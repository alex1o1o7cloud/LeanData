import Mathlib

namespace NUMINAMATH_CALUDE_seed_germination_percentage_l104_10455

theorem seed_germination_percentage 
  (seeds_plot1 : ℕ) 
  (seeds_plot2 : ℕ) 
  (germination_rate_plot1 : ℚ) 
  (germination_rate_plot2 : ℚ) 
  (h1 : seeds_plot1 = 300)
  (h2 : seeds_plot2 = 200)
  (h3 : germination_rate_plot1 = 25 / 100)
  (h4 : germination_rate_plot2 = 30 / 100) :
  (seeds_plot1 * germination_rate_plot1 + seeds_plot2 * germination_rate_plot2) / 
  (seeds_plot1 + seeds_plot2) = 27 / 100 := by
sorry

end NUMINAMATH_CALUDE_seed_germination_percentage_l104_10455


namespace NUMINAMATH_CALUDE_polynomial_factorization_isosceles_triangle_l104_10418

-- Part 1: Polynomial factorization
theorem polynomial_factorization (x y : ℝ) :
  x^2 - 2*x*y + y^2 - 16 = (x - y + 4) * (x - y - 4) := by sorry

-- Part 2: Triangle shape determination
def is_triangle (a b c : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem isosceles_triangle (a b c : ℝ) (h : is_triangle a b c) :
  a^2 - a*b + a*c - b*c = 0 → a = b := by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_isosceles_triangle_l104_10418


namespace NUMINAMATH_CALUDE_point_on_extension_line_l104_10416

/-- Given two points P₁ and P₂ in ℝ², and a point P on the extension line of P₁P₂
    such that the distance between P₁ and P is twice the distance between P and P₂,
    prove that P has the coordinates (-2, 11). -/
theorem point_on_extension_line (P₁ P₂ P : ℝ × ℝ) : 
  P₁ = (2, -1) → 
  P₂ = (0, 5) → 
  (∃ t : ℝ, P = P₁ + t • (P₂ - P₁)) →
  dist P₁ P = 2 * dist P P₂ →
  P = (-2, 11) := by
  sorry

end NUMINAMATH_CALUDE_point_on_extension_line_l104_10416


namespace NUMINAMATH_CALUDE_quadratic_function_bound_l104_10445

/-- Theorem: Bound on quadratic function -/
theorem quadratic_function_bound (a b c : ℝ) (ha : a > 0) (hb : b ≠ 0)
  (hf0 : |a * 0^2 + b * 0 + c| ≤ 1)
  (hfn1 : |a * (-1)^2 + b * (-1) + c| ≤ 1)
  (hf1 : |a * 1^2 + b * 1 + c| ≤ 1)
  (hba : |b| ≤ a) :
  ∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 5/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_bound_l104_10445


namespace NUMINAMATH_CALUDE_basketball_team_selection_l104_10495

def total_players : Nat := 18
def quadruplets : Nat := 4
def starters : Nat := 6
def quadruplets_in_lineup : Nat := 2

theorem basketball_team_selection :
  (Nat.choose quadruplets quadruplets_in_lineup) *
  (Nat.choose (total_players - quadruplets) (starters - quadruplets_in_lineup)) = 6006 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l104_10495


namespace NUMINAMATH_CALUDE_odd_number_representation_l104_10480

theorem odd_number_representation (k : ℤ) : 
  (k % 2 = 1) → 
  ((∃ n : ℤ, 2 * n + 3 = k) ∧ 
   ¬(∀ k : ℤ, k % 2 = 1 → ∃ n : ℤ, 4 * n - 1 = k)) := by
sorry

end NUMINAMATH_CALUDE_odd_number_representation_l104_10480


namespace NUMINAMATH_CALUDE_max_d_value_l104_10483

def a (n : ℕ+) : ℕ := 100 + n^2

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  ∃ (m : ℕ+), ∀ (n : ℕ+), d n ≤ d m ∧ d m = 401 :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l104_10483


namespace NUMINAMATH_CALUDE_all_left_probability_l104_10454

/-- Represents the particle movement experiment -/
structure ParticleExperiment where
  total_particles : ℕ
  initial_left : ℕ
  initial_right : ℕ

/-- The probability of all particles ending on the left side -/
def probability_all_left (exp : ParticleExperiment) : ℚ :=
  1 / 2

/-- The main theorem stating the probability of all particles ending on the left side -/
theorem all_left_probability (exp : ParticleExperiment) 
  (h1 : exp.total_particles = 100)
  (h2 : exp.initial_left = 32)
  (h3 : exp.initial_right = 68)
  (h4 : exp.initial_left + exp.initial_right = exp.total_particles) :
  probability_all_left exp = 1 / 2 := by
  sorry

#eval (100 * 1 + 2 : ℕ)

end NUMINAMATH_CALUDE_all_left_probability_l104_10454


namespace NUMINAMATH_CALUDE_slower_bike_speed_l104_10472

theorem slower_bike_speed 
  (distance : ℝ) 
  (fast_speed : ℝ) 
  (time_difference : ℝ) 
  (h1 : distance = 960) 
  (h2 : fast_speed = 64) 
  (h3 : time_difference = 1) :
  ∃ (slow_speed : ℝ), 
    slow_speed > 0 ∧ 
    distance / slow_speed = distance / fast_speed + time_difference ∧ 
    slow_speed = 60 := by
sorry

end NUMINAMATH_CALUDE_slower_bike_speed_l104_10472


namespace NUMINAMATH_CALUDE_non_dividing_diagonals_count_l104_10468

/-- The number of sides in the regular polygon -/
def n : ℕ := 150

/-- The total number of diagonals in a polygon with n sides -/
def total_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of diagonals that divide the polygon into two equal parts -/
def equal_dividing_diagonals (n : ℕ) : ℕ := n / 2

/-- The number of diagonals that do not divide the polygon into two equal parts -/
def non_dividing_diagonals (n : ℕ) : ℕ := total_diagonals n - equal_dividing_diagonals n

theorem non_dividing_diagonals_count :
  non_dividing_diagonals n = 10950 :=
by sorry

end NUMINAMATH_CALUDE_non_dividing_diagonals_count_l104_10468


namespace NUMINAMATH_CALUDE_sin_sum_specific_angles_l104_10494

theorem sin_sum_specific_angles (θ φ : ℝ) :
  Complex.exp (θ * Complex.I) = (4 / 5 : ℂ) + (3 / 5 : ℂ) * Complex.I ∧
  Complex.exp (φ * Complex.I) = -(5 / 13 : ℂ) - (12 / 13 : ℂ) * Complex.I →
  Real.sin (θ + φ) = -(63 / 65) := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_specific_angles_l104_10494


namespace NUMINAMATH_CALUDE_problem_statement_l104_10458

theorem problem_statement (x : ℝ) (h : x + 1/x = 3) :
  (x - 3)^2 + 16/((x - 3)^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l104_10458


namespace NUMINAMATH_CALUDE_permutation_of_6_choose_2_l104_10490

def A (n : ℕ) (k : ℕ) : ℕ := n * (n - 1)

theorem permutation_of_6_choose_2 : A 6 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_permutation_of_6_choose_2_l104_10490


namespace NUMINAMATH_CALUDE_monomial_sum_condition_l104_10433

/-- If the sum of the monomials $-2x^{4}y^{m-1}$ and $5x^{n-1}y^{2}$ is a monomial, then $m-2n = -7$. -/
theorem monomial_sum_condition (m n : ℤ) : 
  (∃ (a : ℚ) (b c : ℕ), -2 * X^4 * Y^(m-1) + 5 * X^(n-1) * Y^2 = a * X^b * Y^c) → 
  m - 2*n = -7 :=
by sorry

end NUMINAMATH_CALUDE_monomial_sum_condition_l104_10433


namespace NUMINAMATH_CALUDE_cubic_function_property_l104_10484

/-- A cubic function g(x) with coefficients p, q, r, and s. -/
def g (p q r s : ℝ) (x : ℝ) : ℝ := p * x^3 + q * x^2 + r * x + s

/-- Theorem stating that for a cubic function g(x) = px³ + qx² + rx + s,
    if g(-3) = 4, then 10p - 5q + 3r - 2s = 40. -/
theorem cubic_function_property (p q r s : ℝ) : 
  g p q r s (-3) = 4 → 10*p - 5*q + 3*r - 2*s = 40 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_property_l104_10484


namespace NUMINAMATH_CALUDE_smallest_coin_collection_l104_10498

def num_factors (n : ℕ) : ℕ := (Nat.divisors n).card

def proper_factors (n : ℕ) : Finset ℕ :=
  (Nat.divisors n).filter (λ x => x > 1 ∧ x < n)

theorem smallest_coin_collection :
  ∃ (n : ℕ), n > 0 ∧ num_factors n = 13 ∧ (proper_factors n).card ≥ 11 ∧
  ∀ (m : ℕ), m > 0 → num_factors m = 13 → (proper_factors m).card ≥ 11 → n ≤ m :=
by
  use 4096
  sorry

end NUMINAMATH_CALUDE_smallest_coin_collection_l104_10498


namespace NUMINAMATH_CALUDE_matrix_multiplication_example_l104_10400

theorem matrix_multiplication_example :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 2, -4]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![7, -3; 0, 2]
  A * B = !![21, -7; 14, -14] := by
  sorry

end NUMINAMATH_CALUDE_matrix_multiplication_example_l104_10400


namespace NUMINAMATH_CALUDE_middle_number_of_seven_consecutive_l104_10448

def is_middle_of_seven_consecutive (n : ℕ) : Prop :=
  ∃ (a : ℕ), a + (a + 1) + (a + 2) + n + (n + 1) + (n + 2) + (n + 3) = 63

theorem middle_number_of_seven_consecutive :
  ∃ (n : ℕ), is_middle_of_seven_consecutive n ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_of_seven_consecutive_l104_10448


namespace NUMINAMATH_CALUDE_stacy_extra_berries_l104_10450

/-- The number of berries each person has -/
structure BerryCount where
  stacy : ℕ
  steve : ℕ
  skylar : ℕ

/-- The given conditions for the berry problem -/
def berry_conditions (b : BerryCount) : Prop :=
  b.stacy > 3 * b.steve ∧
  2 * b.steve = b.skylar ∧
  b.skylar = 20 ∧
  b.stacy = 32

/-- The theorem to prove -/
theorem stacy_extra_berries (b : BerryCount) (h : berry_conditions b) :
  b.stacy - 3 * b.steve = 2 := by
  sorry

end NUMINAMATH_CALUDE_stacy_extra_berries_l104_10450


namespace NUMINAMATH_CALUDE_cookies_per_sitting_l104_10474

/-- The number of times Theo eats cookies per day -/
def eats_per_day : ℕ := 3

/-- The number of days Theo eats cookies per month -/
def days_per_month : ℕ := 20

/-- The total number of cookies Theo eats in 3 months -/
def total_cookies : ℕ := 2340

/-- The number of months considered -/
def months : ℕ := 3

/-- Theorem stating the number of cookies Theo can eat in one sitting -/
theorem cookies_per_sitting :
  total_cookies / (eats_per_day * days_per_month * months) = 13 := by sorry

end NUMINAMATH_CALUDE_cookies_per_sitting_l104_10474


namespace NUMINAMATH_CALUDE_geometry_propositions_l104_10401

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallelPL : Line → Plane → Prop)
variable (perpendicularPL : Line → Plane → Prop)
variable (parallelPlanes : Plane → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)
variable (intersection : Plane → Plane → Line)

-- Theorem statement
theorem geometry_propositions 
  (α β : Plane) (m n : Line) : 
  -- Proposition 2
  (∀ m, perpendicularPL m α ∧ perpendicularPL m β → parallelPlanes α β) ∧
  -- Proposition 3
  (intersection α β = n ∧ parallelPL m α ∧ parallelPL m β → parallel m n) ∧
  -- Proposition 4
  (perpendicularPlanes α β ∧ perpendicularPL m α ∧ perpendicularPL n β → perpendicular m n) :=
by sorry

end NUMINAMATH_CALUDE_geometry_propositions_l104_10401


namespace NUMINAMATH_CALUDE_odot_properties_l104_10471

/-- The custom operation ⊙ -/
def odot (a : ℝ) (x y : ℝ) : ℝ := 18 + x - a * y

/-- Theorem stating the properties of the ⊙ operation -/
theorem odot_properties :
  ∃ a : ℝ, (odot a 2 3 = 8) ∧ (odot a 3 5 = 1) ∧ (odot a 5 3 = 11) := by
  sorry

end NUMINAMATH_CALUDE_odot_properties_l104_10471


namespace NUMINAMATH_CALUDE_chef_almond_weight_l104_10414

/-- The weight of pecans bought by the chef in kilograms. -/
def pecan_weight : ℝ := 0.38

/-- The total weight of nuts bought by the chef in kilograms. -/
def total_nut_weight : ℝ := 0.52

/-- The weight of almonds bought by the chef in kilograms. -/
def almond_weight : ℝ := total_nut_weight - pecan_weight

theorem chef_almond_weight :
  almond_weight = 0.14 := by sorry

end NUMINAMATH_CALUDE_chef_almond_weight_l104_10414


namespace NUMINAMATH_CALUDE_heather_blocks_l104_10431

/-- The number of blocks Heather ends up with after sharing -/
def blocks_remaining (initial : ℕ) (shared : ℕ) : ℕ :=
  initial - shared

/-- Theorem stating that Heather ends up with 45 blocks -/
theorem heather_blocks : blocks_remaining 86 41 = 45 := by
  sorry

end NUMINAMATH_CALUDE_heather_blocks_l104_10431


namespace NUMINAMATH_CALUDE_vacation_pictures_l104_10426

theorem vacation_pictures (zoo museum beach amusement_park deleted : ℕ) :
  zoo = 802 →
  museum = 526 →
  beach = 391 →
  amusement_park = 868 →
  deleted = 1395 →
  zoo + museum + beach + amusement_park - deleted = 1192 := by
  sorry

end NUMINAMATH_CALUDE_vacation_pictures_l104_10426


namespace NUMINAMATH_CALUDE_eight_power_division_l104_10451

theorem eight_power_division (x : ℕ) (y : ℕ) (z : ℕ) :
  x^15 / (x^2)^3 = x^9 :=
by sorry

end NUMINAMATH_CALUDE_eight_power_division_l104_10451


namespace NUMINAMATH_CALUDE_student_lineup_theorem_l104_10488

theorem student_lineup_theorem (N : ℕ) (heights : Finset ℤ) :
  heights.card = 3 * N + 1 →
  ∃ (subset : Finset ℤ),
    subset ⊆ heights ∧
    subset.card = N + 1 ∧
    ∀ (x y : ℤ), x ∈ subset → y ∈ subset → x ≠ y → |x - y| ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_student_lineup_theorem_l104_10488


namespace NUMINAMATH_CALUDE_ant_on_red_after_six_moves_probability_on_red_after_six_moves_l104_10469

/-- Represents the color of a dot on the lattice -/
inductive DotColor
| Red
| Blue

/-- Represents the state of the ant's position -/
structure AntState :=
  (color : DotColor)

/-- Defines a single move of the ant -/
def move (state : AntState) : AntState :=
  match state.color with
  | DotColor.Red => { color := DotColor.Blue }
  | DotColor.Blue => { color := DotColor.Red }

/-- Applies n moves to the initial state -/
def apply_moves (initial : AntState) (n : ℕ) : AntState :=
  match n with
  | 0 => initial
  | n + 1 => move (apply_moves initial n)

/-- The main theorem to prove -/
theorem ant_on_red_after_six_moves (initial : AntState) :
  initial.color = DotColor.Red →
  (apply_moves initial 6).color = DotColor.Red :=
sorry

/-- The probability of the ant being on a red dot after 6 moves -/
theorem probability_on_red_after_six_moves (initial : AntState) :
  initial.color = DotColor.Red →
  ∃ (p : ℝ), p = 1 ∧ 
  (∀ (final : AntState), (apply_moves initial 6).color = DotColor.Red → p = 1) :=
sorry

end NUMINAMATH_CALUDE_ant_on_red_after_six_moves_probability_on_red_after_six_moves_l104_10469


namespace NUMINAMATH_CALUDE_string_length_problem_l104_10464

theorem string_length_problem (total_length remaining_length used_length : ℝ) : 
  total_length = 90 →
  remaining_length = total_length - 30 →
  used_length = (8 / 15) * remaining_length →
  used_length = 32 := by
sorry

end NUMINAMATH_CALUDE_string_length_problem_l104_10464


namespace NUMINAMATH_CALUDE_festival_selection_probability_l104_10440

-- Define the number of festivals
def total_festivals : ℕ := 5

-- Define the number of festivals to be selected
def selected_festivals : ℕ := 2

-- Define the number of specific festivals we're interested in
def specific_festivals : ℕ := 2

-- Define the probability of selecting at least one of the specific festivals
def probability : ℚ := 0.7

-- Theorem statement
theorem festival_selection_probability :
  1 - (Nat.choose (total_festivals - specific_festivals) selected_festivals) / 
      (Nat.choose total_festivals selected_festivals) = probability := by
  sorry

end NUMINAMATH_CALUDE_festival_selection_probability_l104_10440


namespace NUMINAMATH_CALUDE_decimal_to_percentage_l104_10413

theorem decimal_to_percentage (d : ℝ) (h : d = 0.05) : d * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_percentage_l104_10413


namespace NUMINAMATH_CALUDE_smaller_field_area_l104_10475

/-- Given two square fields where one is 1% broader than the other,
    and the difference in their areas is 201 square meters,
    prove that the area of the smaller field is 10,000 square meters. -/
theorem smaller_field_area (s : ℝ) (h1 : s > 0) :
  (s * 1.01)^2 - s^2 = 201 → s^2 = 10000 := by sorry

end NUMINAMATH_CALUDE_smaller_field_area_l104_10475


namespace NUMINAMATH_CALUDE_charlie_has_32_cards_l104_10410

/-- The number of soccer cards Chris has -/
def chris_cards : ℕ := 18

/-- The difference in cards between Charlie and Chris -/
def card_difference : ℕ := 14

/-- Charlie's number of soccer cards -/
def charlie_cards : ℕ := chris_cards + card_difference

/-- Theorem stating that Charlie has 32 soccer cards -/
theorem charlie_has_32_cards : charlie_cards = 32 := by
  sorry

end NUMINAMATH_CALUDE_charlie_has_32_cards_l104_10410


namespace NUMINAMATH_CALUDE_divisor_not_zero_l104_10430

theorem divisor_not_zero (a b : ℝ) : b ≠ 0 → ∃ (c : ℝ), a / b = c := by
  sorry

end NUMINAMATH_CALUDE_divisor_not_zero_l104_10430


namespace NUMINAMATH_CALUDE_cat_adoption_cost_l104_10465

/-- The cost to get each cat ready for adoption -/
def cat_cost : ℝ := 50

/-- The cost to get each adult dog ready for adoption -/
def adult_dog_cost : ℝ := 100

/-- The cost to get each puppy ready for adoption -/
def puppy_cost : ℝ := 150

/-- The number of cats adopted -/
def num_cats : ℕ := 2

/-- The number of adult dogs adopted -/
def num_adult_dogs : ℕ := 3

/-- The number of puppies adopted -/
def num_puppies : ℕ := 2

/-- The total cost to get all adopted animals ready -/
def total_cost : ℝ := 700

/-- Theorem stating that the cost to get each cat ready for adoption is $50 -/
theorem cat_adoption_cost : 
  cat_cost * num_cats + adult_dog_cost * num_adult_dogs + puppy_cost * num_puppies = total_cost :=
by sorry

end NUMINAMATH_CALUDE_cat_adoption_cost_l104_10465


namespace NUMINAMATH_CALUDE_jess_walked_five_blocks_l104_10449

/-- The number of blocks Jess has already walked -/
def blocks_walked (total_blocks remaining_blocks : ℕ) : ℕ :=
  total_blocks - remaining_blocks

/-- Proof that Jess has walked 5 blocks -/
theorem jess_walked_five_blocks :
  blocks_walked 25 20 = 5 := by
  sorry

end NUMINAMATH_CALUDE_jess_walked_five_blocks_l104_10449


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_segments_ratio_l104_10463

theorem right_triangle_hypotenuse_segments_ratio 
  (a b c : ℝ) 
  (h_right : a^2 + b^2 = c^2) 
  (h_ratio : a / b = 3 / 4) :
  let d := (a * c) / (a + b)
  (c - d) / d = 16 / 9 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_segments_ratio_l104_10463


namespace NUMINAMATH_CALUDE_cube_cross_section_theorem_l104_10499

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  vertices : Fin 8 → Point3D

/-- Represents a plane in 3D space -/
structure Plane where
  normal : Point3D
  point : Point3D

/-- Represents a polygon in 3D space -/
structure Polygon where
  vertices : List Point3D

def isPerpendicularTo (p : Plane) (v : Point3D) : Prop := sorry

def intersectsAllFaces (p : Plane) (c : Cube) : Prop := sorry

def crossSectionPolygon (p : Plane) (c : Cube) : Polygon := sorry

def perimeter (poly : Polygon) : ℝ := sorry

def area (poly : Polygon) : ℝ := sorry

theorem cube_cross_section_theorem (c : Cube) (p : Plane) (ac' : Point3D) :
  isPerpendicularTo p ac' →
  intersectsAllFaces p c →
  (∃ l : ℝ, ∀ α : Plane, isPerpendicularTo α ac' → intersectsAllFaces α c →
    perimeter (crossSectionPolygon α c) = l) ∧
  (¬∃ s : ℝ, ∀ α : Plane, isPerpendicularTo α ac' → intersectsAllFaces α c →
    area (crossSectionPolygon α c) = s) := by
  sorry

end NUMINAMATH_CALUDE_cube_cross_section_theorem_l104_10499


namespace NUMINAMATH_CALUDE_mental_math_competition_l104_10436

theorem mental_math_competition :
  ∃! (numbers : Finset ℕ),
    numbers.card = 4 ∧
    (∀ n ∈ numbers,
      ∃ (M m : ℕ),
        n = 15 * M + 11 * m ∧
        M > 1 ∧ m > 1 ∧
        Odd M ∧ Odd m ∧
        (∀ d : ℕ, d > 1 → Odd d → d ∣ n → m ≤ d ∧ d ≤ M) ∧
        numbers = {528, 880, 1232, 1936}) :=
by sorry

end NUMINAMATH_CALUDE_mental_math_competition_l104_10436


namespace NUMINAMATH_CALUDE_max_value_of_z_l104_10432

/-- Given a system of inequalities, prove that the maximum value of z = 2x + 3y is 8 -/
theorem max_value_of_z (x y : ℝ) 
  (h1 : x + y - 1 ≥ 0) 
  (h2 : y - x - 1 ≤ 0) 
  (h3 : x ≤ 1) : 
  (∀ x' y' : ℝ, x' + y' - 1 ≥ 0 → y' - x' - 1 ≤ 0 → x' ≤ 1 → 2*x' + 3*y' ≤ 2*x + 3*y) →
  2*x + 3*y = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_z_l104_10432


namespace NUMINAMATH_CALUDE_sector_arc_length_l104_10473

theorem sector_arc_length (r : ℝ) (A : ℝ) (l : ℝ) : 
  r = 2 → A = π / 3 → A = 1 / 2 * r * l → l = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l104_10473


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l104_10461

def U : Set Nat := {1,2,3,4,5,6}
def M : Set Nat := {1,2,4}

theorem complement_of_M_in_U :
  (U \ M) = {3,5,6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l104_10461


namespace NUMINAMATH_CALUDE_pen_arrangement_count_l104_10496

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def multinomial (n : ℕ) (ks : List ℕ) : ℕ :=
  factorial n / (ks.map factorial).prod

theorem pen_arrangement_count :
  let total_pens := 15
  let blue_pens := 7
  let red_pens := 3
  let green_pens := 3
  let black_pens := 2
  let total_arrangements := multinomial total_pens [blue_pens, red_pens, green_pens, black_pens]
  let adjacent_green_arrangements := 
    (multinomial (total_pens - green_pens + 1) [blue_pens, red_pens, 1, black_pens]) * (factorial green_pens)
  total_arrangements - adjacent_green_arrangements = 6098400 := by
  sorry

end NUMINAMATH_CALUDE_pen_arrangement_count_l104_10496


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l104_10403

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The median of a sequence with an odd number of terms is the middle term. -/
def median (a : ℕ → ℝ) (n : ℕ) : ℝ := a ((n + 1) / 2)

theorem arithmetic_sequence_first_term
  (a : ℕ → ℝ)  -- The sequence
  (n : ℕ)      -- The number of terms in the sequence
  (h1 : is_arithmetic_sequence a)
  (h2 : median a n = 1010)
  (h3 : a n = 2015) :
  a 1 = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l104_10403


namespace NUMINAMATH_CALUDE_stock_sold_percentage_l104_10442

/-- Given the cash realized, brokerage rate, and total amount including brokerage,
    prove that the percentage of stock sold is 100% -/
theorem stock_sold_percentage
  (cash_realized : ℝ)
  (brokerage_rate : ℝ)
  (total_amount : ℝ)
  (h1 : cash_realized = 106.25)
  (h2 : brokerage_rate = 1 / 4 / 100)
  (h3 : total_amount = 106) :
  let sale_amount := cash_realized + (cash_realized * brokerage_rate)
  let stock_percentage := sale_amount / sale_amount * 100
  stock_percentage = 100 := by sorry

end NUMINAMATH_CALUDE_stock_sold_percentage_l104_10442


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l104_10478

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 6) (h2 : x ≠ -5) :
  (7 * x + 11) / (x^2 - x - 30) = (53 / 11) / (x - 6) + (24 / 11) / (x + 5) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l104_10478


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l104_10489

theorem sqrt_equation_solutions :
  ∀ x : ℝ, (Real.sqrt (5 * x - 6) + 8 / Real.sqrt (5 * x - 6) = 6) ↔ (x = 22/5 ∨ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l104_10489


namespace NUMINAMATH_CALUDE_solve_logarithmic_equation_l104_10487

-- Define the base-10 logarithm function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem solve_logarithmic_equation :
  ∃ x : ℝ, log10 (3 * x + 4) = 1 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_logarithmic_equation_l104_10487


namespace NUMINAMATH_CALUDE_trapezoid_median_length_l104_10481

theorem trapezoid_median_length :
  let large_side : ℝ := 4
  let large_area : ℝ := (Real.sqrt 3 / 4) * large_side^2
  let small_area : ℝ := large_area / 3
  let small_side : ℝ := Real.sqrt ((4 * small_area) / Real.sqrt 3)
  let median : ℝ := (large_side + small_side) / 2
  median = (2 * (Real.sqrt 3 + 1)) / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_median_length_l104_10481


namespace NUMINAMATH_CALUDE_can_capacity_is_30_liters_l104_10456

/-- Represents the contents of a can with milk and water -/
structure CanContents where
  milk : ℝ
  water : ℝ

/-- The capacity of the can in liters -/
def canCapacity : ℝ := 30

/-- The amount of milk added in liters -/
def milkAdded : ℝ := 10

/-- Checks if the given contents match the initial ratio of 4:3 -/
def isInitialRatio (contents : CanContents) : Prop :=
  contents.milk / contents.water = 4 / 3

/-- Checks if the given contents match the final ratio of 5:2 -/
def isFinalRatio (contents : CanContents) : Prop :=
  contents.milk / contents.water = 5 / 2

/-- Theorem stating that given the conditions, the can capacity is 30 liters -/
theorem can_capacity_is_30_liters 
  (initialContents : CanContents) 
  (hInitialRatio : isInitialRatio initialContents)
  (hFinalRatio : isFinalRatio { milk := initialContents.milk + milkAdded, water := initialContents.water })
  (hFull : initialContents.milk + initialContents.water + milkAdded = canCapacity) : 
  canCapacity = 30 := by
  sorry


end NUMINAMATH_CALUDE_can_capacity_is_30_liters_l104_10456


namespace NUMINAMATH_CALUDE_z_mod_nine_l104_10424

theorem z_mod_nine (z : ℤ) (h : ∃ k : ℤ, (z + 3) / 9 = k) : z % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_z_mod_nine_l104_10424


namespace NUMINAMATH_CALUDE_sqrt_123454321_l104_10493

theorem sqrt_123454321 : Int.sqrt 123454321 = 11111 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_123454321_l104_10493


namespace NUMINAMATH_CALUDE_fish_count_l104_10492

/-- The number of fishbowls -/
def num_fishbowls : ℕ := 261

/-- The number of fish in each fishbowl -/
def fish_per_bowl : ℕ := 23

/-- The total number of fish -/
def total_fish : ℕ := num_fishbowls * fish_per_bowl

theorem fish_count : total_fish = 6003 := by
  sorry

end NUMINAMATH_CALUDE_fish_count_l104_10492


namespace NUMINAMATH_CALUDE_ball_probabilities_l104_10421

/-- Represents the number of red balls initially in the bag -/
def initial_red_balls : ℕ := 5

/-- Represents the number of yellow balls initially in the bag -/
def initial_yellow_balls : ℕ := 10

/-- Represents the total number of balls added to the bag -/
def added_balls : ℕ := 9

/-- Calculates the probability of drawing a red ball -/
def prob_red_ball : ℚ := initial_red_balls / (initial_red_balls + initial_yellow_balls)

/-- Represents the number of red balls added to the bag -/
def red_balls_added : ℕ := 7

/-- Represents the number of yellow balls added to the bag -/
def yellow_balls_added : ℕ := 2

theorem ball_probabilities :
  (prob_red_ball = 1/3) ∧
  ((initial_red_balls + red_balls_added) / (initial_red_balls + initial_yellow_balls + added_balls) =
   (initial_yellow_balls + yellow_balls_added) / (initial_red_balls + initial_yellow_balls + added_balls)) :=
by sorry

end NUMINAMATH_CALUDE_ball_probabilities_l104_10421


namespace NUMINAMATH_CALUDE_optimal_shopping_solution_l104_10441

/-- Represents the shopping problem with discounts --/
structure ShoppingProblem where
  budget : ℕ
  jacket_price : ℕ
  tshirt_price : ℕ
  jeans_price : ℕ

/-- Calculates the cost of jackets with the buy 2 get 1 free discount --/
def jacket_cost (n : ℕ) (price : ℕ) : ℕ :=
  (n / 3 * 2 + n % 3) * price

/-- Calculates the cost of t-shirts with the buy 3 get 1 free discount --/
def tshirt_cost (n : ℕ) (price : ℕ) : ℕ :=
  (n / 4 * 3 + n % 4) * price

/-- Calculates the cost of jeans with the 50% discount on every other pair --/
def jeans_cost (n : ℕ) (price : ℕ) : ℕ :=
  (n / 2 * 3 + n % 2) * (price / 2)

/-- Represents the optimal shopping solution --/
structure ShoppingSolution where
  jackets : ℕ
  tshirts : ℕ
  jeans : ℕ
  total_spent : ℕ
  remaining : ℕ

/-- Theorem stating the optimal solution for the shopping problem --/
theorem optimal_shopping_solution (p : ShoppingProblem)
    (h : p = { budget := 400, jacket_price := 50, tshirt_price := 25, jeans_price := 40 }) :
    ∃ (s : ShoppingSolution),
      s.jackets = 4 ∧
      s.tshirts = 12 ∧
      s.jeans = 3 ∧
      s.total_spent = 380 ∧
      s.remaining = 20 ∧
      jacket_cost s.jackets p.jacket_price +
      tshirt_cost s.tshirts p.tshirt_price +
      jeans_cost s.jeans p.jeans_price = s.total_spent ∧
      s.total_spent + s.remaining = p.budget ∧
      ∀ (s' : ShoppingSolution),
        jacket_cost s'.jackets p.jacket_price +
        tshirt_cost s'.tshirts p.tshirt_price +
        jeans_cost s'.jeans p.jeans_price ≤ p.budget →
        s'.jackets + s'.tshirts + s'.jeans ≤ s.jackets + s.tshirts + s.jeans :=
by sorry

end NUMINAMATH_CALUDE_optimal_shopping_solution_l104_10441


namespace NUMINAMATH_CALUDE_fraction_power_seven_l104_10462

theorem fraction_power_seven : (5 / 7 : ℚ) ^ 7 = 78125 / 823543 := by sorry

end NUMINAMATH_CALUDE_fraction_power_seven_l104_10462


namespace NUMINAMATH_CALUDE_probability_two_girls_l104_10446

theorem probability_two_girls (total_members : ℕ) (girl_members : ℕ) : 
  total_members = 12 → 
  girl_members = 7 → 
  (Nat.choose girl_members 2 : ℚ) / (Nat.choose total_members 2 : ℚ) = 7 / 22 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_girls_l104_10446


namespace NUMINAMATH_CALUDE_simplify_expression_l104_10434

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) :
  (y - 1) * x⁻¹ - y = -((y * x - y + 1) / x) := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l104_10434


namespace NUMINAMATH_CALUDE_pyramid_properties_l104_10477

/-- Given a sphere of radius r and a regular four-sided pyramid constructed
    such that:
    1. The sphere is divided into two parts
    2. The part towards the center is the mean proportional between the entire radius and the other part
    3. A plane is placed perpendicularly to the radius at the dividing point
    4. The pyramid is constructed in the larger segment of the sphere
    5. The apex of the pyramid is on the surface of the sphere

    Then the following properties hold for the pyramid:
    1. Its volume is 2/3 * r^3
    2. Its surface area is r^2 * (√(2√5 + 10) + √5 - 1)
    3. The tangent of its inclination angle is 1/2 * (√(√5 + 1))^3
-/
theorem pyramid_properties (r : ℝ) (h : r > 0) :
  ∃ (V F : ℝ) (tan_α : ℝ),
    V = 2/3 * r^3 ∧
    F = r^2 * (Real.sqrt (2 * Real.sqrt 5 + 10) + Real.sqrt 5 - 1) ∧
    tan_α = 1/2 * (Real.sqrt (Real.sqrt 5 + 1))^3 :=
sorry

end NUMINAMATH_CALUDE_pyramid_properties_l104_10477


namespace NUMINAMATH_CALUDE_intersection_equals_open_interval_l104_10420

-- Define sets A and B
def A : Set ℝ := {x | (4*x - 3)*(x + 3) < 0}
def B : Set ℝ := {x | 2*x > 1}

-- Define the open interval (1/2, 3/4)
def openInterval : Set ℝ := {x | 1/2 < x ∧ x < 3/4}

-- Theorem statement
theorem intersection_equals_open_interval : A ∩ B = openInterval := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_open_interval_l104_10420


namespace NUMINAMATH_CALUDE_set_operations_and_intersection_intersection_empty_iff_m_range_l104_10404

def A : Set ℝ := {x | -4 < x ∧ x < 2}
def B : Set ℝ := {x | x < -5 ∨ x > 1}
def C (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ m + 1}

theorem set_operations_and_intersection :
  (A ∪ B = {x | x < -5 ∨ x > -4}) ∧
  (A ∩ (Set.univ \ B) = {x | -4 < x ∧ x ≤ 1}) := by sorry

theorem intersection_empty_iff_m_range (m : ℝ) :
  (B ∩ C m = ∅) ↔ (-4 ≤ m ∧ m ≤ 0) := by sorry

end NUMINAMATH_CALUDE_set_operations_and_intersection_intersection_empty_iff_m_range_l104_10404


namespace NUMINAMATH_CALUDE_tan_product_greater_than_one_l104_10417

/-- In an acute triangle ABC, where a, b, c are sides opposite to angles A, B, C respectively,
    and a² = b² + bc, the product of tan A and tan B is always greater than 1. -/
theorem tan_product_greater_than_one (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  A + B + C = π →
  a^2 = b^2 + b*c →
  Real.tan A * Real.tan B > 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_greater_than_one_l104_10417


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_equation_l104_10408

theorem no_integer_solutions_for_equation :
  ¬ ∃ (x y z t : ℤ), x^2 + y^2 + z^2 = 8*t - 1 :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_equation_l104_10408


namespace NUMINAMATH_CALUDE_rectangle_square_overlap_ratio_l104_10439

/-- Given a rectangle ABCD and a square JKLM, if the rectangle shares 40% of its area with the square,
    and the square shares 25% of its area with the rectangle, then the ratio of the length (AB) to 
    the width (AD) of the rectangle is 15.625. -/
theorem rectangle_square_overlap_ratio : 
  ∀ (AB AD s : ℝ), 
  AB > 0 → AD > 0 → s > 0 →
  0.4 * AB * AD = 0.25 * s^2 →
  AB / AD = 15.625 := by
    sorry

end NUMINAMATH_CALUDE_rectangle_square_overlap_ratio_l104_10439


namespace NUMINAMATH_CALUDE_angle_value_for_given_function_l104_10407

/-- Given a function f(x) = sin x + √3 * cos x, prove that if there exists an acute angle θ
    such that f(θ) = 2, then θ = π/6 -/
theorem angle_value_for_given_function (θ : Real) :
  (∃ f : Real → Real, f = λ x => Real.sin x + Real.sqrt 3 * Real.cos x) →
  (0 < θ ∧ θ < π / 2) →
  (∃ f : Real → Real, f θ = 2) →
  θ = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_value_for_given_function_l104_10407


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l104_10459

noncomputable section

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the right focus F
def right_focus (a b c : ℝ) : ℝ × ℝ := (c, 0)

-- Define the line l passing through the origin
def line_through_origin (m n : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | ∃ t : ℝ, x = m * t ∧ y = n * t}

-- Define the condition that M and N are on the hyperbola and the line
def points_on_hyperbola_and_line (a b m n : ℝ) (M N : ℝ × ℝ) : Prop :=
  hyperbola a b M.1 M.2 ∧ hyperbola a b N.1 N.2 ∧
  M ∈ line_through_origin m n ∧ N ∈ line_through_origin m n

-- Define the perpendicularity condition
def perpendicular_vectors (F M N : ℝ × ℝ) : Prop :=
  (M.1 - F.1) * (N.1 - F.1) + (M.2 - F.2) * (N.2 - F.2) = 0

-- Define the area condition
def triangle_area (F M N : ℝ × ℝ) (a b : ℝ) : Prop :=
  abs ((M.1 - F.1) * (N.2 - F.2) - (N.1 - F.1) * (M.2 - F.2)) / 2 = a * b

-- Main theorem
theorem hyperbola_eccentricity
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (F : ℝ × ℝ)
  (hF : F = right_focus a b c)
  (M N : ℝ × ℝ)
  (h_points : ∃ m n : ℝ, points_on_hyperbola_and_line a b m n M N)
  (h_perp : perpendicular_vectors F M N)
  (h_area : triangle_area F M N a b) :
  c^2 / a^2 = 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l104_10459


namespace NUMINAMATH_CALUDE_garden_breadth_l104_10412

theorem garden_breadth (perimeter length : ℕ) (h1 : perimeter = 1200) (h2 : length = 360) :
  let breadth := (perimeter / 2) - length
  breadth = 240 :=
by sorry

end NUMINAMATH_CALUDE_garden_breadth_l104_10412


namespace NUMINAMATH_CALUDE_fraction_equality_l104_10419

theorem fraction_equality : (5 * 7 - 3) / 9 = 32 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l104_10419


namespace NUMINAMATH_CALUDE_f_has_one_zero_a_equals_one_l104_10438

noncomputable def f (x : ℝ) : ℝ := Real.exp x - 1 / (x^2)

theorem f_has_one_zero :
  ∃! x, f x = 0 :=
sorry

theorem a_equals_one (a : ℝ) :
  (∀ x > 0, f x ≥ (2 * a * Real.log x) / x^2 + a / x) ↔ a = 1 :=
sorry

end NUMINAMATH_CALUDE_f_has_one_zero_a_equals_one_l104_10438


namespace NUMINAMATH_CALUDE_apple_production_total_l104_10429

/-- The number of apples produced by a tree over three years -/
def appleProduction : ℕ → ℕ
| 1 => 40
| 2 => 2 * appleProduction 1 + 8
| 3 => appleProduction 2 - (appleProduction 2 / 4)
| _ => 0

/-- The total number of apples produced over three years -/
def totalApples : ℕ := appleProduction 1 + appleProduction 2 + appleProduction 3

theorem apple_production_total : totalApples = 194 := by
  sorry

end NUMINAMATH_CALUDE_apple_production_total_l104_10429


namespace NUMINAMATH_CALUDE_f_max_value_l104_10405

-- Define the function
def f (x : ℝ) : ℝ := 3 * x - x^3

-- State the theorem
theorem f_max_value :
  ∃ (c : ℝ), c > 0 ∧ f c = 2 ∧ ∀ x > 0, f x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_f_max_value_l104_10405


namespace NUMINAMATH_CALUDE_solve_system_l104_10437

theorem solve_system (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 17) 
  (eq2 : 6 * p + 5 * q = 20) : 
  q = 2 / 11 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l104_10437


namespace NUMINAMATH_CALUDE_negative_power_division_l104_10411

theorem negative_power_division : -2^5 / (-2)^3 = 4 := by sorry

end NUMINAMATH_CALUDE_negative_power_division_l104_10411


namespace NUMINAMATH_CALUDE_total_fruits_picked_l104_10447

theorem total_fruits_picked (sara_pears tim_pears lily_apples max_oranges : ℕ)
  (h1 : sara_pears = 6)
  (h2 : tim_pears = 5)
  (h3 : lily_apples = 4)
  (h4 : max_oranges = 3) :
  sara_pears + tim_pears + lily_apples + max_oranges = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_picked_l104_10447


namespace NUMINAMATH_CALUDE_seventeen_stations_tickets_l104_10485

/-- The number of unique, non-directional tickets needed for travel between any two stations -/
def num_tickets (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

/-- Theorem: For 17 stations, the number of unique, non-directional tickets is 68 -/
theorem seventeen_stations_tickets :
  num_tickets 17 = 68 := by
  sorry

end NUMINAMATH_CALUDE_seventeen_stations_tickets_l104_10485


namespace NUMINAMATH_CALUDE_angle_B_measure_l104_10470

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ)
  (sum_angles : A + B + C + D = 360)

-- Define the theorem
theorem angle_B_measure (q : Quadrilateral) (h : q.A + q.C = 150) : q.B = 105 := by
  sorry

end NUMINAMATH_CALUDE_angle_B_measure_l104_10470


namespace NUMINAMATH_CALUDE_quadratic_equation_root_zero_l104_10479

theorem quadratic_equation_root_zero (k : ℝ) : 
  (∀ x : ℝ, (k - 1) * x^2 + 3 * x + k^2 - 1 = 0 → x = 0 ∨ x ≠ 0) →
  ((k - 1) * 0^2 + 3 * 0 + k^2 - 1 = 0) →
  k = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_zero_l104_10479


namespace NUMINAMATH_CALUDE_remainder_two_power_thirty_plus_three_mod_seven_l104_10425

theorem remainder_two_power_thirty_plus_three_mod_seven :
  (2^30 + 3) % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_two_power_thirty_plus_three_mod_seven_l104_10425


namespace NUMINAMATH_CALUDE_kristin_laps_theorem_l104_10428

/-- Kristin's running speed relative to Sarith's -/
def kristin_speed_ratio : ℚ := 3

/-- Ratio of adult field size to children's field size -/
def field_size_ratio : ℚ := 2

/-- Number of times Sarith went around the children's field -/
def sarith_laps : ℕ := 8

/-- Number of times Kristin went around the adult field -/
def kristin_laps : ℕ := 12

theorem kristin_laps_theorem (speed_ratio : ℚ) (field_ratio : ℚ) (sarith_runs : ℕ) :
  speed_ratio = kristin_speed_ratio →
  field_ratio = field_size_ratio →
  sarith_runs = sarith_laps →
  ↑kristin_laps = ↑sarith_runs * (speed_ratio / field_ratio) := by
  sorry

end NUMINAMATH_CALUDE_kristin_laps_theorem_l104_10428


namespace NUMINAMATH_CALUDE_cube_root_sixteen_to_sixth_l104_10406

theorem cube_root_sixteen_to_sixth (x : ℝ) : x = (16 ^ (1/3 : ℝ)) → x^6 = 256 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_sixteen_to_sixth_l104_10406


namespace NUMINAMATH_CALUDE_cube_of_negative_two_times_t_l104_10415

theorem cube_of_negative_two_times_t (t : ℝ) : (-2 * t)^3 = -8 * t^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_negative_two_times_t_l104_10415


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l104_10486

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁ + 1) * (x₁ - 1) = 2 * x₁ + 3 ∧ 
  (x₂ + 1) * (x₂ - 1) = 2 * x₂ + 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l104_10486


namespace NUMINAMATH_CALUDE_exists_monochromatic_isosceles_right_triangle_l104_10467

/-- A color type with three possible values -/
inductive Color
  | Red
  | Green
  | Blue

/-- A point in the infinite grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A coloring function that assigns a color to each point in the grid -/
def Coloring := GridPoint → Color

/-- An isosceles right triangle in the grid -/
structure IsoscelesRightTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint
  is_isosceles : (p1.x - p2.x)^2 + (p1.y - p2.y)^2 = (p1.x - p3.x)^2 + (p1.y - p3.y)^2
  is_right : (p2.x - p1.x) * (p3.x - p1.x) + (p2.y - p1.y) * (p3.y - p1.y) = 0

/-- The main theorem: In any coloring of an infinite grid with three colors,
    there exists an isosceles right triangle with vertices of the same color -/
theorem exists_monochromatic_isosceles_right_triangle (c : Coloring) :
  ∃ (t : IsoscelesRightTriangle), c t.p1 = c t.p2 ∧ c t.p2 = c t.p3 := by
  sorry

end NUMINAMATH_CALUDE_exists_monochromatic_isosceles_right_triangle_l104_10467


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l104_10476

/-- Regular tetrahedron with given midpoint distances -/
structure RegularTetrahedron where
  midpoint_to_face : ℝ
  midpoint_to_edge : ℝ

/-- Volume of a regular tetrahedron -/
def volume (t : RegularTetrahedron) : ℝ := sorry

/-- Theorem stating the volume of the specific regular tetrahedron -/
theorem specific_tetrahedron_volume :
  ∃ (t : RegularTetrahedron),
    t.midpoint_to_face = 2 ∧
    t.midpoint_to_edge = Real.sqrt 10 ∧
    volume t = 80 * Real.sqrt 15 := by sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l104_10476


namespace NUMINAMATH_CALUDE_non_shaded_perimeter_l104_10444

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

theorem non_shaded_perimeter (outer : Rectangle) (attached : Rectangle) (shaded : Rectangle) 
  (h_outer_width : outer.width = 12)
  (h_outer_height : outer.height = 10)
  (h_attached_width : attached.width = 3)
  (h_attached_height : attached.height = 4)
  (h_shaded_width : shaded.width = 3)
  (h_shaded_height : shaded.height = 5)
  (h_shaded_area : area shaded = 120)
  (h_shaded_center : shaded.width < outer.width ∧ shaded.height < outer.height) :
  ∃ (non_shaded : Rectangle), perimeter non_shaded = 19 := by
sorry

end NUMINAMATH_CALUDE_non_shaded_perimeter_l104_10444


namespace NUMINAMATH_CALUDE_candy_box_solution_l104_10457

/-- Represents the number of candies of each type in a box -/
structure CandyBox where
  chocolate : ℕ
  hard : ℕ
  jelly : ℕ

/-- Conditions for the candy box problem -/
def CandyBoxConditions (box : CandyBox) : Prop :=
  (box.chocolate + box.hard + box.jelly = 110) ∧
  (box.chocolate + box.hard = 100) ∧
  (box.hard + box.jelly = box.chocolate + box.jelly + 20)

/-- Theorem stating the solution to the candy box problem -/
theorem candy_box_solution :
  ∃ (box : CandyBox), CandyBoxConditions box ∧ 
    box.chocolate = 40 ∧ box.hard = 60 ∧ box.jelly = 10 := by
  sorry

end NUMINAMATH_CALUDE_candy_box_solution_l104_10457


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l104_10466

theorem solution_set_of_inequality (x : ℝ) : 
  (x + 3)^2 < 1 ↔ -4 < x ∧ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l104_10466


namespace NUMINAMATH_CALUDE_power_relations_l104_10409

theorem power_relations (a m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : a^m = 4) (h4 : a^n = 3) :
  a^(-m/2) = 1/2 ∧ a^(2*m-n) = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_power_relations_l104_10409


namespace NUMINAMATH_CALUDE_sisters_candy_count_l104_10497

theorem sisters_candy_count 
  (debbys_candy : ℕ) 
  (eaten_candy : ℕ) 
  (remaining_candy : ℕ) 
  (h1 : debbys_candy = 32) 
  (h2 : eaten_candy = 35) 
  (h3 : remaining_candy = 39) : 
  ∃ (sisters_candy : ℕ), 
    sisters_candy = 42 ∧ 
    debbys_candy + sisters_candy = eaten_candy + remaining_candy :=
by
  sorry

end NUMINAMATH_CALUDE_sisters_candy_count_l104_10497


namespace NUMINAMATH_CALUDE_rectangular_field_ratio_l104_10452

theorem rectangular_field_ratio (perimeter width : ℝ) :
  perimeter = 432 →
  width = 90 →
  let length := (perimeter - 2 * width) / 2
  (length / width) = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_ratio_l104_10452


namespace NUMINAMATH_CALUDE_solution_set_supremum_a_l104_10453

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem 1: The solution set of f(x) > 3
theorem solution_set (x : ℝ) : f x > 3 ↔ x < 0 ∨ x > 3 := by sorry

-- Theorem 2: The supremum of a for which f(x) > a holds for all x
theorem supremum_a : ∀ a : ℝ, (∀ x : ℝ, f x > a) ↔ a < 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_supremum_a_l104_10453


namespace NUMINAMATH_CALUDE_tennis_racket_price_l104_10422

theorem tennis_racket_price
  (sneakers_cost sports_outfit_cost total_spent : ℝ)
  (racket_discount sales_tax : ℝ)
  (h1 : sneakers_cost = 200)
  (h2 : sports_outfit_cost = 250)
  (h3 : racket_discount = 0.2)
  (h4 : sales_tax = 0.1)
  (h5 : total_spent = 750)
  : ∃ (original_price : ℝ),
    (1 + sales_tax) * ((1 - racket_discount) * original_price + sneakers_cost + sports_outfit_cost) = total_spent ∧
    original_price = 255 / 0.88 :=
by sorry

end NUMINAMATH_CALUDE_tennis_racket_price_l104_10422


namespace NUMINAMATH_CALUDE_track_length_track_length_is_350_l104_10460

/-- The length of a circular track given specific running conditions -/
theorem track_length : ℝ → ℝ → ℝ → Prop :=
  λ first_meet second_meet track_length =>
    -- Brenda and Sally start at diametrically opposite points
    -- They first meet after Brenda has run 'first_meet' meters
    -- They next meet after Sally has run 'second_meet' meters past their first meeting point
    -- 'track_length' is the length of the circular track
    first_meet = 150 ∧
    second_meet = 200 ∧
    track_length = 350 ∧
    -- The total distance run by both runners is twice the track length
    2 * track_length = 2 * first_meet + second_meet

theorem track_length_is_350 : ∃ (l : ℝ), track_length 150 200 l :=
  sorry

end NUMINAMATH_CALUDE_track_length_track_length_is_350_l104_10460


namespace NUMINAMATH_CALUDE_hotdog_cost_l104_10491

theorem hotdog_cost (h s : ℕ) : 
  3 * h + 2 * s = 360 →
  2 * h + 3 * s = 390 →
  h = 60 := by sorry

end NUMINAMATH_CALUDE_hotdog_cost_l104_10491


namespace NUMINAMATH_CALUDE_most_likely_genotype_combination_l104_10435

/-- Represents the alleles for rabbit fur type -/
inductive Allele
| H  -- Dominant hairy allele
| h  -- Recessive hairy allele
| S  -- Dominant smooth allele
| s  -- Recessive smooth allele

/-- Represents the genotype of a rabbit -/
structure Genotype where
  allele1 : Allele
  allele2 : Allele

/-- Determines if a rabbit has hairy fur based on its genotype -/
def hasHairyFur (g : Genotype) : Bool :=
  match g.allele1, g.allele2 with
  | Allele.H, _ => true
  | _, Allele.H => true
  | _, _ => false

/-- The probability of the hairy fur allele in the population -/
def p : ℝ := 0.1

/-- Represents the result of mating two rabbits -/
structure MatingResult where
  parent1 : Genotype
  parent2 : Genotype
  offspringCount : Nat
  allOffspringHairy : Bool

/-- The theorem to be proved -/
theorem most_likely_genotype_combination (result : MatingResult) 
  (h1 : result.parent1.allele1 = Allele.H ∨ result.parent1.allele2 = Allele.H)
  (h2 : result.parent2.allele1 = Allele.S ∨ result.parent2.allele2 = Allele.S)
  (h3 : result.offspringCount = 4)
  (h4 : result.allOffspringHairy = true) :
  (result.parent1 = Genotype.mk Allele.H Allele.H ∧ 
   result.parent2 = Genotype.mk Allele.S Allele.h) :=
sorry

end NUMINAMATH_CALUDE_most_likely_genotype_combination_l104_10435


namespace NUMINAMATH_CALUDE_travel_time_equation_l104_10402

theorem travel_time_equation (x : ℝ) : x > 3 → 
  (30 / (x - 3) - 30 / x = 40 / 60) ↔ 
  (30 = (x - 3) * (40 / 60) ∧ 30 = x * ((40 / 60) + (30 / (x - 3)))) := by
  sorry

#check travel_time_equation

end NUMINAMATH_CALUDE_travel_time_equation_l104_10402


namespace NUMINAMATH_CALUDE_april_price_index_april_price_increase_l104_10443

/-- Represents the price index for a given month -/
structure PriceIndex where
  month : Nat
  value : Real

/-- Calculates the price index for a given month based on the initial index and monthly decrease rate -/
def calculate_price_index (initial_index : Real) (monthly_decrease : Real) (month : Nat) : Real :=
  initial_index - (month - 1) * monthly_decrease

/-- Theorem stating that the price index in April is 1.12 given the conditions -/
theorem april_price_index 
  (january_index : PriceIndex)
  (monthly_decrease : Real)
  (h1 : january_index.month = 1)
  (h2 : january_index.value = 1.15)
  (h3 : monthly_decrease = 0.01)
  : ∃ (april_index : PriceIndex), 
    april_index.month = 4 ∧ 
    april_index.value = calculate_price_index january_index.value monthly_decrease 4 ∧
    april_index.value = 1.12 :=
sorry

/-- Theorem stating that the price in April has increased by 12% compared to the same month last year -/
theorem april_price_increase 
  (april_index : PriceIndex)
  (h : april_index.value = 1.12)
  : (april_index.value - 1) * 100 = 12 :=
sorry

end NUMINAMATH_CALUDE_april_price_index_april_price_increase_l104_10443


namespace NUMINAMATH_CALUDE_min_sum_product_l104_10427

theorem min_sum_product (a1 a2 a3 b1 b2 b3 c1 c2 c3 d : ℕ) : 
  (({a1, a2, a3, b1, b2, b3, c1, c2, c3, d} : Finset ℕ) = Finset.range 10) →
  a1 * a2 * a3 + b1 * b2 * b3 + c1 * c2 * c3 + d ≥ 609 ∧
  ∃ (p1 p2 p3 q1 q2 q3 r1 r2 r3 s : ℕ),
    ({p1, p2, p3, q1, q2, q3, r1, r2, r3, s} : Finset ℕ) = Finset.range 10 ∧
    p1 * p2 * p3 + q1 * q2 * q3 + r1 * r2 * r3 + s = 609 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_product_l104_10427


namespace NUMINAMATH_CALUDE_binomial_12_3_l104_10482

theorem binomial_12_3 : Nat.choose 12 3 = 220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_3_l104_10482


namespace NUMINAMATH_CALUDE_transportation_cost_comparison_l104_10423

/-- The cost function for company A -/
def cost_A (x : ℝ) : ℝ := 0.6 * x

/-- The cost function for company B -/
def cost_B (x : ℝ) : ℝ := 0.3 * x + 750

theorem transportation_cost_comparison (x : ℝ) 
  (h_x_pos : 0 < x) (h_x_upper : x < 5000) :
  (x < 2500 → cost_A x < cost_B x) ∧
  (x > 2500 → cost_B x < cost_A x) ∧
  (x = 2500 → cost_A x = cost_B x) := by
  sorry


end NUMINAMATH_CALUDE_transportation_cost_comparison_l104_10423
