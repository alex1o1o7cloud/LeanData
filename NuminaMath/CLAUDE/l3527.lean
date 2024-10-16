import Mathlib

namespace NUMINAMATH_CALUDE_expected_rounds_range_l3527_352739

/-- Represents the game between players A and B -/
structure Game where
  p : ℝ
  h_p_pos : 0 < p
  h_p_lt_one : p < 1

/-- The expected number of rounds in the game -/
noncomputable def expected_rounds (g : Game) : ℝ :=
  2 * (1 - (2 * g.p * (1 - g.p))^10) / (1 - 2 * g.p * (1 - g.p))

/-- Theorem stating the range of the expected number of rounds -/
theorem expected_rounds_range (g : Game) :
  2 < expected_rounds g ∧ expected_rounds g ≤ 4 - (1/2)^8 :=
sorry

end NUMINAMATH_CALUDE_expected_rounds_range_l3527_352739


namespace NUMINAMATH_CALUDE_division_remainder_problem_l3527_352714

theorem division_remainder_problem (N : ℕ) (Q2 : ℕ) :
  (∃ R1 : ℕ, N = 44 * 432 + R1) ∧ 
  (∃ Q2 : ℕ, N = 38 * Q2 + 8) →
  N % 44 = 0 :=
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l3527_352714


namespace NUMINAMATH_CALUDE_complex_sum_of_parts_l3527_352722

theorem complex_sum_of_parts (z : ℂ) (h : z * Complex.I = -1 + Complex.I) : 
  z.re + z.im = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_of_parts_l3527_352722


namespace NUMINAMATH_CALUDE_point_in_region_l3527_352709

def satisfies_inequality (x y : ℝ) : Prop := 3 + 2*y < 6

theorem point_in_region :
  satisfies_inequality 1 1 ∧
  ¬(satisfies_inequality 0 0 ∧ satisfies_inequality 0 2 ∧ satisfies_inequality 2 0) :=
by sorry

end NUMINAMATH_CALUDE_point_in_region_l3527_352709


namespace NUMINAMATH_CALUDE_extra_apples_l3527_352723

-- Define the number of red apples
def red_apples : ℕ := 6

-- Define the number of green apples
def green_apples : ℕ := 15

-- Define the number of students who wanted fruit
def students_wanting_fruit : ℕ := 5

-- Define the number of apples each student takes
def apples_per_student : ℕ := 1

-- Theorem to prove
theorem extra_apples : 
  (red_apples + green_apples) - (students_wanting_fruit * apples_per_student) = 16 := by
  sorry

end NUMINAMATH_CALUDE_extra_apples_l3527_352723


namespace NUMINAMATH_CALUDE_ball_probability_l3527_352717

theorem ball_probability (m : ℕ) : 
  (3 : ℝ) / ((m : ℝ) + 3) = (1 : ℝ) / 4 → m = 9 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l3527_352717


namespace NUMINAMATH_CALUDE_smallest_five_digit_palindrome_div_by_6_l3527_352718

/-- A function that checks if a number is a five-digit palindrome -/
def is_five_digit_palindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧ 
  (n / 10000 = n % 10) ∧ 
  ((n / 1000) % 10 = (n / 10) % 10)

/-- The theorem stating that 20002 is the smallest five-digit palindrome divisible by 6 -/
theorem smallest_five_digit_palindrome_div_by_6 :
  ∀ n : ℕ, is_five_digit_palindrome n ∧ n % 6 = 0 → n ≥ 20002 :=
sorry

end NUMINAMATH_CALUDE_smallest_five_digit_palindrome_div_by_6_l3527_352718


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l3527_352763

def is_divisible_by (n m : ℕ) : Prop := m ∣ n

def has_additional_prime_factors (n : ℕ) (count : ℕ) : Prop :=
  ∃ (factors : List ℕ), 
    factors.length = count ∧ 
    (∀ p ∈ factors, Nat.Prime p) ∧
    (∀ p ∈ factors, p ∣ n) ∧
    (∀ p ∈ factors, p ≠ 2 ∧ p ≠ 5)

theorem smallest_number_with_conditions : 
  (∀ n : ℕ, n < 840 → 
    ¬(is_divisible_by n 8 ∧ 
      is_divisible_by n 5 ∧ 
      has_additional_prime_factors n 2)) ∧
  (is_divisible_by 840 8 ∧ 
   is_divisible_by 840 5 ∧ 
   has_additional_prime_factors 840 2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l3527_352763


namespace NUMINAMATH_CALUDE_max_abs_z_on_circle_l3527_352760

open Complex

theorem max_abs_z_on_circle (z : ℂ) : 
  (abs (z - 2*I) = 1) → (abs z ≤ 3) ∧ ∃ w : ℂ, abs (w - 2*I) = 1 ∧ abs w = 3 :=
by sorry

end NUMINAMATH_CALUDE_max_abs_z_on_circle_l3527_352760


namespace NUMINAMATH_CALUDE_samantha_buys_four_toys_l3527_352780

/-- Represents the number of toys Samantha buys -/
def num_toys : ℕ := 4

/-- Represents the cost of each toy before discount -/
def toy_cost : ℚ := 12

/-- Represents the total amount Samantha spends -/
def total_spent : ℚ := 36

/-- Calculates the cost of n toys with the "buy one get one half off" promotion -/
def promotion_cost (n : ℕ) : ℚ :=
  (n / 2 : ℚ) * toy_cost + (n - n / 2 : ℚ) * (toy_cost / 2)

theorem samantha_buys_four_toys :
  promotion_cost num_toys = total_spent :=
by sorry

end NUMINAMATH_CALUDE_samantha_buys_four_toys_l3527_352780


namespace NUMINAMATH_CALUDE_moles_of_CO₂_equals_two_l3527_352774

/-- Represents a chemical compound --/
inductive Compound
| HNO₃
| NaHCO₃
| NH₄Cl
| NaNO₃
| H₂O
| CO₂
| NH₄NO₃
| HCl

/-- Represents a chemical reaction --/
structure Reaction :=
(reactants : List (Compound × ℕ))
(products : List (Compound × ℕ))

/-- Represents the two-step reaction process --/
def two_step_reaction : List Reaction :=
[
  { reactants := [(Compound.NaHCO₃, 1), (Compound.HNO₃, 1)],
    products := [(Compound.NaNO₃, 1), (Compound.H₂O, 1), (Compound.CO₂, 1)] },
  { reactants := [(Compound.NH₄Cl, 1), (Compound.HNO₃, 1)],
    products := [(Compound.NH₄NO₃, 1), (Compound.HCl, 1)] }
]

/-- Initial amounts of compounds --/
def initial_amounts : List (Compound × ℕ) :=
[(Compound.HNO₃, 2), (Compound.NaHCO₃, 2), (Compound.NH₄Cl, 1)]

/-- Calculates the moles of CO₂ formed in the two-step reaction --/
def moles_of_CO₂_formed (reactions : List Reaction) (initial : List (Compound × ℕ)) : ℕ :=
sorry

/-- Theorem stating that the moles of CO₂ formed is 2 --/
theorem moles_of_CO₂_equals_two :
  moles_of_CO₂_formed two_step_reaction initial_amounts = 2 :=
sorry

end NUMINAMATH_CALUDE_moles_of_CO₂_equals_two_l3527_352774


namespace NUMINAMATH_CALUDE_sqrt_ratio_equality_implies_y_value_l3527_352727

theorem sqrt_ratio_equality_implies_y_value (y : ℝ) :
  y > 2 →
  (Real.sqrt (7 * y)) / (Real.sqrt (4 * (y - 2))) = 3 →
  y = 72 / 29 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_ratio_equality_implies_y_value_l3527_352727


namespace NUMINAMATH_CALUDE_rectangle_area_from_circular_wire_l3527_352777

/-- The area of a rectangle formed by bending a circular wire -/
theorem rectangle_area_from_circular_wire (r : ℝ) (ratio_l : ℝ) (ratio_b : ℝ) : 
  r = 3.5 → 
  ratio_l = 6 → 
  ratio_b = 5 → 
  let circumference := 2 * π * r
  let length := (circumference * ratio_l) / (2 * (ratio_l + ratio_b))
  let breadth := (circumference * ratio_b) / (2 * (ratio_l + ratio_b))
  length * breadth = (735 * π^2) / 242 := by
  sorry

#check rectangle_area_from_circular_wire

end NUMINAMATH_CALUDE_rectangle_area_from_circular_wire_l3527_352777


namespace NUMINAMATH_CALUDE_walking_problem_l3527_352790

/-- Represents the walking problem from "The Nine Chapters on the Mathematical Art" -/
theorem walking_problem (x : ℝ) :
  (∀ d : ℝ, d > 0 → 100 * (d / 60) = d) →  -- Good walker takes 100 steps for every 60 steps of bad walker
  x = 100 + (60 / 100) * x →               -- The equation to be proved
  x = (100 * 100) / 40                     -- The solution (not given in the original problem, but included for completeness)
  := by sorry

end NUMINAMATH_CALUDE_walking_problem_l3527_352790


namespace NUMINAMATH_CALUDE_smallest_m_no_real_roots_l3527_352730

theorem smallest_m_no_real_roots : 
  let equation (m x : ℝ) := 3 * x * ((m + 1) * x - 5) - x^2 + 8
  ∀ m : ℤ, (∀ x : ℝ, equation m x ≠ 0) → m ≥ 2 ∧ 
  ∃ m' : ℤ, m' < 2 ∧ ∃ x : ℝ, equation (m' : ℝ) x = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_no_real_roots_l3527_352730


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l3527_352746

/-- Given a quarter circle sector with radius 5, the radius of the inscribed circle
    tangent to both radii and the arc is 5√2 - 5. -/
theorem inscribed_circle_radius (r : ℝ) : 
  r > 0 ∧ 
  r * (1 + Real.sqrt 2) = 5 → 
  r = 5 * Real.sqrt 2 - 5 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l3527_352746


namespace NUMINAMATH_CALUDE_young_inequality_l3527_352766

theorem young_inequality (p q a b : ℝ) (hp : 0 < p) (hq : 0 < q) (hpq : 1/p + 1/q = 1) (ha : 0 < a) (hb : 0 < b) :
  a * b ≤ a^p / p + b^q / q := by
  sorry

end NUMINAMATH_CALUDE_young_inequality_l3527_352766


namespace NUMINAMATH_CALUDE_middle_circle_radius_l3527_352757

/-- A sequence of five circles tangent to two parallel lines and to each other -/
structure CircleSequence where
  radii : Fin 5 → ℝ
  tangent_to_lines : Bool
  sequentially_tangent : Bool

/-- The property that the radii form a geometric sequence -/
def is_geometric_sequence (cs : CircleSequence) : Prop :=
  ∃ r : ℝ, ∀ i : Fin 4, cs.radii i.succ = cs.radii i * r

theorem middle_circle_radius 
  (cs : CircleSequence)
  (h_tangent : cs.tangent_to_lines = true)
  (h_seq_tangent : cs.sequentially_tangent = true)
  (h_geometric : is_geometric_sequence cs)
  (h_smallest : cs.radii 0 = 8)
  (h_largest : cs.radii 4 = 18) :
  cs.radii 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_middle_circle_radius_l3527_352757


namespace NUMINAMATH_CALUDE_platform_length_l3527_352712

/-- Given a train of length 600 m that takes 78 seconds to cross a platform
    and 52 seconds to cross a signal pole, prove that the length of the platform is 300 m. -/
theorem platform_length (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ)
  (h1 : train_length = 600)
  (h2 : time_platform = 78)
  (h3 : time_pole = 52) :
  (train_length * time_platform / time_pole) - train_length = 300 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l3527_352712


namespace NUMINAMATH_CALUDE_remainder_of_2367905_div_5_l3527_352734

theorem remainder_of_2367905_div_5 : 2367905 % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_2367905_div_5_l3527_352734


namespace NUMINAMATH_CALUDE_smallest_n_not_divisible_by_ten_l3527_352783

theorem smallest_n_not_divisible_by_ten (n : ℕ) :
  (n > 2016 ∧ ¬(10 ∣ (1^n + 2^n + 3^n + 4^n)) ∧
   ∀ m, 2016 < m ∧ m < n → (10 ∣ (1^m + 2^m + 3^m + 4^m))) →
  n = 2020 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_not_divisible_by_ten_l3527_352783


namespace NUMINAMATH_CALUDE_special_polynomial_is_x_squared_plus_one_l3527_352755

/-- A polynomial satisfying specific conditions -/
def SpecialPolynomial (p : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, p (x * y) = p x * p y - p x - p y + 2) ∧
  p 3 = 10 ∧
  p 4 = 17

/-- The theorem stating that the special polynomial is x^2 + 1 -/
theorem special_polynomial_is_x_squared_plus_one 
  (p : ℝ → ℝ) (hp : SpecialPolynomial p) :
  ∀ x : ℝ, p x = x^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_is_x_squared_plus_one_l3527_352755


namespace NUMINAMATH_CALUDE_new_alcohol_percentage_l3527_352719

/-- Calculates the new alcohol percentage after adding alcohol and water to a solution -/
theorem new_alcohol_percentage
  (initial_volume : ℝ)
  (initial_percentage : ℝ)
  (added_alcohol : ℝ)
  (added_water : ℝ)
  (h1 : initial_volume = 40)
  (h2 : initial_percentage = 0.05)
  (h3 : added_alcohol = 3.5)
  (h4 : added_water = 6.5) :
  let initial_alcohol := initial_volume * initial_percentage
  let new_alcohol := initial_alcohol + added_alcohol
  let new_volume := initial_volume + added_alcohol + added_water
  let new_percentage := new_alcohol / new_volume
  new_percentage = 0.11 :=
sorry

end NUMINAMATH_CALUDE_new_alcohol_percentage_l3527_352719


namespace NUMINAMATH_CALUDE_total_deposit_l3527_352708

def mark_deposit : ℕ := 88
def bryan_deposit : ℕ := 5 * mark_deposit - 40

theorem total_deposit : mark_deposit + bryan_deposit = 488 := by
  sorry

end NUMINAMATH_CALUDE_total_deposit_l3527_352708


namespace NUMINAMATH_CALUDE_sandra_brought_twenty_pairs_l3527_352756

/-- Calculates the number of sock pairs Sandra brought given the initial conditions --/
def sandras_socks (initial_pairs : ℕ) (final_pairs : ℕ) : ℕ :=
  let moms_pairs := 3 * initial_pairs + 8
  let s := (final_pairs - initial_pairs - moms_pairs) * 5 / 6
  s

theorem sandra_brought_twenty_pairs :
  sandras_socks 12 80 = 20 := by
  sorry

end NUMINAMATH_CALUDE_sandra_brought_twenty_pairs_l3527_352756


namespace NUMINAMATH_CALUDE_field_trip_students_l3527_352705

/-- Proves that the number of students on a field trip is equal to the product of seats per bus and number of buses -/
theorem field_trip_students (seats_per_bus : ℕ) (num_buses : ℕ) (h1 : seats_per_bus = 9) (h2 : num_buses = 5) :
  seats_per_bus * num_buses = 45 := by
  sorry

#check field_trip_students

end NUMINAMATH_CALUDE_field_trip_students_l3527_352705


namespace NUMINAMATH_CALUDE_game_solution_l3527_352764

def game_result (x y z : ℚ) : Prop :=
  let a1 := x + y/3 + z/3
  let b1 := 2*y/3
  let c1 := 2*z/3
  let a2 := 2*a1/3
  let b2 := b1 + c1/3
  let c2 := 2*c1/3
  let a3 := 2*a2/3
  let b3 := 2*b2/3
  let c3 := c2 + b2/3 + a2/3
  x - a3 = 2 ∧ c3 - z = 2*z + 8 ∧ x + y + z < 1000

theorem game_solution :
  ∃ x y z : ℚ, game_result x y z ∧ x = 54 ∧ y = 162 ∧ z = 27 :=
by sorry

end NUMINAMATH_CALUDE_game_solution_l3527_352764


namespace NUMINAMATH_CALUDE_exponent_division_l3527_352794

theorem exponent_division (a : ℝ) : a^12 / a^6 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l3527_352794


namespace NUMINAMATH_CALUDE_factorial_fraction_equality_l3527_352725

theorem factorial_fraction_equality : (4 * Nat.factorial 6 + 20 * Nat.factorial 5) / Nat.factorial 7 = 22 / 21 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equality_l3527_352725


namespace NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l3527_352707

theorem smallest_part_of_proportional_division (total : ℚ) (a b c : ℚ) 
  (h_total : total = 150)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_prop : ∃ (k : ℚ), a = 3 * k ∧ b = 5 * k ∧ c = (7/2) * k)
  (h_sum : a + b + c = total) :
  min a (min b c) = 900 / 23 := by
sorry

end NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l3527_352707


namespace NUMINAMATH_CALUDE_equilateral_triangle_sum_product_l3527_352748

def is_equilateral_triangle (x y z : ℂ) : Prop :=
  Complex.abs (y - x) = Complex.abs (z - y) ∧ 
  Complex.abs (z - y) = Complex.abs (x - z) ∧
  Complex.abs (x - z) = Complex.abs (y - x)

theorem equilateral_triangle_sum_product (x y z : ℂ) :
  is_equilateral_triangle x y z →
  Complex.abs (y - x) = 24 →
  Complex.abs (x + y + z) = 72 →
  Complex.abs (x * y + x * z + y * z) = 1728 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_sum_product_l3527_352748


namespace NUMINAMATH_CALUDE_greatest_k_value_l3527_352773

theorem greatest_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 85) →
  k ≤ Real.sqrt 117 :=
sorry

end NUMINAMATH_CALUDE_greatest_k_value_l3527_352773


namespace NUMINAMATH_CALUDE_line_relationship_undetermined_l3527_352772

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A parabola defined by y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a parabola -/
def pointOnParabola (p : Point) (para : Parabola) : Prop :=
  p.y = para.a * p.x^2 + para.b * p.x + para.c

/-- The relationship between two lines -/
inductive LineRelationship
  | Parallel
  | Perpendicular
  | Intersecting

/-- Theorem: The relationship between AD and BC cannot be determined -/
theorem line_relationship_undetermined 
  (A B C D : Point) 
  (para : Parabola) 
  (h_a_nonzero : para.a ≠ 0)
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_on_parabola : pointOnParabola A para ∧ pointOnParabola B para ∧ 
                   pointOnParabola C para ∧ pointOnParabola D para)
  (h_x_sum : A.x + D.x - B.x + C.x = 0) :
  ∀ r : LineRelationship, ∃ para' : Parabola, 
    para'.a ≠ 0 ∧
    (pointOnParabola A para' ∧ pointOnParabola B para' ∧ 
     pointOnParabola C para' ∧ pointOnParabola D para') ∧
    A.x + D.x - B.x + C.x = 0 :=
  sorry

end NUMINAMATH_CALUDE_line_relationship_undetermined_l3527_352772


namespace NUMINAMATH_CALUDE_total_baseball_cards_l3527_352762

theorem total_baseball_cards (rob_doubles jess_doubles alex_doubles rob_total alex_total : ℕ) : 
  rob_doubles = 8 →
  jess_doubles = 40 →
  alex_doubles = 12 →
  rob_total = 24 →
  alex_total = 48 →
  rob_doubles * 3 = rob_total →
  jess_doubles = 5 * rob_doubles →
  alex_total = 2 * rob_total →
  alex_doubles * 4 = alex_total →
  rob_total + jess_doubles + alex_total = 112 := by
  sorry

end NUMINAMATH_CALUDE_total_baseball_cards_l3527_352762


namespace NUMINAMATH_CALUDE_sqrt500_approx_l3527_352789

/-- Approximate value of √5 -/
def sqrt5_approx : ℝ := 2.236

/-- Theorem stating that √500 is approximately 22.36 -/
theorem sqrt500_approx : ‖Real.sqrt 500 - 22.36‖ < 0.01 :=
  sorry

end NUMINAMATH_CALUDE_sqrt500_approx_l3527_352789


namespace NUMINAMATH_CALUDE_min_m_value_l3527_352742

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin x ^ 2 + 4 * Real.sqrt 3 * Real.sin x * Real.cos x + 5

theorem min_m_value (m : ℝ) :
  (∃ x, x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x ≤ m) →
  m ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_min_m_value_l3527_352742


namespace NUMINAMATH_CALUDE_smallest_multiple_thirty_two_works_smallest_x_is_32_l3527_352740

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 800 ∣ (450 * x) → x ≥ 32 :=
by sorry

theorem thirty_two_works : 800 ∣ (450 * 32) :=
by sorry

theorem smallest_x_is_32 : ∃ x : ℕ, x > 0 ∧ 800 ∣ (450 * x) ∧ ∀ y : ℕ, (y > 0 ∧ 800 ∣ (450 * y)) → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_thirty_two_works_smallest_x_is_32_l3527_352740


namespace NUMINAMATH_CALUDE_hadley_walk_back_home_l3527_352704

/-- The distance Hadley walked back home -/
def distance_back_home (distance_to_grocery : ℝ) (distance_to_pet : ℝ) (total_distance : ℝ) : ℝ :=
  total_distance - (distance_to_grocery + distance_to_pet)

/-- Theorem: Hadley walked 3 miles back home -/
theorem hadley_walk_back_home :
  let distance_to_grocery : ℝ := 2
  let distance_to_pet : ℝ := 2 - 1
  let total_distance : ℝ := 6
  distance_back_home distance_to_grocery distance_to_pet total_distance = 3 := by
sorry

end NUMINAMATH_CALUDE_hadley_walk_back_home_l3527_352704


namespace NUMINAMATH_CALUDE_triangle_properties_l3527_352765

/-- Given a triangle ABC with vertices A(0,1), B(0,-1), and C(-2,1) -/
def triangle_ABC : Set (ℝ × ℝ) := {(0, 1), (0, -1), (-2, 1)}

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The equation of a circle in the form x^2 + y^2 + Dx + Ey + F = 0 -/
structure CircleEquation where
  D : ℝ
  E : ℝ
  F : ℝ

/-- Theorem stating the equations of altitude, midline, and circumcircle -/
theorem triangle_properties (ABC : Set (ℝ × ℝ)) 
  (h : ABC = triangle_ABC) : 
  ∃ (altitude_eq : LineEquation) 
    (midline_eq : LineEquation) 
    (circumcircle_eq : CircleEquation),
  altitude_eq = ⟨1, -1, 1⟩ ∧ 
  midline_eq = ⟨1, 0, 1⟩ ∧
  circumcircle_eq = ⟨2, 0, -1⟩ := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3527_352765


namespace NUMINAMATH_CALUDE_square_ratio_side_length_l3527_352779

theorem square_ratio_side_length (area_ratio : ℚ) :
  area_ratio = 270 / 125 →
  ∃ (a b c : ℕ), 
    (a = 3 ∧ b = 30 ∧ c = 25) ∧
    (Real.sqrt area_ratio = a * Real.sqrt b / c) ∧
    (a + b + c = 58) := by
  sorry

end NUMINAMATH_CALUDE_square_ratio_side_length_l3527_352779


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l3527_352770

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 * x - 1) → x ≥ 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l3527_352770


namespace NUMINAMATH_CALUDE_a_range_l3527_352750

def A (a : ℝ) : Set ℝ := {x | -3 ≤ x ∧ x ≤ a}

def B (a : ℝ) : Set ℝ := {y | ∃ x ∈ A a, y = 3*x + 10}

def C (a : ℝ) : Set ℝ := {z | ∃ x ∈ A a, z = 5 - x}

theorem a_range (a : ℝ) : B a ∩ C a = C a → a ∈ Set.Icc (-2/3) 4 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l3527_352750


namespace NUMINAMATH_CALUDE_x_mod_105_l3527_352711

theorem x_mod_105 (x : ℤ) 
  (h1 : (3 + x) % (3^3) = 2^2 % (3^3))
  (h2 : (5 + x) % (5^3) = 3^2 % (5^3))
  (h3 : (7 + x) % (7^3) = 5^2 % (7^3)) :
  x % 105 = 4 := by
  sorry

end NUMINAMATH_CALUDE_x_mod_105_l3527_352711


namespace NUMINAMATH_CALUDE_expression_equality_l3527_352775

theorem expression_equality : (3 / 5 : ℚ) * ((2 / 3 + 3 / 8) / 2) - 1 / 16 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3527_352775


namespace NUMINAMATH_CALUDE_baseball_cards_per_pack_l3527_352758

theorem baseball_cards_per_pack : 
  ∀ (total_cards : ℕ) (num_people : ℕ) (cards_per_person : ℕ) (total_packs : ℕ),
    num_people = 4 →
    cards_per_person = 540 →
    total_packs = 108 →
    total_cards = num_people * cards_per_person →
    total_cards / total_packs = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_baseball_cards_per_pack_l3527_352758


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l3527_352749

/-- The number of combinations of k items chosen from n items -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of available toppings -/
def num_toppings : ℕ := 7

/-- The number of toppings to choose -/
def toppings_to_choose : ℕ := 3

theorem pizza_toppings_combinations :
  binomial num_toppings toppings_to_choose = 35 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l3527_352749


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3527_352745

theorem quadratic_factorization (a b : ℤ) : 
  (∀ x, 25 * x^2 - 115 * x - 150 = (5 * x + a) * (5 * x + b)) →
  a + 2 * b = -55 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3527_352745


namespace NUMINAMATH_CALUDE_cube_edges_sum_l3527_352743

/-- Given a cube-shaped toy made up of 27 small cubes, with the total length of all edges
    of the large cube being 82.8 cm, prove that the sum of the length of one edge of the
    large cube and one edge of a small cube is 9.2 cm. -/
theorem cube_edges_sum (total_edge_length : ℝ) (num_small_cubes : ℕ) :
  total_edge_length = 82.8 ∧ num_small_cubes = 27 →
  ∃ (large_edge small_edge : ℝ),
    large_edge = total_edge_length / 12 ∧
    small_edge = large_edge / 3 ∧
    large_edge + small_edge = 9.2 :=
by sorry

end NUMINAMATH_CALUDE_cube_edges_sum_l3527_352743


namespace NUMINAMATH_CALUDE_gcd_of_256_180_600_l3527_352713

theorem gcd_of_256_180_600 : Nat.gcd 256 (Nat.gcd 180 600) = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_256_180_600_l3527_352713


namespace NUMINAMATH_CALUDE_jim_unknown_row_trees_l3527_352716

/-- Represents the production of a lemon grove over 5 years -/
structure LemonGrove where
  normal_production : ℕ  -- lemons per year for a normal tree
  increase_percent : ℕ   -- percentage increase for Jim's trees
  known_row : ℕ          -- number of trees in the known row
  total_production : ℕ   -- total lemons produced in 5 years

/-- Calculates the number of trees in the unknown row of Jim's lemon grove -/
def unknown_row_trees (grove : LemonGrove) : ℕ :=
  let jim_tree_production := grove.normal_production * (100 + grove.increase_percent) / 100
  let total_trees := grove.total_production / (jim_tree_production * 5)
  total_trees - grove.known_row

/-- Theorem stating the number of trees in the unknown row of Jim's lemon grove -/
theorem jim_unknown_row_trees :
  let grove := LemonGrove.mk 60 50 30 675000
  unknown_row_trees grove = 1470 := by
  sorry

end NUMINAMATH_CALUDE_jim_unknown_row_trees_l3527_352716


namespace NUMINAMATH_CALUDE_lcm_of_20_and_36_l3527_352700

theorem lcm_of_20_and_36 : Nat.lcm 20 36 = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_20_and_36_l3527_352700


namespace NUMINAMATH_CALUDE_airplane_stop_time_l3527_352771

/-- The distance function for an airplane after landing -/
def distance (t : ℝ) : ℝ := 75 * t - 1.5 * t^2

/-- The time at which the airplane stops -/
def stop_time : ℝ := 25

theorem airplane_stop_time :
  (∀ t : ℝ, distance t ≤ distance stop_time) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ t, |t - stop_time| < δ → distance stop_time - distance t < ε) :=
sorry

end NUMINAMATH_CALUDE_airplane_stop_time_l3527_352771


namespace NUMINAMATH_CALUDE_intersection_distance_l3527_352751

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 36 = 1

-- Define the parabola (we don't know its exact equation, but we know it exists)
def parabola (x y : ℝ) : Prop := ∃ (a b c : ℝ), y = a * x^2 + b * x + c

-- Define the shared focus condition
def shared_focus (e p : ℝ → ℝ → Prop) : Prop :=
  ∃ (x₀ y₀ : ℝ), (∀ x y, e x y → (x - x₀)^2 + (y - y₀)^2 = 20) ∧
                 (∀ x y, p x y → (x - x₀)^2 + (y - y₀)^2 ≤ 20)

-- Define the directrix condition
def directrix_on_major_axis (p : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), ∀ x y, p x y → y = k * x^2

-- Theorem statement
theorem intersection_distance :
  ∀ (p : ℝ → ℝ → Prop),
  shared_focus ellipse p →
  directrix_on_major_axis p →
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse x₁ y₁ ∧ p x₁ y₁ ∧
    ellipse x₂ y₂ ∧ p x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = (4 * Real.sqrt 5 / 3)^2 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_l3527_352751


namespace NUMINAMATH_CALUDE_tan_alpha_eq_neg_one_third_l3527_352753

theorem tan_alpha_eq_neg_one_third (α : ℝ) 
  (h : (Real.cos (π/4 - α)) / (Real.cos (π/4 + α)) = 1/2) : 
  Real.tan α = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_eq_neg_one_third_l3527_352753


namespace NUMINAMATH_CALUDE_remaining_wallpaper_time_l3527_352752

/-- Time to remove wallpaper from one wall -/
def time_per_wall : ℕ := 2

/-- Number of walls in dining room -/
def dining_walls : ℕ := 4

/-- Number of walls in living room -/
def living_walls : ℕ := 4

/-- Time already spent removing wallpaper -/
def time_spent : ℕ := 2

/-- Theorem: The remaining time to remove wallpaper is 14 hours -/
theorem remaining_wallpaper_time :
  (time_per_wall * dining_walls + time_per_wall * living_walls) - time_spent = 14 := by
  sorry

end NUMINAMATH_CALUDE_remaining_wallpaper_time_l3527_352752


namespace NUMINAMATH_CALUDE_polar_equation_is_line_and_circle_l3527_352796

-- Define the polar equation
def polar_equation (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 2 * Real.sin (2 * θ)

-- Define what it means for a curve to be a line in polar coordinates
def is_line (f : ℝ → ℝ → Prop) : Prop :=
  ∃ θ₀ : ℝ, ∀ ρ θ : ℝ, f ρ θ → θ = θ₀

-- Define what it means for a curve to be a circle in polar coordinates
def is_circle (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b r : ℝ, ∀ ρ θ : ℝ, f ρ θ → (ρ * Real.cos θ - a)^2 + (ρ * Real.sin θ - b)^2 = r^2

-- Theorem statement
theorem polar_equation_is_line_and_circle :
  is_line polar_equation ∧ is_circle polar_equation := by sorry

end NUMINAMATH_CALUDE_polar_equation_is_line_and_circle_l3527_352796


namespace NUMINAMATH_CALUDE_johns_quarters_l3527_352715

theorem johns_quarters (quarters dimes nickels : ℕ) : 
  quarters + dimes + nickels = 63 →
  dimes = quarters + 3 →
  nickels = quarters - 6 →
  quarters = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_johns_quarters_l3527_352715


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3527_352778

theorem negation_of_universal_proposition :
  (¬ ∀ n : ℕ, 3^n ≥ n^2 + 1) ↔ (∃ n₀ : ℕ, 3^n₀ < n₀^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3527_352778


namespace NUMINAMATH_CALUDE_Q_times_E_times_D_l3527_352785

def Q : ℂ := 3 + 4 * Complex.I
def E : ℂ := Complex.I ^ 2
def D : ℂ := 3 - 4 * Complex.I

theorem Q_times_E_times_D : Q * E * D = -25 := by
  sorry

end NUMINAMATH_CALUDE_Q_times_E_times_D_l3527_352785


namespace NUMINAMATH_CALUDE_multiple_of_seven_proposition_l3527_352792

theorem multiple_of_seven_proposition : 
  (∃ k : ℤ, 47 = 7 * k) ∨ (∃ m : ℤ, 49 = 7 * m) := by sorry

end NUMINAMATH_CALUDE_multiple_of_seven_proposition_l3527_352792


namespace NUMINAMATH_CALUDE_log_sum_condition_l3527_352736

theorem log_sum_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ a b, a > 1 ∧ b > 1 → Real.log a + Real.log b > 0) ∧
  (∃ a b, Real.log a + Real.log b > 0 ∧ ¬(a > 1 ∧ b > 1)) := by
  sorry

end NUMINAMATH_CALUDE_log_sum_condition_l3527_352736


namespace NUMINAMATH_CALUDE_inverse_f_at_negative_three_over_128_l3527_352786

noncomputable def f (x : ℝ) : ℝ := (x^7 - 1) / 4

theorem inverse_f_at_negative_three_over_128 :
  f⁻¹ (-3/128) = (29/32)^(1/7) := by
  sorry

end NUMINAMATH_CALUDE_inverse_f_at_negative_three_over_128_l3527_352786


namespace NUMINAMATH_CALUDE_infinitely_many_lovely_numbers_no_lovely_square_greater_than_one_l3527_352701

def is_lovely (n : ℕ+) : Prop :=
  ∃ (k : ℕ+) (d : Fin k → ℕ+),
    n = (Finset.range k).prod (λ i => d i) ∧
    ∀ i : Fin k, (d i)^2 ∣ (n + d i)

theorem infinitely_many_lovely_numbers :
  ∀ N : ℕ, ∃ n : ℕ+, n > N ∧ is_lovely n :=
sorry

theorem no_lovely_square_greater_than_one :
  ¬∃ m : ℕ+, m > 1 ∧ is_lovely (m^2) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_lovely_numbers_no_lovely_square_greater_than_one_l3527_352701


namespace NUMINAMATH_CALUDE_left_translation_exponential_l3527_352744

/-- Given a function f: ℝ → ℝ, we say it's a left translation by 2 units of g 
    if f(x) = g(x + 2) for all x ∈ ℝ -/
def is_left_translation_by_two (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g (x + 2)

/-- The theorem stating that if f is a left translation by 2 units of the function
    x ↦ 2^(2x-1), then f(x) = 2^(2x-5) for all x ∈ ℝ -/
theorem left_translation_exponential 
  (f : ℝ → ℝ) 
  (h : is_left_translation_by_two f (fun x ↦ 2^(2*x - 1))) :
  ∀ x, f x = 2^(2*x - 5) := by
  sorry

end NUMINAMATH_CALUDE_left_translation_exponential_l3527_352744


namespace NUMINAMATH_CALUDE_min_distance_MN_l3527_352731

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x - 2

-- Define a point on the parabola
def point_on_parabola (x y : ℝ) : Prop := parabola x y

-- Define a line passing through F(0,1)
def line_through_F (k : ℝ) (x y : ℝ) : Prop := y = k*x + 1

-- Define the intersection of a line with the parabola
def intersection_line_parabola (k : ℝ) (x y : ℝ) : Prop :=
  point_on_parabola x y ∧ line_through_F k x y

-- Define the line AO (or BO)
def line_AO (x₁ y₁ : ℝ) (x y : ℝ) : Prop := y = (y₁/x₁) * x

-- Define the intersection of line AO (or BO) with line l
def intersection_AO_l (x₁ y₁ : ℝ) (x y : ℝ) : Prop :=
  line_AO x₁ y₁ x y ∧ line_l x y

-- The main theorem
theorem min_distance_MN :
  ∃ (min_dist : ℝ),
    min_dist = 8 * Real.sqrt 2 / 5 ∧
    ∀ (k : ℝ) (x₁ y₁ x₂ y₂ xM yM xN yN : ℝ),
      intersection_line_parabola k x₁ y₁ →
      intersection_line_parabola k x₂ y₂ →
      intersection_AO_l x₁ y₁ xM yM →
      intersection_AO_l x₂ y₂ xN yN →
      Real.sqrt ((xM - xN)^2 + (yM - yN)^2) ≥ min_dist :=
sorry

end NUMINAMATH_CALUDE_min_distance_MN_l3527_352731


namespace NUMINAMATH_CALUDE_john_squat_wrap_vs_sleeves_l3527_352767

/-- Given a raw squat weight, calculates the difference between
    the additional weight from wraps versus sleeves -/
def wrapVsSleevesDifference (rawSquat : ℝ) : ℝ :=
  0.25 * rawSquat - 30

theorem john_squat_wrap_vs_sleeves :
  wrapVsSleevesDifference 600 = 120 := by
  sorry

end NUMINAMATH_CALUDE_john_squat_wrap_vs_sleeves_l3527_352767


namespace NUMINAMATH_CALUDE_quadratic_root_range_l3527_352787

theorem quadratic_root_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - 4*x + a = 0 → x > 1) → 
  (3 < a ∧ a ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l3527_352787


namespace NUMINAMATH_CALUDE_plot_length_is_60_l3527_352729

/-- Represents a rectangular plot with its dimensions and fencing cost. -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  fencingCostPerMetre : ℝ
  totalFencingCost : ℝ

/-- The length of the plot is 20 metres more than its breadth. -/
def lengthCondition (plot : RectangularPlot) : Prop :=
  plot.length = plot.breadth + 20

/-- The cost of fencing the plot at the given rate equals the total fencing cost. -/
def fencingCostCondition (plot : RectangularPlot) : Prop :=
  plot.fencingCostPerMetre * (2 * plot.length + 2 * plot.breadth) = plot.totalFencingCost

/-- The theorem stating that under the given conditions, the length of the plot is 60 metres. -/
theorem plot_length_is_60 (plot : RectangularPlot)
    (h1 : lengthCondition plot)
    (h2 : fencingCostCondition plot)
    (h3 : plot.fencingCostPerMetre = 26.5)
    (h4 : plot.totalFencingCost = 5300) :
    plot.length = 60 := by
  sorry


end NUMINAMATH_CALUDE_plot_length_is_60_l3527_352729


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3527_352741

theorem rationalize_denominator : 
  3 / (Real.sqrt 5 - 2) = 3 * Real.sqrt 5 + 6 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3527_352741


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3527_352782

theorem complex_modulus_problem (i : ℂ) (h : i^2 = -1) :
  Complex.abs (i / (1 + i^3)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3527_352782


namespace NUMINAMATH_CALUDE_oplus_inequality_solutions_l3527_352799

def oplus (a b : ℤ) : ℤ := 1 - a * b

theorem oplus_inequality_solutions :
  (∃! (n : ℕ), ∀ (x : ℕ), oplus x 2 ≥ -3 ↔ x ≤ n) ∧
  (∃ (s : Finset ℕ), s.card = 3 ∧ ∀ (x : ℕ), x ∈ s ↔ oplus x 2 ≥ -3) :=
by sorry

end NUMINAMATH_CALUDE_oplus_inequality_solutions_l3527_352799


namespace NUMINAMATH_CALUDE_square_of_binomial_constant_l3527_352768

theorem square_of_binomial_constant (a : ℚ) : 
  (∃ b : ℚ, ∀ x : ℚ, 9*x^2 + 27*x + a = (3*x + b)^2) → a = 81/4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_constant_l3527_352768


namespace NUMINAMATH_CALUDE_terminating_decimal_count_l3527_352784

/-- A fraction a/b has a terminating decimal representation if and only if
    the denominator b can be factored as 2^m * 5^n * d, where d is coprime to 10 -/
def has_terminating_decimal (a b : ℕ) : Prop := sorry

/-- Count of integers in a given range satisfying a property -/
def count_satisfying (lower upper : ℕ) (P : ℕ → Prop) : ℕ := sorry

theorem terminating_decimal_count :
  count_satisfying 1 508 (λ k => has_terminating_decimal k 425) = 29 := by sorry

end NUMINAMATH_CALUDE_terminating_decimal_count_l3527_352784


namespace NUMINAMATH_CALUDE_range_of_a_l3527_352776

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 2*a - 4}
def B : Set ℝ := {x | x^2 - 5*x - 6 < 0}

-- State the theorem
theorem range_of_a (a : ℝ) : A a ∩ B = A a → a < 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3527_352776


namespace NUMINAMATH_CALUDE_security_to_bag_check_ratio_l3527_352754

def total_time : ℕ := 180
def uber_to_house : ℕ := 10
def check_bag_time : ℕ := 15
def wait_for_boarding : ℕ := 20

def uber_to_airport : ℕ := 5 * uber_to_house
def wait_for_takeoff : ℕ := 2 * wait_for_boarding

def known_time : ℕ := uber_to_house + uber_to_airport + check_bag_time + wait_for_boarding + wait_for_takeoff
def security_time : ℕ := total_time - known_time

theorem security_to_bag_check_ratio :
  security_time / check_bag_time = 3 ∧ security_time % check_bag_time = 0 :=
by sorry

end NUMINAMATH_CALUDE_security_to_bag_check_ratio_l3527_352754


namespace NUMINAMATH_CALUDE_solution_range_l3527_352702

theorem solution_range (m : ℝ) : 
  (∃ x : ℝ, x ≥ 0 ∧ (2 * x + m) / (x - 1) = 1) → 
  (m ≤ -1 ∧ m ≠ -2) :=
by sorry

end NUMINAMATH_CALUDE_solution_range_l3527_352702


namespace NUMINAMATH_CALUDE_fraction_sum_l3527_352706

theorem fraction_sum : (3 : ℚ) / 8 + (9 : ℚ) / 12 = (9 : ℚ) / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l3527_352706


namespace NUMINAMATH_CALUDE_min_square_area_l3527_352737

/-- A monic quartic polynomial with integer coefficients -/
structure MonicQuarticPolynomial where
  p : ℤ
  q : ℤ
  r : ℤ
  s : ℤ

/-- The roots of a polynomial form a square on the complex plane -/
def roots_form_square (poly : MonicQuarticPolynomial) : Prop :=
  sorry

/-- The area of the square formed by the roots of a polynomial -/
def square_area (poly : MonicQuarticPolynomial) : ℝ :=
  sorry

/-- The minimum possible area of a square formed by the roots of a monic quartic polynomial
    with integer coefficients is 2 -/
theorem min_square_area (poly : MonicQuarticPolynomial) 
  (h : roots_form_square poly) : 
  ∃ (min_area : ℝ), min_area = 2 ∧ ∀ (p : MonicQuarticPolynomial), 
  roots_form_square p → square_area p ≥ min_area :=
sorry

end NUMINAMATH_CALUDE_min_square_area_l3527_352737


namespace NUMINAMATH_CALUDE_inequality_system_no_solution_l3527_352703

theorem inequality_system_no_solution (a : ℝ) :
  (∀ x : ℝ, ¬(x > a + 2 ∧ x < 3*a - 2)) ↔ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_no_solution_l3527_352703


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l3527_352733

theorem quadratic_roots_sum_of_squares (p q r s : ℝ) : 
  (∀ x, x^2 - 2*p*x + 3*q = 0 ↔ x = r ∨ x = s) → 
  r^2 + s^2 = 4*p^2 - 6*q := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l3527_352733


namespace NUMINAMATH_CALUDE_sams_remaining_pennies_l3527_352769

/-- Given an initial amount of pennies and an amount spent, calculate the remaining pennies. -/
def remaining_pennies (initial : ℕ) (spent : ℕ) : ℕ := initial - spent

/-- Theorem: Sam's remaining pennies -/
theorem sams_remaining_pennies :
  let initial := 98
  let spent := 93
  remaining_pennies initial spent = 5 := by
  sorry

end NUMINAMATH_CALUDE_sams_remaining_pennies_l3527_352769


namespace NUMINAMATH_CALUDE_prob_sum_24_four_dice_l3527_352735

/-- The number of sides on a standard die -/
def standard_die_sides : ℕ := 6

/-- The target sum we're aiming for -/
def target_sum : ℕ := 24

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- Probability of rolling a specific number on a fair, standard six-sided die -/
def single_die_prob : ℚ := 1 / standard_die_sides

/-- Theorem: The probability of rolling a sum of 24 with four fair, standard six-sided dice is 1/1296 -/
theorem prob_sum_24_four_dice : 
  (single_die_prob ^ num_dice : ℚ) = 1 / 1296 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_24_four_dice_l3527_352735


namespace NUMINAMATH_CALUDE_tan_difference_equals_one_eighth_l3527_352797

theorem tan_difference_equals_one_eighth 
  (α β : ℝ) 
  (h1 : Real.tan (α - β) = 2/3) 
  (h2 : Real.tan (π/6 - β) = 1/2) : 
  Real.tan (α - π/6) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_tan_difference_equals_one_eighth_l3527_352797


namespace NUMINAMATH_CALUDE_sequence_inequality_l3527_352732

theorem sequence_inequality (n : ℕ) (a : ℝ) (seq : ℕ → ℝ) 
  (h1 : seq 1 = a)
  (h2 : seq n = a)
  (h3 : ∀ k ∈ Finset.range (n - 2), seq (k + 2) ≤ (seq (k + 1) + seq (k + 3)) / 2) :
  ∀ k ∈ Finset.range n, seq (k + 1) ≤ a := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l3527_352732


namespace NUMINAMATH_CALUDE_range_of_a_l3527_352759

/-- Given propositions p and q, where p: x^2 + 2x - 3 > 0 and q: x > a,
    and a sufficient but not necessary condition for ¬q is ¬p,
    prove that the range of values for a is a ≥ 1 -/
theorem range_of_a (x a : ℝ) : 
  (∀ x, (x^2 + 2*x - 3 > 0 → x > a) ∧ 
       (x ≤ a → x^2 + 2*x - 3 ≤ 0)) → 
  a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3527_352759


namespace NUMINAMATH_CALUDE_train_final_speed_train_final_speed_zero_l3527_352747

/-- Proves that a train with given initial speed and deceleration comes to a stop before traveling 4 km -/
theorem train_final_speed (v_i : Real) (a : Real) (d : Real) :
  v_i = 189 * (1000 / 3600) →
  a = -0.5 →
  d = 4000 →
  v_i^2 + 2 * a * d < 0 →
  ∃ (d_stop : Real), d_stop < d ∧ v_i^2 + 2 * a * d_stop = 0 :=
by sorry

/-- Proves that the final speed of the train after traveling 4 km is 0 m/s -/
theorem train_final_speed_zero (v_i : Real) (a : Real) (d : Real) (v_f : Real) :
  v_i = 189 * (1000 / 3600) →
  a = -0.5 →
  d = 4000 →
  v_f^2 = v_i^2 + 2 * a * d →
  v_f = 0 :=
by sorry

end NUMINAMATH_CALUDE_train_final_speed_train_final_speed_zero_l3527_352747


namespace NUMINAMATH_CALUDE_largest_angle_in_convex_pentagon_l3527_352720

theorem largest_angle_in_convex_pentagon (x : ℝ) : 
  (x + 2 + (2*x + 3) + (3*x - 4) + (4*x + 5) + (5*x - 6) = 540) →
  max (x + 2) (max (2*x + 3) (max (3*x - 4) (max (4*x + 5) (5*x - 6)))) = 174 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_convex_pentagon_l3527_352720


namespace NUMINAMATH_CALUDE_range_of_m_l3527_352781

theorem range_of_m (x y m : ℝ) : 
  x > 0 → 
  y > 0 → 
  2 / x + 1 / y = 1 → 
  (∀ (x y : ℝ), x > 0 → y > 0 → 2 / x + 1 / y = 1 → x^2 + 2*x*y > m^2 + 2*m) → 
  m > -4 ∧ m < 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l3527_352781


namespace NUMINAMATH_CALUDE_angle_sum_90_l3527_352788

-- Define the necessary structures
structure Plane :=
(π : Type)

structure Line :=
(l : Type)

-- Define the perpendicular relation between a line and a plane
def perpendicular (p : Line) (π : Plane) : Prop :=
sorry

-- Define the angle between a line and a plane
def angle_line_plane (l : Line) (π : Plane) : ℝ :=
sorry

-- Define the angle between two lines
def angle_between_lines (l1 l2 : Line) : ℝ :=
sorry

-- State the theorem
theorem angle_sum_90 (p : Line) (π : Plane) (l : Line) 
  (h : perpendicular p π) :
  angle_line_plane l π + angle_between_lines l p = 90 :=
sorry

end NUMINAMATH_CALUDE_angle_sum_90_l3527_352788


namespace NUMINAMATH_CALUDE_correct_object_clause_introducer_l3527_352721

-- Define a type for words that can introduce clauses
inductive ClauseIntroducer
  | That
  | What
  | Where
  | Which

-- Define a function to check if a word is the correct introducer for an object clause
def isCorrectObjectClauseIntroducer (word : ClauseIntroducer) : Prop :=
  word = ClauseIntroducer.What

-- Theorem stating that "what" is the correct word to introduce the object clause
theorem correct_object_clause_introducer :
  isCorrectObjectClauseIntroducer ClauseIntroducer.What :=
by sorry

end NUMINAMATH_CALUDE_correct_object_clause_introducer_l3527_352721


namespace NUMINAMATH_CALUDE_star_seven_three_l3527_352761

/-- Custom binary operation ∗ -/
def star (a b : ℤ) : ℤ := 4*a + 5*b - a*b

/-- Theorem stating that 7 ∗ 3 = 22 -/
theorem star_seven_three : star 7 3 = 22 := by
  sorry

end NUMINAMATH_CALUDE_star_seven_three_l3527_352761


namespace NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l3527_352726

theorem sphere_volume_equals_surface_area (r : ℝ) : 
  (4 / 3 : ℝ) * Real.pi * r^3 = 4 * Real.pi * r^2 → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l3527_352726


namespace NUMINAMATH_CALUDE_trader_profit_loss_percentage_trader_overall_loss_l3527_352793

/-- Calculates the overall profit or loss percentage for a trader selling two cars -/
theorem trader_profit_loss_percentage 
  (selling_price : ℝ) 
  (gain_percentage : ℝ) 
  (loss_percentage : ℝ) : ℝ :=
  let cost_price1 := selling_price / (1 + gain_percentage / 100)
  let cost_price2 := selling_price / (1 - loss_percentage / 100)
  let total_cost := cost_price1 + cost_price2
  let total_selling := 2 * selling_price
  let profit_loss := total_selling - total_cost
  (profit_loss / total_cost) * 100

/-- Proof that the trader's overall loss is approximately 1.44% -/
theorem trader_overall_loss :
  ∃ ε > 0, abs (trader_profit_loss_percentage 325475 12 12 + 1.44) < ε :=
sorry

end NUMINAMATH_CALUDE_trader_profit_loss_percentage_trader_overall_loss_l3527_352793


namespace NUMINAMATH_CALUDE_traffic_light_change_probability_l3527_352738

/-- Represents the duration of a traffic light cycle in seconds -/
def cycle_duration : ℕ := 95

/-- Represents the number of color changes in a cycle -/
def color_changes : ℕ := 3

/-- Represents the duration of each color change in seconds -/
def change_duration : ℕ := 5

/-- Represents the duration of the observation interval in seconds -/
def observation_interval : ℕ := 5

/-- The probability of observing a color change during a random observation interval -/
theorem traffic_light_change_probability :
  (color_changes * change_duration : ℚ) / cycle_duration = 3 / 19 := by sorry

end NUMINAMATH_CALUDE_traffic_light_change_probability_l3527_352738


namespace NUMINAMATH_CALUDE_abs_neg_four_equals_four_l3527_352710

theorem abs_neg_four_equals_four : |(-4 : ℤ)| = 4 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_four_equals_four_l3527_352710


namespace NUMINAMATH_CALUDE_complex_cube_root_l3527_352795

theorem complex_cube_root (a b : ℕ+) :
  (a + b * Complex.I) ^ 3 = 2 + 11 * Complex.I →
  a + b * Complex.I = 2 + Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_cube_root_l3527_352795


namespace NUMINAMATH_CALUDE_peter_stamps_l3527_352728

theorem peter_stamps (M : ℕ) : 
  M > 1 ∧ 
  M % 5 = 2 ∧ 
  M % 11 = 2 ∧ 
  M % 13 = 2 → 
  (∀ n : ℕ, n > 1 ∧ n % 5 = 2 ∧ n % 11 = 2 ∧ n % 13 = 2 → n ≥ M) → 
  M = 717 := by
sorry

end NUMINAMATH_CALUDE_peter_stamps_l3527_352728


namespace NUMINAMATH_CALUDE_spinner_points_south_l3527_352724

/-- Represents the four cardinal directions --/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents a rotation of the spinner --/
structure Rotation :=
  (revolutions : ℚ)
  (clockwise : Bool)

/-- Calculates the final direction after applying a net rotation --/
def finalDirection (netRotation : ℚ) : Direction :=
  match netRotation.num % 4 with
  | 0 => Direction.North
  | 1 => Direction.East
  | 2 => Direction.South
  | _ => Direction.West

/-- Theorem stating that the given sequence of rotations results in the spinner pointing south --/
theorem spinner_points_south (initialDirection : Direction)
    (rotation1 : Rotation)
    (rotation2 : Rotation)
    (rotation3 : Rotation) :
    initialDirection = Direction.North ∧
    rotation1 = { revolutions := 7/2, clockwise := true } ∧
    rotation2 = { revolutions := 16/3, clockwise := false } ∧
    rotation3 = { revolutions := 13/6, clockwise := true } →
    finalDirection (
      rotation1.revolutions * (if rotation1.clockwise then 1 else -1) +
      rotation2.revolutions * (if rotation2.clockwise then 1 else -1) +
      rotation3.revolutions * (if rotation3.clockwise then 1 else -1)
    ) = Direction.South :=
by sorry

end NUMINAMATH_CALUDE_spinner_points_south_l3527_352724


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l3527_352798

theorem z_in_first_quadrant : 
  ∃ (z : ℂ), (Complex.I + 1) * z = Complex.I^2013 ∧ 
  z.re > 0 ∧ z.im > 0 :=
by sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l3527_352798


namespace NUMINAMATH_CALUDE_final_F_position_l3527_352791

-- Define the letter F as a type with base and stem directions
inductive LetterF
  | mk (base : ℝ × ℝ) (stem : ℝ × ℝ)

-- Define the initial position of F
def initial_F : LetterF := LetterF.mk (-1, 0) (0, -1)

-- Define the transformations
def rotate_180 (f : LetterF) : LetterF :=
  match f with
  | LetterF.mk (x, y) (a, b) => LetterF.mk (-x, -y) (-a, -b)

def reflect_y_axis (f : LetterF) : LetterF :=
  match f with
  | LetterF.mk (x, y) (a, b) => LetterF.mk (-x, y) (-a, b)

def rotate_90 (f : LetterF) : LetterF :=
  match f with
  | LetterF.mk (x, y) (a, b) => LetterF.mk (y, -x) (b, -a)

-- Define the final transformation as a composition of the three transformations
def final_transformation (f : LetterF) : LetterF :=
  rotate_90 (reflect_y_axis (rotate_180 f))

-- Theorem: The final position of F after transformations
theorem final_F_position :
  final_transformation initial_F = LetterF.mk (0, -1) (-1, 0) :=
by sorry

end NUMINAMATH_CALUDE_final_F_position_l3527_352791
