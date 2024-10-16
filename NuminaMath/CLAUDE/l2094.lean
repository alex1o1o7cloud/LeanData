import Mathlib

namespace NUMINAMATH_CALUDE_ellipse_dot_product_range_l2094_209432

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the left focus and left vertex
def F₁ : ℝ × ℝ := (-1, 0)
def A : ℝ × ℝ := (-2, 0)

-- Define the dot product of PF₁ and PA
def dot_product (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  (x + 1) * (x + 2) + y^2

-- Theorem statement
theorem ellipse_dot_product_range :
  ∀ P : ℝ × ℝ, ellipse P.1 P.2 → 0 ≤ dot_product P ∧ dot_product P ≤ 12 :=
sorry

end NUMINAMATH_CALUDE_ellipse_dot_product_range_l2094_209432


namespace NUMINAMATH_CALUDE_budget_supplies_percentage_l2094_209443

theorem budget_supplies_percentage (transportation research_development utilities equipment salaries supplies : ℝ)
  (h1 : transportation = 15)
  (h2 : research_development = 9)
  (h3 : utilities = 5)
  (h4 : equipment = 4)
  (h5 : salaries = 234 / 360 * 100)
  (h6 : transportation + research_development + utilities + equipment + salaries + supplies = 100) :
  supplies = 2 := by
  sorry

end NUMINAMATH_CALUDE_budget_supplies_percentage_l2094_209443


namespace NUMINAMATH_CALUDE_flower_bee_butterfly_difference_l2094_209417

theorem flower_bee_butterfly_difference (flowers bees butterflies : ℕ) 
  (h1 : flowers = 12) 
  (h2 : bees = 7) 
  (h3 : butterflies = 4) : 
  (flowers - bees) - butterflies = 1 := by
  sorry

end NUMINAMATH_CALUDE_flower_bee_butterfly_difference_l2094_209417


namespace NUMINAMATH_CALUDE_melted_ice_cream_depth_l2094_209458

/-- Given a sphere of radius 3 inches and a cylinder of radius 12 inches with the same volume,
    prove that the height of the cylinder is 1/4 inch. -/
theorem melted_ice_cream_depth (sphere_radius : ℝ) (cylinder_radius : ℝ) (cylinder_height : ℝ) :
  sphere_radius = 3 →
  cylinder_radius = 12 →
  (4 / 3) * Real.pi * sphere_radius ^ 3 = Real.pi * cylinder_radius ^ 2 * cylinder_height →
  cylinder_height = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_melted_ice_cream_depth_l2094_209458


namespace NUMINAMATH_CALUDE_consecutive_sum_33_l2094_209420

theorem consecutive_sum_33 (m : ℕ) (h1 : m > 1) :
  (∃ a : ℕ, (Finset.range m).sum (λ i => a + i) = 33) ↔ m = 2 ∨ m = 3 ∨ m = 6 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_sum_33_l2094_209420


namespace NUMINAMATH_CALUDE_eulers_conjecture_counterexample_l2094_209475

theorem eulers_conjecture_counterexample : 133^5 + 110^5 + 84^5 + 27^5 = 144^5 := by
  sorry

end NUMINAMATH_CALUDE_eulers_conjecture_counterexample_l2094_209475


namespace NUMINAMATH_CALUDE_blank_value_l2094_209454

theorem blank_value : ∃ x : ℝ, x * (-2) = 1 ∧ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_blank_value_l2094_209454


namespace NUMINAMATH_CALUDE_no_lines_satisfy_conditions_l2094_209433

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def line_through_point (a b : ℝ) : Prop :=
  6 / a + 5 / b = 1

def satisfies_conditions (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ is_prime a ∧ a + b < 20 ∧ line_through_point a b

theorem no_lines_satisfy_conditions :
  ¬ ∃ a b : ℕ, satisfies_conditions a b :=
sorry

end NUMINAMATH_CALUDE_no_lines_satisfy_conditions_l2094_209433


namespace NUMINAMATH_CALUDE_probability_diamond_is_one_fourth_l2094_209456

/-- A special deck of cards -/
structure SpecialDeck :=
  (total_cards : ℕ)
  (num_ranks : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)
  (h1 : total_cards = num_ranks * num_suits)
  (h2 : cards_per_suit = num_ranks)

/-- The probability of drawing a diamond from the special deck -/
def probability_diamond (deck : SpecialDeck) : ℚ :=
  deck.cards_per_suit / deck.total_cards

/-- Theorem stating that the probability of drawing a diamond is 1/4 -/
theorem probability_diamond_is_one_fourth (deck : SpecialDeck) 
  (h3 : deck.num_suits = 4) : 
  probability_diamond deck = 1/4 := by
  sorry

#check probability_diamond_is_one_fourth

end NUMINAMATH_CALUDE_probability_diamond_is_one_fourth_l2094_209456


namespace NUMINAMATH_CALUDE_two_triangles_exist_l2094_209450

/-- Given a side length, ratio of other sides, and circumradius, prove existence of two triangles -/
theorem two_triangles_exist (a : ℝ) (k : ℝ) (R : ℝ) 
    (h_a : a > 0) (h_k : k > 0) (h_R : R > 0) (h_aR : a < 2*R) : 
  ∃ (b₁ c₁ b₂ c₂ : ℝ), 
    (b₁ > 0 ∧ c₁ > 0 ∧ b₂ > 0 ∧ c₂ > 0) ∧ 
    (b₁/c₁ = k ∧ b₂/c₂ = k) ∧
    (a + b₁ > c₁ ∧ b₁ + c₁ > a ∧ c₁ + a > b₁) ∧
    (a + b₂ > c₂ ∧ b₂ + c₂ > a ∧ c₂ + a > b₂) ∧
    (b₁ ≠ b₂ ∨ c₁ ≠ c₂) ∧
    (4 * R * R * (a + b₁ + c₁) = (a * b₁ * c₁) / R) ∧
    (4 * R * R * (a + b₂ + c₂) = (a * b₂ * c₂) / R) :=
by sorry

end NUMINAMATH_CALUDE_two_triangles_exist_l2094_209450


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l2094_209451

def M : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def N : Set ℝ := {x | 2*x + 1 < 5}

theorem union_of_M_and_N : M ∪ N = {x | x < 3} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l2094_209451


namespace NUMINAMATH_CALUDE_retail_price_increase_l2094_209410

theorem retail_price_increase (wholesale_cost employee_paid : ℝ) (employee_discount : ℝ) :
  wholesale_cost = 200 →
  employee_discount = 0.15 →
  employee_paid = 204 →
  ∃ (retail_price_increase : ℝ),
    retail_price_increase = 0.20 ∧
    employee_paid = wholesale_cost * (1 + retail_price_increase) * (1 - employee_discount) :=
by
  sorry

end NUMINAMATH_CALUDE_retail_price_increase_l2094_209410


namespace NUMINAMATH_CALUDE_seating_arrangements_with_restrictions_l2094_209472

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def total_arrangements (n : ℕ) : ℕ := factorial n

def pair_together (n : ℕ) : ℕ := factorial (n - 1) * factorial 2

def both_pairs_together (n : ℕ) : ℕ := factorial (n - 2) * factorial 2 * factorial 2

theorem seating_arrangements_with_restrictions :
  let n : ℕ := 8
  let total := total_arrangements n
  let alice_bob := pair_together n
  let cindy_dave := pair_together n
  let both_pairs := both_pairs_together n
  total - (alice_bob + cindy_dave - both_pairs) = 23040 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_with_restrictions_l2094_209472


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l2094_209416

theorem ceiling_floor_sum : ⌈(7 : ℚ) / 3⌉ + ⌊-(7 : ℚ) / 3⌋ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l2094_209416


namespace NUMINAMATH_CALUDE_natural_number_representation_with_distinct_powers_l2094_209497

theorem natural_number_representation_with_distinct_powers : ∃ (N : ℕ) 
  (a₁ a₂ b₁ b₂ c₁ c₂ d₁ d₂ : ℕ), 
  (∃ (x₁ x₂ : ℕ), a₁ = x₁^2 ∧ a₂ = x₂^2) ∧
  (∃ (y₁ y₂ : ℕ), b₁ = y₁^3 ∧ b₂ = y₂^3) ∧
  (∃ (z₁ z₂ : ℕ), c₁ = z₁^5 ∧ c₂ = z₂^5) ∧
  (∃ (w₁ w₂ : ℕ), d₁ = w₁^7 ∧ d₂ = w₂^7) ∧
  N = a₁ - a₂ ∧ N = b₁ - b₂ ∧ N = c₁ - c₂ ∧ N = d₁ - d₂ ∧
  a₁ ≠ b₁ ∧ a₁ ≠ c₁ ∧ a₁ ≠ d₁ ∧ b₁ ≠ c₁ ∧ b₁ ≠ d₁ ∧ c₁ ≠ d₁ :=
by
  sorry


end NUMINAMATH_CALUDE_natural_number_representation_with_distinct_powers_l2094_209497


namespace NUMINAMATH_CALUDE_custom_mult_example_l2094_209457

/-- Custom multiplication operation for fractions -/
def custom_mult (m n p q : ℚ) : ℚ := m * p * (n / q)

/-- Theorem stating that 5/4 * 6/2 = 60 under the custom multiplication -/
theorem custom_mult_example : custom_mult 5 4 6 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_example_l2094_209457


namespace NUMINAMATH_CALUDE_compute_expression_l2094_209473

theorem compute_expression : 9 + 7 * (5 - Real.sqrt 16)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l2094_209473


namespace NUMINAMATH_CALUDE_min_value_f_l2094_209455

/-- Given positive real numbers x₁ and x₂, and a function f satisfying certain conditions,
    the value of f(x₁ + x₂) has a lower bound of 4/5. -/
theorem min_value_f (x₁ x₂ : ℝ) (f : ℝ → ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0)
  (hf : ∀ x, 4^x = (1 + f x) / (1 - f x))
  (hsum : f x₁ + f x₂ = 1) :
  f (x₁ + x₂) ≥ 4/5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_f_l2094_209455


namespace NUMINAMATH_CALUDE_quadratic_roots_expression_l2094_209411

theorem quadratic_roots_expression (x₁ x₂ : ℝ) : 
  (-2 * x₁^2 + x₁ + 5 = 0) → 
  (-2 * x₂^2 + x₂ + 5 = 0) → 
  x₁^2 * x₂ + x₁ * x₂^2 = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_expression_l2094_209411


namespace NUMINAMATH_CALUDE_infinitely_many_H_points_l2094_209481

/-- The curve C defined by x/4 + y^2 = 1 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 / 4 + p.2^2 = 1}

/-- The line l defined by x = 4 -/
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 4}

/-- A point P is an H point if there exists a line through P intersecting C at A 
    and l at B, with either |PA| = |PB| or |PA| = |AB| -/
def is_H_point (P : ℝ × ℝ) : Prop :=
  P ∈ C ∧ ∃ (A B : ℝ × ℝ), A ∈ C ∧ B ∈ l ∧ A ≠ P ∧
    (∃ (k m : ℝ), ∀ x y, y = k * x + m → 
      ((x, y) = P ∨ (x, y) = A ∨ (x, y) = B)) ∧
    (dist P A = dist P B ∨ dist P A = dist A B)

/-- There are infinitely many H points on C, but not all points on C are H points -/
theorem infinitely_many_H_points : 
  (∃ (S : Set (ℝ × ℝ)), S ⊆ C ∧ Infinite S ∧ ∀ p ∈ S, is_H_point p) ∧
  (∃ p ∈ C, ¬is_H_point p) := by sorry


end NUMINAMATH_CALUDE_infinitely_many_H_points_l2094_209481


namespace NUMINAMATH_CALUDE_lcm_problem_l2094_209460

theorem lcm_problem (a b : ℕ+) (h : Nat.gcd a b = 9) (p : a * b = 1800) :
  Nat.lcm a b = 200 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l2094_209460


namespace NUMINAMATH_CALUDE_probability_six_heads_twelve_flips_l2094_209422

/-- The probability of getting exactly 6 heads when flipping a fair coin 12 times -/
theorem probability_six_heads_twelve_flips : 
  (Nat.choose 12 6 : ℚ) / 2^12 = 231 / 1024 := by sorry

end NUMINAMATH_CALUDE_probability_six_heads_twelve_flips_l2094_209422


namespace NUMINAMATH_CALUDE_x_between_one_third_and_two_thirds_l2094_209429

theorem x_between_one_third_and_two_thirds (x : ℝ) :
  (1 < 4 * x ∧ 4 * x < 3) ∧ (2 < 6 * x ∧ 6 * x < 4) → (1/3 < x ∧ x < 2/3) := by
  sorry

end NUMINAMATH_CALUDE_x_between_one_third_and_two_thirds_l2094_209429


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l2094_209448

-- Problem 1
theorem problem_one : (-1/3)⁻¹ + Real.sqrt 9 + (2 - Real.pi)^0 = 1 := by sorry

-- Problem 2
theorem problem_two (a : ℝ) (ha : a ≠ 0) (ha' : a ≠ 1) : 
  ((1/a - 1) / ((a^2 - 2*a + 1) / a)) = 1 / (1 - a) := by sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l2094_209448


namespace NUMINAMATH_CALUDE_interest_credited_proof_l2094_209421

/-- The interest rate per annum as a decimal -/
def interest_rate : ℚ := 5 / 100

/-- The time period in years -/
def time_period : ℚ := 2 / 12

/-- The total amount after interest -/
def total_amount : ℚ := 255.31

/-- The simple interest formula -/
def simple_interest (principal : ℚ) : ℚ :=
  principal * (1 + interest_rate * time_period)

theorem interest_credited_proof :
  ∃ (principal : ℚ),
    simple_interest principal = total_amount ∧
    (total_amount - principal) * 100 = 210 := by
  sorry

end NUMINAMATH_CALUDE_interest_credited_proof_l2094_209421


namespace NUMINAMATH_CALUDE_negation_of_implication_l2094_209486

theorem negation_of_implication (x y : ℝ) :
  ¬(x + y = 1 → x * y ≤ 1) ↔ (x + y = 1 → x * y > 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2094_209486


namespace NUMINAMATH_CALUDE_car_profit_percentage_l2094_209440

theorem car_profit_percentage (P : ℝ) (P_positive : P > 0) : 
  let discount_rate : ℝ := 0.20
  let increase_rate : ℝ := 0.70
  let buying_price : ℝ := P * (1 - discount_rate)
  let selling_price : ℝ := buying_price * (1 + increase_rate)
  let profit : ℝ := selling_price - P
  let profit_percentage : ℝ := (profit / P) * 100
  profit_percentage = 36 := by sorry

end NUMINAMATH_CALUDE_car_profit_percentage_l2094_209440


namespace NUMINAMATH_CALUDE_geometric_solid_surface_area_l2094_209403

/-- Given a geometric solid that is a quarter of a cylinder with height 2,
    base area π, and radius 2, prove that its surface area is 8 + 4π. -/
theorem geometric_solid_surface_area
  (h : ℝ) (base_area : ℝ) (radius : ℝ) :
  h = 2 →
  base_area = π →
  radius = 2 →
  (2 * base_area + 2 * radius * h + (1/4) * 2 * π * radius * h) = 8 + 4 * π :=
by sorry

end NUMINAMATH_CALUDE_geometric_solid_surface_area_l2094_209403


namespace NUMINAMATH_CALUDE_sum_of_squares_l2094_209449

theorem sum_of_squares (a b c : ℝ) 
  (h_arithmetic : (a + b + c) / 3 = 10)
  (h_geometric : (a * b * c) ^ (1/3 : ℝ) = 6)
  (h_harmonic : 3 / (1/a + 1/b + 1/c) = 4) :
  a^2 + b^2 + c^2 = 576 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2094_209449


namespace NUMINAMATH_CALUDE_max_value_of_f_l2094_209445

def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 4*x + a

theorem max_value_of_f (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f a x ≥ -2) ∧
  (∃ x ∈ Set.Icc 0 1, f a x = -2) →
  (∃ x ∈ Set.Icc 0 1, f a x = 1) ∧
  (∀ x ∈ Set.Icc 0 1, f a x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2094_209445


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2094_209478

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℚ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 2 + a 8 = 12) :
  a 5 = 6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2094_209478


namespace NUMINAMATH_CALUDE_g_expression_l2094_209459

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

-- Define the property of g being a linear function
def is_linear (g : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ x, g x = a * x + b

-- State the theorem
theorem g_expression (g : ℝ → ℝ) (h_linear : is_linear g) 
    (h_comp : ∀ x, f (g x) = 4 * x^2) :
  (∀ x, g x = 2 * x + 1) ∨ (∀ x, g x = -2 * x + 1) := by sorry

end NUMINAMATH_CALUDE_g_expression_l2094_209459


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2094_209444

theorem max_value_of_expression (x y z : ℝ) (h : x + y + 2*z = 5) :
  ∃ (max : ℝ), max = 25/6 ∧ ∀ (a b c : ℝ), a + b + 2*c = 5 → a*b + a*c + b*c ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2094_209444


namespace NUMINAMATH_CALUDE_tan_five_pi_four_equals_one_l2094_209489

theorem tan_five_pi_four_equals_one : Real.tan (5 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_five_pi_four_equals_one_l2094_209489


namespace NUMINAMATH_CALUDE_octal_1072_equals_base5_4240_l2094_209470

def octal_to_decimal (n : List Nat) : Nat :=
  n.enum.foldr (λ (i, d) acc => acc + d * (8 ^ i)) 0

def decimal_to_base5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem octal_1072_equals_base5_4240 :
  decimal_to_base5 (octal_to_decimal [2, 7, 0, 1]) = [4, 2, 4, 0] := by
  sorry

end NUMINAMATH_CALUDE_octal_1072_equals_base5_4240_l2094_209470


namespace NUMINAMATH_CALUDE_min_subset_size_for_acute_triangle_l2094_209480

def is_acute_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b ∧
  a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

theorem min_subset_size_for_acute_triangle :
  ∃ (k : ℕ), k = 29 ∧
  (∀ (S : Finset ℕ), S ⊆ Finset.range 2004 → S.card ≥ k →
    ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ is_acute_triangle a b c) ∧
  (∀ (k' : ℕ), k' < k →
    ∃ (S : Finset ℕ), S ⊆ Finset.range 2004 ∧ S.card = k' ∧
      ∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → c ≠ a → ¬is_acute_triangle a b c) :=
by sorry

end NUMINAMATH_CALUDE_min_subset_size_for_acute_triangle_l2094_209480


namespace NUMINAMATH_CALUDE_min_half_tiles_for_29_l2094_209474

/-- Represents a tiling of a square area -/
structure Tiling where
  size : ℕ  -- The size of the square area in unit squares
  whole_tiles : ℕ  -- Number of whole tiles used
  half_tiles : ℕ  -- Number of tiles cut in half

/-- Checks if a tiling is valid for the given area -/
def is_valid_tiling (t : Tiling) : Prop :=
  t.whole_tiles + t.half_tiles / 2 = t.size

/-- Theorem: The minimum number of tiles to be cut in half for a 29-unit square area is 12 -/
theorem min_half_tiles_for_29 :
  ∀ t : Tiling, t.size = 29 → is_valid_tiling t →
  t.half_tiles ≥ 12 ∧ ∃ t' : Tiling, t'.size = 29 ∧ is_valid_tiling t' ∧ t'.half_tiles = 12 :=
by sorry

#check min_half_tiles_for_29

end NUMINAMATH_CALUDE_min_half_tiles_for_29_l2094_209474


namespace NUMINAMATH_CALUDE_smallest_distance_to_i_l2094_209418

theorem smallest_distance_to_i (w : ℂ) (h : Complex.abs (w^2 - 3) = Complex.abs (w * (2*w + 3*Complex.I))) :
  ∃ (min_dist : ℝ), 
    (∀ w', Complex.abs (w'^2 - 3) = Complex.abs (w' * (2*w' + 3*Complex.I)) → 
      Complex.abs (w' - Complex.I) ≥ min_dist) ∧
    min_dist = Complex.abs ((Real.sqrt 3 - Real.sqrt 6) / 3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_distance_to_i_l2094_209418


namespace NUMINAMATH_CALUDE_parallelogram_area_l2094_209484

/-- The area of a parallelogram with one angle of 100 degrees and two consecutive sides of lengths 10 and 20 is approximately 196.96 -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h₁ : a = 10) (h₂ : b = 20) (h₃ : θ = 100 * π / 180) :
  abs (a * b * Real.sin θ - 196.96) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2094_209484


namespace NUMINAMATH_CALUDE_point_3_0_on_line_point_0_4_on_line_is_line_equation_main_theorem_l2094_209425

/-- The line passing through points (3, 0) and (0, 4) -/
def line_equation (x y : ℝ) : Prop := 4*x + 3*y - 12 = 0

/-- Point (3, 0) lies on the line -/
theorem point_3_0_on_line : line_equation 3 0 := by sorry

/-- Point (0, 4) lies on the line -/
theorem point_0_4_on_line : line_equation 0 4 := by sorry

/-- The equation represents a line -/
theorem is_line_equation : ∃ (m b : ℝ), ∀ (x y : ℝ), line_equation x y ↔ y = m*x + b := by sorry

/-- Main theorem: The given equation represents the unique line passing through (3, 0) and (0, 4) -/
theorem main_theorem : 
  ∀ (f : ℝ → ℝ → Prop), 
  (f 3 0 ∧ f 0 4 ∧ (∃ (m b : ℝ), ∀ (x y : ℝ), f x y ↔ y = m*x + b)) → 
  (∀ (x y : ℝ), f x y ↔ line_equation x y) := by sorry

end NUMINAMATH_CALUDE_point_3_0_on_line_point_0_4_on_line_is_line_equation_main_theorem_l2094_209425


namespace NUMINAMATH_CALUDE_minimum_commission_rate_l2094_209469

/-- The minimum commission rate problem -/
theorem minimum_commission_rate 
  (old_salary : ℝ) 
  (new_base_salary : ℝ) 
  (sale_value : ℝ) 
  (min_sales : ℝ) 
  (h1 : old_salary = 75000)
  (h2 : new_base_salary = 45000)
  (h3 : sale_value = 750)
  (h4 : min_sales = 266.67)
  : ∃ (commission_rate : ℝ), 
    commission_rate ≥ (old_salary - new_base_salary) / min_sales ∧ 
    commission_rate ≥ 112.50 :=
sorry

end NUMINAMATH_CALUDE_minimum_commission_rate_l2094_209469


namespace NUMINAMATH_CALUDE_system_solution_l2094_209430

theorem system_solution (x y z : ℚ) : 
  (x * y = 6 * (x + y) ∧ 
   x * z = 4 * (x + z) ∧ 
   y * z = 2 * (y + z)) ↔ 
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ 
   (x = -24 ∧ y = 24/5 ∧ z = 24/7)) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2094_209430


namespace NUMINAMATH_CALUDE_remainder_theorem_l2094_209439

theorem remainder_theorem (n : ℤ) (h : n % 28 = 15) : (2 * n) % 14 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2094_209439


namespace NUMINAMATH_CALUDE_intersection_when_m_3_range_of_m_when_B_subset_A_l2094_209404

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem intersection_when_m_3 : A ∩ B 3 = {x | 4 ≤ x ∧ x ≤ 5} := by sorry

theorem range_of_m_when_B_subset_A : 
  ∀ m : ℝ, B m ⊆ A → m ≤ 3 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_3_range_of_m_when_B_subset_A_l2094_209404


namespace NUMINAMATH_CALUDE_gcd_75_100_l2094_209452

theorem gcd_75_100 : Nat.gcd 75 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcd_75_100_l2094_209452


namespace NUMINAMATH_CALUDE_f_composition_of_three_l2094_209413

def f (x : ℝ) : ℝ := 3 * x + 2

theorem f_composition_of_three : f (f (f 3)) = 107 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_of_three_l2094_209413


namespace NUMINAMATH_CALUDE_min_value_theorem_l2094_209483

theorem min_value_theorem (a : ℝ) (h : a > 2) :
  a + 1 / (a - 2) ≥ 4 ∧ (a + 1 / (a - 2) = 4 ↔ a = 3) := by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2094_209483


namespace NUMINAMATH_CALUDE_min_red_chips_l2094_209490

theorem min_red_chips (w b r : ℕ) : 
  b ≥ w / 3 →
  b ≤ r / 4 →
  w + b ≥ 70 →
  r ≥ 72 ∧ ∀ r' : ℕ, (∃ w' b' : ℕ, b' ≥ w' / 3 ∧ b' ≤ r' / 4 ∧ w' + b' ≥ 70) → r' ≥ 72 :=
by sorry

end NUMINAMATH_CALUDE_min_red_chips_l2094_209490


namespace NUMINAMATH_CALUDE_second_shift_participation_theorem_l2094_209466

/-- The percentage of second shift employees participating in the pension program -/
def second_shift_participation_rate : ℝ := 40

theorem second_shift_participation_theorem :
  let total_employees : ℕ := 60 + 50 + 40
  let first_shift : ℕ := 60
  let second_shift : ℕ := 50
  let third_shift : ℕ := 40
  let first_shift_rate : ℝ := 20
  let third_shift_rate : ℝ := 10
  let total_participation_rate : ℝ := 24
  let first_shift_participants : ℝ := first_shift_rate / 100 * first_shift
  let third_shift_participants : ℝ := third_shift_rate / 100 * third_shift
  let total_participants : ℝ := total_participation_rate / 100 * total_employees
  let second_shift_participants : ℝ := total_participants - first_shift_participants - third_shift_participants
  second_shift_participation_rate = second_shift_participants / second_shift * 100 :=
by
  sorry

end NUMINAMATH_CALUDE_second_shift_participation_theorem_l2094_209466


namespace NUMINAMATH_CALUDE_longest_side_of_triangle_l2094_209476

theorem longest_side_of_triangle (x : ℝ) : 
  9 + (x + 5) + (2*x + 3) = 40 →
  max 9 (max (x + 5) (2*x + 3)) = 55/3 :=
by sorry

end NUMINAMATH_CALUDE_longest_side_of_triangle_l2094_209476


namespace NUMINAMATH_CALUDE_charlie_flutes_l2094_209487

theorem charlie_flutes (charlie_flutes : ℕ) (charlie_horns : ℕ) (charlie_harps : ℕ) 
  (carli_flutes : ℕ) (carli_horns : ℕ) (carli_harps : ℕ) : 
  charlie_horns = 2 →
  charlie_harps = 1 →
  carli_flutes = 2 * charlie_flutes →
  carli_horns = charlie_horns / 2 →
  carli_harps = 0 →
  charlie_flutes + charlie_horns + charlie_harps + carli_flutes + carli_horns + carli_harps = 7 →
  charlie_flutes = 1 := by
sorry

end NUMINAMATH_CALUDE_charlie_flutes_l2094_209487


namespace NUMINAMATH_CALUDE_natalie_bushes_needed_l2094_209424

/-- Represents the number of containers of blueberries per bush -/
def containers_per_bush : ℕ := 10

/-- Represents the number of containers needed to trade for one cabbage -/
def containers_per_cabbage : ℕ := 4

/-- Represents the number of cabbages Natalie wants to obtain -/
def target_cabbages : ℕ := 20

/-- Calculates the number of bushes needed to obtain a given number of cabbages -/
def bushes_needed (cabbages : ℕ) : ℕ :=
  (cabbages * containers_per_cabbage) / containers_per_bush

theorem natalie_bushes_needed : bushes_needed target_cabbages = 8 := by
  sorry

end NUMINAMATH_CALUDE_natalie_bushes_needed_l2094_209424


namespace NUMINAMATH_CALUDE_max_matches_theorem_l2094_209493

/-- The maximum number of matches that cannot form a triangle with any two sides differing by at least 10 matches -/
def max_matches : ℕ := 62

/-- A function that checks if three numbers can form a triangle -/
def is_triangle (a b c : ℕ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that checks if any two sides differ by at least 10 -/
def sides_differ_by_10 (a b c : ℕ) : Prop :=
  (a ≥ b + 10 ∨ b ≥ a + 10) ∧ (b ≥ c + 10 ∨ c ≥ b + 10) ∧ (c ≥ a + 10 ∨ a ≥ c + 10)

theorem max_matches_theorem :
  ∀ n : ℕ, n > max_matches →
    ∃ a b c : ℕ, a + b + c = n ∧ is_triangle a b c ∧ sides_differ_by_10 a b c :=
sorry

end NUMINAMATH_CALUDE_max_matches_theorem_l2094_209493


namespace NUMINAMATH_CALUDE_min_students_all_activities_l2094_209477

theorem min_students_all_activities 
  (total : Nat) 
  (swim : Nat) 
  (cycle : Nat) 
  (tennis : Nat) 
  (h1 : total = 52) 
  (h2 : swim = 30) 
  (h3 : cycle = 35) 
  (h4 : tennis = 42) :
  total - ((total - swim) + (total - cycle) + (total - tennis)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_students_all_activities_l2094_209477


namespace NUMINAMATH_CALUDE_b_72_mod_50_l2094_209426

/-- The sequence b_n defined as 7^n + 9^n -/
def b (n : ℕ) : ℕ := 7^n + 9^n

/-- The theorem stating that b_72 is congruent to 2 modulo 50 -/
theorem b_72_mod_50 : b 72 ≡ 2 [ZMOD 50] := by
  sorry

end NUMINAMATH_CALUDE_b_72_mod_50_l2094_209426


namespace NUMINAMATH_CALUDE_cd_purchase_remaining_money_l2094_209496

theorem cd_purchase_remaining_money 
  (total_money : ℚ) 
  (total_cds : ℚ) 
  (cd_price : ℚ) 
  (h1 : cd_price > 0) 
  (h2 : total_money > 0) 
  (h3 : total_cds > 0) 
  (h4 : total_money / 5 = cd_price * total_cds / 3) :
  total_money - cd_price * total_cds = 2 * total_money / 5 := by
sorry

end NUMINAMATH_CALUDE_cd_purchase_remaining_money_l2094_209496


namespace NUMINAMATH_CALUDE_vertex_x_is_three_l2094_209419

/-- A quadratic function passing through specific points -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  point1 : (2 : ℝ)^2 * a + 2 * b + c = 8
  point2 : (4 : ℝ)^2 * a + 4 * b + c = 8
  point3 : c = 3

/-- The x-coordinate of the vertex of the quadratic function -/
def vertex_x (f : QuadraticFunction) : ℝ := sorry

/-- Theorem stating that the x-coordinate of the vertex is 3 -/
theorem vertex_x_is_three (f : QuadraticFunction) : vertex_x f = 3 := by sorry

end NUMINAMATH_CALUDE_vertex_x_is_three_l2094_209419


namespace NUMINAMATH_CALUDE_cameron_work_time_l2094_209479

theorem cameron_work_time (cameron_alone : ℝ) 
  (h1 : cameron_alone > 0)
  (h2 : 9 / cameron_alone + 1 / 2 = 1)
  (h3 : (1 / cameron_alone + 1 / 7) * 7 = 1) : 
  cameron_alone = 18 := by
sorry

end NUMINAMATH_CALUDE_cameron_work_time_l2094_209479


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2094_209499

theorem quadratic_equation_roots (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  (a * x^2 + b * x + c = 0) → (
    (|r₁ - r₂| = 1 ∧ max r₁ r₂ = 4) ↔ (a = 1 ∧ b = -7 ∧ c = 12)
  ) := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2094_209499


namespace NUMINAMATH_CALUDE_simplify_sqrt_difference_l2094_209461

theorem simplify_sqrt_difference : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |((Real.sqrt 528 / Real.sqrt 32) - (Real.sqrt 297 / Real.sqrt 99)) - 2.318| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_difference_l2094_209461


namespace NUMINAMATH_CALUDE_rockville_baseball_league_members_l2094_209498

/-- The cost of a pair of cleats in dollars -/
def cleatCost : ℕ := 6

/-- The additional cost of a jersey compared to cleats in dollars -/
def jerseyAdditionalCost : ℕ := 7

/-- The total cost for all members in dollars -/
def totalCost : ℕ := 3360

/-- The number of sets (cleats and jersey) each member needs -/
def setsPerMember : ℕ := 2

/-- The cost of one set (cleats and jersey) for a member -/
def setCost : ℕ := cleatCost + (cleatCost + jerseyAdditionalCost)

/-- The total cost for one member -/
def memberCost : ℕ := setsPerMember * setCost

/-- The number of members in the Rockville Baseball League -/
def numberOfMembers : ℕ := totalCost / memberCost

theorem rockville_baseball_league_members :
  numberOfMembers = 88 := by sorry

end NUMINAMATH_CALUDE_rockville_baseball_league_members_l2094_209498


namespace NUMINAMATH_CALUDE_sixth_number_is_eight_l2094_209465

/-- A structure representing an increasing list of consecutive integers -/
structure ConsecutiveIntegerList where
  start : ℤ
  length : ℕ
  increasing : 0 < length

/-- The nth number in the list -/
def ConsecutiveIntegerList.nthNumber (list : ConsecutiveIntegerList) (n : ℕ) : ℤ :=
  list.start + n - 1

/-- The property that the sum of the 3rd and 4th numbers is 11 -/
def sumProperty (list : ConsecutiveIntegerList) : Prop :=
  list.nthNumber 3 + list.nthNumber 4 = 11

theorem sixth_number_is_eight (list : ConsecutiveIntegerList) 
    (h : sumProperty list) : list.nthNumber 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sixth_number_is_eight_l2094_209465


namespace NUMINAMATH_CALUDE_annie_total_spent_l2094_209447

/-- The total amount Annie spent on televisions and figurines -/
def total_spent (num_tvs : ℕ) (price_tv : ℕ) (num_figurines : ℕ) (price_figurine : ℕ) : ℕ :=
  num_tvs * price_tv + num_figurines * price_figurine

/-- Theorem: Annie spent $260 in total -/
theorem annie_total_spent :
  total_spent 5 50 10 1 = 260 := by
  sorry

end NUMINAMATH_CALUDE_annie_total_spent_l2094_209447


namespace NUMINAMATH_CALUDE_exam_average_theorem_l2094_209446

/-- Calculates the overall average marks for a group of students given the total number of students,
    the number of passed students, the average marks of passed and failed students. -/
def overall_average (total_students : ℕ) (passed_students : ℕ) (passed_avg : ℚ) (failed_avg : ℚ) : ℚ :=
  let failed_students := total_students - passed_students
  let total_marks := passed_avg * passed_students + failed_avg * failed_students
  total_marks / total_students

/-- Theorem stating that given the conditions from the problem, 
    the overall average of marks obtained by all boys is 37. -/
theorem exam_average_theorem :
  let total_students : ℕ := 120
  let passed_students : ℕ := 110
  let passed_avg : ℚ := 39
  let failed_avg : ℚ := 15
  overall_average total_students passed_students passed_avg failed_avg = 37 := by
sorry


end NUMINAMATH_CALUDE_exam_average_theorem_l2094_209446


namespace NUMINAMATH_CALUDE_bug_meeting_point_l2094_209428

/-- Triangle PQR with side lengths PQ=6, QR=8, and PR=9 -/
structure Triangle :=
  (PQ : ℝ) (QR : ℝ) (PR : ℝ)

/-- Two bugs crawling along the perimeter of the triangle -/
structure BugMeeting (t : Triangle) :=
  (S : ℝ)  -- Position of meeting point S on side QR

/-- Main theorem: QS = 5.5 when bugs meet -/
theorem bug_meeting_point (t : Triangle) (b : BugMeeting t) :
  t.PQ = 6 → t.QR = 8 → t.PR = 9 → b.S = 5.5 := by
  sorry

#check bug_meeting_point

end NUMINAMATH_CALUDE_bug_meeting_point_l2094_209428


namespace NUMINAMATH_CALUDE_simplify_expression_l2094_209494

theorem simplify_expression (x y : ℝ) : 3 * x^2 - 2 * x * y - 3 * x^2 + 4 * x * y - 1 = 2 * x * y - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2094_209494


namespace NUMINAMATH_CALUDE_m_range_is_correct_l2094_209453

/-- Proposition p: The solution set of (x-1)^2 > m-1 is ℝ -/
def prop_p (m : ℝ) : Prop :=
  ∀ x, (x - 1)^2 > m - 1

/-- Proposition q: f(x) = (5-2m)^x is an increasing function on ℝ -/
def prop_q (m : ℝ) : Prop :=
  ∀ x y, x < y → (5 - 2*m)^x < (5 - 2*m)^y

/-- The range of m satisfying the given conditions -/
def m_range : Set ℝ :=
  {m | (prop_p m ∨ prop_q m) ∧ ¬(prop_p m ∧ prop_q m)}

theorem m_range_is_correct :
  m_range = Set.Ici 1 ∩ Set.Iio 2 :=
by
  sorry

end NUMINAMATH_CALUDE_m_range_is_correct_l2094_209453


namespace NUMINAMATH_CALUDE_coffee_price_increase_l2094_209415

/-- Calculates the percentage increase in coffee price given the original conditions and savings. -/
theorem coffee_price_increase 
  (original_price : ℝ) 
  (original_quantity : ℕ) 
  (new_quantity : ℕ) 
  (daily_savings : ℝ) 
  (h1 : original_price = 2)
  (h2 : original_quantity = 4)
  (h3 : new_quantity = 2)
  (h4 : daily_savings = 2) : 
  (((original_price * original_quantity - daily_savings) / new_quantity - original_price) / original_price) * 100 = 50 := by
  sorry

#check coffee_price_increase

end NUMINAMATH_CALUDE_coffee_price_increase_l2094_209415


namespace NUMINAMATH_CALUDE_marcella_shoes_theorem_l2094_209491

/-- Given an initial number of shoe pairs and a number of individual shoes lost,
    calculate the maximum number of complete pairs remaining. -/
def max_pairs_remaining (initial_pairs : ℕ) (shoes_lost : ℕ) : ℕ :=
  initial_pairs - shoes_lost

/-- Theorem: Given 25 initial pairs of shoes and losing 9 individual shoes,
    the maximum number of complete pairs remaining is 16. -/
theorem marcella_shoes_theorem :
  max_pairs_remaining 25 9 = 16 := by
  sorry

#eval max_pairs_remaining 25 9

end NUMINAMATH_CALUDE_marcella_shoes_theorem_l2094_209491


namespace NUMINAMATH_CALUDE_root_reciprocal_sum_l2094_209401

theorem root_reciprocal_sum (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ - 1 = 0 → x₂^2 - 3*x₂ - 1 = 0 → x₁ ≠ x₂ → 
  (1/x₁) + (1/x₂) = -3 := by
  sorry

end NUMINAMATH_CALUDE_root_reciprocal_sum_l2094_209401


namespace NUMINAMATH_CALUDE_missing_number_proof_l2094_209414

theorem missing_number_proof : ∃ n : ℝ, n * 120 = 173 * 240 ∧ n = 345.6 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l2094_209414


namespace NUMINAMATH_CALUDE_map_scale_l2094_209438

/-- Given a map scale where 15 cm represents 90 km, 
    prove that 20 cm on the map represents 120 km in reality. -/
theorem map_scale (scale : ℝ → ℝ) 
  (h1 : scale 15 = 90) -- 15 cm on map represents 90 km in reality
  (h2 : ∀ x : ℝ, scale x = (x / 15) * 90) -- scale is linear
  : scale 20 = 120 := by
  sorry

end NUMINAMATH_CALUDE_map_scale_l2094_209438


namespace NUMINAMATH_CALUDE_abs_cube_complex_l2094_209464

/-- The absolute value of (3 + √7i)^3 is equal to 64, where i is the imaginary unit. -/
theorem abs_cube_complex : Complex.abs ((3 + Complex.I * Real.sqrt 7) ^ 3) = 64 := by
  sorry

end NUMINAMATH_CALUDE_abs_cube_complex_l2094_209464


namespace NUMINAMATH_CALUDE_annie_village_trick_or_treat_l2094_209431

/-- The number of blocks in Annie's village -/
def num_blocks : ℕ := 9

/-- The number of children on each block -/
def children_per_block : ℕ := 6

/-- The total number of children going trick or treating in Annie's village -/
def total_children : ℕ := num_blocks * children_per_block

theorem annie_village_trick_or_treat : total_children = 54 := by
  sorry

end NUMINAMATH_CALUDE_annie_village_trick_or_treat_l2094_209431


namespace NUMINAMATH_CALUDE_trajectory_is_straight_line_l2094_209427

/-- The set of points P(x,y) satisfying the given equation forms a straight line -/
theorem trajectory_is_straight_line :
  ∃ (a b c : ℝ), a ≠ 0 ∨ b ≠ 0 ∧
  ∀ (x y : ℝ), Real.sqrt ((x - 1)^2 + (y - 1)^2) = |x + y - 2| / Real.sqrt 2 →
  a * x + b * y + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_is_straight_line_l2094_209427


namespace NUMINAMATH_CALUDE_fraction_equality_expression_equality_l2094_209400

-- Problem 1
theorem fraction_equality : (2021 * 2023) / (2022^2 - 1) = 1 := by sorry

-- Problem 2
theorem expression_equality : 2 * 101^2 + 2 * 101 * 98 + 2 * 49^2 = 45000 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_expression_equality_l2094_209400


namespace NUMINAMATH_CALUDE_test_scores_mode_l2094_209492

/-- Represents a stem-and-leaf plot entry -/
structure StemLeafEntry where
  stem : ℕ
  leaves : List ℕ

/-- The stem-and-leaf plot data -/
def testScores : List StemLeafEntry := [
  ⟨4, [5, 5, 5]⟩,
  ⟨5, [2, 6, 6]⟩,
  ⟨6, [1, 3, 3, 3, 3]⟩,
  ⟨7, [2, 4, 5, 5, 5, 5, 5]⟩,
  ⟨8, [0, 3, 6]⟩,
  ⟨9, [1, 1, 4, 7]⟩
]

/-- Convert a stem-leaf entry to a list of full scores -/
def toFullScores (entry : StemLeafEntry) : List ℕ :=
  entry.leaves.map (λ leaf => entry.stem * 10 + leaf)

/-- Find the mode of a list of numbers -/
def mode (numbers : List ℕ) : ℕ := sorry

/-- The main theorem stating that the mode of the test scores is 75 -/
theorem test_scores_mode :
  mode (testScores.bind toFullScores) = 75 := by sorry

end NUMINAMATH_CALUDE_test_scores_mode_l2094_209492


namespace NUMINAMATH_CALUDE_pencil_price_l2094_209437

theorem pencil_price (total_pencils : ℕ) (total_cost : ℚ) (h1 : total_pencils = 10) (h2 : total_cost = 2) :
  total_cost / total_pencils = 1/5 := by
sorry

end NUMINAMATH_CALUDE_pencil_price_l2094_209437


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l2094_209435

theorem greatest_divisor_four_consecutive_integers :
  ∀ n : ℕ, n > 0 →
  ∃ m : ℕ, m > 0 ∧ m ∣ (n * (n + 1) * (n + 2) * (n + 3)) ∧
  ∀ k : ℕ, k > m → ¬(∀ p : ℕ, p > 0 → k ∣ (p * (p + 1) * (p + 2) * (p + 3))) →
  m = 24 :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l2094_209435


namespace NUMINAMATH_CALUDE_tennis_players_l2094_209402

theorem tennis_players (total : ℕ) (badminton : ℕ) (neither : ℕ) (both : ℕ)
  (h1 : total = 42)
  (h2 : badminton = 20)
  (h3 : neither = 6)
  (h4 : both = 7) :
  total - (badminton - both) - neither = 23 :=
by sorry

end NUMINAMATH_CALUDE_tennis_players_l2094_209402


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2094_209442

theorem complex_fraction_equality : (2 - I) / (1 + 2*I) = -I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2094_209442


namespace NUMINAMATH_CALUDE_aron_cleaning_time_l2094_209467

/-- Calculates the total cleaning time per week for Aron -/
def total_cleaning_time (vacuum_time : ℕ) (vacuum_freq : ℕ) (dust_time : ℕ) (dust_freq : ℕ) : ℕ :=
  vacuum_time * vacuum_freq + dust_time * dust_freq

/-- Proves that Aron spends 130 minutes per week cleaning -/
theorem aron_cleaning_time :
  total_cleaning_time 30 3 20 2 = 130 := by
  sorry

end NUMINAMATH_CALUDE_aron_cleaning_time_l2094_209467


namespace NUMINAMATH_CALUDE_multiplication_problem_l2094_209408

theorem multiplication_problem : ∃ x : ℕ, 582964 * x = 58293485180 ∧ x = 100000 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_problem_l2094_209408


namespace NUMINAMATH_CALUDE_smallest_value_l2094_209406

theorem smallest_value (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  x^3 < x^2 ∧ x^3 < 3*x ∧ x^3 < Real.sqrt x ∧ x^3 < 1/x := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_l2094_209406


namespace NUMINAMATH_CALUDE_zeros_in_20_pow_10_eq_11_l2094_209434

/-- The number of zeros in the decimal representation of 20^10 -/
def zeros_in_20_pow_10 : ℕ :=
  let base_20_pow_10 := (20 : ℕ) ^ 10
  let digits := base_20_pow_10.digits 10
  digits.count 0

/-- Theorem stating that the number of zeros in 20^10 is 11 -/
theorem zeros_in_20_pow_10_eq_11 : zeros_in_20_pow_10 = 11 := by
  sorry

end NUMINAMATH_CALUDE_zeros_in_20_pow_10_eq_11_l2094_209434


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2094_209495

/-- An arithmetic sequence is a sequence where the difference between consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that for an arithmetic sequence where the 20th term is 15 
    and the 21st term is 18, the 5th term is -30. -/
theorem arithmetic_sequence_fifth_term 
  (a : ℕ → ℤ) (h : ArithmeticSequence a) 
  (h20 : a 20 = 15) (h21 : a 21 = 18) : 
  a 5 = -30 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2094_209495


namespace NUMINAMATH_CALUDE_football_tournament_l2094_209412

theorem football_tournament (n : ℕ) (k : ℕ) : 
  n > 0 →  -- Ensure n is positive
  (n * (n - 1)) / 2 + k * n = 77 →  -- Total matches equation
  2 * n = 14  -- Prove that the initial number of teams is 14
  := by sorry

end NUMINAMATH_CALUDE_football_tournament_l2094_209412


namespace NUMINAMATH_CALUDE_five_pairs_l2094_209471

/-- The number of ordered pairs (b,c) of positive integers satisfying the given conditions -/
def count_pairs : ℕ := 
  (Finset.filter (fun p : ℕ × ℕ => 
    let b := p.1
    let c := p.2
    b > 0 ∧ c > 0 ∧ b^2 ≤ 9*c ∧ c^2 ≤ 9*b) 
  (Finset.product (Finset.range 4) (Finset.range 4))).card

/-- The theorem stating that there are exactly 5 such pairs -/
theorem five_pairs : count_pairs = 5 := by sorry

end NUMINAMATH_CALUDE_five_pairs_l2094_209471


namespace NUMINAMATH_CALUDE_triangle_isosceles_theorem_l2094_209468

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- A triangle is isosceles if at least two of its sides are equal. -/
def isIsosceles (a b c : ℕ) : Prop := a = b ∨ b = c ∨ a = c

theorem triangle_isosceles_theorem (a b c : ℕ) 
  (ha : isPrime a) (hb : isPrime b) (hc : isPrime c) 
  (hsum : a + b + c = 16) : 
  isIsosceles a b c := by
  sorry

#check triangle_isosceles_theorem

end NUMINAMATH_CALUDE_triangle_isosceles_theorem_l2094_209468


namespace NUMINAMATH_CALUDE_root_sum_squares_l2094_209423

theorem root_sum_squares (p q r : ℝ) : 
  (p + q + r = 15) → (p * q + q * r + r * p = 25) → 
  (p + q)^2 + (q + r)^2 + (r + p)^2 = 400 := by
sorry

end NUMINAMATH_CALUDE_root_sum_squares_l2094_209423


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2094_209462

theorem quadratic_inequality_solution (x : ℝ) :
  2 * x^2 - 4 * x - 70 > 0 ∧ x ≠ -2 ∧ x ≠ 0 →
  x < -5 ∨ x > 7 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2094_209462


namespace NUMINAMATH_CALUDE_calculation_problem_1_calculation_problem_2_l2094_209485

-- Question 1
theorem calculation_problem_1 :
  (-1/4)⁻¹ - |Real.sqrt 3 - 1| + 3 * Real.tan (30 * π / 180) + (2017 - π) = -2 := by sorry

-- Question 2
theorem calculation_problem_2 (x : ℝ) (h : x = 2) :
  (2 * x^2) / (x^2 - 2*x + 1) / ((2*x + 1) / (x + 1) + 1 / (x - 1)) = 3 := by sorry

end NUMINAMATH_CALUDE_calculation_problem_1_calculation_problem_2_l2094_209485


namespace NUMINAMATH_CALUDE_at_least_four_same_prob_l2094_209407

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The probability of rolling a specific value on a single die -/
def single_prob : ℚ := 1 / sides

/-- The probability that all five dice show the same number -/
def all_same_prob : ℚ := single_prob ^ (num_dice - 1)

/-- The probability that exactly four dice show the same number and one die shows a different number -/
def four_same_prob : ℚ := num_dice * (single_prob ^ (num_dice - 2)) * ((sides - 1) / sides)

/-- The theorem stating the probability of at least four out of five fair six-sided dice showing the same value -/
theorem at_least_four_same_prob : all_same_prob + four_same_prob = 13 / 648 := by
  sorry

end NUMINAMATH_CALUDE_at_least_four_same_prob_l2094_209407


namespace NUMINAMATH_CALUDE_ducks_to_chickens_ratio_l2094_209436

/-- Represents the farm animals --/
structure Farm :=
  (chickens : ℕ)
  (ducks : ℕ)
  (turkeys : ℕ)

/-- The conditions of Mr. Valentino's farm --/
def valentino_farm (f : Farm) : Prop :=
  f.chickens = 200 ∧
  f.turkeys = 3 * f.ducks ∧
  f.chickens + f.ducks + f.turkeys = 1800

/-- The theorem stating the ratio of ducks to chickens --/
theorem ducks_to_chickens_ratio (f : Farm) :
  valentino_farm f → (f.ducks : ℚ) / f.chickens = 2 := by
  sorry

#check ducks_to_chickens_ratio

end NUMINAMATH_CALUDE_ducks_to_chickens_ratio_l2094_209436


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2094_209441

/-- The complex number z = 2i / (1 + i) has both positive real and imaginary parts -/
theorem complex_number_in_first_quadrant : 
  let z : ℂ := (2 * Complex.I) / (1 + Complex.I)
  0 < z.re ∧ 0 < z.im := by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2094_209441


namespace NUMINAMATH_CALUDE_four_inch_cube_painted_faces_l2094_209463

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ := n

/-- Counts the number of 1-inch cubes with at least two painted faces in a painted n×n×n cube -/
def count_painted_cubes (c : Cube n) : ℕ :=
  sorry

/-- Theorem: In a 4x4x4 painted cube, there are 56 1-inch cubes with at least two painted faces -/
theorem four_inch_cube_painted_faces :
  ∃ (c : Cube 4), count_painted_cubes c = 56 := by
  sorry

end NUMINAMATH_CALUDE_four_inch_cube_painted_faces_l2094_209463


namespace NUMINAMATH_CALUDE_product_of_base8_digits_8675_l2094_209482

/-- Converts a natural number from base 10 to base 8 -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the product of a list of natural numbers -/
def productOfList (l : List ℕ) : ℕ :=
  sorry

/-- The product of the digits in the base 8 representation of 8675 (base 10) is 0 -/
theorem product_of_base8_digits_8675 :
  productOfList (toBase8 8675) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_of_base8_digits_8675_l2094_209482


namespace NUMINAMATH_CALUDE_hundred_hours_before_seven_am_l2094_209409

/-- Represents a time of day in hours and minutes -/
structure TimeOfDay where
  hours : Nat
  minutes : Nat
  valid : hours < 24 ∧ minutes < 60

/-- Calculates the time a given number of hours before a specified time -/
def timeBefore (t : TimeOfDay) (h : Nat) : TimeOfDay :=
  sorry

/-- Theorem: 100 hours before 7:00 a.m. is 3:00 a.m. -/
theorem hundred_hours_before_seven_am :
  let start_time : TimeOfDay := ⟨7, 0, by sorry⟩
  let end_time : TimeOfDay := ⟨3, 0, by sorry⟩
  timeBefore start_time 100 = end_time := by
  sorry

end NUMINAMATH_CALUDE_hundred_hours_before_seven_am_l2094_209409


namespace NUMINAMATH_CALUDE_crayons_in_box_l2094_209405

def crayons_problem (given_away lost : ℕ) (difference : ℤ) : Prop :=
  given_away = 90 ∧
  lost = 412 ∧
  difference = lost - given_away ∧
  difference = 322

theorem crayons_in_box (given_away lost : ℕ) (difference : ℤ) 
  (h : crayons_problem given_away lost difference) : 
  given_away + lost = 502 := by
  sorry

end NUMINAMATH_CALUDE_crayons_in_box_l2094_209405


namespace NUMINAMATH_CALUDE_six_foldable_configurations_l2094_209488

/-- Represents a square in the puzzle -/
inductive Square
| A | B | C | D | E | F | G | H

/-- Represents the T-shaped figure -/
structure TShape :=
  (squares : Finset Square)
  (h_count : squares.card = 4)

/-- Represents a configuration of the puzzle -/
structure Configuration :=
  (base : TShape)
  (added : Square)

/-- Predicate to check if a configuration can be folded into a topless cubical box -/
def is_foldable (c : Configuration) : Prop :=
  sorry  -- Definition of foldability

/-- The main theorem statement -/
theorem six_foldable_configurations :
  ∃ (valid_configs : Finset Configuration),
    valid_configs.card = 6 ∧
    (∀ c ∈ valid_configs, is_foldable c) ∧
    (∀ c : Configuration, is_foldable c → c ∈ valid_configs) :=
  sorry

end NUMINAMATH_CALUDE_six_foldable_configurations_l2094_209488
