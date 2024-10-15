import Mathlib

namespace NUMINAMATH_CALUDE_functional_equation_solutions_l3329_332947

/-- A function satisfying the given functional equation. -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x * f y) + f (x + y) = f (x * y)

/-- The theorem stating that any function satisfying the functional equation
    must be one of the three specified functions. -/
theorem functional_equation_solutions (f : ℝ → ℝ) 
  (h : SatisfiesFunctionalEquation f) : 
  (∀ x, f x = 0) ∨ (∀ x, f x = x - 1) ∨ (∀ x, f x = 1 - x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l3329_332947


namespace NUMINAMATH_CALUDE_expression_equality_l3329_332911

theorem expression_equality : (2 + Real.sqrt 6) * (2 - Real.sqrt 6) - (Real.sqrt 3 + 1)^2 = -6 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3329_332911


namespace NUMINAMATH_CALUDE_divisible_by_two_l3329_332936

theorem divisible_by_two (a : ℤ) (h : 2 ∣ a^2) : 2 ∣ a := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_two_l3329_332936


namespace NUMINAMATH_CALUDE_sum_of_b_and_c_is_eleven_l3329_332907

theorem sum_of_b_and_c_is_eleven
  (a b c : ℕ+)
  (ha : a ≠ 1)
  (hb : b ≤ 9)
  (hc : c ≤ 9)
  (hbc : b ≠ c)
  (heq : (10 * a + b) * (10 * a + c) = 100 * a^2 + 110 * a + b * c) :
  b + c = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_b_and_c_is_eleven_l3329_332907


namespace NUMINAMATH_CALUDE_fraction_order_l3329_332962

theorem fraction_order : 
  (21 : ℚ) / 17 < 18 / 13 ∧ 18 / 13 < 16 / 11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_order_l3329_332962


namespace NUMINAMATH_CALUDE_share_ratio_a_to_b_l3329_332914

/-- Prove that the ratio of A's share to B's share is 4:1 -/
theorem share_ratio_a_to_b (total amount : ℕ) (a_share b_share c_share : ℕ) :
  amount = 578 →
  b_share = c_share / 4 →
  a_share = 408 →
  b_share = 102 →
  c_share = 68 →
  a_share + b_share + c_share = amount →
  a_share / b_share = 4 := by
  sorry

end NUMINAMATH_CALUDE_share_ratio_a_to_b_l3329_332914


namespace NUMINAMATH_CALUDE_rosa_phone_calls_l3329_332961

theorem rosa_phone_calls (last_week : ℝ) (this_week : ℝ) (total : ℝ) 
  (h1 : last_week = 10.2)
  (h2 : this_week = 8.6)
  (h3 : total = last_week + this_week) :
  total = 18.8 := by
sorry

end NUMINAMATH_CALUDE_rosa_phone_calls_l3329_332961


namespace NUMINAMATH_CALUDE_problem_solution_l3329_332977

def f (x : ℝ) : ℝ := |x + 1| - |x - 4|

theorem problem_solution :
  (∀ m : ℝ, (∀ x : ℝ, f x ≤ -m^2 + 6*m) ↔ (1 ≤ m ∧ m ≤ 5)) ∧
  (∃ m₀ : ℝ, m₀ = 1 ∧ ∀ m : ℝ, (∀ x : ℝ, f x ≤ -m^2 + 6*m) → m₀ ≤ m) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 3*a + 4*b + 5*c = 1 →
    a^2 + b^2 + c^2 ≥ 1/50 ∧ ∃ a₀ b₀ c₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧
      3*a₀ + 4*b₀ + 5*c₀ = 1 ∧ a₀^2 + b₀^2 + c₀^2 = 1/50) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3329_332977


namespace NUMINAMATH_CALUDE_gigi_additional_batches_l3329_332915

/-- Represents the number of cups of flour required for one batch of cookies -/
def flour_per_batch : ℕ := 2

/-- Represents the number of batches Gigi has already baked -/
def batches_baked : ℕ := 3

/-- Represents the total amount of flour in Gigi's bag -/
def total_flour : ℕ := 20

/-- Calculates the number of additional batches Gigi can make with the remaining flour -/
def additional_batches : ℕ := (total_flour - batches_baked * flour_per_batch) / flour_per_batch

/-- Proves that Gigi can make 7 more batches of cookies with the remaining flour -/
theorem gigi_additional_batches : additional_batches = 7 := by
  sorry

end NUMINAMATH_CALUDE_gigi_additional_batches_l3329_332915


namespace NUMINAMATH_CALUDE_indeterminate_roots_l3329_332987

/-- Given that the equation mx^2 - 2(m+2)x + m + 5 = 0 has no real roots,
    the number of real roots of (m-5)x^2 - 2(m+2)x + m = 0 cannot be determined
    to be exclusively 0, 1, or 2. -/
theorem indeterminate_roots (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - 2*(m+2)*x + m + 5 ≠ 0) →
  ¬(∀ x : ℝ, (m-5) * x^2 - 2*(m+2)*x + m ≠ 0) ∧
  ¬(∃! x : ℝ, (m-5) * x^2 - 2*(m+2)*x + m = 0) ∧
  ¬(∃ x y : ℝ, x ≠ y ∧ (m-5) * x^2 - 2*(m+2)*x + m = 0 ∧ (m-5) * y^2 - 2*(m+2)*y + m = 0) :=
sorry

end NUMINAMATH_CALUDE_indeterminate_roots_l3329_332987


namespace NUMINAMATH_CALUDE_product_of_largest_and_smallest_three_digit_l3329_332964

def largest_three_digit (a b c : Nat) : Nat :=
  100 * max a (max b c) + 10 * (if a > b ∧ a > c then max b c else if b > a ∧ b > c then max a c else max a b) +
  min a (min b c)

def smallest_three_digit (a b c : Nat) : Nat :=
  100 * (if a > 0 then a else if b > 0 then b else c) +
  10 * (if a > 0 ∧ b > 0 then min a b else if a > 0 ∧ c > 0 then min a c else min b c) +
  (if a = 0 then 0 else if b = 0 then 0 else c)

theorem product_of_largest_and_smallest_three_digit :
  largest_three_digit 6 0 2 * smallest_three_digit 6 0 2 = 127720 := by
  sorry

end NUMINAMATH_CALUDE_product_of_largest_and_smallest_three_digit_l3329_332964


namespace NUMINAMATH_CALUDE_smallest_divisible_by_12_13_14_l3329_332979

theorem smallest_divisible_by_12_13_14 : ∃ n : ℕ, n > 0 ∧ 12 ∣ n ∧ 13 ∣ n ∧ 14 ∣ n ∧ ∀ m : ℕ, m > 0 → 12 ∣ m → 13 ∣ m → 14 ∣ m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_12_13_14_l3329_332979


namespace NUMINAMATH_CALUDE_contractor_fine_calculation_l3329_332986

/-- Calculates the daily fine for absence given the contract parameters -/
def calculate_daily_fine (total_days : ℕ) (daily_pay : ℚ) (total_payment : ℚ) (absent_days : ℕ) : ℚ :=
  let worked_days := total_days - absent_days
  let earned_amount := daily_pay * worked_days
  (earned_amount - total_payment) / absent_days

/-- Proves that the daily fine is 7.5 given the contract parameters -/
theorem contractor_fine_calculation :
  calculate_daily_fine 30 25 490 8 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_contractor_fine_calculation_l3329_332986


namespace NUMINAMATH_CALUDE_simplify_expression_l3329_332952

theorem simplify_expression (y : ℝ) : 4*y + 9*y^2 + 6 - (3 - 4*y - 9*y^2) = 18*y^2 + 8*y + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3329_332952


namespace NUMINAMATH_CALUDE_a_gt_b_neither_sufficient_nor_necessary_for_a_sq_gt_b_sq_l3329_332985

theorem a_gt_b_neither_sufficient_nor_necessary_for_a_sq_gt_b_sq :
  ∃ a b : ℝ, (a > b ∧ ¬(a^2 > b^2)) ∧ ∃ c d : ℝ, (c^2 > d^2 ∧ ¬(c > d)) := by
  sorry

end NUMINAMATH_CALUDE_a_gt_b_neither_sufficient_nor_necessary_for_a_sq_gt_b_sq_l3329_332985


namespace NUMINAMATH_CALUDE_remainder_theorem_l3329_332938

theorem remainder_theorem (x y q r : ℕ) (h1 : x = q * y + r) (h2 : r < y) :
  (x - 3 * q * y) % y = r := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3329_332938


namespace NUMINAMATH_CALUDE_tangent_line_and_monotonicity_l3329_332978

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 6*a*x^2

theorem tangent_line_and_monotonicity :
  -- Part I: Tangent line equation
  (∀ x y : ℝ, y = f (-1) x → (x = 1 ∧ y = 7) →
    ∃ k m : ℝ, k = 15 ∧ m = -8 ∧ k*x + (-1)*y + m = 0) ∧
  -- Part II: Monotonicity
  (∀ a : ℝ,
    -- Case a = 0
    (a = 0 → ∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂) ∧
    -- Case a < 0
    (a < 0 → ∀ x₁ x₂ : ℝ,
      ((x₁ < x₂ ∧ x₂ < 4*a) ∨ (x₁ < x₂ ∧ 0 < x₁)) → f a x₁ < f a x₂) ∧
    (a < 0 → ∀ x₁ x₂ : ℝ, (4*a < x₁ ∧ x₁ < x₂ ∧ x₂ < 0) → f a x₁ > f a x₂) ∧
    -- Case a > 0
    (a > 0 → ∀ x₁ x₂ : ℝ,
      ((x₁ < x₂ ∧ x₂ < 0) ∨ (4*a < x₁ ∧ x₁ < x₂)) → f a x₁ < f a x₂) ∧
    (a > 0 → ∀ x₁ x₂ : ℝ, (0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 4*a) → f a x₁ > f a x₂)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_monotonicity_l3329_332978


namespace NUMINAMATH_CALUDE_units_digit_of_99_factorial_l3329_332908

theorem units_digit_of_99_factorial (n : ℕ) : n = 99 → n.factorial % 10 = 0 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_99_factorial_l3329_332908


namespace NUMINAMATH_CALUDE_remainder_puzzle_l3329_332944

theorem remainder_puzzle : (9^4 + 8^5 + 7^6 + 5^3) % 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_puzzle_l3329_332944


namespace NUMINAMATH_CALUDE_divisors_of_90_l3329_332967

def n : ℕ := 90

/-- The number of positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- The sum of all positive divisors of n -/
def sum_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem divisors_of_90 :
  num_divisors n = 12 ∧ sum_divisors n = 234 := by sorry

end NUMINAMATH_CALUDE_divisors_of_90_l3329_332967


namespace NUMINAMATH_CALUDE_largest_sum_of_squared_differences_l3329_332933

theorem largest_sum_of_squared_differences (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a > 0 ∧ b > 0 ∧ c > 0 →
  ∃ x y z : ℕ, (b + c - a = x^2) ∧ (c + a - b = y^2) ∧ (a + b - c = z^2) →
  a + b + c < 100 →
  a + b + c ≤ 91 :=
by sorry

end NUMINAMATH_CALUDE_largest_sum_of_squared_differences_l3329_332933


namespace NUMINAMATH_CALUDE_factorization_equality_l3329_332929

theorem factorization_equality (z : ℝ) :
  70 * z^20 + 154 * z^40 + 224 * z^60 = 14 * z^20 * (5 + 11 * z^20 + 16 * z^40) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3329_332929


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3329_332909

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (x > Real.exp 1 → x > 1) ∧ ∃ x, x > 1 ∧ x ≤ Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3329_332909


namespace NUMINAMATH_CALUDE_solution_to_linear_equation_l3329_332906

theorem solution_to_linear_equation :
  ∃ (x y : ℝ), 2 * x + y = 6 ∧ x = 2 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_linear_equation_l3329_332906


namespace NUMINAMATH_CALUDE_forest_trees_l3329_332951

/-- Calculates the total number of trees in a forest given the conditions --/
theorem forest_trees (street_side : ℝ) (forest_area_multiplier : ℝ) (trees_per_sqm : ℝ) : 
  street_side = 100 →
  forest_area_multiplier = 3 →
  trees_per_sqm = 4 →
  (forest_area_multiplier * street_side^2 * trees_per_sqm : ℝ) = 120000 := by
  sorry

#check forest_trees

end NUMINAMATH_CALUDE_forest_trees_l3329_332951


namespace NUMINAMATH_CALUDE_puppies_per_cage_l3329_332969

def initial_puppies : ℕ := 56
def sold_puppies : ℕ := 24
def num_cages : ℕ := 8

theorem puppies_per_cage :
  (initial_puppies - sold_puppies) / num_cages = 4 :=
by sorry

end NUMINAMATH_CALUDE_puppies_per_cage_l3329_332969


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3329_332924

theorem polynomial_factorization (x : ℝ) :
  x^4 - 5*x^2 + 4 = (x + 1)*(x - 1)*(x + 2)*(x - 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3329_332924


namespace NUMINAMATH_CALUDE_chopstick_length_l3329_332976

/-- The length of a chopstick given specific wetness conditions -/
theorem chopstick_length (wetted_length : ℝ) (h1 : wetted_length = 8) 
  (h2 : wetted_length + wetted_length / 2 + wetted_length = 24) : ℝ :=
by
  sorry

#check chopstick_length

end NUMINAMATH_CALUDE_chopstick_length_l3329_332976


namespace NUMINAMATH_CALUDE_least_exponent_sum_for_400_l3329_332948

def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

def sum_of_distinct_powers_of_two (n : ℕ) (powers : List ℕ) : Prop :=
  (powers.length ≥ 2) ∧
  (∀ p ∈ powers, is_power_of_two p) ∧
  (powers.sum = n) ∧
  (powers.toFinset.card = powers.length)

def exponent_sum (powers : List ℕ) : ℕ :=
  (powers.map (λ p => (Nat.log p 2))).sum

theorem least_exponent_sum_for_400 :
  ∀ powers : List ℕ,
    sum_of_distinct_powers_of_two 400 powers →
    exponent_sum powers ≥ 19 :=
sorry

end NUMINAMATH_CALUDE_least_exponent_sum_for_400_l3329_332948


namespace NUMINAMATH_CALUDE_rubber_band_length_l3329_332984

theorem rubber_band_length (r₁ r₂ d : ℝ) (hr₁ : r₁ = 3) (hr₂ : r₂ = 9) (hd : d = 12) :
  ∃ (L : ℝ), L = 4 * Real.pi + 12 * Real.sqrt 3 ∧
  L = 2 * (r₁ * Real.arctan ((Real.sqrt (d^2 - (r₂ - r₁)^2)) / (r₂ - r₁)) +
           r₂ * Real.arctan ((Real.sqrt (d^2 - (r₂ - r₁)^2)) / (r₂ - r₁)) +
           Real.sqrt (d^2 - (r₂ - r₁)^2)) :=
by sorry

end NUMINAMATH_CALUDE_rubber_band_length_l3329_332984


namespace NUMINAMATH_CALUDE_problem_solution_l3329_332922

/-- The number of positive factors of n -/
def num_factors (n : ℕ) : ℕ := sorry

theorem problem_solution (y : ℕ) 
  (h1 : num_factors y = 18) 
  (h2 : 14 ∣ y) 
  (h3 : 18 ∣ y) : 
  y = 252 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3329_332922


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l3329_332939

theorem min_value_quadratic_form (x y : ℝ) : x^2 - x*y + y^2 ≥ 0 ∧ 
  (x^2 - x*y + y^2 = 0 ↔ x = 0 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l3329_332939


namespace NUMINAMATH_CALUDE_empty_set_proof_l3329_332901

theorem empty_set_proof : {x : ℝ | x > 6 ∧ x < 1} = ∅ := by
  sorry

end NUMINAMATH_CALUDE_empty_set_proof_l3329_332901


namespace NUMINAMATH_CALUDE_extension_point_coordinates_l3329_332900

/-- Given two points P₁ and P₂ in ℝ², and a point P on the extension line of P₁P₂
    such that the distance from P₁ to P is twice the distance from P to P₂,
    prove that P has the specified coordinates. -/
theorem extension_point_coordinates (P₁ P₂ P : ℝ × ℝ) : 
  P₁ = (2, -1) →
  P₂ = (0, 5) →
  (∃ t : ℝ, t ∉ [0, 1] ∧ P = P₁ + t • (P₂ - P₁)) →
  ‖P - P₁‖ = 2 * ‖P - P₂‖ →
  P = (-2, 11) := by sorry

end NUMINAMATH_CALUDE_extension_point_coordinates_l3329_332900


namespace NUMINAMATH_CALUDE_soccer_team_goalies_l3329_332916

theorem soccer_team_goalies :
  ∀ (goalies defenders midfielders strikers : ℕ),
    defenders = 10 →
    midfielders = 2 * defenders →
    strikers = 7 →
    goalies + defenders + midfielders + strikers = 40 →
    goalies = 3 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_goalies_l3329_332916


namespace NUMINAMATH_CALUDE_max_value_F_l3329_332932

/-- The function f(x) = ax² + bx + c -/
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The function g(x) = cx² + bx + a -/
def g (a b c x : ℝ) : ℝ := c * x^2 + b * x + a

/-- The function F(x) = |f(x) · g(x)| -/
def F (a b c x : ℝ) : ℝ := |f a b c x * g a b c x|

theorem max_value_F (a b c : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, |f a b c x| ≤ 1) →
  ∃ M, M = 2 ∧ ∀ x ∈ Set.Icc (-1 : ℝ) 1, F a b c x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_F_l3329_332932


namespace NUMINAMATH_CALUDE_function_identity_l3329_332918

theorem function_identity (f : ℕ+ → ℕ+) 
  (h : ∀ (m n : ℕ+), f (f m ^ 2 + 2 * f n ^ 2) = m ^ 2 + 2 * n ^ 2) : 
  ∀ (n : ℕ+), f n = n := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l3329_332918


namespace NUMINAMATH_CALUDE_factor_75x_plus_50_l3329_332972

theorem factor_75x_plus_50 (x : ℝ) : 75 * x + 50 = 25 * (3 * x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_75x_plus_50_l3329_332972


namespace NUMINAMATH_CALUDE_xia_initial_stickers_l3329_332902

/-- The number of stickers Xia shared with her friends -/
def shared_stickers : ℕ := 100

/-- The number of sheets of stickers Xia had left -/
def remaining_sheets : ℕ := 5

/-- The number of stickers on each sheet -/
def stickers_per_sheet : ℕ := 10

/-- Theorem: Xia had 150 stickers at the beginning -/
theorem xia_initial_stickers :
  shared_stickers + remaining_sheets * stickers_per_sheet = 150 := by
  sorry

end NUMINAMATH_CALUDE_xia_initial_stickers_l3329_332902


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3329_332923

theorem decimal_to_fraction : (3.675 : ℚ) = 147 / 40 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3329_332923


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l3329_332943

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 + 8*x + 9 = 0 ↔ (x + 4)^2 = 7 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l3329_332943


namespace NUMINAMATH_CALUDE_joshua_bottle_caps_l3329_332966

theorem joshua_bottle_caps (initial : ℕ) (bought : ℕ) (total : ℕ) : 
  initial = 40 → bought = 7 → total = initial + bought → total = 47 := by
  sorry

end NUMINAMATH_CALUDE_joshua_bottle_caps_l3329_332966


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l3329_332995

theorem rectangle_area_increase (L W : ℝ) (h : L > 0 ∧ W > 0) : 
  let new_area := (1.1 * L) * (1.1 * W)
  let original_area := L * W
  (new_area - original_area) / original_area * 100 = 21 := by
sorry


end NUMINAMATH_CALUDE_rectangle_area_increase_l3329_332995


namespace NUMINAMATH_CALUDE_meeting_time_l3329_332982

/-- The speed of l in km/hr -/
def speed_l : ℝ := 50

/-- The speed of k in km/hr -/
def speed_k : ℝ := speed_l * 1.5

/-- The time difference between k's and l's start times in hours -/
def time_difference : ℝ := 1

/-- The total distance between k and l in km -/
def total_distance : ℝ := 300

/-- The time when l starts -/
def start_time_l : ℕ := 9

/-- The time when k starts -/
def start_time_k : ℕ := 10

theorem meeting_time :
  let distance_traveled_by_l := speed_l * time_difference
  let remaining_distance := total_distance - distance_traveled_by_l
  let relative_speed := speed_l + speed_k
  let time_to_meet := remaining_distance / relative_speed
  start_time_k + ⌊time_to_meet⌋ = 12 := by sorry

end NUMINAMATH_CALUDE_meeting_time_l3329_332982


namespace NUMINAMATH_CALUDE_bills_toddler_count_l3329_332910

/-- The number of toddlers Bill thinks he counted -/
def billsCount (actualCount doubleCount missedCount : ℕ) : ℕ :=
  actualCount + doubleCount - missedCount

/-- Theorem stating that Bill thinks he counted 26 toddlers -/
theorem bills_toddler_count :
  let actualCount : ℕ := 21
  let doubleCount : ℕ := 8
  let missedCount : ℕ := 3
  billsCount actualCount doubleCount missedCount = 26 := by
  sorry

end NUMINAMATH_CALUDE_bills_toddler_count_l3329_332910


namespace NUMINAMATH_CALUDE_find_divisor_l3329_332920

theorem find_divisor : ∃ (d : ℕ), d = 675 ∧ 
  (9679 - 4) % d = 0 ∧ 
  ∀ (k : ℕ), 0 < k → k < 4 → (9679 - k) % d ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l3329_332920


namespace NUMINAMATH_CALUDE_blackjack_payout_ratio_l3329_332931

/-- Represents the payout ratio for a blackjack in a casino game -/
structure BlackjackPayout where
  original_bet : ℚ
  total_payout : ℚ

/-- Calculates the payout ratio for a blackjack given the original bet and total payout -/
def payout_ratio (bp : BlackjackPayout) : ℚ × ℚ :=
  let winnings := bp.total_payout - bp.original_bet
  (winnings, bp.original_bet)

/-- Theorem stating that for the given conditions, the payout ratio is 1:2 -/
theorem blackjack_payout_ratio :
  let bp := BlackjackPayout.mk 40 60
  payout_ratio bp = (1, 2) := by
  sorry

end NUMINAMATH_CALUDE_blackjack_payout_ratio_l3329_332931


namespace NUMINAMATH_CALUDE_museum_visitors_l3329_332971

theorem museum_visitors (V T E : ℕ) : 
  V = 6 * T →
  E = 180 →
  E = 3 * T / 5 →
  V = 1800 :=
by sorry

end NUMINAMATH_CALUDE_museum_visitors_l3329_332971


namespace NUMINAMATH_CALUDE_factorial_equation_solutions_l3329_332927

theorem factorial_equation_solutions :
  ∀ x y z : ℕ, 2^x + 5^y + 63 = z! → ((x = 5 ∧ y = 2 ∧ z = 5) ∨ (x = 4 ∧ y = 4 ∧ z = 6)) :=
by sorry

end NUMINAMATH_CALUDE_factorial_equation_solutions_l3329_332927


namespace NUMINAMATH_CALUDE_basket_apples_theorem_l3329_332998

/-- The total number of apples in the basket -/
def total_apples : ℕ := 5

/-- The probability of selecting at least one spoiled apple when picking 2 apples randomly -/
def prob_spoiled : ℚ := 2/5

/-- The number of spoiled apples in the basket -/
def spoiled_apples : ℕ := 1

/-- The number of good apples in the basket -/
def good_apples : ℕ := total_apples - spoiled_apples

theorem basket_apples_theorem :
  (total_apples = spoiled_apples + good_apples) ∧
  (prob_spoiled = 1 - (good_apples / total_apples) * ((good_apples - 1) / (total_apples - 1))) :=
by sorry

end NUMINAMATH_CALUDE_basket_apples_theorem_l3329_332998


namespace NUMINAMATH_CALUDE_floor_of_e_l3329_332983

theorem floor_of_e : ⌊Real.exp 1⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_floor_of_e_l3329_332983


namespace NUMINAMATH_CALUDE_real_number_line_bijection_l3329_332955

/-- A point on the number line -/
structure NumberLinePoint where
  position : ℝ

/-- The bijective function between real numbers and points on the number line -/
def realToPoint : ℝ → NumberLinePoint :=
  λ x ↦ ⟨x⟩

theorem real_number_line_bijection :
  Function.Bijective realToPoint :=
sorry

end NUMINAMATH_CALUDE_real_number_line_bijection_l3329_332955


namespace NUMINAMATH_CALUDE_interval_constraint_l3329_332974

theorem interval_constraint (x : ℝ) : (1 < 2*x ∧ 2*x < 2) ∧ (1 < 3*x ∧ 3*x < 2) ↔ 1/2 < x ∧ x < 2/3 := by
  sorry

end NUMINAMATH_CALUDE_interval_constraint_l3329_332974


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l3329_332993

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (∀ k : ℕ, k * (k + 1) < 500 → k ≤ n) → 
  n * (n + 1) < 500 → 
  n + (n + 1) = 43 :=
by sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l3329_332993


namespace NUMINAMATH_CALUDE_rectangle_perimeter_in_square_l3329_332968

/-- Given a square of side length y containing a smaller square of side length x,
    the perimeter of one of the four congruent rectangles formed in the remaining area
    is equal to 2y. -/
theorem rectangle_perimeter_in_square (y x : ℝ) (h1 : 0 < y) (h2 : 0 < x) (h3 : x < y) :
  2 * (y - x) + 2 * x = 2 * y :=
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_in_square_l3329_332968


namespace NUMINAMATH_CALUDE_reciprocal_problem_l3329_332925

theorem reciprocal_problem (x : ℝ) (h : 8 * x = 5) : 50 * (1 / x) = 80 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l3329_332925


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3329_332946

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2

/-- Theorem: If 2S_3 = 3S_2 + 6 for an arithmetic sequence, then its common difference is 2 -/
theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence) 
  (h : 2 * seq.S 3 = 3 * seq.S 2 + 6) : 
  seq.d = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3329_332946


namespace NUMINAMATH_CALUDE_min_socks_for_twenty_pairs_l3329_332957

/-- Represents the number of socks of each color in the drawer -/
structure SockDrawer :=
  (red : Nat)
  (green : Nat)
  (blue : Nat)
  (black : Nat)

/-- Calculates the minimum number of socks needed to guarantee a certain number of pairs -/
def minSocksForPairs (drawer : SockDrawer) (requiredPairs : Nat) : Nat :=
  5 + 2 * (requiredPairs - 1)

/-- Theorem stating the minimum number of socks needed for 20 pairs -/
theorem min_socks_for_twenty_pairs (drawer : SockDrawer) 
  (h1 : drawer.red = 120)
  (h2 : drawer.green = 100)
  (h3 : drawer.blue = 80)
  (h4 : drawer.black = 50) :
  minSocksForPairs drawer 20 = 43 := by
  sorry

#eval minSocksForPairs { red := 120, green := 100, blue := 80, black := 50 } 20

end NUMINAMATH_CALUDE_min_socks_for_twenty_pairs_l3329_332957


namespace NUMINAMATH_CALUDE_rectangle_cylinder_volume_ratio_l3329_332904

theorem rectangle_cylinder_volume_ratio :
  let rectangle_width : ℝ := 7
  let rectangle_height : ℝ := 9
  let cylinder1_height : ℝ := rectangle_height
  let cylinder1_circumference : ℝ := rectangle_width
  let cylinder2_height : ℝ := rectangle_width
  let cylinder2_circumference : ℝ := rectangle_height
  let cylinder1_volume : ℝ := (cylinder1_circumference ^ 2 * cylinder1_height) / (4 * Real.pi)
  let cylinder2_volume : ℝ := (cylinder2_circumference ^ 2 * cylinder2_height) / (4 * Real.pi)
  let larger_volume : ℝ := max cylinder1_volume cylinder2_volume
  let smaller_volume : ℝ := min cylinder1_volume cylinder2_volume
  larger_volume / smaller_volume = 1 / 7 := by
sorry

end NUMINAMATH_CALUDE_rectangle_cylinder_volume_ratio_l3329_332904


namespace NUMINAMATH_CALUDE_intersection_integer_points_l3329_332992

theorem intersection_integer_points (k : ℝ) : 
  (∃! (n : ℕ), n = 3 ∧ 
    (∀ (x y : ℤ), 
      (y = 4*k*x - 1/k ∧ y = (1/k)*x + 2) → 
      (∃ (k₁ k₂ k₃ : ℝ), k = k₁ ∨ k = k₂ ∨ k = k₃))) :=
by sorry

end NUMINAMATH_CALUDE_intersection_integer_points_l3329_332992


namespace NUMINAMATH_CALUDE_min_value_expression_l3329_332949

theorem min_value_expression (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 1) (hab : a + b = 1) :
  ((2 * a + b) / (a * b) - 3) * c + Real.sqrt 2 / (c - 1) ≥ 4 + 2 * Real.sqrt 2 ∧
  (((2 * a + b) / (a * b) - 3) * c + Real.sqrt 2 / (c - 1) = 4 + 2 * Real.sqrt 2 ↔ 
    a = Real.sqrt 2 - 1 ∧ b = 2 - Real.sqrt 2 ∧ c = 1 + Real.sqrt 2 / 2) :=
by sorry

#check min_value_expression

end NUMINAMATH_CALUDE_min_value_expression_l3329_332949


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3329_332965

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 2*x + 2 > 0) ↔ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3329_332965


namespace NUMINAMATH_CALUDE_spinster_cat_difference_l3329_332941

theorem spinster_cat_difference (spinster_count : ℕ) (cat_count : ℕ) : 
  spinster_count = 14 →
  (2 : ℚ) / 7 = spinster_count / cat_count →
  cat_count > spinster_count →
  cat_count - spinster_count = 35 := by
sorry

end NUMINAMATH_CALUDE_spinster_cat_difference_l3329_332941


namespace NUMINAMATH_CALUDE_geometric_sequence_specific_form_l3329_332928

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- The theorem states that for a geometric sequence satisfying certain conditions,
    its general term has a specific form. -/
theorem geometric_sequence_specific_form (a : ℕ → ℝ) 
    (h_geo : GeometricSequence a) 
    (h_a2 : a 2 = 1)
    (h_relation : a 3 * a 5 = 2 * a 7) :
    ∀ n : ℕ, a n = 1 / 2^(n - 2) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_specific_form_l3329_332928


namespace NUMINAMATH_CALUDE_tank_water_supply_l3329_332970

theorem tank_water_supply (C V : ℝ) 
  (h1 : C = 15 * (V + 10))
  (h2 : C = 12 * (V + 20)) :
  C / V = 20 := by
sorry

end NUMINAMATH_CALUDE_tank_water_supply_l3329_332970


namespace NUMINAMATH_CALUDE_parabola_circle_theorem_l3329_332945

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define point E
def E : ℝ × ℝ := (2, 2)

-- Define line l
def line_l (y : ℝ) : ℝ := 2 * y + 2

-- Define points A and B on the parabola and line l
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define points M and N
def M : ℝ × ℝ := sorry
def N : ℝ × ℝ := sorry

-- Origin
def O : ℝ × ℝ := (0, 0)

theorem parabola_circle_theorem :
  parabola E.1 E.2 →
  parabola A.1 A.2 →
  parabola B.1 B.2 →
  A.1 = line_l A.2 →
  B.1 = line_l B.2 →
  A ≠ E →
  B ≠ E →
  M.2 = -2 →
  N.2 = -2 →
  (∃ t : ℝ, M = (1 - t) • E + t • A) →
  (∃ s : ℝ, N = (1 - s) • E + s • B) →
  (O.1 - M.1) * (O.1 - N.1) + (O.2 - M.2) * (O.2 - N.2) = 0 :=
sorry

end NUMINAMATH_CALUDE_parabola_circle_theorem_l3329_332945


namespace NUMINAMATH_CALUDE_floor_equation_solutions_l3329_332942

theorem floor_equation_solutions :
  let S := {x : ℤ | ⌊(x : ℚ) / 2⌋ + ⌊(x : ℚ) / 4⌋ = x}
  S = {0, -3, -2, -5} := by
  sorry

end NUMINAMATH_CALUDE_floor_equation_solutions_l3329_332942


namespace NUMINAMATH_CALUDE_combination_problem_l3329_332917

theorem combination_problem (n : ℕ) 
  (h : Nat.choose (n + 1) 7 - Nat.choose n 7 = Nat.choose n 8) : n = 14 := by
  sorry

end NUMINAMATH_CALUDE_combination_problem_l3329_332917


namespace NUMINAMATH_CALUDE_coin_authenticity_test_l3329_332988

/-- Represents the type of coin -/
inductive CoinType
| Real
| Fake

/-- Represents the weight difference between real and fake coins -/
def weightDifference : ℤ := 1

/-- Represents the total number of coins -/
def totalCoins (n : ℕ) : ℕ := 2 * n + 1

/-- Represents the number of fake coins -/
def fakeCoins (k : ℕ) : ℕ := 2 * k

/-- Represents the scale reading when weighing n coins against n coins -/
def scaleReading (n k₁ k₂ : ℕ) : ℤ := (k₁ : ℤ) - (k₂ : ℤ)

/-- Main theorem: The parity of the scale reading determines the type of the chosen coin -/
theorem coin_authenticity_test (n k : ℕ) (h : k ≤ n) :
  ∀ (chosenCoin : CoinType) (k₁ k₂ : ℕ) (h₁ : k₁ + k₂ = fakeCoins k - 1),
    chosenCoin = CoinType.Fake ↔ scaleReading n k₁ k₂ % 2 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_coin_authenticity_test_l3329_332988


namespace NUMINAMATH_CALUDE_angle_CED_measure_l3329_332963

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the circles
def circle1 : Set (ℝ × ℝ) := sorry
def circle2 : Set (ℝ × ℝ) := sorry

-- State the conditions
axiom circles_congruent : circle1 = circle2
axiom A_center_circle1 : A ∈ circle1
axiom B_center_circle2 : B ∈ circle2
axiom B_on_circle1 : B ∈ circle1
axiom A_on_circle2 : A ∈ circle2
axiom C_on_line_AB : sorry
axiom D_on_line_AB : sorry
axiom E_intersection : E ∈ circle1 ∩ circle2

-- Define the angle CED
def angle_CED : ℝ := sorry

-- Theorem to prove
theorem angle_CED_measure : angle_CED = 120 := by sorry

end NUMINAMATH_CALUDE_angle_CED_measure_l3329_332963


namespace NUMINAMATH_CALUDE_conic_sections_eccentricity_l3329_332934

theorem conic_sections_eccentricity (x : ℝ) : 
  (2 * x^2 - 5 * x + 2 = 0) →
  (x = 2 ∨ x = 1/2) ∧ 
  ((0 < x ∧ x < 1) ∨ x > 1) :=
sorry

end NUMINAMATH_CALUDE_conic_sections_eccentricity_l3329_332934


namespace NUMINAMATH_CALUDE_amy_hourly_rate_l3329_332919

/-- Calculates the hourly rate given total earnings, hours worked, and tips received. -/
def hourly_rate (total_earnings hours_worked tips : ℚ) : ℚ :=
  (total_earnings - tips) / hours_worked

/-- Proves that Amy's hourly rate is $2, given the conditions from the problem. -/
theorem amy_hourly_rate :
  let total_earnings : ℚ := 23
  let hours_worked : ℚ := 7
  let tips : ℚ := 9
  hourly_rate total_earnings hours_worked tips = 2 := by
  sorry

end NUMINAMATH_CALUDE_amy_hourly_rate_l3329_332919


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_one_l3329_332973

theorem at_least_one_greater_than_one (a b : ℝ) :
  a + b > 2 → max a b > 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_one_l3329_332973


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3329_332958

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (6 * x₁^2 + 5 * x₁ - 11 = 0) → 
  (6 * x₂^2 + 5 * x₂ - 11 = 0) → 
  (x₁ ≠ x₂) →
  x₁^2 + x₂^2 = 157 / 36 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3329_332958


namespace NUMINAMATH_CALUDE_game_price_is_correct_l3329_332912

/-- The price of each game Zachary sold -/
def game_price : ℝ := 5

/-- The number of games Zachary sold -/
def zachary_games : ℕ := 40

/-- The amount of money Zachary received -/
def zachary_amount : ℝ := game_price * zachary_games

/-- The amount of money Jason received -/
def jason_amount : ℝ := zachary_amount * 1.3

/-- The amount of money Ryan received -/
def ryan_amount : ℝ := jason_amount + 50

/-- The total amount received by all three friends -/
def total_amount : ℝ := 770

theorem game_price_is_correct : 
  zachary_amount + jason_amount + ryan_amount = total_amount := by sorry

end NUMINAMATH_CALUDE_game_price_is_correct_l3329_332912


namespace NUMINAMATH_CALUDE_original_purchase_price_l3329_332937

/-- Represents the original purchase price of the pants -/
def purchase_price : ℝ := sorry

/-- Represents the original selling price of the pants -/
def selling_price : ℝ := sorry

/-- The markup is 25% of the selling price -/
axiom markup_condition : selling_price = purchase_price + 0.25 * selling_price

/-- The new selling price after 20% decrease -/
def new_selling_price : ℝ := 0.8 * selling_price

/-- The gross profit is $5.40 -/
axiom gross_profit_condition : new_selling_price - purchase_price = 5.40

/-- Theorem stating that the original purchase price is $81 -/
theorem original_purchase_price : purchase_price = 81 := by sorry

end NUMINAMATH_CALUDE_original_purchase_price_l3329_332937


namespace NUMINAMATH_CALUDE_rope_cutting_problem_l3329_332999

theorem rope_cutting_problem :
  Nat.gcd 48 (Nat.gcd 72 (Nat.gcd 96 120)) = 24 := by sorry

end NUMINAMATH_CALUDE_rope_cutting_problem_l3329_332999


namespace NUMINAMATH_CALUDE_majka_numbers_unique_l3329_332960

/-- A three-digit number with alternating odd-even-odd digits -/
structure FunnyNumber :=
  (hundreds : Nat) (tens : Nat) (ones : Nat)
  (hundreds_odd : Odd hundreds)
  (tens_even : Even tens)
  (ones_odd : Odd ones)
  (is_three_digit : 100 ≤ hundreds * 100 + tens * 10 + ones ∧ hundreds * 100 + tens * 10 + ones < 1000)

/-- A three-digit number with alternating even-odd-even digits -/
structure CheerfulNumber :=
  (hundreds : Nat) (tens : Nat) (ones : Nat)
  (hundreds_even : Even hundreds)
  (tens_odd : Odd tens)
  (ones_even : Even ones)
  (is_three_digit : 100 ≤ hundreds * 100 + tens * 10 + ones ∧ hundreds * 100 + tens * 10 + ones < 1000)

/-- Convert a FunnyNumber to a natural number -/
def FunnyNumber.toNat (n : FunnyNumber) : Nat :=
  n.hundreds * 100 + n.tens * 10 + n.ones

/-- Convert a CheerfulNumber to a natural number -/
def CheerfulNumber.toNat (n : CheerfulNumber) : Nat :=
  n.hundreds * 100 + n.tens * 10 + n.ones

/-- The main theorem stating the unique solution to Majka's problem -/
theorem majka_numbers_unique (f : FunnyNumber) (c : CheerfulNumber) 
  (sum_eq : f.toNat + c.toNat = 1617)
  (product_ends_40 : (f.toNat * c.toNat) % 100 = 40)
  (all_digits_different : f.hundreds ≠ f.tens ∧ f.hundreds ≠ f.ones ∧ f.tens ≠ f.ones ∧
                          c.hundreds ≠ c.tens ∧ c.hundreds ≠ c.ones ∧ c.tens ≠ c.ones ∧
                          f.hundreds ≠ c.hundreds ∧ f.hundreds ≠ c.tens ∧ f.hundreds ≠ c.ones ∧
                          f.tens ≠ c.hundreds ∧ f.tens ≠ c.tens ∧ f.tens ≠ c.ones ∧
                          f.ones ≠ c.hundreds ∧ f.ones ≠ c.tens ∧ f.ones ≠ c.ones)
  (all_digits_nonzero : f.hundreds ≠ 0 ∧ f.tens ≠ 0 ∧ f.ones ≠ 0 ∧
                        c.hundreds ≠ 0 ∧ c.tens ≠ 0 ∧ c.ones ≠ 0) :
  f.toNat = 945 ∧ c.toNat = 672 ∧ f.toNat * c.toNat = 635040 := by
  sorry


end NUMINAMATH_CALUDE_majka_numbers_unique_l3329_332960


namespace NUMINAMATH_CALUDE_special_polygon_diagonals_l3329_332954

/-- A polygon with 10 vertices, where 4 vertices lie on a straight line
    and the remaining 6 form a regular hexagon. -/
structure SpecialPolygon where
  vertices : Fin 10
  line_vertices : Fin 4
  hexagon_vertices : Fin 6

/-- The number of diagonals in the special polygon. -/
def num_diagonals (p : SpecialPolygon) : ℕ := 33

/-- Theorem stating that the number of diagonals in the special polygon is 33. -/
theorem special_polygon_diagonals (p : SpecialPolygon) : num_diagonals p = 33 := by
  sorry

end NUMINAMATH_CALUDE_special_polygon_diagonals_l3329_332954


namespace NUMINAMATH_CALUDE_age_difference_l3329_332921

/-- Given three people a, b, and c, where b is twice as old as c, 
    the total of their ages is 12, and b is 4 years old, 
    prove that a is 2 years older than b. -/
theorem age_difference (a b c : ℕ) : 
  b = 2 * c →
  a + b + c = 12 →
  b = 4 →
  a = b + 2 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l3329_332921


namespace NUMINAMATH_CALUDE_opposite_of_neg_three_l3329_332930

-- Define the concept of opposite for real numbers
def opposite (x : ℝ) : ℝ := -x

-- Theorem stating that the opposite of -3 is 3
theorem opposite_of_neg_three : opposite (-3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_neg_three_l3329_332930


namespace NUMINAMATH_CALUDE_another_square_possible_l3329_332950

/-- Represents a grid of cells -/
structure Grid :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a square cut out from the grid -/
structure Square :=
  (size : Nat)

/-- Function to check if another square can be cut out -/
def can_cut_another_square (g : Grid) (cut_squares : List Square) : Prop :=
  ∃ (new_square : Square), 
    new_square.size = 2 ∧ 
    (g.rows ≥ 2 ∧ g.cols ≥ 2) ∧
    (List.length cut_squares < (g.rows / 2) * (g.cols / 2))

/-- Theorem statement -/
theorem another_square_possible (g : Grid) (cut_squares : List Square) :
  g.rows = 29 ∧ g.cols = 29 ∧ 
  List.length cut_squares = 99 ∧
  ∀ s ∈ cut_squares, s.size = 2
  →
  can_cut_another_square g cut_squares :=
sorry

end NUMINAMATH_CALUDE_another_square_possible_l3329_332950


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3329_332975

theorem simplify_and_evaluate :
  let x : ℚ := -1/3
  let a : ℤ := -2
  let b : ℤ := -1
  (6*x^2 + 5*x^2 - 2*(3*x - 2*x^2) = 11/3) ∧
  (5*a^2 - a*b - 2*(3*a*b - (a*b - 2*a^2)) = -6) := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3329_332975


namespace NUMINAMATH_CALUDE_percentage_of_120_to_40_l3329_332905

theorem percentage_of_120_to_40 : ∀ (x y : ℝ), x = 120 ∧ y = 40 → (x / y) * 100 = 300 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_120_to_40_l3329_332905


namespace NUMINAMATH_CALUDE_smallest_equal_partition_is_seven_l3329_332980

/-- The sum of squares from 1 to n -/
def sumOfSquares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- Checks if there exists a subset of squares that sum to half the total sum -/
def existsEqualPartition (n : ℕ) : Prop :=
  ∃ (subset : Finset ℕ), subset ⊆ Finset.range n ∧ 
    subset.sum (λ i => (i + 1)^2) = sumOfSquares n / 2

/-- The smallest n for which an equal partition exists -/
def smallestEqualPartition : ℕ := 7

theorem smallest_equal_partition_is_seven :
  (smallestEqualPartition = 7) ∧ 
  (existsEqualPartition 7) ∧ 
  (∀ k < 7, ¬ existsEqualPartition k) :=
sorry

end NUMINAMATH_CALUDE_smallest_equal_partition_is_seven_l3329_332980


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l3329_332990

/-- Evaluates |3-7i| + |3+7i| - arg(3+7i) -/
theorem complex_expression_evaluation :
  let z₁ : ℂ := 3 - 7*I
  let z₂ : ℂ := 3 + 7*I
  Complex.abs z₁ + Complex.abs z₂ - Complex.arg z₂ = 2 * Real.sqrt 58 - Real.arctan (7/3) := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l3329_332990


namespace NUMINAMATH_CALUDE_max_y_value_l3329_332953

theorem max_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -2) : 
  ∃ (max_y : ℤ), (∀ (y' : ℤ), ∃ (x' : ℤ), x' * y' + 3 * x' + 2 * y' = -2 → y' ≤ max_y) ∧ max_y = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_y_value_l3329_332953


namespace NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l3329_332991

theorem smallest_integer_with_given_remainders : ∃ b : ℕ+, 
  (b : ℕ) % 4 = 3 ∧ 
  (b : ℕ) % 6 = 5 ∧ 
  ∀ k : ℕ+, (k : ℕ) % 4 = 3 ∧ (k : ℕ) % 6 = 5 → k ≥ b :=
by
  use 23
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l3329_332991


namespace NUMINAMATH_CALUDE_new_jasmine_concentration_l3329_332994

/-- Calculates the new jasmine concentration after adding pure jasmine and water to a solution -/
theorem new_jasmine_concentration
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (added_jasmine : ℝ)
  (added_water : ℝ)
  (h1 : initial_volume = 80)
  (h2 : initial_concentration = 0.1)
  (h3 : added_jasmine = 5)
  (h4 : added_water = 15) :
  let initial_jasmine := initial_volume * initial_concentration
  let new_jasmine := initial_jasmine + added_jasmine
  let new_volume := initial_volume + added_jasmine + added_water
  new_jasmine / new_volume = 0.13 :=
by sorry

end NUMINAMATH_CALUDE_new_jasmine_concentration_l3329_332994


namespace NUMINAMATH_CALUDE_custom_operation_result_l3329_332935

-- Define the custom operation *
def star (a b : ℝ) : ℝ := a^2 - a*b

-- State the theorem
theorem custom_operation_result : star (star (-1) 2) 3 = 0 := by sorry

end NUMINAMATH_CALUDE_custom_operation_result_l3329_332935


namespace NUMINAMATH_CALUDE_smallest_configuration_l3329_332997

/-- A configuration of points on a plane where each point is 1 unit away from exactly four others -/
structure PointConfiguration where
  n : ℕ
  points : Fin n → ℝ × ℝ
  distinct : ∀ i j, i ≠ j → points i ≠ points j
  distance_condition : ∀ i, ∃ s : Finset (Fin n), s.card = 4 ∧ 
    ∀ j ∈ s, (i ≠ j) ∧ Real.sqrt (((points i).1 - (points j).1)^2 + ((points i).2 - (points j).2)^2) = 1

/-- The smallest possible number of points in a valid configuration is 9 -/
theorem smallest_configuration : 
  (∃ c : PointConfiguration, c.n = 9) ∧ 
  (∀ c : PointConfiguration, c.n ≥ 9) :=
sorry

end NUMINAMATH_CALUDE_smallest_configuration_l3329_332997


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3329_332913

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, a > |b| → a^3 > b^3) ∧
  (∃ a b : ℝ, a^3 > b^3 ∧ a ≤ |b|) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3329_332913


namespace NUMINAMATH_CALUDE_ellipse_focal_length_l3329_332989

/-- An ellipse with equation x²/m + y²/5 = 1 and focal length 2 has m = 4 -/
theorem ellipse_focal_length (m : ℝ) : 
  (∀ x y : ℝ, x^2 / m + y^2 / 5 = 1) →  -- Ellipse equation
  2 = 2 * (Real.sqrt (5 - m)) →         -- Focal length is 2
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_ellipse_focal_length_l3329_332989


namespace NUMINAMATH_CALUDE_variable_value_l3329_332959

theorem variable_value (w x v : ℝ) 
  (h1 : 2 / w + 2 / x = 2 / v) 
  (h2 : w * x = v) 
  (h3 : (w + x) / 2 = 0.5) : 
  v = 0.25 := by
sorry

end NUMINAMATH_CALUDE_variable_value_l3329_332959


namespace NUMINAMATH_CALUDE_frequency_converges_to_half_l3329_332996

/-- A fair coin toss experiment -/
structure CoinTossExperiment where
  n : ℕ  -- number of tosses
  m : ℕ  -- number of heads
  h_m_le_n : m ≤ n  -- m cannot exceed n

/-- The frequency of heads in a coin toss experiment -/
def frequency (e : CoinTossExperiment) : ℚ :=
  e.m / e.n

/-- The theoretical probability of heads for a fair coin -/
def fairCoinProbability : ℚ := 1 / 2

/-- The main theorem: as n approaches infinity, the frequency converges to 1/2 -/
theorem frequency_converges_to_half :
  ∀ ε > 0, ∃ N, ∀ e : CoinTossExperiment, e.n ≥ N →
    |frequency e - fairCoinProbability| < ε :=
sorry

end NUMINAMATH_CALUDE_frequency_converges_to_half_l3329_332996


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3329_332940

theorem sufficient_not_necessary (a : ℝ) :
  (a < -1 → ∃ x₀ : ℝ, a * Real.cos x₀ + 1 < 0) ∧
  (∃ a : ℝ, a ≥ -1 ∧ ∃ x₀ : ℝ, a * Real.cos x₀ + 1 < 0) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3329_332940


namespace NUMINAMATH_CALUDE_equation_solution_l3329_332981

theorem equation_solution (x : ℝ) : (x + 1) ^ (x + 3) = 1 ↔ x = -3 ∨ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3329_332981


namespace NUMINAMATH_CALUDE_accommodation_arrangements_theorem_l3329_332956

/-- The number of ways to arrange 5 people in 3 rooms with constraints -/
def accommodationArrangements (n : ℕ) (r : ℕ) (maxPerRoom : ℕ) : ℕ :=
  sorry

/-- The number of ways to arrange 5 people in 3 rooms with A and B not sharing -/
def accommodationArrangementsWithConstraint (n : ℕ) (r : ℕ) (maxPerRoom : ℕ) : ℕ :=
  sorry

theorem accommodation_arrangements_theorem :
  accommodationArrangementsWithConstraint 5 3 2 = 72 :=
sorry

end NUMINAMATH_CALUDE_accommodation_arrangements_theorem_l3329_332956


namespace NUMINAMATH_CALUDE_product_calculation_l3329_332903

theorem product_calculation : 200 * 19.9 * 1.99 * 100 = 791620 := by
  sorry

end NUMINAMATH_CALUDE_product_calculation_l3329_332903


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3329_332926

theorem quadratic_roots_property (x₁ x₂ : ℝ) : 
  (x₁^2 + x₁ - 3 = 0) → 
  (x₂^2 + x₂ - 3 = 0) → 
  (x₁^3 - 4*x₂^2 + 19 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3329_332926
