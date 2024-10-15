import Mathlib

namespace NUMINAMATH_CALUDE_sequence_property_l3442_344296

def sequence_condition (a : ℕ → ℝ) (m r : ℝ) : Prop :=
  a 1 = m ∧
  (∀ k : ℕ, a (2*k) = 2 * a (2*k - 1)) ∧
  (∀ k : ℕ, a (2*k + 1) = a (2*k) + r) ∧
  (∀ n : ℕ, n > 0 → a (n + 2) = a n)

theorem sequence_property (a : ℕ → ℝ) (m r : ℝ) 
  (h : sequence_condition a m r) : m + r = 0 := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l3442_344296


namespace NUMINAMATH_CALUDE_two_digit_product_digits_l3442_344253

theorem two_digit_product_digits (a b : ℕ) (ha : 40 < a ∧ a < 100) (hb : 40 < b ∧ b < 100) :
  (1000 ≤ a * b ∧ a * b < 10000) ∨ (100 ≤ a * b ∧ a * b < 1000) :=
sorry

end NUMINAMATH_CALUDE_two_digit_product_digits_l3442_344253


namespace NUMINAMATH_CALUDE_max_value_theorem_max_value_achievable_l3442_344284

theorem max_value_theorem (x y : ℝ) :
  (3 * x + 4 * y + 6) / Real.sqrt (x^2 + 4 * y^2 + 4) ≤ Real.sqrt 61 :=
by sorry

theorem max_value_achievable :
  ∃ x y : ℝ, (3 * x + 4 * y + 6) / Real.sqrt (x^2 + 4 * y^2 + 4) = Real.sqrt 61 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_max_value_achievable_l3442_344284


namespace NUMINAMATH_CALUDE_self_inverse_matrix_l3442_344203

def A (c d : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![4, -2; c, d]

theorem self_inverse_matrix (c d : ℚ) :
  A c d * A c d = 1 → c = 15/2 ∧ d = -4 := by
  sorry

end NUMINAMATH_CALUDE_self_inverse_matrix_l3442_344203


namespace NUMINAMATH_CALUDE_negation_of_absolute_value_nonnegative_l3442_344288

theorem negation_of_absolute_value_nonnegative :
  (¬ ∀ x : ℝ, |x| ≥ 0) ↔ (∃ x : ℝ, |x| < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_absolute_value_nonnegative_l3442_344288


namespace NUMINAMATH_CALUDE_cos_neg_570_deg_l3442_344210

theorem cos_neg_570_deg : Real.cos ((-570 : ℝ) * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_neg_570_deg_l3442_344210


namespace NUMINAMATH_CALUDE_assignment_plans_l3442_344221

theorem assignment_plans (n_females : ℕ) (n_males : ℕ) (n_positions : ℕ) 
  (h_females : n_females = 10)
  (h_males : n_males = 40)
  (h_positions : n_positions = 5) :
  (n_females.choose 2) * 3 * 24 * (n_males.choose 3) = 
    Nat.choose n_females 2 * (Nat.factorial 3 / Nat.factorial 2) * 
    (Nat.factorial 4 / Nat.factorial 0) * Nat.choose n_males 3 :=
by sorry

end NUMINAMATH_CALUDE_assignment_plans_l3442_344221


namespace NUMINAMATH_CALUDE_inequality_proof_l3442_344293

theorem inequality_proof (a₁ a₂ a₃ S : ℝ) 
  (h₁ : a₁ > 1) (h₂ : a₂ > 1) (h₃ : a₃ > 1)
  (hS : S = a₁ + a₂ + a₃)
  (hₐ₁ : a₁^2 / (a₁ - 1) > S)
  (hₐ₂ : a₂^2 / (a₂ - 1) > S)
  (hₐ₃ : a₃^2 / (a₃ - 1) > S) :
  1 / (a₁ + a₂) + 1 / (a₂ + a₃) + 1 / (a₃ + a₁) > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3442_344293


namespace NUMINAMATH_CALUDE_skittle_groups_l3442_344204

/-- The number of groups formed when dividing Skittles into equal-sized groups -/
def number_of_groups (total_skittles : ℕ) (skittles_per_group : ℕ) : ℕ :=
  total_skittles / skittles_per_group

/-- Theorem stating that dividing 5929 Skittles into groups of 77 results in 77 groups -/
theorem skittle_groups : number_of_groups 5929 77 = 77 := by
  sorry

end NUMINAMATH_CALUDE_skittle_groups_l3442_344204


namespace NUMINAMATH_CALUDE_salary_expenditure_l3442_344292

theorem salary_expenditure (salary : ℝ) (rent_fraction : ℝ) (clothes_fraction : ℝ) (remaining : ℝ) 
  (h1 : salary = 170000)
  (h2 : rent_fraction = 1/10)
  (h3 : clothes_fraction = 3/5)
  (h4 : remaining = 17000)
  (h5 : remaining / salary + rent_fraction + clothes_fraction < 1) :
  let food_fraction := 1 - (remaining / salary + rent_fraction + clothes_fraction)
  food_fraction = 1/5 := by
sorry

end NUMINAMATH_CALUDE_salary_expenditure_l3442_344292


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3442_344228

theorem fraction_sum_equality (p q r : ℝ) 
  (h : p / (30 - p) + q / (75 - q) + r / (45 - r) = 8) :
  6 / (30 - p) + 15 / (75 - q) + 9 / (45 - r) = 11 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3442_344228


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3442_344218

theorem inequality_solution_set (a : ℝ) (x₁ x₂ : ℝ) : 
  a > 0 → 
  (∀ x, x^2 - 2*a*x - 3*a^2 < 0 ↔ x₁ < x ∧ x < x₂) → 
  |x₁ - x₂| = 8 → 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3442_344218


namespace NUMINAMATH_CALUDE_range_of_b_l3442_344261

theorem range_of_b (a b c : ℝ) (h1 : a * c = b^2) (h2 : a + b + c = 3) :
  -3 ≤ b ∧ b ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_b_l3442_344261


namespace NUMINAMATH_CALUDE_candy_bars_problem_l3442_344226

theorem candy_bars_problem (fred : ℕ) (bob : ℕ) (jacqueline : ℕ) : 
  fred = 12 →
  bob = fred + 6 →
  jacqueline = 10 * (fred + bob) →
  (40 : ℝ) / 100 * jacqueline = 120 := by
  sorry

end NUMINAMATH_CALUDE_candy_bars_problem_l3442_344226


namespace NUMINAMATH_CALUDE_race_heartbeats_l3442_344216

/-- Calculates the total number of heartbeats during a race -/
def total_heartbeats (initial_rate : ℕ) (rate_increase : ℕ) (distance : ℕ) (pace : ℕ) : ℕ :=
  let final_rate := initial_rate + (distance - 1) * rate_increase
  let avg_rate := (initial_rate + final_rate) / 2
  avg_rate * distance * pace

/-- Theorem stating that the total heartbeats for the given conditions is 9750 -/
theorem race_heartbeats :
  total_heartbeats 140 5 10 6 = 9750 := by
  sorry

#eval total_heartbeats 140 5 10 6

end NUMINAMATH_CALUDE_race_heartbeats_l3442_344216


namespace NUMINAMATH_CALUDE_gcf_of_75_and_100_l3442_344245

theorem gcf_of_75_and_100 : Nat.gcd 75 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_75_and_100_l3442_344245


namespace NUMINAMATH_CALUDE_power_multiplication_l3442_344252

theorem power_multiplication (t : ℝ) : t^3 * t^4 = t^7 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3442_344252


namespace NUMINAMATH_CALUDE_equation_solution_l3442_344233

theorem equation_solution (x : ℝ) : 
  (1 - 2 * Real.sin (x / 2) * Real.cos (x / 2) = 
   (Real.sin (x / 2) - Real.cos (x / 2)) / Real.cos (x / 2)) ↔ 
  (∃ k : ℤ, x = π / 2 + 2 * k * π) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3442_344233


namespace NUMINAMATH_CALUDE_divisibility_by_six_l3442_344268

theorem divisibility_by_six (m n : ℤ) 
  (h1 : ∃ x y : ℤ, x^2 + m*x - n = 0 ∧ y^2 + m*y - n = 0)
  (h2 : ∃ x y : ℤ, x^2 - m*x + n = 0 ∧ y^2 - m*y + n = 0) : 
  6 ∣ n :=
sorry

end NUMINAMATH_CALUDE_divisibility_by_six_l3442_344268


namespace NUMINAMATH_CALUDE_seed_germination_percentage_l3442_344254

/-- Given an agricultural experiment with two plots of seeds, calculate the percentage of total seeds that germinated. -/
theorem seed_germination_percentage 
  (seeds_plot1 : ℕ) 
  (seeds_plot2 : ℕ) 
  (germination_rate_plot1 : ℚ) 
  (germination_rate_plot2 : ℚ) 
  (h1 : seeds_plot1 = 300)
  (h2 : seeds_plot2 = 200)
  (h3 : germination_rate_plot1 = 25 / 100)
  (h4 : germination_rate_plot2 = 40 / 100) :
  (((seeds_plot1 : ℚ) * germination_rate_plot1 + (seeds_plot2 : ℚ) * germination_rate_plot2) / 
   ((seeds_plot1 : ℚ) + (seeds_plot2 : ℚ))) = 31 / 100 := by
  sorry

end NUMINAMATH_CALUDE_seed_germination_percentage_l3442_344254


namespace NUMINAMATH_CALUDE_adams_purchase_cost_l3442_344282

/-- The cost of Adam's purchases given the quantities and prices of nuts and dried fruits -/
theorem adams_purchase_cost (nuts_quantity : ℝ) (nuts_price : ℝ) (fruits_quantity : ℝ) (fruits_price : ℝ) 
  (h1 : nuts_quantity = 3)
  (h2 : nuts_price = 12)
  (h3 : fruits_quantity = 2.5)
  (h4 : fruits_price = 8) :
  nuts_quantity * nuts_price + fruits_quantity * fruits_price = 56 := by
  sorry

end NUMINAMATH_CALUDE_adams_purchase_cost_l3442_344282


namespace NUMINAMATH_CALUDE_sculpture_cost_in_yen_l3442_344258

/-- Exchange rate from US dollars to Namibian dollars -/
def usd_to_nad : ℚ := 8

/-- Exchange rate from US dollars to Japanese yen -/
def usd_to_jpy : ℚ := 110

/-- Cost of the sculpture in Namibian dollars -/
def sculpture_cost_nad : ℚ := 136

/-- Theorem stating the cost of the sculpture in Japanese yen -/
theorem sculpture_cost_in_yen : 
  (sculpture_cost_nad / usd_to_nad) * usd_to_jpy = 1870 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_cost_in_yen_l3442_344258


namespace NUMINAMATH_CALUDE_no_real_roots_l3442_344242

def polynomial (x p : ℝ) : ℝ := x^4 + 4*p*x^3 + 6*x^2 + 4*p*x + 1

theorem no_real_roots (p : ℝ) : 
  (∀ x : ℝ, polynomial x p ≠ 0) ↔ p > -Real.sqrt 5 / 2 ∧ p < Real.sqrt 5 / 2 :=
sorry

end NUMINAMATH_CALUDE_no_real_roots_l3442_344242


namespace NUMINAMATH_CALUDE_hot_dogs_remainder_l3442_344224

theorem hot_dogs_remainder : 35867413 % 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hot_dogs_remainder_l3442_344224


namespace NUMINAMATH_CALUDE_triangle_division_2005_l3442_344217

theorem triangle_division_2005 : ∃ n : ℕ, n^2 + (2005 - n^2)^2 = 2005 := by
  sorry

end NUMINAMATH_CALUDE_triangle_division_2005_l3442_344217


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3442_344275

/-- An arithmetic sequence with sum formula S_n = n^2 - 3n has general term a_n = 2n - 4 -/
theorem arithmetic_sequence_general_term 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h_sum : ∀ n : ℕ, S n = n^2 - 3*n)
  (h_arithmetic : ∀ n : ℕ, a (n+1) - a n = a (n+2) - a (n+1))
  (h_relation : ∀ n : ℕ, n ≥ 1 → a n = S n - S (n-1))
  : ∀ n : ℕ, a n = 2*n - 4 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3442_344275


namespace NUMINAMATH_CALUDE_cubic_quadratic_comparison_quadratic_inequality_l3442_344289

-- Problem 1
theorem cubic_quadratic_comparison (x : ℝ) (h : x ≥ -1) :
  x^3 + 1 ≥ x^2 + x ∧ (x^3 + 1 = x^2 + x ↔ x = 1 ∨ x = -1) := by sorry

-- Problem 2
theorem quadratic_inequality (a x : ℝ) (h : a < 0) :
  x^2 - a*x - 6*a^2 > 0 ↔ x < 3*a ∨ x > -2*a := by sorry

end NUMINAMATH_CALUDE_cubic_quadratic_comparison_quadratic_inequality_l3442_344289


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3442_344295

/-- Given a right triangle, prove that if rotation about one leg produces a cone
    of volume 1620π cm³ and rotation about the other leg produces a cone
    of volume 3240π cm³, then the length of the hypotenuse is √507 cm. -/
theorem right_triangle_hypotenuse (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (1 / 3 * π * a * b^2 = 1620 * π) →
  (1 / 3 * π * b * a^2 = 3240 * π) →
  Real.sqrt (a^2 + b^2) = Real.sqrt 507 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3442_344295


namespace NUMINAMATH_CALUDE_factorization_x3_minus_9xy2_l3442_344276

theorem factorization_x3_minus_9xy2 (x y : ℝ) : 
  x^3 - 9*x*y^2 = x*(x+3*y)*(x-3*y) := by sorry

end NUMINAMATH_CALUDE_factorization_x3_minus_9xy2_l3442_344276


namespace NUMINAMATH_CALUDE_rhombus_side_length_l3442_344267

/-- Represents a rhombus with given diagonal and area -/
structure Rhombus where
  diagonal : ℝ
  area : ℝ

/-- Calculates the length of the side of a rhombus -/
def side_length (r : Rhombus) : ℝ :=
  sorry

theorem rhombus_side_length (r : Rhombus) (h1 : r.diagonal = 30) (h2 : r.area = 600) :
  side_length r = 25 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l3442_344267


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l3442_344251

theorem smallest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l3442_344251


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l3442_344291

theorem arithmetic_sequence_length (first last step : ℤ) (h : first ≥ last) : 
  (first - last) / step + 1 = (first - 44) / 4 + 1 → (first - 44) / 4 + 1 = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l3442_344291


namespace NUMINAMATH_CALUDE_emily_toys_left_l3442_344265

/-- The number of toys Emily started with -/
def initial_toys : ℕ := 7

/-- The number of toys Emily sold -/
def sold_toys : ℕ := 3

/-- The number of toys Emily has left -/
def remaining_toys : ℕ := initial_toys - sold_toys

/-- Theorem stating that Emily has 4 toys left -/
theorem emily_toys_left : remaining_toys = 4 := by
  sorry

end NUMINAMATH_CALUDE_emily_toys_left_l3442_344265


namespace NUMINAMATH_CALUDE_longest_chord_in_quarter_circle_l3442_344299

theorem longest_chord_in_quarter_circle (d : ℝ) (h : d = 16) : 
  let r := d / 2
  let chord_length := (2 * r ^ 2) ^ (1/2)
  chord_length ^ 2 = 128 :=
by sorry

end NUMINAMATH_CALUDE_longest_chord_in_quarter_circle_l3442_344299


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l3442_344232

theorem no_positive_integer_solutions :
  ¬ ∃ (x₁ x₂ : ℕ), 903 * x₁ + 731 * x₂ = 1106 := by
sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l3442_344232


namespace NUMINAMATH_CALUDE_safe_code_count_l3442_344278

/-- The set of digits from 0 to 9 -/
def Digits : Finset ℕ := Finset.range 10

/-- The length of the safe code -/
def CodeLength : ℕ := 4

/-- The set of forbidden first digits -/
def ForbiddenFirstDigits : Finset ℕ := {5, 7}

/-- The number of valid safe codes -/
def ValidCodes : ℕ := 10^CodeLength - ForbiddenFirstDigits.card * 10^(CodeLength - 1)

theorem safe_code_count : ValidCodes = 9900 := by
  sorry

end NUMINAMATH_CALUDE_safe_code_count_l3442_344278


namespace NUMINAMATH_CALUDE_only_14_satisfies_l3442_344246

def is_multiple_of_three (n : ℤ) : Prop := ∃ k : ℤ, n = 3 * k

def is_perfect_square (n : ℤ) : Prop := ∃ k : ℤ, n = k * k

def sum_of_digits (n : ℤ) : ℕ :=
  (n.natAbs.repr.toList.map (λ c => c.toNat - '0'.toNat)).sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, 1 < m → m < n → ¬(n % m = 0)

def satisfies_conditions (n : ℤ) : Prop :=
  ¬(is_multiple_of_three n) ∧
  ¬(is_perfect_square n) ∧
  is_prime (sum_of_digits n)

theorem only_14_satisfies :
  satisfies_conditions 14 ∧
  ¬(satisfies_conditions 12) ∧
  ¬(satisfies_conditions 16) ∧
  ¬(satisfies_conditions 21) ∧
  ¬(satisfies_conditions 26) :=
sorry

end NUMINAMATH_CALUDE_only_14_satisfies_l3442_344246


namespace NUMINAMATH_CALUDE_car_average_mpg_l3442_344205

def initial_odometer : ℕ := 57300
def final_odometer : ℕ := 58300
def initial_gas : ℕ := 8
def second_gas : ℕ := 14
def final_gas : ℕ := 22

def total_distance : ℕ := final_odometer - initial_odometer
def total_gas : ℕ := initial_gas + second_gas + final_gas

def average_mpg : ℚ := total_distance / total_gas

theorem car_average_mpg :
  (round (average_mpg * 10) / 10 : ℚ) = 227/10 := by sorry

end NUMINAMATH_CALUDE_car_average_mpg_l3442_344205


namespace NUMINAMATH_CALUDE_min_value_of_f_range_of_a_l3442_344208

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + a

-- Theorem for the minimum value of f when a = 2
theorem min_value_of_f (x : ℝ) (h : x ≥ 1) :
  f 2 x ≥ 5 :=
sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∀ x ≥ 1, f a x > 0) ↔ a > -3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_range_of_a_l3442_344208


namespace NUMINAMATH_CALUDE_constant_function_theorem_l3442_344219

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

/-- The property that a function satisfies the given functional equation -/
def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * floor y) = floor (f x) * f y

/-- The theorem stating that functions satisfying the equation are constant functions with values in [1, 2) -/
theorem constant_function_theorem (f : ℝ → ℝ) (h : satisfies_equation f) :
  ∃ c : ℝ, (∀ x : ℝ, f x = c) ∧ 1 ≤ c ∧ c < 2 := by
  sorry

end NUMINAMATH_CALUDE_constant_function_theorem_l3442_344219


namespace NUMINAMATH_CALUDE_ellipse_slope_product_l3442_344290

theorem ellipse_slope_product (x₁ y₁ x₂ y₂ : ℝ) :
  x₁ ≠ 0 →
  y₁ ≠ 0 →
  x₁^2 + 4*y₁^2/9 = 1 →
  x₂^2 + 4*y₂^2/9 = 1 →
  3*y₁/(4*x₁) = (y₁ + y₂)/(x₁ + x₂) →
  (y₁/x₁) * ((y₁ - y₂)/(x₁ - x₂)) = -1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_slope_product_l3442_344290


namespace NUMINAMATH_CALUDE_isosceles_triangle_dimensions_l3442_344248

/-- An isosceles triangle with base b and leg length l -/
structure IsoscelesTriangle where
  b : ℝ
  l : ℝ
  h : ℝ
  isPositive : 0 < b ∧ 0 < l ∧ 0 < h
  isIsosceles : l = b - 1
  areaRelation : (1/2) * b * h = (1/3) * b^2

/-- Theorem about the dimensions of a specific isosceles triangle -/
theorem isosceles_triangle_dimensions (t : IsoscelesTriangle) :
  t.b = 6 ∧ t.l = 5 ∧ t.h = 4 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_dimensions_l3442_344248


namespace NUMINAMATH_CALUDE_min_value_of_b_in_geometric_sequence_l3442_344259

theorem min_value_of_b_in_geometric_sequence (a b c : ℝ) : 
  (∃ r : ℝ, (a = b / r ∧ c = b * r) ∨ (a = b * r ∧ c = b / r)) →  -- geometric sequence condition
  ((a = 1 ∧ c = 4) ∨ (a = 4 ∧ c = 1) ∨ (a = 1 ∧ b = 4) ∨ (a = 4 ∧ b = 1) ∨ (b = 1 ∧ c = 4) ∨ (b = 4 ∧ c = 1)) →  -- 1 and 4 are in the sequence
  b ≥ -2 ∧ ∃ b₀ : ℝ, b₀ = -2 ∧ 
    (∃ r : ℝ, (b₀ = b₀ / r ∧ 4 = b₀ * r) ∨ (1 = b₀ * r ∧ 4 = b₀ / r)) ∧
    ((1 = 1 ∧ 4 = 4) ∨ (1 = 4 ∧ 4 = 1) ∨ (1 = 1 ∧ b₀ = 4) ∨ (1 = 4 ∧ b₀ = 1) ∨ (b₀ = 1 ∧ 4 = 4) ∨ (b₀ = 4 ∧ 4 = 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_of_b_in_geometric_sequence_l3442_344259


namespace NUMINAMATH_CALUDE_negation_truth_values_l3442_344262

theorem negation_truth_values :
  (¬ ∃ x : ℝ, x^2 + x + 1 ≤ 0) ∧
  (¬ ∀ x y : ℝ, Real.sqrt ((x - 1)^2) + (y + 1)^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_truth_values_l3442_344262


namespace NUMINAMATH_CALUDE_nth_equation_l3442_344225

theorem nth_equation (n : ℕ) (hn : n > 0) :
  1 / (n + 2 : ℚ) + 2 / (n^2 + 2*n : ℚ) = 1 / n :=
by sorry

end NUMINAMATH_CALUDE_nth_equation_l3442_344225


namespace NUMINAMATH_CALUDE_perimeter_of_similar_triangle_l3442_344286

/-- Represents a triangle with side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- Determines if two triangles are similar -/
def similar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ t2.a = k * t1.a ∧ t2.b = k * t1.b ∧ t2.c = k * t1.c

theorem perimeter_of_similar_triangle (abc pqr : Triangle) :
  abc.a = abc.b ∧ 
  abc.a = 12 ∧ 
  abc.c = 14 ∧ 
  similar abc pqr ∧
  max pqr.a (max pqr.b pqr.c) = 35 →
  perimeter pqr = 95 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_similar_triangle_l3442_344286


namespace NUMINAMATH_CALUDE_paper_tearing_impossibility_l3442_344250

theorem paper_tearing_impossibility : ¬ ∃ (n : ℕ), 1 + 2 * n = 100 := by
  sorry

end NUMINAMATH_CALUDE_paper_tearing_impossibility_l3442_344250


namespace NUMINAMATH_CALUDE_ellipse_condition_l3442_344269

/-- The equation of an ellipse -/
def ellipse_equation (x y b : ℝ) : Prop :=
  4 * x^2 + 9 * y^2 - 16 * x + 18 * y + 12 = b

/-- A non-degenerate ellipse condition -/
def is_non_degenerate_ellipse (b : ℝ) : Prop :=
  b > -13

/-- Theorem: The given equation represents a non-degenerate ellipse iff b > -13 -/
theorem ellipse_condition (b : ℝ) :
  (∃ x y : ℝ, ellipse_equation x y b) ↔ is_non_degenerate_ellipse b :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l3442_344269


namespace NUMINAMATH_CALUDE_mary_bill_difference_l3442_344281

/-- Represents the candy distribution problem --/
def candy_distribution (total : ℕ) (kate robert mary bill : ℕ) : Prop :=
  total = 20 ∧
  robert = kate + 2 ∧
  mary > bill ∧
  mary = robert + 2 ∧
  kate = bill + 2 ∧
  kate = 4

/-- Theorem stating the difference between Mary's and Bill's candy pieces --/
theorem mary_bill_difference (total kate robert mary bill : ℕ) 
  (h : candy_distribution total kate robert mary bill) : 
  mary - bill = 6 := by sorry

end NUMINAMATH_CALUDE_mary_bill_difference_l3442_344281


namespace NUMINAMATH_CALUDE_notebook_sales_plan_exists_l3442_344249

/-- Represents the sales data for a month -/
structure MonthSales where
  price : ℝ
  sales : ℝ

/-- Represents the notebook sales problem -/
structure NotebookSales where
  initial_inventory : ℕ
  purchase_price : ℝ
  min_sell_price : ℝ
  max_sell_price : ℝ
  july_oct_sales : List MonthSales
  price_sales_relation : ℝ → ℝ

/-- Represents a pricing plan for November and December -/
structure PricingPlan where
  nov_price : ℝ
  nov_sales : ℝ
  dec_price : ℝ
  dec_sales : ℝ

/-- Main theorem statement -/
theorem notebook_sales_plan_exists (problem : NotebookSales) :
  problem.initial_inventory = 550 ∧
  problem.purchase_price = 6 ∧
  problem.min_sell_price = 9 ∧
  problem.max_sell_price = 12 ∧
  problem.july_oct_sales = [⟨9, 115⟩, ⟨10, 100⟩, ⟨11, 85⟩, ⟨12, 70⟩] ∧
  (∀ x, problem.price_sales_relation x = -15 * x + 250) →
  ∃ (plan : PricingPlan),
    -- Remaining inventory after 4 months is 180
    (problem.initial_inventory - (problem.july_oct_sales.map (λ s => s.sales)).sum = 180) ∧
    -- Highest monthly profit in first 4 months is 425, occurring in September
    ((problem.july_oct_sales.map (λ s => (s.price - problem.purchase_price) * s.sales)).maximum = some 425) ∧
    -- Total sales profit for November and December is at least 800
    ((plan.nov_price - problem.purchase_price) * plan.nov_sales +
     (plan.dec_price - problem.purchase_price) * plan.dec_sales ≥ 800) ∧
    -- Pricing plan follows the price-sales relationship
    (problem.price_sales_relation plan.nov_price = plan.nov_sales ∧
     problem.price_sales_relation plan.dec_price = plan.dec_sales) ∧
    -- Prices are within the allowed range
    (plan.nov_price ≥ problem.min_sell_price ∧ plan.nov_price ≤ problem.max_sell_price ∧
     plan.dec_price ≥ problem.min_sell_price ∧ plan.dec_price ≤ problem.max_sell_price) :=
by sorry


end NUMINAMATH_CALUDE_notebook_sales_plan_exists_l3442_344249


namespace NUMINAMATH_CALUDE_complex_product_simplification_l3442_344230

theorem complex_product_simplification (a b x y : ℝ) : 
  (a * x + Complex.I * b * y) * (a * x - Complex.I * b * y) = a^2 * x^2 - b^2 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_simplification_l3442_344230


namespace NUMINAMATH_CALUDE_preimage_of_three_l3442_344234

def f (x : ℝ) : ℝ := 2 * x - 1

theorem preimage_of_three (x : ℝ) : f x = 3 ↔ x = 2 := by sorry

end NUMINAMATH_CALUDE_preimage_of_three_l3442_344234


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3442_344207

theorem arithmetic_calculation : 5 * 7.5 + 2 * 12 + 8.5 * 4 + 7 * 6 = 137.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3442_344207


namespace NUMINAMATH_CALUDE_units_digit_F_500_l3442_344294

-- Define the modified Fermat number function
def F (n : ℕ) : ℕ := 2^(2^(2*n)) + 1

-- Theorem statement
theorem units_digit_F_500 : F 500 % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_units_digit_F_500_l3442_344294


namespace NUMINAMATH_CALUDE_class_age_problem_l3442_344223

theorem class_age_problem (total_students : ℕ) (total_avg_age : ℝ) 
  (group_a_students : ℕ) (group_a_avg_age : ℝ)
  (group_b_students : ℕ) (group_b_avg_age : ℝ) :
  total_students = 50 →
  total_avg_age = 24 →
  group_a_students = 15 →
  group_a_avg_age = 20 →
  group_b_students = 25 →
  group_b_avg_age = 25 →
  let group_c_students := total_students - (group_a_students + group_b_students)
  let total_age := total_students * total_avg_age
  let group_a_total_age := group_a_students * group_a_avg_age
  let group_b_total_age := group_b_students * group_b_avg_age
  let group_c_total_age := total_age - (group_a_total_age + group_b_total_age)
  let group_c_avg_age := group_c_total_age / group_c_students
  group_c_avg_age = 27.5 := by
    sorry

end NUMINAMATH_CALUDE_class_age_problem_l3442_344223


namespace NUMINAMATH_CALUDE_candidate_votes_l3442_344279

theorem candidate_votes (total_votes : ℕ) (invalid_percentage : ℚ) (candidate_percentage : ℚ) :
  total_votes = 560000 →
  invalid_percentage = 15 / 100 →
  candidate_percentage = 70 / 100 →
  ∃ (valid_votes : ℕ) (candidate_votes : ℕ),
    valid_votes = (1 - invalid_percentage) * total_votes ∧
    candidate_votes = candidate_percentage * valid_votes ∧
    candidate_votes = 333200 := by
  sorry

end NUMINAMATH_CALUDE_candidate_votes_l3442_344279


namespace NUMINAMATH_CALUDE_complete_square_sum_l3442_344244

theorem complete_square_sum (a b c : ℤ) (h1 : a > 0) 
  (h2 : ∀ x : ℝ, 49 * x^2 + 56 * x - 64 = 0 ↔ (a * x + b)^2 = c) : 
  a + b + c = 91 := by
sorry

end NUMINAMATH_CALUDE_complete_square_sum_l3442_344244


namespace NUMINAMATH_CALUDE_train_length_l3442_344243

/-- Calculates the length of a train given its speed and the time it takes to pass through a tunnel of known length. -/
theorem train_length (train_speed : ℝ) (tunnel_length : ℝ) (time_to_pass : ℝ) : 
  train_speed = 54 * 1000 / 3600 →
  tunnel_length = 1200 →
  time_to_pass = 100 →
  (train_speed * time_to_pass) - tunnel_length = 300 := by
sorry

end NUMINAMATH_CALUDE_train_length_l3442_344243


namespace NUMINAMATH_CALUDE_rectangle_division_l3442_344238

/-- Represents a rectangle with side lengths a and b -/
structure Rectangle where
  a : ℝ
  b : ℝ

/-- Represents the large rectangle ABCD -/
def large_rectangle : Rectangle := { a := 18, b := 16 }

/-- Represents a small rectangle within ABCD -/
structure SmallRectangle where
  x : ℝ
  y : ℝ

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.a + r.b)

/-- The perimeter of a small rectangle -/
def small_perimeter (r : SmallRectangle) : ℝ := 2 * (r.x + r.y)

/-- The theorem to be proved -/
theorem rectangle_division (small1 small2 small3 small4 : SmallRectangle) :
  large_rectangle.a = 18 ∧ large_rectangle.b = 16 ∧
  small_perimeter small1 = small_perimeter small2 ∧
  small_perimeter small2 = small_perimeter small3 ∧
  small_perimeter small3 = small_perimeter small4 ∧
  small1.x + small2.x + small3.x = large_rectangle.a ∧
  small1.y + small2.y + small3.y + small4.y = large_rectangle.b →
  (small1.x = 2 ∧ small1.y = 18 ∧
   small2.x = 6 ∧ small2.y = 14 ∧
   small3.x = 6 ∧ small3.y = 14 ∧
   small4.x = 6 ∧ small4.y = 14) :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_division_l3442_344238


namespace NUMINAMATH_CALUDE_total_chickens_l3442_344239

def farm_animals (ducks rabbits : ℕ) : Prop :=
  ∃ (hens roosters chickens : ℕ),
    hens = ducks + 20 ∧
    roosters = rabbits - 10 ∧
    chickens = hens + roosters ∧
    chickens = 80

theorem total_chickens : farm_animals 40 30 := by
  sorry

end NUMINAMATH_CALUDE_total_chickens_l3442_344239


namespace NUMINAMATH_CALUDE_positive_expression_l3442_344273

theorem positive_expression (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 1) : 
  0 < b + 3 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_positive_expression_l3442_344273


namespace NUMINAMATH_CALUDE_track_length_satisfies_conditions_l3442_344270

/-- The length of a circular track satisfying the given conditions -/
def track_length : ℝ := 766.67

/-- Two runners on a circular track -/
structure Runners :=
  (track_length : ℝ)
  (initial_separation : ℝ)
  (first_meeting_distance : ℝ)
  (second_meeting_distance : ℝ)

/-- The conditions of the problem -/
def problem_conditions (r : Runners) : Prop :=
  r.initial_separation = 0.75 * r.track_length ∧
  r.first_meeting_distance = 120 ∧
  r.second_meeting_distance = 180

/-- The theorem stating that the track length satisfies the problem conditions -/
theorem track_length_satisfies_conditions :
  ∃ (r : Runners), r.track_length = track_length ∧ problem_conditions r :=
sorry

end NUMINAMATH_CALUDE_track_length_satisfies_conditions_l3442_344270


namespace NUMINAMATH_CALUDE_inequality_solutions_l3442_344214

theorem inequality_solutions :
  (∀ x : ℝ, x^2 + 3*x - 10 ≥ 0 ↔ x ≤ -5 ∨ x ≥ 2) ∧
  (∀ x : ℝ, x^2 - 3*x - 2 ≤ 0 ↔ (3 - Real.sqrt 17) / 2 ≤ x ∧ x ≤ (3 + Real.sqrt 17) / 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solutions_l3442_344214


namespace NUMINAMATH_CALUDE_average_weight_BCDE_l3442_344237

/-- Given the weights of individuals A, B, C, D, and E, prove that the average weight of B, C, D, and E is 97.25 kg. -/
theorem average_weight_BCDE (w_A w_B w_C w_D w_E : ℝ) : 
  w_A = 77 →
  (w_A + w_B + w_C) / 3 = 84 →
  (w_A + w_B + w_C + w_D) / 4 = 80 →
  w_E = w_D + 5 →
  (w_B + w_C + w_D + w_E) / 4 = 97.25 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_BCDE_l3442_344237


namespace NUMINAMATH_CALUDE_horner_method_example_l3442_344215

def f (x : ℝ) : ℝ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

theorem horner_method_example : f 2 = -80 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_example_l3442_344215


namespace NUMINAMATH_CALUDE_hexagon_area_in_circle_l3442_344255

/-- The area of a regular hexagon inscribed in a circle with radius 2 units is 6√3 square units. -/
theorem hexagon_area_in_circle (r : ℝ) (h : r = 2) : 
  let hexagon_area := 6 * (r^2 * Real.sqrt 3 / 4)
  hexagon_area = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_in_circle_l3442_344255


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l3442_344285

theorem triangle_angle_problem (A B C : Real) (a b c : Real) :
  (A + B + C = π) →
  (a * Real.cos B - b * Real.cos A = c) →
  (C = π / 5) →
  (B = 3 * π / 10) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l3442_344285


namespace NUMINAMATH_CALUDE_area_ratio_is_one_fourth_l3442_344231

/-- A square with vertices A, B, C, D -/
structure Square where
  side_length : ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- A particle moving along the edges of a square -/
structure Particle where
  position : ℝ → ℝ × ℝ  -- position as a function of time
  speed : ℝ

/-- The region enclosed by the path of the midpoint -/
def enclosed_region (p1 p2 : Particle) : Set (ℝ × ℝ) := sorry

/-- The area of a set in ℝ² -/
def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating the ratio of areas -/
theorem area_ratio_is_one_fourth (sq : Square) (p1 p2 : Particle) :
  sq.A = (0, 0) ∧ 
  sq.B = (sq.side_length, 0) ∧ 
  sq.C = (sq.side_length, sq.side_length) ∧ 
  sq.D = (0, sq.side_length) ∧
  p1.position 0 = sq.A ∧
  p2.position 0 = ((sq.C.1 + sq.D.1) / 2, sq.C.2) ∧
  p1.speed = p2.speed →
  area (enclosed_region p1 p2) / area {p | p.1 ∈ Set.Icc 0 sq.side_length ∧ p.2 ∈ Set.Icc 0 sq.side_length} = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_is_one_fourth_l3442_344231


namespace NUMINAMATH_CALUDE_number_symmetry_equation_l3442_344280

theorem number_symmetry_equation (a b : ℕ) (h : 2 ≤ a + b ∧ a + b ≤ 9) :
  (10 * a + b) * (100 * b + 10 * (a + b) + a) = (100 * a + 10 * (a + b) + b) * (10 * b + a) := by
  sorry

end NUMINAMATH_CALUDE_number_symmetry_equation_l3442_344280


namespace NUMINAMATH_CALUDE_sufficient_necessary_condition_l3442_344200

theorem sufficient_necessary_condition (a : ℝ) :
  (∀ x : ℝ, x > 0 → x + 1/x > a) ↔ a < 2 := by sorry

end NUMINAMATH_CALUDE_sufficient_necessary_condition_l3442_344200


namespace NUMINAMATH_CALUDE_inverse_proportion_doubling_l3442_344213

/-- Given two positive real numbers x and y that are inversely proportional,
    if x doubles, then y decreases by 50%. -/
theorem inverse_proportion_doubling (x y x' y' : ℝ) (k : ℝ) (hxy_pos : x > 0 ∧ y > 0) 
    (hk_pos : k > 0) (hxy : x * y = k) (hx'y' : x' * y' = k) (hx_double : x' = 2 * x) : 
    y' = y / 2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_doubling_l3442_344213


namespace NUMINAMATH_CALUDE_junior_basketball_league_bad_teams_l3442_344264

/-- Given a total of 18 teams in a junior basketball league, where half are rich,
    and there cannot be 10 teams that are both rich and bad,
    prove that the fraction of bad teams must be less than or equal to 1/2. -/
theorem junior_basketball_league_bad_teams
  (total_teams : ℕ)
  (rich_teams : ℕ)
  (bad_fraction : ℚ)
  (h1 : total_teams = 18)
  (h2 : rich_teams = total_teams / 2)
  (h3 : ¬(bad_fraction * ↑total_teams ≥ 10 ∧ bad_fraction * ↑total_teams ≤ ↑rich_teams)) :
  bad_fraction ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_junior_basketball_league_bad_teams_l3442_344264


namespace NUMINAMATH_CALUDE_diagonal_length_of_prism_l3442_344206

/-- 
Given a rectangular prism with dimensions x, y, and z,
if the projections of its diagonal on each plane (xy, xz, yz) have length √2,
then the length of the diagonal is √3.
-/
theorem diagonal_length_of_prism (x y z : ℝ) 
  (h1 : x^2 + y^2 = 2)
  (h2 : x^2 + z^2 = 2)
  (h3 : y^2 + z^2 = 2) :
  Real.sqrt (x^2 + y^2 + z^2) = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_diagonal_length_of_prism_l3442_344206


namespace NUMINAMATH_CALUDE_differential_equation_satisfaction_l3442_344287

open Real

theorem differential_equation_satisfaction (n : ℝ) (x : ℝ) (h : x ≠ -1) :
  let y : ℝ → ℝ := λ x => (x + 1)^n * (exp x - 1)
  deriv y x - (n * y x) / (x + 1) = exp x * (1 + x)^n := by
  sorry

end NUMINAMATH_CALUDE_differential_equation_satisfaction_l3442_344287


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l3442_344212

/-- The area of an isosceles right triangle with hypotenuse 6 is 9 -/
theorem isosceles_right_triangle_area (h : ℝ) (A : ℝ) : 
  h = 6 →  -- The hypotenuse is 6 units
  A = h^2 / 4 →  -- Area formula for isosceles right triangle
  A = 9 := by
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l3442_344212


namespace NUMINAMATH_CALUDE_no_real_solution_for_equation_and_convergence_l3442_344220

theorem no_real_solution_for_equation_and_convergence : 
  ¬∃ y : ℝ, y = 2 / (1 + y) ∧ abs y < 1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_for_equation_and_convergence_l3442_344220


namespace NUMINAMATH_CALUDE_no_natural_solution_l3442_344263

theorem no_natural_solution : ¬∃ (x y : ℕ), x^4 - y^4 = x^3 + y^3 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solution_l3442_344263


namespace NUMINAMATH_CALUDE_car_distance_traveled_l3442_344277

-- Define constants
def tire_diameter : ℝ := 15
def revolutions : ℝ := 672.1628045157456
def inches_per_mile : ℝ := 63360

-- Define the theorem
theorem car_distance_traveled (ε : ℝ) (h_ε : ε > 0) :
  ∃ (distance : ℝ), 
    abs (distance - 0.5) < ε ∧ 
    distance = (π * tire_diameter * revolutions) / inches_per_mile :=
sorry

end NUMINAMATH_CALUDE_car_distance_traveled_l3442_344277


namespace NUMINAMATH_CALUDE_characterization_of_n_l3442_344241

def has_finite_multiples_with_n_divisors (n : ℕ+) : Prop :=
  ∃ (S : Finset ℕ+), ∀ (k : ℕ+), (n ∣ k) → (Nat.card (Nat.divisors k) = n) → k ∈ S

def not_divisible_by_square_of_prime (n : ℕ+) : Prop :=
  ∀ (p : ℕ+), Nat.Prime p → (p * p ∣ n) → False

theorem characterization_of_n (n : ℕ+) :
  has_finite_multiples_with_n_divisors n ↔ not_divisible_by_square_of_prime n ∨ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_characterization_of_n_l3442_344241


namespace NUMINAMATH_CALUDE_evaluate_expression_l3442_344211

theorem evaluate_expression : (8^5 / 8^2) * 2^12 = 2^21 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3442_344211


namespace NUMINAMATH_CALUDE_sum_of_powers_l3442_344256

/-- Given two real numbers a and b satisfying certain conditions, 
    prove that a^10 + b^10 = 123 -/
theorem sum_of_powers (a b : ℝ) 
  (h1 : a = Real.sqrt 6)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^4 + b^4 = 7)
  (h4 : a^5 + b^5 = 11) : 
  a^10 + b^10 = 123 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l3442_344256


namespace NUMINAMATH_CALUDE_smallest_value_u_cube_plus_v_cube_l3442_344266

theorem smallest_value_u_cube_plus_v_cube (u v : ℂ) 
  (h1 : Complex.abs (u + v) = 2)
  (h2 : Complex.abs (u^2 + v^2) = 17) :
  Complex.abs (u^3 + v^3) = 47 := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_u_cube_plus_v_cube_l3442_344266


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3442_344202

theorem complex_fraction_simplification (z : ℂ) (h : z = 1 - 2*I) : 
  (z^2 + 3) / (z - 1) = 2 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3442_344202


namespace NUMINAMATH_CALUDE_work_completion_time_l3442_344283

theorem work_completion_time 
  (a_rate : ℚ) 
  (b_rate : ℚ) 
  (joint_work_days : ℕ) 
  (a_rate_def : a_rate = 1 / 5) 
  (b_rate_def : b_rate = 1 / 15) 
  (joint_work_days_def : joint_work_days = 2) : 
  (1 - (a_rate + b_rate) * joint_work_days) / b_rate = 7 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3442_344283


namespace NUMINAMATH_CALUDE_dice_game_winning_probability_l3442_344240

/-- Represents the outcome of rolling three dice -/
inductive DiceOutcome
  | AllSame
  | TwoSame
  | AllDifferent

/-- The probability of winning the dice game -/
def winning_probability : ℚ := 2177 / 10000

/-- The strategy for rerolling dice based on the initial outcome -/
def reroll_strategy (outcome : DiceOutcome) : ℕ :=
  match outcome with
  | DiceOutcome.AllSame => 0
  | DiceOutcome.TwoSame => 1
  | DiceOutcome.AllDifferent => 3

theorem dice_game_winning_probability :
  ∀ (num_rolls : ℕ) (max_rerolls : ℕ),
    num_rolls = 3 ∧ max_rerolls = 2 →
    (∀ (outcome : DiceOutcome), reroll_strategy outcome ≤ num_rolls) →
    winning_probability = 2177 / 10000 := by
  sorry


end NUMINAMATH_CALUDE_dice_game_winning_probability_l3442_344240


namespace NUMINAMATH_CALUDE_profit_difference_l3442_344272

/-- Calculates the difference in profit share between two partners given their investments and total profit -/
theorem profit_difference (mary_investment mike_investment total_profit : ℚ) :
  mary_investment = 650 →
  mike_investment = 350 →
  total_profit = 2999.9999999999995 →
  let total_investment := mary_investment + mike_investment
  let equal_share := (1/3) * total_profit / 2
  let remaining_profit := (2/3) * total_profit
  let mary_ratio := mary_investment / total_investment
  let mike_ratio := mike_investment / total_investment
  let mary_total := equal_share + mary_ratio * remaining_profit
  let mike_total := equal_share + mike_ratio * remaining_profit
  mary_total - mike_total = 600 := by sorry

end NUMINAMATH_CALUDE_profit_difference_l3442_344272


namespace NUMINAMATH_CALUDE_sum_of_digit_products_2019_l3442_344227

/-- Product of digits of a natural number -/
def digitProduct (n : ℕ) : ℕ := sorry

/-- Sum of products of digits for numbers from 1 to n -/
def sumOfDigitProducts (n : ℕ) : ℕ := sorry

/-- Theorem stating that the sum of products of digits for integers from 1 to 2019 is 184320 -/
theorem sum_of_digit_products_2019 : sumOfDigitProducts 2019 = 184320 := by sorry

end NUMINAMATH_CALUDE_sum_of_digit_products_2019_l3442_344227


namespace NUMINAMATH_CALUDE_sum_of_digits_equation_l3442_344274

/-- Sum of digits function -/
def S (x : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem sum_of_digits_equation : 
  ∃ x : ℕ, x + S x = 2001 ∧ x = 1977 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_equation_l3442_344274


namespace NUMINAMATH_CALUDE_sum_of_rectangle_areas_l3442_344235

def first_six_odd_numbers : List ℕ := [1, 3, 5, 7, 9, 11]

def rectangle_areas (width : ℕ) (lengths : List ℕ) : List ℕ :=
  lengths.map (λ l => width * l)

theorem sum_of_rectangle_areas :
  let width := 2
  let lengths := first_six_odd_numbers.map (λ n => n * n)
  let areas := rectangle_areas width lengths
  areas.sum = 572 := by sorry

end NUMINAMATH_CALUDE_sum_of_rectangle_areas_l3442_344235


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l3442_344257

theorem negation_of_existence_proposition :
  (¬ ∃ n : ℕ, n^2 ≥ 2^n) ↔ (∀ n : ℕ, n^2 < 2^n) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l3442_344257


namespace NUMINAMATH_CALUDE_kitchen_guest_bath_living_area_l3442_344297

/-- Calculates the area of the kitchen, guest bath, and living area given the areas of other rooms and rent information -/
theorem kitchen_guest_bath_living_area 
  (master_bath_area : ℝ) 
  (guest_bedroom_area : ℝ) 
  (num_guest_bedrooms : ℕ) 
  (total_rent : ℝ) 
  (cost_per_sqft : ℝ) 
  (h1 : master_bath_area = 500) 
  (h2 : guest_bedroom_area = 200) 
  (h3 : num_guest_bedrooms = 2) 
  (h4 : total_rent = 3000) 
  (h5 : cost_per_sqft = 2) : 
  ℝ := by
  sorry

#check kitchen_guest_bath_living_area

end NUMINAMATH_CALUDE_kitchen_guest_bath_living_area_l3442_344297


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3442_344229

theorem tan_alpha_value (α : Real) (h : Real.tan (α + π/4) = 1/5) : Real.tan α = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3442_344229


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3442_344236

theorem triangle_angle_measure (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_eq : a^2 + b^2 + Real.sqrt 2 * a * b = c^2) :
  let C := Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
  C = 3 * π / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3442_344236


namespace NUMINAMATH_CALUDE_proposition_conjunction_false_perpendicular_lines_condition_converse_equivalence_l3442_344222

-- Statement 1
theorem proposition_conjunction_false :
  ¬(∃ x : ℝ, Real.tan x = 1 ∧ ¬(∀ x : ℝ, x^2 + 1 > 0)) := by sorry

-- Statement 2
theorem perpendicular_lines_condition :
  ∃ a b : ℝ, (∀ x y : ℝ, a * x + 3 * y - 1 = 0 ↔ x + b * y + 1 = 0) ∧
             (a * 1 + b * 3 = 0) ∧
             (a / b ≠ -3) := by sorry

-- Statement 3
theorem converse_equivalence :
  (∀ x : ℝ, x ≠ 1 → x^2 - 3*x + 2 ≠ 0) ↔
  (∀ x : ℝ, x^2 - 3*x + 2 = 0 → x = 1) := by sorry

end NUMINAMATH_CALUDE_proposition_conjunction_false_perpendicular_lines_condition_converse_equivalence_l3442_344222


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3442_344298

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ,
  2 * X^5 + 11 * X^4 - 48 * X^3 - 60 * X^2 + 20 * X + 50 =
  (X^3 + 7 * X^2 + 4) * q + (-27 * X^3 - 68 * X^2 + 32 * X + 50) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3442_344298


namespace NUMINAMATH_CALUDE_tangent_lines_to_cubic_l3442_344271

noncomputable def f (x : ℝ) := x^3

def P : ℝ × ℝ := (1, 1)

theorem tangent_lines_to_cubic (x : ℝ) :
  -- The tangent line at point P(1, 1) is y = 3x - 2
  (HasDerivAt f 3 1 ∧ f 1 = 1) →
  -- There are exactly two tangent lines to the curve that pass through point P(1, 1)
  (∃! (m₁ m₂ : ℝ), m₁ ≠ m₂ ∧
    -- First tangent line: y = 3x - 2
    (m₁ = 3 ∧ P.2 = m₁ * P.1 - 2) ∧
    -- Second tangent line: y = 3/4x + 1/4
    (m₂ = 3/4 ∧ P.2 = m₂ * P.1 + 1/4) ∧
    -- Both lines are tangent to the curve
    (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
      HasDerivAt f (3 * x₁^2) x₁ ∧
      HasDerivAt f (3 * x₂^2) x₂ ∧
      f x₁ = m₁ * x₁ - 2 ∧
      f x₂ = m₂ * x₂ + 1/4)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_lines_to_cubic_l3442_344271


namespace NUMINAMATH_CALUDE_hall_covering_cost_l3442_344201

/-- Calculates the total expenditure for covering the interior of a rectangular hall with mat -/
def total_expenditure (length width height cost_per_sqm : ℝ) : ℝ :=
  let floor_area := length * width
  let wall_area := 2 * (length * height + width * height)
  let total_area := floor_area + wall_area
  total_area * cost_per_sqm

/-- Proves that the total expenditure for the given hall dimensions and mat cost is Rs. 39,000 -/
theorem hall_covering_cost : 
  total_expenditure 20 15 5 60 = 39000 := by
  sorry

end NUMINAMATH_CALUDE_hall_covering_cost_l3442_344201


namespace NUMINAMATH_CALUDE_triangle_is_equilateral_l3442_344247

/-- A triangle with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Predicate for angles forming an arithmetic progression -/
def angles_in_arithmetic_progression (t : Triangle) : Prop :=
  t.A + t.C = 2 * t.B

/-- Predicate for sides forming a geometric progression -/
def sides_in_geometric_progression (t : Triangle) : Prop :=
  t.b^2 = t.a * t.c

/-- Theorem stating that a triangle with angles in arithmetic progression
    and sides in geometric progression is equilateral -/
theorem triangle_is_equilateral (t : Triangle)
  (h1 : angles_in_arithmetic_progression t)
  (h2 : sides_in_geometric_progression t) :
  t.A = 60 ∧ t.B = 60 ∧ t.C = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_equilateral_l3442_344247


namespace NUMINAMATH_CALUDE_investment_relationship_l3442_344209

def initial_AA : ℝ := 150
def initial_BB : ℝ := 100
def initial_CC : ℝ := 200

def year1_AA_change : ℝ := 0.10
def year1_BB_change : ℝ := -0.20
def year1_CC_change : ℝ := 0.05

def year2_AA_change : ℝ := -0.05
def year2_BB_change : ℝ := 0.15
def year2_CC_change : ℝ := -0.10

def final_AA : ℝ := initial_AA * (1 + year1_AA_change) * (1 + year2_AA_change)
def final_BB : ℝ := initial_BB * (1 + year1_BB_change) * (1 + year2_BB_change)
def final_CC : ℝ := initial_CC * (1 + year1_CC_change) * (1 + year2_CC_change)

theorem investment_relationship : final_BB < final_AA ∧ final_AA < final_CC := by
  sorry

end NUMINAMATH_CALUDE_investment_relationship_l3442_344209


namespace NUMINAMATH_CALUDE_x_percent_plus_six_equals_ten_l3442_344260

theorem x_percent_plus_six_equals_ten (x : ℝ) (h1 : x > 0) 
  (h2 : x * (x / 100) + 6 = 10) : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_x_percent_plus_six_equals_ten_l3442_344260
