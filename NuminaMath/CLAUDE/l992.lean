import Mathlib

namespace NUMINAMATH_CALUDE_imaginary_part_of_2_minus_i_l992_99241

theorem imaginary_part_of_2_minus_i :
  Complex.im (2 - Complex.I) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_2_minus_i_l992_99241


namespace NUMINAMATH_CALUDE_savings_calculation_l992_99209

def calculate_savings (initial_winnings : ℚ) (first_saving_ratio : ℚ) (profit_ratio : ℚ) (second_saving_ratio : ℚ) : ℚ :=
  let first_saving := initial_winnings * first_saving_ratio
  let second_bet := initial_winnings * (1 - first_saving_ratio)
  let second_bet_earnings := second_bet * (1 + profit_ratio)
  let second_saving := second_bet_earnings * second_saving_ratio
  first_saving + second_saving

theorem savings_calculation :
  calculate_savings 100 (1/2) (3/5) (1/2) = 90 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l992_99209


namespace NUMINAMATH_CALUDE_probability_all_odd_is_correct_l992_99247

def total_slips : ℕ := 10
def odd_slips : ℕ := 5
def drawn_slips : ℕ := 4

def probability_all_odd : ℚ := (odd_slips.choose drawn_slips) / (total_slips.choose drawn_slips)

theorem probability_all_odd_is_correct : 
  probability_all_odd = 1 / 42 := by sorry

end NUMINAMATH_CALUDE_probability_all_odd_is_correct_l992_99247


namespace NUMINAMATH_CALUDE_homework_time_calculation_l992_99258

theorem homework_time_calculation (total_time : ℝ) :
  (0.3 * total_time = 0.3 * total_time) ∧  -- Time spent on math
  (0.4 * total_time = 0.4 * total_time) ∧  -- Time spent on science
  (total_time - 0.3 * total_time - 0.4 * total_time = 45) →  -- Time spent on other subjects
  total_time = 150 := by
sorry

end NUMINAMATH_CALUDE_homework_time_calculation_l992_99258


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l992_99206

def M : Set ℝ := {-2, 0, 1}
def N : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l992_99206


namespace NUMINAMATH_CALUDE_contestant_speaking_orders_l992_99227

theorem contestant_speaking_orders :
  let total_contestants : ℕ := 6
  let restricted_contestant : ℕ := 1
  let available_positions : ℕ := total_contestants - 2

  available_positions * Nat.factorial (total_contestants - restricted_contestant) = 480 :=
by sorry

end NUMINAMATH_CALUDE_contestant_speaking_orders_l992_99227


namespace NUMINAMATH_CALUDE_range_of_f_l992_99283

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- State the theorem
theorem range_of_f :
  (∀ x : ℝ, (1 + x > 0 ∧ 1 - x > 0) → (3/4 ≤ f x ∧ f x ≤ 57)) →
  Set.range f = Set.Ici 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l992_99283


namespace NUMINAMATH_CALUDE_additional_cats_needed_l992_99250

def current_cats : ℕ := 11
def target_cats : ℕ := 43

theorem additional_cats_needed : target_cats - current_cats = 32 := by
  sorry

end NUMINAMATH_CALUDE_additional_cats_needed_l992_99250


namespace NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l992_99244

/-- Represents the different employee categories in the company -/
inductive EmployeeCategory
  | Senior
  | Intermediate
  | General

/-- Represents the company's employee distribution -/
structure CompanyDistribution where
  total : Nat
  senior : Nat
  intermediate : Nat
  general : Nat
  senior_count : senior ≤ total
  intermediate_count : intermediate ≤ total
  general_count : general ≤ total
  total_sum : senior + intermediate + general = total

/-- Represents a sampling method -/
inductive SamplingMethod
  | Simple
  | Stratified
  | Cluster
  | Systematic

/-- Determines the most appropriate sampling method given a company distribution and sample size -/
def mostAppropriateSamplingMethod (dist : CompanyDistribution) (sampleSize : Nat) : SamplingMethod :=
  sorry

/-- Theorem stating that stratified sampling is the most appropriate method for the given scenario -/
theorem stratified_sampling_most_appropriate (dist : CompanyDistribution) (sampleSize : Nat) :
  dist.total = 150 ∧ dist.senior = 15 ∧ dist.intermediate = 45 ∧ dist.general = 90 ∧ sampleSize = 30 →
  mostAppropriateSamplingMethod dist sampleSize = SamplingMethod.Stratified :=
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l992_99244


namespace NUMINAMATH_CALUDE_stating_parking_arrangement_count_l992_99265

/-- Represents the number of parking spaces -/
def num_spaces : ℕ := 8

/-- Represents the number of trucks -/
def num_trucks : ℕ := 2

/-- Represents the number of cars -/
def num_cars : ℕ := 2

/-- Represents the total number of vehicles -/
def total_vehicles : ℕ := num_trucks + num_cars

/-- 
Represents the number of ways to arrange trucks and cars in a row of parking spaces,
where vehicles of the same type must be adjacent.
-/
def parking_arrangements (spaces : ℕ) (trucks : ℕ) (cars : ℕ) : ℕ :=
  sorry

/-- 
Theorem stating that the number of ways to arrange 2 trucks and 2 cars
in a row of 8 parking spaces, where vehicles of the same type must be adjacent,
is equal to 120.
-/
theorem parking_arrangement_count :
  parking_arrangements num_spaces num_trucks num_cars = 120 := by
  sorry

end NUMINAMATH_CALUDE_stating_parking_arrangement_count_l992_99265


namespace NUMINAMATH_CALUDE_factorial_fraction_simplification_l992_99234

theorem factorial_fraction_simplification :
  (4 * Nat.factorial 6 + 20 * Nat.factorial 5 + 48 * Nat.factorial 4) / Nat.factorial 7 = 134 / 105 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_simplification_l992_99234


namespace NUMINAMATH_CALUDE_root_transformation_l992_99242

theorem root_transformation {a₁ a₂ a₃ b c₁ c₂ c₃ : ℝ} 
  (h_distinct : c₁ ≠ c₂ ∧ c₂ ≠ c₃ ∧ c₁ ≠ c₃)
  (h_roots : ∀ x : ℝ, (x - a₁) * (x - a₂) * (x - a₃) = b ↔ x = c₁ ∨ x = c₂ ∨ x = c₃) :
  ∀ x : ℝ, (x + c₁) * (x + c₂) * (x + c₃) = b ↔ x = -a₁ ∨ x = -a₂ ∨ x = -a₃ := by
sorry

end NUMINAMATH_CALUDE_root_transformation_l992_99242


namespace NUMINAMATH_CALUDE_one_third_minus_decimal_l992_99248

theorem one_third_minus_decimal : (1 : ℚ) / 3 - 333333 / 1000000 = 1 / (3 * 1000000) := by
  sorry

end NUMINAMATH_CALUDE_one_third_minus_decimal_l992_99248


namespace NUMINAMATH_CALUDE_books_bought_l992_99272

theorem books_bought (initial_amount : ℕ) (cost_per_book : ℕ) (remaining_amount : ℕ) :
  initial_amount = 79 →
  cost_per_book = 7 →
  remaining_amount = 16 →
  (initial_amount - remaining_amount) / cost_per_book = 9 :=
by sorry

end NUMINAMATH_CALUDE_books_bought_l992_99272


namespace NUMINAMATH_CALUDE_product_from_lcm_gcd_l992_99208

theorem product_from_lcm_gcd (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 48) 
  (h_gcd : Nat.gcd a b = 8) : 
  a * b = 384 := by
sorry

end NUMINAMATH_CALUDE_product_from_lcm_gcd_l992_99208


namespace NUMINAMATH_CALUDE_solution_difference_l992_99294

theorem solution_difference (p q : ℝ) : 
  (p - 5) * (p + 5) = 26 * p - 130 →
  (q - 5) * (q + 5) = 26 * q - 130 →
  p ≠ q →
  p > q →
  p - q = 16 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l992_99294


namespace NUMINAMATH_CALUDE_yellow_apples_probability_l992_99292

/-- The probability of choosing 2 yellow apples out of 10 apples, where 4 are yellow -/
theorem yellow_apples_probability (total_apples : ℕ) (yellow_apples : ℕ) (chosen_apples : ℕ)
  (h1 : total_apples = 10)
  (h2 : yellow_apples = 4)
  (h3 : chosen_apples = 2) :
  (yellow_apples.choose chosen_apples : ℚ) / (total_apples.choose chosen_apples) = 2 / 15 :=
by sorry

end NUMINAMATH_CALUDE_yellow_apples_probability_l992_99292


namespace NUMINAMATH_CALUDE_square_difference_pattern_l992_99282

theorem square_difference_pattern (n : ℕ+) : (2*n + 1)^2 - (2*n - 1)^2 = 8*n := by
  sorry

end NUMINAMATH_CALUDE_square_difference_pattern_l992_99282


namespace NUMINAMATH_CALUDE_otimes_properties_l992_99291

def otimes (a b : ℝ) : ℝ := a * b + a + b

theorem otimes_properties :
  (∀ a b : ℝ, otimes a b = otimes b a) ∧
  (∃ a b c : ℝ, otimes (otimes a b) c ≠ otimes a (otimes b c)) ∧
  (∃ a b c : ℝ, otimes (a + b) c ≠ otimes a c + otimes b c) := by
  sorry

end NUMINAMATH_CALUDE_otimes_properties_l992_99291


namespace NUMINAMATH_CALUDE_curve_expression_bound_l992_99202

theorem curve_expression_bound (x y : ℝ) : 
  4 * x^2 + y^2 = 16 → -4 ≤ Real.sqrt 3 * x + (1/2) * y ∧ Real.sqrt 3 * x + (1/2) * y ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_curve_expression_bound_l992_99202


namespace NUMINAMATH_CALUDE_halves_in_two_sevenths_l992_99212

theorem halves_in_two_sevenths : (2 : ℚ) / 7 / (1 : ℚ) / 2 = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_halves_in_two_sevenths_l992_99212


namespace NUMINAMATH_CALUDE_warehouse_boxes_theorem_l992_99235

/-- The number of boxes in two warehouses -/
def total_boxes (first_warehouse : ℕ) (second_warehouse : ℕ) : ℕ :=
  first_warehouse + second_warehouse

theorem warehouse_boxes_theorem (first_warehouse second_warehouse : ℕ) 
  (h1 : first_warehouse = 400)
  (h2 : first_warehouse = 2 * second_warehouse) : 
  total_boxes first_warehouse second_warehouse = 600 := by
  sorry

end NUMINAMATH_CALUDE_warehouse_boxes_theorem_l992_99235


namespace NUMINAMATH_CALUDE_m_range_theorem_l992_99273

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem m_range_theorem (f : ℝ → ℝ) 
  (h1 : ∀ x ∈ Set.Icc (-2) 2, f x ≠ 0 → True)  -- f is defined on [-2, 2]
  (h2 : is_even f)
  (h3 : monotone_decreasing_on f 0 2)
  (h4 : ∀ m, f (1 - m) < f m) :
  ∀ m, -2 ≤ m ∧ m < (1/2) := by
sorry

end NUMINAMATH_CALUDE_m_range_theorem_l992_99273


namespace NUMINAMATH_CALUDE_tangent_lines_count_l992_99205

/-- The function f(x) = -x³ + 6x² - 9x + 8 -/
def f (x : ℝ) : ℝ := -x^3 + 6*x^2 - 9*x + 8

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := -3*x^2 + 12*x - 9

/-- Condition for a point (x₀, f(x₀)) to be on a tangent line passing through (0, 0) -/
def is_tangent_point (x₀ : ℝ) : Prop :=
  f x₀ = (f' x₀) * x₀

theorem tangent_lines_count :
  ∃ (S : Finset ℝ), (∀ x ∈ S, is_tangent_point x) ∧ S.card = 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_count_l992_99205


namespace NUMINAMATH_CALUDE_distance_to_origin_of_complex_number_l992_99243

theorem distance_to_origin_of_complex_number : ∃ (z : ℂ), 
  z = (2 * Complex.I) / (1 - Complex.I) ∧ 
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_of_complex_number_l992_99243


namespace NUMINAMATH_CALUDE_omega_properties_l992_99203

/-- The weight function ω(n) that returns the sum of binary digits of n -/
def ω (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 2 + ω (n / 2))

/-- Theorem stating the properties of the ω function -/
theorem omega_properties :
  ∀ n : ℕ,
  (ω (2 * n) = ω n) ∧
  (ω (8 * n + 5) = ω (4 * n + 3)) ∧
  (ω ((2 ^ n) - 1) = n) :=
by sorry

end NUMINAMATH_CALUDE_omega_properties_l992_99203


namespace NUMINAMATH_CALUDE_horner_method_equals_f_at_2_l992_99255

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * x^3 + 2 * x^2 + x + 1

-- Define Horner's method for this specific polynomial
def horner_method (x : ℝ) : ℝ := x * (x * (3 * x + 2) + 1) + 1

-- Theorem statement
theorem horner_method_equals_f_at_2 : 
  horner_method 2 = f 2 ∧ f 2 = 35 := by sorry

end NUMINAMATH_CALUDE_horner_method_equals_f_at_2_l992_99255


namespace NUMINAMATH_CALUDE_initial_overs_correct_l992_99210

/-- Represents the number of overs played initially in a cricket game. -/
def initial_overs : ℝ := 10

/-- The target score for the cricket game. -/
def target_score : ℝ := 282

/-- The initial run rate in runs per over. -/
def initial_run_rate : ℝ := 6.2

/-- The required run rate for the remaining overs in runs per over. -/
def required_run_rate : ℝ := 5.5

/-- The number of remaining overs. -/
def remaining_overs : ℝ := 40

/-- Theorem stating that the initial number of overs is correct given the conditions. -/
theorem initial_overs_correct : 
  target_score = initial_run_rate * initial_overs + required_run_rate * remaining_overs :=
by sorry

end NUMINAMATH_CALUDE_initial_overs_correct_l992_99210


namespace NUMINAMATH_CALUDE_same_terminal_side_angles_l992_99240

open Set Real

def isObtuse (α : ℝ) : Prop := π / 2 < α ∧ α < π

theorem same_terminal_side_angles
  (α : ℝ)
  (h_obtuse : isObtuse α)
  (h_sin : sin α = 1 / 2) :
  {β | ∃ k : ℤ, β = 5 * π / 6 + 2 * π * k} =
  {β | ∃ k : ℤ, β = α + 2 * π * k} :=
by sorry

end NUMINAMATH_CALUDE_same_terminal_side_angles_l992_99240


namespace NUMINAMATH_CALUDE_number_of_factors_of_n_l992_99238

def n : ℕ := 2^2 * 3^2 * 7^2

theorem number_of_factors_of_n : (Finset.card (Nat.divisors n)) = 27 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_of_n_l992_99238


namespace NUMINAMATH_CALUDE_choose_four_from_seven_l992_99228

theorem choose_four_from_seven (n : ℕ) (k : ℕ) : n = 7 ∧ k = 4 → Nat.choose n k = 35 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_seven_l992_99228


namespace NUMINAMATH_CALUDE_erdos_mordell_two_points_l992_99275

/-- The Erdős–Mordell inequality for two points -/
theorem erdos_mordell_two_points
  (a b c : ℝ)
  (a₁ b₁ c₁ : ℝ)
  (a₂ b₂ c₂ : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (ha₁ : 0 ≤ a₁) (hb₁ : 0 ≤ b₁) (hc₁ : 0 ≤ c₁)
  (ha₂ : 0 ≤ a₂) (hb₂ : 0 ≤ b₂) (hc₂ : 0 ≤ c₂)
  (htri : a + b > c ∧ b + c > a ∧ c + a > b) :
  a * a₁ * a₂ + b * b₁ * b₂ + c * c₁ * c₂ ≥ a * b * c :=
by sorry

end NUMINAMATH_CALUDE_erdos_mordell_two_points_l992_99275


namespace NUMINAMATH_CALUDE_apec_photo_arrangements_l992_99268

def arrangement_count (n : ℕ) (k : ℕ) : ℕ := n.factorial

theorem apec_photo_arrangements :
  let total_leaders : ℕ := 21
  let front_row : ℕ := 11
  let back_row : ℕ := 10
  let fixed_positions : ℕ := 3
  let remaining_leaders : ℕ := total_leaders - fixed_positions
  let us_russia_arrangements : ℕ := arrangement_count 2 2
  let other_arrangements : ℕ := arrangement_count remaining_leaders remaining_leaders
  
  (us_russia_arrangements * other_arrangements : ℕ) = 
    arrangement_count 2 2 * arrangement_count 18 18 :=
by sorry

end NUMINAMATH_CALUDE_apec_photo_arrangements_l992_99268


namespace NUMINAMATH_CALUDE_power_equality_natural_numbers_l992_99274

theorem power_equality_natural_numbers (a b : ℕ) :
  a ^ b = b ^ a ↔ (a = b) ∨ (a = 2 ∧ b = 4) ∨ (a = 4 ∧ b = 2) := by
sorry

end NUMINAMATH_CALUDE_power_equality_natural_numbers_l992_99274


namespace NUMINAMATH_CALUDE_six_tangent_circles_l992_99207

-- Define the circles C₁ and C₂
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the property of being tangent
def are_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

-- Define the problem setup
def problem_setup (C₁ C₂ : Circle) : Prop :=
  C₁.radius = 2 ∧
  C₂.radius = 2 ∧
  are_tangent C₁ C₂

-- Define a function to count tangent circles
def count_tangent_circles (C₁ C₂ : Circle) : ℕ :=
  sorry -- The actual counting logic would go here

-- The main theorem
theorem six_tangent_circles (C₁ C₂ : Circle) :
  problem_setup C₁ C₂ → count_tangent_circles C₁ C₂ = 6 :=
by sorry


end NUMINAMATH_CALUDE_six_tangent_circles_l992_99207


namespace NUMINAMATH_CALUDE_product_mod_600_l992_99261

theorem product_mod_600 : (1234 * 2047) % 600 = 198 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_600_l992_99261


namespace NUMINAMATH_CALUDE_ellipse_axes_l992_99224

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  2 * x^2 + y^2 - 12 = 2 * x + 4 * y

-- Define the standard form of the ellipse
def standard_form (a b h k : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

-- Theorem stating the semi-major and semi-minor axes of the ellipse
theorem ellipse_axes :
  ∃ h k : ℝ, 
    (∀ x y : ℝ, ellipse_equation x y ↔ standard_form 17 8.5 h k x y) ∧
    (17 > 8.5) :=
sorry

end NUMINAMATH_CALUDE_ellipse_axes_l992_99224


namespace NUMINAMATH_CALUDE_f_of_two_equals_five_l992_99290

-- Define the function f
def f : ℝ → ℝ := fun x ↦ 2 * (x - 1) + 3

-- State the theorem
theorem f_of_two_equals_five : f 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_of_two_equals_five_l992_99290


namespace NUMINAMATH_CALUDE_money_lasts_9_weeks_l992_99262

def lawn_mowing_earnings : ℕ := 9
def weed_eating_earnings : ℕ := 18
def weekly_spending : ℕ := 3

theorem money_lasts_9_weeks : 
  (lawn_mowing_earnings + weed_eating_earnings) / weekly_spending = 9 := by
  sorry

end NUMINAMATH_CALUDE_money_lasts_9_weeks_l992_99262


namespace NUMINAMATH_CALUDE_special_polynomial_p_count_l992_99260

/-- Represents a polynomial of degree 4 with specific properties -/
structure SpecialPolynomial where
  m : ℤ
  n : ℤ
  p : ℤ
  zeros : Fin 4 → ℝ
  is_zero : ∀ i, (zeros i)^4 - 2004 * (zeros i)^3 + m * (zeros i)^2 + n * (zeros i) + p = 0
  distinct_zeros : ∀ i j, i ≠ j → zeros i ≠ zeros j
  positive_zeros : ∀ i, zeros i > 0
  integer_zero : ∃ i, ∃ k : ℤ, zeros i = k
  sum_property : ∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ zeros i = zeros j + zeros k
  product_property : ∃ i j k l, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i ∧ zeros i = zeros j * zeros k * zeros l

/-- The number of possible values for p in a SpecialPolynomial -/
def count_p_values : ℕ := 63000

/-- Theorem stating that there are exactly 63000 possible values for p -/
theorem special_polynomial_p_count :
  (∃ f : Set SpecialPolynomial → ℕ, f {sp | sp.p = p} = count_p_values) :=
sorry

end NUMINAMATH_CALUDE_special_polynomial_p_count_l992_99260


namespace NUMINAMATH_CALUDE_r_amount_l992_99211

theorem r_amount (total : ℝ) (p_q_amount : ℝ) (r_amount : ℝ) : 
  total = 6000 →
  r_amount = (2/3) * p_q_amount →
  total = p_q_amount + r_amount →
  r_amount = 2400 := by
sorry

end NUMINAMATH_CALUDE_r_amount_l992_99211


namespace NUMINAMATH_CALUDE_binary_equals_octal_l992_99296

-- Define the binary number
def binary_num : List Bool := [true, false, true, true, true, false]

-- Define the octal number
def octal_num : Nat := 56

-- Function to convert binary to decimal
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Function to convert decimal to octal
def decimal_to_octal (n : Nat) : Nat :=
  if n < 8 then n
  else 10 * (decimal_to_octal (n / 8)) + (n % 8)

-- Theorem stating that the binary number is equal to the octal number
theorem binary_equals_octal : 
  decimal_to_octal (binary_to_decimal binary_num) = octal_num := by
  sorry

end NUMINAMATH_CALUDE_binary_equals_octal_l992_99296


namespace NUMINAMATH_CALUDE_smallest_divisible_by_one_to_ten_l992_99287

theorem smallest_divisible_by_one_to_ten : 
  ∃ n : ℕ, (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ n) ∧ 
    (∀ m : ℕ, m < n → ∃ j : ℕ, 1 ≤ j ∧ j ≤ 10 ∧ ¬(j ∣ m)) :=
by
  use 2520
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_one_to_ten_l992_99287


namespace NUMINAMATH_CALUDE_molecular_weight_CCl4_proof_l992_99216

/-- The molecular weight of CCl4 in g/mol -/
def molecular_weight_CCl4 : ℝ := 152

/-- The number of moles used in the given condition -/
def given_moles : ℝ := 7

/-- The total weight of the given moles of CCl4 in g/mol -/
def given_total_weight : ℝ := 1064

/-- Theorem stating that the molecular weight of CCl4 is correct given the condition -/
theorem molecular_weight_CCl4_proof :
  molecular_weight_CCl4 * given_moles = given_total_weight :=
by sorry

end NUMINAMATH_CALUDE_molecular_weight_CCl4_proof_l992_99216


namespace NUMINAMATH_CALUDE_circle_line_tangent_l992_99225

/-- A circle C in the xy-plane -/
def Circle (a : ℝ) (x y : ℝ) : Prop :=
  x^2 - 2*a*x + y^2 = 0

/-- A line l in the xy-plane -/
def Line (x y : ℝ) : Prop :=
  x - Real.sqrt 3 * y + 3 = 0

/-- The circle and line are tangent if they intersect at exactly one point -/
def Tangent (a : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, Circle a p.1 p.2 ∧ Line p.1 p.2

theorem circle_line_tangent (a : ℝ) (h1 : a > 0) (h2 : Tangent a) : a = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_line_tangent_l992_99225


namespace NUMINAMATH_CALUDE_carbon_neutral_olympics_emissions_l992_99204

theorem carbon_neutral_olympics_emissions (emissions : ℝ) : 
  emissions = 320000 → emissions = 3.2 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_carbon_neutral_olympics_emissions_l992_99204


namespace NUMINAMATH_CALUDE_percentage_men_science_majors_l992_99289

/-- Given a college class, proves that 28% of men are science majors -/
theorem percentage_men_science_majors 
  (women_science_major_ratio : Real) 
  (non_science_ratio : Real) 
  (men_ratio : Real) 
  (h1 : women_science_major_ratio = 0.2)
  (h2 : non_science_ratio = 0.6)
  (h3 : men_ratio = 0.4) :
  (1 - non_science_ratio - women_science_major_ratio * (1 - men_ratio)) / men_ratio = 0.28 := by
  sorry

end NUMINAMATH_CALUDE_percentage_men_science_majors_l992_99289


namespace NUMINAMATH_CALUDE_cosine_function_period_l992_99259

/-- Given a cosine function y = a cos(bx + c) + d where a, b, c, and d are positive constants,
    if the graph covers two periods in an interval of 2π, then b = 2. -/
theorem cosine_function_period (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ x : ℝ, ∃ y : ℝ, y = a * Real.cos (b * x + c) + d) →
  (∀ x : ℝ, ∃ y : ℝ, y = a * Real.cos (b * (x + 2 * Real.pi) + c) + d) →
  b = 2 := by sorry

end NUMINAMATH_CALUDE_cosine_function_period_l992_99259


namespace NUMINAMATH_CALUDE_money_left_calculation_l992_99297

-- Define the initial amount, spent amount, and amount given to each friend
def initial_amount : ℚ := 5.10
def spent_on_sweets : ℚ := 1.05
def given_to_friend : ℚ := 1.00
def number_of_friends : ℕ := 2

-- Theorem to prove
theorem money_left_calculation :
  initial_amount - (spent_on_sweets + number_of_friends * given_to_friend) = 2.05 := by
  sorry


end NUMINAMATH_CALUDE_money_left_calculation_l992_99297


namespace NUMINAMATH_CALUDE_symmetry_problem_l992_99277

/-- Two complex numbers are symmetric with respect to the imaginary axis if their real parts are negatives of each other and their imaginary parts are equal. -/
def symmetric_wrt_imaginary_axis (z₁ z₂ : ℂ) : Prop :=
  z₁.re = -z₂.re ∧ z₁.im = z₂.im

/-- Given that z₁ = 1 - 2i and z₁ and z₂ are symmetric with respect to the imaginary axis,
    prove that z₂ = -1 - 2i. -/
theorem symmetry_problem (z₁ z₂ : ℂ) 
    (h₁ : z₁ = 1 - 2*I) 
    (h₂ : symmetric_wrt_imaginary_axis z₁ z₂) : 
  z₂ = -1 - 2*I :=
sorry

end NUMINAMATH_CALUDE_symmetry_problem_l992_99277


namespace NUMINAMATH_CALUDE_abs_complex_value_l992_99279

theorem abs_complex_value : Complex.abs (-3 - (9/4)*Complex.I) = 15/4 := by
  sorry

end NUMINAMATH_CALUDE_abs_complex_value_l992_99279


namespace NUMINAMATH_CALUDE_function_identically_zero_l992_99218

/-- A function satisfying the given conditions is identically zero. -/
theorem function_identically_zero (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f)
  (h_zero : f 0 = 0)
  (h_bound : ∀ x : ℝ, 0 < |f x| → |f x| < (1/2) → 
    |deriv f x| ≤ |f x * Real.log (|f x|)|) : 
  ∀ x : ℝ, f x = 0 := by
sorry

end NUMINAMATH_CALUDE_function_identically_zero_l992_99218


namespace NUMINAMATH_CALUDE_sally_quarters_l992_99213

/-- Given an initial quantity of quarters and an additional amount received,
    calculate the total number of quarters. -/
def total_quarters (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem: Given 760 initial quarters and 418 additional quarters,
    the total number of quarters is 1178. -/
theorem sally_quarters : total_quarters 760 418 = 1178 := by
  sorry

end NUMINAMATH_CALUDE_sally_quarters_l992_99213


namespace NUMINAMATH_CALUDE_infinite_sqrt_twelve_l992_99217

theorem infinite_sqrt_twelve (x : ℝ) : x = Real.sqrt (12 + x) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_infinite_sqrt_twelve_l992_99217


namespace NUMINAMATH_CALUDE_max_value_theorem_l992_99231

theorem max_value_theorem (a b c : ℝ) 
  (nonneg_a : a ≥ 0) (nonneg_b : b ≥ 0) (nonneg_c : c ≥ 0)
  (sum_squares : a^2 + b^2 + c^2 = 1) :
  2 * a * b * Real.sqrt 2 + 2 * a * c ≤ 1 ∧ 
  ∃ a₀ b₀ c₀ : ℝ, a₀ ≥ 0 ∧ b₀ ≥ 0 ∧ c₀ ≥ 0 ∧ 
    a₀^2 + b₀^2 + c₀^2 = 1 ∧
    2 * a₀ * b₀ * Real.sqrt 2 + 2 * a₀ * c₀ = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l992_99231


namespace NUMINAMATH_CALUDE_integer_solutions_x_squared_plus_15_eq_y_squared_l992_99230

theorem integer_solutions_x_squared_plus_15_eq_y_squared :
  {(x, y) : ℤ × ℤ | x^2 + 15 = y^2} =
  {(7, 8), (-7, -8), (-7, 8), (7, -8), (1, 4), (-1, -4), (-1, 4), (1, -4)} := by
sorry

end NUMINAMATH_CALUDE_integer_solutions_x_squared_plus_15_eq_y_squared_l992_99230


namespace NUMINAMATH_CALUDE_polynomial_equality_l992_99215

-- Define the polynomials
variable (x : ℝ)
def f (x : ℝ) : ℝ := x^3 - 3*x - 1
def h (x : ℝ) : ℝ := -x^3 + 5*x^2 + 3*x

-- State the theorem
theorem polynomial_equality :
  (∀ x, f x + h x = 5*x^2 - 1) ∧ 
  (∀ x, f x = x^3 - 3*x - 1) →
  (∀ x, h x = -x^3 + 5*x^2 + 3*x) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l992_99215


namespace NUMINAMATH_CALUDE_darwin_gas_expense_l992_99246

def initial_amount : ℝ := 600
def final_amount : ℝ := 300

theorem darwin_gas_expense (x : ℝ) 
  (h1 : 0 < x ∧ x < 1) 
  (h2 : final_amount = initial_amount - x * initial_amount - (1/4) * (initial_amount - x * initial_amount)) :
  x = 1/3 := by
sorry

end NUMINAMATH_CALUDE_darwin_gas_expense_l992_99246


namespace NUMINAMATH_CALUDE_partnership_profit_l992_99286

/-- Given the investment ratios and C's profit share, calculate the total profit -/
theorem partnership_profit (a b c : ℚ) (c_profit : ℚ) : 
  a = 6 ∧ b = 2 ∧ c = 9 ∧ c_profit = 6000.000000000001 →
  (a + b + c) * c_profit / c = 11333.333333333336 :=
by sorry

end NUMINAMATH_CALUDE_partnership_profit_l992_99286


namespace NUMINAMATH_CALUDE_least_frood_number_l992_99251

def drop_score (n : ℕ) : ℕ := n * (n + 1) / 2

def eat_score (n : ℕ) : ℕ := 10 * n

theorem least_frood_number : ∀ k : ℕ, k < 20 → drop_score k ≤ eat_score k ∧ drop_score 20 > eat_score 20 := by
  sorry

end NUMINAMATH_CALUDE_least_frood_number_l992_99251


namespace NUMINAMATH_CALUDE_largest_r_same_range_l992_99263

/-- A quadratic polynomial function -/
def f (r : ℝ) (x : ℝ) : ℝ := 2 * x^2 - 3 * x + r

/-- The theorem stating the largest value of r for which f and f ∘ f have the same range -/
theorem largest_r_same_range :
  ∃ (r_max : ℝ), r_max = 15/8 ∧
  ∀ (r : ℝ), Set.range (f r) = Set.range (f r ∘ f r) ↔ r ≤ r_max :=
sorry

end NUMINAMATH_CALUDE_largest_r_same_range_l992_99263


namespace NUMINAMATH_CALUDE_lemon_heads_distribution_l992_99232

theorem lemon_heads_distribution (total : Nat) (friends : Nat) (each : Nat) : 
  total = 72 → friends = 6 → total / friends = each → each = 12 := by sorry

end NUMINAMATH_CALUDE_lemon_heads_distribution_l992_99232


namespace NUMINAMATH_CALUDE_proportion_problem_l992_99288

theorem proportion_problem (x y : ℚ) : 
  (3/4 : ℚ) / x = 7/8 → x / y = 5/6 → x = 6/7 ∧ y = 36/35 := by
  sorry

end NUMINAMATH_CALUDE_proportion_problem_l992_99288


namespace NUMINAMATH_CALUDE_oranges_and_cookies_donation_l992_99278

theorem oranges_and_cookies_donation (total_oranges : ℕ) (total_cookies : ℕ) (num_children : ℕ) 
  (h_oranges : total_oranges = 81)
  (h_cookies : total_cookies = 65)
  (h_children : num_children = 7) :
  (total_oranges % num_children = 4) ∧ (total_cookies % num_children = 2) :=
by sorry

end NUMINAMATH_CALUDE_oranges_and_cookies_donation_l992_99278


namespace NUMINAMATH_CALUDE_amanda_remaining_budget_l992_99256

/- Define the budgets -/
def samuel_budget : ℚ := 25
def kevin_budget : ℚ := 20
def laura_budget : ℚ := 18
def amanda_budget : ℚ := 15

/- Define the regular ticket prices -/
def samuel_ticket_price : ℚ := 14
def kevin_ticket_price : ℚ := 10
def laura_ticket_price : ℚ := 10
def amanda_ticket_price : ℚ := 8

/- Define the discount rates -/
def general_discount : ℚ := 0.1
def student_discount : ℚ := 0.1

/- Define Samuel's additional expenses -/
def samuel_drink : ℚ := 6
def samuel_popcorn : ℚ := 3
def samuel_candy : ℚ := 1

/- Define Kevin's additional expense -/
def kevin_combo : ℚ := 7

/- Define Laura's additional expenses -/
def laura_popcorn : ℚ := 4
def laura_drink : ℚ := 2

/- Calculate discounted ticket prices -/
def samuel_discounted_ticket : ℚ := samuel_ticket_price * (1 - general_discount)
def kevin_discounted_ticket : ℚ := kevin_ticket_price * (1 - general_discount)
def laura_discounted_ticket : ℚ := laura_ticket_price * (1 - general_discount)
def amanda_discounted_ticket : ℚ := amanda_ticket_price * (1 - general_discount) * (1 - student_discount)

/- Define the theorem -/
theorem amanda_remaining_budget :
  amanda_budget - amanda_discounted_ticket = 8.52 := by sorry

end NUMINAMATH_CALUDE_amanda_remaining_budget_l992_99256


namespace NUMINAMATH_CALUDE_inequality_proof_l992_99239

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / Real.sqrt (a^2 + 8*b*c) + b / Real.sqrt (b^2 + 8*c*a) + c / Real.sqrt (c^2 + 8*a*b) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l992_99239


namespace NUMINAMATH_CALUDE_factorization_equality_l992_99284

/-- For all real numbers a and b, ab² - 2ab + a = a(b-1)² --/
theorem factorization_equality (a b : ℝ) : a * b^2 - 2 * a * b + a = a * (b - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l992_99284


namespace NUMINAMATH_CALUDE_quadratic_monotonic_condition_l992_99276

/-- A quadratic function f(x) = x^2 - 2mx + 3 is monotonic on [2, 3] if and only if m ∈ (-∞, 2] ∪ [3, +∞) -/
theorem quadratic_monotonic_condition (m : ℝ) :
  (∀ x ∈ Set.Icc 2 3, Monotone (fun x => x^2 - 2*m*x + 3)) ↔ 
  (m ≤ 2 ∨ m ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_monotonic_condition_l992_99276


namespace NUMINAMATH_CALUDE_marks_garden_flowers_l992_99295

/-- The number of flowers in Mark's garden -/
def total_flowers : ℕ := by sorry

/-- The number of yellow flowers -/
def yellow_flowers : ℕ := 10

/-- The number of purple flowers -/
def purple_flowers : ℕ := yellow_flowers + (yellow_flowers * 8 / 10)

/-- The number of green flowers -/
def green_flowers : ℕ := (yellow_flowers + purple_flowers) * 25 / 100

/-- The number of red flowers -/
def red_flowers : ℕ := (yellow_flowers + purple_flowers + green_flowers) * 35 / 100

theorem marks_garden_flowers :
  total_flowers = yellow_flowers + purple_flowers + green_flowers + red_flowers ∧
  total_flowers = 47 := by sorry

end NUMINAMATH_CALUDE_marks_garden_flowers_l992_99295


namespace NUMINAMATH_CALUDE_opposite_of_neg_five_l992_99271

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (a : ℤ) : ℤ := -a

/-- Theorem: The opposite of -5 is 5. -/
theorem opposite_of_neg_five : opposite (-5) = 5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_neg_five_l992_99271


namespace NUMINAMATH_CALUDE_unique_number_with_seven_coprimes_l992_99293

def connection (a b : ℕ) : ℚ :=
  (Nat.lcm a b : ℚ) / (a * b : ℚ)

def isCoprimeWithExactlyN (x n : ℕ) : Prop :=
  (∃ (S : Finset ℕ), S.card = n ∧ 
    (∀ y ∈ S, y < 20 ∧ connection x y = 1) ∧
    (∀ y < 20, y ∉ S → connection x y ≠ 1))

theorem unique_number_with_seven_coprimes :
  ∃! x, isCoprimeWithExactlyN x 7 :=
sorry

end NUMINAMATH_CALUDE_unique_number_with_seven_coprimes_l992_99293


namespace NUMINAMATH_CALUDE_courtyard_length_l992_99200

/-- Calculates the length of a rectangular courtyard given its width, number of bricks, and brick dimensions --/
theorem courtyard_length 
  (width : ℝ) 
  (num_bricks : ℕ) 
  (brick_length : ℝ) 
  (brick_width : ℝ) 
  (h1 : width = 16) 
  (h2 : num_bricks = 20000) 
  (h3 : brick_length = 0.2) 
  (h4 : brick_width = 0.1) : 
  (num_bricks : ℝ) * brick_length * brick_width / width = 25 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_length_l992_99200


namespace NUMINAMATH_CALUDE_complex_square_root_of_negative_one_l992_99264

theorem complex_square_root_of_negative_one (z : ℂ) : 
  (z - 1)^2 = -1 → z = 1 + I ∨ z = 1 - I :=
by sorry

end NUMINAMATH_CALUDE_complex_square_root_of_negative_one_l992_99264


namespace NUMINAMATH_CALUDE_sum_of_fractions_simplest_form_l992_99257

theorem sum_of_fractions : 
  7 / 12 + 11 / 15 = 79 / 60 :=
by sorry

theorem simplest_form : 
  ∀ n m : ℕ, n ≠ 0 → m ≠ 0 → Nat.gcd n m = 1 → (n : ℚ) / m = 79 / 60 → n = 79 ∧ m = 60 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_simplest_form_l992_99257


namespace NUMINAMATH_CALUDE_find_k_l992_99266

theorem find_k (k : ℝ) (h1 : k ≠ 0) : 
  (∀ x : ℝ, (x^2 - k)*(x + k) = x^3 + k*(x^2 - x - 7)) → k = 7 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l992_99266


namespace NUMINAMATH_CALUDE_cube_root_function_l992_99249

theorem cube_root_function (k : ℝ) : 
  (∀ x : ℝ, x > 0 → ∃ y : ℝ, y = k * x^(1/3)) →
  (4 * Real.sqrt 3 = k * 64^(1/3)) →
  (2 * Real.sqrt 3 = k * 8^(1/3)) := by sorry

end NUMINAMATH_CALUDE_cube_root_function_l992_99249


namespace NUMINAMATH_CALUDE_distance_2_neg5_distance_neg2_neg5_distance_1_neg3_solutions_abs_x_plus_1_eq_2_min_value_range_l992_99298

-- Define distance between two points on a number line
def distance (a b : ℝ) : ℝ := |a - b|

-- Theorem 1: Distance between 2 and -5
theorem distance_2_neg5 : distance 2 (-5) = 7 := by sorry

-- Theorem 2: Distance between -2 and -5
theorem distance_neg2_neg5 : distance (-2) (-5) = 3 := by sorry

-- Theorem 3: Distance between 1 and -3
theorem distance_1_neg3 : distance 1 (-3) = 4 := by sorry

-- Theorem 4: Solutions for |x + 1| = 2
theorem solutions_abs_x_plus_1_eq_2 : 
  ∀ x : ℝ, |x + 1| = 2 ↔ x = 1 ∨ x = -3 := by sorry

-- Theorem 5: Range of x for minimum value of |x+1| + |x-2|
theorem min_value_range : 
  ∀ x : ℝ, (∀ y : ℝ, |x+1| + |x-2| ≤ |y+1| + |y-2|) → -1 ≤ x ∧ x ≤ 2 := by sorry

end NUMINAMATH_CALUDE_distance_2_neg5_distance_neg2_neg5_distance_1_neg3_solutions_abs_x_plus_1_eq_2_min_value_range_l992_99298


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l992_99222

theorem trigonometric_equation_solution (z : ℂ) : 
  (Complex.sin z + Complex.sin (2 * z) + Complex.sin (3 * z) = 
   Complex.cos z + Complex.cos (2 * z) + Complex.cos (3 * z)) ↔ 
  (∃ (k : ℤ), z = (2 / 3 : ℂ) * π * (3 * k + 1) ∨ z = (2 / 3 : ℂ) * π * (3 * k - 1)) ∨
  (∃ (n : ℤ), z = (π / 8 : ℂ) * (4 * n + 1)) :=
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l992_99222


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l992_99229

theorem quadratic_minimum_value :
  ∃ (min : ℝ), min = -3 ∧ ∀ x : ℝ, (x - 1)^2 - 3 ≥ min := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l992_99229


namespace NUMINAMATH_CALUDE_remainder_sum_of_powers_l992_99219

theorem remainder_sum_of_powers (n : ℕ) : (8^6 + 7^7 + 6^8) % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_of_powers_l992_99219


namespace NUMINAMATH_CALUDE_simplify_expression_l992_99233

theorem simplify_expression :
  2 - (1 / (2 + Real.sqrt 5)) + (1 / (2 - Real.sqrt 5)) = 2 * (1 - Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l992_99233


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l992_99252

/-- The line l is tangent to the circle C -/
theorem line_tangent_to_circle :
  ∀ (x y : ℝ),
  (x - y + 4 = 0) →
  ((x - 2)^2 + (y - 2)^2 = 8) →
  ∃! (p : ℝ × ℝ), p.1 - p.2 + 4 = 0 ∧ (p.1 - 2)^2 + (p.2 - 2)^2 = 8 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l992_99252


namespace NUMINAMATH_CALUDE_volunteer_distribution_count_l992_99281

/-- The number of volunteers --/
def num_volunteers : ℕ := 7

/-- The number of positions --/
def num_positions : ℕ := 4

/-- The number of ways to choose 2 people from 5 --/
def choose_two_from_five : ℕ := (5 * 4) / (2 * 1)

/-- The number of ways to permute 4 items --/
def permute_four : ℕ := 4 * 3 * 2 * 1

/-- The total number of ways to distribute volunteers when A and B can be in the same group --/
def total_ways : ℕ := choose_two_from_five * permute_four

/-- The number of ways where A and B are in the same position --/
def same_position_ways : ℕ := permute_four

/-- The number of ways for A and B not to serve at the same position --/
def different_position_ways : ℕ := total_ways - same_position_ways

theorem volunteer_distribution_count :
  different_position_ways = 216 :=
sorry

end NUMINAMATH_CALUDE_volunteer_distribution_count_l992_99281


namespace NUMINAMATH_CALUDE_minimal_tile_placement_l992_99221

/-- Represents a tile placement on a grid -/
structure TilePlacement where
  tiles : ℕ
  grid_size : ℕ
  is_valid : Bool

/-- Checks if a tile placement is valid -/
def is_valid_placement (p : TilePlacement) : Prop :=
  p.is_valid ∧ 
  p.grid_size = 8 ∧ 
  p.tiles > 0 ∧ 
  p.tiles ≤ 32 ∧
  ∀ (t : TilePlacement), t.tiles < p.tiles → ¬t.is_valid

theorem minimal_tile_placement : 
  ∃ (p : TilePlacement), is_valid_placement p ∧ p.tiles = 28 := by
  sorry

end NUMINAMATH_CALUDE_minimal_tile_placement_l992_99221


namespace NUMINAMATH_CALUDE_pyramid_volume_l992_99280

/-- The volume of a triangular pyramid with an equilateral base of side length 6√3 and height 9 is 81√3 -/
theorem pyramid_volume : 
  let s : ℝ := 6 * Real.sqrt 3
  let base_area : ℝ := (Real.sqrt 3 / 4) * s^2
  let height : ℝ := 9
  let volume : ℝ := (1/3) * base_area * height
  volume = 81 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_pyramid_volume_l992_99280


namespace NUMINAMATH_CALUDE_rhombus_side_length_l992_99236

/-- The length of a side of a rhombus given one diagonal and its area -/
theorem rhombus_side_length 
  (d1 : ℝ) 
  (area : ℝ) 
  (h1 : d1 = 16) 
  (h2 : area = 327.90242451070714) : 
  ∃ (side : ℝ), abs (side - 37.73592452822641) < 1e-10 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l992_99236


namespace NUMINAMATH_CALUDE_mingyoungs_animals_l992_99267

theorem mingyoungs_animals (chickens ducks rabbits : ℕ) : 
  chickens = 4 * ducks →
  ducks = rabbits + 17 →
  rabbits = 8 →
  chickens + ducks + rabbits = 133 := by
sorry

end NUMINAMATH_CALUDE_mingyoungs_animals_l992_99267


namespace NUMINAMATH_CALUDE_min_cos_sum_acute_angles_l992_99223

theorem min_cos_sum_acute_angles (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan α + Real.tan β = 4 * Real.sin (α + β)) :
  Real.cos (α + β) ≥ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_min_cos_sum_acute_angles_l992_99223


namespace NUMINAMATH_CALUDE_fruit_basket_problem_l992_99253

theorem fruit_basket_problem (total_fruits : ℕ) (mango_count : ℕ) (pear_count : ℕ) (pawpaw_count : ℕ) 
  (h1 : total_fruits = 58)
  (h2 : mango_count = 18)
  (h3 : pear_count = 10)
  (h4 : pawpaw_count = 12) :
  ∃ (lemon_count : ℕ), 
    lemon_count = (total_fruits - (mango_count + pear_count + pawpaw_count)) / 2 ∧ 
    lemon_count = 9 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_problem_l992_99253


namespace NUMINAMATH_CALUDE_cosine_symmetry_l992_99201

/-- A function f is symmetric about the origin if f(-x) = -f(x) for all x -/
def SymmetricAboutOrigin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem cosine_symmetry (φ : ℝ) :
  SymmetricAboutOrigin (fun x ↦ Real.cos (3 * x + φ)) →
  ¬ ∃ k : ℤ, φ = k * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cosine_symmetry_l992_99201


namespace NUMINAMATH_CALUDE_points_separated_by_line_l992_99269

/-- Definition of a line in 2D space --/
def Line (a b c : ℝ) : ℝ × ℝ → ℝ :=
  fun p => a * p.1 + b * p.2 + c

/-- Definition of η for two points with respect to a line --/
def eta (l : ℝ × ℝ → ℝ) (p1 p2 : ℝ × ℝ) : ℝ :=
  (l p1) * (l p2)

/-- Definition of two points being separated by a line --/
def separatedByLine (l : ℝ × ℝ → ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  eta l p1 p2 < 0

/-- Theorem: Points A(1,2) and B(-1,0) are separated by the line x+y-1=0 --/
theorem points_separated_by_line :
  let l := Line 1 1 (-1)
  let A := (1, 2)
  let B := (-1, 0)
  separatedByLine l A B := by
  sorry


end NUMINAMATH_CALUDE_points_separated_by_line_l992_99269


namespace NUMINAMATH_CALUDE_problem_solution_l992_99237

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x - a / x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + a * x - 6 * log x

def h (m : ℝ) (x : ℝ) : ℝ := x^2 - m * x + 4

theorem problem_solution :
  -- Part I
  (∀ a x, x > 0 → 
    (a ≥ 0 → (deriv (f a)) x > 0) ∧ 
    (a < 0 → ((0 < x ∧ x < -a) → (deriv (f a)) x < 0) ∧ 
             (x > -a → (deriv (f a)) x > 0))) ∧
  -- Part II
  (∀ a, (∀ x, x > 0 → (deriv (g a)) x ≥ 0) → a ≥ 5/2) ∧
  -- Part III
  (∀ m, (∃ x₁, 0 < x₁ ∧ x₁ < 1 ∧ 
        ∀ x₂, 1 ≤ x₂ ∧ x₂ ≤ 2 → g 2 x₁ ≥ h m x₂) → 
        m ≥ 8 - 5 * log 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l992_99237


namespace NUMINAMATH_CALUDE_inequality_implies_a_positive_l992_99254

theorem inequality_implies_a_positive (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-2) (-1) → x^2 + x + a > 0) →
  a > 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_a_positive_l992_99254


namespace NUMINAMATH_CALUDE_chocolate_probability_l992_99226

/-- Represents a chocolate bar with dark and white segments -/
structure ChocolateBar :=
  (segments : List (Float × Bool))  -- List of (length, isDark) pairs

/-- The process of cutting and switching chocolate bars -/
def cutAndSwitch (p : Float) (bar1 bar2 : ChocolateBar) : ChocolateBar × ChocolateBar :=
  sorry

/-- Checks if the chocolate at 1/3 and 2/3 are the same type -/
def sameTypeAt13And23 (bar : ChocolateBar) : Bool :=
  sorry

/-- Performs the cutting and switching process for n steps -/
def processSteps (n : Nat) (bar1 bar2 : ChocolateBar) : ChocolateBar × ChocolateBar :=
  sorry

/-- The probability of getting the same type at 1/3 and 2/3 after n steps -/
def probabilitySameType (n : Nat) : Float :=
  sorry

theorem chocolate_probability :
  probabilitySameType 100 = 1/2 * (1 + (1/3)^100) :=
sorry

end NUMINAMATH_CALUDE_chocolate_probability_l992_99226


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l992_99285

theorem quadratic_root_difference (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (∃ x y : ℝ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ |x - y| = 1) →
  p = Real.sqrt (4*q + 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l992_99285


namespace NUMINAMATH_CALUDE_scooter_distance_l992_99220

/-- Proves that a scooter traveling 5/8 as fast as a motorcycle going 96 miles per hour will cover 40 miles in 40 minutes. -/
theorem scooter_distance (motorcycle_speed : ℝ) (scooter_ratio : ℝ) (travel_time : ℝ) :
  motorcycle_speed = 96 →
  scooter_ratio = 5/8 →
  travel_time = 40/60 →
  scooter_ratio * motorcycle_speed * travel_time = 40 :=
by sorry

end NUMINAMATH_CALUDE_scooter_distance_l992_99220


namespace NUMINAMATH_CALUDE_probability_three_games_probability_best_of_five_l992_99214

-- Define the probability of A winning a single game
def p_A : ℚ := 2/3

-- Define the probability of B winning a single game
def p_B : ℚ := 1/3

-- Theorem for part (1)
theorem probability_three_games 
  (h1 : p_A + p_B = 1) 
  (h2 : p_A = 2/3) 
  (h3 : p_B = 1/3) :
  let p_A_wins_two := 3 * (p_A^2 * p_B)
  let p_B_wins_at_least_one := 1 - p_A^3
  (p_A_wins_two = 4/9) ∧ (p_B_wins_at_least_one = 19/27) := by
  sorry

-- Theorem for part (2)
theorem probability_best_of_five
  (h1 : p_A + p_B = 1)
  (h2 : p_A = 2/3)
  (h3 : p_B = 1/3) :
  let p_A_wins_three_straight := p_A^3
  let p_A_wins_in_four := 3 * (p_A^3 * p_B)
  let p_A_wins_in_five := 6 * (p_A^3 * p_B^2)
  p_A_wins_three_straight + p_A_wins_in_four + p_A_wins_in_five = 64/81 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_games_probability_best_of_five_l992_99214


namespace NUMINAMATH_CALUDE_min_even_integers_l992_99299

theorem min_even_integers (x y z a b c m n o : ℤ) : 
  x + y + z = 30 →
  x + y + z + a + b + c = 55 →
  x + y + z + a + b + c + m + n + o = 88 →
  ∃ (count : ℕ), count = (if Even x then 1 else 0) + 
                         (if Even y then 1 else 0) + 
                         (if Even z then 1 else 0) + 
                         (if Even a then 1 else 0) + 
                         (if Even b then 1 else 0) + 
                         (if Even c then 1 else 0) + 
                         (if Even m then 1 else 0) + 
                         (if Even n then 1 else 0) + 
                         (if Even o then 1 else 0) ∧
  count ≥ 1 ∧
  ∀ (other_count : ℕ), other_count ≥ count →
    ∃ (x' y' z' a' b' c' m' n' o' : ℤ),
      x' + y' + z' = 30 ∧
      x' + y' + z' + a' + b' + c' = 55 ∧
      x' + y' + z' + a' + b' + c' + m' + n' + o' = 88 ∧
      other_count = (if Even x' then 1 else 0) + 
                    (if Even y' then 1 else 0) + 
                    (if Even z' then 1 else 0) + 
                    (if Even a' then 1 else 0) + 
                    (if Even b' then 1 else 0) + 
                    (if Even c' then 1 else 0) + 
                    (if Even m' then 1 else 0) + 
                    (if Even n' then 1 else 0) + 
                    (if Even o' then 1 else 0) :=
by
  sorry

end NUMINAMATH_CALUDE_min_even_integers_l992_99299


namespace NUMINAMATH_CALUDE_strictly_increasing_function_property_l992_99245

/-- A function f: ℝ → ℝ that satisfies the given conditions -/
def StrictlyIncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 0

theorem strictly_increasing_function_property
  (f : ℝ → ℝ)
  (h : StrictlyIncreasingFunction f)
  (h1 : f 5 = -1)
  (h2 : f 7 = 0) :
  f (-3) < -1 := by
  sorry

end NUMINAMATH_CALUDE_strictly_increasing_function_property_l992_99245


namespace NUMINAMATH_CALUDE_unique_base_eight_l992_99270

/-- Converts a list of digits in base b to its decimal representation -/
def toDecimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldl (fun acc d => acc * b + d) 0

/-- Checks if the equation 243₍ᵦ₎ + 152₍ᵦ₎ = 415₍ᵦ₎ holds for a given base b -/
def equationHolds (b : Nat) : Prop :=
  toDecimal [2, 4, 3] b + toDecimal [1, 5, 2] b = toDecimal [4, 1, 5] b

theorem unique_base_eight :
  ∃! b, b > 5 ∧ equationHolds b :=
sorry

end NUMINAMATH_CALUDE_unique_base_eight_l992_99270
