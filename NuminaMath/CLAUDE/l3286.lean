import Mathlib

namespace NUMINAMATH_CALUDE_max_value_theorem_l3286_328627

theorem max_value_theorem (x y : ℝ) (h : x * y > 0) :
  ∃ (max : ℝ), max = 4 - 2 * Real.sqrt 2 ∧
  ∀ (z : ℝ), z = x / (x + y) + 2 * y / (x + 2 * y) → z ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3286_328627


namespace NUMINAMATH_CALUDE_solution_range_l3286_328643

theorem solution_range (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (1 - m) * x = 2 - 3 * x) → m < 4 := by
sorry

end NUMINAMATH_CALUDE_solution_range_l3286_328643


namespace NUMINAMATH_CALUDE_square_side_length_l3286_328662

theorem square_side_length (s : ℝ) : s^2 + s - 4*s = 4 → s = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3286_328662


namespace NUMINAMATH_CALUDE_partnership_profit_difference_l3286_328665

/-- Given a partnership scenario with specific investments and profit-sharing rules, 
    calculate the difference in profit shares between two partners. -/
theorem partnership_profit_difference 
  (john_investment mike_investment : ℚ)
  (total_profit : ℚ)
  (effort_share investment_share : ℚ)
  (h1 : john_investment = 700)
  (h2 : mike_investment = 300)
  (h3 : total_profit = 3000.0000000000005)
  (h4 : effort_share = 1/3)
  (h5 : investment_share = 2/3)
  (h6 : effort_share + investment_share = 1) :
  let total_investment := john_investment + mike_investment
  let john_investment_ratio := john_investment / total_investment
  let mike_investment_ratio := mike_investment / total_investment
  let john_share := (effort_share * total_profit / 2) + 
                    (investment_share * total_profit * john_investment_ratio)
  let mike_share := (effort_share * total_profit / 2) + 
                    (investment_share * total_profit * mike_investment_ratio)
  john_share - mike_share = 800.0000000000001 := by
  sorry


end NUMINAMATH_CALUDE_partnership_profit_difference_l3286_328665


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l3286_328683

theorem vector_sum_magnitude (a b : ℝ × ℝ) : 
  ‖a - 2 • b‖ = 1 → a • b = 1 → ‖a + 2 • b‖ = 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l3286_328683


namespace NUMINAMATH_CALUDE_y_minus_x_value_l3286_328632

theorem y_minus_x_value (x y : ℝ) (hx : |x| = 5) (hy : |y| = 9) (hxy : x < y) :
  y - x = 4 ∨ y - x = 14 := by
  sorry

end NUMINAMATH_CALUDE_y_minus_x_value_l3286_328632


namespace NUMINAMATH_CALUDE_alcohol_bottle_problem_l3286_328668

/-- The amount of alcohol originally in the bottle -/
def original_amount : ℝ := 750

/-- The amount poured back in after the first pour -/
def amount_added : ℝ := 40

/-- The amount poured out in the third pour -/
def third_pour : ℝ := 180

/-- The amount remaining after all pours -/
def final_amount : ℝ := 60

theorem alcohol_bottle_problem :
  let first_pour := original_amount * (1/3)
  let after_first_pour := original_amount - first_pour + amount_added
  let second_pour := after_first_pour * (5/9)
  let after_second_pour := after_first_pour - second_pour
  after_second_pour - third_pour = final_amount :=
sorry


end NUMINAMATH_CALUDE_alcohol_bottle_problem_l3286_328668


namespace NUMINAMATH_CALUDE_gcd_8_factorial_12_factorial_l3286_328630

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem gcd_8_factorial_12_factorial :
  Nat.gcd (factorial 8) (factorial 12) = factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8_factorial_12_factorial_l3286_328630


namespace NUMINAMATH_CALUDE_other_root_of_equation_l3286_328607

theorem other_root_of_equation (m : ℤ) : 
  (∃ x : ℤ, x^2 - 3*x - m = 0 ∧ x = ⌊Real.sqrt 6⌋) →
  (∃ y : ℤ, y^2 - 3*y - m = 0 ∧ y ≠ ⌊Real.sqrt 6⌋ ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_equation_l3286_328607


namespace NUMINAMATH_CALUDE_least_multiple_of_25_greater_than_450_l3286_328651

theorem least_multiple_of_25_greater_than_450 : 
  ∀ n : ℕ, n * 25 > 450 → n * 25 ≥ 475 :=
sorry

end NUMINAMATH_CALUDE_least_multiple_of_25_greater_than_450_l3286_328651


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3286_328628

theorem quadratic_inequality_solution (x : ℝ) :
  -10 * x^2 + 6 * x + 8 < 0 ↔ -0.64335 < x ∧ x < 1.24335 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3286_328628


namespace NUMINAMATH_CALUDE_greatest_common_multiple_under_150_l3286_328621

theorem greatest_common_multiple_under_150 :
  ∃ (n : ℕ), n = 120 ∧ 
  n % 15 = 0 ∧ 
  n % 20 = 0 ∧ 
  n < 150 ∧ 
  ∀ (m : ℕ), m % 15 = 0 → m % 20 = 0 → m < 150 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_under_150_l3286_328621


namespace NUMINAMATH_CALUDE_louis_oranges_l3286_328657

/-- Given the fruit distribution among Louis, Samantha, and Marley, prove that Louis has 5 oranges. -/
theorem louis_oranges :
  ∀ (louis_oranges louis_apples samantha_oranges samantha_apples marley_oranges marley_apples : ℕ),
  louis_apples = 3 →
  samantha_oranges = 8 →
  samantha_apples = 7 →
  marley_oranges = 2 * louis_oranges →
  marley_apples = 3 * samantha_apples →
  marley_oranges + marley_apples = 31 →
  louis_oranges = 5 := by
sorry


end NUMINAMATH_CALUDE_louis_oranges_l3286_328657


namespace NUMINAMATH_CALUDE_johnny_fish_count_l3286_328625

theorem johnny_fish_count (total : ℕ) (sony_multiplier : ℕ) (johnny_count : ℕ) : 
  total = 40 →
  sony_multiplier = 4 →
  total = johnny_count + sony_multiplier * johnny_count →
  johnny_count = 8 := by
sorry

end NUMINAMATH_CALUDE_johnny_fish_count_l3286_328625


namespace NUMINAMATH_CALUDE_power_difference_over_sum_l3286_328603

theorem power_difference_over_sum : (3^2016 - 3^2014) / (3^2016 + 3^2014) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_over_sum_l3286_328603


namespace NUMINAMATH_CALUDE_equation_linear_iff_a_eq_neg_two_l3286_328663

/-- The equation (a-2)x^(|a|^(-1)+3) = 0 is linear in x if and only if a = -2 -/
theorem equation_linear_iff_a_eq_neg_two (a : ℝ) :
  (∀ x, ∃ b c : ℝ, (a - 2) * x^(|a|⁻¹ + 3) = b * x + c) ↔ a = -2 :=
sorry

end NUMINAMATH_CALUDE_equation_linear_iff_a_eq_neg_two_l3286_328663


namespace NUMINAMATH_CALUDE_initial_depth_calculation_l3286_328686

theorem initial_depth_calculation (men_initial : ℕ) (hours_initial : ℕ) (men_extra : ℕ) (hours_final : ℕ) (depth_final : ℕ) :
  men_initial = 75 →
  hours_initial = 8 →
  men_extra = 65 →
  hours_final = 6 →
  depth_final = 70 →
  ∃ (depth_initial : ℕ), 
    (men_initial * hours_initial * depth_final = (men_initial + men_extra) * hours_final * depth_initial) ∧
    depth_initial = 50 := by
  sorry

#check initial_depth_calculation

end NUMINAMATH_CALUDE_initial_depth_calculation_l3286_328686


namespace NUMINAMATH_CALUDE_negative_sum_positive_product_l3286_328636

theorem negative_sum_positive_product (a b : ℝ) : 
  a + b < 0 → ab > 0 → a < 0 ∧ b < 0 := by sorry

end NUMINAMATH_CALUDE_negative_sum_positive_product_l3286_328636


namespace NUMINAMATH_CALUDE_arrangement_count_l3286_328676

/-- The number of people in the row -/
def n : ℕ := 5

/-- The number of arrangements where A and B are adjacent -/
def adjacent_arrangements : ℕ := 48

/-- The total number of arrangements of n people -/
def total_arrangements (n : ℕ) : ℕ := n.factorial

/-- The number of arrangements where A and B are not adjacent -/
def non_adjacent_arrangements (n : ℕ) : ℕ := total_arrangements n - adjacent_arrangements

/-- The number of arrangements where A and B are not adjacent and A is to the left of B -/
def target_arrangements (n : ℕ) : ℕ := non_adjacent_arrangements n / 2

theorem arrangement_count : target_arrangements n = 36 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l3286_328676


namespace NUMINAMATH_CALUDE_value_of_b_l3286_328647

theorem value_of_b (a c b : ℝ) : 
  a = 105 → 
  c = 70 → 
  a^4 = 21 * 25 * 15 * b * c^3 → 
  b = 0.045 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_l3286_328647


namespace NUMINAMATH_CALUDE_flour_spill_ratio_l3286_328672

def initial_flour : ℕ := 500
def used_flour : ℕ := 240
def needed_flour : ℕ := 370

theorem flour_spill_ratio :
  let flour_after_baking := initial_flour - used_flour
  let flour_after_spill := initial_flour - needed_flour
  let spilled_flour := flour_after_baking - flour_after_spill
  (spilled_flour : ℚ) / flour_after_baking = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_flour_spill_ratio_l3286_328672


namespace NUMINAMATH_CALUDE_min_value_expression_l3286_328670

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 64) :
  x^2 + 4*x*y + 4*y^2 + 4*z^2 ≥ 192 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 64 ∧ x₀^2 + 4*x₀*y₀ + 4*y₀^2 + 4*z₀^2 = 192 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3286_328670


namespace NUMINAMATH_CALUDE_non_parallel_diagonals_32gon_l3286_328654

/-- The number of diagonals not parallel to any side in a regular n-gon -/
def non_parallel_diagonals (n : ℕ) : ℕ :=
  let total_diagonals := n * (n - 3) / 2
  let parallel_pairs := n / 2
  let diagonals_per_pair := (n - 4) / 2
  let parallel_diagonals := parallel_pairs * diagonals_per_pair
  total_diagonals - parallel_diagonals

/-- Theorem: In a regular 32-gon, the number of diagonals not parallel to any of its sides is 240 -/
theorem non_parallel_diagonals_32gon :
  non_parallel_diagonals 32 = 240 := by
  sorry


end NUMINAMATH_CALUDE_non_parallel_diagonals_32gon_l3286_328654


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3286_328629

-- Define the sets P and Q
def P : Set ℝ := {x | x < 1}
def Q : Set ℝ := {x | x^2 < 4}

-- State the theorem
theorem intersection_of_P_and_Q : P ∩ Q = {x : ℝ | -2 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3286_328629


namespace NUMINAMATH_CALUDE_major_axis_length_l3286_328692

/-- Represents a right circular cylinder --/
structure RightCircularCylinder where
  radius : ℝ

/-- Represents an ellipse formed by intersecting a plane with a cylinder --/
structure Ellipse where
  minorAxis : ℝ
  majorAxis : ℝ

/-- The ellipse formed by intersecting a plane with a right circular cylinder --/
def cylinderEllipse (c : RightCircularCylinder) : Ellipse :=
  { minorAxis := 2 * c.radius,
    majorAxis := 3 * c.radius }

theorem major_axis_length (c : RightCircularCylinder) 
  (h : c.radius = 1) :
  (cylinderEllipse c).majorAxis = 3 ∧
  (cylinderEllipse c).majorAxis = 1.5 * (cylinderEllipse c).minorAxis :=
by sorry

end NUMINAMATH_CALUDE_major_axis_length_l3286_328692


namespace NUMINAMATH_CALUDE_dvd_money_calculation_l3286_328694

/-- Given the cost of one pack of DVDs and the number of packs that can be bought,
    calculate the total amount of money available. -/
theorem dvd_money_calculation (cost_per_pack : ℕ) (num_packs : ℕ) :
  cost_per_pack = 12 → num_packs = 11 → cost_per_pack * num_packs = 132 := by
  sorry

end NUMINAMATH_CALUDE_dvd_money_calculation_l3286_328694


namespace NUMINAMATH_CALUDE_customer_satisfaction_probability_l3286_328671

-- Define the probability of a customer being satisfied
def p : ℝ := sorry

-- Define the conditions
def dissatisfied_review_rate : ℝ := 0.80
def satisfied_review_rate : ℝ := 0.15
def angry_reviews : ℕ := 60
def positive_reviews : ℕ := 20

-- Theorem statement
theorem customer_satisfaction_probability :
  dissatisfied_review_rate * (1 - p) * (angry_reviews + positive_reviews) = angry_reviews ∧
  satisfied_review_rate * p * (angry_reviews + positive_reviews) = positive_reviews →
  p = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_customer_satisfaction_probability_l3286_328671


namespace NUMINAMATH_CALUDE_exists_n_power_half_eq_twenty_l3286_328688

theorem exists_n_power_half_eq_twenty :
  ∃ n : ℝ, n > 0 ∧ n^(n/2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_exists_n_power_half_eq_twenty_l3286_328688


namespace NUMINAMATH_CALUDE_spoonfuls_per_bowl_l3286_328646

/-- Proves that the number of spoonfuls in each bowl is 25 -/
theorem spoonfuls_per_bowl
  (clusters_per_spoonful : ℕ)
  (clusters_per_box : ℕ)
  (bowls_per_box : ℕ)
  (h1 : clusters_per_spoonful = 4)
  (h2 : clusters_per_box = 500)
  (h3 : bowls_per_box = 5) :
  clusters_per_box / (bowls_per_box * clusters_per_spoonful) = 25 := by
  sorry

end NUMINAMATH_CALUDE_spoonfuls_per_bowl_l3286_328646


namespace NUMINAMATH_CALUDE_max_non_managers_l3286_328695

theorem max_non_managers (num_managers : ℕ) (ratio_managers : ℚ) (ratio_non_managers : ℚ) :
  num_managers = 8 →
  ratio_managers / ratio_non_managers > 7 / 24 →
  ∃ (max_non_managers : ℕ),
    (↑num_managers : ℚ) / (↑max_non_managers : ℚ) > ratio_managers / ratio_non_managers ∧
    ∀ (n : ℕ), n > max_non_managers →
      (↑num_managers : ℚ) / (↑n : ℚ) ≤ ratio_managers / ratio_non_managers →
      max_non_managers = 27 :=
by sorry

end NUMINAMATH_CALUDE_max_non_managers_l3286_328695


namespace NUMINAMATH_CALUDE_mobile_profit_percentage_l3286_328680

-- Define the given values
def grinder_cost : ℝ := 15000
def mobile_cost : ℝ := 8000
def grinder_loss_percentage : ℝ := 0.02
def overall_profit : ℝ := 500

-- Define the theorem
theorem mobile_profit_percentage :
  let grinder_selling_price := grinder_cost * (1 - grinder_loss_percentage)
  let total_cost := grinder_cost + mobile_cost
  let total_selling_price := total_cost + overall_profit
  let mobile_selling_price := total_selling_price - grinder_selling_price
  let mobile_profit := mobile_selling_price - mobile_cost
  (mobile_profit / mobile_cost) * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_mobile_profit_percentage_l3286_328680


namespace NUMINAMATH_CALUDE_rotate_point_A_about_C_l3286_328698

-- Define the rotation function
def rotate90ClockwiseAboutPoint (p center : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  let (cx, cy) := center
  (cx + (y - cy), cy - (x - cx))

-- Theorem statement
theorem rotate_point_A_about_C :
  let A : ℝ × ℝ := (-3, 2)
  let C : ℝ × ℝ := (-2, 2)
  rotate90ClockwiseAboutPoint A C = (-2, 3) := by sorry

end NUMINAMATH_CALUDE_rotate_point_A_about_C_l3286_328698


namespace NUMINAMATH_CALUDE_julian_needs_80_more_legos_l3286_328615

/-- The number of legos Julian has -/
def julian_legos : ℕ := 400

/-- The number of airplane models Julian wants to make -/
def num_airplanes : ℕ := 2

/-- The number of legos required for each airplane model -/
def legos_per_airplane : ℕ := 240

/-- The number of additional legos Julian needs -/
def additional_legos : ℕ := num_airplanes * legos_per_airplane - julian_legos

theorem julian_needs_80_more_legos : additional_legos = 80 :=
by sorry

end NUMINAMATH_CALUDE_julian_needs_80_more_legos_l3286_328615


namespace NUMINAMATH_CALUDE_fourth_root_simplification_l3286_328633

theorem fourth_root_simplification : Real.sqrt (Real.sqrt (2^8 * 3^4 * 11^0)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_simplification_l3286_328633


namespace NUMINAMATH_CALUDE_two_digit_square_sum_l3286_328673

/-- Two-digit integer -/
def TwoDigitInt (x : ℕ) : Prop := 10 ≤ x ∧ x < 100

/-- Reverse digits of a two-digit integer -/
def reverseDigits (x : ℕ) : ℕ := 
  let tens := x / 10
  let ones := x % 10
  10 * ones + tens

theorem two_digit_square_sum (x y n : ℕ) : 
  TwoDigitInt x → TwoDigitInt y → y = reverseDigits x → x^2 + y^2 = n^2 → x + y + n = 264 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_square_sum_l3286_328673


namespace NUMINAMATH_CALUDE_apple_distribution_l3286_328674

theorem apple_distribution (x : ℕ) : 
  x > 0 →
  x - x / 5 - x / 12 - x / 8 - x / 20 - x / 4 - x / 7 - x / 30 - 4 * (x / 30) - 300 ≤ 50 →
  x = 3360 := by
sorry

end NUMINAMATH_CALUDE_apple_distribution_l3286_328674


namespace NUMINAMATH_CALUDE_multivariable_jensen_inequality_l3286_328600

/-- A function F: ℝⁿ → ℝ is convex if for any two points x and y in ℝⁿ and weights q₁, q₂ ≥ 0 with q₁ + q₂ = 1,
    F(q₁x + q₂y) ≤ q₁F(x) + q₂F(y) -/
def IsConvex (n : ℕ) (F : (Fin n → ℝ) → ℝ) : Prop :=
  ∀ (x y : Fin n → ℝ) (q₁ q₂ : ℝ), q₁ ≥ 0 → q₂ ≥ 0 → q₁ + q₂ = 1 →
    F (fun i => q₁ * x i + q₂ * y i) ≤ q₁ * F x + q₂ * F y

/-- Jensen's inequality for multivariable convex functions -/
theorem multivariable_jensen_inequality {n : ℕ} (F : (Fin n → ℝ) → ℝ) (h_convex : IsConvex n F)
    (x y : Fin n → ℝ) (q₁ q₂ : ℝ) (hq₁ : q₁ ≥ 0) (hq₂ : q₂ ≥ 0) (hsum : q₁ + q₂ = 1) :
    F (fun i => q₁ * x i + q₂ * y i) ≤ q₁ * F x + q₂ * F y := by
  sorry

end NUMINAMATH_CALUDE_multivariable_jensen_inequality_l3286_328600


namespace NUMINAMATH_CALUDE_rectangle_perimeter_and_area_l3286_328699

theorem rectangle_perimeter_and_area :
  ∀ (length width perimeter area : ℝ),
    length = 10 →
    width = length - 3 →
    perimeter = 2 * (length + width) →
    area = length * width →
    perimeter = 34 ∧ area = 70 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_and_area_l3286_328699


namespace NUMINAMATH_CALUDE_tan_fifteen_ratio_equals_sqrt_three_l3286_328611

theorem tan_fifteen_ratio_equals_sqrt_three :
  (1 + Real.tan (15 * π / 180)) / (1 - Real.tan (15 * π / 180)) = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_tan_fifteen_ratio_equals_sqrt_three_l3286_328611


namespace NUMINAMATH_CALUDE_xy_max_value_l3286_328691

theorem xy_max_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2*y = 16) :
  x * y ≤ 32 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + 2*y = 16 ∧ x * y = 32 :=
sorry

end NUMINAMATH_CALUDE_xy_max_value_l3286_328691


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3286_328689

def is_in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

theorem point_in_fourth_quadrant :
  is_in_fourth_quadrant 3 (-4) :=
sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3286_328689


namespace NUMINAMATH_CALUDE_ratio_equality_l3286_328610

theorem ratio_equality (a b c x y z : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0)
  (h_abc : a^2 + b^2 + c^2 = 49)
  (h_xyz : x^2 + y^2 + z^2 = 64)
  (h_dot : a*x + b*y + c*z = 56) :
  (a + b + c) / (x + y + z) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l3286_328610


namespace NUMINAMATH_CALUDE_tan_sum_specific_l3286_328677

theorem tan_sum_specific (a b : Real) 
  (ha : Real.tan a = 1/2) (hb : Real.tan b = 1/3) : 
  Real.tan (a + b) = 1 := by sorry

end NUMINAMATH_CALUDE_tan_sum_specific_l3286_328677


namespace NUMINAMATH_CALUDE_workshop_average_salary_l3286_328652

theorem workshop_average_salary
  (num_technicians : ℕ)
  (num_total_workers : ℕ)
  (avg_salary_technicians : ℚ)
  (avg_salary_others : ℚ)
  (h1 : num_technicians = 7)
  (h2 : num_total_workers = 56)
  (h3 : avg_salary_technicians = 12000)
  (h4 : avg_salary_others = 6000) :
  let num_other_workers := num_total_workers - num_technicians
  let total_salary := num_technicians * avg_salary_technicians + num_other_workers * avg_salary_others
  total_salary / num_total_workers = 6750 :=
sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l3286_328652


namespace NUMINAMATH_CALUDE_max_k_value_l3286_328697

theorem max_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 72) →
  k ≤ 2 * Real.sqrt 26 :=
by sorry

end NUMINAMATH_CALUDE_max_k_value_l3286_328697


namespace NUMINAMATH_CALUDE_tree_space_calculation_l3286_328612

/-- Given a road of length 166 feet where 16 trees are planted with 10 feet between each tree,
    prove that each tree occupies 1 square foot of sidewalk space. -/
theorem tree_space_calculation (road_length : ℝ) (num_trees : ℕ) (space_between : ℝ) : 
  road_length = 166 ∧ num_trees = 16 ∧ space_between = 10 → 
  (road_length - space_between * (num_trees - 1)) / num_trees = 1 :=
by sorry

end NUMINAMATH_CALUDE_tree_space_calculation_l3286_328612


namespace NUMINAMATH_CALUDE_inequality_system_solutions_l3286_328641

def has_exactly_two_integer_solutions (m : ℝ) : Prop :=
  ∃! (x y : ℤ), (x < 1 ∧ x > m - 1) ∧ (y < 1 ∧ y > m - 1) ∧ x ≠ y

theorem inequality_system_solutions (m : ℝ) :
  has_exactly_two_integer_solutions m ↔ -1 ≤ m ∧ m < 0 :=
sorry

end NUMINAMATH_CALUDE_inequality_system_solutions_l3286_328641


namespace NUMINAMATH_CALUDE_purely_imaginary_modulus_l3286_328658

theorem purely_imaginary_modulus (a : ℝ) :
  let z : ℂ := a + Complex.I
  (z.re = 0) → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_modulus_l3286_328658


namespace NUMINAMATH_CALUDE_new_average_weight_l3286_328623

theorem new_average_weight 
  (initial_students : ℕ) 
  (initial_average : ℚ) 
  (new_student_weight : ℚ) : 
  initial_students = 29 →
  initial_average = 28 →
  new_student_weight = 13 →
  (initial_students * initial_average + new_student_weight) / (initial_students + 1) = 27.5 := by
  sorry

end NUMINAMATH_CALUDE_new_average_weight_l3286_328623


namespace NUMINAMATH_CALUDE_fifth_number_is_24_l3286_328649

/-- Definition of the sequence function -/
def f (n : ℕ) : ℕ := n^2 - 1

/-- Theorem stating that the fifth number in the sequence is 24 -/
theorem fifth_number_is_24 : f 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_fifth_number_is_24_l3286_328649


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3286_328619

theorem geometric_sequence_product (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n = a 1 * q ^ (n - 1)) →  -- Definition of geometric sequence
  a 1 = 1 →                        -- First term is 1
  a 5 = 16 →                       -- Last term is 16
  a 2 * a 3 * a 4 = 64 :=           -- Product of middle three terms is 64
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3286_328619


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3286_328614

-- Define the property that f must satisfy
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, f (x * y + z) = f x * f y + f z

-- State the theorem
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesProperty f →
    (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3286_328614


namespace NUMINAMATH_CALUDE_total_applications_eq_600_l3286_328602

def in_state_applications : ℕ := 200

def out_state_applications : ℕ := 2 * in_state_applications

def total_applications : ℕ := in_state_applications + out_state_applications

theorem total_applications_eq_600 : total_applications = 600 := by
  sorry

end NUMINAMATH_CALUDE_total_applications_eq_600_l3286_328602


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l3286_328696

theorem ellipse_major_axis_length :
  let ellipse := {(x, y) : ℝ × ℝ | x^2 / 9 + y^2 / 4 = 1}
  ∃ a b : ℝ, a > b ∧ a > 0 ∧ b > 0 ∧
    ellipse = {(x, y) : ℝ × ℝ | x^2 / a^2 + y^2 / b^2 = 1} ∧
    2 * a = 6 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l3286_328696


namespace NUMINAMATH_CALUDE_fifth_reading_calculation_l3286_328675

theorem fifth_reading_calculation (r1 r2 r3 r4 : ℝ) (mean : ℝ) (h1 : r1 = 2) (h2 : r2 = 2.1) (h3 : r3 = 2) (h4 : r4 = 2.2) (h_mean : mean = 2) :
  ∃ r5 : ℝ, (r1 + r2 + r3 + r4 + r5) / 5 = mean ∧ r5 = 1.7 :=
by sorry

end NUMINAMATH_CALUDE_fifth_reading_calculation_l3286_328675


namespace NUMINAMATH_CALUDE_sin_cos_equation_solution_l3286_328644

theorem sin_cos_equation_solution (x : Real) 
  (h1 : x ∈ Set.Icc 0 Real.pi) 
  (h2 : Real.sin (x + Real.sin x) = Real.cos (x - Real.cos x)) : 
  x = Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_equation_solution_l3286_328644


namespace NUMINAMATH_CALUDE_vector_equation_holds_l3286_328660

def vector2D := ℝ × ℝ

def dot_product (v w : vector2D) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def scale_vector (s : ℝ) (v : vector2D) : vector2D :=
  (s * v.1, s * v.2)

theorem vector_equation_holds (a c : vector2D) (b : vector2D) : 
  a = (1, 1) → c = (2, 2) → 
  scale_vector (dot_product a b) c = scale_vector (dot_product b c) a := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_holds_l3286_328660


namespace NUMINAMATH_CALUDE_sin_2x_derivative_l3286_328606

theorem sin_2x_derivative (f : ℝ → ℝ) (x : ℝ) :
  (∀ x, f x = Real.sin (2 * x)) →
  (deriv f) x = 2 * Real.cos (2 * x) := by
sorry

end NUMINAMATH_CALUDE_sin_2x_derivative_l3286_328606


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3286_328609

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (3 - 4 * z) = 7 :=
by
  use -23/2
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3286_328609


namespace NUMINAMATH_CALUDE_smallest_addend_to_palindrome_l3286_328667

/-- A function that checks if a positive integer is a palindrome -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- The smallest positive integer that can be added to 2002 to produce a larger palindrome -/
def smallestAddend : ℕ := 110

theorem smallest_addend_to_palindrome : 
  (isPalindrome 2002) ∧ 
  (isPalindrome (2002 + smallestAddend)) ∧ 
  (∀ k : ℕ, k < smallestAddend → ¬ isPalindrome (2002 + k)) := by sorry

end NUMINAMATH_CALUDE_smallest_addend_to_palindrome_l3286_328667


namespace NUMINAMATH_CALUDE_same_number_probability_l3286_328693

/-- The upper bound for the selected numbers -/
def upperBound : ℕ := 300

/-- Billy's number is a multiple of this value -/
def billyMultiple : ℕ := 36

/-- Bobbi's number is a multiple of this value -/
def bobbiMultiple : ℕ := 48

/-- The probability of Billy and Bobbi selecting the same number -/
def sameProbability : ℚ := 1 / 24

theorem same_number_probability :
  (∃ (b₁ b₂ : ℕ), b₁ > 0 ∧ b₂ > 0 ∧ b₁ < upperBound ∧ b₂ < upperBound ∧
    b₁ % billyMultiple = 0 ∧ b₂ % bobbiMultiple = 0) →
  (∃ (n : ℕ), n > 0 ∧ n < upperBound ∧ n % billyMultiple = 0 ∧ n % bobbiMultiple = 0) →
  sameProbability = (Nat.card {n : ℕ | n > 0 ∧ n < upperBound ∧ n % billyMultiple = 0 ∧ n % bobbiMultiple = 0} : ℚ) /
                    ((Nat.card {n : ℕ | n > 0 ∧ n < upperBound ∧ n % billyMultiple = 0} : ℚ) *
                     (Nat.card {n : ℕ | n > 0 ∧ n < upperBound ∧ n % bobbiMultiple = 0} : ℚ)) :=
by sorry

end NUMINAMATH_CALUDE_same_number_probability_l3286_328693


namespace NUMINAMATH_CALUDE_quadratic_equation_k_l3286_328639

/-- The equation (k-1)x^(|k|+1)-x+5=0 is quadratic in x -/
def is_quadratic (k : ℝ) : Prop :=
  (k - 1 ≠ 0) ∧ (|k| + 1 = 2)

theorem quadratic_equation_k (k : ℝ) :
  is_quadratic k → k = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_k_l3286_328639


namespace NUMINAMATH_CALUDE_inverse_variation_cube_and_sqrt_l3286_328661

theorem inverse_variation_cube_and_sqrt (k : ℝ) :
  (∀ x > 0, x^3 * Real.sqrt x = k) →
  (4^3 * Real.sqrt 4 = 2 * k) →
  (16^3 * Real.sqrt 16 = 128 * k) := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_cube_and_sqrt_l3286_328661


namespace NUMINAMATH_CALUDE_work_efficiency_ratio_l3286_328608

-- Define the work efficiencies of A and B
def work_efficiency_A : ℚ := 1 / 45
def work_efficiency_B : ℚ := 1 / 22.5

-- Define the combined work time
def combined_work_time : ℚ := 15

-- Define B's individual work time
def B_work_time : ℚ := 22.5

-- Theorem statement
theorem work_efficiency_ratio :
  (work_efficiency_A / work_efficiency_B) = 45 / 2 := by
  sorry

end NUMINAMATH_CALUDE_work_efficiency_ratio_l3286_328608


namespace NUMINAMATH_CALUDE_lukes_trip_time_l3286_328635

/-- Calculates the total trip time for Luke's journey to London --/
theorem lukes_trip_time :
  let bus_time : ℚ := 75 / 60
  let walk_time : ℚ := 15 / 60
  let wait_time : ℚ := 2 * walk_time
  let train_time : ℚ := 6
  bus_time + walk_time + wait_time + train_time = 8 :=
by sorry

end NUMINAMATH_CALUDE_lukes_trip_time_l3286_328635


namespace NUMINAMATH_CALUDE_no_integer_solution_x4_plus_6_eq_y3_l3286_328638

theorem no_integer_solution_x4_plus_6_eq_y3 :
  ∀ (x y : ℤ), (x^4 + 6) % 13 ≠ y^3 % 13 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_x4_plus_6_eq_y3_l3286_328638


namespace NUMINAMATH_CALUDE_sports_books_count_l3286_328604

theorem sports_books_count (total_books : ℕ) (school_books : ℕ) (sports_books : ℕ) 
  (h1 : total_books = 344)
  (h2 : school_books = 136)
  (h3 : total_books = school_books + sports_books) :
  sports_books = 208 := by
sorry

end NUMINAMATH_CALUDE_sports_books_count_l3286_328604


namespace NUMINAMATH_CALUDE_students_with_average_age_16_l3286_328637

theorem students_with_average_age_16 (total_students : ℕ) (total_avg_age : ℕ) 
  (students_avg_14 : ℕ) (age_15th_student : ℕ) :
  total_students = 15 →
  total_avg_age = 15 →
  students_avg_14 = 5 →
  age_15th_student = 11 →
  ∃ (students_avg_16 : ℕ),
    students_avg_16 = 9 ∧
    students_avg_16 * 16 = total_students * total_avg_age - students_avg_14 * 14 - age_15th_student :=
by sorry

end NUMINAMATH_CALUDE_students_with_average_age_16_l3286_328637


namespace NUMINAMATH_CALUDE_complex_modulus_l3286_328681

theorem complex_modulus (r : ℝ) (z : ℂ) (hr : |r| < 1) (hz : z - 1/z = r) :
  Complex.abs z = Real.sqrt (1 + r^2/2) := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3286_328681


namespace NUMINAMATH_CALUDE_product_of_numbers_l3286_328605

theorem product_of_numbers (x y : ℝ) 
  (sum_eq : x + y = 22) 
  (sum_squares_eq : x^2 + y^2 = 404) : 
  x * y = 40 := by sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3286_328605


namespace NUMINAMATH_CALUDE_f_greater_g_iff_a_geq_half_l3286_328622

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - a - Real.log x

noncomputable def g (x : ℝ) : ℝ := 1 / x - 1 / Real.exp (x - 1)

theorem f_greater_g_iff_a_geq_half (a : ℝ) :
  (∀ x > 1, f a x > g x) ↔ a ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_greater_g_iff_a_geq_half_l3286_328622


namespace NUMINAMATH_CALUDE_lcm_factor_42_l3286_328620

theorem lcm_factor_42 (A B : ℕ+) : 
  Nat.gcd A B = 42 → 
  max A B = 840 → 
  42 ∣ Nat.lcm A B :=
by
  sorry

end NUMINAMATH_CALUDE_lcm_factor_42_l3286_328620


namespace NUMINAMATH_CALUDE_matrix_multiplication_example_l3286_328618

theorem matrix_multiplication_example :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, -2; -1, 4]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![2, 5; 0, -3]
  A * B = !![6, 21; -2, -17] := by
  sorry

end NUMINAMATH_CALUDE_matrix_multiplication_example_l3286_328618


namespace NUMINAMATH_CALUDE_three_digit_power_ending_l3286_328684

theorem three_digit_power_ending (N : ℕ) : 
  (100 ≤ N ∧ N < 1000) → 
  (∀ k : ℕ, k > 0 → N^k ≡ N [ZMOD 1000]) → 
  (N = 625 ∨ N = 376) :=
sorry

end NUMINAMATH_CALUDE_three_digit_power_ending_l3286_328684


namespace NUMINAMATH_CALUDE_inverse_variation_example_l3286_328631

-- Define the inverse variation relationship
def inverse_variation (p q : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ p * q = k

-- State the theorem
theorem inverse_variation_example :
  ∀ p q : ℝ,
  inverse_variation p q →
  (p = 1500 → q = 0.25) →
  (p = 3000 → q = 0.125) :=
by sorry

end NUMINAMATH_CALUDE_inverse_variation_example_l3286_328631


namespace NUMINAMATH_CALUDE_final_price_correct_l3286_328666

/-- The final selling price of an item after two discounts -/
def final_price (m : ℝ) : ℝ :=
  0.8 * m - 10

/-- Theorem stating the correctness of the final price calculation -/
theorem final_price_correct (m : ℝ) :
  let first_discount := 0.2
  let second_discount := 10
  let price_after_first := m * (1 - first_discount)
  let final_price := price_after_first - second_discount
  final_price = 0.8 * m - 10 :=
by sorry

end NUMINAMATH_CALUDE_final_price_correct_l3286_328666


namespace NUMINAMATH_CALUDE_article_pricing_loss_l3286_328682

/-- Proves that for an article with a given cost price, selling at 216 results in a 20% profit,
    and selling at 153 results in a 15% loss. -/
theorem article_pricing_loss (CP : ℝ) : 
  CP * 1.2 = 216 → (CP - 153) / CP * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_article_pricing_loss_l3286_328682


namespace NUMINAMATH_CALUDE_evaluate_expression_l3286_328656

theorem evaluate_expression : 5 * (9 - 3) + 8 = 38 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3286_328656


namespace NUMINAMATH_CALUDE_joan_oranges_l3286_328685

theorem joan_oranges (total_oranges sara_oranges : ℕ) 
  (h1 : total_oranges = 47) 
  (h2 : sara_oranges = 10) : 
  total_oranges - sara_oranges = 37 := by
  sorry

end NUMINAMATH_CALUDE_joan_oranges_l3286_328685


namespace NUMINAMATH_CALUDE_square_sum_division_theorem_l3286_328669

theorem square_sum_division_theorem (a b : ℕ+) :
  let q : ℕ := (a.val^2 + b.val^2) / (a.val + b.val)
  let r : ℕ := (a.val^2 + b.val^2) % (a.val + b.val)
  q^2 + r = 1977 →
  ((a.val = 50 ∧ b.val = 37) ∨
   (a.val = 37 ∧ b.val = 50) ∨
   (a.val = 50 ∧ b.val = 7) ∨
   (a.val = 7 ∧ b.val = 50)) :=
by sorry

end NUMINAMATH_CALUDE_square_sum_division_theorem_l3286_328669


namespace NUMINAMATH_CALUDE_smallest_factor_l3286_328645

def is_divisible (a b : ℕ) : Prop := ∃ k, a = b * k

theorem smallest_factor (w n : ℕ) : 
  w > 0 → 
  n > 0 → 
  (∀ w' : ℕ, w' ≥ w → is_divisible (w' * n) (2^5)) →
  (∀ w' : ℕ, w' ≥ w → is_divisible (w' * n) (3^3)) →
  (∀ w' : ℕ, w' ≥ w → is_divisible (w' * n) (10^2)) →
  w = 120 →
  n ≥ 180 :=
sorry

end NUMINAMATH_CALUDE_smallest_factor_l3286_328645


namespace NUMINAMATH_CALUDE_sunzi_wood_measurement_l3286_328634

/-- Proves the equation for the wood measurement problem from "The Mathematical Classic of Sunzi" -/
theorem sunzi_wood_measurement (x : ℝ) 
  (h1 : ∃ (rope_length : ℝ), rope_length = x + 4.5) 
  (h2 : ∃ (half_rope : ℝ), half_rope = x - 1 ∧ half_rope = (x + 4.5) / 2) : 
  (x + 4.5) / 2 = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_sunzi_wood_measurement_l3286_328634


namespace NUMINAMATH_CALUDE_circles_intersect_distance_between_centers_l3286_328640

/-- Given two circles M and N, prove that they intersect --/
theorem circles_intersect : ∀ (a : ℝ),
  a > 0 →
  (∃ (x y : ℝ), x^2 + y^2 - 2*a*y = 0 ∧ x + y = 0 ∧ (x - (-x))^2 = 4) →
  a = Real.sqrt 2 →
  ∃ (x y : ℝ), 
    x^2 + (y - a)^2 = a^2 ∧
    (x - 1)^2 + (y - 1)^2 = 1 :=
by
  sorry

/-- The distance between the centers of the circles is between |R-r| and R+r --/
theorem distance_between_centers (a : ℝ) (h : a = Real.sqrt 2) :
  Real.sqrt 2 - 1 < Real.sqrt (1 + (Real.sqrt 2 - 1)^2) ∧
  Real.sqrt (1 + (Real.sqrt 2 - 1)^2) < Real.sqrt 2 + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_circles_intersect_distance_between_centers_l3286_328640


namespace NUMINAMATH_CALUDE_correlation_relationships_l3286_328650

/-- Represents a relationship between two factors --/
inductive Relationship
| TeacherStudent
| SphereVolumeRadius
| AppleProductionClimate
| CrowsCawingOmen
| TreeDiameterHeight
| StudentIDNumber

/-- Defines whether a relationship has a correlation --/
def has_correlation (r : Relationship) : Prop :=
  match r with
  | Relationship.TeacherStudent => true
  | Relationship.SphereVolumeRadius => false
  | Relationship.AppleProductionClimate => true
  | Relationship.CrowsCawingOmen => false
  | Relationship.TreeDiameterHeight => true
  | Relationship.StudentIDNumber => false

/-- Theorem stating which relationships have correlations --/
theorem correlation_relationships :
  (has_correlation Relationship.TeacherStudent) ∧
  (has_correlation Relationship.AppleProductionClimate) ∧
  (has_correlation Relationship.TreeDiameterHeight) ∧
  (¬ has_correlation Relationship.SphereVolumeRadius) ∧
  (¬ has_correlation Relationship.CrowsCawingOmen) ∧
  (¬ has_correlation Relationship.StudentIDNumber) := by
  sorry


end NUMINAMATH_CALUDE_correlation_relationships_l3286_328650


namespace NUMINAMATH_CALUDE_triangle_area_l3286_328613

theorem triangle_area (a b c : ℝ) (h1 : a = 21) (h2 : b = 72) (h3 : c = 75) : 
  (1/2 : ℝ) * a * b = 756 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3286_328613


namespace NUMINAMATH_CALUDE_like_terms_ratio_l3286_328601

theorem like_terms_ratio (m n : ℕ) : 
  (∃ (x y : ℝ), 2 * x^(m-2) * y^3 = -1/2 * x^2 * y^(2*n-1)) → 
  m / n = 2 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_ratio_l3286_328601


namespace NUMINAMATH_CALUDE_angle_P_measure_l3286_328616

-- Define the triangle PQR
structure Triangle :=
  (P Q R : Real)

-- Define the properties of the triangle
def valid_triangle (t : Triangle) : Prop :=
  t.P > 0 ∧ t.Q > 0 ∧ t.R > 0 ∧ t.P + t.Q + t.R = 180

-- Define the theorem
theorem angle_P_measure (t : Triangle) 
  (h1 : valid_triangle t) 
  (h2 : t.Q = 3 * t.R) 
  (h3 : t.R = 18) : 
  t.P = 108 := by
  sorry

end NUMINAMATH_CALUDE_angle_P_measure_l3286_328616


namespace NUMINAMATH_CALUDE_fraction_sum_equals_one_l3286_328653

theorem fraction_sum_equals_one (x y : ℝ) 
  (h1 : 3 * x + 2 * y ≠ 0) (h2 : 3 * x - 2 * y ≠ 0) : 
  (7 * x - 5 * y) / (3 * x + 2 * y) + 
  (5 * x - 8 * y) / (3 * x - 2 * y) - 
  (x - 9 * y) / (3 * x + 2 * y) - 
  (8 * x - 10 * y) / (3 * x - 2 * y) = 1 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_one_l3286_328653


namespace NUMINAMATH_CALUDE_solve_for_q_l3286_328617

theorem solve_for_q (k l q : ℚ) : 
  (2/3 : ℚ) = k/45 ∧ (2/3 : ℚ) = (k+l)/75 ∧ (2/3 : ℚ) = (q-l)/105 → q = 90 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_q_l3286_328617


namespace NUMINAMATH_CALUDE_min_cups_to_fill_cylinder_l3286_328648

def cylinder_capacity : ℚ := 980
def cup_capacity : ℚ := 80

theorem min_cups_to_fill_cylinder :
  ⌈cylinder_capacity / cup_capacity⌉ = 13 := by
  sorry

end NUMINAMATH_CALUDE_min_cups_to_fill_cylinder_l3286_328648


namespace NUMINAMATH_CALUDE_diophantine_equation_solvability_l3286_328687

theorem diophantine_equation_solvability (m : ℤ) :
  ∃ (k : ℕ+) (a b c d : ℕ+), a * b - c * d = m := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solvability_l3286_328687


namespace NUMINAMATH_CALUDE_sum_digits_ratio_bound_l3286_328678

/-- Sum of digits function -/
def S (n : ℕ+) : ℕ := sorry

/-- The theorem stating the upper bound and its achievability -/
theorem sum_digits_ratio_bound :
  (∀ n : ℕ+, (S n : ℚ) / (S (16 * n) : ℚ) ≤ 13) ∧
  (∃ n : ℕ+, (S n : ℚ) / (S (16 * n) : ℚ) = 13) :=
sorry

end NUMINAMATH_CALUDE_sum_digits_ratio_bound_l3286_328678


namespace NUMINAMATH_CALUDE_ball_returns_to_start_l3286_328659

/-- The number of girls in the circle -/
def n : ℕ := 13

/-- The number of positions to advance in each throw -/
def k : ℕ := 5

/-- The function that determines the next girl to receive the ball -/
def next (x : ℕ) : ℕ := (x + k) % n

/-- The sequence of girls who receive the ball, starting from position 1 -/
def ball_sequence : ℕ → ℕ
  | 0 => 1
  | i + 1 => next (ball_sequence i)

theorem ball_returns_to_start :
  ∃ m : ℕ, m > 0 ∧ ball_sequence m = 1 ∧ ∀ i < m, ball_sequence i ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_ball_returns_to_start_l3286_328659


namespace NUMINAMATH_CALUDE_units_digit_of_sum_of_cubes_l3286_328690

theorem units_digit_of_sum_of_cubes : ∃ n : ℕ, n < 10 ∧ (41^3 + 23^3) % 10 = n ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_of_cubes_l3286_328690


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3286_328679

theorem arithmetic_calculations : 
  (23 - 17 - (-6) + (-16) = -4) ∧ 
  (0 - 32 / ((-2)^3 - (-4)) = 8) := by sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3286_328679


namespace NUMINAMATH_CALUDE_circle_radius_range_l3286_328664

/-- Given points P and C in a 2D Cartesian coordinate system, 
    if there exist two distinct points A and B on the circle centered at C with radius r, 
    such that PA - 2AB = 0, then r is in the range [1, 5). -/
theorem circle_radius_range (P C A B : ℝ × ℝ) (r : ℝ) : 
  P = (2, 2) →
  C = (5, 6) →
  A ≠ B →
  (∃ (A B : ℝ × ℝ), 
    (A.1 - C.1)^2 + (A.2 - C.2)^2 = r^2 ∧ 
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = r^2 ∧
    (A.1 - P.1, A.2 - P.2) = 2 • (B.1 - A.1, B.2 - A.2)) →
  r ∈ Set.Icc 1 5 ∧ r ≠ 5 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_range_l3286_328664


namespace NUMINAMATH_CALUDE_grisha_hat_color_l3286_328655

/-- Represents the color of a hat -/
inductive HatColor
| White
| Black

/-- Represents a person in the game -/
structure Person where
  name : String
  hatColor : HatColor
  canSee : List String

/-- The game setup -/
structure GameSetup where
  totalHats : Nat
  whiteHats : Nat
  blackHats : Nat
  persons : List Person
  remainingHats : Nat

/-- Predicate to check if a person can determine their hat color -/
def canDetermineColor (setup : GameSetup) (person : Person) : Prop := sorry

/-- The main theorem -/
theorem grisha_hat_color (setup : GameSetup) 
  (h1 : setup.totalHats = 5)
  (h2 : setup.whiteHats = 2)
  (h3 : setup.blackHats = 3)
  (h4 : setup.remainingHats = 2)
  (h5 : setup.persons.length = 3)
  (h6 : ∃ zhenya ∈ setup.persons, zhenya.name = "Zhenya" ∧ zhenya.canSee = ["Lyova", "Grisha"])
  (h7 : ∃ lyova ∈ setup.persons, lyova.name = "Lyova" ∧ lyova.canSee = ["Grisha"])
  (h8 : ∃ grisha ∈ setup.persons, grisha.name = "Grisha" ∧ grisha.canSee = [])
  (h9 : ∃ zhenya ∈ setup.persons, zhenya.name = "Zhenya" ∧ ¬canDetermineColor setup zhenya)
  (h10 : ∃ lyova ∈ setup.persons, lyova.name = "Lyova" ∧ ¬canDetermineColor setup lyova) :
  ∃ grisha ∈ setup.persons, grisha.name = "Grisha" ∧ grisha.hatColor = HatColor.Black ∧ canDetermineColor setup grisha :=
sorry

end NUMINAMATH_CALUDE_grisha_hat_color_l3286_328655


namespace NUMINAMATH_CALUDE_always_odd_l3286_328626

theorem always_odd (p m : ℤ) (h_p : Odd p) : Odd (p^3 + 3*p*m^2 + 2*m) := by
  sorry

end NUMINAMATH_CALUDE_always_odd_l3286_328626


namespace NUMINAMATH_CALUDE_coronavirus_cases_difference_l3286_328624

theorem coronavirus_cases_difference (new_york california texas : ℕ) : 
  new_york = 2000 →
  california = new_york / 2 →
  new_york + california + texas = 3600 →
  texas < california →
  california - texas = 400 :=
by sorry

end NUMINAMATH_CALUDE_coronavirus_cases_difference_l3286_328624


namespace NUMINAMATH_CALUDE_largest_intersection_is_eight_l3286_328642

/-- A polynomial of degree 6 -/
def P (a b c : ℝ) (x : ℝ) : ℝ :=
  x^6 - 14*x^5 + 45*x^4 - 30*x^3 + a*x^2 + b*x + c

/-- A linear function -/
def L (d e : ℝ) (x : ℝ) : ℝ :=
  d*x + e

/-- The difference between P and L -/
def Q (a b c d e : ℝ) (x : ℝ) : ℝ :=
  P a b c x - L d e x

theorem largest_intersection_is_eight (a b c d e : ℝ) :
  (∃ p q r : ℝ, p < q ∧ q < r ∧
    ∀ x : ℝ, Q a b c d e x = 0 ↔ (x = p ∨ x = q ∨ x = r) ∧
    ∀ x : ℝ, x ≠ p ∧ x ≠ q ∧ x ≠ r → Q a b c d e x > 0) →
  r = 8 :=
sorry

end NUMINAMATH_CALUDE_largest_intersection_is_eight_l3286_328642
