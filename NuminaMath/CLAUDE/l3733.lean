import Mathlib

namespace NUMINAMATH_CALUDE_books_read_l3733_373340

theorem books_read (total : ℕ) (unread : ℕ) (h1 : total = 21) (h2 : unread = 8) :
  total - unread = 13 := by
  sorry

end NUMINAMATH_CALUDE_books_read_l3733_373340


namespace NUMINAMATH_CALUDE_alcohol_dilution_l3733_373307

theorem alcohol_dilution (original_volume : ℝ) (original_percentage : ℝ) 
  (added_water : ℝ) (new_percentage : ℝ) :
  original_volume = 15 →
  original_percentage = 0.2 →
  added_water = 3 →
  new_percentage = 1/6 →
  (original_volume * original_percentage) / (original_volume + added_water) = new_percentage := by
  sorry

#check alcohol_dilution

end NUMINAMATH_CALUDE_alcohol_dilution_l3733_373307


namespace NUMINAMATH_CALUDE_vector_operation_l3733_373392

/-- Given vectors a and b in R², prove that 2a - b equals (-1, 0) --/
theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (3, 4)) :
  2 • a - b = (-1, 0) := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_l3733_373392


namespace NUMINAMATH_CALUDE_y_in_terms_of_z_l3733_373323

theorem y_in_terms_of_z (x y z : ℝ) : 
  x = 90 * (1 + 0.11) →
  y = x * (1 - 0.27) →
  z = y/2 + 3 →
  y = 2*z - 6 := by
  sorry

end NUMINAMATH_CALUDE_y_in_terms_of_z_l3733_373323


namespace NUMINAMATH_CALUDE_cos_160_eq_neg_cos_20_l3733_373345

/-- Proves that cos 160° equals -cos 20° --/
theorem cos_160_eq_neg_cos_20 : 
  Real.cos (160 * π / 180) = - Real.cos (20 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_cos_160_eq_neg_cos_20_l3733_373345


namespace NUMINAMATH_CALUDE_combined_girls_average_is_89_l3733_373315

/-- Represents a high school with average test scores -/
structure School where
  boyAvg : ℝ
  girlAvg : ℝ
  combinedAvg : ℝ

/-- Calculates the combined average score for girls given two schools and the combined boys' average -/
def combinedGirlsAverage (lincoln : School) (monroe : School) (combinedBoysAvg : ℝ) : ℝ :=
  sorry

theorem combined_girls_average_is_89 (lincoln : School) (monroe : School) (combinedBoysAvg : ℝ) :
  lincoln.boyAvg = 75 ∧
  lincoln.girlAvg = 78 ∧
  lincoln.combinedAvg = 76 ∧
  monroe.boyAvg = 85 ∧
  monroe.girlAvg = 92 ∧
  monroe.combinedAvg = 88 ∧
  combinedBoysAvg = 82 →
  combinedGirlsAverage lincoln monroe combinedBoysAvg = 89 := by
  sorry

end NUMINAMATH_CALUDE_combined_girls_average_is_89_l3733_373315


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l3733_373344

/-- The eccentricity of a hyperbola given its equation and a point in the "up" region -/
theorem hyperbola_eccentricity_range (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_up : b / a < 2) : ∃ e : ℝ, 1 < e ∧ e < Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l3733_373344


namespace NUMINAMATH_CALUDE_cos_2alpha_minus_2pi_3_l3733_373384

theorem cos_2alpha_minus_2pi_3 (α : Real) (h : Real.sin (π/6 + α) = 3/5) : 
  Real.cos (2*α - 2*π/3) = -7/25 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_minus_2pi_3_l3733_373384


namespace NUMINAMATH_CALUDE_semi_annual_compound_interest_rate_l3733_373359

/-- Proves that the annual interest rate of a semi-annually compounded account is approximately 7.96%
    given specific conditions on the initial investment and interest earned. -/
theorem semi_annual_compound_interest_rate (principal : ℝ) (simple_rate : ℝ) (diff : ℝ) :
  principal = 5000 →
  simple_rate = 0.08 →
  diff = 6 →
  ∃ (compound_rate : ℝ),
    (principal * (1 + compound_rate / 2)^2 - principal) = 
    (principal * simple_rate + diff) ∧
    abs (compound_rate - 0.0796) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_semi_annual_compound_interest_rate_l3733_373359


namespace NUMINAMATH_CALUDE_function_inequality_l3733_373332

theorem function_inequality 
  (f : Real → Real) 
  (h_domain : ∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x) 
  (h_ineq : ∀ x y, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 → 
    (f x + f y) / 2 ≤ f ((x + y) / 2) + 1) 
  (u v w : Real) 
  (h_order : 0 ≤ u ∧ u < v ∧ v < w ∧ w ≤ 1) : 
  ((w - v) / (w - u)) * f u + ((v - u) / (w - u)) * f w ≤ f v + 2 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l3733_373332


namespace NUMINAMATH_CALUDE_stratified_sampling_female_count_l3733_373303

theorem stratified_sampling_female_count 
  (total_employees : ℕ) 
  (female_employees : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_employees = 200)
  (h2 : female_employees = 80)
  (h3 : sample_size = 20) :
  (female_employees : ℚ) / total_employees * sample_size = 8 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_female_count_l3733_373303


namespace NUMINAMATH_CALUDE_complex_sum_cube_ratio_l3733_373329

theorem complex_sum_cube_ratio (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 18)
  (h_diff_sq : (x - y)^2 + (x - z)^2 + (y - z)^2 = (x*y*z)/3) :
  (x^3 + y^3 + z^3) / (x*y*z) = 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_cube_ratio_l3733_373329


namespace NUMINAMATH_CALUDE_regular_ngon_max_area_and_perimeter_l3733_373311

/-- A circle in a 2D plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- An n-gon inscribed in a circle. -/
structure InscribedNGon (n : ℕ) (c : Circle) where
  vertices : Fin n → ℝ × ℝ
  inscribed : ∀ i, ((vertices i).1 - c.center.1)^2 + ((vertices i).2 - c.center.2)^2 = c.radius^2

/-- The area of an n-gon. -/
def area {n : ℕ} {c : Circle} (ngon : InscribedNGon n c) : ℝ :=
  sorry

/-- The perimeter of an n-gon. -/
def perimeter {n : ℕ} {c : Circle} (ngon : InscribedNGon n c) : ℝ :=
  sorry

/-- A regular n-gon inscribed in a circle. -/
def regularNGon (n : ℕ) (c : Circle) : InscribedNGon n c :=
  sorry

/-- Theorem: The regular n-gon has maximum area and perimeter among all inscribed n-gons. -/
theorem regular_ngon_max_area_and_perimeter (n : ℕ) (c : Circle) :
  ∀ (ngon : InscribedNGon n c),
    area ngon ≤ area (regularNGon n c) ∧
    perimeter ngon ≤ perimeter (regularNGon n c) :=
  sorry

end NUMINAMATH_CALUDE_regular_ngon_max_area_and_perimeter_l3733_373311


namespace NUMINAMATH_CALUDE_x_squared_mod_25_l3733_373300

theorem x_squared_mod_25 (x : ℤ) 
  (h1 : 5 * x ≡ 10 [ZMOD 25]) 
  (h2 : 4 * x ≡ 20 [ZMOD 25]) : 
  x^2 ≡ 4 [ZMOD 25] := by
  sorry

end NUMINAMATH_CALUDE_x_squared_mod_25_l3733_373300


namespace NUMINAMATH_CALUDE_candy_problem_l3733_373362

/-- Given an initial amount of candy and the amounts eaten in two stages,
    calculate the remaining amount of candy. -/
def remaining_candy (initial : ℕ) (first_eaten : ℕ) (second_eaten : ℕ) : ℕ :=
  initial - (first_eaten + second_eaten)

/-- Theorem stating that given 36 initial pieces of candy, 
    after eating 17 and then 15 pieces, 4 pieces remain. -/
theorem candy_problem : remaining_candy 36 17 15 = 4 := by
  sorry

end NUMINAMATH_CALUDE_candy_problem_l3733_373362


namespace NUMINAMATH_CALUDE_hotdogs_served_today_l3733_373350

/-- The number of hot dogs served during lunch today -/
def lunch_hotdogs : ℕ := 9

/-- The number of hot dogs served during dinner today -/
def dinner_hotdogs : ℕ := 2

/-- The total number of hot dogs served today -/
def total_hotdogs : ℕ := lunch_hotdogs + dinner_hotdogs

theorem hotdogs_served_today : total_hotdogs = 11 := by
  sorry

end NUMINAMATH_CALUDE_hotdogs_served_today_l3733_373350


namespace NUMINAMATH_CALUDE_specific_event_handshakes_l3733_373358

/-- Represents a social event with two groups of people -/
structure SocialEvent where
  total_people : ℕ
  group_a_size : ℕ
  group_b_size : ℕ
  h_total : total_people = group_a_size + group_b_size
  h_group_a : group_a_size > 0
  h_group_b : group_b_size > 0

/-- Calculates the number of handshakes in a social event -/
def handshakes (event : SocialEvent) : ℕ :=
  event.group_a_size * event.group_b_size

/-- Theorem stating the number of handshakes in the specific social event -/
theorem specific_event_handshakes :
  ∃ (event : SocialEvent),
    event.total_people = 40 ∧
    event.group_a_size = 25 ∧
    event.group_b_size = 15 ∧
    handshakes event = 375 := by
  sorry

end NUMINAMATH_CALUDE_specific_event_handshakes_l3733_373358


namespace NUMINAMATH_CALUDE_cross_product_zero_implies_values_l3733_373316

def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.1 * b.2.2 - a.2.2 * b.2.1,
   a.2.2 * b.1 - a.1 * b.2.2,
   a.1 * b.2.1 - a.2.1 * b.1)

theorem cross_product_zero_implies_values (x y : ℝ) :
  cross_product (3, x, -9) (4, 6, y) = (0, 0, 0) →
  x = 9/2 ∧ y = -12 := by
  sorry

end NUMINAMATH_CALUDE_cross_product_zero_implies_values_l3733_373316


namespace NUMINAMATH_CALUDE_trailing_zeros_340_factorial_l3733_373330

-- Define a function to count trailing zeros in a factorial
def trailingZerosInFactorial (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

-- Theorem statement
theorem trailing_zeros_340_factorial :
  trailingZerosInFactorial 340 = 83 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_340_factorial_l3733_373330


namespace NUMINAMATH_CALUDE_quadratic_greater_than_linear_l3733_373373

theorem quadratic_greater_than_linear (x : ℝ) :
  let y₁ : ℝ → ℝ := λ x => x + 1
  let y₂ : ℝ → ℝ := λ x => (1/2) * x^2 - (1/2) * x - 1
  (y₂ x > y₁ x) ↔ (x < -1 ∨ x > 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_greater_than_linear_l3733_373373


namespace NUMINAMATH_CALUDE_calculation_difference_l3733_373310

def correct_calculation : ℤ := 12 - (3 * 4)
def incorrect_calculation : ℤ := (12 - 3) * 4

theorem calculation_difference :
  correct_calculation - incorrect_calculation = -36 := by
  sorry

end NUMINAMATH_CALUDE_calculation_difference_l3733_373310


namespace NUMINAMATH_CALUDE_point_on_unit_circle_l3733_373390

/-- The coordinates of a point on the unit circle after moving counterclockwise from (1,0) by an arc length of 4π/3 -/
theorem point_on_unit_circle (Q : ℝ × ℝ) : 
  (∃ θ : ℝ, θ = 4 * Real.pi / 3 ∧ 
   Q.1 = Real.cos θ ∧ 
   Q.2 = Real.sin θ) →
  Q = (-1/2, -Real.sqrt 3 / 2) := by
sorry


end NUMINAMATH_CALUDE_point_on_unit_circle_l3733_373390


namespace NUMINAMATH_CALUDE_unique_solution_l3733_373321

theorem unique_solution (a b c : ℝ) : 
  a > 4 ∧ b > 4 ∧ c > 4 ∧
  (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 48 →
  a = 9 ∧ b = 8 ∧ c = 8 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l3733_373321


namespace NUMINAMATH_CALUDE_coupon_a_best_at_220_l3733_373336

def coupon_a_discount (price : ℝ) : ℝ := 0.12 * price

def coupon_b_discount (price : ℝ) : ℝ := 25

def coupon_c_discount (price : ℝ) : ℝ := 0.2 * (price - 120)

theorem coupon_a_best_at_220 :
  let price := 220
  coupon_a_discount price > coupon_b_discount price ∧
  coupon_a_discount price > coupon_c_discount price :=
by sorry

end NUMINAMATH_CALUDE_coupon_a_best_at_220_l3733_373336


namespace NUMINAMATH_CALUDE_cost_of_stationery_l3733_373391

/-- Given the cost of erasers, pens, and markers satisfying certain conditions,
    prove that the total cost of 3 erasers, 4 pens, and 6 markers is 520 rubles. -/
theorem cost_of_stationery (E P M : ℕ) 
    (h1 : E + 3 * P + 2 * M = 240)
    (h2 : 2 * E + 5 * P + 4 * M = 440) :
  3 * E + 4 * P + 6 * M = 520 := by
  sorry


end NUMINAMATH_CALUDE_cost_of_stationery_l3733_373391


namespace NUMINAMATH_CALUDE_certain_number_problem_l3733_373354

theorem certain_number_problem (x : ℝ) : 
  (15 - 2 + x) / 2 * 8 = 77 → x = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3733_373354


namespace NUMINAMATH_CALUDE_smallest_cube_divisor_l3733_373304

theorem smallest_cube_divisor (p q r s : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r → Nat.Prime s →
  p ≠ q → p ≠ r → p ≠ s → q ≠ r → q ≠ s → r ≠ s →
  (∀ m : ℕ, m > 0 → m^3 % (p * q^2 * r^4 * s^3) = 0 → m ≥ p * q * r^2 * s) ∧
  (p * q * r^2 * s)^3 % (p * q^2 * r^4 * s^3) = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_cube_divisor_l3733_373304


namespace NUMINAMATH_CALUDE_det_M_eq_26_l3733_373365

/-- The determinant of a 2x2 matrix -/
def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

/-- The specific 2x2 matrix we're interested in -/
def M : Matrix (Fin 2) (Fin 2) ℝ := !![5, 3; -2, 4]

/-- Theorem stating that the determinant of M is 26 -/
theorem det_M_eq_26 : det2x2 (M 0 0) (M 0 1) (M 1 0) (M 1 1) = 26 := by
  sorry

end NUMINAMATH_CALUDE_det_M_eq_26_l3733_373365


namespace NUMINAMATH_CALUDE_line_not_in_second_quadrant_l3733_373325

/-- A line that does not pass through the second quadrant -/
theorem line_not_in_second_quadrant (a : ℝ) (h : a ≠ 0) :
  ∀ x y : ℝ, x - y - a^2 = 0 → ¬(x < 0 ∧ y > 0) :=
by sorry

end NUMINAMATH_CALUDE_line_not_in_second_quadrant_l3733_373325


namespace NUMINAMATH_CALUDE_children_going_to_zoo_l3733_373322

/-- The number of children per seat in the bus -/
def children_per_seat : ℕ := 2

/-- The total number of seats needed in the bus -/
def total_seats : ℕ := 29

/-- The total number of children taking the bus to the zoo -/
def total_children : ℕ := children_per_seat * total_seats

theorem children_going_to_zoo : total_children = 58 := by
  sorry

end NUMINAMATH_CALUDE_children_going_to_zoo_l3733_373322


namespace NUMINAMATH_CALUDE_system_solution_ratio_l3733_373339

theorem system_solution_ratio (x y a b : ℝ) (h1 : 4*x - 3*y = a) (h2 : 6*y - 8*x = b) (h3 : b ≠ 0) :
  a / b = -1 / 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l3733_373339


namespace NUMINAMATH_CALUDE_y_over_z_equals_negative_five_l3733_373382

theorem y_over_z_equals_negative_five 
  (x y z : ℚ) 
  (eq1 : x + y = 2*x + z) 
  (eq2 : x - 2*y = 4*z) 
  (eq3 : x + y + z = 21) : 
  y / z = -5 := by sorry

end NUMINAMATH_CALUDE_y_over_z_equals_negative_five_l3733_373382


namespace NUMINAMATH_CALUDE_range_of_m_l3733_373333

def p (x : ℝ) : Prop := |1 - (x - 1) / 3| ≥ 2

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≥ 0

def not_p (x : ℝ) : Prop := -2 < x ∧ x < 10

def not_q (x m : ℝ) : Prop := 1 - m < x ∧ x < 1 + m

theorem range_of_m :
  ∀ m : ℝ, (m > 0 ∧
    (∀ x : ℝ, not_q x m → not_p x) ∧
    (∃ x : ℝ, not_p x ∧ ¬(not_q x m))) ↔
  (0 < m ∧ m ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3733_373333


namespace NUMINAMATH_CALUDE_cd_combined_length_l3733_373372

/-- The combined length of 3 CDs is 6 hours, given that two CDs are 1.5 hours each and the third CD is twice as long as the shorter ones. -/
theorem cd_combined_length : 
  let short_cd_length : ℝ := 1.5
  let long_cd_length : ℝ := 2 * short_cd_length
  let total_length : ℝ := 2 * short_cd_length + long_cd_length
  total_length = 6 := by sorry

end NUMINAMATH_CALUDE_cd_combined_length_l3733_373372


namespace NUMINAMATH_CALUDE_three_valid_plans_l3733_373388

/-- Represents the cost and construction details of parking spaces -/
structure ParkingProject where
  aboveGroundCost : ℚ
  undergroundCost : ℚ
  totalSpaces : ℕ
  minInvestment : ℚ
  maxInvestment : ℚ

/-- Calculates the number of valid construction plans -/
def validConstructionPlans (project : ParkingProject) : ℕ :=
  (project.totalSpaces + 1).fold
    (λ count aboveGround =>
      let underground := project.totalSpaces - aboveGround
      let cost := project.aboveGroundCost * aboveGround + project.undergroundCost * underground
      if project.minInvestment < cost ∧ cost ≤ project.maxInvestment then
        count + 1
      else
        count)
    0

/-- Theorem stating that there are exactly 3 valid construction plans -/
theorem three_valid_plans (project : ParkingProject)
  (h1 : project.aboveGroundCost + project.undergroundCost = 0.6)
  (h2 : 3 * project.aboveGroundCost + 2 * project.undergroundCost = 1.3)
  (h3 : project.totalSpaces = 50)
  (h4 : project.minInvestment = 12)
  (h5 : project.maxInvestment = 13) :
  validConstructionPlans project = 3 := by
  sorry

#eval validConstructionPlans {
  aboveGroundCost := 0.1,
  undergroundCost := 0.5,
  totalSpaces := 50,
  minInvestment := 12,
  maxInvestment := 13
}

end NUMINAMATH_CALUDE_three_valid_plans_l3733_373388


namespace NUMINAMATH_CALUDE_modular_difference_in_range_l3733_373389

theorem modular_difference_in_range (a b : ℤ) : 
  a ≡ 25 [ZMOD 60] →
  b ≡ 84 [ZMOD 60] →
  ∃! n : ℤ, 150 ≤ n ∧ n ≤ 200 ∧ a - b ≡ n [ZMOD 60] ∧ n = 181 :=
by sorry

end NUMINAMATH_CALUDE_modular_difference_in_range_l3733_373389


namespace NUMINAMATH_CALUDE_savings_account_theorem_l3733_373374

def initial_deposit : ℚ := 5 / 100
def daily_multiplier : ℚ := 3
def target_amount : ℚ := 500

def total_amount (n : ℕ) : ℚ :=
  initial_deposit * (1 - daily_multiplier^n) / (1 - daily_multiplier)

def exceeds_target (n : ℕ) : Prop :=
  total_amount n > target_amount

theorem savings_account_theorem :
  ∃ (n : ℕ), exceeds_target n ∧ ∀ (m : ℕ), m < n → ¬(exceeds_target m) :=
by sorry

end NUMINAMATH_CALUDE_savings_account_theorem_l3733_373374


namespace NUMINAMATH_CALUDE_equilateral_triangle_semi_regular_hexagon_l3733_373305

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Represents a hexagon -/
structure Hexagon :=
  (vertices : Fin 6 → Point)

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Divides each side of a triangle into three equal parts -/
def divideSides (t : Triangle) : Fin 6 → Point := sorry

/-- Forms a hexagon from the division points and opposite vertices -/
def formHexagon (t : Triangle) (divisionPoints : Fin 6 → Point) : Hexagon := sorry

/-- Checks if a hexagon is semi-regular -/
def isSemiRegular (h : Hexagon) : Prop := sorry

/-- Main theorem: The hexagon formed by dividing the sides of an equilateral triangle
    and connecting division points to opposite vertices is semi-regular -/
theorem equilateral_triangle_semi_regular_hexagon 
  (t : Triangle) (h : isEquilateral t) : 
  isSemiRegular (formHexagon t (divideSides t)) := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_semi_regular_hexagon_l3733_373305


namespace NUMINAMATH_CALUDE_x_value_l3733_373309

theorem x_value : ∃ x : ℝ, (3 * x) / 7 = 15 ∧ x = 35 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3733_373309


namespace NUMINAMATH_CALUDE_garbage_ratio_proof_l3733_373334

def garbage_problem (collection_days_per_week : ℕ) 
                    (avg_collection_per_day : ℕ) 
                    (weeks_without_collection : ℕ) 
                    (total_accumulated : ℕ) : Prop :=
  let weekly_collection := collection_days_per_week * avg_collection_per_day
  let total_normal_collection := weekly_collection * weeks_without_collection
  let first_week_garbage := weekly_collection
  let second_week_garbage := total_accumulated - first_week_garbage
  (2 : ℚ) * second_week_garbage = first_week_garbage

theorem garbage_ratio_proof : 
  garbage_problem 3 200 2 900 := by
  sorry

#check garbage_ratio_proof

end NUMINAMATH_CALUDE_garbage_ratio_proof_l3733_373334


namespace NUMINAMATH_CALUDE_exponential_function_property_l3733_373347

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

theorem exponential_function_property (a : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1) :
  ∀ x y : ℝ, f a (x + y) = f a x * f a y :=
by sorry

end NUMINAMATH_CALUDE_exponential_function_property_l3733_373347


namespace NUMINAMATH_CALUDE_roller_skate_attendance_l3733_373337

/-- The number of wheels on the floor when all people skated -/
def total_wheels : ℕ := 320

/-- The number of roller skates per person -/
def skates_per_person : ℕ := 2

/-- The number of wheels per roller skate -/
def wheels_per_skate : ℕ := 2

/-- The number of people who showed up to roller skate -/
def num_people : ℕ := total_wheels / (skates_per_person * wheels_per_skate)

theorem roller_skate_attendance : num_people = 80 := by
  sorry

end NUMINAMATH_CALUDE_roller_skate_attendance_l3733_373337


namespace NUMINAMATH_CALUDE_fraction_equals_negative_one_l3733_373396

theorem fraction_equals_negative_one (a b : ℝ) (h : a + b ≠ 0) :
  (-a - b) / (a + b) = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_negative_one_l3733_373396


namespace NUMINAMATH_CALUDE_vector_magnitude_l3733_373328

def m : ℝ × ℝ := (2, 4)

theorem vector_magnitude (m : ℝ × ℝ) (n : ℝ × ℝ) : 
  let angle := π / 3
  norm m = 2 * Real.sqrt 5 →
  norm n = Real.sqrt 5 →
  m.1 * n.1 + m.2 * n.2 = norm m * norm n * Real.cos angle →
  norm (2 • m - 3 • n) = Real.sqrt 65 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l3733_373328


namespace NUMINAMATH_CALUDE_student_mistake_fraction_l3733_373361

theorem student_mistake_fraction (original_number : ℕ) 
  (h1 : original_number = 384) 
  (correct_fraction : ℚ) 
  (h2 : correct_fraction = 5 / 16) 
  (mistake_fraction : ℚ) : 
  (mistake_fraction * original_number = correct_fraction * original_number + 200) → 
  mistake_fraction = 5 / 6 := by
sorry

end NUMINAMATH_CALUDE_student_mistake_fraction_l3733_373361


namespace NUMINAMATH_CALUDE_division_problem_l3733_373395

theorem division_problem (L S q : ℕ) : 
  L - S = 1325 → 
  L = 1650 → 
  L = S * q + 5 → 
  q = 5 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3733_373395


namespace NUMINAMATH_CALUDE_apples_per_pie_l3733_373313

theorem apples_per_pie (initial_apples : Nat) (handed_out : Nat) (num_pies : Nat)
  (h1 : initial_apples = 96)
  (h2 : handed_out = 42)
  (h3 : num_pies = 9) :
  (initial_apples - handed_out) / num_pies = 6 := by
  sorry

end NUMINAMATH_CALUDE_apples_per_pie_l3733_373313


namespace NUMINAMATH_CALUDE_system_solution_l3733_373343

theorem system_solution : ∃ (x y z : ℝ), 
  (x^2 + y - 2*z = -3) ∧ 
  (3*x + y + z^2 = 14) ∧ 
  (7*x - y^2 + 4*z = 25) ∧
  (x = 2 ∧ y = -1 ∧ z = 3) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3733_373343


namespace NUMINAMATH_CALUDE_rectangular_field_dimensions_l3733_373398

theorem rectangular_field_dimensions (m : ℝ) : 
  (3 * m + 8) * (m - 3) = 72 → m = (1 + Real.sqrt 1153) / 6 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_dimensions_l3733_373398


namespace NUMINAMATH_CALUDE_part_one_part_two_l3733_373346

-- Define propositions p and q
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := 2 < x ∧ x ≤ 3

-- Part 1
theorem part_one (x : ℝ) (h1 : p 1 x) (h2 : q x) : 2 < x ∧ x < 3 := by
  sorry

-- Part 2
theorem part_two (a : ℝ) (h1 : a > 0) (h2 : ∀ x, q x → p a x) : 1 < a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3733_373346


namespace NUMINAMATH_CALUDE_inequality_implication_l3733_373385

theorem inequality_implication (a b : ℝ) (h : a < b) : -3 * a > -3 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l3733_373385


namespace NUMINAMATH_CALUDE_least_number_with_special_division_property_l3733_373369

theorem least_number_with_special_division_property : ∃ k : ℕ, 
  k > 0 ∧ 
  k / 5 = k % 34 + 8 ∧ 
  (∀ m : ℕ, m > 0 → m / 5 = m % 34 + 8 → k ≤ m) ∧
  k = 68 := by
sorry

end NUMINAMATH_CALUDE_least_number_with_special_division_property_l3733_373369


namespace NUMINAMATH_CALUDE_money_ratio_l3733_373327

theorem money_ratio (rodney ian jessica : ℕ) : 
  rodney = ian + 35 →
  jessica = 100 →
  jessica = rodney + 15 →
  ian * 2 = jessica := by sorry

end NUMINAMATH_CALUDE_money_ratio_l3733_373327


namespace NUMINAMATH_CALUDE_three_digit_with_repeat_l3733_373376

/-- The number of digits available (0 to 9) -/
def num_digits : ℕ := 10

/-- The total number of three-digit numbers -/
def total_three_digit : ℕ := 900

/-- The number of three-digit numbers without repeated digits -/
def no_repeat_three_digit : ℕ := 9 * 9 * 8

/-- Theorem: The number of three-digit numbers with repeated digits using digits 0 to 9 is 252 -/
theorem three_digit_with_repeat : 
  total_three_digit - no_repeat_three_digit = 252 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_with_repeat_l3733_373376


namespace NUMINAMATH_CALUDE_football_team_progress_l3733_373378

/-- Calculates the total progress of a football team given yards lost and gained -/
def footballProgress (yardsLost : Int) (yardsGained : Int) : Int :=
  yardsGained - yardsLost

/-- Theorem: A football team that lost 5 yards and then gained 11 yards has a total progress of 6 yards -/
theorem football_team_progress :
  footballProgress 5 11 = 6 := by
  sorry

end NUMINAMATH_CALUDE_football_team_progress_l3733_373378


namespace NUMINAMATH_CALUDE_license_plate_palindrome_probability_l3733_373324

/-- Probability of a palindrome in a four-letter sequence -/
def prob_letter_palindrome : ℚ := 1 / 676

/-- Probability of a palindrome in a four-digit sequence -/
def prob_digit_palindrome : ℚ := 1 / 100

/-- Total number of possible license plate arrangements -/
def total_arrangements : ℕ := 26^4 * 10^4

theorem license_plate_palindrome_probability :
  let prob_at_least_one_palindrome := prob_letter_palindrome + prob_digit_palindrome - 
                                      (prob_letter_palindrome * prob_digit_palindrome)
  prob_at_least_one_palindrome = 775 / 67600 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_palindrome_probability_l3733_373324


namespace NUMINAMATH_CALUDE_gravel_path_cost_l3733_373301

/-- The cost of gravelling a path inside a rectangular plot -/
theorem gravel_path_cost 
  (length width path_width : ℝ) 
  (cost_per_sqm : ℝ) 
  (h_length : length = 110) 
  (h_width : width = 65) 
  (h_path_width : path_width = 2.5) 
  (h_cost_per_sqm : cost_per_sqm = 0.4) : 
  ((length + 2 * path_width) * (width + 2 * path_width) - length * width) * cost_per_sqm = 360 := by
sorry

end NUMINAMATH_CALUDE_gravel_path_cost_l3733_373301


namespace NUMINAMATH_CALUDE_cubic_function_two_zeros_l3733_373335

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 3 * x^3 - x + a

-- State the theorem
theorem cubic_function_two_zeros (a : ℝ) (h : a > 0) :
  (∃! x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0) ↔ a = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_two_zeros_l3733_373335


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3733_373308

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) :
  surface_area = 150 →
  volume = (surface_area / 6) ^ (3/2) →
  volume = 125 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3733_373308


namespace NUMINAMATH_CALUDE_novels_pages_per_book_l3733_373375

theorem novels_pages_per_book (novels_per_month : ℕ) (pages_per_year : ℕ) : 
  novels_per_month = 4 → pages_per_year = 9600 → 
  (pages_per_year / (novels_per_month * 12) : ℚ) = 200 := by
  sorry

end NUMINAMATH_CALUDE_novels_pages_per_book_l3733_373375


namespace NUMINAMATH_CALUDE_wax_required_for_feathers_l3733_373306

/-- The amount of wax Icarus has, in grams. -/
def total_wax : ℕ := 557

/-- The amount of wax needed for the feathers, in grams. -/
def wax_needed : ℕ := 17

/-- Theorem stating that the amount of wax required for the feathers is equal to the amount needed, regardless of the total amount available. -/
theorem wax_required_for_feathers : wax_needed = 17 := by
  sorry

end NUMINAMATH_CALUDE_wax_required_for_feathers_l3733_373306


namespace NUMINAMATH_CALUDE_polygon_sides_when_interior_triple_exterior_polygon_sides_proof_l3733_373357

theorem polygon_sides_when_interior_triple_exterior : ℕ → Prop :=
  fun n =>
    (((n : ℝ) - 2) * 180 = 3 * 360) →
    n = 8

-- Proof
theorem polygon_sides_proof : polygon_sides_when_interior_triple_exterior 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_when_interior_triple_exterior_polygon_sides_proof_l3733_373357


namespace NUMINAMATH_CALUDE_a_3_value_l3733_373319

/-- Given a polynomial expansion of (1+x)(a-x)^6, prove that a₃ = -5 when the sum of all coefficients is zero. -/
theorem a_3_value (a : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) : 
  (∀ x, (1 + x) * (a - x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 0) →
  a₃ = -5 := by
sorry

end NUMINAMATH_CALUDE_a_3_value_l3733_373319


namespace NUMINAMATH_CALUDE_meeting_distance_l3733_373352

/-- Proves that given two people 75 miles apart, walking towards each other at constant speeds of 4 mph and 6 mph respectively, the person walking at 6 mph will have walked 45 miles when they meet. -/
theorem meeting_distance (initial_distance : ℝ) (speed_fred : ℝ) (speed_sam : ℝ) 
  (h1 : initial_distance = 75)
  (h2 : speed_fred = 4)
  (h3 : speed_sam = 6) :
  let distance_sam := initial_distance * speed_sam / (speed_fred + speed_sam)
  distance_sam = 45 := by
  sorry

#check meeting_distance

end NUMINAMATH_CALUDE_meeting_distance_l3733_373352


namespace NUMINAMATH_CALUDE_prism_volume_l3733_373302

/-- The volume of a right rectangular prism given its face areas -/
theorem prism_volume (a b c : ℝ) (h1 : a * b = 50) (h2 : a * c = 72) (h3 : b * c = 45) :
  a * b * c = 180 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l3733_373302


namespace NUMINAMATH_CALUDE_geometric_sum_ratio_l3733_373312

/-- Given a geometric sequence {a_n}, S_n represents the sum of its first n terms -/
def S (n : ℕ) : ℝ := sorry

/-- The ratio of S_5 to S_10 is 1/3 -/
axiom ratio_condition : S 5 / S 10 = 1 / 3

/-- Theorem: If S_5 / S_10 = 1/3, then S_5 / (S_20 + S_10) = 1/18 -/
theorem geometric_sum_ratio : S 5 / (S 20 + S 10) = 1 / 18 := by sorry

end NUMINAMATH_CALUDE_geometric_sum_ratio_l3733_373312


namespace NUMINAMATH_CALUDE_largest_s_value_l3733_373379

/-- The largest possible value of s for regular polygons satisfying given conditions -/
theorem largest_s_value : ∃ (s : ℕ), s = 121 ∧ 
  (∀ (r s' : ℕ), r ≥ s' ∧ s' ≥ 3 →
    (r - 2 : ℚ) / r * 60 = (s' - 2 : ℚ) / s' * 61 →
    s' ≤ s) :=
sorry

end NUMINAMATH_CALUDE_largest_s_value_l3733_373379


namespace NUMINAMATH_CALUDE_min_balls_for_same_color_l3733_373377

def box : Finset (Fin 6) := Finset.univ
def color : Fin 6 → ℕ
  | 0 => 28  -- red
  | 1 => 20  -- green
  | 2 => 19  -- yellow
  | 3 => 13  -- blue
  | 4 => 11  -- white
  | 5 => 9   -- black

theorem min_balls_for_same_color : 
  ∀ n : ℕ, (∀ s : Finset (Fin 6), s.card = n → 
    (∃ c : Fin 6, (s.filter (λ i => color i = color c)).card < 15)) → 
  n < 76 :=
sorry

end NUMINAMATH_CALUDE_min_balls_for_same_color_l3733_373377


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_five_l3733_373353

theorem arithmetic_square_root_of_five :
  ∃ x : ℝ, x > 0 ∧ x^2 = 5 ∧ ∀ y : ℝ, y^2 = 5 → y = x ∨ y = -x :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_five_l3733_373353


namespace NUMINAMATH_CALUDE_salt_solution_concentration_l3733_373318

/-- Given a mixture of pure water and salt solution, prove the original salt solution concentration -/
theorem salt_solution_concentration 
  (pure_water_volume : ℝ) 
  (salt_solution_volume : ℝ) 
  (final_mixture_concentration : ℝ) 
  (h1 : pure_water_volume = 1)
  (h2 : salt_solution_volume = 0.5)
  (h3 : final_mixture_concentration = 15) :
  let total_volume := pure_water_volume + salt_solution_volume
  let salt_amount := final_mixture_concentration / 100 * total_volume
  salt_amount / salt_solution_volume * 100 = 45 := by
  sorry

end NUMINAMATH_CALUDE_salt_solution_concentration_l3733_373318


namespace NUMINAMATH_CALUDE_square_difference_given_sum_and_product_l3733_373314

theorem square_difference_given_sum_and_product (x y : ℝ) 
  (h1 : x^2 + y^2 = 10) (h2 : x * y = 2) : (x - y)^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_given_sum_and_product_l3733_373314


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3733_373366

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 3 - 4) :
  (4 - x) / (x - 2) / (x + 2 - 12 / (x - 2)) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3733_373366


namespace NUMINAMATH_CALUDE_range_of_f_l3733_373364

noncomputable def f (x : ℝ) : ℝ := 2 * (x + 7) * (x - 5) / (x + 7)

theorem range_of_f :
  Set.range f = {y | y < -24 ∨ y > -24} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3733_373364


namespace NUMINAMATH_CALUDE_new_person_weight_l3733_373387

/-- Given a group of 8 people where one person weighing 55 kg is replaced,
    if the average weight increases by 2.5 kg, then the new person weighs 75 kg. -/
theorem new_person_weight (initial_total : ℝ) (new_weight : ℝ) : 
  (initial_total - 55 + new_weight) / 8 = initial_total / 8 + 2.5 →
  new_weight = 75 := by
sorry

end NUMINAMATH_CALUDE_new_person_weight_l3733_373387


namespace NUMINAMATH_CALUDE_power_of_power_equals_base_l3733_373338

theorem power_of_power_equals_base (x : ℝ) (h : x > 0) : (x^(4/5))^(5/4) = x := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_equals_base_l3733_373338


namespace NUMINAMATH_CALUDE_cubic_sum_of_roots_l3733_373370

theorem cubic_sum_of_roots (p q r s : ℝ) : 
  (r^2 - p*r - q = 0) → (s^2 - p*s - q = 0) → (r^3 + s^3 = p^3 + 3*p*q) :=
by sorry

end NUMINAMATH_CALUDE_cubic_sum_of_roots_l3733_373370


namespace NUMINAMATH_CALUDE_first_apartment_utility_cost_l3733_373386

/-- Represents the monthly cost structure for an apartment --/
structure ApartmentCost where
  rent : ℝ
  utilities : ℝ
  drivingDistance : ℝ

/-- Calculates the total monthly cost for an apartment --/
def totalMonthlyCost (apt : ApartmentCost) (drivingCostPerMile : ℝ) (workingDays : ℝ) : ℝ :=
  apt.rent + apt.utilities + (apt.drivingDistance * drivingCostPerMile * workingDays)

/-- Theorem stating the utility cost of the first apartment --/
theorem first_apartment_utility_cost :
  let firstApt : ApartmentCost := { rent := 800, utilities := U, drivingDistance := 31 }
  let secondApt : ApartmentCost := { rent := 900, utilities := 200, drivingDistance := 21 }
  let drivingCostPerMile : ℝ := 0.58
  let workingDays : ℝ := 20
  totalMonthlyCost firstApt drivingCostPerMile workingDays - 
    totalMonthlyCost secondApt drivingCostPerMile workingDays = 76 →
  U = 259.60 := by
  sorry


end NUMINAMATH_CALUDE_first_apartment_utility_cost_l3733_373386


namespace NUMINAMATH_CALUDE_repeating_decimal_multiplication_l3733_373348

/-- Given a real number x where x = 0.000272727... (27 repeats indefinitely),
    prove that (10^5 - 10^3) * x = 27 -/
theorem repeating_decimal_multiplication (x : ℝ) : 
  (∃ (n : ℕ), x * 10^(n+5) - x * 10^5 = 27 * (10^n - 1) / 99) → 
  (10^5 - 10^3) * x = 27 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_multiplication_l3733_373348


namespace NUMINAMATH_CALUDE_division_equivalence_l3733_373317

theorem division_equivalence (h : 43 * 47 = 2021) : (-43) / (1 / 47) = -2021 := by
  sorry

end NUMINAMATH_CALUDE_division_equivalence_l3733_373317


namespace NUMINAMATH_CALUDE_polynomial_root_l3733_373320

theorem polynomial_root : ∃ (x : ℝ), 2 * x^5 + x^4 - 20 * x^3 - 10 * x^2 + 2 * x + 1 = 0 ∧ x = Real.sqrt 3 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_l3733_373320


namespace NUMINAMATH_CALUDE_normal_distribution_symmetry_l3733_373368

/-- A random variable following a normal distribution -/
structure NormalRV where
  μ : ℝ
  σ : ℝ
  hσ : σ > 0

/-- Probability of a normal random variable being less than or equal to a value -/
noncomputable def probability (X : NormalRV) (x : ℝ) : ℝ := sorry

theorem normal_distribution_symmetry (X : NormalRV) (h : X.μ = 2) (h_prob : probability X 4 = 0.84) :
  probability X 0 = 0.16 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_symmetry_l3733_373368


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l3733_373351

/-- The volume of a rectangular prism with face areas √2, √3, and √6 is √6 -/
theorem rectangular_prism_volume (a b c : ℝ) 
  (h1 : a * b = Real.sqrt 2)
  (h2 : b * c = Real.sqrt 3)
  (h3 : a * c = Real.sqrt 6) :
  a * b * c = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l3733_373351


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3733_373383

theorem quadratic_equation_solution :
  let x₁ : ℝ := (3 + Real.sqrt 3) / 2
  let x₂ : ℝ := (3 - Real.sqrt 3) / 2
  2 * x₁^2 - 6 * x₁ + 3 = 0 ∧ 2 * x₂^2 - 6 * x₂ + 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3733_373383


namespace NUMINAMATH_CALUDE_april_rose_price_l3733_373326

/-- Calculates the price per rose given the initial number of roses, remaining roses, and total earnings. -/
def price_per_rose (initial_roses : ℕ) (remaining_roses : ℕ) (total_earnings : ℕ) : ℚ :=
  total_earnings / (initial_roses - remaining_roses)

/-- Proves that the price per rose is $4 given the problem conditions. -/
theorem april_rose_price : price_per_rose 13 4 36 = 4 := by
  sorry

end NUMINAMATH_CALUDE_april_rose_price_l3733_373326


namespace NUMINAMATH_CALUDE_inscribed_cylinder_height_l3733_373393

theorem inscribed_cylinder_height (r_cylinder : ℝ) (r_sphere : ℝ) :
  r_cylinder = 3 →
  r_sphere = 7 →
  let h := 2 * (2 * Real.sqrt 10)
  h = 2 * Real.sqrt (r_sphere^2 - r_cylinder^2) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cylinder_height_l3733_373393


namespace NUMINAMATH_CALUDE_square_of_101_l3733_373363

theorem square_of_101 : 101 * 101 = 10201 := by
  sorry

end NUMINAMATH_CALUDE_square_of_101_l3733_373363


namespace NUMINAMATH_CALUDE_rotation_90_degrees_l3733_373394

def rotate90(z : ℂ) : ℂ := z * Complex.I

theorem rotation_90_degrees :
  rotate90 (-4 - 2 * Complex.I) = 2 - 4 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_rotation_90_degrees_l3733_373394


namespace NUMINAMATH_CALUDE_necklace_bead_count_l3733_373380

/-- Proves that the total number of beads in a necklace is 40 -/
theorem necklace_bead_count :
  let amethyst_count : ℕ := 7
  let amber_count : ℕ := 2 * amethyst_count
  let turquoise_count : ℕ := 19
  let total_count : ℕ := amethyst_count + amber_count + turquoise_count
  total_count = 40 := by sorry

end NUMINAMATH_CALUDE_necklace_bead_count_l3733_373380


namespace NUMINAMATH_CALUDE_q_squared_minus_one_div_fifteen_l3733_373331

/-- The largest prime with 2023 digits -/
def q : ℕ := sorry

/-- q is prime -/
axiom q_prime : Nat.Prime q

/-- q has 2023 digits -/
axiom q_digits : 10^2022 ≤ q ∧ q < 10^2023

/-- q is the largest prime with 2023 digits -/
axiom q_largest : ∀ p, Nat.Prime p → 10^2022 ≤ p ∧ p < 10^2023 → p ≤ q

theorem q_squared_minus_one_div_fifteen : 15 ∣ (q^2 - 1) := by sorry

end NUMINAMATH_CALUDE_q_squared_minus_one_div_fifteen_l3733_373331


namespace NUMINAMATH_CALUDE_probability_even_sum_is_one_third_l3733_373355

def digits : Finset ℕ := {2, 3, 5}

def is_valid_arrangement (a b c d : ℕ) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  (a = 2 ∧ b = 2) ∨ (a = 2 ∧ c = 2) ∨ (a = 2 ∧ d = 2) ∨
  (b = 2 ∧ c = 2) ∨ (b = 2 ∧ d = 2) ∨ (c = 2 ∧ d = 2)

def sum_first_last_even (a d : ℕ) : Prop :=
  (a + d) % 2 = 0

def count_valid_arrangements : ℕ := 12

def count_even_sum_arrangements : ℕ := 4

theorem probability_even_sum_is_one_third :
  (count_even_sum_arrangements : ℚ) / count_valid_arrangements = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_even_sum_is_one_third_l3733_373355


namespace NUMINAMATH_CALUDE_origin_on_circle_circle_through_P_l3733_373349

-- Define the parabola C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 2 * p.1}

-- Define the point (2, 0)
def point_2_0 : ℝ × ℝ := (2, 0)

-- Define the line l passing through (2, 0)
def l (k : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = k * (p.1 - 2)}

-- Define the intersection points A and B
def A (k : ℝ) : ℝ × ℝ := sorry
def B (k : ℝ) : ℝ × ℝ := sorry

-- Define the circle M with diameter AB
def M (k : ℝ) : Set (ℝ × ℝ) := sorry

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define point P
def P : ℝ × ℝ := (4, -2)

-- Theorem 1: The origin O is on circle M
theorem origin_on_circle (k : ℝ) : O ∈ M k := sorry

-- Theorem 2: If M passes through P, then l and M have specific equations
theorem circle_through_P (k : ℝ) (h : P ∈ M k) :
  (k = -2 ∧ l k = {p : ℝ × ℝ | p.2 = -2 * p.1 + 4} ∧ 
   M k = {p : ℝ × ℝ | (p.1 - 9/4)^2 + (p.2 + 1/2)^2 = 85/16}) ∨
  (k = 1 ∧ l k = {p : ℝ × ℝ | p.2 = p.1 - 2} ∧ 
   M k = {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 1)^2 = 10}) := sorry

end NUMINAMATH_CALUDE_origin_on_circle_circle_through_P_l3733_373349


namespace NUMINAMATH_CALUDE_fenced_square_cost_l3733_373381

/-- A square with fenced sides -/
structure FencedSquare where
  side_cost : ℕ
  sides : ℕ

/-- The total cost of fencing a square -/
def total_fencing_cost (s : FencedSquare) : ℕ :=
  s.side_cost * s.sides

/-- Theorem: The total cost of fencing a square with 4 sides at $69 per side is $276 -/
theorem fenced_square_cost :
  ∀ (s : FencedSquare), s.side_cost = 69 → s.sides = 4 → total_fencing_cost s = 276 :=
by
  sorry

end NUMINAMATH_CALUDE_fenced_square_cost_l3733_373381


namespace NUMINAMATH_CALUDE_magnitude_of_z_l3733_373367

/-- The magnitude of the complex number z = (1+i)/i is equal to √2 -/
theorem magnitude_of_z (i : ℂ) (h : i^2 = -1) : 
  Complex.abs ((1 + i) / i) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l3733_373367


namespace NUMINAMATH_CALUDE_three_prime_divisors_l3733_373360

theorem three_prime_divisors (p : Nat) (h_prime : Prime p) 
  (h_cong : (2^(p-1)) % (p^2) = 1) (n : Nat) : 
  ∃ (q₁ q₂ q₃ : Nat), Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ 
  q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₂ ≠ q₃ ∧
  (q₁ ∣ ((p-1) * (Nat.factorial p + 2^n))) ∧
  (q₂ ∣ ((p-1) * (Nat.factorial p + 2^n))) ∧
  (q₃ ∣ ((p-1) * (Nat.factorial p + 2^n))) := by
sorry

end NUMINAMATH_CALUDE_three_prime_divisors_l3733_373360


namespace NUMINAMATH_CALUDE_garden_feet_count_l3733_373342

/-- Calculates the total number of feet in a garden with dogs and ducks -/
def total_feet (num_dogs : ℕ) (num_ducks : ℕ) (feet_per_dog : ℕ) (feet_per_duck : ℕ) : ℕ :=
  num_dogs * feet_per_dog + num_ducks * feet_per_duck

/-- Theorem: The total number of feet in a garden with 6 dogs and 2 ducks is 28 -/
theorem garden_feet_count : total_feet 6 2 4 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_garden_feet_count_l3733_373342


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l3733_373397

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a = 1, b = √2, and A = 30°, then B = 45° or B = 135°. -/
theorem triangle_angle_calculation (a b c A B C : Real) : 
  a = 1 → b = Real.sqrt 2 → A = π / 6 → 
  (B = π / 4 ∨ B = 3 * π / 4) := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_calculation_l3733_373397


namespace NUMINAMATH_CALUDE_watermelon_price_l3733_373356

theorem watermelon_price : 
  let base_price : ℕ := 5000
  let additional_cost : ℕ := 200
  let total_price : ℕ := base_price + additional_cost
  let price_in_thousands : ℚ := total_price / 1000
  price_in_thousands = 5.2 := by sorry

end NUMINAMATH_CALUDE_watermelon_price_l3733_373356


namespace NUMINAMATH_CALUDE_cubic_difference_zero_l3733_373399

theorem cubic_difference_zero (a b : ℝ) (h1 : a - b = 1) (h2 : a * b ≠ 0) :
  a^3 - b^3 - a*b - a^2 - b^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_difference_zero_l3733_373399


namespace NUMINAMATH_CALUDE_intersection_of_logarithmic_graphs_l3733_373371

theorem intersection_of_logarithmic_graphs :
  ∃! x : ℝ, x > 0 ∧ 3 * Real.log x = Real.log (3 * x) := by sorry

end NUMINAMATH_CALUDE_intersection_of_logarithmic_graphs_l3733_373371


namespace NUMINAMATH_CALUDE_negation_of_forall_geq_two_squared_geq_four_l3733_373341

theorem negation_of_forall_geq_two_squared_geq_four :
  (¬ (∀ x : ℝ, x ≥ 2 → x^2 ≥ 4)) ↔ (∃ x₀ : ℝ, x₀ ≥ 2 ∧ x₀^2 < 4) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_geq_two_squared_geq_four_l3733_373341
