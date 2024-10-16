import Mathlib

namespace NUMINAMATH_CALUDE_square_perimeter_from_area_l3942_394203

-- Define a square with a given area
def Square (area : ℝ) : Type :=
  { side : ℝ // side * side = area }

-- Define the perimeter of a square
def perimeter (s : Square 625) : ℝ :=
  4 * s.val

-- Theorem statement
theorem square_perimeter_from_area :
  ∀ s : Square 625, perimeter s = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_from_area_l3942_394203


namespace NUMINAMATH_CALUDE_charity_event_probability_l3942_394299

theorem charity_event_probability :
  let n : ℕ := 5  -- number of students
  let d : ℕ := 2  -- number of days (Saturday and Sunday)
  let total_outcomes : ℕ := d^n
  let same_day_outcomes : ℕ := 2  -- all choose Saturday or all choose Sunday
  let both_days_outcomes : ℕ := total_outcomes - same_day_outcomes
  (both_days_outcomes : ℚ) / total_outcomes = 15 / 16 :=
by sorry

end NUMINAMATH_CALUDE_charity_event_probability_l3942_394299


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3942_394275

/-- Given a geometric sequence {a_n} with common ratio q and sum of first n terms S_n,
    prove that if a_3 = 2S_2 + 1 and a_4 = 2S_3 + 1, then q = 3. -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- a_n is a geometric sequence with common ratio q
  (∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) →  -- S_n is the sum of first n terms
  a 3 = 2 * S 2 + 1 →  -- a_3 = 2S_2 + 1
  a 4 = 2 * S 3 + 1 →  -- a_4 = 2S_3 + 1
  q = 3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3942_394275


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3942_394256

theorem inequality_system_solution (m : ℝ) : 
  (∀ x : ℝ, (x + 2 > 2*x - 6 ∧ x < m) ↔ x < 8) → m ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3942_394256


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3942_394240

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (3 - i) / (2 + i) = 1 - i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3942_394240


namespace NUMINAMATH_CALUDE_square_and_parallelogram_area_l3942_394253

theorem square_and_parallelogram_area : 
  let square_side : ℝ := 3
  let parallelogram_base : ℝ := 3
  let parallelogram_height : ℝ := 2
  let square_area := square_side * square_side
  let parallelogram_area := parallelogram_base * parallelogram_height
  square_area + parallelogram_area = 15 := by sorry

end NUMINAMATH_CALUDE_square_and_parallelogram_area_l3942_394253


namespace NUMINAMATH_CALUDE_restaurant_glasses_count_l3942_394272

/-- Represents the number of glasses in a restaurant with two box sizes --/
def total_glasses (small_box_count : ℕ) (large_box_count : ℕ) : ℕ :=
  small_box_count * 12 + large_box_count * 16

/-- Represents the average number of glasses per box --/
def average_glasses_per_box (small_box_count : ℕ) (large_box_count : ℕ) : ℚ :=
  (total_glasses small_box_count large_box_count : ℚ) / (small_box_count + large_box_count : ℚ)

theorem restaurant_glasses_count :
  ∃ (small_box_count : ℕ) (large_box_count : ℕ),
    small_box_count > 0 ∧
    large_box_count = small_box_count + 16 ∧
    average_glasses_per_box small_box_count large_box_count = 15 ∧
    total_glasses small_box_count large_box_count = 480 := by
  sorry


end NUMINAMATH_CALUDE_restaurant_glasses_count_l3942_394272


namespace NUMINAMATH_CALUDE_altitude_to_largerBase_ratio_l3942_394285

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- The length of the smaller base -/
  smallerBase : ℝ
  /-- The length of the larger base -/
  largerBase : ℝ
  /-- The length of the altitude -/
  altitude : ℝ
  /-- The smaller base is positive -/
  smallerBase_pos : 0 < smallerBase
  /-- The larger base is positive -/
  largerBase_pos : 0 < largerBase
  /-- The altitude is positive -/
  altitude_pos : 0 < altitude
  /-- The smaller base is less than the larger base -/
  smallerBase_lt_largerBase : smallerBase < largerBase
  /-- The smaller base equals the length of a diagonal -/
  smallerBase_eq_diagonal : smallerBase = Real.sqrt (smallerBase^2 + altitude^2)
  /-- The larger base equals twice the altitude -/
  largerBase_eq_twice_altitude : largerBase = 2 * altitude

/-- The ratio of the altitude to the larger base is 1/2 -/
theorem altitude_to_largerBase_ratio (t : IsoscelesTrapezoid) : 
  t.altitude / t.largerBase = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_altitude_to_largerBase_ratio_l3942_394285


namespace NUMINAMATH_CALUDE_work_time_ratio_l3942_394271

/-- Proves that the ratio of Celeste's work time to Bianca's work time is 2:1 given the specified conditions. -/
theorem work_time_ratio (bianca_time : ℝ) (celeste_multiplier : ℝ) :
  bianca_time = 12.5 →
  bianca_time * celeste_multiplier + (bianca_time * celeste_multiplier - 8.5) + bianca_time = 54 →
  celeste_multiplier = 2 := by
  sorry

#check work_time_ratio

end NUMINAMATH_CALUDE_work_time_ratio_l3942_394271


namespace NUMINAMATH_CALUDE_string_length_problem_l3942_394237

theorem string_length_problem (num_strings : ℕ) (total_length : ℝ) (h1 : num_strings = 7) (h2 : total_length = 98) :
  total_length / num_strings = 14 := by
  sorry

end NUMINAMATH_CALUDE_string_length_problem_l3942_394237


namespace NUMINAMATH_CALUDE_tens_digit_of_8_power_23_l3942_394297

theorem tens_digit_of_8_power_23 : ∃ n : ℕ, 8^23 = 10 * n + 12 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_8_power_23_l3942_394297


namespace NUMINAMATH_CALUDE_original_fraction_problem_l3942_394250

theorem original_fraction_problem (N D : ℚ) :
  (N > 0) →
  (D > 0) →
  ((1.4 * N) / (0.5 * D) = 4/5) →
  (N / D = 2/7) :=
by sorry

end NUMINAMATH_CALUDE_original_fraction_problem_l3942_394250


namespace NUMINAMATH_CALUDE_cylinder_volume_l3942_394269

/-- Given a cylinder with height 2 and lateral surface area 4π, its volume is 2π -/
theorem cylinder_volume (h : ℝ) (lateral_area : ℝ) (volume : ℝ) : 
  h = 2 → lateral_area = 4 * Real.pi → volume = 2 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_l3942_394269


namespace NUMINAMATH_CALUDE_salary_change_l3942_394264

theorem salary_change (S : ℝ) : 
  let increase := S * 1.2
  let decrease := increase * 0.8
  decrease = S * 0.96 := by sorry

end NUMINAMATH_CALUDE_salary_change_l3942_394264


namespace NUMINAMATH_CALUDE_books_read_l3942_394226

theorem books_read (total : ℕ) (remaining : ℕ) (read : ℕ) : 
  total = 14 → remaining = 6 → read = total - remaining → read = 8 := by
sorry

end NUMINAMATH_CALUDE_books_read_l3942_394226


namespace NUMINAMATH_CALUDE_domestic_needs_fraction_l3942_394274

def total_income : ℚ := 200
def provident_fund_rate : ℚ := 1/16
def insurance_premium_rate : ℚ := 1/15
def bank_deposit : ℚ := 50

def remaining_after_provident_fund : ℚ := total_income * (1 - provident_fund_rate)
def remaining_after_insurance : ℚ := remaining_after_provident_fund * (1 - insurance_premium_rate)

theorem domestic_needs_fraction :
  (remaining_after_insurance - bank_deposit) / remaining_after_insurance = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_domestic_needs_fraction_l3942_394274


namespace NUMINAMATH_CALUDE_labourer_absence_solution_l3942_394229

/-- Represents the problem of calculating a labourer's absence days --/
def LabourerAbsence (total_days work_pay absence_fine total_received : ℚ) : Prop :=
  ∃ (days_worked days_absent : ℚ),
    days_worked + days_absent = total_days ∧
    work_pay * days_worked - absence_fine * days_absent = total_received ∧
    days_absent = 5

/-- Theorem stating the solution to the labourer absence problem --/
theorem labourer_absence_solution :
  LabourerAbsence 25 2 (1/2) (75/2) :=
sorry

end NUMINAMATH_CALUDE_labourer_absence_solution_l3942_394229


namespace NUMINAMATH_CALUDE_product_of_ab_values_l3942_394267

theorem product_of_ab_values (a b : ℝ) (h1 : a + 1/b = 4) (h2 : 1/a + b = 16/15) : 
  (5/3 * 3/5 : ℝ) = 1 := by sorry

end NUMINAMATH_CALUDE_product_of_ab_values_l3942_394267


namespace NUMINAMATH_CALUDE_sum_of_quadratic_solutions_l3942_394209

theorem sum_of_quadratic_solutions : 
  let f (x : ℝ) := x^2 - 4*x - 14 - (3*x + 16)
  let solutions := {x : ℝ | f x = 0}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ solutions ∧ x₂ ∈ solutions ∧ x₁ ≠ x₂ ∧ x₁ + x₂ = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_quadratic_solutions_l3942_394209


namespace NUMINAMATH_CALUDE_maple_pine_height_difference_l3942_394216

/-- The height of the pine tree in feet -/
def pine_height : ℚ := 24 + 1/4

/-- The height of the maple tree in feet -/
def maple_height : ℚ := 31 + 2/3

/-- The difference in height between the maple and pine trees -/
def height_difference : ℚ := maple_height - pine_height

theorem maple_pine_height_difference :
  height_difference = 7 + 5/12 := by sorry

end NUMINAMATH_CALUDE_maple_pine_height_difference_l3942_394216


namespace NUMINAMATH_CALUDE_athlete_exercise_time_l3942_394204

/-- Prove that given an athlete who burns 10 calories per minute while running,
    4 calories per minute while walking, burns 450 calories in total,
    and spends 35 minutes running, the total exercise time is 60 minutes. -/
theorem athlete_exercise_time
  (calories_per_minute_running : ℕ)
  (calories_per_minute_walking : ℕ)
  (total_calories_burned : ℕ)
  (time_running : ℕ)
  (h1 : calories_per_minute_running = 10)
  (h2 : calories_per_minute_walking = 4)
  (h3 : total_calories_burned = 450)
  (h4 : time_running = 35) :
  time_running + (total_calories_burned - calories_per_minute_running * time_running) / calories_per_minute_walking = 60 :=
by sorry

end NUMINAMATH_CALUDE_athlete_exercise_time_l3942_394204


namespace NUMINAMATH_CALUDE_imaginary_part_of_3_minus_4i_l3942_394210

theorem imaginary_part_of_3_minus_4i :
  Complex.im (3 - 4 * Complex.I) = -4 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_3_minus_4i_l3942_394210


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3942_394214

theorem pure_imaginary_complex_number (x : ℝ) : 
  (((x^2 - 4) : ℂ) + (x^2 + 3*x + 2)*I).im ≠ 0 ∧ 
  (((x^2 - 4) : ℂ) + (x^2 + 3*x + 2)*I).re = 0 → 
  x = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3942_394214


namespace NUMINAMATH_CALUDE_first_discount_percentage_l3942_394235

theorem first_discount_percentage 
  (original_price : ℝ) 
  (final_price : ℝ) 
  (second_discount : ℝ) : 
  original_price = 400 →
  final_price = 342 →
  second_discount = 5 →
  ∃ (first_discount : ℝ),
    final_price = original_price * (100 - first_discount) / 100 * (100 - second_discount) / 100 ∧
    first_discount = 10 := by
  sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l3942_394235


namespace NUMINAMATH_CALUDE_root_equation_r_values_l3942_394233

theorem root_equation_r_values (r : ℤ) : 
  (∃ x y : ℤ, x > 0 ∧ y > 0 ∧ 
    r * x^2 - (2*r + 7) * x + r + 7 = 0 ∧
    r * y^2 - (2*r + 7) * y + r + 7 = 0) →
  r = 7 ∨ r = 0 ∨ r = 1 :=
by sorry

end NUMINAMATH_CALUDE_root_equation_r_values_l3942_394233


namespace NUMINAMATH_CALUDE_product_sum_fractions_l3942_394227

theorem product_sum_fractions : (3 * 4 * 5) * (1/3 + 1/4 + 1/5) = 47 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_fractions_l3942_394227


namespace NUMINAMATH_CALUDE_odd_function_property_l3942_394248

-- Define an odd function f on ℝ
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- State the theorem
theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h1 : f 1 = 2) 
  (h2 : f 2 = 3) : 
  f (f (-1)) = -3 := by
  sorry


end NUMINAMATH_CALUDE_odd_function_property_l3942_394248


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_l3942_394268

def polynomial (x : ℝ) : ℝ := 4 * (x^3 - 2*x^2) + 3 * (x^2 - x^3 + 2*x^4) - 5 * (x^4 - 2*x^3)

theorem coefficient_of_x_cubed :
  ∃ (a b c d : ℝ), ∀ x, polynomial x = a*x^4 + 11*x^3 + b*x^2 + c*x + d :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_l3942_394268


namespace NUMINAMATH_CALUDE_axis_of_symmetry_sin_l3942_394228

open Real

theorem axis_of_symmetry_sin (φ : ℝ) :
  (∀ x, |sin (2*x + φ)| ≤ |sin (π/3 + φ)|) →
  ∃ k : ℤ, 2*(2*π/3) + φ = k*π + π/2 :=
by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_sin_l3942_394228


namespace NUMINAMATH_CALUDE_abs_inequality_l3942_394221

theorem abs_inequality (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| := by
  sorry

end NUMINAMATH_CALUDE_abs_inequality_l3942_394221


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3942_394231

theorem contrapositive_equivalence (p q : Prop) :
  (¬p → q) → (¬q → p) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3942_394231


namespace NUMINAMATH_CALUDE_five_digit_division_sum_l3942_394219

theorem five_digit_division_sum (ABCDE : ℕ) : 
  ABCDE ≥ 10000 ∧ ABCDE < 100000 ∧ ABCDE % 6 = 0 ∧ ABCDE / 6 = 13579 →
  (ABCDE / 100) + (ABCDE % 100) = 888 := by
sorry

end NUMINAMATH_CALUDE_five_digit_division_sum_l3942_394219


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3942_394293

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2*y = 1) :
  (1/x + 1/(2*y) ≥ 4) ∧ (1/x + 1/(2*y) = 4 ↔ x = 2*y) := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3942_394293


namespace NUMINAMATH_CALUDE_seminar_fee_calculation_l3942_394244

/-- Proves that the regular seminar fee is $150 given the problem conditions --/
theorem seminar_fee_calculation (F : ℝ) : 
  (∃ (total_spent discounted_fee : ℝ),
    -- 5% discount applied
    discounted_fee = F * 0.95 ∧
    -- 10 teachers registered
    -- $10 food allowance per teacher
    total_spent = 10 * discounted_fee + 10 * 10 ∧
    -- Total spent is $1525
    total_spent = 1525) →
  F = 150 := by
  sorry

end NUMINAMATH_CALUDE_seminar_fee_calculation_l3942_394244


namespace NUMINAMATH_CALUDE_ellipse_equation_l3942_394282

/-- An ellipse with foci and points satisfying certain conditions -/
structure Ellipse where
  a : ℝ
  b : ℝ
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  h₁ : a > b
  h₂ : b > 0
  h₃ : A.1^2 / a^2 + A.2^2 / b^2 = 1  -- A is on the ellipse
  h₄ : B.1^2 / a^2 + B.2^2 / b^2 = 1  -- B is on the ellipse
  h₅ : (A.1 - B.1) * (F₁.1 - F₂.1) + (A.2 - B.2) * (F₁.2 - F₂.2) = 0  -- AB ⟂ F₁F₂
  h₆ : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 16  -- |AB| = 4
  h₇ : (F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2 = 12  -- |F₁F₂| = 2√3

/-- The equation of the ellipse is x²/9 + y²/6 = 1 -/
theorem ellipse_equation (e : Ellipse) : e.a^2 = 9 ∧ e.b^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3942_394282


namespace NUMINAMATH_CALUDE_no_matrix_transformation_l3942_394276

theorem no_matrix_transformation (a b c d : ℝ) : 
  ¬ ∃ (N : Matrix (Fin 2) (Fin 2) ℝ), 
    N • !![a, b; c, d] = !![d, c; b, a] := by
  sorry

end NUMINAMATH_CALUDE_no_matrix_transformation_l3942_394276


namespace NUMINAMATH_CALUDE_tv_show_average_episodes_l3942_394243

theorem tv_show_average_episodes (total_years : ℕ) (seasons_15 : ℕ) (seasons_20 : ℕ) (seasons_12 : ℕ)
  (h1 : total_years = 14)
  (h2 : seasons_15 = 8)
  (h3 : seasons_20 = 4)
  (h4 : seasons_12 = 2) :
  (seasons_15 * 15 + seasons_20 * 20 + seasons_12 * 12) / total_years = 16 := by
  sorry

end NUMINAMATH_CALUDE_tv_show_average_episodes_l3942_394243


namespace NUMINAMATH_CALUDE_min_ratio_of_valid_partition_l3942_394206

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def is_valid_partition (partition : List ℕ × List ℕ) : Prop :=
  let (group1, group2) := partition
  (group1 ++ group2).toFinset = Finset.range 30
  ∧ (group1.prod % group2.prod = 0)

def ratio (partition : List ℕ × List ℕ) : ℚ :=
  let (group1, group2) := partition
  (group1.prod : ℚ) / (group2.prod : ℚ)

theorem min_ratio_of_valid_partition :
  ∀ partition : List ℕ × List ℕ,
    is_valid_partition partition →
    ratio partition ≥ 1077205 :=
sorry

end NUMINAMATH_CALUDE_min_ratio_of_valid_partition_l3942_394206


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3942_394230

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (∃ r s : ℝ, (36 - 18 * x - x^2 = 0 ↔ x = r ∨ x = s) ∧ r + s = 18) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3942_394230


namespace NUMINAMATH_CALUDE_sqrt_x_minus_2_meaningful_l3942_394266

theorem sqrt_x_minus_2_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_2_meaningful_l3942_394266


namespace NUMINAMATH_CALUDE_milk_consumption_l3942_394279

/-- The amount of regular milk consumed by Mitch's family in 1 week -/
def regular_milk : ℝ := 0.5

/-- The amount of soy milk consumed by Mitch's family in 1 week -/
def soy_milk : ℝ := 0.1

/-- The total amount of milk consumed by Mitch's family in 1 week -/
def total_milk : ℝ := regular_milk + soy_milk

theorem milk_consumption :
  total_milk = 0.6 := by sorry

end NUMINAMATH_CALUDE_milk_consumption_l3942_394279


namespace NUMINAMATH_CALUDE_dawn_savings_l3942_394252

/-- Dawn's financial situation --/
def dawn_finances : Prop :=
  let annual_income : ℝ := 48000
  let monthly_income : ℝ := annual_income / 12
  let tax_rate : ℝ := 0.20
  let variable_expense_rate : ℝ := 0.30
  let stock_investment_rate : ℝ := 0.05
  let retirement_contribution_rate : ℝ := 0.15
  let savings_rate : ℝ := 0.10
  let after_tax_income : ℝ := monthly_income * (1 - tax_rate)
  let variable_expenses : ℝ := after_tax_income * variable_expense_rate
  let stock_investment : ℝ := after_tax_income * stock_investment_rate
  let retirement_contribution : ℝ := after_tax_income * retirement_contribution_rate
  let total_deductions : ℝ := variable_expenses + stock_investment + retirement_contribution
  let remaining_income : ℝ := after_tax_income - total_deductions
  let monthly_savings : ℝ := remaining_income * savings_rate
  monthly_savings = 160

theorem dawn_savings : dawn_finances := by
  sorry

end NUMINAMATH_CALUDE_dawn_savings_l3942_394252


namespace NUMINAMATH_CALUDE_trig_equation_roots_l3942_394289

open Real

theorem trig_equation_roots (α β : ℝ) : 
  0 < α ∧ α < π ∧ 0 < β ∧ β < π →
  (∃ x y : ℝ, x^2 - 5*x + 6 = 0 ∧ y^2 - 5*y + 6 = 0 ∧ x = tan α ∧ y = tan β) →
  tan (α + β) = -1 ∧ cos (α - β) = (7 * Real.sqrt 2) / 10 := by
  sorry

end NUMINAMATH_CALUDE_trig_equation_roots_l3942_394289


namespace NUMINAMATH_CALUDE_salary_savings_percentage_l3942_394294

theorem salary_savings_percentage 
  (salary : ℝ) 
  (savings_after_increase : ℝ) 
  (expense_increase_percentage : ℝ) :
  salary = 5750 →
  savings_after_increase = 230 →
  expense_increase_percentage = 20 →
  ∃ (savings_percentage : ℝ),
    savings_percentage = 20 ∧
    savings_after_increase = salary - (1 + expense_increase_percentage / 100) * ((100 - savings_percentage) / 100 * salary) :=
by sorry

end NUMINAMATH_CALUDE_salary_savings_percentage_l3942_394294


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l3942_394249

/-- The function f satisfying the given conditions -/
noncomputable def f (x : ℝ) : ℝ := -5 * (4^x - 5^x)

/-- Theorem stating that f satisfies the required conditions -/
theorem f_satisfies_conditions :
  (f 1 = 5) ∧
  (∀ x y : ℝ, f (x + y) = 4^y * f x + 5^x * f y) :=
by sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l3942_394249


namespace NUMINAMATH_CALUDE_f_of_3_eq_5_l3942_394200

/-- The function f defined on ℝ -/
def f : ℝ → ℝ := fun x ↦ 2 * x - 1

/-- Theorem: f(3) = 5 -/
theorem f_of_3_eq_5 : f 3 = 5 := by sorry

end NUMINAMATH_CALUDE_f_of_3_eq_5_l3942_394200


namespace NUMINAMATH_CALUDE_environmental_policy_support_percentage_l3942_394281

theorem environmental_policy_support_percentage : 
  let total_surveyed : ℕ := 150 + 850
  let men_surveyed : ℕ := 150
  let women_surveyed : ℕ := 850
  let men_support_percentage : ℚ := 70 / 100
  let women_support_percentage : ℚ := 75 / 100
  let men_supporters : ℚ := men_surveyed * men_support_percentage
  let women_supporters : ℚ := women_surveyed * women_support_percentage
  let total_supporters : ℚ := men_supporters + women_supporters
  let overall_support_percentage : ℚ := total_supporters / total_surveyed * 100
  overall_support_percentage = 743 / 10 := by sorry

end NUMINAMATH_CALUDE_environmental_policy_support_percentage_l3942_394281


namespace NUMINAMATH_CALUDE_number_problem_l3942_394236

theorem number_problem : ∃ x : ℝ, 0.65 * x = 0.05 * 60 + 23 ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3942_394236


namespace NUMINAMATH_CALUDE_work_completion_time_l3942_394283

/-- The time taken to complete a work given two workers with different rates and a partial work completion scenario -/
theorem work_completion_time
  (amit_rate : ℚ)
  (ananthu_rate : ℚ)
  (amit_days : ℕ)
  (h_amit_rate : amit_rate = 1 / 15)
  (h_ananthu_rate : ananthu_rate = 1 / 45)
  (h_amit_days : amit_days = 3)
  : ∃ (total_days : ℕ), total_days = amit_days + ⌈(1 - amit_rate * amit_days) / ananthu_rate⌉ ∧ total_days = 39 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l3942_394283


namespace NUMINAMATH_CALUDE_original_number_l3942_394222

theorem original_number (x : ℝ) : 3 * (2 * x + 5) = 111 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l3942_394222


namespace NUMINAMATH_CALUDE_winning_candidate_percentage_l3942_394258

def candidate1_votes : ℕ := 1136
def candidate2_votes : ℕ := 8236
def candidate3_votes : ℕ := 11628

def total_votes : ℕ := candidate1_votes + candidate2_votes + candidate3_votes

def winning_votes : ℕ := max candidate1_votes (max candidate2_votes candidate3_votes)

theorem winning_candidate_percentage :
  (winning_votes : ℚ) / (total_votes : ℚ) * 100 = 58.14 := by
  sorry

end NUMINAMATH_CALUDE_winning_candidate_percentage_l3942_394258


namespace NUMINAMATH_CALUDE_circle_area_theorem_l3942_394211

theorem circle_area_theorem (r : ℝ) (h : 8 / (2 * Real.pi * r) = (2 * r)^2) :
  π * r^2 = Real.pi^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_theorem_l3942_394211


namespace NUMINAMATH_CALUDE_ivy_cupcakes_l3942_394261

def morning_cupcakes : ℕ := 20
def afternoon_difference : ℕ := 15

def total_cupcakes : ℕ := morning_cupcakes + (morning_cupcakes + afternoon_difference)

theorem ivy_cupcakes : total_cupcakes = 55 := by
  sorry

end NUMINAMATH_CALUDE_ivy_cupcakes_l3942_394261


namespace NUMINAMATH_CALUDE_two_heads_probability_l3942_394292

/-- The probability of getting heads on a single flip of a fair coin -/
def prob_heads : ℚ := 1/2

/-- The probability of getting heads on both of the first two flips of a fair coin -/
def prob_two_heads : ℚ := prob_heads * prob_heads

theorem two_heads_probability : prob_two_heads = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_two_heads_probability_l3942_394292


namespace NUMINAMATH_CALUDE_three_operations_to_one_tile_l3942_394288

/-- Represents the set of tiles -/
def TileSet := Finset Nat

/-- The operation of removing perfect squares and renumbering -/
def remove_squares_and_renumber (s : TileSet) : TileSet :=
  sorry

/-- The initial set of tiles from 1 to 49 -/
def initial_set : TileSet :=
  sorry

/-- Applies the operation n times -/
def apply_n_times (n : Nat) (s : TileSet) : TileSet :=
  sorry

theorem three_operations_to_one_tile :
  ∃ (n : Nat), n = 3 ∧ (apply_n_times n initial_set).card = 1 ∧
  ∀ (m : Nat), m < n → (apply_n_times m initial_set).card > 1 :=
sorry

end NUMINAMATH_CALUDE_three_operations_to_one_tile_l3942_394288


namespace NUMINAMATH_CALUDE_katie_sugar_calculation_l3942_394217

/-- Given a recipe that requires a total amount of sugar and an amount already added,
    calculate the remaining amount needed. -/
def remaining_sugar (total : ℝ) (added : ℝ) : ℝ :=
  total - added

theorem katie_sugar_calculation :
  let total_required : ℝ := 3
  let already_added : ℝ := 0.5
  remaining_sugar total_required already_added = 2.5 := by
sorry

end NUMINAMATH_CALUDE_katie_sugar_calculation_l3942_394217


namespace NUMINAMATH_CALUDE_ratio_problem_l3942_394247

theorem ratio_problem (a b x m : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a / b = 4 / 5) 
  (h4 : x = a + 0.25 * a) (h5 : m = b - 0.6 * b) : m / x = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3942_394247


namespace NUMINAMATH_CALUDE_triangle_inequalities_triangle_equality_condition_l3942_394254

/-- Triangle properties -/
structure Triangle :=
  (a b c : ℝ)
  (r R : ℝ)
  (h_a h_b h_c : ℝ)
  (β_a β_b β_c : ℝ)
  (m_a m_b m_c : ℝ)
  (r_a r_b r_c : ℝ)
  (p : ℝ)

/-- Main theorem -/
theorem triangle_inequalities (t : Triangle) :
  (9 * t.r ≤ t.h_a + t.h_b + t.h_c) ∧
  (t.h_a + t.h_b + t.h_c ≤ t.β_a + t.β_b + t.β_c) ∧
  (t.β_a + t.β_b + t.β_c ≤ t.m_a + t.m_b + t.m_c) ∧
  (t.m_a + t.m_b + t.m_c ≤ 9/2 * t.R) ∧
  (t.β_a + t.β_b + t.β_c ≤ Real.sqrt (t.r_a * t.r_b) + Real.sqrt (t.r_b * t.r_c) + Real.sqrt (t.r_c * t.r_a)) ∧
  (Real.sqrt (t.r_a * t.r_b) + Real.sqrt (t.r_b * t.r_c) + Real.sqrt (t.r_c * t.r_a) ≤ t.p * Real.sqrt 3) ∧
  (t.p * Real.sqrt 3 ≤ t.r_a + t.r_b + t.r_c) ∧
  (t.r_a + t.r_b + t.r_c = t.r + 4 * t.R) ∧
  (27 * t.r^2 ≤ t.h_a^2 + t.h_b^2 + t.h_c^2) ∧
  (t.h_a^2 + t.h_b^2 + t.h_c^2 ≤ t.β_a^2 + t.β_b^2 + t.β_c^2) ∧
  (t.β_a^2 + t.β_b^2 + t.β_c^2 ≤ t.p^2) ∧
  (t.p^2 ≤ t.m_a^2 + t.m_b^2 + t.m_c^2) ∧
  (t.m_a^2 + t.m_b^2 + t.m_c^2 = 3/4 * (t.a^2 + t.b^2 + t.c^2)) ∧
  (3/4 * (t.a^2 + t.b^2 + t.c^2) ≤ 27/4 * t.R^2) ∧
  (1/t.r = 1/t.r_a + 1/t.r_b + 1/t.r_c) ∧
  (1/t.r = 1/t.h_a + 1/t.h_b + 1/t.h_c) ∧
  (1/t.h_a + 1/t.h_b + 1/t.h_c ≥ 1/t.β_a + 1/t.β_b + 1/t.β_c) ∧
  (1/t.β_a + 1/t.β_b + 1/t.β_c ≥ 1/t.m_a + 1/t.m_b + 1/t.m_c) ∧
  (1/t.m_a + 1/t.m_b + 1/t.m_c ≥ 2/t.R) :=
sorry

/-- Equality condition -/
theorem triangle_equality_condition (t : Triangle) :
  (9 * t.r = t.h_a + t.h_b + t.h_c) ∧
  (t.h_a + t.h_b + t.h_c = t.β_a + t.β_b + t.β_c) ∧
  (t.β_a + t.β_b + t.β_c = t.m_a + t.m_b + t.m_c) ∧
  (t.m_a + t.m_b + t.m_c = 9/2 * t.R) ∧
  (t.β_a + t.β_b + t.β_c = Real.sqrt (t.r_a * t.r_b) + Real.sqrt (t.r_b * t.r_c) + Real.sqrt (t.r_c * t.r_a)) ∧
  (Real.sqrt (t.r_a * t.r_b) + Real.sqrt (t.r_b * t.r_c) + Real.sqrt (t.r_c * t.r_a) = t.p * Real.sqrt 3) ∧
  (t.p * Real.sqrt 3 = t.r_a + t.r_b + t.r_c) ∧
  (27 * t.r^2 = t.h_a^2 + t.h_b^2 + t.h_c^2) ∧
  (t.h_a^2 + t.h_b^2 + t.h_c^2 = t.β_a^2 + t.β_b^2 + t.β_c^2) ∧
  (t.β_a^2 + t.β_b^2 + t.β_c^2 = t.p^2) ∧
  (t.p^2 = t.m_a^2 + t.m_b^2 + t.m_c^2) ∧
  (3/4 * (t.a^2 + t.b^2 + t.c^2) = 27/4 * t.R^2) ∧
  (1/t.h_a + 1/t.h_b + 1/t.h_c = 1/t.β_a + 1/t.β_b + 1/t.β_c) ∧
  (1/t.β_a + 1/t.β_b + 1/t.β_c = 1/t.m_a + 1/t.m_b + 1/t.m_c) ∧
  (1/t.m_a + 1/t.m_b + 1/t.m_c = 2/t.R) ↔
  (t.a = t.b ∧ t.b = t.c) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequalities_triangle_equality_condition_l3942_394254


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3942_394213

theorem chess_tournament_games (n : Nat) (h : n = 5) : 
  n * (n - 1) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3942_394213


namespace NUMINAMATH_CALUDE_ratio_problem_l3942_394246

theorem ratio_problem (first_part : ℝ) (ratio_percent : ℝ) (second_part : ℝ) :
  first_part = 5 →
  ratio_percent = 25 →
  first_part / (first_part + second_part) = ratio_percent / 100 →
  second_part = 15 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3942_394246


namespace NUMINAMATH_CALUDE_solve_system_l3942_394263

theorem solve_system (x y : ℝ) (h1 : x - 2*y = 10) (h2 : x * y = 40) : y = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3942_394263


namespace NUMINAMATH_CALUDE_final_output_is_25_l3942_394220

def algorithm_output : ℕ → ℕ
| 0 => 25
| (n+1) => if 2*n + 1 < 10 then algorithm_output n else 2*(2*n + 1) + 3

theorem final_output_is_25 : algorithm_output 0 = 25 := by
  sorry

end NUMINAMATH_CALUDE_final_output_is_25_l3942_394220


namespace NUMINAMATH_CALUDE_largest_number_divisible_by_89_l3942_394290

def largest_n_digit_number (n : ℕ) : ℕ := 10^n - 1

def is_divisible_by_89 (x : ℕ) : Prop := x % 89 = 0

theorem largest_number_divisible_by_89 :
  ∃ (n : ℕ), 
    (n % 2 = 1) ∧ 
    (3 ≤ n) ∧ 
    (n ≤ 7) ∧ 
    (is_divisible_by_89 (largest_n_digit_number n)) ∧
    (∀ (m : ℕ), 
      (m % 2 = 1) → 
      (3 ≤ m) → 
      (m ≤ 7) → 
      (is_divisible_by_89 (largest_n_digit_number m)) → 
      (largest_n_digit_number m ≤ largest_n_digit_number n)) ∧
    (largest_n_digit_number n = 9999951) := by
  sorry

end NUMINAMATH_CALUDE_largest_number_divisible_by_89_l3942_394290


namespace NUMINAMATH_CALUDE_expansion_properties_l3942_394273

/-- Given an expression (3x - 1/(2*3x))^n where the ratio of the binomial coefficient 
    of the fifth term to that of the third term is 14:3, this theorem proves various 
    properties about the expansion. -/
theorem expansion_properties (n : ℕ) 
  (h : (Nat.choose n 4 : ℚ) / (Nat.choose n 2 : ℚ) = 14 / 3) :
  n = 10 ∧ 
  (let coeff_x2 := (Nat.choose 10 2 : ℚ) * (-1/2)^2 * 3^2;
   coeff_x2 = 45/4) ∧
  (let rational_terms := [
     (Nat.choose 10 2 : ℚ) * (-1/2)^2,
     (Nat.choose 10 5 : ℚ) * (-1/2)^5,
     (Nat.choose 10 8 : ℚ) * (-1/2)^8
   ];
   rational_terms.length = 3) :=
by sorry


end NUMINAMATH_CALUDE_expansion_properties_l3942_394273


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3942_394298

theorem quadratic_factorization (a : ℝ) : a^2 - a + 1/4 = (a - 1/2)^2 := by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3942_394298


namespace NUMINAMATH_CALUDE_allocation_schemes_count_l3942_394241

/-- The number of ways to allocate three distinct individuals to seven laboratories,
    where each laboratory can hold at most two people. -/
def allocationSchemes : ℕ := 336

/-- The number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of permutations of n distinct objects. -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem allocation_schemes_count :
  allocationSchemes = 
    choose 7 3 * factorial 3 + choose 7 2 * choose 3 2 * 2 :=
by sorry

end NUMINAMATH_CALUDE_allocation_schemes_count_l3942_394241


namespace NUMINAMATH_CALUDE_correlation_coefficient_relationship_l3942_394286

-- Define the data points
def x_data : List ℝ := [1, 2, 3, 4, 5]
def y_data : List ℝ := [3, 5.3, 6.9, 9.1, 10.8]
def U_data : List ℝ := [1, 2, 3, 4, 5]
def V_data : List ℝ := [12.7, 10.2, 7, 3.6, 1]

-- Define linear correlation coefficient
def linear_correlation_coefficient (x y : List ℝ) : ℝ := sorry

-- Define r₁ and r₂
def r₁ : ℝ := linear_correlation_coefficient x_data y_data
def r₂ : ℝ := linear_correlation_coefficient U_data V_data

-- Theorem to prove
theorem correlation_coefficient_relationship : r₂ < 0 ∧ 0 < r₁ := by
  sorry

end NUMINAMATH_CALUDE_correlation_coefficient_relationship_l3942_394286


namespace NUMINAMATH_CALUDE_club_member_age_difference_l3942_394245

/-- Given a club with 10 members, prove that replacing one member
    results in a 50-year difference between the old and new member's ages
    if the average age remains the same after 5 years. -/
theorem club_member_age_difference
  (n : ℕ) -- number of club members
  (a : ℝ) -- average age of members 5 years ago
  (o : ℝ) -- age of the old (replaced) member
  (n' : ℝ) -- age of the new member
  (h1 : n = 10) -- there are 10 members
  (h2 : n * a = (n - 1) * (a + 5) + n') -- average age remains the same after 5 years and replacement
  : |o - n'| = 50 := by
  sorry


end NUMINAMATH_CALUDE_club_member_age_difference_l3942_394245


namespace NUMINAMATH_CALUDE_unique_pair_existence_l3942_394242

theorem unique_pair_existence : ∃! (c d : Real),
  c ∈ Set.Ioo 0 (Real.pi / 2) ∧
  d ∈ Set.Ioo 0 (Real.pi / 2) ∧
  c < d ∧
  Real.sin (Real.cos c) = c ∧
  Real.cos (Real.sin d) = d := by
  sorry

end NUMINAMATH_CALUDE_unique_pair_existence_l3942_394242


namespace NUMINAMATH_CALUDE_inequality_solutions_l3942_394208

-- Define the solution sets
def solution_set1 : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 1}
def solution_set2 : Set ℝ := {x : ℝ | x < -2 ∨ x > 3}

-- State the theorem
theorem inequality_solutions :
  (∀ x : ℝ, x^2 + 2*x - 3 ≤ 0 ↔ x ∈ solution_set1) ∧
  (∀ x : ℝ, x - x^2 + 6 < 0 ↔ x ∈ solution_set2) := by
sorry

end NUMINAMATH_CALUDE_inequality_solutions_l3942_394208


namespace NUMINAMATH_CALUDE_robot_cost_calculation_l3942_394270

def number_of_robots : ℕ := 7
def total_tax : ℚ := 7.22
def remaining_change : ℚ := 11.53
def initial_amount : ℚ := 80

theorem robot_cost_calculation (number_of_robots : ℕ) (total_tax : ℚ) (remaining_change : ℚ) (initial_amount : ℚ) :
  let total_spent : ℚ := initial_amount - remaining_change
  let robots_cost : ℚ := total_spent - total_tax
  let cost_per_robot : ℚ := robots_cost / number_of_robots
  cost_per_robot = 8.75 :=
by sorry

end NUMINAMATH_CALUDE_robot_cost_calculation_l3942_394270


namespace NUMINAMATH_CALUDE_mike_peaches_picked_l3942_394287

/-- The number of peaches Mike picked -/
def peaches_picked (initial : ℕ) (final : ℕ) : ℕ := final - initial

theorem mike_peaches_picked : 
  peaches_picked 34 86 = 52 := by sorry

end NUMINAMATH_CALUDE_mike_peaches_picked_l3942_394287


namespace NUMINAMATH_CALUDE_circle_through_points_equation_l3942_394284

/-- A circle passing through three given points -/
structure CircleThroughPoints where
  -- Define the three points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- Ensure the points are distinct
  distinct_points : A ≠ B ∧ B ≠ C ∧ A ≠ C

/-- The equation of a circle in standard form -/
def circle_equation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Main theorem: The circle through the given points has the specified equation -/
theorem circle_through_points_equation (c : CircleThroughPoints) :
  c.A = (-1, -1) →
  c.B = (-8, 0) →
  c.C = (0, 6) →
  ∃ (h k r : ℝ), 
    (h = -4 ∧ k = 3 ∧ r = 5) ∧
    (∀ x y, circle_equation h k r x y ↔ 
      ((x, y) = c.A ∨ (x, y) = c.B ∨ (x, y) = c.C)) :=
by sorry

#check circle_through_points_equation

end NUMINAMATH_CALUDE_circle_through_points_equation_l3942_394284


namespace NUMINAMATH_CALUDE_erica_ride_time_l3942_394201

/-- The time Dave can ride the merry-go-round in minutes -/
def dave_time : ℝ := 10

/-- The factor by which Chuck can ride longer than Dave -/
def chuck_factor : ℝ := 5

/-- The percentage longer Erica can stay compared to Chuck -/
def erica_percentage : ℝ := 0.3

/-- The time Chuck can ride the merry-go-round in minutes -/
def chuck_time : ℝ := dave_time * chuck_factor

/-- The time Erica can ride the merry-go-round in minutes -/
def erica_time : ℝ := chuck_time * (1 + erica_percentage)

theorem erica_ride_time : erica_time = 65 := by
  sorry

end NUMINAMATH_CALUDE_erica_ride_time_l3942_394201


namespace NUMINAMATH_CALUDE_reciprocal_problem_l3942_394259

theorem reciprocal_problem (x : ℝ) (h : 8 * x = 3) : 200 / x = 1600 / 3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l3942_394259


namespace NUMINAMATH_CALUDE_complex_number_location_l3942_394280

theorem complex_number_location : 
  let i : ℂ := Complex.I
  (1 + i) / (1 - i) + i ^ 2012 = 1 + i := by sorry

end NUMINAMATH_CALUDE_complex_number_location_l3942_394280


namespace NUMINAMATH_CALUDE_min_sum_squares_l3942_394262

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), m = 4 ∧ (∀ a b c : ℝ, a^3 + b^3 + c^3 - 3*a*b*c = 8 → a^2 + b^2 + c^2 ≥ m) ∧
  (∃ p q r : ℝ, p^3 + q^3 + r^3 - 3*p*q*r = 8 ∧ p^2 + q^2 + r^2 = m) :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3942_394262


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3942_394205

theorem max_value_of_expression (a b c d : ℝ) 
  (nonneg_a : 0 ≤ a) (nonneg_b : 0 ≤ b) (nonneg_c : 0 ≤ c) (nonneg_d : 0 ≤ d)
  (sum_condition : a + b + c + d = 200)
  (a_condition : a = 2 * d) :
  a * b + b * c + c * d ≤ 42500 / 3 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3942_394205


namespace NUMINAMATH_CALUDE_largest_number_with_quotient_30_l3942_394212

theorem largest_number_with_quotient_30 : 
  ∀ n : ℕ, n ≤ 960 ∧ n / 31 = 30 → n = 960 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_number_with_quotient_30_l3942_394212


namespace NUMINAMATH_CALUDE_updated_mean_after_decrement_l3942_394278

theorem updated_mean_after_decrement (n : ℕ) (original_mean : ℝ) (decrement : ℝ) :
  n > 0 →
  n = 50 →
  original_mean = 200 →
  decrement = 6 →
  (n * original_mean - n * decrement) / n = 194 := by
  sorry

end NUMINAMATH_CALUDE_updated_mean_after_decrement_l3942_394278


namespace NUMINAMATH_CALUDE_baker_cakes_problem_l3942_394207

theorem baker_cakes_problem (sold : ℕ) (left : ℕ) (h1 : sold = 41) (h2 : left = 13) :
  sold + left = 54 :=
by sorry

end NUMINAMATH_CALUDE_baker_cakes_problem_l3942_394207


namespace NUMINAMATH_CALUDE_evaluate_expression_l3942_394291

theorem evaluate_expression :
  (3 * Real.sqrt 8) / (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7) =
  -(6 * (Real.sqrt 2 - Real.sqrt 6 - Real.sqrt 10 - Real.sqrt 14)) / 13 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3942_394291


namespace NUMINAMATH_CALUDE_tank_fill_problem_l3942_394260

theorem tank_fill_problem (tank_capacity : ℚ) (added_amount : ℚ) (final_fraction : ℚ) :
  tank_capacity = 48 →
  added_amount = 8 →
  final_fraction = 9/10 →
  (tank_capacity * final_fraction - added_amount) / tank_capacity = 4/10 :=
by sorry

end NUMINAMATH_CALUDE_tank_fill_problem_l3942_394260


namespace NUMINAMATH_CALUDE_billy_younger_than_gladys_l3942_394238

def billy_age : ℕ := sorry
def lucas_age : ℕ := sorry
def gladys_age : ℕ := 30

axiom lucas_future_age : lucas_age + 3 = 8
axiom gladys_age_relation : gladys_age = 2 * (billy_age + lucas_age)

theorem billy_younger_than_gladys : gladys_age / billy_age = 3 := by sorry

end NUMINAMATH_CALUDE_billy_younger_than_gladys_l3942_394238


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3942_394218

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 / x + 9 / y) ≥ 16 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ 1 / x + 9 / y = 16 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3942_394218


namespace NUMINAMATH_CALUDE_cube_root_125_times_fourth_root_256_times_sixth_root_64_l3942_394251

theorem cube_root_125_times_fourth_root_256_times_sixth_root_64 :
  (125 : ℝ) ^ (1/3) * (256 : ℝ) ^ (1/4) * (64 : ℝ) ^ (1/6) = 40 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_125_times_fourth_root_256_times_sixth_root_64_l3942_394251


namespace NUMINAMATH_CALUDE_direction_vector_b_l3942_394295

/-- Given a line passing through points (-4, 6) and (3, -3), 
    prove that the direction vector of the form (b, 1) has b = -7/9 -/
theorem direction_vector_b (p1 p2 : ℝ × ℝ) (b : ℝ) : 
  p1 = (-4, 6) → p2 = (3, -3) → 
  ∃ k : ℝ, k • (p2.1 - p1.1, p2.2 - p1.2) = (b, 1) → 
  b = -7/9 := by
sorry

end NUMINAMATH_CALUDE_direction_vector_b_l3942_394295


namespace NUMINAMATH_CALUDE_additional_time_is_24_minutes_l3942_394234

/-- Time to fill one barrel normally (in minutes) -/
def normal_time : ℕ := 3

/-- Time to fill one barrel with leak (in minutes) -/
def leak_time : ℕ := 5

/-- Number of barrels to fill -/
def num_barrels : ℕ := 12

/-- Additional time required to fill barrels with leak -/
def additional_time : ℕ := (leak_time * num_barrels) - (normal_time * num_barrels)

theorem additional_time_is_24_minutes : additional_time = 24 := by
  sorry

end NUMINAMATH_CALUDE_additional_time_is_24_minutes_l3942_394234


namespace NUMINAMATH_CALUDE_fence_price_per_foot_l3942_394239

theorem fence_price_per_foot 
  (area : ℝ) 
  (total_cost : ℝ) 
  (h1 : area = 289) 
  (h2 : total_cost = 3672) : 
  total_cost / (4 * Real.sqrt area) = 54 := by
  sorry

end NUMINAMATH_CALUDE_fence_price_per_foot_l3942_394239


namespace NUMINAMATH_CALUDE_visitors_equal_cats_l3942_394277

/-- In a cat show scenario -/
structure CatShow where
  /-- The set of visitors -/
  visitors : Type
  /-- The set of cats -/
  cats : Type
  /-- The relation representing a visitor petting a cat -/
  pets : visitors → cats → Prop
  /-- Each visitor pets exactly three cats -/
  visitor_pets_three : ∀ v : visitors, ∃! (c₁ c₂ c₃ : cats), pets v c₁ ∧ pets v c₂ ∧ pets v c₃ ∧ c₁ ≠ c₂ ∧ c₁ ≠ c₃ ∧ c₂ ≠ c₃
  /-- Each cat is petted by exactly three visitors -/
  cat_petted_by_three : ∀ c : cats, ∃! (v₁ v₂ v₃ : visitors), pets v₁ c ∧ pets v₂ c ∧ pets v₃ c ∧ v₁ ≠ v₂ ∧ v₁ ≠ v₃ ∧ v₂ ≠ v₃

/-- The number of visitors is equal to the number of cats -/
theorem visitors_equal_cats (cs : CatShow) : Nonempty (Equiv cs.visitors cs.cats) := by
  sorry

end NUMINAMATH_CALUDE_visitors_equal_cats_l3942_394277


namespace NUMINAMATH_CALUDE_first_year_more_rabbits_l3942_394232

def squirrels (k : ℕ) : ℕ := 2020 * 2^k - 2019

def rabbits (k : ℕ) : ℕ := (4^k + 2) / 3

def more_rabbits_than_squirrels (k : ℕ) : Prop :=
  rabbits k > squirrels k

theorem first_year_more_rabbits : 
  (∀ n < 13, ¬(more_rabbits_than_squirrels n)) ∧ 
  more_rabbits_than_squirrels 13 := by
  sorry

#check first_year_more_rabbits

end NUMINAMATH_CALUDE_first_year_more_rabbits_l3942_394232


namespace NUMINAMATH_CALUDE_book_sale_profit_l3942_394257

/-- Prove that given two books with a total cost of 480, where the first book costs 280 and is sold
    at a 15% loss, and both books are sold at the same price, the gain percentage on the second book
    is 19%. -/
theorem book_sale_profit (total_cost : ℝ) (cost_book1 : ℝ) (loss_percentage : ℝ) 
  (h1 : total_cost = 480)
  (h2 : cost_book1 = 280)
  (h3 : loss_percentage = 15)
  (h4 : ∃ (sell_price : ℝ), sell_price = cost_book1 * (1 - loss_percentage / 100) ∧ 
        sell_price = (total_cost - cost_book1) * (1 + x / 100)) :
  x = 19 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_profit_l3942_394257


namespace NUMINAMATH_CALUDE_max_plus_min_equals_16_l3942_394202

def f (x : ℝ) : ℝ := x^3 - 12*x + 8

theorem max_plus_min_equals_16 :
  ∃ (M m : ℝ),
    (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≤ M) ∧
    (∃ x ∈ Set.Icc (-3 : ℝ) 3, f x = M) ∧
    (∀ x ∈ Set.Icc (-3 : ℝ) 3, m ≤ f x) ∧
    (∃ x ∈ Set.Icc (-3 : ℝ) 3, f x = m) ∧
    M + m = 16 :=
by sorry

end NUMINAMATH_CALUDE_max_plus_min_equals_16_l3942_394202


namespace NUMINAMATH_CALUDE_quadratic_to_vertex_form_l3942_394215

theorem quadratic_to_vertex_form (x : ℝ) : 
  x^2 - 4*x + 5 = (x - 2)^2 + 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_to_vertex_form_l3942_394215


namespace NUMINAMATH_CALUDE_fraction_power_and_multiply_l3942_394265

theorem fraction_power_and_multiply :
  (2 / 3 : ℚ) ^ 3 * (1 / 4 : ℚ) = 2 / 27 := by sorry

end NUMINAMATH_CALUDE_fraction_power_and_multiply_l3942_394265


namespace NUMINAMATH_CALUDE_oliver_video_games_l3942_394224

/-- The number of working video games Oliver bought -/
def working_games : ℕ := 6

/-- The number of bad video games Oliver bought -/
def bad_games : ℕ := 5

/-- The total number of video games Oliver bought -/
def total_games : ℕ := working_games + bad_games

theorem oliver_video_games : 
  total_games = working_games + bad_games := by sorry

end NUMINAMATH_CALUDE_oliver_video_games_l3942_394224


namespace NUMINAMATH_CALUDE_william_max_moves_l3942_394225

/-- Represents a player in the game -/
inductive Player : Type
| Mark : Player
| William : Player

/-- Represents a move in the game -/
inductive Move : Type
| Double : Move  -- Multiply by 2 and add 1
| Quadruple : Move  -- Multiply by 4 and add 3

/-- Applies a move to the current value -/
def applyMove (value : ℕ) (move : Move) : ℕ :=
  match move with
  | Move.Double => 2 * value + 1
  | Move.Quadruple => 4 * value + 3

/-- Checks if the game is over -/
def isGameOver (value : ℕ) : Prop :=
  value > 2^100

/-- Represents the state of the game -/
structure GameState :=
  (value : ℕ)
  (currentPlayer : Player)

/-- Represents an optimal strategy for the game -/
def OptimalStrategy : Type :=
  GameState → Move

/-- The maximum number of moves William can make -/
def maxWilliamMoves : ℕ := 33

/-- The main theorem to be proved -/
theorem william_max_moves 
  (strategy : OptimalStrategy) : 
  ∃ (game : List Move), 
    game.length = 2 * maxWilliamMoves + 1 ∧ 
    isGameOver (game.foldl applyMove 1) ∧
    ∀ (game' : List Move), 
      game'.length > 2 * maxWilliamMoves + 1 → 
      ¬isGameOver (game'.foldl applyMove 1) :=
sorry

end NUMINAMATH_CALUDE_william_max_moves_l3942_394225


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l3942_394223

theorem triangle_angle_proof (a b c : ℝ) (A : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Ensure positive side lengths
  0 < A ∧ A < π →  -- Angle A is between 0 and π
  a^2 = b^2 + Real.sqrt 3 * b * c + c^2 →  -- Given condition
  A = 5 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l3942_394223


namespace NUMINAMATH_CALUDE_smallest_z_value_l3942_394296

/-- Given an equation of consecutive perfect cubes, find the smallest possible value of the largest cube. -/
theorem smallest_z_value (u w x y z : ℕ) : 
  u^3 + w^3 + x^3 + y^3 = z^3 ∧ 
  (∃ k : ℕ, u = k ∧ w = k + 1 ∧ x = k + 2 ∧ y = k + 3 ∧ z = k + 4) ∧
  0 < u ∧ u < w ∧ w < x ∧ x < y ∧ y < z →
  z = 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_z_value_l3942_394296


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l3942_394255

/-- The radius of the inscribed circle of a triangle with sides 15, 16, and 17 is √21 -/
theorem inscribed_circle_radius (a b c : ℝ) (ha : a = 15) (hb : b = 16) (hc : c = 17) :
  let s := (a + b + c) / 2
  let r := Real.sqrt (s * (s - a) * (s - b) * (s - c)) / s
  r = Real.sqrt 21 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l3942_394255
