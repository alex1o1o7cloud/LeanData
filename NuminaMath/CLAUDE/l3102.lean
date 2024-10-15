import Mathlib

namespace NUMINAMATH_CALUDE_jimmy_speed_l3102_310280

theorem jimmy_speed (mary_speed : ℝ) (total_distance : ℝ) (time : ℝ) (jimmy_speed : ℝ) : 
  mary_speed = 5 →
  total_distance = 9 →
  time = 1 →
  jimmy_speed = total_distance - mary_speed * time →
  jimmy_speed = 4 :=
by sorry

end NUMINAMATH_CALUDE_jimmy_speed_l3102_310280


namespace NUMINAMATH_CALUDE_bobbys_candy_consumption_l3102_310219

/-- Represents the number of candies Bobby takes during each of the remaining days of the week. -/
def candies_on_remaining_days (
  packets : ℕ)  -- Number of candy packets
  (candies_per_packet : ℕ)  -- Number of candies in each packet
  (candies_per_weekday : ℕ)  -- Number of candies eaten per weekday
  (weekdays : ℕ)  -- Number of weekdays per week
  (weeks : ℕ)  -- Number of weeks to finish all candies
  : ℕ :=
  let total_candies := packets * candies_per_packet
  let weekday_candies := candies_per_weekday * weekdays * weeks
  let remaining_candies := total_candies - weekday_candies
  let remaining_days := (7 - weekdays) * weeks
  remaining_candies / remaining_days

/-- Theorem stating that Bobby takes 1 candy during each of the remaining days of the week. -/
theorem bobbys_candy_consumption :
  candies_on_remaining_days 2 18 2 5 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_bobbys_candy_consumption_l3102_310219


namespace NUMINAMATH_CALUDE_total_questions_l3102_310209

/-- Represents the examination structure -/
structure Examination where
  typeA : Nat
  typeB : Nat
  totalTime : Nat
  typeATime : Nat

/-- The given examination parameters -/
def givenExam : Examination where
  typeA := 25
  typeB := 0  -- We don't know this value yet
  totalTime := 180  -- 3 hours * 60 minutes
  typeATime := 40

/-- Theorem stating the total number of questions in the examination -/
theorem total_questions (e : Examination) (h1 : e.typeA = givenExam.typeA)
    (h2 : e.totalTime = givenExam.totalTime) (h3 : e.typeATime = givenExam.typeATime)
    (h4 : 2 * (e.totalTime - e.typeATime) = 7 * e.typeATime) :
    e.typeA + e.typeB = 200 := by
  sorry

#check total_questions

end NUMINAMATH_CALUDE_total_questions_l3102_310209


namespace NUMINAMATH_CALUDE_amusement_park_tickets_l3102_310282

theorem amusement_park_tickets (adult_price child_price total_paid total_tickets : ℕ)
  (h1 : adult_price = 8)
  (h2 : child_price = 5)
  (h3 : total_paid = 201)
  (h4 : total_tickets = 33)
  : ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_tickets * adult_price + child_tickets * child_price = total_paid ∧
    child_tickets = 21 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_tickets_l3102_310282


namespace NUMINAMATH_CALUDE_min_value_sin_cos_min_value_achievable_l3102_310238

theorem min_value_sin_cos (x : ℝ) : 
  Real.sin x ^ 6 + 2 * Real.cos x ^ 6 ≥ 2/3 :=
sorry

theorem min_value_achievable : 
  ∃ x : ℝ, Real.sin x ^ 6 + 2 * Real.cos x ^ 6 = 2/3 :=
sorry

end NUMINAMATH_CALUDE_min_value_sin_cos_min_value_achievable_l3102_310238


namespace NUMINAMATH_CALUDE_alexei_weekly_loss_l3102_310213

/-- Given the weight loss information for Aleesia and Alexei, calculate Alexei's weekly weight loss. -/
theorem alexei_weekly_loss 
  (aleesia_weekly_loss : ℝ) 
  (aleesia_weeks : ℕ) 
  (alexei_weeks : ℕ) 
  (total_loss : ℝ) 
  (h1 : aleesia_weekly_loss = 1.5)
  (h2 : aleesia_weeks = 10)
  (h3 : alexei_weeks = 8)
  (h4 : total_loss = 35)
  : (total_loss - aleesia_weekly_loss * aleesia_weeks) / alexei_weeks = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_alexei_weekly_loss_l3102_310213


namespace NUMINAMATH_CALUDE_same_constant_term_similar_structure_l3102_310245

-- Define a polynomial with distinct positive real coefficients
def P (x : ℝ) : ℝ := sorry

-- Define the median of the coefficients of P
def median_coeff : ℝ := sorry

-- Define Q using the median of coefficients of P
def Q (x : ℝ) : ℝ := sorry

-- Theorem stating that P and Q have the same constant term
theorem same_constant_term : P 0 = Q 0 := by sorry

-- Theorem stating that P and Q have similar structure
-- (We can't precisely define "similar structure" without more information,
-- so we'll use a placeholder property)
theorem similar_structure : ∃ (k : ℝ), k > 0 ∧ ∀ x, |P x - Q x| ≤ k := by sorry

end NUMINAMATH_CALUDE_same_constant_term_similar_structure_l3102_310245


namespace NUMINAMATH_CALUDE_teacup_rows_per_box_l3102_310248

def total_boxes : ℕ := 26
def boxes_with_pans : ℕ := 6
def cups_per_row : ℕ := 4
def cups_broken_per_box : ℕ := 2
def teacups_left : ℕ := 180

def boxes_with_teacups : ℕ := (total_boxes - boxes_with_pans) / 2

theorem teacup_rows_per_box :
  let total_teacups := teacups_left + cups_broken_per_box * boxes_with_teacups
  let teacups_per_box := total_teacups / boxes_with_teacups
  teacups_per_box / cups_per_row = 5 := by
  sorry

end NUMINAMATH_CALUDE_teacup_rows_per_box_l3102_310248


namespace NUMINAMATH_CALUDE_log_xy_value_l3102_310214

theorem log_xy_value (x y : ℝ) (h1 : Real.log (x * y^2) = 2) (h2 : Real.log (x^3 * y) = 3) :
  Real.log (x * y) = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_log_xy_value_l3102_310214


namespace NUMINAMATH_CALUDE_largest_value_problem_l3102_310261

theorem largest_value_problem : 
  let a := 12345 + 1/5678
  let b := 12345 - 1/5678
  let c := 12345 * 1/5678
  let d := 12345 / (1/5678)
  let e := 12345.5678
  d > a ∧ d > b ∧ d > c ∧ d > e := by
  sorry

end NUMINAMATH_CALUDE_largest_value_problem_l3102_310261


namespace NUMINAMATH_CALUDE_smallest_w_l3102_310236

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w : 
  ∀ w : ℕ, 
    w > 0 →
    is_factor (2^6) (1152 * w) →
    is_factor (3^4) (1152 * w) →
    is_factor (5^3) (1152 * w) →
    is_factor (7^2) (1152 * w) →
    is_factor 11 (1152 * w) →
    w ≥ 16275 :=
by sorry

end NUMINAMATH_CALUDE_smallest_w_l3102_310236


namespace NUMINAMATH_CALUDE_problem_1_l3102_310259

theorem problem_1 : 
  (-1.75) - 6.3333333333 - 2.25 + (10/3) = -7 := by sorry

end NUMINAMATH_CALUDE_problem_1_l3102_310259


namespace NUMINAMATH_CALUDE_triangle_properties_l3102_310237

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The cosine rule for triangle ABC -/
axiom cosine_rule (t : Triangle) : t.a^2 + t.b^2 - t.c^2 = 2 * t.a * t.b * Real.cos t.C

/-- The area formula for triangle ABC -/
axiom area_formula (t : Triangle) (S : ℝ) : S = 1/2 * t.a * t.b * Real.sin t.C

theorem triangle_properties (t : Triangle) (S : ℝ) :
  (t.a^2 + t.b^2 - t.c^2 = t.a * t.b → t.C = π/3) ∧
  (t.a^2 + t.b^2 - t.c^2 = t.a * t.b ∧ t.c = Real.sqrt 7 ∧ S = 3 * Real.sqrt 3 / 2 → t.a + t.b = 5) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3102_310237


namespace NUMINAMATH_CALUDE_union_equals_B_implies_m_leq_neg_three_l3102_310208

def A : Set ℝ := {x | x^2 - 3*x - 10 < 0}
def B (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 1 - 3*m}

theorem union_equals_B_implies_m_leq_neg_three (m : ℝ) : A ∪ B m = B m → m ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_union_equals_B_implies_m_leq_neg_three_l3102_310208


namespace NUMINAMATH_CALUDE_fraction_problem_l3102_310284

theorem fraction_problem (x : ℚ) : 
  x < 0.4 ∧ x * 180 = 48 → x = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3102_310284


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l3102_310250

theorem no_solution_for_equation : 
  ¬ ∃ (x : ℝ), x ≠ 3 ∧ (2 - x) / (x - 3) = 1 / (3 - x) - 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l3102_310250


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l3102_310291

theorem isosceles_right_triangle_area (DE DF : ℝ) (angle_EDF : ℝ) :
  DE = 5 →
  DF = 5 →
  angle_EDF = Real.pi / 2 →
  (1 / 2) * DE * DF = 25 / 2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l3102_310291


namespace NUMINAMATH_CALUDE_marble_arrangement_remainder_l3102_310218

def green_marbles : ℕ := 7

-- m is the maximum number of red marbles
def red_marbles (m : ℕ) : Prop := 
  m = 19 ∧ ∀ k, k > m → ¬∃ (arr : List (Fin 2)), 
    arr.length = green_marbles + k ∧ 
    (arr.countP (λ i => arr[i]? = arr[i+1]?)) = 
    (arr.countP (λ i => arr[i]? ≠ arr[i+1]?))

-- N is the number of valid arrangements
def valid_arrangements (m : ℕ) : ℕ := Nat.choose (m + green_marbles) green_marbles

theorem marble_arrangement_remainder (m : ℕ) : 
  red_marbles m → valid_arrangements m % 1000 = 388 := by sorry

end NUMINAMATH_CALUDE_marble_arrangement_remainder_l3102_310218


namespace NUMINAMATH_CALUDE_total_distance_traveled_l3102_310295

/-- Given the conditions of the problem, prove that the total distance traveled is 5 miles. -/
theorem total_distance_traveled (total_time : ℝ) (walking_time : ℝ) (walking_rate : ℝ) 
  (break_time : ℝ) (running_time : ℝ) (running_rate : ℝ) :
  total_time = 75 / 60 → 
  walking_time = 1 →
  walking_rate = 3 →
  break_time = 5 / 60 →
  running_time = 1 / 6 →
  running_rate = 12 →
  walking_time * walking_rate + running_time * running_rate = 5 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_traveled_l3102_310295


namespace NUMINAMATH_CALUDE_no_solution_system_l3102_310220

theorem no_solution_system :
  ¬∃ (x y : ℝ), (3 * x - 4 * y = 12) ∧ (9 * x - 12 * y = 15) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_system_l3102_310220


namespace NUMINAMATH_CALUDE_previous_year_300th_day_is_monday_l3102_310269

/-- Represents days of the week -/
inductive DayOfWeek
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday
  | sunday

/-- Represents a year -/
structure Year where
  isLeapYear : Bool

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.monday => DayOfWeek.tuesday
  | DayOfWeek.tuesday => DayOfWeek.wednesday
  | DayOfWeek.wednesday => DayOfWeek.thursday
  | DayOfWeek.thursday => DayOfWeek.friday
  | DayOfWeek.friday => DayOfWeek.saturday
  | DayOfWeek.saturday => DayOfWeek.sunday
  | DayOfWeek.sunday => DayOfWeek.monday

/-- Calculates the day of the week after a given number of days -/
def advanceDays (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match days with
  | 0 => start
  | n + 1 => advanceDays (nextDay start) n

/-- Main theorem -/
theorem previous_year_300th_day_is_monday 
  (currentYear : Year)
  (nextYear : Year)
  (h1 : advanceDays DayOfWeek.sunday 200 = DayOfWeek.sunday)
  (h2 : advanceDays DayOfWeek.sunday 100 = DayOfWeek.sunday) :
  advanceDays DayOfWeek.monday 300 = DayOfWeek.sunday :=
sorry

end NUMINAMATH_CALUDE_previous_year_300th_day_is_monday_l3102_310269


namespace NUMINAMATH_CALUDE_rectangle_area_l3102_310293

theorem rectangle_area (ratio_long : ℕ) (ratio_short : ℕ) (perimeter : ℕ) :
  ratio_long = 4 →
  ratio_short = 3 →
  perimeter = 126 →
  ∃ (length width : ℕ),
    length * ratio_short = width * ratio_long ∧
    2 * (length + width) = perimeter ∧
    length * width = 972 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3102_310293


namespace NUMINAMATH_CALUDE_students_per_class_l3102_310286

theorem students_per_class 
  (cards_per_student : ℕ) 
  (periods_per_day : ℕ) 
  (cards_per_pack : ℕ) 
  (cost_per_pack : ℕ) 
  (total_spent : ℕ) 
  (h1 : cards_per_student = 10)
  (h2 : periods_per_day = 6)
  (h3 : cards_per_pack = 50)
  (h4 : cost_per_pack = 3)
  (h5 : total_spent = 108) :
  total_spent / cost_per_pack * cards_per_pack / cards_per_student / periods_per_day = 30 := by
sorry

end NUMINAMATH_CALUDE_students_per_class_l3102_310286


namespace NUMINAMATH_CALUDE_function_inequality_l3102_310211

open Real

/-- Given two differentiable functions f and g on ℝ, if f'(x) > g'(x) for all x,
    then for a < x < b, we have f(x) + g(b) < g(x) + f(b) and f(x) + g(a) > g(x) + f(a) -/
theorem function_inequality (f g : ℝ → ℝ) (hf : Differentiable ℝ f) (hg : Differentiable ℝ g)
    (h_deriv : ∀ x, deriv f x > deriv g x) (a b x : ℝ) (h_x : a < x ∧ x < b) :
    (f x + g b < g x + f b) ∧ (f x + g a > g x + f a) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3102_310211


namespace NUMINAMATH_CALUDE_base_conversion_problem_l3102_310278

theorem base_conversion_problem :
  ∀ (a b : ℕ),
  (a < 10 ∧ b < 10) →
  (5 * 7^2 + 2 * 7 + 5 = 3 * 10 * a + b) →
  (a * b) / 15 = 8 / 5 := by
sorry

end NUMINAMATH_CALUDE_base_conversion_problem_l3102_310278


namespace NUMINAMATH_CALUDE_xyz_value_l3102_310201

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19) :
  x * y * z = 10 := by sorry

end NUMINAMATH_CALUDE_xyz_value_l3102_310201


namespace NUMINAMATH_CALUDE_final_value_after_four_iterations_l3102_310217

def iterate_operation (x : ℕ) (s : ℕ) : ℕ := s * x + 1

def final_value (x : ℕ) (initial_s : ℕ) (iterations : ℕ) : ℕ :=
  match iterations with
  | 0 => initial_s
  | n + 1 => iterate_operation x (final_value x initial_s n)

theorem final_value_after_four_iterations :
  final_value 2 0 4 = 15 := by sorry

end NUMINAMATH_CALUDE_final_value_after_four_iterations_l3102_310217


namespace NUMINAMATH_CALUDE_linear_equation_power_l3102_310225

theorem linear_equation_power (n m : ℕ) :
  (∃ a b c : ℝ, ∀ x y : ℝ, a * x + b * y = c ↔ 2 * x^(n - 3) - (1/3) * y^(2*m + 1) = 0) →
  n^m = 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_power_l3102_310225


namespace NUMINAMATH_CALUDE_joan_video_game_spending_l3102_310206

/-- The cost of the basketball game Joan purchased -/
def basketball_cost : ℚ := 5.20

/-- The cost of the racing game Joan purchased -/
def racing_cost : ℚ := 4.23

/-- The total amount Joan spent on video games -/
def total_spent : ℚ := basketball_cost + racing_cost

/-- Theorem stating that the total amount Joan spent on video games is $9.43 -/
theorem joan_video_game_spending :
  total_spent = 9.43 := by sorry

end NUMINAMATH_CALUDE_joan_video_game_spending_l3102_310206


namespace NUMINAMATH_CALUDE_apps_added_minus_deleted_l3102_310240

theorem apps_added_minus_deleted (initial_apps added_apps final_apps : ℕ) :
  initial_apps = 115 →
  added_apps = 235 →
  final_apps = 178 →
  added_apps - (initial_apps + added_apps - final_apps) = 63 :=
by
  sorry

end NUMINAMATH_CALUDE_apps_added_minus_deleted_l3102_310240


namespace NUMINAMATH_CALUDE_system_solution_l3102_310216

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 5 * x^2 - 14 * x * y + 10 * y^2 = 17
def equation2 (x y : ℝ) : Prop := 4 * x^2 - 10 * x * y + 6 * y^2 = 8

-- Define the solution set
def solutions : List (ℝ × ℝ) := [(-1, -2), (11, 7), (-11, -7), (1, 2)]

-- Theorem statement
theorem system_solution :
  ∀ (p : ℝ × ℝ), p ∈ solutions → equation1 p.1 p.2 ∧ equation2 p.1 p.2 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3102_310216


namespace NUMINAMATH_CALUDE_no_solution_for_qt_plus_q_plus_t_eq_6_l3102_310230

theorem no_solution_for_qt_plus_q_plus_t_eq_6 :
  ∀ (q t : ℕ), q > 0 ∧ t > 0 → q * t + q + t ≠ 6 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_qt_plus_q_plus_t_eq_6_l3102_310230


namespace NUMINAMATH_CALUDE_arithmetic_proof_l3102_310289

theorem arithmetic_proof : (3 + 2) - (2 + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_proof_l3102_310289


namespace NUMINAMATH_CALUDE_inequality_proof_l3102_310267

theorem inequality_proof (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum_squares : a^2 + b^2 + c^2 = 3) : 
  1/(4-a^2) + 1/(4-b^2) + 1/(4-c^2) ≤ 9/((a+b+c)^2) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3102_310267


namespace NUMINAMATH_CALUDE_internally_tangent_circles_distance_l3102_310270

theorem internally_tangent_circles_distance (r₁ r₂ : ℝ) (h₁ : r₁ = 12) (h₂ : r₂ = 4) :
  let d := (r₁ - r₂)^2 + r₂^2
  d = (4 * Real.sqrt 10)^2 :=
by sorry

end NUMINAMATH_CALUDE_internally_tangent_circles_distance_l3102_310270


namespace NUMINAMATH_CALUDE_money_sum_l3102_310243

/-- Given two people a and b with some amount of money, 
    if 2/3 of a's amount equals 1/2 of b's amount, 
    and b has 484 rupees, then their total amount is 847 rupees. -/
theorem money_sum (a b : ℕ) (h1 : 2 * a = 3 * (b / 2)) (h2 : b = 484) : 
  a + b = 847 := by
  sorry

end NUMINAMATH_CALUDE_money_sum_l3102_310243


namespace NUMINAMATH_CALUDE_park_fencing_cost_l3102_310292

theorem park_fencing_cost (length width area perimeter total_cost : ℝ) : 
  length / width = 3 / 2 →
  length * width = 3750 →
  perimeter = 2 * (length + width) →
  total_cost = 175 →
  (total_cost / perimeter) * 100 = 70 :=
by sorry

end NUMINAMATH_CALUDE_park_fencing_cost_l3102_310292


namespace NUMINAMATH_CALUDE_part1_part2_l3102_310272

-- Define the quadratic function f(x)
def f (q : ℝ) (x : ℝ) : ℝ := x^2 - 16*x + q + 3

-- Part 1: Prove that if f has a root in [-1, 1], then q ∈ [-20, 12]
theorem part1 (q : ℝ) : 
  (∃ x ∈ Set.Icc (-1) 1, f q x = 0) → q ∈ Set.Icc (-20) 12 :=
by sorry

-- Part 2: Prove that if f(x) + 51 ≥ 0 for all x ∈ [q, 10], then q ∈ [9, 10)
theorem part2 (q : ℝ) :
  (∀ x ∈ Set.Icc q 10, f q x + 51 ≥ 0) → q ∈ Set.Ici 9 ∩ Set.Iio 10 :=
by sorry

end NUMINAMATH_CALUDE_part1_part2_l3102_310272


namespace NUMINAMATH_CALUDE_max_b_value_l3102_310281

-- Define a lattice point
def is_lattice_point (x y : ℤ) : Prop := true

-- Define the line equation
def line_equation (m : ℚ) (x : ℤ) : ℚ := m * x + 4

-- Define the condition for the line not passing through lattice points
def no_lattice_intersection (m : ℚ) : Prop :=
  ∀ x y : ℤ, 0 < x ∧ x ≤ 150 → is_lattice_point x y →
    line_equation m x ≠ y

-- State the theorem
theorem max_b_value :
  ∃ b : ℚ, b = 50/147 ∧
  (∀ m : ℚ, 1/3 < m ∧ m < b → no_lattice_intersection m) ∧
  (∀ b' : ℚ, b < b' →
    ∃ m : ℚ, 1/3 < m ∧ m < b' ∧ ¬(no_lattice_intersection m)) :=
sorry

end NUMINAMATH_CALUDE_max_b_value_l3102_310281


namespace NUMINAMATH_CALUDE_equation_solutions_l3102_310242

theorem equation_solutions :
  (∀ x : ℝ, 4 * x^2 - 9 = 0 ↔ x = 3/2 ∨ x = -3/2) ∧
  (∀ x : ℝ, 64 * (x + 1)^3 = -125 ↔ x = -9/4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3102_310242


namespace NUMINAMATH_CALUDE_pure_imaginary_implies_m_eq_neg_three_l3102_310228

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPureImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- Given that i is the imaginary unit, if the complex number z=(m^2+2m-3)+(m-1)i
    is a pure imaginary number, then m = -3. -/
theorem pure_imaginary_implies_m_eq_neg_three (m : ℝ) :
  IsPureImaginary (Complex.mk (m^2 + 2*m - 3) (m - 1)) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_implies_m_eq_neg_three_l3102_310228


namespace NUMINAMATH_CALUDE_prob_five_three_l3102_310274

/-- Represents the probability of reaching (0,0) from a given point (x,y) -/
def P (x y : ℕ) : ℚ :=
  sorry

/-- The probability of reaching (0,0) from any point on the x-axis (except origin) is 0 -/
axiom P_x_axis (x : ℕ) : x > 0 → P x 0 = 0

/-- The probability of reaching (0,0) from any point on the y-axis (except origin) is 0 -/
axiom P_y_axis (y : ℕ) : y > 0 → P 0 y = 0

/-- The probability at the origin is 1 -/
axiom P_origin : P 0 0 = 1

/-- The recursive relation for the probability function -/
axiom P_recursive (x y : ℕ) : x > 0 → y > 0 → 
  P x y = (1/3 : ℚ) * (P (x-1) y + P x (y-1) + P (x-1) (y-1))

/-- The main theorem: probability of reaching (0,0) from (5,3) is 121/729 -/
theorem prob_five_three : P 5 3 = 121 / 729 :=
  sorry

end NUMINAMATH_CALUDE_prob_five_three_l3102_310274


namespace NUMINAMATH_CALUDE_box_volume_is_3888_l3102_310247

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  height : ℝ
  length : ℝ
  width : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.height * d.length * d.width

/-- Theorem: The volume of the box with given dimensions is 3888 cubic inches -/
theorem box_volume_is_3888 :
  let d : BoxDimensions := {
    height := 12,
    length := 12 * 3,
    width := 12 * 3 / 4
  }
  boxVolume d = 3888 := by
  sorry


end NUMINAMATH_CALUDE_box_volume_is_3888_l3102_310247


namespace NUMINAMATH_CALUDE_negation_equivalence_l3102_310299

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - 2*x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3102_310299


namespace NUMINAMATH_CALUDE_high_school_language_study_l3102_310255

theorem high_school_language_study (total_students : ℕ) 
  (spanish_min spanish_max french_min french_max : ℕ) :
  total_students = 2001 →
  spanish_min = 1601 →
  spanish_max = 1700 →
  french_min = 601 →
  french_max = 800 →
  let m := spanish_min + french_min - total_students
  let M := spanish_max + french_max - total_students
  M - m = 298 := by
sorry

end NUMINAMATH_CALUDE_high_school_language_study_l3102_310255


namespace NUMINAMATH_CALUDE_equation_solution_l3102_310200

theorem equation_solution : 
  ∃ x : ℚ, (x ≠ 0 ∧ x ≠ 1) ∧ ((x - 1) / x + 3 * x / (x - 1) = 4) ∧ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3102_310200


namespace NUMINAMATH_CALUDE_next_price_reduction_l3102_310298

def price_sequence (n : ℕ) : ℚ :=
  (1024 : ℚ) * (5/8 : ℚ)^n

theorem next_price_reduction : price_sequence 4 = 156.25 := by
  sorry

end NUMINAMATH_CALUDE_next_price_reduction_l3102_310298


namespace NUMINAMATH_CALUDE_product_of_integers_l3102_310297

theorem product_of_integers (p q r : ℤ) : 
  p ≠ 0 → q ≠ 0 → r ≠ 0 →
  p + q + r = 27 →
  1 / p + 1 / q + 1 / r + 300 / (p * q * r) = 1 →
  p * q * r = 984 := by
sorry

end NUMINAMATH_CALUDE_product_of_integers_l3102_310297


namespace NUMINAMATH_CALUDE_gcd_1230_990_l3102_310215

theorem gcd_1230_990 : Nat.gcd 1230 990 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1230_990_l3102_310215


namespace NUMINAMATH_CALUDE_ac_equals_twelve_l3102_310287

theorem ac_equals_twelve (a b c d : ℝ) 
  (h1 : a = 2 * b)
  (h2 : c = d * b)
  (h3 : d + d = b * c)
  (h4 : d = 3) : 
  a * c = 12 := by
sorry

end NUMINAMATH_CALUDE_ac_equals_twelve_l3102_310287


namespace NUMINAMATH_CALUDE_circle_area_when_eight_times_reciprocal_circumference_equals_diameter_l3102_310244

theorem circle_area_when_eight_times_reciprocal_circumference_equals_diameter :
  ∀ (r : ℝ), r > 0 → (8 * (1 / (2 * Real.pi * r)) = 2 * r) → (Real.pi * r^2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_area_when_eight_times_reciprocal_circumference_equals_diameter_l3102_310244


namespace NUMINAMATH_CALUDE_function_domain_implies_m_range_l3102_310262

/-- If f(x) = √(mx² - 6mx + m + 8) has a domain of ℝ, then m ∈ [0, 1] -/
theorem function_domain_implies_m_range (m : ℝ) : 
  (∀ x : ℝ, mx^2 - 6*m*x + m + 8 ≥ 0) → 0 ≤ m ∧ m ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_function_domain_implies_m_range_l3102_310262


namespace NUMINAMATH_CALUDE_sum_of_digits_253_l3102_310204

/-- Given a three-digit number with specific properties, prove that the sum of its digits is 10 -/
theorem sum_of_digits_253 (a b c : ℕ) : 
  -- The number is 253
  100 * a + 10 * b + c = 253 →
  -- The middle digit is the sum of the other two
  b = a + c →
  -- Reversing the digits increases the number by 99
  100 * c + 10 * b + a = 100 * a + 10 * b + c + 99 →
  -- The sum of the digits is 10
  a + b + c = 10 := by
sorry


end NUMINAMATH_CALUDE_sum_of_digits_253_l3102_310204


namespace NUMINAMATH_CALUDE_power_sum_difference_l3102_310260

theorem power_sum_difference : 2^1 + 1^0 - 3^2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_difference_l3102_310260


namespace NUMINAMATH_CALUDE_original_number_from_sum_l3102_310257

/-- Represents a three-digit number in base 10 -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_hundreds : hundreds < 10
  h_tens : tens < 10
  h_ones : ones < 10
  h_not_zero : hundreds ≠ 0

/-- Calculates the sum of a three-digit number and its permutations -/
def sumOfPermutations (n : ThreeDigitNumber) : Nat :=
  let a := n.hundreds
  let b := n.tens
  let c := n.ones
  222 * (a + b + c)

/-- The main theorem -/
theorem original_number_from_sum (N : Nat) (h_N : N = 3237) :
  ∃ (n : ThreeDigitNumber), sumOfPermutations n = N ∧ n.hundreds = 4 ∧ n.tens = 2 ∧ n.ones = 9 := by
  sorry

end NUMINAMATH_CALUDE_original_number_from_sum_l3102_310257


namespace NUMINAMATH_CALUDE_individual_can_cost_l3102_310207

def pack_size : ℕ := 12
def pack_cost : ℚ := 299 / 100  -- $2.99 represented as a rational number

theorem individual_can_cost :
  let cost_per_can := pack_cost / pack_size
  cost_per_can = 299 / (100 * 12) := by sorry

end NUMINAMATH_CALUDE_individual_can_cost_l3102_310207


namespace NUMINAMATH_CALUDE_seating_arrangements_with_restriction_l3102_310264

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def circularArrangements (n : ℕ) : ℕ := factorial (n - 1)

def adjacentPairArrangements (n : ℕ) : ℕ := 2 * factorial (n - 2)

theorem seating_arrangements_with_restriction (total : ℕ) (restricted_pair : ℕ) :
  total = 12 →
  restricted_pair = 2 →
  circularArrangements total - adjacentPairArrangements total = 32659200 := by
  sorry

#eval circularArrangements 12 - adjacentPairArrangements 12

end NUMINAMATH_CALUDE_seating_arrangements_with_restriction_l3102_310264


namespace NUMINAMATH_CALUDE_area_2018_correct_l3102_310205

/-- Calculates the area to be converted after a given number of years -/
def area_to_convert (initial_area : ℝ) (annual_increase : ℝ) (years : ℕ) : ℝ :=
  initial_area * (1 + annual_increase) ^ years

/-- Proves that the area to be converted in 2018 is correct -/
theorem area_2018_correct (initial_area : ℝ) (annual_increase : ℝ) :
  initial_area = 8 →
  annual_increase = 0.1 →
  area_to_convert initial_area annual_increase 5 = 8 * 1.1^5 := by
  sorry

#check area_2018_correct

end NUMINAMATH_CALUDE_area_2018_correct_l3102_310205


namespace NUMINAMATH_CALUDE_pizza_theorem_l3102_310265

def pizza_eaten (n : ℕ) : ℚ :=
  if n = 0 then 0
  else if n = 1 then 1/3
  else pizza_eaten (n-1) + (1 - pizza_eaten (n-1)) / 2

theorem pizza_theorem :
  pizza_eaten 4 = 11/12 ∧ (1 - pizza_eaten 4) = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_pizza_theorem_l3102_310265


namespace NUMINAMATH_CALUDE_ellipse_and_triangle_area_l3102_310263

-- Define the ellipse C
def ellipse_C (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the inscribed circle
def inscribed_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 2

-- Define the parabola E
def parabola_E (p : ℝ) (x y : ℝ) : Prop :=
  y^2 = 2 * p * x ∧ p > 0

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  y = x + m ∧ 0 ≤ m ∧ m ≤ 1

-- State the theorem
theorem ellipse_and_triangle_area :
  ∀ (a b p m : ℝ) (x y : ℝ),
  ellipse_C a b x y →
  inscribed_circle x y →
  parabola_E p x y →
  line_l m x y →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    parabola_E p x₁ y₁ ∧ 
    parabola_E p x₂ y₂ ∧ 
    line_l m x₁ y₁ ∧ 
    line_l m x₂ y₂ ∧ 
    x₁ ≠ x₂) →
  (a^2 = 8 ∧ b^2 = 4) ∧
  (∃ (S : ℝ), S = (32 * Real.sqrt 6) / 9 ∧
    ∀ (S' : ℝ), S' ≤ S) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_triangle_area_l3102_310263


namespace NUMINAMATH_CALUDE_total_peaches_l3102_310253

/-- The number of baskets -/
def num_baskets : ℕ := 11

/-- The number of red peaches in each basket -/
def red_peaches_per_basket : ℕ := 10

/-- The number of green peaches in each basket -/
def green_peaches_per_basket : ℕ := 18

/-- Theorem: The total number of peaches in all baskets is 308 -/
theorem total_peaches :
  (num_baskets * (red_peaches_per_basket + green_peaches_per_basket)) = 308 := by
  sorry

end NUMINAMATH_CALUDE_total_peaches_l3102_310253


namespace NUMINAMATH_CALUDE_log_equation_solution_l3102_310224

theorem log_equation_solution (k x : ℝ) (h : k > 0) (h2 : x > 0) :
  (Real.log x / Real.log k) * (Real.log (k^2) / Real.log 5) = 3 →
  x = 5 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3102_310224


namespace NUMINAMATH_CALUDE_quadratic_form_h_value_l3102_310246

theorem quadratic_form_h_value (a k h : ℝ) :
  (∀ x, 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k) →
  h = -3/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_h_value_l3102_310246


namespace NUMINAMATH_CALUDE_largest_multiple_of_seven_under_hundred_l3102_310251

theorem largest_multiple_of_seven_under_hundred : 
  ∀ n : ℕ, n * 7 < 100 → n * 7 ≤ 98 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_seven_under_hundred_l3102_310251


namespace NUMINAMATH_CALUDE_jazmin_dolls_count_l3102_310254

theorem jazmin_dolls_count (geraldine_dolls : ℝ) (difference : ℕ) :
  geraldine_dolls = 2186.0 →
  difference = 977 →
  geraldine_dolls - difference = 1209 :=
by sorry

end NUMINAMATH_CALUDE_jazmin_dolls_count_l3102_310254


namespace NUMINAMATH_CALUDE_trout_catfish_ratio_is_three_to_one_l3102_310294

/-- Represents the fishing challenge scenario -/
structure FishingChallenge where
  will_catfish : ℕ
  will_eels : ℕ
  total_fish : ℕ

/-- Calculates the ratio of trout to catfish Henry challenged himself to catch -/
def trout_catfish_ratio (challenge : FishingChallenge) : ℚ :=
  let will_total := challenge.will_catfish + challenge.will_eels
  let henry_fish := challenge.total_fish - will_total
  (henry_fish : ℚ) / (challenge.will_catfish : ℚ) / 2

/-- Theorem stating the ratio of trout to catfish Henry challenged himself to catch -/
theorem trout_catfish_ratio_is_three_to_one (challenge : FishingChallenge)
  (h1 : challenge.will_catfish = 16)
  (h2 : challenge.will_eels = 10)
  (h3 : challenge.total_fish = 50) :
  trout_catfish_ratio challenge = 3 := by
  sorry

end NUMINAMATH_CALUDE_trout_catfish_ratio_is_three_to_one_l3102_310294


namespace NUMINAMATH_CALUDE_odd_divisors_of_factorial_20_l3102_310223

/-- The factorial of 20 -/
def factorial_20 : ℕ := 2432902008176640000

/-- The total number of natural divisors of 20! -/
def total_divisors : ℕ := 41040

/-- Theorem: The number of odd natural divisors of 20! is 2160 -/
theorem odd_divisors_of_factorial_20 : 
  (Finset.filter (fun d => d % 2 = 1) (Nat.divisors factorial_20)).card = 2160 := by
  sorry

end NUMINAMATH_CALUDE_odd_divisors_of_factorial_20_l3102_310223


namespace NUMINAMATH_CALUDE_sum_equals_two_n_cubed_l3102_310210

/-- The sum of numbers in the nth group of positive integers -/
def A (n : ℕ+) : ℕ := sorry

/-- The difference between the latter and former number in the nth group of cubes of natural numbers -/
def B (n : ℕ+) : ℕ := sorry

/-- The theorem stating that A_n + B_n = 2n^3 for all positive integers n -/
theorem sum_equals_two_n_cubed (n : ℕ+) : A n + B n = 2 * n.val ^ 3 := by sorry

end NUMINAMATH_CALUDE_sum_equals_two_n_cubed_l3102_310210


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_implies_function_ordering_l3102_310296

/-- The quadratic function f(x) = ax^2 + bx + c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequality_solution_implies_function_ordering
  (a b c : ℝ)
  (h : ∀ x, a * x^2 + b * x + c > 0 ↔ -2 < x ∧ x < 4) :
  f a b c 5 < f a b c (-1) ∧ f a b c (-1) < f a b c 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_implies_function_ordering_l3102_310296


namespace NUMINAMATH_CALUDE_doubled_container_volume_l3102_310234

/-- The volume of a container after doubling its dimensions -/
def doubled_volume (original_volume : ℝ) : ℝ := 8 * original_volume

/-- Theorem: Doubling the dimensions of a 4-gallon container results in a 32-gallon container -/
theorem doubled_container_volume : doubled_volume 4 = 32 := by
  sorry

end NUMINAMATH_CALUDE_doubled_container_volume_l3102_310234


namespace NUMINAMATH_CALUDE_total_pencils_count_l3102_310276

/-- The number of colors in the rainbow -/
def rainbow_colors : ℕ := 7

/-- The number of pencils in a color box -/
def pencils_per_box : ℕ := rainbow_colors

/-- The number of Emily's friends who bought a color box -/
def emilys_friends : ℕ := 7

/-- The total number of pencils Emily and her friends have -/
def total_pencils : ℕ := pencils_per_box + emilys_friends * pencils_per_box

theorem total_pencils_count : total_pencils = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_count_l3102_310276


namespace NUMINAMATH_CALUDE_ellipse_theorem_l3102_310279

/-- Ellipse with focus at (-√3, 0) and point (1, y) on it --/
structure Ellipse where
  a : ℝ
  b : ℝ
  y : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_y_pos : y > 0
  h_eq : 1 / a^2 + y^2 / b^2 = 1
  h_focus : -Real.sqrt 3 = -Real.sqrt (a^2 - b^2)
  h_area : 1/2 * Real.sqrt 3 * y = 3/4

/-- The main theorem --/
theorem ellipse_theorem (e : Ellipse) :
  (∃ (k : ℝ), (∀ (x y : ℝ),
    (x^2 / 4 + y^2 = 1 ↔ x^2 / e.a^2 + y^2 / e.b^2 = 1) ∧
    (y = k * (x - 2) →
      ∃ (x₁ y₁ x₂ y₂ : ℝ),
        x₁^2 / 4 + y₁^2 = 1 ∧
        x₂^2 / 4 + y₂^2 = 1 ∧
        y₁ = k * (x₁ - 2) ∧
        y₂ = k * (x₂ - 2) ∧
        ∃ (t₁ t₂ : ℝ),
          t₁^2 + t₂^2 = 1 ∧
          t₁ = Real.sqrt 5 / 5 * (2 + x₂) ∧
          t₂ = Real.sqrt 5 / 5 * (y₂)))) ∧
  (k = 1/2 ∨ k = -1/2) := by sorry

end NUMINAMATH_CALUDE_ellipse_theorem_l3102_310279


namespace NUMINAMATH_CALUDE_circular_garden_ratio_l3102_310258

theorem circular_garden_ratio (r : ℝ) (h : r = 6) : 
  (2 * Real.pi * r) / (Real.pi * r^2) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_circular_garden_ratio_l3102_310258


namespace NUMINAMATH_CALUDE_min_value_problem_l3102_310285

theorem min_value_problem (m n : ℝ) (hm : m ≥ 0) (hn : n ≥ 0) (hmn : m + n = 1) :
  m^2 / (m + 2) + n^2 / (n + 1) ≥ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l3102_310285


namespace NUMINAMATH_CALUDE_fraction_simplification_l3102_310241

theorem fraction_simplification : (4 * 5) / 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3102_310241


namespace NUMINAMATH_CALUDE_subset_P_l3102_310239

def P : Set ℝ := {x | x ≤ 3}

theorem subset_P : {-1} ⊆ P := by sorry

end NUMINAMATH_CALUDE_subset_P_l3102_310239


namespace NUMINAMATH_CALUDE_loan_interest_rate_calculation_l3102_310222

/-- The interest rate for the second part of a loan, given specific conditions -/
theorem loan_interest_rate_calculation (total : ℝ) (second_part : ℝ) : 
  total = 2743 →
  second_part = 1688 →
  let first_part := total - second_part
  let interest_rate_first := 0.03
  let time_first := 8
  let time_second := 3
  let interest_first := first_part * interest_rate_first * time_first
  let interest_second := second_part * time_second
  ∃ (r : ℝ), interest_first = r * interest_second ∧ 
             r ≥ 0.0499 ∧ r ≤ 0.05 := by
  sorry

#check loan_interest_rate_calculation

end NUMINAMATH_CALUDE_loan_interest_rate_calculation_l3102_310222


namespace NUMINAMATH_CALUDE_expected_replant_is_200_l3102_310235

/-- The expected number of seeds to be replanted -/
def expected_replant (p : ℝ) (n : ℕ) (r : ℕ) : ℝ :=
  n * (1 - p) * r

/-- Theorem: The expected number of seeds to be replanted is 200 -/
theorem expected_replant_is_200 :
  expected_replant 0.9 1000 2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_expected_replant_is_200_l3102_310235


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l3102_310233

theorem square_sum_reciprocal (x : ℝ) (h : x + (1/x) = 5) : x^2 + (1/x)^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l3102_310233


namespace NUMINAMATH_CALUDE_range_of_a_for_two_distinct_roots_l3102_310268

theorem range_of_a_for_two_distinct_roots : 
  ∀ a : ℝ, (∃! x y : ℝ, x ≠ y ∧ |x^2 - 5*x| = a) → (a = 0 ∨ a > 25/4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_two_distinct_roots_l3102_310268


namespace NUMINAMATH_CALUDE_aaron_erasers_l3102_310229

theorem aaron_erasers (initial : ℕ) (given_away : ℕ) (remaining : ℕ) : 
  initial = 81 → given_away = 34 → remaining = initial - given_away → remaining = 47 := by
  sorry

end NUMINAMATH_CALUDE_aaron_erasers_l3102_310229


namespace NUMINAMATH_CALUDE_mirasol_account_balance_l3102_310226

def remaining_amount (initial : ℕ) (expense1 : ℕ) (expense2 : ℕ) : ℕ :=
  initial - (expense1 + expense2)

theorem mirasol_account_balance : remaining_amount 50 10 30 = 10 := by
  sorry

end NUMINAMATH_CALUDE_mirasol_account_balance_l3102_310226


namespace NUMINAMATH_CALUDE_division_of_decimals_l3102_310212

theorem division_of_decimals : (0.25 : ℚ) / (0.005 : ℚ) = 50 := by sorry

end NUMINAMATH_CALUDE_division_of_decimals_l3102_310212


namespace NUMINAMATH_CALUDE_circle_sum_puzzle_l3102_310266

/-- A solution is a 6-tuple of natural numbers representing the values in circles A, B, C, D, E, F --/
def Solution := (Nat × Nat × Nat × Nat × Nat × Nat)

/-- Check if a solution satisfies all conditions --/
def is_valid_solution (s : Solution) : Prop :=
  let (a, b, c, d, e, f) := s
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f ∧
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧
  b + c + a = 22 ∧
  d + c + f = 11 ∧
  e + b + d = 19 ∧
  a + e + c = 22

theorem circle_sum_puzzle :
  ∃! (s1 s2 : Solution),
    is_valid_solution s1 ∧
    is_valid_solution s2 ∧
    (∀ s, is_valid_solution s → (s = s1 ∨ s = s2)) :=
sorry

end NUMINAMATH_CALUDE_circle_sum_puzzle_l3102_310266


namespace NUMINAMATH_CALUDE_melanies_dimes_l3102_310203

theorem melanies_dimes (initial_dimes : ℕ) (mother_dimes : ℕ) (total_dimes : ℕ) (dad_dimes : ℕ) :
  initial_dimes = 7 →
  mother_dimes = 4 →
  total_dimes = 19 →
  total_dimes = initial_dimes + mother_dimes + dad_dimes →
  dad_dimes = 8 := by
sorry

end NUMINAMATH_CALUDE_melanies_dimes_l3102_310203


namespace NUMINAMATH_CALUDE_two_color_theorem_l3102_310252

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A region in the plane --/
inductive Region
  | Inside (n : ℕ) -- Inside n circles
  | Outside        -- Outside all circles

/-- The type of coloring function --/
def Coloring := Region → Fin 2

/-- Two regions are adjacent if they differ by crossing one circle boundary --/
def adjacent (r1 r2 : Region) : Prop :=
  match r1, r2 with
  | Region.Inside n, Region.Inside m => n + 1 = m ∨ m + 1 = n
  | Region.Inside 1, Region.Outside => True
  | Region.Outside, Region.Inside 1 => True
  | _, _ => False

/-- A coloring is valid if adjacent regions have different colors --/
def valid_coloring (c : Coloring) : Prop :=
  ∀ r1 r2, adjacent r1 r2 → c r1 ≠ c r2

theorem two_color_theorem (circles : List Circle) :
  ∃ c : Coloring, valid_coloring c :=
sorry

end NUMINAMATH_CALUDE_two_color_theorem_l3102_310252


namespace NUMINAMATH_CALUDE_greatest_value_l3102_310283

theorem greatest_value (a b : ℝ) (ha : a = 2) (hb : b = 5) :
  let expr1 := a / b
  let expr2 := b / a
  let expr3 := a - b
  let expr4 := b - a
  let expr5 := (1 / 2) * a
  (expr4 ≥ expr1) ∧ (expr4 ≥ expr2) ∧ (expr4 ≥ expr3) ∧ (expr4 ≥ expr5) :=
by sorry

end NUMINAMATH_CALUDE_greatest_value_l3102_310283


namespace NUMINAMATH_CALUDE_sum_of_sqrt_inequality_l3102_310288

theorem sum_of_sqrt_inequality (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_sum : a + b + c = 1) : 
  Real.sqrt (4 * a + 1) + Real.sqrt (4 * b + 1) + Real.sqrt (4 * c + 1) < 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sqrt_inequality_l3102_310288


namespace NUMINAMATH_CALUDE_point_on_transformed_graph_l3102_310202

-- Define the function g
variable (g : ℝ → ℝ)

-- State the theorem
theorem point_on_transformed_graph (h : g 3 = 10) :
  ∃ (x y : ℝ), 3 * y = 4 * g (3 * x) + 6 ∧ x = 1 ∧ y = 46 / 3 ∧ x + y = 49 / 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_transformed_graph_l3102_310202


namespace NUMINAMATH_CALUDE_lighter_box_identification_l3102_310275

/-- A weighing strategy for identifying a lighter box among n boxes. -/
def WeighingStrategy (n : ℕ) := ℕ

/-- The number of weighings required to identify a lighter box among n boxes. -/
def NumWeighings (strategy : WeighingStrategy n) : ℕ := sorry

/-- Checks if a strategy correctly identifies the lighter box. -/
def IsValidStrategy (strategy : WeighingStrategy n) : Prop := sorry

theorem lighter_box_identification :
  ∃ (strategy : WeighingStrategy 15),
    IsValidStrategy strategy ∧ NumWeighings strategy ≤ 4 := by sorry

end NUMINAMATH_CALUDE_lighter_box_identification_l3102_310275


namespace NUMINAMATH_CALUDE_even_increasing_function_inequality_l3102_310231

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x < f y

theorem even_increasing_function_inequality (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_incr : increasing_on f (Set.Ici 0)) : 
  f (-2) < f (-3) ∧ f (-3) < f π := by
  sorry

end NUMINAMATH_CALUDE_even_increasing_function_inequality_l3102_310231


namespace NUMINAMATH_CALUDE_algebraic_expression_symmetry_l3102_310227

theorem algebraic_expression_symmetry (a b c : ℝ) : 
  a * (-5)^4 + b * (-5)^2 + c = 3 → a * 5^4 + b * 5^2 + c = 3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_symmetry_l3102_310227


namespace NUMINAMATH_CALUDE_f_odd_and_increasing_l3102_310273

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * |x|

-- Statement to prove
theorem f_odd_and_increasing : 
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) := by
sorry

end NUMINAMATH_CALUDE_f_odd_and_increasing_l3102_310273


namespace NUMINAMATH_CALUDE_max_fraction_sum_l3102_310249

theorem max_fraction_sum (a b c d : ℕ) (h1 : a/b + c/d < 1) (h2 : a + c = 20) :
  ∃ (a₀ b₀ c₀ d₀ : ℕ), 
    a₀/b₀ + c₀/d₀ = 1385/1386 ∧ 
    a₀ + c₀ = 20 ∧
    a₀/b₀ + c₀/d₀ < 1 ∧
    ∀ (x y z w : ℕ), x + z = 20 → x/y + z/w < 1 → x/y + z/w ≤ 1385/1386 :=
sorry

end NUMINAMATH_CALUDE_max_fraction_sum_l3102_310249


namespace NUMINAMATH_CALUDE_lakers_win_in_seven_l3102_310277

def probability_celtics_win : ℚ := 3/4

def probability_lakers_win : ℚ := 1 - probability_celtics_win

def games_to_win : ℕ := 4

def total_games : ℕ := 7

theorem lakers_win_in_seven (probability_celtics_win : ℚ) 
  (h1 : probability_celtics_win = 3/4) 
  (h2 : games_to_win = 4) 
  (h3 : total_games = 7) : 
  ℚ :=
by
  sorry

end NUMINAMATH_CALUDE_lakers_win_in_seven_l3102_310277


namespace NUMINAMATH_CALUDE_proportion_not_recent_boarders_l3102_310256

/-- Represents the proportion of passengers who boarded at a given dock -/
def boardingProportion : ℚ := 1/4

/-- Represents the proportion of departing passengers who boarded at the previous dock -/
def previousDockProportion : ℚ := 1/10

/-- Calculates the proportion of passengers who boarded at either of the two previous docks -/
def recentBoardersProportion : ℚ := 2 * boardingProportion - boardingProportion * previousDockProportion

/-- Theorem stating the proportion of passengers who did not board at either of the two previous docks -/
theorem proportion_not_recent_boarders :
  1 - recentBoardersProportion = 21/40 := by sorry

end NUMINAMATH_CALUDE_proportion_not_recent_boarders_l3102_310256


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_l3102_310232

theorem pure_imaginary_complex (m : ℝ) : 
  (m * (m - 1) : ℂ) + m * Complex.I = (0 : ℝ) + (b : ℝ) * Complex.I ∧ b ≠ 0 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_l3102_310232


namespace NUMINAMATH_CALUDE_cubic_root_equation_solutions_l3102_310271

theorem cubic_root_equation_solutions :
  let f : ℝ → ℝ := λ x => (17*x - 2)^(1/3) + (11*x + 2)^(1/3) - 2*(9*x)^(1/3)
  ∀ x : ℝ, f x = 0 ↔ x = 0 ∨ x = (2 + Real.sqrt 35) / 31 ∨ x = (2 - Real.sqrt 35) / 31 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_equation_solutions_l3102_310271


namespace NUMINAMATH_CALUDE_prob_ratio_balls_in_bins_l3102_310290

def factorial (n : ℕ) : ℕ := sorry

def multinomial_coefficient (n : ℕ) (x : List ℕ) : ℝ := sorry

def p (n : ℕ) (k : ℕ) : ℝ := 
  multinomial_coefficient n [3, 6, 5, 4, 2, 10]

def q (n : ℕ) (k : ℕ) : ℝ := 
  multinomial_coefficient n [5, 5, 5, 5, 5, 5]

theorem prob_ratio_balls_in_bins : 
  p 30 6 / q 30 6 = 0.125 := by sorry

end NUMINAMATH_CALUDE_prob_ratio_balls_in_bins_l3102_310290


namespace NUMINAMATH_CALUDE_triangle_theorem_l3102_310221

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b * Real.tan t.A = 2 * t.a * Real.sin t.B ∧
  t.a = Real.sqrt 7 ∧
  2 * t.b - t.c = 4

-- Define the theorem
theorem triangle_theorem (t : Triangle) (h : triangle_conditions t) :
  t.A = π / 3 ∧ (1/2 * t.b * t.c * Real.sin t.A = 3 * Real.sqrt 3 / 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3102_310221
