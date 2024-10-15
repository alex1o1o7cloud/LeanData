import Mathlib

namespace NUMINAMATH_CALUDE_system_solution_unique_l892_89223

theorem system_solution_unique (x y : ℚ) : 
  (4 * x + 7 * y = -19) ∧ (4 * x - 5 * y = 17) ↔ (x = 1/2) ∧ (y = -3) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l892_89223


namespace NUMINAMATH_CALUDE_zachs_bike_savings_l892_89200

/-- Represents the problem of calculating how much more money Zach needs to earn --/
theorem zachs_bike_savings (bike_cost : ℕ) (discount_rate : ℚ) 
  (weekly_allowance : ℕ) (lawn_mowing_min lawn_mowing_max : ℕ) 
  (garage_cleaning : ℕ) (babysitting_rate babysitting_hours : ℕ) 
  (loan_to_repay : ℕ) (current_savings : ℕ) : 
  bike_cost = 150 →
  discount_rate = 1/10 →
  weekly_allowance = 5 →
  lawn_mowing_min = 8 →
  lawn_mowing_max = 12 →
  garage_cleaning = 15 →
  babysitting_rate = 7 →
  babysitting_hours = 3 →
  loan_to_repay = 10 →
  current_savings = 65 →
  ∃ (additional_money : ℕ), additional_money = 27 ∧ 
    (bike_cost - (discount_rate * bike_cost).floor) - current_savings + loan_to_repay = 
    weekly_allowance + lawn_mowing_max + garage_cleaning + (babysitting_rate * babysitting_hours) + additional_money :=
by sorry

end NUMINAMATH_CALUDE_zachs_bike_savings_l892_89200


namespace NUMINAMATH_CALUDE_a_less_than_two_necessary_not_sufficient_l892_89221

/-- A quadratic equation x^2 + ax + 1 = 0 with real coefficient a -/
def quadratic_equation (a : ℝ) (x : ℂ) : Prop :=
  x^2 + a*x + 1 = 0

/-- The equation has complex roots -/
def has_complex_roots (a : ℝ) : Prop :=
  ∃ x : ℂ, quadratic_equation a x ∧ x.im ≠ 0

theorem a_less_than_two_necessary_not_sufficient :
  (∀ a : ℝ, has_complex_roots a → a < 2) ∧
  (∃ a : ℝ, a < 2 ∧ ¬has_complex_roots a) :=
sorry

end NUMINAMATH_CALUDE_a_less_than_two_necessary_not_sufficient_l892_89221


namespace NUMINAMATH_CALUDE_log_base_10_of_7_l892_89229

theorem log_base_10_of_7 (p q : ℝ) 
  (hp : Real.log 5 / Real.log 4 = p) 
  (hq : Real.log 7 / Real.log 5 = q) : 
  Real.log 7 / Real.log 10 = 2 * p * q / (2 * p + 1) := by
  sorry

end NUMINAMATH_CALUDE_log_base_10_of_7_l892_89229


namespace NUMINAMATH_CALUDE_hotel_assignment_theorem_l892_89230

/-- The number of ways to assign 6 friends to 6 rooms with given constraints -/
def assignmentWays : ℕ := sorry

/-- The total number of rooms available -/
def totalRooms : ℕ := 6

/-- The number of friends to be assigned -/
def totalFriends : ℕ := 6

/-- The maximum number of friends allowed per room -/
def maxFriendsPerRoom : ℕ := 2

/-- The maximum number of rooms that can be used -/
def maxRoomsUsed : ℕ := 5

theorem hotel_assignment_theorem :
  assignmentWays = 10440 ∧
  totalRooms = 6 ∧
  totalFriends = 6 ∧
  maxFriendsPerRoom = 2 ∧
  maxRoomsUsed = 5 := by sorry

end NUMINAMATH_CALUDE_hotel_assignment_theorem_l892_89230


namespace NUMINAMATH_CALUDE_system_solution_l892_89202

theorem system_solution (x y u v : ℝ) : 
  (x^2 + y^2 + u^2 + v^2 = 4) ∧
  (x*u + y*v + x*v + y*u = 0) ∧
  (x*y*u + y*u*v + u*v*x + v*x*y = -2) ∧
  (x*y*u*v = -1) →
  ((x = 1 ∧ y = 1 ∧ u = 1 ∧ v = -1) ∨
   (x = -1 + Real.sqrt 2 ∧ y = -1 - Real.sqrt 2 ∧ u = 1 ∧ v = -1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l892_89202


namespace NUMINAMATH_CALUDE_line_inclination_is_30_degrees_l892_89267

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 2 = 0

-- Define the angle of inclination
def angle_of_inclination (α : ℝ) : Prop := 
  ∃ (k : ℝ), k = Real.sqrt 3 / 3 ∧ k = Real.tan α

-- Theorem statement
theorem line_inclination_is_30_degrees : 
  ∃ (α : ℝ), angle_of_inclination α ∧ α = π / 6 :=
sorry

end NUMINAMATH_CALUDE_line_inclination_is_30_degrees_l892_89267


namespace NUMINAMATH_CALUDE_peter_statement_consistency_l892_89296

/-- Represents the day of the week -/
inductive Day
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

/-- Represents whether a person is telling the truth or lying -/
inductive TruthState
| Truthful | Lying

/-- Represents a statement that can be made -/
inductive Statement
| A | B | C | D | E

/-- Function to determine if a day follows another -/
def follows (d1 d2 : Day) : Prop := sorry

/-- Function to determine if a number is divisible by another -/
def is_divisible_by (n m : Nat) : Prop := sorry

/-- Peter's truth-telling state on a given day -/
def peter_truth_state (d : Day) : TruthState := sorry

/-- The content of each statement -/
def statement_content (s : Statement) (today : Day) : Prop :=
  match s with
  | Statement.A => peter_truth_state (sorry : Day) = TruthState.Lying ∧ 
                   peter_truth_state (sorry : Day) = TruthState.Lying
  | Statement.B => peter_truth_state today = TruthState.Truthful ∧ 
                   peter_truth_state (sorry : Day) = TruthState.Truthful
  | Statement.C => is_divisible_by 2024 11
  | Statement.D => (sorry : Day) = Day.Wednesday
  | Statement.E => follows (sorry : Day) Day.Saturday

/-- The main theorem -/
theorem peter_statement_consistency 
  (today : Day) 
  (statements : Finset Statement) 
  (h1 : statements.card = 4) 
  (h2 : Statement.C ∉ statements) :
  ∀ s ∈ statements, 
    (peter_truth_state today = TruthState.Truthful → statement_content s today) ∧
    (peter_truth_state today = TruthState.Lying → ¬statement_content s today) := by
  sorry


end NUMINAMATH_CALUDE_peter_statement_consistency_l892_89296


namespace NUMINAMATH_CALUDE_miles_driven_l892_89238

/-- Given a journey with a total distance and remaining distance, 
    calculate the distance already traveled. -/
theorem miles_driven (total_journey : ℕ) (remaining : ℕ) 
    (h1 : total_journey = 1200) 
    (h2 : remaining = 816) : 
  total_journey - remaining = 384 := by
  sorry

end NUMINAMATH_CALUDE_miles_driven_l892_89238


namespace NUMINAMATH_CALUDE_sector_max_area_l892_89207

/-- Given a sector with circumference 36, the radian measure of the central angle
    that maximizes the area of the sector is 2. -/
theorem sector_max_area (r : ℝ) (α : ℝ) (h1 : r > 0) (h2 : α > 0) 
  (h3 : 2 * r + r * α = 36) : 
  (∀ β : ℝ, β > 0 → 2 * r + r * β = 36 → r * r * α / 2 ≤ r * r * β / 2) → α = 2 := 
sorry

end NUMINAMATH_CALUDE_sector_max_area_l892_89207


namespace NUMINAMATH_CALUDE_favorite_season_fall_l892_89295

theorem favorite_season_fall (total_students : ℕ) (winter_angle spring_angle : ℝ) :
  total_students = 600 →
  winter_angle = 90 →
  spring_angle = 60 →
  (total_students : ℝ) * (360 - winter_angle - spring_angle - 180) / 360 = 50 :=
by sorry

end NUMINAMATH_CALUDE_favorite_season_fall_l892_89295


namespace NUMINAMATH_CALUDE_log_2_base_10_bound_l892_89241

theorem log_2_base_10_bound (h1 : 2^11 = 2048) (h2 : 2^12 = 4096) (h3 : 10^4 = 10000) :
  Real.log 2 / Real.log 10 < 4/11 := by
sorry

end NUMINAMATH_CALUDE_log_2_base_10_bound_l892_89241


namespace NUMINAMATH_CALUDE_line_circle_intersection_l892_89288

/-- The line x - y + 1 = 0 intersects the circle (x - a)² + y² = 2 
    if and only if a is in the closed interval [-3, 1] -/
theorem line_circle_intersection (a : ℝ) : 
  (∃ x y : ℝ, x - y + 1 = 0 ∧ (x - a)^2 + y^2 = 2) ↔ a ∈ Set.Icc (-3) 1 := by
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l892_89288


namespace NUMINAMATH_CALUDE_cannot_be_even_after_odd_operations_l892_89213

/-- Represents the parity of a number -/
inductive Parity
  | Even
  | Odd

/-- Function to determine the parity of a number -/
def getParity (n : ℕ) : Parity :=
  if n % 2 = 0 then Parity.Even else Parity.Odd

/-- Function to toggle the parity -/
def toggleParity (p : Parity) : Parity :=
  match p with
  | Parity.Even => Parity.Odd
  | Parity.Odd => Parity.Even

theorem cannot_be_even_after_odd_operations
  (initial : ℕ)
  (operations : ℕ)
  (h_initial_even : getParity initial = Parity.Even)
  (h_operations_odd : getParity operations = Parity.Odd) :
  ∃ (final : ℕ), getParity final = Parity.Odd ∧
    ∃ (f : ℕ → ℕ), (∀ n, f n = n + 1 ∨ f n = n - 1) ∧
      (f^[operations] initial = final) :=
sorry

end NUMINAMATH_CALUDE_cannot_be_even_after_odd_operations_l892_89213


namespace NUMINAMATH_CALUDE_equation_represents_parabola_l892_89268

/-- The equation |y - 3| = √((x+4)² + (y-1)²) represents a parabola -/
theorem equation_represents_parabola :
  ∃ (a b c : ℝ), a ≠ 0 ∧
  (∀ x y : ℝ, |y - 3| = Real.sqrt ((x + 4)^2 + (y - 1)^2) →
    y = a * x^2 + b * x + c) :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_parabola_l892_89268


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l892_89203

theorem fraction_sum_equality (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  1 / (x - 2) + 2 / (x + 2) + 4 / (4 - x^2) = 3 / (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l892_89203


namespace NUMINAMATH_CALUDE_mark_payment_l892_89297

def hours : ℕ := 3
def hourly_rate : ℚ := 15
def tip_percentage : ℚ := 20 / 100

def total_paid : ℚ :=
  let base_cost := hours * hourly_rate
  let tip := base_cost * tip_percentage
  base_cost + tip

theorem mark_payment : total_paid = 54 := by
  sorry

end NUMINAMATH_CALUDE_mark_payment_l892_89297


namespace NUMINAMATH_CALUDE_proportion_problem_l892_89252

theorem proportion_problem (x y : ℝ) : 
  (0.60 : ℝ) / x = y / 2 → 
  x = 0.19999999999999998 → 
  y = 6 := by sorry

end NUMINAMATH_CALUDE_proportion_problem_l892_89252


namespace NUMINAMATH_CALUDE_sum_first_100_even_integers_l892_89226

/-- The sum of the first n positive even integers -/
def sumFirstNEvenIntegers (n : ℕ) : ℕ := n * (n + 1)

/-- Theorem: The sum of the first 100 positive even integers is 10100 -/
theorem sum_first_100_even_integers : sumFirstNEvenIntegers 100 = 10100 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_100_even_integers_l892_89226


namespace NUMINAMATH_CALUDE_tommy_savings_tommy_current_savings_l892_89219

/-- Calculates the amount of money Tommy already has -/
theorem tommy_savings (num_books : ℕ) (cost_per_book : ℕ) (amount_to_save : ℕ) : ℕ :=
  num_books * cost_per_book - amount_to_save

/-- Proves that Tommy already has $13 -/
theorem tommy_current_savings : tommy_savings 8 5 27 = 13 := by
  sorry

end NUMINAMATH_CALUDE_tommy_savings_tommy_current_savings_l892_89219


namespace NUMINAMATH_CALUDE_abc_product_l892_89285

theorem abc_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * (b + c) = 152) (h2 : b * (c + a) = 162) (h3 : c * (a + b) = 170) :
  a * b * c = 720 := by
sorry

end NUMINAMATH_CALUDE_abc_product_l892_89285


namespace NUMINAMATH_CALUDE_unique_solution_l892_89274

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the equation TETA + BETA = GAMMA -/
def EquationSatisfied (T E B G M A : Digit) : Prop :=
  1000 * T.val + 100 * E.val + 10 * T.val + A.val +
  1000 * B.val + 100 * E.val + 10 * T.val + A.val =
  10000 * G.val + 1000 * A.val + 100 * M.val + 10 * M.val + A.val

/-- All digits are different except for repeated letters -/
def DigitsDifferent (T E B G M A : Digit) : Prop :=
  T ≠ E ∧ T ≠ B ∧ T ≠ G ∧ T ≠ M ∧ T ≠ A ∧
  E ≠ B ∧ E ≠ G ∧ E ≠ M ∧ E ≠ A ∧
  B ≠ G ∧ B ≠ M ∧ B ≠ A ∧
  G ≠ M ∧ G ≠ A ∧
  M ≠ A

theorem unique_solution :
  ∃! (T E B G M A : Digit),
    EquationSatisfied T E B G M A ∧
    DigitsDifferent T E B G M A ∧
    T.val = 4 ∧ E.val = 9 ∧ B.val = 5 ∧ G.val = 1 ∧ M.val = 8 ∧ A.val = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l892_89274


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l892_89234

theorem sufficient_but_not_necessary (x : ℝ) :
  (∀ x, x = 1 → x^2 ≠ 1) ∧ 
  ¬(∀ x, x^2 ≠ 1 → x = 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l892_89234


namespace NUMINAMATH_CALUDE_unique_root_when_b_zero_c_positive_odd_function_when_c_zero_symmetric_about_zero_one_iff_c_one_l892_89263

-- Define the function f
def f (x b c : ℝ) : ℝ := |x| * x + b * x + c

-- Theorem 1: When b=0 and c>0, f(x) = 0 has only one root
theorem unique_root_when_b_zero_c_positive (c : ℝ) (hc : c > 0) :
  ∃! x : ℝ, f x 0 c = 0 :=
sorry

-- Theorem 2: When c=0, y=f(x) is an odd function
theorem odd_function_when_c_zero (b : ℝ) :
  ∀ x : ℝ, f (-x) b 0 = -f x b 0 :=
sorry

-- Theorem 3: The graph of y=f(x) is symmetric about (0,1) iff c=1
theorem symmetric_about_zero_one_iff_c_one (b : ℝ) :
  (∀ x : ℝ, f x b 1 = 2 - f (-x) b 1) ↔ c = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_root_when_b_zero_c_positive_odd_function_when_c_zero_symmetric_about_zero_one_iff_c_one_l892_89263


namespace NUMINAMATH_CALUDE_system_solution_l892_89201

-- Define the solution set
def solution_set := {x : ℝ | 0 < x ∧ x < 1}

-- Define the system of inequalities
def inequality1 (x : ℝ) := |x| - 1 < 0
def inequality2 (x : ℝ) := x^2 - 3*x < 0

-- Theorem statement
theorem system_solution :
  ∀ x : ℝ, x ∈ solution_set ↔ (inequality1 x ∧ inequality2 x) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l892_89201


namespace NUMINAMATH_CALUDE_fourth_selected_is_34_l892_89272

/-- Represents a systematic sampling of students -/
structure SystematicSampling where
  total_students : Nat
  num_groups : Nat
  first_selected : Nat
  second_selected : Nat

/-- Calculates the number of the selected student for a given group -/
def selected_student (s : SystematicSampling) (group : Nat) : Nat :=
  s.first_selected + (s.total_students / s.num_groups) * group

/-- Theorem stating that the fourth selected student will be number 34 -/
theorem fourth_selected_is_34 (s : SystematicSampling) 
  (h1 : s.total_students = 50)
  (h2 : s.num_groups = 5)
  (h3 : s.first_selected = 4)
  (h4 : s.second_selected = 14) :
  selected_student s 3 = 34 := by
  sorry

end NUMINAMATH_CALUDE_fourth_selected_is_34_l892_89272


namespace NUMINAMATH_CALUDE_kaleb_fair_expense_l892_89294

/-- Calculates the total cost of rides given the number of tickets used and the cost per ticket -/
def total_cost (tickets_used : ℕ) (cost_per_ticket : ℕ) : ℕ :=
  tickets_used * cost_per_ticket

theorem kaleb_fair_expense :
  let initial_tickets : ℕ := 6
  let ferris_wheel_cost : ℕ := 2
  let bumper_cars_cost : ℕ := 1
  let roller_coaster_cost : ℕ := 2
  let ticket_price : ℕ := 9
  let total_tickets_used : ℕ := ferris_wheel_cost + bumper_cars_cost + roller_coaster_cost
  total_cost total_tickets_used ticket_price = 45 := by
  sorry

#eval total_cost 5 9

end NUMINAMATH_CALUDE_kaleb_fair_expense_l892_89294


namespace NUMINAMATH_CALUDE_estimated_probability_is_two_fifths_l892_89210

/-- Represents a set of three-digit numbers -/
def RandomSet : Type := List Nat

/-- Checks if a number represents a rainy day (1-6) -/
def isRainyDay (n : Nat) : Bool :=
  1 ≤ n ∧ n ≤ 6

/-- Counts the number of rainy days in a three-digit number -/
def countRainyDays (n : Nat) : Nat :=
  (if isRainyDay (n / 100) then 1 else 0) +
  (if isRainyDay ((n / 10) % 10) then 1 else 0) +
  (if isRainyDay (n % 10) then 1 else 0)

/-- Checks if a number represents exactly two rainy days -/
def hasTwoRainyDays (n : Nat) : Bool :=
  countRainyDays n = 2

/-- The given set of random numbers -/
def givenSet : RandomSet :=
  [180, 792, 454, 417, 165, 809, 798, 386, 196, 206]

/-- Theorem: The estimated probability of exactly two rainy days is 2/5 -/
theorem estimated_probability_is_two_fifths :
  (givenSet.filter hasTwoRainyDays).length / givenSet.length = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_estimated_probability_is_two_fifths_l892_89210


namespace NUMINAMATH_CALUDE_shooting_probability_l892_89275

theorem shooting_probability (p : ℝ) : 
  (p ≥ 0 ∧ p ≤ 1) →  -- Ensure p is a valid probability
  (1 - (1 - 1/2) * (1 - 2/3) * (1 - p) = 7/8) → 
  p = 1/4 := by
sorry

end NUMINAMATH_CALUDE_shooting_probability_l892_89275


namespace NUMINAMATH_CALUDE_linear_equation_solution_l892_89298

theorem linear_equation_solution (x y : ℝ) : x - 3 * y = 4 ↔ x = 1 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l892_89298


namespace NUMINAMATH_CALUDE_three_digit_square_sum_l892_89262

theorem three_digit_square_sum (N : ℕ) : 
  (100 ≤ N ∧ N ≤ 999) →
  (∃ (a b c : ℕ), 
    0 ≤ a ∧ a ≤ 9 ∧ a ≠ 0 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    N = 100 * a + 10 * b + c ∧
    N = 11 * (a^2 + b^2 + c^2)) →
  (N = 550 ∨ N = 803) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_square_sum_l892_89262


namespace NUMINAMATH_CALUDE_art_museum_survey_l892_89244

theorem art_museum_survey (total : ℕ) (not_enjoyed_not_understood : ℕ) (enjoyed : ℕ) (understood : ℕ) :
  total = 400 →
  not_enjoyed_not_understood = 100 →
  enjoyed = understood →
  (enjoyed : ℚ) / total = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_art_museum_survey_l892_89244


namespace NUMINAMATH_CALUDE_simplify_expression_l892_89227

theorem simplify_expression : 1 - 1 / (1 + Real.sqrt 5) + 1 / (1 - Real.sqrt 5) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l892_89227


namespace NUMINAMATH_CALUDE_car_expenses_sum_l892_89280

theorem car_expenses_sum : 
  let speakers_cost : ℚ := 118.54
  let tires_cost : ℚ := 106.33
  let tints_cost : ℚ := 85.27
  let maintenance_cost : ℚ := 199.75
  let cover_cost : ℚ := 15.63
  speakers_cost + tires_cost + tints_cost + maintenance_cost + cover_cost = 525.52 := by
  sorry

end NUMINAMATH_CALUDE_car_expenses_sum_l892_89280


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l892_89255

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3}

theorem complement_of_A_in_U :
  {x ∈ U | x ∉ A} = {2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l892_89255


namespace NUMINAMATH_CALUDE_quadratic_factorization_l892_89256

theorem quadratic_factorization (a b c : ℤ) :
  (∀ x, x^2 + 11*x + 28 = (x + a) * (x + b)) →
  (∀ x, x^2 + 7*x - 60 = (x + b) * (x - c)) →
  a + b + c = 21 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l892_89256


namespace NUMINAMATH_CALUDE_quincy_peter_picture_difference_l892_89248

theorem quincy_peter_picture_difference :
  ∀ (peter_pictures randy_pictures quincy_pictures total_pictures : ℕ),
    peter_pictures = 8 →
    randy_pictures = 5 →
    total_pictures = 41 →
    total_pictures = peter_pictures + randy_pictures + quincy_pictures →
    quincy_pictures - peter_pictures = 20 := by
  sorry

end NUMINAMATH_CALUDE_quincy_peter_picture_difference_l892_89248


namespace NUMINAMATH_CALUDE_houses_with_pool_count_l892_89235

/-- Represents the number of houses in a development with various features -/
structure Development where
  total : ℕ
  with_garage : ℕ
  with_both : ℕ
  with_neither : ℕ

/-- The number of houses with an in-the-ground swimming pool in the development -/
def houses_with_pool (d : Development) : ℕ :=
  d.total - d.with_garage + d.with_both - d.with_neither

/-- Theorem stating that in the given development, 40 houses have an in-the-ground swimming pool -/
theorem houses_with_pool_count (d : Development) 
  (h1 : d.total = 65)
  (h2 : d.with_garage = 50)
  (h3 : d.with_both = 35)
  (h4 : d.with_neither = 10) : 
  houses_with_pool d = 40 := by
  sorry

end NUMINAMATH_CALUDE_houses_with_pool_count_l892_89235


namespace NUMINAMATH_CALUDE_tennis_balls_problem_l892_89249

theorem tennis_balls_problem (brian frodo lily : ℕ) : 
  brian = 2 * frodo ∧ 
  frodo = lily + 8 ∧ 
  brian = 22 → 
  lily = 3 := by sorry

end NUMINAMATH_CALUDE_tennis_balls_problem_l892_89249


namespace NUMINAMATH_CALUDE_pyramid_scheme_characterization_l892_89247

/-- Represents a financial scheme -/
structure FinancialScheme where
  returns : ℝ
  information_completeness : ℝ
  advertising_aggressiveness : ℝ

/-- Defines the average market return -/
def average_market_return : ℝ := sorry

/-- Defines the threshold for complete information -/
def complete_information_threshold : ℝ := sorry

/-- Defines the threshold for aggressive advertising -/
def aggressive_advertising_threshold : ℝ := sorry

/-- Determines if a financial scheme is a pyramid scheme -/
def is_pyramid_scheme (scheme : FinancialScheme) : Prop :=
  scheme.returns > average_market_return ∧
  scheme.information_completeness < complete_information_threshold ∧
  scheme.advertising_aggressiveness > aggressive_advertising_threshold

theorem pyramid_scheme_characterization (scheme : FinancialScheme) :
  is_pyramid_scheme scheme ↔
    scheme.returns > average_market_return ∧
    scheme.information_completeness < complete_information_threshold ∧
    scheme.advertising_aggressiveness > aggressive_advertising_threshold := by
  sorry

end NUMINAMATH_CALUDE_pyramid_scheme_characterization_l892_89247


namespace NUMINAMATH_CALUDE_two_Z_six_l892_89281

/-- Definition of the operation Z -/
def Z (a b : ℤ) : ℤ := b + 10 * a - a ^ 2

/-- Theorem stating that 2Z6 = 22 -/
theorem two_Z_six : Z 2 6 = 22 := by
  sorry

end NUMINAMATH_CALUDE_two_Z_six_l892_89281


namespace NUMINAMATH_CALUDE_max_sum_given_sum_of_squares_and_product_l892_89289

theorem max_sum_given_sum_of_squares_and_product (x y : ℝ) :
  x^2 + y^2 = 98 → xy = 36 → x + y ≤ Real.sqrt 170 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_given_sum_of_squares_and_product_l892_89289


namespace NUMINAMATH_CALUDE_derivative_product_at_one_and_neg_one_l892_89204

-- Define the function f
def f (x : ℝ) : ℝ := x * (1 + abs x)

-- State the theorem
theorem derivative_product_at_one_and_neg_one :
  (deriv f 1) * (deriv f (-1)) = 9 := by sorry

end NUMINAMATH_CALUDE_derivative_product_at_one_and_neg_one_l892_89204


namespace NUMINAMATH_CALUDE_right_angled_triangle_set_l892_89209

theorem right_angled_triangle_set : ∃! (a b c : ℝ), 
  ((a = 1 ∧ b = Real.sqrt 2 ∧ c = Real.sqrt 3) ∨
   (a = 2 ∧ b = 3 ∧ c = 4) ∨
   (a = 4 ∧ b = 6 ∧ c = 8) ∨
   (a = 5 ∧ b = 12 ∧ c = 15)) ∧
  a^2 + b^2 = c^2 := by
sorry

end NUMINAMATH_CALUDE_right_angled_triangle_set_l892_89209


namespace NUMINAMATH_CALUDE_total_pictures_sum_l892_89269

/-- Represents the number of pictures Zoe has taken -/
structure PictureCount where
  initial : ℕ
  dolphinShow : ℕ
  total : ℕ

/-- Theorem: The total number of pictures is the sum of initial and dolphin show pictures -/
theorem total_pictures_sum (z : PictureCount) 
  (h1 : z.initial = 28)
  (h2 : z.dolphinShow = 16)
  (h3 : z.total = 44) :
  z.total = z.initial + z.dolphinShow := by
  sorry

#check total_pictures_sum

end NUMINAMATH_CALUDE_total_pictures_sum_l892_89269


namespace NUMINAMATH_CALUDE_parabola_distance_theorem_l892_89283

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define point B
def B : ℝ × ℝ := (3, 0)

-- Theorem statement
theorem parabola_distance_theorem (A : ℝ × ℝ) :
  parabola A.1 A.2 →  -- A lies on the parabola
  (A.1 - focus.1)^2 + (A.2 - focus.2)^2 = (B.1 - focus.1)^2 + (B.2 - focus.2)^2 →  -- |AF| = |BF|
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 8 :=  -- |AB|^2 = 8, which implies |AB| = 2√2
by
  sorry


end NUMINAMATH_CALUDE_parabola_distance_theorem_l892_89283


namespace NUMINAMATH_CALUDE_triangle_problem_l892_89265

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) →
  (a * Real.sin C) / (1 - Real.cos A) = Real.sqrt 3 * c →
  b + c = 10 →
  (1 / 2) * b * c * Real.sin A = 4 * Real.sqrt 3 →
  (A = π / 3) ∧ (a = 2 * Real.sqrt 13) := by
sorry


end NUMINAMATH_CALUDE_triangle_problem_l892_89265


namespace NUMINAMATH_CALUDE_equation_solution_l892_89251

theorem equation_solution :
  ∃! x : ℝ, (3 : ℝ)^x * (9 : ℝ)^x = (27 : ℝ)^(x - 4) :=
by
  use -6
  sorry

end NUMINAMATH_CALUDE_equation_solution_l892_89251


namespace NUMINAMATH_CALUDE_meat_for_hamburgers_l892_89279

/-- Given that 5 pounds of meat can make 12 hamburgers, 
    prove that 15 pounds of meat are needed to make 36 hamburgers. -/
theorem meat_for_hamburgers : 
  ∀ (meat_per_batch : ℝ) (hamburgers_per_batch : ℝ) (total_hamburgers : ℝ),
    meat_per_batch = 5 →
    hamburgers_per_batch = 12 →
    total_hamburgers = 36 →
    (meat_per_batch / hamburgers_per_batch) * total_hamburgers = 15 := by
  sorry

end NUMINAMATH_CALUDE_meat_for_hamburgers_l892_89279


namespace NUMINAMATH_CALUDE_floor_pi_plus_four_l892_89206

theorem floor_pi_plus_four : ⌊Real.pi + 4⌋ = 7 := by
  sorry

end NUMINAMATH_CALUDE_floor_pi_plus_four_l892_89206


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l892_89240

theorem purely_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := (m^2 - m - 2) + (m + 1)*Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → m = 2 := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l892_89240


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l892_89299

theorem negation_of_existential_proposition :
  (¬ ∃ n : ℕ, n + 10 / n < 4) ↔ (∀ n : ℕ, n + 10 / n ≥ 4) := by sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l892_89299


namespace NUMINAMATH_CALUDE_unique_pie_solution_l892_89220

/-- Represents the number of pies of each type -/
structure PieCount where
  raspberry : ℕ
  blueberry : ℕ
  strawberry : ℕ

/-- Checks if the given pie counts satisfy the problem conditions -/
def satisfiesConditions (pies : PieCount) : Prop :=
  pies.raspberry = (pies.raspberry + pies.blueberry + pies.strawberry) / 2 ∧
  pies.blueberry = pies.raspberry - 14 ∧
  pies.strawberry = (pies.raspberry + pies.blueberry) / 2

/-- Theorem stating that the given pie counts are the unique solution -/
theorem unique_pie_solution :
  ∃! (pies : PieCount), satisfiesConditions pies ∧
    pies.raspberry = 21 ∧ pies.blueberry = 7 ∧ pies.strawberry = 14 := by
  sorry

end NUMINAMATH_CALUDE_unique_pie_solution_l892_89220


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l892_89260

theorem contrapositive_equivalence (x y : ℝ) :
  (¬(x = 0 ∧ y = 0) → x^2 + y^2 ≠ 0) ↔ (x^2 + y^2 = 0 → x = 0 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l892_89260


namespace NUMINAMATH_CALUDE_sum_in_base10_l892_89212

/-- Converts a number from base 14 to base 10 -/
def base14ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 13 to base 10 -/
def base13ToBase10 (n : ℕ) : ℕ := sorry

/-- Represents the digit C in base 13 -/
def C : ℕ := 12

theorem sum_in_base10 : 
  base14ToBase10 356 + base13ToBase10 409 = 1505 := by sorry

end NUMINAMATH_CALUDE_sum_in_base10_l892_89212


namespace NUMINAMATH_CALUDE_flea_return_probability_l892_89237

/-- A flea jumps on a number line with the following properties:
    - It starts at 0
    - Each jump has a length of 1
    - The probability of jumping in the same direction as the previous jump is p
    - The probability of jumping in the opposite direction is 1-p -/
def FleaJump (p : ℝ) := 
  {flea : ℕ → ℝ // flea 0 = 0 ∧ ∀ n, |flea (n+1) - flea n| = 1}

/-- The probability that the flea returns to 0 -/
noncomputable def ReturnProbability (p : ℝ) : ℝ := sorry

/-- The theorem stating the probability of the flea returning to 0 -/
theorem flea_return_probability (p : ℝ) : 
  ReturnProbability p = if p = 1 then 0 else 1 := by sorry

end NUMINAMATH_CALUDE_flea_return_probability_l892_89237


namespace NUMINAMATH_CALUDE_complex_number_solution_l892_89271

theorem complex_number_solution (z : ℂ) (h : z + Complex.abs z = 2 + 8 * I) : z = -15 + 8 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_solution_l892_89271


namespace NUMINAMATH_CALUDE_students_behind_minyoung_l892_89208

/-- Given a line of students with Minyoung in it, this theorem proves
    that the number of students behind Minyoung is equal to the total
    number of students minus the number of students in front of Minyoung
    minus 1 (Minyoung herself). -/
theorem students_behind_minyoung
  (total_students : ℕ)
  (students_in_front : ℕ)
  (h1 : total_students = 35)
  (h2 : students_in_front = 27) :
  total_students - students_in_front - 1 = 7 :=
by sorry

end NUMINAMATH_CALUDE_students_behind_minyoung_l892_89208


namespace NUMINAMATH_CALUDE_jennas_driving_speed_l892_89290

/-- Proves that Jenna's driving speed is 50 miles per hour given the road trip conditions -/
theorem jennas_driving_speed 
  (total_distance : ℝ) 
  (jenna_distance : ℝ) 
  (friend_distance : ℝ)
  (total_time : ℝ) 
  (break_time : ℝ) 
  (friend_speed : ℝ) 
  (h1 : total_distance = jenna_distance + friend_distance)
  (h2 : total_distance = 300)
  (h3 : jenna_distance = 200)
  (h4 : friend_distance = 100)
  (h5 : total_time = 10)
  (h6 : break_time = 1)
  (h7 : friend_speed = 20) : 
  jenna_distance / (total_time - break_time - friend_distance / friend_speed) = 50 := by
  sorry

#check jennas_driving_speed

end NUMINAMATH_CALUDE_jennas_driving_speed_l892_89290


namespace NUMINAMATH_CALUDE_power_of_three_l892_89205

theorem power_of_three (m n : ℕ) (h1 : 3^m = 4) (h2 : 3^n = 5) : 3^(2*m + n) = 80 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_l892_89205


namespace NUMINAMATH_CALUDE_mauras_seashells_l892_89292

/-- Represents the number of seashells Maura found during her summer vacation. -/
def total_seashells : ℕ := 75

/-- Represents the number of seashells Maura kept after giving some to her sister. -/
def kept_seashells : ℕ := 57

/-- Represents the number of seashells Maura gave to her sister. -/
def given_seashells : ℕ := 18

/-- Represents the number of days Maura's family stayed at the beach house. -/
def beach_days : ℕ := 21

/-- Proves that the total number of seashells Maura found is equal to the sum of
    the seashells she kept and the seashells she gave away. -/
theorem mauras_seashells : total_seashells = kept_seashells + given_seashells := by
  sorry

end NUMINAMATH_CALUDE_mauras_seashells_l892_89292


namespace NUMINAMATH_CALUDE_jaguar_snake_consumption_l892_89282

theorem jaguar_snake_consumption 
  (beetles_per_bird : ℕ) 
  (birds_per_snake : ℕ) 
  (total_jaguars : ℕ) 
  (total_beetles_eaten : ℕ) 
  (h1 : beetles_per_bird = 12)
  (h2 : birds_per_snake = 3)
  (h3 : total_jaguars = 6)
  (h4 : total_beetles_eaten = 1080) :
  total_beetles_eaten / total_jaguars / beetles_per_bird / birds_per_snake = 5 := by
  sorry

end NUMINAMATH_CALUDE_jaguar_snake_consumption_l892_89282


namespace NUMINAMATH_CALUDE_sin_alpha_for_point_l892_89246

/-- If the terminal side of angle α passes through point (-2, 4), then sin α = (2√5) / 5 -/
theorem sin_alpha_for_point (α : Real) : 
  (∃ (r : Real), r > 0 ∧ r * (Real.cos α) = -2 ∧ r * (Real.sin α) = 4) →
  Real.sin α = (2 * Real.sqrt 5) / 5 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_for_point_l892_89246


namespace NUMINAMATH_CALUDE_problem_statement_l892_89233

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) 
  (h : b * Real.log a - a * Real.log b = a - b) : 
  (a + b - a * b > 1) ∧ (a + b > 2) ∧ (1 / a + 1 / b > 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l892_89233


namespace NUMINAMATH_CALUDE_rita_remaining_money_l892_89236

def initial_amount : ℕ := 400
def dress_cost : ℕ := 20
def pants_cost : ℕ := 12
def jacket_cost : ℕ := 30
def transportation_cost : ℕ := 5
def dress_quantity : ℕ := 5
def pants_quantity : ℕ := 3
def jacket_quantity : ℕ := 4

theorem rita_remaining_money :
  initial_amount - 
  (dress_cost * dress_quantity + 
   pants_cost * pants_quantity + 
   jacket_cost * jacket_quantity + 
   transportation_cost) = 139 := by
sorry

end NUMINAMATH_CALUDE_rita_remaining_money_l892_89236


namespace NUMINAMATH_CALUDE_winning_candidate_vote_percentage_l892_89254

/-- The percentage of votes received by the winning candidate in an election with three candidates -/
theorem winning_candidate_vote_percentage 
  (votes : Fin 3 → ℕ)
  (h1 : votes 0 = 3000)
  (h2 : votes 1 = 5000)
  (h3 : votes 2 = 15000) :
  (votes 2 : ℚ) / (votes 0 + votes 1 + votes 2) * 100 = 15000 / 23000 * 100 := by
  sorry

#eval (15000 : ℚ) / 23000 * 100 -- To display the approximate result

end NUMINAMATH_CALUDE_winning_candidate_vote_percentage_l892_89254


namespace NUMINAMATH_CALUDE_carpet_cost_l892_89266

/-- Calculates the total cost of carpeting a rectangular floor with square carpet tiles -/
theorem carpet_cost (floor_length floor_width carpet_side_length carpet_cost : ℝ) :
  floor_length = 6 ∧ 
  floor_width = 10 ∧ 
  carpet_side_length = 2 ∧ 
  carpet_cost = 15 →
  (floor_length * floor_width) / (carpet_side_length * carpet_side_length) * carpet_cost = 225 := by
  sorry

#check carpet_cost

end NUMINAMATH_CALUDE_carpet_cost_l892_89266


namespace NUMINAMATH_CALUDE_sum_of_factors_of_30_l892_89222

def sum_of_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_factors_of_30 : sum_of_factors 30 = 72 := by sorry

end NUMINAMATH_CALUDE_sum_of_factors_of_30_l892_89222


namespace NUMINAMATH_CALUDE_largest_number_l892_89259

theorem largest_number (π : ℝ) (h : 3 < π ∧ π < 4) : 
  π = max π (max 3 (max (1 - π) (-π^2))) := by
sorry

end NUMINAMATH_CALUDE_largest_number_l892_89259


namespace NUMINAMATH_CALUDE_no_valid_n_l892_89293

theorem no_valid_n : ¬∃ (n : ℕ), n > 0 ∧ 
  (∃ (x : ℕ), n^2 - 21*n + 110 = x^2) ∧ 
  (15 % n = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_n_l892_89293


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l892_89291

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Property of geometric sequences: if m + n = p + q, then a_m * a_n = a_p * a_q -/
axiom geometric_property {a : ℕ → ℝ} (h : GeometricSequence a) :
  ∀ m n p q : ℕ, m + n = p + q → a m * a n = a p * a q

theorem geometric_sequence_problem (a : ℕ → ℝ) (h : GeometricSequence a) 
    (h_sum : a 4 + a 8 = -3) :
  a 6 * (a 2 + 2 * a 6 + a 10) = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l892_89291


namespace NUMINAMATH_CALUDE_cylinder_min_surface_area_l892_89231

/-- Given a cone with base radius 4 and slant height 5, and a cylinder with equal volume,
    the surface area of the cylinder is minimized when its base radius is 2. -/
theorem cylinder_min_surface_area (r h : ℝ) : 
  r > 0 → h > 0 → 
  (π * r^2 * h = (1/3) * π * 4^2 * 3) →
  (∀ r' h' : ℝ, r' > 0 → h' > 0 → π * r'^2 * h' = (1/3) * π * 4^2 * 3 → 
    2 * π * r * (r + h) ≤ 2 * π * r' * (r' + h')) →
  r = 2 := by sorry

end NUMINAMATH_CALUDE_cylinder_min_surface_area_l892_89231


namespace NUMINAMATH_CALUDE_fb_is_80_l892_89284

/-- A right-angled triangle ABC with a point F on BC -/
structure TriangleABCF where
  /-- The length of side AB -/
  ab : ℝ
  /-- The length of side AC -/
  ac : ℝ
  /-- The length of side BC -/
  bc : ℝ
  /-- The length of BF -/
  bf : ℝ
  /-- The length of CF -/
  cf : ℝ
  /-- AB is 120 meters -/
  hab : ab = 120
  /-- AC is 160 meters -/
  hac : ac = 160
  /-- ABC is a right-angled triangle -/
  hright : ab^2 + ac^2 = bc^2
  /-- F is on BC -/
  hf_on_bc : bf + cf = bc
  /-- Jack and Jill jog the same distance -/
  heq_dist : ac + cf = ab + bf

/-- The main theorem: FB is 80 meters -/
theorem fb_is_80 (t : TriangleABCF) : t.bf = 80 := by
  sorry

end NUMINAMATH_CALUDE_fb_is_80_l892_89284


namespace NUMINAMATH_CALUDE_job_completion_time_l892_89215

theorem job_completion_time (y : ℝ) : y > 0 → (
  (1 / (y + 4) + 1 / (y + 2) + 1 / (2 * y) = 1 / y) ↔ y = 2
) := by sorry

end NUMINAMATH_CALUDE_job_completion_time_l892_89215


namespace NUMINAMATH_CALUDE_quiz_score_impossibility_l892_89287

theorem quiz_score_impossibility :
  ∀ (c u i : ℕ),
    c + u + i = 25 →
    4 * c + 2 * u - i ≠ 79 :=
by
  sorry

end NUMINAMATH_CALUDE_quiz_score_impossibility_l892_89287


namespace NUMINAMATH_CALUDE_simplify_expression_l892_89225

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) :
  Real.sqrt (4 + ((x^3 - 2) / (3*x))^2) = (Real.sqrt (x^6 - 4*x^3 + 36*x^2 + 4)) / (3*x) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l892_89225


namespace NUMINAMATH_CALUDE_least_integer_y_l892_89261

theorem least_integer_y : ∃ y : ℤ, (∀ z : ℤ, |3*z - 4| ≤ 25 → y ≤ z) ∧ |3*y - 4| ≤ 25 :=
by sorry

end NUMINAMATH_CALUDE_least_integer_y_l892_89261


namespace NUMINAMATH_CALUDE_right_triangle_count_l892_89232

/-- A point in 2D space -/
structure Point := (x : ℝ) (y : ℝ)

/-- Definition of the rectangle ABCD and points E, F, G -/
def rectangle_setup :=
  let A := Point.mk 0 0
  let B := Point.mk 6 0
  let C := Point.mk 6 4
  let D := Point.mk 0 4
  let E := Point.mk 3 0
  let F := Point.mk 3 4
  let G := Point.mk 2 4
  (A, B, C, D, E, F, G)

/-- Function to count right triangles -/
def count_right_triangles (points : Point × Point × Point × Point × Point × Point × Point) : ℕ :=
  sorry -- Implementation details omitted

/-- Theorem stating that the number of right triangles is 16 -/
theorem right_triangle_count :
  count_right_triangles rectangle_setup = 16 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_count_l892_89232


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l892_89264

-- Define the complex number z
def z : ℂ := (3 + Complex.I) * (1 - Complex.I)

-- Theorem stating that z is in the fourth quadrant
theorem z_in_fourth_quadrant :
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l892_89264


namespace NUMINAMATH_CALUDE_prob_at_least_three_speak_l892_89218

/-- The probability of a single baby speaking -/
def p : ℚ := 1/5

/-- The number of babies in the cluster -/
def n : ℕ := 7

/-- The probability that exactly k out of n babies will speak -/
def prob_exactly (k : ℕ) : ℚ :=
  (n.choose k) * (1 - p)^(n - k) * p^k

/-- The probability that at least 3 out of 7 babies will speak -/
theorem prob_at_least_three_speak : 
  1 - (prob_exactly 0 + prob_exactly 1 + prob_exactly 2) = 45349/78125 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_three_speak_l892_89218


namespace NUMINAMATH_CALUDE_inequalities_theorem_l892_89273

theorem inequalities_theorem (a b c d : ℝ) 
  (h1 : a > 0) (h2 : 0 > b) (h3 : b > -a) (h4 : c < d) (h5 : d < 0) : 
  (a * d ≤ b * c) ∧ 
  (a / d + b / c < 0) ∧ 
  (a - c > b - d) ∧ 
  (a * (d - c) > b * (d - c)) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_theorem_l892_89273


namespace NUMINAMATH_CALUDE_units_digit_G_100_l892_89216

/-- The sequence G_n defined as 3^(3^n) + 1 -/
def G (n : ℕ) : ℕ := 3^(3^n) + 1

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Theorem stating that the units digit of G_100 is 4 -/
theorem units_digit_G_100 : unitsDigit (G 100) = 4 := by sorry

end NUMINAMATH_CALUDE_units_digit_G_100_l892_89216


namespace NUMINAMATH_CALUDE_smallest_n_divisible_l892_89217

/-- The smallest positive integer n such that (x+1)^n - 1 is divisible by x^2 + 1 modulo 3 -/
def smallest_n : ℕ := 8

/-- The divisor polynomial -/
def divisor_poly (x : ℤ) : ℤ := x^2 + 1

/-- The dividend polynomial -/
def dividend_poly (x : ℤ) (n : ℕ) : ℤ := (x + 1)^n - 1

/-- Divisibility modulo 3 -/
def is_divisible_mod_3 (a b : ℤ → ℤ) : Prop :=
  ∃ (p q : ℤ → ℤ), ∀ x, a x = b x * p x + 3 * q x

theorem smallest_n_divisible :
  (∀ n < smallest_n, ¬ is_divisible_mod_3 (dividend_poly · n) divisor_poly) ∧
  is_divisible_mod_3 (dividend_poly · smallest_n) divisor_poly :=
sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_l892_89217


namespace NUMINAMATH_CALUDE_joanne_part_time_hours_l892_89258

/-- Calculates the number of hours Joanne works at her part-time job each day -/
def part_time_hours_per_day (main_job_hourly_rate : ℚ) (main_job_hours_per_day : ℚ) 
  (part_time_hourly_rate : ℚ) (days_per_week : ℚ) (total_weekly_earnings : ℚ) : ℚ :=
  let main_job_daily_earnings := main_job_hourly_rate * main_job_hours_per_day
  let main_job_weekly_earnings := main_job_daily_earnings * days_per_week
  let part_time_weekly_earnings := total_weekly_earnings - main_job_weekly_earnings
  let part_time_weekly_hours := part_time_weekly_earnings / part_time_hourly_rate
  part_time_weekly_hours / days_per_week

theorem joanne_part_time_hours : 
  part_time_hours_per_day 16 8 (27/2) 5 775 = 2 := by
  sorry

end NUMINAMATH_CALUDE_joanne_part_time_hours_l892_89258


namespace NUMINAMATH_CALUDE_bus_journey_distance_l892_89270

/-- Represents the bus journey with an obstruction --/
structure BusJourney where
  initialSpeed : ℝ
  totalDistance : ℝ
  obstructionTime : ℝ
  delayTime : ℝ
  speedReductionFactor : ℝ
  lateArrivalTime : ℝ
  alternativeObstructionDistance : ℝ
  alternativeLateArrivalTime : ℝ

/-- Theorem stating that given the conditions, the total distance of the journey is 570 miles --/
theorem bus_journey_distance (j : BusJourney) 
  (h1 : j.obstructionTime = 2)
  (h2 : j.delayTime = 2/3)
  (h3 : j.speedReductionFactor = 5/6)
  (h4 : j.lateArrivalTime = 2.75)
  (h5 : j.alternativeObstructionDistance = 50)
  (h6 : j.alternativeLateArrivalTime = 2 + 1/3)
  : j.totalDistance = 570 := by
  sorry

end NUMINAMATH_CALUDE_bus_journey_distance_l892_89270


namespace NUMINAMATH_CALUDE_elena_garden_petals_l892_89277

/-- The number of lilies in Elena's garden -/
def num_lilies : ℕ := 8

/-- The number of tulips in Elena's garden -/
def num_tulips : ℕ := 5

/-- The number of petals each lily has -/
def petals_per_lily : ℕ := 6

/-- The number of petals each tulip has -/
def petals_per_tulip : ℕ := 3

/-- The total number of flower petals in Elena's garden -/
def total_petals : ℕ := num_lilies * petals_per_lily + num_tulips * petals_per_tulip

theorem elena_garden_petals : total_petals = 63 := by
  sorry

end NUMINAMATH_CALUDE_elena_garden_petals_l892_89277


namespace NUMINAMATH_CALUDE_sum_of_numbers_l892_89253

theorem sum_of_numbers : 0.45 + 0.003 + (1/4 : ℚ) = 0.703 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l892_89253


namespace NUMINAMATH_CALUDE_probability_bounds_l892_89224

/-- A segment of natural numbers -/
structure Segment where
  start : ℕ
  length : ℕ

/-- The probability of a number in a segment being divisible by 10 -/
def probability_divisible_by_10 (s : Segment) : ℚ :=
  (s.length.div 10) / s.length

/-- The maximum probability of a number in any segment being divisible by 10 -/
def max_probability : ℚ := 1

/-- The minimum non-zero probability of a number in any segment being divisible by 10 -/
def min_nonzero_probability : ℚ := 1 / 19

theorem probability_bounds :
  ∀ s : Segment, 
    probability_divisible_by_10 s ≤ max_probability ∧
    (probability_divisible_by_10 s ≠ 0 → probability_divisible_by_10 s ≥ min_nonzero_probability) :=
by sorry

end NUMINAMATH_CALUDE_probability_bounds_l892_89224


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l892_89276

/-- An arithmetic sequence with first term 10, last term 140, and common difference 5 has 27 terms. -/
theorem arithmetic_sequence_terms : ∀ (a : ℕ → ℕ),
  (∀ n, a (n + 1) = a n + 5) →  -- arithmetic sequence with common difference 5
  a 1 = 10 →                    -- first term is 10
  (∃ m, a m = 140) →            -- last term is 140
  (∃ m, a m = 140 ∧ ∀ k, k > m → a k > 140) →  -- 140 is the last term not exceeding 140
  (∃ m, m = 27 ∧ a m = 140) :=  -- the sequence has exactly 27 terms
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l892_89276


namespace NUMINAMATH_CALUDE_square_of_1037_l892_89214

theorem square_of_1037 : (1037 : ℕ)^2 = 1074369 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_square_of_1037_l892_89214


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l892_89245

theorem solution_set_quadratic_inequality :
  Set.Ioo 0 3 = {x : ℝ | x^2 - 3*x < 0} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l892_89245


namespace NUMINAMATH_CALUDE_chessboard_tiling_l892_89278

/-- A type representing a chessboard -/
structure Chessboard :=
  (size : ℕ)

/-- A type representing a tiling piece -/
structure TilingPiece :=
  (coverage : ℕ)

/-- Function to check if a chessboard can be tiled with given pieces -/
def can_tile (board : Chessboard) (piece : TilingPiece) : Prop :=
  (board.size * board.size) % piece.coverage = 0

theorem chessboard_tiling :
  (∃ (piece : TilingPiece), piece.coverage = 4 ∧ can_tile ⟨8⟩ piece) ∧
  (∀ (piece : TilingPiece), piece.coverage = 4 → ¬can_tile ⟨10⟩ piece) :=
sorry

end NUMINAMATH_CALUDE_chessboard_tiling_l892_89278


namespace NUMINAMATH_CALUDE_custom_op_result_l892_89211

/-- Define the custom operation ã — -/
def custom_op (a b : ℤ) : ℤ := 2*a - 3*b + a*b

/-- Theorem stating that (1 ã — 2) - 2 = -4 -/
theorem custom_op_result : custom_op 1 2 - 2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_result_l892_89211


namespace NUMINAMATH_CALUDE_vector_operation_result_l892_89286

/-- Prove that the result of 3 * (-3, 2, 6) + (4, -5, 2) is (-5, 1, 20) -/
theorem vector_operation_result :
  (3 : ℝ) • ((-3 : ℝ), (2 : ℝ), (6 : ℝ)) + ((4 : ℝ), (-5 : ℝ), (2 : ℝ)) = ((-5 : ℝ), (1 : ℝ), (20 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_result_l892_89286


namespace NUMINAMATH_CALUDE_five_by_five_to_fifty_l892_89239

/-- Represents a square cut into pieces --/
structure CutSquare :=
  (side : ℕ)
  (pieces : ℕ)

/-- Represents the result of reassembling cut pieces --/
structure ReassembledSquares :=
  (count : ℕ)
  (side : ℚ)

/-- Function that cuts a square and reassembles the pieces --/
def cut_and_reassemble (s : CutSquare) : ReassembledSquares :=
  sorry

/-- Theorem stating that a 5x5 square can be cut and reassembled into 50 equal squares --/
theorem five_by_five_to_fifty :
  ∃ (cs : CutSquare) (rs : ReassembledSquares),
    cs.side = 5 ∧
    rs = cut_and_reassemble cs ∧
    rs.count = 50 ∧
    rs.side * rs.side * rs.count = cs.side * cs.side :=
  sorry

end NUMINAMATH_CALUDE_five_by_five_to_fifty_l892_89239


namespace NUMINAMATH_CALUDE_sum_reciprocals_l892_89243

theorem sum_reciprocals (a b c d : ℝ) (ω : ℂ) 
  (ha : a ≠ -2) (hb : b ≠ -2) (hc : c ≠ -2) (hd : d ≠ -2)
  (hω1 : ω^4 = 1) (hω2 : ω ≠ 1)
  (h : 1/(a + ω) + 1/(b + ω) + 1/(c + ω) + 1/(d + ω) = 4/ω) :
  1/(a + 2) + 1/(b + 2) + 1/(c + 2) + 1/(d + 2) = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocals_l892_89243


namespace NUMINAMATH_CALUDE_quadratic_two_real_roots_l892_89242

theorem quadratic_two_real_roots 
  (a b c : ℝ) 
  (h : a * (a + b + c) < 0) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_real_roots_l892_89242


namespace NUMINAMATH_CALUDE_quadratic_form_l892_89250

theorem quadratic_form (x : ℝ) : ∃ (b c : ℝ), x^2 - 16*x + 64 = (x + b)^2 + c ∧ b + c = -8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_l892_89250


namespace NUMINAMATH_CALUDE_reflected_beam_angle_l892_89228

/-- Given a fixed beam of light falling on a mirror at an acute angle α with its projection
    on the mirror plane, and the mirror rotated by an acute angle β around this projection,
    the angle θ between the two reflected beams (before and after rotation) is given by
    θ = arccos(1 - 2 * sin²α * sin²β) -/
theorem reflected_beam_angle (α β : Real) (h_α : 0 < α ∧ α < π/2) (h_β : 0 < β ∧ β < π/2) :
  ∃ θ : Real, θ = Real.arccos (1 - 2 * Real.sin α ^ 2 * Real.sin β ^ 2) :=
sorry

end NUMINAMATH_CALUDE_reflected_beam_angle_l892_89228


namespace NUMINAMATH_CALUDE_min_value_quadratic_l892_89257

theorem min_value_quadratic (x y : ℝ) : 2*x^2 + 3*y^2 - 8*x + 12*y + 40 ≥ 20 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l892_89257
