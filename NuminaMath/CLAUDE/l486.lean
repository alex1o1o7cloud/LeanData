import Mathlib

namespace NUMINAMATH_CALUDE_fathers_age_l486_48650

/-- Proves that given the conditions, the father's age is 70 years. -/
theorem fathers_age (man_age : ℕ) (father_age : ℕ) : 
  man_age = (2 / 5 : ℚ) * father_age →
  man_age + 14 = (1 / 2 : ℚ) * (father_age + 14) →
  father_age = 70 :=
by sorry

end NUMINAMATH_CALUDE_fathers_age_l486_48650


namespace NUMINAMATH_CALUDE_birthday_250_years_ago_l486_48632

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Calculates the day of the week that is n days before the given day -/
def daysBefore (d : DayOfWeek) (n : ℕ) : DayOfWeek :=
  sorry

/-- Calculates the number of leap years in a 250-year period, excluding certain century years -/
def leapYearsIn250Years : ℕ :=
  sorry

/-- Represents the number of days to go backwards for 250 years -/
def daysBackFor250Years : ℕ :=
  sorry

theorem birthday_250_years_ago (anniversary_day : DayOfWeek) : 
  anniversary_day = DayOfWeek.Tuesday → 
  daysBefore anniversary_day daysBackFor250Years = DayOfWeek.Saturday :=
sorry

end NUMINAMATH_CALUDE_birthday_250_years_ago_l486_48632


namespace NUMINAMATH_CALUDE_average_increase_food_expenditure_l486_48686

/-- Represents the regression line equation for annual income and food expenditure -/
def regression_line (x : ℝ) : ℝ := 0.245 * x + 0.321

/-- Theorem stating that the average increase in food expenditure for a unit increase in income is 0.245 -/
theorem average_increase_food_expenditure :
  ∀ x : ℝ, regression_line (x + 1) - regression_line x = 0.245 := by
sorry


end NUMINAMATH_CALUDE_average_increase_food_expenditure_l486_48686


namespace NUMINAMATH_CALUDE_arithmetic_seq_no_geometric_subseq_l486_48628

theorem arithmetic_seq_no_geometric_subseq
  (a : ℕ → ℝ)
  (h_arith : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n + r)
  (h_contains : ∃ k l : ℕ, a k = 1 ∧ a l = Real.sqrt 2)
  (h_index : ∀ n : ℕ, n ≥ 1) :
  ¬ ∃ m n p : ℕ, m ≠ n ∧ n ≠ p ∧ m ≠ p ∧
    (a n)^2 = (a m) * (a p) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_seq_no_geometric_subseq_l486_48628


namespace NUMINAMATH_CALUDE_mass_percentage_H_in_NH4I_l486_48635

-- Define atomic masses
def atomic_mass_N : ℝ := 14.01
def atomic_mass_H : ℝ := 1.01
def atomic_mass_I : ℝ := 126.90

-- Define the composition of NH4I
def NH4I_composition : Fin 3 → ℕ
  | 0 => 1  -- N
  | 1 => 4  -- H
  | 2 => 1  -- I
  | _ => 0

-- Define the molar mass of NH4I
def molar_mass_NH4I : ℝ :=
  NH4I_composition 0 * atomic_mass_N +
  NH4I_composition 1 * atomic_mass_H +
  NH4I_composition 2 * atomic_mass_I

-- Define the mass of hydrogen in NH4I
def mass_H_in_NH4I : ℝ := NH4I_composition 1 * atomic_mass_H

-- Theorem statement
theorem mass_percentage_H_in_NH4I :
  abs ((mass_H_in_NH4I / molar_mass_NH4I) * 100 - 2.79) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_mass_percentage_H_in_NH4I_l486_48635


namespace NUMINAMATH_CALUDE_percentage_relationship_l486_48672

theorem percentage_relationship (x : ℝ) (p : ℝ) :
  x = 120 →
  5.76 = p * (0.4 * x) →
  p = 0.12 :=
by sorry

end NUMINAMATH_CALUDE_percentage_relationship_l486_48672


namespace NUMINAMATH_CALUDE_inequality_solution_l486_48689

theorem inequality_solution (x : ℝ) : 
  (7 / 30 : ℝ) + |x - 13 / 60| < 11 / 20 ↔ -1 / 5 < x ∧ x < 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l486_48689


namespace NUMINAMATH_CALUDE_set_c_is_well_defined_l486_48600

-- Define the universe of discourse
def Student : Type := sorry

-- Define the properties
def SeniorHighSchool (s : Student) : Prop := sorry
def EnrolledAtDudeSchool (s : Student) : Prop := sorry
def EnrolledInJanuary2013 (s : Student) : Prop := sorry

-- Define the set C
def SetC : Set Student :=
  {s : Student | SeniorHighSchool s ∧ EnrolledAtDudeSchool s ∧ EnrolledInJanuary2013 s}

-- Define the property of being well-defined
def WellDefined (S : Set Student) : Prop :=
  ∀ s : Student, s ∈ S → (∃ (criterion : Student → Prop), criterion s)

-- Theorem statement
theorem set_c_is_well_defined :
  WellDefined SetC ∧ 
  (∀ S : Set Student, S ≠ SetC → ¬WellDefined S) :=
sorry

end NUMINAMATH_CALUDE_set_c_is_well_defined_l486_48600


namespace NUMINAMATH_CALUDE_line_inclination_angle_l486_48699

theorem line_inclination_angle (x y : ℝ) :
  y - 3 = Real.sqrt 3 * (x - 4) →
  ∃ α : ℝ, 0 ≤ α ∧ α < π ∧ Real.tan α = Real.sqrt 3 ∧ α = π / 3 :=
by sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l486_48699


namespace NUMINAMATH_CALUDE_arithmetic_operations_l486_48616

theorem arithmetic_operations : 
  (12 - (-18) + (-7) + (-15) = 8) ∧ 
  ((-1)^7 * 2 + (-3)^2 / 9 = -1) := by sorry

end NUMINAMATH_CALUDE_arithmetic_operations_l486_48616


namespace NUMINAMATH_CALUDE_correct_group_sizes_l486_48678

/-- Represents the pricing structure and group information for a scenic area in Xi'an --/
structure ScenicAreaPricing where
  regularPrice : ℕ
  nonHolidayDiscount : ℚ
  holidayDiscountThreshold : ℕ
  holidayDiscount : ℚ
  totalPeople : ℕ
  totalCost : ℕ

/-- Calculates the cost for a group visiting on a non-holiday --/
def nonHolidayCost (pricing : ScenicAreaPricing) (people : ℕ) : ℚ :=
  pricing.regularPrice * (1 - pricing.nonHolidayDiscount) * people

/-- Calculates the cost for a group visiting on a holiday --/
def holidayCost (pricing : ScenicAreaPricing) (people : ℕ) : ℚ :=
  if people ≤ pricing.holidayDiscountThreshold then
    pricing.regularPrice * people
  else
    pricing.regularPrice * pricing.holidayDiscountThreshold +
    pricing.regularPrice * (1 - pricing.holidayDiscount) * (people - pricing.holidayDiscountThreshold)

/-- Theorem stating the correct number of people in each group --/
theorem correct_group_sizes (pricing : ScenicAreaPricing)
  (h1 : pricing.regularPrice = 50)
  (h2 : pricing.nonHolidayDiscount = 0.4)
  (h3 : pricing.holidayDiscountThreshold = 10)
  (h4 : pricing.holidayDiscount = 0.2)
  (h5 : pricing.totalPeople = 50)
  (h6 : pricing.totalCost = 1840) :
  ∃ (groupA groupB : ℕ),
    groupA + groupB = pricing.totalPeople ∧
    holidayCost pricing groupA + nonHolidayCost pricing groupB = pricing.totalCost ∧
    groupA = 24 ∧ groupB = 26 := by
  sorry


end NUMINAMATH_CALUDE_correct_group_sizes_l486_48678


namespace NUMINAMATH_CALUDE_pizza_slice_count_l486_48614

/-- The total number of pizza slices given the conditions -/
def totalPizzaSlices (totalPizzas smallPizzaSlices largePizzaSlices : ℕ) : ℕ :=
  let smallPizzas := totalPizzas / 3
  let largePizzas := 2 * smallPizzas
  smallPizzas * smallPizzaSlices + largePizzas * largePizzaSlices

/-- Theorem stating that the total number of pizza slices is 384 -/
theorem pizza_slice_count :
  totalPizzaSlices 36 8 12 = 384 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slice_count_l486_48614


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l486_48667

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := circle_radius / 6
  let rectangle_area : ℝ := rectangle_length * rectangle_breadth
  rectangle_area

theorem rectangle_area_proof :
  rectangle_area 1296 10 = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l486_48667


namespace NUMINAMATH_CALUDE_circle_and_m_value_l486_48642

-- Define the curve
def curve (x y : ℝ) : Prop := y = x^2 - 4*x + 3

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 5

-- Define the line
def line (x y m : ℝ) : Prop := x + y + m = 0

-- Define perpendicularity condition
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem circle_and_m_value :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (curve 0 3 ∧ curve 1 0 ∧ curve 3 0) ∧  -- Intersection points with axes
    (circle_C 0 3 ∧ circle_C 1 0 ∧ circle_C 3 0) ∧  -- These points lie on circle C
    (∃ m : ℝ, 
      line x₁ y₁ m ∧ line x₂ y₂ m ∧  -- A and B lie on the line
      circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧  -- A and B lie on circle C
      perpendicular x₁ y₁ x₂ y₂ ∧  -- OA is perpendicular to OB
      (m = -1 ∨ m = -3))  -- The value of m
  :=
sorry

end NUMINAMATH_CALUDE_circle_and_m_value_l486_48642


namespace NUMINAMATH_CALUDE_remainder_53_pow_10_mod_8_l486_48655

theorem remainder_53_pow_10_mod_8 : 53^10 % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_53_pow_10_mod_8_l486_48655


namespace NUMINAMATH_CALUDE_contractor_fine_calculation_l486_48613

/-- Calculates the fine per day of absence for a contractor --/
def calculate_fine (contract_days : ℕ) (daily_pay : ℚ) (absent_days : ℕ) (total_payment : ℚ) : ℚ :=
  let worked_days := contract_days - absent_days
  let earned_amount := worked_days * daily_pay
  (earned_amount - total_payment) / absent_days

theorem contractor_fine_calculation :
  let contract_days : ℕ := 30
  let daily_pay : ℚ := 25
  let absent_days : ℕ := 6
  let total_payment : ℚ := 555
  calculate_fine contract_days daily_pay absent_days total_payment = 7.5 := by
  sorry

#eval calculate_fine 30 25 6 555

end NUMINAMATH_CALUDE_contractor_fine_calculation_l486_48613


namespace NUMINAMATH_CALUDE_money_remaining_l486_48603

/-- Given an initial amount of money and an amount spent, 
    the remaining amount is the difference between the two. -/
theorem money_remaining (initial spent : ℕ) : 
  initial = 16 → spent = 8 → initial - spent = 8 := by
  sorry

end NUMINAMATH_CALUDE_money_remaining_l486_48603


namespace NUMINAMATH_CALUDE_min_value_of_a_l486_48629

theorem min_value_of_a (a : ℝ) : 
  (∀ x ∈ Set.Ioc (0 : ℝ) (1/2), x^2 + a*x + 1 ≥ 0) → a ≥ -5/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_a_l486_48629


namespace NUMINAMATH_CALUDE_sum_of_inverse_conjugates_l486_48626

theorem sum_of_inverse_conjugates (m n : ℝ) : 
  m = (Real.sqrt 2 - 1)⁻¹ → n = (Real.sqrt 2 + 1)⁻¹ → m + n = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_inverse_conjugates_l486_48626


namespace NUMINAMATH_CALUDE_earnings_ratio_l486_48661

theorem earnings_ratio (total_earnings lottie_earnings jerusha_earnings : ℕ)
  (h1 : total_earnings = 85)
  (h2 : jerusha_earnings = 68)
  (h3 : total_earnings = lottie_earnings + jerusha_earnings)
  (h4 : ∃ k : ℕ, jerusha_earnings = k * lottie_earnings) :
  jerusha_earnings = 4 * lottie_earnings :=
by sorry

end NUMINAMATH_CALUDE_earnings_ratio_l486_48661


namespace NUMINAMATH_CALUDE_ball_probability_l486_48621

/-- Given a bag of 100 balls with specific color distributions, 
    prove that the probability of choosing a ball that is neither red nor purple is 0.8 -/
theorem ball_probability (total : ℕ) (white green yellow red purple : ℕ) 
  (h_total : total = 100)
  (h_white : white = 50)
  (h_green : green = 20)
  (h_yellow : yellow = 10)
  (h_red : red = 17)
  (h_purple : purple = 3)
  (h_sum : white + green + yellow + red + purple = total) :
  (white + green + yellow : ℚ) / total = 0.8 := by
sorry

end NUMINAMATH_CALUDE_ball_probability_l486_48621


namespace NUMINAMATH_CALUDE_probability_sqrt_less_than_9_l486_48630

/-- A two-digit whole number is an integer between 10 and 99, inclusive. -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The count of two-digit whole numbers whose square root is less than 9. -/
def CountLessThan9 : ℕ := 71

/-- The total count of two-digit whole numbers. -/
def TotalTwoDigitNumbers : ℕ := 90

/-- The probability that the square root of a randomly selected two-digit whole number is less than 9. -/
theorem probability_sqrt_less_than_9 :
  (CountLessThan9 : ℚ) / TotalTwoDigitNumbers = 71 / 90 := by sorry

end NUMINAMATH_CALUDE_probability_sqrt_less_than_9_l486_48630


namespace NUMINAMATH_CALUDE_haley_sunday_tv_hours_l486_48676

/-- Represents the number of hours Haley watched TV -/
structure TVWatchingHours where
  saturday : ℕ
  total : ℕ

/-- Calculates the number of hours Haley watched TV on Sunday -/
def sunday_hours (h : TVWatchingHours) : ℕ :=
  h.total - h.saturday

/-- Theorem stating that Haley watched TV for 3 hours on Sunday -/
theorem haley_sunday_tv_hours :
  ∀ h : TVWatchingHours, h.saturday = 6 → h.total = 9 → sunday_hours h = 3 := by
  sorry

end NUMINAMATH_CALUDE_haley_sunday_tv_hours_l486_48676


namespace NUMINAMATH_CALUDE_three_X_five_equals_two_l486_48698

def X (a b : ℝ) : ℝ := b + 8 * a - a^3

theorem three_X_five_equals_two : X 3 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_three_X_five_equals_two_l486_48698


namespace NUMINAMATH_CALUDE_same_function_l486_48636

theorem same_function (x : ℝ) : (x^3 + x) / (x^2 + 1) = x := by
  sorry

end NUMINAMATH_CALUDE_same_function_l486_48636


namespace NUMINAMATH_CALUDE_cricket_team_avg_age_l486_48605

-- Define the team and its properties
structure CricketTeam where
  captain_age : ℝ
  wicket_keeper_age : ℝ
  num_bowlers : ℕ
  num_batsmen : ℕ
  team_avg_age : ℝ
  bowlers_avg_age : ℝ
  batsmen_avg_age : ℝ

-- Define the conditions
def team_conditions (team : CricketTeam) : Prop :=
  team.captain_age = 28 ∧
  team.wicket_keeper_age = team.captain_age + 3 ∧
  team.num_bowlers = 5 ∧
  team.num_batsmen = 4 ∧
  team.bowlers_avg_age = team.team_avg_age - 2 ∧
  team.batsmen_avg_age = team.team_avg_age + 3

-- Theorem statement
theorem cricket_team_avg_age (team : CricketTeam) :
  team_conditions team →
  team.team_avg_age = 30.5 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_avg_age_l486_48605


namespace NUMINAMATH_CALUDE_rabbit_weeks_calculation_l486_48693

/-- The number of weeks Julia has had the rabbit -/
def weeks_with_rabbit : ℕ := 2

/-- The total weekly cost for both animals' food -/
def total_weekly_cost : ℕ := 30

/-- The number of weeks Julia has had the parrot -/
def weeks_with_parrot : ℕ := 3

/-- The total spent on food so far -/
def total_spent : ℕ := 114

/-- The weekly cost of rabbit food -/
def weekly_rabbit_cost : ℕ := 12

theorem rabbit_weeks_calculation :
  weeks_with_rabbit * weekly_rabbit_cost + weeks_with_parrot * total_weekly_cost = total_spent :=
sorry

end NUMINAMATH_CALUDE_rabbit_weeks_calculation_l486_48693


namespace NUMINAMATH_CALUDE_fred_baseball_cards_l486_48637

theorem fred_baseball_cards (initial_cards : ℕ) (cards_bought : ℕ) (remaining_cards : ℕ) : 
  initial_cards = 40 → cards_bought = 22 → remaining_cards = initial_cards - cards_bought → 
  remaining_cards = 18 := by
sorry

end NUMINAMATH_CALUDE_fred_baseball_cards_l486_48637


namespace NUMINAMATH_CALUDE_race_participants_race_result_l486_48692

theorem race_participants (group_size : ℕ) (start_position : ℕ) (end_position : ℕ) : ℕ :=
  let total_groups := start_position + end_position - 1
  total_groups * group_size

theorem race_result : race_participants 3 7 5 = 33 := by
  sorry

end NUMINAMATH_CALUDE_race_participants_race_result_l486_48692


namespace NUMINAMATH_CALUDE_function_characterization_l486_48675

def is_valid_function (f : ℝ → ℝ) : Prop :=
  (∀ m n : ℝ, f (m + n) = f m + f n - 6) ∧
  (∃ k : ℕ, 0 < k ∧ k ≤ 5 ∧ f (-1) = k) ∧
  (∀ x : ℝ, x > -1 → f x > 0)

theorem function_characterization (f : ℝ → ℝ) (h : is_valid_function f) :
  ∃ k : ℕ, 0 < k ∧ k ≤ 5 ∧ ∀ x : ℝ, f x = k * x + 6 :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l486_48675


namespace NUMINAMATH_CALUDE_outer_boundary_diameter_is_44_l486_48618

/-- The diameter of the circular fountain in feet. -/
def fountain_diameter : ℝ := 12

/-- The width of the garden ring in feet. -/
def garden_width : ℝ := 10

/-- The width of the walking path in feet. -/
def path_width : ℝ := 6

/-- The diameter of the circle forming the outer boundary of the walking path. -/
def outer_boundary_diameter : ℝ := fountain_diameter + 2 * (garden_width + path_width)

/-- Theorem stating that the diameter of the circle forming the outer boundary of the walking path is 44 feet. -/
theorem outer_boundary_diameter_is_44 : outer_boundary_diameter = 44 := by
  sorry

end NUMINAMATH_CALUDE_outer_boundary_diameter_is_44_l486_48618


namespace NUMINAMATH_CALUDE_collinear_vectors_l486_48671

-- Define the vectors
def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![2, 3]

-- Define the sum vector
def sum_vector : Fin 2 → ℝ := ![3, 5]

-- Define the collinear vector
def collinear_vector : Fin 2 → ℝ := ![6, 10]

-- Theorem statement
theorem collinear_vectors :
  (∃ k : ℝ, ∀ i : Fin 2, collinear_vector i = k * sum_vector i) ∧
  (∀ i : Fin 2, sum_vector i = a i + b i) := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_l486_48671


namespace NUMINAMATH_CALUDE_red_balls_count_l486_48682

theorem red_balls_count (total : ℕ) (prob : ℚ) : 
  total = 15 → 
  prob = 1 / 21 →
  ∃ (red : ℕ), red ≤ total ∧ 
    (red : ℚ) / total * (red - 1) / (total - 1) = prob ∧
    red = 5 :=
by sorry

end NUMINAMATH_CALUDE_red_balls_count_l486_48682


namespace NUMINAMATH_CALUDE_complement_union_theorem_l486_48683

universe u

def U : Set ℕ := {1, 2, 3, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_theorem :
  (U \ M) ∪ N = {3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l486_48683


namespace NUMINAMATH_CALUDE_surveyed_not_population_l486_48608

/-- Represents the total number of students in the seventh grade. -/
def total_students : ℕ := 800

/-- Represents the number of students surveyed. -/
def surveyed_students : ℕ := 200

/-- Represents whether a given number of students constitutes the entire population. -/
def is_population (n : ℕ) : Prop := n = total_students

/-- Theorem stating that the surveyed students do not constitute the entire population. -/
theorem surveyed_not_population : ¬(is_population surveyed_students) := by
  sorry

end NUMINAMATH_CALUDE_surveyed_not_population_l486_48608


namespace NUMINAMATH_CALUDE_x_squared_is_quadratic_x_squared_eq_zero_is_quadratic_l486_48607

/-- A quadratic equation in one variable is of the form ax² + bx + c = 0, where a, b, and c are constants, and a ≠ 0 -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = x² is a quadratic equation in one variable -/
theorem x_squared_is_quadratic : is_quadratic_equation (λ x => x^2) := by
  sorry

/-- The equation x² = 0 is equivalent to the function f(x) = x² -/
theorem x_squared_eq_zero_is_quadratic : is_quadratic_equation (λ x => x^2 - 0) := by
  sorry

end NUMINAMATH_CALUDE_x_squared_is_quadratic_x_squared_eq_zero_is_quadratic_l486_48607


namespace NUMINAMATH_CALUDE_triangle_angle_adjustment_l486_48640

/-- 
Given a triangle with interior angles in a 3:4:9 ratio, prove that if the largest angle is 
decreased by x degrees such that the smallest angle doubles its initial value while 
maintaining the sum of angles as 180 degrees, then x = 33.75 degrees.
-/
theorem triangle_angle_adjustment (k : ℝ) (x : ℝ) 
  (h1 : 3*k + 4*k + 9*k = 180)  -- Sum of initial angles is 180 degrees
  (h2 : 3*k + 4*k + (9*k - x) = 180)  -- Sum of angles after adjustment is 180 degrees
  (h3 : 2*(3*k) = 3*k + 4*k)  -- Smallest angle doubles its initial value
  : x = 33.75 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_adjustment_l486_48640


namespace NUMINAMATH_CALUDE_sherman_weekly_driving_time_l486_48647

-- Define the daily commute time in minutes
def daily_commute : ℕ := 30 + 30

-- Define the number of workdays in a week
def workdays : ℕ := 5

-- Define the weekend driving time in hours
def weekend_driving : ℕ := 2 * 2

-- Theorem statement
theorem sherman_weekly_driving_time :
  (workdays * daily_commute) / 60 + weekend_driving = 9 := by
  sorry

end NUMINAMATH_CALUDE_sherman_weekly_driving_time_l486_48647


namespace NUMINAMATH_CALUDE_possible_signs_l486_48694

theorem possible_signs (a b c : ℝ) : 
  a + b + c = 0 → 
  abs a > abs b → 
  abs b > abs c → 
  ∃ (a' b' c' : ℝ), a' + b' + c' = 0 ∧ 
                     abs a' > abs b' ∧ 
                     abs b' > abs c' ∧ 
                     c' > 0 ∧ 
                     a' < 0 :=
sorry

end NUMINAMATH_CALUDE_possible_signs_l486_48694


namespace NUMINAMATH_CALUDE_inequality_proof_l486_48666

theorem inequality_proof (r p q : ℝ) (hr : r > 0) (hp : p > 0) (hq : q > 0) (h : p^2 * r > q^2 * r) :
  1 > -q/p := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l486_48666


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_a_l486_48622

/-- Given a hyperbola with equation x²/a² - y²/9 = 1 where a > 0,
    if the asymptote equations are 3x ± 2y = 0, then a = 2 -/
theorem hyperbola_asymptote_a (a : ℝ) (h1 : a > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / 9 = 1 →
    (3 * x + 2 * y = 0 ∨ 3 * x - 2 * y = 0)) →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_a_l486_48622


namespace NUMINAMATH_CALUDE_ratio_equality_l486_48615

theorem ratio_equality (a b : ℚ) (h : a / b = 7 / 6) : 6 * a = 7 * b := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l486_48615


namespace NUMINAMATH_CALUDE_searchlight_dark_period_l486_48633

/-- Given a searchlight that makes 3 revolutions per minute, 
    prove that if the probability of staying in the dark is 0.75, 
    then the duration of the dark period is 15 seconds. -/
theorem searchlight_dark_period 
  (revolutions_per_minute : ℝ) 
  (probability_dark : ℝ) 
  (h1 : revolutions_per_minute = 3) 
  (h2 : probability_dark = 0.75) : 
  (probability_dark * (60 / revolutions_per_minute)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_searchlight_dark_period_l486_48633


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l486_48620

theorem min_value_sum_squares (x y z : ℝ) (h : x + 2*y + 3*z = 1) :
  ∃ (m : ℝ), (∀ a b c : ℝ, a + 2*b + 3*c = 1 → a^2 + b^2 + c^2 ≥ m) ∧
             (x^2 + y^2 + z^2 = m) ∧
             (m = 1/14) := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l486_48620


namespace NUMINAMATH_CALUDE_positive_solution_condition_l486_48602

theorem positive_solution_condition (a b : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧
    x₁ - x₂ = a ∧ x₃ - x₄ = b ∧ x₁ + x₂ + x₃ + x₄ = 1) ↔
  abs a + abs b < 1 :=
by sorry

end NUMINAMATH_CALUDE_positive_solution_condition_l486_48602


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l486_48660

theorem least_subtraction_for_divisibility : 
  ∃! x : ℕ, x ≤ 953 ∧ (218791 - x) % 953 = 0 ∧ ∀ y : ℕ, y < x → (218791 - y) % 953 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l486_48660


namespace NUMINAMATH_CALUDE_solution_set_of_f_greater_than_4_range_of_a_l486_48663

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 3| + |x - 1|

-- Statement 1
theorem solution_set_of_f_greater_than_4 :
  {x : ℝ | f x > 4} = Set.Ioi (-2) ∪ Set.Ioi 0 :=
sorry

-- Statement 2
theorem range_of_a (a : ℝ) :
  (∃ x ∈ Set.Icc (-3/2) 1, a + 1 > f x) → a ∈ Set.Ioi (3/2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_f_greater_than_4_range_of_a_l486_48663


namespace NUMINAMATH_CALUDE_solution_set_a_neg_one_range_of_a_for_nonnegative_f_l486_48654

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 2 * x

-- Part 1: Solution set when a = -1
theorem solution_set_a_neg_one :
  {x : ℝ | f (-1) x ≤ 0} = {x : ℝ | x ≤ -1/3} := by sorry

-- Part 2: Range of a for f(x) ≥ 0 when x ≥ -1
theorem range_of_a_for_nonnegative_f :
  ∀ a : ℝ, (∀ x : ℝ, x ≥ -1 → f a x ≥ 0) ↔ (a ≤ -3 ∨ a ≥ 1) := by sorry

end NUMINAMATH_CALUDE_solution_set_a_neg_one_range_of_a_for_nonnegative_f_l486_48654


namespace NUMINAMATH_CALUDE_total_length_S_l486_48688

-- Define the set S
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
    ((|x| - 2)^2 + (|y| - 2)^2)^(1/2) = 2 - |1 - ((|x| - 2)^2 + (|y| - 2)^2)^(1/2)|}

-- Define the length function for S
noncomputable def length_S : ℝ := sorry

-- Theorem statement
theorem total_length_S : length_S = 20 * Real.pi := by sorry

end NUMINAMATH_CALUDE_total_length_S_l486_48688


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_sum_l486_48612

theorem lcm_of_ratio_and_sum (a b : ℕ+) : 
  (a : ℚ) / b = 2 / 3 → a + b = 40 → Nat.lcm a b = 24 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_sum_l486_48612


namespace NUMINAMATH_CALUDE_milk_addition_rate_l486_48609

/-- Calculates the rate of milk addition given initial conditions --/
theorem milk_addition_rate
  (initial_milk : ℝ)
  (pump_rate : ℝ)
  (pump_time : ℝ)
  (addition_time : ℝ)
  (final_milk : ℝ)
  (h1 : initial_milk = 30000)
  (h2 : pump_rate = 2880)
  (h3 : pump_time = 4)
  (h4 : addition_time = 7)
  (h5 : final_milk = 28980) :
  let milk_pumped := pump_rate * pump_time
  let milk_before_addition := initial_milk - milk_pumped
  let milk_added := final_milk - milk_before_addition
  milk_added / addition_time = 1500 := by
  sorry

end NUMINAMATH_CALUDE_milk_addition_rate_l486_48609


namespace NUMINAMATH_CALUDE_choose_three_from_eight_l486_48648

theorem choose_three_from_eight : Nat.choose 8 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_eight_l486_48648


namespace NUMINAMATH_CALUDE_triangle_equal_area_l486_48656

/-- Given two triangles FGH and IJK with the specified properties, prove that JK = 10 -/
theorem triangle_equal_area (FG FH IJ IK : ℝ) (angle_GFH angle_IJK : ℝ) :
  FG = 5 →
  FH = 4 →
  angle_GFH = 30 * π / 180 →
  IJ = 2 →
  IK = 6 →
  angle_IJK = 30 * π / 180 →
  angle_GFH = angle_IJK →
  (1/2 * FG * FH * Real.sin angle_GFH) = (1/2 * IJ * 10 * Real.sin angle_IJK) →
  ∃ (JK : ℝ), JK = 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_equal_area_l486_48656


namespace NUMINAMATH_CALUDE_increasing_function_condition_l486_48690

theorem increasing_function_condition (f : ℝ → ℝ) (h : Monotone f) :
  ∀ a b : ℝ, (a + b > 0 ↔ f a + f b > f (-a) + f (-b)) :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_condition_l486_48690


namespace NUMINAMATH_CALUDE_homer_investment_interest_l486_48680

/-- Calculates the interest earned on an investment with annual compounding -/
def interest_earned (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

/-- Proves that the interest earned on a $2000 investment at 2% for 3 years is $122.416 -/
theorem homer_investment_interest :
  let principal : ℝ := 2000
  let rate : ℝ := 0.02
  let time : ℕ := 3
  abs (interest_earned principal rate time - 122.416) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_homer_investment_interest_l486_48680


namespace NUMINAMATH_CALUDE_fiftieth_term_is_346_l486_48669

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem fiftieth_term_is_346 : 
  arithmetic_sequence 3 7 50 = 346 := by
sorry

end NUMINAMATH_CALUDE_fiftieth_term_is_346_l486_48669


namespace NUMINAMATH_CALUDE_f_max_min_l486_48684

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + x + 1

-- Define the domain
def domain : Set ℝ := { x | 0 ≤ x ∧ x ≤ 3/2 }

theorem f_max_min :
  ∃ (max min : ℝ),
    (∀ x ∈ domain, f x ≤ max) ∧
    (∃ x ∈ domain, f x = max) ∧
    (∀ x ∈ domain, min ≤ f x) ∧
    (∃ x ∈ domain, f x = min) ∧
    max = 5/4 ∧
    min = 1/4 := by sorry

end NUMINAMATH_CALUDE_f_max_min_l486_48684


namespace NUMINAMATH_CALUDE_inequality_transformation_l486_48687

theorem inequality_transformation (a b c : ℝ) (hc : c ≠ 0) :
  a * c^2 > b * c^2 → a > b := by sorry

end NUMINAMATH_CALUDE_inequality_transformation_l486_48687


namespace NUMINAMATH_CALUDE_coin_division_l486_48631

theorem coin_division (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < n → (m % 8 ≠ 6 ∨ m % 9 ≠ 7)) → 
  n % 8 = 6 → 
  n % 9 = 7 → 
  n % 11 = 8 := by
sorry

end NUMINAMATH_CALUDE_coin_division_l486_48631


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l486_48604

/-- The number of ways to choose 7 starters from a volleyball team -/
def volleyball_starters_count : ℕ := 2376

/-- The total number of players in the team -/
def total_players : ℕ := 15

/-- The number of triplets in the team -/
def triplets_count : ℕ := 3

/-- The number of starters to be chosen -/
def starters_count : ℕ := 7

/-- The number of triplets that must be in the starting lineup -/
def required_triplets : ℕ := 2

theorem volleyball_team_selection :
  volleyball_starters_count = 
    (Nat.choose triplets_count required_triplets) * 
    (Nat.choose (total_players - triplets_count) (starters_count - required_triplets)) := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_l486_48604


namespace NUMINAMATH_CALUDE_set_representation_l486_48651

theorem set_representation :
  {x : ℕ | 8 < x ∧ x < 12} = {9, 10, 11} := by
  sorry

end NUMINAMATH_CALUDE_set_representation_l486_48651


namespace NUMINAMATH_CALUDE_collins_total_petals_l486_48643

/-- The number of petals Collin has after receiving flowers from Ingrid -/
theorem collins_total_petals (collins_initial_flowers ingrid_flowers petals_per_flower : ℕ) : 
  collins_initial_flowers = 25 →
  ingrid_flowers = 33 →
  petals_per_flower = 4 →
  (collins_initial_flowers + ingrid_flowers / 3) * petals_per_flower = 144 := by
  sorry

#check collins_total_petals

end NUMINAMATH_CALUDE_collins_total_petals_l486_48643


namespace NUMINAMATH_CALUDE_inequality_preservation_l486_48685

theorem inequality_preservation (a b c : ℝ) (h : a > b) (h' : b > 0) :
  a + c > b + c := by
sorry

end NUMINAMATH_CALUDE_inequality_preservation_l486_48685


namespace NUMINAMATH_CALUDE_santinos_mango_trees_l486_48652

theorem santinos_mango_trees :
  let papaya_trees : ℕ := 2
  let papayas_per_tree : ℕ := 10
  let mangos_per_tree : ℕ := 20
  let total_fruits : ℕ := 80
  ∃ mango_trees : ℕ,
    papaya_trees * papayas_per_tree + mango_trees * mangos_per_tree = total_fruits ∧
    mango_trees = 3 :=
by sorry

end NUMINAMATH_CALUDE_santinos_mango_trees_l486_48652


namespace NUMINAMATH_CALUDE_tank_volume_ratio_l486_48665

theorem tank_volume_ratio :
  ∀ (V₁ V₂ : ℝ), V₁ > 0 → V₂ > 0 →
  (3/4 : ℝ) * V₁ = (5/8 : ℝ) * V₂ →
  V₁ / V₂ = 5/6 := by
sorry

end NUMINAMATH_CALUDE_tank_volume_ratio_l486_48665


namespace NUMINAMATH_CALUDE_sqrt_20_in_terms_of_a_and_b_l486_48659

theorem sqrt_20_in_terms_of_a_and_b (a b : ℝ) (ha : a = Real.sqrt 2) (hb : b = Real.sqrt 10) :
  Real.sqrt 20 = a * b := by
  sorry

end NUMINAMATH_CALUDE_sqrt_20_in_terms_of_a_and_b_l486_48659


namespace NUMINAMATH_CALUDE_collinear_points_condition_l486_48657

/-- Given non-collinear plane vectors a and b, and points A, B, C such that
    AB = a - 2b and BC = 3a + kb, prove that A, B, and C are collinear iff k = -6 -/
theorem collinear_points_condition (a b : ℝ × ℝ) (k : ℝ) 
  (h_non_collinear : ¬ ∃ (r : ℝ), a = r • b) 
  (A B C : ℝ × ℝ) 
  (h_AB : B - A = a - 2 • b) 
  (h_BC : C - B = 3 • a + k • b) :
  (∃ (t : ℝ), C - A = t • (B - A)) ↔ k = -6 :=
sorry

end NUMINAMATH_CALUDE_collinear_points_condition_l486_48657


namespace NUMINAMATH_CALUDE_cost_price_for_given_profit_l486_48646

/-- Given a profit percentage, calculates the cost price as a percentage of the selling price -/
def cost_price_percentage (profit_percentage : Real) : Real :=
  100 - profit_percentage

/-- Theorem stating that when the profit percentage is 4.166666666666666%,
    the cost price is 95.83333333333334% of the selling price -/
theorem cost_price_for_given_profit :
  cost_price_percentage 4.166666666666666 = 95.83333333333334 := by
  sorry

#eval cost_price_percentage 4.166666666666666

end NUMINAMATH_CALUDE_cost_price_for_given_profit_l486_48646


namespace NUMINAMATH_CALUDE_ellipse_intersection_dot_product_l486_48691

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the foci of the ellipse
def focus_1 : ℝ × ℝ := (1, 0)
def focus_2 : ℝ × ℝ := (-1, 0)

-- Define a line passing through a focus at 45°
def line_through_focus (f : ℝ × ℝ) (x y : ℝ) : Prop :=
  y - f.2 = (x - f.1)

-- Define the intersection points
def intersection_points (f : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p | is_on_ellipse p.1 p.2 ∧ line_through_focus f p.1 p.2}

-- Theorem statement
theorem ellipse_intersection_dot_product :
  ∀ (f : ℝ × ℝ) (A B : ℝ × ℝ),
    (f = focus_1 ∨ f = focus_2) →
    A ∈ intersection_points f →
    B ∈ intersection_points f →
    A ≠ B →
    A.1 * B.1 + A.2 * B.2 = -1/3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_dot_product_l486_48691


namespace NUMINAMATH_CALUDE_manuscript_revision_l486_48673

/-- Proves that the number of pages revised twice is 15, given the manuscript typing conditions --/
theorem manuscript_revision (total_pages : ℕ) (revised_once : ℕ) (total_cost : ℕ) 
  (first_typing_cost : ℕ) (revision_cost : ℕ) :
  total_pages = 100 →
  revised_once = 35 →
  total_cost = 860 →
  first_typing_cost = 6 →
  revision_cost = 4 →
  ∃ (revised_twice : ℕ),
    revised_twice = 15 ∧
    total_cost = (total_pages - revised_once - revised_twice) * first_typing_cost +
                 revised_once * (first_typing_cost + revision_cost) +
                 revised_twice * (first_typing_cost + 2 * revision_cost) :=
by sorry


end NUMINAMATH_CALUDE_manuscript_revision_l486_48673


namespace NUMINAMATH_CALUDE_num_perfect_square_factors_is_440_l486_48662

/-- The number of positive perfect square factors of (2^14)(3^9)(5^20) -/
def num_perfect_square_factors : ℕ :=
  (Finset.range 8).card * (Finset.range 5).card * (Finset.range 11).card

/-- Theorem stating that the number of positive perfect square factors of (2^14)(3^9)(5^20) is 440 -/
theorem num_perfect_square_factors_is_440 : num_perfect_square_factors = 440 := by
  sorry

end NUMINAMATH_CALUDE_num_perfect_square_factors_is_440_l486_48662


namespace NUMINAMATH_CALUDE_copper_alloy_percentage_l486_48677

/-- Proves that the percentage of copper in the alloy that we need 32 kg of is 43.75% --/
theorem copper_alloy_percentage :
  ∀ (x : ℝ),
  -- Total mass of the final alloy
  let total_mass : ℝ := 40
  -- Percentage of copper in the final alloy
  let final_copper_percentage : ℝ := 45
  -- Mass of the alloy with unknown copper percentage
  let mass_unknown : ℝ := 32
  -- Mass of the alloy with 50% copper
  let mass_known : ℝ := 8
  -- Percentage of copper in the known alloy
  let known_copper_percentage : ℝ := 50
  -- The equation representing the mixture of alloys
  (mass_unknown * x / 100 + mass_known * known_copper_percentage / 100 = total_mass * final_copper_percentage / 100) →
  x = 43.75 := by
sorry

end NUMINAMATH_CALUDE_copper_alloy_percentage_l486_48677


namespace NUMINAMATH_CALUDE_line_intersects_y_axis_l486_48664

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The y-intercept of a line -/
def y_intercept (l : Line) : ℝ × ℝ := sorry

/-- The theorem stating that the line passing through (2, 9) and (5, 21) 
    intersects the y-axis at (0, 1) -/
theorem line_intersects_y_axis : 
  let l : Line := { x₁ := 2, y₁ := 9, x₂ := 5, y₂ := 21 }
  y_intercept l = (0, 1) := by sorry

end NUMINAMATH_CALUDE_line_intersects_y_axis_l486_48664


namespace NUMINAMATH_CALUDE_fibonacci_arithmetic_sequence_l486_48653

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fibonacci_arithmetic_sequence (n : ℕ) :
  n > 0 ∧ 
  n + (n + 3) + (n + 4) = 3000 ∧ 
  (fib n < fib (n + 3) ∧ fib (n + 3) < fib (n + 4)) ∧
  (fib (n + 4) - fib (n + 3) = fib (n + 3) - fib n) →
  n = 997 := by
sorry

end NUMINAMATH_CALUDE_fibonacci_arithmetic_sequence_l486_48653


namespace NUMINAMATH_CALUDE_inequality_range_l486_48641

theorem inequality_range (a : ℝ) : 
  (∀ x > 1, x + 1 / (x - 1) ≥ a) → a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l486_48641


namespace NUMINAMATH_CALUDE_three_digit_palindrome_average_l486_48681

/-- Reverses the digits of a three-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 100 + ((n / 10) % 10) * 10 + (n / 100)

/-- Checks if a three-digit number is a palindrome -/
def is_palindrome (n : ℕ) : Prop :=
  n % 10 = n / 100

theorem three_digit_palindrome_average (m n : ℕ) : 
  100 ≤ m ∧ m < 1000 ∧
  100 ≤ n ∧ n < 1000 ∧
  is_palindrome m ∧
  (m + n) / 2 = reverse_digits m ∧
  m = 161 ∧ n = 161 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_palindrome_average_l486_48681


namespace NUMINAMATH_CALUDE_complex_equality_l486_48670

theorem complex_equality (z : ℂ) : z = -15/8 + 5/4*I → Complex.abs (z - 2*I) = Complex.abs (z + 4) ∧ Complex.abs (z - 2*I) = Complex.abs (z + I) := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l486_48670


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l486_48638

/-- An isosceles triangle with two sides measuring 5 and 6 -/
structure IsoscelesTriangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (is_isosceles : side1 = side2 ∨ side1 = 6 ∨ side2 = 6)
  (has_sides_5_6 : (side1 = 5 ∧ side2 = 6) ∨ (side1 = 6 ∧ side2 = 5))

/-- The perimeter of an isosceles triangle with sides 5 and 6 is either 16 or 17 -/
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) :
  ∃ (p : ℝ), (p = 16 ∨ p = 17) ∧ p = t.side1 + t.side2 + (if t.side1 = t.side2 then 5 else 6) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l486_48638


namespace NUMINAMATH_CALUDE_average_stickers_per_pack_l486_48610

def sticker_counts : List ℕ := [5, 7, 9, 9, 11, 15, 15, 17, 19, 21]

def total_stickers : ℕ := sticker_counts.sum

def num_packs : ℕ := sticker_counts.length

theorem average_stickers_per_pack :
  (total_stickers : ℚ) / num_packs = 12.8 := by
  sorry

end NUMINAMATH_CALUDE_average_stickers_per_pack_l486_48610


namespace NUMINAMATH_CALUDE_relationship_theorem_l486_48623

theorem relationship_theorem (x y z w : ℝ) :
  (x + y) / (y + z) = (z + w) / (w + x) →
  x = z ∨ x + y + w + z = 0 :=
by sorry

end NUMINAMATH_CALUDE_relationship_theorem_l486_48623


namespace NUMINAMATH_CALUDE_father_age_l486_48624

/-- Represents the ages and relationships of family members -/
structure FamilyAges where
  peter : ℕ
  jane : ℕ
  harriet : ℕ
  emily : ℕ
  mother : ℕ
  aunt_lucy : ℕ
  father : ℕ

/-- The conditions given in the problem -/
def family_conditions (f : FamilyAges) : Prop :=
  f.peter + 12 = 2 * (f.harriet + 12) ∧
  f.jane = f.emily + 10 ∧
  3 * f.peter = f.mother ∧
  f.mother = 60 ∧
  f.peter = f.jane + 5 ∧
  f.aunt_lucy = 52 ∧
  f.aunt_lucy = f.mother + 4 ∧
  f.father = f.aunt_lucy + 20

/-- The theorem to be proved -/
theorem father_age (f : FamilyAges) : 
  family_conditions f → f.father = 72 := by
  sorry

end NUMINAMATH_CALUDE_father_age_l486_48624


namespace NUMINAMATH_CALUDE_weekly_card_pack_size_l486_48619

theorem weekly_card_pack_size (total_weeks : ℕ) (remaining_cards : ℕ) : 
  total_weeks = 52 →
  remaining_cards = 520 →
  (remaining_cards * 2) / total_weeks = 20 :=
by sorry

end NUMINAMATH_CALUDE_weekly_card_pack_size_l486_48619


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l486_48639

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 6 = 30 → a 3 + a 9 = 60 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l486_48639


namespace NUMINAMATH_CALUDE_set_inclusion_equivalence_l486_48601

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 ≤ 5/4}

def B (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | |p.1 - 1| + 2*|p.2 - 2| ≤ a}

-- State the theorem
theorem set_inclusion_equivalence (a : ℝ) : A ⊆ B a ↔ a ≥ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_set_inclusion_equivalence_l486_48601


namespace NUMINAMATH_CALUDE_store_profit_ratio_l486_48606

/-- Represents the cost and sales information for a product. -/
structure Product where
  cost : ℝ
  markup : ℝ
  salesRatio : ℝ

/-- Represents the store's product lineup. -/
structure Store where
  peachSlices : Product
  riceCrispyTreats : Product
  sesameSnacks : Product

theorem store_profit_ratio (s : Store) : 
  s.peachSlices.cost = 2 * s.sesameSnacks.cost ∧
  s.peachSlices.markup = 0.2 ∧
  s.riceCrispyTreats.markup = 0.3 ∧
  s.sesameSnacks.markup = 0.2 ∧
  s.peachSlices.salesRatio = 1 ∧
  s.riceCrispyTreats.salesRatio = 3 ∧
  s.sesameSnacks.salesRatio = 2 ∧
  (s.peachSlices.markup * s.peachSlices.cost * s.peachSlices.salesRatio +
   s.riceCrispyTreats.markup * s.riceCrispyTreats.cost * s.riceCrispyTreats.salesRatio +
   s.sesameSnacks.markup * s.sesameSnacks.cost * s.sesameSnacks.salesRatio) = 
  0.25 * (s.peachSlices.cost * s.peachSlices.salesRatio +
          s.riceCrispyTreats.cost * s.riceCrispyTreats.salesRatio +
          s.sesameSnacks.cost * s.sesameSnacks.salesRatio) →
  s.riceCrispyTreats.cost / s.sesameSnacks.cost = 4 / 3 := by
sorry


end NUMINAMATH_CALUDE_store_profit_ratio_l486_48606


namespace NUMINAMATH_CALUDE_square_sum_equals_two_l486_48695

theorem square_sum_equals_two (x y : ℝ) 
  (h1 : x - y = -1) 
  (h2 : x * y = 1/2) : 
  x^2 + y^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_two_l486_48695


namespace NUMINAMATH_CALUDE_gardening_project_total_cost_l486_48644

def gardening_project_cost (rose_bushes : ℕ) (rose_bush_cost : ℕ) 
  (gardener_hourly_rate : ℕ) (hours_per_day : ℕ) (days_worked : ℕ)
  (soil_volume : ℕ) (soil_cost_per_unit : ℕ) : ℕ :=
  rose_bushes * rose_bush_cost + 
  gardener_hourly_rate * hours_per_day * days_worked +
  soil_volume * soil_cost_per_unit

theorem gardening_project_total_cost : 
  gardening_project_cost 20 150 30 5 4 100 5 = 4100 := by
  sorry

end NUMINAMATH_CALUDE_gardening_project_total_cost_l486_48644


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l486_48658

theorem necessary_but_not_sufficient_condition 
  (A B C : Set α) 
  (h_nonempty_A : A.Nonempty) 
  (h_nonempty_B : B.Nonempty) 
  (h_nonempty_C : C.Nonempty)
  (h_union : A ∪ B = C) 
  (h_not_subset : ¬(B ⊆ A)) :
  (∀ x, x ∈ A → x ∈ C) ∧ (∃ x, x ∈ C ∧ x ∉ A) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l486_48658


namespace NUMINAMATH_CALUDE_probability_of_prime_is_two_fifths_l486_48674

/-- A function that determines if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- The set of numbers from 1 to 10 -/
def numberSet : Finset ℕ := sorry

/-- The set of prime numbers in the numberSet -/
def primeSet : Finset ℕ := sorry

/-- The probability of selecting a prime number from the numberSet -/
def probabilityOfPrime : ℚ := sorry

theorem probability_of_prime_is_two_fifths : 
  probabilityOfPrime = 2 / 5 := sorry

end NUMINAMATH_CALUDE_probability_of_prime_is_two_fifths_l486_48674


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l486_48679

/-- Given a right triangle with sides 5, 12, and 13, x is the side length of a square
    inscribed with one vertex at the right angle, and y is the side length of a square
    inscribed with one side on the longest leg (12). -/
def triangle_with_squares (x y : ℝ) : Prop :=
  -- Right triangle condition
  5^2 + 12^2 = 13^2 ∧
  -- Condition for square with side x
  x / 5 = x / 12 ∧
  -- Condition for square with side y
  y + y = 12

/-- The ratio of the side lengths of the two inscribed squares is 10/17. -/
theorem inscribed_squares_ratio :
  ∀ x y : ℝ, triangle_with_squares x y → x / y = 10 / 17 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l486_48679


namespace NUMINAMATH_CALUDE_triangle_area_l486_48696

/-- A triangle with integral sides and perimeter 12 has area 6 -/
theorem triangle_area (a b c : ℕ) : 
  a + b + c = 12 → 
  a + b > c → b + c > a → a + c > b → 
  (a : ℝ) * (b : ℝ) / 2 = 6 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_l486_48696


namespace NUMINAMATH_CALUDE_valentino_farm_birds_l486_48668

/-- The number of birds on Mr. Valentino's farm -/
def total_birds (chickens ducks turkeys : ℕ) : ℕ := chickens + ducks + turkeys

/-- Theorem stating the total number of birds on Mr. Valentino's farm -/
theorem valentino_farm_birds :
  ∃ (chickens ducks turkeys : ℕ),
    chickens = 200 ∧
    ducks = 2 * chickens ∧
    turkeys = 3 * ducks ∧
    total_birds chickens ducks turkeys = 1800 :=
by sorry

end NUMINAMATH_CALUDE_valentino_farm_birds_l486_48668


namespace NUMINAMATH_CALUDE_mod_power_difference_l486_48645

theorem mod_power_difference (n : ℕ) : 35^1723 - 16^1723 ≡ 1 [ZMOD 6] := by sorry

end NUMINAMATH_CALUDE_mod_power_difference_l486_48645


namespace NUMINAMATH_CALUDE_polynomial_factorization_l486_48625

theorem polynomial_factorization (a x : ℝ) : -a*x^2 + 2*a*x - a = -a*(x-1)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l486_48625


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_coefficients_l486_48634

theorem quadratic_roots_imply_coefficients (a b : ℝ) :
  (∀ x : ℝ, x^2 + (a + 1)*x + a*b = 0 ↔ x = -1 ∨ x = 4) →
  a = -4 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_coefficients_l486_48634


namespace NUMINAMATH_CALUDE_area_of_triangle_ABC_l486_48697

-- Define the points in the plane
variable (A B C D : ℝ × ℝ)

-- Define the distances
def AC : ℝ := 15
def AB : ℝ := 17
def DC : ℝ := 9

-- Define the angle D as a right angle
def angle_D_is_right : Prop := sorry

-- Define that the points are coplanar
def points_are_coplanar : Prop := sorry

-- Define the area of triangle ABC
def area_ABC : ℝ := sorry

-- Theorem statement
theorem area_of_triangle_ABC :
  points_are_coplanar →
  angle_D_is_right →
  area_ABC = 54 + 6 * Real.sqrt 145 :=
sorry

end NUMINAMATH_CALUDE_area_of_triangle_ABC_l486_48697


namespace NUMINAMATH_CALUDE_circle_radius_l486_48649

/-- The radius of the circle defined by x^2 + y^2 - 8x = 0 is 4 -/
theorem circle_radius (x y : ℝ) : (x^2 + y^2 - 8*x = 0) → ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 4^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l486_48649


namespace NUMINAMATH_CALUDE_min_value_sum_squares_and_reciprocal_cube_min_value_achievable_l486_48611

theorem min_value_sum_squares_and_reciprocal_cube (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a^2 + b^2 + c^2 + 1 / (a + b + c)^3 ≥ (1/12)^(1/3) := by
  sorry

theorem min_value_achievable (ε : ℝ) (hε : ε > 0) : 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a^2 + b^2 + c^2 + 1 / (a + b + c)^3 < (1/12)^(1/3) + ε := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_and_reciprocal_cube_min_value_achievable_l486_48611


namespace NUMINAMATH_CALUDE_binomial_expansion_problem_l486_48627

theorem binomial_expansion_problem (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (3 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃ + a₅)^2 = 3125 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_problem_l486_48627


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l486_48617

theorem consecutive_odd_integers_sum (n : ℤ) : 
  (∃ (a b c : ℤ), 
    (a = n - 2 ∧ b = n ∧ c = n + 2) ∧  -- Three consecutive odd integers
    (Odd a ∧ Odd b ∧ Odd c) ∧           -- All are odd
    (a + c = 152)) →                    -- Sum of first and third is 152
  n = 76 :=                             -- Second integer is 76
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l486_48617
