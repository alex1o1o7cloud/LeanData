import Mathlib

namespace NUMINAMATH_CALUDE_hiking_trip_up_rate_l100_10068

/-- Represents the hiking trip parameters -/
structure HikingTrip where
  upRate : ℝ  -- Rate of ascent in miles per day
  downRate : ℝ  -- Rate of descent in miles per day
  upTime : ℝ  -- Time taken for ascent in days
  downTime : ℝ  -- Time taken for descent in days
  downDistance : ℝ  -- Distance of the descent route in miles

/-- The hiking trip satisfies the given conditions -/
def validHikingTrip (trip : HikingTrip) : Prop :=
  trip.upTime = trip.downTime ∧  -- Same time for each route
  trip.downRate = 1.5 * trip.upRate ∧  -- Down rate is 1.5 times up rate
  trip.upTime = 2 ∧  -- 2 days to go up
  trip.downDistance = 9  -- 9 miles down

theorem hiking_trip_up_rate (trip : HikingTrip) 
  (h : validHikingTrip trip) : trip.upRate = 3 := by
  sorry

end NUMINAMATH_CALUDE_hiking_trip_up_rate_l100_10068


namespace NUMINAMATH_CALUDE_f_derivative_l100_10035

def binomial (n k : ℕ) : ℕ := Nat.choose n k

def f (x : ℝ) : ℝ :=
  binomial 4 0 - binomial 4 1 * x + binomial 4 2 * x^2 - binomial 4 3 * x^3 + binomial 4 4 * x^4

theorem f_derivative (x : ℝ) : deriv f x = 4 * (-1 + x)^3 := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_l100_10035


namespace NUMINAMATH_CALUDE_operation_problem_l100_10040

-- Define the type for our operations
inductive Operation
| Add
| Sub
| Mul
| Div

-- Define the function to apply an operation
def apply (op : Operation) (a b : ℚ) : ℚ :=
  match op with
  | Operation.Add => a + b
  | Operation.Sub => a - b
  | Operation.Mul => a * b
  | Operation.Div => a / b

-- State the theorem
theorem operation_problem (diamond circ : Operation) 
    (h : (apply diamond 15 3) / (apply circ 8 2) = 3) :
  (apply diamond 9 4) / (apply circ 14 7) = 13 / 7 := by
  sorry

end NUMINAMATH_CALUDE_operation_problem_l100_10040


namespace NUMINAMATH_CALUDE_geometric_progression_product_l100_10038

/-- For a geometric progression with n terms, first term a, and common ratio r,
    where P is the product of the n terms and T is the sum of the squares of the terms,
    the following equation holds. -/
theorem geometric_progression_product (n : ℕ) (a r : ℝ) (P T : ℝ) 
    (h1 : P = a^n * r^(n * (n - 1) / 2))
    (h2 : T = a^2 * (1 - r^(2*n)) / (1 - r^2)) 
    (h3 : r ≠ 1) : 
  P = T^(n/2) * ((1 - r^2) / (1 - r^(2*n)))^(n/2) * r^(n*(n-1)/2) := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_product_l100_10038


namespace NUMINAMATH_CALUDE_calculate_total_earnings_l100_10003

/-- Represents the number of days it takes for a person to complete the job alone. -/
structure WorkRate where
  days : ℕ
  days_pos : days > 0

/-- Represents the daily work rate of a person. -/
def daily_rate (w : WorkRate) : ℚ := 1 / w.days

/-- Calculates the total daily rate when multiple people work together. -/
def total_daily_rate (rates : List ℚ) : ℚ := rates.sum

/-- Represents the earnings of the workers. -/
structure Earnings where
  total : ℚ
  total_pos : total > 0

/-- Main theorem: Given the work rates and b's earnings, prove the total earnings. -/
theorem calculate_total_earnings
  (a b c : WorkRate)
  (h_a : a.days = 6)
  (h_b : b.days = 8)
  (h_c : c.days = 12)
  (b_earnings : ℚ)
  (h_b_earnings : b_earnings = 390)
  : ∃ (e : Earnings), e.total = 1170 := by
  sorry

end NUMINAMATH_CALUDE_calculate_total_earnings_l100_10003


namespace NUMINAMATH_CALUDE_fraction_of_sales_for_ingredients_l100_10012

/-- Proves that the fraction of sales used to buy ingredients is 3/5 -/
theorem fraction_of_sales_for_ingredients
  (num_pies : ℕ)
  (price_per_pie : ℚ)
  (amount_remaining : ℚ)
  (h1 : num_pies = 200)
  (h2 : price_per_pie = 20)
  (h3 : amount_remaining = 1600) :
  (num_pies * price_per_pie - amount_remaining) / (num_pies * price_per_pie) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_sales_for_ingredients_l100_10012


namespace NUMINAMATH_CALUDE_diary_pieces_not_complete_l100_10024

theorem diary_pieces_not_complete : ¬∃ (n : ℕ), 4^n = 50 := by
  sorry

end NUMINAMATH_CALUDE_diary_pieces_not_complete_l100_10024


namespace NUMINAMATH_CALUDE_analytic_method_characterization_l100_10064

/-- Enumeration of proof methods --/
inductive ProofMethod
  | MathematicalInduction
  | ProofByContradiction
  | AnalyticMethod
  | SyntheticMethod

/-- Characteristic of a proof method --/
def isCharacterizedBy (m : ProofMethod) (c : String) : Prop :=
  match m with
  | ProofMethod.AnalyticMethod => c = "seeking the cause from the effect"
  | _ => c ≠ "seeking the cause from the effect"

/-- Theorem stating that the Analytic Method is characterized by "seeking the cause from the effect" --/
theorem analytic_method_characterization :
  isCharacterizedBy ProofMethod.AnalyticMethod "seeking the cause from the effect" :=
by sorry

end NUMINAMATH_CALUDE_analytic_method_characterization_l100_10064


namespace NUMINAMATH_CALUDE_equal_days_count_l100_10066

/-- Represents the days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Counts the number of Tuesdays and Thursdays in a 30-day month starting on the given day -/
def countTuesdaysAndThursdays (startDay : DayOfWeek) : Nat × Nat :=
  let rec count (currentDay : DayOfWeek) (daysLeft : Nat) (tuesdays : Nat) (thursdays : Nat) : Nat × Nat :=
    if daysLeft = 0 then
      (tuesdays, thursdays)
    else
      match currentDay with
      | DayOfWeek.Tuesday => count (nextDay currentDay) (daysLeft - 1) (tuesdays + 1) thursdays
      | DayOfWeek.Thursday => count (nextDay currentDay) (daysLeft - 1) tuesdays (thursdays + 1)
      | _ => count (nextDay currentDay) (daysLeft - 1) tuesdays thursdays
  count startDay 30 0 0

/-- Checks if the number of Tuesdays and Thursdays are equal for a given starting day -/
def hasEqualTuesdaysAndThursdays (startDay : DayOfWeek) : Bool :=
  let (tuesdays, thursdays) := countTuesdaysAndThursdays startDay
  tuesdays = thursdays

/-- Counts the number of days that result in equal Tuesdays and Thursdays -/
def countEqualDays : Nat :=
  let days := [DayOfWeek.Sunday, DayOfWeek.Monday, DayOfWeek.Tuesday, DayOfWeek.Wednesday,
               DayOfWeek.Thursday, DayOfWeek.Friday, DayOfWeek.Saturday]
  days.filter hasEqualTuesdaysAndThursdays |>.length

theorem equal_days_count :
  countEqualDays = 4 :=
sorry

end NUMINAMATH_CALUDE_equal_days_count_l100_10066


namespace NUMINAMATH_CALUDE_extreme_points_imply_a_negative_l100_10085

/-- A cubic function with a linear term -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

/-- A function has two extreme points if its derivative has two distinct real roots -/
def has_two_extreme_points (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f' a x₁ = 0 ∧ f' a x₂ = 0

theorem extreme_points_imply_a_negative (a : ℝ) :
  has_two_extreme_points a → a < 0 := by sorry

end NUMINAMATH_CALUDE_extreme_points_imply_a_negative_l100_10085


namespace NUMINAMATH_CALUDE_exists_grid_with_more_than_20_components_l100_10086

/-- Represents a diagonal in a cell --/
inductive Diagonal
| TopLeft
| TopRight

/-- Represents the grid --/
def Grid := Matrix (Fin 8) (Fin 8) Diagonal

/-- A function that counts the number of connected components in a grid --/
def countComponents (g : Grid) : ℕ := sorry

/-- Theorem stating that there exists a grid configuration with more than 20 components --/
theorem exists_grid_with_more_than_20_components :
  ∃ (g : Grid), countComponents g > 20 :=
sorry

end NUMINAMATH_CALUDE_exists_grid_with_more_than_20_components_l100_10086


namespace NUMINAMATH_CALUDE_triangle_midpoint_sum_l100_10044

theorem triangle_midpoint_sum (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) + (a + c) + (b + c) = 30 := by
  sorry

end NUMINAMATH_CALUDE_triangle_midpoint_sum_l100_10044


namespace NUMINAMATH_CALUDE_amount_saved_is_30_l100_10002

/-- Calculates the amount saved after clearing debt given income and expenses -/
def amountSavedAfterDebt (monthlyIncome : ℕ) (initialExpense : ℕ) (reducedExpense : ℕ) : ℕ :=
  let initialPeriod := 6
  let reducedPeriod := 4
  let initialDebt := initialPeriod * initialExpense - initialPeriod * monthlyIncome
  let totalIncome := (initialPeriod + reducedPeriod) * monthlyIncome
  let totalExpense := initialPeriod * initialExpense + reducedPeriod * reducedExpense
  totalIncome - (totalExpense + initialDebt)

/-- Theorem: Given the specified income and expenses, the amount saved after clearing debt is 30 -/
theorem amount_saved_is_30 :
  amountSavedAfterDebt 69 70 60 = 30 := by
  sorry

end NUMINAMATH_CALUDE_amount_saved_is_30_l100_10002


namespace NUMINAMATH_CALUDE_P₁_subset_P₂_l100_10047

/-- P₁ is the set of real numbers x such that x² + ax + 1 > 0 -/
def P₁ (a : ℝ) : Set ℝ := {x | x^2 + a*x + 1 > 0}

/-- P₂ is the set of real numbers x such that x² + ax + 2 > 0 -/
def P₂ (a : ℝ) : Set ℝ := {x | x^2 + a*x + 2 > 0}

/-- For all real numbers a, P₁(a) is a subset of P₂(a) -/
theorem P₁_subset_P₂ : ∀ a : ℝ, P₁ a ⊆ P₂ a := by
  sorry

end NUMINAMATH_CALUDE_P₁_subset_P₂_l100_10047


namespace NUMINAMATH_CALUDE_last_two_digits_sum_l100_10059

theorem last_two_digits_sum (n : ℕ) : n = 25 → (15^n + 5^n) % 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_l100_10059


namespace NUMINAMATH_CALUDE_mph_to_fps_conversion_l100_10070

/-- Conversion factor from miles per hour to feet per second -/
def mph_to_fps : ℝ := 1.5

/-- Cheetah's speed in miles per hour -/
def cheetah_speed : ℝ := 60

/-- Gazelle's speed in miles per hour -/
def gazelle_speed : ℝ := 40

/-- Initial distance between cheetah and gazelle in feet -/
def initial_distance : ℝ := 210

/-- Time for cheetah to catch up to gazelle in seconds -/
def catch_up_time : ℝ := 7

theorem mph_to_fps_conversion :
  (cheetah_speed * mph_to_fps * catch_up_time) - (gazelle_speed * mph_to_fps * catch_up_time) = initial_distance := by
  sorry

#check mph_to_fps_conversion

end NUMINAMATH_CALUDE_mph_to_fps_conversion_l100_10070


namespace NUMINAMATH_CALUDE_total_slices_needed_l100_10032

/-- The number of sandwiches Ryan wants to make -/
def num_sandwiches : ℕ := 5

/-- The number of bread slices needed for each sandwich -/
def slices_per_sandwich : ℕ := 3

/-- Theorem stating the total number of bread slices needed -/
theorem total_slices_needed : num_sandwiches * slices_per_sandwich = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_slices_needed_l100_10032


namespace NUMINAMATH_CALUDE_three_in_all_curriculums_l100_10069

/-- Represents the number of people in different curriculum groups -/
structure CurriculumGroups where
  yoga : ℕ
  cooking : ℕ
  weaving : ℕ
  cookingOnly : ℕ
  cookingAndYoga : ℕ
  cookingAndWeaving : ℕ

/-- Calculates the number of people participating in all curriculums -/
def allCurriculums (g : CurriculumGroups) : ℕ :=
  g.cooking - g.cookingOnly - g.cookingAndYoga - g.cookingAndWeaving

/-- Theorem stating that 3 people participate in all curriculums -/
theorem three_in_all_curriculums (g : CurriculumGroups) 
  (h1 : g.yoga = 35)
  (h2 : g.cooking = 20)
  (h3 : g.weaving = 15)
  (h4 : g.cookingOnly = 7)
  (h5 : g.cookingAndYoga = 5)
  (h6 : g.cookingAndWeaving = 5) :
  allCurriculums g = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_in_all_curriculums_l100_10069


namespace NUMINAMATH_CALUDE_lottery_prize_probability_l100_10062

/-- The probability of getting a prize in a lottery with 10 prizes and 25 blanks -/
theorem lottery_prize_probability :
  let num_prizes : ℕ := 10
  let num_blanks : ℕ := 25
  let total_outcomes : ℕ := num_prizes + num_blanks
  let probability : ℚ := num_prizes / total_outcomes
  probability = 2 / 7 := by
sorry

end NUMINAMATH_CALUDE_lottery_prize_probability_l100_10062


namespace NUMINAMATH_CALUDE_E_parity_l100_10029

def E : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | 2 => 0
  | n + 3 => E (n + 2) + E (n + 1)

def isEven (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

def isOdd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem E_parity : isEven (E 2023) ∧ isOdd (E 2024) ∧ isOdd (E 2025) := by sorry

end NUMINAMATH_CALUDE_E_parity_l100_10029


namespace NUMINAMATH_CALUDE_largest_multiple_of_seven_less_than_negative_eightyfive_l100_10098

theorem largest_multiple_of_seven_less_than_negative_eightyfive :
  ∀ n : ℤ, n * 7 < -85 → n * 7 ≤ -91 :=
sorry

end NUMINAMATH_CALUDE_largest_multiple_of_seven_less_than_negative_eightyfive_l100_10098


namespace NUMINAMATH_CALUDE_incorrect_observation_value_l100_10071

/-- Given a set of observations with known properties, determine the value of an incorrectly recorded observation. -/
theorem incorrect_observation_value
  (n : ℕ)  -- Total number of observations
  (original_mean : ℝ)  -- Original mean of all observations
  (correct_value : ℝ)  -- The correct value of the misrecorded observation
  (new_mean : ℝ)  -- The new mean after correcting the misrecorded observation
  (h_n : n = 50)  -- There are 50 observations
  (h_original_mean : original_mean = 36)  -- The original mean was 36
  (h_correct_value : correct_value = 30)  -- The correct value should have been 30
  (h_new_mean : new_mean = 36.5)  -- The new mean after correction is 36.5
  : ∃ (incorrect_value : ℝ), incorrect_value = 55 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_observation_value_l100_10071


namespace NUMINAMATH_CALUDE_solution_in_interval_l100_10096

def f (x : ℝ) := x^2 + 12*x - 15

theorem solution_in_interval :
  ∃ x : ℝ, x ∈ (Set.Ioo 1.1 1.2) ∧ f x = 0 :=
by
  have h1 : f 1.1 < 0 := by sorry
  have h2 : f 1.2 > 0 := by sorry
  sorry

end NUMINAMATH_CALUDE_solution_in_interval_l100_10096


namespace NUMINAMATH_CALUDE_inequality_proof_l100_10018

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
  (h4 : x^2 + y^2 + z^2 = 2*(x*y + y*z + z*x)) : 
  (x + y + z) / 3 ≥ (2*x*y*z)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l100_10018


namespace NUMINAMATH_CALUDE_shape_e_not_in_square_pieces_l100_10093

/-- Represents a shape in the diagram -/
structure Shape :=
  (id : String)

/-- Represents the set of shapes in the divided square -/
def SquarePieces : Finset Shape := sorry

/-- Represents the set of given shapes to check -/
def GivenShapes : Finset Shape := sorry

/-- Shape E is defined separately for the theorem -/
def ShapeE : Shape := { id := "E" }

theorem shape_e_not_in_square_pieces :
  ShapeE ∉ SquarePieces ∧
  ∀ s ∈ GivenShapes, s ≠ ShapeE → s ∈ SquarePieces :=
sorry

end NUMINAMATH_CALUDE_shape_e_not_in_square_pieces_l100_10093


namespace NUMINAMATH_CALUDE_apollo_chariot_payment_l100_10065

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of months before the price increase -/
def months_before_increase : ℕ := 6

/-- The initial price in golden apples per month -/
def initial_price : ℕ := 3

/-- The price increase factor -/
def price_increase_factor : ℕ := 2

/-- The total number of golden apples paid for the year -/
def total_apples : ℕ := 
  initial_price * months_before_increase + 
  initial_price * price_increase_factor * (months_in_year - months_before_increase)

theorem apollo_chariot_payment :
  total_apples = 54 := by sorry

end NUMINAMATH_CALUDE_apollo_chariot_payment_l100_10065


namespace NUMINAMATH_CALUDE_integer_pairs_satisfying_conditions_l100_10026

theorem integer_pairs_satisfying_conditions :
  ∀ m n : ℤ, 
    m^2 = n^5 + n^4 + 1 ∧ 
    (m - 7*n) ∣ (m - 4*n) → 
    ((m = -1 ∧ n = 0) ∨ (m = 1 ∧ n = 0)) := by
  sorry

end NUMINAMATH_CALUDE_integer_pairs_satisfying_conditions_l100_10026


namespace NUMINAMATH_CALUDE_xiao_ying_pe_grade_l100_10023

/-- Calculates the final grade based on component weights and scores -/
def calculate_final_grade (weight1 weight2 weight3 score1 score2 score3 : ℝ) : ℝ :=
  weight1 * score1 + weight2 * score2 + weight3 * score3

/-- Xiao Ying's physical education grade calculation -/
theorem xiao_ying_pe_grade :
  let weight1 : ℝ := 0.3  -- Regular physical activity performance weight
  let weight2 : ℝ := 0.2  -- Physical education theory test weight
  let weight3 : ℝ := 0.5  -- Physical education skills test weight
  let score1 : ℝ := 90    -- Regular physical activity performance score
  let score2 : ℝ := 80    -- Physical education theory test score
  let score3 : ℝ := 94    -- Physical education skills test score
  calculate_final_grade weight1 weight2 weight3 score1 score2 score3 = 90
  := by sorry

end NUMINAMATH_CALUDE_xiao_ying_pe_grade_l100_10023


namespace NUMINAMATH_CALUDE_initial_average_calculation_l100_10058

theorem initial_average_calculation (n : ℕ) (correct_sum incorrect_sum : ℝ) 
  (h1 : n = 10)
  (h2 : correct_sum / n = 18)
  (h3 : incorrect_sum = correct_sum - 46 + 26) :
  incorrect_sum / n = 16 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_calculation_l100_10058


namespace NUMINAMATH_CALUDE_cos_equality_317_degrees_l100_10004

theorem cos_equality_317_degrees (n : ℕ) (h1 : n ≤ 180) (h2 : Real.cos (n * π / 180) = Real.cos (317 * π / 180)) : n = 43 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_317_degrees_l100_10004


namespace NUMINAMATH_CALUDE_unique_number_with_three_prime_factors_l100_10021

theorem unique_number_with_three_prime_factors (x n : ℕ) : 
  x = 6^n + 1 →
  Odd n →
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧ x = 11 * p * q) →
  (∀ r : ℕ, Prime r ∧ r ∣ x → r = 11 ∨ r = p ∨ r = q) →
  x = 7777 := by
sorry

end NUMINAMATH_CALUDE_unique_number_with_three_prime_factors_l100_10021


namespace NUMINAMATH_CALUDE_conference_attendance_l100_10074

/-- The number of writers at the conference -/
def writers : ℕ := 45

/-- The number of editors at the conference -/
def editors : ℕ := 37

/-- The number of people who are both writers and editors -/
def both : ℕ := 18

/-- The number of people who are neither writers nor editors -/
def neither : ℕ := 2 * both

/-- The total number of people attending the conference -/
def total : ℕ := writers + editors - both + neither

theorem conference_attendance :
  editors > 36 ∧ both ≤ 18 → total = 100 := by sorry

end NUMINAMATH_CALUDE_conference_attendance_l100_10074


namespace NUMINAMATH_CALUDE_unit_digit_of_4137_to_754_l100_10089

theorem unit_digit_of_4137_to_754 : (4137^754) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_4137_to_754_l100_10089


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_fraction_l100_10001

theorem greatest_integer_less_than_negative_fraction :
  ⌊-22/5⌋ = -5 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_fraction_l100_10001


namespace NUMINAMATH_CALUDE_range_of_a_l100_10020

theorem range_of_a (a : ℝ) 
  (h1 : ∀ x : ℝ, x^2 - a ≥ 0)
  (h2 : ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) :
  a ≤ -2 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l100_10020


namespace NUMINAMATH_CALUDE_bouquet_cost_55_l100_10072

/-- The cost of a bouquet of lilies given the number of lilies -/
def bouquet_cost (n : ℕ) : ℚ :=
  (30 : ℚ) * n / 24

theorem bouquet_cost_55 : bouquet_cost 55 = (68750 : ℚ) / 1000 := by
  sorry

end NUMINAMATH_CALUDE_bouquet_cost_55_l100_10072


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l100_10030

def S : Set ℝ := {x | (x - 2) * (x + 3) > 0}

def T : Set ℝ := {x | ∃ y, y = Real.sqrt (3 - x)}

theorem intersection_equals_interval : S ∩ T = Set.Ioo 2 3 ∪ Set.singleton 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l100_10030


namespace NUMINAMATH_CALUDE_cubic_monotonicity_l100_10083

def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

def f' (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

theorem cubic_monotonicity 
  (a b c d : ℝ) 
  (h1 : f a b c d 0 = -4)
  (h2 : f' a b c 0 = 12)
  (h3 : f a b c d 2 = 0)
  (h4 : f' a b c 2 = 0) :
  ∃ (x₁ x₂ : ℝ), x₁ = 1 ∧ x₂ = 2 ∧
  (∀ x < x₁, f' a b c x > 0) ∧
  (∀ x ∈ Set.Ioo x₁ x₂, f' a b c x < 0) ∧
  (∀ x > x₂, f' a b c x > 0) :=
sorry

end NUMINAMATH_CALUDE_cubic_monotonicity_l100_10083


namespace NUMINAMATH_CALUDE_problem_solution_l100_10056

theorem problem_solution (a b : ℝ) 
  (h1 : 5 + a = 6 - b) 
  (h2 : 3 + b = 8 + a) : 
  5 - a = 7 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l100_10056


namespace NUMINAMATH_CALUDE_parallel_line_slope_l100_10015

/-- The slope of a line parallel to 5x - 3y = 12 is 5/3 -/
theorem parallel_line_slope : 
  ∀ (m : ℚ), (∃ b : ℚ, ∀ x y : ℚ, 5 * x - 3 * y = 12 ↔ y = m * x + b) → m = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l100_10015


namespace NUMINAMATH_CALUDE_simplify_fraction_l100_10087

theorem simplify_fraction (x y : ℝ) (hx : x = 3) (hy : y = 2) :
  15 * x^2 * y^3 / (9 * x * y^2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l100_10087


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l100_10078

/-- The speed of a boat in still water, given downstream travel information -/
theorem boat_speed_in_still_water (current_speed : ℝ) (downstream_distance : ℝ) (travel_time_minutes : ℝ) :
  current_speed = 7 →
  downstream_distance = 35.93 →
  travel_time_minutes = 44 →
  ∃ (v : ℝ), abs (v - 42) < 0.01 ∧ downstream_distance = (v + current_speed) * (travel_time_minutes / 60) :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l100_10078


namespace NUMINAMATH_CALUDE_bike_rides_ratio_l100_10014

/-- Proves that the ratio of John's bike rides to Billy's bike rides is 2:1 --/
theorem bike_rides_ratio : 
  ∀ (john_rides : ℕ),
  (17 : ℕ) + john_rides + (john_rides + 10) = 95 →
  (john_rides : ℚ) / 17 = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_bike_rides_ratio_l100_10014


namespace NUMINAMATH_CALUDE_inequality_and_equality_conditions_l100_10037

theorem inequality_and_equality_conditions (x y z : ℝ) 
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 2) :
  3*x + 8*x*y + 16*x*y*z ≤ 12 ∧ 
  (3*x + 8*x*y + 16*x*y*z = 12 ↔ x = 1 ∧ y = 3/4 ∧ z = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_conditions_l100_10037


namespace NUMINAMATH_CALUDE_place_value_ratio_l100_10019

def number : ℚ := 56439.2071

theorem place_value_ratio : 
  (10000 : ℚ) * (number - number.floor) * 10 = (number.floor % 100000 - number.floor % 10000) / 10 := by
  sorry

end NUMINAMATH_CALUDE_place_value_ratio_l100_10019


namespace NUMINAMATH_CALUDE_ship_speed_calculation_l100_10076

/-- The speed of the train in km/h -/
def train_speed : ℝ := 48

/-- The total distance traveled in km -/
def total_distance : ℝ := 480

/-- The additional time taken by the train in hours -/
def additional_time : ℝ := 2

/-- The speed of the ship in km/h -/
def ship_speed : ℝ := 60

theorem ship_speed_calculation : 
  (total_distance / ship_speed) + additional_time = total_distance / train_speed := by
  sorry

#check ship_speed_calculation

end NUMINAMATH_CALUDE_ship_speed_calculation_l100_10076


namespace NUMINAMATH_CALUDE_expression_equality_l100_10060

theorem expression_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x = 1 / y) :
  (x - 1 / x) * (y + 1 / y) = x^2 - y^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l100_10060


namespace NUMINAMATH_CALUDE_parabola_hyperbola_intersection_l100_10055

/-- Parabola C₁ with focus F and equation y² = 2px (p > 0) -/
structure Parabola where
  p : ℝ
  focus : ℝ × ℝ
  eq : (x y : ℝ) → Prop

/-- Hyperbola C₂ with equation y²/4 - x²/3 = 1 -/
structure Hyperbola where
  eq : (x y : ℝ) → Prop

/-- Two points A and B in the first quadrant -/
structure IntersectionPoints where
  A : ℝ × ℝ
  B : ℝ × ℝ
  first_quadrant : Prop

/-- Area of triangle FAB -/
def triangleArea (F A B : ℝ × ℝ) : ℝ := sorry

/-- Dot product of vectors FA and FB -/
def dotProduct (F A B : ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem parabola_hyperbola_intersection
  (C₁ : Parabola)
  (C₂ : Hyperbola)
  (points : IntersectionPoints)
  (h₁ : C₁.p > 0)
  (h₂ : C₁.eq = fun x y ↦ y^2 = 2 * C₁.p * x)
  (h₃ : C₂.eq = fun x y ↦ y^2 / 4 - x^2 / 3 = 1)
  (h₄ : C₁.focus = (C₁.p / 2, 0))
  (h₅ : triangleArea C₁.focus points.A points.B = 2/3 * dotProduct C₁.focus points.A points.B) :
  C₁.p = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_intersection_l100_10055


namespace NUMINAMATH_CALUDE_probability_second_green_given_first_green_l100_10092

def total_balls : ℕ := 14
def green_balls : ℕ := 8
def red_balls : ℕ := 6

theorem probability_second_green_given_first_green :
  (green_balls : ℚ) / total_balls = 
  (green_balls : ℚ) / (green_balls + red_balls) :=
by sorry

end NUMINAMATH_CALUDE_probability_second_green_given_first_green_l100_10092


namespace NUMINAMATH_CALUDE_mistaken_quotient_l100_10011

theorem mistaken_quotient (D : ℕ) (h1 : D % 21 = 0) (h2 : D / 21 = 32) : D / 12 = 56 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_quotient_l100_10011


namespace NUMINAMATH_CALUDE_sandy_tokens_l100_10027

theorem sandy_tokens (total_tokens : ℕ) (num_siblings : ℕ) : 
  total_tokens = 1000000 →
  num_siblings = 4 →
  let sandy_share := total_tokens / 2
  let remaining_tokens := total_tokens - sandy_share
  let sibling_share := remaining_tokens / num_siblings
  sandy_share - sibling_share = 375000 :=
by
  sorry

end NUMINAMATH_CALUDE_sandy_tokens_l100_10027


namespace NUMINAMATH_CALUDE_count_six_digit_numbers_middle_same_is_90000_l100_10080

/-- Counts the number of six-digit numbers where only the middle two digits are the same -/
def count_six_digit_numbers_middle_same : ℕ :=
  -- First digit: 9 choices (1-9)
  9 * 
  -- Second digit: 10 choices (0-9)
  10 * 
  -- Third digit: 10 choices (0-9)
  10 * 
  -- Fourth digit: 1 choice (same as third)
  1 * 
  -- Fifth digit: 10 choices (0-9)
  10 * 
  -- Sixth digit: 10 choices (0-9)
  10

/-- Theorem stating that the count of six-digit numbers with only middle digits the same is 90000 -/
theorem count_six_digit_numbers_middle_same_is_90000 :
  count_six_digit_numbers_middle_same = 90000 := by
  sorry

end NUMINAMATH_CALUDE_count_six_digit_numbers_middle_same_is_90000_l100_10080


namespace NUMINAMATH_CALUDE_intersection_y_diff_zero_l100_10036

def f (x : ℝ) : ℝ := 2 - x^2 + x^4
def g (x : ℝ) : ℝ := -1 + x^2 + x^4

theorem intersection_y_diff_zero :
  let intersection_points := {x : ℝ | f x = g x}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ intersection_points ∧ x₂ ∈ intersection_points ∧
    ∀ (y₁ y₂ : ℝ), y₁ = f x₁ ∧ y₂ = f x₂ → |y₁ - y₂| = 0 :=
sorry

end NUMINAMATH_CALUDE_intersection_y_diff_zero_l100_10036


namespace NUMINAMATH_CALUDE_third_term_of_geometric_series_l100_10091

/-- Given an infinite geometric series with common ratio 1/4 and sum 16, 
    the third term of the sequence is 3/4. -/
theorem third_term_of_geometric_series (a : ℝ) : 
  (∃ (S : ℝ), S = 16 ∧ S = a / (1 - (1/4))) →
  a * (1/4)^2 = 3/4 := by
sorry

end NUMINAMATH_CALUDE_third_term_of_geometric_series_l100_10091


namespace NUMINAMATH_CALUDE_pyramid_cone_properties_l100_10010

/-- Represents a square pyramid with a cone resting on its base --/
structure PyramidWithCone where
  pyramid_height : ℝ
  cone_base_radius : ℝ
  -- The cone is tangent to the other four faces of the pyramid
  is_tangent : Bool

/-- Calculates the edge length of the pyramid's base --/
def calculate_edge_length (p : PyramidWithCone) : ℝ := sorry

/-- Calculates the surface area of the cone not in contact with the pyramid --/
def calculate_cone_surface_area (p : PyramidWithCone) : ℝ := sorry

/-- Theorem stating the properties of the specific pyramid and cone configuration --/
theorem pyramid_cone_properties :
  let p : PyramidWithCone := {
    pyramid_height := 9,
    cone_base_radius := 3,
    is_tangent := true
  }
  calculate_edge_length p = 9 ∧
  calculate_cone_surface_area p = 30 * Real.pi := by sorry

end NUMINAMATH_CALUDE_pyramid_cone_properties_l100_10010


namespace NUMINAMATH_CALUDE_volume_of_one_gram_l100_10079

/-- Given a substance where 1 cubic meter has a mass of 100 kg, 
    prove that 1 gram of this substance has a volume of 10 cubic centimeters. -/
theorem volume_of_one_gram (substance_mass : ℝ) (substance_volume : ℝ) 
  (h1 : substance_mass = 100) 
  (h2 : substance_volume = 1) 
  (h3 : (1 : ℝ) = 1000 * (1 / 1000)) -- 1 kg = 1000 g
  (h4 : (1 : ℝ) = 1000000 * (1 / 1000000)) -- 1 m³ = 1,000,000 cm³
  : (1 / 1000) / (substance_mass / substance_volume) = 10 * (1 / 1000000) := by
  sorry

end NUMINAMATH_CALUDE_volume_of_one_gram_l100_10079


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l100_10017

theorem ceiling_floor_sum : ⌈(5/4 : ℝ)⌉ + ⌊-(5/4 : ℝ)⌋ = 0 :=
by sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l100_10017


namespace NUMINAMATH_CALUDE_absolute_value_of_five_minus_e_l100_10050

-- Define e as a constant approximation
def e : ℝ := 2.71828

-- State the theorem
theorem absolute_value_of_five_minus_e : |5 - e| = 2.28172 := by sorry

end NUMINAMATH_CALUDE_absolute_value_of_five_minus_e_l100_10050


namespace NUMINAMATH_CALUDE_train_cars_count_l100_10028

/-- The number of cars counted in the first 15 seconds -/
def initial_cars : ℕ := 9

/-- The time in seconds for the initial count -/
def initial_time : ℕ := 15

/-- The total time in seconds for the train to clear the crossing -/
def total_time : ℕ := 210

/-- The number of cars in the train -/
def train_cars : ℕ := (initial_cars * total_time) / initial_time

theorem train_cars_count : train_cars = 126 := by
  sorry

end NUMINAMATH_CALUDE_train_cars_count_l100_10028


namespace NUMINAMATH_CALUDE_gauss_family_mean_age_l100_10033

/-- The ages of the Gauss family children -/
def gauss_ages : List ℕ := [7, 7, 7, 14, 15]

/-- The number of children in the Gauss family -/
def num_children : ℕ := gauss_ages.length

/-- The mean age of the Gauss family children -/
def mean_age : ℚ := (gauss_ages.sum : ℚ) / num_children

theorem gauss_family_mean_age : mean_age = 10 := by
  sorry

end NUMINAMATH_CALUDE_gauss_family_mean_age_l100_10033


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l100_10006

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)  -- subset relation for a line in a plane
variable (perpendicular : Line → Line → Prop)  -- perpendicular relation between lines
variable (perpendicularToPlane : Line → Plane → Prop)  -- perpendicular relation between a line and a plane
variable (parallel : Plane → Plane → Prop)  -- parallel relation between planes

-- State the theorem
theorem sufficient_but_not_necessary
  (a b : Line) (α β : Plane)
  (h1 : subset a α)
  (h2 : perpendicularToPlane b β)
  (h3 : parallel α β) :
  perpendicular a b ∧
  ¬(∀ (a b : Line) (α β : Plane),
    perpendicular a b →
    subset a α ∧ perpendicularToPlane b β ∧ parallel α β) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l100_10006


namespace NUMINAMATH_CALUDE_mitzi_food_expense_l100_10054

/-- Proves that the amount spent on food is $13 given the conditions of Mitzi's amusement park expenses --/
theorem mitzi_food_expense (
  total_brought : ℕ)
  (ticket_cost : ℕ)
  (tshirt_cost : ℕ)
  (money_left : ℕ)
  (h1 : total_brought = 75)
  (h2 : ticket_cost = 30)
  (h3 : tshirt_cost = 23)
  (h4 : money_left = 9)
  : total_brought - money_left - (ticket_cost + tshirt_cost) = 13 := by
  sorry

end NUMINAMATH_CALUDE_mitzi_food_expense_l100_10054


namespace NUMINAMATH_CALUDE_fraction_order_l100_10013

theorem fraction_order : 
  (20 : ℚ) / 15 < 25 / 18 ∧ 25 / 18 < 23 / 16 ∧ 23 / 16 < 21 / 14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_order_l100_10013


namespace NUMINAMATH_CALUDE_hallway_floor_design_ratio_l100_10075

/-- Given a rectangle with semicircles on either side, where the ratio of length to width
    is 4:1 and the width is 20 inches, the ratio of the area of the rectangle to the
    combined area of the semicircles is 16/π. -/
theorem hallway_floor_design_ratio : 
  ∀ (length width : ℝ),
  width = 20 →
  length = 4 * width →
  (length * width) / (π * (width / 2)^2) = 16 / π :=
by sorry

end NUMINAMATH_CALUDE_hallway_floor_design_ratio_l100_10075


namespace NUMINAMATH_CALUDE_isosceles_triangle_smallest_angle_l100_10039

/-- Given an isosceles triangle with one angle 80% larger than a right angle,
    prove that one of the two smallest angles measures 9 degrees. -/
theorem isosceles_triangle_smallest_angle :
  ∀ (a b c : ℝ),
  -- The triangle is isosceles
  a = b →
  -- One angle is 80% larger than a right angle
  c = 1.8 * 90 →
  -- The sum of angles in a triangle is 180°
  a + b + c = 180 →
  -- One of the two smallest angles measures 9°
  a = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_smallest_angle_l100_10039


namespace NUMINAMATH_CALUDE_gold_copper_ratio_l100_10082

/-- Proves that the ratio of gold to copper in an alloy that is 17 times as heavy as water is 4:1,
    given that gold is 19 times as heavy as water and copper is 9 times as heavy as water. -/
theorem gold_copper_ratio (g c : ℝ) 
  (h1 : g > 0) 
  (h2 : c > 0) 
  (h_gold : 19 * g = 17 * (g + c)) 
  (h_copper : 9 * c = 17 * (g + c) - 19 * g) : 
  g / c = 4 := by
sorry

end NUMINAMATH_CALUDE_gold_copper_ratio_l100_10082


namespace NUMINAMATH_CALUDE_testicular_cell_properties_l100_10081

-- Define the possible bases
inductive Base
| A
| C
| T

-- Define the possible cell cycle periods
inductive Period
| Interphase
| EarlyMitosis
| LateMitosis
| EarlyMeiosis1
| LateMeiosis1
| EarlyMeiosis2
| LateMeiosis2

-- Define the structure of a testicular cell
structure TesticularCell where
  nucleotideTypes : Finset (List Base)
  lowestStabilityPeriod : Period
  dnaSeperationPeriod : Period

-- Define the theorem
theorem testicular_cell_properties : ∃ (cell : TesticularCell),
  (cell.nucleotideTypes.card = 3) ∧
  (cell.lowestStabilityPeriod = Period.Interphase) ∧
  (cell.dnaSeperationPeriod = Period.LateMeiosis1 ∨ cell.dnaSeperationPeriod = Period.LateMeiosis2) :=
by
  sorry

end NUMINAMATH_CALUDE_testicular_cell_properties_l100_10081


namespace NUMINAMATH_CALUDE_existence_of_person_with_few_amicable_foes_l100_10008

structure Society where
  n : ℕ  -- number of persons
  q : ℕ  -- number of amicable pairs
  is_valid : q ≤ n * (n - 1) / 2  -- maximum possible number of pairs

def is_hostile (S : Society) (a b : Fin S.n) : Prop := sorry

def is_amicable (S : Society) (a b : Fin S.n) : Prop := ¬(is_hostile S a b)

axiom society_property (S : Society) :
  ∀ (a b c : Fin S.n), a ≠ b ∧ b ≠ c ∧ a ≠ c →
    is_hostile S a b ∨ is_hostile S b c ∨ is_hostile S a c

def foes (S : Society) (a : Fin S.n) : Set (Fin S.n) :=
  {b | is_hostile S a b}

def amicable_pairs_among_foes (S : Society) (a : Fin S.n) : ℕ := sorry

theorem existence_of_person_with_few_amicable_foes (S : Society) :
  ∃ (a : Fin S.n), amicable_pairs_among_foes S a ≤ S.q * (1 - 4 * S.q / (S.n * S.n)) :=
sorry

end NUMINAMATH_CALUDE_existence_of_person_with_few_amicable_foes_l100_10008


namespace NUMINAMATH_CALUDE_carries_cake_profit_l100_10022

/-- Calculates the profit for a cake decorator given their work hours, pay rate, and supply cost. -/
def cake_decorator_profit (hours_per_day : ℕ) (days_worked : ℕ) (hourly_rate : ℕ) (supply_cost : ℕ) : ℕ :=
  hours_per_day * days_worked * hourly_rate - supply_cost

/-- Proves that Carrie's profit from decorating a wedding cake is $122. -/
theorem carries_cake_profit :
  cake_decorator_profit 2 4 22 54 = 122 := by
  sorry

end NUMINAMATH_CALUDE_carries_cake_profit_l100_10022


namespace NUMINAMATH_CALUDE_proportion_solution_l100_10043

theorem proportion_solution (x : ℝ) (h : 0.25 / x = 2 / 6) : x = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l100_10043


namespace NUMINAMATH_CALUDE_page_number_divisibility_l100_10057

theorem page_number_divisibility (n : ℕ) (k : ℕ) : 
  n ≥ 52 → 
  52 ≤ n → 
  n % 13 = 0 → 
  n % k = 0 → 
  ∀ m, m < n → (m % 13 = 0 → m % k = 0) → m < 52 →
  k = 4 := by
  sorry

end NUMINAMATH_CALUDE_page_number_divisibility_l100_10057


namespace NUMINAMATH_CALUDE_equation_solution_l100_10009

theorem equation_solution : 
  ∃! x : ℝ, (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) ∧ x = -9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l100_10009


namespace NUMINAMATH_CALUDE_not_p_false_sufficient_not_necessary_for_p_or_q_true_l100_10000

theorem not_p_false_sufficient_not_necessary_for_p_or_q_true (p q : Prop) :
  (¬¬p → p ∨ q) ∧ ∃ (p q : Prop), (p ∨ q) ∧ ¬(¬¬p) := by sorry

end NUMINAMATH_CALUDE_not_p_false_sufficient_not_necessary_for_p_or_q_true_l100_10000


namespace NUMINAMATH_CALUDE_vivian_yogurt_count_l100_10077

/-- The number of banana slices per yogurt -/
def slices_per_yogurt : ℕ := 8

/-- The number of slices one banana yields -/
def slices_per_banana : ℕ := 10

/-- The number of bananas Vivian needs to buy -/
def bananas_to_buy : ℕ := 4

/-- The number of yogurts Vivian needs to make -/
def yogurts_to_make : ℕ := (bananas_to_buy * slices_per_banana) / slices_per_yogurt

theorem vivian_yogurt_count : yogurts_to_make = 5 := by
  sorry

end NUMINAMATH_CALUDE_vivian_yogurt_count_l100_10077


namespace NUMINAMATH_CALUDE_complex_equation_solution_l100_10061

theorem complex_equation_solution (z : ℂ) : (Complex.I - z = 2 - Complex.I) → z = -2 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l100_10061


namespace NUMINAMATH_CALUDE_f_81_product_remainder_l100_10097

def p : ℕ := 2^16 + 1

-- S is implicitly defined as the set of positive integers not divisible by p

def is_in_S (x : ℕ) : Prop := x > 0 ∧ ¬(p ∣ x)

axiom p_is_prime : Nat.Prime p

axiom f_exists : ∃ (f : ℕ → ℕ), 
  (∀ x, is_in_S x → f x < p) ∧
  (∀ x y, is_in_S x → is_in_S y → (f x * f y) % p = (f (x * y) + f (x * y^(p-2))) % p) ∧
  (∀ x, is_in_S x → f (x + p) = f x)

def N : ℕ := sorry  -- Definition of N as the product of nonzero f(81) values

theorem f_81_product_remainder : N % p = 16384 := by sorry

end NUMINAMATH_CALUDE_f_81_product_remainder_l100_10097


namespace NUMINAMATH_CALUDE_two_digit_number_interchange_l100_10041

theorem two_digit_number_interchange (a b j : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : 10 * a + b = j * (a + b)) :
  10 * b + a = (10 * j - 9) * (a + b) :=
sorry

end NUMINAMATH_CALUDE_two_digit_number_interchange_l100_10041


namespace NUMINAMATH_CALUDE_triangle_side_range_l100_10052

theorem triangle_side_range (a b c : ℝ) : 
  (|a - 3| + (b - 7)^2 = 0) →
  (c ≥ a ∧ c ≥ b) →
  (c < a + b) →
  (7 ≤ c ∧ c < 10) :=
sorry

end NUMINAMATH_CALUDE_triangle_side_range_l100_10052


namespace NUMINAMATH_CALUDE_mindmaster_secret_codes_l100_10067

/-- The number of different colors available for pegs -/
def num_colors : ℕ := 8

/-- The number of slots in the code -/
def num_slots : ℕ := 4

/-- The total number of options for each slot (colors + empty) -/
def options_per_slot : ℕ := num_colors + 1

/-- The number of possible secret codes in the Mindmaster variation -/
theorem mindmaster_secret_codes :
  (options_per_slot ^ num_slots) - 1 = 6560 := by sorry

end NUMINAMATH_CALUDE_mindmaster_secret_codes_l100_10067


namespace NUMINAMATH_CALUDE_correct_transformation_l100_10005

theorem correct_transformation (a x y : ℝ) : 
  ax = ay → 3 - ax = 3 - ay := by
sorry

end NUMINAMATH_CALUDE_correct_transformation_l100_10005


namespace NUMINAMATH_CALUDE_rectangle_square_ratio_l100_10063

/-- Configuration of rectangles around a square -/
structure RectangleSquareConfig where
  /-- Side length of the inner square -/
  inner_side : ℝ
  /-- Shorter side of each rectangle -/
  rect_short : ℝ
  /-- Longer side of each rectangle -/
  rect_long : ℝ

/-- Theorem: If the area of the outer square is 9 times that of the inner square,
    then the ratio of the longer side to the shorter side of each rectangle is 2 -/
theorem rectangle_square_ratio (config : RectangleSquareConfig) 
    (h_area : (config.inner_side + 2 * config.rect_short)^2 = 9 * config.inner_side^2) :
    config.rect_long / config.rect_short = 2 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_square_ratio_l100_10063


namespace NUMINAMATH_CALUDE_unique_c_for_unique_solution_l100_10046

/-- The quadratic equation in x with parameter b -/
def quadratic (b : ℝ) (c : ℝ) (x : ℝ) : Prop :=
  x^2 + (b^2 + 3*b + 1/b)*x + c = 0

/-- The statement to be proved -/
theorem unique_c_for_unique_solution :
  ∃! c : ℝ, c ≠ 0 ∧
    ∃! b : ℝ, b > 0 ∧
      (∃! x : ℝ, quadratic b c x) ∧
      c = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_unique_c_for_unique_solution_l100_10046


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l100_10095

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: There are 243 ways to distribute 5 distinguishable balls into 3 distinguishable boxes -/
theorem five_balls_three_boxes : distribute_balls 5 3 = 243 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l100_10095


namespace NUMINAMATH_CALUDE_opposite_sign_sum_zero_l100_10031

theorem opposite_sign_sum_zero (a b : ℝ) : 
  (|a - 2| + (b + 1)^2 = 0) → (a - b = 3) := by
  sorry

end NUMINAMATH_CALUDE_opposite_sign_sum_zero_l100_10031


namespace NUMINAMATH_CALUDE_solve_linear_system_l100_10099

theorem solve_linear_system (a b : ℝ) 
  (eq1 : 3 * a + 2 * b = 5) 
  (eq2 : 2 * a + 3 * b = 4) : 
  a - b = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_system_l100_10099


namespace NUMINAMATH_CALUDE_ruined_tomatoes_percentage_l100_10042

/-- The percentage of ruined and discarded tomatoes -/
def ruined_percentage : ℝ := 15

/-- The purchase price per pound of tomatoes -/
def purchase_price : ℝ := 0.80

/-- The desired profit percentage on the cost of tomatoes -/
def profit_percentage : ℝ := 8

/-- The selling price per pound of remaining tomatoes -/
def selling_price : ℝ := 1.0165

/-- Theorem stating that given the purchase price, profit percentage, and selling price,
    the percentage of ruined and discarded tomatoes is approximately 15% -/
theorem ruined_tomatoes_percentage :
  ∀ (W : ℝ), W > 0 →
  selling_price * (100 - ruined_percentage) / 100 * W - purchase_price * W =
  profit_percentage / 100 * purchase_price * W :=
by sorry

end NUMINAMATH_CALUDE_ruined_tomatoes_percentage_l100_10042


namespace NUMINAMATH_CALUDE_original_number_proof_l100_10007

theorem original_number_proof (x : ℝ) : 
  x * 16 = 3408 → 0.16 * 2.13 = 0.3408 → x = 213 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l100_10007


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l100_10034

theorem necessary_but_not_sufficient (a : ℝ) :
  (∀ a, (0 < a ∧ a < 1) → (a + 1) * (a - 2) < 0) ∧
  (∃ a, (a + 1) * (a - 2) < 0 ∧ (a ≤ 0 ∨ a ≥ 1)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l100_10034


namespace NUMINAMATH_CALUDE_root_sum_absolute_value_l100_10094

theorem root_sum_absolute_value (m : ℤ) (a b c : ℤ) : 
  (∃ (m : ℤ), ∀ (x : ℤ), x^3 - 2023*x + m = 0 ↔ x = a ∨ x = b ∨ x = c) →
  |a| + |b| + |c| = 102 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_absolute_value_l100_10094


namespace NUMINAMATH_CALUDE_grid_path_theorem_l100_10016

/-- Represents a closed path on a grid that is not self-intersecting -/
structure GridPath (m n : ℕ) where
  -- Add necessary fields to represent the path

/-- Counts the number of points on the path where it does not turn -/
def count_no_turn_points (p : GridPath m n) : ℕ := sorry

/-- Counts the number of squares that the path goes through two non-adjacent sides -/
def count_two_side_squares (p : GridPath m n) : ℕ := sorry

/-- Counts the number of squares with no side in the path -/
def count_empty_squares (p : GridPath m n) : ℕ := sorry

theorem grid_path_theorem {m n : ℕ} (hm : m ≥ 4) (hn : n ≥ 4) (p : GridPath m n) :
  count_no_turn_points p = count_two_side_squares p - count_empty_squares p + m + n - 1 := by
  sorry

end NUMINAMATH_CALUDE_grid_path_theorem_l100_10016


namespace NUMINAMATH_CALUDE_tigers_wins_l100_10045

theorem tigers_wins (total_games : ℕ) (games_lost_more : ℕ) 
  (h1 : total_games = 120)
  (h2 : games_lost_more = 38) :
  let games_won := (total_games - games_lost_more) / 2
  games_won = 41 := by
sorry

end NUMINAMATH_CALUDE_tigers_wins_l100_10045


namespace NUMINAMATH_CALUDE_sqrt_meaningful_iff_x_gt_one_l100_10073

theorem sqrt_meaningful_iff_x_gt_one (x : ℝ) : 
  (∃ y : ℝ, y * y = 1 / (x - 1)) ↔ x > 1 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_iff_x_gt_one_l100_10073


namespace NUMINAMATH_CALUDE_largest_eight_digit_with_even_digits_l100_10084

def even_digits : List Nat := [0, 2, 4, 6, 8]

def is_eight_digit (n : Nat) : Prop :=
  n ≥ 10000000 ∧ n < 100000000

def contains_all_even_digits (n : Nat) : Prop :=
  ∀ d ∈ even_digits, ∃ k, n / (10^k) % 10 = d

theorem largest_eight_digit_with_even_digits :
  ∀ n : Nat, is_eight_digit n → contains_all_even_digits n →
  n ≤ 99986420 :=
by sorry

end NUMINAMATH_CALUDE_largest_eight_digit_with_even_digits_l100_10084


namespace NUMINAMATH_CALUDE_chocolate_milk_probability_l100_10025

theorem chocolate_milk_probability : 
  let n : ℕ := 7  -- total number of days
  let k : ℕ := 5  -- number of days with chocolate milk
  let p : ℚ := 3/4  -- probability of bottling chocolate milk on any given day
  Nat.choose n k * p^k * (1-p)^(n-k) = 5103/16384 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_milk_probability_l100_10025


namespace NUMINAMATH_CALUDE_first_number_in_ratio_l100_10048

theorem first_number_in_ratio (A B : ℕ) (h1 : A > 0) (h2 : B > 0) : 
  A * 4 = B * 5 → lcm A B = 80 → A = 10 := by
  sorry

end NUMINAMATH_CALUDE_first_number_in_ratio_l100_10048


namespace NUMINAMATH_CALUDE_smaller_number_proof_l100_10049

theorem smaller_number_proof (S L : ℕ) 
  (h1 : L - S = 2468) 
  (h2 : L = 8 * S + 27) : 
  S = 349 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l100_10049


namespace NUMINAMATH_CALUDE_min_cans_correct_l100_10051

/-- The volume of soda in a single can (in ounces) -/
def can_volume : ℝ := 15

/-- The conversion factor from liters to ounces -/
def liter_to_ounce : ℝ := 33.814

/-- The required volume of soda (in liters) -/
def required_volume : ℝ := 3.8

/-- The minimum number of cans required to provide at least the required volume of soda -/
def min_cans : ℕ := 9

/-- Theorem stating that the minimum number of cans required to provide at least
    the required volume of soda is 9 -/
theorem min_cans_correct :
  ∀ n : ℕ, (n : ℝ) * can_volume ≥ required_volume * liter_to_ounce → n ≥ min_cans :=
by sorry

end NUMINAMATH_CALUDE_min_cans_correct_l100_10051


namespace NUMINAMATH_CALUDE_cats_favorite_number_l100_10090

def is_two_digit_positive (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

def has_distinct_nonzero_digits (n : ℕ) : Prop :=
  let tens := n / 10
  let ones := n % 10
  tens ≠ ones ∧ tens ≠ 0 ∧ ones ≠ 0

def digits_are_factors (n : ℕ) : Prop :=
  let tens := n / 10
  let ones := n % 10
  n % tens = 0 ∧ n % ones = 0

def satisfies_four_number_property (a b c d : ℕ) : Prop :=
  a + b - c = d ∧ b + c - a = d ∧ c + d - b = a ∧ d + a - c = b

theorem cats_favorite_number :
  ∃! n : ℕ,
    is_two_digit_positive n ∧
    has_distinct_nonzero_digits n ∧
    digits_are_factors n ∧
    ∃ a b c : ℕ,
      satisfies_four_number_property n a b c ∧
      n^2 = a * b ∧
      (a ≠ n ∧ b ≠ n ∧ c ≠ n) :=
by
  sorry

end NUMINAMATH_CALUDE_cats_favorite_number_l100_10090


namespace NUMINAMATH_CALUDE_factorial_fraction_simplification_l100_10053

theorem factorial_fraction_simplification : 
  (4 * Nat.factorial 6 + 24 * Nat.factorial 5) / Nat.factorial 7 = 8 / 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_simplification_l100_10053


namespace NUMINAMATH_CALUDE_base_n_representation_of_d_l100_10088

/-- Represents a number in base n --/
structure BaseN (n : ℕ) where
  digits : List ℕ
  all_less : ∀ d ∈ digits, d < n

/-- Convert a base-n number to its decimal representation --/
def toDecimal (n : ℕ) (b : BaseN n) : ℕ :=
  b.digits.enum.foldl (fun acc (i, d) => acc + d * n ^ i) 0

theorem base_n_representation_of_d (n : ℕ) (c d : ℕ) :
  n > 8 →
  n ^ 2 - c * n + d = 0 →
  toDecimal n ⟨[2, 1], by sorry⟩ = c →
  toDecimal n ⟨[0, 1, 1], by sorry⟩ = d :=
by sorry

end NUMINAMATH_CALUDE_base_n_representation_of_d_l100_10088
