import Mathlib

namespace NUMINAMATH_CALUDE_interesting_iff_prime_power_l279_27994

def is_interesting (n : ℕ) : Prop :=
  n > 1 ∧ ∀ x y : ℕ, (Nat.gcd x n ≠ 1 ∧ Nat.gcd y n ≠ 1) → Nat.gcd (x + y) n ≠ 1

theorem interesting_iff_prime_power (n : ℕ) :
  is_interesting n ↔ ∃ p k : ℕ, Nat.Prime p ∧ k > 0 ∧ n = p^k :=
sorry

end NUMINAMATH_CALUDE_interesting_iff_prime_power_l279_27994


namespace NUMINAMATH_CALUDE_num_new_candles_l279_27910

/-- The amount of wax left in a candle as a percentage of its original weight -/
def waxLeftPercentage : ℚ := 1 / 10

/-- The weight of a large candle in ounces -/
def largeCandle : ℚ := 20

/-- The weight of a medium candle in ounces -/
def mediumCandle : ℚ := 5

/-- The weight of a small candle in ounces -/
def smallCandle : ℚ := 1

/-- The number of large candles -/
def numLargeCandles : ℕ := 5

/-- The number of medium candles -/
def numMediumCandles : ℕ := 5

/-- The number of small candles -/
def numSmallCandles : ℕ := 25

/-- The weight of a new candle to be made in ounces -/
def newCandleWeight : ℚ := 5

/-- Theorem: The number of new candles that can be made is 3 -/
theorem num_new_candles :
  (waxLeftPercentage * (numLargeCandles * largeCandle + 
                        numMediumCandles * mediumCandle + 
                        numSmallCandles * smallCandle)) / newCandleWeight = 3 := by
  sorry

end NUMINAMATH_CALUDE_num_new_candles_l279_27910


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_of_8_pow_2004_l279_27919

/-- The sum of the tens digit and the units digit in the decimal representation of 8^2004 -/
def sum_of_last_two_digits : ℕ :=
  let n : ℕ := 8^2004
  let tens_digit : ℕ := (n / 10) % 10
  let units_digit : ℕ := n % 10
  tens_digit + units_digit

theorem sum_of_last_two_digits_of_8_pow_2004 :
  sum_of_last_two_digits = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_of_8_pow_2004_l279_27919


namespace NUMINAMATH_CALUDE_basketball_time_l279_27937

theorem basketball_time (n : ℕ) (last_activity_time : ℝ) : 
  n = 5 ∧ last_activity_time = 160 →
  (let seq := fun i => (2 ^ i) * (last_activity_time / (2 ^ (n - 1)))
   seq 0 = 10) := by
  sorry

end NUMINAMATH_CALUDE_basketball_time_l279_27937


namespace NUMINAMATH_CALUDE_range_of_m_l279_27992

-- Define p and q as predicates on real numbers
def p (x : ℝ) : Prop := |x - 3| ≤ 2
def q (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≤ 0

-- Define the theorem
theorem range_of_m :
  (∀ x m : ℝ, (¬(p x) → ¬(q x m)) ∧ ¬(¬(q x m) → ¬(p x))) →
  ∃ a b : ℝ, a = 2 ∧ b = 4 ∧ ∀ m : ℝ, a ≤ m ∧ m ≤ b :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l279_27992


namespace NUMINAMATH_CALUDE_truck_driver_pay_l279_27973

/-- Calculates the pay for a round trip given the pay rate per mile and one-way distance -/
def round_trip_pay (pay_rate : ℚ) (one_way_distance : ℕ) : ℚ :=
  2 * pay_rate * one_way_distance

/-- Proves that given a pay rate of $0.40 per mile and a one-way trip distance of 400 miles,
    the total pay for a round trip is $320 -/
theorem truck_driver_pay : round_trip_pay (40/100) 400 = 320 := by
  sorry

end NUMINAMATH_CALUDE_truck_driver_pay_l279_27973


namespace NUMINAMATH_CALUDE_expo_visit_arrangements_l279_27924

/-- The number of schools visiting the Expo Park -/
def num_schools : ℕ := 10

/-- The number of days available for visits -/
def total_days : ℕ := 30

/-- The number of days required by the larger school -/
def large_school_days : ℕ := 2

/-- The number of days required by each of the other schools -/
def other_school_days : ℕ := 1

/-- The number of ways to arrange the school visits -/
def arrangement_count : ℕ := Nat.choose 29 1 * (Nat.factorial 28 / Nat.factorial (28 - 9))

theorem expo_visit_arrangements :
  arrangement_count = 
    Nat.choose (total_days - 1) 1 * 
    (Nat.factorial (total_days - large_school_days) / 
     Nat.factorial (total_days - large_school_days - (num_schools - 1))) :=
by sorry

end NUMINAMATH_CALUDE_expo_visit_arrangements_l279_27924


namespace NUMINAMATH_CALUDE_quadratic_root_expression_l279_27912

theorem quadratic_root_expression (x₁ x₂ : ℝ) : 
  x₁^2 - 4*x₁ + 2 = 0 →
  x₂^2 - 4*x₂ + 2 = 0 →
  x₁ + x₂ = 4 →
  x₁ * x₂ = 2 →
  x₁^2 - 4*x₁ + 2*x₁*x₂ = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_expression_l279_27912


namespace NUMINAMATH_CALUDE_intersection_M_N_l279_27968

def M : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

def N : Set ℕ := {x | Real.sqrt (2^x - 1) < 5}

theorem intersection_M_N : M ∩ N = {1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l279_27968


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l279_27983

/-- Represents the number of male students in a stratified sample -/
def male_students_in_sample (total_male : ℕ) (total_female : ℕ) (sample_size : ℕ) : ℕ :=
  (total_male * sample_size) / (total_male + total_female)

/-- Theorem: In a school with 560 male students and 420 female students,
    a stratified sample of 140 students will contain 80 male students -/
theorem stratified_sample_theorem :
  male_students_in_sample 560 420 140 = 80 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_theorem_l279_27983


namespace NUMINAMATH_CALUDE_jeremys_beads_l279_27911

theorem jeremys_beads (n : ℕ) : n > 1 ∧ 
  n % 5 = 2 ∧ 
  n % 7 = 2 ∧ 
  n % 9 = 2 → 
  (∀ m : ℕ, m > 1 ∧ m % 5 = 2 ∧ m % 7 = 2 ∧ m % 9 = 2 → m ≥ n) →
  n = 317 := by
sorry

end NUMINAMATH_CALUDE_jeremys_beads_l279_27911


namespace NUMINAMATH_CALUDE_system_solution_l279_27954

-- Define the system of equations
def system_equations (x₁ x₂ x₃ x₄ : ℝ) : Prop :=
  (x₁ + x₂*x₃*x₄ = 2) ∧
  (x₂ + x₁*x₃*x₄ = 2) ∧
  (x₃ + x₁*x₂*x₄ = 2) ∧
  (x₄ + x₁*x₂*x₃ = 2)

-- Define the set of solutions
def solution_set : Set (ℝ × ℝ × ℝ × ℝ) :=
  {(1, 1, 1, 1), (-1, -1, -1, 3), (-1, -1, 3, -1), (-1, 3, -1, -1), (3, -1, -1, -1)}

-- Theorem statement
theorem system_solution :
  ∀ x₁ x₂ x₃ x₄ : ℝ, system_equations x₁ x₂ x₃ x₄ ↔ (x₁, x₂, x₃, x₄) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_system_solution_l279_27954


namespace NUMINAMATH_CALUDE_brittany_rebecca_age_difference_l279_27974

/-- The age difference between Brittany and Rebecca -/
def ageDifference (rebecca_age : ℕ) (brittany_age_after_vacation : ℕ) (vacation_duration : ℕ) : ℕ :=
  brittany_age_after_vacation - vacation_duration - rebecca_age

/-- Proof that Brittany is 3 years older than Rebecca -/
theorem brittany_rebecca_age_difference :
  ageDifference 25 32 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_brittany_rebecca_age_difference_l279_27974


namespace NUMINAMATH_CALUDE_max_x_value_l279_27961

theorem max_x_value (x : ℝ) : 
  ((4*x - 16)/(3*x - 4))^2 + ((4*x - 16)/(3*x - 4)) = 12 → x ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_x_value_l279_27961


namespace NUMINAMATH_CALUDE_even_and_increasing_order_l279_27989

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f x < f y

theorem even_and_increasing_order (f : ℝ → ℝ) 
  (h_even : is_even f) (h_incr : increasing_on_nonneg f) : 
  f (-2) < f 3 ∧ f 3 < f (-π) := by
  sorry

end NUMINAMATH_CALUDE_even_and_increasing_order_l279_27989


namespace NUMINAMATH_CALUDE_cubic_function_properties_l279_27925

/-- A cubic function with parameters m and n -/
def f (m n x : ℝ) : ℝ := x^3 + m*x^2 + n*x

/-- The derivative of f with respect to x -/
def f' (m n x : ℝ) : ℝ := 3*x^2 + 2*m*x + n

theorem cubic_function_properties (m n : ℝ) :
  (∀ x, f' m n x ≤ f' m n 1) →
  (f' m n 1 = 0 ∧ ∃! (a b : ℝ), a ≠ b ∧ 
    ∃ (t : ℝ), f m n t = a*t + (1 - a) ∧
    f m n t = b*t + (1 - b)) →
  (m < -3 ∧ m = -3) :=
sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l279_27925


namespace NUMINAMATH_CALUDE_initial_men_employed_is_300_l279_27995

/-- Represents the highway construction scenario --/
structure HighwayConstruction where
  totalLength : ℝ
  totalDays : ℕ
  initialHoursPerDay : ℕ
  daysWorked : ℕ
  workCompleted : ℝ
  additionalMen : ℕ
  newHoursPerDay : ℕ

/-- Calculates the initial number of men employed --/
def initialMenEmployed (h : HighwayConstruction) : ℕ :=
  sorry

/-- Theorem stating that the initial number of men employed is 300 --/
theorem initial_men_employed_is_300 (h : HighwayConstruction) 
  (h_total_length : h.totalLength = 2)
  (h_total_days : h.totalDays = 50)
  (h_initial_hours : h.initialHoursPerDay = 8)
  (h_days_worked : h.daysWorked = 25)
  (h_work_completed : h.workCompleted = 1/3)
  (h_additional_men : h.additionalMen = 60)
  (h_new_hours : h.newHoursPerDay = 10) :
  initialMenEmployed h = 300 :=
sorry

end NUMINAMATH_CALUDE_initial_men_employed_is_300_l279_27995


namespace NUMINAMATH_CALUDE_tan_ratio_equals_two_l279_27906

theorem tan_ratio_equals_two (a β : ℝ) (h : 3 * Real.sin β = Real.sin (2 * a + β)) :
  Real.tan (a + β) / Real.tan a = 2 := by sorry

end NUMINAMATH_CALUDE_tan_ratio_equals_two_l279_27906


namespace NUMINAMATH_CALUDE_isabellas_hair_growth_l279_27940

/-- Given Isabella's initial and final hair lengths, prove that her hair growth is 6 inches. -/
theorem isabellas_hair_growth 
  (initial_length : ℝ) 
  (final_length : ℝ) 
  (h1 : initial_length = 18) 
  (h2 : final_length = 24) : 
  final_length - initial_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_hair_growth_l279_27940


namespace NUMINAMATH_CALUDE_equation_solutions_l279_27947

theorem equation_solutions : 
  let f (r : ℝ) := (r^2 - 6*r + 9) / (r^2 - 9*r + 14)
  let g (r : ℝ) := (r^2 - 4*r - 21) / (r^2 - 2*r - 35)
  ∀ r : ℝ, f r = g r ↔ (r = 3 ∨ r = (-1 + Real.sqrt 69) / 2 ∨ r = (-1 - Real.sqrt 69) / 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l279_27947


namespace NUMINAMATH_CALUDE_min_value_theorem_l279_27902

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 27) :
  ∃ (min : ℝ), min = 30 ∧ ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a * b * c = 27 → 
    a^2 + 3*b + 6*c ≥ min := by
  sorry

#check min_value_theorem

end NUMINAMATH_CALUDE_min_value_theorem_l279_27902


namespace NUMINAMATH_CALUDE_infinite_series_sum_l279_27932

theorem infinite_series_sum : 
  (∑' n : ℕ+, (1 : ℝ) / (n * (n + 3))) = 11 / 18 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l279_27932


namespace NUMINAMATH_CALUDE_wine_equation_correctness_l279_27933

/-- Represents the wine consumption and intoxication scenario --/
def wine_scenario (x y : ℚ) : Prop :=
  -- Total bottles of wine
  x + y = 19 ∧
  -- Intoxication effect
  3 * x + (1/3) * y = 33 ∧
  -- x represents good wine bottles
  x ≥ 0 ∧
  -- y represents inferior wine bottles
  y ≥ 0

/-- The system of equations correctly represents the wine scenario --/
theorem wine_equation_correctness :
  ∃ x y : ℚ, wine_scenario x y :=
sorry

end NUMINAMATH_CALUDE_wine_equation_correctness_l279_27933


namespace NUMINAMATH_CALUDE_largest_lcm_with_15_l279_27916

theorem largest_lcm_with_15 :
  let lcm_list := [lcm 15 3, lcm 15 5, lcm 15 6, lcm 15 9, lcm 15 10, lcm 15 12]
  List.maximum lcm_list = some 60 := by
  sorry

end NUMINAMATH_CALUDE_largest_lcm_with_15_l279_27916


namespace NUMINAMATH_CALUDE_apple_solution_l279_27900

/-- The number of apples each person has. -/
structure Apples where
  rebecca : ℕ
  jackie : ℕ
  adam : ℕ

/-- The conditions of the apple distribution problem. -/
def AppleConditions (a : Apples) : Prop :=
  a.rebecca = 2 * a.jackie ∧
  a.adam = a.jackie + 3 ∧
  a.adam = 9

/-- The solution to the apple distribution problem. -/
theorem apple_solution (a : Apples) (h : AppleConditions a) : a.jackie = 6 ∧ a.rebecca = 12 := by
  sorry


end NUMINAMATH_CALUDE_apple_solution_l279_27900


namespace NUMINAMATH_CALUDE_sqrt_of_four_l279_27944

-- Define the square root function
def sqrt (x : ℝ) : Set ℝ := {y : ℝ | y^2 = x}

-- Theorem statement
theorem sqrt_of_four : sqrt 4 = {2, -2} := by sorry

end NUMINAMATH_CALUDE_sqrt_of_four_l279_27944


namespace NUMINAMATH_CALUDE_stack_weight_error_l279_27914

/-- The weight of a disc with exactly 1 meter diameter in kg -/
def standard_weight : ℝ := 100

/-- The nominal radius of a disc in meters -/
def nominal_radius : ℝ := 0.5

/-- The standard deviation of the radius in meters -/
def radius_std_dev : ℝ := 0.01

/-- The number of discs in the stack -/
def num_discs : ℕ := 100

/-- The expected weight of a single disc given the manufacturing variation -/
def expected_single_disc_weight : ℝ := sorry

/-- The expected weight of the stack of discs -/
def expected_stack_weight : ℝ := sorry

/-- Engineer Sidorov's estimate of the stack weight -/
def sidorov_estimate : ℝ := 10000

theorem stack_weight_error :
  expected_stack_weight - sidorov_estimate = 4 := by sorry

end NUMINAMATH_CALUDE_stack_weight_error_l279_27914


namespace NUMINAMATH_CALUDE_uncle_jude_cookies_l279_27980

/-- The number of cookies Uncle Jude baked -/
def total_cookies : ℕ := 256

/-- The number of cookies given to Tim -/
def tim_cookies : ℕ := 15

/-- The number of cookies given to Mike -/
def mike_cookies : ℕ := 23

/-- The number of cookies kept in the fridge -/
def fridge_cookies : ℕ := 188

/-- The number of cookies given to Anna -/
def anna_cookies : ℕ := 2 * tim_cookies

theorem uncle_jude_cookies : 
  total_cookies = tim_cookies + mike_cookies + anna_cookies + fridge_cookies := by
  sorry

end NUMINAMATH_CALUDE_uncle_jude_cookies_l279_27980


namespace NUMINAMATH_CALUDE_f_3_equals_130_l279_27926

def f (x : ℝ) : ℝ := 3 * x^4 - 5 * x^3 + 2 * x^2 - x + 7

theorem f_3_equals_130 : f 3 = 130 := by
  sorry

end NUMINAMATH_CALUDE_f_3_equals_130_l279_27926


namespace NUMINAMATH_CALUDE_circle_reflection_translation_l279_27915

/-- Given a point (3, -4), prove that reflecting it across the x-axis
    and then translating it 5 units to the left results in the point (-2, 4) -/
theorem circle_reflection_translation :
  let initial_point : ℝ × ℝ := (3, -4)
  let reflected_point : ℝ × ℝ := (initial_point.1, -initial_point.2)
  let final_point : ℝ × ℝ := (reflected_point.1 - 5, reflected_point.2)
  final_point = (-2, 4) := by
sorry

end NUMINAMATH_CALUDE_circle_reflection_translation_l279_27915


namespace NUMINAMATH_CALUDE_platform_length_l279_27922

/-- The length of a train platform given crossing times and known lengths -/
theorem platform_length 
  (train_length : ℝ) 
  (platform2_length : ℝ) 
  (time1 : ℝ) 
  (time2 : ℝ) 
  (h1 : train_length = 310)
  (h2 : platform2_length = 250)
  (h3 : time1 = 15)
  (h4 : time2 = 20) :
  ∃ (platform1_length : ℝ), 
    (train_length + platform1_length) / (train_length + platform2_length) = time1 / time2 ∧ 
    platform1_length = 110 := by
sorry

end NUMINAMATH_CALUDE_platform_length_l279_27922


namespace NUMINAMATH_CALUDE_kevin_kangaroo_hops_l279_27941

theorem kevin_kangaroo_hops (n : ℕ) (a : ℚ) (r : ℚ) : 
  n = 7 ∧ a = 1 ∧ r = 3/4 → 
  4 * (a * (1 - r^n) / (1 - r)) = 7086/2048 := by
sorry

end NUMINAMATH_CALUDE_kevin_kangaroo_hops_l279_27941


namespace NUMINAMATH_CALUDE_complex_rectangle_perimeter_l279_27997

/-- A structure representing a rectangle with an internal complex shape. -/
structure ComplexRectangle where
  width : ℝ
  height : ℝ
  enclosed_area : ℝ

/-- The perimeter of a ComplexRectangle is equal to 2 * (width + height) -/
def perimeter (r : ComplexRectangle) : ℝ := 2 * (r.width + r.height)

theorem complex_rectangle_perimeter :
  ∀ (r : ComplexRectangle),
    r.width = 15 ∧ r.height = 10 ∧ r.enclosed_area = 108 →
    perimeter r = 50 := by
  sorry

end NUMINAMATH_CALUDE_complex_rectangle_perimeter_l279_27997


namespace NUMINAMATH_CALUDE_consecutive_integers_sqrt_3_l279_27936

theorem consecutive_integers_sqrt_3 (a b : ℤ) : 
  (b = a + 1) → (↑a < Real.sqrt 3 ∧ Real.sqrt 3 < ↑b) → a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sqrt_3_l279_27936


namespace NUMINAMATH_CALUDE_equation_proof_l279_27934

theorem equation_proof : (36 / 18) * (36 / 72) = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l279_27934


namespace NUMINAMATH_CALUDE_math_grade_calculation_l279_27928

theorem math_grade_calculation (history_grade third_subject_grade : ℝ) 
  (h1 : history_grade = 84)
  (h2 : third_subject_grade = 67)
  (h3 : (math_grade + history_grade + third_subject_grade) / 3 = 75) :
  math_grade = 74 := by
sorry

end NUMINAMATH_CALUDE_math_grade_calculation_l279_27928


namespace NUMINAMATH_CALUDE_balloon_count_l279_27945

/-- Represents the number of balloons of each color and their arrangement --/
structure BalloonArrangement where
  red : Nat
  yellow : Nat
  blue : Nat
  yellow_spaces : Nat
  yellow_unfilled : Nat

/-- Calculates the total number of balloons --/
def total_balloons (arrangement : BalloonArrangement) : Nat :=
  arrangement.red + arrangement.yellow + arrangement.blue

/-- Theorem stating the correct number of yellow and blue balloons --/
theorem balloon_count (arrangement : BalloonArrangement) 
  (h1 : arrangement.red = 40)
  (h2 : arrangement.yellow_spaces = arrangement.red - 1)
  (h3 : arrangement.yellow_unfilled = 3)
  (h4 : arrangement.yellow = arrangement.yellow_spaces + arrangement.yellow_unfilled)
  (h5 : arrangement.blue = total_balloons arrangement - 1) :
  arrangement.yellow = 42 ∧ arrangement.blue = 81 := by
  sorry

#check balloon_count

end NUMINAMATH_CALUDE_balloon_count_l279_27945


namespace NUMINAMATH_CALUDE_inequality_solution_l279_27931

theorem inequality_solution (x : ℝ) : 
  (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2)) < 1 / 3) ↔ 
  (x < -2 ∨ (-1 < x ∧ x < 0) ∨ 1 < x) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l279_27931


namespace NUMINAMATH_CALUDE_paige_homework_problems_l279_27903

/-- The number of math problems Paige had for homework -/
def math_problems : ℕ := 43

/-- The number of science problems Paige had for homework -/
def science_problems : ℕ := 12

/-- The number of problems Paige finished at school -/
def finished_problems : ℕ := 44

/-- The number of problems Paige had to do for homework -/
def homework_problems : ℕ := math_problems + science_problems - finished_problems

theorem paige_homework_problems :
  homework_problems = 11 := by sorry

end NUMINAMATH_CALUDE_paige_homework_problems_l279_27903


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l279_27967

theorem opposite_of_negative_2023 : 
  (-((-2023 : ℝ)) = (2023 : ℝ)) := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l279_27967


namespace NUMINAMATH_CALUDE_prob_at_least_one_ace_value_l279_27953

/-- The number of cards in two standard decks -/
def total_cards : ℕ := 104

/-- The number of aces in two standard decks -/
def total_aces : ℕ := 8

/-- The probability of drawing at least one ace when two cards are chosen
    sequentially with replacement from a deck of two standard decks -/
def prob_at_least_one_ace : ℚ :=
  1 - (1 - total_aces / total_cards) ^ 2

theorem prob_at_least_one_ace_value :
  prob_at_least_one_ace = 25 / 169 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_ace_value_l279_27953


namespace NUMINAMATH_CALUDE_cricket_players_l279_27959

theorem cricket_players (total : ℕ) (football : ℕ) (neither : ℕ) (both : ℕ) 
  (h1 : total = 460)
  (h2 : football = 325)
  (h3 : neither = 50)
  (h4 : both = 90) :
  ∃ cricket : ℕ, cricket = 175 ∧ 
  cricket = total - neither - (football - both) := by
  sorry

end NUMINAMATH_CALUDE_cricket_players_l279_27959


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l279_27952

/-- A line passing through (4,1) with equal intercepts on both axes -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line passes through (4,1) -/
  point_condition : m * 4 + b = 1
  /-- The line has equal intercepts on both axes -/
  equal_intercepts : b ≠ 0 → -b = b / m

/-- The equation of an EqualInterceptLine is either x - 4y = 0 or x + y - 5 = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.m = 1/4 ∧ l.b = 0) ∨ (l.m = -1 ∧ l.b = 5) := by
  sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l279_27952


namespace NUMINAMATH_CALUDE_bottle_capacity_ratio_l279_27962

theorem bottle_capacity_ratio (c1 c2 : ℝ) : 
  c1 > 0 ∧ c2 > 0 →  -- Capacities are positive
  c1 / 2 + c2 / 4 = (c1 + c2) / 3 →  -- Oil is 1/3 of total mixture
  c2 / c1 = 2 := by sorry

end NUMINAMATH_CALUDE_bottle_capacity_ratio_l279_27962


namespace NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l279_27918

-- Define the repeating decimal 4.8̄
def repeating_decimal : ℚ := 4 + 8/9

-- Theorem to prove
theorem repeating_decimal_as_fraction : repeating_decimal = 44/9 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l279_27918


namespace NUMINAMATH_CALUDE_complex_modulus_l279_27951

theorem complex_modulus (z : ℂ) : z = (1 - 2*I) / (1 - I) → Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l279_27951


namespace NUMINAMATH_CALUDE_binomial_coefficient_congruence_l279_27999

theorem binomial_coefficient_congruence (p a b : ℕ) : 
  Nat.Prime p → p > 3 → a > b → b > 1 → 
  (Nat.choose (a * p) (b * p)) ≡ (Nat.choose a b) [MOD p^3] := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_congruence_l279_27999


namespace NUMINAMATH_CALUDE_equation_characterizes_triangles_l279_27946

/-- A triangle with sides a, b, and c. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The equation given in the problem. -/
def satisfies_equation (t : Triangle) : Prop :=
  (t.c^2 - t.a^2) / t.b + (t.b^2 - t.c^2) / t.a = t.b - t.a

/-- A right-angled triangle. -/
def is_right_angled (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2 ∨ t.b^2 + t.c^2 = t.a^2 ∨ t.c^2 + t.a^2 = t.b^2

/-- An isosceles triangle. -/
def is_isosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

/-- The main theorem. -/
theorem equation_characterizes_triangles (t : Triangle) :
  satisfies_equation t ↔ is_right_angled t ∨ is_isosceles t := by
  sorry

end NUMINAMATH_CALUDE_equation_characterizes_triangles_l279_27946


namespace NUMINAMATH_CALUDE_point_transformation_difference_l279_27993

-- Define the rotation and reflection transformations
def rotate90CounterClockwise (center x : ℝ × ℝ) : ℝ × ℝ :=
  let (cx, cy) := center
  let (px, py) := x
  (cx - (py - cy), cy + (px - cx))

def reflectAboutYEqualNegX (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (-y, -x)

-- State the theorem
theorem point_transformation_difference (a b : ℝ) :
  let p : ℝ × ℝ := (a, b)
  let rotated := rotate90CounterClockwise (2, 6) p
  let final := reflectAboutYEqualNegX rotated
  final = (-5, 2) → b - a = 15 := by
  sorry


end NUMINAMATH_CALUDE_point_transformation_difference_l279_27993


namespace NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_min_value_is_nine_l279_27986

theorem min_value_sum_of_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (4^a * 2^b)) :
  ∀ x y, x > 0 → y > 0 → 2/a + 1/b ≤ 2/x + 1/y :=
by
  sorry

theorem min_value_is_nine (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (4^a * 2^b)) :
  ∃ x y, x > 0 ∧ y > 0 ∧ 2/x + 1/y = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_min_value_is_nine_l279_27986


namespace NUMINAMATH_CALUDE_stadium_length_conversion_l279_27988

/-- Converts yards to feet given the number of yards and the conversion factor. -/
def yards_to_feet (yards : ℕ) (conversion_factor : ℕ) : ℕ :=
  yards * conversion_factor

/-- Proves that 62 yards is equal to 186 feet when converted. -/
theorem stadium_length_conversion :
  let stadium_length_yards : ℕ := 62
  let yards_to_feet_conversion : ℕ := 3
  yards_to_feet stadium_length_yards yards_to_feet_conversion = 186 := by
  sorry

#check stadium_length_conversion

end NUMINAMATH_CALUDE_stadium_length_conversion_l279_27988


namespace NUMINAMATH_CALUDE_specific_conference_handshakes_l279_27913

/-- The number of distinct handshakes in a conference --/
def conference_handshakes (num_companies : ℕ) (reps_per_company : ℕ) : ℕ :=
  let total_people := num_companies * reps_per_company
  let handshakes_per_person := total_people - reps_per_company
  (total_people * handshakes_per_person) / 2

/-- Theorem stating the number of handshakes for the specific conference scenario --/
theorem specific_conference_handshakes :
  conference_handshakes 3 5 = 75 := by
  sorry

#eval conference_handshakes 3 5

end NUMINAMATH_CALUDE_specific_conference_handshakes_l279_27913


namespace NUMINAMATH_CALUDE_twice_x_minus_y_negative_l279_27976

theorem twice_x_minus_y_negative (x y : ℝ) : 
  (2 * x - y < 0) ↔ (∃ z : ℝ, z < 0 ∧ 2 * x - y = z) :=
sorry

end NUMINAMATH_CALUDE_twice_x_minus_y_negative_l279_27976


namespace NUMINAMATH_CALUDE_remainder_14_power_53_mod_7_l279_27935

theorem remainder_14_power_53_mod_7 : 14^53 % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_14_power_53_mod_7_l279_27935


namespace NUMINAMATH_CALUDE_two_car_garage_count_l279_27938

theorem two_car_garage_count (total : ℕ) (pool : ℕ) (both : ℕ) (neither : ℕ) :
  total = 70 →
  pool = 40 →
  both = 35 →
  neither = 15 →
  ∃ garage : ℕ, garage = 50 ∧ garage + pool - both + neither = total :=
by sorry

end NUMINAMATH_CALUDE_two_car_garage_count_l279_27938


namespace NUMINAMATH_CALUDE_net_population_increase_l279_27982

/-- The net population increase in one day given specific birth and death rates -/
theorem net_population_increase (birth_rate : ℚ) (death_rate : ℚ) (seconds_per_day : ℕ) : 
  birth_rate = 5 / 2 → death_rate = 3 / 2 → seconds_per_day = 24 * 60 * 60 →
  (birth_rate - death_rate) * seconds_per_day = 86400 := by
  sorry

end NUMINAMATH_CALUDE_net_population_increase_l279_27982


namespace NUMINAMATH_CALUDE_triangle_base_and_area_l279_27960

/-- Proves the base and area of a triangle given its height and the ratio of height to base -/
theorem triangle_base_and_area (h : ℝ) (ratio : ℝ) : 
  h = 12 → ratio = 2/3 → h = ratio * (h / ratio) → 
  let b := h / ratio
  let A := (1/2) * b * h
  b = 18 ∧ A = 108 := by
  sorry

#check triangle_base_and_area

end NUMINAMATH_CALUDE_triangle_base_and_area_l279_27960


namespace NUMINAMATH_CALUDE_unique_base_for_good_number_l279_27998

def is_good_number (m : ℕ) : Prop :=
  ∃ (p n : ℕ), n ≥ 2 ∧ Nat.Prime p ∧ m = p^n

theorem unique_base_for_good_number :
  ∀ b : ℕ, (is_good_number (b^2 - 2*b - 3)) ↔ b = 7 :=
by sorry

end NUMINAMATH_CALUDE_unique_base_for_good_number_l279_27998


namespace NUMINAMATH_CALUDE_probability_point_on_subsegment_l279_27917

/-- The probability of a randomly chosen point on a segment also lying on its subsegment -/
theorem probability_point_on_subsegment 
  (L ℓ : ℝ) 
  (hL : L = 40) 
  (hℓ : ℓ = 15) 
  (h_pos_L : L > 0) 
  (h_pos_ℓ : ℓ > 0) 
  (h_subsegment : ℓ ≤ L) :
  ℓ / L = 3 / 8 :=
sorry

end NUMINAMATH_CALUDE_probability_point_on_subsegment_l279_27917


namespace NUMINAMATH_CALUDE_abs_diff_plus_smaller_l279_27939

theorem abs_diff_plus_smaller (a b : ℝ) (h : a > b) : |a - b| + b = a := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_plus_smaller_l279_27939


namespace NUMINAMATH_CALUDE_jack_jogging_speed_l279_27908

/-- Represents the problem of Jack jogging to the beach with an ice cream cone -/
theorem jack_jogging_speed 
  (normal_melt_time : ℝ) 
  (wind_speed : ℝ) 
  (temperature : ℝ) 
  (melt_rate_increase : ℝ) 
  (num_blocks : ℕ) 
  (block_length : ℝ) 
  (slope_percent : ℝ) 
  (speed_reduction : ℝ) :
  normal_melt_time = 10 →
  wind_speed = 15 →
  temperature = 85 →
  melt_rate_increase = 0.25 →
  num_blocks = 16 →
  block_length = 1/8 →
  slope_percent = 5 →
  speed_reduction = 0.2 →
  ∃ (required_speed : ℝ), 
    required_speed = 20 ∧ 
    required_speed * (1 - speed_reduction) * (normal_melt_time * (1 - melt_rate_increase) / 60) ≥ 
    (num_blocks : ℝ) * block_length :=
by sorry

end NUMINAMATH_CALUDE_jack_jogging_speed_l279_27908


namespace NUMINAMATH_CALUDE_g_inv_composition_l279_27907

/-- Function g defined on a finite domain -/
def g : Fin 5 → Fin 5
| 0 => 3  -- represents g(1) = 4
| 1 => 4  -- represents g(2) = 5
| 2 => 1  -- represents g(3) = 2
| 3 => 2  -- represents g(4) = 3
| 4 => 0  -- represents g(5) = 1

/-- g is bijective -/
axiom g_bijective : Function.Bijective g

/-- The inverse function of g -/
noncomputable def g_inv : Fin 5 → Fin 5 := Function.invFun g

/-- Theorem stating that g^(-1)(g^(-1)(g^(-1)(4))) = 2 -/
theorem g_inv_composition :
  g_inv (g_inv (g_inv 3)) = 1 := by sorry

end NUMINAMATH_CALUDE_g_inv_composition_l279_27907


namespace NUMINAMATH_CALUDE_movie_screening_attendance_l279_27943

theorem movie_screening_attendance (total_guests : ℕ) 
  (h1 : total_guests = 50)
  (h2 : ∃ women : ℕ, women = total_guests / 2)
  (h3 : ∃ men : ℕ, men = 15)
  (h4 : ∃ children : ℕ, children = total_guests - (total_guests / 2 + 15))
  (h5 : ∃ men_left : ℕ, men_left = 15 / 5)
  (h6 : ∃ children_left : ℕ, children_left = 4) :
  total_guests - (15 / 5 + 4) = 43 := by
sorry


end NUMINAMATH_CALUDE_movie_screening_attendance_l279_27943


namespace NUMINAMATH_CALUDE_unique_pair_sum_and_quotient_l279_27981

theorem unique_pair_sum_and_quotient :
  ∃! (x y : ℕ), x + y = 2015 ∧ ∃ (s : ℕ), x = 25 * y + s ∧ s < y := by
  sorry

end NUMINAMATH_CALUDE_unique_pair_sum_and_quotient_l279_27981


namespace NUMINAMATH_CALUDE_triangle_problem_l279_27984

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a - b = 2 →
  c = 4 →
  Real.sin A = 2 * Real.sin B →
  (a = 4 ∧ b = 2) ∧
  Real.cos B = 7/8 ∧
  Real.sin (2*B - π/6) = (21 * Real.sqrt 5 - 17) / 64 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l279_27984


namespace NUMINAMATH_CALUDE_fish_population_estimation_l279_27970

/-- Calculates the number of fish in a lake on May 1st given certain conditions --/
theorem fish_population_estimation (marked_may : ℕ) (caught_sept : ℕ) (marked_sept : ℕ)
  (death_rate : ℚ) (new_fish_rate : ℚ) :
  marked_may = 60 →
  caught_sept = 70 →
  marked_sept = 3 →
  death_rate = 1/4 →
  new_fish_rate = 2/5 →
  ∃ (fish_may : ℕ), fish_may = 840 ∧ 
    (fish_may : ℚ) * (1 - death_rate) * (marked_sept : ℚ) / (caught_sept : ℚ) = 
    (marked_may : ℚ) * (1 - death_rate) ∧
    (fish_may : ℚ) * (1 - death_rate) / (1 - new_fish_rate) = 
    (fish_may : ℚ) * (1 - death_rate) * (marked_sept : ℚ) / (caught_sept : ℚ) / (1 - new_fish_rate) :=
by
  sorry


end NUMINAMATH_CALUDE_fish_population_estimation_l279_27970


namespace NUMINAMATH_CALUDE_probability_of_drawing_ball_two_l279_27949

/-- A box containing labeled balls. -/
structure Box where
  balls : Finset ℕ
  labels_distinct : balls.card = balls.toList.length

/-- The probability of drawing a specific ball from a box. -/
def probability_of_drawing (box : Box) (ball : ℕ) : ℚ :=
  if ball ∈ box.balls then 1 / box.balls.card else 0

/-- Theorem stating the probability of drawing ball 2 from a box with 3 balls labeled 1, 2, and 3. -/
theorem probability_of_drawing_ball_two :
  ∃ (box : Box), box.balls = {1, 2, 3} ∧ probability_of_drawing box 2 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_drawing_ball_two_l279_27949


namespace NUMINAMATH_CALUDE_function_is_identity_l279_27957

open Real

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, f (x^2 + y * f z + 1) = x * f x + z * f y + 1

theorem function_is_identity (f : ℝ → ℝ) (h : FunctionalEquation f) :
  (∀ x : ℝ, f x = x) ∧ f 5 = 5 := by
  sorry


end NUMINAMATH_CALUDE_function_is_identity_l279_27957


namespace NUMINAMATH_CALUDE_decimal_49_to_binary_l279_27972

/-- The binary representation of a natural number -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- Convert a list of booleans to a natural number in binary representation -/
def fromBinary (l : List Bool) : ℕ :=
  l.foldr (fun b n => 2 * n + if b then 1 else 0) 0

theorem decimal_49_to_binary :
  toBinary 49 = [true, true, false, false, false, true] :=
by sorry

end NUMINAMATH_CALUDE_decimal_49_to_binary_l279_27972


namespace NUMINAMATH_CALUDE_otimes_equation_solution_l279_27965

-- Define the custom operation
noncomputable def otimes (a b : ℝ) : ℝ :=
  a * Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- Theorem statement
theorem otimes_equation_solution :
  ∃ z : ℝ, otimes 3 z = 27 ∧ z = 72 :=
by sorry

end NUMINAMATH_CALUDE_otimes_equation_solution_l279_27965


namespace NUMINAMATH_CALUDE_interest_equality_implies_second_sum_l279_27920

/-- Proves that given a total sum of 2678, if the interest on the first part for 8 years at 3% per annum
    is equal to the interest on the second part for 3 years at 5% per annum, then the second part is 1648. -/
theorem interest_equality_implies_second_sum (total : ℝ) (first : ℝ) (second : ℝ) :
  total = 2678 →
  first + second = total →
  (first * (3/100) * 8) = (second * (5/100) * 3) →
  second = 1648 := by
sorry

end NUMINAMATH_CALUDE_interest_equality_implies_second_sum_l279_27920


namespace NUMINAMATH_CALUDE_correct_calculation_l279_27901

theorem correct_calculation (x y : ℝ) : 3 * x * y - 2 * y * x = x * y := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l279_27901


namespace NUMINAMATH_CALUDE_picture_placement_l279_27929

theorem picture_placement (wall_width picture_width : ℝ) 
  (hw : wall_width = 22)
  (hp : picture_width = 4) : 
  (wall_width - picture_width) / 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_picture_placement_l279_27929


namespace NUMINAMATH_CALUDE_negation_of_nonnegative_product_l279_27979

theorem negation_of_nonnegative_product (a b : ℝ) :
  ¬(a ≥ 0 ∧ b ≥ 0 → a * b ≥ 0) ↔ (a < 0 ∨ b < 0 → a * b < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_nonnegative_product_l279_27979


namespace NUMINAMATH_CALUDE_combined_mean_score_l279_27971

/-- Given two classes with different average scores and a ratio of students, 
    calculate the combined mean score. -/
theorem combined_mean_score (avg1 avg2 : ℝ) (ratio1 ratio2 : ℕ) : 
  avg1 = 90 →
  avg2 = 75 →
  ratio1 = 2 →
  ratio2 = 3 →
  (avg1 * ratio1 + avg2 * ratio2) / (ratio1 + ratio2) = 81 := by
sorry

end NUMINAMATH_CALUDE_combined_mean_score_l279_27971


namespace NUMINAMATH_CALUDE_x_value_l279_27942

theorem x_value : ∃ x : ℝ, x ≠ 0 ∧ x = 3 * (1 / x * (-x)) + 3 → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l279_27942


namespace NUMINAMATH_CALUDE_max_green_socks_l279_27978

/-- Represents the number of socks in a drawer -/
structure SockDrawer where
  green : ℕ
  yellow : ℕ
  total_bound : green + yellow ≤ 2500

/-- The probability of choosing two socks of the same color -/
def same_color_probability (d : SockDrawer) : ℚ :=
  let t := d.green + d.yellow
  (d.green * (d.green - 1) + d.yellow * (d.yellow - 1)) / (t * (t - 1))

/-- The theorem stating the maximum number of green socks possible -/
theorem max_green_socks (d : SockDrawer) 
  (h : same_color_probability d = 2/3) : 
  d.green ≤ 1275 ∧ ∃ d' : SockDrawer, d'.green = 1275 ∧ same_color_probability d' = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_max_green_socks_l279_27978


namespace NUMINAMATH_CALUDE_fibonacci_factorial_last_two_digits_sum_l279_27987

def fibonacci_factorial_series : List ℕ := [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

def last_two_digits (n : ℕ) : ℕ := n % 100

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_last_two_digits (series : List ℕ) : ℕ :=
  (series.map (λ n => last_two_digits (factorial n))).sum

theorem fibonacci_factorial_last_two_digits_sum :
  sum_last_two_digits fibonacci_factorial_series = 5 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_factorial_last_two_digits_sum_l279_27987


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l279_27950

theorem cubic_equation_roots (a : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    x^3 - 3*x^2 - a = 0 ∧ 
    y^3 - 3*y^2 - a = 0 ∧ 
    z^3 - 3*z^2 - a = 0) → 
  -4 < a ∧ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l279_27950


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l279_27923

/-- Given two vectors a and b in R², where a = (2,3) and b = (x,-6),
    if 2a is parallel to b, then x = -4. -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (x, -6)
  (∃ (k : ℝ), k ≠ 0 ∧ (2 * a.1, 2 * a.2) = (k * b.1, k * b.2)) →
  x = -4 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l279_27923


namespace NUMINAMATH_CALUDE_smallest_cut_length_l279_27904

theorem smallest_cut_length (x : ℕ) : x > 0 ∧ x ≤ 13 →
  (∀ y : ℕ, y > 0 ∧ y ≤ 13 → (13 - y) + (20 - y) ≤ 25 - y → y ≥ x) →
  (13 - x) + (20 - x) ≤ 25 - x →
  x = 8 := by
  sorry

end NUMINAMATH_CALUDE_smallest_cut_length_l279_27904


namespace NUMINAMATH_CALUDE_joan_took_25_marbles_l279_27930

-- Define the initial number of yellow marbles
def initial_yellow_marbles : ℕ := 86

-- Define the remaining number of yellow marbles
def remaining_yellow_marbles : ℕ := 61

-- Define the number of yellow marbles Joan took
def marbles_taken : ℕ := initial_yellow_marbles - remaining_yellow_marbles

-- Theorem to prove
theorem joan_took_25_marbles : marbles_taken = 25 := by
  sorry

end NUMINAMATH_CALUDE_joan_took_25_marbles_l279_27930


namespace NUMINAMATH_CALUDE_expand_and_simplify_l279_27963

theorem expand_and_simplify (x : ℝ) :
  5 * (6 * x^3 - 3 * x^2 + 4 * x - 2) = 30 * x^3 - 15 * x^2 + 20 * x - 10 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l279_27963


namespace NUMINAMATH_CALUDE_profit_share_difference_l279_27955

/-- Given the investments and profit share of B, calculate the difference between profit shares of A and C -/
theorem profit_share_difference (investment_A investment_B investment_C profit_B : ℕ) : 
  investment_A = 8000 →
  investment_B = 10000 →
  investment_C = 12000 →
  profit_B = 1900 →
  ∃ (profit_A profit_C : ℕ),
    profit_A * investment_B = profit_B * investment_A ∧
    profit_C * investment_B = profit_B * investment_C ∧
    profit_C - profit_A = 760 :=
by sorry

end NUMINAMATH_CALUDE_profit_share_difference_l279_27955


namespace NUMINAMATH_CALUDE_not_penetrating_function_l279_27996

/-- Definition of a penetrating function -/
def isPenetratingFunction (f : ℝ → ℝ) : Prop :=
  ∀ a : ℝ, a > 0 ∧ a ≠ 1 → ∀ x : ℝ, f (a * x) = a * f x

/-- The function f(x) = x + 1 -/
def f (x : ℝ) : ℝ := x + 1

/-- Theorem: f(x) = x + 1 is not a penetrating function -/
theorem not_penetrating_function : ¬ isPenetratingFunction f := by
  sorry

end NUMINAMATH_CALUDE_not_penetrating_function_l279_27996


namespace NUMINAMATH_CALUDE_square_minus_eight_equals_power_of_three_l279_27958

theorem square_minus_eight_equals_power_of_three (b n : ℕ) :
  b^2 - 8 = 3^n ↔ b = 3 ∧ n = 0 := by sorry

end NUMINAMATH_CALUDE_square_minus_eight_equals_power_of_three_l279_27958


namespace NUMINAMATH_CALUDE_minimum_amount_spent_on_boxes_l279_27948

/-- The minimum amount spent on boxes for packaging a collection -/
theorem minimum_amount_spent_on_boxes
  (box_length : ℝ)
  (box_width : ℝ)
  (box_height : ℝ)
  (box_cost : ℝ)
  (total_volume : ℝ)
  (h1 : box_length = 20)
  (h2 : box_width = 20)
  (h3 : box_height = 12)
  (h4 : box_cost = 0.5)
  (h5 : total_volume = 2160000) :
  ⌈total_volume / (box_length * box_width * box_height)⌉ * box_cost = 225 :=
sorry

end NUMINAMATH_CALUDE_minimum_amount_spent_on_boxes_l279_27948


namespace NUMINAMATH_CALUDE_xy_max_and_x2_y2_min_l279_27921

theorem xy_max_and_x2_y2_min (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2*y = 1) :
  (∃ (x0 y0 : ℝ), x0 > 0 ∧ y0 > 0 ∧ x0 + 2*y0 = 1 ∧ x0*y0 = 1/8 ∧ ∀ x' y' : ℝ, x' > 0 → y' > 0 → x' + 2*y' = 1 → x'*y' ≤ 1/8) ∧
  (x^2 + y^2 ≥ 1/5 ∧ ∃ (x1 y1 : ℝ), x1 > 0 ∧ y1 > 0 ∧ x1 + 2*y1 = 1 ∧ x1^2 + y1^2 = 1/5) :=
by sorry

end NUMINAMATH_CALUDE_xy_max_and_x2_y2_min_l279_27921


namespace NUMINAMATH_CALUDE_f_monotone_increasing_l279_27985

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 + (1/2) * x^2

theorem f_monotone_increasing :
  (∀ x y, x < y ∧ y < -1 → f x < f y) ∧
  (∀ x y, 0 < x ∧ x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_l279_27985


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_max_area_l279_27969

/-- A quadrilateral with side lengths a, b, c, d -/
structure Quadrilateral (a b c d : ℝ) where
  angle_sum : ℝ -- Sum of all interior angles
  area : ℝ -- Area of the quadrilateral

/-- Definition of a cyclic quadrilateral -/
def is_cyclic (q : Quadrilateral a b c d) : Prop :=
  q.angle_sum = 2 * Real.pi

/-- Theorem: Among all quadrilaterals with given side lengths, 
    the cyclic quadrilateral has the largest area -/
theorem cyclic_quadrilateral_max_area 
  {a b c d : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  ∀ q : Quadrilateral a b c d, 
    ∃ q_cyclic : Quadrilateral a b c d, 
      is_cyclic q_cyclic ∧ q.area ≤ q_cyclic.area :=
sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_max_area_l279_27969


namespace NUMINAMATH_CALUDE_line_l_and_AB_are_skew_l279_27991

-- Define the types for points, lines, and planes
variable (Point Line Plane : Type)

-- Define the relationships
variable (on_plane : Point → Plane → Prop)
variable (on_line : Point → Line → Prop)
variable (line_on_plane : Line → Plane → Prop)
variable (plane_intersection : Plane → Plane → Line)
variable (line_through_points : Point → Point → Line)
variable (skew_lines : Line → Line → Prop)

-- Define the given conditions
variable (α β : Plane)
variable (A B : Point)
variable (l : Line)

-- Theorem statement
theorem line_l_and_AB_are_skew :
  on_plane A α →
  on_plane B β →
  plane_intersection α β = l →
  ¬ on_line A l →
  ¬ on_line B l →
  skew_lines l (line_through_points A B) :=
by sorry

end NUMINAMATH_CALUDE_line_l_and_AB_are_skew_l279_27991


namespace NUMINAMATH_CALUDE_cosine_product_eleven_l279_27927

theorem cosine_product_eleven : 
  Real.cos (π / 11) * Real.cos (2 * π / 11) * Real.cos (3 * π / 11) * 
  Real.cos (4 * π / 11) * Real.cos (5 * π / 11) = 1 / 32 :=
by sorry

end NUMINAMATH_CALUDE_cosine_product_eleven_l279_27927


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l279_27975

theorem complex_fraction_simplification (x : ℝ) 
  (h1 : x ≠ 2) (h2 : x ≠ 4) (h3 : x ≠ 5) : 
  (x^2 - 4*x + 3) / (x^2 - 6*x + 8) / ((x^2 - 6*x + 5) / (x^2 - 8*x + 15)) = 
  1 / ((x - 2) * (x - 4)) := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l279_27975


namespace NUMINAMATH_CALUDE_three_element_subsets_of_eight_l279_27964

theorem three_element_subsets_of_eight (S : Finset Nat) :
  S.card = 8 → (S.powerset.filter (fun s => s.card = 3)).card = 56 := by
  sorry

end NUMINAMATH_CALUDE_three_element_subsets_of_eight_l279_27964


namespace NUMINAMATH_CALUDE_evaluate_expression_l279_27990

theorem evaluate_expression (b y : ℤ) (h : y = b + 9) : y - b + 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l279_27990


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l279_27909

theorem divisibility_implies_equality (a b n : ℕ) 
  (h : ∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) : 
  a = b^n := by
sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l279_27909


namespace NUMINAMATH_CALUDE_circle_max_min_distances_l279_27956

/-- Circle C with center (3,4) and radius 1 -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 4)^2 = 1}

/-- Point A -/
def A : ℝ × ℝ := (-1, 0)

/-- Point B -/
def B : ℝ × ℝ := (1, 0)

/-- Distance squared between two points -/
def distanceSquared (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

/-- The expression to be maximized and minimized -/
def d (P : ℝ × ℝ) : ℝ :=
  distanceSquared P A + distanceSquared P B

theorem circle_max_min_distances :
  (∀ P ∈ C, d P ≤ 74) ∧ (∀ P ∈ C, d P ≥ 34) ∧ (∃ P ∈ C, d P = 74) ∧ (∃ P ∈ C, d P = 34) := by
  sorry

end NUMINAMATH_CALUDE_circle_max_min_distances_l279_27956


namespace NUMINAMATH_CALUDE_mode_of_dataset_l279_27966

def dataset : List ℕ := [2, 2, 2, 3, 3, 4]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_of_dataset : mode dataset = 2 := by
  sorry

end NUMINAMATH_CALUDE_mode_of_dataset_l279_27966


namespace NUMINAMATH_CALUDE_min_abs_sum_l279_27905

theorem min_abs_sum (x₁ x₂ : ℝ) 
  (h : (2 + Real.sin x₁) * (2 + Real.sin (2 * x₂)) = 1) : 
  ∃ (k m : ℤ), |x₁ + x₂| ≥ π / 4 ∧ 
  |x₁ + x₂| = π / 4 ↔ x₁ = 3 * π / 2 + 2 * π * k ∧ x₂ = 3 * π / 4 + π * m := by
  sorry

end NUMINAMATH_CALUDE_min_abs_sum_l279_27905


namespace NUMINAMATH_CALUDE_center_is_nine_l279_27977

/-- Represents a 3x3 grid --/
def Grid := Fin 3 → Fin 3 → Nat

/-- Check if two positions are adjacent in the grid --/
def adjacent (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2.val + 1 = p2.2.val ∨ p2.2.val + 1 = p1.2.val)) ∨
  (p1.2 = p2.2 ∧ (p1.1.val + 1 = p2.1.val ∨ p2.1.val + 1 = p1.1.val))

/-- Check if a grid satisfies the consecutive number condition --/
def consecutiveAdjacent (g : Grid) : Prop :=
  ∀ i j k l : Fin 3, (g i j).succ = g k l → adjacent (i, j) (k, l)

/-- Sum of corner numbers in the grid --/
def cornerSum (g : Grid) : Nat :=
  g 0 0 + g 0 2 + g 2 0 + g 2 2

/-- The center number in the grid --/
def centerNumber (g : Grid) : Nat := g 1 1

/-- All numbers from 1 to 9 are used in the grid --/
def usesAllNumbers (g : Grid) : Prop :=
  ∀ n : Nat, n ≥ 1 → n ≤ 9 → ∃ i j : Fin 3, g i j = n

theorem center_is_nine (g : Grid) 
    (h1 : usesAllNumbers g)
    (h2 : consecutiveAdjacent g)
    (h3 : cornerSum g = 20) :
  centerNumber g = 9 := by
  sorry

end NUMINAMATH_CALUDE_center_is_nine_l279_27977
