import Mathlib

namespace NUMINAMATH_CALUDE_kaleb_total_score_l907_90707

/-- Kaleb's score in the first half of the game -/
def first_half_score : ℕ := 43

/-- Kaleb's score in the second half of the game -/
def second_half_score : ℕ := 23

/-- Kaleb's total score in the game -/
def total_score : ℕ := first_half_score + second_half_score

/-- Theorem stating that Kaleb's total score is 66 points -/
theorem kaleb_total_score : total_score = 66 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_total_score_l907_90707


namespace NUMINAMATH_CALUDE_max_value_x_plus_reciprocal_l907_90725

theorem max_value_x_plus_reciprocal (x : ℝ) (h : 10 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 12 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 12 :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_plus_reciprocal_l907_90725


namespace NUMINAMATH_CALUDE_park_visitors_l907_90705

/-- Given a park with visitors on Saturday and Sunday, calculate the total number of visitors over two days. -/
theorem park_visitors (saturday_visitors : ℕ) (sunday_extra : ℕ) : 
  saturday_visitors = 200 → sunday_extra = 40 → 
  saturday_visitors + (saturday_visitors + sunday_extra) = 440 := by
  sorry

#check park_visitors

end NUMINAMATH_CALUDE_park_visitors_l907_90705


namespace NUMINAMATH_CALUDE_trip_duration_l907_90748

/-- A car trip with varying speeds -/
structure CarTrip where
  totalTime : ℝ
  averageSpeed : ℝ

/-- The conditions of the car trip -/
def tripConditions (trip : CarTrip) : Prop :=
  ∃ (additionalTime : ℝ),
    trip.totalTime = 4 + additionalTime ∧
    50 * 4 + 80 * additionalTime = 65 * trip.totalTime ∧
    trip.averageSpeed = 65

/-- The theorem stating that the trip duration is 8 hours -/
theorem trip_duration (trip : CarTrip) 
    (h : tripConditions trip) : trip.totalTime = 8 := by
  sorry

#check trip_duration

end NUMINAMATH_CALUDE_trip_duration_l907_90748


namespace NUMINAMATH_CALUDE_total_painting_time_l907_90758

/-- Given that Hadassah paints 12 paintings in 6 hours and adds 20 more paintings,
    prove that the total time to finish all paintings is 16 hours. -/
theorem total_painting_time (initial_paintings : ℕ) (initial_time : ℝ) (additional_paintings : ℕ) :
  initial_paintings = 12 →
  initial_time = 6 →
  additional_paintings = 20 →
  (initial_time + (additional_paintings * (initial_time / initial_paintings))) = 16 :=
by sorry

end NUMINAMATH_CALUDE_total_painting_time_l907_90758


namespace NUMINAMATH_CALUDE_megan_water_consumption_l907_90776

/-- The number of glasses of water Megan drinks in a given time period -/
def glasses_of_water (minutes : ℕ) : ℕ :=
  minutes / 20

theorem megan_water_consumption : glasses_of_water 220 = 11 := by
  sorry

end NUMINAMATH_CALUDE_megan_water_consumption_l907_90776


namespace NUMINAMATH_CALUDE_kenneth_earnings_l907_90766

theorem kenneth_earnings (spent_percentage : Real) (remaining_amount : Real) (total_earnings : Real) : 
  spent_percentage = 0.1 →
  remaining_amount = 405 →
  remaining_amount = (1 - spent_percentage) * total_earnings →
  total_earnings = 450 :=
by sorry

end NUMINAMATH_CALUDE_kenneth_earnings_l907_90766


namespace NUMINAMATH_CALUDE_sum_of_cubes_zero_l907_90762

theorem sum_of_cubes_zero (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / (b - c)^3 + b / (c - a)^3 + c / (a - b)^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_zero_l907_90762


namespace NUMINAMATH_CALUDE_linear_system_solution_l907_90736

theorem linear_system_solution (x y : ℝ) : 
  3 * x + 2 * y = 2 → 2 * x + 3 * y = 8 → x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_linear_system_solution_l907_90736


namespace NUMINAMATH_CALUDE_chord_length_circle_line_l907_90732

/-- The length of the chord intercepted by a circle on a line -/
theorem chord_length_circle_line (t : ℝ → ℝ × ℝ) (c : ℝ × ℝ → Prop) :
  (∀ r, t r = (-2 + r, 1 - r)) →  -- Line definition
  (∀ p, c p ↔ (p.1 - 3)^2 + (p.2 + 1)^2 = 25) →  -- Circle definition
  ∃ t₁ t₂, t₁ ≠ t₂ ∧ c (t t₁) ∧ c (t t₂) ∧ 
    Real.sqrt ((t t₁).1 - (t t₂).1)^2 + ((t t₁).2 - (t t₂).2)^2 = Real.sqrt 82 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_circle_line_l907_90732


namespace NUMINAMATH_CALUDE_equation_solutions_l907_90712

theorem equation_solutions :
  (∀ x : ℝ, (x - 2)^2 = 25 ↔ x = 7 ∨ x = -3) ∧
  (∀ x : ℝ, (x - 5)^2 = 2*(5 - x) ↔ x = 5 ∨ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l907_90712


namespace NUMINAMATH_CALUDE_tv_show_length_specific_l907_90770

/-- The length of a TV show, given the total airtime and duration of commercials and breaks -/
def tv_show_length (total_airtime : ℕ) (commercial_durations : List ℕ) (break_durations : List ℕ) : ℚ :=
  let total_minutes : ℕ := total_airtime
  let commercial_time : ℕ := commercial_durations.sum
  let break_time : ℕ := break_durations.sum
  let show_time : ℕ := total_minutes - commercial_time - break_time
  (show_time : ℚ) / 60

/-- Theorem stating the length of the TV show given specific conditions -/
theorem tv_show_length_specific : 
  let total_airtime : ℕ := 150  -- 2 hours and 30 minutes
  let commercial_durations : List ℕ := [7, 7, 13, 5, 9, 9]
  let break_durations : List ℕ := [4, 2, 8]
  abs (tv_show_length total_airtime commercial_durations break_durations - 1.4333) < 0.0001 := by
  sorry

#eval tv_show_length 150 [7, 7, 13, 5, 9, 9] [4, 2, 8]

end NUMINAMATH_CALUDE_tv_show_length_specific_l907_90770


namespace NUMINAMATH_CALUDE_original_jellybeans_proof_l907_90796

/-- The original number of jellybeans in Jenny's jar -/
def original_jellybeans : ℕ := 50

/-- The fraction of jellybeans remaining after each day -/
def daily_remaining_fraction : ℚ := 4/5

/-- The number of days that have passed -/
def days_passed : ℕ := 2

/-- The number of jellybeans remaining after two days -/
def remaining_jellybeans : ℕ := 32

/-- Theorem stating that the original number of jellybeans is correct -/
theorem original_jellybeans_proof :
  (daily_remaining_fraction ^ days_passed) * original_jellybeans = remaining_jellybeans := by
  sorry

end NUMINAMATH_CALUDE_original_jellybeans_proof_l907_90796


namespace NUMINAMATH_CALUDE_chips_calories_is_310_l907_90771

/-- Represents the calorie content of various food items and daily calorie limits --/
structure CalorieData where
  cake : ℕ
  coke : ℕ
  breakfast : ℕ
  lunch : ℕ
  daily_limit : ℕ
  remaining : ℕ

/-- Calculates the calorie content of the pack of chips --/
def calculate_chips_calories (data : CalorieData) : ℕ :=
  data.daily_limit - data.remaining - (data.cake + data.coke + data.breakfast + data.lunch)

/-- Theorem stating that the calorie content of the pack of chips is 310 --/
theorem chips_calories_is_310 (data : CalorieData) 
    (h1 : data.cake = 110)
    (h2 : data.coke = 215)
    (h3 : data.breakfast = 560)
    (h4 : data.lunch = 780)
    (h5 : data.daily_limit = 2500)
    (h6 : data.remaining = 525) :
  calculate_chips_calories data = 310 := by
  sorry

end NUMINAMATH_CALUDE_chips_calories_is_310_l907_90771


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l907_90704

theorem arithmetic_sequence_sum : 2016 - 2017 + 2018 - 2019 + 2020 = 2018 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l907_90704


namespace NUMINAMATH_CALUDE_point_movement_theorem_l907_90797

theorem point_movement_theorem (A : ℝ) : 
  (A + 7 - 4 = 0) → A = -3 := by
  sorry

end NUMINAMATH_CALUDE_point_movement_theorem_l907_90797


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l907_90709

/-- Given two inversely proportional quantities p and q, if p = 30 when q = 4,
    then p = 12 when q = 10. -/
theorem inverse_proportion_problem (p q : ℝ) (h : ∃ k : ℝ, ∀ x y : ℝ, p = x ∧ q = y → x * y = k) :
  (p = 30 ∧ q = 4) → (q = 10 → p = 12) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l907_90709


namespace NUMINAMATH_CALUDE_herd_division_l907_90723

theorem herd_division (total : ℕ) (fourth_son : ℕ) : 
  (1 : ℚ) / 3 + (1 : ℚ) / 6 + (1 : ℚ) / 9 + (fourth_son : ℚ) / total = 1 →
  fourth_son = 11 →
  total = 54 := by
sorry

end NUMINAMATH_CALUDE_herd_division_l907_90723


namespace NUMINAMATH_CALUDE_impossible_all_multiples_of_10_l907_90726

/-- Represents a grid operation (adding 1 to each cell in a subgrid) -/
structure GridOperation where
  startRow : Fin 8
  startCol : Fin 8
  size : Fin 2  -- 0 for 3x3, 1 for 4x4

/-- Represents the 8x8 grid of non-negative integers -/
def Grid := Fin 8 → Fin 8 → ℕ

/-- Applies a single grid operation to the given grid -/
def applyOperation (grid : Grid) (op : GridOperation) : Grid :=
  sorry

/-- Checks if all numbers in the grid are multiples of 10 -/
def allMultiplesOf10 (grid : Grid) : Prop :=
  ∀ i j, (grid i j) % 10 = 0

/-- Main theorem: It's impossible to make all numbers multiples of 10 -/
theorem impossible_all_multiples_of_10 (initialGrid : Grid) :
  ¬∃ (ops : List GridOperation), allMultiplesOf10 (ops.foldl applyOperation initialGrid) :=
sorry

end NUMINAMATH_CALUDE_impossible_all_multiples_of_10_l907_90726


namespace NUMINAMATH_CALUDE_find_number_l907_90773

theorem find_number : ∃ N : ℕ, N = (555 + 445) * (2 * (555 - 445)) + 70 ∧ N = 220070 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l907_90773


namespace NUMINAMATH_CALUDE_equation_3y_plus_1_eq_6_is_linear_l907_90721

/-- Definition of a linear equation in one variable -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The equation 3y + 1 = 6 is a linear equation -/
theorem equation_3y_plus_1_eq_6_is_linear :
  is_linear_equation (λ y => 3 * y + 1) :=
by
  sorry

#check equation_3y_plus_1_eq_6_is_linear

end NUMINAMATH_CALUDE_equation_3y_plus_1_eq_6_is_linear_l907_90721


namespace NUMINAMATH_CALUDE_billion_to_scientific_notation_l907_90700

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to its scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem billion_to_scientific_notation :
  toScientificNotation (463.4 * 10^9) = ScientificNotation.mk 4.634 11 sorry :=
sorry

end NUMINAMATH_CALUDE_billion_to_scientific_notation_l907_90700


namespace NUMINAMATH_CALUDE_product_equals_sum_of_squares_l907_90717

theorem product_equals_sum_of_squares 
  (nums : List ℕ) 
  (count : nums.length = 116) 
  (sum_of_squares : (nums.map (λ x => x^2)).sum = 144) : 
  nums.prod = 144 := by
sorry

end NUMINAMATH_CALUDE_product_equals_sum_of_squares_l907_90717


namespace NUMINAMATH_CALUDE_simplify_expression_l907_90714

theorem simplify_expression (x : ℝ) : (3*x - 4)*(x + 9) - (x + 6)*(3*x - 2) = 7*x - 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l907_90714


namespace NUMINAMATH_CALUDE_fixed_fee_december_l907_90764

/-- Represents the billing information for an online service provider --/
structure BillingInfo where
  dec_fixed_fee : ℝ
  hourly_charge : ℝ
  dec_connect_time : ℝ
  jan_connect_time : ℝ
  dec_bill : ℝ
  jan_bill : ℝ
  jan_fee_increase : ℝ

/-- The fixed monthly fee in December is $10.80 --/
theorem fixed_fee_december (info : BillingInfo) : info.dec_fixed_fee = 10.80 :=
  by
  have h1 : info.dec_bill = 15.00 := by sorry
  have h2 : info.jan_bill = 25.40 := by sorry
  have h3 : info.jan_connect_time = 3 * info.dec_connect_time := by sorry
  have h4 : info.jan_fee_increase = 2 := by sorry
  have h5 : info.dec_fixed_fee + info.hourly_charge * info.dec_connect_time = info.dec_bill := by sorry
  have h6 : (info.dec_fixed_fee + info.jan_fee_increase) + info.hourly_charge * info.jan_connect_time = info.jan_bill := by sorry
  sorry

#check fixed_fee_december

end NUMINAMATH_CALUDE_fixed_fee_december_l907_90764


namespace NUMINAMATH_CALUDE_triangle_angle_inequality_l907_90777

-- Define a triangle
structure Triangle where
  A : ℝ  -- angle A
  B : ℝ  -- angle B
  C : ℝ  -- angle C
  a : ℝ  -- side opposite to A
  b : ℝ  -- side opposite to B
  c : ℝ  -- side opposite to C
  angle_sum : A + B + C = π
  law_of_sines : a / Real.sin A = b / Real.sin B
  
-- State the theorem
theorem triangle_angle_inequality (t : Triangle) :
  Real.sin t.A * Real.sin t.B > Real.sin t.C ^ 2 → t.C < π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_inequality_l907_90777


namespace NUMINAMATH_CALUDE_homework_questions_l907_90708

theorem homework_questions (first_hour second_hour third_hour : ℕ) : 
  third_hour = 132 → 
  third_hour = 2 * second_hour → 
  third_hour = 3 * first_hour → 
  first_hour + second_hour + third_hour = 264 :=
by
  sorry

end NUMINAMATH_CALUDE_homework_questions_l907_90708


namespace NUMINAMATH_CALUDE_unique_solution_l907_90742

/-- Represents a 3-digit number AAA where A is a single digit -/
def three_digit_AAA (A : ℕ) : ℕ := 100 * A + 10 * A + A

/-- Represents a 6-digit number AAABBB where A and B are single digits -/
def six_digit_AAABBB (A B : ℕ) : ℕ := 1000 * (three_digit_AAA A) + 100 * B + 10 * B + B

/-- Proves that the only solution to AAA × AAA + AAA = AAABBB is A = 9 and B = 0 -/
theorem unique_solution : 
  ∀ A B : ℕ, 
  A ≠ 0 → 
  A < 10 → 
  B < 10 → 
  (three_digit_AAA A) * (three_digit_AAA A) + (three_digit_AAA A) = six_digit_AAABBB A B → 
  A = 9 ∧ B = 0 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l907_90742


namespace NUMINAMATH_CALUDE_greatest_common_multiple_8_12_under_90_l907_90774

theorem greatest_common_multiple_8_12_under_90 : 
  ∃ (n : ℕ), n = 72 ∧ 
  (∀ m : ℕ, m < 90 → m % 8 = 0 → m % 12 = 0 → m ≤ n) ∧
  72 % 8 = 0 ∧ 72 % 12 = 0 ∧ 72 < 90 :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_8_12_under_90_l907_90774


namespace NUMINAMATH_CALUDE_tg_plus_ctg_values_l907_90706

-- Define the trigonometric functions
noncomputable def sec (x : ℝ) : ℝ := 1 / Real.cos x
noncomputable def cosec (x : ℝ) : ℝ := 1 / Real.sin x
noncomputable def tg (x : ℝ) : ℝ := Real.tan x
noncomputable def ctg (x : ℝ) : ℝ := 1 / Real.tan x

-- State the theorem
theorem tg_plus_ctg_values (x : ℝ) :
  sec x - cosec x = 4 * Real.sqrt 3 →
  (tg x + ctg x = -6 ∨ tg x + ctg x = 8) :=
by sorry

end NUMINAMATH_CALUDE_tg_plus_ctg_values_l907_90706


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_l907_90743

/-- The sum of complex numbers 1 + i + i² + ... + i¹⁰ equals i -/
theorem sum_of_powers_of_i : 
  (Finset.range 11).sum (fun k => (Complex.I : ℂ) ^ k) = Complex.I :=
by sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_l907_90743


namespace NUMINAMATH_CALUDE_johns_piggy_bank_l907_90744

theorem johns_piggy_bank (quarters dimes nickels : ℕ) : 
  dimes = quarters + 3 →
  nickels = quarters - 6 →
  quarters + dimes + nickels = 63 →
  quarters = 22 :=
by sorry

end NUMINAMATH_CALUDE_johns_piggy_bank_l907_90744


namespace NUMINAMATH_CALUDE_average_work_hours_l907_90735

theorem average_work_hours (total_people : ℕ) (people_on_duty : ℕ) (hours_per_day : ℕ) :
  total_people = 8 →
  people_on_duty = 3 →
  hours_per_day = 24 →
  (hours_per_day * people_on_duty : ℚ) / total_people = 9 := by
sorry

end NUMINAMATH_CALUDE_average_work_hours_l907_90735


namespace NUMINAMATH_CALUDE_hike_up_time_l907_90775

/-- Proves that the time taken to hike up a hill is 1.8 hours given specific conditions -/
theorem hike_up_time (up_speed down_speed total_time : ℝ) 
  (h1 : up_speed = 4)
  (h2 : down_speed = 6)
  (h3 : total_time = 3) : 
  ∃ (t : ℝ), t * up_speed = (total_time - t) * down_speed ∧ t = 1.8 := by
  sorry

#check hike_up_time

end NUMINAMATH_CALUDE_hike_up_time_l907_90775


namespace NUMINAMATH_CALUDE_fifth_score_proof_l907_90792

theorem fifth_score_proof (s1 s2 s3 s4 s5 : ℕ) : 
  s1 = 90 ∧ s2 = 93 ∧ s3 = 85 ∧ s4 = 97 →
  (s1 + s2 + s3 + s4 + s5) / 5 = 92 →
  s5 = 95 := by
sorry

end NUMINAMATH_CALUDE_fifth_score_proof_l907_90792


namespace NUMINAMATH_CALUDE_ants_meet_at_66cm_l907_90788

/-- Represents a point on the tile grid -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- Represents a movement on the grid -/
inductive GridMove
  | Right
  | Up
  | Left
  | Down

/-- The path of an ant on the grid -/
def AntPath := List GridMove

/-- Calculate the distance traveled given a path -/
def pathDistance (path : AntPath) (tileWidth tileLength : ℕ) : ℕ :=
  path.foldl (fun acc move =>
    acc + match move with
      | GridMove.Right => tileLength
      | GridMove.Up => tileWidth
      | GridMove.Left => tileLength
      | GridMove.Down => tileWidth) 0

/-- Check if two paths meet at the same point -/
def pathsMeet (path1 path2 : AntPath) (start1 start2 : GridPoint) : Prop :=
  sorry

theorem ants_meet_at_66cm (tileWidth tileLength : ℕ) (startM startN : GridPoint) 
    (pathM pathN : AntPath) : 
  tileWidth = 4 →
  tileLength = 6 →
  startM = ⟨0, 0⟩ →
  startN = ⟨14, 12⟩ →
  pathsMeet pathM pathN startM startN →
  pathDistance pathM tileWidth tileLength = 66 ∧
  pathDistance pathN tileWidth tileLength = 66 :=
by
  sorry

#check ants_meet_at_66cm

end NUMINAMATH_CALUDE_ants_meet_at_66cm_l907_90788


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l907_90755

theorem quadratic_inequality_empty_solution_set (a : ℝ) :
  (∀ x : ℝ, x^2 - 4*x + a^2 > 0) → (a < -2 ∨ a > 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l907_90755


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l907_90718

theorem triangle_ABC_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  a > 0 → b > 0 → c > 0 →
  Real.sqrt 3 * b * Real.cos A = Real.sin A * (a * Real.cos C + c * Real.cos A) →
  a = 2 * Real.sqrt 3 →
  (5 * Real.sqrt 3) / 4 = (1 / 2) * a * b * Real.sin C →
  (A = π / 3) ∧ (a + b + c = 5 * Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l907_90718


namespace NUMINAMATH_CALUDE_triangle_side_length_l907_90783

theorem triangle_side_length (A B C : ℝ × ℝ) : 
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let angleB := Real.arccos ((BC^2 + (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2))^2 - (Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2))^2) / (2 * BC * Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)))
  let area := (1/2) * BC * Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) * Real.sin angleB
  BC = 1 → angleB = π/3 → area = Real.sqrt 3 → 
  Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l907_90783


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l907_90786

theorem nested_fraction_equality : 
  1 + (1 / (1 + (1 / (1 + (1 / (1 + 2)))))) = 11 / 7 := by sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l907_90786


namespace NUMINAMATH_CALUDE_mango_crates_problem_l907_90798

theorem mango_crates_problem (total_cost : ℝ) (lost_crates : ℕ) (selling_price : ℝ) (profit_percentage : ℝ) :
  total_cost = 160 →
  lost_crates = 2 →
  selling_price = 25 →
  profit_percentage = 0.25 →
  ∃ (initial_crates : ℕ),
    initial_crates = 10 ∧
    (initial_crates - lost_crates : ℝ) * selling_price = total_cost * (1 + profit_percentage) :=
by sorry

end NUMINAMATH_CALUDE_mango_crates_problem_l907_90798


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l907_90740

/-- Two vectors in ℝ² are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (2, m)
  parallel a b → m = -4 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l907_90740


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l907_90778

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 1 > 0) ↔ (∃ x : ℝ, x^2 - x + 1 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l907_90778


namespace NUMINAMATH_CALUDE_range_of_m_l907_90749

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - 3 * m * x + 9 ≥ 0) → 
  m ∈ Set.Icc (-2 * Real.sqrt 2) (2 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l907_90749


namespace NUMINAMATH_CALUDE_square_difference_l907_90789

theorem square_difference (a b : ℕ) (h1 : b - a = 3) (h2 : b = 8) : b^2 - a^2 = 39 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l907_90789


namespace NUMINAMATH_CALUDE_area_of_ring_area_of_specific_ring_l907_90790

/-- The area of a ring-shaped region formed by two concentric circles -/
theorem area_of_ring (r₁ r₂ : ℝ) (h : r₁ > r₂) :
  (π * r₁^2 - π * r₂^2) = π * (r₁^2 - r₂^2) :=
by sorry

/-- The area of a ring-shaped region formed by two concentric circles with radii 12 and 5 -/
theorem area_of_specific_ring :
  π * (12^2 - 5^2) = 119 * π :=
by sorry

end NUMINAMATH_CALUDE_area_of_ring_area_of_specific_ring_l907_90790


namespace NUMINAMATH_CALUDE_bubble_arrangements_l907_90754

def word_length : ℕ := 6
def repeated_letter_count : ℕ := 3

theorem bubble_arrangements :
  (word_length.factorial) / (repeated_letter_count.factorial) = 120 :=
by sorry

end NUMINAMATH_CALUDE_bubble_arrangements_l907_90754


namespace NUMINAMATH_CALUDE_calculation_proof_l907_90737

theorem calculation_proof :
  (let a := 3 + 4/5
   let b := (1 - 9/10) / (1/100)
   a * b = 38) ∧
  (let c := 5/6 + 20
   let d := 5/4
   c / d = 50/3) ∧
  (3/7 * 5/9 * 28 * 45 = 300) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l907_90737


namespace NUMINAMATH_CALUDE_min_value_condition_l907_90745

def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| + |a * x + 1|

theorem min_value_condition (a : ℝ) : 
  (∀ x : ℝ, f a x ≥ 3/2) ∧ (∃ x : ℝ, f a x = 3/2) ↔ a = -1/2 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_condition_l907_90745


namespace NUMINAMATH_CALUDE_right_triangle_9_40_41_l907_90751

theorem right_triangle_9_40_41 : 
  ∀ (a b c : ℝ), a = 9 ∧ b = 40 ∧ c = 41 → a^2 + b^2 = c^2 :=
by
  sorry

#check right_triangle_9_40_41

end NUMINAMATH_CALUDE_right_triangle_9_40_41_l907_90751


namespace NUMINAMATH_CALUDE_barge_power_increase_l907_90787

/-- Given a barge pushed by tugboats in water, this theorem proves that
    doubling the force results in a power increase by a factor of 2√2,
    when water resistance is proportional to the square of speed. -/
theorem barge_power_increase
  (F : ℝ) -- Initial force
  (v : ℝ) -- Initial velocity
  (k : ℝ) -- Constant of proportionality for water resistance
  (h1 : F = k * v^2) -- Initial force equals water resistance
  (h2 : 2 * F = k * v_new^2) -- New force equals new water resistance
  : (2 * F * v_new) / (F * v) = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_barge_power_increase_l907_90787


namespace NUMINAMATH_CALUDE_frogs_on_lily_pads_l907_90767

/-- Given the total number of frogs in a pond, the number of frogs on logs, and the number of baby frogs on a rock,
    calculate the number of frogs on lily pads. -/
theorem frogs_on_lily_pads (total : ℕ) (on_logs : ℕ) (on_rock : ℕ) 
    (h1 : total = 32) 
    (h2 : on_logs = 3) 
    (h3 : on_rock = 24) : 
  total - on_logs - on_rock = 5 := by
  sorry

end NUMINAMATH_CALUDE_frogs_on_lily_pads_l907_90767


namespace NUMINAMATH_CALUDE_x_value_l907_90780

theorem x_value (x : Real) : 
  Real.sin (π / 2 - x) = -Real.sqrt 3 / 2 → 
  π < x → 
  x < 2 * π → 
  x = 7 * π / 6 := by
sorry

end NUMINAMATH_CALUDE_x_value_l907_90780


namespace NUMINAMATH_CALUDE_james_toy_cost_l907_90760

/-- Calculates the cost per toy given the total number of toys, percentage sold, selling price, and profit. -/
def cost_per_toy (total_toys : ℕ) (percent_sold : ℚ) (selling_price : ℚ) (profit : ℚ) : ℚ :=
  let sold_toys : ℚ := total_toys * percent_sold
  let revenue : ℚ := sold_toys * selling_price
  let cost : ℚ := revenue - profit
  cost / sold_toys

/-- Proves that the cost per toy is $25 given the problem conditions. -/
theorem james_toy_cost :
  cost_per_toy 200 (80 / 100) 30 800 = 25 := by
  sorry

end NUMINAMATH_CALUDE_james_toy_cost_l907_90760


namespace NUMINAMATH_CALUDE_square_sum_from_means_l907_90703

theorem square_sum_from_means (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 24) 
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 168) : 
  x^2 + y^2 = 1968 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_from_means_l907_90703


namespace NUMINAMATH_CALUDE_factorial_divisibility_l907_90733

theorem factorial_divisibility (p : ℕ) (h : Prime p) : 
  (Nat.factorial (p^2)) % (Nat.factorial p)^(p+1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_factorial_divisibility_l907_90733


namespace NUMINAMATH_CALUDE_binomial_sum_divides_power_of_two_l907_90746

theorem binomial_sum_divides_power_of_two (n : ℕ) : 
  n > 3 → (1 + n.choose 1 + n.choose 2 + n.choose 3 ∣ 2^2000) ↔ (n = 7 ∨ n = 23) := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_divides_power_of_two_l907_90746


namespace NUMINAMATH_CALUDE_remainder_theorem_l907_90710

theorem remainder_theorem (n : ℤ) (h : n % 7 = 2) : (3 * n - 7) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l907_90710


namespace NUMINAMATH_CALUDE_ln_cube_inequality_l907_90753

theorem ln_cube_inequality (a b : ℝ) : 
  (∃ a b, a^3 < b^3 ∧ ¬(Real.log a < Real.log b)) ∧ 
  (∀ a b, Real.log a < Real.log b → a^3 < b^3) :=
sorry

end NUMINAMATH_CALUDE_ln_cube_inequality_l907_90753


namespace NUMINAMATH_CALUDE_equation_solution_l907_90769

-- Define the functions
def f (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 4
def g (x : ℝ) : ℝ := x^4 + 2*x^3 + x^2 + 11*x + 11
def h (x : ℝ) : ℝ := x + 1

-- Define the set of solutions
def solution_set : Set ℝ := {x | x = 1 ∨ x = (-3 + Real.sqrt 5) / 2 ∨ x = (-3 - Real.sqrt 5) / 2}

-- State the theorem
theorem equation_solution :
  ∀ x ∈ solution_set, ∃ y, f y = g x ∧ y = h x :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l907_90769


namespace NUMINAMATH_CALUDE_email_count_correct_l907_90702

/-- Calculates the number of emails in Jackson's inbox after deletion and reception process -/
def final_email_count (deleted1 deleted2 received1 received2 received_after : ℕ) : ℕ :=
  received1 + received2 + received_after

/-- Theorem stating that the final email count is correct given the problem conditions -/
theorem email_count_correct :
  let deleted1 := 50
  let deleted2 := 20
  let received1 := 15
  let received2 := 5
  let received_after := 10
  final_email_count deleted1 deleted2 received1 received2 received_after = 30 := by sorry

end NUMINAMATH_CALUDE_email_count_correct_l907_90702


namespace NUMINAMATH_CALUDE_house_painting_cost_is_1900_l907_90784

/-- The cost of painting a house given three contributors' amounts -/
def housePaintingCost (judsonContribution : ℕ) : ℕ :=
  let kennyContribution := judsonContribution + judsonContribution / 5
  let camiloContribution := kennyContribution + 200
  judsonContribution + kennyContribution + camiloContribution

/-- Theorem stating that the total cost of painting the house is 1900 -/
theorem house_painting_cost_is_1900 : housePaintingCost 500 = 1900 := by
  sorry

end NUMINAMATH_CALUDE_house_painting_cost_is_1900_l907_90784


namespace NUMINAMATH_CALUDE_share_difference_l907_90741

/-- Given a distribution of money among three people with a specific ratio and one known share,
    calculate the difference between the largest and smallest shares. -/
theorem share_difference (total_parts ratio_faruk ratio_vasim ratio_ranjith vasim_share : ℕ) 
    (h1 : total_parts = ratio_faruk + ratio_vasim + ratio_ranjith)
    (h2 : ratio_faruk = 3)
    (h3 : ratio_vasim = 5)
    (h4 : ratio_ranjith = 11)
    (h5 : vasim_share = 1500) :
    ratio_ranjith * (vasim_share / ratio_vasim) - ratio_faruk * (vasim_share / ratio_vasim) = 2400 := by
  sorry

end NUMINAMATH_CALUDE_share_difference_l907_90741


namespace NUMINAMATH_CALUDE_abs_frac_inequality_l907_90722

theorem abs_frac_inequality (x : ℝ) : 
  |((x - 3) / x)| > ((x - 3) / x) ↔ 0 < x ∧ x < 3 :=
by sorry

end NUMINAMATH_CALUDE_abs_frac_inequality_l907_90722


namespace NUMINAMATH_CALUDE_average_sitting_time_l907_90785

theorem average_sitting_time 
  (total_students : ℕ) 
  (available_seats : ℕ) 
  (total_travel_time : ℕ) 
  (h1 : total_students = 8) 
  (h2 : available_seats = 5) 
  (h3 : total_travel_time = 152) : 
  (total_travel_time * available_seats) / total_students = 95 :=
by sorry

end NUMINAMATH_CALUDE_average_sitting_time_l907_90785


namespace NUMINAMATH_CALUDE_f_neg_two_l907_90729

def f (x : ℝ) : ℝ := x^2 + 3*x - 5

theorem f_neg_two : f (-2) = -7 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_l907_90729


namespace NUMINAMATH_CALUDE_sphere_radius_ratio_l907_90716

theorem sphere_radius_ratio (V_large V_small : ℝ) (r_large r_small : ℝ) : 
  V_large = 576 * Real.pi ∧ 
  V_small = 0.0625 * V_large ∧
  V_large = (4/3) * Real.pi * r_large^3 ∧
  V_small = (4/3) * Real.pi * r_small^3 →
  r_small / r_large = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sphere_radius_ratio_l907_90716


namespace NUMINAMATH_CALUDE_sector_area_l907_90747

/-- Given a sector with perimeter 16 cm and central angle 2 radians, its area is 16 cm² -/
theorem sector_area (perimeter : ℝ) (central_angle : ℝ) (area : ℝ) : 
  perimeter = 16 → central_angle = 2 → area = (1/2) * central_angle * ((perimeter / (2 + central_angle))^2) → area = 16 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l907_90747


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l907_90728

def set_A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + 3}
def set_B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 3 * p.1 - 1}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {(2, 5)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l907_90728


namespace NUMINAMATH_CALUDE_indefinite_stick_shortening_l907_90799

theorem indefinite_stick_shortening :
  ∃ t : ℝ, t > 1 ∧ ∀ n : ℕ, t^(3-n) > t^(2-n) + t^(1-n) := by
  sorry

end NUMINAMATH_CALUDE_indefinite_stick_shortening_l907_90799


namespace NUMINAMATH_CALUDE_water_bottle_cost_l907_90779

theorem water_bottle_cost (cola_price : ℝ) (juice_price : ℝ) (water_price : ℝ)
  (cola_sold : ℕ) (juice_sold : ℕ) (water_sold : ℕ) (total_revenue : ℝ)
  (h1 : cola_price = 3)
  (h2 : juice_price = 1.5)
  (h3 : cola_sold = 15)
  (h4 : juice_sold = 12)
  (h5 : water_sold = 25)
  (h6 : total_revenue = 88)
  (h7 : total_revenue = cola_price * cola_sold + juice_price * juice_sold + water_price * water_sold) :
  water_price = 1 := by
sorry

end NUMINAMATH_CALUDE_water_bottle_cost_l907_90779


namespace NUMINAMATH_CALUDE_laundry_time_l907_90781

def total_time : ℕ := 120
def bathroom_time : ℕ := 15
def room_time : ℕ := 35
def homework_time : ℕ := 40

theorem laundry_time : 
  ∃ (laundry_time : ℕ), 
    laundry_time + bathroom_time + room_time + homework_time = total_time ∧ 
    laundry_time = 30 :=
by sorry

end NUMINAMATH_CALUDE_laundry_time_l907_90781


namespace NUMINAMATH_CALUDE_kims_candy_bars_l907_90738

/-- The number of candy bars Kim's dad buys her each week -/
def candy_bars_per_week : ℕ := 2

/-- The number of weeks in the problem -/
def total_weeks : ℕ := 16

/-- The number of candy bars Kim eats in 4 weeks -/
def candy_bars_eaten_per_4_weeks : ℕ := 1

/-- The number of candy bars Kim has saved after 16 weeks -/
def candy_bars_saved : ℕ := 28

/-- Theorem stating that the number of candy bars Kim's dad buys her each week is 2 -/
theorem kims_candy_bars : 
  candy_bars_per_week * total_weeks - 
  (total_weeks / 4 * candy_bars_eaten_per_4_weeks) = 
  candy_bars_saved := by
  sorry

end NUMINAMATH_CALUDE_kims_candy_bars_l907_90738


namespace NUMINAMATH_CALUDE_paving_cost_example_l907_90713

/-- The cost of paving a rectangular floor -/
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

/-- Theorem: The cost of paving a rectangular floor with length 5.5 m and width 3.75 m
    at a rate of $600 per square metre is $12,375. -/
theorem paving_cost_example : paving_cost 5.5 3.75 600 = 12375 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_example_l907_90713


namespace NUMINAMATH_CALUDE_YZ_squared_equals_33_l907_90750

/-- Triangle ABC with given side lengths -/
structure Triangle :=
  (AB BC CA : ℝ)
  (AB_pos : AB > 0)
  (BC_pos : BC > 0)
  (CA_pos : CA > 0)

/-- Circumcircle of a triangle -/
def Circumcircle (t : Triangle) : Set (ℝ × ℝ) := sorry

/-- Incircle of a triangle -/
def Incircle (t : Triangle) : Set (ℝ × ℝ) := sorry

/-- Circle tangent to circumcircle and two sides of the triangle -/
def TangentCircle (t : Triangle) : Set (ℝ × ℝ) := sorry

/-- Intersection point of TangentCircle and Circumcircle -/
def X (t : Triangle) : ℝ × ℝ := sorry

/-- Points Y and Z on the circumcircle such that XY and YZ are tangent to the incircle -/
def Y (t : Triangle) : ℝ × ℝ := sorry
def Z (t : Triangle) : ℝ × ℝ := sorry

/-- Square of the distance between two points -/
def dist_squared (p q : ℝ × ℝ) : ℝ := sorry

theorem YZ_squared_equals_33 (t : Triangle) 
  (h1 : t.AB = 4) 
  (h2 : t.BC = 5) 
  (h3 : t.CA = 6) : 
  dist_squared (Y t) (Z t) = 33 := by sorry

end NUMINAMATH_CALUDE_YZ_squared_equals_33_l907_90750


namespace NUMINAMATH_CALUDE_polygon_sides_from_interior_angle_l907_90763

theorem polygon_sides_from_interior_angle (n : ℕ) (angle : ℝ) : 
  (n ≥ 3) → (angle = 140) → (n * angle = (n - 2) * 180) → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_interior_angle_l907_90763


namespace NUMINAMATH_CALUDE_valid_lineup_count_l907_90759

def team_size : ℕ := 15
def lineup_size : ℕ := 6

def cannot_play_together (p1 p2 : ℕ) : Prop := p1 ≠ p2

def excludes_player (p1 p2 : ℕ) : Prop := p1 ≠ p2

def valid_lineup (lineup : Finset ℕ) : Prop :=
  lineup.card = lineup_size ∧
  (∀ p ∈ lineup, p ≤ team_size) ∧
  ¬(1 ∈ lineup ∧ 2 ∈ lineup) ∧
  (1 ∈ lineup → 3 ∉ lineup)

def count_valid_lineups : ℕ := sorry

theorem valid_lineup_count :
  count_valid_lineups = 3795 := by sorry

end NUMINAMATH_CALUDE_valid_lineup_count_l907_90759


namespace NUMINAMATH_CALUDE_distinct_roots_iff_k_gt_three_fourths_roots_condition_implies_k_value_l907_90727

-- Define the quadratic equation
def quadratic_equation (k x : ℝ) : Prop :=
  x^2 + (2*k + 1)*x + k^2 + 1 = 0

-- Define the condition for two distinct real roots
def has_two_distinct_roots (k : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic_equation k x1 ∧ quadratic_equation k x2

-- Define the condition for the sum and product of roots
def roots_condition (k : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, quadratic_equation k x1 ∧ quadratic_equation k x2 ∧ 
    x1 + x2 = 2 - x1 * x2

-- Theorem for part 1
theorem distinct_roots_iff_k_gt_three_fourths :
  ∀ k : ℝ, has_two_distinct_roots k ↔ k > 3/4 :=
sorry

-- Theorem for part 2
theorem roots_condition_implies_k_value :
  ∀ k : ℝ, roots_condition k → k = 1 + Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_distinct_roots_iff_k_gt_three_fourths_roots_condition_implies_k_value_l907_90727


namespace NUMINAMATH_CALUDE_t_divides_t_2n_plus_1_l907_90711

def t : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => 2 * t (n + 1) + t n

theorem t_divides_t_2n_plus_1 (n : ℕ) : t n ∣ t (2 * n + 1) := by
  sorry

end NUMINAMATH_CALUDE_t_divides_t_2n_plus_1_l907_90711


namespace NUMINAMATH_CALUDE_cricket_team_size_l907_90756

theorem cricket_team_size :
  ∀ n : ℕ,
  n > 2 →
  let captain_age : ℕ := 25
  let keeper_age : ℕ := captain_age + 5
  let team_avg_age : ℕ := 23
  let remaining_avg_age : ℕ := team_avg_age - 1
  n * team_avg_age = captain_age + keeper_age + (n - 2) * remaining_avg_age →
  n = 11 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_size_l907_90756


namespace NUMINAMATH_CALUDE_shark_stingray_ratio_l907_90791

theorem shark_stingray_ratio :
  ∀ (total_fish sharks stingrays : ℕ),
    total_fish = 84 →
    stingrays = 28 →
    sharks + stingrays = total_fish →
    sharks / stingrays = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_shark_stingray_ratio_l907_90791


namespace NUMINAMATH_CALUDE_salary_problem_l907_90734

theorem salary_problem (total : ℝ) (a_spend_percent : ℝ) (b_spend_percent : ℝ)
  (h_total : total = 7000)
  (h_a_spend : a_spend_percent = 95)
  (h_b_spend : b_spend_percent = 85)
  (h_equal_savings : (100 - a_spend_percent) * a_salary = (100 - b_spend_percent) * (total - a_salary)) :
  a_salary = 5250 :=
by
  sorry

#check salary_problem

end NUMINAMATH_CALUDE_salary_problem_l907_90734


namespace NUMINAMATH_CALUDE_equation_solution_sum_l907_90757

theorem equation_solution_sum : ∃ x₁ x₂ : ℝ, 
  (6 * x₁) / 30 = 7 / x₁ ∧
  (6 * x₂) / 30 = 7 / x₂ ∧
  x₁ + x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_sum_l907_90757


namespace NUMINAMATH_CALUDE_profit_percentage_is_36_percent_l907_90724

def selling_price : ℝ := 850
def profit : ℝ := 225

theorem profit_percentage_is_36_percent :
  (profit / (selling_price - profit)) * 100 = 36 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_is_36_percent_l907_90724


namespace NUMINAMATH_CALUDE_quadratic_roots_implications_l907_90768

theorem quadratic_roots_implications (a b c : ℝ) 
  (h_roots : ∃ (α β : ℝ), α > 0 ∧ β ≠ 0 ∧ 
    (∀ x : ℂ, x^2 - (a + b + c) * x + (a * b + b * c + c * a) = 0 ↔ x = α + β * I ∨ x = α - β * I)) :
  (a > 0 ∧ b > 0 ∧ c > 0) ∧
  (Real.sqrt a + Real.sqrt b > Real.sqrt c ∧
   Real.sqrt b + Real.sqrt c > Real.sqrt a ∧
   Real.sqrt c + Real.sqrt a > Real.sqrt b) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_implications_l907_90768


namespace NUMINAMATH_CALUDE_subtraction_of_negative_two_minus_negative_four_equals_six_l907_90719

theorem subtraction_of_negative (a b : ℤ) : a - (-b) = a + b := by sorry

theorem two_minus_negative_four_equals_six : 2 - (-4) = 6 := by sorry

end NUMINAMATH_CALUDE_subtraction_of_negative_two_minus_negative_four_equals_six_l907_90719


namespace NUMINAMATH_CALUDE_lucas_L10_units_digit_l907_90795

/-- Lucas numbers sequence -/
def lucas : ℕ → ℕ
  | 0 => 3
  | 1 => 1
  | (n + 2) => lucas (n + 1) + lucas n

/-- Units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

theorem lucas_L10_units_digit :
  unitsDigit (lucas (lucas 10)) = 4 := by sorry

end NUMINAMATH_CALUDE_lucas_L10_units_digit_l907_90795


namespace NUMINAMATH_CALUDE_original_number_proof_l907_90739

theorem original_number_proof (x : ℝ) : 
  (x * 1.2 * 0.6 = 1080) → x = 1500 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l907_90739


namespace NUMINAMATH_CALUDE_car_wash_goal_remaining_l907_90730

def car_wash_fundraiser (goal : ℕ) (high_donors : ℕ) (high_donation : ℕ) (low_donors : ℕ) (low_donation : ℕ) : ℕ :=
  goal - (high_donors * high_donation + low_donors * low_donation)

theorem car_wash_goal_remaining :
  car_wash_fundraiser 150 3 10 15 5 = 45 := by sorry

end NUMINAMATH_CALUDE_car_wash_goal_remaining_l907_90730


namespace NUMINAMATH_CALUDE_planes_through_three_points_l907_90701

/-- Three points in 3D space -/
structure ThreePoints where
  p1 : ℝ × ℝ × ℝ
  p2 : ℝ × ℝ × ℝ
  p3 : ℝ × ℝ × ℝ

/-- Possible number of planes through three points -/
inductive NumPlanes
  | one
  | infinite

/-- The number of planes that can be constructed through three points in 3D space 
    is either one or infinite -/
theorem planes_through_three_points (points : ThreePoints) : 
  ∃ (n : NumPlanes), n = NumPlanes.one ∨ n = NumPlanes.infinite :=
sorry

end NUMINAMATH_CALUDE_planes_through_three_points_l907_90701


namespace NUMINAMATH_CALUDE_num_factors_41040_eq_80_l907_90752

/-- The number of positive factors of 41040 -/
def num_factors_41040 : ℕ :=
  (Finset.filter (· ∣ 41040) (Finset.range 41041)).card

/-- Theorem stating that the number of positive factors of 41040 is 80 -/
theorem num_factors_41040_eq_80 : num_factors_41040 = 80 := by
  sorry

end NUMINAMATH_CALUDE_num_factors_41040_eq_80_l907_90752


namespace NUMINAMATH_CALUDE_greatest_x_value_l907_90782

theorem greatest_x_value (x : ℤ) : 
  (2.134 * (10 : ℝ) ^ (x : ℝ) < 21000) ↔ x ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_greatest_x_value_l907_90782


namespace NUMINAMATH_CALUDE_divisor_power_difference_l907_90794

theorem divisor_power_difference (k : ℕ) : 
  (15 ^ k ∣ 759325) → 3 ^ k - k ^ 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisor_power_difference_l907_90794


namespace NUMINAMATH_CALUDE_peanut_mixture_solution_l907_90715

/-- Represents the peanut mixture problem -/
def peanut_mixture (virginia_weight : ℝ) (virginia_cost : ℝ) (spanish_cost : ℝ) (mixture_cost : ℝ) : ℝ → Prop :=
  λ spanish_weight : ℝ =>
    (virginia_weight * virginia_cost + spanish_weight * spanish_cost) / (virginia_weight + spanish_weight) = mixture_cost

/-- Proves that 2.5 pounds of Spanish peanuts is the correct amount for the desired mixture -/
theorem peanut_mixture_solution :
  peanut_mixture 10 3.5 3 3.4 2.5 := by
  sorry

end NUMINAMATH_CALUDE_peanut_mixture_solution_l907_90715


namespace NUMINAMATH_CALUDE_james_writing_speed_l907_90793

/-- Calculates the number of pages written per hour given the total writing time and book length -/
def pages_per_hour (hours_per_day : ℕ) (days_per_week : ℕ) (total_weeks : ℕ) (total_pages : ℕ) : ℚ :=
  total_pages / (hours_per_day * days_per_week * total_weeks)

/-- Proves that writing 3 hours a day for 7 weeks to complete a 735-page book results in 5 pages per hour -/
theorem james_writing_speed : pages_per_hour 3 7 7 735 = 5 := by
  sorry

end NUMINAMATH_CALUDE_james_writing_speed_l907_90793


namespace NUMINAMATH_CALUDE_no_three_similar_piles_l907_90761

theorem no_three_similar_piles (x : ℝ) (hx : x > 0) :
  ¬∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = x ∧
    a ≤ b ∧ b ≤ c ∧
    c ≤ Real.sqrt 2 * b ∧
    b ≤ Real.sqrt 2 * a :=
by
  sorry

end NUMINAMATH_CALUDE_no_three_similar_piles_l907_90761


namespace NUMINAMATH_CALUDE_sum_of_squares_l907_90720

theorem sum_of_squares (w x y z a b c : ℝ) 
  (hw : w ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : w * x = a^2) (h2 : w * y = b^2) (h3 : w * z = c^2) : 
  x^2 + y^2 + z^2 = (a^4 + b^4 + c^4) / w^2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l907_90720


namespace NUMINAMATH_CALUDE_second_hour_billboards_l907_90731

/-- The number of billboards counted in the second hour -/
def billboards_second_hour (first_hour : ℕ) (third_hour : ℕ) (total_hours : ℕ) (average : ℕ) : ℕ :=
  average * total_hours - first_hour - third_hour

theorem second_hour_billboards :
  billboards_second_hour 17 23 3 20 = 20 := by
  sorry

#eval billboards_second_hour 17 23 3 20

end NUMINAMATH_CALUDE_second_hour_billboards_l907_90731


namespace NUMINAMATH_CALUDE_days_to_eat_candy_correct_l907_90772

/-- Given the initial number of candies, the number of candies eaten per day for the first week,
    and the number of candies to be eaten per day after the first week,
    calculate the number of additional days Yuna can eat candy. -/
def days_to_eat_candy (initial_candies : ℕ) (candies_per_day_week1 : ℕ) (candies_per_day_after : ℕ) : ℕ :=
  let candies_eaten_week1 := candies_per_day_week1 * 7
  let remaining_candies := initial_candies - candies_eaten_week1
  remaining_candies / candies_per_day_after

theorem days_to_eat_candy_correct (initial_candies : ℕ) (candies_per_day_week1 : ℕ) (candies_per_day_after : ℕ) 
  (h1 : initial_candies = 60)
  (h2 : candies_per_day_week1 = 6)
  (h3 : candies_per_day_after = 3) :
  days_to_eat_candy initial_candies candies_per_day_week1 candies_per_day_after = 6 := by
  sorry

end NUMINAMATH_CALUDE_days_to_eat_candy_correct_l907_90772


namespace NUMINAMATH_CALUDE_coffee_shop_multiple_l907_90765

theorem coffee_shop_multiple (x : ℕ) : 
  (32 = x * 6 + 8) → x = 4 := by sorry

end NUMINAMATH_CALUDE_coffee_shop_multiple_l907_90765
