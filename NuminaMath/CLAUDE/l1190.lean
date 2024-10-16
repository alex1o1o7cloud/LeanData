import Mathlib

namespace NUMINAMATH_CALUDE_traffic_survey_l1190_119033

theorem traffic_survey (N : ℕ) 
  (drivers_A : ℕ) (sample_A : ℕ) (sample_B : ℕ) (sample_C : ℕ) (sample_D : ℕ) : 
  drivers_A = 96 →
  sample_A = 12 →
  sample_B = 21 →
  sample_C = 25 →
  sample_D = 43 →
  N = (sample_A + sample_B + sample_C + sample_D) * drivers_A / sample_A →
  N = 808 := by
sorry

end NUMINAMATH_CALUDE_traffic_survey_l1190_119033


namespace NUMINAMATH_CALUDE_total_notes_count_l1190_119004

def total_amount : ℕ := 10350
def note_50_value : ℕ := 50
def note_500_value : ℕ := 500
def note_50_count : ℕ := 37

theorem total_notes_count : 
  ∃ (note_500_count : ℕ), 
    note_50_count * note_50_value + note_500_count * note_500_value = total_amount ∧
    note_50_count + note_500_count = 54 :=
by sorry

end NUMINAMATH_CALUDE_total_notes_count_l1190_119004


namespace NUMINAMATH_CALUDE_annie_children_fruits_l1190_119040

/-- The number of fruits Annie's children received -/
def total_fruits (mike_oranges matt_apples mark_bananas : ℕ) : ℕ :=
  mike_oranges + matt_apples + mark_bananas

theorem annie_children_fruits :
  ∃ (mike_oranges matt_apples mark_bananas : ℕ),
    mike_oranges = 3 ∧
    matt_apples = 2 * mike_oranges ∧
    mark_bananas = mike_oranges + matt_apples ∧
    total_fruits mike_oranges matt_apples mark_bananas = 18 := by
  sorry

end NUMINAMATH_CALUDE_annie_children_fruits_l1190_119040


namespace NUMINAMATH_CALUDE_rightmost_three_digits_of_7_to_1993_l1190_119029

theorem rightmost_three_digits_of_7_to_1993 : 7^1993 % 1000 = 343 := by
  sorry

end NUMINAMATH_CALUDE_rightmost_three_digits_of_7_to_1993_l1190_119029


namespace NUMINAMATH_CALUDE_parking_garage_floors_l1190_119021

/-- Represents a parking garage with the given properties -/
structure ParkingGarage where
  floors : ℕ
  drive_time : ℕ
  id_check_time : ℕ
  total_time : ℕ

/-- Calculates the number of ID checks required -/
def id_checks (g : ParkingGarage) : ℕ := (g.floors - 1) / 3

/-- Calculates the total time to traverse the garage -/
def calculate_total_time (g : ParkingGarage) : ℕ :=
  g.drive_time * (g.floors - 1) + g.id_check_time * id_checks g

/-- Theorem stating that a parking garage with the given properties has 13 floors -/
theorem parking_garage_floors :
  ∃ (g : ParkingGarage), 
    g.drive_time = 80 ∧ 
    g.id_check_time = 120 ∧ 
    g.total_time = 1440 ∧ 
    calculate_total_time g = g.total_time ∧ 
    g.floors = 13 := by
  sorry

end NUMINAMATH_CALUDE_parking_garage_floors_l1190_119021


namespace NUMINAMATH_CALUDE_B_equals_set_l1190_119014

def A : Set Int := {-2, -1, 1, 2, 3, 4}

def B : Set Int := {x | ∃ t ∈ A, x = t^2}

theorem B_equals_set : B = {1, 4, 9, 16} := by sorry

end NUMINAMATH_CALUDE_B_equals_set_l1190_119014


namespace NUMINAMATH_CALUDE_joe_initial_cars_l1190_119075

/-- The number of cars Joe will have after getting more -/
def total_cars : ℕ := 62

/-- The number of additional cars Joe will get -/
def additional_cars : ℕ := 12

/-- Joe's initial number of cars -/
def initial_cars : ℕ := total_cars - additional_cars

theorem joe_initial_cars : initial_cars = 50 := by sorry

end NUMINAMATH_CALUDE_joe_initial_cars_l1190_119075


namespace NUMINAMATH_CALUDE_at_most_two_match_count_l1190_119002

/-- The number of ways to arrange 5 balls in 5 boxes -/
def total_arrangements : ℕ := 120

/-- The number of ways to arrange 5 balls in 5 boxes where exactly 3 balls match their box number -/
def three_match_arrangements : ℕ := 10

/-- The number of ways to arrange 5 balls in 5 boxes where all 5 balls match their box number -/
def all_match_arrangement : ℕ := 1

/-- The number of ways to arrange 5 balls in 5 boxes such that at most two balls have the same number as their respective boxes -/
def at_most_two_match : ℕ := total_arrangements - three_match_arrangements - all_match_arrangement

theorem at_most_two_match_count : at_most_two_match = 109 := by
  sorry

end NUMINAMATH_CALUDE_at_most_two_match_count_l1190_119002


namespace NUMINAMATH_CALUDE_snake_revenue_theorem_l1190_119086

/-- Calculates the total revenue from selling Jake's baby snakes --/
def calculate_snake_revenue (num_snakes : ℕ) (eggs_per_snake : ℕ) (regular_price : ℕ) (rare_multiplier : ℕ) : ℕ :=
  let total_babies := num_snakes * eggs_per_snake
  let regular_babies := total_babies - 1
  let regular_revenue := regular_babies * regular_price
  let rare_revenue := regular_price * rare_multiplier
  regular_revenue + rare_revenue

/-- Proves that the total revenue from selling Jake's baby snakes is $2250 --/
theorem snake_revenue_theorem :
  calculate_snake_revenue 3 2 250 4 = 2250 := by
  sorry

end NUMINAMATH_CALUDE_snake_revenue_theorem_l1190_119086


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l1190_119092

theorem product_of_three_numbers (a b c m : ℝ) : 
  a + b + c = 180 ∧
  5 * a = m ∧
  b = m + 12 ∧
  c = m - 6 →
  a * b * c = 42184 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l1190_119092


namespace NUMINAMATH_CALUDE_midpoint_trajectory_of_moving_chord_l1190_119070

/-- Given a circle and a moving chord, prove the equation of the midpoint's trajectory -/
theorem midpoint_trajectory_of_moving_chord 
  (x y : ℝ) (M : ℝ × ℝ) : 
  (∀ (C D : ℝ × ℝ), 
    (C.1^2 + C.2^2 = 25) ∧ 
    (D.1^2 + D.2^2 = 25) ∧ 
    ((C.1 - D.1)^2 + (C.2 - D.2)^2 = 64) ∧ 
    (M = ((C.1 + D.1)/2, (C.2 + D.2)/2))) →
  (M.1^2 + M.2^2 = 9) := by
sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_of_moving_chord_l1190_119070


namespace NUMINAMATH_CALUDE_second_bill_overdue_months_l1190_119018

/-- Calculates the number of months a bill is overdue given the total amount owed and the conditions of three bills -/
def months_overdue (total_owed : ℚ) (bill1_amount : ℚ) (bill1_interest_rate : ℚ) (bill1_months : ℕ)
                   (bill2_amount : ℚ) (bill2_fee : ℚ)
                   (bill3_fee1 : ℚ) (bill3_fee2 : ℚ) : ℕ :=
  let bill1_total := bill1_amount + bill1_amount * bill1_interest_rate * bill1_months
  let bill3_total := bill3_fee1 + bill3_fee2
  let bill2_overdue := total_owed - bill1_total - bill3_total
  Nat.ceil (bill2_overdue / bill2_fee)

/-- The number of months the second bill is overdue is 18 -/
theorem second_bill_overdue_months :
  months_overdue 1234 200 (1/10) 2 130 50 40 80 = 18 := by
  sorry

end NUMINAMATH_CALUDE_second_bill_overdue_months_l1190_119018


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l1190_119085

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def N : Set ℝ := {x | Real.log (x - 2) < 1}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = {x : ℝ | -1 < x ∧ x < 12} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l1190_119085


namespace NUMINAMATH_CALUDE_trig_inequality_l1190_119035

theorem trig_inequality (a b : Real) (ha : 0 < a ∧ a < π/2) (hb : 0 < b ∧ b < π/2) :
  (Real.sin a)^3 / Real.sin b + (Real.cos a)^3 / Real.cos b ≥ 1 / Real.cos (a - b) := by
  sorry

end NUMINAMATH_CALUDE_trig_inequality_l1190_119035


namespace NUMINAMATH_CALUDE_aaron_initial_cards_l1190_119055

theorem aaron_initial_cards (found : ℕ) (final : ℕ) (h1 : found = 62) (h2 : final = 67) :
  final - found = 5 := by
  sorry

end NUMINAMATH_CALUDE_aaron_initial_cards_l1190_119055


namespace NUMINAMATH_CALUDE_square_root_of_25_l1190_119097

-- Define the concept of square root
def is_square_root (x y : ℝ) : Prop := y^2 = x

-- Theorem statement
theorem square_root_of_25 : 
  ∃ (a b : ℝ), a ≠ b ∧ is_square_root 25 a ∧ is_square_root 25 b :=
sorry

end NUMINAMATH_CALUDE_square_root_of_25_l1190_119097


namespace NUMINAMATH_CALUDE_boy_girl_ratio_l1190_119073

/-- Represents the number of students in the class -/
def total_students : ℕ := 25

/-- Represents the difference between the number of boys and girls -/
def boy_girl_difference : ℕ := 9

/-- Theorem stating that the ratio of boys to girls is 17:8 -/
theorem boy_girl_ratio :
  ∃ (boys girls : ℕ),
    boys + girls = total_students ∧
    boys = girls + boy_girl_difference ∧
    boys = 17 ∧
    girls = 8 :=
by sorry

end NUMINAMATH_CALUDE_boy_girl_ratio_l1190_119073


namespace NUMINAMATH_CALUDE_giorgio_cookies_l1190_119020

theorem giorgio_cookies (total_students : ℕ) (oatmeal_ratio : ℚ) (oatmeal_cookies : ℕ) 
  (h1 : total_students = 40)
  (h2 : oatmeal_ratio = 1/10)
  (h3 : oatmeal_cookies = 8) :
  (oatmeal_cookies : ℚ) / (oatmeal_ratio * total_students) = 2 := by
  sorry

end NUMINAMATH_CALUDE_giorgio_cookies_l1190_119020


namespace NUMINAMATH_CALUDE_sin_B_value_l1190_119041

-- Define a right triangle ABC
structure RightTriangle :=
  (A B C : Real)
  (right_angle : C = 90)
  (bc_half_ac : B = 1/2 * A)

-- Theorem statement
theorem sin_B_value (t : RightTriangle) : Real.sin (t.B) = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_B_value_l1190_119041


namespace NUMINAMATH_CALUDE_triangle_side_length_l1190_119087

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  (a + b + c) * (b + c - a) = 3 * b * c →
  a = Real.sqrt 3 →
  Real.tan B = Real.sqrt 2 / 4 →
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = Real.pi →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A →
  b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * Real.cos B →
  c ^ 2 = a ^ 2 + b ^ 2 - 2 * a * b * Real.cos C →
  b = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1190_119087


namespace NUMINAMATH_CALUDE_percent_calculation_l1190_119052

theorem percent_calculation (x : ℝ) (h : 0.6 * x = 42) : 0.5 * x = 35 := by
  sorry

end NUMINAMATH_CALUDE_percent_calculation_l1190_119052


namespace NUMINAMATH_CALUDE_minimizes_y_l1190_119068

/-- The function y in terms of x, a, b, and c -/
def y (x a b c : ℝ) : ℝ := (x - a)^2 + (x - b)^2 + (x - c)^2

/-- The theorem stating that (a + b + c) / 3 minimizes y -/
theorem minimizes_y (a b c : ℝ) :
  let x_min : ℝ := (a + b + c) / 3
  ∀ x : ℝ, y x_min a b c ≤ y x a b c :=
sorry

end NUMINAMATH_CALUDE_minimizes_y_l1190_119068


namespace NUMINAMATH_CALUDE_cost_of_apples_and_bananas_l1190_119036

/-- The cost of apples in dollars per pound -/
def apple_cost : ℚ := 3 / 3

/-- The cost of bananas in dollars per pound -/
def banana_cost : ℚ := 2 / 2

/-- The total cost of apples and bananas -/
def total_cost (apple_pounds banana_pounds : ℚ) : ℚ :=
  apple_pounds * apple_cost + banana_pounds * banana_cost

theorem cost_of_apples_and_bananas :
  total_cost 9 6 = 15 := by sorry

end NUMINAMATH_CALUDE_cost_of_apples_and_bananas_l1190_119036


namespace NUMINAMATH_CALUDE_vaishali_hats_l1190_119049

/-- The number of hats with 4 stripes each that Vaishali has -/
def hats_with_four_stripes : ℕ :=
  let three_stripe_hats := 4
  let three_stripe_count := 3
  let no_stripe_hats := 6
  let five_stripe_hats := 2
  let five_stripe_count := 5
  let total_stripes := 34
  let remaining_stripes := total_stripes - 
    (three_stripe_hats * three_stripe_count + 
     no_stripe_hats * 0 + 
     five_stripe_hats * five_stripe_count)
  remaining_stripes / 4

theorem vaishali_hats : hats_with_four_stripes = 3 := by
  sorry

end NUMINAMATH_CALUDE_vaishali_hats_l1190_119049


namespace NUMINAMATH_CALUDE_cot_thirty_degrees_l1190_119071

theorem cot_thirty_degrees : Real.cos (π / 6) / Real.sin (π / 6) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cot_thirty_degrees_l1190_119071


namespace NUMINAMATH_CALUDE_probability_multiple_of_15_l1190_119011

/-- The set of digits used to form the numbers -/
def digits : Finset Nat := {1, 2, 3, 4, 5}

/-- A five-digit number without repeating digits -/
def FiveDigitNumber := {n : Finset Nat // n.card = 5 ∧ n ⊆ digits}

/-- The set of all possible five-digit numbers -/
def allNumbers : Finset FiveDigitNumber := sorry

/-- Predicate to check if a number is a multiple of 15 -/
def isMultipleOf15 (n : FiveDigitNumber) : Prop := sorry

/-- The set of five-digit numbers that are multiples of 15 -/
def multiplesOf15 : Finset FiveDigitNumber := sorry

/-- The probability of drawing a multiple of 15 -/
def probabilityMultipleOf15 : ℚ := (multiplesOf15.card : ℚ) / (allNumbers.card : ℚ)

theorem probability_multiple_of_15 : probabilityMultipleOf15 = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_multiple_of_15_l1190_119011


namespace NUMINAMATH_CALUDE_units_digit_problem_l1190_119013

theorem units_digit_problem : ∃ n : ℤ, (8 * 19 * 1981 + 6^3 - 2^5) % 10 = 6 ∧ 0 ≤ n ∧ n < 10 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l1190_119013


namespace NUMINAMATH_CALUDE_onion_chop_time_is_four_l1190_119098

/-- Represents the time in minutes for Bill's omelet preparation tasks -/
structure OmeletPrep where
  pepper_chop_time : ℕ
  cheese_grate_time : ℕ
  assemble_cook_time : ℕ
  total_peppers : ℕ
  total_onions : ℕ
  total_omelets : ℕ
  total_prep_time : ℕ

/-- Calculates the time to chop an onion given the omelet preparation details -/
def time_to_chop_onion (prep : OmeletPrep) : ℕ :=
  let pepper_time := prep.pepper_chop_time * prep.total_peppers
  let cheese_time := prep.cheese_grate_time * prep.total_omelets
  let cook_time := prep.assemble_cook_time * prep.total_omelets
  let remaining_time := prep.total_prep_time - (pepper_time + cheese_time + cook_time)
  remaining_time / prep.total_onions

/-- Theorem stating that it takes 4 minutes to chop an onion given the specific conditions -/
theorem onion_chop_time_is_four : 
  let prep : OmeletPrep := {
    pepper_chop_time := 3,
    cheese_grate_time := 1,
    assemble_cook_time := 5,
    total_peppers := 4,
    total_onions := 2,
    total_omelets := 5,
    total_prep_time := 50
  }
  time_to_chop_onion prep = 4 := by
  sorry

end NUMINAMATH_CALUDE_onion_chop_time_is_four_l1190_119098


namespace NUMINAMATH_CALUDE_commute_time_difference_l1190_119088

theorem commute_time_difference (distance : Real) (walk_speed : Real) (train_speed : Real) (time_difference : Real) :
  distance = 1.5 ∧ 
  walk_speed = 3 ∧ 
  train_speed = 20 ∧ 
  time_difference = 25 →
  ∃ x : Real, 
    (distance / walk_speed) * 60 = (distance / train_speed) * 60 + x + time_difference ∧ 
    x = 0.5 := by
  sorry


end NUMINAMATH_CALUDE_commute_time_difference_l1190_119088


namespace NUMINAMATH_CALUDE_ellipse_intersection_dot_product_range_l1190_119077

-- Define the ellipses
def C₁ (x y : ℝ) : Prop := y^2/16 + x^2/4 = 1
def C₂ (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define points and lines
def B : ℝ × ℝ := (1, 0)
def intersectionPoint (x y : ℝ) : Prop := C₂ x y ∧ ∃ k : ℝ, y = k * (x - 1)
def lineAE (x y : ℝ) (xE yE : ℝ) : Prop := y = (yE / (xE - 2)) * (x - 2)
def lineAF (x y : ℝ) (xF yF : ℝ) : Prop := y = (yF / (xF - 2)) * (x - 2)

-- Define the dot product of vectors EM and FN
def dotProduct (xE yE xM yM xF yF xN yN : ℝ) : ℝ :=
  (xM - xE) * (xN - xF) + (yM - yE) * (yN - yF)

-- State the theorem
theorem ellipse_intersection_dot_product_range :
  ∀ xE yE xF yF xM yM xN yN : ℝ,
  C₂ 2 0 →  -- Point A(2, 0) is on C₂
  intersectionPoint xE yE →
  intersectionPoint xF yF →
  xM = 3 ∧ lineAE xM yM xE yE →
  xN = 3 ∧ lineAF xN yN xF yF →
  1 ≤ dotProduct xE yE xM yM xF yF xN yN ∧
  dotProduct xE yE xM yM xF yF xN yN < 5/4 :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_dot_product_range_l1190_119077


namespace NUMINAMATH_CALUDE_total_money_is_252_l1190_119076

/-- Represents the money redistribution process among three friends -/
def MoneyRedistribution (a j t : ℕ) : Prop :=
  -- Initial condition: Toy starts with 36 dollars
  t = 36 ∧
  -- After three rounds of redistribution, Toy ends with 36 dollars
  ∃ (a' j' : ℕ),
    -- First round: Amy doubles Jan's and Toy's amounts
    ∃ (a1 j1 t1 : ℕ),
      j1 = 2 * j ∧
      t1 = 2 * t ∧
      a1 + j1 + t1 = a + j + t ∧
    -- Second round: Jan doubles Amy's and Toy's amounts
    ∃ (a2 j2 t2 : ℕ),
      a2 = 2 * a1 ∧
      t2 = 2 * t1 ∧
      a2 + j2 + t2 = a1 + j1 + t1 ∧
    -- Third round: Toy doubles Amy's and Jan's amounts
    a' = 2 * a2 ∧
    j' = 2 * j2 ∧
    36 = t2 - (a' - a2 + j' - j2) ∧
    -- The total amount after redistribution
    a' + j' + 36 = 252

/-- The theorem stating that the total amount of money is 252 dollars -/
theorem total_money_is_252 (a j t : ℕ) :
  MoneyRedistribution a j t → a + j + t = 252 := by
  sorry

end NUMINAMATH_CALUDE_total_money_is_252_l1190_119076


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_digit_sum_27_l1190_119019

/-- Given a natural number, returns the sum of its digits. -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Returns true if the number is a three-digit number. -/
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∀ n : ℕ, isThreeDigit n → n % 9 = 0 → digitSum n = 27 → n ≤ 999 := by sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_digit_sum_27_l1190_119019


namespace NUMINAMATH_CALUDE_triangle_area_maximized_l1190_119091

theorem triangle_area_maximized (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  a / (Real.sin A) = c / (Real.sin C) ∧
  Real.tan A = 2 * Real.tan B ∧
  b = Real.sqrt 2 →
  (∀ A' B' C' a' b' c' : ℝ,
    0 < A' ∧ 0 < B' ∧ 0 < C' ∧
    A' + B' + C' = π ∧
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧
    a' / (Real.sin A') = b' / (Real.sin B') ∧
    a' / (Real.sin A') = c' / (Real.sin C') ∧
    Real.tan A' = 2 * Real.tan B' ∧
    b' = Real.sqrt 2 →
    1/2 * a * b * Real.sin C ≥ 1/2 * a' * b' * Real.sin C') →
  a = Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_maximized_l1190_119091


namespace NUMINAMATH_CALUDE_fruit_cost_proof_l1190_119005

/-- Given the cost of fruits, prove the cost of a different combination -/
theorem fruit_cost_proof (cost_six_apples_three_oranges : ℝ) 
                         (cost_one_apple : ℝ) : 
  cost_six_apples_three_oranges = 1.77 →
  cost_one_apple = 0.21 →
  2 * cost_one_apple + 5 * ((cost_six_apples_three_oranges - 6 * cost_one_apple) / 3) = 1.27 := by
  sorry

end NUMINAMATH_CALUDE_fruit_cost_proof_l1190_119005


namespace NUMINAMATH_CALUDE_meeting_speed_l1190_119022

theorem meeting_speed
  (total_distance : ℝ)
  (time : ℝ)
  (speed_diff : ℝ)
  (h1 : total_distance = 45)
  (h2 : time = 5)
  (h3 : speed_diff = 1)
  (h4 : ∀ (v_a v_b : ℝ), v_a = v_b + speed_diff → v_a * time + v_b * time = total_distance)
  : ∃ (v_a : ℝ), v_a = 5 ∧ ∃ (v_b : ℝ), v_a = v_b + speed_diff ∧ v_a * time + v_b * time = total_distance :=
by sorry

end NUMINAMATH_CALUDE_meeting_speed_l1190_119022


namespace NUMINAMATH_CALUDE_volume_of_rhombus_revolution_l1190_119083

/-- A rhombus with side length 1 and shorter diagonal equal to its side -/
structure Rhombus where
  side_length : ℝ
  side_length_is_one : side_length = 1
  shorter_diagonal_eq_side : ℝ
  shorter_diagonal_eq_side_prop : shorter_diagonal_eq_side = side_length

/-- The volume of the solid of revolution formed by rotating the rhombus -/
noncomputable def volume_of_revolution (r : Rhombus) : ℝ := 
  3 * Real.pi / 2

/-- Theorem stating that the volume of the solid of revolution is 3π/2 -/
theorem volume_of_rhombus_revolution (r : Rhombus) : 
  volume_of_revolution r = 3 * Real.pi / 2 := by sorry

end NUMINAMATH_CALUDE_volume_of_rhombus_revolution_l1190_119083


namespace NUMINAMATH_CALUDE_range_of_y_l1190_119065

-- Define the function f
def f (x : ℝ) : ℝ := |x^2 - x| - 4*x

-- State the theorem
theorem range_of_y (y : ℝ) :
  (∃ x : ℝ, 0 < x ∧ x < 2 ∧ f x = 0) ↔ -4 < y ∧ y < 12 :=
sorry

end NUMINAMATH_CALUDE_range_of_y_l1190_119065


namespace NUMINAMATH_CALUDE_no_loops_in_process_flowchart_l1190_119043

-- Define the basic concepts
def ProcessFlowchart : Type := Unit
def AlgorithmFlowchart : Type := Unit
def Process : Type := Unit
def FlowLine : Type := Unit

-- Define the properties of process flowcharts
def is_similar_to (pf : ProcessFlowchart) (af : AlgorithmFlowchart) : Prop := sorry
def refine_step_by_step (p : Process) : Prop := sorry
def connect_adjacent_processes (fl : FlowLine) : Prop := sorry
def is_directional (fl : FlowLine) : Prop := sorry

-- Define the concept of a loop
def Loop : Type := Unit
def contains_loop (pf : ProcessFlowchart) (l : Loop) : Prop := sorry

-- State the theorem
theorem no_loops_in_process_flowchart (pf : ProcessFlowchart) (af : AlgorithmFlowchart) 
  (p : Process) (fl : FlowLine) :
  is_similar_to pf af →
  refine_step_by_step p →
  connect_adjacent_processes fl →
  is_directional fl →
  ∀ l : Loop, ¬ contains_loop pf l := by
  sorry

end NUMINAMATH_CALUDE_no_loops_in_process_flowchart_l1190_119043


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1190_119016

/-- A point is in the fourth quadrant if its x-coordinate is positive and its y-coordinate is negative -/
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- The point P has coordinates (2, -3) -/
def P : ℝ × ℝ := (2, -3)

theorem point_in_fourth_quadrant :
  fourth_quadrant P.1 P.2 := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1190_119016


namespace NUMINAMATH_CALUDE_coefficient_value_l1190_119026

-- Define the polynomial P(x)
def P (c : ℝ) (x : ℝ) : ℝ := x^3 + 4*x^2 + c*x - 20

-- Theorem statement
theorem coefficient_value (c : ℝ) : 
  (∀ x, P c x = 0 → x = 5) → c = -41 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_value_l1190_119026


namespace NUMINAMATH_CALUDE_sin_135_degrees_l1190_119079

theorem sin_135_degrees : Real.sin (135 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_135_degrees_l1190_119079


namespace NUMINAMATH_CALUDE_intersection_implies_m_value_subset_implies_m_range_l1190_119001

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 6 / (x + 1) ≥ 1}
def B (m : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*x + 2*m < 0}

-- Theorem 1
theorem intersection_implies_m_value :
  ∀ m : ℝ, (A ∩ B m = {x : ℝ | -1 < x ∧ x < 4}) → m = -4 := by sorry

-- Theorem 2
theorem subset_implies_m_range :
  ∀ m : ℝ, (B m ⊆ A) → m ≥ -3/2 := by sorry

end NUMINAMATH_CALUDE_intersection_implies_m_value_subset_implies_m_range_l1190_119001


namespace NUMINAMATH_CALUDE_triangle_angle_determinant_l1190_119031

theorem triangle_angle_determinant (A B C : ℝ) (h : A + B + C = Real.pi) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := λ i j =>
    if i = j then Real.sin (2 * (if i = 0 then A else if i = 1 then B else C))
    else 1
  Matrix.det M = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_determinant_l1190_119031


namespace NUMINAMATH_CALUDE_periodic_even_function_extension_l1190_119099

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem periodic_even_function_extension
  (f : ℝ → ℝ)
  (h_periodic : is_periodic f 2)
  (h_even : is_even f)
  (h_defined : ∀ x ∈ Set.Icc 2 3, f x = -2 * (x - 3)^2 + 4) :
  ∀ x ∈ Set.Icc 0 2, f x = -2 * (x - 1)^2 + 4 :=
sorry

end NUMINAMATH_CALUDE_periodic_even_function_extension_l1190_119099


namespace NUMINAMATH_CALUDE_total_distance_is_6300_l1190_119093

/-- The distance Bomin walked in kilometers -/
def bomin_km : ℝ := 2

/-- The additional distance Bomin walked in meters -/
def bomin_additional_m : ℝ := 600

/-- The distance Yunshik walked in meters -/
def yunshik_m : ℝ := 3700

/-- Conversion factor from kilometers to meters -/
def km_to_m : ℝ := 1000

/-- The total distance walked by Bomin and Yunshik in meters -/
def total_distance : ℝ := (bomin_km * km_to_m + bomin_additional_m) + yunshik_m

theorem total_distance_is_6300 : total_distance = 6300 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_is_6300_l1190_119093


namespace NUMINAMATH_CALUDE_base_4_divisible_by_19_l1190_119090

def base_4_to_decimal (a b c d : ℕ) : ℕ := a * 4^3 + b * 4^2 + c * 4 + d

theorem base_4_divisible_by_19 :
  ∃! x : ℕ, x < 4 ∧ 19 ∣ base_4_to_decimal 2 1 x 2 :=
by
  sorry

end NUMINAMATH_CALUDE_base_4_divisible_by_19_l1190_119090


namespace NUMINAMATH_CALUDE_sum_of_leading_digits_is_seven_l1190_119038

/-- N is a 200-digit number where each digit is 8 -/
def N : ℕ := 8888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888

/-- f(r) is the leading digit of the r-th root of N -/
def f (r : ℕ) : ℕ := sorry

/-- The sum of f(r) for r from 3 to 7 is 7 -/
theorem sum_of_leading_digits_is_seven :
  f 3 + f 4 + f 5 + f 6 + f 7 = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_leading_digits_is_seven_l1190_119038


namespace NUMINAMATH_CALUDE_symmetric_point_y_axis_l1190_119032

/-- A point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- The symmetric point with respect to the y-axis --/
def symmetricYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- The original point (2,5) --/
def originalPoint : Point :=
  { x := 2, y := 5 }

/-- The expected symmetric point (-2,5) --/
def expectedSymmetricPoint : Point :=
  { x := -2, y := 5 }

theorem symmetric_point_y_axis :
  symmetricYAxis originalPoint = expectedSymmetricPoint := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_y_axis_l1190_119032


namespace NUMINAMATH_CALUDE_sqrt_two_sqrt_two_power_l1190_119000

theorem sqrt_two_sqrt_two_power : (((2 * Real.sqrt 2) ^ 4).sqrt) ^ 3 = 512 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_sqrt_two_power_l1190_119000


namespace NUMINAMATH_CALUDE_tangent_line_problem_l1190_119025

theorem tangent_line_problem (x y a : ℝ) :
  (∃ m : ℝ, y = 3 * x - 2 ∧ y = x^3 - 2 * a ∧ 3 * x^2 = 3) →
  (a = 0 ∨ a = 2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_problem_l1190_119025


namespace NUMINAMATH_CALUDE_triangle_cannot_be_formed_l1190_119063

theorem triangle_cannot_be_formed (a b c : ℝ) (h1 : a = 8) (h2 : b = 6) (h3 : c = 9) : 
  ¬ (∃ (a' b' c' : ℝ), a' = a * 1.5 ∧ b' = b * (1 - 0.333) ∧ c' = c ∧ 
    a' + b' > c' ∧ a' + c' > b' ∧ b' + c' > a') :=
by sorry

end NUMINAMATH_CALUDE_triangle_cannot_be_formed_l1190_119063


namespace NUMINAMATH_CALUDE_x_one_value_l1190_119060

theorem x_one_value (x₁ x₂ x₃ x₄ : ℝ) 
  (h_order : 0 ≤ x₄ ∧ x₄ ≤ x₃ ∧ x₃ ≤ x₂ ∧ x₂ ≤ x₁ ∧ x₁ ≤ 1)
  (h_eq : (1 - x₁)^2 + (x₁ - x₂)^2 + (x₂ - x₃)^2 + (x₃ - x₄)^2 + x₄^2 = 1/3) :
  x₁ = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_x_one_value_l1190_119060


namespace NUMINAMATH_CALUDE_seven_balls_three_boxes_l1190_119080

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 64 ways to distribute 7 distinguishable balls into 3 indistinguishable boxes -/
theorem seven_balls_three_boxes : distribute 7 3 = 64 := by sorry

end NUMINAMATH_CALUDE_seven_balls_three_boxes_l1190_119080


namespace NUMINAMATH_CALUDE_tangent_point_coordinates_l1190_119050

def f (x : ℝ) := x^4 - x

theorem tangent_point_coordinates :
  ∀ m n : ℝ,
  (∃ k : ℝ, f m = n ∧ 4 * m^3 - 1 = 3) →
  m = 1 ∧ n = 0 := by
sorry

end NUMINAMATH_CALUDE_tangent_point_coordinates_l1190_119050


namespace NUMINAMATH_CALUDE_congruence_problem_l1190_119028

theorem congruence_problem (m : ℕ) : 
  163 * 937 ≡ m [ZMOD 60] → 0 ≤ m → m < 60 → m = 11 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l1190_119028


namespace NUMINAMATH_CALUDE_cafeteria_apples_l1190_119094

theorem cafeteria_apples (initial : ℕ) : 
  initial - 20 + 28 = 46 → initial = 38 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_apples_l1190_119094


namespace NUMINAMATH_CALUDE_square_sum_lower_bound_l1190_119046

theorem square_sum_lower_bound (a b : ℝ) 
  (h1 : a^3 - b^3 = 2) 
  (h2 : a^5 - b^5 ≥ 4) : 
  a^2 + b^2 ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_square_sum_lower_bound_l1190_119046


namespace NUMINAMATH_CALUDE_james_carrot_sticks_l1190_119024

/-- Given that James originally had 50 carrot sticks, ate 22 before dinner,
    ate 15 after dinner, and gave away 8 during dinner, prove that he has 5 left. -/
theorem james_carrot_sticks (original : ℕ) (eaten_before : ℕ) (eaten_after : ℕ) (given_away : ℕ)
    (h1 : original = 50)
    (h2 : eaten_before = 22)
    (h3 : eaten_after = 15)
    (h4 : given_away = 8) :
    original - eaten_before - eaten_after - given_away = 5 := by
  sorry

end NUMINAMATH_CALUDE_james_carrot_sticks_l1190_119024


namespace NUMINAMATH_CALUDE_marks_initial_money_l1190_119066

theorem marks_initial_money (x : ℝ) : 
  x / 2 + 14 + x / 3 + 16 = x → x = 180 :=
by sorry

end NUMINAMATH_CALUDE_marks_initial_money_l1190_119066


namespace NUMINAMATH_CALUDE_frog_jump_probability_l1190_119010

/-- Represents a jump of the frog -/
structure Jump where
  direction : ℝ × ℝ
  length : ℝ
  random_direction : Bool

/-- Represents the frog's journey -/
structure FrogJourney where
  jumps : List Jump
  final_position : ℝ × ℝ

/-- The probability of the frog's final position being within 1 meter of the start -/
def probability_within_one_meter (journey : FrogJourney) : ℝ :=
  sorry

/-- Theorem stating the probability of the frog's final position being within 1 meter of the start -/
theorem frog_jump_probability :
  ∀ (journey : FrogJourney),
    journey.jumps.length = 4 ∧
    (∀ jump ∈ journey.jumps, jump.length = 1 ∧ jump.random_direction) →
    probability_within_one_meter journey = 1/5 :=
  sorry

end NUMINAMATH_CALUDE_frog_jump_probability_l1190_119010


namespace NUMINAMATH_CALUDE_a_to_b_value_l1190_119058

theorem a_to_b_value (a b : ℝ) (h : Real.sqrt (a + 2) + (b - 3)^2 = 0) : a^b = -8 := by
  sorry

end NUMINAMATH_CALUDE_a_to_b_value_l1190_119058


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l1190_119059

theorem complex_number_in_fourth_quadrant (a : ℝ) :
  let z : ℂ := (a^2 - 4*a + 5) - 6*Complex.I
  (z.re > 0) ∧ (z.im < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l1190_119059


namespace NUMINAMATH_CALUDE_expand_expression_l1190_119062

theorem expand_expression (x : ℝ) : 20 * (3 * x + 4) = 60 * x + 80 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1190_119062


namespace NUMINAMATH_CALUDE_trig_ratio_problem_l1190_119072

theorem trig_ratio_problem (x : Real) (h : (1 + Real.sin x) / Real.cos x = 2) :
  (1 - Real.sin x) / Real.cos x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_ratio_problem_l1190_119072


namespace NUMINAMATH_CALUDE_coefficient_x2y2_l1190_119056

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the expansion of (1+x)^7(1+y)^4
def expansion : ℕ → ℕ → ℕ := sorry

-- Theorem statement
theorem coefficient_x2y2 : expansion 2 2 = 126 := by sorry

end NUMINAMATH_CALUDE_coefficient_x2y2_l1190_119056


namespace NUMINAMATH_CALUDE_pants_price_decrease_percentage_l1190_119042

/-- Proves that the percentage decrease in selling price is 20% given the conditions --/
theorem pants_price_decrease_percentage (purchase_price : ℝ) (markup_percentage : ℝ) (gross_profit : ℝ) :
  purchase_price = 210 →
  markup_percentage = 0.25 →
  gross_profit = 14 →
  let original_price := purchase_price / (1 - markup_percentage)
  let final_price := purchase_price + gross_profit
  let price_decrease := original_price - final_price
  let percentage_decrease := (price_decrease / original_price) * 100
  percentage_decrease = 20 := by
  sorry

end NUMINAMATH_CALUDE_pants_price_decrease_percentage_l1190_119042


namespace NUMINAMATH_CALUDE_inequality_solution_l1190_119017

-- Define the inequality
def inequality (x a : ℝ) : Prop := x^2 - 2*a*x - 3*a^2 < 0

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ :=
  {x | inequality x a}

-- Theorem statement
theorem inequality_solution :
  ∀ a : ℝ,
    (a = 0 → solution_set a = ∅) ∧
    (a > 0 → solution_set a = {x | -a < x ∧ x < 3*a}) ∧
    (a < 0 → solution_set a = {x | 3*a < x ∧ x < -a}) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1190_119017


namespace NUMINAMATH_CALUDE_integral_of_f_with_min_neg_one_l1190_119089

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + m

-- State the theorem
theorem integral_of_f_with_min_neg_one (m : ℝ) :
  (∃ (x : ℝ), ∀ (y : ℝ), f m x ≤ f m y) ∧ 
  (∃ (x : ℝ), f m x = -1) →
  ∫ x in (1 : ℝ)..(2 : ℝ), f m x = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_integral_of_f_with_min_neg_one_l1190_119089


namespace NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l1190_119053

-- Define a function to get the nth prime number
def nthPrime (n : ℕ) : ℕ := sorry

-- Define a function to sum the first n prime numbers
def sumFirstNPrimes (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_first_six_primes_mod_seventh_prime :
  sumFirstNPrimes 6 % nthPrime 7 = 7 := by sorry

end NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l1190_119053


namespace NUMINAMATH_CALUDE_distance_between_specific_lines_l1190_119039

/-- Line represented by a parametric equation -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Line represented by a slope-intercept equation -/
structure SlopeInterceptLine where
  slope : ℝ
  intercept : ℝ

/-- The distance between two lines -/
def distance_between_lines (l₁ : ParametricLine) (l₂ : SlopeInterceptLine) : ℝ :=
  sorry

/-- The given problem statement -/
theorem distance_between_specific_lines :
  let l₁ : ParametricLine := {
    x := λ t => 1 + t,
    y := λ t => 1 + 3*t
  }
  let l₂ : SlopeInterceptLine := {
    slope := 3,
    intercept := 4
  }
  distance_between_lines l₁ l₂ = 3 * Real.sqrt 10 / 5 :=
sorry

end NUMINAMATH_CALUDE_distance_between_specific_lines_l1190_119039


namespace NUMINAMATH_CALUDE_stan_magician_payment_l1190_119027

def magician_payment (hourly_rate : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) (num_weeks : ℕ) : ℕ :=
  hourly_rate * hours_per_day * days_per_week * num_weeks

theorem stan_magician_payment :
  magician_payment 60 3 7 2 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_stan_magician_payment_l1190_119027


namespace NUMINAMATH_CALUDE_gcd_315_168_l1190_119078

theorem gcd_315_168 : Nat.gcd 315 168 = 21 := by
  sorry

end NUMINAMATH_CALUDE_gcd_315_168_l1190_119078


namespace NUMINAMATH_CALUDE_james_night_out_cost_l1190_119044

/-- Calculate the total amount James spent for a night out -/
theorem james_night_out_cost : 
  let entry_fee : ℚ := 25
  let friends_count : ℕ := 8
  let rounds_count : ℕ := 3
  let james_drinks_count : ℕ := 7
  let cocktail_price : ℚ := 8
  let non_alcoholic_price : ℚ := 4
  let james_cocktails_count : ℕ := 6
  let burger_price : ℚ := 18
  let tip_percentage : ℚ := 0.25

  let friends_drinks_cost := friends_count * rounds_count * cocktail_price
  let james_drinks_cost := james_cocktails_count * cocktail_price + 
                           (james_drinks_count - james_cocktails_count) * non_alcoholic_price
  let food_cost := burger_price
  let subtotal := entry_fee + friends_drinks_cost + james_drinks_cost + food_cost
  let tip := subtotal * tip_percentage
  let total_cost := subtotal + tip

  total_cost = 358.75 := by sorry

end NUMINAMATH_CALUDE_james_night_out_cost_l1190_119044


namespace NUMINAMATH_CALUDE_log_101600_value_l1190_119064

-- Define the logarithm function
noncomputable def log : ℝ → ℝ := Real.log

-- State the theorem
theorem log_101600_value (h : log 102 = 0.3010) : log 101600 = 3.3010 := by
  sorry

end NUMINAMATH_CALUDE_log_101600_value_l1190_119064


namespace NUMINAMATH_CALUDE_acme_vowel_soup_sequences_l1190_119023

/-- The number of distinct elements in the set -/
def n : ℕ := 5

/-- The number of times each element appears -/
def k : ℕ := 6

/-- The length of the sequences to be formed -/
def seq_length : ℕ := 6

/-- The number of possible sequences -/
def num_sequences : ℕ := n ^ seq_length

theorem acme_vowel_soup_sequences :
  num_sequences = 15625 :=
sorry

end NUMINAMATH_CALUDE_acme_vowel_soup_sequences_l1190_119023


namespace NUMINAMATH_CALUDE_prob_sum_less_than_10_l1190_119081

/-- A fair die with faces labeled 1 to 6 -/
def FairDie : Finset ℕ := Finset.range 6

/-- The sample space of rolling a fair die twice -/
def SampleSpace : Finset (ℕ × ℕ) :=
  FairDie.product FairDie

/-- The event where the sum of face values is less than 10 -/
def EventSumLessThan10 : Finset (ℕ × ℕ) :=
  SampleSpace.filter (fun (x, y) => x + y < 10)

/-- Theorem: The probability of the sum of face values being less than 10
    when rolling a fair six-sided die twice is 5/6 -/
theorem prob_sum_less_than_10 :
  (EventSumLessThan10.card : ℚ) / SampleSpace.card = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_less_than_10_l1190_119081


namespace NUMINAMATH_CALUDE_decagon_circle_intersection_undecagon_no_circle_intersection_l1190_119047

/-- Represents a polygon with n sides -/
structure Polygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- Represents a circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Constructs circles on each side of a polygon -/
def constructCircles {n : ℕ} (p : Polygon n) : Fin n → Circle :=
  sorry

/-- Checks if a point is a common intersection of all circles -/
def isCommonIntersection {n : ℕ} (p : Polygon n) (circles : Fin n → Circle) (point : ℝ × ℝ) : Prop :=
  sorry

/-- Checks if a point is a vertex of the polygon -/
def isVertex {n : ℕ} (p : Polygon n) (point : ℝ × ℝ) : Prop :=
  sorry

theorem decagon_circle_intersection :
  ∃ (p : Polygon 10) (point : ℝ × ℝ),
    isCommonIntersection p (constructCircles p) point ∧
    ¬isVertex p point :=
  sorry

theorem undecagon_no_circle_intersection :
  ∀ (p : Polygon 11) (point : ℝ × ℝ),
    isCommonIntersection p (constructCircles p) point →
    isVertex p point :=
  sorry

end NUMINAMATH_CALUDE_decagon_circle_intersection_undecagon_no_circle_intersection_l1190_119047


namespace NUMINAMATH_CALUDE_range_of_a_l1190_119051

open Set Real

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | -2 - a < x ∧ x < a}

-- Define propositions p and q
def p (a : ℝ) : Prop := (1 : ℝ) ∈ A a
def q (a : ℝ) : Prop := (2 : ℝ) ∈ A a

-- Theorem statement
theorem range_of_a (a : ℝ) 
  (h1 : a > 0) 
  (h2 : p a ∨ q a) 
  (h3 : ¬(p a ∧ q a)) : 
  1 < a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1190_119051


namespace NUMINAMATH_CALUDE_abie_chips_count_l1190_119006

theorem abie_chips_count (initial bags_given bags_bought : ℕ) 
  (h1 : initial = 20)
  (h2 : bags_given = 4)
  (h3 : bags_bought = 6) : 
  initial - bags_given + bags_bought = 22 := by
  sorry

end NUMINAMATH_CALUDE_abie_chips_count_l1190_119006


namespace NUMINAMATH_CALUDE_seven_unit_disks_cover_radius_two_disk_l1190_119096

-- Define a disk as a pair (center, radius)
def Disk := ℝ × ℝ × ℝ

-- Define a function to check if a point is covered by a disk
def is_covered (point : ℝ × ℝ) (disk : Disk) : Prop :=
  let (cx, cy, r) := disk
  (point.1 - cx)^2 + (point.2 - cy)^2 ≤ r^2

-- Define a function to check if a point is covered by any disk in a list
def is_covered_by_any (point : ℝ × ℝ) (disks : List Disk) : Prop :=
  ∃ d ∈ disks, is_covered point d

-- Define the main theorem
theorem seven_unit_disks_cover_radius_two_disk :
  ∃ (arrangement : List Disk),
    (arrangement.length = 7) ∧
    (∀ d ∈ arrangement, d.2.2 = 1) ∧
    (∀ point : ℝ × ℝ, point.1^2 + point.2^2 ≤ 4 → is_covered_by_any point arrangement) :=
sorry

end NUMINAMATH_CALUDE_seven_unit_disks_cover_radius_two_disk_l1190_119096


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1190_119030

/-- Given a boat that travels 11 km/hr along a stream and 7 km/hr against the same stream,
    its speed in still water is 9 km/hr. -/
theorem boat_speed_in_still_water :
  ∀ (boat_speed stream_speed : ℝ),
    boat_speed + stream_speed = 11 →
    boat_speed - stream_speed = 7 →
    boat_speed = 9 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1190_119030


namespace NUMINAMATH_CALUDE_ryosuke_trip_cost_l1190_119003

/-- Calculates the cost of gas for a trip given the odometer readings, fuel efficiency, and gas price -/
def gas_cost (initial_reading final_reading : ℕ) (fuel_efficiency : ℚ) (gas_price : ℚ) : ℚ :=
  let distance := final_reading - initial_reading
  let gas_used := (distance : ℚ) / fuel_efficiency
  gas_used * gas_price

/-- Proves that the cost of gas for Ryosuke's trip is $5.04 -/
theorem ryosuke_trip_cost :
  let initial_reading : ℕ := 74580
  let final_reading : ℕ := 74610
  let fuel_efficiency : ℚ := 25
  let gas_price : ℚ := 21/5
  gas_cost initial_reading final_reading fuel_efficiency gas_price = 504/100 := by
  sorry

end NUMINAMATH_CALUDE_ryosuke_trip_cost_l1190_119003


namespace NUMINAMATH_CALUDE_longer_piece_length_l1190_119034

def board_length : ℝ := 69

theorem longer_piece_length :
  ∀ (short_piece long_piece : ℝ),
  short_piece + long_piece = board_length →
  long_piece = 2 * short_piece →
  long_piece = 46 :=
by
  sorry

end NUMINAMATH_CALUDE_longer_piece_length_l1190_119034


namespace NUMINAMATH_CALUDE_tunnel_length_l1190_119054

/-- The length of a tunnel given train parameters -/
theorem tunnel_length (train_length : ℝ) (train_speed : ℝ) (time : ℝ) :
  train_length = 90 →
  train_speed = 160 →
  time = 3 →
  train_speed * time - train_length = 390 := by
  sorry

end NUMINAMATH_CALUDE_tunnel_length_l1190_119054


namespace NUMINAMATH_CALUDE_female_officers_count_l1190_119082

theorem female_officers_count 
  (total_on_duty : ℕ) 
  (female_percentage : ℚ) 
  (female_on_duty_ratio : ℚ) :
  total_on_duty = 144 →
  female_percentage = 18 / 100 →
  female_on_duty_ratio = 1 / 2 →
  ∃ (total_female : ℕ), 
    (↑(total_on_duty) * female_on_duty_ratio : ℚ) = 
    (↑total_female * female_percentage : ℚ) ∧
    total_female = 400 :=
by sorry

end NUMINAMATH_CALUDE_female_officers_count_l1190_119082


namespace NUMINAMATH_CALUDE_lollipops_eaten_by_children_l1190_119067

/-- The number of lollipops Sushi's father bought -/
def initial_lollipops : ℕ := 12

/-- The number of lollipops left -/
def remaining_lollipops : ℕ := 7

/-- The number of lollipops eaten by the children -/
def eaten_lollipops : ℕ := initial_lollipops - remaining_lollipops

theorem lollipops_eaten_by_children : eaten_lollipops = 5 := by
  sorry

end NUMINAMATH_CALUDE_lollipops_eaten_by_children_l1190_119067


namespace NUMINAMATH_CALUDE_gloria_pine_tree_price_l1190_119007

/-- Proves that the price per pine tree is $200 given the conditions of Gloria's cabin purchase --/
theorem gloria_pine_tree_price :
  let cabin_price : ℕ := 129000
  let initial_cash : ℕ := 150
  let cypress_trees : ℕ := 20
  let pine_trees : ℕ := 600
  let maple_trees : ℕ := 24
  let cypress_price : ℕ := 100
  let maple_price : ℕ := 300
  let remaining_cash : ℕ := 350
  let pine_price : ℕ := (cabin_price - initial_cash + remaining_cash - 
    (cypress_trees * cypress_price + maple_trees * maple_price)) / pine_trees
  pine_price = 200 := by sorry

end NUMINAMATH_CALUDE_gloria_pine_tree_price_l1190_119007


namespace NUMINAMATH_CALUDE_five_points_two_small_triangles_l1190_119074

-- Define a triangular region with unit area
def UnitTriangle : Set (ℝ × ℝ) := sorry

-- Define a function to calculate the area of a triangle given three points
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem five_points_two_small_triangles 
  (points : Finset (ℝ × ℝ)) 
  (h1 : points.card = 5) 
  (h2 : ∀ p ∈ points, p ∈ UnitTriangle) : 
  ∃ (t1 t2 : Finset (ℝ × ℝ)), 
    t1 ⊆ points ∧ t2 ⊆ points ∧ 
    t1.card = 3 ∧ t2.card = 3 ∧ 
    t1 ≠ t2 ∧
    (∃ (p1 p2 p3 : ℝ × ℝ), p1 ∈ t1 ∧ p2 ∈ t1 ∧ p3 ∈ t1 ∧ triangleArea p1 p2 p3 ≤ 1/4) ∧
    (∃ (q1 q2 q3 : ℝ × ℝ), q1 ∈ t2 ∧ q2 ∈ t2 ∧ q3 ∈ t2 ∧ triangleArea q1 q2 q3 ≤ 1/4) :=
by sorry

end NUMINAMATH_CALUDE_five_points_two_small_triangles_l1190_119074


namespace NUMINAMATH_CALUDE_map_distance_calculation_l1190_119037

/-- Represents the scale of a map in feet per inch -/
def map_scale : ℝ := 500

/-- Represents the length of a line segment on the map in inches -/
def map_length : ℝ := 7.2

/-- Calculates the actual distance represented by a map length -/
def actual_distance (scale : ℝ) (map_length : ℝ) : ℝ := scale * map_length

/-- Theorem: The actual distance represented by a 7.2-inch line segment on a map with a scale of 1 inch = 500 feet is 3600 feet -/
theorem map_distance_calculation :
  actual_distance map_scale map_length = 3600 := by
  sorry

end NUMINAMATH_CALUDE_map_distance_calculation_l1190_119037


namespace NUMINAMATH_CALUDE_phone_number_remainder_l1190_119069

theorem phone_number_remainder :
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧
  (312837 % n = 96) ∧ (310650 % n = 96) := by
  sorry

end NUMINAMATH_CALUDE_phone_number_remainder_l1190_119069


namespace NUMINAMATH_CALUDE_N_squared_eq_N_minus_26I_l1190_119061

def N : Matrix (Fin 2) (Fin 2) ℝ := !![3, 8; -4, -2]

theorem N_squared_eq_N_minus_26I : 
  N ^ 2 = N - 26 • (1 : Matrix (Fin 2) (Fin 2) ℝ) := by sorry

end NUMINAMATH_CALUDE_N_squared_eq_N_minus_26I_l1190_119061


namespace NUMINAMATH_CALUDE_complement_not_always_smaller_than_supplement_l1190_119045

theorem complement_not_always_smaller_than_supplement :
  ¬ (∀ θ : Real, 0 < θ ∧ θ < π → (π / 2 - θ) < (π - θ)) := by
  sorry

end NUMINAMATH_CALUDE_complement_not_always_smaller_than_supplement_l1190_119045


namespace NUMINAMATH_CALUDE_cn_relation_sqrt_c_equals_c8_l1190_119057

-- Define cn as a function that returns a natural number with n ones
def cn (n : ℕ) : ℕ :=
  -- Implementation details omitted
  sorry

-- Define the relation between cn and cn+1
theorem cn_relation (n : ℕ) : cn (n + 1) = 10 * cn n + 1 := by sorry

-- Define c
def c : ℕ := 123456787654321

-- Theorem to prove
theorem sqrt_c_equals_c8 : ∃ (x : ℕ), x * x = c ∧ x = cn 8 := by sorry

end NUMINAMATH_CALUDE_cn_relation_sqrt_c_equals_c8_l1190_119057


namespace NUMINAMATH_CALUDE_blocks_added_l1190_119048

theorem blocks_added (initial_blocks final_blocks : ℕ) 
  (h1 : initial_blocks = 86) 
  (h2 : final_blocks = 95) : 
  final_blocks - initial_blocks = 9 := by
  sorry

end NUMINAMATH_CALUDE_blocks_added_l1190_119048


namespace NUMINAMATH_CALUDE_reflection_line_equation_l1190_119095

-- Define the points
def P : ℝ × ℝ := (3, 4)
def Q : ℝ × ℝ := (8, 9)
def R : ℝ × ℝ := (-5, 7)
def P' : ℝ × ℝ := (3, -6)
def Q' : ℝ × ℝ := (8, -11)
def R' : ℝ × ℝ := (-5, -9)

-- Define the line of reflection
def M : ℝ → ℝ := λ x => -1

-- Theorem statement
theorem reflection_line_equation :
  (∀ x y, M x = y ↔ y = -1) ∧
  (P.1 = P'.1 ∧ P.2 + P'.2 = 2 * (M P.1)) ∧
  (Q.1 = Q'.1 ∧ Q.2 + Q'.2 = 2 * (M Q.1)) ∧
  (R.1 = R'.1 ∧ R.2 + R'.2 = 2 * (M R.1)) :=
by sorry

end NUMINAMATH_CALUDE_reflection_line_equation_l1190_119095


namespace NUMINAMATH_CALUDE_divisibility_of_sum_of_powers_l1190_119015

theorem divisibility_of_sum_of_powers : 
  let n : ℕ := 3^105 + 4^105
  ∃ (a b c d : ℕ), n = 13 * a ∧ n = 49 * b ∧ n = 181 * c ∧ n = 379 * d ∧
  ¬(∃ (e : ℕ), n = 5 * e) ∧ ¬(∃ (f : ℕ), n = 11 * f) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_sum_of_powers_l1190_119015


namespace NUMINAMATH_CALUDE_sqrt_diff_approx_three_l1190_119084

theorem sqrt_diff_approx_three (k : ℕ) (h : k ≥ 7) :
  |Real.sqrt (9 * (k + 1)^2 + (k + 1)) - Real.sqrt (9 * k^2 + k) - 3| < (1 : ℝ) / 1000 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_diff_approx_three_l1190_119084


namespace NUMINAMATH_CALUDE_preimage_of_four_l1190_119012

def f (x : ℝ) : ℝ := x^2

theorem preimage_of_four (x : ℝ) : f x = 4 ↔ x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_four_l1190_119012


namespace NUMINAMATH_CALUDE_last_two_digits_of_sum_l1190_119008

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def lastTwoDigits (n : ℕ) : ℕ := n % 100

def sumSequence : ℕ → ℕ
  | 0 => 0
  | n + 1 => factorial (7 * (n + 1)) * 3 + sumSequence n

theorem last_two_digits_of_sum :
  lastTwoDigits (sumSequence 15) = 20 := by sorry

end NUMINAMATH_CALUDE_last_two_digits_of_sum_l1190_119008


namespace NUMINAMATH_CALUDE_prob_three_even_d20_l1190_119009

/-- A fair 20-sided die -/
def D20 : Type := Fin 20

/-- The probability of rolling an even number on a D20 -/
def prob_even : ℚ := 1/2

/-- The number of dice rolled -/
def n : ℕ := 5

/-- The number of dice showing even numbers -/
def k : ℕ := 3

/-- The probability of rolling exactly k even numbers out of n rolls -/
def prob_k_even (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

theorem prob_three_even_d20 :
  prob_k_even n k prob_even = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_even_d20_l1190_119009
