import Mathlib

namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1947_194713

theorem absolute_value_inequality (x : ℝ) : 
  abs x + abs (2 * x - 3) ≥ 6 ↔ x ≤ -1 ∨ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1947_194713


namespace NUMINAMATH_CALUDE_garden_perimeter_l1947_194769

theorem garden_perimeter : 
  ∀ (length breadth perimeter : ℝ),
  length = 260 →
  breadth = 190 →
  perimeter = 2 * (length + breadth) →
  perimeter = 900 :=
by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l1947_194769


namespace NUMINAMATH_CALUDE_workshop_payment_digit_l1947_194798

-- Define the total payment as 2B0 where B is a single digit
def total_payment (B : Nat) : Nat := 200 + 10 * B

-- Define the condition that B is a single digit
def is_single_digit (B : Nat) : Prop := B ≥ 0 ∧ B ≤ 9

-- Define the condition that the payment is equally divisible among 15 people
def is_equally_divisible (payment : Nat) : Prop := 
  ∃ (individual_payment : Nat), payment = 15 * individual_payment

-- Theorem statement
theorem workshop_payment_digit :
  ∀ B : Nat, is_single_digit B → 
  (is_equally_divisible (total_payment B) ↔ (B = 1 ∨ B = 4)) :=
sorry

end NUMINAMATH_CALUDE_workshop_payment_digit_l1947_194798


namespace NUMINAMATH_CALUDE_min_value_xy_l1947_194784

theorem min_value_xy (x y : ℝ) (hx : x > 1) (hy : y > 1) (h_log : Real.log x * Real.log y = Real.log 3) :
  ∀ z, x * y ≥ z → z ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_xy_l1947_194784


namespace NUMINAMATH_CALUDE_AlF3_MgCl2_cell_potential_l1947_194746

/-- Standard reduction potential for Al^3+/Al in volts -/
def E_Al : ℝ := -1.66

/-- Standard reduction potential for Mg^2+/Mg in volts -/
def E_Mg : ℝ := -2.37

/-- Calculate the cell potential of an electrochemical cell -/
def cell_potential (E_reduction E_oxidation : ℝ) : ℝ :=
  E_reduction - E_oxidation

/-- Theorem: The cell potential of an electrochemical cell involving 
    Aluminum Fluoride and Magnesium Chloride is 0.71 V -/
theorem AlF3_MgCl2_cell_potential : 
  cell_potential E_Al (-E_Mg) = 0.71 := by
  sorry

end NUMINAMATH_CALUDE_AlF3_MgCl2_cell_potential_l1947_194746


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l1947_194762

theorem opposite_of_negative_two : 
  ∀ x : ℤ, x + (-2) = 0 → x = 2 := by
sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l1947_194762


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1947_194743

theorem inequality_system_solution : 
  {x : ℤ | x > 0 ∧ 
           (1 + 2*x : ℚ)/4 - (1 - 3*x)/10 > -1/5 ∧ 
           (3*x - 1 : ℚ) < 2*(x + 1)} = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1947_194743


namespace NUMINAMATH_CALUDE_book_selection_combination_l1947_194785

theorem book_selection_combination : ∃ n : ℕ, n * 10^9 + 306249080 = Nat.choose 20 8 := by sorry

end NUMINAMATH_CALUDE_book_selection_combination_l1947_194785


namespace NUMINAMATH_CALUDE_stacy_bought_two_packs_l1947_194754

/-- The number of sheets per pack of printer paper -/
def sheets_per_pack : ℕ := 240

/-- The number of sheets used per day -/
def sheets_per_day : ℕ := 80

/-- The number of days the paper lasts -/
def days_lasted : ℕ := 6

/-- The number of packs of printer paper Stacy bought -/
def packs_bought : ℕ := (sheets_per_day * days_lasted) / sheets_per_pack

theorem stacy_bought_two_packs : packs_bought = 2 := by
  sorry

end NUMINAMATH_CALUDE_stacy_bought_two_packs_l1947_194754


namespace NUMINAMATH_CALUDE_blue_hat_cost_l1947_194759

/-- Proves that the cost of each blue hat is $6 given the conditions of the hat purchase problem. -/
theorem blue_hat_cost (total_hats : ℕ) (green_hat_cost : ℕ) (total_price : ℕ) (green_hats : ℕ) :
  total_hats = 85 →
  green_hat_cost = 7 →
  total_price = 530 →
  green_hats = 20 →
  (total_price - green_hat_cost * green_hats) / (total_hats - green_hats) = 6 := by
sorry

end NUMINAMATH_CALUDE_blue_hat_cost_l1947_194759


namespace NUMINAMATH_CALUDE_minimal_sum_for_equal_last_digits_l1947_194722

theorem minimal_sum_for_equal_last_digits (m n : ℕ) : 
  n > m ∧ m ≥ 1 ∧ 
  (1978^m : ℕ) % 1000 = (1978^n : ℕ) % 1000 ∧
  (∀ m' n' : ℕ, n' > m' ∧ m' ≥ 1 ∧ 
    (1978^m' : ℕ) % 1000 = (1978^n' : ℕ) % 1000 → 
    m + n ≤ m' + n') →
  m = 3 ∧ n = 103 := by
sorry

end NUMINAMATH_CALUDE_minimal_sum_for_equal_last_digits_l1947_194722


namespace NUMINAMATH_CALUDE_a_plus_b_value_l1947_194797

theorem a_plus_b_value (a b : ℤ) (h1 : |a| = 6) (h2 : |b| = 4) (h3 : a * b < 0) :
  a + b = 2 ∨ a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_value_l1947_194797


namespace NUMINAMATH_CALUDE_ship_lighthouse_distance_l1947_194788

/-- The distance between a ship and a lighthouse given specific sailing conditions -/
theorem ship_lighthouse_distance 
  (speed : ℝ) 
  (time : ℝ) 
  (angle_A : ℝ) 
  (angle_B : ℝ) : 
  speed = 15 → 
  time = 4 → 
  angle_A = 60 * π / 180 → 
  angle_B = 15 * π / 180 → 
  ∃ (d : ℝ), d = 800 * Real.sqrt 3 - 240 ∧ 
    d = Real.sqrt ((speed * time * (Real.cos angle_B - Real.cos angle_A) / (Real.sin angle_A - Real.sin angle_B))^2 + 
                   (speed * time * (Real.sin angle_B * Real.cos angle_A - Real.sin angle_A * Real.cos angle_B) / (Real.sin angle_A - Real.sin angle_B))^2) := by
  sorry

end NUMINAMATH_CALUDE_ship_lighthouse_distance_l1947_194788


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1947_194715

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The common difference of an arithmetic sequence. -/
def CommonDifference (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ) (p q : ℕ) (h_arith : ArithmeticSequence a)
  (h_p : a p = 4) (h_q : a q = 2) (h_pq : p = 4 + q) :
  ∃ d : ℝ, CommonDifference a d ∧ d = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1947_194715


namespace NUMINAMATH_CALUDE_min_points_to_guarantee_highest_score_eighteen_points_achievable_smallest_points_to_guarantee_highest_score_l1947_194740

/-- Represents the possible points for a single race -/
inductive RacePoints
  | first : RacePoints
  | second : RacePoints
  | third : RacePoints
  | fourth : RacePoints

/-- Converts RacePoints to their numerical value -/
def pointValue (p : RacePoints) : Nat :=
  match p with
  | .first => 7
  | .second => 4
  | .third => 2
  | .fourth => 1

/-- Calculates the total points for a sequence of three races -/
def totalPoints (r1 r2 r3 : RacePoints) : Nat :=
  pointValue r1 + pointValue r2 + pointValue r3

/-- Theorem stating that 18 points is the minimum to guarantee the highest score -/
theorem min_points_to_guarantee_highest_score :
  ∀ (r1 r2 r3 : RacePoints),
    totalPoints r1 r2 r3 ≥ 18 →
    ∀ (s1 s2 s3 : RacePoints),
      totalPoints r1 r2 r3 > totalPoints s1 s2 s3 ∨
      (r1 = s1 ∧ r2 = s2 ∧ r3 = s3) :=
by sorry

/-- Theorem stating that 18 points is achievable -/
theorem eighteen_points_achievable :
  ∃ (r1 r2 r3 : RacePoints), totalPoints r1 r2 r3 = 18 :=
by sorry

/-- Main theorem combining the above results -/
theorem smallest_points_to_guarantee_highest_score :
  (∃ (r1 r2 r3 : RacePoints), totalPoints r1 r2 r3 = 18) ∧
  (∀ (r1 r2 r3 : RacePoints),
    totalPoints r1 r2 r3 ≥ 18 →
    ∀ (s1 s2 s3 : RacePoints),
      totalPoints r1 r2 r3 > totalPoints s1 s2 s3 ∨
      (r1 = s1 ∧ r2 = s2 ∧ r3 = s3)) ∧
  (∀ n : Nat, n < 18 →
    ∃ (s1 s2 s3 r1 r2 r3 : RacePoints),
      totalPoints s1 s2 s3 = n ∧
      totalPoints r1 r2 r3 > n) :=
by sorry

end NUMINAMATH_CALUDE_min_points_to_guarantee_highest_score_eighteen_points_achievable_smallest_points_to_guarantee_highest_score_l1947_194740


namespace NUMINAMATH_CALUDE_contrapositive_truth_l1947_194725

theorem contrapositive_truth : 
  (∀ x : ℝ, (x^2 < 1 → -1 < x ∧ x < 1)) ↔ 
  (∀ x : ℝ, (x ≤ -1 ∨ 1 ≤ x) → x^2 ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_truth_l1947_194725


namespace NUMINAMATH_CALUDE_train_speed_with_36_coaches_l1947_194771

/-- Represents the speed of a train given the number of coaches attached. -/
noncomputable def train_speed (initial_speed : ℝ) (k : ℝ) (coaches : ℝ) : ℝ :=
  initial_speed - k * Real.sqrt coaches

/-- The theorem states that given the initial conditions, 
    the speed of the train with 36 coaches is 48 kmph. -/
theorem train_speed_with_36_coaches 
  (initial_speed : ℝ) 
  (k : ℝ) 
  (speed_reduction : ∀ (c : ℝ), train_speed initial_speed k c = initial_speed - k * Real.sqrt c) 
  (h1 : initial_speed = 60) 
  (h2 : train_speed initial_speed k 36 = 48) :
  train_speed initial_speed k 36 = 48 := by
  sorry

#check train_speed_with_36_coaches

end NUMINAMATH_CALUDE_train_speed_with_36_coaches_l1947_194771


namespace NUMINAMATH_CALUDE_uncle_jerry_tomatoes_l1947_194776

theorem uncle_jerry_tomatoes (day1 day2 total : ℕ) 
  (h1 : day2 = day1 + 50)
  (h2 : day1 + day2 = total)
  (h3 : total = 290) : 
  day1 = 120 := by
sorry

end NUMINAMATH_CALUDE_uncle_jerry_tomatoes_l1947_194776


namespace NUMINAMATH_CALUDE_test_point_value_l1947_194709

theorem test_point_value
  (total_points : ℕ)
  (total_questions : ℕ)
  (two_point_questions : ℕ)
  (other_type_questions : ℕ)
  (h1 : total_points = 100)
  (h2 : total_questions = 40)
  (h3 : other_type_questions = 10)
  (h4 : two_point_questions + other_type_questions = total_questions)
  (h5 : 2 * two_point_questions + other_type_questions * (total_points - 2 * two_point_questions) / other_type_questions = total_points) :
  (total_points - 2 * two_point_questions) / other_type_questions = 4 :=
by sorry

end NUMINAMATH_CALUDE_test_point_value_l1947_194709


namespace NUMINAMATH_CALUDE_cosine_sine_sum_equals_half_l1947_194752

theorem cosine_sine_sum_equals_half : 
  Real.cos (36 * π / 180) * Real.cos (96 * π / 180) + 
  Real.sin (36 * π / 180) * Real.sin (84 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_sum_equals_half_l1947_194752


namespace NUMINAMATH_CALUDE_commute_days_calculation_l1947_194707

theorem commute_days_calculation (morning_bus afternoon_bus train_commute : ℕ) 
  (h1 : morning_bus = 8)
  (h2 : afternoon_bus = 15)
  (h3 : train_commute = 9) :
  ∃ (morning_train afternoon_train both_bus : ℕ),
    morning_train + afternoon_train = train_commute ∧
    morning_bus = afternoon_train + both_bus ∧
    afternoon_bus = morning_train + both_bus ∧
    morning_train + afternoon_train + both_bus = 16 :=
by sorry

end NUMINAMATH_CALUDE_commute_days_calculation_l1947_194707


namespace NUMINAMATH_CALUDE_probability_factor_less_than_eight_l1947_194772

def factors (n : ℕ) : Finset ℕ :=
  Finset.filter (· ∣ n) (Finset.range (n + 1))

theorem probability_factor_less_than_eight :
  let f := factors 120
  (f.filter (· < 8)).card / f.card = 3 / 8 := by sorry

end NUMINAMATH_CALUDE_probability_factor_less_than_eight_l1947_194772


namespace NUMINAMATH_CALUDE_old_clock_slow_l1947_194727

/-- Represents the number of minutes between hand overlaps on the old clock -/
def overlap_interval : ℕ := 66

/-- Represents the number of minutes in a standard day -/
def standard_day_minutes : ℕ := 24 * 60

/-- Represents the number of hand overlaps in a standard day -/
def overlaps_per_day : ℕ := 22

theorem old_clock_slow (old_clock_day : ℕ) 
  (h1 : old_clock_day = overlap_interval * overlaps_per_day) : 
  old_clock_day - standard_day_minutes = 12 := by
  sorry

end NUMINAMATH_CALUDE_old_clock_slow_l1947_194727


namespace NUMINAMATH_CALUDE_frog_jump_probability_l1947_194708

/-- Represents the probability of ending on a horizontal side -/
def probability_horizontal_end (x y : ℝ) : ℝ := sorry

/-- The rectangle's dimensions -/
def rectangle_width : ℝ := 5
def rectangle_height : ℝ := 5

/-- The frog's starting position -/
def start_x : ℝ := 2
def start_y : ℝ := 3

/-- Theorem stating the probability of ending on a horizontal side -/
theorem frog_jump_probability :
  probability_horizontal_end start_x start_y = 13 / 14 := by sorry

end NUMINAMATH_CALUDE_frog_jump_probability_l1947_194708


namespace NUMINAMATH_CALUDE_banana_bread_pieces_l1947_194750

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents a pan of banana bread -/
structure BananaBreadPan where
  dimensions : Dimensions

/-- Represents a piece of banana bread -/
structure BananaBreadPiece where
  dimensions : Dimensions

/-- Calculates the number of pieces that can be cut from a pan -/
def num_pieces (pan : BananaBreadPan) (piece : BananaBreadPiece) : ℕ :=
  (area pan.dimensions) / (area piece.dimensions)

theorem banana_bread_pieces : 
  let pan := BananaBreadPan.mk (Dimensions.mk 24 20)
  let piece := BananaBreadPiece.mk (Dimensions.mk 3 4)
  num_pieces pan piece = 40 := by
  sorry

end NUMINAMATH_CALUDE_banana_bread_pieces_l1947_194750


namespace NUMINAMATH_CALUDE_desired_interest_percentage_l1947_194745

/-- Calculates the desired interest percentage for a share investment. -/
theorem desired_interest_percentage
  (face_value : ℝ)
  (dividend_rate : ℝ)
  (market_value : ℝ)
  (h1 : face_value = 20)
  (h2 : dividend_rate = 0.09)
  (h3 : market_value = 15) :
  (dividend_rate * face_value) / market_value = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_desired_interest_percentage_l1947_194745


namespace NUMINAMATH_CALUDE_sun_rise_position_l1947_194741

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the visibility of a circle above a line -/
inductive Visibility
  | Small
  | Half
  | Full

/-- Determines the positional relationship between a line and a circle -/
inductive PositionalRelationship
  | Tangent
  | Separate
  | ExternallyTangent
  | Intersecting

/-- 
  Given a circle and a line where only a small portion of the circle is visible above the line,
  prove that the positional relationship between the line and circle is intersecting.
-/
theorem sun_rise_position (c : Circle) (l : Line) (v : Visibility) :
  v = Visibility.Small → PositionalRelationship.Intersecting = 
    (let relationship := sorry -- Define the actual relationship based on c and l
     relationship) := by
  sorry


end NUMINAMATH_CALUDE_sun_rise_position_l1947_194741


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l1947_194766

/-- Given three lines that intersect at the same point, prove the value of k -/
theorem intersection_of_three_lines (x y : ℝ) :
  (y = 4 * x + 3) ∧ 
  (y = -2 * x - 25) ∧ 
  (y = 3 * x + k) →
  k = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l1947_194766


namespace NUMINAMATH_CALUDE_exists_fifteen_classmates_l1947_194735

/-- A type representing students. -/
def Student : Type := ℕ

/-- The total number of students. -/
def total_students : ℕ := 60

/-- A function that returns true if the given students are classmates. -/
def are_classmates : List Student → Prop := sorry

/-- The property that among any 10 students, there are always 3 classmates. -/
axiom three_classmates_in_ten : 
  ∀ (s : Finset Student), s.card = 10 → ∃ (t : Finset Student), t ⊆ s ∧ t.card = 3 ∧ are_classmates t.toList

/-- The theorem to be proved. -/
theorem exists_fifteen_classmates :
  ∃ (s : Finset Student), s.card ≥ 15 ∧ are_classmates s.toList :=
sorry

end NUMINAMATH_CALUDE_exists_fifteen_classmates_l1947_194735


namespace NUMINAMATH_CALUDE_grade_assignment_count_l1947_194716

/-- The number of different grades that can be assigned to each student. -/
def numGrades : ℕ := 4

/-- The number of students in the class. -/
def numStudents : ℕ := 15

/-- The theorem stating the number of ways to assign grades to students. -/
theorem grade_assignment_count : numGrades ^ numStudents = 1073741824 := by
  sorry

end NUMINAMATH_CALUDE_grade_assignment_count_l1947_194716


namespace NUMINAMATH_CALUDE_interest_years_calculation_l1947_194705

/-- Given simple interest, compound interest, and interest rate, calculate the number of years -/
theorem interest_years_calculation (simple_interest compound_interest : ℝ) (rate : ℝ) 
  (h1 : simple_interest = 600)
  (h2 : compound_interest = 609)
  (h3 : rate = 0.03)
  (h4 : simple_interest = rate * (compound_interest / (rate * ((1 + rate)^2 - 1))))
  (h5 : compound_interest = (simple_interest / (rate * 2)) * ((1 + rate)^2 - 1)) :
  ∃ (n : ℕ), n = 2 ∧ 
    simple_interest = (compound_interest / ((1 + rate)^n - 1)) * rate * n ∧
    compound_interest = (simple_interest / (rate * n)) * ((1 + rate)^n - 1) :=
sorry

end NUMINAMATH_CALUDE_interest_years_calculation_l1947_194705


namespace NUMINAMATH_CALUDE_unique_seating_arrangement_l1947_194791

/-- Represents a seating arrangement --/
structure SeatingArrangement where
  rows_with_8 : ℕ
  rows_with_7 : ℕ

/-- Checks if a seating arrangement is valid --/
def is_valid (s : SeatingArrangement) : Prop :=
  s.rows_with_8 * 8 + s.rows_with_7 * 7 = 55

/-- Theorem stating the unique valid seating arrangement --/
theorem unique_seating_arrangement :
  ∃! s : SeatingArrangement, is_valid s ∧ s.rows_with_8 = 6 := by sorry

end NUMINAMATH_CALUDE_unique_seating_arrangement_l1947_194791


namespace NUMINAMATH_CALUDE_x_squared_plus_4x_plus_5_range_l1947_194710

theorem x_squared_plus_4x_plus_5_range :
  ∀ x : ℝ, x^2 - 7*x + 12 < 0 →
  ∃ y ∈ Set.Ioo 26 37, y = x^2 + 4*x + 5 ∧
  ∀ z, z = x^2 + 4*x + 5 → z ∈ Set.Ioo 26 37 :=
by sorry

end NUMINAMATH_CALUDE_x_squared_plus_4x_plus_5_range_l1947_194710


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1947_194751

theorem absolute_value_inequality (x : ℝ) : 
  |((7 - x) / 4)| < 3 ↔ 2 < x ∧ x < 19 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1947_194751


namespace NUMINAMATH_CALUDE_solution_set_abs_b_greater_than_two_l1947_194757

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2|

-- Part 1: Solution set of the inequality
theorem solution_set (x : ℝ) : f x + f (x + 1) ≥ 5 ↔ x ≥ 4 ∨ x ≤ -1 := by
  sorry

-- Part 2: Proof that |b| > 2
theorem abs_b_greater_than_two (a b : ℝ) (h1 : |a| > 1) (h2 : f (a * b) > |a| * f (b / a)) : |b| > 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_abs_b_greater_than_two_l1947_194757


namespace NUMINAMATH_CALUDE_tax_reduction_theorem_l1947_194700

/-- Proves that if a tax rate is reduced by X%, consumption increases by 12%,
    and the resulting revenue decreases by 14.88%, then X = 24. -/
theorem tax_reduction_theorem (X : ℝ) (T C : ℝ) (hT : T > 0) (hC : C > 0) :
  let new_tax_rate := T - (X / 100) * T
  let new_consumption := C + (12 / 100) * C
  let original_revenue := T * C
  let new_revenue := new_tax_rate * new_consumption
  new_revenue = (1 - 14.88 / 100) * original_revenue →
  X = 24 := by
  sorry

end NUMINAMATH_CALUDE_tax_reduction_theorem_l1947_194700


namespace NUMINAMATH_CALUDE_cube_root_of_a_plus_b_l1947_194793

theorem cube_root_of_a_plus_b (a b : ℝ) (ha : a > 0) 
  (h1 : (2*b - 1)^2 = a) (h2 : (b + 4)^2 = a) (h3 : (2*b - 1) + (b + 4) = 0) : 
  (a + b)^(1/3 : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_a_plus_b_l1947_194793


namespace NUMINAMATH_CALUDE_max_rooks_per_color_exists_sixteen_rooks_config_max_rooks_is_sixteen_l1947_194758

/-- Represents a chessboard configuration with white and black rooks -/
structure ChessboardConfig where
  board_size : Nat
  white_rooks : Nat
  black_rooks : Nat
  non_threatening : Bool

/-- Defines a valid chessboard configuration -/
def is_valid_config (c : ChessboardConfig) : Prop :=
  c.board_size = 8 ∧ 
  c.white_rooks = c.black_rooks ∧ 
  c.non_threatening = true

/-- Theorem stating the maximum number of rooks for each color -/
theorem max_rooks_per_color (c : ChessboardConfig) : 
  is_valid_config c → c.white_rooks ≤ 16 := by
  sorry

/-- Theorem proving the existence of a configuration with 16 rooks per color -/
theorem exists_sixteen_rooks_config : 
  ∃ c : ChessboardConfig, is_valid_config c ∧ c.white_rooks = 16 := by
  sorry

/-- Main theorem proving 16 is the maximum number of rooks per color -/
theorem max_rooks_is_sixteen : 
  ∀ c : ChessboardConfig, is_valid_config c → 
    c.white_rooks ≤ 16 ∧ (∃ c' : ChessboardConfig, is_valid_config c' ∧ c'.white_rooks = 16) := by
  sorry

end NUMINAMATH_CALUDE_max_rooks_per_color_exists_sixteen_rooks_config_max_rooks_is_sixteen_l1947_194758


namespace NUMINAMATH_CALUDE_total_amount_after_four_years_l1947_194753

/-- Jo's annual earnings in USD -/
def annual_earnings : ℕ := 3^5 - 3^4 + 3^3 - 3^2 + 3

/-- Annual investment return in USD -/
def investment_return : ℕ := 2^5 - 2^4 + 2^3 - 2^2 + 2

/-- Number of years -/
def years : ℕ := 4

/-- Theorem stating the total amount after four years -/
theorem total_amount_after_four_years : 
  (annual_earnings + investment_return) * years = 820 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_after_four_years_l1947_194753


namespace NUMINAMATH_CALUDE_louisa_travel_speed_l1947_194781

/-- Louisa's vacation travel problem -/
theorem louisa_travel_speed :
  ∀ (v : ℝ),
  v > 0 →
  200 / v + 3 = 350 / v →
  v = 50 :=
by
  sorry

#check louisa_travel_speed

end NUMINAMATH_CALUDE_louisa_travel_speed_l1947_194781


namespace NUMINAMATH_CALUDE_ratio_BL_LC_l1947_194733

/-- A square with side length 5 -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (is_square : A = (0, 0) ∧ B = (5, 0) ∧ C = (5, 5) ∧ D = (0, 5))

/-- A point K on side AB of the square -/
def K : ℝ × ℝ := (3, 0)

/-- A point L on side BC of the square -/
def L (y : ℝ) : ℝ × ℝ := (5, y)

/-- Distance function between a point and a line -/
def distance_to_line (p : ℝ × ℝ) (l : ℝ × ℝ → Prop) : ℝ := sorry

/-- The theorem to be proved -/
theorem ratio_BL_LC (ABCD : Square) :
  ∃ y : ℝ, 0 ≤ y ∧ y ≤ 5 ∧
  distance_to_line K (fun p => p.2 = (y - 5) / 5 * p.1 + 5) = 3 →
  (5 - y) / y = 8 / 7 := by sorry

end NUMINAMATH_CALUDE_ratio_BL_LC_l1947_194733


namespace NUMINAMATH_CALUDE_arc_length_from_central_angle_l1947_194760

theorem arc_length_from_central_angle (D : Real) (EF : Real) (DEF : Real) : 
  D = 80 → DEF = 45 → EF = 10 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_from_central_angle_l1947_194760


namespace NUMINAMATH_CALUDE_negation_equivalence_l1947_194780

open Real

theorem negation_equivalence :
  (¬ ∃ x₀ ∈ Set.Ioo (0 : ℝ) (π / 2), Real.log x₀ + Real.tan x₀ < 0) ↔
  (∀ x ∈ Set.Ioo (0 : ℝ) (π / 2), Real.log x + Real.tan x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1947_194780


namespace NUMINAMATH_CALUDE_mary_chestnut_pick_l1947_194755

/-- Given three people picking chestnuts with specific relationships between their picks,
    prove that one person picked a certain amount. -/
theorem mary_chestnut_pick (peter lucy mary : ℝ) 
  (h1 : mary = 2 * peter)
  (h2 : lucy = peter + 2)
  (h3 : peter + mary + lucy = 26) :
  mary = 12 := by
  sorry

end NUMINAMATH_CALUDE_mary_chestnut_pick_l1947_194755


namespace NUMINAMATH_CALUDE_grocer_sales_problem_l1947_194773

/-- Calculates the first month's sale given sales for the next 4 months and desired average -/
def first_month_sale (month2 month3 month4 month5 desired_average : ℕ) : ℕ :=
  5 * desired_average - (month2 + month3 + month4 + month5)

/-- Proves that the first month's sale is 6790 given the problem conditions -/
theorem grocer_sales_problem : 
  first_month_sale 5660 6200 6350 6500 6300 = 6790 := by
  sorry

#eval first_month_sale 5660 6200 6350 6500 6300

end NUMINAMATH_CALUDE_grocer_sales_problem_l1947_194773


namespace NUMINAMATH_CALUDE_min_value_of_c_l1947_194777

theorem min_value_of_c (a b c d e : ℕ) : 
  a > 0 → 
  b = a + 1 → 
  c = b + 1 → 
  d = c + 1 → 
  e = d + 1 → 
  ∃ n : ℕ, a + b + c + d + e = n^3 → 
  ∃ m : ℕ, b + c + d = m^2 → 
  ∀ c' : ℕ, (∃ a' b' d' e' : ℕ, 
    a' > 0 ∧ 
    b' = a' + 1 ∧ 
    d' = c' + 1 ∧ 
    e' = d' + 1 ∧ 
    (∃ n' : ℕ, a' + b' + c' + d' + e' = n'^3) ∧ 
    (∃ m' : ℕ, b' + c' + d' = m'^2)) → 
  c' ≥ c → 
  c = 675 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_c_l1947_194777


namespace NUMINAMATH_CALUDE_simplify_polynomial_expression_l1947_194721

theorem simplify_polynomial_expression (x : ℝ) :
  6 * x^2 + 4 * x + 9 - (7 - 5 * x - 9 * x^3 + 8 * x^2) = 9 * x^3 - 2 * x^2 + 9 * x + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_expression_l1947_194721


namespace NUMINAMATH_CALUDE_orchard_area_distribution_l1947_194789

/-- Represents an orange orchard with flat and hilly land. -/
structure Orchard where
  total_area : ℝ
  flat_area : ℝ
  hilly_area : ℝ
  sampled_flat : ℝ
  sampled_hilly : ℝ

/-- Checks if the orchard satisfies the given conditions. -/
def is_valid_orchard (o : Orchard) : Prop :=
  o.total_area = 120 ∧
  o.flat_area + o.hilly_area = o.total_area ∧
  o.sampled_flat + o.sampled_hilly = 10 ∧
  o.sampled_hilly = 2 * o.sampled_flat + 1

/-- Theorem stating the correct distribution of flat and hilly land in the orchard. -/
theorem orchard_area_distribution (o : Orchard) (h : is_valid_orchard o) :
  o.flat_area = 36 ∧ o.hilly_area = 84 := by
  sorry

end NUMINAMATH_CALUDE_orchard_area_distribution_l1947_194789


namespace NUMINAMATH_CALUDE_ellipse_properties_l1947_194770

open Real

theorem ellipse_properties (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  let e := (Real.sqrt 6) / 3
  let d := (Real.sqrt 3) / 2
  let ellipse := fun (x y : ℝ) => x^2 / a^2 + y^2 / b^2 = 1
  let line := fun (k m x : ℝ) => k * x + m
  let A := (0, -b)
  let B := (a, 0)
  let distance_to_AB := d

  (e^2 * a^2 = a^2 - b^2) →
  (distance_to_AB^2 * (a^2 + b^2) = a^2 * b^2) →
  (∃ (C D : ℝ × ℝ) (k m : ℝ), k ≠ 0 ∧ m ≠ 0 ∧
    ellipse C.1 C.2 ∧ ellipse D.1 D.2 ∧
    C.2 = line k m C.1 ∧ D.2 = line k m D.1 ∧
    (C.1 - A.1)^2 + (C.2 - A.2)^2 = (D.1 - A.1)^2 + (D.2 - A.2)^2) →
  (a^2 = 3 ∧ b^2 = 1 ∧
   (let k := (Real.sqrt 6) / 3
    let m := 3 / 2
    let area_ACD := 5 / 4
    ∃ (C D : ℝ × ℝ),
      ellipse C.1 C.2 ∧ ellipse D.1 D.2 ∧
      C.2 = line k m C.1 ∧ D.2 = line k m D.1 ∧
      (C.1 - A.1)^2 + (C.2 - A.2)^2 = (D.1 - A.1)^2 + (D.2 - A.2)^2 ∧
      area_ACD = 1/2 * abs ((C.1 - A.1) * (D.2 - A.2) - (C.2 - A.2) * (D.1 - A.1))))
  := by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1947_194770


namespace NUMINAMATH_CALUDE_sharp_nested_30_l1947_194706

-- Define the # operation
def sharp (N : ℝ) : ℝ := 0.6 * N + 2

-- Theorem statement
theorem sharp_nested_30 : sharp (sharp (sharp (sharp 30))) = 8.24 := by sorry

end NUMINAMATH_CALUDE_sharp_nested_30_l1947_194706


namespace NUMINAMATH_CALUDE_quadratic_equation_sum_l1947_194723

theorem quadratic_equation_sum (x r s : ℝ) : 
  (15 * x^2 + 30 * x - 450 = 0) →
  ((x + r)^2 = s) →
  (r + s = 32) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_sum_l1947_194723


namespace NUMINAMATH_CALUDE_distribute_four_to_three_l1947_194742

/-- The number of ways to distribute n students to k villages, where each village gets at least one student -/
def distribute_students (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem stating that distributing 4 students to 3 villages results in 36 different plans -/
theorem distribute_four_to_three : distribute_students 4 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_distribute_four_to_three_l1947_194742


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l1947_194738

theorem matrix_equation_solution : ∃ (N : Matrix (Fin 2) (Fin 2) ℝ),
  N^3 - 3 * N^2 + 4 * N = !![8, 16; 4, 8] :=
by
  -- Define the matrix N
  let N : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, 2]
  
  -- Assert that N satisfies the equation
  have h : N^3 - 3 * N^2 + 4 * N = !![8, 16; 4, 8] := by sorry
  
  -- Prove existence
  exact ⟨N, h⟩

#check matrix_equation_solution

end NUMINAMATH_CALUDE_matrix_equation_solution_l1947_194738


namespace NUMINAMATH_CALUDE_opposite_of_reciprocal_of_negative_five_l1947_194765

theorem opposite_of_reciprocal_of_negative_five :
  -(1 / -5) = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_opposite_of_reciprocal_of_negative_five_l1947_194765


namespace NUMINAMATH_CALUDE_upstream_distance_is_48_l1947_194712

/-- Represents the problem of calculating the upstream distance rowed --/
def UpstreamRowingProblem (downstream_distance : ℝ) (time : ℝ) (stream_speed : ℝ) : Prop :=
  ∃ (upstream_distance : ℝ) (boat_speed : ℝ),
    downstream_distance = (boat_speed + stream_speed) * time ∧
    upstream_distance = (boat_speed - stream_speed) * time ∧
    upstream_distance = 48

/-- Theorem stating that given the problem conditions, the upstream distance is 48 km --/
theorem upstream_distance_is_48 :
  UpstreamRowingProblem 84 2 9 :=
sorry

end NUMINAMATH_CALUDE_upstream_distance_is_48_l1947_194712


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1947_194799

theorem quadratic_inequality_solution_set (a : ℝ) (h : a > 0) :
  let S := {x : ℝ | a * x^2 - (a + 1) * x + 1 < 0}
  (0 < a ∧ a < 1 → S = {x : ℝ | 1 < x ∧ x < 1/a}) ∧
  (a = 1 → S = ∅) ∧
  (a > 1 → S = {x : ℝ | 1/a < x ∧ x < 1}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1947_194799


namespace NUMINAMATH_CALUDE_total_stickers_count_l1947_194749

/-- The number of stickers on each page -/
def stickers_per_page : ℕ := 10

/-- The number of pages -/
def number_of_pages : ℕ := 22

/-- The total number of stickers -/
def total_stickers : ℕ := stickers_per_page * number_of_pages

theorem total_stickers_count : total_stickers = 220 := by
  sorry

end NUMINAMATH_CALUDE_total_stickers_count_l1947_194749


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l1947_194748

theorem quadratic_roots_product (a b : ℝ) : 
  (a^2 + 2012*a + 1 = 0) → 
  (b^2 + 2012*b + 1 = 0) → 
  (2 + 2013*a + a^2) * (2 + 2013*b + b^2) = -2010 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l1947_194748


namespace NUMINAMATH_CALUDE_initial_number_of_persons_l1947_194718

theorem initial_number_of_persons (n : ℕ) 
  (h1 : 4 * n = 48) : n = 12 := by
  sorry

#check initial_number_of_persons

end NUMINAMATH_CALUDE_initial_number_of_persons_l1947_194718


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l1947_194720

theorem cube_root_equation_solution :
  ∃! x : ℝ, (5 - x / 3) ^ (1/3 : ℝ) = -4 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l1947_194720


namespace NUMINAMATH_CALUDE_same_color_probability_l1947_194714

/-- The probability of drawing two balls of the same color from a box with 2 red balls and 3 white balls,
    when drawing with replacement. -/
theorem same_color_probability (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) 
  (h1 : total_balls = red_balls + white_balls)
  (h2 : red_balls = 2)
  (h3 : white_balls = 3) :
  (red_balls : ℚ) / total_balls * (red_balls : ℚ) / total_balls + 
  (white_balls : ℚ) / total_balls * (white_balls : ℚ) / total_balls = 13 / 25 :=
sorry

end NUMINAMATH_CALUDE_same_color_probability_l1947_194714


namespace NUMINAMATH_CALUDE_product_of_largest_primes_l1947_194763

def largest_one_digit_primes : List Nat := [7, 5]
def largest_two_digit_prime : Nat := 97
def largest_three_digit_prime : Nat := 997

theorem product_of_largest_primes : 
  (List.prod largest_one_digit_primes) * largest_two_digit_prime * largest_three_digit_prime = 3383815 := by
  sorry

end NUMINAMATH_CALUDE_product_of_largest_primes_l1947_194763


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1947_194795

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) :
  let z := (1 + i^3) / (2 - i)
  Complex.im z = -1/5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1947_194795


namespace NUMINAMATH_CALUDE_ratio_bounds_l1947_194756

theorem ratio_bounds (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : 5 - 3*a ≤ b) (h2 : b ≤ 4 - a) (h3 : Real.log b ≥ a) :
  e ≤ b/a ∧ b/a ≤ 7 := by
sorry

end NUMINAMATH_CALUDE_ratio_bounds_l1947_194756


namespace NUMINAMATH_CALUDE_marks_trip_length_l1947_194796

theorem marks_trip_length (total : ℚ) 
  (h1 : total / 4 + 30 + total / 6 = total) : 
  total = 360 / 7 := by
sorry

end NUMINAMATH_CALUDE_marks_trip_length_l1947_194796


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1947_194779

def U : Set Nat := {1,2,3,4,5,6,7}
def A : Set Nat := {1,3,5,7}
def B : Set Nat := {1,3,5,6,7}

theorem complement_intersection_theorem :
  (U \ (A ∩ B)) = {2,4,6} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1947_194779


namespace NUMINAMATH_CALUDE_smallest_staircase_steps_l1947_194728

theorem smallest_staircase_steps : ∃ (n : ℕ), 
  n > 20 ∧ 
  n % 6 = 5 ∧ 
  n % 7 = 1 ∧ 
  (∀ m : ℕ, m > 20 ∧ m % 6 = 5 ∧ m % 7 = 1 → m ≥ n) ∧
  n = 29 := by
sorry

end NUMINAMATH_CALUDE_smallest_staircase_steps_l1947_194728


namespace NUMINAMATH_CALUDE_polynomial_irreducibility_l1947_194778

theorem polynomial_irreducibility (n : ℕ) (hn : n > 1) :
  ¬∃ (g h : Polynomial ℤ), (Polynomial.degree g ≥ 1) ∧ (Polynomial.degree h ≥ 1) ∧
  (x^n + 5 * x^(n-1) + 3 : Polynomial ℤ) = g * h :=
by sorry

end NUMINAMATH_CALUDE_polynomial_irreducibility_l1947_194778


namespace NUMINAMATH_CALUDE_set_equality_l1947_194704

-- Define the universal set U as ℝ
def U := ℝ

-- Define set M
def M : Set ℝ := {x | x < 1}

-- Define set N
def N : Set ℝ := {x | -1 < x ∧ x < 2}

-- Theorem statement
theorem set_equality : {x : ℝ | x ≥ 2} = (M ∪ N)ᶜ := by sorry

end NUMINAMATH_CALUDE_set_equality_l1947_194704


namespace NUMINAMATH_CALUDE_ellipse_cartesian_eq_l1947_194732

def ellipse_eq (x y : ℝ) : Prop :=
  ∃ t : ℝ, x = (3 * (Real.sin t - 2)) / (3 - Real.cos t) ∧
            y = (4 * (Real.cos t - 6)) / (3 - Real.cos t)

theorem ellipse_cartesian_eq :
  ∀ x y : ℝ, ellipse_eq x y ↔ 9*x^2 + 36*x*y + 9*y^2 + 216*x + 432*y + 1440 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_cartesian_eq_l1947_194732


namespace NUMINAMATH_CALUDE_certain_number_proof_l1947_194794

theorem certain_number_proof : ∃ x : ℝ, (7.5 * 7.5) + x + (2.5 * 2.5) = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1947_194794


namespace NUMINAMATH_CALUDE_square_perimeter_l1947_194719

/-- Given a square with side length 15 cm, prove that its perimeter is 60 cm. -/
theorem square_perimeter (side_length : ℝ) (h : side_length = 15) : 
  4 * side_length = 60 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l1947_194719


namespace NUMINAMATH_CALUDE_circle_equation_l1947_194775

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def is_in_first_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0

def is_tangent_to_line (C : Circle) (a b c : ℝ) : Prop :=
  let (x, y) := C.center
  abs (a * x + b * y + c) / Real.sqrt (a^2 + b^2) = C.radius

def is_tangent_to_x_axis (C : Circle) : Prop :=
  C.center.2 = C.radius

-- State the theorem
theorem circle_equation (C : Circle) :
  C.radius = 1 →
  is_in_first_quadrant C.center →
  is_tangent_to_line C 4 (-3) 0 →
  is_tangent_to_x_axis C →
  ∀ x y : ℝ, (x - 2)^2 + (y - 1)^2 = 1 ↔ (x, y) ∈ {p | (p.1 - C.center.1)^2 + (p.2 - C.center.2)^2 = C.radius^2} :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l1947_194775


namespace NUMINAMATH_CALUDE_central_angle_twice_inscribed_l1947_194736

/-- A circle with a diameter and a point on its circumference -/
structure CircleWithDiameterAndPoint where
  /-- The center of the circle -/
  O : ℝ × ℝ
  /-- One end of the diameter -/
  A : ℝ × ℝ
  /-- The other end of the diameter -/
  B : ℝ × ℝ
  /-- An arbitrary point on the circle -/
  C : ℝ × ℝ
  /-- AB is a diameter -/
  diameter : dist O A = dist O B
  /-- C is on the circle -/
  on_circle : dist O C = dist O A

/-- The angle between two vectors -/
def angle (v w : ℝ × ℝ) : ℝ := sorry

/-- The theorem: Central angle COB is twice the inscribed angle CAB -/
theorem central_angle_twice_inscribed 
  (circle : CircleWithDiameterAndPoint) : 
  angle (circle.C - circle.O) (circle.B - circle.O) = 
  2 * angle (circle.C - circle.A) (circle.B - circle.A) := by
  sorry

end NUMINAMATH_CALUDE_central_angle_twice_inscribed_l1947_194736


namespace NUMINAMATH_CALUDE_optimal_sampling_methods_l1947_194787

/-- Represents different income levels of families -/
inductive IncomeLevel
  | High
  | Middle
  | Low

/-- Represents different sampling methods -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

structure Community where
  totalFamilies : Nat
  highIncomeFamilies : Nat
  middleIncomeFamilies : Nat
  lowIncomeFamilies : Nat
  sampleSize : Nat

structure School where
  femaleAthletes : Nat
  selectionSize : Nat

def optimalSamplingMethod (c : Community) (s : School) : 
  (SamplingMethod × SamplingMethod) :=
  sorry

theorem optimal_sampling_methods 
  (c : Community) 
  (s : School) 
  (h1 : c.totalFamilies = 500)
  (h2 : c.highIncomeFamilies = 125)
  (h3 : c.middleIncomeFamilies = 280)
  (h4 : c.lowIncomeFamilies = 95)
  (h5 : c.sampleSize = 100)
  (h6 : s.femaleAthletes = 12)
  (h7 : s.selectionSize = 3) :
  optimalSamplingMethod c s = (SamplingMethod.Stratified, SamplingMethod.SimpleRandom) :=
  sorry

end NUMINAMATH_CALUDE_optimal_sampling_methods_l1947_194787


namespace NUMINAMATH_CALUDE_infinitely_many_non_representable_l1947_194724

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- The statement that there exist infinitely many positive integers which cannot be written as a^(d(a)) + b^(d(b)) -/
theorem infinitely_many_non_representable : 
  ∃ (S : Set ℕ+), Set.Infinite S ∧ 
    ∀ (n : ℕ+), n ∈ S → 
      ∀ (a b : ℕ+), n ≠ a ^ (num_divisors a) + b ^ (num_divisors b) := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_non_representable_l1947_194724


namespace NUMINAMATH_CALUDE_expression_simplification_l1947_194703

theorem expression_simplification :
  1 / ((1 / (Real.sqrt 2 + 1)) + (2 / (Real.sqrt 3 - 1))) = Real.sqrt 3 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1947_194703


namespace NUMINAMATH_CALUDE_acute_triangle_inequality_l1947_194731

-- Define an acute-angled triangle
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  r : ℝ
  R : ℝ
  acute : 0 < a ∧ 0 < b ∧ 0 < c
  inradius_positive : 0 < r
  circumradius_positive : 0 < R
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  acute_angles : a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

-- State the theorem
theorem acute_triangle_inequality (t : AcuteTriangle) :
  t.a^2 + t.b^2 + t.c^2 ≥ 4 * (t.R + t.r)^2 := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_inequality_l1947_194731


namespace NUMINAMATH_CALUDE_power_of_one_third_l1947_194734

theorem power_of_one_third (a b : ℕ) : 
  (2^a = 8 ∧ 5^b = 25) → (1/3 : ℚ)^(b - a) = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_one_third_l1947_194734


namespace NUMINAMATH_CALUDE_total_leaked_equals_1958_l1947_194730

/-- Represents the data for an oil pipe leak -/
structure PipeLeak where
  name : String
  leakRate : ℕ  -- gallons per hour
  fixTime : ℕ  -- hours

/-- Calculates the total amount of oil leaked from a pipe during repair -/
def totalLeakedDuringRepair (pipe : PipeLeak) : ℕ :=
  pipe.leakRate * pipe.fixTime

/-- The set of all pipe leaks -/
def pipeLeaks : List PipeLeak := [
  { name := "A", leakRate := 25, fixTime := 10 },
  { name := "B", leakRate := 37, fixTime := 7 },
  { name := "C", leakRate := 55, fixTime := 12 },
  { name := "D", leakRate := 41, fixTime := 9 },
  { name := "E", leakRate := 30, fixTime := 14 }
]

/-- Calculates the total amount of oil leaked from all pipes during repair -/
def totalLeaked : ℕ :=
  (pipeLeaks.map totalLeakedDuringRepair).sum

theorem total_leaked_equals_1958 : totalLeaked = 1958 := by
  sorry

#eval totalLeaked  -- This will print the result

end NUMINAMATH_CALUDE_total_leaked_equals_1958_l1947_194730


namespace NUMINAMATH_CALUDE_claire_photos_l1947_194768

/-- Given that:
    - Lisa and Robert have taken the same number of photos
    - Lisa has taken 3 times as many photos as Claire
    - Robert has taken 28 more photos than Claire
    Prove that Claire has taken 14 photos. -/
theorem claire_photos (claire lisa robert : ℕ) 
  (h1 : lisa = robert)
  (h2 : lisa = 3 * claire)
  (h3 : robert = claire + 28) :
  claire = 14 := by
  sorry

end NUMINAMATH_CALUDE_claire_photos_l1947_194768


namespace NUMINAMATH_CALUDE_equation_solution_l1947_194737

theorem equation_solution : ∃ x : ℝ, (27 - 5 = 4 + x) ∧ (x = 18) := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1947_194737


namespace NUMINAMATH_CALUDE_pentagon_y_coordinate_l1947_194717

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Pentagon with vertices A, B, C, D, E -/
structure Pentagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point

/-- Check if a pentagon has a vertical line of symmetry -/
def hasVerticalSymmetry (p : Pentagon) : Prop :=
  p.A.x = 0 ∧ p.B.x = 0 ∧ p.D.x = p.E.x ∧ p.C.x = (p.D.x / 2)

/-- Calculate the area of a pentagon -/
noncomputable def pentagonArea (p : Pentagon) : ℝ :=
  sorry -- Actual implementation would go here

theorem pentagon_y_coordinate (p : Pentagon) :
  p.A = ⟨0, 0⟩ →
  p.B = ⟨0, 5⟩ →
  p.D = ⟨6, 5⟩ →
  p.E = ⟨6, 0⟩ →
  p.C.x = 3 →
  hasVerticalSymmetry p →
  pentagonArea p = 50 →
  p.C.y = 35/3 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_y_coordinate_l1947_194717


namespace NUMINAMATH_CALUDE_vacuum_cost_proof_l1947_194767

/-- The cost of the vacuum cleaner Daria is saving for -/
def vacuum_cost : ℕ := 120

/-- The initial amount Daria has collected -/
def initial_amount : ℕ := 20

/-- The amount Daria adds to her savings each week -/
def weekly_savings : ℕ := 10

/-- The number of weeks Daria needs to save -/
def weeks_to_save : ℕ := 10

/-- Theorem stating that the vacuum cost is correct given the initial amount,
    weekly savings, and number of weeks to save -/
theorem vacuum_cost_proof :
  vacuum_cost = initial_amount + weekly_savings * weeks_to_save := by
  sorry

end NUMINAMATH_CALUDE_vacuum_cost_proof_l1947_194767


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_fraction_simplification_l1947_194702

-- Problem 1
theorem sqrt_expression_equality : 
  Real.sqrt 12 + Real.sqrt 3 * (Real.sqrt 2 - 1) = Real.sqrt 3 + Real.sqrt 6 := by
  sorry

-- Problem 2
theorem fraction_simplification (a b : ℝ) (ha : a ≠ 0) :
  (a + (2 * a * b + b^2) / a) / ((a + b) / a) = a + b := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_fraction_simplification_l1947_194702


namespace NUMINAMATH_CALUDE_circle_radius_c_value_l1947_194764

theorem circle_radius_c_value :
  ∀ c : ℝ,
  (∀ x y : ℝ, x^2 + 8*x + y^2 + 2*y + c = 0 ↔ (x + 4)^2 + (y + 1)^2 = 25) →
  c = -8 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_radius_c_value_l1947_194764


namespace NUMINAMATH_CALUDE_cross_section_perimeter_bound_l1947_194711

/-- A regular tetrahedron with edge length a -/
structure RegularTetrahedron where
  a : ℝ
  edge_positive : 0 < a

/-- A triangular cross-section through one vertex of a regular tetrahedron -/
structure TriangularCrossSection (t : RegularTetrahedron) where
  perimeter : ℝ

/-- The perimeter of any triangular cross-section through one vertex of a regular tetrahedron
    is greater than twice the edge length -/
theorem cross_section_perimeter_bound (t : RegularTetrahedron) 
  (s : TriangularCrossSection t) : s.perimeter > 2 * t.a := by
  sorry

end NUMINAMATH_CALUDE_cross_section_perimeter_bound_l1947_194711


namespace NUMINAMATH_CALUDE_consecutive_draws_count_l1947_194774

/-- The number of ways to draw 4 consecutively numbered balls from a set of 20 balls. -/
def consecutiveDraws : ℕ := 17

/-- The total number of balls in the bin. -/
def totalBalls : ℕ := 20

/-- The number of balls to be drawn. -/
def ballsDrawn : ℕ := 4

theorem consecutive_draws_count :
  consecutiveDraws = totalBalls - ballsDrawn + 1 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_draws_count_l1947_194774


namespace NUMINAMATH_CALUDE_balance_disruption_possible_l1947_194786

/-- Represents a coin with a weight of either 7 or 8 grams -/
inductive Coin
  | Light : Coin  -- 7 grams
  | Heavy : Coin  -- 8 grams

/-- Represents the state of the balance scale -/
structure BalanceState :=
  (left : List Coin)
  (right : List Coin)

/-- Checks if the balance scale is in equilibrium -/
def isBalanced (state : BalanceState) : Bool :=
  (state.left.length = state.right.length) &&
  (state.left.foldl (fun acc c => acc + match c with
    | Coin.Light => 7
    | Coin.Heavy => 8) 0 =
   state.right.foldl (fun acc c => acc + match c with
    | Coin.Light => 7
    | Coin.Heavy => 8) 0)

/-- Performs a swap operation on the balance scale -/
def swapCoins (state : BalanceState) (n : Nat) : BalanceState :=
  { left := state.right.take n ++ state.left.drop n,
    right := state.left.take n ++ state.right.drop n }

/-- The main theorem to be proved -/
theorem balance_disruption_possible :
  ∀ (initialState : BalanceState),
    initialState.left.length = 144 →
    initialState.right.length = 144 →
    isBalanced initialState →
    ∃ (finalState : BalanceState),
      ∃ (numOperations : Nat),
        numOperations ≤ 11 ∧
        ¬isBalanced finalState ∧
        (∃ (swaps : List Nat),
          swaps.length = numOperations ∧
          finalState = swaps.foldl swapCoins initialState) :=
sorry

end NUMINAMATH_CALUDE_balance_disruption_possible_l1947_194786


namespace NUMINAMATH_CALUDE_range_of_m_l1947_194744

def p (x : ℝ) : Prop := |1 - (x - 1) / 3| ≤ 2

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, ¬(p x) → ¬(q x m)) →
  (∃ x, ¬(p x) ∧ (q x m)) →
  m ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1947_194744


namespace NUMINAMATH_CALUDE_dynaco_shares_sold_is_150_l1947_194782

/-- Represents the stock portfolio problem --/
structure StockProblem where
  microtron_price : ℕ
  dynaco_price : ℕ
  total_shares : ℕ
  average_price : ℕ

/-- Calculates the number of Dynaco shares sold --/
def dynaco_shares_sold (p : StockProblem) : ℕ :=
  (p.total_shares * p.average_price - p.microtron_price * p.total_shares) / (p.dynaco_price - p.microtron_price)

/-- Theorem stating that given the problem conditions, 150 Dynaco shares were sold --/
theorem dynaco_shares_sold_is_150 (p : StockProblem) 
  (h1 : p.microtron_price = 36)
  (h2 : p.dynaco_price = 44)
  (h3 : p.total_shares = 300)
  (h4 : p.average_price = 40) :
  dynaco_shares_sold p = 150 := by
  sorry

#eval dynaco_shares_sold { microtron_price := 36, dynaco_price := 44, total_shares := 300, average_price := 40 }

end NUMINAMATH_CALUDE_dynaco_shares_sold_is_150_l1947_194782


namespace NUMINAMATH_CALUDE_factorization_equality_l1947_194792

theorem factorization_equality (a b : ℝ) : a^2*b - 2*a*b + b = b*(a-1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1947_194792


namespace NUMINAMATH_CALUDE_circle_tangent_to_x_axis_at_origin_l1947_194701

-- Define a circle using its general equation
def Circle (D E F : ℝ) := {(x, y) : ℝ × ℝ | x^2 + y^2 + D*x + E*y + F = 0}

-- Define what it means for a circle to be tangent to the x-axis at the origin
def TangentToXAxisAtOrigin (c : Set (ℝ × ℝ)) : Prop :=
  (0, 0) ∈ c ∧ ∀ y ≠ 0, (0, y) ∉ c

-- Theorem statement
theorem circle_tangent_to_x_axis_at_origin (D E F : ℝ) :
  TangentToXAxisAtOrigin (Circle D E F) → D = 0 ∧ E ≠ 0 ∧ F ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_x_axis_at_origin_l1947_194701


namespace NUMINAMATH_CALUDE_function_value_at_half_l1947_194747

theorem function_value_at_half (f : ℝ → ℝ) :
  (∀ x ∈ Set.Icc 0 (π / 2), f (Real.sin x) = x) →
  f (1 / 2) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_half_l1947_194747


namespace NUMINAMATH_CALUDE_jack_christina_lindy_meeting_l1947_194726

/-- The problem of Jack, Christina, and Lindy meeting --/
theorem jack_christina_lindy_meeting 
  (jack_speed : ℝ) 
  (christina_speed : ℝ) 
  (lindy_speed : ℝ) 
  (lindy_distance : ℝ) : 
  jack_speed = 3 → 
  christina_speed = 3 → 
  lindy_speed = 10 → 
  lindy_distance = 400 → 
  ∃ (initial_distance : ℝ), 
    initial_distance = 240 ∧ 
    (initial_distance / 2) / jack_speed = lindy_distance / lindy_speed :=
sorry

end NUMINAMATH_CALUDE_jack_christina_lindy_meeting_l1947_194726


namespace NUMINAMATH_CALUDE_dirk_profit_l1947_194729

/-- Calculates the profit for selling amulets at a Ren Faire --/
def amulet_profit (days : ℕ) (amulets_per_day : ℕ) (sell_price : ℕ) (cost_price : ℕ) (faire_fee_percent : ℕ) : ℕ :=
  let total_amulets := days * amulets_per_day
  let revenue := total_amulets * sell_price
  let faire_fee := revenue * faire_fee_percent / 100
  let revenue_after_fee := revenue - faire_fee
  let total_cost := total_amulets * cost_price
  revenue_after_fee - total_cost

/-- Theorem stating that Dirk's profit is 300 dollars --/
theorem dirk_profit :
  amulet_profit 2 25 40 30 10 = 300 := by
  sorry

end NUMINAMATH_CALUDE_dirk_profit_l1947_194729


namespace NUMINAMATH_CALUDE_largest_even_not_sum_of_composite_odds_l1947_194790

/-- A function that checks if a natural number is composite -/
def isComposite (n : ℕ) : Prop :=
  ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

/-- A function that checks if a natural number is odd -/
def isOdd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

/-- A function that checks if a natural number can be expressed as the sum of two composite odd positive integers -/
def isSumOfTwoCompositeOdds (n : ℕ) : Prop :=
  ∃ a b, isComposite a ∧ isComposite b ∧ isOdd a ∧ isOdd b ∧ n = a + b

/-- Theorem stating that 38 is the largest even positive integer that cannot be expressed as the sum of two composite odd positive integers -/
theorem largest_even_not_sum_of_composite_odds :
  (∀ n : ℕ, n > 38 → isSumOfTwoCompositeOdds n) ∧
  ¬isSumOfTwoCompositeOdds 38 ∧
  (∀ n : ℕ, n < 38 → n % 2 = 0 → isSumOfTwoCompositeOdds n ∨ n < 38) :=
sorry

end NUMINAMATH_CALUDE_largest_even_not_sum_of_composite_odds_l1947_194790


namespace NUMINAMATH_CALUDE_largest_equal_cost_number_l1947_194783

/-- Calculates the sum of digits in decimal representation -/
def sumDecimalDigits (n : Nat) : Nat :=
  sorry

/-- Calculates the sum of digits in binary representation -/
def sumBinaryDigits (n : Nat) : Nat :=
  sorry

/-- Checks if the costs are equal for both options -/
def equalCost (n : Nat) : Prop :=
  sumDecimalDigits n = sumBinaryDigits n

theorem largest_equal_cost_number :
  (∀ m : Nat, m < 500 → m > 404 → ¬(equalCost m)) ∧
  equalCost 404 :=
sorry

end NUMINAMATH_CALUDE_largest_equal_cost_number_l1947_194783


namespace NUMINAMATH_CALUDE_population_growth_theorem_l1947_194761

/-- The annual population growth rate due to natural growth -/
def natural_growth_rate : ℝ := 0.06

/-- The overall population growth rate over 3 years -/
def total_growth_rate : ℝ := 0.157625

/-- The annual population decrease rate due to migration -/
def migration_decrease_rate : ℝ := 0.009434

theorem population_growth_theorem :
  ∃ (x : ℝ),
    (((1 + natural_growth_rate) * (1 - x))^3 = 1 + total_growth_rate) ∧
    (abs (x - migration_decrease_rate) < 0.00001) := by
  sorry

end NUMINAMATH_CALUDE_population_growth_theorem_l1947_194761


namespace NUMINAMATH_CALUDE_inner_set_area_of_specific_triangle_l1947_194739

/-- Triangle with side lengths a, b, c -/
structure Triangle (a b c : ℝ) where
  side_a : a > 0
  side_b : b > 0
  side_c : c > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Set of points inside a triangle not within distance d of any side -/
def InnerSet (T : Triangle a b c) (d : ℝ) : Set (ℝ × ℝ) :=
  sorry

/-- Area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Main theorem -/
theorem inner_set_area_of_specific_triangle :
  let T : Triangle 26 51 73 := ⟨sorry, sorry, sorry, sorry⟩
  let S := InnerSet T 5
  area S = 135 / 28 := by
  sorry

end NUMINAMATH_CALUDE_inner_set_area_of_specific_triangle_l1947_194739
