import Mathlib

namespace NUMINAMATH_CALUDE_rectangle_longest_side_l2364_236483

/-- Given a rectangle with perimeter 240 feet and area equal to eight times its perimeter,
    the length of its longest side is 101 feet. -/
theorem rectangle_longest_side : ∀ l w : ℝ,
  (2 * l + 2 * w = 240) →  -- perimeter is 240 feet
  (l * w = 8 * 240) →      -- area is 8 times the perimeter
  (l ≥ 0 ∧ w ≥ 0) →        -- length and width are non-negative
  (max l w = 101) :=       -- the longest side is 101 feet
by sorry

end NUMINAMATH_CALUDE_rectangle_longest_side_l2364_236483


namespace NUMINAMATH_CALUDE_old_edition_pages_l2364_236493

theorem old_edition_pages (new_edition : ℕ) (h1 : new_edition = 450) 
  (h2 : new_edition = 2 * old_edition - 230) : old_edition = 340 :=
by
  sorry

end NUMINAMATH_CALUDE_old_edition_pages_l2364_236493


namespace NUMINAMATH_CALUDE_permutation_count_mod_500_l2364_236482

/-- Represents the number of ways to arrange letters in specific positions -/
def arrange (n m : ℕ) : ℕ := Nat.choose n m

/-- Calculates the sum of products of arrangements for different k values -/
def sum_arrangements : ℕ :=
  (arrange 5 1 * arrange 6 0 * arrange 7 2) +
  (arrange 5 2 * arrange 6 1 * arrange 7 3) +
  (arrange 5 3 * arrange 6 2 * arrange 7 4) +
  (arrange 5 4 * arrange 6 3 * arrange 7 5) +
  (arrange 5 5 * arrange 6 4 * arrange 7 6)

/-- The main theorem stating the result of the permutation count modulo 500 -/
theorem permutation_count_mod_500 :
  sum_arrangements % 500 = 160 := by sorry

end NUMINAMATH_CALUDE_permutation_count_mod_500_l2364_236482


namespace NUMINAMATH_CALUDE_shelby_gold_stars_yesterday_l2364_236401

/-- Proves that Shelby earned 4 gold stars yesterday -/
theorem shelby_gold_stars_yesterday (yesterday : ℕ) (today : ℕ) (total : ℕ)
  (h1 : today = 3)
  (h2 : total = 7)
  (h3 : yesterday + today = total) :
  yesterday = 4 := by
  sorry

end NUMINAMATH_CALUDE_shelby_gold_stars_yesterday_l2364_236401


namespace NUMINAMATH_CALUDE_paving_cost_l2364_236446

/-- The cost of paving a rectangular floor -/
theorem paving_cost (length width rate : ℝ) (h1 : length = 5.5) (h2 : width = 3.75) (h3 : rate = 1000) :
  length * width * rate = 20625 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_l2364_236446


namespace NUMINAMATH_CALUDE_sum_of_cubes_divisibility_l2364_236428

theorem sum_of_cubes_divisibility (a b c : ℤ) : 
  (3 ∣ (a + b + c)) → (3 ∣ (a^3 + b^3 + c^3)) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_divisibility_l2364_236428


namespace NUMINAMATH_CALUDE_selection_with_girl_count_l2364_236405

def num_boys : Nat := 4
def num_girls : Nat := 3
def num_selected : Nat := 3
def num_tasks : Nat := 3

theorem selection_with_girl_count :
  (Nat.choose (num_boys + num_girls) num_selected * Nat.factorial num_tasks) -
  (Nat.choose num_boys num_selected * Nat.factorial num_tasks) = 186 := by
  sorry

end NUMINAMATH_CALUDE_selection_with_girl_count_l2364_236405


namespace NUMINAMATH_CALUDE_bobs_improvement_percentage_l2364_236414

theorem bobs_improvement_percentage (bob_time sister_time : ℕ) 
  (h1 : bob_time = 640) 
  (h2 : sister_time = 320) : 
  (bob_time - sister_time) / bob_time * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_bobs_improvement_percentage_l2364_236414


namespace NUMINAMATH_CALUDE_problem_solution_l2364_236485

def B : Set ℝ := {m | ∀ x ∈ Set.Icc (-1) 1, x^2 - x - m < 0}

def A (a : ℝ) : Set ℝ := {x | (x - 3*a) * (x - a - 2) < 0}

theorem problem_solution :
  (B = Set.Ioi 2) ∧
  ({a : ℝ | A a ⊆ B ∧ A a ≠ B} = Set.Ici (2/3)) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2364_236485


namespace NUMINAMATH_CALUDE_compound_interest_calculation_l2364_236454

/-- Compound interest calculation --/
theorem compound_interest_calculation 
  (principal : ℝ) 
  (rate : ℝ) 
  (time : ℕ) 
  (h1 : principal = 5000)
  (h2 : rate = 0.1)
  (h3 : time = 2) :
  principal * (1 + rate) ^ time = 6050 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_calculation_l2364_236454


namespace NUMINAMATH_CALUDE_inverse_matrices_solution_l2364_236404

/-- Given two 2x2 matrices that are inverses of each other, prove that a = 6 and b = 3/25 -/
theorem inverse_matrices_solution :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, 3; 2, 5]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![1/5, -1/5; b, 2/5]
  A * B = 1 → a = 6 ∧ b = 3/25 := by
  sorry

#check inverse_matrices_solution

end NUMINAMATH_CALUDE_inverse_matrices_solution_l2364_236404


namespace NUMINAMATH_CALUDE_x_gt_neg_two_necessary_not_sufficient_for_x_sq_lt_four_l2364_236410

theorem x_gt_neg_two_necessary_not_sufficient_for_x_sq_lt_four :
  (∀ x : ℝ, x^2 < 4 → x > -2) ∧
  (∃ x : ℝ, x > -2 ∧ x^2 ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_neg_two_necessary_not_sufficient_for_x_sq_lt_four_l2364_236410


namespace NUMINAMATH_CALUDE_collinearity_condition_l2364_236417

/-- Three points in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Collinearity condition for three points in a 2D plane -/
def collinear (A B C : Point2D) : Prop :=
  A.x * B.y + B.x * C.y + C.x * A.y = A.y * B.x + B.y * C.x + C.y * A.x

/-- Theorem: Three points are collinear iff they satisfy the collinearity condition -/
theorem collinearity_condition (A B C : Point2D) :
  collinear A B C ↔ A.x * B.y + B.x * C.y + C.x * A.y = A.y * B.x + B.y * C.x + C.y * A.x :=
sorry

end NUMINAMATH_CALUDE_collinearity_condition_l2364_236417


namespace NUMINAMATH_CALUDE_class_mean_calculation_l2364_236406

/-- Calculates the overall mean score for a class given two groups of students and their respective mean scores -/
theorem class_mean_calculation 
  (total_students : ℕ) 
  (group1_students : ℕ) 
  (group2_students : ℕ) 
  (group1_mean : ℚ) 
  (group2_mean : ℚ) 
  (h1 : total_students = group1_students + group2_students)
  (h2 : total_students = 32)
  (h3 : group1_students = 24)
  (h4 : group2_students = 8)
  (h5 : group1_mean = 85 / 100)
  (h6 : group2_mean = 90 / 100) :
  (group1_students * group1_mean + group2_students * group2_mean) / total_students = 8625 / 10000 := by
  sorry


end NUMINAMATH_CALUDE_class_mean_calculation_l2364_236406


namespace NUMINAMATH_CALUDE_f_minimum_value_l2364_236467

noncomputable def f (x : ℝ) : ℝ := x + 1/x + 1/(x + 1/x) + 1/x^2

theorem f_minimum_value (x : ℝ) (hx : x > 0) : f x ≥ 3.5 ∧ f 1 = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_f_minimum_value_l2364_236467


namespace NUMINAMATH_CALUDE_x_less_equal_two_l2364_236438

theorem x_less_equal_two (x : ℝ) (h : Real.sqrt ((x - 2)^2) = 2 - x) : x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_x_less_equal_two_l2364_236438


namespace NUMINAMATH_CALUDE_ammonia_formation_l2364_236432

-- Define the chemical reaction
structure Reaction where
  koh : ℝ  -- moles of Potassium hydroxide
  nh4i : ℝ  -- moles of Ammonium iodide
  nh3 : ℝ  -- moles of Ammonia formed

-- Define the balanced equation
axiom balanced_equation (r : Reaction) : r.koh = r.nh4i ∧ r.koh = r.nh3

-- Theorem: Given 3 moles of KOH, the number of moles of NH3 formed is 3
theorem ammonia_formation (r : Reaction) (h : r.koh = 3) : r.nh3 = 3 := by
  sorry

-- The proof is omitted as per instructions

end NUMINAMATH_CALUDE_ammonia_formation_l2364_236432


namespace NUMINAMATH_CALUDE_correct_calculation_l2364_236491

theorem correct_calculation (x : ℝ) : 8 * x + 8 = 56 → (x / 8) + 7 = 7.75 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2364_236491


namespace NUMINAMATH_CALUDE_gretchen_earnings_l2364_236437

/-- Gretchen's caricature business --/
def caricature_problem (price_per_drawing : ℚ) (saturday_sales : ℕ) (sunday_sales : ℕ) : ℚ :=
  (saturday_sales + sunday_sales : ℚ) * price_per_drawing

/-- Theorem stating the total money Gretchen made --/
theorem gretchen_earnings :
  caricature_problem 20 24 16 = 800 := by
  sorry

end NUMINAMATH_CALUDE_gretchen_earnings_l2364_236437


namespace NUMINAMATH_CALUDE_fraction_sum_zero_l2364_236499

theorem fraction_sum_zero (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_zero_l2364_236499


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l2364_236400

/-- Two lines are parallel if their slopes are equal -/
def parallel (m1 n1 c1 m2 n2 c2 : ℝ) : Prop :=
  m1 * n2 = m2 * n1

/-- The problem statement -/
theorem parallel_lines_a_value (a : ℝ) :
  parallel (1 + a) 1 1 2 a 2 → a = 1 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l2364_236400


namespace NUMINAMATH_CALUDE_suzy_final_book_count_l2364_236418

/-- Calculates the final number of books Suzy has after a series of transactions -/
def final_book_count (initial_books : ℕ) 
                     (wed_checkout : ℕ) 
                     (thu_return thu_checkout : ℕ) 
                     (fri_return : ℕ) : ℕ :=
  initial_books - wed_checkout + thu_return - thu_checkout + fri_return

/-- Theorem stating that Suzy ends up with 80 books given the specific transactions -/
theorem suzy_final_book_count : 
  final_book_count 98 43 23 5 7 = 80 := by
  sorry

end NUMINAMATH_CALUDE_suzy_final_book_count_l2364_236418


namespace NUMINAMATH_CALUDE_rafael_monday_hours_l2364_236477

/-- Represents the number of hours Rafael worked on Monday -/
def monday_hours : ℕ := sorry

/-- Represents the number of hours Rafael worked on Tuesday -/
def tuesday_hours : ℕ := 8

/-- Represents the number of hours Rafael has left to work in the week -/
def remaining_hours : ℕ := 20

/-- Represents Rafael's hourly pay rate in dollars -/
def hourly_rate : ℕ := 20

/-- Represents Rafael's total earnings for the week in dollars -/
def total_earnings : ℕ := 760

/-- Theorem stating that Rafael worked 10 hours on Monday -/
theorem rafael_monday_hours :
  monday_hours = 10 :=
by sorry

end NUMINAMATH_CALUDE_rafael_monday_hours_l2364_236477


namespace NUMINAMATH_CALUDE_absolute_value_reciprocal_2023_l2364_236459

theorem absolute_value_reciprocal_2023 :
  {x : ℝ | |x| = (1 : ℝ) / 2023} = {-(1 : ℝ) / 2023, (1 : ℝ) / 2023} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_reciprocal_2023_l2364_236459


namespace NUMINAMATH_CALUDE_toy_pricing_and_profit_l2364_236426

/-- Represents the order quantity and price for toys -/
structure ToyOrder where
  quantity : ℕ
  price : ℚ

/-- Calculates the factory price based on order quantity -/
def factoryPrice (x : ℕ) : ℚ :=
  if x ≤ 100 then 60
  else if x < 600 then max (62 - x / 50) 50
  else 50

/-- Calculates the profit for a given order quantity -/
def profit (x : ℕ) : ℚ := (factoryPrice x - 40) * x

theorem toy_pricing_and_profit :
  (∃ x : ℕ, x > 100 ∧ factoryPrice x = 50 → x = 600) ∧
  (∀ x : ℕ, x > 0 → factoryPrice x = 
    if x ≤ 100 then 60
    else if x < 600 then 62 - x / 50
    else 50) ∧
  profit 500 = 6000 := by
  sorry


end NUMINAMATH_CALUDE_toy_pricing_and_profit_l2364_236426


namespace NUMINAMATH_CALUDE_smallest_square_addition_l2364_236455

theorem smallest_square_addition (n : ℕ) (h : n = 2020) : 
  ∃ k : ℕ, k = 1 ∧ 
  (∃ m : ℕ, (n - 1) * n * (n + 1) * (n + 2) + k = m^2) ∧
  (∀ j : ℕ, j < k → ¬∃ m : ℕ, (n - 1) * n * (n + 1) * (n + 2) + j = m^2) :=
by sorry

#check smallest_square_addition

end NUMINAMATH_CALUDE_smallest_square_addition_l2364_236455


namespace NUMINAMATH_CALUDE_unique_residue_mod_16_l2364_236453

theorem unique_residue_mod_16 : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 15 ∧ n ≡ -3125 [ZMOD 16] := by
  sorry

end NUMINAMATH_CALUDE_unique_residue_mod_16_l2364_236453


namespace NUMINAMATH_CALUDE_system_solution_l2364_236462

theorem system_solution :
  let x : ℝ := -1
  let y : ℝ := (Real.sqrt 3 + 1) / 2
  (Real.sqrt 3 * x + 2 * y = 1) ∧ (x + 2 * y = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2364_236462


namespace NUMINAMATH_CALUDE_point_on_parabola_l2364_236416

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = 2 * x^2 - 3 * x + 1

/-- Theorem: The point (1/2, 0) lies on the parabola y = 2x^2 - 3x + 1 -/
theorem point_on_parabola : parabola (1/2) 0 := by sorry

end NUMINAMATH_CALUDE_point_on_parabola_l2364_236416


namespace NUMINAMATH_CALUDE_prime_satisfying_condition_l2364_236497

def satisfies_condition (p : Nat) : Prop :=
  Nat.Prime p ∧
  ∀ q : Nat, Nat.Prime q → q < p →
    ∀ k r : Nat, p = k * q + r → 0 ≤ r → r < q →
      ∀ a : Nat, a > 1 → ¬(a^2 ∣ r)

theorem prime_satisfying_condition :
  {p : Nat | satisfies_condition p} = {2, 3, 5, 7, 13} := by sorry

end NUMINAMATH_CALUDE_prime_satisfying_condition_l2364_236497


namespace NUMINAMATH_CALUDE_actual_distance_traveled_l2364_236479

theorem actual_distance_traveled (original_speed : ℝ) (increased_speed : ℝ) (additional_distance : ℝ) :
  original_speed = 15 →
  increased_speed = 25 →
  additional_distance = 35 →
  (∃ (time : ℝ), time > 0 ∧ time * increased_speed = time * original_speed + additional_distance) →
  ∃ (actual_distance : ℝ), actual_distance = 52.5 ∧ actual_distance = original_speed * (actual_distance / original_speed) :=
by sorry

end NUMINAMATH_CALUDE_actual_distance_traveled_l2364_236479


namespace NUMINAMATH_CALUDE_positive_real_inequalities_l2364_236412

theorem positive_real_inequalities (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b^2 + c^2 ≥ a*b + b*c + c*a) ∧ ((a + b + c)^2 ≥ 3*(a*b + b*c + c*a)) := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequalities_l2364_236412


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2364_236434

theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, (x - 2*a)*(a*x - 1) < 0 ↔ (x > 1/a ∨ x < 2*a)) →
  a ≤ -Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2364_236434


namespace NUMINAMATH_CALUDE_solution_in_interval_l2364_236449

theorem solution_in_interval : ∃ x : ℝ, x ∈ Set.Ioo 0 1 ∧ 2^x + x - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_in_interval_l2364_236449


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2364_236444

/-- Arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  a 1 = 1/2 ∧ S 4 = 20 ∧
  ∀ n : ℕ, S n = n * (a 1) + (n * (n - 1)) / 2 * (a 2 - a 1)

/-- Theorem stating the common difference and S_6 for the given arithmetic sequence -/
theorem arithmetic_sequence_properties (a : ℕ → ℚ) (S : ℕ → ℚ) 
  (h : arithmetic_sequence a S) : (a 2 - a 1 = 3) ∧ (S 6 = 48) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2364_236444


namespace NUMINAMATH_CALUDE_fruit_mix_grapes_l2364_236447

theorem fruit_mix_grapes (b r g c : ℚ) : 
  b + r + g + c = 400 →
  r = 3 * b →
  g = 2 * c →
  c = 5 * r →
  g = 12000 / 49 := by
sorry

end NUMINAMATH_CALUDE_fruit_mix_grapes_l2364_236447


namespace NUMINAMATH_CALUDE_median_is_six_l2364_236481

/-- Represents the attendance data for a group of students -/
structure AttendanceData where
  total_students : Nat
  attend_4_times : Nat
  attend_5_times : Nat
  attend_6_times : Nat
  attend_7_times : Nat
  attend_8_times : Nat

/-- Calculates the median attendance for a given AttendanceData -/
def median_attendance (data : AttendanceData) : Nat :=
  sorry

/-- Theorem stating that the median attendance for the given data is 6 -/
theorem median_is_six (data : AttendanceData) 
  (h1 : data.total_students = 20)
  (h2 : data.attend_4_times = 1)
  (h3 : data.attend_5_times = 5)
  (h4 : data.attend_6_times = 7)
  (h5 : data.attend_7_times = 4)
  (h6 : data.attend_8_times = 3) :
  median_attendance data = 6 := by
  sorry

end NUMINAMATH_CALUDE_median_is_six_l2364_236481


namespace NUMINAMATH_CALUDE_parallel_lines_corresponding_angles_l2364_236466

/-- Represents a line in a plane -/
structure Line where
  -- Add necessary fields here
  
/-- Represents an angle in a plane -/
structure Angle where
  -- Add necessary fields here

/-- Represents a plane -/
structure Plane where
  -- Add necessary fields here

/-- Two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  sorry

/-- A line intersects two other lines -/
def intersects (l : Line) (l1 l2 : Line) : Prop :=
  sorry

/-- Two angles are corresponding angles -/
def corresponding_angles (a1 a2 : Angle) (l1 l2 l : Line) : Prop :=
  sorry

/-- Two angles are equal -/
def angles_equal (a1 a2 : Angle) : Prop :=
  sorry

/-- Two angles are supplementary -/
def angles_supplementary (a1 a2 : Angle) : Prop :=
  sorry

/-- Main theorem: If two lines are parallel and intersected by a transversal,
    then the corresponding angles are either equal or supplementary -/
theorem parallel_lines_corresponding_angles 
  (p : Plane) (l1 l2 l : Line) (a1 a2 : Angle) :
  parallel l1 l2 → intersects l l1 l2 → corresponding_angles a1 a2 l1 l2 l →
  angles_equal a1 a2 ∨ angles_supplementary a1 a2 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_corresponding_angles_l2364_236466


namespace NUMINAMATH_CALUDE_problem_solution_l2364_236452

theorem problem_solution (p q r : ℝ) : 
  (∀ x : ℝ, (x - p) * (x - q) / (x - r) ≤ 0 ↔ (x < -6 ∨ |x - 30| ≤ 2)) →
  p < q →
  p + 2*q + 3*r = 74 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2364_236452


namespace NUMINAMATH_CALUDE_apples_difference_l2364_236494

/-- The number of apples Yanna bought -/
def total_apples : ℕ := 60

/-- The number of apples Yanna gave to Zenny -/
def apples_to_zenny : ℕ := 18

/-- The number of apples Yanna kept for herself -/
def apples_kept : ℕ := 36

/-- The number of apples Yanna gave to Andrea -/
def apples_to_andrea : ℕ := total_apples - apples_to_zenny - apples_kept

theorem apples_difference :
  apples_to_zenny - apples_to_andrea = 12 :=
by sorry

end NUMINAMATH_CALUDE_apples_difference_l2364_236494


namespace NUMINAMATH_CALUDE_set_equality_implies_subset_l2364_236473

theorem set_equality_implies_subset (A B C : Set α) :
  A ∪ B = B ∩ C → A ⊆ C := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_subset_l2364_236473


namespace NUMINAMATH_CALUDE_pablo_puzzle_pieces_l2364_236407

/-- The number of pieces Pablo can put together per hour -/
def piecesPerHour : ℕ := 100

/-- The number of 300-piece puzzles Pablo has -/
def numLargePuzzles : ℕ := 8

/-- The number of puzzles with unknown pieces Pablo has -/
def numSmallPuzzles : ℕ := 5

/-- The maximum number of hours Pablo works on puzzles per day -/
def hoursPerDay : ℕ := 7

/-- The number of days it takes Pablo to complete all puzzles -/
def totalDays : ℕ := 7

/-- The number of pieces in each of the large puzzles -/
def piecesPerLargePuzzle : ℕ := 300

/-- The number of pieces in each of the small puzzles -/
def piecesPerSmallPuzzle : ℕ := 500

theorem pablo_puzzle_pieces :
  piecesPerSmallPuzzle * numSmallPuzzles + piecesPerLargePuzzle * numLargePuzzles = 
  piecesPerHour * hoursPerDay * totalDays :=
by sorry

end NUMINAMATH_CALUDE_pablo_puzzle_pieces_l2364_236407


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l2364_236495

theorem negation_of_proposition (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_specific_proposition :
  (¬ ∀ x : ℝ, x^2 + x - 1 < 0) ↔ (∃ x : ℝ, x^2 + x - 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l2364_236495


namespace NUMINAMATH_CALUDE_cube_sum_of_roots_l2364_236448

theorem cube_sum_of_roots (a b c : ℝ) : 
  (3 * a^3 - 2 * a^2 + 5 * a - 7 = 0) ∧ 
  (3 * b^3 - 2 * b^2 + 5 * b - 7 = 0) ∧ 
  (3 * c^3 - 2 * c^2 + 5 * c - 7 = 0) → 
  a^3 + b^3 + c^3 = 137 / 27 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_of_roots_l2364_236448


namespace NUMINAMATH_CALUDE_additional_cards_l2364_236427

theorem additional_cards (total_cards : ℕ) (complete_decks : ℕ) (cards_per_deck : ℕ) : 
  total_cards = 160 ∧ complete_decks = 3 ∧ cards_per_deck = 52 →
  total_cards - (complete_decks * cards_per_deck) = 4 := by
sorry

end NUMINAMATH_CALUDE_additional_cards_l2364_236427


namespace NUMINAMATH_CALUDE_a_spending_percentage_l2364_236458

def total_salary : ℝ := 4000
def a_salary : ℝ := 3000
def b_spending_percentage : ℝ := 0.85

theorem a_spending_percentage :
  ∃ (a_spending : ℝ),
    a_spending = 0.95 ∧
    a_salary * (1 - a_spending) = (total_salary - a_salary) * (1 - b_spending_percentage) :=
by sorry

end NUMINAMATH_CALUDE_a_spending_percentage_l2364_236458


namespace NUMINAMATH_CALUDE_three_digit_seven_divisible_by_five_l2364_236439

theorem three_digit_seven_divisible_by_five (N : ℕ) : 
  (100 ≤ N ∧ N ≤ 999) →  -- N is a three-digit number
  (N % 10 = 7) →         -- N has a ones digit of 7
  (N % 5 = 0) →          -- N is divisible by 5
  False :=               -- This is impossible
by sorry

end NUMINAMATH_CALUDE_three_digit_seven_divisible_by_five_l2364_236439


namespace NUMINAMATH_CALUDE_scientific_notation_of_29150000_l2364_236470

theorem scientific_notation_of_29150000 : 
  29150000 = 2.915 * (10 : ℝ)^7 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_29150000_l2364_236470


namespace NUMINAMATH_CALUDE_first_wheat_rate_calculation_l2364_236488

-- Define the variables and constants
def first_wheat_quantity : ℝ := 30
def second_wheat_quantity : ℝ := 20
def second_wheat_rate : ℝ := 14.25
def profit_percentage : ℝ := 0.10
def selling_rate : ℝ := 13.86

-- Define the theorem
theorem first_wheat_rate_calculation (x : ℝ) : 
  (1 + profit_percentage) * (first_wheat_quantity * x + second_wheat_quantity * second_wheat_rate) = 
  (first_wheat_quantity + second_wheat_quantity) * selling_rate → 
  x = 11.50 := by
sorry

end NUMINAMATH_CALUDE_first_wheat_rate_calculation_l2364_236488


namespace NUMINAMATH_CALUDE_cube_dimension_ratio_l2364_236433

theorem cube_dimension_ratio (v1 v2 : ℝ) (h1 : v1 = 216) (h2 : v2 = 1728) :
  (v2 / v1) ^ (1/3 : ℝ) = 2 := by
sorry

end NUMINAMATH_CALUDE_cube_dimension_ratio_l2364_236433


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l2364_236480

theorem arithmetic_expression_equality : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l2364_236480


namespace NUMINAMATH_CALUDE_atomic_weight_Ba_value_l2364_236441

/-- The atomic weight of Fluorine (F) -/
def atomic_weight_F : ℝ := 19

/-- The molecular weight of the compound BaF₂ -/
def molecular_weight_BaF2 : ℝ := 175

/-- The number of F atoms in the compound -/
def num_F_atoms : ℕ := 2

/-- The atomic weight of Barium (Ba) -/
def atomic_weight_Ba : ℝ := molecular_weight_BaF2 - num_F_atoms * atomic_weight_F

theorem atomic_weight_Ba_value : atomic_weight_Ba = 137 := by sorry

end NUMINAMATH_CALUDE_atomic_weight_Ba_value_l2364_236441


namespace NUMINAMATH_CALUDE_center_of_given_hyperbola_l2364_236429

/-- The center of a hyperbola is the point (h, k) in the standard form 
    (x-h)^2/a^2 - (y-k)^2/b^2 = 1 or (y-k)^2/a^2 - (x-h)^2/b^2 = 1 -/
def center_of_hyperbola (a b c d e f : ℝ) : ℝ × ℝ := sorry

/-- The equation of a hyperbola in general form is ax^2 + bxy + cy^2 + dx + ey + f = 0 -/
def is_hyperbola (a b c d e f : ℝ) : Prop := sorry

theorem center_of_given_hyperbola :
  let a : ℝ := 9
  let b : ℝ := 0
  let c : ℝ := -16
  let d : ℝ := -54
  let e : ℝ := 128
  let f : ℝ := -400
  is_hyperbola a b c d e f →
  center_of_hyperbola a b c d e f = (3, 4) := by sorry

end NUMINAMATH_CALUDE_center_of_given_hyperbola_l2364_236429


namespace NUMINAMATH_CALUDE_square_area_with_circles_l2364_236456

theorem square_area_with_circles (r : ℝ) (h : r = 7) : 
  (4 * r) ^ 2 = 784 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_circles_l2364_236456


namespace NUMINAMATH_CALUDE_primitive_triples_theorem_l2364_236411

/-- A triple of positive integers (a, b, c) is primitive if they have no common prime factors -/
def isPrimitive (a b c : ℕ+) : Prop :=
  ∀ p : ℕ, Nat.Prime p → ¬(p ∣ a.val ∧ p ∣ b.val ∧ p ∣ c.val)

/-- Each number in the triple divides the sum of the other two -/
def eachDividesSumOfOthers (a b c : ℕ+) : Prop :=
  a ∣ (b + c) ∧ b ∣ (a + c) ∧ c ∣ (a + b)

/-- The main theorem -/
theorem primitive_triples_theorem :
  ∀ a b c : ℕ+, a ≤ b → b ≤ c →
  isPrimitive a b c → eachDividesSumOfOthers a b c →
  (a, b, c) = (1, 1, 1) ∨ (a, b, c) = (1, 1, 2) ∨ (a, b, c) = (1, 2, 3) :=
sorry

end NUMINAMATH_CALUDE_primitive_triples_theorem_l2364_236411


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2364_236476

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, ax^2 + bx + 1 > 0 ↔ -1 < x ∧ x < 1/3) →
  a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2364_236476


namespace NUMINAMATH_CALUDE_negative_three_point_fourteen_greater_than_negative_pi_l2364_236490

theorem negative_three_point_fourteen_greater_than_negative_pi : -3.14 > -π := by
  sorry

end NUMINAMATH_CALUDE_negative_three_point_fourteen_greater_than_negative_pi_l2364_236490


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2364_236423

theorem complex_modulus_problem : 
  Complex.abs ((3 + Complex.I) / (1 - Complex.I)) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2364_236423


namespace NUMINAMATH_CALUDE_ceiling_sqrt_count_l2364_236498

theorem ceiling_sqrt_count (x : ℤ) : (∃ (count : ℕ), count = 39 ∧ 
  (∀ y : ℤ, ⌈Real.sqrt (y : ℝ)⌉ = 20 ↔ 362 ≤ y ∧ y ≤ 400) ∧
  count = (Finset.range 39).card) :=
sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_count_l2364_236498


namespace NUMINAMATH_CALUDE_simplify_and_ratio_l2364_236469

theorem simplify_and_ratio (m n : ℤ) : 
  let simplified := (5*m + 15*n + 20) / 5
  ∃ (a b c : ℤ), 
    simplified = a*m + b*n + c ∧ 
    (a + b) / c = 1 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_ratio_l2364_236469


namespace NUMINAMATH_CALUDE_square_diagonal_l2364_236450

theorem square_diagonal (A : ℝ) (h : A = 338) : 
  ∃ d : ℝ, d^2 = 2 * A ∧ d = 26 :=
by sorry

end NUMINAMATH_CALUDE_square_diagonal_l2364_236450


namespace NUMINAMATH_CALUDE_carl_index_card_cost_l2364_236487

/-- The cost of index cards for Carl's classes -/
def total_cost (cards_per_student : ℕ) (periods : ℕ) (students_per_class : ℕ) (pack_size : ℕ) (pack_cost : ℚ) : ℚ :=
  let total_cards := cards_per_student * periods * students_per_class
  let packs_needed := (total_cards + pack_size - 1) / pack_size  -- Ceiling division
  packs_needed * pack_cost

/-- Proof that Carl spent $108 on index cards -/
theorem carl_index_card_cost :
  total_cost 10 6 30 50 3 = 108 := by
  sorry

end NUMINAMATH_CALUDE_carl_index_card_cost_l2364_236487


namespace NUMINAMATH_CALUDE_store_visitors_l2364_236403

theorem store_visitors (first_hour_left second_hour_in second_hour_out final_count : ℕ) :
  first_hour_left = 27 →
  second_hour_in = 18 →
  second_hour_out = 9 →
  final_count = 76 →
  ∃ first_hour_in : ℕ, first_hour_in = 94 ∧
    final_count = first_hour_in - first_hour_left + second_hour_in - second_hour_out :=
by sorry

end NUMINAMATH_CALUDE_store_visitors_l2364_236403


namespace NUMINAMATH_CALUDE_quiz_show_winning_probability_l2364_236430

def num_questions : ℕ := 4
def choices_per_question : ℕ := 3
def min_correct_to_win : ℕ := 3

def probability_of_correct_answer : ℚ := 1 / choices_per_question

/-- The probability of winning the quiz show -/
def probability_of_winning : ℚ :=
  (num_questions.choose min_correct_to_win) * (probability_of_correct_answer ^ min_correct_to_win) * ((1 - probability_of_correct_answer) ^ (num_questions - min_correct_to_win)) +
  (num_questions.choose (min_correct_to_win + 1)) * (probability_of_correct_answer ^ (min_correct_to_win + 1)) * ((1 - probability_of_correct_answer) ^ (num_questions - (min_correct_to_win + 1)))

theorem quiz_show_winning_probability :
  probability_of_winning = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_quiz_show_winning_probability_l2364_236430


namespace NUMINAMATH_CALUDE_chimney_bricks_proof_l2364_236413

/-- The time it takes Brenda to build the chimney alone (in hours) -/
def brenda_time : ℝ := 9

/-- The time it takes Brandon to build the chimney alone (in hours) -/
def brandon_time : ℝ := 10

/-- The decrease in combined output when working together (in bricks per hour) -/
def output_decrease : ℝ := 10

/-- The time it takes Brenda and Brandon to build the chimney together (in hours) -/
def combined_time : ℝ := 5

/-- The number of bricks in the chimney -/
def chimney_bricks : ℝ := 900

theorem chimney_bricks_proof :
  let brenda_rate := chimney_bricks / brenda_time
  let brandon_rate := chimney_bricks / brandon_time
  let combined_rate := brenda_rate + brandon_rate - output_decrease
  chimney_bricks = combined_rate * combined_time := by
  sorry

end NUMINAMATH_CALUDE_chimney_bricks_proof_l2364_236413


namespace NUMINAMATH_CALUDE_smallest_degree_for_horizontal_asymptote_l2364_236478

/-- The numerator of our rational function -/
def f (x : ℝ) : ℝ := 5 * x^7 + 2 * x^4 - 7

/-- A proposition stating that a rational function has a horizontal asymptote -/
def has_horizontal_asymptote (num den : ℝ → ℝ) : Prop :=
  ∃ (L : ℝ), ∀ ε > 0, ∃ M, ∀ x, |x| > M → |num x / den x - L| < ε

/-- The main theorem: the smallest possible degree of p(x) is 7 -/
theorem smallest_degree_for_horizontal_asymptote :
  ∀ p : ℝ → ℝ, has_horizontal_asymptote f p → (∃ n : ℕ, ∀ x, p x = x^n) → 
  (∀ m : ℕ, (∃ x, p x = x^m) → m ≥ 7) :=
sorry

end NUMINAMATH_CALUDE_smallest_degree_for_horizontal_asymptote_l2364_236478


namespace NUMINAMATH_CALUDE_sugar_calculation_l2364_236415

/-- Given a recipe with a sugar to flour ratio and an amount of flour,
    calculate the amount of sugar needed. -/
def sugar_amount (sugar_flour_ratio : ℚ) (flour_amount : ℚ) : ℚ :=
  sugar_flour_ratio * flour_amount

theorem sugar_calculation (sugar_flour_ratio flour_amount : ℚ) :
  sugar_flour_ratio = 10 / 1 →
  flour_amount = 5 →
  sugar_amount sugar_flour_ratio flour_amount = 50 := by
sorry

end NUMINAMATH_CALUDE_sugar_calculation_l2364_236415


namespace NUMINAMATH_CALUDE_largest_divisor_for_multiples_of_three_l2364_236489

def f (n : ℕ) : ℕ := n * (n + 2) * (n + 4) * (n + 6) * (n + 8)

theorem largest_divisor_for_multiples_of_three :
  ∃ (d : ℕ), d = 288 ∧
  (∀ (n : ℕ), 3 ∣ n → d ∣ f n) ∧
  (∀ (m : ℕ), m > d → ∃ (n : ℕ), 3 ∣ n ∧ ¬(m ∣ f n)) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_for_multiples_of_three_l2364_236489


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2364_236464

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2364_236464


namespace NUMINAMATH_CALUDE_pipe_fill_time_l2364_236457

theorem pipe_fill_time (fill_time_A fill_time_all empty_time : ℝ) 
  (h1 : fill_time_A = 60)
  (h2 : fill_time_all = 50)
  (h3 : empty_time = 100.00000000000001) :
  ∃ fill_time_B : ℝ, fill_time_B = 75 ∧ 
  (1 / fill_time_A + 1 / fill_time_B - 1 / empty_time = 1 / fill_time_all) := by
  sorry

#check pipe_fill_time

end NUMINAMATH_CALUDE_pipe_fill_time_l2364_236457


namespace NUMINAMATH_CALUDE_may_friday_to_monday_l2364_236402

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a day in May -/
structure DayInMay where
  day : Nat
  dayOfWeek : DayOfWeek

/-- The function that determines the day of the week for a given day in May -/
def dayOfWeekInMay (d : Nat) : DayOfWeek :=
  sorry

theorem may_friday_to_monday (r n : Nat) 
  (h1 : dayOfWeekInMay r = DayOfWeek.Friday)
  (h2 : dayOfWeekInMay n = DayOfWeek.Monday)
  (h3 : 15 < n)
  (h4 : n < 25) :
  n = 20 := by
  sorry

end NUMINAMATH_CALUDE_may_friday_to_monday_l2364_236402


namespace NUMINAMATH_CALUDE_smallest_prime_in_sum_l2364_236421

theorem smallest_prime_in_sum (p q r s : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r → Nat.Prime s →
  p + q + r = 2 * s →
  1 < p → p < q → q < r →
  p = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_in_sum_l2364_236421


namespace NUMINAMATH_CALUDE_shoe_difference_l2364_236445

def scott_shoes : ℕ := 7
def anthony_shoes : ℕ := 3 * scott_shoes
def jim_shoes : ℕ := anthony_shoes - 2

theorem shoe_difference : anthony_shoes - jim_shoes = 2 := by
  sorry

end NUMINAMATH_CALUDE_shoe_difference_l2364_236445


namespace NUMINAMATH_CALUDE_batsman_average_increase_l2364_236475

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  innings : ℕ
  totalScore : ℕ
  average : ℚ

/-- Calculates the new average after an additional inning -/
def newAverage (stats : BatsmanStats) (newScore : ℕ) : ℚ :=
  (stats.totalScore + newScore) / (stats.innings + 1)

/-- Theorem: If a batsman's average increases by 1 after scoring 69 in the 11th inning,
    then the new average is 59 -/
theorem batsman_average_increase (stats : BatsmanStats) :
  stats.innings = 10 →
  newAverage stats 69 = stats.average + 1 →
  newAverage stats 69 = 59 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l2364_236475


namespace NUMINAMATH_CALUDE_expression_simplification_l2364_236422

theorem expression_simplification 
  (x y z p q r : ℝ) 
  (hx : x ≠ p) 
  (hy : y ≠ q) 
  (hz : z ≠ r) 
  (hpq : p ≠ q) 
  (hqr : q ≠ r) 
  (hpr : p ≠ r) : 
  (2 * (x - p)) / (3 * (r - z)) * 
  (2 * (y - q)) / (3 * (p - x)) * 
  (2 * (z - r)) / (3 * (q - y)) = -8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2364_236422


namespace NUMINAMATH_CALUDE_six_times_r_of_30_l2364_236471

def r (θ : ℚ) : ℚ := 1 / (2 - θ)

theorem six_times_r_of_30 : r (r (r (r (r (r 30))))) = 144 / 173 := by
  sorry

end NUMINAMATH_CALUDE_six_times_r_of_30_l2364_236471


namespace NUMINAMATH_CALUDE_total_earnings_l2364_236461

theorem total_earnings (jerusha_earnings lottie_earnings : ℕ) :
  jerusha_earnings = 68 →
  jerusha_earnings = 4 * lottie_earnings →
  jerusha_earnings + lottie_earnings = 85 :=
by
  sorry

end NUMINAMATH_CALUDE_total_earnings_l2364_236461


namespace NUMINAMATH_CALUDE_model_evaluation_criteria_l2364_236440

-- Define the concept of a model
def Model : Type := ℝ → ℝ

-- Define the concept of residuals
def Residuals (m : Model) (data : Set (ℝ × ℝ)) : Set ℝ := sorry

-- Define the concept of residual plot distribution
def EvenlyDistributedInHorizontalBand (r : Set ℝ) : Prop := sorry

-- Define the sum of squared residuals
def SumSquaredResiduals (r : Set ℝ) : ℝ := sorry

-- Define the concept of model appropriateness
def ModelAppropriate (m : Model) (data : Set (ℝ × ℝ)) : Prop := 
  EvenlyDistributedInHorizontalBand (Residuals m data)

-- Define the concept of better fitting model
def BetterFittingModel (m1 m2 : Model) (data : Set (ℝ × ℝ)) : Prop :=
  SumSquaredResiduals (Residuals m1 data) < SumSquaredResiduals (Residuals m2 data)

-- Theorem statement
theorem model_evaluation_criteria 
  (m : Model) (data : Set (ℝ × ℝ)) (m1 m2 : Model) :
  (ModelAppropriate m data ↔ 
    EvenlyDistributedInHorizontalBand (Residuals m data)) ∧
  (BetterFittingModel m1 m2 data ↔ 
    SumSquaredResiduals (Residuals m1 data) < SumSquaredResiduals (Residuals m2 data)) :=
by sorry

end NUMINAMATH_CALUDE_model_evaluation_criteria_l2364_236440


namespace NUMINAMATH_CALUDE_intersection_and_lines_l2364_236409

-- Define the lines
def l₁ (x y : ℝ) : Prop := 2 * x + 3 * y - 5 = 0
def l₂ (x y : ℝ) : Prop := x + 2 * y - 3 = 0
def l₃ (x y : ℝ) : Prop := 2 * x + y - 5 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (1, 1)

-- Define the parallel and perpendicular lines
def parallel_line (x y : ℝ) : Prop := 2 * x + y - 3 = 0
def perpendicular_line (x y : ℝ) : Prop := x - 2 * y + 1 = 0

theorem intersection_and_lines :
  (∃ (x y : ℝ), l₁ x y ∧ l₂ x y) →
  (l₁ P.1 P.2 ∧ l₂ P.1 P.2) →
  (∀ (x y : ℝ), parallel_line x y ↔ (∃ (t : ℝ), x = P.1 + t ∧ y = P.2 + t * (-2))) ∧
  (∀ (x y : ℝ), perpendicular_line x y ↔ (∃ (t : ℝ), x = P.1 + t ∧ y = P.2 + t * (1/2))) :=
sorry

end NUMINAMATH_CALUDE_intersection_and_lines_l2364_236409


namespace NUMINAMATH_CALUDE_simplified_tax_for_leonid_business_l2364_236436

-- Define the types of tax regimes
inductive TaxRegime
  | UnifiedAgricultural
  | Simplified
  | General
  | Patent

-- Define the characteristics of a business
structure Business where
  isAgricultural : Bool
  isSmall : Bool
  hasComplexAccounting : Bool
  isNewEntrepreneur : Bool

-- Define the function to determine the appropriate tax regime
def appropriateTaxRegime (b : Business) : TaxRegime :=
  if b.isAgricultural then TaxRegime.UnifiedAgricultural
  else if b.isSmall && b.isNewEntrepreneur && !b.hasComplexAccounting then TaxRegime.Simplified
  else if !b.isSmall || b.hasComplexAccounting then TaxRegime.General
  else TaxRegime.Patent

-- Theorem statement
theorem simplified_tax_for_leonid_business :
  let leonidBusiness : Business := {
    isAgricultural := false,
    isSmall := true,
    hasComplexAccounting := false,
    isNewEntrepreneur := true
  }
  appropriateTaxRegime leonidBusiness = TaxRegime.Simplified :=
by sorry


end NUMINAMATH_CALUDE_simplified_tax_for_leonid_business_l2364_236436


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l2364_236451

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, |x| + x^2 ≥ 0) ↔ (∃ x₀ : ℝ, |x₀| + x₀^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l2364_236451


namespace NUMINAMATH_CALUDE_max_b_plus_c_is_negative_four_l2364_236443

/-- A quadratic function passing through two points with two x-intercepts -/
structure QuadraticFunction where
  a : ℕ+  -- a is a positive integer
  b : ℝ
  c : ℝ
  passes_through_points : (a : ℝ) * (-1)^2 + b * (-1) + c = 4 ∧
                          (a : ℝ) * 2^2 + b * 2 + c = 1
  two_intercepts : (b^2 : ℝ) - 4 * (a : ℝ) * c > 0

/-- The maximum value of b + c for a quadratic function with given properties is -4 -/
theorem max_b_plus_c_is_negative_four (f : QuadraticFunction) :
  ∃ (max : ℝ), max = -4 ∧ ∀ (g : QuadraticFunction), g.b + g.c ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_b_plus_c_is_negative_four_l2364_236443


namespace NUMINAMATH_CALUDE_unique_solution_to_equation_l2364_236492

theorem unique_solution_to_equation : ∃! (x : ℝ), x ≠ 0 ∧ (7 * x)^5 = (14 * x)^4 ∧ x = 16/7 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_to_equation_l2364_236492


namespace NUMINAMATH_CALUDE_two_bedroom_square_footage_l2364_236408

/-- Calculates the total square footage of two bedrooms -/
def total_square_footage (martha_bedroom : ℕ) (jenny_bedroom_difference : ℕ) : ℕ :=
  martha_bedroom + (martha_bedroom + jenny_bedroom_difference)

/-- Proves that the total square footage of two bedrooms is 300 square feet -/
theorem two_bedroom_square_footage :
  total_square_footage 120 60 = 300 := by
  sorry

end NUMINAMATH_CALUDE_two_bedroom_square_footage_l2364_236408


namespace NUMINAMATH_CALUDE_f_is_even_and_increasing_l2364_236435

-- Define the function f(x) = 2x^2 + 4
def f (x : ℝ) : ℝ := 2 * x^2 + 4

-- State the theorem
theorem f_is_even_and_increasing :
  (∀ x, f x = f (-x)) ∧ 
  (∀ x y, 0 < x ∧ x < y ∧ y < 1 → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_is_even_and_increasing_l2364_236435


namespace NUMINAMATH_CALUDE_skt_lineups_l2364_236431

/-- The total number of StarCraft progamers -/
def total_progamers : ℕ := 111

/-- The number of progamers SKT starts with -/
def initial_skt_progamers : ℕ := 11

/-- The number of progamers in a lineup -/
def lineup_size : ℕ := 5

/-- The number of different ordered lineups SKT could field -/
def num_lineups : ℕ := 4015440

theorem skt_lineups :
  (total_progamers : ℕ) = 111 →
  (initial_skt_progamers : ℕ) = 11 →
  (lineup_size : ℕ) = 5 →
  num_lineups = (Nat.choose initial_skt_progamers lineup_size +
                 Nat.choose initial_skt_progamers (lineup_size - 1) * (total_progamers - initial_skt_progamers)) *
                (Nat.factorial lineup_size) :=
by sorry

end NUMINAMATH_CALUDE_skt_lineups_l2364_236431


namespace NUMINAMATH_CALUDE_complex_square_equality_l2364_236420

theorem complex_square_equality (c d : ℕ+) :
  (c + d * Complex.I) ^ 2 = 7 + 24 * Complex.I →
  c + d * Complex.I = 4 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_equality_l2364_236420


namespace NUMINAMATH_CALUDE_empty_quadratic_set_l2364_236486

theorem empty_quadratic_set (a : ℝ) :
  ({x : ℝ | a * x^2 - 2 * a * x + 1 < 0} = ∅) ↔ (0 ≤ a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_empty_quadratic_set_l2364_236486


namespace NUMINAMATH_CALUDE_slope_range_for_intersection_l2364_236424

/-- A line with slope k intersects a hyperbola at two distinct points -/
def intersects_at_two_points (k : ℝ) : Prop :=
  ∃ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    y₁ = k * x₁ ∧ y₂ = k * x₂ ∧
    x₁^2 - y₁^2 = 2 ∧ x₂^2 - y₂^2 = 2

/-- The theorem stating the range of k for which the line intersects the hyperbola at two points -/
theorem slope_range_for_intersection :
  ∀ k : ℝ, intersects_at_two_points k ↔ -1 < k ∧ k < 1 :=
sorry

end NUMINAMATH_CALUDE_slope_range_for_intersection_l2364_236424


namespace NUMINAMATH_CALUDE_tournament_handshakes_eq_24_l2364_236472

/-- The number of handshakes in a tournament with 4 teams of 2 players each -/
def tournament_handshakes : ℕ :=
  let num_teams : ℕ := 4
  let players_per_team : ℕ := 2
  let total_players : ℕ := num_teams * players_per_team
  let handshakes_per_player : ℕ := total_players - players_per_team
  (total_players * handshakes_per_player) / 2

theorem tournament_handshakes_eq_24 : tournament_handshakes = 24 := by
  sorry

end NUMINAMATH_CALUDE_tournament_handshakes_eq_24_l2364_236472


namespace NUMINAMATH_CALUDE_acute_triangle_sine_sum_l2364_236496

theorem acute_triangle_sine_sum (α β γ : Real) 
  (acute_triangle : 0 < α ∧ α < π/2 ∧ 0 < β ∧ β < π/2 ∧ 0 < γ ∧ γ < π/2)
  (angle_sum : α + β + γ = π) : 
  Real.sin α + Real.sin β + Real.sin γ > 2 := by
sorry

end NUMINAMATH_CALUDE_acute_triangle_sine_sum_l2364_236496


namespace NUMINAMATH_CALUDE_remaining_children_fed_theorem_l2364_236465

/-- Represents the capacity of a meal in terms of adults and children -/
structure MealCapacity where
  adults : ℕ
  children : ℕ

/-- Calculates the number of children that can be fed with the remaining food -/
def remainingChildrenFed (capacity : MealCapacity) (adultsEaten : ℕ) : ℕ :=
  let remainingAdults := capacity.adults - adultsEaten
  (remainingAdults * capacity.children) / capacity.adults

/-- Theorem stating that given a meal for 70 adults or 90 children, 
    if 42 adults have eaten, the remaining food can feed 36 children -/
theorem remaining_children_fed_theorem (capacity : MealCapacity) 
  (h1 : capacity.adults = 70)
  (h2 : capacity.children = 90)
  (h3 : adultsEaten = 42) :
  remainingChildrenFed capacity adultsEaten = 36 := by
  sorry

#eval remainingChildrenFed { adults := 70, children := 90 } 42

end NUMINAMATH_CALUDE_remaining_children_fed_theorem_l2364_236465


namespace NUMINAMATH_CALUDE_diplomats_not_speaking_russian_l2364_236474

theorem diplomats_not_speaking_russian (total : ℕ) (french : ℕ) (neither_percent : ℚ) (both_percent : ℚ) :
  total = 150 →
  french = 17 →
  neither_percent = 1/5 →
  both_percent = 1/10 →
  ∃ (not_russian : ℕ), not_russian = 32 :=
by sorry

end NUMINAMATH_CALUDE_diplomats_not_speaking_russian_l2364_236474


namespace NUMINAMATH_CALUDE_product_division_problem_l2364_236442

theorem product_division_problem (x y : ℝ) (h1 : x = 1.4) (h2 : x ≠ 0) :
  Real.sqrt ((7 * x) / y) = x → y = 5 := by
  sorry

end NUMINAMATH_CALUDE_product_division_problem_l2364_236442


namespace NUMINAMATH_CALUDE_principal_amount_l2364_236468

/-- Calculates the total interest paid over 11 years given the principal amount -/
def totalInterest (principal : ℝ) : ℝ :=
  principal * 0.06 * 3 + principal * 0.09 * 5 + principal * 0.13 * 3

/-- Theorem stating that the principal amount borrowed is 8000 given the total interest paid -/
theorem principal_amount (totalInterestPaid : ℝ) 
  (h : totalInterestPaid = 8160) : 
  ∃ (principal : ℝ), totalInterest principal = totalInterestPaid ∧ principal = 8000 := by
  sorry

#check principal_amount

end NUMINAMATH_CALUDE_principal_amount_l2364_236468


namespace NUMINAMATH_CALUDE_expression_equality_l2364_236484

theorem expression_equality : (5^1003 + 6^1004)^2 - (5^1003 - 6^1004)^2 = 24 * 30^1003 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2364_236484


namespace NUMINAMATH_CALUDE_sum_congruence_modulo_nine_l2364_236425

theorem sum_congruence_modulo_nine : 
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_congruence_modulo_nine_l2364_236425


namespace NUMINAMATH_CALUDE_blue_to_yellow_ratio_l2364_236463

/-- Represents the number of fish of each color in the aquarium -/
structure FishCount where
  yellow : ℕ
  blue : ℕ
  green : ℕ
  other : ℕ

/-- The conditions of the aquarium -/
def aquariumConditions (f : FishCount) : Prop :=
  f.yellow = 12 ∧
  f.green = 2 * f.yellow ∧
  f.yellow + f.blue + f.green + f.other = 42

/-- The theorem stating the ratio of blue to yellow fish -/
theorem blue_to_yellow_ratio (f : FishCount) 
  (h : aquariumConditions f) : 
  f.blue * 2 = f.yellow := by sorry

end NUMINAMATH_CALUDE_blue_to_yellow_ratio_l2364_236463


namespace NUMINAMATH_CALUDE_cubic_equation_result_l2364_236419

theorem cubic_equation_result (x : ℝ) (h : x^3 + 4*x^2 = 8) :
  x^5 + 80*x^3 = -376*x^2 - 32*x + 768 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_result_l2364_236419


namespace NUMINAMATH_CALUDE_clerical_staff_reduction_l2364_236460

theorem clerical_staff_reduction (total_employees : ℕ) 
  (initial_clerical_ratio : ℚ) (clerical_reduction_ratio : ℚ) : 
  total_employees = 3600 →
  initial_clerical_ratio = 1/3 →
  clerical_reduction_ratio = 1/2 →
  let initial_clerical := (initial_clerical_ratio * total_employees : ℚ).num
  let remaining_clerical := (1 - clerical_reduction_ratio) * initial_clerical
  let new_total := total_employees - (clerical_reduction_ratio * initial_clerical : ℚ).num
  (remaining_clerical / new_total : ℚ) = 1/5 := by
sorry

end NUMINAMATH_CALUDE_clerical_staff_reduction_l2364_236460
