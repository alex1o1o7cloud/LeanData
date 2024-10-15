import Mathlib

namespace NUMINAMATH_CALUDE_three_intersection_points_l2808_280854

/-- The number of distinct points satisfying the given equations -/
def num_intersection_points : ℕ := 3

/-- First equation -/
def equation1 (x y : ℝ) : Prop :=
  (x + y - 7) * (2*x - 3*y + 7) = 0

/-- Second equation -/
def equation2 (x y : ℝ) : Prop :=
  (x - y + 3) * (3*x + 2*y - 18) = 0

/-- Theorem stating that there are exactly 3 distinct points satisfying both equations -/
theorem three_intersection_points :
  ∃! (points : Finset (ℝ × ℝ)), 
    points.card = num_intersection_points ∧
    ∀ p ∈ points, equation1 p.1 p.2 ∧ equation2 p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_three_intersection_points_l2808_280854


namespace NUMINAMATH_CALUDE_max_value_x_2y_plus_1_l2808_280821

theorem max_value_x_2y_plus_1 (x y : ℝ) 
  (hx : |x - 1| ≤ 1) 
  (hy : |y - 2| ≤ 1) : 
  ∃ (M : ℝ), M = 5 ∧ 
  (∀ z, |x - 2*y + 1| ≤ z ↔ M ≤ z) :=
sorry

end NUMINAMATH_CALUDE_max_value_x_2y_plus_1_l2808_280821


namespace NUMINAMATH_CALUDE_triangle_side_length_l2808_280856

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the length of a line segment
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define a median
def is_median (t : Triangle) (M : ℝ × ℝ) : Prop :=
  ∃ (m : ℝ × ℝ), m = ((t.B.1 + t.C.1) / 2, (t.B.2 + t.C.2) / 2) ∧ M = m

theorem triangle_side_length (t : Triangle) :
  length t.A t.B = 7 →
  length t.B t.C = 10 →
  (∃ (M : ℝ × ℝ), is_median t M ∧ length t.A M = 5) →
  length t.A t.C = Real.sqrt 51 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2808_280856


namespace NUMINAMATH_CALUDE_s_range_l2808_280819

theorem s_range (a b c : ℝ) 
  (ha : 1/2 ≤ a ∧ a ≤ 1) 
  (hb : 1/2 ≤ b ∧ b ≤ 1) 
  (hc : 1/2 ≤ c ∧ c ≤ 1) : 
  let s := (a + b) / (1 + c) + (b + c) / (1 + a) + (c + a) / (1 + b)
  2 ≤ s ∧ s ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_s_range_l2808_280819


namespace NUMINAMATH_CALUDE_integer_triple_solution_l2808_280814

theorem integer_triple_solution (x y z : ℤ) :
  x * y * z + 4 * (x + y + z) = 2 * (x * y + x * z + y * z) + 7 ↔
  (x = 1 ∧ y = 1 ∧ z = 1) ∨
  (x = 3 ∧ y = 3 ∧ z = 1) ∨
  (x = 3 ∧ y = 1 ∧ z = 3) ∨
  (x = 1 ∧ y = 3 ∧ z = 3) :=
by sorry

end NUMINAMATH_CALUDE_integer_triple_solution_l2808_280814


namespace NUMINAMATH_CALUDE_product_unit_digit_is_one_l2808_280847

def unit_digit (n : ℕ) : ℕ := n % 10

def numbers : List ℕ := [7858413, 10864231, 45823797, 97833129, 51679957, 
                         38213827, 75946153, 27489543, 94837311, 37621597]

theorem product_unit_digit_is_one :
  unit_digit (numbers.prod) = 1 := by
  sorry

#check product_unit_digit_is_one

end NUMINAMATH_CALUDE_product_unit_digit_is_one_l2808_280847


namespace NUMINAMATH_CALUDE_total_distance_walked_and_run_l2808_280844

/-- Calculates the total distance traveled when walking and running at different speeds for different durations. -/
theorem total_distance_walked_and_run 
  (walk_time : ℝ) (walk_speed : ℝ) (run_time : ℝ) (run_speed : ℝ) :
  walk_time = 60 →  -- 60 minutes walking
  walk_speed = 3 →  -- 3 mph walking speed
  run_time = 45 →   -- 45 minutes running
  run_speed = 8 →   -- 8 mph running speed
  (walk_time + run_time) / 60 = 1.75 →  -- Total time in hours
  walk_time / 60 * walk_speed + run_time / 60 * run_speed = 9 := by
  sorry

#check total_distance_walked_and_run

end NUMINAMATH_CALUDE_total_distance_walked_and_run_l2808_280844


namespace NUMINAMATH_CALUDE_perpendicular_line_l2808_280896

/-- Given a line L1 with equation 3x - 6y = 9 and a point P (2, -3),
    prove that the line L2 with equation y = -2x + 1 is perpendicular to L1 and passes through P. -/
theorem perpendicular_line (x y : ℝ) : 
  (∃ (m b : ℝ), (3 * x - 6 * y = 9) ∧ (y = m * x + b)) →  -- L1 equation
  (y = -2 * x + 1) →                                     -- L2 equation
  ((-2) * (1/2) = -1) →                                  -- Perpendicularity condition
  ((-3) = -2 * 2 + 1) →                                  -- Point P satisfies L2
  (∃ (x₀ y₀ : ℝ), x₀ = 2 ∧ y₀ = -3 ∧ y₀ = -2 * x₀ + 1) -- L2 passes through P
  := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_l2808_280896


namespace NUMINAMATH_CALUDE_clock_angle_at_15_40_clock_angle_at_15_40_is_130_l2808_280806

/-- The angle between clock hands at 15:40 --/
theorem clock_angle_at_15_40 : ℝ :=
  let minutes_past_hour : ℝ := 40
  let hours_past_12 : ℝ := 3
  let minutes_per_hour : ℝ := 60
  let degrees_per_circle : ℝ := 360
  let hours_per_revolution : ℝ := 12

  let minute_hand_angle : ℝ := (minutes_past_hour / minutes_per_hour) * degrees_per_circle
  let hour_hand_angle : ℝ := (hours_past_12 / hours_per_revolution +
                              minutes_past_hour / (minutes_per_hour * hours_per_revolution)) *
                             degrees_per_circle

  let angle_between : ℝ := |minute_hand_angle - hour_hand_angle|

  130 -- The actual proof is omitted

theorem clock_angle_at_15_40_is_130 : clock_angle_at_15_40 = 130 := by
  sorry -- Proof is omitted

end NUMINAMATH_CALUDE_clock_angle_at_15_40_clock_angle_at_15_40_is_130_l2808_280806


namespace NUMINAMATH_CALUDE_inequality_solution_l2808_280830

theorem inequality_solution (x : ℝ) : 
  (x^3 - 3*x^2 + 2*x) / (x^2 - 3*x + 2) ≤ 0 ↔ x ≤ 0 ∧ x ≠ 1 ∧ x ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2808_280830


namespace NUMINAMATH_CALUDE_units_digit_of_23_power_23_l2808_280804

theorem units_digit_of_23_power_23 : (23^23) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_23_power_23_l2808_280804


namespace NUMINAMATH_CALUDE_paint_containers_left_over_l2808_280838

/-- Calculates the number of paint containers left over after repainting a bathroom --/
theorem paint_containers_left_over 
  (initial_containers : ℕ) 
  (total_walls : ℕ) 
  (tiled_walls : ℕ) 
  (ceiling_containers : ℕ) 
  (gradient_containers_per_wall : ℕ) : 
  initial_containers = 16 →
  total_walls = 4 →
  tiled_walls = 1 →
  ceiling_containers = 1 →
  gradient_containers_per_wall = 1 →
  initial_containers - 
    (ceiling_containers + 
     (total_walls - tiled_walls) * (1 + gradient_containers_per_wall)) = 11 := by
  sorry


end NUMINAMATH_CALUDE_paint_containers_left_over_l2808_280838


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l2808_280888

theorem arithmetic_evaluation : 8 / 2 - 3 * 2 + 5^2 / 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l2808_280888


namespace NUMINAMATH_CALUDE_student_arrangement_probabilities_l2808_280840

/-- Represents the probability of various arrangements of 4 students in a row. -/
structure StudentArrangementProbabilities where
  /-- The total number of possible arrangements for 4 students. -/
  total_arrangements : ℕ
  /-- The number of arrangements where a specific student is at one end. -/
  student_at_end : ℕ
  /-- The number of arrangements where two specific students are at both ends. -/
  two_students_at_ends : ℕ

/-- Theorem stating the probabilities of various student arrangements. -/
theorem student_arrangement_probabilities 
  (probs : StudentArrangementProbabilities)
  (h1 : probs.total_arrangements = 24)
  (h2 : probs.student_at_end = 12)
  (h3 : probs.two_students_at_ends = 4) :
  let p1 := probs.student_at_end / probs.total_arrangements
  let p2 := probs.two_students_at_ends / probs.total_arrangements
  let p3 := 1 - (probs.total_arrangements - probs.student_at_end - probs.student_at_end + probs.two_students_at_ends) / probs.total_arrangements
  let p4 := (probs.total_arrangements - probs.student_at_end - probs.student_at_end + probs.two_students_at_ends) / probs.total_arrangements
  (p1 = 1/2) ∧ 
  (p2 = 1/6) ∧ 
  (p3 = 5/6) ∧ 
  (p4 = 1/6) := by
  sorry

end NUMINAMATH_CALUDE_student_arrangement_probabilities_l2808_280840


namespace NUMINAMATH_CALUDE_brittany_vacation_duration_l2808_280885

/-- The duration of Brittany's vacation --/
def vacation_duration (rebecca_age : ℕ) (age_difference : ℕ) (brittany_age_after : ℕ) : ℕ :=
  brittany_age_after - (rebecca_age + age_difference)

/-- Theorem stating that Brittany's vacation lasted 4 years --/
theorem brittany_vacation_duration :
  vacation_duration 25 3 32 = 4 := by
  sorry

end NUMINAMATH_CALUDE_brittany_vacation_duration_l2808_280885


namespace NUMINAMATH_CALUDE_division_multiplication_result_l2808_280845

theorem division_multiplication_result : 
  let x : ℝ := 6.5
  let y : ℝ := (x / 6) * 12
  y = 13 := by sorry

end NUMINAMATH_CALUDE_division_multiplication_result_l2808_280845


namespace NUMINAMATH_CALUDE_crucian_carps_count_l2808_280858

/-- The number of bags of feed each fish eats -/
def bags_per_fish : ℕ := 3

/-- The number of individual feed bags prepared -/
def individual_bags : ℕ := 60

/-- The number of 8-packet feed bags prepared -/
def multi_bags : ℕ := 15

/-- The number of packets in each multi-bag -/
def packets_per_multi_bag : ℕ := 8

/-- The number of colored carps in the tank -/
def colored_carps : ℕ := 52

/-- The total number of feed packets available -/
def total_packets : ℕ := individual_bags + multi_bags * packets_per_multi_bag

/-- The total number of fish that can be fed -/
def total_fish : ℕ := total_packets / bags_per_fish

/-- The number of crucian carps in the tank -/
def crucian_carps : ℕ := total_fish - colored_carps

theorem crucian_carps_count : crucian_carps = 8 := by
  sorry

end NUMINAMATH_CALUDE_crucian_carps_count_l2808_280858


namespace NUMINAMATH_CALUDE_prob_sum_le_4_l2808_280822

/-- The number of possible outcomes for a single die. -/
def die_outcomes : ℕ := 6

/-- The set of all possible outcomes when throwing two dice. -/
def all_outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range die_outcomes) (Finset.range die_outcomes)

/-- The set of favorable outcomes where the sum is less than or equal to 4. -/
def favorable_outcomes : Finset (ℕ × ℕ) :=
  all_outcomes.filter (fun p => p.1 + p.2 ≤ 4)

/-- The probability of the sum of two dice being less than or equal to 4. -/
theorem prob_sum_le_4 :
    (favorable_outcomes.card : ℚ) / all_outcomes.card = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_le_4_l2808_280822


namespace NUMINAMATH_CALUDE_smallest_number_of_students_l2808_280898

/-- Represents the number of students in each grade --/
structure Students where
  ninth : ℕ
  eighth : ℕ
  seventh : ℕ

/-- The ratio of 9th-graders to 8th-graders is 7:4 --/
def ratio_9th_8th (s : Students) : Prop :=
  7 * s.eighth = 4 * s.ninth

/-- The ratio of 9th-graders to 7th-graders is 10:3 --/
def ratio_9th_7th (s : Students) : Prop :=
  10 * s.seventh = 3 * s.ninth

/-- The total number of students --/
def total_students (s : Students) : ℕ :=
  s.ninth + s.eighth + s.seventh

/-- The main theorem stating the smallest possible number of students --/
theorem smallest_number_of_students :
  ∃ (s : Students),
    ratio_9th_8th s ∧
    ratio_9th_7th s ∧
    total_students s = 131 ∧
    (∀ (t : Students),
      ratio_9th_8th t → ratio_9th_7th t →
      total_students t ≥ total_students s) :=
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_students_l2808_280898


namespace NUMINAMATH_CALUDE_michelle_gas_usage_l2808_280853

theorem michelle_gas_usage (initial_gas final_gas : ℚ) : 
  initial_gas = 1/2 → 
  final_gas = 1/6 → 
  initial_gas - final_gas = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_michelle_gas_usage_l2808_280853


namespace NUMINAMATH_CALUDE_grid_product_problem_l2808_280836

theorem grid_product_problem (x y : ℚ) 
  (h1 : x * 3 = y) 
  (h2 : 7 * y = 350) : 
  x = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_grid_product_problem_l2808_280836


namespace NUMINAMATH_CALUDE_debby_water_bottles_l2808_280841

theorem debby_water_bottles (initial_bottles : ℕ) (days : ℕ) (remaining_bottles : ℕ) 
  (h1 : initial_bottles = 264)
  (h2 : days = 11)
  (h3 : remaining_bottles = 99) :
  (initial_bottles - remaining_bottles) / days = 15 := by
  sorry

end NUMINAMATH_CALUDE_debby_water_bottles_l2808_280841


namespace NUMINAMATH_CALUDE_max_working_groups_l2808_280815

theorem max_working_groups (total_instructors : ℕ) (group_size : ℕ) (max_membership : ℕ) :
  total_instructors = 36 →
  group_size = 4 →
  max_membership = 2 →
  (∃ (n : ℕ), n ≤ 18 ∧ 
    n * group_size ≤ total_instructors * max_membership ∧
    ∀ (m : ℕ), m > n → m * group_size > total_instructors * max_membership) :=
by sorry

end NUMINAMATH_CALUDE_max_working_groups_l2808_280815


namespace NUMINAMATH_CALUDE_object_max_height_time_l2808_280887

/-- The height function of the thrown object -/
def h (t : ℝ) : ℝ := -15 * (t - 3)^2 + 150

/-- The time at which the object reaches its maximum height -/
def t_max : ℝ := 3

theorem object_max_height_time :
  (∀ t, h t ≤ h t_max) ∧ h (t_max + 2) = 90 :=
by sorry

end NUMINAMATH_CALUDE_object_max_height_time_l2808_280887


namespace NUMINAMATH_CALUDE_divisibility_by_five_l2808_280883

theorem divisibility_by_five (k m n : ℕ+) 
  (hk : ¬ 5 ∣ k.val) (hm : ¬ 5 ∣ m.val) (hn : ¬ 5 ∣ n.val) : 
  5 ∣ (k.val^2 - m.val^2) ∨ 5 ∣ (m.val^2 - n.val^2) ∨ 5 ∣ (n.val^2 - k.val^2) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l2808_280883


namespace NUMINAMATH_CALUDE_complex_abs_value_l2808_280893

theorem complex_abs_value : Complex.abs (-3 - (8/5)*Complex.I) = 17/5 := by
  sorry

end NUMINAMATH_CALUDE_complex_abs_value_l2808_280893


namespace NUMINAMATH_CALUDE_square_side_length_l2808_280817

theorem square_side_length (diagonal : ℝ) (h : diagonal = 2 * Real.sqrt 2) :
  ∃ (side : ℝ), side * side * 2 = diagonal * diagonal ∧ side = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2808_280817


namespace NUMINAMATH_CALUDE_inequality_proof_l2808_280862

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a^2 - a^3) * (1 / b^2 - b^3) ≥ (31/8)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2808_280862


namespace NUMINAMATH_CALUDE_mathcounts_teach_probability_l2808_280825

def mathcounts_letters : Finset Char := {'M', 'A', 'T', 'H', 'C', 'O', 'U', 'N', 'T', 'S'}
def teach_letters : Finset Char := {'T', 'E', 'A', 'C', 'H'}

theorem mathcounts_teach_probability :
  let common_letters := mathcounts_letters ∩ teach_letters
  (common_letters.card : ℚ) / mathcounts_letters.card = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_mathcounts_teach_probability_l2808_280825


namespace NUMINAMATH_CALUDE_mascot_sales_theorem_l2808_280877

/-- Represents the monthly sales quantity as a function of selling price -/
def sales_quantity (x : ℝ) : ℝ := -2 * x + 360

/-- Represents the monthly sales profit as a function of selling price -/
def sales_profit (x : ℝ) : ℝ := sales_quantity x * (x - 30)

theorem mascot_sales_theorem :
  -- 1. The linear function satisfies the given conditions
  (sales_quantity 30 = 300 ∧ sales_quantity 45 = 270) ∧
  -- 2. The maximum profit occurs at x = 105 and equals 11250
  (∀ x : ℝ, sales_profit x ≤ sales_profit 105) ∧
  (sales_profit 105 = 11250) ∧
  -- 3. The minimum selling price for profit ≥ 10000 is 80
  (∀ x : ℝ, x ≥ 80 → sales_profit x ≥ 10000) ∧
  (∀ x : ℝ, x < 80 → sales_profit x < 10000) :=
by sorry

end NUMINAMATH_CALUDE_mascot_sales_theorem_l2808_280877


namespace NUMINAMATH_CALUDE_rearrangements_of_13358_l2808_280889

/-- The number of different five-digit numbers that can be formed by rearranging the digits in 13358 -/
def rearrangements : ℕ :=
  Nat.factorial 5 / (Nat.factorial 2 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1)

/-- Theorem stating that the number of rearrangements is 60 -/
theorem rearrangements_of_13358 : rearrangements = 60 := by
  sorry

end NUMINAMATH_CALUDE_rearrangements_of_13358_l2808_280889


namespace NUMINAMATH_CALUDE_number_operation_result_l2808_280870

theorem number_operation_result : 
  let x : ℚ := 33
  (x / 4) + 9 = 17.25 := by sorry

end NUMINAMATH_CALUDE_number_operation_result_l2808_280870


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l2808_280850

theorem arithmetic_evaluation : 2 * 7 + 9 * 4 - 6 * 5 + 8 * 3 = 44 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l2808_280850


namespace NUMINAMATH_CALUDE_exists_N_average_twelve_l2808_280803

theorem exists_N_average_twelve : ∃ N : ℝ, 11 < N ∧ N < 21 ∧ (8 + 12 + N) / 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_exists_N_average_twelve_l2808_280803


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2808_280881

theorem quadratic_minimum (x y : ℝ) : 
  y = x^2 + 16*x + 20 → (∀ z : ℝ, z = x^2 + 16*x + 20 → y ≤ z) → y = -44 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2808_280881


namespace NUMINAMATH_CALUDE_cookies_per_row_l2808_280878

theorem cookies_per_row (num_trays : ℕ) (rows_per_tray : ℕ) (total_cookies : ℕ) :
  num_trays = 4 →
  rows_per_tray = 5 →
  total_cookies = 120 →
  total_cookies / (num_trays * rows_per_tray) = 6 := by
sorry

end NUMINAMATH_CALUDE_cookies_per_row_l2808_280878


namespace NUMINAMATH_CALUDE_cubic_function_properties_l2808_280824

/-- A cubic function with a linear term -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + m*x

theorem cubic_function_properties (m : ℝ) (h : f m 1 = 5) :
  m = 4 ∧ ∀ x : ℝ, f m (-x) = -(f m x) := by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l2808_280824


namespace NUMINAMATH_CALUDE_exploration_writing_ratio_l2808_280828

theorem exploration_writing_ratio :
  let exploring_time : ℝ := 3
  let book_writing_time : ℝ := 0.5
  let total_time : ℝ := 5
  let notes_writing_time : ℝ := total_time - exploring_time - book_writing_time
  (notes_writing_time / exploring_time = 1 / 2) := by
sorry

end NUMINAMATH_CALUDE_exploration_writing_ratio_l2808_280828


namespace NUMINAMATH_CALUDE_quadratic_root_range_l2808_280868

def f (a x : ℝ) : ℝ := -x^2 + 2*a*x + 4*a + 1

theorem quadratic_root_range (a : ℝ) :
  (∃ r₁ r₂ : ℝ, r₁ < -1 ∧ r₂ > 3 ∧ f a r₁ = 0 ∧ f a r₂ = 0) →
  a > 4/5 ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l2808_280868


namespace NUMINAMATH_CALUDE_unique_angle_solution_l2808_280882

theorem unique_angle_solution :
  ∃! x : ℝ, 0 < x ∧ x < 180 ∧
    Real.tan ((150 - x) * π / 180) = 
      (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) / 
      (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) ∧
    x = 110 := by
  sorry

end NUMINAMATH_CALUDE_unique_angle_solution_l2808_280882


namespace NUMINAMATH_CALUDE_least_divisible_by_second_smallest_consecutive_primes_l2808_280812

def second_smallest_consecutive_primes : List Nat := [11, 13, 17, 19]

theorem least_divisible_by_second_smallest_consecutive_primes :
  (∀ n : Nat, n > 0 ∧ (∀ p ∈ second_smallest_consecutive_primes, p ∣ n) → n ≥ 46189) ∧
  (∀ p ∈ second_smallest_consecutive_primes, p ∣ 46189) :=
sorry

end NUMINAMATH_CALUDE_least_divisible_by_second_smallest_consecutive_primes_l2808_280812


namespace NUMINAMATH_CALUDE_product_is_very_large_l2808_280807

theorem product_is_very_large : 
  (3 + 2) * 
  (3^2 + 2^2) * 
  (3^4 + 2^4) * 
  (3^8 + 2^8) * 
  (3^16 + 2^16) * 
  (3^32 + 2^32) * 
  (3^64 + 2^64) > 10^400 := by
sorry

end NUMINAMATH_CALUDE_product_is_very_large_l2808_280807


namespace NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l2808_280894

theorem unique_solution_sqrt_equation :
  ∃! (x : ℝ), 2 * x + Real.sqrt (x - 3) = 7 :=
by
  -- The unique solution is x = 3.25
  use 3.25
  sorry

end NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l2808_280894


namespace NUMINAMATH_CALUDE_vector_sum_problem_l2808_280802

/-- Given two vectors a and b in ℝ³, prove that a + 2b equals the expected result. -/
theorem vector_sum_problem (a b : Fin 3 → ℝ) 
  (ha : a = ![1, 2, 3]) 
  (hb : b = ![-1, 0, 1]) : 
  a + 2 • b = ![-1, 2, 5] := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_problem_l2808_280802


namespace NUMINAMATH_CALUDE_binary_arithmetic_equality_l2808_280809

def binary_to_nat (s : String) : Nat :=
  s.foldl (fun acc c => 2 * acc + c.toNat - '0'.toNat) 0

theorem binary_arithmetic_equality : 
  let a := binary_to_nat "1011101"
  let b := binary_to_nat "1101"
  let c := binary_to_nat "101010"
  let d := binary_to_nat "110"
  let result := binary_to_nat "1110111100"
  ((a + b) * c) / d = result := by
  sorry

end NUMINAMATH_CALUDE_binary_arithmetic_equality_l2808_280809


namespace NUMINAMATH_CALUDE_quadratic_root_range_l2808_280829

theorem quadratic_root_range (a : ℝ) (x₁ x₂ : ℝ) :
  (∃ x, x^2 + a*x - 2 = 0) ∧ 
  (x₁^2 + a*x₁ - 2 = 0) ∧ 
  (x₂^2 + a*x₂ - 2 = 0) ∧ 
  (x₁ < 1) ∧ (1 < x₂) →
  a < 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l2808_280829


namespace NUMINAMATH_CALUDE_debby_drinks_six_bottles_per_day_l2808_280861

-- Define the total number of bottles
def total_bottles : ℕ := 12

-- Define the number of days the bottles last
def days_last : ℕ := 2

-- Define the function to calculate bottles per day
def bottles_per_day (total : ℕ) (days : ℕ) : ℚ :=
  (total : ℚ) / (days : ℚ)

-- Theorem statement
theorem debby_drinks_six_bottles_per_day :
  bottles_per_day total_bottles days_last = 6 := by
  sorry

end NUMINAMATH_CALUDE_debby_drinks_six_bottles_per_day_l2808_280861


namespace NUMINAMATH_CALUDE_cosine_range_theorem_l2808_280865

theorem cosine_range_theorem (f : ℝ → ℝ) (x : ℝ) :
  (f = λ x => Real.cos (x - π/3)) →
  (x ∈ Set.Icc 0 (π/2)) →
  (∀ y, y ∈ Set.range f ↔ y ∈ Set.Icc (1/2) 1) :=
sorry

end NUMINAMATH_CALUDE_cosine_range_theorem_l2808_280865


namespace NUMINAMATH_CALUDE_sin_negative_780_degrees_l2808_280869

theorem sin_negative_780_degrees : 
  Real.sin ((-780 : ℝ) * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_780_degrees_l2808_280869


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2808_280857

theorem min_value_of_expression (a b : ℝ) (h : a ≠ -1) :
  |a + b| + |1 / (a + 1) - b| ≥ 1 := by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2808_280857


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2808_280816

def i : ℂ := Complex.I

theorem complex_magnitude_problem (z : ℂ) (h : z * (i + 1) = i) : Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2808_280816


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2808_280852

theorem complex_fraction_equality : ∃ (i : ℂ), i^2 = -1 ∧ i / (1 + Real.sqrt 3 * i) = (Real.sqrt 3 / 4 : ℂ) + (1 / 4 : ℂ) * i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2808_280852


namespace NUMINAMATH_CALUDE_trapezoid_area_in_circle_l2808_280864

/-- The area of a trapezoid inscribed in a circle -/
theorem trapezoid_area_in_circle (R : ℝ) (α : ℝ) (h : 0 < α ∧ α < π) :
  let trapezoid_area := R^2 * (1 + Real.sin (α/2)) * Real.cos (α/2)
  let diameter := 2 * R
  let chord := 2 * R * Real.sin (α/2)
  let height := R * Real.cos (α/2)
  trapezoid_area = (diameter + chord) * height / 2 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_area_in_circle_l2808_280864


namespace NUMINAMATH_CALUDE_project_completion_time_l2808_280860

/-- Represents the number of days it takes to complete the project. -/
def total_days : ℕ := 21

/-- Represents the rate at which A completes the project per day. -/
def rate_A : ℚ := 1 / 20

/-- Represents the rate at which B completes the project per day. -/
def rate_B : ℚ := 1 / 30

/-- Represents the combined rate at which A and B complete the project per day when working together. -/
def combined_rate : ℚ := rate_A + rate_B

theorem project_completion_time (x : ℕ) :
  (↑(total_days - x) * combined_rate + ↑x * rate_B = 1) → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_project_completion_time_l2808_280860


namespace NUMINAMATH_CALUDE_rose_work_days_l2808_280874

/-- Proves that if John completes a work in 320 days, and both John and Rose together complete
    the same work in 192 days, then Rose completes the work alone in 384 days. -/
theorem rose_work_days (john_days : ℕ) (together_days : ℕ) (rose_days : ℕ) : 
  john_days = 320 → together_days = 192 → 
  1 / john_days + 1 / rose_days = 1 / together_days → 
  rose_days = 384 := by
sorry

end NUMINAMATH_CALUDE_rose_work_days_l2808_280874


namespace NUMINAMATH_CALUDE_compare_numbers_l2808_280871

-- Define the base conversion function
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun digit acc => acc * base + digit) 0

-- Define the numbers in their respective bases
def num1 : List Nat := [8, 5]
def base1 : Nat := 9

def num2 : List Nat := [2, 1, 0]
def base2 : Nat := 6

def num3 : List Nat := [1, 0, 0, 0]
def base3 : Nat := 4

def num4 : List Nat := [1, 1, 1, 1, 1, 1]
def base4 : Nat := 2

-- State the theorem
theorem compare_numbers :
  to_decimal num2 base2 > to_decimal num1 base1 ∧
  to_decimal num1 base1 > to_decimal num3 base3 ∧
  to_decimal num3 base3 > to_decimal num4 base4 := by
  sorry

end NUMINAMATH_CALUDE_compare_numbers_l2808_280871


namespace NUMINAMATH_CALUDE_max_value_of_b_l2808_280872

/-- Given functions f and g with a common point and tangent, prove the maximum value of b -/
theorem max_value_of_b (a : ℝ) (h_a : a > 0) : 
  let f := fun x : ℝ => (1/2) * x^2 + 2 * a * x
  let g := fun x b : ℝ => 3 * a^2 * Real.log x + b
  ∃ (x₀ b₀ : ℝ), 
    (f x₀ = g x₀ b₀) ∧ 
    (deriv f x₀ = deriv (fun x => g x b₀) x₀) →
  (∀ b : ℝ, ∃ (x : ℝ), (f x = g x b) ∧ (deriv f x = deriv (fun x => g x b) x) → b ≤ (3/2) * Real.exp ((2/3) : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_b_l2808_280872


namespace NUMINAMATH_CALUDE_product_of_decimals_l2808_280851

theorem product_of_decimals : 0.3 * 0.7 = 0.21 := by
  sorry

end NUMINAMATH_CALUDE_product_of_decimals_l2808_280851


namespace NUMINAMATH_CALUDE_soda_cost_l2808_280891

/-- Given the total cost of an order and the cost of sandwiches, 
    calculate the cost of each soda. -/
theorem soda_cost (total_cost sandwich_cost : ℚ) 
  (h1 : total_cost = 10.46)
  (h2 : sandwich_cost = 3.49)
  (h3 : 2 * sandwich_cost + 4 * (total_cost - 2 * sandwich_cost) / 4 = total_cost) :
  (total_cost - 2 * sandwich_cost) / 4 = 0.87 := by sorry

end NUMINAMATH_CALUDE_soda_cost_l2808_280891


namespace NUMINAMATH_CALUDE_sqrt_200_simplification_l2808_280835

theorem sqrt_200_simplification : Real.sqrt 200 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_200_simplification_l2808_280835


namespace NUMINAMATH_CALUDE_dog_eaten_cost_calculation_l2808_280831

def cake_cost (flour_cost sugar_cost butter_cost eggs_cost : ℚ) : ℚ :=
  flour_cost + sugar_cost + butter_cost + eggs_cost

def dog_eaten_cost (total_cost : ℚ) (total_slices mother_eaten_slices : ℕ) : ℚ :=
  (total_cost * (total_slices - mother_eaten_slices : ℚ)) / total_slices

theorem dog_eaten_cost_calculation :
  let flour_cost : ℚ := 4
  let sugar_cost : ℚ := 2
  let butter_cost : ℚ := 2.5
  let eggs_cost : ℚ := 0.5
  let total_slices : ℕ := 6
  let mother_eaten_slices : ℕ := 2
  let total_cost := cake_cost flour_cost sugar_cost butter_cost eggs_cost
  dog_eaten_cost total_cost total_slices mother_eaten_slices = 6 := by
  sorry

#eval dog_eaten_cost (cake_cost 4 2 2.5 0.5) 6 2

end NUMINAMATH_CALUDE_dog_eaten_cost_calculation_l2808_280831


namespace NUMINAMATH_CALUDE_difference_of_squares_65_35_l2808_280886

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_65_35_l2808_280886


namespace NUMINAMATH_CALUDE_divisibility_3_power_l2808_280811

theorem divisibility_3_power (n : ℕ) : 
  (∃ k : ℤ, 3^n + 1 = 10 * k) → (∃ m : ℤ, 3^(n+4) + 1 = 10 * m) := by
sorry

end NUMINAMATH_CALUDE_divisibility_3_power_l2808_280811


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l2808_280884

/-- The number of ways to distribute n identical balls into k distinct boxes with at least one ball in each box -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 4 ways to distribute 5 identical balls into 4 distinct boxes with at least one ball in each box -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 4 := by sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l2808_280884


namespace NUMINAMATH_CALUDE_missing_consonants_fraction_l2808_280818

theorem missing_consonants_fraction 
  (total_letters : ℕ) 
  (total_vowels : ℕ) 
  (total_missing : ℕ) 
  (missing_vowels : ℕ) 
  (h1 : total_letters = 26) 
  (h2 : total_vowels = 5) 
  (h3 : total_missing = 5) 
  (h4 : missing_vowels = 2) :
  (total_missing - missing_vowels) / (total_letters - total_vowels) = 1 / 7 := by
sorry

end NUMINAMATH_CALUDE_missing_consonants_fraction_l2808_280818


namespace NUMINAMATH_CALUDE_three_number_problem_l2808_280880

theorem three_number_problem (a b c : ℤ) 
  (sum_ab : a + b = 35)
  (sum_bc : b + c = 47)
  (sum_ca : c + a = 52) :
  (a + b + c = 67) ∧ (a * b * c = 9600) := by
sorry

end NUMINAMATH_CALUDE_three_number_problem_l2808_280880


namespace NUMINAMATH_CALUDE_frank_apples_l2808_280843

theorem frank_apples (frank : ℕ) (susan : ℕ) : 
  susan = 3 * frank →  -- Susan picked 3 times as many apples as Frank
  (2 * frank / 3 + 3 * susan / 2 : ℚ) = 78 →  -- Remaining apples after Frank sold 1/3 and Susan gave out 1/2
  frank = 36 := by
sorry

end NUMINAMATH_CALUDE_frank_apples_l2808_280843


namespace NUMINAMATH_CALUDE_triangle_properties_l2808_280859

open Real

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Triangle inequality
  A + B + C = π ∧  -- Sum of angles in a triangle
  A < π/2 ∧ B < π/2 ∧ C < π/2 ∧  -- Acute triangle
  sqrt 3 * tan A * tan B - tan A - tan B = sqrt 3 ∧  -- Given condition
  c = 2 →  -- Given side length
  C = π/3 ∧ 20/3 < a^2 + b^2 ∧ a^2 + b^2 ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2808_280859


namespace NUMINAMATH_CALUDE_cookingAndYogaCount_l2808_280875

/-- Represents a group of people participating in various curriculums -/
structure CurriculumGroup where
  yoga : ℕ
  cooking : ℕ
  weaving : ℕ
  cookingOnly : ℕ
  allCurriculums : ℕ
  cookingAndWeaving : ℕ

/-- The number of people who study both cooking and yoga -/
def bothCookingAndYoga (g : CurriculumGroup) : ℕ :=
  g.cooking - g.cookingOnly - g.cookingAndWeaving + g.allCurriculums

/-- Theorem stating the number of people who study both cooking and yoga -/
theorem cookingAndYogaCount (g : CurriculumGroup) 
  (h1 : g.yoga = 35)
  (h2 : g.cooking = 20)
  (h3 : g.weaving = 15)
  (h4 : g.cookingOnly = 7)
  (h5 : g.allCurriculums = 3)
  (h6 : g.cookingAndWeaving = 5) :
  bothCookingAndYoga g = 5 := by
  sorry

#eval bothCookingAndYoga { yoga := 35, cooking := 20, weaving := 15, cookingOnly := 7, allCurriculums := 3, cookingAndWeaving := 5 }

end NUMINAMATH_CALUDE_cookingAndYogaCount_l2808_280875


namespace NUMINAMATH_CALUDE_hyperbola_intersection_length_l2808_280879

/-- Given a hyperbola with imaginary axis length 4 and eccentricity √6/2,
    if a line through the left focus intersects the left branch at points A and B
    such that |AB| is the arithmetic mean of |AF₂| and |BF₂|, then |AB| = 8√2 -/
theorem hyperbola_intersection_length
  (b : ℝ) (e : ℝ) (A B F₁ F₂ : ℝ × ℝ)
  (h_b : b = 2)
  (h_e : e = Real.sqrt 6 / 2)
  (h_foci : F₁.1 < F₂.1)
  (h_left_branch : A.1 < F₁.1 ∧ B.1 < F₁.1)
  (h_line : ∃ (m k : ℝ), A.2 = m * A.1 + k ∧ B.2 = m * B.1 + k ∧ F₁.2 = m * F₁.1 + k)
  (h_arithmetic_mean : 2 * dist A B = dist A F₂ + dist B F₂)
  (h_hyperbola : dist A F₂ - dist A F₁ = dist B F₂ - dist B F₁) :
  dist A B = 8 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_length_l2808_280879


namespace NUMINAMATH_CALUDE_total_candies_l2808_280810

/-- The number of candies in each gift box -/
def candies_per_box : ℕ := 156

/-- The number of children receiving gift boxes -/
def num_children : ℕ := 20

/-- Theorem: The total number of candies needed is 3120 -/
theorem total_candies : candies_per_box * num_children = 3120 := by
  sorry

end NUMINAMATH_CALUDE_total_candies_l2808_280810


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2808_280849

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b, a > b + 1 → a > b) ∧ 
  ¬(∀ a b, a > b → a > b + 1) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2808_280849


namespace NUMINAMATH_CALUDE_subtracted_value_l2808_280801

theorem subtracted_value (N : ℝ) (x : ℝ) 
  (h1 : (N - x) / 7 = 7) 
  (h2 : (N - 14) / 10 = 4) : 
  x = 5 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_l2808_280801


namespace NUMINAMATH_CALUDE_books_obtained_l2808_280867

/-- The number of additional books obtained by the class -/
def additional_books (initial final : ℕ) : ℕ := final - initial

/-- Proves that the number of additional books is 23 given the initial and final counts -/
theorem books_obtained (initial final : ℕ) 
  (h_initial : initial = 54)
  (h_final : final = 77) :
  additional_books initial final = 23 := by
  sorry

end NUMINAMATH_CALUDE_books_obtained_l2808_280867


namespace NUMINAMATH_CALUDE_average_speed_two_segments_l2808_280820

/-- Given a 100-mile trip where the first 50 miles are traveled at 20 mph
    and the remaining 50 miles at 50 mph, prove that the average speed
    for the entire trip is 100 / (50/20 + 50/50) miles per hour. -/
theorem average_speed_two_segments (total_distance : ℝ) (first_segment : ℝ) (second_segment : ℝ)
  (first_speed : ℝ) (second_speed : ℝ)
  (h1 : total_distance = 100)
  (h2 : first_segment = 50)
  (h3 : second_segment = 50)
  (h4 : first_speed = 20)
  (h5 : second_speed = 50)
  (h6 : total_distance = first_segment + second_segment) :
  (total_distance / (first_segment / first_speed + second_segment / second_speed)) =
  100 / (50 / 20 + 50 / 50) :=
by sorry

end NUMINAMATH_CALUDE_average_speed_two_segments_l2808_280820


namespace NUMINAMATH_CALUDE_last_segment_speed_l2808_280890

def total_distance : ℝ := 120
def total_time : ℝ := 1.5
def segment_time : ℝ := 0.5
def speed_segment1 : ℝ := 50
def speed_segment2 : ℝ := 70

theorem last_segment_speed :
  ∃ (speed_segment3 : ℝ),
    (speed_segment1 * segment_time + speed_segment2 * segment_time + speed_segment3 * segment_time) / total_time = total_distance / total_time ∧
    speed_segment3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_last_segment_speed_l2808_280890


namespace NUMINAMATH_CALUDE_problem_solution_l2808_280839

theorem problem_solution (x : ℕ+) : 
  x^2 + 4*x + 29 = x*(4*x + 9) + 13 → x = 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2808_280839


namespace NUMINAMATH_CALUDE_probability_tamika_greater_carlos_l2808_280832

def tamika_set : Finset ℕ := {8, 9, 10, 11}
def carlos_set : Finset ℕ := {3, 5, 6, 7}

def tamika_result (a b : ℕ) : ℕ := a + b

def carlos_result (a b : ℕ) : ℕ := a * b - 2

def valid_pair (s : Finset ℕ) (a b : ℕ) : Prop :=
  a ∈ s ∧ b ∈ s ∧ a ≠ b

def favorable_outcomes : ℕ := 26
def total_outcomes : ℕ := 36

theorem probability_tamika_greater_carlos :
  (↑favorable_outcomes / ↑total_outcomes : ℚ) = 13 / 18 := by sorry

end NUMINAMATH_CALUDE_probability_tamika_greater_carlos_l2808_280832


namespace NUMINAMATH_CALUDE_systematic_sampling_l2808_280800

theorem systematic_sampling (population : ℕ) (sample_size : ℕ) 
  (h_pop : population = 1650) (h_sample : sample_size = 35) :
  ∃ (removed : ℕ) (segments : ℕ) (per_segment : ℕ),
    removed = 5 ∧ 
    segments = sample_size ∧
    per_segment = 47 ∧
    (population - removed) = segments * per_segment :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_l2808_280800


namespace NUMINAMATH_CALUDE_diagonal_length_l2808_280866

/-- A quadrilateral with specific side lengths and an integer diagonal -/
structure Quadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  AC : ℤ
  h1 : AB = 9
  h2 : BC = 2
  h3 : CD = 14
  h4 : DA = 5

/-- The diagonal AC of the quadrilateral is 10 -/
theorem diagonal_length (q : Quadrilateral) : q.AC = 10 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_length_l2808_280866


namespace NUMINAMATH_CALUDE_log_inequality_l2808_280892

theorem log_inequality (a : ℝ) (h : 0 < a ∧ a < 1/4) :
  ∀ x : ℝ, (0 < x ∧ x ≠ 1 ∧ x + a > 0 ∧ x + a ≠ 1) →
  (Real.log 2 / Real.log (x + a) < Real.log 4 / Real.log x ↔
    (0 < x ∧ x < 1/2 - a - Real.sqrt (1/4 - a)) ∨
    (1/2 - a + Real.sqrt (1/4 - a) < x ∧ x < 1 - a) ∨
    (1 < x)) :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_l2808_280892


namespace NUMINAMATH_CALUDE_lcm_of_five_numbers_l2808_280848

theorem lcm_of_five_numbers :
  Nat.lcm 456 (Nat.lcm 783 (Nat.lcm 935 (Nat.lcm 1024 1297))) = 2308474368000 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_five_numbers_l2808_280848


namespace NUMINAMATH_CALUDE_expected_defective_meters_l2808_280899

/-- Proves that given a rejection rate of 1.5% and a sample size of 10,000 meters,
    the expected number of defective meters is 150. -/
theorem expected_defective_meters
  (rejection_rate : ℝ)
  (sample_size : ℕ)
  (h1 : rejection_rate = 0.015)
  (h2 : sample_size = 10000) :
  ↑sample_size * rejection_rate = 150 := by
  sorry

end NUMINAMATH_CALUDE_expected_defective_meters_l2808_280899


namespace NUMINAMATH_CALUDE_crayons_per_box_l2808_280834

theorem crayons_per_box (total_crayons : ℕ) (num_boxes : ℕ) (h1 : total_crayons = 35) (h2 : num_boxes = 7) :
  total_crayons / num_boxes = 5 := by
sorry

end NUMINAMATH_CALUDE_crayons_per_box_l2808_280834


namespace NUMINAMATH_CALUDE_roots_of_equation_l2808_280855

theorem roots_of_equation : 
  {x : ℝ | x * (x + 5)^3 * (5 - x) = 0} = {-5, 0, 5} := by
sorry

end NUMINAMATH_CALUDE_roots_of_equation_l2808_280855


namespace NUMINAMATH_CALUDE_quadratic_root_in_interval_l2808_280873

/-- A quadratic function f(x) = ax^2 + bx + c has a root in the interval (-2, 0),
    given that 2a + c/2 > b and c < 0. -/
theorem quadratic_root_in_interval (a b c : ℝ) (h1 : 2 * a + c / 2 > b) (h2 : c < 0) :
  ∃ x : ℝ, x ∈ Set.Ioo (-2 : ℝ) 0 ∧ a * x^2 + b * x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_in_interval_l2808_280873


namespace NUMINAMATH_CALUDE_original_number_proof_l2808_280808

theorem original_number_proof (x y : ℝ) : 
  10 * x + 22 * y = 780 →
  y = 30.333333333333332 →
  y > x →
  x + y = 41.6 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l2808_280808


namespace NUMINAMATH_CALUDE_max_intersections_three_circles_one_line_l2808_280833

/-- The maximum number of intersection points between three circles -/
def max_circle_intersections : ℕ := 6

/-- The maximum number of intersection points between a line and three circles -/
def max_line_circle_intersections : ℕ := 6

/-- The maximum number of intersection points between three circles and one line -/
theorem max_intersections_three_circles_one_line :
  max_circle_intersections + max_line_circle_intersections = 12 := by
  sorry

end NUMINAMATH_CALUDE_max_intersections_three_circles_one_line_l2808_280833


namespace NUMINAMATH_CALUDE_minimum_games_for_winning_percentage_l2808_280805

theorem minimum_games_for_winning_percentage (N : ℕ) : 
  (∀ k : ℕ, k < N → (3 + k : ℚ) / (4 + k) < 4/5) ∧ 
  (3 + N : ℚ) / (4 + N) ≥ 4/5 → 
  N = 1 :=
by sorry

end NUMINAMATH_CALUDE_minimum_games_for_winning_percentage_l2808_280805


namespace NUMINAMATH_CALUDE_dress_final_cost_l2808_280863

/-- The final cost of a dress after applying a discount --/
theorem dress_final_cost (original_price discount_percentage : ℚ) 
  (h1 : original_price = 50)
  (h2 : discount_percentage = 30) :
  original_price * (1 - discount_percentage / 100) = 35 := by
  sorry

#check dress_final_cost

end NUMINAMATH_CALUDE_dress_final_cost_l2808_280863


namespace NUMINAMATH_CALUDE_function_equation_implies_zero_l2808_280813

/-- A function satisfying f(x + |y|) = f(|x|) + f(y) for all real x and y is identically zero. -/
theorem function_equation_implies_zero (f : ℝ → ℝ) 
    (h : ∀ x y : ℝ, f (x + |y|) = f (|x|) + f y) : 
    ∀ x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_equation_implies_zero_l2808_280813


namespace NUMINAMATH_CALUDE_tangent_circles_radius_l2808_280897

/-- Two circles are tangent if their centers' distance equals the sum or difference of their radii -/
def are_tangent (r₁ r₂ d : ℝ) : Prop := d = r₁ + r₂ ∨ d = |r₁ - r₂|

theorem tangent_circles_radius (r₁ r₂ d : ℝ) (h₁ : r₁ = 2) (h₂ : d = 5) 
  (h₃ : are_tangent r₁ r₂ d) : r₂ = 3 ∨ r₂ = 7 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_radius_l2808_280897


namespace NUMINAMATH_CALUDE_hexagon_perimeter_l2808_280837

/-- An equilateral hexagon with specific properties -/
structure EquilateralHexagon where
  -- Side length of the hexagon
  side : ℝ
  -- Assertion that three nonadjacent interior angles are 60°
  angle_property : True
  -- The area of the hexagon is 9√3
  area_eq : side^2 * Real.sqrt 3 = 9 * Real.sqrt 3

/-- The perimeter of an equilateral hexagon is 18 given the specified conditions -/
theorem hexagon_perimeter (h : EquilateralHexagon) : h.side * 6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_perimeter_l2808_280837


namespace NUMINAMATH_CALUDE_computer_price_problem_l2808_280826

theorem computer_price_problem (sticker_price : ℝ) : 
  (0.85 * sticker_price - 50 = 0.7 * sticker_price - 20) → 
  sticker_price = 200 := by
sorry

end NUMINAMATH_CALUDE_computer_price_problem_l2808_280826


namespace NUMINAMATH_CALUDE_auction_result_l2808_280895

def auction_total (tv_initial : ℝ) (tv_increase : ℝ) (phone_initial : ℝ) (phone_increase : ℝ) 
                  (laptop_initial : ℝ) (laptop_decrease : ℝ) (auction_fee_rate : ℝ) : ℝ :=
  let tv_final := tv_initial * (1 + tv_increase)
  let phone_final := phone_initial * (1 + phone_increase)
  let laptop_final := laptop_initial * (1 - laptop_decrease)
  let total_before_fee := tv_final + phone_final + laptop_final
  let fee := total_before_fee * auction_fee_rate
  total_before_fee - fee

theorem auction_result : 
  auction_total 500 (2/5) 400 0.4 800 0.15 0.05 = 1843 := by
  sorry

end NUMINAMATH_CALUDE_auction_result_l2808_280895


namespace NUMINAMATH_CALUDE_paint_cost_decrease_l2808_280842

theorem paint_cost_decrease (canvas_original : ℝ) (paint_original : ℝ) 
  (h1 : paint_original = 4 * canvas_original)
  (h2 : canvas_original > 0)
  (h3 : paint_original > 0) :
  let canvas_new := 0.6 * canvas_original
  let total_original := paint_original + canvas_original
  let total_new := 0.4400000000000001 * total_original
  ∃ (paint_new : ℝ), paint_new = 0.4 * paint_original ∧ total_new = paint_new + canvas_new :=
by sorry

end NUMINAMATH_CALUDE_paint_cost_decrease_l2808_280842


namespace NUMINAMATH_CALUDE_candy_problem_l2808_280846

theorem candy_problem (initial_candy : ℕ) (talitha_took : ℕ) (remaining_candy : ℕ) 
  (h1 : initial_candy = 349)
  (h2 : talitha_took = 108)
  (h3 : remaining_candy = 88) :
  initial_candy - talitha_took - remaining_candy = 153 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_problem_l2808_280846


namespace NUMINAMATH_CALUDE_sales_tax_difference_example_l2808_280876

/-- The difference between two sales tax amounts -/
def sales_tax_difference (price : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  price * rate1 - price * rate2

/-- Theorem: The difference between a 7.25% sales tax and a 7% sales tax on an item priced at $50 before tax is $0.125 -/
theorem sales_tax_difference_example : 
  sales_tax_difference 50 0.0725 0.07 = 0.125 := by
sorry

end NUMINAMATH_CALUDE_sales_tax_difference_example_l2808_280876


namespace NUMINAMATH_CALUDE_infinite_prime_divisors_of_derived_set_l2808_280823

/-- A subset of natural numbers with infinite members -/
def InfiniteNatSubset (S : Set ℕ) : Prop := Set.Infinite S

/-- The set S' derived from S -/
def DerivedSet (S : Set ℕ) : Set ℕ :=
  {n | ∃ x y, x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ n = x^y + y^x}

/-- The set of prime divisors of a set of natural numbers -/
def PrimeDivisors (S : Set ℕ) : Set ℕ :=
  {p | Nat.Prime p ∧ ∃ n ∈ S, p ∣ n}

/-- Main theorem: The set of prime divisors of S' is infinite -/
theorem infinite_prime_divisors_of_derived_set (S : Set ℕ) 
  (h : InfiniteNatSubset S) : Set.Infinite (PrimeDivisors (DerivedSet S)) :=
sorry

end NUMINAMATH_CALUDE_infinite_prime_divisors_of_derived_set_l2808_280823


namespace NUMINAMATH_CALUDE_point_coordinates_l2808_280827

/-- A point in the two-dimensional plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- The fourth quadrant of the two-dimensional plane. -/
def FourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The distance between a point and the x-axis. -/
def DistanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- The distance between a point and the y-axis. -/
def DistanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem: A point P in the fourth quadrant with distance 2 to the x-axis
    and distance 3 to the y-axis has coordinates (3, -2). -/
theorem point_coordinates (P : Point)
    (h1 : FourthQuadrant P)
    (h2 : DistanceToXAxis P = 2)
    (h3 : DistanceToYAxis P = 3) :
    P = Point.mk 3 (-2) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l2808_280827
