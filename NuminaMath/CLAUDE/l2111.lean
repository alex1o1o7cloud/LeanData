import Mathlib

namespace NUMINAMATH_CALUDE_beverage_distribution_l2111_211180

/-- Represents the number of cans of beverage -/
def total_cans : ℚ := 5

/-- Represents the number of children -/
def num_children : ℚ := 8

/-- Represents each child's share of the total beverage -/
def share_of_total : ℚ := 1 / num_children

/-- Represents each child's share in terms of cans -/
def share_in_cans : ℚ := total_cans / num_children

theorem beverage_distribution :
  share_of_total = 1 / 8 ∧ share_in_cans = 5 / 8 := by sorry

end NUMINAMATH_CALUDE_beverage_distribution_l2111_211180


namespace NUMINAMATH_CALUDE_final_number_lower_bound_board_game_result_l2111_211167

/-- 
Given a positive integer n and a real number a ≥ n, 
we define a sequence of operations on a multiset of n real numbers,
initially all equal to a. In each step, we replace any two numbers
x and y in the multiset with (x+y)/4 until only one number remains.
-/
def final_number (n : ℕ+) (a : ℝ) (h : a ≥ n) : ℝ :=
  sorry

/-- 
The final number obtained after performing the operations
is always greater than or equal to a/n.
-/
theorem final_number_lower_bound (n : ℕ+) (a : ℝ) (h : a ≥ n) :
  final_number n a h ≥ a / n :=
  sorry

/--
For the specific case of 2023 numbers, each initially equal to 2023,
the final number is greater than 1.
-/
theorem board_game_result :
  final_number 2023 2023 (by norm_num) > 1 :=
  sorry

end NUMINAMATH_CALUDE_final_number_lower_bound_board_game_result_l2111_211167


namespace NUMINAMATH_CALUDE_namjoon_and_taehyung_trucks_l2111_211160

/-- The number of trucks Namjoon and Taehyung have together -/
def total_trucks (namjoon_trucks taehyung_trucks : ℕ) : ℕ :=
  namjoon_trucks + taehyung_trucks

/-- Theorem: Given Namjoon has 3 trucks and Taehyung has 2 trucks, 
    they have 5 trucks in total -/
theorem namjoon_and_taehyung_trucks : 
  total_trucks 3 2 = 5 := by sorry

end NUMINAMATH_CALUDE_namjoon_and_taehyung_trucks_l2111_211160


namespace NUMINAMATH_CALUDE_inequality_proof_l2111_211121

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^4 - y^4 = x - y) :
  (x - y) / (x^6 - y^6) ≤ (4/3) * (x + y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2111_211121


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l2111_211131

/-- Represents a trapezoid EFGH with specific properties -/
structure Trapezoid where
  EF : ℝ
  angleEGF : ℝ
  angleFHE : ℝ
  height : ℝ
  EF_length : EF = 60
  angleEGF_value : angleEGF = 45 * π / 180
  angleFHE_value : angleFHE = 45 * π / 180
  height_value : height = 30 * Real.sqrt 2

/-- The perimeter of the trapezoid EFGH is 180 + 60√2 -/
theorem trapezoid_perimeter (t : Trapezoid) : 
  ∃ (perimeter : ℝ), perimeter = 180 + 60 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_l2111_211131


namespace NUMINAMATH_CALUDE_min_value_and_max_product_l2111_211143

def f (x : ℝ) : ℝ := 2 * abs (x + 1) - abs (x - 1)

theorem min_value_and_max_product :
  (∃ k : ℝ, ∀ x : ℝ, f x ≥ k ∧ ∃ x₀ : ℝ, f x₀ = k) ∧
  (∀ a b c : ℝ, a^2 + c^2 + b^2/2 = 2 → b*(a+c) ≤ 2) ∧
  (∃ a b c : ℝ, a^2 + c^2 + b^2/2 = 2 ∧ b*(a+c) = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_max_product_l2111_211143


namespace NUMINAMATH_CALUDE_first_day_is_thursday_l2111_211101

/-- Represents days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a month with specific properties -/
structure Month where
  days : Nat
  saturdays : Nat
  sundays : Nat

/-- Function to determine the first day of the month -/
def firstDayOfMonth (m : Month) : DayOfWeek :=
  sorry

/-- Theorem stating that in a month with 31 days, 5 Saturdays, and 4 Sundays, 
    the first day is Thursday -/
theorem first_day_is_thursday :
  ∀ (m : Month), m.days = 31 → m.saturdays = 5 → m.sundays = 4 →
  firstDayOfMonth m = DayOfWeek.Thursday :=
  sorry

end NUMINAMATH_CALUDE_first_day_is_thursday_l2111_211101


namespace NUMINAMATH_CALUDE_perpendicular_parallel_imply_perpendicular_l2111_211177

-- Define the types for lines and planes
def Line : Type := ℝ → ℝ → ℝ → Prop
def Plane : Type := ℝ → ℝ → ℝ → Prop

-- Define the relations
def parallel (l₁ l₂ : Line) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular (l₁ l₂ : Line) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry

-- Define non-coincident
def non_coincident_lines (l₁ l₂ : Line) : Prop := sorry
def non_coincident_planes (p₁ p₂ : Plane) : Prop := sorry

theorem perpendicular_parallel_imply_perpendicular 
  (a b : Line) (α : Plane) 
  (h_non_coincident_lines : non_coincident_lines a b)
  (h_non_coincident_planes : non_coincident_planes α β)
  (h1 : perpendicular_line_plane a α) 
  (h2 : parallel_line_plane b α) : 
  perpendicular a b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_imply_perpendicular_l2111_211177


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binomial_200_100_l2111_211171

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Predicate to check if a number is prime -/
def isPrime (p : ℕ) : Prop := sorry

/-- The largest 2-digit prime factor of (200 choose 100) -/
def largestTwoDigitPrimeFactor : ℕ := 61

theorem largest_two_digit_prime_factor_of_binomial_200_100 :
  ∀ p : ℕ, 
    10 ≤ p → p < 100 → isPrime p → 
    p ∣ binomial 200 100 →
    p ≤ largestTwoDigitPrimeFactor ∧
    isPrime largestTwoDigitPrimeFactor ∧
    largestTwoDigitPrimeFactor ∣ binomial 200 100 := by
  sorry

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binomial_200_100_l2111_211171


namespace NUMINAMATH_CALUDE_find_multiplier_l2111_211173

theorem find_multiplier (x : ℕ) : 72514 * x = 724777430 → x = 10001 := by
  sorry

end NUMINAMATH_CALUDE_find_multiplier_l2111_211173


namespace NUMINAMATH_CALUDE_cubic_factorization_l2111_211141

theorem cubic_factorization (x : ℝ) : x^3 - 2*x^2 + x - 2 = (x^2 + 1)*(x - 2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l2111_211141


namespace NUMINAMATH_CALUDE_smallest_square_area_l2111_211156

/-- The smallest square area containing two non-overlapping rectangles -/
theorem smallest_square_area (r1_width r1_height r2_width r2_height : ℕ) 
  (h1 : r1_width = 3 ∧ r1_height = 5)
  (h2 : r2_width = 4 ∧ r2_height = 6) :
  (max (r1_width + r2_height) (r1_height + r2_width))^2 = 81 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_area_l2111_211156


namespace NUMINAMATH_CALUDE_geometric_sequence_constant_l2111_211190

/-- A geometric sequence with sum S_n = 3 · 2^n + k -/
def geometric_sequence (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) (k : ℝ) : Prop :=
  ∀ n : ℕ+, S n = 3 * 2^(n : ℝ) + k

theorem geometric_sequence_constant (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) :
  geometric_sequence a S (-3) →
  (∀ n : ℕ+, a n = S n - S (n - 1)) →
  a 1 = 3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_constant_l2111_211190


namespace NUMINAMATH_CALUDE_johns_first_second_distance_l2111_211116

/-- Represents the race scenario with John and James --/
structure RaceScenario where
  john_total_time : ℝ
  john_total_distance : ℝ
  james_top_speed_diff : ℝ
  james_initial_distance : ℝ
  james_initial_time : ℝ
  james_total_time : ℝ
  james_total_distance : ℝ

/-- Theorem stating John's distance in the first second --/
theorem johns_first_second_distance 
  (race : RaceScenario)
  (h_john_time : race.john_total_time = 13)
  (h_john_dist : race.john_total_distance = 100)
  (h_james_speed_diff : race.james_top_speed_diff = 2)
  (h_james_initial_dist : race.james_initial_distance = 10)
  (h_james_initial_time : race.james_initial_time = 2)
  (h_james_time : race.james_total_time = 11)
  (h_james_dist : race.james_total_distance = 100) :
  ∃ d : ℝ, d = 4 ∧ 
    (race.john_total_distance - d) / (race.john_total_time - 1) = 
    (race.james_total_distance - race.james_initial_distance) / (race.james_total_time - race.james_initial_time) - race.james_top_speed_diff :=
by sorry

end NUMINAMATH_CALUDE_johns_first_second_distance_l2111_211116


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2111_211133

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | -1 < x ∧ x < 2}

theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2111_211133


namespace NUMINAMATH_CALUDE_digit_57_of_21_over_22_l2111_211197

def decimal_representation (n d : ℕ) : ℕ → ℕ
  | 0 => (n * 10 / d) % 10
  | i + 1 => decimal_representation n d i

theorem digit_57_of_21_over_22 :
  decimal_representation 21 22 56 = 4 := by
  sorry

end NUMINAMATH_CALUDE_digit_57_of_21_over_22_l2111_211197


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_l2111_211154

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (-3/8, 17/8)

/-- The first line equation -/
def line1 (x y : ℚ) : Prop := y = -3 * x + 1

/-- The second line equation -/
def line2 (x y : ℚ) : Prop := y = 5 * x + 4

theorem intersection_point_is_unique :
  let (x, y) := intersection_point
  (line1 x y ∧ line2 x y) ∧
  ∀ x' y', (line1 x' y' ∧ line2 x' y') → (x' = x ∧ y' = y) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_l2111_211154


namespace NUMINAMATH_CALUDE_total_cost_of_pen_and_pencil_l2111_211187

theorem total_cost_of_pen_and_pencil (pencil_cost : ℝ) (h1 : pencil_cost = 8) :
  let pen_cost := pencil_cost / 2
  pencil_cost + pen_cost = 12 := by
sorry

end NUMINAMATH_CALUDE_total_cost_of_pen_and_pencil_l2111_211187


namespace NUMINAMATH_CALUDE_not_in_fourth_quadrant_l2111_211152

/-- A linear function defined by y = 3x + 2 -/
def linear_function (x : ℝ) : ℝ := 3 * x + 2

/-- The fourth quadrant of the Cartesian plane -/
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- Theorem stating that the linear function y = 3x + 2 does not pass through the fourth quadrant -/
theorem not_in_fourth_quadrant :
  ∀ x : ℝ, ¬(fourth_quadrant x (linear_function x)) :=
by sorry

end NUMINAMATH_CALUDE_not_in_fourth_quadrant_l2111_211152


namespace NUMINAMATH_CALUDE_shanghai_expo_2010_l2111_211137

/-- Represents a day of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Determines if a year is a leap year -/
def isLeapYear (year : Nat) : Bool :=
  year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)

/-- Calculates the number of days in a year -/
def daysInYear (year : Nat) : Nat :=
  if isLeapYear year then 366 else 365

/-- Calculates the day of the week for a given date -/
def dayOfWeek (year month day : Nat) : DayOfWeek := sorry

/-- Calculates the number of days between two dates in the same year -/
def daysBetween (year startMonth startDay endMonth endDay : Nat) : Nat := sorry

theorem shanghai_expo_2010 :
  let year := 2010
  let mayFirst := DayOfWeek.Saturday
  ¬isLeapYear year ∧
  daysInYear year = 365 ∧
  dayOfWeek year 5 31 = DayOfWeek.Monday ∧
  daysBetween year 5 1 10 31 = 184 := by sorry

end NUMINAMATH_CALUDE_shanghai_expo_2010_l2111_211137


namespace NUMINAMATH_CALUDE_profit_share_b_profit_share_b_is_1500_l2111_211136

theorem profit_share_b (capital_a capital_b capital_c : ℕ) 
  (profit_diff_ac : ℚ) (profit_share_b : ℚ) : Prop :=
  capital_a = 8000 ∧ 
  capital_b = 10000 ∧ 
  capital_c = 12000 ∧ 
  profit_diff_ac = 600 ∧
  profit_share_b = 1500 ∧
  ∃ (total_profit : ℚ),
    total_profit * (capital_b : ℚ) / (capital_a + capital_b + capital_c : ℚ) = profit_share_b ∧
    total_profit * (capital_c - capital_a : ℚ) / (capital_a + capital_b + capital_c : ℚ) = profit_diff_ac

-- Proof
theorem profit_share_b_is_1500 : 
  profit_share_b 8000 10000 12000 600 1500 := by
  sorry

end NUMINAMATH_CALUDE_profit_share_b_profit_share_b_is_1500_l2111_211136


namespace NUMINAMATH_CALUDE_parabola_vertex_on_line_l2111_211161

/-- The parabola function -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 10*x + c

/-- The x-coordinate of the vertex of the parabola -/
def vertex_x : ℝ := 5

/-- The y-coordinate of the vertex of the parabola -/
def vertex_y (c : ℝ) : ℝ := f c vertex_x

/-- The theorem stating that the value of c for which the vertex of the parabola
    y = x^2 - 10x + c lies on the line y = 3 is 28 -/
theorem parabola_vertex_on_line : ∃ c : ℝ, vertex_y c = 3 ∧ c = 28 := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_on_line_l2111_211161


namespace NUMINAMATH_CALUDE_consecutive_pairs_49_6_l2111_211165

/-- The number of ways to choose 6 elements among the first 49 positive integers
    with at least two consecutive elements -/
def consecutivePairs (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k - Nat.choose (n - k + 1) k

theorem consecutive_pairs_49_6 :
  consecutivePairs 49 6 = Nat.choose 49 6 - Nat.choose 44 6 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_pairs_49_6_l2111_211165


namespace NUMINAMATH_CALUDE_sine_of_alpha_l2111_211194

-- Define the angle α
variable (α : Real)

-- Define the point on the terminal side of α
def point : ℝ × ℝ := (3, 4)

-- Define sine function
noncomputable def sine (θ : Real) : Real :=
  point.2 / Real.sqrt (point.1^2 + point.2^2)

-- Theorem statement
theorem sine_of_alpha : sine α = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_sine_of_alpha_l2111_211194


namespace NUMINAMATH_CALUDE_least_n_divisibility_l2111_211117

theorem least_n_divisibility (a b : ℕ+) : 
  (∃ (n : ℕ+), n = 1296 ∧ 
    (∀ (a b : ℕ+), 36 ∣ (a + b) → n ∣ (a * b) → 36 ∣ a ∧ 36 ∣ b) ∧
    (∀ (m : ℕ+), m < n → 
      ∃ (x y : ℕ+), 36 ∣ (x + y) ∧ m ∣ (x * y) ∧ (¬(36 ∣ x) ∨ ¬(36 ∣ y)))) :=
by
  sorry

#check least_n_divisibility

end NUMINAMATH_CALUDE_least_n_divisibility_l2111_211117


namespace NUMINAMATH_CALUDE_pen_collection_theorem_l2111_211113

/-- Calculates the final number of pens after a series of operations --/
def final_pen_count (initial : ℕ) (mike_gives : ℕ) (cindy_multiplier : ℕ) (sharon_takes : ℕ) : ℕ :=
  ((initial + mike_gives) * cindy_multiplier) - sharon_takes

/-- Proves that the final number of pens is 31 given the specific conditions --/
theorem pen_collection_theorem :
  final_pen_count 5 20 2 19 = 31 := by
  sorry

end NUMINAMATH_CALUDE_pen_collection_theorem_l2111_211113


namespace NUMINAMATH_CALUDE_prob_sum_five_is_one_third_l2111_211139

/-- Representation of the cube faces -/
def cube_faces : Finset ℕ := {1, 2, 2, 3, 3, 3}

/-- The number of faces on the cube -/
def num_faces : ℕ := Finset.card cube_faces

/-- The set of all possible outcomes when throwing the cube twice -/
def all_outcomes : Finset (ℕ × ℕ) :=
  Finset.product cube_faces cube_faces

/-- The set of outcomes that sum to 5 -/
def sum_five_outcomes : Finset (ℕ × ℕ) :=
  all_outcomes.filter (fun p => p.1 + p.2 = 5)

/-- The probability of getting a sum of 5 when throwing the cube twice -/
def prob_sum_five : ℚ :=
  (Finset.card sum_five_outcomes : ℚ) / (Finset.card all_outcomes : ℚ)

theorem prob_sum_five_is_one_third :
  prob_sum_five = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_five_is_one_third_l2111_211139


namespace NUMINAMATH_CALUDE_smallest_n_for_four_sum_divisible_by_four_l2111_211104

theorem smallest_n_for_four_sum_divisible_by_four :
  ∃ (n : ℕ), n > 0 ∧
  (∀ (S : Finset ℤ), S.card = n →
    ∃ (a b c d : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (a + b + c + d) % 4 = 0) ∧
  (∀ (m : ℕ), m < n →
    ∃ (T : Finset ℤ), T.card = m ∧
    ∀ (a b c d : ℤ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ d ∈ T ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    (a + b + c + d) % 4 ≠ 0) ∧
  n = 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_four_sum_divisible_by_four_l2111_211104


namespace NUMINAMATH_CALUDE_camping_items_l2111_211153

theorem camping_items (total_items : ℕ) 
  (tent_stakes : ℕ) 
  (drink_mix : ℕ) 
  (water_bottles : ℕ) 
  (food_cans : ℕ) : 
  total_items = 32 → 
  drink_mix = 2 * tent_stakes → 
  water_bottles = tent_stakes + 2 → 
  food_cans * 2 = tent_stakes → 
  tent_stakes + drink_mix + water_bottles + food_cans = total_items → 
  tent_stakes = 6 := by
sorry

end NUMINAMATH_CALUDE_camping_items_l2111_211153


namespace NUMINAMATH_CALUDE_stamps_received_l2111_211145

/-- Given Simon's initial and final stamp counts, prove he received 27 stamps from friends -/
theorem stamps_received (initial_stamps final_stamps : ℕ) 
  (h1 : initial_stamps = 34)
  (h2 : final_stamps = 61) :
  final_stamps - initial_stamps = 27 := by
  sorry

end NUMINAMATH_CALUDE_stamps_received_l2111_211145


namespace NUMINAMATH_CALUDE_mabel_transactions_l2111_211162

/-- Represents the number of transactions handled by each person -/
structure Transactions where
  mabel : ℕ
  anthony : ℕ
  cal : ℕ
  jade : ℕ

/-- The conditions of the problem -/
def problem_conditions (t : Transactions) : Prop :=
  t.anthony = t.mabel + t.mabel / 10 ∧
  t.cal = (2 * t.anthony) / 3 ∧
  t.jade = t.cal + 16 ∧
  t.jade = 82

/-- The theorem to prove -/
theorem mabel_transactions :
  ∀ t : Transactions, problem_conditions t → t.mabel = 90 := by
  sorry

end NUMINAMATH_CALUDE_mabel_transactions_l2111_211162


namespace NUMINAMATH_CALUDE_debugging_time_l2111_211144

theorem debugging_time (total_hours : ℝ) (flow_chart_frac : ℝ) (coding_frac : ℝ) (meeting_frac : ℝ)
  (h1 : total_hours = 192)
  (h2 : flow_chart_frac = 3 / 10)
  (h3 : coding_frac = 3 / 8)
  (h4 : meeting_frac = 1 / 5)
  (h5 : flow_chart_frac + coding_frac + meeting_frac < 1) :
  total_hours - (flow_chart_frac + coding_frac + meeting_frac) * total_hours = 24 := by
  sorry

end NUMINAMATH_CALUDE_debugging_time_l2111_211144


namespace NUMINAMATH_CALUDE_circle_equation_l2111_211163

/-- A circle with center on y = 3x and tangent to x-axis -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  center_on_line : center.2 = 3 * center.1
  tangent_to_x_axis : center.2 = radius

/-- The line 2x + y - 10 = 0 -/
def intercepting_line (x y : ℝ) : Prop := 2 * x + y - 10 = 0

/-- The chord intercepted by the line has length 4 -/
def chord_length (c : TangentCircle) : ℝ := 4

theorem circle_equation (c : TangentCircle) 
  (h : ∃ (x y : ℝ), intercepting_line x y ∧ 
       ((x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) ∧
       ((x - c.center.1)^2 + (y - c.center.2)^2 = (chord_length c / 2)^2)) :
  ((c.center.1 = 1 ∧ c.center.2 = 3 ∧ c.radius = 3) ∨
   (c.center.1 = -6 ∧ c.center.2 = -18 ∧ c.radius = 18)) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l2111_211163


namespace NUMINAMATH_CALUDE_jellybean_probability_l2111_211199

theorem jellybean_probability : 
  let total_jellybeans : ℕ := 12
  let red_jellybeans : ℕ := 5
  let blue_jellybeans : ℕ := 3
  let white_jellybeans : ℕ := 4
  let picked_jellybeans : ℕ := 3
  
  total_jellybeans = red_jellybeans + blue_jellybeans + white_jellybeans →
  
  (Nat.choose blue_jellybeans 2 * Nat.choose white_jellybeans 1) / 
  Nat.choose total_jellybeans picked_jellybeans = 3 / 55 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_probability_l2111_211199


namespace NUMINAMATH_CALUDE_miss_stevie_payment_l2111_211166

def jerry_painting_hours : ℕ := 8
def jerry_painting_rate : ℚ := 15
def jerry_mowing_hours : ℕ := 6
def jerry_mowing_rate : ℚ := 10
def jerry_plumbing_hours : ℕ := 4
def jerry_plumbing_rate : ℚ := 18
def jerry_discount : ℚ := 0.1

def randy_painting_hours : ℕ := 7
def randy_painting_rate : ℚ := 12
def randy_mowing_hours : ℕ := 4
def randy_mowing_rate : ℚ := 8
def randy_electrical_hours : ℕ := 3
def randy_electrical_rate : ℚ := 20
def randy_discount : ℚ := 0.05

def total_payment : ℚ := 394

theorem miss_stevie_payment :
  let jerry_total := (jerry_painting_hours * jerry_painting_rate +
                      jerry_mowing_hours * jerry_mowing_rate +
                      jerry_plumbing_hours * jerry_plumbing_rate) * (1 - jerry_discount)
  let randy_total := (randy_painting_hours * randy_painting_rate +
                      randy_mowing_hours * randy_mowing_rate +
                      randy_electrical_hours * randy_electrical_rate) * (1 - randy_discount)
  jerry_total + randy_total = total_payment := by
    sorry

end NUMINAMATH_CALUDE_miss_stevie_payment_l2111_211166


namespace NUMINAMATH_CALUDE_water_tank_capacity_l2111_211103

theorem water_tank_capacity : ∀ x : ℚ, 
  (3/4 : ℚ) * x - (1/3 : ℚ) * x = 15 → x = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l2111_211103


namespace NUMINAMATH_CALUDE_chord_length_l2111_211169

-- Define the line L: 3x + 4y - 5 = 0
def L : Set (ℝ × ℝ) := {p : ℝ × ℝ | 3 * p.1 + 4 * p.2 - 5 = 0}

-- Define the circle C: x^2 + y^2 = 4
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- State the theorem
theorem chord_length : 
  A ∈ L ∧ A ∈ C ∧ B ∈ L ∧ B ∈ C → 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_l2111_211169


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2111_211172

theorem solution_set_inequality (x : ℝ) : 
  (Set.Ioo 1 2 : Set ℝ) = {x | (x - 1) * (2 - x) > 0} :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2111_211172


namespace NUMINAMATH_CALUDE_cubic_and_square_sum_l2111_211176

theorem cubic_and_square_sum (x y : ℝ) : 
  x + y = 12 → xy = 20 → (x^3 + y^3 = 1008 ∧ x^2 + y^2 = 104) := by
  sorry

end NUMINAMATH_CALUDE_cubic_and_square_sum_l2111_211176


namespace NUMINAMATH_CALUDE_equation_satisfaction_l2111_211193

theorem equation_satisfaction (a b c : ℤ) (h1 : a = c) (h2 : b + 1 = c) :
  a * (b - c) + b * (c - a) + c * (a - b) = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_satisfaction_l2111_211193


namespace NUMINAMATH_CALUDE_sin_double_angle_special_case_l2111_211129

/-- Given an angle θ in the Cartesian coordinate system with vertex at the origin,
    initial side on the positive x-axis, and terminal side on the line y = 3x,
    prove that sin 2θ = 3/5 -/
theorem sin_double_angle_special_case (θ : Real) :
  (∃ (x y : Real), y = 3 * x ∧ x > 0 ∧ y > 0 ∧ (θ = Real.arctan (y / x))) →
  Real.sin (2 * θ) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_special_case_l2111_211129


namespace NUMINAMATH_CALUDE_added_number_after_doubling_l2111_211122

theorem added_number_after_doubling (original : ℕ) (added : ℕ) : 
  original = 6 →
  3 * (2 * original + added) = 63 →
  added = 9 := by
sorry

end NUMINAMATH_CALUDE_added_number_after_doubling_l2111_211122


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l2111_211164

theorem simultaneous_equations_solution (m : ℝ) :
  ∃ (x y : ℝ), y = 3 * m * x + 4 ∧ y = (3 * m - 1) * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l2111_211164


namespace NUMINAMATH_CALUDE_average_speed_calculation_l2111_211120

-- Define the cycling parameters
def cycling_speed : ℝ := 20
def cycling_time : ℝ := 1

-- Define the walking parameters
def walking_speed : ℝ := 3
def walking_time : ℝ := 2

-- Define the total distance and time
def total_distance : ℝ := cycling_speed * cycling_time + walking_speed * walking_time
def total_time : ℝ := cycling_time + walking_time

-- Theorem statement
theorem average_speed_calculation :
  total_distance / total_time = 26 / 3 := by sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l2111_211120


namespace NUMINAMATH_CALUDE_prop_values_l2111_211128

theorem prop_values (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬(¬p ∨ q)) : 
  p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_prop_values_l2111_211128


namespace NUMINAMATH_CALUDE_tank_base_diameter_calculation_l2111_211115

/-- The volume of a cylindrical tank in cubic meters. -/
def tank_volume : ℝ := 1848

/-- The depth of the cylindrical tank in meters. -/
def tank_depth : ℝ := 12.00482999321725

/-- The diameter of the base of the cylindrical tank in meters. -/
def tank_base_diameter : ℝ := 24.838

/-- Theorem stating that the diameter of the base of a cylindrical tank with given volume and depth is approximately equal to the calculated value. -/
theorem tank_base_diameter_calculation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |2 * Real.sqrt (tank_volume / (Real.pi * tank_depth)) - tank_base_diameter| < ε :=
sorry

end NUMINAMATH_CALUDE_tank_base_diameter_calculation_l2111_211115


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l2111_211183

theorem smallest_number_of_eggs (total_containers : ℕ) (deficient_containers : ℕ) : 
  deficient_containers = 3 →
  (15 * total_containers - deficient_containers > 150) →
  (∀ n : ℕ, 15 * n - deficient_containers > 150 → n ≥ total_containers) →
  15 * total_containers - deficient_containers = 162 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l2111_211183


namespace NUMINAMATH_CALUDE_acid_mixture_concentration_exists_l2111_211191

theorem acid_mixture_concentration_exists :
  ∃! P : ℝ, ∃ a w : ℝ,
    a > 0 ∧ w > 0 ∧
    (a / (a + w + 2)) * 100 = 30 ∧
    ((a + 1) / (a + w + 3)) * 100 = 40 ∧
    (a / (a + w)) * 100 = P ∧
    (P = 50 ∨ P = 52 ∨ P = 55 ∨ P = 57 ∨ P = 60) :=
by sorry

end NUMINAMATH_CALUDE_acid_mixture_concentration_exists_l2111_211191


namespace NUMINAMATH_CALUDE_bathroom_cleaning_time_is_15_l2111_211108

/-- Represents the time spent on various tasks in minutes -/
structure TaskTimes where
  total : ℕ
  laundry : ℕ
  room : ℕ
  homework : ℕ

/-- Calculates the time spent cleaning the bathroom given the times for other tasks -/
def bathroomCleaningTime (t : TaskTimes) : ℕ :=
  t.total - (t.laundry + t.room + t.homework)

theorem bathroom_cleaning_time_is_15 (t : TaskTimes) 
  (h1 : t.total = 120)
  (h2 : t.laundry = 30)
  (h3 : t.room = 35)
  (h4 : t.homework = 40) :
  bathroomCleaningTime t = 15 := by
  sorry

#eval bathroomCleaningTime { total := 120, laundry := 30, room := 35, homework := 40 }

end NUMINAMATH_CALUDE_bathroom_cleaning_time_is_15_l2111_211108


namespace NUMINAMATH_CALUDE_survey_solution_l2111_211102

def survey_problem (mac_preference : ℕ) (no_preference : ℕ) : Prop :=
  let both_preference : ℕ := mac_preference / 3
  let total_students : ℕ := mac_preference + both_preference + no_preference
  (mac_preference = 60) ∧ (no_preference = 90) → (total_students = 170)

theorem survey_solution : survey_problem 60 90 := by
  sorry

end NUMINAMATH_CALUDE_survey_solution_l2111_211102


namespace NUMINAMATH_CALUDE_jake_initial_balloons_count_l2111_211127

/-- The number of balloons Jake brought initially to the park -/
def jake_initial_balloons : ℕ := 3

/-- The number of balloons Allan brought to the park -/
def allan_balloons : ℕ := 6

/-- The number of additional balloons Jake bought at the park -/
def jake_additional_balloons : ℕ := 4

theorem jake_initial_balloons_count :
  jake_initial_balloons = 3 ∧
  allan_balloons = 6 ∧
  jake_additional_balloons = 4 ∧
  jake_initial_balloons + jake_additional_balloons = allan_balloons + 1 :=
by sorry

end NUMINAMATH_CALUDE_jake_initial_balloons_count_l2111_211127


namespace NUMINAMATH_CALUDE_function_properties_l2111_211157

noncomputable section

def f (k a x : ℝ) : ℝ := 
  if x ≥ 0 then k * x + k * (1 - a^2) else x^2 + (a^2 - 4*a) * x + (3 - a)^2

theorem function_properties (a : ℝ) 
  (h1 : ∀ (x₁ : ℝ), x₁ ≠ 0 → ∃! (x₂ : ℝ), x₂ ≠ 0 ∧ x₂ ≠ x₁ ∧ f k a x₂ = f k a x₁) :
  ∃ k : ℝ, k = (3 - a)^2 / (1 - a^2) ∧ 0 ≤ a ∧ a < 1 :=
sorry

end NUMINAMATH_CALUDE_function_properties_l2111_211157


namespace NUMINAMATH_CALUDE_pen_ratio_is_one_l2111_211148

theorem pen_ratio_is_one (initial_pens : ℕ) (mike_pens : ℕ) (sharon_pens : ℕ) (final_pens : ℕ)
  (h1 : initial_pens = 25)
  (h2 : mike_pens = 22)
  (h3 : sharon_pens = 19)
  (h4 : final_pens = 75) :
  (final_pens + sharon_pens - (initial_pens + mike_pens)) / (initial_pens + mike_pens) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_pen_ratio_is_one_l2111_211148


namespace NUMINAMATH_CALUDE_integer_x_value_l2111_211150

theorem integer_x_value : ∀ x : ℤ,
  (3 < x ∧ x < 10) ∧
  (5 < x ∧ x < 18) ∧
  (9 > x ∧ x > -2) ∧
  (8 > x ∧ x > 0) ∧
  (x + 1 < 9) →
  x = 6 ∨ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_integer_x_value_l2111_211150


namespace NUMINAMATH_CALUDE_sum_of_two_squares_not_prime_l2111_211182

theorem sum_of_two_squares_not_prime (p a b x y : ℤ) 
  (sum1 : p = a^2 + b^2) 
  (sum2 : p = x^2 + y^2) 
  (diff_rep : (a, b) ≠ (x, y) ∧ (a, b) ≠ (y, x)) : 
  ¬ Prime p := by
sorry

end NUMINAMATH_CALUDE_sum_of_two_squares_not_prime_l2111_211182


namespace NUMINAMATH_CALUDE_factorization_x4_minus_y4_factorization_x3y_minus_2x2y2_plus_xy3_factorization_4x2_minus_4x_plus_1_factorization_4ab2_plus_1_plus_4ab_l2111_211151

-- (1)
theorem factorization_x4_minus_y4 (x y : ℝ) :
  x^4 - y^4 = (x^2 + y^2) * (x + y) * (x - y) := by sorry

-- (2)
theorem factorization_x3y_minus_2x2y2_plus_xy3 (x y : ℝ) :
  x^3*y - 2*x^2*y^2 + x*y^3 = x*y*(x - y)^2 := by sorry

-- (3)
theorem factorization_4x2_minus_4x_plus_1 (x : ℝ) :
  4*x^2 - 4*x + 1 = (2*x - 1)^2 := by sorry

-- (4)
theorem factorization_4ab2_plus_1_plus_4ab (a b : ℝ) :
  4*(a - b)^2 + 1 + 4*(a - b) = (2*a - 2*b + 1)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_x4_minus_y4_factorization_x3y_minus_2x2y2_plus_xy3_factorization_4x2_minus_4x_plus_1_factorization_4ab2_plus_1_plus_4ab_l2111_211151


namespace NUMINAMATH_CALUDE_discount_percentages_l2111_211147

theorem discount_percentages :
  ∃ (x y : ℕ), 0 < x ∧ x < 10 ∧ 0 < y ∧ y < 10 ∧
  69000 * (100 - x) * (100 - y) / 10000 = 60306 ∧
  ((x = 5 ∧ y = 8) ∨ (x = 8 ∧ y = 5)) := by
  sorry

end NUMINAMATH_CALUDE_discount_percentages_l2111_211147


namespace NUMINAMATH_CALUDE_jilin_coldest_l2111_211155

structure City where
  name : String
  temperature : Int

def beijing : City := { name := "Beijing", temperature := -5 }
def shanghai : City := { name := "Shanghai", temperature := 6 }
def shenzhen : City := { name := "Shenzhen", temperature := 19 }
def jilin : City := { name := "Jilin", temperature := -22 }

def cities : List City := [beijing, shanghai, shenzhen, jilin]

theorem jilin_coldest : 
  ∀ c ∈ cities, jilin.temperature ≤ c.temperature :=
by sorry

end NUMINAMATH_CALUDE_jilin_coldest_l2111_211155


namespace NUMINAMATH_CALUDE_grazing_area_fence_posts_l2111_211140

/-- Calculates the number of fence posts needed for a rectangular grazing area -/
def fencePostsRequired (length width postSpacing : ℕ) : ℕ :=
  let longSide := max length width
  let shortSide := min length width
  let longSidePosts := longSide / postSpacing + 1
  let shortSidePosts := (shortSide / postSpacing + 1) * 2 - 2
  longSidePosts + shortSidePosts

/-- The problem statement -/
theorem grazing_area_fence_posts :
  fencePostsRequired 70 50 10 = 18 := by
  sorry


end NUMINAMATH_CALUDE_grazing_area_fence_posts_l2111_211140


namespace NUMINAMATH_CALUDE_unique_four_digit_prime_product_l2111_211149

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem unique_four_digit_prime_product :
  ∃! n : ℕ,
    1000 ≤ n ∧ n ≤ 9999 ∧
    ∃ (p q r s : ℕ),
      is_prime p ∧ is_prime q ∧ is_prime r ∧ is_prime s ∧
      p < q ∧ q < r ∧
      n = p * q * r ∧
      p + q = r - q ∧
      p + q + r = s^2 ∧
      n = 2015 := by
  sorry

end NUMINAMATH_CALUDE_unique_four_digit_prime_product_l2111_211149


namespace NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_l2111_211178

theorem regular_polygon_with_150_degree_angles (n : ℕ) : 
  n > 2 → (∀ θ : ℝ, θ = 150 → n * θ = (n - 2) * 180) → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_l2111_211178


namespace NUMINAMATH_CALUDE_fifteen_percent_problem_l2111_211118

theorem fifteen_percent_problem (x : ℝ) (h : (15 / 100) * x = 60) : x = 400 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_percent_problem_l2111_211118


namespace NUMINAMATH_CALUDE_angle_sum_theorem_l2111_211142

theorem angle_sum_theorem (θ φ : Real) (h1 : 0 < θ ∧ θ < π/2) (h2 : 0 < φ ∧ φ < π/2)
  (h3 : Real.tan θ = 2/5) (h4 : Real.cos φ = 1/2) :
  2 * θ + φ = π/4 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_theorem_l2111_211142


namespace NUMINAMATH_CALUDE_different_size_circles_not_one_tangent_l2111_211106

-- Define a circle in a plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_positive : radius > 0

-- Define the number of common tangents between two circles
def num_common_tangents (c1 c2 : Circle) : ℕ := sorry

-- Theorem statement
theorem different_size_circles_not_one_tangent (c1 c2 : Circle) :
  c1.radius ≠ c2.radius →
  num_common_tangents c1 c2 ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_different_size_circles_not_one_tangent_l2111_211106


namespace NUMINAMATH_CALUDE_finite_consecutive_divisible_pairs_infinite_highly_divisible_multiples_l2111_211184

-- Define the number of divisors function
def d (n : ℕ) : ℕ := (Nat.divisors n).card

-- Define highly divisible property
def is_highly_divisible (n : ℕ) : Prop :=
  ∀ m : ℕ, m < n → d m < d n

-- Define consecutive highly divisible property
def consecutive_highly_divisible (m n : ℕ) : Prop :=
  is_highly_divisible m ∧ is_highly_divisible n ∧ m < n ∧
  ∀ s : ℕ, m < s → s < n → ¬is_highly_divisible s

-- Theorem for part (a)
theorem finite_consecutive_divisible_pairs :
  {p : ℕ × ℕ | consecutive_highly_divisible p.1 p.2 ∧ p.1 ∣ p.2}.Finite :=
sorry

-- Theorem for part (b)
theorem infinite_highly_divisible_multiples (p : ℕ) (hp : Nat.Prime p) :
  {r : ℕ | is_highly_divisible r ∧ is_highly_divisible (p * r)}.Infinite :=
sorry

end NUMINAMATH_CALUDE_finite_consecutive_divisible_pairs_infinite_highly_divisible_multiples_l2111_211184


namespace NUMINAMATH_CALUDE_sum_of_roots_l2111_211181

-- Define the function g
variable (g : ℝ → ℝ)

-- Define the symmetry property of g
def symmetric_about_3 (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (3 + x) = g (3 - x)

-- Define a proposition that g has exactly six distinct real roots
def has_six_distinct_roots (g : ℝ → ℝ) : Prop :=
  ∃ (a b c d e f : ℝ), (g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ g d = 0 ∧ g e = 0 ∧ g f = 0) ∧
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
     b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
     c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
     d ≠ e ∧ d ≠ f ∧
     e ≠ f) ∧
    (∀ x : ℝ, g x = 0 → (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e ∨ x = f))

-- State the theorem
theorem sum_of_roots (g : ℝ → ℝ) 
    (h1 : symmetric_about_3 g) 
    (h2 : has_six_distinct_roots g) : 
  ∃ (a b c d e f : ℝ), (g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ g d = 0 ∧ g e = 0 ∧ g f = 0) ∧
    (a + b + c + d + e + f = 18) :=
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2111_211181


namespace NUMINAMATH_CALUDE_triangle_problem_l2111_211114

open Real

noncomputable def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  a / sin A = b / sin B ∧ b / sin B = c / sin C

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : triangle_ABC a b c A B C)
  (h_c : c = 2 * b * cos B)
  (h_C : C = 2 * π / 3) :
  B = π / 6 ∧ 
  (∀ p, p = 4 + 2 * sqrt 3 → a + b + c = p → 
    ∃ m, m = sqrt 7 ∧ m^2 = (a^2 + b^2) / 4 + c^2 / 16) ∧
  (∀ S, S = 3 * sqrt 3 / 4 → (1/2) * a * b * sin C = S → 
    ∃ m, m = sqrt 21 / 2 ∧ m^2 = (a^2 + b^2) / 4 + c^2 / 16) :=
by sorry


end NUMINAMATH_CALUDE_triangle_problem_l2111_211114


namespace NUMINAMATH_CALUDE_smallest_number_with_55_divisors_l2111_211175

def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem smallest_number_with_55_divisors :
  ∃ (n : ℕ), num_divisors n = 55 ∧ 
  (∀ m : ℕ, num_divisors m = 55 → n ≤ m) ∧
  n = 3^4 * 2^10 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_55_divisors_l2111_211175


namespace NUMINAMATH_CALUDE_rice_profit_l2111_211109

/-- Calculates the profit from selling a sack of rice -/
def calculate_profit (weight : ℝ) (cost : ℝ) (price_per_kg : ℝ) : ℝ :=
  weight * price_per_kg - cost

/-- Theorem: The profit from selling a 50kg sack of rice that costs $50 at $1.20 per kg is $10 -/
theorem rice_profit : calculate_profit 50 50 1.20 = 10 := by
  sorry

end NUMINAMATH_CALUDE_rice_profit_l2111_211109


namespace NUMINAMATH_CALUDE_classmate_heights_most_suitable_l2111_211132

/-- Represents a survey option -/
inductive SurveyOption
  | LightBulbs
  | RiverWater
  | TVViewership
  | ClassmateHeights

/-- Characteristics of a survey -/
structure SurveyCharacteristics where
  population_size : ℕ
  accessibility : Bool
  non_destructive : Bool

/-- Defines what makes a survey comprehensive -/
def is_comprehensive (s : SurveyCharacteristics) : Prop :=
  s.population_size < 1000 ∧ s.accessibility ∧ s.non_destructive

/-- Assigns characteristics to each survey option -/
def survey_properties : SurveyOption → SurveyCharacteristics
  | SurveyOption.LightBulbs => ⟨100, true, false⟩
  | SurveyOption.RiverWater => ⟨10000, false, true⟩
  | SurveyOption.TVViewership => ⟨1000000, false, true⟩
  | SurveyOption.ClassmateHeights => ⟨30, true, true⟩

/-- Theorem stating that surveying classmate heights is the most suitable for a comprehensive survey -/
theorem classmate_heights_most_suitable :
  ∀ (s : SurveyOption), s ≠ SurveyOption.ClassmateHeights →
  ¬(is_comprehensive (survey_properties s)) ∧
  (is_comprehensive (survey_properties SurveyOption.ClassmateHeights)) :=
sorry


end NUMINAMATH_CALUDE_classmate_heights_most_suitable_l2111_211132


namespace NUMINAMATH_CALUDE_emery_shoe_alteration_cost_l2111_211188

theorem emery_shoe_alteration_cost :
  let num_pairs : ℕ := 17
  let cost_per_shoe : ℕ := 29
  let total_shoes : ℕ := num_pairs * 2
  let total_cost : ℕ := total_shoes * cost_per_shoe
  total_cost = 986 := by sorry

end NUMINAMATH_CALUDE_emery_shoe_alteration_cost_l2111_211188


namespace NUMINAMATH_CALUDE_a_to_m_equals_2023_l2111_211125

theorem a_to_m_equals_2023 (a m : ℝ) : 
  m = Real.sqrt (a - 2023) - Real.sqrt (2023 - a) + 1 → 
  a^m = 2023 := by
sorry

end NUMINAMATH_CALUDE_a_to_m_equals_2023_l2111_211125


namespace NUMINAMATH_CALUDE_ellipse_theorem_l2111_211195

/-- An ellipse with center at the origin, foci on the x-axis, 
    minor axis length 8√2, and eccentricity 1/3 -/
structure Ellipse where
  b : ℝ
  e : ℝ
  minor_axis : b = 4 * Real.sqrt 2
  eccentricity : e = 1/3

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 36 + y^2 / 32 = 1

/-- Theorem stating that the given ellipse satisfies the equation -/
theorem ellipse_theorem (E : Ellipse) (x y : ℝ) :
  ellipse_equation x y := by
  sorry

end NUMINAMATH_CALUDE_ellipse_theorem_l2111_211195


namespace NUMINAMATH_CALUDE_book_price_reduction_l2111_211146

theorem book_price_reduction : 
  let initial_discount : ℝ := 0.3
  let price_increase : ℝ := 0.2
  let final_discount : ℝ := 0.5
  let original_price : ℝ := 1
  let discounted_price := original_price * (1 - initial_discount)
  let increased_price := discounted_price * (1 + price_increase)
  let final_price := increased_price * (1 - final_discount)
  let total_reduction := (original_price - final_price) / original_price
  total_reduction = 0.58
:= by sorry

end NUMINAMATH_CALUDE_book_price_reduction_l2111_211146


namespace NUMINAMATH_CALUDE_not_always_prime_l2111_211170

def P (n : ℤ) : ℤ := n^2 + n + 41

theorem not_always_prime : ∃ n : ℤ, ¬(Nat.Prime (Int.natAbs (P n))) := by
  sorry

end NUMINAMATH_CALUDE_not_always_prime_l2111_211170


namespace NUMINAMATH_CALUDE_triangle_perimeter_range_l2111_211138

/-- Given a triangle ABC with sides a, b, c, where a = 1 and 2cos(C) + c = 2b,
    the perimeter p satisfies 2 < p ≤ 3 -/
theorem triangle_perimeter_range (b c : ℝ) (C : ℝ) : 
  let a : ℝ := 1
  let p := a + b + c
  2 * Real.cos C + c = 2 * b →
  2 < p ∧ p ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_range_l2111_211138


namespace NUMINAMATH_CALUDE_triangle_area_implies_angle_l2111_211100

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area S_ABC = (a^2 + b^2 - c^2) / 4, then the measure of angle C is π/4. -/
theorem triangle_area_implies_angle (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_area : (a^2 + b^2 - c^2) / 4 = (1/2) * a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b)))) :
  Real.arccos ((a^2 + b^2 - c^2) / (2*a*b)) = π/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_implies_angle_l2111_211100


namespace NUMINAMATH_CALUDE_lisa_candy_consumption_l2111_211168

/-- The number of candies Lisa has initially -/
def initial_candies : ℕ := 36

/-- The number of candies Lisa eats on Mondays and Wednesdays -/
def candies_on_mon_wed : ℕ := 2

/-- The number of candies Lisa eats on other days -/
def candies_on_other_days : ℕ := 1

/-- The number of days Lisa eats 2 candies per week -/
def days_with_two_candies : ℕ := 2

/-- The number of days Lisa eats 1 candy per week -/
def days_with_one_candy : ℕ := 5

/-- The total number of candies Lisa eats in a week -/
def candies_per_week : ℕ := 
  days_with_two_candies * candies_on_mon_wed + 
  days_with_one_candy * candies_on_other_days

/-- The number of weeks it takes for Lisa to eat all the candies -/
def weeks_to_eat_all_candies : ℕ := initial_candies / candies_per_week

theorem lisa_candy_consumption : weeks_to_eat_all_candies = 4 := by
  sorry

end NUMINAMATH_CALUDE_lisa_candy_consumption_l2111_211168


namespace NUMINAMATH_CALUDE_squirrels_in_tree_l2111_211186

theorem squirrels_in_tree (nuts : ℕ) (squirrels : ℕ) : 
  nuts = 2 → squirrels = nuts + 2 → squirrels = 4 := by
  sorry

end NUMINAMATH_CALUDE_squirrels_in_tree_l2111_211186


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2111_211107

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The statement of the problem -/
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 ^ 2 + 6 * a 2 + 2 = 0 →
  a 16 ^ 2 + 6 * a 16 + 2 = 0 →
  (a 2 * a 16 / a 9 = Real.sqrt 2 ∨ a 2 * a 16 / a 9 = -Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2111_211107


namespace NUMINAMATH_CALUDE_connie_marbles_theorem_l2111_211179

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℕ := 183

/-- The number of marbles Connie has left -/
def marbles_left : ℕ := 593

/-- The initial number of marbles Connie had -/
def initial_marbles : ℕ := marbles_given + marbles_left

theorem connie_marbles_theorem : initial_marbles = 776 := by sorry

end NUMINAMATH_CALUDE_connie_marbles_theorem_l2111_211179


namespace NUMINAMATH_CALUDE_probability_no_shaded_l2111_211192

/-- Represents a rectangle in the 2 by 1001 grid --/
structure Rectangle where
  left : Nat
  right : Nat
  top : Nat
  bottom : Nat

/-- The total number of possible rectangles in the grid --/
def total_rectangles : Nat := 501501

/-- The number of rectangles containing at least one shaded square --/
def shaded_rectangles : Nat := 252002

/-- Checks if a rectangle contains a shaded square --/
def contains_shaded (r : Rectangle) : Prop :=
  (r.left = 1 ∧ r.right ≥ 1) ∨ 
  (r.left ≤ 501 ∧ r.right ≥ 501) ∨ 
  (r.left ≤ 1001 ∧ r.right = 1001)

/-- The main theorem stating the probability of choosing a rectangle without a shaded square --/
theorem probability_no_shaded : 
  (total_rectangles - shaded_rectangles) / total_rectangles = 249499 / 501501 := by
  sorry

end NUMINAMATH_CALUDE_probability_no_shaded_l2111_211192


namespace NUMINAMATH_CALUDE_unsold_bars_l2111_211159

theorem unsold_bars (total_bars : ℕ) (price_per_bar : ℕ) (total_sold : ℕ) :
  total_bars = 13 →
  price_per_bar = 6 →
  total_sold = 42 →
  total_bars - (total_sold / price_per_bar) = 6 := by
sorry

end NUMINAMATH_CALUDE_unsold_bars_l2111_211159


namespace NUMINAMATH_CALUDE_b_is_composite_greatest_number_of_factors_l2111_211158

/-- The greatest number of positive factors for b^m -/
def max_factors : ℕ := 81

/-- b is a positive integer less than or equal to 20 -/
def b : ℕ := 16

/-- m is a positive integer less than or equal to 20 -/
def m : ℕ := 20

/-- b is composite -/
theorem b_is_composite : ¬ Nat.Prime b := by sorry

theorem greatest_number_of_factors :
  ∀ b' m' : ℕ,
  b' ≤ 20 → m' ≤ 20 → b' > 1 → ¬ Nat.Prime b' →
  (Nat.divisors (b' ^ m')).card ≤ max_factors := by sorry

end NUMINAMATH_CALUDE_b_is_composite_greatest_number_of_factors_l2111_211158


namespace NUMINAMATH_CALUDE_water_park_admission_l2111_211185

/-- The admission charge for a child in a water park. -/
def child_admission : ℚ :=
  3⁻¹ * (13 / 4 - 1)

/-- The total amount paid by an adult. -/
def total_paid : ℚ := 13 / 4

/-- The number of children accompanying the adult. -/
def num_children : ℕ := 3

/-- The admission charge for an adult. -/
def adult_admission : ℚ := 1

theorem water_park_admission :
  child_admission * num_children + adult_admission = total_paid :=
sorry

end NUMINAMATH_CALUDE_water_park_admission_l2111_211185


namespace NUMINAMATH_CALUDE_at_least_two_heads_probability_l2111_211123

def coin_toss_probability : ℕ → ℕ → ℚ
  | n, k => (Nat.choose n k : ℚ) * (1/2)^k * (1/2)^(n-k)

theorem at_least_two_heads_probability :
  coin_toss_probability 4 2 + coin_toss_probability 4 3 + coin_toss_probability 4 4 = 11/16 := by
  sorry

end NUMINAMATH_CALUDE_at_least_two_heads_probability_l2111_211123


namespace NUMINAMATH_CALUDE_daily_production_is_2170_l2111_211198

/-- The number of toys produced per week -/
def weekly_production : ℕ := 4340

/-- The number of working days per week -/
def working_days : ℕ := 2

/-- The number of toys produced each day -/
def daily_production : ℕ := weekly_production / working_days

/-- Theorem stating that the daily production is 2170 toys -/
theorem daily_production_is_2170 : daily_production = 2170 := by
  sorry

end NUMINAMATH_CALUDE_daily_production_is_2170_l2111_211198


namespace NUMINAMATH_CALUDE_acute_triangle_cosine_inequality_l2111_211174

theorem acute_triangle_cosine_inequality (A B C : ℝ) 
  (h_acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) :
  (Real.cos A / Real.cos (B - C)) + 
  (Real.cos B / Real.cos (C - A)) + 
  (Real.cos C / Real.cos (A - B)) ≥ 3/2 := by sorry

end NUMINAMATH_CALUDE_acute_triangle_cosine_inequality_l2111_211174


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2111_211119

theorem polynomial_simplification (x : ℝ) :
  (3 * x^2 + 4 * x + 8) * (2 * x + 1) - (2 * x + 1) * (x^2 + 5 * x - 72) + (4 * x - 15) * (2 * x + 1) * (x + 6) =
  12 * x^3 + 22 * x^2 - 12 * x - 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2111_211119


namespace NUMINAMATH_CALUDE_log_sum_equality_l2111_211196

-- Define the problem
theorem log_sum_equality : Real.log 50 + Real.log 20 + Real.log 4 = 3.60206 := by
  sorry

#check log_sum_equality

end NUMINAMATH_CALUDE_log_sum_equality_l2111_211196


namespace NUMINAMATH_CALUDE_toy_cost_l2111_211189

/-- The cost of each toy given Paul's savings and allowance -/
theorem toy_cost (initial_savings : ℕ) (allowance : ℕ) (num_toys : ℕ) 
  (h1 : initial_savings = 3)
  (h2 : allowance = 7)
  (h3 : num_toys = 2)
  (h4 : num_toys > 0) :
  (initial_savings + allowance) / num_toys = 5 := by
  sorry


end NUMINAMATH_CALUDE_toy_cost_l2111_211189


namespace NUMINAMATH_CALUDE_initial_worksheets_l2111_211110

theorem initial_worksheets (graded : ℕ) (new_worksheets : ℕ) (total : ℕ) :
  graded = 7 → new_worksheets = 36 → total = 63 →
  ∃ initial : ℕ, initial - graded + new_worksheets = total ∧ initial = 34 :=
by sorry

end NUMINAMATH_CALUDE_initial_worksheets_l2111_211110


namespace NUMINAMATH_CALUDE_sin_double_angle_plus_pi_sixth_l2111_211126

theorem sin_double_angle_plus_pi_sixth (α : Real) 
  (h : Real.sin (α - π/6) = 1/3) : 
  Real.sin (2*α + π/6) = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_plus_pi_sixth_l2111_211126


namespace NUMINAMATH_CALUDE_golden_ratio_properties_l2111_211134

theorem golden_ratio_properties :
  let a : ℝ := (Real.sqrt 5 + 1) / 2
  let b : ℝ := (Real.sqrt 5 - 1) / 2
  (b / a + a / b = 3) ∧ (a^2 + b^2 + a*b = 4) := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_properties_l2111_211134


namespace NUMINAMATH_CALUDE_rectangle_shading_theorem_l2111_211124

theorem rectangle_shading_theorem :
  let r : ℝ := 1/4
  let series_sum : ℝ := r / (1 - r)
  series_sum = 1/3 := by sorry

end NUMINAMATH_CALUDE_rectangle_shading_theorem_l2111_211124


namespace NUMINAMATH_CALUDE_consecutive_integers_square_sum_l2111_211130

theorem consecutive_integers_square_sum (x : ℤ) : 
  x^2 + (x+1)^2 + x^2 * (x+1)^2 = (x^2 + x + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_sum_l2111_211130


namespace NUMINAMATH_CALUDE_arithmetic_progression_ratio_l2111_211111

theorem arithmetic_progression_ratio (a d : ℝ) : 
  (15 * a + 105 * d = 4 * (8 * a + 28 * d)) → (a / d = -7 / 17) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_ratio_l2111_211111


namespace NUMINAMATH_CALUDE_fraction_equality_l2111_211105

theorem fraction_equality (a : ℕ+) : 
  (a : ℚ) / ((a : ℚ) + 35) = 865 / 1000 → a = 225 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2111_211105


namespace NUMINAMATH_CALUDE_cube_shadow_problem_l2111_211112

/-- Given a cube with edge length 2 cm and a light source x cm above one upper vertex,
    if the shadow area (excluding the area beneath the cube) is 192 cm²,
    then the greatest integer not exceeding 1000x is 25780. -/
theorem cube_shadow_problem (x : ℝ) : 
  let cube_edge : ℝ := 2
  let shadow_area : ℝ := 192
  let total_shadow_area : ℝ := shadow_area + cube_edge^2
  let shadow_side : ℝ := (total_shadow_area).sqrt
  x = (shadow_side - cube_edge) / 2 →
  ⌊1000 * x⌋ = 25780 := by sorry

end NUMINAMATH_CALUDE_cube_shadow_problem_l2111_211112


namespace NUMINAMATH_CALUDE_lateral_surface_area_regular_triangular_prism_l2111_211135

/-- Given a regular triangular prism with height h, where a line passing through 
    the center of the upper base and the midpoint of the side of the lower base 
    is inclined at an angle 60° to the plane of the base, 
    the lateral surface area of the prism is 6h². -/
theorem lateral_surface_area_regular_triangular_prism 
  (h : ℝ) 
  (h_pos : h > 0) 
  (incline_angle : ℝ) 
  (incline_angle_eq : incline_angle = 60 * π / 180) : 
  ∃ (S : ℝ), S = 6 * h^2 ∧ S > 0 := by
  sorry

end NUMINAMATH_CALUDE_lateral_surface_area_regular_triangular_prism_l2111_211135
