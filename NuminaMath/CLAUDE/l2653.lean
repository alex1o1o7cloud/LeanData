import Mathlib

namespace NUMINAMATH_CALUDE_systematic_sampling_correspondence_l2653_265355

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  total_students : Nat
  num_groups : Nat
  students_per_group : Nat
  selected_student : Nat
  selected_group : Nat

/-- Theorem stating the relationship between selected students in different groups. -/
theorem systematic_sampling_correspondence
  (s : SystematicSampling)
  (h1 : s.total_students = 60)
  (h2 : s.num_groups = 5)
  (h3 : s.students_per_group = s.total_students / s.num_groups)
  (h4 : s.selected_student = 16)
  (h5 : s.selected_group = 2)
  : (s.selected_student - (s.selected_group - 1) * s.students_per_group) + 3 * s.students_per_group = 40 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_correspondence_l2653_265355


namespace NUMINAMATH_CALUDE_blocks_used_for_first_building_l2653_265373

/-- Given the number of building blocks Jesse started with, used for farmhouse and fenced-in area, and left at the end, 
    calculate the number of blocks used for the first building. -/
theorem blocks_used_for_first_building 
  (total_blocks : ℕ) 
  (farmhouse_blocks : ℕ) 
  (fenced_area_blocks : ℕ) 
  (blocks_left : ℕ) 
  (h1 : total_blocks = 344) 
  (h2 : farmhouse_blocks = 123) 
  (h3 : fenced_area_blocks = 57) 
  (h4 : blocks_left = 84) :
  total_blocks - farmhouse_blocks - fenced_area_blocks - blocks_left = 80 :=
by sorry

end NUMINAMATH_CALUDE_blocks_used_for_first_building_l2653_265373


namespace NUMINAMATH_CALUDE_max_value_xyz_expression_l2653_265327

theorem max_value_xyz_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x * y * z * (x + y + z) / ((x + y)^2 * (y + z)^2) ≤ 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_xyz_expression_l2653_265327


namespace NUMINAMATH_CALUDE_no_prime_solution_to_equation_l2653_265352

theorem no_prime_solution_to_equation :
  ∀ p q r s t : ℕ, 
    Prime p → Prime q → Prime r → Prime s → Prime t →
    p^2 + q^2 ≠ r^2 + s^2 + t^2 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_solution_to_equation_l2653_265352


namespace NUMINAMATH_CALUDE_candy_bar_consumption_l2653_265354

theorem candy_bar_consumption (calories_per_bar : ℕ) (total_calories : ℕ) (num_bars : ℕ) : 
  calories_per_bar = 8 → total_calories = 24 → num_bars = total_calories / calories_per_bar → num_bars = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_bar_consumption_l2653_265354


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_one_forty_satisfies_conditions_one_forty_is_greatest_l2653_265358

theorem greatest_integer_with_gcd_six (n : ℕ) : n < 150 ∧ n.gcd 18 = 6 → n ≤ 140 :=
by sorry

theorem one_forty_satisfies_conditions : 140 < 150 ∧ Nat.gcd 140 18 = 6 :=
by sorry

theorem one_forty_is_greatest : 
  ∀ m : ℕ, m < 150 ∧ m.gcd 18 = 6 → m ≤ 140 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_one_forty_satisfies_conditions_one_forty_is_greatest_l2653_265358


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2653_265340

theorem quadratic_equation_roots (a : ℝ) : 
  ((a + 1) * (-1)^2 + (-1) - 1 = 0) → 
  (a = 1 ∧ ∃ x : ℝ, x ≠ -1 ∧ (2 * x^2 + x - 1 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2653_265340


namespace NUMINAMATH_CALUDE_fraction_subtraction_l2653_265303

theorem fraction_subtraction (d : ℝ) : (6 - 5 * d) / 9 - 3 = (-21 - 5 * d) / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l2653_265303


namespace NUMINAMATH_CALUDE_arithmetic_sequence_equivalence_l2653_265312

theorem arithmetic_sequence_equivalence
  (a b c : ℕ → ℝ)
  (h1 : ∀ n, b n = a (n + 1) - a n)
  (h2 : ∀ n, c n = a n + 2 * a (n + 1)) :
  (∃ d, ∀ n, a (n + 1) - a n = d) ↔
  ((∃ D, ∀ n, c (n + 1) - c n = D) ∧ (∀ n, b n ≤ b (n + 1))) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_equivalence_l2653_265312


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l2653_265375

theorem right_triangle_perimeter (a b c r : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 →
  c = 10 → r = 1 →
  a^2 + b^2 = c^2 →
  (a + b - c) * r = a * b / 2 →
  a + b + c = 24 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l2653_265375


namespace NUMINAMATH_CALUDE_only_234_not_right_triangle_l2653_265316

/-- A function that checks if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- Theorem stating that among the given sets, only (2, 3, 4) is not a right triangle --/
theorem only_234_not_right_triangle :
  ¬(is_right_triangle 2 3 4) ∧
  (is_right_triangle 1 1 (Real.sqrt 2)) ∧
  (is_right_triangle (Real.sqrt 2) (Real.sqrt 3) (Real.sqrt 5)) ∧
  (is_right_triangle 3 4 5) :=
by sorry


end NUMINAMATH_CALUDE_only_234_not_right_triangle_l2653_265316


namespace NUMINAMATH_CALUDE_h_is_even_l2653_265304

-- Define g as an odd function
def g : ℝ → ℝ := sorry

-- Axiom stating that g is an odd function
axiom g_odd : ∀ x : ℝ, g (-x) = -g x

-- Define h using g
def h (x : ℝ) : ℝ := |g (x^4)|

-- Theorem stating that h is an even function
theorem h_is_even : ∀ x : ℝ, h (-x) = h x := by
  sorry

end NUMINAMATH_CALUDE_h_is_even_l2653_265304


namespace NUMINAMATH_CALUDE_luncheon_invitees_l2653_265363

/-- The number of people who didn't show up -/
def no_shows : ℕ := 7

/-- The number of people each table can hold -/
def people_per_table : ℕ := 5

/-- The number of tables needed -/
def tables_needed : ℕ := 8

/-- The original number of invited people -/
def original_invitees : ℕ := (tables_needed * people_per_table) + no_shows

theorem luncheon_invitees : original_invitees = 47 := by
  sorry

end NUMINAMATH_CALUDE_luncheon_invitees_l2653_265363


namespace NUMINAMATH_CALUDE_jane_test_probability_l2653_265336

theorem jane_test_probability (pass_prob : ℚ) (h : pass_prob = 4/7) :
  1 - pass_prob = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_jane_test_probability_l2653_265336


namespace NUMINAMATH_CALUDE_expression_evaluation_l2653_265306

theorem expression_evaluation : 
  8^(1/4) * 42 + (32 * Real.sqrt 3)^6 + Real.log 2 / Real.log 3 * (Real.log (Real.log 27 / Real.log 3) / Real.log 2) = 111 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2653_265306


namespace NUMINAMATH_CALUDE_marco_score_percentage_l2653_265393

/-- Proves that Marco scored 10% less than the average test score -/
theorem marco_score_percentage (average_score : ℝ) (margaret_score : ℝ) (marco_score : ℝ) :
  average_score = 90 →
  margaret_score = 86 →
  margaret_score = marco_score + 5 →
  (average_score - marco_score) / average_score = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_marco_score_percentage_l2653_265393


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l2653_265395

theorem cubic_sum_minus_product (x y z : ℝ) 
  (h1 : x + y + z = 10) 
  (h2 : x*y + x*z + y*z = 30) : 
  x^3 + y^3 + z^3 - 3*x*y*z = 100 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l2653_265395


namespace NUMINAMATH_CALUDE_island_marriage_proportion_l2653_265334

theorem island_marriage_proportion (men women : ℕ) (h1 : 2 * men = 3 * women) :
  (2 * men + 2 * women : ℚ) / (3 * men + 5 * women : ℚ) = 12 / 19 := by
  sorry

end NUMINAMATH_CALUDE_island_marriage_proportion_l2653_265334


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2653_265396

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (5 * x + 9) = 12 → x = 27 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2653_265396


namespace NUMINAMATH_CALUDE_inequality_range_l2653_265328

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x > 1 → x + 1 / (x - 1) ≥ a) → a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l2653_265328


namespace NUMINAMATH_CALUDE_total_dolls_l2653_265333

theorem total_dolls (jazmin_dolls geraldine_dolls : ℕ) 
  (h1 : jazmin_dolls = 1209) 
  (h2 : geraldine_dolls = 2186) : 
  jazmin_dolls + geraldine_dolls = 3395 := by
  sorry

end NUMINAMATH_CALUDE_total_dolls_l2653_265333


namespace NUMINAMATH_CALUDE_intersection_of_intervals_solution_interval_l2653_265385

theorem intersection_of_intervals : Set.Ioo (1/2 : ℝ) (3/5 : ℝ) = 
  Set.inter 
    (Set.Ioo (1/2 : ℝ) (3/4 : ℝ)) 
    (Set.Ioo (2/5 : ℝ) (3/5 : ℝ)) := by sorry

theorem solution_interval (x : ℝ) : 
  (2 < 4 * x ∧ 4 * x < 3) ∧ (2 < 5 * x ∧ 5 * x < 3) ↔ 
  x ∈ Set.Ioo (1/2 : ℝ) (3/5 : ℝ) := by sorry

end NUMINAMATH_CALUDE_intersection_of_intervals_solution_interval_l2653_265385


namespace NUMINAMATH_CALUDE_rectangular_plot_area_l2653_265357

theorem rectangular_plot_area 
  (L B : ℝ) 
  (h_ratio : L / B = 7 / 5) 
  (h_perimeter : 2 * (L + B) = 288) : 
  L * B = 5040 := by sorry

end NUMINAMATH_CALUDE_rectangular_plot_area_l2653_265357


namespace NUMINAMATH_CALUDE_cubic_factorization_l2653_265345

theorem cubic_factorization (x : ℝ) : x^3 - 6*x^2 + 9*x = x*(x-3)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l2653_265345


namespace NUMINAMATH_CALUDE_find_a_l2653_265394

theorem find_a : ∃ a : ℝ, (2 * 1 - a * (-1) = 3) ∧ a = 1 := by sorry

end NUMINAMATH_CALUDE_find_a_l2653_265394


namespace NUMINAMATH_CALUDE_smallest_gcd_of_multiples_l2653_265308

theorem smallest_gcd_of_multiples (a b : ℕ+) (h : Nat.gcd a b = 18) :
  (Nat.gcd (12 * a) (20 * b)).min = 72 := by
  sorry

end NUMINAMATH_CALUDE_smallest_gcd_of_multiples_l2653_265308


namespace NUMINAMATH_CALUDE_rice_seedling_stats_l2653_265317

def dataset : List Nat := [25, 26, 27, 26, 27, 28, 29, 26, 29]

def mode (l : List Nat) : Nat := sorry

def median (l : List Nat) : Nat := sorry

theorem rice_seedling_stats :
  mode dataset = 26 ∧ median dataset = 27 := by sorry

end NUMINAMATH_CALUDE_rice_seedling_stats_l2653_265317


namespace NUMINAMATH_CALUDE_gcd_78_36_l2653_265367

theorem gcd_78_36 : Nat.gcd 78 36 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_78_36_l2653_265367


namespace NUMINAMATH_CALUDE_tic_tac_toe_winning_probability_l2653_265379

/-- A tic-tac-toe board is a 3x3 grid. -/
def TicTacToeBoard := Fin 3 → Fin 3 → Bool

/-- A winning position is a line (row, column, or diagonal) on the board. -/
def WinningPosition : Type := List (Fin 3 × Fin 3)

/-- The set of all winning positions on a tic-tac-toe board. -/
def allWinningPositions : List WinningPosition :=
  -- 3 horizontal lines
  [[(0,0), (0,1), (0,2)], [(1,0), (1,1), (1,2)], [(2,0), (2,1), (2,2)]] ++
  -- 3 vertical lines
  [[(0,0), (1,0), (2,0)], [(0,1), (1,1), (2,1)], [(0,2), (1,2), (2,2)]] ++
  -- 2 diagonal lines
  [[(0,0), (1,1), (2,2)], [(0,2), (1,1), (2,0)]]

/-- The number of ways to arrange 3 noughts on a 3x3 board. -/
def totalArrangements : ℕ := 84

/-- The probability of three noughts being in a winning position. -/
def winningProbability : ℚ := 2 / 21

theorem tic_tac_toe_winning_probability :
  (List.length allWinningPositions : ℚ) / totalArrangements = winningProbability := by
  sorry

end NUMINAMATH_CALUDE_tic_tac_toe_winning_probability_l2653_265379


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2653_265351

theorem sqrt_meaningful_range (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 2) → x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2653_265351


namespace NUMINAMATH_CALUDE_first_quadrant_iff_sin_cos_sum_gt_one_l2653_265325

theorem first_quadrant_iff_sin_cos_sum_gt_one (α : Real) :
  (0 < α ∧ α < Real.pi / 2) ↔ (Real.sin α + Real.cos α > 1) := by
  sorry

end NUMINAMATH_CALUDE_first_quadrant_iff_sin_cos_sum_gt_one_l2653_265325


namespace NUMINAMATH_CALUDE_a_range_l2653_265356

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x, a^x > 1 ↔ x < 0

def q (a : ℝ) : Prop := ∀ x, x^2 - x + a > 0

-- Define the range of a
def range_of_a (a : ℝ) : Prop := (0 < a ∧ a ≤ 1/4) ∨ (a > 1)

-- Theorem statement
theorem a_range (a : ℝ) : 
  ((p a ∨ q a) ∧ ¬(p a ∧ q a)) → range_of_a a :=
sorry

end NUMINAMATH_CALUDE_a_range_l2653_265356


namespace NUMINAMATH_CALUDE_chord_count_l2653_265300

/-- The number of different chords that can be drawn by connecting any two of ten points 
    on the circumference of a circle, where four of these points form a square. -/
def num_chords : ℕ := 45

/-- The total number of points on the circumference of the circle. -/
def total_points : ℕ := 10

/-- The number of points that form a square. -/
def square_points : ℕ := 4

theorem chord_count : 
  num_chords = (total_points * (total_points - 1)) / 2 :=
sorry

end NUMINAMATH_CALUDE_chord_count_l2653_265300


namespace NUMINAMATH_CALUDE_sara_marbles_l2653_265342

def marble_problem (initial : ℕ) (given : ℕ) (lost : ℕ) (traded : ℕ) : Prop :=
  initial + given - lost - traded = 5

theorem sara_marbles : marble_problem 10 5 7 3 := by
  sorry

end NUMINAMATH_CALUDE_sara_marbles_l2653_265342


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l2653_265386

/-- Given a line segment with midpoint (3, 4) and one endpoint (7, 10), 
    prove that the other endpoint is (-1, -2). -/
theorem line_segment_endpoint (M A B : ℝ × ℝ) : 
  M = (3, 4) → A = (7, 10) → M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → B = (-1, -2) := by
  sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l2653_265386


namespace NUMINAMATH_CALUDE_first_three_squares_s_3_equals_149_l2653_265326

/-- s(n) is the n-digit number formed by attaching the first n perfect squares in order -/
def s (n : ℕ) : ℕ :=
  sorry

/-- The first three perfect squares are 1, 4, and 9 -/
theorem first_three_squares : List ℕ := [1, 4, 9]

/-- s(3) is equal to 149 -/
theorem s_3_equals_149 : s 3 = 149 := by
  sorry

end NUMINAMATH_CALUDE_first_three_squares_s_3_equals_149_l2653_265326


namespace NUMINAMATH_CALUDE_min_squares_to_exceed_1000_l2653_265347

/-- Represents the squaring operation on a calculator --/
def square (n : ℕ) : ℕ := n * n

/-- Applies the squaring operation n times to the initial value --/
def repeated_square (initial : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => initial
  | n + 1 => square (repeated_square initial n)

/-- The theorem to be proved --/
theorem min_squares_to_exceed_1000 :
  (∀ k < 3, repeated_square 3 k ≤ 1000) ∧
  repeated_square 3 3 > 1000 :=
sorry

end NUMINAMATH_CALUDE_min_squares_to_exceed_1000_l2653_265347


namespace NUMINAMATH_CALUDE_non_monotonic_quadratic_l2653_265387

/-- A function f is not monotonic on an interval [a, b] if there exists
    x, y in [a, b] such that x < y and f(x) > f(y) -/
def NotMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y, a ≤ x ∧ x < y ∧ y ≤ b ∧ f x > f y

/-- The quadratic function f(x) = 4x^2 - kx - 8 -/
def f (k : ℝ) (x : ℝ) : ℝ := 4 * x^2 - k * x - 8

theorem non_monotonic_quadratic (k : ℝ) :
  NotMonotonic (f k) 5 8 ↔ k ∈ Set.Ioo 40 64 := by
  sorry

end NUMINAMATH_CALUDE_non_monotonic_quadratic_l2653_265387


namespace NUMINAMATH_CALUDE_success_arrangements_l2653_265376

def word_length : ℕ := 7
def s_count : ℕ := 3
def c_count : ℕ := 2
def u_count : ℕ := 1
def e_count : ℕ := 1

theorem success_arrangements : 
  (word_length.factorial) / (s_count.factorial * c_count.factorial * u_count.factorial * e_count.factorial) = 420 :=
sorry

end NUMINAMATH_CALUDE_success_arrangements_l2653_265376


namespace NUMINAMATH_CALUDE_constant_term_expansion_l2653_265388

def constant_term (n : ℕ) : ℕ :=
  Nat.choose n 0 + 
  Nat.choose n 2 * Nat.choose 2 1 + 
  Nat.choose n 4 * Nat.choose 4 2 + 
  Nat.choose n 6 * Nat.choose 6 3

theorem constant_term_expansion : constant_term 6 = 141 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l2653_265388


namespace NUMINAMATH_CALUDE_max_cubes_in_box_l2653_265341

/-- The maximum number of cubes that can fit in a rectangular box -/
theorem max_cubes_in_box (box_length box_width box_height cube_volume : ℕ) :
  box_length = 8 →
  box_width = 9 →
  box_height = 12 →
  cube_volume = 27 →
  (box_length * box_width * box_height) / cube_volume = 32 := by
  sorry

#check max_cubes_in_box

end NUMINAMATH_CALUDE_max_cubes_in_box_l2653_265341


namespace NUMINAMATH_CALUDE_hyperbolic_and_linear_functions_l2653_265353

/-- The hyperbolic and linear functions with their properties -/
theorem hyperbolic_and_linear_functions (k : ℝ) (h : |k| < 1) :
  (∀ x y : ℝ, y = (k - 1) / x → (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)) ∧
  (k * (-1 + 1) = 0) := by
  sorry

end NUMINAMATH_CALUDE_hyperbolic_and_linear_functions_l2653_265353


namespace NUMINAMATH_CALUDE_different_colors_probability_l2653_265377

structure Box where
  red : ℕ
  black : ℕ
  white : ℕ
  yellow : ℕ

def boxA : Box := { red := 3, black := 3, white := 3, yellow := 0 }
def boxB : Box := { red := 0, black := 2, white := 2, yellow := 2 }

def totalBalls (box : Box) : ℕ := box.red + box.black + box.white + box.yellow

def probabilityDifferentColors (boxA boxB : Box) : ℚ :=
  let totalA := totalBalls boxA
  let totalB := totalBalls boxB
  let sameColor := boxA.black * boxB.black + boxA.white * boxB.white
  (totalA * totalB - sameColor) / (totalA * totalB)

theorem different_colors_probability :
  probabilityDifferentColors boxA boxB = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_different_colors_probability_l2653_265377


namespace NUMINAMATH_CALUDE_triangle_cosine_rule_l2653_265364

/-- Given a triangle ABC where 6 sin A = 4 sin B = 3 sin C, prove that cos C = -1/4 -/
theorem triangle_cosine_rule (A B C : ℝ) (h : 6 * Real.sin A = 4 * Real.sin B ∧ 4 * Real.sin B = 3 * Real.sin C) :
  Real.cos C = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_rule_l2653_265364


namespace NUMINAMATH_CALUDE_monkey_climb_time_l2653_265313

/-- A monkey climbing a tree problem -/
theorem monkey_climb_time (tree_height : ℕ) (hop_distance : ℕ) (slip_distance : ℕ) 
  (h1 : tree_height = 20)
  (h2 : hop_distance = 3)
  (h3 : slip_distance = 2) :
  ∃ (time : ℕ), time = 17 ∧ 
  time * (hop_distance - slip_distance) + hop_distance ≥ tree_height :=
by
  sorry

end NUMINAMATH_CALUDE_monkey_climb_time_l2653_265313


namespace NUMINAMATH_CALUDE_ben_votes_l2653_265320

/-- Given a total of 60 votes and a ratio of 2:3 between Ben's and Matt's votes,
    prove that Ben received 24 votes. -/
theorem ben_votes (total_votes : ℕ) (ben_votes : ℕ) (matt_votes : ℕ) :
  total_votes = 60 →
  ben_votes + matt_votes = total_votes →
  3 * ben_votes = 2 * matt_votes →
  ben_votes = 24 := by
sorry

end NUMINAMATH_CALUDE_ben_votes_l2653_265320


namespace NUMINAMATH_CALUDE_multiple_of_10_average_l2653_265344

theorem multiple_of_10_average (N : ℕ) : 
  N % 10 = 0 → -- N is a multiple of 10
  (10 + N) / 2 = 305 → -- The average of multiples of 10 from 10 to N inclusive is 305
  N = 600 := by
sorry

end NUMINAMATH_CALUDE_multiple_of_10_average_l2653_265344


namespace NUMINAMATH_CALUDE_minimum_m_value_l2653_265310

theorem minimum_m_value (a x : ℝ) (ha : |a| ≤ 1) (hx : |x| ≤ 1) :
  ∃ m : ℝ, (∀ a x, |a| ≤ 1 → |x| ≤ 1 → |x^2 - a*x - a^2| ≤ m) ∧ 
  (∀ m' : ℝ, m' < m → ∃ a x, |a| ≤ 1 ∧ |x| ≤ 1 ∧ |x^2 - a*x - a^2| > m') ∧
  m = 5/4 :=
sorry

end NUMINAMATH_CALUDE_minimum_m_value_l2653_265310


namespace NUMINAMATH_CALUDE_y1_less_than_y2_l2653_265331

/-- Given a linear function y = 3x - b and two points P₁(3, y₁) and P₂(4, y₂) on its graph,
    prove that y₁ < y₂. -/
theorem y1_less_than_y2 (b : ℝ) (y₁ y₂ : ℝ) 
    (h₁ : y₁ = 3 * 3 - b) 
    (h₂ : y₂ = 3 * 4 - b) : 
  y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_less_than_y2_l2653_265331


namespace NUMINAMATH_CALUDE_decreasing_g_implies_a_nonpositive_l2653_265369

-- Define the function g(x)
def g (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x

-- Define what it means for g to be decreasing on ℝ
def isDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

-- Theorem statement
theorem decreasing_g_implies_a_nonpositive :
  ∀ a : ℝ, isDecreasing (g a) → a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_decreasing_g_implies_a_nonpositive_l2653_265369


namespace NUMINAMATH_CALUDE_probability_at_least_one_strike_l2653_265348

theorem probability_at_least_one_strike (p : ℝ) (h : p = 2/5) :
  1 - (1 - p)^2 = 16/25 := by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_strike_l2653_265348


namespace NUMINAMATH_CALUDE_angle_ABC_is_30_degrees_l2653_265361

theorem angle_ABC_is_30_degrees (BA BC : ℝ × ℝ) : 
  BA = (1/2, Real.sqrt 3/2) → 
  BC = (Real.sqrt 3/2, 1/2) → 
  Real.arccos ((BA.1 * BC.1 + BA.2 * BC.2) / (Real.sqrt (BA.1^2 + BA.2^2) * Real.sqrt (BC.1^2 + BC.2^2))) = π/6 := by
  sorry

end NUMINAMATH_CALUDE_angle_ABC_is_30_degrees_l2653_265361


namespace NUMINAMATH_CALUDE_half_pond_fill_time_l2653_265370

/-- Represents the growth of water hyacinth in a pond -/
def WaterHyacinthGrowth :=
  {growth : ℕ → ℝ // 
    (∀ n, growth (n + 1) = 2 * growth n) ∧ 
    (growth 10 = 1)}

theorem half_pond_fill_time (g : WaterHyacinthGrowth) : 
  g.val 9 = 1/2 := by sorry

end NUMINAMATH_CALUDE_half_pond_fill_time_l2653_265370


namespace NUMINAMATH_CALUDE_min_value_of_f_l2653_265335

def f (x : ℝ) : ℝ := x^2 + 2*x + 4

theorem min_value_of_f :
  ∃ (x_min : ℝ), (∀ x, f x ≥ f x_min) ∧ (x_min = -1) ∧ (f x_min = 3) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2653_265335


namespace NUMINAMATH_CALUDE_sufficient_questions_sufficient_questions_10n_l2653_265371

/-- Represents the origin of a scientist -/
inductive Origin
| Piripocs
| Nekeresd

/-- Represents a scientist at the congress -/
structure Scientist where
  origin : Origin

/-- Represents the congress of scientists -/
structure Congress where
  n : ℕ
  scientists : Fin n → Scientist
  more_piripocs : ∃ (p : ℕ), 2 * p > n ∧ (∀ i : Fin n, (scientists i).origin = Origin.Piripocs → i.val < p)

/-- Function to ask a question about a scientist's origin -/
def ask_question (c : Congress) (asker : Fin c.n) (about : Fin c.n) : Origin :=
  match (c.scientists asker).origin with
  | Origin.Piripocs => (c.scientists about).origin
  | Origin.Nekeresd => sorry  -- This can be either true or false

/-- Theorem stating that n^2 / 2 questions are sufficient -/
theorem sufficient_questions (c : Congress) :
  ∃ (strategy : (Fin c.n → Fin c.n → Origin) → Fin c.n → Origin),
    (∀ f : Fin c.n → Fin c.n → Origin, 
      (∀ i j : Fin c.n, f i j = ask_question c i j) → 
      (∀ i : Fin c.n, strategy f i = (c.scientists i).origin)) ∧
    (∃ m : ℕ, 2 * m ≤ c.n * c.n ∧ 
      ∀ f : Fin c.n → Fin c.n → Origin, 
        (∃ s : Finset (Fin c.n × Fin c.n), s.card ≤ m ∧ 
          ∀ i j : Fin c.n, f i j = ask_question c i j → (i, j) ∈ s)) :=
sorry

/-- Theorem stating that 10n questions are also sufficient -/
theorem sufficient_questions_10n (c : Congress) :
  ∃ (strategy : (Fin c.n → Fin c.n → Origin) → Fin c.n → Origin),
    (∀ f : Fin c.n → Fin c.n → Origin, 
      (∀ i j : Fin c.n, f i j = ask_question c i j) → 
      (∀ i : Fin c.n, strategy f i = (c.scientists i).origin)) ∧
    (∃ s : Finset (Fin c.n × Fin c.n), s.card ≤ 10 * c.n ∧ 
      ∀ f : Fin c.n → Fin c.n → Origin, 
        (∀ i j : Fin c.n, f i j = ask_question c i j → (i, j) ∈ s)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_questions_sufficient_questions_10n_l2653_265371


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l2653_265330

theorem r_value_when_n_is_3 :
  let n : ℕ := 3
  let t : ℕ := 2^n + n
  let r : ℕ := 3^t - t
  r = 177136 := by sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l2653_265330


namespace NUMINAMATH_CALUDE_dragon_poker_ways_l2653_265343

/-- The number of points to be scored -/
def target_points : ℕ := 2018

/-- The number of suits in the deck -/
def num_suits : ℕ := 4

/-- Calculates the number of ways to partition a given number into a specified number of parts -/
def partition_ways (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The main theorem: The number of ways to score exactly 2018 points in Dragon Poker -/
theorem dragon_poker_ways : partition_ways target_points num_suits = 1373734330 := by
  sorry

end NUMINAMATH_CALUDE_dragon_poker_ways_l2653_265343


namespace NUMINAMATH_CALUDE_airplane_trip_people_count_l2653_265366

/-- Represents the airplane trip scenario --/
structure AirplaneTrip where
  bagsPerPerson : ℕ
  weightPerBag : ℕ
  currentCapacity : ℕ
  additionalBags : ℕ

/-- Calculate the number of people on the trip --/
def numberOfPeople (trip : AirplaneTrip) : ℕ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating the number of people on the trip --/
theorem airplane_trip_people_count :
  let trip := AirplaneTrip.mk 5 50 6000 90
  numberOfPeople trip = 42 := by
  sorry

end NUMINAMATH_CALUDE_airplane_trip_people_count_l2653_265366


namespace NUMINAMATH_CALUDE_negation_equivalence_l2653_265321

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 2*x + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2*x + 5 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2653_265321


namespace NUMINAMATH_CALUDE_lawrence_county_summer_break_l2653_265372

/-- The number of kids who stayed home during summer break in Lawrence county -/
theorem lawrence_county_summer_break (total_kids : ℕ) (camp_kids : ℕ) (h1 : total_kids = 1538832) (h2 : camp_kids = 893835) :
  total_kids - camp_kids = 644997 := by
  sorry

#check lawrence_county_summer_break

end NUMINAMATH_CALUDE_lawrence_county_summer_break_l2653_265372


namespace NUMINAMATH_CALUDE_quadratic_always_positive_inequality_implication_existence_of_divisible_number_l2653_265398

-- Problem 1
theorem quadratic_always_positive : ∀ x : ℝ, x^2 - 8*x + 17 > 0 := by sorry

-- Problem 2
theorem inequality_implication : ∀ x : ℝ, (x+2)^2 - (x-3)^2 ≥ 0 → x ≥ 1/2 := by sorry

-- Problem 3
theorem existence_of_divisible_number : ∃ n : ℕ, 11 ∣ (6*n^2 - 7) := by sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_inequality_implication_existence_of_divisible_number_l2653_265398


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l2653_265301

/-- The line equation passing through a fixed point for all values of m -/
def line_equation (m x y : ℝ) : Prop :=
  (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

/-- The fixed point P -/
def P : ℝ × ℝ := (3, 1)

/-- Theorem stating that P lies on the line for all real m -/
theorem fixed_point_on_line : ∀ m : ℝ, line_equation m P.1 P.2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l2653_265301


namespace NUMINAMATH_CALUDE_sum_of_equal_expressions_l2653_265350

theorem sum_of_equal_expressions (a b c d : ℝ) :
  a + 2 = b + 3 ∧ 
  b + 3 = c + 4 ∧ 
  c + 4 = d + 5 ∧ 
  d + 5 = a + b + c + d + 10 →
  a + b + c + d = -26/3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_equal_expressions_l2653_265350


namespace NUMINAMATH_CALUDE_balanced_distribution_exists_l2653_265318

-- Define the weights of each letter
def weight (letter : Char) : ℕ :=
  match letter with
  | 'O' => 300
  | 'B' => 300
  | 'M' => 200
  | 'E' => 200
  | 'P' => 100
  | _ => 0

-- Define the initial setup
def initial_left : List Char := ['M', 'B']
def initial_right : List Char := ['P', 'E']
def top : Char := 'P'

-- Define the remaining letters to be placed
def remaining_letters : List Char := ['O', 'O', 'B', 'B', 'M', 'M', 'E', 'E', 'P', 'P']

-- Function to calculate the total weight of a list of letters
def total_weight (letters : List Char) : ℕ :=
  letters.map weight |>.sum

-- Theorem stating that a balanced distribution exists
theorem balanced_distribution_exists :
  ∃ (left right : List Char),
    left.length + right.length = remaining_letters.length ∧
    (left ++ initial_left).toFinset ∪ (right ++ initial_right).toFinset ∪ {top} = remaining_letters.toFinset ∪ initial_left.toFinset ∪ initial_right.toFinset ∪ {top} ∧
    total_weight (left ++ initial_left) = total_weight (right ++ initial_right) :=
  sorry

end NUMINAMATH_CALUDE_balanced_distribution_exists_l2653_265318


namespace NUMINAMATH_CALUDE_line_equation_from_conditions_l2653_265383

/-- Vector in R² -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Line in R² -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point in R² -/
structure Point2D where
  x : ℝ
  y : ℝ

def vector_add (v w : Vector2D) : Vector2D :=
  ⟨v.x + w.x, v.y + w.y⟩

def vector_scale (k : ℝ) (v : Vector2D) : Vector2D :=
  ⟨k * v.x, k * v.y⟩

def is_perpendicular (v : Vector2D) (l : Line2D) : Prop :=
  v.x * l.a + v.y * l.b = 0

def point_on_line (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem line_equation_from_conditions 
  (a b : Vector2D)
  (A : Point2D)
  (l : Line2D)
  (h1 : a = ⟨6, 2⟩)
  (h2 : b = ⟨-4, 1/2⟩)
  (h3 : A = ⟨3, -1⟩)
  (h4 : is_perpendicular (vector_add a (vector_scale 2 b)) l)
  (h5 : point_on_line A l) :
  l = ⟨2, -3, -9⟩ :=
sorry

end NUMINAMATH_CALUDE_line_equation_from_conditions_l2653_265383


namespace NUMINAMATH_CALUDE_emma_ball_lists_l2653_265399

/-- The number of balls in the bin -/
def n : ℕ := 24

/-- The number of draws -/
def k : ℕ := 4

/-- The number of possible lists when drawing with replacement from n balls, k times -/
def num_lists (n k : ℕ) : ℕ := n^k

theorem emma_ball_lists : num_lists n k = 331776 := by
  sorry

end NUMINAMATH_CALUDE_emma_ball_lists_l2653_265399


namespace NUMINAMATH_CALUDE_puzzle_solution_l2653_265362

theorem puzzle_solution :
  ∀ (F I V T E N : ℕ),
    F = 8 →
    N % 2 = 1 →
    F ≠ I ∧ F ≠ V ∧ F ≠ T ∧ F ≠ E ∧ F ≠ N ∧
    I ≠ V ∧ I ≠ T ∧ I ≠ E ∧ I ≠ N ∧
    V ≠ T ∧ V ≠ E ∧ V ≠ N ∧
    T ≠ E ∧ T ≠ N ∧
    E ≠ N →
    F < 10 ∧ I < 10 ∧ V < 10 ∧ T < 10 ∧ E < 10 ∧ N < 10 →
    100 * F + 10 * I + V + 100 * F + 10 * I + V = 1000 * T + 100 * E + 10 * N →
    I = 4 :=
by sorry

end NUMINAMATH_CALUDE_puzzle_solution_l2653_265362


namespace NUMINAMATH_CALUDE_relationship_abc_l2653_265389

noncomputable def a : ℝ := (1.1 : ℝ) ^ (0.1 : ℝ)
noncomputable def b : ℝ := Real.log 2
noncomputable def c : ℝ := Real.log (Real.sqrt 3 / 3) / Real.log (1/3)

theorem relationship_abc : a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l2653_265389


namespace NUMINAMATH_CALUDE_reciprocal_sum_fourths_sixths_l2653_265339

theorem reciprocal_sum_fourths_sixths : (1 / (1/4 + 1/6) : ℚ) = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_fourths_sixths_l2653_265339


namespace NUMINAMATH_CALUDE_intersection_area_greater_than_half_l2653_265381

/-- Represents a rectangle in a 2D plane -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents the intersection of two rectangles -/
structure Intersection (r1 r2 : Rectangle) where
  area : ℝ

/-- Theorem: Given two equal rectangles whose contours intersect at 8 points,
    the area of their intersection is greater than half the area of each rectangle -/
theorem intersection_area_greater_than_half 
  (r1 r2 : Rectangle) 
  (h_equal : r1 = r2) 
  (h_intersect : ∃ (pts : Finset (ℝ × ℝ)), pts.card = 8) 
  (i : Intersection r1 r2) : 
  i.area > (1/2) * r1.area := by
  sorry

end NUMINAMATH_CALUDE_intersection_area_greater_than_half_l2653_265381


namespace NUMINAMATH_CALUDE_jellybean_count_l2653_265337

/-- The number of jellybeans in a bag with specific color distributions -/
def total_jellybeans (black green orange red yellow : ℕ) : ℕ :=
  black + green + orange + red + yellow

/-- Theorem stating the total number of jellybeans in the bag -/
theorem jellybean_count : ∃ (black green orange red yellow : ℕ),
  black = 8 ∧
  green = black + 4 ∧
  orange = green - 5 ∧
  red = orange + 3 ∧
  yellow = black - 2 ∧
  total_jellybeans black green orange red yellow = 43 := by
  sorry


end NUMINAMATH_CALUDE_jellybean_count_l2653_265337


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2653_265311

theorem trigonometric_identities (α : ℝ) (h : Real.tan α = 2) :
  (Real.tan (α + π/4) = -3) ∧
  (Real.sin (2*α) / (Real.sin α ^ 2 + Real.sin α * Real.cos α - Real.cos (2*α) - 1) = 1) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2653_265311


namespace NUMINAMATH_CALUDE_negative_values_range_l2653_265319

/-- A quadratic function that takes negative values for some x -/
def takes_negative_values (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - a*x + 1 < 0

/-- The theorem stating the range of a for which f(x) takes negative values -/
theorem negative_values_range (a : ℝ) :
  takes_negative_values a ↔ a > 2 ∨ a < -2 :=
sorry

end NUMINAMATH_CALUDE_negative_values_range_l2653_265319


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l2653_265324

theorem geometric_arithmetic_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence condition
  (a 1 + a 3 = 2 * (2 * a 2)) →  -- arithmetic sequence condition
  q = 2 - Real.sqrt 3 ∨ q = 2 + Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l2653_265324


namespace NUMINAMATH_CALUDE_min_value_theorem_l2653_265392

theorem min_value_theorem (a : ℝ) (x₁ x₂ : ℝ) 
  (h_a : a > 0)
  (h_sol : ∀ x, -x^2 + 6*a*x - 3*a^2 ≥ 0 ↔ x₁ ≤ x ∧ x ≤ x₂) :
  ∃ (m : ℝ), m = 2 * Real.sqrt 6 ∧ 
    ∀ y₁ y₂, (∀ x, -x^2 + 6*a*x - 3*a^2 ≥ 0 ↔ y₁ ≤ x ∧ x ≤ y₂) → 
      y₁ + y₂ + 3*a / (y₁ * y₂) ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2653_265392


namespace NUMINAMATH_CALUDE_point_A_coordinates_l2653_265368

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y - 4 = 0

-- Define a point on the circle
def point_on_circle (P : ℝ × ℝ) : Prop := circle_O P.1 P.2

-- Define a point on the line
def point_on_line (A : ℝ × ℝ) : Prop := line_l A.1 A.2

-- Define the angle PAQ
def angle_PAQ (A P Q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem point_A_coordinates :
  ∀ A : ℝ × ℝ,
  point_on_line A →
  (∀ P Q : ℝ × ℝ, point_on_circle P → point_on_circle Q → angle_PAQ A P Q ≤ 90) →
  (∃ P Q : ℝ × ℝ, point_on_circle P ∧ point_on_circle Q ∧ angle_PAQ A P Q = 90) →
  A = (1, 3) :=
sorry

end NUMINAMATH_CALUDE_point_A_coordinates_l2653_265368


namespace NUMINAMATH_CALUDE_residue_products_l2653_265305

theorem residue_products (n k : ℕ+) : 
  (∃ (a : Fin n → ℤ) (b : Fin k → ℤ), 
    ∀ (i j i' j' : ℕ) (hi : i < n) (hj : j < k) (hi' : i' < n) (hj' : j' < k),
      (i ≠ i' ∨ j ≠ j') → 
      (a ⟨i, hi⟩ * b ⟨j, hj⟩) % (n * k : ℕ) ≠ (a ⟨i', hi'⟩ * b ⟨j', hj'⟩) % (n * k : ℕ)) ↔ 
  Nat.gcd n k = 1 :=
sorry

end NUMINAMATH_CALUDE_residue_products_l2653_265305


namespace NUMINAMATH_CALUDE_specific_book_arrangement_l2653_265346

/-- The number of arrangements for placing math and English books on a shelf. -/
def book_arrangements (n_math : ℕ) (n_english : ℕ) (adjacent_math : ℕ) : ℕ :=
  Nat.factorial n_english * 
  (n_english - 1) * 
  Nat.choose (n_english + adjacent_math - 1) (n_math - adjacent_math)

/-- Theorem stating the number of arrangements for the specific book problem. -/
theorem specific_book_arrangement : book_arrangements 6 5 2 = 2400 := by
  sorry

#eval book_arrangements 6 5 2

end NUMINAMATH_CALUDE_specific_book_arrangement_l2653_265346


namespace NUMINAMATH_CALUDE_sin_seven_pi_thirds_l2653_265359

theorem sin_seven_pi_thirds : Real.sin (7 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_seven_pi_thirds_l2653_265359


namespace NUMINAMATH_CALUDE_part_one_part_two_l2653_265322

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a - 1 < x ∧ x < -a}
def B : Set ℝ := {x | |x - 1| < 2}

-- Part 1
theorem part_one : (Aᶜ (-1) ∪ B) = {x | x ≤ -3 ∨ x > -1} := by sorry

-- Part 2
theorem part_two : ∀ a : ℝ, (A a ⊆ B ∧ A a ≠ B) ↔ a ≥ 0 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2653_265322


namespace NUMINAMATH_CALUDE_fraction_calculation_l2653_265323

theorem fraction_calculation : (1 / 3 : ℚ) * (4 / 7 : ℚ) * (9 / 13 : ℚ) + (1 / 2 : ℚ) = 49 / 78 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l2653_265323


namespace NUMINAMATH_CALUDE_smallest_integer_divisible_by_24_and_8_l2653_265314

theorem smallest_integer_divisible_by_24_and_8 : ∃ n : ℕ+, 
  (∀ m : ℕ+, m < n → (¬(24 ∣ m^2) ∨ ¬(8 ∣ m))) ∧ 
  24 ∣ n^2 ∧ 
  8 ∣ n ∧
  ∀ d : ℕ+, d ∣ n → d ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_divisible_by_24_and_8_l2653_265314


namespace NUMINAMATH_CALUDE_nearest_multiple_of_11_to_457_l2653_265338

theorem nearest_multiple_of_11_to_457 :
  ∃ (n : ℤ), n % 11 = 0 ∧ 
  ∀ (m : ℤ), m % 11 = 0 → |n - 457| ≤ |m - 457| ∧
  n = 462 := by
  sorry

end NUMINAMATH_CALUDE_nearest_multiple_of_11_to_457_l2653_265338


namespace NUMINAMATH_CALUDE_birthday_problem_solution_l2653_265315

/-- Represents a person's age -/
structure Age :=
  (value : ℕ)

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Represents the ages of Alice, Bob, and Carl -/
structure FamilyAges :=
  (alice : Age)
  (bob : Age)
  (carl : Age)

/-- Checks if one age is a multiple of another -/
def isMultipleOf (a b : Age) : Prop :=
  ∃ k : ℕ, a.value = k * b.value

/-- Represents the conditions of the problem -/
structure BirthdayProblem :=
  (ages : FamilyAges)
  (aliceOlderThanBob : ages.alice.value = ages.bob.value + 2)
  (carlAgeToday : ages.carl.value = 3)
  (bobMultipleOfCarl : isMultipleOf ages.bob ages.carl)
  (firstOfFourBirthdays : ∀ n : ℕ, n < 4 → isMultipleOf ⟨ages.bob.value + n⟩ ⟨ages.carl.value + n⟩)

/-- The main theorem to prove -/
theorem birthday_problem_solution (problem : BirthdayProblem) :
  ∃ (futureAliceAge : ℕ),
    futureAliceAge > problem.ages.alice.value ∧
    isMultipleOf ⟨futureAliceAge⟩ ⟨problem.ages.carl.value + (futureAliceAge - problem.ages.alice.value)⟩ ∧
    sumOfDigits futureAliceAge = 6 :=
  sorry

end NUMINAMATH_CALUDE_birthday_problem_solution_l2653_265315


namespace NUMINAMATH_CALUDE_marble_probability_l2653_265397

/-- Given a box of 100 marbles with specified probabilities for white and green marbles,
    prove that the probability of drawing either a red or blue marble is 11/20. -/
theorem marble_probability (total : ℕ) (p_white p_green : ℚ) 
    (h_total : total = 100)
    (h_white : p_white = 1 / 4)
    (h_green : p_green = 1 / 5) :
    (total - (p_white * total + p_green * total)) / total = 11 / 20 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l2653_265397


namespace NUMINAMATH_CALUDE_division_subtraction_problem_l2653_265360

theorem division_subtraction_problem (x : ℝ) : 
  (800 / x) - 154 = 6 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_division_subtraction_problem_l2653_265360


namespace NUMINAMATH_CALUDE_smallest_four_digit_unique_divisible_by_digits_with_five_l2653_265382

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def has_unique_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 4 ∧ digits.toFinset.card = 4

def divisible_by_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≠ 0 → n % d = 0

def includes_digit_five (n : ℕ) : Prop :=
  5 ∈ n.digits 10

theorem smallest_four_digit_unique_divisible_by_digits_with_five :
  ∀ n : ℕ, is_four_digit n →
           has_unique_digits n →
           divisible_by_digits n →
           includes_digit_five n →
           1560 ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_unique_divisible_by_digits_with_five_l2653_265382


namespace NUMINAMATH_CALUDE_line_intersects_parabola_vertex_l2653_265380

theorem line_intersects_parabola_vertex (b : ℝ) : 
  (∃! (x y : ℝ), y = x + b ∧ y = x^2 + 2*b^2 ∧ x = 0) ↔ (b = 0 ∨ b = 1/2) :=
sorry

end NUMINAMATH_CALUDE_line_intersects_parabola_vertex_l2653_265380


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l2653_265374

/-- Given a hyperbola with equation x²/9 - y²/m = 1 and focal distance length 8,
    prove that its asymptote equation is y = ±(√7/3)x -/
theorem hyperbola_asymptote (m : ℝ) :
  (∀ x y, x^2 / 9 - y^2 / m = 1) →  -- Hyperbola equation
  (∃ c, c = 4 ∧ c^2 = 9 + m) →      -- Focal distance condition
  (∃ k, ∀ x, y = k * x ∨ y = -k * x) ∧ k = Real.sqrt 7 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l2653_265374


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2653_265309

theorem sqrt_equation_solution (x : ℚ) : 
  (Real.sqrt (3 * x + 5) / Real.sqrt (6 * x + 5) = Real.sqrt 5 / 3) → x = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2653_265309


namespace NUMINAMATH_CALUDE_intersection_line_equation_l2653_265384

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def line_through_intersections (c1 c2 : Circle) : ℝ → ℝ → Prop :=
  fun x y => x + y = 26/3

theorem intersection_line_equation :
  let c1 : Circle := ⟨(2, -3), 10⟩
  let c2 : Circle := ⟨(-4, 7), 6⟩
  ∀ x y : ℝ,
    (x - c1.center.1)^2 + (y - c1.center.2)^2 = c1.radius^2 ∧
    (x - c2.center.1)^2 + (y - c2.center.2)^2 = c2.radius^2 →
    line_through_intersections c1 c2 x y :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l2653_265384


namespace NUMINAMATH_CALUDE_dynamic_number_sum_divisible_by_three_l2653_265332

/-- A dynamic number is a four-digit positive integer where each digit is not 0,
    and the two-digit number formed by the tenth and unit places is twice
    the two-digit number formed by the thousandth and hundredth places. -/
def isDynamicNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  ∀ a b c d : ℕ,
    n = 1000 * a + 100 * b + 10 * c + d →
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    10 * c + d = 2 * (10 * a + b)

theorem dynamic_number_sum_divisible_by_three (a : ℕ) (h : 10 ≤ a ∧ a < 100) :
  ∃ k : ℕ, 102 * a + (200 * a + a) = 3 * k := by
  sorry

end NUMINAMATH_CALUDE_dynamic_number_sum_divisible_by_three_l2653_265332


namespace NUMINAMATH_CALUDE_pet_store_puppies_l2653_265378

theorem pet_store_puppies (initial_kittens : ℕ) (sold_puppies sold_kittens remaining_pets : ℕ) 
  (h1 : initial_kittens = 6)
  (h2 : sold_puppies = 2)
  (h3 : sold_kittens = 3)
  (h4 : remaining_pets = 8) :
  ∃ initial_puppies : ℕ, 
    initial_puppies - sold_puppies + initial_kittens - sold_kittens = remaining_pets ∧ 
    initial_puppies = 7 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_puppies_l2653_265378


namespace NUMINAMATH_CALUDE_john_money_left_l2653_265307

/-- The amount of money John has left after buying pizzas and drinks -/
def money_left (q : ℝ) : ℝ :=
  let drink_cost := q
  let small_pizza_cost := q
  let large_pizza_cost := 4 * q
  let total_cost := 4 * drink_cost + 2 * small_pizza_cost + large_pizza_cost
  50 - total_cost

/-- Theorem stating that John will have 50 - 10q dollars left -/
theorem john_money_left (q : ℝ) : money_left q = 50 - 10 * q := by
  sorry

end NUMINAMATH_CALUDE_john_money_left_l2653_265307


namespace NUMINAMATH_CALUDE_problem1_l2653_265329

theorem problem1 (x y : ℝ) : x^2 * (-2*x*y^2)^3 = -8*x^5*y^6 := by sorry

end NUMINAMATH_CALUDE_problem1_l2653_265329


namespace NUMINAMATH_CALUDE_girls_count_in_school_l2653_265302

/-- Represents the number of students in a school with a given boy-to-girl ratio. -/
structure School where
  total : ℕ
  ratio : ℚ
  boys : ℕ
  girls : ℕ
  ratio_def : ratio = boys / girls
  total_def : total = boys + girls

/-- Theorem: In a school with 90 students and a 1:2 boy-to-girl ratio, there are 60 girls. -/
theorem girls_count_in_school (s : School) 
    (h_total : s.total = 90)
    (h_ratio : s.ratio = 1/2) : 
    s.girls = 60 := by
  sorry

end NUMINAMATH_CALUDE_girls_count_in_school_l2653_265302


namespace NUMINAMATH_CALUDE_peter_green_notebooks_l2653_265391

/-- Represents the number of green notebooks Peter bought -/
def green_notebooks (total notebooks : ℕ) (black_notebooks pink_notebooks : ℕ) 
  (total_cost black_cost pink_cost : ℕ) : ℕ :=
  total - black_notebooks - pink_notebooks

/-- Theorem stating that Peter bought 2 green notebooks -/
theorem peter_green_notebooks : 
  green_notebooks 4 1 1 45 15 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_peter_green_notebooks_l2653_265391


namespace NUMINAMATH_CALUDE_max_m_value_l2653_265349

theorem max_m_value (b a m : ℝ) (h_b : b > 0) :
  (∀ a, (b - (a - 2))^2 + (Real.log b - (a - 1))^2 ≥ m^2 - m) →
  m ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l2653_265349


namespace NUMINAMATH_CALUDE_unique_assignment_l2653_265365

-- Define the type for tables
inductive Table
| T1 | T2 | T3 | T4

-- Define the type for students
inductive Student
| Albert | Bogdan | Vadim | Denis

-- Define a function to represent the assignment of tables to students
def assignment : Student → Table
| Student.Albert => Table.T4
| Student.Bogdan => Table.T2
| Student.Vadim => Table.T1
| Student.Denis => Table.T3

-- Define a predicate for table intersection
def intersects (t1 t2 : Table) : Prop := sorry

-- Albert and Bogdan colored some cells
axiom albert_bogdan_colored : ∀ (t : Table), t ≠ Table.T1 → intersects (assignment Student.Albert) t ∨ intersects (assignment Student.Bogdan) t

-- Vadim's table doesn't intersect with Albert's or Bogdan's
axiom vadim_condition : ¬(intersects (assignment Student.Vadim) (assignment Student.Albert)) ∧ 
                        ¬(intersects (assignment Student.Vadim) (assignment Student.Bogdan))

-- Denis's table doesn't intersect with Bogdan's or Vadim's
axiom denis_condition : ¬(intersects (assignment Student.Denis) (assignment Student.Bogdan)) ∧ 
                        ¬(intersects (assignment Student.Denis) (assignment Student.Vadim))

-- Theorem stating that the given assignment is the only valid solution
theorem unique_assignment : 
  ∀ (f : Student → Table), 
    (∀ (s1 s2 : Student), s1 ≠ s2 → f s1 ≠ f s2) →
    (∀ (t : Table), t ≠ Table.T1 → intersects (f Student.Albert) t ∨ intersects (f Student.Bogdan) t) →
    (¬(intersects (f Student.Vadim) (f Student.Albert)) ∧ ¬(intersects (f Student.Vadim) (f Student.Bogdan))) →
    (¬(intersects (f Student.Denis) (f Student.Bogdan)) ∧ ¬(intersects (f Student.Denis) (f Student.Vadim))) →
    f = assignment := by sorry

end NUMINAMATH_CALUDE_unique_assignment_l2653_265365


namespace NUMINAMATH_CALUDE_locus_is_ellipse_l2653_265390

/-- A complex number z tracing a circle centered at the origin with radius 3 -/
def z_on_circle (z : ℂ) : Prop := Complex.abs z = 3

/-- The locus of points (x, y) satisfying x + yi = z + 1/z -/
def locus (z : ℂ) (x y : ℝ) : Prop := x + y * Complex.I = z + 1 / z

/-- The equation of an ellipse in standard form -/
def is_ellipse (x y : ℝ) : Prop := ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ x^2 / a^2 + y^2 / b^2 = 1

theorem locus_is_ellipse :
  ∀ z : ℂ, z_on_circle z →
  ∀ x y : ℝ, locus z x y →
  is_ellipse x y :=
sorry

end NUMINAMATH_CALUDE_locus_is_ellipse_l2653_265390
