import Mathlib

namespace NUMINAMATH_GPT_train_crosses_bridge_in_30_seconds_l2095_209574

noncomputable def train_length : ℝ := 100
noncomputable def bridge_length : ℝ := 200
noncomputable def train_speed_kmph : ℝ := 36

noncomputable def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)

noncomputable def total_distance : ℝ := train_length + bridge_length

noncomputable def crossing_time : ℝ := total_distance / train_speed_mps

theorem train_crosses_bridge_in_30_seconds :
  crossing_time = 30 := 
by
  sorry

end NUMINAMATH_GPT_train_crosses_bridge_in_30_seconds_l2095_209574


namespace NUMINAMATH_GPT_box_2008_count_l2095_209531

noncomputable def box_count (a : ℕ → ℕ) : Prop :=
  a 1 = 7 ∧ a 4 = 8 ∧ ∀ n : ℕ, 1 ≤ n ∧ n + 3 ≤ 2008 → a n + a (n + 1) + a (n + 2) + a (n + 3) = 30

theorem box_2008_count (a : ℕ → ℕ) (h : box_count a) : a 2008 = 8 :=
by
  sorry

end NUMINAMATH_GPT_box_2008_count_l2095_209531


namespace NUMINAMATH_GPT_cricket_run_rate_l2095_209543

theorem cricket_run_rate
  (run_rate_first_10_overs : ℝ)
  (overs_first_10_overs : ℕ)
  (target_runs : ℕ)
  (remaining_overs : ℕ)
  (run_rate_required : ℝ) :
  run_rate_first_10_overs = 3.2 →
  overs_first_10_overs = 10 →
  target_runs = 242 →
  remaining_overs = 40 →
  run_rate_required = 5.25 →
  (target_runs - (run_rate_first_10_overs * overs_first_10_overs)) = 210 →
  (target_runs - (run_rate_first_10_overs * overs_first_10_overs)) / remaining_overs = run_rate_required :=
by
  sorry

end NUMINAMATH_GPT_cricket_run_rate_l2095_209543


namespace NUMINAMATH_GPT_no_nontrivial_integer_solutions_l2095_209570

theorem no_nontrivial_integer_solutions (a b c d : ℤ) :
  6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * d^2 → a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_no_nontrivial_integer_solutions_l2095_209570


namespace NUMINAMATH_GPT_min_distance_between_parallel_lines_distance_when_line_parallel_to_x_axis_l2095_209547

noncomputable def line_equation (A B C x y : ℝ) : Prop := A * x + B * y + C = 0

noncomputable def point_on_line (x y A B C : ℝ) : Prop := line_equation A B C x y

noncomputable def distance_between_parallel_lines (A B C1 C2 : ℝ) : ℝ :=
  (|C2 - C1|) / (Real.sqrt (A^2 + B^2))

theorem min_distance_between_parallel_lines :
  ∀ (A B C1 C2 x y : ℝ),
  point_on_line x y A B C1 ∧ point_on_line x y A B C2 →
  distance_between_parallel_lines A B C1 C2 = 3 :=
by
  intros A B C1 C2 x y h
  sorry

theorem distance_when_line_parallel_to_x_axis :
  ∀ (x1 x2 y k A B C1 C2 : ℝ),
  k = 3 →
  point_on_line x1 k A B C1 →
  point_on_line x2 k A B C2 →
  |x2 - x1| = 5 :=
by
  intros x1 x2 y k A B C1 C2 hk h1 h2
  sorry

end NUMINAMATH_GPT_min_distance_between_parallel_lines_distance_when_line_parallel_to_x_axis_l2095_209547


namespace NUMINAMATH_GPT_price_first_variety_is_126_l2095_209522

variable (x : ℝ) -- price of the first variety per kg (unknown we need to solve for)
variable (p2 : ℝ := 135) -- price of the second variety per kg
variable (p3 : ℝ := 175.5) -- price of the third variety per kg
variable (mix_ratio : ℝ := 4) -- total weight ratio of the mixture
variable (mix_price : ℝ := 153) -- price of the mixture per kg
variable (w1 w2 w3 : ℝ := 1) -- weights of the first two varieties
variable (w4 : ℝ := 2) -- weight of the third variety

theorem price_first_variety_is_126:
  (w1 * x + w2 * p2 + w4 * p3) / mix_ratio = mix_price → x = 126 := by
  sorry

end NUMINAMATH_GPT_price_first_variety_is_126_l2095_209522


namespace NUMINAMATH_GPT_statement_A_statement_B_statement_C_statement_D_statement_E_l2095_209579

def diamond (x y : ℝ) : ℝ := x^2 - 2*x*y + y^2

theorem statement_A : ∀ (x y : ℝ), diamond x y = diamond y x := sorry

theorem statement_B : ∀ (x y : ℝ), 2 * (diamond x y) ≠ diamond (2 * x) (2 * y) := sorry

theorem statement_C : ∀ (x : ℝ), diamond x 0 = x^2 := sorry

theorem statement_D : ∀ (x : ℝ), diamond x x = 0 := sorry

theorem statement_E : ∀ (x y : ℝ), x = y → diamond x y = 0 := sorry

end NUMINAMATH_GPT_statement_A_statement_B_statement_C_statement_D_statement_E_l2095_209579


namespace NUMINAMATH_GPT_triangle_is_isosceles_l2095_209523

theorem triangle_is_isosceles 
  (A B C : Real) 
  (h_triangle : A + B + C = Real.pi) 
  (h_sin_identity : Real.sin A = 2 * Real.sin C * Real.cos B) : 
  (B = C) :=
sorry

end NUMINAMATH_GPT_triangle_is_isosceles_l2095_209523


namespace NUMINAMATH_GPT_value_diff_l2095_209516

theorem value_diff (a b : ℕ) (h1 : a * b = 2 * (a + b) + 14) (h2 : b = 8) : b - a = 3 :=
by
  sorry

end NUMINAMATH_GPT_value_diff_l2095_209516


namespace NUMINAMATH_GPT_sine_triangle_inequality_l2095_209544

theorem sine_triangle_inequality 
  {a b c : ℝ} (h_triangle : a + b + c ≤ 2 * Real.pi) (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c) 
  (ha_lt_pi : a < Real.pi) (hb_lt_pi : b < Real.pi) (hc_lt_pi : c < Real.pi) :
  (Real.sin a + Real.sin b > Real.sin c) ∧ 
  (Real.sin a + Real.sin c > Real.sin b) ∧ 
  (Real.sin b + Real.sin c > Real.sin a) :=
by
  sorry

end NUMINAMATH_GPT_sine_triangle_inequality_l2095_209544


namespace NUMINAMATH_GPT_quadratic_one_real_root_l2095_209550

theorem quadratic_one_real_root (a : ℝ) :
  (∀ x : ℝ, (a * x^2 - 2 * x + 1 = 0) ↔ ((a = 0) ∨ (a = 1))) :=
sorry

end NUMINAMATH_GPT_quadratic_one_real_root_l2095_209550


namespace NUMINAMATH_GPT_exists_nat_pair_l2095_209586

theorem exists_nat_pair 
  (k : ℕ) : 
  let a := 2 * k
  let b := 2 * k * k + 2 * k + 1
  (b - 1) % (a + 1) = 0 ∧ (a * a + a + 2) % b = 0 := by
  sorry

end NUMINAMATH_GPT_exists_nat_pair_l2095_209586


namespace NUMINAMATH_GPT_intersection_with_y_axis_l2095_209557

theorem intersection_with_y_axis (y : ℝ) : 
  (∃ y, (0, y) ∈ {(x, 2 * x + 4) | x : ℝ}) ↔ y = 4 :=
by 
  sorry

end NUMINAMATH_GPT_intersection_with_y_axis_l2095_209557


namespace NUMINAMATH_GPT_max_abs_sum_l2095_209526

-- Define the condition for the ellipse equation
def ellipse_condition (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 2) = 1

-- Prove that the largest possible value of |x| + |y| given the condition is 2√3
theorem max_abs_sum (x y : ℝ) (h : ellipse_condition x y) : |x| + |y| ≤ 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_max_abs_sum_l2095_209526


namespace NUMINAMATH_GPT_smallest_base10_integer_l2095_209520

theorem smallest_base10_integer {a b n : ℕ} (ha : a > 2) (hb : b > 2)
  (h₁ : 2 * a + 1 = n) (h₂ : 1 * b + 2 = n) :
  n = 7 :=
sorry

end NUMINAMATH_GPT_smallest_base10_integer_l2095_209520


namespace NUMINAMATH_GPT_xanthia_hot_dogs_l2095_209576

theorem xanthia_hot_dogs (a b : ℕ) (h₁ : a = 5) (h₂ : b = 7) :
  ∃ n m : ℕ, n * a = m * b ∧ n = 7 := by 
sorry

end NUMINAMATH_GPT_xanthia_hot_dogs_l2095_209576


namespace NUMINAMATH_GPT_line_intersects_parabola_once_l2095_209540

theorem line_intersects_parabola_once (k : ℝ) :
  (x = k)
  ∧ (x = -3 * y^2 - 4 * y + 7)
  ∧ (3 * y^2 + 4 * y + (k - 7)) = 0
  ∧ ((4)^2 - 4 * 3 * (k - 7) = 0)
  → k = 25 / 3 := 
by
  sorry

end NUMINAMATH_GPT_line_intersects_parabola_once_l2095_209540


namespace NUMINAMATH_GPT_student_correct_answers_l2095_209594

theorem student_correct_answers (C W : ℕ) (h₁ : C + W = 50) (h₂ : 4 * C - W = 130) : C = 36 := 
by
  sorry

end NUMINAMATH_GPT_student_correct_answers_l2095_209594


namespace NUMINAMATH_GPT_quentavious_gum_count_l2095_209506

def initial_nickels : Nat := 5
def remaining_nickels : Nat := 2
def gum_per_nickel : Nat := 2
def traded_nickels (initial remaining : Nat) : Nat := initial - remaining
def total_gum (trade_n gum_per_n : Nat) : Nat := trade_n * gum_per_n

theorem quentavious_gum_count : total_gum (traded_nickels initial_nickels remaining_nickels) gum_per_nickel = 6 := by
  sorry

end NUMINAMATH_GPT_quentavious_gum_count_l2095_209506


namespace NUMINAMATH_GPT_sum_areas_frequency_distribution_histogram_l2095_209524

theorem sum_areas_frequency_distribution_histogram :
  ∀ (rectangles : List ℝ), (∀ r ∈ rectangles, 0 ≤ r ∧ r ≤ 1) → rectangles.sum = 1 := 
  by
    intro rectangles h
    sorry

end NUMINAMATH_GPT_sum_areas_frequency_distribution_histogram_l2095_209524


namespace NUMINAMATH_GPT_John_age_l2095_209596

theorem John_age (Drew Maya Peter John Jacob : ℕ)
  (h1 : Drew = Maya + 5)
  (h2 : Peter = Drew + 4)
  (h3 : John = 2 * Maya)
  (h4 : (Jacob + 2) * 2 = Peter + 2)
  (h5 : Jacob = 11) : John = 30 :=
by 
  sorry

end NUMINAMATH_GPT_John_age_l2095_209596


namespace NUMINAMATH_GPT_problem1_problem2_l2095_209577

-- Problem 1
theorem problem1 (a b : ℤ) (h1 : a = 4) (h2 : b = 5) : a - b = -1 := 
by {
  sorry
}

-- Problem 2
theorem problem2 (a b m n s : ℤ) (h1 : a + b = 0) (h2 : m * n = 1) (h3 : |s| = 3) :
  a + b + m * n + s = 4 ∨ a + b + m * n + s = -2 := 
by {
  sorry
}

end NUMINAMATH_GPT_problem1_problem2_l2095_209577


namespace NUMINAMATH_GPT_part1_part2_l2095_209510

open Real

variable {x y a: ℝ}

-- Condition for the second proof to avoid division by zero
variable (h1 : a ≠ 1) (h2 : a ≠ 4) (h3 : a ≠ -4)

theorem part1 : (x + y)^2 + y * (3 * x - y) = x^2 + 5 * (x * y) := 
by sorry

theorem part2 (h1: a ≠ 1) (h2: a ≠ 4) (h3: a ≠ -4) : 
  ((4 - a^2) / (a - 1) + a) / ((a^2 - 16) / (a - 1)) = -1 / (a + 4) := 
by sorry

end NUMINAMATH_GPT_part1_part2_l2095_209510


namespace NUMINAMATH_GPT_number_of_pupils_l2095_209538

theorem number_of_pupils (n : ℕ) 
  (h1 : 83 - 63 = 20) 
  (h2 : (20 : ℝ) / n = 1 / 2) : 
  n = 40 := 
sorry

end NUMINAMATH_GPT_number_of_pupils_l2095_209538


namespace NUMINAMATH_GPT_number_of_convex_quadrilaterals_l2095_209532

-- Each definition used in Lean 4 statement should directly appear in the conditions problem.

variable {n : ℕ} -- Definition of n in Lean

-- Conditions
def distinct_points_on_circle (n : ℕ) : Prop := n = 10

-- Question and correct answer
theorem number_of_convex_quadrilaterals (h : distinct_points_on_circle n) : 
    (n.choose 4) = 210 := by
  sorry

end NUMINAMATH_GPT_number_of_convex_quadrilaterals_l2095_209532


namespace NUMINAMATH_GPT_baez_marble_loss_l2095_209512

theorem baez_marble_loss :
  ∃ p : ℚ, (p > 0 ∧ (p / 100) * 25 * 2 = 60) ∧ p = 20 :=
by
  sorry

end NUMINAMATH_GPT_baez_marble_loss_l2095_209512


namespace NUMINAMATH_GPT_f_increasing_f_odd_zero_l2095_209560

noncomputable def f (a x : ℝ) : ℝ := a - 2 / (2^x + 1)

-- 1. Prove that f(x) is always an increasing function for any real a.
theorem f_increasing (a : ℝ) : ∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2 :=
by
  sorry

-- 2. Determine the value of a such that f(-x) + f(x) = 0 always holds.
theorem f_odd_zero (a : ℝ) : (∀ x : ℝ, f a (-x) + f a x = 0) → a = 1 :=
by
  sorry

end NUMINAMATH_GPT_f_increasing_f_odd_zero_l2095_209560


namespace NUMINAMATH_GPT_cartesian_coordinates_problem_l2095_209537

theorem cartesian_coordinates_problem
  (x1 y1 x2 y2 : ℕ)
  (h1 : x1 < y1)
  (h2 : x2 > y2)
  (h3 : x2 * y2 = x1 * y1 + 67)
  (h4 : 0 < x1 ∧ 0 < y1 ∧ 0 < x2 ∧ 0 < y2)
  : Nat.digits 10 (x1 * 1000 + y1 * 100 + x2 * 10 + y2) = [1, 9, 8, 5] :=
by
  sorry

end NUMINAMATH_GPT_cartesian_coordinates_problem_l2095_209537


namespace NUMINAMATH_GPT_difference_of_squares_l2095_209511

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 10) (h2 : x - y = 8) : x^2 - y^2 = 80 := 
sorry

end NUMINAMATH_GPT_difference_of_squares_l2095_209511


namespace NUMINAMATH_GPT_brown_house_number_l2095_209567

-- Defining the problem conditions
def sum_arithmetic_series (k : ℕ) := k * (k + 1) / 2

theorem brown_house_number (t n : ℕ) (h1 : 20 < t) (h2 : t < 500)
    (h3 : sum_arithmetic_series n = sum_arithmetic_series t / 2) : n = 84 := by
  sorry

end NUMINAMATH_GPT_brown_house_number_l2095_209567


namespace NUMINAMATH_GPT_carmen_sold_1_box_of_fudge_delights_l2095_209500

noncomputable def boxes_of_fudge_delights (total_earned: ℝ) (samoas_price: ℝ) (thin_mints_price: ℝ) (fudge_delights_price: ℝ) (sugar_cookies_price: ℝ) (samoas_sold: ℝ) (thin_mints_sold: ℝ) (sugar_cookies_sold: ℝ): ℝ :=
  let samoas_total := samoas_price * samoas_sold
  let thin_mints_total := thin_mints_price * thin_mints_sold
  let sugar_cookies_total := sugar_cookies_price * sugar_cookies_sold
  let other_cookies_total := samoas_total + thin_mints_total + sugar_cookies_total
  (total_earned - other_cookies_total) / fudge_delights_price

theorem carmen_sold_1_box_of_fudge_delights: boxes_of_fudge_delights 42 4 3.5 5 2 3 2 9 = 1 :=
by
  sorry

end NUMINAMATH_GPT_carmen_sold_1_box_of_fudge_delights_l2095_209500


namespace NUMINAMATH_GPT_more_pencils_than_pens_l2095_209534

theorem more_pencils_than_pens : 
  ∀ (P L : ℕ), L = 30 → (P / L: ℚ) = 5 / 6 → ((L - P) = 5) := by
  intros P L hL hRatio
  sorry

end NUMINAMATH_GPT_more_pencils_than_pens_l2095_209534


namespace NUMINAMATH_GPT_greatest_number_of_groups_l2095_209502

theorem greatest_number_of_groups (s a t b n : ℕ) (hs : s = 10) (ha : a = 15) (ht : t = 12) (hb : b = 18) :
  (∀ n, n ≤ n ∧ n ∣ s ∧ n ∣ a ∧ n ∣ t ∧ n ∣ b ∧ n > 1 → 
  (s / n < (a / n) + (t / n) + (b / n))
  ∧ (∃ groups, groups = n)) → n = 3 :=
sorry

end NUMINAMATH_GPT_greatest_number_of_groups_l2095_209502


namespace NUMINAMATH_GPT_point_in_second_quadrant_l2095_209568

theorem point_in_second_quadrant {x : ℝ} (h1 : 6 - 2 * x < 0) (h2 : x - 5 > 0) : x > 5 :=
sorry

end NUMINAMATH_GPT_point_in_second_quadrant_l2095_209568


namespace NUMINAMATH_GPT_pastries_made_initially_l2095_209535

theorem pastries_made_initially 
  (sold : ℕ) (remaining : ℕ) (initial : ℕ) 
  (h1 : sold = 103) (h2 : remaining = 45) : 
  initial = 148 :=
by
  have h := h1
  have r := h2
  sorry

end NUMINAMATH_GPT_pastries_made_initially_l2095_209535


namespace NUMINAMATH_GPT_sticks_form_equilateral_triangle_l2095_209515

theorem sticks_form_equilateral_triangle (n : ℕ) (h1 : n ≥ 5) (h2 : n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) :
  ∃ k, k * 3 = (n * (n + 1)) / 2 :=
by
  sorry

end NUMINAMATH_GPT_sticks_form_equilateral_triangle_l2095_209515


namespace NUMINAMATH_GPT_log_base_250_2662sqrt10_l2095_209581

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

variables (a b : ℝ)
variables (h1 : log_base 50 55 = a) (h2 : log_base 55 20 = b)

theorem log_base_250_2662sqrt10 : log_base 250 (2662 * Real.sqrt 10) = (18 * a + 11 * a * b - 13) / (10 - 2 * a * b) :=
by
  sorry

end NUMINAMATH_GPT_log_base_250_2662sqrt10_l2095_209581


namespace NUMINAMATH_GPT_quadratic_sequence_l2095_209571

theorem quadratic_sequence (a x₁ b x₂ c : ℝ)
  (h₁ : a + b = 2 * x₁)
  (h₂ : x₁ + x₂ = 2 * b)
  (h₃ : a + c = 2 * b)
  (h₄ : x₁ + x₂ = -6 / a)
  (h₅ : x₁ * x₂ = c / a) :
  b = -2 * a ∧ c = -5 * a :=
by
  sorry

end NUMINAMATH_GPT_quadratic_sequence_l2095_209571


namespace NUMINAMATH_GPT_number_of_circles_l2095_209548

theorem number_of_circles (side : ℝ) (enclosed_area : ℝ) (num_circles : ℕ) (radius : ℝ) :
  side = 14 ∧ enclosed_area = 42.06195997410015 ∧ 2 * radius = side ∧ π * radius^2 = 49 * π → num_circles = 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_number_of_circles_l2095_209548


namespace NUMINAMATH_GPT_calculate_difference_square_l2095_209563

theorem calculate_difference_square (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 15) : (x - y)^2 = 4 :=
by sorry

end NUMINAMATH_GPT_calculate_difference_square_l2095_209563


namespace NUMINAMATH_GPT_square_paintings_size_l2095_209505

theorem square_paintings_size (total_area : ℝ) (small_paintings_count : ℕ) (small_painting_area : ℝ) 
                              (large_painting_area : ℝ) (square_paintings_count : ℕ) (square_paintings_total_area : ℝ) : 
  total_area = small_paintings_count * small_painting_area + large_painting_area + square_paintings_total_area → 
  square_paintings_count = 3 → 
  small_paintings_count = 4 → 
  small_painting_area = 2 * 3 → 
  large_painting_area = 10 * 15 → 
  square_paintings_total_area = 3 * 6^2 → 
  ∃ side_length, side_length^2 = (square_paintings_total_area / square_paintings_count) ∧ side_length = 6 := 
by
  intro h_total h_square_count h_small_count h_small_area h_large_area h_square_total 
  use 6
  sorry

end NUMINAMATH_GPT_square_paintings_size_l2095_209505


namespace NUMINAMATH_GPT_total_handshakes_l2095_209589

theorem total_handshakes (twins_num : ℕ) (triplets_num : ℕ) (twins_sets : ℕ) (triplets_sets : ℕ) (h_twins : twins_sets = 9) (h_triplets : triplets_sets = 6) (h_twins_num : twins_num = 2 * twins_sets) (h_triplets_num: triplets_num = 3 * triplets_sets) (h_handshakes : twins_num * (twins_num - 2) + triplets_num * (triplets_num - 3) + 2 * twins_num * (triplets_num / 2) = 882): 
  (twins_num * (twins_num - 2) + triplets_num * (triplets_num - 3) + 2 * twins_num * (triplets_num / 2)) / 2 = 441 :=
by
  sorry

end NUMINAMATH_GPT_total_handshakes_l2095_209589


namespace NUMINAMATH_GPT_inverse_g_of_87_l2095_209507

noncomputable def g (x : ℝ) : ℝ := 3 * x^3 + 6

theorem inverse_g_of_87 : (g x = 87) → (x = 3) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_inverse_g_of_87_l2095_209507


namespace NUMINAMATH_GPT_max_sequence_length_l2095_209572

theorem max_sequence_length (a : ℕ → ℝ) (n : ℕ)
  (H1 : ∀ k : ℕ, k + 4 < n → (a k + a (k+1) + a (k+2) + a (k+3) + a (k+4)) < 0)
  (H2 : ∀ k : ℕ, k + 8 < n → (a k + a (k+1) + a (k+2) + a (k+3) + a (k+4) + a (k+5) + a (k+6) + a (k+7) + a (k+8)) > 0) : 
  n ≤ 12 :=
sorry

end NUMINAMATH_GPT_max_sequence_length_l2095_209572


namespace NUMINAMATH_GPT_triangles_in_extended_figure_l2095_209558

theorem triangles_in_extended_figure : 
  ∀ (row1_tri : ℕ) (row2_tri : ℕ) (row3_tri : ℕ) (row4_tri : ℕ) 
  (row1_2_med_tri : ℕ) (row2_3_med_tri : ℕ) (row3_4_med_tri : ℕ) 
  (large_tri : ℕ), 
  row1_tri = 6 →
  row2_tri = 5 →
  row3_tri = 4 →
  row4_tri = 3 →
  row1_2_med_tri = 5 →
  row2_3_med_tri = 2 →
  row3_4_med_tri = 1 →
  large_tri = 1 →
  row1_tri + row2_tri + row3_tri + row4_tri
  + row1_2_med_tri + row2_3_med_tri + row3_4_med_tri
  + large_tri = 27 :=
by
  intro row1_tri row2_tri row3_tri row4_tri
  intro row1_2_med_tri row2_3_med_tri row3_4_med_tri
  intro large_tri
  intro h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end NUMINAMATH_GPT_triangles_in_extended_figure_l2095_209558


namespace NUMINAMATH_GPT_sum_of_two_numbers_l2095_209590

variable {x y : ℝ}

theorem sum_of_two_numbers (h1 : x * y = 120) (h2 : x^2 + y^2 = 289) : x + y = 22 :=
sorry

end NUMINAMATH_GPT_sum_of_two_numbers_l2095_209590


namespace NUMINAMATH_GPT_shaded_area_fraction_l2095_209533

theorem shaded_area_fraction (total_grid_squares : ℕ) (number_1_squares : ℕ) (number_9_squares : ℕ) (number_8_squares : ℕ) (partial_squares_1 : ℕ) (partial_squares_2 : ℕ) (partial_squares_3 : ℕ) :
  total_grid_squares = 18 * 8 →
  number_1_squares = 8 →
  number_9_squares = 15 →
  number_8_squares = 16 →
  partial_squares_1 = 6 →
  partial_squares_2 = 6 →
  partial_squares_3 = 8 →
  (2 * (number_1_squares + number_9_squares + number_9_squares + number_8_squares) + (partial_squares_1 + partial_squares_2 + partial_squares_3)) = 2 * (74 : ℕ) →
  (74 / 144 : ℚ) = 37 / 72 :=
by
  intros _ _ _ _ _ _ _ _
  sorry

end NUMINAMATH_GPT_shaded_area_fraction_l2095_209533


namespace NUMINAMATH_GPT_value_of_alpha_beta_l2095_209509

variable (α β : ℝ)

-- Conditions
def quadratic_eq (x: ℝ) : Prop := x^2 + 2*x - 2005 = 0

-- Lean 4 statement
theorem value_of_alpha_beta 
  (hα : quadratic_eq α) 
  (hβ : quadratic_eq β)
  (sum_roots : α + β = -2) :
  α^2 + 3*α + β = 2003 :=
sorry

end NUMINAMATH_GPT_value_of_alpha_beta_l2095_209509


namespace NUMINAMATH_GPT_not_both_hit_prob_l2095_209587

-- Defining the probabilities
def prob_archer_A_hits : ℚ := 1 / 3
def prob_archer_B_hits : ℚ := 1 / 2

-- Defining event B as both hit the bullseye
def prob_both_hit : ℚ := prob_archer_A_hits * prob_archer_B_hits

-- Defining the complementary event of not both hitting the bullseye
def prob_not_both_hit : ℚ := 1 - prob_both_hit

theorem not_both_hit_prob : prob_not_both_hit = 5 / 6 := by
  -- This is the statement we are trying to prove.
  sorry

end NUMINAMATH_GPT_not_both_hit_prob_l2095_209587


namespace NUMINAMATH_GPT_justin_current_age_l2095_209501

theorem justin_current_age (angelina_future_age : ℕ) (years_until_future : ℕ) (age_difference : ℕ)
  (h_future_age : angelina_future_age = 40) (h_years_until_future : years_until_future = 5)
  (h_age_difference : age_difference = 4) : 
  (angelina_future_age - years_until_future) - age_difference = 31 :=
by
  -- This is where the proof would go.
  sorry

end NUMINAMATH_GPT_justin_current_age_l2095_209501


namespace NUMINAMATH_GPT_how_many_toys_l2095_209565

theorem how_many_toys (initial_savings : ℕ) (allowance : ℕ) (toy_cost : ℕ)
  (h1 : initial_savings = 21)
  (h2 : allowance = 15)
  (h3 : toy_cost = 6) :
  (initial_savings + allowance) / toy_cost = 6 :=
by
  sorry

end NUMINAMATH_GPT_how_many_toys_l2095_209565


namespace NUMINAMATH_GPT_min_value_x2_2xy_y2_l2095_209525

theorem min_value_x2_2xy_y2 (x y : ℝ) : ∃ (a b : ℝ), (x = a ∧ y = b) → x^2 + 2*x*y + y^2 = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_value_x2_2xy_y2_l2095_209525


namespace NUMINAMATH_GPT_student_rank_left_l2095_209582

theorem student_rank_left {n m : ℕ} (h1 : n = 10) (h2 : m = 6) : (n - m + 1) = 5 := by
  sorry

end NUMINAMATH_GPT_student_rank_left_l2095_209582


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l2095_209599

theorem quadratic_inequality_solution (x : ℝ) :
  (x < -7 ∨ x > 3) → x^2 + 4 * x - 21 > 0 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l2095_209599


namespace NUMINAMATH_GPT_inequality_solution_l2095_209553

theorem inequality_solution (x : ℝ) (h : 4 ≤ |x + 2| ∧ |x + 2| ≤ 8) :
  (-10 : ℝ) ≤ x ∧ x ≤ -6 ∨ (2 : ℝ) ≤ x ∧ x ≤ 6 :=
sorry

end NUMINAMATH_GPT_inequality_solution_l2095_209553


namespace NUMINAMATH_GPT_find_least_positive_x_l2095_209584

theorem find_least_positive_x :
  ∃ x : ℕ, x + 5419 ≡ 3789 [MOD 15] ∧ x = 5 :=
by
  use 5
  constructor
  · sorry
  · rfl

end NUMINAMATH_GPT_find_least_positive_x_l2095_209584


namespace NUMINAMATH_GPT_value_of_expression_l2095_209564

theorem value_of_expression (a b : ℝ) (h1 : a^2 + 2012 * a + 1 = 0) (h2 : b^2 + 2012 * b + 1 = 0) :
  (2 + 2013 * a + a^2) * (2 + 2013 * b + b^2) = -2010 := 
  sorry

end NUMINAMATH_GPT_value_of_expression_l2095_209564


namespace NUMINAMATH_GPT_pencil_pen_eraser_cost_l2095_209518

-- Define the problem conditions and question
theorem pencil_pen_eraser_cost 
  (p q : ℝ)
  (h1 : 3 * p + 2 * q = 4.10)
  (h2 : 2 * p + 3 * q = 3.70) :
  p + q + 0.85 = 2.41 :=
sorry

end NUMINAMATH_GPT_pencil_pen_eraser_cost_l2095_209518


namespace NUMINAMATH_GPT_area_of_original_triangle_l2095_209575

variable (H : ℝ) (H' : ℝ := 0.65 * H) 
variable (A' : ℝ := 14.365)
variable (k : ℝ := 0.65) 
variable (A : ℝ)

theorem area_of_original_triangle (h₁ : H' = k * H) (h₂ : A' = 14.365) (h₃ : k = 0.65) : A = 34 := by
  sorry

end NUMINAMATH_GPT_area_of_original_triangle_l2095_209575


namespace NUMINAMATH_GPT_luke_games_l2095_209597

theorem luke_games (F G : ℕ) (H1 : G = 2) (H2 : F + G - 2 = 2) : F = 2 := by
  sorry

end NUMINAMATH_GPT_luke_games_l2095_209597


namespace NUMINAMATH_GPT_price_of_large_pizza_l2095_209556

variable {price_small_pizza : ℕ}
variable {total_revenue : ℕ}
variable {small_pizzas_sold : ℕ}
variable {large_pizzas_sold : ℕ}
variable {price_large_pizza : ℕ}

theorem price_of_large_pizza
  (h1 : price_small_pizza = 2)
  (h2 : total_revenue = 40)
  (h3 : small_pizzas_sold = 8)
  (h4 : large_pizzas_sold = 3) :
  price_large_pizza = 8 :=
by
  sorry

end NUMINAMATH_GPT_price_of_large_pizza_l2095_209556


namespace NUMINAMATH_GPT_exists_numbering_for_nonagon_no_numbering_for_decagon_l2095_209530

-- Definitions for the problem setup
variable (n : ℕ) 
variable (A : Fin n → Point)
variable (O : Point)

-- Definition for the numbering function
variable (f : Fin (2 * n) → ℕ)

-- First statement for n = 9
theorem exists_numbering_for_nonagon :
  ∃ (f : Fin 18 → ℕ), (∀ i : Fin 9, f (i : Fin 9) + f (i + 9) + f ((i + 1) % 9) = 15) :=
sorry

-- Second statement for n = 10
theorem no_numbering_for_decagon :
  ¬ ∃ (f : Fin 20 → ℕ), (∀ i : Fin 10, f (i : Fin 10) + f (i + 10) + f ((i + 1) % 10) = 16) :=
sorry

end NUMINAMATH_GPT_exists_numbering_for_nonagon_no_numbering_for_decagon_l2095_209530


namespace NUMINAMATH_GPT_correct_calculation_l2095_209504

theorem correct_calculation (x : ℝ) (h : 3 * x - 12 = 60) : (x / 3) + 12 = 20 :=
by 
  sorry

end NUMINAMATH_GPT_correct_calculation_l2095_209504


namespace NUMINAMATH_GPT_find_c_l2095_209514

-- Define the problem
def parabola (x y : ℝ) (a : ℝ) : Prop := 
  x = a * (y - 3) ^ 2 + 5

def point (x y : ℝ) (a : ℝ) : Prop := 
  7 = a * (6 - 3) ^ 2 + 5

-- Theorem to be proved
theorem find_c (a : ℝ) (c : ℝ) (h1 : parabola 7 6 a) (h2 : point 7 6 a) : c = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_c_l2095_209514


namespace NUMINAMATH_GPT_point_in_fourth_quadrant_l2095_209508

theorem point_in_fourth_quadrant (m : ℝ) : 0 < m ∧ 2 - m < 0 ↔ m > 2 := 
by 
  sorry

end NUMINAMATH_GPT_point_in_fourth_quadrant_l2095_209508


namespace NUMINAMATH_GPT_find_y_given_conditions_l2095_209580

def is_value_y (x y : ℕ) : Prop :=
  (100 + 200 + 300 + x) / 4 = 250 ∧ (300 + 150 + 100 + x + y) / 5 = 200

theorem find_y_given_conditions : ∃ y : ℕ, ∀ x : ℕ, (100 + 200 + 300 + x) / 4 = 250 ∧ (300 + 150 + 100 + x + y) / 5 = 200 → y = 50 :=
by
  sorry

end NUMINAMATH_GPT_find_y_given_conditions_l2095_209580


namespace NUMINAMATH_GPT_total_marbles_l2095_209536

-- Definitions to state the problem
variables {r b g : ℕ}
axiom ratio_condition : r / b = 2 / 4 ∧ r / g = 2 / 6
axiom blue_marbles : b = 30

-- Theorem statement
theorem total_marbles : r + b + g = 90 :=
by sorry

end NUMINAMATH_GPT_total_marbles_l2095_209536


namespace NUMINAMATH_GPT_value_of_x_l2095_209578

theorem value_of_x (x : ℝ) (h₁ : x > 0) (h₂ : x^3 = 19683) : x = 27 :=
sorry

end NUMINAMATH_GPT_value_of_x_l2095_209578


namespace NUMINAMATH_GPT_part1_part2_l2095_209595

noncomputable def Sn (a : ℕ → ℚ) (n : ℕ) (p : ℚ) : ℚ :=
4 * a n - p

theorem part1 (a : ℕ → ℚ) (S : ℕ → ℚ) (p : ℚ) (hp : p ≠ 0)
  (hS : ∀ n, S n = Sn a n p) : 
  ∃ q, ∀ n, a (n + 1) = q * a n :=
sorry

noncomputable def an_formula (n : ℕ) : ℚ := (4/3)^(n - 1)

theorem part2 (b : ℕ → ℚ) (a : ℕ → ℚ)
  (p : ℚ) (hp : p = 3)
  (hb : b 1 = 2)
  (ha1 : a 1 = 1) 
  (h_rec : ∀ n, b (n + 1) = b n + a n) :
  ∀ n, b n = 3 * ((4/3)^(n - 1)) - 1 :=
sorry

end NUMINAMATH_GPT_part1_part2_l2095_209595


namespace NUMINAMATH_GPT_tan_alpha_value_l2095_209583

theorem tan_alpha_value (α β : ℝ) (h₁ : Real.tan (α + β) = 3) (h₂ : Real.tan β = 2) : 
  Real.tan α = 1 / 7 := 
by 
  sorry

end NUMINAMATH_GPT_tan_alpha_value_l2095_209583


namespace NUMINAMATH_GPT_volunteers_meet_again_in_360_days_l2095_209588

-- Definitions of the given values for the problem
def ella_days := 5
def fiona_days := 6
def george_days := 8
def harry_days := 9

-- Statement of the problem in Lean 4
theorem volunteers_meet_again_in_360_days :
  Nat.lcm (Nat.lcm (Nat.lcm ella_days fiona_days) george_days) harry_days = 360 :=
by
  sorry

end NUMINAMATH_GPT_volunteers_meet_again_in_360_days_l2095_209588


namespace NUMINAMATH_GPT_sin_sum_diff_l2095_209561

theorem sin_sum_diff (α β : ℝ) 
  (hα : Real.sin α = 1/3) 
  (hβ : Real.sin β = 1/2) : 
  Real.sin (α + β) * Real.sin (α - β) = -5/36 := 
sorry

end NUMINAMATH_GPT_sin_sum_diff_l2095_209561


namespace NUMINAMATH_GPT_sample_size_l2095_209591

theorem sample_size (n : ℕ) (h_ratio : 2 + 3 + 5 = 10) (h_sample : 8 = n * 2 / 10) : n = 40 :=
by
  sorry

end NUMINAMATH_GPT_sample_size_l2095_209591


namespace NUMINAMATH_GPT_inverse_matrix_correct_l2095_209539

def A : Matrix (Fin 3) (Fin 3) ℚ :=
  ![
    ![1, 2, 3],
    ![0, -1, 2],
    ![3, 0, 7]
  ]

def A_inv_correct : Matrix (Fin 3) (Fin 3) ℚ :=
  ![
    ![-1/2, -1, 1/2],
    ![3/7, -1/7, -1/7],
    ![3/14, 3/7, -1/14]
  ]

theorem inverse_matrix_correct : A⁻¹ = A_inv_correct := by
  sorry

end NUMINAMATH_GPT_inverse_matrix_correct_l2095_209539


namespace NUMINAMATH_GPT_sum_of_cubes_eq_91_l2095_209549

theorem sum_of_cubes_eq_91 (a b : ℤ) (h₁ : a^3 + b^3 = 91) (h₂ : a * b = 12) : a^3 + b^3 = 91 :=
by
  exact h₁

end NUMINAMATH_GPT_sum_of_cubes_eq_91_l2095_209549


namespace NUMINAMATH_GPT_pencils_initial_count_l2095_209545

theorem pencils_initial_count (pencils_given : ℕ) (pencils_left : ℕ) (initial_pencils : ℕ) :
  pencils_given = 31 → pencils_left = 111 → initial_pencils = pencils_given + pencils_left → initial_pencils = 142 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_pencils_initial_count_l2095_209545


namespace NUMINAMATH_GPT_perimeter_of_cube_face_is_28_l2095_209569

-- Define the volume of the cube
def volume_of_cube : ℝ := 343

-- Define the side length of the cube based on the volume
def side_length_of_cube : ℝ := volume_of_cube^(1/3)

-- Define the perimeter of one face of the cube
def perimeter_of_one_face (side_length : ℝ) : ℝ := 4 * side_length

-- Theorem: Prove the perimeter of one face of the cube is 28 cm given the volume is 343 cm³
theorem perimeter_of_cube_face_is_28 : 
  perimeter_of_one_face side_length_of_cube = 28 := 
by
  sorry

end NUMINAMATH_GPT_perimeter_of_cube_face_is_28_l2095_209569


namespace NUMINAMATH_GPT_simplify_expression_l2095_209593

theorem simplify_expression :
  (1 / ((3 / (Real.sqrt 5 + 2)) + (4 / (Real.sqrt 6 - 2)))) =
  ((3 * Real.sqrt 5 + 2 * Real.sqrt 6 + 2) / 29) :=
  sorry

end NUMINAMATH_GPT_simplify_expression_l2095_209593


namespace NUMINAMATH_GPT_geometric_series_sum_eq_l2095_209541

theorem geometric_series_sum_eq (a r : ℝ) 
  (h_sum : (∑' n:ℕ, a * r^n) = 20) 
  (h_odd_sum : (∑' n:ℕ, a * r^(2 * n + 1)) = 8) : 
  r = 2 / 3 := 
sorry

end NUMINAMATH_GPT_geometric_series_sum_eq_l2095_209541


namespace NUMINAMATH_GPT_yz_sub_zx_sub_xy_l2095_209552

theorem yz_sub_zx_sub_xy (x y z : ℝ) (h1 : x - y - z = 19) (h2 : x^2 + y^2 + z^2 ≠ 19) :
  yz - zx - xy = 171 := by
  sorry

end NUMINAMATH_GPT_yz_sub_zx_sub_xy_l2095_209552


namespace NUMINAMATH_GPT_sum_geometric_sequence_l2095_209566

theorem sum_geometric_sequence {a : ℕ → ℝ} (ha : ∃ q, ∀ n, a n = 3 * q ^ n)
  (h1 : a 1 = 3) (h2 : a 1 + a 2 + a 3 = 9) :
  a 4 + a 5 + a 6 = 9 ∨ a 4 + a 5 + a 6 = -72 :=
sorry

end NUMINAMATH_GPT_sum_geometric_sequence_l2095_209566


namespace NUMINAMATH_GPT_smallest_a_undefined_inverse_l2095_209542

theorem smallest_a_undefined_inverse (a : ℕ) (ha : a = 2) :
  (∀ (a : ℕ), 0 < a → ((Nat.gcd a 40 > 1) ∧ (Nat.gcd a 90 > 1)) ↔ a = 2) :=
by
  sorry

end NUMINAMATH_GPT_smallest_a_undefined_inverse_l2095_209542


namespace NUMINAMATH_GPT_go_stones_perimeter_count_l2095_209554

def stones_per_side : ℕ := 6
def sides_of_square : ℕ := 4
def corner_stones : ℕ := 4

theorem go_stones_perimeter_count :
  (stones_per_side * sides_of_square) - corner_stones = 20 := 
by
  sorry

end NUMINAMATH_GPT_go_stones_perimeter_count_l2095_209554


namespace NUMINAMATH_GPT_solve_for_x_l2095_209555

theorem solve_for_x (x : ℝ) (hp : 0 < x) (h : 4 * x^2 = 1024) : x = 16 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l2095_209555


namespace NUMINAMATH_GPT_sin_cos_inequality_l2095_209519

open Real

theorem sin_cos_inequality 
  (x : ℝ) (hx : 0 < x ∧ x < π / 2) 
  (m n : ℕ) (hmn : n > m)
  : 2 * abs (sin x ^ n - cos x ^ n) ≤ 3 * abs (sin x ^ m - cos x ^ m) :=
sorry

end NUMINAMATH_GPT_sin_cos_inequality_l2095_209519


namespace NUMINAMATH_GPT_joan_remaining_oranges_l2095_209559

def total_oranges_joan_picked : ℕ := 37
def oranges_sara_sold : ℕ := 10

theorem joan_remaining_oranges : total_oranges_joan_picked - oranges_sara_sold = 27 := by
  sorry

end NUMINAMATH_GPT_joan_remaining_oranges_l2095_209559


namespace NUMINAMATH_GPT_matrix_equation_l2095_209592

def M : Matrix (Fin 2) (Fin 2) ℤ := ![![4, 5], ![-6, -2]]
def p : ℤ := 2
def q : ℤ := -18

theorem matrix_equation :
  M * M = p • M + q • (1 : Matrix (Fin 2) (Fin 2) ℤ) :=
by
  sorry

end NUMINAMATH_GPT_matrix_equation_l2095_209592


namespace NUMINAMATH_GPT_find_x_l2095_209551

open Real

noncomputable def log_base (b x : ℝ) : ℝ := log x / log b

theorem find_x :
  ∃ x : ℝ, 0 < x ∧
  log_base 5 (x - 1) + log_base (sqrt 5) (x^2 - 1) + log_base (1/5) (x - 1) = 3 ∧
  x = sqrt (5 * sqrt 5 + 1) :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2095_209551


namespace NUMINAMATH_GPT_symmetric_y_axis_function_l2095_209528

theorem symmetric_y_axis_function (f g : ℝ → ℝ) (h : ∀ (x : ℝ), g x = 3^x + 1) :
  (∀ x, f x = f (-x)) → (∀ x, f x = g (-x)) → (∀ x, f x = 3^(-x) + 1) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_symmetric_y_axis_function_l2095_209528


namespace NUMINAMATH_GPT_sam_investment_l2095_209546

noncomputable def compound_interest (P: ℝ) (r: ℝ) (n: ℕ) (t: ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem sam_investment :
  compound_interest 3000 0.10 4 1 = 3311.44 :=
by
  sorry

end NUMINAMATH_GPT_sam_investment_l2095_209546


namespace NUMINAMATH_GPT_Mike_got_18_cards_l2095_209585

theorem Mike_got_18_cards (original_cards : ℕ) (total_cards : ℕ) : 
  original_cards = 64 → total_cards = 82 → total_cards - original_cards = 18 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_Mike_got_18_cards_l2095_209585


namespace NUMINAMATH_GPT_car_robot_collections_l2095_209517

variable (t m b s j : ℕ)

axiom tom_has_15 : t = 15
axiom michael_robots : m = 3 * t - 5
axiom bob_robots : b = 8 * (t + m)
axiom sarah_robots : s = b / 2 - 7
axiom jane_robots : j = (s - t) / 3

theorem car_robot_collections :
  t = 15 ∧
  m = 40 ∧
  b = 440 ∧
  s = 213 ∧
  j = 66 :=
  by
    sorry

end NUMINAMATH_GPT_car_robot_collections_l2095_209517


namespace NUMINAMATH_GPT_valid_words_count_l2095_209598

noncomputable def count_valid_words : Nat :=
  let total_possible_words : Nat := ((25^1) + (25^2) + (25^3) + (25^4) + (25^5))
  let total_possible_words_without_B : Nat := ((24^1) + (24^2) + (24^3) + (24^4) + (24^5))
  total_possible_words - total_possible_words_without_B

theorem valid_words_count : count_valid_words = 1864701 :=
by
  let total_1_letter_words := 25^1
  let total_2_letter_words := 25^2
  let total_3_letter_words := 25^3
  let total_4_letter_words := 25^4
  let total_5_letter_words := 25^5

  let total_words_without_B_1_letter := 24^1
  let total_words_without_B_2_letter := 24^2
  let total_words_without_B_3_letter := 24^3
  let total_words_without_B_4_letter := 24^4
  let total_words_without_B_5_letter := 24^5

  let valid_1_letter_words := total_1_letter_words - total_words_without_B_1_letter
  let valid_2_letter_words := total_2_letter_words - total_words_without_B_2_letter
  let valid_3_letter_words := total_3_letter_words - total_words_without_B_3_letter
  let valid_4_letter_words := total_4_letter_words - total_words_without_B_4_letter
  let valid_5_letter_words := total_5_letter_words - total_words_without_B_5_letter

  let valid_words := valid_1_letter_words + valid_2_letter_words + valid_3_letter_words + valid_4_letter_words + valid_5_letter_words
  sorry

end NUMINAMATH_GPT_valid_words_count_l2095_209598


namespace NUMINAMATH_GPT_obtuse_triangle_side_range_l2095_209573

theorem obtuse_triangle_side_range (a : ℝ) (h1 : 0 < a)
  (h2 : a + (a + 1) > a + 2)
  (h3 : (a + 1) + (a + 2) > a)
  (h4 : (a + 2) + a > a + 1)
  (h5 : (a + 2)^2 > a^2 + (a + 1)^2) : 1 < a ∧ a < 3 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_obtuse_triangle_side_range_l2095_209573


namespace NUMINAMATH_GPT_pipe_A_fill_time_l2095_209527

theorem pipe_A_fill_time :
  (∃ x : ℕ, (1 / (x : ℝ) + 1 / 60 - 1 / 72 = 1 / 40) ∧ x = 45) :=
sorry

end NUMINAMATH_GPT_pipe_A_fill_time_l2095_209527


namespace NUMINAMATH_GPT_boxes_neither_markers_nor_crayons_l2095_209513

theorem boxes_neither_markers_nor_crayons (total boxes_markers boxes_crayons boxes_both: ℕ)
  (htotal : total = 15)
  (hmarkers : boxes_markers = 9)
  (hcrayons : boxes_crayons = 4)
  (hboth : boxes_both = 5) :
  total - (boxes_markers + boxes_crayons - boxes_both) = 7 := by
  sorry

end NUMINAMATH_GPT_boxes_neither_markers_nor_crayons_l2095_209513


namespace NUMINAMATH_GPT_area_of_ABCD_l2095_209503

noncomputable def AB := 6
noncomputable def BC := 8
noncomputable def CD := 15
noncomputable def DA := 17
def right_angle_BCD := true
def convex_ABCD := true

theorem area_of_ABCD : ∃ area : ℝ, area = 110 := by
  -- Given conditions
  have hAB : AB = 6 := rfl
  have hBC : BC = 8 := rfl
  have hCD : CD = 15 := rfl
  have hDA : DA = 17 := rfl
  have hAngle : right_angle_BCD = true := rfl
  have hConvex : convex_ABCD = true := rfl

  -- skip the proof
  sorry

end NUMINAMATH_GPT_area_of_ABCD_l2095_209503


namespace NUMINAMATH_GPT_part1_part2_l2095_209529

section
variable (x y : ℝ)

def A : ℝ := 3 * x^2 + 2 * y^2 - 2 * x * y
def B : ℝ := y^2 - x * y + 2 * x^2

-- Part (1): Prove that 2A - 3B = y^2 - xy
theorem part1 : 2 * A x y - 3 * B x y = y^2 - x * y := 
sorry

-- Part (2): Given |2x - 3| + (y + 2)^2 = 0, prove that 2A - 3B = 7
theorem part2 (h : |2 * x - 3| + (y + 2)^2 = 0) : 2 * A x y - 3 * B x y = 7 :=
sorry

end

end NUMINAMATH_GPT_part1_part2_l2095_209529


namespace NUMINAMATH_GPT_sector_area_l2095_209562

theorem sector_area (r : ℝ) (θ : ℝ) (h_r : r = 10) (h_θ : θ = 42) : 
  (θ / 360) * Real.pi * r^2 = (35 * Real.pi) / 3 :=
by
  -- Using the provided conditions to simplify the expression
  rw [h_r, h_θ]
  -- Simplify and solve the expression
  sorry

end NUMINAMATH_GPT_sector_area_l2095_209562


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2095_209521

theorem solution_set_of_inequality (x : ℝ) : 
  |x + 3| - |x - 2| ≥ 3 ↔ x ≥ 1 :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2095_209521
