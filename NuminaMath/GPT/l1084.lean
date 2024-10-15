import Mathlib

namespace NUMINAMATH_GPT_fourth_root_of_expression_l1084_108434

theorem fourth_root_of_expression (x : ℝ) (h : 0 < x) : Real.sqrt (x^3 * Real.sqrt (x^2)) ^ (1 / 4) = x := sorry

end NUMINAMATH_GPT_fourth_root_of_expression_l1084_108434


namespace NUMINAMATH_GPT_geometric_sequence_a5_value_l1084_108407

theorem geometric_sequence_a5_value :
  ∃ (a : ℕ → ℝ) (r : ℝ), (a 3)^2 - 4 * a 3 + 3 = 0 ∧ 
                         (a 7)^2 - 4 * a 7 + 3 = 0 ∧ 
                         (a 3) * (a 7) = 3 ∧ 
                         (a 3) + (a 7) = 4 ∧ 
                         a 5 = (a 3 * a 7).sqrt :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a5_value_l1084_108407


namespace NUMINAMATH_GPT_equivalent_trigonometric_identity_l1084_108496

variable (α : ℝ)

theorem equivalent_trigonometric_identity
  (h1 : α ∈ Set.Ioo (-(Real.pi/2)) 0)
  (h2 : Real.sin (α + (Real.pi/4)) = -1/3) :
  (Real.sin (2*α) / Real.cos ((Real.pi/4) - α)) = 7/3 := 
by
  sorry

end NUMINAMATH_GPT_equivalent_trigonometric_identity_l1084_108496


namespace NUMINAMATH_GPT_smallest_multiple_of_8_and_9_l1084_108481

theorem smallest_multiple_of_8_and_9 : ∃ n : ℕ, n > 0 ∧ (n % 8 = 0) ∧ (n % 9 = 0) ∧ (∀ m : ℕ, m > 0 ∧ (m % 8 = 0) ∧ (m % 9 = 0) → n ≤ m) ∧ n = 72 :=
by
  sorry

end NUMINAMATH_GPT_smallest_multiple_of_8_and_9_l1084_108481


namespace NUMINAMATH_GPT_perpendicular_lines_condition_l1084_108417

theorem perpendicular_lines_condition (m : ℝ) :
    (m = 1 → (∀ (x y : ℝ), (∀ (c d : ℝ), c * (m * x + y - 1) = 0 → d * (x - m * y - 1) = 0 → (c * m + d / m) ^ 2 = 1))) ∧ (∀ (m' : ℝ), m' ≠ 1 → ¬ (∀ (x y : ℝ), (∀ (c d : ℝ), c * (m' * x + y - 1) = 0 → d * (x - m' * y - 1) = 0 → (c * m' + d / m') ^ 2 = 1))) :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_condition_l1084_108417


namespace NUMINAMATH_GPT_candles_on_rituprts_cake_l1084_108413

theorem candles_on_rituprts_cake (peter_candles : ℕ) (rupert_factor : ℝ) 
  (h_peter : peter_candles = 10) (h_rupert : rupert_factor = 3.5) : 
  ∃ rupert_candles : ℕ, rupert_candles = 35 :=
by
  sorry

end NUMINAMATH_GPT_candles_on_rituprts_cake_l1084_108413


namespace NUMINAMATH_GPT_abs_x_plus_7_eq_0_has_no_solution_l1084_108497

theorem abs_x_plus_7_eq_0_has_no_solution : ¬∃ x : ℝ, |x| + 7 = 0 :=
by
  sorry

end NUMINAMATH_GPT_abs_x_plus_7_eq_0_has_no_solution_l1084_108497


namespace NUMINAMATH_GPT_find_smaller_integer_l1084_108488

noncomputable def average_equals_decimal (m n : ℕ) : Prop :=
  (m + n) / 2 = m + n / 100

theorem find_smaller_integer (m n : ℕ) (h1 : 10 ≤ m ∧ m < 100) (h2 : 10 ≤ n ∧ n < 100) (h3 : 25 ∣ n) (h4 : average_equals_decimal m n) : m = 49 :=
by
  sorry

end NUMINAMATH_GPT_find_smaller_integer_l1084_108488


namespace NUMINAMATH_GPT_slope_of_line_l1084_108420

theorem slope_of_line (x y : ℝ) (h : 2 * y = -3 * x + 6) : (∃ m b : ℝ, y = m * x + b) ∧  (m = -3 / 2) :=
by 
  sorry

end NUMINAMATH_GPT_slope_of_line_l1084_108420


namespace NUMINAMATH_GPT_polygon_with_150_degree_interior_angles_has_12_sides_l1084_108451

-- Define the conditions as Lean definitions
def regular_polygon (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ (a : ℝ), a = 150 ∧ 180 * (n - 2) = 150 * n

-- Theorem statement
theorem polygon_with_150_degree_interior_angles_has_12_sides :
  ∃ (n : ℕ), regular_polygon n ∧ n = 12 := 
by
  sorry

end NUMINAMATH_GPT_polygon_with_150_degree_interior_angles_has_12_sides_l1084_108451


namespace NUMINAMATH_GPT_min_distance_symmetry_l1084_108483

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^2 + x + 1

def line (x y : ℝ) : Prop := 2 * x - y = 3

theorem min_distance_symmetry :
  ∀ (P Q : ℝ × ℝ),
    line P.1 P.2 → line Q.1 Q.2 →
    (exists (x : ℝ), P = (x, f x)) ∧
    (exists (x : ℝ), Q = (x, f x)) →
    ∃ (d : ℝ), d = 2 * Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_min_distance_symmetry_l1084_108483


namespace NUMINAMATH_GPT_remainder_of_addition_and_division_l1084_108446

theorem remainder_of_addition_and_division :
  (3452179 + 50) % 7 = 4 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_addition_and_division_l1084_108446


namespace NUMINAMATH_GPT_total_logs_in_stack_l1084_108472

/-- The total number of logs in a stack where the top row has 5 logs,
each succeeding row has one more log than the one above,
and the bottom row has 15 logs. -/
theorem total_logs_in_stack :
  let a := 5               -- first term (logs in the top row)
  let l := 15              -- last term (logs in the bottom row)
  let n := l - a + 1       -- number of terms (rows)
  let S := n / 2 * (a + l) -- sum of the arithmetic series
  S = 110 := sorry

end NUMINAMATH_GPT_total_logs_in_stack_l1084_108472


namespace NUMINAMATH_GPT_Isabella_paint_area_l1084_108418

def bedroom1_length : ℕ := 14
def bedroom1_width : ℕ := 11
def bedroom1_height : ℕ := 9

def bedroom2_length : ℕ := 13
def bedroom2_width : ℕ := 12
def bedroom2_height : ℕ := 9

def unpaintable_area_per_bedroom : ℕ := 70

theorem Isabella_paint_area :
  let wall_area (length width height : ℕ) := 2 * (length * height) + 2 * (width * height)
  let paintable_area (length width height : ℕ) := wall_area length width height - unpaintable_area_per_bedroom
  paintable_area bedroom1_length bedroom1_width bedroom1_height +
  paintable_area bedroom1_length bedroom1_width bedroom1_height +
  paintable_area bedroom2_length bedroom2_width bedroom2_height +
  paintable_area bedroom2_length bedroom2_width bedroom2_height =
  1520 := 
by
  sorry

end NUMINAMATH_GPT_Isabella_paint_area_l1084_108418


namespace NUMINAMATH_GPT_avg_salary_increase_l1084_108402

theorem avg_salary_increase (A1 : ℝ) (M : ℝ) (n : ℕ) (N : ℕ) 
  (h1 : n = 20) (h2 : A1 = 1500) (h3 : M = 4650) (h4 : N = n + 1) :
  (20 * A1 + M) / N - A1 = 150 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_avg_salary_increase_l1084_108402


namespace NUMINAMATH_GPT_differential_savings_l1084_108456

theorem differential_savings (income : ℕ) (tax_rate_before : ℝ) (tax_rate_after : ℝ) : 
  income = 36000 → tax_rate_before = 0.46 → tax_rate_after = 0.32 →
  ((income * tax_rate_before) - (income * tax_rate_after)) = 5040 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_differential_savings_l1084_108456


namespace NUMINAMATH_GPT_tan_neg_five_pi_over_four_l1084_108465

theorem tan_neg_five_pi_over_four : Real.tan (-5 * Real.pi / 4) = -1 :=
  sorry

end NUMINAMATH_GPT_tan_neg_five_pi_over_four_l1084_108465


namespace NUMINAMATH_GPT_solve_for_x_l1084_108453

-- Defining the given conditions
def y : ℕ := 6
def lhs (x : ℕ) : ℕ := Nat.pow x y
def rhs : ℕ := Nat.pow 3 12

-- Theorem statement to prove
theorem solve_for_x (x : ℕ) (hypothesis : lhs x = rhs) : x = 9 :=
by sorry

end NUMINAMATH_GPT_solve_for_x_l1084_108453


namespace NUMINAMATH_GPT_rainfall_mondays_l1084_108443

theorem rainfall_mondays
  (M : ℕ)
  (rain_monday : ℝ)
  (rain_tuesday : ℝ)
  (num_tuesdays : ℕ)
  (extra_rain_tuesdays : ℝ)
  (h1 : rain_monday = 1.5)
  (h2 : rain_tuesday = 2.5)
  (h3 : num_tuesdays = 9)
  (h4 : num_tuesdays * rain_tuesday = rain_monday * M + extra_rain_tuesdays)
  (h5 : extra_rain_tuesdays = 12) :
  M = 7 := 
sorry

end NUMINAMATH_GPT_rainfall_mondays_l1084_108443


namespace NUMINAMATH_GPT_pyramid_height_correct_l1084_108487

noncomputable def pyramid_height : ℝ :=
  let ab := 15 * Real.sqrt 3
  let bc := 14 * Real.sqrt 3
  let base_area := ab * bc
  let volume := 750
  let height := 3 * volume / base_area
  height

theorem pyramid_height_correct : pyramid_height = 25 / 7 :=
by
  sorry

end NUMINAMATH_GPT_pyramid_height_correct_l1084_108487


namespace NUMINAMATH_GPT_wire_cut_min_area_l1084_108414

theorem wire_cut_min_area :
  ∃ x : ℝ, 0 < x ∧ x < 100 ∧ S = π * (x / (2 * π))^2 + ((100 - x) / 4)^2 ∧ 
  (∀ y : ℝ, 0 < y ∧ y < 100 → (π * (y / (2 * π))^2 + ((100 - y) / 4)^2 ≥ S)) ∧
  x = 100 * π / (16 + π) :=
sorry

end NUMINAMATH_GPT_wire_cut_min_area_l1084_108414


namespace NUMINAMATH_GPT_find_abc_l1084_108431

theorem find_abc (a b c : ℕ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : (a-1) * (b-1) * (c-1) ∣ a * b * c - 1) :
  (a = 3 ∧ b = 5 ∧ c = 15) ∨ (a = 2 ∧ b = 4 ∧ c = 8) :=
by 
  sorry

end NUMINAMATH_GPT_find_abc_l1084_108431


namespace NUMINAMATH_GPT_match_graph_l1084_108475

theorem match_graph (x : ℝ) (h : x ≤ 0) : 
  Real.sqrt (-2 * x^3) = -x * Real.sqrt (-2 * x) :=
by
  sorry

end NUMINAMATH_GPT_match_graph_l1084_108475


namespace NUMINAMATH_GPT_p_sufficient_for_q_q_not_necessary_for_p_l1084_108467

variable (x : ℝ)

def p := |x - 2| < 1
def q := 1 < x ∧ x < 5

theorem p_sufficient_for_q : p x → q x :=
by sorry

theorem q_not_necessary_for_p : ¬ (q x → p x) :=
by sorry

end NUMINAMATH_GPT_p_sufficient_for_q_q_not_necessary_for_p_l1084_108467


namespace NUMINAMATH_GPT_total_cantaloupes_l1084_108457

theorem total_cantaloupes (fred_cantaloupes : ℕ) (tim_cantaloupes : ℕ) (h1 : fred_cantaloupes = 38) (h2 : tim_cantaloupes = 44) : fred_cantaloupes + tim_cantaloupes = 82 :=
by
  sorry

end NUMINAMATH_GPT_total_cantaloupes_l1084_108457


namespace NUMINAMATH_GPT_total_simple_interest_l1084_108438

theorem total_simple_interest (P R T : ℝ) (hP : P = 6178.846153846154) (hR : R = 0.13) (hT : T = 5) :
    P * R * T = 4011.245192307691 := by
  rw [hP, hR, hT]
  norm_num
  sorry

end NUMINAMATH_GPT_total_simple_interest_l1084_108438


namespace NUMINAMATH_GPT_true_proposition_among_provided_l1084_108468

theorem true_proposition_among_provided :
  ∃ (x0 : ℝ), |x0| ≤ 0 :=
by
  exists 0
  simp

end NUMINAMATH_GPT_true_proposition_among_provided_l1084_108468


namespace NUMINAMATH_GPT_train_cross_bridge_time_l1084_108473

noncomputable def length_train : ℝ := 130
noncomputable def length_bridge : ℝ := 320
noncomputable def speed_kmh : ℝ := 54
noncomputable def speed_ms : ℝ := speed_kmh * 1000 / 3600

theorem train_cross_bridge_time :
  (length_train + length_bridge) / speed_ms = 30 := by
  sorry

end NUMINAMATH_GPT_train_cross_bridge_time_l1084_108473


namespace NUMINAMATH_GPT_yi_reads_more_than_jia_by_9_pages_l1084_108454

-- Define the number of pages in the book
def total_pages : ℕ := 120

-- Define number of pages read per day by Jia and Yi
def pages_per_day_jia : ℕ := 8
def pages_per_day_yi : ℕ := 13

-- Define the number of days in the period
def total_days : ℕ := 7

-- Calculate total pages read by Jia in the given period
def pages_read_by_jia : ℕ := total_days * pages_per_day_jia

-- Calculate the number of reading days by Yi in the given period
def reading_days_yi : ℕ := (total_days / 3) * 2 + (total_days % 3).min 2

-- Calculate total pages read by Yi in the given period
def pages_read_by_yi : ℕ := reading_days_yi * pages_per_day_yi

-- Given all conditions, prove that Yi reads 9 pages more than Jia over the 7-day period
theorem yi_reads_more_than_jia_by_9_pages :
  pages_read_by_yi - pages_read_by_jia = 9 :=
by
  sorry

end NUMINAMATH_GPT_yi_reads_more_than_jia_by_9_pages_l1084_108454


namespace NUMINAMATH_GPT_stones_on_one_side_l1084_108435

theorem stones_on_one_side (total_perimeter_stones : ℕ) (h : total_perimeter_stones = 84) :
  ∃ s : ℕ, 4 * s - 4 = total_perimeter_stones ∧ s = 22 :=
by
  use 22
  sorry

end NUMINAMATH_GPT_stones_on_one_side_l1084_108435


namespace NUMINAMATH_GPT_part1_part2_l1084_108410

theorem part1 : (π - 3)^0 + (-1)^(2023) - Real.sqrt 8 = -2 * Real.sqrt 2 := sorry

theorem part2 (x : ℝ) : (4 * x - 3 > 9) ∧ (2 + x ≥ 0) ↔ x > 3 := sorry

end NUMINAMATH_GPT_part1_part2_l1084_108410


namespace NUMINAMATH_GPT_number_of_solutions_l1084_108489

theorem number_of_solutions (x : ℤ) (h1 : 0 < x) (h2 : x < 150) (h3 : (x + 17) % 46 = 75 % 46) : 
  ∃ n : ℕ, n = 3 :=
sorry

end NUMINAMATH_GPT_number_of_solutions_l1084_108489


namespace NUMINAMATH_GPT_donuts_percentage_missing_l1084_108441

noncomputable def missing_donuts_percentage (initial_donuts : ℕ) (remaining_donuts : ℕ) : ℝ :=
  ((initial_donuts - remaining_donuts : ℕ) : ℝ) / initial_donuts * 100

theorem donuts_percentage_missing
  (h_initial : ℕ := 30)
  (h_remaining : ℕ := 9) :
  missing_donuts_percentage h_initial h_remaining = 70 :=
by
  sorry

end NUMINAMATH_GPT_donuts_percentage_missing_l1084_108441


namespace NUMINAMATH_GPT_unique_solution_3_pow_x_minus_2_pow_y_eq_7_l1084_108448

theorem unique_solution_3_pow_x_minus_2_pow_y_eq_7 :
  ∀ x y : ℕ, (1 ≤ x) → (1 ≤ y) → (3 ^ x - 2 ^ y = 7) → (x = 2 ∧ y = 1) :=
by
  intros x y hx hy hxy
  sorry

end NUMINAMATH_GPT_unique_solution_3_pow_x_minus_2_pow_y_eq_7_l1084_108448


namespace NUMINAMATH_GPT_sin_diff_angle_identity_l1084_108482

open Real

noncomputable def alpha : ℝ := sorry -- α is an obtuse angle

axiom h1 : 90 < alpha ∧ alpha < 180 -- α is an obtuse angle
axiom h2 : cos alpha = -3 / 5 -- given cosine value

theorem sin_diff_angle_identity :
  sin (π / 4 - alpha) = - (7 * sqrt 2) / 10 :=
by
  sorry

end NUMINAMATH_GPT_sin_diff_angle_identity_l1084_108482


namespace NUMINAMATH_GPT_solve_equation_solve_inequality_l1084_108463

-- Defining the first problem
theorem solve_equation (x : ℝ) : 3 * (x - 2) - (1 - 2 * x) = 3 ↔ x = 2 := 
by
  sorry

-- Defining the second problem
theorem solve_inequality (x : ℝ) : (2 * x - 1 < 4 * x + 3) ↔ (x > -2) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_solve_inequality_l1084_108463


namespace NUMINAMATH_GPT_students_prefer_windows_to_mac_l1084_108437

-- Define the conditions
def total_students : ℕ := 210
def students_prefer_mac : ℕ := 60
def students_equally_prefer_both : ℕ := 20
def students_no_preference : ℕ := 90

-- The proof problem
theorem students_prefer_windows_to_mac :
  total_students - students_prefer_mac - students_equally_prefer_both - students_no_preference = 40 :=
by sorry

end NUMINAMATH_GPT_students_prefer_windows_to_mac_l1084_108437


namespace NUMINAMATH_GPT_find_cos_alpha_l1084_108411

variable (α β : ℝ)

-- Conditions
def acute_angles (α β : ℝ) : Prop := 0 < α ∧ α < (Real.pi / 2) ∧ 0 < β ∧ β < (Real.pi / 2)
def cos_alpha_beta : Prop := Real.cos (α + β) = 12 / 13
def cos_2alpha_beta : Prop := Real.cos (2 * α + β) = 3 / 5

-- Main theorem
theorem find_cos_alpha (h1 : acute_angles α β) (h2 : cos_alpha_beta α β) (h3 : cos_2alpha_beta α β) : 
  Real.cos α = 56 / 65 :=
sorry

end NUMINAMATH_GPT_find_cos_alpha_l1084_108411


namespace NUMINAMATH_GPT_convert_base_10_to_base_6_l1084_108419

theorem convert_base_10_to_base_6 : 
  ∃ (digits : List ℕ), (digits.length = 4 ∧
    List.foldr (λ (x : ℕ) (acc : ℕ) => acc * 6 + x) 0 digits = 314 ∧
    digits = [1, 2, 4, 2]) := by
  sorry

end NUMINAMATH_GPT_convert_base_10_to_base_6_l1084_108419


namespace NUMINAMATH_GPT_arithmetic_seq_of_equal_roots_l1084_108466

theorem arithmetic_seq_of_equal_roots (a b c : ℝ) (h : b ≠ 0) 
    (h_eq_roots : ∃ x, b*x^2 - 4*b*x + 2*(a + c) = 0 ∧ (∀ y, b*y^2 - 4*b*y + 2*(a + c) = 0 → x = y)) : 
    b - a = c - b := 
by 
  -- placeholder for proof body
  sorry

end NUMINAMATH_GPT_arithmetic_seq_of_equal_roots_l1084_108466


namespace NUMINAMATH_GPT_minimum_m_n_squared_l1084_108479

theorem minimum_m_n_squared (a b c m n : ℝ) (h1 : c > a) (h2 : c > b) (h3 : c = Real.sqrt (a^2 + b^2)) 
    (h4 : a * m + b * n + c = 0) : m^2 + n^2 ≥ 1 := by
  sorry

end NUMINAMATH_GPT_minimum_m_n_squared_l1084_108479


namespace NUMINAMATH_GPT_find_pairs_l1084_108459

theorem find_pairs (m n : ℕ) : 
  (20^m - 10 * m^2 + 1 = 19^n ↔ (m = 0 ∧ n = 0) ∨ (m = 2 ∧ n = 2)) :=
by
  sorry

end NUMINAMATH_GPT_find_pairs_l1084_108459


namespace NUMINAMATH_GPT_perfect_square_trinomial_l1084_108492

theorem perfect_square_trinomial (m : ℝ) :
  ∃ (a : ℝ), (∀ (x : ℝ), x^2 - 2*(m-3)*x + 16 = (x - a)^2) ↔ (m = 7 ∨ m = -1) := by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_l1084_108492


namespace NUMINAMATH_GPT_remainder_of_3056_div_78_l1084_108432

-- Define the necessary conditions and the statement
theorem remainder_of_3056_div_78 : (3056 % 78) = 14 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_3056_div_78_l1084_108432


namespace NUMINAMATH_GPT_area_of_region_l1084_108400

theorem area_of_region : 
    ∃ (area : ℝ), 
    (∀ (x y : ℝ), (x^2 + y^2 + 6 * x - 10 * y + 5 = 0) → 
    area = 29 * Real.pi) := 
by
  use 29 * Real.pi
  intros x y h
  sorry

end NUMINAMATH_GPT_area_of_region_l1084_108400


namespace NUMINAMATH_GPT_geometric_sequence_condition_l1084_108461

-- Define the condition ac = b^2
def condition (a b c : ℝ) : Prop := a * c = b ^ 2

-- Define what it means for a, b, c to form a geometric sequence
def geometric_sequence (a b c : ℝ) : Prop := 
  (b ≠ 0 → a / b = b / c) ∧ (a = 0 → b = 0 ∧ c = 0)

-- The goal is to prove the necessary but not sufficient condition
theorem geometric_sequence_condition (a b c : ℝ) :
  condition a b c ↔ (geometric_sequence a b c → condition a b c) ∧ (¬ (geometric_sequence a b c) → condition a b c ∧ ¬ (geometric_sequence (2 : ℝ) (0 : ℝ) (0 : ℝ))) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_condition_l1084_108461


namespace NUMINAMATH_GPT_carrots_not_used_l1084_108421

theorem carrots_not_used :
  let total_carrots := 300
  let carrots_before_lunch := (2 / 5) * total_carrots
  let remaining_after_lunch := total_carrots - carrots_before_lunch
  let carrots_by_end_of_day := (3 / 5) * remaining_after_lunch
  remaining_after_lunch - carrots_by_end_of_day = 72
:= by
  sorry

end NUMINAMATH_GPT_carrots_not_used_l1084_108421


namespace NUMINAMATH_GPT_charlie_ride_distance_l1084_108412

-- Define the known values
def oscar_ride : ℝ := 0.75
def difference : ℝ := 0.5

-- Define Charlie's bus ride distance
def charlie_ride : ℝ := oscar_ride - difference

-- The theorem to be proven
theorem charlie_ride_distance : charlie_ride = 0.25 := 
by sorry

end NUMINAMATH_GPT_charlie_ride_distance_l1084_108412


namespace NUMINAMATH_GPT_integral_evaluation_l1084_108447

noncomputable def definite_integral (a b : ℝ) (f : ℝ → ℝ) : ℝ :=
  ∫ x in a..b, f x

theorem integral_evaluation : 
  definite_integral 1 2 (fun x => 1 / x + x) = Real.log 2 + 3 / 2 :=
  sorry

end NUMINAMATH_GPT_integral_evaluation_l1084_108447


namespace NUMINAMATH_GPT_tom_hockey_games_l1084_108430

def tom_hockey_games_last_year (games_this_year missed_this_year total_games : Nat) : Nat :=
  total_games - games_this_year

theorem tom_hockey_games :
  ∀ (games_this_year missed_this_year total_games : Nat),
    games_this_year = 4 →
    missed_this_year = 7 →
    total_games = 13 →
    tom_hockey_games_last_year games_this_year total_games = 9 := by
  intros games_this_year missed_this_year total_games h1 h2 h3
  -- The proof steps would go here
  sorry

end NUMINAMATH_GPT_tom_hockey_games_l1084_108430


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1084_108493

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = 1 + Real.sqrt 3) :
  ((x + 3) / (x^2 - 2*x + 1) * (x - 1) / (x^2 + 3*x) + 1 / x) = Real.sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1084_108493


namespace NUMINAMATH_GPT_man_total_pay_l1084_108439

def regular_rate : ℕ := 3
def regular_hours : ℕ := 40
def overtime_hours : ℕ := 13

def regular_pay : ℕ := regular_rate * regular_hours
def overtime_rate : ℕ := 2 * regular_rate
def overtime_pay : ℕ := overtime_rate * overtime_hours

def total_pay : ℕ := regular_pay + overtime_pay

theorem man_total_pay : total_pay = 198 := by
  sorry

end NUMINAMATH_GPT_man_total_pay_l1084_108439


namespace NUMINAMATH_GPT_all_lights_on_l1084_108498

def light_on (n : ℕ) : Prop := sorry

axiom light_rule_1 (k : ℕ) (hk: light_on k): light_on (2 * k) ∧ light_on (2 * k + 1)
axiom light_rule_2 (k : ℕ) (hk: ¬ light_on k): ¬ light_on (4 * k + 1) ∧ ¬ light_on (4 * k + 3)
axiom light_2023_on : light_on 2023

theorem all_lights_on (n : ℕ) (hn : n < 2023) : light_on n :=
by sorry

end NUMINAMATH_GPT_all_lights_on_l1084_108498


namespace NUMINAMATH_GPT_intersection_of_M_and_N_is_0_and_2_l1084_108415

open Set

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

theorem intersection_of_M_and_N_is_0_and_2 : M ∩ N = {0, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_is_0_and_2_l1084_108415


namespace NUMINAMATH_GPT_average_of_expressions_l1084_108474

theorem average_of_expressions (y : ℝ) :
  (1 / 3:ℝ) * ((2 * y + 5) + (3 * y + 4) + (7 * y - 2)) = 4 * y + 7 / 3 :=
by sorry

end NUMINAMATH_GPT_average_of_expressions_l1084_108474


namespace NUMINAMATH_GPT_typing_pages_l1084_108445

theorem typing_pages (typists : ℕ) (pages min : ℕ) 
  (h_typists_can_type_two_pages_in_two_minutes : typists * 2 / min = pages / min) 
  (h_10_typists_type_25_pages_in_5_minutes : 10 * 25 / 5 = pages / min) :
  pages / min = 2 := 
sorry

end NUMINAMATH_GPT_typing_pages_l1084_108445


namespace NUMINAMATH_GPT_hexagon_area_l1084_108499

theorem hexagon_area :
  let points := [(0, 0), (2, 4), (5, 4), (7, 0), (5, -4), (2, -4), (0, 0)]
  ∃ (area : ℝ), area = 52 := by
  sorry

end NUMINAMATH_GPT_hexagon_area_l1084_108499


namespace NUMINAMATH_GPT_prob1_converse_prob1_inverse_prob1_contrapositive_prob2_converse_prob2_inverse_prob2_contrapositive_l1084_108476

-- Problem 1: Original proposition converse, inverse, contrapositive
theorem prob1_converse (x y : ℝ) (h : x = 0 ∨ y = 0) : x * y = 0 :=
sorry

theorem prob1_inverse (x y : ℝ) (h : x * y ≠ 0) : x ≠ 0 ∧ y ≠ 0 :=
sorry

theorem prob1_contrapositive (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) : x * y ≠ 0 :=
sorry

-- Problem 2: Original proposition converse, inverse, contrapositive
theorem prob2_converse (x y : ℝ) (h : x * y > 0) : x > 0 ∧ y > 0 :=
sorry

theorem prob2_inverse (x y : ℝ) (h : x ≤ 0 ∨ y ≤ 0) : x * y ≤ 0 :=
sorry

theorem prob2_contrapositive (x y : ℝ) (h : x * y ≤ 0) : x ≤ 0 ∨ y ≤ 0 :=
sorry

end NUMINAMATH_GPT_prob1_converse_prob1_inverse_prob1_contrapositive_prob2_converse_prob2_inverse_prob2_contrapositive_l1084_108476


namespace NUMINAMATH_GPT_correct_calculation_l1084_108409

theorem correct_calculation (a b : ℝ) : (a * b) ^ 2 = a ^ 2 * b ^ 2 := by
  sorry

end NUMINAMATH_GPT_correct_calculation_l1084_108409


namespace NUMINAMATH_GPT_equation_of_line_containing_chord_l1084_108405

theorem equation_of_line_containing_chord (x y : ℝ) : 
  (y^2 = -8 * x) ∧ ((-1, 1) = ((x + x) / 2, (y + y) / 2)) →
  4 * x + y + 3 = 0 :=
by 
  sorry

end NUMINAMATH_GPT_equation_of_line_containing_chord_l1084_108405


namespace NUMINAMATH_GPT_ratio_Smax_Smin_l1084_108433

-- Define the area of a cube's diagonal cross-section through BD1
def cross_section_area (a : ℝ) : ℝ := sorry

theorem ratio_Smax_Smin (a : ℝ) (S S_min S_max : ℝ) :
  cross_section_area a = S →
  S_min = (a^2 * Real.sqrt 6) / 2 →
  S_max = a^2 * Real.sqrt 6 →
  S_max / S_min = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_Smax_Smin_l1084_108433


namespace NUMINAMATH_GPT_value_of_x_l1084_108442

theorem value_of_x (x : ℝ) (h : 3 * x + 15 = (1/3) * (7 * x + 45)) : x = 0 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l1084_108442


namespace NUMINAMATH_GPT_inequality_holds_l1084_108408

theorem inequality_holds (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) : 
  ((2 + x)/(1 + x))^2 + ((2 + y)/(1 + y))^2 ≥ 9/2 := 
sorry

end NUMINAMATH_GPT_inequality_holds_l1084_108408


namespace NUMINAMATH_GPT_total_female_officers_l1084_108404

theorem total_female_officers
  (percent_female_on_duty : ℝ)
  (total_on_duty : ℝ)
  (half_of_total_on_duty : ℝ)
  (num_females_on_duty : ℝ) :
  percent_female_on_duty = 0.10 →
  total_on_duty = 200 →
  half_of_total_on_duty = total_on_duty / 2 →
  num_females_on_duty = half_of_total_on_duty →
  num_females_on_duty = percent_female_on_duty * (1000 : ℝ) :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_total_female_officers_l1084_108404


namespace NUMINAMATH_GPT_number_of_valid_partitions_l1084_108495

-- Define the condition to check if a list of integers has all elements same or exactly differ by 1
def validPartition (l : List ℕ) : Prop :=
  l ≠ [] ∧ (∀ (a b : ℕ), a ∈ l → b ∈ l → a = b ∨ a = b + 1 ∨ b = a + 1)

-- Count valid partitions of n (integer partitions meeting the given condition)
noncomputable def countValidPartitions (n : ℕ) : ℕ :=
  if n = 0 then 0 else n

-- Main theorem
theorem number_of_valid_partitions (n : ℕ) : countValidPartitions n = n :=
by
  sorry

end NUMINAMATH_GPT_number_of_valid_partitions_l1084_108495


namespace NUMINAMATH_GPT_simplify_expression_l1084_108458

theorem simplify_expression : 4 * (15 / 5) * (24 / -60) = - (24 / 5) := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1084_108458


namespace NUMINAMATH_GPT_triangle_angle_contradiction_l1084_108425

theorem triangle_angle_contradiction :
  ∀ (α β γ : ℝ), (α + β + γ = 180) →
  (α > 60) ∧ (β > 60) ∧ (γ > 60) →
  false :=
by
  intros α β γ h_sum h_angles
  sorry

end NUMINAMATH_GPT_triangle_angle_contradiction_l1084_108425


namespace NUMINAMATH_GPT_modular_inverse_7_10000_l1084_108484

theorem modular_inverse_7_10000 :
  (7 * 8571) % 10000 = 1 := 
sorry

end NUMINAMATH_GPT_modular_inverse_7_10000_l1084_108484


namespace NUMINAMATH_GPT_jake_and_luke_items_l1084_108449

theorem jake_and_luke_items :
  ∃ (p j : ℕ), 6 * p + 2 * j ≤ 50 ∧ (∀ (p' : ℕ), 6 * p' + 2 * j ≤ 50 → p' ≤ p) ∧ p + j = 9 :=
by
  sorry

end NUMINAMATH_GPT_jake_and_luke_items_l1084_108449


namespace NUMINAMATH_GPT_initial_pants_l1084_108477

theorem initial_pants (pairs_per_year : ℕ) (pants_per_pair : ℕ) (years : ℕ) (total_pants : ℕ) 
  (h1 : pairs_per_year = 4) (h2 : pants_per_pair = 2) (h3 : years = 5) (h4 : total_pants = 90) : 
  ∃ (initial_pants : ℕ), initial_pants = total_pants - (pairs_per_year * pants_per_pair * years) :=
by
  use 50
  sorry

end NUMINAMATH_GPT_initial_pants_l1084_108477


namespace NUMINAMATH_GPT_loads_of_laundry_l1084_108444

theorem loads_of_laundry (families : ℕ) (days : ℕ) (adults_per_family : ℕ) (children_per_family : ℕ)
  (adult_towels_per_day : ℕ) (child_towels_per_day : ℕ) (initial_capacity : ℕ) (reduced_capacity : ℕ)
  (initial_days : ℕ) (remaining_days : ℕ) : 
  families = 7 → days = 12 → adults_per_family = 2 → children_per_family = 4 → 
  adult_towels_per_day = 2 → child_towels_per_day = 1 → initial_capacity = 8 → 
  reduced_capacity = 6 → initial_days = 6 → remaining_days = 6 → 
  (families * (adults_per_family * adult_towels_per_day + children_per_family * child_towels_per_day) * initial_days / initial_capacity) +
  (families * (adults_per_family * adult_towels_per_day + children_per_family * child_towels_per_day) * remaining_days / reduced_capacity) = 98 :=
by 
  intros _ _ _ _ _ _ _ _ _ _
  sorry

end NUMINAMATH_GPT_loads_of_laundry_l1084_108444


namespace NUMINAMATH_GPT_Joan_balloons_l1084_108416

variable (J : ℕ) -- Joan's blue balloons

theorem Joan_balloons (h : J + 41 = 81) : J = 40 :=
by
  sorry

end NUMINAMATH_GPT_Joan_balloons_l1084_108416


namespace NUMINAMATH_GPT_solve_quadratic_inequality_l1084_108422

theorem solve_quadratic_inequality (x : ℝ) : (-x^2 - 2 * x + 3 < 0) ↔ (x < -3 ∨ x > 1) := 
sorry

end NUMINAMATH_GPT_solve_quadratic_inequality_l1084_108422


namespace NUMINAMATH_GPT_percent_twelve_equals_eighty_four_l1084_108490

theorem percent_twelve_equals_eighty_four (x : ℝ) (h : (12 / 100) * x = 84) : x = 700 :=
by
  sorry

end NUMINAMATH_GPT_percent_twelve_equals_eighty_four_l1084_108490


namespace NUMINAMATH_GPT_box_volume_l1084_108440

-- Definitions for the dimensions of the box: Length (L), Width (W), and Height (H)
variables (L W H : ℝ)

-- Condition 1: Area of the front face is half the area of the top face
def condition1 := L * W = 0.5 * (L * H)

-- Condition 2: Area of the top face is 1.5 times the area of the side face
def condition2 := L * H = 1.5 * (W * H)

-- Condition 3: Area of the side face is 200
def condition3 := W * H = 200

-- Theorem stating the volume of the box is 3000 given the above conditions
theorem box_volume : condition1 L W H ∧ condition2 L W H ∧ condition3 W H → L * W * H = 3000 :=
by sorry

end NUMINAMATH_GPT_box_volume_l1084_108440


namespace NUMINAMATH_GPT_proof_problem_l1084_108486

theorem proof_problem
  (x y z : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0) :
  (x + 1 / y ≥ 2) ∨ (y + 1 / z ≥ 2) ∨ (z + 1 / x ≥ 2) :=
sorry

end NUMINAMATH_GPT_proof_problem_l1084_108486


namespace NUMINAMATH_GPT_euler_totient_inequality_l1084_108428

open Int

def is_power_of_prime (m : ℕ) : Prop :=
  ∃ p k : ℕ, (Nat.Prime p) ∧ (k ≥ 1) ∧ (m = p^k)

def φ (n m : ℕ) (h : m ≠ 1) : ℕ := -- This is a placeholder, you would need an actual implementation for φ
  sorry

theorem euler_totient_inequality (m : ℕ) (h : m ≠ 1) :
  (is_power_of_prime m) ↔ (∀ n > 0, (φ n m h) / n ≥ (φ m m h) / m) :=
sorry

end NUMINAMATH_GPT_euler_totient_inequality_l1084_108428


namespace NUMINAMATH_GPT_point_in_fourth_quadrant_l1084_108429

theorem point_in_fourth_quadrant (x y : Real) (hx : x = 2) (hy : y = Real.tan 300) : 
  (0 < x) → (y < 0) → (x = 2 ∧ y = -Real.sqrt 3) :=
by
  intro hx_trans hy_trans
  -- Here you will provide statements or tactics to assist the proof if you were completing it
  sorry

end NUMINAMATH_GPT_point_in_fourth_quadrant_l1084_108429


namespace NUMINAMATH_GPT_simplify_expression_l1084_108494

theorem simplify_expression (x : ℝ) (h : x^2 + 2 * x = 1) :
  (1 - x) ^ 2 - (x + 3) * (3 - x) - (x - 3) * (x - 1) = -10 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l1084_108494


namespace NUMINAMATH_GPT_shaded_square_percentage_l1084_108426

theorem shaded_square_percentage (total_squares shaded_squares : ℕ) (h_total: total_squares = 25) (h_shaded: shaded_squares = 13) : 
(shaded_squares * 100) / total_squares = 52 := 
by
  sorry

end NUMINAMATH_GPT_shaded_square_percentage_l1084_108426


namespace NUMINAMATH_GPT_smallest_fraction_l1084_108480

theorem smallest_fraction (x : ℝ) (h : x > 2022) :
  min (min (min (min (x / 2022) (2022 / (x - 1))) ((x + 1) / 2022)) (2022 / x)) (2022 / (x + 1)) = 2022 / (x + 1) :=
sorry

end NUMINAMATH_GPT_smallest_fraction_l1084_108480


namespace NUMINAMATH_GPT_simplify_expression_l1084_108436

theorem simplify_expression (a b : ℚ) (ha : a = -1) (hb : b = 1/4) : 
  (a + 2 * b)^2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l1084_108436


namespace NUMINAMATH_GPT_nat_implies_int_incorrect_reasoning_due_to_minor_premise_l1084_108460

-- Definitions for conditions
def is_integer (x : ℚ) : Prop := ∃ (n : ℤ), x = n
def is_natural (x : ℚ) : Prop := ∃ (n : ℕ), x = n

-- Major premise: Natural numbers are integers
theorem nat_implies_int (n : ℕ) : is_integer n := 
  ⟨n, rfl⟩

-- Minor premise: 1 / 3 is a natural number
def one_div_three_is_natural : Prop := is_natural (1 / 3)

-- Conclusion: 1 / 3 is an integer
def one_div_three_is_integer : Prop := is_integer (1 / 3)

-- The proof problem
theorem incorrect_reasoning_due_to_minor_premise :
  ¬one_div_three_is_natural :=
sorry

end NUMINAMATH_GPT_nat_implies_int_incorrect_reasoning_due_to_minor_premise_l1084_108460


namespace NUMINAMATH_GPT_initial_sale_price_percent_l1084_108464

theorem initial_sale_price_percent (P S : ℝ) (h1 : S * 0.90 = 0.63 * P) :
  S = 0.70 * P :=
by
  sorry

end NUMINAMATH_GPT_initial_sale_price_percent_l1084_108464


namespace NUMINAMATH_GPT_gemstones_count_l1084_108470

theorem gemstones_count (F B S W SN : ℕ) 
  (hS : S = 1)
  (hSpaatz : S = F / 2 - 2)
  (hBinkie : B = 4 * F)
  (hWhiskers : W = S + 3)
  (hSnowball : SN = 2 * W) :
  B = 24 :=
by
  sorry

end NUMINAMATH_GPT_gemstones_count_l1084_108470


namespace NUMINAMATH_GPT_total_cows_l1084_108427

theorem total_cows (n : ℕ) 
  (h₁ : n / 3 + n / 6 + n / 9 + 8 = n) : n = 144 :=
by sorry

end NUMINAMATH_GPT_total_cows_l1084_108427


namespace NUMINAMATH_GPT_find_line_equation_l1084_108491

open Real

-- Define the parabola
def Parabola (x y : ℝ) : Prop := y^2 = 2 * x

-- Define the line passing through (0,2)
def LineThruPoint (x y k : ℝ) : Prop := y = k * x + 2

-- Define when line intersects parabola
def LineIntersectsParabola (x1 y1 x2 y2 k : ℝ) : Prop :=
  LineThruPoint x1 y1 k ∧ LineThruPoint x2 y2 k ∧ Parabola x1 y1 ∧ Parabola x2 y2

-- Define when circle with diameter MN passes through origin O
def CircleThroughOrigin (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 * x2 + y1 * y2 = 0

theorem find_line_equation (k : ℝ) 
    (h₀ : k ≠ 0)
    (h₁ : ∃ x1 y1 x2 y2, LineIntersectsParabola x1 y1 x2 y2 k)
    (h₂ : ∃ x1 y1 x2 y2, LineIntersectsParabola x1 y1 x2 y2 k ∧ CircleThroughOrigin x1 y1 x2 y2) :
  (∃ x y, LineThruPoint x y k ∧ y = -x + 2) :=
sorry

end NUMINAMATH_GPT_find_line_equation_l1084_108491


namespace NUMINAMATH_GPT_turtle_distance_during_rabbit_rest_l1084_108462

theorem turtle_distance_during_rabbit_rest
  (D : ℕ)
  (vr vt : ℕ)
  (rabbit_speed_multiple : vr = 15 * vt)
  (rabbit_remaining_distance : D - 100 = 900)
  (turtle_finish_time : true)
  (rabbit_to_be_break : true)
  (turtle_finish_during_rabbit_rest : true) :
  (D - (900 / 15) = 940) :=
by
  sorry

end NUMINAMATH_GPT_turtle_distance_during_rabbit_rest_l1084_108462


namespace NUMINAMATH_GPT_total_sales_l1084_108450

theorem total_sales (T : ℝ) (h1 : (2 / 5) * T = (2 / 5) * T) (h2 : (3 / 5) * T = 48) : T = 80 :=
by
  -- added sorry to skip proofs as per the requirement
  sorry

end NUMINAMATH_GPT_total_sales_l1084_108450


namespace NUMINAMATH_GPT_at_least_one_genuine_product_l1084_108423

-- Definitions of the problem conditions
structure Products :=
  (total : ℕ)
  (genuine : ℕ)
  (defective : ℕ)

def products : Products := { total := 12, genuine := 10, defective := 2 }

-- Definition of the event
def certain_event (p : Products) (selected : ℕ) : Prop :=
  selected > p.defective

-- The theorem stating that there is at least one genuine product among the selected ones
theorem at_least_one_genuine_product : certain_event products 3 :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_genuine_product_l1084_108423


namespace NUMINAMATH_GPT_average_age_combined_l1084_108403

theorem average_age_combined (fifth_graders_count : ℕ) (fifth_graders_avg_age : ℚ)
                             (parents_count : ℕ) (parents_avg_age : ℚ)
                             (grandparents_count : ℕ) (grandparents_avg_age : ℚ) :
  fifth_graders_count = 40 →
  fifth_graders_avg_age = 10 →
  parents_count = 60 →
  parents_avg_age = 35 →
  grandparents_count = 20 →
  grandparents_avg_age = 65 →
  (fifth_graders_count * fifth_graders_avg_age + 
   parents_count * parents_avg_age + 
   grandparents_count * grandparents_avg_age) / 
  (fifth_graders_count + parents_count + grandparents_count) = 95 / 3 := sorry

end NUMINAMATH_GPT_average_age_combined_l1084_108403


namespace NUMINAMATH_GPT_opposite_of_three_l1084_108469

theorem opposite_of_three : -3 = -3 := 
by sorry

end NUMINAMATH_GPT_opposite_of_three_l1084_108469


namespace NUMINAMATH_GPT_geometric_sequence_a5_l1084_108485

variable {a : ℕ → ℝ}
variable (h₁ : a 3 * a 7 = 3)
variable (h₂ : a 3 + a 7 = 4)

theorem geometric_sequence_a5 : a 5 = Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_a5_l1084_108485


namespace NUMINAMATH_GPT_find_point_coordinates_l1084_108424

open Real

-- Define circles C1 and C2
def circle_C1 (x y : ℝ) : Prop := (x + 4)^2 + (y - 2)^2 = 9
def circle_C2 (x y : ℝ) : Prop := (x - 5)^2 + (y - 6)^2 = 9

-- Define mutually perpendicular lines passing through point P
def line_l1 (P : ℝ × ℝ) (k : ℝ) (x y : ℝ) : Prop := y - P.2 = k * (x - P.1)
def line_l2 (P : ℝ × ℝ) (k : ℝ) (x y : ℝ) : Prop := y - P.2 = -1/k * (x - P.1)

-- Define the condition that chord lengths intercepted by lines on respective circles are equal
def equal_chord_lengths (P : ℝ × ℝ) (k : ℝ) : Prop :=
  abs (-4 * k - 2 + P.2 - k * P.1) / sqrt ((k^2) + 1) = abs (5 + 6 * k - k * P.2 - P.1) / sqrt ((k^2) + 1)

-- Main statement to be proved
theorem find_point_coordinates :
  ∃ (P : ℝ × ℝ), 
  circle_C1 (P.1) (P.2) ∧
  circle_C2 (P.1) (P.2) ∧
  (∀ k : ℝ, k ≠ 0 → equal_chord_lengths P k) ∧
  (P = (-3/2, 17/2) ∨ P = (5/2, -1/2)) :=
sorry

end NUMINAMATH_GPT_find_point_coordinates_l1084_108424


namespace NUMINAMATH_GPT_complement_M_eq_45_l1084_108471

open Set Nat

/-- Define the universal set U and the set M in Lean -/
def U : Set ℕ := {1, 2, 3, 4, 5, 6}

def M : Set ℕ := {x | 6 % x = 0 ∧ x ∈ U}

/-- Lean theorem statement for the complement of M in U -/
theorem complement_M_eq_45 : (U \ M) = {4, 5} :=
by
  sorry

end NUMINAMATH_GPT_complement_M_eq_45_l1084_108471


namespace NUMINAMATH_GPT_age_of_teacher_l1084_108478

theorem age_of_teacher (S T : ℕ) (avg_students avg_total : ℕ) (num_students num_total : ℕ)
  (h1 : num_students = 50)
  (h2 : avg_students = 14)
  (h3 : num_total = 51)
  (h4 : avg_total = 15)
  (h5 : S = avg_students * num_students)
  (h6 : S + T = avg_total * num_total) :
  T = 65 := 
by {
  sorry
}

end NUMINAMATH_GPT_age_of_teacher_l1084_108478


namespace NUMINAMATH_GPT_remainder_of_sum_of_squares_mod_n_l1084_108401

theorem remainder_of_sum_of_squares_mod_n (a b n : ℤ) (hn : n > 1) 
  (ha : a * a ≡ 1 [ZMOD n]) (hb : b * b ≡ 1 [ZMOD n]) : 
  (a^2 + b^2) % n = 2 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_of_sum_of_squares_mod_n_l1084_108401


namespace NUMINAMATH_GPT_regression_eq_change_in_y_l1084_108455

-- Define the regression equation
def regression_eq (x : ℝ) : ℝ := 2 - 1.5 * x

-- Define the statement to be proved
theorem regression_eq_change_in_y (x : ℝ) :
  regression_eq (x + 1) = regression_eq x - 1.5 :=
by sorry

end NUMINAMATH_GPT_regression_eq_change_in_y_l1084_108455


namespace NUMINAMATH_GPT_exam_total_boys_l1084_108452

theorem exam_total_boys (T F : ℕ) (avg_total avg_passed avg_failed : ℕ) 
    (H1 : avg_total = 40) (H2 : avg_passed = 39) (H3 : avg_failed = 15) (H4 : 125 > 0) (H5 : 125 * avg_passed + (T - 125) * avg_failed = T * avg_total) : T = 120 :=
by
  sorry

end NUMINAMATH_GPT_exam_total_boys_l1084_108452


namespace NUMINAMATH_GPT_desk_height_l1084_108406

variables (h l w : ℝ)

theorem desk_height
  (h_eq_2l_50 : h + 2 * l = 50)
  (h_eq_2w_40 : h + 2 * w = 40)
  (l_minus_w_eq_5 : l - w = 5) :
  h = 30 :=
by {
  sorry
}

end NUMINAMATH_GPT_desk_height_l1084_108406
