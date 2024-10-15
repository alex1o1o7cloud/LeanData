import Mathlib

namespace NUMINAMATH_GPT_cylinder_ratio_l1615_161548

theorem cylinder_ratio (h r : ℝ) (h_eq : h = 2 * Real.pi * r) : 
  h / r = 2 * Real.pi := 
by 
  sorry

end NUMINAMATH_GPT_cylinder_ratio_l1615_161548


namespace NUMINAMATH_GPT_range_of_2x_minus_y_l1615_161560

variable {x y : ℝ}

theorem range_of_2x_minus_y (h1 : 2 < x) (h2 : x < 4) (h3 : -1 < y) (h4 : y < 3) :
  ∃ (a b : ℝ), (1 < a) ∧ (a < 2 * x - y) ∧ (2 * x - y < b) ∧ (b < 9) :=
by
  sorry

end NUMINAMATH_GPT_range_of_2x_minus_y_l1615_161560


namespace NUMINAMATH_GPT_find_minimum_m_l1615_161555

theorem find_minimum_m (m : ℕ) (h1 : 1350 + 36 * m < 2136) (h2 : 1500 + 45 * m ≥ 2365) :
  m = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_minimum_m_l1615_161555


namespace NUMINAMATH_GPT_function_properties_l1615_161549

noncomputable def f (x : ℝ) : ℝ := 3^x - 3^(-x)

theorem function_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, x < y → f x < f y) :=
by
  sorry

end NUMINAMATH_GPT_function_properties_l1615_161549


namespace NUMINAMATH_GPT_cubes_with_one_colored_face_l1615_161590

theorem cubes_with_one_colored_face (n : ℕ) (c1 : ℕ) (c2 : ℕ) :
  (n = 64) ∧ (c1 = 4) ∧ (c2 = 4) → ((4 * n) * 2) / n = 32 :=
by 
  sorry

end NUMINAMATH_GPT_cubes_with_one_colored_face_l1615_161590


namespace NUMINAMATH_GPT_a_and_b_together_30_days_l1615_161550

variable (R_a R_b : ℝ)

-- Conditions
axiom condition1 : R_a = 3 * R_b
axiom condition2 : R_a * 40 = (R_a + R_b) * 30

-- Question: prove that a and b together can complete the work in 30 days.
theorem a_and_b_together_30_days (R_a R_b : ℝ) (condition1 : R_a = 3 * R_b) (condition2 : R_a * 40 = (R_a + R_b) * 30) : true :=
by
  sorry

end NUMINAMATH_GPT_a_and_b_together_30_days_l1615_161550


namespace NUMINAMATH_GPT_sum_of_coeffs_in_expansion_l1615_161523

theorem sum_of_coeffs_in_expansion (n : ℕ) : 
  (1 - 2 : ℤ)^n = (-1 : ℤ)^n :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coeffs_in_expansion_l1615_161523


namespace NUMINAMATH_GPT_least_three_digit_product_of_digits_is_8_l1615_161525

theorem least_three_digit_product_of_digits_is_8 :
  ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ (n.digits 10).prod = 8 ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ (m.digits 10).prod = 8 → n ≤ m :=
sorry

end NUMINAMATH_GPT_least_three_digit_product_of_digits_is_8_l1615_161525


namespace NUMINAMATH_GPT_new_average_after_17th_l1615_161530

def old_average (A : ℕ) (n : ℕ) : ℕ :=
  A -- A is the average before the 17th inning

def runs_in_17th : ℕ := 84 -- The score in the 17th inning

def average_increase : ℕ := 3 -- The increase in average after the 17th inning

theorem new_average_after_17th (A : ℕ) (n : ℕ) (h1 : n = 16) (h2 : old_average A n + average_increase = A + 3) :
  (old_average A n) + average_increase = 36 :=
by
  sorry

end NUMINAMATH_GPT_new_average_after_17th_l1615_161530


namespace NUMINAMATH_GPT_expand_expression_l1615_161594

theorem expand_expression (a b : ℤ) : (-1 + a * b^2)^2 = 1 - 2 * a * b^2 + a^2 * b^4 :=
by sorry

end NUMINAMATH_GPT_expand_expression_l1615_161594


namespace NUMINAMATH_GPT_bus_empty_seats_l1615_161583

theorem bus_empty_seats : 
  let initial_seats : ℕ := 23 * 4
  let people_at_start : ℕ := 16
  let first_board : ℕ := 15
  let first_alight : ℕ := 3
  let second_board : ℕ := 17
  let second_alight : ℕ := 10
  let seats_after_init : ℕ := initial_seats - people_at_start
  let seats_after_first : ℕ := seats_after_init - (first_board - first_alight)
  let seats_after_second : ℕ := seats_after_first - (second_board - second_alight)
  seats_after_second = 57 :=
by
  sorry

end NUMINAMATH_GPT_bus_empty_seats_l1615_161583


namespace NUMINAMATH_GPT_eggs_processed_per_day_l1615_161533

/-- In a certain egg-processing plant, every egg must be inspected, and is either accepted for processing or rejected. For every 388 eggs accepted for processing, 12 eggs are rejected.

If, on a particular day, 37 additional eggs were accepted, but the overall number of eggs inspected remained the same, the ratio of those accepted to those rejected would be 405 to 3.

Prove that the number of eggs processed per day, given these conditions, is 125763.
-/
theorem eggs_processed_per_day : ∃ (E : ℕ), (∃ (R : ℕ), 38 * R = 3 * (E - 37) ∧  E = 32 * R + E / 33 ) ∧ (E = 125763) :=
sorry

end NUMINAMATH_GPT_eggs_processed_per_day_l1615_161533


namespace NUMINAMATH_GPT_invest_today_for_future_value_l1615_161584

-- Define the given future value, interest rate, and number of years as constants
def FV : ℝ := 600000
def r : ℝ := 0.04
def n : ℕ := 15
def target : ℝ := 333087.66

-- Define the present value calculation
noncomputable def PV : ℝ := FV / (1 + r)^n

-- State the theorem that PV is approximately equal to the target value
theorem invest_today_for_future_value : PV = target := 
by sorry

end NUMINAMATH_GPT_invest_today_for_future_value_l1615_161584


namespace NUMINAMATH_GPT_value_of_a_b_c_l1615_161563

theorem value_of_a_b_c 
    (a b c : Int)
    (h1 : ∀ x : Int, x^2 + 10*x + 21 = (x + a) * (x + b))
    (h2 : ∀ x : Int, x^2 + 3*x - 88 = (x + b) * (x - c))
    :
    a + b + c = 18 := 
sorry

end NUMINAMATH_GPT_value_of_a_b_c_l1615_161563


namespace NUMINAMATH_GPT_parallel_slope_l1615_161515

theorem parallel_slope (x y : ℝ) : (∃ b : ℝ, 3 * x - 6 * y = 12) → (∀ (x' y' : ℝ), (∃ b' : ℝ, 3 * x' - 6 * y' = b') → (∃ m : ℝ, m = 1 / 2)) :=
by
  sorry

end NUMINAMATH_GPT_parallel_slope_l1615_161515


namespace NUMINAMATH_GPT_star_area_l1615_161564

-- Conditions
def square_ABCD_area (s : ℝ) := s^2 = 72

-- Question and correct answer
theorem star_area (s : ℝ) (h : square_ABCD_area s) : 24 = 24 :=
by sorry

end NUMINAMATH_GPT_star_area_l1615_161564


namespace NUMINAMATH_GPT_cost_price_of_watch_l1615_161572

theorem cost_price_of_watch (C : ℝ) 
  (h1 : ∃ (SP1 SP2 : ℝ), SP1 = 0.54 * C ∧ SP2 = 1.04 * C ∧ SP2 = SP1 + 140) : 
  C = 280 :=
by
  obtain ⟨SP1, SP2, H1, H2, H3⟩ := h1
  sorry

end NUMINAMATH_GPT_cost_price_of_watch_l1615_161572


namespace NUMINAMATH_GPT_curve_C2_eqn_l1615_161582

theorem curve_C2_eqn (p : ℝ) (x y : ℝ) :
  (∃ x y, (x^2 - y^2 = 1) ∧ (y^2 = 2 * p * x) ∧ (2 * p = 3/4)) →
  (y^2 = (3/2) * x) :=
by
  sorry

end NUMINAMATH_GPT_curve_C2_eqn_l1615_161582


namespace NUMINAMATH_GPT_fraction_remain_same_l1615_161541

theorem fraction_remain_same (x y : ℝ) : (2 * x + y) / (3 * x + y) = (2 * (10 * x) + (10 * y)) / (3 * (10 * x) + (10 * y)) :=
by sorry

end NUMINAMATH_GPT_fraction_remain_same_l1615_161541


namespace NUMINAMATH_GPT_at_least_two_solutions_l1615_161578

theorem at_least_two_solutions (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  (∃ x, (x - a) * (x - b) = x - c) ∨ (∃ x, (x - b) * (x - c) = x - a) ∨ (∃ x, (x - c) * (x - a) = x - b) ∨
    (((x - a) * (x - b) = x - c) ∧ ((x - b) * (x - c) = x - a)) ∨ 
    (((x - b) * (x + c) = x - a) ∧ ((x - c) * (x - a) = x - b)) ∨ 
    (((x - c) * (x - a) = x - b) ∧ ((x - a) * (x - b) = x - c)) :=
sorry

end NUMINAMATH_GPT_at_least_two_solutions_l1615_161578


namespace NUMINAMATH_GPT_jonathan_fourth_task_completion_l1615_161551

-- Conditions
def start_time : Nat := 9 * 60 -- 9:00 AM in minutes
def third_task_completion_time : Nat := 11 * 60 + 30 -- 11:30 AM in minutes
def number_of_tasks : Nat := 4
def number_of_completed_tasks : Nat := 3

-- Calculation of time duration
def total_time_first_three_tasks : Nat :=
  third_task_completion_time - start_time

def duration_of_one_task : Nat :=
  total_time_first_three_tasks / number_of_completed_tasks
  
-- Statement to prove
theorem jonathan_fourth_task_completion :
  (third_task_completion_time + duration_of_one_task) = (12 * 60 + 20) :=
  by
    -- We do not need to provide the proof steps as per instructions
    sorry

end NUMINAMATH_GPT_jonathan_fourth_task_completion_l1615_161551


namespace NUMINAMATH_GPT_Joan_attended_games_l1615_161538

def total_games : ℕ := 864
def games_missed_by_Joan : ℕ := 469
def games_attended_by_Joan : ℕ := total_games - games_missed_by_Joan

theorem Joan_attended_games : games_attended_by_Joan = 395 := 
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_Joan_attended_games_l1615_161538


namespace NUMINAMATH_GPT_smallest_three_digit_perfect_square_l1615_161561

theorem smallest_three_digit_perfect_square :
  ∀ (a : ℕ), 100 ≤ a ∧ a < 1000 → (∃ (n : ℕ), 1001 * a + 1 = n^2) → a = 183 :=
by
  sorry

end NUMINAMATH_GPT_smallest_three_digit_perfect_square_l1615_161561


namespace NUMINAMATH_GPT_marble_remainder_l1615_161566

theorem marble_remainder
  (r p : ℕ)
  (h_r : r % 5 = 2)
  (h_p : p % 5 = 4) :
  (r + p) % 5 = 1 :=
by
  sorry

end NUMINAMATH_GPT_marble_remainder_l1615_161566


namespace NUMINAMATH_GPT_product_of_place_values_l1615_161587

theorem product_of_place_values : 
  let place_value_1 := 800000
  let place_value_2 := 80
  let place_value_3 := 0.08
  place_value_1 * place_value_2 * place_value_3 = 5120000 := 
by 
  -- proof will be provided here 
  sorry

end NUMINAMATH_GPT_product_of_place_values_l1615_161587


namespace NUMINAMATH_GPT_meet_starting_point_together_at_7_40_AM_l1615_161576

-- Definitions of the input conditions
def Charlie_time : Nat := 5
def Alex_time : Nat := 8
def Taylor_time : Nat := 10

-- The combined time when they meet again at the starting point
def LCM_time (a b c : Nat) : Nat := Nat.lcm a (Nat.lcm b c)

-- Proving that the earliest time they all coincide again is 40 minutes after the start
theorem meet_starting_point_together_at_7_40_AM :
  LCM_time Charlie_time Alex_time Taylor_time = 40 := 
by
  unfold Charlie_time Alex_time Taylor_time LCM_time
  sorry

end NUMINAMATH_GPT_meet_starting_point_together_at_7_40_AM_l1615_161576


namespace NUMINAMATH_GPT_factor_quadratic_expression_l1615_161557

theorem factor_quadratic_expression (a b : ℤ) :
  (∃ a b : ℤ, (5 * a + 5 * b = -125) ∧ (a * b = -100) → (a + b = -25)) → (25 * x^2 - 125 * x - 100 = (5 * x + a) * (5 * x + b)) := 
by
  sorry

end NUMINAMATH_GPT_factor_quadratic_expression_l1615_161557


namespace NUMINAMATH_GPT_distinct_real_roots_l1615_161585

theorem distinct_real_roots (p : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 2 * |x1| - p = 0) ∧ (x2^2 - 2 * |x2| - p = 0)) → p > -1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_distinct_real_roots_l1615_161585


namespace NUMINAMATH_GPT_part1_solution_set_part2_range_a_l1615_161524

noncomputable def f (x a : ℝ) := 5 - abs (x + a) - abs (x - 2)

-- Part 1
theorem part1_solution_set (x : ℝ) (a : ℝ) (h : a = 1) :
  (f x a ≥ 0) ↔ (-2 ≤ x ∧ x ≤ 3) := sorry

-- Part 2
theorem part2_range_a (a : ℝ) :
  (∀ x, f x a ≤ 1) ↔ (a ≤ -6 ∨ a ≥ 2) := sorry

end NUMINAMATH_GPT_part1_solution_set_part2_range_a_l1615_161524


namespace NUMINAMATH_GPT_operation_5_7_eq_35_l1615_161503

noncomputable def operation (x y : ℝ) : ℝ := sorry

axiom condition1 :
  ∀ (x y : ℝ), (x * y > 0) → (operation (x * y) y = x * (operation y y))

axiom condition2 :
  ∀ (x : ℝ), (x > 0) → (operation (operation x 1) x = operation x 1)

axiom condition3 :
  (operation 1 1 = 2)

theorem operation_5_7_eq_35 : operation 5 7 = 35 :=
by
  sorry

end NUMINAMATH_GPT_operation_5_7_eq_35_l1615_161503


namespace NUMINAMATH_GPT_Mark_hours_left_l1615_161529

theorem Mark_hours_left (sick_days vacation_days : ℕ) (hours_per_day : ℕ) 
  (h1 : sick_days = 10) (h2 : vacation_days = 10) (h3 : hours_per_day = 8) 
  (used_sick_days : ℕ) (used_vacation_days : ℕ) 
  (h4 : used_sick_days = sick_days / 2) (h5 : used_vacation_days = vacation_days / 2) 
  : (sick_days + vacation_days - used_sick_days - used_vacation_days) * hours_per_day = 80 :=
by
  sorry

end NUMINAMATH_GPT_Mark_hours_left_l1615_161529


namespace NUMINAMATH_GPT_total_amount_spent_l1615_161543

def cost_of_soft_drink : ℕ := 2
def cost_per_candy_bar : ℕ := 5
def number_of_candy_bars : ℕ := 5

theorem total_amount_spent : cost_of_soft_drink + cost_per_candy_bar * number_of_candy_bars = 27 := by
  sorry

end NUMINAMATH_GPT_total_amount_spent_l1615_161543


namespace NUMINAMATH_GPT_inequality_ab_l1615_161520

theorem inequality_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) :
  1 / (a^2 + 1) + 1 / (b^2 + 1) ≥ 1 := 
sorry

end NUMINAMATH_GPT_inequality_ab_l1615_161520


namespace NUMINAMATH_GPT_units_digit_A_is_1_l1615_161571

def units_digit (n : ℕ) : ℕ := n % 10

noncomputable def A : ℕ := 2 * (3 + 1) * (3^2 + 1) * (3^4 + 1) + 1

theorem units_digit_A_is_1 : units_digit A = 1 := by
  sorry

end NUMINAMATH_GPT_units_digit_A_is_1_l1615_161571


namespace NUMINAMATH_GPT_simplify_expression_l1615_161596

variables (a b : ℝ)
noncomputable def x := (1 / 2) * (Real.sqrt (a / b) - Real.sqrt (b / a))

theorem simplify_expression (ha : a > 0) (hb : b > 0) :
  (2 * a * Real.sqrt (1 + x a b ^ 2)) / (x a b + Real.sqrt (1 + x a b ^ 2)) = a + b :=
sorry

end NUMINAMATH_GPT_simplify_expression_l1615_161596


namespace NUMINAMATH_GPT_smallest_y_for_perfect_cube_l1615_161528

-- Define the given conditions
def x : ℕ := 5 * 24 * 36

-- State the theorem to prove
theorem smallest_y_for_perfect_cube (y : ℕ) (h : y = 50) : 
  ∃ y, (x * y) % (y * y * y) = 0 :=
by
  sorry

end NUMINAMATH_GPT_smallest_y_for_perfect_cube_l1615_161528


namespace NUMINAMATH_GPT_value_of_M_l1615_161588

theorem value_of_M (x y z M : ℚ) 
  (h1 : x + y + z = 120)
  (h2 : x - 10 = M)
  (h3 : y + 10 = M)
  (h4 : 10 * z = M) : 
  M = 400 / 7 :=
sorry

end NUMINAMATH_GPT_value_of_M_l1615_161588


namespace NUMINAMATH_GPT_number_of_girls_l1615_161505

variable (b g d : ℕ)

-- Conditions
axiom boys_count : b = 1145
axiom difference : d = 510
axiom boys_equals_girls_plus_difference : b = g + d

-- Theorem to prove
theorem number_of_girls : g = 635 := by
  sorry

end NUMINAMATH_GPT_number_of_girls_l1615_161505


namespace NUMINAMATH_GPT_simplify_fraction_l1615_161517

theorem simplify_fraction (a b gcd : ℕ) (h1 : a = 72) (h2 : b = 108) (h3 : gcd = Nat.gcd a b) : (a / gcd) / (b / gcd) = 2 / 3 :=
by
  -- the proof is omitted here
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1615_161517


namespace NUMINAMATH_GPT_no_corner_cut_possible_l1615_161501

-- Define the cube and the triangle sides
def cube_edge_length : ℝ := 15
def triangle_side1 : ℝ := 5
def triangle_side2 : ℝ := 6
def triangle_side3 : ℝ := 8

-- Main statement: Prove that it's not possible to cut off a corner of the cube to form the given triangle
theorem no_corner_cut_possible :
  ¬ (∃ (a b c : ℝ),
    a^2 + b^2 = triangle_side1^2 ∧
    b^2 + c^2 = triangle_side2^2 ∧
    c^2 + a^2 = triangle_side3^2 ∧
    a^2 + b^2 + c^2 = 62.5) :=
sorry

end NUMINAMATH_GPT_no_corner_cut_possible_l1615_161501


namespace NUMINAMATH_GPT_number_of_lines_through_focus_intersecting_hyperbola_l1615_161554

open Set

noncomputable def hyperbola (x y : ℝ) : Prop := (x^2 / 2) - y^2 = 1

-- The coordinates of the focuses of the hyperbola
def right_focus : ℝ × ℝ := (2, 0)

-- Definition to express that a line passes through the right focus
def line_through_focus (l : ℝ → ℝ) : Prop := l 2 = 0

-- Definition for the length of segment AB being 4
def length_AB_is_4 (A B : ℝ × ℝ) : Prop := dist A B = 4

-- The statement asserting the number of lines satisfying the given condition
theorem number_of_lines_through_focus_intersecting_hyperbola:
  ∃ (n : ℕ), n = 3 ∧ ∀ (l : ℝ → ℝ),
  line_through_focus l →
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ length_AB_is_4 A B :=
sorry

end NUMINAMATH_GPT_number_of_lines_through_focus_intersecting_hyperbola_l1615_161554


namespace NUMINAMATH_GPT_range_of_k_l1615_161542

theorem range_of_k :
  ∀ (a k : ℝ) (f : ℝ → ℝ),
    (∀ x, f x = if x ≥ 0 then k^2 * x + a^2 - k else x^2 + (a^2 + 4 * a) * x + (2 - a)^2) →
    (∀ x1 x2 : ℝ, x1 ≠ 0 → x2 ≠ 0 → x1 ≠ x2 → f x1 = f x2 → False) →
    -20 ≤ k ∧ k ≤ -4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l1615_161542


namespace NUMINAMATH_GPT_smallest_sum_of_four_distinct_numbers_l1615_161599

theorem smallest_sum_of_four_distinct_numbers 
  (S : Finset ℤ) 
  (h : S = {8, 26, -2, 13, -4, 0}) :
  ∃ (a b c d : ℤ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a + b + c + d = 2 :=
sorry

end NUMINAMATH_GPT_smallest_sum_of_four_distinct_numbers_l1615_161599


namespace NUMINAMATH_GPT_initial_punch_amount_l1615_161569

theorem initial_punch_amount (P : ℝ) (h1 : 16 = (P / 2 + 2 + 12)) : P = 4 :=
by
  sorry

end NUMINAMATH_GPT_initial_punch_amount_l1615_161569


namespace NUMINAMATH_GPT_power_function_increasing_l1615_161597

theorem power_function_increasing (m : ℝ) : 
  (∀ x : ℝ, 0 < x → (m^2 - 2*m - 2) * x^(-4*m - 2) > 0) ↔ m = -1 :=
by sorry

end NUMINAMATH_GPT_power_function_increasing_l1615_161597


namespace NUMINAMATH_GPT_cost_of_fruits_l1615_161573

-- Definitions based on the conditions
variables (x y z : ℝ)

-- Conditions
axiom h1 : 2 * x + y + 4 * z = 6
axiom h2 : 4 * x + 2 * y + 2 * z = 4

-- Question to prove
theorem cost_of_fruits : 4 * x + 2 * y + 5 * z = 8 :=
sorry

end NUMINAMATH_GPT_cost_of_fruits_l1615_161573


namespace NUMINAMATH_GPT_scientific_notation_correct_l1615_161581

-- Define the number to be converted
def number : ℕ := 3790000

-- Define the correct scientific notation representation
def scientific_notation : ℝ := 3.79 * (10 ^ 6)

-- Statement to prove that number equals scientific_notation
theorem scientific_notation_correct :
  number = 3790000 → scientific_notation = 3.79 * (10 ^ 6) :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_correct_l1615_161581


namespace NUMINAMATH_GPT_number_of_three_digit_integers_congruent_to_2_mod_4_l1615_161598

theorem number_of_three_digit_integers_congruent_to_2_mod_4 : 
  ∃ (count : ℕ), count = 225 ∧ ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ n % 4 = 2 ↔ (∃ k : ℕ, 25 ≤ k ∧ k ≤ 249 ∧ n = 4 * k + 2) := 
by {
  sorry
}

end NUMINAMATH_GPT_number_of_three_digit_integers_congruent_to_2_mod_4_l1615_161598


namespace NUMINAMATH_GPT_inequality_system_solution_l1615_161552

theorem inequality_system_solution (x: ℝ) (h1: 5 * x - 2 < 3 * (x + 2)) (h2: (2 * x - 1) / 3 - (5 * x + 1) / 2 <= 1) : 
  -1 ≤ x ∧ x < 4 :=
sorry

end NUMINAMATH_GPT_inequality_system_solution_l1615_161552


namespace NUMINAMATH_GPT_value_added_to_each_number_is_12_l1615_161579

theorem value_added_to_each_number_is_12
    (sum_original : ℕ)
    (sum_new : ℕ)
    (n : ℕ)
    (avg_original : ℕ)
    (avg_new : ℕ)
    (value_added : ℕ) :
  (n = 15) →
  (avg_original = 40) →
  (avg_new = 52) →
  (sum_original = n * avg_original) →
  (sum_new = n * avg_new) →
  (value_added = (sum_new - sum_original) / n) →
  value_added = 12 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_value_added_to_each_number_is_12_l1615_161579


namespace NUMINAMATH_GPT_problem_solution_l1615_161565

/-- Define the repeating decimal 0.\overline{49} as a rational number. --/
def rep49 := 7 / 9

/-- Define the repeating decimal 0.\overline{4} as a rational number. --/
def rep4 := 4 / 9

/-- The main theorem stating that 99 times the difference between 
    the repeating decimals 0.\overline{49} and 0.\overline{4} equals 5. --/
theorem problem_solution : 99 * (rep49 - rep4) = 5 := by
  sorry

end NUMINAMATH_GPT_problem_solution_l1615_161565


namespace NUMINAMATH_GPT_find_x_l1615_161513

theorem find_x (x : ℝ) (hx1 : x > 0) 
  (h1 : 0.20 * x + 14 = (1 / 3) * ((3 / 4) * x + 21)) : x = 140 :=
sorry

end NUMINAMATH_GPT_find_x_l1615_161513


namespace NUMINAMATH_GPT_line_tangent_to_ellipse_l1615_161577

theorem line_tangent_to_ellipse (k : ℝ) :
  (∃ x : ℝ, 2 * x ^ 2 + 8 * (k * x + 2) ^ 2 = 8 ∧
             ∀ x1 x2 : ℝ, (2 + 8 * k ^ 2) * x1 ^ 2 + 32 * k * x1 + 24 = 0 →
             (2 + 8 * k ^ 2) * x2 ^ 2 + 32 * k * x2 + 24 = 0 → x1 = x2) →
  k^2 = 3 / 4 := by
  sorry

end NUMINAMATH_GPT_line_tangent_to_ellipse_l1615_161577


namespace NUMINAMATH_GPT_probability_sum_eight_l1615_161516

def total_outcomes : ℕ := 36
def favorable_outcomes : ℕ := 5

theorem probability_sum_eight :
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 36 := by
  sorry

end NUMINAMATH_GPT_probability_sum_eight_l1615_161516


namespace NUMINAMATH_GPT_opposite_of_2023_is_neg_2023_l1615_161510

theorem opposite_of_2023_is_neg_2023 (x : ℝ) (h : x = 2023) : -x = -2023 :=
by
  /- proof begins here, but we are skipping it with sorry -/
  sorry

end NUMINAMATH_GPT_opposite_of_2023_is_neg_2023_l1615_161510


namespace NUMINAMATH_GPT_probability_of_fourth_three_is_correct_l1615_161586

noncomputable def p_plus_q : ℚ := 41 + 84

theorem probability_of_fourth_three_is_correct :
  let fair_die_prob := (1 / 6 : ℚ)
  let biased_die_prob := (1 / 2 : ℚ)
  -- Probability of rolling three threes with the fair die:
  let fair_die_three_three_prob := fair_die_prob ^ 3
  -- Probability of rolling three threes with the biased die:
  let biased_die_three_three_prob := biased_die_prob ^ 3
  -- Probability of rolling three threes in total:
  let total_three_three_prob := fair_die_three_three_prob + biased_die_three_three_prob
  -- Probability of using the fair die given three threes
  let fair_die_given_three := fair_die_three_three_prob / total_three_three_prob
  -- Probability of using the biased die given three threes
  let biased_die_given_three := biased_die_three_three_prob / total_three_three_prob
  -- Probability of rolling another three:
  let fourth_three_prob := fair_die_given_three * fair_die_prob + biased_die_given_three * biased_die_prob
  -- Simplifying fraction
  let result_fraction := (41 / 84 : ℚ)
  -- Final answer p + q is 125
  p_plus_q = 125 ∧ fourth_three_prob = result_fraction
:= by
  sorry

end NUMINAMATH_GPT_probability_of_fourth_three_is_correct_l1615_161586


namespace NUMINAMATH_GPT_product_of_solutions_l1615_161562

theorem product_of_solutions :
  (∀ y : ℝ, (|y| = 2 * (|y| - 1)) → y = 2 ∨ y = -2) →
  (∀ y1 y2 : ℝ, (y1 = 2 ∧ y2 = -2) → y1 * y2 = -4) :=
by
  intro h
  have h1 := h 2
  have h2 := h (-2)
  sorry

end NUMINAMATH_GPT_product_of_solutions_l1615_161562


namespace NUMINAMATH_GPT_percentage_error_in_area_l1615_161521

theorem percentage_error_in_area (s : ℝ) (h_s_pos: s > 0) :
  let measured_side := 1.01 * s
  let actual_area := s ^ 2
  let measured_area := measured_side ^ 2
  let error_in_area := measured_area - actual_area
  (error_in_area / actual_area) * 100 = 2.01 :=
by
  sorry

end NUMINAMATH_GPT_percentage_error_in_area_l1615_161521


namespace NUMINAMATH_GPT_arithmetic_example_l1615_161559

theorem arithmetic_example : 15 * 30 + 45 * 15 = 1125 := by
  sorry

end NUMINAMATH_GPT_arithmetic_example_l1615_161559


namespace NUMINAMATH_GPT_positive_difference_of_complementary_angles_in_ratio_5_to_4_l1615_161535

-- Definitions for given conditions
def is_complementary (a b : ℝ) : Prop :=
  a + b = 90

def ratio_5_to_4 (a b : ℝ) : Prop :=
  ∃ x : ℝ, a = 5 * x ∧ b = 4 * x

-- Theorem to prove the measure of their positive difference is 10 degrees
theorem positive_difference_of_complementary_angles_in_ratio_5_to_4
  {a b : ℝ} (h_complementary : is_complementary a b) (h_ratio : ratio_5_to_4 a b) :
  abs (a - b) = 10 :=
by 
  sorry

end NUMINAMATH_GPT_positive_difference_of_complementary_angles_in_ratio_5_to_4_l1615_161535


namespace NUMINAMATH_GPT_log2_bounds_158489_l1615_161540

theorem log2_bounds_158489 :
  (2^16 = 65536) ∧ (2^17 = 131072) ∧ (65536 < 158489 ∧ 158489 < 131072) →
  (16 < Real.log 158489 / Real.log 2 ∧ Real.log 158489 / Real.log 2 < 17) ∧ 16 + 17 = 33 :=
by
  intro h
  have h1 : 2^16 = 65536 := h.1
  have h2 : 2^17 = 131072 := h.2.1
  have h3 : 65536 < 158489 := h.2.2.1
  have h4 : 158489 < 131072 := h.2.2.2
  sorry

end NUMINAMATH_GPT_log2_bounds_158489_l1615_161540


namespace NUMINAMATH_GPT_linear_function_positive_in_interval_abc_sum_greater_negative_one_l1615_161537

-- Problem 1
theorem linear_function_positive_in_interval (f : ℝ → ℝ) (k h m n : ℝ) (hk : k ≠ 0) (hmn : m < n)
  (hf_m : f m > 0) (hf_n : f n > 0) : (∀ x : ℝ, m < x ∧ x < n → f x > 0) :=
sorry

-- Problem 2
theorem abc_sum_greater_negative_one (a b c : ℝ)
  (ha : abs a < 1) (hb : abs b < 1) (hc : abs c < 1) : a * b + b * c + c * a > -1 :=
sorry

end NUMINAMATH_GPT_linear_function_positive_in_interval_abc_sum_greater_negative_one_l1615_161537


namespace NUMINAMATH_GPT_slope_range_l1615_161514

noncomputable def directed_distance (a b c x0 y0 : ℝ) : ℝ :=
  (a * x0 + b * y0 + c) / (Real.sqrt (a^2 + b^2))

theorem slope_range {A B P : ℝ × ℝ} (hA : A = (-1, 0)) (hB : B = (1, 0)) (hP : P = (3, 0))
                   {C : ℝ × ℝ} (hC : ∃ θ : ℝ, C = (9 * Real.cos θ, 18 + 9 * Real.sin θ))
                   {a b c : ℝ} (h_line : c = -3 * a)
                   (h_sum_distances : directed_distance a b c (-1) 0 +
                                      directed_distance a b c 1 0 +
                                      directed_distance a b c (9 * Real.cos θ) (18 + 9 * Real.sin θ) = 0) :
  -3 ≤ - (a / b) ∧ - (a / b) ≤ -1 := sorry

end NUMINAMATH_GPT_slope_range_l1615_161514


namespace NUMINAMATH_GPT_prime_not_divisor_ab_cd_l1615_161532

theorem prime_not_divisor_ab_cd {a b c d : ℕ} (ha: 0 < a) (hb: 0 < b) (hc: 0 < c) (hd: 0 < d) 
  (p : ℕ) (hp : p = a + b + c + d) (hprime : Nat.Prime p) : ¬ p ∣ (a * b - c * d) := 
sorry

end NUMINAMATH_GPT_prime_not_divisor_ab_cd_l1615_161532


namespace NUMINAMATH_GPT_average_earning_week_l1615_161507

theorem average_earning_week (D1 D2 D3 D4 D5 D6 D7 : ℝ) 
  (h1 : (D1 + D2 + D3 + D4) / 4 = 18)
  (h2 : (D4 + D5 + D6 + D7) / 4 = 22)
  (h3 : D4 = 13) : 
  (D1 + D2 + D3 + D4 + D5 + D6 + D7) / 7 = 22.86 := 
by 
  sorry

end NUMINAMATH_GPT_average_earning_week_l1615_161507


namespace NUMINAMATH_GPT_find_y_l1615_161504

theorem find_y (x y : ℕ) (h1 : x^2 = y + 3) (h2 : x = 6) : y = 33 := 
by
  sorry

end NUMINAMATH_GPT_find_y_l1615_161504


namespace NUMINAMATH_GPT_book_stack_sum_l1615_161545

theorem book_stack_sum : 
  let a := 15 -- first term
  let d := -2 -- common difference
  let l := 1 -- last term
  -- n = (l - a) / d + 1
  let n := (l - a) / d + 1
  -- S = n * (a + l) / 2
  let S := n * (a + l) / 2
  S = 64 :=
by
  -- The given conditions
  let a := 15 -- first term
  let d := -2 -- common difference
  let l := 1 -- last term
  -- Calculate the number of terms (n)
  let n := (l - a) / d + 1
  -- Calculate the total sum (S)
  let S := n * (a + l) / 2
  -- Prove the sum is 64
  show S = 64
  sorry

end NUMINAMATH_GPT_book_stack_sum_l1615_161545


namespace NUMINAMATH_GPT_eggs_used_to_bake_cake_l1615_161500

theorem eggs_used_to_bake_cake
    (initial_eggs : ℕ)
    (omelet_eggs : ℕ)
    (aunt_eggs : ℕ)
    (meal_eggs : ℕ)
    (num_meals : ℕ)
    (remaining_eggs_after_omelet : initial_eggs - omelet_eggs = 22)
    (eggs_given_to_aunt : 2 * aunt_eggs = initial_eggs - omelet_eggs)
    (remaining_eggs_after_aunt : initial_eggs - omelet_eggs - aunt_eggs = 11)
    (total_eggs_for_meals : meal_eggs * num_meals = 9)
    (remaining_eggs_after_meals : initial_eggs - omelet_eggs - aunt_eggs - meal_eggs * num_meals = 2) :
  initial_eggs - omelet_eggs - aunt_eggs - meal_eggs * num_meals = 2 :=
sorry

end NUMINAMATH_GPT_eggs_used_to_bake_cake_l1615_161500


namespace NUMINAMATH_GPT_abs_inequality_no_solution_l1615_161522

theorem abs_inequality_no_solution (a : ℝ) : (∀ x : ℝ, |x - 5| + |x + 3| ≥ a) ↔ a ≤ 8 :=
by sorry

end NUMINAMATH_GPT_abs_inequality_no_solution_l1615_161522


namespace NUMINAMATH_GPT_avg_difference_l1615_161595

def avg (a b c : ℕ) := (a + b + c) / 3

theorem avg_difference : avg 14 32 53 - avg 21 47 22 = 3 :=
by
  sorry

end NUMINAMATH_GPT_avg_difference_l1615_161595


namespace NUMINAMATH_GPT_relationship_xy_l1615_161575

def M (x : ℤ) : Prop := ∃ m : ℤ, x = 3 * m + 1
def N (y : ℤ) : Prop := ∃ n : ℤ, y = 3 * n + 2

theorem relationship_xy (x y : ℤ) (hx : M x) (hy : N y) : N (x * y) ∧ ¬ M (x * y) :=
by
  sorry

end NUMINAMATH_GPT_relationship_xy_l1615_161575


namespace NUMINAMATH_GPT_fraction_of_salary_spent_on_house_rent_l1615_161511

theorem fraction_of_salary_spent_on_house_rent
    (S : ℕ) (H : ℚ)
    (cond1 : S = 180000)
    (cond2 : S / 5 + H * S + 3 * S / 5 + 18000 = S) :
    H = 1 / 10 := by
  sorry

end NUMINAMATH_GPT_fraction_of_salary_spent_on_house_rent_l1615_161511


namespace NUMINAMATH_GPT_f_bounded_l1615_161570

noncomputable def f : ℝ → ℝ := sorry

axiom f_property : ∀ x : ℝ, f (3 * x) = 3 * f x - 4 * (f x) ^ 3

axiom f_continuous_at_zero : ContinuousAt f 0

theorem f_bounded : ∀ x : ℝ, |f x| ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_f_bounded_l1615_161570


namespace NUMINAMATH_GPT_stephen_total_distance_l1615_161519

def speed_first_segment := 16 -- miles per hour
def time_first_segment := 10 / 60 -- hours

def speed_second_segment := 12 -- miles per hour
def headwind := 2 -- miles per hour
def actual_speed_second_segment := speed_second_segment - headwind
def time_second_segment := 20 / 60 -- hours

def speed_third_segment := 20 -- miles per hour
def tailwind := 4 -- miles per hour
def actual_speed_third_segment := speed_third_segment + tailwind
def time_third_segment := 15 / 60 -- hours

def distance_first_segment := speed_first_segment * time_first_segment
def distance_second_segment := actual_speed_second_segment * time_second_segment
def distance_third_segment := actual_speed_third_segment * time_third_segment

theorem stephen_total_distance : distance_first_segment + distance_second_segment + distance_third_segment = 12 := by
  sorry

end NUMINAMATH_GPT_stephen_total_distance_l1615_161519


namespace NUMINAMATH_GPT_black_beans_count_l1615_161593

theorem black_beans_count (B G O : ℕ) (h₁ : G = B + 2) (h₂ : O = G - 1) (h₃ : B + G + O = 27) : B = 8 := by
  sorry

end NUMINAMATH_GPT_black_beans_count_l1615_161593


namespace NUMINAMATH_GPT_total_oranges_proof_l1615_161512

def jeremyMonday : ℕ := 100
def jeremyTuesdayPlusBrother : ℕ := 3 * jeremyMonday
def jeremyWednesdayPlusBrotherPlusCousin : ℕ := 2 * jeremyTuesdayPlusBrother
def jeremyThursday : ℕ := (70 * jeremyMonday) / 100
def cousinWednesday : ℕ := jeremyTuesdayPlusBrother - (20 * jeremyTuesdayPlusBrother) / 100
def cousinThursday : ℕ := cousinWednesday + (30 * cousinWednesday) / 100

def total_oranges : ℕ :=
  jeremyMonday + jeremyTuesdayPlusBrother + jeremyWednesdayPlusBrotherPlusCousin + (jeremyThursday + (jeremyWednesdayPlusBrotherPlusCousin - cousinWednesday) + cousinThursday)

theorem total_oranges_proof : total_oranges = 1642 :=
by
  sorry

end NUMINAMATH_GPT_total_oranges_proof_l1615_161512


namespace NUMINAMATH_GPT_curves_intersect_exactly_three_points_l1615_161534

theorem curves_intersect_exactly_three_points (a : ℝ) :
  (∃! (p : ℝ × ℝ), p.1 ^ 2 + p.2 ^ 2 = a ^ 2 ∧ p.2 = p.1 ^ 2 - a) ↔ a > (1 / 2) :=
by sorry

end NUMINAMATH_GPT_curves_intersect_exactly_three_points_l1615_161534


namespace NUMINAMATH_GPT_eggs_per_chicken_per_week_l1615_161544

-- Define the conditions
def chickens : ℕ := 10
def price_per_dozen : ℕ := 2  -- in dollars
def earnings_in_2_weeks : ℕ := 20  -- in dollars
def weeks : ℕ := 2
def eggs_per_dozen : ℕ := 12

-- Define the question as a theorem to be proved
theorem eggs_per_chicken_per_week : 
  (earnings_in_2_weeks / price_per_dozen) * eggs_per_dozen / (chickens * weeks) = 6 :=
by
  -- proof steps
  sorry

end NUMINAMATH_GPT_eggs_per_chicken_per_week_l1615_161544


namespace NUMINAMATH_GPT_binary_predecessor_l1615_161531

def M : ℕ := 84
def N : ℕ := 83
def M_bin : ℕ := 0b1010100
def N_bin : ℕ := 0b1010011

theorem binary_predecessor (H : M = M_bin ∧ N = M - 1) : N = N_bin := by
  sorry

end NUMINAMATH_GPT_binary_predecessor_l1615_161531


namespace NUMINAMATH_GPT_consecutive_odd_integers_sum_l1615_161526

theorem consecutive_odd_integers_sum (a b c : ℤ) (h1 : a % 2 = 1) (h2 : b % 2 = 1) (h3 : c % 2 = 1) (h4 : a < b) (h5 : b < c) (h6 : c = -47) : a + b + c = -141 := 
sorry

end NUMINAMATH_GPT_consecutive_odd_integers_sum_l1615_161526


namespace NUMINAMATH_GPT_jesse_initial_blocks_l1615_161509

def total_blocks_initial (blocks_cityscape blocks_farmhouse blocks_zoo blocks_first_area blocks_second_area blocks_third_area blocks_left : ℕ) : ℕ :=
  blocks_cityscape + blocks_farmhouse + blocks_zoo + blocks_first_area + blocks_second_area + blocks_third_area + blocks_left

theorem jesse_initial_blocks :
  total_blocks_initial 80 123 95 57 43 62 84 = 544 :=
sorry

end NUMINAMATH_GPT_jesse_initial_blocks_l1615_161509


namespace NUMINAMATH_GPT_equilateral_triangle_l1615_161556

theorem equilateral_triangle
  (a b c : ℝ) (α β γ : ℝ) (p R : ℝ)
  (h : (a * Real.cos α + b * Real.cos β + c * Real.cos γ) / (a * Real.sin β + b * Real.sin γ + c * Real.sin α) = p / (9 * R)) :
  a = b ∧ b = c ∧ a = c :=
sorry

end NUMINAMATH_GPT_equilateral_triangle_l1615_161556


namespace NUMINAMATH_GPT_no_real_solution_l1615_161589

theorem no_real_solution :
  ¬ ∃ x : ℝ, (1 / (x + 2) + 8 / (x + 6) ≥ 2) ∧ (5 / (x + 1) - 2 ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_no_real_solution_l1615_161589


namespace NUMINAMATH_GPT_total_ticket_sales_cost_l1615_161527

theorem total_ticket_sales_cost
  (num_orchestra num_balcony : ℕ)
  (price_orchestra price_balcony : ℕ)
  (total_tickets total_revenue : ℕ)
  (h1 : num_orchestra + num_balcony = 370)
  (h2 : num_balcony = num_orchestra + 190)
  (h3 : price_orchestra = 12)
  (h4 : price_balcony = 8)
  (h5 : total_tickets = 370)
  : total_revenue = 3320 := by
  sorry

end NUMINAMATH_GPT_total_ticket_sales_cost_l1615_161527


namespace NUMINAMATH_GPT_original_prices_sum_l1615_161558

theorem original_prices_sum
  (new_price_candy_box : ℝ)
  (new_price_soda_can : ℝ)
  (increase_candy_box : ℝ)
  (increase_soda_can : ℝ)
  (h1 : new_price_candy_box = 10)
  (h2 : new_price_soda_can = 9)
  (h3 : increase_candy_box = 0.25)
  (h4 : increase_soda_can = 0.50) :
  let original_price_candy_box := new_price_candy_box / (1 + increase_candy_box)
  let original_price_soda_can := new_price_soda_can / (1 + increase_soda_can)
  original_price_candy_box + original_price_soda_can = 19 :=
by
  sorry

end NUMINAMATH_GPT_original_prices_sum_l1615_161558


namespace NUMINAMATH_GPT_greatest_odd_factors_under_150_l1615_161580

theorem greatest_odd_factors_under_150 : ∃ (n : ℕ), n < 150 ∧ ( ∃ (k : ℕ), n = k * k ) ∧ (∀ m : ℕ, m < 150 ∧ ( ∃ (k : ℕ), m = k * k ) → m ≤ 144) :=
by
  sorry

end NUMINAMATH_GPT_greatest_odd_factors_under_150_l1615_161580


namespace NUMINAMATH_GPT_function_behavior_l1615_161508

noncomputable def f (x : ℝ) : ℝ := abs (2^x - 2)

theorem function_behavior :
  (∀ x y : ℝ, x < y ∧ y ≤ 1 → f x ≥ f y) ∧ (∀ x y : ℝ, x < y ∧ x ≥ 1 → f x ≤ f y) :=
by
  sorry

end NUMINAMATH_GPT_function_behavior_l1615_161508


namespace NUMINAMATH_GPT_find_num_3_year_olds_l1615_161574

noncomputable def num_4_year_olds := 20
noncomputable def num_5_year_olds := 15
noncomputable def num_6_year_olds := 22
noncomputable def average_class_size := 35
noncomputable def num_students_class1 (num_3_year_olds : ℕ) := num_3_year_olds + num_4_year_olds
noncomputable def num_students_class2 := num_5_year_olds + num_6_year_olds
noncomputable def total_students (num_3_year_olds : ℕ) := num_students_class1 num_3_year_olds + num_students_class2

theorem find_num_3_year_olds (num_3_year_olds : ℕ) : 
  (total_students num_3_year_olds) / 2 = average_class_size → num_3_year_olds = 13 :=
by
  sorry

end NUMINAMATH_GPT_find_num_3_year_olds_l1615_161574


namespace NUMINAMATH_GPT_find_ab_l1615_161506

-- Define the "¤" operation
def op (x y : ℝ) := (x + y)^2 - (x - y)^2

-- The Lean 4 theorem statement
theorem find_ab (a b : ℝ) (h : op a b = 24) : a * b = 6 := 
by
  -- We leave the proof as an exercise
  sorry

end NUMINAMATH_GPT_find_ab_l1615_161506


namespace NUMINAMATH_GPT_reporters_covering_local_politics_l1615_161502

theorem reporters_covering_local_politics (R : ℕ) (P Q A B : ℕ)
  (h1 : P = 70)
  (h2 : Q = 100 - P)
  (h3 : A = 40)
  (h4 : B = 100 - A) :
  B % 30 = 18 :=
by
  sorry

end NUMINAMATH_GPT_reporters_covering_local_politics_l1615_161502


namespace NUMINAMATH_GPT_find_number_of_boys_l1615_161518

noncomputable def number_of_boys (B G : ℕ) : Prop :=
  (B : ℚ) / (G : ℚ) = 7.5 / 15.4 ∧ G = B + 174

theorem find_number_of_boys : ∃ B G : ℕ, number_of_boys B G ∧ B = 165 := 
by 
  sorry

end NUMINAMATH_GPT_find_number_of_boys_l1615_161518


namespace NUMINAMATH_GPT_compute_105_squared_l1615_161592

theorem compute_105_squared :
  let a := 100
  let b := 5
  (a + b)^2 = 11025 :=
by
  sorry

end NUMINAMATH_GPT_compute_105_squared_l1615_161592


namespace NUMINAMATH_GPT_original_number_l1615_161547

theorem original_number (x y : ℝ) (h1 : 10 * x + 22 * y = 780) (h2 : y = 34) : x + y = 37.2 :=
sorry

end NUMINAMATH_GPT_original_number_l1615_161547


namespace NUMINAMATH_GPT_determine_c_plus_d_l1615_161568

theorem determine_c_plus_d (x : ℝ) (c d : ℤ) (h1 : x^2 + 5*x + (5/x) + (1/(x^2)) = 35) (h2 : x = c + Real.sqrt d) : c + d = 5 :=
sorry

end NUMINAMATH_GPT_determine_c_plus_d_l1615_161568


namespace NUMINAMATH_GPT_increasing_iff_a_gt_neg1_l1615_161539

noncomputable def increasing_function_condition (a : ℝ) (b : ℝ) (x : ℝ) : Prop :=
  let y := (a + 1) * x + b
  a > -1

theorem increasing_iff_a_gt_neg1 (a : ℝ) (b : ℝ) : (∀ x : ℝ, (a + 1) > 0) ↔ a > -1 :=
by
  sorry

end NUMINAMATH_GPT_increasing_iff_a_gt_neg1_l1615_161539


namespace NUMINAMATH_GPT_ascending_order_conversion_l1615_161546

def convert_base (num : Nat) (base : Nat) : Nat :=
  match num with
  | 0 => 0
  | _ => (num / 10) * base + (num % 10)

theorem ascending_order_conversion :
  let num16 := 12
  let num7 := 25
  let num4 := 33
  let base16 := 16
  let base7 := 7
  let base4 := 4
  convert_base num4 base4 < convert_base num16 base16 ∧ 
  convert_base num16 base16 < convert_base num7 base7 :=
by
  -- Here would be the proof, but we skip it
  sorry

end NUMINAMATH_GPT_ascending_order_conversion_l1615_161546


namespace NUMINAMATH_GPT_solution_set_inequality_l1615_161536

open Set

variable {a b : ℝ}

/-- Proof Problem Statement -/
theorem solution_set_inequality (h : ∀ x : ℝ, -3 < x ∧ x < -1 ↔ a * x^2 - 1999 * x + b > 0) : 
  ∀ x : ℝ, 1 < x ∧ x < 3 ↔ a * x^2 + 1999 * x + b > 0 :=
sorry

end NUMINAMATH_GPT_solution_set_inequality_l1615_161536


namespace NUMINAMATH_GPT_calvin_score_l1615_161591

theorem calvin_score (C : ℚ) (h_paislee_score : (3/4) * C = 125) : C = 167 := 
  sorry

end NUMINAMATH_GPT_calvin_score_l1615_161591


namespace NUMINAMATH_GPT_next_in_sequence_is_80_l1615_161567

def seq (n : ℕ) : ℕ := n^2 - 1

theorem next_in_sequence_is_80 :
  seq 9 = 80 :=
by
  sorry

end NUMINAMATH_GPT_next_in_sequence_is_80_l1615_161567


namespace NUMINAMATH_GPT_whole_process_time_is_9_l1615_161553

variable (BleachingTime : ℕ)
variable (DyeingTime : ℕ)

-- Conditions
axiom bleachingTime_is_3 : BleachingTime = 3
axiom dyeingTime_is_twice_bleachingTime : DyeingTime = 2 * BleachingTime

-- Question and Proof Problem
theorem whole_process_time_is_9 (BleachingTime : ℕ) (DyeingTime : ℕ)
  (h1 : BleachingTime = 3) (h2 : DyeingTime = 2 * BleachingTime) : 
  (BleachingTime + DyeingTime) = 9 :=
  by
  sorry

end NUMINAMATH_GPT_whole_process_time_is_9_l1615_161553
