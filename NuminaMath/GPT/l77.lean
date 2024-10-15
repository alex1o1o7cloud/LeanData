import Mathlib

namespace NUMINAMATH_GPT_find_circle_center_l77_7738

def circle_center_eq : Prop :=
  ∃ (x y : ℝ), (x^2 - 6 * x + y^2 + 2 * y - 12 = 0) ∧ (x = 3) ∧ (y = -1)

theorem find_circle_center : circle_center_eq :=
sorry

end NUMINAMATH_GPT_find_circle_center_l77_7738


namespace NUMINAMATH_GPT_sum_digits_10_pow_100_minus_100_l77_7750

open Nat

/-- Define the condition: 10^100 - 100 as an expression. -/
def subtract_100_from_power_10 (n : ℕ) : ℕ :=
  10^n - 100

/-- Sum the digits of a natural number. -/
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- The goal is to prove the sum of the digits of 10^100 - 100 equals 882. -/
theorem sum_digits_10_pow_100_minus_100 :
  sum_of_digits (subtract_100_from_power_10 100) = 882 :=
by
  sorry

end NUMINAMATH_GPT_sum_digits_10_pow_100_minus_100_l77_7750


namespace NUMINAMATH_GPT_percentage_increase_in_freelance_l77_7785

open Real

def initial_part_time_earnings := 65
def new_part_time_earnings := 72
def initial_freelance_earnings := 45
def new_freelance_earnings := 72

theorem percentage_increase_in_freelance :
  (new_freelance_earnings - initial_freelance_earnings) / initial_freelance_earnings * 100 = 60 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_percentage_increase_in_freelance_l77_7785


namespace NUMINAMATH_GPT_find_line_through_midpoint_of_hyperbola_l77_7795

theorem find_line_through_midpoint_of_hyperbola
  (x1 y1 x2 y2 : ℝ)
  (P : ℝ × ℝ := (4, 1))
  (A : ℝ × ℝ := (x1, y1))
  (B : ℝ × ℝ := (x2, y2))
  (H_midpoint : P = ((x1 + x2) / 2, (y1 + y2) / 2))
  (H_hyperbola_A : (x1^2 / 4 - y1^2 = 1))
  (H_hyperbola_B : (x2^2 / 4 - y2^2 = 1)) :
  ∃ m b : ℝ, (m = 1) ∧ (b = 3) ∧ (∀ x y : ℝ, y = m * x + b → x - y - 3 = 0) := by
  sorry

end NUMINAMATH_GPT_find_line_through_midpoint_of_hyperbola_l77_7795


namespace NUMINAMATH_GPT_intersection_nonempty_l77_7766

open Nat

theorem intersection_nonempty (a : ℕ) (ha : a ≥ 2) :
  ∃ (b : ℕ), b = 1 ∨ b = a ∧
  ∃ y, (∃ x, y = a^x ∧ x ≥ 1) ∧
       (∃ x, y = (a + 1)^x + b ∧ x ≥ 1) :=
by sorry

end NUMINAMATH_GPT_intersection_nonempty_l77_7766


namespace NUMINAMATH_GPT_solve_eq_integers_l77_7703

theorem solve_eq_integers (x y : ℤ) : 
    x^2 - x * y - 6 * y^2 + 2 * x + 19 * y = 18 ↔ (x = 2 ∧ y = 2) ∨ (x = -2 ∧ y = 2) := by
    sorry

end NUMINAMATH_GPT_solve_eq_integers_l77_7703


namespace NUMINAMATH_GPT_sequence_length_l77_7709

theorem sequence_length 
  (a : ℕ)
  (b : ℕ)
  (d : ℕ)
  (steps : ℕ)
  (h1 : a = 160)
  (h2 : b = 28)
  (h3 : d = 4)
  (h4 : (28:ℕ) = (160:ℕ) - steps * 4) :
  steps + 1 = 34 :=
by
  sorry

end NUMINAMATH_GPT_sequence_length_l77_7709


namespace NUMINAMATH_GPT_remainder_8_pow_2023_mod_5_l77_7732

theorem remainder_8_pow_2023_mod_5 :
  8 ^ 2023 % 5 = 2 :=
by
  sorry

end NUMINAMATH_GPT_remainder_8_pow_2023_mod_5_l77_7732


namespace NUMINAMATH_GPT_commute_times_abs_diff_l77_7769

def commute_times_avg (x y : ℝ) : Prop := (x + y + 7 + 8 + 9) / 5 = 8
def commute_times_var (x y : ℝ) : Prop := ((x - 8)^2 + (y - 8)^2 + (7 - 8)^2 + (8 - 8)^2 + (9 - 8)^2) / 5 = 4

theorem commute_times_abs_diff (x y : ℝ) (h_avg : commute_times_avg x y) (h_var : commute_times_var x y) :
  |x - y| = 6 :=
sorry

end NUMINAMATH_GPT_commute_times_abs_diff_l77_7769


namespace NUMINAMATH_GPT_sum_and_divide_repeating_decimals_l77_7794

noncomputable def repeating_decimal_83 : ℚ := 83 / 99
noncomputable def repeating_decimal_18 : ℚ := 18 / 99

theorem sum_and_divide_repeating_decimals :
  (repeating_decimal_83 + repeating_decimal_18) / (1 / 5) = 505 / 99 :=
by
  sorry

end NUMINAMATH_GPT_sum_and_divide_repeating_decimals_l77_7794


namespace NUMINAMATH_GPT_mark_hours_per_week_l77_7788

theorem mark_hours_per_week (w_historical : ℕ) (w_spring : ℕ) (h_spring : ℕ) (e_spring : ℕ) (e_goal : ℕ) (w_goal : ℕ) (h_goal : ℚ) :
  (e_spring : ℚ) / (w_historical * w_spring) = h_spring / w_spring →
  e_goal = 21000 →
  w_goal = 50 →
  h_spring = 35 →
  w_spring = 15 →
  e_spring = 4200 →
  (h_goal : ℚ) = 2625 / w_goal →
  h_goal = 52.5 :=
sorry

end NUMINAMATH_GPT_mark_hours_per_week_l77_7788


namespace NUMINAMATH_GPT_find_a_plus_b_l77_7776

theorem find_a_plus_b {f : ℝ → ℝ} (a b : ℝ) :
  (∀ x, f x = x^3 + 3*x^2 + 6*x + 14) →
  f a = 1 →
  f b = 19 →
  a + b = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l77_7776


namespace NUMINAMATH_GPT_number_of_girls_joined_l77_7717

-- Define the initial conditions
def initial_girls := 18
def initial_boys := 15
def boys_quit := 4
def total_children_after_changes := 36

-- Define the changes
def boys_after_quit := initial_boys - boys_quit
def girls_after_changes := total_children_after_changes - boys_after_quit
def girls_joined := girls_after_changes - initial_girls

-- State the theorem
theorem number_of_girls_joined :
  girls_joined = 7 :=
by
  sorry

end NUMINAMATH_GPT_number_of_girls_joined_l77_7717


namespace NUMINAMATH_GPT_blueBirdChessTeam72_l77_7762

def blueBirdChessTeamArrangements : Nat :=
  let boys_girls_ends := 3 * 3 + 3 * 3
  let alternate_arrangements := 2 * 2
  boys_girls_ends * alternate_arrangements

theorem blueBirdChessTeam72 : blueBirdChessTeamArrangements = 72 := by
  unfold blueBirdChessTeamArrangements
  sorry

end NUMINAMATH_GPT_blueBirdChessTeam72_l77_7762


namespace NUMINAMATH_GPT_tan_3theta_l77_7758

theorem tan_3theta (θ : ℝ) (h : Real.tan θ = 3 / 4) : Real.tan (3 * θ) = -12.5 :=
sorry

end NUMINAMATH_GPT_tan_3theta_l77_7758


namespace NUMINAMATH_GPT_smallest_n_l77_7779

theorem smallest_n (n : ℕ) (h1 : n % 6 = 4) (h2 : n % 7 = 3) (h3 : n % 8 = 5) (h4 : n > 20) : n = 136 := by
  sorry

end NUMINAMATH_GPT_smallest_n_l77_7779


namespace NUMINAMATH_GPT_necessary_condition_for_positive_on_interval_l77_7715

theorem necessary_condition_for_positive_on_interval (a b : ℝ) (h : a + 2 * b > 0) :
  (∀ x, 0 ≤ x → x ≤ 1 → (a * x + b) > 0) ↔ ∃ c, 0 < c ∧ c ≤ 1 ∧ a + 2 * b > 0 ∧ ¬∀ d, 0 < d ∧ d ≤ 1 → a * d + b > 0 := 
by 
  sorry

end NUMINAMATH_GPT_necessary_condition_for_positive_on_interval_l77_7715


namespace NUMINAMATH_GPT_intersection_M_N_union_complements_M_N_l77_7740

open Set

def U : Set ℝ := univ
def M : Set ℝ := {x | x ≥ 1}
def N : Set ℝ := {x | 0 ≤ x ∧ x < 5}

theorem intersection_M_N :
  M ∩ N = {x | 1 ≤ x ∧ x < 5} :=
by {
  sorry
}

theorem union_complements_M_N :
  (compl M) ∪ (compl N) = {x | x < 1 ∨ x ≥ 5} :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_M_N_union_complements_M_N_l77_7740


namespace NUMINAMATH_GPT_triangle_sides_consecutive_obtuse_l77_7774

/-- Given the sides of a triangle are consecutive natural numbers 
    and the largest angle is obtuse, 
    the lengths of the sides in ascending order are 2, 3, 4. -/
theorem triangle_sides_consecutive_obtuse 
    (x : ℕ) (hx : x > 1) 
    (cos_alpha_neg : (x - 4) < 0) 
    (x_lt_4 : x < 4) :
    (x = 3) → (∃ a b c : ℕ, a < b ∧ b < c ∧ a + b > c ∧ a = 2 ∧ b = 3 ∧ c = 4) :=
by
  intro hx3
  use 2, 3, 4
  repeat {split}
  any_goals {linarith}
  all_goals {sorry}

end NUMINAMATH_GPT_triangle_sides_consecutive_obtuse_l77_7774


namespace NUMINAMATH_GPT_problem1_problem2_l77_7742

noncomputable def f (x a b c : ℝ) : ℝ := abs (x + a) + abs (x - b) + c

theorem problem1 (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c)
  (h₃ : ∃ x, f x a b c = 4) : a + b + c = 4 :=
sorry

theorem problem2 (a b c : ℝ) (h : a + b + c = 4) : (1 / a) + (1 / b) + (1 / c) ≥ 9 / 4 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l77_7742


namespace NUMINAMATH_GPT_square_perimeter_ratio_l77_7723

theorem square_perimeter_ratio (x y : ℝ)
(h : (x / y) ^ 2 = 16 / 25) : (4 * x) / (4 * y) = 4 / 5 :=
by sorry

end NUMINAMATH_GPT_square_perimeter_ratio_l77_7723


namespace NUMINAMATH_GPT_largest_and_next_largest_difference_l77_7771

theorem largest_and_next_largest_difference (a b c : ℕ) (h1: a = 10) (h2: b = 11) (h3: c = 12) : 
  let largest := max a (max b c)
  let next_largest := min (max a b) (max (min a b) c)
  largest - next_largest = 1 :=
by
  -- Proof to be filled in for verification
  sorry

end NUMINAMATH_GPT_largest_and_next_largest_difference_l77_7771


namespace NUMINAMATH_GPT_trigonometric_expression_evaluation_l77_7713

theorem trigonometric_expression_evaluation (θ : ℝ) (hθ : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_expression_evaluation_l77_7713


namespace NUMINAMATH_GPT_power_function_properties_l77_7772

theorem power_function_properties (m : ℤ) :
  (m^2 - 2 * m - 2 ≠ 0) ∧ (m^2 + 4 * m < 0) ∧ (m^2 + 4 * m % 2 = 1) → m = -1 := by
  intro h
  sorry

end NUMINAMATH_GPT_power_function_properties_l77_7772


namespace NUMINAMATH_GPT_smallest_sum_of_consecutive_primes_divisible_by_5_l77_7764

def consecutive_primes (n : Nat) : Prop :=
  -- Define what it means to be 4 consecutive prime numbers
  Nat.Prime n ∧ Nat.Prime (n + 2) ∧ Nat.Prime (n + 6) ∧ Nat.Prime (n + 8)

def sum_of_consecutive_primes (n : Nat) : Nat :=
  n + (n + 2) + (n + 6) + (n + 8)

theorem smallest_sum_of_consecutive_primes_divisible_by_5 :
  ∃ n, n > 10 ∧ consecutive_primes n ∧ sum_of_consecutive_primes n % 5 = 0 ∧ sum_of_consecutive_primes n = 60 :=
by
  sorry

end NUMINAMATH_GPT_smallest_sum_of_consecutive_primes_divisible_by_5_l77_7764


namespace NUMINAMATH_GPT_dimes_given_l77_7786

theorem dimes_given (initial_dimes final_dimes dimes_dad_gave : ℕ)
  (h1 : initial_dimes = 9)
  (h2 : final_dimes = 16)
  (h3 : final_dimes = initial_dimes + dimes_dad_gave) :
  dimes_dad_gave = 7 :=
by
  rw [h1, h2] at h3
  linarith

end NUMINAMATH_GPT_dimes_given_l77_7786


namespace NUMINAMATH_GPT_amy_bike_total_l77_7792

-- Define the miles Amy biked yesterday
def y : ℕ := 12

-- Define the miles Amy biked today
def t : ℕ := 2 * y - 3

-- Define the total miles Amy biked in two days
def total : ℕ := y + t

-- The theorem stating the total distance biked equals 33 miles
theorem amy_bike_total : total = 33 := by
  sorry

end NUMINAMATH_GPT_amy_bike_total_l77_7792


namespace NUMINAMATH_GPT_calculate_result_l77_7760

theorem calculate_result (x : ℝ) : (-x^3)^3 = -x^9 :=
by {
  sorry  -- Proof not required per instructions
}

end NUMINAMATH_GPT_calculate_result_l77_7760


namespace NUMINAMATH_GPT_termite_ridden_fraction_l77_7702

theorem termite_ridden_fraction (T : ℝ)
  (h1 : (3 / 10) * T = 0.1) : T = 1 / 3 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_termite_ridden_fraction_l77_7702


namespace NUMINAMATH_GPT_arithmetic_seq_proof_l77_7710

open Nat

-- Define the arithmetic sequence and its properties
def arithmetic_seq (a d : ℕ → ℤ) : Prop :=
∀ n, a (n + 1) = a n + d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_of_arithmetic_seq (a : ℕ → ℤ) (d : ℤ) (n : ℕ) : ℤ :=
n * (a 1) + n * (n - 1) / 2 * d

theorem arithmetic_seq_proof (a : ℕ → ℤ) (d : ℤ)
  (h1 : arithmetic_seq a d)
  (h2 : a 2 = 0)
  (h3 : sum_of_arithmetic_seq a d 3 + sum_of_arithmetic_seq a d 4 = 6) :
  a 5 + a 6 = 21 :=
sorry

end NUMINAMATH_GPT_arithmetic_seq_proof_l77_7710


namespace NUMINAMATH_GPT_nabla_four_seven_l77_7731

def nabla (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

theorem nabla_four_seven : nabla 4 7 = 11 / 29 :=
by
  sorry

end NUMINAMATH_GPT_nabla_four_seven_l77_7731


namespace NUMINAMATH_GPT_find_p_l77_7798

theorem find_p 
  (h : {x | x^2 - 5 * x + p ≥ 0} = {x | x ≤ -1 ∨ x ≥ 6}) : p = -6 :=
by
  sorry

end NUMINAMATH_GPT_find_p_l77_7798


namespace NUMINAMATH_GPT_least_four_digit_multiple_3_5_7_l77_7745

theorem least_four_digit_multiple_3_5_7 : ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ (3 ∣ n) ∧ (5 ∣ n) ∧ (7 ∣ n) ∧ n = 1050 :=
by
  use 1050
  repeat {sorry}

end NUMINAMATH_GPT_least_four_digit_multiple_3_5_7_l77_7745


namespace NUMINAMATH_GPT_non_degenerate_ellipse_l77_7730

theorem non_degenerate_ellipse (k : ℝ) : 
    (∃ x y : ℝ, x^2 + 9 * y^2 - 6 * x + 18 * y = k) ↔ k > -18 :=
sorry

end NUMINAMATH_GPT_non_degenerate_ellipse_l77_7730


namespace NUMINAMATH_GPT_rowing_upstream_speed_l77_7700

theorem rowing_upstream_speed (V_down V_m : ℝ) (h_down : V_down = 35) (h_still : V_m = 31) : ∃ V_up, V_up = V_m - (V_down - V_m) ∧ V_up = 27 := by
  sorry

end NUMINAMATH_GPT_rowing_upstream_speed_l77_7700


namespace NUMINAMATH_GPT_arithmetic_sequence_a7_l77_7777

variable {a : ℕ → ℚ}

def isArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_a7 (h_arith : isArithmeticSequence a) (h_a1 : a 1 = 2) (h_a3_a5 : a 3 + a 5 = 8) :
  a 7 = 6 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a7_l77_7777


namespace NUMINAMATH_GPT_real_solution_count_l77_7778

/-- Given \( \lfloor x \rfloor \) is the greatest integer less than or equal to \( x \),
prove that the number of real solutions to the equation \( 9x^2 - 36\lfloor x \rfloor + 20 = 0 \) is 2. --/
theorem real_solution_count (x : ℝ) (h : ⌊x⌋ = Int.floor x) :
  ∃ (S : Finset ℝ), S.card = 2 ∧ ∀ a ∈ S, 9 * a^2 - 36 * ⌊a⌋ + 20 = 0 :=
sorry

end NUMINAMATH_GPT_real_solution_count_l77_7778


namespace NUMINAMATH_GPT_common_solution_l77_7765

theorem common_solution (x : ℚ) : 
  (8 * x^2 + 7 * x - 1 = 0) ∧ (40 * x^2 + 89 * x - 9 = 0) → x = 1 / 8 :=
by { sorry }

end NUMINAMATH_GPT_common_solution_l77_7765


namespace NUMINAMATH_GPT_proof_prob_at_least_one_die_3_or_5_l77_7797

def probability_at_least_one_die_3_or_5 (total_outcomes : ℕ) (favorable_outcomes : ℕ) : ℚ :=
  favorable_outcomes / total_outcomes

theorem proof_prob_at_least_one_die_3_or_5 :
  let total_outcomes := 36
  let favorable_outcomes := 20
  probability_at_least_one_die_3_or_5 total_outcomes favorable_outcomes = 5 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_proof_prob_at_least_one_die_3_or_5_l77_7797


namespace NUMINAMATH_GPT_double_grandfather_pension_l77_7773

-- Define the total family income and individual contributions
def total_income (masha mother father grandfather : ℝ) : ℝ :=
  masha + mother + father + grandfather

-- Define the conditions provided in the problem
variables
  (masha mother father grandfather : ℝ)
  (cond1 : 2 * masha = total_income masha mother father grandfather * 1.05)
  (cond2 : 2 * mother = total_income masha mother father grandfather * 1.15)
  (cond3 : 2 * father = total_income masha mother father grandfather * 1.25)

-- Define the statement to be proved
theorem double_grandfather_pension :
  2 * grandfather = total_income masha mother father grandfather * 1.55 :=
by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_double_grandfather_pension_l77_7773


namespace NUMINAMATH_GPT_repeating_block_length_of_three_elevens_l77_7728

def smallest_repeating_block_length (x : ℚ) : ℕ :=
  sorry -- Definition to compute smallest repeating block length if needed

theorem repeating_block_length_of_three_elevens :
  smallest_repeating_block_length (3 / 11) = 2 :=
sorry

end NUMINAMATH_GPT_repeating_block_length_of_three_elevens_l77_7728


namespace NUMINAMATH_GPT_students_count_rental_cost_l77_7734

theorem students_count (k m : ℕ) (n : ℕ) 
  (h1 : n = 35 * k)
  (h2 : n = 55 * (m - 1) + 45) : 
  n = 175 := 
by {
  sorry
}

theorem rental_cost (x y : ℕ) 
  (total_buses : x + y = 4)
  (cost_limit : 35 * x + 55 * y ≤ 1500) : 
  320 * x + 400 * y = 1440 := 
by {
  sorry 
}

end NUMINAMATH_GPT_students_count_rental_cost_l77_7734


namespace NUMINAMATH_GPT_num_boys_l77_7726

variable (B G : ℕ)

def ratio_boys_girls (B G : ℕ) : Prop := B = 7 * G
def total_students (B G : ℕ) : Prop := B + G = 48

theorem num_boys (B G : ℕ) (h1 : ratio_boys_girls B G) (h2 : total_students B G) : 
  B = 42 :=
by
  sorry

end NUMINAMATH_GPT_num_boys_l77_7726


namespace NUMINAMATH_GPT_frank_maze_time_l77_7744

theorem frank_maze_time 
    (n mazes : ℕ)
    (avg_time_per_maze completed_time total_allowable_time remaining_maze_time extra_time_inside current_time : ℕ) 
    (h1 : mazes = 5)
    (h2 : avg_time_per_maze = 60)
    (h3 : completed_time = 200)
    (h4 : total_allowable_time = mazes * avg_time_per_maze)
    (h5 : total_allowable_time = 300)
    (h6 : remaining_maze_time = total_allowable_time - completed_time) 
    (h7 : extra_time_inside = 55)
    (h8 : current_time + extra_time_inside ≤ remaining_maze_time) :
  current_time = 45 :=
by
  sorry

end NUMINAMATH_GPT_frank_maze_time_l77_7744


namespace NUMINAMATH_GPT_jane_reading_speed_second_half_l77_7708

-- Definitions from the problem's conditions
def total_pages : ℕ := 500
def first_half_pages : ℕ := total_pages / 2
def first_half_speed : ℕ := 10
def total_days : ℕ := 75

-- The number of days spent reading the first half
def first_half_days : ℕ := first_half_pages / first_half_speed

-- The number of days spent reading the second half
def second_half_days : ℕ := total_days - first_half_days

-- The number of pages in the second half
def second_half_pages : ℕ := total_pages - first_half_pages

-- The actual theorem stating that Jane's reading speed for the second half was 5 pages per day
theorem jane_reading_speed_second_half :
  second_half_pages / second_half_days = 5 :=
by
  sorry

end NUMINAMATH_GPT_jane_reading_speed_second_half_l77_7708


namespace NUMINAMATH_GPT_irrational_number_l77_7759

noncomputable def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem irrational_number : 
  is_rational (Real.sqrt 4) ∧ 
  is_rational (22 / 7 : ℝ) ∧ 
  is_rational (1.0101 : ℝ) ∧ 
  ¬ is_rational (Real.pi / 3) 
  :=
sorry

end NUMINAMATH_GPT_irrational_number_l77_7759


namespace NUMINAMATH_GPT_optimal_solution_range_l77_7767

theorem optimal_solution_range (a : ℝ) (x y : ℝ) :
  (x + y - 4 ≥ 0) → (2 * x - y - 5 ≤ 0) → (x = 1) → (y = 3) →
  (-2 < a) ∧ (a < 1) :=
by
  intros h1 h2 hx hy
  sorry

end NUMINAMATH_GPT_optimal_solution_range_l77_7767


namespace NUMINAMATH_GPT_total_junk_mail_l77_7743

-- Definitions for conditions
def houses_per_block : Nat := 17
def pieces_per_house : Nat := 4
def blocks : Nat := 16

-- Theorem stating that the mailman gives out 1088 pieces of junk mail in total
theorem total_junk_mail : houses_per_block * pieces_per_house * blocks = 1088 := by
  sorry

end NUMINAMATH_GPT_total_junk_mail_l77_7743


namespace NUMINAMATH_GPT_certain_number_eq_neg17_l77_7752

theorem certain_number_eq_neg17 (x : Int) : 47 + x = 30 → x = -17 := by
  intro h
  have : x = 30 - 47 := by
    sorry  -- This is just to demonstrate the proof step. Actual manipulation should prove x = -17
  simp [this]

end NUMINAMATH_GPT_certain_number_eq_neg17_l77_7752


namespace NUMINAMATH_GPT_total_passengers_transportation_l77_7735

theorem total_passengers_transportation : 
  let passengers_one_way := 100
  let passengers_return := 60
  let first_trip_total := passengers_one_way + passengers_return
  let additional_trips := 3
  let additional_trips_total := additional_trips * first_trip_total
  let total_passengers := first_trip_total + additional_trips_total
  total_passengers = 640 := 
by
  sorry

end NUMINAMATH_GPT_total_passengers_transportation_l77_7735


namespace NUMINAMATH_GPT_find_a_20_l77_7749

-- Definitions
def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ (a₀ d : ℤ), ∀ n, a n = a₀ + n * d

def sum_first_n (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = n * (a 0 + a (n - 1)) / 2

-- Conditions and question
theorem find_a_20 (a S : ℕ → ℤ) (a₀ d : ℤ) :
  arithmetic_seq a ∧ sum_first_n a S ∧ 
  S 6 = 8 * (S 3) ∧ a 3 - a 5 = 8 → a 20 = -74 :=
by
  sorry

end NUMINAMATH_GPT_find_a_20_l77_7749


namespace NUMINAMATH_GPT_arithmetic_mean_first_n_positive_integers_l77_7747

theorem arithmetic_mean_first_n_positive_integers (n : ℕ) (Sn : ℕ) (h : Sn = n * (n + 1) / 2) : 
  (Sn / n) = (n + 1) / 2 := by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_arithmetic_mean_first_n_positive_integers_l77_7747


namespace NUMINAMATH_GPT_pyramid_base_side_length_l77_7741

theorem pyramid_base_side_length (A : ℝ) (h : ℝ) (s : ℝ)
  (hA : A = 200)
  (hh : h = 40)
  (hface : A = (1 / 2) * s * h) : 
  s = 10 :=
by
  sorry

end NUMINAMATH_GPT_pyramid_base_side_length_l77_7741


namespace NUMINAMATH_GPT_expand_and_simplify_l77_7736

theorem expand_and_simplify (x : ℝ) (hx : x ≠ 0) :
  (3 / 7) * (14 / x^3 + 15 * x - 6 * x^5) = (6 / x^3) + (45 * x / 7) - (18 * x^5 / 7) :=
by
  sorry

end NUMINAMATH_GPT_expand_and_simplify_l77_7736


namespace NUMINAMATH_GPT_problem_inequality_l77_7714

variable (a b : ℝ)

theorem problem_inequality (h1 : a < 0) (h2 : -1 < b) (h3 : b < 0) : ab > ab^2 ∧ ab^2 > a :=
by
  sorry

end NUMINAMATH_GPT_problem_inequality_l77_7714


namespace NUMINAMATH_GPT_area_of_isosceles_triangle_l77_7739

open Real

theorem area_of_isosceles_triangle 
  (PQ PR QR : ℝ) (PQ_eq_PR : PQ = PR) (PQ_val : PQ = 13) (QR_val : QR = 10) : 
  1 / 2 * QR * sqrt (PQ^2 - (QR / 2)^2) = 60 := 
by 
sorry

end NUMINAMATH_GPT_area_of_isosceles_triangle_l77_7739


namespace NUMINAMATH_GPT_shifted_parabola_sum_l77_7727

theorem shifted_parabola_sum :
  let f (x : ℝ) := 3 * x^2 - 2 * x + 5
  let g (x : ℝ) := 3 * (x - 3)^2 - 2 * (x - 3) + 5
  let a := 3
  let b := -20
  let c := 38
  a + b + c = 21 :=
by
  sorry

end NUMINAMATH_GPT_shifted_parabola_sum_l77_7727


namespace NUMINAMATH_GPT_rectangle_square_ratio_l77_7719

theorem rectangle_square_ratio (s x y : ℝ) (h1 : 0.1 * s ^ 2 = 0.25 * x * y) (h2 : y = s / 4) :
  x / y = 6 := 
sorry

end NUMINAMATH_GPT_rectangle_square_ratio_l77_7719


namespace NUMINAMATH_GPT_pure_alcohol_addition_l77_7782

variable (x : ℝ)

def initial_volume : ℝ := 6
def initial_concentration : ℝ := 0.25
def final_concentration : ℝ := 0.50

theorem pure_alcohol_addition :
  (1.5 + x) / (initial_volume + x) = final_concentration → x = 3 :=
by
  sorry

end NUMINAMATH_GPT_pure_alcohol_addition_l77_7782


namespace NUMINAMATH_GPT_comb_identity_l77_7755

theorem comb_identity (n : Nat) (h : 0 < n) (h_eq : Nat.choose n 2 = Nat.choose (n-1) 2 + Nat.choose (n-1) 3) : n = 5 := by
  sorry

end NUMINAMATH_GPT_comb_identity_l77_7755


namespace NUMINAMATH_GPT_ratio_of_dogs_to_cats_l77_7720

theorem ratio_of_dogs_to_cats (D C : ℕ) (hC : C = 40) (h : D + 20 = 2 * C) :
  D / Nat.gcd D C = 3 ∧ C / Nat.gcd D C = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_dogs_to_cats_l77_7720


namespace NUMINAMATH_GPT_unique_real_root_of_quadratic_l77_7780

theorem unique_real_root_of_quadratic (k : ℝ) :
  (∃ a : ℝ, ∀ b : ℝ, ((k^2 - 9) * b^2 - 2 * (k + 1) * b + 1 = 0 → b = a)) ↔ (k = 3 ∨ k = -3 ∨ k = -5) :=
by
  sorry

end NUMINAMATH_GPT_unique_real_root_of_quadratic_l77_7780


namespace NUMINAMATH_GPT_tony_water_intake_l77_7729

theorem tony_water_intake (yesterday water_two_days_ago : ℝ) 
    (h1 : yesterday = 48) (h2 : yesterday = 0.96 * water_two_days_ago) :
    water_two_days_ago = 50 :=
by
  sorry

end NUMINAMATH_GPT_tony_water_intake_l77_7729


namespace NUMINAMATH_GPT_probability_of_draw_l77_7704

-- Define the probabilities as constants
def prob_not_lose_xiao_ming : ℚ := 3 / 4
def prob_lose_xiao_dong : ℚ := 1 / 2

-- State the theorem we want to prove
theorem probability_of_draw :
  prob_not_lose_xiao_ming - prob_lose_xiao_dong = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_draw_l77_7704


namespace NUMINAMATH_GPT_gum_pieces_per_package_l77_7770

theorem gum_pieces_per_package (packages : ℕ) (extra : ℕ) (total : ℕ) (pieces_per_package : ℕ) :
    packages = 43 → extra = 8 → total = 997 → 43 * pieces_per_package + extra = total → pieces_per_package = 23 :=
by
  intros hpkg hextra htotal htotal_eq
  sorry

end NUMINAMATH_GPT_gum_pieces_per_package_l77_7770


namespace NUMINAMATH_GPT_team_not_losing_probability_l77_7793

theorem team_not_losing_probability
  (p_center_forward : ℝ) (p_winger : ℝ) (p_attacking_midfielder : ℝ)
  (rate_center_forward : ℝ) (rate_winger : ℝ) (rate_attacking_midfielder : ℝ)
  (h_center_forward : p_center_forward = 0.2) (h_winger : p_winger = 0.5) (h_attacking_midfielder : p_attacking_midfielder = 0.3)
  (h_rate_center_forward : rate_center_forward = 0.4) (h_rate_winger : rate_winger = 0.2) (h_rate_attacking_midfielder : rate_attacking_midfielder = 0.2) :
  (p_center_forward * (1 - rate_center_forward) + p_winger * (1 - rate_winger) + p_attacking_midfielder * (1 - rate_attacking_midfielder)) = 0.76 :=
by
  sorry

end NUMINAMATH_GPT_team_not_losing_probability_l77_7793


namespace NUMINAMATH_GPT_new_cases_first_week_l77_7724

theorem new_cases_first_week
  (X : ℕ)
  (second_week_cases : X / 2 = X / 2)
  (third_week_cases : X / 2 + 2000 = (X / 2) + 2000)
  (total_cases : X + X / 2 + (X / 2 + 2000) = 9500) :
  X = 3750 := 
by sorry

end NUMINAMATH_GPT_new_cases_first_week_l77_7724


namespace NUMINAMATH_GPT_fraction_boxes_loaded_by_day_crew_l77_7796

variables {D W_d : ℝ}

theorem fraction_boxes_loaded_by_day_crew
  (h1 : ∀ (D W_d: ℝ), D > 0 → W_d > 0 → ∃ (D' W_n : ℝ), (D' = 0.5 * D) ∧ (W_n = 0.8 * W_d))
  (h2 : ∃ (D W_d : ℝ), ∀ (D' W_n : ℝ), (D' = 0.5 * D) → (W_n = 0.8 * W_d) → 
        (D * W_d / (D * W_d + D' * W_n)) = (5 / 7)) :
  (∃ (D W_d : ℝ), D > 0 → W_d > 0 → (D * W_d)/(D * W_d + 0.5 * D * 0.8 * W_d) = (5/7)) := 
  sorry 

end NUMINAMATH_GPT_fraction_boxes_loaded_by_day_crew_l77_7796


namespace NUMINAMATH_GPT_value_of_ratios_l77_7791

variable (x y z : ℝ)

-- Conditions
def geometric_sequence : Prop :=
  4 * y / (3 * x) = 5 * z / (4 * y)

def arithmetic_sequence : Prop :=
  2 / y = 1 / x + 1 / z

-- Theorem/Proof Statement
theorem value_of_ratios (h1 : geometric_sequence x y z) (h2 : arithmetic_sequence x y z) :
  (x / z) + (z / x) = 34 / 15 :=
by
  sorry

end NUMINAMATH_GPT_value_of_ratios_l77_7791


namespace NUMINAMATH_GPT_B_investment_amount_l77_7787

-- Define given conditions in Lean 4

def A_investment := 400
def total_months := 12
def B_investment_months := 6
def total_profit := 100
def A_share := 80
def B_share := total_profit - A_share

-- The problem statement in Lean 4 that needs to be proven:
theorem B_investment_amount (A_investment B_investment_months total_profit A_share B_share: ℕ)
  (hA_investment : A_investment = 400)
  (htotal_months : total_months = 12)
  (hB_investment_months : B_investment_months = 6)
  (htotal_profit : total_profit = 100)
  (hA_share : A_share = 80)
  (hB_share : B_share = total_profit - A_share) 
  : (∃ (B: ℕ), 
       (5 * (A_investment * total_months) = 4 * (400 * total_months + B * B_investment_months)) 
       ∧ B = 200) :=
sorry

end NUMINAMATH_GPT_B_investment_amount_l77_7787


namespace NUMINAMATH_GPT_exist_triangle_l77_7781

-- Definitions of points and properties required in the conditions
structure Point :=
(x : ℝ) (y : ℝ)

def orthocenter (M : Point) := M 
def centroid (S : Point) := S 
def vertex (C : Point) := C 

-- The problem statement that needs to be proven
theorem exist_triangle (M S C : Point) 
    (h_orthocenter : orthocenter M = M)
    (h_centroid : centroid S = S)
    (h_vertex : vertex C = C) : 
    ∃ (A B : Point), 
        -- A, B, and C form a triangle ABC
        -- S is the centroid of this triangle
        -- M is the orthocenter of this triangle
        -- C is one of the vertices
        true := 
sorry

end NUMINAMATH_GPT_exist_triangle_l77_7781


namespace NUMINAMATH_GPT_opposite_and_reciprocal_numbers_l77_7711

theorem opposite_and_reciprocal_numbers (a b c d : ℝ)
  (h1 : a + b = 0)
  (h2 : c * d = 1) :
  2019 * a + (7 / (c * d)) + 2019 * b = 7 :=
sorry

end NUMINAMATH_GPT_opposite_and_reciprocal_numbers_l77_7711


namespace NUMINAMATH_GPT_sale_in_fifth_month_l77_7712

theorem sale_in_fifth_month
  (s1 s2 s3 s4 s6 : ℕ)
  (avg : ℕ)
  (h1 : s1 = 5435)
  (h2 : s2 = 5927)
  (h3 : s3 = 5855)
  (h4 : s4 = 6230)
  (h6 : s6 = 3991)
  (hav : avg = 5500) :
  ∃ s5 : ℕ, s1 + s2 + s3 + s4 + s5 + s6 = avg * 6 ∧ s5 = 5562 := 
by
  sorry

end NUMINAMATH_GPT_sale_in_fifth_month_l77_7712


namespace NUMINAMATH_GPT_determine_digit_phi_l77_7706

theorem determine_digit_phi (Φ : ℕ) (h1 : Φ > 0) (h2 : Φ < 10) (h3 : 504 / Φ = 40 + 3 * Φ) : Φ = 8 :=
by
  sorry

end NUMINAMATH_GPT_determine_digit_phi_l77_7706


namespace NUMINAMATH_GPT_find_geometric_sequence_l77_7775

def geometric_sequence (b1 b2 b3 b4 : ℤ) :=
  ∃ q : ℤ, b2 = b1 * q ∧ b3 = b1 * q^2 ∧ b4 = b1 * q^3

theorem find_geometric_sequence :
  ∃ b1 b2 b3 b4 : ℤ, 
    geometric_sequence b1 b2 b3 b4 ∧
    (b1 + b4 = -49) ∧
    (b2 + b3 = 14) ∧ 
    ((b1, b2, b3, b4) = (7, -14, 28, -56) ∨ (b1, b2, b3, b4) = (-56, 28, -14, 7)) :=
by
  sorry

end NUMINAMATH_GPT_find_geometric_sequence_l77_7775


namespace NUMINAMATH_GPT_loss_percent_l77_7757

theorem loss_percent (cost_price selling_price loss_percent : ℝ) 
  (h_cost_price : cost_price = 600)
  (h_selling_price : selling_price = 550)
  (h_loss_percent : loss_percent = 8.33) : 
  (loss_percent = ((cost_price - selling_price) / cost_price) * 100) := 
by
  rw [h_cost_price, h_selling_price]
  sorry

end NUMINAMATH_GPT_loss_percent_l77_7757


namespace NUMINAMATH_GPT_relay_race_total_time_l77_7746

noncomputable def mary_time (susan_time : ℕ) : ℕ := 2 * susan_time
noncomputable def susan_time (jen_time : ℕ) : ℕ := jen_time + 10
def jen_time : ℕ := 30
noncomputable def tiffany_time (mary_time : ℕ) : ℕ := mary_time - 7

theorem relay_race_total_time :
  let mary_time := mary_time (susan_time jen_time)
  let susan_time := susan_time jen_time
  let tiffany_time := tiffany_time mary_time
  mary_time + susan_time + jen_time + tiffany_time = 223 := by
  sorry

end NUMINAMATH_GPT_relay_race_total_time_l77_7746


namespace NUMINAMATH_GPT_linear_function_common_quadrants_l77_7705

theorem linear_function_common_quadrants {k b : ℝ} (h : k * b < 0) :
  (exists (q1 q2 : ℕ), q1 = 1 ∧ q2 = 4) := 
sorry

end NUMINAMATH_GPT_linear_function_common_quadrants_l77_7705


namespace NUMINAMATH_GPT_proof_x_squared_minus_y_squared_l77_7733

theorem proof_x_squared_minus_y_squared (x y : ℚ) (h1 : x + y = 9 / 14) (h2 : x - y = 3 / 14) :
  x^2 - y^2 = 27 / 196 := by
  sorry

end NUMINAMATH_GPT_proof_x_squared_minus_y_squared_l77_7733


namespace NUMINAMATH_GPT_trivia_team_students_per_group_l77_7768

theorem trivia_team_students_per_group (total_students : ℕ) (not_picked : ℕ) (num_groups : ℕ) 
  (h1 : total_students = 58) (h2 : not_picked = 10) (h3 : num_groups = 8) :
  (total_students - not_picked) / num_groups = 6 :=
by
  sorry

end NUMINAMATH_GPT_trivia_team_students_per_group_l77_7768


namespace NUMINAMATH_GPT_tan_double_angle_l77_7761

theorem tan_double_angle (α : ℝ) (h : Real.tan (π - α) = 2) : Real.tan (2 * α) = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_double_angle_l77_7761


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l77_7753

theorem arithmetic_sequence_common_difference (a : ℕ → ℤ) (d : ℤ) :
  (a 5 = 8) → (a 1 + a 2 + a 3 = 6) → (∀ n : ℕ, a (n + 1) = a 1 + n * d) → d = 2 :=
by
  intros ha5 hsum harr
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l77_7753


namespace NUMINAMATH_GPT_mean_of_all_students_l77_7701

variable (M A m a : ℕ)
variable (M_val : M = 84)
variable (A_val : A = 70)
variable (ratio : m = 3 * a / 4)

theorem mean_of_all_students (M A m a : ℕ) (M_val : M = 84) (A_val : A = 70) (ratio : m = 3 * a / 4) :
    (63 * a + 70 * a) / (7 * a / 4) = 76 := by
  sorry

end NUMINAMATH_GPT_mean_of_all_students_l77_7701


namespace NUMINAMATH_GPT_gcd_lcm_product_360_l77_7790

theorem gcd_lcm_product_360 (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 360) :
    {d : ℕ | d = Nat.gcd a b } =
    {1, 2, 4, 8, 3, 6, 12, 24} := 
by
  sorry

end NUMINAMATH_GPT_gcd_lcm_product_360_l77_7790


namespace NUMINAMATH_GPT_trig_identity_l77_7751

theorem trig_identity (α : ℝ) 
  (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) 
  : Real.cos (5 / 6 * π + α) + (Real.cos (4 * π / 3 + α))^2 = (2 - Real.sqrt 3) / 3 := 
sorry

end NUMINAMATH_GPT_trig_identity_l77_7751


namespace NUMINAMATH_GPT_tank_capacity_l77_7737

theorem tank_capacity (w c : ℝ) (h1 : w / c = 1 / 6) (h2 : (w + 5) / c = 1 / 3) : c = 30 :=
by
  sorry

end NUMINAMATH_GPT_tank_capacity_l77_7737


namespace NUMINAMATH_GPT_Batman_game_cost_l77_7725

theorem Batman_game_cost (football_cost strategy_cost total_spent batman_cost : ℝ)
  (h₁ : football_cost = 14.02)
  (h₂ : strategy_cost = 9.46)
  (h₃ : total_spent = 35.52)
  (h₄ : total_spent = football_cost + strategy_cost + batman_cost) :
  batman_cost = 12.04 := by
  sorry

end NUMINAMATH_GPT_Batman_game_cost_l77_7725


namespace NUMINAMATH_GPT_find_matrix_M_l77_7722

-- Define the given matrix with real entries
def matrix_M : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 2], ![-1, 0]]

-- Define the function for matrix operations
def M_calc (M : Matrix (Fin 2) (Fin 2) ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  (M * M * M) - (M * M) + (2 • M)

-- Define the target matrix
def target_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 3], ![-2, 0]]

-- Problem statement: The matrix M should satisfy the given matrix equation
theorem find_matrix_M (M : Matrix (Fin 2) (Fin 2) ℝ) :
  M_calc M = target_matrix ↔ M = matrix_M :=
sorry

end NUMINAMATH_GPT_find_matrix_M_l77_7722


namespace NUMINAMATH_GPT_Liam_cycling_speed_l77_7718

theorem Liam_cycling_speed :
  ∀ (Eugene_speed Claire_speed Liam_speed : ℝ),
    Eugene_speed = 6 →
    Claire_speed = (3/4) * Eugene_speed →
    Liam_speed = (4/3) * Claire_speed →
    Liam_speed = 6 :=
by
  intros
  sorry

end NUMINAMATH_GPT_Liam_cycling_speed_l77_7718


namespace NUMINAMATH_GPT_find_prime_pair_l77_7748

noncomputable def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def has_integer_root (p q : ℕ) : Prop :=
  ∃ x : ℤ, x^4 + p * x^3 - q = 0

theorem find_prime_pair :
  ∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ has_integer_root p q ∧ p = 2 ∧ q = 3 := by
  sorry

end NUMINAMATH_GPT_find_prime_pair_l77_7748


namespace NUMINAMATH_GPT_A_fraction_simplification_l77_7716

noncomputable def A : ℚ := 
  ((3/8) * (13/5)) / ((5/2) * (6/5)) +
  ((5/8) * (8/5)) / (3 * (6/5) * (25/6)) +
  (20/3) * (3/25) +
  28 +
  (1 / 9) / 7 +
  (1/5) / (9 * 22)

theorem A_fraction_simplification :
  let num := 1901
  let denom := 3360
  (A = num / denom) :=
sorry

end NUMINAMATH_GPT_A_fraction_simplification_l77_7716


namespace NUMINAMATH_GPT_solve_for_a_l77_7783

theorem solve_for_a (x a : ℝ) (h : x = 3) (eqn : 2 * (x - 1) - a = 0) : a = 4 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_a_l77_7783


namespace NUMINAMATH_GPT_find_a_l77_7763

-- Define the lines as given
def line1 (x y : ℝ) := 2 * x + y - 5 = 0
def line2 (x y : ℝ) := x - y - 1 = 0
def line3 (a x y : ℝ) := a * x + y - 3 = 0

-- Define the condition that they intersect at a single point
def lines_intersect_at_point (x y a : ℝ) := line1 x y ∧ line2 x y ∧ line3 a x y

-- To prove: If lines intersect at a certain point, then a = 1
theorem find_a (a : ℝ) : (∃ x y, lines_intersect_at_point x y a) → a = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l77_7763


namespace NUMINAMATH_GPT_length_of_second_edge_l77_7799

-- Define the edge lengths and volume
def edge1 : ℕ := 6
def edge3 : ℕ := 6
def volume : ℕ := 180

-- The theorem to state the length of the second edge
theorem length_of_second_edge (edge2 : ℕ) (h : edge1 * edge2 * edge3 = volume) :
  edge2 = 5 :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_length_of_second_edge_l77_7799


namespace NUMINAMATH_GPT_initial_speed_100kmph_l77_7756

theorem initial_speed_100kmph (v x : ℝ) (h1 : 0 < v) (h2 : 100 - x = v / 2) 
  (h3 : (80 - x) / (v - 10) - 20 / (v - 20) = 1 / 12) : v = 100 :=
by 
  sorry

end NUMINAMATH_GPT_initial_speed_100kmph_l77_7756


namespace NUMINAMATH_GPT_missing_angle_in_convex_polygon_l77_7754

theorem missing_angle_in_convex_polygon (n : ℕ) (x : ℝ) 
  (h1 : n ≥ 5) 
  (h2 : 180 * (n - 2) - 3 * x = 3330) : 
  x = 54 := 
by 
  sorry

end NUMINAMATH_GPT_missing_angle_in_convex_polygon_l77_7754


namespace NUMINAMATH_GPT_mike_initial_games_l77_7707

theorem mike_initial_games (v w: ℕ)
  (h_non_working : v - w = 8)
  (h_earnings : 7 * w = 56)
  : v = 16 :=
by
  sorry

end NUMINAMATH_GPT_mike_initial_games_l77_7707


namespace NUMINAMATH_GPT_sin_cos_value_l77_7789

theorem sin_cos_value (x : ℝ) (h : Real.cos x - 3 * Real.sin x = 2) :
  (3 * Real.sin x + Real.cos x = 0) ∨ (3 * Real.sin x + Real.cos x = -4) :=
sorry

end NUMINAMATH_GPT_sin_cos_value_l77_7789


namespace NUMINAMATH_GPT_gcd_97_pow_10_plus_1_and_97_pow_10_plus_97_pow_3_plus_1_l77_7721

theorem gcd_97_pow_10_plus_1_and_97_pow_10_plus_97_pow_3_plus_1 :
  Int.gcd (97 ^ 10 + 1) (97 ^ 10 + 97 ^ 3 + 1) = 1 := sorry

end NUMINAMATH_GPT_gcd_97_pow_10_plus_1_and_97_pow_10_plus_97_pow_3_plus_1_l77_7721


namespace NUMINAMATH_GPT_sum_F_G_H_l77_7784

theorem sum_F_G_H : 
  ∀ (F G H : ℕ), 
    (F < 10 ∧ G < 10 ∧ H < 10) ∧ 
    ∃ k : ℤ, 
      (F - 8 + 6 - 1 + G - 2 - H - 11 * k = 0) → 
        F + G + H = 23 :=
by sorry

end NUMINAMATH_GPT_sum_F_G_H_l77_7784
