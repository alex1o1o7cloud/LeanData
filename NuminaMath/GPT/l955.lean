import Mathlib

namespace solve_inequality_l955_95575

def f (a x : ℝ) : ℝ := a * x * (x + 1) + 1

theorem solve_inequality (a x : ℝ) (h : f a x < 0) : x < (1 / a) ∨ (x > 1 ∧ a ≠ 0) := by
  sorry

end solve_inequality_l955_95575


namespace find_f_l955_95583

noncomputable def func_satisfies_eq (f : ℝ → ℝ) :=
  ∀ x y : ℝ, f (x ^ 2 - y ^ 2) = x * f x - y * f y

theorem find_f (f : ℝ → ℝ) (h : func_satisfies_eq f) : ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end find_f_l955_95583


namespace integer_modulo_problem_l955_95525

theorem integer_modulo_problem : ∃ n : ℤ, 0 ≤ n ∧ n < 23 ∧ (-250 % 23 = n) := 
  sorry

end integer_modulo_problem_l955_95525


namespace triangle_inequality_inequality_l955_95511

variable {a b c : ℝ}
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)
variable (triangle_ineq : a + b > c)

theorem triangle_inequality_inequality (ha : a > 0) (hb : b > 0) (hc : c > 0) (triangle_ineq : a + b > c) :
  a^3 + b^3 + 3 * a * b * c > c^3 :=
sorry

end triangle_inequality_inequality_l955_95511


namespace remainder_div_x_minus_2_l955_95587

noncomputable def q (x : ℝ) (A B C : ℝ) : ℝ := A * x^6 + B * x^4 + C * x^2 + 10

theorem remainder_div_x_minus_2 (A B C : ℝ) (h : q 2 A B C = 20) : q (-2) A B C = 20 :=
by sorry

end remainder_div_x_minus_2_l955_95587


namespace arithmetic_sequence_l955_95560

theorem arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) (h1 : a 2 = 3) (h2 : a (n - 1) = 17) (h3 : n ≥ 2) (h4 : (n * (3 + 17)) / 2 = 100) : n = 10 :=
sorry

end arithmetic_sequence_l955_95560


namespace positive_difference_balances_l955_95519

noncomputable def cedric_balance (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r) ^ t

noncomputable def daniel_balance (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r) ^ t

theorem positive_difference_balances :
  let P : ℝ := 15000
  let r_cedric : ℝ := 0.06
  let r_daniel : ℝ := 0.08
  let t : ℕ := 15
  let A_cedric := cedric_balance P r_cedric t
  let A_daniel := daniel_balance P r_daniel t
  (A_daniel - A_cedric) = 11632.65 :=
by
  sorry

end positive_difference_balances_l955_95519


namespace count_multiples_of_5_not_10_or_15_l955_95535

theorem count_multiples_of_5_not_10_or_15 : 
  ∃ n : ℕ, n = 33 ∧ (∀ x : ℕ, x < 500 ∧ (x % 5 = 0) ∧ (x % 10 ≠ 0) ∧ (x % 15 ≠ 0) → x < 500 ∧ (x % 5 = 0) ∧ (x % 10 ≠ 0) ∧ (x % 15 ≠ 0)) :=
by
  sorry

end count_multiples_of_5_not_10_or_15_l955_95535


namespace ratio_of_product_of_composites_l955_95503

theorem ratio_of_product_of_composites :
  let A := [4, 6, 8, 9, 10, 12]
  let B := [14, 15, 16, 18, 20, 21]
  (A.foldl (λ x y => x * y) 1) / (B.foldl (λ x y => x * y) 1) = 1 / 49 :=
by
  -- Proof will be filled here
  sorry

end ratio_of_product_of_composites_l955_95503


namespace friends_playing_video_game_l955_95585

def total_lives : ℕ := 64
def lives_per_player : ℕ := 8

theorem friends_playing_video_game (num_friends : ℕ) :
  num_friends = total_lives / lives_per_player :=
sorry

end friends_playing_video_game_l955_95585


namespace smallest_integer_for_perfect_square_l955_95510

-- Given condition: y = 2^3 * 3^2 * 4^6 * 5^5 * 7^8 * 8^3 * 9^10 * 11^11
def y : ℕ := 2^3 * 3^2 * 4^6 * 5^5 * 7^8 * 8^3 * 9^10 * 11^11

-- The statement to prove
theorem smallest_integer_for_perfect_square (y : ℕ) : ∃ n : ℕ, n = 110 ∧ ∃ m : ℕ, (y * n) = m^2 := 
by {
  sorry
}

end smallest_integer_for_perfect_square_l955_95510


namespace problem_series_sum_l955_95565

noncomputable def series_sum : ℝ := ∑' n : ℕ, (4 * n + 3) / ((4 * n + 1)^2 * (4 * n + 5)^2)

theorem problem_series_sum :
  series_sum = 1 / 200 :=
sorry

end problem_series_sum_l955_95565


namespace second_die_sides_l955_95551

theorem second_die_sides (p : ℚ) (n : ℕ) (h1 : p = 0.023809523809523808) (h2 : n ≠ 0) :
  let first_die_sides := 6
  let probability := (1 : ℚ) / first_die_sides * (1 : ℚ) / n
  probability = p → n = 7 :=
by
  intro h
  sorry

end second_die_sides_l955_95551


namespace apples_given_to_father_l955_95593

theorem apples_given_to_father
  (total_apples : ℤ) 
  (people_sharing : ℤ) 
  (apples_per_person : ℤ)
  (jack_and_friends : ℤ) :
  total_apples = 55 →
  people_sharing = 5 →
  apples_per_person = 9 →
  jack_and_friends = 4 →
  (total_apples - people_sharing * apples_per_person) = 10 :=
by 
  intros h1 h2 h3 h4
  sorry

end apples_given_to_father_l955_95593


namespace work_problem_l955_95596

theorem work_problem (W : ℕ) (T_AB T_A T_B together_worked alone_worked remaining_work : ℕ)
  (h1 : T_AB = 30)
  (h2 : T_A = 60)
  (h3 : together_worked = 20)
  (h4 : T_B = 30)
  (h5 : remaining_work = W / 3)
  (h6 : alone_worked = 20)
  : alone_worked = 20 :=
by
  /- Proof is not required -/
  sorry

end work_problem_l955_95596


namespace unique_seating_arrangements_l955_95594

/--
There are five couples including Charlie and his wife. The five men sit on the 
inner circle and each man's wife sits directly opposite him on the outer circle.
Prove that the number of unique seating arrangements where each man has another 
man seated directly to his right on the inner circle, counting all seat 
rotations as the same but not considering inner to outer flips as different, is 30.
-/
theorem unique_seating_arrangements : 
  ∃ (n : ℕ), n = 30 := 
sorry

end unique_seating_arrangements_l955_95594


namespace not_every_constant_is_geometric_l955_95567

def is_constant_sequence (s : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, s n = s m

def is_geometric_sequence (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, s (n + 1) = r * s n

theorem not_every_constant_is_geometric :
  (¬ ∀ s : ℕ → ℝ, is_constant_sequence s → is_geometric_sequence s) ↔
  ∃ s : ℕ → ℝ, is_constant_sequence s ∧ ¬ is_geometric_sequence s := 
by
  sorry

end not_every_constant_is_geometric_l955_95567


namespace larger_box_cost_l955_95572

-- Definitions based on the conditions

def ounces_large : ℕ := 30
def ounces_small : ℕ := 20
def cost_small : ℝ := 3.40
def price_per_ounce_better_value : ℝ := 0.16

-- The statement to prove
theorem larger_box_cost :
  30 * price_per_ounce_better_value = 4.80 :=
by sorry

end larger_box_cost_l955_95572


namespace duration_of_period_l955_95570

noncomputable def birth_rate : ℕ := 7
noncomputable def death_rate : ℕ := 3
noncomputable def net_increase : ℕ := 172800

theorem duration_of_period : (net_increase / ((birth_rate - death_rate) / 2)) / 3600 = 12 := by
  sorry

end duration_of_period_l955_95570


namespace quadratic_two_distinct_real_roots_l955_95537

def quadratic_function_has_two_distinct_real_roots (k : ℝ) : Prop :=
  let a := k
  let b := -4
  let c := -2
  b * b - 4 * a * c > 0 ∧ a ≠ 0

theorem quadratic_two_distinct_real_roots (k : ℝ) :
  quadratic_function_has_two_distinct_real_roots k ↔ (k > -2 ∧ k ≠ 0) :=
by
  sorry

end quadratic_two_distinct_real_roots_l955_95537


namespace express_2011_with_digit_1_l955_95513

theorem express_2011_with_digit_1 :
  ∃ (a b c d e: ℕ), 2011 = a * b - c * d + e - f + g ∧
  (a = 1111 ∧ b = 1111) ∧ (c = 111 ∧ d = 11111) ∧ (e = 1111) ∧ (f = 111) ∧ (g = 11) ∧
  (a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ e ≠ f ∧ f ≠ g) :=
sorry

end express_2011_with_digit_1_l955_95513


namespace sum_of_common_ratios_l955_95550

variable {k a_2 a_3 b_2 b_3 p r : ℝ}
variable (hp : a_2 = k * p) (ha3 : a_3 = k * p^2)
variable (hr : b_2 = k * r) (hb3 : b_3 = k * r^2)
variable (hcond : a_3 - b_3 = 5 * (a_2 - b_2))

theorem sum_of_common_ratios (h_nonconst : k ≠ 0) (p_ne_r : p ≠ r) : p + r = 5 :=
by
  sorry

end sum_of_common_ratios_l955_95550


namespace four_at_three_equals_thirty_l955_95540

def custom_operation (a b : ℕ) : ℕ :=
  3 * a^2 - 2 * b^2

theorem four_at_three_equals_thirty : custom_operation 4 3 = 30 :=
by
  sorry

end four_at_three_equals_thirty_l955_95540


namespace min_distance_from_origin_l955_95581

-- Define the condition of the problem
def condition (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x + 6 * y + 4 = 0

-- Statement of the problem in Lean 4
theorem min_distance_from_origin (x y : ℝ) (h : condition x y) : 
  ∃ m : ℝ, m = Real.sqrt (x^2 + y^2) ∧ m = Real.sqrt 13 - 3 := 
sorry

end min_distance_from_origin_l955_95581


namespace part1_part2_l955_95586

def p (a : ℝ) : Prop := a^2 - 5*a - 6 > 0
def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a * x + 1 = 0 → x < 0

theorem part1 (a : ℝ) (hp : p a) : a ∈ Set.Iio (-1) ∪ Set.Ioi 6 :=
sorry

theorem part2 (a : ℝ) (h_or : p a ∨ q a) (h_and : ¬ (p a ∧ q a)) : a ∈ Set.Iio (-1) ∪ Set.Ioc 2 6 :=
sorry

end part1_part2_l955_95586


namespace therapy_hours_l955_95559

theorem therapy_hours (x n : ℕ) : 
  (x + 30) + 2 * x = 252 → 
  104 + (n - 1) * x = 400 → 
  x = 74 → 
  n = 5 := 
by
  sorry

end therapy_hours_l955_95559


namespace initial_amounts_l955_95530

theorem initial_amounts (x y z : ℕ) (h1 : x + y + z = 24)
  (h2 : z = 24 - x - y)
  (h3 : x - (y + z) = 8)
  (h4 : y - (x + z) = 12) :
  x = 13 ∧ y = 7 ∧ z = 4 :=
by
  sorry

end initial_amounts_l955_95530


namespace smallest_right_triangle_area_l955_95558

theorem smallest_right_triangle_area
  (a b : ℕ)
  (h₁ : a = 6)
  (h₂ : b = 8)
  (h₃ : ∃ c : ℕ, a * a + b * b = c * c) :
  (∃ A : ℕ, A = (1 / 2) * a * b) :=
by
  use 24
  sorry

end smallest_right_triangle_area_l955_95558


namespace count_integers_with_block_178_l955_95518

theorem count_integers_with_block_178 (a b : ℕ) : 10000 ≤ a ∧ a < 100000 → 10000 ≤ b ∧ b < 100000 → a = b → b - a = 99999 → ∃ n, n = 280 ∧ (n = a + b) := sorry

end count_integers_with_block_178_l955_95518


namespace tangent_line_at_1_l955_95512

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.log x
def tangent_line_eq : ℝ × ℝ → ℝ := fun ⟨x, y⟩ => x - y - 1

theorem tangent_line_at_1 : tangent_line_eq (1, f 1) = 0 := by
  -- Proof would go here
  sorry

end tangent_line_at_1_l955_95512


namespace smallest_fraction_l955_95599

theorem smallest_fraction (a x y : ℕ) (ha : a > 100) (hx : x > 100) (hy : y > 100) (eqn : y^2 - 1 = a^2 * (x^2 - 1)) :
  2 ≤ a / x :=
sorry

end smallest_fraction_l955_95599


namespace expected_balls_in_original_pos_after_two_transpositions_l955_95507

theorem expected_balls_in_original_pos_after_two_transpositions :
  ∃ (n : ℚ), n = 3.2 := 
sorry

end expected_balls_in_original_pos_after_two_transpositions_l955_95507


namespace find_n_mod_60_l955_95516

theorem find_n_mod_60 {x y : ℤ} (hx : x ≡ 45 [ZMOD 60]) (hy : y ≡ 98 [ZMOD 60]) :
  ∃ n, 150 ≤ n ∧ n ≤ 210 ∧ (x - y ≡ n [ZMOD 60]) ∧ n = 187 := by
  sorry

end find_n_mod_60_l955_95516


namespace textopolis_word_count_l955_95509

theorem textopolis_word_count :
  let alphabet_size := 26
  let total_one_letter := 2 -- only "A" and "B"
  let total_two_letter := alphabet_size^2
  let excl_two_letter := (alphabet_size - 2)^2
  let total_three_letter := alphabet_size^3
  let excl_three_letter := (alphabet_size - 2)^3
  let total_four_letter := alphabet_size^4
  let excl_four_letter := (alphabet_size - 2)^4
  let valid_two_letter := total_two_letter - excl_two_letter
  let valid_three_letter := total_three_letter - excl_three_letter
  let valid_four_letter := total_four_letter - excl_four_letter
  2 + valid_two_letter + valid_three_letter + valid_four_letter = 129054 := by
  -- To be proved
  sorry

end textopolis_word_count_l955_95509


namespace overall_average_marks_l955_95576

theorem overall_average_marks 
  (n1 : ℕ) (m1 : ℕ) 
  (n2 : ℕ) (m2 : ℕ) 
  (n3 : ℕ) (m3 : ℕ) 
  (n4 : ℕ) (m4 : ℕ) 
  (h1 : n1 = 70) (h2 : m1 = 50) 
  (h3 : n2 = 35) (h4 : m2 = 60)
  (h5 : n3 = 45) (h6 : m3 = 55)
  (h7 : n4 = 42) (h8 : m4 = 45) :
  (n1 * m1 + n2 * m2 + n3 * m3 + n4 * m4) / (n1 + n2 + n3 + n4) = 9965 / 192 :=
by
  sorry

end overall_average_marks_l955_95576


namespace solve_equation_l955_95536

theorem solve_equation (x : ℝ) : (3 * x - 2 * (10 - x) = 5) → x = 5 :=
by {
  sorry
}

end solve_equation_l955_95536


namespace constant_sums_l955_95534

theorem constant_sums (n : ℕ) 
  (x y z : ℝ) 
  (h₁ : x + y + z = 0) 
  (h₂ : x * y * z = 1) 
  : (x^n + y^n + z^n = 0 ∨ x^n + y^n + z^n = 3) ↔ (n = 1 ∨ n = 3) :=
by sorry

end constant_sums_l955_95534


namespace solve_basketball_court_dimensions_l955_95529

theorem solve_basketball_court_dimensions 
  (A B C D E F : ℕ) 
  (h1 : A - B = C) 
  (h2 : D = 2 * (A + B)) 
  (h3 : E = A * B) 
  (h4 : F = 3) : 
  A = 28 ∧ B = 15 ∧ C = 13 ∧ D = 86 ∧ E = 420 ∧ F = 3 := 
by 
  sorry

end solve_basketball_court_dimensions_l955_95529


namespace percentage_difference_l955_95538

variables (G P R : ℝ)

-- Conditions
def condition1 : Prop := P = 0.9 * G
def condition2 : Prop := R = 3.0000000000000006 * G

-- Theorem to prove
theorem percentage_difference (h1 : condition1 P G) (h2 : condition2 R G) : 
  (R - P) / R * 100 = 70 :=
sorry

end percentage_difference_l955_95538


namespace evaluate_expression_l955_95557

theorem evaluate_expression
  (p q r s : ℚ)
  (h1 : p / q = 4 / 5)
  (h2 : r / s = 3 / 7) :
  (18 / 7) + ((2 * q - p) / (2 * q + p)) - ((3 * s + r) / (3 * s - r)) = 5 / 3 := by
  sorry

end evaluate_expression_l955_95557


namespace total_new_cans_l955_95521

-- Define the condition
def initial_cans : ℕ := 256
def first_term : ℕ := 64
def ratio : ℚ := 1 / 4
def terms : ℕ := 4

-- Define the sum of the geometric series
noncomputable def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * ((1 - r ^ n) / (1 - r))

-- Problem statement in Lean 4
theorem total_new_cans : geometric_series_sum first_term ratio terms = 85 := by
  sorry

end total_new_cans_l955_95521


namespace divisibility_of_n_squared_plus_n_plus_two_l955_95592

-- Definition: n is a natural number.
def n (n : ℕ) : Prop := True

-- Theorem: For any natural number n, n^2 + n + 2 is always divisible by 2, but not necessarily divisible by 5.
theorem divisibility_of_n_squared_plus_n_plus_two (n : ℕ) : 
  (∃ k : ℕ, n^2 + n + 2 = 2 * k) ∧ (¬ ∃ m : ℕ, n^2 + n + 2 = 5 * m) :=
by
  sorry

end divisibility_of_n_squared_plus_n_plus_two_l955_95592


namespace bernoulli_inequality_l955_95591

theorem bernoulli_inequality (n : ℕ) (h : 1 ≤ n) (x : ℝ) (h1 : x > -1) : (1 + x) ^ n ≥ 1 + n * x := 
sorry

end bernoulli_inequality_l955_95591


namespace melanie_turnips_l955_95563

theorem melanie_turnips (b : ℕ) (d : ℕ) (h_b : b = 113) (h_d : d = 26) : b + d = 139 :=
by
  sorry

end melanie_turnips_l955_95563


namespace martha_meeting_distance_l955_95561

theorem martha_meeting_distance (t : ℝ) (d : ℝ)
  (h1 : 0 < t)
  (h2 : d = 45 * (t + 0.75))
  (h3 : d - 45 = 55 * (t - 1)) :
  d = 230.625 := 
  sorry

end martha_meeting_distance_l955_95561


namespace Uncle_Bradley_bills_l955_95523

theorem Uncle_Bradley_bills :
  let total_money := 1000
  let fifty_bills_portion := 3 / 10
  let fifty_bill_value := 50
  let hundred_bill_value := 100
  -- Calculate the number of $50 bills
  let fifty_bills_count := (total_money * fifty_bills_portion) / fifty_bill_value
  -- Calculate the number of $100 bills
  let hundred_bills_count := (total_money * (1 - fifty_bills_portion)) / hundred_bill_value
  -- Calculate the total number of bills
  fifty_bills_count + hundred_bills_count = 13 :=
by 
  -- Note: Proof omitted, as it is not required 
  sorry

end Uncle_Bradley_bills_l955_95523


namespace servant_service_duration_l955_95541

variables (x : ℕ) (total_compensation full_months received_compensation : ℕ)
variables (price_uniform compensation_cash : ℕ)

theorem servant_service_duration :
  total_compensation = 1000 →
  full_months = 12 →
  received_compensation = (compensation_cash + price_uniform) →
  received_compensation = 750 →
  total_compensation = (compensation_cash + price_uniform) →
  x / full_months = 750 / total_compensation →
  x = 9 :=
by sorry

end servant_service_duration_l955_95541


namespace expr_value_l955_95522

-- Define the constants
def w : ℤ := 3
def x : ℤ := -2
def y : ℤ := 1
def z : ℤ := 4

-- Define the expression
def expr : ℤ := (w^2 * x^2 * y * z) - (w * x^2 * y * z^2) + (w * y^3 * z^2) - (w * y^2 * x * z^4)

-- Statement to be proved
theorem expr_value : expr = 1536 :=
by
  -- Proof is omitted, so we use sorry.
  sorry

end expr_value_l955_95522


namespace equation1_solution_equation2_solution_l955_95528

theorem equation1_solution (x : ℝ) : x^2 - 10*x + 16 = 0 ↔ x = 2 ∨ x = 8 :=
by sorry

theorem equation2_solution (x : ℝ) : 2*x*(x-1) = x-1 ↔ x = 1 ∨ x = 1/2 :=
by sorry

end equation1_solution_equation2_solution_l955_95528


namespace percent_not_participating_music_sports_l955_95520

theorem percent_not_participating_music_sports
  (total_students : ℕ) 
  (both : ℕ) 
  (music_only : ℕ) 
  (sports_only : ℕ) 
  (not_participating : ℕ)
  (percentage_not_participating : ℝ) :
  total_students = 50 →
  both = 5 →
  music_only = 15 →
  sports_only = 20 →
  not_participating = total_students - (both + music_only + sports_only) →
  percentage_not_participating = (not_participating : ℝ) / (total_students : ℝ) * 100 →
  percentage_not_participating = 20 :=
by
  sorry

end percent_not_participating_music_sports_l955_95520


namespace arrangements_7_people_no_A_at_head_no_B_in_middle_l955_95549

theorem arrangements_7_people_no_A_at_head_no_B_in_middle :
  let n := 7
  let total_arrangements := Nat.factorial n
  let A_at_head := Nat.factorial (n - 1)
  let B_in_middle := A_at_head
  let overlap := Nat.factorial (n - 2)
  total_arrangements - 2 * A_at_head + overlap = 3720 :=
by
  let n := 7
  let total_arrangements := Nat.factorial n
  let A_at_head := Nat.factorial (n - 1)
  let B_in_middle := A_at_head
  let overlap := Nat.factorial (n - 2)
  show total_arrangements - 2 * A_at_head + overlap = 3720
  sorry

end arrangements_7_people_no_A_at_head_no_B_in_middle_l955_95549


namespace sufficient_not_necessary_condition_m_eq_1_sufficient_m_eq_1_not_necessary_l955_95568

variable (m : ℝ)

def vector_a : ℝ × ℝ := (1, m)
def vector_b : ℝ × ℝ := (4, -2)

def perp_vectors (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem sufficient_not_necessary_condition :
  perp_vectors (vector_a m) ((vector_a m).1 - (vector_b).1, (vector_a m).2 - (vector_b).2) ↔ (m = 1 ∨ m = -3) :=
by
  sorry

theorem m_eq_1_sufficient :
  (m = 1) → perp_vectors (vector_a m) ((vector_a m).1 - (vector_b).1, (vector_a m).2 - (vector_b).2) :=
by
  sorry

theorem m_eq_1_not_necessary :
  perp_vectors (vector_a m) ((vector_a m).1 - (vector_b).1, (vector_a m).2 - (vector_b).2) → (m = 1 ∨ m = -3) :=
by
  sorry

end sufficient_not_necessary_condition_m_eq_1_sufficient_m_eq_1_not_necessary_l955_95568


namespace proof_inequality_l955_95526

noncomputable def problem_statement (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) : Prop :=
  a + b + c ≤ (a ^ 4 + b ^ 4 + c ^ 4) / (a * b * c)

theorem proof_inequality (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) :
  problem_statement a b c h_a h_b h_c :=
by
  sorry

end proof_inequality_l955_95526


namespace smallest_5digit_palindrome_base2_expressed_as_3digit_palindrome_base5_l955_95542

def is_palindrome (n : ℕ) (b : ℕ) : Prop :=
  let digits := n.digits b
  digits = digits.reverse

theorem smallest_5digit_palindrome_base2_expressed_as_3digit_palindrome_base5 :
  ∃ n : ℕ, n = 0b11011 ∧ is_palindrome n 2 ∧ is_palindrome n 5 :=
by
  existsi 0b11011
  sorry

end smallest_5digit_palindrome_base2_expressed_as_3digit_palindrome_base5_l955_95542


namespace range_of_k_l955_95578

theorem range_of_k {x k : ℝ} :
  (∀ x, ((x - 2) * (x + 1) > 0) → ((2 * x + 7) * (x + k) < 0)) →
  (x = -3 ∨ x = -2) → 
  -3 ≤ k ∧ k < 2 :=
sorry

end range_of_k_l955_95578


namespace pencils_per_student_l955_95590

theorem pencils_per_student (total_pencils : ℤ) (num_students : ℤ) (pencils_per_student : ℤ)
  (h1 : total_pencils = 195)
  (h2 : num_students = 65) :
  total_pencils / num_students = 3 :=
by
  sorry

end pencils_per_student_l955_95590


namespace max_expression_value_l955_95548

theorem max_expression_value :
  ∀ (a b : ℝ), (100 ≤ a ∧ a ≤ 500) → (500 ≤ b ∧ b ≤ 1500) → 
  (∃ x, x = (b - 100) / (a + 50) ∧ ∀ y, y = (b - 100) / (a + 50) → y ≤ (28 / 3)) :=
by
  sorry

end max_expression_value_l955_95548


namespace number_of_ways_to_write_2024_l955_95508

theorem number_of_ways_to_write_2024 :
  (∃ a b c : ℕ, 2 * a + 3 * b + 4 * c = 2024) -> 
  (∃ n m p : ℕ, a = 3 * n + 2 * m + p ∧ n + m + p = 337) ->
  (∃ n m p : ℕ, n + m + p = 337 ∧ 2 * n * 3 + m * 2 + p * 6 = 2 * (57231 + 498)) :=
sorry

end number_of_ways_to_write_2024_l955_95508


namespace minimum_value_of_a_squared_plus_b_squared_l955_95505

def quadratic (a b x : ℝ) : ℝ := a * x^2 + (2 * b + 1) * x - a - 2

theorem minimum_value_of_a_squared_plus_b_squared (a b : ℝ) (hab : a ≠ 0)
  (hroot : ∃ (x : ℝ), 3 ≤ x ∧ x ≤ 4 ∧ quadratic a b x = 0) :
  a^2 + b^2 = 1 / 100 :=
sorry

end minimum_value_of_a_squared_plus_b_squared_l955_95505


namespace Olivia_paint_area_l955_95555

theorem Olivia_paint_area
  (length width height : ℕ) (door_window_area : ℕ) (bedrooms : ℕ)
  (h_length : length = 14) 
  (h_width : width = 11) 
  (h_height : height = 9) 
  (h_door_window_area : door_window_area = 70) 
  (h_bedrooms : bedrooms = 4) :
  (2 * (length * height) + 2 * (width * height) - door_window_area) * bedrooms = 1520 :=
by
  sorry

end Olivia_paint_area_l955_95555


namespace area_change_correct_l955_95545

theorem area_change_correct (L B : ℝ) (A : ℝ) (x : ℝ) (hx1 : A = L * B)
  (hx2 : ((L + (x / 100) * L) * (B - (x / 100) * B)) = A - (1 / 100) * A) :
  x = 10 := by
  sorry

end area_change_correct_l955_95545


namespace net_calorie_deficit_l955_95562

-- Define the conditions as constants.
def total_distance : ℕ := 3
def calories_burned_per_mile : ℕ := 150
def calories_in_candy_bar : ℕ := 200

-- Prove the net calorie deficit.
theorem net_calorie_deficit : total_distance * calories_burned_per_mile - calories_in_candy_bar = 250 := by
  sorry

end net_calorie_deficit_l955_95562


namespace ordered_triple_solution_l955_95544

theorem ordered_triple_solution (a b c : ℝ) (h1 : a > 5) (h2 : b > 5) (h3 : c > 5)
  (h4 : (a + 3) * (a + 3) / (b + c - 5) + (b + 5) * (b + 5) / (c + a - 7) + (c + 7) * (c + 7) / (a + b - 9) = 49) :
  (a, b, c) = (13, 9, 6) :=
sorry

end ordered_triple_solution_l955_95544


namespace inequality_proof_l955_95564

theorem inequality_proof (x y z : ℝ) (hx : x ≥ y) (hy : y ≥ z) (hz : z > 0) :
  (x^2 * y / z + y^2 * z / x + z^2 * x / y) ≥ (x^2 + y^2 + z^2) := 
  sorry

end inequality_proof_l955_95564


namespace area_transformation_l955_95539

variable {g : ℝ → ℝ}

theorem area_transformation (h : ∫ x, g x = 20) : ∫ x, -4 * g (x + 3) = 80 := by
  sorry

end area_transformation_l955_95539


namespace acute_triangle_angle_A_is_60_degrees_l955_95566

open Real

variables {A B C : ℝ} -- Assume A, B, C are reals representing the angles of the triangle

theorem acute_triangle_angle_A_is_60_degrees
  (h_acute : A < 90 ∧ B < 90 ∧ C < 90)
  (h_eq_dist : dist A O = dist A H) : A = 60 :=
  sorry

end acute_triangle_angle_A_is_60_degrees_l955_95566


namespace evaluate_nested_fraction_l955_95588

theorem evaluate_nested_fraction :
  (1 / (3 - (1 / (2 - (1 / (3 - (1 / (2 - (1 / 2))))))))) = 11 / 26 :=
by
  sorry

end evaluate_nested_fraction_l955_95588


namespace painters_workdays_l955_95543

theorem painters_workdays (d₁ d₂ : ℚ) (p₁ p₂ : ℕ)
  (h1 : p₁ = 5) (h2 : p₂ = 4) (rate: 5 * d₁ = 7.5) :
  (p₂:ℚ) * d₂ = 7.5 → d₂ = 1 + 7 / 8 :=
by
  sorry

end painters_workdays_l955_95543


namespace smallest_positive_m_l955_95532

theorem smallest_positive_m (m : ℕ) (h : ∃ n : ℤ, m^3 - 90 = n * (m + 9)) : m = 12 :=
by
  sorry

end smallest_positive_m_l955_95532


namespace center_of_tangent_circle_l955_95514

theorem center_of_tangent_circle (x y : ℝ) 
    (h1 : 3 * x - 4 * y = 20) 
    (h2 : 3 * x - 4 * y = -40) 
    (h3 : x - 3 * y = 0) : 
    (x, y) = (-6, -2) := 
by
    sorry

end center_of_tangent_circle_l955_95514


namespace smallest_third_term_arith_seq_l955_95553

theorem smallest_third_term_arith_seq {a d : ℕ} 
  (h1 : a > 0) 
  (h2 : d > 0) 
  (sum_eq : 5 * a + 10 * d = 80) : 
  a + 2 * d = 16 := 
by {
  sorry
}

end smallest_third_term_arith_seq_l955_95553


namespace positive_difference_of_fraction_results_l955_95527

theorem positive_difference_of_fraction_results :
  let a := 8
  let expr1 := (a ^ 2 - a ^ 2) / a
  let expr2 := (a ^ 2 * a ^ 2) / a
  expr1 = 0 ∧ expr2 = 512 ∧ (expr2 - expr1) = 512 := 
by
  sorry

end positive_difference_of_fraction_results_l955_95527


namespace total_amount_740_l955_95580

theorem total_amount_740 (x y z : ℝ) (hz : z = 200) (hy : y = 1.20 * z) (hx : x = 1.25 * y) : 
  x + y + z = 740 := by
  sorry

end total_amount_740_l955_95580


namespace hiking_rate_up_the_hill_l955_95573

theorem hiking_rate_up_the_hill (r_down : ℝ) (t_total : ℝ) (t_up : ℝ) (r_up : ℝ) :
  r_down = 6 ∧ t_total = 3 ∧ t_up = 1.2 → r_up * t_up = 9 * t_up :=
by
  intro h
  let ⟨hrd, htt, htu⟩ := h
  sorry

end hiking_rate_up_the_hill_l955_95573


namespace inequality_proof_l955_95500

theorem inequality_proof (a : ℝ) (h1 : 0 < a) (h2 : a < 1) : 
  (1 / a + 4 / (1 - a) ≥ 9) := 
sorry

end inequality_proof_l955_95500


namespace floor_double_l955_95515

theorem floor_double (a : ℝ) (h : 0 < a) : 
  ⌊2 * a⌋ = ⌊a⌋ + ⌊a + 1/2⌋ :=
sorry

end floor_double_l955_95515


namespace max_length_shortest_arc_l955_95597

theorem max_length_shortest_arc (C : ℝ) (hC : C = 84) : 
  ∃ shortest_arc_length : ℝ, shortest_arc_length = 2 :=
by
  -- now prove it
  sorry

end max_length_shortest_arc_l955_95597


namespace train_crossing_time_l955_95574

noncomputable def length_of_train : ℕ := 250
noncomputable def length_of_bridge : ℕ := 350
noncomputable def speed_of_train_kmph : ℕ := 72

noncomputable def speed_of_train_mps : ℕ := (speed_of_train_kmph * 1000) / 3600

noncomputable def total_distance : ℕ := length_of_train + length_of_bridge

theorem train_crossing_time : total_distance / speed_of_train_mps = 30 := by
  sorry

end train_crossing_time_l955_95574


namespace ratio_of_lateral_edges_l955_95531

theorem ratio_of_lateral_edges (A B : ℝ) (hA : A > 0) (hB : B > 0) (h : A / B = 4 / 9) : 
  let upper_length_ratio := 2
  let lower_length_ratio := 3
  upper_length_ratio / lower_length_ratio = 2 / 3 :=
by 
  sorry

end ratio_of_lateral_edges_l955_95531


namespace sum_of_abcd_is_1_l955_95569

theorem sum_of_abcd_is_1
  (a b c d : ℤ)
  (h1 : (x^2 + a*x + b)*(x^2 + c*x + d) = x^4 + 2*x^3 + x^2 + 8*x - 12) :
  a + b + c + d = 1 := by
  sorry

end sum_of_abcd_is_1_l955_95569


namespace domain_of_expression_l955_95584

theorem domain_of_expression (x : ℝ) : 
  x + 3 ≥ 0 → 7 - x > 0 → (x ∈ Set.Icc (-3) 7) :=
by 
  intros h1 h2
  sorry

end domain_of_expression_l955_95584


namespace terry_age_proof_l955_95571

-- Condition 1: In 10 years, Terry will be 4 times the age that Nora is currently.
-- Condition 2: Nora is currently 10 years old.
-- We need to prove that Terry's current age is 30 years old.

variable (Terry_now Terry_in_10 Nora_now : ℕ)

theorem terry_age_proof (h1: Terry_in_10 = 4 * Nora_now) (h2: Nora_now = 10) (h3: Terry_in_10 = Terry_now + 10) : Terry_now = 30 := 
by
  sorry

end terry_age_proof_l955_95571


namespace mushroom_pickers_at_least_50_l955_95595

-- Given conditions
variables (a : Fin 7 → ℕ) -- Each picker collects a different number of mushrooms.
variables (distinct : ∀ i j, i ≠ j → a i ≠ a j)
variable (total_mushrooms : (Finset.univ.sum a) = 100)

-- The proof that at least three of the pickers collected at least 50 mushrooms together
theorem mushroom_pickers_at_least_50 (a : Fin 7 → ℕ) (distinct : ∀ i j, i ≠ j → a i ≠ a j)
    (total_mushrooms : (Finset.univ.sum a) = 100) :
    ∃ i j k : Fin 7, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ (a i + a j + a k) ≥ 50 :=
sorry

end mushroom_pickers_at_least_50_l955_95595


namespace finish_remaining_work_l955_95556

theorem finish_remaining_work (x y : ℕ) (hx : x = 30) (hy : y = 15) (hy_work_days : y_work_days = 10) :
  x = 10 :=
by
  sorry

end finish_remaining_work_l955_95556


namespace find_x_l955_95577

def star (a b c d : ℤ) : ℤ × ℤ := (a + c, b - d)

theorem find_x 
  (x y : ℤ) 
  (h_star1 : star 5 4 2 2 = (7, 2)) 
  (h_eq : star x y 3 3 = (7, 2)) : 
  x = 4 := 
sorry

end find_x_l955_95577


namespace percentage_of_burpees_is_10_l955_95524

-- Definitions for each exercise count
def jumping_jacks : ℕ := 25
def pushups : ℕ := 15
def situps : ℕ := 30
def burpees : ℕ := 10
def lunges : ℕ := 20

-- Total number of exercises
def total_exercises : ℕ := jumping_jacks + pushups + situps + burpees + lunges

-- The proof statement
theorem percentage_of_burpees_is_10 :
  (burpees * 100) / total_exercises = 10 :=
by
  sorry

end percentage_of_burpees_is_10_l955_95524


namespace largest_lcm_l955_95506

theorem largest_lcm :
  ∀ (a b c d e f : ℕ),
  a = Nat.lcm 18 2 →
  b = Nat.lcm 18 4 →
  c = Nat.lcm 18 6 →
  d = Nat.lcm 18 9 →
  e = Nat.lcm 18 12 →
  f = Nat.lcm 18 16 →
  max (max (max (max (max a b) c) d) e) f = 144 :=
by
  intros a b c d e f ha hb hc hd he hf
  sorry

end largest_lcm_l955_95506


namespace tangent_triangle_perimeter_acute_tangent_triangle_perimeter_obtuse_l955_95552

theorem tangent_triangle_perimeter_acute (a b c: ℝ) (h1: a^2 + b^2 > c^2) (h2: b^2 + c^2 > a^2) (h3: c^2 + a^2 > b^2) :
  2 * a * b * c * (1 / (b^2 + c^2 - a^2) + 1 / (c^2 + a^2 - b^2) + 1 / (a^2 + b^2 - c^2)) = 
  2 * a * b * c * (1 / (b^2 + c^2 - a^2) + 1 / (c^2 + a^2 - b^2) + 1 / (a^2 + b^2 - c^2)) := 
by sorry -- proof goes here

theorem tangent_triangle_perimeter_obtuse (a b c: ℝ) (h1: a^2 > b^2 + c^2) :
  2 * a * b * c / (a^2 - b^2 - c^2) = 2 * a * b * c / (a^2 - b^2 - c^2) := 
by sorry -- proof goes here

end tangent_triangle_perimeter_acute_tangent_triangle_perimeter_obtuse_l955_95552


namespace builder_total_amount_paid_l955_95554

theorem builder_total_amount_paid :
  let cost_drill_bits := 5 * 6
  let tax_drill_bits := 0.10 * cost_drill_bits
  let total_cost_drill_bits := cost_drill_bits + tax_drill_bits

  let cost_hammers := 3 * 8
  let discount_hammers := 0.05 * cost_hammers
  let total_cost_hammers := cost_hammers - discount_hammers

  let cost_toolbox := 25
  let tax_toolbox := 0.15 * cost_toolbox
  let total_cost_toolbox := cost_toolbox + tax_toolbox

  let total_amount_paid := total_cost_drill_bits + total_cost_hammers + total_cost_toolbox

  total_amount_paid = 84.55 :=
by
  sorry

end builder_total_amount_paid_l955_95554


namespace arithmetic_geometric_fraction_l955_95547

theorem arithmetic_geometric_fraction (a x₁ x₂ b y₁ y₂ : ℝ) 
  (h₁ : x₁ + x₂ = a + b) 
  (h₂ : y₁ * y₂ = ab) : 
  (x₁ + x₂) / (y₁ * y₂) = (a + b) / (ab) := 
by
  sorry

end arithmetic_geometric_fraction_l955_95547


namespace trig_identity_l955_95582

theorem trig_identity (α : ℝ) (h1 : Real.cos α = -4/5) (h2 : π/2 < α ∧ α < π) : 
  - (Real.sin (2 * α) / Real.cos α) = -6/5 :=
by
  sorry

end trig_identity_l955_95582


namespace find_k_l955_95546

theorem find_k (k : ℝ) (h : 64 / k = 8) : k = 8 := 
sorry

end find_k_l955_95546


namespace angle_terminal_side_l955_95502

def angle_on_line (β : ℝ) : Prop :=
  ∃ n : ℤ, β = 135 + n * 180

def angle_in_range (β : ℝ) : Prop :=
  -360 < β ∧ β < 360

theorem angle_terminal_side :
  ∀ β, angle_on_line β → angle_in_range β → β = -225 ∨ β = -45 ∨ β = 135 ∨ β = 315 :=
by
  intros β h_line h_range
  sorry

end angle_terminal_side_l955_95502


namespace law_of_sines_l955_95579

theorem law_of_sines (a b c : ℝ) (A B C : ℝ) (R : ℝ) 
  (hA : a = 2 * R * Real.sin A)
  (hEquilateral1 : b = 2 * R * Real.sin B)
  (hEquilateral2 : c = 2 * R * Real.sin C):
  (a / Real.sin A) = (b / Real.sin B) ∧ 
  (b / Real.sin B) = (c / Real.sin C) ∧ 
  (c / Real.sin C) = 2 * R :=
by
  sorry

end law_of_sines_l955_95579


namespace art_club_students_l955_95598

theorem art_club_students 
    (students artworks_per_student_per_quarter quarters_per_year artworks_in_two_years : ℕ) 
    (h1 : artworks_per_student_per_quarter = 2)
    (h2 : quarters_per_year = 4) 
    (h3 : artworks_in_two_years = 240) 
    (h4 : students * (artworks_per_student_per_quarter * quarters_per_year) * 2 = artworks_in_two_years) :
    students = 15 := 
by
    -- Given conditions for the problem
    sorry

end art_club_students_l955_95598


namespace alcohol_percentage_l955_95501

theorem alcohol_percentage (x : ℝ)
  (h1 : 8 * x / 100 + 2 * 12 / 100 = 22.4 * 10 / 100) : x = 25 :=
by
  -- skip the proof
  sorry

end alcohol_percentage_l955_95501


namespace gcd_9011_2147_l955_95517

theorem gcd_9011_2147 : Int.gcd 9011 2147 = 1 := sorry

end gcd_9011_2147_l955_95517


namespace snow_probability_january_first_week_l955_95589

noncomputable def P_snow_at_least_once_first_week : ℚ :=
  1 - ((2 / 3) ^ 4 * (3 / 4) ^ 3)

theorem snow_probability_january_first_week :
  P_snow_at_least_once_first_week = 11 / 12 :=
by
  sorry

end snow_probability_january_first_week_l955_95589


namespace sum_of_selected_terms_l955_95504

variable {a : ℕ → ℚ} -- Define the arithmetic sequence as a function from natural numbers to rational numbers

noncomputable def sum_first_n_terms (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))

theorem sum_of_selected_terms (h₁ : sum_first_n_terms a 13 = 39) : a 6 + a 7 + a 8 = 13 :=
sorry

end sum_of_selected_terms_l955_95504


namespace martin_boxes_l955_95533

theorem martin_boxes (total_crayons : ℕ) (crayons_per_box : ℕ) (number_of_boxes : ℕ) 
  (h1 : total_crayons = 56) (h2 : crayons_per_box = 7) 
  (h3 : total_crayons = crayons_per_box * number_of_boxes) : 
  number_of_boxes = 8 :=
by 
  sorry

end martin_boxes_l955_95533
