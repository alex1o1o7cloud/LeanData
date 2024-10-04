import Mathlib

namespace Tim_age_l9_9875

theorem Tim_age : ∃ (T : ℕ), (T = (3 * T + 2 - 12)) ∧ (T = 5) :=
by
  existsi 5
  sorry

end Tim_age_l9_9875


namespace vector_problem_solution_l9_9076

variables (a b c : ℤ × ℤ) (m n : ℤ)

def parallel (v1 v2 : ℤ × ℤ) : Prop := v1.1 * v2.2 = v1.2 * v2.1
def perpendicular (v1 v2 : ℤ × ℤ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem vector_problem_solution
  (a_eq : a = (1, -2))
  (b_eq : b = (2, m - 1))
  (c_eq : c = (4, n))
  (h1 : parallel a b)
  (h2 : perpendicular b c) :
  m + n = -1 := by
  sorry

end vector_problem_solution_l9_9076


namespace rain_on_Tuesday_correct_l9_9674

-- Let the amount of rain on Monday be represented by m
def rain_on_Monday : ℝ := 0.9

-- Let the difference in rain between Monday and Tuesday be represented by d
def rain_difference : ℝ := 0.7

-- Define the calculated amount of rain on Tuesday
def rain_on_Tuesday : ℝ := rain_on_Monday - rain_difference

-- The statement we need to prove
theorem rain_on_Tuesday_correct : rain_on_Tuesday = 0.2 := 
by
  -- Proof omitted (to be provided)
  sorry

end rain_on_Tuesday_correct_l9_9674


namespace complex_solution_l9_9049

noncomputable def solution : ℂ := mk (Real.of_rat (-11/7)) (Real.of_rat 4/7)

theorem complex_solution :
  let z := solution
  in 5 * z - 2 * complex.I * conj z = -9 + 6 * complex.I :=
by
  let z := solution
  trivial -- This is where the proof would go.

end complex_solution_l9_9049


namespace opposite_of_2023_l9_9448

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l9_9448


namespace grant_school_students_l9_9690

theorem grant_school_students (S : ℕ) 
  (h1 : S / 3 = x) 
  (h2 : x / 4 = 15) : 
  S = 180 := 
sorry

end grant_school_students_l9_9690


namespace inequality_solution_sets_l9_9575

noncomputable def solve_inequality (m : ℝ) : Set ℝ :=
  if m = 0 then Set.Iic (-2)
  else if m > 0 then Set.Iic (-2) ∪ Set.Ici (1 / m)
  else if (-(1/2) < m ∧ m < 0) then Set.Ioo (1 / m) (-2)
  else if m = -(1 / 2) then ∅
  else Set.Ioo (-2) (1 / m)

theorem inequality_solution_sets (m : ℝ) :
  solve_inequality m = 
    if m = 0 then Set.Iic (-2)
    else if m > 0 then Set.Iic (-2) ∪ Set.Ici (1 / m)
    else if (-(1/2) < m ∧ m < 0) then Set.Ioo (1 / m) (-2)
    else if m = -(1 / 2) then ∅
    else Set.Ioo (-2) (1 / m) :=
sorry

end inequality_solution_sets_l9_9575


namespace parabola_focus_l9_9055

noncomputable def focus_of_parabola (a b : ℝ) : ℝ × ℝ :=
  (0, 1 / (4 * a) - b)

theorem parabola_focus : focus_of_parabola 4 3 = (0, -47 / 16) :=
by
  -- Function definition: focus_of_parabola a b gives the focus of y = ax^2 - b
  -- Given: a = 4, b = 3
  -- Focus: (0, 1 / (4 * 4) - 3)
  -- Proof: Skipping detailed algebraic manipulation, assume function correctness
  sorry

end parabola_focus_l9_9055


namespace venus_hall_meal_cost_l9_9121

theorem venus_hall_meal_cost (V : ℕ) :
  let caesars_total_cost := 800 + 30 * 60;
  let venus_hall_total_cost := 500 + V * 60;
  caesars_total_cost = venus_hall_total_cost → V = 35 :=
by
  let caesars_total_cost := 800 + 30 * 60
  let venus_hall_total_cost := 500 + V * 60
  intros h
  sorry

end venus_hall_meal_cost_l9_9121


namespace maximum_value_of_f_intervals_of_monotonic_increase_l9_9559

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := 
  let a1 := a x
  let b1 := b x
  a1.1 * (a1.1 + b1.1) + a1.2 * (a1.2 + b1.2)

theorem maximum_value_of_f :
  ∃ x : ℝ, f x = 3 / 2 + Real.sqrt 2 / 2 := sorry

theorem intervals_of_monotonic_increase :
  ∃ I1 I2 : Set ℝ, 
  I1 = Set.Icc 0 (Real.pi / 8) ∧ 
  I2 = Set.Icc (5 * Real.pi / 8) Real.pi ∧ 
  (∀ x ∈ I1, ∀ y ∈ I2, x ≤ y ∧ f x ≤ f y) ∧
  (∀ x y, x ∈ I1 → y ∈ I1 → x < y → f x < f y) ∧
  (∀ x y, x ∈ I2 → y ∈ I2 → x < y → f x < f y) := sorry

end maximum_value_of_f_intervals_of_monotonic_increase_l9_9559


namespace sum_gcd_lcm_l9_9013

theorem sum_gcd_lcm (a₁ a₂ : ℕ) (h₁ : a₁ = 36) (h₂ : a₂ = 495) :
  Nat.gcd a₁ a₂ + Nat.lcm a₁ a₂ = 1989 :=
by
  -- Proof can be added here
  sorry

end sum_gcd_lcm_l9_9013


namespace meaningful_expr_l9_9089

theorem meaningful_expr (x : ℝ) : 
  (∃ (y : ℝ), y = sqrt (x + 2) ∧ x - 1 ≠ 0) ↔ (x ≥ -2 ∧ x ≠ 1) :=
by
  sorry

end meaningful_expr_l9_9089


namespace number_of_ways_to_enter_and_exit_l9_9033

theorem number_of_ways_to_enter_and_exit (n : ℕ) (h : n = 4) : (n * n) = 16 := by
  sorry

end number_of_ways_to_enter_and_exit_l9_9033


namespace part1_B_correct_part2_at_least_two_correct_l9_9667

noncomputable def P_A : ℚ := 1 / 2
noncomputable def P_C : ℚ := 3 / 4
noncomputable def P_B : ℚ := 2 / 3

def prob_A_correct : ℚ := P_A
def prob_AC_incorrect : ℚ := (1 - P_A) * (1 - P_C)
def prob_BC_correct : ℚ := P_B * P_C

-- Question(I): Prove the probabilities of B and C answering correctly
theorem part1_B_correct :
  prob_AC_incorrect = 1 / 8 ∧ 
  prob_BC_correct = 1 / 2 → 
  P_B = 2 / 3 ∧ P_C = 3 / 4 :=
sorry

-- Question(II): Prove the probability that at least 2 out of A, B, and C answer correctly
theorem part2_at_least_two_correct :
  P_A = 1 / 2 ∧ 
  P_B = 2 / 3 ∧ 
  P_C = 3 / 4 →
  let P_at_least_two := P_A * P_B * P_C + (1 - P_A) * P_B * P_C + P_A * (1 - P_B) * P_C + P_A * P_B * (1 - P_C) in
  P_at_least_two = 17 / 24 :=
sorry

end part1_B_correct_part2_at_least_two_correct_l9_9667


namespace free_donut_coupons_total_l9_9692

theorem free_donut_coupons_total :
  let books_per_coupon := 5
  let quinn_books := 5 * 5 -- (2 books + 3 books) * 5 cycles
  let taylor_books := 1 + 4 * 9
  let jordan_books := 3 * 10
  let quinn_coupons := quinn_books / books_per_coupon
  let taylor_coupons := taylor_books / books_per_coupon
  let jordan_coupons := jordan_books / books_per_coupon
  quinn_coupons + taylor_coupons + jordan_coupons = 18 :=
by {
  let books_per_coupon := 5
  let quinn_books := 5 * 5 -- (2 books + 3 books) * 5 cycles
  let taylor_books := 1 + 4 * 9
  let jordan_books := 3 * 10
  let quinn_coupons := quinn_books / books_per_coupon
  let taylor_coupons := taylor_books / books_per_coupon
  let jordan_coupons := jordan_books / books_per_coupon
  have h_quinn_coupons : quinn_coupons = 5 := by sorry
  have h_taylor_coupons : taylor_coupons = 7 := by sorry
  have h_jordan_coupons : jordan_coupons = 6 := by sorry
  show quinn_coupons + taylor_coupons + jordan_coupons = 18, from by {
    calc
      5 + 7 + 6 = 18 : by norm_num
  }
}

end free_donut_coupons_total_l9_9692


namespace opposite_of_2023_l9_9205

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l9_9205


namespace pipe_c_empty_time_l9_9007

theorem pipe_c_empty_time (x : ℝ) :
  (4/20 + 4/30 + 4/x) * 3 = 1 → x = 6 :=
by
  sorry

end pipe_c_empty_time_l9_9007


namespace opposite_of_2023_l9_9408

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l9_9408


namespace happy_number_part1_happy_number_part2_happy_number_part3_l9_9843

section HappyEquations

def is_happy_eq (a b c : ℤ) : Prop :=
  ∃ x1 x2 : ℤ, a ≠ 0 ∧ a * x1 * x1 + b * x1 + c = 0 ∧ a * x2 * x2 + b * x2 + c = 0

def happy_number (a b c : ℤ) : ℚ :=
  (4 * a * c - b ^ 2) / (4 * a)

def happy_to_each_other (a b c p q r : ℤ) : Prop :=
  let Fa : ℚ := happy_number a b c
  let Fb : ℚ := happy_number p q r
  |r * Fa - c * Fb| = 0

theorem happy_number_part1 :
  happy_number 1 (-2) (-3) = -4 :=
by sorry

theorem happy_number_part2 (m : ℤ) (h : 1 < m ∧ m < 6) :
  is_happy_eq 1 (2 * m - 1) (m ^ 2 - 2 * m - 3) →
  m = 3 ∧ happy_number 1 (2 * m - 1) (m ^ 2 - 2 * m - 3) = -25 / 4 :=
by sorry

theorem happy_number_part3 (m n : ℤ) :
  is_happy_eq 1 (-m) (m + 1) ∧ is_happy_eq 1 (-(n + 2)) (2 * n) →
  happy_to_each_other 1 (-m) (m + 1) 1 (-(n + 2)) (2 * n) →
  n = 0 ∨ n = 3 ∨ n = 3 / 2 :=
by sorry

end HappyEquations

end happy_number_part1_happy_number_part2_happy_number_part3_l9_9843


namespace opposite_of_2023_is_neg2023_l9_9322

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l9_9322


namespace max_value_of_expr_l9_9981

noncomputable def max_expr (a b : ℝ) (h : a + b = 5) : ℝ :=
  a^4 * b + a^3 * b + a^2 * b + a * b + a * b^2 + a * b^3 + a * b^4

theorem max_value_of_expr (a b : ℝ) (h : a + b = 5) : max_expr a b h ≤ 6084 / 17 :=
sorry

end max_value_of_expr_l9_9981


namespace tan_negative_angle_l9_9662

theorem tan_negative_angle (m : ℝ) (h1 : m = Real.cos (80 * Real.pi / 180)) (h2 : m = Real.sin (10 * Real.pi / 180)) :
  Real.tan (-440 * Real.pi / 180) = - (Real.sqrt (1 - m^2)) / m :=
by
  sorry

end tan_negative_angle_l9_9662


namespace minimum_value_expression_l9_9929

theorem minimum_value_expression 
  (a b c d : ℝ)
  (h1 : (2 * a^2 - Real.log a) / b = 1)
  (h2 : (3 * c - 2) / d = 1) :
  ∃ min_val : ℝ, min_val = (a - c)^2 + (b - d)^2 ∧ min_val = 1 / 10 :=
by {
  sorry
}

end minimum_value_expression_l9_9929


namespace opposite_of_2023_l9_9443

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l9_9443


namespace opposite_of_2023_l9_9140

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l9_9140


namespace arithmetic_sequence_terms_l9_9959

theorem arithmetic_sequence_terms (a d n : ℕ) 
  (h_sum_first_3 : 3 * a + 3 * d = 34)
  (h_sum_last_3 : 3 * a + 3 * d * (n - 1) = 146)
  (h_sum_all : n * (2 * a + (n - 1) * d) = 2 * 390) : 
  n = 13 :=
by
  sorry

end arithmetic_sequence_terms_l9_9959


namespace first_candidate_percentage_l9_9970

-- Conditions
def total_votes : ℕ := 600
def second_candidate_votes : ℕ := 240
def first_candidate_votes : ℕ := total_votes - second_candidate_votes

-- Question and correct answer
theorem first_candidate_percentage : (first_candidate_votes * 100) / total_votes = 60 := by
  sorry

end first_candidate_percentage_l9_9970


namespace value_of_expression_l9_9963

theorem value_of_expression (x : ℝ) (h : 2 * x^2 + 3 * x + 7 = 8) : 4 * x^2 + 6 * x - 9 = -7 := by
  sorry

end value_of_expression_l9_9963


namespace range_of_a_l9_9958

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x| ≥ a * x) → -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l9_9958


namespace four_star_three_l9_9657

def star (a b : ℕ) : ℕ := a^2 - a * b + b^2 + 2 * a * b

theorem four_star_three : star 4 3 = 37 :=
by
  -- here we would normally provide the proof steps
  sorry

end four_star_three_l9_9657


namespace quadratic_intersects_x_axis_l9_9942

theorem quadratic_intersects_x_axis (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 7 * x - 7 = 0) ↔ k ≥ -7/4 ∧ k ≠ 0 :=
by
  sorry

end quadratic_intersects_x_axis_l9_9942


namespace opposite_of_2023_l9_9198

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l9_9198


namespace distinct_license_plates_count_l9_9601

def num_digit_choices : Nat := 10
def num_letter_choices : Nat := 26
def num_digits : Nat := 5
def num_letters : Nat := 3

theorem distinct_license_plates_count :
  (num_digit_choices ^ num_digits) * (num_letter_choices ^ num_letters) = 1757600000 := 
sorry

end distinct_license_plates_count_l9_9601


namespace acorns_given_is_correct_l9_9990

-- Define initial conditions
def initial_acorns : ℕ := 16
def remaining_acorns : ℕ := 9

-- Define the number of acorns given to her sister
def acorns_given : ℕ := initial_acorns - remaining_acorns

-- Theorem statement
theorem acorns_given_is_correct : acorns_given = 7 := by
  sorry

end acorns_given_is_correct_l9_9990


namespace opposite_of_2023_l9_9167

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l9_9167


namespace problem1_problem2_l9_9544

variable {A B C : ℝ} {AC BC : ℝ}

-- Condition: BC = 2AC
def condition1 (AC BC : ℝ) : Prop := BC = 2 * AC

-- Problem 1: Prove 4cos^2(B) - cos^2(A) = 3
theorem problem1 (h : condition1 AC BC) :
  4 * Real.cos B ^ 2 - Real.cos A ^ 2 = 3 :=
sorry

-- Problem 2: Prove the maximum value of (sin(A) / (2cos(B) + cos(A))) is 2/3 for A ∈ (0, π)
theorem problem2 (h : condition1 AC BC) (hA : 0 < A ∧ A < Real.pi) :
  ∃ t : ℝ, (t = Real.sin A / (2 * Real.cos B + Real.cos A) ∧ t ≤ 2/3) :=
sorry

end problem1_problem2_l9_9544


namespace radius_of_C3_correct_l9_9554

noncomputable def radius_of_C3
  (C1 C2 C3 : Type)
  (r1 r2 : ℝ)
  (A B T : Type)
  (TA : ℝ) : ℝ :=
if h1 : r1 = 2 ∧ r2 = 3
    ∧ (TA = 4) -- Conditions 1 and 2
   then 8
   else 0

-- Proof statement
theorem radius_of_C3_correct
  (C1 C2 C3 : Type)
  (r1 r2 : ℝ)
  (A B T : Type)
  (TA : ℝ)
  (h1 : r1 = 2)
  (h2 : r2 = 3)
  (h3 : TA = 4) :
  radius_of_C3 C1 C2 C3 r1 r2 A B T TA = 8 :=
by 
  sorry

end radius_of_C3_correct_l9_9554


namespace intersection_A_B_l9_9550

def A : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ y = 2 * x + 5}
def B : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ y = 1 - 2 * x}
def inter : Set (ℝ × ℝ) := {(x, y) | x = -1 ∧ y = 3}

theorem intersection_A_B :
  A ∩ B = inter :=
sorry

end intersection_A_B_l9_9550


namespace percentage_more_likely_to_lose_both_l9_9041

def first_lawsuit_win_probability : ℝ := 0.30
def first_lawsuit_lose_probability : ℝ := 0.70
def second_lawsuit_win_probability : ℝ := 0.50
def second_lawsuit_lose_probability : ℝ := 0.50

theorem percentage_more_likely_to_lose_both :
  (second_lawsuit_lose_probability * first_lawsuit_lose_probability - second_lawsuit_win_probability * first_lawsuit_win_probability) / (second_lawsuit_win_probability * first_lawsuit_win_probability) * 100 = 133.33 :=
by
  sorry

end percentage_more_likely_to_lose_both_l9_9041


namespace sum_r_p_values_l9_9075

def p (x : ℝ) : ℝ := |x| - 2
def r (x : ℝ) : ℝ := -|p x - 1|
def r_p (x : ℝ) : ℝ := r (p x)

theorem sum_r_p_values :
  (r_p (-4) + r_p (-3) + r_p (-2) + r_p (-1) + r_p 0 + r_p 1 + r_p 2 + r_p 3 + r_p 4) = -11 :=
by 
  -- Proof omitted
  sorry

end sum_r_p_values_l9_9075


namespace cannot_be_sum_of_six_consecutive_odd_integers_l9_9016

theorem cannot_be_sum_of_six_consecutive_odd_integers (S : ℕ) :
  (S = 90 ∨ S = 150) ->
  ∀ n : ℤ, ¬(S = n + (n+2) + (n+4) + (n+6) + (n+8) + (n+10)) :=
by
  intro h
  intro n
  cases h
  case inl => 
    sorry
  case inr => 
    sorry

end cannot_be_sum_of_six_consecutive_odd_integers_l9_9016


namespace opposite_of_2023_l9_9253

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l9_9253


namespace sequence_expression_l9_9643

theorem sequence_expression {a : ℕ → ℝ} (h1 : ∀ n, a (n + 1) ^ 2 = a n ^ 2 + 4)
  (h2 : a 1 = 1) (h3 : ∀ n, a n > 0) : ∀ n, a n = Real.sqrt (4 * n - 3) := by
  sorry

end sequence_expression_l9_9643


namespace roots_magnitudes_less_than_one_l9_9649

theorem roots_magnitudes_less_than_one
  (A B C D : ℝ)
  (h1 : ∀ (r : ℝ), |r| < 1 → r ≠ 0 → (r ^ 2 + A * r + B = 0))
  (h2 : ∀ (r : ℝ), |r| < 1 → r ≠ 0 → (r ^ 2 + C * r + D = 0)) :
  ∀ (r : ℝ), |r| < 1 → r ≠ 0 → (r ^ 2 + (1 / 2 * (A + C)) * r + (1 / 2 * (B + D)) = 0) :=
by
  sorry

end roots_magnitudes_less_than_one_l9_9649


namespace logarithmic_inequality_and_integral_l9_9557

theorem logarithmic_inequality_and_integral :
  let a := Real.log 3 / Real.log 2
  let b := Real.log 2 / Real.log 3
  let c := 2 / Real.pi^2
  a > b ∧ b > c :=
by
  let a := Real.log 3 / Real.log 2
  let b := Real.log 2 / Real.log 3
  let c := 2 / Real.pi^2
  sorry

end logarithmic_inequality_and_integral_l9_9557


namespace largest_multiple_of_8_less_than_100_l9_9463

theorem largest_multiple_of_8_less_than_100 : ∃ (n : ℕ), n < 100 ∧ 8 ∣ n ∧ ∀ (m : ℕ), m < 100 ∧ 8 ∣ m → m ≤ n :=
sorry

end largest_multiple_of_8_less_than_100_l9_9463


namespace negation_one_zero_l9_9707

theorem negation_one_zero (a b : ℝ) (h : a ≠ 0):
  ¬ (∃! x : ℝ, a * x + b = 0) ↔ (¬ ∃ x : ℝ, a * x + b = 0 ∨ ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁ + b = 0 ∧ a * x₂ + b = 0) := by
sorry

end negation_one_zero_l9_9707


namespace opposite_of_2023_l9_9152

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l9_9152


namespace slightly_used_crayons_count_l9_9590

-- Definitions
def total_crayons := 120
def new_crayons := total_crayons * (1/3)
def broken_crayons := total_crayons * (20/100)
def slightly_used_crayons := total_crayons - new_crayons - broken_crayons

-- Theorem statement
theorem slightly_used_crayons_count :
  slightly_used_crayons = 56 :=
by
  sorry

end slightly_used_crayons_count_l9_9590


namespace rectangle_area_l9_9609

theorem rectangle_area (x : ℝ) (w : ℝ) (h : ℝ) (H1 : x^2 = w^2 + h^2) (H2 : h = 3 * w) : 
  (w * h = (3 * x^2) / 10) :=
by sorry

end rectangle_area_l9_9609


namespace total_cost_of_tshirts_l9_9505

theorem total_cost_of_tshirts
  (White_packs : ℕ := 3) (Blue_packs : ℕ := 2) (Red_packs : ℕ := 4) (Green_packs : ℕ := 1) 
  (White_price_per_pack : ℝ := 12) (Blue_price_per_pack : ℝ := 8) (Red_price_per_pack : ℝ := 10) (Green_price_per_pack : ℝ := 6) 
  (White_discount : ℝ := 0.10) (Blue_discount : ℝ := 0.05) (Red_discount : ℝ := 0.15) (Green_discount : ℝ := 0.00) :
  White_packs * White_price_per_pack * (1 - White_discount) +
  Blue_packs * Blue_price_per_pack * (1 - Blue_discount) +
  Red_packs * Red_price_per_pack * (1 - Red_discount) +
  Green_packs * Green_price_per_pack * (1 - Green_discount) = 87.60 := by
    sorry

end total_cost_of_tshirts_l9_9505


namespace opposite_of_2023_l9_9179

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l9_9179


namespace sum_of_coordinates_B_l9_9996

theorem sum_of_coordinates_B
  (x y : ℤ)
  (Mx My : ℤ)
  (Ax Ay : ℤ)
  (M : Mx = 2 ∧ My = -3)
  (A : Ax = -4 ∧ Ay = -5)
  (midpoint_x : (x + Ax) / 2 = Mx)
  (midpoint_y : (y + Ay) / 2 = My) :
  x + y = 7 :=
by
  sorry

end sum_of_coordinates_B_l9_9996


namespace opposite_of_2023_l9_9401

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9401


namespace total_money_collected_l9_9754

theorem total_money_collected (attendees : ℕ) (reserved_price unreserved_price : ℝ) (reserved_sold unreserved_sold : ℕ)
  (h_attendees : attendees = 1096)
  (h_reserved_price : reserved_price = 25.00)
  (h_unreserved_price : unreserved_price = 20.00)
  (h_reserved_sold : reserved_sold = 246)
  (h_unreserved_sold : unreserved_sold = 246) :
  (reserved_price * reserved_sold + unreserved_price * unreserved_sold) = 11070.00 :=
by
  sorry

end total_money_collected_l9_9754


namespace expected_sixes_in_three_rolls_l9_9868

theorem expected_sixes_in_three_rolls : 
  (∑ k in Finset.range 4, k * (Nat.choose 3 k) * (1/6)^k * (5/6)^(3-k)) = 1/2 := 
by
  sorry

end expected_sixes_in_three_rolls_l9_9868


namespace zhang_san_not_losing_probability_l9_9765

theorem zhang_san_not_losing_probability (p_win p_draw : ℚ) (h_win : p_win = 1 / 3) (h_draw : p_draw = 1 / 4) : 
  p_win + p_draw = 7 / 12 := by
  sorry

end zhang_san_not_losing_probability_l9_9765


namespace find_AC_l9_9965

def angle_A := 60
def angle_B := 45
def side_BC := 12
def sin_60 := (Real.sin (Real.pi / 3)) -- 60 degrees in radians
def sin_45 := (Real.sin (Real.pi / 4)) -- 45 degrees in radians

theorem find_AC : 
  let AC := side_BC * sin_45 / sin_60 in
  AC = 4 * Real.sqrt 6 :=
sorry

end find_AC_l9_9965


namespace opposite_of_2023_is_neg2023_l9_9353

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l9_9353


namespace probability_jack_hearts_queen_l9_9457

def standardDeck : Finset (ℕ × ℕ) := Finset.product (Finset.range 4) (Finset.range 13)
def isJack (card : ℕ × ℕ) : Prop := card.2 = 10
def isQueen (card : ℕ × ℕ) : Prop := card.2 = 11
def isHearts (card : ℕ × ℕ) : Prop := card.1 = 2

theorem probability_jack_hearts_queen :
  let draw (n : ℕ) (deck : Finset (ℕ × ℕ)) := (deck, deck.chooseSubset n)
  in let first_draw := (0, 10)
     let second_draw := (2, _)
     let third_draw := (_, 11)
     let success_cases := 
       {*
        draw 1 (standardDeck \ {(first_draw, second_draw, third_draw)}) *
        {(first_draw, second_draw, third_draw)}, let prob := success_cases.size.to_rat / 
        (@Finset.card ((ℕ × ℕ) ^ 52))
  in prob = 1 / 663 :=
  sorry

end probability_jack_hearts_queen_l9_9457


namespace simplify_expr_l9_9118

open Real

theorem simplify_expr (x : ℝ) (hx : x ≠ 0) : 
  sqrt (1 + ( (x^6 - 2) / (3 * x^3) )^2) = sqrt (x^12 + 5 * x^6 + 4) / (3 * x^3) :=
by
  sorry

end simplify_expr_l9_9118


namespace geometric_sequence_sum_l9_9824

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∃ r > 0, ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : ∀ n, a n > 0)
  (h3 : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) : 
  a 3 + a 5 = 5 := 
sorry

end geometric_sequence_sum_l9_9824


namespace opposite_of_2023_is_neg_2023_l9_9281

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l9_9281


namespace expected_sixes_in_three_rolls_l9_9869

theorem expected_sixes_in_three_rolls : 
  (∑ k in Finset.range 4, k * (Nat.choose 3 k) * (1/6)^k * (5/6)^(3-k)) = 1/2 := 
by
  sorry

end expected_sixes_in_three_rolls_l9_9869


namespace solve_inequality_l9_9573

theorem solve_inequality (a : ℝ) :
  (a > 0 → ∀ x : ℝ, (12 * x^2 - a * x - a^2 < 0 ↔ -a / 4 < x ∧ x < a / 3)) ∧
  (a = 0 → ∀ x : ℝ, ¬ (12 * x^2 - a * x - a^2 < 0)) ∧ 
  (a < 0 → ∀ x : ℝ, (12 * x^2 - a * x - a^2 < 0 ↔ a / 3 < x ∧ x < -a / 4)) :=
by
  sorry

end solve_inequality_l9_9573


namespace tan_of_sine_plus_cosine_eq_neg_4_over_3_l9_9788

variable {A : ℝ}

theorem tan_of_sine_plus_cosine_eq_neg_4_over_3 
  (h : Real.sin A + Real.cos A = -4/3) : 
  Real.tan A = -4/3 :=
sorry

end tan_of_sine_plus_cosine_eq_neg_4_over_3_l9_9788


namespace evaluate_expression_l9_9623

theorem evaluate_expression :
  500 * 997 * 0.0997 * 10^2 = 5 * (997:ℝ)^2 :=
by
  sorry

end evaluate_expression_l9_9623


namespace power_sum_greater_than_linear_l9_9065

theorem power_sum_greater_than_linear (x : ℝ) (n : ℕ) (hx1 : x > -1) (hx2 : x ≠ 0) (hn : n ≥ 2) :
  (1 + x) ^ n > 1 + n * x :=
sorry

end power_sum_greater_than_linear_l9_9065


namespace opposite_of_2023_l9_9407

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l9_9407


namespace geometric_arithmetic_seq_unique_ratio_l9_9644

variable (d : ℚ) (q : ℚ) (k : ℤ)
variable (h_d_nonzero : d ≠ 0)
variable (h_q_pos : 0 < q) (h_q_lt_one : q < 1)
variable (h_integer : 14 / (1 + q + q^2) = k)

theorem geometric_arithmetic_seq_unique_ratio :
  q = 1 / 2 :=
by
  sorry

end geometric_arithmetic_seq_unique_ratio_l9_9644


namespace quadratic_intersection_l9_9934

theorem quadratic_intersection (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 7 * x - 7 = 0) ↔ k ≥ -7/4 ∧ k ≠ 0 :=
by
  sorry

end quadratic_intersection_l9_9934


namespace opposite_of_2023_l9_9273

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9273


namespace opposite_of_2023_is_neg2023_l9_9324

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l9_9324


namespace unbroken_seashells_l9_9876

theorem unbroken_seashells (total_seashells broken_seashells unbroken_seashells : ℕ) 
  (h_total : total_seashells = 7) (h_broken : broken_seashells = 4) 
  (h_unbroken : unbroken_seashells = total_seashells - broken_seashells) : 
  unbroken_seashells = 3 :=
by 
  rw [h_total, h_broken] at h_unbroken
  exact h_unbroken

end unbroken_seashells_l9_9876


namespace problem1_problem2_problem3_l9_9654

-- Problem 1
theorem problem1
  (α : ℝ)
  (a : ℝ × ℝ := (1 / 2, - (Real.sqrt 3) / 2))
  (b : ℝ × ℝ := (Real.cos α, Real.sin α))
  (hα : 0 < α ∧ α < 2 * Real.pi / 3) :
  (a + b) • (a - b) = 0 :=
sorry

-- Problem 2
theorem problem2
  (α k : ℝ)
  (a : ℝ × ℝ := (1 / 2, - (Real.sqrt 3) / 2))
  (b : ℝ × ℝ := (Real.cos α, Real.sin α))
  (x : ℝ × ℝ := k • a + 3 • b)
  (y : ℝ × ℝ := a + (1 / k) • b)
  (hk : 0 < k)
  (hα : 0 < α ∧ α < 2 * Real.pi / 3)
  (hxy : x • y = 0) :
  k + 3 / k + 4 * Real.sin (Real.pi / 6 - α) = 0 :=
sorry

-- Problem 3
theorem problem3
  (α k : ℝ)
  (h_eq : k + 3 / k + 4 * Real.sin (Real.pi / 6 - α) = 0)
  (hα : 0 < α ∧ α < 2 * Real.pi / 3)
  (hk : 0 < k) :
  Real.pi / 2 ≤ α ∧ α < 2 * Real.pi / 3 :=
sorry

end problem1_problem2_problem3_l9_9654


namespace opposite_of_2023_l9_9420

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l9_9420


namespace evaluation_at_2_l9_9101

def f (x : ℚ) : ℚ := (2 * x^2 + 7 * x + 12) / (x^2 + 2 * x + 5)
def g (x : ℚ) : ℚ := x - 2

theorem evaluation_at_2 :
  f (g 2) + g (f 2) = 196 / 65 := by
  sorry

end evaluation_at_2_l9_9101


namespace exists_two_factorizations_in_C_another_number_with_property_l9_9799

def in_set_C (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 4 * k + 1

def is_prime_wrt_C (k : ℕ) : Prop :=
  ¬ ∃ a b : ℕ, in_set_C a ∧ in_set_C b ∧ k = a * b

theorem exists_two_factorizations_in_C : 
  ∃ (a b a' b' : ℕ), 
  in_set_C 4389 ∧ 
  in_set_C a ∧ in_set_C b ∧ in_set_C a' ∧ in_set_C b' ∧ 
  (4389 = a * b ∧ 4389 = a' * b') ∧ (a ≠ a' ∨ b ≠ b') :=
sorry

theorem another_number_with_property : 
 ∃ (n a b a' b' : ℕ), 
 n ≠ 4389 ∧ in_set_C n ∧ 
 in_set_C a ∧ in_set_C b ∧ in_set_C a' ∧ in_set_C b' ∧ 
 (n = a * b ∧ n = a' * b') ∧ (a ≠ a' ∨ b ≠ b') :=
sorry

end exists_two_factorizations_in_C_another_number_with_property_l9_9799


namespace trebled_result_of_original_number_is_72_l9_9030

theorem trebled_result_of_original_number_is_72:
  ∀ (x : ℕ), x = 9 → 3 * (2 * x + 6) = 72 :=
by
  intro x h
  sorry

end trebled_result_of_original_number_is_72_l9_9030


namespace harry_total_hours_l9_9910

variable (x h y : ℕ)

theorem harry_total_hours :
  ((h + 2 * y) = 42) → ∃ t, t = h + y :=
  by
    sorry -- Proof is omitted as per the instructions

end harry_total_hours_l9_9910


namespace number_of_assembled_desks_and_chairs_students_cannot_complete_tasks_simultaneously_l9_9096

-- Defining the conditions
def wooden_boards_type_A := 400
def wooden_boards_type_B := 500
def desk_needs_type_A := 2
def desk_needs_type_B := 1
def chair_needs_type_A := 1
def chair_needs_type_B := 2
def total_students := 30
def desk_assembly_time := 10
def chair_assembly_time := 7

-- Theorem for the number of assembled desks and chairs
theorem number_of_assembled_desks_and_chairs :
  ∃ x y : ℕ, 2 * x + y = wooden_boards_type_A ∧ x + 2 * y = wooden_boards_type_B ∧ x = 100 ∧ y = 200 :=
by {
  sorry
}

-- Theorem for the feasibility of students completing the tasks simultaneously
theorem students_cannot_complete_tasks_simultaneously :
  ¬ ∃ a : ℕ, (a ≤ total_students) ∧ (total_students - a > 0) ∧ 
  (100 / a) * desk_assembly_time = (200 / (total_students - a)) * chair_assembly_time :=
by {
  sorry
}

end number_of_assembled_desks_and_chairs_students_cannot_complete_tasks_simultaneously_l9_9096


namespace opposite_of_2023_l9_9263

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9263


namespace range_positive_of_odd_increasing_l9_9928

-- Define f as an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

-- Define f as an increasing function on (-∞,0)
def is_increasing_on_neg (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → y < 0 → f (x) < f (y)

-- Given an odd function that is increasing on (-∞,0) and f(-1) = 0, prove the range of x for which f(x) > 0 is (-1, 0) ∪ (1, +∞)
theorem range_positive_of_odd_increasing (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_increasing : is_increasing_on_neg f)
  (h_f_neg_one : f (-1) = 0) :
  {x : ℝ | f x > 0} = {x : ℝ | -1 < x ∧ x < 0} ∪ {x : ℝ | 1 < x} :=
by
  sorry

end range_positive_of_odd_increasing_l9_9928


namespace opposite_of_2023_l9_9418

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l9_9418


namespace quadratic_intersects_x_axis_l9_9941

theorem quadratic_intersects_x_axis (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 7 * x - 7 = 0) ↔ k ≥ -7/4 ∧ k ≠ 0 :=
by
  sorry

end quadratic_intersects_x_axis_l9_9941


namespace smallest_value_of_m_plus_n_l9_9582

theorem smallest_value_of_m_plus_n :
  ∃ m n : ℕ, 1 < m ∧ 
  (∃ l : ℝ, l = (m^2 - 1 : ℝ) / (m * n) ∧ l = 1 / 2021) ∧
  m + n = 85987 := 
sorry

end smallest_value_of_m_plus_n_l9_9582


namespace opposite_of_2023_l9_9166

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l9_9166


namespace part_a_max_cells_crossed_part_b_max_cells_crossed_by_needle_l9_9465

theorem part_a_max_cells_crossed (m n : ℕ) : 
  ∃ max_cells : ℕ, max_cells = m + n - 1 := sorry

theorem part_b_max_cells_crossed_by_needle : 
  ∃ max_cells : ℕ, max_cells = 285 := sorry

end part_a_max_cells_crossed_part_b_max_cells_crossed_by_needle_l9_9465


namespace sin_value_l9_9645

theorem sin_value (x : ℝ) (h : Real.sin (x + π / 3) = Real.sqrt 3 / 3) :
  Real.sin (2 * π / 3 - x) = Real.sqrt 3 / 3 :=
by
  sorry

end sin_value_l9_9645


namespace crystal_barrette_sets_l9_9689

-- Definitional and situational context
def cost_of_barrette : ℕ := 3
def cost_of_comb : ℕ := 1
def kristine_total_cost : ℕ := 4
def total_spent : ℕ := 14

-- The Lean 4 theorem statement to prove that Crystal bought 3 sets of barrettes
theorem crystal_barrette_sets (x : ℕ) 
  (kristine_cost : kristine_total_cost = cost_of_barrette + cost_of_comb + 1)
  (total_cost_eq : kristine_total_cost + (x * cost_of_barrette + cost_of_comb) = total_spent) 
  : x = 3 := 
sorry

end crystal_barrette_sets_l9_9689


namespace right_triangle_sides_l9_9000

theorem right_triangle_sides (a b c : ℕ) (h1 : a < b) 
  (h2 : 2 * c / 2 = c) 
  (h3 : exists x y, (x + y = 8 ∧ a < b) ∨ (x + y = 9 ∧ a < b)) 
  (h4 : a^2 + b^2 = c^2) : 
  a = 3 ∧ b = 4 ∧ c = 5 := 
by
  sorry

end right_triangle_sides_l9_9000


namespace sophomore_spaghetti_tortellini_ratio_l9_9543

theorem sophomore_spaghetti_tortellini_ratio
    (total_students : ℕ)
    (spaghetti_lovers : ℕ)
    (tortellini_lovers : ℕ)
    (grade_levels : ℕ)
    (spaghetti_sophomores : ℕ)
    (tortellini_sophomores : ℕ)
    (h1 : total_students = 800)
    (h2 : spaghetti_lovers = 300)
    (h3 : tortellini_lovers = 120)
    (h4 : grade_levels = 4)
    (h5 : spaghetti_sophomores = spaghetti_lovers / grade_levels)
    (h6 : tortellini_sophomores = tortellini_lovers / grade_levels) :
    (spaghetti_sophomores : ℚ) / (tortellini_sophomores : ℚ) = 5 / 2 := by
  sorry

end sophomore_spaghetti_tortellini_ratio_l9_9543


namespace parabola_intercept_sum_l9_9701

theorem parabola_intercept_sum (a b c : ℝ) : 
  (∃ y : ℝ, a = 3 * y^2 - 9 * y + 5) ∧ (∀ x : ℝ, x = 0 → b ≠ c → 3 * b^2 - 9 * b + 5 = 0 ∧ 3 * c^2 - 9 * c + 5 = 0 ∧ b + c = 3) → 
  a + b + c = 8 :=
begin
  sorry
end

end parabola_intercept_sum_l9_9701


namespace opposite_of_2023_l9_9269

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9269


namespace opposite_of_2023_l9_9433

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l9_9433


namespace opposite_of_2023_is_neg2023_l9_9310

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l9_9310


namespace speed_ratio_l9_9599

variable (vA vB : ℝ)
variable (H1 : 3 * vA = abs (-400 + 3 * vB))
variable (H2 : 10 * vA = abs (-400 + 10 * vB))

theorem speed_ratio (vA vB : ℝ) (H1 : 3 * vA = abs (-400 + 3 * vB)) (H2 : 10 * vA = abs (-400 + 10 * vB)) : 
  vA / vB = 5 / 6 :=
  sorry

end speed_ratio_l9_9599


namespace total_votes_is_240_l9_9023

-- Defining the problem conditions
variables (total_votes : ℕ) (likes : ℕ) (dislikes : ℕ)
def score : ℤ := likes - dislikes
def percentage_likes : ℚ := 3 / 4
def percentage_dislikes : ℚ := 1 / 4

-- Stating the given conditions
axiom h1 : total_votes = likes + dislikes
axiom h2 : (likes : ℤ) = (percentage_likes * total_votes)
axiom h3 : (dislikes : ℤ) = (percentage_dislikes * total_votes)
axiom h4 : score = 120

-- The statement to prove
theorem total_votes_is_240 : total_votes = 240 :=
by
  sorry

end total_votes_is_240_l9_9023


namespace solve_for_y_l9_9488

variable {b c y : Real}

theorem solve_for_y (h : b > c) (h_eq : y^2 + c^2 = (b - y)^2) : y = (b^2 - c^2) / (2 * b) := 
sorry

end solve_for_y_l9_9488


namespace opposite_of_2023_l9_9275

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9275


namespace part_a_part_b_l9_9022

-- Part (a): Prove that if 2^n - 1 divides m^2 + 9 for positive integers m and n, then n must be a power of 2.
theorem part_a (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : (2^n - 1) ∣ (m^2 + 9)) : ∃ k : ℕ, n = 2^k := 
sorry

-- Part (b): Prove that if n is a power of 2, then there exists a positive integer m such that 2^n - 1 divides m^2 + 9.
theorem part_b (n : ℕ) (hn : ∃ k : ℕ, n = 2^k) : ∃ m : ℕ, 0 < m ∧ (2^n - 1) ∣ (m^2 + 9) := 
sorry

end part_a_part_b_l9_9022


namespace rectangle_perimeters_l9_9995

theorem rectangle_perimeters (w h : ℝ) 
  (h1 : 2 * (w + h) = 20)
  (h2 : 2 * (4 * w + h) = 56) : 
  4 * (w + h) = 40 ∧ 2 * (w + 4 * h) = 44 := 
by
  sorry

end rectangle_perimeters_l9_9995


namespace coins_remainder_l9_9484

theorem coins_remainder (N : ℕ) (h1 : N % 8 = 5) (h2 : N % 7 = 2) (hN_min : ∀ M : ℕ, (M % 8 = 5 ∧ M % 7 = 2) → N ≤ M) : N % 9 = 1 :=
sorry

end coins_remainder_l9_9484


namespace opposite_of_2023_is_neg_2023_l9_9282

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l9_9282


namespace conditional_probability_l9_9514

open Finset

noncomputable def pairs := (finset.powersetLen 2 (range 8)).filter (λ s, s.card = 2)
noncomputable def eventA := pairs.filter (λ s, s.toList.map id.sum % 2 = 0)
noncomputable def eventB := pairs.filter (λ s, s.toList.all (λ x, x % 2 = 0))

noncomputable def probA := (eventA.card : ℚ) / (pairs.card : ℚ)
noncomputable def probAB := (eventB ∩ eventA).card / pairs.card

theorem conditional_probability : probAB / probA = 1 / 3 := 
by sorry

end conditional_probability_l9_9514


namespace opposite_of_2023_l9_9260

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l9_9260


namespace unit_triangle_count_bound_l9_9497

variable {L : ℝ} (L_pos : L > 0)
variable {n : ℕ}

/--
  Let \( \Delta \) be an equilateral triangle with side length \( L \), and suppose that \( n \) unit 
  equilateral triangles are drawn inside \( \Delta \) with non-overlapping interiors and each having 
  sides parallel to \( \Delta \) but with opposite orientation. Then,
  we must have \( n \leq \frac{2}{3} L^2 \).
-/
theorem unit_triangle_count_bound (L_pos : L > 0) (n : ℕ) :
  n ≤ (2 / 3) * (L ^ 2) := 
sorry

end unit_triangle_count_bound_l9_9497


namespace value_of_expression_l9_9962

theorem value_of_expression (x : ℝ) (h : 2 * x^2 + 3 * x + 7 = 8) : 4 * x^2 + 6 * x - 9 = -7 := by
  sorry

end value_of_expression_l9_9962


namespace will_remaining_money_l9_9884

theorem will_remaining_money : 
  ∀ (initial_money sweater_cost tshirt_cost shoes_cost refund_percentage remaining_money : ℕ),
  initial_money = 74 →
  sweater_cost = 9 →
  tshirt_cost = 11 →
  shoes_cost = 30 →
  refund_percentage = 90 →
  remaining_money = 51 → 
  let total_spent := sweater_cost + tshirt_cost + ((shoes_cost * (100 - refund_percentage)) / 100) 
  in remaining_money = initial_money - total_spent :=
by 
  intros initial_money sweater_cost tshirt_cost shoes_cost refund_percentage remaining_money 
         h_initial_money h_sweater_cost h_tshirt_cost h_shoes_cost h_refund_percentage h_remaining_money
  let total_spent := sweater_cost + tshirt_cost + ((shoes_cost * (100 - refund_percentage)) / 100)
  have h_total_spent : total_spent = 20 + 3 := by 
    unfold total_spent
    rw [h_sweater_cost, h_tshirt_cost, h_shoes_cost, h_refund_percentage]
    norm_num
  have h_remaining_money_computed : initial_money - total_spent = 51 := by
    rw [h_initial_money, h_total_spent]
    norm_num
  rw [h_remaining_money] at h_remaining_money_computed
  exact h_remaining_money_computed

end will_remaining_money_l9_9884


namespace intersection_complement_l9_9100

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 3}

theorem intersection_complement :
  A ∩ (U \ B) = {2} :=
by {
  sorry
}

end intersection_complement_l9_9100


namespace opposite_of_2023_l9_9336

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l9_9336


namespace opposite_of_2023_l9_9445

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l9_9445


namespace sequence_inequality_l9_9854

theorem sequence_inequality
  (n : ℕ) (h1 : 1 < n)
  (a : ℕ → ℕ)
  (h2 : ∀ i, i < n → a i < a (i + 1))
  (h3 : ∀ i, i < n - 1 → ∃ k : ℕ, (a i ^ 2 + a (i + 1) ^ 2) / 2 = k ^ 2) :
  a (n - 1) ≥ 2 * n ^ 2 - 1 :=
sorry

end sequence_inequality_l9_9854


namespace solve_equation_in_natural_numbers_l9_9021

theorem solve_equation_in_natural_numbers (x y : ℕ) :
  2 * y^2 - x * y - x^2 + 2 * y + 7 * x - 84 = 0 ↔ (x = 1 ∧ y = 6) ∨ (x = 14 ∧ y = 13) := 
sorry

end solve_equation_in_natural_numbers_l9_9021


namespace mark_sideline_time_l9_9967

def total_game_time : ℕ := 90
def initial_play : ℕ := 20
def second_play : ℕ := 35
def total_play_time : ℕ := initial_play + second_play
def sideline_time : ℕ := total_game_time - total_play_time

theorem mark_sideline_time : sideline_time = 35 := by
  sorry

end mark_sideline_time_l9_9967


namespace kneading_time_is_correct_l9_9989

def total_time := 280
def rising_time_per_session := 120
def number_of_rising_sessions := 2
def baking_time := 30

def total_rising_time := rising_time_per_session * number_of_rising_sessions
def total_non_kneading_time := total_rising_time + baking_time
def kneading_time := total_time - total_non_kneading_time

theorem kneading_time_is_correct : kneading_time = 10 := by
  have h1 : total_rising_time = 240 := by
    sorry
  have h2 : total_non_kneading_time = 270 := by
    sorry
  have h3 : kneading_time = 10 := by
    sorry
  exact h3

end kneading_time_is_correct_l9_9989


namespace intersection_count_l9_9002

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x
noncomputable def g (x : ℝ) : ℝ := x^2 - 4 * x + 5

theorem intersection_count : ∃! (x1 x2 : ℝ), 
  x1 > 0 ∧ x2 > 0 ∧ f x1 = g x1 ∧ f x2 = g x2 ∧ x1 ≠ x2 :=
sorry

end intersection_count_l9_9002


namespace complement_union_l9_9104

-- Definitions
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 3}
def B : Set ℕ := {2, 3}

-- Theorem Statement
theorem complement_union (hU: U = {0, 1, 2, 3, 4}) (hA: A = {0, 1, 3}) (hB: B = {2, 3}) :
  (U \ (A ∪ B)) = {4} :=
sorry

end complement_union_l9_9104


namespace total_pairs_sold_l9_9757

theorem total_pairs_sold
  (H S : ℕ)
  (price_soft : ℕ := 150)
  (price_hard : ℕ := 85)
  (diff_lenses : S = H + 5)
  (total_sales_eq : price_soft * S + price_hard * H = 1455) :
  H + S = 11 := by
sorry

end total_pairs_sold_l9_9757


namespace quadratic_distinct_real_roots_l9_9661

-- Definitions
def is_quadratic_eq (a b c x : ℝ) (fx : ℝ) := a * x^2 + b * x + c = fx

-- Theorem statement
theorem quadratic_distinct_real_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ is_quadratic_eq 1 (-2) m x₁ 0 ∧ is_quadratic_eq 1 (-2) m x₂ 0) → m < 1 :=
sorry -- Proof omitted

end quadratic_distinct_real_roots_l9_9661


namespace opposite_of_2023_l9_9306

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l9_9306


namespace opposite_of_2023_l9_9338

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l9_9338


namespace opposite_of_2023_l9_9297

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l9_9297


namespace find_c_d_l9_9950

theorem find_c_d (y : ℝ) (c d : ℕ) (hy : y^2 + 4*y + 4/y + 1/y^2 = 35)
  (hform : ∃ (c d : ℕ), y = c + Real.sqrt d) : c + d = 42 :=
sorry

end find_c_d_l9_9950


namespace opposite_of_2023_l9_9398

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9398


namespace problem_1_problem_2_l9_9047

variable (a : ℝ) (x : ℝ)

theorem problem_1 (h : a ≠ 1) : (a^2 / (a - 1)) - (a / (a - 1)) = a := 
sorry

theorem problem_2 (h : x ≠ -1) : (x^2 / (x + 1)) - x + 1 = 1 / (x + 1) := 
sorry

end problem_1_problem_2_l9_9047


namespace cars_already_parked_l9_9031

-- Define the levels and their parking spaces based on given conditions
def first_level_spaces : Nat := 90
def second_level_spaces : Nat := first_level_spaces + 8
def third_level_spaces : Nat := second_level_spaces + 12
def fourth_level_spaces : Nat := third_level_spaces - 9

-- Compute total spaces in the garage
def total_spaces : Nat := first_level_spaces + second_level_spaces + third_level_spaces + fourth_level_spaces

-- Define the available spaces for more cars
def available_spaces : Nat := 299

-- Prove the number of cars already parked
theorem cars_already_parked : total_spaces - available_spaces = 100 :=
by
  exact Nat.sub_eq_of_eq_add sorry -- Fill in with the actual proof step

end cars_already_parked_l9_9031


namespace dormitory_problem_l9_9746

theorem dormitory_problem (x : ℕ) :
  9 < x ∧ x < 12
  → (x = 10 ∧ 4 * x + 18 = 58)
  ∨ (x = 11 ∧ 4 * x + 18 = 62) :=
by
  intros h
  sorry

end dormitory_problem_l9_9746


namespace opposite_of_2023_l9_9211

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l9_9211


namespace florist_bouquets_is_36_l9_9028

noncomputable def florist_bouquets : Prop :=
  let r := 125
  let y := 125
  let o := 125
  let p := 125
  let rk := 45
  let yk := 61
  let ok := 30
  let pk := 40
  let initial_flowers := r + y + o + p
  let total_killed := rk + yk + ok + pk
  let remaining_flowers := initial_flowers - total_killed
  let flowers_per_bouquet := 9
  let bouquets := remaining_flowers / flowers_per_bouquet
  bouquets = 36

theorem florist_bouquets_is_36 : florist_bouquets :=
  by
    sorry

end florist_bouquets_is_36_l9_9028


namespace no_solution_for_x_l9_9090

theorem no_solution_for_x (a : ℝ) (h : a ≤ 8) : ¬ ∃ x : ℝ, |x - 5| + |x + 3| < a :=
by
  sorry

end no_solution_for_x_l9_9090


namespace opposite_of_2023_l9_9190

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9190


namespace minimum_value_proof_l9_9984

noncomputable def minimum_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 2) : ℝ :=
  (x + y) / (x * y * z)

theorem minimum_value_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 2) : 
  minimum_value x y z hx hy hz h ≥ 4 :=
sorry

end minimum_value_proof_l9_9984


namespace solve_inequality_system_l9_9574

-- Define the inequalities as conditions.
def cond1 (x : ℝ) := 2 * x + 1 < 3 * x - 2
def cond2 (x : ℝ) := 3 * (x - 2) - x ≤ 4

-- Formulate the theorem to prove that these conditions give the solution 3 < x ≤ 5.
theorem solve_inequality_system (x : ℝ) : cond1 x ∧ cond2 x ↔ 3 < x ∧ x ≤ 5 := 
sorry

end solve_inequality_system_l9_9574


namespace sum_of_numbers_l9_9558

open Function

theorem sum_of_numbers (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) 
  (h3 : b = 8) 
  (h4 : (a + b + c) / 3 = a + 7) 
  (h5 : (a + b + c) / 3 = c - 20) : 
  a + b + c = 63 := 
by 
  sorry

end sum_of_numbers_l9_9558


namespace opposite_of_2023_l9_9219

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l9_9219


namespace florist_bouquets_is_36_l9_9029

noncomputable def florist_bouquets : Prop :=
  let r := 125
  let y := 125
  let o := 125
  let p := 125
  let rk := 45
  let yk := 61
  let ok := 30
  let pk := 40
  let initial_flowers := r + y + o + p
  let total_killed := rk + yk + ok + pk
  let remaining_flowers := initial_flowers - total_killed
  let flowers_per_bouquet := 9
  let bouquets := remaining_flowers / flowers_per_bouquet
  bouquets = 36

theorem florist_bouquets_is_36 : florist_bouquets :=
  by
    sorry

end florist_bouquets_is_36_l9_9029


namespace molecular_weight_H2O_7_moles_l9_9466

noncomputable def atomic_weight_H : ℝ := 1.008
noncomputable def atomic_weight_O : ℝ := 16.00
noncomputable def num_atoms_H_in_H2O : ℝ := 2
noncomputable def num_atoms_O_in_H2O : ℝ := 1
noncomputable def moles_H2O : ℝ := 7

theorem molecular_weight_H2O_7_moles :
  (num_atoms_H_in_H2O * atomic_weight_H + num_atoms_O_in_H2O * atomic_weight_O) * moles_H2O = 126.112 := by
  sorry

end molecular_weight_H2O_7_moles_l9_9466


namespace reduction_in_consumption_l9_9504

def rate_last_month : ℝ := 16
def rate_current : ℝ := 20
def initial_consumption (X : ℝ) : ℝ := X

theorem reduction_in_consumption (X : ℝ) : initial_consumption X - (initial_consumption X * rate_last_month / rate_current) = initial_consumption X * 0.2 :=
by
  sorry

end reduction_in_consumption_l9_9504


namespace price_per_bottle_is_half_l9_9999

theorem price_per_bottle_is_half (P : ℚ) 
  (Remy_bottles_morning : ℕ) (Nick_bottles_morning : ℕ) 
  (Total_sales_evening : ℚ) (Evening_more : ℚ) : 
  Remy_bottles_morning = 55 → 
  Nick_bottles_morning = Remy_bottles_morning - 6 → 
  Total_sales_evening = 55 → 
  Evening_more = 3 → 
  104 * P + 3 = 55 → 
  P = 1 / 2 := 
by
  intros h_remy_55 h_nick_remy h_total_55 h_evening_3 h_sales_eq
  sorry

end price_per_bottle_is_half_l9_9999


namespace debby_photos_of_friends_l9_9687

theorem debby_photos_of_friends (F : ℕ) (h1 : 23 + F = 86) : F = 63 := by
  -- Proof steps will go here
  sorry

end debby_photos_of_friends_l9_9687


namespace problem_l9_9555

theorem problem (f : ℕ → ℝ) 
  (h_def : ∀ x, f x = Real.cos (x * Real.pi / 3)) 
  (h_period : ∀ x, f (x + 6) = f x) : 
  (Finset.sum (Finset.range 2018) f) = 0 := 
by
  sorry

end problem_l9_9555


namespace tetrahedron_inscribed_sphere_radius_l9_9697

theorem tetrahedron_inscribed_sphere_radius (a : ℝ) (r : ℝ) (a_pos : 0 < a) :
  (r = a * (Real.sqrt 6 + 1) / 8) ∨ 
  (r = a * (Real.sqrt 6 - 1) / 8) :=
sorry

end tetrahedron_inscribed_sphere_radius_l9_9697


namespace probability_exactly_3_positive_l9_9098

noncomputable def probability_positive : ℚ := 3 / 7
noncomputable def probability_negative : ℚ := 4 / 7

theorem probability_exactly_3_positive : 
  (Nat.choose 7 3 : ℚ) * (probability_positive^3) * (probability_negative^4) = 242112 / 823543 := by
  sorry

end probability_exactly_3_positive_l9_9098


namespace opposite_of_2023_l9_9399

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9399


namespace find_quadratic_polynomial_with_root_l9_9638

noncomputable def quadratic_polynomial (a b c : ℝ) : ℝ := 3 * a^2 - 30 * b + 87

theorem find_quadratic_polynomial_with_root (x : ℂ) (h₁ : x = 5 + 2 * complex.I) 
        (h₂ : x.conj = 5 - 2 * complex.I) : 
        quadratic_polynomial x.re x.im (3) = 3 * (x^2).re - 30 * x.re + 87  :=
by
  -- Proof goes here
  sorry

end find_quadratic_polynomial_with_root_l9_9638


namespace range_of_t_l9_9648

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def decreasing_function (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y⦄, x < y → f y < f x

variable {f : ℝ → ℝ}

theorem range_of_t (h_odd : odd_function f) 
  (h_decreasing : decreasing_function f)
  (h_ineq : ∀ t, -1 < t → t < 1 → f (1 - t) + f (1 - t^2) < 0) 
  : ∀ t, 0 < t → t < 1 :=
by sorry

end range_of_t_l9_9648


namespace space_convex_polyhedron_euler_characteristic_l9_9668

-- Definition of space convex polyhedron
structure Polyhedron where
  F : ℕ    -- number of faces
  V : ℕ    -- number of vertices
  E : ℕ    -- number of edges

-- Problem statement: Prove that for any space convex polyhedron, F + V - E = 2
theorem space_convex_polyhedron_euler_characteristic (P : Polyhedron) : P.F + P.V - P.E = 2 := by
  sorry

end space_convex_polyhedron_euler_characteristic_l9_9668


namespace parabola_intercepts_sum_l9_9702

noncomputable def a : ℝ := 5

noncomputable def b : ℝ := (9 + Real.sqrt 21) / 6

noncomputable def c : ℝ := (9 - Real.sqrt 21) / 6

theorem parabola_intercepts_sum : a + b + c = 8 := by
  -- definition of a
  have ha : a = 5 := rfl
  
  -- definitions of b and c from roots of 3y^2 - 9y + 5 = 0
  have hb : b = (9 + Real.sqrt 21) / 6 := rfl
  have hc : c = (9 - Real.sqrt 21) / 6 := rfl
  
  -- Vieta's formulas implies b + c = 3
  have hb_c : b + c = 3 := by
    calc
    b + c = (9 + Real.sqrt 21) / 6 + (9 - Real.sqrt 21) / 6 : by rw [hb, hc]
    ... = (9 + 9) / 6 : by ring
    ... = 18 / 6 : by norm_num
    ... = 3 : by norm_num
  
  -- Sum a + b + c
  calc
  a + b + c = 5 + (b + c) : by rw [ha]
  ... = 5 + 3 : by rw [hb_c]
  ... = 8 : by norm_num

end parabola_intercepts_sum_l9_9702


namespace scientific_notation_1_3_billion_l9_9564

theorem scientific_notation_1_3_billion : 1300000000 = 1.3 * 10^9 := 
sorry

end scientific_notation_1_3_billion_l9_9564


namespace find_y_in_range_l9_9536

theorem find_y_in_range (y : ℝ) (h1 : y < 0) (h2 : ⌈y⌉ * ⌊y⌋ = 72) : y ∈ Ioo (-9 : ℝ) (-8 : ℝ) :=
sorry

end find_y_in_range_l9_9536


namespace opposite_of_2023_l9_9309

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l9_9309


namespace mark_sideline_time_l9_9966

def total_game_time : ℕ := 90
def initial_play : ℕ := 20
def second_play : ℕ := 35
def total_play_time : ℕ := initial_play + second_play
def sideline_time : ℕ := total_game_time - total_play_time

theorem mark_sideline_time : sideline_time = 35 := by
  sorry

end mark_sideline_time_l9_9966


namespace opposite_of_2023_l9_9327

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l9_9327


namespace ratio_monkeys_camels_l9_9624

-- Definitions corresponding to conditions
variables (zebras camels monkeys giraffes : ℕ)
variables (multiple : ℕ)

-- Conditions
def condition1 := zebras = 12
def condition2 := camels = zebras / 2
def condition3 := monkeys = camels * multiple
def condition4 := giraffes = 2
def condition5 := monkeys = giraffes + 22

-- Question: What is the ratio of monkeys to camels? Prove it is 4:1 given the conditions.
theorem ratio_monkeys_camels (zebras camels monkeys giraffes multiple : ℕ) 
  (h1 : condition1 zebras) 
  (h2 : condition2 zebras camels)
  (h3 : condition3 camels monkeys multiple)
  (h4 : condition4 giraffes)
  (h5 : condition5 monkeys giraffes) :
  multiple = 4 :=
sorry

end ratio_monkeys_camels_l9_9624


namespace total_road_length_l9_9570

theorem total_road_length (L : ℚ) : (1/3) * L + (2/5) * (2/3) * L = 135 → L = 225 := 
by
  intro h
  sorry

end total_road_length_l9_9570


namespace number_of_boys_l9_9026

theorem number_of_boys (n : ℕ) (handshakes : ℕ) (h_handshakes : handshakes = n * (n - 1) / 2) (h_total : handshakes = 55) : n = 11 := by
  sorry

end number_of_boys_l9_9026


namespace compute_n_pow_m_l9_9532

-- Given conditions
variables (n m : ℕ)
axiom n_eq : n = 3
axiom n_plus_one_eq_2m : n + 1 = 2 * m

-- Goal: Prove n^m = 9
theorem compute_n_pow_m : n^m = 9 :=
by {
  -- Proof goes here
  sorry
}

end compute_n_pow_m_l9_9532


namespace opposite_of_2023_l9_9221

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l9_9221


namespace opposite_of_2023_is_neg_2023_l9_9288

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l9_9288


namespace colorful_family_total_children_l9_9093

theorem colorful_family_total_children (x : ℕ) (b : ℕ) :
  -- Initial equal number of white, blue, and striped children
  -- After some blue children become striped
  -- Total number of blue and white children was 10,
  -- Total number of white and striped children was 18
  -- We need to prove the total number of children is 21
  (x = 5) →
  (x + x = 10) →
  (10 + b = 18) →
  (3*x = 21) :=
by
  intros h1 h2 h3
  -- x initially represents the number of white, blue, and striped children
  -- We know x is 5 and satisfy the conditions
  sorry

end colorful_family_total_children_l9_9093


namespace opposite_of_2023_l9_9326

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l9_9326


namespace problem_I_l9_9795

def f (x : ℝ) : ℝ := 4 - |x| - |x - 3|

theorem problem_I {x : ℝ} : f (x + 3 / 2) ≥ 0 ↔ -2 ≤ x ∧ x ≤ 2 :=
by
  sorry

end problem_I_l9_9795


namespace opposite_of_2023_is_neg_2023_l9_9279

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l9_9279


namespace chess_club_mixed_groups_l9_9719

theorem chess_club_mixed_groups (total_children : ℕ) (num_groups : ℕ)
  (games_per_group : ℕ) (boy_vs_boy_games : ℕ) (girl_vs_girl_games : ℕ) :
  total_children = 90 ∧ num_groups = 30 ∧ games_per_group = 3 ∧ boy_vs_boy_games = 30 ∧ girl_vs_girl_games = 14 →
  by let mixed_games := 90 - 30 - 14 in let mixed_groups := mixed_games / 2 in mixed_groups = 23 :=
begin
  intros h,
  cases h with h1 h234, cases h234 with h2 h34, cases h34 with h3 h4,
  simp at h1 h2 h3 h4,
  have h5 : 46 = 90 - 30 - 14 := by norm_num,
  have h6 : 2 * 23 = 46 := by norm_num,
  exact h6
end

end chess_club_mixed_groups_l9_9719


namespace S15_constant_l9_9786

-- Define the arithmetic sequence a_n
noncomputable def a (n : ℕ) (a1 d : ℤ) : ℤ := a1 + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
noncomputable def S (n : ℕ) (a1 d : ℤ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

-- Given condition: a_5 + a_8 + a_11 is constant
axiom const_sum : ∀ (a1 d : ℤ), a 5 a1 d + a 8 a1 d + a 11 a1 d = 3 * a1 + 21 * d

-- The equivalent proof problem
theorem S15_constant (a1 d : ℤ) : S 15 a1 d = 5 * (3 * a1 + 21 * d) :=
by
  sorry

end S15_constant_l9_9786


namespace jellybean_ratio_l9_9109

-- Define the conditions
def Matilda_jellybeans := 420
def Steve_jellybeans := 84
def Matt_jellybeans := 10 * Steve_jellybeans

-- State the theorem to prove the ratio
theorem jellybean_ratio : (Matilda_jellybeans : Nat) / (Matt_jellybeans : Nat) = 1 / 2 :=
by
  sorry

end jellybean_ratio_l9_9109


namespace f_properties_l9_9517

noncomputable def f : ℝ → ℝ := sorry -- we define f as a noncomputable function for generality 

-- Given conditions as Lean hypotheses
axiom functional_eq : ∀ x y : ℝ, f x + f y = 2 * f ((x + y) / 2) * f ((x - y) / 2)
axiom not_always_zero : ¬(∀ x : ℝ, f x = 0)

-- The statement we need to prove
theorem f_properties : f 0 = 1 ∧ (∀ x : ℝ, f (-x) = f x) := 
  by 
    sorry

end f_properties_l9_9517


namespace annual_interest_rate_is_approx_14_87_percent_l9_9085

-- Let P be the principal amount, r the annual interest rate, and n the number of years
-- Given: A = P(1 + r)^n, where A is the amount of money after n years
-- In this problem: A = 2P, n = 5

theorem annual_interest_rate_is_approx_14_87_percent
    (P : Real) (r : Real) (n : Real) (A : Real) (condition1 : n = 5)
    (condition2 : A = 2 * P)
    (condition3 : A = P * (1 + r)^n) :
  r = 2^(1/5) - 1 := 
  sorry

end annual_interest_rate_is_approx_14_87_percent_l9_9085


namespace opposite_of_2023_l9_9155

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l9_9155


namespace expected_number_of_sixes_l9_9872

-- Define the problem context and conditions
def die_prob := (1 : ℝ) / 6

def expected_six (n : ℕ) : ℝ :=
  n * die_prob

-- The main proposition to prove
theorem expected_number_of_sixes (n : ℕ) (hn : n = 3) : expected_six n = 1 / 2 :=
by
  rw [hn]
  have fact1 : (3 : ℝ) * die_prob = 3 / 6 := by norm_cast; norm_num
  rw [fact1]
  norm_num

-- We add sorry to indicate incomplete proof, fulfilling criteria 4
sorry

end expected_number_of_sixes_l9_9872


namespace opposite_of_2023_l9_9374

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9374


namespace opposite_of_2023_l9_9177

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l9_9177


namespace minimum_value_y_l9_9060

theorem minimum_value_y (x : ℝ) (hx : x > 2) : 
  ∃ y, y = x + 4 / (x - 2) ∧ ∀ z, (z = x + 4 / (x - 2) → z ≥ 6) :=
by
  sorry

end minimum_value_y_l9_9060


namespace opposite_of_2023_l9_9150

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l9_9150


namespace opposite_of_2023_l9_9256

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l9_9256


namespace opposite_of_2023_l9_9379

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9379


namespace consecutive_numbers_product_l9_9017

theorem consecutive_numbers_product (a b c d : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : c + 1 = d) (h4 : a + d = 109) :
  b * c = 2970 :=
by {
  -- Proof goes here
  sorry
}

end consecutive_numbers_product_l9_9017


namespace opposite_of_2023_l9_9255

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l9_9255


namespace opposite_of_2023_l9_9369

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l9_9369


namespace opposite_of_2023_is_neg2023_l9_9352

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l9_9352


namespace mass_percentage_O_mixture_l9_9915

noncomputable def molar_mass_Al2O3 : ℝ := (2 * 26.98) + (3 * 16.00)
noncomputable def molar_mass_Cr2O3 : ℝ := (2 * 51.99) + (3 * 16.00)
noncomputable def mass_of_O_in_Al2O3 : ℝ := 3 * 16.00
noncomputable def mass_of_O_in_Cr2O3 : ℝ := 3 * 16.00
noncomputable def mass_percentage_O_in_Al2O3 : ℝ := (mass_of_O_in_Al2O3 / molar_mass_Al2O3) * 100
noncomputable def mass_percentage_O_in_Cr2O3 : ℝ := (mass_of_O_in_Cr2O3 / molar_mass_Cr2O3) * 100
noncomputable def mass_percentage_O_in_mixture : ℝ := (0.50 * mass_percentage_O_in_Al2O3) + (0.50 * mass_percentage_O_in_Cr2O3)

theorem mass_percentage_O_mixture : mass_percentage_O_in_mixture = 39.325 := by
  sorry

end mass_percentage_O_mixture_l9_9915


namespace opposite_of_2023_l9_9438

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l9_9438


namespace union_complement_l9_9944

open Set

def U : Set ℤ := {x | -3 < x ∧ x < 3}

def A : Set ℤ := {1, 2}

def B : Set ℤ := {-2, -1, 2}

theorem union_complement :
  A ∪ (U \ B) = {0, 1, 2} := by
  sorry

end union_complement_l9_9944


namespace speed_difference_l9_9675

theorem speed_difference (distance : ℝ) (time_heavy : ℝ) (time_no_traffic : ℝ) (d : distance = 200) (th : time_heavy = 5) (tn : time_no_traffic = 4) :
  (distance / time_no_traffic) - (distance / time_heavy) = 10 :=
by
  -- Proof goes here
  sorry

end speed_difference_l9_9675


namespace intersection_of_A_and_B_l9_9520

open Finset

def A : Finset ℤ := {-2, -1, 0, 1, 2}
def B : Finset ℤ := {1, 2, 3}

theorem intersection_of_A_and_B :
  A ∩ B = {1, 2} :=
by sorry

end intersection_of_A_and_B_l9_9520


namespace sum_is_composite_l9_9678

theorem sum_is_composite (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a * b = c * d) : 
  ∃ x y : ℕ, 1 < x ∧ 1 < y ∧ x * y = a + b + c + d :=
sorry

end sum_is_composite_l9_9678


namespace opposite_of_2023_is_neg2023_l9_9355

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l9_9355


namespace more_radishes_correct_l9_9902

def total_radishes : ℕ := 88
def radishes_first_basket : ℕ := 37

def more_radishes_in_second_basket := total_radishes - radishes_first_basket - radishes_first_basket

theorem more_radishes_correct : more_radishes_in_second_basket = 14 :=
by
  sorry

end more_radishes_correct_l9_9902


namespace opposite_of_2023_l9_9371

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l9_9371


namespace digit_x_base_7_l9_9849

theorem digit_x_base_7 (x : ℕ) : 
    (4 * 7^3 + 5 * 7^2 + x * 7 + 2) % 9 = 0 → x = 4 := 
by {
    sorry
}

end digit_x_base_7_l9_9849


namespace opposite_of_2023_is_neg_2023_l9_9236

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l9_9236


namespace isosceles_triangle_exists_l9_9927

-- Definitions for a triangle vertex and side lengths
structure Triangle :=
  (A B C : ℝ × ℝ) -- Vertices A, B, C
  (AB AC BC : ℝ)  -- Sides AB, AC, BC

-- Definition for all sides being less than 1 unit
def sides_less_than_one (T : Triangle) : Prop :=
  T.AB < 1 ∧ T.AC < 1 ∧ T.BC < 1

-- Definition for isosceles triangle containing the original one
def exists_isosceles_containing (T : Triangle) : Prop :=
  ∃ (T' : Triangle), 
    (T'.AB = T'.AC ∨ T'.AB = T'.BC ∨ T'.AC = T'.BC) ∧
    T'.A = T.A ∧ -- T'.A vertex is same as T.A
    (T'.AB < 1 ∧ T'.AC < 1 ∧ T'.BC < 1) ∧
    (∃ (B1 : ℝ × ℝ), -- There exists point B1 such that new triangle T' incorporates B1
      T'.B = B1 ∧
      T'.C = T.C) -- T' also has vertex C of original triangle

-- Complete theorem statement
theorem isosceles_triangle_exists (T : Triangle) (hT : sides_less_than_one T) : exists_isosceles_containing T :=
by 
  sorry

end isosceles_triangle_exists_l9_9927


namespace remainder_of_product_divided_by_7_l9_9511

theorem remainder_of_product_divided_by_7 :
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 2 :=
by
  sorry

end remainder_of_product_divided_by_7_l9_9511


namespace triangle_largest_angle_l9_9943

theorem triangle_largest_angle {k : ℝ} (h1 : k > 0)
  (h2 : k + 2 * k + 3 * k = 180) : 3 * k = 90 := 
sorry

end triangle_largest_angle_l9_9943


namespace sum_of_interior_ninth_row_l9_9672

theorem sum_of_interior_ninth_row : 
  Sum of the interior numbers of the fourth row is 6 ∧
  Sum of the interior numbers of the fifth row is 14 →
  Sum of the interior numbers of the ninth row = 254 := 
by 
  -- Assuming the conditions hold, we will prove the conclusion.
  sorry

end sum_of_interior_ninth_row_l9_9672


namespace opposite_of_2023_l9_9375

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9375


namespace total_parcel_boxes_l9_9578

theorem total_parcel_boxes (a b c d : ℕ) (row_boxes column_boxes total_boxes : ℕ)
  (h_left : a = 7) (h_right : b = 13)
  (h_front : c = 8) (h_back : d = 14)
  (h_row : row_boxes = a - 1 + 1 + b) -- boxes in a row: (a - 1) + 1 (parcel itself) + b
  (h_column : column_boxes = c - 1 + 1 + d) -- boxes in a column: (c -1) + 1(parcel itself) + d
  (h_total : total_boxes = row_boxes * column_boxes) :
  total_boxes = 399 := by
  sorry

end total_parcel_boxes_l9_9578


namespace opposite_of_2023_is_neg_2023_l9_9286

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l9_9286


namespace kristin_annual_income_l9_9716

theorem kristin_annual_income (p : ℝ) :
  ∃ A : ℝ, 
  (0.01 * p * 28000 + 0.01 * (p + 2) * (A - 28000) = (0.01 * (p + 0.25) * A)) ∧
  A = 32000 :=
by
  sorry

end kristin_annual_income_l9_9716


namespace opposite_of_2023_l9_9388

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9388


namespace opposite_of_2023_l9_9331

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l9_9331


namespace quiz_competition_l9_9751

theorem quiz_competition (x : ℕ) :
  (10 * x - 4 * (20 - x) ≥ 88) ↔ (x ≥ 12) :=
by 
  sorry

end quiz_competition_l9_9751


namespace range_of_y_l9_9084

theorem range_of_y (y : ℝ) (hy : y < 0) (h : ⌈y⌉ * ⌊y⌋ = 132) : -12 < y ∧ y < -11 := 
by 
  sorry

end range_of_y_l9_9084


namespace opposite_of_2023_l9_9447

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l9_9447


namespace probability_red_or_blue_l9_9596

theorem probability_red_or_blue
  (total_marbles : ℕ)
  (p_white : ℚ)
  (p_green : ℚ)
  (p_red_or_blue : ℚ) :
  total_marbles = 84 →
  p_white = 1 / 4 →
  p_green = 2 / 7 →
  p_red_or_blue = 1 - p_white - p_green →
  p_red_or_blue = 13 / 28 :=
by
  intros h_total h_white h_green h_red_or_blue
  sorry

end probability_red_or_blue_l9_9596


namespace odd_function_iff_l9_9001

def f (x a b : ℝ) : ℝ := x * abs (x + a) + b

theorem odd_function_iff (a b : ℝ) : 
  (∀ x, f x a b = -f (-x) a b) ↔ (a ^ 2 + b ^ 2 = 0) :=
by
  sorry

end odd_function_iff_l9_9001


namespace add_fractions_l9_9632

theorem add_fractions : (7 / 12) + (3 / 8) = 23 / 24 := by
  sorry

end add_fractions_l9_9632


namespace opposite_of_2023_l9_9209

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l9_9209


namespace opposite_of_2023_l9_9397

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9397


namespace intercepts_sum_eq_eight_l9_9699

theorem intercepts_sum_eq_eight :
  let parabola_eq := λ y : ℝ, 3 * y^2 - 9 * y + 5
  let a := parabola_eq 0
  let roots := (λ a b c : ℝ, (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a), (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a))
  let b := roots 3 (-9) 5).1
  let c := roots 3 (-9) 5).2
  a + b + c = 8 :=
by
  let parabola_eq := λ y : ℝ, 3 * y^2 - 9 * y + 5
  let a := parabola_eq 0
  let roots := (λ a b c : ℝ, (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a), (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a))
  let b := roots 3 (-9) 5).1
  let c := roots 3 (-9) 5).2
  -- Proof will go here
  sorry

end intercepts_sum_eq_eight_l9_9699


namespace opposite_of_2023_l9_9217

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l9_9217


namespace natalie_bushes_to_zucchinis_l9_9772

/-- Each of Natalie's blueberry bushes yields ten containers of blueberries,
    and she trades six containers of blueberries for three zucchinis.
    Given this setup, prove that the number of bushes Natalie needs to pick
    in order to get sixty zucchinis is twelve. --/
theorem natalie_bushes_to_zucchinis :
  (∀ (bush_yield containers_needed : ℕ), bush_yield = 10 ∧ containers_needed = 60 * (6 / 3)) →
  (∀ (containers_total bushes_needed : ℕ), containers_total = 60 * (6 / 3) ∧ bushes_needed = containers_total * (1 / bush_yield)) →
  bushes_needed = 12 :=
by
  sorry

end natalie_bushes_to_zucchinis_l9_9772


namespace average_rate_of_interest_l9_9493

def invested_amount_total : ℝ := 5000
def rate1 : ℝ := 0.03
def rate2 : ℝ := 0.05
def annual_return (amount : ℝ) (rate : ℝ) : ℝ := amount * rate

theorem average_rate_of_interest : 
  (∃ (x : ℝ), x > 0 ∧ x < invested_amount_total ∧ 
    annual_return (invested_amount_total - x) rate1 = annual_return x rate2) → 
  ((annual_return (invested_amount_total - 1875) rate1 + annual_return 1875 rate2) / invested_amount_total = 0.0375) := 
by
  sorry

end average_rate_of_interest_l9_9493


namespace event_A_muffins_correct_event_B_muffins_correct_event_C_muffins_correct_l9_9760

-- Event A
def total_muffins_needed_A := 200
def arthur_muffins_A := 35
def beatrice_muffins_A := 48
def charles_muffins_A := 29
def total_muffins_baked_A := arthur_muffins_A + beatrice_muffins_A + charles_muffins_A
def additional_muffins_needed_A := total_muffins_needed_A - total_muffins_baked_A

-- Event B
def total_muffins_needed_B := 150
def arthur_muffins_B := 20
def beatrice_muffins_B := 35
def charles_muffins_B := 25
def total_muffins_baked_B := arthur_muffins_B + beatrice_muffins_B + charles_muffins_B
def additional_muffins_needed_B := total_muffins_needed_B - total_muffins_baked_B

-- Event C
def total_muffins_needed_C := 250
def arthur_muffins_C := 45
def beatrice_muffins_C := 60
def charles_muffins_C := 30
def total_muffins_baked_C := arthur_muffins_C + beatrice_muffins_C + charles_muffins_C
def additional_muffins_needed_C := total_muffins_needed_C - total_muffins_baked_C

-- Proof Statements
theorem event_A_muffins_correct : additional_muffins_needed_A = 88 := by
  sorry

theorem event_B_muffins_correct : additional_muffins_needed_B = 70 := by
  sorry

theorem event_C_muffins_correct : additional_muffins_needed_C = 115 := by
  sorry

end event_A_muffins_correct_event_B_muffins_correct_event_C_muffins_correct_l9_9760


namespace smallest_m_n_sum_l9_9580

noncomputable def smallestPossibleSum (m n : ℕ) : ℕ :=
  m + n

theorem smallest_m_n_sum :
  ∃ (m n : ℕ), (m > 1) ∧ (m * n * (2021 * (m^2 - 1)) = 2021 * m * m * n) ∧ smallestPossibleSum m n = 4323 :=
by
  sorry

end smallest_m_n_sum_l9_9580


namespace vertical_strips_count_l9_9827

/- Define the conditions -/

variables {a b x y : ℕ}

-- The outer rectangle has a perimeter of 50 cells
axiom outer_perimeter : 2 * a + 2 * b = 50

-- The inner hole has a perimeter of 32 cells
axiom inner_perimeter : 2 * x + 2 * y = 32

-- Cutting along all horizontal lines produces 20 strips
axiom horizontal_cuts : a + x = 20

-- We want to prove that cutting along all vertical grid lines produces 21 strips
theorem vertical_strips_count : b + y = 21 :=
by
  sorry

end vertical_strips_count_l9_9827


namespace opposite_of_2023_l9_9187

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9187


namespace thabo_book_ratio_l9_9846

theorem thabo_book_ratio :
  ∃ (P_f P_nf H_nf : ℕ), H_nf = 35 ∧ P_nf = H_nf + 20 ∧ P_f + P_nf + H_nf = 200 ∧ P_f / P_nf = 2 :=
by
  sorry

end thabo_book_ratio_l9_9846


namespace number_of_mixed_groups_l9_9723

theorem number_of_mixed_groups (n_children n_groups n_games boy_vs_boy girl_vs_girl mixed_games : ℕ) (h_children : n_children = 90) (h_groups : n_groups = 30) (h_games_per_group : n_games = 3) (h_boy_vs_boy : boy_vs_boy = 30) (h_girl_vs_girl : girl_vs_girl = 14) (h_total_games : mixed_games = 46) :
  (∀ g : ℕ, g * 2 = mixed_games → g = 23) :=
by
  intros g hg
  sorry

end number_of_mixed_groups_l9_9723


namespace geese_left_park_l9_9618

noncomputable def initial_ducks : ℕ := 25
noncomputable def initial_geese (ducks : ℕ) : ℕ := 2 * ducks - 10
noncomputable def final_ducks (ducks_added : ℕ) (ducks : ℕ) : ℕ := ducks + ducks_added
noncomputable def geese_after_leaving (geese_before : ℕ) (geese_left : ℕ) : ℕ := geese_before - geese_left

theorem geese_left_park
    (ducks : ℕ)
    (ducks_added : ℕ)
    (initial_geese : ℕ := 2 * ducks - 10)
    (final_ducks : ℕ := ducks + ducks_added)
    (geese_left : ℕ)
    (geese_remaining : ℕ := initial_geese - geese_left) :
    geese_remaining = final_ducks + 1 → geese_left = 10 := by
  sorry

end geese_left_park_l9_9618


namespace hockey_stick_identity_sum_find_binomial_coefficient_l9_9481

-- Declare binomial coefficients in Lean
open Nat

-- Step 1: Prove that the sum of the binomial coefficients equals 462
theorem hockey_stick_identity_sum :
    (Nat.choose 5 0) + (Nat.choose 6 5) + (Nat.choose 7 5) +
    (Nat.choose 8 5) + (Nat.choose 9 5) + (Nat.choose 10 5) =
    Nat.choose 11 6 := by
    sorry

-- Step 2: Prove that given the relation, the value of the binomial coefficient
theorem find_binomial_coefficient (m : ℕ) (h : 1 / (Nat.choose 5 m) - 1 / (Nat.choose 6 m) = 7 / (10 * Nat.choose 7 m)) :
    Nat.choose 8 m = 28 := by
    sorry

end hockey_stick_identity_sum_find_binomial_coefficient_l9_9481


namespace opposite_of_2023_l9_9246

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l9_9246


namespace opposite_of_2023_l9_9193

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9193


namespace opposite_of_2023_l9_9339

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l9_9339


namespace sparrow_pecks_seeds_l9_9563

theorem sparrow_pecks_seeds (x : ℕ) (h1 : 9 * x < 1001) (h2 : 10 * x > 1100) : x = 111 :=
by
  sorry

end sparrow_pecks_seeds_l9_9563


namespace stream_speed_is_one_l9_9475

noncomputable def speed_of_stream (downstream_speed upstream_speed : ℝ) : ℝ :=
  (downstream_speed - upstream_speed) / 2

theorem stream_speed_is_one : speed_of_stream 10 8 = 1 := by
  sorry

end stream_speed_is_one_l9_9475


namespace second_field_area_percent_greater_l9_9855

theorem second_field_area_percent_greater (r1 r2 : ℝ) (h : r1 / r2 = 2 / 5) : 
  (π * (r2^2) - π * (r1^2)) / (π * (r1^2)) * 100 = 525 := 
by
  sorry

end second_field_area_percent_greater_l9_9855


namespace expected_number_of_sixes_when_three_dice_are_rolled_l9_9867

theorem expected_number_of_sixes_when_three_dice_are_rolled : 
  ∑ n in finset.range 4, (n * (↑(finset.filter (λ xs : fin 3 → fin 6, xs.count (λ x, x = 5) = n) finset.univ).card / 216 : ℚ)) = 1 / 2 :=
by
  -- Conclusion of proof is omitted as per instructions
  sorry

end expected_number_of_sixes_when_three_dice_are_rolled_l9_9867


namespace convex_polyhedron_property_l9_9745

-- Given conditions as definitions
def num_faces : ℕ := 40
def num_hexagons : ℕ := 8
def num_triangles_eq_twice_pentagons (P : ℕ) (T : ℕ) : Prop := T = 2 * P
def num_pentagons_eq_twice_hexagons (P : ℕ) (H : ℕ) : Prop := P = 2 * H

-- Main statement for the proof problem
theorem convex_polyhedron_property (P T V : ℕ) :
  num_triangles_eq_twice_pentagons P T ∧ num_pentagons_eq_twice_hexagons P num_hexagons ∧ 
  num_faces = T + P + num_hexagons ∧ V = (T * 3 + P * 5 + num_hexagons * 6) / 2 + num_faces - 2 →
  100 * P + 10 * T + V = 535 :=
by
  sorry

end convex_polyhedron_property_l9_9745


namespace opposite_of_2023_l9_9169

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l9_9169


namespace opposite_of_2023_l9_9164

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l9_9164


namespace problem_a_problem_d_l9_9805

theorem problem_a (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) : (1 / (a * b)) ≥ 1 / 4 :=
by
  sorry

theorem problem_d (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) : a^2 + b^2 ≥ 8 :=
by
  sorry

end problem_a_problem_d_l9_9805


namespace opposite_of_2023_is_neg_2023_l9_9237

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l9_9237


namespace find_four_numbers_l9_9729

theorem find_four_numbers (a b c d : ℕ) (h1 : b^2 = a * c) (h2 : a * b * c = 216) (h3 : 2 * c = b + d) (h4 : b + c + d = 12) :
  a = 9 ∧ b = 6 ∧ c = 4 ∧ d = 2 :=
sorry

end find_four_numbers_l9_9729


namespace polar_to_rect_l9_9627

theorem polar_to_rect (r θ : ℝ) (hr : r = 5) (hθ : θ = 5 * Real.pi / 3) :
  (r * Real.cos θ, r * Real.sin θ) = (2.5, 5 * Real.sqrt 3 / 2) :=
by
  rw [hr, hθ]
  sorry

end polar_to_rect_l9_9627


namespace opposite_of_2023_l9_9222

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l9_9222


namespace quadratic_zeros_l9_9589

theorem quadratic_zeros : ∀ x : ℝ, (x = 3 ∨ x = -1) ↔ (x^2 - 2*x - 3 = 0) := by
  intro x
  sorry

end quadratic_zeros_l9_9589


namespace opposite_of_2023_is_neg_2023_l9_9293

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l9_9293


namespace opposite_of_2023_l9_9147

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l9_9147


namespace max_forced_terms_in_1000_seq_l9_9487

def is_reg_seq (a : ℕ → ℤ) (n : ℕ) : Prop :=
  ∃ x : ℝ, ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → a k = int.floor (k * x)

def is_forced_term (a : ℕ → ℤ) (n : ℕ) (k : ℕ) : Prop :=
  ∀ b : ℤ, (∃ x : ℝ, ∀ i : ℕ, 1 ≤ i ∧ i < k → a i = int.floor (i * x)) ↔ b = a k

theorem max_forced_terms_in_1000_seq : 
  ∀ a : ℕ → ℤ, is_reg_seq a 1000 → 
  ∃ forced : ℕ → Prop, 
    (∀ k, (1 ≤ k ∧ k ≤ 1000) → (forced k ↔ is_forced_term a 1000 k)) 
    ∧ (finset.card (finset.filter forced (finset.range 1000.succ)) = 985) :=
by
  sorry

end max_forced_terms_in_1000_seq_l9_9487


namespace opposite_of_2023_l9_9382

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9382


namespace opposite_of_2023_l9_9259

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l9_9259


namespace rectangle_perimeter_eq_26_l9_9878

theorem rectangle_perimeter_eq_26 (a b c W : ℕ) (h_tri : a = 5 ∧ b = 12 ∧ c = 13)
  (h_right_tri : a^2 + b^2 = c^2) (h_W : W = 3) (h_area_eq : 1/2 * (a * b) = (W * L))
  (A L : ℕ) (hA : A = 30) (hL : L = A / W) :
  2 * (L + W) = 26 :=
by
  sorry

end rectangle_perimeter_eq_26_l9_9878


namespace prod_eq_one_l9_9521

noncomputable def is_parity_equal (A : Finset ℝ) (a : ℝ) : Prop :=
  (A.filter (fun x => x > a)).card % 2 = (A.filter (fun x => x < 1/a)).card % 2

theorem prod_eq_one
  (A : Finset ℝ)
  (hA : ∀ (a : ℝ), 0 < a → is_parity_equal A a)
  (hA_pos : ∀ x ∈ A, 0 < x) :
  A.prod id = 1 :=
sorry

end prod_eq_one_l9_9521


namespace reciprocal_expression_l9_9046

theorem reciprocal_expression :
  (1 / ((1 / 4 : ℚ) + (1 / 5 : ℚ)) / (1 / 3)) = (20 / 27 : ℚ) :=
by
  sorry

end reciprocal_expression_l9_9046


namespace triangle_equilateral_l9_9540

noncomputable def is_equilateral (a b c : ℝ) (A B C : ℝ) : Prop :=
  A = 60 ∧ B = 60 ∧ C = 60 ∧ a = b ∧ b = c

theorem triangle_equilateral (A B C a b c : ℝ) (hB : B = 60) (hb : b^2 = a * c) :
  is_equilateral a b c A B C :=
by
  sorry

end triangle_equilateral_l9_9540


namespace crescent_moon_falcata_area_l9_9587

/-
Prove that the area of the crescent moon falcata, which is bounded by:
1. A portion of the circle with radius 4 centered at (0,0) in the second quadrant.
2. A portion of the circle with radius 2 centered at (0,2) in the second quadrant.
3. The line segment from (0,0) to (-4,0).
is equal to 6π.
-/
theorem crescent_moon_falcata_area :
  let radius_large := 4
  let radius_small := 2
  let area_large := (1 / 2) * (π * (radius_large ^ 2))
  let area_small := (1 / 2) * (π * (radius_small ^ 2))
  (area_large - area_small) = 6 * π := by
  sorry

end crescent_moon_falcata_area_l9_9587


namespace no_int_solutions_for_equation_l9_9655

theorem no_int_solutions_for_equation : 
  ∀ x y : ℤ, x ^ 2022 + y^2 = 2 * y + 2 → false := 
by
  -- By the given steps in the solution, we can conclude that no integer solutions exist
  sorry

end no_int_solutions_for_equation_l9_9655


namespace smallest_n_for_sum_condition_l9_9629

theorem smallest_n_for_sum_condition :
  ∃ n, n ≥ 4 ∧ (∀ S : Finset ℤ, S.card = n → ∃ (a b c d : ℤ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ (a + b - c - d) % 20 = 0) ∧ n = 9 :=
by
  sorry

end smallest_n_for_sum_condition_l9_9629


namespace money_left_after_bike_purchase_l9_9974

-- Definitions based on conditions
def jars : ℕ := 5
def quarters_per_jar : ℕ := 160
def quarter_value : ℝ := 0.25
def bike_cost : ℝ := 180

-- The theorem statement
theorem money_left_after_bike_purchase : (jars * quarters_per_jar * quarter_value) - bike_cost = 20 := by
  sorry

end money_left_after_bike_purchase_l9_9974


namespace opposite_of_2023_l9_9395

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9395


namespace real_solutions_count_l9_9530

noncomputable def number_of_real_solutions : ℕ := 2

theorem real_solutions_count (x : ℝ) :
  (x^2 - 5)^2 = 36 → number_of_real_solutions = 2 := by
  sorry

end real_solutions_count_l9_9530


namespace algebraic_expression_value_l9_9960

theorem algebraic_expression_value (x : ℝ) (h : 2 * x^2 + 3 * x + 7 = 8) : 4 * x^2 + 6 * x - 9 = -7 :=
by
  sorry

end algebraic_expression_value_l9_9960


namespace possible_ages_that_sum_to_a_perfect_square_l9_9602

def two_digit_number (a b : ℕ) := 10 * a + b
def reversed_number (a b : ℕ) := 10 * b + a

def sum_of_number_and_its_reversed (a b : ℕ) : ℕ := 
  two_digit_number a b + reversed_number a b

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem possible_ages_that_sum_to_a_perfect_square :
  ∃ (s : Finset ℕ), s.card = 6 ∧ 
  ∀ x ∈ s, ∃ a b : ℕ, a + b = 11 ∧ s = {two_digit_number a b} ∧ is_perfect_square (sum_of_number_and_its_reversed a b) :=
  sorry

end possible_ages_that_sum_to_a_perfect_square_l9_9602


namespace solve_complex_addition_l9_9014

noncomputable def complex_addition : Prop :=
  let i := Complex.I
  let z1 := 3 - 5 * i
  let z2 := -1 + 12 * i
  let result := 2 + 7 * i
  z1 + z2 = result

theorem solve_complex_addition :
  complex_addition :=
by
  sorry

end solve_complex_addition_l9_9014


namespace smallest_range_between_allocations_l9_9038

-- Problem statement in Lean
theorem smallest_range_between_allocations :
  ∀ (A B C D E : ℕ), 
  (A = 30000) →
  (B < 18000 ∨ B > 42000) →
  (C < 18000 ∨ C > 42000) →
  (D < 58802 ∨ D > 82323) →
  (E < 58802 ∨ E > 82323) →
  min B (min C (min D E)) = 17999 →
  max B (max C (max D E)) = 82323 →
  82323 - 17999 = 64324 :=
by
  intros A B C D E hA hB hC hD hE hmin hmax
  sorry

end smallest_range_between_allocations_l9_9038


namespace factoring_options_count_l9_9822

theorem factoring_options_count :
  (∃ (n : ℕ), n = 14) ↔
  ∃ a b c : ℕ, a + b + c = 10 ∧ a ≥ b ∧ b ≥ c ∧ fintype.card {p : ℕ × ℕ // p.1 + p.2 = 10 ∧ p.1 ≥ p.2} + 
    fintype.card {p : ℕ × ℕ // p.1 + p.2 = 9 ∧ p.1 ≥ p.2} + 
    fintype.card {p : ℕ × ℕ // p.1 + p.2 = 8 ∧ p.1 ≥ p.2} + 
    fintype.card {p : ℕ × ℕ // p.1 + p.2 = 7 ∧ p.1 ≥ p.2} = 14 := 
by
  sorry

end factoring_options_count_l9_9822


namespace proof_equivalence_l9_9911

noncomputable def compute_expression (N : ℕ) (M : ℕ) : ℚ :=
  ((N - 3)^3 + (N - 2)^3 + (N - 1)^3 + N^3 + (N + 1)^3 + (N + 2)^3 + (N + 3)^3) /
  ((M - 3) * (M - 2) + (M - 1) * M + M * (M + 1) + (M + 2) * (M + 3))

theorem proof_equivalence:
  let N := 65536
  let M := 32768
  compute_expression N M = 229376 := 
  by
    sorry

end proof_equivalence_l9_9911


namespace opposite_of_2023_l9_9409

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l9_9409


namespace opposite_of_2023_l9_9223

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l9_9223


namespace inequality_problem_l9_9647

theorem inequality_problem (a b c : ℝ) (h : a < b ∧ b < 0) : a^2 > a * b ∧ a * b > b^2 :=
by
  -- The proof is supposed to be here
  sorry

end inequality_problem_l9_9647


namespace one_plus_x_pow_gt_one_plus_nx_l9_9064

theorem one_plus_x_pow_gt_one_plus_nx (x : ℝ) (n : ℕ) (hx1 : x > -1) (hx2 : x ≠ 0)
  (hn1 : n ≥ 2) : (1 + x)^n > 1 + n * x :=
sorry

end one_plus_x_pow_gt_one_plus_nx_l9_9064


namespace opposite_of_2023_is_neg2023_l9_9315

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l9_9315


namespace number_of_classmates_l9_9004

theorem number_of_classmates (n m : ℕ) (h₁ : n < 100) (h₂ : m = 9)
:(2 ^ 6 - 1) = 63 → 63 / m = 7 := by
  intros 
  sorry

end number_of_classmates_l9_9004


namespace probability_not_losing_l9_9767

theorem probability_not_losing (P_winning P_drawing : ℚ)
  (h_winning : P_winning = 1/3)
  (h_drawing : P_drawing = 1/4) :
  P_winning + P_drawing = 7/12 := 
by
  sorry

end probability_not_losing_l9_9767


namespace opposite_of_2023_l9_9410

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l9_9410


namespace P_3_eq_seven_eighths_P_4_ne_fifteen_sixteenths_P_decreasing_P_recurrence_l9_9492

open ProbabilityTheory

section
/-- Probability of not getting three consecutive heads -/
def P (n : ℕ) : ℚ := sorry

theorem P_3_eq_seven_eighths : P 3 = 7 / 8 := sorry

theorem P_4_ne_fifteen_sixteenths : P 4 ≠ 15 / 16 := sorry

theorem P_decreasing (n : ℕ) (h : 2 ≤ n) : P (n + 1) < P n := sorry

theorem P_recurrence (n : ℕ) (h : 4 ≤ n) : P n = (1 / 2) * P (n - 1) + (1 / 4) * P (n - 2) + (1 / 8) * P (n - 3) := sorry
end

end P_3_eq_seven_eighths_P_4_ne_fifteen_sixteenths_P_decreasing_P_recurrence_l9_9492


namespace white_clothing_probability_l9_9025

theorem white_clothing_probability (total_athletes sample_size k_min k_max : ℕ) 
  (red_upper_bound white_upper_bound yellow_upper_bound sampled_start_interval : ℕ)
  (h_total : total_athletes = 600)
  (h_sample : sample_size = 50)
  (h_intervals : total_athletes / sample_size = 12)
  (h_group_start : sampled_start_interval = 4)
  (h_red_upper : red_upper_bound = 311)
  (h_white_upper : white_upper_bound = 496)
  (h_yellow_upper : yellow_upper_bound = 600)
  (h_k_min : k_min = 26)   -- Calculated from 312 <= 12k + 4
  (h_k_max : k_max = 41)  -- Calculated from 12k + 4 <= 496
  : (k_max - k_min + 1) / sample_size = 8 / 25 := 
by
  sorry

end white_clothing_probability_l9_9025


namespace value_set_for_a_non_empty_proper_subsets_l9_9515

def A : Set ℝ := {x | x^2 - 4 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 6 = 0}

theorem value_set_for_a (M : Set ℝ) : 
  (∀ (a : ℝ), B a ⊆ A → a ∈ M) :=
sorry

theorem non_empty_proper_subsets (M : Set ℝ) :
  M = {0, 3, -3} →
  (∃ S : Set (Set ℝ), S = {{0}, {3}, {-3}, {0, 3}, {0, -3}, {3, -3}}) :=
sorry

end value_set_for_a_non_empty_proper_subsets_l9_9515


namespace exponent_problem_l9_9081

theorem exponent_problem 
  (a : ℝ) (x : ℝ) (y : ℝ) 
  (h1 : a > 0) 
  (h2 : a^x = 3) 
  (h3 : a^y = 5) : 
  a^(2*x + y/2) = 9 * Real.sqrt 5 :=
by
  sorry

end exponent_problem_l9_9081


namespace opposite_of_2023_l9_9141

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l9_9141


namespace opposite_of_2023_l9_9195

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9195


namespace find_ab_pairs_l9_9776

theorem find_ab_pairs (a b s : ℕ) (a_pos : a > 0) (b_pos : b > 0) (s_gt_one : s > 1) :
  (a = 2^s ∧ b = 2^(2*s) - 1) ↔
  (∃ p k : ℕ, Prime p ∧ (a^2 + b + 1 = p^k) ∧
   (a^2 + b + 1 ∣ b^2 - a^3 - 1) ∧
   ¬ (a^2 + b + 1 ∣ (a + b - 1)^2)) :=
sorry

end find_ab_pairs_l9_9776


namespace horse_saddle_ratio_l9_9839

variable (H S : ℝ)
variable (m : ℝ)
variable (total_value saddle_value : ℝ)

theorem horse_saddle_ratio :
  total_value = 100 ∧ saddle_value = 12.5 ∧ H = m * saddle_value ∧ H + saddle_value = total_value → m = 7 :=
by
  sorry

end horse_saddle_ratio_l9_9839


namespace period_ending_time_l9_9923

theorem period_ending_time (start_time : ℕ) (rain_duration : ℕ) (no_rain_duration : ℕ) (end_time : ℕ) :
  start_time = 8 ∧ rain_duration = 4 ∧ no_rain_duration = 5 ∧ end_time = 8 + rain_duration + no_rain_duration
  → end_time = 17 :=
by
  sorry

end period_ending_time_l9_9923


namespace domain_of_function_l9_9859

theorem domain_of_function :
  ∀ x : ℝ, (0 ≤ x ∧ x ≠ 1) ↔ (∃ y : ℝ, y = 1 / (Real.sqrt x - 1)) := by
  sorry

end domain_of_function_l9_9859


namespace number_of_houses_on_block_l9_9605

theorem number_of_houses_on_block 
  (total_mail : ℕ) 
  (white_mailboxes : ℕ) 
  (red_mailboxes : ℕ) 
  (mail_per_house : ℕ) 
  (total_white_mail : ℕ) 
  (total_red_mail : ℕ) 
  (remaining_mail : ℕ)
  (additional_houses : ℕ)
  (total_houses : ℕ) :
  total_mail = 48 ∧ 
  white_mailboxes = 2 ∧ 
  red_mailboxes = 3 ∧ 
  mail_per_house = 6 ∧ 
  total_white_mail = white_mailboxes * mail_per_house ∧
  total_red_mail = red_mailboxes * mail_per_house ∧
  remaining_mail = total_mail - (total_white_mail + total_red_mail) ∧
  additional_houses = remaining_mail / mail_per_house ∧
  total_houses = white_mailboxes + red_mailboxes + additional_houses →
  total_houses = 8 :=
by 
  sorry

end number_of_houses_on_block_l9_9605


namespace find_m_n_and_sqrt_l9_9524

-- definitions based on conditions
def condition_1 (m : ℤ) : Prop := m + 3 = 1
def condition_2 (n : ℤ) : Prop := 2 * n - 12 = 64

-- the proof problem statement
theorem find_m_n_and_sqrt (m n : ℤ) (h1 : condition_1 m) (h2 : condition_2 n) : 
  m = -2 ∧ n = 38 ∧ Int.sqrt (m + n) = 6 := 
sorry

end find_m_n_and_sqrt_l9_9524


namespace opposite_of_2023_l9_9185

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9185


namespace smallest_integer_to_make_1008_perfect_square_l9_9739

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem smallest_integer_to_make_1008_perfect_square : ∃ k : ℕ, k > 0 ∧ 
  (∀ m : ℕ, m > 0 → (is_perfect_square (1008 * m) → m ≥ k)) ∧ is_perfect_square (1008 * k) :=
by
  sorry

end smallest_integer_to_make_1008_perfect_square_l9_9739


namespace evaluate_box_2_neg1_0_l9_9773

def box (a b c : ℤ) : ℚ := a^b - b^c + c^a

theorem evaluate_box_2_neg1_0 : box 2 (-1) 0 = -1/2 := 
by
  sorry

end evaluate_box_2_neg1_0_l9_9773


namespace polynomial_is_perfect_square_trinomial_l9_9803

-- The definition of a perfect square trinomial
def isPerfectSquareTrinomial (a b c m : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a * b = c ∧ 4 * a * a + m * b = 4 * a * b * b

-- The main theorem to prove that if the polynomial is a perfect square trinomial, then m = 20
theorem polynomial_is_perfect_square_trinomial (a b : ℝ) (h : isPerfectSquareTrinomial 2 1 5 25) :
  ∀ x, (4 * x * x + 20 * x + 25 = (2 * x + 5) * (2 * x + 5)) :=
by
  sorry

end polynomial_is_perfect_square_trinomial_l9_9803


namespace f_periodic_l9_9682

noncomputable def f (x : ℝ) : ℝ := sorry

theorem f_periodic (f : ℝ → ℝ)
  (h_bound : ∀ x : ℝ, |f x| ≤ 1)
  (h_func : ∀ x : ℝ, f (x + 13 / 42) + f x = f (x + 1 / 6) + f (x + 1 / 7)) :
  ∀ x : ℝ, f (x + 1) = f x :=
sorry

end f_periodic_l9_9682


namespace money_left_after_bike_purchase_l9_9975

-- Definitions based on conditions
def jars : ℕ := 5
def quarters_per_jar : ℕ := 160
def quarter_value : ℝ := 0.25
def bike_cost : ℝ := 180

-- The theorem statement
theorem money_left_after_bike_purchase : (jars * quarters_per_jar * quarter_value) - bike_cost = 20 := by
  sorry

end money_left_after_bike_purchase_l9_9975


namespace correct_negation_of_p_l9_9568

open Real

def proposition_p (x : ℝ) := x > 0 → sin x ≥ -1

theorem correct_negation_of_p :
  ¬ (∀ x, proposition_p x) ↔ (∃ x, x > 0 ∧ sin x < -1) :=
by
  sorry

end correct_negation_of_p_l9_9568


namespace opposite_of_2023_l9_9296

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l9_9296


namespace opposite_of_2023_l9_9250

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l9_9250


namespace opposite_of_2023_l9_9149

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l9_9149


namespace continuous_zero_point_condition_l9_9933

theorem continuous_zero_point_condition (f : ℝ → ℝ) {a b : ℝ} (h_cont : ContinuousOn f (Set.Icc a b)) :
  (f a * f b < 0) → (∃ c ∈ Set.Ioo a b, f c = 0) ∧ ¬ (∃ c ∈ Set.Ioo a b, f c = 0 → f a * f b < 0) :=
sorry

end continuous_zero_point_condition_l9_9933


namespace rahul_share_is_100_l9_9569

-- Definitions of the conditions
def rahul_rate := 1/3
def rajesh_rate := 1/2
def total_payment := 250

-- Definition of their work rate when they work together
def combined_rate := rahul_rate + rajesh_rate

-- Definition of the total value of the work done in one day when both work together
noncomputable def combined_work_value := total_payment / combined_rate

-- Definition of Rahul's share for the work done in one day
noncomputable def rahul_share := rahul_rate * combined_work_value

-- The theorem we need to prove
theorem rahul_share_is_100 : rahul_share = 100 := by
  sorry

end rahul_share_is_100_l9_9569


namespace c_rent_share_l9_9737

-- Definitions based on conditions
def a_oxen := 10
def a_months := 7
def b_oxen := 12
def b_months := 5
def c_oxen := 15
def c_months := 3
def total_rent := 105

-- Calculate the shares in ox-months
def share_a := a_oxen * a_months
def share_b := b_oxen * b_months
def share_c := c_oxen * c_months

-- Calculate the total ox-months
def total_ox_months := share_a + share_b + share_c

-- Calculate the rent per ox-month
def rent_per_ox_month := total_rent / total_ox_months

-- Calculate the amount C should pay
def amount_c_should_pay := share_c * rent_per_ox_month

-- Prove the statement
theorem c_rent_share : amount_c_should_pay = 27 := by
  sorry

end c_rent_share_l9_9737


namespace jenn_money_left_over_l9_9976

-- Definitions based on problem conditions
def num_jars : ℕ := 5
def quarters_per_jar : ℕ := 160
def value_per_quarter : ℚ := 0.25   -- Rational number to represent $0.25
def cost_of_bike : ℚ := 180         -- Rational number to represent $180

-- Statement to prove that Jenn will have $20 left after buying the bike
theorem jenn_money_left_over : 
  (num_jars * quarters_per_jar * value_per_quarter) - cost_of_bike = 20 :=
by
  sorry

end jenn_money_left_over_l9_9976


namespace opposite_of_2023_l9_9191

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9191


namespace sum_of_squares_of_biking_jogging_swimming_rates_l9_9510

theorem sum_of_squares_of_biking_jogging_swimming_rates (b j s : ℕ) 
  (h1 : 2 * b + 3 * j + 4 * s = 74) 
  (h2 : 4 * b + 2 * j + 3 * s = 91) : 
  (b^2 + j^2 + s^2 = 314) :=
sorry

end sum_of_squares_of_biking_jogging_swimming_rates_l9_9510


namespace rectangle_ratio_l9_9670

-- Given conditions
variable (w : ℕ) -- width is a natural number

-- Definitions based on conditions 
def length := 10
def perimeter := 30

-- Theorem to prove
theorem rectangle_ratio (h : 2 * length + 2 * w = perimeter) : w = 5 ∧ 1 = 1 ∧ 2 = 2 :=
by
  sorry

end rectangle_ratio_l9_9670


namespace opposite_of_2023_l9_9274

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9274


namespace min_value_is_four_l9_9986

noncomputable def min_value (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 2) : ℝ :=
  (x + y) / (x * y * z)

theorem min_value_is_four (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 2) :
  min_value x y z h1 h2 h3 h4 = 4 :=
sorry

end min_value_is_four_l9_9986


namespace opposite_of_2023_is_neg_2023_l9_9230

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l9_9230


namespace rhombus_diagonal_length_l9_9579

theorem rhombus_diagonal_length
  (d1 d2 A : ℝ)
  (h1 : d1 = 20)
  (h2 : A = 250)
  (h3 : A = (d1 * d2) / 2) :
  d2 = 25 :=
by
  sorry

end rhombus_diagonal_length_l9_9579


namespace find_valid_pairs_l9_9634

def divides (a b : Nat) : Prop := ∃ k, b = a * k

def valid_pair (a b : Nat) : Prop :=
  divides (a^2 * b) (b^2 + 3 * a)

theorem find_valid_pairs :
  {ab | valid_pair ab.1 ab.2} = ({(1, 1), (1, 3)} : Set (Nat × Nat)) :=
by
  sorry

end find_valid_pairs_l9_9634


namespace additional_carpet_needed_is_94_l9_9547

noncomputable def area_room_a : ℝ := 4 * 20

noncomputable def area_room_b : ℝ := area_room_a / 2.5

noncomputable def total_area : ℝ := area_room_a + area_room_b

noncomputable def carpet_jessie_has : ℝ := 18

noncomputable def additional_carpet_needed : ℝ := total_area - carpet_jessie_has

theorem additional_carpet_needed_is_94 :
  additional_carpet_needed = 94 := by
  sorry

end additional_carpet_needed_is_94_l9_9547


namespace opposite_of_2023_l9_9439

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l9_9439


namespace subset_condition_l9_9069

def A : Set ℝ := {2, 0, 1, 6}
def B (a : ℝ) : Set ℝ := {x | x + a > 0}

theorem subset_condition (a : ℝ) (h : A ⊆ B a) : a > 0 :=
sorry

end subset_condition_l9_9069


namespace divide_friends_among_teams_l9_9947

theorem divide_friends_among_teams :
  let friends_num := 8
  let teams_num := 4
  (teams_num ^ friends_num) = 65536 := by
  sorry

end divide_friends_among_teams_l9_9947


namespace opposite_of_2023_l9_9162

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l9_9162


namespace clare_remaining_money_l9_9769

-- Definitions based on conditions
def clare_initial_money : ℕ := 47
def bread_quantity : ℕ := 4
def milk_quantity : ℕ := 2
def bread_cost : ℕ := 2
def milk_cost : ℕ := 2

-- The goal is to prove that Clare has $35 left after her purchases.
theorem clare_remaining_money : 
  clare_initial_money - (bread_quantity * bread_cost + milk_quantity * milk_cost) = 35 := 
sorry

end clare_remaining_money_l9_9769


namespace opposite_of_2023_l9_9215

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l9_9215


namespace tea_sale_price_correct_l9_9035

noncomputable def cost_price (weight: ℕ) (unit_price: ℕ) : ℕ := weight * unit_price
noncomputable def desired_profit (cost: ℕ) (percentage: ℕ) : ℕ := cost * percentage / 100
noncomputable def sale_price (cost: ℕ) (profit: ℕ) : ℕ := cost + profit
noncomputable def sale_price_per_kg (total_sale_price: ℕ) (weight: ℕ) : ℚ := total_sale_price / weight

theorem tea_sale_price_correct :
  ∀ (weight_A weight_B weight_C weight_D cost_per_kg_A cost_per_kg_B cost_per_kg_C cost_per_kg_D
     profit_percent_A profit_percent_B profit_percent_C profit_percent_D : ℕ),

  weight_A = 80 →
  weight_B = 20 →
  weight_C = 50 →
  weight_D = 30 →
  cost_per_kg_A = 15 →
  cost_per_kg_B = 20 →
  cost_per_kg_C = 25 →
  cost_per_kg_D = 30 →
  profit_percent_A = 25 →
  profit_percent_B = 30 →
  profit_percent_C = 20 →
  profit_percent_D = 15 →
  
  sale_price_per_kg (sale_price (cost_price weight_A cost_per_kg_A) (desired_profit (cost_price weight_A cost_per_kg_A) profit_percent_A)) weight_A = 18.75 →
  sale_price_per_kg (sale_price (cost_price weight_B cost_per_kg_B) (desired_profit (cost_price weight_B cost_per_kg_B) profit_percent_B)) weight_B = 26 →
  sale_price_per_kg (sale_price (cost_price weight_C cost_per_kg_C) (desired_profit (cost_price weight_C cost_per_kg_C) profit_percent_C)) weight_C = 30 →
  sale_price_per_kg (sale_price (cost_price weight_D cost_per_kg_D) (desired_profit (cost_price weight_D cost_per_kg_D) profit_percent_D)) weight_D = 34.5 :=
by
  intros
  sorry

end tea_sale_price_correct_l9_9035


namespace opposite_of_2023_l9_9138

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l9_9138


namespace find_original_number_l9_9840

theorem find_original_number (x : ℚ) (h : 1 + 1 / x = 8 / 3) : x = 3 / 5 := by
  sorry

end find_original_number_l9_9840


namespace opposite_of_2023_l9_9165

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l9_9165


namespace triangle_parallel_vectors_l9_9784

noncomputable def collinear {V : Type*} [AddCommGroup V] [Module ℝ V]
  (P₁ P₂ P₃ : V) : Prop :=
∃ t : ℝ, P₃ = P₁ + t • (P₂ - P₁)

theorem triangle_parallel_vectors
  (A B C C₁ A₁ B₁ C₂ A₂ B₂ : ℝ × ℝ)
  (h1 : collinear A B C₁) (h2 : collinear B C A₁) (h3 : collinear C A B₁)
  (ratio1 : ∀ (AC1 CB : ℝ), AC1 / CB = 1) (ratio2 : ∀ (BA1 AC : ℝ), BA1 / AC = 1) (ratio3 : ∀ (CB B1A : ℝ), CB / B1A = 1)
  (h4 : collinear A₁ B₁ C₂) (h5 : collinear B₁ C₁ A₂) (h6 : collinear C₁ A₁ B₂)
  (n : ℝ)
  (ratio4 : ∀ (A1C2 C2B1 : ℝ), A1C2 / C2B1 = n) (ratio5 : ∀ (B1A2 A2C1 : ℝ), B1A2 / A2C1 = n) (ratio6 : ∀ (C1B2 B2A1 : ℝ), C1B2 / B2A1 = n) :
  collinear A C A₂ ∧ collinear C B C₂ ∧ collinear B A B₂ :=
sorry

end triangle_parallel_vectors_l9_9784


namespace students_with_no_preference_l9_9819

def total_students : ℕ := 210
def prefer_mac : ℕ := 60
def equally_prefer_both (x : ℕ) : ℕ := x / 3

def no_preference_students : ℕ :=
  total_students - (prefer_mac + equally_prefer_both prefer_mac)

theorem students_with_no_preference :
  no_preference_students = 130 :=
by
  sorry

end students_with_no_preference_l9_9819


namespace probability_not_losing_l9_9768

theorem probability_not_losing (P_winning P_drawing : ℚ)
  (h_winning : P_winning = 1/3)
  (h_drawing : P_drawing = 1/4) :
  P_winning + P_drawing = 7/12 := 
by
  sorry

end probability_not_losing_l9_9768


namespace opposite_of_2023_l9_9204

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l9_9204


namespace opposite_of_2023_l9_9411

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l9_9411


namespace smallest_positive_period_minimum_value_of_f_l9_9073

open Real

noncomputable def f (x : ℝ) : ℝ :=
  2 * cos x * sin (x + π / 3) - sqrt 3 * sin x ^ 2 + sin x * cos x

theorem smallest_positive_period :
  ∀ x, f (x + π) = f x :=
sorry

theorem minimum_value_of_f :
  ∀ k : ℤ, f (k * π - 5 * π / 12) = -2 :=
sorry

end smallest_positive_period_minimum_value_of_f_l9_9073


namespace opposite_of_2023_l9_9392

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9392


namespace expected_number_of_sixes_l9_9864

-- Define the probability of not rolling a 6 on one die
def prob_not_six : ℚ := 5 / 6

-- Define the probability of rolling zero 6's on three dice
def prob_zero_six : ℚ := prob_not_six ^ 3

-- Define the probability of rolling exactly one 6 among the three dice
def prob_one_six (n : ℕ) : ℚ := n * (1 / 6) * (prob_not_six ^ (n - 1))

-- Calculate the probabilities of each specific outcomes
def prob_exactly_zero_six : ℚ := prob_zero_six
def prob_exactly_one_six : ℚ := prob_one_six 3 * (prob_not_six ^ 2)
def prob_exactly_two_six : ℚ := prob_one_six 3 * (1 / 6) * prob_not_six
def prob_exactly_three_six : ℚ := (1 / 6) ^ 3

-- Define the expected value calculation
noncomputable def expected_value : ℚ :=
  0 * prob_exactly_zero_six
  + 1 * prob_exactly_one_six
  + 2 * prob_exactly_two_six
  + 3 * prob_exactly_three_six

-- Prove that the expected value equals to 1/2
theorem expected_number_of_sixes : expected_value = 1 / 2 :=
  by
    sorry

end expected_number_of_sixes_l9_9864


namespace pair_D_equal_l9_9471

theorem pair_D_equal: (-1)^3 = (-1)^2023 := by
  sorry

end pair_D_equal_l9_9471


namespace minimum_value_proof_l9_9985

noncomputable def minimum_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 2) : ℝ :=
  (x + y) / (x * y * z)

theorem minimum_value_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 2) : 
  minimum_value x y z hx hy hz h ≥ 4 :=
sorry

end minimum_value_proof_l9_9985


namespace bagel_spending_l9_9500

variable (B D : ℝ)

theorem bagel_spending (h1 : B - D = 12.50) (h2 : D = B * 0.75) : B + D = 87.50 := 
sorry

end bagel_spending_l9_9500


namespace weight_of_mixture_l9_9889

noncomputable def total_weight_of_mixture (zinc_weight: ℝ) (zinc_ratio: ℝ) (total_ratio: ℝ) : ℝ :=
  (zinc_weight / zinc_ratio) * total_ratio

theorem weight_of_mixture (zinc_ratio: ℝ) (copper_ratio: ℝ) (tin_ratio: ℝ) (zinc_weight: ℝ) :
  total_weight_of_mixture zinc_weight zinc_ratio (zinc_ratio + copper_ratio + tin_ratio) = 98.95 :=
by 
  let ratio_sum := zinc_ratio + copper_ratio + tin_ratio
  let part_weight := zinc_weight / zinc_ratio
  let mixture_weight := part_weight * ratio_sum
  have h : mixture_weight = 98.95 := sorry
  exact h

end weight_of_mixture_l9_9889


namespace opposite_of_2023_is_neg_2023_l9_9233

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l9_9233


namespace opposite_of_2023_l9_9414

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l9_9414


namespace value_of_fraction_l9_9835

variable {x y : ℝ}

theorem value_of_fraction (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 4) : 
  (x + y) / (x - y) = Real.sqrt 3 := 
sorry

end value_of_fraction_l9_9835


namespace opposite_of_2023_l9_9202

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l9_9202


namespace arithmetic_sequence_geometric_subsequence_l9_9785

theorem arithmetic_sequence_geometric_subsequence :
  ∀ (a : ℕ → ℝ), (∀ n, a (n + 1) = a n + 2) ∧ (a 1 * a 3 = a 2 ^ 2) → a 2 = 4 :=
by
  intros a h
  sorry

end arithmetic_sequence_geometric_subsequence_l9_9785


namespace no_solution_perfect_square_abcd_l9_9056

theorem no_solution_perfect_square_abcd (x : ℤ) :
  (x ≤ 24) → (∃ (m : ℤ), 104 * x = m * m) → false :=
by
  sorry

end no_solution_perfect_square_abcd_l9_9056


namespace opposite_of_2023_l9_9394

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9394


namespace determine_g_10_l9_9584

noncomputable def g : ℝ → ℝ := sorry

-- Given condition
axiom g_condition : ∀ x y : ℝ, g x + g (2 * x + y) + 7 * x * y = g (3 * x - y) + 3 * x ^ 2 + 4

-- Theorem to prove
theorem determine_g_10 : g 10 = -46 := 
by
  -- skipping the proof here
  sorry

end determine_g_10_l9_9584


namespace fraction_equality_l9_9904

noncomputable def x := (4 : ℚ) / 6
noncomputable def y := (8 : ℚ) / 12

theorem fraction_equality : (6 * x + 8 * y) / (48 * x * y) = (7 : ℚ) / 16 := 
by 
  sorry

end fraction_equality_l9_9904


namespace opposite_of_2023_l9_9226

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l9_9226


namespace opposite_of_2023_l9_9362

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l9_9362


namespace calculate_expression_l9_9620

theorem calculate_expression :
  ((2000000000000 - 1234567890123) * 3 = 2296296329631) :=
by 
  sorry

end calculate_expression_l9_9620


namespace opposite_of_2023_is_neg_2023_l9_9245

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l9_9245


namespace opposite_of_2023_l9_9372

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l9_9372


namespace opposite_of_2023_is_neg_2023_l9_9243

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l9_9243


namespace trigonometric_identity_l9_9646

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) :
  Real.cos (2 * α) - Real.sin α * Real.cos α = -1 :=
sorry

end trigonometric_identity_l9_9646


namespace opposite_of_2023_l9_9172

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l9_9172


namespace opposite_of_2023_l9_9403

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9403


namespace opposite_of_2023_is_neg_2023_l9_9240

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l9_9240


namespace opposite_of_2023_l9_9265

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9265


namespace certain_number_x_l9_9086

theorem certain_number_x (p q x : ℕ) (hp : p > 1) (hq : q > 1)
  (h_eq : x * (p + 1) = 21 * (q + 1)) 
  (h_sum : p + q = 36) : x = 245 := 
by 
  sorry

end certain_number_x_l9_9086


namespace sequence_third_order_and_nth_term_l9_9752

-- Define the given sequence
def a (n : ℕ) : ℕ :=
  match n with
  | 0 => 4
  | 1 => 6
  | 2 => 13
  | 3 => 27
  | 4 => 50
  | 5 => 84
  | _ => sorry -- let’s define the general form for other terms later

-- Define first differences
def first_diff (n : ℕ) : ℤ := a (n + 1) - a n

-- Define second differences
def second_diff (n : ℕ) : ℤ := first_diff (n + 1) - first_diff n

-- Define third differences
def third_diff (n : ℕ) : ℤ := second_diff (n + 1) - second_diff n

-- Define the nth term formula
noncomputable def nth_term (n : ℕ) : ℚ := (1 / 6) * (2 * n^3 + 3 * n^2 - 11 * n + 30)

-- Theorem stating the least possible order is 3 and the nth term formula
theorem sequence_third_order_and_nth_term :
  (∀ n, third_diff n = 2) ∧ (∀ n, a n = nth_term n) :=
by
  sorry

end sequence_third_order_and_nth_term_l9_9752


namespace opposite_of_2023_l9_9390

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9390


namespace opposite_of_2023_l9_9216

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l9_9216


namespace multiple_of_4_multiple_of_8_multiple_of_16_not_multiple_of_32_l9_9679

def y : ℕ := 32 + 48 + 64 + 96 + 200 + 224 + 1600

theorem multiple_of_4 : y % 4 = 0 := by
  -- proof needed
  sorry

theorem multiple_of_8 : y % 8 = 0 := by
  -- proof needed
  sorry

theorem multiple_of_16 : y % 16 = 0 := by
  -- proof needed
  sorry

theorem not_multiple_of_32 : y % 32 ≠ 0 := by
  -- proof needed
  sorry

end multiple_of_4_multiple_of_8_multiple_of_16_not_multiple_of_32_l9_9679


namespace number_of_mixed_groups_l9_9724

theorem number_of_mixed_groups (n_children n_groups n_games boy_vs_boy girl_vs_girl mixed_games : ℕ) (h_children : n_children = 90) (h_groups : n_groups = 30) (h_games_per_group : n_games = 3) (h_boy_vs_boy : boy_vs_boy = 30) (h_girl_vs_girl : girl_vs_girl = 14) (h_total_games : mixed_games = 46) :
  (∀ g : ℕ, g * 2 = mixed_games → g = 23) :=
by
  intros g hg
  sorry

end number_of_mixed_groups_l9_9724


namespace linear_function_quadrant_l9_9472

theorem linear_function_quadrant (x y : ℝ) (h : y = 2 * x - 3) : ¬(∃ x y : ℝ, x < 0 ∧ y > 0 ∧ y = 2 * x - 3) :=
sorry

end linear_function_quadrant_l9_9472


namespace pyramid_side_length_l9_9847

-- Definitions for our conditions
def area_of_lateral_face : ℝ := 150
def slant_height : ℝ := 25

-- Theorem statement
theorem pyramid_side_length (A : ℝ) (h : ℝ) (s : ℝ) (hA : A = area_of_lateral_face) (hh : h = slant_height) :
  A = (1 / 2) * s * h → s = 12 :=
by
  intro h_eq
  rw [hA, hh, area_of_lateral_face, slant_height] at h_eq
  -- Steps to verify s = 12
  sorry

end pyramid_side_length_l9_9847


namespace relationship_among_abc_l9_9070

noncomputable def a : ℝ := real.sqrt 2
noncomputable def b : ℝ := real.log 2 / real.log 3
noncomputable def c : ℝ := real.log (real.sin 1) / real.log 2

theorem relationship_among_abc : a > b ∧ b > c :=
by
  -- proof to be provided
  sorry

end relationship_among_abc_l9_9070


namespace opposite_of_2023_l9_9358

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l9_9358


namespace domain_of_function_l9_9858

theorem domain_of_function :
  ∀ x : ℝ, (0 ≤ x ∧ x ≠ 1) ↔ (∃ y : ℝ, y = 1 / (Real.sqrt x - 1)) := by
  sorry

end domain_of_function_l9_9858


namespace opposite_of_2023_l9_9214

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l9_9214


namespace remaining_pens_l9_9748

theorem remaining_pens (blue_initial black_initial red_initial green_initial purple_initial : ℕ)
                        (blue_removed black_removed red_removed green_removed purple_removed : ℕ) :
  blue_initial = 15 → black_initial = 27 → red_initial = 12 → green_initial = 10 → purple_initial = 8 →
  blue_removed = 8 → black_removed = 9 → red_removed = 3 → green_removed = 5 → purple_removed = 6 →
  blue_initial - blue_removed + black_initial - black_removed + red_initial - red_removed +
  green_initial - green_removed + purple_initial - purple_removed = 41 :=
by
  intros
  sorry

end remaining_pens_l9_9748


namespace part_a_part_b_l9_9730

-- Define the conditions
def digit5 : ℕ := 1
def digit3 : ℕ := 2
def digit2 : ℕ := 100
def total_digits : ℕ := 10

-- Define the problem statement in Lean
def ways_to_form_valid_numbers : ℕ :=
  let zero_3s := Nat.choose total_digits digit5
  let one_3 := Nat.choose total_digits digit5 * Nat.choose (total_digits - digit5) 1
  let two_3s := Nat.choose total_digits digit5 * Nat.choose (total_digits - digit5) digit3
  zero_3s + one_3 + two_3s

def number_in_position (n : ℕ) : ℕ := 5322222322

theorem part_a : ways_to_form_valid_numbers = 460 := by
  sorry

theorem part_b : number_in_position 455 = 5322222322 := by
  sorry

end part_a_part_b_l9_9730


namespace tea_price_l9_9755

theorem tea_price 
  (x : ℝ)
  (total_cost_80kg_tea : ℝ := 80 * x)
  (total_cost_20kg_tea : ℝ := 20 * 20)
  (total_selling_price : ℝ := 1920)
  (profit_condition : 1.2 * (total_cost_80kg_tea + total_cost_20kg_tea) = total_selling_price) :
  x = 15 :=
by
  sorry

end tea_price_l9_9755


namespace opposite_of_2023_is_neg2023_l9_9356

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l9_9356


namespace opposite_of_2023_l9_9330

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l9_9330


namespace square_side_length_l9_9490

variable (s : ℕ)
variable (P A : ℕ)

theorem square_side_length (h1 : P = 52) (h2 : A = 169) (h3 : P = 4 * s) (h4 : A = s * s) : s = 13 :=
sorry

end square_side_length_l9_9490


namespace range_a_of_function_has_two_zeros_l9_9660

noncomputable def f (a x : ℝ) : ℝ := a^x - x - a

theorem range_a_of_function_has_two_zeros (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) : 
  1 < a :=
sorry

end range_a_of_function_has_two_zeros_l9_9660


namespace find_purple_balls_count_l9_9743

theorem find_purple_balls_count (k : ℕ) (h : ∃ k > 0, (21 - 3 * k) = (3 / 4) * (7 + k)) : k = 4 :=
sorry

end find_purple_balls_count_l9_9743


namespace range_of_a_l9_9917

theorem range_of_a (a x y : ℝ)
  (h1 : x + 3 * y = 2 + a)
  (h2 : 3 * x + y = -4 * a)
  (hxy : x + y > 2) : a < -2 := 
sorry

end range_of_a_l9_9917


namespace opposite_of_2023_is_neg_2023_l9_9278

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l9_9278


namespace opposite_of_2023_l9_9249

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l9_9249


namespace opposite_of_2023_l9_9423

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l9_9423


namespace sqrt_mixed_number_simplified_l9_9053

theorem sqrt_mixed_number_simplified : 
  (Real.sqrt (12 + 1 / 9) = Real.sqrt 109 / 3) := by
  sorry

end sqrt_mixed_number_simplified_l9_9053


namespace range_of_a_l9_9916

theorem range_of_a (a x y : ℝ)
  (h1 : x + 3 * y = 2 + a)
  (h2 : 3 * x + y = -4 * a)
  (hxy : x + y > 2) : a < -2 := 
sorry

end range_of_a_l9_9916


namespace find_x_l9_9741

theorem find_x (x : ℝ) (h : 0.009 / x = 0.03) : x = 0.3 :=
sorry

end find_x_l9_9741


namespace opposite_of_2023_is_neg_2023_l9_9284

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l9_9284


namespace opposite_of_2023_l9_9158

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l9_9158


namespace sequence_geometric_proof_l9_9685

theorem sequence_geometric_proof (a : ℕ → ℕ) (h1 : a 1 = 5) (h2 : ∀ n, a (n + 1) = 2 * a n) :
  ∀ n, a n = 5 * 2 ^ (n - 1) :=
by
  sorry

end sequence_geometric_proof_l9_9685


namespace smallest_fourth_number_l9_9896

theorem smallest_fourth_number :
  ∃ (a b : ℕ), 145 + 10 * a + b = 4 * (28 + a + b) ∧ 10 * a + b = 35 :=
by
  sorry

end smallest_fourth_number_l9_9896


namespace quad_intersects_x_axis_l9_9938

theorem quad_intersects_x_axis (k : ℝ) :
  (∃ x : ℝ, k * x ^ 2 - 7 * x - 7 = 0) ↔ (k ≥ -7 / 4 ∧ k ≠ 0) :=
by sorry

end quad_intersects_x_axis_l9_9938


namespace candy_distribution_count_l9_9123

-- Definitions of the problem setup
def ten_distinct_candies : ℕ := 10
def num_bags : ℕ := 4

-- Conditions: each bag (red, blue, and brown) must get at least 1 candy
def red_bag : ℕ := 1
def blue_bag : ℕ := 1
def brown_bag : ℕ := 1

-- The main statement/question that needs proof
theorem candy_distribution_count :
  (∑ r in finset.Ico 1 (ten_distinct_candies - red_bag + 1),
   ∑ b in finset.Ico 1 (ten_distinct_candies - r + 1),
   ∑ n in finset.Ico 1 (ten_distinct_candies - r - b + 1),
   nat.choose (ten_distinct_candies) r * nat.choose (ten_distinct_candies - r) b * nat.choose (ten_distinct_candies - r - b) n * nat.choose (ten_distinct_candies - r - b - n) (10 - r - b - n)) = 3176 := 
sorry

end candy_distribution_count_l9_9123


namespace inversely_proportional_rs_l9_9576

theorem inversely_proportional_rs (r s : ℝ) (k : ℝ) 
(h_invprop : r * s = k) 
(h1 : r = 40) (h2 : s = 5) 
(h3 : s = 8) : r = 25 := by
  sorry

end inversely_proportional_rs_l9_9576


namespace probability_two_dice_same_number_l9_9460

-- Suppose we roll two fair 6-sided dice
noncomputable def dice_probability_same_number : ℚ :=
  let total_outcomes := 36
  let successful_outcomes := 6
  successful_outcomes / total_outcomes

theorem probability_two_dice_same_number :
  dice_probability_same_number = 1 / 6 :=
by sorry

end probability_two_dice_same_number_l9_9460


namespace opposite_of_2023_is_neg2023_l9_9318

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l9_9318


namespace total_nephews_correct_l9_9494

namespace Nephews

-- Conditions
variable (ten_years_ago : Nat)
variable (current_alden_nephews : Nat)
variable (vihaan_extra_nephews : Nat)
variable (alden_nephews_10_years_ago : ten_years_ago = 50)
variable (alden_nephews_double : ten_years_ago * 2 = current_alden_nephews)
variable (vihaan_nephews : vihaan_extra_nephews = 60)

-- Answer
def total_nephews (alden_nephews_now vihaan_nephews_now : Nat) : Nat :=
  alden_nephews_now + vihaan_nephews_now

-- Proof statement
theorem total_nephews_correct :
  ∃ (alden_nephews_now vihaan_nephews_now : Nat), 
    alden_nephews_10_years_ago →
    alden_nephews_double →
    vihaan_nephews →
    alden_nephews_now = current_alden_nephews →
    vihaan_nephews_now = current_alden_nephews + vihaan_extra_nephews →
    total_nephews alden_nephews_now vihaan_nephews_now = 260 :=
by
  sorry

end Nephews

end total_nephews_correct_l9_9494


namespace opposite_of_2023_l9_9148

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l9_9148


namespace quadrilateral_circumscribed_circle_l9_9539

theorem quadrilateral_circumscribed_circle (a : ℝ) :
  ((a + 2) * x + (1 - a) * y - 3 = 0) ∧ ((a - 1) * x + (2 * a + 3) * y + 2 = 0) →
  ( a = 1 ∨ a = -1 ) :=
by
  intro h
  sorry

end quadrilateral_circumscribed_circle_l9_9539


namespace opposite_of_2023_l9_9373

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l9_9373


namespace opposite_of_2023_l9_9145

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l9_9145


namespace upsilon_value_l9_9079

theorem upsilon_value (Upsilon : ℤ) (h : 5 * (-3) = Upsilon - 3) : Upsilon = -12 :=
by
  sorry

end upsilon_value_l9_9079


namespace polyhedron_value_calculation_l9_9024

noncomputable def calculate_value (P T V : ℕ) : ℕ :=
  100 * P + 10 * T + V

theorem polyhedron_value_calculation :
  ∀ (P T V E F : ℕ),
    F = 36 ∧
    T + P = 36 ∧
    E = (3 * T + 5 * P) / 2 ∧
    V = E - F + 2 →
    calculate_value P T V = 2018 :=
by
  intros P T V E F h
  sorry

end polyhedron_value_calculation_l9_9024


namespace unique_representation_reciprocal_sum_l9_9809
noncomputable theory

open Classical Real Nat

theorem unique_representation (x : ℚ) (hx : 0 < x) : 
    ∃! (a : ℕ → ℤ) (n : ℕ), x = (∑ i in Finset.range n, a i / Nat.factorial (i + 1)) ∧ 
                             (∀ i, 1 ≤ i → i < n → 0 ≤ a i ∧ a i < i + 1) :=
sorry

theorem reciprocal_sum (x : ℚ) (hx : 0 < x) : 
    ∃ (n : ℕ → ℕ), (x = ∑ i, (1 / n i : ℝ)) ∧ (∀ i, 10^6 < n i) :=
sorry

end unique_representation_reciprocal_sum_l9_9809


namespace sum_of_super_cool_triangle_areas_l9_9611

noncomputable def super_cool_triangle_sum_area : ℕ :=
  let leg_pairs := [(7, 42), (8, 24), (9, 18), (10, 15), (12, 12)] in
  let areas := leg_pairs.map (λ p, (p.1 * p.2) / 2) in
  areas.sum

theorem sum_of_super_cool_triangle_areas : super_cool_triangle_sum_area = 471 :=
by
  sorry

end sum_of_super_cool_triangle_areas_l9_9611


namespace gifted_subscribers_l9_9829

theorem gifted_subscribers (initial_subs : ℕ) (revenue_per_sub : ℕ) (total_revenue : ℕ) (h1 : initial_subs = 150) (h2 : revenue_per_sub = 9) (h3 : total_revenue = 1800) :
  total_revenue / revenue_per_sub - initial_subs = 50 :=
by
  sorry

end gifted_subscribers_l9_9829


namespace opposite_of_2023_l9_9422

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l9_9422


namespace time_on_sideline_l9_9968

def total_game_time : ℕ := 90
def time_mark_played_first_period : ℕ := 20
def time_mark_played_second_period : ℕ := 35
def total_time_mark_played : ℕ := time_mark_played_first_period + time_mark_played_second_period

theorem time_on_sideline : total_game_time - total_time_mark_played = 35 := by
  sorry

end time_on_sideline_l9_9968


namespace opposite_of_2023_l9_9391

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9391


namespace arithmetic_sequence_value_l9_9626

theorem arithmetic_sequence_value 
    (a1 : ℤ) (a2 a3 a4 : ℤ) (a1_a4 : a1 = 18) 
    (b1 b2 b3 : ℤ) 
    (b1_b3 : b3 - b2 = 6 ∧ b2 - b1 = 6 ∧ b2 = 15 ∧ b3 = 21)
    (b1_a3 : a3 = b1 - 6 ∧ a4 = a1 + (a3 - 18) / 3) 
    (c1 c2 c3 c4 : ℝ) 
    (c1_b3 : c1 = a4) 
    (c2 : c2 = -14) 
    (c4 : ∃ m, c4 = b1 - m * (6 :ℝ) + - 0.5) 
    (n : ℝ) : 
    n = -12.5 := by 
  sorry

end arithmetic_sequence_value_l9_9626


namespace remainder_of_x_mod_11_l9_9881

theorem remainder_of_x_mod_11 {x : ℤ} (h : x % 66 = 14) : x % 11 = 3 :=
sorry

end remainder_of_x_mod_11_l9_9881


namespace price_of_36kgs_l9_9759

namespace Apples

-- Define the parameters l and q
variables (l q : ℕ)

-- Define the conditions
def cost_first_30kgs (l : ℕ) : ℕ := 30 * l
def cost_first_15kgs : ℕ := 150
def cost_33kgs (l q : ℕ) : ℕ := (30 * l) + (3 * q)
def cost_36kgs (l q : ℕ) : ℕ := (30 * l) + (6 * q)

-- Define the hypothesis for l and q based on given conditions
axiom l_value (h1 : cost_first_15kgs = 150) : l = 10
axiom q_value (h2 : cost_33kgs l q = 333) : q = 11

-- Prove the price of 36 kilograms of apples
theorem price_of_36kgs (h1 : cost_first_15kgs = 150) (h2 : cost_33kgs l q = 333) : cost_36kgs l q = 366 :=
sorry

end Apples

end price_of_36kgs_l9_9759


namespace mixed_groups_count_l9_9726

theorem mixed_groups_count
    (total_children : ℕ)
    (total_groups : ℕ)
    (children_per_group : ℕ)
    (boy_games : ℕ)
    (girl_games : ℕ)
    (all_play : ∀ g, g ∈ Finset.range total_groups → multiset.card (filter (λ x, x ∈ g) total_children) = children_per_group)
    (each_plays_twice : ∀ g, g ∈ Finset.range total_groups → multiset.card (filter (λ x, x ∈ g) total_children) = 2)
    (boy_game_count : ℕ)
    (girl_game_count : ℕ)
    (total_games : ℕ := total_groups * (children_per_group * (children_per_group - 1) / 2))
    (mixed_groups : ℕ)
    (game_equations : total_games = boy_games + girl_games + 2 * mixed_groups) :
    mixed_groups = 23 :=
begin
  sorry
end

end mixed_groups_count_l9_9726


namespace opposite_of_2023_l9_9449

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l9_9449


namespace jericho_money_left_l9_9459

/--
Given:
1. Twice the money Jericho has is 60.
2. Jericho owes Annika $14.
3. Jericho owes Manny half as much as he owes Annika.

Prove:
Jericho will be left with $9 after paying off all his debts.
-/
theorem jericho_money_left (j_money : ℕ) (annika_owes : ℕ) (manny_multiplier : ℕ) (debt : ℕ) (remaining_money : ℕ) :
  2 * j_money = 60 →
  annika_owes = 14 →
  manny_multiplier = 1 / 2 →
  debt = annika_owes + manny_multiplier * annika_owes →
  remaining_money = j_money - debt →
  remaining_money = 9 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end jericho_money_left_l9_9459


namespace opposite_of_2023_l9_9442

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l9_9442


namespace inequality_subtraction_l9_9926

theorem inequality_subtraction {a b c : ℝ} (h : a > b) : a - c > b - c := 
sorry

end inequality_subtraction_l9_9926


namespace cookies_from_dough_l9_9736

theorem cookies_from_dough :
  ∀ (length width : ℕ), length = 24 → width = 18 →
  ∃ (side : ℕ), side = Nat.gcd length width ∧ (length / side) * (width / side) = 12 :=
by
  intros length width h_length h_width
  simp only [h_length, h_width]
  use Nat.gcd length width
  simp only [Nat.gcd_rec]
  sorry

end cookies_from_dough_l9_9736


namespace parabola_point_distance_eq_l9_9489

open Real

theorem parabola_point_distance_eq (P : ℝ × ℝ) (V : ℝ × ℝ) (F : ℝ × ℝ)
    (hV: V = (0, 0)) (hF : F = (0, 2)) (P_on_parabola : P.1 ^ 2 = 8 * P.2) 
    (hPf : dist P F = 150) (P_in_first_quadrant : 0 ≤ P.1 ∧ 0 ≤ P.2) :
    P = (sqrt 1184, 148) :=
sorry

end parabola_point_distance_eq_l9_9489


namespace find_AC_l9_9964

noncomputable def isTriangle (A B C : Type) : Type := sorry

-- Define angles and lengths.
variables (A B C : Type)
variables (angle_A angle_B : ℝ)
variables (BC AC : ℝ)

-- Assume the given conditions.
axiom angle_A_60 : angle_A = 60 * Real.pi / 180
axiom angle_B_45 : angle_B = 45 * Real.pi / 180
axiom BC_12 : BC = 12

-- Statement to prove.
theorem find_AC 
  (h_triangle : isTriangle A B C)
  (h_angle_A : angle_A = 60 * Real.pi / 180)
  (h_angle_B : angle_B = 45 * Real.pi / 180)
  (h_BC : BC = 12) :
  ∃ AC : ℝ, AC = 8 * Real.sqrt 3 / 3 :=
sorry

end find_AC_l9_9964


namespace opposite_of_2023_is_neg_2023_l9_9235

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l9_9235


namespace joint_probability_l9_9595

open ProbabilityTheory

variable (Ω : Type) [MeasurableSpace Ω] (P : Measure Ω)

theorem joint_probability (a b : Set Ω) (hp_a : P a = 0.18)
  (hp_b_given_a : condCount P a b = 0.2) : P (a ∩ b) = 0.036 :=
by
  sorry

end joint_probability_l9_9595


namespace max_value_fx_when_a_neg1_find_a_when_max_fx_is_neg3_inequality_gx_if_a_pos_l9_9652

noncomputable def f (a x : ℝ) := a * x + Real.log x
noncomputable def g (a x : ℝ) := x * f a x
noncomputable def e := Real.exp 1

-- Statement for part (1)
theorem max_value_fx_when_a_neg1 : 
  ∀ x : ℝ, 0 < x → (f (-1) x ≤ f (-1) 1) :=
sorry

-- Statement for part (2)
theorem find_a_when_max_fx_is_neg3 : 
  (∀ x : ℝ, 0 < x ∧ x ≤ e → (f (-e^2) x ≤ -3)) →
  (∃ a : ℝ, a = -e^2) :=
sorry

-- Statement for part (3)
theorem inequality_gx_if_a_pos (a : ℝ) (hapos : 0 < a) 
  (x1 x2 : ℝ) (hxpos1 : 0 < x1) (hxpos2 : 0 < x2) (hx12 : x1 ≠ x2) :
  2 * g a ((x1 + x2) / 2) < g a x1 + g a x2 :=
sorry

end max_value_fx_when_a_neg1_find_a_when_max_fx_is_neg3_inequality_gx_if_a_pos_l9_9652


namespace find_shares_l9_9921

def shareA (B : ℝ) : ℝ := 3 * B
def shareC (B : ℝ) : ℝ := B - 25
def shareD (A B : ℝ) : ℝ := A + B - 10
def total_share (A B C D : ℝ) : ℝ := A + B + C + D

theorem find_shares :
  ∃ (A B C D : ℝ),
  A = 744.99 ∧
  B = 248.33 ∧
  C = 223.33 ∧
  D = 983.32 ∧
  A = shareA B ∧
  C = shareC B ∧
  D = shareD A B ∧
  total_share A B C D = 2200 := 
sorry

end find_shares_l9_9921


namespace opposite_of_2023_l9_9252

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l9_9252


namespace pauly_omelets_l9_9115

theorem pauly_omelets :
  let total_eggs := 3 * 12 in
  let eggs_per_omelet := 4 in
  let num_people := 3 in
  (total_eggs / eggs_per_omelet) / num_people = 3 :=
by
  let total_eggs := 3 * 12
  let eggs_per_omelet := 4
  let num_people := 3
  have h1 : total_eggs = 36 := by sorry
  have h2 : 36 / eggs_per_omelet = 9 := by sorry
  have h3 : 9 / num_people = 3 := by sorry
  exact h3

end pauly_omelets_l9_9115


namespace opposite_of_2023_l9_9201

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l9_9201


namespace lowest_degree_for_divisibility_by_7_lowest_degree_for_divisibility_by_12_l9_9464

-- Define a polynomial and conditions for divisibility by 7
def poly_deg_6 (a b c d e f g x : ℤ) : ℤ :=
  a * x^6 + b * x^5 + c * x^4 + d * x^3 + e * x^2 + f * x + g

-- Theorem for divisibility by 7
theorem lowest_degree_for_divisibility_by_7 : 
  (∀ x : ℤ, poly_deg_6 a b c d e f g x % 7 = 0) → false :=
sorry

-- Define a polynomial and conditions for divisibility by 12
def poly_deg_3 (a b c d x : ℤ) : ℤ :=
  a * x^3 + b * x^2 + c * x + d

-- Theorem for divisibility by 12
theorem lowest_degree_for_divisibility_by_12 : 
  (∀ x : ℤ, poly_deg_3 a b c d x % 12 = 0) → false :=
sorry

end lowest_degree_for_divisibility_by_7_lowest_degree_for_divisibility_by_12_l9_9464


namespace opposite_of_2023_l9_9378

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9378


namespace prob_at_least_one_interested_l9_9738

noncomputable def finance_club_probability (total_members interested_members : ℕ) : ℚ :=
  let prob_not_interested_first : ℚ := (total_members - interested_members) / total_members
  let prob_not_interested_second : ℚ := (total_members - interested_members) / (total_members - 1)
  let prob_both_not_interested : ℚ := prob_not_interested_first * prob_not_interested_second
  1 - prob_both_not_interested

theorem prob_at_least_one_interested : finance_club_probability 25 20 = 23 / 24 := 
by
  -- definitions
  let total_members := 25
  let interested_members := 20
  
  -- calculations
  let prob_not_interested_first : ℚ := (total_members - interested_members) / total_members
  let prob_not_interested_second : ℚ := (total_members - interested_members) / (total_members - 1)
  let prob_both_not_interested : ℚ := prob_not_interested_first * prob_not_interested_second
  let prob_at_least_one := 1 - prob_both_not_interested
  
  -- prove the statement
  have : prob_at_least_one = 23 / 24 := by
    simp [prob_not_interested_first, prob_not_interested_second, prob_both_not_interested, prob_at_least_one]

  exact this

end prob_at_least_one_interested_l9_9738


namespace compute_expression_l9_9680

-- Define the conditions
variables (ω : ℂ) (hω_nonreal : ω^3 = 1) (hω_eq : ω^2 + ω + 1 = 0)

-- State the theorem to be proved
theorem compute_expression (ω : ℂ) (hω_nonreal : ω^3 = 1) (hω_eq : ω^2 + ω + 1 = 0) :
  (1 - ω + ω^2)^6 + (1 + ω - ω^2)^6 = 128 := 
sorry

end compute_expression_l9_9680


namespace pigeonhole_principle_useful_inequality_l9_9980

open Finset

theorem pigeonhole_principle_useful_inequality (n : ℕ) (A : Finset ℕ) (hA : A = range (2^(n+1))) (B : Finset ℕ) (hB : B ⊆ A) (h_cardB : B.card = 2 * n + 1) :
  ∃ a b c ∈ B, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ b * c < 2 * a^2 ∧ 2 * a^2 < 4 * b * c := 
by
  sorry

end pigeonhole_principle_useful_inequality_l9_9980


namespace opposite_of_2023_is_neg_2023_l9_9244

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l9_9244


namespace minimal_disks_needed_l9_9764

-- Define the capacity of one disk
def disk_capacity : ℝ := 2.0

-- Define the number of files and their sizes
def num_files_0_9 : ℕ := 5
def size_file_0_9 : ℝ := 0.9

def num_files_0_8 : ℕ := 15
def size_file_0_8 : ℝ := 0.8

def num_files_0_5 : ℕ := 20
def size_file_0_5 : ℝ := 0.5

-- Total number of files
def total_files : ℕ := num_files_0_9 + num_files_0_8 + num_files_0_5

-- Proof statement: the minimal number of disks needed to store all files given their sizes and the disk capacity
theorem minimal_disks_needed : 
  ∀ (d : ℕ), 
    d = 18 → 
    total_files = 40 → 
    disk_capacity = 2.0 → 
    ((num_files_0_9 * size_file_0_9 + num_files_0_8 * size_file_0_8 + num_files_0_5 * size_file_0_5) / disk_capacity) ≤ d
  :=
by
  sorry

end minimal_disks_needed_l9_9764


namespace opposite_of_2023_l9_9176

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l9_9176


namespace opposite_of_2023_is_neg2023_l9_9357

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l9_9357


namespace angle_measure_l9_9087

theorem angle_measure (x : ℝ) (h : 180 - x = (90 - x) - 4) : x = 60 := by
  sorry

end angle_measure_l9_9087


namespace interior_sum_nine_l9_9673

-- Defining the function for the sum of the interior numbers in the nth row of Pascal's Triangle
def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

-- Given conditions
axiom interior_sum_4 : interior_sum 4 = 6
axiom interior_sum_5 : interior_sum 5 = 14

-- Goal to prove
theorem interior_sum_nine : interior_sum 9 = 254 := by
  sorry

end interior_sum_nine_l9_9673


namespace opposite_of_2023_l9_9441

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l9_9441


namespace opposite_of_2023_l9_9419

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l9_9419


namespace area_of_circle_l9_9691

noncomputable def point : Type := ℝ × ℝ

def A : point := (8, 15)
def B : point := (14, 9)

def is_on_circle (P : point) (r : ℝ) (C : point) : Prop :=
  (P.1 - C.1) ^ 2 + (P.2 - C.2) ^ 2 = r ^ 2

def tangent_intersects_x_axis (tangent_point : point) (circle_center : point) : Prop :=
  ∃ x : ℝ, ∃ C : point, C.2 = 0 ∧ tangent_point = C ∧ circle_center = (x, 0)

theorem area_of_circle :
  ∃ C : point, ∃ r : ℝ,
    is_on_circle A r C ∧ 
    is_on_circle B r C ∧ 
    tangent_intersects_x_axis A C ∧ 
    tangent_intersects_x_axis B C ∧ 
    (↑(π * r ^ 2) = (117 * π) / 8) :=
sorry

end area_of_circle_l9_9691


namespace exists_large_p_l9_9787

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sin (Real.pi * x)

theorem exists_large_p (d : ℝ) (h : d > 0) : ∃ p : ℝ, ∀ x : ℝ, |f (x + p) - f x| < d ∧ ∃ M : ℝ, M > 0 ∧ p > M :=
by {
  sorry
}

end exists_large_p_l9_9787


namespace opposite_of_2023_l9_9156

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l9_9156


namespace opposite_of_2023_l9_9266

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9266


namespace opposite_of_2023_l9_9220

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l9_9220


namespace mixed_groups_count_l9_9725

theorem mixed_groups_count
    (total_children : ℕ)
    (total_groups : ℕ)
    (children_per_group : ℕ)
    (boy_games : ℕ)
    (girl_games : ℕ)
    (all_play : ∀ g, g ∈ Finset.range total_groups → multiset.card (filter (λ x, x ∈ g) total_children) = children_per_group)
    (each_plays_twice : ∀ g, g ∈ Finset.range total_groups → multiset.card (filter (λ x, x ∈ g) total_children) = 2)
    (boy_game_count : ℕ)
    (girl_game_count : ℕ)
    (total_games : ℕ := total_groups * (children_per_group * (children_per_group - 1) / 2))
    (mixed_groups : ℕ)
    (game_equations : total_games = boy_games + girl_games + 2 * mixed_groups) :
    mixed_groups = 23 :=
begin
  sorry
end

end mixed_groups_count_l9_9725


namespace opposite_of_2023_is_neg2023_l9_9349

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l9_9349


namespace gcd_45345_34534_l9_9635

theorem gcd_45345_34534 : Nat.gcd 45345 34534 = 71 := by
  sorry

end gcd_45345_34534_l9_9635


namespace find_geometric_sequence_first_term_and_ratio_l9_9567

theorem find_geometric_sequence_first_term_and_ratio 
  (a1 a2 a3 a4 a5 : ℕ) 
  (h : a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5)
  (geo_seq : a2 = a1 * 3 / 2 ∧ a3 = a2 * 3 / 2 ∧ a4 = a3 * 3 / 2 ∧ a5 = a4 * 3 / 2)
  (sum_cond : a1 + a2 + a3 + a4 + a5 = 211) :
  (a1 = 16) ∧ (3 / 2 = 3 / 2) := 
by {
  sorry
}

end find_geometric_sequence_first_term_and_ratio_l9_9567


namespace jamie_collects_oysters_l9_9483

theorem jamie_collects_oysters (d : ℕ) (p : ℕ) (r : ℕ) (x : ℕ)
  (h1 : d = 14)
  (h2 : p = 56)
  (h3 : r = 25)
  (h4 : x = p / d * 100 / r) :
  x = 16 :=
by
  sorry

end jamie_collects_oysters_l9_9483


namespace opposite_of_2023_l9_9365

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l9_9365


namespace curve_intersections_l9_9794

def C1_param (t : ℝ) : ℝ × ℝ := (1 + (√2 / 2) * t, (√2 / 2) * t)

def C2_polar (ρ θ : ℝ) : ℝ := 1 / (ρ^2) - (cos θ)^2 / 2 - (sin θ)^2

def M : ℝ × ℝ := (1, 0)

theorem curve_intersections (t : ℝ) :
  (C1_param t).1 - (C1_param t).2 - 1 = 0 ∧ 
  (C2_polar ((C1_param t).1 / cos t) t) = 0 ∧
  let A : ℝ × ℝ := (0, -1) in
  let B : ℝ × ℝ := (4/3, 1/3) in
  (1 - 0)^2 + (0 + 1)^2 = 2 ∧
  ((4/3 - 1)^2 + (1/3 - 0)^2) = (2/9) ∧
  ((4/3 - 0)^2 + (1/3 + 1)^2) = (16/9) ∧
  (sqrt 2 * sqrt (2/9)) / sqrt (16/9) = sqrt 2 / 4 := sorry

end curve_intersections_l9_9794


namespace recruits_total_l9_9586

theorem recruits_total (x y z : ℕ) (total_people : ℕ) :
  (x = total_people - 51) ∧
  (y = total_people - 101) ∧
  (z = total_people - 171) ∧
  (x = 4 * y ∨ y = 4 * z ∨ x = 4 * z) ∧
  (∃ total_people, total_people = 211) :=
sorry

end recruits_total_l9_9586


namespace converse_even_sum_l9_9851

def is_even (n : Int) : Prop := ∃ k : Int, n = 2 * k

theorem converse_even_sum (a b : Int) :
  (is_even a ∧ is_even b → is_even (a + b)) →
  (is_even (a + b) → is_even a ∧ is_even b) :=
by
  sorry

end converse_even_sum_l9_9851


namespace card_tag_sum_l9_9499

noncomputable def W : ℕ := 200
noncomputable def X : ℝ := 2 / 3 * W
noncomputable def Y : ℝ := W + X
noncomputable def Z : ℝ := Real.sqrt Y
noncomputable def P : ℝ := X^3
noncomputable def Q : ℝ := Nat.factorial W / 100000
noncomputable def R : ℝ := 3 / 5 * (P + Q)
noncomputable def S : ℝ := W^1 + X^2 + Z^3

theorem card_tag_sum :
  W + X + Y + Z + P + S = 2373589.26 + Q + R :=
by
  sorry

end card_tag_sum_l9_9499


namespace tan_sum_angles_l9_9622

theorem tan_sum_angles : (Real.tan (17 * Real.pi / 180) + Real.tan (28 * Real.pi / 180)) / (1 - Real.tan (17 * Real.pi / 180) * Real.tan (28 * Real.pi / 180)) = 1 := 
by sorry

end tan_sum_angles_l9_9622


namespace repeating_decimal_eq_fraction_l9_9914

theorem repeating_decimal_eq_fraction :
  let a := (85 : ℝ) / 100
  let r := (1 : ℝ) / 100
  (∑' n : ℕ, a * (r ^ n)) = 85 / 99 := by
  let a := (85 : ℝ) / 100
  let r := (1 : ℝ) / 100
  exact sorry

end repeating_decimal_eq_fraction_l9_9914


namespace opposite_of_2023_is_neg_2023_l9_9291

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l9_9291


namespace root_polynomial_h_l9_9656

theorem root_polynomial_h (h : ℤ) : (2^3 + h * 2 + 10 = 0) → h = -9 :=
by
  sorry

end root_polynomial_h_l9_9656


namespace number_of_social_science_papers_selected_is_18_l9_9747

def total_social_science_papers : ℕ := 54
def total_humanities_papers : ℕ := 60
def total_other_papers : ℕ := 39
def total_selected_papers : ℕ := 51

def number_of_social_science_papers_selected : ℕ :=
  (total_social_science_papers * total_selected_papers) / (total_social_science_papers + total_humanities_papers + total_other_papers)

theorem number_of_social_science_papers_selected_is_18 :
  number_of_social_science_papers_selected = 18 :=
by 
  -- Proof to be provided
  sorry

end number_of_social_science_papers_selected_is_18_l9_9747


namespace prob_negative_one_to_zero_l9_9783

noncomputable theory
open MeasureTheory

variables {ξ : ℝ →ₘ measure_theory.real_measurable_space} 

-- Assume ξ follows a normal distribution N(0,1)
axiom ξ_normal : ξ = distribution.normal_std

-- Given condition: P(ξ>1)=a
variable (a : ℝ)
axiom P_ξ_gt_1 : measure_theory.measure_space.prob (λ x, x > 1) ξ = a

-- We need to prove: P(-1≤ξ≤0) = 1/2 - a
theorem prob_negative_one_to_zero :
  measure_theory.measure_space.prob (λ x, -1 ≤ x ∧ x ≤ 0) ξ = 1/2 - a :=
  sorry

end prob_negative_one_to_zero_l9_9783


namespace correct_statements_l9_9793

-- Given the values of x and y on the parabola
def parabola (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

-- Define the points on the parabola
def points_on_parabola (a b c : ℝ) : Prop :=
  parabola a b c (-1) = 3 ∧
  parabola a b c 0 = 0 ∧
  parabola a b c 1 = -1 ∧
  parabola a b c 2 = 0 ∧
  parabola a b c 3 = 3

-- Prove the correct statements
theorem correct_statements (a b c : ℝ) (h : points_on_parabola a b c) : 
  ¬(∃ x, parabola a b c x < 0 ∧ x < 0) ∧
  parabola a b c 2 = 0 :=
by 
  sorry

end correct_statements_l9_9793


namespace square_area_from_circle_area_l9_9615

variable (square_area : ℝ) (circle_area : ℝ)

theorem square_area_from_circle_area 
  (h1 : circle_area = 9 * Real.pi) 
  (h2 : square_area = (2 * Real.sqrt (circle_area / Real.pi))^2) : 
  square_area = 36 := 
by
  sorry

end square_area_from_circle_area_l9_9615


namespace opposite_of_2023_l9_9412

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l9_9412


namespace find_B_max_f_A_l9_9664

namespace ProofProblem

-- Definitions
variables {A B C a b c : ℝ} -- Angles and sides in the triangle
noncomputable def givenCondition (A B C a b c : ℝ) : Prop :=
  2 * b * Real.cos A = 2 * c - Real.sqrt 3 * a

noncomputable def f (x : ℝ) : ℝ :=
  Real.cos x * Real.sin (x + Real.pi / 3) - Real.sqrt 3 / 4

-- Problem Statements (to be proved)
theorem find_B (h : givenCondition A B C a b c) : B = Real.pi / 6 := sorry

theorem max_f_A (A : ℝ) (B : ℝ) (h1 : 0 < A) (h2 : A < 5 * Real.pi / 6) (h3 : B = Real.pi / 6) : (∃ (x : ℝ), f x = 1 / 2) := sorry

end ProofProblem

end find_B_max_f_A_l9_9664


namespace opposite_of_2023_l9_9144

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l9_9144


namespace woman_work_completion_days_l9_9749

def work_completion_days_man := 6
def work_completion_days_boy := 9
def work_completion_days_combined := 3

theorem woman_work_completion_days : 
  (1 / work_completion_days_man + W + 1 / work_completion_days_boy = 1 / work_completion_days_combined) →
  W = 1 / 18 → 
  1 / W = 18 :=
by
  intros h₁ h₂
  sorry

end woman_work_completion_days_l9_9749


namespace remainder_of_a_squared_l9_9552

theorem remainder_of_a_squared (n : ℕ) (a : ℤ) (h : a % n * a % n % n = 1) : (a * a) % n = 1 := by
  sorry

end remainder_of_a_squared_l9_9552


namespace opposite_of_2023_l9_9335

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l9_9335


namespace quadratic_intersects_x_axis_l9_9940

theorem quadratic_intersects_x_axis (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 7 * x - 7 = 0) ↔ k ≥ -7/4 ∧ k ≠ 0 :=
by
  sorry

end quadratic_intersects_x_axis_l9_9940


namespace opposite_of_2023_l9_9295

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l9_9295


namespace cos_two_pi_over_three_l9_9479

theorem cos_two_pi_over_three : Real.cos (2 * Real.pi / 3) = -1 / 2 :=
by sorry

end cos_two_pi_over_three_l9_9479


namespace tan_2x_period_l9_9705

-- Define the tangent function and its properties
def tan := Real.tan

-- Define the problem
theorem tan_2x_period :
  (∃ P > 0, ∀ x, tan (2 * x) = tan (2 * x + P)) → P = π / 2 :=
by
  sorry

end tan_2x_period_l9_9705


namespace two_pow_65537_mod_19_l9_9482

theorem two_pow_65537_mod_19 : (2 ^ 65537) % 19 = 2 := by
  -- We will use Fermat's Little Theorem and given conditions.
  sorry

end two_pow_65537_mod_19_l9_9482


namespace independent_variable_range_l9_9856

theorem independent_variable_range (x : ℝ) :
  (∃ y : ℝ, y = 1 / (Real.sqrt x - 1)) ↔ x ≥ 0 ∧ x ≠ 1 := 
by
  sorry

end independent_variable_range_l9_9856


namespace percent_students_in_range_l9_9895

theorem percent_students_in_range
    (n1 n2 n3 n4 n5 : ℕ)
    (h1 : n1 = 5)
    (h2 : n2 = 7)
    (h3 : n3 = 8)
    (h4 : n4 = 4)
    (h5 : n5 = 3) :
  ((n3 : ℝ) / (n1 + n2 + n3 + n4 + n5) * 100) = 29.63 :=
by
  sorry

end percent_students_in_range_l9_9895


namespace degree_of_polynomial_l9_9011

theorem degree_of_polynomial (p : ℕ) (n : ℕ) (m : ℕ) (h : p = 3) (k : n = 15) :
  m = p * n := by
  sorry

-- Given p = 3 (degree of 5x^3) and n = 15 (exponent in (5x^3 + 7)^15)
-- Prove that m = 45 (degree of (5x^3 + 7)^15)
noncomputable def main_theorem : Prop :=
  (degree_of_polynomial 3 15 45 rfl rfl)

end degree_of_polynomial_l9_9011


namespace cos_double_angle_sin_double_angle_l9_9804

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 1/2) : Real.cos (2 * θ) = -1/2 :=
by sorry

theorem sin_double_angle (θ : ℝ) (h : Real.cos θ = 1/2) : Real.sin (2 * θ) = (Real.sqrt 3) / 2 :=
by sorry

end cos_double_angle_sin_double_angle_l9_9804


namespace opposite_of_2023_is_neg_2023_l9_9283

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l9_9283


namespace opposite_of_2023_l9_9160

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l9_9160


namespace opposite_of_2023_l9_9189

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9189


namespace candy_problem_l9_9616

-- Define the given conditions
def numberOfStudents : Nat := 43
def piecesOfCandyPerStudent : Nat := 8

-- Formulate the problem statement
theorem candy_problem : numberOfStudents * piecesOfCandyPerStudent = 344 := by
  sorry

end candy_problem_l9_9616


namespace opposite_of_2023_l9_9400

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9400


namespace SUCCESS_rearrangement_l9_9531

theorem SUCCESS_rearrangement: 
  let vowels := ['U', 'E'],
      consonants := ['S', 'S', 'S', 'C', 'C'] in
  (vowels.length.factorial) * (consonants.length.factorial / (3.factorial * 2.factorial)) = 20 :=
by
  sorry

end SUCCESS_rearrangement_l9_9531


namespace correct_factorization_l9_9734

theorem correct_factorization :
  (∀ a b : ℝ, ¬ (a^2 + b^2 = (a + b) * (a - b))) ∧
  (∀ a : ℝ, ¬ (a^4 - 1 = (a^2 + 1) * (a^2 - 1))) ∧
  (∀ x : ℝ, ¬ (x^2 + 2 * x + 4 = (x + 2)^2)) ∧
  (∀ x : ℝ, x^2 - 3 * x + 2 = (x - 1) * (x - 2)) :=
by
  sorry

end correct_factorization_l9_9734


namespace ticket_cost_correct_l9_9112

def metro_sells (tickets_per_minute : ℕ) (minutes : ℕ) : ℕ :=
  tickets_per_minute * minutes

def total_earnings (tickets_sold : ℕ) (ticket_cost : ℕ) : ℕ :=
  tickets_sold * ticket_cost

theorem ticket_cost_correct (ticket_cost : ℕ) : 
  (metro_sells 5 6 = 30) ∧ (total_earnings 30 ticket_cost = 90) → ticket_cost = 3 :=
by
  intro h
  sorry

end ticket_cost_correct_l9_9112


namespace painters_completing_rooms_l9_9816

theorem painters_completing_rooms (three_painters_three_rooms_three_hours : 3 * 3 * 3 ≥ 3 * 3) :
  9 * 3 * 9 ≥ 9 * 27 :=
by 
  sorry

end painters_completing_rooms_l9_9816


namespace P_inter_Q_eq_l9_9480

def P (x : ℝ) : Prop := -1 < x ∧ x < 3
def Q (x : ℝ) : Prop := -2 < x ∧ x < 1

theorem P_inter_Q_eq : {x | P x} ∩ {x | Q x} = {x : ℝ | -1 < x ∧ x < 1} :=
by
  sorry

end P_inter_Q_eq_l9_9480


namespace third_smallest_triangular_square_l9_9880

theorem third_smallest_triangular_square :
  ∃ n : ℕ, n = 1225 ∧ 
           (∃ x y : ℕ, y^2 - 8 * x^2 = 1 ∧ 
                        y = 99 ∧ x = 35) :=
by
  sorry

end third_smallest_triangular_square_l9_9880


namespace opposite_of_2023_l9_9404

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9404


namespace cos2alpha_plus_sin2alpha_l9_9782

theorem cos2alpha_plus_sin2alpha (α : Real) (h : Real.tan (Real.pi + α) = 2) : 
  Real.cos (2 * α) + Real.sin (2 * α) = 1 / 5 :=
sorry

end cos2alpha_plus_sin2alpha_l9_9782


namespace opposite_of_2023_l9_9435

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l9_9435


namespace number_of_bought_bottle_caps_l9_9099

/-- Define the initial number of bottle caps and the final number of bottle caps --/
def initial_bottle_caps : ℕ := 40
def final_bottle_caps : ℕ := 47

/-- Proof that the number of bottle caps Joshua bought is equal to 7 --/
theorem number_of_bought_bottle_caps : final_bottle_caps - initial_bottle_caps = 7 :=
by
  sorry

end number_of_bought_bottle_caps_l9_9099


namespace correct_operation_l9_9592

variable (a b : ℝ)

theorem correct_operation (h1 : a^2 + a^3 ≠ a^5)
                          (h2 : (-a^2)^3 ≠ a^6)
                          (h3 : -2*a^3*b / (a*b) ≠ -2*a^2*b) :
                          a^2 * a^3 = a^5 :=
by sorry

end correct_operation_l9_9592


namespace opposite_of_2023_l9_9370

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l9_9370


namespace surface_area_ratio_l9_9792

noncomputable def surface_area (r : ℝ) : ℝ :=
  4 * Real.pi * r ^ 2

theorem surface_area_ratio (k : ℝ) :
  let r1 := k
  let r2 := 2 * k
  let r3 := 3 * k
  let A1 := surface_area r1
  let A2 := surface_area r2
  let A3 := surface_area r3
  A3 / (A1 + A2) = 9 / 5 :=
by
  sorry

end surface_area_ratio_l9_9792


namespace min_value_x_y_l9_9082

theorem min_value_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 4/y = 1) : x + y ≥ 9 :=
sorry

end min_value_x_y_l9_9082


namespace minimum_period_tan_2x_l9_9706

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x)

theorem minimum_period_tan_2x : ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = Real.pi / 2 :=
by
  sorry

end minimum_period_tan_2x_l9_9706


namespace simplify_expression_l9_9598

noncomputable def expr := (-1 : ℝ)^2023 + Real.sqrt 9 - Real.pi^0 + Real.sqrt (1 / 8) * Real.sqrt 32

theorem simplify_expression : expr = 3 := 
by sorry

end simplify_expression_l9_9598


namespace opposite_of_2023_is_neg2023_l9_9354

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l9_9354


namespace find_quadratic_polynomial_l9_9637

-- Given conditions to construct a quadratic polynomial with real coefficients
noncomputable def quadratic_polynomial (a b c : ℂ) (h1 : a = 5 + 2 * complex.I) : polynomial ℂ :=
  3 * (X - C a) * (X - C (conj a))

-- The required proof problem statement
theorem find_quadratic_polynomial (x : ℂ) :
    quadratic_polynomial 5 2 0 rfl = 3 * X^2 - 30 * X + 87 :=
sorry

end find_quadratic_polynomial_l9_9637


namespace opposite_of_2023_l9_9340

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l9_9340


namespace first_day_more_than_200_paperclips_l9_9561

def paperclips_after_days (k : ℕ) : ℕ :=
  3 * 2^k

theorem first_day_more_than_200_paperclips : (∀ k, 3 * 2^k <= 200) → k <= 7 → 3 * 2^7 > 200 → k = 7 :=
by
  intro h_le h_lt h_gt
  sorry

end first_day_more_than_200_paperclips_l9_9561


namespace train_length_l9_9020

-- Definitions based on conditions
def faster_train_speed := 46 -- speed in km/hr
def slower_train_speed := 36 -- speed in km/hr
def time_to_pass := 72 -- time in seconds
def relative_speed_kmph := faster_train_speed - slower_train_speed
def relative_speed_mps : ℚ := (relative_speed_kmph * 1000) / 3600

theorem train_length :
  ∃ L : ℚ, (2 * L = relative_speed_mps * time_to_pass / 1) ∧ L = 100 := 
by
  sorry

end train_length_l9_9020


namespace minuend_is_not_integer_l9_9588

theorem minuend_is_not_integer (M S D : ℚ) (h1 : M + S + D = 555) (h2 : M - S = D) : ¬ ∃ n : ℤ, M = n := 
by
  sorry

end minuend_is_not_integer_l9_9588


namespace mid_point_between_fractions_l9_9057

theorem mid_point_between_fractions : (1 / 12 + 1 / 20) / 2 = 1 / 15 := by
  sorry

end mid_point_between_fractions_l9_9057


namespace find_mn_expression_l9_9062

-- Define the conditions
variables (m n : ℤ)
axiom abs_m_eq_3 : |m| = 3
axiom abs_n_eq_2 : |n| = 2
axiom m_lt_n : m < n

-- State the problem
theorem find_mn_expression : m^2 + m * n + n^2 = 7 ∨ m^2 + m * n + n^2 = 19 :=
by
  sorry

end find_mn_expression_l9_9062


namespace binomial_identity_l9_9506

theorem binomial_identity (n k : ℕ) (h1 : 0 < k) (h2 : k < n)
    (h3 : Nat.choose n (k-1) + Nat.choose n (k+1) = 2 * Nat.choose n k) :
  ∃ c : ℤ, k = (c^2 + c - 2) / 2 ∧ n = c^2 - 2 := sorry

end binomial_identity_l9_9506


namespace expected_number_of_sixes_when_three_dice_are_rolled_l9_9866

theorem expected_number_of_sixes_when_three_dice_are_rolled : 
  ∑ n in finset.range 4, (n * (↑(finset.filter (λ xs : fin 3 → fin 6, xs.count (λ x, x = 5) = n) finset.univ).card / 216 : ℚ)) = 1 / 2 :=
by
  -- Conclusion of proof is omitted as per instructions
  sorry

end expected_number_of_sixes_when_three_dice_are_rolled_l9_9866


namespace Ron_eats_24_pickle_slices_l9_9571

theorem Ron_eats_24_pickle_slices : 
  ∀ (pickle_slices_Sammy Tammy Ron : ℕ), 
    pickle_slices_Sammy = 15 → 
    Tammy = 2 * pickle_slices_Sammy → 
    Ron = Tammy - (20 * Tammy / 100) → 
    Ron = 24 := by
  intros pickle_slices_Sammy Tammy Ron h_sammy h_tammy h_ron
  sorry

end Ron_eats_24_pickle_slices_l9_9571


namespace rachel_picked_apples_l9_9998

-- Define relevant variables based on problem conditions
variable (trees : ℕ) (apples_per_tree : ℕ) (remaining_apples : ℕ)
variable (total_apples_picked : ℕ)

-- Assume the given conditions
axiom num_trees : trees = 4
axiom apples_each_tree : apples_per_tree = 7
axiom apples_left : remaining_apples = 29

-- Define the number of apples picked
def total_apples_picked_def := trees * apples_per_tree

-- State the theorem to prove the total apples picked
theorem rachel_picked_apples :
  total_apples_picked_def trees apples_per_tree = 28 :=
by
  -- Proof omitted
  sorry

end rachel_picked_apples_l9_9998


namespace opposite_of_2023_l9_9151

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l9_9151


namespace smallest_value_of_m_plus_n_l9_9583

theorem smallest_value_of_m_plus_n :
  ∃ m n : ℕ, 1 < m ∧ 
  (∃ l : ℝ, l = (m^2 - 1 : ℝ) / (m * n) ∧ l = 1 / 2021) ∧
  m + n = 85987 := 
sorry

end smallest_value_of_m_plus_n_l9_9583


namespace opposite_of_2023_l9_9437

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l9_9437


namespace last_digit_of_two_exp_sum_l9_9130

theorem last_digit_of_two_exp_sum (m : ℕ) (h : 0 < m) : 
  ((2 ^ (m + 2007) + 2 ^ (m + 1)) % 10) = 0 :=
by
  -- proof will go here
  sorry

end last_digit_of_two_exp_sum_l9_9130


namespace find_y_l9_9092

variable (x y : ℤ)

-- Conditions
def cond1 : Prop := x + y = 280
def cond2 : Prop := x - y = 200

-- Proof statement
theorem find_y (h1 : cond1 x y) (h2 : cond2 x y) : y = 40 := 
by 
  sorry

end find_y_l9_9092


namespace money_last_weeks_l9_9113

-- Define the conditions
def dollars_mowing : ℕ := 68
def dollars_weed_eating : ℕ := 13
def dollars_per_week : ℕ := 9

-- Define the total money made
def total_dollars := dollars_mowing + dollars_weed_eating

-- State the theorem to prove the question
theorem money_last_weeks : (total_dollars / dollars_per_week) = 9 :=
by
  sorry

end money_last_weeks_l9_9113


namespace soccer_ball_diameter_l9_9614

theorem soccer_ball_diameter 
  (h : ℝ)
  (s : ℝ)
  (d : ℝ)
  (h_eq : h = 1.25)
  (s_eq : s = 1)
  (d_eq : d = 0.23) : 2 * (d * h / (s - h)) = 0.46 :=
by
  sorry

end soccer_ball_diameter_l9_9614


namespace measure_of_angle_f_l9_9671

theorem measure_of_angle_f (angle_D angle_E angle_F : ℝ)
  (h1 : angle_D = 75)
  (h2 : angle_E = 4 * angle_F + 30)
  (h3 : angle_D + angle_E + angle_F = 180) : 
  angle_F = 15 :=
by
  sorry

end measure_of_angle_f_l9_9671


namespace difference_in_probabilities_is_twenty_percent_l9_9043

-- Definition of the problem conditions
def prob_win_first_lawsuit : ℝ := 0.30
def prob_lose_first_lawsuit : ℝ := 0.70
def prob_win_second_lawsuit : ℝ := 0.50
def prob_lose_second_lawsuit : ℝ := 0.50

-- We need to prove that the difference in probability of losing both lawsuits and winning both lawsuits is 20%
theorem difference_in_probabilities_is_twenty_percent :
  (prob_lose_first_lawsuit * prob_lose_second_lawsuit) -
  (prob_win_first_lawsuit * prob_win_second_lawsuit) = 0.20 := 
by
  sorry

end difference_in_probabilities_is_twenty_percent_l9_9043


namespace opposite_of_2023_l9_9363

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l9_9363


namespace opposite_of_2023_l9_9432

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l9_9432


namespace picnic_total_persons_l9_9607

-- Definitions based on given conditions
variables (W M A C : ℕ)
axiom cond1 : M = W + 80
axiom cond2 : A = C + 80
axiom cond3 : M = 120

-- Proof problem: Total persons = 240
theorem picnic_total_persons : W + M + A + C = 240 :=
by
  -- Proof will be filled here
  sorry

end picnic_total_persons_l9_9607


namespace team_sports_competed_l9_9091

theorem team_sports_competed (x : ℕ) (n : ℕ) 
  (h1 : (97 + n) / x = 90) 
  (h2 : (73 + n) / x = 87) : 
  x = 8 := 
by sorry

end team_sports_competed_l9_9091


namespace repeating_decimal_as_fraction_l9_9770

-- Given conditions
def repeating_decimal : ℚ := 7 + 832 / 999

-- Goal: Prove that the repeating decimal 7.\overline{832} equals 70/9
theorem repeating_decimal_as_fraction : repeating_decimal = 70 / 9 := by
  unfold repeating_decimal
  sorry

end repeating_decimal_as_fraction_l9_9770


namespace infant_weight_in_4th_month_l9_9826

-- Given conditions
def a : ℕ := 3000
def x : ℕ := 4
def y : ℕ := a + 700 * x

-- Theorem stating the weight of the infant in the 4th month equals 5800 grams
theorem infant_weight_in_4th_month : y = 5800 := by
  sorry

end infant_weight_in_4th_month_l9_9826


namespace M_inter_N_eq_l9_9103

-- Definitions based on the problem conditions
def M : Set ℝ := { x | abs x ≥ 3 }
def N : Set ℝ := { y | ∃ x ∈ M, y = x^2 }

-- The statement we want to prove
theorem M_inter_N_eq : M ∩ N = { x : ℝ | x ≥ 3 } :=
by
  sorry

end M_inter_N_eq_l9_9103


namespace geometric_diff_l9_9502

-- Definitions based on conditions
def is_geometric (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧ (d2 * d2 = d1 * d3)

-- Problem statement
theorem geometric_diff :
  let largest_geometric := 964
  let smallest_geometric := 124
  is_geometric largest_geometric ∧ is_geometric smallest_geometric ∧
  (largest_geometric - smallest_geometric = 840) :=
by
  sorry

end geometric_diff_l9_9502


namespace opposite_of_2023_is_neg2023_l9_9344

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l9_9344


namespace opposite_of_2023_l9_9194

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9194


namespace proof_problem_l9_9604

-- Definitions for the conditions and the events in the problem
def P_A : ℚ := 2 / 3
def P_B : ℚ := 1 / 4
def P_not_any_module : ℚ := 1 - (P_A + P_B)

-- Definition for the binomial coefficient
def C (n k : ℕ) := Nat.choose n k

-- Definition for the event where at least 3 out of 4 students have taken "Selected Topics in Geometric Proofs"
def P_at_least_three_taken : ℚ := 
  C 4 3 * (P_A ^ 3) * ((1 - P_A) ^ 1) + C 4 4 * (P_A ^ 4)

-- The main theorem to prove
theorem proof_problem : 
  P_not_any_module = 1 / 12 ∧ P_at_least_three_taken = 16 / 27 :=
by
  sorry

end proof_problem_l9_9604


namespace probability_increasing_function_g_l9_9931

theorem probability_increasing_function_g (a : ℝ) (h : 0 ≤ a ∧ a ≤ 10) : 
  ∃ p : ℝ, p = 1 / 5 ∧ (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → ∀ a, (0 ≤ a ∧ a < 2) → g a x₁ < g a x₂ ) :=
by
  let g (a x : ℝ) : ℝ := (a - 2) / x
  use 1 / 5
  sorry

end probability_increasing_function_g_l9_9931


namespace infinite_solutions_xyz_t_l9_9841

theorem infinite_solutions_xyz_t (x y z t : ℕ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) (h4 : t ≠ 0) (h5 : gcd (gcd x y) (gcd z t) = 1) :
  ∃ (x y z t : ℕ), x^3 + y^3 + z^3 = t^4 ∧ gcd (gcd x y) (gcd z t) = 1 :=
sorry

end infinite_solutions_xyz_t_l9_9841


namespace smallest_integer_in_consecutive_set_l9_9818

theorem smallest_integer_in_consecutive_set :
  ∃ (n : ℤ), 2 < n ∧ ∀ m : ℤ, m < n → ¬ (m + 6 < 2 * (m + 3) - 2) :=
sorry

end smallest_integer_in_consecutive_set_l9_9818


namespace calculate_total_prime_dates_l9_9903

-- Define the prime months
def prime_months : List Nat := [2, 3, 5, 7, 11, 13]

-- Define the number of days in each month for a non-leap year
def days_in_month (month : Nat) : Nat :=
  if month = 2 then 28
  else if month = 3 then 31
  else if month = 5 then 31
  else if month = 7 then 31
  else if month = 11 then 30
  else if month = 13 then 31
  else 0

-- Define the prime days in a month
def prime_days : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

-- Calculate the number of prime dates in a given month
def prime_dates_in_month (month : Nat) : Nat :=
  (prime_days.filter (λ d => d <= days_in_month month)).length

-- Calculate the total number of prime dates for the year
def total_prime_dates : Nat :=
  (prime_months.map prime_dates_in_month).sum

theorem calculate_total_prime_dates : total_prime_dates = 62 := by
  sorry

end calculate_total_prime_dates_l9_9903


namespace european_customer_savings_l9_9750

noncomputable def popcorn_cost : ℝ := 8 - 3
noncomputable def drink_cost : ℝ := popcorn_cost + 1
noncomputable def candy_cost : ℝ := drink_cost / 2

noncomputable def discounted_popcorn_cost : ℝ := popcorn_cost * (1 - 0.15)
noncomputable def discounted_candy_cost : ℝ := candy_cost * (1 - 0.1)

noncomputable def total_normal_cost : ℝ := 8 + discounted_popcorn_cost + drink_cost + discounted_candy_cost
noncomputable def deal_price : ℝ := 20
noncomputable def savings_in_dollars : ℝ := total_normal_cost - deal_price

noncomputable def exchange_rate : ℝ := 0.85
noncomputable def savings_in_euros : ℝ := savings_in_dollars * exchange_rate

theorem european_customer_savings : savings_in_euros = 0.81 := by
  sorry

end european_customer_savings_l9_9750


namespace line_passes_through_fixed_point_l9_9790

theorem line_passes_through_fixed_point (a b : ℝ) (x y : ℝ) (h : a + b = 1) (h1 : 2 * a * x - b * y = 1) : x = 1/2 ∧ y = -1 :=
by 
  sorry

end line_passes_through_fixed_point_l9_9790


namespace opposite_of_2023_l9_9218

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l9_9218


namespace binomial_9_pow5_eq_binomial_11_pow5_eq_pow_9_and_11_l9_9501

noncomputable def pow_9 : ℕ := 9^5
noncomputable def pow_11 : ℕ := 11^5

theorem binomial_9_pow5_eq :
  ∑ k in Finset.range 6, Nat.choose 5 k * 10^(5-k) * (-1)^k = 59149 := sorry

theorem binomial_11_pow5_eq :
  ∑ k in Finset.range 6, Nat.choose 5 k * 10^(5-k) * 1^k = 161051 := sorry

theorem pow_9_and_11 :
  pow_9 = 59149 ∧ pow_11 = 161051 :=
by
  unfold pow_9 pow_11
  apply And.intro
  · apply binomial_9_pow5_eq
  · apply binomial_11_pow5_eq

end binomial_9_pow5_eq_binomial_11_pow5_eq_pow_9_and_11_l9_9501


namespace initial_back_squat_weight_l9_9548

-- Define a structure to encapsulate the conditions
structure squat_conditions where
  initial_back_squat : ℝ
  front_squat_ratio : ℝ := 0.8
  back_squat_increase : ℝ := 50
  front_squat_triple_ratio : ℝ := 0.9
  total_weight_moved : ℝ := 540

-- Using the conditions provided to prove John's initial back squat weight
theorem initial_back_squat_weight (c : squat_conditions) :
  (3 * 3 * (c.front_squat_triple_ratio * (c.front_squat_ratio * c.initial_back_squat)) = c.total_weight_moved) →
  c.initial_back_squat = 540 / 6.48 := sorry

end initial_back_squat_weight_l9_9548


namespace half_is_greater_than_third_by_one_sixth_l9_9133

theorem half_is_greater_than_third_by_one_sixth : (0.5 : ℝ) - (1 / 3 : ℝ) = 1 / 6 := by
  sorry

end half_is_greater_than_third_by_one_sixth_l9_9133


namespace perpendicular_lines_condition_l9_9597

theorem perpendicular_lines_condition (k : ℝ) : 
  (k = 5 → (∃ x y : ℝ, k * x + 5 * y - 2 = 0 ∧ (4 - k) * x + y - 7 = 0 ∧ x ≠ 0 ∧ y ≠ 0)) ∧
  (∀ x y : ℝ, k * x + 5 * y - 2 = 0 ∧ (4 - k) * x + y - 7 = 0 → (k = 5 ∨ k = -1)) :=
sorry

end perpendicular_lines_condition_l9_9597


namespace mrs_berkeley_A_students_first_class_mrs_berkeley_A_students_extended_class_l9_9562

noncomputable def ratio_of_A_students (total_students_A : ℕ) (A_students_A : ℕ) : ℚ :=
  A_students_A / total_students_A

theorem mrs_berkeley_A_students_first_class :
  ∀ (total_students_A : ℕ) (A_students_A : ℕ) (total_students_B : ℕ),
    total_students_A = 30 →
    A_students_A = 20 →
    total_students_B = 18 →
    (A_students_A / total_students_A) * total_students_B = 12 :=
by
  intros total_students_A A_students_A total_students_B hA1 hA2 hB
  sorry

theorem mrs_berkeley_A_students_extended_class :
  ∀ (total_students_A : ℕ) (A_students_A : ℕ) (total_students_B : ℕ),
    total_students_A = 30 →
    A_students_A = 20 →
    total_students_B = 27 →
    (A_students_A / total_students_A) * total_students_B = 18 :=
by
  intros total_students_A A_students_A total_students_B hA1 hA2 hB
  sorry

end mrs_berkeley_A_students_first_class_mrs_berkeley_A_students_extended_class_l9_9562


namespace general_term_a_n_general_term_b_n_sum_of_first_n_terms_D_n_l9_9525

def seq_a : ℕ → ℕ 
| 0 => 0  -- a_0 is not defined in natural numbers, put it as zero for base case
| (n+1) => 2^(n+1)

def seq_b : ℕ → ℕ 
| 0 => 0  -- b_0 is not defined in natural numbers, put it as zero for base case
| (n+1) => 2*(n+1) -1

def sum_S (n : ℕ) : ℕ := (seq_a (n+1) * 2) - 2

def sum_T : ℕ → ℕ 
| 0 => 0  -- T_0 is not defined in natural numbers, put it as zero for base case too
| (n+1) => (n+1)^2

def sum_D : ℕ → ℕ
| 0 => 0
| (n+1) => (seq_a (n+1) * seq_b (n+1)) + sum_D n

theorem general_term_a_n (n : ℕ) : seq_a n = 2^n := sorry

theorem general_term_b_n (n : ℕ) : seq_b n = 2*n - 1 := sorry

theorem sum_of_first_n_terms_D_n (n : ℕ) : sum_D n = (2*n - 3)*2^(n+1) + 6 := sorry

end general_term_a_n_general_term_b_n_sum_of_first_n_terms_D_n_l9_9525


namespace compute_expression_l9_9811

theorem compute_expression (x : ℝ) (hx : x + 1 / x = 7) : 
  (x - 3)^2 + 36 / (x - 3)^2 = 12.375 := 
  sorry

end compute_expression_l9_9811


namespace opposite_of_2023_is_neg2023_l9_9325

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l9_9325


namespace smaller_number_is_22_l9_9718

noncomputable def smaller_number (x y : ℕ) : ℕ := 
x

theorem smaller_number_is_22 (x y : ℕ) (h1 : x + y = 56) (h2 : y = x + 12) : x = 22 :=
by
  sorry

end smaller_number_is_22_l9_9718


namespace opposite_of_2023_is_neg_2023_l9_9287

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l9_9287


namespace opposite_of_2023_l9_9254

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l9_9254


namespace opposite_of_2023_l9_9228

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l9_9228


namespace income_of_deceased_l9_9594

def average_income (total_income : ℕ) (members : ℕ) : ℕ :=
  total_income / members

theorem income_of_deceased
  (total_income_before : ℕ) (members_before : ℕ) (avg_income_before : ℕ)
  (total_income_after : ℕ) (members_after : ℕ) (avg_income_after : ℕ) :
  total_income_before = members_before * avg_income_before →
  total_income_after = members_after * avg_income_after →
  members_before = 4 →
  members_after = 3 →
  avg_income_before = 735 →
  avg_income_after = 650 →
  total_income_before - total_income_after = 990 :=
by
  sorry

end income_of_deceased_l9_9594


namespace opposite_of_2023_l9_9298

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l9_9298


namespace opposite_of_2023_l9_9251

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l9_9251


namespace odd_function_condition_l9_9908

noncomputable def f (x a b : ℝ) : ℝ := x * abs (x + a) + b

theorem odd_function_condition (a b : ℝ) :
  (∀ x : ℝ, f (-x) a b = -f x a b) ↔ (a = 0 ∧ b = 0) :=
by
  sorry

end odd_function_condition_l9_9908


namespace algebraic_expression_value_l9_9961

theorem algebraic_expression_value (x : ℝ) (h : 2 * x^2 + 3 * x + 7 = 8) : 4 * x^2 + 6 * x - 9 = -7 :=
by
  sorry

end algebraic_expression_value_l9_9961


namespace expression_value_l9_9932

theorem expression_value :
  3 * 12^2 - 3 * 13 + 2 * 16 * 11^2 = 4265 :=
by
  sorry

end expression_value_l9_9932


namespace translate_down_by_2_l9_9877

theorem translate_down_by_2 (x y : ℝ) (h : y = -2 * x + 3) : y - 2 = -2 * x + 1 := 
by 
  sorry

end translate_down_by_2_l9_9877


namespace opposite_of_2023_l9_9417

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l9_9417


namespace zhang_san_not_losing_probability_l9_9766

theorem zhang_san_not_losing_probability (p_win p_draw : ℚ) (h_win : p_win = 1 / 3) (h_draw : p_draw = 1 / 4) : 
  p_win + p_draw = 7 / 12 := by
  sorry

end zhang_san_not_losing_probability_l9_9766


namespace absent_children_count_l9_9993

theorem absent_children_count (total_children : ℕ) (bananas_per_child : ℕ) (extra_bananas_per_child : ℕ)
    (absent_children : ℕ) (total_bananas : ℕ) (present_children : ℕ) :
    total_children = 640 →
    bananas_per_child = 2 →
    extra_bananas_per_child = 2 →
    total_bananas = (total_children * bananas_per_child) →
    present_children = (total_children - absent_children) →
    total_bananas = (present_children * (bananas_per_child + extra_bananas_per_child)) →
    absent_children = 320 := by
  intros h1 h2 h3 h4 h5 h6
  sorry

end absent_children_count_l9_9993


namespace b_range_l9_9528

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := Real.log x + b / (x + 1)

theorem b_range (b : ℝ) (hb : 0 < b)
  (h : ∀ x1 x2 : ℝ, 1 ≤ x1 ∧ x1 ≤ 2 → 1 ≤ x2 ∧ x2 ≤ 2 → x1 ≠ x2 → (f x1 b - f x2 b) / (x1 - x2) < -1) :
  b > 27 / 2 :=
sorry

end b_range_l9_9528


namespace opposite_of_2023_is_neg_2023_l9_9232

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l9_9232


namespace opposite_of_2023_is_neg2023_l9_9319

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l9_9319


namespace necessary_but_not_sufficient_l9_9059

def p (a : ℝ) : Prop := ∃ (x : ℝ), x^2 + 2 * a * x - a ≤ 0

def q (a : ℝ) : Prop := a > 0 ∨ a < -1

theorem necessary_but_not_sufficient (a : ℝ) : (∃ (x : ℝ), x^2 + 2 * a * x - a ≤ 0) → (a > 0 ∨ a < -1) ∧ ¬((a > 0 ∨ a < -1) → (∃ (x : ℝ), x^2 + 2 * a * x - a ≤ 0)) :=
by
  sorry

end necessary_but_not_sufficient_l9_9059


namespace fraction_upgraded_l9_9897

theorem fraction_upgraded :
  ∀ (N U : ℕ), 24 * N = 6 * U → (U : ℚ) / (24 * N + U) = 1 / 7 :=
by
  intros N U h_eq
  sorry

end fraction_upgraded_l9_9897


namespace triangle_side_lengths_l9_9820

theorem triangle_side_lengths (a b c : ℝ) (h1 : a / b = 3 / 4) (h2 : b / c = 4 / 5) (h3 : a + b + c = 60) :
  (a = 15 ∧ b = 20 ∧ c = 25) :=
sorry

end triangle_side_lengths_l9_9820


namespace sun_city_population_l9_9845

theorem sun_city_population (W R S : ℕ) (h1 : W = 2000)
    (h2 : R = 3 * W - 500) (h3 : S = 2 * R + 1000) : S = 12000 :=
by
    -- Use the provided conditions (h1, h2, h3) to state the theorem
    sorry

end sun_city_population_l9_9845


namespace opposite_of_2023_is_neg_2023_l9_9280

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l9_9280


namespace percentage_more_likely_to_lose_both_l9_9042

def first_lawsuit_win_probability : ℝ := 0.30
def first_lawsuit_lose_probability : ℝ := 0.70
def second_lawsuit_win_probability : ℝ := 0.50
def second_lawsuit_lose_probability : ℝ := 0.50

theorem percentage_more_likely_to_lose_both :
  (second_lawsuit_lose_probability * first_lawsuit_lose_probability - second_lawsuit_win_probability * first_lawsuit_win_probability) / (second_lawsuit_win_probability * first_lawsuit_win_probability) * 100 = 133.33 :=
by
  sorry

end percentage_more_likely_to_lose_both_l9_9042


namespace convex_pentadecagon_diagonals_l9_9050

def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem convex_pentadecagon_diagonals :
  number_of_diagonals 15 = 90 :=
by sorry

end convex_pentadecagon_diagonals_l9_9050


namespace opposite_of_2023_l9_9304

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l9_9304


namespace total_slices_left_is_14_l9_9628

-- Define the initial conditions
def large_pizza_slices : ℕ := 12
def small_pizza_slices : ℕ := 8
def hawaiian_pizza (num_large : ℕ) : ℕ := num_large * large_pizza_slices
def cheese_pizza (num_large : ℕ) : ℕ := num_large * large_pizza_slices
def pepperoni_pizza (num_small : ℕ) : ℕ := num_small * small_pizza_slices

-- Number of large pizzas ordered (Hawaiian and cheese)
def num_large_pizzas : ℕ := 2

-- Number of small pizzas received in promotion
def num_small_pizzas : ℕ := 1

-- Slices eaten by each person
def dean_slices (hawaiian_slices : ℕ) : ℕ := hawaiian_slices / 2
def frank_slices : ℕ := 3
def sammy_slices (cheese_slices : ℕ) : ℕ := cheese_slices / 3
def nancy_cheese_slices : ℕ := 2
def nancy_pepperoni_slice : ℕ := 1
def olivia_slices : ℕ := 2

-- Total slices eaten from each pizza
def total_hawaiian_slices_eaten (hawaiian_slices : ℕ) : ℕ := dean_slices hawaiian_slices + frank_slices
def total_cheese_slices_eaten (cheese_slices : ℕ) : ℕ := sammy_slices cheese_slices + nancy_cheese_slices
def total_pepperoni_slices_eaten : ℕ := nancy_pepperoni_slice + olivia_slices

-- Total slices left over
def total_slices_left (hawaiian_slices : ℕ) (cheese_slices : ℕ) (pepperoni_slices : ℕ) : ℕ := 
  (hawaiian_slices - total_hawaiian_slices_eaten hawaiian_slices) + 
  (cheese_slices - total_cheese_slices_eaten cheese_slices) + 
  (pepperoni_slices - total_pepperoni_slices_eaten)

-- The actual Lean 4 statement to be verified
theorem total_slices_left_is_14 : total_slices_left (hawaiian_pizza num_large_pizzas) (cheese_pizza num_large_pizzas) (pepperoni_pizza num_small_pizzas) = 14 := 
  sorry

end total_slices_left_is_14_l9_9628


namespace opposite_of_2023_l9_9360

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l9_9360


namespace HorseKeepsPower_l9_9957

/-- If the Little Humpbacked Horse does not eat for seven days or does not sleep for seven days,
    he will lose his magic power. Suppose he did not eat or sleep for a whole week. 
    Prove that by the end of the seventh day, he must do the activity he did not do right before 
    the start of the first period of seven days in order to keep his power. -/
theorem HorseKeepsPower (eat sleep : ℕ → Prop) :
  (∀ (n : ℕ), (n ≥ 7 → ¬eat n) ∨ (n ≥ 7 → ¬sleep n)) →
  (∀ (n : ℕ), n < 7 → (¬eat n ∧ ¬sleep n)) →
  ∃ (t : ℕ), t > 7 → (eat t ∨ sleep t) :=
sorry

end HorseKeepsPower_l9_9957


namespace trapezoid_leg_length_proof_l9_9695

noncomputable def circumscribed_trapezoid_leg_length 
  (area : ℝ) (acute_angle_base : ℝ) : ℝ :=
  -- Hypothesis: Given conditions of the problem
  if h : area = 32 * Real.sqrt 3 ∧ acute_angle_base = Real.pi / 3 then
    -- The length of the trapezoid's leg
    8
  else
    0

-- Statement of the proof problem
theorem trapezoid_leg_length_proof 
  (area : ℝ) (acute_angle_base : ℝ)
  (h : area = 32 * Real.sqrt 3 ∧ acute_angle_base = Real.pi / 3) :
  circumscribed_trapezoid_leg_length area acute_angle_base = 8 := 
by {
  -- skipping actual proof
  sorry
}

end trapezoid_leg_length_proof_l9_9695


namespace opposite_of_2023_l9_9301

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l9_9301


namespace opposite_of_2023_l9_9248

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l9_9248


namespace enclosed_area_of_curve_l9_9850

noncomputable def radius_of_arcs := 1

noncomputable def arc_length := (1 / 2) * Real.pi

noncomputable def side_length_of_octagon := 3

noncomputable def area_of_octagon (s : ℝ) := 
  2 * (1 + Real.sqrt 2) * s ^ 2

noncomputable def area_of_sectors (n : ℕ) (arc_radius : ℝ) (arc_theta : ℝ) := 
  n * (1 / 4) * Real.pi

theorem enclosed_area_of_curve : 
  area_of_octagon side_length_of_octagon + area_of_sectors 12 radius_of_arcs arc_length 
  = 54 + 54 * Real.sqrt 2 + 3 * Real.pi := 
by
  sorry

end enclosed_area_of_curve_l9_9850


namespace minimum_people_to_save_cost_l9_9111

-- Define the costs for the two event planners.
def cost_first_planner (x : ℕ) : ℕ := 120 + 18 * x
def cost_second_planner (x : ℕ) : ℕ := 250 + 15 * x

-- State the theorem to prove the minimum number of people required for the second event planner to be less expensive.
theorem minimum_people_to_save_cost : ∃ x : ℕ, cost_second_planner x < cost_first_planner x ∧ ∀ y : ℕ, y < x → cost_second_planner y ≥ cost_first_planner y :=
sorry

end minimum_people_to_save_cost_l9_9111


namespace c_impossible_value_l9_9516

theorem c_impossible_value (a b c : ℤ) (h : (∀ x : ℤ, (x + a) * (x + b) = x^2 + c * x - 8)) : c ≠ 4 :=
by
  sorry

end c_impossible_value_l9_9516


namespace kristin_annual_income_l9_9715

theorem kristin_annual_income (p : ℝ) :
  ∃ A : ℝ, 
  (0.01 * p * 28000 + 0.01 * (p + 2) * (A - 28000) = (0.01 * (p + 0.25) * A)) ∧
  A = 32000 :=
by
  sorry

end kristin_annual_income_l9_9715


namespace sum_of_nine_consecutive_parity_l9_9828

theorem sum_of_nine_consecutive_parity (n : ℕ) : 
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) + (n + 7) + (n + 8)) % 2 = n % 2 := 
  sorry

end sum_of_nine_consecutive_parity_l9_9828


namespace opposite_of_2023_l9_9181

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l9_9181


namespace negation_example_l9_9132

theorem negation_example :
  (¬ (∀ a : ℕ, a > 0 → 2^a ≥ a^2)) ↔ (∃ a : ℕ, a > 0 ∧ 2^a < a^2) :=
by sorry

end negation_example_l9_9132


namespace flour_needed_for_one_batch_l9_9994

theorem flour_needed_for_one_batch (F : ℝ) (h1 : 8 * F + 8 * 1.5 = 44) : F = 4 := 
by
    sorry

end flour_needed_for_one_batch_l9_9994


namespace opposite_of_2023_l9_9262

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9262


namespace range_of_m_l9_9945

-- Define the two vectors a and b
def vector_a := (1, 2)
def vector_b (m : ℝ) := (m, 3 * m - 2)

-- Define the condition for non-collinearity
def non_collinear (m : ℝ) := ¬ (m / 1 = (3 * m - 2) / 2)

theorem range_of_m (m : ℝ) : non_collinear m ↔ m ≠ 2 :=
  sorry

end range_of_m_l9_9945


namespace base8_digits_sum_l9_9810

-- Define digits and their restrictions
variables {A B C : ℕ}

-- Main theorem
theorem base8_digits_sum (h1 : 0 < A ∧ A < 8)
                         (h2 : 0 < B ∧ B < 8)
                         (h3 : 0 < C ∧ C < 8)
                         (distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
                         (condition : (8^2 * A + 8 * B + C) + (8^2 * B + 8 * C + A) + (8^2 * C + 8 * A + B) = (8^2 + 8 + 1) * 8 * A) :
  A + B + C = 8 := 
sorry

end base8_digits_sum_l9_9810


namespace sum_of_two_integers_l9_9861

theorem sum_of_two_integers (x y : ℝ) (h₁ : x^2 + y^2 = 130) (h₂ : x * y = 45) : x + y = 2 * Real.sqrt 55 :=
sorry

end sum_of_two_integers_l9_9861


namespace weight_of_B_l9_9893

/-- Let A, B, and C be the weights in kg of three individuals. If the average weight of A, B, and C is 45 kg,
and the average weight of A and B is 41 kg, and the average weight of B and C is 43 kg,
then the weight of B is 33 kg. -/
theorem weight_of_B (A B C : ℝ) 
  (h1 : A + B + C = 135) 
  (h2 : A + B = 82) 
  (h3 : B + C = 86) : 
  B = 33 := 
by 
  sorry

end weight_of_B_l9_9893


namespace sum_of_first_49_odd_numbers_l9_9892

theorem sum_of_first_49_odd_numbers : 
  let seq : List ℕ := List.range' 1 (2 * 49 - 1) 2 in
  seq.sum = 2401 :=
by
  sorry

end sum_of_first_49_odd_numbers_l9_9892


namespace power_sum_greater_than_linear_l9_9066

theorem power_sum_greater_than_linear (x : ℝ) (n : ℕ) (hx1 : x > -1) (hx2 : x ≠ 0) (hn : n ≥ 2) :
  (1 + x) ^ n > 1 + n * x :=
sorry

end power_sum_greater_than_linear_l9_9066


namespace range_of_m_l9_9781

variable {x m : ℝ}
variable (q: ℝ → Prop) (p: ℝ → Prop)

-- Definition of q
def q_cond : Prop := (x - (1 + m)) * (x - (1 - m)) ≤ 0

-- Definition of p
def p_cond : Prop := |1 - (x - 1) / 3| ≤ 2

-- Statement of the proof problem
theorem range_of_m (h1 : ∀ x, q x → p x) (h2 : ∃ x, ¬p x → q x) 
  (h3 : m > 0) :
  0 < m ∧ m ≤ 3 :=
by
  sorry

end range_of_m_l9_9781


namespace max_mn_value_min_4m_square_n_square_l9_9659

variable {m n : ℝ}
variable (h_cond1 : m > 0)
variable (h_cond2 : n > 0)
variable (h_eq : 2 * m + n = 1)

theorem max_mn_value : (∃ m n : ℝ, m > 0 ∧ n > 0 ∧ 2 * m + n = 1 ∧ m * n = 1/8) := 
  sorry

theorem min_4m_square_n_square : (∃ m n : ℝ, m > 0 ∧ n > 0 ∧ 2 * m + n = 1 ∧ 4 * m^2 + n^2 = 1/2) := 
  sorry

end max_mn_value_min_4m_square_n_square_l9_9659


namespace company_employee_count_l9_9894

/-- 
 Given the employees are divided into three age groups: A, B, and C, with a ratio of 5:4:1,
 a stratified sampling method is used to draw a sample of size 20 from the population,
 and the probability of selecting both person A and person B from group C is 1/45.
 Prove the total number of employees in the company is 100.
-/
theorem company_employee_count :
  ∃ (total_employees : ℕ),
    (∃ (ratio_A : ℕ) (ratio_B : ℕ) (ratio_C : ℕ),
      ratio_A = 5 ∧ 
      ratio_B = 4 ∧ 
      ratio_C = 1 ∧
      ∃ (sample_size : ℕ), 
        sample_size = 20 ∧
        ∃ (prob_selecting_two_from_C : ℚ),
          prob_selecting_two_from_C = 1 / 45 ∧
          total_employees = 100) :=
sorry

end company_employee_count_l9_9894


namespace william_wins_tic_tac_toe_l9_9888

-- Define the conditions
variables (total_rounds : ℕ) (extra_wins : ℕ) (william_wins : ℕ) (harry_wins : ℕ)

-- Setting the conditions
def william_harry_tic_tac_toe_conditions : Prop :=
  total_rounds = 15 ∧
  extra_wins = 5 ∧
  william_wins = harry_wins + extra_wins ∧
  total_rounds = william_wins + harry_wins

-- The goal is to prove that William won 10 rounds given the conditions above
theorem william_wins_tic_tac_toe : william_harry_tic_tac_toe_conditions total_rounds extra_wins william_wins harry_wins → william_wins = 10 :=
by
  intro h
  have total_rounds_eq := and.left h
  have extra_wins_eq := and.right (and.left (and.right h))
  have william_harry_diff := and.left (and.right (and.right h))
  have total_wins_eq := and.right (and.right (and.right h))
  sorry

end william_wins_tic_tac_toe_l9_9888


namespace total_nephews_correct_l9_9495

def alden_nephews_10_years_ago : ℕ := 50

def alden_nephews_now : ℕ :=
  alden_nephews_10_years_ago * 2

def vihaan_nephews_now : ℕ :=
  alden_nephews_now + 60

def total_nephews : ℕ :=
  alden_nephews_now + vihaan_nephews_now

theorem total_nephews_correct : total_nephews = 260 := by
  sorry

end total_nephews_correct_l9_9495


namespace alice_leaves_30_minutes_after_bob_l9_9852

theorem alice_leaves_30_minutes_after_bob :
  ∀ (distance : ℝ) (speed_bob : ℝ) (speed_alice : ℝ) (time_diff : ℝ),
  distance = 220 ∧ speed_bob = 40 ∧ speed_alice = 44 ∧ 
  time_diff = (distance / speed_bob) - (distance / speed_alice) →
  (time_diff * 60 = 30) := by
  intro distance speed_bob speed_alice time_diff
  intro h
  have h1 : distance = 220 := h.1
  have h2 : speed_bob = 40 := h.2.1
  have h3 : speed_alice = 44 := h.2.2.1
  have h4 : time_diff = (distance / speed_bob) - (distance / speed_alice) := h.2.2.2
  sorry

end alice_leaves_30_minutes_after_bob_l9_9852


namespace mk97_x_eq_one_l9_9131

noncomputable def mk97_initial_number (x : ℝ) : Prop := 
  x ≠ 0 ∧ 4 * (x^2 - x) = 0

theorem mk97_x_eq_one (x : ℝ) (h : mk97_initial_number x) : x = 1 := by
  sorry

end mk97_x_eq_one_l9_9131


namespace quadratic_polynomial_real_coeff_l9_9639

theorem quadratic_polynomial_real_coeff (a b : ℂ) (h₁ : a = 5 + 2*i) 
  (h₂ : b = 5 - 2*i) (c : ℂ) (hc : c = 3) :
  3 * (X - C a) * (X - C b) = 3*X^2 - 30*X + 87 := 
by {
  sorry
}

end quadratic_polynomial_real_coeff_l9_9639


namespace visitors_inversely_proportional_l9_9694

theorem visitors_inversely_proportional (k : ℝ) (v₁ v₂ t₁ t₂ : ℝ) (h1 : v₁ * t₁ = k) (h2 : t₁ = 20) (h3 : v₁ = 150) (h4 : t₂ = 30) : v₂ = 100 :=
by
  -- This is a placeholder line; the actual proof would go here.
  sorry

end visitors_inversely_proportional_l9_9694


namespace range_of_m_l9_9800

theorem range_of_m (x y m : ℝ) 
  (h1: 3 * x + y = 1 + 3 * m) 
  (h2: x + 3 * y = 1 - m) 
  (h3: x + y > 0) : 
  m > -1 :=
sorry

end range_of_m_l9_9800


namespace find_smaller_number_l9_9860

theorem find_smaller_number (x : ℕ) (h : 3 * x + 4 * x = 420) : 3 * x = 180 :=
by
  sorry

end find_smaller_number_l9_9860


namespace smallest_m_n_sum_l9_9581

noncomputable def smallestPossibleSum (m n : ℕ) : ℕ :=
  m + n

theorem smallest_m_n_sum :
  ∃ (m n : ℕ), (m > 1) ∧ (m * n * (2021 * (m^2 - 1)) = 2021 * m * m * n) ∧ smallestPossibleSum m n = 4323 :=
by
  sorry

end smallest_m_n_sum_l9_9581


namespace expand_product_l9_9052

theorem expand_product (x : ℝ) (hx : x ≠ 0) : (3 / 7) * (7 / x - 5 * x ^ 3) = 3 / x - (15 / 7) * x ^ 3 :=
by
  sorry

end expand_product_l9_9052


namespace statement_1_statement_2_statement_3_all_statements_correct_l9_9642

-- Define the function f and the axioms/conditions given in the problem
def f : ℕ → ℕ → ℕ := sorry

-- Conditions
axiom f_initial : f 1 1 = 1
axiom f_nat : ∀ m n : ℕ, m > 0 → n > 0 → f m n > 0
axiom f_condition_1 : ∀ m n : ℕ, m > 0 → n > 0 → f m (n + 1) = f m n + 2
axiom f_condition_2 : ∀ m : ℕ, m > 0 → f (m + 1) 1 = 2 * f m 1

-- Statements to be proved
theorem statement_1 : f 1 5 = 9 := sorry
theorem statement_2 : f 5 1 = 16 := sorry
theorem statement_3 : f 5 6 = 26 := sorry

theorem all_statements_correct : (f 1 5 = 9) ∧ (f 5 1 = 16) ∧ (f 5 6 = 26) := by
  exact ⟨statement_1, statement_2, statement_3⟩

end statement_1_statement_2_statement_3_all_statements_correct_l9_9642


namespace annual_income_is_32000_l9_9714

noncomputable def compute_tax (p A: ℝ) : ℝ := 
  0.01 * p * 28000 + 0.01 * (p + 2) * (A - 28000)

noncomputable def stated_tax (p A: ℝ) : ℝ := 
  0.01 * (p + 0.25) * A

theorem annual_income_is_32000 (p : ℝ) (A : ℝ) :
  compute_tax p A = stated_tax p A → A = 32000 :=
by
  intros h
  have : 0.01 * p * 28000 + 0.01 * (p + 2) * (A - 28000) = 0.01 * (p + 0.25) * A := h
  sorry

end annual_income_is_32000_l9_9714


namespace mixed_groups_count_l9_9727

theorem mixed_groups_count 
    (total_children : ℕ)
    (total_groups : ℕ)
    (group_size : ℕ)
    (total_games : ℕ)
    (boy_vs_boy_games : ℕ)
    (girl_vs_girl_games : ℕ)
    (total_mixed_games : ℕ)
    (games_per_group : ℕ)
    (mixed_group_games_per_group : ℕ)
    (mixed_groups : ℕ) :
    total_children = 90 →
    total_groups = 30 →
    group_size = 3 →
    total_games = 90 →
    boy_vs_boy_games = 30 →
    girl_vs_girl_games = 14 →
    total_mixed_games = 46 →
    games_per_group = 3 →
    mixed_group_games_per_group = 2 →
    mixed_groups = total_mixed_games / mixed_group_games_per_group →
    mixed_groups = 23 :=
by
    intros
    sorry

end mixed_groups_count_l9_9727


namespace opposite_of_2023_l9_9197

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9197


namespace opposite_of_2023_l9_9428

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l9_9428


namespace y_range_l9_9535

theorem y_range (y : ℝ) (h1 : y < 0) (h2 : ⌈y⌉ * ⌊y⌋ = 72) : y ∈ set.Ioo (-9 : ℝ) (-8 : ℝ) :=
sorry

end y_range_l9_9535


namespace burglary_charge_sentence_l9_9546

theorem burglary_charge_sentence (B : ℕ) 
  (arson_counts : ℕ := 3) 
  (arson_sentence : ℕ := 36)
  (burglary_charges : ℕ := 2)
  (petty_larceny_factor : ℕ := 6)
  (total_jail_time : ℕ := 216) :
  arson_counts * arson_sentence + burglary_charges * B + (burglary_charges * petty_larceny_factor) * (B / 3) = total_jail_time → B = 18 := 
by
  sorry

end burglary_charge_sentence_l9_9546


namespace range_of_a_l9_9813

theorem range_of_a (a : ℝ) :
  (∃ (x : ℝ), (2 - 2^(-|x - 3|))^2 = 3 + a) ↔ -2 ≤ a ∧ a < 1 :=
by
  sorry

end range_of_a_l9_9813


namespace min_value_is_four_l9_9987

noncomputable def min_value (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 2) : ℝ :=
  (x + y) / (x * y * z)

theorem min_value_is_four (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 2) :
  min_value x y z h1 h2 h3 h4 = 4 :=
sorry

end min_value_is_four_l9_9987


namespace brown_dog_count_l9_9665

theorem brown_dog_count:
  ∀ (T L N : ℕ), T = 45 → L = 36 → N = 8 → (T - N - (T - L - N) = 37) :=
by
  intros T L N hT hL hN
  sorry

end brown_dog_count_l9_9665


namespace will_earnings_l9_9882

-- Defining the conditions
def hourly_wage : ℕ := 8
def monday_hours : ℕ := 8
def tuesday_hours : ℕ := 2

-- Calculating the earnings
def monday_earnings := monday_hours * hourly_wage
def tuesday_earnings := tuesday_hours * hourly_wage
def total_earnings := monday_earnings + tuesday_earnings

-- Stating the problem
theorem will_earnings : total_earnings = 80 := by
  -- sorry is used to skip the actual proof
  sorry

end will_earnings_l9_9882


namespace mixed_groups_count_l9_9722

-- Number of children in the chess club
def numChildren : Nat := 90

-- Number of groups
def numGroups : Nat := 30

-- Number of children per group
def childrenPerGroup : Nat := 3

-- Total number of games (3 chooses 2) times the number of groups
def totalGames : Nat := (childrenPerGroup * (childrenPerGroup - 1) / 2) * numGroups

-- Number of "boy vs boy" games
def boyGames : Nat := 30

-- Number of "girl vs girl" games
def girlGames : Nat := 14

-- Number of mixed games
def mixedGames : Nat := totalGames - boyGames - girlGames

-- Number of mixed groups
def mixedGroups : Nat := mixedGames / 2

-- Statement to prove that there are 23 mixed groups
theorem mixed_groups_count : mixedGroups = 23 := by
  -- Instantiates all the constants with their solid values and verifies the equation.
  have totalGames := 90
  have mixedGames := 46
  exact eq.symm (Nat.div_eq_of_eq_mul_right (by simp) (by simp[46]))

end mixed_groups_count_l9_9722


namespace tim_income_percentage_less_l9_9106

theorem tim_income_percentage_less (M T J : ℝ)
  (h₁ : M = 1.60 * T)
  (h₂ : M = 0.96 * J) :
  100 - (T / J) * 100 = 40 :=
by sorry

end tim_income_percentage_less_l9_9106


namespace opposite_of_2023_l9_9305

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l9_9305


namespace tree_planting_equation_l9_9666

variables (x : ℝ)

theorem tree_planting_equation (h1 : x > 50) :
  (300 / (x - 50) = 400 / x) ≠ False :=
by
  sorry

end tree_planting_equation_l9_9666


namespace tangent_line_equation_l9_9698

theorem tangent_line_equation (y : ℝ → ℝ) (x : ℝ) (dy_dx : ℝ → ℝ) (tangent_eq : ℝ → ℝ → Prop):
  (∀ x, y x = x^2 + Real.log x) →
  (∀ x, dy_dx x = (deriv y) x) →
  (dy_dx 1 = 3) →
  (tangent_eq x (y x) ↔ (3 * x - y x - 2 = 0)) →
  tangent_eq 1 (y 1) :=
by
  intros y_def dy_dx_def slope_at_1 tangent_line_char
  sorry

end tangent_line_equation_l9_9698


namespace quadratic_intersection_l9_9936

theorem quadratic_intersection (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 7 * x - 7 = 0) ↔ k ≥ -7/4 ∧ k ≠ 0 :=
by
  sorry

end quadratic_intersection_l9_9936


namespace at_most_one_existence_l9_9834

theorem at_most_one_existence
  (p : ℕ) (hp : Nat.Prime p)
  (A B : Finset (Fin p))
  (h_non_empty_A : A.Nonempty) (h_non_empty_B : B.Nonempty)
  (h_union : A ∪ B = Finset.univ) (h_disjoint : A ∩ B = ∅) :
  ∃! a : Fin p, ¬ (∃ x y : Fin p, (x ∈ A ∧ y ∈ B ∧ x + y = a) ∨ (x + y = a + p)) :=
sorry

end at_most_one_existence_l9_9834


namespace kids_in_group_l9_9761

open Nat

theorem kids_in_group (A K : ℕ) (h1 : A + K = 11) (h2 : 8 * A = 72) : K = 2 := by
  sorry

end kids_in_group_l9_9761


namespace determine_original_number_l9_9817

theorem determine_original_number (a b c : ℕ) (m : ℕ) (N : ℕ) 
  (h1 : N = 4410) 
  (h2 : (a + b + c) % 2 = 0)
  (h3 : m = 100 * a + 10 * b + c)
  (h4 : N + m = 222 * (a + b + c)) : 
  a = 4 ∧ b = 4 ∧ c = 4 :=
by 
  sorry

end determine_original_number_l9_9817


namespace opposite_of_2023_l9_9154

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l9_9154


namespace initial_donuts_30_l9_9078

variable (x y : ℝ)
variable (p : ℝ := 0.30)

theorem initial_donuts_30 (h1 : y = 9) (h2 : y = p * x) : x = 30 := by
  sorry

end initial_donuts_30_l9_9078


namespace opposite_of_2023_is_neg2023_l9_9351

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l9_9351


namespace binomial_square_l9_9808

theorem binomial_square (p : ℝ) : (∃ b : ℝ, (3 * x + b)^2 = 9 * x^2 + 24 * x + p) → p = 16 := by
  sorry

end binomial_square_l9_9808


namespace expected_value_of_sixes_l9_9871

theorem expected_value_of_sixes (n : ℕ) (k : ℕ) (p q : ℚ) 
  (h1 : n = 3) 
  (h2 : k = 6)
  (h3 : p = 1/6) 
  (h4 : q = 5/6) : 
  (1 : ℚ) / 2 = ∑ i in finset.range (n + 1), (i * (nat.choose n i * p^i * q^(n-i))) := 
sorry

end expected_value_of_sixes_l9_9871


namespace opposite_of_2023_l9_9381

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9381


namespace opposite_of_2023_l9_9168

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l9_9168


namespace opposite_of_2023_is_neg2023_l9_9311

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l9_9311


namespace super_cool_triangles_area_sum_l9_9613

theorem super_cool_triangles_area_sum : 
  ∃ (areas : List ℕ), 
  (∀ (a b : ℕ), (a * b / 2 = 3 * (a + b)) → 
  (a * b / 2 ∈ areas)) ∧ 
  areas.sum = 471 := 
sorry

end super_cool_triangles_area_sum_l9_9613


namespace opposite_of_2023_l9_9212

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l9_9212


namespace average_of_integers_is_ten_l9_9848

theorem average_of_integers_is_ten (k m r s t : ℕ) 
  (h1 : k < m) (h2 : m < r) (h3 : r < s) (h4 : s < t)
  (h5 : k > 0) (h6 : m > 0)
  (h7 : t = 20) (h8 : r = 13)
  (h9 : k = 1) (h10 : m = 2) (h11 : s = 14) :
  (k + m + r + s + t) / 5 = 10 := by
  sorry

end average_of_integers_is_ten_l9_9848


namespace intersection_counts_l9_9032

theorem intersection_counts (f g h : ℝ → ℝ)
  (hf : ∀ x, f x = -x^2 + 4 * x - 3)
  (hg : ∀ x, g x = -f x)
  (hh : ∀ x, h x = f (-x))
  (c : ℕ) (hc : c = 2)
  (d : ℕ) (hd : d = 1):
  10 * c + d = 21 :=
by
  sorry

end intersection_counts_l9_9032


namespace trigonometric_expression_eq_neg3_l9_9641

theorem trigonometric_expression_eq_neg3
  {α : ℝ} (h : Real.tan α = 1 / 2) :
  (1 + 2 * Real.sin (π - α) * Real.cos (-2 * π - α)) /
  ((Real.sin (-α))^2 - (Real.sin (5 * π / 2 - α))^2) = -3 :=
sorry

end trigonometric_expression_eq_neg3_l9_9641


namespace opposite_of_2023_l9_9192

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9192


namespace sum_is_zero_l9_9522

variable (a b c x y : ℝ)

theorem sum_is_zero (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) 
(h4 : a^3 + a * x + y = 0)
(h5 : b^3 + b * x + y = 0)
(h6 : c^3 + c * x + y = 0) : a + b + c = 0 :=
sorry

end sum_is_zero_l9_9522


namespace opposite_of_2023_l9_9271

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9271


namespace opposite_of_2023_l9_9173

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l9_9173


namespace opposite_of_2023_is_neg2023_l9_9343

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l9_9343


namespace technician_round_trip_percentage_l9_9034

theorem technician_round_trip_percentage
  (D : ℝ) 
  (H1 : D > 0) -- Assume D is positive
  (H2 : true) -- The technician completes the drive to the center
  (H3 : true) -- The technician completes 20% of the drive from the center
  : (1.20 * D / (2 * D)) * 100 = 60 := 
by
  simp [H1, H2, H3]
  sorry

end technician_round_trip_percentage_l9_9034


namespace domain_of_f_l9_9507

def function_domain (f : ℝ → ℝ) (domain : Set ℝ) : Prop :=
  ∀ x, x ∈ domain ↔ ∃ y, f y = x

noncomputable def f (x : ℝ) : ℝ :=
  (x + 6) / Real.sqrt (x^2 - 5 * x + 6)

theorem domain_of_f :
  function_domain f ((Set.Iio 2) ∪ (Set.Ioi 3)) :=
by
  sorry

end domain_of_f_l9_9507


namespace solve_system_l9_9119

def x : ℚ := 2.7 / 13
def y : ℚ := 1.0769

theorem solve_system :
  (∃ (x' y' : ℚ), 4 * x' - 3 * y' = -2.4 ∧ 5 * x' + 6 * y' = 7.5) ↔
  (x = 2.7 / 13 ∧ y = 1.0769) :=
by
  sorry

end solve_system_l9_9119


namespace range_of_a_l9_9918

theorem range_of_a (a x y : ℝ) (h1: x + 3 * y = 2 + a) (h2: 3 * x + y = -4 * a) (h3: x + y > 2) : a < -2 :=
sorry

end range_of_a_l9_9918


namespace people_on_williams_bus_l9_9591

theorem people_on_williams_bus
  (P : ℕ)
  (dutch_people : ℕ)
  (dutch_americans : ℕ)
  (window_seats : ℕ)
  (h1 : dutch_people = (3 * P) / 5)
  (h2 : dutch_americans = dutch_people / 2)
  (h3 : window_seats = dutch_americans / 3)
  (h4 : window_seats = 9) : 
  P = 90 :=
sorry

end people_on_williams_bus_l9_9591


namespace problem_ineq_l9_9836

variable {a b c : ℝ}

theorem problem_ineq 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2) ≥ 9 * a^2 * b^2 * c^2 := 
sorry

end problem_ineq_l9_9836


namespace sufficient_but_not_necessary_not_necessary_l9_9806

theorem sufficient_but_not_necessary (m x y a : ℝ) (h₀ : m > 0) (h₁ : |x - a| < m) (h₂ : |y - a| < m) : |x - y| < 2 * m :=
by
  sorry

theorem not_necessary (m : ℝ) (h₀ : m > 0) : ∃ x y a : ℝ, |x - y| < 2 * m ∧ ¬ (|x - a| < m ∧ |y - a| < m) :=
by
  sorry

end sufficient_but_not_necessary_not_necessary_l9_9806


namespace standard_normal_prob_gt_neg1_l9_9684

open ProbabilityTheory

noncomputable def standard_normal : ProbabilityDistribution :=
  normalPdf 0 1

theorem standard_normal_prob_gt_neg1 (p : ℝ) :
  (∀ (s : Set ℝ), standard_normal.prob s = ∫ x in s, normalPdf 0 1 x ∂volume) →
  (standard_normal.prob {x | x > 1} = p) →
  standard_normal.prob {x | x > -1} = 1 - p :=
by
  intros hProb hp
  have hSym : standard_normal.prob {x | x < -1} = p := sorry
  have hTotal : standard_normal.prob {x | x ≤ -1} = p := sorry
  have hComplement : standard_normal.prob {x | x > -1} = 1 - standard_normal.prob {x | x ≤ -1} := sorry
  exact hComplement.trans hTotal.trans hp.symm

end standard_normal_prob_gt_neg1_l9_9684


namespace opposite_of_2023_l9_9406

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l9_9406


namespace opposite_of_2023_l9_9229

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l9_9229


namespace opposite_of_2023_is_neg_2023_l9_9242

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l9_9242


namespace car_second_half_speed_l9_9474

theorem car_second_half_speed (D : ℝ) (V : ℝ) :
  let average_speed := 60  -- km/hr
  let first_half_speed := 75 -- km/hr
  average_speed = D / ((D / 2) / first_half_speed + (D / 2) / V) ->
  V = 150 :=
by
  sorry

end car_second_half_speed_l9_9474


namespace opposite_of_2023_l9_9337

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l9_9337


namespace sum_of_powers_of_i_l9_9551

-- Let i be the imaginary unit
def i : ℂ := Complex.I

theorem sum_of_powers_of_i : (1 + i + i^2 + i^3 + i^4 + i^5 + i^6 + i^7 + i^8 + i^9 + i^10) = i := by
  sorry

end sum_of_powers_of_i_l9_9551


namespace exists_integers_cubes_sum_product_l9_9477

theorem exists_integers_cubes_sum_product :
  ∃ (a b : ℤ), a^3 + b^3 = 91 ∧ a * b = 12 :=
by
  sorry

end exists_integers_cubes_sum_product_l9_9477


namespace tangent_inclination_point_l9_9711

theorem tangent_inclination_point :
  ∃ a : ℝ, (2 * a = 1) ∧ ((a, a^2) = (1 / 2, 1 / 4)) :=
by
  sorry

end tangent_inclination_point_l9_9711


namespace meaningful_expression_range_l9_9088

theorem meaningful_expression_range (x : ℝ) :
  (x + 2 ≥ 0) ∧ (x - 1 ≠ 0) ↔ (x ≥ -2) ∧ (x ≠ 1) :=
by
  sorry

end meaningful_expression_range_l9_9088


namespace tripod_max_height_l9_9756

noncomputable def tripod_new_height (original_height : ℝ) (original_leg_length : ℝ) (broken_leg_length : ℝ) : ℝ :=
  (broken_leg_length / original_leg_length) * original_height

theorem tripod_max_height :
  let original_height := 5
  let original_leg_length := 6
  let broken_leg_length := 4
  let h := tripod_new_height original_height original_leg_length broken_leg_length
  h = (10 / 3) :=
by
  sorry

end tripod_max_height_l9_9756


namespace find_denominators_l9_9863

theorem find_denominators (f1 f2 f3 f4 f5 f6 f7 f8 f9 : ℚ)
  (h1 : f1 = 1/3) (h2 : f2 = 1/7) (h3 : f3 = 1/9) (h4 : f4 = 1/11) (h5 : f5 = 1/33)
  (h6 : ∃ (d₁ d₂ d₃ d₄ : ℕ), f6 = 1/d₁ ∧ f7 = 1/d₂ ∧ f8 = 1/d₃ ∧ f9 = 1/d₄ ∧
    (∀ d, d ∈ [d₁, d₂, d₃, d₄] → d % 10 = 5))
  (h7 : f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 = 1) :
  ∃ (d₁ d₂ d₃ d₄ : ℕ), (d₁ = 5) ∧ (d₂ = 15) ∧ (d₃ = 45) ∧ (d₄ = 385) :=
by
  sorry

end find_denominators_l9_9863


namespace opposite_of_2023_l9_9170

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l9_9170


namespace opposite_of_2023_l9_9277

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9277


namespace opposite_of_2023_l9_9210

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l9_9210


namespace cubes_sum_eq_zero_l9_9814

theorem cubes_sum_eq_zero (a b : ℝ) (h1 : a + b = 0) (h2 : a * b = -7) : a^3 + b^3 = 0 :=
by
  sorry

end cubes_sum_eq_zero_l9_9814


namespace intersection_of_A_and_B_l9_9797

def A : Set ℝ := { x | 0 < x }
def B : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | 0 < x ∧ x ≤ 1 } := 
sorry

end intersection_of_A_and_B_l9_9797


namespace quadratic_relationship_l9_9508

theorem quadratic_relationship (a b c : ℝ) (α : ℝ) (h₁ : α + α^2 = -b / a) (h₂ : α^3 = c / a) : b^2 = 3 * a * c + c^2 :=
by
  sorry

end quadratic_relationship_l9_9508


namespace train_length_l9_9476

theorem train_length (L S : ℝ) 
  (h1 : L = S * 15) 
  (h2 : L + 100 = S * 25) : 
  L = 150 :=
by
  sorry

end train_length_l9_9476


namespace opposite_of_2023_l9_9303

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l9_9303


namespace rita_book_pages_l9_9844

theorem rita_book_pages (x : ℕ) (h1 : ∃ n₁, n₁ = (1/6 : ℚ) * x + 10) 
                                  (h2 : ∃ n₂, n₂ = (1/5 : ℚ) * ((5/6 : ℚ) * x - 10) + 20)
                                  (h3 : ∃ n₃, n₃ = (1/4 : ℚ) * ((4/5 : ℚ) * ((5/6 : ℚ) * x - 10) - 20) + 25)
                                  (h4 : ((3/4 : ℚ) * ((2/3 : ℚ) * x - 28) - 25) = 50) :
    x = 192 := 
sorry

end rita_book_pages_l9_9844


namespace opposite_of_2023_l9_9453

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l9_9453


namespace license_plate_combinations_l9_9900

theorem license_plate_combinations : 
  let letters := 26
  let non_repeated_letters := 25
  let positions_for_unique := Nat.choose 4 1
  let digits := 10 * 9 * 8
  (letters * non_repeated_letters * positions_for_unique * digits) = 187200 := by
  let letters := 26
  let non_repeated_letters := 25
  let positions_for_unique := Nat.choose 4 1
  let digits := 10 * 9 * 8
  have h : letters * non_repeated_letters * positions_for_unique * digits = 187200 := by sorry
  exact h


end license_plate_combinations_l9_9900


namespace square_side_length_l9_9467

theorem square_side_length (A : ℝ) (s : ℝ) (hA : A = 64) (h_s : A = s * s) : s = 8 := by
  sorry

end square_side_length_l9_9467


namespace judys_school_week_l9_9978

theorem judys_school_week
  (pencils_used : ℕ)
  (packs_cost : ℕ)
  (total_cost : ℕ)
  (days_period : ℕ)
  (pencils_per_pack : ℕ)
  (pencils_in_school_days : ℕ)
  (total_pencil_use : ℕ) :
  (total_cost / packs_cost * pencils_per_pack = total_pencil_use) →
  (total_pencil_use / days_period = pencils_used) →
  (pencils_in_school_days / pencils_used = 5) :=
sorry

end judys_school_week_l9_9978


namespace one_plus_x_pow_gt_one_plus_nx_l9_9063

theorem one_plus_x_pow_gt_one_plus_nx (x : ℝ) (n : ℕ) (hx1 : x > -1) (hx2 : x ≠ 0)
  (hn1 : n ≥ 2) : (1 + x)^n > 1 + n * x :=
sorry

end one_plus_x_pow_gt_one_plus_nx_l9_9063


namespace max_min_PA_l9_9072

-- Definition of the curve C and the parametric equations for the line l.
def curve (x y : ℝ) : Prop :=
  (x^2 / 4 + y^2 / 9 = 1)

def line (t x y : ℝ) : Prop :=
  (x = 2 + t) ∧ (y = 2 - 2 * t)

-- Angle in radians for 30 degrees
def θ := Real.pi / 6 

-- Distance function
def distance (θ α : ℝ) : ℝ := (2 * Real.sqrt 5 / 5) * Real.abs (5 * Real.sin (θ + α) - 6)

-- Proof statement for max and min values of |PA|
theorem max_min_PA : 
  ∀ θ α : ℝ,
  (α > 0) ∧ (α < Real.pi / 2) →
  (distance θ α = 2 * Real.sqrt 5 / 5 * Real.abs (5 * Real.sin (θ + α) - 6)) →
  (distance θ α = 22 * Real.sqrt 5 / 5 ∨ distance θ α = 2 * Real.sqrt 5 / 5) := 
by sorry

end max_min_PA_l9_9072


namespace independent_variable_range_l9_9857

theorem independent_variable_range (x : ℝ) :
  (∃ y : ℝ, y = 1 / (Real.sqrt x - 1)) ↔ x ≥ 0 ∧ x ≠ 1 := 
by
  sorry

end independent_variable_range_l9_9857


namespace opposite_of_2023_l9_9208

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l9_9208


namespace area_of_triangle_F1PF2_l9_9102

noncomputable def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  let x := P.1
  let y := P.2
  (x^2 / 25) + (y^2 / 16) = 1

def is_focus (f : ℝ × ℝ) : Prop := 
  f = (3, 0) ∨ f = (-3, 0)

def right_angle_at_P (F1 P F2 : ℝ × ℝ) : Prop := 
  let a1 := (F1.1 - P.1, F1.2 - P.2)
  let a2 := (F2.1 - P.1, F2.2 - P.2)
  a1.1 * a2.1 + a1.2 * a2.2 = 0

theorem area_of_triangle_F1PF2
  (P F1 F2 : ℝ × ℝ)
  (hP : point_on_ellipse P)
  (hF1 : is_focus F1)
  (hF2 : is_focus F2)
  (h_angle : right_angle_at_P F1 P F2) :
  1/2 * (P.1 - F1.1) * (P.2 - F2.2) = 16 :=
sorry

end area_of_triangle_F1PF2_l9_9102


namespace quadratic_to_standard_form_div_l9_9630

theorem quadratic_to_standard_form_div (b c : ℤ)
  (h : ∀ x : ℤ, x^2 - 2100 * x - 8400 = (x + b)^2 + c) :
  c / b = 1058 :=
sorry

end quadratic_to_standard_form_div_l9_9630


namespace opposite_of_2023_is_neg_2023_l9_9289

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l9_9289


namespace lcm_of_nt_and_16_l9_9008

open Int

def n : ℤ := 24
def m : ℤ := 16
def gcf_n_m : ℤ := 8

theorem lcm_of_nt_and_16 :
  (gcd n m = gcf_n_m) →
  (Nat.lcm n.nat_abs m.nat_abs = 48) :=
by
  intro h
  sorry

end lcm_of_nt_and_16_l9_9008


namespace percentage_increase_overtime_rate_l9_9742

theorem percentage_increase_overtime_rate :
  let regular_rate := 16
  let regular_hours_limit := 30
  let total_earnings := 760
  let total_hours_worked := 40
  let overtime_rate := 28 -- This is calculated as $280/10 from the solution.
  let increase_in_hourly_rate := overtime_rate - regular_rate
  let percentage_increase := (increase_in_hourly_rate / regular_rate) * 100
  percentage_increase = 75 :=
by {
  sorry
}

end percentage_increase_overtime_rate_l9_9742


namespace difference_in_probabilities_is_twenty_percent_l9_9044

-- Definition of the problem conditions
def prob_win_first_lawsuit : ℝ := 0.30
def prob_lose_first_lawsuit : ℝ := 0.70
def prob_win_second_lawsuit : ℝ := 0.50
def prob_lose_second_lawsuit : ℝ := 0.50

-- We need to prove that the difference in probability of losing both lawsuits and winning both lawsuits is 20%
theorem difference_in_probabilities_is_twenty_percent :
  (prob_lose_first_lawsuit * prob_lose_second_lawsuit) -
  (prob_win_first_lawsuit * prob_win_second_lawsuit) = 0.20 := 
by
  sorry

end difference_in_probabilities_is_twenty_percent_l9_9044


namespace opposite_of_2023_is_neg_2023_l9_9239

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l9_9239


namespace maximum_area_rhombus_l9_9825

theorem maximum_area_rhombus 
    (x₀ y₀ k : ℝ)
    (h1 : 2 ≤ x₀ ∧ x₀ ≤ 4)
    (h2 : y₀ = k / x₀)
    (h3 : ∀ x > 0, ∃ y, y = k / x) :
    (∀ (x₀ : ℝ), 2 ≤ x₀ ∧ x₀ ≤ 4 → ∃ (S : ℝ), S = 3 * (Real.sqrt 2 / 2 * x₀^2) → S ≤ 24 * Real.sqrt 2) :=
by
  sorry

end maximum_area_rhombus_l9_9825


namespace opposite_of_2023_l9_9450

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l9_9450


namespace absolute_value_inequality_l9_9117

variable (a b c d : ℝ)

theorem absolute_value_inequality (h₁ : a + b + c + d > 0) (h₂ : a > c) (h₃ : b > d) : 
  |a + b| > |c + d| := sorry

end absolute_value_inequality_l9_9117


namespace winning_strategy_ping_pong_l9_9455

theorem winning_strategy_ping_pong:
  ∀ {n : ℕ}, n = 18 → (∀ a : ℕ, 1 ≤ a ∧ a ≤ 4 → (∀ k : ℕ, k = 3 * a → (∃ b : ℕ, 1 ≤ b ∧ b ≤ 4 ∧ n - k - b = 18 - (k + b))) → (∃ c : ℕ, c = 3)) :=
by
sorry

end winning_strategy_ping_pong_l9_9455


namespace opposite_of_2023_l9_9451

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l9_9451


namespace jerrys_age_l9_9991

theorem jerrys_age (M J : ℕ) (h1 : M = 3 * J - 4) (h2 : M = 14) : J = 6 :=
by 
  sorry

end jerrys_age_l9_9991


namespace will_money_left_l9_9883

theorem will_money_left (initial sweater tshirt shoes refund_percentage : ℕ) 
  (h_initial : initial = 74)
  (h_sweater : sweater = 9)
  (h_tshirt : tshirt = 11)
  (h_shoes : shoes = 30)
  (h_refund_percentage : refund_percentage = 90) : 
  initial - (sweater + tshirt + (100 - refund_percentage) * shoes / 100) = 51 := by
  sorry

end will_money_left_l9_9883


namespace points_per_bag_l9_9010

/-
Wendy had 11 bags but didn't recycle 2 of them. She would have earned 
45 points for recycling all 11 bags. Prove that Wendy earns 5 points 
per bag of cans she recycles.
-/

def total_bags : Nat := 11
def unrecycled_bags : Nat := 2
def recycled_bags : Nat := total_bags - unrecycled_bags
def total_points : Nat := 45

theorem points_per_bag : total_points / recycled_bags = 5 := by
  sorry

end points_per_bag_l9_9010


namespace Levi_has_5_lemons_l9_9105

theorem Levi_has_5_lemons
  (Levi Jayden Eli Ian : ℕ)
  (h1 : Jayden = Levi + 6)
  (h2 : Eli = 3 * Jayden)
  (h3 : Ian = 2 * Eli)
  (h4 : Levi + Jayden + Eli + Ian = 115) :
  Levi = 5 := 
sorry

end Levi_has_5_lemons_l9_9105


namespace opposite_of_2023_l9_9426

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l9_9426


namespace expected_number_of_sixes_l9_9873

-- Define the problem context and conditions
def die_prob := (1 : ℝ) / 6

def expected_six (n : ℕ) : ℝ :=
  n * die_prob

-- The main proposition to prove
theorem expected_number_of_sixes (n : ℕ) (hn : n = 3) : expected_six n = 1 / 2 :=
by
  rw [hn]
  have fact1 : (3 : ℝ) * die_prob = 3 / 6 := by norm_cast; norm_num
  rw [fact1]
  norm_num

-- We add sorry to indicate incomplete proof, fulfilling criteria 4
sorry

end expected_number_of_sixes_l9_9873


namespace opposite_of_2023_l9_9186

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9186


namespace smallest_geometric_third_term_l9_9753

theorem smallest_geometric_third_term (d : ℝ) (a₁ a₂ a₃ g₁ g₂ g₃ : ℝ) 
  (h_AP : a₁ = 5 ∧ a₂ = 5 + d ∧ a₃ = 5 + 2 * d)
  (h_GP : g₁ = a₁ ∧ g₂ = a₂ + 3 ∧ g₃ = a₃ + 15)
  (h_geom : (g₂)^2 = g₁ * g₃) : g₃ = -4 := 
by
  -- We would provide the proof here.
  sorry

end smallest_geometric_third_term_l9_9753


namespace difference_in_total_cost_l9_9491

theorem difference_in_total_cost
  (item_price : ℝ := 15)
  (tax_rate1 : ℝ := 0.08)
  (tax_rate2 : ℝ := 0.072)
  (discount : ℝ := 0.005)
  (correct_difference : ℝ := 0.195) :
  let discounted_tax_rate := tax_rate2 - discount
  let total_price_with_tax_rate1 := item_price * (1 + tax_rate1)
  let total_price_with_discounted_tax_rate := item_price * (1 + discounted_tax_rate)
  total_price_with_tax_rate1 - total_price_with_discounted_tax_rate = correct_difference := by
  sorry

end difference_in_total_cost_l9_9491


namespace A_share_in_profit_l9_9039

def investment_A := 6300
def investment_B := 4200
def investment_C := 10500
def total_profit := 12500

def total_investment := investment_A + investment_B + investment_C
def A_ratio := investment_A / total_investment

theorem A_share_in_profit : (total_profit * A_ratio) = 3750 := by
  sorry

end A_share_in_profit_l9_9039


namespace opposite_of_2023_l9_9184

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9184


namespace largest_unachievable_score_l9_9542

theorem largest_unachievable_score :
  ∀ (x y : ℕ), 3 * x + 7 * y ≠ 11 :=
by
  sorry

end largest_unachievable_score_l9_9542


namespace number_of_positive_integers_with_positive_log_l9_9946

theorem number_of_positive_integers_with_positive_log (b : ℕ) (h : ∃ n : ℕ, n > 0 ∧ b ^ n = 1024) : 
  ∃ L, L = 4 :=
sorry

end number_of_positive_integers_with_positive_log_l9_9946


namespace inequality_condition_necessary_not_sufficient_l9_9097

theorem inequality_condition (a b : ℝ) (h1 : 0 < a) (h2 : a < b) : 
  (1 / a > 1 / b) :=
by
  sorry

theorem necessary_not_sufficient (a b : ℝ) :
  (1 / a > 1 / b → 0 < a ∧ a < b) ∧ ¬ (0 < a ∧ a < b → 1 / a > 1 / b) :=
by
  sorry

end inequality_condition_necessary_not_sufficient_l9_9097


namespace function_passes_through_point_l9_9585

noncomputable def func_graph (a : ℝ) (x : ℝ) : ℝ := a ^ (x - 1) + 2

theorem function_passes_through_point (a : ℝ) (h0 : a > 0) (h1 : a ≠ 1) :
  func_graph a 1 = 3 :=
by
  -- Proof logic is omitted
  sorry

end function_passes_through_point_l9_9585


namespace compare_abc_l9_9658

-- Definitions and conditions from the problem
def a : ℝ := Real.sqrt 2
def b : ℝ := Real.exp (1 / Real.exp 1)
def c : ℝ := Real.cbrt 6

-- Theorem statement for the problem
theorem compare_abc : a < b ∧ b < c :=
sorry

end compare_abc_l9_9658


namespace find_quadruples_l9_9912

def valid_quadruple (x1 x2 x3 x4 : ℝ) : Prop :=
  x1 + x2 * x3 * x4 = 2 ∧ 
  x2 + x3 * x4 * x1 = 2 ∧ 
  x3 + x4 * x1 * x2 = 2 ∧ 
  x4 + x1 * x2 * x3 = 2

theorem find_quadruples (x1 x2 x3 x4 : ℝ) :
  valid_quadruple x1 x2 x3 x4 ↔ (x1, x2, x3, x4) = (1, 1, 1, 1) ∨ 
                                   (x1, x2, x3, x4) = (3, -1, -1, -1) ∨ 
                                   (x1, x2, x3, x4) = (-1, 3, -1, -1) ∨ 
                                   (x1, x2, x3, x4) = (-1, -1, 3, -1) ∨ 
                                   (x1, x2, x3, x4) = (-1, -1, -1, 3) := by
  sorry

end find_quadruples_l9_9912


namespace opposite_of_2023_l9_9203

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l9_9203


namespace tan_2x_period_l9_9704

-- Define the tangent function and its standard period
def tan_period : ℝ := Real.pi

-- Define the function y = tan 2x
def f (x : ℝ) := Real.tan (2 * x)

-- State the property to be proved: The period of f is π/2
theorem tan_2x_period : ∀ x: ℝ, f(x) = f(x + π/2) := 
sorry

end tan_2x_period_l9_9704


namespace probability_of_picking_letter_from_MATHEMATICS_l9_9952

theorem probability_of_picking_letter_from_MATHEMATICS : 
  (8 : ℤ) / 26 = (4 : ℤ) / 13 :=
by
  norm_num

end probability_of_picking_letter_from_MATHEMATICS_l9_9952


namespace prime_square_mod_30_l9_9842

theorem prime_square_mod_30 (p : ℕ) (hp : Nat.Prime p) (hp_gt_5 : p > 5) : 
  p^2 % 30 = 1 ∨ p^2 % 30 = 19 := 
sorry

end prime_square_mod_30_l9_9842


namespace part_one_part_two_part_three_l9_9862

def numberOfWaysToPlaceBallsInBoxes : ℕ :=
  4 ^ 4

def numberOfWaysOneBoxEmpty : ℕ :=
  Nat.choose 4 2 * (Nat.factorial 4 / Nat.factorial 1)

def numberOfWaysTwoBoxesEmpty : ℕ :=
  (Nat.choose 4 1 * (Nat.factorial 4 / Nat.factorial 2)) + (Nat.choose 4 2 * (Nat.factorial 4 / (Nat.factorial 2 * Nat.factorial 2)))

theorem part_one : numberOfWaysToPlaceBallsInBoxes = 256 := by
  sorry

theorem part_two : numberOfWaysOneBoxEmpty = 144 := by
  sorry

theorem part_three : numberOfWaysTwoBoxesEmpty = 120 := by
  sorry

end part_one_part_two_part_three_l9_9862


namespace defective_product_probabilities_l9_9899

-- Defining the probabilities of production by machines a, b, and c
def P_H1 := 0.20
def P_H2 := 0.35
def P_H3 := 0.45

-- Defining the defect rates for machines a, b, and c
def P_A_H1 := 0.03
def P_A_H2 := 0.02
def P_A_H3 := 0.04

-- Total Probability of defect
def P_A := (P_A_H1 * P_H1) + (P_A_H2 * P_H2) + (P_A_H3 * P_H3)

-- Conditional probabilities using Bayes' Theorem
def P_H1_A := (P_H1 * P_A_H1) / P_A
def P_H2_A := (P_H2 * P_A_H2) / P_A
def P_H3_A := (P_H3 * P_A_H3) / P_A

-- Main theorem
theorem defective_product_probabilities:
  P_H1_A ≈ 0.1936 ∧ P_H2_A ≈ 0.2258 ∧ P_H3_A ≈ 0.5806 :=
by
  sorry

end defective_product_probabilities_l9_9899


namespace opposite_of_2023_l9_9429

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l9_9429


namespace company_profit_growth_l9_9600

theorem company_profit_growth (x : ℝ) (h : 1.6 * (1 + x / 100)^2 = 2.5) : x = 25 :=
sorry

end company_profit_growth_l9_9600


namespace shift_sine_cos_graph_l9_9693

theorem shift_sine_cos_graph (m : ℝ) (m_pos: m > 0):
    (∃ k : ℤ, f (x + m) at (0) = 1) → (m = π / 4) :=
by
    -- Define the function f
    let f := λ x, 2 * sin (2 * x + π / 3)
    -- Define the shifted function g
    let g := λ x, 2 * sin (2 * (x + m) + π / 3)
    -- Apply the conditions
    sorry

end shift_sine_cos_graph_l9_9693


namespace inequality_solution_l9_9454

theorem inequality_solution (x : ℝ) (h : 1 - x > x - 1) : x < 1 :=
sorry

end inequality_solution_l9_9454


namespace opposite_of_2023_l9_9199

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l9_9199


namespace abs_abc_eq_abs_k_l9_9832

variable {a b c k : ℝ}

noncomputable def distinct_nonzero (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a

theorem abs_abc_eq_abs_k (h_distinct : distinct_nonzero a b c)
                          (h_nonzero_k : k ≠ 0)
                          (h_eq : a + k / b = b + k / c ∧ b + k / c = c + k / a) :
  |a * b * c| = |k| :=
by
  sorry

end abs_abc_eq_abs_k_l9_9832


namespace opposite_of_2023_l9_9361

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l9_9361


namespace solve_for_a_l9_9534

theorem solve_for_a (a : ℝ) (h : a⁻¹ = (-1 : ℝ)^0) : a = 1 :=
sorry

end solve_for_a_l9_9534


namespace ratio_M_N_l9_9080

-- Definitions of M, Q and N based on the given conditions
variables (M Q P N : ℝ)
variable (h1 : M = 0.40 * Q)
variable (h2 : Q = 0.30 * P)
variable (h3 : N = 0.50 * P)

theorem ratio_M_N : M / N = 6 / 25 :=
by
  -- Proof steps would go here
  sorry

end ratio_M_N_l9_9080


namespace opposite_of_2023_l9_9452

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l9_9452


namespace opposite_of_2023_l9_9139

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l9_9139


namespace Andy_solves_correct_number_of_problems_l9_9758

-- Define the problem boundaries
def first_problem : ℕ := 80
def last_problem : ℕ := 125

-- The goal is to prove that Andy solves 46 problems given the range
theorem Andy_solves_correct_number_of_problems : (last_problem - first_problem + 1) = 46 :=
by
  sorry

end Andy_solves_correct_number_of_problems_l9_9758


namespace opposite_of_2023_l9_9224

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l9_9224


namespace opposite_of_2023_l9_9146

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l9_9146


namespace opposite_of_2023_l9_9434

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l9_9434


namespace bus_profit_problem_l9_9486

def independent_variable := "number of passengers per month"
def dependent_variable := "monthly profit"

-- Given monthly profit equation
def monthly_profit (x : ℕ) : ℤ := 2 * x - 4000

-- 1. Independent and Dependent variables
def independent_variable_defined_correctly : Prop :=
  independent_variable = "number of passengers per month"

def dependent_variable_defined_correctly : Prop :=
  dependent_variable = "monthly profit"

-- 2. Minimum passenger volume to avoid losses
def minimum_passenger_volume_no_loss : Prop :=
  ∀ x : ℕ, (monthly_profit x >= 0) → (x >= 2000)

-- 3. Monthly profit prediction for 4230 passengers
def monthly_profit_prediction_4230 (x : ℕ) : Prop :=
  x = 4230 → monthly_profit x = 4460

theorem bus_profit_problem :
  independent_variable_defined_correctly ∧
  dependent_variable_defined_correctly ∧
  minimum_passenger_volume_no_loss ∧
  monthly_profit_prediction_4230 4230 :=
by
  sorry

end bus_profit_problem_l9_9486


namespace opposite_of_2023_l9_9258

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l9_9258


namespace opposite_of_2023_is_neg2023_l9_9347

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l9_9347


namespace opposite_of_2023_l9_9227

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l9_9227


namespace units_of_Product_C_sold_l9_9560

-- Definitions of commission rates
def commission_rate_A : ℝ := 0.05
def commission_rate_B : ℝ := 0.07
def commission_rate_C : ℝ := 0.10

-- Definitions of revenues per unit
def revenue_A : ℝ := 1500
def revenue_B : ℝ := 2000
def revenue_C : ℝ := 3500

-- Definition of units sold
def units_A : ℕ := 5
def units_B : ℕ := 3

-- Commission calculations for Product A and B
def commission_A : ℝ := commission_rate_A * revenue_A * units_A
def commission_B : ℝ := commission_rate_B * revenue_B * units_B

-- Previous average commission and new average commission
def previous_avg_commission : ℝ := 100
def new_avg_commission : ℝ := 250

-- The main proof statement
theorem units_of_Product_C_sold (x : ℝ) (h1 : new_avg_commission = previous_avg_commission + 150)
  (h2 : total_units = units_A + units_B + x)
  (h3 : total_new_commission = commission_A + commission_B + (commission_rate_C * revenue_C * x))
  : x = 12 :=
by
  sorry

end units_of_Product_C_sold_l9_9560


namespace expected_number_of_sixes_l9_9865

-- Define the probability of not rolling a 6 on one die
def prob_not_six : ℚ := 5 / 6

-- Define the probability of rolling zero 6's on three dice
def prob_zero_six : ℚ := prob_not_six ^ 3

-- Define the probability of rolling exactly one 6 among the three dice
def prob_one_six (n : ℕ) : ℚ := n * (1 / 6) * (prob_not_six ^ (n - 1))

-- Calculate the probabilities of each specific outcomes
def prob_exactly_zero_six : ℚ := prob_zero_six
def prob_exactly_one_six : ℚ := prob_one_six 3 * (prob_not_six ^ 2)
def prob_exactly_two_six : ℚ := prob_one_six 3 * (1 / 6) * prob_not_six
def prob_exactly_three_six : ℚ := (1 / 6) ^ 3

-- Define the expected value calculation
noncomputable def expected_value : ℚ :=
  0 * prob_exactly_zero_six
  + 1 * prob_exactly_one_six
  + 2 * prob_exactly_two_six
  + 3 * prob_exactly_three_six

-- Prove that the expected value equals to 1/2
theorem expected_number_of_sixes : expected_value = 1 / 2 :=
  by
    sorry

end expected_number_of_sixes_l9_9865


namespace opposite_of_2023_l9_9333

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l9_9333


namespace triangle_angle_C_and_area_l9_9663

theorem triangle_angle_C_and_area (A B C : ℝ) (a b c : ℝ) 
  (h1 : 2 * c * Real.cos B = 2 * a - b)
  (h2 : c = Real.sqrt 3)
  (h3 : b - a = 1) :
  (C = Real.pi / 3) ∧
  (1 / 2 * a * b * Real.sin C = Real.sqrt 3 / 2) :=
by
  sorry

end triangle_angle_C_and_area_l9_9663


namespace initial_amount_l9_9922

theorem initial_amount (x : ℕ) (h1 : x - 3 + 14 = 22) : x = 11 :=
sorry

end initial_amount_l9_9922


namespace triangle_side_range_l9_9650

theorem triangle_side_range (x : ℝ) (hx1 : 8 + 10 > x) (hx2 : 10 + x > 8) (hx3 : x + 8 > 10) : 2 < x ∧ x < 18 :=
by
  sorry

end triangle_side_range_l9_9650


namespace prove_sets_l9_9074

noncomputable def A := { y : ℝ | ∃ x : ℝ, y = 3^x }
def B := { x : ℝ | x^2 - 4 ≤ 0 }

theorem prove_sets :
  A ∪ B = { x : ℝ | x ≥ -2 } ∧ A ∩ B = { x : ℝ | 0 < x ∧ x ≤ 2 } :=
by {
  sorry
}

end prove_sets_l9_9074


namespace ellipse_parametric_form_l9_9617

theorem ellipse_parametric_form :
  (∃ A B C D E F : ℤ,
    ((∀ t : ℝ, (3 * (Real.sin t - 2)) / (3 - Real.cos t) = x ∧ 
     (2 * (Real.cos t - 4)) / (3 - Real.cos t) = y) → 
    (A * x^2 + B * x * y + C * y^2 + D * x + E * y + F = 0)) ∧
    Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.gcd (Int.natAbs D) (Int.gcd (Int.natAbs E) (Int.natAbs F))))) = 1 ∧
    (Int.natAbs A + Int.natAbs B + Int.natAbs C + Int.natAbs D + Int.natAbs E + Int.natAbs F = 1846)) := 
sorry

end ellipse_parametric_form_l9_9617


namespace opposite_of_2023_l9_9413

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l9_9413


namespace profit_growth_rate_and_expected_profit_l9_9619

theorem profit_growth_rate_and_expected_profit
  (profit_April : ℕ)
  (profit_June : ℕ)
  (months : ℕ)
  (avg_growth_rate : ℝ)
  (profit_July : ℕ) :
  profit_April = 6000 ∧ profit_June = 7260 ∧ months = 2 ∧ 
  (profit_April : ℝ) * (1 + avg_growth_rate)^months = profit_June →
  avg_growth_rate = 0.1 ∧ 
  (profit_June : ℝ) * (1 + avg_growth_rate) = profit_July →
  profit_July = 7986 := 
sorry

end profit_growth_rate_and_expected_profit_l9_9619


namespace problem_G6_1_problem_G6_2_problem_G6_3_problem_G6_4_l9_9954

-- Problem G6.1
theorem problem_G6_1 : (21 ^ 3 - 11 ^ 3) / (21 ^ 2 + 21 * 11 + 11 ^ 2) = 10 := 
  sorry

-- Problem G6.2
theorem problem_G6_2 (p q : ℕ) (h1 : (p : ℚ) * 6 = 4 * (q : ℚ)) : q = 3 * p / 2 := 
  sorry

-- Problem G6.3
theorem problem_G6_3 (q r : ℕ) (h1 : q % 7 = 3) (h2 : r % 7 = 5) (h3 : 18 < r) (h4 : r < 26) : r = 24 := 
  sorry

-- Problem G6.4
def star (a b : ℕ) : ℕ := a * b + 1

theorem problem_G6_4 : star (star 3 4) 2 = 27 := 
  sorry

end problem_G6_1_problem_G6_2_problem_G6_3_problem_G6_4_l9_9954


namespace opposite_of_2023_is_neg2023_l9_9346

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l9_9346


namespace opposite_of_2023_is_neg2023_l9_9345

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l9_9345


namespace lucy_additional_kilometers_l9_9107

theorem lucy_additional_kilometers
  (mary_distance : ℚ := (3/8) * 24)
  (edna_distance : ℚ := (2/3) * mary_distance)
  (lucy_distance : ℚ := (5/6) * edna_distance) :
  (mary_distance - lucy_distance) = 4 :=
by
  sorry

end lucy_additional_kilometers_l9_9107


namespace opposite_of_2023_l9_9180

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l9_9180


namespace singers_in_choir_l9_9744

variable (X : ℕ)

/-- In the first verse, only half of the total singers sang -/ 
def first_verse_not_singing (X : ℕ) : ℕ := X / 2

/-- In the second verse, a third of the remaining singers joined in -/
def second_verse_joining (X : ℕ) : ℕ := (X / 2) / 3

/-- In the final third verse, 10 people joined so that the whole choir sang together -/
def remaining_singers_after_second_verse (X : ℕ) : ℕ := first_verse_not_singing X - second_verse_joining X

def final_verse_joining_condition (X : ℕ) : Prop := remaining_singers_after_second_verse X = 10

theorem singers_in_choir : ∃ (X : ℕ), final_verse_joining_condition X ∧ X = 30 :=
by
  sorry

end singers_in_choir_l9_9744


namespace PropositionA_necessary_not_sufficient_l9_9116

variable (a : ℝ)

def PropositionA : Prop := a < 2
def PropositionB : Prop := a^2 < 4

theorem PropositionA_necessary_not_sufficient : 
  (PropositionA a → PropositionB a) ∧ ¬ (PropositionB a → PropositionA a) :=
sorry

end PropositionA_necessary_not_sufficient_l9_9116


namespace opposite_of_2023_is_neg_2023_l9_9238

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l9_9238


namespace cellphone_loading_time_approximately_l9_9973

noncomputable def cellphone_loading_time_minutes : ℝ :=
  let T := 533.78 -- Solution for T from solving the given equation
  T / 60

theorem cellphone_loading_time_approximately :
  abs (cellphone_loading_time_minutes - 8.90) < 0.01 :=
by 
  -- The proof goes here, but we are just required to state it
  sorry

end cellphone_loading_time_approximately_l9_9973


namespace unique_number_not_in_range_l9_9982

noncomputable def g (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem unique_number_not_in_range
  (a b c d : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : g a b c d 13 = 13)
  (h2 : g a b c d 31 = 31)
  (h3 : ∀ x, x ≠ -d / c → g a b c d (g a b c d x) = x) :
  ∀ y, ∃! x, g a b c d x = y :=
by {
  sorry
}

end unique_number_not_in_range_l9_9982


namespace opposite_of_2023_l9_9161

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l9_9161


namespace true_false_question_count_l9_9608

theorem true_false_question_count (n : ℕ) (h : (1 / 3) * (1 / 2)^n = 1 / 12) : n = 2 := by
  sorry

end true_false_question_count_l9_9608


namespace tan_alpha_not_unique_l9_9948

theorem tan_alpha_not_unique (α : ℝ) (h1 : α > 0) (h2 : α < Real.pi) (h3 : (Real.sin α)^2 + Real.cos (2 * α) = 1) :
  ¬(∃ t : ℝ, Real.tan α = t) :=
by
  sorry

end tan_alpha_not_unique_l9_9948


namespace sufficient_but_not_necessary_for_circle_l9_9126

theorem sufficient_but_not_necessary_for_circle (m : ℝ) :
  (∀ (x y : ℝ), (x^2 + y^2 - 4 * x + 2 * y + m = 0) → (m = 0) → (∃ (a b : ℝ), (x - a)^2 + (y - b)^2 = r^2)) ∧
  ¬(∀ (x y : ℝ), (x^2 + y^2 - 4 * x + 2 * y + m = 0) → (∃ (a b : ℝ), (x - a)^2 + (y - b)^2 = r^2) → (m = 0)) := sorry

end sufficient_but_not_necessary_for_circle_l9_9126


namespace opposite_of_2023_l9_9405

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9405


namespace mixed_number_multiplication_l9_9763

def mixed_to_improper (a : Int) (b : Int) (c : Int) : Rat :=
  a + (b / c)

theorem mixed_number_multiplication : 
  let a := 5
  let b := mixed_to_improper 7 2 5
  a * b = (37 : Rat) :=
by
  intros
  sorry

end mixed_number_multiplication_l9_9763


namespace opposite_of_2023_l9_9364

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l9_9364


namespace expression_value_l9_9468

theorem expression_value (x : ℤ) (h : x = 2) : (2 * x + 5)^3 = 729 := by
  sorry

end expression_value_l9_9468


namespace opposite_of_2023_l9_9196

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9196


namespace pauly_omelets_l9_9114

/-- Pauly is making omelets for his family. There are three dozen eggs, and he plans to use them all. 
Each omelet requires 4 eggs. Including himself, there are 3 people. 
Prove that each person will get 3 omelets. -/

def total_eggs := 3 * 12

def eggs_per_omelet := 4

def total_omelets := total_eggs / eggs_per_omelet

def number_of_people := 3

def omelets_per_person := total_omelets / number_of_people

theorem pauly_omelets : omelets_per_person = 3 :=
by
  -- Placeholder proof
  sorry

end pauly_omelets_l9_9114


namespace weightlifter_one_hand_l9_9898

theorem weightlifter_one_hand (total_weight : ℕ) (h : total_weight = 20) (even_distribution : total_weight % 2 = 0) : total_weight / 2 = 10 :=
by
  sorry

end weightlifter_one_hand_l9_9898


namespace decimal_equiv_of_one_fourth_cubed_l9_9462

theorem decimal_equiv_of_one_fourth_cubed : (1 / 4 : ℝ) ^ 3 = 0.015625 := 
by sorry

end decimal_equiv_of_one_fourth_cubed_l9_9462


namespace opposite_of_2023_l9_9387

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9387


namespace opposite_of_2023_l9_9415

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l9_9415


namespace opposite_of_2023_l9_9329

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l9_9329


namespace paving_time_together_l9_9473

/-- Define the rate at which Mary alone paves the driveway -/
noncomputable def Mary_rate : ℝ := 1 / 4

/-- Define the rate at which Hillary alone paves the driveway -/
noncomputable def Hillary_rate : ℝ := 1 / 3

/-- Define the increased rate of Mary when working together -/
noncomputable def Mary_rate_increased := Mary_rate + (0.3333 * Mary_rate)

/-- Define the decreased rate of Hillary when working together -/
noncomputable def Hillary_rate_decreased := Hillary_rate - (0.5 * Hillary_rate)

/-- Combine their rates when working together -/
noncomputable def combined_rate := Mary_rate_increased + Hillary_rate_decreased

/-- Prove that the time taken to pave the driveway together is approximately 2 hours -/
theorem paving_time_together : abs ((1 / combined_rate) - 2) < 0.0001 :=
by
  sorry

end paving_time_together_l9_9473


namespace diameter_of_circular_field_l9_9913

noncomputable def diameter (C : ℝ) : ℝ := C / Real.pi

theorem diameter_of_circular_field :
  let cost_per_meter := 3
  let total_cost := 376.99
  let circumference := total_cost / cost_per_meter
  diameter circumference = 40 :=
by
  let cost_per_meter : ℝ := 3
  let total_cost : ℝ := 376.99
  let circumference : ℝ := total_cost / cost_per_meter
  have : circumference = 125.66333333333334 := by sorry
  have : diameter circumference = 40 := by sorry
  sorry

end diameter_of_circular_field_l9_9913


namespace opposite_of_2023_is_neg2023_l9_9313

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l9_9313


namespace mabel_tomatoes_l9_9988

theorem mabel_tomatoes (x : ℕ)
  (plant_1_bore : ℕ)
  (plant_2_bore : ℕ := x + 4)
  (total_first_two_plants : ℕ := x + plant_2_bore)
  (plant_3_bore : ℕ := 3 * total_first_two_plants)
  (plant_4_bore : ℕ := 3 * total_first_two_plants)
  (total_tomatoes : ℕ)
  (h1 : total_first_two_plants = 2 * x + 4)
  (h2 : plant_3_bore = 3 * (2 * x + 4))
  (h3 : plant_4_bore = 3 * (2 * x + 4))
  (h4 : total_tomatoes = x + plant_2_bore + plant_3_bore + plant_4_bore)
  (h5 : total_tomatoes = 140) :
   x = 8 :=
by
  sorry

end mabel_tomatoes_l9_9988


namespace unit_digit_power3_58_l9_9478

theorem unit_digit_power3_58 : (3 ^ 58) % 10 = 9 := by
  -- proof steps will be provided here
  sorry

end unit_digit_power3_58_l9_9478


namespace more_students_than_guinea_pigs_l9_9909

theorem more_students_than_guinea_pigs (students_per_classroom guinea_pigs_per_classroom classrooms : ℕ)
  (h1 : students_per_classroom = 24) 
  (h2 : guinea_pigs_per_classroom = 3) 
  (h3 : classrooms = 6) : 
  (students_per_classroom * classrooms) - (guinea_pigs_per_classroom * classrooms) = 126 := 
by
  sorry

end more_students_than_guinea_pigs_l9_9909


namespace bisecting_line_exists_l9_9874

noncomputable theory

-- Definitions of the geometric entities
variable (A : Point) (l : Line) (S : Circle)

-- Assumptions on the intersection and reflection
variable (l' : Line) (B : Point)
variable (h_reflection : l' = reflection th A l)
variable (h_intersection : B ∈ l'.intersect_circle S)

-- The main theorem to prove
theorem bisecting_line_exists : ∃ (AB : Line), AB.passes_through A ∧
(∃ P Q : Point, P ∈ l ∧ Q ∈ S ∧ (segment_intersected_by l S AB P Q).bisected_by A) := sorry

end bisecting_line_exists_l9_9874


namespace opposite_of_2023_l9_9136

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l9_9136


namespace opposite_of_2023_l9_9174

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l9_9174


namespace sum_of_areas_of_super_cool_triangles_l9_9610

def is_super_cool_triangle (a b : ℕ) : Prop :=
  (a * b / 2 = 3 * (a + b))

theorem sum_of_areas_of_super_cool_triangles :
  (∑ p in {p : ℕ × ℕ | is_super_cool_triangle p.1 p.2}, (p.1 * p.2) / 2) = 471 := 
by
  sorry

end sum_of_areas_of_super_cool_triangles_l9_9610


namespace distance_to_parabola_focus_l9_9067

theorem distance_to_parabola_focus :
  ∀ (x : ℝ), ((4 : ℝ) = (1 / 4) * x^2) → dist (0, 4) (0, 5) = 5 := 
by
  intro x
  intro hyp
  -- initial conditions indicate the distance is 5 and can be directly given
  sorry

end distance_to_parabola_focus_l9_9067


namespace opposite_of_2023_l9_9386

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9386


namespace opposite_of_2023_l9_9178

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l9_9178


namespace opposite_of_2023_l9_9334

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l9_9334


namespace opposite_of_2023_l9_9430

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l9_9430


namespace opposite_of_2023_l9_9431

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l9_9431


namespace simplify_1_simplify_2_l9_9572

theorem simplify_1 (a b : ℤ) : 2 * a - (a + b) = a - b :=
by
  sorry

theorem simplify_2 (x y : ℤ) : (x^2 - 2 * y^2) - 2 * (3 * y^2 - 2 * x^2) = 5 * x^2 - 8 * y^2 :=
by
  sorry

end simplify_1_simplify_2_l9_9572


namespace number_of_games_l9_9735

-- Definitions based on the conditions
def initial_money : ℕ := 104
def cost_of_blades : ℕ := 41
def cost_per_game : ℕ := 9

-- Lean 4 statement asserting the number of games Will can buy is 7
theorem number_of_games : (initial_money - cost_of_blades) / cost_per_game = 7 := by
  sorry

end number_of_games_l9_9735


namespace f_20_plus_f_neg20_l9_9556

noncomputable def f (a b : ℝ) (x : ℝ) := a * x^4 + b * x^2 + 5

theorem f_20_plus_f_neg20 (a b : ℝ) (h : f a b 20 = 3) : f a b 20 + f a b (-20) = 6 := by
  sorry

end f_20_plus_f_neg20_l9_9556


namespace shaded_area_fraction_l9_9906

theorem shaded_area_fraction :
  let A := (0, 0)
  let B := (4, 0)
  let C := (4, 4)
  let D := (0, 4)
  let P := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let Q := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let R := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  let S := ((D.1 + A.1) / 2, (D.2 + A.2) / 2)
  let area_triangle := 1 / 2 * 2 * 2
  let shaded_area := 2 * area_triangle
  let total_area := 4 * 4
  shaded_area / total_area = 1 / 4 :=
by
  sorry

end shaded_area_fraction_l9_9906


namespace equal_constant_difference_l9_9051

theorem equal_constant_difference (x : ℤ) (k : ℤ) :
  x^2 - 6*x + 11 = k ∧ -x^2 + 8*x - 13 = k ∧ 3*x^2 - 16*x + 19 = k → x = 4 :=
by
  sorry

end equal_constant_difference_l9_9051


namespace prob_A_wins_match_expected_games_won_variance_games_won_l9_9461

-- Definitions of probabilities
def prob_A_win := 0.6
def prob_B_win := 0.4

-- Prove that the probability of A winning the match is 0.648
theorem prob_A_wins_match : 
  prob_A_win * prob_A_win + 2 * prob_B_win * prob_A_win * prob_A_win = 0.648 :=
  sorry

-- Define the expected number of games won by A
noncomputable def expected_games_won_by_A := 
  0 * (prob_B_win * prob_B_win) + 1 * (2 * prob_A_win * prob_B_win * prob_B_win) + 
  2 * (prob_A_win * prob_A_win + 2 * prob_B_win * prob_A_win * prob_A_win)

-- Prove the expected number of games won by A is 1.5
theorem expected_games_won : 
  expected_games_won_by_A = 1.5 :=
  sorry

-- Define the variance of the number of games won by A
noncomputable def variance_games_won_by_A := 
  (prob_B_win * prob_B_win) * (0 - 1.5)^2 + 
  (2 * prob_A_win * prob_B_win * prob_B_win) * (1 - 1.5)^2 + 
  (prob_A_win * prob_A_win + 2 * prob_B_win * prob_A_win * prob_A_win) * (2 - 1.5)^2

-- Prove the variance of the number of games won by A is 0.57
theorem variance_games_won : 
  variance_games_won_by_A = 0.57 :=
  sorry

end prob_A_wins_match_expected_games_won_variance_games_won_l9_9461


namespace volume_to_surface_area_ratio_l9_9593

theorem volume_to_surface_area_ratio (base_layer: ℕ) (top_layer: ℕ) (unit_cube_volume: ℕ) (unit_cube_faces_exposed_base: ℕ) (unit_cube_faces_exposed_top: ℕ) 
  (V : ℕ := base_layer * top_layer * unit_cube_volume) 
  (S : ℕ := base_layer * unit_cube_faces_exposed_base + top_layer * unit_cube_faces_exposed_top) 
  (ratio := V / S) : ratio = 1 / 2 :=
by
  -- Base Layer: 4 cubes, 3 faces exposed per cube
  have base_layer_faces : ℕ := 4 * 3
  -- Top Layer: 4 cubes, 1 face exposed per cube
  have top_layer_faces : ℕ := 4 * 1
  -- Total volume is 8
  have V : ℕ := 4 * 2
  -- Total surface area is 16
  have S : ℕ := base_layer_faces + top_layer_faces
  -- Volume to surface area ratio computation
  have ratio : ℕ := V / S
  sorry

end volume_to_surface_area_ratio_l9_9593


namespace average_diesel_rate_l9_9631

theorem average_diesel_rate (r1 r2 r3 r4 : ℝ) (H1: (r1 + r2 + r3 + r4) / 4 = 1.52) :
    ((r1 + r2 + r3 + r4) / 4 = 1.52) :=
by
  exact H1

end average_diesel_rate_l9_9631


namespace no_perfect_square_with_one_digit_appending_l9_9708

def append_digit (n : Nat) (d : Fin 10) : Nat :=
  n * 10 + d.val

theorem no_perfect_square_with_one_digit_appending :
  ∀ n : Nat, (∃ k : Nat, k * k = n) → 
  (¬ (∃ d1 : Fin 10, ∃ k : Nat, k * k = append_digit n d1.val) ∧
   ¬ (∃ d2 : Fin 10, ∃ d3 : Fin 10, ∃ k : Nat, k * k = d2.val * 10 ^ (Nat.digits 10 n).length + n * 10 + d3.val)) :=
by sorry

end no_perfect_square_with_one_digit_appending_l9_9708


namespace largest_fraction_is_36_l9_9470

theorem largest_fraction_is_36 : 
  let A := (1 : ℚ) / 5
  let B := (2 : ℚ) / 10
  let C := (7 : ℚ) / 15
  let D := (9 : ℚ) / 20
  let E := (3 : ℚ) / 6
  A < E ∧ B < E ∧ C < E ∧ D < E :=
by
  let A := (1 : ℚ) / 5
  let B := (2 : ℚ) / 10
  let C := (7 : ℚ) / 15
  let D := (9 : ℚ) / 20
  let E := (3 : ℚ) / 6
  sorry

end largest_fraction_is_36_l9_9470


namespace opposite_of_2023_l9_9135

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l9_9135


namespace opposite_of_2023_l9_9137

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l9_9137


namespace opposite_of_2023_l9_9200

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l9_9200


namespace shifted_graph_sum_l9_9469

theorem shifted_graph_sum :
  let f (x : ℝ) := 3*x^2 - 2*x + 8
  let g (x : ℝ) := f (x - 6)
  let a := 3
  let b := -38
  let c := 128
  a + b + c = 93 :=
by
  sorry

end shifted_graph_sum_l9_9469


namespace find_d_l9_9503

theorem find_d (a b c d : ℝ) (hac : 0 < a) (hbc : 0 < b) (hcc : 0 < c) (hdc : 0 < d)
  (oscillates : ∀ x, -2 ≤ a * Real.sin (b * x + c) + d ∧ a * Real.sin (b * x + c) + d ≤ 4) :
  d = 1 :=
sorry

end find_d_l9_9503


namespace number_division_l9_9003

theorem number_division (m k n : ℤ) (h : n = m * k + 1) : n = m * k + 1 :=
by
  exact h

end number_division_l9_9003


namespace Mike_exercises_l9_9992

theorem Mike_exercises :
  let pull_ups_per_visit := 2
  let push_ups_per_visit := 5
  let squats_per_visit := 10
  let office_visits_per_day := 5
  let kitchen_visits_per_day := 8
  let living_room_visits_per_day := 7
  let days_per_week := 7
  let total_pull_ups := pull_ups_per_visit * office_visits_per_day * days_per_week
  let total_push_ups := push_ups_per_visit * kitchen_visits_per_day * days_per_week
  let total_squats := squats_per_visit * living_room_visits_per_day * days_per_week
  total_pull_ups = 70 ∧ total_push_ups = 280 ∧ total_squats = 490 :=
by
  let pull_ups_per_visit := 2
  let push_ups_per_visit := 5
  let squats_per_visit := 10
  let office_visits_per_day := 5
  let kitchen_visits_per_day := 8
  let living_room_visits_per_day := 7
  let days_per_week := 7
  let total_pull_ups := pull_ups_per_visit * office_visits_per_day * days_per_week
  let total_push_ups := push_ups_per_visit * kitchen_visits_per_day * days_per_week
  let total_squats := squats_per_visit * living_room_visits_per_day * days_per_week
  have h1 : total_pull_ups = 2 * 5 * 7 := rfl
  have h2 : total_push_ups = 5 * 8 * 7 := rfl
  have h3 : total_squats = 10 * 7 * 7 := rfl
  show total_pull_ups = 70 ∧ total_push_ups = 280 ∧ total_squats = 490
  sorry

end Mike_exercises_l9_9992


namespace hyperbola_slope_condition_l9_9798

-- Define the setup
variables (a b : ℝ) (P F1 F2 : ℝ × ℝ)
variables (h : a > 0) (k : b > 0)
variables (hyperbola : (∀ x y : ℝ, ((x^2 / a^2) - (y^2 / b^2) = 1)))

-- Define the condition
variables (cond : ∃ (P : ℝ × ℝ), 3 * abs (dist P F1 + dist P F2) ≤ 2 * dist F1 F2)

-- The proof goal
theorem hyperbola_slope_condition : (b / a) ≥ (Real.sqrt 5 / 2) :=
sorry

end hyperbola_slope_condition_l9_9798


namespace opposite_of_2023_is_neg2023_l9_9316

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l9_9316


namespace find_f_neg_3_l9_9771

variable {f : ℝ → ℝ}

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def functional_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, f (1 - x) = f (1 + x)

def function_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x^2

theorem find_f_neg_3 
  (hf_even : even_function f) 
  (hf_condition : functional_condition f)
  (hf_interval : function_on_interval f) : 
  f (-3) = 1 := 
by
  sorry

end find_f_neg_3_l9_9771


namespace opposite_of_2023_l9_9385

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9385


namespace venus_meal_cost_l9_9120

variable (V : ℕ)
variable cost_caesars : ℕ := 800 + 30 * 60
variable cost_venus : ℕ := 500 + V * 60

theorem venus_meal_cost :
  cost_caesars = cost_venus → V = 35 :=
by
  intro h
  sorry

end venus_meal_cost_l9_9120


namespace Lucy_additional_km_l9_9108

variables (Mary_distance Edna_distance Lucy_distance Additional_km : ℝ)

def problem_conditions : Prop :=
  let total_field := 24 in
  let Mary_fraction := 3 / 8 in
  let Edna_fraction := 2 / 3 in
  let Lucy_fraction := 5 / 6 in
  Mary_distance = Mary_fraction * total_field ∧
  Edna_distance = Edna_fraction * Mary_distance ∧
  Lucy_distance = Lucy_fraction * Edna_distance

theorem Lucy_additional_km (h : problem_conditions Mary_distance Edna_distance Lucy_distance) :
  Additional_km = Mary_distance - Lucy_distance :=
by { sorry }

end Lucy_additional_km_l9_9108


namespace smallest_triangle_perimeter_l9_9037

theorem smallest_triangle_perimeter :
  ∃ (y : ℕ), (y % 2 = 0) ∧ (y < 17) ∧ (y > 3) ∧ (7 + 10 + y = 21) :=
by
  sorry

end smallest_triangle_perimeter_l9_9037


namespace problem1_l9_9740

theorem problem1 (a b : ℤ) (h1 : abs a = 5) (h2 : abs b = 3) (h3 : abs (a - b) = b - a) : a - b = -8 ∨ a - b = -2 := by 
  sorry

end problem1_l9_9740


namespace ordered_triples_count_10_factorial_l9_9077

noncomputable def ordered_triples_count (a b c : ℕ) : ℕ :=
if (Nat.lcm a (Nat.lcm b c) = Nat.factorial 10 ∧ Nat.gcd a (Nat.gcd b c) = 1) then 1 else 0

theorem ordered_triples_count_10_factorial :
  ∑ a b c : ℕ, ordered_triples_count a b c = 82944 :=
begin
  sorry
end

end ordered_triples_count_10_factorial_l9_9077


namespace opposite_of_2023_l9_9424

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l9_9424


namespace range_of_m_l9_9653

theorem range_of_m (m : ℝ) :
  ( ∀ x : ℝ, |x + m| ≤ 4 → -2 ≤ x ∧ x ≤ 8) ↔ -4 ≤ m ∧ m ≤ -2 := 
by
  sorry

end range_of_m_l9_9653


namespace mixed_groups_count_l9_9728

theorem mixed_groups_count 
    (total_children : ℕ)
    (total_groups : ℕ)
    (group_size : ℕ)
    (total_games : ℕ)
    (boy_vs_boy_games : ℕ)
    (girl_vs_girl_games : ℕ)
    (total_mixed_games : ℕ)
    (games_per_group : ℕ)
    (mixed_group_games_per_group : ℕ)
    (mixed_groups : ℕ) :
    total_children = 90 →
    total_groups = 30 →
    group_size = 3 →
    total_games = 90 →
    boy_vs_boy_games = 30 →
    girl_vs_girl_games = 14 →
    total_mixed_games = 46 →
    games_per_group = 3 →
    mixed_group_games_per_group = 2 →
    mixed_groups = total_mixed_games / mixed_group_games_per_group →
    mixed_groups = 23 :=
by
    intros
    sorry

end mixed_groups_count_l9_9728


namespace opposite_of_2023_l9_9183

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9183


namespace opposite_of_2023_l9_9307

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l9_9307


namespace six_identities_l9_9496

theorem six_identities :
    (∀ x, (2 * x - 1) * (x - 3) = 2 * x^2 - 7 * x + 3) ∧
    (∀ x, (2 * x + 1) * (x + 3) = 2 * x^2 + 7 * x + 3) ∧
    (∀ x, (2 - x) * (1 - 3 * x) = 2 - 7 * x + 3 * x^2) ∧
    (∀ x, (2 + x) * (1 + 3 * x) = 2 + 7 * x + 3 * x^2) ∧
    (∀ x y, (2 * x - y) * (x - 3 * y) = 2 * x^2 - 7 * x * y + 3 * y^2) ∧
    (∀ x y, (2 * x + y) * (x + 3 * y) = 2 * x^2 + 7 * x * y + 3 * y^2) →
    6 = 6 :=
by
  intros
  sorry

end six_identities_l9_9496


namespace find_k_l9_9717

-- Define the sum of even integers from 2 to 2k
def sum_even_integers (k : ℕ) : ℕ :=
  2 * (k * (k + 1)) / 2

-- Define the condition that this sum equals 132
def sum_condition (t : ℕ) (k : ℕ) : Prop :=
  sum_even_integers k = t

theorem find_k (k : ℕ) (t : ℕ) (h₁ : t = 132) (h₂ : sum_condition t k) : k = 11 := by
  sorry

end find_k_l9_9717


namespace range_of_a_l9_9919

theorem range_of_a (a x y : ℝ) (h1: x + 3 * y = 2 + a) (h2: 3 * x + y = -4 * a) (h3: x + y > 2) : a < -2 :=
sorry

end range_of_a_l9_9919


namespace opposite_of_2023_l9_9213

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l9_9213


namespace number_of_integer_solutions_l9_9709

theorem number_of_integer_solutions : 
  (∃ (sols : List (ℤ × ℤ)), 
    (∀ (x y : ℤ), (x, y) ∈ sols ↔ (1 : ℚ)/x + (1 : ℚ)/y = 1/7) ∧ sols.length = 5) := 
sorry

end number_of_integer_solutions_l9_9709


namespace a_star_b_value_l9_9907

theorem a_star_b_value (a b : ℤ) (h1 : a + b = 12) (h2 : a * b = 32) (h3 : b = 8) :
  (1 / (a : ℚ) + 1 / (b : ℚ)) = 3 / 8 := by
sorry

end a_star_b_value_l9_9907


namespace metallic_sheet_width_l9_9606

theorem metallic_sheet_width 
  (length_of_cut_square : ℝ) (original_length_of_sheet : ℝ) (volume_of_box : ℝ) (w : ℝ)
  (h1 : length_of_cut_square = 5) 
  (h2 : original_length_of_sheet = 48) 
  (h3 : volume_of_box = 4940) : 
  (38 * (w - 10) * 5 = 4940) → w = 36 :=
by
  intros
  sorry

end metallic_sheet_width_l9_9606


namespace cost_per_millisecond_l9_9127

theorem cost_per_millisecond
  (C : ℝ)
  (h1 : 1.07 + (C * 1500) + 5.35 = 40.92) :
  C = 0.023 :=
sorry

end cost_per_millisecond_l9_9127


namespace intersection_A_B_l9_9519

-- Define the sets A and B based on the given conditions
def A := { x : ℝ | (1 / 9) ≤ (3:ℝ)^x ∧ (3:ℝ)^x ≤ 1 }
def B := { x : ℝ | x^2 < 1 }

-- State the theorem for the intersection of sets A and B
theorem intersection_A_B :
  A ∩ B = { x : ℝ | -1 < x ∧ x ≤ 0 } :=
by
  sorry

end intersection_A_B_l9_9519


namespace opposite_of_2023_l9_9328

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l9_9328


namespace opposite_of_2023_l9_9383

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9383


namespace number_of_cats_l9_9541

-- Defining the context and conditions
variables (x y z : Nat)
variables (h1 : x + y + z = 29) (h2 : x = z)

-- Proving the number of cats
theorem number_of_cats (x y z : Nat) (h1 : x + y + z = 29) (h2 : x = z) :
  6 * x + 3 * y = 87 := by
  sorry

end number_of_cats_l9_9541


namespace sum_of_super_cool_areas_l9_9612

def is_super_cool (a b : ℕ) : Prop :=
  (a - 9) * (b - 9) = 81

theorem sum_of_super_cool_areas : 
  let areas := [(90 * 10) / 2, (36 * 12) / 2, (18 * 18) / 2].erase_dup
  areas.sum = 828 :=
by
  sorry

end sum_of_super_cool_areas_l9_9612


namespace harry_items_left_l9_9802

def sea_stars : ℕ := 34
def seashells : ℕ := 21
def snails : ℕ := 29
def lost_items : ℕ := 25

def total_items : ℕ := sea_stars + seashells + snails
def remaining_items : ℕ := total_items - lost_items

theorem harry_items_left : remaining_items = 59 := by
  -- proof skipped
  sorry

end harry_items_left_l9_9802


namespace mms_pack_count_l9_9676

def mms_per_pack (sundaes_monday : Nat) (mms_monday : Nat) (sundaes_tuesday : Nat) (mms_tuesday : Nat) (packs : Nat) : Nat :=
  (sundaes_monday * mms_monday + sundaes_tuesday * mms_tuesday) / packs

theorem mms_pack_count 
  (sundaes_monday : Nat)
  (mms_monday : Nat)
  (sundaes_tuesday : Nat)
  (mms_tuesday : Nat)
  (packs : Nat)
  (monday_total_mms : sundaes_monday * mms_monday = 240)
  (tuesday_total_mms : sundaes_tuesday * mms_tuesday = 200)
  (total_packs : packs = 11)
  : mms_per_pack sundaes_monday mms_monday sundaes_tuesday mms_tuesday packs = 40 := by
  sorry

end mms_pack_count_l9_9676


namespace mixed_groups_count_l9_9721

-- Number of children in the chess club
def numChildren : Nat := 90

-- Number of groups
def numGroups : Nat := 30

-- Number of children per group
def childrenPerGroup : Nat := 3

-- Total number of games (3 chooses 2) times the number of groups
def totalGames : Nat := (childrenPerGroup * (childrenPerGroup - 1) / 2) * numGroups

-- Number of "boy vs boy" games
def boyGames : Nat := 30

-- Number of "girl vs girl" games
def girlGames : Nat := 14

-- Number of mixed games
def mixedGames : Nat := totalGames - boyGames - girlGames

-- Number of mixed groups
def mixedGroups : Nat := mixedGames / 2

-- Statement to prove that there are 23 mixed groups
theorem mixed_groups_count : mixedGroups = 23 := by
  -- Instantiates all the constants with their solid values and verifies the equation.
  have totalGames := 90
  have mixedGames := 46
  exact eq.symm (Nat.div_eq_of_eq_mul_right (by simp) (by simp[46]))

end mixed_groups_count_l9_9721


namespace jenn_money_left_over_l9_9977

-- Definitions based on problem conditions
def num_jars : ℕ := 5
def quarters_per_jar : ℕ := 160
def value_per_quarter : ℚ := 0.25   -- Rational number to represent $0.25
def cost_of_bike : ℚ := 180         -- Rational number to represent $180

-- Statement to prove that Jenn will have $20 left after buying the bike
theorem jenn_money_left_over : 
  (num_jars * quarters_per_jar * value_per_quarter) - cost_of_bike = 20 :=
by
  sorry

end jenn_money_left_over_l9_9977


namespace trajectory_eq_ellipse_range_sum_inv_dist_l9_9523

-- Conditions for circle M
def CircleM := { center : ℝ × ℝ // center = (-3, 0) }
def radiusM := 1

-- Conditions for circle N
def CircleN := { center : ℝ × ℝ // center = (3, 0) }
def radiusN := 9

-- Conditions for circle P
def CircleP (x y : ℝ) (r : ℝ) := 
  (dist (x, y) (-3, 0) = r + radiusM) ∧
  (dist (x, y) (3, 0) = radiusN - r)

-- Proof for the equation of the trajectory
theorem trajectory_eq_ellipse :
  ∃ (x y : ℝ), CircleP x y r → x^2 / 25 + y^2 / 16 = 1 :=
sorry

-- Proof for the range of 1/PM + 1/PN
theorem range_sum_inv_dist :
  ∃ (r PM PN : ℝ), 
    PM ∈ [2, 8] ∧ 
    PN = 10 - PM ∧ 
    CircleP (PM - radiusM) (PN - radiusN) r → 
    (2/5 ≤ (1/PM + 1/PN) ∧ (1/PM + 1/PN) ≤ 5/8) :=
sorry

end trajectory_eq_ellipse_range_sum_inv_dist_l9_9523


namespace hemisphere_surface_area_l9_9125

theorem hemisphere_surface_area (r : ℝ) (h : π * r^2 = 225 * π) : 2 * π * r^2 + π * r^2 = 675 * π := 
by
  sorry

end hemisphere_surface_area_l9_9125


namespace opposite_of_2023_l9_9384

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9384


namespace comparison_of_powers_l9_9712

theorem comparison_of_powers : 6 ^ 0.7 > 0.7 ^ 6 ∧ 0.7 ^ 6 > 0.6 ^ 7 := by
  sorry

end comparison_of_powers_l9_9712


namespace proof_problem_l9_9526

theorem proof_problem (α : ℝ) (h1 : 0 < α ∧ α < π)
    (h2 : Real.sin α + Real.cos α = 1 / 5) :
    (Real.tan α = -4 / 3) ∧ 
    ((Real.sin (3 * Real.pi / 2 + α) * Real.sin (Real.pi / 2 - α) * (Real.tan (Real.pi - α))^3) / 
    (Real.cos (Real.pi / 2 + α) * Real.cos (3 * Real.pi / 2 - α)) = -4 / 3) :=
by
  sorry

end proof_problem_l9_9526


namespace master_bedroom_and_bath_area_l9_9731

-- Definitions of the problem conditions
def guest_bedroom_area : ℕ := 200
def two_guest_bedrooms_area : ℕ := 2 * guest_bedroom_area
def kitchen_guest_bath_living_area : ℕ := 600
def total_rent : ℕ := 3000
def cost_per_sq_ft : ℕ := 2
def total_area_of_house : ℕ := total_rent / cost_per_sq_ft
def expected_master_bedroom_and_bath_area : ℕ := 500

-- Theorem statement to prove the desired area
theorem master_bedroom_and_bath_area :
  total_area_of_house - (two_guest_bedrooms_area + kitchen_guest_bath_living_area) = expected_master_bedroom_and_bath_area :=
by
  sorry

end master_bedroom_and_bath_area_l9_9731


namespace first_triangular_number_year_in_21st_century_l9_9565

theorem first_triangular_number_year_in_21st_century :
  ∃ n : ℕ, (n * (n + 1)) / 2 = 2016 ∧ 2000 ≤ 2016 ∧ 2016 < 2100 :=
by
  sorry

end first_triangular_number_year_in_21st_century_l9_9565


namespace opposite_of_2023_l9_9341

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l9_9341


namespace urn_problem_probability_l9_9498

theorem urn_problem_probability :
  let urn_initial_red_balls := 1 in
  let urn_initial_blue_balls := 1 in
  let total_operations := 5 in
  let final_red_balls := 3 in
  let final_blue_balls := 4 in
  let box_contains_additional_balls := true in -- This is implicit in the problem.
  (probability (λ s, s = (final_red_balls, final_blue_balls)) | urn_initial_red_balls := 1, urn_initial_blue_balls := 1, total_operations := total_operations, box_contains_additional_balls := box_contains_additional_balls) = 1/6 :=
sorry

end urn_problem_probability_l9_9498


namespace max_value_on_interval_l9_9527

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := (a * x^2 + b * x + c) / Real.exp x
noncomputable def f' (a b c : ℝ) (x : ℝ) : ℝ := ((2 * a * x + b) * Real.exp x - (a * x^2 + b * x + c)) / Real.exp (2 * x)

variable (a b c : ℝ)

-- Given conditions
axiom pos_a : a > 0
axiom zero_point_neg3 : f' a b c (-3) = 0
axiom zero_point_0 : f' a b c 0 = 0
axiom min_value_neg3 : f a b c (-3) = -Real.exp 3

-- Goal: Maximum value of f(x) on the interval [-5, ∞) is 5e^5.
theorem max_value_on_interval : ∃ y ∈ Set.Ici (-5), f a b c y = 5 * Real.exp 5 := by
  sorry

end max_value_on_interval_l9_9527


namespace opposite_of_2023_is_neg2023_l9_9323

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l9_9323


namespace integer_solutions_pxy_eq_xy_l9_9775

theorem integer_solutions_pxy_eq_xy (p : ℤ) (hp : Prime p) :
  ∃ x y : ℤ, p * (x + y) = x * y ∧ 
  ((x, y) = (2 * p, 2 * p) ∨ 
  (x, y) = (0, 0) ∨ 
  (x, y) = (p + 1, p + p^2) ∨ 
  (x, y) = (p - 1, p - p^2) ∨ 
  (x, y) = (p + p^2, p + 1) ∨ 
  (x, y) = (p - p^2, p - 1)) :=
by
  sorry

end integer_solutions_pxy_eq_xy_l9_9775


namespace opposite_of_2023_l9_9257

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l9_9257


namespace opposite_of_2023_l9_9332

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l9_9332


namespace arithmetic_sequence_properties_l9_9068

def arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∃ d a_1, ∀ n, a n = a_1 + d * (n - 1)

def sum_n (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

def sum_b (b : ℕ → ℕ) (T : ℕ → ℕ) : Prop :=
  ∀ n, T n = n^2 + n + (3^(n+1) - 3)/2

theorem arithmetic_sequence_properties :
  ∀ (a : ℕ → ℕ) (S : ℕ → ℕ),
    (arithmetic_seq a) →
    a 5 = 10 →
    S 7 = 56 →
    (∀ n, a n = 2 * n) ∧
    ∃ (b T : ℕ → ℕ), (∀ n, b n = a n + 3^n) ∧ sum_b b T :=
by
  intros a S ha h5 hS7
  sorry

end arithmetic_sequence_properties_l9_9068


namespace determine_b_perpendicular_l9_9509

theorem determine_b_perpendicular :
  ∀ (b : ℝ),
  (b * 2 + (-3) * (-1) + 2 * 4 = 0) → 
  b = -11/2 :=
by
  intros b h
  sorry

end determine_b_perpendicular_l9_9509


namespace parabola_intercepts_sum_l9_9700

theorem parabola_intercepts_sum (a b c : ℝ)
  (h₁ : a = 5)
  (h₂ : b = (9 + Real.sqrt 21) / 6)
  (h₃ : c = (9 - Real.sqrt 21) / 6) :
  a + b + c = 8 :=
by
  sorry

end parabola_intercepts_sum_l9_9700


namespace find_m_and_n_l9_9054

namespace BinomialProof

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Given conditions
def condition1 (n m : ℕ) : Prop :=
  binom (n+1) (m+1) = binom (n+1) m

def condition2 (n m : ℕ) : Prop :=
  binom (n+1) m / binom (n+1) (m-1) = 5 / 3

-- Problem statement
theorem find_m_and_n : ∃ (m n : ℕ), 
  (condition1 n m) ∧ 
  (condition2 n m) ∧ 
  m = 3 ∧ n = 6 := sorry

end BinomialProof

end find_m_and_n_l9_9054


namespace area_under_the_curve_l9_9577

theorem area_under_the_curve : 
  ∫ x in (0 : ℝ)..1, (x^2 + 1) = 4 / 3 := 
by
  sorry

end area_under_the_curve_l9_9577


namespace ratio_of_amount_divided_to_total_savings_is_half_l9_9027

theorem ratio_of_amount_divided_to_total_savings_is_half :
  let husband_weekly_contribution := 335
  let wife_weekly_contribution := 225
  let weeks_in_six_months := 6 * 4
  let total_weekly_contribution := husband_weekly_contribution + wife_weekly_contribution
  let total_savings := total_weekly_contribution * weeks_in_six_months
  let amount_per_child := 1680
  let number_of_children := 4
  let total_amount_divided := amount_per_child * number_of_children
  (total_amount_divided : ℝ) / total_savings = 0.5 := 
by
  sorry

end ratio_of_amount_divided_to_total_savings_is_half_l9_9027


namespace consecutive_integer_sets_l9_9710

theorem consecutive_integer_sets (S : ℕ) (hS : S = 180) : 
  ∃ n_values : Finset ℕ, 
  (∀ n ∈ n_values, (∃ a : ℕ, n * (2 * a + n - 1) = 2 * S) ∧ n >= 2) ∧ 
  n_values.card = 4 :=
by
  sorry

end consecutive_integer_sets_l9_9710


namespace focus_of_parabola_l9_9777

theorem focus_of_parabola :
  (∃ f : ℝ, ∀ y : ℝ, (x = -1 / 4 * y^2) = (x = (y^2 / 4 + f)) -> f = -1) :=
by
  sorry

end focus_of_parabola_l9_9777


namespace opposite_of_2023_l9_9267

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9267


namespace quad_intersects_x_axis_l9_9937

theorem quad_intersects_x_axis (k : ℝ) :
  (∃ x : ℝ, k * x ^ 2 - 7 * x - 7 = 0) ↔ (k ≥ -7 / 4 ∧ k ≠ 0) :=
by sorry

end quad_intersects_x_axis_l9_9937


namespace opposite_of_2023_is_neg_2023_l9_9234

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l9_9234


namespace opposite_of_2023_l9_9302

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l9_9302


namespace smallest_integer_l9_9703

theorem smallest_integer (n : ℕ) (h : n > 0) (h1 : lcm 36 n / gcd 36 n = 24) : n = 96 :=
sorry

end smallest_integer_l9_9703


namespace opposite_of_2023_is_neg_2023_l9_9231

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l9_9231


namespace opposite_of_2023_is_neg2023_l9_9312

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l9_9312


namespace find_other_root_l9_9688

theorem find_other_root 
  (m : ℚ) 
  (h : 3 * 3^2 + m * 3 - 5 = 0) :
  (1 - 3) * (x : ℚ) = 0 :=
sorry

end find_other_root_l9_9688


namespace golden_section_AP_length_l9_9071

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

noncomputable def golden_ratio_recip : ℝ := (Real.sqrt 5 - 1) / 2

theorem golden_section_AP_length (AB : ℝ) (P : ℝ) 
  (h1 : AB = 2) (h2 : P = golden_ratio_recip * AB) : 
  P = Real.sqrt 5 - 1 :=
by
  sorry

end golden_section_AP_length_l9_9071


namespace find_divisor_l9_9019

theorem find_divisor (remainder quotient dividend divisor : ℕ) 
  (h_rem : remainder = 8)
  (h_quot : quotient = 43)
  (h_div : dividend = 997)
  (h_eq : dividend = divisor * quotient + remainder) : 
  divisor = 23 :=
by
  sorry

end find_divisor_l9_9019


namespace find_y_from_triangle_properties_l9_9095

-- Define angle measures according to the given conditions
def angle_BAC := 45
def angle_CDE := 72

-- Define the proof problem
theorem find_y_from_triangle_properties
: ∀ (y : ℝ), (∃ (BAC ACB ABC ADC ADE AED DEB : ℝ),
    angle_BAC = 45 ∧
    angle_CDE = 72 ∧
    BAC + ACB + ABC = 180 ∧
    ADC = 180 ∧
    ADE = 180 - angle_CDE ∧
    EAD = angle_BAC ∧
    AED + ADE + EAD = 180 ∧
    DEB = 180 - AED ∧
    y = DEB) →
    y = 153 :=
by sorry

end find_y_from_triangle_properties_l9_9095


namespace problem_part_1_problem_part_2_problem_part_3_l9_9518

open Set

universe u

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 1 < x ∧ x < 6}
def C (a : ℝ) : Set ℝ := {x | x > a}
def U : Set ℝ := univ

theorem problem_part_1 : A ∪ B = {x | 1 < x ∧ x ≤ 8} :=
sorry

theorem problem_part_2 : (U \ A) ∩ B = {x | 1 < x ∧ x < 2} :=
sorry

theorem problem_part_3 (a : ℝ) (h : (A ∩ C a) ≠ ∅) : a < 8 :=
sorry

end problem_part_1_problem_part_2_problem_part_3_l9_9518


namespace MaryIncomeIs64PercentOfJuanIncome_l9_9686

variable {J T M : ℝ}

-- Conditions
def TimIncome (J : ℝ) : ℝ := 0.40 * J
def MaryIncome (T : ℝ) : ℝ := 1.60 * T

-- Theorem to prove
theorem MaryIncomeIs64PercentOfJuanIncome (J : ℝ) :
  MaryIncome (TimIncome J) = 0.64 * J :=
by
  sorry

end MaryIncomeIs64PercentOfJuanIncome_l9_9686


namespace opposite_of_2023_is_neg2023_l9_9348

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l9_9348


namespace opposite_of_2023_l9_9300

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l9_9300


namespace part1_part2_part3_l9_9796

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x + 1

theorem part1 (a : ℝ) (x : ℝ) (h : 0 < x) :
  (a ≤ 0 → (∀ x > 0, f a x < 0)) ∧
  (a > 0 → (∀ x ∈ Set.Ioo 0 a, f a x > 0) ∧ (∀ x ∈ Set.Ioi a, f a x < 0)) :=
sorry

theorem part2 {a : ℝ} : (∀ x > 0, f a x ≤ 0) → a = 1 :=
sorry

theorem part3 (n : ℕ) (h : 0 < n) :
  (1 + 1 / n : ℝ)^n < Real.exp 1 ∧ Real.exp 1 < (1 + 1 / n : ℝ)^(n + 1) :=
sorry

end part1_part2_part3_l9_9796


namespace tan_of_log_conditions_l9_9061

theorem tan_of_log_conditions (x : ℝ) (h1 : 0 < x ∧ x < (Real.pi / 2))
  (h2 : Real.log (Real.sin (2 * x)) - Real.log (Real.sin x) = Real.log (1 / 2)) :
  Real.tan x = Real.sqrt 15 :=
sorry

end tan_of_log_conditions_l9_9061


namespace quadratic_intersection_l9_9935

theorem quadratic_intersection (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 7 * x - 7 = 0) ↔ k ≥ -7/4 ∧ k ≠ 0 :=
by
  sorry

end quadratic_intersection_l9_9935


namespace opposite_of_2023_l9_9377

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9377


namespace length_of_train_proof_l9_9036

-- Definitions based on conditions
def speed_of_train := 54.99520038396929 -- in km/h
def speed_of_man := 5 -- in km/h
def time_to_cross := 6 -- in seconds

-- Conversion factor from km/h to m/s
def kmph_to_mps (speed_kmph: ℝ) : ℝ := speed_kmph * (5 / 18)

-- Relative speed in m/s
def relative_speed_mps := kmph_to_mps (speed_of_train + speed_of_man)

-- Length of the train (distance)
def length_of_train := relative_speed_mps * time_to_cross

-- The proof problem statement
theorem length_of_train_proof :
  length_of_train = 99.99180063994882 := by
  sorry

end length_of_train_proof_l9_9036


namespace opposite_of_2023_is_neg_2023_l9_9292

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l9_9292


namespace inequality_solution_set_l9_9538

theorem inequality_solution_set (a b c : ℝ)
  (h1 : a < 0)
  (h2 : b = -a)
  (h3 : c = -2 * a) :
  ∀ x : ℝ, (c * x^2 + b * x + a > 0) ↔ (x < -1 ∨ x > 1 / 2) :=
by
  sorry

end inequality_solution_set_l9_9538


namespace probability_is_zero_l9_9762

noncomputable def probability_same_number (b d : ℕ) (h_b : b < 150) (h_d : d < 150)
    (h_b_multiple: b % 15 = 0) (h_d_multiple: d % 20 = 0) (h_square: b * b = b ∨ d * d = d) : ℝ :=
  0

theorem probability_is_zero (b d : ℕ) (h_b : b < 150) (h_d : d < 150)
    (h_b_multiple: b % 15 = 0) (h_d_multiple: d % 20 = 0) (h_square: b * b = b ∨ d * d = d) : 
    probability_same_number b d h_b h_d h_b_multiple h_d_multiple h_square = 0 :=
  sorry

end probability_is_zero_l9_9762


namespace factorization_1_factorization_2_factorization_3_factorization_4_l9_9774

-- Problem 1
theorem factorization_1 (a b : ℝ) : 
  4 * a^2 + 12 * a * b + 9 * b^2 = (2 * a + 3 * b)^2 :=
by sorry

-- Problem 2
theorem factorization_2 (a b : ℝ) : 
  16 * a^2 * (a - b) + 4 * b^2 * (b - a) = 4 * (a - b) * (2 * a - b) * (2 * a + b) :=
by sorry

-- Problem 3
theorem factorization_3 (m n : ℝ) : 
  25 * (m + n)^2 - 9 * (m - n)^2 = 4 * (4 * m + n) * (m + 4 * n) :=
by sorry

-- Problem 4
theorem factorization_4 (a b : ℝ) : 
  4 * a^2 - b^2 - 4 * a + 1 = (2 * a - 1 + b) * (2 * a - 1 - b) :=
by sorry

end factorization_1_factorization_2_factorization_3_factorization_4_l9_9774


namespace correct_equation_l9_9733

theorem correct_equation (a b : ℝ) : (a - b) ^ 3 * (b - a) ^ 4 = (a - b) ^ 7 :=
sorry

end correct_equation_l9_9733


namespace opposite_of_2023_is_neg_2023_l9_9290

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l9_9290


namespace minimum_omega_l9_9956

theorem minimum_omega (ω : ℕ) (h_pos : ω ∈ {n : ℕ | n > 0}) (h_cos_center : ∃ k : ℤ, ω * (π / 6) + (π / 6) = k * π + π / 2) :
  ω = 2 :=
by { sorry }

end minimum_omega_l9_9956


namespace quad_intersects_x_axis_l9_9939

theorem quad_intersects_x_axis (k : ℝ) :
  (∃ x : ℝ, k * x ^ 2 - 7 * x - 7 = 0) ↔ (k ≥ -7 / 4 ∧ k ≠ 0) :=
by sorry

end quad_intersects_x_axis_l9_9939


namespace find_a_minus_b_l9_9122

theorem find_a_minus_b
  (a b : ℝ)
  (f g h h_inv : ℝ → ℝ)
  (hf : ∀ x, f x = a * x + b)
  (hg : ∀ x, g x = -4 * x + 3)
  (hh : ∀ x, h x = f (g x))
  (hinv : ∀ x, h_inv x = 2 * x + 6)
  (h_comp : ∀ x, h x = (x - 6) / 2) :
  a - b = 5 / 2 :=
sorry

end find_a_minus_b_l9_9122


namespace n_square_divisible_by_144_l9_9537

theorem n_square_divisible_by_144 (n : ℤ) (hn : n > 0)
  (hw : ∃ k : ℤ, n = 12 * k) : ∃ m : ℤ, n^2 = 144 * m :=
by {
  sorry
}

end n_square_divisible_by_144_l9_9537


namespace sector_area_correct_l9_9812

-- Define the initial conditions
def arc_length := 4 -- Length of the arc in cm
def central_angle := 2 -- Central angle in radians
def radius := arc_length / central_angle -- Radius of the circle

-- Define the formula for the area of the sector
def sector_area := (1 / 2) * radius * arc_length

-- The statement of our theorem
theorem sector_area_correct : sector_area = 4 := by
  -- Proof goes here
  sorry

end sector_area_correct_l9_9812


namespace factor_1024_count_l9_9821

theorem factor_1024_count :
  ∃ (n : ℕ), 
  (∀ (a b c : ℕ), (a >= b) → (b >= c) → (2^a * 2^b * 2^c = 1024) → a + b + c = 10) ∧ n = 14 :=
sorry

end factor_1024_count_l9_9821


namespace opposite_of_2023_l9_9446

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l9_9446


namespace correct_insights_l9_9823

def insight1 := ∀ connections : Type, (∃ journey : connections → Prop, ∀ (x : connections), ¬journey x)
def insight2 := ∀ connections : Type, (∃ (beneficial : connections → Prop), ∀ (x : connections), beneficial x → True)
def insight3 := ∀ connections : Type, (∃ (accidental : connections → Prop), ∀ (x : connections), accidental x → False)
def insight4 := ∀ connections : Type, (∃ (conditional : connections → Prop), ∀ (x : connections), conditional x → True)

theorem correct_insights : ¬ insight1 ∧ insight2 ∧ ¬ insight3 ∧ insight4 :=
by sorry

end correct_insights_l9_9823


namespace inequality_range_m_l9_9513

theorem inequality_range_m:
  (∀ x ∈ Set.Icc (Real.sqrt 2) 4, (5 / 2) * x^2 ≥ m * (x - 1)) → m ≤ 10 :=
by 
  intros h 
  sorry

end inequality_range_m_l9_9513


namespace complement_union_equals_l9_9801

def universal_set : Set ℤ := {-2, -1, 0, 1, 2, 3, 4, 5}
def A : Set ℤ := {-1, 0, 1, 2, 3}
def B : Set ℤ := {-2, 0, 2}

def C_I (I : Set ℤ) (s : Set ℤ) : Set ℤ := I \ s

theorem complement_union_equals :
  C_I universal_set (A ∪ B) = {4, 5} :=
by
  sorry

end complement_union_equals_l9_9801


namespace opposite_of_2023_l9_9276

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9276


namespace totalGoals_l9_9669

-- Define the conditions
def louieLastMatchGoals : Nat := 4
def louiePreviousGoals : Nat := 40
def gamesPerSeason : Nat := 50
def seasons : Nat := 3
def brotherGoalsPerGame := 2 * louieLastMatchGoals

-- Define the properties derived from the conditions
def totalBrotherGoals : Nat := brotherGoalsPerGame * gamesPerSeason * seasons
def totalLouieGoals : Nat := louiePreviousGoals + louieLastMatchGoals

-- State what needs to be proved
theorem totalGoals : louiePreviousGoals + louieLastMatchGoals + brotherGoalsPerGame * gamesPerSeason * seasons = 1244 := by
  sorry

end totalGoals_l9_9669


namespace Cindy_walking_speed_l9_9905

noncomputable def walking_speed (total_time : ℕ) (running_speed : ℕ) (running_distance : ℚ) (walking_distance : ℚ) : ℚ := 
  let time_to_run := running_distance / running_speed
  let walking_time := total_time - (time_to_run * 60)
  walking_distance / (walking_time / 60)

theorem Cindy_walking_speed : walking_speed 40 3 0.5 0.5 = 1 := 
  sorry

end Cindy_walking_speed_l9_9905


namespace total_amount_l9_9040

def g_weight : ℝ := 2.5
def g_price : ℝ := 2.79
def r_weight : ℝ := 1.8
def r_price : ℝ := 3.25
def c_weight : ℝ := 1.2
def c_price : ℝ := 4.90
def o_weight : ℝ := 0.9
def o_price : ℝ := 5.75

theorem total_amount :
  g_weight * g_price + r_weight * r_price + c_weight * c_price + o_weight * o_price = 23.88 := by
  sorry

end total_amount_l9_9040


namespace opposite_of_2023_l9_9380

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9380


namespace xy_value_l9_9533

theorem xy_value (x y : ℝ) (h : (x - 3)^2 + |y + 2| = 0) : x * y = -6 :=
by {
  sorry
}

end xy_value_l9_9533


namespace opposite_of_2023_l9_9268

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9268


namespace remainder_sum_15_div_11_l9_9879

theorem remainder_sum_15_div_11 :
  let n := 15 
  let a := 1 
  let l := 15 
  let S := (n * (a + l)) / 2
  S % 11 = 10 :=
by
  let n := 15
  let a := 1
  let l := 15
  let S := (n * (a + l)) / 2
  show S % 11 = 10
  sorry

end remainder_sum_15_div_11_l9_9879


namespace decreasing_interval_of_f_minimum_value_of_f_on_interval_l9_9651

noncomputable def f (a x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

theorem decreasing_interval_of_f :
  ∃ a : ℝ, ∀ x : ℝ, (f a x').deriv < 0 → x < -1 ∨ x > 3 := sorry

theorem minimum_value_of_f_on_interval (a : ℝ) (h_max : ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 2 → f a x ≤ 20) :
  f -2 (-1) = -7 := sorry

end decreasing_interval_of_f_minimum_value_of_f_on_interval_l9_9651


namespace complement_of_M_is_correct_l9_9529

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 3}

-- Define the complement of M in U
def complement_M_in_U : Set ℝ := {x : ℝ | x < -1 ∨ x > 3}

-- State the theorem
theorem complement_of_M_is_correct : (U \ M) = complement_M_in_U := by sorry

end complement_of_M_is_correct_l9_9529


namespace basketball_free_throws_l9_9696

theorem basketball_free_throws
  (a b x : ℕ)
  (h1 : 3 * b = 2 * a)
  (h2 : x = 2 * a)
  (h3 : 2 * a + 3 * b + x = 72)
  : x = 24 := by
  sorry

end basketball_free_throws_l9_9696


namespace equality_of_arithmetic_sums_l9_9780

def sum_arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem equality_of_arithmetic_sums (n : ℕ) (h : n ≠ 0) :
  sum_arithmetic_sequence 8 4 n = sum_arithmetic_sequence 17 2 n ↔ n = 10 :=
by
  sorry

end equality_of_arithmetic_sums_l9_9780


namespace opposite_of_2023_is_neg_2023_l9_9285

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l9_9285


namespace opposite_of_2023_is_neg2023_l9_9317

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l9_9317


namespace opposite_of_2023_l9_9159

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l9_9159


namespace opposite_of_2023_l9_9368

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l9_9368


namespace magic_square_sum_l9_9094

theorem magic_square_sum (S a b c d e : ℤ) (h1 : x + 15 + 100 = S)
                        (h2 : 23 + d + e = S)
                        (h3 : x + a + 23 = S)
                        (h4 : a = 92)
                        (h5 : 92 + b + d = x + 15 + 100)
                        (h6 : b = 0)
                        (h7 : d = 100) : x = 77 :=
by {
  sorry
}

end magic_square_sum_l9_9094


namespace opposite_of_2023_l9_9393

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9393


namespace problem1_problem2_l9_9930

variable (α : ℝ)

-- First problem statement
theorem problem1 (h : Real.tan α = 2) : 
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.sin α + 3 * Real.cos α) = 6 / 13 :=
by 
  sorry

-- Second problem statement
theorem problem2 (h : Real.tan α = 2) :
  3 * (Real.sin α)^2 + 3 * Real.sin α * Real.cos α - 2 * (Real.cos α)^2 = 16 / 5 :=
by 
  sorry

end problem1_problem2_l9_9930


namespace sum_of_coordinates_of_point_B_l9_9566

theorem sum_of_coordinates_of_point_B
  (x y : ℝ)
  (A : (ℝ × ℝ) := (2, 1))
  (B : (ℝ × ℝ) := (x, y))
  (h_line : y = 6)
  (h_slope : (y - 1) / (x - 2) = 4 / 5) :
  x + y = 14.25 :=
by {
  -- convert hypotheses to Lean terms and finish the proof
  sorry
}

end sum_of_coordinates_of_point_B_l9_9566


namespace sum_of_decimals_l9_9621

theorem sum_of_decimals : (5.47 + 4.96) = 10.43 :=
by
  sorry

end sum_of_decimals_l9_9621


namespace P_plus_Q_l9_9949

theorem P_plus_Q (P Q : ℕ) (h1 : 4 / 7 = P / 49) (h2 : 4 / 7 = 84 / Q) : P + Q = 175 :=
by
  sorry

end P_plus_Q_l9_9949


namespace total_marbles_l9_9815

theorem total_marbles (boxes : ℕ) (marbles_per_box : ℕ) (h1 : boxes = 10) (h2 : marbles_per_box = 100) : (boxes * marbles_per_box = 1000) :=
by
  sorry

end total_marbles_l9_9815


namespace subtracted_number_l9_9807

def least_sum_is (x y z : ℤ) (a : ℤ) : Prop :=
  (x - a) * (y - 5) * (z - 2) = 1000 ∧ x + y + z = 7

theorem subtracted_number (x y z a : ℤ) (h : least_sum_is x y z a) : a = 30 :=
sorry

end subtracted_number_l9_9807


namespace food_initially_meant_to_last_22_days_l9_9456

variable (D : ℕ)   -- Denoting the initial number of days the food was meant to last
variable (m : ℕ := 760)  -- Initial number of men
variable (total_men : ℕ := 1520)  -- Total number of men after 2 days

-- The first condition derived from the problem: total amount of food
def total_food := m * D

-- The second condition derived from the problem: Remaining food after 2 days
def remaining_food_after_2_days := total_food - m * 2

-- The third condition derived from the problem: Remaining food to last for 10 more days
def remaining_food_to_last_10_days := total_men * 10

-- Statement to prove
theorem food_initially_meant_to_last_22_days :
  D - 2 = 10 →
  D = 22 :=
by
  sorry

end food_initially_meant_to_last_22_days_l9_9456


namespace opposite_of_2023_l9_9153

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l9_9153


namespace opposite_of_2023_l9_9247

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l9_9247


namespace opposite_of_2023_is_neg2023_l9_9314

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l9_9314


namespace more_uniform_team_l9_9006

-- Define the parameters and the variances
def average_height := 1.85
def variance_team_A := 0.32
def variance_team_B := 0.26

-- Main theorem statement
theorem more_uniform_team : variance_team_B < variance_team_A → "Team B" = "Team with more uniform heights" :=
by
  -- Placeholder for the actual proof
  sorry

end more_uniform_team_l9_9006


namespace opposite_of_2023_is_neg_2023_l9_9241

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l9_9241


namespace opposite_of_2023_l9_9270

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9270


namespace min_value_a_plus_3b_l9_9831

theorem min_value_a_plus_3b (a b : ℝ) (h_positive : 0 < a ∧ 0 < b)
  (h_condition : (1 / (a + 3) + 1 / (b + 3) = 1 / 4)) :
  a + 3 * b ≥ 4 + 8 * Real.sqrt 3 := 
sorry

end min_value_a_plus_3b_l9_9831


namespace opposite_of_2023_l9_9402

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9402


namespace find_angle3_l9_9789

theorem find_angle3 (angle1 angle2 angle3 : ℝ)
  (h1 : angle1 + angle2 = 90)
  (h2 : angle2 + angle3 = 180)
  (h3 : angle1 = 20) :
  angle3 = 110 :=
sorry

end find_angle3_l9_9789


namespace opposite_of_2023_l9_9396

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9396


namespace product_of_possible_values_l9_9953

theorem product_of_possible_values (x : ℚ) (h : abs ((18 : ℚ) / (2 * x) - 4) = 3) : (x = 9 ∨ x = 9/7) → (9 * (9/7) = 81/7) :=
by
  intros
  sorry

end product_of_possible_values_l9_9953


namespace opposite_of_2023_l9_9308

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l9_9308


namespace opposite_of_2023_l9_9143

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l9_9143


namespace william_wins_tic_tac_toe_l9_9887

-- Define the conditions
variables (total_rounds : ℕ) (extra_wins : ℕ) (william_wins : ℕ) (harry_wins : ℕ)

-- Setting the conditions
def william_harry_tic_tac_toe_conditions : Prop :=
  total_rounds = 15 ∧
  extra_wins = 5 ∧
  william_wins = harry_wins + extra_wins ∧
  total_rounds = william_wins + harry_wins

-- The goal is to prove that William won 10 rounds given the conditions above
theorem william_wins_tic_tac_toe : william_harry_tic_tac_toe_conditions total_rounds extra_wins william_wins harry_wins → william_wins = 10 :=
by
  intro h
  have total_rounds_eq := and.left h
  have extra_wins_eq := and.right (and.left (and.right h))
  have william_harry_diff := and.left (and.right (and.right h))
  have total_wins_eq := and.right (and.right (and.right h))
  sorry

end william_wins_tic_tac_toe_l9_9887


namespace focus_of_parabola_x_eq_neg_1_div_4_y_squared_is_neg_1_0_l9_9778

theorem focus_of_parabola_x_eq_neg_1_div_4_y_squared_is_neg_1_0 :
  let P (y : ℝ) := (-1/4 * y^2, y)
  let F := (-1, 0)
  let d := 1
  ∀ (y : ℝ), (P y).fst = -1/4 * y^2 → (F.fst - P y.fst)^2 + (F.snd - P y.snd)^2 = (d + P y.fst)^2 → F = (-1, 0) :=
by
  intros P F d y h1 h2
  sorry

end focus_of_parabola_x_eq_neg_1_div_4_y_squared_is_neg_1_0_l9_9778


namespace min_value_of_expression_l9_9837

noncomputable def problem_statement : Prop :=
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ ((1/x) + (1/y) + (1/z) = 9) ∧ (x^2 * y^3 * z^2 = 1/2268)

theorem min_value_of_expression :
  problem_statement := 
sorry

end min_value_of_expression_l9_9837


namespace interval_of_monotonic_increase_l9_9129

-- Define the function f(x)
noncomputable def f (x: ℝ) : ℝ := log 0.5 (-x^2 - 3*x + 4)

-- Define the interval of monotonic increase we need to prove
theorem interval_of_monotonic_increase : 
    ∀ (x: ℝ), 
    -4 < x -> x < 1 -> (-x^2 - 3*x + 4 > 0) ->
    (log (0.5 : ℝ) (-x^2 - 3*x + 4) = f x) ->
    [(-3/2), 1) := sorry

end interval_of_monotonic_increase_l9_9129


namespace opposite_of_2023_l9_9272

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9272


namespace opposite_of_2023_is_neg2023_l9_9342

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l9_9342


namespace opposite_of_2023_l9_9367

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l9_9367


namespace biker_bob_east_distance_l9_9901

noncomputable def distance_between_towns : ℝ := 28.30194339616981
noncomputable def distance_west : ℝ := 30
noncomputable def distance_north_1 : ℝ := 6
noncomputable def distance_north_2 : ℝ := 18
noncomputable def total_distance_north : ℝ := distance_north_1 + distance_north_2
noncomputable def unknown_distance_east : ℝ := 45.0317 -- Expected distance east

theorem biker_bob_east_distance :
  ∃ (E : ℝ), (total_distance_north ^ 2 + (-distance_west + E) ^ 2 = distance_between_towns ^ 2) ∧ E = unknown_distance_east :=
by 
  sorry

end biker_bob_east_distance_l9_9901


namespace opposite_of_2023_is_neg2023_l9_9321

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l9_9321


namespace problem_correct_options_l9_9681

open ProbabilityTheory

variables {Ω : Type*} [MeasureSpace Ω]
variables {A B : Set Ω} (P : Measure Ω)

theorem problem_correct_options
  (hA : 0 < P A) (hA1 : P A < 1)
  (hB : 0 < P B) (hB1 : P B < 1):
  P[B | A] + P[complement B | A] = 1 :=
begin
  sorry
end

end problem_correct_options_l9_9681


namespace katya_female_classmates_l9_9979

theorem katya_female_classmates (g b : ℕ) (h1 : b = 2 * g) (h2 : b = g + 7) :
  g - 1 = 6 :=
by
  sorry

end katya_female_classmates_l9_9979


namespace opposite_of_2023_l9_9427

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l9_9427


namespace opposite_of_2023_is_neg2023_l9_9350

theorem opposite_of_2023_is_neg2023 : ∀ x : ℤ, x = 2023 → -x = -2023 := by
  intro x h
  rw [h]
  rfl

end opposite_of_2023_is_neg2023_l9_9350


namespace compare_solutions_l9_9553

variables (p q r s : ℝ)
variables (hp : p ≠ 0) (hr : r ≠ 0)

theorem compare_solutions :
  ((-q / p) > (-s / r)) ↔ (s * r > q * p) :=
by sorry

end compare_solutions_l9_9553


namespace opposite_of_2023_l9_9425

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l9_9425


namespace trigonometric_identity_l9_9924

-- Definition for the given condition
def tan_alpha (α : ℝ) : Prop := Real.tan α = 2

-- The proof goal
theorem trigonometric_identity (α : ℝ) (h : tan_alpha α) : 
  Real.cos (π + α) * Real.cos (π / 2 + α) = 2 / 5 :=
by
  sorry

end trigonometric_identity_l9_9924


namespace chess_club_mixed_groups_l9_9720

theorem chess_club_mixed_groups (total_children : ℕ) (num_groups : ℕ)
  (games_per_group : ℕ) (boy_vs_boy_games : ℕ) (girl_vs_girl_games : ℕ) :
  total_children = 90 ∧ num_groups = 30 ∧ games_per_group = 3 ∧ boy_vs_boy_games = 30 ∧ girl_vs_girl_games = 14 →
  by let mixed_games := 90 - 30 - 14 in let mixed_groups := mixed_games / 2 in mixed_groups = 23 :=
begin
  intros h,
  cases h with h1 h234, cases h234 with h2 h34, cases h34 with h3 h4,
  simp at h1 h2 h3 h4,
  have h5 : 46 = 90 - 30 - 14 := by norm_num,
  have h6 : 2 * 23 = 46 := by norm_num,
  exact h6
end

end chess_club_mixed_groups_l9_9720


namespace range_of_theta_l9_9853

-- Conditions
def theta_in_triangle (theta : ℝ) : Prop :=
  0 < theta ∧ theta < real.pi

def function_always_positive (theta : ℝ) : Prop :=
  ∀ x : ℝ, (real.cos theta) * x^2 - 4 * (real.sin theta) * x + 6 > 0

-- Proof statement
theorem range_of_theta
  (theta : ℝ)
  (h₁ : theta_in_triangle theta)
  (h₂ : function_always_positive theta) :
  0 < theta ∧ theta < real.pi / 3 :=
sorry

end range_of_theta_l9_9853


namespace opposite_of_2023_l9_9389

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9389


namespace Chloe_total_score_l9_9048

-- Definitions
def points_per_treasure : ℕ := 9
def treasures_first_level : ℕ := 6
def treasures_second_level : ℕ := 3

-- Statement of the theorem
theorem Chloe_total_score : (points_per_treasure * treasures_first_level) + (points_per_treasure * treasures_second_level) = 81 := by
  sorry

end Chloe_total_score_l9_9048


namespace opposite_of_2023_l9_9436

theorem opposite_of_2023 : ∃ x, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · exact rfl
  · exact rfl

end opposite_of_2023_l9_9436


namespace opposite_of_2023_l9_9299

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l9_9299


namespace grid_X_value_l9_9625

theorem grid_X_value :
  ∃ X, (∃ b d1 d2 d3 d4, 
    b = 16 ∧
    d1 = (25 - 20) ∧
    d2 = (16 - 15) / 3 ∧
    d3 = (d1 * 5) / 4 ∧
    d4 = d1 - d3 ∧
    (-12 - d4 * 4) = -30 ∧ 
    X = d4 ∧
    X = 10.5) :=
sorry

end grid_X_value_l9_9625


namespace total_snacks_l9_9677

variable (peanuts : ℝ) (raisins : ℝ)

theorem total_snacks (h1 : peanuts = 0.1) (h2 : raisins = 0.4) : peanuts + raisins = 0.5 :=
by
  sorry

end total_snacks_l9_9677


namespace jericho_money_left_l9_9458

/--
Given:
1. Twice the money Jericho has is 60.
2. Jericho owes Annika $14.
3. Jericho owes Manny half as much as he owes Annika.

Prove:
Jericho will be left with $9 after paying off all his debts.
-/
theorem jericho_money_left (j_money : ℕ) (annika_owes : ℕ) (manny_multiplier : ℕ) (debt : ℕ) (remaining_money : ℕ) :
  2 * j_money = 60 →
  annika_owes = 14 →
  manny_multiplier = 1 / 2 →
  debt = annika_owes + manny_multiplier * annika_owes →
  remaining_money = j_money - debt →
  remaining_money = 9 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end jericho_money_left_l9_9458


namespace opposite_of_2023_is_neg2023_l9_9320

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l9_9320


namespace opposite_of_2023_l9_9188

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9188


namespace opposite_of_2023_l9_9225

theorem opposite_of_2023 : ∀ x : ℤ, x = 2023 → -x = -2023 :=
by
  intro x hx
  rw [hx]
  apply eq.refl (-2023)

end opposite_of_2023_l9_9225


namespace opposite_of_2023_l9_9261

theorem opposite_of_2023 : -2023 = Int.neg 2023 := 
by
  rw Int.neg_eq_neg
  refl

end opposite_of_2023_l9_9261


namespace opposite_of_2023_l9_9207

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l9_9207


namespace annual_income_is_32000_l9_9713

noncomputable def compute_tax (p A: ℝ) : ℝ := 
  0.01 * p * 28000 + 0.01 * (p + 2) * (A - 28000)

noncomputable def stated_tax (p A: ℝ) : ℝ := 
  0.01 * (p + 0.25) * A

theorem annual_income_is_32000 (p : ℝ) (A : ℝ) :
  compute_tax p A = stated_tax p A → A = 32000 :=
by
  intros h
  have : 0.01 * p * 28000 + 0.01 * (p + 2) * (A - 28000) = 0.01 * (p + 0.25) * A := h
  sorry

end annual_income_is_32000_l9_9713


namespace mps_to_kmph_conversion_l9_9018

/-- Define the conversion factor from meters per second to kilometers per hour. -/
def mps_to_kmph : ℝ := 3.6

/-- Define the speed in meters per second. -/
def speed_mps : ℝ := 5

/-- Define the converted speed in kilometers per hour. -/
def speed_kmph : ℝ := 18

/-- Statement asserting the conversion from meters per second to kilometers per hour. -/
theorem mps_to_kmph_conversion : speed_mps * mps_to_kmph = speed_kmph := by 
  sorry

end mps_to_kmph_conversion_l9_9018


namespace selection_assignment_ways_l9_9603

-- Define the group of students
def male_students : ℕ := 4
def female_students : ℕ := 3

-- Define the selection conditions
def selected_people : ℕ := 4
def min_females : ℕ := 2

-- Prove the number of different ways to select and assign the individuals
theorem selection_assignment_ways : (C 4 2 * C 3 2 + C 3 3 * C 4 1) * (C 4 2 * A 3 3) = 792 :=
by
  -- Sorry to omit the proof details
  sorry

end selection_assignment_ways_l9_9603


namespace problem_inequality_l9_9983

theorem problem_inequality 
  (a b c : ℝ) 
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c)
  (h : a^4 + b^4 + c^4 ≤ 2 * (a^2 * b^2 + b^2 * c^2 + c^2 * a^2)) : 
  a^2 + b^2 + c^2 ≤ 2 * (a * b + b * c + c * a) :=
by
  sorry

end problem_inequality_l9_9983


namespace ReuleauxTriangleFitsAll_l9_9015

-- Assume definitions for fits into various slots

def FitsTriangular (s : Type) : Prop := sorry
def FitsSquare (s : Type) : Prop := sorry
def FitsCircular (s : Type) : Prop := sorry
def ReuleauxTriangle (s : Type) : Prop := sorry

theorem ReuleauxTriangleFitsAll (s : Type) (h : ReuleauxTriangle s) : 
  FitsTriangular s ∧ FitsSquare s ∧ FitsCircular s := 
  sorry

end ReuleauxTriangleFitsAll_l9_9015


namespace ab_sum_l9_9951

theorem ab_sum (a b : ℝ) (h : |a - 4| + (b + 1)^2 = 0) : a + b = 3 :=
by
  sorry -- this is where the proof would go

end ab_sum_l9_9951


namespace opposite_of_2023_l9_9444

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l9_9444


namespace cricket_runs_l9_9890

theorem cricket_runs (x a b c d : ℕ) 
    (h1 : a = 1 * x) 
    (h2 : b = 3 * x) 
    (h3 : c = 5 * x) 
    (h4 : d = 4 * x) 
    (total_runs : 1 * x + 3 * x + 5 * x + 4 * x = 234) :
  a = 18 ∧ b = 54 ∧ c = 90 ∧ d = 72 := by
  sorry

end cricket_runs_l9_9890


namespace proof_problem_l9_9833

variable (a b c d : ℝ)
variable (ω : ℂ)

-- Conditions
def conditions : Prop :=
  a ≠ -1 ∧ b ≠ -1 ∧ c ≠ -1 ∧ d ≠ -1 ∧
  ω^4 = 1 ∧ ω ≠ 1 ∧
  (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 4 / ω^2)

theorem proof_problem (h : conditions a b c d ω) :
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1)) = 2 := 
sorry

end proof_problem_l9_9833


namespace total_trip_length_is570_l9_9485

theorem total_trip_length_is570 (v D : ℝ) (h1 : (2:ℝ) + (2/3) + (6 * (D - 2 * v) / (5 * v)) = 2.75)
(h2 : (2:ℝ) + (50 / v) + (2/3) + (6 * (D - 2 * v - 50) / (5 * v)) = 2.33) :
D = 570 :=
sorry

end total_trip_length_is570_l9_9485


namespace number_increased_by_one_fourth_l9_9732

theorem number_increased_by_one_fourth (n : ℕ) (h : 25 * 80 / 100 = 20) (h1 : 80 - 20 = 60) :
  n + n / 4 = 60 ↔ n = 48 :=
by
  -- Conditions
  have h2 : 80 - 25 * 80 / 100 = 60 := by linarith [h, h1]
  have h3 : n + n / 4 = 60 := sorry
  -- Assertion (Proof to show is omitted)
  sorry

end number_increased_by_one_fourth_l9_9732


namespace inequality_holds_l9_9997

theorem inequality_holds (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) : 
  ((2 + x)/(1 + x))^2 + ((2 + y)/(1 + y))^2 ≥ 9/2 := 
sorry

end inequality_holds_l9_9997


namespace opposite_of_2023_l9_9416

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l9_9416


namespace time_on_sideline_l9_9969

def total_game_time : ℕ := 90
def time_mark_played_first_period : ℕ := 20
def time_mark_played_second_period : ℕ := 35
def total_time_mark_played : ℕ := time_mark_played_first_period + time_mark_played_second_period

theorem time_on_sideline : total_game_time - total_time_mark_played = 35 := by
  sorry

end time_on_sideline_l9_9969


namespace william_wins_10_rounds_l9_9885

-- Definitions from the problem conditions
variable (W H : ℕ)
variable (total_rounds : ℕ := 15)
variable (additional_wins : ℕ := 5)

-- Conditions
def total_game_condition : Prop := W + H = total_rounds
def win_difference_condition : Prop := W = H + additional_wins

-- Statement to be proved
theorem william_wins_10_rounds (h1 : total_game_condition W H) (h2 : win_difference_condition W H) : W = 10 :=
by
  sorry

end william_wins_10_rounds_l9_9885


namespace first_player_always_wins_l9_9009

theorem first_player_always_wins (A B : ℤ) (hA : A ≠ 0) (hB : B ≠ 0) : A + B + 1998 = 0 → 
  (∃ (a b c : ℤ), (a = A ∨ a = B ∨ a = 1998) ∧ (b = A ∨ b = B ∨ b = 1998) ∧ (c = A ∨ c = B ∨ c = 1998) ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  (∃ (r1 r2 : ℚ), r1 ≠ r2 ∧ r1 * r1 * a + r1 * b + c = 0 ∧ r2 * r2 * a + r2 * b + c = 0)) :=
sorry

end first_player_always_wins_l9_9009


namespace opposite_of_2023_l9_9182

def opposite (x : Int) : Int :=
  -x

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9182


namespace max_k_inequality_l9_9636

theorem max_k_inequality (a b c d : ℝ) (h₀ : 0 ≤ a) (h₁ : a ≤ 1) 
                                      (h₂ : 0 ≤ b) (h₃ : b ≤ 1) 
                                      (h₄ : 0 ≤ c) (h₅ : c ≤ 1) 
                                      (h₆ : 0 ≤ d) (h₇ : d ≤ 1) :
  a^2 * b + b^2 * c + c^2 * d + d^2 * a + 4 ≥ 2 * (a^2 + b^2 + c^2 + d^2) :=
sorry

end max_k_inequality_l9_9636


namespace opposite_of_2023_l9_9171

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l9_9171


namespace fraction_not_on_time_l9_9891

theorem fraction_not_on_time (total_attendees : ℕ) (male_fraction female_fraction male_on_time_fraction female_on_time_fraction : ℝ)
  (H1 : male_fraction = 3/5)
  (H2 : male_on_time_fraction = 7/8)
  (H3 : female_on_time_fraction = 4/5)
  : ((1 - (male_fraction * male_on_time_fraction + (1 - male_fraction) * female_on_time_fraction)) = 3/20) :=
sorry

end fraction_not_on_time_l9_9891


namespace geometric_sequence_product_proof_l9_9971

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

theorem geometric_sequence_product_proof (a : ℕ → ℝ) (q : ℝ)
  (h_geo : geometric_sequence a q) 
  (h1 : a 2010 * a 2011 * a 2012 = 3)
  (h2 : a 2013 * a 2014 * a 2015 = 24) :
  a 2016 * a 2017 * a 2018 = 192 :=
sorry

end geometric_sequence_product_proof_l9_9971


namespace find_area_of_plot_l9_9124

def area_of_plot (B : ℝ) (L : ℝ) (A : ℝ) : Prop :=
  L = 0.75 * B ∧ B = 21.908902300206645 ∧ A = L * B

theorem find_area_of_plot (B L A : ℝ) (h : area_of_plot B L A) : A = 360 := by
  sorry

end find_area_of_plot_l9_9124


namespace opposite_of_2023_l9_9157

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l9_9157


namespace fewer_students_played_thursday_l9_9110

variable (w t : ℕ)

theorem fewer_students_played_thursday (h1 : w = 37) (h2 : w + t = 65) : w - t = 9 :=
by
  sorry

end fewer_students_played_thursday_l9_9110


namespace william_wins_10_rounds_l9_9886

-- Definitions from the problem conditions
variable (W H : ℕ)
variable (total_rounds : ℕ := 15)
variable (additional_wins : ℕ := 5)

-- Conditions
def total_game_condition : Prop := W + H = total_rounds
def win_difference_condition : Prop := W = H + additional_wins

-- Statement to be proved
theorem william_wins_10_rounds (h1 : total_game_condition W H) (h2 : win_difference_condition W H) : W = 10 :=
by
  sorry

end william_wins_10_rounds_l9_9886


namespace opposite_of_2023_l9_9175

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l9_9175


namespace opposite_of_2023_l9_9134

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l9_9134


namespace opposite_of_2023_l9_9421

theorem opposite_of_2023 : 2023 + (-2023) = 0 :=
by
  sorry

end opposite_of_2023_l9_9421


namespace painting_clock_57_painting_clock_1913_l9_9972

-- Part (a)
theorem painting_clock_57 (h : ∀ n : ℕ, (n = 12 ∨ (exists k : ℕ, n = (12 + k * 57) % 12))) :
  ∃ m : ℕ, m = 4 :=
by { sorry }

-- Part (b)
theorem painting_clock_1913 (h : ∀ n : ℕ, (n = 12 ∨ (exists k : ℕ, n = (12 + k * 1913) % 12))) :
  ∃ m : ℕ, m = 12 :=
by { sorry }

end painting_clock_57_painting_clock_1913_l9_9972


namespace pipes_height_l9_9920

theorem pipes_height (d : ℝ) (h : ℝ) (r : ℝ) (s : ℝ)
  (hd : d = 12)
  (hs : s = d)
  (hr : r = d / 2)
  (heq : h = 6 * Real.sqrt 3 + r) :
  h = 6 * Real.sqrt 3 + 6 :=
by
  sorry

end pipes_height_l9_9920


namespace line_passes_fixed_point_l9_9925

theorem line_passes_fixed_point (a b : ℝ) (h : a + 2 * b = 1) : 
  a * (1/2) + 3 * (-1/6) + b = 0 :=
by
  sorry

end line_passes_fixed_point_l9_9925


namespace opposite_of_2023_l9_9376

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9376


namespace average_speed_l9_9128

theorem average_speed (D T : ℝ) (h1 : D = 100) (h2 : T = 6) : (D / T) = 50 / 3 := by
  sorry

end average_speed_l9_9128


namespace volume_in_region_l9_9549

def satisfies_conditions (x y : ℝ) : Prop :=
  |8 - x| + y ≤ 10 ∧ 3 * y - x ≥ 15

def in_region (x y : ℝ) : Prop :=
  satisfies_conditions x y

theorem volume_in_region (x y p m n : ℕ) (hx : x ≠ 0) (hy : y ≠ 0) (hn : n ≠ 0) (V : ℝ) 
  (hvol : V = (m * Real.pi) / (n * Real.sqrt p))
  (hprime : m.gcd n = 1 ∧ ¬(∃ k : ℕ, k^2 ∣ p ∧ k ≥ 2)) 
  (hpoints : ∀ (x y : ℝ), in_region x y → 3 * y - x = 15) : 
  m + n + p = 365 := 
sorry

end volume_in_region_l9_9549


namespace constant_function_of_inequality_l9_9633

theorem constant_function_of_inequality (f : ℝ → ℝ) 
  (h : ∀ x y z : ℝ, f (x + y) + f (y + z) + f (z + x) ≥ 3 * f (x + 2 * y + 3 * z)) : 
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
sorry

end constant_function_of_inequality_l9_9633


namespace geometric_sequence_ratio_l9_9683

theorem geometric_sequence_ratio (a1 : ℕ) (S : ℕ → ℕ) (q : ℕ) (h1 : q = 2)
  (h2 : ∀ n, S n = a1 * (1 - q ^ (n + 1)) / (1 - q)) :
  S 4 / S 2 = 5 :=
by
  sorry

end geometric_sequence_ratio_l9_9683


namespace opposite_of_2023_l9_9359

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l9_9359


namespace opposite_of_2023_l9_9206

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end opposite_of_2023_l9_9206


namespace compute_remainder_l9_9830

/-- T is the sum of all three-digit positive integers 
  where the digits are distinct, the hundreds digit is at least 2,
  and the digit 1 is not used in any place. -/
def T : ℕ := 
  let hundreds_sum := (2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) * 56 * 100
  let tens_sum := (2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) * 49 * 10
  let units_sum := (2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) * 49
  hundreds_sum + tens_sum + units_sum

/-- Theorem: Compute the remainder when T is divided by 1000. -/
theorem compute_remainder : T % 1000 = 116 := by
  sorry

end compute_remainder_l9_9830


namespace xy_difference_l9_9083

theorem xy_difference (x y : ℝ) (h1 : x + y = 6) (h2 : x^2 - y^2 = 12) : x - y = 2 := by
  sorry

end xy_difference_l9_9083


namespace shortest_tree_height_proof_l9_9005

def tallest_tree_height : ℕ := 150
def middle_tree_height : ℕ := (2 * tallest_tree_height) / 3
def shortest_tree_height : ℕ := middle_tree_height / 2

theorem shortest_tree_height_proof : shortest_tree_height = 50 := by
  sorry

end shortest_tree_height_proof_l9_9005


namespace point_on_transformed_graph_l9_9791

variable (f : ℝ → ℝ)

theorem point_on_transformed_graph :
  (f 12 = 10) →
  3 * (19 / 9) = (f (3 * 4)) / 3 + 3 ∧ (4 + 19 / 9 = 55 / 9) :=
by
  sorry

end point_on_transformed_graph_l9_9791


namespace opposite_of_2023_l9_9163

theorem opposite_of_2023 :
  ∃ x : Int, 2023 + x = 0 ∧ x = -2023 :=
by
  use -2023
  split
  · simp [add_eq_zero_iff_neg_eq]  
  · rfl


end opposite_of_2023_l9_9163


namespace percentage_of_students_in_band_l9_9045

theorem percentage_of_students_in_band 
  (students_in_band : ℕ)
  (total_students : ℕ)
  (students_in_band_eq : students_in_band = 168)
  (total_students_eq : total_students = 840) :
  (students_in_band / total_students : ℚ) * 100 = 20 :=
by
  sorry

end percentage_of_students_in_band_l9_9045


namespace problem_l9_9640

variable (α : ℝ)

def setA : Set ℝ := {Real.sin α, Real.cos α, 1}
def setB : Set ℝ := {Real.sin α ^ 2, Real.sin α + Real.cos α, 0}
theorem problem (h : setA α = setB α) : Real.sin α ^ 2009 + Real.cos α ^ 2009 = -1 := 
by 
  sorry

end problem_l9_9640


namespace opposite_of_2023_l9_9294

theorem opposite_of_2023 : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  use (-2023)
  constructor
  . exact eq.refl (-2023)
  . linarith

end opposite_of_2023_l9_9294


namespace imaginary_part_of_z_l9_9955

open Complex

theorem imaginary_part_of_z (z : ℂ) (h : (3 - 4 * I) * z = abs (4 + 3 * I)) : im z = 4 / 5 :=
sorry

end imaginary_part_of_z_l9_9955


namespace intersection_complement_l9_9838

def M (x : ℝ) : Prop := x^2 - 2 * x < 0
def N (x : ℝ) : Prop := x < 1

theorem intersection_complement (x : ℝ) :
  (M x ∧ ¬N x) ↔ (1 ≤ x ∧ x < 2) := 
sorry

end intersection_complement_l9_9838


namespace opposite_of_2023_l9_9142

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l9_9142


namespace opposite_of_2023_l9_9440

theorem opposite_of_2023 : -2023 = - (2023) :=
by
  sorry

end opposite_of_2023_l9_9440


namespace money_distribution_l9_9779

theorem money_distribution (a b : ℝ) 
  (h1 : 4 * a - b = 40)
  (h2 : 6 * a + b = 110) :
  a = 15 ∧ b = 20 :=
by
  sorry

end money_distribution_l9_9779


namespace opposite_of_2023_l9_9264

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l9_9264


namespace opposite_of_2023_l9_9366

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l9_9366


namespace degree_of_polynomial_raised_to_power_l9_9012

def polynomial_degree (p : Polynomial ℤ) : ℕ := p.natDegree

theorem degree_of_polynomial_raised_to_power :
  let p : Polynomial ℤ := Polynomial.C 5 * Polynomial.X ^ 3 + Polynomial.C 7
  in polynomial_degree (p ^ 15) = 45 :=
by
  sorry

end degree_of_polynomial_raised_to_power_l9_9012


namespace find_a10_l9_9512

variable {G : Type*} [LinearOrderedField G]
variable (a : ℕ → G)

-- Conditions
def geometric_sequence (a : ℕ → G) (r : G) := ∀ n, a (n + 1) = r * a n
def positive_terms (a : ℕ → G) := ∀ n, 0 < a n
def specific_condition (a : ℕ → G) := a 3 * a 11 = 16

theorem find_a10
  (h_geom : geometric_sequence a 2)
  (h_pos : positive_terms a)
  (h_cond : specific_condition a) :
  a 10 = 32 := by
  sorry

end find_a10_l9_9512


namespace tan_alpha_value_l9_9058

theorem tan_alpha_value (α : ℝ) (h : Real.tan (π / 4 + α) = 1 / 2) : Real.tan α = -1 / 3 := 
by
  sorry

end tan_alpha_value_l9_9058


namespace expected_value_of_sixes_l9_9870

theorem expected_value_of_sixes (n : ℕ) (k : ℕ) (p q : ℚ) 
  (h1 : n = 3) 
  (h2 : k = 6)
  (h3 : p = 1/6) 
  (h4 : q = 5/6) : 
  (1 : ℚ) / 2 = ∑ i in finset.range (n + 1), (i * (nat.choose n i * p^i * q^(n-i))) := 
sorry

end expected_value_of_sixes_l9_9870


namespace find_a_l9_9545

noncomputable def triangle_side (a b c : ℝ) (A : ℝ) (area : ℝ) : ℝ :=
if b + c = 2 * Real.sqrt 3 ∧ A = Real.pi / 3 ∧ area = Real.sqrt 3 / 2 then
  Real.sqrt (b^2 + c^2 - 2 * b * c * Real.cos A)
else 0

theorem find_a (b c : ℝ) (h1 : b + c = 2 * Real.sqrt 3) (h2 : Real.cos (Real.pi / 3) = 1 / 2) (area : ℝ)
  (h3 : area = Real.sqrt 3 / 2)
  (a := triangle_side (Real.sqrt 6) b c (Real.pi / 3) (Real.sqrt 3 / 2)) :
  a = Real.sqrt 6 :=
sorry

end find_a_l9_9545
