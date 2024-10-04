import Mathlib

namespace smallest_N_l194_194046

theorem smallest_N (N : ℕ) : 
  (N = 484) ∧ 
  (∃ k : ℕ, 484 = 4 * k) ∧
  (∃ k : ℕ, 485 = 25 * k) ∧
  (∃ k : ℕ, 486 = 9 * k) ∧
  (∃ k : ℕ, 487 = 121 * k) :=
by
  -- Proof omitted (replaced by sorry)
  sorry

end smallest_N_l194_194046


namespace max_ab_min_inv_a_plus_4_div_b_l194_194410

theorem max_ab (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_sum : a + 4 * b = 4) : 
  ab ≤ 1 :=
by
  sorry

theorem min_inv_a_plus_4_div_b (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_sum : a + 4 * b = 4) :
  1 / a + 4 / b ≥ 25 / 4 :=
by
  sorry

end max_ab_min_inv_a_plus_4_div_b_l194_194410


namespace terry_current_age_l194_194931

theorem terry_current_age (T : ℕ) (nora_current_age : ℕ) (h1 : nora_current_age = 10)
  (h2 : T + 10 = 4 * nora_current_age) : T = 30 :=
by
  sorry

end terry_current_age_l194_194931


namespace find_x_l194_194403

theorem find_x (x : ℝ) (h : 3.5 * ( (3.6 * 0.48 * 2.50) / (x * 0.09 * 0.5) ) = 2800.0000000000005) : x = 0.3 :=
sorry

end find_x_l194_194403


namespace evaluate_pow_l194_194702

theorem evaluate_pow : (-64 : ℝ)^(4/3) = 256 := 
by
  sorry

end evaluate_pow_l194_194702


namespace initial_tomatoes_l194_194537

/-- 
Given the conditions:
  - The farmer picked 134 tomatoes yesterday.
  - The farmer picked 30 tomatoes today.
  - The farmer will have 7 tomatoes left after today.
Prove that the initial number of tomatoes in the farmer's garden was 171.
--/

theorem initial_tomatoes (picked_yesterday : ℕ) (picked_today : ℕ) (left_tomatoes : ℕ)
  (h1 : picked_yesterday = 134)
  (h2 : picked_today = 30)
  (h3 : left_tomatoes = 7) :
  (picked_yesterday + picked_today + left_tomatoes) = 171 :=
by 
  sorry

end initial_tomatoes_l194_194537


namespace g_at_6_l194_194946

def g (x : ℝ) : ℝ := 2 * x^4 - 13 * x^3 + 28 * x^2 - 32 * x - 48

theorem g_at_6 : g 6 = 552 :=
by sorry

end g_at_6_l194_194946


namespace jacob_total_bill_l194_194667

def base_cost : ℝ := 25
def included_hours : ℕ := 25
def cost_per_text : ℝ := 0.08
def cost_per_extra_minute : ℝ := 0.13
def jacob_texts : ℕ := 150
def jacob_hours : ℕ := 31

theorem jacob_total_bill : 
  let extra_minutes := (jacob_hours - included_hours) * 60
  let total_cost := base_cost + jacob_texts * cost_per_text + extra_minutes * cost_per_extra_minute
  total_cost = 83.80 := 
by 
  -- Placeholder for proof
  sorry

end jacob_total_bill_l194_194667


namespace sin_45_degree_l194_194326

theorem sin_45_degree : sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_degree_l194_194326


namespace correctly_calculated_expression_l194_194658

theorem correctly_calculated_expression (x : ℝ) :
  ¬ (x^3 + x^2 = x^5) ∧ 
  ¬ (x^3 * x^2 = x^6) ∧ 
  (x^3 / x^2 = x) ∧ 
  ¬ ((x^3)^2 = x^9) := by
sorry

end correctly_calculated_expression_l194_194658


namespace find_bloom_day_l194_194977

def days := {d : Fin 7 // 1 ≤ d.val ∧ d.val ≤ 7}

def sunflowers_bloom (d : days) : Prop :=
¬ (d.val = 2 ∨ d.val = 4 ∨ d.val = 7)

def lilies_bloom (d : days) : Prop :=
¬ (d.val = 4 ∨ d.val = 6)

def magnolias_bloom (d : days) : Prop :=
¬ (d.val = 7)

def all_bloom_together (d : days) : Prop :=
sunflowers_bloom d ∧ lilies_bloom d ∧ magnolias_bloom d

def blooms_simultaneously (d : days) : Prop :=
∀ d1 d2 d3 : days, (d1 = d ∧ d2 = d ∧ d3 = d) →
(all_bloom_together d1 ∧ all_bloom_together d2 ∧ all_bloom_together d3)

theorem find_bloom_day :
  ∃ d : days, blooms_simultaneously d :=
sorry

end find_bloom_day_l194_194977


namespace sin_45_degree_eq_sqrt2_div_2_l194_194352

theorem sin_45_degree_eq_sqrt2_div_2 :
  let θ := (real.pi / 4)
  in sin θ = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_degree_eq_sqrt2_div_2_l194_194352


namespace find_unknown_towel_rate_l194_194871

theorem find_unknown_towel_rate 
    (cost_known1 : ℕ := 300)
    (cost_known2 : ℕ := 750)
    (total_towels : ℕ := 10)
    (average_price : ℕ := 150)
    (total_cost : ℕ := total_towels * average_price) :
  let total_cost_known := cost_known1 + cost_known2
  let cost_unknown := 2 * x
  300 + 750 + 2 * x = total_cost → x = 225 :=
by
  sorry

end find_unknown_towel_rate_l194_194871


namespace sum_of_four_consecutive_even_integers_l194_194809

theorem sum_of_four_consecutive_even_integers (x : ℕ) (hx : x > 4) :
  (x - 4) * (x - 2) * x * (x + 2) = 48 * (4 * x) → (x - 4) + (x - 2) + x + (x + 2) = 28 := by
{
  sorry
}

end sum_of_four_consecutive_even_integers_l194_194809


namespace smallest_number_in_set_l194_194548

open Real

theorem smallest_number_in_set :
  ∀ (a b c d : ℝ), a = -3 → b = 3⁻¹ → c = -abs (-1 / 3) → d = 0 →
    a < b ∧ a < c ∧ a < d :=
by
  intros a b c d ha hb hc hd
  rw [ha, hb, hc, hd]
  sorry

end smallest_number_in_set_l194_194548


namespace greatest_integer_less_than_neg22_div_3_l194_194499

def greatest_integer_less_than (x : ℝ) : ℤ :=
  int.floor x

theorem greatest_integer_less_than_neg22_div_3 : greatest_integer_less_than (-22 / 3) = -8 := by
  sorry

end greatest_integer_less_than_neg22_div_3_l194_194499


namespace man_walking_speed_l194_194867

theorem man_walking_speed (length_of_bridge : ℝ) (time_to_cross : ℝ) 
  (h1 : length_of_bridge = 1250) (h2 : time_to_cross = 15) : 
  (length_of_bridge / time_to_cross) * (60 / 1000) = 5 := 
sorry

end man_walking_speed_l194_194867


namespace num_factors_of_90_multiple_of_6_l194_194430

def is_factor (m n : ℕ) : Prop := n % m = 0
def is_multiple_of (m n : ℕ) : Prop := n % m = 0

theorem num_factors_of_90_multiple_of_6 : 
  ∃ (count : ℕ), count = 4 ∧ ∀ x, is_factor x 90 → is_multiple_of 6 x → x > 0 :=
sorry

end num_factors_of_90_multiple_of_6_l194_194430


namespace fraction_to_decimal_l194_194886

theorem fraction_to_decimal : (58 : ℚ) / 160 = 0.3625 := 
by sorry

end fraction_to_decimal_l194_194886


namespace train_speed_l194_194251

noncomputable def speed_of_train (length_of_train length_of_overbridge time: ℝ) : ℝ :=
  (length_of_train + length_of_overbridge) / time

theorem train_speed (length_of_train length_of_overbridge time speed: ℝ)
  (h1 : length_of_train = 600)
  (h2 : length_of_overbridge = 100)
  (h3 : time = 70)
  (h4 : speed = 10) :
  speed_of_train length_of_train length_of_overbridge time = speed :=
by
  simp [speed_of_train, h1, h2, h3, h4]
  sorry

end train_speed_l194_194251


namespace joan_mortgage_payment_l194_194455

noncomputable def geometric_series_sum (a r : ℕ) (n : ℕ) : ℕ :=
  a * (1 - r^n) / (1 - r)

theorem joan_mortgage_payment : 
  ∃ n : ℕ, geometric_series_sum 100 3 n = 109300 ∧ n = 7 :=
by
  sorry

end joan_mortgage_payment_l194_194455


namespace smallest_consecutive_divisible_by_17_l194_194005

def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_consecutive_divisible_by_17 :
  ∃ (n m : ℕ), 
    (m = n + 1) ∧
    sum_digits n % 17 = 0 ∧ 
    sum_digits m % 17 = 0 ∧ 
    n = 8899 ∧ 
    m = 8900 := 
by
  sorry

end smallest_consecutive_divisible_by_17_l194_194005


namespace probability_heart_then_club_l194_194822

noncomputable def numHearts : ℕ := 13
noncomputable def numClubs : ℕ := 13
noncomputable def totalCards (n : ℕ) : ℕ := 52 - n

noncomputable def probabilityFirstHeart : ℚ := numHearts / totalCards 0
noncomputable def probabilitySecondClubGivenFirstHeart : ℚ := numClubs / totalCards 1

theorem probability_heart_then_club :
  (probabilityFirstHeart * probabilitySecondClubGivenFirstHeart) = 13 / 204 :=
by
  sorry

end probability_heart_then_club_l194_194822


namespace time_without_moving_walkway_l194_194511

/--
Assume a person walks from one end to the other of a 90-meter long moving walkway at a constant rate in 30 seconds, assisted by the walkway. When this person reaches the end, they reverse direction and continue walking with the same speed, but this time it takes 120 seconds because the person is traveling against the direction of the moving walkway.

Prove that if the walkway were to stop moving, it would take this person 48 seconds to walk from one end of the walkway to the other.
-/
theorem time_without_moving_walkway : 
  ∀ (v_p v_w : ℝ),
  (v_p + v_w) * 30 = 90 →
  (v_p - v_w) * 120 = 90 →
  90 / v_p = 48 :=
by
  intros v_p v_w h1 h2
  have hpw := eq_of_sub_eq_zero (sub_eq_zero.mpr h1)
  have hmw := eq_of_sub_eq_zero (sub_eq_zero.mpr h2)
  sorry

end time_without_moving_walkway_l194_194511


namespace sin_eleven_pi_over_three_l194_194896

theorem sin_eleven_pi_over_three : Real.sin (11 * Real.pi / 3) = -((Real.sqrt 3) / 2) :=
by
  -- Conversion factor between radians and degrees
  -- periodicity of sine function: sin theta = sin (theta + n * 360 degrees) for any integer n
  -- the sine function is odd: sin (-theta) = -sin theta
  -- sin 60 degrees = sqrt(3)/2
  sorry

end sin_eleven_pi_over_three_l194_194896


namespace sin_cos_unique_solution_l194_194052

theorem sin_cos_unique_solution (α : ℝ) (hα1 : 0 < α) (hα2 : α < (π / 2)) :
  ∃! x : ℝ, (Real.sin α) ^ x + (Real.cos α) ^ x = 1 :=
sorry

end sin_cos_unique_solution_l194_194052


namespace polynomial_divisible_by_5040_l194_194114

theorem polynomial_divisible_by_5040 (n : ℤ) (hn : n > 3) :
  5040 ∣ (n^7 - 14 * n^5 + 49 * n^3 - 36 * n) :=
sorry

end polynomial_divisible_by_5040_l194_194114


namespace sum_of_distinct_digits_l194_194057

theorem sum_of_distinct_digits
  (w x y z : ℕ)
  (h1 : y + w = 10)
  (h2 : x + y = 9)
  (h3 : w + z = 10)
  (h4 : w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z)
  (hw : w < 10) (hx : x < 10) (hy : y < 10) (hz : z < 10) :
  w + x + y + z = 20 := sorry

end sum_of_distinct_digits_l194_194057


namespace inequality_holds_for_all_x_l194_194729

theorem inequality_holds_for_all_x : 
  ∀ (a : ℝ), (∀ (x : ℝ), |x| ≤ 1 → x^2 - (a + 1) * x + a + 1 > 0) ↔ a < -1 := 
sorry

end inequality_holds_for_all_x_l194_194729


namespace sin_45_deg_eq_one_div_sqrt_two_l194_194332

def unit_circle_radius : ℝ := 1

def forty_five_degrees_in_radians : ℝ := (Real.pi / 4)

def cos_45 : ℝ := Real.cos forty_five_degrees_in_radians

def sin_45 : ℝ := Real.sin forty_five_degrees_in_radians

theorem sin_45_deg_eq_one_div_sqrt_two : 
  sin_45 = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_deg_eq_one_div_sqrt_two_l194_194332


namespace intersection_point_l194_194002

def L1 (x y : ℚ) : Prop := y = -3 * x
def L2 (x y : ℚ) : Prop := y + 4 = 9 * x

theorem intersection_point : ∃ x y : ℚ, L1 x y ∧ L2 x y ∧ x = 1/3 ∧ y = -1 := sorry

end intersection_point_l194_194002


namespace sufficient_condition_not_monotonic_l194_194746

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - 4 * a * x - Real.log x

def sufficient_not_monotonic (a : ℝ) : Prop :=
  (a > 1 / 6) ∨ (a < -1 / 2)

theorem sufficient_condition_not_monotonic (a : ℝ) :
  sufficient_not_monotonic a → ¬(∀ x y : ℝ, 1 < x ∧ x < 3 ∧ 1 < y ∧ y < 3 ∧ x ≠ y → ((f a x - f a y) / (x - y) ≥ 0 ∨ (f a y - f a x) / (y - x) ≥ 0)) :=
by
  sorry

end sufficient_condition_not_monotonic_l194_194746


namespace find_principal_amount_l194_194853

variable (x y : ℝ)

-- conditions given in the problem
def simple_interest_condition : Prop :=
  600 = (x * y * 2) / 100

def compound_interest_condition : Prop :=
  615 = x * ((1 + y / 100)^2 - 1)

-- target statement to be proven
theorem find_principal_amount (h1 : simple_interest_condition x y) (h2 : compound_interest_condition x y) :
  x = 285.7142857 :=
  sorry

end find_principal_amount_l194_194853


namespace witch_votes_is_seven_l194_194768

-- Definitions
def votes_for_witch (W : ℕ) : ℕ := W
def votes_for_unicorn (W : ℕ) : ℕ := 3 * W
def votes_for_dragon (W : ℕ) : ℕ := W + 25
def total_votes (W : ℕ) : ℕ := votes_for_witch W + votes_for_unicorn W + votes_for_dragon W

-- Proof Statement
theorem witch_votes_is_seven (W : ℕ) (h1 : total_votes W = 60) : W = 7 :=
by
  sorry

end witch_votes_is_seven_l194_194768


namespace cubic_eq_solutions_l194_194854

theorem cubic_eq_solutions (x : ℝ) :
  x^3 - 4 * x = 0 ↔ x = 0 ∨ x = -2 ∨ x = 2 := by
  sorry

end cubic_eq_solutions_l194_194854


namespace island_connectivity_after_years_l194_194736

noncomputable def ferry_network (n : ℕ) (h : n ≥ 3) : Prop :=
  ∀ G : SimpleGraph (Fin n),
    (∀ s t : Finset (Fin n), s ∪ t = Finset.univ → s ≠ ∅ → t ≠ ∅ → ∃ (x : Fin n), G.exists_edge (x, G.neighbors x)) →
    (∃ x : Fin n, ∀ y : Fin n, y ≠ x → x ∈ G.neighbors y)

theorem island_connectivity_after_years (n : ℕ) (h : n ≥ 3) :
  ferry_network n h :=
begin
  sorry
end

end island_connectivity_after_years_l194_194736


namespace solve_inequality_l194_194217

theorem solve_inequality (a x : ℝ) : 
  (a = 0 ∧ x ≤ -1) ∨ 
  (a > 0 ∧ (x ≥ 2 / a ∨ x ≤ -1)) ∨ 
  (-2 < a ∧ a < 0 ∧ 2 / a ≤ x ∧ x ≤ -1) ∨ 
  (a = -2 ∧ x = -1) ∨
  (a < -2 ∧ -1 ≤ x ∧ x ≤ 2 / a) ↔ 
  a * x ^ 2 + (a - 2) * x - 2 ≥ 0 := 
sorry

end solve_inequality_l194_194217


namespace no_rain_either_day_l194_194934

noncomputable def P_A := 0.62
noncomputable def P_B := 0.54
noncomputable def P_A_and_B := 0.44
noncomputable def P_A_or_B := P_A + P_B - P_A_and_B -- Applying Inclusion-Exclusion principle.
noncomputable def P_A_and_B_complement := 1 - P_A_or_B -- Complement of P(A ∪ B).

theorem no_rain_either_day :
  P_A_and_B_complement = 0.28 :=
by
  unfold P_A_and_B_complement P_A_or_B
  unfold P_A P_B P_A_and_B
  simp
  sorry

end no_rain_either_day_l194_194934


namespace sin_45_eq_1_div_sqrt_2_l194_194338

theorem sin_45_eq_1_div_sqrt_2 : Real.sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_eq_1_div_sqrt_2_l194_194338


namespace solve_eq1_solve_eq2_l194_194475

theorem solve_eq1 (y : ℝ) : 6 - 3 * y = 15 + 6 * y ↔ y = -1 := by
  sorry

theorem solve_eq2 (x : ℝ) : (1 - 2 * x) / 3 = (3 * x + 1) / 7 - 2 ↔ x = 2 := by
  sorry

end solve_eq1_solve_eq2_l194_194475


namespace number_of_possible_scenarios_l194_194491

theorem number_of_possible_scenarios 
  (subjects : ℕ) 
  (students : ℕ) 
  (h_subjects : subjects = 4) 
  (h_students : students = 3) : 
  (subjects ^ students) = 64 := 
by
  -- Provide proof here
  sorry

end number_of_possible_scenarios_l194_194491


namespace saving_percentage_l194_194208

variable (S : ℝ) (saved_percent_last_year : ℝ) (made_more : ℝ) (saved_percent_this_year : ℝ)

-- Conditions from problem
def condition1 := saved_percent_last_year = 0.06
def condition2 := made_more = 1.20
def condition3 := saved_percent_this_year = 0.05 * made_more

-- The problem statement to prove
theorem saving_percentage (S : ℝ) (saved_percent_last_year : ℝ) (made_more : ℝ) (saved_percent_this_year : ℝ) :
  condition1 saved_percent_last_year →
  condition2 made_more →
  condition3 saved_percent_this_year made_more →
  (saved_percent_this_year * made_more = saved_percent_last_year * S * 1) :=
by 
  intros h1 h2 h3
  sorry

end saving_percentage_l194_194208


namespace total_percentage_increase_l194_194772

def initial_salary : Float := 60
def first_raise (s : Float) : Float := s + 0.10 * s
def second_raise (s : Float) : Float := s + 0.15 * s
def deduction (s : Float) : Float := s - 0.05 * s
def promotion_raise (s : Float) : Float := s + 0.20 * s
def final_salary (s : Float) : Float := promotion_raise (deduction (second_raise (first_raise s)))

theorem total_percentage_increase :
  final_salary initial_salary = initial_salary * 1.4421 :=
by
  sorry

end total_percentage_increase_l194_194772


namespace bc_together_l194_194165

theorem bc_together (A B C : ℕ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : C = 20) : B + C = 320 :=
by
  sorry

end bc_together_l194_194165


namespace incorrect_statement_A_l194_194248

-- Definitions for the conditions
def conditionA (x : ℝ) : Prop := -3 * x > 9
def conditionB (x : ℝ) : Prop := 2 * x - 1 < 0
def conditionC (x : ℤ) : Prop := x < 10
def conditionD (x : ℤ) : Prop := x < 2

-- Formal theorem statement
theorem incorrect_statement_A : ¬ (∀ x : ℝ, conditionA x ↔ x < -3) :=
by 
  sorry

end incorrect_statement_A_l194_194248


namespace power_evaluation_l194_194717

theorem power_evaluation : (-64 : ℝ)^(4/3) = 256 :=
by 
  have step1 : (-64 : ℝ)^(4/3) = ((-4 : ℝ)^3)^(4/3),
  { sorry },
  have step2 : ((-4 : ℝ)^3)^(4/3) = (-4 : ℝ)^4,
  { sorry },
  have step3 : (-4 : ℝ)^4 = 256,
  { sorry },
  rwa [step1, step2, step3]

end power_evaluation_l194_194717


namespace fraction_to_decimal_l194_194888

theorem fraction_to_decimal : (58 : ℚ) / 160 = 0.3625 := 
by sorry

end fraction_to_decimal_l194_194888


namespace number_of_real_pairs_l194_194967

theorem number_of_real_pairs :
  ∃! (x y : ℝ), 11 * x^2 + 2 * x * y + 9 * y^2 + 8 * x - 12 * y + 6 = 0 :=
sorry

end number_of_real_pairs_l194_194967


namespace solution_set_of_inequality_l194_194129

open Set

theorem solution_set_of_inequality :
  {x : ℝ | x^2 - x - 6 < 0} = Ioo (-2 : ℝ) 3 := 
sorry

end solution_set_of_inequality_l194_194129


namespace sin_45_degree_l194_194302

def Q : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), real.sin (real.pi / 4))
def E : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), 0)
def O : (x:ℝ) × (y:ℝ) := (0,0)
def OQ : ℝ := real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2)

theorem sin_45_degree : ∃ x: ℝ, x = real.sin (real.pi / 4) ∧ x = real.sqrt 2 / 2 :=
by sorry

end sin_45_degree_l194_194302


namespace forty_ab_l194_194141

theorem forty_ab (a b : ℝ) (h₁ : 4 * a = 30) (h₂ : 5 * b = 30) : 40 * a * b = 1800 :=
by
  sorry

end forty_ab_l194_194141


namespace guessing_probability_l194_194988

theorem guessing_probability :
  let P_correct : ℚ := 1 - (5/6) ^ 6
  P_correct = 31031 / 46656 :=
by sorry

end guessing_probability_l194_194988


namespace number_of_valid_3_digit_numbers_l194_194072

theorem number_of_valid_3_digit_numbers : 
  ∃ (n : ℕ), 
    (∀ (h t u : ℕ), 
      (n = h * 100 + t * 10 + u) ∧ 
      (1 ≤ h ∧ h ≤ 9) ∧ 
      (0 ≤ t ∧ t ≤ 9) ∧ 
      (0 ≤ u ∧ u ≤ 9) ∧ 
      (u ≥ 3 * t)) →
      n = 198 := 
by
  sorry

end number_of_valid_3_digit_numbers_l194_194072


namespace fraction_expression_l194_194682

theorem fraction_expression : (1 / 3) ^ 3 * (1 / 8) = 1 / 216 :=
by
  sorry

end fraction_expression_l194_194682


namespace find_value_of_A_l194_194811

theorem find_value_of_A (A ω φ c : ℝ)
  (a : ℕ+ → ℝ)
  (h_seq : ∀ n : ℕ+, a n * a (n + 1) * a (n + 2) = a n + a (n + 1) + a (n + 2))
  (h_neq : ∀ n : ℕ+, a n * a (n + 1) ≠ 1)
  (h_a1 : a 1 = 1)
  (h_a2 : a 2 = 2)
  (h_form : ∀ n : ℕ+, a n = A * Real.sin (ω * n + φ) + c)
  (h_ω_gt_0 : ω > 0)
  (h_phi_lt_pi_div_2 : |φ| < Real.pi / 2) :
  A = -2 * Real.sqrt 3 / 3 := 
sorry

end find_value_of_A_l194_194811


namespace tangent_line_at_P_eq_2x_l194_194226

noncomputable def tangentLineEq (f : ℝ → ℝ) (P : ℝ × ℝ) : ℝ → ℝ :=
  let slope := deriv f P.1
  fun x => slope * (x - P.1) + P.2

theorem tangent_line_at_P_eq_2x : 
  ∀ (f : ℝ → ℝ) (x y : ℝ),
    f x = x^2 + 1 → 
    (x = 1) → (y = 2) →
    tangentLineEq f (x, y) x = 2 * x :=
by
  intros f x y f_eq hx hy
  sorry

end tangent_line_at_P_eq_2x_l194_194226


namespace plane_second_trace_line_solutions_l194_194409

noncomputable def num_solutions_second_trace_line
  (first_trace_line : Line)
  (angle_with_projection_plane : ℝ)
  (intersection_outside_paper : Prop) : ℕ :=
2

theorem plane_second_trace_line_solutions
  (first_trace_line : Line)
  (angle_with_projection_plane : ℝ)
  (intersection_outside_paper : Prop) :
  num_solutions_second_trace_line first_trace_line angle_with_projection_plane intersection_outside_paper = 2 := by
sorry

end plane_second_trace_line_solutions_l194_194409


namespace nth_equation_l194_194469

theorem nth_equation (n : ℕ) : (2 * n + 2) ^ 2 - (2 * n) ^ 2 = 4 * (2 * n + 1) :=
by
  sorry

end nth_equation_l194_194469


namespace sin_45_eq_sqrt_two_over_two_l194_194366

theorem sin_45_eq_sqrt_two_over_two : Real.sin (π / 4) = sqrt 2 / 2 :=
by
  sorry

end sin_45_eq_sqrt_two_over_two_l194_194366


namespace sin_45_degree_l194_194310

noncomputable section

open Real

theorem sin_45_degree : sin (π / 4) = sqrt 2 / 2 := sorry

end sin_45_degree_l194_194310


namespace sin_45_deg_eq_l194_194372

noncomputable def sin_45_deg : ℝ :=
  real.sin (real.pi / 4)

theorem sin_45_deg_eq : sin_45_deg = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_deg_eq_l194_194372


namespace common_chord_length_l194_194653

theorem common_chord_length (r : ℝ) (h_r : r = 15) (h_overlap : 2 * r) :
    ∃ l : ℝ, l = 15 * Real.sqrt 3 :=
by 
  sorry

end common_chord_length_l194_194653


namespace train_cross_signal_in_18_sec_l194_194518

-- Definitions of the given conditions
def train_length := 300 -- meters
def platform_length := 350 -- meters
def time_cross_platform := 39 -- seconds

-- Speed of the train
def train_speed := (train_length + platform_length) / time_cross_platform -- meters/second

-- Time to cross the signal pole
def time_cross_signal_pole := train_length / train_speed -- seconds

theorem train_cross_signal_in_18_sec : time_cross_signal_pole = 18 := by sorry

end train_cross_signal_in_18_sec_l194_194518


namespace remainder_25197629_mod_4_l194_194757

theorem remainder_25197629_mod_4 : 25197629 % 4 = 1 := by
  sorry

end remainder_25197629_mod_4_l194_194757


namespace identical_digits_satisfy_l194_194488

theorem identical_digits_satisfy (n : ℕ) (hn : n ≥ 2) (x y z : ℕ) :
  (∃ (x y z : ℕ),
     (∃ (x y z : ℕ), 
         x = 3 ∧ y = 2 ∧ z = 1) ∨
     (∃ (x y z : ℕ), 
         x = 6 ∧ y = 8 ∧ z = 4) ∨
     (∃ (x y z : ℕ), 
         x = 8 ∧ y = 3 ∧ z = 7)) :=
by sorry

end identical_digits_satisfy_l194_194488


namespace basketball_probability_l194_194834

variable (A B : Event)
variable (P : Probability)
variable (P_A : P A = 0.8)
variable (P_B' : P (¬ B) = 0.1)
variable (ind : independent A B)

theorem basketball_probability :
  (P (A ∩ B) = 0.72) ∧ 
  (P (A ∩ (¬ B)) + P ((¬ A) ∩ B) = 0.26) :=
by
  sorry

end basketball_probability_l194_194834


namespace tan_add_pi_div_four_sine_cosine_ratio_l194_194181

-- Definition of the tangent function and trigonometric identities
variable {α : ℝ}

-- Given condition: tan(α) = 2
axiom tan_alpha_eq_2 : Real.tan α = 2

-- Problem 1: Prove that tan(α + π/4) = -3
theorem tan_add_pi_div_four : Real.tan ( α + Real.pi / 4 ) = -3 :=
by
  sorry

-- Problem 2: Prove that (6 * sin(α) + cos(α)) / (3 * sin(α) - cos(α)) = 13 / 5
theorem sine_cosine_ratio : 
  ( 6 * Real.sin α + Real.cos α ) / ( 3 * Real.sin α - Real.cos α ) = 13 / 5 :=
by
  sorry

end tan_add_pi_div_four_sine_cosine_ratio_l194_194181


namespace fraction_of_bikinis_or_trunks_l194_194168

theorem fraction_of_bikinis_or_trunks (h_bikinis : Real := 0.38) (h_trunks : Real := 0.25) :
  h_bikinis + h_trunks = 0.63 :=
by
  sorry

end fraction_of_bikinis_or_trunks_l194_194168


namespace evaluate_neg_sixtyfour_exp_four_thirds_l194_194707

theorem evaluate_neg_sixtyfour_exp_four_thirds : (-64 : ℝ) ^ (4/3 : ℝ) = 256 := 
by {
  have h1 : (-64 : ℝ) = (-4 : ℝ) ^ 3, by norm_num,
  have h2 : ((-4 : ℝ) ^ 3) ^ (4/3) = (-4 : ℝ) ^ (3 * (4/3)), by rw [←real.rpow_mul],
  rw [←h1],
  rw [h2],
  norm_num,
  rw [pow_nat_cast],
  norm_num,
  norm_num,
  rw [pow_nat_cast],
  norm_num
}

end evaluate_neg_sixtyfour_exp_four_thirds_l194_194707


namespace xy_zero_l194_194134

theorem xy_zero (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 162) : x * y = 0 :=
by
  sorry

end xy_zero_l194_194134


namespace time_for_a_alone_l194_194851

theorem time_for_a_alone
  (b_work_time : ℕ := 20)
  (c_work_time : ℕ := 45)
  (together_work_time : ℕ := 72 / 10) :
  ∃ (a_work_time : ℕ), a_work_time = 15 :=
by
  sorry

end time_for_a_alone_l194_194851


namespace distance_from_rachel_to_nicholas_l194_194636

def distance (speed time : ℝ) := speed * time

theorem distance_from_rachel_to_nicholas :
  distance 2 5 = 10 :=
by
  -- Proof goes here
  sorry

end distance_from_rachel_to_nicholas_l194_194636


namespace window_area_ratio_l194_194541

variables (AB AD : ℝ) (r : ℝ) (A_rectangle A_circle : ℝ)
          (h_AB : AB = 36)
          (h_ratio : AD / AB = 4 / 3)
          (h_r : r = AB / 2)
          (h_A_circle : A_circle = (Real.pi * (r^2)))
          (h_A_rectangle : A_rectangle = AD * AB)

theorem window_area_ratio : (A_rectangle / A_circle) = (16 / (3 * Real.pi)) :=
by
  sorry

end window_area_ratio_l194_194541


namespace contractor_absent_days_l194_194530

theorem contractor_absent_days (W A : ℕ) : 
  (W + A = 30 ∧ 25 * W - 7.5 * A = 425) → A = 10 :=
by
 sorry

end contractor_absent_days_l194_194530


namespace solve_star_eq_l194_194262

noncomputable def star (a b : ℤ) : ℤ := if a = b then 2 else sorry

axiom star_assoc : ∀ (a b c : ℤ), star a (star b c) = (star a b) - c
axiom star_self_eq_two : ∀ (a : ℤ), star a a = 2

theorem solve_star_eq : ∀ (x : ℤ), star 100 (star 5 x) = 20 → x = 20 :=
by
  intro x hx
  sorry

end solve_star_eq_l194_194262


namespace tan_angle_sum_l194_194104

noncomputable def tan_sum (θ : ℝ) : ℝ := Real.tan (θ + (Real.pi / 4))

theorem tan_angle_sum :
  let x := 1
  let y := 2
  let θ := Real.arctan (y / x)
  tan_sum θ = -3 := by
  sorry

end tan_angle_sum_l194_194104


namespace evaluate_neg_64_exp_4_over_3_l194_194713

theorem evaluate_neg_64_exp_4_over_3 : (-64 : ℝ) ^ (4 / 3) = 256 := 
by
  sorry

end evaluate_neg_64_exp_4_over_3_l194_194713


namespace complex_number_proof_l194_194210

open Complex

noncomputable def problem_complex (z : ℂ) (h1 : z ^ 7 = 1) (h2 : z ≠ 1) : ℂ :=
  (z - 1) * (z^2 - 1) * (z^3 - 1) * (z^4 - 1) * (z^5 - 1) * (z^6 - 1)

theorem complex_number_proof (z : ℂ) (h1 : z ^ 7 = 1) (h2 : z ≠ 1) :
  problem_complex z h1 h2 = 8 :=
  sorry

end complex_number_proof_l194_194210


namespace sin_45_degree_l194_194304

def Q : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), real.sin (real.pi / 4))
def E : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), 0)
def O : (x:ℝ) × (y:ℝ) := (0,0)
def OQ : ℝ := real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2)

theorem sin_45_degree : ∃ x: ℝ, x = real.sin (real.pi / 4) ∧ x = real.sqrt 2 / 2 :=
by sorry

end sin_45_degree_l194_194304


namespace scalene_triangle_height_ratio_l194_194087

theorem scalene_triangle_height_ratio {a b c : ℝ} (h1 : a > b ∧ b > c ∧ a > c)
  (h2 : a + c = 2 * b) : 
  1 / 3 < c / a ∧ c / a < 1 :=
by sorry

end scalene_triangle_height_ratio_l194_194087


namespace inequality_solution_l194_194723

theorem inequality_solution {x : ℝ} :
  {x | (2 * x - 8) * (x - 4) / x ≥ 0} = {x | x < 0} ∪ {x | x > 0} :=
by
  sorry

end inequality_solution_l194_194723


namespace probability_of_first_hearts_and_second_clubs_l194_194831

noncomputable def probability_first_hearts_second_clubs : ℚ :=
  let total_cards := 52
  let hearts_count := 13
  let clubs_count := 13
  let probability_first_hearts := hearts_count / total_cards
  let probability_second_clubs_given_first_hearts := clubs_count / (total_cards - 1)
  probability_first_hearts * probability_second_clubs_given_first_hearts

theorem probability_of_first_hearts_and_second_clubs :
  probability_first_hearts_second_clubs = 13 / 204 :=
by
  sorry

end probability_of_first_hearts_and_second_clubs_l194_194831


namespace hyperbola_line_intersections_l194_194421

-- Define the hyperbola and line equations
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 4
def line (x y k : ℝ) : Prop := y = k * (x - 1)

-- Conditions for intersecting the hyperbola at two points
def intersect_two_points (k : ℝ) : Prop := 
  k ∈ Set.Ioo (-2 * Real.sqrt 3 / 3) (-1) ∨ 
  k ∈ Set.Ioo (-1) 1 ∨ 
  k ∈ Set.Ioo 1 (2 * Real.sqrt 3 / 3)

-- Conditions for intersecting the hyperbola at exactly one point
def intersect_one_point (k : ℝ) : Prop := 
  k = 1 ∨ 
  k = -1 ∨ 
  k = 2 * Real.sqrt 3 / 3 ∨ 
  k = -2 * Real.sqrt 3 / 3

-- Proof that k is in the appropriate ranges
theorem hyperbola_line_intersections (k : ℝ) :
  ((∃ x y : ℝ, hyperbola x y ∧ line x y k) 
  → (∃ x₁ x₂ y₁ y₂ : ℝ, (x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧ hyperbola x₁ y₁ ∧ line x₁ y₁ k ∧ hyperbola x₂ y₂ ∧ line x₂ y₂ k) 
  → intersect_two_points k))
  ∧ ((∃ x y : ℝ, hyperbola x y ∧ line x y k) 
  → (∃ x y : ℝ, (hyperbola x y ∧ line x y k ∧ (∀ x' y', hyperbola x' y' ∧ line x' y' k → (x' ≠ x ∨ y' ≠ y) = false)) 
  → intersect_one_point k)) := 
sorry

end hyperbola_line_intersections_l194_194421


namespace probability_of_first_hearts_and_second_clubs_l194_194830

noncomputable def probability_first_hearts_second_clubs : ℚ :=
  let total_cards := 52
  let hearts_count := 13
  let clubs_count := 13
  let probability_first_hearts := hearts_count / total_cards
  let probability_second_clubs_given_first_hearts := clubs_count / (total_cards - 1)
  probability_first_hearts * probability_second_clubs_given_first_hearts

theorem probability_of_first_hearts_and_second_clubs :
  probability_first_hearts_second_clubs = 13 / 204 :=
by
  sorry

end probability_of_first_hearts_and_second_clubs_l194_194830


namespace evaluation_of_expression_l194_194435

theorem evaluation_of_expression
  (a b x y m : ℤ)
  (h1 : a + b = 0)
  (h2 : x * y = 1)
  (h3 : m = -1) :
  2023 * (a + b) + 3 * (|m|) - 2 * (x * y) = 1 :=
by
  -- skipping the proof
  sorry

end evaluation_of_expression_l194_194435


namespace number_of_sequences_l194_194405

theorem number_of_sequences :
  let letters := Finset.singleton 'P' ∪ Finset.singleton 'Q' ∪ Finset.singleton 'R' ∪ Finset.singleton 'S'
  let numbers := Finset.range 10
  (∑ l1 in letters, ∑ l2 in (letters \ {l1}), ∑ n1 in numbers, ∑ n2 in (numbers \ {n1}),
    if l1 = 'Q' ∧ l2 = 'Q' then 0 else if n1 = 0 ∧ n2 = 0 then 0 else 1) = 5832 := by
  sorry

end number_of_sequences_l194_194405


namespace range_of_a_min_value_reciprocals_l194_194418

noncomputable def f (x a : ℝ) : ℝ := |x - 2| + |x - a^2|

theorem range_of_a (a : ℝ) : (∃ x : ℝ, f x a ≤ a) ↔ 1 ≤ a ∧ a ≤ 2 := by
  sorry

theorem min_value_reciprocals (m n a : ℝ) (h : m + 2 * n = a) (ha : a = 2) : (1/m + 1/n) ≥ (3/2 + Real.sqrt 2) := by
  sorry

end range_of_a_min_value_reciprocals_l194_194418


namespace circles_externally_tangent_l194_194686

theorem circles_externally_tangent :
  let C1x := -3
  let C1y := 2
  let r1 := 2
  let C2x := 3
  let C2y := -6
  let r2 := 8
  let d := Real.sqrt ((C2x - C1x)^2 + (C2y - C1y)^2)
  (d = r1 + r2) → 
  ((x + 3)^2 + (y - 2)^2 = 4) → ((x - 3)^2 + (y + 6)^2 = 64) → 
  ∃ (P : ℝ × ℝ), (P.1 + 3)^2 + (P.2 - 2)^2 = 4 ∧ (P.1 - 3)^2 + (P.2 + 6)^2 = 64 :=
by
  intros
  sorry

end circles_externally_tangent_l194_194686


namespace car_total_distance_l194_194151

noncomputable def distance_first_segment (speed1 : ℝ) (time1 : ℝ) : ℝ :=
  speed1 * time1

noncomputable def distance_second_segment (speed2 : ℝ) (time2 : ℝ) : ℝ :=
  speed2 * time2

noncomputable def distance_final_segment (speed3 : ℝ) (time3 : ℝ) : ℝ :=
  speed3 * time3

noncomputable def total_distance (d1 d2 d3 : ℝ) : ℝ :=
  d1 + d2 + d3

theorem car_total_distance :
  let d1 := distance_first_segment 65 2
  let d2 := distance_second_segment 80 1.5
  let d3 := distance_final_segment 50 2
  total_distance d1 d2 d3 = 350 :=
by
  sorry

end car_total_distance_l194_194151


namespace polynomial_multiplication_l194_194555

theorem polynomial_multiplication (x y : ℝ) : 
  (2 * x - 3 * y + 1) * (2 * x + 3 * y - 1) = 4 * x^2 - 9 * y^2 + 6 * y - 1 := by
  sorry

end polynomial_multiplication_l194_194555


namespace find_part_of_number_l194_194594

theorem find_part_of_number (x y : ℕ) (h₁ : x = 1925) (h₂ : x / 7 = y + 100) : y = 175 :=
sorry

end find_part_of_number_l194_194594


namespace amy_remaining_money_l194_194677

-- Definitions based on conditions
def initial_money : ℕ := 100
def doll_cost : ℕ := 1
def number_of_dolls : ℕ := 3

-- The theorem we want to prove
theorem amy_remaining_money : initial_money - number_of_dolls * doll_cost = 97 :=
by 
  sorry

end amy_remaining_money_l194_194677


namespace ratio_of_Steve_speeds_l194_194646

noncomputable def Steve_speeds_ratio : Nat := 
  let d := 40 -- distance in km
  let T := 6  -- total time in hours
  let v2 := 20 -- speed on the way back in km/h
  let t2 := d / v2 -- time taken on the way back in hours
  let t1 := T - t2 -- time taken on the way to work in hours
  let v1 := d / t1 -- speed on the way to work in km/h
  v2 / v1

theorem ratio_of_Steve_speeds :
  Steve_speeds_ratio = 2 := 
  by sorry

end ratio_of_Steve_speeds_l194_194646


namespace union_of_sets_l194_194233

def setA : Set ℝ := { x : ℝ | (x - 2) / (x + 1) ≤ 0 }
def setB : Set ℝ := { x : ℝ | -2 * x^2 + 7 * x + 4 > 0 }
def unionAB : Set ℝ := { x : ℝ | -1 < x ∧ x < 4 }

theorem union_of_sets :
  ∀ x : ℝ, x ∈ setA ∨ x ∈ setB ↔ x ∈ unionAB :=
by sorry

end union_of_sets_l194_194233


namespace find_annual_interest_rate_l194_194288

/-- 
  Given:
  - Principal P = 10000
  - Interest I = 450
  - Time period T = 0.75 years

  Prove that the annual interest rate is 0.08.
-/
theorem find_annual_interest_rate (P I : ℝ) (T : ℝ) (hP : P = 10000) (hI : I = 450) (hT : T = 0.75) : 
  (I / (P * T) / T) = 0.08 :=
by
  sorry

end find_annual_interest_rate_l194_194288


namespace initial_games_l194_194094

def games_given_away : ℕ := 91
def games_left : ℕ := 92

theorem initial_games :
  games_given_away + games_left = 183 :=
by
  sorry

end initial_games_l194_194094


namespace anne_more_drawings_l194_194550

/-- Anne's markers problem setup. -/
structure MarkerProblem :=
  (markers : ℕ)
  (drawings_per_marker : ℚ)
  (drawings_made : ℕ)

-- Given conditions
def anne_conditions : MarkerProblem :=
  { markers := 12, drawings_per_marker := 1.5, drawings_made := 8 }

-- Equivalent proof problem statement in Lean
theorem anne_more_drawings(conditions : MarkerProblem) : 
  conditions.markers * conditions.drawings_per_marker - conditions.drawings_made = 10 :=
by
  -- The proof of this theorem is omitted
  sorry

end anne_more_drawings_l194_194550


namespace eval_expression_correct_l194_194697

noncomputable def eval_expression : ℝ := (-64)^(4/3)

theorem eval_expression_correct : eval_expression = 256 := by
  sorry

end eval_expression_correct_l194_194697


namespace actual_distance_between_towns_l194_194122

def map_distance := 20 -- distance between towns on the map in inches
def scale := 10 -- scale: 1 inch = 10 miles

theorem actual_distance_between_towns : map_distance * scale = 200 := by
  sorry

end actual_distance_between_towns_l194_194122


namespace at_least_one_not_less_than_two_l194_194061

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) :=
sorry

end at_least_one_not_less_than_two_l194_194061


namespace yola_past_weight_l194_194839

variable (W Y Y_past : ℕ)

-- Conditions
def condition1 : Prop := W = Y + 30
def condition2 : Prop := W = Y_past + 80
def condition3 : Prop := Y = 220

-- Theorem statement
theorem yola_past_weight : condition1 W Y → condition2 W Y_past → condition3 Y → Y_past = 170 :=
by
  intros h_condition1 h_condition2 h_condition3
  -- Placeholder for the proof, not required in the solution
  sorry

end yola_past_weight_l194_194839


namespace minimum_value_expr_C_l194_194547

theorem minimum_value_expr_C : 
  ∃ (x : ℝ), (∀ (z : ℝ), z = real.exp x + 4 * real.exp (-x) → z ≥ 4) := 
sorry

end minimum_value_expr_C_l194_194547


namespace nancy_carrots_l194_194621

theorem nancy_carrots (picked_day_1 threw_out total_left total_final picked_next_day : ℕ)
  (h1 : picked_day_1 = 12)
  (h2 : threw_out = 2)
  (h3 : total_final = 31)
  (h4 : total_left = picked_day_1 - threw_out)
  (h5 : total_final = total_left + picked_next_day) :
  picked_next_day = 21 :=
by
  sorry

end nancy_carrots_l194_194621


namespace sin_45_deg_eq_l194_194374

noncomputable def sin_45_deg : ℝ :=
  real.sin (real.pi / 4)

theorem sin_45_deg_eq : sin_45_deg = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_deg_eq_l194_194374


namespace shopkeeper_total_cards_l194_194024

-- Conditions
def num_standard_decks := 3
def cards_per_standard_deck := 52
def num_tarot_decks := 2
def cards_per_tarot_deck := 72
def num_trading_sets := 5
def cards_per_trading_set := 100
def additional_random_cards := 27

-- Calculate total cards
def total_standard_cards := num_standard_decks * cards_per_standard_deck
def total_tarot_cards := num_tarot_decks * cards_per_tarot_deck
def total_trading_cards := num_trading_sets * cards_per_trading_set
def total_cards := total_standard_cards + total_tarot_cards + total_trading_cards + additional_random_cards

-- Proof statement
theorem shopkeeper_total_cards : total_cards = 827 := by
    sorry

end shopkeeper_total_cards_l194_194024


namespace miles_per_dollar_l194_194684

def car_mpg : ℝ := 32
def gas_cost_per_gallon : ℝ := 4

theorem miles_per_dollar (X : ℝ) : 
  (X / gas_cost_per_gallon) * car_mpg = 8 * X :=
by
  sorry

end miles_per_dollar_l194_194684


namespace find_F_l194_194080

-- Define the condition and the equation
def C (F : ℤ) : ℤ := (5 * (F - 30)) / 9

-- Define the assumption that C = 25
def C_condition : ℤ := 25

-- The theorem to prove that F = 75 given the conditions
theorem find_F (F : ℤ) (h : C F = C_condition) : F = 75 :=
sorry

end find_F_l194_194080


namespace expression_evaluation_l194_194556

theorem expression_evaluation : 2^2 - Real.tan (Real.pi / 3) + abs (Real.sqrt 3 - 1) - (3 - Real.pi)^0 = 2 :=
by
  sorry

end expression_evaluation_l194_194556


namespace simplified_expression_l194_194639

theorem simplified_expression (x : ℝ) : 
  x * (3 * x^2 - 2) - 5 * (x^2 - 2 * x + 7) = 3 * x^3 - 5 * x^2 + 8 * x - 35 := 
by
  sorry

end simplified_expression_l194_194639


namespace eval_expression_correct_l194_194699

noncomputable def eval_expression : ℝ := (-64)^(4/3)

theorem eval_expression_correct : eval_expression = 256 := by
  sorry

end eval_expression_correct_l194_194699


namespace three_digit_numbers_with_properties_l194_194966

noncomputable def valid_numbers_with_properties : List Nat :=
  [179, 239, 299, 359, 419, 479, 539, 599, 659, 719, 779, 839, 899, 959]

theorem three_digit_numbers_with_properties (N : ℕ) :
  N >= 100 ∧ N < 1000 ∧ 
  N ≡ 1 [MOD 2] ∧
  N ≡ 2 [MOD 3] ∧
  N ≡ 3 [MOD 4] ∧
  N ≡ 4 [MOD 5] ∧
  N ≡ 5 [MOD 6] ↔ N ∈ valid_numbers_with_properties :=
by
  sorry

end three_digit_numbers_with_properties_l194_194966


namespace sin_45_deg_eq_l194_194375

noncomputable def sin_45_deg : ℝ :=
  real.sin (real.pi / 4)

theorem sin_45_deg_eq : sin_45_deg = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_deg_eq_l194_194375


namespace area_of_region_enclosed_by_parabolas_l194_194291

-- Define the given parabolas
def parabola1 (y : ℝ) : ℝ := -3 * y^2
def parabola2 (y : ℝ) : ℝ := 1 - 4 * y^2

-- Define the integral representing the area between the parabolas
noncomputable def areaBetweenParabolas : ℝ :=
  2 * (∫ y in (0 : ℝ)..1, (parabola2 y - parabola1 y))

-- The statement to be proved
theorem area_of_region_enclosed_by_parabolas :
  areaBetweenParabolas = 4 / 3 := 
sorry

end area_of_region_enclosed_by_parabolas_l194_194291


namespace probability_sum_of_three_dice_is_9_l194_194764

def sum_of_three_dice_is_9 : Prop :=
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧ a + b + c = 9)

theorem probability_sum_of_three_dice_is_9 : 
  (∀ (a b c : ℕ), 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 → a + b + c = 9 → sum_of_three_dice_is_9) ∧ 
  (1 / 216 = 25 / 216) := 
by
  sorry

end probability_sum_of_three_dice_is_9_l194_194764


namespace sin_45_degree_eq_sqrt2_div_2_l194_194350

theorem sin_45_degree_eq_sqrt2_div_2 :
  let θ := (real.pi / 4)
  in sin θ = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_degree_eq_sqrt2_div_2_l194_194350


namespace find_a_l194_194419

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 5 * x^2 + a * x

theorem find_a (a : ℝ) : 
  (∃ x : ℝ, x = -3 ∧ ∀ x' : ℝ, f (f'':=f' (a:=a) : (x : ℝ) -> x^3 + 5x^2 + ax
  := 3x^2 + 10x + a) x' = 0 -> x = -3) → a = 3 :=
by
  sorry

end find_a_l194_194419


namespace unique_solution_system_l194_194144

noncomputable def f (x : ℝ) := 4 * x ^ 3 + x - 4

theorem unique_solution_system :
  (∃ x y z : ℝ, y^2 = 4*x^3 + x - 4 ∧ z^2 = 4*y^3 + y - 4 ∧ x^2 = 4*z^3 + z - 4) ↔
  (x = 1 ∧ y = 1 ∧ z = 1) :=
by
  sorry

end unique_solution_system_l194_194144


namespace find_m_eq_5_l194_194482

-- Definitions for the problem conditions
def f (x m : ℝ) := 2 * x + m

theorem find_m_eq_5 (m : ℝ) (a b : ℝ) :
  (a = f 0 m) ∧ (b = f m m) ∧ ((b - a) = (m - 0 + 5)) → m = 5 :=
by
  sorry

end find_m_eq_5_l194_194482


namespace digit_contains_zero_l194_194028

theorem digit_contains_zero 
  (n₁ n₂ : ℕ) 
  (h₁ : 10000 ≤ n₁ ∧ n₁ ≤ 99999)
  (h₂ : 10000 ≤ n₂ ∧ n₂ ≤ 99999)
  (h₃ : ∃ (i j : ℕ), i ≠ j ∧ n₂ = n₁.swap_digits i j)
  (h₄ : n₁ + n₂ = 111111) 
  : ∃ (d : ℕ), d = 0 :=
sorry

end digit_contains_zero_l194_194028


namespace fraction_is_terminating_decimal_l194_194890

noncomputable def fraction_to_decimal : ℚ :=
  58 / 160

theorem fraction_is_terminating_decimal : fraction_to_decimal = 3625 / 10000 :=
by
  sorry

end fraction_is_terminating_decimal_l194_194890


namespace molecular_weight_of_acid_l194_194243

theorem molecular_weight_of_acid (molecular_weight : ℕ) (n : ℕ) (h : molecular_weight = 792) (hn : n = 9) :
  molecular_weight = 792 :=
by 
  sorry

end molecular_weight_of_acid_l194_194243


namespace boards_nailing_l194_194109

variables {x y a b : ℕ} 

theorem boards_nailing :
  (2 * x + 3 * y = 87) ∧
  (3 * a + 5 * b = 94) →
  (x + y = 30) ∧ (a + b = 30) :=
by
  sorry

end boards_nailing_l194_194109


namespace find_a_and_theta_find_sin_alpha_plus_pi_over_3_l194_194413

noncomputable def f (a θ x : ℝ) : ℝ :=
  (a + 2 * Real.cos x ^ 2) * Real.cos (2 * x + θ)

theorem find_a_and_theta (a θ : ℝ) (h1 : f a θ (Real.pi / 4) = 0)
  (h2 : ∀ x, f a θ (-x) = -f a θ x) :
  a = -1 ∧ θ = Real.pi / 2 :=
sorry

theorem find_sin_alpha_plus_pi_over_3 (α θ : ℝ) (h1 : α ∈ Set.Ioo (Real.pi / 2) Real.pi)
  (h2 : f (-1) (Real.pi / 2) (α / 4) = -2 / 5) :
  Real.sin (α + Real.pi / 3) = (4 - 3 * Real.sqrt 3) / 10 :=
sorry

end find_a_and_theta_find_sin_alpha_plus_pi_over_3_l194_194413


namespace blue_candies_count_l194_194146

theorem blue_candies_count (total_pieces red_pieces : Nat) (h1 : total_pieces = 3409) (h2 : red_pieces = 145) : total_pieces - red_pieces = 3264 := 
by
  -- Proof will be provided here
  sorry

end blue_candies_count_l194_194146


namespace eval_expression_correct_l194_194701

noncomputable def eval_expression : ℝ := (-64)^(4/3)

theorem eval_expression_correct : eval_expression = 256 := by
  sorry

end eval_expression_correct_l194_194701


namespace proof_goal_l194_194462

noncomputable def exp_value (k m n : ℕ) : ℤ :=
  (6^k - k^6 + 2^m - 4^m + n^3 - 3^n : ℤ)

theorem proof_goal (k m n : ℕ) (h_k : 18^k ∣ 624938) (h_m : 24^m ∣ 819304) (h_n : n = 2 * k + m) :
  exp_value k m n = 0 := by
  sorry

end proof_goal_l194_194462


namespace problem_solution_l194_194097

theorem problem_solution
  (a b c : ℝ)
  (habc_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_condition : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a^2 / (b - c)^2 + b^2 / (c - a)^2 + c^2 / (a - b)^2 = 0 :=
by
  sorry

end problem_solution_l194_194097


namespace johns_speed_l194_194941

theorem johns_speed :
  ∀ (v : ℝ), 
    (∀ (t : ℝ), 24 = 30 * (t + 4 / 60) → 24 = v * (t - 8 / 60)) → 
    v = 40 :=
by
  intros
  sorry

end johns_speed_l194_194941


namespace cost_price_per_meter_l194_194250

-- Definitions
def selling_price : ℝ := 9890
def meters_sold : ℕ := 92
def profit_per_meter : ℝ := 24

-- Theorem
theorem cost_price_per_meter : (selling_price - profit_per_meter * meters_sold) / meters_sold = 83.5 :=
by
  sorry

end cost_price_per_meter_l194_194250


namespace sin_45_eq_sqrt2_div_2_l194_194361

theorem sin_45_eq_sqrt2_div_2 :
  Real.sin (π / 4) = Real.sqrt 2 / 2 := 
by
  sorry

end sin_45_eq_sqrt2_div_2_l194_194361


namespace sin_45_degree_l194_194308

noncomputable section

open Real

theorem sin_45_degree : sin (π / 4) = sqrt 2 / 2 := sorry

end sin_45_degree_l194_194308


namespace f_leq_2x_l194_194484

noncomputable def f : ℝ → ℝ := sorry
axiom f_nonneg {x : ℝ} (hx : 0 ≤ x ∧ x ≤ 1) : 0 ≤ f x
axiom f_one : f 1 = 1
axiom f_superadditive {x y : ℝ} (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hxy : x + y ≤ 1) : f (x + y) ≥ f x + f y

-- The theorem statement to be proved
theorem f_leq_2x {x : ℝ} (hx : 0 ≤ x ∧ x ≤ 1) : f x ≤ 2 * x := sorry

end f_leq_2x_l194_194484


namespace smallest_10_digit_number_with_sum_81_l194_194569

def sum_of_digits (n : Nat) : Nat :=
  n.digits 10 |>.sum

theorem smallest_10_digit_number_with_sum_81 {n : Nat} :
  n ≥ 1000000000 ∧ n < 10000000000 ∧ sum_of_digits n ≥ 81 → 
  n = 1899999999 :=
sorry

end smallest_10_digit_number_with_sum_81_l194_194569


namespace mildred_total_oranges_l194_194953

-- Conditions
def initial_oranges : ℕ := 77
def additional_oranges : ℕ := 2

-- Question/Goal
theorem mildred_total_oranges : initial_oranges + additional_oranges = 79 := by
  sorry

end mildred_total_oranges_l194_194953


namespace james_toys_l194_194202

-- Define the conditions and the problem statement
theorem james_toys (x : ℕ) (h1 : ∀ x, 2 * x = 60 - x) : x = 20 :=
sorry

end james_toys_l194_194202


namespace train_passes_tree_in_28_seconds_l194_194163

def km_per_hour_to_meter_per_second (km_per_hour : ℕ) : ℕ :=
  km_per_hour * 1000 / 3600

def pass_tree_time (length : ℕ) (speed_kmh : ℕ) : ℕ :=
  length / (km_per_hour_to_meter_per_second speed_kmh)

theorem train_passes_tree_in_28_seconds :
  pass_tree_time 490 63 = 28 :=
by
  sorry

end train_passes_tree_in_28_seconds_l194_194163


namespace triangle_inequality_l194_194732

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  (a^2 + 2 * b * c) / (b^2 + c^2) + (b^2 + 2 * a * c) / (c^2 + a^2) + (c^2 + 2 * a * b) / (a^2 + b^2) > 3 :=
by
  sorry

end triangle_inequality_l194_194732


namespace find_solution_l194_194898

theorem find_solution (x y : ℕ) (h1 : y ∣ (x^2 + 1)) (h2 : x^2 ∣ (y^3 + 1)) : (x = 1 ∧ y = 1) :=
sorry

end find_solution_l194_194898


namespace basketball_probability_l194_194833

variable (A B : Event)
variable (P : Probability)
variable (P_A : P A = 0.8)
variable (P_B' : P (¬ B) = 0.1)
variable (ind : independent A B)

theorem basketball_probability :
  (P (A ∩ B) = 0.72) ∧ 
  (P (A ∩ (¬ B)) + P ((¬ A) ∩ B) = 0.26) :=
by
  sorry

end basketball_probability_l194_194833


namespace local_min_at_neg_one_l194_194580

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem local_min_at_neg_one : 
  ∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f x ≥ f (-1) := by
  sorry

end local_min_at_neg_one_l194_194580


namespace tire_price_l194_194805

theorem tire_price (x : ℝ) (h1 : 2 * x + 5 = 185) : x = 90 :=
by
  sorry

end tire_price_l194_194805


namespace product_of_integers_whose_cubes_sum_to_189_l194_194663

theorem product_of_integers_whose_cubes_sum_to_189 :
  ∃ (a b : ℤ), a^3 + b^3 = 189 ∧ a * b = 20 :=
by
  sorry

end product_of_integers_whose_cubes_sum_to_189_l194_194663


namespace find_cheese_calories_l194_194610

noncomputable def lettuce_calories := 50
noncomputable def carrots_calories := 2 * lettuce_calories
noncomputable def dressing_calories := 210

noncomputable def crust_calories := 600
noncomputable def pepperoni_calories := crust_calories / 3

noncomputable def total_salad_calories := lettuce_calories + carrots_calories + dressing_calories
noncomputable def total_pizza_calories (cheese_calories : ℕ) := crust_calories + pepperoni_calories + cheese_calories

theorem find_cheese_calories (consumed_calories : ℕ) (cheese_calories : ℕ) :
  consumed_calories = 330 →
  1/4 * total_salad_calories + 1/5 * total_pizza_calories cheese_calories = consumed_calories →
  cheese_calories = 400 := by
  sorry

end find_cheese_calories_l194_194610


namespace find_integer_solutions_xy_l194_194388

theorem find_integer_solutions_xy :
  ∀ (x y : ℕ), (x * y = x + y + 3) → (x, y) = (2, 5) ∨ (x, y) = (5, 2) ∨ (x, y) = (3, 3) := by
  intros x y h
  sorry

end find_integer_solutions_xy_l194_194388


namespace number_of_ways_correct_l194_194681

-- Definition used for the problem
def number_of_ways : Nat :=
  -- sorry is used here to ignore the function body, since we focus on statement
  sorry

-- Statement to be proved
theorem number_of_ways_correct : 
  number_of_ways = 114 := sorry

end number_of_ways_correct_l194_194681


namespace students_left_l194_194602

theorem students_left (initial_students new_students final_students students_left : ℕ)
  (h1 : initial_students = 10)
  (h2 : new_students = 42)
  (h3 : final_students = 48)
  : initial_students + new_students - students_left = final_students → students_left = 4 :=
by
  intros
  sorry

end students_left_l194_194602


namespace number_of_football_players_l194_194766

theorem number_of_football_players
  (cricket_players : ℕ)
  (hockey_players : ℕ)
  (softball_players : ℕ)
  (total_players : ℕ) :
  cricket_players = 22 →
  hockey_players = 15 →
  softball_players = 19 →
  total_players = 77 →
  total_players - (cricket_players + hockey_players + softball_players) = 21 :=
by
  intros h1 h2 h3 h4
  sorry

end number_of_football_players_l194_194766


namespace parabola_equation_conditions_l194_194730

def focus_on_x_axis (focus : ℝ × ℝ) := (∃ x : ℝ, focus = (x, 0))
def foot_of_perpendicular (line : ℝ × ℝ → Prop) (focus : ℝ × ℝ) :=
  (∃ point : ℝ × ℝ, point = (2, 1) ∧ line focus ∧ line point ∧ line (0, 0))

theorem parabola_equation_conditions (focus : ℝ × ℝ) (line : ℝ × ℝ → Prop) :
  focus_on_x_axis focus →
  foot_of_perpendicular line focus →
  ∃ a : ℝ, ∀ x y : ℝ, y^2 = a * x ↔ y^2 = 10 * x :=
by
  intros h1 h2
  use 10
  sorry

end parabola_equation_conditions_l194_194730


namespace option_b_is_correct_l194_194676

theorem option_b_is_correct :
  ∃ (f : ℝ → ℝ), (∀ x : ℝ, f x = - 1 / x)
  ∧ (∀ x : ℝ, f (-x) = - f (x))
  ∧ (∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y)
:=
begin
  use (λ x : ℝ, - 1 / x),
  split,
  { intro x,
    refl, },
  split,
  { intros x,
    simp, },
  { intros x y hx hy hxy,
    simp [hx, hy, hxy], },
end

end option_b_is_correct_l194_194676


namespace find_direction_vector_l194_194125

noncomputable def projection_matrix : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![2/15, -1/15, -1/3],
    ![-1/15, 1/30, 1/6],
    ![-1/3, 1/6, 5/6]]

def valid_direction_vector (v : Vector ℤ 3) : Prop :=
  v.head > 0 ∧ (v.to_list.map Int.natAbs).gcd = 1

theorem find_direction_vector :
  ∃ (v : Vector ℤ 3), projection_matrix.mulVec ![1, 0, 0] = (1 / 15 : ℚ) • v ∧ valid_direction_vector v :=
begin
  let v := Vector.of [2, -1, -5],
  use v,
  simp only [Matrix.mulVec, Matrix.dotProduct, Pi.smul_apply, Matrix.smulVec, Matrix.cons_val', Matrix.head_cons, Matrix.fin_zero_eq_zero, Matrix.row_vec_lin_equiv_apply],
  split,
  { simp, },
  { split,
    { norm_num, },
    { simp [valid_direction_vector],
      norm_num, } },
end

end find_direction_vector_l194_194125


namespace value_of_expression_l194_194436

noncomputable def largestNegativeInteger : Int := -1

theorem value_of_expression (a b x y : ℝ) (m : Int)
  (h1 : a + b = 0)
  (h2 : x * y = 1)
  (h3 : m = largestNegativeInteger) :
  2023 * (a + b) + 3 * |m| - 2 * (x * y) = 1 :=
by
  sorry

end value_of_expression_l194_194436


namespace range_of_b_l194_194950

noncomputable def f (a x : ℝ) : ℝ := 
  Real.log x - (1 / 2) * a * x^2 - 2 * x

theorem range_of_b (a : ℝ) (b : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x ≤ 1 → a ∈ Set.Ico (-1 : ℝ) (0 : ℝ) → f a x < b) ↔ b > -3 / 2 :=
by
  sorry

end range_of_b_l194_194950


namespace album_cost_l194_194290

-- Definitions for given conditions
def M (X : ℕ) : ℕ := X - 2
def K (X : ℕ) : ℕ := X - 34
def F (X : ℕ) : ℕ := X - 35

-- We need to prove that X = 35
theorem album_cost : ∃ X : ℕ, (M X) + (K X) + (F X) < X ∧ X = 35 :=
by
  sorry -- Proof not required.

end album_cost_l194_194290


namespace hexagon_side_squares_sum_l194_194679

variables {P Q R P' Q' R' A B C D E F : Type}
variables (a1 a2 a3 b1 b2 b3 : ℝ)
variables (h_eq_triangles : congruent (triangle P Q R) (triangle P' Q' R'))
variables (h_sides : 
  AB = a1 ∧ BC = b1 ∧ CD = a2 ∧ 
  DE = b2 ∧ EF = a3 ∧ FA = b3)
  
theorem hexagon_side_squares_sum :
  a1^2 + a2^2 + a3^2 = b1^2 + b2^2 + b3^2 :=
sorry

end hexagon_side_squares_sum_l194_194679


namespace paint_gallons_needed_l194_194133

theorem paint_gallons_needed (n : ℕ) (h : n = 16) (h_col_height : ℝ) (h_col_height_val : h_col_height = 24)
  (h_col_diameter : ℝ) (h_col_diameter_val : h_col_diameter = 8) (cover_area : ℝ) 
  (cover_area_val : cover_area = 350) : 
  ∃ (gallons : ℤ), gallons = 33 := 
by
  sorry

end paint_gallons_needed_l194_194133


namespace value_of_expression_l194_194923

theorem value_of_expression (a b : ℝ) (h : a + b = 4) : a^2 + 2 * a * b + b^2 = 16 := by
  sorry

end value_of_expression_l194_194923


namespace total_students_in_class_l194_194447

theorem total_students_in_class :
  ∃ x, (10 * 90 + 15 * 80 + x * 60) / (10 + 15 + x) = 72 → 10 + 15 + x = 50 :=
by
  -- Providing an existence proof and required conditions
  use 25
  intro h
  sorry

end total_students_in_class_l194_194447


namespace total_number_of_fish_l194_194553

noncomputable def number_of_stingrays : ℕ := 28

noncomputable def number_of_sharks : ℕ := 2 * number_of_stingrays

theorem total_number_of_fish : number_of_sharks + number_of_stingrays = 84 :=
by
  sorry

end total_number_of_fish_l194_194553


namespace people_got_off_at_second_stop_l194_194025

theorem people_got_off_at_second_stop (x : ℕ) :
  (10 - x) + 20 - 18 + 2 = 12 → x = 2 :=
  by sorry

end people_got_off_at_second_stop_l194_194025


namespace prob_same_color_is_correct_l194_194079

-- Define the sides of one die
def blue_sides := 6
def yellow_sides := 8
def green_sides := 10
def purple_sides := 6
def total_sides := 30

-- Define the probability each die shows a specific color
def prob_blue := blue_sides / total_sides
def prob_yellow := yellow_sides / total_sides
def prob_green := green_sides / total_sides
def prob_purple := purple_sides / total_sides

-- The probability that both dice show the same color
def prob_same_color :=
  (prob_blue * prob_blue) + 
  (prob_yellow * prob_yellow) + 
  (prob_green * prob_green) + 
  (prob_purple * prob_purple)

-- We should prove that the computed probability is equal to the given answer
theorem prob_same_color_is_correct :
  prob_same_color = 59 / 225 := 
sorry

end prob_same_color_is_correct_l194_194079


namespace desired_depth_is_50_l194_194999

noncomputable def desired_depth_dig (d days : ℝ) : ℝ :=
  let initial_man_hours := 45 * 8 * d
  let additional_man_hours := 100 * 6 * d
  (initial_man_hours / additional_man_hours) * 30

theorem desired_depth_is_50 (d : ℝ) : desired_depth_dig d = 50 :=
  sorry

end desired_depth_is_50_l194_194999


namespace plate_and_rollers_acceleration_l194_194001

-- Definitions for conditions
def roller_radii := (1 : ℝ, 0.4 : ℝ)
def plate_mass := 150 -- kg
def inclination_angle := Real.arccos 0.68
def gravity_acceleration := 10 -- m/s^2

-- Theorem statement for the problem
theorem plate_and_rollers_acceleration :
  let R := roller_radii.1,
      r := roller_radii.2,
      m := plate_mass,
      α := inclination_angle,
      g := gravity_acceleration in
  ∃ (a_plate a_rollers : ℝ), a_plate = a_rollers ∧ a_plate = 4 :=
  sorry

end plate_and_rollers_acceleration_l194_194001


namespace terry_current_age_l194_194930

theorem terry_current_age (T : ℕ) (nora_current_age : ℕ) (h1 : nora_current_age = 10)
  (h2 : T + 10 = 4 * nora_current_age) : T = 30 :=
by
  sorry

end terry_current_age_l194_194930


namespace sin_45_eq_sqrt2_div_2_l194_194321

theorem sin_45_eq_sqrt2_div_2 : Real.sin (π / 4) = Real.sqrt 2 / 2 := 
sorry

end sin_45_eq_sqrt2_div_2_l194_194321


namespace cost_of_pencils_and_pens_l194_194223

theorem cost_of_pencils_and_pens (p q : ℝ) 
  (h₁ : 3 * p + 2 * q = 3.60) 
  (h₂ : 2 * p + 3 * q = 3.15) : 
  3 * p + 3 * q = 4.05 :=
sorry

end cost_of_pencils_and_pens_l194_194223


namespace kevin_leap_day_2024_is_monday_l194_194614

def days_between_leap_birthdays (years: ℕ) (leap_year_count: ℕ) : ℕ :=
  (years - leap_year_count) * 365 + leap_year_count * 366

def day_of_week_after_days (start_day: ℕ) (days: ℕ) : ℕ :=
  (start_day + days) % 7

noncomputable def kevin_leap_day_weekday_2024 : ℕ :=
  let days := days_between_leap_birthdays 24 6
  let start_day := 2 -- Tuesday as 2 (assuming 0 = Sunday, 1 = Monday,..., 6 = Saturday)
  day_of_week_after_days start_day days

theorem kevin_leap_day_2024_is_monday :
  kevin_leap_day_weekday_2024 = 1 -- 1 represents Monday
  :=
by
  sorry

end kevin_leap_day_2024_is_monday_l194_194614


namespace collinear_vectors_l194_194781

theorem collinear_vectors :
  ∃ x : ℝ, (collinear ({ x, 1 }, { 4, x }))
    ∧ ∀ y : ℝ, collinear ({ y, 1 }, { 4, y }) → (y = 2 ∨ y = -2) :=
by
  sorry

end collinear_vectors_l194_194781


namespace inequality_for_positive_reals_l194_194187

theorem inequality_for_positive_reals (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) (k : ℕ) (h_k : 2 ≤ k) :
  (a^k / (a + b) + b^k / (b + c) + c^k / (c + a) ≥ 3 / 2) :=
by
  intros
  sorry

end inequality_for_positive_reals_l194_194187


namespace initial_hamburgers_count_is_nine_l194_194157

-- Define the conditions
def hamburgers_initial (total_hamburgers : ℕ) (additional_hamburgers : ℕ) : ℕ :=
  total_hamburgers - additional_hamburgers

-- The statement to be proved
theorem initial_hamburgers_count_is_nine :
  hamburgers_initial 12 3 = 9 :=
by
  sorry

end initial_hamburgers_count_is_nine_l194_194157


namespace malfunctioning_clock_fraction_correct_l194_194864

noncomputable def malfunctioning_clock_correct_time_fraction : ℚ := 5 / 8

theorem malfunctioning_clock_fraction_correct :
  malfunctioning_clock_correct_time_fraction = 5 / 8 := 
by
  sorry

end malfunctioning_clock_fraction_correct_l194_194864


namespace max_f_value_l194_194400

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin x + 12 * Real.cos x

theorem max_f_value : ∃ x : ℝ, f x = 13 :=
sorry

end max_f_value_l194_194400


namespace max_f_value_l194_194398

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin x + 12 * Real.cos x

theorem max_f_value : ∃ x : ℝ, f x = 13 :=
sorry

end max_f_value_l194_194398


namespace monthly_rent_of_shop_l194_194278

theorem monthly_rent_of_shop
  (length width : ℕ)
  (annual_rent_per_sq_ft : ℕ)
  (length_def : length = 18)
  (width_def : width = 22)
  (annual_rent_per_sq_ft_def : annual_rent_per_sq_ft = 68) :
  (18 * 22 * 68) / 12 = 2244 := 
by
  sorry

end monthly_rent_of_shop_l194_194278


namespace count_valid_numbers_l194_194075

def is_valid_number (n : Nat) : Prop := 
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  1 ≤ hundreds ∧ hundreds ≤ 9 ∧
  0 ≤ tens ∧ tens ≤ 9 ∧
  units ≥ 3 * tens

theorem count_valid_numbers : 
  (Finset.filter is_valid_number (Finset.range 1000)).card = 198 :=
by sorry

end count_valid_numbers_l194_194075


namespace sin_45_deg_l194_194295

theorem sin_45_deg : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by 
  -- placeholder for the actual proof
  sorry

end sin_45_deg_l194_194295


namespace printers_finish_tasks_l194_194869

theorem printers_finish_tasks :
  ∀ (start_time_1 finish_half_time_1 start_time_2 : ℕ) (half_task_duration full_task_duration second_task_duration : ℕ),
    start_time_1 = 9 * 60 ∧
    finish_half_time_1 = 12 * 60 + 30 ∧
    half_task_duration = finish_half_time_1 - start_time_1 ∧
    full_task_duration = 2 * half_task_duration ∧
    start_time_2 = 13 * 60 ∧
    second_task_duration = 2 * 60 ∧
    start_time_1 + full_task_duration = 4 * 60 ∧
    start_time_2 + second_task_duration = 15 * 60 →
  max (start_time_1 + full_task_duration) (start_time_2 + second_task_duration) = 16 * 60 := 
by
  intros start_time_1 finish_half_time_1 start_time_2 half_task_duration full_task_duration second_task_duration
  intro h
  rcases h with ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩
  sorry

end printers_finish_tasks_l194_194869


namespace greatest_teams_l194_194975

-- Define the number of girls and boys as constants
def numGirls : ℕ := 40
def numBoys : ℕ := 32

-- Define the greatest number of teams possible with equal number of girls and boys as teams.
theorem greatest_teams : Nat.gcd numGirls numBoys = 8 := sorry

end greatest_teams_l194_194975


namespace minimum_product_value_l194_194954

-- Problem conditions
def total_stones : ℕ := 40
def b_min : ℕ := 20
def b_max : ℕ := 32

-- Define the product function
def P (b : ℕ) : ℕ := b * (total_stones - b)

-- Goal: Prove the minimum value of P(b) for b in [20, 32] is 256
theorem minimum_product_value : ∃ (b : ℕ), b_min ≤ b ∧ b ≤ b_max ∧ P b = 256 := by
  sorry

end minimum_product_value_l194_194954


namespace find_winner_votes_l194_194978

-- Define the conditions
variables (V : ℝ) (winner_votes second_votes : ℝ)
def election_conditions :=
  winner_votes = 0.468 * V ∧
  second_votes = 0.326 * V ∧
  winner_votes - second_votes = 752

-- State the theorem
theorem find_winner_votes (h : election_conditions V winner_votes second_votes) :
  winner_votes = 2479 :=
sorry

end find_winner_votes_l194_194978


namespace distance_BC_l194_194664

theorem distance_BC (AB AC CD DA: ℝ) (hAB: AB = 50) (hAC: AC = 40) (hCD: CD = 25) (hDA: DA = 35):
  BC = 10 ∨ BC = 90 :=
by
  sorry

end distance_BC_l194_194664


namespace factorize_expression_l194_194042

theorem factorize_expression (a b : ℝ) :
  ab^(3 : ℕ) - 4 * ab = ab * (b + 2) * (b - 2) :=
by
  -- proof to be provided
  sorry

end factorize_expression_l194_194042


namespace minimum_b_l194_194596

open Real

noncomputable def tangent_min_b (a : ℝ) (h : 0 < a) : ℝ := 2 * a * log a - 2 * a

theorem minimum_b (h : ∀ a : ℝ, 0 < a → tangent_min_b a ≥ tangent_min_b 1) : tangent_min_b 1 = -2 :=
by
  -- proof omitted
  sorry

end minimum_b_l194_194596


namespace cos_function_max_value_l194_194572

theorem cos_function_max_value (k : ℤ) : (2 * Real.cos (2 * k * Real.pi) - 1) = 1 :=
by
  -- Proof not included
  sorry

end cos_function_max_value_l194_194572


namespace linear_elimination_l194_194509

theorem linear_elimination (a b : ℤ) (x y : ℤ) :
  (a = 2) ∧ (b = -5) → 
  (a * (5 * x - 2 * y) + b * (2 * x + 3 * y) = 0) → 
  (10 * x - 4 * y + -10 * x - 15 * y = 8 + -45) :=
by
  sorry

end linear_elimination_l194_194509


namespace input_statement_is_INPUT_l194_194284

-- Define the type for statements
inductive Statement
| PRINT
| INPUT
| IF
| END

-- Define roles for the types of statements
def isOutput (s : Statement) : Prop := s = Statement.PRINT
def isInput (s : Statement) : Prop := s = Statement.INPUT
def isConditional (s : Statement) : Prop := s = Statement.IF
def isTermination (s : Statement) : Prop := s = Statement.END

-- Theorem to prove INPUT is the input statement
theorem input_statement_is_INPUT :
  isInput Statement.INPUT := by
  -- Proof to be provided
  sorry

end input_statement_is_INPUT_l194_194284


namespace ratio_of_ages_l194_194222

theorem ratio_of_ages (F C : ℕ) (h1 : F = C) (h2 : F = 75) :
  (C + 5 * 15) / (F + 15) = 5 / 3 :=
by
  sorry

end ratio_of_ages_l194_194222


namespace theta_in_fourth_quadrant_l194_194919

theorem theta_in_fourth_quadrant (θ : ℝ) (h1 : Real.cos θ > 0) (h2 : Real.tan θ < 0) : 
  (π < θ ∧ θ < 2 * π) :=
by
  sorry

end theta_in_fourth_quadrant_l194_194919


namespace vertex_closest_point_l194_194763

theorem vertex_closest_point (a : ℝ) (x y : ℝ) :
  (x^2 = 2 * y) ∧ (y ≥ 0) ∧ ((y^2 + 2 * (1 - a) * y + a^2) ≤ 0) → a ≤ 1 :=
by 
  sorry

end vertex_closest_point_l194_194763


namespace problem_statement_l194_194883

noncomputable def G (x : ℝ) : ℝ := ((x + 1) ^ 2) / 2 - 4

theorem problem_statement : G (G (G 0)) = -3.9921875 :=
by
  sorry

end problem_statement_l194_194883


namespace largest_c_value_l194_194043

theorem largest_c_value (c : ℝ) :
  (∃ x : ℝ, x^2 + 5 * x + c = -3) → c ≤ 13 / 4 :=
sorry

end largest_c_value_l194_194043


namespace convert_quadratic_l194_194173

theorem convert_quadratic (x : ℝ) :
  (1 + 3 * x) * (x - 3) = 2 * x ^ 2 + 1 ↔ x ^ 2 - 8 * x - 4 = 0 := 
by sorry

end convert_quadratic_l194_194173


namespace unique_solution_sin_tan_eq_l194_194588

noncomputable def S (x : ℝ) : ℝ := Real.tan (Real.sin x) - Real.sin x

theorem unique_solution_sin_tan_eq (h : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ Real.arcsin (1/2) → S x < S y) :
  ∃! x, 0 ≤ x ∧ x ≤ Real.arcsin (1/2) ∧ Real.sin x = Real.tan (Real.sin x) := by
sorry

end unique_solution_sin_tan_eq_l194_194588


namespace fraction_to_decimal_l194_194893

-- We define the fraction and its simplified form
def fraction : ℚ := 58 / 160
def simplified_fraction : ℚ := 29 / 80

-- We state that the fraction simplifies correctly
lemma simplify_fraction : fraction = simplified_fraction := by
  sorry

-- Define the factorization of the denominator
def denominator_factorization : ℕ := 2^4 * 5

-- Verify the fraction when multiplied by 125/125
def equalized_fraction : ℚ := 29 * 125 / 10000

-- State the final result as a decimal
theorem fraction_to_decimal : fraction = 0.3625 := by
  sorry

end fraction_to_decimal_l194_194893


namespace Greg_harvested_acres_l194_194195

-- Defining the conditions
def Sharon_harvested : ℝ := 0.1
def Greg_harvested (additional: ℝ) (Sharon: ℝ) : ℝ := Sharon + additional

-- Proving the statement
theorem Greg_harvested_acres : Greg_harvested 0.3 Sharon_harvested = 0.4 :=
by
  sorry

end Greg_harvested_acres_l194_194195


namespace fraction_is_terminating_decimal_l194_194891

noncomputable def fraction_to_decimal : ℚ :=
  58 / 160

theorem fraction_is_terminating_decimal : fraction_to_decimal = 3625 / 10000 :=
by
  sorry

end fraction_is_terminating_decimal_l194_194891


namespace price_increase_is_12_percent_l194_194874

theorem price_increase_is_12_percent
    (P : ℝ) (d : ℝ) (P' : ℝ) (sale_price : ℝ) (increase : ℝ) (percentage_increase : ℝ) :
    P = 470 → d = 0.16 → P' = 442.18 → 
    sale_price = P - P * d →
    increase = P' - sale_price →
    percentage_increase = (increase / sale_price) * 100 →
    percentage_increase = 12 :=
  by
  sorry

end price_increase_is_12_percent_l194_194874


namespace average_annual_growth_rate_l194_194270

variable (a b : ℝ)

theorem average_annual_growth_rate :
  ∃ x : ℝ, (1 + x)^2 = (1 + a) * (1 + b) ∧ x = Real.sqrt ((1 + a) * (1 + b)) - 1 := by
  sorry

end average_annual_growth_rate_l194_194270


namespace usable_area_is_correct_l194_194275

variable (x : ℝ)

def total_field_area : ℝ := (x + 9) * (x + 7)
def flooded_area : ℝ := (2 * x - 2) * (x - 1)
def usable_area : ℝ := total_field_area x - flooded_area x

theorem usable_area_is_correct : usable_area x = -x^2 + 20 * x + 61 :=
by
  sorry

end usable_area_is_correct_l194_194275


namespace minimum_value_expression_l194_194619

variable (a b c : ℝ)
variable (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 4)

theorem minimum_value_expression : (a + 3 * b) * (2 * b + 3 * c) * (a * c + 2) = 192 := by
  sorry

end minimum_value_expression_l194_194619


namespace well_defined_interval_l194_194687

def is_well_defined (x : ℝ) : Prop :=
  (5 - x > 0) ∧ (x ≠ 2)

theorem well_defined_interval : 
  ∀ x : ℝ, (is_well_defined x) ↔ (x < 5 ∧ x ≠ 2) :=
by 
  sorry

end well_defined_interval_l194_194687


namespace square_free_even_less_than_200_count_l194_194916

def is_square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m^2 ∣ n → m = 1

def is_multiple_of_2 (n : ℕ) : Prop := n % 2 = 0

theorem square_free_even_less_than_200_count : ∃ (count : ℕ), count = 38 ∧ (∀ n : ℕ, n < 200 ∧ is_multiple_of_2 n ∧ is_square_free n → count = 38) :=
by
  sorry

end square_free_even_less_than_200_count_l194_194916


namespace maria_savings_percentage_is_33_l194_194876

noncomputable def regular_price : ℝ := 60
noncomputable def second_pair_price : ℝ := regular_price - (0.4 * regular_price)
noncomputable def third_pair_price : ℝ := regular_price - (0.6 * regular_price)
noncomputable def total_regular_price : ℝ := 3 * regular_price
noncomputable def total_discounted_price : ℝ := regular_price + second_pair_price + third_pair_price
noncomputable def savings : ℝ := total_regular_price - total_discounted_price
noncomputable def savings_percentage : ℝ := (savings / total_regular_price) * 100

theorem maria_savings_percentage_is_33 :
  savings_percentage = 33 :=
by
  sorry

end maria_savings_percentage_is_33_l194_194876


namespace represent_1917_as_sum_diff_of_squares_l194_194112

theorem represent_1917_as_sum_diff_of_squares : ∃ a b c : ℤ, 1917 = a^2 - b^2 + c^2 :=
by
  use 480, 478, 1
  sorry

end represent_1917_as_sum_diff_of_squares_l194_194112


namespace bea_has_max_profit_l194_194289

theorem bea_has_max_profit : 
  let price_bea := 25
  let price_dawn := 28
  let price_carla := 35
  let sold_bea := 10
  let sold_dawn := 8
  let sold_carla := 6
  let cost_bea := 10
  let cost_dawn := 12
  let cost_carla := 15
  let profit_bea := (price_bea * sold_bea) - (cost_bea * sold_bea)
  let profit_dawn := (price_dawn * sold_dawn) - (cost_dawn * sold_dawn)
  let profit_carla := (price_carla * sold_carla) - (cost_carla * sold_carla)
  profit_bea = 150 ∧ profit_dawn = 128 ∧ profit_carla = 120 ∧ ∀ p, p ∈ [profit_bea, profit_dawn, profit_carla] → p ≤ 150 :=
by
  sorry

end bea_has_max_profit_l194_194289


namespace value_of_expression_l194_194437

noncomputable def largestNegativeInteger : Int := -1

theorem value_of_expression (a b x y : ℝ) (m : Int)
  (h1 : a + b = 0)
  (h2 : x * y = 1)
  (h3 : m = largestNegativeInteger) :
  2023 * (a + b) + 3 * |m| - 2 * (x * y) = 1 :=
by
  sorry

end value_of_expression_l194_194437


namespace set_operation_correct_l194_194578

-- Define the sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

-- Define the operation A * B
def set_operation (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

-- State the theorem to be proved
theorem set_operation_correct : set_operation A B = {1, 3} :=
sorry

end set_operation_correct_l194_194578


namespace square_sum_inverse_eq_23_l194_194512

theorem square_sum_inverse_eq_23 {x : ℝ} (h : x + 1/x = 5) : x^2 + (1/x)^2 = 23 :=
by
  sorry

end square_sum_inverse_eq_23_l194_194512


namespace solve_inequality_l194_194792

noncomputable def solution_set : Set ℝ := {x | x < -4/3 ∨ x > -13/9}

theorem solve_inequality (x : ℝ) : 
  2 - 1 / (3 * x + 4) < 5 → x ∈ solution_set :=
by
  sorry

end solve_inequality_l194_194792


namespace number_of_intersections_is_four_l194_194685

def LineA (x y : ℝ) : Prop := 3 * x - 2 * y + 4 = 0
def LineB (x y : ℝ) : Prop := 6 * x + 4 * y - 12 = 0
def LineC (x y : ℝ) : Prop := x - y + 1 = 0
def LineD (x y : ℝ) : Prop := y - 2 = 0

def is_intersection (L1 L2 : ℝ → ℝ → Prop) (p : ℝ × ℝ) : Prop := L1 p.1 p.2 ∧ L2 p.1 p.2

theorem number_of_intersections_is_four :
  (∃ p1 : ℝ × ℝ, is_intersection LineA LineB p1) ∧
  (∃ p2 : ℝ × ℝ, is_intersection LineC LineD p2) ∧
  (∃ p3 : ℝ × ℝ, is_intersection LineA LineD p3) ∧
  (∃ p4 : ℝ × ℝ, is_intersection LineB LineD p4) ∧
  (p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4) :=
by
  sorry

end number_of_intersections_is_four_l194_194685


namespace first_team_more_points_l194_194769

/-
Conditions:
  - Beth scored 12 points.
  - Jan scored 10 points.
  - Judy scored 8 points.
  - Angel scored 11 points.
Question:
  - How many more points did the first team get than the second team?
Prove that the first team scored 3 points more than the second team.
-/

theorem first_team_more_points
  (Beth_score : ℕ)
  (Jan_score : ℕ)
  (Judy_score : ℕ)
  (Angel_score : ℕ)
  (First_team_total : ℕ := Beth_score + Jan_score)
  (Second_team_total : ℕ := Judy_score + Angel_score)
  (Beth_score_val : Beth_score = 12)
  (Jan_score_val : Jan_score = 10)
  (Judy_score_val : Judy_score = 8)
  (Angel_score_val : Angel_score = 11)
  : First_team_total - Second_team_total = 3 := by
  sorry

end first_team_more_points_l194_194769


namespace tan_theta_eq_sqrt3_div_3_l194_194425

theorem tan_theta_eq_sqrt3_div_3
  (θ : ℝ)
  (h : (Real.cos θ * Real.sqrt 3 + Real.sin θ) = 2) :
  Real.tan θ = Real.sqrt 3 / 3 := by
  sorry

end tan_theta_eq_sqrt3_div_3_l194_194425


namespace specimen_exchange_l194_194264

theorem specimen_exchange (x : ℕ) (h : x * (x - 1) = 110) : x * (x - 1) = 110 := by
  exact h

end specimen_exchange_l194_194264


namespace find_annual_interest_rate_l194_194612

open Real

-- Definitions of initial conditions
def P : ℝ := 10000
def A : ℝ := 10815.83
def n : ℝ := 2
def t : ℝ := 2

-- Statement of the problem
theorem find_annual_interest_rate (r : ℝ) : A = P * (1 + r / n) ^ (n * t) → r = 0.0398 :=
by
  sorry

end find_annual_interest_rate_l194_194612


namespace parallel_vectors_sin_cos_l194_194913

theorem parallel_vectors_sin_cos (θ : ℝ) (a := (6, 3)) (b := (Real.sin θ, Real.cos θ))
  (h : (∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2)) :
  Real.sin (2 * θ) - 2 * (Real.cos θ)^2 = 2 / 5 :=
by
  sorry

end parallel_vectors_sin_cos_l194_194913


namespace phil_packs_duration_l194_194471

noncomputable def total_cards_left_after_fire : ℕ := 520
noncomputable def total_cards_initially : ℕ := total_cards_left_after_fire * 2
noncomputable def cards_per_pack : ℕ := 20
noncomputable def packs_bought_weeks : ℕ := total_cards_initially / cards_per_pack

theorem phil_packs_duration : packs_bought_weeks = 52 := by
  sorry

end phil_packs_duration_l194_194471


namespace jamshid_takes_less_time_l194_194203

open Real

theorem jamshid_takes_less_time (J : ℝ) (hJ : J < 15) (h_work_rate : (1 / J) + (1 / 15) = 1 / 5) :
  (15 - J) / 15 * 100 = 50 :=
by
  sorry

end jamshid_takes_less_time_l194_194203


namespace find_x_plus_y_l194_194754

theorem find_x_plus_y (x y : ℚ) (h1 : 3 * x - 4 * y = 18) (h2 : x + 3 * y = -1) :
  x + y = 29 / 13 :=
sorry

end find_x_plus_y_l194_194754


namespace sin_45_degree_l194_194300

def Q : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), real.sin (real.pi / 4))
def E : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), 0)
def O : (x:ℝ) × (y:ℝ) := (0,0)
def OQ : ℝ := real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2)

theorem sin_45_degree : ∃ x: ℝ, x = real.sin (real.pi / 4) ∧ x = real.sqrt 2 / 2 :=
by sorry

end sin_45_degree_l194_194300


namespace sin_45_deg_eq_one_div_sqrt_two_l194_194331

def unit_circle_radius : ℝ := 1

def forty_five_degrees_in_radians : ℝ := (Real.pi / 4)

def cos_45 : ℝ := Real.cos forty_five_degrees_in_radians

def sin_45 : ℝ := Real.sin forty_five_degrees_in_radians

theorem sin_45_deg_eq_one_div_sqrt_two : 
  sin_45 = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_deg_eq_one_div_sqrt_two_l194_194331


namespace fifth_score_l194_194047

theorem fifth_score (r : ℕ) 
  (h1 : r % 5 = 0)
  (h2 : (60 + 75 + 85 + 95 + r) / 5 = 80) : 
  r = 85 := by 
  sorry

end fifth_score_l194_194047


namespace sin_45_deg_l194_194297

theorem sin_45_deg : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by 
  -- placeholder for the actual proof
  sorry

end sin_45_deg_l194_194297


namespace solve_system_part1_solve_system_part3_l194_194011

noncomputable def solution_part1 : Prop :=
  ∃ (x y : ℝ), (x + y = 2) ∧ (5 * x - 2 * (x + y) = 6) ∧ (x = 2) ∧ (y = 0)

-- Part (1) Statement
theorem solve_system_part1 : solution_part1 := sorry

noncomputable def solution_part3 : Prop :=
  ∃ (a b c : ℝ), (a + b = 3) ∧ (5 * a + 3 * c = 1) ∧ (a + b + c = 0) ∧ (a = 2) ∧ (b = 1) ∧ (c = -3)

-- Part (3) Statement
theorem solve_system_part3 : solution_part3 := sorry

end solve_system_part1_solve_system_part3_l194_194011


namespace sin_45_eq_sqrt2_div_2_l194_194323

theorem sin_45_eq_sqrt2_div_2 : Real.sin (π / 4) = Real.sqrt 2 / 2 := 
sorry

end sin_45_eq_sqrt2_div_2_l194_194323


namespace prime_square_implies_equal_l194_194274

theorem prime_square_implies_equal (p : ℕ) (hp : Nat.Prime p) (hp_gt_2 : p > 2)
  (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ (p-1)/2) (hy : 1 ≤ y ∧ y ≤ (p-1)/2)
  (h_square: ∃ k : ℕ, x * (p - x) * y * (p - y) = k ^ 2) : x = y :=
sorry

end prime_square_implies_equal_l194_194274


namespace totalBooksOnShelves_l194_194259

-- Define the conditions
def numShelves : Nat := 150
def booksPerShelf : Nat := 15

-- Define the statement to be proved
theorem totalBooksOnShelves : numShelves * booksPerShelf = 2250 :=
by
  -- Skipping the proof
  sorry

end totalBooksOnShelves_l194_194259


namespace sleeping_bag_selling_price_l194_194180

def wholesale_cost : ℝ := 24.56
def gross_profit_percentage : ℝ := 0.14

def gross_profit (x : ℝ) : ℝ := gross_profit_percentage * x

def selling_price (x y : ℝ) : ℝ := x + y

theorem sleeping_bag_selling_price :
  selling_price wholesale_cost (gross_profit wholesale_cost) = 28 := by
  sorry

end sleeping_bag_selling_price_l194_194180


namespace gather_half_of_nuts_l194_194167

open Nat

theorem gather_half_of_nuts (a b c : ℕ) (h₀ : (a + b + c) % 2 = 0) : ∃ k, k = (a + b + c) / 2 :=
  sorry

end gather_half_of_nuts_l194_194167


namespace binary_11101_to_decimal_l194_194561

theorem binary_11101_to_decimal : 
  (1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 29) := by
  sorry

end binary_11101_to_decimal_l194_194561


namespace greatest_integer_l194_194204

theorem greatest_integer (n : ℕ) (h1 : n < 150) (h2 : ∃ k : ℤ, n = 9 * k - 2) (h3 : ∃ l : ℤ, n = 8 * l - 4) : n = 124 := 
sorry

end greatest_integer_l194_194204


namespace minimum_ticket_cost_l194_194199

theorem minimum_ticket_cost 
  (N : ℕ)
  (southern_cities : Fin 4)
  (northern_cities : Fin 5)
  (one_way_cost : ∀ (A B : city), A ≠ B → ticket_cost A B = N)
  (round_trip_cost : ∀ (A B : city), A ≠ B → ticket_cost_round_trip A B = 1.6 * N) :
  ∃ (minimum_cost : ℕ), minimum_cost = 6.4 * N := 
sorry

end minimum_ticket_cost_l194_194199


namespace time_to_cross_signal_pole_l194_194515

/-- Definitions representing the given conditions --/
def length_of_train : ℕ := 300
def time_to_cross_platform : ℕ := 39
def length_of_platform : ℕ := 350
def total_distance := length_of_train + length_of_platform
def speed_of_train := total_distance / time_to_cross_platform

/-- Main statement to be proven --/
theorem time_to_cross_signal_pole : length_of_train / speed_of_train = 18 := by
  sorry

end time_to_cross_signal_pole_l194_194515


namespace sin_45_eq_sqrt_two_over_two_l194_194370

theorem sin_45_eq_sqrt_two_over_two : Real.sin (π / 4) = sqrt 2 / 2 :=
by
  sorry

end sin_45_eq_sqrt_two_over_two_l194_194370


namespace time_to_cross_signal_pole_l194_194516

/-- Definitions representing the given conditions --/
def length_of_train : ℕ := 300
def time_to_cross_platform : ℕ := 39
def length_of_platform : ℕ := 350
def total_distance := length_of_train + length_of_platform
def speed_of_train := total_distance / time_to_cross_platform

/-- Main statement to be proven --/
theorem time_to_cross_signal_pole : length_of_train / speed_of_train = 18 := by
  sorry

end time_to_cross_signal_pole_l194_194516


namespace sin_gamma_isosceles_l194_194088

theorem sin_gamma_isosceles (a c m_a m_c s_1 s_2 : ℝ) (γ : ℝ) 
  (h1 : a + m_c = s_1) (h2 : c + m_a = s_2) :
  Real.sin γ = (s_2 / (2 * s_1)) * Real.sqrt ((4 * s_1^2) - s_2^2) :=
sorry

end sin_gamma_isosceles_l194_194088


namespace sin_45_degree_eq_sqrt2_div_2_l194_194351

theorem sin_45_degree_eq_sqrt2_div_2 :
  let θ := (real.pi / 4)
  in sin θ = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_degree_eq_sqrt2_div_2_l194_194351


namespace eval_expression_correct_l194_194700

noncomputable def eval_expression : ℝ := (-64)^(4/3)

theorem eval_expression_correct : eval_expression = 256 := by
  sorry

end eval_expression_correct_l194_194700


namespace zoey_finished_on_monday_l194_194007

def total_days_read (n : ℕ) : ℕ :=
  2 * ((2^n) - 1)

def day_of_week_finished (start_day : ℕ) (total_days : ℕ) : ℕ :=
  (start_day + total_days) % 7

theorem zoey_finished_on_monday :
  day_of_week_finished 1 (total_days_read 18) = 1 :=
by
  sorry

end zoey_finished_on_monday_l194_194007


namespace not_possible_to_get_105_single_stone_piles_l194_194976

noncomputable def piles : List Nat := [51, 49, 5]
def combine (a b : Nat) : Nat := a + b
def split (a : Nat) : List Nat := if a % 2 = 0 then [a / 2, a / 2] else [a]

theorem not_possible_to_get_105_single_stone_piles 
  (initial_piles : List Nat := piles) 
  (combine : Nat → Nat → Nat := combine) 
  (split : Nat → List Nat := split) :
  ¬ ∃ (final_piles : List Nat), final_piles.length = 105 ∧ (∀ n ∈ final_piles, n = 1) :=
by
  sorry

end not_possible_to_get_105_single_stone_piles_l194_194976


namespace contractor_absent_days_l194_194528

variable (x y : ℝ) -- x for the number of days worked, y for the number of days absent

-- Conditions
def eng_days := x + y = 30
def total_money := 25 * x - 7.5 * y = 425

-- Theorem
theorem contractor_absent_days (x y : ℝ) (h1 : eng_days x y) (h2 : total_money x y) : y = 10 := 
sorry

end contractor_absent_days_l194_194528


namespace eleven_pow_603_mod_500_eq_331_l194_194503

theorem eleven_pow_603_mod_500_eq_331 : 11^603 % 500 = 331 := by
  sorry

end eleven_pow_603_mod_500_eq_331_l194_194503


namespace true_root_30_40_l194_194379

noncomputable def u (x : ℝ) : ℝ := Real.sqrt (x + 15)
noncomputable def original_eqn (x : ℝ) : Prop := u x - 3 / (u x) = 4

theorem true_root_30_40 : ∃ (x : ℝ), 30 < x ∧ x < 40 ∧ original_eqn x :=
by
  sorry

end true_root_30_40_l194_194379


namespace smallest_solution_x4_50x2_576_eq_0_l194_194138

theorem smallest_solution_x4_50x2_576_eq_0 :
  ∃ x : ℝ, (x^4 - 50*x^2 + 576 = 0) ∧ ∀ y : ℝ, y^4 - 50*y^2 + 576 = 0 → x ≤ y :=
sorry

end smallest_solution_x4_50x2_576_eq_0_l194_194138


namespace range_of_x_l194_194737

variable {f : ℝ → ℝ}
variable (hf1 : ∀ x : ℝ, has_deriv_at f (derivative f x) x)
variable (hf2 : ∀ x : ℝ, derivative f x > - f x)

theorem range_of_x (h : f (Real.log 3) = 1/3) : 
  {x : ℝ | f x > 1 / Real.exp x} = Set.Ioi (Real.log 3) := 
by 
  sorry

end range_of_x_l194_194737


namespace max_value_sin_cos_combination_l194_194396

theorem max_value_sin_cos_combination :
  ∀ x : ℝ, (5 * Real.sin x + 12 * Real.cos x) ≤ 13 :=
by
  intro x
  sorry

end max_value_sin_cos_combination_l194_194396


namespace average_speed_train_l194_194252

theorem average_speed_train (d1 d2 : ℝ) (t1 t2 : ℝ) 
  (h_d1 : d1 = 325) (h_d2 : d2 = 470)
  (h_t1 : t1 = 3.5) (h_t2 : t2 = 4) :
  (d1 + d2) / (t1 + t2) = 106 :=
by
  sorry

end average_speed_train_l194_194252


namespace total_legs_l194_194798

def animals_legs (dogs : Nat) (birds : Nat) (insects : Nat) : Nat :=
  (dogs * 4) + (birds * 2) + (insects * 6)

theorem total_legs :
  animals_legs 3 2 2 = 22 := by
  sorry

end total_legs_l194_194798


namespace sin_45_eq_sqrt2_div_2_l194_194320

theorem sin_45_eq_sqrt2_div_2 : Real.sin (π / 4) = Real.sqrt 2 / 2 := 
sorry

end sin_45_eq_sqrt2_div_2_l194_194320


namespace atomic_weight_of_chlorine_l194_194178

theorem atomic_weight_of_chlorine (molecular_weight_AlCl3 : ℝ) (atomic_weight_Al : ℝ) (atomic_weight_Cl : ℝ) :
  molecular_weight_AlCl3 = 132 ∧ atomic_weight_Al = 26.98 →
  132 = 26.98 + 3 * atomic_weight_Cl →
  atomic_weight_Cl = 35.007 :=
by
  intros h1 h2
  sorry

end atomic_weight_of_chlorine_l194_194178


namespace oil_output_per_capita_correctness_l194_194996

variable (population_west : ℝ := 1)
variable (output_west : ℝ := 55.084)
variable (population_non_west : ℝ := 6.9)
variable (output_non_west : ℝ := 1480.689)
variable (output_russia_9_percent : ℝ := 13737.1)
variable (percentage : ℝ := 9)
variable (total_population_russia : ℝ := 147)

def west_output_per_capita : ℝ :=
  output_west / population_west

def non_west_output_per_capita : ℝ :=
  output_non_west / population_non_west

def total_output_russia : ℝ :=
  (output_russia_9_percent * 100) / percentage

def russia_output_per_capita : ℝ :=
  total_output_russia / total_population_russia

theorem oil_output_per_capita_correctness :
  west_output_per_capita = 55.084 ∧
  non_west_output_per_capita = 214.59 ∧
  total_output_russia = 152634.44 ∧
  russia_output_per_capita = 1038.33 :=
by
  sorry

end oil_output_per_capita_correctness_l194_194996


namespace train_lengths_equal_l194_194009

theorem train_lengths_equal (v_fast v_slow : ℝ) (t : ℝ) (L : ℝ)  
  (h1 : v_fast = 46) 
  (h2 : v_slow = 36) 
  (h3 : t = 36.00001) : 
  2 * L = (v_fast - v_slow) / 3600 * t → L = 1800.0005 := 
by
  sorry

end train_lengths_equal_l194_194009


namespace probability_at_least_four_8s_in_five_rolls_l194_194154

-- Definitions 
def prob_three_favorable : ℚ := 3 / 10

def prob_at_least_four_times_in_five_rolls : ℚ := 5 * (prob_three_favorable^4) * ((7 : ℚ)/10) + (prob_three_favorable)^5

-- The proof statement
theorem probability_at_least_four_8s_in_five_rolls : prob_at_least_four_times_in_five_rolls = 2859.3 / 10000 :=
by
  sorry

end probability_at_least_four_8s_in_five_rolls_l194_194154


namespace minimize_expr_l194_194616

-- Define the function we need to minimize
noncomputable def expr (α β : ℝ) : ℝ := 
  (2 * Real.cos α + 5 * Real.sin β - 8)^2 + (2 * Real.sin α + 5 * Real.cos β - 15)^2

-- State the theorem to prove the minimum value of this expression
theorem minimize_expr (α β : ℝ) : ∃ (α β : ℝ), expr α β = 100 := 
sorry

end minimize_expr_l194_194616


namespace prob_both_A_B_prob_exactly_one_l194_194836

def prob_A : ℝ := 0.8
def prob_not_B : ℝ := 0.1
def prob_B : ℝ := 1 - prob_not_B

lemma prob_independent (a b : Prop) : Prop := -- Placeholder for actual independence definition
sorry

-- Given conditions
variables (P_A : ℝ := prob_A) (P_not_B : ℝ := prob_not_B) (P_B : ℝ := prob_B) (indep : ∀ A B, prob_independent A B)

-- Questions translated to Lean statements
theorem prob_both_A_B : P_A * P_B = 0.72 := sorry

theorem prob_exactly_one : (P_A * P_not_B) + ((1 - P_A) * P_B) = 0.26 := sorry

end prob_both_A_B_prob_exactly_one_l194_194836


namespace sum_of_cubes_l194_194959

theorem sum_of_cubes (x y z : ℝ) (h1 : x + y + z = 7) (h2 : xy + xz + yz = 9) (h3 : xyz = -18) :
  x^3 + y^3 + z^3 = 100 :=
by
  sorry

end sum_of_cubes_l194_194959


namespace minimum_disks_needed_l194_194206

-- Define the conditions
def total_files : ℕ := 25
def disk_capacity : ℝ := 2.0
def files_06MB : ℕ := 5
def size_06MB_file : ℝ := 0.6
def files_10MB : ℕ := 10
def size_10MB_file : ℝ := 1.0
def files_03MB : ℕ := total_files - files_06MB - files_10MB
def size_03MB_file : ℝ := 0.3

-- Define the theorem that needs to be proved
theorem minimum_disks_needed : 
    ∃ (disks: ℕ), disks = 10 ∧ 
    (5 * size_06MB_file + 10 * size_10MB_file + 10 * size_03MB_file) ≤ disks * disk_capacity := 
by
  sorry

end minimum_disks_needed_l194_194206


namespace solution_set_inequality_l194_194727

theorem solution_set_inequality (x : ℝ) :
  (|x + 3| - |x - 3| > 3) ↔ (x > 3 / 2) := 
sorry

end solution_set_inequality_l194_194727


namespace length_of_AP_l194_194445

variables {x : ℝ} (M B C P A : Point) (circle : Circle)
  (BC AB MP : Line)

-- Definitions of conditions
def is_midpoint_of_arc (M B C : Point) (circle : Circle) : Prop := sorry
def is_perpendicular (MP AB : Line) (P : Point) : Prop := sorry
def chord_length (BC : Line) (length : ℝ) : Prop := sorry
def segment_length (BP : Line) (length : ℝ) : Prop := sorry

-- Prove statement
theorem length_of_AP
  (h1 : is_midpoint_of_arc M B C circle)
  (h2 : is_perpendicular MP AB P)
  (h3 : chord_length BC (2 * x))
  (h4 : segment_length BP (3 * x)) :
  ∃AP : Line, segment_length AP (2 * x) :=
sorry

end length_of_AP_l194_194445


namespace remainder_m_n_mod_1000_l194_194098

noncomputable def m : ℕ :=
  Nat.card {p : ℕ × ℕ × ℕ // 4 * p.1 + 3 * p.2 + 2 * p.2.1 = 2009 ∧ 0 < p.1 ∧ 0 < p.2 ∧ 0 < p.2.1}

noncomputable def n : ℕ :=
  Nat.card {p : ℕ × ℕ × ℕ // 4 * p.1 + 3 * p.2 + 2 * p.2.1 = 2000 ∧ 0 < p.1 ∧ 0 < p.2 ∧ 0 < p.2.1}

theorem remainder_m_n_mod_1000 : (m - n) % 1000 = 0 :=
by
  sorry

end remainder_m_n_mod_1000_l194_194098


namespace logarithmic_condition_solution_l194_194895

noncomputable def logarithmic_points_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | log (p.1^3 + (1 / 3) * p.2^3 + (1 / 9)) = log p.1 + log p.2}

theorem logarithmic_condition_solution :
  logarithmic_points_set = { (⟨ real.cbrt 3 / 3, real.cbrt 9 / 3 ⟩) } := 
by 
  sorry

end logarithmic_condition_solution_l194_194895


namespace find_b_in_triangle_l194_194083

theorem find_b_in_triangle (c : ℝ) (B C : ℝ) (h1 : c = Real.sqrt 3)
  (h2 : B = Real.pi / 4) (h3 : C = Real.pi / 3) : ∃ b : ℝ, b = Real.sqrt 2 :=
by
  sorry

end find_b_in_triangle_l194_194083


namespace evaluate_expr_l194_194386

def x := 2
def y := -1
def z := 3
def expr := 2 * x^2 + y^2 - z^2 + 3 * x * y

theorem evaluate_expr : expr = -6 :=
by sorry

end evaluate_expr_l194_194386


namespace fred_walking_speed_l194_194404

/-- 
Fred and Sam are standing 55 miles apart and they start walking in a straight line toward each other
at the same time. Fred walks at a certain speed and Sam walks at a constant speed of 5 miles per hour.
Sam has walked 25 miles when they meet.
-/
theorem fred_walking_speed
  (initial_distance : ℕ) 
  (sam_speed : ℕ)
  (sam_distance : ℕ) 
  (meeting_time : ℕ)
  (fred_distance : ℕ) 
  (fred_speed : ℕ)
  (h_initial_distance : initial_distance = 55)
  (h_sam_speed : sam_speed = 5)
  (h_sam_distance : sam_distance = 25)
  (h_meeting_time : meeting_time = 5)
  (h_fred_distance : fred_distance = 30)
  (h_fred_speed : fred_speed = 6)
  : fred_speed = fred_distance / meeting_time :=
by sorry

end fred_walking_speed_l194_194404


namespace maximize_tetrahedron_volume_l194_194542

noncomputable def volume_of_tetrahedron (a : ℝ) : ℝ :=
  a / 6

theorem maximize_tetrahedron_volume (a : ℝ) (h_a : 0 < a) 
  (P Q X Y : ℝ × ℝ × ℝ) (h_PQ : dist P Q = 1) (h_XY : dist X Y = 1) :
  volume_of_tetrahedron a = a / 6 :=
by
  sorry

end maximize_tetrahedron_volume_l194_194542


namespace curtain_additional_material_l194_194381

theorem curtain_additional_material
  (room_height_feet : ℕ)
  (curtain_length_inches : ℕ)
  (height_conversion_factor : ℕ)
  (desired_length : ℕ)
  (h_room_height_conversion : room_height_feet * height_conversion_factor = 96)
  (h_desired_length : desired_length = 101) :
  curtain_length_inches = desired_length - (room_height_feet * height_conversion_factor) :=
by
  sorry

end curtain_additional_material_l194_194381


namespace regular_polygon_sides_l194_194276

theorem regular_polygon_sides (n : ℕ) (h1 : 2 ≤ n) (h2 : (n - 2) * 180 / n = 120) : n = 6 :=
by
  sorry

end regular_polygon_sides_l194_194276


namespace appropriate_grouping_43_neg78_27_neg52_l194_194849

theorem appropriate_grouping_43_neg78_27_neg52 :
  let a := 43
  let b := -78
  let c := 27
  let d := -52
  (a + c) + (b + d) = -60 :=
by
  let a := 43
  let b := -78
  let c := 27
  let d := -52
  sorry

end appropriate_grouping_43_neg78_27_neg52_l194_194849


namespace find_other_number_l194_194228

theorem find_other_number (a b : ℕ) (gcd_ab : Nat.gcd a b = 45) (lcm_ab : Nat.lcm a b = 1260) (a_eq : a = 180) : b = 315 :=
by
  -- proof goes here
  sorry

end find_other_number_l194_194228


namespace range_of_a_l194_194444

noncomputable def f (a x : ℝ) := a * x - x^2 - Real.log x

theorem range_of_a (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ 2*x₁*x₁ - a*x₁ + 1 = 0 ∧ 
  2*x₂*x₂ - a*x₂ + 1 = 0 ∧ f a x₁ + f a x₂ ≥ 4 + Real.log 2) ↔ 
  a ∈ Set.Ici (2 * Real.sqrt 3) := 
sorry

end range_of_a_l194_194444


namespace total_people_expression_l194_194526

variable {X : ℕ}

def men (X : ℕ) := 24 * X
def women (X : ℕ) := 12 * X
def teenagers (X : ℕ) := 4 * X
def children (X : ℕ) := X

def total_people (X : ℕ) := men X + women X + teenagers X + children X

theorem total_people_expression (X : ℕ) : total_people X = 41 * X :=
by 
  unfold total_people
  unfold men women teenagers children
  sorry

end total_people_expression_l194_194526


namespace min_number_of_bags_l194_194779

theorem min_number_of_bags (a b : ℕ) : 
  ∃ K : ℕ, K = a + b - Nat.gcd a b :=
by
  sorry

end min_number_of_bags_l194_194779


namespace probability_three_white_balls_l194_194862

theorem probability_three_white_balls (total_balls: ℕ) (white_balls: ℕ) (black_balls: ℕ) (drawn_balls: ℕ) 
    (h_total: total_balls = 15) (h_white: white_balls = 7) (h_black: black_balls = 8) (h_drawn: drawn_balls = 3) : 
    ((choose white_balls drawn_balls) / (choose total_balls drawn_balls) : ℚ) = 1 / 13 := 
by {
    -- Definitions and conditions come from part (a)
    -- The lean code should be able to be built successfully
    sorry
} 

end probability_three_white_balls_l194_194862


namespace segment_length_in_meters_l194_194273

-- Conditions
def inch_to_meters : ℝ := 500
def segment_length_in_inches : ℝ := 7.25

-- Theorem to prove
theorem segment_length_in_meters : segment_length_in_inches * inch_to_meters = 3625 := by
  sorry

end segment_length_in_meters_l194_194273


namespace total_books_l194_194523

-- Define the amount of tables
def tables : ℕ := 500

-- Define the proportion of books per table
def proportion_of_books : ℝ := 2 / 5

-- Using these definitions, prove the total number of books is 100,000
theorem total_books (tables : ℕ) (proportion_of_books : ℝ) : 
  let books_per_table := proportion_of_books * tables in
  let total_books := tables * books_per_table in
  total_books = 100000 := 
by 
  -- Here we declare the desired result
  sorry

end total_books_l194_194523


namespace probability_of_first_hearts_and_second_clubs_l194_194832

noncomputable def probability_first_hearts_second_clubs : ℚ :=
  let total_cards := 52
  let hearts_count := 13
  let clubs_count := 13
  let probability_first_hearts := hearts_count / total_cards
  let probability_second_clubs_given_first_hearts := clubs_count / (total_cards - 1)
  probability_first_hearts * probability_second_clubs_given_first_hearts

theorem probability_of_first_hearts_and_second_clubs :
  probability_first_hearts_second_clubs = 13 / 204 :=
by
  sorry

end probability_of_first_hearts_and_second_clubs_l194_194832


namespace sin_45_degree_l194_194305

def Q : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), real.sin (real.pi / 4))
def E : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), 0)
def O : (x:ℝ) × (y:ℝ) := (0,0)
def OQ : ℝ := real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2)

theorem sin_45_degree : ∃ x: ℝ, x = real.sin (real.pi / 4) ∧ x = real.sqrt 2 / 2 :=
by sorry

end sin_45_degree_l194_194305


namespace three_digit_number_units_digit_condition_l194_194077

theorem three_digit_number_units_digit_condition :
  let valid_numbers := (λ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ ((n % 10) ≥ 3 * ((n / 10) % 10))) in
  (Finset.filter valid_numbers (Finset.range 900)) = 198 :=
by
  sorry

end three_digit_number_units_digit_condition_l194_194077


namespace joel_average_speed_l194_194613

theorem joel_average_speed :
  let start_time := (8, 50)
  let end_time := (14, 35)
  let total_distance := 234
  let total_time := (14 - 8) + (35 - 50) / 60
  ∀ start_time end_time total_distance,
    (start_time = (8, 50)) →
    (end_time = (14, 35)) →
    total_distance = 234 →
    (total_time = (14 - 8) + (35 - 50) / 60) →
    total_distance / total_time = 41 :=
by
  sorry

end joel_average_speed_l194_194613


namespace correct_expression_l194_194657

theorem correct_expression (x : ℝ) :
  (x^3 / x^2 = x) :=
by sorry

end correct_expression_l194_194657


namespace perpendicular_lines_condition_l194_194260

theorem perpendicular_lines_condition (a : ℝ) : 
  (∀ x y : ℝ, x + y = 0 ∧ x - ay = 0 → x = 0) ↔ (a = 1) := 
sorry

end perpendicular_lines_condition_l194_194260


namespace students_got_off_l194_194237

-- Define the number of students originally on the bus
def original_students : ℕ := 10

-- Define the number of students left on the bus after the first stop
def students_left : ℕ := 7

-- Prove that the number of students who got off the bus at the first stop is 3
theorem students_got_off : original_students - students_left = 3 :=
by
  sorry

end students_got_off_l194_194237


namespace shaded_percentage_of_large_square_l194_194224

theorem shaded_percentage_of_large_square
  (side_length_small_square : ℕ)
  (side_length_large_square : ℕ)
  (total_border_squares : ℕ)
  (shaded_border_squares : ℕ)
  (central_region_shaded_fraction : ℚ)
  (total_area_large_square : ℚ)
  (shaded_area_border_squares : ℚ)
  (shaded_area_central_region : ℚ) :
  side_length_small_square = 1 →
  side_length_large_square = 5 →
  total_border_squares = 16 →
  shaded_border_squares = 8 →
  central_region_shaded_fraction = 3 / 4 →
  total_area_large_square = 25 →
  shaded_area_border_squares = 8 →
  shaded_area_central_region = (3 / 4) * 9 →
  (shaded_area_border_squares + shaded_area_central_region) / total_area_large_square = 0.59 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end shaded_percentage_of_large_square_l194_194224


namespace annual_interest_rate_l194_194796

theorem annual_interest_rate (initial_amount final_amount : ℝ) 
  (h_initial : initial_amount = 90) 
  (h_final : final_amount = 99) : 
  ((final_amount - initial_amount) / initial_amount) * 100 = 10 :=
by {
  sorry
}

end annual_interest_rate_l194_194796


namespace inequality_with_xy_l194_194905

theorem inequality_with_xy
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x * y = 4) :
  (1 / (x + 3)) + (1 / (y + 3)) ≤ 2 / 5 :=
sorry

end inequality_with_xy_l194_194905


namespace solve_f_lt_zero_solve_f_plus_4_abs_x_minus_4_gt_m_l194_194618

def f (x : ℝ) : ℝ := |3 * x + 1| - |x - 4|

theorem solve_f_lt_zero :
  { x : ℝ | f x < 0 } = { x : ℝ | -5 / 2 < x ∧ x < 3 / 4 } := 
sorry

theorem solve_f_plus_4_abs_x_minus_4_gt_m (m : ℝ) :
  (∀ x : ℝ, f x + 4 * |x - 4| > m) → m < 15 :=
sorry

end solve_f_lt_zero_solve_f_plus_4_abs_x_minus_4_gt_m_l194_194618


namespace fraction_sum_of_lcm_and_gcd_l194_194220

theorem fraction_sum_of_lcm_and_gcd 
  (m n : ℕ) 
  (h_gcd : Nat.gcd m n = 6) 
  (h_lcm : Nat.lcm m n = 210) 
  (h_sum : m + n = 72) :
  1 / (m : ℚ) + 1 / (n : ℚ) = 12 / 210 := 
by
sorry

end fraction_sum_of_lcm_and_gcd_l194_194220


namespace tram_speed_l194_194665

variables (V : ℝ)

theorem tram_speed (h : (V + 5) / (V - 5) = 600 / 225) : V = 11 :=
sorry

end tram_speed_l194_194665


namespace case_one_case_two_l194_194634

theorem case_one (n : ℝ) (h : n > -1) : n^3 + 1 > n^2 + n :=
sorry

theorem case_two (n : ℝ) (h : n < -1) : n^3 + 1 < n^2 + n :=
sorry

end case_one_case_two_l194_194634


namespace cows_in_group_l194_194086

variable (c h : ℕ)

theorem cows_in_group (hcow : 4 * c + 2 * h = 2 * (c + h) + 18) : c = 9 := 
by 
  sorry

end cows_in_group_l194_194086


namespace mod_1237_17_l194_194041

theorem mod_1237_17 : 1237 % 17 = 13 := by
  sorry

end mod_1237_17_l194_194041


namespace sufficient_but_not_necessary_condition_l194_194949

variable (x : ℝ)

theorem sufficient_but_not_necessary_condition (h : x > 1) : x > 0 :=
by
  sorry

end sufficient_but_not_necessary_condition_l194_194949


namespace rectangle_area_l194_194142

theorem rectangle_area (b l : ℕ) (h1: l = 3 * b) (h2: 2 * (l + b) = 120) : l * b = 675 := by
  sorry

end rectangle_area_l194_194142


namespace citizen_income_l194_194662

theorem citizen_income (I : ℝ) (h1 : ∀ I ≤ 40000, 0.15 * I = 8000) 
  (h2 : ∀ I > 40000, (0.15 * 40000 + 0.20 * (I - 40000)) = 8000) : 
  I = 50000 :=
by
  sorry

end citizen_income_l194_194662


namespace value_of_a_l194_194193

def P : Set ℝ := { x | x^2 ≤ 4 }
def M (a : ℝ) : Set ℝ := { a }

theorem value_of_a (a : ℝ) (h : P ∪ {a} = P) : a ∈ { x : ℝ | -2 ≤ x ∧ x ≤ 2 } := by
  sorry

end value_of_a_l194_194193


namespace right_triangle_angle_l194_194450

theorem right_triangle_angle {A B C : ℝ} (hABC : A + B + C = 180) (hC : C = 90) (hA : A = 70) : B = 20 :=
sorry

end right_triangle_angle_l194_194450


namespace flu_infection_l194_194670

theorem flu_infection (x : ℕ) (H : 1 + x + x^2 = 36) : True :=
begin
  sorry
end

end flu_infection_l194_194670


namespace complement_union_covers_until_1_l194_194012

open Set

noncomputable def S := {x : ℝ | x > -2}
noncomputable def T := {x : ℝ | x^2 + 3*x - 4 ≤ 0}
noncomputable def complement_R_S := {x : ℝ | x ≤ -2}
noncomputable def union := complement_R_S ∪ T

theorem complement_union_covers_until_1 : union = {x : ℝ | x ≤ 1} := by
  sorry

end complement_union_covers_until_1_l194_194012


namespace find_min_value_expression_l194_194044

noncomputable def minValueExpression (θ : ℝ) : ℝ :=
  2 * Real.cos θ + 1 / Real.sin θ + Real.sqrt 2 * Real.tan θ

theorem find_min_value_expression :
  ∃ θ : ℝ, 0 < θ ∧ θ < Real.pi / 2 ∧ minValueExpression θ = 3 * Real.sqrt 2 :=
sorry

end find_min_value_expression_l194_194044


namespace container_volume_ratio_l194_194282

variables (A B C : ℝ)

theorem container_volume_ratio (h1 : (2 / 3) * A = (1 / 2) * B) (h2 : (1 / 2) * B = (3 / 5) * C) :
  A / C = 6 / 5 :=
sorry

end container_volume_ratio_l194_194282


namespace evaluate_expression_l194_194253

theorem evaluate_expression :
  |7 - (8^2) * (3 - 12)| - |(5^3) - (Real.sqrt 11)^4| = 579 := 
by 
  sorry

end evaluate_expression_l194_194253


namespace sin_45_eq_sqrt_two_over_two_l194_194371

theorem sin_45_eq_sqrt_two_over_two : Real.sin (π / 4) = sqrt 2 / 2 :=
by
  sorry

end sin_45_eq_sqrt_two_over_two_l194_194371


namespace average_books_collected_per_day_l194_194090

theorem average_books_collected_per_day :
  let n := 7
  let a := 12
  let d := 12
  let S_n := (n * (2 * a + (n - 1) * d)) / 2
  S_n / n = 48 :=
by
  let n := 7
  let a := 12
  let d := 12
  let S_n := (n * (2 * a + (n - 1) * d)) / 2
  show S_n / n = 48
  sorry

end average_books_collected_per_day_l194_194090


namespace find_center_and_radius_sum_l194_194774

-- Define the given equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 - 16 * x + y^2 + 10 * y = -75

-- Define the center of the circle
def center (a b : ℝ) : Prop :=
  ∀ x y : ℝ, circle_eq x y → (x = a) ∧ (y = b)

-- Define the radius of the circle
def radius (r : ℝ) : Prop :=
  ∀ x y : ℝ, circle_eq x y → (x^2 - 16 * x + y^2 + 10 * y = r^2)

-- Main theorem to prove a + b + r = 3 + sqrt 14
theorem find_center_and_radius_sum (a b r : ℝ) (h_cen : center a b) (h_rad : radius r) : 
  a + b + r = 3 + Real.sqrt 14 :=
  sorry

end find_center_and_radius_sum_l194_194774


namespace remaining_budget_correct_l194_194546

def cost_item1 := 13
def cost_item2 := 24
def last_year_remaining_budget := 6
def this_year_budget := 50

theorem remaining_budget_correct :
    (last_year_remaining_budget + this_year_budget - (cost_item1 + cost_item2) = 19) :=
by
  -- This is the statement only, with the proof omitted
  sorry

end remaining_budget_correct_l194_194546


namespace stock_worth_l194_194991

theorem stock_worth (profit_part loss_part total_loss : ℝ) 
  (h1 : profit_part = 0.10) 
  (h2 : loss_part = 0.90) 
  (h3 : total_loss = 400) 
  (profit_rate : ℝ := 0.20) 
  (loss_rate : ℝ := 0.05)
  (profit_value := profit_rate * profit_part)
  (loss_value := loss_rate * loss_part)
  (overall_loss := total_loss)
  (h4 : loss_value - profit_value = overall_loss) :
  ∃ X : ℝ, X = 16000 :=
by
  sorry

end stock_worth_l194_194991


namespace general_formula_an_bounds_Mn_l194_194780

variable {n : ℕ}

-- Define the sequence Sn
def S : ℕ → ℚ := λ n => n * (4 * n - 3) - 2 * n * (n - 1)

-- Define the sequence an based on Sn
def a : ℕ → ℚ := λ n =>
  if n = 0 then 0 else S n - S (n - 1)

-- Define the sequence Mn and the bounds to prove
def M : ℕ → ℚ := λ n => (1 / 4) * (1 - (1 / (4 * n + 1)))

-- Theorem: General formula for the sequence {a_n}
theorem general_formula_an (n : ℕ) (hn : 1 ≤ n) : a n = 4 * n - 3 :=
  sorry

-- Theorem: Bounds for the sequence {M_n}
theorem bounds_Mn (n : ℕ) (hn : 1 ≤ n) : (1 / 5 : ℚ) ≤ M n ∧ M n < (1 / 4) :=
  sorry

end general_formula_an_bounds_Mn_l194_194780


namespace quadratic_inequality_solution_range_l194_194192

open Set Real

theorem quadratic_inequality_solution_range
  (a : ℝ) : (∃ (x1 x2 : ℤ), x1 ≠ x2 ∧ (∀ x : ℝ, x^2 - a * x + 2 * a < 0 ↔ ↑x1 < x ∧ x < ↑x2)) ↔ 
    (a ∈ Icc (-1 : ℝ) ((-1:ℝ)/3)) ∨ (a ∈ Ioo (25 / 3 : ℝ) 9) :=
sorry

end quadratic_inequality_solution_range_l194_194192


namespace smallest_x_2_abs_eq_24_l194_194904

theorem smallest_x_2_abs_eq_24 : ∃ x : ℝ, (2 * |x - 10| = 24) ∧ (∀ y : ℝ, (2 * |y - 10| = 24) -> x ≤ y) := 
sorry

end smallest_x_2_abs_eq_24_l194_194904


namespace fraction_to_decimal_l194_194892

-- We define the fraction and its simplified form
def fraction : ℚ := 58 / 160
def simplified_fraction : ℚ := 29 / 80

-- We state that the fraction simplifies correctly
lemma simplify_fraction : fraction = simplified_fraction := by
  sorry

-- Define the factorization of the denominator
def denominator_factorization : ℕ := 2^4 * 5

-- Verify the fraction when multiplied by 125/125
def equalized_fraction : ℚ := 29 * 125 / 10000

-- State the final result as a decimal
theorem fraction_to_decimal : fraction = 0.3625 := by
  sorry

end fraction_to_decimal_l194_194892


namespace min_value_of_a_plus_b_l194_194740

theorem min_value_of_a_plus_b 
  (a b : ℝ)
  (h_pos_a : a > 0)
  (h_pos_b : b > 0)
  (h_eq : 1 / a + 2 / b = 2) :
  a + b ≥ (3 + 2 * Real.sqrt 2) / 2 :=
sorry

end min_value_of_a_plus_b_l194_194740


namespace addition_problem_l194_194448

theorem addition_problem (x y S : ℕ) 
    (h1 : x = S - 2000)
    (h2 : S = y + 6) :
    x = 6 ∧ y = 2000 ∧ S = 2006 :=
by
  -- The proof will go here
  sorry

end addition_problem_l194_194448


namespace lcm_18_45_l194_194847

theorem lcm_18_45 : Int.lcm 18 45 = 90 :=
by
  -- Prime factorizations
  have h1 : Nat.factors 18 = [2, 3, 3] := by sorry
  have h2 : Nat.factors 45 = [3, 3, 5] := by sorry
  
  -- Calculate LCM
  rw [←Int.lcm_def, Nat.factors_mul, List.perm.ext']
  apply List.Permutation.sublist
  sorry

end lcm_18_45_l194_194847


namespace constant_term_in_binomial_expansion_max_coef_sixth_term_l194_194121

theorem constant_term_in_binomial_expansion_max_coef_sixth_term 
  (n : ℕ) (h : n = 10) : 
  (∃ C : ℕ → ℕ → ℕ, C 10 2 * (Nat.sqrt 2) ^ 8 = 720) :=
sorry

end constant_term_in_binomial_expansion_max_coef_sixth_term_l194_194121


namespace cos_double_angle_value_l194_194438

theorem cos_double_angle_value (θ : ℝ) (h : Real.sin (Real.pi / 2 + θ) = 3 / 5) : 
  Real.cos (2 * θ) = -7 / 25 :=
by
  sorry

end cos_double_angle_value_l194_194438


namespace assign_grades_l194_194785

def is_not_first_grader : Type := sorry
def one_year_older (misha dima : Type) : Prop := sorry
def different_streets (vasya ivanov : Type) : Prop := sorry
def neighbors (boris orlov : Type) : Prop := sorry
def met_one_year_ago (krylov petrov : Type) : Prop := sorry
def gave_textbook_last_year (vasya boris : Type) : Prop := sorry

theorem assign_grades 
  (name : Type) 
  (surname : Type) 
  (grade : Type) 
  (Dima Misha Boris Vasya : name) 
  (Ivanov Krylov Petrov Orlov : surname)
  (first second third fourth : grade)
  (h1 : ¬is_not_first_grader Boris)
  (h2 : different_streets Vasya Ivanov)
  (h3 : one_year_older Misha Dima)
  (h4 : neighbors Boris Orlov)
  (h5 : met_one_year_ago Krylov Petrov)
  (h6 : gave_textbook_last_year Vasya Boris) : 
  (Dima, Ivanov, first) ∧
  (Misha, Krylov, second) ∧
  (Boris, Petrov, third) ∧
  (Vasya, Orlov, fourth) :=
sorry

end assign_grades_l194_194785


namespace total_masks_correct_l194_194974

-- Define the conditions
def boxes := 18
def capacity_per_box := 15
def deficiency_per_box := 3
def masks_per_box := capacity_per_box - deficiency_per_box
def total_masks := boxes * masks_per_box

-- The theorem statement we need to prove
theorem total_masks_correct : total_masks = 216 := by
  unfold total_masks boxes masks_per_box capacity_per_box deficiency_per_box
  sorry

end total_masks_correct_l194_194974


namespace harry_terry_difference_l194_194915

theorem harry_terry_difference :
  let H := 12 - (3 * 4)
  let T := 12 - (3 * 4) -- Correcting Terry's mistake
  H - T = 0 := by
  sorry

end harry_terry_difference_l194_194915


namespace sin_45_eq_one_div_sqrt_two_l194_194315

theorem sin_45_eq_one_div_sqrt_two
  (Q : ℝ × ℝ)
  (h1 : Q = (real.cos (real.pi / 4), real.sin (real.pi / 4)))
  (h2 : Q.2 = real.sin (real.pi / 4)) :
  real.sin (real.pi / 4) = 1 / real.sqrt 2 := 
sorry

end sin_45_eq_one_div_sqrt_two_l194_194315


namespace width_of_room_l194_194965

-- Define the givens
def length_of_room : ℝ := 5.5
def total_cost : ℝ := 20625
def rate_per_sq_meter : ℝ := 1000

-- Define the required proof statement
theorem width_of_room : (total_cost / rate_per_sq_meter) / length_of_room = 3.75 :=
by
  sorry

end width_of_room_l194_194965


namespace value_of_m_l194_194808

theorem value_of_m
  (m : ℤ)
  (h1 : ∃ p : ℕ → ℝ, p 4 = 1/3 ∧ p 1 = -(m + 4) ∧ p 0 = -11 ∧ (∀ (n : ℕ), (n ≠ 4 ∧ n ≠ 1 ∧ n ≠ 0) → p n = 0) ∧ 1 ≤ p 4 + p 1 + p 0) :
  m = 4 :=
  sorry

end value_of_m_l194_194808


namespace arithmetic_sequence_root_sum_l194_194935

theorem arithmetic_sequence_root_sum (a : ℕ → ℝ) (h_arith : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) 
    (h_roots : (a 3) * (a 8) + 3 * (a 3) + 3 * (a 8) - 18 = 0) : a 5 + a 6 = 3 := by
  sorry

end arithmetic_sequence_root_sum_l194_194935


namespace mini_marshmallows_count_l194_194467

theorem mini_marshmallows_count (total_marshmallows large_marshmallows : ℕ) (h1 : total_marshmallows = 18) (h2 : large_marshmallows = 8) :
  total_marshmallows - large_marshmallows = 10 :=
by 
  sorry

end mini_marshmallows_count_l194_194467


namespace quadratic_vertex_position_l194_194743

theorem quadratic_vertex_position (a p q m : ℝ) (ha : 0 < a) (hpq : p < q) (hA : p = a * (-1 - m)^2) (hB : q = a * (3 - m)^2) : m ≠ 2 :=
by
  sorry

end quadratic_vertex_position_l194_194743


namespace power_simplification_l194_194840

theorem power_simplification :
  (1 / ((-5) ^ 4) ^ 2) * (-5) ^ 9 = -5 :=
by 
  sorry

end power_simplification_l194_194840


namespace total_bushels_needed_l194_194037

def cows := 5
def sheep := 4
def chickens := 8
def pigs := 6
def horses := 2

def cow_bushels := 3.5
def sheep_bushels := 1.75
def chicken_bushels := 1.25
def pig_bushels := 4.5
def horse_bushels := 5.75

theorem total_bushels_needed
  (cows : ℕ) (sheep : ℕ) (chickens : ℕ) (pigs : ℕ) (horses : ℕ)
  (cow_bushels: ℝ) (sheep_bushels: ℝ) (chicken_bushels: ℝ) (pig_bushels: ℝ) (horse_bushels: ℝ) :
  cows * cow_bushels + sheep * sheep_bushels + chickens * chicken_bushels + pigs * pig_bushels + horses * horse_bushels = 73 :=
by
  -- Skipping the proof
  sorry

end total_bushels_needed_l194_194037


namespace irreducible_fraction_l194_194633

theorem irreducible_fraction (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
  sorry

end irreducible_fraction_l194_194633


namespace oil_leakage_problem_l194_194493

theorem oil_leakage_problem :
    let l_A := 25  -- Leakage rate of Pipe A (gallons/hour)
    let l_B := 37  -- Leakage rate of Pipe B (gallons/hour)
    let l_C := 55  -- Leakage rate of Pipe C (gallons/hour)
    let l_D := 41  -- Leakage rate of Pipe D (gallons/hour)
    let l_E := 30  -- Leakage rate of Pipe E (gallons/hour)

    let t_A := 10  -- Time taken to fix Pipe A (hours)
    let t_B := 7   -- Time taken to fix Pipe B (hours)
    let t_C := 12  -- Time taken to fix Pipe C (hours)
    let t_D := 9   -- Time taken to fix Pipe D (hours)
    let t_E := 14  -- Time taken to fix Pipe E (hours)

    let leak_A := l_A * t_A  -- Total leaked from Pipe A (gallons)
    let leak_B := l_B * t_B  -- Total leaked from Pipe B (gallons)
    let leak_C := l_C * t_C  -- Total leaked from Pipe C (gallons)
    let leak_D := l_D * t_D  -- Total leaked from Pipe D (gallons)
    let leak_E := l_E * t_E  -- Total leaked from Pipe E (gallons)
  
    let overall_total := leak_A + leak_B + leak_C + leak_D + leak_E
  
    leak_A = 250 ∧
    leak_B = 259 ∧
    leak_C = 660 ∧
    leak_D = 369 ∧
    leak_E = 420 ∧
    overall_total = 1958 :=
by
    sorry

end oil_leakage_problem_l194_194493


namespace number_of_balls_selected_is_three_l194_194265

-- Definitions of conditions
def total_balls : ℕ := 100
def odd_balls_selected : ℕ := 2
def even_balls_selected : ℕ := 1
def probability_first_ball_odd : ℚ := 2 / 3

-- The number of balls selected
def balls_selected := odd_balls_selected + even_balls_selected

-- Statement of the proof problem
theorem number_of_balls_selected_is_three 
(h1 : total_balls = 100)
(h2 : odd_balls_selected = 2)
(h3 : even_balls_selected = 1)
(h4 : probability_first_ball_odd = 2 / 3) :
  balls_selected = 3 :=
sorry

end number_of_balls_selected_is_three_l194_194265


namespace intersection_points_count_l194_194560

theorem intersection_points_count : 
  ∃ (x1 y1 x2 y2 : ℝ), 
  (x1 - ⌊x1⌋)^2 + (y1 - 1)^2 = x1 - ⌊x1⌋ ∧ 
  y1 = 1/5 * x1 + 1 ∧ 
  (x2 - ⌊x2⌋)^2 + (y2 - 1)^2 = x2 - ⌊x2⌋ ∧ 
  y2 = 1/5 * x2 + 1 ∧ 
  (x1, y1) ≠ (x2, y2) :=
sorry

end intersection_points_count_l194_194560


namespace polygons_sides_l194_194813

theorem polygons_sides 
  (n1 n2 : ℕ)
  (h1 : n1 * (n1 - 3) / 2 + n2 * (n2 - 3) / 2 = 158)
  (h2 : 180 * (n1 + n2 - 4) = 4320) :
  (n1 = 16 ∧ n2 = 12) ∨ (n1 = 12 ∧ n2 = 16) :=
sorry

end polygons_sides_l194_194813


namespace contains_zero_if_sum_is_111111_l194_194031

variables (x y : ℕ)
variables (digits_x digits_y : Fin 5 → ℕ)

-- Conditions
def is_five_digit (n : ℕ) : Prop := 
  10000 ≤ n ∧ n < 100000

def differs_by_two_digit_swap (dx dy : Fin 5 → ℕ) : Prop :=
  ∃ i j : Fin 5, i ≠ j ∧ 
  (∀ k : Fin 5, (k = i → dy k = dx j) ∧ 
                (k = j → dy k = dx i) ∧ 
                (k ≠ i ∧ k ≠ j → dy k = dx k))

theorem contains_zero_if_sum_is_111111 
  (hx : is_five_digit x) (hy : is_five_digit y)
  (h_digits_x : digits_x = fun i : Fin 5 => (x / 10^(4-i) % 10))
  (h_digits_y : digits_y = fun i : Fin 5 => (y / 10^(4-i) % 10))
  (h_swap : differs_by_two_digit_swap digits_x digits_y)
  (h_sum : x + y = 111111) :
  ∃ i : Fin 5, digits_x i = 0 :=
begin
  -- Proof omitted
  sorry,
end

end contains_zero_if_sum_is_111111_l194_194031


namespace solution_set_of_inequality_l194_194402

theorem solution_set_of_inequality :
  {x : ℝ | 1 / x < 1 / 3} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 3} :=
by
  sorry

end solution_set_of_inequality_l194_194402


namespace initial_roses_in_vase_l194_194238

/-- 
There were some roses in a vase. Mary cut roses from her flower garden 
and put 16 more roses in the vase. There are now 22 roses in the vase.
Prove that the initial number of roses in the vase was 6. 
-/
theorem initial_roses_in_vase (initial_roses added_roses current_roses : ℕ) 
  (h_add : added_roses = 16) 
  (h_current : current_roses = 22) 
  (h_current_eq : current_roses = initial_roses + added_roses) : 
  initial_roses = 6 := 
by
  subst h_add
  subst h_current
  linarith

end initial_roses_in_vase_l194_194238


namespace Collin_total_petals_l194_194881

-- Definitions of the conditions
def initial_flowers_Collin : ℕ := 25
def flowers_Ingrid : ℕ := 33
def petals_per_flower : ℕ := 4
def third_of_flowers_Ingrid : ℕ := flowers_Ingrid / 3

-- Total number of flowers Collin has after receiving from Ingrid
def total_flowers_Collin : ℕ := initial_flowers_Collin + third_of_flowers_Ingrid

-- Total number of petals Collin has
def total_petals_Collin : ℕ := total_flowers_Collin * petals_per_flower

-- The theorem to be proved
theorem Collin_total_petals : total_petals_Collin = 144 := by
  -- Proof goes here
  sorry

end Collin_total_petals_l194_194881


namespace senior_junior_ratio_l194_194286

variable (S J : ℕ) (k : ℕ)

theorem senior_junior_ratio (h1 : S = k * J) 
                           (h2 : (1/8 : ℚ) * S + (3/4 : ℚ) * J = (1/3 : ℚ) * (S + J)) : 
                           k = 2 :=
by
  sorry

end senior_junior_ratio_l194_194286


namespace probability_of_drawing_three_white_balls_l194_194860

theorem probability_of_drawing_three_white_balls
  (total_balls white_balls black_balls: ℕ)
  (h_total: total_balls = 15)
  (h_white: white_balls = 7)
  (h_black: black_balls = 8)
  (draws: ℕ)
  (h_draws: draws = 3) :
  (Nat.choose white_balls draws / Nat.choose total_balls draws) = (7 / 91) :=
by sorry

end probability_of_drawing_three_white_balls_l194_194860


namespace hours_in_one_year_l194_194817

/-- Given that there are 24 hours in a day and 365 days in a year,
    prove that there are 8760 hours in one year. -/
theorem hours_in_one_year (hours_per_day : ℕ) (days_per_year : ℕ) (hours_value : ℕ := 8760) : hours_per_day = 24 → days_per_year = 365 → hours_per_day * days_per_year = hours_value :=
by
  intros
  sorry

end hours_in_one_year_l194_194817


namespace largest_final_digit_l194_194158

theorem largest_final_digit (seq : Fin 1002 → Fin 10) 
  (h1 : seq 0 = 2) 
  (h2 : ∀ n : Fin 1001, (17 ∣ (10 * seq n + seq (n + 1))) ∨ (29 ∣ (10 * seq n + seq (n + 1)))) : 
  seq 1001 = 5 :=
sorry

end largest_final_digit_l194_194158


namespace last_digit_2008_pow_2005_l194_194845

def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_2008_pow_2005 : last_digit (2008 ^ 2005) = 8 :=
by
  sorry

end last_digit_2008_pow_2005_l194_194845


namespace probability_of_valid_quadrilateral_l194_194479

-- Define a regular octagon
def regular_octagon_sides : ℕ := 8

-- Total number of ways to choose 4 sides from 8 sides
def total_ways_choose_four_sides : ℕ := Nat.choose 8 4

-- Number of ways to choose 4 adjacent sides (invalid)
def invalid_adjacent_ways : ℕ := 8

-- Number of ways to choose 4 sides with 3 adjacent unchosen sides (invalid)
def invalid_three_adjacent_unchosen_ways : ℕ := 8 * 3

-- Total number of invalid ways
def total_invalid_ways : ℕ := invalid_adjacent_ways + invalid_three_adjacent_unchosen_ways

-- Total number of valid ways
def total_valid_ways : ℕ := total_ways_choose_four_sides - total_invalid_ways

-- Probability of forming a quadrilateral that contains the octagon
def probability_valid_quadrilateral : ℚ :=
  (total_valid_ways : ℚ) / (total_ways_choose_four_sides : ℚ)

-- Theorem statement
theorem probability_of_valid_quadrilateral :
  probability_valid_quadrilateral = 19 / 35 :=
by
  sorry

end probability_of_valid_quadrilateral_l194_194479


namespace discriminant_zero_geometric_progression_l194_194380

variable (a b c : ℝ)

theorem discriminant_zero_geometric_progression
  (h : b^2 = 4 * a * c) : (b / (2 * a)) = (2 * c / b) :=
by
  sorry

end discriminant_zero_geometric_progression_l194_194380


namespace range_a_for_false_proposition_l194_194927

theorem range_a_for_false_proposition :
  {a : ℝ | ¬ ∃ x : ℝ, x^2 + (1 - a) * x < 0} = {1} :=
sorry

end range_a_for_false_proposition_l194_194927


namespace students_scoring_above_115_l194_194152

noncomputable def number_of_students : ℕ := 50
noncomputable def mean : ℝ := 105
noncomputable def variance : ℝ := 10^2
noncomputable def distribution (x : ℝ) : ℝ := (1 / (Math.sqrt (2 * Math.pi * variance))) * Math.exp (-(x - mean)^2 / (2 * variance))
noncomputable def prob_range_95_105 : ℝ := 0.32

theorem students_scoring_above_115 : 
  (number_of_students * (1 - 2 * prob_range_95_105)) / 2 = 9 :=
by
  sorry

end students_scoring_above_115_l194_194152


namespace hyeongjun_older_sister_age_l194_194431

-- Define the ages of Hyeongjun and his older sister
variables (H S : ℕ)

-- Conditions
def age_gap := S = H + 2
def sum_of_ages := H + S = 26

-- Theorem stating that the older sister's age is 14
theorem hyeongjun_older_sister_age (H S : ℕ) (h1 : age_gap H S) (h2 : sum_of_ages H S) : S = 14 := 
by 
  sorry

end hyeongjun_older_sister_age_l194_194431


namespace sin_45_eq_sqrt2_div_2_l194_194359

theorem sin_45_eq_sqrt2_div_2 : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by sorry

end sin_45_eq_sqrt2_div_2_l194_194359


namespace problem1_problem2_l194_194147

-- Problem 1: Simplify the calculation: 6.9^2 + 6.2 * 6.9 + 3.1^2
theorem problem1 : 6.9^2 + 6.2 * 6.9 + 3.1^2 = 100 := 
by
  sorry

-- Problem 2: Simplify and find the value of the expression with given conditions
theorem problem2 (a b : ℝ) (h1 : a = 1) (h2 : b = 0.5) :
  (a^2 * b^3 + 2 * a^3 * b) / (2 * a * b) - (a + 2 * b) * (a - 2 * b) = 9 / 8 :=
by
  sorry

end problem1_problem2_l194_194147


namespace percentage_paid_l194_194647

/-- 
Given the marked price is 80% of the suggested retail price,
and Alice paid 60% of the marked price,
prove that the percentage of the suggested retail price Alice paid is 48%.
-/
theorem percentage_paid (P : ℝ) (MP : ℝ) (price_paid : ℝ)
  (h1 : MP = 0.80 * P)
  (h2 : price_paid = 0.60 * MP) :
  (price_paid / P) * 100 = 48 := 
sorry

end percentage_paid_l194_194647


namespace groups_needed_l194_194267

theorem groups_needed (h_camper_count : 36 > 0) (h_group_limit : 12 > 0) : 
  ∃ x : ℕ, x = 36 / 12 ∧ x = 3 := by
  sorry

end groups_needed_l194_194267


namespace sum_of_reciprocals_of_squares_roots_eq_14_3125_l194_194964

theorem sum_of_reciprocals_of_squares_roots_eq_14_3125
  (α β γ : ℝ)
  (h1 : α + β + γ = 15)
  (h2 : α * β + β * γ + γ * α = 26)
  (h3 : α * β * γ = -8) :
  (1 / α^2) + (1 / β^2) + (1 / γ^2) = 14.3125 := 
by
  sorry

end sum_of_reciprocals_of_squares_roots_eq_14_3125_l194_194964


namespace evaluate_pow_l194_194704

theorem evaluate_pow : (-64 : ℝ)^(4/3) = 256 := 
by
  sorry

end evaluate_pow_l194_194704


namespace knights_and_liars_l194_194626

theorem knights_and_liars (N : ℕ) (hN : N = 30)
  (sees : Π (I : fin N), finset (fin N))
  (h_sees : ∀ (I : fin N), sees I = (finset.univ.erase I).erase (I - 1)).erase (I + 1))
  (statement : Π (I : fin N), Prop)
  (h_statement : ∀ (I : fin N), statement I = ∀ J ∈ sees I, ¬ statement J) :
  ∃ K L : ℕ, K + L = 30 ∧ K = 2 ∧ L = 28 :=
by {
  use 2,
  use 28,
  split,
  exact hN,
  split,
  refl,
  refl
}

end knights_and_liars_l194_194626


namespace time_3339_minutes_after_midnight_l194_194986

def minutes_since_midnight (minutes : ℕ) : ℕ × ℕ :=
  let hours := minutes / 60
  let remaining_minutes := minutes % 60
  (hours, remaining_minutes)

def time_after_midnight (start_time : ℕ × ℕ) (hours : ℕ) (minutes : ℕ) : ℕ × ℕ :=
  let (start_hours, start_minutes) := start_time
  let total_minutes := start_hours * 60 + start_minutes + hours * 60 + minutes
  let end_hours := total_minutes / 60
  let end_minutes := total_minutes % 60
  (end_hours, end_minutes)

theorem time_3339_minutes_after_midnight :
  time_after_midnight (0, 0) 55 39 = (7, 39) :=
by
  sorry

end time_3339_minutes_after_midnight_l194_194986


namespace sin_45_deg_eq_one_div_sqrt_two_l194_194330

def unit_circle_radius : ℝ := 1

def forty_five_degrees_in_radians : ℝ := (Real.pi / 4)

def cos_45 : ℝ := Real.cos forty_five_degrees_in_radians

def sin_45 : ℝ := Real.sin forty_five_degrees_in_radians

theorem sin_45_deg_eq_one_div_sqrt_two : 
  sin_45 = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_deg_eq_one_div_sqrt_two_l194_194330


namespace find_m_from_split_l194_194411

theorem find_m_from_split (m : ℕ) (h1 : m > 1) (h2 : m^2 - m + 1 = 211) : True :=
by
  -- This theorem states that under the conditions that m is a positive integer greater than 1
  -- and m^2 - m + 1 = 211, there exists an integer value for m that satisfies these conditions.
  trivial

end find_m_from_split_l194_194411


namespace total_boys_eq_350_l194_194255

variable (Total : ℕ)
variable (SchoolA : ℕ)
variable (NotScience : ℕ)

axiom h1 : SchoolA = 20 * Total / 100
axiom h2 : NotScience = 70 * SchoolA / 100
axiom h3 : NotScience = 49

theorem total_boys_eq_350 : Total = 350 :=
by
  sorry

end total_boys_eq_350_l194_194255


namespace problem_1_problem_2_problem_3_l194_194525

-- Definitions of assumptions and conditions.
structure Problem :=
  (boys : ℕ) -- number of boys
  (girls : ℕ) -- number of girls
  (subjects : ℕ) -- number of subjects
  (boyA_not_math : Prop) -- Boy A can't be a representative of the mathematics course
  (girlB_chinese : Prop) -- Girl B must be a representative of the Chinese language course

-- Problem 1: Calculate the number of ways satisfying condition (1)
theorem problem_1 (p : Problem) (h1 : p.girls < p.boys) :
  ∃ n : ℕ, n = 5520 := sorry

-- Problem 2: Calculate the number of ways satisfying condition (2)
theorem problem_2 (p : Problem) (h1 : p.boys ≥ 1) (h2 : p.boyA_not_math) :
  ∃ n : ℕ, n = 3360 := sorry

-- Problem 3: Calculate the number of ways satisfying condition (3)
theorem problem_3 (p : Problem) (h1 : p.boys ≥ 1) (h2 : p.boyA_not_math) (h3 : p.girlB_chinese) :
  ∃ n : ℕ, n = 360 := sorry

end problem_1_problem_2_problem_3_l194_194525


namespace vector_dot_product_identity_l194_194750

-- Define the vectors a, b, and c in ℝ²
def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (-1, 1)
def c : ℝ × ℝ := (-3, 1)

-- Define vector addition and dot product in ℝ²
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Prove that c · (a + b) = 9
theorem vector_dot_product_identity : dot_product c (vector_add a b) = 9 := 
by 
sorry

end vector_dot_product_identity_l194_194750


namespace KHSO4_formed_l194_194401

-- Define the reaction condition and result using moles
def KOH_moles : ℕ := 2
def H2SO4_moles : ℕ := 2

-- The balanced chemical reaction in terms of moles
-- 1 mole of KOH reacts with 1 mole of H2SO4 to produce 
-- 1 mole of KHSO4
def react (koh : ℕ) (h2so4 : ℕ) : ℕ := 
  -- stoichiometry 1:1 ratio of KOH and H2SO4 to KHSO4
  if koh ≤ h2so4 then koh else h2so4

-- The proof statement that verifies the expected number of moles of KHSO4
theorem KHSO4_formed (koh : ℕ) (h2so4 : ℕ) (hrs : react koh h2so4 = koh) : 
  koh = KOH_moles → h2so4 = H2SO4_moles → react koh h2so4 = 2 := 
by
  intros 
  sorry

end KHSO4_formed_l194_194401


namespace find_equation_of_tangent_line_perpendicular_l194_194725

noncomputable def tangent_line_perpendicular_to_curve (a b : ℝ) : Prop :=
  (∃ (P : ℝ × ℝ), P = (-1, -3) ∧ 2 * P.1 - 6 * P.2 + 1 = 0 ∧ P.2 = P.1^3 + 5 * P.1^2 - 5) ∧
  (-3) = 3 * (-1)^2 + 6 * (-1)

theorem find_equation_of_tangent_line_perpendicular :
  tangent_line_perpendicular_to_curve (-1) (-3) →
  ∀ x y : ℝ, 3 * x + y + 6 = 0 :=
by
  sorry

end find_equation_of_tangent_line_perpendicular_l194_194725


namespace part1_3_neg5_is_pair_part1_neg2_4_is_not_pair_part2_find_n_part3_find_k_l194_194067

def is_equation_number_pair (a b : ℝ) : Prop :=
  ∀ x : ℝ, (x = 1 / (a + b) ↔ a / x + 1 = b)

theorem part1_3_neg5_is_pair : is_equation_number_pair 3 (-5) :=
sorry

theorem part1_neg2_4_is_not_pair : ¬ is_equation_number_pair (-2) 4 :=
sorry

theorem part2_find_n (n : ℝ) : is_equation_number_pair n (3 - n) ↔ n = 1 / 2 :=
sorry

theorem part3_find_k (m k : ℝ) (hm : m ≠ -1) (hm0 : m ≠ 0) (hk1 : k ≠ 1) :
  is_equation_number_pair (m - k) k → k = (m^2 + 1) / (m + 1) :=
sorry

end part1_3_neg5_is_pair_part1_neg2_4_is_not_pair_part2_find_n_part3_find_k_l194_194067


namespace sequence_geometric_and_sum_l194_194738

variables {S : ℕ → ℝ} (a1 : S 1 = 1)
variable (n : ℕ)
def a := (S (n+1) - 2 * S n, S n)
def b := (2, n)

/-- Prove that the sequence {S n / n} is a geometric sequence 
with first term 1 and common ratio 2, and find the sum of the first 
n terms of the sequence {S n} -/
theorem sequence_geometric_and_sum {S : ℕ → ℝ} (a1 : S 1 = 1)
  (n : ℕ)
  (parallel : ∀ n, n * (S (n + 1) - 2 * S n) = 2 * S n) :
  ∃ r : ℝ, r = 2 ∧ ∃ T : ℕ → ℝ, T n = (n-1)*2^n + 1 :=
by
  sorry

end sequence_geometric_and_sum_l194_194738


namespace sum_of_logs_in_acute_triangle_l194_194604

theorem sum_of_logs_in_acute_triangle (A B C : ℝ)
  (hA : 0 < A ∧ A < π / 2) (hB : 0 < B ∧ B < π / 2) (hC : 0 < C ∧ C < π / 2) 
  (h_triangle : A + B + C = π) :
  (Real.log (Real.sin B) / Real.log (Real.sin A)) +
  (Real.log (Real.sin C) / Real.log (Real.sin B)) +
  (Real.log (Real.sin A) / Real.log (Real.sin C)) ≥ 3 := by
  sorry

end sum_of_logs_in_acute_triangle_l194_194604


namespace binom_150_150_eq_one_l194_194558

theorem binom_150_150_eq_one : nat.choose 150 150 = 1 := by
  sorry

end binom_150_150_eq_one_l194_194558


namespace find_third_number_l194_194924

-- Define the given conditions
def proportion_condition (x y : ℝ) : Prop :=
  (0.75 / x) = (y / 8)

-- The main statement to be proven
theorem find_third_number (x y : ℝ) (hx : x = 1.2) (h_proportion : proportion_condition x y) : y = 5 :=
by
  -- Using the assumptions and the definition provided.
  sorry

end find_third_number_l194_194924


namespace two_buckets_have_40_liters_l194_194021

def liters_in_jug := 5
def jugs_in_bucket := 4
def liters_in_bucket := liters_in_jug * jugs_in_bucket
def buckets := 2

theorem two_buckets_have_40_liters :
  buckets * liters_in_bucket = 40 :=
by
  sorry

end two_buckets_have_40_liters_l194_194021


namespace geometric_difference_l194_194557

def is_geometric_sequence (n : ℕ) : Prop :=
∃ (a b c : ℤ), n = a * 100 + b * 10 + c ∧
a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
(b^2 = a * c) ∧
(b % 2 = 1)

theorem geometric_difference :
  ∃ (n1 n2 : ℕ), is_geometric_sequence n1 ∧ is_geometric_sequence n2 ∧
  n2 > n1 ∧
  n2 - n1 = 220 :=
sorry

end geometric_difference_l194_194557


namespace sin_45_eq_sqrt2_div_2_l194_194318

theorem sin_45_eq_sqrt2_div_2 : Real.sin (π / 4) = Real.sqrt 2 / 2 := 
sorry

end sin_45_eq_sqrt2_div_2_l194_194318


namespace find_vector_c_l194_194050

def angle_equal_coordinates (c : ℝ × ℝ) : Prop :=
  let a : ℝ × ℝ := (1, 0)
  let b : ℝ × ℝ := (1, -Real.sqrt 3)
  let cos_angle_ab (u v : ℝ × ℝ) : ℝ :=
    (u.1 * v.1 + u.2 * v.2) / (Real.sqrt (u.1^2 + u.2^2) * Real.sqrt (v.1^2 + v.2^2))
  cos_angle_ab c a = cos_angle_ab c b

theorem find_vector_c :
  angle_equal_coordinates (Real.sqrt 3, -1) :=
sorry

end find_vector_c_l194_194050


namespace delta_maximum_success_ratio_l194_194906

theorem delta_maximum_success_ratio (x y z w : ℕ) (h1 : 0 < x ∧ x * 5 < y * 3)
    (h2 : 0 < z ∧ z * 5 < w * 3) (h3 : y + w = 600) :
    (x + z) / 600 ≤ 359 / 600 :=
by
  sorry

end delta_maximum_success_ratio_l194_194906


namespace avg_adults_proof_l194_194963

variable (n_total : ℕ) (n_girls : ℕ) (n_boys : ℕ) (n_adults : ℕ)
variable (avg_total : ℕ) (avg_girls : ℕ) (avg_boys : ℕ)

def avg_age_adults (n_total n_girls n_boys n_adults avg_total avg_girls avg_boys : ℕ) : ℕ :=
  let sum_total := n_total * avg_total
  let sum_girls := n_girls * avg_girls
  let sum_boys := n_boys * avg_boys
  let sum_adults := sum_total - sum_girls - sum_boys
  sum_adults / n_adults

theorem avg_adults_proof :
  avg_age_adults 50 25 20 5 21 18 20 = 40 := 
by
  -- Proof will go here
  sorry

end avg_adults_proof_l194_194963


namespace power_simplification_l194_194842

theorem power_simplification :
  (1 / ((-5) ^ 4) ^ 2) * (-5) ^ 9 = -5 :=
by 
  sorry

end power_simplification_l194_194842


namespace sin_45_eq_sqrt_two_over_two_l194_194369

theorem sin_45_eq_sqrt_two_over_two : Real.sin (π / 4) = sqrt 2 / 2 :=
by
  sorry

end sin_45_eq_sqrt_two_over_two_l194_194369


namespace winner_votes_percentage_l194_194936

-- Define the total votes as V
def total_votes (winner_votes : ℕ) (winning_margin : ℕ) : ℕ :=
  winner_votes + (winner_votes - winning_margin)

-- Define the percentage function
def percentage_of_votes (part : ℕ) (total : ℕ) : ℕ :=
  (part * 100) / total

-- Lean statement to prove the result
theorem winner_votes_percentage
  (winner_votes : ℕ)
  (winning_margin : ℕ)
  (H_winner_votes : winner_votes = 550)
  (H_winning_margin : winning_margin = 100) :
  percentage_of_votes winner_votes (total_votes winner_votes winning_margin) = 55 := by
  sorry

end winner_votes_percentage_l194_194936


namespace smallest_solution_x4_50x2_576_eq_0_l194_194139

theorem smallest_solution_x4_50x2_576_eq_0 :
  ∃ x : ℝ, (x^4 - 50*x^2 + 576 = 0) ∧ ∀ y : ℝ, y^4 - 50*y^2 + 576 = 0 → x ≤ y :=
sorry

end smallest_solution_x4_50x2_576_eq_0_l194_194139


namespace boards_nailing_l194_194108

variables {x y a b : ℕ}

theorem boards_nailing (h1 : 2 * x + 3 * y = 87)
                       (h2 : 3 * a + 5 * b = 94) :
                       x + y = 30 ∧ a + b = 30 :=
sorry

end boards_nailing_l194_194108


namespace surface_area_of_cube_edge_8_l194_194505

-- Definition of surface area of a cube
def surface_area_of_cube (edge_length : ℕ) : ℕ :=
  6 * (edge_length * edge_length)

-- Theorem to prove the surface area for a cube with edge length of 8 cm is 384 cm²
theorem surface_area_of_cube_edge_8 : surface_area_of_cube 8 = 384 :=
by
  -- The proof will be inserted here. We use sorry to indicate the missing proof.
  sorry

end surface_area_of_cube_edge_8_l194_194505


namespace sin_45_degree_l194_194303

def Q : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), real.sin (real.pi / 4))
def E : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), 0)
def O : (x:ℝ) × (y:ℝ) := (0,0)
def OQ : ℝ := real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2)

theorem sin_45_degree : ∃ x: ℝ, x = real.sin (real.pi / 4) ∧ x = real.sqrt 2 / 2 :=
by sorry

end sin_45_degree_l194_194303


namespace perpendicular_line_through_point_l194_194123

open Real

theorem perpendicular_line_through_point (B : ℝ × ℝ) (x y : ℝ) (c : ℝ)
  (hB : B = (3, 0)) (h_perpendicular : 2 * x + y - 5 = 0) :
  x - 2 * y + 3 = 0 :=
sorry

end perpendicular_line_through_point_l194_194123


namespace five_a_squared_plus_one_divisible_by_three_l194_194635

theorem five_a_squared_plus_one_divisible_by_three (a : ℤ) (h : a % 3 ≠ 0) : (5 * a^2 + 1) % 3 = 0 :=
sorry

end five_a_squared_plus_one_divisible_by_three_l194_194635


namespace employees_working_abroad_l194_194651

theorem employees_working_abroad
  (total_employees : ℕ)
  (fraction_abroad : ℝ)
  (h_total : total_employees = 450)
  (h_fraction : fraction_abroad = 0.06) :
  total_employees * fraction_abroad = 27 := 
by
  sorry

end employees_working_abroad_l194_194651


namespace squirrel_calories_l194_194878

def rabbits_caught_per_hour := 2
def rabbits_calories := 800
def squirrels_caught_per_hour := 6
def extra_calories_squirrels := 200

theorem squirrel_calories : 
  ∀ (S : ℕ), 
  (6 * S = (2 * 800) + 200) → S = 300 := by
  intros S h
  sorry

end squirrel_calories_l194_194878


namespace sin_45_deg_l194_194296

theorem sin_45_deg : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by 
  -- placeholder for the actual proof
  sorry

end sin_45_deg_l194_194296


namespace sphere_radius_geometric_mean_l194_194956

-- Definitions from conditions
variable (r R ρ : ℝ)
variable (r_nonneg : 0 ≤ r)
variable (R_relation : R = 3 * r)
variable (ρ_relation : ρ = Real.sqrt 3 * r)

-- Problem statement
theorem sphere_radius_geometric_mean (tetrahedron : Prop):
  ρ * ρ = R * r :=
by
  sorry

end sphere_radius_geometric_mean_l194_194956


namespace sin_45_eq_sqrt_two_over_two_l194_194367

theorem sin_45_eq_sqrt_two_over_two : Real.sin (π / 4) = sqrt 2 / 2 :=
by
  sorry

end sin_45_eq_sqrt_two_over_two_l194_194367


namespace sum_of_xi_l194_194734

theorem sum_of_xi {x1 x2 x3 x4 : ℝ} (h1: (x1 - 3) * Real.sin (π * x1) = 1)
  (h2: (x2 - 3) * Real.sin (π * x2) = 1)
  (h3: (x3 - 3) * Real.sin (π * x3) = 1)
  (h4: (x4 - 3) * Real.sin (π * x4) = 1)
  (hx1 : x1 > 0) (hx2: x2 > 0) (hx3 : x3 > 0) (hx4: x4 > 0) :
  x1 + x2 + x3 + x4 = 12 :=
by
  sorry

end sum_of_xi_l194_194734


namespace find_value_of_question_mark_l194_194016

theorem find_value_of_question_mark (q : ℕ) : q * 40 = 173 * 240 → q = 1036 :=
by
  intro h
  sorry

end find_value_of_question_mark_l194_194016


namespace smallest_enclosing_sphere_radius_l194_194690

theorem smallest_enclosing_sphere_radius :
  let r := 2
  let d := 4 * Real.sqrt 3
  let total_diameter := d + 2*r
  let radius_enclosing_sphere := total_diameter / 2
  radius_enclosing_sphere = 2 + 2 * Real.sqrt 3 := by
  -- Define the radius of the smaller spheres
  let r : ℝ := 2
  -- Space diagonal of the cube which is 4√3 where 4 is the side length
  let d : ℝ := 4 * Real.sqrt 3
  -- Total diameter of the sphere containing the cube (space diagonal + 2 radius of one sphere)
  let total_diameter : ℝ := d + 2 * r
  -- Radius of the enclosing sphere
  let radius_enclosing_sphere : ℝ := total_diameter / 2
  -- We need to prove that this radius equals 2 + 2√3
  sorry

end smallest_enclosing_sphere_radius_l194_194690


namespace solve_equation_l194_194234

theorem solve_equation (x : ℝ) : (x - 2) ^ 2 = 9 ↔ x = 5 ∨ x = -1 :=
by
  sorry -- Proof is skipped

end solve_equation_l194_194234


namespace relationship_in_size_l194_194731

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.sqrt 2.1
noncomputable def c : ℝ := Real.log (1.5) / Real.log (2)

theorem relationship_in_size : b > a ∧ a > c := by
  sorry

end relationship_in_size_l194_194731


namespace intersecting_lines_l194_194177

theorem intersecting_lines (p : ℝ) :
    (∃ x y : ℝ, y = 3 * x - 6 ∧ y = -4 * x + 8 ∧ y = 7 * x + p) ↔ p = -14 :=
by {
    sorry
}

end intersecting_lines_l194_194177


namespace hyperbola_eccentricity_sqrt_five_l194_194747

/-- Given a hyperbola with the equation x^2/a^2 - y^2/b^2 = 1 where a > 0 and b > 0,
and its focus lies symmetrically with respect to the asymptote lines and on the hyperbola,
proves that the eccentricity of the hyperbola is sqrt(5). -/
theorem hyperbola_eccentricity_sqrt_five 
  (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) 
  (c : ℝ) (h_focus : c^2 = 5 * a^2) : 
  (c / a = Real.sqrt 5) := sorry

end hyperbola_eccentricity_sqrt_five_l194_194747


namespace min_triangle_perimeter_l194_194055

/-- Given a point (a, b) with 0 < b < a,
    determine the minimum perimeter of a triangle with one vertex at (a, b),
    one on the x-axis, and one on the line y = x. 
    The minimum perimeter is √(2(a^2 + b^2)).
-/
theorem min_triangle_perimeter (a b : ℝ) (h : 0 < b ∧ b < a) 
  : ∃ c d : ℝ, c^2 + d^2 = 2 * (a^2 + b^2) := sorry

end min_triangle_perimeter_l194_194055


namespace circumference_of_jack_head_l194_194454

theorem circumference_of_jack_head (J C : ℝ) (h1 : (2 / 3) * C = 10) (h2 : (1 / 2) * J + 9 = 15) :
  J = 12 :=
by
  sorry

end circumference_of_jack_head_l194_194454


namespace evaluate_expression_l194_194650

theorem evaluate_expression : (-2)^3 - (-3)^2 = -17 :=
by sorry

end evaluate_expression_l194_194650


namespace tangent_lines_count_l194_194745

def f (x : ℝ) : ℝ := x^3

theorem tangent_lines_count :
  (∃ x : ℝ, deriv f x = 3) ∧ 
  (∃ y : ℝ, deriv f y = 3 ∧ y ≠ x) := 
by
  -- Since f(x) = x^3, its derivative is f'(x) = 3x^2
  -- We need to solve 3x^2 = 3
  -- Therefore, x^2 = 1 and x = ±1
  -- Thus, there are two tangent lines
  sorry

end tangent_lines_count_l194_194745


namespace maximum_value_of_f_l194_194392

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin x + 12 * Real.cos x

theorem maximum_value_of_f : ∃ x : ℝ, f x = 13 :=
by 
  sorry

end maximum_value_of_f_l194_194392


namespace ratio_surface_area_l194_194576

open Real

theorem ratio_surface_area (R a : ℝ) 
  (h1 : 4 * R^2 = 6 * a^2) 
  (H : R = (sqrt 6 / 2) * a) : 
  3 * π * R^2 / (6 * a^2) = 3 * π / 4 :=
by {
  sorry
}

end ratio_surface_area_l194_194576


namespace maria_ends_up_with_22_towels_l194_194006

-- Define the number of green towels Maria bought
def green_towels : Nat := 35

-- Define the number of white towels Maria bought
def white_towels : Nat := 21

-- Define the number of towels Maria gave to her mother
def given_towels : Nat := 34

-- Total towels Maria initially bought
def total_towels := green_towels + white_towels

-- Towels Maria ended up with
def remaining_towels := total_towels - given_towels

theorem maria_ends_up_with_22_towels :
  remaining_towels = 22 :=
by
  sorry

end maria_ends_up_with_22_towels_l194_194006


namespace inner_ring_speed_minimum_train_distribution_l194_194492

theorem inner_ring_speed_minimum
  (l_inner : ℝ) (num_trains_inner : ℕ) (max_wait_inner : ℝ) (speed_min : ℝ) :
  l_inner = 30 →
  num_trains_inner = 9 →
  max_wait_inner = 10 →
  speed_min = 20 :=
by 
  sorry

theorem train_distribution
  (l_inner : ℝ) (speed_inner : ℝ) (speed_outer : ℝ) (total_trains : ℕ) (max_wait_diff : ℝ) (trains_inner : ℕ) (trains_outer : ℕ) :
  l_inner = 30 →
  speed_inner = 25 →
  speed_outer = 30 →
  total_trains = 18 →
  max_wait_diff = 1 →
  trains_inner = 10 →
  trains_outer = 8 :=
by 
  sorry

end inner_ring_speed_minimum_train_distribution_l194_194492


namespace sin_45_deg_eq_one_div_sqrt_two_l194_194334

def unit_circle_radius : ℝ := 1

def forty_five_degrees_in_radians : ℝ := (Real.pi / 4)

def cos_45 : ℝ := Real.cos forty_five_degrees_in_radians

def sin_45 : ℝ := Real.sin forty_five_degrees_in_radians

theorem sin_45_deg_eq_one_div_sqrt_two : 
  sin_45 = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_deg_eq_one_div_sqrt_two_l194_194334


namespace proof_set_intersection_l194_194748

noncomputable def U := ℝ
noncomputable def M := {x : ℝ | 0 ≤ x ∧ x < 5}
noncomputable def N := {x : ℝ | x ≥ 2}
noncomputable def compl_U_N := {x : ℝ | x < 2}
noncomputable def intersection := { x : ℝ | 0 ≤ x ∧ x < 2 }

theorem proof_set_intersection : ((compl_U_N ∩ M) = {x : ℝ | 0 ≤ x ∧ x < 2}) :=
by
  sorry

end proof_set_intersection_l194_194748


namespace three_digit_numbers_with_units_at_least_three_times_tens_l194_194073

open Nat

noncomputable def valid_units_and_tens_digit (tens units : ℕ) : Prop :=
  (tens = 0 ∧ units ∈ range 10) ∨
  (tens = 1 ∧ units ∈ range' 3 10) ∨
  (tens = 2 ∧ units ∈ range' 6 10)

noncomputable def valid_hundreds_digit (hundreds : ℕ) : Prop :=
  hundreds ∈ range' 1 10

theorem three_digit_numbers_with_units_at_least_three_times_tens :
  let total_numbers := 189
  ∃ total_numbers : ℕ, total_numbers = ∑ h in range' 1 10, ∑ t in range 10, ∑ u in range 10, if valid_hundreds_digit h ∧ valid_units_and_tens_digit t u then 1 else 0 :=
begin
  sorry
end

end three_digit_numbers_with_units_at_least_three_times_tens_l194_194073


namespace power_evaluation_l194_194718

theorem power_evaluation : (-64 : ℝ)^(4/3) = 256 :=
by 
  have step1 : (-64 : ℝ)^(4/3) = ((-4 : ℝ)^3)^(4/3),
  { sorry },
  have step2 : ((-4 : ℝ)^3)^(4/3) = (-4 : ℝ)^4,
  { sorry },
  have step3 : (-4 : ℝ)^4 = 256,
  { sorry },
  rwa [step1, step2, step3]

end power_evaluation_l194_194718


namespace square_side_lengths_l194_194277

theorem square_side_lengths (x y : ℝ) (h1 : x + y = 20) (h2 : x^2 - y^2 = 120) :
  (x = 13 ∧ y = 7) ∨ (x = 7 ∧ y = 13) :=
by {
  -- skip proof
  sorry
}

end square_side_lengths_l194_194277


namespace planes_parallel_or_intersect_l194_194911

variables {Plane : Type} {Line : Type}
variables (α β : Plane) (a b : Line)

-- Conditions
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def not_parallel (l1 l2 : Line) : Prop := sorry

-- Given conditions
axiom h₁ : line_in_plane a α
axiom h₂ : line_in_plane b β
axiom h₃ : not_parallel a b

-- The theorem statement
theorem planes_parallel_or_intersect : (exists l : Line, line_in_plane l α ∧ line_in_plane l β) ∨ (α = β) :=
sorry

end planes_parallel_or_intersect_l194_194911


namespace sin_45_deg_eq_one_div_sqrt_two_l194_194335

def unit_circle_radius : ℝ := 1

def forty_five_degrees_in_radians : ℝ := (Real.pi / 4)

def cos_45 : ℝ := Real.cos forty_five_degrees_in_radians

def sin_45 : ℝ := Real.sin forty_five_degrees_in_radians

theorem sin_45_deg_eq_one_div_sqrt_two : 
  sin_45 = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_deg_eq_one_div_sqrt_two_l194_194335


namespace relationship_among_abc_l194_194908

noncomputable def a := Real.log 2 / Real.log (1/5)
noncomputable def b := 3 ^ (3/5)
noncomputable def c := 4 ^ (1/5)

theorem relationship_among_abc : a < c ∧ c < b := 
by
  sorry

end relationship_among_abc_l194_194908


namespace average_visitors_per_day_l194_194022

theorem average_visitors_per_day (avg_sunday : ℕ) (avg_other_day : ℕ) (days_in_month : ℕ) (starts_on_sunday : Bool) :
  avg_sunday = 570 →
  avg_other_day = 240 →
  days_in_month = 30 →
  starts_on_sunday = true →
  (5 * avg_sunday + 25 * avg_other_day) / days_in_month = 295 :=
by
  intros
  sorry

end average_visitors_per_day_l194_194022


namespace value_of_a_minus_b_l194_194680

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the invertible function f

theorem value_of_a_minus_b (a b : ℝ) (hf_inv : Function.Injective f)
  (hfa : f a = b) (hfb : f b = 6) (ha1 : f 3 = 1) (hb1 : f 1 = 6) : a - b = 2 :=
sorry

end value_of_a_minus_b_l194_194680


namespace intersection_of_lines_l194_194382

theorem intersection_of_lines : 
  ∃ (x y : ℚ), 
  (5 * x - 3 * y = 20) ∧ (3 * x + 4 * y = 6) ∧ 
  x = 98 / 29 ∧ 
  y = 87 / 58 :=
by 
  sorry

end intersection_of_lines_l194_194382


namespace sin_45_eq_1_div_sqrt_2_l194_194341

theorem sin_45_eq_1_div_sqrt_2 : Real.sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_eq_1_div_sqrt_2_l194_194341


namespace coeff_x12_in_q_squared_is_zero_l194_194592

noncomputable def q (x : ℝ) : ℝ := x^5 - 2 * x^3 + 3

theorem coeff_x12_in_q_squared_is_zero : polynomial.coeff ((q x)^2) 12 = 0 :=
  sorry

end coeff_x12_in_q_squared_is_zero_l194_194592


namespace magnitude_diff_is_correct_l194_194053

def vector_a : ℝ × ℝ × ℝ := (2, -3, 1)
def vector_b : ℝ × ℝ × ℝ := (-1, 1, -4)

theorem magnitude_diff_is_correct : 
  ‖(2, -3, 1) - (-1, 1, -4)‖ = 5 * Real.sqrt 2 := 
by
  sorry

end magnitude_diff_is_correct_l194_194053


namespace only_possible_b_l194_194899

theorem only_possible_b (b : ℕ) (h : ∃ a k l : ℕ, k ≠ l ∧ (b > 0) ∧ (a > 0) ∧ (b ^ (k + l)) ∣ (a ^ k + b ^ l) ∧ (b ^ (k + l)) ∣ (a ^ l + b ^ k)) : 
  b = 1 :=
sorry

end only_possible_b_l194_194899


namespace islanders_liars_count_l194_194625

def number_of_liars (N : ℕ) : ℕ :=
  if N = 30 then 28 else 0

theorem islanders_liars_count : number_of_liars 30 = 28 :=
  sorry

end islanders_liars_count_l194_194625


namespace cost_of_slices_eaten_by_dog_is_correct_l194_194095

noncomputable def total_cost_before_tax : ℝ :=
  2 * 3 + 1 * 2 + 1 * 5 + 3 * 0.5 + 0.25 + 1.5 + 1.25

noncomputable def sales_tax_rate : ℝ := 0.06

noncomputable def sales_tax : ℝ := total_cost_before_tax * sales_tax_rate

noncomputable def total_cost_after_tax : ℝ := total_cost_before_tax + sales_tax

noncomputable def slices : ℝ := 8

noncomputable def cost_per_slice : ℝ := total_cost_after_tax / slices

noncomputable def slices_eaten_by_dog : ℝ := 8 - 3

noncomputable def cost_of_slices_eaten_by_dog : ℝ := cost_per_slice * slices_eaten_by_dog

theorem cost_of_slices_eaten_by_dog_is_correct : 
  cost_of_slices_eaten_by_dog = 11.59 := by
    sorry

end cost_of_slices_eaten_by_dog_is_correct_l194_194095


namespace correctly_calculated_expression_l194_194659

theorem correctly_calculated_expression (x : ℝ) :
  ¬ (x^3 + x^2 = x^5) ∧ 
  ¬ (x^3 * x^2 = x^6) ∧ 
  (x^3 / x^2 = x) ∧ 
  ¬ ((x^3)^2 = x^9) := by
sorry

end correctly_calculated_expression_l194_194659


namespace number_of_games_in_complete_season_l194_194159

-- Define the number of teams in each division
def teams_in_division_A : Nat := 6
def teams_in_division_B : Nat := 7
def teams_in_division_C : Nat := 5

-- Define the number of games each team must play within their division
def games_per_team_within_division (teams : Nat) : Nat :=
  (teams - 1) * 2

-- Calculate the total number of games within a division
def total_games_within_division (teams : Nat) : Nat :=
  (games_per_team_within_division teams * teams) / 2

-- Calculate cross-division games for a team in one division
def cross_division_games_per_team (teams_other_div1 : Nat) (teams_other_div2 : Nat) : Nat :=
  (teams_other_div1 + teams_other_div2) * 2

-- Calculate total cross-division games from all teams in one division
def total_cross_division_games (teams_div : Nat) (teams_other_div1 : Nat) (teams_other_div2 : Nat) : Nat :=
  cross_division_games_per_team teams_other_div1 teams_other_div2 * teams_div

-- Given conditions translated to definitions
def games_in_division_A : Nat := total_games_within_division teams_in_division_A
def games_in_division_B : Nat := total_games_within_division teams_in_division_B
def games_in_division_C : Nat := total_games_within_division teams_in_division_C

def cross_division_games_A : Nat := total_cross_division_games teams_in_division_A teams_in_division_B teams_in_division_C
def cross_division_games_B : Nat := total_cross_division_games teams_in_division_B teams_in_division_A teams_in_division_C
def cross_division_games_C : Nat := total_cross_division_games teams_in_division_C teams_in_division_A teams_in_division_B

-- Total cross-division games with each game counted twice
def total_cross_division_games_in_season : Nat :=
  (cross_division_games_A + cross_division_games_B + cross_division_games_C) / 2

-- Total number of games in the season
def total_games_in_season : Nat :=
  games_in_division_A + games_in_division_B + games_in_division_C + total_cross_division_games_in_season

-- The final proof statement
theorem number_of_games_in_complete_season : total_games_in_season = 306 :=
by
  -- This is the place where the proof would go if it were required.
  sorry

end number_of_games_in_complete_season_l194_194159


namespace die_probability_greater_than_4_given_tail_l194_194153

theorem die_probability_greater_than_4_given_tail :
  let outcomes := [1, 2, 3, 4, 5, 6]
  let favorable_outcomes := [5, 6]
  let total_outcomes := 6
  let favorable_count := List.length favorable_outcomes
  let probability := (favorable_count / total_outcomes : ℚ)
  probability = (1 / 3 : ℚ) :=
by
  -- Definitions of outcomes, conditions and computation will go here.
  sorry

end die_probability_greater_than_4_given_tail_l194_194153


namespace simplification_of_expression_l194_194035

theorem simplification_of_expression (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) : 
  ( (x - 2) / (x^2 - 2 * x + 1) / (x / (x - 1)) + 1 / (x^2 - x) ) = 1 / x := 
by 
  sorry

end simplification_of_expression_l194_194035


namespace gift_distribution_l194_194649

theorem gift_distribution :
  let bags := [1, 2, 3, 4, 5]
  let num_people := 4
  ∃ d: ℕ, d = 96 := by
  -- Proof to be completed
  sorry

end gift_distribution_l194_194649


namespace main_theorem_l194_194058

-- Definitions based on conditions
variable {Ω : Type} [ProbabilitySpace Ω]

-- X is a binomial random variable.
def X : Ω → ℕ := sorry

-- Y is a random variable such that X + Y = 8.
def Y : Ω → ℕ := λ ω => 8 - X ω

axiom X_binomial : ∀ ω, ∃ m : ℕ, X ω ~ binomial m 0.6
axiom X_binomial_params : ∀ ω, X ω ~ binomial 10 0.6

-- Expected value and variance properties of X.
noncomputable def E_X : ℝ := 6
noncomputable def D_X : ℝ := 2.4

-- Expected value of Y.
noncomputable def E_Y : ℝ := 8 - E_X

-- The final statement to prove
theorem main_theorem : D_X + E_Y = 4.4 := by
  sorry

end main_theorem_l194_194058


namespace power_evaluation_l194_194721

theorem power_evaluation : (-64 : ℝ)^(4/3) = 256 :=
by 
  have step1 : (-64 : ℝ)^(4/3) = ((-4 : ℝ)^3)^(4/3),
  { sorry },
  have step2 : ((-4 : ℝ)^3)^(4/3) = (-4 : ℝ)^4,
  { sorry },
  have step3 : (-4 : ℝ)^4 = 256,
  { sorry },
  rwa [step1, step2, step3]

end power_evaluation_l194_194721


namespace express_105_9_billion_in_scientific_notation_l194_194221

def express_in_scientific_notation (n: ℝ) : ℝ × ℤ :=
  let exponent := int.of_nat (nat.floor $ real.logb 10 n)
  let coefficient := n / real.pow 10 exponent
  (coefficient, exponent)

theorem express_105_9_billion_in_scientific_notation :
  express_in_scientific_notation (105.9 * 10^9) = (1.059, 10) :=
by
  sorry

end express_105_9_billion_in_scientific_notation_l194_194221


namespace crafts_sold_l194_194287

theorem crafts_sold (x : ℕ) 
  (h1 : ∃ (n : ℕ), 12 * n = x * 12)
  (h2 : x * 12 + 7 - 18 = 25):
  x = 3 :=
by
  sorry

end crafts_sold_l194_194287


namespace fraction_is_terminating_decimal_l194_194889

noncomputable def fraction_to_decimal : ℚ :=
  58 / 160

theorem fraction_is_terminating_decimal : fraction_to_decimal = 3625 / 10000 :=
by
  sorry

end fraction_is_terminating_decimal_l194_194889


namespace average_calculation_l194_194118

def average (a b c : ℚ) : ℚ := (a + b + c) / 3
def pairAverage (a b : ℚ) : ℚ := (a + b) / 2

theorem average_calculation :
  average (average (pairAverage 2 2) 3 1) (pairAverage 1 2) 1 = 3 / 2 := sorry

end average_calculation_l194_194118


namespace total_oysters_and_crabs_is_195_l194_194236

-- Define the initial conditions
def oysters_day1 : ℕ := 50
def crabs_day1 : ℕ := 72

-- Define the calculations for the second day
def oysters_day2 : ℕ := oysters_day1 / 2
def crabs_day2 : ℕ := crabs_day1 * 2 / 3

-- Define the total counts over the two days
def total_oysters : ℕ := oysters_day1 + oysters_day2
def total_crabs : ℕ := crabs_day1 + crabs_day2
def total_count : ℕ := total_oysters + total_crabs

-- The goal specification
theorem total_oysters_and_crabs_is_195 : total_count = 195 :=
by
  sorry

end total_oysters_and_crabs_is_195_l194_194236


namespace chord_to_diameter_ratio_l194_194145

open Real

theorem chord_to_diameter_ratio
  (r R : ℝ) (h1 : r = R / 2)
  (a : ℝ)
  (h2 : r^2 = a^2 * 3 / 2) :
  3 * a / (2 * R) = 3 * sqrt 6 / 8 :=
by
  sorry

end chord_to_diameter_ratio_l194_194145


namespace sin_45_eq_1_div_sqrt_2_l194_194336

theorem sin_45_eq_1_div_sqrt_2 : Real.sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_eq_1_div_sqrt_2_l194_194336


namespace exists_indices_for_sequences_l194_194257

theorem exists_indices_for_sequences 
  (a b c : ℕ → ℕ) :
  ∃ (p q : ℕ), p ≠ q ∧ a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q :=
sorry

end exists_indices_for_sequences_l194_194257


namespace sin_45_eq_sqrt2_div_2_l194_194322

theorem sin_45_eq_sqrt2_div_2 : Real.sin (π / 4) = Real.sqrt 2 / 2 := 
sorry

end sin_45_eq_sqrt2_div_2_l194_194322


namespace terry_age_proof_l194_194929

-- Condition 1: In 10 years, Terry will be 4 times the age that Nora is currently.
-- Condition 2: Nora is currently 10 years old.
-- We need to prove that Terry's current age is 30 years old.

variable (Terry_now Terry_in_10 Nora_now : ℕ)

theorem terry_age_proof (h1: Terry_in_10 = 4 * Nora_now) (h2: Nora_now = 10) (h3: Terry_in_10 = Terry_now + 10) : Terry_now = 30 := 
by
  sorry

end terry_age_proof_l194_194929


namespace contractor_absent_days_proof_l194_194534

def contractor_absent_days (x y : ℝ) : Prop :=
  x + y = 30 ∧ 25 * x - 7.5 * y = 425

theorem contractor_absent_days_proof : ∃ (y : ℝ), contractor_absent_days x y ∧ y = 10 :=
by
  sorry

end contractor_absent_days_proof_l194_194534


namespace not_perfect_square_2_2049_and_4_2051_l194_194247

theorem not_perfect_square_2_2049_and_4_2051 (
  hA: ∃ x : ℝ, 1^{2048} = x^2,
  hB: ∀ x : ℝ, 2^{2049} ≠ x^2,
  hC: ∃ x : ℝ, 3^{2050} = x^2,
  hD: ∀ x : ℝ, 4^{2051} ≠ x^2,
  hE: ∃ x : ℝ, 5^{2052} = x^2
  ) : ∀ x : ℝ, (2^{2049} ≠ x^2) ∧ (4^{2051} ≠ x^2) :=
by {
  sorry,
}

end not_perfect_square_2_2049_and_4_2051_l194_194247


namespace total_oysters_and_crabs_is_195_l194_194235

-- Define the initial conditions
def oysters_day1 : ℕ := 50
def crabs_day1 : ℕ := 72

-- Define the calculations for the second day
def oysters_day2 : ℕ := oysters_day1 / 2
def crabs_day2 : ℕ := crabs_day1 * 2 / 3

-- Define the total counts over the two days
def total_oysters : ℕ := oysters_day1 + oysters_day2
def total_crabs : ℕ := crabs_day1 + crabs_day2
def total_count : ℕ := total_oysters + total_crabs

-- The goal specification
theorem total_oysters_and_crabs_is_195 : total_count = 195 :=
by
  sorry

end total_oysters_and_crabs_is_195_l194_194235


namespace product_ab_l194_194554

noncomputable def a : ℝ := 1           -- From the condition 1 = a * tan(π / 4)
noncomputable def b : ℝ := 2           -- From the condition π / b = π / 2

theorem product_ab (a b : ℝ)
  (ha : a > 0) (hb : b > 0)
  (period_condition : (π / b = π / 2))
  (point_condition : a * Real.tan ((π / 8) * b) = 1) :
  a * b = 2 := sorry

end product_ab_l194_194554


namespace value_of_expression_l194_194993

theorem value_of_expression : (4.7 * 13.26 + 4.7 * 9.43 + 4.7 * 77.31) = 470 :=
by
  sorry

end value_of_expression_l194_194993


namespace solve_inequality_l194_194793

noncomputable def solution_set : Set ℝ := {x | x < -4/3 ∨ x > -13/9}

theorem solve_inequality (x : ℝ) : 
  2 - 1 / (3 * x + 4) < 5 → x ∈ solution_set :=
by
  sorry

end solve_inequality_l194_194793


namespace toms_age_is_16_l194_194091

variable (J T : ℕ) -- John's current age is J and Tom's current age is T

-- Condition 1: John was thrice as old as Tom 6 years ago
axiom h1 : J - 6 = 3 * (T - 6)

-- Condition 2: John will be 2 times as old as Tom in 4 years
axiom h2 : J + 4 = 2 * (T + 4)

-- Proving Tom's current age is 16
theorem toms_age_is_16 : T = 16 := by
  sorry

end toms_age_is_16_l194_194091


namespace power_evaluation_l194_194720

theorem power_evaluation : (-64 : ℝ)^(4/3) = 256 :=
by 
  have step1 : (-64 : ℝ)^(4/3) = ((-4 : ℝ)^3)^(4/3),
  { sorry },
  have step2 : ((-4 : ℝ)^3)^(4/3) = (-4 : ℝ)^4,
  { sorry },
  have step3 : (-4 : ℝ)^4 = 256,
  { sorry },
  rwa [step1, step2, step3]

end power_evaluation_l194_194720


namespace none_of_these_valid_l194_194544

variables {x y z w u v : ℝ}

def statement_1 (x y z w : ℝ) := x > y → z < w
def statement_2 (z w u v : ℝ) := z > w → u < v

theorem none_of_these_valid (h₁ : statement_1 x y z w) (h₂ : statement_2 z w u v) :
  ¬ ( (x < y → u < v) ∨ (u < v → x < y) ∨ (u > v → x > y) ∨ (x > y → u > v) ) :=
by {
  sorry
}

end none_of_these_valid_l194_194544


namespace Isabella_total_items_l194_194608

theorem Isabella_total_items (A_pants A_dresses I_pants I_dresses : ℕ) 
  (h1 : A_pants = 3 * I_pants) 
  (h2 : A_dresses = 3 * I_dresses)
  (h3 : A_pants = 21) 
  (h4 : A_dresses = 18) : 
  I_pants + I_dresses = 13 :=
by
  -- Proof goes here
  sorry

end Isabella_total_items_l194_194608


namespace find_x_weeks_l194_194875

-- Definition of the problem conditions:
def archibald_first_two_weeks_apples : Nat := 14
def archibald_next_x_weeks_apples (x : Nat) : Nat := 14
def archibald_last_two_weeks_apples : Nat := 42
def total_weeks : Nat := 7
def weekly_average : Nat := 10

-- Statement of the theorem to prove that x = 2 given the conditions
theorem find_x_weeks :
  ∃ x : Nat, (archibald_first_two_weeks_apples + archibald_next_x_weeks_apples x + archibald_last_two_weeks_apples = total_weeks * weekly_average) 
  ∧ (archibald_next_x_weeks_apples x / x = 7) 
  → x = 2 :=
by
  sorry

end find_x_weeks_l194_194875


namespace probability_of_sum_4_or_16_l194_194135

open Finset

def dice_rolls : Finset (ℕ × ℕ) :=
  (range 6).product (range 6)

def successful_outcomes : Finset (ℕ × ℕ) :=
  filter (λ (roll : ℕ × ℕ), roll.1 + roll.2 + 2 = 4) dice_rolls

theorem probability_of_sum_4_or_16 : 
  (card successful_outcomes : ℚ) / (card dice_rolls) = 1 / 12 :=
by
  sorry

end probability_of_sum_4_or_16_l194_194135


namespace average_speed_problem_l194_194863

noncomputable def average_speed (d₁ d₂ d₃ d₄ t₁ t₂ t₃ t₄ : ℝ) : ℝ :=
  (d₁ + d₂ + d₃ + d₄) / (t₁ + t₂ + t₃ + t₄)

theorem average_speed_problem :
  average_speed 30 40 37.5 7 (30 / 35) (40 / 55) 0.5 (10 / 60) = 51 :=
by
  -- skip the proof
  sorry

end average_speed_problem_l194_194863


namespace seventh_observation_is_4_l194_194256

def avg_six := 11 -- Average of the first six observations
def sum_six := 6 * avg_six -- Total sum of the first six observations
def new_avg := avg_six - 1 -- New average after including the new observation
def new_sum := 7 * new_avg -- Total sum after including the new observation

theorem seventh_observation_is_4 : 
  (new_sum - sum_six) = 4 :=
by
  sorry

end seventh_observation_is_4_l194_194256


namespace sin_45_deg_l194_194294

theorem sin_45_deg : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by 
  -- placeholder for the actual proof
  sorry

end sin_45_deg_l194_194294


namespace largest_value_of_y_l194_194117

theorem largest_value_of_y :
  (∃ x y : ℝ, x^2 + 3 * x * y - y^2 = 27 ∧ 3 * x^2 - x * y + y^2 = 27 ∧ y ≤ 3) → (∃ y : ℝ, y = 3) :=
by
  intro h
  obtain ⟨x, y, h1, h2, h3⟩ := h
  -- proof steps go here
  sorry

end largest_value_of_y_l194_194117


namespace candy_bar_calories_l194_194816

theorem candy_bar_calories (calories : ℕ) (bars : ℕ) (dozen : ℕ) (total_calories : ℕ) 
  (H1 : total_calories = 2016) (H2 : bars = 42) (H3 : dozen = 12) 
  (H4 : total_calories = bars * calories) : 
  calories / dozen = 4 := 
by 
  sorry

end candy_bar_calories_l194_194816


namespace values_of_quadratic_expression_l194_194648

variable {x : ℝ}

theorem values_of_quadratic_expression (h : x^2 - 4 * x + 3 < 0) : 
  (8 < x^2 + 4 * x + 3) ∧ (x^2 + 4 * x + 3 < 24) :=
sorry

end values_of_quadratic_expression_l194_194648


namespace total_population_eq_51b_over_40_l194_194933

variable (b g t : Nat)

-- Conditions
def boys_eq_four_times_girls (b g : Nat) : Prop := b = 4 * g
def girls_eq_ten_times_teachers (g t : Nat) : Prop := g = 10 * t

-- Statement to prove
theorem total_population_eq_51b_over_40 (b g t : Nat) 
  (h1 : boys_eq_four_times_girls b g) 
  (h2 : girls_eq_ten_times_teachers g t) : 
  b + g + t = (51 * b) / 40 := 
sorry

end total_population_eq_51b_over_40_l194_194933


namespace evaluate_neg_sixtyfour_exp_four_thirds_l194_194709

theorem evaluate_neg_sixtyfour_exp_four_thirds : (-64 : ℝ) ^ (4/3 : ℝ) = 256 := 
by {
  have h1 : (-64 : ℝ) = (-4 : ℝ) ^ 3, by norm_num,
  have h2 : ((-4 : ℝ) ^ 3) ^ (4/3) = (-4 : ℝ) ^ (3 * (4/3)), by rw [←real.rpow_mul],
  rw [←h1],
  rw [h2],
  norm_num,
  rw [pow_nat_cast],
  norm_num,
  norm_num,
  rw [pow_nat_cast],
  norm_num
}

end evaluate_neg_sixtyfour_exp_four_thirds_l194_194709


namespace how_many_months_to_buy_tv_l194_194258

-- Definitions based on given conditions
def monthly_income : ℕ := 30000
def food_expenses : ℕ := 15000
def utilities_expenses : ℕ := 5000
def other_expenses : ℕ := 2500

def total_expenses := food_expenses + utilities_expenses + other_expenses
def current_savings : ℕ := 10000
def tv_cost : ℕ := 25000
def monthly_savings := monthly_income - total_expenses

-- Theorem statement based on the problem
theorem how_many_months_to_buy_tv 
    (H_income : monthly_income = 30000)
    (H_food : food_expenses = 15000)
    (H_utilities : utilities_expenses = 5000)
    (H_other : other_expenses = 2500)
    (H_savings : current_savings = 10000)
    (H_tv_cost : tv_cost = 25000)
    : (tv_cost - current_savings) / monthly_savings = 2 :=
by
  sorry

end how_many_months_to_buy_tv_l194_194258


namespace quadratic_roots_eq1_quadratic_roots_eq2_l194_194476

theorem quadratic_roots_eq1 :
  ∀ x : ℝ, (x^2 + 3 * x - 1 = 0) ↔ (x = (-3 + Real.sqrt 13) / 2 ∨ x = (-3 - Real.sqrt 13) / 2) :=
by
  intros x
  sorry

theorem quadratic_roots_eq2 :
  ∀ x : ℝ, ((x + 2)^2 = (x + 2)) ↔ (x = -2 ∨ x = -1) :=
by
  intros x
  sorry

end quadratic_roots_eq1_quadratic_roots_eq2_l194_194476


namespace triangular_array_nth_row_4th_number_l194_194164

theorem triangular_array_nth_row_4th_number (n : ℕ) (h : n ≥ 4) :
  ∃ k : ℕ, k = 4 ∧ (2: ℕ)^(n * (n - 1) / 2 + 3) = 2^((n^2 - n + 6) / 2) :=
by
  sorry

end triangular_array_nth_row_4th_number_l194_194164


namespace sin_45_degree_l194_194328

theorem sin_45_degree : sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_degree_l194_194328


namespace square_pyramid_sum_l194_194171

def square_pyramid_faces : Nat := 5
def square_pyramid_edges : Nat := 8
def square_pyramid_vertices : Nat := 5

theorem square_pyramid_sum : square_pyramid_faces + square_pyramid_edges + square_pyramid_vertices = 18 := by
  sorry

end square_pyramid_sum_l194_194171


namespace parallel_line_slope_l194_194655

theorem parallel_line_slope (x y : ℝ) :
  ∃ m b : ℝ, (3 * x - 6 * y = 21) → ∀ (x₁ y₁ : ℝ), (3 * x₁ - 6 * y₁ = 21) → m = 1 / 2 :=
by
  sorry

end parallel_line_slope_l194_194655


namespace polygon_sides_l194_194760

theorem polygon_sides (n : ℕ) (h : 180 * (n - 2) = 720) : n = 6 :=
sorry

end polygon_sides_l194_194760


namespace total_nuts_correct_l194_194093

-- Definitions for conditions
def w : ℝ := 0.25
def a : ℝ := 0.25
def p : ℝ := 0.15
def c : ℝ := 0.40

-- The theorem to be proven
theorem total_nuts_correct : w + a + p + c = 1.05 := by
  sorry

end total_nuts_correct_l194_194093


namespace prop_A_prop_B_prop_C_prop_D_l194_194617

variable {a b : ℝ}

-- Proposition A
theorem prop_A (h : a^2 - b^2 = 1) (a_pos : 0 < a) (b_pos : 0 < b) : a - b < 1 := sorry

-- Proposition B (negation of the original proposition since B is incorrect)
theorem prop_B (h : (1 / b) - (1 / a) = 1) (a_pos : 0 < a) (b_pos : 0 < b) : a - b ≥ 1 := sorry

-- Proposition C
theorem prop_C (h : a > b + 1) (a_pos : 0 < a) (b_pos : 0 < b) : a^2 > b^2 + 1 := sorry

-- Proposition D (negation of the original proposition since D is incorrect)
theorem prop_D (h1 : a ≤ 1) (h2 : b ≤ 1) (a_pos : 0 < a) (b_pos : 0 < b) : |a - b| < |1 - a * b| := sorry

end prop_A_prop_B_prop_C_prop_D_l194_194617


namespace prime_factors_difference_l194_194502

theorem prime_factors_difference (n : ℤ) (h₁ : n = 180181) : ∃ p q : ℤ, Prime p ∧ Prime q ∧ p > q ∧ n % p = 0 ∧ n % q = 0 ∧ (p - q) = 2 :=
by
  sorry

end prime_factors_difference_l194_194502


namespace general_term_a_n_sum_of_b_n_l194_194910

-- Proof Problem 1: General term of sequence {a_n}
theorem general_term_a_n (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) (h1 : a 1 = 2) (h2 : a 2 = 4) 
    (h3 : ∀ n ≥ 2, a (n+1) - a n = 2) : 
    ∀ n, a n = 2 * n :=
by
  sorry

-- Proof Problem 2: Sum of the first n terms of sequence {b_n}
theorem sum_of_b_n (a : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ) (n : ℕ)
    (h : ∀ n, (1 / (a n ^ 2 - 1) : ℝ) + b n = 2^n) :
    T n = 2^(n+1) - n / (2*n + 1) :=
by
  sorry

end general_term_a_n_sum_of_b_n_l194_194910


namespace circle_equation_l194_194390

theorem circle_equation (a : ℝ) (h : a = 1) :
  (∀ (C : ℝ × ℝ), C = (a, a) →
  (∀ (r : ℝ), r = dist C (1, 0) →
  r = 1 → ((x - a) ^ 2 + (y - a) ^ 2 = r ^ 2))) :=
by
  sorry

end circle_equation_l194_194390


namespace calc_first_term_l194_194573

theorem calc_first_term (a d : ℚ)
    (h1 : 15 * (2 * a + 29 * d) = 300)
    (h2 : 20 * (2 * a + 99 * d) = 2200) :
    a = -121 / 14 :=
by
  -- We can add the sorry placeholder here as we are not providing the complete proof steps
  sorry

end calc_first_term_l194_194573


namespace common_difference_in_arithmetic_sequence_l194_194189

theorem common_difference_in_arithmetic_sequence
  (a : ℕ → ℝ) (d : ℝ)
  (h1 : a 2 = 3)
  (h2 : a 5 = 12) :
  d = 3 :=
by
  sorry

end common_difference_in_arithmetic_sequence_l194_194189


namespace functions_are_even_l194_194004

noncomputable def f_A (x : ℝ) : ℝ := -|x| + 2
noncomputable def f_B (x : ℝ) : ℝ := x^2 - 3
noncomputable def f_C (x : ℝ) : ℝ := Real.sqrt (1 - x^2)

theorem functions_are_even :
  (∀ x : ℝ, f_A x = f_A (-x)) ∧
  (∀ x : ℝ, f_B x = f_B (-x)) ∧
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f_C x = f_C (-x)) :=
by
  sorry

end functions_are_even_l194_194004


namespace probability_useful_parts_l194_194674

noncomputable def probability_three_parts_useful (pipe_length : ℝ) (min_length : ℝ) : ℝ :=
  let total_area := (pipe_length * pipe_length) / 2
  let feasible_area := ((pipe_length - min_length) * (pipe_length - min_length)) / 2
  feasible_area / total_area

theorem probability_useful_parts :
  probability_three_parts_useful 300 75 = 1 / 16 :=
by
  sorry

end probability_useful_parts_l194_194674


namespace exists_pairs_angle_120_degrees_l194_194201

theorem exists_pairs_angle_120_degrees :
  ∃ a b : ℤ, a + b ≠ 0 ∧ a + b ≠ a ^ 2 - a * b + b ^ 2 ∧ (a + b) * 13 = 3 * (a ^ 2 - a * b + b ^ 2) :=
sorry

end exists_pairs_angle_120_degrees_l194_194201


namespace math_expr_evaluation_l194_194504

theorem math_expr_evaluation :
  3 + 15 / 3 - 2^2 + 1 = 5 :=
by
  -- The proof will be filled here
  sorry

end math_expr_evaluation_l194_194504


namespace sin_45_eq_sqrt2_div_2_l194_194356

theorem sin_45_eq_sqrt2_div_2 : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by sorry

end sin_45_eq_sqrt2_div_2_l194_194356


namespace mary_principal_amount_l194_194465

theorem mary_principal_amount (t1 t2 t3 t4:ℕ) (P R:ℕ) :
  (t1 = 2) →
  (t2 = 260) →
  (t3 = 5) →
  (t4 = 350) →
  (P + 2 * P * R = t2) →
  (P + 5 * P * R = t4) →
  P = 200 :=
by
  intros
  sorry

end mary_principal_amount_l194_194465


namespace find_smallest_n_l194_194802

/-- 
Define the doubling sum function D(a, n)
-/
def doubling_sum (a : ℕ) (n : ℕ) : ℕ := a * (2^n - 1)

/--
Main theorem statement that proves the smallest n for the given conditions
-/
theorem find_smallest_n :
  ∃ (n : ℕ), (∀ (i : ℕ), 1 ≤ i ∧ i ≤ 6 → ∃ (ai : ℕ), doubling_sum ai i = n) ∧ n = 9765 := 
sorry

end find_smallest_n_l194_194802


namespace min_value_I_is_3_l194_194631

noncomputable def min_value_I (a b c x y : ℝ) : ℝ :=
  1 / (2 * a^3 * x + b^3 * y^2) + 1 / (2 * b^3 * x + c^3 * y^2) + 1 / (2 * c^3 * x + a^3 * y^2)

theorem min_value_I_is_3 {a b c x y : ℝ} (h1 : a^6 + b^6 + c^6 = 3) (h2 : (x + 1)^2 + y^2 ≤ 2) :
  3 ≤ min_value_I a b c x y :=
sorry

end min_value_I_is_3_l194_194631


namespace equation_solutions_l194_194513

theorem equation_solutions
  (a : ℝ) :
  (∃ x : ℝ, (1 < a ∧ a < 2) ∧ (x = (1 - a) / a ∨ x = -1)) ∨
  (a = 2 ∧ (∃ x : ℝ, x = -1 ∨ x = -1/2)) ∨
  (a > 2 ∧ (∃ x : ℝ, x = (1 - a) / a ∨ x = -1 ∨ x = 1 - a)) ∨
  (0 ≤ a ∧ a ≤ 1 ∧ (∃ x : ℝ, x = -1)) ∨
  (a < 0 ∧ (∃ x : ℝ, x = (1 - a) / a ∨ x = -1)) := sorry

end equation_solutions_l194_194513


namespace multiple_6_9_statements_false_l194_194478

theorem multiple_6_9_statements_false
    (a b : ℤ)
    (h₁ : ∃ m : ℤ, a = 6 * m)
    (h₂ : ∃ n : ℤ, b = 9 * n) :
    ¬ (∀ m n : ℤ,  a = 6 * m → b = 9 * n → ((a + b) % 2 = 0)) ∧
    ¬ (∀ m n : ℤ,  a = 6 * m → b = 9 * n → (a + b) % 6 = 0) ∧
    ¬ (∀ m n : ℤ,  a = 6 * m → b = 9 * n → (a + b) % 9 = 0) ∧
    ¬ (∀ m n : ℤ,  a = 6 * m → b = 9 * n → (a + b) % 9 ≠ 0) :=
by
  sorry

end multiple_6_9_statements_false_l194_194478


namespace gum_pieces_bought_correct_l194_194675

-- Define initial number of gum pieces
def initial_gum_pieces : ℕ := 10

-- Define number of friends Adrianna gave gum to
def friends_given_gum : ℕ := 11

-- Define the number of pieces Adrianna has left
def remaining_gum_pieces : ℕ := 2

-- Define a function to calculate the number of gum pieces Adrianna bought at the store
def gum_pieces_bought (initial_gum : ℕ) (given_gum : ℕ) (remaining_gum : ℕ) : ℕ :=
  (given_gum + remaining_gum) - initial_gum

-- Now state the theorem to prove the number of pieces bought is 3
theorem gum_pieces_bought_correct : 
  gum_pieces_bought initial_gum_pieces friends_given_gum remaining_gum_pieces = 3 :=
by
  sorry

end gum_pieces_bought_correct_l194_194675


namespace sin_45_degree_eq_sqrt2_div_2_l194_194353

theorem sin_45_degree_eq_sqrt2_div_2 :
  let θ := (real.pi / 4)
  in sin θ = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_degree_eq_sqrt2_div_2_l194_194353


namespace initial_nickels_l194_194652

variable (q0 n0 : Nat)
variable (d_nickels : Nat := 3) -- His dad gave him 3 nickels
variable (final_nickels : Nat := 12) -- Tim has now 12 nickels

theorem initial_nickels (q0 : Nat) (n0 : Nat) (d_nickels : Nat) (final_nickels : Nat) :
  final_nickels = n0 + d_nickels → n0 = 9 :=
by
  sorry

end initial_nickels_l194_194652


namespace smallest_x_l194_194082

theorem smallest_x (x y : ℝ) (h1 : 4 < x) (h2 : x < 6) (h3 : 6 < y) (h4 : y < 10) (h5 : y - x = 5) :
  x = 4 :=
sorry

end smallest_x_l194_194082


namespace max_a_condition_l194_194432

theorem max_a_condition (a : ℝ) :
  (∀ x : ℝ, x < a → |x| > 2) ∧ (∃ x : ℝ, |x| > 2 ∧ ¬ (x < a)) →
  a ≤ -2 :=
by 
  sorry

end max_a_condition_l194_194432


namespace sum_of_A_H_l194_194939

theorem sum_of_A_H (A B C D E F G H : ℝ) (h1 : C = 10) 
  (h2 : A + B + C = 40) (h3 : B + C + D = 40) (h4 : C + D + E = 40) 
  (h5 : D + E + F = 40) (h6 : E + F + G = 40) (h7 : F + G + H = 40) :
  A + H = 30 := 
sorry

end sum_of_A_H_l194_194939


namespace number_of_tacos_l194_194609

-- Define the conditions and prove the statement
theorem number_of_tacos (T : ℕ) :
  (4 * 7 + 9 * T = 37) → T = 1 :=
by
  intro h
  sorry

end number_of_tacos_l194_194609


namespace horner_value_x_neg2_l194_194838

noncomputable def horner (x : ℝ) : ℝ :=
  (((((x - 5) * x + 6) * x + 0) * x + 1) * x + 0.3) * x + 2

theorem horner_value_x_neg2 : horner (-2) = -40 :=
by
  sorry

end horner_value_x_neg2_l194_194838


namespace find_years_l194_194280

def sum_interest_years (P R : ℝ) (T : ℝ) : Prop :=
  (P * (R + 5) / 100 * T = P * R / 100 * T + 300) ∧ P = 600

theorem find_years {R : ℝ} {T : ℝ} (h1 : sum_interest_years 600 R T) : T = 10 :=
by
  -- proof omitted
  sorry

end find_years_l194_194280


namespace probability_heart_then_club_l194_194821

noncomputable def numHearts : ℕ := 13
noncomputable def numClubs : ℕ := 13
noncomputable def totalCards (n : ℕ) : ℕ := 52 - n

noncomputable def probabilityFirstHeart : ℚ := numHearts / totalCards 0
noncomputable def probabilitySecondClubGivenFirstHeart : ℚ := numClubs / totalCards 1

theorem probability_heart_then_club :
  (probabilityFirstHeart * probabilitySecondClubGivenFirstHeart) = 13 / 204 :=
by
  sorry

end probability_heart_then_club_l194_194821


namespace count_valid_three_digit_numbers_l194_194071

-- Define the property of a number being three digits
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

-- Define the property of the units digit being at least three times the tens digit
def validUnitsTimesTens (n : ℕ) : Prop :=
  let units := n % 10
  let tens := (n / 10) % 10
  units ≥ 3 * tens

-- Definition combining both conditions
def validThreeDigitNumber (n : ℕ) : Prop :=
  isThreeDigit n ∧ validUnitsTimesTens n

-- The main theorem stating the number of valid three-digit numbers
theorem count_valid_three_digit_numbers :
  Finset.card (Finset.filter validThreeDigitNumber (Finset.range 1000)) = 198 :=
by
  sorry

end count_valid_three_digit_numbers_l194_194071


namespace area_of_square_field_l194_194843

theorem area_of_square_field (d : ℝ) (s : ℝ) (A : ℝ) (h_d : d = 28) (h_relation : d = s * Real.sqrt 2) (h_area : A = s^2) :
  A = 391.922 :=
by sorry

end area_of_square_field_l194_194843


namespace man_is_26_years_older_l194_194543

variable (S : ℕ) (M : ℕ)

-- conditions
def present_age_of_son : Prop := S = 24
def future_age_relation : Prop := M + 2 = 2 * (S + 2)

-- question transformed to a proof problem
theorem man_is_26_years_older
  (h1 : present_age_of_son S)
  (h2 : future_age_relation S M) : M - S = 26 := by
  sorry

end man_is_26_years_older_l194_194543


namespace reflect_parallelogram_l194_194215

theorem reflect_parallelogram :
  let D : ℝ × ℝ := (4,1)
  let Dx : ℝ × ℝ := (D.1, -D.2) -- Reflect across x-axis
  let Dxy : ℝ × ℝ := (Dx.2 - 1, Dx.1 - 1) -- Translate point down by 1 unit and reflect across y=x
  let D'' : ℝ × ℝ := (Dxy.1 + 1, Dxy.2 + 1) -- Translate point back up by 1 unit
  D'' = (-2, 5) := by
  sorry

end reflect_parallelogram_l194_194215


namespace average_age_after_person_leaves_l194_194481

theorem average_age_after_person_leaves 
  (initial_people : ℕ) 
  (initial_average_age : ℕ) 
  (person_leaving_age : ℕ) 
  (remaining_people : ℕ) 
  (new_average_age : ℝ)
  (h1 : initial_people = 7) 
  (h2 : initial_average_age = 32) 
  (h3 : person_leaving_age = 22) 
  (h4 : remaining_people = 6) :
  new_average_age = 34 := 
by 
  sorry

end average_age_after_person_leaves_l194_194481


namespace friends_in_group_l194_194540

theorem friends_in_group : 
  ∀ (total_chicken_wings cooked_wings additional_wings chicken_wings_per_person : ℕ), 
    cooked_wings = 8 →
    additional_wings = 10 →
    chicken_wings_per_person = 6 →
    total_chicken_wings = cooked_wings + additional_wings →
    total_chicken_wings / chicken_wings_per_person = 3 :=
by
  intros total_chicken_wings cooked_wings additional_wings chicken_wings_per_person hcooked hadditional hperson htotal
  sorry

end friends_in_group_l194_194540


namespace solution_set_condition_l194_194130

theorem solution_set_condition (a : ℝ) :
  (∀ x : ℝ, x * (x - a + 1) > a ↔ (x < -1 ∨ x > a)) → a > -1 :=
sorry

end solution_set_condition_l194_194130


namespace parabola_hyperbola_tangent_l194_194230

open Real

theorem parabola_hyperbola_tangent (n : ℝ) : 
  (∀ x y : ℝ, y = x^2 + 6 → y^2 - n * x^2 = 4 → y ≥ 6) ↔ (n = 12 + 4 * sqrt 7 ∨ n = 12 - 4 * sqrt 7) :=
by
  sorry

end parabola_hyperbola_tangent_l194_194230


namespace boards_nailing_l194_194110

variables {x y a b : ℕ} 

theorem boards_nailing :
  (2 * x + 3 * y = 87) ∧
  (3 * a + 5 * b = 94) →
  (x + y = 30) ∧ (a + b = 30) :=
by
  sorry

end boards_nailing_l194_194110


namespace molecular_weight_is_122_l194_194982

noncomputable def molecular_weight_of_compound := 
  let atomic_weight_C := 12.01
  let atomic_weight_H := 1.008
  let atomic_weight_O := 16.00
  7 * atomic_weight_C + 6 * atomic_weight_H + 2 * atomic_weight_O

theorem molecular_weight_is_122 :
  molecular_weight_of_compound = 122 := by
  sorry

end molecular_weight_is_122_l194_194982


namespace roots_quadratic_square_diff_10_l194_194066

-- Definition and theorem statement in Lean 4
theorem roots_quadratic_square_diff_10 :
  ∀ x1 x2 : ℝ, (2 * x1^2 + 4 * x1 - 3 = 0) ∧ (2 * x2^2 + 4 * x2 - 3 = 0) →
  (x1 - x2)^2 = 10 :=
by
  sorry

end roots_quadratic_square_diff_10_l194_194066


namespace remainder_2027_div_28_l194_194003

theorem remainder_2027_div_28 : 2027 % 28 = 3 :=
by
  sorry

end remainder_2027_div_28_l194_194003


namespace rahul_share_payment_l194_194637

theorem rahul_share_payment
  (rahul_days : ℕ)
  (rajesh_days : ℕ)
  (total_payment : ℚ)
  (H1 : rahul_days = 3)
  (H2 : rajesh_days = 2)
  (H3 : total_payment = 2250) :
  let rahul_work_per_day := (1 : ℚ) / rahul_days
  let rajesh_work_per_day := (1 : ℚ) / rajesh_days
  let total_work_per_day := rahul_work_per_day + rajesh_work_per_day
  let rahul_fraction_of_total_work := rahul_work_per_day / total_work_per_day
  let rahul_share := rahul_fraction_of_total_work * total_payment
  rahul_share = 900 := by
  sorry

end rahul_share_payment_l194_194637


namespace cannot_equal_120_l194_194672

def positive_even (n : ℕ) : Prop := n > 0 ∧ n % 2 = 0

theorem cannot_equal_120 (a b : ℕ) (ha : positive_even a) (hb : positive_even b) :
  let A := a * b
  let P' := 2 * (a + b) + 6
  A + P' ≠ 120 :=
sorry

end cannot_equal_120_l194_194672


namespace contains_zero_l194_194033

open Nat

theorem contains_zero 
  (x y : ℕ)
  (hx : 10000 ≤ x ∧ x < 100000)
  (hy : 10000 ≤ y ∧ y < 100000)
  (h_swap : ∃ (i j : ℕ), i ≠ j ∧ swap_digits x i j = y)
  (h_sum : x + y = 111111) :
  (∃ i, digit x i = 0) ∨ (∃ j, digit y j = 0) :=
sorry

-- Swap two digits in a number
def swap_digits (n : ℕ) (i j : ℕ) : ℕ := 
  -- Dummy definition for the proof placeholder
  n -- replace with actual implementation

-- Extract the digit at position i
def digit (n : ℕ) (i : ℕ) : ℕ :=
  -- Dummy definition for the proof placeholder
  0 -- replace with actual implementation

end contains_zero_l194_194033


namespace card_probability_l194_194826

theorem card_probability :
  let hearts := 13
  let clubs := 13
  let total_cards := 52
  let first_card_is_heart := (hearts.to_rat / total_cards.to_rat)
  let second_card_is_club_given_first_is_heart := (clubs.to_rat / (total_cards - 1).to_rat)
  first_card_is_heart * second_card_is_club_given_first_is_heart = (13.to_rat / 204.to_rat) := by
  sorry

end card_probability_l194_194826


namespace remainder_avg_is_correct_l194_194593

-- Definitions based on the conditions
variables (total_avg : ℝ) (first_part_avg : ℝ) (second_part_avg : ℝ) (first_part_percent : ℝ) (second_part_percent : ℝ)

-- The conditions stated mathematically
def overall_avg_contribution 
  (remainder_avg : ℝ) : Prop :=
  first_part_percent * first_part_avg + 
  second_part_percent * second_part_avg + 
  (1 - first_part_percent - second_part_percent) * remainder_avg =  total_avg
  
-- The question
theorem remainder_avg_is_correct : overall_avg_contribution 75 80 65 0.25 0.50 90 := sorry

end remainder_avg_is_correct_l194_194593


namespace computer_cost_l194_194089

theorem computer_cost (C : ℝ) (h1 : 0.10 * C = a) (h2 : 3 * C = b) (h3 : b - 1.10 * C = 2700) : 
  C = 2700 / 2.90 :=
by
  sorry

end computer_cost_l194_194089


namespace total_books_in_class_l194_194524

theorem total_books_in_class (Tables : ℕ) (BooksPerTable : ℕ) (TotalBooks : ℕ) 
  (h1 : Tables = 500)
  (h2 : BooksPerTable = (2 * Tables) / 5)
  (h3 : TotalBooks = Tables * BooksPerTable) :
  TotalBooks = 100000 := 
sorry

end total_books_in_class_l194_194524


namespace circle_through_points_l194_194900

-- Definitions of the points
def O : (ℝ × ℝ) := (0, 0)
def M1 : (ℝ × ℝ) := (1, 1)
def M2 : (ℝ × ℝ) := (4, 2)

-- Definition of the center and radius of the circle
def center : (ℝ × ℝ) := (4, -3)
def radius : ℝ := 5

-- The circle equation function
def circle_eq (x y : ℝ) (c : ℝ × ℝ) (r : ℝ) : Prop :=
  (x - c.1)^2 + (y + c.2)^2 = r^2

theorem circle_through_points :
  circle_eq 0 0 center radius ∧ circle_eq 1 1 center radius ∧ circle_eq 4 2 center radius :=
by
  -- This is where the proof would go
  sorry

end circle_through_points_l194_194900


namespace unique_function_satisfies_condition_l194_194726

theorem unique_function_satisfies_condition :
  ∃! f : ℝ → ℝ, ∀ x y z : ℝ, f (x * Real.sin y) + f (x * Real.sin z) -
    f x * f (Real.sin y * Real.sin z) + Real.sin (Real.pi * x) ≥ 1 := sorry

end unique_function_satisfies_condition_l194_194726


namespace power_evaluation_l194_194719

theorem power_evaluation : (-64 : ℝ)^(4/3) = 256 :=
by 
  have step1 : (-64 : ℝ)^(4/3) = ((-4 : ℝ)^3)^(4/3),
  { sorry },
  have step2 : ((-4 : ℝ)^3)^(4/3) = (-4 : ℝ)^4,
  { sorry },
  have step3 : (-4 : ℝ)^4 = 256,
  { sorry },
  rwa [step1, step2, step3]

end power_evaluation_l194_194719


namespace passengers_from_other_continents_l194_194756

theorem passengers_from_other_continents :
  (∀ (n NA EU AF AS : ℕ),
     NA = n / 4 →
     EU = n / 8 →
     AF = n / 12 →
     AS = n / 6 →
     96 = n →
     n - (NA + EU + AF + AS) = 36) :=
by
  sorry

end passengers_from_other_continents_l194_194756


namespace flu_infection_equation_l194_194671

theorem flu_infection_equation (x : ℕ) (h : 1 + x + x^2 = 36) : 1 + x + x^2 = 36 :=
by
  sorry

end flu_infection_equation_l194_194671


namespace express_A_using_roster_method_l194_194422

def A := {x : ℕ | ∃ (n : ℕ), 8 / (2 - x) = n }

theorem express_A_using_roster_method :
  A = {0, 1} :=
sorry

end express_A_using_roster_method_l194_194422


namespace cafeteria_pies_l194_194666

theorem cafeteria_pies (initial_apples handed_out_apples apples_per_pie : ℕ)
  (h1 : initial_apples = 96)
  (h2 : handed_out_apples = 42)
  (h3 : apples_per_pie = 6) :
  (initial_apples - handed_out_apples) / apples_per_pie = 9 := by
  sorry

end cafeteria_pies_l194_194666


namespace determine_m_if_root_exists_l194_194759

def fractional_equation_has_root (x m : ℝ) : Prop :=
  (3 / (x - 4) + (x + m) / (4 - x) = 1)

theorem determine_m_if_root_exists (x : ℝ) (h : fractional_equation_has_root x m) : m = -1 :=
sorry

end determine_m_if_root_exists_l194_194759


namespace sin_theta_correct_l194_194775

noncomputable def sin_theta : ℝ :=
  let d := (4, 5, 7)
  let n := (3, -4, 5)
  let d_dot_n := 4 * 3 + 5 * (-4) + 7 * 5
  let norm_d := Real.sqrt (4^2 + 5^2 + 7^2)
  let norm_n := Real.sqrt (3^2 + (-4)^2 + 5^2)
  let cos_theta := d_dot_n / (norm_d * norm_n)
  cos_theta

theorem sin_theta_correct :
  sin_theta = 27 / Real.sqrt 4500 :=
by
  sorry

end sin_theta_correct_l194_194775


namespace evaluate_pow_l194_194705

theorem evaluate_pow : (-64 : ℝ)^(4/3) = 256 := 
by
  sorry

end evaluate_pow_l194_194705


namespace sin_45_deg_l194_194299

theorem sin_45_deg : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by 
  -- placeholder for the actual proof
  sorry

end sin_45_deg_l194_194299


namespace power_ineq_for_n_geq_5_l194_194632

noncomputable def power_ineq (n : ℕ) : Prop := 2^n > n^2 + 1

theorem power_ineq_for_n_geq_5 (n : ℕ) (h : n ≥ 5) : power_ineq n :=
  sorry

end power_ineq_for_n_geq_5_l194_194632


namespace quadratic_point_comparison_l194_194762

theorem quadratic_point_comparison (c y1 y2 y3 : ℝ) 
  (h1 : y1 = -(-2:ℝ)^2 + c)
  (h2 : y2 = -(1:ℝ)^2 + c)
  (h3 : y3 = -(3:ℝ)^2 + c) : y2 > y1 ∧ y1 > y3 := 
by
  sorry

end quadratic_point_comparison_l194_194762


namespace sum_of_xyz_l194_194441

theorem sum_of_xyz (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x * y = 18) (hxz : x * z = 3) (hyz : y * z = 6) : x + y + z = 10 := 
sorry

end sum_of_xyz_l194_194441


namespace remainder_polynomial_l194_194460

noncomputable def p (x : ℝ) : ℝ := sorry
noncomputable def r (x : ℝ) : ℝ := x^2 + x

theorem remainder_polynomial (p : ℝ → ℝ) (r : ℝ → ℝ) :
  (p 2 = 6) ∧ (p 4 = 20) ∧ (p 6 = 42) →
  (r 2 = 2^2 + 2) ∧ (r 4 = 4^2 + 4) ∧ (r 6 = 6^2 + 6) :=
sorry

end remainder_polynomial_l194_194460


namespace small_boxes_count_correct_l194_194562

-- Definitions of constants
def feet_per_large_box_seal : ℕ := 4
def feet_per_medium_box_seal : ℕ := 2
def feet_per_small_box_seal : ℕ := 1
def feet_per_box_label : ℕ := 1

def large_boxes_packed : ℕ := 2
def medium_boxes_packed : ℕ := 8
def total_tape_used : ℕ := 44

-- Definition for the total tape used for large and medium boxes
def tape_used_large_boxes : ℕ := (large_boxes_packed * feet_per_large_box_seal) + (large_boxes_packed * feet_per_box_label)
def tape_used_medium_boxes : ℕ := (medium_boxes_packed * feet_per_medium_box_seal) + (medium_boxes_packed * feet_per_box_label)
def tape_used_large_and_medium_boxes : ℕ := tape_used_large_boxes + tape_used_medium_boxes
def tape_used_small_boxes : ℕ := total_tape_used - tape_used_large_and_medium_boxes

-- The number of small boxes packed
def small_boxes_packed : ℕ := tape_used_small_boxes / (feet_per_small_box_seal + feet_per_box_label)

-- Proof problem statement
theorem small_boxes_count_correct (n : ℕ) (h : small_boxes_packed = n) : n = 5 :=
by
  sorry

end small_boxes_count_correct_l194_194562


namespace problem_statement_l194_194744

noncomputable def A := 5 * Real.pi / 12
noncomputable def B := Real.pi / 3
noncomputable def C := Real.pi / 4
noncomputable def b := Real.sqrt 3
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x - Real.sqrt 3

theorem problem_statement :
  (Set.Icc (-2 : ℝ) 2 = Set.image f Set.univ) ∧
  (∀ x : ℝ, f (x + Real.pi) = f x) ∧
  (∃ (area : ℝ), area = (3 + Real.sqrt 3) / 4)
:= sorry

end problem_statement_l194_194744


namespace johnson_potatoes_l194_194773

/-- Given that Johnson has a sack of 300 potatoes, 
    gives some to Gina, twice that amount to Tom, and 
    one-third of the amount given to Tom to Anne,
    and has 47 potatoes left, we prove that 
    Johnson gave Gina 69 potatoes. -/
theorem johnson_potatoes : 
  ∃ G : ℕ, 
  ∀ (Gina Tom Anne total : ℕ), 
    total = 300 ∧ 
    total - (Gina + Tom + Anne) = 47 ∧ 
    Tom = 2 * Gina ∧ 
    Anne = (1 / 3 : ℚ) * Tom ∧ 
    (Gina + Tom + (Anne : ℕ)) = (11 / 3 : ℚ) * Gina ∧ 
    (Gina + Tom + Anne) = 253 
    ∧ total = Gina + Tom + Anne + 47 
    → Gina = 69 := sorry


end johnson_potatoes_l194_194773


namespace yuna_average_score_l194_194850

theorem yuna_average_score (avg_may_june : ℕ) (score_july : ℕ) (h1 : avg_may_june = 84) (h2 : score_july = 96) :
  (avg_may_june * 2 + score_july) / 3 = 88 := by
  sorry

end yuna_average_score_l194_194850


namespace evaluate_neg_sixtyfour_exp_four_thirds_l194_194710

theorem evaluate_neg_sixtyfour_exp_four_thirds : (-64 : ℝ) ^ (4/3 : ℝ) = 256 := 
by {
  have h1 : (-64 : ℝ) = (-4 : ℝ) ^ 3, by norm_num,
  have h2 : ((-4 : ℝ) ^ 3) ^ (4/3) = (-4 : ℝ) ^ (3 * (4/3)), by rw [←real.rpow_mul],
  rw [←h1],
  rw [h2],
  norm_num,
  rw [pow_nat_cast],
  norm_num,
  norm_num,
  rw [pow_nat_cast],
  norm_num
}

end evaluate_neg_sixtyfour_exp_four_thirds_l194_194710


namespace total_company_pay_monthly_l194_194457

-- Define the given conditions
def hours_josh_works_daily : ℕ := 8
def days_josh_works_weekly : ℕ := 5
def weeks_josh_works_monthly : ℕ := 4
def hourly_rate_josh : ℕ := 9

-- Define Carl's working hours and rate based on the conditions
def hours_carl_works_daily : ℕ := hours_josh_works_daily - 2
def hourly_rate_carl : ℕ := hourly_rate_josh / 2

-- Calculate total hours worked monthly by Josh and Carl
def total_hours_josh_monthly : ℕ := hours_josh_works_daily * days_josh_works_weekly * weeks_josh_works_monthly
def total_hours_carl_monthly : ℕ := hours_carl_works_daily * days_josh_works_weekly * weeks_josh_works_monthly

-- Calculate monthly pay for Josh and Carl
def monthly_pay_josh : ℕ := total_hours_josh_monthly * hourly_rate_josh
def monthly_pay_carl : ℕ := total_hours_carl_monthly * hourly_rate_carl

-- Theorem to prove the total pay for both Josh and Carl in one month
theorem total_company_pay_monthly : monthly_pay_josh + monthly_pay_carl = 1980 := by
  sorry

end total_company_pay_monthly_l194_194457


namespace sin_45_eq_one_div_sqrt_two_l194_194314

theorem sin_45_eq_one_div_sqrt_two
  (Q : ℝ × ℝ)
  (h1 : Q = (real.cos (real.pi / 4), real.sin (real.pi / 4)))
  (h2 : Q.2 = real.sin (real.pi / 4)) :
  real.sin (real.pi / 4) = 1 / real.sqrt 2 := 
sorry

end sin_45_eq_one_div_sqrt_two_l194_194314


namespace root_properties_of_polynomial_l194_194778

variables {r s t : ℝ}

def polynomial (x : ℝ) : ℝ := 6 * x^3 + 4 * x^2 + 1500 * x + 3000

theorem root_properties_of_polynomial :
  (∀ x : ℝ, polynomial x = 0 → (x = r ∨ x = s ∨ x = t)) →
  (r + s + t = -2 / 3) →
  (r * s + r * t + s * t = 250) →
  (r * s * t = -500) →
  (r + s)^3 + (s + t)^3 + (t + r)^3 = -5992 / 27 :=
by
  sorry

end root_properties_of_polynomial_l194_194778


namespace jason_money_l194_194615

theorem jason_money (fred_money_before : ℕ) (jason_money_before : ℕ)
  (fred_money_after : ℕ) (total_earned : ℕ) :
  fred_money_before = 111 →
  jason_money_before = 40 →
  fred_money_after = 115 →
  total_earned = 4 →
  jason_money_before = 40 := by
  intros h1 h2 h3 h4
  sorry

end jason_money_l194_194615


namespace ellipse_foci_coordinates_l194_194644

theorem ellipse_foci_coordinates :
  (∀ (x y : ℝ), (x^2 / 16 + y^2 / 25 = 1) → (∃ (c : ℝ), c = 3 ∧ (x = 0 ∧ (y = c ∨ y = -c)))) :=
by
  sorry

end ellipse_foci_coordinates_l194_194644


namespace parallel_line_eq_l194_194391

theorem parallel_line_eq (x y : ℝ) (c : ℝ) :
  (∀ x y, x - 2 * y - 2 = 0 → x - 2 * y + c = 0) ∧ (x = 1 ∧ y = 0) → c = -1 :=
by
  sorry

end parallel_line_eq_l194_194391


namespace desired_butterfat_percentage_l194_194522

theorem desired_butterfat_percentage (milk1 milk2 : ℝ) (butterfat1 butterfat2 : ℝ) :
  milk1 = 8 →
  butterfat1 = 0.10 →
  milk2 = 8 →
  butterfat2 = 0.30 →
  ((butterfat1 * milk1) + (butterfat2 * milk2)) / (milk1 + milk2) * 100 = 20 := 
by
  intros
  sorry

end desired_butterfat_percentage_l194_194522


namespace fraction_computation_l194_194907

theorem fraction_computation (p q s u : ℚ)
  (hpq : p / q = 5 / 2)
  (hsu : s / u = 7 / 11) :
  (5 * p * s - 3 * q * u) / (7 * q * u - 4 * p * s) = 109 / 14 := 
by
  sorry

end fraction_computation_l194_194907


namespace inverse_proposition_false_l194_194486

-- Define the original proposition
def original_proposition (a b : ℝ) : Prop :=
  a = b → abs a = abs b

-- Define the inverse proposition
def inverse_proposition (a b : ℝ) : Prop :=
  abs a = abs b → a = b

-- The theorem to prove
theorem inverse_proposition_false : ∃ (a b : ℝ), abs a = abs b ∧ a ≠ b :=
sorry

end inverse_proposition_false_l194_194486


namespace E_plays_2_games_l194_194815

-- Definitions for the students and the number of games they played
def students := ["A", "B", "C", "D", "E"]
def games_played_by (S : String) : Nat :=
  if S = "A" then 4 else
  if S = "B" then 3 else
  if S = "C" then 2 else 
  if S = "D" then 1 else
  2  -- this is the number of games we need to prove for student E 

-- Theorem stating the number of games played by E
theorem E_plays_2_games : games_played_by "E" = 2 :=
  sorry

end E_plays_2_games_l194_194815


namespace solve_for_y_l194_194641

theorem solve_for_y (x y : ℝ) (h1 : 3 * x^2 + 4 * x + 7 * y + 2 = 0) (h2 : 3 * x + 2 * y + 5 = 0) : 4 * y^2 + 33 * y + 11 = 0 :=
sorry

end solve_for_y_l194_194641


namespace eval_neg_pow_l194_194694

theorem eval_neg_pow (a b : ℝ) (h1 : a = (-4)^3) (h2 : b = (-64)) : (-64 : ℝ)^(4/3) = 256 :=
by {
  have h_eq : b = a := by rw h1,
  rw [h2, h_eq],
  have : (a : ℝ)^(4/3) = ((-4)^3)^(4/3) := by rw h1,
  rw this,
  have : ((-4)^3)^(4/3) = (-4)^4 := by norm_num,
  rw this,
  norm_num,
  exact rfl,
}

end eval_neg_pow_l194_194694


namespace integer_solution_system_l194_194722

theorem integer_solution_system (n : ℕ) (H : n ≥ 2) : 
  ∃ (x : ℕ → ℤ), (
    ∀ i : ℕ, x ((i % n) + 1)^2 + x (((i + 1) % n) + 1)^2 + 50 = 16 * x ((i % n) + 1) + 12 * x (((i + 1) % n) + 1)
  ) ↔ n % 3 = 0 :=
by
  sorry

end integer_solution_system_l194_194722


namespace museum_admission_ratio_l194_194120

theorem museum_admission_ratio (a c : ℕ) (h1 : 30 * a + 15 * c = 2700) (h2 : 2 ≤ a) (h3 : 2 ≤ c) :
  a / (180 - 2 * a) = 2 :=
by
  sorry

end museum_admission_ratio_l194_194120


namespace total_attendance_l194_194872

theorem total_attendance (A C : ℕ) (adult_ticket_price child_ticket_price total_revenue : ℕ) 
(h1 : adult_ticket_price = 11) (h2 : child_ticket_price = 10) (h3 : total_revenue = 246) 
(h4 : C = 7) (h5 : adult_ticket_price * A + child_ticket_price * C = total_revenue) : 
A + C = 23 :=
by {
  sorry
}

end total_attendance_l194_194872


namespace find_M_l194_194758

theorem find_M : 
  let S := (981 + 983 + 985 + 987 + 989 + 991 + 993 + 995 + 997 + 999)
  let Target := 5100 - M
  S = Target → M = 4800 :=
by
  sorry

end find_M_l194_194758


namespace sin_45_eq_one_div_sqrt_two_l194_194316

theorem sin_45_eq_one_div_sqrt_two
  (Q : ℝ × ℝ)
  (h1 : Q = (real.cos (real.pi / 4), real.sin (real.pi / 4)))
  (h2 : Q.2 = real.sin (real.pi / 4)) :
  real.sin (real.pi / 4) = 1 / real.sqrt 2 := 
sorry

end sin_45_eq_one_div_sqrt_two_l194_194316


namespace solution_set_inequality_l194_194188

variable (a b c : ℝ)
variable (condition1 : ∀ x : ℝ, ax^2 + bx + c < 0 ↔ x < -1 ∨ 2 < x)

theorem solution_set_inequality (h : a < 0 ∧ b = -a ∧ c = -2 * a) :
  ∀ x : ℝ, (bx^2 + ax - c ≤ 0) ↔ (-1 ≤ x ∧ x ≤ 2) :=
by
  intro x
  sorry

end solution_set_inequality_l194_194188


namespace power_function_expression_l194_194191

theorem power_function_expression (α : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = x ^ α) (h_point : f 2 = 4) :
  α = 2 ∧ (∀ x, f x = x ^ 2) :=
by
  sorry

end power_function_expression_l194_194191


namespace cubic_root_identity_l194_194776

theorem cubic_root_identity (a b c : ℝ) 
  (h1 : a + b + c = 0) 
  (h2 : a * b + a * c + b * c = -3)
  (h3 : a * b * c = -2) : 
  a * (b + c) ^ 2 + b * (c + a) ^ 2 + c * (a + b) ^ 2 = -6 := 
by
  sorry

end cubic_root_identity_l194_194776


namespace probability_of_specific_individual_drawn_on_third_attempt_l194_194240

theorem probability_of_specific_individual_drawn_on_third_attempt :
  let population_size := 6
  let sample_size := 3
  let prob_not_drawn_first_attempt := 5 / 6
  let prob_not_drawn_second_attempt := 4 / 5
  let prob_drawn_third_attempt := 1 / 4
  (prob_not_drawn_first_attempt * prob_not_drawn_second_attempt * prob_drawn_third_attempt) = 1 / 6 :=
by sorry

end probability_of_specific_individual_drawn_on_third_attempt_l194_194240


namespace sample_variance_l194_194814

theorem sample_variance (a : ℝ) (h_avg : (a + 0 + 1 + 2 + 3) / 5 = 1) :
  (1 / 5) * ((a - 1)^2 + (0 - 1)^2 + (1 - 1)^2 + (2 - 1)^2 + (3 - 1)^2) = 2 :=
by
  sorry

end sample_variance_l194_194814


namespace three_digit_numbers_count_l194_194070

def is_3_digit_number (n : Nat) : Prop :=
  100 ≤ n ∧ n ≤ 999

def units_digit (n : Nat) : Nat :=
  n % 10

def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

def hundreds_digit (n : Nat) : Nat :=
  (n / 100) % 10

def valid_number (n : Nat) : Prop :=
  is_3_digit_number n ∧ units_digit n ≥ 3 * tens_digit n

theorem three_digit_numbers_count : 
  (Finset.card (Finset.filter valid_number (Finset.range 1000))) = 198 :=
by
  sorry

end three_digit_numbers_count_l194_194070


namespace cars_meet_time_l194_194994

theorem cars_meet_time 
  (L : ℕ) (v1 v2 : ℕ) (t : ℕ)
  (H1 : L = 333)
  (H2 : v1 = 54)
  (H3 : v2 = 57)
  (H4 : v1 * t + v2 * t = L) : 
  t = 3 :=
by
  -- Insert proof here
  sorry

end cars_meet_time_l194_194994


namespace tea_customers_count_l194_194552

theorem tea_customers_count :
  ∃ T : ℕ, 7 * 5 + T * 4 = 67 ∧ T = 8 :=
by
  sorry

end tea_customers_count_l194_194552


namespace simplify_expression_l194_194196

theorem simplify_expression (p q r s : ℝ) (hp : p ≠ 6) (hq : q ≠ 7) (hr : r ≠ 8) (hs : s ≠ 9) :
    (p - 6) / (8 - r) * (q - 7) / (6 - p) * (r - 8) / (7 - q) * (s - 9) / (9 - s) = 1 := by
  sorry

end simplify_expression_l194_194196


namespace tank_breadth_l194_194538

/-
  We need to define the conditions:
  1. The field dimensions.
  2. The tank dimensions (length and depth), and the unknown breadth.
  3. The relationship after the tank is dug.
-/

noncomputable def field_length : ℝ := 90
noncomputable def field_breadth : ℝ := 50
noncomputable def tank_length : ℝ := 25
noncomputable def tank_depth : ℝ := 4
noncomputable def rise_in_level : ℝ := 0.5

theorem tank_breadth (B : ℝ) (h : 100 * B = (field_length * field_breadth - tank_length * B) * rise_in_level) : B = 20 :=
by sorry

end tank_breadth_l194_194538


namespace stratified_sampling_l194_194232

theorem stratified_sampling
  (ratio_first : ℕ)
  (ratio_second : ℕ)
  (ratio_third : ℕ)
  (sample_size : ℕ)
  (h_ratio : ratio_first = 3 ∧ ratio_second = 4 ∧ ratio_third = 3)
  (h_sample_size : sample_size = 50) :
  (ratio_second * sample_size) / (ratio_first + ratio_second + ratio_third) = 20 :=
by
  sorry

end stratified_sampling_l194_194232


namespace joint_probability_l194_194415

noncomputable def P (A B : Prop) : ℝ := sorry
def A : Prop := sorry
def B : Prop := sorry

axiom prob_A : P A true = 0.005
axiom prob_B_given_A : P B true = 0.99

theorem joint_probability :
  P A B = 0.00495 :=
by sorry

end joint_probability_l194_194415


namespace bowling_average_decrease_l194_194272

theorem bowling_average_decrease 
  (original_average : ℚ) 
  (wickets_last_match : ℚ) 
  (runs_last_match : ℚ) 
  (original_wickets : ℚ) 
  (original_total_runs : ℚ := original_wickets * original_average) 
  (new_total_wickets : ℚ := original_wickets + wickets_last_match) 
  (new_total_runs : ℚ := original_total_runs + runs_last_match)
  (new_average : ℚ := new_total_runs / new_total_wickets) :
  original_wickets = 85 → original_average = 12.4 → wickets_last_match = 5 → runs_last_match = 26 → new_average = 12 →
  original_average - new_average = 0.4 := 
by 
  intros 
  sorry

end bowling_average_decrease_l194_194272


namespace geometric_progression_fourth_term_eq_one_l194_194227

theorem geometric_progression_fourth_term_eq_one :
  let a₁ := (2:ℝ)^(1/4)
  let a₂ := (2:ℝ)^(1/6)
  let a₃ := (2:ℝ)^(1/12)
  let r := a₂ / a₁
  let a₄ := a₃ * r
  a₄ = 1 := by
  sorry

end geometric_progression_fourth_term_eq_one_l194_194227


namespace triangle_rectangle_area_l194_194820

theorem triangle_rectangle_area (DE EF DF : ℕ) (h1 : DE = 15) (h2 : EF = 39) (h3 : DF = 36)
  (Area_DEF : ℕ) (h4 : Area_DEF = 270)
  (Area_WXYZ : ℕ → ℕ) (h5 : ∀ ω, Area_WXYZ ω = 39 * ω - ((60 / 169) * ω^2))
: ∃ p q : ℕ, p + q = 229 :=
by
  use 60
  use 169
  norm_num
  exact rfl

end triangle_rectangle_area_l194_194820


namespace shaded_area_of_logo_l194_194023

theorem shaded_area_of_logo 
  (side_length_of_square : ℝ)
  (side_length_of_square_eq : side_length_of_square = 30)
  (radius_of_circle : ℝ)
  (radius_eq : radius_of_circle = side_length_of_square / 4)
  (number_of_circles : ℕ)
  (number_of_circles_eq : number_of_circles = 4)
  : (side_length_of_square^2) - (number_of_circles * Real.pi * (radius_of_circle^2)) = 900 - 225 * Real.pi := by
    sorry

end shaded_area_of_logo_l194_194023


namespace volume_of_sand_pile_l194_194020

theorem volume_of_sand_pile (d h : ℝ) (π : ℝ) (r : ℝ) (vol : ℝ) :
  d = 8 →
  h = (3 / 4) * d →
  r = d / 2 →
  vol = (1 / 3) * π * r^2 * h →
  vol = 32 * π :=
by
  intros hd hh hr hv
  subst hd
  subst hh
  subst hr
  subst hv
  sorry

end volume_of_sand_pile_l194_194020


namespace gcd_polynomial_l194_194742

theorem gcd_polynomial (b : ℤ) (h1 : ∃ k : ℤ, b = 7 * k ∧ k % 2 = 1) : 
  Int.gcd (3 * b ^ 2 + 34 * b + 76) (b + 16) = 7 := 
sorry

end gcd_polynomial_l194_194742


namespace inequality_proof_l194_194179

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c):
  1/a + 1/b + 1/c ≥ 2/(a + b) + 2/(b + c) + 2/(c + a) ∧ 2/(a + b) + 2/(b + c) + 2/(c + a) ≥ 9/(a + b + c) :=
sorry

end inequality_proof_l194_194179


namespace sin_750_eq_one_half_l194_194383

theorem sin_750_eq_one_half :
  ∀ (θ: ℝ), (∀ n: ℤ, Real.sin (θ + n * 360) = Real.sin θ) → Real.sin 30 = 1 / 2 → Real.sin 750 = 1 / 2 :=
by 
  intros θ periodic_sine sin_30
  -- insert proof here
  sorry

end sin_750_eq_one_half_l194_194383


namespace decompose_number_4705_l194_194998

theorem decompose_number_4705 :
  4.705 = 4 * 1 + 7 * 0.1 + 0 * 0.01 + 5 * 0.001 := by
  sorry

end decompose_number_4705_l194_194998


namespace solve_arcsin_eq_l194_194216

noncomputable def arcsin (x : ℝ) : ℝ := Real.arcsin x
noncomputable def pi : ℝ := Real.pi

theorem solve_arcsin_eq :
  ∃ x : ℝ, arcsin x + arcsin (3 * x) = pi / 4 ∧ x = 1 / Real.sqrt 19 :=
sorry

end solve_arcsin_eq_l194_194216


namespace problem_1_problem_2_l194_194407

-- Definitions of conditions
variables {a b : ℝ}
axiom h_pos_a : a > 0
axiom h_pos_b : b > 0
axiom h_sum : a + b = 1

-- The statements to prove
theorem problem_1 : 
  (1 / (a^2)) + (1 / (b^2)) ≥ 8 := 
sorry

theorem problem_2 : 
  (1 / a) + (1 / b) + (1 / (a * b)) ≥ 8 := 
sorry

end problem_1_problem_2_l194_194407


namespace train_passing_time_l194_194661

theorem train_passing_time :
  ∀ (length : ℕ) (speed_km_hr : ℕ), length = 300 ∧ speed_km_hr = 90 →
  (length / (speed_km_hr * (1000 / 3600)) = 12) := 
by
  intros length speed_km_hr h
  have h_length : length = 300 := h.1
  have h_speed : speed_km_hr = 90 := h.2
  sorry

end train_passing_time_l194_194661


namespace volume_relation_l194_194973

noncomputable def A (r : ℝ) : ℝ := (2 / 3) * Real.pi * r^3
noncomputable def M (r : ℝ) : ℝ := 2 * Real.pi * r^3
noncomputable def C (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem volume_relation (r : ℝ) : A r - M r + C r = 0 :=
by
  sorry

end volume_relation_l194_194973


namespace necessary_and_sufficient_l194_194149

theorem necessary_and_sufficient (a b : ℝ) : a > b ↔ a * |a| > b * |b| := sorry

end necessary_and_sufficient_l194_194149


namespace evaluate_neg_64_exp_4_over_3_l194_194714

theorem evaluate_neg_64_exp_4_over_3 : (-64 : ℝ) ^ (4 / 3) = 256 := 
by
  sorry

end evaluate_neg_64_exp_4_over_3_l194_194714


namespace plate_729_driving_days_l194_194451

def plate (n : ℕ) : Prop := n >= 0 ∧ n <= 999

def monday (n : ℕ) : Prop := n % 2 = 1

def sum_digits (n : ℕ) : ℕ :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 + d2 + d3

def tuesday (n : ℕ) : Prop := sum_digits n >= 11

def wednesday (n : ℕ) : Prop := n % 3 = 0

def thursday (n : ℕ) : Prop := sum_digits n <= 14

def count_digits (n : ℕ) : ℕ × ℕ × ℕ :=
  (n / 100, (n / 10) % 10, n % 10)

def friday (n : ℕ) : Prop :=
  let (d1, d2, d3) := count_digits n
  d1 = d2 ∨ d2 = d3 ∨ d1 = d3

def saturday (n : ℕ) : Prop := n < 500

def sunday (n : ℕ) : Prop := 
  let (d1, d2, d3) := count_digits n
  d1 <= 5 ∧ d2 <= 5 ∧ d3 <= 5

def can_drive (n : ℕ) (day : String) : Prop :=
  plate n ∧ 
  (day = "Monday" → monday n) ∧ 
  (day = "Tuesday" → tuesday n) ∧ 
  (day = "Wednesday" → wednesday n) ∧ 
  (day = "Thursday" → thursday n) ∧ 
  (day = "Friday" → friday n) ∧ 
  (day = "Saturday" → saturday n) ∧ 
  (day = "Sunday" → sunday n)

theorem plate_729_driving_days :
  can_drive 729 "Monday" ∧
  can_drive 729 "Tuesday" ∧
  can_drive 729 "Wednesday" ∧
  ¬ can_drive 729 "Thursday" ∧
  ¬ can_drive 729 "Friday" ∧
  ¬ can_drive 729 "Saturday" ∧
  ¬ can_drive 729 "Sunday" :=
by
  sorry

end plate_729_driving_days_l194_194451


namespace range_of_a_l194_194483

theorem range_of_a {a : ℝ} : (∀ x1 x2 : ℝ, 2 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 4 → -x1^2 + 4*a*x1 ≤ -x2^2 + 4*a*x2)
  ∨ (∀ x1 x2 : ℝ, 2 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 4 → -x1^2 + 4*a*x1 ≥ -x2^2 + 4*a*x2) ↔ (a ≤ 1 ∨ a ≥ 2) :=
by
  sorry

end range_of_a_l194_194483


namespace eight_packets_weight_l194_194673

variable (weight_per_can : ℝ)
variable (weight_per_packet : ℝ)

-- Conditions
axiom h1 : weight_per_can = 1
axiom h2 : 3 * weight_per_can = 8 * weight_per_packet
axiom h3 : weight_per_packet = 6 * weight_per_can

-- Question to be proved: 8 packets weigh 12 kg
theorem eight_packets_weight : 8 * weight_per_packet = 12 :=
by 
  -- Proof would go here
  sorry

end eight_packets_weight_l194_194673


namespace x_less_than_2_necessary_not_sufficient_x_less_than_2_is_necessary_not_sufficient_l194_194688

theorem x_less_than_2_necessary_not_sufficient (x : ℝ) :
  (x^2 - 3 * x + 2 < 0) ↔ (1 < x ∧ x < 2) := sorry

theorem x_less_than_2_is_necessary_not_sufficient : 
  (∀ x : ℝ, x^2 - 3*x + 2 < 0 → x < 2) ∧ 
  (¬ ∀ x : ℝ, x < 2 → x^2 - 3*x + 2 < 0) := sorry

end x_less_than_2_necessary_not_sufficient_x_less_than_2_is_necessary_not_sufficient_l194_194688


namespace salary_C_more_than_A_ratio_salary_E_to_A_and_B_l194_194969

variable (x : ℝ)
variables (salary_A salary_B salary_C salary_D salary_E combined_salary_BCD : ℝ)

-- Conditions
def conditions : Prop :=
  salary_B = 2 * salary_A ∧
  salary_C = 3 * salary_A ∧
  salary_D = 4 * salary_A ∧
  salary_E = 5 * salary_A ∧
  combined_salary_BCD = 15000 ∧
  combined_salary_BCD = salary_B + salary_C + salary_D

-- Statements to prove
theorem salary_C_more_than_A
  (cond : conditions salary_A salary_B salary_C salary_D salary_E combined_salary_BCD) :
  (salary_C - salary_A) / salary_A * 100 = 200 := by
  sorry

theorem ratio_salary_E_to_A_and_B
  (cond : conditions salary_A salary_B salary_C salary_D salary_E combined_salary_BCD) :
  salary_E / (salary_A + salary_B) = 5 / 3 := by
  sorry

end salary_C_more_than_A_ratio_salary_E_to_A_and_B_l194_194969


namespace sin_45_eq_sqrt2_div_2_l194_194363

theorem sin_45_eq_sqrt2_div_2 :
  Real.sin (π / 4) = Real.sqrt 2 / 2 := 
by
  sorry

end sin_45_eq_sqrt2_div_2_l194_194363


namespace math_problem_l194_194755

theorem math_problem 
  (a : ℤ) 
  (h_a : a = -1) 
  (b : ℚ) 
  (h_b : b = 0) 
  (c : ℕ) 
  (h_c : c = 1)
  : a^2024 + 2023 * b - c^2023 = 0 := by
  sorry

end math_problem_l194_194755


namespace original_price_vase_l194_194155

-- Definitions based on the conditions and problem elements
def original_price (P : ℝ) : Prop :=
  0.825 * P = 165

-- Statement to prove equivalence
theorem original_price_vase : ∃ P : ℝ, original_price P ∧ P = 200 :=
  by
    sorry

end original_price_vase_l194_194155


namespace klinker_age_l194_194782

theorem klinker_age (K D : ℕ) (h1 : D = 10) (h2 : K + 15 = 2 * (D + 15)) : K = 35 :=
by
  sorry

end klinker_age_l194_194782


namespace islanders_liars_count_l194_194627

theorem islanders_liars_count :
  ∀ (n : ℕ), n = 30 → 
  ∀ (I : fin n → Prop), -- predicate indicating if an islander is a knight (true) or a liar (false)
  (∀ i : fin n, 
    ((I i → (∀ j : fin n, i ≠ j ∧ abs (i - j) ≤ 1 → ¬ I j)) ∧ -- if i is a knight, all except neighbors are liars
    (¬ I i → (∃ k : fin n, j ≠ j ∧ abs (i - j) ≤ 1 ∧ I k)) -- if i is a liar, there exists at least one knight among non-neighbors
  )) → 
  (Σ (liars : fin n), (liars.card = 28)) :=
sorry

end islanders_liars_count_l194_194627


namespace average_percentage_revenue_fall_l194_194979

theorem average_percentage_revenue_fall
  (initial_revenue_A final_revenue_A : ℝ)
  (initial_revenue_B final_revenue_B : ℝ) (exchange_rate_B : ℝ)
  (initial_revenue_C final_revenue_C : ℝ) (exchange_rate_C : ℝ) :
  initial_revenue_A = 72.0 →
  final_revenue_A = 48.0 →
  initial_revenue_B = 20.0 →
  final_revenue_B = 15.0 →
  exchange_rate_B = 1.30 →
  initial_revenue_C = 6000.0 →
  final_revenue_C = 5500.0 →
  exchange_rate_C = 0.0091 →
  (33.33 + 25 + 8.33) / 3 = 22.22 :=
by
  sorry

end average_percentage_revenue_fall_l194_194979


namespace absolute_difference_probability_l194_194085

-- Define the conditions
def num_red_marbles : ℕ := 1500
def num_black_marbles : ℕ := 2000
def total_marbles : ℕ := num_red_marbles + num_black_marbles

def P_s : ℚ :=
  let ways_to_choose_2_red := (num_red_marbles * (num_red_marbles - 1)) / 2
  let ways_to_choose_2_black := (num_black_marbles * (num_black_marbles - 1)) / 2
  let total_favorable_outcomes := ways_to_choose_2_red + ways_to_choose_2_black
  total_favorable_outcomes / (total_marbles * (total_marbles - 1) / 2)

def P_d : ℚ :=
  (num_red_marbles * num_black_marbles) / (total_marbles * (total_marbles - 1) / 2)

-- Prove the statement
theorem absolute_difference_probability : |P_s - P_d| = 1 / 50 := by
  sorry

end absolute_difference_probability_l194_194085


namespace eval_neg_pow_l194_194692

theorem eval_neg_pow (a b : ℝ) (h1 : a = (-4)^3) (h2 : b = (-64)) : (-64 : ℝ)^(4/3) = 256 :=
by {
  have h_eq : b = a := by rw h1,
  rw [h2, h_eq],
  have : (a : ℝ)^(4/3) = ((-4)^3)^(4/3) := by rw h1,
  rw this,
  have : ((-4)^3)^(4/3) = (-4)^4 := by norm_num,
  rw this,
  norm_num,
  exact rfl,
}

end eval_neg_pow_l194_194692


namespace find_f1_plus_g1_l194_194063

variables (f g : ℝ → ℝ)

-- Conditions
def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g x
def function_equation (f g : ℝ → ℝ) : Prop := ∀ x : ℝ, f x - g x = x^3 - 2*x^2 + 1

theorem find_f1_plus_g1 
  (hf : even_function f)
  (hg : odd_function g)
  (hfg : function_equation f g):
  f 1 + g 1 = -2 :=
by {
  sorry
}

end find_f1_plus_g1_l194_194063


namespace contractor_absent_days_l194_194529

variable (x y : ℝ) -- x for the number of days worked, y for the number of days absent

-- Conditions
def eng_days := x + y = 30
def total_money := 25 * x - 7.5 * y = 425

-- Theorem
theorem contractor_absent_days (x y : ℝ) (h1 : eng_days x y) (h2 : total_money x y) : y = 10 := 
sorry

end contractor_absent_days_l194_194529


namespace dogs_in_shelter_l194_194932

theorem dogs_in_shelter (D C : ℕ) (h1 : D * 7 = 15 * C) (h2 : D * 11 = 15 * (C + 8)) :
  D = 30 :=
sorry

end dogs_in_shelter_l194_194932


namespace sin_45_eq_sqrt2_div_2_l194_194357

theorem sin_45_eq_sqrt2_div_2 : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by sorry

end sin_45_eq_sqrt2_div_2_l194_194357


namespace value_of_n_l194_194920

theorem value_of_n (n : ℕ) (h : sqrt (10 + n) = 9) : n = 71 :=
by
  sorry

end value_of_n_l194_194920


namespace sin_45_degree_l194_194324

theorem sin_45_degree : sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_degree_l194_194324


namespace probability_heart_then_club_l194_194827

theorem probability_heart_then_club :
  let P_heart := 13 / 52
  let P_club_given_heart := 13 / 51
  P_heart * P_club_given_heart = 13 / 204 := 
by
  let P_heart := (13 : ℚ) / 52
  let P_club_given_heart := (13 : ℚ) / 51
  have h : P_heart * P_club_given_heart = 13 / 204 := by
    calc
      P_heart * P_club_given_heart
        = (13 / 52) * (13 / 51) : rfl
    ... = (13 * 13) / (52 * 51) : by rw [mul_div_mul_comm]
    ... = 169 / 2652 : rfl
    ... = 13 / 204 : by norm_num
  exact h

end probability_heart_then_club_l194_194827


namespace probability_three_white_balls_l194_194859

open Nat

def totalWaysToDrawThreeBalls : ℕ := choose 15 3
def waysToDrawThreeWhiteBalls : ℕ := choose 7 3

theorem probability_three_white_balls :
  (waysToDrawThreeWhiteBalls : ℚ) / (totalWaysToDrawThreeBalls : ℚ) = 1 / 13 := 
sorry

end probability_three_white_balls_l194_194859


namespace find_x_squared_minus_y_squared_l194_194051

theorem find_x_squared_minus_y_squared 
  (x y : ℝ)
  (h1 : x + y = 5)
  (h2 : x - y = 1) :
  x^2 - y^2 = 5 := 
by
  sorry

end find_x_squared_minus_y_squared_l194_194051


namespace eval_expression_correct_l194_194698

noncomputable def eval_expression : ℝ := (-64)^(4/3)

theorem eval_expression_correct : eval_expression = 256 := by
  sorry

end eval_expression_correct_l194_194698


namespace maximum_value_of_f_l194_194901

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 * Real.sqrt x

theorem maximum_value_of_f :
  ∃ x_max : ℝ, x_max > 0 ∧ (∀ x : ℝ, x > 0 → f x ≤ f x_max) ∧ f x_max = -2 :=
by
  sorry

end maximum_value_of_f_l194_194901


namespace domain_of_f_l194_194801

noncomputable def f (x : ℝ) := Real.sqrt (x - 1) + (1 / (x - 2))

theorem domain_of_f : { x : ℝ | x ≥ 1 ∧ x ≠ 2 } = { x : ℝ | ∃ (y : ℝ), f x = y } :=
sorry

end domain_of_f_l194_194801


namespace determine_friends_l194_194784

inductive Grade
| first
| second
| third
| fourth

inductive Name
| Petya
| Kolya
| Alyosha
| Misha
| Dima
| Borya
| Vasya

inductive Surname
| Ivanov
| Krylov
| Petrov
| Orlov

structure Friend :=
  (name : Name)
  (surname : Surname)
  (grade : Grade)

def friends : List Friend :=
  [ {name := Name.Dima, surname := Surname.Ivanov, grade := Grade.first},
    {name := Name.Misha, surname := Surname.Krylov, grade := Grade.second},
    {name := Name.Borya, surname := Surname.Petrov, grade := Grade.third},
    {name := Name.Vasya, surname := Surname.Orlov, grade := Grade.fourth} ]

theorem determine_friends : ∃ l : List Friend, 
  {name := Name.Dima, surname := Surname.Ivanov, grade := Grade.first} ∈ l ∧
  {name := Name.Misha, surname := Surname.Krylov, grade := Grade.second} ∈ l ∧
  {name := Name.Borya, surname := Surname.Petrov, grade := Grade.third} ∈ l ∧
  {name := Name.Vasya, surname := Surname.Orlov, grade := Grade.fourth} ∈ l :=
by 
  use friends
  repeat { simp [friends] }


end determine_friends_l194_194784


namespace calculate_expression_l194_194292

theorem calculate_expression : 4 + (-8) / (-4) - (-1) = 7 := 
by 
  sorry

end calculate_expression_l194_194292


namespace find_number_eq_150_l194_194857

variable {x : ℝ}

theorem find_number_eq_150 (h : 0.60 * x - 40 = 50) : x = 150 :=
sorry

end find_number_eq_150_l194_194857


namespace factorize_ab_factorize_x_l194_194387

-- Problem 1: Factorization of a^3 b - 2 a^2 b^2 + a b^3
theorem factorize_ab (a b : ℤ) : a^3 * b - 2 * a^2 * b^2 + a * b^3 = a * b * (a - b)^2 := 
by sorry

-- Problem 2: Factorization of (x^2 + 4)^2 - 16 x^2
theorem factorize_x (x : ℤ) : (x^2 + 4)^2 - 16 * x^2 = (x + 2)^2 * (x - 2)^2 :=
by sorry

end factorize_ab_factorize_x_l194_194387


namespace kth_term_in_sequence_l194_194105

theorem kth_term_in_sequence (k : ℕ) (hk : 0 < k) : ℚ :=
  (2 * k) / (2 * k + 1)

end kth_term_in_sequence_l194_194105


namespace number_of_liars_l194_194628

-- Definitions based on the conditions
def num_islanders : ℕ := 30

def can_see (i j : ℕ) (n : ℕ) : Prop :=
  i ≠ j ∧ (j ≠ ((i + 1) % n)) ∧ (j ≠ ((i - 1 + n) % n))

def says_all_liars (i : ℕ) (see_liars : ℕ → Prop) : Prop :=
  ∀ j, can_see i j num_islanders → see_liars j

inductive Islander
| knight : Islander
| liar   : Islander

-- Knights always tell the truth and liars always lie
def is_knight (i : ℕ) : Prop := sorry

def is_liar (i : ℕ) : Prop := sorry

def see_liars (i : ℕ) : Prop :=
  if is_knight i then
    ∀ j, can_see i j num_islanders → is_liar j
  else
    ∃ j, can_see i j num_islanders ∧ is_knight j

-- Main theorem
theorem number_of_liars :
  ∃ liars, liars = num_islanders - 2 :=
sorry

end number_of_liars_l194_194628


namespace inverse_proposition_false_l194_194485

-- Define the original proposition: ∀ a b, if a = b then |a| = |b|
def original_proposition (a b : ℝ) : Prop := a = b → abs a = abs b

-- Define the inverse proposition: ∀ a b, if |a| = |b| then a = b
def inverse_proposition (a b : ℝ) : Prop := abs a = abs b → a = b

-- Prove that the inverse proposition is false
theorem inverse_proposition_false : ¬ (∀ a b : ℝ, inverse_proposition a b) := 
by {
  intro h,
  have h1 : inverse_proposition 1 (-1) := h 1 (-1),
  have h2 : abs 1 = abs (-1),
  { rfl },
  exact h1 h2,
  have h3 : 1 = -1,
  { sorry }
}

end inverse_proposition_false_l194_194485


namespace minimum_ticket_cost_l194_194198

-- Definitions of the conditions in Lean
def southern_cities : ℕ := 4
def northern_cities : ℕ := 5
def one_way_ticket_cost (N : ℝ) : ℝ := N
def round_trip_ticket_cost (N : ℝ) : ℝ := 1.6 * N

-- The main theorem to prove
theorem minimum_ticket_cost (N : ℝ) : 
  (∀ (Y1 Y2 Y3 Y4 : ℕ), 
  (∀ (S1 S2 S3 S4 S5 : ℕ), 
  southern_cities = 4 → northern_cities = 5 →
  one_way_ticket_cost N = N →
  round_trip_ticket_cost N = 1.6 * N →
  ∃ (total_cost : ℝ), total_cost = 6.4 * N)) :=
sorry

end minimum_ticket_cost_l194_194198


namespace dvd_book_capacity_l194_194150

/--
Theorem: Given that there are 81 DVDs already in the DVD book and it can hold 45 more DVDs,
the total capacity of the DVD book is 126 DVDs.
-/
theorem dvd_book_capacity : 
  (already_in_book additional_capacity : ℕ) (h1 : already_in_book = 81) (h2 : additional_capacity = 45) :
  already_in_book + additional_capacity = 126 :=
by
  sorry

end dvd_book_capacity_l194_194150


namespace train_cross_signal_in_18_sec_l194_194517

-- Definitions of the given conditions
def train_length := 300 -- meters
def platform_length := 350 -- meters
def time_cross_platform := 39 -- seconds

-- Speed of the train
def train_speed := (train_length + platform_length) / time_cross_platform -- meters/second

-- Time to cross the signal pole
def time_cross_signal_pole := train_length / train_speed -- seconds

theorem train_cross_signal_in_18_sec : time_cross_signal_pole = 18 := by sorry

end train_cross_signal_in_18_sec_l194_194517


namespace growth_factor_condition_l194_194470

open BigOperators

theorem growth_factor_condition {n : ℕ} (h : ∏ i in Finset.range n, (i + 2) / (i + 1) = 50) : n = 49 := by
  sorry

end growth_factor_condition_l194_194470


namespace sin_45_eq_sqrt2_div_2_l194_194364

theorem sin_45_eq_sqrt2_div_2 :
  Real.sin (π / 4) = Real.sqrt 2 / 2 := 
by
  sorry

end sin_45_eq_sqrt2_div_2_l194_194364


namespace eval_neg_pow_l194_194693

theorem eval_neg_pow (a b : ℝ) (h1 : a = (-4)^3) (h2 : b = (-64)) : (-64 : ℝ)^(4/3) = 256 :=
by {
  have h_eq : b = a := by rw h1,
  rw [h2, h_eq],
  have : (a : ℝ)^(4/3) = ((-4)^3)^(4/3) := by rw h1,
  rw this,
  have : ((-4)^3)^(4/3) = (-4)^4 := by norm_num,
  rw this,
  norm_num,
  exact rfl,
}

end eval_neg_pow_l194_194693


namespace dante_walk_time_l194_194624

-- Define conditions and problem
variables (T R : ℝ)

-- Conditions as per the problem statement
def wind_in_favor_condition : Prop := 0.8 * T = 15
def wind_against_condition : Prop := 1.25 * T = 7
def total_walk_time_condition : Prop := 15 + 7 = 22
def total_time_away_condition : Prop := 32 - 22 = 10
def lake_park_restaurant_condition : Prop := 0.8 * R = 10

-- Proof statement
theorem dante_walk_time :
  wind_in_favor_condition T ∧
  wind_against_condition T ∧
  total_walk_time_condition ∧
  total_time_away_condition ∧
  lake_park_restaurant_condition R →
  R = 12.5 :=
by
  intros
  sorry

end dante_walk_time_l194_194624


namespace average_monthly_income_is_2125_l194_194660

noncomputable def calculate_average_monthly_income (expenses_3_months: ℕ) (expenses_4_months: ℕ) (expenses_5_months: ℕ) (savings_per_year: ℕ) : ℕ :=
  (expenses_3_months * 3 + expenses_4_months * 4 + expenses_5_months * 5 + savings_per_year) / 12

theorem average_monthly_income_is_2125 :
  calculate_average_monthly_income 1700 1550 1800 5200 = 2125 :=
by
  sorry

end average_monthly_income_is_2125_l194_194660


namespace contains_zero_l194_194030

theorem contains_zero (a b c d e f: ℕ) (h1: 1 ≤ a ∧ a ≤ 9) (h2: 1 ≤ b ∧ b ≤ 9) (h3: 0 ≤ c ∧ c ≤ 9) 
  (h4: 1 ≤ d ∧ d ≤ 9) (h5: 0 ≤ e ∧ e ≤ 9) (h6: 0 ≤ f ∧ f ≤ 9) 
  (h7: c ≠ f) (h8: 10^4*a + 10^3*b + 10^2*c + 10^1*d + e + 10^4*a + 10^3*b + 10^2*f + 10^1*d + e = 111111) :
  c = 0 ∨ f = 0 := 
sorry

end contains_zero_l194_194030


namespace inequalities_correct_l194_194741

theorem inequalities_correct (a b : ℝ) (h : a * b > 0) :
  |b| > |a| ∧ |a + b| < |b| := sorry

end inequalities_correct_l194_194741


namespace ned_long_sleeve_shirts_l194_194622

-- Define the conditions
def total_shirts_washed_before_school : ℕ := 29
def short_sleeve_shirts : ℕ := 9
def unwashed_shirts : ℕ := 1

-- Define the proof problem
theorem ned_long_sleeve_shirts (total_shirts_washed_before_school short_sleeve_shirts unwashed_shirts: ℕ) : 
(total_shirts_washed_before_school - unwashed_shirts - short_sleeve_shirts) = 19 :=
by
  -- It is given: 29 total shirts - 1 unwashed shirt = 28 washed shirts
  -- Out of the 28 washed shirts, 9 are short sleeve shirts
  -- Therefore, Ned washed 28 - 9 = 19 long sleeve shirts
  sorry

end ned_long_sleeve_shirts_l194_194622


namespace max_value_sin_cos_combination_l194_194397

theorem max_value_sin_cos_combination :
  ∀ x : ℝ, (5 * Real.sin x + 12 * Real.cos x) ≤ 13 :=
by
  intro x
  sorry

end max_value_sin_cos_combination_l194_194397


namespace trigonometric_relationship_l194_194945

noncomputable def a : ℝ := Real.sin (393 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (55 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (50 * Real.pi / 180)

theorem trigonometric_relationship : a < b ∧ b < c := by
  sorry

end trigonometric_relationship_l194_194945


namespace find_x_l194_194574

-- Define the custom operation on m and n
def operation (m n : ℤ) : ℤ := 2 * m - 3 * n

-- Lean statement of the problem
theorem find_x (x : ℤ) (h : operation x 7 = operation 7 x) : x = 7 :=
by {
  -- Placeholder for the proof
  sorry
}

end find_x_l194_194574


namespace taxi_company_charges_l194_194268

theorem taxi_company_charges
  (X : ℝ)  -- charge for the first 1/5 of a mile
  (C : ℝ)  -- charge for each additional 1/5 of a mile
  (total_charge : ℝ)  -- total charge for an 8-mile ride
  (remaining_distance_miles : ℝ)  -- remaining miles after the first 1/5 mile
  (remaining_increments : ℝ)  -- remaining 1/5 mile increments
  (charge_increments : ℝ)  -- total charge for remaining increments
  (X_val : X = 2.50)
  (C_val : C = 0.40)
  (total_charge_val : total_charge = 18.10)
  (remaining_distance_miles_val : remaining_distance_miles = 7.8)
  (remaining_increments_val : remaining_increments = remaining_distance_miles * 5)
  (charge_increments_val : charge_increments = remaining_increments * C)
  (proof_1: charge_increments = 15.60)
  (proof_2: total_charge - charge_increments = X) : X = 2.50 := 
by
  sorry

end taxi_company_charges_l194_194268


namespace continued_fraction_l194_194925

theorem continued_fraction {w x y : ℕ} (hw : 0 < w) (hx : 0 < x) (hy : 0 < y)
  (h_eq : (97:ℚ) / 19 = w + 1 / (x + 1 / y)) : w + x + y = 16 :=
sorry

end continued_fraction_l194_194925


namespace simplify_fraction_l194_194788

theorem simplify_fraction (x y z : ℕ) (hx : x = 3) (hy : y = 2) (hz : z = 4) : 
  (15 * x^2 * y^4 * z^2) / (9 * x * y^3 * z) = 10 := 
by
  sorry

end simplify_fraction_l194_194788


namespace find_xy_value_l194_194597

theorem find_xy_value (x y z w : ℕ) (h1 : x = w) (h2 : y = z) (h3 : w + w = z * w) (h4 : y = w)
    (h5 : w + w = w * w) (h6 : z = 3) : x * y = 4 := by
  -- Given that w = 2 based on the conditions
  sorry

end find_xy_value_l194_194597


namespace circle_equation_exists_l194_194185

def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (2, -2)
def l (p : ℝ × ℝ) : Prop := p.1 - p.2 + 1 = 0
def is_on_circle (C : ℝ × ℝ) (p : ℝ × ℝ) (r : ℝ) : Prop :=
  (p.1 - C.1)^2 + (p.2 - C.2)^2 = r^2

theorem circle_equation_exists :
  ∃ C : ℝ × ℝ, C.1 - C.2 + 1 = 0 ∧
  (is_on_circle C A 5) ∧
  (is_on_circle C B 5) ∧
  is_on_circle C (-3, -2) 5 :=
sorry

end circle_equation_exists_l194_194185


namespace find_a_l194_194943

-- Definitions of the sets based on given conditions
def A : Set ℝ := { x | x^2 - 4 ≤ 0 }
def B (a : ℝ) : Set ℝ := { x | 2 * x + a ≤ 0 }
def intersectionAB : Set ℝ := { x | -2 ≤ x ∧ x ≤ 1 }

-- The theorem to prove
theorem find_a (a : ℝ) (h : A ∩ B a = intersectionAB) : a = -2 := by
  sorry

end find_a_l194_194943


namespace total_time_l194_194506

theorem total_time {minutes seconds : ℕ} (hmin : minutes = 3450) (hsec : seconds = 7523) :
  ∃ h m s : ℕ, h = 59 ∧ m = 35 ∧ s = 23 :=
by
  sorry

end total_time_l194_194506


namespace largest_integer_solution_l194_194844

theorem largest_integer_solution (x : ℤ) : 
  x < (92 / 21 : ℝ) → ∀ y : ℤ, y < (92 / 21 : ℝ) → y ≤ x :=
by
  sorry

end largest_integer_solution_l194_194844


namespace maximum_value_of_f_l194_194393

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin x + 12 * Real.cos x

theorem maximum_value_of_f : ∃ x : ℝ, f x = 13 :=
by 
  sorry

end maximum_value_of_f_l194_194393


namespace standard_deviation_is_one_l194_194799

def mean : ℝ := 10.5
def value : ℝ := 8.5

theorem standard_deviation_is_one (σ : ℝ) (h : value = mean - 2 * σ) : σ = 1 :=
by {
  sorry
}

end standard_deviation_is_one_l194_194799


namespace wrapping_paper_solution_l194_194880

variable (P1 P2 P3 : ℝ)

def wrapping_paper_problem : Prop :=
  P1 = 2 ∧
  P3 = P1 + P2 ∧
  P1 + P2 + P3 = 7 →
  (P2 / P1) = 3 / 4

theorem wrapping_paper_solution : wrapping_paper_problem P1 P2 P3 :=
by
  sorry

end wrapping_paper_solution_l194_194880


namespace probability_of_observing_change_l194_194162

noncomputable def traffic_light_cycle := 45 + 5 + 45
noncomputable def observable_duration := 5 + 5 + 5
noncomputable def probability_observe_change := observable_duration / (traffic_light_cycle : ℝ)

theorem probability_of_observing_change :
  probability_observe_change = (3 / 19 : ℝ) :=
  by sorry

end probability_of_observing_change_l194_194162


namespace find_ordered_pair_l194_194568

theorem find_ordered_pair :
  ∃ (x y : ℚ), 7 * x - 3 * y = 6 ∧ 4 * x + 5 * y = 23 ∧ 
               x = 99 / 47 ∧ y = 137 / 47 :=
by
  sorry

end find_ordered_pair_l194_194568


namespace sum_of_arithmetic_sequences_l194_194948

theorem sum_of_arithmetic_sequences (n : ℕ) (h : n ≠ 0) :
  (2 * n * (n + 3) = n * (n + 12)) → (n = 6) :=
by
  intro h_eq
  have h_nonzero : n ≠ 0 := h
  sorry

end sum_of_arithmetic_sequences_l194_194948


namespace interval_a_b_l194_194182

noncomputable def f (x : ℝ) : ℝ := |Real.log (x - 1)|

theorem interval_a_b (a b : ℝ) (x1 x2 : ℝ) (h1 : 1 < x1) (h2 : x1 < x2) (h3 : x2 < b) (h4 : f x1 > f x2) :
  a < 2 := 
sorry

end interval_a_b_l194_194182


namespace sum_of_roots_of_quadratic_eq_l194_194753

theorem sum_of_roots_of_quadratic_eq (x : ℝ) :
  (x + 3) * (x - 4) = 18 → (∃ a b : ℝ, x ^ 2 + a * x + b = 0) ∧ (a = -1) ∧ (b = -30) :=
by
  sorry

end sum_of_roots_of_quadratic_eq_l194_194753


namespace relationship_m_n_l194_194102

variable (a b : ℝ)
variable (m n : ℝ)

theorem relationship_m_n (h1 : a > b) (h2 : b > 0) (hm : m = Real.sqrt a - Real.sqrt b) (hn : n = Real.sqrt (a - b)) : m < n := sorry

end relationship_m_n_l194_194102


namespace range_of_M_l194_194439

theorem range_of_M (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z)
  (h1 : x + y + z = 30) (h2 : 3 * x + y - z = 50) :
  120 ≤ 5 * x + 4 * y + 2 * z ∧ 5 * x + 4 * y + 2 * z ≤ 130 :=
by
  -- We would start the proof here by using the given constraints
  sorry

end range_of_M_l194_194439


namespace problem_259_problem_260_l194_194261

theorem problem_259 (x a b : ℝ) (h : x ^ 3 = a * x ^ 2 + b * x) (hx : x ≠ 0) : x ^ 2 = a * x + b :=
by sorry

theorem problem_260 (x a b : ℝ) (h : x ^ 4 = a * x ^ 2 + b) : 
  x ^ 2 = (a + Real.sqrt (a ^ 2 + 4 * b)) / 2 ∨ x ^ 2 = (a - Real.sqrt (a ^ 2 + 4 * b)) / 2 :=
by sorry

end problem_259_problem_260_l194_194261


namespace problem_l194_194212

open Real

theorem problem (x y : ℝ) (h_posx : 0 < x) (h_posy : 0 < y) (h_cond : x + y^(2016) ≥ 1) : 
  x^(2016) + y > 1 - 1/100 :=
by sorry

end problem_l194_194212


namespace probability_three_white_balls_l194_194861

noncomputable def probability_all_white (white black total_drawn : ℕ) : ℚ :=
  (nat.choose white total_drawn : ℚ) / (nat.choose (white + black) total_drawn : ℚ)

theorem probability_three_white_balls :
  probability_all_white 7 8 3 = 1 / 13 :=
by 
  sorry

end probability_three_white_balls_l194_194861


namespace angle_sum_straight_line_l194_194452

theorem angle_sum_straight_line (x : ℝ) (h : 4 * x + x = 180) : x = 36 :=
sorry

end angle_sum_straight_line_l194_194452


namespace find_a_and_b_l194_194148

theorem find_a_and_b (a b : ℝ) :
  {-1, 3} = {x : ℝ | x^2 + a * x + b = 0} ↔ a = -2 ∧ b = -3 :=
by 
  sorry

end find_a_and_b_l194_194148


namespace sin_45_degree_l194_194309

noncomputable section

open Real

theorem sin_45_degree : sin (π / 4) = sqrt 2 / 2 := sorry

end sin_45_degree_l194_194309


namespace cistern_fill_time_l194_194269

theorem cistern_fill_time (hA : ∀ C : ℝ, 0 < C → ∀ t : ℝ, 0 < t → C / t = C / 10) 
                          (hB : ∀ C : ℝ, 0 < C → ∀ t : ℝ, 0 < t → C / t = -(C / 15)) :
  ∀ C : ℝ, 0 < C → ∃ t : ℝ, t = 30 := 
by 
  sorry

end cistern_fill_time_l194_194269


namespace evaluate_pow_l194_194703

theorem evaluate_pow : (-64 : ℝ)^(4/3) = 256 := 
by
  sorry

end evaluate_pow_l194_194703


namespace slopes_of_line_intersecting_ellipse_l194_194271

noncomputable def possible_slopes : Set ℝ := {m : ℝ | m ≤ -1/Real.sqrt 20 ∨ m ≥ 1/Real.sqrt 20}

theorem slopes_of_line_intersecting_ellipse (m : ℝ) (h : ∃ x y, y = m * x - 3 ∧ 4 * x^2 + 25 * y^2 = 100) : 
  m ∈ possible_slopes :=
sorry

end slopes_of_line_intersecting_ellipse_l194_194271


namespace cotton_equals_iron_l194_194855

theorem cotton_equals_iron (cotton_weight : ℝ) (iron_weight : ℝ)
  (h_cotton : cotton_weight = 1)
  (h_iron : iron_weight = 4) :
  (4 / 5) * cotton_weight = (1 / 5) * iron_weight :=
by
  rw [h_cotton, h_iron]
  simp
  sorry

end cotton_equals_iron_l194_194855


namespace shirts_sold_l194_194870

theorem shirts_sold (initial final : ℕ) (h : initial = 49) (h1 : final = 28) : initial - final = 21 :=
sorry

end shirts_sold_l194_194870


namespace anne_remaining_drawings_l194_194551

/-- Given that Anne has 12 markers and each marker lasts for about 1.5 drawings,
    and she has already made 8 drawings, prove that Anne can make 10 more drawings 
    before she runs out of markers. -/
theorem anne_remaining_drawings (markers : ℕ) (drawings_per_marker : ℝ)
    (drawings_made : ℕ) : markers = 12 → drawings_per_marker = 1.5 → drawings_made = 8 →
    (markers * drawings_per_marker - drawings_made = 10) :=
begin
  intros h1 h2 h3,
  rw h1,
  rw h2,
  rw h3,
  norm_num,
  sorry
end

end anne_remaining_drawings_l194_194551


namespace contractor_absent_days_l194_194531

theorem contractor_absent_days (W A : ℕ) : 
  (W + A = 30 ∧ 25 * W - 7.5 * A = 425) → A = 10 :=
by
 sorry

end contractor_absent_days_l194_194531


namespace num_ordered_triples_l194_194902

theorem num_ordered_triples :
  (∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a ∣ b ∧ a ∣ c ∧ a + b + c = 100) :=
  sorry

end num_ordered_triples_l194_194902


namespace infinite_points_on_line_with_positive_rational_coordinates_l194_194563

theorem infinite_points_on_line_with_positive_rational_coordinates :
  ∃ (S : Set (ℚ × ℚ)), (∀ p ∈ S, p.1 + p.2 = 4 ∧ 0 < p.1 ∧ 0 < p.2) ∧ S.Infinite :=
sorry

end infinite_points_on_line_with_positive_rational_coordinates_l194_194563


namespace largest_exterior_angle_l194_194128

theorem largest_exterior_angle (x : ℝ) (h1 : 3 * x + 4 * x + 5 * x = 180) : 180 - 3 * (180 / 12) = 135 :=
by {
  -- Sorry is a placeholder for the actual proof
  sorry
}

end largest_exterior_angle_l194_194128


namespace cost_of_five_plastic_chairs_l194_194494

theorem cost_of_five_plastic_chairs (C T : ℕ) (h1 : 3 * C = T) (h2 : T + 2 * C = 55) : 5 * C = 55 :=
by {
  sorry
}

end cost_of_five_plastic_chairs_l194_194494


namespace problem1_problem2_problem3_l194_194640

-- Proof Problem 1: $A$ and $B$ are not standing together
theorem problem1 : 
  ∃ (n : ℕ), n = 480 ∧ 
  ∀ (students : Fin 6 → String),
    students 0 ≠ "A" ∨ students 1 ≠ "B" :=
sorry

-- Proof Problem 2: $C$ and $D$ must stand together
theorem problem2 : 
  ∃ (n : ℕ), n = 240 ∧ 
  ∀ (students : Fin 6 → String),
    (students 0 = "C" ∧ students 1 = "D") ∨ 
    (students 1 = "C" ∧ students 2 = "D") :=
sorry

-- Proof Problem 3: $E$ is not at the beginning and $F$ is not at the end
theorem problem3 : 
  ∃ (n : ℕ), n = 504 ∧ 
  ∀ (students : Fin 6 → String),
    students 0 ≠ "E" ∧ students 5 ≠ "F" :=
sorry

end problem1_problem2_problem3_l194_194640


namespace inequality_solution_l194_194790

theorem inequality_solution (x : ℝ) (h : x > -4/3) : 2 - 1 / (3 * x + 4) < 5 :=
sorry

end inequality_solution_l194_194790


namespace contractor_absent_days_l194_194527

variable (x y : ℝ) -- x for the number of days worked, y for the number of days absent

-- Conditions
def eng_days := x + y = 30
def total_money := 25 * x - 7.5 * y = 425

-- Theorem
theorem contractor_absent_days (x y : ℝ) (h1 : eng_days x y) (h2 : total_money x y) : y = 10 := 
sorry

end contractor_absent_days_l194_194527


namespace minimum_apples_collected_l194_194285

-- Anya, Vanya, Dania, Sanya, and Tanya each collected an integer percentage of the total number of apples,
-- with all these percentages distinct and greater than zero.
-- Prove that the minimum total number of apples is 20.

theorem minimum_apples_collected :
  ∃ (n : ℕ), (∀ (a v d s t : ℕ), 
    1 ≤ a ∧ 1 ≤ v ∧ 1 ≤ d ∧ 1 ≤ s ∧ 1 ≤ t ∧
    a ≠ v ∧ a ≠ d ∧ a ≠ s ∧ a ≠ t ∧ 
    v ≠ d ∧ v ≠ s ∧ v ≠ t ∧ 
    d ≠ s ∧ d ≠ t ∧ 
    s ≠ t ∧
    a + v + d + s + t = 100) →
  n ≥ 20 :=
by 
  sorry

end minimum_apples_collected_l194_194285


namespace hyperbola_asymptote_eq_l194_194069

-- Define the given hyperbola equation and its asymptote
def hyperbola_eq (a : ℝ) : Prop :=
  ∃ x y : ℝ, (x^2 / a^2) - (y^2 / 4) = 1

def asymptote_eq (a : ℝ) : Prop :=
  ∀ x : ℝ, ∃ y : ℝ, y = (1/2) * x

-- State the main theorem
theorem hyperbola_asymptote_eq :
  (∃ a : ℝ, hyperbola_eq a ∧ asymptote_eq a) →
  (∃ x y : ℝ, (x^2 / 16) - (y^2 / 4) = 1) := 
by
  sorry

end hyperbola_asymptote_eq_l194_194069


namespace inequality_system_solution_l194_194794

theorem inequality_system_solution (x: ℝ) (h1: 5 * x - 2 < 3 * (x + 2)) (h2: (2 * x - 1) / 3 - (5 * x + 1) / 2 <= 1) : 
  -1 ≤ x ∧ x < 4 :=
sorry

end inequality_system_solution_l194_194794


namespace find_random_discount_l194_194520

theorem find_random_discount
  (initial_price : ℝ) (final_price : ℝ) (autumn_discount : ℝ) (loyalty_discount : ℝ) (random_discount : ℝ) :
  initial_price = 230 ∧ final_price = 69 ∧ autumn_discount = 0.25 ∧ loyalty_discount = 0.20 ∧ 
  final_price = initial_price * (1 - autumn_discount) * (1 - loyalty_discount) * (1 - random_discount / 100) →
  random_discount = 50 :=
by
  intros h
  sorry

end find_random_discount_l194_194520


namespace find_David_marks_in_Physics_l194_194884

theorem find_David_marks_in_Physics
  (english_marks : ℕ) (math_marks : ℕ) (chem_marks : ℕ) (biology_marks : ℕ)
  (avg_marks : ℕ) (num_subjects : ℕ)
  (h_english : english_marks = 76)
  (h_math : math_marks = 65)
  (h_chem : chem_marks = 67)
  (h_bio : biology_marks = 85)
  (h_avg : avg_marks = 75) 
  (h_num_subjects : num_subjects = 5) :
  english_marks + math_marks + chem_marks + biology_marks + physics_marks = avg_marks * num_subjects → physics_marks = 82 := 
  sorry

end find_David_marks_in_Physics_l194_194884


namespace vertex_of_parabola_l194_194645

theorem vertex_of_parabola :
  ∀ (x y : ℝ), y = (1 / 3) * (x - 7) ^ 2 + 5 → ∃ h k : ℝ, h = 7 ∧ k = 5 ∧ y = (1 / 3) * (x - h) ^ 2 + k :=
by
  intro x y h
  sorry

end vertex_of_parabola_l194_194645


namespace coffee_on_Thursday_coffee_on_Friday_average_coffee_l194_194806

noncomputable def coffee_consumption (k h : ℝ) : ℝ := k / h

theorem coffee_on_Thursday : coffee_consumption 24 4 = 6 :=
by sorry

theorem coffee_on_Friday : coffee_consumption 24 10 = 2.4 :=
by sorry

theorem average_coffee : 
  (coffee_consumption 24 8 + coffee_consumption 24 4 + coffee_consumption 24 10) / 3 = 3.8 :=
by sorry

end coffee_on_Thursday_coffee_on_Friday_average_coffee_l194_194806


namespace sin_45_eq_1_div_sqrt_2_l194_194340

theorem sin_45_eq_1_div_sqrt_2 : Real.sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_eq_1_div_sqrt_2_l194_194340


namespace sum_mod_7_remainder_l194_194137

def sum_to (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem sum_mod_7_remainder : (sum_to 140) % 7 = 0 :=
by
  sorry

end sum_mod_7_remainder_l194_194137


namespace boards_nailing_l194_194107

variables {x y a b : ℕ}

theorem boards_nailing (h1 : 2 * x + 3 * y = 87)
                       (h2 : 3 * a + 5 * b = 94) :
                       x + y = 30 ∧ a + b = 30 :=
sorry

end boards_nailing_l194_194107


namespace evaluate_neg_64_exp_4_over_3_l194_194715

theorem evaluate_neg_64_exp_4_over_3 : (-64 : ℝ) ^ (4 / 3) = 256 := 
by
  sorry

end evaluate_neg_64_exp_4_over_3_l194_194715


namespace symmetric_point_coordinates_l194_194767

def point_symmetric_to_x_axis (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  match p with
  | (x, y, z) => (x, -y, -z)

theorem symmetric_point_coordinates :
  point_symmetric_to_x_axis (-2, 1, 4) = (-2, -1, -4) := by
  sorry

end symmetric_point_coordinates_l194_194767


namespace algebraic_expression_value_l194_194575

theorem algebraic_expression_value (x y : ℝ) (h : x + 2 * y = 1) :
  (2 * x + 4 * y) / (x^2 + 4 * x * y + 4 * y^2) = 2 :=
by
  sorry

end algebraic_expression_value_l194_194575


namespace eval_neg_pow_l194_194695

theorem eval_neg_pow (a b : ℝ) (h1 : a = (-4)^3) (h2 : b = (-64)) : (-64 : ℝ)^(4/3) = 256 :=
by {
  have h_eq : b = a := by rw h1,
  rw [h2, h_eq],
  have : (a : ℝ)^(4/3) = ((-4)^3)^(4/3) := by rw h1,
  rw this,
  have : ((-4)^3)^(4/3) = (-4)^4 := by norm_num,
  rw this,
  norm_num,
  exact rfl,
}

end eval_neg_pow_l194_194695


namespace yeast_population_correct_l194_194263

noncomputable def yeast_population_estimation 
    (count_per_small_square : ℕ)
    (dimension_large_square : ℝ)
    (dilution_factor : ℝ)
    (thickness : ℝ)
    (total_volume : ℝ) 
    : ℝ :=
    (count_per_small_square:ℝ) / ((dimension_large_square * dimension_large_square * thickness) / 400) * dilution_factor * total_volume

theorem yeast_population_correct:
    yeast_population_estimation 5 1 10 0.1 10 = 2 * 10^9 :=
by
    sorry

end yeast_population_correct_l194_194263


namespace kishore_savings_l194_194852

noncomputable def total_expenses : ℝ :=
  5000 + 1500 + 4500 + 2500 + 2000 + 5200

def percentage_saved : ℝ := 0.10

theorem kishore_savings (salary : ℝ) :
  (total_expenses + percentage_saved * salary) = salary → 
  (percentage_saved * salary = 2077.78) :=
by
  intros h
  rw [← h]
  sorry

end kishore_savings_l194_194852


namespace isosceles_triangle_construction_l194_194683

noncomputable def isosceles_triangle_construction_impossible 
  (hb lb : ℝ) : Prop :=
  ∀ (α β : ℝ), 
  3 * β ≠ α

theorem isosceles_triangle_construction : 
  ∃ (hb lb : ℝ), isosceles_triangle_construction_impossible hb lb :=
sorry

end isosceles_triangle_construction_l194_194683


namespace quadratic_root_k_l194_194926

theorem quadratic_root_k (k : ℝ) : (∃ x : ℝ, x^2 - 2 * x + k = 0 ∧ x = 1) → k = 1 :=
by
  sorry

end quadratic_root_k_l194_194926


namespace TeresaTotalMarks_l194_194119

/-- Teresa's scores in various subjects as given conditions -/
def ScienceScore := 70
def MusicScore := 80
def SocialStudiesScore := 85
def PhysicsScore := 1 / 2 * MusicScore

/-- Total marks Teresa scored in all the subjects -/
def TotalMarks := ScienceScore + MusicScore + SocialStudiesScore + PhysicsScore

/-- Proof statement: The total marks scored by Teresa in all subjects is 275. -/
theorem TeresaTotalMarks : TotalMarks = 275 := by
  sorry

end TeresaTotalMarks_l194_194119


namespace two_pow_gt_square_for_n_ge_5_l194_194245

theorem two_pow_gt_square_for_n_ge_5 (n : ℕ) (hn : n ≥ 5) : 2^n > n^2 :=
sorry

end two_pow_gt_square_for_n_ge_5_l194_194245


namespace quadratic_equation_solution_l194_194115

theorem quadratic_equation_solution :
  ∃ x1 x2 : ℝ, (x1 = (-1 + Real.sqrt 13) / 2 ∧ x2 = (-1 - Real.sqrt 13) / 2 
  ∧ (∀ x : ℝ, x^2 + x - 3 = 0 → x = x1 ∨ x = x2)) :=
sorry

end quadratic_equation_solution_l194_194115


namespace probability_event1_probability_event2_l194_194449

-- Define the balls and the probability distribution
def balls : List ℕ := [1, 2, 3, 4]

-- Two integers x and y are drawn with replacement
def draws_with_replacement : List (ℕ × ℕ) := 
  (balls.product balls)

-- Probability Mass Function for equal probability draws
def pmf := ProbabilityMassFunction.ofFinsetUniform succeeds,
  λ s, s ∈ draws_with_replacement.toFinset

-- Define the events
def event1 (s : ℕ × ℕ) : Prop := (s.fst + s.snd = 5)
def event2 (s : ℕ × ℕ) : Prop := (2 * s.fst + abs (s.fst - s.snd) = 6)

-- Propositions to be proved
theorem probability_event1 : pmf.event (λ s, event1 s) = 1/4 :=
sorry

theorem probability_event2 : pmf.event (λ s, event2 s) = 1/8 :=
sorry

end probability_event1_probability_event2_l194_194449


namespace extreme_points_inequality_l194_194803

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^2 + m * Real.log (1 + x)

-- Given m > 0 and f(x) has extreme points x1 and x2 such that x1 < x2
theorem extreme_points_inequality {m x1 x2 : ℝ} (h_m : m > 0)
    (h_extreme1 : x1 = (-1 - Real.sqrt (1 - 2 * m)) / 2)
    (h_extreme2 : x2 = (-1 + Real.sqrt (1 - 2 * m)) / 2)
    (h_order : x1 < x2) :
    2 * f x2 m > -x1 + 2 * x1 * Real.log 2 := sorry

end extreme_points_inequality_l194_194803


namespace inverse_prop_relation_l194_194577

theorem inverse_prop_relation (y₁ y₂ y₃ : ℝ) :
  (y₁ = (1 : ℝ) / (-1)) →
  (y₂ = (1 : ℝ) / (-2)) →
  (y₃ = (1 : ℝ) / (3)) →
  y₃ > y₂ ∧ y₂ > y₁ :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  constructor
  · norm_num
  · norm_num

end inverse_prop_relation_l194_194577


namespace cos_product_value_l194_194406

open Real

theorem cos_product_value (α : ℝ) (h : sin α = 1 / 3) : 
  cos (π / 4 + α) * cos (π / 4 - α) = 7 / 18 :=
by
  sorry

end cos_product_value_l194_194406


namespace sin_45_eq_one_div_sqrt_two_l194_194317

theorem sin_45_eq_one_div_sqrt_two
  (Q : ℝ × ℝ)
  (h1 : Q = (real.cos (real.pi / 4), real.sin (real.pi / 4)))
  (h2 : Q.2 = real.sin (real.pi / 4)) :
  real.sin (real.pi / 4) = 1 / real.sqrt 2 := 
sorry

end sin_45_eq_one_div_sqrt_two_l194_194317


namespace simplify_complex_expression_l194_194065

theorem simplify_complex_expression (i : ℂ) (h_i : i * i = -1) : 
  (11 - 3 * i) / (1 + 2 * i) = 3 - 5 * i :=
sorry

end simplify_complex_expression_l194_194065


namespace num_colorings_l194_194818

def color := {red, blue, green}

def adjacency (i j : Fin 3 × Fin 3) : Prop :=
  (i.1 = j.1 ∧ (i.2 = j.2 + 1 ∨ i.2 = j.2 - 1)) ∨
  (i.2 = j.2 ∧ (i.1 = j.1 + 1 ∨ i.1 = j.1 - 1))

def valid_coloring (grid : Fin 3 × Fin 3 → color) : Prop :=
  ∀ i j, adjacency i j → grid i ≠ grid j

noncomputable def count_valid_colorings : Nat :=
  Finset.card {c : Fin 3 × Fin 3 → color | valid_coloring c}

theorem num_colorings : count_valid_colorings = 3 :=
sorry

end num_colorings_l194_194818


namespace flour_price_increase_l194_194819

theorem flour_price_increase (x : ℝ) (hx : x > 0) :
  (9600 / (1.5 * x) - 6000 / x = 0.4) :=
by 
  sorry

end flour_price_increase_l194_194819


namespace probability_heart_then_club_l194_194823

noncomputable def numHearts : ℕ := 13
noncomputable def numClubs : ℕ := 13
noncomputable def totalCards (n : ℕ) : ℕ := 52 - n

noncomputable def probabilityFirstHeart : ℚ := numHearts / totalCards 0
noncomputable def probabilitySecondClubGivenFirstHeart : ℚ := numClubs / totalCards 1

theorem probability_heart_then_club :
  (probabilityFirstHeart * probabilitySecondClubGivenFirstHeart) = 13 / 204 :=
by
  sorry

end probability_heart_then_club_l194_194823


namespace number_of_correct_propositions_l194_194103

def f (x b c : ℝ) := x * |x| + b * x + c

def proposition1 (b : ℝ) : Prop :=
  ∀ (x : ℝ), f x b 0 = -f (-x) b 0

def proposition2 (c : ℝ) : Prop :=
  c > 0 → ∃ (x : ℝ), ∀ (y : ℝ), f y 0 c = 0 → y = x

def proposition3 (b c : ℝ) : Prop :=
  ∀ (x : ℝ), f x b c = f (-x) b c + 2 * c

def proposition4 (b c : ℝ) : Prop :=
  ∀ (x₁ x₂ x₃ : ℝ), f x₁ b c = 0 → f x₂ b c = 0 → f x₃ b c = 0 → x₁ = x₂ ∨ x₂ = x₃ ∨ x₁ = x₃

theorem number_of_correct_propositions (b c : ℝ) : 
  1 + (if c > 0 then 1 else 0) + 1 + 0 = 3 :=
  sorry

end number_of_correct_propositions_l194_194103


namespace john_profit_percentage_is_50_l194_194456

noncomputable def profit_percentage
  (P : ℝ)  -- The sum of money John paid for purchasing 30 pens
  (recovered_amount : ℝ)  -- The amount John recovered when he sold 20 pens
  (condition : recovered_amount = P) -- Condition that John recovered the full amount P when he sold 20 pens
  : ℝ := 
  ((P / 20) - (P / 30)) / (P / 30) * 100

theorem john_profit_percentage_is_50
  (P : ℝ)
  (recovered_amount : ℝ)
  (condition : recovered_amount = P) :
  profit_percentage P recovered_amount condition = 50 := 
  by 
  sorry

end john_profit_percentage_is_50_l194_194456


namespace sum_of_distinct_squares_l194_194795

theorem sum_of_distinct_squares:
  ∀ (a b c : ℕ),
  a + b + c = 23 ∧ Nat.gcd a b + Nat.gcd b c + Nat.gcd c a = 9 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 + b^2 + c^2 = 179 ∨ a^2 + b^2 + c^2 = 259 →
  a^2 + b^2 + c^2 = 438 :=
by
  sorry

end sum_of_distinct_squares_l194_194795


namespace sin_45_deg_eq_one_div_sqrt_two_l194_194333

def unit_circle_radius : ℝ := 1

def forty_five_degrees_in_radians : ℝ := (Real.pi / 4)

def cos_45 : ℝ := Real.cos forty_five_degrees_in_radians

def sin_45 : ℝ := Real.sin forty_five_degrees_in_radians

theorem sin_45_deg_eq_one_div_sqrt_two : 
  sin_45 = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_deg_eq_one_div_sqrt_two_l194_194333


namespace sqrt_x_minus_2_real_iff_x_ge_2_l194_194433

theorem sqrt_x_minus_2_real_iff_x_ge_2 (x : ℝ) : (∃ r : ℝ, r * r = x - 2) ↔ x ≥ 2 := by
  sorry

end sqrt_x_minus_2_real_iff_x_ge_2_l194_194433


namespace blocks_to_beach_l194_194770

theorem blocks_to_beach (melt_time_in_minutes : ℕ) (block_length_in_miles : ℚ) (speed_in_miles_per_hour : ℚ)
  (h1 : melt_time_in_minutes = 10)
  (h2 : block_length_in_miles = 1 / 8)
  (h3 : speed_in_miles_per_hour = 12) :
  let melt_time_in_hours := melt_time_in_minutes / 60
      distance_in_miles := speed_in_miles_per_hour * melt_time_in_hours
      blocks := distance_in_miles / block_length_in_miles
  in blocks = 16 :=
by
  let melt_time_in_hours := melt_time_in_minutes / 60
  let distance_in_miles := speed_in_miles_per_hour * melt_time_in_hours
  let blocks := distance_in_miles / block_length_in_miles
  -- Proof steps would go here.
  have : blocks = 16 := sorry
  exact this

end blocks_to_beach_l194_194770


namespace polynomial_pattern_1_polynomial_pattern_2_polynomial_calculation_polynomial_factorization_l194_194989

theorem polynomial_pattern_1 (a b : ℝ) : (a + b) * (a ^ 2 - a * b + b ^ 2) = a ^ 3 + b ^ 3 :=
sorry

theorem polynomial_pattern_2 (a b : ℝ) : (a - b) * (a ^ 2 + a * b + b ^ 2) = a ^ 3 - b ^ 3 :=
sorry

theorem polynomial_calculation (a b : ℝ) : (a + 2 * b) * (a ^ 2 - 2 * a * b + 4 * b ^ 2) = a ^ 3 + 8 * b ^ 3 :=
sorry

theorem polynomial_factorization (a : ℝ) : a ^ 3 - 8 = (a - 2) * (a ^ 2 + 2 * a + 4) :=
sorry

end polynomial_pattern_1_polynomial_pattern_2_polynomial_calculation_polynomial_factorization_l194_194989


namespace set_equality_l194_194458

def M : Set ℝ := {x | x^2 - x > 0}

def N : Set ℝ := {x | 1 / x < 1}

theorem set_equality : M = N := 
by
  sorry

end set_equality_l194_194458


namespace sin_45_eq_sqrt2_div_2_l194_194355

theorem sin_45_eq_sqrt2_div_2 : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by sorry

end sin_45_eq_sqrt2_div_2_l194_194355


namespace jared_annual_salary_l194_194048

def monthly_salary_diploma_holder : ℕ := 4000
def factor_degree_to_diploma : ℕ := 3
def months_in_year : ℕ := 12

theorem jared_annual_salary :
  (factor_degree_to_diploma * monthly_salary_diploma_holder) * months_in_year = 144000 :=
by
  sorry

end jared_annual_salary_l194_194048


namespace initial_volume_of_mixture_l194_194017

theorem initial_volume_of_mixture 
  (V : ℝ)
  (h1 : 0 < V) 
  (h2 : 0.20 * V = 0.15 * (V + 5)) :
  V = 15 :=
by 
  -- proof steps 
  sorry

end initial_volume_of_mixture_l194_194017


namespace cost_per_pound_of_sausages_l194_194771

/-- Jake buys 2-pound packages of sausages. He buys 3 packages. He pays $24. 
To find the cost per pound of sausages. --/
theorem cost_per_pound_of_sausages 
  (pkg_weight : ℕ) 
  (num_pkg : ℕ) 
  (total_cost : ℕ) 
  (cost_per_pound : ℕ) 
  (h_pkg_weight : pkg_weight = 2) 
  (h_num_pkg : num_pkg = 3) 
  (h_total_cost : total_cost = 24) 
  (h_total_weight : num_pkg * pkg_weight = 6) :
  total_cost / (num_pkg * pkg_weight) = cost_per_pound :=
sorry

end cost_per_pound_of_sausages_l194_194771


namespace shopkeeper_profit_percentage_l194_194249

theorem shopkeeper_profit_percentage (C : ℝ) (hC : C > 0) :
  let selling_price := 12 * C
  let cost_price := 10 * C
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = 20 :=
by
  sorry

end shopkeeper_profit_percentage_l194_194249


namespace inequality_solution_set_l194_194064

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def f' (x : ℝ) : ℝ := sorry

axiom deriv_cond : ∀ (x : ℝ), x ≠ 0 → f' x < (2 * f x) / x
axiom zero_points : f (-2) = 0 ∧ f 1 = 0

theorem inequality_solution_set :
  {x : ℝ | x * f x < 0} = { x : ℝ | (-2 ≤ x ∧ x < 0) ∨ (0 < x ∧ x ≤ 1) } :=
sorry

end inequality_solution_set_l194_194064


namespace evaluate_neg_64_exp_4_over_3_l194_194712

theorem evaluate_neg_64_exp_4_over_3 : (-64 : ℝ) ^ (4 / 3) = 256 := 
by
  sorry

end evaluate_neg_64_exp_4_over_3_l194_194712


namespace consecutive_sum_36_unique_l194_194752

def is_consecutive_sum (a b n : ℕ) :=
  (0 < n) ∧ ((n ≥ 2) ∧ (b = a + n - 1) ∧ (2 * a + n - 1) * n = 72)

theorem consecutive_sum_36_unique :
  ∃! n, ∃ a b, is_consecutive_sum a b n :=
by
  sorry

end consecutive_sum_36_unique_l194_194752


namespace total_books_in_bookcase_l194_194918

def num_bookshelves := 8
def num_layers_per_bookshelf := 5
def books_per_layer := 85

theorem total_books_in_bookcase : 
  (num_bookshelves * num_layers_per_bookshelf * books_per_layer) = 3400 := by
  sorry

end total_books_in_bookcase_l194_194918


namespace all_words_synonymous_l194_194601

namespace SynonymousWords

inductive Letter
| a | b | c | d | e | f | g
deriving DecidableEq

open Letter

def transform : Letter → List Letter
| a => [b, c]
| b => [c, d]
| c => [d, e]
| d => [e, f]
| e => [f, g]
| f => [g, a]
| g => [a, b]

def remove_delimiter : List Letter → List Letter
| x :: y :: xs =>
  if x = y then remove_delimiter xs else x :: remove_delimiter (y :: xs)
| xs => xs

def synonymous (w1 w2 : List Letter) : Prop :=
  ∃ n, (remove_delimiter ∘ (List.bind transform^[n])) w1 = w2

theorem all_words_synonymous (w1 w2 : List Letter) : synonymous w1 w2 :=
sorry

end SynonymousWords

end all_words_synonymous_l194_194601


namespace oxen_eat_as_much_as_buffaloes_or_cows_l194_194014

theorem oxen_eat_as_much_as_buffaloes_or_cows
  (B C O : ℝ)
  (h1 : 3 * B = 4 * C)
  (h2 : (15 * B + 8 * O + 24 * C) * 36 = (30 * B + 8 * O + 64 * C) * 18) :
  3 * B = 4 * O :=
by sorry

end oxen_eat_as_much_as_buffaloes_or_cows_l194_194014


namespace total_wolves_l194_194868

theorem total_wolves (x y : ℕ) :
  (x + 2 * y = 20) →
  (4 * x + 3 * y = 55) →
  (x + y = 15) :=
by
  intro h1 h2
  sorry

end total_wolves_l194_194868


namespace evaluate_neg_sixtyfour_exp_four_thirds_l194_194711

theorem evaluate_neg_sixtyfour_exp_four_thirds : (-64 : ℝ) ^ (4/3 : ℝ) = 256 := 
by {
  have h1 : (-64 : ℝ) = (-4 : ℝ) ^ 3, by norm_num,
  have h2 : ((-4 : ℝ) ^ 3) ^ (4/3) = (-4 : ℝ) ^ (3 * (4/3)), by rw [←real.rpow_mul],
  rw [←h1],
  rw [h2],
  norm_num,
  rw [pow_nat_cast],
  norm_num,
  norm_num,
  rw [pow_nat_cast],
  norm_num
}

end evaluate_neg_sixtyfour_exp_four_thirds_l194_194711


namespace approximate_value_correct_l194_194241

noncomputable def P1 : ℝ := (47 / 100) * 1442
noncomputable def P2 : ℝ := (36 / 100) * 1412
noncomputable def result : ℝ := (P1 - P2) + 63

theorem approximate_value_correct : abs (result - 232.42) < 0.01 := 
by
  -- Proof to be completed
  sorry

end approximate_value_correct_l194_194241


namespace cannot_fill_box_exactly_l194_194266

def box_length : ℝ := 70
def box_width : ℝ := 40
def box_height : ℝ := 25
def cube_side : ℝ := 4.5

theorem cannot_fill_box_exactly : 
  ¬ (∃ n : ℕ, n * cube_side^3 = box_length * box_width * box_height ∧
               (∃ x y z : ℕ, x * cube_side = box_length ∧ 
                             y * cube_side = box_width ∧ 
                             z * cube_side = box_height)) :=
by sorry

end cannot_fill_box_exactly_l194_194266


namespace sin_45_degree_l194_194325

theorem sin_45_degree : sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_degree_l194_194325


namespace original_price_l194_194489

theorem original_price (P : ℝ) (final_price : ℝ) (percent_increase : ℝ) (h1 : final_price = 450) (h2 : percent_increase = 0.50) : 
  P + percent_increase * P = final_price → P = 300 :=
by
  sorry

end original_price_l194_194489


namespace prob_both_A_B_prob_exactly_one_l194_194835

def prob_A : ℝ := 0.8
def prob_not_B : ℝ := 0.1
def prob_B : ℝ := 1 - prob_not_B

lemma prob_independent (a b : Prop) : Prop := -- Placeholder for actual independence definition
sorry

-- Given conditions
variables (P_A : ℝ := prob_A) (P_not_B : ℝ := prob_not_B) (P_B : ℝ := prob_B) (indep : ∀ A B, prob_independent A B)

-- Questions translated to Lean statements
theorem prob_both_A_B : P_A * P_B = 0.72 := sorry

theorem prob_exactly_one : (P_A * P_not_B) + ((1 - P_A) * P_B) = 0.26 := sorry

end prob_both_A_B_prob_exactly_one_l194_194835


namespace plate_and_roller_acceleration_l194_194000

noncomputable def m : ℝ := 150
noncomputable def g : ℝ := 10
noncomputable def R : ℝ := 1
noncomputable def r : ℝ := 0.4
noncomputable def alpha : ℝ := Real.arccos 0.68

theorem plate_and_roller_acceleration :
  let sin_alpha_half := Real.sin (alpha / 2)
  sin_alpha_half = 0.4 →
  plate_acceleration == 4 ∧ direction == Real.arcsin 0.4 ∧ rollers_acceleration == 4 :=
by
  sorry

end plate_and_roller_acceleration_l194_194000


namespace find_2a_plus_b_l194_194209

open Real

-- Define the given conditions
variables (a b : ℝ)

-- a and b are acute angles
axiom acute_a : 0 < a ∧ a < π / 2
axiom acute_b : 0 < b ∧ b < π / 2

axiom condition1 : 4 * sin a ^ 2 + 3 * sin b ^ 2 = 1
axiom condition2 : 4 * sin (2 * a) - 3 * sin (2 * b) = 0

-- Define the theorem we want to prove
theorem find_2a_plus_b : 2 * a + b = π / 2 :=
sorry

end find_2a_plus_b_l194_194209


namespace calories_needed_l194_194629

def calories_per_orange : ℕ := 80
def cost_per_orange : ℝ := 1.2
def initial_amount : ℝ := 10
def remaining_amount : ℝ := 4

theorem calories_needed : calories_per_orange * (initial_amount - remaining_amount) / cost_per_orange = 400 := 
by 
  sorry

end calories_needed_l194_194629


namespace compound_percentage_increase_l194_194942

noncomputable def weeklyEarningsAfterRaises (initial : ℝ) (raises : List ℝ) : ℝ :=
  raises.foldl (λ sal raise_rate => sal * (1 + raise_rate / 100)) initial

theorem compound_percentage_increase :
  let initial := 60
  let raises := [10, 15, 12, 8]
  weeklyEarningsAfterRaises initial raises = 91.80864 ∧
  ((weeklyEarningsAfterRaises initial raises - initial) / initial * 100 = 53.0144) :=
by
  sorry

end compound_percentage_increase_l194_194942


namespace consecutive_log_sum_l194_194170

theorem consecutive_log_sum : 
  ∃ c d: ℤ, (c + 1 = d) ∧ (c < Real.logb 5 125) ∧ (Real.logb 5 125 < d) ∧ (c + d = 5) :=
sorry

end consecutive_log_sum_l194_194170


namespace cube_volume_l194_194019

theorem cube_volume
  (s : ℝ) 
  (surface_area_eq : 6 * s^2 = 54) :
  s^3 = 27 := 
by 
  sorry

end cube_volume_l194_194019


namespace hannah_monday_run_l194_194914

-- Definitions of the conditions
def ran_on_wednesday : ℕ := 4816
def ran_on_friday : ℕ := 2095
def extra_on_monday : ℕ := 2089

-- Translations to set the total combined distance and the distance ran on Monday
def combined_distance := ran_on_wednesday + ran_on_friday
def ran_on_monday := combined_distance + extra_on_monday

-- A statement to show she ran 9 kilometers on Monday
theorem hannah_monday_run :
  ran_on_monday = 9000 / 1000 * 1000 := sorry

end hannah_monday_run_l194_194914


namespace range_of_a_l194_194416

def line_intersects_circle (a : ℝ) : Prop :=
  let distance_from_center_to_line := |1 - a| / Real.sqrt 2
  distance_from_center_to_line ≤ Real.sqrt 2

theorem range_of_a :
  {a : ℝ | line_intersects_circle a} = {a : ℝ | -1 ≤ a ∧ a ≤ 3} :=
by
  sorry

end range_of_a_l194_194416


namespace false_proposition_l194_194166

theorem false_proposition :
  ¬ (∀ x : ℕ, (x > 0) → (x - 2)^2 > 0) :=
by
  sorry

end false_proposition_l194_194166


namespace convert_to_rectangular_and_find_line_l194_194127

noncomputable def circle_eq1 (x y : ℝ) : Prop := x^2 + y^2 = 4 * x
noncomputable def circle_eq2 (x y : ℝ) : Prop := x^2 + y^2 + 4 * y = 0
noncomputable def line_eq (x y : ℝ) : Prop := y = -x

theorem convert_to_rectangular_and_find_line :
  (∀ x y : ℝ, circle_eq1 x y → x^2 + y^2 = 4 * x) →
  (∀ x y : ℝ, circle_eq2 x y → x^2 + y^2 + 4 * y = 0) →
  (∀ x y : ℝ, circle_eq1 x y ∧ circle_eq2 x y → line_eq x y)
:=
sorry

end convert_to_rectangular_and_find_line_l194_194127


namespace complement_intersection_l194_194464

open Set

-- Definitions of U, A, and B
def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

-- Proof statement
theorem complement_intersection : 
  ((U \ A) ∩ (U \ B)) = ({0, 2, 4} : Set ℕ) :=
by sorry

end complement_intersection_l194_194464


namespace sufficiency_of_inequality_l194_194581

theorem sufficiency_of_inequality (x : ℝ) (h : x > 5) : x^2 > 25 :=
sorry

end sufficiency_of_inequality_l194_194581


namespace sin_45_eq_one_div_sqrt_two_l194_194313

theorem sin_45_eq_one_div_sqrt_two
  (Q : ℝ × ℝ)
  (h1 : Q = (real.cos (real.pi / 4), real.sin (real.pi / 4)))
  (h2 : Q.2 = real.sin (real.pi / 4)) :
  real.sin (real.pi / 4) = 1 / real.sqrt 2 := 
sorry

end sin_45_eq_one_div_sqrt_two_l194_194313


namespace car_speed_is_80_l194_194008

theorem car_speed_is_80 
  (d : ℝ) (t_delay : ℝ) (v_train_factor : ℝ)
  (t_car t_train : ℝ) (v : ℝ) :
  ((d = 75) ∧ (t_delay = 12.5 / 60) ∧ (v_train_factor = 1.5) ∧ 
   (d = v * t_car) ∧ (d = v_train_factor * v * (t_car - t_delay))) →
  v = 80 := 
sorry

end car_speed_is_80_l194_194008


namespace inequality_proof_l194_194096

variable (a b c d : ℝ)

theorem inequality_proof (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : a + b + c + d = 1) : 
  (1 / (4 * a + 3 * b + c) + 1 / (3 * a + b + 4 * d) + 1 / (a + 4 * c + 3 * d) + 1 / (4 * b + 3 * c + d)) ≥ 2 :=
by
  sorry

end inequality_proof_l194_194096


namespace geometric_sequence_150th_term_l194_194724

-- Given conditions
def a1 : ℤ := 5
def a2 : ℤ := -10

-- Computation of common ratio
def r : ℤ := a2 / a1

-- Definition of the n-th term in geometric sequence
def nth_term (n : ℕ) : ℤ :=
  a1 * r^(n-1)

-- Statement to prove
theorem geometric_sequence_150th_term :
  nth_term 150 = -5 * 2^149 :=
by
  sorry

end geometric_sequence_150th_term_l194_194724


namespace collin_total_petals_l194_194882

variable (collin_flowers initially given_flowers received_flowers each_flower_petals total_petals : ℕ)

-- Conditions as definitions in Lean
def collin_initial_flowers := 25
def ingrid_total_flowers := 33 / 3
def each_flower_petals := 4

-- Collin receives 11 flowers from Ingrid
def received_flowers := ingrid_total_flowers

-- Total flowers Collin has
def collin_flowers := 25 + received_flowers

-- Total petals Collin has
def total_petals := collin_flowers * each_flower_petals

-- Proof that Collin has 144 petals in total
theorem collin_total_petals : total_petals = 144 := by
  sorry

end collin_total_petals_l194_194882


namespace negation_of_exists_sin_gt_one_l194_194126

theorem negation_of_exists_sin_gt_one : 
  (¬ ∃ x : ℝ, Real.sin x > 1) ↔ (∀ x : ℝ, Real.sin x ≤ 1) := 
by
  sorry

end negation_of_exists_sin_gt_one_l194_194126


namespace find_d10_bills_l194_194654

variable (V : Int) (d10 d20 : Int)

-- Given conditions
def spent_money (d10 d20 : Int) : Int := 10 * d10 + 20 * d20

axiom spent_amount : spent_money d10 d20 = 80
axiom more_20_bills : d20 = d10 + 1

-- Question to prove
theorem find_d10_bills : d10 = 2 :=
by {
  -- We mark the theorem to be proven
  sorry
}

end find_d10_bills_l194_194654


namespace problem_1_problem_2_l194_194116

theorem problem_1 (x : ℝ) : (2 * x + 3)^2 = 16 ↔ x = 1/2 ∨ x = -7/2 := by
  sorry

theorem problem_2 (x : ℝ) : x^2 - 4 * x - 3 = 0 ↔ x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7 := by
  sorry

end problem_1_problem_2_l194_194116


namespace calculate_interest_rate_l194_194027

theorem calculate_interest_rate
  (total_investment : ℝ)
  (invested_at_eleven_percent : ℝ)
  (total_interest : ℝ)
  (interest_rate_first_type : ℝ) :
  total_investment = 100000 ∧ 
  invested_at_eleven_percent = 30000 ∧ 
  total_interest = 9.6 → 
  interest_rate_first_type = 9 :=
by
  intros
  sorry

end calculate_interest_rate_l194_194027


namespace length_CD_l194_194487

-- Definitions of the edge lengths provided in the problem
def edge_lengths : Set ℕ := {7, 13, 18, 27, 36, 41}

-- Assumption that AB = 41
def AB := 41
def BC : ℕ := 13
def AC : ℕ := 36

-- Main theorem to prove that CD = 13
theorem length_CD (AB BC AC : ℕ) (edges : Set ℕ) (hAB : AB = 41) (hedges : edges = edge_lengths) :
  ∃ (CD : ℕ), CD ∈ edges ∧ CD = 13 :=
by
  sorry

end length_CD_l194_194487


namespace remainder_3_pow_2040_mod_11_l194_194848

theorem remainder_3_pow_2040_mod_11 : (3 ^ 2040) % 11 = 1 := by
  have h1 : 3 % 11 = 3 := by norm_num
  have h2 : (3 ^ 2) % 11 = 9 := by norm_num
  have h3 : (3 ^ 3) % 11 = 5 := by norm_num
  have h4 : (3 ^ 4) % 11 = 4 := by norm_num
  have h5 : (3 ^ 5) % 11 = 1 := by norm_num
  have h_mod : 2040 % 5 = 0 := by norm_num
  sorry

end remainder_3_pow_2040_mod_11_l194_194848


namespace total_price_correct_l194_194131

-- Definitions based on given conditions
def basic_computer_price : ℝ := 2125
def enhanced_computer_price : ℝ := 2125 + 500
def printer_price (P : ℝ) := P = 1/8 * (enhanced_computer_price + P)

-- Statement to prove the total price of the basic computer and printer
theorem total_price_correct (P : ℝ) (h : printer_price P) : 
  basic_computer_price + P = 2500 :=
by
  sorry

end total_price_correct_l194_194131


namespace min_value_of_expression_l194_194459

open Real

theorem min_value_of_expression {a b c d e f : ℝ} (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f)
    (h_sum : a + b + c + d + e + f = 10) :
    (∃ x, x = 44.1 ∧ ∀ y, y = 1 / a + 4 / b + 9 / c + 16 / d + 25 / e + 36 / f → x ≤ y) :=
sorry

end min_value_of_expression_l194_194459


namespace smallest_N_l194_194545

-- Definitions for conditions
variable (a b c : ℕ) (N : ℕ)

-- Define the conditions for the given problem
def valid_block (a b c : ℕ) : Prop :=
  (a - 1) * (b - 1) * (c - 1) = 252

def block_volume (a b c : ℕ) : ℕ := a * b * c

-- The target theorem to be proved
theorem smallest_N (h : valid_block a b c) : N = 224 :=
  sorry

end smallest_N_l194_194545


namespace largest_prime_divisor_in_range_l194_194176

theorem largest_prime_divisor_in_range (n : ℕ) (h1 : 1000 ≤ n) (h2 : n ≤ 1100) :
  ∃ p, Prime p ∧ p ≤ Int.floor (Real.sqrt n) ∧ 
  (∀ q, Prime q ∧ q ≤ Int.floor (Real.sqrt n) → q ≤ p) :=
sorry

end largest_prime_divisor_in_range_l194_194176


namespace num_outfits_l194_194957

-- Define the number of trousers, shirts, and jackets available
def num_trousers : Nat := 5
def num_shirts : Nat := 6
def num_jackets : Nat := 4

-- Define the main theorem
theorem num_outfits (t : Nat) (s : Nat) (j : Nat) (ht : t = num_trousers) (hs : s = num_shirts) (hj : j = num_jackets) :
  t * s * j = 120 :=
by 
  rw [ht, hs, hj]
  exact rfl

end num_outfits_l194_194957


namespace intersection_points_of_graph_and_line_l194_194427

theorem intersection_points_of_graph_and_line (f : ℝ → ℝ) :
  (∀ x : ℝ, f x ≠ my_special_value) → (∀ x₁ x₂ : ℝ, f x₁ = f x₂ → x₁ = x₂) →
  ∃! x : ℝ, x = 1 ∧ ∃ y : ℝ, y = f x :=
by
  sorry

end intersection_points_of_graph_and_line_l194_194427


namespace basic_printer_total_price_l194_194971

theorem basic_printer_total_price (C P : ℝ) (hC : C = 1500) (hP : P = (1/3) * (C + 500 + P)) : C + P = 2500 := 
by
  sorry

end basic_printer_total_price_l194_194971


namespace problem_statement_l194_194442

theorem problem_statement (a b : ℝ) (h1 : a - b = 5) (h2 : a * b = 2) : a^2 + b^2 = 29 := 
by
  sorry

end problem_statement_l194_194442


namespace fraction_to_decimal_l194_194887

theorem fraction_to_decimal : (58 : ℚ) / 160 = 0.3625 := 
by sorry

end fraction_to_decimal_l194_194887


namespace exponent_subtraction_l194_194909

theorem exponent_subtraction (a : ℝ) (m n : ℕ) (h1 : a^m = 6) (h2 : a^n = 2) : a^(m - n) = 3 := by
  sorry

end exponent_subtraction_l194_194909


namespace determine_t_l194_194214

theorem determine_t (t : ℝ) : 
  (3 * t - 9) * (4 * t - 3) = (4 * t - 16) * (3 * t - 9) → t = 7.8 :=
by
  intros h
  sorry

end determine_t_l194_194214


namespace minimum_f_zero_iff_t_is_2sqrt2_l194_194417

noncomputable def f (x t : ℝ) : ℝ := 4 * x^4 - 6 * t * x^3 + (2 * t + 6) * x^2 - 3 * t * x + 1

theorem minimum_f_zero_iff_t_is_2sqrt2 :
  (∀ x > 0, f x t ≥ 0) ∧ (∃ x > 0, f x t = 0) ↔ t = 2 * Real.sqrt 2 := 
sorry

end minimum_f_zero_iff_t_is_2sqrt2_l194_194417


namespace cities_real_distance_l194_194955

def map_scale := 7 -- number of centimeters representing 35 kilometers
def real_distance_equiv := 35 -- number of kilometers that corresponds to map_scale

def centimeters_per_kilometer := real_distance_equiv / map_scale -- kilometers per centimeter

def distance_on_map := 49 -- number of centimeters cities are separated by on the map

theorem cities_real_distance : distance_on_map * centimeters_per_kilometer = 245 :=
by
  sorry

end cities_real_distance_l194_194955


namespace sin_45_eq_l194_194342

noncomputable def sin_45_degrees := Real.sin (π / 4)

theorem sin_45_eq : sin_45_degrees = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_eq_l194_194342


namespace min_value_expression_l194_194590

variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)

theorem min_value_expression : (1 + b / a) * (4 * a / b) ≥ 9 :=
sorry

end min_value_expression_l194_194590


namespace mrs_wilsborough_vip_tickets_l194_194468

theorem mrs_wilsborough_vip_tickets:
  let S := 500 -- Initial savings
  let PVIP := 100 -- Price per VIP ticket
  let preg := 50 -- Price per regular ticket
  let nreg := 3 -- Number of regular tickets
  let R := 150 -- Remaining savings after purchase
  
  -- The total amount spent on tickets is S - R
  S - R = PVIP * 2 + preg * nreg := 
by sorry

end mrs_wilsborough_vip_tickets_l194_194468


namespace candy_division_l194_194611

theorem candy_division 
  (total_candy : ℕ)
  (total_bags : ℕ)
  (candies_per_bag : ℕ)
  (chocolate_heart_bags : ℕ)
  (fruit_jelly_bags : ℕ)
  (caramel_chew_bags : ℕ) 
  (H1 : total_candy = 260)
  (H2 : total_bags = 13)
  (H3 : candies_per_bag = total_candy / total_bags)
  (H4 : chocolate_heart_bags = 4)
  (H5 : fruit_jelly_bags = 3)
  (H6 : caramel_chew_bags = total_bags - chocolate_heart_bags - fruit_jelly_bags)
  (H7 : candies_per_bag = 20) :
  (chocolate_heart_bags * candies_per_bag) + 
  (fruit_jelly_bags * candies_per_bag) + 
  (caramel_chew_bags * candies_per_bag) = 260 :=
sorry

end candy_division_l194_194611


namespace f_has_one_zero_l194_194564

noncomputable def f (x : ℝ) : ℝ := 2 * x - 5 - Real.log x

theorem f_has_one_zero : ∃! x : ℝ, x > 0 ∧ f x = 0 :=
by
  sorry

end f_has_one_zero_l194_194564


namespace average_score_after_19_innings_l194_194254

/-
  Problem Statement:
  Prove that the cricketer's average score after 19 innings is 24,
  given that scoring 96 runs in the 19th inning increased his average by 4.
-/

theorem average_score_after_19_innings :
  ∀ A : ℕ,
  (18 * A + 96) / 19 = A + 4 → A + 4 = 24 :=
by
  intros A h
  /- Skipping proof by adding "sorry" -/
  sorry

end average_score_after_19_innings_l194_194254


namespace cherries_in_mix_l194_194669

theorem cherries_in_mix (total_fruit : ℕ) (blueberries : ℕ) (raspberries : ℕ) (cherries : ℕ) 
  (H1 : total_fruit = 300)
  (H2: raspberries = 3 * blueberries)
  (H3: cherries = 5 * blueberries)
  (H4: total_fruit = blueberries + raspberries + cherries) : cherries = 167 :=
by
  sorry

end cherries_in_mix_l194_194669


namespace sin_45_eq_sqrt2_div_2_l194_194362

theorem sin_45_eq_sqrt2_div_2 :
  Real.sin (π / 4) = Real.sqrt 2 / 2 := 
by
  sorry

end sin_45_eq_sqrt2_div_2_l194_194362


namespace prove_smallest_solution_l194_194571

noncomputable def smallest_solution : ℝ :=
  if h : 0 ≤ (3 - Real.sqrt 17) / 2 then min ((3 - Real.sqrt 17) / 2) 1
  else (3 - Real.sqrt 17) / 2  -- Assumption as sqrt(17) > 3, so (3 - sqrt(17))/2 < 0

theorem prove_smallest_solution :
  ∃ x : ℝ, (x * |x| = 3 * x - 2) ∧ 
           (∀ y : ℝ, (y * |y| = 3 * y - 2) → x ≤ y) ∧
           x = (3 - Real.sqrt 17) / 2 :=
sorry

end prove_smallest_solution_l194_194571


namespace mary_number_l194_194466

-- Definitions for conditions
def has_factor_150 (m : ℕ) : Prop := 150 ∣ m
def is_multiple_of_45 (m : ℕ) : Prop := 45 ∣ m
def in_range (m : ℕ) : Prop := 1000 < m ∧ m < 3000

-- Theorem stating that Mary's number is one of {1350, 1800, 2250, 2700} given the conditions
theorem mary_number 
  (m : ℕ) 
  (h1 : has_factor_150 m)
  (h2 : is_multiple_of_45 m)
  (h3 : in_range m) :
  m = 1350 ∨ m = 1800 ∨ m = 2250 ∨ m = 2700 :=
sorry

end mary_number_l194_194466


namespace solution_set_of_inequality_l194_194970

theorem solution_set_of_inequality (x : ℝ) : 
  (2 * x - 1) / (x + 2) > 1 ↔ x < -2 ∨ x > 3 :=
by
  sorry

end solution_set_of_inequality_l194_194970


namespace inequality_solution_l194_194791

theorem inequality_solution (x : ℝ) (h : x > -4/3) : 2 - 1 / (3 * x + 4) < 5 :=
sorry

end inequality_solution_l194_194791


namespace sin_45_eq_sqrt2_div_2_l194_194319

theorem sin_45_eq_sqrt2_div_2 : Real.sin (π / 4) = Real.sqrt 2 / 2 := 
sorry

end sin_45_eq_sqrt2_div_2_l194_194319


namespace isabella_purchases_l194_194606

def isabella_items_total (alexis_pants alexis_dresses isabella_pants isabella_dresses : ℕ) : ℕ :=
  isabella_pants + isabella_dresses

theorem isabella_purchases
  (alexis_pants : ℕ) (alexis_dresses : ℕ)
  (h_pants : alexis_pants = 21)
  (h_dresses : alexis_dresses = 18)
  (h_ratio : ∀ (x : ℕ), alexis_pants = 3 * x → alexis_dresses = 3 * x):
  isabella_items_total (21 / 3) (18 / 3) = 13 :=
by
  sorry

end isabella_purchases_l194_194606


namespace constants_inequality_value_l194_194944

theorem constants_inequality_value
  (a b c d : ℝ)
  (h1 : a < b)
  (h2 : b < c)
  (h3 : ∀ x, (1 ≤ x ∧ x ≤ 5) ∨ (24 ≤ x ∧ x ≤ 26) ∨ x < -4 ↔ (x - a) * (x - b) * (x - c) / (x - d) ≤ 0) :
  a + 3 * b + 3 * c + 4 * d = 72 :=
sorry

end constants_inequality_value_l194_194944


namespace sin_45_eq_one_div_sqrt_two_l194_194312

theorem sin_45_eq_one_div_sqrt_two
  (Q : ℝ × ℝ)
  (h1 : Q = (real.cos (real.pi / 4), real.sin (real.pi / 4)))
  (h2 : Q.2 = real.sin (real.pi / 4)) :
  real.sin (real.pi / 4) = 1 / real.sqrt 2 := 
sorry

end sin_45_eq_one_div_sqrt_two_l194_194312


namespace shaded_squares_percentage_l194_194508

theorem shaded_squares_percentage : 
  let grid_size := 6
  let total_squares := grid_size * grid_size
  let shaded_squares := total_squares / 2
  (shaded_squares / total_squares) * 100 = 50 :=
by
  /- Definitions and conditions -/
  let grid_size := 6
  let total_squares := grid_size * grid_size
  let shaded_squares := total_squares / 2

  /- Required proof statement -/
  have percentage_shaded : (shaded_squares / total_squares) * 100 = 50 := sorry

  /- Return the proof -/
  exact percentage_shaded

end shaded_squares_percentage_l194_194508


namespace negation_proposition_l194_194229

theorem negation_proposition :
  ¬ (∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ ∃ x0 : ℝ, x0^2 - 2*x0 + 4 > 0 :=
by
  sorry

end negation_proposition_l194_194229


namespace min_orders_to_minimize_spent_l194_194013

-- Definitions for the given conditions
def original_price (n p : ℕ) : ℕ := n * p
def discounted_price (T : ℕ) : ℕ := (3 * T) / 5  -- Equivalent to 0.6 * T, using integer math

-- Define the conditions
theorem min_orders_to_minimize_spent 
  (n p : ℕ)
  (h1 : n = 42)
  (h2 : p = 48)
  : ∃ m : ℕ, m = 3 :=
by 
  sorry

end min_orders_to_minimize_spent_l194_194013


namespace Isabella_total_items_l194_194607

theorem Isabella_total_items (A_pants A_dresses I_pants I_dresses : ℕ) 
  (h1 : A_pants = 3 * I_pants) 
  (h2 : A_dresses = 3 * I_dresses)
  (h3 : A_pants = 21) 
  (h4 : A_dresses = 18) : 
  I_pants + I_dresses = 13 :=
by
  -- Proof goes here
  sorry

end Isabella_total_items_l194_194607


namespace sin_45_eq_l194_194345

noncomputable def sin_45_degrees := Real.sin (π / 4)

theorem sin_45_eq : sin_45_degrees = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_eq_l194_194345


namespace math_problem_l194_194068

variable {f : ℝ → ℝ}

theorem math_problem (h1 : ∀ x y : ℝ, f (x + y) = f x + f y)
                     (h2 : ∀ x : ℝ, x > 0 → f x > 0)
                     (h3 : f 1 = 2) :
                     f 0 = 0 ∧
                     (∀ x : ℝ, f (-x) = -f x) ∧
                     (∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2) ∧
                     (∃ a : ℝ, f (2 - a) = 6 ∧ a = -1) := 
by
  sorry

end math_problem_l194_194068


namespace largest_divisor_expression_l194_194081

theorem largest_divisor_expression (y : ℤ) (h : y % 2 = 1) : 
  4320 ∣ (15 * y + 3) * (15 * y + 9) * (10 * y + 10) :=
sorry  

end largest_divisor_expression_l194_194081


namespace molecular_weight_of_10_moles_l194_194501

-- Define the molecular weight of a compound as a constant
def molecular_weight (compound : Type) : ℝ := 840

-- Prove that the molecular weight of 10 moles of the compound is the same as the molecular weight of 1 mole of the compound
theorem molecular_weight_of_10_moles (compound : Type) :
  molecular_weight compound = 840 :=
by
  -- Proof
  sorry

end molecular_weight_of_10_moles_l194_194501


namespace rainfall_hydroville_2012_l194_194598

-- Define the average monthly rainfall for each year
def avg_rainfall_2010 : ℝ := 37.2
def avg_rainfall_2011 : ℝ := avg_rainfall_2010 + 3.5
def avg_rainfall_2012 : ℝ := avg_rainfall_2011 - 1.2

-- Define the total rainfall for 2012
def total_rainfall_2012 : ℝ := 12 * avg_rainfall_2012

-- The theorem to be proved
theorem rainfall_hydroville_2012 : total_rainfall_2012 = 474 := by
  sorry

end rainfall_hydroville_2012_l194_194598


namespace sum_of_adjacent_to_14_l194_194968

/-!
# Problem Statement
The positive integer divisors of 294, except 1, are arranged around a circle so that every pair of adjacent integers has a common factor greater than 1. Prove that the sum of the two integers adjacent to 14 is 140.
-/

noncomputable def divisors_except_one (n : ℕ) : List ℕ :=
  (List.range (n+1)).filter (λ x, x > 1 ∧ n % x = 0)

noncomputable def arranged_circle (n : ℕ) : List ℕ := 
  [2, 3, 7, 6, 14, 21, 42, 49, 98, 147, 294] -- manually derived

theorem sum_of_adjacent_to_14 : 
  let divisors := divisors_except_one 294 in
  let adjacent_to_14 := List.filter (λ x, Nat.gcd 14 x > 1) (arranged_circle 294) in
  adjacent_to_14 = [42, 98] →
  adjacent_to_14.sum = 140 :=
by
  intro divisors adjacent_to_14 H
  have h1 : adjacent_to_14 = [42, 98] := H
  rw h1
  norm_num
  rfl

example : sum_of_adjacent_to_14 :=
by
  unfold sum_of_adjacent_to_14
  rw List.filter_eq_of_sublist
  sorry

end sum_of_adjacent_to_14_l194_194968


namespace b_in_terms_of_a_axis_of_symmetry_and_a_value_range_of_a_l194_194054

-- Definitions based on problem conditions
def parabola (a b x : ℝ) : ℝ := a * x^2 + b * x
def passes_through_A (a b : ℝ) : Prop := parabola a b 3 = 3
def points_on_parabola (a b x1 x2 : ℝ) : Prop := x1 < x2 ∧ x1 + x2 = 2
def equal_y_values (a b x1 x2 : ℝ) : Prop := parabola a b x1 = parabola a b x2
def less_than_y_values (a b x1 x2 : ℝ) : Prop := parabola a b x1 < parabola a b x2

-- 1) Express b in terms of a
theorem b_in_terms_of_a (a : ℝ) (h : passes_through_A a (1 - 3 * a)) : True := sorry

-- 2) Axis of symmetry and the value of a when y1 = y2
theorem axis_of_symmetry_and_a_value (a : ℝ) (x1 x2 : ℝ) 
    (h1 : points_on_parabola a (1 - 3 * a) x1 x2)
    (h2 : equal_y_values a (1 - 3 * a) x1 x2) 
    : a = 1 ∧ -1 / 2 * (1 - 3 * a) / a = 1 := sorry

-- 3) Range of values for a when y1 < y2
theorem range_of_a (a x1 x2 : ℝ) 
    (h1 : points_on_parabola a (1 - 3 * a) x1 x2)
    (h2 : less_than_y_values a (1 - 3 * a) x1 x2) 
    (h3 : a ≠ 0) : 0 < a ∧ a < 1 := sorry

end b_in_terms_of_a_axis_of_symmetry_and_a_value_range_of_a_l194_194054


namespace evaluate_pow_l194_194706

theorem evaluate_pow : (-64 : ℝ)^(4/3) = 256 := 
by
  sorry

end evaluate_pow_l194_194706


namespace sin_45_eq_sqrt2_div_2_l194_194365

theorem sin_45_eq_sqrt2_div_2 :
  Real.sin (π / 4) = Real.sqrt 2 / 2 := 
by
  sorry

end sin_45_eq_sqrt2_div_2_l194_194365


namespace sin_45_deg_eq_l194_194377

noncomputable def sin_45_deg : ℝ :=
  real.sin (real.pi / 4)

theorem sin_45_deg_eq : sin_45_deg = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_deg_eq_l194_194377


namespace average_population_is_1000_l194_194231

-- Define the populations of the villages.
def populations : List ℕ := [803, 900, 1100, 1023, 945, 980, 1249]

-- Define the number of villages.
def num_villages : ℕ := 7

-- Define the total population.
def total_population (pops : List ℕ) : ℕ :=
  pops.foldl (λ acc x => acc + x) 0

-- Define the average population computation.
def average_population (pops : List ℕ) (n : ℕ) : ℕ :=
  total_population pops / n

-- Prove that the average population of the 7 villages is 1000.
theorem average_population_is_1000 :
  average_population populations num_villages = 1000 := by
  -- Proof omitted.
  sorry

end average_population_is_1000_l194_194231


namespace sin_45_degree_l194_194301

def Q : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), real.sin (real.pi / 4))
def E : (x:ℝ) × (y:ℝ) := (real.cos (real.pi / 4), 0)
def O : (x:ℝ) × (y:ℝ) := (0,0)
def OQ : ℝ := real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2)

theorem sin_45_degree : ∃ x: ℝ, x = real.sin (real.pi / 4) ∧ x = real.sqrt 2 / 2 :=
by sorry

end sin_45_degree_l194_194301


namespace min_sum_of_inverses_l194_194461

theorem min_sum_of_inverses 
  (x y z p q r : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) 
  (h_sum : x + y + z + p + q + r = 10) :
  (1 / x + 9 / y + 4 / z + 25 / p + 16 / q + 36 / r) = 44.1 :=
sorry

end min_sum_of_inverses_l194_194461


namespace postman_pete_mileage_l194_194111

theorem postman_pete_mileage :
  let initial_steps := 30000
  let resets := 72
  let final_steps := 45000
  let steps_per_mile := 1500
  let steps_per_full_cycle := 99999 + 1
  let total_steps := initial_steps + resets * steps_per_full_cycle + final_steps
  total_steps / steps_per_mile = 4850 := 
by 
  sorry

end postman_pete_mileage_l194_194111


namespace area_of_given_trapezium_l194_194992

def area_of_trapezium (a b h : ℕ) : ℕ :=
  (1 / 2) * (a + b) * h

theorem area_of_given_trapezium :
  area_of_trapezium 20 18 25 = 475 :=
by
  sorry

end area_of_given_trapezium_l194_194992


namespace initial_population_l194_194603

theorem initial_population (P : ℝ) (h1 : 1.20 * P = P_1) (h2 : 0.96 * P = P_2) (h3 : P_2 = 9600) : P = 10000 :=
by
  sorry

end initial_population_l194_194603


namespace sin_45_eq_1_div_sqrt_2_l194_194339

theorem sin_45_eq_1_div_sqrt_2 : Real.sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_eq_1_div_sqrt_2_l194_194339


namespace value_range_of_f_l194_194972

open Set

def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem value_range_of_f : {y : ℝ | ∃ x ∈ Icc (-2 : ℝ) (2 : ℝ), f x = y} = Icc (-1 : ℝ) 8 := 
by
  sorry

end value_range_of_f_l194_194972


namespace blue_more_than_white_l194_194521

theorem blue_more_than_white :
  ∃ (B R : ℕ), (B > 16) ∧ (R = 2 * B) ∧ (B + R + 16 = 100) ∧ (B - 16 = 12) :=
sorry

end blue_more_than_white_l194_194521


namespace price_of_book_l194_194106

variables (D B : ℝ)

def younger_brother : ℝ := 10

theorem price_of_book 
  (h1 : D = 1/2 * (B + younger_brother))
  (h2 : B = 1/3 * (D + younger_brother)) : 
  D + B + younger_brother = 24 := 
sorry

end price_of_book_l194_194106


namespace abc_sub_c_minus_2023_eq_2023_l194_194922

theorem abc_sub_c_minus_2023_eq_2023 (a b c : ℝ) (h : a * b = 1) : 
  a * b * c - (c - 2023) = 2023 := 
by sorry

end abc_sub_c_minus_2023_eq_2023_l194_194922


namespace multiple_of_kids_finishing_early_l194_194015

-- Definitions based on conditions
def num_10_percent_kids (total_kids : ℕ) : ℕ := (total_kids * 10) / 100

def num_remaining_kids (total_kids kids_less_6 kids_more_14 : ℕ) : ℕ := total_kids - kids_less_6 - kids_more_14

def num_multiple_finishing_less_8 (total_kids : ℕ) (multiple : ℕ) : ℕ := multiple * num_10_percent_kids total_kids

-- Main theorem statement
theorem multiple_of_kids_finishing_early 
  (total_kids : ℕ)
  (h_total_kids : total_kids = 40)
  (kids_more_14 : ℕ)
  (h_kids_more_14 : kids_more_14 = 4)
  (h_1_6_remaining : kids_more_14 = num_remaining_kids total_kids (num_10_percent_kids total_kids) kids_more_14 / 6)
  : (num_multiple_finishing_less_8 total_kids 3) = (total_kids - num_10_percent_kids total_kids - kids_more_14) := 
by 
  sorry

end multiple_of_kids_finishing_early_l194_194015


namespace ambika_candles_count_l194_194678

-- Definitions
def Aniyah_candles (A : ℕ) : ℕ := 6 * A
def combined_candles (A : ℕ) : ℕ := A + Aniyah_candles A

-- Problem Statement:
theorem ambika_candles_count : ∃ A : ℕ, combined_candles A = 28 ∧ A = 4 :=
by
  sorry

end ambika_candles_count_l194_194678


namespace find_x_l194_194858

variable (P T S : Point)
variable (angle_PTS angle_TSR x : ℝ)
variable (reflector : Point)

-- Given conditions
axiom angle_PTS_is_90 : angle_PTS = 90
axiom angle_TSR_is_26 : angle_TSR = 26

-- Proof problem
theorem find_x : x = 32 := by
  sorry

end find_x_l194_194858


namespace quadratic_inequality_min_value_l194_194408

noncomputable def min_value (a b: ℝ) : ℝ := 2 * a^2 + b^2

theorem quadratic_inequality_min_value
  (a b: ℝ) (hx: ∀ x : ℝ, a * x^2 + 2 * x + b ≥ 0)
  (x0: ℝ) (hx0: a * x0^2 + 2 * x0 + b = 0) :
  a > b → min_value a b = 2 * Real.sqrt 2 := 
sorry

end quadratic_inequality_min_value_l194_194408


namespace range_of_a_l194_194735

variable {x a : ℝ}

def p (x : ℝ) := x^2 - 8 * x - 20 > 0
def q (a : ℝ) (x : ℝ) := x^2 - 2 * x + 1 - a^2 > 0

theorem range_of_a (h₀ : ∀ x, p x → q a x) (h₁ : a > 0) : 0 < a ∧ a ≤ 3 := 
by 
  sorry

end range_of_a_l194_194735


namespace ratio_of_inscribed_squares_l194_194378

-- Definitions of the conditions
def right_triangle_sides (a b c : ℕ) : Prop := a = 6 ∧ b = 8 ∧ c = 10 ∧ a^2 + b^2 = c^2

def inscribed_square_1 (x : ℚ) : Prop := x = 18 / 7

def inscribed_square_2 (y : ℚ) : Prop := y = 32 / 7

-- Statement of the problem
theorem ratio_of_inscribed_squares (x y : ℚ) : right_triangle_sides 6 8 10 ∧ inscribed_square_1 x ∧ inscribed_square_2 y → (x / y) = 9 / 16 :=
by
  sorry

end ratio_of_inscribed_squares_l194_194378


namespace find_a_for_symmetric_and_parallel_lines_l194_194587

theorem find_a_for_symmetric_and_parallel_lines :
  ∃ (a : ℝ), (∀ (x y : ℝ), y = a * x + 3 ↔ x = a * y + 3) ∧ (∀ (x y : ℝ), x + 2 * y - 1 = 0 ↔ x = a * y + 3) ∧ ∃ (a : ℝ), a = -2 := 
sorry

end find_a_for_symmetric_and_parallel_lines_l194_194587


namespace sum_of_squares_inequality_l194_194062

theorem sum_of_squares_inequality (a b c : ℝ) : a^2 + b^2 + c^2 ≥ (1/3)*(a + b + c)^2 := sorry

end sum_of_squares_inequality_l194_194062


namespace min_max_calculation_l194_194777

theorem min_max_calculation
  (p q r s : ℝ)
  (h1 : p + q + r + s = 8)
  (h2 : p^2 + q^2 + r^2 + s^2 = 20) :
  -32 ≤ 5 * (p^3 + q^3 + r^3 + s^3) - (p^4 + q^4 + r^4 + s^4) ∧
  5 * (p^3 + q^3 + r^3 + s^3) - (p^4 + q^4 + r^4 + s^4) ≤ 12 :=
sorry

end min_max_calculation_l194_194777


namespace tangent_slope_through_origin_l194_194761

noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := x^a + 1

theorem tangent_slope_through_origin (a : ℝ) (h : curve a 1 = 2) 
  (tangent_passing_through_origin : ∀ y, (y - 2 = a * (1 - 0)) → y = 0): a = 2 := 
sorry

end tangent_slope_through_origin_l194_194761


namespace inequality_proof_l194_194733

variable (m : ℕ) (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1)

theorem inequality_proof :
    (m > 0) →
    (x^m / ((1 + y) * (1 + z)) + y^m / ((1 + x) * (1 + z)) + z^m / ((1 + x) * (1 + y)) >= 3/4) :=
by
  intro hm_pos
  -- Proof skipped
  sorry

end inequality_proof_l194_194733


namespace shifted_function_expression_l194_194414

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  Real.sin (ω * x + Real.pi / 3)

theorem shifted_function_expression (ω : ℝ) (h : ℝ) (x : ℝ) (h_positive : ω > 0) (h_period : Real.pi = 2 * Real.pi / ω) :
  f ω (x + h) = Real.cos (2 * x) :=
by
  -- We assume h = π/12, ω = 2
  have ω_val : ω = 2 := by sorry
  have h_val : h = Real.pi / 12 := by sorry
  rw [ω_val, h_val]
  sorry

end shifted_function_expression_l194_194414


namespace find_smaller_number_l194_194498

variable (x y : ℕ)

theorem find_smaller_number (h1 : ∃ k : ℕ, x = 2 * k ∧ y = 5 * k) (h2 : x + y = 21) : x = 6 :=
by
  sorry

end find_smaller_number_l194_194498


namespace sqrt3_minus_sqrt2_abs_plus_sqrt2_eq_sqrt3_sqrt2_times_sqrt2_plus_2_eq_2_plus_2sqrt2_l194_194172

-- Problem 1
theorem sqrt3_minus_sqrt2_abs_plus_sqrt2_eq_sqrt3 : |Real.sqrt 3 - Real.sqrt 2| + Real.sqrt 2 = Real.sqrt 3 := by
  sorry

-- Problem 2
theorem sqrt2_times_sqrt2_plus_2_eq_2_plus_2sqrt2 : Real.sqrt 2 * (Real.sqrt 2 + 2) = 2 + 2 * Real.sqrt 2 := by
  sorry

end sqrt3_minus_sqrt2_abs_plus_sqrt2_eq_sqrt3_sqrt2_times_sqrt2_plus_2_eq_2_plus_2sqrt2_l194_194172


namespace first_machine_rate_l194_194536

theorem first_machine_rate (x : ℕ) (h1 : 30 * x + 30 * 65 = 3000) : x = 35 := sorry

end first_machine_rate_l194_194536


namespace solve_for_x_l194_194474

theorem solve_for_x : 
  ∃ x : ℝ, 7 * (4 * x + 3) - 5 = -3 * (2 - 8 * x) + 1 / 2 ∧ x = -5.375 :=
by
  sorry

end solve_for_x_l194_194474


namespace average_age_is_25_l194_194643

theorem average_age_is_25 (A B C : ℝ) (h_avg_ac : (A + C) / 2 = 29) (h_b : B = 17) :
  (A + B + C) / 3 = 25 := 
  by
    sorry

end average_age_is_25_l194_194643


namespace hash_hash_hash_100_l194_194174

def hash (N : ℝ) : ℝ := 0.4 * N + 3

theorem hash_hash_hash_100 : hash (hash (hash 100)) = 11.08 :=
by sorry

end hash_hash_hash_100_l194_194174


namespace second_bounce_distance_correct_l194_194160

noncomputable def second_bounce_distance (R v g : ℝ) : ℝ := 2 * R - (2 * v / 3) * (Real.sqrt (R / g))

theorem second_bounce_distance_correct (R v g : ℝ) (hR : R > 0) (hv : v > 0) (hg : g > 0) :
  second_bounce_distance R v g = 2 * R - (2 * v / 3) * (Real.sqrt (R / g)) := 
by
  -- Placeholder for the proof
  sorry

end second_bounce_distance_correct_l194_194160


namespace greatest_points_for_top_teams_l194_194200

-- Definitions as per the conditions
def teams := 9 -- Number of teams
def games_per_pair := 2 -- Each team plays every other team twice
def points_win := 3 -- Points for a win
def points_draw := 1 -- Points for a draw
def points_loss := 0 -- Points for a loss

-- Total number of games played
def total_games := (teams * (teams - 1) / 2) * games_per_pair

-- Total points available in the tournament
def total_points := total_games * points_win

-- Given the conditions, prove that the greatest possible number of total points each of the top three teams can accumulate is 42.
theorem greatest_points_for_top_teams :
  ∃ k, (∀ A B C : ℕ, A = B ∧ B = C → A ≤ k) ∧ k = 42 :=
sorry

end greatest_points_for_top_teams_l194_194200


namespace f_strictly_increasing_on_l194_194183

-- Define the function
def f (x : ℝ) : ℝ := x^2 * (2 - x)

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := -3 * x^2 + 4 * x

-- Define the property that the function is strictly increasing on an interval
def strictly_increasing_on (a b : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- State the theorem
theorem f_strictly_increasing_on : strictly_increasing_on 0 (4/3) f :=
sorry

end f_strictly_increasing_on_l194_194183


namespace allan_initial_balloons_l194_194283

theorem allan_initial_balloons (jake_balloons allan_bought_more allan_total_balloons : ℕ) 
  (h1 : jake_balloons = 4)
  (h2 : allan_bought_more = 3)
  (h3 : allan_total_balloons = 8) :
  ∃ (allan_initial_balloons : ℕ), allan_total_balloons = allan_initial_balloons + allan_bought_more ∧ allan_initial_balloons = 5 := 
by
  sorry

end allan_initial_balloons_l194_194283


namespace sin_45_degree_l194_194306

noncomputable section

open Real

theorem sin_45_degree : sin (π / 4) = sqrt 2 / 2 := sorry

end sin_45_degree_l194_194306


namespace contains_zero_l194_194029

-- Define what constitutes a five-digit number
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

-- Define the main proof statement
theorem contains_zero (n1 n2 : ℕ) (c1 c2 : Prop) :
  is_five_digit n1 →
  is_five_digit n2 →
  c1 ∧ c2 →
  n1 + n2 = 111111 →
  ∃ (d : ℕ), d < 10 ∧ (d = 0 ∧ (∃ p1 p2 : ℕ, n1 = p1 * 10 + d ∧ n2 = p2 * 10 + d) ∨ 
             ∃ p1 p2 : ℕ, n1 = d * (10^4) + p1 ∧ n2 = d * (10^4) + p2) :=
begin
  sorry
end

end contains_zero_l194_194029


namespace height_of_shorter_pot_is_20_l194_194981

-- Define the conditions as given
def height_of_taller_pot := 40
def shadow_of_taller_pot := 20
def shadow_of_shorter_pot := 10

-- Define the height of the shorter pot to be determined
def height_of_shorter_pot (h : ℝ) := h

-- Define the relationship using the concept of similar triangles
theorem height_of_shorter_pot_is_20 (h : ℝ) :
  (height_of_taller_pot / shadow_of_taller_pot = height_of_shorter_pot h / shadow_of_shorter_pot) → h = 20 :=
by
  intros
  sorry

end height_of_shorter_pot_is_20_l194_194981


namespace batsman_average_after_17th_innings_l194_194519

theorem batsman_average_after_17th_innings :
  ∀ (A : ℕ), (80 + 16 * A) = 17 * (A + 2) → A + 2 = 48 := by
  intro A h
  sorry

end batsman_average_after_17th_innings_l194_194519


namespace log_equation_solution_l194_194473

theorem log_equation_solution (a b x : ℝ) (h : 5 * (Real.log x / Real.log b) ^ 2 + 2 * (Real.log x / Real.log a) ^ 2 = 10 * (Real.log x) ^ 2 / (Real.log a * Real.log b)) :
    b = a ^ (1 + Real.sqrt 15 / 5) ∨ b = a ^ (1 - Real.sqrt 15 / 5) :=
sorry

end log_equation_solution_l194_194473


namespace find_m_if_a_b_parallel_l194_194749

theorem find_m_if_a_b_parallel :
  ∃ m : ℝ, (∃ a : ℝ × ℝ, a = (-2, 1)) ∧ (∃ b : ℝ × ℝ, b = (1, m)) ∧ (m * -2 = 1) ∧ (m = -1 / 2) :=
by
  sorry

end find_m_if_a_b_parallel_l194_194749


namespace smallest_digit_is_one_l194_194440

-- Given a 4-digit integer x.
def four_digit_integer (x : ℕ) : Prop :=
  1000 ≤ x ∧ x < 10000

-- Define function for the product of digits of x.
def product_of_digits (x : ℕ) : ℕ :=
  let d1 := x % 10
  let d2 := (x / 10) % 10
  let d3 := (x / 100) % 10
  let d4 := (x / 1000) % 10
  d1 * d2 * d3 * d4

-- Define function for the sum of digits of x.
def sum_of_digits (x : ℕ) : ℕ :=
  let d1 := x % 10
  let d2 := (x / 10) % 10
  let d3 := (x / 100) % 10
  let d4 := (x / 1000) % 10
  d1 + d2 + d3 + d4

-- Assume p is a prime number.
def is_prime (p : ℕ) : Prop :=
  ¬ ∃ d, d ∣ p ∧ d ≠ 1 ∧ d ≠ p

-- Proof problem: Given conditions for T(x) and S(x),
-- prove that the smallest digit in x is 1.
theorem smallest_digit_is_one (x p k : ℕ) (h1 : four_digit_integer x)
  (h2 : is_prime p) (h3 : product_of_digits x = p^k)
  (h4 : sum_of_digits x = p^p - 5) : 
  ∃ d1 d2 d3 d4, d1 <= d2 ∧ d1 <= d3 ∧ d1 <= d4 ∧ d1 = 1 
  ∧ (d1 + d2 + d3 + d4 = p^p - 5) 
  ∧ (d1 * d2 * d3 * d4 = p^k) := 
sorry

end smallest_digit_is_one_l194_194440


namespace tangent_line_equation_l194_194579

theorem tangent_line_equation 
  (A : ℝ × ℝ)
  (hA : A = (-1, 2))
  (parabola : ℝ → ℝ)
  (h_parabola : ∀ x, parabola x = 2 * x ^ 2) 
  (tangent : ℝ × ℝ → ℝ)
  (h_tangent : ∀ P, tangent P = -4 * P.1 + 4 * (-1) + 2) : 
  tangent A = 4 * (-1) + 2 :=
by
  sorry

end tangent_line_equation_l194_194579


namespace smallest_solution_is_9_l194_194570

noncomputable def smallest_positive_solution (x : ℝ) : Prop :=
  (3*x / (x - 3) + (3*x^2 - 45) / (x + 3) = 14) ∧ (x > 3) ∧ (∀ y : ℝ, (3*y / (y - 3) + (3*y^2 - 45) / (y + 3) = 14) → (y > 3) → (y ≥ 9))

theorem smallest_solution_is_9 : ∃ x : ℝ, smallest_positive_solution x ∧ x = 9 :=
by
  exists 9
  have : smallest_positive_solution 9 := sorry
  exact ⟨this, rfl⟩

end smallest_solution_is_9_l194_194570


namespace sin_45_degree_eq_sqrt2_div_2_l194_194349

theorem sin_45_degree_eq_sqrt2_div_2 :
  let θ := (real.pi / 4)
  in sin θ = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_degree_eq_sqrt2_div_2_l194_194349


namespace susan_mean_l194_194219

def susan_scores : List ℝ := [87, 90, 95, 98, 100]

theorem susan_mean :
  (susan_scores.sum) / (susan_scores.length) = 94 := by
  sorry

end susan_mean_l194_194219


namespace S_8_arithmetic_sequence_l194_194056

theorem S_8_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (h2 : a 4 = 18 - a 5):
  S 8 = 72 :=
by
  sorry

end S_8_arithmetic_sequence_l194_194056


namespace problem_inequality_minimum_value_l194_194582

noncomputable def f (x y z : ℝ) : ℝ := 
  (3 * x^2 - x) / (1 + x^2) + 
  (3 * y^2 - y) / (1 + y^2) + 
  (3 * z^2 - z) / (1 + z^2)

theorem problem_inequality (x y z : ℝ) 
  (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h_sum : x + y + z = 1) :
  f x y z ≥ 0 :=
sorry

theorem minimum_value (x y z : ℝ) 
  (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h_sum : x + y + z = 1) :
  f x y z = 0 ↔ x = 1/3 ∧ y = 1/3 ∧ z = 1/3 :=
sorry

end problem_inequality_minimum_value_l194_194582


namespace evaluate_neg_64_exp_4_over_3_l194_194716

theorem evaluate_neg_64_exp_4_over_3 : (-64 : ℝ) ^ (4 / 3) = 256 := 
by
  sorry

end evaluate_neg_64_exp_4_over_3_l194_194716


namespace sin_45_degree_l194_194329

theorem sin_45_degree : sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_degree_l194_194329


namespace solution_interval_l194_194040

theorem solution_interval (x : ℝ) (h1 : x / 2 ≤ 5 - x) (h2 : 5 - x < -3 * (2 + x)) :
  x < -11 / 2 := 
sorry

end solution_interval_l194_194040


namespace math_problem_l194_194566

theorem math_problem : 2^5 + (5^2 / 5^1) - 3^3 = 10 :=
by
  sorry

end math_problem_l194_194566


namespace evaluate_neg_sixtyfour_exp_four_thirds_l194_194708

theorem evaluate_neg_sixtyfour_exp_four_thirds : (-64 : ℝ) ^ (4/3 : ℝ) = 256 := 
by {
  have h1 : (-64 : ℝ) = (-4 : ℝ) ^ 3, by norm_num,
  have h2 : ((-4 : ℝ) ^ 3) ^ (4/3) = (-4 : ℝ) ^ (3 * (4/3)), by rw [←real.rpow_mul],
  rw [←h1],
  rw [h2],
  norm_num,
  rw [pow_nat_cast],
  norm_num,
  norm_num,
  rw [pow_nat_cast],
  norm_num
}

end evaluate_neg_sixtyfour_exp_four_thirds_l194_194708


namespace train_length_in_terms_of_james_cycle_l194_194940

/-- Define the mathematical entities involved: L (train length), J (James's cycle length), T (train length per cycle) -/
theorem train_length_in_terms_of_james_cycle 
  (L J T : ℝ) 
  (h1 : 130 * J = L + 130 * T) 
  (h2 : 26 * J = L - 26 * T) 
    : L = 58 * J := 
by 
  sorry

end train_length_in_terms_of_james_cycle_l194_194940


namespace radius_squared_l194_194018

theorem radius_squared (r : ℝ) (AB_len CD_len BP_len : ℝ) (angle_APD : ℝ) (r_squared : ℝ) :
  AB_len = 10 →
  CD_len = 7 →
  BP_len = 8 →
  angle_APD = 60 →
  r_squared = r^2 →
  r_squared = 73 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end radius_squared_l194_194018


namespace solve_inequality_group_l194_194642

theorem solve_inequality_group (x : ℝ) (h1 : -9 < 2 * x - 1) (h2 : 2 * x - 1 ≤ 6) :
  -4 < x ∧ x ≤ 3.5 := 
sorry

end solve_inequality_group_l194_194642


namespace angle_B_is_pi_over_3_l194_194599

theorem angle_B_is_pi_over_3
  (A B C : ℝ) (a b c : ℝ)
  (h_triangle : a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2)
  (h_sin_ratios : ∃ k > 0, a = 5*k ∧ b = 7*k ∧ c = 8*k) :
  B = π / 3 := 
by
  sorry

end angle_B_is_pi_over_3_l194_194599


namespace sale_in_fifth_month_l194_194539

theorem sale_in_fifth_month (a1 a2 a3 a4 a5 a6 avg : ℝ)
  (h1 : a1 = 5420) (h2 : a2 = 5660) (h3 : a3 = 6200) (h4 : a4 = 6350) (h6 : a6 = 6470) (h_avg : avg = 6100) :
  a5 = 6500 :=
by
  sorry

end sale_in_fifth_month_l194_194539


namespace max_value_a_l194_194246

theorem max_value_a (a : ℝ) : 
  (∀ x : ℝ, x > -1 → x + 1 > 0 → x + 1 + 1 / (x + 1) - 2 ≥ a) → a ≤ 0 :=
by
  -- Proof omitted
  sorry

end max_value_a_l194_194246


namespace steve_reading_pages_l194_194958

theorem steve_reading_pages (total_pages: ℕ) (weeks: ℕ) (reading_days_per_week: ℕ) 
  (reads_on_monday: ℕ) (reads_on_wednesday: ℕ) (reads_on_friday: ℕ) :
  total_pages = 2100 → weeks = 7 → reading_days_per_week = 3 → 
  (reads_on_monday = reads_on_wednesday ∧ reads_on_wednesday = reads_on_friday) → 
  ((weeks * reading_days_per_week) > 0) → 
  (total_pages / (weeks * reading_days_per_week)) = reads_on_monday :=
by
  intro h_total_pages h_weeks h_reading_days_per_week h_reads_on_days h_nonzero
  sorry

end steve_reading_pages_l194_194958


namespace unique_prime_sum_diff_l194_194389

theorem unique_prime_sum_diff (p : ℕ) (primeP : Prime p)
  (hx : ∃ (x y : ℕ), Prime x ∧ Prime y ∧ p = x + y)
  (hz : ∃ (z w : ℕ), Prime z ∧ Prime w ∧ p = z - w) : p = 5 :=
sorry

end unique_prime_sum_diff_l194_194389


namespace max_value_sin_cos_combination_l194_194395

theorem max_value_sin_cos_combination :
  ∀ x : ℝ, (5 * Real.sin x + 12 * Real.cos x) ≤ 13 :=
by
  intro x
  sorry

end max_value_sin_cos_combination_l194_194395


namespace sin_45_eq_l194_194344

noncomputable def sin_45_degrees := Real.sin (π / 4)

theorem sin_45_eq : sin_45_degrees = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_eq_l194_194344


namespace intersect_parabolas_l194_194497

theorem intersect_parabolas :
  ∀ (x y : ℝ),
    ((y = 2 * x^2 - 7 * x + 1 ∧ y = 8 * x^2 + 5 * x + 1) ↔ 
     ((x = -2 ∧ y = 23) ∨ (x = 0 ∧ y = 1))) :=
by sorry

end intersect_parabolas_l194_194497


namespace symmetric_points_origin_a_plus_b_l194_194412

theorem symmetric_points_origin_a_plus_b (a b : ℤ) 
  (h1 : a + 3 * b = 5)
  (h2 : a + 2 * b = -3) :
  a + b = -11 :=
by
  sorry

end symmetric_points_origin_a_plus_b_l194_194412


namespace arithmetic_sequence_a18_value_l194_194186

theorem arithmetic_sequence_a18_value 
  (a : ℕ → ℕ) (d : ℕ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_incr : ∀ n, a n < a (n + 1))
  (h_sum : a 2 + a 5 + a 8 = 33)
  (h_geom : (a 5 + 1) ^ 2 = (a 2 + 1) * (a 8 + 7)) :
  a 18 = 37 :=
sorry

end arithmetic_sequence_a18_value_l194_194186


namespace inverse_proportion_l194_194480

theorem inverse_proportion (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : x = 3) (h3 : y = 15) (h4 : y = -30) : x = -3 / 2 :=
by
  sorry

end inverse_proportion_l194_194480


namespace sqrt_square_eq_14_l194_194010

theorem sqrt_square_eq_14 : Real.sqrt (14 ^ 2) = 14 :=
by
  sorry

end sqrt_square_eq_14_l194_194010


namespace eccentricity_of_hyperbola_l194_194583

variable (a b c e : ℝ)

-- The hyperbola definition and conditions.
def hyperbola (a b : ℝ) := (a > 0) ∧ (b > 0) ∧ (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)

-- Eccentricity is greater than 1 and less than the specified upper bound
def eccentricity_range (e : ℝ) := 1 < e ∧ e < (2 * Real.sqrt 3) / 3

-- Main theorem statement: Given the hyperbola with conditions, prove eccentricity lies in the specified range.
theorem eccentricity_of_hyperbola (h : hyperbola a b) (h_line : ∀ (x y : ℝ), y = x * (Real.sqrt 3) / 3 - 0 -> y^2 ≤ (c^2 - x^2 * a^2)) :
  eccentricity_range e :=
sorry

end eccentricity_of_hyperbola_l194_194583


namespace find_b_value_l194_194084

theorem find_b_value (a b c A B C : ℝ) 
  (h1 : a = 1)
  (h2 : B = 120 * (π / 180))
  (h3 : c = b * Real.cos C + c * Real.cos B)
  (h4 : c = 1) : 
  b = Real.sqrt 3 :=
by
  sorry

end find_b_value_l194_194084


namespace complement_of_A_in_U_is_4_l194_194194

-- Define the universal set U
def U : Set ℕ := { x | 1 < x ∧ x < 5 }

-- Define the set A
def A : Set ℕ := {2, 3}

-- Define the complement of A in U
def complement_U_of_A : Set ℕ := { x ∈ U | x ∉ A }

-- State the theorem
theorem complement_of_A_in_U_is_4 : complement_U_of_A = {4} :=
by
  sorry

end complement_of_A_in_U_is_4_l194_194194


namespace common_divisor_greater_than_1_l194_194739
open Nat

theorem common_divisor_greater_than_1 (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
    (h_ab : (a + b) ∣ (a * b)) (h_bc : (b + c) ∣ (b * c)) (h_ca : (c + a) ∣ (c * a)) :
    ∃ k : ℕ, k > 1 ∧ k ∣ a ∧ k ∣ b ∧ k ∣ c := 
by
  sorry

end common_divisor_greater_than_1_l194_194739


namespace remainder_div_3005_95_l194_194983

theorem remainder_div_3005_95 : 3005 % 95 = 60 := 
by {
  sorry
}

end remainder_div_3005_95_l194_194983


namespace cute_polynomial_zero_l194_194293

open Polynomial

def is_prime (n : ℕ) : Prop := nat.prime n
def is_composite (n : ℕ) : Prop := ¬is_prime n ∧ n > 1

def is_cute_subset (s : set ℕ) : Prop :=
  ∃ p q, s = {p, q} ∧ is_prime p ∧ is_composite q ∨ is_prime q ∧ is_composite p

theorem cute_polynomial_zero (f : Polynomial ℤ) :
  (∀ s : set ℕ, is_cute_subset s → is_cute_subset (s.image (λ p, f.eval p))) →
  f = 0 :=
begin
  sorry
end

end cute_polynomial_zero_l194_194293


namespace inequality_proof_l194_194638

theorem inequality_proof (n : ℕ) (h : n > 1) : 
  1 / (2 * n * Real.exp 1) < 1 / Real.exp 1 - (1 - 1 / n) ^ n ∧ 
  1 / Real.exp 1 - (1 - 1 / n) ^ n < 1 / (n * Real.exp 1) := 
by
  sorry

end inequality_proof_l194_194638


namespace problem_statement_l194_194960

variable (a : ℝ)

theorem problem_statement (h : 5 = a + a⁻¹) : a^4 + (a⁻¹)^4 = 527 := 
by 
  sorry

end problem_statement_l194_194960


namespace sin_45_degree_eq_sqrt2_div_2_l194_194348

theorem sin_45_degree_eq_sqrt2_div_2 :
  let θ := (real.pi / 4)
  in sin θ = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_degree_eq_sqrt2_div_2_l194_194348


namespace julia_paint_area_l194_194092

noncomputable def area_to_paint (bedroom_length: ℕ) (bedroom_width: ℕ) (bedroom_height: ℕ) (non_paint_area: ℕ) (num_bedrooms: ℕ) : ℕ :=
  let wall_area_one_bedroom := 2 * (bedroom_length * bedroom_height) + 2 * (bedroom_width * bedroom_height)
  let paintable_area_one_bedroom := wall_area_one_bedroom - non_paint_area
  num_bedrooms * paintable_area_one_bedroom

theorem julia_paint_area :
  area_to_paint 14 11 9 70 4 = 1520 :=
by
  sorry

end julia_paint_area_l194_194092


namespace solve_fraction_l194_194984

theorem solve_fraction :
  (144^2 - 100^2) / 22 = 488 := 
by 
  sorry

end solve_fraction_l194_194984


namespace lcm_18_45_l194_194846

theorem lcm_18_45 : Nat.lcm 18 45 = 90 := by
  sorry

end lcm_18_45_l194_194846


namespace three_digit_numbers_count_l194_194074

theorem three_digit_numbers_count :
  let t_range := finset.range 10
  let h_range := finset.range 9
  let valid_u t := if t * 3 ≤ 9 then 9 - (t * 3) + 1 else 0
  let total_count := h_range.card * t_range.sum (λ t, valid_u t)
  total_count = 198 :=
by
  sorry

end three_digit_numbers_count_l194_194074


namespace largest_interior_angle_l194_194804

theorem largest_interior_angle (x : ℝ) (h₀ : 50 + 55 + x = 180) : 
  max 50 (max 55 x) = 75 := by
  sorry

end largest_interior_angle_l194_194804


namespace contractor_absent_days_proof_l194_194535

def contractor_absent_days (x y : ℝ) : Prop :=
  x + y = 30 ∧ 25 * x - 7.5 * y = 425

theorem contractor_absent_days_proof : ∃ (y : ℝ), contractor_absent_days x y ∧ y = 10 :=
by
  sorry

end contractor_absent_days_proof_l194_194535


namespace domain_of_function_l194_194036

def domain_of_f (x: ℝ) : Prop :=
x >= -1 ∧ x <= 48

theorem domain_of_function :
  ∀ x, (x + 1 >= 0 ∧ 7 - Real.sqrt (x + 1) >= 0 ∧ 4 - Real.sqrt (7 - Real.sqrt (x + 1)) >= 0)
  ↔ domain_of_f x := by
  sorry

end domain_of_function_l194_194036


namespace final_S_is_correct_l194_194026

/-- Define a function to compute the final value of S --/
def final_value_of_S : ℕ :=
  let S := 0
  let I_values := List.range' 1 27 3 -- generate list [1, 4, 7, ..., 28]
  I_values.foldl (fun S I => S + I) 0  -- compute the sum of the list

/-- Theorem stating the final value of S is 145 --/
theorem final_S_is_correct : final_value_of_S = 145 := by
  sorry

end final_S_is_correct_l194_194026


namespace sin_45_eq_sqrt_two_over_two_l194_194368

theorem sin_45_eq_sqrt_two_over_two : Real.sin (π / 4) = sqrt 2 / 2 :=
by
  sorry

end sin_45_eq_sqrt_two_over_two_l194_194368


namespace lines_connecting_intersections_l194_194917

def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

theorem lines_connecting_intersections (n : ℕ) (h : n ≥ 2) :
  let N := binomial n 2
  binomial N 2 = (n * n * (n - 1) * (n - 1) - 2 * n * (n - 1)) / 8 :=
by {
  sorry
}

end lines_connecting_intersections_l194_194917


namespace sin_45_eq_l194_194346

noncomputable def sin_45_degrees := Real.sin (π / 4)

theorem sin_45_eq : sin_45_degrees = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_eq_l194_194346


namespace paco_initial_cookies_l194_194783

theorem paco_initial_cookies (cookies_ate : ℕ) (cookies_left : ℕ) (cookies_initial : ℕ) 
  (h1 : cookies_ate = 15) (h2 : cookies_left = 78) :
  cookies_initial = cookies_ate + cookies_left → cookies_initial = 93 :=
by
  sorry

end paco_initial_cookies_l194_194783


namespace eval_neg_pow_l194_194696

theorem eval_neg_pow (a b : ℝ) (h1 : a = (-4)^3) (h2 : b = (-64)) : (-64 : ℝ)^(4/3) = 256 :=
by {
  have h_eq : b = a := by rw h1,
  rw [h2, h_eq],
  have : (a : ℝ)^(4/3) = ((-4)^3)^(4/3) := by rw h1,
  rw this,
  have : ((-4)^3)^(4/3) = (-4)^4 := by norm_num,
  rw this,
  norm_num,
  exact rfl,
}

end eval_neg_pow_l194_194696


namespace regular_polygon_sides_l194_194443

theorem regular_polygon_sides (n : ℕ) (h₁ : n ≥ 3) (h₂ : 120 = 180 * (n - 2) / n) : n = 6 :=
by
  sorry

end regular_polygon_sides_l194_194443


namespace distance_between_peaks_correct_l194_194797

noncomputable def distance_between_peaks 
    (α β γ δ ε : ℝ) 
    (hα : α = 6 + 50/60 + 33/3600)
    (hβ : β = 7 + 25/60 + 52/3600)
    (hγ : γ = 5 + 24/60 + 52/3600)
    (hδ : δ = 5 + 55/60 + 36/3600)
    (hε : ε = 31 + 4/60 + 34/3600)
    (h_unit_conversion : ∀ θ : ℝ, θ * π / 180 = θ * 3.141592653589793 / 180)
    : ℝ :=
    let α_rad := α * π / 180,
    β_rad := β * π / 180,
    γ_rad := γ * π / 180,
    δ_rad := δ * π / 180,
    ε_rad := ε * π / 180,
    MA := 200 * cos α_rad * sin β_rad / sin (β_rad - α_rad),
    M1A := 200 * cos γ_rad * sin δ_rad / sin (δ_rad - γ_rad),
    MM1 := (MA^2 + M1A^2 - 2 * MA * M1A * cos ε_rad).sqrt,
    OM := MA * tan α_rad,
    O1M1 := M1A * tan γ_rad
in (MM1^2 + (OM - O1M1)^2).sqrt

theorem distance_between_peaks_correct :
    distance_between_peaks (6 + 50 / 60 + 33 / 3600) (7 + 25 / 60 + 52 / 3600) (5 + 24 / 60 + 52 / 3600) (5 + 55 / 60 + 36 / 3600) (31 + 4 / 60 + 34 / 3600) 
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (fun _ => by norm_cast; simp) = 1303 :=
sorry

end distance_between_peaks_correct_l194_194797


namespace n_not_both_perfect_squares_l194_194786

open Int

theorem n_not_both_perfect_squares (n x y : ℤ) (h1 : n > 0) :
  ¬ ((n + 1 = x^2) ∧ (4 * n + 1 = y^2)) :=
by {
  -- Problem restated in Lean, proof not required
  sorry
}

end n_not_both_perfect_squares_l194_194786


namespace Cedar_school_earnings_l194_194477

noncomputable def total_earnings_Cedar_school : ℝ :=
  let total_payment := 774
  let total_student_days := 6 * 4 + 5 * 6 + 3 * 10
  let daily_wage := total_payment / total_student_days
  let Cedar_student_days := 3 * 10
  daily_wage * Cedar_student_days

theorem Cedar_school_earnings :
  total_earnings_Cedar_school = 276.43 :=
by
  sorry

end Cedar_school_earnings_l194_194477


namespace book_total_pages_l194_194990

theorem book_total_pages (x : ℕ) (h1 : x * (3 / 5) * (3 / 8) = 36) : x = 120 := 
by
  -- Proof should be supplied here, but we only need the statement
  sorry

end book_total_pages_l194_194990


namespace function_graph_intersection_l194_194429

theorem function_graph_intersection (f : ℝ → ℝ) :
  (∃ y : ℝ, f 1 = y) → (∃! y : ℝ, f 1 = y) :=
by
  sorry

end function_graph_intersection_l194_194429


namespace equal_cylinder_volumes_l194_194496

theorem equal_cylinder_volumes (x : ℝ) (hx : x > 0) :
  π * (5 + x) ^ 2 * 4 = π * 25 * (4 + x) → x = 35 / 4 :=
by
  sorry

end equal_cylinder_volumes_l194_194496


namespace graph_passes_through_point_l194_194124

theorem graph_passes_through_point (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) :
  ∃ p : ℝ × ℝ, p = (2, 0) ∧ ∀ x, (x = 2 → a ^ (x - 2) - 1 = 0) :=
by
  sorry

end graph_passes_through_point_l194_194124


namespace angle_between_bisectors_is_zero_l194_194549

-- Let's define the properties of the triangle and the required proof.

open Real

-- Define the side lengths of the isosceles triangle
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a ∧ a > 0 ∧ b > 0 ∧ c > 0

def is_isosceles (a b c : ℝ) : Prop :=
  (a = b ∨ a = c ∨ b = c) ∧ is_triangle a b c

-- Define the specific isosceles triangle in the problem
def triangle_ABC : Prop := is_isosceles 5 5 6

-- Prove that the angle φ between the two lines is 0°
theorem angle_between_bisectors_is_zero :
  triangle_ABC → ∃ φ : ℝ, φ = 0 :=
by sorry

end angle_between_bisectors_is_zero_l194_194549


namespace find_abc_sum_l194_194059

theorem find_abc_sum :
  ∀ (a b c : ℝ),
    2 * |a + 3| + 4 - b = 0 →
    c^2 + 4 * b - 4 * c - 12 = 0 →
    a + b + c = 5 :=
by
  intros a b c h1 h2
  sorry

end find_abc_sum_l194_194059


namespace david_older_than_rosy_l194_194038

theorem david_older_than_rosy
  (R D : ℕ) 
  (h1 : R = 12) 
  (h2 : D + 6 = 2 * (R + 6)) : 
  D - R = 18 := 
by
  sorry

end david_older_than_rosy_l194_194038


namespace find_value_of_k_l194_194453

noncomputable def value_of_k (m n : ℝ) : ℝ :=
  let p := 0.4
  let point1 := (m, n)
  let point2 := (m + 2, n + p)
  let k := 5
  k

theorem find_value_of_k (m n : ℝ) : value_of_k m n = 5 :=
sorry

end find_value_of_k_l194_194453


namespace sum_of_series_l194_194034

open BigOperators

-- Define the sequence a(n) = 2 / (n * (n + 3))
def a (n : ℕ) : ℚ := 2 / (n * (n + 3))

-- Prove the sum of the first 20 terms of sequence a equals 10 / 9.
theorem sum_of_series : (∑ n in Finset.range 20, a (n + 1)) = 10 / 9 := by
  sorry

end sum_of_series_l194_194034


namespace sin_45_eq_sqrt2_div_2_l194_194358

theorem sin_45_eq_sqrt2_div_2 : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by sorry

end sin_45_eq_sqrt2_div_2_l194_194358


namespace inequality_proof_l194_194184

theorem inequality_proof 
  (x y z : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hz : z > 0)
  (hxz : x * z = 1) 
  (h₁ : x * (1 + z) > 1) 
  (h₂ : y * (1 + x) > 1) 
  (h₃ : z * (1 + y) > 1) :
  2 * (x + y + z) ≥ -1/x + 1/y + 1/z + 3 :=
sorry

end inequality_proof_l194_194184


namespace max_value_of_a_l194_194591

noncomputable def f : ℝ → ℝ := sorry

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def decreasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y ≤ f x

theorem max_value_of_a
  (odd_f : odd_function f)
  (decr_f : decreasing_function f)
  (h : ∀ x : ℝ, f (Real.cos (2 * x) + Real.sin x) + f (Real.sin x - a) ≤ 0) :
  a ≤ -3 :=
sorry

end max_value_of_a_l194_194591


namespace card_probability_l194_194824

theorem card_probability :
  let hearts := 13
  let clubs := 13
  let total_cards := 52
  let first_card_is_heart := (hearts.to_rat / total_cards.to_rat)
  let second_card_is_club_given_first_is_heart := (clubs.to_rat / (total_cards - 1).to_rat)
  first_card_is_heart * second_card_is_club_given_first_is_heart = (13.to_rat / 204.to_rat) := by
  sorry

end card_probability_l194_194824


namespace inradius_circumradius_le_height_l194_194951

theorem inradius_circumradius_le_height
    {α β γ : ℝ}
    (hα : 0 < α ∧ α ≤ 90)
    (hβ : 0 < β ∧ β ≤ 90)
    (hγ : 0 < γ ∧ γ ≤ 90)
    (α_ge_β : α ≥ β)
    (β_ge_γ : β ≥ γ)
    {r R h : ℝ} :
  r + R ≤ h := 
sorry

end inradius_circumradius_le_height_l194_194951


namespace Ian_hourly_wage_l194_194589

variable (hours_worked : ℕ)
variable (money_left : ℕ)
variable (hourly_wage : ℕ)

theorem Ian_hourly_wage :
  hours_worked = 8 ∧
  money_left = 72 ∧
  hourly_wage = 18 →
  2 * money_left = hours_worked * hourly_wage :=
by
  intros
  sorry

end Ian_hourly_wage_l194_194589


namespace value_of_y_l194_194143

theorem value_of_y : exists y : ℝ, (∀ k : ℝ, (∀ x y : ℝ, x = k / y^2 → (x = 1 → y = 2 → k = 4)) ∧ (x = 0.1111111111111111 → k = 4 → y = 6)) := by
  sorry

end value_of_y_l194_194143


namespace power_simplification_l194_194841

theorem power_simplification :
  (1 / ((-5) ^ 4) ^ 2) * (-5) ^ 9 = -5 :=
by 
  sorry

end power_simplification_l194_194841


namespace problem1_problem2_l194_194190

-- Given the function f(x) = 2x * ln x, prove that the tangent line at (1, f(1)) is perpendicular to y = -1/2*x
theorem problem1 {a : ℝ} (h : ∀ x, f x = a * x * log x) :
  f' (1 : ℝ) = 2 :=
by
  sorry

-- Given the function f(x) = 2x * ln x and the inequality f(x) - m*x + 2 ≥ 0 ∀ x ≥ 1, show m ∈ (-∞, 2]
theorem problem2 {m : ℝ} (h : ∀ (x : ℝ), f x = 2 * x * log x ∧ x ≥ 1 → f x - m * x + 2 ≥ 0) :
  m ∈ set.Iic 2 :=
by
  sorry

end problem1_problem2_l194_194190


namespace find_bc_l194_194585

noncomputable def setA : Set ℝ := {x | x^2 + x - 2 ≤ 0}
noncomputable def setB : Set ℝ := {x | 2 < x + 1 ∧ x + 1 ≤ 4}
noncomputable def setAB : Set ℝ := setA ∪ setB
noncomputable def setC (b c : ℝ) : Set ℝ := {x | x^2 + b * x + c > 0}

theorem find_bc (b c : ℝ) :
  (setAB ∩ setC b c = ∅) ∧ (setAB ∪ setC b c = Set.univ) →
  b = -1 ∧ c = -6 :=
by
  sorry

end find_bc_l194_194585


namespace sin_45_deg_l194_194298

theorem sin_45_deg : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by 
  -- placeholder for the actual proof
  sorry

end sin_45_deg_l194_194298


namespace intersection_points_of_graph_and_line_l194_194426

theorem intersection_points_of_graph_and_line (f : ℝ → ℝ) :
  (∀ x : ℝ, f x ≠ my_special_value) → (∀ x₁ x₂ : ℝ, f x₁ = f x₂ → x₁ = x₂) →
  ∃! x : ℝ, x = 1 ∧ ∃ y : ℝ, y = f x :=
by
  sorry

end intersection_points_of_graph_and_line_l194_194426


namespace complement_of_M_in_U_l194_194423

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 4}

theorem complement_of_M_in_U : U \ M = {3, 5, 6} := by
  sorry

end complement_of_M_in_U_l194_194423


namespace contains_zero_l194_194032

-- Define a five-digit number in terms of its digits.
def five_digit_number (A B C D E : ℕ) : ℕ :=
  10^4 * A + 10^3 * B + 10^2 * C + 10 * D + E

-- Define the conditions:
noncomputable def switch_two_digits (A B C D E F : ℕ) : Prop :=
  (10^4 * A + 10^3 * B + 10^2 * C + 10 * D + E ≠ 10^4 * A + 10^3 * B + 10^2 * F + 10 * D + E)

-- Lean statement for the proof problem.
theorem contains_zero (A B C D E F : ℕ) :
  let ABCDE := five_digit_number A B C D E,
      ABFDE := five_digit_number A B F D E in
  switch_two_digits A B C D E F ∧ (ABCDE + ABFDE = 111111) → (A = 0 ∨ B = 0 ∨ C = 0 ∨ D = 0 ∨ E = 0) :=
by
  -- This is where the proof would go.
  sorry

end contains_zero_l194_194032


namespace solve_for_x_l194_194789

theorem solve_for_x (x : ℝ) :
  (x - 2)^6 + (x - 6)^6 = 64 → x = 3 ∨ x = 5 :=
by
  intros h
  sorry

end solve_for_x_l194_194789


namespace k_is_3_l194_194211

noncomputable def k_solution (k : ℝ) : Prop :=
  k > 1 ∧ (∑' n : ℕ, (n^2 + 3 * n - 2) / k^n = 2)

theorem k_is_3 : ∃ k : ℝ, k_solution k ∧ k = 3 :=
by
  sorry

end k_is_3_l194_194211


namespace last_two_digits_l194_194136

theorem last_two_digits :
  (2 * 5^2 * 2^2 * 13 * 2 * 27 * 2^3 * 7 * 2 * 29 * 2^2 * 3 * 5 / (2^6 * 10^3)) % 100 = 22 :=
by sorry

end last_two_digits_l194_194136


namespace complex_pure_imaginary_l194_194595

theorem complex_pure_imaginary (a : ℂ) : (∃ (b : ℂ), (2 - I) * (a + 2 * I) = b * I) → a = -1 :=
by
  sorry

end complex_pure_imaginary_l194_194595


namespace bella_bracelets_l194_194877

theorem bella_bracelets (h_beads_per_bracelet : Nat)
  (h_initial_beads : Nat) 
  (h_additional_beads : Nat) 
  (h_friends : Nat):
  h_beads_per_bracelet = 8 →
  h_initial_beads = 36 →
  h_additional_beads = 12 →
  h_friends = (h_initial_beads + h_additional_beads) / h_beads_per_bracelet →
  h_friends = 6 :=
by
  intros h_beads_per_bracelet_eq h_initial_beads_eq h_additional_beads_eq h_friends_eq
  subst_vars
  sorry

end bella_bracelets_l194_194877


namespace function_graph_intersection_l194_194428

theorem function_graph_intersection (f : ℝ → ℝ) :
  (∃ y : ℝ, f 1 = y) → (∃! y : ℝ, f 1 = y) :=
by
  sorry

end function_graph_intersection_l194_194428


namespace inequality_proof_l194_194463

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c ≤ 3) : 
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l194_194463


namespace binomial_probability_X_eq_3_l194_194218

theorem binomial_probability_X_eq_3 :
  let n := 6
  let p := 1 / 2
  let k := 3
  let binom := Nat.choose n k
  (binom * p ^ k * (1 - p) ^ (n - k)) = 5 / 16 := by 
  sorry

end binomial_probability_X_eq_3_l194_194218


namespace shaded_region_perimeter_l194_194446

theorem shaded_region_perimeter (C : Real) (r : Real) (L : Real) (P : Real)
  (h0 : C = 48)
  (h1 : r = C / (2 * Real.pi))
  (h2 : L = (90 / 360) * C)
  (h3 : P = 3 * L) :
  P = 36 := by
  sorry

end shaded_region_perimeter_l194_194446


namespace max_value_of_sequence_l194_194630

theorem max_value_of_sequence :
  ∃ a : ℕ → ℕ, (∀ i, 1 ≤ i ∧ i ≤ 101 → 0 < a i) →
              (∀ i, 1 ≤ i ∧ i < 101 → (a i + 1) % a (i + 1) = 0) →
              (a 102 = a 1) →
              (∀ n, (1 ≤ n ∧ n ≤ 101) → a n ≤ 201) :=
by
  sorry

end max_value_of_sequence_l194_194630


namespace sin_45_degree_l194_194307

noncomputable section

open Real

theorem sin_45_degree : sin (π / 4) = sqrt 2 / 2 := sorry

end sin_45_degree_l194_194307


namespace tim_will_attend_game_probability_l194_194980

theorem tim_will_attend_game_probability :
  let P_rain := 0.60
  let P_sunny := 1 - P_rain
  let P_attends_given_rain := 0.25
  let P_attends_given_sunny := 0.70
  let P_rain_and_attends := P_rain * P_attends_given_rain
  let P_sunny_and_attends := P_sunny * P_attends_given_sunny
  (P_rain_and_attends + P_sunny_and_attends) = 0.43 :=
by
  sorry

end tim_will_attend_game_probability_l194_194980


namespace oil_output_per_capita_l194_194995

theorem oil_output_per_capita 
  (total_oil_output_russia : ℝ := 13737.1 * 100 / 9)
  (population_russia : ℝ := 147)
  (population_non_west : ℝ := 6.9)
  (oil_output_non_west : ℝ := 1480.689)
  : 
  (55.084 : ℝ) = 55.084 ∧ 
    (214.59 : ℝ) = (1480.689 / 6.9) ∧ 
    (1038.33 : ℝ) = (total_oil_output_russia / population_russia) :=
by
  sorry

end oil_output_per_capita_l194_194995


namespace six_hundred_sixes_not_square_l194_194472

theorem six_hundred_sixes_not_square : 
  ∀ (n : ℕ), (n = 66666666666666666666666666666666666666666666666666666666666 -- continued 600 times
  ∨ n = 66666666666666666666666666666666666666666666666666666666666 -- continued with some zeros
  ) → ¬ (∃ k : ℕ, k * k = n) := 
by
  sorry

end six_hundred_sixes_not_square_l194_194472


namespace total_onions_l194_194113

theorem total_onions (S SA F J : ℕ) (h1 : S = 4) (h2 : SA = 5) (h3 : F = 9) (h4 : J = 7) : S + SA + F + J = 25 :=
by {
  sorry
}

end total_onions_l194_194113


namespace f_has_two_zeros_l194_194807

def f (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem f_has_two_zeros : ∃ (x1 x2 : ℝ), f x1 = 0 ∧ f x2 = 0 ∧ x1 ≠ x2 := 
by
  sorry

end f_has_two_zeros_l194_194807


namespace car_travel_distance_l194_194751

theorem car_travel_distance 
  (v_train : ℝ) (h_train_speed : v_train = 90) 
  (v_car : ℝ) (h_car_speed : v_car = (2 / 3) * v_train) 
  (t : ℝ) (h_time : t = 0.5) :
  ∃ d : ℝ, d = v_car * t ∧ d = 30 := 
sorry

end car_travel_distance_l194_194751


namespace distinct_x_intercepts_l194_194078

theorem distinct_x_intercepts : 
  ∃ (s : Finset ℝ), s.card = 3 ∧ ∀ x, (x + 5) * (x^2 + 5 * x - 6) = 0 ↔ x ∈ s :=
by { 
  sorry 
}

end distinct_x_intercepts_l194_194078


namespace largest_number_l194_194424

theorem largest_number (a b c : ℕ) (h1 : c = a + 6) (h2 : b = (a + c) / 2) (h3 : a * b * c = 46332) : 
  c = 39 := 
sorry

end largest_number_l194_194424


namespace hyperbola_asymptote_b_value_l194_194584

theorem hyperbola_asymptote_b_value (b : ℝ) (hb : b > 0)
  (asymptote : ∀ x y : ℝ, y = 2 * x → x^2 - (y^2 / b^2) = 1) :
  b = 2 :=
sorry

end hyperbola_asymptote_b_value_l194_194584


namespace betty_bracelets_l194_194169

theorem betty_bracelets : (140 / 14) = 10 := 
by
  norm_num

end betty_bracelets_l194_194169


namespace part1_positive_integer_solutions_part2_value_of_m_part3_fixed_solution_l194_194586

-- Part 1: Proof that the solutions of 2x + y - 6 = 0 under positive integer constraints are (2, 2) and (1, 4)
theorem part1_positive_integer_solutions : 
  (∃ x y : ℤ, 2 * x + y - 6 = 0 ∧ x > 0 ∧ y > 0) → 
  ({(x, y) | 2 * x + y - 6 = 0 ∧ x > 0 ∧ y > 0} = {(2, 2), (1, 4)})
:= sorry

-- Part 2: Proof that if x = y, the value of m that satisfies the system of equations is -4
theorem part2_value_of_m (x y m : ℤ) : 
  x = y → (∃ m, (2 * x + y - 6 = 0 ∧ 2 * x - 2 * y + m * y + 8 = 0)) → m = -4
:= sorry

-- Part 3: Proof that regardless of m, there is a fixed solution (x, y) = (-4, 0) for the equation 2x - 2y + my + 8 = 0
theorem part3_fixed_solution (m : ℤ) : 
  2 * x - 2 * y + m * y + 8 = 0 → (x, y) = (-4, 0)
:= sorry

end part1_positive_integer_solutions_part2_value_of_m_part3_fixed_solution_l194_194586


namespace find_three_digit_number_l194_194897

theorem find_three_digit_number : 
  ∃ x : ℕ, (x >= 100 ∧ x < 1000) ∧ (2 * x = 3 * x - 108) :=
by
  have h : ∀ x : ℕ, 100 ≤ x → x < 1000 → 2 * x = 3 * x - 108 → x = 108 := sorry
  exact ⟨108, by sorry⟩

end find_three_digit_number_l194_194897


namespace vector_expression_l194_194060

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (i j k a b : V)
variables (h_i_j_k_non_coplanar : ∃ (l m n : ℝ), l • i + m • j + n • k = 0 → l = 0 ∧ m = 0 ∧ n = 0)
variables (h_a : a = (1 / 2 : ℝ) • i - j + k)
variables (h_b : b = 5 • i - 2 • j - k)

theorem vector_expression :
  4 • a - 3 • b = -13 • i + 2 • j + 7 • k :=
by
  sorry

end vector_expression_l194_194060


namespace contrapositive_l194_194912

theorem contrapositive (q p : Prop) (h : q → p) : ¬p → ¬q :=
by
  -- Proof will be filled in later.
  sorry

end contrapositive_l194_194912


namespace situationD_not_represented_l194_194987

def situationA := -2 + 10 = 8

def situationB := -2 + 10 = 8

def situationC := 10 - 2 = 8 ∧ -2 + 10 = 8

def situationD := |10 - (-2)| = 12

theorem situationD_not_represented : ¬ (|10 - (-2)| = -2 + 10) := 
by
  sorry

end situationD_not_represented_l194_194987


namespace sin_45_eq_l194_194343

noncomputable def sin_45_degrees := Real.sin (π / 4)

theorem sin_45_eq : sin_45_degrees = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_eq_l194_194343


namespace total_travel_cost_is_47100_l194_194156

-- Define the dimensions of the lawn
def lawn_length : ℝ := 200
def lawn_breadth : ℝ := 150

-- Define the roads' widths and their respective travel costs per sq m
def road1_width : ℝ := 12
def road1_travel_cost : ℝ := 4
def road2_width : ℝ := 15
def road2_travel_cost : ℝ := 5
def road3_width : ℝ := 10
def road3_travel_cost : ℝ := 3
def road4_width : ℝ := 20
def road4_travel_cost : ℝ := 6

-- Define the areas of the roads
def road1_area : ℝ := lawn_length * road1_width
def road2_area : ℝ := lawn_length * road2_width
def road3_area : ℝ := lawn_breadth * road3_width
def road4_area : ℝ := lawn_breadth * road4_width

-- Define the costs for the roads
def road1_cost : ℝ := road1_area * road1_travel_cost
def road2_cost : ℝ := road2_area * road2_travel_cost
def road3_cost : ℝ := road3_area * road3_travel_cost
def road4_cost : ℝ := road4_area * road4_travel_cost

-- Define the total cost
def total_cost : ℝ := road1_cost + road2_cost + road3_cost + road4_cost

-- The theorem statement
theorem total_travel_cost_is_47100 : total_cost = 47100 := by
  sorry

end total_travel_cost_is_47100_l194_194156


namespace cars_15th_time_l194_194132

noncomputable def minutes_since_8am (hour : ℕ) (minute : ℕ) : ℕ :=
  hour * 60 + minute

theorem cars_15th_time :
  let initial_time := minutes_since_8am 8 0
  let interval := 5
  let obstacles_time := 3 * 10
  let minutes_passed := (15 - 1) * interval + obstacles_time
  let total_time := initial_time + minutes_passed
  let expected_time := minutes_since_8am 9 40
  total_time = expected_time :=
by
  let initial_time := minutes_since_8am 8 0
  let interval := 5
  let obstacles_time := 3 * 10
  let minutes_passed := (15 - 1) * interval + obstacles_time
  let total_time := initial_time + minutes_passed
  let expected_time := minutes_since_8am 9 40
  show total_time = expected_time
  sorry

end cars_15th_time_l194_194132


namespace solve_exponential_equation_l194_194490

theorem solve_exponential_equation :
  ∃ x, (2:ℝ)^(2*x) - 8 * (2:ℝ)^x + 12 = 0 ↔ x = 1 ∨ x = 1 + Real.log 3 / Real.log 2 :=
by
  sorry

end solve_exponential_equation_l194_194490


namespace max_f_value_l194_194399

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin x + 12 * Real.cos x

theorem max_f_value : ∃ x : ℝ, f x = 13 :=
sorry

end max_f_value_l194_194399


namespace number_of_ways_to_choose_chairs_l194_194961

def choose_chairs_equivalent (chairs : Nat) (students : Nat) (professors : Nat) : Nat :=
  let positions := (chairs - 2)  -- exclude first and last chair
  Nat.choose positions professors * Nat.factorial professors

theorem number_of_ways_to_choose_chairs : choose_chairs_equivalent 10 5 4 = 1680 :=
by
  -- The positions for professors are available from chairs 2 through 9 which are 8 positions.
  /- Calculation for choosing 4 positions out of these 8:
     C(8,4) * 4! = 70 * 24 = 1680 -/
  sorry

end number_of_ways_to_choose_chairs_l194_194961


namespace proof_problem_l194_194810

-- Definitions of propositions p and q
def p (a b : ℝ) : Prop := a < b → ∀ c : ℝ, c ≠ 0 → a * c^2 < b * c^2
def q : Prop := ∃ x₀ > 0, x₀ - 1 + Real.log x₀ = 0

-- Conditions for the problem
variable (a b : ℝ)
variable (p_false : ¬ p a b)
variable (q_true : q)

-- Proving which compound proposition is true
theorem proof_problem : (¬ p a b) ∧ q := by
  exact ⟨p_false, q_true⟩

end proof_problem_l194_194810


namespace terry_age_proof_l194_194928

-- Condition 1: In 10 years, Terry will be 4 times the age that Nora is currently.
-- Condition 2: Nora is currently 10 years old.
-- We need to prove that Terry's current age is 30 years old.

variable (Terry_now Terry_in_10 Nora_now : ℕ)

theorem terry_age_proof (h1: Terry_in_10 = 4 * Nora_now) (h2: Nora_now = 10) (h3: Terry_in_10 = Terry_now + 10) : Terry_now = 30 := 
by
  sorry

end terry_age_proof_l194_194928


namespace remainder_91_pow_91_mod_100_l194_194045

theorem remainder_91_pow_91_mod_100 : Nat.mod (91 ^ 91) 100 = 91 :=
by
  sorry

end remainder_91_pow_91_mod_100_l194_194045


namespace snail_distance_round_100_l194_194140

def snail_distance (n : ℕ) : ℕ :=
  if n = 0 then 100 else (100 * (n + 2)) / (n + 1)

theorem snail_distance_round_100 : snail_distance 100 = 5050 :=
  sorry

end snail_distance_round_100_l194_194140


namespace greatest_int_less_neg_22_3_l194_194500

theorem greatest_int_less_neg_22_3 : ∃ n : ℤ, n = -8 ∧ n < -22 / 3 ∧ ∀ m : ℤ, m < -22 / 3 → m ≤ n :=
by
  sorry

end greatest_int_less_neg_22_3_l194_194500


namespace least_three_digit_multiple_of_3_4_5_l194_194242

def is_multiple_of (a b : ℕ) : Prop := b % a = 0

theorem least_three_digit_multiple_of_3_4_5 : 
  ∃ n : ℕ, is_multiple_of 3 n ∧ is_multiple_of 4 n ∧ is_multiple_of 5 n ∧ 100 ≤ n ∧ n < 1000 ∧ (∀ m : ℕ, is_multiple_of 3 m ∧ is_multiple_of 4 m ∧ is_multiple_of 5 m ∧ 100 ≤ m ∧ m < 1000 → n ≤ m) ∧ n = 120 :=
by
  sorry

end least_three_digit_multiple_of_3_4_5_l194_194242


namespace sin_45_deg_eq_l194_194373

noncomputable def sin_45_deg : ℝ :=
  real.sin (real.pi / 4)

theorem sin_45_deg_eq : sin_45_deg = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_deg_eq_l194_194373


namespace evaluation_of_expression_l194_194434

theorem evaluation_of_expression
  (a b x y m : ℤ)
  (h1 : a + b = 0)
  (h2 : x * y = 1)
  (h3 : m = -1) :
  2023 * (a + b) + 3 * (|m|) - 2 * (x * y) = 1 :=
by
  -- skipping the proof
  sorry

end evaluation_of_expression_l194_194434


namespace fraction_to_decimal_l194_194894

-- We define the fraction and its simplified form
def fraction : ℚ := 58 / 160
def simplified_fraction : ℚ := 29 / 80

-- We state that the fraction simplifies correctly
lemma simplify_fraction : fraction = simplified_fraction := by
  sorry

-- Define the factorization of the denominator
def denominator_factorization : ℕ := 2^4 * 5

-- Verify the fraction when multiplied by 125/125
def equalized_fraction : ℚ := 29 * 125 / 10000

-- State the final result as a decimal
theorem fraction_to_decimal : fraction = 0.3625 := by
  sorry

end fraction_to_decimal_l194_194894


namespace arithmetic_sequence_a7_l194_194937

variable {a : ℕ → ℚ}

def isArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_a7 (h_arith : isArithmeticSequence a) (h_a1 : a 1 = 2) (h_a3_a5 : a 3 + a 5 = 8) :
  a 7 = 6 :=
sorry

end arithmetic_sequence_a7_l194_194937


namespace lisa_time_to_complete_l194_194213

theorem lisa_time_to_complete 
  (hotdogs_record : ℕ) 
  (eaten_so_far : ℕ) 
  (rate_per_minute : ℕ) 
  (remaining_hotdogs : ℕ) 
  (time_to_complete : ℕ) 
  (h1 : hotdogs_record = 75) 
  (h2 : eaten_so_far = 20) 
  (h3 : rate_per_minute = 11) 
  (h4 : remaining_hotdogs = hotdogs_record - eaten_so_far)
  (h5 : time_to_complete = remaining_hotdogs / rate_per_minute) :
  time_to_complete = 5 :=
sorry

end lisa_time_to_complete_l194_194213


namespace correct_expression_l194_194656

theorem correct_expression (x : ℝ) :
  (x^3 / x^2 = x) :=
by sorry

end correct_expression_l194_194656


namespace sin_45_degree_l194_194327

theorem sin_45_degree : sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_degree_l194_194327


namespace binom_150_150_eq_one_l194_194559

theorem binom_150_150_eq_one :
  nat.choose 150 150 = 1 :=
by {
  sorry
}

end binom_150_150_eq_one_l194_194559


namespace koala_fiber_consumption_l194_194207

theorem koala_fiber_consumption
  (absorbed_fiber : ℝ) (total_fiber : ℝ) 
  (h1 : absorbed_fiber = 0.40 * total_fiber)
  (h2 : absorbed_fiber = 12) :
  total_fiber = 30 := 
by
  sorry

end koala_fiber_consumption_l194_194207


namespace find_n_l194_194921

theorem find_n (n : ℤ) (h : Real.sqrt (10 + n) = 9) : n = 71 :=
sorry

end find_n_l194_194921


namespace part1_part2_l194_194856

open Set

-- Definitions from conditions in a)
def R : Set ℝ := univ
def A : Set ℝ := {x | (x + 2) * (x - 3) < 0}
def B (a : ℝ) : Set ℝ := {x | x - a > 0}

-- Question part (1)
theorem part1 (a : ℝ) (h : a = 1) :
  (compl A) ∪ B a = {x | x ≤ -2 ∨ x > 1} :=
by 
  simp [h]
  sorry

-- Question part (2)
theorem part2 (a : ℝ) :
  A ⊆ B a → a ≤ -2 :=
by 
  sorry

end part1_part2_l194_194856


namespace probability_of_insight_l194_194866

noncomputable def students_in_both_classes : ℕ := 30 + 35 - 40

noncomputable def only_mandarin : ℕ := 30 - students_in_both_classes
noncomputable def only_german : ℕ := 35 - students_in_both_classes
noncomputable def total_ways_to_choose_2_students : ℕ := (Nat.choose 40 2)
noncomputable def ways_to_choose_2_only_mandarin : ℕ := (Nat.choose only_mandarin 2)
noncomputable def ways_to_choose_2_only_german : ℕ := (Nat.choose only_german 2)

theorem probability_of_insight : 
  (1 - ((ways_to_choose_2_only_mandarin + ways_to_choose_2_only_german) / total_ways_to_choose_2_students)) = (145 / 156) := 
by
  have h1 : students_in_both_classes = 25 := by sorry
  have h2 : only_mandarin = 5 := by sorry
  have h3 : only_german = 10 := by sorry
  have h4 : total_ways_to_choose_2_students = 780 := by sorry
  have h5 : ways_to_choose_2_only_mandarin = 10 := by sorry
  have h6 : ways_to_choose_2_only_german = 45 := by sorry
  sorry

end probability_of_insight_l194_194866


namespace total_money_difference_l194_194205

-- Define the number of quarters each sibling has
def quarters_Karen : ℕ := 32
def quarters_Christopher : ℕ := 64
def quarters_Emily : ℕ := 20
def quarters_Michael : ℕ := 12

-- Define the value of each quarter
def value_per_quarter : ℚ := 0.25

-- Prove that the total money difference between the pairs of siblings is $16.00
theorem total_money_difference : 
  (quarters_Karen - quarters_Emily) * value_per_quarter + 
  (quarters_Christopher - quarters_Michael) * value_per_quarter = 16 := by
sorry

end total_money_difference_l194_194205


namespace ratio_of_intercepts_l194_194239

variable (b1 b2 : ℝ)
variable (s t : ℝ)
variable (Hs : s = -b1 / 8)
variable (Ht : t = -b2 / 3)

theorem ratio_of_intercepts (hb1 : b1 ≠ 0) (hb2 : b2 ≠ 0) : s / t = 3 * b1 / (8 * b2) :=
by
  sorry

end ratio_of_intercepts_l194_194239


namespace sqrt_of_9_eq_pm_3_l194_194812

theorem sqrt_of_9_eq_pm_3 : (∃ x : ℤ, x * x = 9) → (∃ x : ℤ, x = 3 ∨ x = -3) :=
by
  sorry

end sqrt_of_9_eq_pm_3_l194_194812


namespace jogger_distance_ahead_l194_194668

noncomputable def jogger_speed_kmph : ℤ := 9
noncomputable def train_speed_kmph : ℤ := 45
noncomputable def train_length_m : ℤ := 120
noncomputable def time_to_pass_seconds : ℤ := 38

theorem jogger_distance_ahead
  (jogger_speed_kmph : ℤ)
  (train_speed_kmph : ℤ)
  (train_length_m : ℤ)
  (time_to_pass_seconds : ℤ) :
  jogger_speed_kmph = 9 →
  train_speed_kmph = 45 →
  train_length_m = 120 →
  time_to_pass_seconds = 38 →
  ∃ distance_ahead : ℤ, distance_ahead = 260 :=
by 
  -- the proof would go here
  sorry  

end jogger_distance_ahead_l194_194668


namespace sin_45_eq_sqrt2_div_2_l194_194360

theorem sin_45_eq_sqrt2_div_2 :
  Real.sin (π / 4) = Real.sqrt 2 / 2 := 
by
  sorry

end sin_45_eq_sqrt2_div_2_l194_194360


namespace contractor_absent_days_l194_194532

theorem contractor_absent_days (W A : ℕ) : 
  (W + A = 30 ∧ 25 * W - 7.5 * A = 425) → A = 10 :=
by
 sorry

end contractor_absent_days_l194_194532


namespace chess_tournament_l194_194800

theorem chess_tournament (m p k n : ℕ) 
  (h1 : m * 9 = p * 6) 
  (h2 : m * n = k * 8) 
  (h3 : p * 2 = k * 6) : 
  n = 4 := 
by 
  sorry

end chess_tournament_l194_194800


namespace find_modulus_z_l194_194100

open Complex

noncomputable def z_w_condition1 (z w : ℂ) : Prop := abs (3 * z - w) = 17
noncomputable def z_w_condition2 (z w : ℂ) : Prop := abs (z + 3 * w) = 4
noncomputable def z_w_condition3 (z w : ℂ) : Prop := abs (z + w) = 6

theorem find_modulus_z (z w : ℂ) (h1 : z_w_condition1 z w) (h2 : z_w_condition2 z w) (h3 : z_w_condition3 z w) :
  abs z = 5 :=
by
  sorry

end find_modulus_z_l194_194100


namespace batch_production_equation_l194_194865

theorem batch_production_equation (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 20) :
  (500 / x) = (300 / (x - 20)) :=
sorry

end batch_production_equation_l194_194865


namespace trader_bags_correct_l194_194161

-- Definitions according to given conditions
def initial_bags := 55
def sold_bags := 23
def restocked_bags := 132

-- Theorem that encapsulates the problem's question and the proven answer
theorem trader_bags_correct :
  (initial_bags - sold_bags + restocked_bags) = 164 :=
by
  sorry

end trader_bags_correct_l194_194161


namespace coffee_grinder_assembly_time_l194_194787

-- Variables for the assembly rates
variables (h r : ℝ)

-- Definitions of conditions
def condition1 : Prop := h / 4 = r
def condition2 : Prop := r / 4 = h
def condition3 : Prop := ∀ start_time end_time net_added, 
  start_time = 9 ∧ end_time = 12 ∧ net_added = 27 → 3 * 3/4 * h = net_added
def condition4 : Prop := ∀ start_time end_time net_added, 
  start_time = 13 ∧ end_time = 19 ∧ net_added = 120 → 6 * 3/4 * r = net_added

-- Theorem statement
theorem coffee_grinder_assembly_time
  (h r : ℝ)
  (c1 : condition1 h r)
  (c2 : condition2 h r)
  (c3 : condition3 h)
  (c4 : condition4 r) :
  h = 12 ∧ r = 80 / 3 :=
sorry

end coffee_grinder_assembly_time_l194_194787


namespace evaluate_given_condition_l194_194565

noncomputable def evaluate_expression (b : ℚ) : ℚ :=
  (7 * b^2 - 15 * b + 5) * (3 * b - 4)

theorem evaluate_given_condition (b : ℚ) (h : b = 4 / 3) : evaluate_expression b = 0 := by
  sorry

end evaluate_given_condition_l194_194565


namespace minimize_expression_l194_194099

theorem minimize_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (cond1 : x + y > z) (cond2 : y + z > x) (cond3 : z + x > y) :
  (x + y + z) * (1 / (x + y - z) + 1 / (y + z - x) + 1 / (z + x - y)) ≥ 9 :=
by
  sorry

end minimize_expression_l194_194099


namespace sandwich_cost_l194_194049

theorem sandwich_cost 
  (loaf_sandwiches : ℕ) (target_sandwiches : ℕ) 
  (bread_cost : ℝ) (meat_cost : ℝ) (cheese_cost : ℝ) 
  (cheese_coupon : ℝ) (meat_coupon : ℝ) (total_threshold : ℝ) 
  (discount_rate : ℝ)
  (h1 : loaf_sandwiches = 10) 
  (h2 : target_sandwiches = 50) 
  (h3 : bread_cost = 4) 
  (h4 : meat_cost = 5) 
  (h5 : cheese_cost = 4) 
  (h6 : cheese_coupon = 1) 
  (h7 : meat_coupon = 1) 
  (h8 : total_threshold = 60) 
  (h9 : discount_rate = 0.1) :
  ( ∃ cost_per_sandwich : ℝ, 
      cost_per_sandwich = 1.944 ) :=
  sorry

end sandwich_cost_l194_194049


namespace third_twenty_third_wise_superior_number_l194_194039

def wise_superior_number (x : ℕ) : Prop :=
  ∃ m n : ℕ, m > n ∧ m - n > 1 ∧ x = m^2 - n^2

theorem third_twenty_third_wise_superior_number :
  ∃ T_3 T_23 : ℕ, wise_superior_number T_3 ∧ wise_superior_number T_23 ∧ T_3 = 15 ∧ T_23 = 57 :=
by
  sorry

end third_twenty_third_wise_superior_number_l194_194039


namespace probability_of_convex_quadrilateral_l194_194689

open Finset

-- Define the number of points on the circle
def num_points : ℕ := 8

-- Define the number of ways to choose 2 points out of num_points
def num_chords : ℕ := choose num_points 2

-- Define the number of ways to choose 4 chords from num_chords
def num_ways_to_choose_4_chords : ℕ := choose num_chords 4

-- Define the number of ways to choose 4 points out of num_points, each forming a convex quadrilateral
def num_ways_to_form_convex_quad : ℕ := choose num_points 4

-- Define the probability calculation
def probability : ℚ := num_ways_to_form_convex_quad / num_ways_to_choose_4_chords

-- Main theorem to prove
theorem probability_of_convex_quadrilateral : probability = 2 / 585 := by sorry

end probability_of_convex_quadrilateral_l194_194689


namespace evaluate_expression_l194_194507

theorem evaluate_expression : 4 * (8 - 3) - 6 / 3 = 18 :=
by sorry

end evaluate_expression_l194_194507


namespace sum_base8_l194_194728

theorem sum_base8 (a b c : ℕ) (h₁ : a = 7*8^2 + 7*8 + 7)
                           (h₂ : b = 7*8 + 7)
                           (h₃ : c = 7) :
  a + b + c = 1*8^3 + 1*8^2 + 0*8 + 5 :=
by
  sorry

end sum_base8_l194_194728


namespace average_income_of_all_customers_l194_194281

theorem average_income_of_all_customers
  (n m : ℕ) 
  (a b : ℝ) 
  (customers_responded : n = 50) 
  (wealthiest_count : m = 10) 
  (other_customers_count : n - m = 40) 
  (wealthiest_avg_income : a = 55000) 
  (other_avg_income : b = 42500) : 
  (m * a + (n - m) * b) / n = 45000 := 
by
  -- transforming given conditions into useful expressions
  have h1 : m = 10 := by assumption
  have h2 : n = 50 := by assumption
  have h3 : n - m = 40 := by assumption
  have h4 : a = 55000 := by assumption
  have h5 : b = 42500 := by assumption
  sorry

end average_income_of_all_customers_l194_194281


namespace gcd_153_119_l194_194837

theorem gcd_153_119 : Nat.gcd 153 119 = 17 := by
  sorry

end gcd_153_119_l194_194837


namespace arithmetic_sequence_n_terms_l194_194903

theorem arithmetic_sequence_n_terms:
  ∀ (a₁ d aₙ n: ℕ), 
  a₁ = 6 → d = 3 → aₙ = 300 → aₙ = a₁ + (n - 1) * d → n = 99 :=
by
  intros a₁ d aₙ n h1 h2 h3 h4
  sorry

end arithmetic_sequence_n_terms_l194_194903


namespace isabella_purchases_l194_194605

def isabella_items_total (alexis_pants alexis_dresses isabella_pants isabella_dresses : ℕ) : ℕ :=
  isabella_pants + isabella_dresses

theorem isabella_purchases
  (alexis_pants : ℕ) (alexis_dresses : ℕ)
  (h_pants : alexis_pants = 21)
  (h_dresses : alexis_dresses = 18)
  (h_ratio : ∀ (x : ℕ), alexis_pants = 3 * x → alexis_dresses = 3 * x):
  isabella_items_total (21 / 3) (18 / 3) = 13 :=
by
  sorry

end isabella_purchases_l194_194605


namespace sin_45_eq_1_div_sqrt_2_l194_194337

theorem sin_45_eq_1_div_sqrt_2 : Real.sin (π / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end sin_45_eq_1_div_sqrt_2_l194_194337


namespace probability_heart_then_club_l194_194829

theorem probability_heart_then_club :
  let P_heart := 13 / 52
  let P_club_given_heart := 13 / 51
  P_heart * P_club_given_heart = 13 / 204 := 
by
  let P_heart := (13 : ℚ) / 52
  let P_club_given_heart := (13 : ℚ) / 51
  have h : P_heart * P_club_given_heart = 13 / 204 := by
    calc
      P_heart * P_club_given_heart
        = (13 / 52) * (13 / 51) : rfl
    ... = (13 * 13) / (52 * 51) : by rw [mul_div_mul_comm]
    ... = 169 / 2652 : rfl
    ... = 13 / 204 : by norm_num
  exact h

end probability_heart_then_club_l194_194829


namespace range_of_a_minimum_value_of_b_l194_194885

def is_fixed_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop := f x₀ = x₀

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + (2 * b - 1) * x + b - 2
noncomputable def g (a x : ℝ) : ℝ := -x + a / (3 * a^2 - 2 * a + 1)

theorem range_of_a (h : ∀ b : ℝ, ∃ x1 x2 : ℝ, is_fixed_point (f a b) x1 ∧ is_fixed_point (f a b) x2) : 0 < a ∧ a < 4 :=
sorry

theorem minimum_value_of_b (hx1 : is_fixed_point (f a b) x₁) (hx2 : is_fixed_point (f a b) x₂)
  (hm : g a ((x₁ + x₂) / 2) = (x₁ + x₂) / 2) (ha : 0 < a ∧ a < 4) : b ≥ 3/4 :=
sorry

end range_of_a_minimum_value_of_b_l194_194885


namespace min_value_f_l194_194510

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos x)^2 / (Real.cos x * Real.sin x - (Real.sin x)^2)

theorem min_value_f :
  ∃ x : ℝ, 0 < x ∧ x < Real.pi / 4 ∧ f x = 4 := 
sorry

end min_value_f_l194_194510


namespace extremum_at_neg3_l194_194420

variable (a : ℝ)
def f (x : ℝ) : ℝ := x^3 + 5 * x^2 + a * x
def f_deriv (x : ℝ) : ℝ := 3 * x^2 + 10 * x + a

theorem extremum_at_neg3 (h : f_deriv a (-3) = 0) : a = 3 := 
  by
  sorry

end extremum_at_neg3_l194_194420


namespace product_sum_of_roots_l194_194947

theorem product_sum_of_roots
  {p q r : ℝ}
  (h : (∀ x : ℝ, (4 * x^3 - 8 * x^2 + 16 * x - 12) = 0 → (x = p ∨ x = q ∨ x = r))) :
  p * q + q * r + r * p = 4 := 
sorry

end product_sum_of_roots_l194_194947


namespace emily_has_28_beads_l194_194691

def beads_per_necklace : ℕ := 7
def necklaces : ℕ := 4

def total_beads : ℕ := necklaces * beads_per_necklace

theorem emily_has_28_beads : total_beads = 28 := by
  sorry

end emily_has_28_beads_l194_194691


namespace sin_45_eq_l194_194347

noncomputable def sin_45_degrees := Real.sin (π / 4)

theorem sin_45_eq : sin_45_degrees = 1 / Real.sqrt 2 :=
by
  sorry

end sin_45_eq_l194_194347


namespace sin_45_degree_l194_194311

noncomputable section

open Real

theorem sin_45_degree : sin (π / 4) = sqrt 2 / 2 := sorry

end sin_45_degree_l194_194311


namespace sin_45_deg_eq_l194_194376

noncomputable def sin_45_deg : ℝ :=
  real.sin (real.pi / 4)

theorem sin_45_deg_eq : sin_45_deg = real.sqrt 2 / 2 :=
by
  sorry

end sin_45_deg_eq_l194_194376


namespace marvin_next_birthday_monday_l194_194620

def is_leap_year (y : ℕ) : Prop :=
  (y % 4 = 0 ∧ y % 100 ≠ 0) ∨ (y % 400 = 0)

def day_of_week_after_leap_years (start_day : ℕ) (leap_years : ℕ) : ℕ :=
  (start_day + 2 * leap_years) % 7

def next_birthday_on_monday (year : ℕ) (start_day : ℕ) : ℕ :=
  let next_day := day_of_week_after_leap_years start_day ((year - 2012)/4)
  year + 4 * ((7 - next_day + 1) / 2)

theorem marvin_next_birthday_monday : next_birthday_on_monday 2012 3 = 2016 :=
by sorry

end marvin_next_birthday_monday_l194_194620


namespace percentage_divisible_by_7_l194_194985

-- Define the total integers and the condition for being divisible by 7
def total_ints := 140
def divisible_by_7 (n : ℕ) : Prop := n % 7 = 0

-- Calculate the number of integers between 1 and 140 that are divisible by 7
def count_divisible_by_7 : ℕ := Nat.succ (140 / 7)

-- The theorem to prove
theorem percentage_divisible_by_7 : (count_divisible_by_7 / total_ints : ℚ) * 100 = 14.29 := by
  sorry

end percentage_divisible_by_7_l194_194985


namespace sin_45_eq_sqrt2_div_2_l194_194354

theorem sin_45_eq_sqrt2_div_2 : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by sorry

end sin_45_eq_sqrt2_div_2_l194_194354


namespace animal_shelter_l194_194600

theorem animal_shelter : ∃ D C : ℕ, (D = 75) ∧ (D / C = 15 / 7) ∧ (D / (C + 20) = 15 / 11) :=
by
  sorry

end animal_shelter_l194_194600


namespace find_number_l194_194765

theorem find_number (x n : ℤ) (h1 : |x| = 9 * x - n) (h2 : x = 2) : n = 16 := by 
  sorry

end find_number_l194_194765


namespace card_probability_l194_194825

theorem card_probability :
  let hearts := 13
  let clubs := 13
  let total_cards := 52
  let first_card_is_heart := (hearts.to_rat / total_cards.to_rat)
  let second_card_is_club_given_first_is_heart := (clubs.to_rat / (total_cards - 1).to_rat)
  first_card_is_heart * second_card_is_club_given_first_is_heart = (13.to_rat / 204.to_rat) := by
  sorry

end card_probability_l194_194825


namespace dog_weight_ratio_l194_194962

theorem dog_weight_ratio :
  ∀ (brown black white grey : ℕ),
    brown = 4 →
    black = brown + 1 →
    grey = black - 2 →
    (brown + black + white + grey) / 4 = 5 →
    white / brown = 2 :=
by
  intros brown black white grey h_brown h_black h_grey h_avg
  sorry

end dog_weight_ratio_l194_194962


namespace contractor_absent_days_proof_l194_194533

def contractor_absent_days (x y : ℝ) : Prop :=
  x + y = 30 ∧ 25 * x - 7.5 * y = 425

theorem contractor_absent_days_proof : ∃ (y : ℝ), contractor_absent_days x y ∧ y = 10 :=
by
  sorry

end contractor_absent_days_proof_l194_194533


namespace count_valid_3_digit_numbers_l194_194076

def valid_units_digit (tens_digit units_digit : ℕ) : Prop :=
  units_digit ≥ 3 * tens_digit

def count_valid_numbers : ℕ :=
  (∑ tens_digit in {0, 1, 2, 3}, (finset.filter (valid_units_digit tens_digit) (finset.range 10)).card) * 9

theorem count_valid_3_digit_numbers : count_valid_numbers = 198 :=
by
  sorry

end count_valid_3_digit_numbers_l194_194076


namespace ratio_of_boys_to_total_students_l194_194197

theorem ratio_of_boys_to_total_students
  (p : ℝ)
  (h : p = (3/4) * (1 - p)) :
  p = 3 / 7 :=
by
  sorry

end ratio_of_boys_to_total_students_l194_194197


namespace probability_heart_then_club_l194_194828

theorem probability_heart_then_club :
  let P_heart := 13 / 52
  let P_club_given_heart := 13 / 51
  P_heart * P_club_given_heart = 13 / 204 := 
by
  let P_heart := (13 : ℚ) / 52
  let P_club_given_heart := (13 : ℚ) / 51
  have h : P_heart * P_club_given_heart = 13 / 204 := by
    calc
      P_heart * P_club_given_heart
        = (13 / 52) * (13 / 51) : rfl
    ... = (13 * 13) / (52 * 51) : by rw [mul_div_mul_comm]
    ... = 169 / 2652 : rfl
    ... = 13 / 204 : by norm_num
  exact h

end probability_heart_then_club_l194_194828


namespace real_solutions_of_fraction_eqn_l194_194567

theorem real_solutions_of_fraction_eqn (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ 7) :
  ( x = 3 + Real.sqrt 3 ∨ x = 3 + Real.sqrt 5 ∨ x = 3 - Real.sqrt 5 ) ↔
    ((x - 1) * (x - 3) * (x - 5) * (x - 7) * (x - 3) * (x - 5) * (x - 1)) / ((x - 3) * (x - 7) * (x - 3)) = 1 :=
sorry

end real_solutions_of_fraction_eqn_l194_194567


namespace graduation_problem_l194_194495

def valid_xs : List ℕ :=
  [10, 12, 15, 18, 20, 24, 30]

noncomputable def sum_valid_xs (l : List ℕ) : ℕ :=
  l.foldr (λ x sum => x + sum) 0

theorem graduation_problem :
  sum_valid_xs valid_xs = 129 :=
by
  sorry

end graduation_problem_l194_194495


namespace Запад_oil_output_per_capita_Не_Запад_oil_output_per_capita_Россия_oil_output_per_capita_l194_194997

noncomputable def oil_output_per_capita (total_output : ℝ) (population : ℝ) : ℝ := total_output / population

theorem Запад_oil_output_per_capita :
  oil_output_per_capita 55.084 1 = 55.084 :=
by
  sorry

theorem Не_Запад_oil_output_per_capita :
  oil_output_per_capita 1480.689 6.9 = 214.59 :=
by
  sorry

theorem Россия_oil_output_per_capita :
  oil_output_per_capita (13737.1 * 100 / 9) 147 = 1038.33 :=
by
  sorry

end Запад_oil_output_per_capita_Не_Запад_oil_output_per_capita_Россия_oil_output_per_capita_l194_194997


namespace minimum_sum_of_dimensions_l194_194225

   theorem minimum_sum_of_dimensions (a b c : ℕ) (habc : a * b * c = 3003) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
     a + b + c = 45 :=
   sorry
   
end minimum_sum_of_dimensions_l194_194225


namespace mod7_remainder_problem_l194_194101

theorem mod7_remainder_problem (a b c : ℤ) (h1 : 0 ≤ a ∧ a < 7) (h2 : 0 ≤ b ∧ b < 7) (h3 : 0 ≤ c ∧ c < 7)
    (hc1 : a + 3 * b + 2 * c ≡ 2 [ZMOD 7])
    (hc2 : 2 * a + b + 3 * c ≡ 3 [ZMOD 7])
    (hc3 : 3 * a + 2 * b + c ≡ 5 [ZMOD 7]) : a * b * c ≡ 1 [ZMOD 7] := by
  sorry

end mod7_remainder_problem_l194_194101


namespace unit_digit_7_power_2023_l194_194623

theorem unit_digit_7_power_2023 : (7 ^ 2023) % 10 = 3 := by
  sorry

end unit_digit_7_power_2023_l194_194623


namespace calculate_fraction_l194_194879

theorem calculate_fraction : (1 / (1 + 1 / (4 + 1 / 5))) = (21 / 26) :=
by
  sorry

end calculate_fraction_l194_194879


namespace alpha_convex_implies_J_convex_alpha_convex_implies_general_convex_l194_194175

variable {D : Set ℝ} {f : ℝ → ℝ} (α : ℝ) (hα : 0 < α ∧ α < 1)
(hα_convex : ∀ x1 x2 ∈ D, α * f x1 + (1 - α) * f x2 ≥ f (α * x1 + (1 - α) * x2))

/-- If a function is α-convex on a domain, then it is J-convex (midpoint-convex) on that domain. -/
theorem alpha_convex_implies_J_convex (hf : ∀ x1 x2 ∈ D, α * f x1 + (1 - α) * f x2 ≥ f (α * x1 + (1 - α) * x2)) :
  ∀ x1 x2 ∈ D, f x1 + f x2 ≥ 2 * f ((x1 + x2) / 2) :=
by
  sorry

/-- If a function is α-convex on a domain, it is also (α^n / ((1-α)^n + α^n))-convex on that domain for any natural number n ≥ 1. -/
theorem alpha_convex_implies_general_convex
  (hf : ∀ x1 x2 ∈ D, α * f x1 + (1 - α) * f x2 ≥ f (α * x1 + (1 - α) * x2)) :
  ∀ (n : ℕ) (hn : n ≥ 1) (x1 x2 ∈ D), ((α^n * f x1 + (1 - α)^n * f x2) / (α^n + (1 - α)^n)) ≥ f ((α^n * x1 + (1 - α)^n * x2) / (α^n + (1 - α)^n)) :=
by
  sorry

end alpha_convex_implies_J_convex_alpha_convex_implies_general_convex_l194_194175


namespace arithmetic_expression_eval_l194_194244

theorem arithmetic_expression_eval : 3 + (12 / 3 - 1) ^ 2 = 12 := by
  sorry

end arithmetic_expression_eval_l194_194244


namespace liz_three_pointers_l194_194952

-- Define the points scored by Liz's team in the final quarter.
def points_scored_by_liz (free_throws jump_shots three_pointers : ℕ) : ℕ :=
  free_throws * 1 + jump_shots * 2 + three_pointers * 3

-- Define the points needed to tie the game.
def points_needed_to_tie (initial_deficit points_lost other_team_points : ℕ) : ℕ :=
  points_lost + (initial_deficit - points_lost) + other_team_points

-- The total points scored by Liz from free throws and jump shots.
def liz_regular_points (free_throws jump_shots : ℕ) : ℕ :=
  free_throws * 1 + jump_shots * 2

theorem liz_three_pointers :
  ∀ (free_throws jump_shots liz_team_deficit_final quarter_deficit other_team_points liz_team_deficit_end final_deficit : ℕ),
    liz_team_deficit_final = 20 →
    free_throws = 5 →
    jump_shots = 4 →
    other_team_points = 10 →
    liz_team_deficit_end = 8 →
    final_deficit = liz_team_deficit_final - liz_team_deficit_end →
    (free_throws * 1 + jump_shots * 2 + 3 * final_deficit) = 
      points_needed_to_tie 20 other_team_points 8 →
    (3 * final_deficit) = 9 →
    final_deficit = 3 →
    final_deficit = 3 :=
by
  intros 
  try sorry

end liz_three_pointers_l194_194952


namespace length_of_AP_l194_194938

noncomputable def square_side_length : ℝ := 8
noncomputable def rect_width : ℝ := 12
noncomputable def rect_height : ℝ := 8

axiom AD_perpendicular_WX : true
axiom shaded_area_half_WXYZ : true

theorem length_of_AP (AP : ℝ) (shaded_area : ℝ)
  (h1 : shaded_area = (rect_width * rect_height) / 2)
  (h2 : shaded_area = (square_side_length - AP) * square_side_length)
  : AP = 2 := by
  sorry

end length_of_AP_l194_194938


namespace expected_greetings_l194_194385

theorem expected_greetings :
  let p1 := 1       -- Probability 1
  let p2 := 0.8     -- Probability 0.8
  let p3 := 0.5     -- Probability 0.5
  let p4 := 0       -- Probability 0
  let n1 := 8       -- Number of colleagues with probability 1
  let n2 := 15      -- Number of colleagues with probability 0.8
  let n3 := 14      -- Number of colleagues with probability 0.5
  let n4 := 3       -- Number of colleagues with probability 0
  p1 * n1 + p2 * n2 + p3 * n3 + p4 * n4 = 27 :=
by
  sorry

end expected_greetings_l194_194385


namespace coupons_used_l194_194279

theorem coupons_used
  (initial_books : ℝ)
  (sold_books : ℝ)
  (coupons_per_book : ℝ)
  (remaining_books := initial_books - sold_books)
  (total_coupons := remaining_books * coupons_per_book) :
  initial_books = 40.0 →
  sold_books = 20.0 →
  coupons_per_book = 4.0 →
  total_coupons = 80.0 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end coupons_used_l194_194279


namespace proof_problem_l194_194384

-- Definitions
variable (T : Type) (Sam : T)
variable (solves_all : T → Prop) (passes : T → Prop)

-- Given condition (Dr. Evans's statement)
axiom dr_evans_statement : ∀ x : T, solves_all x → passes x

-- Statement to be proven
theorem proof_problem : ¬ (passes Sam) → ¬ (solves_all Sam) :=
  by sorry

end proof_problem_l194_194384


namespace maximum_value_of_f_l194_194394

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin x + 12 * Real.cos x

theorem maximum_value_of_f : ∃ x : ℝ, f x = 13 :=
by 
  sorry

end maximum_value_of_f_l194_194394


namespace total_loaves_served_l194_194514

-- Given conditions
def wheat_bread := 0.5
def white_bread := 0.4

-- Proof that total loaves served is 0.9
theorem total_loaves_served : wheat_bread + white_bread = 0.9 :=
by sorry

end total_loaves_served_l194_194514


namespace probability_one_from_each_l194_194873

-- Define the total number of cards
def total_cards : ℕ := 10

-- Define the number of cards from Amelia's name
def amelia_cards : ℕ := 6

-- Define the number of cards from Lucas's name
def lucas_cards : ℕ := 4

-- Define the probability that one letter is from each person's name
theorem probability_one_from_each : (amelia_cards / total_cards) * (lucas_cards / (total_cards - 1)) +
                                    (lucas_cards / total_cards) * (amelia_cards / (total_cards - 1)) = 8 / 15 :=
by
  sorry

end probability_one_from_each_l194_194873
