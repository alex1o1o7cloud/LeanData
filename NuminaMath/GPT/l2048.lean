import Mathlib

namespace Lowella_score_l2048_204889

theorem Lowella_score
  (Mandy_score : ℕ)
  (Pamela_score : ℕ)
  (Lowella_score : ℕ)
  (h1 : Mandy_score = 84) 
  (h2 : Mandy_score = 2 * Pamela_score)
  (h3 : Pamela_score = Lowella_score + 20) :
  Lowella_score = 22 := by
  sorry

end Lowella_score_l2048_204889


namespace geometric_series_sum_l2048_204869

theorem geometric_series_sum (a r : ℚ) (h_a : a = 1) (h_r : r = 1/3) :
  (∑' n : ℕ, a * r^n) = 3/2 :=
by
  -- proof goes here
  sorry

end geometric_series_sum_l2048_204869


namespace distinct_elements_in_T_l2048_204832

def sequence1 (k : ℕ) : ℤ := 3 * k - 1
def sequence2 (m : ℕ) : ℤ := 8 * m + 2

def setC : Finset ℤ := Finset.image sequence1 (Finset.range 3000)
def setD : Finset ℤ := Finset.image sequence2 (Finset.range 3000)
def setT : Finset ℤ := setC ∪ setD

theorem distinct_elements_in_T : setT.card = 3000 := by
  sorry

end distinct_elements_in_T_l2048_204832


namespace probability_of_drawing_red_ball_l2048_204823

theorem probability_of_drawing_red_ball (total_balls red_balls white_balls: ℕ) 
    (h1 : total_balls = 5) 
    (h2 : red_balls = 2) 
    (h3 : white_balls = 3) : 
    (red_balls : ℚ) / total_balls = 2 / 5 := 
by 
    sorry

end probability_of_drawing_red_ball_l2048_204823


namespace common_tangent_lines_l2048_204884

theorem common_tangent_lines (m : ℝ) (hm : 0 < m) :
  (∀ x y : ℝ, x^2 + y^2 - (4 * m + 2) * x - 2 * m * y + 4 * m^2 + 4 * m + 1 = 0 →
     (y = 0 ∨ y = 4 / 3 * x - 4 / 3)) :=
by sorry

end common_tangent_lines_l2048_204884


namespace find_S_9_l2048_204839

-- Conditions
def aₙ (n : ℕ) : ℕ := sorry  -- arithmetic sequence

def Sₙ (n : ℕ) : ℕ := sorry  -- sum of the first n terms of the sequence

axiom condition_1 : 2 * aₙ 8 = 6 + aₙ 11

-- Proof goal
theorem find_S_9 : Sₙ 9 = 54 :=
sorry

end find_S_9_l2048_204839


namespace mashed_potatoes_vs_tomatoes_l2048_204807

theorem mashed_potatoes_vs_tomatoes :
  let m := 144
  let t := 79
  m - t = 65 :=
by 
  repeat { sorry }

end mashed_potatoes_vs_tomatoes_l2048_204807


namespace cost_of_1000_pairs_pairs_for_48000_yuan_minimum_pairs_to_avoid_loss_l2048_204873

-- Define the production cost function
def production_cost (n : ℕ) : ℕ := 4000 + 50 * n

-- Define the profit function
def profit (n : ℕ) : ℤ := 90 * n - 4000 - 50 * n

-- 1. Prove that the cost for producing 1000 pairs of shoes is 54,000 yuan
theorem cost_of_1000_pairs : production_cost 1000 = 54000 := 
by sorry

-- 2. Prove that if the production cost is 48,000 yuan, then 880 pairs of shoes were produced
theorem pairs_for_48000_yuan (n : ℕ) (h : production_cost n = 48000) : n = 880 := 
by sorry

-- 3. Prove that at least 100 pairs of shoes must be produced each day to avoid a loss
theorem minimum_pairs_to_avoid_loss (n : ℕ) : profit n ≥ 0 ↔ n ≥ 100 := 
by sorry

end cost_of_1000_pairs_pairs_for_48000_yuan_minimum_pairs_to_avoid_loss_l2048_204873


namespace function_is_quadratic_l2048_204854

-- Definitions for the conditions
def is_quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), (a ≠ 0) ∧ ∀ (x : ℝ), f x = a * x^2 + b * x + c

-- The function to be proved as a quadratic function
def f (x : ℝ) : ℝ := 2 * x^2 - 2 * x + 1

-- The theorem statement: f must be a quadratic function
theorem function_is_quadratic : is_quadratic_function f :=
  sorry

end function_is_quadratic_l2048_204854


namespace solution_set_of_inequality_minimum_value_2a_plus_b_l2048_204850

noncomputable def f (x : ℝ) : ℝ := x + 1 + |3 - x|

theorem solution_set_of_inequality :
  {x : ℝ | x ≥ -1 ∧ f x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 4} :=
by
  sorry

theorem minimum_value_2a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 8 * a * b = a + 2 * b) :
  2 * a + b = 9 / 8 :=
by
  sorry

end solution_set_of_inequality_minimum_value_2a_plus_b_l2048_204850


namespace line_through_point_with_slope_l2048_204843

theorem line_through_point_with_slope (x y : ℝ) (h : y - 2 = -3 * (x - 1)) : 3 * x + y - 5 = 0 :=
sorry

example : 3 * 1 + 2 - 5 = 0 := by sorry

end line_through_point_with_slope_l2048_204843


namespace domain_of_function_l2048_204896

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) + 1 / Real.sqrt (2 - x^2)

theorem domain_of_function : 
  {x : ℝ | x > -1 ∧ x < Real.sqrt 2} = {x : ℝ | x ∈ Set.Ioo (-1) (Real.sqrt 2)} :=
by
  sorry

end domain_of_function_l2048_204896


namespace proposition_D_l2048_204871

-- Definitions extracted from the conditions
variables {a b : ℝ} (c d : ℝ)

-- Proposition D to be proven
theorem proposition_D (ha : a < b) (hb : b < 0) : a^2 > b^2 := sorry

end proposition_D_l2048_204871


namespace f_neg_l2048_204857

variable (f : ℝ → ℝ)

-- Given condition that f is an odd function
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- The form of f for x ≥ 0
def f_pos (x : ℝ) (h : 0 ≤ x) : f x = -x^2 + 2 * x := sorry

-- Objective to prove f(x) for x < 0
theorem f_neg {x : ℝ} (h : x < 0) (hf_odd : odd_function f) (hf_pos : ∀ x, 0 ≤ x → f x = -x^2 + 2 * x) : f x = x^2 + 2 * x := 
by 
  sorry

end f_neg_l2048_204857


namespace solution_mn_l2048_204887

theorem solution_mn (m n : ℤ) (h1 : |m| = 4) (h2 : |n| = 5) (h3 : n < 0) : m + n = -1 ∨ m + n = -9 := 
by
  sorry

end solution_mn_l2048_204887


namespace triangle_third_side_range_l2048_204852

variable (a b c : ℝ)

theorem triangle_third_side_range 
  (h₁ : |a + b - 4| + (a - b + 2)^2 = 0)
  (h₂ : a + b > c)
  (h₃ : a + c > b)
  (h₄ : b + c > a) : 2 < c ∧ c < 4 := 
sorry

end triangle_third_side_range_l2048_204852


namespace cannot_buy_same_number_of_notebooks_l2048_204874

theorem cannot_buy_same_number_of_notebooks
  (price_softcover : ℝ)
  (price_hardcover : ℝ)
  (notebooks_ming : ℝ)
  (notebooks_li : ℝ)
  (h1 : price_softcover = 12)
  (h2 : price_hardcover = 21)
  (h3 : price_hardcover = price_softcover + 1.2) :
  notebooks_ming = 12 / price_softcover ∧
  notebooks_li = 21 / price_hardcover →
  ¬ (notebooks_ming = notebooks_li) :=
by
  sorry

end cannot_buy_same_number_of_notebooks_l2048_204874


namespace product_of_random_numbers_greater_zero_l2048_204881

noncomputable def random_product_positive_probability : ℝ := 
  let interval_length := 45
  let neg_interval_length := 30
  let pos_interval_length := 15
  let prob_neg := (neg_interval_length : ℝ) / interval_length
  let prob_pos := (pos_interval_length : ℝ) / interval_length
  prob_pos * prob_pos + prob_neg * prob_neg

-- Prove that the probability that the product of two randomly selected numbers
-- from the interval [-30, 15] is greater than zero is 5/9.
theorem product_of_random_numbers_greater_zero : 
  random_product_positive_probability = 5 / 9 := by
  sorry

end product_of_random_numbers_greater_zero_l2048_204881


namespace value_of_f_neg2011_l2048_204802

def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x - 2

theorem value_of_f_neg2011 (a b : ℝ) (h : f 2011 a b = 10) : f (-2011) a b = -14 := by
  sorry

end value_of_f_neg2011_l2048_204802


namespace algebraic_identity_l2048_204872

theorem algebraic_identity (theta : ℝ) (x : ℂ) (n : ℕ) (h1 : 0 < theta) (h2 : theta < π) (h3 : x + x⁻¹ = 2 * Real.cos theta) : 
  x^n + (x⁻¹)^n = 2 * Real.cos (n * theta) :=
by
  sorry

end algebraic_identity_l2048_204872


namespace notebook_problem_l2048_204848

theorem notebook_problem :
  ∃ (x y z : ℕ), x + y + z = 20 ∧ 2 * x + 5 * y + 6 * z = 62 ∧ x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1 ∧ x = 14 :=
by
  sorry

end notebook_problem_l2048_204848


namespace round_time_of_A_l2048_204862

theorem round_time_of_A (T_a T_b : ℝ) 
  (h1 : 4 * T_b = 5 * T_a) 
  (h2 : 4 * T_b = 4 * T_a + 10) : T_a = 10 :=
by
  sorry

end round_time_of_A_l2048_204862


namespace value_of_2_pow_a_l2048_204826

theorem value_of_2_pow_a (a b : ℕ) (ha : 0 < a) (hb : 0 < b) 
(h1 : (2^a)^b = 2^2) (h2 : 2^a * 2^b = 8): 2^a = 2 := 
by
  sorry

end value_of_2_pow_a_l2048_204826


namespace M_inter_N_eq_l2048_204828

def M : Set ℝ := {x | -4 < x ∧ x < 2}
def N : Set ℝ := {x | x^2 - x - 6 < 0}

theorem M_inter_N_eq : {x | -2 < x ∧ x < 2} = M ∩ N := by
  sorry

end M_inter_N_eq_l2048_204828


namespace even_sum_probability_l2048_204806

-- Define the probabilities of even and odd outcomes for each wheel
def probability_even_first_wheel : ℚ := 2 / 3
def probability_odd_first_wheel : ℚ := 1 / 3
def probability_even_second_wheel : ℚ := 3 / 5
def probability_odd_second_wheel : ℚ := 2 / 5

-- Define the probabilities of the scenarios that result in an even sum
def probability_both_even : ℚ := probability_even_first_wheel * probability_even_second_wheel
def probability_both_odd : ℚ := probability_odd_first_wheel * probability_odd_second_wheel

-- Define the total probability of an even sum
def probability_even_sum : ℚ := probability_both_even + probability_both_odd

-- The theorem statement to be proven
theorem even_sum_probability :
  probability_even_sum = 8 / 15 :=
by
  sorry

end even_sum_probability_l2048_204806


namespace min_cost_to_fence_land_l2048_204809

theorem min_cost_to_fence_land (w l : ℝ) (h1 : l = 2 * w) (h2 : 2 * w ^ 2 ≥ 500) : 
  5 * (2 * (l + w)) = 150 * Real.sqrt 10 := 
by
  sorry

end min_cost_to_fence_land_l2048_204809


namespace functional_equation_solution_l2048_204810

noncomputable def f : ℚ → ℚ := sorry

theorem functional_equation_solution :
  (∀ x y : ℚ, f (f x + x * f y) = x + f x * y) →
  (∀ x : ℚ, f x = x) :=
by
  intro h
  sorry

end functional_equation_solution_l2048_204810


namespace tan_sum_identity_l2048_204805

theorem tan_sum_identity (a b : ℝ) (h₁ : Real.tan a = 1/2) (h₂ : Real.tan b = 1/3) : 
  Real.tan (a + b) = 1 := 
by
  sorry

end tan_sum_identity_l2048_204805


namespace sum_x_y_m_l2048_204800

theorem sum_x_y_m (a b x y m : ℕ) (ha : a - b = 3) (hx : x = 10 * a + b) (hy : y = 10 * b + a) (hxy : x^2 - y^2 = m^2) : x + y + m = 178 := sorry

end sum_x_y_m_l2048_204800


namespace stuffed_animal_cost_l2048_204861

variable (S : ℝ)  -- Cost of the stuffed animal
variable (total_cost_after_discount_gave_30_dollars : S * 0.10 = 3.6) 
-- Condition: cost of stuffed animal = $4.44
theorem stuffed_animal_cost :
  S = 4.44 :=
by
  sorry

end stuffed_animal_cost_l2048_204861


namespace sum_of_numbers_l2048_204833

theorem sum_of_numbers (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 100 ≤ y ∧ y < 1000)
(h_eq : 100 * x + y = 7 * x * y) : x + y = 18 :=
sorry

end sum_of_numbers_l2048_204833


namespace cube_root_expression_l2048_204822

theorem cube_root_expression (x : ℝ) (hx : x ≥ 0) : (x * Real.sqrt (x * x^(1/3)))^(1/3) = x^(5/9) :=
by
  sorry

end cube_root_expression_l2048_204822


namespace new_trailer_homes_added_l2048_204837

theorem new_trailer_homes_added (n : ℕ) (h1 : (20 * 20 + 2 * n)/(20 + n) = 14) : n = 10 :=
by
  sorry

end new_trailer_homes_added_l2048_204837


namespace no_bijective_function_l2048_204817

open Set

def is_bijective {α β : Type*} (f : α → β) : Prop :=
  Function.Bijective f

def are_collinear {P : Type*} (A B C : P) : Prop :=
  sorry -- placeholder for the collinearity predicate on points

def are_parallel_or_concurrent {L : Type*} (l₁ l₂ l₃ : L) : Prop :=
  sorry -- placeholder for the condition that lines are parallel or concurrent

theorem no_bijective_function (P : Type*) (D : Type*) :
  ¬ ∃ (f : P → D), is_bijective f ∧
    ∀ A B C : P, are_collinear A B C → are_parallel_or_concurrent (f A) (f B) (f C) :=
by
  sorry

end no_bijective_function_l2048_204817


namespace average_t_value_is_15_l2048_204842

noncomputable def average_of_distinct_t_values (t_vals : List ℤ) : ℤ :=
t_vals.sum / t_vals.length

theorem average_t_value_is_15 :
  average_of_distinct_t_values [8, 14, 18, 20] = 15 :=
by
  sorry

end average_t_value_is_15_l2048_204842


namespace completing_the_square_l2048_204821

theorem completing_the_square :
  ∀ x : ℝ, x^2 - 4 * x - 2 = 0 ↔ (x - 2)^2 = 6 :=
by
  sorry

end completing_the_square_l2048_204821


namespace problem_f_sum_zero_l2048_204808

variable (f : ℝ → ℝ)

def odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def symmetrical (f : ℝ → ℝ) : Prop := ∀ x, f (1 - x) = f x

-- Prove the required sum is zero given the conditions.
theorem problem_f_sum_zero (hf_odd : odd f) (hf_symm : symmetrical f) : 
  f 1 + f 2 + f 3 + f 4 + f 5 = 0 := by
  sorry

end problem_f_sum_zero_l2048_204808


namespace total_money_before_spending_l2048_204840

-- Define the amounts for each friend
variables (J P Q A: ℝ)

-- Define the conditions from the problem
def condition1 := P = 2 * J
def condition2 := Q = P + 20
def condition3 := A = 1.15 * Q
def condition4 := J + P + Q + A = 1211
def cost_of_item : ℝ := 1200

-- The total amount before buying the item
theorem total_money_before_spending (J P Q A : ℝ)
  (h1 : condition1 J P)
  (h2 : condition2 P Q)
  (h3 : condition3 Q A)
  (h4 : condition4 J P Q A) : 
  J + P + Q + A - cost_of_item = 11 :=
by
  sorry

end total_money_before_spending_l2048_204840


namespace grocer_sales_l2048_204849

theorem grocer_sales 
  (s1 s2 s3 s4 s5 s6 s7 s8 sales : ℝ)
  (h_sales_1 : s1 = 5420)
  (h_sales_2 : s2 = 5660)
  (h_sales_3 : s3 = 6200)
  (h_sales_4 : s4 = 6350)
  (h_sales_5 : s5 = 6500)
  (h_sales_6 : s6 = 6780)
  (h_sales_7 : s7 = 7000)
  (h_sales_8 : s8 = 7200)
  (h_avg : (5420 + 5660 + 6200 + 6350 + 6500 + 6780 + 7000 + 7200 + 2 * sales) / 10 = 6600) :
  sales = 9445 := 
  by 
  sorry

end grocer_sales_l2048_204849


namespace distance_home_to_school_l2048_204863

def speed_walk := 5
def speed_car := 15
def time_difference := 2

variable (d : ℝ) -- distance from home to school
variable (T1 T2 : ℝ) -- T1: time to school, T2: time back home

-- Conditions
axiom h1 : T1 = d / speed_walk / 2 + d / speed_car / 2
axiom h2 : d = speed_car * T2 / 3 + speed_walk * 2 * T2 / 3
axiom h3 : T1 = T2 + time_difference

-- Theorem to prove
theorem distance_home_to_school : d = 150 :=
by
  sorry

end distance_home_to_school_l2048_204863


namespace nickels_count_l2048_204835

theorem nickels_count (N Q : ℕ) 
  (h_eq : N = Q) 
  (h_total_value : 5 * N + 25 * Q = 1200) :
  N = 40 := 
by 
  sorry

end nickels_count_l2048_204835


namespace trains_meet_480_km_away_l2048_204864

-- Define the conditions
def bombay_express_speed : ℕ := 60 -- speed in km/h
def rajdhani_express_speed : ℕ := 80 -- speed in km/h
def bombay_express_start_time : ℕ := 1430 -- 14:30 in 24-hour format
def rajdhani_express_start_time : ℕ := 1630 -- 16:30 in 24-hour format

-- Define the function to calculate the meeting point distance
noncomputable def meeting_distance (bombay_speed rajdhani_speed : ℕ) (bombay_start rajdhani_start : ℕ) : ℕ :=
  let t := 6 -- time taken for Rajdhani to catch up in hours, derived from the solution
  rajdhani_speed * t

-- The statement we need to prove:
theorem trains_meet_480_km_away :
  meeting_distance bombay_express_speed rajdhani_express_speed bombay_express_start_time rajdhani_express_start_time = 480 := by
  sorry

end trains_meet_480_km_away_l2048_204864


namespace bridget_apples_l2048_204894

theorem bridget_apples :
  ∃ x : ℕ, (x - x / 3 - 4) = 6 :=
by
  sorry

end bridget_apples_l2048_204894


namespace sum_five_consecutive_l2048_204878

theorem sum_five_consecutive (n : ℤ) : (n + (n + 1) + (n + 2) + (n + 3) + (n + 4)) = 5 * n + 10 := by
  sorry

end sum_five_consecutive_l2048_204878


namespace fg_of_2_l2048_204820

def f (x : ℝ) : ℝ := x^2 + 1
def g (x : ℝ) : ℝ := 2 * x - 1

theorem fg_of_2 : f (g 2) = 10 :=
by
  sorry

end fg_of_2_l2048_204820


namespace sum_of_sequence_l2048_204834

def a (n : ℕ) : ℕ := 2 * n + 1 + 2^n

def S (n : ℕ) : ℕ := (Finset.range n).sum (λ k => a (k + 1))

theorem sum_of_sequence (n : ℕ) : S n = n^2 + 2 * n + 2^(n + 1) - 2 := 
by 
  sorry

end sum_of_sequence_l2048_204834


namespace num_divisors_630_l2048_204825

theorem num_divisors_630 : ∃ d : ℕ, (d = 24) ∧ ∀ n : ℕ, (∃ (a b c d : ℕ), (n = 2^a * 3^b * 5^c * 7^d) ∧ a ≤ 1 ∧ b ≤ 2 ∧ c ≤ 1 ∧ d ≤ 1) ↔ (n ∣ 630) := sorry

end num_divisors_630_l2048_204825


namespace triangle_inequality_l2048_204879

theorem triangle_inequality (a b c : ℝ) (h₁ : a + b > c) (h₂ : b + c > a) (h₃ : c + a > b) : 
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := 
by
  sorry

end triangle_inequality_l2048_204879


namespace perpendicular_lines_condition_l2048_204812

theorem perpendicular_lines_condition (k : ℝ) : 
  (k = 5 → (∃ x y : ℝ, k * x + 5 * y - 2 = 0 ∧ (4 - k) * x + y - 7 = 0 ∧ x ≠ 0 ∧ y ≠ 0)) ∧
  (∀ x y : ℝ, k * x + 5 * y - 2 = 0 ∧ (4 - k) * x + y - 7 = 0 → (k = 5 ∨ k = -1)) :=
sorry

end perpendicular_lines_condition_l2048_204812


namespace gym_monthly_cost_l2048_204866

theorem gym_monthly_cost (down_payment total_cost total_months : ℕ) (h_down_payment : down_payment = 50) (h_total_cost : total_cost = 482) (h_total_months : total_months = 36) : 
  (total_cost - down_payment) / total_months = 12 := by 
  sorry

end gym_monthly_cost_l2048_204866


namespace gcd_m_n_l2048_204895

def m : ℕ := 3333333
def n : ℕ := 66666666

theorem gcd_m_n : Nat.gcd m n = 3 := by
  sorry

end gcd_m_n_l2048_204895


namespace find_n_l2048_204898

def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 3 ∧ ∀ n : ℕ, a (n + 1) = -1 / (a n + 1)

theorem find_n (a : ℕ → ℚ) (h : seq a) : ∃ n : ℕ, a n = 3 ∧ n = 16 :=
by
  sorry

end find_n_l2048_204898


namespace compare_xyz_l2048_204860

open Real

theorem compare_xyz (x y z : ℝ) : x = Real.log π → y = log 2 / log 5 → z = exp (-1 / 2) → y < z ∧ z < x := by
  intros h_x h_y h_z
  sorry

end compare_xyz_l2048_204860


namespace solve_system_l2048_204875

variable {x y z : ℝ}

theorem solve_system :
  (y + z = 16 - 4 * x) →
  (x + z = -18 - 4 * y) →
  (x + y = 13 - 4 * z) →
  2 * x + 2 * y + 2 * z = 11 / 3 :=
by
  intros h1 h2 h3
  -- proof skips, to be completed
  sorry

end solve_system_l2048_204875


namespace Ram_Shyam_weight_ratio_l2048_204858

theorem Ram_Shyam_weight_ratio :
  ∃ (R S : ℝ), 
    (1.10 * R + 1.21 * S = 82.8) ∧ 
    (1.15 * (R + S) = 82.8) ∧ 
    (R / S = 1.20) :=
by {
  sorry
}

end Ram_Shyam_weight_ratio_l2048_204858


namespace number_of_pairs_satisfying_equation_l2048_204855

theorem number_of_pairs_satisfying_equation :
  ∃ n : ℕ, n = 4998 ∧ (∀ x y : ℤ, x^2 + 7 * x * y + 6 * y^2 = 15^50 → (x, y) ≠ (0, 0)) ∧
  (∀ x y : ℤ, x^2 + 7 * x * y + 6 * y^2 = 15^50 → ((x + 6 * y) = (3 * 5) ^ a ∧ (x + y) = (3 ^ (50 - a) * 5 ^ (50 - b)) ∨
        (x + 6 * y) = -(3 * 5) ^ a ∧ (x + y) = -(3 ^ (50 - a) * 5 ^ (50 - b)) → (a + b = 50))) :=
sorry

end number_of_pairs_satisfying_equation_l2048_204855


namespace math_problem_l2048_204838

noncomputable def canA_red_balls := 3
noncomputable def canA_black_balls := 4
noncomputable def canB_red_balls := 2
noncomputable def canB_black_balls := 3

noncomputable def prob_event_A := canA_red_balls / (canA_red_balls + canA_black_balls) -- P(A)
noncomputable def prob_event_B := 
  (canA_red_balls / (canA_red_balls + canA_black_balls)) * (canB_red_balls + 1) / (6) +
  (canA_black_balls / (canA_red_balls + canA_black_balls)) * (canB_red_balls) / (6) -- P(B)

theorem math_problem : 
  (prob_event_A = 3 / 7) ∧ 
  (prob_event_B = 17 / 42) ∧
  (¬ (prob_event_A * prob_event_B = (3 / 7) * (17 / 42))) ∧
  ((prob_event_A * (canB_red_balls + 1) / 6) / prob_event_A = 1 / 2) := by
  repeat { sorry }

end math_problem_l2048_204838


namespace inverse_solution_correct_l2048_204892

noncomputable def f (a b c x : ℝ) : ℝ :=
  1 / (a * x^2 + b * x + c)

theorem inverse_solution_correct (a b c x : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  f a b c x = 1 ↔ x = (-b + Real.sqrt (b^2 - 4 * a * (c - 1))) / (2 * a) ∨
               x = (-b - Real.sqrt (b^2 - 4 * a * (c - 1))) / (2 * a) :=
by
  sorry

end inverse_solution_correct_l2048_204892


namespace intersection_of_A_and_B_l2048_204891

def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {x | 1 < x ∧ x < 2} := by
  sorry

end intersection_of_A_and_B_l2048_204891


namespace largest_number_value_l2048_204877

theorem largest_number_value (x : ℕ) (h : 7 * x - 3 * x = 40) : 7 * x = 70 :=
by
  sorry

end largest_number_value_l2048_204877


namespace compose_f_g_f_l2048_204831

def f (x : ℝ) : ℝ := 2 * x + 5
def g (x : ℝ) : ℝ := 3 * x + 4

theorem compose_f_g_f (x : ℝ) : f (g (f 3)) = 79 := by
  sorry

end compose_f_g_f_l2048_204831


namespace term_sequence_10th_l2048_204803

theorem term_sequence_10th :
  let a (n : ℕ) := (-1:ℚ)^(n+1) * (2*n)/(2*n + 1)
  a 10 = -20/21 := 
by
  sorry

end term_sequence_10th_l2048_204803


namespace xyz_value_l2048_204836

-- Define the real numbers x, y, and z
variables (x y z : ℝ)

-- Condition 1
def condition1 := (x + y + z) * (x * y + x * z + y * z) = 49

-- Condition 2
def condition2 := x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19

-- Main theorem statement
theorem xyz_value (h1 : condition1 x y z) (h2 : condition2 x y z) : x * y * z = 10 :=
sorry

end xyz_value_l2048_204836


namespace expression_value_l2048_204844

theorem expression_value (y : ℤ) (h : y = 5) : (y^2 - y - 12) / (y - 4) = 8 :=
by
  rw[h]
  sorry

end expression_value_l2048_204844


namespace avg_temp_Brookdale_l2048_204830

noncomputable def avg_temp (temps : List ℚ) : ℚ :=
  temps.sum / temps.length

theorem avg_temp_Brookdale : avg_temp [51, 67, 64, 61, 50, 65, 47] = 57.9 :=
by
  sorry

end avg_temp_Brookdale_l2048_204830


namespace distance_ratio_l2048_204856

variables (dw dr : ℝ)

theorem distance_ratio (h1 : 4 * (dw / 4) + 8 * (dr / 8) = 8)
  (h2 : dw + dr = 8)
  (h3 : (dw / 4) + (dr / 8) = 1.5) :
  dw / dr = 1 :=
by
  sorry

end distance_ratio_l2048_204856


namespace p_more_than_q_l2048_204883

def stamps (p q : ℕ) : Prop :=
  p / q = 7 / 4 ∧ (p - 8) / (q + 8) = 6 / 5

theorem p_more_than_q (p q : ℕ) (h : stamps p q) : p - 8 - (q + 8) = 8 :=
by {
  sorry
}

end p_more_than_q_l2048_204883


namespace part_1_part_3_500_units_part_3_1000_units_l2048_204853

/-- Define the pricing function P as per the given conditions -/
def P (x : ℕ) : ℝ :=
  if 0 < x ∧ x ≤ 100 then 60
  else if 100 < x ∧ x <= 550 then 62 - 0.02 * x
  else 51

/-- Verify that ordering 550 units results in a per-unit price of 51 yuan -/
theorem part_1 : P 550 = 51 := sorry

/-- Compute profit for given order quantities -/
def profit (x : ℕ) : ℝ :=
  x * (P x - 40)

/-- Verify that an order of 500 units results in a profit of 6000 yuan -/
theorem part_3_500_units : profit 500 = 6000 := sorry

/-- Verify that an order of 1000 units results in a profit of 11000 yuan -/
theorem part_3_1000_units : profit 1000 = 11000 := sorry

end part_1_part_3_500_units_part_3_1000_units_l2048_204853


namespace abs_sum_eq_3_given_condition_l2048_204846

theorem abs_sum_eq_3_given_condition (m n p : ℤ)
  (h : |m - n|^3 + |p - m|^5 = 1) :
  |p - m| + |m - n| + 2 * |n - p| = 3 :=
sorry

end abs_sum_eq_3_given_condition_l2048_204846


namespace find_integer_pairs_l2048_204870

theorem find_integer_pairs (x y : ℤ) (h : x^3 - y^3 = 2 * x * y + 8) : 
  (x = 0 ∧ y = -2) ∨ (x = 2 ∧ y = 0) := 
by {
  sorry
}

end find_integer_pairs_l2048_204870


namespace relationship_abc_l2048_204824

noncomputable def a : ℝ := Real.sqrt 3
noncomputable def b : ℝ := Real.sqrt 15 - Real.sqrt 7
noncomputable def c : ℝ := Real.sqrt 11 - Real.sqrt 3

theorem relationship_abc : a > c ∧ c > b := 
by
  unfold a b c
  sorry

end relationship_abc_l2048_204824


namespace shoe_cost_l2048_204804

def initial_amount : ℕ := 91
def cost_sweater : ℕ := 24
def cost_tshirt : ℕ := 6
def amount_left : ℕ := 50
def cost_shoes : ℕ := 11

theorem shoe_cost :
  initial_amount - (cost_sweater + cost_tshirt) - amount_left = cost_shoes :=
by
  sorry

end shoe_cost_l2048_204804


namespace proof_statement_l2048_204814

open Classical

variable (Person : Type) (Nationality : Type) (Occupation : Type)

variable (A B C D : Person)
variable (UnitedKingdom UnitedStates Germany France : Nationality)
variable (Doctor Teacher : Occupation)

variable (nationality : Person → Nationality)
variable (occupation : Person → Occupation)
variable (can_swim : Person → Prop)
variable (play_sports_together : Person → Person → Prop)

noncomputable def proof :=
  (nationality A = UnitedKingdom ∧ nationality D = Germany)

axiom condition1 : occupation A = Doctor ∧ ∃ x : Person, nationality x = UnitedStates ∧ occupation x = Doctor
axiom condition2 : occupation B = Teacher ∧ ∃ x : Person, nationality x = Germany ∧ occupation x = Teacher 
axiom condition3 : can_swim C ∧ ∀ x : Person, nationality x = Germany → ¬ can_swim x
axiom condition4 : ∃ x : Person, nationality x = France ∧ play_sports_together A x

theorem proof_statement : 
  (nationality A = UnitedKingdom ∧ nationality D = Germany) :=
by {
  sorry
}

end proof_statement_l2048_204814


namespace total_travel_distance_l2048_204876

noncomputable def total_distance_traveled (DE DF : ℝ) : ℝ :=
  let EF := Real.sqrt (DE^2 - DF^2)
  DE + EF + DF

theorem total_travel_distance
  (DE DF : ℝ)
  (hDE : DE = 4500)
  (hDF : DF = 4000)
  : total_distance_traveled DE DF = 10560.992 :=
by
  rw [hDE, hDF]
  unfold total_distance_traveled
  norm_num
  sorry

end total_travel_distance_l2048_204876


namespace johns_yearly_grass_cutting_cost_l2048_204868

-- Definitions of the conditions
def initial_height : ℝ := 2.0
def growth_rate : ℝ := 0.5
def cutting_height : ℝ := 4.0
def cost_per_cut : ℝ := 100.0
def months_per_year : ℝ := 12.0

-- Formulate the statement
theorem johns_yearly_grass_cutting_cost :
  let months_to_grow : ℝ := (cutting_height - initial_height) / growth_rate
  let cuts_per_year : ℝ := months_per_year / months_to_grow
  let total_cost_per_year : ℝ := cuts_per_year * cost_per_cut
  total_cost_per_year = 300.0 :=
by
  sorry

end johns_yearly_grass_cutting_cost_l2048_204868


namespace radius_of_circle_eq_l2048_204811

-- Define the given quadratic equation representing the circle
noncomputable def circle_eq (x y : ℝ) : ℝ :=
  16 * x^2 - 32 * x + 16 * y^2 - 48 * y + 68

-- State that the radius of the circle given by the equation is 1
theorem radius_of_circle_eq : ∃ r, (∀ x y, circle_eq x y = 0 ↔ (x - 1)^2 + (y - 1.5)^2 = r^2) ∧ r = 1 :=
by 
  use 1
  sorry

end radius_of_circle_eq_l2048_204811


namespace right_triangle_legs_sum_l2048_204886

theorem right_triangle_legs_sum : 
  ∃ (x : ℕ), (x^2 + (x + 1)^2 = 41^2) ∧ (x + (x + 1) = 57) :=
by
  sorry

end right_triangle_legs_sum_l2048_204886


namespace tan_alpha_eq_inv_3_tan_alpha_add_beta_eq_1_l2048_204813

open Real

axiom sin_add_half_pi_div_4_eq_zero (α : ℝ) : 
  sin (α + π / 4) + 2 * sin (α - π / 4) = 0

axiom tan_sub_half_pi_div_4_eq_inv_3 (β : ℝ) : 
  tan (π / 4 - β) = 1 / 3

theorem tan_alpha_eq_inv_3 (α : ℝ) (h : sin (α + π / 4) + 2 * sin (α - π / 4) = 0) : 
  tan α = 1 / 3 := sorry

theorem tan_alpha_add_beta_eq_1 (α β : ℝ) 
  (h1 : tan α = 1 / 3) (h2 : tan (π / 4 - β) = 1 / 3) : 
  tan (α + β) = 1 := sorry

end tan_alpha_eq_inv_3_tan_alpha_add_beta_eq_1_l2048_204813


namespace new_team_average_weight_l2048_204897

theorem new_team_average_weight :
  let original_team_weight := 7 * 94
  let new_players_weight := 110 + 60
  let new_total_weight := original_team_weight + new_players_weight
  let new_player_count := 9
  (new_total_weight / new_player_count) = 92 :=
by
  let original_team_weight := 7 * 94
  let new_players_weight := 110 + 60
  let new_total_weight := original_team_weight + new_players_weight
  let new_player_count := 9
  sorry

end new_team_average_weight_l2048_204897


namespace nina_basketball_cards_l2048_204880

theorem nina_basketball_cards (cost_toy cost_shirt cost_card total_spent : ℕ) (n_toys n_shirts n_cards n_packs_result : ℕ)
  (h1 : cost_toy = 10)
  (h2 : cost_shirt = 6)
  (h3 : cost_card = 5)
  (h4 : n_toys = 3)
  (h5 : n_shirts = 5)
  (h6 : total_spent = 70)
  (h7 : n_packs_result =  2)
  : (3 * cost_toy + 5 * cost_shirt + n_cards * cost_card = total_spent) → n_cards = n_packs_result :=
by
  sorry

end nina_basketball_cards_l2048_204880


namespace min_max_values_in_interval_l2048_204845

def func (x y : ℝ) : ℝ := 3 * x^2 * y - 2 * x * y^2

theorem min_max_values_in_interval :
  (∀ x y, 0 ≤ x → x ≤ 1 → 0 ≤ y → y ≤ 1 → func x y ≥ -1/3) ∧
  (∃ x y, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ func x y = -1/3) ∧
  (∀ x y, 0 ≤ x → x ≤ 1 → 0 ≤ y → y ≤ 1 → func x y ≤ 9/8) ∧
  (∃ x y, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ func x y = 9/8) :=
by
  sorry

end min_max_values_in_interval_l2048_204845


namespace square_diff_l2048_204829

-- Definitions and conditions from the problem
def three_times_sum_eq (a b : ℝ) : Prop := 3 * (a + b) = 18
def diff_eq (a b : ℝ) : Prop := a - b = 4

-- Goal to prove that a^2 - b^2 = 24 under the given conditions
theorem square_diff (a b : ℝ) (h₁ : three_times_sum_eq a b) (h₂ : diff_eq a b) : a^2 - b^2 = 24 :=
sorry

end square_diff_l2048_204829


namespace tobias_mowed_four_lawns_l2048_204888

-- Let’s define the conditions
def shoe_cost : ℕ := 95
def allowance_per_month : ℕ := 5
def savings_months : ℕ := 3
def lawn_mowing_charge : ℕ := 15
def shoveling_charge : ℕ := 7
def change_after_purchase : ℕ := 15
def num_driveways_shoveled : ℕ := 5

-- Total money Tobias had before buying the shoes
def total_money : ℕ := shoe_cost + change_after_purchase

-- Money saved from allowance
def money_from_allowance : ℕ := allowance_per_month * savings_months

-- Money earned from shoveling driveways
def money_from_shoveling : ℕ := shoveling_charge * num_driveways_shoveled

-- Money earned from mowing lawns
def money_from_mowing : ℕ := total_money - money_from_allowance - money_from_shoveling

-- Number of lawns mowed
def num_lawns_mowed : ℕ := money_from_mowing / lawn_mowing_charge

-- The theorem stating the number of lawns mowed is 4
theorem tobias_mowed_four_lawns : num_lawns_mowed = 4 :=
by
  sorry

end tobias_mowed_four_lawns_l2048_204888


namespace triangle_angle_B_l2048_204885

theorem triangle_angle_B {A B C : ℝ} (h1 : A = 60) (h2 : B = 2 * C) (h3 : A + B + C = 180) : B = 80 :=
sorry

end triangle_angle_B_l2048_204885


namespace proof_expr_l2048_204859

theorem proof_expr (a b c : ℤ) (h1 : a - b = 3) (h2 : b - c = 2) : (a - c)^2 + 3 * a + 1 - 3 * c = 41 := by {
  sorry
}

end proof_expr_l2048_204859


namespace roots_of_polynomial_l2048_204801

theorem roots_of_polynomial :
  ∀ x : ℝ, (x^2 - 5*x + 6)*(x)*(x-5) = 0 ↔ x = 0 ∨ x = 2 ∨ x = 3 ∨ x = 5 :=
by
  sorry

end roots_of_polynomial_l2048_204801


namespace min_sum_of_consecutive_natural_numbers_l2048_204865

theorem min_sum_of_consecutive_natural_numbers (a b c : ℕ) 
  (h1 : a + 1 = b)
  (h2 : a + 2 = c)
  (h3 : a % 9 = 0)
  (h4 : b % 8 = 0)
  (h5 : c % 7 = 0) :
  a + b + c = 1488 :=
sorry

end min_sum_of_consecutive_natural_numbers_l2048_204865


namespace round_balloons_burst_l2048_204867

theorem round_balloons_burst :
  let round_balloons := 5 * 20
  let long_balloons := 4 * 30
  let total_balloons := round_balloons + long_balloons
  let balloons_left := 215
  ((total_balloons - balloons_left) = 5) :=
by 
  sorry

end round_balloons_burst_l2048_204867


namespace find_a3_l2048_204818

-- Define the sequence sum S_n
def S (n : ℕ) : ℚ := (n + 1) / (n + 2)

-- Define the sequence term a_n using S_n
def a (n : ℕ) : ℚ :=
  if h : n = 1 then S 1 else S n - S (n - 1)

-- State the theorem to find the value of a_3
theorem find_a3 : a 3 = 1 / 20 :=
by
  -- The proof is omitted, use sorry to skip it
  sorry

end find_a3_l2048_204818


namespace original_amount_l2048_204841

theorem original_amount (x : ℝ) (h : 0.25 * x = 200) : x = 800 := 
by
  sorry

end original_amount_l2048_204841


namespace largest_number_of_square_plots_l2048_204827

theorem largest_number_of_square_plots (n : ℕ) 
  (field_length : ℕ := 30) 
  (field_width : ℕ := 60) 
  (total_fence : ℕ := 2400) 
  (square_length : ℕ := field_length / n) 
  (fencing_required : ℕ := 60 * n) :
  field_length % n = 0 → 
  field_width % square_length = 0 → 
  fencing_required = total_fence → 
  2 * n^2 = 3200 :=
by
  intros h1 h2 h3
  sorry

end largest_number_of_square_plots_l2048_204827


namespace compositeQuotientCorrect_l2048_204899

namespace CompositeNumbersProof

def firstFiveCompositesProduct : ℕ :=
  21 * 22 * 24 * 25 * 26

def subsequentFiveCompositesProduct : ℕ :=
  27 * 28 * 30 * 32 * 33

def compositeQuotient : ℚ :=
  firstFiveCompositesProduct / subsequentFiveCompositesProduct

theorem compositeQuotientCorrect : compositeQuotient = 1 / 1964 := by sorry

end CompositeNumbersProof

end compositeQuotientCorrect_l2048_204899


namespace number_of_sequences_l2048_204815

theorem number_of_sequences (n k : ℕ) (h₁ : 1 ≤ k) (h₂ : k ≤ n) :
  ∃ C : ℕ, C = Nat.choose (Nat.floor ((n + 2 - k) / 2) + k - 1) k :=
sorry

end number_of_sequences_l2048_204815


namespace problem_arithmetic_sequence_l2048_204847

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) := a + d * (n - 1)

theorem problem_arithmetic_sequence (a d : ℝ) (h₁ : d < 0) (h₂ : (arithmetic_sequence a d 1)^2 = (arithmetic_sequence a d 9)^2):
  (arithmetic_sequence a d 5) = 0 :=
by
  -- This is where the proof would go
  sorry

end problem_arithmetic_sequence_l2048_204847


namespace sequences_properties_l2048_204819

-- Definition of sequences and their properties
variable {n : ℕ}

noncomputable def S (n : ℕ) : ℕ := n^2 - n
noncomputable def a (n : ℕ) : ℕ := if n = 1 then 0 else 2 * n - 2
noncomputable def b (n : ℕ) : ℕ := 3^(n-1)
noncomputable def c (n : ℕ) : ℕ := (2 * (n - 1)) / 3^(n - 1)
noncomputable def T (n : ℕ) : ℕ := 3 / 2 - (2 * n + 1) / (2 * 3^(n-1))

-- Main theorem
theorem sequences_properties (n : ℕ) (hn : n > 0) :
  S n = n^2 - n ∧
  (∀ n, a n = if n = 1 then 0 else 2 * n - 2) ∧
  (∀ n, b n = 3^(n-1)) ∧
  (∀ n, T n = 3 / 2 - (2 * n + 1) / (2 * 3^(n-1))) :=
by sorry

end sequences_properties_l2048_204819


namespace no_integer_solution_for_Q_square_l2048_204890

def Q (x : ℤ) : ℤ := x^4 + 5*x^3 + 10*x^2 + 5*x + 56

theorem no_integer_solution_for_Q_square :
  ∀ x : ℤ, ∃ k : ℤ, Q x = k^2 → false :=
by
  sorry

end no_integer_solution_for_Q_square_l2048_204890


namespace rowing_distance_l2048_204893

theorem rowing_distance (D : ℝ) 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (total_time : ℝ) 
  (downstream_speed : ℝ := boat_speed + stream_speed) 
  (upstream_speed : ℝ := boat_speed - stream_speed)
  (downstream_time : ℝ := D / downstream_speed)
  (upstream_time : ℝ := D / upstream_speed)
  (round_trip_time : ℝ := downstream_time + upstream_time) 
  (h1 : boat_speed = 16) 
  (h2 : stream_speed = 2) 
  (h3 : total_time = 914.2857142857143)
  (h4 : round_trip_time = total_time) :
  D = 720 :=
by sorry

end rowing_distance_l2048_204893


namespace rate_per_square_meter_l2048_204851

-- Define the conditions
def length (L : ℝ) := L = 8
def width (W : ℝ) := W = 4.75
def total_cost (C : ℝ) := C = 34200
def area (A : ℝ) (L W : ℝ) := A = L * W
def rate (R C A : ℝ) := R = C / A

-- The theorem to prove
theorem rate_per_square_meter (L W C A R : ℝ) 
  (hL : length L) (hW : width W) (hC : total_cost C) (hA : area A L W) : 
  rate R C A :=
by
  -- By the conditions, length is 8, width is 4.75, and total cost is 34200.
  simp [length, width, total_cost, area, rate] at hL hW hC hA ⊢
  -- It remains to calculate the rate and use conditions
  have hA : A = L * W := hA
  rw [hL, hW] at hA
  have hA' : A = 8 * 4.75 := by simp [hA]
  rw [hA']
  simp [rate]
  sorry -- The detailed proof is omitted.

end rate_per_square_meter_l2048_204851


namespace count_marble_pairs_l2048_204816

-- Define conditions:
structure Marbles :=
(red : ℕ) (green : ℕ) (blue : ℕ) (yellow : ℕ) (white : ℕ)

def tomsMarbles : Marbles :=
  { red := 1, green := 1, blue := 1, yellow := 3, white := 2 }

-- Define a function to count pairs of marbles:
def count_pairs (m : Marbles) : ℕ :=
  -- Count pairs of identical marbles:
  (if m.yellow >= 2 then 1 else 0) + 
  (if m.white >= 2 then 1 else 0) +
  -- Count pairs of different colored marbles:
  (Nat.choose 5 2)

-- Theorem statement:
theorem count_marble_pairs : count_pairs tomsMarbles = 12 :=
  by
    sorry

end count_marble_pairs_l2048_204816


namespace biology_marks_correct_l2048_204882

-- Define the known marks in other subjects
def math_marks : ℕ := 76
def science_marks : ℕ := 65
def social_studies_marks : ℕ := 82
def english_marks : ℕ := 62

-- Define the total number of subjects
def total_subjects : ℕ := 5

-- Define the average marks
def average_marks : ℕ := 74

-- Calculate the total marks of the known four subjects
def total_known_marks : ℕ := math_marks + science_marks + social_studies_marks + english_marks

-- Define a variable to represent the marks in biology
def biology_marks : ℕ := 370 - total_known_marks

-- Statement to prove
theorem biology_marks_correct : biology_marks = 85 := by
  sorry

end biology_marks_correct_l2048_204882
