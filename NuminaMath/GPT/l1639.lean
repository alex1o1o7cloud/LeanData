import Mathlib

namespace actors_in_one_hour_l1639_163966

theorem actors_in_one_hour (actors_per_set : ℕ) (minutes_per_set : ℕ) (total_minutes : ℕ) :
  actors_per_set = 5 → minutes_per_set = 15 → total_minutes = 60 →
  (total_minutes / minutes_per_set) * actors_per_set = 20 :=
by
  intros h1 h2 h3
  sorry

end actors_in_one_hour_l1639_163966


namespace origin_in_circle_m_gt_5_l1639_163990

theorem origin_in_circle_m_gt_5 (m : ℝ) : ((0 - 1)^2 + (0 + 2)^2 < m) → (m > 5) :=
by
  intros h
  sorry

end origin_in_circle_m_gt_5_l1639_163990


namespace evaluate_expression_l1639_163910

theorem evaluate_expression : 5000 * (5000 ^ 3000) = 5000 ^ 3001 := 
by
  sorry

end evaluate_expression_l1639_163910


namespace correct_calculation_l1639_163936

-- Definition of the conditions
def condition1 (a : ℕ) : Prop := a^2 * a^3 = a^6
def condition2 (a : ℕ) : Prop := (a^2)^10 = a^20
def condition3 (a : ℕ) : Prop := (2 * a) * (3 * a) = 6 * a
def condition4 (a : ℕ) : Prop := a^12 / a^2 = a^6

-- The main theorem to state that condition2 is the correct calculation
theorem correct_calculation (a : ℕ) : condition2 a :=
sorry

end correct_calculation_l1639_163936


namespace trig_identity_l1639_163985

theorem trig_identity : 4 * Real.sin (20 * Real.pi / 180) + Real.tan (20 * Real.pi / 180) = Real.sqrt 3 := 
by sorry

end trig_identity_l1639_163985


namespace initially_calculated_average_weight_l1639_163951

-- Define the conditions
def num_boys : ℕ := 20
def correct_average_weight : ℝ := 58.7
def misread_weight : ℝ := 56
def correct_weight : ℝ := 62
def weight_difference : ℝ := correct_weight - misread_weight

-- State the goal
theorem initially_calculated_average_weight :
  let correct_total_weight := correct_average_weight * num_boys
  let initial_total_weight := correct_total_weight - weight_difference
  let initially_calculated_weight := initial_total_weight / num_boys
  initially_calculated_weight = 58.4 :=
by
  sorry

end initially_calculated_average_weight_l1639_163951


namespace salary_increase_l1639_163953

variable (S : ℝ) (P : ℝ)

theorem salary_increase (h1 : 1.16 * S = 406) (h2 : 350 + 350 * P = 420) : P * 100 = 20 := 
by
  sorry

end salary_increase_l1639_163953


namespace mean_home_runs_correct_l1639_163988

-- Define the total home runs in April
def total_home_runs_April : ℕ := 5 * 4 + 6 * 4 + 8 * 2 + 10

-- Define the total home runs in May
def total_home_runs_May : ℕ := 5 * 2 + 6 * 2 + 8 * 3 + 10 * 2 + 11

-- Define the total number of top hitters/players
def total_players : ℕ := 12

-- Define the total home runs over two months
def total_home_runs : ℕ := total_home_runs_April + total_home_runs_May

-- Calculate the mean number of home runs
def mean_home_runs : ℚ := total_home_runs / total_players

-- Prove that the calculated mean is equal to the expected result
theorem mean_home_runs_correct : mean_home_runs = 12.08 := by
  sorry

end mean_home_runs_correct_l1639_163988


namespace product_of_two_numbers_l1639_163922

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 26) (h2 : x - y = 8) : x * y = 153 :=
by
  -- proof goes here
  sorry

end product_of_two_numbers_l1639_163922


namespace tangent_lengths_l1639_163975

noncomputable def internal_tangent_length (r1 r2 d : ℝ) : ℝ :=
  Real.sqrt (d^2 - (r1 + r2)^2)

noncomputable def external_tangent_length (r1 r2 d : ℝ) : ℝ :=
  Real.sqrt (d^2 - (r1 - r2)^2)

theorem tangent_lengths (r1 r2 d : ℝ) (h_r1 : r1 = 8) (h_r2 : r2 = 10) (h_d : d = 50) :
  internal_tangent_length r1 r2 d = 46.67 ∧ external_tangent_length r1 r2 d = 49.96 :=
by
  sorry

end tangent_lengths_l1639_163975


namespace find_possible_sets_C_l1639_163970

open Set

def A : Set ℕ := {3, 4}
def B : Set ℕ := {0, 1, 2, 3, 4}
def possible_C_sets : Set (Set ℕ) :=
  { {3, 4}, {3, 4, 0}, {3, 4, 1}, {3, 4, 2}, {3, 4, 0, 1},
    {3, 4, 0, 2}, {3, 4, 1, 2}, {0, 1, 2, 3, 4} }

theorem find_possible_sets_C :
  {C : Set ℕ | A ⊆ C ∧ C ⊆ B} = possible_C_sets :=
by
  sorry

end find_possible_sets_C_l1639_163970


namespace P_plus_Q_l1639_163979

theorem P_plus_Q (P Q : ℝ) (h : (P / (x - 3) + Q * (x - 2)) = (-5 * x^2 + 18 * x + 27) / (x - 3)) : P + Q = 31 := 
by {
  sorry
}

end P_plus_Q_l1639_163979


namespace problem_I_problem_II_l1639_163916

-- Define the function f(x) = |x+1| + |x+m+1|
def f (x : ℝ) (m : ℝ) : ℝ := |x+1| + |x+(m+1)|

-- Define the problem (Ⅰ): f(x) ≥ |m-2| for all x implies m ≥ 1
theorem problem_I (m : ℝ) (h : ∀ x : ℝ, f x m ≥ |m-2|) : m ≥ 1 := sorry

-- Define the problem (Ⅱ): Find the solution set for f(-x) < 2m
theorem problem_II (m : ℝ) :
  (m ≤ 0 → ∀ x : ℝ, ¬ (f (-x) m < 2 * m)) ∧
  (m > 0 → ∀ x : ℝ, (1 - m / 2 < x ∧ x < 3 * m / 2 + 1) ↔ f (-x) m < 2 * m) := sorry

end problem_I_problem_II_l1639_163916


namespace quadratic_inequality_solution_l1639_163935

theorem quadratic_inequality_solution (x : ℝ) : x^2 + 3 * x - 18 < 0 ↔ -6 < x ∧ x < 3 := 
sorry

end quadratic_inequality_solution_l1639_163935


namespace problem_solution_l1639_163915

open Real

def system_satisfied (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ k : ℕ, k < n → 
    (a (2 * k + 1) = (1 / a (2 * (k + n) - 1) + 1 / a (2 * k + 2))) ∧ 
    (a (2 * k + 2) = a (2 * k + 1) + a (2 * k + 3))

theorem problem_solution (n : ℕ) (a : ℕ → ℝ) 
  (h1 : n ≥ 4)
  (h2 : ∀ k, 0 ≤ k → k < 2 * n → a k > 0)
  (h3 : system_satisfied a n) :
  ∀ k, 0 ≤ k ∧ k < n → a (2 * k + 1) = 1 ∧ a (2 * k + 2) = 2 :=
sorry

end problem_solution_l1639_163915


namespace convert_fraction_to_decimal_l1639_163919

theorem convert_fraction_to_decimal : (3 / 40 : ℝ) = 0.075 := 
by
  sorry

end convert_fraction_to_decimal_l1639_163919


namespace odd_multiple_of_9_is_multiple_of_3_l1639_163941

theorem odd_multiple_of_9_is_multiple_of_3 (n : ℕ) (h1 : n % 2 = 1) (h2 : n % 9 = 0) : n % 3 = 0 := 
by sorry

end odd_multiple_of_9_is_multiple_of_3_l1639_163941


namespace remainder_23_to_2047_mod_17_l1639_163957

theorem remainder_23_to_2047_mod_17 :
  23^2047 % 17 = 11 := 
by {
  sorry
}

end remainder_23_to_2047_mod_17_l1639_163957


namespace domain_g_l1639_163960

noncomputable def f : ℝ → ℝ := sorry  -- f is a real-valued function

theorem domain_g:
  (∀ x, x ∈ [-2, 4] ↔ f x ∈ [-2, 4]) →  -- The domain of f(x) is [-2, 4]
  (∀ x, x ∈ [-2, 2] ↔ (f x + f (-x)) ∈ [-2, 2]) :=  -- The domain of g(x) = f(x) + f(-x) is [-2, 2]
by
  intros h
  sorry

end domain_g_l1639_163960


namespace average_pages_correct_l1639_163956

noncomputable def total_pages : ℝ := 50 + 75 + 80 + 120 + 100 + 90 + 110 + 130
def num_books : ℝ := 8
noncomputable def average_pages : ℝ := total_pages / num_books

theorem average_pages_correct : average_pages = 94.375 :=
by
  sorry

end average_pages_correct_l1639_163956


namespace subtraction_result_l1639_163958

theorem subtraction_result :
  let x := 567.89
  let y := 123.45
  (x - y) = 444.44 :=
by
  sorry

end subtraction_result_l1639_163958


namespace inscribed_pentagon_angles_sum_l1639_163900

theorem inscribed_pentagon_angles_sum (α β γ δ ε : ℝ) (h1 : α + β + γ + δ + ε = 360) 
(h2 : α / 2 + β / 2 + γ / 2 + δ / 2 + ε / 2 = 180) : 
(α / 2) + (β / 2) + (γ / 2) + (δ / 2) + (ε / 2) = 180 :=
by
  sorry

end inscribed_pentagon_angles_sum_l1639_163900


namespace geometric_sequence_term_formula_l1639_163972

theorem geometric_sequence_term_formula (a n : ℕ) (a_seq : ℕ → ℕ)
  (h1 : a_seq 0 = a - 1) (h2 : a_seq 1 = a + 1) (h3 : a_seq 2 = a + 4)
  (geometric_seq : ∀ n, a_seq (n + 1) = a_seq n * ((a_seq 1) / (a_seq 0))) :
  a = 5 ∧ a_seq n = 4 * (3 / 2) ^ (n - 1) :=
by
  sorry

end geometric_sequence_term_formula_l1639_163972


namespace Mary_age_is_10_l1639_163911

-- Define the parameters for the ages of Rahul and Mary
variables (Rahul Mary : ℕ)

-- Conditions provided in the problem
def condition1 := Rahul = Mary + 30
def condition2 := Rahul + 20 = 2 * (Mary + 20)

-- Stating the theorem to be proved
theorem Mary_age_is_10 (Rahul Mary : ℕ) 
  (h1 : Rahul = Mary + 30) 
  (h2 : Rahul + 20 = 2 * (Mary + 20)) : 
  Mary = 10 :=
by 
  sorry

end Mary_age_is_10_l1639_163911


namespace kyro_percentage_paid_l1639_163943

theorem kyro_percentage_paid
    (aryan_debt : ℕ) -- Aryan owes Fernanda $1200
    (kyro_debt : ℕ) -- Kyro owes Fernanda
    (aryan_debt_twice_kyro_debt : aryan_debt = 2 * kyro_debt) -- Aryan's debt is twice what Kyro owes
    (aryan_payment : ℕ) -- Aryan's payment
    (aryan_payment_percentage : aryan_payment = 60 * aryan_debt / 100) -- Aryan pays 60% of her debt
    (initial_savings : ℕ) -- Initial savings in Fernanda's account
    (final_savings : ℕ) -- Final savings in Fernanda's account
    (initial_savings_cond : initial_savings = 300) -- Fernanda's initial savings is $300
    (final_savings_cond : final_savings = 1500) -- Fernanda's final savings is $1500
    : kyro_payment = 80 * kyro_debt / 100 := -- Kyro paid 80% of her debt
by {
    sorry
}

end kyro_percentage_paid_l1639_163943


namespace velocity_volleyball_league_members_l1639_163959

theorem velocity_volleyball_league_members (total_cost : ℕ) (socks_cost t_shirt_cost cost_per_member members : ℕ)
  (h_socks_cost : socks_cost = 6)
  (h_t_shirt_cost : t_shirt_cost = socks_cost + 7)
  (h_cost_per_member : cost_per_member = 2 * (socks_cost + t_shirt_cost))
  (h_total_cost : total_cost = 3510)
  (h_total_cost_eq : total_cost = cost_per_member * members) :
  members = 92 :=
by
  sorry

end velocity_volleyball_league_members_l1639_163959


namespace construct_pairwise_tangent_circles_l1639_163913

-- Define the three points A, B, and C in a 2D plane.
variables (A B C : EuclideanSpace ℝ (Fin 2))

/--
  Given three points A, B, and C in the plane, 
  it is possible to construct three circles that are pairwise tangent at these points.
-/
theorem construct_pairwise_tangent_circles (A B C : EuclideanSpace ℝ (Fin 2)) :
  ∃ (O1 O2 O3 : EuclideanSpace ℝ (Fin 2)) (r1 r2 r3 : ℝ),
    r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧
    dist O1 O2 = r1 + r2 ∧
    dist O2 O3 = r2 + r3 ∧
    dist O3 O1 = r3 + r1 ∧
    dist O1 A = r1 ∧ dist O2 B = r2 ∧ dist O3 C = r3 :=
sorry

end construct_pairwise_tangent_circles_l1639_163913


namespace area_of_trapezium_l1639_163931

/-- Two parallel sides of a trapezium are 4 cm and 5 cm respectively. 
    The perpendicular distance between the parallel sides is 6 cm.
    Prove that the area of the trapezium is 27 cm². -/
theorem area_of_trapezium (a b h : ℝ) (ha : a = 4) (hb : b = 5) (hh : h = 6) : 
  (1/2) * (a + b) * h = 27 := 
by 
  sorry

end area_of_trapezium_l1639_163931


namespace sequence_integers_l1639_163963

theorem sequence_integers (a : ℕ → ℤ) (h1 : a 1 = 1) (h2 : a 2 = 1) 
  (h3 : ∀ n ≥ 3, a n = (a (n - 1))^2 + 2 / a (n - 2)) : ∀ n, ∃ k : ℤ, a n = k :=
sorry

end sequence_integers_l1639_163963


namespace Douglas_won_in_county_Y_l1639_163997

def total_percentage (x y t r : ℝ) : Prop :=
  (0.74 * 2 + y * 1 = 0.66 * (2 + 1))

theorem Douglas_won_in_county_Y :
  ∀ (x y t r : ℝ), x = 0.74 → t = 0.66 → r = 2 →
  total_percentage x y t r → y = 0.50 := 
by
  intros x y t r hx ht hr H
  rw [hx, hr, ht] at H
  sorry

end Douglas_won_in_county_Y_l1639_163997


namespace difference_divisible_by_10_l1639_163973

theorem difference_divisible_by_10 : (43 ^ 43 - 17 ^ 17) % 10 = 0 := by
  sorry

end difference_divisible_by_10_l1639_163973


namespace max_b_squared_l1639_163947

theorem max_b_squared (a b : ℤ) (h : (a + b) * (a + b) + a * (a + b) + b = 0) : b^2 ≤ 81 :=
sorry

end max_b_squared_l1639_163947


namespace mod_remainder_l1639_163993

theorem mod_remainder (n : ℤ) (h : n % 5 = 3) : (4 * n - 5) % 5 = 2 := by
  sorry

end mod_remainder_l1639_163993


namespace distance_between_X_and_Y_l1639_163978

theorem distance_between_X_and_Y :
  ∀ (D : ℝ), 
  (10 : ℝ) * (D / (10 : ℝ) + D / (4 : ℝ)) / (10 + 4) = 142.85714285714286 → 
  D = 1000 :=
by
  intro D
  sorry

end distance_between_X_and_Y_l1639_163978


namespace necessary_and_sufficient_conditions_l1639_163967

open Real

def cubic_has_arithmetic_sequence_roots (a b c : ℝ) : Prop :=
∃ x y : ℝ,
  (x - y) * (x) * (x + y) + a * (x^2 + x - y + x + y) + b * x + c = 0 ∧
  3 * x = -a

theorem necessary_and_sufficient_conditions
  (a b c : ℝ) (h : cubic_has_arithmetic_sequence_roots a b c) :
  2 * a^3 - 9 * a * b + 27 * c = 0 ∧ a^2 - 3 * b ≥ 0 :=
sorry

end necessary_and_sufficient_conditions_l1639_163967


namespace f_2007_eq_0_l1639_163920

-- Define even function and odd function properties
def is_even (f : ℝ → ℝ) := ∀ x, f (-x) = f x
def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- Define functions f and g
variables (f g : ℝ → ℝ)

-- Assume the given conditions
axiom even_f : is_even f
axiom odd_g : is_odd g
axiom g_def : ∀ x, g x = f (x - 1)

-- Prove that f(2007) = 0
theorem f_2007_eq_0 : f 2007 = 0 :=
sorry

end f_2007_eq_0_l1639_163920


namespace determine_a_l1639_163932

theorem determine_a :
  ∃ a : ℝ, (∀ x : ℝ, y = -((x - a) / (x - a - 1)) ↔ x = (3 - a) / (3 - a - 1)) → a = 2 :=
sorry

end determine_a_l1639_163932


namespace watch_cost_price_l1639_163981

theorem watch_cost_price 
  (C : ℝ)
  (h1 : 0.9 * C + 180 = 1.05 * C) :
  C = 1200 :=
sorry

end watch_cost_price_l1639_163981


namespace salary_increase_l1639_163945

theorem salary_increase (new_salary increase : ℝ) (h_new : new_salary = 25000) (h_inc : increase = 5000) : 
  ((increase / (new_salary - increase)) * 100) = 25 :=
by
  -- We will write the proof to satisfy the requirement, but it is currently left out as per the instructions.
  sorry

end salary_increase_l1639_163945


namespace ratio_mark_to_jenna_l1639_163969

-- Definitions based on the given conditions
def total_problems : ℕ := 20

def problems_angela : ℕ := 9
def problems_martha : ℕ := 2
def problems_jenna : ℕ := 4 * problems_martha - 2

def problems_completed : ℕ := problems_angela + problems_martha + problems_jenna
def problems_mark : ℕ := total_problems - problems_completed

-- The proof statement based on the question and conditions
theorem ratio_mark_to_jenna :
  (problems_mark : ℚ) / problems_jenna = 1 / 2 :=
by
  sorry

end ratio_mark_to_jenna_l1639_163969


namespace onions_total_l1639_163904

theorem onions_total (Sara_onions : ℕ) (Sally_onions : ℕ) (Fred_onions : ℕ)
  (h1 : Sara_onions = 4) (h2 : Sally_onions = 5) (h3 : Fred_onions = 9) :
  Sara_onions + Sally_onions + Fred_onions = 18 := by
  sorry

end onions_total_l1639_163904


namespace calculate_expression_l1639_163980

theorem calculate_expression (a : ℝ) : 3 * a * (2 * a^2 - 4 * a) - 2 * a^2 * (3 * a + 4) = -20 * a^2 :=
by
  sorry

end calculate_expression_l1639_163980


namespace birdseed_mix_percentage_l1639_163926

theorem birdseed_mix_percentage (x : ℝ) :
  (0.40 * x + 0.65 * (100 - x) = 50) → x = 60 :=
by
  sorry

end birdseed_mix_percentage_l1639_163926


namespace cistern_depth_l1639_163991

theorem cistern_depth (h : ℝ) :
  (6 * 4 + 2 * (h * 6) + 2 * (h * 4) = 49) → (h = 1.25) :=
by
  sorry

end cistern_depth_l1639_163991


namespace length_of_diagonal_l1639_163928

theorem length_of_diagonal (h1 h2 area : ℝ) (h1_val : h1 = 7) (h2_val : h2 = 3) (area_val : area = 50) :
  ∃ d : ℝ, d = 10 :=
by
  sorry

end length_of_diagonal_l1639_163928


namespace f_even_l1639_163996

noncomputable def f : ℝ → ℝ := sorry

axiom f_not_const : ¬ (∀ x y : ℝ, f x = f y)
axiom f_equiv1 : ∀ x : ℝ, f (x + 2) = f (2 - x)
axiom f_equiv2 : ∀ x : ℝ, f (1 + x) = -f x

theorem f_even : ∀ x : ℝ, f x = f (-x) :=
by
  sorry

end f_even_l1639_163996


namespace ratio_sphere_locus_l1639_163962

noncomputable def sphere_locus_ratio (r : ℝ) : ℝ :=
  let F1 := 2 * Real.pi * r^2 * (1 - Real.sqrt (2 / 3))
  let F2 := Real.pi * r^2 * (2 * Real.sqrt 3 / 3)
  F1 / F2

theorem ratio_sphere_locus (r : ℝ) (h : r > 0) : sphere_locus_ratio r = Real.sqrt 3 - 1 :=
by
  sorry

end ratio_sphere_locus_l1639_163962


namespace pattern_proof_l1639_163961

theorem pattern_proof (h1 : 1 = 6) (h2 : 2 = 36) (h3 : 3 = 363) (h4 : 4 = 364) (h5 : 5 = 365) : 36 = 3636 := by
  sorry

end pattern_proof_l1639_163961


namespace oranges_equiv_frac_bananas_l1639_163905

theorem oranges_equiv_frac_bananas :
  (3 / 4) * 16 * (1 / 3) * 9 = (3 / 2) * 6 :=
by
  sorry

end oranges_equiv_frac_bananas_l1639_163905


namespace original_number_is_80_l1639_163903

theorem original_number_is_80 (x : ℝ) (h1 : 1.125 * x - 0.75 * x = 30) : x = 80 :=
by
  sorry

end original_number_is_80_l1639_163903


namespace bottles_from_shop_c_correct_l1639_163946

-- Definitions for the given conditions
def total_bottles := 550
def bottles_from_shop_a := 150
def bottles_from_shop_b := 180

-- Definition for the bottles from Shop C
def bottles_from_shop_c := total_bottles - (bottles_from_shop_a + bottles_from_shop_b)

-- The statement to prove
theorem bottles_from_shop_c_correct : bottles_from_shop_c = 220 :=
by
  -- proof will be filled later
  sorry

end bottles_from_shop_c_correct_l1639_163946


namespace problem_a_problem_b_l1639_163906

-- Part (a)
theorem problem_a (n: Nat) : ∃ k: ℤ, (32^ (3 * n) - 1312^ n) = 1966 * k := sorry

-- Part (b)
theorem problem_b (n: Nat) : ∃ m: ℤ, (843^ (2 * n + 1) - 1099^ (2 * n + 1) + 16^ (4 * n + 2)) = 1967 * m := sorry

end problem_a_problem_b_l1639_163906


namespace alex_hours_per_week_l1639_163983

theorem alex_hours_per_week
  (summer_earnings : ℕ)
  (summer_weeks : ℕ)
  (summer_hours_per_week : ℕ)
  (academic_year_weeks : ℕ)
  (academic_year_earnings : ℕ)
  (same_hourly_rate : Prop) :
  summer_earnings = 4000 →
  summer_weeks = 8 →
  summer_hours_per_week = 40 →
  academic_year_weeks = 32 →
  academic_year_earnings = 8000 →
  same_hourly_rate →
  (academic_year_earnings / ((summer_earnings : ℚ) / (summer_weeks * summer_hours_per_week)) / academic_year_weeks) = 20 :=
by
  sorry

end alex_hours_per_week_l1639_163983


namespace length_of_train_l1639_163924

theorem length_of_train (speed_kmh : ℕ) (time_s : ℕ) (length_bridge_m : ℕ) (length_train_m : ℕ) :
  speed_kmh = 45 → time_s = 30 → length_bridge_m = 275 → length_train_m = 475 :=
by
  intros h1 h2 h3
  sorry

end length_of_train_l1639_163924


namespace intersection_M_N_l1639_163921

def M : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def N : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_M_N :
  M ∩ N = {-1, 0, 1, 2} :=
sorry

end intersection_M_N_l1639_163921


namespace cost_price_of_book_l1639_163955

theorem cost_price_of_book (SP : ℝ) (rate_of_profit : ℝ) (CP : ℝ) 
  (h1 : SP = 90) 
  (h2 : rate_of_profit = 0.8) 
  (h3 : rate_of_profit = (SP - CP) / CP) : 
  CP = 50 :=
sorry

end cost_price_of_book_l1639_163955


namespace bristol_to_carlisle_routes_l1639_163992

-- Given conditions
def r_bb := 6
def r_bs := 3
def r_sc := 2

-- The theorem we want to prove
theorem bristol_to_carlisle_routes :
  (r_bb * r_bs * r_sc) = 36 :=
by
  sorry

end bristol_to_carlisle_routes_l1639_163992


namespace max_students_can_be_equally_distributed_l1639_163902

def num_pens : ℕ := 2730
def num_pencils : ℕ := 1890

theorem max_students_can_be_equally_distributed : Nat.gcd num_pens num_pencils = 210 := by
  sorry

end max_students_can_be_equally_distributed_l1639_163902


namespace num_true_propositions_l1639_163939

theorem num_true_propositions : 
  (∀ (a b : ℝ), a = 0 → ab = 0) ∧
  (∀ (a b : ℝ), ab ≠ 0 → a ≠ 0) ∧
  ¬ (∀ (a b : ℝ), ab = 0 → a = 0) ∧
  ¬ (∀ (a b : ℝ), a ≠ 0 → ab ≠ 0) → 
  2 = 2 :=
by 
  sorry

end num_true_propositions_l1639_163939


namespace javier_average_hits_per_game_l1639_163942

theorem javier_average_hits_per_game (total_games_first_part : ℕ) (average_hits_first_part : ℕ) 
  (remaining_games : ℕ) (average_hits_remaining : ℕ) : 
  total_games_first_part = 20 → average_hits_first_part = 2 → 
  remaining_games = 10 → average_hits_remaining = 5 →
  (total_games_first_part * average_hits_first_part + 
  remaining_games * average_hits_remaining) /
  (total_games_first_part + remaining_games) = 3 := 
by intros h1 h2 h3 h4;
   sorry

end javier_average_hits_per_game_l1639_163942


namespace paul_initial_books_l1639_163929

theorem paul_initial_books (sold_books : ℕ) (left_books : ℕ) (initial_books : ℕ) 
  (h_sold_books : sold_books = 109)
  (h_left_books : left_books = 27)
  (h_initial_books_formula : initial_books = sold_books + left_books) : 
  initial_books = 136 :=
by
  rw [h_sold_books, h_left_books] at h_initial_books_formula
  exact h_initial_books_formula

end paul_initial_books_l1639_163929


namespace unique_solution_m_l1639_163923

theorem unique_solution_m :
  ∃! m : ℝ, ∀ x y : ℝ, (y = x^2 ∧ y = 4*x + m) → m = -4 :=
by 
  sorry

end unique_solution_m_l1639_163923


namespace value_of_f_at_2_l1639_163908

theorem value_of_f_at_2 (a b : ℝ) (h : (a + -b + 8) = (9 * a + 3 * b + 8)) :
  (a * 2 ^ 2 + b * 2 + 8) = 8 := 
by
  sorry

end value_of_f_at_2_l1639_163908


namespace tennis_tournament_total_rounds_l1639_163965

theorem tennis_tournament_total_rounds
  (participants : ℕ)
  (points_win : ℕ)
  (points_loss : ℕ)
  (pairs_formation : ℕ → ℕ)
  (single_points_award : ℕ → ℕ)
  (elimination_condition : ℕ → Prop)
  (tournament_continues : ℕ → Prop)
  (progression_condition : ℕ → ℕ → ℕ)
  (group_split : Π (n : ℕ), Π (k : ℕ), (ℕ × ℕ))
  (rounds_needed : ℕ) :
  participants = 1152 →
  points_win = 1 →
  points_loss = 0 →
  pairs_formation participants ≥ 0 →
  single_points_award participants ≥ 0 →
  (∀ p, p > 1 → participants / p > 0 → tournament_continues participants) →
  (∀ m n, progression_condition m n = n - m) →
  (group_split 1152 1024 = (1024, 128)) →
  rounds_needed = 14 :=
by
  sorry

end tennis_tournament_total_rounds_l1639_163965


namespace triangle_angle_equality_l1639_163987

theorem triangle_angle_equality (A B C : ℝ) (h : ∃ (x : ℝ), x^2 - x * (Real.cos A * Real.cos B) - Real.cos (C / 2)^2 = 0 ∧ x = 1) : A = B :=
by {
  sorry
}

end triangle_angle_equality_l1639_163987


namespace determine_value_of_a_l1639_163989

theorem determine_value_of_a (a : ℝ) (h : 1 < a) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ a → 
  1 ≤ (1 / 2 * x^2 - x + 3 / 2) ∧ (1 / 2 * x^2 - x + 3 / 2) ≤ a) →
  a = 3 :=
by
  sorry

end determine_value_of_a_l1639_163989


namespace part1_part2_l1639_163927

-- Condition for exponents of x to be equal
def condition1 (a : ℤ) : Prop := (3 : ℤ) = 2 * a - 3

-- Condition for exponents of y to be equal
def condition2 (b : ℤ) : Prop := b = 1

noncomputable def a_value : ℤ := 3
noncomputable def b_value : ℤ := 1

-- Theorem for part (1): values of a and b
theorem part1 : condition1 3 ∧ condition2 1 :=
by
  have ha : condition1 3 := by sorry
  have hb : condition2 1 := by sorry
  exact And.intro ha hb

-- Theorem for part (2): value of (7a - 22)^2024 given a = 3
theorem part2 : (7 * a_value - 22) ^ 2024 = 1 :=
by
  have hx : 7 * a_value - 22 = -1 := by sorry
  have hres : (-1) ^ 2024 = 1 := by sorry
  exact Eq.trans (congrArg (fun x => x ^ 2024) hx) hres

end part1_part2_l1639_163927


namespace benny_leftover_money_l1639_163994

-- Define the conditions
def initial_money : ℕ := 67
def spent_money : ℕ := 34

-- Define the leftover money calculation
def leftover_money : ℕ := initial_money - spent_money

-- Prove that Benny had 33 dollars left over
theorem benny_leftover_money : leftover_money = 33 :=
by 
  -- Proof
  sorry

end benny_leftover_money_l1639_163994


namespace not_perfect_square_l1639_163938

theorem not_perfect_square (n : ℤ) (hn : n > 4) : ¬ (∃ k : ℕ, n^2 - 3*n = k^2) :=
sorry

end not_perfect_square_l1639_163938


namespace leo_total_travel_cost_l1639_163950

-- Define the conditions as variables and assumptions in Lean
def cost_one_way : ℕ := 24
def working_days : ℕ := 20

-- Define the total travel cost as a function
def total_travel_cost (cost_one_way : ℕ) (working_days : ℕ) : ℕ :=
  cost_one_way * 2 * working_days

-- State the theorem to prove the total travel cost
theorem leo_total_travel_cost : total_travel_cost 24 20 = 960 :=
sorry

end leo_total_travel_cost_l1639_163950


namespace words_on_each_page_l1639_163974

theorem words_on_each_page (p : ℕ) (h : 150 * p ≡ 198 [MOD 221]) : p = 93 :=
sorry

end words_on_each_page_l1639_163974


namespace right_triangle_acute_angle_30_l1639_163984

theorem right_triangle_acute_angle_30 (α β : ℝ) (h1 : α = 60) (h2 : α + β + 90 = 180) : β = 30 :=
by
  sorry

end right_triangle_acute_angle_30_l1639_163984


namespace no_solution_ineq_system_l1639_163917

def inequality_system (x : ℝ) : Prop :=
  (x / 6 + 7 / 2 > (3 * x + 29) / 5) ∧
  (x + 9 / 2 > x / 8) ∧
  (11 / 3 - x / 6 < (34 - 3 * x) / 5)

theorem no_solution_ineq_system : ¬ ∃ x : ℝ, inequality_system x :=
  sorry

end no_solution_ineq_system_l1639_163917


namespace course_length_l1639_163971

noncomputable def timeBicycling := 12 / 60 -- hours
noncomputable def avgRateBicycling := 30 -- miles per hour
noncomputable def timeRunning := (117 - 12) / 60 -- hours
noncomputable def avgRateRunning := 8 -- miles per hour

theorem course_length : avgRateBicycling * timeBicycling + avgRateRunning * timeRunning = 20 := 
by
  sorry

end course_length_l1639_163971


namespace rectangular_field_area_l1639_163995

noncomputable def length (c : ℚ) : ℚ := 3 * c / 2
noncomputable def width (c : ℚ) : ℚ := 4 * c / 2
noncomputable def area (c : ℚ) : ℚ := (length c) * (width c)
noncomputable def field_area (c1 : ℚ) (c2 : ℚ) : ℚ :=
  let l := length c1
  let w := width c1
  if 25 * c2 = 101.5 * 100 then
    area c1
  else
    0

theorem rectangular_field_area :
  ∃ (c : ℚ), field_area c 25 = 10092 := by
  sorry

end rectangular_field_area_l1639_163995


namespace parallel_line_slope_l1639_163909

theorem parallel_line_slope (x y : ℝ) (h : 3 * x - 6 * y = 9) : ∃ (m : ℝ), m = 1 / 2 := 
sorry

end parallel_line_slope_l1639_163909


namespace distance_to_focus_2_l1639_163933

-- Definition of the ellipse and the given distance to one focus
def ellipse (P : ℝ × ℝ) : Prop := (P.1^2)/25 + (P.2^2)/16 = 1
def distance_to_focus_1 (P : ℝ × ℝ) : Prop := dist P (5, 0) = 3

-- Proof problem statement
theorem distance_to_focus_2 (P : ℝ × ℝ) (h₁ : ellipse P) (h₂ : distance_to_focus_1 P) :
  dist P (-5, 0) = 7 :=
sorry

end distance_to_focus_2_l1639_163933


namespace cole_drive_time_l1639_163986

noncomputable def time_to_drive_to_work (D : ℝ) : ℝ :=
  D / 50

theorem cole_drive_time (D : ℝ) (h₁ : time_to_drive_to_work D + (D / 110) = 2) : time_to_drive_to_work D * 60 = 82.5 :=
by
  sorry

end cole_drive_time_l1639_163986


namespace min_dist_l1639_163944

open Real

theorem min_dist (a b : ℝ) :
  let A := (0, -1)
  let B := (1, 3)
  let C := (2, 6)
  let D := (0, b)
  let E := (1, a + b)
  let F := (2, 2 * a + b)
  let AD_sq := (b + 1) ^ 2
  let BE_sq := (a + b - 3) ^ 2
  let CF_sq := (2 * a + b - 6) ^ 2
  AD_sq + BE_sq + CF_sq = (b + 1) ^ 2 + (a + b - 3) ^ 2 + (2 * a + b - 6) ^ 2 → 
  a = 7 / 2 ∧ b = -5 / 6 :=
sorry

end min_dist_l1639_163944


namespace sufficient_not_necessary_l1639_163949

theorem sufficient_not_necessary (x y : ℝ) :
  (x ≥ 2 ∧ y ≥ 2) ↔ (x + y ≥ 4) :=
by sorry

end sufficient_not_necessary_l1639_163949


namespace minimum_value_g_l1639_163907

noncomputable def g (x : ℝ) : ℝ :=
  x + (2 * x) / (x^2 + 1) + (x * (x + 3)) / (x^2 + 3) + (3 * (x + 1)) / (x * (x^2 + 3))

theorem minimum_value_g : ∀ x : ℝ, x > 0 → g x ≥ 7 :=
by
  intros x hx
  sorry

end minimum_value_g_l1639_163907


namespace range_of_f_l1639_163968

noncomputable def f (x : ℝ) : ℝ :=
  (Real.arccos (x / 2))^2 + 
  Real.pi * Real.arcsin (x / 2) - 
  (Real.arcsin (x / 2))^2 + 
  (Real.pi^2 / 6) * (x^2 + 2 * x + 1)

theorem range_of_f (x : ℝ) (hx : -2 ≤ x ∧ x ≤ 2) :
  ∃ y : ℝ, (f y) = x ∧  (Real.pi^2 / 4) ≤ y ∧ y ≤ (39 * Real.pi^2 / 96) := 
sorry

end range_of_f_l1639_163968


namespace hyperbola_eccentricity_l1639_163977

theorem hyperbola_eccentricity (a : ℝ) (h1 : a > 0) (h2 : 2 = (Real.sqrt (a^2 + 3)) / a) : a = 1 := 
by
  sorry

end hyperbola_eccentricity_l1639_163977


namespace peanuts_remaining_l1639_163918

theorem peanuts_remaining (initial_peanuts brock_ate bonita_ate brock_fraction : ℕ) (h_initial : initial_peanuts = 148) (h_brock_fraction : brock_fraction = 4) (h_brock_ate : brock_ate = initial_peanuts / brock_fraction) (h_bonita_ate : bonita_ate = 29) :
  (initial_peanuts - brock_ate - bonita_ate) = 82 :=
by
  sorry

end peanuts_remaining_l1639_163918


namespace A_and_D_mut_exclusive_not_complementary_l1639_163930

-- Define the events based on the conditions
inductive Die
| one | two | three | four | five | six

def is_odd (d : Die) : Prop :=
  d = Die.one ∨ d = Die.three ∨ d = Die.five

def is_even (d : Die) : Prop :=
  d = Die.two ∨ d = Die.four ∨ d = Die.six

def is_multiple_of_2 (d : Die) : Prop :=
  d = Die.two ∨ d = Die.four ∨ d = Die.six

def is_two_or_four (d : Die) : Prop :=
  d = Die.two ∨ d = Die.four

-- Define the predicate for mutually exclusive but not complementary
def mutually_exclusive_but_not_complementary (P Q : Die → Prop) : Prop :=
  (∀ d, ¬ (P d ∧ Q d)) ∧ ¬ (∀ d, P d ∨ Q d)

-- Verify that "A and D" are mutually exclusive but not complementary
theorem A_and_D_mut_exclusive_not_complementary :
  mutually_exclusive_but_not_complementary is_odd is_two_or_four :=
  by
    sorry

end A_and_D_mut_exclusive_not_complementary_l1639_163930


namespace solve_for_r_l1639_163952

theorem solve_for_r : ∃ r : ℝ, r ≠ 4 ∧ r ≠ 5 ∧ 
  (r^2 - 6*r + 8) / (r^2 - 9*r + 20) = (r^2 - 3*r - 10) / (r^2 - 2*r - 15) ↔ 
  r = 2*Real.sqrt 2 ∨ r = -2*Real.sqrt 2 := 
by {
  sorry
}

end solve_for_r_l1639_163952


namespace num_possibilities_for_asima_integer_l1639_163964

theorem num_possibilities_for_asima_integer (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 65) :
  ∃ (n : ℕ), n = 64 :=
by
  sorry

end num_possibilities_for_asima_integer_l1639_163964


namespace unique_a_for_set_A_l1639_163925

def A (a : ℝ) : Set ℝ := {a^2, 2 - a, 4}

theorem unique_a_for_set_A (a : ℝ) : A a = {x : ℝ // x = a^2 ∨ x = 2 - a ∨ x = 4} → a = -1 :=
by
  sorry

end unique_a_for_set_A_l1639_163925


namespace shirt_cost_is_ten_l1639_163934

theorem shirt_cost_is_ten (S J : ℝ) (h1 : J = 2 * S) 
    (h2 : 20 * S + 10 * J = 400) : S = 10 :=
by
  -- proof skipped
  sorry

end shirt_cost_is_ten_l1639_163934


namespace equation_has_two_distinct_real_roots_l1639_163998

open Real

theorem equation_has_two_distinct_real_roots (m : ℝ) :
  (∃ (x1 x2 : ℝ), 0 < x1 ∧ x1 < 16 ∧ 0 < x2 ∧ x2 < 16 ∧ x1 ≠ x2 ∧ exp (m * x1) = x1^2 ∧ exp (m * x2) = x2^2) ↔
  (log 2 / 2 < m ∧ m < 2 / exp 1) :=
by sorry

end equation_has_two_distinct_real_roots_l1639_163998


namespace evaluate_sqrt_log_expression_l1639_163940

noncomputable def evaluate_log_expression : ℝ :=
  let log3 (x : ℝ) := Real.log x / Real.log 3
  let log4 (x : ℝ) := Real.log x / Real.log 4
  Real.sqrt (log3 8 + log4 8)

theorem evaluate_sqrt_log_expression : evaluate_log_expression = Real.sqrt 3 := 
by
  sorry

end evaluate_sqrt_log_expression_l1639_163940


namespace secret_known_on_monday_l1639_163999

def students_know_secret (n : ℕ) : ℕ :=
  (3^(n + 1) - 1) / 2

theorem secret_known_on_monday :
  ∃ n : ℕ, students_know_secret n = 3280 ∧ (n + 1) % 7 = 0 :=
by
  sorry

end secret_known_on_monday_l1639_163999


namespace sale_in_third_month_l1639_163948

theorem sale_in_third_month (s_1 s_2 s_4 s_5 s_6 : ℝ) (avg_sale : ℝ) (h1 : s_1 = 6435) (h2 : s_2 = 6927) (h4 : s_4 = 7230) (h5 : s_5 = 6562) (h6 : s_6 = 6191) (h_avg : avg_sale = 6700) :
  ∃ s_3 : ℝ, s_1 + s_2 + s_3 + s_4 + s_5 + s_6 = 6 * avg_sale ∧ s_3 = 6855 :=
by 
  sorry

end sale_in_third_month_l1639_163948


namespace B_work_days_proof_l1639_163954

-- Define the main variables
variables (W : ℝ) (x : ℝ) (daysA : ℝ) (daysBworked : ℝ) (daysAremaining : ℝ)

-- Given conditions from the problem
def A_work_days : ℝ := 6
def B_work_days : ℝ := x
def B_worked_days : ℝ := 10
def A_remaining_days : ℝ := 2

-- We are asked to prove this statement
theorem B_work_days_proof (h1 : daysA = A_work_days)
                           (h2 : daysBworked = B_worked_days)
                           (h3 : daysAremaining = A_remaining_days) 
                           (hx : (W/6 = (W - 10*W/x) / 2)) : x = 15 :=
by 
  -- Proof omitted
  sorry 

end B_work_days_proof_l1639_163954


namespace sales_tax_difference_l1639_163901

def item_price : ℝ := 20
def sales_tax_rate1 : ℝ := 0.065
def sales_tax_rate2 : ℝ := 0.06

theorem sales_tax_difference :
  (item_price * sales_tax_rate1) - (item_price * sales_tax_rate2) = 0.1 := 
by
  sorry

end sales_tax_difference_l1639_163901


namespace maximize_profit_l1639_163976

noncomputable def profit (x a : ℝ) : ℝ :=
  19 - 24 / (x + 2) - (3 / 2) * x

theorem maximize_profit (a : ℝ) (ha : 0 < a) :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ a) ∧ 
  (if a ≥ 2 then x = 2 else x = a) :=
by
  sorry

end maximize_profit_l1639_163976


namespace outfit_combinations_l1639_163912

theorem outfit_combinations 
  (shirts : Fin 5)
  (pants : Fin 6)
  (restricted_shirt : Fin 1)
  (restricted_pants : Fin 2) :
  ∃ total_combinations : ℕ, total_combinations = 28 :=
sorry

end outfit_combinations_l1639_163912


namespace incorrect_C_l1639_163982

variable (D : ℝ → ℝ)

-- Definitions to encapsulate conditions
def range_D : Set ℝ := {0, 1}
def is_even := ∀ x, D x = D (-x)
def is_periodic := ∀ T > 0, ∃ p, ∀ x, D (x + p) = D x
def is_monotonic := ∀ x y, x < y → D x ≤ D y

-- The proof statement
theorem incorrect_C : ¬ is_periodic D :=
sorry

end incorrect_C_l1639_163982


namespace combined_population_l1639_163914

theorem combined_population (W PP LH : ℕ) 
  (hW : W = 900)
  (hPP : PP = 7 * W)
  (hLH : PP = LH + 800) : 
  (PP + LH) = 11800 :=
by
  sorry

end combined_population_l1639_163914


namespace digits_base8_sum_l1639_163937

open Nat

theorem digits_base8_sum (X Y Z : ℕ) (hX : 0 < X) (hY : 0 < Y) (hZ : 0 < Z) 
  (h_distinct : X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z) (h_base8 : X < 8 ∧ Y < 8 ∧ Z < 8) 
  (h_eq : (8^2 * X + 8 * Y + Z) + (8^2 * Y + 8 * Z + X) + (8^2 * Z + 8 * X + Y) = 8^3 * X + 8^2 * X + 8 * X) : 
  Y + Z = 7 :=
by
  sorry

end digits_base8_sum_l1639_163937
