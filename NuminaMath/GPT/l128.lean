import Mathlib

namespace cos_135_eq_neg_sqrt_two_div_two_l128_128975

theorem cos_135_eq_neg_sqrt_two_div_two : real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt_two_div_two_l128_128975


namespace simplify_fraction_l128_128654

variable {a b c : ℝ}

theorem simplify_fraction (h : a + b + c ≠ 0) :
  (a^2 + 3*a*b + b^2 - c^2) / (a^2 + 3*a*c + c^2 - b^2) = (a + b - c) / (a - b + c) := 
by
  sorry

end simplify_fraction_l128_128654


namespace smallest_n_l128_128190

theorem smallest_n (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h_sum : a + b + c = 2012) :
  ∃ m n : ℕ, a.factorial * b.factorial * c.factorial = m * 10 ^ n ∧ ¬ (10 ∣ m) ∧ n = 501 :=
by
  sorry

end smallest_n_l128_128190


namespace genevieve_initial_amount_l128_128544

def cost_per_kg : ℕ := 8
def kg_bought : ℕ := 250
def short_amount : ℕ := 400
def total_cost : ℕ := kg_bought * cost_per_kg
def initial_amount := total_cost - short_amount

theorem genevieve_initial_amount : initial_amount = 1600 := by
  unfold initial_amount total_cost cost_per_kg kg_bought short_amount
  sorry

end genevieve_initial_amount_l128_128544


namespace fraction_meaningful_range_l128_128056

theorem fraction_meaningful_range (x : ℝ) : 5 - x ≠ 0 ↔ x ≠ 5 :=
by sorry

end fraction_meaningful_range_l128_128056


namespace floor_plus_x_eq_17_over_4_l128_128402

theorem floor_plus_x_eq_17_over_4
  (x : ℚ)
  (h : ⌊x⌋ + x = 17 / 4)
  : x = 9 / 4 :=
sorry

end floor_plus_x_eq_17_over_4_l128_128402


namespace last_non_zero_digit_of_30_factorial_l128_128406

theorem last_non_zero_digit_of_30_factorial : 
  ∃ d: ℕ, d < 10 ∧ d ≠ 0 ∧ (30.factorial % 10^8 % 10 = d) := 
begin
  use 8,
  split,
  { exact nat.lt.base (10-1), },
  split,
  { norm_num, },
  { sorry }  -- Proof omitted
end

end last_non_zero_digit_of_30_factorial_l128_128406


namespace proof_remove_terms_sum_is_one_l128_128084

noncomputable def remove_terms_sum_is_one : Prop :=
  let initial_sum := (1/2) + (1/4) + (1/6) + (1/8) + (1/10) + (1/12)
  let terms_to_remove := (1/8) + (1/10)
  initial_sum - terms_to_remove = 1

theorem proof_remove_terms_sum_is_one : remove_terms_sum_is_one :=
by
  -- proof will go here but is not required
  sorry

end proof_remove_terms_sum_is_one_l128_128084


namespace prove_problem_l128_128863

noncomputable def proof_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) : Prop :=
  (1 + 1 / x) * (1 + 1 / y) ≥ 9

theorem prove_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) : proof_problem x y hx hy h :=
  sorry

end prove_problem_l128_128863


namespace cos_135_eq_neg_sqrt2_div_2_l128_128946

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
by 
  sorry

end cos_135_eq_neg_sqrt2_div_2_l128_128946


namespace sin_30_eq_half_l128_128344

theorem sin_30_eq_half
  (Q : ℝ × ℝ)
  (h1 : Q = (cos (π / 6), sin (π / 6)))
  (E : ℝ × ℝ)
  (h2 : E = (Q.1, 0))
  (h3 : ∃ x y z : ℝ, x / 2 = Q.2 ∧ y = Q.1 ∧ sqrt (x^2 + y^2) = z ∧ z = 1) :
  sin (π / 6) = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l128_128344


namespace account_balance_after_transfer_l128_128118

def account_after_transfer (initial_balance transfer_amount : ℕ) : ℕ :=
  initial_balance - transfer_amount

theorem account_balance_after_transfer :
  account_after_transfer 27004 69 = 26935 :=
by
  sorry

end account_balance_after_transfer_l128_128118


namespace reducibility_implies_divisibility_l128_128173

theorem reducibility_implies_divisibility
  (a b c d l k : ℤ)
  (p q : ℤ)
  (h1 : a * l + b = k * p)
  (h2 : c * l + d = k * q) :
  k ∣ (a * d - b * c) :=
sorry

end reducibility_implies_divisibility_l128_128173


namespace solution_set_of_inequality_l128_128894

theorem solution_set_of_inequality : {x : ℝ | x^2 + x - 2 < 0} = {x : ℝ | -2 < x ∧ x < 1} :=
by
  sorry

end solution_set_of_inequality_l128_128894


namespace circumradius_geq_3_times_inradius_l128_128459

-- Define the variables representing the circumradius and inradius
variables {R r : ℝ}

-- Assume the conditions that R is the circumradius and r is the inradius of a tetrahedron
def tetrahedron_circumradius (R : ℝ) : Prop := true
def tetrahedron_inradius (r : ℝ) : Prop := true

-- State the theorem
theorem circumradius_geq_3_times_inradius (hR : tetrahedron_circumradius R) (hr : tetrahedron_inradius r) : R ≥ 3 * r :=
sorry

end circumradius_geq_3_times_inradius_l128_128459


namespace max_value_of_sum_max_value_achievable_l128_128185

theorem max_value_of_sum (a b c d : ℝ) 
  (h : a^6 + b^6 + c^6 + d^6 = 64) : a^7 + b^7 + c^7 + d^7 ≤ 128 :=
sorry

theorem max_value_achievable : ∃ a b c d : ℝ,
  a^6 + b^6 + c^6 + d^6 = 64 ∧ a^7 + b^7 + c^7 + d^7 = 128 :=
sorry

end max_value_of_sum_max_value_achievable_l128_128185


namespace find_f_of_2_l128_128827

theorem find_f_of_2 (f : ℝ → ℝ) (h : ∀ (x : ℝ), x > 0 → f (Real.log x / Real.log 2) = 2 ^ x) : f 2 = 16 :=
by
  sorry

end find_f_of_2_l128_128827


namespace evaluate_f_at_2_l128_128823

def f (x : ℕ) : ℕ := 5 * x + 2

theorem evaluate_f_at_2 : f 2 = 12 := by
  sorry

end evaluate_f_at_2_l128_128823


namespace profit_percentage_l128_128781

-- Definitions and conditions
variable (SP : ℝ) (CP : ℝ)
variable (h : CP = 0.98 * SP)

-- Lean statement to prove the profit percentage is 2.04%
theorem profit_percentage (h : CP = 0.98 * SP) : (SP - CP) / CP * 100 = 2.04 := 
sorry

end profit_percentage_l128_128781


namespace borgnine_lizards_l128_128809

theorem borgnine_lizards (chimps lions tarantulas total_legs : ℕ) (legs_per_chimp legs_per_lion legs_per_tarantula legs_per_lizard lizards : ℕ)
  (H_chimps : chimps = 12)
  (H_lions : lions = 8)
  (H_tarantulas : tarantulas = 125)
  (H_total_legs : total_legs = 1100)
  (H_legs_per_chimp : legs_per_chimp = 4)
  (H_legs_per_lion : legs_per_lion = 4)
  (H_legs_per_tarantula : legs_per_tarantula = 8)
  (H_legs_per_lizard : legs_per_lizard = 4)
  (H_seen_legs : total_legs = (chimps * legs_per_chimp) + (lions * legs_per_lion) + (tarantulas * legs_per_tarantula) + (lizards * legs_per_lizard)) :
  lizards = 5 := 
by
  sorry

end borgnine_lizards_l128_128809


namespace geometric_sequence_general_term_and_arithmetic_sequence_max_sum_l128_128825

theorem geometric_sequence_general_term_and_arithmetic_sequence_max_sum :
  (∃ a_n : ℕ → ℕ, ∃ b_n : ℕ → ℤ, ∃ T_n : ℕ → ℤ,
    (∀ n, a_n n = 2^(n-1)) ∧
    (a_n 1 + a_n 2 = 3) ∧
    (b_n 2 = a_n 3) ∧
    (b_n 3 = -b_n 5) ∧
    (∀ n, T_n n = n * (b_n 1 + b_n n) / 2) ∧
    (T_n 3 = 12) ∧
    (T_n 4 = 12)) :=
by
  sorry

end geometric_sequence_general_term_and_arithmetic_sequence_max_sum_l128_128825


namespace sin_thirty_degrees_l128_128330

theorem sin_thirty_degrees : Real.sin (Real.pi / 6) = 1 / 2 := 
by sorry

end sin_thirty_degrees_l128_128330


namespace find_frac_a_b_c_l128_128645

theorem find_frac_a_b_c (a b c : ℝ) (h1 : a = 2 * b) (h2 : a^2 + b^2 = c^2) : (a + b) / c = (3 * Real.sqrt 5) / 5 :=
by
  sorry

end find_frac_a_b_c_l128_128645


namespace number_of_players_l128_128803

variable (total_socks : ℕ) (socks_per_player : ℕ)

theorem number_of_players (h1 : total_socks = 16) (h2 : socks_per_player = 2) :
  total_socks / socks_per_player = 8 := by
  -- proof steps will go here
  sorry

end number_of_players_l128_128803


namespace smallest_solution_l128_128673

theorem smallest_solution (x : ℝ) (h : (1 / (x - 3)) + (1 / (x - 5)) = (4 / (x - 4))) : 
  x = 5 - 2 * Real.sqrt 2 :=
sorry

end smallest_solution_l128_128673


namespace probability_of_2_reds_before_3_greens_l128_128446

theorem probability_of_2_reds_before_3_greens :
  let total_chips := 7
  let red_chips := 4
  let green_chips := 3
  let total_arrangements := Nat.choose total_chips green_chips
  let favorable_arrangements := Nat.choose 5 2
  (favorable_arrangements / total_arrangements : ℚ) = (2 / 7 : ℚ) :=
by
  let total_chips := 7
  let red_chips := 4
  let green_chips := 3
  let total_arrangements := Nat.choose total_chips green_chips
  let favorable_arrangements := Nat.choose 5 2
  have fraction_computation :
    (favorable_arrangements : ℚ) / (total_arrangements : ℚ) = (2 / 7 : ℚ)
  {
    sorry
  }
  exact fraction_computation

end probability_of_2_reds_before_3_greens_l128_128446


namespace sin_30_eq_half_l128_128312

noncomputable def P : ℝ × ℝ := 
  have angle := real.pi / 6
  have x := real.cos angle
  have y := real.sin angle
  (x, y)  -- The point on the unit circle at 30 degrees

theorem sin_30_eq_half : real.sin (real.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l128_128312


namespace number_of_four_digit_integers_with_digit_sum_nine_l128_128539

theorem number_of_four_digit_integers_with_digit_sum_nine :
  ∃ (n : ℕ), (n = 165) ∧ (
    ∃ (a b c d : ℕ), 
      1 ≤ a ∧ 
      a + b + c + d = 9 ∧ 
      (1 ≤ a ∧ a ≤ 9) ∧ 
      (0 ≤ b ∧ b ≤ 9) ∧ 
      (0 ≤ c ∧ c ≤ 9) ∧ 
      (0 ≤ d ∧ d ≤ 9)) := 
sorry

end number_of_four_digit_integers_with_digit_sum_nine_l128_128539


namespace arcade_playtime_l128_128163

noncomputable def cost_per_six_minutes : ℝ := 0.50
noncomputable def total_spent : ℝ := 15
noncomputable def minutes_per_interval : ℝ := 6
noncomputable def minutes_per_hour : ℝ := 60

theorem arcade_playtime :
  (total_spent / cost_per_six_minutes) * minutes_per_interval / minutes_per_hour = 3 :=
by
  sorry

end arcade_playtime_l128_128163


namespace probability_both_in_photo_correct_l128_128159

noncomputable def probability_both_in_photo (lap_time_Emily : ℕ) (lap_time_John : ℕ) (observation_start : ℕ) (observation_end : ℕ) : ℚ := 
  let GCD := Nat.gcd lap_time_Emily lap_time_John
  let cycle_time := lap_time_Emily * lap_time_John / GCD
  let visible_time := 2 * min (lap_time_Emily / 3) (lap_time_John / 3)
  visible_time / cycle_time

theorem probability_both_in_photo_correct : 
  probability_both_in_photo 100 75 900 1200 = 1 / 6 :=
by
  -- Use previous calculations and observations here to construct the proof.
  -- sorry is used to indicate that proof steps are omitted.
  sorry

end probability_both_in_photo_correct_l128_128159


namespace sin_thirty_degrees_l128_128334

theorem sin_thirty_degrees : Real.sin (Real.pi / 6) = 1 / 2 := 
by sorry

end sin_thirty_degrees_l128_128334


namespace range_of_c_l128_128413

variable (c : ℝ)

def p := 2 < 3 * c
def q := ∀ x : ℝ, 2 * x^2 + 4 * c * x + 1 > 0

theorem range_of_c (hp : p c) (hq : q c) : (2 / 3) < c ∧ c < (Real.sqrt 2 / 2) :=
by
  sorry

end range_of_c_l128_128413


namespace sin_30_eq_half_l128_128300

theorem sin_30_eq_half : Real.sin (π / 6) = 1 / 2 := 
  sorry

end sin_30_eq_half_l128_128300


namespace distance_from_B_l128_128511

theorem distance_from_B (s y : ℝ) 
  (h1 : s^2 = 12)
  (h2 : ∀y, (1 / 2) * y^2 = 12 - y^2)
  (h3 : y = 2 * Real.sqrt 2)
: Real.sqrt ((2 * Real.sqrt 2)^2 + (2 * Real.sqrt 2)^2) = 4 := by
  sorry

end distance_from_B_l128_128511


namespace f_perf_square_iff_n_eq_one_l128_128695

def f (n : ℕ+) : ℕ :=
  (Finset.filter (fun s : Finset (Fin n) => nat.gcd (s.val.filter (λ x, x ∈ s)) = 1)
  (Finset.powerset (Finset.range (n : ℕ)))).card

theorem f_perf_square_iff_n_eq_one (n : ℕ+) : (∃ k : ℕ, k * k = f n) ↔ n = 1 := sorry

end f_perf_square_iff_n_eq_one_l128_128695


namespace problem1_problem2_l128_128003

-- Definitions of the conditions
def periodic_func (f: ℝ → ℝ) (a: ℝ) (x: ℝ) : Prop :=
(∀ x, f (x + 3) = f x) ∧ 
(∀ x, -2 ≤ x ∧ x < 0 → f x = x + a) ∧ 
(∀ x, 0 ≤ x ∧ x < 1 → f x = (1/2)^x)

-- 1. Prove f(13/2) = sqrt(2)/2
theorem problem1 (f: ℝ → ℝ) (a: ℝ) (h: periodic_func f a x) : f (13/2) = (Real.sqrt 2) / 2 := 
sorry

-- 2. Prove that if f(x) has a minimum value but no maximum value, then 1 < a ≤ 5/2
theorem problem2 (f: ℝ → ℝ) (a: ℝ) (h: periodic_func f a x) (hmin: ∃ m, ∀ x, f x ≥ m) (hmax: ¬∃ M, ∀ x, f x ≤ M) : 1 < a ∧ a ≤ 5/2 :=
sorry

end problem1_problem2_l128_128003


namespace cos_135_eq_neg_sqrt2_div_2_l128_128992

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = - Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l128_128992


namespace cos_135_degree_l128_128982

theorem cos_135_degree : 
  let Q := (Complex.exp (135 * Real.pi / 180 * Complex.I)) in 
  Complex.re Q = -Real.sqrt 2 / 2 :=
by sorry

end cos_135_degree_l128_128982


namespace average_salary_increase_l128_128192

theorem average_salary_increase :
  let avg_salary := 1200
  let num_employees := 20
  let manager_salary := 3300
  let new_num_people := num_employees + 1
  let total_salary := num_employees * avg_salary
  let new_total_salary := total_salary + manager_salary
  let new_avg_salary := new_total_salary / new_num_people
  let increase := new_avg_salary - avg_salary
  increase = 100 :=
by
  sorry

end average_salary_increase_l128_128192


namespace cos_135_eq_neg_inv_sqrt2_l128_128957

theorem cos_135_eq_neg_inv_sqrt2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt2_l128_128957


namespace solve_equation_1_solve_equation_2_l128_128744

theorem solve_equation_1 (x : ℝ) : x * (x + 2) = 2 * (x + 2) ↔ x = -2 ∨ x = 2 := 
by sorry

theorem solve_equation_2 (x : ℝ) : 3 * x^2 - x - 1 = 0 ↔ x = (1 + Real.sqrt 13) / 6 ∨ x = (1 - Real.sqrt 13) / 6 := 
by sorry

end solve_equation_1_solve_equation_2_l128_128744


namespace men_women_equal_after_city_Y_l128_128638

variable (M W M' W' : ℕ)

-- Initial conditions: total passengers, women to men ratio
variable (h1 : M + W = 72)
variable (h2 : W = M / 2)

-- Changes in city Y: men leave, women enter
variable (h3 : M' = M - 16)
variable (h4 : W' = W + 8)

theorem men_women_equal_after_city_Y (h1 : M + W = 72) (h2 : W = M / 2) (h3 : M' = M - 16) (h4 : W' = W + 8) : 
  M' = W' := 
by 
  sorry

end men_women_equal_after_city_Y_l128_128638


namespace find_c_l128_128476

-- Define the two points as given in the problem
def pointA : ℝ × ℝ := (-6, 1)
def pointB : ℝ × ℝ := (-3, 4)

-- Define the direction vector as subtraction of the two points
def directionVector : ℝ × ℝ := (pointB.1 - pointA.1, pointB.2 - pointA.2)

-- Define the target direction vector format with unknown c
def targetDirectionVector (c : ℝ) : ℝ × ℝ := (3, c)

-- The theorem stating that c must be 3
theorem find_c : ∃ c : ℝ, directionVector = targetDirectionVector c ∧ c = 3 := 
by
  -- Prove the statement or show it is derivable
  sorry

end find_c_l128_128476


namespace sin_30_eq_half_l128_128315

noncomputable def P : ℝ × ℝ := 
  have angle := real.pi / 6
  have x := real.cos angle
  have y := real.sin angle
  (x, y)  -- The point on the unit circle at 30 degrees

theorem sin_30_eq_half : real.sin (real.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l128_128315


namespace mark_total_cents_l128_128466

theorem mark_total_cents (dimes nickels : ℕ) (h1 : nickels = dimes + 3) (h2 : dimes = 5) : 
  dimes * 10 + nickels * 5 = 90 := by
  sorry

end mark_total_cents_l128_128466


namespace sin_thirty_degrees_l128_128335

theorem sin_thirty_degrees : Real.sin (Real.pi / 6) = 1 / 2 := 
by sorry

end sin_thirty_degrees_l128_128335


namespace odd_coefficients_in_binomial_expansion_l128_128549

theorem odd_coefficients_in_binomial_expansion :
  let a : Fin 9 → ℕ := fun k => Nat.choose 8 k
  (Finset.filter (fun k => a k % 2 = 1) (Finset.Icc 0 8)).card = 2 := by
  sorry

end odd_coefficients_in_binomial_expansion_l128_128549


namespace find_linear_function_l128_128707

theorem find_linear_function (a : ℝ) (a_pos : 0 < a) :
  ∃ (b : ℝ), ∀ (f : ℕ → ℝ),
  (∀ (k m : ℕ), (a * m ≤ k ∧ k < (a + 1) * m) → f (k + m) = f k + f m) →
  ∀ n : ℕ, f n = b * n :=
sorry

end find_linear_function_l128_128707


namespace new_concentration_of_solution_l128_128640

theorem new_concentration_of_solution 
  (Q : ℚ) 
  (initial_concentration : ℚ := 0.4) 
  (new_concentration : ℚ := 0.25) 
  (replacement_fraction : ℚ := 1/3) 
  (new_solution_concentration : ℚ := 0.35) :
  (initial_concentration * (1 - replacement_fraction) + new_concentration * replacement_fraction)
  = new_solution_concentration := 
by 
  sorry

end new_concentration_of_solution_l128_128640


namespace calories_consumed_in_week_l128_128377

-- Define the calorie content of each type of burger
def calorie_A := 350
def calorie_B := 450
def calorie_C := 550

-- Define Dimitri's burger consumption over the 7 days
def consumption_day1 := (2 * calorie_A) + (1 * calorie_B)
def consumption_day2 := (1 * calorie_A) + (2 * calorie_B) + (1 * calorie_C)
def consumption_day3 := (1 * calorie_A) + (1 * calorie_B) + (2 * calorie_C)
def consumption_day4 := (3 * calorie_B)
def consumption_day5 := (1 * calorie_A) + (1 * calorie_B) + (1 * calorie_C)
def consumption_day6 := (2 * calorie_A) + (3 * calorie_C)
def consumption_day7 := (1 * calorie_B) + (2 * calorie_C)

-- Define the total weekly calorie consumption
def total_weekly_calories :=
  consumption_day1 + consumption_day2 + consumption_day3 +
  consumption_day4 + consumption_day5 + consumption_day6 + consumption_day7

-- State and prove the main theorem
theorem calories_consumed_in_week :
  total_weekly_calories = 11450 := 
by
  sorry

end calories_consumed_in_week_l128_128377


namespace carl_marbles_l128_128649

-- Define initial conditions
def initial_marbles : ℕ := 12
def lost_marbles : ℕ := initial_marbles / 2
def remaining_marbles : ℕ := initial_marbles - lost_marbles
def additional_marbles : ℕ := 10
def new_marbles_from_mother : ℕ := 25

-- Define the final number of marbles Carl will put back in the jar
def total_marbles_put_back : ℕ := remaining_marbles + additional_marbles + new_marbles_from_mother

-- Statement to be proven
theorem carl_marbles : total_marbles_put_back = 41 :=
by
  sorry

end carl_marbles_l128_128649


namespace number_of_customers_per_month_l128_128904

-- Define the constants and conditions
def price_lettuce_per_head : ℝ := 1
def price_tomato_per_piece : ℝ := 0.5
def num_lettuce_per_customer : ℕ := 2
def num_tomato_per_customer : ℕ := 4
def monthly_sales : ℝ := 2000

-- Calculate the cost per customer
def cost_per_customer : ℝ := 
  (num_lettuce_per_customer * price_lettuce_per_head) + 
  (num_tomato_per_customer * price_tomato_per_piece)

-- Prove the number of customers per month
theorem number_of_customers_per_month : monthly_sales / cost_per_customer = 500 :=
  by
    -- Here, we would write the proof steps
    sorry

end number_of_customers_per_month_l128_128904


namespace satisfactory_grades_fraction_l128_128232

def total_satisfactory_students (gA gB gC gD gE : Nat) : Nat :=
  gA + gB + gC + gD + gE

def total_students (gA gB gC gD gE gF : Nat) : Nat :=
  total_satisfactory_students gA gB gC gD gE + gF

def satisfactory_fraction (gA gB gC gD gE gF : Nat) : Rat :=
  total_satisfactory_students gA gB gC gD gE / total_students gA gB gC gD gE gF

theorem satisfactory_grades_fraction :
  satisfactory_fraction 3 5 4 2 1 4 = (15 : Rat) / 19 :=
by
  sorry

end satisfactory_grades_fraction_l128_128232


namespace cos_135_degree_l128_128983

theorem cos_135_degree : 
  let Q := (Complex.exp (135 * Real.pi / 180 * Complex.I)) in 
  Complex.re Q = -Real.sqrt 2 / 2 :=
by sorry

end cos_135_degree_l128_128983


namespace Ruth_math_class_percentage_l128_128017

theorem Ruth_math_class_percentage :
  let hours_school_day := 8
  let days_school_week := 5
  let hours_math_week := 10
  let total_school_hours_week := hours_school_day * days_school_week
  (hours_math_week / total_school_hours_week) * 100 = 25 := 
by 
  let hours_school_day := 8
  let days_school_week := 5
  let hours_math_week := 10
  let total_school_hours_week := hours_school_day * days_school_week
  -- skip the proof here
  sorry

end Ruth_math_class_percentage_l128_128017


namespace red_more_than_yellow_l128_128209

-- Define the total number of marbles
def total_marbles : ℕ := 19

-- Define the number of yellow marbles
def yellow_marbles : ℕ := 5

-- Calculate the number of remaining marbles
def remaining_marbles : ℕ := total_marbles - yellow_marbles

-- Define the ratio of blue to red marbles
def blue_ratio : ℕ := 3
def red_ratio : ℕ := 4

-- Calculate the sum of ratio parts
def sum_ratio : ℕ := blue_ratio + red_ratio

-- Calculate the number of shares per ratio part
def share_per_part : ℕ := remaining_marbles / sum_ratio

-- Calculate the number of red marbles
def red_marbles : ℕ := red_ratio * share_per_part

-- Theorem to prove: the difference between red marbles and yellow marbles is 3
theorem red_more_than_yellow : red_marbles - yellow_marbles = 3 :=
by
  sorry

end red_more_than_yellow_l128_128209


namespace cos_135_eq_neg_sqrt_two_div_two_l128_128976

theorem cos_135_eq_neg_sqrt_two_div_two : real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt_two_div_two_l128_128976


namespace geometric_sequence_4th_term_is_2_5_l128_128036

variables (a r : ℝ) (n : ℕ)

def geometric_term (a r : ℝ) (n : ℕ) : ℝ := a * r^(n-1)

theorem geometric_sequence_4th_term_is_2_5 (a r : ℝ)
  (h1 : a = 125) 
  (h2 : geometric_term a r 8 = 72) :
  geometric_term a r 4 = 5 / 2 := 
sorry

end geometric_sequence_4th_term_is_2_5_l128_128036


namespace fraction_invariant_l128_128566

variable {R : Type*} [Field R]
variables (x y : R)

theorem fraction_invariant : (2 * x) / (3 * x - y) = (6 * x) / (9 * x - 3 * y) :=
by
  sorry

end fraction_invariant_l128_128566


namespace collinear_condition_l128_128534

variable {R : Type*} [LinearOrderedField R]
variable {x1 y1 x2 y2 x3 y3 : R}

theorem collinear_condition : 
  x1 * y2 + x2 * y3 + x3 * y1 = y1 * x2 + y2 * x3 + y3 * x1 →
  ∃ k l m : R, k * (x2 - x1) = l * (y2 - y1) ∧ k * (x3 - x1) = m * (y3 - y1) :=
by
  sorry

end collinear_condition_l128_128534


namespace smallest_solution_l128_128680

theorem smallest_solution :
  (∃ x : ℝ, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧
  ∀ y : ℝ, (1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4)) → x ≤ y) →
  x = 4 - Real.sqrt 2 :=
sorry

end smallest_solution_l128_128680


namespace hyperbola_equation_Q_on_fixed_circle_l128_128142

-- Define the hyperbola and necessary conditions
def hyperbola (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / (3 * a^2) = 1

-- Given conditions
variables (a : ℝ) (h_pos : a > 0)
variables (F1 F2 : ℝ × ℝ)
variables (dist_F2_asymptote : ℝ) (h_dist : dist_F2_asymptote = sqrt 3)
variables (left_vertex : ℝ × ℝ) (right_branch_intersect : ℝ × ℝ)
variables (line_x_half : ℝ × ℝ)
variables (line_PF2 : ℝ × ℝ)
variables (point_Q : ℝ × ℝ)

-- Prove that the equation of the hyperbola is correct
theorem hyperbola_equation :
  hyperbola a x y ↔ x^2 - y^2 / 3 = 1 :=
sorry

-- Prove that point Q lies on a fixed circle
theorem Q_on_fixed_circle :
  dist point_Q F2 = 4 :=
sorry

end hyperbola_equation_Q_on_fixed_circle_l128_128142


namespace find_fraction_l128_128720

theorem find_fraction :
  ∀ (t k : ℝ) (frac : ℝ),
    t = frac * (k - 32) →
    t = 20 → 
    k = 68 → 
    frac = 5 / 9 :=
by
  intro t k frac h_eq h_t h_k
  -- Start from the conditions and end up showing frac = 5/9
  sorry

end find_fraction_l128_128720


namespace ticket_price_profit_condition_maximize_profit_at_7_point_5_l128_128181

-- Define the ticket price increase and the total profit function
def ticket_price (x : ℝ) := (10 + x) * (500 - 20 * x)

-- Prove that the function equals 6000 at x = 10 and x = 25
theorem ticket_price_profit_condition (x : ℝ) :
  ticket_price x = 6000 ↔ (x = 10 ∨ x = 25) :=
by sorry

-- Prove that m = 7.5 maximizes the profit
def profit (m : ℝ) := -20 * m^2 + 300 * m + 5000

theorem maximize_profit_at_7_point_5 (m : ℝ) :
  m = 7.5 ↔ (∀ m, profit 7.5 ≥ profit m) :=
by sorry

end ticket_price_profit_condition_maximize_profit_at_7_point_5_l128_128181


namespace intersection_sets_l128_128548

-- Define the sets A and B as given in the problem conditions
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {0, 2, 4}

-- Lean theorem statement for proving the intersection of sets A and B is {0, 2}
theorem intersection_sets : A ∩ B = {0, 2} := 
by
  sorry

end intersection_sets_l128_128548


namespace tangent_periodic_solution_l128_128662

theorem tangent_periodic_solution :
  ∃ n : ℤ, -180 < n ∧ n < 180 ∧ (Real.tan (n * Real.pi / 180) = Real.tan (345 * Real.pi / 180)) := by
  sorry

end tangent_periodic_solution_l128_128662


namespace determine_x_l128_128556

theorem determine_x (x : ℝ) (A B : Set ℝ) (H1 : A = {-1, 0}) (H2 : B = {0, 1, x + 2}) (H3 : A ⊆ B) : x = -3 :=
sorry

end determine_x_l128_128556


namespace unoccupied_seats_in_business_class_l128_128238

def airplane_seating (fc bc ec : ℕ) (pfc pbc pec : ℕ) : Nat :=
  let num_ec := ec / 2
  let num_bc := num_ec - pfc
  bc - num_bc

theorem unoccupied_seats_in_business_class :
  airplane_seating 10 30 50 3 (50/2) (50/2) = 8 := by
    sorry

end unoccupied_seats_in_business_class_l128_128238


namespace cos_135_eq_neg_inv_sqrt2_l128_128958

theorem cos_135_eq_neg_inv_sqrt2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt2_l128_128958


namespace fraction_of_earth_surface_inhabitable_l128_128719

theorem fraction_of_earth_surface_inhabitable (f_land : ℚ) (f_inhabitable_land : ℚ)
  (h1 : f_land = 1 / 3)
  (h2 : f_inhabitable_land = 2 / 3) :
  f_land * f_inhabitable_land = 2 / 9 :=
by
  sorry

end fraction_of_earth_surface_inhabitable_l128_128719


namespace michael_number_l128_128871

theorem michael_number (m : ℕ) (h1 : m % 75 = 0) (h2 : m % 40 = 0) (h3 : 1000 < m) (h4 : m < 3000) :
  m = 1800 ∨ m = 2400 ∨ m = 3000 :=
sorry

end michael_number_l128_128871


namespace smallest_solution_l128_128674

theorem smallest_solution (x : ℝ) (h : (1 / (x - 3)) + (1 / (x - 5)) = (4 / (x - 4))) : 
  x = 5 - 2 * Real.sqrt 2 :=
sorry

end smallest_solution_l128_128674


namespace train_length_l128_128107

theorem train_length :
  (∃ (L : ℝ), (L / 30 = (L + 2500) / 120) ∧ L = 75000 / 90) :=
sorry

end train_length_l128_128107


namespace max_value_fraction_l128_128407

theorem max_value_fraction (x y : ℝ) (hx : 1 / 3 ≤ x ∧ x ≤ 3 / 5) (hy : 1 / 4 ≤ y ∧ y ≤ 1 / 2) :
  (∃ x y, (1 / 3 ≤ x ∧ x ≤ 3 / 5) ∧ (1 / 4 ≤ y ∧ y ≤ 1 / 2) ∧ (xy / (x^2 + y^2) = 6 / 13)) :=
by
  sorry

end max_value_fraction_l128_128407


namespace length_of_GH_l128_128032

-- Define the lengths of the segments as given in the conditions
def AB : ℕ := 11
def FE : ℕ := 13
def CD : ℕ := 5

-- Define what we need to prove: the length of GH is 29
theorem length_of_GH (AB FE CD : ℕ) : AB = 11 → FE = 13 → CD = 5 → (AB + CD + FE = 29) :=
by
  sorry

end length_of_GH_l128_128032


namespace sin_30_eq_half_l128_128317

-- Conditions
def point_Q (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

-- The angle 30 degrees in radians
def angle_30_deg := Real.pi / 6

-- Point Q at Angle 30 degrees to the positive x-axis
def Q := point_Q angle_30_deg

-- Formalized problem statement in Lean 4
theorem sin_30_eq_half : Real.sin angle_30_deg = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l128_128317


namespace find_number_of_eggs_l128_128123

namespace HalloweenCleanup

def eggs (E : ℕ) (seconds_per_egg : ℕ) (minutes_per_roll : ℕ) (total_time : ℕ) (num_rolls : ℕ) : Prop :=
  seconds_per_egg = 15 ∧
  minutes_per_roll = 30 ∧
  total_time = 225 ∧
  num_rolls = 7 ∧
  E * (seconds_per_egg / 60) + num_rolls * minutes_per_roll = total_time

theorem find_number_of_eggs : ∃ E : ℕ, eggs E 15 30 225 7 :=
  by
    use 60
    unfold eggs
    simp
    exact sorry

end HalloweenCleanup

end find_number_of_eggs_l128_128123


namespace find_number_of_women_l128_128095

-- Define the work rate variables and the equations from conditions
variables (m w : ℝ) (x : ℝ)

-- Define the first condition
def condition1 : Prop := 3 * m + x * w = 6 * m + 2 * w

-- Define the second condition
def condition2 : Prop := 4 * m + 2 * w = (5 / 7) * (3 * m + x * w)

-- The theorem stating that, given the above conditions, x must be 23
theorem find_number_of_women (hmw : m = 7 * w) (h1 : condition1 m w x) (h2 : condition2 m w x) : x = 23 :=
sorry

end find_number_of_women_l128_128095


namespace sin_thirty_degrees_l128_128332

theorem sin_thirty_degrees : Real.sin (Real.pi / 6) = 1 / 2 := 
by sorry

end sin_thirty_degrees_l128_128332


namespace sin_thirty_deg_l128_128327

theorem sin_thirty_deg : Real.sin (π / 6) = 1 / 2 := 
by 
-- Assume that the unit circle point corresponding to 30° is (√3/2, 1/2)
-- Sine of an angle in the unit circle is the y-coordinate of the corresponding point
sorry

end sin_thirty_deg_l128_128327


namespace cover_large_square_l128_128581

theorem cover_large_square :
  ∃ (small_squares : Fin 8 → Set (ℝ × ℝ)),
    (∀ i, small_squares i = {p : ℝ × ℝ | (p.1 - x_i)^2 + (p.2 - y_i)^2 < (3/2)^2}) ∧
    (∃ (large_square : Set (ℝ × ℝ)),
      large_square = {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 7 ∧ 0 ≤ p.2 ∧ p.2 ≤ 7} ∧
      large_square ⊆ ⋃ i, small_squares i) :=
sorry

end cover_large_square_l128_128581


namespace apple_trees_count_l128_128573

-- Conditions
def num_peach_trees : ℕ := 45
def kg_per_peach_tree : ℕ := 65
def total_mass_fruit : ℕ := 7425
def kg_per_apple_tree : ℕ := 150
variable (A : ℕ)

-- Proof goal
theorem apple_trees_count (h : A * kg_per_apple_tree + num_peach_trees * kg_per_peach_tree = total_mass_fruit) : A = 30 := 
sorry

end apple_trees_count_l128_128573


namespace pi_is_irrational_l128_128489

theorem pi_is_irrational :
  (¬ ∃ (p q : ℤ), q ≠ 0 ∧ π = p / q) :=
by
  sorry

end pi_is_irrational_l128_128489


namespace expected_value_of_sum_of_marbles_l128_128714

theorem expected_value_of_sum_of_marbles :
  let marbles := [1, 2, 3, 4, 5, 6]
  let sets_of_three := Nat.choose 6 3  -- The number of ways to choose 3 marbles out of 6
  let all_sums := [
    1 + 2 + 3, 1 + 2 + 4, 1 + 2 + 5, 1 + 2 + 6, 1 + 3 + 4,
    1 + 3 + 5, 1 + 3 + 6, 1 + 4 + 5, 1 + 4 + 6, 1 + 5 + 6,
    2 + 3 + 4, 2 + 3 + 5, 2 + 3 + 6, 2 + 4 + 5, 2 + 4 + 6,
    2 + 5 + 6, 3 + 4 + 5, 3 + 4 + 6, 3 + 5 + 6, 4 + 5 + 6
  ]
  let total_sum := List.foldr (.+.) 0 all_sums
  let expected_value := total_sum / sets_of_three
  in expected_value = 10.5 := by
  sorry

end expected_value_of_sum_of_marbles_l128_128714


namespace cos_135_eq_neg_sqrt_two_div_two_l128_128970

theorem cos_135_eq_neg_sqrt_two_div_two : real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt_two_div_two_l128_128970


namespace prove_axisymmetric_char4_l128_128915

-- Predicates representing whether a character is an axisymmetric figure
def is_axisymmetric (ch : Char) : Prop := sorry

-- Definitions for the conditions given in the problem
def char1 := '月'
def char2 := '右'
def char3 := '同'
def char4 := '干'

-- Statement that needs to be proven
theorem prove_axisymmetric_char4 (h1 : ¬ is_axisymmetric char1) 
                                  (h2 : ¬ is_axisymmetric char2) 
                                  (h3 : ¬ is_axisymmetric char3) : 
                                  is_axisymmetric char4 :=
sorry

end prove_axisymmetric_char4_l128_128915


namespace sin_30_eq_one_half_l128_128366

def Q_coords (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

theorem sin_30_eq_one_half :
  let Q := Q_coords (Real.pi / 6)
  Q.2 = 1 / 2 :=
by
  sorry

end sin_30_eq_one_half_l128_128366


namespace ticket_difference_l128_128807

/-- 
  Define the initial number of tickets Billy had,
  the number of tickets after buying a yoyo,
  and state the proof that the difference is 16.
--/

theorem ticket_difference (initial_tickets : ℕ) (remaining_tickets : ℕ) 
  (h₁ : initial_tickets = 48) (h₂ : remaining_tickets = 32) : 
  initial_tickets - remaining_tickets = 16 :=
by
  /- This is where the prover would go, 
     no need to implement it as we know the expected result -/
  sorry

end ticket_difference_l128_128807


namespace sum_of_factors_36_l128_128071

theorem sum_of_factors_36 : (∑ d in (Finset.range (36 + 1)).filter (λ d, 36 % d = 0), d) = 91 :=
by
  sorry

end sum_of_factors_36_l128_128071


namespace total_pieces_l128_128646

def pieces_from_friend : ℕ := 123
def pieces_from_brother : ℕ := 136
def pieces_needed : ℕ := 117

theorem total_pieces :
  pieces_from_friend + pieces_from_brother + pieces_needed = 376 :=
by
  unfold pieces_from_friend pieces_from_brother pieces_needed
  sorry

end total_pieces_l128_128646


namespace sin_30_eq_half_l128_128313

noncomputable def P : ℝ × ℝ := 
  have angle := real.pi / 6
  have x := real.cos angle
  have y := real.sin angle
  (x, y)  -- The point on the unit circle at 30 degrees

theorem sin_30_eq_half : real.sin (real.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l128_128313


namespace iso_triangle_perimeter_l128_128418

theorem iso_triangle_perimeter :
  ∃ p : ℕ, (p = 11 ∨ p = 13) ∧ ∃ a b : ℕ, a ≠ b ∧ a^2 - 8 * a + 15 = 0 ∧ b^2 - 8 * b + 15 = 0 :=
by
  sorry

end iso_triangle_perimeter_l128_128418


namespace geometric_mean_4_16_l128_128140

theorem geometric_mean_4_16 (x : ℝ) (h : x^2 = 4 * 16) : x = 8 ∨ x = -8 :=
sorry

end geometric_mean_4_16_l128_128140


namespace model1_best_fitting_effect_l128_128852

-- Definitions for the correlation coefficients of the models
def R1 : ℝ := 0.98
def R2 : ℝ := 0.80
def R3 : ℝ := 0.50
def R4 : ℝ := 0.25

-- Main theorem stating Model 1 has the best fitting effect
theorem model1_best_fitting_effect : |R1| > |R2| ∧ |R1| > |R3| ∧ |R1| > |R4| :=
by sorry

end model1_best_fitting_effect_l128_128852


namespace smallest_solution_l128_128671

theorem smallest_solution (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ 4) (h₃ : x ≠ 5) 
    (h_eq : 1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) : x = 4 - Real.sqrt 2 := 
by 
  sorry

end smallest_solution_l128_128671


namespace age_difference_l128_128089

-- Denote the ages of A, B, and C as a, b, and c respectively.
variables (a b c : ℕ)

-- The given condition
def condition : Prop := a + b = b + c + 12

-- Prove that C is 12 years younger than A.
theorem age_difference (h : condition a b c) : c = a - 12 :=
by {
  -- skip the actual proof here, as instructed
  sorry
}

end age_difference_l128_128089


namespace spike_crickets_hunted_morning_l128_128189

def crickets_hunted_in_morning (C : ℕ) (total_daily_crickets : ℕ) : Prop :=
  4 * C = total_daily_crickets

theorem spike_crickets_hunted_morning (C : ℕ) (total_daily_crickets : ℕ) :
  total_daily_crickets = 20 → crickets_hunted_in_morning C total_daily_crickets → C = 5 :=
by
  intros h1 h2
  sorry

end spike_crickets_hunted_morning_l128_128189


namespace xy_eq_one_l128_128704

theorem xy_eq_one (x y : ℝ) (h : x + y = (1 / x) + (1 / y) ∧ x + y ≠ 0) : x * y = 1 := by
  sorry

end xy_eq_one_l128_128704


namespace least_whole_number_subtracted_l128_128786

theorem least_whole_number_subtracted (x : ℕ) :
  ((6 - x) / (7 - x) < (16 / 21)) → x = 3 :=
by
  sorry

end least_whole_number_subtracted_l128_128786


namespace smallest_solution_l128_128668

theorem smallest_solution (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ 4) (h₃ : x ≠ 5) 
    (h_eq : 1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) : x = 4 - Real.sqrt 2 := 
by 
  sorry

end smallest_solution_l128_128668


namespace solution_set_of_inequality_l128_128897

theorem solution_set_of_inequality :
  { x : ℝ | x^2 + x - 2 < 0 } = { x : ℝ | -2 < x ∧ x < 1 } :=
sorry

end solution_set_of_inequality_l128_128897


namespace sin_of_30_degrees_l128_128342

-- Define P and D and their relationships on a unit circle
def P : ℝ × ℝ := (cos (π / 6), sin (π / 6))  -- Point P on the unit circle at 30 degrees
def D : ℝ × ℝ := (P.1, 0)  -- Foot of the altitude to the x-axis

theorem sin_of_30_degrees : sin (π / 6) = 1 / 2 :=
by 
  -- Use the fact that the triangle P,O,D is a 30-60-90 triangle
  -- and the unit circle properties.
  sorry

end sin_of_30_degrees_l128_128342


namespace round_table_six_people_l128_128726

-- Definition of the problem
def round_table_arrangements (n : ℕ) : ℕ :=
  nat.factorial n / n

-- The main theorem statement
theorem round_table_six_people : round_table_arrangements 6 = 120 :=
by 
  -- We implement the definition and calculations inline here directly.
  sorry

end round_table_six_people_l128_128726


namespace find_x_eq_nine_fourths_l128_128385

theorem find_x_eq_nine_fourths (x n : ℚ) (β : ℚ) (h1 : x = n + β) (h2 : n = ⌊x⌋) (h3 : ⌊x⌋ + x = 17 / 4) : x = 9 / 4 :=
by
  sorry

end find_x_eq_nine_fourths_l128_128385


namespace abs_sum_l128_128440

theorem abs_sum (a b c : ℚ) (h₁ : a = -1/4) (h₂ : b = -2) (h₃ : c = -11/4) :
  |a| + |b| - |c| = -1/2 :=
by {
  sorry
}

end abs_sum_l128_128440


namespace solution_set_of_inequality_l128_128892

theorem solution_set_of_inequality : {x : ℝ | x^2 + x - 2 < 0} = {x : ℝ | -2 < x ∧ x < 1} :=
by
  sorry

end solution_set_of_inequality_l128_128892


namespace smallest_solution_exists_l128_128690

noncomputable def is_solution (x : ℝ) : Prop := (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧ x ≠ 3 ∧ x ≠ 5 ∧ x ≠ 4

-- Statement of the problem without proof
theorem smallest_solution_exists : ∃ (x : ℝ), is_solution x ∧ ∀ (y : ℝ), is_solution y → x ≤ y :=
sorry

end smallest_solution_exists_l128_128690


namespace unoccupied_seats_in_business_class_l128_128239

def airplane_seating (fc bc ec : ℕ) (pfc pbc pec : ℕ) : Nat :=
  let num_ec := ec / 2
  let num_bc := num_ec - pfc
  bc - num_bc

theorem unoccupied_seats_in_business_class :
  airplane_seating 10 30 50 3 (50/2) (50/2) = 8 := by
    sorry

end unoccupied_seats_in_business_class_l128_128239


namespace original_average_rent_is_800_l128_128749

def original_rent (A : ℝ) : Prop :=
  let friends : ℝ := 4
  let old_rent : ℝ := 800
  let increased_rent : ℝ := old_rent * 1.25
  let new_total_rent : ℝ := (850 * friends)
  old_rent * 4 - 800 + increased_rent = new_total_rent

theorem original_average_rent_is_800 (A : ℝ) : original_rent A → A = 800 :=
by 
  sorry

end original_average_rent_is_800_l128_128749


namespace letters_written_l128_128452

theorem letters_written (nathan_rate : ℕ) (jacob_rate : ℕ) (combined_rate : ℕ) (hours : ℕ) :
  nathan_rate = 25 →
  jacob_rate = 2 * nathan_rate →
  combined_rate = nathan_rate + jacob_rate →
  hours = 10 →
  combined_rate * hours = 750 :=
by
  intros
  sorry

end letters_written_l128_128452


namespace find_x_l128_128391

theorem find_x (x : ℚ) (n : ℤ) (f : ℚ) (h1 : x = n + f) (h2 : n = ⌊x⌋) (h3 : f < 1): 
  ⌊x⌋ + x = 17 / 4 → x = 9 / 4 :=
by 
  sorry

end find_x_l128_128391


namespace final_sale_price_l128_128042

theorem final_sale_price (P P₁ P₂ P₃ : ℝ) (d₁ d₂ d₃ dx : ℝ) (x : ℝ)
  (h₁ : P = 600) 
  (h_d₁ : d₁ = 20) (h_d₂ : d₂ = 15) (h_d₃ : d₃ = 10)
  (h₁₁ : P₁ = P * (1 - d₁ / 100))
  (h₁₂ : P₂ = P₁ * (1 - d₂ / 100))
  (h₁₃ : P₃ = P₂ * (1 - d₃ / 100))
  (h_P₃_final : P₃ = 367.2) :
  P₃ * (100 - dx) / 100 = 367.2 * (100 - x) / 100 :=
by
  sorry

end final_sale_price_l128_128042


namespace moles_of_C6H6_l128_128437

def balanced_reaction (a b c d : ℕ) : Prop :=
  a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ a + b + c + d = 4

theorem moles_of_C6H6 (a b c d : ℕ) (h_balanced : balanced_reaction a b c d) :
  a = 1 := 
by 
  sorry

end moles_of_C6H6_l128_128437


namespace intersect_point_on_BC_l128_128847

open EuclideanGeometry

theorem intersect_point_on_BC (A B C M N X : Point) :
  is_midpoint M A B →
  is_midpoint N A C →
  tangent AX (circumcircle A B C) →
  ∃ ω_B : Circle, centroid BC M B ∧ tangent_at M (circumcircle M B X) ω_B ∧
  ∃ ω_C : Circle, centroid BC N C ∧ tangent_at N (circumcircle N C X) ω_C ∧
  ∃ D : Point, D ∈ intersection_points ω_B ω_C ∧ collinear B C D :=
sorry

end intersect_point_on_BC_l128_128847


namespace sin_30_eq_half_l128_128260

theorem sin_30_eq_half : Real.sin (30 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_eq_half_l128_128260


namespace sum_of_positive_factors_36_l128_128066

def sum_of_factors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, n % d = 0).sum

theorem sum_of_positive_factors_36 : sum_of_factors 36 = 91 :=
by
  sorry

end sum_of_positive_factors_36_l128_128066


namespace square_area_increase_l128_128634

theorem square_area_increase (s : ℝ) (h : s > 0) :
  ((1.15 * s) ^ 2 - s ^ 2) / s ^ 2 * 100 = 32.25 :=
by
  sorry

end square_area_increase_l128_128634


namespace twenty_kopeck_greater_than_ten_kopeck_l128_128195

-- Definitions of the conditions
variables (x y z : ℕ)
axiom total_coins : x + y + z = 30 
axiom total_value : 10 * x + 15 * y + 20 * z = 500 

-- The proof statement
theorem twenty_kopeck_greater_than_ten_kopeck : z > x :=
sorry

end twenty_kopeck_greater_than_ten_kopeck_l128_128195


namespace commodity_x_increase_rate_l128_128043

variable (x_increase : ℕ) -- annual increase in cents of commodity X
variable (y_increase : ℕ := 20) -- annual increase in cents of commodity Y
variable (x_2001_price : ℤ := 420) -- price of commodity X in cents in 2001
variable (y_2001_price : ℤ := 440) -- price of commodity Y in cents in 2001
variable (year_difference : ℕ := 2010 - 2001) -- difference in years between 2010 and 2001
variable (x_y_diff_2010 : ℕ := 70) -- cents by which X is more expensive than Y in 2010

theorem commodity_x_increase_rate :
  x_increase * year_difference = (x_2001_price + x_increase * year_difference) - (y_2001_price + y_increase * year_difference) + x_y_diff_2010 := by
  sorry

end commodity_x_increase_rate_l128_128043


namespace each_niece_gets_13_l128_128217

-- Define the conditions
def total_sandwiches : ℕ := 143
def number_of_nieces : ℕ := 11

-- Prove that each niece can get 13 ice cream sandwiches
theorem each_niece_gets_13 : total_sandwiches / number_of_nieces = 13 :=
by
  -- Proof omitted
  sorry

end each_niece_gets_13_l128_128217


namespace molecular_weight_proof_l128_128060

def atomic_weight_N : ℝ := 14.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_I : ℝ := 126.90

def molecular_weight (n_N n_H n_I : ℕ) : ℝ :=
  n_N * atomic_weight_N + n_H * atomic_weight_H + n_I * atomic_weight_I

theorem molecular_weight_proof : molecular_weight 1 4 1 = 144.95 :=
by {
  sorry
}

end molecular_weight_proof_l128_128060


namespace number_of_ordered_triples_l128_128129

theorem number_of_ordered_triples (a b c : ℤ) : 
  ∃ (n : ℕ), -31 <= a ∧ a <= 31 ∧ -31 <= b ∧ b <= 31 ∧ -31 <= c ∧ c <= 31 ∧ 
  (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ (a + b + c > 0) ∧ n = 117690 :=
by sorry

end number_of_ordered_triples_l128_128129


namespace range_of_f_l128_128410

noncomputable def f (x : ℝ) : ℝ := 3^(x - 2)

theorem range_of_f : Set.Icc 1 9 = {y : ℝ | ∃ x : ℝ, 2 ≤ x ∧ x ≤ 4 ∧ f x = y} :=
by
  sorry

end range_of_f_l128_128410


namespace year_2022_form_l128_128450

theorem year_2022_form :
  ∃ (a b c d e f g h i j : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
    f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
    g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
    h ≠ i ∧ h ≠ j ∧
    i ≠ j ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧ g < 10 ∧ h < 10 ∧ i < 10 ∧ j < 10 ∧
    2001 ≤ (a + b * c * d * e) / (f + g * h * i * j) ∧ (a + b * c * d * e) / (f + g * h * i * j) ≤ 2100 ∧
    (a + b * c * d * e) / (f + g * h * i * j) = 2022 :=
sorry

end year_2022_form_l128_128450


namespace large_box_chocolate_bars_l128_128505

theorem large_box_chocolate_bars (num_small_boxes : ℕ) (chocolates_per_box : ℕ) 
  (h1 : num_small_boxes = 18) (h2 : chocolates_per_box = 28) : 
  num_small_boxes * chocolates_per_box = 504 := by
  sorry

end large_box_chocolate_bars_l128_128505


namespace solve_equation1_solve_equation2_l128_128538

theorem solve_equation1 (x : ℝ) (h : 4 * x^2 - 81 = 0) : x = 9/2 ∨ x = -9/2 := 
sorry

theorem solve_equation2 (x : ℝ) (h : 8 * (x + 1)^3 = 27) : x = 1/2 := 
sorry

end solve_equation1_solve_equation2_l128_128538


namespace smallest_solution_l128_128686

-- Defining the equation as a condition
def equation (x : ℝ) : Prop := (1 / (x - 3)) + (1 / (x - 5)) = 4 / (x - 4)

-- Proving that the smallest solution is 4 - sqrt(2)
theorem smallest_solution : ∃ x : ℝ, equation x ∧ x = 4 - Real.sqrt 2 := 
by
  -- Proof is omitted
  sorry

end smallest_solution_l128_128686


namespace circles_externally_tangent_l128_128559

theorem circles_externally_tangent
  (r1 r2 d : ℝ)
  (hr1 : r1 = 2) (hr2 : r2 = 3)
  (hd : d = 5) :
  r1 + r2 = d :=
by
  sorry

end circles_externally_tangent_l128_128559


namespace OJ_perpendicular_PQ_l128_128708

noncomputable def quadrilateral (A B C D : Point) : Prop := sorry

noncomputable def inscribed (A B C D : Point) : Prop := sorry

noncomputable def circumscribed (A B C D : Point) : Prop := sorry

noncomputable def no_diameter (A B C D : Point) : Prop := sorry

noncomputable def intersection_of_external_bisectors (A B C D : Point) (P : Point) : Prop := sorry

noncomputable def incenter (A B C D J : Point) : Prop := sorry

noncomputable def circumcenter (A B C D O : Point) : Prop := sorry

noncomputable def PQ_perpendicular (O J P Q : Point) : Prop := sorry

theorem OJ_perpendicular_PQ (A B C D P Q J O : Point) :
  quadrilateral A B C D →
  inscribed A B C D →
  circumscribed A B C D →
  no_diameter A B C D →
  intersection_of_external_bisectors A B C D P →
  intersection_of_external_bisectors C D A B Q →
  incenter A B C D J →
  circumcenter A B C D O →
  PQ_perpendicular O J P Q :=
sorry

end OJ_perpendicular_PQ_l128_128708


namespace space_needed_between_apple_trees_l128_128875

-- Definitions based on conditions
def apple_tree_width : ℕ := 10
def peach_tree_width : ℕ := 12
def space_between_peach_trees : ℕ := 15
def total_space : ℕ := 71
def number_of_apple_trees : ℕ := 2
def number_of_peach_trees : ℕ := 2

-- Lean 4 theorem statement
theorem space_needed_between_apple_trees :
  (total_space 
   - (number_of_peach_trees * peach_tree_width + space_between_peach_trees))
  - (number_of_apple_trees * apple_tree_width) 
  = 12 := by
  sorry

end space_needed_between_apple_trees_l128_128875


namespace sum_of_positive_factors_of_36_l128_128074

theorem sum_of_positive_factors_of_36 : 
  let n := 36 in
  ∑ i in {0, 1, 2}, (2 ^ i) * ∑ j in {0, 1, 2}, (3 ^ j) = 91 :=
by
  let n := 36
  let sum_p := ∑ i in {0, 1, 2}, (2 ^ i)
  let sum_q := ∑ j in {0, 1, 2}, (3 ^ j)
  have h1 : sum_p = 1 + 2 + 4 := by
    sorry
  have h2 : sum_q = 1 + 3 + 9 := by
    sorry
  have h3 : sum_p * sum_q = 91 := by
    sorry
  exact h3

end sum_of_positive_factors_of_36_l128_128074


namespace percent_of_a_is_20_l128_128843

variable {a b c : ℝ}

theorem percent_of_a_is_20 (h1 : c = (x / 100) * a)
                          (h2 : c = 0.1 * b)
                          (h3 : b = 2 * a) :
  c = 0.2 * a := sorry

end percent_of_a_is_20_l128_128843


namespace sin_30_eq_half_l128_128348

theorem sin_30_eq_half
  (Q : ℝ × ℝ)
  (h1 : Q = (cos (π / 6), sin (π / 6)))
  (E : ℝ × ℝ)
  (h2 : E = (Q.1, 0))
  (h3 : ∃ x y z : ℝ, x / 2 = Q.2 ∧ y = Q.1 ∧ sqrt (x^2 + y^2) = z ∧ z = 1) :
  sin (π / 6) = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l128_128348


namespace range_of_m_l128_128412

variable (m : ℝ)

/-- Proposition p: For any x in ℝ, x^2 + 1 > m -/
def p := ∀ x : ℝ, x^2 + 1 > m

/-- Proposition q: The linear function f(x) = (2 - m) * x + 1 is an increasing function -/
def q := (2 - m) > 0

theorem range_of_m (h1 : p m ∨ q m) (h2 : ¬(p m ∧ q m)) : 1 < m ∧ m < 2 := 
sorry

end range_of_m_l128_128412


namespace algebraic_expression_domain_l128_128760

theorem algebraic_expression_domain (x : ℝ) : 
  (x + 2 ≥ 0) ∧ (x - 3 ≠ 0) ↔ (x ≥ -2) ∧ (x ≠ 3) := by
  sorry

end algebraic_expression_domain_l128_128760


namespace probability_twice_correct_l128_128812

noncomputable def probability_at_least_twice (x y : ℝ) : ℝ :=
if (0 ≤ x ∧ x ≤ 1000) ∧ (0 ≤ y ∧ y ≤ 3000) then
  if y ≥ 2*x then (1/6 : ℝ) else 0
else 0

theorem probability_twice_correct : probability_at_least_twice 500 1000 = (1/6 : ℝ) :=
sorry

end probability_twice_correct_l128_128812


namespace polygon_is_hexagon_l128_128508

-- Definitions
def side_length : ℝ := 8
def perimeter : ℝ := 48

-- The main theorem to prove
theorem polygon_is_hexagon : (perimeter / side_length = 6) ∧ (48 / 8 = 6) := 
by
  sorry

end polygon_is_hexagon_l128_128508


namespace sum_of_positive_factors_of_36_l128_128062

def prime_factorization_36 : Prop :=
  prime.factorization 36 = {2: 2, 3: 2}

def sum_of_divisors_formula (n : ℕ) (p q : ℕ) (np nq : ℕ) : ℕ :=
  (Finset.range (np + 1)).sum (λ i, p^i) * (Finset.range (nq + 1)).sum (λ j, q^j)

def sum_of_divisors_36 : Prop :=
  sum_of_divisors_formula 36 2 3 2 2 = 91

theorem sum_of_positive_factors_of_36 : prime_factorization_36 → sum_of_divisors_36 :=
by {
  sorry
}

end sum_of_positive_factors_of_36_l128_128062


namespace cos_135_eq_neg_inv_sqrt2_l128_128956

theorem cos_135_eq_neg_inv_sqrt2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt2_l128_128956


namespace smallest_positive_period_l128_128869

noncomputable def f (A ω φ : ℝ) (x : ℝ) := A * Real.sin (ω * x + φ)

theorem smallest_positive_period 
  (A ω φ T : ℝ) 
  (hA : A > 0) 
  (hω : ω > 0)
  (h1 : f A ω φ (π / 2) = f A ω φ (2 * π / 3))
  (h2 : f A ω φ (π / 6) = -f A ω φ (π / 2))
  (h3 : ∀ x1 x2, (π / 6) ≤ x1 → x1 ≤ x2 → x2 ≤ (π / 2) → f A ω φ x1 ≤ f A ω φ x2) :
  T = π :=
sorry

end smallest_positive_period_l128_128869


namespace sqrt_of_8_l128_128048

-- Definition of square root
def isSquareRoot (x : ℝ) (a : ℝ) : Prop := x * x = a

-- Theorem statement: The square root of 8 is ±√8
theorem sqrt_of_8 :
  ∃ x : ℝ, isSquareRoot x 8 ∧ (x = Real.sqrt 8 ∨ x = -Real.sqrt 8) :=
by
  sorry

end sqrt_of_8_l128_128048


namespace major_snow_shadow_length_l128_128160

theorem major_snow_shadow_length :
  ∃ (a1 d : ℝ), 
  (3 * a1 + 12 * d = 16.5) ∧ 
  (12 * a1 + 66 * d = 84) ∧
  (a1 + 11 * d = 12.5) := 
sorry

end major_snow_shadow_length_l128_128160


namespace sin_30_eq_half_l128_128319

-- Conditions
def point_Q (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

-- The angle 30 degrees in radians
def angle_30_deg := Real.pi / 6

-- Point Q at Angle 30 degrees to the positive x-axis
def Q := point_Q angle_30_deg

-- Formalized problem statement in Lean 4
theorem sin_30_eq_half : Real.sin angle_30_deg = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l128_128319


namespace pizza_eaten_after_six_trips_l128_128916

theorem pizza_eaten_after_six_trips :
  let a := (1 : ℝ) / 3
  let r := (1 : ℝ) / 3
  let n := 6
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 364 / 729 :=
by
  let a := (1 : ℝ) / 3
  let r := (1 : ℝ) / 3
  let n := 6
  let S_n := a * (1 - r^n) / (1 - r)
  have : S_n = (1 / 3) * (1 - (1 / 3)^6) / (1 - 1 / 3) := by sorry
  have : S_n = 364 / 729 := by sorry
  exact this

end pizza_eaten_after_six_trips_l128_128916


namespace f_prime_neg_one_l128_128721

-- Given conditions and definitions
def f (x : ℝ) (a b c : ℝ) := a * x^4 + b * x^2 + c

def f_prime (x : ℝ) (a b : ℝ) := 4 * a * x^3 + 2 * b * x

-- The theorem we need to prove
theorem f_prime_neg_one (a b c : ℝ) (h : f_prime 1 a b = 2) : f_prime (-1) a b = -2 := by
  sorry

end f_prime_neg_one_l128_128721


namespace sin_30_deg_l128_128308

theorem sin_30_deg : real.sin (30 * real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_deg_l128_128308


namespace find_d_l128_128474

theorem find_d (
  x : ℝ
) (
  h1 : 3 * x + 8 = 5
) (
  d : ℝ
) (
  h2 : d * x - 15 = -7
) : d = -8 :=
by
  sorry

end find_d_l128_128474


namespace sin_30_eq_half_l128_128284

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the point Q at 30 degrees on the unit circle
def Q : ℝ × ℝ := unit_circle (Real.pi / 6)

theorem sin_30_eq_half : (Real.sin (Real.pi / 6)) = 1 / 2 :=
by sorry

end sin_30_eq_half_l128_128284


namespace translate_one_chapter_in_three_hours_l128_128728

-- Definitions representing the conditions:
def jun_seok_time : ℝ := 4
def yoon_yeol_time : ℝ := 12

-- Question and Correct answer as a statement:
theorem translate_one_chapter_in_three_hours :
  (1 / (1 / jun_seok_time + 1 / yoon_yeol_time)) = 3 := by
sorry

end translate_one_chapter_in_three_hours_l128_128728


namespace number_of_tiles_per_row_l128_128607

theorem number_of_tiles_per_row : 
  ∀ (side_length_in_feet room_area_in_sqft : ℕ) (tile_width_in_inches : ℕ), 
  room_area_in_sqft = 256 → tile_width_in_inches = 8 → 
  side_length_in_feet * side_length_in_feet = room_area_in_sqft → 
  12 * side_length_in_feet / tile_width_in_inches = 24 := 
by
  intros side_length_in_feet room_area_in_sqft tile_width_in_inches h_area h_tile_width h_side_length
  sorry

end number_of_tiles_per_row_l128_128607


namespace area_of_large_square_l128_128472

theorem area_of_large_square (s : ℝ) (h : 2 * s^2 = 14) : 9 * s^2 = 63 := by
  sorry

end area_of_large_square_l128_128472


namespace sin_30_eq_half_l128_128362

theorem sin_30_eq_half : Real.sin (Real.pi / 6) = 1 / 2 := sorry

end sin_30_eq_half_l128_128362


namespace expression_equals_negative_two_l128_128553

def f (x : ℝ) : ℝ := x^3 - x - 1
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

theorem expression_equals_negative_two : 
  f 2023 + f' 2023 + f (-2023) - f' (-2023) = -2 :=
by
  sorry

end expression_equals_negative_two_l128_128553


namespace xiaoyu_money_left_l128_128096

def box_prices (x y z : ℝ) : Prop :=
  2 * x + 5 * y = z + 3 ∧ 5 * x + 2 * y = z - 3

noncomputable def money_left (x y z : ℝ) : ℝ :=
  z - 7 * x
  
theorem xiaoyu_money_left (x y z : ℝ) (hx : box_prices x y z) :
  money_left x y z = 7 := by
  sorry

end xiaoyu_money_left_l128_128096


namespace find_x_of_floor_plus_x_eq_17_over_4_l128_128394

theorem find_x_of_floor_plus_x_eq_17_over_4 (x : ℝ) (hx : ⌊x⌋ + x = 17 / 4) : x = 9 / 4 := 
sorry

end find_x_of_floor_plus_x_eq_17_over_4_l128_128394


namespace cubical_cake_l128_128792

noncomputable def cubical_cake_properties : Prop :=
  let a : ℝ := 3
  let top_area := (1 / 2) * 3 * 1.5
  let height := 3
  let volume := top_area * height
  let vertical_triangles_area := 2 * ((1 / 2) * 1.5 * 3)
  let vertical_rectangular_area := 3 * 3
  let iced_area := top_area + vertical_triangles_area + vertical_rectangular_area
  volume + iced_area = 22.5

theorem cubical_cake : cubical_cake_properties := sorry

end cubical_cake_l128_128792


namespace max_value_of_sum_max_value_achievable_l128_128184

theorem max_value_of_sum (a b c d : ℝ) 
  (h : a^6 + b^6 + c^6 + d^6 = 64) : a^7 + b^7 + c^7 + d^7 ≤ 128 :=
sorry

theorem max_value_achievable : ∃ a b c d : ℝ,
  a^6 + b^6 + c^6 + d^6 = 64 ∧ a^7 + b^7 + c^7 + d^7 = 128 :=
sorry

end max_value_of_sum_max_value_achievable_l128_128184


namespace smallest_solution_l128_128684

-- Defining the equation as a condition
def equation (x : ℝ) : Prop := (1 / (x - 3)) + (1 / (x - 5)) = 4 / (x - 4)

-- Proving that the smallest solution is 4 - sqrt(2)
theorem smallest_solution : ∃ x : ℝ, equation x ∧ x = 4 - Real.sqrt 2 := 
by
  -- Proof is omitted
  sorry

end smallest_solution_l128_128684


namespace rectangle_perimeter_l128_128606

theorem rectangle_perimeter 
(area : ℝ) (width : ℝ) (h1 : area = 200) (h2 : width = 10) : 
    ∃ (perimeter : ℝ), perimeter = 60 :=
by
  sorry

end rectangle_perimeter_l128_128606


namespace number_of_possible_values_l128_128753

theorem number_of_possible_values (x : ℕ) (h1 : x > 6) (h2 : x + 4 > 0) :
  ∃ (n : ℕ), n = 24 := 
sorry

end number_of_possible_values_l128_128753


namespace A_inter_complement_B_is_empty_l128_128435

open Set Real

noncomputable def U : Set Real := univ

noncomputable def A : Set Real := { x : Real | ∃ (y : Real), y = sqrt (Real.log x) }

noncomputable def B : Set Real := { y : Real | ∃ (x : Real), y = sqrt x }

theorem A_inter_complement_B_is_empty :
  A ∩ (U \ B) = ∅ :=
by
    sorry

end A_inter_complement_B_is_empty_l128_128435


namespace age_difference_l128_128793

variable (y m e : ℕ)

theorem age_difference (h1 : m = y + 3) (h2 : e = 3 * y) (h3 : e = 15) : 
  ∃ x, e = y + m + x ∧ x = 2 := by
  sorry

end age_difference_l128_128793


namespace inequality_solution_set_l128_128891

theorem inequality_solution_set : 
  { x : ℝ | x^2 + x - 2 < 0 } = { x : ℝ | -2 < x ∧ x < 1 } :=
sorry

end inequality_solution_set_l128_128891


namespace cos_135_eq_neg_sqrt2_div_2_l128_128945

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
by 
  sorry

end cos_135_eq_neg_sqrt2_div_2_l128_128945


namespace monotonic_increasing_interval_l128_128886

theorem monotonic_increasing_interval : ∀ x : ℝ, (x > 2) → ((x-3) * Real.exp x > 0) :=
sorry

end monotonic_increasing_interval_l128_128886


namespace isosceles_triangle_sides_l128_128547

theorem isosceles_triangle_sides (a b c : ℝ) (h_iso : a = b ∨ b = c ∨ c = a) (h_perimeter : a + b + c = 14) (h_side : a = 4 ∨ b = 4 ∨ c = 4) : 
  (a = 4 ∧ b = 5 ∧ c = 5) ∨ (a = 5 ∧ b = 5 ∧ c = 4) ∨ (a = 4 ∧ b = 4 ∧ c = 6) ∨ (a = 4 ∧ b = 6 ∧ c = 4) :=
  sorry

end isosceles_triangle_sides_l128_128547


namespace non_negative_real_inequality_l128_128460

theorem non_negative_real_inequality
  {a b c : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  a^4 + b^4 + c^4 - 2 * (a^2 * b^2 + a^2 * c^2 + b^2 * c^2) + a^2 * b * c + b^2 * a * c + c^2 * a * b ≥ 0 :=
by
  sorry

end non_negative_real_inequality_l128_128460


namespace simplify_fraction_l128_128186

variable {F : Type*} [Field F]

theorem simplify_fraction (a b : F) (h: a ≠ -1) :
  b / (a * b + b) = 1 / (a + 1) :=
by
  sorry

end simplify_fraction_l128_128186


namespace smallest_solution_to_equation_l128_128663

theorem smallest_solution_to_equation :
  ∀ x : ℝ, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) → x = 4 - real.sqrt 2 :=
  sorry

end smallest_solution_to_equation_l128_128663


namespace johns_gym_time_l128_128729

noncomputable def time_spent_at_gym (day : String) : ℝ :=
  match day with
  | "Monday" => 1 + 0.5
  | "Tuesday" => 40/60 + 20/60 + 15/60
  | "Thursday" => 40/60 + 20/60 + 15/60
  | "Saturday" => 1.5 + 0.75
  | "Sunday" => 10/60 + 50/60 + 10/60
  | _ => 0

noncomputable def total_hours_per_week : ℝ :=
  time_spent_at_gym "Monday" 
  + 2 * time_spent_at_gym "Tuesday" 
  + time_spent_at_gym "Saturday" 
  + time_spent_at_gym "Sunday"

theorem johns_gym_time : total_hours_per_week = 7.4167 := by
  sorry

end johns_gym_time_l128_128729


namespace handshake_problem_l128_128636

noncomputable def number_of_handshakes (n : ℕ) : ℕ :=
  n.choose 2

theorem handshake_problem : number_of_handshakes 25 = 300 := 
  by
  sorry

end handshake_problem_l128_128636


namespace value_of_k_l128_128540

theorem value_of_k :
  ∃ k : ℚ, (∀ x : ℝ, (x + 5) * (x + 2) = k + 3 * x →  (x^2 + 7 * x + 10) = (k + 3 * x)) ∧
  ∃ x : ℝ, x^2 + 4 * x + (10 - k) = 0 ∧ discriminant (x^2 + 4 * x + (10 - k)) = 0 :=
begin
  use 6,
  sorry
end

end value_of_k_l128_128540


namespace girls_maple_grove_correct_l128_128112

variables (total_students boys girls pine_ridge_students maple_grove_students boys_pine_ridge : ℕ)
variables (girls_maple_grove : ℕ)

-- Conditions
def conditions : Prop :=
  total_students = 150 ∧ 
  boys = 82 ∧ 
  girls = 68 ∧ 
  pine_ridge_students = 70 ∧ 
  maple_grove_students = 80 ∧ 
  boys_pine_ridge = 36 ∧ 
  girls_maple_grove = girls - (pine_ridge_students - boys_pine_ridge)

-- Question and Answer translated to a proposition
def proof_problem : Prop :=
  conditions total_students boys girls pine_ridge_students maple_grove_students boys_pine_ridge girls_maple_grove → 
  girls_maple_grove = 34

-- Statement
theorem girls_maple_grove_correct : proof_problem total_students boys girls pine_ridge_students maple_grove_students boys_pine_ridge girls_maple_grove :=
by {
  sorry -- Proof omitted
}

end girls_maple_grove_correct_l128_128112


namespace perimeter_of_rectangular_field_l128_128194

theorem perimeter_of_rectangular_field (L B : ℝ) 
    (h1 : B = 0.60 * L) 
    (h2 : L * B = 37500) : 
    2 * L + 2 * B = 800 :=
by 
  -- proof goes here
  sorry

end perimeter_of_rectangular_field_l128_128194


namespace max_handshakes_l128_128635

theorem max_handshakes (n : ℕ) (h : n = 25) : ∃ k : ℕ, k = 300 :=
by
  use (n * (n - 1)) / 2
  have h_eq : n = 25 := h
  rw h_eq
  sorry

end max_handshakes_l128_128635


namespace half_of_number_l128_128922

theorem half_of_number (N : ℝ)
  (h1 : (4 / 15) * (5 / 7) * N = (4 / 9) * (2 / 5) * N + 8) : 
  (N / 2) = 315 := 
sorry

end half_of_number_l128_128922


namespace count_ones_digits_of_numbers_divisible_by_4_and_3_l128_128532

theorem count_ones_digits_of_numbers_divisible_by_4_and_3 :
  let eligible_numbers := { n : ℕ | n < 100 ∧ n % 4 = 0 ∧ n % 3 = 0 }
  ∃ (digits : Finset ℕ), 
    (∀ n ∈ eligible_numbers, n % 10 ∈ digits) ∧
    digits.card = 5 :=
by
  sorry

end count_ones_digits_of_numbers_divisible_by_4_and_3_l128_128532


namespace sum_of_positive_factors_36_l128_128075

def sum_of_divisors (n : Nat) : Nat :=
  (List.range (n + 1)).filter (fun m => n % m = 0) |>.sum

theorem sum_of_positive_factors_36 : sum_of_divisors 36 = 91 := by
  sorry

end sum_of_positive_factors_36_l128_128075


namespace smallest_solution_l128_128676

theorem smallest_solution (x : ℝ) (h : (1 / (x - 3)) + (1 / (x - 5)) = (4 / (x - 4))) : 
  x = 5 - 2 * Real.sqrt 2 :=
sorry

end smallest_solution_l128_128676


namespace m_range_and_simplification_l128_128557

theorem m_range_and_simplification (x y m : ℝ)
  (h1 : (3 * (x + 1) / 2) + y = 2)
  (h2 : 3 * x - m = 2 * y)
  (hx : x ≤ 1)
  (hy : y ≤ 1) :
  (-3 ≤ m) ∧ (m ≤ 5) ∧ (|x - 1| + |y - 1| + |m + 3| + |m - 5| - |x + y - 2| = 8) := 
by sorry

end m_range_and_simplification_l128_128557


namespace inequality_solution_set_l128_128889

theorem inequality_solution_set : 
  { x : ℝ | x^2 + x - 2 < 0 } = { x : ℝ | -2 < x ∧ x < 1 } :=
sorry

end inequality_solution_set_l128_128889


namespace geometric_sequence_problem_l128_128448

noncomputable def a₂ (a₁ q : ℝ) : ℝ := a₁ * q
noncomputable def a₃ (a₁ q : ℝ) : ℝ := a₁ * q^2
noncomputable def a₄ (a₁ q : ℝ) : ℝ := a₁ * q^3
noncomputable def S₆ (a₁ q : ℝ) : ℝ := (a₁ * (1 - q^6)) / (1 - q)

theorem geometric_sequence_problem
  (a₁ q : ℝ)
  (h1 : a₁ * a₂ a₁ q * a₃ a₁ q = 27)
  (h2 : a₂ a₁ q + a₄ a₁ q = 30)
  : ((a₁ = 1 ∧ q = 3) ∨ (a₁ = -1 ∧ q = -3))
    ∧ (if a₁ = 1 ∧ q = 3 then S₆ a₁ q = 364 else true)
    ∧ (if a₁ = -1 ∧ q = -3 then S₆ a₁ q = -182 else true) :=
by
  -- Proof goes here
  sorry

end geometric_sequence_problem_l128_128448


namespace call_center_agents_ratio_l128_128501

theorem call_center_agents_ratio
  (a b : ℕ) -- Number of agents in teams A and B
  (x : ℝ) -- Calls each member of team B processes
  (h1 : (a : ℝ) / (b : ℝ) = 5 / 8)
  (h2 : b * x * 4 / 7 + a * 6 / 5 * x * 3 / 7 = b * x + a * 6 / 5 * x) :
  (a : ℝ) / (b : ℝ) = 5 / 8 :=
by
  sorry

end call_center_agents_ratio_l128_128501


namespace find_divisor_l128_128592

theorem find_divisor (x : ℕ) (h : 144 = (x * 13) + 1) : x = 11 := by
  sorry

end find_divisor_l128_128592


namespace sin_of_30_degrees_l128_128341

-- Define P and D and their relationships on a unit circle
def P : ℝ × ℝ := (cos (π / 6), sin (π / 6))  -- Point P on the unit circle at 30 degrees
def D : ℝ × ℝ := (P.1, 0)  -- Foot of the altitude to the x-axis

theorem sin_of_30_degrees : sin (π / 6) = 1 / 2 :=
by 
  -- Use the fact that the triangle P,O,D is a 30-60-90 triangle
  -- and the unit circle properties.
  sorry

end sin_of_30_degrees_l128_128341


namespace fence_width_l128_128594

theorem fence_width (L W : ℝ) 
  (circumference_eq : 2 * (L + W) = 30)
  (width_eq : W = 2 * L) : 
  W = 10 :=
by 
  sorry

end fence_width_l128_128594


namespace artist_paints_33_square_meters_l128_128804

/-
Conditions:
1. The artist has 14 cubes.
2. Each cube has an edge of 1 meter.
3. The cubes are arranged in a pyramid-like structure with three layers.
4. The top layer has 1 cube, the middle layer has 4 cubes, and the bottom layer has 9 cubes.
-/

def exposed_surface_area (num_cubes : Nat) (layer1 : Nat) (layer2 : Nat) (layer3 : Nat) : Nat :=
  let layer1_area := 5 -- Each top layer cube has 5 faces exposed
  let layer2_edge_cubes := 4 -- Count of cubes on the edge in middle layer
  let layer2_area := layer2_edge_cubes * 3 -- Each middle layer edge cube has 3 faces exposed
  let layer3_area := 9 -- Each bottom layer cube has 1 face exposed
  let top_faces := layer1 + layer2 + layer3 -- All top faces exposed
  layer1_area + layer2_area + layer3_area + top_faces

theorem artist_paints_33_square_meters :
  exposed_surface_area 14 1 4 9 = 33 := 
sorry

end artist_paints_33_square_meters_l128_128804


namespace sin_30_eq_half_l128_128318

-- Conditions
def point_Q (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

-- The angle 30 degrees in radians
def angle_30_deg := Real.pi / 6

-- Point Q at Angle 30 degrees to the positive x-axis
def Q := point_Q angle_30_deg

-- Formalized problem statement in Lean 4
theorem sin_30_eq_half : Real.sin angle_30_deg = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l128_128318


namespace length_of_segment_GH_l128_128028

theorem length_of_segment_GH (a1 a2 a3 a4 : ℕ)
  (h1 : a1 = a2 + 11)
  (h2 : a2 = a3 + 5)
  (h3 : a3 = a4 + 13)
  : a1 - a4 = 29 :=
by
  sorry

end length_of_segment_GH_l128_128028


namespace sin_30_eq_half_l128_128309

noncomputable def P : ℝ × ℝ := 
  have angle := real.pi / 6
  have x := real.cos angle
  have y := real.sin angle
  (x, y)  -- The point on the unit circle at 30 degrees

theorem sin_30_eq_half : real.sin (real.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l128_128309


namespace quadratic_function_min_value_in_interval_l128_128038

noncomputable def quadratic_function (x : ℝ) : ℝ :=
  x^2 - 6 * x + 10

theorem quadratic_function_min_value_in_interval :
  ∀ (x : ℝ), 2 ≤ x ∧ x < 5 → (∃ min_val : ℝ, min_val = 1) ∧ (∀ upper_bound : ℝ, ∃ x0 : ℝ, x0 < 5 ∧ quadratic_function x0 > upper_bound) := 
by
  sorry

end quadratic_function_min_value_in_interval_l128_128038


namespace find_second_number_l128_128789

theorem find_second_number (x : ℝ) : 217 + x + 0.217 + 2.0017 = 221.2357 → x = 2.017 :=
by
  sorry

end find_second_number_l128_128789


namespace total_ounces_of_drink_l128_128859

-- Definitions based on the conditions
def coke_parts : ℕ := 2
def sprite_parts : ℕ := 1
def mountain_dew_parts : ℕ := 3
def coke_ounces : ℕ := 6

-- Theorem stating the total ounces in the drink
theorem total_ounces_of_drink : 
  let total_parts := coke_parts + sprite_parts + mountain_dew_parts
  let ounces_per_part := coke_ounces / coke_parts
in total_parts * ounces_per_part = 18 := by
  sorry

end total_ounces_of_drink_l128_128859


namespace cheryl_material_left_l128_128523

def square_yards_left (bought1 bought2 used : ℚ) : ℚ :=
  bought1 + bought2 - used

theorem cheryl_material_left :
  square_yards_left (4/19) (2/13) (0.21052631578947367 : ℚ) = (0.15384615384615385 : ℚ) :=
by
  sorry

end cheryl_material_left_l128_128523


namespace jackson_emma_probability_l128_128445

open_locale big_operators

theorem jackson_emma_probability :
  ∃ (p : ℚ), p = 55 / 303 ∧ jackson_emma_probability = 55 / 303 :=
begin
  let total_ways := (nat.choose 12 6) * (nat.choose 12 6),
  let successful_ways := (nat.choose 12 3) * (nat.choose 9 3) * (nat.choose 9 3),
  let probability := (successful_ways : ℚ) / total_ways,
  use probability,
  split,
  { refl },
  { 
    norm_num1,
    sorry 
  }
end

end jackson_emma_probability_l128_128445


namespace lean_math_problem_l128_128706

noncomputable theory
open Real

variable {f : ℝ → ℝ}

-- Hypotheses
def differentiable_on_ℝ (f : ℝ → ℝ) := ∀ x : ℝ, differentiable_at ℝ f x

def condition1 (f : ℝ → ℝ) := differentiable_on_ℝ f
def condition2 (f : ℝ → ℝ) := ∀ x : ℝ, deriv f x + f x < 0

-- Conclusion
theorem lean_math_problem
  (h1 : condition1 f)
  (h2 : condition2 f) :
  ∀ m : ℝ, (f (m - m^2)) / (exp (m^2 - m + 1)) > f 1 :=
sorry

end lean_math_problem_l128_128706


namespace find_length_of_MN_l128_128157

theorem find_length_of_MN (A B C M N : ℝ × ℝ)
  (AB AC : ℝ) (M_midpoint : M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2))
  (N_midpoint : N = ((A.1 + C.1) / 2, (A.2 + C.2) / 2))
  (length_AB : abs (B.1 - A.1) + abs (B.2 - A.2) = 15)
  (length_AC : abs (C.1 - A.1) + abs (C.2 - A.2) = 20) :
  abs (N.1 - M.1) + abs (N.2 - M.2) = 40 / 3 := sorry

end find_length_of_MN_l128_128157


namespace frame_interior_edge_sum_l128_128795

theorem frame_interior_edge_sum (y : ℝ) :
  ( ∀ outer_edge1 : ℝ, outer_edge1 = 7 →
    ∀ frame_width : ℝ, frame_width = 2 →
    ∀ frame_area : ℝ, frame_area = 30 →
    7 * y - (3 * (y - 4)) = 30) → 
  (7 * y - (4 * y - 12) ) / 4 = 4.5 → 
  (3 + (y - 4)) * 2 = 7 :=
sorry

end frame_interior_edge_sum_l128_128795


namespace fractionD_is_unchanged_l128_128568

-- Define variables x and y
variable (x y : ℚ)

-- Define the fractions
def fractionA := x / (y + 1)
def fractionB := (x + y) / (x + 1)
def fractionC := (x * y) / (x + y)
def fractionD := (2 * x) / (3 * x - y)

-- Define the transformation
def transform (a b : ℚ) : ℚ × ℚ := (3 * a, 3 * b)

-- Define the new fractions after transformation
def newFractionA := (3 * x) / (3 * y + 1)
def newFractionB := (3 * x + 3 * y) / (3 * x + 1)
def newFractionC := (9 * x * y) / (3 * x + 3 * y)
def newFractionD := (6 * x) / (9 * x - 3 * y)

-- The proof problem statement
theorem fractionD_is_unchanged :
  fractionD x y = newFractionD x y ∧
  fractionA x y ≠ newFractionA x y ∧
  fractionB x y ≠ newFractionB x y ∧
  fractionC x y ≠ newFractionC x y := sorry

end fractionD_is_unchanged_l128_128568


namespace cosine_of_angle_in_second_quadrant_l128_128439

theorem cosine_of_angle_in_second_quadrant
  (α : ℝ)
  (h1 : Real.sin α = 1 / 3)
  (h2 : π / 2 < α ∧ α < π) :
  Real.cos α = - (2 * Real.sqrt 2) / 3 :=
by
  sorry

end cosine_of_angle_in_second_quadrant_l128_128439


namespace adam_money_given_l128_128937

theorem adam_money_given (original_money : ℕ) (final_money : ℕ) (money_given : ℕ) :
  original_money = 79 →
  final_money = 92 →
  money_given = final_money - original_money →
  money_given = 13 := by
sorry

end adam_money_given_l128_128937


namespace probability_allison_greater_l128_128245

open Probability

-- Definitions for the problem
noncomputable def die_A : ℕ := 4

noncomputable def die_B : PMF ℕ :=
  PMF.uniform_of_fin (fin 6) -- Could also explicitly write as PMF.of_list [(1, 1/6), ..., (6, 1/6)]

noncomputable def die_N : PMF ℕ :=
  PMF.of_list [(3, 3/6), (5, 3/6)]

-- The target event: Allison's roll > Brian's roll and Noah's roll
noncomputable def event (roll_A : ℕ) (roll_B roll_N : PMF ℕ) :=
  ∀ b n, b ∈ [1, 2, 3] → n ∈ [3] → roll_A > b ∧ roll_A > n

-- The probability calculation
theorem probability_allison_greater :
  (∑' (b : ℕ) (h_b : b < 4) (n : ℕ) (h_n : n = 3), die_B b * die_N n)
  = 1 / 4 :=
by
  -- assumed rolls
  let roll_A := 4
  let prob_B := 1 / 2
  let prob_N := 1 / 2
  
  -- skip proof, but assert the correct result.
  sorry

end probability_allison_greater_l128_128245


namespace sum_of_s_and_t_eq_neg11_l128_128441

theorem sum_of_s_and_t_eq_neg11 (s t : ℝ) 
  (h1 : ∀ x, x = 3 → x^2 + s * x + t = 0)
  (h2 : ∀ x, x = -4 → x^2 + s * x + t = 0) :
  s + t = -11 :=
sorry

end sum_of_s_and_t_eq_neg11_l128_128441


namespace remaining_digits_count_l128_128608

theorem remaining_digits_count 
  (avg9 : ℝ) (avg4 : ℝ) (avgRemaining : ℝ) (h1 : avg9 = 18) (h2 : avg4 = 8) (h3 : avgRemaining = 26) :
  let S := 9 * avg9
  let S4 := 4 * avg4
  let S_remaining := S - S4
  let N := S_remaining / avgRemaining
  N = 5 := 
by
  sorry

end remaining_digits_count_l128_128608


namespace cost_of_shirts_l128_128131

theorem cost_of_shirts : 
  let shirt1 := 15
  let shirt2 := 15
  let shirt3 := 15
  let shirt4 := 20
  let shirt5 := 20
  shirt1 + shirt2 + shirt3 + shirt4 + shirt5 = 85 := 
by
  sorry

end cost_of_shirts_l128_128131


namespace sin_30_eq_one_half_l128_128368

def Q_coords (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

theorem sin_30_eq_one_half :
  let Q := Q_coords (Real.pi / 6)
  Q.2 = 1 / 2 :=
by
  sorry

end sin_30_eq_one_half_l128_128368


namespace initial_apples_l128_128872

-- Definitions of the conditions
def Minseok_ate : Nat := 3
def Jaeyoon_ate : Nat := 3
def apples_left : Nat := 2

-- The proposition we need to prove
theorem initial_apples : Minseok_ate + Jaeyoon_ate + apples_left = 8 := by
  sorry

end initial_apples_l128_128872


namespace six_by_six_board_partition_l128_128226

theorem six_by_six_board_partition (P : Prop) (Q : Prop) 
(board : ℕ × ℕ) (domino : ℕ × ℕ) 
(h1 : board = (6, 6)) 
(h2 : domino = (2, 1)) 
(h3 : P → Q ∧ Q → P) :
  ∃ R₁ R₂ : ℕ × ℕ, (R₁ = (p, q) ∧ R₂ = (r, s) ∧ ((R₁.1 * R₁.2 + R₂.1 * R₂.2) = 36)) :=
sorry

end six_by_six_board_partition_l128_128226


namespace find_b_l128_128745

theorem find_b (g : ℝ → ℝ) (g_inv : ℝ → ℝ) (b : ℝ) (h_g_def : ∀ x, g x = 1 / (3 * x + b)) (h_g_inv_def : ∀ x, g_inv x = (1 - 3 * x) / (3 * x)) :
  b = 3 :=
by
  sorry

end find_b_l128_128745


namespace minimum_value_amgm_l128_128867

theorem minimum_value_amgm (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 27) : (a + 3 * b + 9 * c) ≥ 27 :=
by
  sorry

end minimum_value_amgm_l128_128867


namespace sin_30_eq_half_l128_128292

theorem sin_30_eq_half : Real.sin (Float.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l128_128292


namespace sin_thirty_deg_l128_128328

theorem sin_thirty_deg : Real.sin (π / 6) = 1 / 2 := 
by 
-- Assume that the unit circle point corresponding to 30° is (√3/2, 1/2)
-- Sine of an angle in the unit circle is the y-coordinate of the corresponding point
sorry

end sin_thirty_deg_l128_128328


namespace sin_30_eq_half_l128_128361

theorem sin_30_eq_half : Real.sin (Real.pi / 6) = 1 / 2 := sorry

end sin_30_eq_half_l128_128361


namespace find_other_side_length_l128_128578

variable (total_shingles : ℕ)
variable (shingles_per_sqft : ℕ)
variable (num_roofs : ℕ)
variable (side_length : ℕ)

theorem find_other_side_length
  (h1 : total_shingles = 38400)
  (h2 : shingles_per_sqft = 8)
  (h3 : num_roofs = 3)
  (h4 : side_length = 20)
  : (total_shingles / shingles_per_sqft / num_roofs / 2) / side_length = 40 :=
by
  sorry

end find_other_side_length_l128_128578


namespace remainder_y150_div_yminus2_4_l128_128536

theorem remainder_y150_div_yminus2_4 (y : ℝ) :
  (y ^ 150) % ((y - 2) ^ 4) = 554350 * (y - 2) ^ 3 + 22350 * (y - 2) ^ 2 + 600 * (y - 2) + 8 * 2 ^ 147 :=
by
  sorry

end remainder_y150_div_yminus2_4_l128_128536


namespace xiaojuan_savings_l128_128085

-- Define the conditions
def spent_on_novel (savings : ℝ) : ℝ := 0.5 * savings
def mother_gave : ℝ := 5
def spent_on_dictionary (amount_given : ℝ) : ℝ := 0.5 * amount_given + 0.4
def remaining_amount : ℝ := 7.2

-- Define the theorem stating the equivalence
theorem xiaojuan_savings : ∃ (savings: ℝ), spent_on_novel savings + mother_gave - spent_on_dictionary mother_gave - remaining_amount = savings / 2 ∧ savings = 20.4 :=
by {
  sorry
}

end xiaojuan_savings_l128_128085


namespace triangle_classification_l128_128421

theorem triangle_classification 
  (a b c : ℝ) 
  (h : (b^2 + a^2) * (b - a) = b * c^2 - a * c^2) : 
  (a = b ∨ a^2 + b^2 = c^2) :=
by sorry

end triangle_classification_l128_128421


namespace max_f_value_area_of_triangle_l128_128822

open Real

-- Definition of vectors a and b
def a (x : ℝ) : ℝ × ℝ := (2 * cos x, sin x - cos x)
def b (x : ℝ) : ℝ × ℝ := (sqrt 3 * sin x, sin x + cos x)

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := (a x).fst * (b x).fst + (a x).snd * (b x).snd

-- Definition of the conditions
def triangle_cond (a b c A B C : ℝ) := a + b = 2 * sqrt 3 ∧ c = sqrt 6 ∧ f C = 2

-- Tuple proving the maximum value condition for f(x)
theorem max_f_value (x : ℝ) : (∃ k : ℤ, x = k * π + π / 3) ↔ f x = 2 :=
sorry

-- Area calculation for triangle ABC given the conditions
theorem area_of_triangle (a b c A B C : ℝ) (h : triangle_cond a b c A B C) : 
  let s := (a + b + c) / 2 in
  sqrt (s * (s - a) * (s - b) * (s - c)) = sqrt 3 / 2 :=
sorry

end max_f_value_area_of_triangle_l128_128822


namespace initial_number_of_machines_l128_128601

theorem initial_number_of_machines
  (x : ℕ)
  (h1 : x * 270 = 1080)
  (h2 : 20 * 3600 = 144000)
  (h3 : ∀ y, (20 * y * 4 = 3600) → y = 45) :
  x = 6 :=
by
  sorry

end initial_number_of_machines_l128_128601


namespace number_of_subsets_with_mean_of_remaining_7_l128_128712

open Finset

def original_set : Finset ℕ := (range 14).filter (λ x, 0 < x)

theorem number_of_subsets_with_mean_of_remaining_7 :
  (∑ s in (original_set.ssubsetsLen 3).filter (λ t, (original_set.sum id - t.sum id) = 70), 1) = 5 :=
by
  sorry

end number_of_subsets_with_mean_of_remaining_7_l128_128712


namespace negation_of_proposition_l128_128204

variable (a b : ℝ)

theorem negation_of_proposition :
  (¬ (a * b = 0 → a = 0 ∨ b = 0)) ↔ (a * b ≠ 0 → a ≠ 0 ∧ b ≠ 0) :=
by
  sorry

end negation_of_proposition_l128_128204


namespace smallest_solution_l128_128677

theorem smallest_solution (x : ℝ) (h : (1 / (x - 3)) + (1 / (x - 5)) = (4 / (x - 4))) : 
  x = 5 - 2 * Real.sqrt 2 :=
sorry

end smallest_solution_l128_128677


namespace ratio_of_term_to_difference_l128_128899

def arithmetic_progression_sum (n a d : ℕ) : ℕ := (n * (2 * a + (n - 1) * d)) / 2

theorem ratio_of_term_to_difference (a d : ℕ) 
  (h1: arithmetic_progression_sum 7 a d = arithmetic_progression_sum 3 a d + 20)
  (h2 : d ≠ 0) : a / d = 1 / 2 := 
by 
  sorry

end ratio_of_term_to_difference_l128_128899


namespace largest_common_term_in_range_l128_128248

theorem largest_common_term_in_range :
  ∃ (a : ℕ), a < 150 ∧ (∃ (n : ℕ), a = 3 + 8 * n) ∧ (∃ (n : ℕ), a = 5 + 9 * n) ∧ a = 131 :=
by
  sorry

end largest_common_term_in_range_l128_128248


namespace sum_of_factors_l128_128078

-- Define 36
def num : ℕ := 36

-- Prime factorization of 36
def prime_factorization : (ℕ × ℕ) := (2, 2) -- represents 2^2
def prime_factorization_2 : (ℕ × ℕ) := (3, 2) -- represents 3^2

-- Sum of divisors function formula
def sigma (n : ℕ) : ℕ := 
  (1 + 2 + 2^2) * (1 + 3 + 3^2)

-- Theorem: sum of the positive factors of 36 is 91
theorem sum_of_factors : sigma num = 91 := 
by {
  -- Skipping the proof steps
  sorry
}

end sum_of_factors_l128_128078


namespace original_price_is_correct_l128_128642

-- Given conditions as Lean definitions
def reduced_price : ℝ := 2468
def reduction_amount : ℝ := 161.46

-- To find the original price including the sales tax
def original_price_including_tax (P : ℝ) : Prop :=
  P - reduction_amount = reduced_price

-- The proof statement to show the price is 2629.46
theorem original_price_is_correct : original_price_including_tax 2629.46 :=
by
  sorry

end original_price_is_correct_l128_128642


namespace find_x_satisfying_conditions_l128_128404

theorem find_x_satisfying_conditions :
  ∃ x : ℕ, (x % 2 = 1) ∧ (x % 3 = 2) ∧ (x % 4 = 3) ∧ (x % 5 = 4) ∧ x = 59 :=
by
  sorry

end find_x_satisfying_conditions_l128_128404


namespace sin_thirty_degree_l128_128277

theorem sin_thirty_degree : Real.sin (30 * Real.pi / 180) = 1 / 2 := 
by
  sorry

end sin_thirty_degree_l128_128277


namespace rectangle_diagonal_l128_128766

theorem rectangle_diagonal (l w : ℝ) (hl : l = 40) (hw : w = 40 * Real.sqrt 2) :
  Real.sqrt (l^2 + w^2) = 40 * Real.sqrt 3 :=
by
  rw [hl, hw]
  sorry

end rectangle_diagonal_l128_128766


namespace annie_budget_l128_128249

theorem annie_budget :
  let budget := 120
  let hamburger_count := 8
  let milkshake_count := 6
  let hamburgerA := 4
  let milkshakeA := 5
  let hamburgerB := 3.5
  let milkshakeB := 6
  let hamburgerC := 5
  let milkshakeC := 4
  let costA := hamburgerA * hamburger_count + milkshakeA * milkshake_count
  let costB := hamburgerB * hamburger_count + milkshakeB * milkshake_count
  let costC := hamburgerC * hamburger_count + milkshakeC * milkshake_count
  let min_cost := min costA (min costB costC)
  budget - min_cost = 58 :=
by {
  sorry
}

end annie_budget_l128_128249


namespace compute_expression_l128_128526

theorem compute_expression : 6^2 - 4 * 5 + 4^2 = 32 := by
  sorry

end compute_expression_l128_128526


namespace kia_vehicle_count_l128_128005

theorem kia_vehicle_count (total_vehicles : Nat) (dodge_vehicles : Nat) (hyundai_vehicles : Nat) 
    (h1 : total_vehicles = 400)
    (h2 : dodge_vehicles = total_vehicles / 2)
    (h3 : hyundai_vehicles = dodge_vehicles / 2) : 
    (total_vehicles - dodge_vehicles - hyundai_vehicles) = 100 := 
by sorry

end kia_vehicle_count_l128_128005


namespace triangle_cos_identity_l128_128856

variable {A B C : ℝ} -- Angle A, B, C are real numbers representing the angles of the triangle
variable {a b c : ℝ} -- Sides a, b, c are real numbers representing the lengths of the sides of the triangle

theorem triangle_cos_identity (h : 2 * b = a + c) : 5 * (Real.cos A) - 4 * (Real.cos A) * (Real.cos C) + 5 * (Real.cos C) = 4 :=
by
  sorry

end triangle_cos_identity_l128_128856


namespace sin_thirty_degree_l128_128280

theorem sin_thirty_degree : Real.sin (30 * Real.pi / 180) = 1 / 2 := 
by
  sorry

end sin_thirty_degree_l128_128280


namespace purely_imaginary_iff_real_iff_second_quadrant_iff_l128_128701

def Z (m : ℝ) : ℂ := ⟨m^2 - 2 * m - 3, m^2 + 3 * m + 2⟩

theorem purely_imaginary_iff (m : ℝ) : (Z m).re = 0 ∧ (Z m).im ≠ 0 ↔ m = 3 :=
by sorry

theorem real_iff (m : ℝ) : (Z m).im = 0 ↔ m = -1 ∨ m = -2 :=
by sorry

theorem second_quadrant_iff (m : ℝ) : (Z m).re < 0 ∧ (Z m).im > 0 ↔ -1 < m ∧ m < 3 :=
by sorry

end purely_imaginary_iff_real_iff_second_quadrant_iff_l128_128701


namespace dogs_left_l128_128052

-- Define the conditions
def total_dogs : ℕ := 50
def dog_houses : ℕ := 17

-- Statement to prove the number of dogs left
theorem dogs_left : (total_dogs % dog_houses) = 16 :=
by sorry

end dogs_left_l128_128052


namespace roots_quadratic_l128_128423

theorem roots_quadratic (m x₁ x₂ : ℝ) (h : m < 0) (h₁ : x₁ < x₂) (hx : ∀ x, (x^2 - x - 6 = m) ↔ (x = x₁ ∨ x = x₂)) : 
  -2 < x₁ ∧ x₁ < x₂ ∧ x₂ < 3 :=
by {
  sorry
}

end roots_quadratic_l128_128423


namespace cos_135_degree_l128_128979

theorem cos_135_degree : 
  let Q := (Complex.exp (135 * Real.pi / 180 * Complex.I)) in 
  Complex.re Q = -Real.sqrt 2 / 2 :=
by sorry

end cos_135_degree_l128_128979


namespace num_ways_seating_l128_128725

theorem num_ways_seating (n : ℕ) (h : n = 6) : (nat.factorial n) / n = nat.factorial (n - 1) :=
by 
  rw h
  calc
    (nat.factorial 6) / 6 = 720 / 6    : by norm_num
                      ... = 120        : by norm_num
                      ... = nat.factorial 5 : by norm_num

end num_ways_seating_l128_128725


namespace probability_of_5_out_of_7_odd_rolls_l128_128905

theorem probability_of_5_out_of_7_odd_rolls :
  let prob_odd := (1 / 2 : ℝ)
  let prob_even := (1 / 2 : ℝ)
  let success_count := 5
  let trial_count := 7
  let combination := nat.choose trial_count success_count
  let prob := combination * prob_odd ^ success_count * prob_even ^ (trial_count - success_count)
  in prob = 21 / 128 := by
  sorry

end probability_of_5_out_of_7_odd_rolls_l128_128905


namespace correct_equation_for_growth_rate_l128_128530

def initial_price : ℝ := 6.2
def final_price : ℝ := 8.9
def growth_rate (x : ℝ) : ℝ := initial_price * (1 + x) ^ 2

theorem correct_equation_for_growth_rate (x : ℝ) : growth_rate x = final_price ↔ initial_price * (1 + x) ^ 2 = 8.9 :=
by sorry

end correct_equation_for_growth_rate_l128_128530


namespace sin_of_30_degrees_l128_128340

-- Define P and D and their relationships on a unit circle
def P : ℝ × ℝ := (cos (π / 6), sin (π / 6))  -- Point P on the unit circle at 30 degrees
def D : ℝ × ℝ := (P.1, 0)  -- Foot of the altitude to the x-axis

theorem sin_of_30_degrees : sin (π / 6) = 1 / 2 :=
by 
  -- Use the fact that the triangle P,O,D is a 30-60-90 triangle
  -- and the unit circle properties.
  sorry

end sin_of_30_degrees_l128_128340


namespace red_marbles_more_than_yellow_l128_128213

-- Define the conditions
def total_marbles : ℕ := 19
def yellow_marbles : ℕ := 5
def ratio_parts_blue : ℕ := 3
def ratio_parts_red : ℕ := 4

-- Prove the number of more red marbles than yellow marbles is equal to 3
theorem red_marbles_more_than_yellow :
  let remainder_marbles := total_marbles - yellow_marbles
  let total_parts := ratio_parts_blue + ratio_parts_red
  let marbles_per_part := remainder_marbles / total_parts
  let blue_marbles := ratio_parts_blue * marbles_per_part
  let red_marbles := ratio_parts_red * marbles_per_part
  red_marbles - yellow_marbles = 3 := by
  sorry

end red_marbles_more_than_yellow_l128_128213


namespace original_quantity_ghee_mixture_is_correct_l128_128444

-- Define the variables
def percentage_ghee (x : ℝ) := 0.55 * x
def percentage_vanasapati (x : ℝ) := 0.35 * x
def percentage_palm_oil (x : ℝ) := 0.10 * x
def new_mixture_weight (x : ℝ) := x + 20
def final_vanasapati_percentage (x : ℝ) := 0.30 * (new_mixture_weight x)

-- State the theorem
theorem original_quantity_ghee_mixture_is_correct (x : ℝ) 
  (h1 : percentage_ghee x = 0.55 * x)
  (h2 : percentage_vanasapati x = 0.35 * x)
  (h3 : percentage_palm_oil x = 0.10 * x)
  (h4 : percentage_vanasapati x = final_vanasapati_percentage x) :
  x = 120 := 
sorry

end original_quantity_ghee_mixture_is_correct_l128_128444


namespace percentage_w_less_x_l128_128888

theorem percentage_w_less_x 
    (z : ℝ) 
    (y : ℝ) 
    (x : ℝ) 
    (w : ℝ) 
    (hy : y = 1.20 * z)
    (hx : x = 1.20 * y)
    (hw : w = 1.152 * z) 
    : (x - w) / x * 100 = 20 :=
by
  sorry

end percentage_w_less_x_l128_128888


namespace sin_30_eq_half_l128_128299

theorem sin_30_eq_half : Real.sin (π / 6) = 1 / 2 := 
  sorry

end sin_30_eq_half_l128_128299


namespace license_plate_combinations_l128_128114

-- Definition for the conditions of the problem
def num_license_plate_combinations : ℕ :=
  let num_letters := 26
  let num_digits := 10
  let choose_two_distinct_letters := (num_letters * (num_letters - 1)) / 2
  let arrange_pairs := 2
  let choose_positions := 6
  let digit_permutations := num_digits ^ 2
  choose_two_distinct_letters * arrange_pairs * choose_positions * digit_permutations

-- The theorem we are proving
theorem license_plate_combinations :
  num_license_plate_combinations = 390000 :=
by
  -- The proof would be provided here.
  sorry

end license_plate_combinations_l128_128114


namespace minimum_value_of_2x_3y_l128_128420

noncomputable def minimum_value (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : (2/x) + (3/y) = 1) : ℝ :=
  2*x + 3*y

theorem minimum_value_of_2x_3y {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hxy : (2/x) + (3/y) = 1) : minimum_value x y hx hy hxy = 25 :=
sorry

end minimum_value_of_2x_3y_l128_128420


namespace remainder_8_pow_215_mod_9_l128_128624

theorem remainder_8_pow_215_mod_9 : (8 ^ 215) % 9 = 8 := by
  -- condition
  have pattern : ∀ n, (8 ^ (2 * n + 1)) % 9 = 8 := by sorry
  -- final proof
  exact pattern 107

end remainder_8_pow_215_mod_9_l128_128624


namespace find_x_eq_nine_fourths_l128_128386

theorem find_x_eq_nine_fourths (x n : ℚ) (β : ℚ) (h1 : x = n + β) (h2 : n = ⌊x⌋) (h3 : ⌊x⌋ + x = 17 / 4) : x = 9 / 4 :=
by
  sorry

end find_x_eq_nine_fourths_l128_128386


namespace red_box_position_l128_128772

theorem red_box_position (n : ℕ) (pos_smallest_to_largest : ℕ) (pos_largest_to_smallest : ℕ) 
  (h1 : n = 45) 
  (h2 : pos_smallest_to_largest = 29) 
  (h3 : pos_largest_to_smallest = n - (pos_smallest_to_largest - 1)) :
  pos_largest_to_smallest = 17 := 
  by
    -- This proof is missing; implementation goes here
    sorry

end red_box_position_l128_128772


namespace area_circle_minus_square_l128_128241

theorem area_circle_minus_square {r : ℝ} (h : r = 1/2) : 
  (π * r^2) - (1^2) = (π / 4) - 1 :=
by
  rw [h]
  sorry

end area_circle_minus_square_l128_128241


namespace find_aa_l128_128086

-- Given conditions
def m : ℕ := 7

-- Definition for checking if a number's tens place is 1
def tens_place_one (n : ℕ) : Prop :=
  (n / 10) % 10 = 1

-- The main statement to prove
theorem find_aa : ∃ x : ℕ, x < 10 ∧ tens_place_one (m * x^3) ∧ x = 6 := by
  -- Proof would go here
  sorry

end find_aa_l128_128086


namespace proof_cos_135_degree_l128_128994

noncomputable def cos_135_degree_is_negative_sqrt2_over_2 : Prop :=
  real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2

theorem proof_cos_135_degree : cos_135_degree_is_negative_sqrt2_over_2 :=
by
  sorry

end proof_cos_135_degree_l128_128994


namespace probability_divisor_twenty_l128_128791

/-
Given a circular spinner divided into 20 equal sections and an arrow that is spun once,
prove that the probability that the arrow stops in a section containing a number
that is a divisor of 20 is \(\frac{3}{10}\).
-/

theorem probability_divisor_twenty : 
  let total_sections : ℕ := 20 in
  let divisors_of_20 := {1, 2, 4, 5, 10, 20}.toFinset in
  (divisors_of_20.card : ℚ) / total_sections = 3 / 10 :=
by
  sorry

end probability_divisor_twenty_l128_128791


namespace sin_30_eq_half_l128_128261

theorem sin_30_eq_half : Real.sin (30 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_eq_half_l128_128261


namespace find_divisor_of_x_l128_128488

theorem find_divisor_of_x (x : ℕ) (q p : ℕ) (h1 : x % n = 5) (h2 : 4 * x % n = 2) : n = 9 :=
by
  sorry

end find_divisor_of_x_l128_128488


namespace right_angled_triangle_l128_128490

theorem right_angled_triangle (a b c : ℕ) (h₀ : a = 7) (h₁ : b = 9) (h₂ : c = 13) :
  a^2 + b^2 ≠ c^2 :=
by
  sorry

end right_angled_triangle_l128_128490


namespace jacob_nathan_total_letters_l128_128451

/-- Jacob and Nathan's combined writing output in 10 hours. -/
theorem jacob_nathan_total_letters (jacob_speed nathan_speed : ℕ) (h1 : jacob_speed = 2 * nathan_speed) (h2 : nathan_speed = 25) : jacob_speed + nathan_speed = 75 → (jacob_speed + nathan_speed) * 10 = 750 :=
by
  intros h3
  rw [h1, h2] at h3
  simp at h3
  rw [h3]
  norm_num

end jacob_nathan_total_letters_l128_128451


namespace bob_distance_when_they_meet_l128_128918

-- Define the conditions
def distance_XY : ℝ := 10
def yolanda_rate : ℝ := 3
def bob_rate : ℝ := 4
def yolanda_start_time : ℝ := 0
def bob_start_time : ℝ := 1

-- The statement we want to prove
theorem bob_distance_when_they_meet : 
  ∃ t : ℝ, (yolanda_rate * (t + 1) + bob_rate * t = distance_XY) ∧ (bob_rate * t = 4) :=
sorry

end bob_distance_when_they_meet_l128_128918


namespace smallest_solution_to_equation_l128_128667

theorem smallest_solution_to_equation :
  ∀ x : ℝ, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) → x = 4 - real.sqrt 2 :=
  sorry

end smallest_solution_to_equation_l128_128667


namespace equal_charges_at_x_l128_128105

theorem equal_charges_at_x (x : ℝ) : 
  (2.75 * x + 125 = 1.50 * x + 140) → (x = 12) := 
by
  sorry

end equal_charges_at_x_l128_128105


namespace eliminate_alpha_l128_128815

theorem eliminate_alpha (α x y : ℝ) (h1 : x = Real.tan α ^ 2) (h2 : y = Real.sin α ^ 2) : 
  x - y = x * y := 
by
  sorry

end eliminate_alpha_l128_128815


namespace sin_30_eq_half_l128_128255

theorem sin_30_eq_half :
  let Q := { (x, y) : ℝ × ℝ // x^2 + y^2 = 1 ∧ y ≠ 0 ∧ 0 ≤ x ∧ x ≤ 1 }
  let OE := (1, 0)
  ∀ (Q : (x, y) : ℝ × ℝ), (Q.1, Q.2) = (√3/2, 1/2) →
  ∃ OE : (x, y) : ℝ × ℝ,
    let OQ := (0, 0)
    let QOE_triangle := { c : ℝ × ℝ // ∃ Q OE OQ, QOE c }
      sin 30° = 1/2 :=
  by
    sorry

end sin_30_eq_half_l128_128255


namespace smallest_x_correct_l128_128656

noncomputable def smallest_x (K : ℤ) : ℤ := 135000

theorem smallest_x_correct (K : ℤ) :
  (∃ x : ℤ, 180 * x = K ^ 5 ∧ x > 0) → smallest_x K = 135000 :=
by
  sorry

end smallest_x_correct_l128_128656


namespace sin_30_eq_half_l128_128289

theorem sin_30_eq_half : Real.sin (Float.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l128_128289


namespace red_marbles_more_than_yellow_l128_128212

-- Define the conditions
def total_marbles : ℕ := 19
def yellow_marbles : ℕ := 5
def ratio_parts_blue : ℕ := 3
def ratio_parts_red : ℕ := 4

-- Prove the number of more red marbles than yellow marbles is equal to 3
theorem red_marbles_more_than_yellow :
  let remainder_marbles := total_marbles - yellow_marbles
  let total_parts := ratio_parts_blue + ratio_parts_red
  let marbles_per_part := remainder_marbles / total_parts
  let blue_marbles := ratio_parts_blue * marbles_per_part
  let red_marbles := ratio_parts_red * marbles_per_part
  red_marbles - yellow_marbles = 3 := by
  sorry

end red_marbles_more_than_yellow_l128_128212


namespace sin_30_eq_half_l128_128266

theorem sin_30_eq_half : Real.sin (30 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_eq_half_l128_128266


namespace brass_players_count_l128_128201

def marching_band_size : ℕ := 110
def woodwinds (b : ℕ) : ℕ := 2 * b
def percussion (w : ℕ) : ℕ := 4 * w
def total_members (b : ℕ) : ℕ := b + woodwinds b + percussion (woodwinds b)

theorem brass_players_count : ∃ b : ℕ, total_members b = marching_band_size ∧ b = 10 :=
by
  sorry

end brass_players_count_l128_128201


namespace total_difference_in_cents_l128_128811

variable (q : ℕ)

def charles_quarters := 6 * q + 2
def charles_dimes := 3 * q - 2

def richard_quarters := 2 * q + 10
def richard_dimes := 4 * q + 3

def cents_from_quarters (n : ℕ) : ℕ := 25 * n
def cents_from_dimes (n : ℕ) : ℕ := 10 * n

theorem total_difference_in_cents : 
  (cents_from_quarters (charles_quarters q) + cents_from_dimes (charles_dimes q)) - 
  (cents_from_quarters (richard_quarters q) + cents_from_dimes (richard_dimes q)) = 
  90 * q - 250 :=
by
  sorry

end total_difference_in_cents_l128_128811


namespace dice_probability_l128_128464

noncomputable def probability_event (event_count : ℕ) (total_count : ℕ) : ℚ := 
  event_count / total_count

theorem dice_probability :
  let event_first_die := 3
  let event_second_die := 3
  let total_outcomes_first := 8
  let total_outcomes_second := 8
  probability_event event_first_die total_outcomes_first * probability_event event_second_die total_outcomes_second = 9 / 64 :=
by
  sorry

end dice_probability_l128_128464


namespace percent_of_g_is_a_l128_128193

-- Definitions of the seven consecutive numbers
def consecutive_7_avg_9 (a b c d e f g : ℝ) : Prop :=
  a + b + c + d + e + f + g = 63

def is_median (d : ℝ) : Prop :=
  d = 9

def express_numbers (a b c d e f g : ℝ) : Prop :=
  a = d - 3 ∧ b = d - 2 ∧ c = d - 1 ∧ d = d ∧ e = d + 1 ∧ f = d + 2 ∧ g = d + 3

-- Main statement asserting the percentage relationship
theorem percent_of_g_is_a (a b c d e f g : ℝ) (h_avg : consecutive_7_avg_9 a b c d e f g)
  (h_median : is_median d) (h_express : express_numbers a b c d e f g) :
  (a / g) * 100 = 50 := by
  sorry

end percent_of_g_is_a_l128_128193


namespace circle_center_and_radius_locus_of_midpoint_l128_128853

-- Part 1: Prove the equation of the circle C:
theorem circle_center_and_radius (a b r: ℝ) (hc: a + b = 2):
  (4 - a)^2 + b^2 = r^2 →
  (2 - a)^2 + (2 - b)^2 = r^2 →
  a = 2 ∧ b = 0 ∧ r = 2 := by
  sorry

-- Part 2: Prove the locus of the midpoint M:
theorem locus_of_midpoint (x y : ℝ) :
  ∃ (x1 y1 : ℝ), (x1 - 2)^2 + y1^2 = 4 ∧ x = (x1 + 5) / 2 ∧ y = y1 / 2 →
  x^2 - 7*x + y^2 + 45/4 = 0 := by
  sorry

end circle_center_and_radius_locus_of_midpoint_l128_128853


namespace response_percentage_is_50_l128_128737

-- Define the initial number of friends
def initial_friends := 100

-- Define the number of friends Mark kept initially
def kept_friends := 40

-- Define the number of friends Mark contacted
def contacted_friends := initial_friends - kept_friends

-- Define the number of friends Mark has after some responded
def remaining_friends := 70

-- Define the number of friends who responded to Mark's contact
def responded_friends := remaining_friends - kept_friends

-- Define the percentage of contacted friends who responded
def response_percentage := (responded_friends / contacted_friends) * 100

theorem response_percentage_is_50 :
  response_percentage = 50 := by
  sorry

end response_percentage_is_50_l128_128737


namespace smallest_solution_exists_l128_128692

noncomputable def is_solution (x : ℝ) : Prop := (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧ x ≠ 3 ∧ x ≠ 5 ∧ x ≠ 4

-- Statement of the problem without proof
theorem smallest_solution_exists : ∃ (x : ℝ), is_solution x ∧ ∀ (y : ℝ), is_solution y → x ≤ y :=
sorry

end smallest_solution_exists_l128_128692


namespace trig_identity_proof_l128_128943

theorem trig_identity_proof :
  2 * (1 / 2) + (Real.sqrt 3 / 2) * Real.sqrt 3 = 5 / 2 :=
by
  sorry

end trig_identity_proof_l128_128943


namespace ratio_of_projection_l128_128653

theorem ratio_of_projection (x y : ℝ)
  (h : ∀ (x y : ℝ), (∃ x y : ℝ, 
  (3/25 * x + 4/25 * y = x) ∧ (4/25 * x + 12/25 * y = y))) : x / y = 2 / 11 :=
sorry

end ratio_of_projection_l128_128653


namespace volunteers_meet_again_in_360_days_l128_128380

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

end volunteers_meet_again_in_360_days_l128_128380


namespace neznaika_discrepancy_l128_128778

-- Definitions of the given conditions
def correct_kilograms_to_kilolunas (kg : ℝ) : ℝ := kg / 0.24
def neznaika_kilograms_to_kilolunas (kg : ℝ) : ℝ := (kg * 4) * 1.04

-- Define the precise value of 1 kiloluna in kilograms for clarity
def one_kiloluna_in_kg : ℝ := (1 / 4) * 0.96

-- Function calculating the discrepancy percentage
def discrepancy_percentage (kg : ℝ) : ℝ :=
  let correct_value := correct_kilograms_to_kilolunas kg
  let neznaika_value := neznaika_kilograms_to_kilolunas kg
  ((correct_value - neznaika_value).abs / correct_value) * 100

-- Statement of the theorem to be proven
theorem neznaika_discrepancy :
  discrepancy_percentage 1 = 0.16 := sorry

end neznaika_discrepancy_l128_128778


namespace intersection_points_relation_l128_128545

-- Suppressing noncomputable theory to focus on the structure
-- of the Lean statement rather than computability aspects.

noncomputable def intersection_points (k : ℕ) : ℕ :=
sorry -- This represents the function f(k)

axiom no_parallel (k : ℕ) : Prop
axiom no_three_intersect (k : ℕ) : Prop

theorem intersection_points_relation (k : ℕ) (h1 : no_parallel k) (h2 : no_three_intersect k) :
  intersection_points (k + 1) = intersection_points k + k :=
sorry

end intersection_points_relation_l128_128545


namespace sin_30_eq_half_l128_128256

theorem sin_30_eq_half :
  let Q := { (x, y) : ℝ × ℝ // x^2 + y^2 = 1 ∧ y ≠ 0 ∧ 0 ≤ x ∧ x ≤ 1 }
  let OE := (1, 0)
  ∀ (Q : (x, y) : ℝ × ℝ), (Q.1, Q.2) = (√3/2, 1/2) →
  ∃ OE : (x, y) : ℝ × ℝ,
    let OQ := (0, 0)
    let QOE_triangle := { c : ℝ × ℝ // ∃ Q OE OQ, QOE c }
      sin 30° = 1/2 :=
  by
    sorry

end sin_30_eq_half_l128_128256


namespace correct_choice_is_B_l128_128110

def draw_ray := "Draw ray OP=3cm"
def connect_points := "Connect points A and B"
def draw_midpoint := "Draw the midpoint of points A and B"
def draw_distance := "Draw the distance between points A and B"

-- Mathematical function to identify the correct statement about drawing
def correct_drawing_statement (s : String) : Prop :=
  s = connect_points

theorem correct_choice_is_B :
  correct_drawing_statement connect_points :=
by
  sorry

end correct_choice_is_B_l128_128110


namespace jesse_max_correct_answers_l128_128849

theorem jesse_max_correct_answers :
  ∃ a b c : ℕ, a + b + c = 60 ∧ 5 * a - 2 * c = 150 ∧ a ≤ 38 :=
sorry

end jesse_max_correct_answers_l128_128849


namespace suzannes_book_pages_l128_128471

-- Conditions
def pages_read_on_monday : ℕ := 15
def pages_read_on_tuesday : ℕ := 31
def pages_left : ℕ := 18

-- Total number of pages in the book
def total_pages : ℕ := pages_read_on_monday + pages_read_on_tuesday + pages_left

-- Problem statement
theorem suzannes_book_pages : total_pages = 64 :=
by
  -- Proof is not required, only the statement
  sorry

end suzannes_book_pages_l128_128471


namespace a_range_l128_128832

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (a - 3) * x + 4 * a

theorem a_range (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) < 0) ↔ (0 < a ∧ a ≤ 1 / 4) :=
by
  sorry

end a_range_l128_128832


namespace proof_cos_135_degree_l128_128998

noncomputable def cos_135_degree_is_negative_sqrt2_over_2 : Prop :=
  real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2

theorem proof_cos_135_degree : cos_135_degree_is_negative_sqrt2_over_2 :=
by
  sorry

end proof_cos_135_degree_l128_128998


namespace thirteen_in_base_two_l128_128383

theorem thirteen_in_base_two : nat.to_binary 13 = "1101" :=
sorry

end thirteen_in_base_two_l128_128383


namespace cos_135_eq_neg_sqrt2_div_2_l128_128961

theorem cos_135_eq_neg_sqrt2_div_2 : 
  Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l128_128961


namespace vectors_not_coplanar_l128_128938

def a : ℝ × ℝ × ℝ := (4, 1, 1)
def b : ℝ × ℝ × ℝ := (-9, -4, -9)
def c : ℝ × ℝ × ℝ := (6, 2, 6)

def scalarTripleProduct (u v w : ℝ × ℝ × ℝ) : ℝ :=
  let (u1, u2, u3) := u
  let (v1, v2, v3) := v
  let (w1, w2, w3) := w
  u1 * (v2 * w3 - v3 * w2) - u2 * (v1 * w3 - v3 * w1) + u3 * (v1 * w2 - v2 * w1)

theorem vectors_not_coplanar : scalarTripleProduct a b c = -18 := by
  sorry

end vectors_not_coplanar_l128_128938


namespace max_marks_l128_128512

theorem max_marks (M : ℝ) (h_pass : 0.33 * M = 165) : M = 500 := 
by
  sorry

end max_marks_l128_128512


namespace sin_30_eq_one_half_l128_128370

def Q_coords (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

theorem sin_30_eq_one_half :
  let Q := Q_coords (Real.pi / 6)
  Q.2 = 1 / 2 :=
by
  sorry

end sin_30_eq_one_half_l128_128370


namespace find_x_l128_128150

noncomputable def leastCommonMultiple (a b : ℕ) : ℕ :=
  a * b / (Nat.gcd a b)

noncomputable def lcm_of_10_to_15 : ℕ :=
  leastCommonMultiple 10 (leastCommonMultiple 11 (leastCommonMultiple 12 (leastCommonMultiple 13 (leastCommonMultiple 14 15))))

theorem find_x :
  (lcm_of_10_to_15 / 2310 = 26) := by
  sorry

end find_x_l128_128150


namespace infinite_primes_of_form_l128_128457

theorem infinite_primes_of_form (p : ℕ) (hp : Nat.Prime p) (hpodd : p % 2 = 1) :
  ∃ᶠ n in at_top, Nat.Prime (2 * p * n + 1) :=
sorry

end infinite_primes_of_form_l128_128457


namespace sin_30_deg_l128_128306

theorem sin_30_deg : real.sin (30 * real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_deg_l128_128306


namespace complement_of_intersection_l128_128733

def U : Set ℕ := {0, 1, 2, 3}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {1, 2, 3}
def intersection : Set ℕ := M ∩ N
def complement : Set ℕ := U \ intersection

theorem complement_of_intersection (U M N : Set ℕ) :
  U = {0, 1, 2, 3} →
  M = {0, 1, 2} →
  N = {1, 2, 3} →
  (U \ (M ∩ N)) = {0, 3} := by
  intro hU hM hN
  simp [hU, hM, hN]
  sorry

end complement_of_intersection_l128_128733


namespace cal_fraction_of_anthony_l128_128013

theorem cal_fraction_of_anthony (mabel_transactions anthony_transactions cal_transactions jade_transactions : ℕ)
  (h_mabel : mabel_transactions = 90)
  (h_anthony : anthony_transactions = mabel_transactions + mabel_transactions / 10)
  (h_jade : jade_transactions = 82)
  (h_jade_cal : jade_transactions = cal_transactions + 16) :
  (cal_transactions : ℚ) / (anthony_transactions : ℚ) = 2 / 3 :=
by
  -- The proof would be here, but it is omitted as per the requirement.
  sorry

end cal_fraction_of_anthony_l128_128013


namespace cos_135_eq_neg_sqrt2_div_2_l128_128966

theorem cos_135_eq_neg_sqrt2_div_2 : 
  Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l128_128966


namespace same_cost_for_same_sheets_l128_128103

def John's_Photo_World_cost (x : ℕ) : ℝ := 2.75 * x + 125
def Sam's_Picture_Emporium_cost (x : ℕ) : ℝ := 1.50 * x + 140

theorem same_cost_for_same_sheets :
  ∃ (x : ℕ), John's_Photo_World_cost x = Sam's_Picture_Emporium_cost x ∧ x = 12 :=
by
  sorry

end same_cost_for_same_sheets_l128_128103


namespace fractional_equation_positive_root_l128_128152

theorem fractional_equation_positive_root (a : ℝ) (ha : ∃ x : ℝ, x > 0 ∧ (6 / (x - 2) - 1 = a * x / (2 - x))) : a = -3 :=
by
  sorry

end fractional_equation_positive_root_l128_128152


namespace problem_statement_l128_128092

-- Definitions
def MagnitudeEqual : Prop := (2.4 : ℝ) = (2.40 : ℝ)
def CountUnit2_4 : Prop := (0.1 : ℝ) = 2.4 / 24
def CountUnit2_40 : Prop := (0.01 : ℝ) = 2.40 / 240

-- Theorem statement
theorem problem_statement : MagnitudeEqual ∧ CountUnit2_4 ∧ CountUnit2_40 → True := by
  intros
  sorry

end problem_statement_l128_128092


namespace prob_Allison_wins_l128_128247

open ProbabilityTheory

def outcome_space (n : ℕ) := {k : ℕ // k < n}

def Brian_roll : outcome_space 6 := sorry -- Representing Brian's possible outcomes: 1, 2, 3, 4, 5, 6
def Noah_roll : outcome_space 2 := sorry -- Representing Noah's possible outcomes: 3 or 5 (3 is 0 and 5 is 1 in the index for simplicity)

noncomputable def prob_Brian_less_than_4 := (|{k : ℕ // k < 3}| : ℝ) / (|{k : ℕ // k < 6}|) -- Probability Brian rolls 1, 2, or 3
noncomputable def prob_Noah_less_than_4 := (|{k : ℕ // k = 0}| : ℝ) / (|{k : ℕ // k < 2}|)  -- Probability Noah rolls 3 (index 0)

theorem prob_Allison_wins : (prob_Brian_less_than_4 * prob_Noah_less_than_4) = 1 / 4 := by
  sorry

end prob_Allison_wins_l128_128247


namespace cos_135_eq_neg_sqrt2_div_2_l128_128989

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = - Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l128_128989


namespace sum_of_three_numbers_l128_128617

theorem sum_of_three_numbers (S F T : ℕ) (h1 : S = 150) (h2 : F = 2 * S) (h3 : T = F / 3) :
  F + S + T = 550 :=
by
  sorry

end sum_of_three_numbers_l128_128617


namespace range_of_x_l128_128132

theorem range_of_x (a : ℝ) (h : -1 ≤ a ∧ a ≤ 1) (x : ℝ) : 
  (x ^ 2 + (a - 4) * x + 4 - 2 * a > 0) ↔ (x < 1 ∨ x > 3) :=
by 
  sorry

end range_of_x_l128_128132


namespace fraction_is_one_third_l128_128087

noncomputable def fraction_studying_japanese (J S : ℕ) (h1 : S = 2 * J) (h2 : 3 / 8 * S + 1 / 4 * J = J) : ℚ :=
  J / (J + S)

theorem fraction_is_one_third (J S : ℕ) (h1 : S = 2 * J) (h2 : 3 / 8 * S + 1 / 4 * J = J) : 
  fraction_studying_japanese J S h1 h2 = 1 / 3 :=
  sorry

end fraction_is_one_third_l128_128087


namespace judy_expense_correct_l128_128381

noncomputable def judy_expense : ℝ :=
  let carrots := 5 * 1
  let milk := 3 * 3
  let pineapples := 2 * 4
  let original_flour_price := 5
  let discount := original_flour_price * 0.25
  let discounted_flour_price := original_flour_price - discount
  let flour := 2 * discounted_flour_price
  let ice_cream := 7
  let total_no_coupon := carrots + milk + pineapples + flour + ice_cream
  if total_no_coupon >= 30 then total_no_coupon - 10 else total_no_coupon

theorem judy_expense_correct : judy_expense = 26.5 := by
  sorry

end judy_expense_correct_l128_128381


namespace proof_cos_135_degree_l128_128995

noncomputable def cos_135_degree_is_negative_sqrt2_over_2 : Prop :=
  real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2

theorem proof_cos_135_degree : cos_135_degree_is_negative_sqrt2_over_2 :=
by
  sorry

end proof_cos_135_degree_l128_128995


namespace caitlins_team_number_l128_128484

noncomputable def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the two-digit prime numbers
def two_digit_prime (n : ℕ) : Prop := 10 ≤ n ∧ n < 100 ∧ is_prime n

-- Lean statement
theorem caitlins_team_number (h_date birthday_before today birthday_after : ℕ)
  (p₁ p₂ p₃ : ℕ)
  (h1 : two_digit_prime p₁)
  (h2 : two_digit_prime p₂)
  (h3 : two_digit_prime p₃)
  (h4 : p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃)
  (h5 : p₁ + p₂ = today ∨ p₁ + p₃ = today ∨ p₂ + p₃ = today)
  (h6 : (p₁ + p₂ = birthday_before ∨ p₁ + p₃ = birthday_before ∨ p₂ + p₃ = birthday_before)
       ∧ birthday_before < today)
  (h7 : (p₁ + p₂ = birthday_after ∨ p₁ + p₃ = birthday_after ∨ p₂ + p₃ = birthday_after)
       ∧ birthday_after > today) :
  p₃ = 11 := by
  sorry

end caitlins_team_number_l128_128484


namespace tangent_line_through_point_and_circle_l128_128125

noncomputable def tangent_line_equation : String :=
  "y - 1 = 0"

theorem tangent_line_through_point_and_circle :
  ∀ (line_eq: String), 
  (∀ (x y: ℝ), (x - 1) ^ 2 + y ^ 2 = 1 ∧ (x, y) = (1, 1) → y - 1 = 0) →
  line_eq = tangent_line_equation :=
by
  intro line_eq h
  sorry

end tangent_line_through_point_and_circle_l128_128125


namespace brass_players_10_l128_128199

theorem brass_players_10:
  ∀ (brass woodwind percussion : ℕ),
    brass + woodwind + percussion = 110 →
    percussion = 4 * woodwind →
    woodwind = 2 * brass →
    brass = 10 :=
by
  intros brass woodwind percussion h1 h2 h3
  sorry

end brass_players_10_l128_128199


namespace sin_30_eq_half_l128_128253

theorem sin_30_eq_half :
  let Q := { (x, y) : ℝ × ℝ // x^2 + y^2 = 1 ∧ y ≠ 0 ∧ 0 ≤ x ∧ x ≤ 1 }
  let OE := (1, 0)
  ∀ (Q : (x, y) : ℝ × ℝ), (Q.1, Q.2) = (√3/2, 1/2) →
  ∃ OE : (x, y) : ℝ × ℝ,
    let OQ := (0, 0)
    let QOE_triangle := { c : ℝ × ℝ // ∃ Q OE OQ, QOE c }
      sin 30° = 1/2 :=
  by
    sorry

end sin_30_eq_half_l128_128253


namespace algebraic_expression_value_l128_128705

-- Define the conditions
variables (x y : ℝ)
-- Condition 1: x - y = 5
def cond1 : Prop := x - y = 5
-- Condition 2: xy = -3
def cond2 : Prop := x * y = -3

-- Define the statement to be proved
theorem algebraic_expression_value :
  cond1 x y → cond2 x y → x^2 * y - x * y^2 = -15 :=
by
  intros h1 h2
  sorry

end algebraic_expression_value_l128_128705


namespace work_completion_l128_128633

theorem work_completion (p q : ℝ) (h1 : p = 1.60 * q) (h2 : (1 / p + 1 / q) = 1 / 16) : p = 1 / 26 := 
by {
  -- This will be followed by the proof steps, but we add sorry since only the statement is required
  sorry
}

end work_completion_l128_128633


namespace values_of_x_defined_l128_128657

noncomputable def problem_statement (x : ℝ) : Prop :=
  (2 * x - 3 > 0) ∧ (5 - 2 * x > 0)

theorem values_of_x_defined (x : ℝ) :
  problem_statement x ↔ (3 / 2 < x ∧ x < 5 / 2) :=
by sorry

end values_of_x_defined_l128_128657


namespace kyle_origami_stars_l128_128580

/-- Kyle bought 2 glass bottles, each can hold 15 origami stars,
    then bought another 3 identical glass bottles.
    Prove that the total number of origami stars needed to fill them is 75. -/
theorem kyle_origami_stars : (2 * 15) + (3 * 15) = 75 := by
  sorry

end kyle_origami_stars_l128_128580


namespace sin_30_eq_half_l128_128270
open real

noncomputable def sin_30_deg : ℝ :=
  sin (π / 6)

theorem sin_30_eq_half : sin_30_deg = 1 / 2 :=
by
  -- Given the conditions of the problem, we need to prove that sin(π / 6) = 1 / 2
  sorry

end sin_30_eq_half_l128_128270


namespace min_points_dodecahedron_min_points_icosahedron_l128_128913

-- Definitions for the dodecahedron
def dodecahedron_faces : ℕ := 12
def vertices_per_face_dodecahedron : ℕ := 3

-- Prove the minimum number of points to mark each face of a dodecahedron
theorem min_points_dodecahedron (n : ℕ) (h : 3 * n >= dodecahedron_faces) : n >= 4 :=
sorry

-- Definitions for the icosahedron
def icosahedron_faces : ℕ := 20
def icosahedron_vertices : ℕ := 12

-- Prove the minimum number of points to mark each face of an icosahedron
theorem min_points_icosahedron (n : ℕ) (h : n >= 6) : n = 6 :=
sorry

end min_points_dodecahedron_min_points_icosahedron_l128_128913


namespace distance_between_cars_l128_128620

-- Definitions representing the initial conditions and distances traveled by the cars
def initial_distance : ℕ := 113
def first_car_distance_on_road : ℕ := 50
def second_car_distance_on_road : ℕ := 35

-- Statement of the theorem to be proved
theorem distance_between_cars : initial_distance - (first_car_distance_on_road + second_car_distance_on_road) = 28 :=
by
  sorry

end distance_between_cars_l128_128620


namespace contrapositive_proof_l128_128026

theorem contrapositive_proof (a : ℝ) (h : a ≤ 2 → a^2 ≤ 4) : a > 2 → a^2 > 4 :=
by
  intros ha
  sorry

end contrapositive_proof_l128_128026


namespace slope_range_l128_128529

noncomputable def directed_distance (a b c x0 y0 : ℝ) : ℝ :=
  (a * x0 + b * y0 + c) / (Real.sqrt (a^2 + b^2))

theorem slope_range {A B P : ℝ × ℝ} (hA : A = (-1, 0)) (hB : B = (1, 0)) (hP : P = (3, 0))
                   {C : ℝ × ℝ} (hC : ∃ θ : ℝ, C = (9 * Real.cos θ, 18 + 9 * Real.sin θ))
                   {a b c : ℝ} (h_line : c = -3 * a)
                   (h_sum_distances : directed_distance a b c (-1) 0 +
                                      directed_distance a b c 1 0 +
                                      directed_distance a b c (9 * Real.cos θ) (18 + 9 * Real.sin θ) = 0) :
  -3 ≤ - (a / b) ∧ - (a / b) ≤ -1 := sorry

end slope_range_l128_128529


namespace value_of_a2012_l128_128045

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 0 ∧ ∀ n, a (n + 1) - a n = 2 * n

theorem value_of_a2012 (a : ℕ → ℤ) (h : seq a) : a 2012 = 2012 * 2011 :=
by 
  sorry

end value_of_a2012_l128_128045


namespace domain_is_correct_l128_128751

def domain_of_function (x : ℝ) : Prop :=
  (3 - x ≥ 0) ∧ (x + 1 ≠ 0) ∧ (x + 2 > 0)

theorem domain_is_correct :
  { x : ℝ | domain_of_function x } = { x : ℝ | -2 < x ∧ x ≤ 3 ∧ x ≠ -1 } :=
by
  sorry

end domain_is_correct_l128_128751


namespace team_selection_l128_128408

theorem team_selection :
  let teachers := 5
  let students := 10
  (teachers * students = 50) :=
by
  sorry

end team_selection_l128_128408


namespace quadratic_roots_bounds_l128_128425

theorem quadratic_roots_bounds (m x1 x2 : ℝ) (h : m < 0)
  (hx : x1 < x2) 
  (hr : ∀ x, x^2 - x - 6 = m → x = x1 ∨ x = x2) :
  -2 < x1 ∧ x2 < 3 :=
by
  sorry

end quadratic_roots_bounds_l128_128425


namespace neznaika_discrepancy_l128_128777

theorem neznaika_discrepancy :
  let KL := 1 -- Assume we start with 1 kiloluna
  let kg := 1 -- Assume we start with 1 kilogram
  let snayka_kg (KL : ℝ) := (KL / 4) * 0.96 -- Conversion rule from kilolunas to kilograms by Snayka
  let neznaika_kl (kg : ℝ) := (kg * 4) * 1.04 -- Conversion rule from kilograms to kilolunas by Neznaika
  let correct_kl (kg : ℝ) := kg / 0.24 -- Correct conversion from kilograms to kilolunas
  
  let result_kl := (neznaika_kl 1) -- Neznaika's computed kilolunas for 1 kilogram
  let correct_kl_val := (correct_kl 1) -- Correct kilolunas for 1 kilogram
  let ratio := result_kl / correct_kl_val -- Ratio of Neznaika's value to Correct value
  let discrepancy := 100 * (1 - ratio) -- Discrepancy percentage

  result_kl = 4.16 ∧ correct_kl_val = 4.1667 ∧ discrepancy = 0.16 := 
by
  sorry

end neznaika_discrepancy_l128_128777


namespace probability_of_odd_numbers_exactly_five_times_in_seven_rolls_l128_128908

theorem probability_of_odd_numbers_exactly_five_times_in_seven_rolls :
  (nat.choose 7 5 * (1/2)^5 * (1/2)^2) = (21 / 128) := by
  sorry

end probability_of_odd_numbers_exactly_five_times_in_seven_rolls_l128_128908


namespace cos_135_eq_neg_sqrt2_div_2_l128_128987

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = - Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l128_128987


namespace unoccupied_seats_in_business_class_l128_128236

/-
Define the numbers for each class and the number of people in each.
-/
def first_class_seats : Nat := 10
def business_class_seats : Nat := 30
def economy_class_seats : Nat := 50
def people_in_first_class : Nat := 3
def people_in_economy_class : Nat := economy_class_seats / 2
def people_in_business_and_first_class : Nat := people_in_economy_class
def people_in_business_class : Nat := people_in_business_and_first_class - people_in_first_class

/-
Prove that the number of unoccupied seats in business class is 8.
-/
theorem unoccupied_seats_in_business_class :
  business_class_seats - people_in_business_class = 8 :=
sorry

end unoccupied_seats_in_business_class_l128_128236


namespace sin_30_eq_half_l128_128259

theorem sin_30_eq_half :
  let Q := { (x, y) : ℝ × ℝ // x^2 + y^2 = 1 ∧ y ≠ 0 ∧ 0 ≤ x ∧ x ≤ 1 }
  let OE := (1, 0)
  ∀ (Q : (x, y) : ℝ × ℝ), (Q.1, Q.2) = (√3/2, 1/2) →
  ∃ OE : (x, y) : ℝ × ℝ,
    let OQ := (0, 0)
    let QOE_triangle := { c : ℝ × ℝ // ∃ Q OE OQ, QOE c }
      sin 30° = 1/2 :=
  by
    sorry

end sin_30_eq_half_l128_128259


namespace triangle_angle_bisector_segment_length_l128_128047

theorem triangle_angle_bisector_segment_length
  (DE DF EF DG EG : ℝ)
  (h_ratio : DE / 12 = 1 ∧ DF / DE = 4 / 3 ∧ EF / DE = 5 / 3)
  (h_angle_bisector : DG / EG = DE / DF ∧ DG + EG = EF) :
  EG = 80 / 7 :=
by
  sorry

end triangle_angle_bisector_segment_length_l128_128047


namespace part1_part2_l128_128427

section

variables {x m : ℝ}

def f (x m : ℝ) : ℝ := 3 * x^2 + (4 - m) * x - 6 * m
def g (x m : ℝ) : ℝ := 2 * x^2 - x - m

theorem part1 (m : ℝ) (h : m = 1) : 
  {x : ℝ | f x m > 0} = {x : ℝ | x < -2 ∨ x > 1} :=
sorry

theorem part2 (m : ℝ) (h : m > 0) : 
  {x : ℝ | f x m ≤ g x m} = {x : ℝ | -5 ≤ x ∧ x ≤ m} :=
sorry
     
end

end part1_part2_l128_128427


namespace maximum_value_of_k_l128_128148

theorem maximum_value_of_k (x y k : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < k)
    (h4 : 4 = k^2 * ((x^2 / y^2) + (y^2 / x^2)) + k * ((x / y) + (y / x))) : k ≤ 1.5 :=
by
  sorry

end maximum_value_of_k_l128_128148


namespace number_of_cats_l128_128562

theorem number_of_cats 
  (n k : ℕ)
  (h1 : n * k = 999919)
  (h2 : k > n) :
  n = 991 :=
sorry

end number_of_cats_l128_128562


namespace smallest_solution_l128_128681

theorem smallest_solution :
  (∃ x : ℝ, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧
  ∀ y : ℝ, (1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4)) → x ≤ y) →
  x = 4 - Real.sqrt 2 :=
sorry

end smallest_solution_l128_128681


namespace sin_thirty_deg_l128_128324

theorem sin_thirty_deg : Real.sin (π / 6) = 1 / 2 := 
by 
-- Assume that the unit circle point corresponding to 30° is (√3/2, 1/2)
-- Sine of an angle in the unit circle is the y-coordinate of the corresponding point
sorry

end sin_thirty_deg_l128_128324


namespace sum_of_positive_factors_of_36_l128_128064

theorem sum_of_positive_factors_of_36 : 
  ∑ d in (Finset.filter (λ d, 36 % d = 0) (Finset.range 37)), d = 91 :=
by
  sorry

end sum_of_positive_factors_of_36_l128_128064


namespace min_value_xyz_l128_128586

theorem min_value_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 8) : 
  x + 2 * y + 4 * z ≥ 12 := sorry

end min_value_xyz_l128_128586


namespace cos_135_eq_neg_sqrt2_div_2_l128_128968

theorem cos_135_eq_neg_sqrt2_div_2 : 
  Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l128_128968


namespace minimum_omega_value_l128_128168

noncomputable def min_omega (ω T x φ : ℝ) (hω : ω > 0) (hφ : 0 < φ ∧ φ < π) (hT : T = 2 * π / ω) 
  (hfT : Real.cos (ω * T + φ) = √3 / 2) (hx : Real.cos (ω * x + φ) = 0) : ℝ :=
  3

theorem minimum_omega_value (ω T x φ : ℝ)
  (hω : ω > 0)
  (hφ : 0 < φ ∧ φ < π)
  (hT : T = 2 * π / ω)
  (hfT : Real.cos (ω * T + φ) = √3 / 2)
  (hx : Real.cos (ω * x + φ) = 0)
  (hx_val : x = π / 9)
  (T_val : T = 2 * π / ω) :
  ω = min_omega ω T (π / 9) (π / 6) :=
begin
  sorry -- Proof steps are skipped as instructed.
end

end minimum_omega_value_l128_128168


namespace length_of_GH_l128_128035

theorem length_of_GH (AB FE CD : ℕ) (side_large side_second side_third side_small : ℕ) 
  (h1 : AB = 11) (h2 : FE = 13) (h3 : CD = 5)
  (h4 : side_large = side_second + AB)
  (h5 : side_second = side_third + CD)
  (h6 : side_third = side_small + FE) :
  GH = 29 :=
by
  -- Proof steps would follow here based on the problem's solution
  -- Using the given conditions and transformations.
  sorry

end length_of_GH_l128_128035


namespace necessary_but_not_sufficient_l128_128828

-- Define α as an interior angle of triangle ABC
def is_interior_angle_of_triangle (α : ℝ) : Prop :=
  0 < α ∧ α < 180

-- Define the sine condition
def sine_condition (α : ℝ) : Prop :=
  Real.sin α = Real.sqrt 2 / 2

-- Define the main theorem
theorem necessary_but_not_sufficient (α : ℝ) (h1 : is_interior_angle_of_triangle α) (h2 : sine_condition α) :
  (sine_condition α) ↔ (α = 45) ∨ (α = 135) := by
  sorry

end necessary_but_not_sufficient_l128_128828


namespace values_of_a_l128_128433

open Set

noncomputable def A : Set ℝ := { x | x^2 - 2*x - 3 = 0 }
noncomputable def B (a : ℝ) : Set ℝ := if a = 0 then ∅ else { x | a * x = 1 }

theorem values_of_a (a : ℝ) : (B a ⊆ A) ↔ (a = -1 ∨ a = 0 ∨ a = 1/3) :=
by 
  sorry

end values_of_a_l128_128433


namespace minimum_omega_formula_l128_128172

noncomputable def minimumOmega (f : ℝ → ℝ) (x T ω φ : ℝ) : ℝ :=
  if (f = λ x, Real.cos (ω * x + φ)) 
     ∧ (ω > 0) 
     ∧ (0 < φ < π)
     ∧ (f T = √3 / 2)
     ∧ (f (π / 9) = 0) 
     ∧ (T = 2 * π / ω) 
  then 3 else 0

theorem minimum_omega_formula {f : ℝ → ℝ} {x T ω φ : ℝ} :
  (f = λ x, Real.cos (ω * x + φ)) 
  ∧ (ω > 0) 
  ∧ (0 < φ < π)
  ∧ (f T = √3 / 2)
  ∧ (f (π / 9) = 0)
  ∧ (T = 2 * π / ω) 
  → minimumOmega f x T ω φ = 3 :=
by
  sorry

end minimum_omega_formula_l128_128172


namespace ratio_of_hardback_books_is_two_to_one_l128_128113

noncomputable def ratio_of_hardback_books : ℕ :=
  let sarah_paperbacks := 6
  let sarah_hardbacks := 4
  let brother_paperbacks := sarah_paperbacks / 3
  let total_books_brother := 10
  let brother_hardbacks := total_books_brother - brother_paperbacks
  brother_hardbacks / sarah_hardbacks

theorem ratio_of_hardback_books_is_two_to_one : 
  ratio_of_hardback_books = 2 :=
by
  sorry

end ratio_of_hardback_books_is_two_to_one_l128_128113


namespace probability_of_5_out_of_7_odd_rolls_l128_128906

theorem probability_of_5_out_of_7_odd_rolls :
  let prob_odd := (1 / 2 : ℝ)
  let prob_even := (1 / 2 : ℝ)
  let success_count := 5
  let trial_count := 7
  let combination := nat.choose trial_count success_count
  let prob := combination * prob_odd ^ success_count * prob_even ^ (trial_count - success_count)
  in prob = 21 / 128 := by
  sorry

end probability_of_5_out_of_7_odd_rolls_l128_128906


namespace smallest_t_circle_sin_l128_128039

theorem smallest_t_circle_sin (t : ℝ) (h0 : 0 ≤ t) (h : ∀ θ, 0 ≤ θ ∧ θ ≤ t → ∃ k : ℤ, θ = (π/2 + 2 * π * k) ∨ θ = (3 * π / 2 + 2 * π * k)) : t = π :=
by {
  sorry
}

end smallest_t_circle_sin_l128_128039


namespace latest_time_to_start_roasting_turkeys_l128_128735

theorem latest_time_to_start_roasting_turkeys
  (turkeys : ℕ) 
  (weight_per_turkey : ℕ) 
  (minutes_per_pound : ℕ) 
  (dinner_time_hours : ℕ)
  (dinner_time_minutes : ℕ) 
  (one_at_a_time : turkeys = 2)
  (weight : weight_per_turkey = 16)
  (roasting_time_per_pound : minutes_per_pound = 15)
  (dinner_hours : dinner_time_hours = 18)
  (dinner_minutes : dinner_time_minutes = 0) :
  (latest_start_hours : ℕ) (latest_start_minutes : ℕ) :=
  latest_start_hours = 10 ∧ latest_start_minutes = 0 := 
sorry

end latest_time_to_start_roasting_turkeys_l128_128735


namespace triangle_side_lengths_exist_l128_128546

theorem triangle_side_lengths_exist :
  ∃ (a b c : ℕ), a ≥ b ∧ b ≥ c ∧ a + b > c ∧ b + c > a ∧ a + c > b ∧ abc = 2 * (a - 1) * (b - 1) * (c - 1) ∧
  ((a, b, c) = (8, 7, 3) ∨ (a, b, c) = (6, 5, 4)) :=
by sorry

end triangle_side_lengths_exist_l128_128546


namespace sum_of_coefficients_is_256_l128_128838

theorem sum_of_coefficients_is_256 :
  ∀ (a a1 a2 a3 a4 a5 a6 a7 a8 : ℤ), 
  ((x : ℤ) - a)^8 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7 + a8 * x^8 → 
  a5 = 56 →
  a + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 256 :=
by
  intros
  sorry

end sum_of_coefficients_is_256_l128_128838


namespace cos_135_eq_neg_sqrt2_div_2_l128_128990

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = - Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l128_128990


namespace sin_30_eq_half_l128_128265

theorem sin_30_eq_half : Real.sin (30 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_eq_half_l128_128265


namespace sin_thirty_degree_l128_128276

theorem sin_thirty_degree : Real.sin (30 * Real.pi / 180) = 1 / 2 := 
by
  sorry

end sin_thirty_degree_l128_128276


namespace triangle_area_l128_128109

-- Define the vertices of the triangle
def point_A : (ℝ × ℝ) := (0, 0)
def point_B : (ℝ × ℝ) := (8, -3)
def point_C : (ℝ × ℝ) := (4, 7)

-- Function to compute the area of a triangle given its vertices
def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Conjecture the area of triangle ABC is 30.0 square units
theorem triangle_area : area_of_triangle point_A point_B point_C = 30.0 := by
  sorry

end triangle_area_l128_128109


namespace apprentice_time_l128_128479

theorem apprentice_time
  (x y : ℝ)
  (h1 : 7 * x + 4 * y = 5 / 9)
  (h2 : 11 * x + 8 * y = 17 / 18)
  (hy : y > 0) :
  1 / y = 24 :=
by
  sorry

end apprentice_time_l128_128479


namespace apples_to_pears_l128_128156

theorem apples_to_pears :
  (∀ (apples oranges pears : ℕ),
  12 * apples = 6 * oranges →
  3 * oranges = 5 * pears →
  24 * apples = 20 * pears) :=
by
  intros apples oranges pears h₁ h₂
  sorry

end apples_to_pears_l128_128156


namespace smallest_solution_l128_128678

theorem smallest_solution :
  (∃ x : ℝ, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧
  ∀ y : ℝ, (1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4)) → x ≤ y) →
  x = 4 - Real.sqrt 2 :=
sorry

end smallest_solution_l128_128678


namespace cos_135_eq_neg_inv_sqrt2_l128_128959

theorem cos_135_eq_neg_inv_sqrt2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt2_l128_128959


namespace solution_set_empty_l128_128696

variable (m x : ℝ)
axiom no_solution (h : (3 - 2 * x) / (x - 3) + (2 + m * x) / (3 - x) = -1) : (1 + m = 0)

theorem solution_set_empty :
  (∀ x, (3 - 2 * x) / (x - 3) + (2 + m * x) / (3 - x) ≠ -1) ↔ m = -1 := by
  sorry

end solution_set_empty_l128_128696


namespace equation_of_line_AB_l128_128926

-- Definition of the given circle
def circle1 : Type := { p : ℝ × ℝ // p.1^2 + (p.2 - 2)^2 = 4 }

-- Definition of the center and point on the second circle
def center : ℝ × ℝ := (0, 2)
def point : ℝ × ℝ := (-2, 6)

-- Definition of the second circle with diameter endpoints
def circle2_eq (x y : ℝ) : Prop := (x + 1)^2 + (y - 4)^2 = 5

-- Statement to be proved
theorem equation_of_line_AB :
  ∃ x y : ℝ, (x^2 + (y - 2)^2 = 4) ∧ ((x + 1)^2 + (y - 4)^2 = 5) ∧ (x - 2*y + 6 = 0) := 
sorry

end equation_of_line_AB_l128_128926


namespace compute_expression_l128_128525

theorem compute_expression : 6^2 - 4 * 5 + 4^2 = 32 := by
  sorry

end compute_expression_l128_128525


namespace red_more_than_yellow_l128_128211

-- Define the total number of marbles
def total_marbles : ℕ := 19

-- Define the number of yellow marbles
def yellow_marbles : ℕ := 5

-- Calculate the number of remaining marbles
def remaining_marbles : ℕ := total_marbles - yellow_marbles

-- Define the ratio of blue to red marbles
def blue_ratio : ℕ := 3
def red_ratio : ℕ := 4

-- Calculate the sum of ratio parts
def sum_ratio : ℕ := blue_ratio + red_ratio

-- Calculate the number of shares per ratio part
def share_per_part : ℕ := remaining_marbles / sum_ratio

-- Calculate the number of red marbles
def red_marbles : ℕ := red_ratio * share_per_part

-- Theorem to prove: the difference between red marbles and yellow marbles is 3
theorem red_more_than_yellow : red_marbles - yellow_marbles = 3 :=
by
  sorry

end red_more_than_yellow_l128_128211


namespace circle_parabola_intersection_l128_128097

theorem circle_parabola_intersection (b : ℝ) : 
  b = 25 / 12 → 
  ∃ (r : ℝ) (cx : ℝ), 
  (∃ p1 p2 : ℝ × ℝ, 
    (p1.2 = 3/4 * p1.1 + b ∧ p2.2 = 3/4 * p2.1 + b) ∧ 
    (p1.2 = 3/4 * p1.1^2 ∧ p2.2 = 3/4 * p2.1^2) ∧ 
    (p1 ≠ (0, 0) ∧ p2 ≠ (0, 0))) ∧ 
  (cx^2 + b^2 = r^2) := 
by 
  sorry

end circle_parabola_intersection_l128_128097


namespace floor_plus_x_eq_17_over_4_l128_128400

theorem floor_plus_x_eq_17_over_4
  (x : ℚ)
  (h : ⌊x⌋ + x = 17 / 4)
  : x = 9 / 4 :=
sorry

end floor_plus_x_eq_17_over_4_l128_128400


namespace limit_at_minus_one_third_l128_128178

theorem limit_at_minus_one_third : 
  ∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧ 
  (∀ (x : ℝ), 0 < |x + 1 / 3| ∧ |x + 1 / 3| < δ → 
  |(9 * x^2 - 1) / (x + 1 / 3) + 6| < ε) :=
sorry

end limit_at_minus_one_third_l128_128178


namespace sum_abc_l128_128588

noncomputable def f (a b c : ℕ) (x : ℤ) : ℤ :=
  if x > 0 then a * x + 3
  else if x = 0 then a * b
  else b * x^2 + c

theorem sum_abc (a b c : ℕ) (h1 : f a b c 2 = 7) (h2 : f a b c 0 = 6) (h3 : f a b c (-1) = 8) :
  a + b + c = 10 :=
by {
  sorry
}

end sum_abc_l128_128588


namespace cos_135_eq_neg_sqrt_two_div_two_l128_128969

theorem cos_135_eq_neg_sqrt_two_div_two : real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt_two_div_two_l128_128969


namespace all_three_white_probability_l128_128500

noncomputable def box_probability : ℚ :=
  let total_white := 4
  let total_black := 7
  let total_balls := total_white + total_black
  let draw_count := 3
  let total_combinations := (total_balls.choose draw_count : ℕ)
  let favorable_combinations := (total_white.choose draw_count : ℕ)
  (favorable_combinations : ℚ) / (total_combinations : ℚ)

theorem all_three_white_probability :
  box_probability = 4 / 165 :=
by
  sorry

end all_three_white_probability_l128_128500


namespace no_solution_for_x_l128_128917

theorem no_solution_for_x (x : ℝ) :
  (1 / (x + 4) + 1 / (x - 4) = 1 / (x - 4)) → False :=
by
  sorry

end no_solution_for_x_l128_128917


namespace science_books_initially_l128_128754

def initial_number_of_books (borrowed left : ℕ) : ℕ := 
borrowed + left

theorem science_books_initially (borrowed left : ℕ) (h1 : borrowed = 18) (h2 : left = 57) :
initial_number_of_books borrowed left = 75 := by
sorry

end science_books_initially_l128_128754


namespace sin_30_eq_half_l128_128288

theorem sin_30_eq_half : Real.sin (Float.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l128_128288


namespace tenth_term_is_correct_l128_128610

-- Conditions and calculation
variable (a l : ℚ)
variable (d : ℚ)
variable (a10 : ℚ)

-- Setting the given values:
noncomputable def first_term : ℚ := 2 / 3
noncomputable def seventeenth_term : ℚ := 3 / 2
noncomputable def common_difference : ℚ := (seventeenth_term - first_term) / 16

-- Calculate the tenth term using the common difference
noncomputable def tenth_term : ℚ := first_term + 9 * common_difference

-- Statement to prove
theorem tenth_term_is_correct : 
  first_term = 2 / 3 →
  seventeenth_term = 3 / 2 →
  common_difference = (3 / 2 - 2 / 3) / 16 →
  tenth_term = 2 / 3 + 9 * ((3 / 2 - 2 / 3) / 16) →
  tenth_term = 109 / 96 :=
  by
    sorry

end tenth_term_is_correct_l128_128610


namespace red_marbles_more_than_yellow_l128_128214

-- Define the conditions
def total_marbles : ℕ := 19
def yellow_marbles : ℕ := 5
def ratio_parts_blue : ℕ := 3
def ratio_parts_red : ℕ := 4

-- Prove the number of more red marbles than yellow marbles is equal to 3
theorem red_marbles_more_than_yellow :
  let remainder_marbles := total_marbles - yellow_marbles
  let total_parts := ratio_parts_blue + ratio_parts_red
  let marbles_per_part := remainder_marbles / total_parts
  let blue_marbles := ratio_parts_blue * marbles_per_part
  let red_marbles := ratio_parts_red * marbles_per_part
  red_marbles - yellow_marbles = 3 := by
  sorry

end red_marbles_more_than_yellow_l128_128214


namespace find_x_of_floor_plus_x_eq_17_over_4_l128_128398

theorem find_x_of_floor_plus_x_eq_17_over_4 (x : ℝ) (hx : ⌊x⌋ + x = 17 / 4) : x = 9 / 4 := 
sorry

end find_x_of_floor_plus_x_eq_17_over_4_l128_128398


namespace value_of_t_l128_128631

theorem value_of_t (t : ℝ) (x y : ℝ) (h1 : x = 1 - 2 * t) (h2 : y = 2 * t - 2) (h3 : x = y) : t = 3 / 4 := 
by
  sorry

end value_of_t_l128_128631


namespace problem_1_exists_unique_tangent_problem_2_range_of_a_l128_128555

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x - 1)

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (a * x - 1) * Real.exp x

theorem problem_1_exists_unique_tangent (a : ℝ) :
  ∃! a : ℝ, ∃ x0 : ℝ, f a x0 = g a x0 ∧ f a x0' = deriv (g a) x0 :=
sorry

theorem problem_2_range_of_a (a : ℝ) :
  (∃ x0 x1 : ℤ, f a x0 > g a x0 ∧ f a x1 > g a x1 ∧ (∀ x : ℤ, x ≠ x0 → x ≠ x1 → f a x ≤ g a x)) →
  a ∈ Set.Ico (Real.exp 2 / (2 * Real.exp 2 - 1)) 1 :=
sorry

end problem_1_exists_unique_tangent_problem_2_range_of_a_l128_128555


namespace find_x_eq_nine_fourths_l128_128388

theorem find_x_eq_nine_fourths (x n : ℚ) (β : ℚ) (h1 : x = n + β) (h2 : n = ⌊x⌋) (h3 : ⌊x⌋ + x = 17 / 4) : x = 9 / 4 :=
by
  sorry

end find_x_eq_nine_fourths_l128_128388


namespace sin_thirty_deg_l128_128325

theorem sin_thirty_deg : Real.sin (π / 6) = 1 / 2 := 
by 
-- Assume that the unit circle point corresponding to 30° is (√3/2, 1/2)
-- Sine of an angle in the unit circle is the y-coordinate of the corresponding point
sorry

end sin_thirty_deg_l128_128325


namespace volume_ratio_of_cubes_l128_128767

-- Given conditions
def edge_length_smaller_cube : ℝ := 6
def edge_length_larger_cube : ℝ := 12

-- Problem statement
theorem volume_ratio_of_cubes : 
  (edge_length_smaller_cube / edge_length_larger_cube) ^ 3 = (1 / 8) := 
by
  sorry

end volume_ratio_of_cubes_l128_128767


namespace jamies_shoes_cost_l128_128485

-- Define the costs of items and the total cost.
def cost_total : ℤ := 110
def cost_coat : ℤ := 40
def cost_one_pair_jeans : ℤ := 20

-- Define the number of pairs of jeans.
def num_pairs_jeans : ℕ := 2

-- Define the cost of Jamie's shoes (to be proved).
def cost_jamies_shoes : ℤ := cost_total - (cost_coat + num_pairs_jeans * cost_one_pair_jeans)

theorem jamies_shoes_cost : cost_jamies_shoes = 30 :=
by
  -- Insert proof here
  sorry

end jamies_shoes_cost_l128_128485


namespace find_h_plus_k_l128_128818

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 14*y - 11 = 0

-- State the problem: Prove h + k = -4 given (h, k) is the center of the circle
theorem find_h_plus_k : (∃ h k, (∀ x y, circle_eq x y ↔ (x - h)^2 + (y - k)^2 = 69) ∧ h + k = -4) :=
by {
  sorry
}

end find_h_plus_k_l128_128818


namespace cos_135_eq_neg_sqrt_two_div_two_l128_128973

theorem cos_135_eq_neg_sqrt_two_div_two : real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt_two_div_two_l128_128973


namespace smallest_solution_l128_128670

theorem smallest_solution (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ 4) (h₃ : x ≠ 5) 
    (h_eq : 1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) : x = 4 - Real.sqrt 2 := 
by 
  sorry

end smallest_solution_l128_128670


namespace extremum_values_of_function_l128_128041

noncomputable def maxValue := Real.sqrt 2 + 1 / Real.sqrt 2
noncomputable def minValue := -Real.sqrt 2 + 1 / Real.sqrt 2

theorem extremum_values_of_function :
  ∀ x : ℝ, - (Real.sqrt 2) + (1 / Real.sqrt 2) ≤ (Real.sin x + Real.cos x + 1 / Real.sqrt (1 + |Real.sin (2 * x)|)) ∧ 
            (Real.sin x + Real.cos x + 1 / Real.sqrt (1 + |Real.sin (2 * x)|)) ≤ (Real.sqrt 2 + 1 / Real.sqrt 2) := 
by
  sorry

end extremum_values_of_function_l128_128041


namespace aaron_age_l128_128802

variable (A : ℕ)
variable (henry_sister_age : ℕ)
variable (henry_age : ℕ)
variable (combined_age : ℕ)

theorem aaron_age (h1 : henry_sister_age = 3 * A)
                 (h2 : henry_age = 4 * henry_sister_age)
                 (h3 : combined_age = henry_sister_age + henry_age)
                 (h4 : combined_age = 240) : A = 16 := by
  sorry

end aaron_age_l128_128802


namespace sin_30_eq_half_l128_128360

theorem sin_30_eq_half : Real.sin (Real.pi / 6) = 1 / 2 := sorry

end sin_30_eq_half_l128_128360


namespace cos_135_eq_neg_sqrt2_div_2_l128_128991

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = - Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l128_128991


namespace truck_capacity_l128_128602

theorem truck_capacity
  (x y : ℝ)
  (h1 : 2 * x + 3 * y = 15.5)
  (h2 : 5 * x + 6 * y = 35) :
  3 * x + 5 * y = 24.5 :=
sorry

end truck_capacity_l128_128602


namespace cos_135_eq_neg_inv_sqrt2_l128_128960

theorem cos_135_eq_neg_inv_sqrt2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt2_l128_128960


namespace sin_30_eq_one_half_l128_128371

def Q_coords (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

theorem sin_30_eq_one_half :
  let Q := Q_coords (Real.pi / 6)
  Q.2 = 1 / 2 :=
by
  sorry

end sin_30_eq_one_half_l128_128371


namespace trigonometric_identity_l128_128942

theorem trigonometric_identity :
  let sin_30 := 1 / 2
  let cos_30 := Real.sqrt 3 / 2
  let tan_60 := Real.sqrt 3
  2 * sin_30 + cos_30 * tan_60 = 5 / 2 :=
by
  let sin_30 := 1 / 2
  let cos_30 := Real.sqrt 3 / 2
  let tan_60 := Real.sqrt 3
  have h1 : 2 * sin_30 = 1 := by norm_num
  have h2 : cos_30 * tan_60 = 3 / 2 := by norm_num
  calc
    2 * sin_30 + cos_30 * tan_60
        = 1 + 3 / 2 : by rw [h1, h2]
    ... = 5 / 2 : by norm_num

end trigonometric_identity_l128_128942


namespace correct_formula_for_xy_l128_128449

theorem correct_formula_for_xy :
  (∀ x y, (x = 1 ∧ y = 3) ∨ (x = 2 ∧ y = 7) ∨ (x = 3 ∧ y = 13) ∨ (x = 4 ∧ y = 21) ∨ (x = 5 ∧ y = 31) →
    y = x^2 + x + 1) :=
sorry

end correct_formula_for_xy_l128_128449


namespace unoccupied_business_class_seats_l128_128234

theorem unoccupied_business_class_seats :
  ∀ (first_class_seats business_class_seats economy_class_seats economy_fullness : ℕ)
  (first_class_occupancy : ℕ) (total_business_first_combined economy_occupancy : ℕ),
  first_class_seats = 10 →
  business_class_seats = 30 →
  economy_class_seats = 50 →
  economy_fullness = economy_class_seats / 2 →
  economy_occupancy = economy_fullness →
  total_business_first_combined = economy_occupancy →
  first_class_occupancy = 3 →
  total_business_first_combined = first_class_occupancy + business_class_seats - (business_class_seats - total_business_first_combined + first_class_occupancy) →
  business_class_seats - (total_business_first_combined - first_class_occupancy) = 8 :=
by
  intros first_class_seats business_class_seats economy_class_seats economy_fullness 
         first_class_occupancy total_business_first_combined economy_occupancy
         h1 h2 h3 h4 h5 h6 h7 h8 
  rw [h1, h2, h3, h4, h5, h6, h7] at h8
  sorry

end unoccupied_business_class_seats_l128_128234


namespace cos_135_eq_neg_sqrt2_div_2_l128_128964

theorem cos_135_eq_neg_sqrt2_div_2 : 
  Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l128_128964


namespace circle_standard_equation_l128_128208

theorem circle_standard_equation (x y : ℝ) (center : ℝ × ℝ) (radius : ℝ) 
  (h_center : center = (2, -1)) (h_radius : radius = 2) :
  (x - center.1) ^ 2 + (y - center.2) ^ 2 = radius ^ 2 ↔ (x - 2) ^ 2 + (y + 1) ^ 2 = 4 := by
  sorry

end circle_standard_equation_l128_128208


namespace greatest_drop_in_june_l128_128933

def monthly_changes := [("January", 1.50), ("February", -2.25), ("March", 0.75), ("April", -3.00), ("May", 1.00), ("June", -4.00)]

theorem greatest_drop_in_june : ∀ months : List (String × Float), (months = monthly_changes) → 
  (∃ month : String, 
    month = "June" ∧ 
    ∀ m p, m ≠ "June" → (m, p) ∈ months → p ≥ -4.00) :=
by
  sorry

end greatest_drop_in_june_l128_128933


namespace function_identity_l128_128124

-- Definitions of the problem
def f (n : ℕ) : ℕ := sorry

-- Main theorem to prove
theorem function_identity (f : ℕ → ℕ) 
  (h : ∀ m n : ℕ, m > 0 → n > 0 → f (m + n) * f (m - n) = f (m * m)) : 
  ∀ n : ℕ, n > 0 → f n = 1 := 
sorry

end function_identity_l128_128124


namespace abs_neg_2_plus_sqrt3_add_tan60_eq_2_l128_128920

theorem abs_neg_2_plus_sqrt3_add_tan60_eq_2 :
  abs (-2 + Real.sqrt 3) + Real.tan (Real.pi / 3) = 2 :=
by
  sorry

end abs_neg_2_plus_sqrt3_add_tan60_eq_2_l128_128920


namespace inverse_comp_inverse_comp_inverse_g4_l128_128756

noncomputable def g : ℕ → ℕ
| 1 := 4
| 2 := 5
| 3 := 1
| 4 := 3
| 5 := 2
| _ := 0 -- not needed, just for completeness

theorem inverse_comp_inverse_comp_inverse_g4 : function.inv_fun g (function.inv_fun g (function.inv_fun g 4)) = 4 :=
by sorry

end inverse_comp_inverse_comp_inverse_g4_l128_128756


namespace smallest_of_five_consecutive_numbers_l128_128773

theorem smallest_of_five_consecutive_numbers (n : ℕ) : 
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 100) → 
  n = 18 :=
by sorry

end smallest_of_five_consecutive_numbers_l128_128773


namespace prank_combinations_l128_128939

theorem prank_combinations :
  let monday_choices := 1
  let tuesday_choices := 2
  let wednesday_choices := 5
  let thursday_choices := 4
  let friday_choices := 1
  (monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices) = 40 :=
by
  sorry

end prank_combinations_l128_128939


namespace smallest_sum_is_4_9_l128_128651

theorem smallest_sum_is_4_9 :
  min
    (min
      (min
        (min (1/3 + 1/4) (1/3 + 1/5))
        (min (1/3 + 1/6) (1/3 + 1/7)))
      (1/3 + 1/9)) = 4/9 :=
  by sorry

end smallest_sum_is_4_9_l128_128651


namespace sin_30_eq_half_l128_128271
open real

noncomputable def sin_30_deg : ℝ :=
  sin (π / 6)

theorem sin_30_eq_half : sin_30_deg = 1 / 2 :=
by
  -- Given the conditions of the problem, we need to prove that sin(π / 6) = 1 / 2
  sorry

end sin_30_eq_half_l128_128271


namespace problem_l128_128887

theorem problem (a b c : ℝ) (h1 : a > b) (h2 : (a - b) * (b - c) * (c - a) > 0) : a > c :=
sorry

end problem_l128_128887


namespace largest_lcm_l128_128764

theorem largest_lcm :
  let l4 := Nat.lcm 18 3
  let l5 := Nat.lcm 18 6
  let l6 := Nat.lcm 18 9
  let l7 := Nat.lcm 18 12
  let l8 := Nat.lcm 18 15
  let l9 := Nat.lcm 18 18
  in max (max (max (max (max l4 l5) l6) l7) l8) l9 = 90 := by
    sorry

end largest_lcm_l128_128764


namespace last_digit_p_minus_q_not_5_l128_128902

theorem last_digit_p_minus_q_not_5 (p q : ℕ) (n : ℕ) 
  (h1 : p * q = 10^n) 
  (h2 : ¬ (p % 10 = 0))
  (h3 : ¬ (q % 10 = 0))
  (h4 : p > q) : (p - q) % 10 ≠ 5 :=
by sorry

end last_digit_p_minus_q_not_5_l128_128902


namespace green_balls_count_l128_128229

def total_balls : ℕ := 60
def white_balls : ℕ := 22
def yellow_balls : ℕ := 2
def red_balls : ℕ := 15
def purple_balls : ℕ := 3
def probability_neither_red_nor_purple : ℝ := 0.7

theorem green_balls_count (G : ℕ) :
  (white_balls + G + yellow_balls) / total_balls = probability_neither_red_nor_purple →
  G = 18 := 
by
  sorry

end green_balls_count_l128_128229


namespace cos_135_eq_neg_inv_sqrt2_l128_128954

theorem cos_135_eq_neg_inv_sqrt2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt2_l128_128954


namespace brass_players_count_l128_128202

def marching_band_size : ℕ := 110
def woodwinds (b : ℕ) : ℕ := 2 * b
def percussion (w : ℕ) : ℕ := 4 * w
def total_members (b : ℕ) : ℕ := b + woodwinds b + percussion (woodwinds b)

theorem brass_players_count : ∃ b : ℕ, total_members b = marching_band_size ∧ b = 10 :=
by
  sorry

end brass_players_count_l128_128202


namespace line_intersection_l128_128099

noncomputable def line1 (t : ℚ) : ℚ × ℚ := (1 - 2 * t, 4 + 3 * t)
noncomputable def line2 (u : ℚ) : ℚ × ℚ := (5 + u, 2 + 6 * u)

theorem line_intersection :
  ∃ t u : ℚ, line1 t = (21 / 5, -4 / 5) ∧ line2 u = (21 / 5, -4 / 5) :=
sorry

end line_intersection_l128_128099


namespace parking_lot_problem_l128_128880

variable (M S : Nat)

theorem parking_lot_problem (h1 : M + S = 30) (h2 : 15 * M + 8 * S = 324) :
  M = 12 ∧ S = 18 :=
by
  -- proof omitted
  sorry

end parking_lot_problem_l128_128880


namespace solve_fractional_equation_l128_128603

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -5) : 
    (2 * x / (x - 1)) - 1 = 4 / (1 - x) → x = -5 := 
by
  sorry

end solve_fractional_equation_l128_128603


namespace part1_part2_l128_128502

-- Definitions based on the conditions
def original_sales : ℕ := 30
def profit_per_shirt_initial : ℕ := 40

-- Additional shirts sold for each 1 yuan price reduction
def additional_shirts_per_yuan : ℕ := 2

-- Price reduction example of 3 yuan
def price_reduction_example : ℕ := 3

-- New sales quantity after 3 yuan reduction
def new_sales_quantity_example := 
  original_sales + (price_reduction_example * additional_shirts_per_yuan)

-- Prove that the sales quantity is 36 shirts for a reduction of 3 yuan
theorem part1 : new_sales_quantity_example = 36 := by
  sorry

-- General price reduction variable
def price_reduction_per_item (x : ℕ) : ℕ := x
def new_profit_per_shirt (x : ℕ) : ℕ := profit_per_shirt_initial - x
def new_sales_quantity (x : ℕ) : ℕ := original_sales + (additional_shirts_per_yuan * x)
def daily_sales_profit (x : ℕ) : ℕ := (new_profit_per_shirt x) * (new_sales_quantity x)

-- Goal for daily sales profit of 1200 yuan
def goal_profit : ℕ := 1200

-- Prove that a price reduction of 25 yuan per shirt achieves a daily sales profit of 1200 yuan
theorem part2 : daily_sales_profit 25 = goal_profit := by
  sorry

end part1_part2_l128_128502


namespace smallest_solution_l128_128679

theorem smallest_solution :
  (∃ x : ℝ, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧
  ∀ y : ℝ, (1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4)) → x ≤ y) →
  x = 4 - Real.sqrt 2 :=
sorry

end smallest_solution_l128_128679


namespace rectangle_square_ratio_l128_128599

theorem rectangle_square_ratio (s x y : ℝ) (h1 : 0.1 * s ^ 2 = 0.25 * x * y) (h2 : y = s / 4) :
  x / y = 6 := 
sorry

end rectangle_square_ratio_l128_128599


namespace sin_30_eq_half_l128_128282

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the point Q at 30 degrees on the unit circle
def Q : ℝ × ℝ := unit_circle (Real.pi / 6)

theorem sin_30_eq_half : (Real.sin (Real.pi / 6)) = 1 / 2 :=
by sorry

end sin_30_eq_half_l128_128282


namespace sin_30_eq_half_l128_128258

theorem sin_30_eq_half :
  let Q := { (x, y) : ℝ × ℝ // x^2 + y^2 = 1 ∧ y ≠ 0 ∧ 0 ≤ x ∧ x ≤ 1 }
  let OE := (1, 0)
  ∀ (Q : (x, y) : ℝ × ℝ), (Q.1, Q.2) = (√3/2, 1/2) →
  ∃ OE : (x, y) : ℝ × ℝ,
    let OQ := (0, 0)
    let QOE_triangle := { c : ℝ × ℝ // ∃ Q OE OQ, QOE c }
      sin 30° = 1/2 :=
  by
    sorry

end sin_30_eq_half_l128_128258


namespace sum_of_digits_next_l128_128866

def sum_of_digits (n : ℕ) : ℕ :=
  (Nat.digits 10 n).sum

theorem sum_of_digits_next (n : ℕ) (h : sum_of_digits n = 1399) : 
  sum_of_digits (n + 1) = 1402 :=
sorry

end sum_of_digits_next_l128_128866


namespace find_x_l128_128393

theorem find_x (x : ℚ) (n : ℤ) (f : ℚ) (h1 : x = n + f) (h2 : n = ⌊x⌋) (h3 : f < 1): 
  ⌊x⌋ + x = 17 / 4 → x = 9 / 4 :=
by 
  sorry

end find_x_l128_128393


namespace smallest_abs_sum_l128_128731

open Matrix

noncomputable def matrix_square_eq (a b c d : ℤ) : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![a, b], ![c, d]] * ![![a, b], ![c, d]]

theorem smallest_abs_sum (a b c d : ℤ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
    (M_eq : matrix_square_eq a b c d = ![![9, 0], ![0, 9]]) :
    |a| + |b| + |c| + |d| = 8 :=
sorry

end smallest_abs_sum_l128_128731


namespace fibonacci_arithmetic_sequence_l128_128025

def fibonacci : ℕ → ℕ
| 0       => 0
| 1       => 1
| (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_arithmetic_sequence (a b c n : ℕ) 
  (h1 : fibonacci 1 = 1)
  (h2 : fibonacci 2 = 1)
  (h3 : ∀ n ≥ 3, fibonacci n = fibonacci (n - 1) + fibonacci (n - 2))
  (h4 : a + b + c = 2500)
  (h5 : (a, b, c) = (n, n + 3, n + 5)) :
  a = 831 := 
sorry

end fibonacci_arithmetic_sequence_l128_128025


namespace closest_point_in_plane_l128_128130

noncomputable def closest_point (x y z : ℚ) : Prop :=
  ∃ (t : ℚ), 
    x = 2 + 2 * t ∧ 
    y = 3 - 3 * t ∧ 
    z = 1 + 4 * t ∧ 
    2 * (2 + 2 * t) - 3 * (3 - 3 * t) + 4 * (1 + 4 * t) = 40

theorem closest_point_in_plane :
  closest_point (92 / 29) (16 / 29) (145 / 29) :=
by
  sorry

end closest_point_in_plane_l128_128130


namespace train_pass_time_l128_128799

-- Assuming conversion factor, length of the train, and speed in km/hr
def conversion_factor := 1000 / 3600
def train_length := 280
def speed_km_hr := 36

-- Defining speed in m/s
def speed_m_s := speed_km_hr * conversion_factor

-- Defining the time to pass a tree
def time_to_pass_tree := train_length / speed_m_s

-- Theorem statement
theorem train_pass_time : time_to_pass_tree = 28 := by
  sorry

end train_pass_time_l128_128799


namespace cos_135_eq_neg_sqrt2_div_2_l128_128965

theorem cos_135_eq_neg_sqrt2_div_2 : 
  Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l128_128965


namespace inequality_for_positive_reals_l128_128019

theorem inequality_for_positive_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (a + b) + 1 / (b + c) + 1 / (c + a)) ≥ ((a + b + c) ^ 2 / (a * b * (a + b) + b * c * (b + c) + c * a * (c + a))) :=
by
  sorry

end inequality_for_positive_reals_l128_128019


namespace neha_amount_removed_l128_128094

theorem neha_amount_removed (N S M : ℝ) (x : ℝ) (total_amnt : ℝ) (M_val : ℝ) (ratio2 : ℝ) (ratio8 : ℝ) (ratio6 : ℝ) :
  total_amnt = 1100 →
  M_val = 102 →
  ratio2 = 2 →
  ratio8 = 8 →
  ratio6 = 6 →
  (M - 4 = ratio6 * x) →
  (S - 8 = ratio8 * x) →
  (N - (N - (ratio2 * x)) = ratio2 * x) →
  (N + S + M = total_amnt) →
  (N - 32.66 = N - (ratio2 * (total_amnt - M_val - 8 - 4) / (ratio2 + ratio8 + ratio6))) →
  N - (N - (ratio2 * ((total_amnt - M_val - 8 - 4) / (ratio2 + ratio8 + ratio6)))) = 826.70 :=
by
  intros
  sorry

end neha_amount_removed_l128_128094


namespace sin_30_eq_half_l128_128268
open real

noncomputable def sin_30_deg : ℝ :=
  sin (π / 6)

theorem sin_30_eq_half : sin_30_deg = 1 / 2 :=
by
  -- Given the conditions of the problem, we need to prove that sin(π / 6) = 1 / 2
  sorry

end sin_30_eq_half_l128_128268


namespace union_complement_A_B_eq_l128_128835

-- Define the universal set U, set A, and set B
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Define the complement of A with respect to U
def complement_U_A : Set ℕ := {x | x ∈ U ∧ x ∉ A}

-- The statement to be proved
theorem union_complement_A_B_eq {U A B : Set ℕ} (hU : U = {0, 1, 2, 3, 4}) 
  (hA : A = {0, 1, 2, 3}) (hB : B = {2, 3, 4}) :
  (complement_U_A) ∪ B = {2, 3, 4} := 
by
  sorry

end union_complement_A_B_eq_l128_128835


namespace find_certain_number_l128_128147

open Real

noncomputable def certain_number (x : ℝ) : Prop :=
  0.75 * x = 0.50 * 900

theorem find_certain_number : certain_number 600 :=
by
  dsimp [certain_number]
  -- We need to show that 0.75 * 600 = 0.50 * 900
  sorry

end find_certain_number_l128_128147


namespace max_square_test_plots_l128_128928

theorem max_square_test_plots 
  (length : ℕ) (width : ℕ) (fence_available : ℕ) 
  (side_length : ℕ) (num_plots : ℕ) 
  (h_length : length = 30)
  (h_width : width = 60)
  (h_fencing : fence_available = 2500)
  (h_side_length : side_length = 10)
  (h_num_plots : num_plots = 18) :
  (length * width / side_length^2 = num_plots) ∧
  (30 * (60 / side_length - 1) + 60 * (30 / side_length - 1) ≤ fence_available) := 
sorry

end max_square_test_plots_l128_128928


namespace sin_30_deg_l128_128302

theorem sin_30_deg : real.sin (30 * real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_deg_l128_128302


namespace boat_speed_in_still_water_l128_128207

theorem boat_speed_in_still_water (x : ℕ) 
  (h1 : x + 17 = 77) (h2 : x - 17 = 43) : x = 60 :=
by
  sorry

end boat_speed_in_still_water_l128_128207


namespace inverse_proportion_function_point_l128_128563

theorem inverse_proportion_function_point (k x y : ℝ) (h₁ : 1 = k / (-6)) (h₂ : y = k / x) :
  k = -6 ∧ (x = 2 ∧ y = -3 ↔ y = -k / x) :=
by
  sorry

end inverse_proportion_function_point_l128_128563


namespace visible_length_red_bus_l128_128885

-- Definitions from the problem
def length_red_bus : ℝ := 48
def length_orange_car : ℝ := length_red_bus / 4
def length_yellow_bus : ℝ := length_orange_car * 3.5

-- Proof statement
theorem visible_length_red_bus : (length_red_bus - length_yellow_bus) = 6 := by
  have h1 : length_orange_car = 12 := by sorry
  have h2 : length_yellow_bus = 42 := by sorry
  have h3 : length_red_bus = 48 := by sorry
  sorry

end visible_length_red_bus_l128_128885


namespace sin_30_eq_half_l128_128286

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the point Q at 30 degrees on the unit circle
def Q : ℝ × ℝ := unit_circle (Real.pi / 6)

theorem sin_30_eq_half : (Real.sin (Real.pi / 6)) = 1 / 2 :=
by sorry

end sin_30_eq_half_l128_128286


namespace sin_30_eq_half_l128_128350

theorem sin_30_eq_half
  (Q : ℝ × ℝ)
  (h1 : Q = (cos (π / 6), sin (π / 6)))
  (E : ℝ × ℝ)
  (h2 : E = (Q.1, 0))
  (h3 : ∃ x y z : ℝ, x / 2 = Q.2 ∧ y = Q.1 ∧ sqrt (x^2 + y^2) = z ∧ z = 1) :
  sin (π / 6) = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l128_128350


namespace sum_of_positive_factors_36_l128_128068

theorem sum_of_positive_factors_36 : 
  ∑ d in (finset.range 37).filter (λ n, 36 ∣ n), d = 91 := 
by
  sorry

end sum_of_positive_factors_36_l128_128068


namespace value_of_expression_l128_128840

theorem value_of_expression (x : ℝ) (h : x^2 - 3 * x = 4) : 3 * x^2 - 9 * x + 8 = 20 := 
by
  sorry

end value_of_expression_l128_128840


namespace gcd_9155_4892_l128_128912

theorem gcd_9155_4892 : Nat.gcd 9155 4892 = 1 := 
by 
  sorry

end gcd_9155_4892_l128_128912


namespace product_of_first_three_terms_l128_128196

/--
  The eighth term of an arithmetic sequence is 20.
  If the difference between two consecutive terms is 2,
  prove that the product of the first three terms of the sequence is 480.
-/
theorem product_of_first_three_terms (a d : ℕ) (h_d : d = 2) (h_eighth_term : a + 7 * d = 20) :
  (a * (a + d) * (a + 2 * d) = 480) :=
by
  sorry

end product_of_first_three_terms_l128_128196


namespace at_least_one_greater_l128_128456

theorem at_least_one_greater (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = a * b * c) :
  a > 17 / 10 ∨ b > 17 / 10 ∨ c > 17 / 10 :=
sorry

end at_least_one_greater_l128_128456


namespace largest_integer_solution_l128_128884

theorem largest_integer_solution (x : ℤ) (h : -x ≥ 2 * x + 3) : x ≤ -1 := sorry

end largest_integer_solution_l128_128884


namespace sum_of_factors_36_l128_128082

theorem sum_of_factors_36 : (∑ (i in Finset.filter (λ x, 36 % x = 0) (Finset.range 37)), i) = 91 :=
by
  -- Proof here
  sorry

end sum_of_factors_36_l128_128082


namespace sin_30_eq_half_l128_128257

theorem sin_30_eq_half :
  let Q := { (x, y) : ℝ × ℝ // x^2 + y^2 = 1 ∧ y ≠ 0 ∧ 0 ≤ x ∧ x ≤ 1 }
  let OE := (1, 0)
  ∀ (Q : (x, y) : ℝ × ℝ), (Q.1, Q.2) = (√3/2, 1/2) →
  ∃ OE : (x, y) : ℝ × ℝ,
    let OQ := (0, 0)
    let QOE_triangle := { c : ℝ × ℝ // ∃ Q OE OQ, QOE c }
      sin 30° = 1/2 :=
  by
    sorry

end sin_30_eq_half_l128_128257


namespace probability_of_odd_numbers_l128_128910

def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

def probability_of_5_odd_numbers_in_7_rolls : ℚ :=
  (binomial 7 5) * (1 / 2)^5 * (1 / 2)^(7-5)

theorem probability_of_odd_numbers :
  probability_of_5_odd_numbers_in_7_rolls = 21 / 128 :=
by
  sorry

end probability_of_odd_numbers_l128_128910


namespace range_of_m_l128_128564

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x + 1| + |x - m| > 4) ↔ m > 3 ∨ m < -5 := 
sorry

end range_of_m_l128_128564


namespace intersection_of_sets_eq_l128_128434

noncomputable def set_intersection (M N : Set ℝ): Set ℝ :=
  {x | x ∈ M ∧ x ∈ N}

theorem intersection_of_sets_eq :
  let M := {x : ℝ | -2 < x ∧ x < 2}
  let N := {x : ℝ | x^2 - 2 * x - 3 < 0}
  set_intersection M N = {x : ℝ | -1 < x ∧ x < 2} := by
  sorry

end intersection_of_sets_eq_l128_128434


namespace sin_30_eq_half_l128_128322

-- Conditions
def point_Q (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

-- The angle 30 degrees in radians
def angle_30_deg := Real.pi / 6

-- Point Q at Angle 30 degrees to the positive x-axis
def Q := point_Q angle_30_deg

-- Formalized problem statement in Lean 4
theorem sin_30_eq_half : Real.sin angle_30_deg = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l128_128322


namespace profit_percentage_is_correct_l128_128932

noncomputable def shopkeeper_profit_percentage : ℚ :=
  let cost_A : ℚ := 12 * (15/16)
  let cost_B : ℚ := 18 * (47/50)
  let profit_A : ℚ := 12 - cost_A
  let profit_B : ℚ := 18 - cost_B
  let total_profit : ℚ := profit_A + profit_B
  let total_cost : ℚ := cost_A + cost_B
  (total_profit / total_cost) * 100

theorem profit_percentage_is_correct :
  shopkeeper_profit_percentage = 6.5 := by
  sorry

end profit_percentage_is_correct_l128_128932


namespace max_a4b2c_l128_128458

-- Define the conditions and required statement
theorem max_a4b2c (a b c : ℝ) (h1: 0 < a) (h2: 0 < b) (h3: 0 < c) (h4: a + b + c = 1) :
    a^4 * b^2 * c ≤ 1024 / 117649 :=
sorry

end max_a4b2c_l128_128458


namespace petya_vasya_common_result_l128_128050

theorem petya_vasya_common_result (a b : ℝ) (h1 : b ≠ 0) (h2 : a/b = (a + b)/(2 * a)) (h3 : a/b ≠ 1) : 
  a/b = -1/2 :=
by 
  sorry

end petya_vasya_common_result_l128_128050


namespace max_value_a7_b7_c7_d7_l128_128183

-- Assume a, b, c, d are real numbers such that a^6 + b^6 + c^6 + d^6 = 64
-- Prove that the maximum value of a^7 + b^7 + c^7 + d^7 is 128
theorem max_value_a7_b7_c7_d7 (a b c d : ℝ) (h : a^6 + b^6 + c^6 + d^6 = 64) : 
  ∃ a b c d, a^6 + b^6 + c^6 + d^6 = 64 ∧ a^7 + b^7 + c^7 + d^7 = 128 :=
by sorry

end max_value_a7_b7_c7_d7_l128_128183


namespace correct_operation_l128_128221

theorem correct_operation (a : ℝ) : 2 * a^3 / a^2 = 2 * a := 
sorry

end correct_operation_l128_128221


namespace proof_cos_135_degree_l128_128993

noncomputable def cos_135_degree_is_negative_sqrt2_over_2 : Prop :=
  real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2

theorem proof_cos_135_degree : cos_135_degree_is_negative_sqrt2_over_2 :=
by
  sorry

end proof_cos_135_degree_l128_128993


namespace sum_abcd_value_l128_128865

theorem sum_abcd_value (a b c d : ℚ) :
  (2 * a + 3 = 2 * b + 5) ∧ 
  (2 * b + 5 = 2 * c + 7) ∧ 
  (2 * c + 7 = 2 * d + 9) ∧ 
  (2 * d + 9 = 2 * (a + b + c + d) + 13) → 
  a + b + c + d = -14 / 3 := 
by
  sorry

end sum_abcd_value_l128_128865


namespace anand_present_age_l128_128788

theorem anand_present_age (A B : ℕ) 
  (h1 : B = A + 10)
  (h2 : A - 10 = (B - 10) / 3) :
  A = 15 :=
sorry

end anand_present_age_l128_128788


namespace compute_expression_l128_128524

theorem compute_expression : 6^2 - 4 * 5 + 4^2 = 32 := by
  sorry

end compute_expression_l128_128524


namespace minimum_omega_l128_128169

noncomputable section

-- Define the function f and its properties
def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.cos (ω * x + φ)

-- Assumptions based on the given conditions
variables {ω φ : ℝ}
variables (T : ℝ) (hω_pos : 0 < ω) (hφ_range : 0 < φ ∧ φ < π)
variables (hT : f T ω φ = Real.sqrt 3 / 2)
variables (hx_zero : f (π / 9) ω φ = 0)

-- Prove the minimum value of ω is 3
theorem minimum_omega : ω = 3 := sorry

end minimum_omega_l128_128169


namespace solution_set_of_inequality_l128_128893

theorem solution_set_of_inequality : {x : ℝ | x^2 + x - 2 < 0} = {x : ℝ | -2 < x ∧ x < 1} :=
by
  sorry

end solution_set_of_inequality_l128_128893


namespace flower_shop_february_roses_l128_128475

theorem flower_shop_february_roses (roses_oct : ℕ) (roses_nov : ℕ) (roses_dec : ℕ) (roses_jan : ℕ) (d : ℕ) :
  roses_oct = 108 →
  roses_nov = 120 →
  roses_dec = 132 →
  roses_jan = 144 →
  roses_nov - roses_oct = d →
  roses_dec - roses_nov = d →
  roses_jan - roses_dec = d →
  (roses_jan + d = 156) :=
by
  intros h_oct h_nov h_dec h_jan h_diff1 h_diff2 h_diff3
  rw [h_jan, h_diff1] at *
  sorry

end flower_shop_february_roses_l128_128475


namespace max_stamps_with_100_dollars_l128_128155

/-- max_stamps function calculates the maximum number of stamps that can be purchased with a given budget, price per stamp, and conditional discount --/
def max_stamps (budget : ℝ) (price_per_stamp : ℝ) (discount_threshold : ℝ) (discount_rate : ℝ) : ℝ :=
  if budget <= discount_threshold * price_per_stamp then
    budget / price_per_stamp
  else
    ((discount_threshold * price_per_stamp) + ((budget - (discount_threshold * price_per_stamp)) / (price_per_stamp * (1 - discount_rate))))

theorem max_stamps_with_100_dollars : max_stamps 100 0.5 100 0.1 = 200 :=
sorry

end max_stamps_with_100_dollars_l128_128155


namespace nathan_subtracts_79_l128_128467

theorem nathan_subtracts_79 (a b : ℤ) (h₁ : a = 40) (h₂ : b = 1) :
  (a - b) ^ 2 = a ^ 2 - 79 := 
by
  sorry

end nathan_subtracts_79_l128_128467


namespace train_passing_pole_l128_128108

variables (v L t_platform D_platform t_pole : ℝ)
variables (H1 : L = 500)
variables (H2 : t_platform = 100)
variables (H3 : D_platform = L + 500)
variables (H4 : t_platform = D_platform / v)

theorem train_passing_pole :
  t_pole = L / v := 
sorry

end train_passing_pole_l128_128108


namespace compute_division_l128_128911

variable (a b c : ℕ)
variable (ha : a = 3)
variable (hb : b = 2)
variable (hc : c = 2)

theorem compute_division : (c * a^3 + c * b^3) / (a^2 - a * b + b^2) = 10 := by
  sorry

end compute_division_l128_128911


namespace first_day_exceeding_100_paperclips_l128_128454

def paperclips_day (k : ℕ) : ℕ := 3 * 2^k

theorem first_day_exceeding_100_paperclips :
  ∃ (k : ℕ), paperclips_day k > 100 ∧ k = 6 := by
  sorry

end first_day_exceeding_100_paperclips_l128_128454


namespace oranges_in_box_l128_128850

theorem oranges_in_box :
  ∃ (A P O : ℕ), A + P + O = 60 ∧ A = 3 * (P + O) ∧ P = (A + O) / 5 ∧ O = 5 :=
by
  sorry

end oranges_in_box_l128_128850


namespace sin_of_30_degrees_l128_128338

-- Define P and D and their relationships on a unit circle
def P : ℝ × ℝ := (cos (π / 6), sin (π / 6))  -- Point P on the unit circle at 30 degrees
def D : ℝ × ℝ := (P.1, 0)  -- Foot of the altitude to the x-axis

theorem sin_of_30_degrees : sin (π / 6) = 1 / 2 :=
by 
  -- Use the fact that the triangle P,O,D is a 30-60-90 triangle
  -- and the unit circle properties.
  sorry

end sin_of_30_degrees_l128_128338


namespace rectangle_width_squared_l128_128637

theorem rectangle_width_squared (w l : ℝ) (h1 : w^2 + l^2 = 400) (h2 : 4 * w^2 + l^2 = 484) : w^2 = 28 := 
by
  sorry

end rectangle_width_squared_l128_128637


namespace linear_function_not_in_fourth_quadrant_l128_128477

theorem linear_function_not_in_fourth_quadrant (a b : ℝ) (h : a = 2 ∧ b = 1) :
  ∀ (x : ℝ), (2 * x + 1 < 0 → x > 0) := 
sorry

end linear_function_not_in_fourth_quadrant_l128_128477


namespace product_of_first_three_terms_l128_128197

/--
  The eighth term of an arithmetic sequence is 20.
  If the difference between two consecutive terms is 2,
  prove that the product of the first three terms of the sequence is 480.
-/
theorem product_of_first_three_terms (a d : ℕ) (h_d : d = 2) (h_eighth_term : a + 7 * d = 20) :
  (a * (a + d) * (a + 2 * d) = 480) :=
by
  sorry

end product_of_first_three_terms_l128_128197


namespace quadratic_has_one_real_solution_l128_128541

theorem quadratic_has_one_real_solution (k : ℝ) (hk : (x + 5) * (x + 2) = k + 3 * x) : k = 6 → ∃! x : ℝ, (x + 5) * (x + 2) = k + 3 * x :=
by
  sorry

end quadratic_has_one_real_solution_l128_128541


namespace unoccupied_seats_in_business_class_l128_128237

/-
Define the numbers for each class and the number of people in each.
-/
def first_class_seats : Nat := 10
def business_class_seats : Nat := 30
def economy_class_seats : Nat := 50
def people_in_first_class : Nat := 3
def people_in_economy_class : Nat := economy_class_seats / 2
def people_in_business_and_first_class : Nat := people_in_economy_class
def people_in_business_class : Nat := people_in_business_and_first_class - people_in_first_class

/-
Prove that the number of unoccupied seats in business class is 8.
-/
theorem unoccupied_seats_in_business_class :
  business_class_seats - people_in_business_class = 8 :=
sorry

end unoccupied_seats_in_business_class_l128_128237


namespace foil_covered_prism_width_l128_128090

theorem foil_covered_prism_width 
    (l w h : ℕ) 
    (h_w_eq_2l : w = 2 * l)
    (h_w_eq_2h : w = 2 * h)
    (h_volume : l * w * h = 128) 
    (h_foiled_width : q = w + 2) :
  q = 10 := 
sorry

end foil_covered_prism_width_l128_128090


namespace total_molecular_weight_correct_l128_128220

-- Defining the molecular weights of elements
def mol_weight_C : ℝ := 12.01
def mol_weight_H : ℝ := 1.01
def mol_weight_Cl : ℝ := 35.45
def mol_weight_O : ℝ := 16.00

-- Defining the number of moles of compounds
def moles_C2H5Cl : ℝ := 15
def moles_O2 : ℝ := 12

-- Calculating the molecular weights of compounds
def mol_weight_C2H5Cl : ℝ := (2 * mol_weight_C) + (5 * mol_weight_H) + mol_weight_Cl
def mol_weight_O2 : ℝ := 2 * mol_weight_O

-- Calculating the total weight of each compound
def total_weight_C2H5Cl : ℝ := moles_C2H5Cl * mol_weight_C2H5Cl
def total_weight_O2 : ℝ := moles_O2 * mol_weight_O2

-- Defining the final total weight
def total_weight : ℝ := total_weight_C2H5Cl + total_weight_O2

-- Statement to prove
theorem total_molecular_weight_correct :
  total_weight = 1351.8 := by
  sorry

end total_molecular_weight_correct_l128_128220


namespace double_neg_five_eq_five_l128_128941

theorem double_neg_five_eq_five : -(-5) = 5 := 
sorry

end double_neg_five_eq_five_l128_128941


namespace decorations_per_box_l128_128010

-- Definitions based on given conditions
def used_decorations : ℕ := 35
def given_away_decorations : ℕ := 25
def number_of_boxes : ℕ := 4

-- Theorem stating the problem
theorem decorations_per_box : (used_decorations + given_away_decorations) / number_of_boxes = 15 := by
  sorry

end decorations_per_box_l128_128010


namespace percent_decrease_l128_128579

theorem percent_decrease (original_price sale_price : ℝ) (h₀ : original_price = 100) (h₁ : sale_price = 30) :
  (original_price - sale_price) / original_price * 100 = 70 :=
by
  rw [h₀, h₁]
  norm_num

end percent_decrease_l128_128579


namespace find_x_l128_128787

theorem find_x (x : ℝ) : (x / 4 * 5 + 10 - 12 = 48) → (x = 40) :=
by
  sorry

end find_x_l128_128787


namespace sector_area_l128_128848

theorem sector_area
  (r : ℝ) (s : ℝ) (h_r : r = 1) (h_s : s = 1) : 
  (1 / 2) * r * s = 1 / 2 := by
  sorry

end sector_area_l128_128848


namespace total_charge_3_hours_l128_128925

-- Define the charges for the first hour (F) and additional hours (A)
variable (F A : ℝ)

-- Given conditions
axiom charge_relation : F = A + 20
axiom total_charge_5_hours : F + 4 * A = 300

-- The theorem stating the total charge for 3 hours of therapy
theorem total_charge_3_hours : 
  (F + 2 * A) = 188 :=
by
  -- Insert the proof here
  sorry

end total_charge_3_hours_l128_128925


namespace infinitely_many_n_divisible_by_n_squared_l128_128743

theorem infinitely_many_n_divisible_by_n_squared :
  ∃ (n : ℕ → ℕ), (∀ k : ℕ, 0 < n k) ∧ (∀ k : ℕ, n k^2 ∣ 2^(n k) + 3^(n k)) :=
sorry

end infinitely_many_n_divisible_by_n_squared_l128_128743


namespace triangle_BC_range_l128_128709

open Real

variable {a C : ℝ} (A : ℝ) (ABC : Triangle A C)

/-- Proof problem statement -/
theorem triangle_BC_range (A C : ℝ) (h0 : 0 < A) (h1 : A < π) (c : ℝ) (h2 : c = sqrt 2) (h3 : a * cos C = c * sin A): 
  ∃ (BC : ℝ), sqrt 2 < BC ∧ BC < 2 :=
sorry

end triangle_BC_range_l128_128709


namespace price_decrease_percentage_l128_128044

-- Definitions based on given conditions
def price_in_2007 (x : ℝ) : ℝ := x
def price_in_2008 (x : ℝ) : ℝ := 1.25 * x
def desired_price_in_2009 (x : ℝ) : ℝ := 1.1 * x

-- Theorem statement to prove the price decrease from 2008 to 2009
theorem price_decrease_percentage (x : ℝ) (h : x > 0) : 
  (1.25 * x - 1.1 * x) / (1.25 * x) = 0.12 := 
sorry

end price_decrease_percentage_l128_128044


namespace num_divisible_by_2_not_by_3_or_5_l128_128517

open Finset

def nums1_to_1000 := (range 1000).image (λ n => n + 1)
def divisible_by (k : ℕ) := nums1_to_1000.filter (λ n => n % k = 0)

def setA := divisible_by 2
def setB := divisible_by 3
def setC := divisible_by 5
    
theorem num_divisible_by_2_not_by_3_or_5 : 
  (setA \ (setB ∪ setC)).card = 267 := 
by
  sorry

end num_divisible_by_2_not_by_3_or_5_l128_128517


namespace largest_lcm_l128_128763

theorem largest_lcm :
  let lcm_value := List.map (λ b => Nat.lcm 18 b) [3, 6, 9, 12, 15, 18]
  in List.maximum! lcm_value = 90 := by
  sorry

end largest_lcm_l128_128763


namespace valid_patents_growth_l128_128516

variable (a b : ℝ)

def annual_growth_rate : ℝ := 0.23

theorem valid_patents_growth (h1 : b = (1 + annual_growth_rate)^2 * a) : b = (1 + 0.23)^2 * a :=
by
  sorry

end valid_patents_growth_l128_128516


namespace students_behind_yoongi_l128_128093

theorem students_behind_yoongi (total_students jungkoo_position students_between_jungkook_yoongi : ℕ) 
    (h1 : total_students = 20)
    (h2 : jungkoo_position = 3)
    (h3 : students_between_jungkook_yoongi = 5) : 
    (total_students - (jungkoo_position + students_between_jungkook_yoongi + 1)) = 11 :=
by
  sorry

end students_behind_yoongi_l128_128093


namespace inequality_not_necessarily_hold_l128_128561

theorem inequality_not_necessarily_hold (a b c d : ℝ) 
  (h1 : a > b) (h2 : c > d) : ¬ (a + d > b + c) :=
sorry

end inequality_not_necessarily_hold_l128_128561


namespace compute_expression_l128_128527

theorem compute_expression : 6^2 - 4 * 5 + 4^2 = 32 := by
  sorry

end compute_expression_l128_128527


namespace cos_135_eq_neg_sqrt2_div_2_l128_128949

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
by 
  sorry

end cos_135_eq_neg_sqrt2_div_2_l128_128949


namespace count_triples_satisfying_conditions_l128_128375

theorem count_triples_satisfying_conditions :
  (∃ a b c : ℕ, 0 < a ∧ 0 < b ∧ 0 < c ∧ ab + bc = 72 ∧ ac + bc = 35) → 
  ∃! t : (ℕ × ℕ × ℕ), 0 < t.1 ∧ 0 < t.2.1 ∧ 0 < t.2.2 ∧ 
                     t.1 * t.2.1 + t.2.1 * t.2.2 = 72 ∧ 
                     t.1 * t.2.2 + t.2.1 * t.2.2 = 35 :=
by sorry

end count_triples_satisfying_conditions_l128_128375


namespace team_X_finishes_with_more_points_than_Y_l128_128659

-- Define the number of teams and games played
def numberOfTeams : ℕ := 8
def gamesPerTeam : ℕ := numberOfTeams - 1

-- Define the probability of winning (since each team has a 50% chance to win any game)
def probOfWin : ℝ := 0.5

-- Define the event that team X finishes with more points than team Y
noncomputable def probXFinishesMorePointsThanY : ℝ := 1 / 2

-- Statement to be proved: 
theorem team_X_finishes_with_more_points_than_Y :
  (∃ p : ℝ, p = probXFinishesMorePointsThanY) :=
sorry

end team_X_finishes_with_more_points_than_Y_l128_128659


namespace unique_solution_l128_128878

theorem unique_solution (x : ℝ) (hx : x ≥ 0) : 2021 * x = 2022 * x ^ (2021 / 2022) - 1 → x = 1 :=
by
  intros h
  sorry

end unique_solution_l128_128878


namespace length_of_GH_l128_128031

-- Define the lengths of the segments as given in the conditions
def AB : ℕ := 11
def FE : ℕ := 13
def CD : ℕ := 5

-- Define what we need to prove: the length of GH is 29
theorem length_of_GH (AB FE CD : ℕ) : AB = 11 → FE = 13 → CD = 5 → (AB + CD + FE = 29) :=
by
  sorry

end length_of_GH_l128_128031


namespace cos_135_eq_neg_sqrt2_div_2_l128_128948

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
by 
  sorry

end cos_135_eq_neg_sqrt2_div_2_l128_128948


namespace solve_for_x_l128_128842

theorem solve_for_x {x : ℤ} (h : x - 2 * x + 3 * x - 4 * x = 120) : x = -60 :=
sorry

end solve_for_x_l128_128842


namespace original_number_is_144_l128_128824

theorem original_number_is_144 :
  ∃ (A B C : ℕ), A ≠ 0 ∧
  (100 * A + 11 * B = 144) ∧
  (A * B^2 = 10 * A + C) ∧
  (A * C = C) ∧
  A = 1 ∧ B = 4 ∧ C = 6 :=
by
  sorry

end original_number_is_144_l128_128824


namespace cos_shifted_eq_l128_128830

noncomputable def cos_shifted (theta : ℝ) (h1 : Real.cos theta = -12 / 13) (h2 : theta ∈ Set.Ioo Real.pi (3 / 2 * Real.pi)) : Real :=
  Real.cos (theta + Real.pi / 4)

theorem cos_shifted_eq (theta : ℝ) (h1 : Real.cos theta = -12 / 13) (h2 : theta ∈ Set.Ioo Real.pi (3 / 2 * Real.pi)) :
  cos_shifted theta h1 h2 = -7 * Real.sqrt 2 / 26 := 
by
  sorry

end cos_shifted_eq_l128_128830


namespace cos_135_degree_l128_128984

theorem cos_135_degree : 
  let Q := (Complex.exp (135 * Real.pi / 180 * Complex.I)) in 
  Complex.re Q = -Real.sqrt 2 / 2 :=
by sorry

end cos_135_degree_l128_128984


namespace vitamin_d_supplements_per_pack_l128_128739

theorem vitamin_d_supplements_per_pack :
  ∃ (x : ℕ), (∀ (n m : ℕ), 7 * n = x * m → 119 <= 7 * n) ∧ (7 * n = 17 * m) :=
by
  -- definition of conditions
  let min_sold := 119
  let vitaminA_per_pack := 7
  -- let x be the number of Vitamin D supplements per pack
  -- the proof is yet to be completed
  sorry

end vitamin_d_supplements_per_pack_l128_128739


namespace smallest_solution_l128_128669

theorem smallest_solution (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ 4) (h₃ : x ≠ 5) 
    (h_eq : 1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) : x = 4 - Real.sqrt 2 := 
by 
  sorry

end smallest_solution_l128_128669


namespace unoccupied_business_class_seats_l128_128235

theorem unoccupied_business_class_seats :
  ∀ (first_class_seats business_class_seats economy_class_seats economy_fullness : ℕ)
  (first_class_occupancy : ℕ) (total_business_first_combined economy_occupancy : ℕ),
  first_class_seats = 10 →
  business_class_seats = 30 →
  economy_class_seats = 50 →
  economy_fullness = economy_class_seats / 2 →
  economy_occupancy = economy_fullness →
  total_business_first_combined = economy_occupancy →
  first_class_occupancy = 3 →
  total_business_first_combined = first_class_occupancy + business_class_seats - (business_class_seats - total_business_first_combined + first_class_occupancy) →
  business_class_seats - (total_business_first_combined - first_class_occupancy) = 8 :=
by
  intros first_class_seats business_class_seats economy_class_seats economy_fullness 
         first_class_occupancy total_business_first_combined economy_occupancy
         h1 h2 h3 h4 h5 h6 h7 h8 
  rw [h1, h2, h3, h4, h5, h6, h7] at h8
  sorry

end unoccupied_business_class_seats_l128_128235


namespace shortest_side_of_right_triangle_l128_128513

theorem shortest_side_of_right_triangle (a b : ℝ) (ha : a = 5) (hb : b = 12) : 
  ∀ c, (c = 5 ∨ c = 12 ∨ c = (Real.sqrt (a^2 + b^2))) → c = 5 :=
by
  intros c h
  sorry

end shortest_side_of_right_triangle_l128_128513


namespace probability_of_odd_numbers_exactly_five_times_in_seven_rolls_l128_128907

theorem probability_of_odd_numbers_exactly_five_times_in_seven_rolls :
  (nat.choose 7 5 * (1/2)^5 * (1/2)^2) = (21 / 128) := by
  sorry

end probability_of_odd_numbers_exactly_five_times_in_seven_rolls_l128_128907


namespace sin_30_eq_half_l128_128297

theorem sin_30_eq_half : Real.sin (π / 6) = 1 / 2 := 
  sorry

end sin_30_eq_half_l128_128297


namespace sin_30_eq_half_l128_128358

theorem sin_30_eq_half : Real.sin (Real.pi / 6) = 1 / 2 := sorry

end sin_30_eq_half_l128_128358


namespace farm_field_area_l128_128717

theorem farm_field_area
  (plough_per_day_planned plough_per_day_actual fields_left : ℕ)
  (D : ℕ) 
  (condition1 : plough_per_day_planned = 100)
  (condition2 : plough_per_day_actual = 85)
  (condition3 : fields_left = 40)
  (additional_days : ℕ) 
  (condition4 : additional_days = 2)
  (initial_days : D + additional_days = 85 * (D + 2) + 40) :
  (100 * D + fields_left = 1440) :=
by
  sorry

end farm_field_area_l128_128717


namespace cos_135_eq_neg_sqrt2_div_2_l128_128967

theorem cos_135_eq_neg_sqrt2_div_2 : 
  Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l128_128967


namespace lucy_snowballs_eq_19_l128_128522

-- Define the conditions
def charlie_snowballs : ℕ := 50
def difference_charlie_lucy : ℕ := 31

-- Define what we want to prove, i.e., Lucy has 19 snowballs
theorem lucy_snowballs_eq_19 : (charlie_snowballs - difference_charlie_lucy = 19) :=
by
  -- We would provide the proof here, but it's not required for this prompt
  sorry

end lucy_snowballs_eq_19_l128_128522


namespace cos_135_eq_neg_sqrt_two_div_two_l128_128974

theorem cos_135_eq_neg_sqrt_two_div_two : real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt_two_div_two_l128_128974


namespace sum_of_x_y_l128_128145

theorem sum_of_x_y :
  ∀ (x y : ℚ), (1 / x + 1 / y = 4) → (1 / x - 1 / y = -8) → x + y = -1 / 3 := 
by
  intros x y h1 h2
  sorry

end sum_of_x_y_l128_128145


namespace quadratic_roots_bounds_l128_128424

theorem quadratic_roots_bounds (m x1 x2 : ℝ) (h : m < 0)
  (hx : x1 < x2) 
  (hr : ∀ x, x^2 - x - 6 = m → x = x1 ∨ x = x2) :
  -2 < x1 ∧ x2 < 3 :=
by
  sorry

end quadratic_roots_bounds_l128_128424


namespace units_digit_2_pow_2015_minus_1_l128_128810

theorem units_digit_2_pow_2015_minus_1 : (2^2015 - 1) % 10 = 7 := by
  sorry

end units_digit_2_pow_2015_minus_1_l128_128810


namespace solve_equation_l128_128604

theorem solve_equation (x : ℝ) : 
  (3 * x + 2) * (x + 3) = x + 3 ↔ (x = -3 ∨ x = -1/3) :=
by sorry

end solve_equation_l128_128604


namespace solve_for_n_l128_128188

theorem solve_for_n (n : ℝ) (h : 0.05 * n + 0.1 * (30 + n) - 0.02 * n = 15.5) : n = 96 := 
by 
  sorry

end solve_for_n_l128_128188


namespace solve_a_range_m_l128_128552

def f (x : ℝ) (a : ℝ) : ℝ := |x - a|

theorem solve_a :
  (∀ x : ℝ, f x 2 ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) ↔ (2 = 2) :=
by {
  sorry
}

theorem range_m :
  (∀ x : ℝ, f (3 * x) 2 + f (x + 3) 2 ≥ m) ↔ (m ≤ 5 / 3) :=
by {
  sorry
}

end solve_a_range_m_l128_128552


namespace race_time_l128_128632

theorem race_time 
  (v t : ℝ)
  (h1 : 1000 = v * t)
  (h2 : 960 = v * (t + 10)) :
  t = 250 :=
by
  sorry

end race_time_l128_128632


namespace sin_30_eq_one_half_l128_128367

def Q_coords (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

theorem sin_30_eq_one_half :
  let Q := Q_coords (Real.pi / 6)
  Q.2 = 1 / 2 :=
by
  sorry

end sin_30_eq_one_half_l128_128367


namespace sin_30_deg_l128_128307

theorem sin_30_deg : real.sin (30 * real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_deg_l128_128307


namespace anna_age_when_married_l128_128862

-- Define constants for the conditions
def j_married : ℕ := 22
def m : ℕ := 30
def combined_age_today : ℕ := 5 * j_married
def j_current : ℕ := j_married + m

-- Define Anna's current age based on the combined age today and Josh's current age
def a_current : ℕ := combined_age_today - j_current

-- Define Anna's age when married
def a_married : ℕ := a_current - m

-- Statement of the theorem to be proved
theorem anna_age_when_married : a_married = 28 :=
by
  sorry

end anna_age_when_married_l128_128862


namespace find_x_of_floor_plus_x_eq_17_over_4_l128_128397

theorem find_x_of_floor_plus_x_eq_17_over_4 (x : ℝ) (hx : ⌊x⌋ + x = 17 / 4) : x = 9 / 4 := 
sorry

end find_x_of_floor_plus_x_eq_17_over_4_l128_128397


namespace combined_work_rate_l128_128504

-- Define the context and the key variables
variable (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b)

-- State the theorem corresponding to the proof problem
theorem combined_work_rate (h_a : a ≠ 0) (h_b : b ≠ 0) : 
  1/a + 1/b = (a * b) / (a + b) * (1/a * 1/b) :=
sorry

end combined_work_rate_l128_128504


namespace sum_of_positive_factors_of_36_l128_128063

theorem sum_of_positive_factors_of_36 : 
  ∑ d in (Finset.filter (λ d, 36 % d = 0) (Finset.range 37)), d = 91 :=
by
  sorry

end sum_of_positive_factors_of_36_l128_128063


namespace cos_135_degree_l128_128978

theorem cos_135_degree : 
  let Q := (Complex.exp (135 * Real.pi / 180 * Complex.I)) in 
  Complex.re Q = -Real.sqrt 2 / 2 :=
by sorry

end cos_135_degree_l128_128978


namespace sin_30_eq_half_l128_128285

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the point Q at 30 degrees on the unit circle
def Q : ℝ × ℝ := unit_circle (Real.pi / 6)

theorem sin_30_eq_half : (Real.sin (Real.pi / 6)) = 1 / 2 :=
by sorry

end sin_30_eq_half_l128_128285


namespace cost_price_of_article_l128_128507

theorem cost_price_of_article (C : ℝ) (SP : ℝ) (C_new : ℝ) (SP_new : ℝ) :
  SP = 1.05 * C →
  C_new = 0.95 * C →
  SP_new = SP - 3 →
  SP_new = 1.045 * C →
  C = 600 :=
by
  intro h1 h2 h3 h4
  -- statement to be proved
  sorry

end cost_price_of_article_l128_128507


namespace one_third_recipe_ingredients_l128_128102

noncomputable def cups_of_flour (f : ℚ) := (f : ℚ)
noncomputable def cups_of_sugar (s : ℚ) := (s : ℚ)
def original_recipe_flour := (27 / 4 : ℚ)  -- mixed number 6 3/4 converted to improper fraction
def original_recipe_sugar := (5 / 2 : ℚ)  -- mixed number 2 1/2 converted to improper fraction

theorem one_third_recipe_ingredients :
  cups_of_flour (original_recipe_flour / 3) = (9 / 4) ∧
  cups_of_sugar (original_recipe_sugar / 3) = (5 / 6) :=
by
  sorry

end one_third_recipe_ingredients_l128_128102


namespace teal_more_green_count_l128_128921

open Set

-- Define the survey data structure
def Survey : Type := {p : ℕ // p ≤ 150}

def people_surveyed : ℕ := 150
def more_blue (s : Survey) : Prop := sorry
def more_green (s : Survey) : Prop := sorry

-- Define the given conditions
def count_more_blue : ℕ := 90
def count_more_both : ℕ := 40
def count_neither : ℕ := 20

-- Define the proof statement
theorem teal_more_green_count :
  (count_more_both + (people_surveyed - (count_neither + (count_more_blue - count_more_both)))) = 80 :=
by {
  -- Sorry is used as a placeholder for the proof
  sorry
}

end teal_more_green_count_l128_128921


namespace no_solution_eq_l128_128698

theorem no_solution_eq (m : ℝ) : 
  (∀ x : ℝ, x ≠ 3 → ((3 - 2 * x) / (x - 3) + (2 + m * x) / (3 - x) = -1) → (m = -1)) :=
by
  sorry

end no_solution_eq_l128_128698


namespace squirrel_nuts_l128_128543

theorem squirrel_nuts :
  ∃ (a b c d : ℕ), 103 ≤ a ∧ 103 ≤ b ∧ 103 ≤ c ∧ 103 ≤ d ∧
                   a ≥ b ∧ a ≥ c ∧ a ≥ d ∧
                   a + b + c + d = 2020 ∧
                   b + c = 1277 ∧
                   a = 640 :=
by {
  -- proof goes here
  sorry
}

end squirrel_nuts_l128_128543


namespace find_x_of_floor_plus_x_eq_17_over_4_l128_128396

theorem find_x_of_floor_plus_x_eq_17_over_4 (x : ℝ) (hx : ⌊x⌋ + x = 17 / 4) : x = 9 / 4 := 
sorry

end find_x_of_floor_plus_x_eq_17_over_4_l128_128396


namespace geraldine_more_than_jazmin_l128_128409

def geraldine_dolls : ℝ := 2186.0
def jazmin_dolls : ℝ := 1209.0
def difference_dolls : ℝ := 977.0

theorem geraldine_more_than_jazmin : geraldine_dolls - jazmin_dolls = difference_dolls :=
by sorry

end geraldine_more_than_jazmin_l128_128409


namespace triangle_angle_not_greater_than_60_l128_128180

theorem triangle_angle_not_greater_than_60 (A B C : ℝ) (h1 : A + B + C = 180) :
  A ≤ 60 ∨ B ≤ 60 ∨ C ≤ 60 :=
sorry -- proof by contradiction to be implemented here

end triangle_angle_not_greater_than_60_l128_128180


namespace find_number_thought_of_l128_128758

theorem find_number_thought_of :
  ∃ x : ℝ, (6 * x^2 - 10) / 3 + 15 = 95 ∧ x = 5 * Real.sqrt 15 / 3 :=
by
  sorry

end find_number_thought_of_l128_128758


namespace min_blocks_for_wall_l128_128233

-- Definitions based on conditions
def length_of_wall := 120
def height_of_wall := 6
def block_height := 1
def block_lengths := [1, 3]
def blocks_third_row := 3

-- Function to calculate the total blocks given the constraints from the conditions
noncomputable def min_blocks_needed : Nat := 164 + 80

-- Theorem assertion that the minimum number of blocks required is 244
theorem min_blocks_for_wall : min_blocks_needed = 244 := by
  -- The proof would go here
  sorry

end min_blocks_for_wall_l128_128233


namespace equation_has_one_real_solution_l128_128542

theorem equation_has_one_real_solution (k : ℚ) :
    (∀ x : ℝ, (x + 5) * (x + 2) = k + 3 * x ↔ x^2 + 4 * x + (10 - k) = 0) →
    (∃ k : ℚ, (∀ x : ℝ, x^2 + 4 * x + (10 - k) = 0 ↔ by sorry (condition for one real solution is equivalent to discriminant being zero), k = 6) := by
    sorry

end equation_has_one_real_solution_l128_128542


namespace probability_at_least_one_male_l128_128821

theorem probability_at_least_one_male (total_students male_students female_students chosen_students : ℕ) 
  (total_comb : ℕ := Nat.choose total_students chosen_students)
  (female_comb : ℕ := Nat.choose female_students chosen_students)
  (prob_at_least_one_male : ℚ := 1 - (female_comb : ℚ) / (total_comb : ℚ)) :
  (total_students = 5) ∧ (male_students = 3) ∧ (female_students = 2) ∧ (chosen_students = 2) → 
  prob_at_least_one_male = 9 / 10 :=
by
  sorry

end probability_at_least_one_male_l128_128821


namespace apple_cost_l128_128806

theorem apple_cost (x l q : ℝ) 
  (h1 : 10 * l = 3.62) 
  (h2 : x * l + (33 - x) * q = 11.67)
  (h3 : x * l + (36 - x) * q = 12.48) : 
  x = 30 :=
by
  sorry

end apple_cost_l128_128806


namespace fg_of_2_l128_128839

def g (x : ℝ) : ℝ := 2 * x^2
def f (x : ℝ) : ℝ := 2 * x - 1

theorem fg_of_2 : f (g 2) = 15 :=
by
  have h1 : g 2 = 8 := by sorry
  have h2 : f 8 = 15 := by sorry
  rw [h1]
  exact h2

end fg_of_2_l128_128839


namespace angle_between_vectors_collinear_points_l128_128584

open Real

-- Part (1)
theorem angle_between_vectors
  (a b : ℝ^3)
  (m : ℝ)
  (h1 : m = -1 / 2)
  (h2 : ∥a∥ = 2 * sqrt 2 * ∥b∥)
  (h3 : ((m + 1) • a + b) ⬝ (a - 3 • b) = 0) :
  ∀ θ, cos θ = 1 / sqrt 2 ↔ θ = π / 4 :=
sorry

-- Part (2)
theorem collinear_points
  (a b : ℝ^3)
  (m : ℝ)
  (OA : ℝ^3 := m • a - b)
  (OB : ℝ^3 := (m + 1) • a + b)
  (OC : ℝ^3 := a - 3 • b)
  (h1 : collinear {OA, OB, OC}) :
  m = 2 :=
sorry

end angle_between_vectors_collinear_points_l128_128584


namespace sin_30_eq_half_l128_128296

theorem sin_30_eq_half : Real.sin (π / 6) = 1 / 2 := 
  sorry

end sin_30_eq_half_l128_128296


namespace function_properties_l128_128587

-- Define the function and conditions
def f : ℝ → ℝ := sorry

axiom condition1 (x : ℝ) : f (10 + x) = f (10 - x)
axiom condition2 (x : ℝ) : f (20 - x) = -f (20 + x)

-- Lean statement to encapsulate the question and expected result
theorem function_properties (x : ℝ) : (f (-x) = -f x) ∧ (f (x + 40) = f x) :=
sorry

end function_properties_l128_128587


namespace money_left_is_40_l128_128738

-- Define the quantities spent by Mildred and Candice, and the total money provided by their mom.
def MildredSpent : ℕ := 25
def CandiceSpent : ℕ := 35
def TotalGiven : ℕ := 100

-- Calculate the total spent and the remaining money.
def TotalSpent := MildredSpent + CandiceSpent
def MoneyLeft := TotalGiven - TotalSpent

-- Prove that the money left is 40.
theorem money_left_is_40 : MoneyLeft = 40 := by
  sorry

end money_left_is_40_l128_128738


namespace sin_30_eq_half_l128_128264

theorem sin_30_eq_half : Real.sin (30 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_eq_half_l128_128264


namespace Tyler_age_l128_128730

variable (T B S : ℕ)

theorem Tyler_age :
  (T = B - 3) ∧
  (S = B + 2) ∧
  (S = 2 * T) ∧
  (T + B + S = 30) →
  T = 5 := by
  sorry

end Tyler_age_l128_128730


namespace find_t_value_l128_128854

theorem find_t_value (k t : ℤ) (h1 : 0 < k) (h2 : k < 10) (h3 : 0 < t) (h4 : t < 10) : t = 6 :=
by
  sorry

end find_t_value_l128_128854


namespace length_of_segment_GH_l128_128027

theorem length_of_segment_GH (a1 a2 a3 a4 : ℕ)
  (h1 : a1 = a2 + 11)
  (h2 : a2 = a3 + 5)
  (h3 : a3 = a4 + 13)
  : a1 - a4 = 29 :=
by
  sorry

end length_of_segment_GH_l128_128027


namespace min_omega_is_three_l128_128170

open Real

noncomputable def min_omega {ω : ℝ} {ϕ : ℝ} (f : ℝ → ℝ) : ℝ :=
  if f = (λ x, cos (ω * x + ϕ)) then
    if 0 < ω ∧ 0 < ϕ ∧ ϕ < π ∧
       (∃ (T : ℝ), T = 2 * π / ω ∧ f T = sqrt 3 / 2) ∧
       f (π / 9) = 0 then
      ω
    else 0
  else 0

theorem min_omega_is_three :
  let f : ℝ → ℝ := λ x, cos (3 * x + π / 6) in
  min_omega f = 3 := by
  sorry

end min_omega_is_three_l128_128170


namespace sin_30_eq_half_l128_128267
open real

noncomputable def sin_30_deg : ℝ :=
  sin (π / 6)

theorem sin_30_eq_half : sin_30_deg = 1 / 2 :=
by
  -- Given the conditions of the problem, we need to prove that sin(π / 6) = 1 / 2
  sorry

end sin_30_eq_half_l128_128267


namespace floor_plus_x_eq_17_over_4_l128_128401

theorem floor_plus_x_eq_17_over_4
  (x : ℚ)
  (h : ⌊x⌋ + x = 17 / 4)
  : x = 9 / 4 :=
sorry

end floor_plus_x_eq_17_over_4_l128_128401


namespace div_problem_l128_128623

theorem div_problem : 150 / (6 / 3) = 75 := by
  sorry

end div_problem_l128_128623


namespace initial_solution_weight_100kg_l128_128228

theorem initial_solution_weight_100kg
  (W : ℝ)
  (initial_salt_percentage : ℝ)
  (added_salt : ℝ)
  (final_salt_percentage : ℝ)
  (H1 : initial_salt_percentage = 0.10)
  (H2 : added_salt = 12.5)
  (H3 : final_salt_percentage = 0.20)
  (H4 : 0.20 * (W + 12.5) = 0.10 * W + 12.5) :
  W = 100 :=   
by 
  sorry

end initial_solution_weight_100kg_l128_128228


namespace sin_thirty_deg_l128_128323

theorem sin_thirty_deg : Real.sin (π / 6) = 1 / 2 := 
by 
-- Assume that the unit circle point corresponding to 30° is (√3/2, 1/2)
-- Sine of an angle in the unit circle is the y-coordinate of the corresponding point
sorry

end sin_thirty_deg_l128_128323


namespace smallest_solution_to_equation_l128_128666

theorem smallest_solution_to_equation :
  ∀ x : ℝ, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) → x = 4 - real.sqrt 2 :=
  sorry

end smallest_solution_to_equation_l128_128666


namespace Olly_needs_24_shoes_l128_128740

def dogs := 3
def cats := 2
def ferrets := 1
def paws_per_dog := 4
def paws_per_cat := 4
def paws_per_ferret := 4

theorem Olly_needs_24_shoes : (dogs * paws_per_dog) + (cats * paws_per_cat) + (ferrets * paws_per_ferret) = 24 :=
by
  sorry

end Olly_needs_24_shoes_l128_128740


namespace circle_chord_area_l128_128851

noncomputable def part_circle_area_between_chords (R : ℝ) : ℝ :=
  (R^2 * (Real.pi + Real.sqrt 3)) / 2

theorem circle_chord_area (R : ℝ) :
  ∀ (a₃ a₆ : ℝ),
    a₃ = Real.sqrt 3 * R →
    a₆ = R →
    part_circle_area_between_chords R = (R^2 * (Real.pi + Real.sqrt 3)) / 2 :=
by
  intros a₃ a₆ h₁ h₂
  sorry

end circle_chord_area_l128_128851


namespace range_of_j_l128_128732

def h (x: ℝ) : ℝ := 2 * x + 1
def j (x: ℝ) : ℝ := h (h (h (h (h x))))

theorem range_of_j :
  ∀ x, -1 ≤ x ∧ x ≤ 3 → -1 ≤ j x ∧ j x ≤ 127 :=
by 
  intros x hx
  sorry

end range_of_j_l128_128732


namespace sum_of_positive_factors_36_l128_128067

theorem sum_of_positive_factors_36 : 
  ∑ d in (finset.range 37).filter (λ n, 36 ∣ n), d = 91 := 
by
  sorry

end sum_of_positive_factors_36_l128_128067


namespace simplify_expr1_simplify_expr2_l128_128187

variable (a b m n : ℝ)

theorem simplify_expr1 : 2 * a - 6 * b - 3 * a + 9 * b = -a + 3 * b := by
  sorry

theorem simplify_expr2 : 2 * (3 * m^2 - m * n) - m * n + m^2 = 7 * m^2 - 3 * m * n := by
  sorry

end simplify_expr1_simplify_expr2_l128_128187


namespace water_pool_amount_is_34_l128_128116

noncomputable def water_in_pool_after_five_hours : ℕ :=
let water_first_hour := 8,
    water_next_two_hours := 2 * 10,
    water_fourth_hour := 14,
    total_water_added := water_first_hour + water_next_two_hours + water_fourth_hour,
    water_leak_fifth_hour := 8
in total_water_added - water_leak_fifth_hour

theorem water_pool_amount_is_34 : water_in_pool_after_five_hours = 34 := by
  sorry

end water_pool_amount_is_34_l128_128116


namespace box_volume_l128_128919

theorem box_volume
  (L W H : ℝ)
  (h1 : L * W = 120)
  (h2 : W * H = 72)
  (h3 : L * H = 60) :
  L * W * H = 720 :=
by
  -- The proof goes here
  sorry

end box_volume_l128_128919


namespace total_students_in_class_l128_128011

theorem total_students_in_class 
  (b : ℕ)
  (boys_jelly_beans : ℕ := b * b)
  (girls_jelly_beans : ℕ := (b + 1) * (b + 1))
  (total_jelly_beans : ℕ := 432) 
  (condition : boys_jelly_beans + girls_jelly_beans = total_jelly_beans) :
  (b + b + 1 = 29) :=
sorry

end total_students_in_class_l128_128011


namespace ratio_of_first_term_to_common_difference_l128_128898

theorem ratio_of_first_term_to_common_difference (a d : ℕ) (h : 15 * a + 105 * d = 3 * (5 * a + 10 * d)) : a = 5 * d :=
by
  sorry

end ratio_of_first_term_to_common_difference_l128_128898


namespace mod_inverse_sum_l128_128814

theorem mod_inverse_sum :
  (2⁻¹ : ℤ) + (4⁻¹ : ℤ) ≡ 5 [MOD 17] :=
by
  have h1 : 2 * 9 ≡ 1 [MOD 17], by norm_num,
  have h2 : 4 * 13 ≡ 1 [MOD 17], by norm_num,
  have inv2 : 2⁻¹ ≡ 9 [MOD 17], from (inv_of_eq_inv h1).symm,
  have inv4 : 4⁻¹ ≡ 13 [MOD 17], from (inv_of_eq_inv h2).symm,
  calc
    (2⁻¹ : ℤ) + (4⁻¹ : ℤ) ≡ 9 + 13 [MOD 17] : by rw [inv2, inv4]
    ... ≡ 22 [MOD 17] : by norm_num
    ... ≡ 5 [MOD 17] : by norm_num

end mod_inverse_sum_l128_128814


namespace remainder_div_72_l128_128625

theorem remainder_div_72 (x : ℤ) (h : x % 8 = 3) : x % 72 = 3 :=
sorry

end remainder_div_72_l128_128625


namespace total_respondents_l128_128494

theorem total_respondents (X Y : ℕ) (h1 : X = 60) (h2 : 3 * Y = X) : X + Y = 80 :=
by
  sorry

end total_respondents_l128_128494


namespace Beethoven_birth_day_l128_128748

theorem Beethoven_birth_day :
  (exists y: ℤ, y.mod 4 = 0 ∧ y.mod 100 ≠ 0 ∨ y.mod 400 = 0 → y ∈ range(1770, 2021) → y)
  → (∃ d: Zmod 7, d = (16 : Zmod 7) - 2) :=
by
  let totalYears := 250
  let leapYears := 60
  let regularYears := 190
  let dayShifts := regularYears + 2 * leapYears
  have modDays : dayShifts % 7 = 2 := by

  -- Proof skipped
  sorry

end Beethoven_birth_day_l128_128748


namespace minimum_value_l128_128128

open Real

theorem minimum_value (x : ℝ) (h : 0 < x) : 
  ∃ y, (∀ z > 0, 3 * sqrt z + 2 / z ≥ y) ∧ y = 5 := by
  sorry

end minimum_value_l128_128128


namespace sin_30_eq_half_l128_128263

theorem sin_30_eq_half : Real.sin (30 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_eq_half_l128_128263


namespace statistical_hypothesis_independence_l128_128162

def independence_test_statistical_hypothesis (A B: Prop) (independence_test: Prop) : Prop :=
  (independence_test ∧ A ∧ B) → (A = B)

theorem statistical_hypothesis_independence (A B: Prop) (independence_test: Prop) :
  (independence_test ∧ A ∧ B) → (A = B) :=
by
  sorry

end statistical_hypothesis_independence_l128_128162


namespace sin_30_eq_half_l128_128294

theorem sin_30_eq_half : Real.sin (Float.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l128_128294


namespace length_of_GH_l128_128030

-- Define the lengths of the segments as given in the conditions
def AB : ℕ := 11
def FE : ℕ := 13
def CD : ℕ := 5

-- Define what we need to prove: the length of GH is 29
theorem length_of_GH (AB FE CD : ℕ) : AB = 11 → FE = 13 → CD = 5 → (AB + CD + FE = 29) :=
by
  sorry

end length_of_GH_l128_128030


namespace sin_30_eq_half_l128_128301

theorem sin_30_eq_half : Real.sin (π / 6) = 1 / 2 := 
  sorry

end sin_30_eq_half_l128_128301


namespace sin_thirty_degrees_l128_128336

theorem sin_thirty_degrees : Real.sin (Real.pi / 6) = 1 / 2 := 
by sorry

end sin_thirty_degrees_l128_128336


namespace parabola_axis_of_symmetry_l128_128049

theorem parabola_axis_of_symmetry (p : ℝ) :
  (∀ x : ℝ, x = 3 → -x^2 - p*x + 2 = -x^2 - (-6)*x + 2) → p = -6 :=
by sorry

end parabola_axis_of_symmetry_l128_128049


namespace sin_30_eq_half_l128_128321

-- Conditions
def point_Q (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

-- The angle 30 degrees in radians
def angle_30_deg := Real.pi / 6

-- Point Q at Angle 30 degrees to the positive x-axis
def Q := point_Q angle_30_deg

-- Formalized problem statement in Lean 4
theorem sin_30_eq_half : Real.sin angle_30_deg = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l128_128321


namespace find_x_when_y_equals_2_l128_128483

theorem find_x_when_y_equals_2 :
  ∀ (y x k : ℝ),
  (y * (Real.sqrt x + 1) = k) →
  (y = 5 → x = 1 → k = 10) →
  (y = 2 → x = 16) := by
  intros y x k h_eq h_initial h_final
  sorry

end find_x_when_y_equals_2_l128_128483


namespace uncle_bradley_money_l128_128903

-- Definitions of the variables and conditions
variables (F H M : ℝ)
variables (h1 : F + H = 13)
variables (h2 : 50 * F = (3 / 10) * M)
variables (h3 : 100 * H = (7 / 10) * M)

-- The theorem statement
theorem uncle_bradley_money : M = 1300 :=
by
  sorry

end uncle_bradley_money_l128_128903


namespace planting_area_l128_128644

variable (x : ℝ)

def garden_length := x + 2
def garden_width := 4
def path_width := 1

def effective_garden_length := garden_length x - 2 * path_width
def effective_garden_width := garden_width - 2 * path_width

theorem planting_area : effective_garden_length x * effective_garden_width = 2 * x := by
  simp [garden_length, garden_width, path_width, effective_garden_length, effective_garden_width]
  sorry

end planting_area_l128_128644


namespace sequence_sum_l128_128481

theorem sequence_sum:
  ∀ (y : ℕ → ℕ), 
  (y 1 = 100) → 
  (∀ k ≥ 2, y k = y (k - 1) ^ 2 + 2 * y (k - 1) + 1) →
  ( ∑' n, 1 / (y n + 1) = 1 / 101 ) :=
by
  sorry

end sequence_sum_l128_128481


namespace proof_problem_l128_128000

variable {a b c d : ℝ}
variable {x1 y1 x2 y2 x3 y3 x4 y4 : ℝ}

-- Assume the conditions
variable (habcd_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
variable (unity_circle : x1^2 + y1^2 = 1 ∧ x2^2 + y2^2 = 1 ∧ x3^2 + y3^2 = 1 ∧ x4^2 + y4^2 = 1)
variable (unit_sum : a * b + c * d = 1)

-- Statement to prove
theorem proof_problem :
  (a * y1 + b * y2 + c * y3 + d * y4)^2 + (a * x4 + b * x3 + c * x2 + d * x1)^2
    ≤ 2 * ((a^2 + b^2) / (a * b) + (c^2 + d^2) / (c * d)) := 
  sorry

end proof_problem_l128_128000


namespace apples_per_box_l128_128216

theorem apples_per_box (crates_apples : ℕ) (num_crates : ℕ) (rotten_apples : ℕ) (num_boxes : ℕ) 
  (h_crates_apples : crates_apples = 42) 
  (h_num_crates : num_crates = 12)
  (h_rotten_apples : rotten_apples = 4)
  (h_num_boxes : num_boxes = 50) :
  (num_crates * crates_apples - rotten_apples) / num_boxes = 10 := 
by
  sorry

end apples_per_box_l128_128216


namespace exists_integers_abcd_l128_128746

theorem exists_integers_abcd (x y z : ℕ) (h : x * y = z^2 + 1) :
  ∃ (a b c d : ℤ), x = a^2 + b^2 ∧ y = c^2 + d^2 ∧ z = a * c + b * d :=
sorry

end exists_integers_abcd_l128_128746


namespace needed_people_l128_128021

theorem needed_people (n t t' k m : ℕ) (h1 : n = 6) (h2 : t = 8) (h3 : t' = 3) 
    (h4 : k = n * t) (h5 : k = m * t') : m - n = 10 :=
by
  sorry

end needed_people_l128_128021


namespace line_intersects_hyperbola_l128_128135

variables (a b : ℝ) (h : a ≠ 0) (k : b ≠ 0)

def line (x y : ℝ) := a * x - y + b = 0

def hyperbola (x y : ℝ) := x^2 / (|a| / |b|) - y^2 / (|b| / |a|) = 1

theorem line_intersects_hyperbola :
  ∃ x y : ℝ, line a b x y ∧ hyperbola a b x y := 
sorry

end line_intersects_hyperbola_l128_128135


namespace gcd_7429_13356_l128_128127

theorem gcd_7429_13356 : Nat.gcd 7429 13356 = 1 := by
  sorry

end gcd_7429_13356_l128_128127


namespace last_integer_in_geometric_sequence_l128_128615

theorem last_integer_in_geometric_sequence (a : ℕ) (r : ℚ) (h_a : a = 2048000) (h_r : r = 1/2) : 
  ∃ n : ℕ, (a : ℚ) * (r^n : ℚ) = 125 := 
by
  sorry

end last_integer_in_geometric_sequence_l128_128615


namespace writing_rate_l128_128453

theorem writing_rate (nathan_rate : ℕ) (jacob_rate : ℕ) : nathan_rate = 25 → jacob_rate = 2 * nathan_rate → (nathan_rate + jacob_rate) * 10 = 750 :=
by
  assume h1 : nathan_rate = 25,
  assume h2 : jacob_rate = 2 * nathan_rate,
  have combined_rate : nathan_rate + jacob_rate = 75, from sorry, -- From calculation in solution step
  show (nathan_rate + jacob_rate) * 10 = 750, from sorry -- Multiplying by 10 as per solution step


end writing_rate_l128_128453


namespace number_of_students_l128_128755

theorem number_of_students (n : ℕ) (h1 : 90 - n = n / 2) : n = 60 :=
by
  sorry

end number_of_students_l128_128755


namespace brass_players_10_l128_128200

theorem brass_players_10:
  ∀ (brass woodwind percussion : ℕ),
    brass + woodwind + percussion = 110 →
    percussion = 4 * woodwind →
    woodwind = 2 * brass →
    brass = 10 :=
by
  intros brass woodwind percussion h1 h2 h3
  sorry

end brass_players_10_l128_128200


namespace sin_30_is_half_l128_128353

noncomputable def sin_30_degrees : ℝ := Real.sin (Real.pi / 6)

theorem sin_30_is_half : sin_30_degrees = 1 / 2 := by
  sorry

end sin_30_is_half_l128_128353


namespace polynomial_abs_sum_l128_128702

theorem polynomial_abs_sum (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ)
  (h : (2*X - 1)^5 = a_5 * X^5 + a_4 * X^4 + a_3 * X^3 + a_2 * X^2 + a_1 * X + a_0) :
  |a_0| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| = 243 :=
by
  sorry

end polynomial_abs_sum_l128_128702


namespace smallest_solution_l128_128685

-- Defining the equation as a condition
def equation (x : ℝ) : Prop := (1 / (x - 3)) + (1 / (x - 5)) = 4 / (x - 4)

-- Proving that the smallest solution is 4 - sqrt(2)
theorem smallest_solution : ∃ x : ℝ, equation x ∧ x = 4 - Real.sqrt 2 := 
by
  -- Proof is omitted
  sorry

end smallest_solution_l128_128685


namespace total_songs_l128_128776

-- Define the number of albums Faye bought and the number of songs per album
def country_albums : ℕ := 2
def pop_albums : ℕ := 3
def songs_per_album : ℕ := 6

-- Define the total number of albums Faye bought
def total_albums : ℕ := country_albums + pop_albums

-- Prove that the total number of songs Faye bought is 30
theorem total_songs : total_albums * songs_per_album = 30 := by
  sorry

end total_songs_l128_128776


namespace race_result_130m_l128_128224

theorem race_result_130m (d : ℕ) (t_a t_b: ℕ) (a_speed b_speed : ℚ) (d_a_t : ℚ) (d_b_t : ℚ) (distance_covered_by_B_in_20_secs : ℚ) :
  d = 130 →
  t_a = 20 →
  t_b = 25 →
  a_speed = (↑d) / t_a →
  b_speed = (↑d) / t_b →
  d_a_t = a_speed * t_a →
  d_b_t = b_speed * t_b →
  distance_covered_by_B_in_20_secs = b_speed * 20 →
  (d - distance_covered_by_B_in_20_secs = 26) :=
by
  sorry

end race_result_130m_l128_128224


namespace value_of_m_l128_128846

theorem value_of_m (m x : ℝ) (h : x - 4 ≠ 0) (hx_pos : x > 0) 
  (eqn : m / (x - 4) - (1 - x) / (4 - x) = 0) : m = 3 := 
by
  sorry

end value_of_m_l128_128846


namespace sum_of_positive_factors_36_l128_128065

def sum_of_factors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, n % d = 0).sum

theorem sum_of_positive_factors_36 : sum_of_factors 36 = 91 :=
by
  sorry

end sum_of_positive_factors_36_l128_128065


namespace geometric_sequence_sum_range_l128_128572

theorem geometric_sequence_sum_range {a : ℕ → ℝ}
  (h4_8: a 4 * a 8 = 9) :
  a 3 + a 9 ∈ Set.Iic (-6) ∪ Set.Ici 6 :=
sorry

end geometric_sequence_sum_range_l128_128572


namespace sum_of_percentages_l128_128734

theorem sum_of_percentages : 
  let x := 80 + (0.2 * 80)
  let y := 60 - (0.3 * 60)
  let z := 40 + (0.5 * 40)
  x + y + z = 198 := by
  sorry

end sum_of_percentages_l128_128734


namespace minimum_value_of_k_l128_128429

theorem minimum_value_of_k (x y : ℝ) (h : x * (x - 1) ≤ y * (1 - y)) : x^2 + y^2 ≤ 2 :=
sorry

end minimum_value_of_k_l128_128429


namespace students_more_than_Yoongi_l128_128627

theorem students_more_than_Yoongi (total_players : ℕ) (less_than_Yoongi : ℕ) (total_players_eq : total_players = 21) (less_than_eq : less_than_Yoongi = 11) : 
  ∃ more_than_Yoongi : ℕ, more_than_Yoongi = (total_players - 1 - less_than_Yoongi) ∧ more_than_Yoongi = 8 :=
by
  sorry

end students_more_than_Yoongi_l128_128627


namespace triangle_ef_sum_l128_128574

noncomputable def triangle_length_ef (D E F : ℝ) (angleD : ℝ) (sideDE : ℝ) (sideDF : ℝ) :=
  let sin45 := Real.sin (45 * Real.pi / 180)
  let angleF1 := 30 * Real.pi / 180
  let angleF2 := 150 * Real.pi / 180
  let sinF1 := Real.sin angleF1
  let sinF2 := Real.sin angleF2
  let cos105 := -Real.sin (15 * Real.pi / 180)
  let sideEF1 := Real.sqrt (100^2 + (100 * Real.sqrt 2)^2 - 2 * 100 * (100 * Real.sqrt 2) * cos105)
  let sideEF2 := 0 -- invalid case flag
  if (angleF1 + angleD) < 180 / 180 * Real.pi then sideEF1
  else if (angleF2 + angleD) < 180 / 180 * Real.pi then sideEF2
  else 0

theorem triangle_ef_sum :
  triangle_length_ef 0 0 0 (45 * Real.pi / 180) 100 (100 * Real.sqrt 2) = 201 :=
sorry

end triangle_ef_sum_l128_128574


namespace change_given_l128_128009

theorem change_given (pants_cost : ℕ) (shirt_cost : ℕ) (tie_cost : ℕ) (total_paid : ℕ) (total_cost : ℕ) (change : ℕ) :
  pants_cost = 140 ∧ shirt_cost = 43 ∧ tie_cost = 15 ∧ total_paid = 200 ∧ total_cost = (pants_cost + shirt_cost + tie_cost) ∧ change = (total_paid - total_cost) → change = 2 :=
by
  sorry

end change_given_l128_128009


namespace value_of_M_correct_l128_128046

noncomputable def value_of_M : ℤ :=
  let d1 := 4        -- First column difference
  let d2 := -7       -- Row difference
  let d3 := 1        -- Second column difference
  let a1 := 25       -- First number in the row
  let a2 := 16 - d1  -- First number in the first column
  let a3 := a1 - d2 * 6  -- Last number in the row
  a3 + d3

theorem value_of_M_correct : value_of_M = -16 :=
  by
    let d1 := 4       -- First column difference
    let d2 := -7      -- Row difference
    let d3 := 1       -- Second column difference
    let a1 := 25      -- First number in the row
    let a2 := 16 - d1 -- First number in the first column
    let a3 := a1 - d2 * 6 -- Last number in the row
    have : a3 + d3 = -16
    · sorry
    exact this

end value_of_M_correct_l128_128046


namespace sum_of_factors_36_l128_128072

theorem sum_of_factors_36 : (∑ d in (Finset.range (36 + 1)).filter (λ d, 36 % d = 0), d) = 91 :=
by
  sorry

end sum_of_factors_36_l128_128072


namespace chad_sandwiches_l128_128521

-- Definitions representing the conditions
def crackers_per_sleeve : ℕ := 28
def sleeves_per_box : ℕ := 4
def boxes : ℕ := 5
def nights : ℕ := 56
def crackers_per_sandwich : ℕ := 2

-- Definition representing the final question about the number of sandwiches
def sandwiches_per_night (crackers_per_sleeve sleeves_per_box boxes nights crackers_per_sandwich : ℕ) : ℕ :=
  (crackers_per_sleeve * sleeves_per_box * boxes) / nights / crackers_per_sandwich

-- The theorem that states Chad makes 5 sandwiches each night
theorem chad_sandwiches :
  sandwiches_per_night crackers_per_sleeve sleeves_per_box boxes nights crackers_per_sandwich = 5 :=
by
  -- Proof outline:
  -- crackers_per_sleeve * sleeves_per_box * boxes = 28 * 4 * 5 = 560
  -- 560 / nights = 560 / 56 = 10 crackers per night
  -- 10 / crackers_per_sandwich = 10 / 2 = 5 sandwiches per night
  sorry

end chad_sandwiches_l128_128521


namespace find_digit_A_l128_128752

theorem find_digit_A (A : ℕ) (h1 : 0 ≤ A ∧ A ≤ 9) (h2 : (2 + A + 3 + A) % 9 = 0) : A = 2 :=
by
  sorry

end find_digit_A_l128_128752


namespace complement_union_eq_l128_128166

namespace SetComplementUnion

-- Defining the universal set U, set M and set N.
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x < 2}
def N : Set ℝ := {x | -1 < x ∧ x < 3}

-- Proving the desired equality
theorem complement_union_eq :
  (U \ M) ∪ N = {x | x > -1} :=
sorry

end SetComplementUnion

end complement_union_eq_l128_128166


namespace sin_thirty_deg_l128_128329

theorem sin_thirty_deg : Real.sin (π / 6) = 1 / 2 := 
by 
-- Assume that the unit circle point corresponding to 30° is (√3/2, 1/2)
-- Sine of an angle in the unit circle is the y-coordinate of the corresponding point
sorry

end sin_thirty_deg_l128_128329


namespace solve_quadratic_equation_l128_128605

theorem solve_quadratic_equation (x : ℝ) :
  x^2 - 6 * x - 3 = 0 ↔ x = 3 + 2 * Real.sqrt 3 ∨ x = 3 - 2 * Real.sqrt 3 :=
by
  sorry

end solve_quadratic_equation_l128_128605


namespace sin_of_30_degrees_l128_128339

-- Define P and D and their relationships on a unit circle
def P : ℝ × ℝ := (cos (π / 6), sin (π / 6))  -- Point P on the unit circle at 30 degrees
def D : ℝ × ℝ := (P.1, 0)  -- Foot of the altitude to the x-axis

theorem sin_of_30_degrees : sin (π / 6) = 1 / 2 :=
by 
  -- Use the fact that the triangle P,O,D is a 30-60-90 triangle
  -- and the unit circle properties.
  sorry

end sin_of_30_degrees_l128_128339


namespace cos_135_eq_neg_sqrt2_div_2_l128_128952

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
by 
  sorry

end cos_135_eq_neg_sqrt2_div_2_l128_128952


namespace magnitude_of_error_l128_128509

theorem magnitude_of_error (x : ℝ) (hx : 0 < x) :
  abs ((4 * x) - (x / 4)) / (4 * x) * 100 = 94 := 
sorry

end magnitude_of_error_l128_128509


namespace squirrel_travel_time_l128_128796

theorem squirrel_travel_time :
  ∀ (speed distance : ℝ), speed = 5 → distance = 3 →
  (distance / speed) * 60 = 36 := by
  intros speed distance h_speed h_distance
  rw [h_speed, h_distance]
  norm_num

end squirrel_travel_time_l128_128796


namespace product_evaluation_l128_128122

-- Define the conditions and the target expression
def product (a : ℕ) : ℕ := (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a

-- Main theorem statement
theorem product_evaluation : product 7 = 5040 :=
by
  -- Lean usually requires some import from the broader Mathlib to support arithmetic simplifications
  sorry

end product_evaluation_l128_128122


namespace cos_135_degree_l128_128980

theorem cos_135_degree : 
  let Q := (Complex.exp (135 * Real.pi / 180 * Complex.I)) in 
  Complex.re Q = -Real.sqrt 2 / 2 :=
by sorry

end cos_135_degree_l128_128980


namespace probability_of_odd_numbers_l128_128909

def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

def probability_of_5_odd_numbers_in_7_rolls : ℚ :=
  (binomial 7 5) * (1 / 2)^5 * (1 / 2)^(7-5)

theorem probability_of_odd_numbers :
  probability_of_5_odd_numbers_in_7_rolls = 21 / 128 :=
by
  sorry

end probability_of_odd_numbers_l128_128909


namespace tangent_line_intersects_y_axis_at_10_l128_128618

-- Define the curve y = x^2 + 11
def curve (x : ℝ) : ℝ := x^2 + 11

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 2 * x

-- Define the point of tangency
def point_of_tangency : ℝ × ℝ := (1, 12)

-- Define the tangent line at point_of_tangency
def tangent_line (x : ℝ) : ℝ :=
  let slope := curve_derivative point_of_tangency.1
  let y_intercept := point_of_tangency.2 - slope * point_of_tangency.1
  slope * x + y_intercept

-- Theorem stating the y-coordinate of the intersection of the tangent line with the y-axis
theorem tangent_line_intersects_y_axis_at_10 :
  tangent_line 0 = 10 :=
by
  sorry

end tangent_line_intersects_y_axis_at_10_l128_128618


namespace length_of_GH_l128_128033

theorem length_of_GH (AB FE CD : ℕ) (side_large side_second side_third side_small : ℕ) 
  (h1 : AB = 11) (h2 : FE = 13) (h3 : CD = 5)
  (h4 : side_large = side_second + AB)
  (h5 : side_second = side_third + CD)
  (h6 : side_third = side_small + FE) :
  GH = 29 :=
by
  -- Proof steps would follow here based on the problem's solution
  -- Using the given conditions and transformations.
  sorry

end length_of_GH_l128_128033


namespace fractionD_is_unchanged_l128_128567

-- Define variables x and y
variable (x y : ℚ)

-- Define the fractions
def fractionA := x / (y + 1)
def fractionB := (x + y) / (x + 1)
def fractionC := (x * y) / (x + y)
def fractionD := (2 * x) / (3 * x - y)

-- Define the transformation
def transform (a b : ℚ) : ℚ × ℚ := (3 * a, 3 * b)

-- Define the new fractions after transformation
def newFractionA := (3 * x) / (3 * y + 1)
def newFractionB := (3 * x + 3 * y) / (3 * x + 1)
def newFractionC := (9 * x * y) / (3 * x + 3 * y)
def newFractionD := (6 * x) / (9 * x - 3 * y)

-- The proof problem statement
theorem fractionD_is_unchanged :
  fractionD x y = newFractionD x y ∧
  fractionA x y ≠ newFractionA x y ∧
  fractionB x y ≠ newFractionB x y ∧
  fractionC x y ≠ newFractionC x y := sorry

end fractionD_is_unchanged_l128_128567


namespace simplify_expression_l128_128020

theorem simplify_expression :
  (Real.sqrt (8^(1/3)) + Real.sqrt (17/4))^2 = (33 + 8 * Real.sqrt 17) / 4 :=
by
  sorry

end simplify_expression_l128_128020


namespace sin_thirty_degree_l128_128279

theorem sin_thirty_degree : Real.sin (30 * Real.pi / 180) = 1 / 2 := 
by
  sorry

end sin_thirty_degree_l128_128279


namespace kia_vehicle_count_l128_128004

theorem kia_vehicle_count (total_vehicles : Nat) (dodge_vehicles : Nat) (hyundai_vehicles : Nat) 
    (h1 : total_vehicles = 400)
    (h2 : dodge_vehicles = total_vehicles / 2)
    (h3 : hyundai_vehicles = dodge_vehicles / 2) : 
    (total_vehicles - dodge_vehicles - hyundai_vehicles) = 100 := 
by sorry

end kia_vehicle_count_l128_128004


namespace investment_value_after_five_years_l128_128111

theorem investment_value_after_five_years :
  let initial_investment := 10000
  let year1 := initial_investment * (1 - 0.05) * (1 + 0.02)
  let year2 := year1 * (1 + 0.10) * (1 + 0.02)
  let year3 := year2 * (1 + 0.04) * (1 + 0.02)
  let year4 := year3 * (1 - 0.03) * (1 + 0.02)
  let year5 := year4 * (1 + 0.08) * (1 + 0.02)
  year5 = 12570.99 :=
  sorry

end investment_value_after_five_years_l128_128111


namespace prob_Allison_wins_l128_128246

open ProbabilityTheory

def outcome_space (n : ℕ) := {k : ℕ // k < n}

def Brian_roll : outcome_space 6 := sorry -- Representing Brian's possible outcomes: 1, 2, 3, 4, 5, 6
def Noah_roll : outcome_space 2 := sorry -- Representing Noah's possible outcomes: 3 or 5 (3 is 0 and 5 is 1 in the index for simplicity)

noncomputable def prob_Brian_less_than_4 := (|{k : ℕ // k < 3}| : ℝ) / (|{k : ℕ // k < 6}|) -- Probability Brian rolls 1, 2, or 3
noncomputable def prob_Noah_less_than_4 := (|{k : ℕ // k = 0}| : ℝ) / (|{k : ℕ // k < 2}|)  -- Probability Noah rolls 3 (index 0)

theorem prob_Allison_wins : (prob_Brian_less_than_4 * prob_Noah_less_than_4) = 1 / 4 := by
  sorry

end prob_Allison_wins_l128_128246


namespace pies_calculation_l128_128600

-- Definition: Number of ingredients per pie
def ingredients_per_pie (apples total_apples pies : ℤ) : ℤ := total_apples / pies

-- Definition: Number of pies that can be made with available ingredients 
def pies_from_ingredients (ingredient_amount per_pie : ℤ) : ℤ := ingredient_amount / per_pie

-- Hypothesis
theorem pies_calculation (apples_per_pie pears_per_pie apples pears pies : ℤ) 
  (h1: ingredients_per_pie apples 12 pies = 4)
  (h2: ingredients_per_pie apples 6 pies = 2)
  (h3: pies_from_ingredients 36 4 = 9)
  (h4: pies_from_ingredients 18 2 = 9): 
  pies = 9 := 
sorry

end pies_calculation_l128_128600


namespace exists_power_of_two_with_consecutive_zeros_l128_128742

theorem exists_power_of_two_with_consecutive_zeros (k : ℕ) (hk : k ≥ 1) :
  ∃ n : ℕ, ∃ a b : ℕ, ∃ m : ℕ, 2^n = a * 10^(m + k) + b ∧ 10^(k - 1) ≤ b ∧ b < 10^k ∧ a % 10 ≠ 0 ∧ b % 10 ≠ 0 :=
sorry

end exists_power_of_two_with_consecutive_zeros_l128_128742


namespace john_unanswered_questions_l128_128455

theorem john_unanswered_questions
  (x y z : ℕ)
  (h1 : 5 * x + 2 * z = 93)
  (h2 : 4 * x - y = 54)
  (h3 : x + y + z = 30) : z = 9 :=
sorry

end john_unanswered_questions_l128_128455


namespace correct_calculation_result_l128_128837

theorem correct_calculation_result (x : ℤ) (h : x - 32 = 25) : x - 23 = 34 :=
by
  sorry

end correct_calculation_result_l128_128837


namespace sum_of_positive_divisors_of_36_l128_128080

theorem sum_of_positive_divisors_of_36 : 
  ( ∑ d in (finset.filter (λ x, 36 % x = 0) (finset.range (36 + 1))), d) = 91 :=
by
  sorry

end sum_of_positive_divisors_of_36_l128_128080


namespace necessary_and_sufficient_condition_l128_128554

noncomputable def f (a b c x : ℝ) : ℝ :=
  a * x ^ 2 + b * x + c

theorem necessary_and_sufficient_condition
  {a b c : ℝ}
  (ha_pos : a > 0) :
  ( (∀ y : ℝ, y ∈ { y : ℝ | ∃ x : ℝ, f a b c x = y } → ∃! x : ℝ, f a b c x = y) ∧ 
    (∀ y : ℝ, y ∈ { y : ℝ | ∃ x : ℝ, y = f a b c x } → ∃! x : ℝ, f a b c x = y)
  ) ↔
  f a b c (f a b c (-b / (2 * a))) < 0 :=
sorry

end necessary_and_sufficient_condition_l128_128554


namespace sin_30_eq_half_l128_128281

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the point Q at 30 degrees on the unit circle
def Q : ℝ × ℝ := unit_circle (Real.pi / 6)

theorem sin_30_eq_half : (Real.sin (Real.pi / 6)) = 1 / 2 :=
by sorry

end sin_30_eq_half_l128_128281


namespace smallest_solution_to_equation_l128_128665

theorem smallest_solution_to_equation :
  ∀ x : ℝ, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) → x = 4 - real.sqrt 2 :=
  sorry

end smallest_solution_to_equation_l128_128665


namespace gcd_840_1764_l128_128478

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 :=
by
  sorry

end gcd_840_1764_l128_128478


namespace smallest_solution_l128_128683

-- Defining the equation as a condition
def equation (x : ℝ) : Prop := (1 / (x - 3)) + (1 / (x - 5)) = 4 / (x - 4)

-- Proving that the smallest solution is 4 - sqrt(2)
theorem smallest_solution : ∃ x : ℝ, equation x ∧ x = 4 - Real.sqrt 2 := 
by
  -- Proof is omitted
  sorry

end smallest_solution_l128_128683


namespace mark_lloyd_ratio_l128_128465

theorem mark_lloyd_ratio (M L C : ℕ) (h1 : M = L) (h2 : M = C - 10) (h3 : C = 100) (h4 : M + L + C + 80 = 300) : M = L :=
by {
  sorry -- proof steps go here
}

end mark_lloyd_ratio_l128_128465


namespace sum_of_positive_factors_36_l128_128076

def sum_of_divisors (n : Nat) : Nat :=
  (List.range (n + 1)).filter (fun m => n % m = 0) |>.sum

theorem sum_of_positive_factors_36 : sum_of_divisors 36 = 91 := by
  sorry

end sum_of_positive_factors_36_l128_128076


namespace sin_30_eq_half_l128_128269
open real

noncomputable def sin_30_deg : ℝ :=
  sin (π / 6)

theorem sin_30_eq_half : sin_30_deg = 1 / 2 :=
by
  -- Given the conditions of the problem, we need to prove that sin(π / 6) = 1 / 2
  sorry

end sin_30_eq_half_l128_128269


namespace sum_of_positive_factors_of_36_l128_128061

def prime_factorization_36 : Prop :=
  prime.factorization 36 = {2: 2, 3: 2}

def sum_of_divisors_formula (n : ℕ) (p q : ℕ) (np nq : ℕ) : ℕ :=
  (Finset.range (np + 1)).sum (λ i, p^i) * (Finset.range (nq + 1)).sum (λ j, q^j)

def sum_of_divisors_36 : Prop :=
  sum_of_divisors_formula 36 2 3 2 2 = 91

theorem sum_of_positive_factors_of_36 : prime_factorization_36 → sum_of_divisors_36 :=
by {
  sorry
}

end sum_of_positive_factors_of_36_l128_128061


namespace johns_total_due_l128_128860

noncomputable def total_amount_due (initial_amount : ℝ) (first_charge_rate : ℝ) 
  (second_charge_rate : ℝ) (third_charge_rate : ℝ) : ℝ := 
  let after_first_charge := initial_amount * first_charge_rate
  let after_second_charge := after_first_charge * second_charge_rate
  let after_third_charge := after_second_charge * third_charge_rate
  after_third_charge

theorem johns_total_due : total_amount_due 500 1.02 1.03 1.025 = 538.43 := 
  by
    -- The proof would go here.
    sorry

end johns_total_due_l128_128860


namespace problem1_problem2_l128_128432

noncomputable def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - (3 * a + 1)) < 0}

noncomputable def B (a : ℝ) : Set ℝ := {x | (x - 2 * a) / (x - (a^2 + 1)) < 0}

variable (a : ℝ) (x : ℝ)

-- Problem 1: Proving intersection of sets when a = 2
theorem problem1 (ha : a = 2) : (A a ∩ B a) = {x | 4 < x ∧ x < 5} :=
sorry

-- Problem 2: Proving the range of a for which B is a subset of A
theorem problem2 : {a | B a ⊆ A a} = {a | (1 < a ∧ a ≤ 3) ∨ a = -1} :=
sorry

end problem1_problem2_l128_128432


namespace men_with_all_items_and_married_l128_128053

theorem men_with_all_items_and_married (total: ℕ) (married: ℕ) (tv: ℕ) (radio: ℕ) (car: ℕ) (refrigerator: ℕ) (ac: ℕ) :
  total = 500 → married = 350 → tv = 375 → radio = 450 → car = 325 → refrigerator = 275 → ac = 300 → 
  ∃ x, x ≤ 275 :=
by
  assume h1 : total = 500
  assume h2 : married = 350
  assume h3 : tv = 375
  assume h4 : radio = 450
  assume h5 : car = 325
  assume h6 : refrigerator = 275
  assume h7 : ac = 300
  use 275
  sorry

end men_with_all_items_and_married_l128_128053


namespace melanie_gave_8_dimes_l128_128176

theorem melanie_gave_8_dimes
  (initial_dimes : ℕ)
  (additional_dimes : ℕ)
  (current_dimes : ℕ)
  (given_away_dimes : ℕ) :
  initial_dimes = 7 →
  additional_dimes = 4 →
  current_dimes = 3 →
  given_away_dimes = (initial_dimes + additional_dimes - current_dimes) →
  given_away_dimes = 8 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end melanie_gave_8_dimes_l128_128176


namespace sin_thirty_degree_l128_128274

theorem sin_thirty_degree : Real.sin (30 * Real.pi / 180) = 1 / 2 := 
by
  sorry

end sin_thirty_degree_l128_128274


namespace cos_135_degree_l128_128977

theorem cos_135_degree : 
  let Q := (Complex.exp (135 * Real.pi / 180 * Complex.I)) in 
  Complex.re Q = -Real.sqrt 2 / 2 :=
by sorry

end cos_135_degree_l128_128977


namespace remainder_when_3n_plus_2_squared_divided_by_11_l128_128001

theorem remainder_when_3n_plus_2_squared_divided_by_11 (n : ℕ) (h : n % 7 = 5) : ((3 * n + 2)^2) % 11 = 3 :=
  sorry

end remainder_when_3n_plus_2_squared_divided_by_11_l128_128001


namespace probability_allison_greater_l128_128244

open Probability

-- Definitions for the problem
noncomputable def die_A : ℕ := 4

noncomputable def die_B : PMF ℕ :=
  PMF.uniform_of_fin (fin 6) -- Could also explicitly write as PMF.of_list [(1, 1/6), ..., (6, 1/6)]

noncomputable def die_N : PMF ℕ :=
  PMF.of_list [(3, 3/6), (5, 3/6)]

-- The target event: Allison's roll > Brian's roll and Noah's roll
noncomputable def event (roll_A : ℕ) (roll_B roll_N : PMF ℕ) :=
  ∀ b n, b ∈ [1, 2, 3] → n ∈ [3] → roll_A > b ∧ roll_A > n

-- The probability calculation
theorem probability_allison_greater :
  (∑' (b : ℕ) (h_b : b < 4) (n : ℕ) (h_n : n = 3), die_B b * die_N n)
  = 1 / 4 :=
by
  -- assumed rolls
  let roll_A := 4
  let prob_B := 1 / 2
  let prob_N := 1 / 2
  
  -- skip proof, but assert the correct result.
  sorry

end probability_allison_greater_l128_128244


namespace find_S_l128_128414

theorem find_S (a b : ℝ) (R : ℝ) (S : ℝ)
  (h1 : a + b = R) 
  (h2 : a^2 + b^2 = 12)
  (h3 : R = 2)
  (h4 : S = a^3 + b^3) : S = 32 :=
by
  sorry

end find_S_l128_128414


namespace prob_leq_zero_l128_128141

noncomputable def normal_distribution (μ σ : ℝ) := measure_theory.probability_measure (measure_theory.gaussian μ σ)

variables {σ : ℝ} (ξ : ℝ → ℝ) (h₁ : ∀ x, ξ x = measure_theory.expectation (measure_theory.gaussian 2 σ)) (h₂ : measure_theory.measure (set.Iic ξ 4) = 0.84)

theorem prob_leq_zero : measure_theory.measure (set.Iic ξ 0) = 0.16 :=
  sorry

end prob_leq_zero_l128_128141


namespace math_problem_l128_128016

-- Definitions of the conditions
variable (x y : ℝ)
axiom h1 : x + y = 5
axiom h2 : x * y = 3

-- Prove the desired equality
theorem math_problem : x + (x^4 / y^3) + (y^4 / x^3) + y = 1021 := 
by 
sorry

end math_problem_l128_128016


namespace sin_30_eq_half_l128_128349

theorem sin_30_eq_half
  (Q : ℝ × ℝ)
  (h1 : Q = (cos (π / 6), sin (π / 6)))
  (E : ℝ × ℝ)
  (h2 : E = (Q.1, 0))
  (h3 : ∃ x y z : ℝ, x / 2 = Q.2 ∧ y = Q.1 ∧ sqrt (x^2 + y^2) = z ∧ z = 1) :
  sin (π / 6) = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l128_128349


namespace find_f_neg_2_l128_128585

noncomputable def f : ℝ → ℝ := sorry

-- Condition 1: f is an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Condition 2: f is defined on ℝ
-- This is implicitly handled as f : ℝ → ℝ

-- Condition 3: f(x+2) = -f(x)
def periodic_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2) = -f x

theorem find_f_neg_2 (h₁ : odd_function f) (h₂ : periodic_function f) : f (-2) = 0 :=
  sorry

end find_f_neg_2_l128_128585


namespace remainder_when_multiplied_and_divided_l128_128782

theorem remainder_when_multiplied_and_divided (n k : ℤ) (h : n % 28 = 15) : (2 * n) % 14 = 2 := 
by
  sorry

end remainder_when_multiplied_and_divided_l128_128782


namespace min_value_sin_function_l128_128550

theorem min_value_sin_function (α β : ℝ) (h : -5 * (Real.sin α) ^ 2 + (Real.sin β) ^ 2 = 3 * Real.sin α) :
  ∃ x : ℝ, x = Real.sin α ∧ (Real.sin α) ^ 2 + (Real.sin β) ^ 2 = 0 :=
sorry

end min_value_sin_function_l128_128550


namespace cos_135_eq_neg_sqrt2_div_2_l128_128986

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = - Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l128_128986


namespace angle_between_a_b_correct_m_l128_128583

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V) (m : ℝ)
variables (OA OB OC : V)

/-- Given conditions as hypotheses --/
def conditions :=
  non_collinear a b ∧ 
  OA = m • a - b ∧ 
  OB = (m + 1) • a + b ∧ 
  OC = a - 3 • b

/-- Question 1: Given additional conditions, prove the angle between a and b --/
theorem angle_between_a_b 
  (h : conditions a b m OA OB OC) 
  (hm : m = -1 / 2) 
  (ha : ∥a∥ = 2 * real.sqrt 2 * ∥b∥) 
  (h_ortho : inner_product_space.dot_product OB OC = 0) : 
  real.angle a b = real.pi / 4 :=
sorry

/-- Question 2: Given collinearity condition, prove the value of m --/
theorem correct_m 
  (h : conditions a b m OA OB OC) 
  (h_collinear : collinear ℝ ![OA, OB, OC]) :
  m = 2 :=
sorry

end angle_between_a_b_correct_m_l128_128583


namespace number_made_l128_128790

theorem number_made (x y : ℕ) (h1 : x + y = 24) (h2 : x = 11) : 7 * x + 5 * y = 142 := by
  sorry

end number_made_l128_128790


namespace arithmetic_sequence_y_value_l128_128537

theorem arithmetic_sequence_y_value (y : ℝ) (h₁ : 2 * y - 3 = -5 * y + 11) : y = 2 := by
  sorry

end arithmetic_sequence_y_value_l128_128537


namespace solve_for_q_l128_128022

variable (k h q : ℝ)

-- Conditions given in the problem
axiom cond1 : (3 / 4) = (k / 48)
axiom cond2 : (3 / 4) = ((h + 36) / 60)
axiom cond3 : (3 / 4) = ((q - 9) / 80)

-- Our goal is to state that q = 69
theorem solve_for_q : q = 69 :=
by
  -- the proof goes here
  sorry

end solve_for_q_l128_128022


namespace bill_experience_l128_128518

theorem bill_experience (B J : ℕ) (h1 : J - 5 = 3 * (B - 5)) (h2 : J = 2 * B) : B = 10 :=
by
  sorry

end bill_experience_l128_128518


namespace fraction_of_white_surface_area_l128_128098

def larger_cube_edge : ℕ := 4
def number_of_smaller_cubes : ℕ := 64
def number_of_white_cubes : ℕ := 8
def number_of_red_cubes : ℕ := 56
def total_surface_area : ℕ := 6 * (larger_cube_edge * larger_cube_edge)
def minimized_white_surface_area : ℕ := 7

theorem fraction_of_white_surface_area :
  minimized_white_surface_area % total_surface_area = 7 % 96 :=
by
  sorry

end fraction_of_white_surface_area_l128_128098


namespace find_m_minus_n_l128_128134

theorem find_m_minus_n (m n : ℤ) (h1 : |m| = 14) (h2 : |n| = 23) (h3 : m + n > 0) : m - n = -9 ∨ m - n = -37 := 
sorry

end find_m_minus_n_l128_128134


namespace total_porridge_l128_128506

variable {c1 c2 c3 c4 c5 c6 : ℝ}

theorem total_porridge (h1 : c3 = c1 + c2)
                      (h2 : c4 = c2 + c3)
                      (h3 : c5 = c3 + c4)
                      (h4 : c6 = c4 + c5)
                      (h5 : c5 = 10) :
                      c1 + c2 + c3 + c4 + c5 + c6 = 40 := 
by
  sorry

end total_porridge_l128_128506


namespace num_integers_satisfying_conditions_l128_128144

theorem num_integers_satisfying_conditions : 
  ∃ n : ℕ, 
    (120 < n) ∧ (n < 250) ∧ (n % 5 = n % 7) :=
sorry

axiom num_integers_with_conditions : ℕ
@[simp] lemma val_num_integers_with_conditions : num_integers_with_conditions = 25 :=
sorry

end num_integers_satisfying_conditions_l128_128144


namespace negation_of_exists_prop_l128_128589

variable (n : ℕ)

theorem negation_of_exists_prop :
  (¬ (∃ n : ℕ, n^2 > 2^n)) ↔ (∀ n : ℕ, n^2 ≤ 2^n) :=
by
  sorry

end negation_of_exists_prop_l128_128589


namespace count_sequences_of_length_15_l128_128438

def countingValidSequences (n : ℕ) : ℕ := sorry

theorem count_sequences_of_length_15 :
  countingValidSequences 15 = 266 :=
  sorry

end count_sequences_of_length_15_l128_128438


namespace right_triangle_distance_l128_128855

theorem right_triangle_distance (x h d : ℝ) :
  x + Real.sqrt ((x + 2 * h) ^ 2 + d ^ 2) = 2 * h + d → 
  x = (h * d) / (2 * h + d) :=
by
  intros h_eq_d
  sorry

end right_triangle_distance_l128_128855


namespace sin_30_eq_half_l128_128311

noncomputable def P : ℝ × ℝ := 
  have angle := real.pi / 6
  have x := real.cos angle
  have y := real.sin angle
  (x, y)  -- The point on the unit circle at 30 degrees

theorem sin_30_eq_half : real.sin (real.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l128_128311


namespace sum_of_digits_N_l128_128164

-- Define a function to compute the least common multiple (LCM) of a list of numbers
def lcm_list (xs : List ℕ) : ℕ :=
  xs.foldr Nat.lcm 1

-- The set of numbers less than 8
def nums : List ℕ := [1, 2, 3, 4, 5, 6, 7]

-- The LCM of numbers less than 8
def N_lcm : ℕ := lcm_list nums

-- The second smallest positive integer that is divisible by every positive integer less than 8
def N : ℕ := 2 * N_lcm

-- Function to compute the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

-- Prove that the sum of the digits of N is 12
theorem sum_of_digits_N : sum_of_digits N = 12 :=
by
  -- Necessary proof steps will be filled here
  sorry

end sum_of_digits_N_l128_128164


namespace sin_of_30_degrees_l128_128343

-- Define P and D and their relationships on a unit circle
def P : ℝ × ℝ := (cos (π / 6), sin (π / 6))  -- Point P on the unit circle at 30 degrees
def D : ℝ × ℝ := (P.1, 0)  -- Foot of the altitude to the x-axis

theorem sin_of_30_degrees : sin (π / 6) = 1 / 2 :=
by 
  -- Use the fact that the triangle P,O,D is a 30-60-90 triangle
  -- and the unit circle properties.
  sorry

end sin_of_30_degrees_l128_128343


namespace simplify_fraction_rationalize_denominator_l128_128470

theorem simplify_fraction_rationalize_denominator :
  (5 : ℝ) / (Real.sqrt 50 + 3 * Real.sqrt 8 + Real.sqrt 18) = 5 * Real.sqrt 2 / 28 :=
by
  have sqrt_50 : Real.sqrt 50 = 5 * Real.sqrt 2 := sorry
  have sqrt_8 : 3 * Real.sqrt 8 = 6 * Real.sqrt 2 := sorry
  have sqrt_18 : Real.sqrt 18 = 3 * Real.sqrt 2 := sorry
  sorry

end simplify_fraction_rationalize_denominator_l128_128470


namespace sin_30_eq_one_half_l128_128369

def Q_coords (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

theorem sin_30_eq_one_half :
  let Q := Q_coords (Real.pi / 6)
  Q.2 = 1 / 2 :=
by
  sorry

end sin_30_eq_one_half_l128_128369


namespace weight_of_b_l128_128750

theorem weight_of_b (a b c d : ℝ)
  (h1 : a + b + c + d = 160)
  (h2 : a + b = 50)
  (h3 : b + c = 56)
  (h4 : c + d = 64) :
  b = 46 :=
by sorry

end weight_of_b_l128_128750


namespace sin_30_eq_half_l128_128272
open real

noncomputable def sin_30_deg : ℝ :=
  sin (π / 6)

theorem sin_30_eq_half : sin_30_deg = 1 / 2 :=
by
  -- Given the conditions of the problem, we need to prove that sin(π / 6) = 1 / 2
  sorry

end sin_30_eq_half_l128_128272


namespace sin_30_eq_half_l128_128262

theorem sin_30_eq_half : Real.sin (30 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_eq_half_l128_128262


namespace floor_plus_x_eq_17_over_4_l128_128399

theorem floor_plus_x_eq_17_over_4
  (x : ℚ)
  (h : ⌊x⌋ + x = 17 / 4)
  : x = 9 / 4 :=
sorry

end floor_plus_x_eq_17_over_4_l128_128399


namespace area_percentage_of_smaller_square_l128_128240

theorem area_percentage_of_smaller_square 
  (radius : ℝ)
  (a A O B: ℝ)
  (side_length_larger_square side_length_smaller_square : ℝ) 
  (hyp1 : side_length_larger_square = 4)
  (hyp2 : radius = 2 * Real.sqrt 2)
  (hyp3 : a = 4) 
  (hyp4 : A = 2 + side_length_smaller_square / 4)
  (hyp5 : O = 2 * Real.sqrt 2)
  (hyp6 : side_length_smaller_square = 0.8) :
  (side_length_smaller_square^2 / side_length_larger_square^2) = 0.04 :=
by
  sorry

end area_percentage_of_smaller_square_l128_128240


namespace no_solution_eq_l128_128699

theorem no_solution_eq (m : ℝ) : 
  (∀ x : ℝ, x ≠ 3 → ((3 - 2 * x) / (x - 3) + (2 + m * x) / (3 - x) = -1) → (m = -1)) :=
by
  sorry

end no_solution_eq_l128_128699


namespace divides_expression_l128_128014

theorem divides_expression (n : ℕ) : 7 ∣ (3^(12 * n^2 + 1) + 2^(6 * n + 2)) := sorry

end divides_expression_l128_128014


namespace mary_marbles_l128_128591

theorem mary_marbles (total_marbles joan_marbles mary_marbles : ℕ) 
  (h1 : total_marbles = 12) 
  (h2 : joan_marbles = 3) 
  (h3 : total_marbles = joan_marbles + mary_marbles) : 
  mary_marbles = 9 := 
by
  rw [h1, h2, add_comm] at h3
  linarith

end mary_marbles_l128_128591


namespace sum_of_drawn_vegetable_oil_and_fruits_vegetables_l128_128231

-- Definitions based on conditions
def varieties_of_grains : ℕ := 40
def varieties_of_vegetable_oil : ℕ := 10
def varieties_of_animal_products : ℕ := 30
def varieties_of_fruits_vegetables : ℕ := 20
def total_sample_size : ℕ := 20

def sampling_fraction : ℚ := total_sample_size / (varieties_of_grains + varieties_of_vegetable_oil + varieties_of_animal_products + varieties_of_fruits_vegetables)

def expected_drawn_vegetable_oil : ℚ := varieties_of_vegetable_oil * sampling_fraction
def expected_drawn_fruits_vegetables : ℚ := varieties_of_fruits_vegetables * sampling_fraction

-- The theorem to be proved
theorem sum_of_drawn_vegetable_oil_and_fruits_vegetables : 
  expected_drawn_vegetable_oil + expected_drawn_fruits_vegetables = 6 := 
by 
  -- Placeholder for proof
  sorry

end sum_of_drawn_vegetable_oil_and_fruits_vegetables_l128_128231


namespace sin_30_eq_half_l128_128283

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the point Q at 30 degrees on the unit circle
def Q : ℝ × ℝ := unit_circle (Real.pi / 6)

theorem sin_30_eq_half : (Real.sin (Real.pi / 6)) = 1 / 2 :=
by sorry

end sin_30_eq_half_l128_128283


namespace find_set_M_l128_128870

variable (U M : Set ℕ)
variable [DecidableEq ℕ]

-- Universel set U is {1, 3, 5, 7}
def universal_set : Set ℕ := {1, 3, 5, 7}

-- define the complement C_U M
def complement (U M : Set ℕ) : Set ℕ := U \ M

-- M is the set to find such that complement of M in U is {5, 7}
theorem find_set_M (M : Set ℕ) (h : complement universal_set M = {5, 7}) : M = {1, 3} := by
  sorry

end find_set_M_l128_128870


namespace find_x_eq_nine_fourths_l128_128384

theorem find_x_eq_nine_fourths (x n : ℚ) (β : ℚ) (h1 : x = n + β) (h2 : n = ⌊x⌋) (h3 : ⌊x⌋ + x = 17 / 4) : x = 9 / 4 :=
by
  sorry

end find_x_eq_nine_fourths_l128_128384


namespace current_average_age_of_seven_persons_l128_128923

theorem current_average_age_of_seven_persons (T : ℕ)
  (h1 : T + 12 = 6 * 43)
  (h2 : 69 = 69)
  : (T + 69) / 7 = 45 := by
  sorry

end current_average_age_of_seven_persons_l128_128923


namespace pool_water_left_l128_128117

theorem pool_water_left 
  (h1_rate: ℝ) (h1_time: ℝ)
  (h2_rate: ℝ) (h2_time: ℝ)
  (h4_rate: ℝ) (h4_time: ℝ)
  (leak_loss: ℝ)
  (h1_rate_eq: h1_rate = 8)
  (h1_time_eq: h1_time = 1)
  (h2_rate_eq: h2_rate = 10)
  (h2_time_eq: h2_time = 2)
  (h4_rate_eq: h4_rate = 14)
  (h4_time_eq: h4_time = 1)
  (leak_loss_eq: leak_loss = 8) :
  (h1_rate * h1_time) + (h2_rate * h2_time) + (h2_rate * h2_time) + (h4_rate * h4_time) - leak_loss = 34 :=
by
  rw [h1_rate_eq, h1_time_eq, h2_rate_eq, h2_time_eq, h4_rate_eq, h4_time_eq, leak_loss_eq]
  norm_num
  sorry

end pool_water_left_l128_128117


namespace total_fruits_l128_128519

-- Define the given conditions
variable (a o : ℕ)
variable (ratio : a = 2 * o)
variable (half_apples_to_ann : a / 2 - 3 = 4)
variable (apples_to_cassie : a - a / 2 - 3 = 0)
variable (oranges_kept : 5 = o - 3)

theorem total_fruits (a o : ℕ) (ratio : a = 2 * o) 
  (half_apples_to_ann : a / 2 - 3 = 4) 
  (apples_to_cassie : a - a / 2 - 3 = 0) 
  (oranges_kept : 5 = o - 3) : a + o = 21 := 
sorry

end total_fruits_l128_128519


namespace estimate_sqrt_diff_l128_128660

-- Defining approximate values for square roots
def approx_sqrt_90 : ℝ := 9.5
def approx_sqrt_88 : ℝ := 9.4

-- Main statement
theorem estimate_sqrt_diff : |(approx_sqrt_90 - approx_sqrt_88) - 0.10| < 0.01 := by
  sorry

end estimate_sqrt_diff_l128_128660


namespace sin_30_eq_half_l128_128310

noncomputable def P : ℝ × ℝ := 
  have angle := real.pi / 6
  have x := real.cos angle
  have y := real.sin angle
  (x, y)  -- The point on the unit circle at 30 degrees

theorem sin_30_eq_half : real.sin (real.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l128_128310


namespace ashwin_rental_hours_l128_128779

theorem ashwin_rental_hours (x : ℕ) 
  (h1 : 25 + 10 * x = 125) : 1 + x = 11 :=
by
  sorry

end ashwin_rental_hours_l128_128779


namespace sum_of_positive_divisors_of_36_l128_128079

theorem sum_of_positive_divisors_of_36 : 
  ( ∑ d in (finset.filter (λ x, 36 % x = 0) (finset.range (36 + 1))), d) = 91 :=
by
  sorry

end sum_of_positive_divisors_of_36_l128_128079


namespace cos_135_eq_neg_inv_sqrt2_l128_128953

theorem cos_135_eq_neg_inv_sqrt2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt2_l128_128953


namespace intersection_A_B_l128_128175

def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 2}
def intersection (S T : Set ℝ) : Set ℝ := {x | x ∈ S ∧ x ∈ T}

theorem intersection_A_B :
  (A ∩ B) = {x | 0 < x ∧ x ≤ 2} := by
  sorry

end intersection_A_B_l128_128175


namespace N_8_12_eq_288_l128_128203

-- Definitions for various polygonal numbers
def N3 (n : ℕ) : ℕ := n * (n + 1) / 2
def N4 (n : ℕ) : ℕ := n^2
def N5 (n : ℕ) : ℕ := 3 * n^2 / 2 - n / 2
def N6 (n : ℕ) : ℕ := 2 * n^2 - n

-- General definition conjectured
def N (n k : ℕ) : ℕ := (k - 2) * n^2 / 2 + (4 - k) * n / 2

-- The problem statement to prove N(8, 12) == 288
theorem N_8_12_eq_288 : N 8 12 = 288 := by
  -- We would need the proofs for the definitional equalities and calculation here
  sorry

end N_8_12_eq_288_l128_128203


namespace sin_30_deg_l128_128304

theorem sin_30_deg : real.sin (30 * real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_deg_l128_128304


namespace numeric_value_of_BAR_l128_128944

variable (b a t c r : ℕ)

-- Conditions from the problem
axiom h1 : b + a + t = 6
axiom h2 : c + a + t = 8
axiom h3 : c + a + r = 12

-- Required to prove
theorem numeric_value_of_BAR : b + a + r = 10 :=
by
  -- Proof goes here
  sorry

end numeric_value_of_BAR_l128_128944


namespace find_number_l128_128151

theorem find_number (n x : ℤ) (h1 : n * x + 3 = 10 * x - 17) (h2 : x = 4) : n = 5 :=
by
  sorry

end find_number_l128_128151


namespace carl_max_rocks_value_l128_128491

/-- 
Carl finds rocks of three different types:
  - 6-pound rocks worth $18 each.
  - 3-pound rocks worth $9 each.
  - 2-pound rocks worth $3 each.
There are at least 15 rocks available for each type.
Carl can carry at most 20 pounds.

Prove that the maximum value, in dollars, of the rocks Carl can carry out of the cave is $57.
-/
theorem carl_max_rocks_value : 
  (∃ x y z : ℕ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ 6 * x + 3 * y + 2 * z ≤ 20 ∧ 18 * x + 9 * y + 3 * z = 57) :=
sorry

end carl_max_rocks_value_l128_128491


namespace increase_in_circumference_by_2_cm_l128_128881

noncomputable def radius_increase_by_two (r : ℝ) : ℝ := r + 2
noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

theorem increase_in_circumference_by_2_cm (r : ℝ) : 
    circumference (radius_increase_by_two r) - circumference r = 12.56 :=
by sorry

end increase_in_circumference_by_2_cm_l128_128881


namespace quadratic_inequality_solution_l128_128616

theorem quadratic_inequality_solution (a : ℝ) :
  (¬ ∀ x : ℝ, a * x^2 - 2 * a * x + 3 > 0) ↔ (a < 0 ∨ a ≥ 3) :=
sorry

end quadratic_inequality_solution_l128_128616


namespace checkout_speed_ratio_l128_128940

theorem checkout_speed_ratio (n x y : ℝ) 
  (h1 : 40 * x = 20 * y + n)
  (h2 : 36 * x = 12 * y + n) : 
  x = 2 * y := 
sorry

end checkout_speed_ratio_l128_128940


namespace total_travel_time_l128_128242

theorem total_travel_time (x y : ℝ) : 
  (x / 50 + y / 70 + 0.5) = (7 * x + 5 * y) / 350 + 0.5 :=
by
  sorry

end total_travel_time_l128_128242


namespace sum_of_factors_36_l128_128081

theorem sum_of_factors_36 : (∑ (i in Finset.filter (λ x, 36 % x = 0) (Finset.range 37)), i) = 91 :=
by
  -- Proof here
  sorry

end sum_of_factors_36_l128_128081


namespace largest_angle_in_ratio_3_4_5_l128_128613

theorem largest_angle_in_ratio_3_4_5 : ∃ (A B C : ℝ), (A / 3 = B / 4 ∧ B / 4 = C / 5) ∧ (A + B + C = 180) ∧ (C = 75) :=
by
  sorry

end largest_angle_in_ratio_3_4_5_l128_128613


namespace volume_tetrahedron_l128_128015

variables (AB AC AD : ℝ) (β γ D : ℝ)
open Real

/-- Prove that the volume of tetrahedron ABCD is equal to 
    (AB * AC * AD * sin β * sin γ * sin D) / 6,
    where β and γ are the plane angles at vertex A opposite to edges AB and AC, 
    and D is the dihedral angle at edge AD. 
-/
theorem volume_tetrahedron (h₁: β ≠ 0) (h₂: γ ≠ 0) (h₃: D ≠ 0):
  (AB * AC * AD * sin β * sin γ * sin D) / 6 =
    abs (AB * AC * AD * sin β * sin γ * sin D) / 6 :=
by sorry

end volume_tetrahedron_l128_128015


namespace rate_of_painting_per_sq_m_l128_128612

def length_of_floor : ℝ := 18.9999683334125
def total_cost : ℝ := 361
def ratio_of_length_to_breadth : ℝ := 3

theorem rate_of_painting_per_sq_m :
  ∃ (rate : ℝ), rate = 3 :=
by
  let B := length_of_floor / ratio_of_length_to_breadth
  let A := length_of_floor * B
  let rate := total_cost / A
  use rate
  sorry  -- Skipping proof as instructed

end rate_of_painting_per_sq_m_l128_128612


namespace no_three_perfect_squares_l128_128174

theorem no_three_perfect_squares (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ¬(∃ k₁ k₂ k₃ : ℕ, k₁^2 = a^2 + b + c ∧ k₂^2 = b^2 + c + a ∧ k₃^2 = c^2 + a + b) :=
sorry

end no_three_perfect_squares_l128_128174


namespace value_of_a_l128_128770

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem value_of_a (a : ℝ) :
  (1 / log_base 2 a) + (1 / log_base 3 a) + (1 / log_base 4 a) + (1 / log_base 5 a) = 7 / 4 ↔
  a = 120 ^ (4 / 7) :=
by
  sorry

end value_of_a_l128_128770


namespace Mina_has_2_25_cent_coins_l128_128008

def MinaCoinProblem : Prop :=
  ∃ (x y z : ℕ), -- number of 5-cent, 10-cent, and 25-cent coins
  x + y + z = 15 ∧
  (74 - 4 * x - 3 * y = 30) ∧ -- corresponds to 30 different values can be obtained
  z = 2

theorem Mina_has_2_25_cent_coins : MinaCoinProblem :=
by 
  sorry

end Mina_has_2_25_cent_coins_l128_128008


namespace goods_train_crossing_time_l128_128929

def speed_kmh : ℕ := 72
def train_length_m : ℕ := 230
def platform_length_m : ℕ := 290

noncomputable def crossing_time_seconds (speed_kmh train_length_m platform_length_m : ℕ) : ℕ :=
  let distance_m := train_length_m + platform_length_m
  let speed_ms := speed_kmh * 1000 / 3600
  distance_m / speed_ms

theorem goods_train_crossing_time :
  crossing_time_seconds speed_kmh train_length_m platform_length_m = 26 :=
by
  -- The proof should be filled in here
  sorry

end goods_train_crossing_time_l128_128929


namespace max_value_vector_sum_l128_128836

theorem max_value_vector_sum (α β : ℝ) :
  let a := (Real.cos α, Real.sin α)
  let b := (Real.sin β, -Real.cos β)
  |(a.1 + b.1, a.2 + b.2)| ≤ 2 := by
  sorry

end max_value_vector_sum_l128_128836


namespace units_digit_six_l128_128769

theorem units_digit_six (n : ℕ) (h : n > 0) : (6 ^ n) % 10 = 6 :=
by sorry

example : (6 ^ 7) % 10 = 6 :=
units_digit_six 7 (by norm_num)

end units_digit_six_l128_128769


namespace nonneg_real_inequality_l128_128463

theorem nonneg_real_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  a^4 + b^4 + c^4 - 2 * (a^2 * b^2 + a^2 * c^2 + b^2 * c^2) + a^2 * b * c + b^2 * a * c + c^2 * a * b ≥ 0 := 
by
  sorry

end nonneg_real_inequality_l128_128463


namespace cody_steps_l128_128252

theorem cody_steps (S steps_week1 steps_week2 steps_week3 steps_week4 total_steps_4weeks : ℕ) 
  (h1 : steps_week1 = 7 * S) 
  (h2 : steps_week2 = 7 * (S + 1000)) 
  (h3 : steps_week3 = 7 * (S + 2000)) 
  (h4 : steps_week4 = 7 * (S + 3000)) 
  (h5 : total_steps_4weeks = steps_week1 + steps_week2 + steps_week3 + steps_week4) 
  (h6 : total_steps_4weeks = 70000) : 
  S = 1000 := 
    sorry

end cody_steps_l128_128252


namespace intersection_unique_one_point_l128_128133

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 7 * x + a
noncomputable def g (x : ℝ) : ℝ := -3 * x^2 + 5 * x - 6

theorem intersection_unique_one_point (a : ℝ) :
  (∃ x y, y = f x a ∧ y = g x) ↔ a = 3 := by
  sorry

end intersection_unique_one_point_l128_128133


namespace largest_divisor_of_m_l128_128630

theorem largest_divisor_of_m (m : ℕ) (h1 : m > 0) (h2 : ∃ k : ℕ, m^3 = 847 * k) : ∃ d : ℕ, d = 77 ∧ ∀ x : ℕ, x > d → ¬ (x ∣ m) :=
sorry

end largest_divisor_of_m_l128_128630


namespace find_c_quadratic_solution_l128_128138

theorem find_c_quadratic_solution (c : ℝ) :
  (Polynomial.eval (-5) (Polynomial.C (-45) + Polynomial.X * Polynomial.C c + Polynomial.X^2) = 0) →
  c = -4 :=
by 
  intros h
  sorry

end find_c_quadratic_solution_l128_128138


namespace salary_of_N_l128_128495

theorem salary_of_N (total_salary : ℝ) (percent_M_from_N : ℝ) (N_salary : ℝ) : 
  (percent_M_from_N * N_salary + N_salary = total_salary) → (N_salary = 280) :=
by
  sorry

end salary_of_N_l128_128495


namespace opponent_score_l128_128722

theorem opponent_score (s g c total opponent : ℕ)
  (h1 : s = 20)
  (h2 : g = 2 * s)
  (h3 : c = 2 * g)
  (h4 : total = s + g + c)
  (h5 : total - 55 = opponent) :
  opponent = 85 := by
  sorry

end opponent_score_l128_128722


namespace drink_total_amount_l128_128858

theorem drink_total_amount (parts_coke parts_sprite parts_mountain_dew ounces_coke total_parts : ℕ)
  (h1 : parts_coke = 2) (h2 : parts_sprite = 1) (h3 : parts_mountain_dew = 3)
  (h4 : total_parts = parts_coke + parts_sprite + parts_mountain_dew)
  (h5 : ounces_coke = 6) :
  ( ounces_coke * total_parts ) / parts_coke = 18 :=
by
  sorry

end drink_total_amount_l128_128858


namespace volume_of_cone_l128_128154

noncomputable def cone_volume (lateral_area : ℝ) (angle: ℝ): ℝ :=
let r := lateral_area / (20 * Mathlib.pi * Math.cos angle) in
let l := r * (Math.tan angle) in
let h := Math.sqrt (l^2 - r^2) in
(1/3) * Mathlib.pi * r^2 * h

theorem volume_of_cone (lateral_area : ℝ) (angle: ℝ) (h_angle: angle = Mathlib.arccos (4/5)) 
(h_lateral_area: lateral_area = 20 * Mathlib.pi): cone_volume lateral_area angle = 16 * Mathlib.pi :=
by
  sorry

end volume_of_cone_l128_128154


namespace binom_inequality_l128_128829

-- Defining the conditions as non-computable functions
def is_nonneg_integer := ℕ

-- Defining the binomial coefficient function
noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

-- The statement of the theorem
theorem binom_inequality (n k h : ℕ) (hn : n ≥ k + h) : binom n (k + h) ≥ binom (n - k) h :=
  sorry

end binom_inequality_l128_128829


namespace small_z_value_l128_128161

noncomputable def w (n : ℕ) := n
noncomputable def x (n : ℕ) := n + 1
noncomputable def y (n : ℕ) := n + 2
noncomputable def z (n : ℕ) := n + 4

theorem small_z_value (n : ℕ) 
  (h : w n ^ 3 + x n ^ 3 + y n ^ 3 = z n ^ 3)
  : z n = 9 :=
sorry

end small_z_value_l128_128161


namespace value_of_z_l128_128146

theorem value_of_z (x y z : ℝ) (h1 : x + y = 6) (h2 : z^2 = x * y - 9) : z = 0 :=
by
  sorry

end value_of_z_l128_128146


namespace correct_operations_l128_128251

variable (x : ℚ)

def incorrect_equation := ((x - 5) * 3) / 7 = 10

theorem correct_operations :
  incorrect_equation x → (3 * x - 5) / 7 = 80 / 7 :=
by
  intro h
  sorry

end correct_operations_l128_128251


namespace price_difference_is_99_cents_l128_128515

-- Definitions for the conditions
def list_price : ℚ := 3996 / 100
def discount_super_savers : ℚ := 9
def discount_penny_wise : ℚ := 25 / 100 * list_price

-- Sale prices calculated based on the given conditions
def sale_price_super_savers : ℚ := list_price - discount_super_savers
def sale_price_penny_wise : ℚ := list_price - discount_penny_wise

-- Difference in prices
def price_difference : ℚ := sale_price_super_savers - sale_price_penny_wise

-- Prove that the price difference in cents is 99
theorem price_difference_is_99_cents : price_difference = 99 / 100 := 
by
  sorry

end price_difference_is_99_cents_l128_128515


namespace total_over_or_underweight_is_8kg_total_selling_price_is_5384_point_8_yuan_l128_128215

-- Definitions based on conditions
def standard_weight : ℝ := 25
def weight_diffs : List ℝ := [-3, -2, -2, -2, -2, -1.5, -1.5, 0, 0, 0, 1, 1, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]
def price_per_kg : ℝ := 10.6

-- Problem 1
theorem total_over_or_underweight_is_8kg :
  (weight_diffs.sum = 8) := 
  sorry

-- Problem 2
theorem total_selling_price_is_5384_point_8_yuan :
  (20 * standard_weight + 8) * price_per_kg = 5384.8 :=
  sorry

end total_over_or_underweight_is_8kg_total_selling_price_is_5384_point_8_yuan_l128_128215


namespace meaning_of_implication_l128_128700

theorem meaning_of_implication (p q : Prop) : (p → q) = ((p → q) = True) :=
sorry

end meaning_of_implication_l128_128700


namespace sin_30_eq_half_l128_128346

theorem sin_30_eq_half
  (Q : ℝ × ℝ)
  (h1 : Q = (cos (π / 6), sin (π / 6)))
  (E : ℝ × ℝ)
  (h2 : E = (Q.1, 0))
  (h3 : ∃ x y z : ℝ, x / 2 = Q.2 ∧ y = Q.1 ∧ sqrt (x^2 + y^2) = z ∧ z = 1) :
  sin (π / 6) = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l128_128346


namespace inappropriate_character_choice_l128_128492

-- Definitions and conditions
def is_main_character (c : String) : Prop := 
  c = "Gryphon" ∨ c = "Mock Turtle"

def characters : List String := ["Lobster", "Gryphon", "Mock Turtle"]

-- Theorem statement
theorem inappropriate_character_choice : 
  ¬ is_main_character "Lobster" :=
by 
  sorry

end inappropriate_character_choice_l128_128492


namespace ellipse_focal_length_l128_128126

theorem ellipse_focal_length :
  let a_squared := 20
    let b_squared := 11
    let c := Real.sqrt (a_squared - b_squared)
    let focal_length := 2 * c
  11 * x^2 + 20 * y^2 = 220 →
  focal_length = 6 :=
by
  sorry

end ellipse_focal_length_l128_128126


namespace max_saved_houses_l128_128724

theorem max_saved_houses (n c : ℕ) (h₁ : 1 ≤ c ∧ c ≤ n / 2) : 
  ∃ k, k = n^2 + c^2 - n * c - c :=
by
  sorry

end max_saved_houses_l128_128724


namespace sin_30_deg_l128_128305

theorem sin_30_deg : real.sin (30 * real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_deg_l128_128305


namespace minimum_value_amgm_l128_128868

theorem minimum_value_amgm (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 27) : (a + 3 * b + 9 * c) ≥ 27 :=
by
  sorry

end minimum_value_amgm_l128_128868


namespace sum_of_positive_factors_36_l128_128070

theorem sum_of_positive_factors_36 : (∑ d in (finset.filter (λ n, 36 % n = 0) (finset.range (36 + 1))), d) = 91 :=
sorry

end sum_of_positive_factors_36_l128_128070


namespace find_k_value_l128_128040

theorem find_k_value : 
  (∃ (x y k : ℝ), x = -6.8 ∧ 
  (y = 0.25 * x + 10) ∧ 
  (k = -3 * x + y) ∧ 
  k = 32.1) :=
sorry

end find_k_value_l128_128040


namespace remainder_of_x_pow_77_eq_6_l128_128768

theorem remainder_of_x_pow_77_eq_6 (x : ℤ) (h : x^77 % 7 = 6) : x^77 % 7 = 6 :=
by
  sorry

end remainder_of_x_pow_77_eq_6_l128_128768


namespace correct_option_l128_128775

theorem correct_option (a b c : ℝ) : 
  (5 * a - (b + 2 * c) = 5 * a + b - 2 * c ∨
   5 * a - (b + 2 * c) = 5 * a - b + 2 * c ∨
   5 * a - (b + 2 * c) = 5 * a + b + 2 * c ∨
   5 * a - (b + 2 * c) = 5 * a - b - 2 * c) ↔ 
  (5 * a - (b + 2 * c) = 5 * a - b - 2 * c) :=
by
  sorry

end correct_option_l128_128775


namespace increase_in_tire_radius_l128_128658

theorem increase_in_tire_radius
  (r : ℝ)
  (d1 d2 : ℝ)
  (conv_factor : ℝ)
  (original_radius : r = 16)
  (odometer_reading_outbound : d1 = 500)
  (odometer_reading_return : d2 = 485)
  (conversion_factor : conv_factor = 63360) :
  ∃ Δr : ℝ, Δr = 0.33 :=
by
  sorry

end increase_in_tire_radius_l128_128658


namespace train_journey_duration_l128_128051

variable (z x : ℝ)
variable (h1 : 1.7 = 1 + 42 / 60)
variable (h2 : (0.9 * z / (1.2 * x) + 0.1 * z / (1.25 * x)) = z / x - 1.7)

theorem train_journey_duration (z x : ℝ)
    (h1 : 1.7 = 1 + 42 / 60)
    (h2 : (0.9 * z / (1.2 * x) + 0.1 * z / (1.25 * x)) = z / x - 1.7):
    z / x = 10 := 
by
  sorry

end train_journey_duration_l128_128051


namespace sin_30_eq_half_l128_128345

theorem sin_30_eq_half
  (Q : ℝ × ℝ)
  (h1 : Q = (cos (π / 6), sin (π / 6)))
  (E : ℝ × ℝ)
  (h2 : E = (Q.1, 0))
  (h3 : ∃ x y z : ℝ, x / 2 = Q.2 ∧ y = Q.1 ∧ sqrt (x^2 + y^2) = z ∧ z = 1) :
  sin (π / 6) = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l128_128345


namespace tom_bought_6_hardcover_l128_128531

-- Given conditions and statements
def toms_books_condition_1 (h p : ℕ) : Prop :=
  h + p = 10

def toms_books_condition_2 (h p : ℕ) : Prop :=
  28 * h + 18 * p = 240

-- The theorem to prove
theorem tom_bought_6_hardcover (h p : ℕ) 
  (h_condition : toms_books_condition_1 h p)
  (c_condition : toms_books_condition_2 h p) : 
  h = 6 :=
sorry

end tom_bought_6_hardcover_l128_128531


namespace sin_thirty_degrees_l128_128333

theorem sin_thirty_degrees : Real.sin (Real.pi / 6) = 1 / 2 := 
by sorry

end sin_thirty_degrees_l128_128333


namespace find_angle_A_l128_128149

theorem find_angle_A (A B C : ℝ)
  (h1 : C = 2 * B)
  (h2 : B = A / 3)
  (h3 : A + B + C = 180) : A = 90 :=
by
  sorry

end find_angle_A_l128_128149


namespace subset_ratio_l128_128831

theorem subset_ratio (S T : ℕ) (hS : S = 256) (hT : T = 56) :
  (T / S : ℚ) = 7 / 32 := by
sorry

end subset_ratio_l128_128831


namespace alcohol_water_ratio_l128_128223

theorem alcohol_water_ratio (alcohol water : ℝ) (h_alcohol : alcohol = 3 / 5) (h_water : water = 2 / 5) :
  alcohol / water = 3 / 2 :=
by 
  sorry

end alcohol_water_ratio_l128_128223


namespace simplify_expression_l128_128877

variable (x : ℝ)

theorem simplify_expression : 3 * x + 4 * x^3 + 2 - (7 - 3 * x - 4 * x^3) = 8 * x^3 + 6 * x - 5 := 
by 
  sorry

end simplify_expression_l128_128877


namespace arithmetic_difference_l128_128844

variables (p q r : ℝ)

theorem arithmetic_difference (h1 : (p + q) / 2 = 10) (h2 : (q + r) / 2 = 27) : r - p = 34 :=
by
  sorry

end arithmetic_difference_l128_128844


namespace sum_of_digits_18_l128_128619

def distinct_digits (A B C D : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

theorem sum_of_digits_18 (A B C D : ℕ) 
(h1 : A + D = 10)
(h2 : B + C + 1 = 10 + D)
(h3 : C + B + 1 = 10 + B)
(h4 : D + A + 1 = 11)
(h_distinct : distinct_digits A B C D) :
  A + B + C + D = 18 :=
sorry

end sum_of_digits_18_l128_128619


namespace red_more_than_yellow_l128_128210

-- Define the total number of marbles
def total_marbles : ℕ := 19

-- Define the number of yellow marbles
def yellow_marbles : ℕ := 5

-- Calculate the number of remaining marbles
def remaining_marbles : ℕ := total_marbles - yellow_marbles

-- Define the ratio of blue to red marbles
def blue_ratio : ℕ := 3
def red_ratio : ℕ := 4

-- Calculate the sum of ratio parts
def sum_ratio : ℕ := blue_ratio + red_ratio

-- Calculate the number of shares per ratio part
def share_per_part : ℕ := remaining_marbles / sum_ratio

-- Calculate the number of red marbles
def red_marbles : ℕ := red_ratio * share_per_part

-- Theorem to prove: the difference between red marbles and yellow marbles is 3
theorem red_more_than_yellow : red_marbles - yellow_marbles = 3 :=
by
  sorry

end red_more_than_yellow_l128_128210


namespace c_put_15_oxen_l128_128243

theorem c_put_15_oxen (x : ℕ):
  (10 * 7 + 12 * 5 + 3 * x = 130 + 3 * x) →
  (175 * 3 * x / (130 + 3 * x) = 45) →
  x = 15 :=
by
  intros h1 h2
  sorry

end c_put_15_oxen_l128_128243


namespace y_comparison_l128_128415

theorem y_comparison :
  let y1 := (-1)^2 - 2*(-1) + 3
  let y2 := (-2)^2 - 2*(-2) + 3
  y2 > y1 := by
  sorry

end y_comparison_l128_128415


namespace find_natural_numbers_l128_128405

theorem find_natural_numbers (n : ℕ) (h : n > 1) : 
  ((n - 1) ∣ (n^3 - 3)) ↔ (n = 2 ∨ n = 3) := 
by 
  sorry

end find_natural_numbers_l128_128405


namespace find_side_length_l128_128723

theorem find_side_length
  (X : ℕ)
  (h1 : 3 + 2 + X + 4 = 12) :
  X = 3 :=
by
  sorry

end find_side_length_l128_128723


namespace smallest_solution_l128_128682

theorem smallest_solution :
  (∃ x : ℝ, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧
  ∀ y : ℝ, (1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4)) → x ≤ y) →
  x = 4 - Real.sqrt 2 :=
sorry

end smallest_solution_l128_128682


namespace find_added_number_l128_128931

theorem find_added_number 
  (initial_number : ℕ)
  (final_result : ℕ)
  (h : initial_number = 8)
  (h_result : 3 * (2 * initial_number + final_result) = 75) : 
  final_result = 9 := by
  sorry

end find_added_number_l128_128931


namespace find_range_OM_l128_128419

noncomputable def distance (a b : ℝ) := (a - b) ^ 2

theorem find_range_OM (m n : ℝ) (hne : ¬ (m = 0 ∧ n = 0)) :
  let line_eq : ℝ → ℝ → Prop := λ x y, 2 * m * x - (4 * m + n) * y + 2 * n = 0 in
  let P : (ℝ × ℝ) := (2, 6) in
  let O : (ℝ × ℝ) := (0, 0) in
  let d := distance (distance O.1 P.1 + distance O.2 P.2) in
  let OM := sqrt ((O.1 - P.1) ^ 2 + (O.2 - P.2) ^ 2) in
  5 - sqrt 5 ≤ OM ∧ OM ≤ 5 + sqrt 5 :=
sorry

end find_range_OM_l128_128419


namespace rice_amount_previously_l128_128614

variables (P X : ℝ) (hP : P > 0) (h : 0.8 * P * 50 = P * X)

theorem rice_amount_previously (hP : P > 0) (h : 0.8 * P * 50 = P * X) : X = 40 := 
by 
  sorry

end rice_amount_previously_l128_128614


namespace ratio_of_x_to_y_l128_128520

variable {x y : ℝ}

theorem ratio_of_x_to_y (h : (12 * x - 5 * y) / (15 * x - 3 * y) = 4 / 7) : x / y = 23 / 24 :=
by
  sorry

end ratio_of_x_to_y_l128_128520


namespace vectors_parallel_iff_m_eq_neg_1_l128_128558

-- Given vectors a and b
def vector_a (m : ℝ) : ℝ × ℝ := (2 * m - 1, m)
def vector_b : ℝ × ℝ := (3, 1)

-- Definition of vectors being parallel
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- The statement to be proven
theorem vectors_parallel_iff_m_eq_neg_1 (m : ℝ) : 
  parallel (vector_a m) vector_b ↔ m = -1 :=
by 
  sorry

end vectors_parallel_iff_m_eq_neg_1_l128_128558


namespace carl_marbles_l128_128650

theorem carl_marbles (initial : ℕ) (lost_frac : ℚ) (additional : ℕ) (gift : ℕ) (lost : ℕ)
  (initial = 12) 
  (lost_frac = 1 / 2)
  (additional = 10)
  (gift = 25)
  (lost = initial * lost_frac) :
  ((initial - lost) + additional + gift = 41) :=
sorry

end carl_marbles_l128_128650


namespace exp_add_exp_nat_mul_l128_128179

noncomputable def Exp (z : ℝ) : ℝ := Real.exp z

theorem exp_add (a b x : ℝ) :
  Exp ((a + b) * x) = Exp (a * x) * Exp (b * x) := sorry

theorem exp_nat_mul (x : ℝ) (k : ℕ) :
  Exp (k * x) = (Exp x) ^ k := sorry

end exp_add_exp_nat_mul_l128_128179


namespace andrei_cannot_ensure_victory_l128_128622

theorem andrei_cannot_ensure_victory :
  ∀ (juice_andrew : ℝ) (juice_masha : ℝ),
    juice_andrew = 24 * 1000 ∧
    juice_masha = 24 * 1000 ∧
    ∀ (andrew_mug : ℝ) (masha_mug1 : ℝ) (masha_mug2 : ℝ),
      andrew_mug = 500 ∧
      masha_mug1 = 240 ∧
      masha_mug2 = 240 ∧
      (¬ (∃ (turns_andrew turns_masha : ℕ), 
        turns_andrew * andrew_mug > 48 * 1000 / 2 ∨
        turns_masha * (masha_mug1 + masha_mug2) > 48 * 1000 / 2)) := sorry

end andrei_cannot_ensure_victory_l128_128622


namespace domain_of_sqrt_l128_128376

theorem domain_of_sqrt (x : ℝ) (h : 2 * x - 3 ≥ 0) : x ≥ 3 / 2 :=
sorry

end domain_of_sqrt_l128_128376


namespace Liz_latest_start_time_l128_128736

noncomputable def latest_start_time (turkey_weight : ℕ) (roast_time_per_pound : ℕ) (number_of_turkeys : ℕ) (dinner_time : Time) : Time :=
  Time.sub dinner_time (
    ((turkey_weight * roast_time_per_pound) * number_of_turkeys) / 60
  )

theorem Liz_latest_start_time : 
  latest_start_time 16 15 2 (Time.mk 18 0) = Time.mk 10 0 := 
by
  sorry

end Liz_latest_start_time_l128_128736


namespace cylinder_ratio_l128_128510

theorem cylinder_ratio (m r : ℝ) (h1 : m + 2 * r = Real.sqrt (m^2 + (r * Real.pi)^2)) :
  m / (2 * r) = (Real.pi^2 - 4) / 8 := by
  sorry

end cylinder_ratio_l128_128510


namespace intersection_of_A_and_B_l128_128431

def A : Set ℕ := {0, 2}
def B : Set ℕ := {1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {2} :=
by
  sorry

end intersection_of_A_and_B_l128_128431


namespace sin_of_30_degrees_l128_128337

-- Define P and D and their relationships on a unit circle
def P : ℝ × ℝ := (cos (π / 6), sin (π / 6))  -- Point P on the unit circle at 30 degrees
def D : ℝ × ℝ := (P.1, 0)  -- Foot of the altitude to the x-axis

theorem sin_of_30_degrees : sin (π / 6) = 1 / 2 :=
by 
  -- Use the fact that the triangle P,O,D is a 30-60-90 triangle
  -- and the unit circle properties.
  sorry

end sin_of_30_degrees_l128_128337


namespace cos_135_eq_neg_sqrt2_div_2_l128_128963

theorem cos_135_eq_neg_sqrt2_div_2 : 
  Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l128_128963


namespace binder_cost_l128_128590

variable (B : ℕ) -- Define B as the cost of each binder

theorem binder_cost :
  let book_cost := 16
  let num_binders := 3
  let notebook_cost := 1
  let num_notebooks := 6
  let total_cost := 28
  (book_cost + num_binders * B + num_notebooks * notebook_cost = total_cost) → (B = 2) :=
by
  sorry

end binder_cost_l128_128590


namespace work_completed_in_30_days_l128_128876

theorem work_completed_in_30_days (ravi_days : ℕ) (prakash_days : ℕ)
  (h1 : ravi_days = 50) (h2 : prakash_days = 75) : 
  let ravi_rate := (1 / 50 : ℚ)
  let prakash_rate := (1 / 75 : ℚ)
  let combined_rate := ravi_rate + prakash_rate
  let days_to_complete := 1 / combined_rate
  days_to_complete = 30 := by
  sorry

end work_completed_in_30_days_l128_128876


namespace sqrt_inequality_l128_128703

theorem sqrt_inequality (n : ℕ) : 
  (n ≥ 0) → (Real.sqrt (n + 2) - Real.sqrt (n + 1) ≤ Real.sqrt (n + 1) - Real.sqrt n) := 
by
  intro h
  sorry

end sqrt_inequality_l128_128703


namespace john_sublets_to_3_people_l128_128861

def monthly_income (n : ℕ) : ℕ := 400 * n
def monthly_cost : ℕ := 900
def annual_profit (n : ℕ) : ℕ := 12 * (monthly_income n - monthly_cost)

theorem john_sublets_to_3_people
  (h1 : forall n : ℕ, monthly_income n - monthly_cost > 0)
  (h2 : annual_profit 3 = 3600) :
  3 = 3 := by
  sorry

end john_sublets_to_3_people_l128_128861


namespace minimum_value_of_z_l128_128416

/-- Given the constraints: 
1. x - y + 5 ≥ 0,
2. x + y ≥ 0,
3. x ≤ 3,

Prove that the minimum value of z = (x + y + 2) / (x + 3) is 1/3.
-/
theorem minimum_value_of_z : 
  ∀ (x y : ℝ), 
    (x - y + 5 ≥ 0) ∧ 
    (x + y ≥ 0) ∧ 
    (x ≤ 3) → 
    ∃ (z : ℝ), 
      z = (x + y + 2) / (x + 3) ∧
      z = 1 / 3 :=
by
  intros x y h
  sorry

end minimum_value_of_z_l128_128416


namespace rowing_upstream_distance_l128_128101

theorem rowing_upstream_distance (b s d : ℝ) (h_stream_speed : s = 5)
    (h_downstream_distance : 60 = (b + s) * 3)
    (h_upstream_time : d = (b - s) * 3) : 
    d = 30 := by
  have h_b : b = 15 := by
    linarith [h_downstream_distance, h_stream_speed]
  rw [h_b, h_stream_speed] at h_upstream_time
  linarith [h_upstream_time]

end rowing_upstream_distance_l128_128101


namespace find_a1_range_a1_l128_128037

variables (a_1 : ℤ) (d : ℤ := -1) (S : ℕ → ℤ)

-- Definition of sum of first n terms in an arithmetic sequence
def arithmetic_sum (n : ℕ) : ℤ := n * (2 * a_1 + (n - 1) * d) / 2

-- Definition of nth term in an arithmetic sequence
def arithmetic_nth_term (n : ℕ) : ℤ := a_1 + (n - 1) * d

-- Given conditions for the problems
axiom S_def : ∀ n, S n = arithmetic_sum a_1 d n

-- Problem 1: Proving a1 = 1 given S_5 = -5
theorem find_a1 (h : S 5 = -5) : a_1 = 1 :=
by
  sorry

-- Problem 2: Proving range of a1 given S_n ≤ a_n for any positive integer n
theorem range_a1 (h : ∀ n : ℕ, n > 0 → S n ≤ arithmetic_nth_term a_1 d n) : a_1 ≤ 0 :=
by
  sorry

end find_a1_range_a1_l128_128037


namespace solution_set_of_inequality_l128_128895

theorem solution_set_of_inequality :
  { x : ℝ | x^2 + x - 2 < 0 } = { x : ℝ | -2 < x ∧ x < 1 } :=
sorry

end solution_set_of_inequality_l128_128895


namespace sin_30_eq_half_l128_128298

theorem sin_30_eq_half : Real.sin (π / 6) = 1 / 2 := 
  sorry

end sin_30_eq_half_l128_128298


namespace cos_135_eq_neg_sqrt_two_div_two_l128_128971

theorem cos_135_eq_neg_sqrt_two_div_two : real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt_two_div_two_l128_128971


namespace sin_30_is_half_l128_128354

noncomputable def sin_30_degrees : ℝ := Real.sin (Real.pi / 6)

theorem sin_30_is_half : sin_30_degrees = 1 / 2 := by
  sorry

end sin_30_is_half_l128_128354


namespace find_x_l128_128390

theorem find_x (x : ℚ) (n : ℤ) (f : ℚ) (h1 : x = n + f) (h2 : n = ⌊x⌋) (h3 : f < 1): 
  ⌊x⌋ + x = 17 / 4 → x = 9 / 4 :=
by 
  sorry

end find_x_l128_128390


namespace parabola_vertex_l128_128609

theorem parabola_vertex :
  ∀ (x : ℝ), (∃ v : ℝ × ℝ, (v.1 = -1 ∧ v.2 = 4) ∧ ∀ (x : ℝ), (x^2 + 2*x + 5 = ((x + 1)^2 + 4))) :=
by
  sorry

end parabola_vertex_l128_128609


namespace polygon_angle_ratio_pairs_count_l128_128487

theorem polygon_angle_ratio_pairs_count :
  ∃ (m n : ℕ), (∃ (k : ℕ), (k > 0) ∧ (180 - 360 / ↑m) / (180 - 360 / ↑n) = 4 / 3
  ∧ Prime n ∧ (m - 6) * (n + 8) = 48 ∧ 
  ∃! (m n : ℕ), (180 - 360 / ↑m = (4 * (180 - 360 / ↑n)) / 3)) :=
sorry  -- Proof omitted, providing only the statement

end polygon_angle_ratio_pairs_count_l128_128487


namespace sequence_general_term_l128_128711

theorem sequence_general_term 
  (a : ℕ → ℝ)
  (h₀ : a 1 = 1)
  (h₁ : a 2 = 1 / 3)
  (h₂ : ∀ n : ℕ, 2 ≤ n → a n * a (n - 1) + a n * a (n + 1) = 2 * a (n - 1) * a (n + 1)) :
  ∀ n : ℕ, 1 ≤ n → a n = 1 / (2 * n - 1) := 
by
  sorry

end sequence_general_term_l128_128711


namespace necessary_condition_l128_128139

theorem necessary_condition (x : ℝ) (h : (x-1) * (x-2) ≤ 0) : x^2 - 3 * x ≤ 0 :=
sorry

end necessary_condition_l128_128139


namespace sin_30_eq_half_l128_128291

theorem sin_30_eq_half : Real.sin (Float.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l128_128291


namespace acute_triangle_B_area_l128_128571

-- Basic setup for the problem statement
variable {A B C : ℝ} -- Angles of the triangle
variable {a b c : ℝ} -- Sides opposite to respective angles

-- The theorem to be proven
theorem acute_triangle_B_area (h_acute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2) 
                              (h_sides : a = 2 * b * Real.sin A)
                              (h_a : a = 3 * Real.sqrt 3) 
                              (h_c : c = 5) : 
  B = π / 6 ∧ (1/2) * a * c * Real.sin B = (15 * Real.sqrt 3) / 4 :=
by
  sorry

end acute_triangle_B_area_l128_128571


namespace sum_b_eq_l128_128834

open BigOperators

-- Given conditions from the problem
def a (n : ℕ) : ℝ := (∑ i in finset.range n, (i + 1) : ℝ) / (n + 1)
def b (n : ℕ) : ℝ := 4 * (1 / n - 1 / (n + 1))

-- Statement to prove the sum of the first n terms of the sequence {b_n}
theorem sum_b_eq (n : ℕ) : (∑ i in finset.range n, b (i + 1)) = 4 * n / (n + 1) :=
sorry

end sum_b_eq_l128_128834


namespace sin_30_eq_half_l128_128290

theorem sin_30_eq_half : Real.sin (Float.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l128_128290


namespace parity_of_f_and_h_l128_128833

-- Define function f
def f (x : ℝ) : ℝ := x^2

-- Define function h
def h (x : ℝ) : ℝ := x

-- Define even and odd function
def even_fun (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x
def odd_fun (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = - g x

-- Theorem statement
theorem parity_of_f_and_h :
  even_fun f ∧ odd_fun h :=
by {
  sorry
}

end parity_of_f_and_h_l128_128833


namespace male_listeners_l128_128100

structure Survey :=
  (males_dont_listen : Nat)
  (females_listen : Nat)
  (total_listeners : Nat)
  (total_dont_listen : Nat)

def number_of_females_dont_listen (s : Survey) : Nat :=
  s.total_dont_listen - s.males_dont_listen

def number_of_males_listen (s : Survey) : Nat :=
  s.total_listeners - s.females_listen

theorem male_listeners (s : Survey) (h : s = { males_dont_listen := 85, females_listen := 75, total_listeners := 180, total_dont_listen := 160 }) :
  number_of_males_listen s = 105 :=
by
  sorry

end male_listeners_l128_128100


namespace cos_135_eq_neg_sqrt2_div_2_l128_128988

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = - Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l128_128988


namespace lion_king_cost_l128_128191

theorem lion_king_cost
  (LK_earned : ℕ := 200) -- The Lion King earned 200 million
  (LK_profit : ℕ := 190) -- The Lion King profit calculated from half of Star Wars' profit
  (SW_cost : ℕ := 25)    -- Star Wars cost 25 million
  (SW_earned : ℕ := 405) -- Star Wars earned 405 million
  (SW_profit : SW_earned - SW_cost = 380) -- Star Wars profit
  (LK_profit_from_SW : LK_profit = 1/2 * (SW_earned - SW_cost)) -- The Lion King profit calculation
  (LK_cost : ℕ := LK_earned - LK_profit) -- The Lion King cost calculation
  : LK_cost = 10 := 
sorry

end lion_king_cost_l128_128191


namespace andy_demerits_l128_128805

theorem andy_demerits (x : ℕ) :
  (∀ x, 6 * x + 15 = 27 → x = 2) :=
by
  intro
  sorry

end andy_demerits_l128_128805


namespace smallest_solution_to_equation_l128_128664

theorem smallest_solution_to_equation :
  ∀ x : ℝ, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) → x = 4 - real.sqrt 2 :=
  sorry

end smallest_solution_to_equation_l128_128664


namespace sequence_G_51_l128_128747

theorem sequence_G_51 :
  ∀ G : ℕ → ℚ, 
  (∀ n : ℕ, G (n + 1) = (3 * G n + 2) / 2) → 
  G 1 = 3 → 
  G 51 = (3^51 + 1) / 2 := by 
  sorry

end sequence_G_51_l128_128747


namespace find_LCM_of_numbers_l128_128900

def HCF (a b : ℕ) : ℕ := sorry  -- A placeholder definition for HCF
def LCM (a b : ℕ) : ℕ := sorry  -- A placeholder definition for LCM

theorem find_LCM_of_numbers (a b : ℕ) 
  (h1 : a + b = 55) 
  (h2 : HCF a b = 5) 
  (h3 : (1 : ℚ) / a + (1 : ℚ) / b = 0.09166666666666666) : 
  LCM a b = 120 := 
by 
  sorry

end find_LCM_of_numbers_l128_128900


namespace find_N_sum_e_l128_128841

theorem find_N_sum_e (N : ℝ) (e1 e2 : ℝ) :
  (2 * abs (2 - e1) = N) ∧
  (2 * abs (2 - e2) = N) ∧
  (e1 ≠ e2) ∧
  (e1 + e2 = 4) →
  N = 0 :=
by
  sorry

end find_N_sum_e_l128_128841


namespace problem_I_problem_II_l128_128426

namespace MathProof

-- Define the function f(x) given m
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 1| - 2 * |x + 1|

-- Problem (I)
theorem problem_I (x : ℝ) : (5 - |x - 1| - 2 * |x + 1| > 2) ↔ (-4/3 < x ∧ x < 0) := 
sorry

-- Define the quadratic function
def y (x : ℝ) : ℝ := x^2 + 2*x + 3

-- Problem (II)
theorem problem_II (m : ℝ) : (∀ x : ℝ, ∃ t : ℝ, t = x^2 + 2*x + 3 ∧ t = f m x) ↔ (m ≥ 4) :=
sorry

end MathProof

end problem_I_problem_II_l128_128426


namespace solve_inequality_l128_128206

theorem solve_inequality (x : ℝ) (h : 1 / (x - 1) < -1) : 0 < x ∧ x < 1 :=
sorry

end solve_inequality_l128_128206


namespace perfect_square_solution_l128_128535

theorem perfect_square_solution (n : ℕ) : ∃ a : ℕ, n * 2^(n+1) + 1 = a^2 ↔ n = 0 ∨ n = 3 := by
  sorry

end perfect_square_solution_l128_128535


namespace time_saved_calculator_l128_128857

-- Define the conditions
def time_with_calculator (n : ℕ) : ℕ := 2 * n
def time_without_calculator (n : ℕ) : ℕ := 5 * n
def total_problems : ℕ := 20

-- State the theorem to prove the time saved is 60 minutes
theorem time_saved_calculator : 
  time_without_calculator total_problems - time_with_calculator total_problems = 60 :=
sorry

end time_saved_calculator_l128_128857


namespace proof_cos_135_degree_l128_128999

noncomputable def cos_135_degree_is_negative_sqrt2_over_2 : Prop :=
  real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2

theorem proof_cos_135_degree : cos_135_degree_is_negative_sqrt2_over_2 :=
by
  sorry

end proof_cos_135_degree_l128_128999


namespace sin_30_eq_half_l128_128363

theorem sin_30_eq_half : Real.sin (Real.pi / 6) = 1 / 2 := sorry

end sin_30_eq_half_l128_128363


namespace find_q_l128_128784

theorem find_q (P J T : ℝ) (Q : ℝ) (q : ℚ) 
  (h1 : J = 0.75 * P)
  (h2 : J = 0.80 * T)
  (h3 : T = P * (1 - Q))
  (h4 : Q = q / 100) :
  q = 6.25 := 
by
  sorry

end find_q_l128_128784


namespace smallest_number_l128_128055

theorem smallest_number (a b c : ℕ) (h1 : b = 29) (h2 : c = b + 7) (h3 : (a + b + c) / 3 = 30) : a = 25 :=
by
  sorry

end smallest_number_l128_128055


namespace smallest_solution_exists_l128_128689

noncomputable def is_solution (x : ℝ) : Prop := (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧ x ≠ 3 ∧ x ≠ 5 ∧ x ≠ 4

-- Statement of the problem without proof
theorem smallest_solution_exists : ∃ (x : ℝ), is_solution x ∧ ∀ (y : ℝ), is_solution y → x ≤ y :=
sorry

end smallest_solution_exists_l128_128689


namespace smallest_solution_l128_128675

theorem smallest_solution (x : ℝ) (h : (1 / (x - 3)) + (1 / (x - 5)) = (4 / (x - 4))) : 
  x = 5 - 2 * Real.sqrt 2 :=
sorry

end smallest_solution_l128_128675


namespace convert_13_to_binary_l128_128382

theorem convert_13_to_binary : (13 : ℕ) = 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 :=
by sorry

end convert_13_to_binary_l128_128382


namespace lcm_of_times_l128_128759

-- Define the times each athlete takes to complete one lap
def time_A : Nat := 4
def time_B : Nat := 5
def time_C : Nat := 6

-- Prove that the LCM of 4, 5, and 6 is 60
theorem lcm_of_times : Nat.lcm time_A (Nat.lcm time_B time_C) = 60 := by
  sorry

end lcm_of_times_l128_128759


namespace smallest_solution_exists_l128_128688

noncomputable def is_solution (x : ℝ) : Prop := (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧ x ≠ 3 ∧ x ≠ 5 ∧ x ≠ 4

-- Statement of the problem without proof
theorem smallest_solution_exists : ∃ (x : ℝ), is_solution x ∧ ∀ (y : ℝ), is_solution y → x ≤ y :=
sorry

end smallest_solution_exists_l128_128688


namespace part1_part2_l128_128143

-- Define the predicate for the inequality
def prop (x m : ℝ) : Prop := x^2 - 2 * m * x - 3 * m^2 < 0

-- Define the set A
def A (m : ℝ) : Prop := m < -2 ∨ m > 2 / 3

-- Define the predicate for the other inequality
def prop_B (x a : ℝ) : Prop := x^2 - 2 * a * x + a^2 - 1 < 0

-- Define the set B in terms of a
def B (x a : ℝ) : Prop := a - 1 < x ∧ x < a + 1

-- Define the propositions required in the problem
theorem part1 (m : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 2 → prop x m) ↔ A m :=
sorry

theorem part2 (a : ℝ) :
  (∀ x, B x a → A x) ∧ (∃ x, A x ∧ ¬ B x a) ↔ (a ≤ -3 ∨ a ≥ 5 / 3) :=
sorry

end part1_part2_l128_128143


namespace sin_30_eq_half_l128_128254

theorem sin_30_eq_half :
  let Q := { (x, y) : ℝ × ℝ // x^2 + y^2 = 1 ∧ y ≠ 0 ∧ 0 ≤ x ∧ x ≤ 1 }
  let OE := (1, 0)
  ∀ (Q : (x, y) : ℝ × ℝ), (Q.1, Q.2) = (√3/2, 1/2) →
  ∃ OE : (x, y) : ℝ × ℝ,
    let OQ := (0, 0)
    let QOE_triangle := { c : ℝ × ℝ // ∃ Q OE OQ, QOE c }
      sin 30° = 1/2 :=
  by
    sorry

end sin_30_eq_half_l128_128254


namespace total_play_time_in_hours_l128_128222

def football_time : ℕ := 60
def basketball_time : ℕ := 60

theorem total_play_time_in_hours : (football_time + basketball_time) / 60 = 2 := by
  sorry

end total_play_time_in_hours_l128_128222


namespace base8_base13_to_base10_sum_l128_128661

-- Definitions for the base 8 and base 13 numbers
def base8_to_base10 (a b c : ℕ) : ℕ := a * 64 + b * 8 + c
def base13_to_base10 (d e f : ℕ) : ℕ := d * 169 + e * 13 + f

-- Constants for the specific numbers in the problem
def num1 := base8_to_base10 5 3 7
def num2 := base13_to_base10 4 12 5

-- The theorem to prove
theorem base8_base13_to_base10_sum : num1 + num2 = 1188 := by
  sorry

end base8_base13_to_base10_sum_l128_128661


namespace pirates_total_coins_l128_128596

theorem pirates_total_coins :
  ∀ (x : ℕ), (∃ (paul_coins pete_coins : ℕ), 
  paul_coins = x ∧ pete_coins = 5 * x ∧ pete_coins = (x * (x + 1)) / 2) → x + 5 * x = 54 := by
  sorry

end pirates_total_coins_l128_128596


namespace sin_thirty_degrees_l128_128331

theorem sin_thirty_degrees : Real.sin (Real.pi / 6) = 1 / 2 := 
by sorry

end sin_thirty_degrees_l128_128331


namespace find_x_l128_128389

theorem find_x (x : ℚ) (n : ℤ) (f : ℚ) (h1 : x = n + f) (h2 : n = ⌊x⌋) (h3 : f < 1): 
  ⌊x⌋ + x = 17 / 4 → x = 9 / 4 :=
by 
  sorry

end find_x_l128_128389


namespace cutting_stick_ways_l128_128373

theorem cutting_stick_ways :
  ∃ (s : Finset (ℕ × ℕ)), 
  (∀ a ∈ s, 2 * a.1 + 3 * a.2 = 14) ∧
  s.card = 2 := 
by
  sorry

end cutting_stick_ways_l128_128373


namespace sector_angle_measure_l128_128442

-- Define the variables
variables (r α : ℝ)

-- Define the conditions
def perimeter_condition := (2 * r + r * α = 4)
def area_condition := (1 / 2 * α * r^2 = 1)

-- State the theorem
theorem sector_angle_measure (h1 : perimeter_condition r α) (h2 : area_condition r α) : α = 2 :=
sorry

end sector_angle_measure_l128_128442


namespace least_four_digit_11_heavy_l128_128801

def is_11_heavy (n : ℕ) : Prop := (n % 11) > 7

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

theorem least_four_digit_11_heavy : ∃ n : ℕ, is_four_digit n ∧ is_11_heavy n ∧ 
  (∀ m : ℕ, is_four_digit m ∧ is_11_heavy m → 1000 ≤ n) := 
sorry

end least_four_digit_11_heavy_l128_128801


namespace length_of_platform_l128_128227

theorem length_of_platform {train_length : ℕ} {time_to_cross_pole : ℕ} {time_to_cross_platform : ℕ} 
  (h1 : train_length = 300) 
  (h2 : time_to_cross_pole = 18) 
  (h3 : time_to_cross_platform = 45) : 
  ∃ platform_length : ℕ, platform_length = 450 :=
by
  sorry

end length_of_platform_l128_128227


namespace stmt_A_stmt_B_stmt_C_stmt_D_l128_128002
open Real

def x_and_y_conditions := ∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x + 2 * y = 3

theorem stmt_A : x_and_y_conditions → ∃ x y : ℝ, (0 < x ∧ 0 < y ∧ x + 2 * y = 3) ∧ (2 * (x * x + y * y) = 4) :=
by sorry

theorem stmt_B : x_and_y_conditions → ∃ x y : ℝ, (0 < x ∧ 0 < y ∧ x + 2 * y = 3) ∧ (x * y = 9 / 8) :=
by sorry

theorem stmt_C : x_and_y_conditions → ¬ (∃ x y : ℝ, (0 < x ∧ 0 < y ∧ x + 2 * y = 3) ∧ (sqrt (x) + sqrt (2 * y) = sqrt 6)) :=
by sorry

theorem stmt_D : x_and_y_conditions → ∃ x y : ℝ, (0 < x ∧ 0 < y ∧ x + 2 * y = 3) ∧ (x^2 + 4 * y^2 = 9 / 2) :=
by sorry

end stmt_A_stmt_B_stmt_C_stmt_D_l128_128002


namespace find_x_l128_128392

theorem find_x (x : ℚ) (n : ℤ) (f : ℚ) (h1 : x = n + f) (h2 : n = ⌊x⌋) (h3 : f < 1): 
  ⌊x⌋ + x = 17 / 4 → x = 9 / 4 :=
by 
  sorry

end find_x_l128_128392


namespace sin_thirty_degree_l128_128278

theorem sin_thirty_degree : Real.sin (30 * Real.pi / 180) = 1 / 2 := 
by
  sorry

end sin_thirty_degree_l128_128278


namespace kia_vehicle_count_l128_128006

theorem kia_vehicle_count (total_vehicles dodge_vehicles hyundai_vehicles kia_vehicles : ℕ)
  (h_total: total_vehicles = 400)
  (h_dodge: dodge_vehicles = total_vehicles / 2)
  (h_hyundai: hyundai_vehicles = dodge_vehicles / 2)
  (h_kia: kia_vehicles = total_vehicles - dodge_vehicles - hyundai_vehicles) : kia_vehicles = 100 := 
by
  sorry

end kia_vehicle_count_l128_128006


namespace solve_ff_eq_x_l128_128023

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x - 5

theorem solve_ff_eq_x (x : ℝ) :
  (f (f x) = x) ↔ 
  (x = (5 + 3 * Real.sqrt 5) / 2 ∨
   x = (5 - 3 * Real.sqrt 5) / 2 ∨
   x = (3 + Real.sqrt 41) / 2 ∨ 
   x = (3 - Real.sqrt 41) / 2) := 
by
  sorry

end solve_ff_eq_x_l128_128023


namespace tan_4356_equals_l128_128652

noncomputable def tan_periodic (x : ℝ) : ℝ := Real.tan x

theorem tan_4356_equals :
  tan_periodic 4356 = 0.7265 :=
by
  -- Condition: tangent function has a period of 360 degrees or 2π in radians.
  have h1 : 4356 % 360 = 36 := sorry,
  -- Thus, tan(4356°) = tan(36°)
  have h2 : tan_periodic 4356 = tan_periodic 36 := by
    rw [tan_periodic, Real.tan_periodic, h1],
  -- Use trigonometric tables or calculation for tan(36°)
  exact sorry

end tan_4356_equals_l128_128652


namespace probability_of_tie_l128_128447

open ProbabilityTheory

-- Define the number of supporters
def num_supporters (candidate : Type) := 5

-- Define the probability for voting and not voting
def voting_probability : ℚ := 1 / 2
def non_voting_probability : ℚ := 1 / 2

-- Define random variables X_A and X_B for the number of votes for candidates A and B, respectively
noncomputable def X_A : ℕ → ℚ := λ k, (nat.choose 5 k : ℕ) * (voting_probability ^ k) * (non_voting_probability ^ (5 - k))
noncomputable def X_B : ℕ → ℚ := λ k, (nat.choose 5 k : ℕ) * (voting_probability ^ k) * (non_voting_probability ^ (5 - k))

-- Define the probability of a tie
noncomputable def prob_of_tie : ℚ :=
  ∑ k in finset.range 6, (X_A k) * (X_B k)
  
theorem probability_of_tie :
  prob_of_tie = 63 / 256 := sorry

end probability_of_tie_l128_128447


namespace cos_135_eq_neg_sqrt2_div_2_l128_128947

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
by 
  sorry

end cos_135_eq_neg_sqrt2_div_2_l128_128947


namespace part_one_part_two_l128_128582

-- Part (1)
theorem part_one (m : ℝ) : 
  (∀ x, -1 ≤ x ∧ x ≤ 2 → (2 * m < x ∧ x < 1 → -1 ≤ x ∧ x ≤ 2 ∧ - (1 / 2) ≤ m)) → 
  (m ≥ - (1 / 2)) :=
by sorry

-- Part (2)
theorem part_two (m : ℝ) : 
  (∃ x : ℤ, (2 * m < x ∧ x < 1) ∧ (x < -1 ∨ x > 2)) ∧ 
  (∀ y : ℤ, (2 * m < y ∧ y < 1) ∧ (y < -1 ∨ y > 2) → y = x) → 
  (- (3 / 2) ≤ m ∧ m < -1) :=
by sorry

end part_one_part_two_l128_128582


namespace length_of_GH_l128_128034

theorem length_of_GH (AB FE CD : ℕ) (side_large side_second side_third side_small : ℕ) 
  (h1 : AB = 11) (h2 : FE = 13) (h3 : CD = 5)
  (h4 : side_large = side_second + AB)
  (h5 : side_second = side_third + CD)
  (h6 : side_third = side_small + FE) :
  GH = 29 :=
by
  -- Proof steps would follow here based on the problem's solution
  -- Using the given conditions and transformations.
  sorry

end length_of_GH_l128_128034


namespace isosceles_triangle_solution_l128_128411

noncomputable def isosceles_triangle_sides (x y : ℝ) : Prop :=
(x + 1/2 * y = 6 ∧ 1/2 * x + y = 12) ∨ (x + 1/2 * y = 12 ∧ 1/2 * x + y = 6)

theorem isosceles_triangle_solution :
  ∃ (x y : ℝ), isosceles_triangle_sides x y ∧ x = 8 ∧ y = 2 :=
sorry

end isosceles_triangle_solution_l128_128411


namespace trig_identity_sin_eq_l128_128826

theorem trig_identity_sin_eq (α : ℝ) (h : Real.cos (π / 6 - α) = 1 / 3) : 
  Real.sin (5 * π / 6 - 2 * α) = -7 / 9 := 
by 
  sorry

end trig_identity_sin_eq_l128_128826


namespace maximize_exponential_sum_l128_128167

theorem maximize_exponential_sum (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 ≤ 4) : 
  e^a + e^b + e^c + e^d ≤ 4 * Real.exp 1 := 
sorry

end maximize_exponential_sum_l128_128167


namespace distance_to_bus_stand_l128_128088

variable (D : ℝ)

theorem distance_to_bus_stand :
  (D / 4 - D / 5 = 1 / 4) → D = 5 :=
sorry

end distance_to_bus_stand_l128_128088


namespace average_visitors_per_day_is_276_l128_128628

-- Define the number of days in the month
def num_days_in_month : ℕ := 30

-- Define the number of Sundays in the month
def num_sundays_in_month : ℕ := 4

-- Define the number of other days in the month
def num_other_days_in_month : ℕ := num_days_in_month - num_sundays_in_month * 7 / 7 + 2

-- Define the average visitors on Sundays
def avg_visitors_sunday : ℕ := 510

-- Define the average visitors on other days
def avg_visitors_other_days : ℕ := 240

-- Calculate total visitors on Sundays
def total_visitors_sundays : ℕ := num_sundays_in_month * avg_visitors_sunday

-- Calculate total visitors on other days
def total_visitors_other_days : ℕ := num_other_days_in_month * avg_visitors_other_days

-- Calculate total visitors in the month
def total_visitors_in_month : ℕ := total_visitors_sundays + total_visitors_other_days

-- Given conditions, prove average visitors per day in a month
theorem average_visitors_per_day_is_276 :
  total_visitors_in_month / num_days_in_month = 276 := by
  sorry

end average_visitors_per_day_is_276_l128_128628


namespace total_pieces_of_chicken_needed_l128_128058

def friedChickenDinnerPieces := 8
def chickenPastaPieces := 2
def barbecueChickenPieces := 4
def grilledChickenSaladPieces := 1

def friedChickenDinners := 4
def chickenPastaOrders := 8
def barbecueChickenOrders := 5
def grilledChickenSaladOrders := 6

def totalChickenPiecesNeeded :=
  (friedChickenDinnerPieces * friedChickenDinners) +
  (chickenPastaPieces * chickenPastaOrders) +
  (barbecueChickenPieces * barbecueChickenOrders) +
  (grilledChickenSaladPieces * grilledChickenSaladOrders)

theorem total_pieces_of_chicken_needed : totalChickenPiecesNeeded = 74 := by
  sorry

end total_pieces_of_chicken_needed_l128_128058


namespace numbers_composite_l128_128012

theorem numbers_composite (a b c d : ℕ) (h : a * b = c * d) : ∃ x y : ℕ, (x > 1 ∧ y > 1) ∧ a^2000 + b^2000 + c^2000 + d^2000 = x * y := 
sorry

end numbers_composite_l128_128012


namespace vinnie_makes_more_l128_128177

-- Define the conditions
def paul_tips : ℕ := 14
def vinnie_tips : ℕ := 30

-- Define the theorem to prove
theorem vinnie_makes_more :
  vinnie_tips - paul_tips = 16 := by
  sorry

end vinnie_makes_more_l128_128177


namespace acme_horseshoes_production_l128_128936

theorem acme_horseshoes_production
  (profit : ℝ)
  (initial_outlay : ℝ)
  (cost_per_set : ℝ)
  (selling_price : ℝ)
  (number_of_sets : ℕ) :
  profit = selling_price * number_of_sets - (initial_outlay + cost_per_set * number_of_sets) →
  profit = 15337.5 →
  initial_outlay = 12450 →
  cost_per_set = 20.75 →
  selling_price = 50 →
  number_of_sets = 950 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end acme_horseshoes_production_l128_128936


namespace resulting_solution_percentage_l128_128794

theorem resulting_solution_percentage :
  ∀ (C_init R C_replace : ℚ), 
  C_init = 0.85 → 
  R = 0.6923076923076923 → 
  C_replace = 0.2 → 
  (C_init * (1 - R) + C_replace * R) = 0.4 :=
by
  intros C_init R C_replace hC_init hR hC_replace
  -- Omitted proof here
  sorry

end resulting_solution_percentage_l128_128794


namespace solution_set_empty_l128_128697

variable (m x : ℝ)
axiom no_solution (h : (3 - 2 * x) / (x - 3) + (2 + m * x) / (3 - x) = -1) : (1 + m = 0)

theorem solution_set_empty :
  (∀ x, (3 - 2 * x) / (x - 3) + (2 + m * x) / (3 - x) ≠ -1) ↔ m = -1 := by
  sorry

end solution_set_empty_l128_128697


namespace neither_necessary_nor_sufficient_l128_128716

theorem neither_necessary_nor_sufficient (x : ℝ) : 
  ¬ ((x = 0) ↔ (x^2 - 2 * x = 0) ∧ (x ≠ 0 → x^2 - 2 * x ≠ 0) ∧ (x = 0 → x^2 - 2 * x = 0)) := 
sorry

end neither_necessary_nor_sufficient_l128_128716


namespace sin_30_is_half_l128_128356

noncomputable def sin_30_degrees : ℝ := Real.sin (Real.pi / 6)

theorem sin_30_is_half : sin_30_degrees = 1 / 2 := by
  sorry

end sin_30_is_half_l128_128356


namespace addition_terms_correct_l128_128718

def first_seq (n : ℕ) : ℕ := 2 * n + 1
def second_seq (n : ℕ) : ℕ := 5 * n - 1

theorem addition_terms_correct :
  first_seq 10 = 21 ∧ second_seq 10 = 49 ∧
  first_seq 80 = 161 ∧ second_seq 80 = 399 :=
by
  sorry

end addition_terms_correct_l128_128718


namespace length_of_segment_GH_l128_128029

theorem length_of_segment_GH (a1 a2 a3 a4 : ℕ)
  (h1 : a1 = a2 + 11)
  (h2 : a2 = a3 + 5)
  (h3 : a3 = a4 + 13)
  : a1 - a4 = 29 :=
by
  sorry

end length_of_segment_GH_l128_128029


namespace find_x_of_floor_plus_x_eq_17_over_4_l128_128395

theorem find_x_of_floor_plus_x_eq_17_over_4 (x : ℝ) (hx : ⌊x⌋ + x = 17 / 4) : x = 9 / 4 := 
sorry

end find_x_of_floor_plus_x_eq_17_over_4_l128_128395


namespace chocolate_chip_cookies_l128_128641

theorem chocolate_chip_cookies (chocolate_chips_per_recipe : ℕ) (num_recipes : ℕ) (total_chocolate_chips : ℕ) 
  (h1 : chocolate_chips_per_recipe = 2) 
  (h2 : num_recipes = 23) 
  (h3 : total_chocolate_chips = chocolate_chips_per_recipe * num_recipes) : 
  total_chocolate_chips = 46 :=
by
  rw [h1, h2] at h3
  exact h3

-- sorry

end chocolate_chip_cookies_l128_128641


namespace polygon_sides_eq_eight_l128_128845

theorem polygon_sides_eq_eight (n : ℕ) 
  (h_diff : (n - 2) * 180 - 360 = 720) :
  n = 8 := 
by 
  sorry

end polygon_sides_eq_eight_l128_128845


namespace toy_store_restock_l128_128528

theorem toy_store_restock 
  (initial_games : ℕ) (games_sold : ℕ) (after_restock_games : ℕ) 
  (initial_games_condition : initial_games = 95)
  (games_sold_condition : games_sold = 68)
  (after_restock_games_condition : after_restock_games = 74) :
  after_restock_games - (initial_games - games_sold) = 47 :=
by {
  sorry
}

end toy_store_restock_l128_128528


namespace compute_factorial_expression_l128_128372

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n+1 => (n+1) * factorial n

theorem compute_factorial_expression :
  factorial 9 - factorial 8 - factorial 7 + factorial 6 = 318240 := by
  sorry

end compute_factorial_expression_l128_128372


namespace sin_30_eq_half_l128_128293

theorem sin_30_eq_half : Real.sin (Float.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l128_128293


namespace largest_lcm_18_l128_128762

theorem largest_lcm_18 :
  max (max (max (max (max (Nat.lcm 18 3) (Nat.lcm 18 6)) (Nat.lcm 18 9)) (Nat.lcm 18 12)) (Nat.lcm 18 15)) (Nat.lcm 18 18) = 90 :=
by sorry

end largest_lcm_18_l128_128762


namespace Caitlin_correct_age_l128_128808

def Aunt_Anna_age := 48
def Brianna_age := Aunt_Anna_age / 2
def Caitlin_age := Brianna_age - 7

theorem Caitlin_correct_age : Caitlin_age = 17 := by
  /- Condon: Aunt Anna is 48 years old. -/
  let ha := Aunt_Anna_age
  /- Condon: Brianna is half as old as Aunt Anna. -/
  let hb := Brianna_age
  /- Condon: Caitlin is 7 years younger than Brianna. -/
  let hc := Caitlin_age
  /- Question: How old is Caitlin? Proof: -/
  sorry

end Caitlin_correct_age_l128_128808


namespace minimum_value_of_f_l128_128817

noncomputable def f (x : ℝ) : ℝ := (Real.cos (2 * x) + 2 * Real.sin x)

theorem minimum_value_of_f : ∃ x : ℝ, (∀ y : ℝ, f x ≤ f y) ∧ f x = -3 := 
  sorry

end minimum_value_of_f_l128_128817


namespace smallest_positive_n_l128_128914

theorem smallest_positive_n : ∃ (n : ℕ), n > 0 ∧ (gcd (5 * n - 3) (11 * n + 4)) > 1 ∧ ∀ (m : ℕ), m > 0 → (gcd (5 * m - 3) (11 * m + 4)) > 1 → n ≤ m :=
begin
  sorry
end

end smallest_positive_n_l128_128914


namespace ratio_problem_l128_128715

theorem ratio_problem
  (a b c d e : ℚ)
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 7)
  (h4 : d / e = 2) :
  e / a = 2 / 35 := 
sorry

end ratio_problem_l128_128715


namespace sin_30_eq_one_half_l128_128365

def Q_coords (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

theorem sin_30_eq_one_half :
  let Q := Q_coords (Real.pi / 6)
  Q.2 = 1 / 2 :=
by
  sorry

end sin_30_eq_one_half_l128_128365


namespace cos_135_degree_l128_128981

theorem cos_135_degree : 
  let Q := (Complex.exp (135 * Real.pi / 180 * Complex.I)) in 
  Complex.re Q = -Real.sqrt 2 / 2 :=
by sorry

end cos_135_degree_l128_128981


namespace total_vowels_written_l128_128648

-- Define the vowels and the condition
def vowels : List Char := ['A', 'E', 'I', 'O', 'U']
def num_vowels : Nat := vowels.length
def times_written : Nat := 2

-- Assert the total number of vowels written
theorem total_vowels_written : (num_vowels * times_written) = 10 := by
  sorry

end total_vowels_written_l128_128648


namespace kia_vehicle_count_l128_128007

theorem kia_vehicle_count (total_vehicles dodge_vehicles hyundai_vehicles kia_vehicles : ℕ)
  (h_total: total_vehicles = 400)
  (h_dodge: dodge_vehicles = total_vehicles / 2)
  (h_hyundai: hyundai_vehicles = dodge_vehicles / 2)
  (h_kia: kia_vehicles = total_vehicles - dodge_vehicles - hyundai_vehicles) : kia_vehicles = 100 := 
by
  sorry

end kia_vehicle_count_l128_128007


namespace sum_of_factors_l128_128077

-- Define 36
def num : ℕ := 36

-- Prime factorization of 36
def prime_factorization : (ℕ × ℕ) := (2, 2) -- represents 2^2
def prime_factorization_2 : (ℕ × ℕ) := (3, 2) -- represents 3^2

-- Sum of divisors function formula
def sigma (n : ℕ) : ℕ := 
  (1 + 2 + 2^2) * (1 + 3 + 3^2)

-- Theorem: sum of the positive factors of 36 is 91
theorem sum_of_factors : sigma num = 91 := 
by {
  -- Skipping the proof steps
  sorry
}

end sum_of_factors_l128_128077


namespace cos_135_eq_neg_sqrt2_div_2_l128_128985

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = - Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l128_128985


namespace minimum_omega_is_3_l128_128171

variable (f : ℝ → ℝ)
variable (ω φ T : ℝ)
variable (x : ℝ)

noncomputable def minimum_omega (ω : ℝ) : ℝ := 3 -- Temporary noncomputable definition to hold the value for min omega

theorem minimum_omega_is_3 :
  (∀ x, f x = Real.cos (ω * x + φ)) →
  ω > 0 →
  0 < φ ∧ φ < Real.pi →
  T = (2 * Real.pi) / ω →
  f T = Real.sqrt 3 / 2 →
  f (Real.pi / 9) = 0 →
  ∃(ω_min : ℝ), ω_min = 3 :=
by
  intros h0 h1 h2 h3 h4 h5
  use 3
  sorry -- Proof of the exact answer

end minimum_omega_is_3_l128_128171


namespace trig_identity_proof_l128_128597

theorem trig_identity_proof 
  (α : ℝ) 
  (h1 : Real.sin (4 * α) = 2 * Real.sin (2 * α) * Real.cos (2 * α))
  (h2 : Real.cos (4 * α) = Real.cos (2 * α) ^ 2 - Real.sin (2 * α) ^ 2) : 
  (1 - 2 * Real.sin (2 * α) ^ 2) / (1 - Real.sin (4 * α)) = 
  (1 + Real.tan (2 * α)) / (1 - Real.tan (2 * α)) := 
by 
  sorry

end trig_identity_proof_l128_128597


namespace polynomial_satisfies_condition_l128_128533

open Polynomial

noncomputable def polynomial_f : Polynomial ℝ := 6 * X ^ 2 + 5 * X + 1
noncomputable def polynomial_g : Polynomial ℝ := 3 * X ^ 2 + 7 * X + 2

def sum_of_squares (p : Polynomial ℝ) : ℝ :=
  p.coeff 0 ^ 2 + p.coeff 1 ^ 2 + p.coeff 2 ^ 2 + p.coeff 3 ^ 2 + -- ...
  sorry -- Extend as necessary for the degree of the polynomial

theorem polynomial_satisfies_condition :
  (∀ n : ℕ, sum_of_squares (polynomial_f ^ n) = sum_of_squares (polynomial_g ^ n)) :=
by
  sorry

end polynomial_satisfies_condition_l128_128533


namespace find_x_l128_128819

def determinant (a b c d : ℚ) : ℚ := a * d - b * c

theorem find_x (x : ℚ) (h : determinant (2 * x) (-4) x 1 = 18) : x = 3 :=
  sorry

end find_x_l128_128819


namespace number_in_scientific_notation_l128_128883

def scientific_notation_form (a : ℝ) (n : ℤ) : Prop :=
  1 ≤ a ∧ a < 10

theorem number_in_scientific_notation : scientific_notation_form 3.7515 7 ∧ 37515000 = 3.7515 * 10^7 :=
by
  sorry

end number_in_scientific_notation_l128_128883


namespace same_cost_for_same_sheets_l128_128104

def John's_Photo_World_cost (x : ℕ) : ℝ := 2.75 * x + 125
def Sam's_Picture_Emporium_cost (x : ℕ) : ℝ := 1.50 * x + 140

theorem same_cost_for_same_sheets :
  ∃ (x : ℕ), John's_Photo_World_cost x = Sam's_Picture_Emporium_cost x ∧ x = 12 :=
by
  sorry

end same_cost_for_same_sheets_l128_128104


namespace prob_two_red_balls_in_four_draws_l128_128570

noncomputable def probability_red_balls (draws : ℕ) (red_in_draw : ℕ) (total_balls : ℕ) (red_balls : ℕ) : ℝ :=
  let prob_red := (red_balls : ℝ) / (total_balls : ℝ)
  let prob_white := 1 - prob_red
  (Nat.choose draws red_in_draw : ℝ) * (prob_red ^ red_in_draw) * (prob_white ^ (draws - red_in_draw))

theorem prob_two_red_balls_in_four_draws :
  probability_red_balls 4 2 10 4 = 0.3456 :=
by
  sorry

end prob_two_red_balls_in_four_draws_l128_128570


namespace sin_30_is_half_l128_128351

noncomputable def sin_30_degrees : ℝ := Real.sin (Real.pi / 6)

theorem sin_30_is_half : sin_30_degrees = 1 / 2 := by
  sorry

end sin_30_is_half_l128_128351


namespace johns_cocktail_not_stronger_l128_128741

-- Define the percentage of alcohol in each beverage
def beer_percent_alcohol : ℝ := 0.05
def liqueur_percent_alcohol : ℝ := 0.10
def vodka_percent_alcohol : ℝ := 0.40
def whiskey_percent_alcohol : ℝ := 0.50

-- Define the weights of the liquids used in the cocktails
def john_liqueur_weight : ℝ := 400
def john_whiskey_weight : ℝ := 100
def ivan_vodka_weight : ℝ := 400
def ivan_beer_weight : ℝ := 100

-- Calculate the alcohol contents in each cocktail
def johns_cocktail_alcohol : ℝ := john_liqueur_weight * liqueur_percent_alcohol + john_whiskey_weight * whiskey_percent_alcohol
def ivans_cocktail_alcohol : ℝ := ivan_vodka_weight * vodka_percent_alcohol + ivan_beer_weight * beer_percent_alcohol

-- The proof statement to prove that John's cocktail is not stronger than Ivan's cocktail
theorem johns_cocktail_not_stronger :
  johns_cocktail_alcohol ≤ ivans_cocktail_alcohol :=
by
  -- Add proof steps here
  sorry

end johns_cocktail_not_stronger_l128_128741


namespace sin_thirty_deg_l128_128326

theorem sin_thirty_deg : Real.sin (π / 6) = 1 / 2 := 
by 
-- Assume that the unit circle point corresponding to 30° is (√3/2, 1/2)
-- Sine of an angle in the unit circle is the y-coordinate of the corresponding point
sorry

end sin_thirty_deg_l128_128326


namespace find_k_l128_128930

variable {V : Type} [AddCommGroup V] [Module ℝ V]
variable (u v : V)

theorem find_k (h : ∃ t : ℝ, t ≠ 0 ∧ t ≠ 1 ∧ u + t • (v - u) = k • u + (5 / 8) • v) :
  k = 3 / 8 := sorry

end find_k_l128_128930


namespace monotonically_increasing_on_pos_real_l128_128774

def f1 (x : ℝ) := - real.log x
def f2 (x : ℝ) := 1 / 2^x
def f3 (x : ℝ) := -1 / x
def f4 (x : ℝ) := 3^|x-1|

theorem monotonically_increasing_on_pos_real (f : ℝ → ℝ)
    (h1 : ∀ x, f = f1 → x ∈ set.Ioi 0 →  deriv f x > 0)
    (h2 : ∀ x, f = f2 → x ∈ set.Ioi 0 →  deriv f x > 0)
    (h3 : ∀ x, f = f3 → x ∈ set.Ioi 0 →  deriv f x > 0)
    (h4 : ∀ x, f = f4 → x ∈ set.Ioi 0 →  deriv f x > 0) :
  f = f3 → ∀ x ∈ set.Ioi 0, deriv f x > 0 := 
sorry

end monotonically_increasing_on_pos_real_l128_128774


namespace Derek_test_score_l128_128560

def Grant_score (John_score : ℕ) : ℕ := John_score + 10
def John_score (Hunter_score : ℕ) : ℕ := 2 * Hunter_score
def Hunter_score : ℕ := 45
def Sarah_score (Grant_score : ℕ) : ℕ := Grant_score - 5
def Derek_score (John_score Grant_score : ℕ) : ℕ := (John_score + Grant_score) / 2

theorem Derek_test_score :
  Derek_score (John_score Hunter_score) (Grant_score (John_score Hunter_score)) = 95 :=
  by
  -- proof here
  sorry

end Derek_test_score_l128_128560


namespace problem1_problem2_l128_128497

-- Problem 1
theorem problem1 : 
  (2 ^ (1 + Real.log 3 / Real.log 2)) / (Real.log 2 * Real.log 2 + Real.log 5 + Real.log 2 * Real.log 5) = 6 := 
  sorry

-- Problem 2
theorem problem2 (α : ℝ) (h1 : 0 < α) (h2 : α < real.pi) (h3 : real.sin α + real.cos α = 3/5) : 
  real.sin α - real.cos α = real.sqrt 41 / 5 := 
  sorry

end problem1_problem2_l128_128497


namespace sin_thirty_degree_l128_128275

theorem sin_thirty_degree : Real.sin (30 * Real.pi / 180) = 1 / 2 := 
by
  sorry

end sin_thirty_degree_l128_128275


namespace smallest_solution_l128_128672

theorem smallest_solution (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ 4) (h₃ : x ≠ 5) 
    (h_eq : 1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) : x = 4 - Real.sqrt 2 := 
by 
  sorry

end smallest_solution_l128_128672


namespace sin_30_eq_half_l128_128320

-- Conditions
def point_Q (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

-- The angle 30 degrees in radians
def angle_30_deg := Real.pi / 6

-- Point Q at Angle 30 degrees to the positive x-axis
def Q := point_Q angle_30_deg

-- Formalized problem statement in Lean 4
theorem sin_30_eq_half : Real.sin angle_30_deg = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l128_128320


namespace train_length_eq_l128_128935

-- Definitions
def train_speed_kmh : Float := 45
def crossing_time_s : Float := 30
def total_length_m : Float := 245

-- Theorem statement
theorem train_length_eq :
  ∃ (train_length bridge_length: Float),
  bridge_length = total_length_m - train_length ∧
  train_speed_kmh * 1000 / 3600 * crossing_time_s = train_length + bridge_length ∧
  train_length = 130 :=
by
  sorry

end train_length_eq_l128_128935


namespace cos_135_eq_neg_sqrt2_div_2_l128_128950

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
by 
  sorry

end cos_135_eq_neg_sqrt2_div_2_l128_128950


namespace sin_30_eq_half_l128_128316

-- Conditions
def point_Q (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

-- The angle 30 degrees in radians
def angle_30_deg := Real.pi / 6

-- Point Q at Angle 30 degrees to the positive x-axis
def Q := point_Q angle_30_deg

-- Formalized problem statement in Lean 4
theorem sin_30_eq_half : Real.sin angle_30_deg = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l128_128316


namespace sin_30_deg_l128_128303

theorem sin_30_deg : real.sin (30 * real.pi / 180) = 1 / 2 := by
  sorry

end sin_30_deg_l128_128303


namespace real_roots_system_l128_128655

theorem real_roots_system :
  ∃ (x y : ℝ), 
    (x * y * (x^2 + y^2) = 78 ∧ x^4 + y^4 = 97) ↔ 
    (x, y) = (3, 2) ∨ (x, y) = (2, 3) ∨ (x, y) = (-3, -2) ∨ (x, y) = (-2, -3) := 
by 
  sorry

end real_roots_system_l128_128655


namespace problem_statement_l128_128165

def T : Set ℤ :=
  {n^2 + (n+2)^2 + (n+4)^2 | n : ℤ }

theorem problem_statement :
  (∀ x ∈ T, ¬ (4 ∣ x)) ∧ (∃ x ∈ T, 13 ∣ x) :=
by
  sorry

end problem_statement_l128_128165


namespace journey_time_ratio_l128_128924

theorem journey_time_ratio (D : ℝ) (h₁ : D > 0) :
  let T1 := D / 48
  let T2 := D / 32
  (T2 / T1) = 3 / 2 :=
by
  sorry

end journey_time_ratio_l128_128924


namespace inequality_solution_set_l128_128890

theorem inequality_solution_set : 
  { x : ℝ | x^2 + x - 2 < 0 } = { x : ℝ | -2 < x ∧ x < 1 } :=
sorry

end inequality_solution_set_l128_128890


namespace sin_30_eq_half_l128_128287

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the point Q at 30 degrees on the unit circle
def Q : ℝ × ℝ := unit_circle (Real.pi / 6)

theorem sin_30_eq_half : (Real.sin (Real.pi / 6)) = 1 / 2 :=
by sorry

end sin_30_eq_half_l128_128287


namespace probability_of_drawing_three_white_balls_l128_128499

open Nat

def binom (n k : ℕ) : ℕ := nat.choose n k

def probability_of_three_white_balls (total_white total_black : ℕ) (drawn : ℕ) : ℚ :=
  (binom total_white drawn) / (binom (total_white + total_black) drawn)

theorem probability_of_drawing_three_white_balls :
  probability_of_three_white_balls 4 7 3 = 4 / 165 := by
  sorry

end probability_of_drawing_three_white_balls_l128_128499


namespace factorial_less_power_l128_128480

open Nat

noncomputable def factorial_200 : ℕ := 200!

noncomputable def power_100_200 : ℕ := 100 ^ 200

theorem factorial_less_power : factorial_200 < power_100_200 :=
by
  -- Proof goes here
  sorry

end factorial_less_power_l128_128480


namespace closest_ratio_adults_children_l128_128473

theorem closest_ratio_adults_children (a c : ℕ) 
  (h1 : 30 * a + 15 * c = 2250) 
  (h2 : a ≥ 50) 
  (h3 : c ≥ 20) : a = 50 ∧ c = 50 :=
by {
  sorry
}

end closest_ratio_adults_children_l128_128473


namespace shirley_boxes_to_cases_l128_128469

theorem shirley_boxes_to_cases (boxes_sold : Nat) (boxes_per_case : Nat) (cases_needed : Nat) 
      (h1 : boxes_sold = 54) (h2 : boxes_per_case = 6) : cases_needed = 9 :=
by
  sorry

end shirley_boxes_to_cases_l128_128469


namespace sum_of_positive_factors_36_l128_128069

theorem sum_of_positive_factors_36 : (∑ d in (finset.filter (λ n, 36 % n = 0) (finset.range (36 + 1))), d) = 91 :=
sorry

end sum_of_positive_factors_36_l128_128069


namespace triangle_sides_possible_k_l128_128428

noncomputable def f (x k : ℝ) : ℝ := x^2 - 4*x + 4 + k^2

theorem triangle_sides_possible_k (a b c k : ℝ) (ha : 0 ≤ a) (hb : a ≤ 3) (ha' : 0 ≤ b) (hb' : b ≤ 3) (ha'' : 0 ≤ c) (hb'' : c ≤ 3) :
  (f a k + f b k > f c k) ∧ (f a k + f c k > f b k) ∧ (f b k + f c k > f a k) ↔ k = 3 ∨ k = 4 :=
by
  sorry

end triangle_sides_possible_k_l128_128428


namespace cos_135_eq_neg_sqrt2_div_2_l128_128962

theorem cos_135_eq_neg_sqrt2_div_2 : 
  Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l128_128962


namespace value_of_a_l128_128198

-- Declare and define the given conditions.
def line1 (y : ℝ) := y = 13
def line2 (x t y : ℝ) := y = 3 * x + t

-- Define the proof statement.
theorem value_of_a (a b t : ℝ) (h1 : line1 b) (h2 : line2 a t b) (ht : t = 1) : a = 4 :=
by
  sorry

end value_of_a_l128_128198


namespace floor_plus_x_eq_17_over_4_l128_128403

theorem floor_plus_x_eq_17_over_4
  (x : ℚ)
  (h : ⌊x⌋ + x = 17 / 4)
  : x = 9 / 4 :=
sorry

end floor_plus_x_eq_17_over_4_l128_128403


namespace at_least_one_miss_l128_128158

variables (p q : Prop)

-- Proposition stating the necessary and sufficient condition.
theorem at_least_one_miss : ¬(p ∧ q) ↔ (¬p ∨ ¬q) :=
by sorry

end at_least_one_miss_l128_128158


namespace remaining_garden_space_l128_128503

theorem remaining_garden_space : 
  let Area_rectangle := 20 * 18
  let Area_square_cutout := 4 * 4
  let Area_triangle := (1 / 2) * 3 * 2
  Area_rectangle - Area_square_cutout + Area_triangle = 347 :=
by
  let Area_rectangle := 20 * 18
  let Area_square_cutout := 4 * 4
  let Area_triangle := (1 / 2) * 3 * 2
  show Area_rectangle - Area_square_cutout + Area_triangle = 347
  sorry

end remaining_garden_space_l128_128503


namespace inserting_eights_is_composite_l128_128205

theorem inserting_eights_is_composite (n : ℕ) : ¬ Nat.Prime (2000 * 10^n + 8 * ((10^n - 1) / 9) + 21) := 
by sorry

end inserting_eights_is_composite_l128_128205


namespace overlapping_rectangles_perimeter_l128_128761

namespace RectangleOverlappingPerimeter

def length := 7
def width := 3

/-- Prove that the perimeter of the shape formed by overlapping two rectangles,
    each measuring 7 cm by 3 cm, is 28 cm. -/
theorem overlapping_rectangles_perimeter : 
  let total_perimeter := 2 * (length + (2 * width))
  total_perimeter = 28 :=
by
  sorry

end RectangleOverlappingPerimeter

end overlapping_rectangles_perimeter_l128_128761


namespace cost_of_fencing_per_meter_l128_128643

def rectangular_farm_area : Real := 1200
def short_side_length : Real := 30
def total_cost : Real := 1440

theorem cost_of_fencing_per_meter : (total_cost / (short_side_length + (rectangular_farm_area / short_side_length) + Real.sqrt ((rectangular_farm_area / short_side_length)^2 + short_side_length^2))) = 12 :=
by
  sorry

end cost_of_fencing_per_meter_l128_128643


namespace mass_percentage_O_in_C6H8O6_l128_128816

theorem mass_percentage_O_in_C6H8O6 :
  let atomic_mass_C := 12.01
  let atomic_mass_H := 1.01
  let atomic_mass_O := 16.00
  let molar_mass_C6H8O6 := (6 * atomic_mass_C) + (8 * atomic_mass_H) + (6 * atomic_mass_O)
  let mass_of_oxygen := 8 * atomic_mass_O
  let mass_percentage_O := (mass_of_oxygen / molar_mass_C6H8O6) * 100
  mass_percentage_O = 72.67 :=
by
  -- Definitions
  let atomic_mass_C := 12.01
  let atomic_mass_H := 1.01
  let atomic_mass_O := 16.00
  let molar_mass_C6H8O6 := (6 * atomic_mass_C) + (8 * atomic_mass_H) + (6 * atomic_mass_O)
  let mass_of_oxygen := 8 * atomic_mass_O
  let mass_percentage_O := (mass_of_oxygen / molar_mass_C6H8O6) * 100
  -- Proof
  sorry

end mass_percentage_O_in_C6H8O6_l128_128816


namespace coin_toss_probability_l128_128780

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * p^k * (1 - p)^(n - k)

theorem coin_toss_probability :
  binomial_probability 3 2 0.5 = 0.375 :=
by
  sorry

end coin_toss_probability_l128_128780


namespace sqrt_ratio_simplify_l128_128496

theorem sqrt_ratio_simplify :
  ( (Real.sqrt 27 + Real.sqrt 243) / Real.sqrt 75 = 12 / 5 ) :=
by
  let sqrt27 := Real.sqrt 27
  let sqrt243 := Real.sqrt 243
  let sqrt75 := Real.sqrt 75
  have h_sqrt27 : sqrt27 = Real.sqrt (3^2 * 3) := by sorry
  have h_sqrt243 : sqrt243 = Real.sqrt (3^5) := by sorry
  have h_sqrt75 : sqrt75 = Real.sqrt (3 * 5^2) := by sorry
  have h_simplified : (sqrt27 + sqrt243) / sqrt75 = 12 / 5 := by sorry
  exact h_simplified

end sqrt_ratio_simplify_l128_128496


namespace find_x_eq_nine_fourths_l128_128387

theorem find_x_eq_nine_fourths (x n : ℚ) (β : ℚ) (h1 : x = n + β) (h2 : n = ⌊x⌋) (h3 : ⌊x⌋ + x = 17 / 4) : x = 9 / 4 :=
by
  sorry

end find_x_eq_nine_fourths_l128_128387


namespace new_salary_after_increase_l128_128593

theorem new_salary_after_increase : 
  ∀ (previous_salary : ℝ) (percentage_increase : ℝ), 
    previous_salary = 2000 → percentage_increase = 0.05 → 
    previous_salary + (previous_salary * percentage_increase) = 2100 :=
by
  intros previous_salary percentage_increase h1 h2
  sorry

end new_salary_after_increase_l128_128593


namespace sin_30_eq_half_l128_128359

theorem sin_30_eq_half : Real.sin (Real.pi / 6) = 1 / 2 := sorry

end sin_30_eq_half_l128_128359


namespace find_x_l128_128710

-- Definitions based on conditions
variables (A B C M O : Type)
variables (OA OB OC OM : vector_space O)
variables (x : ℚ) -- Rational number type for x

-- Condition (1): M lies in the plane ABC
-- Condition (2): OM = x * OA + 1/3 * OB + 1/2 * OC
axiom H : OM = x • OA + (1 / 3 : ℚ) • OB + (1 / 2 : ℚ) • OC

-- The theorem statement
theorem find_x :
  x = 1 / 6 :=
sorry -- Proof is to be provided

end find_x_l128_128710


namespace max_value_a7_b7_c7_d7_l128_128182

-- Assume a, b, c, d are real numbers such that a^6 + b^6 + c^6 + d^6 = 64
-- Prove that the maximum value of a^7 + b^7 + c^7 + d^7 is 128
theorem max_value_a7_b7_c7_d7 (a b c d : ℝ) (h : a^6 + b^6 + c^6 + d^6 = 64) : 
  ∃ a b c d, a^6 + b^6 + c^6 + d^6 = 64 ∧ a^7 + b^7 + c^7 + d^7 = 128 :=
by sorry

end max_value_a7_b7_c7_d7_l128_128182


namespace janet_home_time_l128_128577

def blocks_north := 3
def blocks_west := 7 * blocks_north
def blocks_south := blocks_north
def blocks_east := 2 * blocks_south -- Initially mistaken, recalculating needed
def remaining_blocks_west := blocks_west - blocks_east
def total_blocks_home := blocks_south + remaining_blocks_west
def walking_speed := 2 -- blocks per minute

theorem janet_home_time :
  (blocks_south + remaining_blocks_west) / walking_speed = 9 := by
  -- We assume that Lean can handle the arithmetic properly here.
  sorry

end janet_home_time_l128_128577


namespace sin_30_is_half_l128_128357

noncomputable def sin_30_degrees : ℝ := Real.sin (Real.pi / 6)

theorem sin_30_is_half : sin_30_degrees = 1 / 2 := by
  sorry

end sin_30_is_half_l128_128357


namespace sin_30_is_half_l128_128355

noncomputable def sin_30_degrees : ℝ := Real.sin (Real.pi / 6)

theorem sin_30_is_half : sin_30_degrees = 1 / 2 := by
  sorry

end sin_30_is_half_l128_128355


namespace sin_30_eq_half_l128_128295

theorem sin_30_eq_half : Real.sin (π / 6) = 1 / 2 := 
  sorry

end sin_30_eq_half_l128_128295


namespace uncle_jerry_total_tomatoes_l128_128218

def day1_tomatoes : ℕ := 120
def day2_tomatoes : ℕ := day1_tomatoes + 50
def day3_tomatoes : ℕ := 2 * day2_tomatoes
def total_tomatoes : ℕ := day1_tomatoes + day2_tomatoes + day3_tomatoes

theorem uncle_jerry_total_tomatoes : total_tomatoes = 630 := by
  sorry

end uncle_jerry_total_tomatoes_l128_128218


namespace percentage_shaded_in_square_l128_128771

theorem percentage_shaded_in_square
  (EFGH : Type)
  (square : EFGH → Prop)
  (side_length : EFGH → ℝ)
  (area : EFGH → ℝ)
  (shaded_area : EFGH → ℝ)
  (P : EFGH)
  (h_square : square P)
  (h_side_length : side_length P = 8)
  (h_area : area P = side_length P * side_length P)
  (h_small_shaded : shaded_area P = 4)
  (h_large_shaded : shaded_area P + 7 = 11) :
  (shaded_area P / area P) * 100 = 17.1875 :=
by
  sorry

end percentage_shaded_in_square_l128_128771


namespace average_price_of_dvds_l128_128378

theorem average_price_of_dvds :
  let num_dvds_box1 := 10
  let price_per_dvd_box1 := 2.00
  let num_dvds_box2 := 5
  let price_per_dvd_box2 := 5.00
  let total_cost_box1 := num_dvds_box1 * price_per_dvd_box1
  let total_cost_box2 := num_dvds_box2 * price_per_dvd_box2
  let total_dvds := num_dvds_box1 + num_dvds_box2
  let total_cost := total_cost_box1 + total_cost_box2
  (total_cost / total_dvds) = 3.00 := 
sorry

end average_price_of_dvds_l128_128378


namespace find_integer_l128_128059

theorem find_integer (n : ℤ) (h1 : n + 10 > 11) (h2 : -4 * n > -12) : 
  n = 2 :=
sorry

end find_integer_l128_128059


namespace perimeter_is_32_l128_128800

-- Define the side lengths of the triangle
def a : ℕ := 13
def b : ℕ := 9
def c : ℕ := 10

-- Definition of the perimeter of the triangle
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- Theorem stating the perimeter is 32
theorem perimeter_is_32 : perimeter a b c = 32 :=
by
  sorry

end perimeter_is_32_l128_128800


namespace find_radius_l128_128727

theorem find_radius
  (r_1 r_2 r_3 : ℝ)
  (h_cone : r_2 = 2 * r_1 ∧ r_3 = 3 * r_1 ∧ r_1 + r_2 + r_3 = 18) :
  r_1 = 3 :=
by
  sorry

end find_radius_l128_128727


namespace last_digit_of_4_over_3_power_5_l128_128765

noncomputable def last_digit_of_fraction (n d : ℕ) : ℕ :=
  (n * 10^5 / d) % 10

def four : ℕ := 4
def three_power_five : ℕ := 3^5

theorem last_digit_of_4_over_3_power_5 :
  last_digit_of_fraction four three_power_five = 7 :=
by
  sorry

end last_digit_of_4_over_3_power_5_l128_128765


namespace number_of_mowers_l128_128798

noncomputable section

def area_larger_meadow (A : ℝ) : ℝ := 2 * A

def team_half_day_work (K a : ℝ) : ℝ := (K * a) / 2

def team_remaining_larger_meadow (K a : ℝ) : ℝ := (K * a) / 2

def half_team_half_day_work (K a : ℝ) : ℝ := (K * a) / 4

def larger_meadow_area_leq_sum (K a A : ℝ) : Prop :=
  team_half_day_work K a + team_remaining_larger_meadow K a = 2 * A

def smaller_meadow_area_left (K a A : ℝ) : ℝ :=
  A - half_team_half_day_work K a

def one_mower_one_day_work_rate (K a : ℝ) : ℝ := (K * a) / 4

def eq_total_mowed_by_team (K a A : ℝ) : Prop :=
  larger_meadow_area_leq_sum K a A ∧ smaller_meadow_area_left K a A = (K * a) / 4

theorem number_of_mowers
  (K a A b : ℝ)
  (h1 : larger_meadow_area_leq_sum K a A)
  (h2 : smaller_meadow_area_left K a A = one_mower_one_day_work_rate K a)
  (h3 : one_mower_one_day_work_rate K a = b)
  (h4 : K * a = 2 * A)
  (h5 : 2 * A = 4 * b)
  : K = 8 :=
  sorry

end number_of_mowers_l128_128798


namespace solution_set_of_inequality_l128_128896

theorem solution_set_of_inequality :
  { x : ℝ | x^2 + x - 2 < 0 } = { x : ℝ | -2 < x ∧ x < 1 } :=
sorry

end solution_set_of_inequality_l128_128896


namespace average_growth_rate_le_max_growth_rate_l128_128647

variable (P : ℝ) (a : ℝ) (b : ℝ) (x : ℝ)

theorem average_growth_rate_le_max_growth_rate (h : (1 + x)^2 = (1 + a) * (1 + b)) :
  x ≤ max a b := 
sorry

end average_growth_rate_le_max_growth_rate_l128_128647


namespace remainder_when_divided_by_5_l128_128468

-- Definitions of the conditions
def condition1 (N : ℤ) : Prop := ∃ R1 : ℤ, N = 5 * 2 + R1
def condition2 (N : ℤ) : Prop := ∃ Q2 : ℤ, N = 4 * Q2 + 2

-- Statement to prove
theorem remainder_when_divided_by_5 (N : ℤ) (R1 : ℤ) (Q2 : ℤ) :
  (N = 5 * 2 + R1) ∧ (N = 4 * Q2 + 2) → (R1 = 4) :=
by
  sorry

end remainder_when_divided_by_5_l128_128468


namespace sqrt_expression_evaluation_l128_128901

theorem sqrt_expression_evaluation : (Real.sqrt (1 + Real.sqrt (1 + Real.sqrt 1)))^4 = 3 + 2 * Real.sqrt 2 := 
by
  sorry

end sqrt_expression_evaluation_l128_128901


namespace representable_as_product_l128_128083

theorem representable_as_product (n : ℤ) (p q : ℚ) (h1 : n > 1995) (h2 : 0 < p) (h3 : p < 1) :
  ∃ (terms : List ℚ), p = terms.prod ∧ ∀ t ∈ terms, ∃ n, t = (n^2 - 1995^2) / (n^2 - 1994^2) ∧ n > 1995 :=
sorry

end representable_as_product_l128_128083


namespace line_contains_point_iff_k_eq_neg1_l128_128820

theorem line_contains_point_iff_k_eq_neg1 (k : ℝ) :
  (∃ x y : ℝ, x = 2 ∧ y = -1 ∧ (2 - k * x = -4 * y)) ↔ k = -1 :=
by
  sorry

end line_contains_point_iff_k_eq_neg1_l128_128820


namespace cos_135_eq_neg_inv_sqrt2_l128_128955

theorem cos_135_eq_neg_inv_sqrt2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt2_l128_128955


namespace sin_30_eq_half_l128_128314

noncomputable def P : ℝ × ℝ := 
  have angle := real.pi / 6
  have x := real.cos angle
  have y := real.sin angle
  (x, y)  -- The point on the unit circle at 30 degrees

theorem sin_30_eq_half : real.sin (real.pi / 6) = 1 / 2 := by
  sorry

end sin_30_eq_half_l128_128314


namespace expected_value_of_three_marbles_l128_128713

-- Define the set of marbles
def marbles := {1, 2, 3, 4, 5, 6}

-- Define the set of possible combinations of drawing 3 marbles
def combinations := marbles.powerset.filter (λ s, s.card = 3)

-- Define the sum of the elements in a set
def sum_set (s : Finset ℕ) : ℕ := s.sum id

-- Define the expected value of the sum of the numbers on the drawn marbles
def expected_value : ℚ :=
  (Finset.sum combinations sum_set : ℚ) / combinations.card

theorem expected_value_of_three_marbles :
  expected_value = 10.05 := sorry

end expected_value_of_three_marbles_l128_128713


namespace sin_30_is_half_l128_128352

noncomputable def sin_30_degrees : ℝ := Real.sin (Real.pi / 6)

theorem sin_30_is_half : sin_30_degrees = 1 / 2 := by
  sorry

end sin_30_is_half_l128_128352


namespace sums_correct_l128_128629

theorem sums_correct (x : ℕ) (h : x + 2 * x = 48) : x = 16 :=
by
  sorry

end sums_correct_l128_128629


namespace roots_quadratic_l128_128422

theorem roots_quadratic (m x₁ x₂ : ℝ) (h : m < 0) (h₁ : x₁ < x₂) (hx : ∀ x, (x^2 - x - 6 = m) ↔ (x = x₁ ∨ x = x₂)) : 
  -2 < x₁ ∧ x₁ < x₂ ∧ x₂ < 3 :=
by {
  sorry
}

end roots_quadratic_l128_128422


namespace average_first_three_numbers_l128_128595

theorem average_first_three_numbers (A B C D : ℝ) 
  (hA : A = 33) 
  (hD : D = 18)
  (hBCD : (B + C + D) / 3 = 15) : 
  (A + B + C) / 3 = 20 := 
by 
  sorry

end average_first_three_numbers_l128_128595


namespace probability_sum_is_5_l128_128486

open ProbabilityTheory MeasureTheory

-- Define the dice with numbers 0 through 5.
def diceNumber : Fin 6 := {
  zero,
  one,
  two,
  three,
  four,
  five
}

-- Define the sample space for a single die (six faces).
def dieOutcome := {0, 1, 2, 3, 4, 5}

-- Define the sample space for two dice.
def sampleSpace := (dieOutcome × dieOutcome)

-- Define the event of interest: Sum of top faces is 5.
def eventSum5 (outcome : sampleSpace) : Prop := 
  outcome.1 + outcome.2 = 5

-- Define the probability space.
def probabilitySpace : ProbabilitySpace sampleSpace := prodUniformSpace dieOutcome dieOutcome

-- Define the probability of an event.
noncomputable def probability (e : set sampleSpace) : ℝ :=
  (measureOf probabilitySpace).measure e / (measureOf probabilitySpace).measure univ

theorem probability_sum_is_5 : probability {outcome | eventSum5 outcome} = 1 / 3 :=
by sorry

end probability_sum_is_5_l128_128486


namespace proof_cos_135_degree_l128_128996

noncomputable def cos_135_degree_is_negative_sqrt2_over_2 : Prop :=
  real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2

theorem proof_cos_135_degree : cos_135_degree_is_negative_sqrt2_over_2 :=
by
  sorry

end proof_cos_135_degree_l128_128996


namespace outer_boundary_diameter_l128_128569

theorem outer_boundary_diameter (statue_width garden_width path_width fountain_diameter : ℝ) 
  (h_statue : statue_width = 2) 
  (h_garden : garden_width = 10) 
  (h_path : path_width = 8) 
  (h_fountain : fountain_diameter = 12) : 
  2 * ((fountain_diameter / 2 + statue_width) + garden_width + path_width) = 52 :=
by
  sorry

end outer_boundary_diameter_l128_128569


namespace proof_cos_135_degree_l128_128997

noncomputable def cos_135_degree_is_negative_sqrt2_over_2 : Prop :=
  real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2

theorem proof_cos_135_degree : cos_135_degree_is_negative_sqrt2_over_2 :=
by
  sorry

end proof_cos_135_degree_l128_128997


namespace ferris_wheel_seats_l128_128024

variable (total_people : ℕ) (people_per_seat : ℕ)

theorem ferris_wheel_seats (h1 : total_people = 18) (h2 : people_per_seat = 9) : total_people / people_per_seat = 2 := by
  sorry

end ferris_wheel_seats_l128_128024


namespace midpoint_trajectory_l128_128136

-- Define the data for the problem
variable (P : (ℝ × ℝ)) (Q : ℝ × ℝ)
variable (M : ℝ × ℝ)
variable (x y : ℝ)
variable (hQ : Q = (2*x - 2, 2*y)) -- Definition of point Q based on midpoint M
variable (hC : (Q.1)^2 + (Q.2)^2 = 1) -- Q moves on the circle x^2 + y^2 = 1

-- Define the proof problem
theorem midpoint_trajectory (P : (ℝ × ℝ)) (hP : P = (2, 0)) (M : ℝ × ℝ) (hQ : Q = (2*M.1 - 2, 2*M.2))
  (hC : (Q.1)^2 + (Q.2)^2 = 1) : 4*(M.1 - 1)^2 + 4*(M.2)^2 = 1 := by
  sorry

end midpoint_trajectory_l128_128136


namespace gcd_polynomials_l128_128374

def even_multiple_of_2927 (a : ℤ) : Prop := ∃ k : ℤ, a = 2 * 2927 * k

theorem gcd_polynomials (a : ℤ) (h : even_multiple_of_2927 a) :
  Int.gcd (3 * a ^ 2 + 61 * a + 143) (a + 19) = 7 :=
by
  sorry

end gcd_polynomials_l128_128374


namespace line_through_points_l128_128882

theorem line_through_points (m b: ℝ) 
  (h1: ∃ m, ∀ x y : ℝ, ((x, y) = (1, 3) ∨ (x, y) = (3, 7)) → y = m * x + b) 
  (h2: ∀ x y : ℝ, ((x, y) = (1, 3) ∨ (x, y) = (3, 7)) → y = m * x + b):
  m + b = 3 :=
by
  -- proof goes here
  sorry

end line_through_points_l128_128882


namespace sin_30_eq_half_l128_128347

theorem sin_30_eq_half
  (Q : ℝ × ℝ)
  (h1 : Q = (cos (π / 6), sin (π / 6)))
  (E : ℝ × ℝ)
  (h2 : E = (Q.1, 0))
  (h3 : ∃ x y z : ℝ, x / 2 = Q.2 ∧ y = Q.1 ∧ sqrt (x^2 + y^2) = z ∧ z = 1) :
  sin (π / 6) = 1 / 2 :=
by
  sorry

end sin_30_eq_half_l128_128347


namespace abs_inequality_solution_l128_128693

theorem abs_inequality_solution (x : ℝ) : 2 * |x - 1| - 1 < 0 ↔ (1 / 2 < x ∧ x < 3 / 2) :=
by
  sorry

end abs_inequality_solution_l128_128693


namespace ms_smith_books_divided_l128_128757

theorem ms_smith_books_divided (books_for_girls : ℕ) (girls boys : ℕ) (books_per_girl : ℕ)
  (h1 : books_for_girls = 225)
  (h2 : girls = 15)
  (h3 : boys = 10)
  (h4 : books_for_girls / girls = books_per_girl)
  (h5 : books_per_girl * boys + books_for_girls = 375) : 
  books_for_girls / girls * (girls + boys) = 375 := 
by
  sorry

end ms_smith_books_divided_l128_128757


namespace distinct_permutations_of_12233_l128_128436

def numFiveDigitIntegers : ℕ :=
  Nat.factorial 5 / (Nat.factorial 2 * Nat.factorial 2)

theorem distinct_permutations_of_12233 : numFiveDigitIntegers = 30 := by
  sorry

end distinct_permutations_of_12233_l128_128436


namespace original_number_is_16_l128_128493

theorem original_number_is_16 (x : ℤ) (h1 : 3 * (2 * x + 5) = 111) : x = 16 :=
by
  sorry

end original_number_is_16_l128_128493


namespace AK_eq_CK_l128_128225

variable {α : Type*} [LinearOrder α] [LinearOrder ℝ]

variable (A B C L K : ℝ)
variable (triangle : ℝ)
variable (h₁ : AL = LB)
variable (h₂ : AK = CL)

--  Given that in triangle ABC,
--     AL is a bisector such that AL = LB,
--     and AK is on ray AL with AK = CL,
--     prove that AK = CK.
theorem AK_eq_CK (h₁ : AL = LB) (h₂ : AK = CL) : AK = CK := by
  sorry

end AK_eq_CK_l128_128225


namespace total_payment_mr_benson_made_l128_128927

noncomputable def general_admission_ticket_cost : ℝ := 40
noncomputable def num_general_admission_tickets : ℕ := 10
noncomputable def num_vip_tickets : ℕ := 3
noncomputable def num_premium_tickets : ℕ := 2
noncomputable def vip_ticket_rate_increase : ℝ := 0.20
noncomputable def premium_ticket_rate_increase : ℝ := 0.50
noncomputable def discount_rate : ℝ := 0.05
noncomputable def threshold_tickets : ℕ := 10

noncomputable def vip_ticket_cost : ℝ := general_admission_ticket_cost * (1 + vip_ticket_rate_increase)
noncomputable def premium_ticket_cost : ℝ := general_admission_ticket_cost * (1 + premium_ticket_rate_increase)

noncomputable def total_general_admission_cost : ℝ := num_general_admission_tickets * general_admission_ticket_cost
noncomputable def total_vip_cost : ℝ := num_vip_tickets * vip_ticket_cost
noncomputable def total_premium_cost : ℝ := num_premium_tickets * premium_ticket_cost

noncomputable def total_tickets : ℕ := num_general_admission_tickets + num_vip_tickets + num_premium_tickets
noncomputable def tickets_exceeding_threshold : ℕ := if total_tickets > threshold_tickets then total_tickets - threshold_tickets else 0

noncomputable def discounted_vip_cost : ℝ := vip_ticket_cost * (1 - discount_rate)
noncomputable def discounted_premium_cost : ℝ := premium_ticket_cost * (1 - discount_rate)

noncomputable def total_discounted_vip_cost : ℝ :=  num_vip_tickets * discounted_vip_cost
noncomputable def total_discounted_premium_cost : ℝ := num_premium_tickets * discounted_premium_cost

noncomputable def total_cost_with_discounts : ℝ := total_general_admission_cost + total_discounted_vip_cost + total_discounted_premium_cost

theorem total_payment_mr_benson_made : total_cost_with_discounts = 650.80 :=
by
  -- Proof is omitted
  sorry

end total_payment_mr_benson_made_l128_128927


namespace no_possible_stack_of_1997_sum_l128_128797

theorem no_possible_stack_of_1997_sum :
  ¬ ∃ k : ℕ, 6 * k = 3 * 1997 := by
  sorry

end no_possible_stack_of_1997_sum_l128_128797


namespace number_composite_l128_128874

theorem number_composite (n : ℕ) : 
  n = 10^(2^1974 + 2^1000 - 1) + 1 →
  ∃ a b : ℕ, 1 < a ∧ a < n ∧ n = a * b :=
by sorry

end number_composite_l128_128874


namespace proof_min_k_l128_128379

-- Define the number of teachers
def num_teachers : ℕ := 200

-- Define what it means for a teacher to send a message to another teacher.
-- Represent this as a function where each teacher sends a message to exactly one other teacher.
def sends_message (teachers : Fin num_teachers → Fin num_teachers) : Prop :=
  ∀ i : Fin num_teachers, ∃ j : Fin num_teachers, teachers i = j

-- Define the main proposition: there exists a group of 67 teachers where no one sends a message to anyone else in the group.
def min_k (teachers : Fin num_teachers → Fin num_teachers) : Prop :=
  ∃ (k : ℕ) (reps : Fin k → Fin num_teachers), k ≥ 67 ∧
  ∀ (i j : Fin k), i ≠ j → teachers (reps i) ≠ reps j

theorem proof_min_k : ∀ (teachers : Fin num_teachers → Fin num_teachers),
  sends_message teachers → min_k teachers :=
sorry

end proof_min_k_l128_128379


namespace part1_part2_l128_128551

open Set

def A (x : ℝ) : Prop := -1 < x ∧ x < 6
def B (x : ℝ) (a : ℝ) : Prop := x ≥ 1 + a ∨ x ≤ 1 - a

theorem part1 (a : ℝ) (hpos : 0 < a) :
  (∀ x, A x → ¬ B x a) ↔ a ≥ 5 :=
sorry

theorem part2 (a : ℝ) (hpos : 0 < a) :
  (∀ x, (¬ A x → B x a) ∧ ∃ x, ¬ A x ∧ ¬ B x a) ↔ (0 < a ∧ a ≤ 2) :=
sorry

end part1_part2_l128_128551


namespace problem_statement_l128_128611

theorem problem_statement {f : ℝ → ℝ}
  (Hodd : ∀ x, f (-x) = -f x)
  (Hdecreasing : ∀ x y, x < y → f x > f y)
  (a b : ℝ) (H : f a + f b > 0) : a + b < 0 :=
sorry

end problem_statement_l128_128611


namespace math_problem_l128_128091

noncomputable def a : ℕ := 1265
noncomputable def b : ℕ := 168
noncomputable def c : ℕ := 21
noncomputable def d : ℕ := 6
noncomputable def e : ℕ := 3

theorem math_problem : 
  ( ( b / 100 : ℚ ) * (a ^ 2 / c) / (d - e ^ 2) : ℚ ) = -42646.27 :=
by sorry

end math_problem_l128_128091


namespace find_sum_of_pqr_l128_128785

theorem find_sum_of_pqr (p q r : ℝ) (h1 : p ≠ q) (h2 : q ≠ r) (h3 : p ≠ r) 
(h4 : q = p * (4 - p)) (h5 : r = q * (4 - q)) (h6 : p = r * (4 - r)) : 
p + q + r = 6 :=
sorry

end find_sum_of_pqr_l128_128785


namespace sum_xyz_le_two_l128_128575

theorem sum_xyz_le_two (x y z : ℝ) (h : 2 * x + y^2 + z^2 ≤ 2) : x + y + z ≤ 2 :=
sorry

end sum_xyz_le_two_l128_128575


namespace find_x_l128_128417

variable (x : ℝ)
variable (l : ℝ) (w : ℝ)

def length := 4 * x + 1
def width := x + 7

theorem find_x (h1 : l = length x) (h2 : w = width x) (h3 : l * w = 2 * (2 * l + 2 * w)) :
  x = (-9 + Real.sqrt 481) / 8 :=
by
  subst_vars
  sorry

end find_x_l128_128417


namespace distance_between_meeting_points_is_48_l128_128621

noncomputable def distance_between_meeting_points 
    (d : ℝ) -- total distance between points A and B
    (first_meeting_from_B : ℝ)   -- distance of the first meeting point from B
    (second_meeting_from_A : ℝ) -- distance of the second meeting point from A
    (second_meeting_from_B : ℝ) : ℝ :=
    (second_meeting_from_B - first_meeting_from_B)

theorem distance_between_meeting_points_is_48 
    (d : ℝ)
    (hm1 : first_meeting_from_B = 108)
    (hm2 : second_meeting_from_A = 84) 
    (hm3 : second_meeting_from_B = d - 24) :
    distance_between_meeting_points d first_meeting_from_B second_meeting_from_A second_meeting_from_B = 48 := by
  sorry

end distance_between_meeting_points_is_48_l128_128621


namespace ratio_of_logs_l128_128879

noncomputable def log_base (b x : ℝ) := (Real.log x) / (Real.log b)

theorem ratio_of_logs (a b : ℝ) 
    (h1 : 0 < a) (h2 : 0 < b) 
    (h3 : log_base 8 a = log_base 18 b)
    (h4 : log_base 18 b = log_base 32 (a + b)) : 
    b / a = (1 + Real.sqrt 5) / 2 :=
by sorry

end ratio_of_logs_l128_128879


namespace task_pages_l128_128626

theorem task_pages (A B T : ℕ) (hB : B = A + 5) (hTogether : (A + B) * 18 = T)
  (hAlone : A * 60 = T) : T = 225 :=
by
  sorry

end task_pages_l128_128626


namespace compound_interest_l128_128783

theorem compound_interest (SI : ℝ) (P : ℝ) (R : ℝ) (T : ℝ) (CI : ℝ) :
  SI = 50 →
  R = 5 →
  T = 2 →
  P = (SI * 100) / (R * T) →
  CI = P * (1 + R / 100)^T - P →
  CI = 51.25 :=
by
  intros
  exact sorry -- This placeholder represents the proof that would need to be filled in 

end compound_interest_l128_128783


namespace skittles_distribution_l128_128018

theorem skittles_distribution :
  let initial_skittles := 14
  let additional_skittles := 22
  let total_skittles := initial_skittles + additional_skittles
  let number_of_people := 7
  (total_skittles / number_of_people = 5) :=
by
  sorry

end skittles_distribution_l128_128018


namespace smallest_solution_exists_l128_128691

noncomputable def is_solution (x : ℝ) : Prop := (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧ x ≠ 3 ∧ x ≠ 5 ∧ x ≠ 4

-- Statement of the problem without proof
theorem smallest_solution_exists : ∃ (x : ℝ), is_solution x ∧ ∀ (y : ℝ), is_solution y → x ≤ y :=
sorry

end smallest_solution_exists_l128_128691


namespace problem_solution_l128_128498

theorem problem_solution (N : ℚ) (h : (4/5) * (3/8) * N = 24) : 2.5 * N = 200 :=
by {
  sorry
}

end problem_solution_l128_128498


namespace frank_oranges_correct_l128_128120

def betty_oranges : ℕ := 12
def sandra_oranges : ℕ := 3 * betty_oranges
def emily_oranges : ℕ := 7 * sandra_oranges
def frank_oranges : ℕ := 5 * emily_oranges

theorem frank_oranges_correct : frank_oranges = 1260 := by
  sorry

end frank_oranges_correct_l128_128120


namespace expression_value_l128_128250

theorem expression_value (x y z : ℕ) (hx : x = 2) (hy : y = 5) (hz : z = 3) :
  (3 * x^5 + 4 * y^3 + z^2) / 7 = 605 / 7 := by
  rw [hx, hy, hz]
  sorry

end expression_value_l128_128250


namespace sum_of_squares_l128_128598

theorem sum_of_squares (x y : ℤ) (h : ∃ k : ℤ, (x^2 + y^2) = 5 * k) : 
  ∃ a b : ℤ, (x^2 + y^2) / 5 = a^2 + b^2 :=
by sorry

end sum_of_squares_l128_128598


namespace ratio_of_goals_l128_128119

-- The conditions
def first_period_goals_kickers : ℕ := 2
def second_period_goals_kickers := 4
def first_period_goals_spiders := first_period_goals_kickers / 2
def second_period_goals_spiders := 2 * second_period_goals_kickers
def total_goals := first_period_goals_kickers + second_period_goals_kickers + first_period_goals_spiders + second_period_goals_spiders

-- The ratio to prove
def ratio_goals : ℕ := second_period_goals_kickers / first_period_goals_kickers

theorem ratio_of_goals : total_goals = 15 → ratio_goals = 2 := by
  intro h
  sorry

end ratio_of_goals_l128_128119


namespace isosceles_triangle_base_length_l128_128576

open Real

noncomputable def average_distance_sun_earth : ℝ := 1.5 * 10^8 -- in kilometers
noncomputable def base_length_given_angle_one_second (legs_length : ℝ) : ℝ := 4.848 -- in millimeters when legs are 1 kilometer

theorem isosceles_triangle_base_length 
  (vertex_angle : ℝ) (legs_length : ℝ) 
  (h1 : vertex_angle = 1 / 3600) 
  (h2 : legs_length = average_distance_sun_earth) : 
  ∃ base_length: ℝ, base_length = 727.2 := 
by 
  sorry

end isosceles_triangle_base_length_l128_128576


namespace option2_is_cheaper_l128_128934

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def final_price_option1 (initial_price : ℝ) : ℝ :=
  let price_after_first_discount := apply_discount initial_price 0.30
  let price_after_second_discount := apply_discount price_after_first_discount 0.10
  apply_discount price_after_second_discount 0.05

def final_price_option2 (initial_price : ℝ) : ℝ :=
  let price_after_first_discount := apply_discount initial_price 0.30
  let price_after_second_discount := apply_discount price_after_first_discount 0.05
  apply_discount price_after_second_discount 0.15

theorem option2_is_cheaper (initial_price : ℝ) (h : initial_price = 12000) :
  final_price_option2 initial_price = 6783 ∧ final_price_option1 initial_price = 7182 → 6783 < 7182 :=
by
  intros
  sorry

end option2_is_cheaper_l128_128934


namespace cos_135_eq_neg_sqrt2_div_2_l128_128951

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
by 
  sorry

end cos_135_eq_neg_sqrt2_div_2_l128_128951


namespace sum_of_coordinates_after_reflections_l128_128873

theorem sum_of_coordinates_after_reflections :
  let A := (3, 2)
  let B := (9, 18)
  let N := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let reflect_y (P : ℤ × ℤ) := (-P.1, P.2)
  let reflect_x (P : ℤ × ℤ) := (P.1, -P.2)
  let N' := reflect_y N
  let N'' := reflect_x N'
  N''.1 + N''.2 = -16 := by sorry

end sum_of_coordinates_after_reflections_l128_128873


namespace cone_volume_l128_128153

theorem cone_volume (lateral_area : ℝ) (angle : ℝ) 
  (h₀ : lateral_area = 20 * Real.pi)
  (h₁ : angle = Real.arccos (4/5)) : 
  (1/3) * Real.pi * (4^2) * 3 = 16 * Real.pi :=
by
  sorry

end cone_volume_l128_128153


namespace sum_of_positive_factors_of_36_l128_128073

theorem sum_of_positive_factors_of_36 : 
  let n := 36 in
  ∑ i in {0, 1, 2}, (2 ^ i) * ∑ j in {0, 1, 2}, (3 ^ j) = 91 :=
by
  let n := 36
  let sum_p := ∑ i in {0, 1, 2}, (2 ^ i)
  let sum_q := ∑ j in {0, 1, 2}, (3 ^ j)
  have h1 : sum_p = 1 + 2 + 4 := by
    sorry
  have h2 : sum_q = 1 + 3 + 9 := by
    sorry
  have h3 : sum_p * sum_q = 91 := by
    sorry
  exact h3

end sum_of_positive_factors_of_36_l128_128073


namespace cos_135_eq_neg_sqrt_two_div_two_l128_128972

theorem cos_135_eq_neg_sqrt_two_div_two : real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt_two_div_two_l128_128972


namespace smallest_solution_l128_128687

-- Defining the equation as a condition
def equation (x : ℝ) : Prop := (1 / (x - 3)) + (1 / (x - 5)) = 4 / (x - 4)

-- Proving that the smallest solution is 4 - sqrt(2)
theorem smallest_solution : ∃ x : ℝ, equation x ∧ x = 4 - Real.sqrt 2 := 
by
  -- Proof is omitted
  sorry

end smallest_solution_l128_128687


namespace fraction_invariant_l128_128565

variable {R : Type*} [Field R]
variables (x y : R)

theorem fraction_invariant : (2 * x) / (3 * x - y) = (6 * x) / (9 * x - 3 * y) :=
by
  sorry

end fraction_invariant_l128_128565


namespace intersection_of_A_and_B_l128_128137

def A : Set ℤ := { -3, -1, 0, 1 }
def B : Set ℤ := { x | (-2 < x) ∧ (x < 1) }

theorem intersection_of_A_and_B : A ∩ B = { -1, 0 } := by
  sorry

end intersection_of_A_and_B_l128_128137


namespace negation_equiv_l128_128430

variable (p : Prop) [Nonempty ℝ]

def proposition := ∃ x : ℝ, Real.exp x - x - 1 ≤ 0

def negation_of_proposition : Prop := ∀ x : ℝ, Real.exp x - x - 1 > 0

theorem negation_equiv
  (h : proposition = p) : (¬ proposition) = negation_of_proposition := by
  sorry

end negation_equiv_l128_128430


namespace sin_30_eq_half_l128_128273
open real

noncomputable def sin_30_deg : ℝ :=
  sin (π / 6)

theorem sin_30_eq_half : sin_30_deg = 1 / 2 :=
by
  -- Given the conditions of the problem, we need to prove that sin(π / 6) = 1 / 2
  sorry

end sin_30_eq_half_l128_128273


namespace probability_three_hearts_l128_128054

noncomputable def probability_of_three_hearts : ℚ :=
  (13/52) * (12/51) * (11/50)

theorem probability_three_hearts :
  probability_of_three_hearts = 26/2025 :=
by
  sorry

end probability_three_hearts_l128_128054


namespace nonneg_real_inequality_l128_128462

theorem nonneg_real_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  a^4 + b^4 + c^4 - 2 * (a^2 * b^2 + a^2 * c^2 + b^2 * c^2) + a^2 * b * c + b^2 * a * c + c^2 * a * b ≥ 0 := 
by
  sorry

end nonneg_real_inequality_l128_128462


namespace calculate_share_A_l128_128514

-- Defining the investments
def investment_A : ℕ := 7000
def investment_B : ℕ := 11000
def investment_C : ℕ := 18000
def investment_D : ℕ := 13000
def investment_E : ℕ := 21000
def investment_F : ℕ := 15000
def investment_G : ℕ := 9000

-- Defining B's share
def share_B : ℚ := 3600

-- Function to calculate total investment
def total_investment : ℕ :=
  investment_A + investment_B + investment_C + investment_D + investment_E + investment_F + investment_G

-- Ratio of B's investment to total investment
def ratio_B : ℚ :=
  investment_B / total_investment

-- Calculate total profit using B's share and ratio
def total_profit : ℚ :=
  share_B / ratio_B

-- Ratio of A's investment to total investment
def ratio_A : ℚ :=
  investment_A / total_investment

-- Calculate A's share based on the total profit
def share_A : ℚ :=
  total_profit * ratio_A

-- The theorem to prove the share of A is approximately $2292.34
theorem calculate_share_A : 
  abs (share_A - 2292.34) < 0.01 :=
by
  sorry

end calculate_share_A_l128_128514


namespace non_negative_real_inequality_l128_128461

theorem non_negative_real_inequality
  {a b c : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  a^4 + b^4 + c^4 - 2 * (a^2 * b^2 + a^2 * c^2 + b^2 * c^2) + a^2 * b * c + b^2 * a * c + c^2 * a * b ≥ 0 :=
by
  sorry

end non_negative_real_inequality_l128_128461


namespace percent_of_value_and_divide_l128_128219

theorem percent_of_value_and_divide (x : ℝ) (y : ℝ) (z : ℝ) (h : x = 1/300 * 180) (h1 : y = x / 6) : 
  y = 0.1 := 
by
  sorry

end percent_of_value_and_divide_l128_128219


namespace circle_radius_l128_128230

theorem circle_radius (r x y : ℝ) (hx : x = π * r^2) (hy : y = 2 * π * r) (h : x + y = 90 * π) : r = 9 := by
  sorry

end circle_radius_l128_128230


namespace sin_30_eq_half_l128_128364

theorem sin_30_eq_half : Real.sin (Real.pi / 6) = 1 / 2 := sorry

end sin_30_eq_half_l128_128364


namespace burger_cost_l128_128694

theorem burger_cost (days_in_june : ℕ) (burgers_per_day : ℕ) (total_spent : ℕ) (h1 : days_in_june = 30) (h2 : burgers_per_day = 2) (h3 : total_spent = 720) : 
  total_spent / (burgers_per_day * days_in_june) = 12 :=
by
  -- We will prove this in Lean, but skipping the proof here
  sorry

end burger_cost_l128_128694


namespace exists_i_for_inequality_l128_128864

theorem exists_i_for_inequality (n : ℕ) (x : ℕ → ℝ) (h1 : 2 ≤ n) 
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ n → 0 ≤ x i ∧ x i ≤ 1) :
  ∃ i, 1 ≤ i ∧ i ≤ n - 1 ∧ x i * (1 - x (i + 1)) ≥ (1 / 4) * x 1 * (1 - x n) :=
by
  sorry

end exists_i_for_inequality_l128_128864


namespace eulers_polyhedron_theorem_l128_128121

theorem eulers_polyhedron_theorem 
  (V E F t h : ℕ) (T H : ℕ) :
  (F = 30) →
  (t = 20) →
  (h = 10) →
  (T = 3) →
  (H = 2) →
  (E = (3 * t + 6 * h) / 2) →
  (V - E + F = 2) →
  100 * H + 10 * T + V = 262 :=
by
  intros F_eq t_eq h_eq T_eq H_eq E_eq euler_eq
  rw [F_eq, t_eq, h_eq, T_eq, H_eq, E_eq] at *
  sorry

end eulers_polyhedron_theorem_l128_128121


namespace find_vector_l128_128813

noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  (2 + 3 * t, 3 + 2 * t)

noncomputable def line_m (s : ℝ) : ℝ × ℝ :=
  (-4 + 3 * s, 5 + 2 * s)

def vector_condition (v1 v2 : ℝ) : Prop :=
  v1 - v2 = 1

theorem find_vector :
  ∃ (v1 v2 : ℝ), vector_condition v1 v2 ∧ (v1, v2) = (3, 2) :=
sorry

end find_vector_l128_128813


namespace equal_charges_at_x_l128_128106

theorem equal_charges_at_x (x : ℝ) : 
  (2.75 * x + 125 = 1.50 * x + 140) → (x = 12) := 
by
  sorry

end equal_charges_at_x_l128_128106


namespace evaluate_expression_l128_128115

theorem evaluate_expression : (120 / 6 * 2 / 3 = (40 / 3)) := 
by sorry

end evaluate_expression_l128_128115


namespace hawks_points_l128_128443

theorem hawks_points (x y : ℕ) (h1 : x + y = 50) (h2 : x + 4 - y = 12) : y = 21 :=
by
  sorry

end hawks_points_l128_128443


namespace troy_buys_beef_l128_128057

theorem troy_buys_beef (B : ℕ) 
  (veg_pounds : ℕ := 6)
  (veg_cost_per_pound : ℕ := 2)
  (beef_cost_per_pound : ℕ := 3 * veg_cost_per_pound)
  (total_cost : ℕ := 36) :
  6 * veg_cost_per_pound + B * beef_cost_per_pound = total_cost → B = 4 :=
by
  sorry

end troy_buys_beef_l128_128057


namespace costume_processing_time_l128_128639

theorem costume_processing_time (x : ℕ) : 
  (300 - 60) / (2 * x) + 60 / x = 9 → (60 / x) + (240 / (2 * x)) = 9 :=
by
  sorry

end costume_processing_time_l128_128639


namespace speed_upstream_calculation_l128_128482

def speed_boat_still_water : ℝ := 60
def speed_current : ℝ := 17

theorem speed_upstream_calculation : speed_boat_still_water - speed_current = 43 := by
  sorry

end speed_upstream_calculation_l128_128482
