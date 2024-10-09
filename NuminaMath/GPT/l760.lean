import Mathlib

namespace pages_printed_l760_76037

theorem pages_printed (P : ℕ) 
  (H1 : P % 7 = 0)
  (H2 : P % 3 = 0)
  (H3 : P - (P / 7 + P / 3 - P / 21) = 24) : 
  P = 42 :=
sorry

end pages_printed_l760_76037


namespace range_zero_of_roots_l760_76047

theorem range_zero_of_roots (x y z w : ℝ) (h1 : x + y + z + w = 0) 
                            (h2 : x^7 + y^7 + z^7 + w^7 = 0) :
  w * (w + x) * (w + y) * (w + z) = 0 :=
  sorry

end range_zero_of_roots_l760_76047


namespace recurrence_solution_proof_l760_76040

noncomputable def recurrence_relation (a : ℕ → ℚ) : Prop :=
  (∀ n ≥ 2, a n = 5 * a (n - 1) - 6 * a (n - 2) + n + 2) ∧
  a 0 = 27 / 4 ∧
  a 1 = 49 / 4

noncomputable def solution (a : ℕ → ℚ) : Prop :=
  ∀ n, a n = 3 * 2^n + 3^n + n / 2 + 11 / 4

theorem recurrence_solution_proof : ∃ a : ℕ → ℚ, recurrence_relation a ∧ solution a :=
by { sorry }

end recurrence_solution_proof_l760_76040


namespace min_a1_a7_l760_76044

noncomputable def geom_seq (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, ∃ r : ℝ, r > 0 ∧ a (n + 1) = a n * r

theorem min_a1_a7 (a : ℕ → ℝ) (h : geom_seq a)
  (h1 : a 3 * a 5 = 64) :
  ∃ m, m = (a 1 + a 7) ∧ m = 16 :=
by
  sorry

end min_a1_a7_l760_76044


namespace quadratic_has_two_distinct_real_roots_l760_76006

theorem quadratic_has_two_distinct_real_roots (k : ℝ) : 
  ((k - 1) * x^2 + 2 * x - 2 = 0) → (1 / 2 < k ∧ k ≠ 1) :=
sorry

end quadratic_has_two_distinct_real_roots_l760_76006


namespace julio_twice_james_in_years_l760_76073

noncomputable def years_until_julio_twice_james := 
  let x := 14
  (36 + x = 2 * (11 + x))

theorem julio_twice_james_in_years : 
  years_until_julio_twice_james := 
  by 
  sorry

end julio_twice_james_in_years_l760_76073


namespace mileage_per_gallon_l760_76094

-- Definitions for the conditions
def total_distance_to_grandma (d : ℕ) : Prop := d = 100
def gallons_to_grandma (g : ℕ) : Prop := g = 5

-- The statement to be proved
theorem mileage_per_gallon :
  ∀ (d g m : ℕ), total_distance_to_grandma d → gallons_to_grandma g → m = d / g → m = 20 :=
sorry

end mileage_per_gallon_l760_76094


namespace arithmetic_problem_l760_76033

theorem arithmetic_problem : 1357 + 3571 + 5713 - 7135 = 3506 :=
by
  sorry

end arithmetic_problem_l760_76033


namespace range_of_a_if_q_sufficient_but_not_necessary_for_p_l760_76096

variable {x a : ℝ}

def p (x : ℝ) : Prop := x > 1 ∨ x < -3
def q (x a : ℝ) : Prop := x > a

theorem range_of_a_if_q_sufficient_but_not_necessary_for_p :
  (∀ x, q x a → p x) ∧ ¬(∀ x, p x → q x a) → a ∈ Set.Ici 1 := 
sorry

end range_of_a_if_q_sufficient_but_not_necessary_for_p_l760_76096


namespace value_of_three_inch_cube_l760_76064

theorem value_of_three_inch_cube (value_two_inch: ℝ) (volume_two_inch: ℝ) (volume_three_inch: ℝ) (cost_two_inch: ℝ):
  value_two_inch = cost_two_inch * ((volume_three_inch / volume_two_inch): ℝ) := 
by
  have volume_two_inch := 2^3 -- Volume of two-inch cube
  have volume_three_inch := 3^3 -- Volume of three-inch cube
  let volume_ratio := (volume_three_inch / volume_two_inch: ℝ)
  have := cost_two_inch * volume_ratio
  norm_num
  sorry

end value_of_three_inch_cube_l760_76064


namespace price_of_pastries_is_5_l760_76055

noncomputable def price_of_reuben : ℕ := 3
def price_of_pastries (price_reuben : ℕ) : ℕ := price_reuben + 2

theorem price_of_pastries_is_5 
    (reuben_price cost_pastries : ℕ) 
    (h1 : cost_pastries = reuben_price + 2) 
    (h2 : 10 * reuben_price + 5 * cost_pastries = 55) :
    cost_pastries = 5 :=
by
    sorry

end price_of_pastries_is_5_l760_76055


namespace inequality_solution_l760_76038

theorem inequality_solution (x : ℝ) : (-3 * x^2 - 9 * x - 6 ≥ -12) ↔ (-2 ≤ x ∧ x ≤ 1) := sorry

end inequality_solution_l760_76038


namespace system_of_equations_solutions_l760_76022

theorem system_of_equations_solutions (x y a b : ℝ) 
  (h1 : 2 * x + y = b) 
  (h2 : x - b * y = a) 
  (hx : x = 1)
  (hy : y = 0) : a - b = -1 :=
by 
  sorry

end system_of_equations_solutions_l760_76022


namespace ratio_distance_traveled_by_foot_l760_76082

theorem ratio_distance_traveled_by_foot (D F B C : ℕ) (hD : D = 40) 
(hB : B = D / 2) (hC : C = 10) (hF : F = D - (B + C)) : F / D = 1 / 4 := 
by sorry

end ratio_distance_traveled_by_foot_l760_76082


namespace last_matching_date_2008_l760_76084

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- The last date in 2008 when the sum of the first four digits equals the sum of the last four digits is 25 December 2008. -/
theorem last_matching_date_2008 :
  ∃ d m y, d = 25 ∧ m = 12 ∧ y = 2008 ∧
            sum_of_digits 2512 = sum_of_digits 2008 :=
by {
  sorry
}

end last_matching_date_2008_l760_76084


namespace maria_remaining_money_l760_76065

theorem maria_remaining_money (initial_amount ticket_cost : ℕ) (h_initial : initial_amount = 760) (h_ticket : ticket_cost = 300) :
  let hotel_cost := ticket_cost / 2
  let total_spent := ticket_cost + hotel_cost
  let remaining := initial_amount - total_spent
  remaining = 310 :=
by
  intros
  sorry

end maria_remaining_money_l760_76065


namespace value_of_polynomial_at_2_l760_76036

def f (x : ℝ) : ℝ := 4 * x^5 + 2 * x^4 + 3 * x^3 - 2 * x^2 - 2500 * x + 434

theorem value_of_polynomial_at_2 : f 2 = -3390 := by
  -- proof would go here
  sorry

end value_of_polynomial_at_2_l760_76036


namespace motorcyclist_cross_time_l760_76062

/-- Definitions and conditions -/
def speed_X := 2 -- Rounds per hour
def speed_Y := 4 -- Rounds per hour

/-- Proof statement -/
theorem motorcyclist_cross_time : (1 / (speed_X + speed_Y) * 60 = 10) :=
by
  sorry

end motorcyclist_cross_time_l760_76062


namespace price_difference_l760_76043

noncomputable def original_price (discounted_price : ℝ) : ℝ :=
  discounted_price / 0.85

noncomputable def final_price (discounted_price : ℝ) : ℝ :=
  discounted_price * 1.25

theorem price_difference (discounted_price : ℝ) (h : discounted_price = 71.4) : 
  (final_price discounted_price) - (original_price discounted_price) = 5.25 := 
by
  sorry

end price_difference_l760_76043


namespace arithmetic_mean_of_1_and_4_l760_76052

theorem arithmetic_mean_of_1_and_4 : 
  (1 + 4) / 2 = 5 / 2 := by
  sorry

end arithmetic_mean_of_1_and_4_l760_76052


namespace find_a_if_perpendicular_l760_76053

def m (a : ℝ) : ℝ × ℝ := (3, a - 1)
def n (a : ℝ) : ℝ × ℝ := (a, -2)

theorem find_a_if_perpendicular (a : ℝ) (h : (m a).fst * (n a).fst + (m a).snd * (n a).snd = 0) : a = -2 :=
by sorry

end find_a_if_perpendicular_l760_76053


namespace pow_mod_remainder_l760_76012

theorem pow_mod_remainder (n : ℕ) (h : 9 ≡ 2 [MOD 7]) (h2 : 9^2 ≡ 4 [MOD 7]) (h3 : 9^3 ≡ 1 [MOD 7]) : 9^123 % 7 = 1 := by
  sorry

end pow_mod_remainder_l760_76012


namespace second_pipe_filling_time_l760_76070

theorem second_pipe_filling_time :
  ∃ T : ℝ, (1/20 + 1/T) * 2/3 * 16 = 1 ∧ T = 160/7 :=
by
  use 160 / 7
  sorry

end second_pipe_filling_time_l760_76070


namespace machine_C_time_l760_76028

theorem machine_C_time (T_c : ℝ) : 
  (1/4) + (1/3) + (1/T_c) = (3/4) → T_c = 6 := 
by 
  sorry

end machine_C_time_l760_76028


namespace students_absent_percentage_l760_76081

theorem students_absent_percentage (total_students present_students : ℕ) (h_total : total_students = 50) (h_present : present_students = 45) :
  (total_students - present_students) * 100 / total_students = 10 := 
by
  sorry

end students_absent_percentage_l760_76081


namespace square_problem_solution_l760_76061

theorem square_problem_solution
  (x : ℝ)
  (h1 : ∃ s1 : ℝ, s1^2 = x^2 + 12*x + 36)
  (h2 : ∃ s2 : ℝ, s2^2 = 4*x^2 - 12*x + 9)
  (h3 : 4 * (s1 + s2) = 64) :
  x = 13 / 3 :=
by
  sorry

end square_problem_solution_l760_76061


namespace problem_statement_l760_76056

def prop_p (x : ℝ) : Prop := x^2 >= x
def prop_q : Prop := ∃ x : ℝ, x^2 >= x

theorem problem_statement : (∀ x : ℝ, prop_p x) = false ∧ prop_q = true :=
by 
  sorry

end problem_statement_l760_76056


namespace simplify_fractions_l760_76039

theorem simplify_fractions :
  (30 / 45) * (75 / 128) * (256 / 150) = 1 / 6 := 
by
  sorry

end simplify_fractions_l760_76039


namespace students_taking_history_l760_76027

-- Defining the conditions
def num_students (total_students history_students statistics_students both_students : ℕ) : Prop :=
  total_students = 89 ∧
  statistics_students = 32 ∧
  (history_students + statistics_students - both_students) = 59 ∧
  (history_students - both_students) = 27

-- The theorem stating that given the conditions, the number of students taking history is 54
theorem students_taking_history :
  ∃ history_students, ∃ statistics_students, ∃ both_students, 
  num_students 89 history_students statistics_students both_students ∧ history_students = 54 :=
by
  sorry

end students_taking_history_l760_76027


namespace roots_quadratic_square_diff_10_l760_76021

-- Definition and theorem statement in Lean 4
theorem roots_quadratic_square_diff_10 :
  ∀ x1 x2 : ℝ, (2 * x1^2 + 4 * x1 - 3 = 0) ∧ (2 * x2^2 + 4 * x2 - 3 = 0) →
  (x1 - x2)^2 = 10 :=
by
  sorry

end roots_quadratic_square_diff_10_l760_76021


namespace range_of_m_l760_76046

-- Define the first circle
def circle1 (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 10*y + 1 = 0

-- Define the second circle
def circle2 (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y - m = 0

-- Lean statement for the proof problem
theorem range_of_m (m : ℝ) : 
  (∃ x y : ℝ, circle1 x y ∧ circle2 x y m) ↔ -1 < m ∧ m < 79 :=
by sorry

end range_of_m_l760_76046


namespace largest_circle_radius_l760_76041

theorem largest_circle_radius (a b c : ℝ) (h : a > b ∧ b > c) :
  ∃ radius : ℝ, radius = b :=
by
  sorry

end largest_circle_radius_l760_76041


namespace point_P_quadrant_IV_l760_76019

theorem point_P_quadrant_IV (x y : ℝ) (h1 : x > 0) (h2 : y < 0) : x > 0 ∧ y < 0 :=
by
  sorry

end point_P_quadrant_IV_l760_76019


namespace travel_distance_l760_76000

variables (speed time : ℕ) (distance : ℕ)

theorem travel_distance (hspeed : speed = 75) (htime : time = 4) : distance = speed * time → distance = 300 :=
by
  intros hdist
  rw [hspeed, htime] at hdist
  simp at hdist
  assumption

end travel_distance_l760_76000


namespace percentage_reduction_price_increase_for_profit_price_increase_max_profit_l760_76035

-- Define the conditions
def original_price : ℝ := 50
def final_price : ℝ := 32
def daily_sales : ℝ := 500
def profit_per_kg : ℝ := 10
def sales_decrease_per_yuan : ℝ := 20
def required_daily_profit : ℝ := 6000
def max_possible_profit : ℝ := 6125

-- Proving the percentage reduction each time
theorem percentage_reduction (a : ℝ) :
  (original_price * (1 - a) ^ 2 = final_price) → (a = 0.2) :=
sorry

-- Proving the price increase per kilogram to ensure a daily profit of 6000 yuan
theorem price_increase_for_profit (x : ℝ) :
  ((profit_per_kg + x) * (daily_sales - sales_decrease_per_yuan * x) = required_daily_profit) → (x = 5) :=
sorry

-- Proving the price increase per kilogram to maximize daily profit
theorem price_increase_max_profit (x : ℝ) :
  ((profit_per_kg + x) * (daily_sales - sales_decrease_per_yuan * x) = max_possible_profit) → (x = 7.5) :=
sorry

end percentage_reduction_price_increase_for_profit_price_increase_max_profit_l760_76035


namespace departure_sequences_count_l760_76089

noncomputable def total_departure_sequences (trains: Finset ℕ) (A B : ℕ) 
  (h : A ∈ trains ∧ B ∈ trains ∧ trains.card = 6) 
  (hAB : ∀ g1 g2 : Finset ℕ, g1 ∪ g2 = trains ∧ g1.card = 3 ∧ g2.card = 3 → ¬(A ∈ g1 ∧ B ∈ g1 ∨ A ∈ g2 ∧ B ∈ g2)) 
  : ℕ := 6 * 6 * 6

-- The main theorem statement: given the conditions, prove the total number of different sequences is 216
theorem departure_sequences_count (trains: Finset ℕ) (A B : ℕ)
  (h : A ∈ trains ∧ B ∈ trains ∧ trains.card = 6)
  (hAB : ∀ g1 g2 : Finset ℕ, g1 ∪ g2 = trains ∧ g1.card = 3 ∧ g2.card = 3 → ¬(A ∈ g1 ∧ B ∈ g1 ∨ A ∈ g2 ∧ B ∈ g2)) 
  : total_departure_sequences trains A B h hAB = 216 := 
by 
  sorry

end departure_sequences_count_l760_76089


namespace expression_simplifies_to_one_l760_76072

theorem expression_simplifies_to_one :
  ( (105^2 - 8^2) / (80^2 - 13^2) ) * ( (80 - 13) * (80 + 13) / ( (105 - 8) * (105 + 8) ) ) = 1 :=
by
  sorry

end expression_simplifies_to_one_l760_76072


namespace problem1_problem2_l760_76017

noncomputable def g (x : ℝ) : ℝ := 3 * x^5 - 4 * x^4 + 2 * x^3 - 28 * x^2 + 15 * x - 90

noncomputable def g' (x : ℝ) : ℝ := 15 * x^4 - 16 * x^3 + 6 * x^2 - 56 * x + 15

theorem problem1 : g 6 = 17568 := 
by {
  sorry
}

theorem problem2 : g' 6 = 15879 := 
by {
  sorry
}

end problem1_problem2_l760_76017


namespace congruence_problem_l760_76023

theorem congruence_problem {x : ℤ} (h : 4 * x + 5 ≡ 3 [ZMOD 20]) : 3 * x + 8 ≡ 2 [ZMOD 10] :=
sorry

end congruence_problem_l760_76023


namespace simple_interest_calculation_l760_76097

-- Defining the given values
def principal : ℕ := 1500
def rate : ℕ := 7
def time : ℕ := rate -- time is the same as the rate of interest

-- Define the simple interest calculation
def simple_interest (P R T : ℕ) : ℕ := (P * R * T) / 100

-- Proof statement
theorem simple_interest_calculation : simple_interest principal rate time = 735 := by
  sorry

end simple_interest_calculation_l760_76097


namespace pages_with_same_units_digit_count_l760_76099

def same_units_digit (x : ℕ) (y : ℕ) : Prop :=
  x % 10 = y % 10

theorem pages_with_same_units_digit_count :
  ∃! (n : ℕ), n = 12 ∧ 
  ∀ x, (1 ≤ x ∧ x ≤ 61) → same_units_digit x (62 - x) → 
  (x % 10 = 2 ∨ x % 10 = 7) :=
by
  sorry

end pages_with_same_units_digit_count_l760_76099


namespace arcsin_sqrt3_div_2_eq_pi_div_3_l760_76031

theorem arcsin_sqrt3_div_2_eq_pi_div_3 : Real.arcsin (Real.sqrt 3 / 2) = Real.pi / 3 :=
by
  sorry

end arcsin_sqrt3_div_2_eq_pi_div_3_l760_76031


namespace quadratic_inequality_solution_l760_76004

theorem quadratic_inequality_solution :
  ∀ x : ℝ, -9 * x^2 + 6 * x - 8 < 0 :=
by {
  sorry
}

end quadratic_inequality_solution_l760_76004


namespace quotient_remainder_threefold_l760_76005

theorem quotient_remainder_threefold (a b c d : ℤ)
  (h : a = b * c + d) :
  3 * a = 3 * b * c + 3 * d :=
by sorry

end quotient_remainder_threefold_l760_76005


namespace teresa_jogged_distance_l760_76058

-- Define the conditions as Lean constants.
def teresa_speed : ℕ := 5 -- Speed in kilometers per hour
def teresa_time : ℕ := 5 -- Time in hours

-- Define the distance formula.
def teresa_distance (speed time : ℕ) : ℕ := speed * time

-- State the theorem.
theorem teresa_jogged_distance : teresa_distance teresa_speed teresa_time = 25 := by
  -- Proof is skipped using 'sorry'.
  sorry

end teresa_jogged_distance_l760_76058


namespace alec_correct_problems_l760_76003

-- Definitions of conditions and proof problem
theorem alec_correct_problems (c w : ℕ) (s : ℕ) (H1 : s = 30 + 4 * c - w) (H2 : s > 90)
  (H3 : ∀ s', 90 < s' ∧ s' < s → ¬(∃ c', ∃ w', s' = 30 + 4 * c' - w')) :
  c = 16 :=
by
  sorry

end alec_correct_problems_l760_76003


namespace johns_grandpa_money_l760_76032

theorem johns_grandpa_money :
  ∃ G : ℝ, (G + 3 * G = 120) ∧ (G = 30) := 
by
  sorry

end johns_grandpa_money_l760_76032


namespace class_mean_l760_76060

theorem class_mean
  (num_students_1 : ℕ)
  (num_students_2 : ℕ)
  (total_students : ℕ)
  (mean_score_1 : ℚ)
  (mean_score_2 : ℚ)
  (new_mean_score : ℚ)
  (h1 : num_students_1 + num_students_2 = total_students)
  (h2 : total_students = 30)
  (h3 : num_students_1 = 24)
  (h4 : mean_score_1 = 80)
  (h5 : num_students_2 = 6)
  (h6 : mean_score_2 = 85) :
  new_mean_score = 81 :=
by
  sorry

end class_mean_l760_76060


namespace complex_division_l760_76091

theorem complex_division (i : ℂ) (hi : i = Complex.I) : (2 / (1 + i)) = (1 - i) :=
by
  sorry

end complex_division_l760_76091


namespace earnings_per_puppy_l760_76075

def daily_pay : ℝ := 40
def total_earnings : ℝ := 76
def num_puppies : ℕ := 16

theorem earnings_per_puppy : (total_earnings - daily_pay) / num_puppies = 2.25 := by
  sorry

end earnings_per_puppy_l760_76075


namespace seq_inv_an_is_arithmetic_seq_fn_over_an_has_minimum_l760_76050

-- Problem 1
theorem seq_inv_an_is_arithmetic (a : ℕ → ℝ) (h1 : a 1 = 1/2) (h2 : ∀ n, n ≥ 2 → a (n - 1) / a n = (a (n - 1) + 2) / (2 - a n)) :
  ∃ d, ∀ n, n ≥ 2 → (1 / a n) = 2 + (n - 1) * d :=
sorry

-- Problem 2
theorem seq_fn_over_an_has_minimum (a f : ℕ → ℝ) (h1 : a 1 = 1/2) (h2 : ∀ n, n ≥ 2 → a (n - 1) / a n = (a (n - 1) + 2) / (2 - a n)) (h3 : ∀ n, f n = (9 / 10) ^ n) :
  ∃ m, ∀ n, n ≠ m → f n / a n ≥ f m / a m :=
sorry

end seq_inv_an_is_arithmetic_seq_fn_over_an_has_minimum_l760_76050


namespace loss_is_selling_price_of_16_pencils_l760_76067

theorem loss_is_selling_price_of_16_pencils
  (S : ℝ) -- Assume the selling price of one pencil is S
  (C : ℝ) -- Assume the cost price of one pencil is C
  (h₁ : 80 * C = 1.2 * 80 * S) -- The cost of 80 pencils is 1.2 times the selling price of 80 pencils
  : (80 * C - 80 * S) = 16 * S := -- The loss for selling 80 pencils equals the selling price of 16 pencils
  sorry

end loss_is_selling_price_of_16_pencils_l760_76067


namespace unique_bijective_function_l760_76074

noncomputable def find_bijective_function {n : ℕ}
  (hn : n ≥ 3) (hodd : n % 2 = 1)
  (x : Fin n → ℝ)
  (f : Fin n → ℝ) : Prop :=
∀ i : Fin n, f i = x i

theorem unique_bijective_function (n : ℕ) (hn : n ≥ 3) (hodd : n % 2 = 1)
  (x : Fin n → ℝ) (f : Fin n → ℝ)
  (hf_bij : Function.Bijective f)
  (h_abs_diff : ∀ i, |f i - x i| = 0) : find_bijective_function hn hodd x f :=
by
  sorry

end unique_bijective_function_l760_76074


namespace total_selling_price_correct_l760_76069

def cost_price_1 := 750
def cost_price_2 := 1200
def cost_price_3 := 500

def loss_percent_1 := 10
def loss_percent_2 := 15
def loss_percent_3 := 5

noncomputable def selling_price_1 := cost_price_1 - ((loss_percent_1 / 100) * cost_price_1)
noncomputable def selling_price_2 := cost_price_2 - ((loss_percent_2 / 100) * cost_price_2)
noncomputable def selling_price_3 := cost_price_3 - ((loss_percent_3 / 100) * cost_price_3)

noncomputable def total_selling_price := selling_price_1 + selling_price_2 + selling_price_3

theorem total_selling_price_correct : total_selling_price = 2170 := by
  sorry

end total_selling_price_correct_l760_76069


namespace option_d_correct_l760_76016

variable (a b m n : ℝ)

theorem option_d_correct :
  6 * a + a ≠ 6 * a ^ 2 ∧
  -2 * a + 5 * b ≠ 3 * a * b ∧
  4 * m ^ 2 * n - 2 * m * n ^ 2 ≠ 2 * m * n ∧
  3 * a * b ^ 2 - 5 * b ^ 2 * a = -2 * a * b ^ 2 := by
  sorry

end option_d_correct_l760_76016


namespace problem_l760_76093

def f (x : ℚ) : ℚ :=
  x⁻¹ - (x⁻¹ / (1 - x⁻¹))

theorem problem : f (f (-3)) = 6 / 5 :=
by
  sorry

end problem_l760_76093


namespace probability_of_both_selected_l760_76007

noncomputable def ramSelectionProbability : ℚ := 1 / 7
noncomputable def raviSelectionProbability : ℚ := 1 / 5

theorem probability_of_both_selected : 
  ramSelectionProbability * raviSelectionProbability = 1 / 35 :=
by sorry

end probability_of_both_selected_l760_76007


namespace correct_statements_B_and_C_l760_76054

-- Given real numbers a, b, c satisfying the conditions
variables (a b c : ℝ)
variables (h1 : a > b)
variables (h2 : b > c)
variables (h3 : a + b + c = 0)

theorem correct_statements_B_and_C : (a - c > 2 * b) ∧ (a ^ 2 > b ^ 2) :=
by
  sorry

end correct_statements_B_and_C_l760_76054


namespace b_plus_c_eq_neg3_l760_76079

theorem b_plus_c_eq_neg3 (b c : ℝ)
  (h1 : ∀ x : ℝ, x^2 + b * x + c > 0 ↔ (x < -1 ∨ x > 2)) :
  b + c = -3 :=
sorry

end b_plus_c_eq_neg3_l760_76079


namespace cars_parked_l760_76086

def front_parking_spaces : ℕ := 52
def back_parking_spaces : ℕ := 38
def filled_back_spaces : ℕ := back_parking_spaces / 2
def available_spaces : ℕ := 32
def total_parking_spaces : ℕ := front_parking_spaces + back_parking_spaces
def filled_spaces : ℕ := total_parking_spaces - available_spaces

theorem cars_parked : 
  filled_spaces = 58 := by
  sorry

end cars_parked_l760_76086


namespace investment_double_l760_76045

theorem investment_double (A : ℝ) (r t : ℝ) (hA : 0 < A) (hr : 0 < r) :
  2 * A ≤ A * (1 + r)^t ↔ t ≥ (Real.log 2) / (Real.log (1 + r)) := 
by
  sorry

end investment_double_l760_76045


namespace passengers_on_ship_l760_76080

theorem passengers_on_ship (P : ℕ)
  (h1 : P / 12 + P / 4 + P / 9 + P / 6 + 42 = P) :
  P = 108 := 
by sorry

end passengers_on_ship_l760_76080


namespace workout_days_l760_76034

theorem workout_days (n : ℕ) (squats : ℕ → ℕ) 
  (h1 : squats 1 = 30)
  (h2 : ∀ k, squats (k + 1) = squats k + 5)
  (h3 : squats 4 = 45) :
  n = 4 :=
sorry

end workout_days_l760_76034


namespace MrsBrownCarrotYield_l760_76063

theorem MrsBrownCarrotYield :
  let pacesLength := 25
  let pacesWidth := 30
  let strideLength := 2.5
  let yieldPerSquareFoot := 0.5
  let lengthInFeet := pacesLength * strideLength
  let widthInFeet := pacesWidth * strideLength
  let area := lengthInFeet * widthInFeet
  let yield := area * yieldPerSquareFoot
  yield = 2343.75 :=
by
  sorry

end MrsBrownCarrotYield_l760_76063


namespace friend_gain_is_20_percent_l760_76001

noncomputable def original_cost : ℝ := 52325.58
noncomputable def loss_percentage : ℝ := 0.14
noncomputable def friend_selling_price : ℝ := 54000
noncomputable def friend_percentage_gain : ℝ :=
  ((friend_selling_price - (original_cost * (1 - loss_percentage))) / (original_cost * (1 - loss_percentage))) * 100

theorem friend_gain_is_20_percent :
  friend_percentage_gain = 20 := by
  sorry

end friend_gain_is_20_percent_l760_76001


namespace lcm_36_65_l760_76013

-- Definitions based on conditions
def number1 : ℕ := 36
def number2 : ℕ := 65

-- The prime factorization conditions can be implied through deriving LCM hence added as comments to clarify the conditions.
-- 36 = 2^2 * 3^2
-- 65 = 5 * 13

-- Theorem statement that the LCM of number1 and number2 is 2340
theorem lcm_36_65 : Nat.lcm number1 number2 = 2340 := 
by 
  sorry

end lcm_36_65_l760_76013


namespace necessary_but_not_sufficient_l760_76085

-- Define the function f(x)
def f (a x : ℝ) := |a - 3 * x|

-- Define the condition for the function to be monotonically increasing on [1, +∞)
def is_monotonically_increasing_on_interval (a : ℝ) : Prop :=
  ∀ (x y : ℝ), 1 ≤ x → x ≤ y → (f a x ≤ f a y)

-- Define the condition that a must be 3
def condition_a_eq_3 (a : ℝ) : Prop := (a = 3)

-- Prove that condition_a_eq_3 is a necessary but not sufficient condition
theorem necessary_but_not_sufficient (a : ℝ) :
  (is_monotonically_increasing_on_interval a) →
  condition_a_eq_3 a ↔ (∀ (b : ℝ), b ≠ a → is_monotonically_increasing_on_interval b → false) := 
sorry

end necessary_but_not_sufficient_l760_76085


namespace max_value_S_n_S_m_l760_76011

noncomputable def a (n : ℕ) : ℤ := -(n : ℤ)^2 + 12 * n - 32

noncomputable def S : ℕ → ℤ
| 0       => 0
| (n + 1) => S n + a (n + 1)

theorem max_value_S_n_S_m : ∀ m n : ℕ, m < n → m > 0 → S n - S m ≤ 10 :=
by
  sorry

end max_value_S_n_S_m_l760_76011


namespace find_number_l760_76051

theorem find_number (x : ℝ) (h : 97 * x - 89 * x = 4926) : x = 615.75 :=
by
  sorry

end find_number_l760_76051


namespace value_of_r_minus_q_l760_76014

variable (q r : ℝ)
variable (slope : ℝ)
variable (h_parallel : slope = 3 / 2)
variable (h_points : (r - q) / (-2) = slope)

theorem value_of_r_minus_q (h_parallel : slope = 3 / 2) (h_points : (r - q) / (-2) = slope) : 
  r - q = -3 := by
  sorry

end value_of_r_minus_q_l760_76014


namespace gcd_seven_eight_fact_l760_76030

-- Definitions based on the problem conditions
def seven_fact : ℕ := 1 * 2 * 3 * 4 * 5 * 6 * 7
def eight_fact : ℕ := 8 * seven_fact

-- Statement of the theorem
theorem gcd_seven_eight_fact : Nat.gcd seven_fact eight_fact = seven_fact := by
  sorry

end gcd_seven_eight_fact_l760_76030


namespace sum_primes_reversed_l760_76068

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def reverse_digits (n : ℕ) : ℕ := 
  let tens := n / 10
  let ones := n % 10
  10 * ones + tens

def valid_primes : List ℕ := [31, 37, 71, 73]

theorem sum_primes_reversed :
  (∀ p ∈ valid_primes, 20 < p ∧ p < 80 ∧ is_prime p ∧ is_prime (reverse_digits p)) ∧
  (valid_primes.sum = 212) :=
by
  sorry

end sum_primes_reversed_l760_76068


namespace points_on_circle_l760_76098

theorem points_on_circle (n : ℕ) (h1 : ∃ (k : ℕ), k = (35 - 7) ∧ n = 2 * k) : n = 56 :=
sorry

end points_on_circle_l760_76098


namespace shepherds_sheep_l760_76049

theorem shepherds_sheep (x y : ℕ) 
  (h1 : x - 4 = y + 4) 
  (h2 : x + 4 = 3 * (y - 4)) : 
  x = 20 ∧ y = 12 := 
by 
  sorry

end shepherds_sheep_l760_76049


namespace total_charge_3_hours_l760_76059

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

end total_charge_3_hours_l760_76059


namespace solution_exists_l760_76015

theorem solution_exists (a b c : ℝ) : ∃ x : ℝ, (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a) = 0 := 
sorry

end solution_exists_l760_76015


namespace train_speed_in_kmph_l760_76077

noncomputable def motorbike_speed : ℝ := 64
noncomputable def overtaking_time : ℝ := 40
noncomputable def train_length_meters : ℝ := 400.032

theorem train_speed_in_kmph :
  let train_length_km := train_length_meters / 1000
  let overtaking_time_hours := overtaking_time / 3600
  let relative_speed := train_length_km / overtaking_time_hours
  let train_speed := motorbike_speed + relative_speed
  train_speed = 100.00288 := by
  sorry

end train_speed_in_kmph_l760_76077


namespace max_sum_arith_seq_l760_76066

theorem max_sum_arith_seq (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) :
  (∀ n, a n = 8 + (n - 1) * d) →
  d ≠ 0 →
  a 1 = 8 →
  a 5 ^ 2 = a 1 * a 7 →
  S n = n * a 1 + (n * (n - 1) * d) / 2 →
  ∃ n : ℕ, S n = 36 :=
by
  intros
  sorry

end max_sum_arith_seq_l760_76066


namespace percentage_deposited_l760_76076

theorem percentage_deposited (amount_deposited income : ℝ) 
  (h1 : amount_deposited = 2500) (h2 : income = 10000) : 
  (amount_deposited / income) * 100 = 25 :=
by
  have amount_deposited_val : amount_deposited = 2500 := h1
  have income_val : income = 10000 := h2
  sorry

end percentage_deposited_l760_76076


namespace percentage_vehicles_updated_2003_l760_76010

theorem percentage_vehicles_updated_2003 (a : ℝ) (h1 : 1.1^4 = 1.46) (h2 : 1.1^5 = 1.61) :
  (a * 1 / (a * 1.61) * 100 = 16.4) :=
  by sorry

end percentage_vehicles_updated_2003_l760_76010


namespace average_run_per_day_l760_76026

theorem average_run_per_day (n6 n7 n8 : ℕ) 
  (h1 : 3 * n7 = n6) 
  (h2 : 3 * n8 = n7) 
  (h3 : n6 * 20 + n7 * 18 + n8 * 16 = 250 * n8) : 
  (n6 * 20 + n7 * 18 + n8 * 16) / (n6 + n7 + n8) = 250 / 13 :=
by sorry

end average_run_per_day_l760_76026


namespace range_of_a_l760_76095

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 2 * x^2 - a * x + 1 > 0) ↔ -2 * Real.sqrt 2 < a ∧ a < 2 * Real.sqrt 2 :=
by
  sorry

end range_of_a_l760_76095


namespace general_term_a_n_general_term_b_n_sum_of_first_n_terms_D_n_l760_76048

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

end general_term_a_n_general_term_b_n_sum_of_first_n_terms_D_n_l760_76048


namespace fabric_problem_l760_76078

theorem fabric_problem
  (x y : ℝ)
  (h1 : y > 0)
  (cost_second_piece := x)
  (cost_first_piece := x + 126)
  (cost_per_meter_first := (x + 126) / y)
  (cost_per_meter_second := x / y)
  (h2 : 4 * cost_per_meter_first - 3 * cost_per_meter_second = 135)
  (h3 : 3 * cost_per_meter_first + 4 * cost_per_meter_second = 382.5) :
  y = 5.6 ∧ cost_per_meter_first = 67.5 ∧ cost_per_meter_second = 45 :=
sorry

end fabric_problem_l760_76078


namespace jeans_cost_l760_76042

-- Definitions based on conditions
def shirt_cost : ℕ := 5
def hat_cost : ℕ := 4
def total_cost : ℕ := 51
def n_shirts : ℕ := 3
def n_hats : ℕ := 4
def n_jeans : ℕ := 2

-- The goal is to prove that the cost of one pair of jeans (J) is 10
theorem jeans_cost (J : ℕ) (h : n_shirts * shirt_cost + n_jeans * J + n_hats * hat_cost = total_cost) : J = 10 :=
  sorry

end jeans_cost_l760_76042


namespace gcd_84_294_315_l760_76020

def gcd_3_integers : ℕ := Nat.gcd (Nat.gcd 84 294) 315

theorem gcd_84_294_315 : gcd_3_integers = 21 :=
by
  sorry

end gcd_84_294_315_l760_76020


namespace person_Y_share_l760_76029

theorem person_Y_share (total_amount : ℝ) (r1 r2 r3 r4 r5 : ℝ) (ratio_Y : ℝ) 
  (h1 : total_amount = 1390) 
  (h2 : r1 = 13) 
  (h3 : r2 = 17)
  (h4 : r3 = 23) 
  (h5 : r4 = 29) 
  (h6 : r5 = 37) 
  (h7 : ratio_Y = 29): 
  (total_amount / (r1 + r2 + r3 + r4 + r5) * ratio_Y) = 338.72 :=
by
  sorry

end person_Y_share_l760_76029


namespace lattice_points_on_hyperbola_l760_76083

open Real

theorem lattice_points_on_hyperbola : 
  ∃ (S : Finset (ℤ × ℤ)), 
  (∀ (x y : ℤ), ((x, y) ∈ S ↔ x^2 - y^2 = 65)) ∧ S.card = 8 :=
by
  sorry

end lattice_points_on_hyperbola_l760_76083


namespace honey_harvested_correct_l760_76024

def honey_harvested_last_year : ℕ := 2479
def honey_increase_this_year : ℕ := 6085
def honey_harvested_this_year : ℕ := 8564

theorem honey_harvested_correct :
  honey_harvested_last_year + honey_increase_this_year = honey_harvested_this_year :=
sorry

end honey_harvested_correct_l760_76024


namespace algebra_expression_value_l760_76087

theorem algebra_expression_value (a b : ℝ)
  (h1 : |a + 2| = 0)
  (h2 : (b - 5 / 2) ^ 2 = 0) : (2 * a + 3 * b) * (2 * b - 3 * a) = 26 := by
sorry

end algebra_expression_value_l760_76087


namespace total_soccer_balls_purchased_l760_76018

theorem total_soccer_balls_purchased : 
  (∃ (x : ℝ), 
    800 / x * 2 = 1560 / (x - 2)) → 
  (800 / x + 1560 / (x - 2) = 30) :=
by
  sorry

end total_soccer_balls_purchased_l760_76018


namespace count_multiples_4_or_5_not_20_l760_76071

-- We define the necessary ranges and conditions
def is_multiple_of (n k : ℕ) := n % k = 0

def count_multiples (n k : ℕ) := (n / k)

def not_multiple_of (n k : ℕ) := ¬ is_multiple_of n k

def count_multiples_excluding (n k l : ℕ) :=
  count_multiples n k + count_multiples n l - count_multiples n (Nat.lcm k l)

theorem count_multiples_4_or_5_not_20 : count_multiples_excluding 3010 4 5 = 1204 := 
by
  sorry

end count_multiples_4_or_5_not_20_l760_76071


namespace sum_of_max_min_a_l760_76009

theorem sum_of_max_min_a (a : ℝ) (x : ℝ) :
  (x^2 - a * x - 20 * a^2 < 0) →
  (∀ x1 x2 : ℝ, x1^2 - a * x1 - 20 * a^2 = 0 ∧ x2^2 - a * x2 - 20 * a^2 = 0 → |x1 - x2| ≤ 9) →
  (∀ max_min_sum : ℝ, max_min_sum = 1 + (-1) → max_min_sum = 0) := 
sorry

end sum_of_max_min_a_l760_76009


namespace first_batch_students_l760_76090

theorem first_batch_students 
  (x : ℕ) 
  (avg1 avg2 avg3 overall_avg : ℝ) 
  (n2 n3 : ℕ) 
  (h_avg1 : avg1 = 45) 
  (h_avg2 : avg2 = 55) 
  (h_avg3 : avg3 = 65) 
  (h_n2 : n2 = 50) 
  (h_n3 : n3 = 60) 
  (h_overall_avg : overall_avg = 56.333333333333336) 
  (h_eq : overall_avg = (45 * x + 55 * 50 + 65 * 60) / (x + 50 + 60)) 
  : x = 40 :=
sorry

end first_batch_students_l760_76090


namespace sheila_will_attend_picnic_l760_76088

def P_Rain : ℝ := 0.3
def P_Cloudy : ℝ := 0.4
def P_Sunny : ℝ := 0.3

def P_Attend_if_Rain : ℝ := 0.25
def P_Attend_if_Cloudy : ℝ := 0.5
def P_Attend_if_Sunny : ℝ := 0.75

def P_Attend : ℝ :=
  P_Rain * P_Attend_if_Rain +
  P_Cloudy * P_Attend_if_Cloudy +
  P_Sunny * P_Attend_if_Sunny

theorem sheila_will_attend_picnic : P_Attend = 0.5 := by
  sorry

end sheila_will_attend_picnic_l760_76088


namespace total_weekly_water_consumption_l760_76092

-- Definitions coming from the conditions of the problem
def num_cows : Nat := 40
def water_per_cow_per_day : Nat := 80
def num_sheep : Nat := 10 * num_cows
def water_per_sheep_per_day : Nat := water_per_cow_per_day / 4
def days_in_week : Nat := 7

-- To prove statement: 
theorem total_weekly_water_consumption :
  let weekly_water_cow := water_per_cow_per_day * days_in_week
  let total_weekly_water_cows := weekly_water_cow * num_cows
  let daily_water_sheep := water_per_sheep_per_day
  let weekly_water_sheep := daily_water_sheep * days_in_week
  let total_weekly_water_sheep := weekly_water_sheep * num_sheep
  total_weekly_water_cows + total_weekly_water_sheep = 78400 := 
by
  sorry

end total_weekly_water_consumption_l760_76092


namespace counts_of_arson_l760_76057

-- Define variables A (arson), B (burglary), L (petty larceny)
variables (A B L : ℕ)

-- Conditions given in the problem
def burglary_charges : Prop := B = 2
def petty_larceny_charges_relation : Prop := L = 6 * B
def total_sentence_calculation : Prop := 36 * A + 18 * B + 6 * L = 216

-- Prove that given these conditions, the counts of arson (A) is 3
theorem counts_of_arson (h1 : burglary_charges B)
                        (h2 : petty_larceny_charges_relation B L)
                        (h3 : total_sentence_calculation A B L) :
                        A = 3 :=
sorry

end counts_of_arson_l760_76057


namespace carolyn_initial_marbles_l760_76025

theorem carolyn_initial_marbles (x : ℕ) (h1 : x - 42 = 5) : x = 47 :=
by
  sorry

end carolyn_initial_marbles_l760_76025


namespace cupcake_price_l760_76008

theorem cupcake_price
  (x : ℝ)
  (h1 : 5 * x + 6 * 1 + 4 * 2 + 15 * 0.6 = 33) : x = 2 :=
by
  sorry

end cupcake_price_l760_76008


namespace maddox_more_profit_than_theo_l760_76002

-- Definitions (conditions)
def cost_per_camera : ℕ := 20
def num_cameras : ℕ := 3
def total_cost : ℕ := num_cameras * cost_per_camera

def maddox_selling_price_per_camera : ℕ := 28
def theo_selling_price_per_camera : ℕ := 23

-- Total selling price
def maddox_total_selling_price : ℕ := num_cameras * maddox_selling_price_per_camera
def theo_total_selling_price : ℕ := num_cameras * theo_selling_price_per_camera

-- Profits
def maddox_profit : ℕ := maddox_total_selling_price - total_cost
def theo_profit : ℕ := theo_total_selling_price - total_cost

-- Proof Statement
theorem maddox_more_profit_than_theo : maddox_profit - theo_profit = 15 := by
  sorry

end maddox_more_profit_than_theo_l760_76002
