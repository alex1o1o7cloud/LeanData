import Mathlib

namespace problem_ineq_l1941_194149

variable {a b c : ℝ}

theorem problem_ineq 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2) ≥ 9 * a^2 * b^2 * c^2 := 
sorry

end problem_ineq_l1941_194149


namespace total_original_cost_of_books_l1941_194174

noncomputable def original_cost_price_in_eur (selling_prices : List ℝ) (profit_margin : ℝ) (exchange_rate : ℝ) : ℝ :=
  let original_cost_prices := selling_prices.map (λ price => price / (1 + profit_margin))
  let total_original_cost_usd := original_cost_prices.sum
  total_original_cost_usd * exchange_rate

theorem total_original_cost_of_books : original_cost_price_in_eur [240, 260, 280, 300, 320] 0.20 0.85 = 991.67 :=
  sorry

end total_original_cost_of_books_l1941_194174


namespace ben_and_sara_tie_fraction_l1941_194172

theorem ben_and_sara_tie_fraction (ben_wins sara_wins : ℚ) (h1 : ben_wins = 2 / 5) (h2 : sara_wins = 1 / 4) : 
  1 - (ben_wins + sara_wins) = 7 / 20 :=
by
  rw [h1, h2]
  norm_num

end ben_and_sara_tie_fraction_l1941_194172


namespace solve_inequality_l1941_194151

theorem solve_inequality (x : ℝ) : (x^2 + 7 * x < 8) ↔ x ∈ (Set.Ioo (-8 : ℝ) 1) := by
  sorry

end solve_inequality_l1941_194151


namespace find_f_2000_l1941_194144

variable (f : ℕ → ℕ)
variable (x : ℕ)

axiom initial_condition : f 0 = 1
axiom recurrence_relation : ∀ x, f (x + 2) = f x + 4 * x + 2

theorem find_f_2000 : f 2000 = 3998001 :=
by
  sorry

end find_f_2000_l1941_194144


namespace maximum_price_for_360_skewers_price_for_1920_profit_l1941_194109

-- Define the number of skewers sold as a function of the price
def skewers_sold (price : ℝ) : ℝ := 300 + 60 * (10 - price)

-- Define the profit as a function of the price
def profit (price : ℝ) : ℝ := (skewers_sold price) * (price - 3)

-- Maximum price for selling at least 360 skewers per day
theorem maximum_price_for_360_skewers (price : ℝ) (h : skewers_sold price ≥ 360) : price ≤ 9 :=
by {
    sorry
}

-- Price to achieve a profit of 1920 yuan per day with price constraint
theorem price_for_1920_profit (price : ℝ) (h₁ : profit price = 1920) (h₂ : price ≤ 8) : price = 7 :=
by {
    sorry
}

end maximum_price_for_360_skewers_price_for_1920_profit_l1941_194109


namespace math_proof_problem_l1941_194104

noncomputable def sum_of_distinct_squares (a b c : ℕ) : ℕ :=
3 * ((a^2 + b^2 + c^2 : ℕ))

theorem math_proof_problem (a b c : ℕ)
  (h1 : a + b + c = 27)
  (h2 : Nat.gcd a b + Nat.gcd b c + Nat.gcd c a = 11) :
  sum_of_distinct_squares a b c = 2274 :=
sorry

end math_proof_problem_l1941_194104


namespace a_investment_l1941_194116

theorem a_investment (B C total_profit A_share: ℝ) (hB: B = 7200) (hC: C = 9600) (htotal_profit: total_profit = 9000) 
  (hA_share: A_share = 1125) : ∃ x : ℝ, (A_share / total_profit) = (x / (x + B + C)) ∧ x = 2400 := 
by
  use 2400
  sorry

end a_investment_l1941_194116


namespace range_of_m_l1941_194106

noncomputable def G (x m : ℝ) : ℝ := (8 * x^2 + 24 * x + 5 * m) / 8

theorem range_of_m (G_is_square : ∃ c d, ∀ x, G x m = (c * x + d) ^ 2) : 3 < m ∧ m < 4 :=
by
  sorry

end range_of_m_l1941_194106


namespace solve_for_s_l1941_194112

theorem solve_for_s (s t : ℚ) (h1 : 7 * s + 8 * t = 150) (h2 : s = 2 * t + 3) : s = 162 / 11 := 
by
  sorry

end solve_for_s_l1941_194112


namespace problem_l1941_194182

theorem problem (a b : ℝ) (h1 : abs a = 4) (h2 : b^2 = 9) (h3 : a / b > 0) : a - b = 1 ∨ a - b = -1 := 
sorry

end problem_l1941_194182


namespace max_value_of_expression_l1941_194183

theorem max_value_of_expression (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 2 * a + b = 1) : 
  2 * Real.sqrt (a * b) - 4 * a ^ 2 - b ^ 2 ≤ (Real.sqrt 2 - 1) / 2 :=
sorry

end max_value_of_expression_l1941_194183


namespace tyler_eggs_in_fridge_l1941_194110

def recipe_eggs_for_four : Nat := 2
def people_multiplier : Nat := 2
def eggs_needed : Nat := recipe_eggs_for_four * people_multiplier
def eggs_to_buy : Nat := 1
def eggs_in_fridge : Nat := eggs_needed - eggs_to_buy

theorem tyler_eggs_in_fridge : eggs_in_fridge = 3 := by
  sorry

end tyler_eggs_in_fridge_l1941_194110


namespace jack_paycheck_l1941_194187

theorem jack_paycheck (P : ℝ) (h1 : 0.15 * 150 + 0.25 * (P - 150) + 30 + 70 / 100 * (P - (0.15 * 150 + 0.25 * (P - 150) + 30)) * 30 / 100 = 50) : P = 242.22 :=
sorry

end jack_paycheck_l1941_194187


namespace gcd_390_455_546_l1941_194161

theorem gcd_390_455_546 :
  Nat.gcd (Nat.gcd 390 455) 546 = 13 := 
sorry

end gcd_390_455_546_l1941_194161


namespace muffin_banana_ratio_l1941_194162

variable {R : Type} [LinearOrderedField R]

-- Define the costs of muffins and bananas
variables {m b : R}

-- Susie's cost
def susie_cost (m b : R) := 4 * m + 5 * b

-- Calvin's cost for three times Susie's items
def calvin_cost_tripled (m b : R) := 12 * m + 15 * b

-- Calvin's actual cost
def calvin_cost_actual (m b : R) := 2 * m + 12 * b

theorem muffin_banana_ratio (m b : R) (h : calvin_cost_tripled m b = calvin_cost_actual m b) : m = (3 / 10) * b :=
by sorry

end muffin_banana_ratio_l1941_194162


namespace profit_percentage_l1941_194122

theorem profit_percentage (SP CP : ℤ) (h_SP : SP = 1170) (h_CP : CP = 975) :
  ((SP - CP : ℤ) * 100) / CP = 20 :=
by 
  sorry

end profit_percentage_l1941_194122


namespace meals_given_away_l1941_194131

def initial_meals_colt_and_curt : ℕ := 113
def additional_meals_sole_mart : ℕ := 50
def remaining_meals : ℕ := 78
def total_initial_meals : ℕ := initial_meals_colt_and_curt + additional_meals_sole_mart
def given_away_meals (total : ℕ) (remaining : ℕ) : ℕ := total - remaining

theorem meals_given_away : given_away_meals total_initial_meals remaining_meals = 85 :=
by
  sorry

end meals_given_away_l1941_194131


namespace option_A_incorrect_option_B_incorrect_option_C_incorrect_option_D_correct_l1941_194196

theorem option_A_incorrect : ¬(Real.sqrt 2 + Real.sqrt 6 = Real.sqrt 8) :=
by sorry

theorem option_B_incorrect : ¬(6 * Real.sqrt 3 - 2 * Real.sqrt 3 = 4) :=
by sorry

theorem option_C_incorrect : ¬(4 * Real.sqrt 2 * 2 * Real.sqrt 3 = 6 * Real.sqrt 6) :=
by sorry

theorem option_D_correct : (1 / (2 - Real.sqrt 3) = 2 + Real.sqrt 3) :=
by sorry

end option_A_incorrect_option_B_incorrect_option_C_incorrect_option_D_correct_l1941_194196


namespace range_f_x1_x2_l1941_194173

noncomputable def f (c x : ℝ) : ℝ := 2 * x ^ 3 - 3 * x ^ 2 + c * x + 1

theorem range_f_x1_x2 (c x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : x1 < x2) 
  (h4 : 36 - 24 * c > 0) (h5 : ∀ x, f c x = 2 * x ^ 3 - 3 * x ^ 2 + c * x + 1) :
  1 < f c x1 / x2 ∧ f c x1 / x2 < 5 / 2 :=
sorry

end range_f_x1_x2_l1941_194173


namespace molecular_weight_N2O3_l1941_194164

variable (atomic_weight_N : ℝ) (atomic_weight_O : ℝ)
variable (n_N_atoms : ℝ) (n_O_atoms : ℝ)
variable (expected_molecular_weight : ℝ)

theorem molecular_weight_N2O3 :
  atomic_weight_N = 14.01 →
  atomic_weight_O = 16.00 →
  n_N_atoms = 2 →
  n_O_atoms = 3 →
  expected_molecular_weight = 76.02 →
  (n_N_atoms * atomic_weight_N + n_O_atoms * atomic_weight_O = expected_molecular_weight) :=
by
  intros
  sorry

end molecular_weight_N2O3_l1941_194164


namespace solve_ode_l1941_194179

noncomputable def x (t : ℝ) : ℝ :=
  -((1 : ℝ) / 18) * Real.exp (-t) +
  (25 / 54) * Real.exp (5 * t) -
  (11 / 27) * Real.exp (-4 * t)

theorem solve_ode :
  ∀ t : ℝ, 
    (deriv^[2] x t) - (deriv x t) - 20 * x t = Real.exp (-t) ∧
    x 0 = 0 ∧
    (deriv x 0) = 4 :=
by
  sorry

end solve_ode_l1941_194179


namespace incorrect_statement_D_l1941_194111

theorem incorrect_statement_D (a b r : ℝ) (hr : r > 0) :
  ¬ ∀ b < r, ∃ x, (x - a)^2 + (0 - b)^2 = r^2 :=
by 
  sorry

end incorrect_statement_D_l1941_194111


namespace inequality_holds_l1941_194133

theorem inequality_holds (a b : ℝ) : (6 * a - 3 * b - 3) * (a ^ 2 + a ^ 2 * b - 2 * a ^ 3) ≤ 0 :=
sorry

end inequality_holds_l1941_194133


namespace complex_coordinate_l1941_194163

theorem complex_coordinate (i : ℂ) (h : i * i = -1) : i * (1 - i) = 1 + i :=
by sorry

end complex_coordinate_l1941_194163


namespace solve_x_squared_eq_sixteen_l1941_194159

theorem solve_x_squared_eq_sixteen : ∃ (x1 x2 : ℝ), (x1 = -4 ∧ x2 = 4) ∧ ∀ x : ℝ, x^2 = 16 → (x = x1 ∨ x = x2) :=
by
  sorry

end solve_x_squared_eq_sixteen_l1941_194159


namespace count_arithmetic_sequence_terms_l1941_194105

theorem count_arithmetic_sequence_terms : 
  ∃ n : ℕ, 
  (∀ k : ℕ, k ≥ 1 → 6 + (k - 1) * 4 = 202 → n = k) ∧ n = 50 :=
by
  sorry

end count_arithmetic_sequence_terms_l1941_194105


namespace leonardo_initial_money_l1941_194157

theorem leonardo_initial_money (chocolate_cost : ℝ) (borrowed_amount : ℝ) (needed_amount : ℝ)
  (h_chocolate_cost : chocolate_cost = 5)
  (h_borrowed_amount : borrowed_amount = 0.59)
  (h_needed_amount : needed_amount = 0.41) :
  chocolate_cost + borrowed_amount + needed_amount - (chocolate_cost - borrowed_amount) = 4.41 :=
by
  rw [h_chocolate_cost, h_borrowed_amount, h_needed_amount]
  norm_num
  -- Continue with the proof, eventually obtaining the value 4.41
  sorry

end leonardo_initial_money_l1941_194157


namespace n_times_s_eq_2023_l1941_194120

noncomputable def S := { x : ℝ | x > 0 }

-- Function f: S → ℝ
def f (x : ℝ) : ℝ := sorry

-- Condition: f(x) f(y) = f(xy) + 2023 * (2/x + 2/y + 2022) for all x, y > 0
axiom f_property (x y : ℝ) (hx : x > 0) (hy : y > 0) : f x * f y = f (x * y) + 2023 * (2 / x + 2 / y + 2022)

-- Theorem: Prove n × s = 2023 where n is the number of possible values of f(2) and s is the sum of all possible values of f(2)
theorem n_times_s_eq_2023 (n s : ℕ) : n * s = 2023 :=
sorry

end n_times_s_eq_2023_l1941_194120


namespace marked_price_each_article_l1941_194192

noncomputable def pair_price : ℝ := 50
noncomputable def discount : ℝ := 0.60
noncomputable def marked_price_pair : ℝ := 50 / 0.40
noncomputable def marked_price_each : ℝ := marked_price_pair / 2

theorem marked_price_each_article : 
  marked_price_each = 62.50 := by
  sorry

end marked_price_each_article_l1941_194192


namespace number_of_groups_l1941_194124

-- Define constants
def new_players : Nat := 48
def returning_players : Nat := 6
def players_per_group : Nat := 6

-- Define the theorem to be proven
theorem number_of_groups :
  (new_players + returning_players) / players_per_group = 9 :=
by
  sorry

end number_of_groups_l1941_194124


namespace female_democrats_count_l1941_194158

variable (F M : ℕ)
def total_participants : Prop := F + M = 720
def female_democrats (D_F : ℕ) : Prop := D_F = 1 / 2 * F
def male_democrats (D_M : ℕ) : Prop := D_M = 1 / 4 * M
def total_democrats (D_F D_M : ℕ) : Prop := D_F + D_M = 1 / 3 * 720

theorem female_democrats_count
  (F M D_F D_M : ℕ)
  (h1 : total_participants F M)
  (h2 : female_democrats F D_F)
  (h3 : male_democrats M D_M)
  (h4 : total_democrats D_F D_M) :
  D_F = 120 :=
sorry

end female_democrats_count_l1941_194158


namespace equilibrium_force_l1941_194146

def f1 : ℝ × ℝ := (-2, -1)
def f2 : ℝ × ℝ := (-3, 2)
def f3 : ℝ × ℝ := (4, -3)
def expected_f4 : ℝ × ℝ := (1, 2)

theorem equilibrium_force :
  (1, 2) = -(f1 + f2 + f3) := 
by
  sorry

end equilibrium_force_l1941_194146


namespace average_length_of_two_strings_l1941_194184

theorem average_length_of_two_strings (a b : ℝ) (h1 : a = 3.2) (h2 : b = 4.8) :
  (a + b) / 2 = 4.0 :=
by
  sorry

end average_length_of_two_strings_l1941_194184


namespace gcd_problem_l1941_194189

def gcd3 (x y z : ℕ) : ℕ := Int.gcd x (Int.gcd y z)

theorem gcd_problem (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : gcd3 (a^2 - 1) (b^2 - 1) (c^2 - 1) = 1) :
  gcd3 (a * b + c) (b * c + a) (c * a + b) = gcd3 a b c :=
by
  sorry

end gcd_problem_l1941_194189


namespace parking_lot_length_l1941_194137

theorem parking_lot_length (W : ℝ) (U : ℝ) (A_car : ℝ) (N_cars : ℕ) (H_w : W = 400) (H_u : U = 0.80) (H_Acar : A_car = 10) (H_Ncars : N_cars = 16000) :
  (U * (W * L) = N_cars * A_car) → (L = 500) :=
by
  sorry

end parking_lot_length_l1941_194137


namespace at_least_one_admitted_prob_l1941_194185

theorem at_least_one_admitted_prob (pA pB : ℝ) (hA : pA = 0.6) (hB : pB = 0.7) (independent : ∀ (P Q : Prop), P ∧ Q → P ∧ Q):
  (1 - ((1 - pA) * (1 - pB))) = 0.88 :=
by
  rw [hA, hB]
  -- more steps would follow in a complete proof
  sorry

end at_least_one_admitted_prob_l1941_194185


namespace final_result_always_4_l1941_194190

-- The function that performs the operations described in the problem
def transform (x : Nat) : Nat :=
  let step1 := 2 * x
  let step2 := step1 + 3
  let step3 := step2 * 5
  let step4 := step3 + 7
  let last_digit := step4 % 10
  let step6 := last_digit + 18
  step6 / 5

-- The theorem statement claiming that for any single-digit number x, the result of transform x is always 4
theorem final_result_always_4 (x : Nat) (h : x < 10) : transform x = 4 := by
  sorry

end final_result_always_4_l1941_194190


namespace inequality_sqrt_l1941_194150

open Real

theorem inequality_sqrt (x y : ℝ) :
  (sqrt (x^2 - 2*x*y) > sqrt (1 - y^2)) ↔ 
    ((x - y > 1 ∧ -1 < y ∧ y < 1) ∨ (x - y < -1 ∧ -1 < y ∧ y < 1)) :=
by
  sorry

end inequality_sqrt_l1941_194150


namespace complex_quadrant_l1941_194165

theorem complex_quadrant (x y: ℝ) (h : x = 1 ∧ y = 2) : x > 0 ∧ y > 0 :=
by
  sorry

end complex_quadrant_l1941_194165


namespace g_f_neg3_l1941_194178

def f (x : ℤ) : ℤ := x^3 - 1
def g (x : ℤ) : ℤ := 3 * x^2 + 3 * x + 1

theorem g_f_neg3 : g (f (-3)) = 2285 :=
by
  -- provide the proof here
  sorry

end g_f_neg3_l1941_194178


namespace batsman_average_after_17th_inning_l1941_194129

theorem batsman_average_after_17th_inning :
  ∃ x : ℤ, (63 + (16 * x) = 17 * (x + 3)) ∧ (x + 3 = 17) :=
by
  sorry

end batsman_average_after_17th_inning_l1941_194129


namespace quadratic_inequality_solution_range_l1941_194148

theorem quadratic_inequality_solution_range (m : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 + m * x + 2 > 0) ↔ m > -3 := 
sorry

end quadratic_inequality_solution_range_l1941_194148


namespace ratio_of_lengths_l1941_194107

theorem ratio_of_lengths (total_length short_length : ℕ)
  (h1 : total_length = 35)
  (h2 : short_length = 10) :
  short_length / (total_length - short_length) = 2 / 5 := by
  -- Proof skipped
  sorry

end ratio_of_lengths_l1941_194107


namespace line_direction_vector_correct_l1941_194145

theorem line_direction_vector_correct :
  ∃ (A B C : ℝ), (A = 2 ∧ B = -3 ∧ C = 1) ∧ 
  ∃ (v w : ℝ), (v = A ∧ w = B) :=
by
  sorry

end line_direction_vector_correct_l1941_194145


namespace calculate_f_one_l1941_194193

def f (x : ℝ) : ℝ := x^2 + |x - 2|

theorem calculate_f_one : f 1 = 2 := by
  sorry

end calculate_f_one_l1941_194193


namespace room_perimeter_l1941_194160

theorem room_perimeter (b l : ℝ) (h1 : l = 3 * b) (h2 : l * b = 12) : 2 * (l + b) = 16 :=
by sorry

end room_perimeter_l1941_194160


namespace sum_of_coordinates_l1941_194175

noncomputable def endpoint_x (x : ℤ) := (-3 + x) / 2 = 2
noncomputable def endpoint_y (y : ℤ) := (-15 + y) / 2 = -5

theorem sum_of_coordinates : ∃ x y : ℤ, endpoint_x x ∧ endpoint_y y ∧ x + y = 12 :=
by
  sorry

end sum_of_coordinates_l1941_194175


namespace find_ABC_l1941_194127

-- Define the angles as real numbers in degrees
variables (ABC CBD DBC DBE ABE : ℝ)

-- Assert the given conditions
axiom horz_angle: CBD = 90
axiom DBC_ABC_relation : DBC = ABC + 30
axiom straight_angle: DBE = 180
axiom measure_abe: ABE = 145

-- State the proof problem
theorem find_ABC : ABC = 30 :=
by
  -- Include all steps required to derive the conclusion in the proof
  sorry

end find_ABC_l1941_194127


namespace complex_number_solution_l1941_194154

theorem complex_number_solution (z : ℂ) (i : ℂ) (h1 : i * z = (1 - 2 * i) ^ 2) (h2 : i * i = -1) : z = -4 + 3 * i := by
  sorry

end complex_number_solution_l1941_194154


namespace find_point_C_on_z_axis_l1941_194142

noncomputable def point_c_condition (C : ℝ × ℝ × ℝ) (A B : ℝ × ℝ × ℝ) : Prop :=
  dist C A = dist C B

theorem find_point_C_on_z_axis :
  ∃ C : ℝ × ℝ × ℝ, C = (0, 0, 1) ∧ point_c_condition C (1, 0, 2) (1, 1, 1) :=
by
  use (0, 0, 1)
  simp [point_c_condition]
  sorry

end find_point_C_on_z_axis_l1941_194142


namespace value_at_2007_l1941_194100

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f x = f (-x)
axiom symmetric_property (x : ℝ) : f (2 + x) = f (2 - x)
axiom specific_value : f (-3) = -2

theorem value_at_2007 : f 2007 = -2 :=
sorry

end value_at_2007_l1941_194100


namespace magic_square_l1941_194188

-- Define a 3x3 grid with positions a, b, c and unknowns x, y, z, t, u, v
variables (a b c x y z t u v : ℝ)

-- State the theorem: there exists values for x, y, z, t, u, v
-- such that the sums in each row, column, and both diagonals are the same
theorem magic_square (h1: x = (b + 3*c - 2*a) / 2)
  (h2: y = a + b - c)
  (h3: z = (b + c) / 2)
  (h4: t = 2*c - a)
  (h5: u = b + c - a)
  (h6: v = (2*a + b - c) / 2) :
  x + a + b = y + z + t ∧
  y + z + t = u ∧
  z + t + u = b + z + c ∧
  t + u + v = a + u + c ∧
  x + t + v = u + y + c ∧
  by sorry :=
sorry

end magic_square_l1941_194188


namespace cost_per_meter_l1941_194136

-- Definitions of the conditions
def length_of_plot : ℕ := 63
def breadth_of_plot : ℕ := length_of_plot - 26
def perimeter_of_plot := 2 * length_of_plot + 2 * breadth_of_plot
def total_cost : ℕ := 5300

-- Statement to prove
theorem cost_per_meter : (total_cost : ℚ) / perimeter_of_plot = 26.5 :=
by sorry

end cost_per_meter_l1941_194136


namespace ratio_expression_l1941_194128

-- Given conditions: X : Y : Z = 3 : 2 : 6
def ratio (X Y Z : ℚ) : Prop := X / Y = 3 / 2 ∧ Y / Z = 2 / 6

-- The expression to be evaluated
def expr (X Y Z : ℚ) : ℚ := (4 * X + 3 * Y) / (5 * Z - 2 * X)

-- The proof problem itself
theorem ratio_expression (X Y Z : ℚ) (h : ratio X Y Z) : expr X Y Z = 3 / 4 := by
  sorry

end ratio_expression_l1941_194128


namespace partI_solution_set_partII_range_of_a_l1941_194197

namespace MathProof

-- Define the function f(x)
def f (x a : ℝ) : ℝ := abs (x - a) - abs (x + 3)

-- Part (Ⅰ) Proof Problem
theorem partI_solution_set (x : ℝ) : 
  f x (-1) ≤ 1 ↔ -5/2 ≤ x :=
sorry

-- Part (Ⅱ) Proof Problem
theorem partII_range_of_a (a : ℝ) : 
  (∀ x, 0 ≤ x ∧ x ≤ 3 → f x a ≤ 4) ↔ -7 ≤ a ∧ a ≤ 7 :=
sorry

end MathProof

end partI_solution_set_partII_range_of_a_l1941_194197


namespace square_side_length_exists_l1941_194147

-- Define the dimensions of the tile
structure Tile where
  width : Nat
  length : Nat

-- Define the specific tile used in the problem
def given_tile : Tile :=
  { width := 16, length := 24 }

-- Define the condition of forming a square using 6 tiles
def forms_square_with_6_tiles (tile : Tile) (side_length : Nat) : Prop :=
  (2 * tile.length = side_length) ∧ (3 * tile.width = side_length)

-- Problem statement requiring proof
theorem square_side_length_exists : forms_square_with_6_tiles given_tile 48 :=
  sorry

end square_side_length_exists_l1941_194147


namespace probability_of_correct_match_l1941_194171

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def total_possible_arrangements : ℕ :=
  factorial 4

def correct_arrangements : ℕ :=
  1

def probability_correct_match : ℚ :=
  correct_arrangements / total_possible_arrangements

theorem probability_of_correct_match : probability_correct_match = 1 / 24 :=
by
  -- Proof is omitted
  sorry

end probability_of_correct_match_l1941_194171


namespace part1_part2_part3_l1941_194168

open Real

-- Definition of "$k$-derived point"
def k_derived_point (P : ℝ × ℝ) (k : ℝ) : ℝ × ℝ := (P.1 + k * P.2, k * P.1 + P.2)

-- Problem statements to prove
theorem part1 :
  k_derived_point (-2, 3) 2 = (4, -1) :=
sorry

theorem part2 (P : ℝ × ℝ) (h : k_derived_point P 3 = (9, 11)) :
  P = (3, 2) :=
sorry

theorem part3 (b k : ℝ) (h1 : b > 0) (h2 : |k * b| ≥ 5 * b) :
  k ≥ 5 ∨ k ≤ -5 :=
sorry

end part1_part2_part3_l1941_194168


namespace average_height_of_trees_l1941_194140

def first_tree_height : ℕ := 1000
def half_tree_height : ℕ := first_tree_height / 2
def last_tree_height : ℕ := first_tree_height + 200

def total_height : ℕ := first_tree_height + 2 * half_tree_height + last_tree_height
def number_of_trees : ℕ := 4
def average_height : ℕ := total_height / number_of_trees

theorem average_height_of_trees :
  average_height = 800 :=
by
  -- This line contains a placeholder proof, the actual proof is omitted.
  sorry

end average_height_of_trees_l1941_194140


namespace max_profit_30000_l1941_194126

noncomputable def max_profit (type_A : ℕ) (type_B : ℕ) : ℝ := 
  10000 * type_A + 5000 * type_B

theorem max_profit_30000 :
  ∃ (type_A type_B : ℕ), 
  (4 * type_A + 1 * type_B ≤ 10) ∧
  (18 * type_A + 15 * type_B ≤ 66) ∧
  max_profit type_A type_B = 30000 :=
sorry

end max_profit_30000_l1941_194126


namespace line_equation_135_deg_l1941_194170

theorem line_equation_135_deg (A : ℝ × ℝ) (theta : ℝ) (l : ℝ → ℝ → Prop) :
  A = (1, -2) →
  theta = 135 →
  (∀ x y, l x y ↔ y = -(x - 1) - 2) →
  ∀ x y, l x y ↔ x + y + 1 = 0 :=
by
  intros hA hTheta hl_form
  sorry

end line_equation_135_deg_l1941_194170


namespace ron_pay_cuts_l1941_194176

-- Define percentages as decimals
def cut_1 : ℝ := 0.05
def cut_2 : ℝ := 0.10
def cut_3 : ℝ := 0.15
def overall_cut : ℝ := 0.27325

-- Define the total number of pay cuts
def total_pay_cuts : ℕ := 3

noncomputable def verify_pay_cuts (cut_1 cut_2 cut_3 overall_cut : ℝ) (total_pay_cuts : ℕ) : Prop :=
  (((1 - cut_1) * (1 - cut_2) * (1 - cut_3) = (1 - overall_cut)) ∧ (total_pay_cuts = 3))

theorem ron_pay_cuts 
  (cut_1 : ℝ := 0.05)
  (cut_2 : ℝ := 0.10)
  (cut_3 : ℝ := 0.15)
  (overall_cut : ℝ := 0.27325)
  (total_pay_cuts : ℕ := 3) 
  : verify_pay_cuts cut_1 cut_2 cut_3 overall_cut total_pay_cuts :=
by sorry

end ron_pay_cuts_l1941_194176


namespace C_increases_as_n_increases_l1941_194199

noncomputable def C (e R r n : ℝ) : ℝ := (e * n^2) / (R + n * r)

theorem C_increases_as_n_increases
  (e R r : ℝ) (e_pos : 0 < e) (R_pos : 0 < R) (r_pos : 0 < r) :
  ∀ n : ℝ, 0 < n → ∃ M : ℝ, ∀ N : ℝ, N > n → C e R r N > M :=
by
  sorry

end C_increases_as_n_increases_l1941_194199


namespace one_percent_as_decimal_l1941_194177

theorem one_percent_as_decimal : (1 / 100 : ℝ) = 0.01 := by
  sorry

end one_percent_as_decimal_l1941_194177


namespace dictionary_cost_l1941_194141

def dinosaur_book_cost : ℕ := 19
def children_cookbook_cost : ℕ := 7
def saved_amount : ℕ := 8
def needed_amount : ℕ := 29

def total_amount_needed := saved_amount + needed_amount
def combined_books_cost := dinosaur_book_cost + children_cookbook_cost

theorem dictionary_cost : total_amount_needed - combined_books_cost = 11 :=
by
  -- proof omitted
  sorry

end dictionary_cost_l1941_194141


namespace min_fence_length_l1941_194125

theorem min_fence_length (w l F: ℝ) (h1: l = 2 * w) (h2: 2 * w^2 ≥ 500) : F = 96 :=
by sorry

end min_fence_length_l1941_194125


namespace arrangement_count_correct_l1941_194153

def num_arrangements_exactly_two_females_next_to_each_other (males : ℕ) (females : ℕ) : ℕ :=
  if males = 4 ∧ females = 3 then 3600 else 0

theorem arrangement_count_correct :
  num_arrangements_exactly_two_females_next_to_each_other 4 3 = 3600 :=
by
  sorry

end arrangement_count_correct_l1941_194153


namespace bus_capacity_l1941_194180

theorem bus_capacity :
  ∀ (left_seats right_seats people_per_seat back_seat : ℕ),
  left_seats = 15 →
  right_seats = left_seats - 3 →
  people_per_seat = 3 →
  back_seat = 11 →
  (left_seats * people_per_seat) + 
  (right_seats * people_per_seat) + 
  back_seat = 92 := by
  intros left_seats right_seats people_per_seat back_seat 
  intros h1 h2 h3 h4 
  sorry

end bus_capacity_l1941_194180


namespace greatest_possible_integer_radius_l1941_194130

theorem greatest_possible_integer_radius (r : ℕ) (h : ∀ (A : ℝ), A = Real.pi * (r : ℝ)^2 → A < 75 * Real.pi) : r ≤ 8 :=
by sorry

end greatest_possible_integer_radius_l1941_194130


namespace fraction_ordering_l1941_194169

theorem fraction_ordering :
  (8 / 25 : ℚ) < 6 / 17 ∧ 6 / 17 < 10 / 27 ∧ 8 / 25 < 10 / 27 :=
by
  sorry

end fraction_ordering_l1941_194169


namespace complex_sum_magnitude_eq_three_l1941_194113

open Complex

theorem complex_sum_magnitude_eq_three (a b c : ℂ) 
    (h1 : abs a = 1) 
    (h2 : abs b = 1) 
    (h3 : abs c = 1) 
    (h4 : a^3 / (b * c) + b^3 / (a * c) + c^3 / (a * b) = -3) : 
    abs (a + b + c) = 3 := 
sorry

end complex_sum_magnitude_eq_three_l1941_194113


namespace total_carrots_l1941_194198

-- Definitions from conditions in a)
def JoanCarrots : ℕ := 29
def JessicaCarrots : ℕ := 11

-- Theorem that encapsulates the problem
theorem total_carrots : JoanCarrots + JessicaCarrots = 40 := by
  sorry

end total_carrots_l1941_194198


namespace TimSpentThisMuch_l1941_194167

/-- Tim's lunch cost -/
def lunchCost : ℝ := 50.50

/-- Tip percentage -/
def tipPercent : ℝ := 0.20

/-- Calculate the tip amount -/
def tipAmount := tipPercent * lunchCost

/-- Calculate the total amount spent -/
def totalAmountSpent := lunchCost + tipAmount

/-- Prove that the total amount spent is as expected -/
theorem TimSpentThisMuch : totalAmountSpent = 60.60 :=
  sorry

end TimSpentThisMuch_l1941_194167


namespace intersection_M_N_l1941_194166

def M := { x : ℝ | x < 2011 }
def N := { x : ℝ | 0 < x ∧ x < 1 }

theorem intersection_M_N :
  M ∩ N = { x : ℝ | 0 < x ∧ x < 1 } := 
by 
  sorry

end intersection_M_N_l1941_194166


namespace sum_of_numbers_l1941_194123

theorem sum_of_numbers (a : ℝ) (n : ℕ) (h : a = 5.3) (hn : n = 8) : (a * n) = 42.4 :=
sorry

end sum_of_numbers_l1941_194123


namespace john_reading_time_l1941_194101

theorem john_reading_time:
  let weekday_hours_moses := 1.5
  let weekday_rate_moses := 30
  let saturday_hours_moses := 2
  let saturday_rate_moses := 40
  let pages_moses := 450
  let weekday_hours_rest := 1.5
  let weekday_rate_rest := 45
  let saturday_hours_rest := 2.5
  let saturday_rate_rest := 60
  let pages_rest := 2350
  let weekdays_per_week := 5
  let saturdays_per_week := 1
  let total_pages_per_week_moses := (weekday_hours_moses * weekday_rate_moses * weekdays_per_week) + 
                                    (saturday_hours_moses * saturday_rate_moses * saturdays_per_week)
  let total_pages_per_week_rest := (weekday_hours_rest * weekday_rate_rest * weekdays_per_week) + 
                                   (saturday_hours_rest * saturday_rate_rest * saturdays_per_week)
  let weeks_moses := (pages_moses / total_pages_per_week_moses).ceil
  let weeks_rest := (pages_rest / total_pages_per_week_rest).ceil
  let total_weeks := weeks_moses + weeks_rest
  total_weeks = 7 :=
by
  -- placeholders for the proof steps.
  sorry

end john_reading_time_l1941_194101


namespace spaceship_initial_people_count_l1941_194181

/-- For every 100 additional people that board a spaceship, its speed is halved.
     The speed of the spaceship with a certain number of people on board is 500 km per hour.
     The speed of the spaceship when there are 400 people on board is 125 km/hr.
     Prove that the number of people on board when the spaceship was moving at 500 km/hr is 200. -/
theorem spaceship_initial_people_count (speed : ℕ → ℕ) (n : ℕ) :
  (∀ k, speed (k + 100) = speed k / 2) →
  speed n = 500 →
  speed 400 = 125 →
  n = 200 :=
by
  intro half_speed speed_500 speed_400
  sorry

end spaceship_initial_people_count_l1941_194181


namespace division_problem_l1941_194155

theorem division_problem : 160 / (10 + 11 * 2) = 5 := 
  by 
    sorry

end division_problem_l1941_194155


namespace find_N_l1941_194143

theorem find_N : ∃ N : ℕ, 36^2 * 72^2 = 12^2 * N^2 ∧ N = 216 :=
by
  sorry

end find_N_l1941_194143


namespace rose_price_vs_carnation_price_l1941_194108

variable (x y : ℝ)

theorem rose_price_vs_carnation_price
  (h1 : 3 * x + 2 * y > 8)
  (h2 : 2 * x + 3 * y < 7) :
  x > 2 * y :=
sorry

end rose_price_vs_carnation_price_l1941_194108


namespace sum_of_all_possible_k_values_l1941_194118

theorem sum_of_all_possible_k_values (j k : ℕ) (h : 1 / j + 1 / k = 1 / 4) : 
  (∃ j k : ℕ, (j > 0 ∧ k > 0) ∧ (1 / j + 1 / k = 1 / 4) ∧ (k = 8 ∨ k = 12 ∨ k = 20)) →
  (8 + 12 + 20 = 40) :=
by
  sorry

end sum_of_all_possible_k_values_l1941_194118


namespace empty_solution_set_of_inequalities_l1941_194156

theorem empty_solution_set_of_inequalities (a : ℝ) : 
  (∀ x : ℝ, ¬ ((2 * x < 5 - 3 * x) ∧ ((x - 1) / 2 > a))) ↔ (0 ≤ a) := 
by
  sorry

end empty_solution_set_of_inequalities_l1941_194156


namespace number_division_l1941_194195

theorem number_division (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 :=
sorry

end number_division_l1941_194195


namespace distance_between_points_l1941_194134

theorem distance_between_points (A B : ℝ) (dA : |A| = 2) (dB : |B| = 7) : |A - B| = 5 ∨ |A - B| = 9 := 
by
  sorry

end distance_between_points_l1941_194134


namespace complete_pairs_of_socks_l1941_194119

def initial_pairs_blue : ℕ := 20
def initial_pairs_green : ℕ := 15
def initial_pairs_red : ℕ := 15

def lost_socks_blue : ℕ := 3
def lost_socks_green : ℕ := 2
def lost_socks_red : ℕ := 2

def donated_socks_blue : ℕ := 10
def donated_socks_green : ℕ := 15
def donated_socks_red : ℕ := 10

def purchased_pairs_blue : ℕ := 5
def purchased_pairs_green : ℕ := 3
def purchased_pairs_red : ℕ := 2

def gifted_pairs_blue : ℕ := 2
def gifted_pairs_green : ℕ := 1

theorem complete_pairs_of_socks : 
  (initial_pairs_blue - 1 - (donated_socks_blue / 2) + purchased_pairs_blue + gifted_pairs_blue) +
  (initial_pairs_green - 1 - (donated_socks_green / 2) + purchased_pairs_green + gifted_pairs_green) +
  (initial_pairs_red - 1 - (donated_socks_red / 2) + purchased_pairs_red) = 43 := by
  sorry

end complete_pairs_of_socks_l1941_194119


namespace remainder_of_sum_l1941_194186

theorem remainder_of_sum (f y z : ℤ) 
  (hf : f % 5 = 3)
  (hy : y % 5 = 4)
  (hz : z % 7 = 6) : 
  (f + y + z) % 35 = 13 :=
by
  sorry

end remainder_of_sum_l1941_194186


namespace stock_worth_l1941_194117

theorem stock_worth (X : ℝ)
  (H1 : 0.2 * X * 0.1 = 0.02 * X)  -- 20% of stock at 10% profit given in condition.
  (H2 : 0.8 * X * 0.05 = 0.04 * X) -- Remaining 80% of stock at 5% loss given in condition.
  (H3 : 0.04 * X - 0.02 * X = 400) -- Overall loss incurred is Rs. 400.
  : X = 20000 := 
sorry

end stock_worth_l1941_194117


namespace maximum_value_expression_l1941_194102

theorem maximum_value_expression (a b c : ℝ) (h : a^2 + b^2 + c^2 = 9) : 
  (a - b)^2 + (b - c)^2 + (c - a)^2 ≤ 27 :=
sorry

end maximum_value_expression_l1941_194102


namespace total_money_l1941_194152

variable (A B C : ℕ)

theorem total_money
  (h1 : A + C = 250)
  (h2 : B + C = 450)
  (h3 : C = 100) :
  A + B + C = 600 := by
  sorry

end total_money_l1941_194152


namespace phil_books_remaining_pages_l1941_194121

/-- We define the initial number of books and the number of pages per book. -/
def initial_books : Nat := 10
def pages_per_book : Nat := 100
def lost_books : Nat := 2

/-- The goal is to find the total number of pages Phil has left after losing 2 books. -/
theorem phil_books_remaining_pages : (initial_books - lost_books) * pages_per_book = 800 := by 
  -- The proof will go here
  sorry

end phil_books_remaining_pages_l1941_194121


namespace top_width_is_76_l1941_194139

-- Definitions of the conditions
def bottom_width : ℝ := 4
def area : ℝ := 10290
def depth : ℝ := 257.25

-- The main theorem to prove that the top width equals 76 meters
theorem top_width_is_76 (x : ℝ) (h : 10290 = 1/2 * (x + 4) * 257.25) : x = 76 :=
by {
  sorry
}

end top_width_is_76_l1941_194139


namespace min_diff_f_l1941_194103

def f (x : ℝ) := 2017 * x ^ 2 - 2018 * x + 2019 * 2020

theorem min_diff_f (t : ℝ) : 
  let f_max := max (f t) (f (t + 2))
  let f_min := min (f t) (f (t + 2))
  (f_max - f_min) ≥ 2017 :=
sorry

end min_diff_f_l1941_194103


namespace cosine_function_range_l1941_194115

theorem cosine_function_range : 
  (∀ x ∈ Set.Icc (-Real.pi / 6) (2 * Real.pi / 3), -1/2 ≤ Real.cos x ∧ Real.cos x ≤ 1) ∧
  (∃ a ∈ Set.Icc (-Real.pi / 6) (2 * Real.pi / 3), Real.cos a = 1) ∧
  (∃ b ∈ Set.Icc (-Real.pi / 6) (2 * Real.pi / 3), Real.cos b = -1/2) :=
by
  sorry

end cosine_function_range_l1941_194115


namespace inverse_geometric_sequence_l1941_194135

-- Define that a, b, c form a geometric sequence
def geometric_sequence (a b c : ℝ) := b^2 = a * c

-- Define the theorem: if b^2 = a * c, then a, b, c form a geometric sequence
theorem inverse_geometric_sequence (a b c : ℝ) (h : b^2 = a * c) : geometric_sequence a b c :=
by
  sorry

end inverse_geometric_sequence_l1941_194135


namespace max_length_third_side_l1941_194132

open Real

theorem max_length_third_side (A B C : ℝ) (a b c : ℝ) 
  (h1 : cos (2 * A) + cos (2 * B) + cos (2 * C) = 1)
  (h2 : a = 9) 
  (h3 : b = 12)
  (h4 : a^2 + b^2 = c^2) : 
  c = 15 := 
sorry

end max_length_third_side_l1941_194132


namespace nine_div_repeating_decimal_l1941_194138

noncomputable def repeating_decimal := 1 / 3

theorem nine_div_repeating_decimal : 9 / repeating_decimal = 27 := by
  sorry

end nine_div_repeating_decimal_l1941_194138


namespace Roberta_spent_on_shoes_l1941_194194

-- Define the conditions as per the problem statement
variables (S B L : ℝ) (h1 : B = S - 17) (h2 : L = B / 4) (h3 : 158 - (S + B + L) = 78)

-- State the theorem to be proved
theorem Roberta_spent_on_shoes : S = 45 :=
by
  -- use variables and conditions
  have := h1
  have := h2
  have := h3
  sorry -- Proof steps can be filled later

end Roberta_spent_on_shoes_l1941_194194


namespace prime_pairs_l1941_194114

-- Define the predicate to check whether a number is a prime.
def is_prime (n : Nat) : Prop := Nat.Prime n

-- Define the main theorem.
theorem prime_pairs (p q : Nat) (hp : is_prime p) (hq : is_prime q) : 
  (p^3 - q^5 = (p + q)^2) → (p = 7 ∧ q = 3) :=
by
  sorry

end prime_pairs_l1941_194114


namespace solve_for_x_l1941_194191

def f (x : ℝ) : ℝ := 3 * x - 5

theorem solve_for_x : ∃ x, 2 * f x - 16 = f (x - 6) ∧ x = 1 := by
  exists 1
  sorry

end solve_for_x_l1941_194191
