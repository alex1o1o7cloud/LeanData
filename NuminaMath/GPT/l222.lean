import Mathlib

namespace cost_price_of_apple_l222_22246

theorem cost_price_of_apple (C : ℚ) (h1 : 19 = 5/6 * C) : C = 22.8 := by
  sorry

end cost_price_of_apple_l222_22246


namespace not_divisible_by_3_or_4_l222_22253

theorem not_divisible_by_3_or_4 (n : ℤ) : 
  ¬ (n^2 + 1) % 3 = 0 ∧ ¬ (n^2 + 1) % 4 = 0 := 
by
  sorry

end not_divisible_by_3_or_4_l222_22253


namespace price_of_first_tea_x_l222_22237

theorem price_of_first_tea_x (x : ℝ) :
  let price_second := 135
  let price_third := 173.5
  let avg_price := 152
  let ratio := [1, 1, 2]
  1 * x + 1 * price_second + 2 * price_third = 4 * avg_price -> x = 126 :=
by
  intros price_second price_third avg_price ratio h
  sorry

end price_of_first_tea_x_l222_22237


namespace sin_double_angle_l222_22275

theorem sin_double_angle 
  (α β : ℝ)
  (h1 : 0 < β)
  (h2 : β < α)
  (h3 : α < π / 4)
  (h_cos_diff : Real.cos (α - β) = 12 / 13)
  (h_sin_sum : Real.sin (α + β) = 4 / 5) :
  Real.sin (2 * α) = 63 / 65 := 
sorry

end sin_double_angle_l222_22275


namespace solve_for_x_l222_22270

theorem solve_for_x (x : ℝ) : 
  5 * x + 9 * x = 420 - 12 * (x - 4) -> 
  x = 18 :=
by
  intro h
  -- derivation will follow here
  sorry

end solve_for_x_l222_22270


namespace xy_identity_l222_22244

theorem xy_identity (x y : ℝ) (h1 : (x + y)^2 = 16) (h2 : x * y = 6) : x^2 + y^2 = 4 := 
by 
  sorry

end xy_identity_l222_22244


namespace first_driver_spends_less_time_l222_22289

noncomputable def round_trip_time (d : ℝ) (v₁ v₂ : ℝ) : ℝ := (d / v₁) + (d / v₂)

theorem first_driver_spends_less_time (d : ℝ) : 
  round_trip_time d 80 80 < round_trip_time d 90 70 :=
by
  --We skip the proof here
  sorry

end first_driver_spends_less_time_l222_22289


namespace complement_of_A_is_correct_l222_22263

open Set

variable (U : Set ℝ) (A : Set ℝ)

def complement_of_A (U : Set ℝ) (A : Set ℝ) :=
  {x : ℝ | x ∉ A}

theorem complement_of_A_is_correct :
  (U = univ) →
  (A = {x : ℝ | x^2 - 2 * x > 0}) →
  (complement_of_A U A = {x : ℝ | 0 ≤ x ∧ x ≤ 2}) :=
by
  intros hU hA
  simp [hU, hA, complement_of_A]
  sorry

end complement_of_A_is_correct_l222_22263


namespace total_marble_weight_l222_22285

theorem total_marble_weight (w1 w2 w3 : ℝ) (h_w1 : w1 = 0.33) (h_w2 : w2 = 0.33) (h_w3 : w3 = 0.08) :
  w1 + w2 + w3 = 0.74 :=
by {
  sorry
}

end total_marble_weight_l222_22285


namespace angle_sum_impossible_l222_22212

theorem angle_sum_impossible (A1 A2 A3 : ℝ) (h : A1 + A2 + A3 = 180) :
  ¬ ((A1 > 90 ∧ A2 > 90 ∧ A3 < 90) ∨ (A1 > 90 ∧ A3 > 90 ∧ A2 < 90) ∨ (A2 > 90 ∧ A3 > 90 ∧ A1 < 90)) :=
sorry

end angle_sum_impossible_l222_22212


namespace binomial_sum_zero_l222_22218

open BigOperators

theorem binomial_sum_zero {n m : ℕ} (h1 : 1 ≤ m) (h2 : m < n) :
  ∑ k in Finset.range (n + 1), (-1 : ℤ) ^ k * k ^ m * Nat.choose n k = 0 :=
by
  sorry

end binomial_sum_zero_l222_22218


namespace sum_possible_n_k_l222_22221

theorem sum_possible_n_k (n k : ℕ) (h1 : 3 * k + 3 = n) (h2 : 5 * (k + 2) = 3 * (n - k - 1)) : 
  n + k = 19 := 
by {
  sorry -- proof steps would go here
}

end sum_possible_n_k_l222_22221


namespace transport_cost_is_correct_l222_22250

-- Define the transport cost per kilogram
def transport_cost_per_kg : ℝ := 18000

-- Define the weight of the scientific instrument in kilograms
def weight_kg : ℝ := 0.5

-- Define the discount rate
def discount_rate : ℝ := 0.10

-- Define the cost calculation without the discount
def cost_without_discount : ℝ := weight_kg * transport_cost_per_kg

-- Define the final cost with the discount applied
def discounted_cost : ℝ := cost_without_discount * (1 - discount_rate)

-- The theorem stating that the discounted cost is $8,100
theorem transport_cost_is_correct : discounted_cost = 8100 := by
  sorry

end transport_cost_is_correct_l222_22250


namespace obtuse_triangle_range_a_l222_22291

noncomputable def is_obtuse_triangle (a b c : ℝ) : Prop :=
  ∃ (θ : ℝ), θ > 90 ∧ θ ≤ 120 ∧ c^2 > a^2 + b^2

theorem obtuse_triangle_range_a (a : ℝ) :
  (a + (a + 1) > a + 2) →
  is_obtuse_triangle a (a + 1) (a + 2) →
  (1.5 ≤ a ∧ a < 3) :=
by
  sorry

end obtuse_triangle_range_a_l222_22291


namespace problem1_problem2_l222_22239

theorem problem1 (x : ℕ) : 
  2 / 8^x * 16^x = 2^5 → x = 4 := 
by
  sorry

theorem problem2 (x : ℕ) : 
  2^(x+2) + 2^(x+1) = 24 → x = 2 := 
by
  sorry

end problem1_problem2_l222_22239


namespace original_cube_volume_l222_22268

theorem original_cube_volume (a : ℕ) (h : (a + 2) * (a + 1) * (a - 1) + 6 = a^3) : a = 2 :=
by sorry

example : 2^3 = 8 := by norm_num

end original_cube_volume_l222_22268


namespace original_rice_amount_l222_22208

theorem original_rice_amount (x : ℝ) 
  (h1 : (x / 2) - 3 = 18) : 
  x = 42 :=
sorry

end original_rice_amount_l222_22208


namespace selling_price_is_correct_l222_22271

def profit_percent : ℝ := 0.6
def cost_price : ℝ := 375
def profit : ℝ := profit_percent * cost_price
def selling_price : ℝ := cost_price + profit

theorem selling_price_is_correct : selling_price = 600 :=
by
  -- proof steps would go here
  sorry

end selling_price_is_correct_l222_22271


namespace count_valid_triples_l222_22203

def S (n : ℕ) : ℕ :=
  (n / 100) + (n % 100 / 10) + (n % 10)

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def satisfies_conditions (a b c : ℕ) : Prop :=
  is_three_digit a ∧ is_three_digit b ∧ is_three_digit c ∧ 
  (a + b + c = 2005) ∧ (S a + S b + S c = 61)

def number_of_valid_triples : ℕ := sorry

theorem count_valid_triples : number_of_valid_triples = 17160 :=
sorry

end count_valid_triples_l222_22203


namespace max_squares_at_a1_bksq_l222_22265

noncomputable def maximizePerfectSquares (a b : ℕ) : Prop := 
a ≠ b ∧ 
(∃ k : ℕ, k ≠ 1 ∧ b = k^2) ∧ 
a = 1

theorem max_squares_at_a1_bksq (a b : ℕ) : maximizePerfectSquares a b := 
by 
  sorry

end max_squares_at_a1_bksq_l222_22265


namespace total_campers_l222_22284

def campers_morning : ℕ := 36
def campers_afternoon : ℕ := 13
def campers_evening : ℕ := 49

theorem total_campers : campers_morning + campers_afternoon + campers_evening = 98 := by
  sorry

end total_campers_l222_22284


namespace not_collinear_C_vector_decomposition_l222_22231

namespace VectorProof

open Function

structure Vector2 where
  x : ℝ
  y : ℝ

def add (v1 v2 : Vector2) : Vector2 := ⟨v1.x + v2.x, v1.y + v2.y⟩
def scale (c : ℝ) (v : Vector2) : Vector2 := ⟨c * v.x, c * v.y⟩

def collinear (v1 v2 : Vector2) : Prop :=
  ∃ k : ℝ, v2 = scale k v1

def vector_a : Vector2 := ⟨3, 4⟩
def e₁_C : Vector2 := ⟨-1, 2⟩
def e₂_C : Vector2 := ⟨3, -1⟩

theorem not_collinear_C :
  ¬ collinear e₁_C e₂_C :=
sorry

theorem vector_decomposition :
  ∃ (x y : ℝ), vector_a = add (scale x e₁_C) (scale y e₂_C) :=
sorry

end VectorProof

end not_collinear_C_vector_decomposition_l222_22231


namespace smallest_sum_of_factors_of_8_l222_22297

theorem smallest_sum_of_factors_of_8! :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
  a * b * c * d = Nat.factorial 8 ∧ a + b + c + d = 102 :=
sorry

end smallest_sum_of_factors_of_8_l222_22297


namespace solve_b_values_l222_22235

open Int

theorem solve_b_values :
  {b : ℤ | ∃ x1 x2 x3 : ℤ, x1^2 + b * x1 - 2 ≤ 0 ∧ x2^2 + b * x2 - 2 ≤ 0 ∧ x3^2 + b * x3 - 2 ≤ 0 ∧
  ∀ x : ℤ, x ≠ x1 ∧ x ≠ x2 ∧ x ≠ x3 → x^2 + b * x - 2 > 0} = { -4, -3 } :=
by sorry

end solve_b_values_l222_22235


namespace car_total_distance_l222_22298

theorem car_total_distance (h1 h2 h3 : ℕ) :
  h1 = 180 → h2 = 160 → h3 = 220 → h1 + h2 + h3 = 560 :=
by
  intros h1_eq h2_eq h3_eq
  sorry

end car_total_distance_l222_22298


namespace problem_statement_l222_22233

theorem problem_statement (m : ℝ) (h : m^2 + m - 1 = 0) : m^4 + 2*m^3 - m + 2007 = 2007 := 
by 
  sorry

end problem_statement_l222_22233


namespace value_of_4m_plus_2n_l222_22211

-- Given that the equation 2kx + 2m = 6 - 2x + nk 
-- has a solution independent of k
theorem value_of_4m_plus_2n (m n : ℝ) 
  (h : ∃ x : ℝ, ∀ k : ℝ, 2 * k * x + 2 * m = 6 - 2 * x + n * k) : 
  4 * m + 2 * n = 12 :=
by
  sorry

end value_of_4m_plus_2n_l222_22211


namespace find_real_a_l222_22213

theorem find_real_a (a : ℝ) : 
  (a ^ 2 + 2 * a - 15 = 0) ∧ (a ^ 2 + 4 * a - 5 ≠ 0) → a = 3 :=
by 
  sorry

end find_real_a_l222_22213


namespace division_problem_l222_22210

theorem division_problem 
  (a b c d e f g h i : ℕ) 
  (h1 : a = 7) 
  (h2 : b = 9) 
  (h3 : c = 8) 
  (h4 : d = 1) 
  (h5 : e = 2) 
  (h6 : f = 3) 
  (h7 : g = 4) 
  (h8 : h = 6) 
  (h9 : i = 0) 
  : 7981 / 23 = 347 := 
by 
  sorry

end division_problem_l222_22210


namespace find_three_digit_number_l222_22283

theorem find_three_digit_number (P Q R : ℕ) 
  (h1 : P ≠ Q) 
  (h2 : P ≠ R) 
  (h3 : Q ≠ R) 
  (h4 : P < 7) 
  (h5 : Q < 7) 
  (h6 : R < 7)
  (h7 : P ≠ 0) 
  (h8 : Q ≠ 0) 
  (h9 : R ≠ 0) 
  (h10 : 7 * P + Q + R = 7 * R) 
  (h11 : (7 * P + Q) + (7 * Q + P) = 49 + 7 * R + R)
  : P * 100 + Q * 10 + R = 434 :=
sorry

end find_three_digit_number_l222_22283


namespace max_distance_from_point_to_line_l222_22273

theorem max_distance_from_point_to_line (θ m : ℝ) :
  let P := (Real.cos θ, Real.sin θ)
  let d := (P.1 - m * P.2 - 2) / Real.sqrt (1 + m^2)
  ∃ (θ m : ℝ), d ≤ 3 := sorry

end max_distance_from_point_to_line_l222_22273


namespace fishing_problem_l222_22216

theorem fishing_problem (a b c d : ℕ)
  (h1 : a + b + c + d = 11)
  (h2 : 1 ≤ a) 
  (h3 : 1 ≤ b) 
  (h4 : 1 ≤ c) 
  (h5 : 1 ≤ d) : 
  a < 3 ∨ b < 3 ∨ c < 3 ∨ d < 3 :=
by
  -- This is a placeholder for the proof
  sorry

end fishing_problem_l222_22216


namespace experienced_sailors_monthly_earnings_l222_22232

theorem experienced_sailors_monthly_earnings :
  let total_sailors : Nat := 17
  let inexperienced_sailors : Nat := 5
  let hourly_wage_inexperienced : Nat := 10
  let workweek_hours : Nat := 60
  let weeks_in_month : Nat := 4
  let experienced_sailors : Nat := total_sailors - inexperienced_sailors
  let hourly_wage_experienced := hourly_wage_inexperienced + (hourly_wage_inexperienced / 5)
  let weekly_earnings_experienced := hourly_wage_experienced * workweek_hours
  let total_weekly_earnings_experienced := weekly_earnings_experienced * experienced_sailors
  let monthly_earnings_experienced := total_weekly_earnings_experienced * weeks_in_month
  monthly_earnings_experienced = 34560 := by
  sorry

end experienced_sailors_monthly_earnings_l222_22232


namespace total_earning_correct_l222_22279

-- Definitions based on conditions
def daily_wage_c : ℕ := 105
def days_worked_a : ℕ := 6
def days_worked_b : ℕ := 9
def days_worked_c : ℕ := 4

-- Given the ratio of their daily wages
def ratio_a : ℕ := 3
def ratio_b : ℕ := 4
def ratio_c : ℕ := 5

-- Now we calculate the daily wages based on the ratio
def unit_wage : ℕ := daily_wage_c / ratio_c
def daily_wage_a : ℕ := ratio_a * unit_wage
def daily_wage_b : ℕ := ratio_b * unit_wage

-- Total earnings are calculated by multiplying daily wages and days worked
def total_earning_a : ℕ := days_worked_a * daily_wage_a
def total_earning_b : ℕ := days_worked_b * daily_wage_b
def total_earning_c : ℕ := days_worked_c * daily_wage_c

def total_earning : ℕ := total_earning_a + total_earning_b + total_earning_c

-- Theorem to prove
theorem total_earning_correct : total_earning = 1554 := by
  sorry

end total_earning_correct_l222_22279


namespace maximum_initial_jars_l222_22299

-- Define the conditions given in the problem
def initial_total_weight_carlson (n : ℕ) : ℕ := 13 * n
def new_total_weight_carlson (n a : ℕ) : ℕ := 13 * n - a
def total_weight_after_giving (n a : ℕ) : ℕ := 8 * (n + a)

-- Theorem statement for the maximum possible jars Carlson could have initially had
theorem maximum_initial_jars (n a k : ℕ) (h1 : initial_total_weight_carlson n = 13 * n)
  (h2 : new_total_weight_carlson n a = 8 * (n + a)) (h3 : n = 9 * k)
  (h4 : a = 5 * k) : (initial_total_weight_carlson n / a) ≤ 23 :=
by
  sorry

end maximum_initial_jars_l222_22299


namespace fraction_of_girls_l222_22217

variable (T G B : ℕ) -- The total number of students, number of girls, and number of boys
variable (x : ℚ) -- The fraction of the number of girls

-- Definitions based on the given conditions
def fraction_condition : Prop := x * G = (1/6) * T
def ratio_condition : Prop := (B : ℚ) / (G : ℚ) = 2
def total_students : Prop := T = B + G

-- The statement we need to prove
theorem fraction_of_girls (h1 : fraction_condition T G x)
                          (h2 : ratio_condition B G)
                          (h3 : total_students T G B):
  x = 1/2 :=
by
  sorry

end fraction_of_girls_l222_22217


namespace more_plastic_pipe_l222_22294

variable (m_copper m_plastic : Nat)
variable (total_cost cost_per_meter : Nat)

-- Conditions
variable (h1 : m_copper = 10)
variable (h2 : cost_per_meter = 4)
variable (h3 : total_cost = 100)
variable (h4 : m_copper * cost_per_meter + m_plastic * cost_per_meter = total_cost)

-- Proof that the number of more meters of plastic pipe bought compared to the copper pipe is 5
theorem more_plastic_pipe :
  m_plastic - m_copper = 5 :=
by
  -- Since proof is not required, we place sorry here.
  sorry

end more_plastic_pipe_l222_22294


namespace pies_sold_by_mcgee_l222_22207

/--
If Smith's Bakery sold 70 pies, and they sold 6 more than four times the number of pies that Mcgee's Bakery sold,
prove that Mcgee's Bakery sold 16 pies.
-/
theorem pies_sold_by_mcgee (x : ℕ) (h1 : 4 * x + 6 = 70) : x = 16 :=
by
  sorry

end pies_sold_by_mcgee_l222_22207


namespace sequence_a_l222_22274

theorem sequence_a (a : ℕ → ℝ) 
  (h1 : a 1 = 2)
  (h2 : a 2 = 1)
  (h3 : ∀ n ≥ 2, a n / a (n + 1) + a n / a (n - 1) = 2) :
  a 12 = 1 / 6 :=
sorry

end sequence_a_l222_22274


namespace decreasing_interval_of_f_l222_22214

noncomputable def f (x : ℝ) : ℝ := (1 / 2)^(x^2 - 2 * x)

theorem decreasing_interval_of_f :
  ∀ x y : ℝ, (1 < x ∧ x < y) → f y < f x :=
by
  sorry

end decreasing_interval_of_f_l222_22214


namespace value_of_b_cannot_form_arithmetic_sequence_l222_22278

theorem value_of_b 
  (a1 : ℝ) (a2 : ℝ) (a3 : ℝ) 
  (h1 : a1 = 150)
  (h2 : a2 = b)
  (h3 : a3 = 60 / 36)
  (h4 : b > 0) :
  b = 5 * Real.sqrt 10 := 
sorry

theorem cannot_form_arithmetic_sequence 
  (d : ℝ)
  (a1 : ℝ) (a2 : ℝ) (a3 : ℝ) 
  (h1 : a1 = 150)
  (h2 : a2 = b)
  (h3 : a3 = 60 / 36)
  (h4 : b = 5 * Real.sqrt 10) :
  ¬(∃ d, a1 + d = a2 ∧ a2 + d = a3) := 
sorry

end value_of_b_cannot_form_arithmetic_sequence_l222_22278


namespace sample_size_l222_22280

theorem sample_size {n : ℕ} (h_ratio : 2+3+4 = 9)
  (h_units_A : ∃ a : ℕ, a = 16)
  (h_stratified_sampling : ∃ B C : ℕ, B = 24 ∧ C = 32)
  : n = 16 + 24 + 32 := by
  sorry

end sample_size_l222_22280


namespace cost_of_items_l222_22295

variable (e t d : ℝ)

noncomputable def ques :=
  5 * e + 5 * t + 2 * d

axiom cond1 : 3 * e + 4 * t = 3.40
axiom cond2 : 4 * e + 3 * t = 4.00
axiom cond3 : 5 * e + 4 * t + 3 * d = 7.50

theorem cost_of_items : ques e t d = 6.93 :=
by
  sorry

end cost_of_items_l222_22295


namespace no_such_reals_exist_l222_22238

-- Define the existence of distinct real numbers such that the given condition holds
theorem no_such_reals_exist :
  ¬ ∃ x y z : ℝ, (x ≠ y) ∧ (y ≠ z) ∧ (z ≠ x) ∧ 
  (1 / (x^2 - y^2) + 1 / (y^2 - z^2) + 1 / (z^2 - x^2) = 0) :=
by
  -- Placeholder for proof
  sorry

end no_such_reals_exist_l222_22238


namespace total_money_divided_l222_22241

theorem total_money_divided (x y : ℕ) (hx : x = 1000) (ratioxy : 2 * y = 8 * x) : x + y = 5000 := 
by
  sorry

end total_money_divided_l222_22241


namespace royalty_amount_l222_22256

theorem royalty_amount (x : ℝ) (h1 : x > 800) (h2 : x ≤ 4000) (h3 : (x - 800) * 0.14 = 420) :
  x = 3800 :=
by
  sorry

end royalty_amount_l222_22256


namespace binomial_coefficients_sum_l222_22227

noncomputable def f (m n : ℕ) : ℕ :=
  (Nat.choose 6 m) * (Nat.choose 4 n)

theorem binomial_coefficients_sum : 
  f 3 0 + f 2 1 + f 1 2 + f 0 3 = 120 := 
by
  sorry

end binomial_coefficients_sum_l222_22227


namespace sequence_sum_consecutive_l222_22223

theorem sequence_sum_consecutive 
  (a : ℕ → ℕ) 
  (h1 : a 1 = 20) 
  (h8 : a 8 = 16) 
  (h_sum : ∀ i, 1 ≤ i ∧ i ≤ 6 → a i + a (i+1) + a (i+2) = 100) :
  a 2 = 16 ∧ a 3 = 64 ∧ a 4 = 20 ∧ a 5 = 16 ∧ a 6 = 64 ∧ a 7 = 20 :=
  sorry

end sequence_sum_consecutive_l222_22223


namespace ratio_value_l222_22266

theorem ratio_value (c d : ℝ) (h1 : c = 15 - 4 * d) (h2 : c / d = 4) : d = 15 / 8 :=
by sorry

end ratio_value_l222_22266


namespace bricks_in_wall_l222_22269

-- Definitions of conditions based on the problem statement
def time_first_bricklayer : ℝ := 12 
def time_second_bricklayer : ℝ := 15 
def reduced_productivity : ℝ := 12 
def combined_time : ℝ := 6
def total_bricks : ℝ := 720

-- Lean 4 statement of the proof problem
theorem bricks_in_wall (x : ℝ) 
  (h1 : (x / time_first_bricklayer + x / time_second_bricklayer - reduced_productivity) * combined_time = x) 
  : x = total_bricks := 
by {
  sorry
}

end bricks_in_wall_l222_22269


namespace even_fn_solution_set_l222_22255

theorem even_fn_solution_set (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x)) (h_f_def : ∀ x ≥ 0, f x = x^3 - 8) :
  { x | f (x - 2) > 0 } = { x | x < 0 ∨ x > 4 } :=
by sorry

end even_fn_solution_set_l222_22255


namespace focus_of_given_parabola_is_correct_l222_22201

-- Define the problem conditions
def parabolic_equation (x y : ℝ) : Prop := y = 4 * x^2

-- Define what it means for a point to be the focus of the given parabola
def is_focus_of_parabola (x0 y0 : ℝ) : Prop := 
    x0 = 0 ∧ y0 = 1 / 16

-- Define the theorem to be proven
theorem focus_of_given_parabola_is_correct : 
  ∃ x0 y0, parabolic_equation x0 y0 ∧ is_focus_of_parabola x0 y0 :=
sorry

end focus_of_given_parabola_is_correct_l222_22201


namespace total_respondents_l222_22234

theorem total_respondents (X Y : ℕ) (h1 : X = 60) (h2 : 3 * Y = X) : X + Y = 80 :=
by
  sorry

end total_respondents_l222_22234


namespace joe_total_paint_used_l222_22264

-- Conditions
def initial_paint : ℕ := 360
def paint_first_week : ℕ := initial_paint * 1 / 4
def remaining_paint_after_first_week : ℕ := initial_paint - paint_first_week
def paint_second_week : ℕ := remaining_paint_after_first_week * 1 / 6

-- Theorem statement
theorem joe_total_paint_used : paint_first_week + paint_second_week = 135 := by
  sorry

end joe_total_paint_used_l222_22264


namespace possible_values_of_5x_plus_2_l222_22209

theorem possible_values_of_5x_plus_2 (x : ℝ) :
  (x - 4) * (5 * x + 2) = 0 →
  (5 * x + 2 = 0 ∨ 5 * x + 2 = 22) :=
by
  intro h
  sorry

end possible_values_of_5x_plus_2_l222_22209


namespace minimum_value_2a_plus_b_l222_22287

theorem minimum_value_2a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : (1 / (a + 1)) + (2 / (b - 2)) = 1 / 2) : 2 * a + b ≥ 16 := 
sorry

end minimum_value_2a_plus_b_l222_22287


namespace cubic_polynomial_inequality_l222_22249

theorem cubic_polynomial_inequality
  (A B C : ℝ)
  (h : ∃ (a b c : ℝ), 
    0 < a ∧ 0 < b ∧ 0 < c ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
    A = -(a + b + c) ∧ B = ab + bc + ca ∧ C = -abc) :
  A^2 + B^2 + 18 * C > 0 :=
by
  sorry

end cubic_polynomial_inequality_l222_22249


namespace percentage_chromium_first_alloy_l222_22220

theorem percentage_chromium_first_alloy 
  (x : ℝ) (w1 w2 : ℝ) (p2 p_new : ℝ) 
  (h1 : w1 = 10) 
  (h2 : w2 = 30) 
  (h3 : p2 = 0.08)
  (h4 : p_new = 0.09):
  ((x / 100) * w1 + p2 * w2) = p_new * (w1 + w2) → x = 12 :=
by
  sorry

end percentage_chromium_first_alloy_l222_22220


namespace roots_of_polynomial_l222_22222

theorem roots_of_polynomial :
  (x^2 - 5 * x + 6) * (x - 1) * (x + 3) = 0 ↔ (x = -3 ∨ x = 1 ∨ x = 2 ∨ x = 3) :=
by {
  sorry
}

end roots_of_polynomial_l222_22222


namespace grade_assignment_ways_l222_22206

-- Definitions
def num_students : ℕ := 10
def num_choices_per_student : ℕ := 3

-- Theorem statement
theorem grade_assignment_ways : num_choices_per_student ^ num_students = 59049 := by
  sorry

end grade_assignment_ways_l222_22206


namespace correct_propositions_l222_22247

-- Definitions of the conditions in the Math problem

variable (triangle_outside_plane : Prop)
variable (triangle_side_intersections_collinear : Prop)
variable (parallel_lines_coplanar : Prop)
variable (noncoplanar_points_planes : Prop)

-- Math proof problem statement
theorem correct_propositions :
  (triangle_outside_plane ∧ 
   parallel_lines_coplanar ∧ 
   ¬noncoplanar_points_planes) →
  2 = 2 :=
by
  sorry

end correct_propositions_l222_22247


namespace lcm_12_18_l222_22204

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l222_22204


namespace fraction_identity_l222_22219

variable (x y z : ℝ)

theorem fraction_identity (h : (x / (y + z)) + (y / (z + x)) + (z / (x + y)) = 1) :
  (x^2 / (y + z)) + (y^2 / (z + x)) + (z^2 / (x + y)) = 0 :=
  sorry

end fraction_identity_l222_22219


namespace equation_holds_l222_22257

variable (a b : ℝ)

theorem equation_holds : a^2 - b^2 - (-2 * b^2) = a^2 + b^2 :=
by sorry

end equation_holds_l222_22257


namespace pipe_tank_overflow_l222_22248

theorem pipe_tank_overflow (t : ℕ) :
  let rateA := 1 / 30
  let rateB := 1 / 60
  let combined_rate := rateA + rateB
  let workA := rateA * (t - 15)
  let workB := rateB * t
  (workA + workB = 1) ↔ (t = 25) := by
  sorry

end pipe_tank_overflow_l222_22248


namespace inequality_solution_set_l222_22293

theorem inequality_solution_set {f : ℝ → ℝ} (h_odd : ∀ x, f (-x) = -f x)
  (h_inc : ∀ {x y : ℝ}, 0 < x → x < y → f x ≤ f y)
  (h_value : f 1 = 0) :
  {x | (f x - f (-x)) / x ≤ 0} = {x | -1 ≤ x ∧ x < 0} ∪ {x | 0 < x ∧ x ≤ 1} :=
by
  sorry


end inequality_solution_set_l222_22293


namespace cannot_determine_a_l222_22281

theorem cannot_determine_a 
  (n : ℝ) 
  (p : ℝ) 
  (a : ℝ) 
  (line_eq : ∀ (x y : ℝ), x = 5 * y + 5) 
  (pt1 : a = 5 * n + 5) 
  (pt2 : a + 2 = 5 * (n + p) + 5) : p = 0.4 → ¬∀ a' : ℝ, a = a' :=
by
  sorry

end cannot_determine_a_l222_22281


namespace problem_l222_22224

noncomputable def f (x : ℝ) : ℝ := sorry

theorem problem 
  (h_odd : ∀ x : ℝ, f (-x) = -f x) 
  (h_periodic : ∀ x : ℝ, f (x + 1) = f (1 - x)) 
  (h_interval : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = 2 ^ x - 1) 
  : f 2019 = -1 := 
sorry

end problem_l222_22224


namespace detour_distance_l222_22254

-- Definitions based on conditions:
def D_black : ℕ := sorry -- The original distance along the black route
def D_black_C : ℕ := sorry -- The distance from C to B along the black route
def D_red : ℕ := sorry -- The distance from C to B along the red route

-- Extra distance due to detour calculation
def D_extra := D_red - D_black_C

-- Prove that the extra distance is 14 km
theorem detour_distance : D_extra = 14 := by
  sorry

end detour_distance_l222_22254


namespace branches_on_main_stem_l222_22243

theorem branches_on_main_stem (x : ℕ) (h : 1 + x + x^2 = 57) : x = 7 :=
  sorry

end branches_on_main_stem_l222_22243


namespace inverse_proportion_function_l222_22272

theorem inverse_proportion_function (f : ℝ → ℝ) (h : ∀ x, f x = 1/x) : f 1 = 1 := 
by
  sorry

end inverse_proportion_function_l222_22272


namespace simplify_and_evaluate_l222_22276

theorem simplify_and_evaluate (m : ℝ) (h_root : m^2 + 3 * m - 2 = 0) :
  (m - 3) / (3 * m^2 - 6 * m) / (m + 2 - 5 / (m - 2)) = 1 / 6 :=
by
  sorry

end simplify_and_evaluate_l222_22276


namespace find_x_l222_22230

theorem find_x (x y : ℤ) (h1 : x + y = 4) (h2 : x - y = 36) : x = 20 :=
by
  sorry

end find_x_l222_22230


namespace equidistant_points_l222_22296

theorem equidistant_points (r d1 d2 : ℝ) (d1_eq : d1 = r) (d2_eq : d2 = 6) : 
  ∃ p : ℝ, p = 2 := 
sorry

end equidistant_points_l222_22296


namespace seashells_count_l222_22259

theorem seashells_count (total_seashells broken_seashells : ℕ) (h_total : total_seashells = 7) (h_broken : broken_seashells = 4) : total_seashells - broken_seashells = 3 := by
  sorry

end seashells_count_l222_22259


namespace intersection_point_l222_22286

def line_parametric (t : ℝ) : ℝ × ℝ × ℝ :=
  (3 + 2 * t, -1 + 3 * t, -3 + 2 * t)

def on_plane (x y z : ℝ) : Prop :=
  3 * x + 4 * y + 7 * z - 16 = 0

theorem intersection_point : ∃ t, line_parametric t = (5, 2, -1) ∧ on_plane 5 2 (-1) :=
by
  use 1
  sorry

end intersection_point_l222_22286


namespace quadratic_function_vertex_form_l222_22225

theorem quadratic_function_vertex_form :
  ∃ f : ℝ → ℝ, (∀ x, f x = (x - 2)^2 - 2) ∧ (f 0 = 2) ∧ (∀ x, f x = a * (x - 2)^2 - 2 → a = 1) := by
  sorry

end quadratic_function_vertex_form_l222_22225


namespace primes_square_condition_l222_22277

open Nat

theorem primes_square_condition (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) : 
  ∃ n : ℕ, p^(q+1) + q^(p+1) = n^2 ↔ p = 2 ∧ q = 2 := by
  sorry

end primes_square_condition_l222_22277


namespace fraction_equality_l222_22229

def at_op (a b : ℕ) : ℕ := a * b - b^2 + b^3
def hash_op (a b : ℕ) : ℕ := a + b - a * b^2 + a * b^3

theorem fraction_equality : 
  ∀ (a b : ℕ), a = 7 → b = 3 → (at_op a b : ℚ) / (hash_op a b : ℚ) = 39 / 136 :=
by
  intros a b h_a h_b
  rw [h_a, h_b]
  sorry

end fraction_equality_l222_22229


namespace find_vanessa_age_l222_22205

/-- Define the initial conditions and goal -/
theorem find_vanessa_age (V : ℕ) (Kevin_age current_time future_time : ℕ) :
  Kevin_age = 16 ∧ future_time = current_time + 5 ∧
  (Kevin_age + future_time - current_time) = 3 * (V + future_time - current_time) →
  V = 2 := 
by
  sorry

end find_vanessa_age_l222_22205


namespace max_distance_from_earth_to_sun_l222_22240

-- Assume the semi-major axis 'a' and semi-minor axis 'b' specified in the problem.
def semi_major_axis : ℝ := 1.5 * 10^8
def semi_minor_axis : ℝ := 3 * 10^6

-- Define the theorem stating the maximum distance from the Earth to the Sun.
theorem max_distance_from_earth_to_sun :
  let a := semi_major_axis
  let b := semi_minor_axis
  a + b = 1.53 * 10^8 :=
by
  -- Proof will be completed
  sorry

end max_distance_from_earth_to_sun_l222_22240


namespace cups_of_baking_mix_planned_l222_22200

-- Definitions
def butter_per_cup := 2 -- 2 ounces of butter per 1 cup of baking mix
def coconut_oil_per_butter := 2 -- 2 ounces of coconut oil can substitute 2 ounces of butter
def butter_remaining := 4 -- Chef had 4 ounces of butter
def coconut_oil_used := 8 -- Chef used 8 ounces of coconut oil

-- Statement to be proven
theorem cups_of_baking_mix_planned : 
  (butter_remaining / butter_per_cup) + (coconut_oil_used / coconut_oil_per_butter) = 6 := 
by 
  sorry

end cups_of_baking_mix_planned_l222_22200


namespace mrs_hilt_total_distance_l222_22260

-- Define the distances and number of trips
def distance_to_water_fountain := 30
def distance_to_staff_lounge := 45
def trips_to_water_fountain := 4
def trips_to_staff_lounge := 3

-- Calculate the total distance for Mrs. Hilt's trips
def total_distance := (distance_to_water_fountain * 2 * trips_to_water_fountain) + 
                      (distance_to_staff_lounge * 2 * trips_to_staff_lounge)
                      
theorem mrs_hilt_total_distance : total_distance = 510 := 
by
  sorry

end mrs_hilt_total_distance_l222_22260


namespace ratio_distance_l222_22251

theorem ratio_distance
  (x : ℝ)
  (P : ℝ × ℝ)
  (hP_coords : P = (x, -9))
  (h_distance_y_axis : abs x = 18) :
  abs (-9) / abs x = 1 / 2 :=
by sorry

end ratio_distance_l222_22251


namespace compute_expression_l222_22258

theorem compute_expression : 2 + 4 * 3^2 - 1 + 7 * 2 / 2 = 44 := by
  sorry

end compute_expression_l222_22258


namespace opposite_face_is_D_l222_22292

-- Define the six faces
inductive Face
| A | B | C | D | E | F

open Face

-- Define the adjacency relation
def is_adjacent (x y : Face) : Prop :=
(y = B ∧ x = A) ∨ (y = F ∧ x = A) ∨ (y = C ∧ x = A) ∨ (y = E ∧ x = A)

-- Define the problem statement in Lean
theorem opposite_face_is_D : 
  (∀ (x : Face), is_adjacent A x ↔ x = B ∨ x = F ∨ x = C ∨ x = E) →
  (¬ (is_adjacent A D)) →
  True :=
by
  intro adj_relation non_adj_relation
  sorry

end opposite_face_is_D_l222_22292


namespace repeating_decimals_sum_l222_22215

theorem repeating_decimals_sum : 
  (0.3333333333333333 : ℝ) + (0.0404040404040404 : ℝ) + (0.005005005005005 : ℝ) = (14 / 37 : ℝ) :=
by {
  sorry
}

end repeating_decimals_sum_l222_22215


namespace hyperbola_eqn_l222_22226

theorem hyperbola_eqn
  (P : ℝ × ℝ) (Q : ℝ × ℝ)
  (C1 : P = (-3, 2 * Real.sqrt 7))
  (C2 : Q = (-6 * Real.sqrt 2, -7))
  (asymptote_hyperbola : ∀ x y : ℝ, x^2 / 4 - y^2 / 3 = 1)
  (special_point : ℝ × ℝ)
  (C3 : special_point = (2, 2 * Real.sqrt 3)) :
  ∃ (a b : ℝ), ¬(a = 0) ∧ ¬(b = 0) ∧ 
  (∀ x y : ℝ, (y^2 / b - x^2 / a = 1 → 
    ((y^2 / 25 - x^2 / 75 = 1) ∨ 
    (y^2 / 9 - x^2 / 12 = 1)))) :=
by
  sorry

end hyperbola_eqn_l222_22226


namespace max_value_2xy_sqrt6_8yz2_l222_22252

theorem max_value_2xy_sqrt6_8yz2 (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (h : x^2 + y^2 + z^2 = 1) : 
  2 * x * y * Real.sqrt 6 + 8 * y * z^2 ≤ Real.sqrt 6 :=
sorry

end max_value_2xy_sqrt6_8yz2_l222_22252


namespace find_f_l222_22261

theorem find_f (f : ℝ → ℝ) (h₁ : ∀ x y : ℝ, 0 < x → 0 < y → x ≤ y → f x ≤ f y)
  (h₂ : ∀ x : ℝ, 0 < x → f (x ^ 4) + f (x ^ 2) + f x + f 1 = x ^ 4 + x ^ 2 + x + 1) :
  ∀ x : ℝ, 0 < x → f x = x := 
sorry

end find_f_l222_22261


namespace mikko_should_attempt_least_questions_l222_22202

theorem mikko_should_attempt_least_questions (p : ℝ) (h_p : 0 < p ∧ p < 1) : 
  ∃ (x : ℕ), x ≥ ⌈1 / (2 * p - 1)⌉ :=
by
  sorry

end mikko_should_attempt_least_questions_l222_22202


namespace quadratic_function_points_l222_22290

theorem quadratic_function_points (a c y1 y2 y3 y4 : ℝ) (h_a : a < 0)
    (h_A : y1 = a * (-2)^2 - 4 * a * (-2) + c)
    (h_B : y2 = a * 0^2 - 4 * a * 0 + c)
    (h_C : y3 = a * 3^2 - 4 * a * 3 + c)
    (h_D : y4 = a * 5^2 - 4 * a * 5 + c)
    (h_condition : y2 * y4 < 0) : y1 * y3 < 0 :=
by
  sorry

end quadratic_function_points_l222_22290


namespace scientific_notation_of_number_l222_22245

theorem scientific_notation_of_number :
  ∃ (a : ℝ) (n : ℤ), 0.00000002 = a * 10^n ∧ a = 2 ∧ n = -8 :=
by
  sorry

end scientific_notation_of_number_l222_22245


namespace bounded_infinite_sequence_l222_22288

noncomputable def sequence_x (n : ℕ) : ℝ :=
  4 * (Real.sqrt 2 * n - ⌊Real.sqrt 2 * n⌋)

theorem bounded_infinite_sequence (a : ℝ) (h : a > 1) :
  ∀ i j : ℕ, i ≠ j → (|sequence_x i - sequence_x j| * |(i - j : ℝ)|^a) ≥ 1 := 
by
  intros i j h_ij
  sorry

end bounded_infinite_sequence_l222_22288


namespace g_inv_f_7_l222_22236

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry
noncomputable def g_inv : ℝ → ℝ := sorry

axiom f_inv_g (x : ℝ) : f_inv (g x) = x^3 - 1
axiom g_exists_inv : ∀ y : ℝ, ∃ x : ℝ, g x = y

theorem g_inv_f_7 : g_inv (f 7) = 2 :=
by
  sorry

end g_inv_f_7_l222_22236


namespace tan_diff_l222_22262

theorem tan_diff (x y : ℝ) (hx : Real.tan x = 3) (hy : Real.tan y = 2) : 
  Real.tan (x - y) = 1 / 7 := 
by 
  sorry

end tan_diff_l222_22262


namespace initial_tickets_l222_22267

theorem initial_tickets (tickets_sold_week1 : ℕ) (tickets_sold_week2 : ℕ) (tickets_left : ℕ) 
  (h1 : tickets_sold_week1 = 38) (h2 : tickets_sold_week2 = 17) (h3 : tickets_left = 35) : 
  tickets_sold_week1 + tickets_sold_week2 + tickets_left = 90 :=
by 
  sorry

end initial_tickets_l222_22267


namespace am_gm_hm_inequality_l222_22228

variable {x y : ℝ}

-- Conditions: x and y are positive real numbers and x < y
def conditions (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∧ x < y

-- Proof statement: A.M. > G.M. > H.M. under given conditions
theorem am_gm_hm_inequality (x y : ℝ) (h : conditions x y) :
  (x + y) / 2 > Real.sqrt (x * y) ∧ Real.sqrt (x * y) > (2 * x * y) / (x + y) :=
sorry

end am_gm_hm_inequality_l222_22228


namespace factorize_expression_l222_22282

variable {R : Type} [CommRing R] (m a : R)

theorem factorize_expression : m * a^2 - m = m * (a + 1) * (a - 1) :=
by
  sorry

end factorize_expression_l222_22282


namespace ratio_xz_y2_l222_22242

-- Define the system of equations
def system (k x y z : ℝ) : Prop := 
  x + k * y + 4 * z = 0 ∧ 
  4 * x + k * y - 3 * z = 0 ∧ 
  3 * x + 5 * y - 4 * z = 0

-- Our main theorem to prove the value of xz / y^2 given the system with k = 7.923
theorem ratio_xz_y2 (x y z : ℝ) (h : system 7.923 x y z) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) : 
  ∃ r : ℝ, r = (x * z) / (y ^ 2) :=
sorry

end ratio_xz_y2_l222_22242
