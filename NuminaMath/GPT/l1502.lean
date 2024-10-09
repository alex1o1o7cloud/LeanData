import Mathlib

namespace polynomial_multiplication_l1502_150256

theorem polynomial_multiplication (x z : ℝ) :
  (3*x^5 - 7*z^3) * (9*x^10 + 21*x^5*z^3 + 49*z^6) = 27*x^15 - 343*z^9 :=
by
  sorry

end polynomial_multiplication_l1502_150256


namespace number_of_apartment_complexes_l1502_150276

theorem number_of_apartment_complexes (width_land length_land side_complex : ℕ)
    (h_width : width_land = 262) (h_length : length_land = 185) 
    (h_side : side_complex = 18) :
    width_land / side_complex * length_land / side_complex = 140 := by
  -- given conditions
  rw [h_width, h_length, h_side]
  -- apply calculation steps for clarity (not necessary for final theorem)
  -- calculate number of complexes along width
  have h1 : 262 / 18 = 14 := sorry
  -- calculate number of complexes along length
  have h2 : 185 / 18 = 10 := sorry
  -- final product calculation
  sorry

end number_of_apartment_complexes_l1502_150276


namespace fish_count_l1502_150211

theorem fish_count (initial_fish : ℝ) (bought_fish : ℝ) (total_fish : ℝ) 
  (h1 : initial_fish = 212.0) 
  (h2 : bought_fish = 280.0) 
  (h3 : total_fish = initial_fish + bought_fish) : 
  total_fish = 492.0 := 
by 
  sorry

end fish_count_l1502_150211


namespace instantaneous_acceleration_at_1_second_l1502_150203

-- Assume the velocity function v(t) is given as:
def v (t : ℝ) : ℝ := t^2 + 2 * t + 3

-- We need to prove that the instantaneous acceleration at t = 1 second is 4 m/s^2.
theorem instantaneous_acceleration_at_1_second : 
  deriv v 1 = 4 :=
by 
  sorry

end instantaneous_acceleration_at_1_second_l1502_150203


namespace time_on_sideline_l1502_150208

def total_game_time : ℕ := 90
def time_mark_played_first_period : ℕ := 20
def time_mark_played_second_period : ℕ := 35
def total_time_mark_played : ℕ := time_mark_played_first_period + time_mark_played_second_period

theorem time_on_sideline : total_game_time - total_time_mark_played = 35 := by
  sorry

end time_on_sideline_l1502_150208


namespace small_poster_ratio_l1502_150277

theorem small_poster_ratio (total_posters : ℕ) (medium_posters large_posters small_posters : ℕ)
  (h1 : total_posters = 50)
  (h2 : medium_posters = 50 / 2)
  (h3 : large_posters = 5)
  (h4 : small_posters = total_posters - medium_posters - large_posters)
  (h5 : total_posters ≠ 0) :
  small_posters = 20 ∧ (small_posters : ℚ) / total_posters = 2 / 5 := 
sorry

end small_poster_ratio_l1502_150277


namespace scientific_notation_of_1040000000_l1502_150243

theorem scientific_notation_of_1040000000 : (1.04 * 10^9 = 1040000000) :=
by
  -- Math proof steps can be added here
  sorry

end scientific_notation_of_1040000000_l1502_150243


namespace period_of_f_l1502_150270

noncomputable def f : ℝ → ℝ := sorry

def functional_equation (f : ℝ → ℝ) := ∀ x y : ℝ, f (2 * x) + f (2 * y) = f (x + y) * f (x - y)

def f_pi_zero (f : ℝ → ℝ) := f (Real.pi) = 0

def f_not_identically_zero (f : ℝ → ℝ) := ∃ x : ℝ, f x ≠ 0

theorem period_of_f (f : ℝ → ℝ)
  (hf_eq : functional_equation f)
  (hf_pi_zero : f_pi_zero f)
  (hf_not_zero : f_not_identically_zero f) : 
  ∀ x : ℝ, f (x + 4 * Real.pi) = f x := sorry

end period_of_f_l1502_150270


namespace find_f_log2_5_l1502_150251

variable {f g : ℝ → ℝ}

-- f is an even function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- g is an odd function
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- Given conditions
axiom f_even : is_even f
axiom g_odd : is_odd g
axiom f_g_equation : ∀ x, f x + g x = (2:ℝ)^x + x

-- Proof goal: Compute f(log_2 5) and show it equals 13/5
theorem find_f_log2_5 : f (Real.log 5 / Real.log 2) = (13:ℝ) / 5 := by
  sorry

end find_f_log2_5_l1502_150251


namespace number_of_solutions_l1502_150229

-- Define the equation and the constraints
def equation (x y z : ℕ) : Prop := 2 * x + 3 * y + z = 800

def positive_integer (n : ℕ) : Prop := n > 0

-- The main theorem statement
theorem number_of_solutions : ∃ s, s = 127 ∧ ∀ (x y z : ℕ), positive_integer x → positive_integer y → positive_integer z → equation x y z → s = 127 :=
by
  sorry

end number_of_solutions_l1502_150229


namespace parallel_vectors_sum_is_six_l1502_150253

theorem parallel_vectors_sum_is_six (x y : ℝ) :
  let a := (4, -1, 1)
  let b := (x, y, 2)
  (x / 4 = 2) ∧ (y / -1 = 2) →
  x + y = 6 :=
by
  intros
  sorry

end parallel_vectors_sum_is_six_l1502_150253


namespace triangle_is_obtuse_l1502_150242

theorem triangle_is_obtuse
  (A B C : ℝ)
  (h1 : 3 * A > 5 * B)
  (h2 : 3 * C < 2 * B)
  (h3 : A + B + C = 180) :
  A > 90 :=
sorry

end triangle_is_obtuse_l1502_150242


namespace constant_sequence_from_conditions_l1502_150266

variable (k b : ℝ) [Nontrivial ℝ]
variable (a_n : ℕ → ℝ)

-- Define the conditions function
def cond1 (k b : ℝ) (a_n : ℕ → ℝ) : Prop :=
  (∃ (q : ℝ), ∀ n, a_n (n + 1) = q * a_n n) ∧ 
  (∃ (m : ℝ), ∀ n, k * a_n (n + 1) + b = m * (k * a_n n + b))

def cond2 (k b : ℝ) (a_n : ℕ → ℝ) : Prop :=
  (∃ (d : ℝ), ∀ n, a_n (n + 1) = a_n n + d) ∧ 
  (∃ (m : ℝ), ∀ n, k * a_n (n + 1) + b = m * (k * a_n n + b))

def cond3 (k b : ℝ) (a_n : ℕ → ℝ) : Prop :=
  (∃ (q : ℝ), ∀ n, a_n (n + 1) = q * a_n n) ∧ 
  (∃ (m : ℝ), ∀ n, k * a_n (n + 1) + b - (k * a_n n + b) = m)

-- Lean statement to prove the problem
theorem constant_sequence_from_conditions (k b : ℝ) [Nontrivial ℝ] (a_n : ℕ → ℝ) :
  (cond1 k b a_n ∨ cond2 k b a_n ∨ cond3 k b a_n) → 
  ∃ c : ℝ, ∀ n, a_n n = c :=
by
  -- To be proven
  intros
  sorry

end constant_sequence_from_conditions_l1502_150266


namespace find_k_l1502_150220

-- Define the problem statement
theorem find_k (d : ℝ) (x : ℝ)
  (h_ratio : 3 * x / (5 * x) = 3 / 5)
  (h_diag : (10 * d)^2 = (3 * x)^2 + (5 * x)^2) :
  ∃ k : ℝ, (3 * x) * (5 * x) = k * d^2 ∧ k = 750 / 17 := by
  sorry

end find_k_l1502_150220


namespace roots_squared_sum_l1502_150230

theorem roots_squared_sum (p q r : ℂ) (h : ∀ x : ℂ, 3 * x ^ 3 - 3 * x ^ 2 + 6 * x - 9 = 0 → x = p ∨ x = q ∨ x = r) :
  p^2 + q^2 + r^2 = -3 :=
by
  sorry

end roots_squared_sum_l1502_150230


namespace jake_not_drop_coffee_l1502_150206

theorem jake_not_drop_coffee :
  let p_trip := 0.40
  let p_drop_trip := 0.25
  let p_step := 0.30
  let p_drop_step := 0.20
  let p_no_drop_trip := 1 - (p_trip * p_drop_trip)
  let p_no_drop_step := 1 - (p_step * p_drop_step)
  (p_no_drop_trip * p_no_drop_step) = 0.846 :=
by
  sorry

end jake_not_drop_coffee_l1502_150206


namespace solve_quadratic_and_cubic_eqns_l1502_150254

-- Define the conditions as predicates
def eq1 (x : ℝ) : Prop := (x - 1)^2 = 4
def eq2 (x : ℝ) : Prop := (x - 2)^3 = -125

-- State the theorem
theorem solve_quadratic_and_cubic_eqns : 
  (∃ x : ℝ, eq1 x ∧ (x = 3 ∨ x = -1)) ∧ (∃ x : ℝ, eq2 x ∧ x = -3) :=
by
  sorry

end solve_quadratic_and_cubic_eqns_l1502_150254


namespace arithmetic_sequence_n_l1502_150246

theorem arithmetic_sequence_n {a : ℕ → ℕ} (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = a n + 3) :
  (∃ n : ℕ, a n = 2005) → (∃ n : ℕ, n = 669) :=
by
  sorry

end arithmetic_sequence_n_l1502_150246


namespace tan_2x_value_l1502_150209

noncomputable def f (x : ℝ) := Real.sin x + Real.cos x
noncomputable def f' (x : ℝ) := deriv f x

theorem tan_2x_value (x : ℝ) (h : f' x = 3 * f x) : Real.tan (2 * x) = (4/3) := by
  sorry

end tan_2x_value_l1502_150209


namespace complement_of_A_in_U_l1502_150233

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≥ 1} ∪ {x | x ≤ 0}
def C_UA : Set ℝ := U \ A

theorem complement_of_A_in_U :
  C_UA = {x | 0 < x ∧ x < 1} :=
sorry

end complement_of_A_in_U_l1502_150233


namespace contractor_laborers_l1502_150259

theorem contractor_laborers (x : ℕ) (h1 : 15 * x = 20 * (x - 5)) : x = 20 :=
by sorry

end contractor_laborers_l1502_150259


namespace solution_set_inequality_l1502_150257

theorem solution_set_inequality (m : ℝ) (x : ℝ) 
  (h : 3 - m < 0) : (2 - m) * x + m > 2 ↔ x < 1 :=
by
  sorry

end solution_set_inequality_l1502_150257


namespace meaningful_sqrt_neg_x_squared_l1502_150292

theorem meaningful_sqrt_neg_x_squared (x : ℝ) : (x = 0) ↔ (-(x^2) ≥ 0) :=
by
  sorry

end meaningful_sqrt_neg_x_squared_l1502_150292


namespace chessboard_edge_count_l1502_150228

theorem chessboard_edge_count (n : ℕ) 
  (border_white : ∀ (c : ℕ), c ∈ (Finset.range (4 * (n - 1))) → (∃ w : ℕ, w ≥ n)) 
  (border_black : ∀ (c : ℕ), c ∈ (Finset.range (4 * (n - 1))) → (∃ b : ℕ, b ≥ n)) :
  ∃ e : ℕ, e ≥ n :=
sorry

end chessboard_edge_count_l1502_150228


namespace rectangle_area_l1502_150295

theorem rectangle_area (W : ℕ) (hW : W = 5) (L : ℕ) (hL : L = 4 * W) : ∃ (A : ℕ), A = L * W ∧ A = 100 := 
by
  use 100
  sorry

end rectangle_area_l1502_150295


namespace find_range_of_m_l1502_150293

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + m * x + 1 = 0 ∧ y^2 + m * y + 1 = 0
def q (m : ℝ) : Prop := 1 < m ∧ m < 3

theorem find_range_of_m (m : ℝ) (h1 : ¬(p m ∧ q m)) (h2 : ¬¬p m) : m ≥ 3 ∨ m < -2 :=
by 
  sorry

end find_range_of_m_l1502_150293


namespace min_sum_of_squares_l1502_150280

theorem min_sum_of_squares (y1 y2 y3 : ℝ) (h1 : y1 > 0) (h2 : y2 > 0) (h3 : y3 > 0) (h4 : y1 + 3 * y2 + 4 * y3 = 72) : 
  y1^2 + y2^2 + y3^2 ≥ 2592 / 13 ∧ (∃ k, y1 = k ∧ y2 = 3 * k ∧ y3 = 4 * k ∧ k = 36 / 13) :=
sorry

end min_sum_of_squares_l1502_150280


namespace hall_volume_l1502_150202

theorem hall_volume (l w : ℕ) (h : ℕ) 
    (cond1 : l = 18)
    (cond2 : w = 9)
    (cond3 : (2 * l * w) = (2 * l * h + 2 * w * h)) : 
    (l * w * h = 972) :=
by
  rw [cond1, cond2] at cond3
  have h_eq : h = 324 / 54 := sorry
  rw [h_eq]
  norm_num
  sorry

end hall_volume_l1502_150202


namespace joe_total_spending_at_fair_l1502_150213

-- Definitions based on conditions
def entrance_fee (age : ℕ) : ℝ := if age < 18 then 5 else 6
def ride_cost (rides : ℕ) : ℝ := rides * 0.5

-- Given conditions
def joe_age := 19
def twin_age := 6

def total_cost (joe_age : ℕ) (twin_age : ℕ) (rides_per_person : ℕ) :=
  entrance_fee joe_age + 2 * entrance_fee twin_age + 3 * ride_cost rides_per_person

-- The main statement to be proven
theorem joe_total_spending_at_fair : total_cost joe_age twin_age 3 = 20.5 :=
by
  sorry

end joe_total_spending_at_fair_l1502_150213


namespace ratio_of_x_y_l1502_150262

theorem ratio_of_x_y (x y : ℝ) (h₁ : 3 < (x - y) / (x + y)) (h₂ : (x - y) / (x + y) < 4) (h₃ : ∃ a b : ℤ, x = a * y / b ) (h₄ : x + y = 10) :
  x / y = -2 := sorry

end ratio_of_x_y_l1502_150262


namespace total_dots_correct_l1502_150231

/-- Define the initial conditions -/
def monday_ladybugs : ℕ := 8
def monday_dots_per_ladybug : ℕ := 6
def tuesday_ladybugs : ℕ := 5
def wednesday_ladybugs : ℕ := 4

/-- Define the derived conditions -/
def tuesday_dots_per_ladybug : ℕ := monday_dots_per_ladybug - 1
def wednesday_dots_per_ladybug : ℕ := monday_dots_per_ladybug - 2

/-- Calculate the total number of dots -/
def monday_total_dots : ℕ := monday_ladybugs * monday_dots_per_ladybug
def tuesday_total_dots : ℕ := tuesday_ladybugs * tuesday_dots_per_ladybug
def wednesday_total_dots : ℕ := wednesday_ladybugs * wednesday_dots_per_ladybug
def total_dots : ℕ := monday_total_dots + tuesday_total_dots + wednesday_total_dots

/-- Prove the total dots equal to 89 -/
theorem total_dots_correct : total_dots = 89 := by
  sorry

end total_dots_correct_l1502_150231


namespace card_area_l1502_150287

theorem card_area (length width : ℕ) (h_length : length = 5) (h_width : width = 7)
  (h_area_after_shortening : (length - 1) * width = 24 ∨ length * (width - 1) = 24) :
  length * (width - 1) = 18 :=
by
  sorry

end card_area_l1502_150287


namespace kareem_has_largest_final_number_l1502_150289

def jose_final : ℕ := (15 - 2) * 4 + 5
def thuy_final : ℕ := (15 * 3 - 3) - 4
def kareem_final : ℕ := ((20 - 3) + 4) * 3

theorem kareem_has_largest_final_number :
  kareem_final > jose_final ∧ kareem_final > thuy_final := 
by 
  sorry

end kareem_has_largest_final_number_l1502_150289


namespace max_projection_sum_l1502_150299

-- Define the given conditions
def edge_length : ℝ := 2

def projection_front_view (length : ℝ) : Prop := length = edge_length
def projection_side_view (length : ℝ) : Prop := ∃ a : ℝ, a = length
def projection_top_view (length : ℝ) : Prop := ∃ b : ℝ, b = length

-- State the theorem
theorem max_projection_sum (a b : ℝ) (ha : projection_side_view a) (hb : projection_top_view b) :
  a + b ≤ 4 := sorry

end max_projection_sum_l1502_150299


namespace root_equation_l1502_150275

theorem root_equation (p q : ℝ) (hp : 3 * p^2 - 5 * p - 7 = 0)
                                  (hq : 3 * q^2 - 5 * q - 7 = 0) :
            (3 * p^2 - 3 * q^2) * (p - q)⁻¹ = 5 := 
by sorry

end root_equation_l1502_150275


namespace germs_per_dish_calc_l1502_150207

theorem germs_per_dish_calc :
    let total_germs := 0.036 * 10^5
    let total_dishes := 36000 * 10^(-3)
    (total_germs / total_dishes) = 100 := by
    sorry

end germs_per_dish_calc_l1502_150207


namespace percent_increase_equilateral_triangles_l1502_150219

noncomputable def side_length (n : ℕ) : ℕ :=
  if n = 0 then 3 else 2 ^ n * 3

noncomputable def perimeter (n : ℕ) : ℕ :=
  3 * side_length n

noncomputable def percent_increase (initial : ℕ) (final : ℕ) : ℚ := 
  ((final - initial) / initial) * 100

theorem percent_increase_equilateral_triangles :
  percent_increase (perimeter 0) (perimeter 4) = 1500 := by
  sorry

end percent_increase_equilateral_triangles_l1502_150219


namespace range_of_a_l1502_150245

section
variables (a : ℝ)
def p : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (h : p a ∧ q a) : a = 1 ∨ a ≤ -2 :=
sorry
end

end range_of_a_l1502_150245


namespace quotient_of_division_l1502_150225

theorem quotient_of_division:
  ∀ (n d r q : ℕ), n = 165 → d = 18 → r = 3 → q = (n - r) / d → q = 9 :=
by sorry

end quotient_of_division_l1502_150225


namespace annual_average_growth_rate_l1502_150269

theorem annual_average_growth_rate (x : ℝ) (h : x > 0): 
  100 * (1 + x)^2 = 169 :=
sorry

end annual_average_growth_rate_l1502_150269


namespace range_xf_ge_0_l1502_150263

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -x - 2 else - (-x) - 2

theorem range_xf_ge_0 :
  { x : ℝ | x * f x ≥ 0 } = { x : ℝ | -2 ≤ x ∧ x ≤ 2 } :=
by
  sorry

end range_xf_ge_0_l1502_150263


namespace marked_price_correct_l1502_150216

theorem marked_price_correct
    (initial_price : ℝ)
    (initial_discount_rate : ℝ)
    (profit_margin_rate : ℝ)
    (final_discount_rate : ℝ)
    (purchase_price : ℝ)
    (final_selling_price : ℝ)
    (marked_price : ℝ)
    (h_initial_price : initial_price = 30)
    (h_initial_discount_rate : initial_discount_rate = 0.15)
    (h_profit_margin_rate : profit_margin_rate = 0.20)
    (h_final_discount_rate : final_discount_rate = 0.25)
    (h_purchase_price : purchase_price = initial_price * (1 - initial_discount_rate))
    (h_final_selling_price : final_selling_price = purchase_price * (1 + profit_margin_rate))
    (h_marked_price : marked_price * (1 - final_discount_rate) = final_selling_price) : 
    marked_price = 40.80 :=
by
  sorry

end marked_price_correct_l1502_150216


namespace ways_from_A_to_C_l1502_150227

theorem ways_from_A_to_C (ways_A_to_B : ℕ) (ways_B_to_C : ℕ) (hA_to_B : ways_A_to_B = 3) (hB_to_C : ways_B_to_C = 4) : ways_A_to_B * ways_B_to_C = 12 :=
by
  sorry

end ways_from_A_to_C_l1502_150227


namespace total_noodles_and_pirates_l1502_150212

-- Condition definitions
def pirates : ℕ := 45
def noodles : ℕ := pirates - 7

-- Theorem stating the total number of noodles and pirates
theorem total_noodles_and_pirates : (noodles + pirates) = 83 := by
  sorry

end total_noodles_and_pirates_l1502_150212


namespace discount_percentage_l1502_150271

theorem discount_percentage (CP SP SP_no_discount discount : ℝ)
  (h1 : SP = CP * (1 + 0.44))
  (h2 : SP_no_discount = CP * (1 + 0.50))
  (h3 : discount = SP_no_discount - SP) :
  (discount / SP_no_discount) * 100 = 4 :=
by
  sorry

end discount_percentage_l1502_150271


namespace largest_possible_value_of_n_l1502_150296

open Nat

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ m, m ∣ p → m = 1 ∨ m = p

def largest_product : ℕ :=
  705

theorem largest_possible_value_of_n :
  ∃ (x y : ℕ), x < 10 ∧ y < 10 ∧
  is_prime x ∧ is_prime y ∧
  is_prime (10 * y - x) ∧
  largest_product = x * y * (10 * y - x) :=
by
  sorry

end largest_possible_value_of_n_l1502_150296


namespace fraction_of_butterflies_flew_away_l1502_150232

theorem fraction_of_butterflies_flew_away (original_butterflies : ℕ) (left_butterflies : ℕ) (h1 : original_butterflies = 9) (h2 : left_butterflies = 6) : (original_butterflies - left_butterflies) / original_butterflies = 1 / 3 :=
by
  sorry

end fraction_of_butterflies_flew_away_l1502_150232


namespace find_m_of_hyperbola_l1502_150237

noncomputable def eccen_of_hyperbola (a b : ℝ) : ℝ := Real.sqrt (1 + (b^2) / (a^2))

theorem find_m_of_hyperbola :
  ∃ (m : ℝ), (m > 0) ∧ (eccen_of_hyperbola 2 m = Real.sqrt 3) ∧ (m = 2 * Real.sqrt 2) :=
by
  sorry

end find_m_of_hyperbola_l1502_150237


namespace min_value_of_function_l1502_150273

noncomputable def y (x : ℝ) : ℝ := (Real.cos x) * (Real.sin (2 * x))

theorem min_value_of_function :
  ∃ x ∈ Set.Icc (-Real.pi) Real.pi, y x = -4 * Real.sqrt 3 / 9 :=
sorry

end min_value_of_function_l1502_150273


namespace jamesOreos_count_l1502_150250

noncomputable def jamesOreos (jordanOreos : ℕ) : ℕ := 4 * jordanOreos + 7

theorem jamesOreos_count (J : ℕ) (h1 : J + jamesOreos J = 52) : jamesOreos J = 43 :=
by
  sorry

end jamesOreos_count_l1502_150250


namespace mean_of_remaining_students_l1502_150218

variable (k : ℕ) (h1 : k > 20)

def mean_of_class (mean : ℝ := 10) := mean
def mean_of_20_students (mean : ℝ := 16) := mean

theorem mean_of_remaining_students 
  (h2 : mean_of_class = 10)
  (h3 : mean_of_20_students = 16) :
  let remaining_students := (k - 20)
  let total_score_20 := 20 * mean_of_20_students
  let total_score_class := k * mean_of_class
  let total_score_remaining := total_score_class - total_score_20
  let mean_remaining := total_score_remaining / remaining_students
  mean_remaining = (10 * k - 320) / (k - 20) :=
sorry

end mean_of_remaining_students_l1502_150218


namespace area_triangle_COD_l1502_150255

noncomputable def area_of_triangle (t s : ℝ) : ℝ := 
  1 / 2 * abs (5 + 2 * s + 7 * t)

theorem area_triangle_COD (t s : ℝ) : 
  ∃ (C : ℝ × ℝ) (D : ℝ × ℝ), 
    C = (3 + 5 * t, 2 + 4 * t) ∧ 
    D = (2 + 5 * s, 3 + 4 * s) ∧ 
    area_of_triangle t s = 1 / 2 * abs (5 + 2 * s + 7 * t) :=
by
  sorry

end area_triangle_COD_l1502_150255


namespace ratio_correct_l1502_150291

-- Definitions based on the problem conditions
def initial_cards_before_eating (X : ℤ) : ℤ := X
def cards_bought_new : ℤ := 4
def cards_left_after_eating : ℤ := 34

-- Definition of the number of cards eaten by the dog
def cards_eaten_by_dog (X : ℤ) : ℤ := X + cards_bought_new - cards_left_after_eating

-- Definition of the ratio of the number of cards eaten to the total number of cards before being eaten
def ratio_cards_eaten_to_total (X : ℤ) : ℚ := (cards_eaten_by_dog X : ℚ) / (X + cards_bought_new : ℚ)

-- Statement to prove
theorem ratio_correct (X : ℤ) : ratio_cards_eaten_to_total X = (X - 30) / (X + 4) := by
  sorry

end ratio_correct_l1502_150291


namespace Daniel_correct_answers_l1502_150260

theorem Daniel_correct_answers
  (c w : ℕ)
  (h1 : c + w = 12)
  (h2 : 4 * c - 3 * w = 21) :
  c = 9 :=
sorry

end Daniel_correct_answers_l1502_150260


namespace nth_equation_l1502_150248

theorem nth_equation (n : ℕ) : 
  1 + 6 * n = (3 * n + 1) ^ 2 - 9 * n ^ 2 := 
by 
  sorry

end nth_equation_l1502_150248


namespace range_of_a1_of_arithmetic_sequence_l1502_150264

theorem range_of_a1_of_arithmetic_sequence
  {a : ℕ → ℝ} (S : ℕ → ℝ) (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (h_sum: ∀ n, S n = (n + 1) * (a 0 + a n) / 2)
  (h_min: ∀ n > 0, S n ≥ S 0)
  (h_S1: S 0 = 10) :
  -30 < a 0 ∧ a 0 < -27 := 
sorry

end range_of_a1_of_arithmetic_sequence_l1502_150264


namespace function_D_is_odd_function_D_is_decreasing_l1502_150205

def f_D (x : ℝ) : ℝ := -x * |x|

theorem function_D_is_odd (x : ℝ) : f_D (-x) = -f_D x := by
  sorry

theorem function_D_is_decreasing (x y : ℝ) (h : x < y) : f_D x > f_D y := by
  sorry

end function_D_is_odd_function_D_is_decreasing_l1502_150205


namespace total_games_won_l1502_150200

theorem total_games_won (Betsy_games : ℕ) (Helen_games : ℕ) (Susan_games : ℕ) 
    (hBetsy : Betsy_games = 5)
    (hHelen : Helen_games = 2 * Betsy_games)
    (hSusan : Susan_games = 3 * Betsy_games) : 
    Betsy_games + Helen_games + Susan_games = 30 :=
sorry

end total_games_won_l1502_150200


namespace original_price_of_sarees_l1502_150282

theorem original_price_of_sarees (P : ℝ) (h : 0.72 * P = 108) : P = 150 := 
by 
  sorry

end original_price_of_sarees_l1502_150282


namespace sector_angle_l1502_150238

theorem sector_angle (r l θ : ℝ) (h : 2 * r + l = π * r) : θ = π - 2 :=
sorry

end sector_angle_l1502_150238


namespace original_price_of_candy_box_is_8_l1502_150214

-- Define the given conditions
def candy_box_price_after_increase : ℝ := 10
def candy_box_increase_rate : ℝ := 1.25

-- Define the original price of the candy box
noncomputable def original_candy_box_price : ℝ := candy_box_price_after_increase / candy_box_increase_rate

-- The theorem to prove
theorem original_price_of_candy_box_is_8 :
  original_candy_box_price = 8 := by
  sorry

end original_price_of_candy_box_is_8_l1502_150214


namespace inequality_system_solution_l1502_150284

theorem inequality_system_solution (a b : ℝ) (h : ∀ x : ℝ, x > -a → x > -b) : a ≥ b :=
by
  sorry

end inequality_system_solution_l1502_150284


namespace equation_solution_l1502_150297

theorem equation_solution (x : ℝ) (h : x ≠ 2) :
  (3 * x + 6) / (x^2 + 5 * x - 14) = (3 - x) / (x - 2) ↔ x = 3 ∨ x = -5 :=
by 
  sorry

end equation_solution_l1502_150297


namespace total_population_after_births_l1502_150268

theorem total_population_after_births:
  let initial_population := 300000
  let immigrants := 50000
  let emigrants := 30000
  let pregnancies_fraction := 1 / 8
  let twins_fraction := 1 / 4
  let net_population := initial_population + immigrants - emigrants
  let pregnancies := net_population * pregnancies_fraction
  let twin_pregnancies := pregnancies * twins_fraction
  let twin_children := twin_pregnancies * 2
  let single_births := pregnancies - twin_pregnancies
  net_population + single_births + twin_children = 370000 := by
  sorry

end total_population_after_births_l1502_150268


namespace cost_of_each_skin_l1502_150223

theorem cost_of_each_skin
  (total_value : ℕ)
  (overall_profit : ℚ)
  (profit_first : ℚ)
  (profit_second : ℚ)
  (total_sell : ℕ)
  (equality : (1 : ℚ) + profit_first ≠ 0 ∧ (1 : ℚ) + profit_second ≠ 0) :
  total_value = 2250 → overall_profit = 0.4 → profit_first = 0.25 → profit_second = -0.5 →
  total_sell = 3150 →
  ∃ x y : ℚ, x = 2700 ∧ y = -450 :=
by
  sorry

end cost_of_each_skin_l1502_150223


namespace sister_age_is_one_l1502_150235

variable (B S : ℕ)

theorem sister_age_is_one (h : B = B * S) : S = 1 :=
by {
  sorry
}

end sister_age_is_one_l1502_150235


namespace solution_a_l1502_150274

noncomputable def problem_a (a b c y : ℕ) : Prop :=
  a + b + c = 30 ∧ b + c + y = 30 ∧ a = 2 ∧ y = 3

theorem solution_a (a b c y x : ℕ)
  (h : problem_a a b c y)
  : x = 25 :=
by sorry

end solution_a_l1502_150274


namespace p_sufficient_not_necessary_for_q_l1502_150290

def p (x1 x2 : ℝ) : Prop := x1 > 1 ∧ x2 > 1
def q (x1 x2 : ℝ) : Prop := x1 + x2 > 2 ∧ x1 * x2 > 1

theorem p_sufficient_not_necessary_for_q : 
  (∀ x1 x2 : ℝ, p x1 x2 → q x1 x2) ∧ ¬ (∀ x1 x2 : ℝ, q x1 x2 → p x1 x2) :=
by 
  sorry

end p_sufficient_not_necessary_for_q_l1502_150290


namespace MeatMarket_sales_l1502_150267

theorem MeatMarket_sales :
  let thursday_sales := 210
  let friday_sales := 2 * thursday_sales
  let saturday_sales := 130
  let sunday_sales := saturday_sales / 2
  let total_sales := thursday_sales + friday_sales + saturday_sales + sunday_sales
  let planned_sales := 500
  total_sales - planned_sales = 325 :=
by
  let thursday_sales := 210
  let friday_sales := 2 * thursday_sales
  let saturday_sales := 130
  let sunday_sales := saturday_sales / 2
  let total_sales := thursday_sales + friday_sales + saturday_sales + sunday_sales
  let planned_sales := 500
  show total_sales - planned_sales = 325
  sorry

end MeatMarket_sales_l1502_150267


namespace gcd_888_1147_l1502_150294

/-- Use the Euclidean algorithm to find the greatest common divisor (GCD) of 888 and 1147. -/
theorem gcd_888_1147 : Nat.gcd 888 1147 = 37 := by
  sorry

end gcd_888_1147_l1502_150294


namespace leak_empty_tank_time_l1502_150286

theorem leak_empty_tank_time (fill_time_A : ℝ) (fill_time_A_with_leak : ℝ) (leak_empty_time : ℝ) :
  fill_time_A = 6 → fill_time_A_with_leak = 9 → leak_empty_time = 18 :=
by
  intros hA hL
  -- Here follows the proof we skip
  sorry

end leak_empty_tank_time_l1502_150286


namespace percent_x_of_w_l1502_150236

theorem percent_x_of_w (x y z w : ℝ)
  (h1 : x = 1.2 * y)
  (h2 : y = 0.7 * z)
  (h3 : w = 1.5 * z) : (x / w) * 100 = 56 :=
by
  sorry

end percent_x_of_w_l1502_150236


namespace find_a_l1502_150204

noncomputable def tangent_line (a : ℝ) (x : ℝ) := (3 * a * (1:ℝ)^2 + 1) * (x - 1) + (a * (1:ℝ)^3 + (1:ℝ) + 1)

theorem find_a : ∃ a : ℝ, tangent_line a 2 = 7 := 
sorry

end find_a_l1502_150204


namespace simplify_expression_l1502_150224

variable (z : ℝ)

theorem simplify_expression :
  (z - 2 * z + 4 * z - 6 + 3 + 7 - 2) = (3 * z + 2) := by
  sorry

end simplify_expression_l1502_150224


namespace students_passed_in_both_subjects_l1502_150278

theorem students_passed_in_both_subjects:
  ∀ (F_H F_E F_HE : ℝ), F_H = 0.30 → F_E = 0.42 → F_HE = 0.28 → (1 - (F_H + F_E - F_HE)) = 0.56 :=
by
  intros F_H F_E F_HE h1 h2 h3
  sorry

end students_passed_in_both_subjects_l1502_150278


namespace excluded_twins_lineup_l1502_150217

/-- 
  Prove that the number of ways to choose 5 starters from 15 players,
  such that both Alice and Bob (twins) are not included together in the lineup, is 2717.
-/
theorem excluded_twins_lineup (n : ℕ) (k : ℕ) (t : ℕ) (u : ℕ) (h_n : n = 15) (h_k : k = 5) (h_t : t = 2) (h_u : u = 3) :
  ((n.choose k) - ((n - t).choose u)) = 2717 :=
by {
  sorry
}

end excluded_twins_lineup_l1502_150217


namespace simplify_fractions_l1502_150201

theorem simplify_fractions:
  (3 / 462 : ℚ) + (28 / 42 : ℚ) = 311 / 462 := sorry

end simplify_fractions_l1502_150201


namespace length_of_CD_l1502_150265

theorem length_of_CD (x y u v : ℝ) (R S C D : ℝ → ℝ)
  (h1 : 5 * x = 3 * y)
  (h2 : 7 * u = 4 * v)
  (h3 : u = x + 3)
  (h4 : v = y - 3)
  (h5 : C x + D y = 1) : 
  x + y = 264 :=
by
  sorry

end length_of_CD_l1502_150265


namespace infinite_series_sum_eq_1_div_432_l1502_150239

theorem infinite_series_sum_eq_1_div_432 :
  (∑' n : ℕ, (4 * (n + 1) + 1) / ((4 * (n + 1) - 1)^3 * (4 * (n + 1) + 3)^3)) = (1 / 432) :=
  sorry

end infinite_series_sum_eq_1_div_432_l1502_150239


namespace cookies_left_after_ted_leaves_l1502_150226

theorem cookies_left_after_ted_leaves :
  let f : Nat := 2 -- trays per day
  let d : Nat := 6 -- days
  let e_f : Nat := 1 -- cookies eaten per day by Frank
  let t : Nat := 4 -- cookies eaten by Ted
  let c : Nat := 12 -- cookies per tray
  let total_cookies := f * c * d -- total cookies baked
  let cookies_eaten_by_frank := e_f * d -- total cookies eaten by Frank
  let cookies_before_ted := total_cookies - cookies_eaten_by_frank -- cookies before Ted
  let total_cookies_left := cookies_before_ted - t -- cookies left after Ted
  total_cookies_left = 134
:= by
  sorry

end cookies_left_after_ted_leaves_l1502_150226


namespace average_weight_l1502_150210

theorem average_weight (men women : ℕ) (avg_weight_men avg_weight_women : ℝ) (total_people : ℕ) (combined_avg_weight : ℝ) 
  (h1 : men = 8) (h2 : avg_weight_men = 190) (h3 : women = 6) (h4 : avg_weight_women = 120) (h5 : total_people = 14) 
  (h6 : (men * avg_weight_men + women * avg_weight_women) / total_people = combined_avg_weight) : combined_avg_weight = 160 := 
  sorry

end average_weight_l1502_150210


namespace math_bonanza_2016_8_l1502_150247

def f (x : ℕ) := x^2 + x + 1

theorem math_bonanza_2016_8 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : f p = f q + 242) (hpq : p > q) :
  (p, q) = (61, 59) :=
by sorry

end math_bonanza_2016_8_l1502_150247


namespace largest_pies_without_ingredients_l1502_150288

variable (total_pies : ℕ) (chocolate_pies marshmallow_pies cayenne_pies soy_nut_pies : ℕ)
variable (b : total_pies = 36)
variable (c : chocolate_pies = total_pies / 2)
variable (m : marshmallow_pies = 2 * total_pies / 3)
variable (k : cayenne_pies = 3 * total_pies / 4)
variable (s : soy_nut_pies = total_pies / 6)

theorem largest_pies_without_ingredients (total_pies chocolate_pies marshmallow_pies cayenne_pies soy_nut_pies : ℕ)
  (b : total_pies = 36)
  (c : chocolate_pies = total_pies / 2)
  (m : marshmallow_pies = 2 * total_pies / 3)
  (k : cayenne_pies = 3 * total_pies / 4)
  (s : soy_nut_pies = total_pies / 6) :
  9 = total_pies - chocolate_pies - marshmallow_pies - cayenne_pies - soy_nut_pies + 3 * cayenne_pies := 
by
  sorry

end largest_pies_without_ingredients_l1502_150288


namespace josh_marbles_earlier_l1502_150252

-- Define the conditions
def marbles_lost : ℕ := 11
def marbles_now : ℕ := 8

-- Define the problem statement
theorem josh_marbles_earlier : marbles_lost + marbles_now = 19 :=
by
  sorry

end josh_marbles_earlier_l1502_150252


namespace original_number_of_movies_l1502_150298

/-- Suppose a movie buff owns movies on DVD, Blu-ray, and digital copies in a ratio of 7:2:1.
    After purchasing 5 more Blu-ray movies and 3 more digital copies, the ratio changes to 13:4:2.
    She owns movies on no other medium.
    Prove that the original number of movies in her library before the extra purchase was 390. -/
theorem original_number_of_movies (x : ℕ) (h1 : 7 * x != 0) 
  (h2 : 2 * x != 0) (h3 : x != 0)
  (h4 : 7 * x / (2 * x + 5) = 13 / 4)
  (h5 : 7 * x / (x + 3) = 13 / 2) : 10 * x = 390 :=
by
  sorry

end original_number_of_movies_l1502_150298


namespace intersection_of_sets_l1502_150244

def setA : Set ℝ := {x | (x^2 - x - 2 < 0)}
def setB : Set ℝ := {y | ∃ x ≤ 0, y = 3^x}

theorem intersection_of_sets : (setA ∩ setB) = {z | 0 < z ∧ z ≤ 1} :=
sorry

end intersection_of_sets_l1502_150244


namespace even_function_has_specific_a_l1502_150241

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x ^ 2 + (2 * a ^ 2 - a) * x + 1

-- State the proof problem
theorem even_function_has_specific_a (a : ℝ) :
  (∀ x : ℝ, f a (-x) = f a x) → a = 1 / 2 :=
by
  intros h
  sorry

end even_function_has_specific_a_l1502_150241


namespace sum_of_powers_2017_l1502_150249

theorem sum_of_powers_2017 (n : ℕ) (x : Fin n → ℤ) (h : ∀ i, x i = 0 ∨ x i = 1 ∨ x i = -1) (h_sum : (Finset.univ : Finset (Fin n)).sum x = 1000) :
  (Finset.univ : Finset (Fin n)).sum (λ i => (x i)^2017) = 1000 :=
by
  sorry

end sum_of_powers_2017_l1502_150249


namespace total_wheels_l1502_150258

def cars := 2
def car_wheels := 4
def bikes_with_one_wheel := 1
def bikes_with_two_wheels := 2
def trash_can_wheels := 2
def tricycle_wheels := 3
def roller_skate_wheels := 3 -- since one is missing a wheel
def wheelchair_wheels := 6 -- 4 large + 2 small wheels
def wagon_wheels := 4

theorem total_wheels : cars * car_wheels + 
                        bikes_with_one_wheel * 1 + 
                        bikes_with_two_wheels * 2 + 
                        trash_can_wheels + 
                        tricycle_wheels + 
                        roller_skate_wheels + 
                        wheelchair_wheels + 
                        wagon_wheels = 31 :=
by
  sorry

end total_wheels_l1502_150258


namespace turtle_distance_in_six_minutes_l1502_150261

theorem turtle_distance_in_six_minutes 
  (observers : ℕ)
  (time_interval : ℕ)
  (distance_seen : ℕ)
  (total_time : ℕ)
  (total_distance : ℕ)
  (observation_per_minute : ∀ t ≤ total_time, ∃ n : ℕ, n ≤ observers ∧ (∃ interval : ℕ, interval ≤ time_interval ∧ distance_seen = 1)) :
  total_distance = 10 :=
sorry

end turtle_distance_in_six_minutes_l1502_150261


namespace number_of_men_in_engineering_department_l1502_150221

theorem number_of_men_in_engineering_department (T : ℝ) (h1 : 0.30 * T = 180) : 
  0.70 * T = 420 :=
by 
  -- The proof will be done here, but for now, we skip it.
  sorry

end number_of_men_in_engineering_department_l1502_150221


namespace percentage_relationships_l1502_150285

variable (a b c d e f g : ℝ)

theorem percentage_relationships (h1 : d = 0.22 * b) (h2 : d = 0.35 * f)
                                 (h3 : e = 0.27 * a) (h4 : e = 0.60 * f)
                                 (h5 : c = 0.14 * a) (h6 : c = 0.40 * b)
                                 (h7 : d = 2 * c) (h8 : g = 3 * e):
    b = 0.7 * a ∧ f = 0.45 * a ∧ g = 0.81 * a :=
sorry

end percentage_relationships_l1502_150285


namespace find_c_k_l1502_150272

noncomputable def common_difference (a : ℕ → ℕ) : ℕ := sorry
noncomputable def common_ratio (b : ℕ → ℕ) : ℕ := sorry
noncomputable def arith_seq (d : ℕ) (n : ℕ) : ℕ := 1 + (n - 1) * d
noncomputable def geom_seq (r : ℕ) (n : ℕ) : ℕ := r^(n - 1)
noncomputable def combined_seq (a : ℕ → ℕ) (b : ℕ → ℕ) (n : ℕ) : ℕ := a n + b n

variable (k : ℕ) (d : ℕ) (r : ℕ)

-- Conditions
axiom arith_condition : common_difference (arith_seq d) = d
axiom geom_condition : common_ratio (geom_seq r) = r
axiom combined_k_minus_1 : combined_seq (arith_seq d) (geom_seq r) (k - 1) = 50
axiom combined_k_plus_1 : combined_seq (arith_seq d) (geom_seq r) (k + 1) = 1500

-- Prove that c_k = 2406
theorem find_c_k : combined_seq (arith_seq d) (geom_seq r) k = 2406 := by
  sorry

end find_c_k_l1502_150272


namespace arithmetic_sequence_product_l1502_150215

theorem arithmetic_sequence_product (b : ℕ → ℤ) (d : ℤ) 
  (h_inc : ∀ n, b (n + 1) - b n = d)
  (h_pos : d > 0)
  (h_prod : b 5 * b 6 = 21) 
  : b 4 * b 7 = -779 ∨ b 4 * b 7 = -11 :=
sorry

end arithmetic_sequence_product_l1502_150215


namespace project_completion_time_l1502_150222

theorem project_completion_time (initial_workers : ℕ) (initial_days : ℕ) (extra_workers : ℕ) (extra_days : ℕ) : 
  initial_workers = 10 →
  initial_days = 15 →
  extra_workers = 5 →
  extra_days = 5 →
  total_days = 6 := by
  sorry

end project_completion_time_l1502_150222


namespace average_xyz_l1502_150240

theorem average_xyz (x y z : ℝ) 
  (h1 : 2003 * z - 4006 * x = 1002) 
  (h2 : 2003 * y + 6009 * x = 4004) : (x + y + z) / 3 = 5 / 6 :=
by
  sorry

end average_xyz_l1502_150240


namespace cost_for_flour_for_two_cakes_l1502_150234

theorem cost_for_flour_for_two_cakes 
    (packages_per_cake : ℕ)
    (cost_per_package : ℕ)
    (cakes : ℕ) 
    (total_cost : ℕ)
    (H1 : packages_per_cake = 2)
    (H2 : cost_per_package = 3)
    (H3 : cakes = 2)
    (H4 : total_cost = 12) :
    total_cost = cakes * packages_per_cake * cost_per_package := 
by 
    rw [H1, H2, H3]
    sorry

end cost_for_flour_for_two_cakes_l1502_150234


namespace sample_capacity_l1502_150279

theorem sample_capacity (freq : ℕ) (freq_rate : ℚ) (H_freq : freq = 36) (H_freq_rate : freq_rate = 0.25) : 
  ∃ n : ℕ, n = 144 :=
by
  sorry

end sample_capacity_l1502_150279


namespace semesters_needed_l1502_150281

def total_credits : ℕ := 120
def credits_per_class : ℕ := 3
def classes_per_semester : ℕ := 5

theorem semesters_needed (h1 : total_credits = 120)
                         (h2 : credits_per_class = 3)
                         (h3 : classes_per_semester = 5) :
  total_credits / (credits_per_class * classes_per_semester) = 8 := 
by {
  sorry
}

end semesters_needed_l1502_150281


namespace tangent_line_at_point_l1502_150283

noncomputable def tangent_line_eq (x y : ℝ) : Prop := x^3 - y = 0

theorem tangent_line_at_point :
  tangent_line_eq (-2) (-8) →
  ∃ (k : ℝ), (k = 12) ∧ (12 * x - y + 16 = 0) :=
sorry

end tangent_line_at_point_l1502_150283
