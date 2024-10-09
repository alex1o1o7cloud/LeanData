import Mathlib

namespace arctan_sum_of_roots_eq_pi_div_4_l1989_198944

theorem arctan_sum_of_roots_eq_pi_div_4 (x₁ x₂ x₃ : ℝ) 
  (h₁ : Polynomial.eval x₁ (Polynomial.C 11 - Polynomial.C 10 * Polynomial.X + Polynomial.X ^ 3) = 0)
  (h₂ : Polynomial.eval x₂ (Polynomial.C 11 - Polynomial.C 10 * Polynomial.X + Polynomial.X ^ 3) = 0)
  (h₃ : Polynomial.eval x₃ (Polynomial.C 11 - Polynomial.C 10 * Polynomial.X + Polynomial.X ^ 3) = 0)
  (h_intv : -5 < x₁ ∧ x₁ < 5 ∧ -5 < x₂ ∧ x₂ < 5 ∧ -5 < x₃ ∧ x₃ < 5) :
  Real.arctan x₁ + Real.arctan x₂ + Real.arctan x₃ = Real.pi / 4 :=
sorry

end arctan_sum_of_roots_eq_pi_div_4_l1989_198944


namespace hexagon_vertices_zero_l1989_198934

theorem hexagon_vertices_zero (n : ℕ) (a0 a1 a2 a3 a4 a5 : ℕ) 
  (h_sum : a0 + a1 + a2 + a3 + a4 + a5 = n) 
  (h_pos : 0 < n) :
  (n = 2 ∨ n % 2 = 1) → 
  ∃ (b0 b1 b2 b3 b4 b5 : ℕ), b0 = 0 ∧ b1 = 0 ∧ b2 = 0 ∧ b3 = 0 ∧ b4 = 0 ∧ b5 = 0 := sorry

end hexagon_vertices_zero_l1989_198934


namespace evaluate_expression_l1989_198925

theorem evaluate_expression :
  (4 * 10^2011 - 1) / (4 * (3 * (10^2011 - 1) / 9) + 1) = 3 :=
by
  sorry

end evaluate_expression_l1989_198925


namespace window_design_ratio_l1989_198902

theorem window_design_ratio (AB AD r : ℝ)
  (h1 : AB = 40)
  (h2 : AD / AB = 4 / 3)
  (h3 : r = AB / 2) :
  ((AD - AB) * AB) / (π * r^2 / 2) = 8 / (3 * π) :=
by
  sorry

end window_design_ratio_l1989_198902


namespace brick_wall_l1989_198993

theorem brick_wall (y : ℕ) (h1 : ∀ y, 6 * ((y / 8) + (y / 12) - 12) = y) : y = 288 :=
sorry

end brick_wall_l1989_198993


namespace sum_of_integers_with_product_2720_l1989_198928

theorem sum_of_integers_with_product_2720 (n : ℤ) (h1 : n > 0) (h2 : n * (n + 2) = 2720) : n + (n + 2) = 104 :=
by {
  sorry
}

end sum_of_integers_with_product_2720_l1989_198928


namespace axis_of_symmetry_shifted_sine_function_l1989_198931

open Real

noncomputable def axisOfSymmetry (k : ℤ) : ℝ := k * π / 2 + π / 6

theorem axis_of_symmetry_shifted_sine_function (x : ℝ) (k : ℤ) :
  ∃ k : ℤ, x = axisOfSymmetry k := by
sorry

end axis_of_symmetry_shifted_sine_function_l1989_198931


namespace ratio_of_discretionary_income_l1989_198921

theorem ratio_of_discretionary_income
  (net_monthly_salary : ℝ) 
  (vacation_fund_pct : ℝ) 
  (savings_pct : ℝ) 
  (socializing_pct : ℝ) 
  (gifts_amt : ℝ)
  (D : ℝ) 
  (ratio : ℝ)
  (salary : net_monthly_salary = 3700)
  (vacation_fund : vacation_fund_pct = 0.30)
  (savings : savings_pct = 0.20)
  (socializing : socializing_pct = 0.35)
  (gifts : gifts_amt = 111)
  (discretionary_income : D = gifts_amt / 0.15)
  (net_salary_ratio : ratio = D / net_monthly_salary) :
  ratio = 1 / 5 := sorry

end ratio_of_discretionary_income_l1989_198921


namespace candied_apple_price_l1989_198995

theorem candied_apple_price
  (x : ℝ) -- price of each candied apple in dollars
  (h1 : 15 * x + 12 * 1.5 = 48) -- total earnings equation
  : x = 2 := 
sorry

end candied_apple_price_l1989_198995


namespace find_X_plus_Y_in_base_8_l1989_198997

theorem find_X_plus_Y_in_base_8 (X Y : ℕ) (h1 : 3 * 8^2 + X * 8 + Y + 5 * 8 + 2 = 4 * 8^2 + X * 8 + 3) : X + Y = 1 :=
sorry

end find_X_plus_Y_in_base_8_l1989_198997


namespace cylinder_volume_relation_l1989_198922

theorem cylinder_volume_relation (r h : ℝ) (π_pos : 0 < π) :
  (∀ B_h B_r A_h A_r : ℝ, B_h = r ∧ B_r = h ∧ A_h = h ∧ A_r = r 
   → 3 * (π * h^2 * r) = π * r^2 * h) → 
  ∃ N : ℝ, (π * (3 * h)^2 * h) = N * π * h^3 ∧ N = 9 :=
by 
  sorry

end cylinder_volume_relation_l1989_198922


namespace sum_of_fourth_powers_l1989_198999

theorem sum_of_fourth_powers (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 4) : 
  a^4 + b^4 + c^4 = 8 := 
by 
  sorry

end sum_of_fourth_powers_l1989_198999


namespace oranges_to_put_back_l1989_198984

theorem oranges_to_put_back
  (price_apple price_orange : ℕ)
  (A_all O_all : ℕ)
  (mean_initial_fruit mean_final_fruit : ℕ)
  (A O x : ℕ)
  (h_price_apple : price_apple = 40)
  (h_price_orange : price_orange = 60)
  (h_total_fruit : A_all + O_all = 10)
  (h_mean_initial : mean_initial_fruit = 54)
  (h_mean_final : mean_final_fruit = 50)
  (h_total_cost_initial : price_apple * A_all + price_orange * O_all = mean_initial_fruit * (A_all + O_all))
  (h_total_cost_final : price_apple * A + price_orange * (O - x) = mean_final_fruit * (A + (O - x)))
  : x = 4 := 
  sorry

end oranges_to_put_back_l1989_198984


namespace salary_of_A_l1989_198955

theorem salary_of_A (x y : ℝ) (h₁ : x + y = 4000) (h₂ : 0.05 * x = 0.15 * y) : x = 3000 :=
by {
    sorry
}

end salary_of_A_l1989_198955


namespace solution_of_inequality_l1989_198964

open Set

theorem solution_of_inequality (x : ℝ) :
  x^2 - 2 * x - 3 > 0 ↔ x < -1 ∨ x > 3 :=
by
  sorry

end solution_of_inequality_l1989_198964


namespace MissAisha_height_l1989_198960

theorem MissAisha_height (H : ℝ)
  (legs_length : ℝ := H / 3)
  (head_length : ℝ := H / 4)
  (rest_body_length : ℝ := 25) :
  H = 60 :=
by sorry

end MissAisha_height_l1989_198960


namespace tan_y_eq_tan_x_plus_one_over_cos_x_l1989_198932

theorem tan_y_eq_tan_x_plus_one_over_cos_x 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hxy : x < y) 
  (hy : y < π / 2) 
  (h_tan : Real.tan y = Real.tan x + (1 / Real.cos x)) 
  : y - (x / 2) = π / 6 :=
sorry

end tan_y_eq_tan_x_plus_one_over_cos_x_l1989_198932


namespace total_canoes_built_l1989_198926

-- Defining basic variables and functions for the proof
variable (a : Nat := 5) -- Initial number of canoes in January
variable (r : Nat := 3) -- Common ratio
variable (n : Nat := 6) -- Number of months including January

-- Function to compute sum of the first n terms of a geometric series
def geometric_sum (a r n : Nat) : Nat :=
  a * (r^n - 1) / (r - 1)

-- The proposition we want to prove
theorem total_canoes_built : geometric_sum a r n = 1820 := by
  sorry

end total_canoes_built_l1989_198926


namespace one_fourth_of_8_point_4_is_21_over_10_l1989_198951

theorem one_fourth_of_8_point_4_is_21_over_10 : (8.4 / 4 : ℚ) = 21 / 10 := 
by
  sorry

end one_fourth_of_8_point_4_is_21_over_10_l1989_198951


namespace distance_between_first_and_last_tree_l1989_198935

theorem distance_between_first_and_last_tree (n : ℕ) (d : ℕ) 
  (h₁ : n = 8)
  (h₂ : d = 75)
  : (d / ((4 - 1) : ℕ)) * (n - 1) = 175 := sorry

end distance_between_first_and_last_tree_l1989_198935


namespace unique_a_exists_for_prime_p_l1989_198985

theorem unique_a_exists_for_prime_p (p : ℕ) [Fact p.Prime] :
  (∃! (a : ℕ), a ∈ Finset.range (p + 1) ∧ (a^3 - 3*a + 1) % p = 0) ↔ p = 3 := by
  sorry

end unique_a_exists_for_prime_p_l1989_198985


namespace Integers_and_fractions_are_rational_numbers_l1989_198990

-- Definitions from conditions
def is_fraction (x : ℚ) : Prop :=
  ∃a b : ℤ, b ≠ 0 ∧ x = (a : ℚ) / (b : ℚ)

def is_integer (x : ℤ) : Prop := 
  ∃n : ℤ, x = n

def is_rational (x : ℚ) : Prop := 
  ∃a b : ℤ, b ≠ 0 ∧ x = (a : ℚ) / (b : ℚ)

-- The statement to be proven
theorem Integers_and_fractions_are_rational_numbers (x : ℚ) : 
  (∃n : ℤ, x = (n : ℚ)) ∨ is_fraction x ↔ is_rational x :=
by sorry

end Integers_and_fractions_are_rational_numbers_l1989_198990


namespace xy_nonzero_implies_iff_l1989_198957

variable {x y : ℝ}

theorem xy_nonzero_implies_iff (h : x * y ≠ 0) : (x + y = 0) ↔ (x / y + y / x = -2) :=
sorry

end xy_nonzero_implies_iff_l1989_198957


namespace count_ordered_pairs_l1989_198998

theorem count_ordered_pairs : 
  ∃ n, n = 719 ∧ 
    (∀ (a b : ℕ), a + b = 1100 → 
      (∀ d ∈ [a, b], 
        ¬(∃ k : ℕ, d = 10 * k ∨ d % 10 = 0 ∨ d / 10 % 10 = 0 ∨ d % 5 = 0))) -> n = 719 :=
by
  sorry

end count_ordered_pairs_l1989_198998


namespace hexagon_longest_side_l1989_198950

theorem hexagon_longest_side (x : ℝ) (h₁ : 6 * x = 20) (h₂ : x < 20 - x) : (10 / 3) ≤ x ∧ x < 10 :=
sorry

end hexagon_longest_side_l1989_198950


namespace thursday_to_wednesday_ratio_l1989_198976

-- Let M, T, W, Th be the number of messages sent on Monday, Tuesday, Wednesday, and Thursday respectively.
variables (M T W Th : ℕ)

-- Conditions are given as follows
axiom hM : M = 300
axiom hT : T = 200
axiom hW : W = T + 300
axiom hSum : M + T + W + Th = 2000

-- Define the function to compute the ratio
def ratio (a b : ℕ) : ℚ := a / b

-- The target is to prove that the ratio of the messages sent on Thursday to those sent on Wednesday is 2 / 1
theorem thursday_to_wednesday_ratio : ratio Th W = 2 :=
by {
  sorry
}

end thursday_to_wednesday_ratio_l1989_198976


namespace three_pow_n_minus_one_not_divisible_by_two_pow_n_minus_one_l1989_198958

theorem three_pow_n_minus_one_not_divisible_by_two_pow_n_minus_one (n : ℕ) (h : n > 1) : ¬(2^n - 1) ∣ (3^n - 1) :=
sorry

end three_pow_n_minus_one_not_divisible_by_two_pow_n_minus_one_l1989_198958


namespace determine_a_range_l1989_198914

open Real

theorem determine_a_range (a : ℝ) :
  (∃ x : ℝ, x^2 + 2*x + a ≤ 0) → a ≤ 1 :=
sorry

end determine_a_range_l1989_198914


namespace polygon_vertices_l1989_198924

-- Define the number of diagonals from one vertex
def diagonals_from_one_vertex (n : ℕ) := n - 3

-- The main theorem stating the number of vertices is 9 given 6 diagonals from one vertex
theorem polygon_vertices (D : ℕ) (n : ℕ) (h : D = 6) (h_diagonals : diagonals_from_one_vertex n = D) :
  n = 9 := by
  sorry

end polygon_vertices_l1989_198924


namespace age_of_eldest_boy_l1989_198904

theorem age_of_eldest_boy (x : ℕ) (h1 : (3*x + 5*x + 7*x) / 3 = 15) :
  7 * x = 21 :=
sorry

end age_of_eldest_boy_l1989_198904


namespace total_crosswalk_lines_l1989_198973

theorem total_crosswalk_lines 
  (intersections : ℕ) 
  (crosswalks_per_intersection : ℕ) 
  (lines_per_crosswalk : ℕ)
  (h1 : intersections = 10)
  (h2 : crosswalks_per_intersection = 8)
  (h3 : lines_per_crosswalk = 30) :
  intersections * crosswalks_per_intersection * lines_per_crosswalk = 2400 := 
by {
  sorry
}

end total_crosswalk_lines_l1989_198973


namespace partial_fraction_identity_l1989_198916

theorem partial_fraction_identity
  (P Q R : ℝ)
  (h1 : -2 = P + Q)
  (h2 : 1 = Q + R)
  (h3 : -1 = P + R) :
  (P, Q, R) = (-2, 0, 1) :=
by
  sorry

end partial_fraction_identity_l1989_198916


namespace evaluate_fraction_expression_l1989_198969

theorem evaluate_fraction_expression :
  ( (1 / 5 - 1 / 6) / (1 / 3 - 1 / 4) ) = 2 / 5 :=
by
  sorry

end evaluate_fraction_expression_l1989_198969


namespace paula_paint_cans_l1989_198948

variables (rooms_per_can total_rooms_lost initial_rooms final_rooms cans_lost : ℕ)

theorem paula_paint_cans
  (h1 : initial_rooms = 50)
  (h2 : cans_lost = 2)
  (h3 : final_rooms = 42)
  (h4 : total_rooms_lost = initial_rooms - final_rooms)
  (h5 : rooms_per_can = total_rooms_lost / cans_lost) :
  final_rooms / rooms_per_can = 11 :=
by sorry

end paula_paint_cans_l1989_198948


namespace sum_of_digits_of_d_l1989_198970

noncomputable section

def exchange_rate : ℚ := 8/5
def euros_after_spending (d : ℚ) : ℚ := exchange_rate * d - 80

theorem sum_of_digits_of_d {d : ℚ} (h : euros_after_spending d = d) : 
  d = 135 ∧ 1 + 3 + 5 = 9 := 
by 
  sorry

end sum_of_digits_of_d_l1989_198970


namespace man_late_minutes_l1989_198933

theorem man_late_minutes (v t t' : ℝ) (hv : v' = 3 / 4 * v) (ht : t = 2) (ht' : t' = 4 / 3 * t) :
  t' * 60 - t * 60 = 40 :=
by
  sorry

end man_late_minutes_l1989_198933


namespace expand_and_simplify_l1989_198994

theorem expand_and_simplify (x : ℝ) : 6 * (x - 3) * (x + 10) = 6 * x^2 + 42 * x - 180 :=
by
  sorry

end expand_and_simplify_l1989_198994


namespace dan_stationery_spent_l1989_198953

def total_spent : ℕ := 32
def backpack_cost : ℕ := 15
def notebook_cost : ℕ := 3
def number_of_notebooks : ℕ := 5
def stationery_cost_each : ℕ := 1

theorem dan_stationery_spent : 
  (total_spent - (backpack_cost + notebook_cost * number_of_notebooks)) = 2 :=
by
  sorry

end dan_stationery_spent_l1989_198953


namespace train_stops_time_l1989_198915

theorem train_stops_time 
  (speed_excluding_stoppages : ℝ)
  (speed_including_stoppages : ℝ)
  (h1 : speed_excluding_stoppages = 60)
  (h2 : speed_including_stoppages = 40) : 
  ∃ (stoppage_time : ℝ), stoppage_time = 20 := 
by
  sorry

end train_stops_time_l1989_198915


namespace value_of_a_l1989_198939

theorem value_of_a (a b : ℝ) (h1 : b = 4 * a) (h2 : b = 20 - 7 * a) : a = 20 / 11 := by
  sorry

end value_of_a_l1989_198939


namespace evaluate_polynomial_at_2_l1989_198982

theorem evaluate_polynomial_at_2 : (2^4 + 2^3 + 2^2 + 2 + 2) = 32 := 
by
  sorry

end evaluate_polynomial_at_2_l1989_198982


namespace largest_three_digit_sum_l1989_198930

open Nat

def isDigit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

def areDistinct (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem largest_three_digit_sum : 
  ∀ (X Y Z : ℕ), isDigit X → isDigit Y → isDigit Z → areDistinct X Y Z →
  100 ≤  (110 * X + 11 * Y + 2 * Z) → (110 * X + 11 * Y + 2 * Z) ≤ 999 → 
  110 * X + 11 * Y + 2 * Z ≤ 982 :=
by
  intros
  sorry

end largest_three_digit_sum_l1989_198930


namespace find_f_condition_l1989_198954

theorem find_f_condition {f : ℂ → ℂ} (h : ∀ z : ℂ, f z + z * f (1 - z) = 1 + z) :
  ∀ z : ℂ, f z = 1 :=
by
  sorry

end find_f_condition_l1989_198954


namespace average_of_last_six_l1989_198974

theorem average_of_last_six (avg_13 : ℕ → ℝ) (avg_first_6 : ℕ → ℝ) (middle_number : ℕ → ℝ) :
  (∀ n, avg_13 n = 9) →
  (∀ n, n ≤ 6 → avg_first_6 n = 5) →
  (middle_number 7 = 45) →
  ∃ (A : ℝ), (∀ n, n > 6 → n < 13 → avg_13 n = A) ∧ A = 7 :=
by
  sorry

end average_of_last_six_l1989_198974


namespace molecular_weight_n2o_l1989_198952

theorem molecular_weight_n2o (w : ℕ) (n : ℕ) (h : w = 352 ∧ n = 8) : (w / n = 44) :=
sorry

end molecular_weight_n2o_l1989_198952


namespace solve_quadratic_equation_l1989_198977

theorem solve_quadratic_equation : 
  ∀ x : ℝ, 2 * x^2 = 4 ↔ x = Real.sqrt 2 ∨ x = -Real.sqrt 2 :=
by
  sorry


end solve_quadratic_equation_l1989_198977


namespace exponent_equality_l1989_198962

theorem exponent_equality (n : ℕ) : 
    5^n = 5 * (5^2)^2 * (5^3)^3 → n = 14 := by
    sorry

end exponent_equality_l1989_198962


namespace volume_of_square_pyramid_l1989_198978

theorem volume_of_square_pyramid (a r : ℝ) : 
  a > 0 → r > 0 → volume = (1 / 3) * a^2 * r :=
by 
    sorry

end volume_of_square_pyramid_l1989_198978


namespace gcd_65536_49152_l1989_198917

theorem gcd_65536_49152 : Nat.gcd 65536 49152 = 16384 :=
by
  sorry

end gcd_65536_49152_l1989_198917


namespace misha_contributes_l1989_198910

noncomputable def misha_contribution (k l m : ℕ) : ℕ :=
  if h : k + l + m = 6 ∧ 2 * k ≤ l + m ∧ 2 * l ≤ k + m ∧ 2 * m ≤ k + l ∧ k ≤ 2 ∧ l ≤ 2 ∧ m ≤ 2 then
    2
  else
    0 -- This is a default value; the actual proof will check for exact solution.

theorem misha_contributes (k l m : ℕ) (h1 : k + l + m = 6)
    (h2 : 2 * k ≤ l + m) (h3 : 2 * l ≤ k + m) (h4 : 2 * m ≤ k + l)
    (h5 : k ≤ 2) (h6 : l ≤ 2) (h7 : m ≤ 2) : m = 2 := by
  sorry

end misha_contributes_l1989_198910


namespace range_of_a_l1989_198979

-- Definitions of the conditions
def p (x : ℝ) : Prop := x^2 - 8 * x - 20 < 0
def q (x : ℝ) (a : ℝ) : Prop := x^2 - 2 * x + 1 - a^2 ≤ 0 ∧ a > 0

-- Statement of the theorem that proves the range of a
theorem range_of_a (x : ℝ) (a : ℝ) :
  (¬ (p x) → ¬ (q x a)) ∧ (¬ (q x a) → ¬ (p x)) → (a ≥ 9) :=
by
  sorry

end range_of_a_l1989_198979


namespace joes_total_weight_l1989_198992

theorem joes_total_weight (F S : ℕ) (h1 : F = 700) (h2 : 2 * F = S + 300) :
  F + S = 1800 :=
by
  sorry

end joes_total_weight_l1989_198992


namespace Hillary_left_with_amount_l1989_198927

theorem Hillary_left_with_amount :
  let price_per_craft := 12
  let crafts_sold := 3
  let extra_earnings := 7
  let deposit_amount := 18
  let total_earnings := crafts_sold * price_per_craft + extra_earnings
  let remaining_amount := total_earnings - deposit_amount
  remaining_amount = 25 :=
by
  let price_per_craft := 12
  let crafts_sold := 3
  let extra_earnings := 7
  let deposit_amount := 18
  let total_earnings := crafts_sold * price_per_craft + extra_earnings
  let remaining_amount := total_earnings - deposit_amount
  sorry

end Hillary_left_with_amount_l1989_198927


namespace abc_sum_l1989_198947

theorem abc_sum : ∃ a b c : ℤ, 
  (∀ x : ℤ, x^2 + 13 * x + 30 = (x + a) * (x + b)) ∧ 
  (∀ x : ℤ, x^2 + 5 * x - 50 = (x + b) * (x - c)) ∧
  a + b + c = 18 := by
  sorry

end abc_sum_l1989_198947


namespace postcards_per_day_l1989_198961

variable (income_per_card total_income days : ℕ)
variable (x : ℕ)

theorem postcards_per_day
  (h1 : income_per_card = 5)
  (h2 : total_income = 900)
  (h3 : days = 6)
  (h4 : total_income = income_per_card * x * days) :
  x = 30 :=
by
  rw [h1, h2, h3] at h4
  linarith

end postcards_per_day_l1989_198961


namespace greatest_possible_difference_in_rectangles_area_l1989_198956

theorem greatest_possible_difference_in_rectangles_area :
  ∃ (l1 w1 l2 w2 l3 w3 : ℤ),
    2 * l1 + 2 * w1 = 148 ∧
    2 * l2 + 2 * w2 = 150 ∧
    2 * l3 + 2 * w3 = 152 ∧
    (∃ (A1 A2 A3 : ℤ),
      A1 = l1 * w1 ∧
      A2 = l2 * w2 ∧
      A3 = l3 * w3 ∧
      (max (abs (A1 - A2)) (max (abs (A1 - A3)) (abs (A2 - A3))) = 1372)) :=
by
  sorry

end greatest_possible_difference_in_rectangles_area_l1989_198956


namespace max_triangles_formed_l1989_198981

-- Define the triangles and their properties
structure EquilateralTriangle (α : Type) :=
(midpoint_segment : α) -- Each triangle has a segment connecting the midpoints of two sides

variables {α : Type} [OrderedSemiring α]

-- Define the condition of being mirrored horizontally
def areMirroredHorizontally (A B : EquilateralTriangle α) : Prop := 
  -- Placeholder for any formalization needed to specify mirrored horizontally
  sorry

-- Movement conditions and number of smaller triangles
def numberOfSmallerTrianglesAtMaxOverlap (A B : EquilateralTriangle α) (move_horizontally : α) : ℕ :=
  -- Placeholder function/modeling for counting triangles during movement
  sorry

-- Statement of our main theorem
theorem max_triangles_formed (A B : EquilateralTriangle α) (move_horizontally : α) 
  (h_mirrored : areMirroredHorizontally A B) :
  numberOfSmallerTrianglesAtMaxOverlap A B move_horizontally = 11 :=
sorry

end max_triangles_formed_l1989_198981


namespace subtraction_correctness_l1989_198949

theorem subtraction_correctness : 25.705 - 3.289 = 22.416 := 
by
  sorry

end subtraction_correctness_l1989_198949


namespace simplify_expression_l1989_198909

theorem simplify_expression (w : ℝ) : 3 * w + 6 * w - 9 * w + 12 * w - 15 * w + 21 = -3 * w + 21 :=
by
  sorry

end simplify_expression_l1989_198909


namespace entry_exit_ways_l1989_198996

theorem entry_exit_ways (n : ℕ) (h : n = 8) : n * (n - 1) = 56 :=
by {
  sorry
}

end entry_exit_ways_l1989_198996


namespace find_m_for_unique_solution_l1989_198965

theorem find_m_for_unique_solution :
  ∃ m : ℝ, (m = -8 + 2 * Real.sqrt 15 ∨ m = -8 - 2 * Real.sqrt 15) ∧ 
  ∀ x : ℝ, (mx - 2 ≠ 0 → (x + 3) / (mx - 2) = x + 1 ↔ ∃! x : ℝ, (mx - 2) * (x + 1) = (x + 3)) :=
sorry

end find_m_for_unique_solution_l1989_198965


namespace solve_for_other_diagonal_l1989_198936

noncomputable def length_of_other_diagonal
  (area : ℝ) (d2 : ℝ) : ℝ :=
  (2 * area) / d2

theorem solve_for_other_diagonal 
  (h_area : ℝ) (h_d2 : ℝ) (h_condition : h_area = 75 ∧ h_d2 = 15) :
  length_of_other_diagonal h_area h_d2 = 10 :=
by
  -- using h_condition, prove the required theorem
  sorry

end solve_for_other_diagonal_l1989_198936


namespace each_friend_paid_l1989_198942

def cottage_cost_per_hour : ℕ := 5
def rental_duration_hours : ℕ := 8
def total_cost := cottage_cost_per_hour * rental_duration_hours
def cost_per_person := total_cost / 2

theorem each_friend_paid : cost_per_person = 20 :=
by 
  sorry

end each_friend_paid_l1989_198942


namespace largest_undefined_x_value_l1989_198967

theorem largest_undefined_x_value :
  ∃ x : ℝ, (6 * x^2 - 65 * x + 54 = 0) ∧ (∀ y : ℝ, (6 * y^2 - 65 * y + 54 = 0) → y ≤ x) :=
sorry

end largest_undefined_x_value_l1989_198967


namespace elevator_initial_floors_down_l1989_198972

theorem elevator_initial_floors_down (x : ℕ) (h1 : 9 - x + 3 + 8 = 13) : x = 7 := 
by
  -- Proof
  sorry

end elevator_initial_floors_down_l1989_198972


namespace sum_adjacent_odd_l1989_198903

/-
  Given 2020 natural numbers written in a circle, prove that the sum of any two adjacent numbers is odd.
-/

noncomputable def numbers_in_circle : Fin 2020 → ℕ := sorry

theorem sum_adjacent_odd (k : Fin 2020) :
  (numbers_in_circle k + numbers_in_circle (k + 1)) % 2 = 1 :=
sorry

end sum_adjacent_odd_l1989_198903


namespace Adam_bought_9_cat_food_packages_l1989_198980

def num_cat_food_packages (c : ℕ) : Prop :=
  let cat_cans := 10 * c
  let dog_cans := 7 * 5
  cat_cans = dog_cans + 55

theorem Adam_bought_9_cat_food_packages : num_cat_food_packages 9 :=
by
  unfold num_cat_food_packages
  sorry

end Adam_bought_9_cat_food_packages_l1989_198980


namespace smallest_solution_l1989_198901

noncomputable def equation (x : ℝ) : Prop :=
  (1 / (x - 3) + 1 / (x - 5) + 1 / (x - 6) = 4 / (x - 4))

def valid_x (x : ℝ) : Prop :=
  x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5 ∧ x ≠ 6

theorem smallest_solution (x : ℝ) (h1 : equation x) (h2 : valid_x x) : x = 16 := sorry

end smallest_solution_l1989_198901


namespace evaluate_expression_l1989_198912

theorem evaluate_expression : 4 * (8 - 3) - 6 / 3 = 18 :=
by sorry

end evaluate_expression_l1989_198912


namespace height_of_right_triangle_on_parabola_equals_one_l1989_198991

theorem height_of_right_triangle_on_parabola_equals_one 
    (x0 x1 x2 : ℝ) 
    (h0 : x0 ≠ x1)
    (h1 : x0 ≠ x2) 
    (h2 : x1 ≠ x2) 
    (h3 : x0^2 = x1^2) 
    (h4 : x0^2 < x2^2):
    x2^2 - x0^2 = 1 := by
  sorry

end height_of_right_triangle_on_parabola_equals_one_l1989_198991


namespace time_to_reach_madison_l1989_198971

-- Definitions based on the conditions
def map_distance : ℝ := 5 -- inches
def average_speed : ℝ := 60 -- miles per hour
def map_scale : ℝ := 0.016666666666666666 -- inches per mile

-- The time taken by Pete to arrive in Madison
noncomputable def time_to_madison := map_distance / map_scale / average_speed

-- The theorem to prove
theorem time_to_reach_madison : time_to_madison = 5 := 
by
  sorry

end time_to_reach_madison_l1989_198971


namespace incorrect_conclusion_C_l1989_198986

variable {a : ℕ → ℝ} {q : ℝ}

-- Conditions
def geo_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n+1) = a n * q

theorem incorrect_conclusion_C 
  (h_geo: geo_seq a q)
  (h_cond: a 1 * a 2 < 0) : 
  a 1 * a 5 > 0 :=
by 
  sorry

end incorrect_conclusion_C_l1989_198986


namespace airlines_routes_l1989_198959

open Function

theorem airlines_routes
  (n_regions m_regions : ℕ)
  (h_n_regions : n_regions = 18)
  (h_m_regions : m_regions = 10)
  (A B : Fin n_regions → Fin n_regions → Bool)
  (h_flight : ∀ r1 r2 : Fin n_regions, r1 ≠ r2 → (A r1 r2 = true ∨ B r1 r2 = true) ∧ ¬(A r1 r2 = true ∧ B r1 r2 = true)) :
  ∃ (routes_A routes_B : List (List (Fin n_regions))),
    (∀ route ∈ routes_A, 2 ∣ route.length) ∧
    (∀ route ∈ routes_B, 2 ∣ route.length) ∧
    routes_A ≠ [] ∧
    routes_B ≠ [] :=
sorry

end airlines_routes_l1989_198959


namespace compute_expression_l1989_198900

theorem compute_expression : 3 * 3^4 - 27^63 / 27^61 = -486 := by
  sorry

end compute_expression_l1989_198900


namespace number_line_problem_l1989_198911

theorem number_line_problem (A B C : ℤ) (hA : A = -1) (hB : B = A - 5 + 6) (hC : abs (C - B) = 5) :
  C = 5 ∨ C = -5 :=
by sorry

end number_line_problem_l1989_198911


namespace uruguayan_goals_conceded_l1989_198937

theorem uruguayan_goals_conceded (x : ℕ) (h : 14 = 9 + x) : x = 5 := by
  sorry

end uruguayan_goals_conceded_l1989_198937


namespace arithmetic_geometric_progression_l1989_198905

theorem arithmetic_geometric_progression (a b c : ℤ) (h1 : a < b) (h2 : b < c)
  (h3 : b = 3 * a) (h4 : 2 * b = a + c) (h5 : b * b = a * c) : c = 9 :=
sorry

end arithmetic_geometric_progression_l1989_198905


namespace min_value_seq_l1989_198929

theorem min_value_seq (a : ℕ → ℕ) (n : ℕ) (h₁ : a 1 = 26) (h₂ : ∀ n, a (n + 1) - a n = 2 * n + 1) :
  ∃ m, (m > 0) ∧ (∀ k, k > 0 → (a k / k : ℚ) ≥ 10) ∧ (a m / m : ℚ) = 10 :=
by
  sorry

end min_value_seq_l1989_198929


namespace parabola_point_distance_l1989_198945

theorem parabola_point_distance (x y : ℝ) (h : y^2 = 2 * x) (d : ℝ) (focus_x : ℝ) (focus_y : ℝ) :
    focus_x = 1/2 → focus_y = 0 → d = 3 →
    (x + 1/2 = d) → x = 5/2 :=
by
  intros h_focus_x h_focus_y h_d h_dist
  sorry

end parabola_point_distance_l1989_198945


namespace red_window_exchange_l1989_198941

-- Defining the total transaction amount for online and offline booths
variables (x y : ℝ)

-- Defining conditions
def offlineMoreThanOnline (y x : ℝ) : Prop := y - 7 * x = 1.8
def averageTransactionDifference (y x : ℝ) : Prop := (y / 71) - (x / 44) = 0.3

-- The proof problem
theorem red_window_exchange (x y : ℝ) :
  offlineMoreThanOnline y x ∧ averageTransactionDifference y x := 
sorry

end red_window_exchange_l1989_198941


namespace maximum_value_of_func_l1989_198943

noncomputable def func (x : ℝ) : ℝ := 4 * x - 2 + 1 / (4 * x - 5)

theorem maximum_value_of_func (x : ℝ) (h : x < 5 / 4) : ∃ y, y = 1 ∧ ∀ z, z = func x → z ≤ y :=
sorry

end maximum_value_of_func_l1989_198943


namespace remainder_of_91_pow_92_mod_100_l1989_198907

theorem remainder_of_91_pow_92_mod_100 : (91 ^ 92) % 100 = 81 :=
by
  sorry

end remainder_of_91_pow_92_mod_100_l1989_198907


namespace positive_root_of_real_root_l1989_198918

theorem positive_root_of_real_root (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : b^2 - 4*a*c ≥ 0) (h2 : c^2 - 4*b*a ≥ 0) (h3 : a^2 - 4*c*b ≥ 0) : 
  ∀ (p q r : ℝ), (p = a ∧ q = b ∧ r = c) ∨ (p = b ∧ q = c ∧ r = a) ∨ (p = c ∧ q = a ∧ r = b) →
  (∃ x : ℝ, x > 0 ∧ p*x^2 + q*x + r = 0) :=
by 
  sorry

end positive_root_of_real_root_l1989_198918


namespace probability_at_least_one_die_shows_three_l1989_198908

theorem probability_at_least_one_die_shows_three : 
  let outcomes := 36
  let not_three_outcomes := 25
  (outcomes - not_three_outcomes) / outcomes = 11 / 36 := sorry

end probability_at_least_one_die_shows_three_l1989_198908


namespace song_distribution_l1989_198963

-- Let us define the necessary conditions and the result as a Lean statement.

theorem song_distribution :
    ∃ (AB BC CA A B C N : Finset ℕ),
    -- Six different songs.
    (AB ∪ BC ∪ CA ∪ A ∪ B ∪ C ∪ N) = {1, 2, 3, 4, 5, 6} ∧
    -- No song is liked by all three.
    (∀ song, ¬(song ∈ AB ∩ BC ∩ CA)) ∧
    -- Each girl dislikes at least one song.
    (N ≠ ∅) ∧
    -- For each pair of girls, at least one song liked by those two but disliked by the third.
    (AB ≠ ∅ ∧ BC ≠ ∅ ∧ CA ≠ ∅) ∧
    -- The total number of ways this can be done is 735.
    True := sorry

end song_distribution_l1989_198963


namespace remaining_flour_needed_l1989_198987

-- Define the required total amount of flour
def total_flour : ℕ := 8

-- Define the amount of flour already added
def flour_added : ℕ := 2

-- Define the remaining amount of flour needed
def remaining_flour : ℕ := total_flour - flour_added

-- The theorem we need to prove
theorem remaining_flour_needed : remaining_flour = 6 := by
  sorry

end remaining_flour_needed_l1989_198987


namespace sum_of_squares_of_first_10_primes_l1989_198940

theorem sum_of_squares_of_first_10_primes :
  ((2^2) + (3^2) + (5^2) + (7^2) + (11^2) + (13^2) + (17^2) + (19^2) + (23^2) + (29^2)) = 2397 :=
by
  sorry

end sum_of_squares_of_first_10_primes_l1989_198940


namespace sales_fifth_month_l1989_198920

theorem sales_fifth_month
  (a1 a2 a3 a4 a6 : ℕ)
  (h1 : a1 = 2435)
  (h2 : a2 = 2920)
  (h3 : a3 = 2855)
  (h4 : a4 = 3230)
  (h6 : a6 = 1000)
  (avg : ℕ)
  (h_avg : avg = 2500) :
  a1 + a2 + a3 + a4 + (15000 - 1000 - (a1 + a2 + a3 + a4)) + a6 = avg * 6 :=
by
  sorry

end sales_fifth_month_l1989_198920


namespace strands_of_duct_tape_used_l1989_198919

-- Define the conditions
def hannah_cut_rate : ℕ := 8  -- Hannah's cutting rate
def son_cut_rate : ℕ := 3     -- Son's cutting rate
def minutes : ℕ := 2          -- Time taken to free the younger son

-- Define the total cutting rate
def total_cut_rate : ℕ := hannah_cut_rate + son_cut_rate

-- Define the total number of strands
def total_strands : ℕ := total_cut_rate * minutes

-- State the theorem to prove
theorem strands_of_duct_tape_used : total_strands = 22 :=
by
  sorry

end strands_of_duct_tape_used_l1989_198919


namespace stratified_sampling_l1989_198968

theorem stratified_sampling
  (students_class1 : ℕ)
  (students_class2 : ℕ)
  (formation_slots : ℕ)
  (total_students : ℕ)
  (prob_selected: ℚ)
  (selected_class1 : ℕ)
  (selected_class2 : ℕ)
  (h1 : students_class1 = 54)
  (h2 : students_class2 = 42)
  (h3 : formation_slots = 16)
  (h4 : total_students = students_class1 + students_class2)
  (h5 : prob_selected = formation_slots / total_students)
  (h6 : selected_class1 = students_class1 * prob_selected)
  (h7 : selected_class2 = students_class2 * prob_selected)
  : selected_class1 = 9 ∧ selected_class2 = 7 := by
  sorry

end stratified_sampling_l1989_198968


namespace min_value_of_function_l1989_198946

theorem min_value_of_function (x : ℝ) (h : x > 2) : ∃ y, y = (x^2 - 4*x + 8) / (x - 2) ∧ (∀ z, z = (x^2 - 4*x + 8) / (x - 2) → y ≤ z) :=
sorry

end min_value_of_function_l1989_198946


namespace triangle_angles_geometric_progression_l1989_198989

-- Theorem: If the sides of a triangle whose angles form an arithmetic progression are in geometric progression, then all three angles are 60°.
theorem triangle_angles_geometric_progression (A B C : ℝ) (a b c : ℝ)
  (h_arith_progression : 2 * B = A + C)
  (h_sum_angles : A + B + C = 180)
  (h_geo_progression : (a / b) = (b / c))
  (h_b_angle : B = 60) :
  A = 60 ∧ B = 60 ∧ C = 60 :=
by
  sorry

end triangle_angles_geometric_progression_l1989_198989


namespace problem_conditions_l1989_198906

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x < 1 then x / (1 + x)
else if -1 < x ∧ x < 0 then x / (1 - x)
else 0

theorem problem_conditions (a b : ℝ) (x : ℝ) :
  (∀ x : ℝ, -1 < x → x < 1 → f (-x) = -f x) ∧ 
  (∀ x : ℝ, 0 ≤ x → x < 1 → f x = (-a * x - b) / (1 + x)) ∧ 
  (f (1 / 2) = 1 / 3) →
  (a = -1) ∧ (b = 0) ∧
  (∀ x :  ℝ, -1 < x ∧ x < 1 → 
    (if 0 ≤ x ∧ x < 1 then f x = x / (1 + x) else if -1 < x ∧ x < 0 then f x = x / (1 - x) else True)) ∧ 
  (∀ x1 x2 : ℝ, -1 < x1 → x1 < x2 → x2 < 1 → f x1 < f x2) ∧ 
  (∀ x : ℝ, f (x - 1) + f x > 0 → (1 / 2 < x ∧ x < 1)) :=
by
  sorry

end problem_conditions_l1989_198906


namespace factorize_quadratic_l1989_198913

theorem factorize_quadratic (x : ℝ) : 2 * x^2 + 12 * x + 18 = 2 * (x + 3)^2 :=
by
  sorry

end factorize_quadratic_l1989_198913


namespace find_a10_l1989_198938

noncomputable def a_n (a1 d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

noncomputable def S_n (a1 d : ℝ) (n : ℕ) : ℝ := n * a1 + (n * (n - 1) / 2) * d

theorem find_a10 (a1 d : ℝ)
  (h1 : a_n a1 d 2 + a_n a1 d 4 = 2)
  (h2 : S_n a1 d 2 + S_n a1 d 4 = 1) :
  a_n a1 d 10 = 8 :=
sorry

end find_a10_l1989_198938


namespace proof_problem_l1989_198988

theorem proof_problem (x : ℤ) (h : (x - 34) / 10 = 2) : (x - 5) / 7 = 7 :=
  sorry

end proof_problem_l1989_198988


namespace false_statement_l1989_198975

-- Define propositions p and q
def p := ∀ x : ℝ, (|x| = x) ↔ (x ≥ 0)
def q := ∀ (f : ℝ → ℝ), (∀ x, f (-x) = -f x) → (∃ origin : ℝ, ∀ y : ℝ, f (origin + y) = f (origin - y))

-- Define the possible answers
def option_A := p ∨ q
def option_B := p ∧ q
def option_C := ¬p ∧ q
def option_D := ¬p ∨ q

-- Define the false option (the correct answer was B)
def false_proposition := option_B

-- The statement to prove
theorem false_statement : false_proposition = false :=
by sorry

end false_statement_l1989_198975


namespace find_xyz_l1989_198966

theorem find_xyz (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 25)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 7) : 
  x * y * z = 6 := 
by 
  sorry

end find_xyz_l1989_198966


namespace isosceles_triangle_congruent_side_length_l1989_198983

theorem isosceles_triangle_congruent_side_length 
  (base : ℝ) (area : ℝ) (a b c : ℝ) 
  (h1 : a = c)
  (h2 : a = base / 2)
  (h3 : (base * a) / 2 = area)
  : b = 5 * Real.sqrt 10 := 
by sorry

end isosceles_triangle_congruent_side_length_l1989_198983


namespace intersection_of_sets_l1989_198923

def A := { x : ℝ | 0 ≤ x ∧ x ≤ 2 }
def B := { x : ℝ | x^2 > 1 }
def C := { x : ℝ | 1 < x ∧ x ≤ 2 }

theorem intersection_of_sets : 
  (A ∩ B) = C := 
by sorry

end intersection_of_sets_l1989_198923
