import Mathlib

namespace NUMINAMATH_GPT_real_roots_condition_l2019_201926

-- Definitions based on conditions
def polynomial (x : ℝ) : ℝ := x^4 - 6 * x - 1
def is_root (a : ℝ) : Prop := polynomial a = 0

-- The statement we want to prove
theorem real_roots_condition (a b : ℝ) (ha: is_root a) (hb: is_root b) : 
  (a * b + 2 * a + 2 * b = 1.5 + Real.sqrt 3) := 
sorry

end NUMINAMATH_GPT_real_roots_condition_l2019_201926


namespace NUMINAMATH_GPT_distinct_paths_to_B_and_C_l2019_201925

def paths_to_red_arrows : ℕ × ℕ := (1, 2)
def paths_from_first_red_to_blue : ℕ := 3 * 2
def paths_from_second_red_to_blue : ℕ := 4 * 2
def total_paths_to_blue_arrows : ℕ := paths_from_first_red_to_blue + paths_from_second_red_to_blue

def paths_from_first_two_blue_to_green : ℕ := 5 * 4
def paths_from_third_and_fourth_blue_to_green : ℕ := 6 * 4
def total_paths_to_green_arrows : ℕ := paths_from_first_two_blue_to_green + paths_from_third_and_fourth_blue_to_green

def paths_to_B : ℕ := total_paths_to_green_arrows * 3
def paths_to_C : ℕ := total_paths_to_green_arrows * 4
def total_paths : ℕ := paths_to_B + paths_to_C

theorem distinct_paths_to_B_and_C :
  total_paths = 4312 := 
by
  -- all conditions can be used within this proof
  sorry

end NUMINAMATH_GPT_distinct_paths_to_B_and_C_l2019_201925


namespace NUMINAMATH_GPT_Dodo_is_sane_l2019_201975

-- Declare the names of the characters
inductive Character
| Dodo : Character
| Lori : Character
| Eagle : Character

open Character

-- Definitions of sanity state
def sane (c : Character) : Prop := sorry
def insane (c : Character) : Prop := ¬ sane c

-- Conditions based on the problem statement
axiom Dodo_thinks_Lori_thinks_Eagle_not_sane : (sane Lori → insane Eagle)
axiom Lori_thinks_Dodo_not_sane : insane Dodo
axiom Eagle_thinks_Dodo_sane : sane Dodo

-- Theorem to prove Dodo is sane
theorem Dodo_is_sane : sane Dodo :=
by {
    sorry
}

end NUMINAMATH_GPT_Dodo_is_sane_l2019_201975


namespace NUMINAMATH_GPT_projectiles_initial_distance_l2019_201938

theorem projectiles_initial_distance 
  (v₁ v₂ : ℝ) (t : ℝ) (d₁ d₂ d : ℝ) 
  (hv₁ : v₁ = 445 / 60) -- speed of first projectile in km/min
  (hv₂ : v₂ = 545 / 60) -- speed of second projectile in km/min
  (ht : t = 84) -- time to meet in minutes
  (hd₁ : d₁ = v₁ * t) -- distance traveled by the first projectile
  (hd₂ : d₂ = v₂ * t) -- distance traveled by the second projectile
  (hd : d = d₁ + d₂) -- total initial distance
  : d = 1385.6 :=
by 
  sorry

end NUMINAMATH_GPT_projectiles_initial_distance_l2019_201938


namespace NUMINAMATH_GPT_Marta_max_piles_l2019_201959

theorem Marta_max_piles (a b c : ℕ) (ha : a = 42) (hb : b = 60) (hc : c = 90) : 
  Nat.gcd (Nat.gcd a b) c = 6 := by
  rw [ha, hb, hc]
  have h : Nat.gcd (Nat.gcd 42 60) 90 = Nat.gcd 6 90 := by sorry
  exact h    

end NUMINAMATH_GPT_Marta_max_piles_l2019_201959


namespace NUMINAMATH_GPT_possible_values_of_p_l2019_201969

theorem possible_values_of_p (p : ℕ) (a b : ℕ) (h_fact : (x : ℤ) → x^2 - 5 * x + p = (x - a) * (x - b))
  (h1 : a + b = 5) (h2 : 1 ≤ a ∧ a ≤ 4) (h3 : 1 ≤ b ∧ b ≤ 4) : 
  p = 4 ∨ p = 6 :=
sorry

end NUMINAMATH_GPT_possible_values_of_p_l2019_201969


namespace NUMINAMATH_GPT_saras_sister_ordered_notebooks_l2019_201931

theorem saras_sister_ordered_notebooks (x : ℕ) 
  (initial_notebooks : ℕ := 4) 
  (lost_notebooks : ℕ := 2) 
  (current_notebooks : ℕ := 8) :
  initial_notebooks + x - lost_notebooks = current_notebooks → x = 6 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_saras_sister_ordered_notebooks_l2019_201931


namespace NUMINAMATH_GPT_distances_sum_in_triangle_l2019_201913

variable (A B C O : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O]
variable (a b c P AO BO CO : ℝ)

def triangle_sides (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

def triangle_perimeter (a b c : ℝ) (P : ℝ) : Prop :=
  P = a + b + c

def point_inside_triangle (O : Type) : Prop := 
  ∃ (A B C : Type), True -- Placeholder for the actual geometric condition

def distances_to_vertices (O : Type) (AO BO CO : ℝ) : Prop := 
  AO >= 0 ∧ BO >= 0 ∧ CO >= 0

theorem distances_sum_in_triangle
  (h1 : triangle_sides a b c)
  (h2 : triangle_perimeter a b c P)
  (h3 : point_inside_triangle O)
  (h4 : distances_to_vertices O AO BO CO) :
  P / 2 < AO + BO + CO ∧ AO + BO + CO < P :=
sorry

end NUMINAMATH_GPT_distances_sum_in_triangle_l2019_201913


namespace NUMINAMATH_GPT_total_wet_surface_area_is_62_l2019_201907

-- Define the dimensions of the cistern
def length_cistern : ℝ := 8
def width_cistern : ℝ := 4
def depth_water : ℝ := 1.25

-- Define the calculation of the wet surface area
def bottom_surface_area : ℝ := length_cistern * width_cistern
def longer_side_surface_area : ℝ := length_cistern * depth_water * 2
def shorter_end_surface_area : ℝ := width_cistern * depth_water * 2

-- Sum up all wet surface areas
def total_wet_surface_area : ℝ := bottom_surface_area + longer_side_surface_area + shorter_end_surface_area

-- The theorem stating that the total wet surface area is 62 m²
theorem total_wet_surface_area_is_62 : total_wet_surface_area = 62 := by
  sorry

end NUMINAMATH_GPT_total_wet_surface_area_is_62_l2019_201907


namespace NUMINAMATH_GPT_maximize_net_income_l2019_201955

noncomputable def net_income (x : ℕ) : ℤ :=
  if 60 ≤ x ∧ x ≤ 90 then 750 * x - 1700
  else if 90 < x ∧ x ≤ 300 then -3 * x * x + 1020 * x - 1700
  else 0

theorem maximize_net_income :
  (∀ x : ℕ, 60 ≤ x ∧ x ≤ 300 →
    net_income x ≤ net_income 170) ∧
  net_income 170 = 85000 := 
sorry

end NUMINAMATH_GPT_maximize_net_income_l2019_201955


namespace NUMINAMATH_GPT_largest_divisor_of_expression_l2019_201968

theorem largest_divisor_of_expression (n : ℤ) : ∃ k, ∀ n : ℤ, n^4 - n^2 = k * 12 :=
by sorry

end NUMINAMATH_GPT_largest_divisor_of_expression_l2019_201968


namespace NUMINAMATH_GPT_factor_expression_l2019_201996

theorem factor_expression (x : ℝ) : 54 * x^5 - 135 * x^9 = 27 * x^5 * (2 - 5 * x^4) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l2019_201996


namespace NUMINAMATH_GPT_x_y_solution_l2019_201924

variable (x y : ℕ)

noncomputable def x_wang_speed : ℕ := x - 6

theorem x_y_solution (hx : (5 : ℚ) / 6 * x = y) (hy : (2 : ℚ) / 3 * (x - 6) = y - 10) : x = 36 ∧ y = 30 :=
by {
  sorry
}

end NUMINAMATH_GPT_x_y_solution_l2019_201924


namespace NUMINAMATH_GPT_range_values_y_div_x_l2019_201918

-- Given the equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 6 * y + 12 = 0

-- Prove that the range of values for y / x is [ (6 - 2 * sqrt 3) / 3, (6 + 2 * sqrt 3) / 3 ]
theorem range_values_y_div_x :
  (∀ x y : ℝ, circle_eq x y → (∃ k : ℝ, y = k * x) → 
  ( (6 - 2 * Real.sqrt 3) / 3 ≤ y / x ∧ y / x ≤ (6 + 2 * Real.sqrt 3) / 3 )) :=
sorry

end NUMINAMATH_GPT_range_values_y_div_x_l2019_201918


namespace NUMINAMATH_GPT_intersection_eq_T_l2019_201954

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := sorry

end NUMINAMATH_GPT_intersection_eq_T_l2019_201954


namespace NUMINAMATH_GPT_Tn_lt_half_Sn_l2019_201984

noncomputable def a_n (n : ℕ) : ℝ := (1/3)^(n-1)
noncomputable def b_n (n : ℕ) : ℝ := n * (1/3)^n
noncomputable def S_n (n : ℕ) : ℝ := 3/2 - 1/2 * (1/3)^(n-1)
noncomputable def T_n (n : ℕ) : ℝ := 3/4 - 1/4 * (1/3)^(n-1) - n/2 * (1/3)^n

theorem Tn_lt_half_Sn (n : ℕ) : T_n n < S_n n / 2 :=
by
  sorry

end NUMINAMATH_GPT_Tn_lt_half_Sn_l2019_201984


namespace NUMINAMATH_GPT_set_inter_complement_U_B_l2019_201902

-- Define sets U, A, B
def U : Set ℝ := {x | x ≤ -1 ∨ x ≥ 0}
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x^2 > 1}

-- Statement to prove
theorem set_inter_complement_U_B :
  A ∩ (Uᶜ \ B) = {x | 0 ≤ x ∧ x ≤ 1} :=
by
  sorry

end NUMINAMATH_GPT_set_inter_complement_U_B_l2019_201902


namespace NUMINAMATH_GPT_claire_gift_card_balance_l2019_201991

/--
Claire has a $100 gift card to her favorite coffee shop.
A latte costs $3.75.
A croissant costs $3.50.
Claire buys one latte and one croissant every day for a week.
Claire buys 5 cookies, each costing $1.25.

Prove that the amount of money left on Claire's gift card after a week is $43.00.
-/
theorem claire_gift_card_balance :
  let initial_balance : ℝ := 100
  let latte_cost : ℝ := 3.75
  let croissant_cost : ℝ := 3.50
  let daily_expense : ℝ := latte_cost + croissant_cost
  let weekly_expense : ℝ := daily_expense * 7
  let cookie_cost : ℝ := 1.25
  let total_cookie_expense : ℝ := cookie_cost * 5
  let total_expense : ℝ := weekly_expense + total_cookie_expense
  let remaining_balance : ℝ := initial_balance - total_expense
  remaining_balance = 43 :=
by
  sorry

end NUMINAMATH_GPT_claire_gift_card_balance_l2019_201991


namespace NUMINAMATH_GPT_family_members_to_pay_l2019_201948

theorem family_members_to_pay :
  (∃ (n : ℕ), 
    5 * 12 = 60 ∧ 
    60 * 2 = 120 ∧ 
    120 / 10 = 12 ∧ 
    12 * 2 = 24 ∧ 
    24 / 4 = n ∧ 
    n = 6) :=
by
  sorry

end NUMINAMATH_GPT_family_members_to_pay_l2019_201948


namespace NUMINAMATH_GPT_no_solution_exists_l2019_201952

theorem no_solution_exists (x y : ℝ) : ¬ ((2 * x - 3 * y = 8) ∧ (6 * y - 4 * x = 9)) :=
sorry

end NUMINAMATH_GPT_no_solution_exists_l2019_201952


namespace NUMINAMATH_GPT_valid_parametrizations_l2019_201962

-- Definitions for the given points and directions
def pointA := (0, 4)
def dirA := (3, -1)

def pointB := (4/3, 0)
def dirB := (1, -3)

def pointC := (-2, 10)
def dirC := (-3, 9)

-- Line equation definition
def line (x y : ℝ) : Prop := y = -3 * x + 4

-- Proof statement
theorem valid_parametrizations :
  (line pointB.1 pointB.2 ∧ dirB.2 = -3 * dirB.1) ∧
  (line pointC.1 pointC.2 ∧ dirC.2 / dirC.1 = 3) :=
by
  sorry

end NUMINAMATH_GPT_valid_parametrizations_l2019_201962


namespace NUMINAMATH_GPT_first_discount_correct_l2019_201917

noncomputable def first_discount (x : ℝ) : Prop :=
  let initial_price := 600
  let first_discounted_price := initial_price * (1 - x / 100)
  let final_price := first_discounted_price * (1 - 0.05)
  final_price = 456

theorem first_discount_correct : ∃ x : ℝ, first_discount x ∧ abs (x - 57.29) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_first_discount_correct_l2019_201917


namespace NUMINAMATH_GPT_gcd_lcm_relation_gcd3_lcm3_relation_lcm3_gcd3_relation_l2019_201989

-- GCD as the greatest common divisor
def GCD (a b : ℕ) : ℕ := Nat.gcd a b

-- LCM as the least common multiple
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

-- First proof problem in Lean 4
theorem gcd_lcm_relation (a b : ℕ) : GCD a b = (a * b) / (LCM a b) :=
  sorry

-- GCD function extended to three arguments
def GCD3 (a b c : ℕ) : ℕ := Nat.gcd (Nat.gcd a b) c

-- LCM function extended to three arguments
def LCM3 (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

-- Second proof problem in Lean 4
theorem gcd3_lcm3_relation (a b c : ℕ) : GCD3 a b c = (a * b * c * LCM3 a b c) / (LCM a b * LCM b c * LCM c a) :=
  sorry

-- Third proof problem in Lean 4
theorem lcm3_gcd3_relation (a b c : ℕ) : LCM3 a b c = (a * b * c * GCD3 a b c) / (GCD a b * GCD b c * GCD c a) :=
  sorry

end NUMINAMATH_GPT_gcd_lcm_relation_gcd3_lcm3_relation_lcm3_gcd3_relation_l2019_201989


namespace NUMINAMATH_GPT_count_positive_integers_satisfying_conditions_l2019_201928

theorem count_positive_integers_satisfying_conditions :
  let condition1 (n : ℕ) := (169 * n) ^ 25 > n ^ 75
  let condition2 (n : ℕ) := n ^ 75 > 3 ^ 150
  ∃ (count : ℕ), count = 3 ∧ (∀ (n : ℕ), (condition1 n) ∧ (condition2 n) → 9 < n ∧ n < 13) :=
by
  sorry

end NUMINAMATH_GPT_count_positive_integers_satisfying_conditions_l2019_201928


namespace NUMINAMATH_GPT_MN_squared_l2019_201922

theorem MN_squared (PQ QR RS SP : ℝ) (h1 : PQ = 15) (h2 : QR = 15) (h3 : RS = 20) (h4 : SP = 20) (angle_S : ℝ) (h5 : angle_S = 90)
(M N: ℝ) (Midpoint_M : M = (QR / 2)) (Midpoint_N : N = (SP / 2)) : 
MN^2 = 100 := by
  sorry

end NUMINAMATH_GPT_MN_squared_l2019_201922


namespace NUMINAMATH_GPT_find_pastries_made_l2019_201915

variable (cakes_made pastries_made total_cakes_sold total_pastries_sold extra_cakes more_cakes_than_pastries : ℕ)

def baker_conditions := (cakes_made = 157) ∧ 
                        (total_cakes_sold = 158) ∧ 
                        (total_pastries_sold = 147) ∧ 
                        (more_cakes_than_pastries = 11) ∧ 
                        (extra_cakes = total_cakes_sold - cakes_made) ∧ 
                        (pastries_made = cakes_made - more_cakes_than_pastries)

theorem find_pastries_made : 
  baker_conditions cakes_made pastries_made total_cakes_sold total_pastries_sold extra_cakes more_cakes_than_pastries → 
  pastries_made = 146 :=
by
  sorry

end NUMINAMATH_GPT_find_pastries_made_l2019_201915


namespace NUMINAMATH_GPT_F_sum_l2019_201972

noncomputable def f : ℝ → ℝ := sorry -- even function f(x)
noncomputable def F (x a c : ℝ) : ℝ := 
  let b := (a + c) / 2
  (x - b) * f (x - b) + 2016

theorem F_sum (a c : ℝ) : F a a c + F c a c = 4032 := 
by {
  sorry
}

end NUMINAMATH_GPT_F_sum_l2019_201972


namespace NUMINAMATH_GPT_correct_exponent_operation_l2019_201932

theorem correct_exponent_operation (a b : ℝ) : 
  a^2 * a^3 = a^5 := 
by sorry

end NUMINAMATH_GPT_correct_exponent_operation_l2019_201932


namespace NUMINAMATH_GPT_cube_root_expression_l2019_201901

variable (x : ℝ)

theorem cube_root_expression (h : x + 1 / x = 7) : x^3 + 1 / x^3 = 322 :=
  sorry

end NUMINAMATH_GPT_cube_root_expression_l2019_201901


namespace NUMINAMATH_GPT_inequality_solution_set_l2019_201953

theorem inequality_solution_set (x : ℝ) :
  2 * x^2 - x ≤ 0 → 0 ≤ x ∧ x ≤ 1 / 2 :=
sorry

end NUMINAMATH_GPT_inequality_solution_set_l2019_201953


namespace NUMINAMATH_GPT_cost_of_airplane_l2019_201914

theorem cost_of_airplane (amount : ℝ) (change : ℝ) (h_amount : amount = 5) (h_change : change = 0.72) : 
  amount - change = 4.28 := 
by
  sorry

end NUMINAMATH_GPT_cost_of_airplane_l2019_201914


namespace NUMINAMATH_GPT_kelly_gave_away_games_l2019_201945

theorem kelly_gave_away_games (initial_games : ℕ) (remaining_games : ℕ) (given_away_games : ℕ) 
  (h1 : initial_games = 183) 
  (h2 : remaining_games = 92) 
  (h3 : given_away_games = initial_games - remaining_games) : 
  given_away_games = 91 := 
by 
  sorry

end NUMINAMATH_GPT_kelly_gave_away_games_l2019_201945


namespace NUMINAMATH_GPT_minimum_students_to_share_birthday_l2019_201988

theorem minimum_students_to_share_birthday (k : ℕ) (m : ℕ) (n : ℕ) (hcond1 : k = 366) (hcond2 : m = 2) (hineq : n > k * m) : n ≥ 733 := 
by
  -- since k = 366 and m = 2
  have hk : k = 366 := hcond1
  have hm : m = 2 := hcond2
  -- thus: n > 366 * 2
  have hn : n > 732 := by
    rw [hk, hm] at hineq
    exact hineq
  -- hence, n ≥ 733
  exact Nat.succ_le_of_lt hn

end NUMINAMATH_GPT_minimum_students_to_share_birthday_l2019_201988


namespace NUMINAMATH_GPT_victoria_initial_money_l2019_201978

-- Definitions based on conditions
def cost_rice := 2 * 20
def cost_flour := 3 * 25
def cost_soda := 150
def total_spent := cost_rice + cost_flour + cost_soda
def remaining_balance := 235

-- Theorem to prove
theorem victoria_initial_money (initial_money : ℕ) :
  initial_money = total_spent + remaining_balance :=
by
  sorry

end NUMINAMATH_GPT_victoria_initial_money_l2019_201978


namespace NUMINAMATH_GPT_find_f_of_3_l2019_201920

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  x^7 + a*x^5 + b*x - 5

theorem find_f_of_3 (a b : ℝ) (h : f (-3) a b = 5) : f 3 a b = -15 := by
  sorry

end NUMINAMATH_GPT_find_f_of_3_l2019_201920


namespace NUMINAMATH_GPT_ratio_of_expenditures_l2019_201935

variable (Rajan_income Balan_income Rajan_expenditure Balan_expenditure Rajan_savings Balan_savings: ℤ)
variable (ratio_incomes: ℚ)
variable (savings_amount: ℤ)

-- Given conditions
def conditions : Prop :=
  Rajan_income = 7000 ∧
  ratio_incomes = 7 / 6 ∧
  savings_amount = 1000 ∧
  Rajan_savings = Rajan_income - Rajan_expenditure ∧
  Balan_savings = Balan_income - Balan_expenditure ∧
  Rajan_savings = savings_amount ∧
  Balan_savings = savings_amount

-- The theorem we want to prove
theorem ratio_of_expenditures :
  conditions Rajan_income Balan_income Rajan_expenditure Balan_expenditure Rajan_savings Balan_savings ratio_incomes savings_amount →
  (Rajan_expenditure : ℚ) / (Balan_expenditure : ℚ) = 6 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_expenditures_l2019_201935


namespace NUMINAMATH_GPT_sides_and_diagonals_l2019_201961

def number_of_sides_of_polygon (n : ℕ) :=
  180 * (n - 2) = 360 + (1 / 4 : ℤ) * 360

def number_of_diagonals_of_polygon (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem sides_and_diagonals : 
  (∃ n : ℕ, number_of_sides_of_polygon n ∧ n = 12) ∧ number_of_diagonals_of_polygon 12 = 54 :=
by {
  -- Proof will be filled in later
  sorry
}

end NUMINAMATH_GPT_sides_and_diagonals_l2019_201961


namespace NUMINAMATH_GPT_exists_n_such_that_not_square_l2019_201977

theorem exists_n_such_that_not_square : ∃ n : ℕ, n > 1 ∧ ¬(∃ k : ℕ, k ^ 2 = 2 ^ (2 ^ n - 1) - 7) := 
sorry

end NUMINAMATH_GPT_exists_n_such_that_not_square_l2019_201977


namespace NUMINAMATH_GPT_solve_quadratic_l2019_201994

theorem solve_quadratic (x : ℝ) : (x^2 + x)^2 + (x^2 + x) - 6 = 0 ↔ x = -2 ∨ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_l2019_201994


namespace NUMINAMATH_GPT_lemonade_served_l2019_201965

def glasses_per_pitcher : ℕ := 5
def number_of_pitchers : ℕ := 6
def total_glasses_served : ℕ := glasses_per_pitcher * number_of_pitchers

theorem lemonade_served : total_glasses_served = 30 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_lemonade_served_l2019_201965


namespace NUMINAMATH_GPT_find_pos_ints_a_b_c_p_l2019_201964

theorem find_pos_ints_a_b_c_p (a b c p : ℕ) (hp : Nat.Prime p) : 
  73 * p^2 + 6 = 9 * a^2 + 17 * b^2 + 17 * c^2 ↔
  (p = 2 ∧ a = 1 ∧ b = 4 ∧ c = 1) ∨ (p = 2 ∧ a = 1 ∧ b = 1 ∧ c = 4) :=
by
  sorry

end NUMINAMATH_GPT_find_pos_ints_a_b_c_p_l2019_201964


namespace NUMINAMATH_GPT_minimum_total_length_of_removed_segments_l2019_201939

-- Definitions based on conditions
def right_angled_triangle_sides : Nat × Nat × Nat := (3, 4, 5)

def large_square_side_length : Nat := 7

-- Statement of the problem to be proved
theorem minimum_total_length_of_removed_segments
  (triangles : Fin 4 → (Nat × Nat × Nat) := fun _ => right_angled_triangle_sides)
  (side_length_of_large_square : Nat := large_square_side_length) :
  ∃ (removed_length : Nat), removed_length = 7 :=
sorry

end NUMINAMATH_GPT_minimum_total_length_of_removed_segments_l2019_201939


namespace NUMINAMATH_GPT_domain_of_function_l2019_201910

theorem domain_of_function :
  {x : ℝ | 2 - x > 0 ∧ 1 + x > 0} = {x : ℝ | -1 < x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l2019_201910


namespace NUMINAMATH_GPT_sophia_fraction_of_book_finished_l2019_201936

variable (x : ℕ)

theorem sophia_fraction_of_book_finished (h1 : x + (x + 90) = 270) : (x + 90) / 270 = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_sophia_fraction_of_book_finished_l2019_201936


namespace NUMINAMATH_GPT_no_integer_a_for_integer_roots_l2019_201979

theorem no_integer_a_for_integer_roots :
  ∀ a : ℤ, ¬ (∃ x : ℤ, x^2 - 2023 * x + 2022 * a + 1 = 0) := 
by
  intro a
  rintro ⟨x, hx⟩
  sorry

end NUMINAMATH_GPT_no_integer_a_for_integer_roots_l2019_201979


namespace NUMINAMATH_GPT_does_not_pass_through_third_quadrant_l2019_201987

theorem does_not_pass_through_third_quadrant :
  ¬ ∃ (x y : ℝ), 2 * x + 3 * y = 5 ∧ x < 0 ∧ y < 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_does_not_pass_through_third_quadrant_l2019_201987


namespace NUMINAMATH_GPT_smallest_b_for_fourth_power_l2019_201980

noncomputable def is_fourth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k ^ 4 = n

theorem smallest_b_for_fourth_power :
  ∃ b : ℕ, (0 < b) ∧ (7 + 7 * b + 7 * b ^ 2 = (7 * 1 + 7 * 18 + 7 * 18 ^ 2)) 
  ∧ is_fourth_power (7 + 7 * b + 7 * b ^ 2) := sorry

end NUMINAMATH_GPT_smallest_b_for_fourth_power_l2019_201980


namespace NUMINAMATH_GPT_additional_seasons_is_one_l2019_201906

-- Definitions for conditions
def episodes_per_season : Nat := 22
def episodes_last_season : Nat := episodes_per_season + 4
def episodes_in_9_seasons : Nat := 9 * episodes_per_season
def hours_per_episode : Nat := 1 / 2 -- Stored as half units

-- Given conditions
def total_hours_to_watch_after_last_season: Nat := 112 * 2 -- converted to half-hours
def time_watched_in_9_seasons: Nat := episodes_in_9_seasons * hours_per_episode
def additional_hours: Nat := total_hours_to_watch_after_last_season - time_watched_in_9_seasons

-- Theorem to prove
theorem additional_seasons_is_one : additional_hours / hours_per_episode = episodes_last_season -> 
      additional_hours / hours_per_episode / episodes_per_season = 1 :=
by
  sorry

end NUMINAMATH_GPT_additional_seasons_is_one_l2019_201906


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l2019_201963

open Real

theorem problem_part1 (α : ℝ) (h : (sin (π - α) * cos (2 * π - α)) / (tan (π - α) * sin (π / 2 + α) * cos (π / 2 - α)) = 1 / 2) :
  (cos α - 2 * sin α) / (3 * cos α + sin α) = 5 := sorry

theorem problem_part2 (α : ℝ) (h : tan α = -2) :
  1 - 2 * sin α * cos α + cos α ^ 2 = 2 / 5 := sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l2019_201963


namespace NUMINAMATH_GPT_train_average_speed_l2019_201949

theorem train_average_speed (x : ℝ) (h1 : x > 0) :
  let d1 := x
  let d2 := 2 * x
  let s1 := 50
  let s2 := 20
  let t1 := d1 / s1
  let t2 := d2 / s2
  let total_distance := d1 + d2
  let total_time := t1 + t2
  let average_speed := total_distance / total_time
  average_speed = 25 := 
by
  sorry

end NUMINAMATH_GPT_train_average_speed_l2019_201949


namespace NUMINAMATH_GPT_sweets_neither_red_nor_green_l2019_201933

theorem sweets_neither_red_nor_green (total_sweets red_sweets green_sweets : ℕ)
  (h1 : total_sweets = 285)
  (h2 : red_sweets = 49)
  (h3 : green_sweets = 59) : total_sweets - (red_sweets + green_sweets) = 177 :=
by
  sorry

end NUMINAMATH_GPT_sweets_neither_red_nor_green_l2019_201933


namespace NUMINAMATH_GPT_distinct_triangles_in_3x3_grid_l2019_201903

theorem distinct_triangles_in_3x3_grid : 
  let num_points := 9 
  let total_combinations := Nat.choose num_points 3 
  let degenerate_cases := 8
  total_combinations - degenerate_cases = 76 := 
by
  sorry

end NUMINAMATH_GPT_distinct_triangles_in_3x3_grid_l2019_201903


namespace NUMINAMATH_GPT_sections_capacity_l2019_201937

theorem sections_capacity (total_people sections : ℕ) 
  (h1 : total_people = 984) 
  (h2 : sections = 4) : 
  total_people / sections = 246 := 
by
  sorry

end NUMINAMATH_GPT_sections_capacity_l2019_201937


namespace NUMINAMATH_GPT_no_solution_ineq_range_a_l2019_201976

theorem no_solution_ineq_range_a (a : ℝ) :
  (∀ x : ℝ, x^2 + a * x + 4 < 0 → false) ↔ (-4 ≤ a ∧ a ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_no_solution_ineq_range_a_l2019_201976


namespace NUMINAMATH_GPT_katie_travel_distance_l2019_201912

theorem katie_travel_distance (d_train d_bus d_bike d_car d_total d1 d2 d3 : ℕ)
  (h1 : d_train = 162)
  (h2 : d_bus = 124)
  (h3 : d_bike = 88)
  (h4 : d_car = 224)
  (h_total : d_total = d_train + d_bus + d_bike + d_car)
  (h1_distance : d1 = 96)
  (h2_distance : d2 = 108)
  (h3_distance : d3 = 130)
  (h1_prob : 30 = 30)
  (h2_prob : 50 = 50)
  (h3_prob : 20 = 20) :
  (d_total + d1 = 694) ∧
  (d_total + d2 = 706) ∧
  (d_total + d3 = 728) :=
sorry

end NUMINAMATH_GPT_katie_travel_distance_l2019_201912


namespace NUMINAMATH_GPT_find_inlet_rate_l2019_201916

-- definitions for the given conditions
def volume_cubic_feet : ℝ := 20
def conversion_factor : ℝ := 12^3
def volume_cubic_inches : ℝ := volume_cubic_feet * conversion_factor

def outlet_rate1 : ℝ := 9
def outlet_rate2 : ℝ := 8
def empty_time : ℕ := 2880

-- theorem that captures the proof problem
theorem find_inlet_rate (volume_cubic_inches : ℝ) (outlet_rate1 outlet_rate2 empty_time : ℝ) :
  ∃ (inlet_rate : ℝ), volume_cubic_inches = (outlet_rate1 + outlet_rate2 - inlet_rate) * empty_time ↔ inlet_rate = 5 := 
by
  sorry

end NUMINAMATH_GPT_find_inlet_rate_l2019_201916


namespace NUMINAMATH_GPT_semicircle_area_in_quarter_circle_l2019_201909

theorem semicircle_area_in_quarter_circle (r : ℝ) (A : ℝ) (π : ℝ) (one : ℝ) :
    r = 1 / (Real.sqrt (2) + 1) →
    A = π * r^2 →
    120 * A / π = 20 :=
sorry

end NUMINAMATH_GPT_semicircle_area_in_quarter_circle_l2019_201909


namespace NUMINAMATH_GPT_circle_radius_is_2_chord_length_is_2sqrt3_l2019_201943

-- Define the given conditions
def inclination_angle_line_incl60 : Prop := ∃ m, m = Real.sqrt 3
def circle_eq : Prop := ∀ x y, x^2 + y^2 - 4 * y = 0

-- Prove: radius of the circle
theorem circle_radius_is_2 (h : circle_eq) : radius = 2 := sorry

-- Prove: length of the chord cut by the line
theorem chord_length_is_2sqrt3 
  (h1 : inclination_angle_line_incl60) 
  (h2 : circle_eq) : chord_length = 2 * Real.sqrt 3 := sorry

end NUMINAMATH_GPT_circle_radius_is_2_chord_length_is_2sqrt3_l2019_201943


namespace NUMINAMATH_GPT_train_b_leaves_after_train_a_l2019_201941

noncomputable def time_difference := 2

theorem train_b_leaves_after_train_a 
  (speedA speedB distance t : ℝ) 
  (h1 : speedA = 30)
  (h2 : speedB = 38)
  (h3 : distance = 285)
  (h4 : distance = speedB * t)
  : time_difference = (distance - speedA * t) / speedA := 
by 
  sorry

end NUMINAMATH_GPT_train_b_leaves_after_train_a_l2019_201941


namespace NUMINAMATH_GPT_correct_algebraic_expression_l2019_201971

theorem correct_algebraic_expression
  (A : String := "1 1/2 a")
  (B : String := "a × b")
  (C : String := "a ÷ b")
  (D : String := "2a") :
  D = "2a" :=
by {
  -- Explanation based on the conditions provided
  -- A: "1 1/2 a" is not properly formatted. Correct format involves improper fraction for multiplication.
  -- B: "a × b" should avoid using the multiplication sign explicitly.
  -- C: "a ÷ b" should be written as a fraction a/b.
  -- D: "2a" is correctly formatted.
  sorry
}

end NUMINAMATH_GPT_correct_algebraic_expression_l2019_201971


namespace NUMINAMATH_GPT_common_chord_eqn_circle_with_center_on_line_smallest_area_circle_l2019_201951

noncomputable def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 8 = 0
noncomputable def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 10*y - 24 = 0

theorem common_chord_eqn :
  ∀ x y : ℝ, (circle1 x y ∧ circle2 x y) ↔ (x - 2*y + 4 = 0) :=
sorry

noncomputable def A : ℝ × ℝ := (-4, 0)
noncomputable def B : ℝ × ℝ := (0, 2)
noncomputable def line_y_eq_neg_x (x y : ℝ) : Prop := y = -x

theorem circle_with_center_on_line :
  ∃ (x y : ℝ), line_y_eq_neg_x x y ∧ ((x + 3)^2 + (y - 3)^2 = 10) :=
sorry

theorem smallest_area_circle :
  ∃ (x y : ℝ), ((x + 2)^2 + (y - 1)^2 = 5) :=
sorry

end NUMINAMATH_GPT_common_chord_eqn_circle_with_center_on_line_smallest_area_circle_l2019_201951


namespace NUMINAMATH_GPT_quadratic_square_binomial_l2019_201944

theorem quadratic_square_binomial (k : ℝ) : 
  (∃ a : ℝ, (x : ℝ) → x^2 - 20 * x + k = (x + a)^2) ↔ k = 100 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_square_binomial_l2019_201944


namespace NUMINAMATH_GPT_problem1_problem2_l2019_201927

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2019_201927


namespace NUMINAMATH_GPT_solve_for_x_l2019_201911

theorem solve_for_x (n m x : ℕ) (h1 : 5 / 7 = n / 91) (h2 : 5 / 7 = (m + n) / 105) (h3 : 5 / 7 = (x - m) / 140) :
    x = 110 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l2019_201911


namespace NUMINAMATH_GPT_fraction_comparison_l2019_201957

noncomputable def one_seventh : ℚ := 1 / 7
noncomputable def decimal_0_point_14285714285 : ℚ := 14285714285 / 10^11
noncomputable def eps_1 : ℚ := 1 / (7 * 10^11)
noncomputable def eps_2 : ℚ := 1 / (7 * 10^12)

theorem fraction_comparison :
  one_seventh = decimal_0_point_14285714285 + eps_1 :=
sorry

end NUMINAMATH_GPT_fraction_comparison_l2019_201957


namespace NUMINAMATH_GPT_smallest_b_l2019_201960

theorem smallest_b (b : ℝ) : b^2 - 16 * b + 63 ≤ 0 → (∃ b : ℝ, b = 7) :=
sorry

end NUMINAMATH_GPT_smallest_b_l2019_201960


namespace NUMINAMATH_GPT_parabola_transformation_zeros_sum_l2019_201997

theorem parabola_transformation_zeros_sum :
  let y := fun x => (x - 3)^2 + 4
  let y_rotated := fun x => -(x - 3)^2 + 4
  let y_shifted_right := fun x => -(x - 7)^2 + 4
  let y_final := fun x => -(x - 7)^2 + 7
  ∃ a b, y_final a = 0 ∧ y_final b = 0 ∧ (a + b) = 14 :=
by
  sorry

end NUMINAMATH_GPT_parabola_transformation_zeros_sum_l2019_201997


namespace NUMINAMATH_GPT_inequality_solution_l2019_201929

noncomputable def operation (a b : ℝ) : ℝ := (a + 3 * b) - a * b

theorem inequality_solution (x : ℝ) : operation 5 x < 13 → x > -4 := by
  sorry

end NUMINAMATH_GPT_inequality_solution_l2019_201929


namespace NUMINAMATH_GPT_total_balloons_l2019_201999

theorem total_balloons (T : ℕ) 
    (h1 : T / 4 = 100)
    : T = 400 := 
by
  sorry

end NUMINAMATH_GPT_total_balloons_l2019_201999


namespace NUMINAMATH_GPT_colton_share_l2019_201998

-- Definitions
def footToInch (foot : ℕ) : ℕ := 12 * foot -- 1 foot equals 12 inches

-- Problem conditions
def coltonBurgerLength := footToInch 1 -- Colton bought a foot long burger
def sharedBurger (length : ℕ) : ℕ := length / 2 -- shared half with his brother

-- Equivalent proof problem statement
theorem colton_share : sharedBurger coltonBurgerLength = 6 := 
by sorry

end NUMINAMATH_GPT_colton_share_l2019_201998


namespace NUMINAMATH_GPT_table_length_l2019_201982

theorem table_length (area_m2 : ℕ) (width_cm : ℕ) (length_cm : ℕ) 
  (h_area : area_m2 = 54)
  (h_width : width_cm = 600)
  :
  length_cm = 900 := 
  sorry

end NUMINAMATH_GPT_table_length_l2019_201982


namespace NUMINAMATH_GPT_value_of_m_l2019_201930

noncomputable def A (m : ℝ) : Set ℝ := {3, m}
noncomputable def B (m : ℝ) : Set ℝ := {3 * m, 3}

theorem value_of_m (m : ℝ) (h : A m = B m) : m = 0 :=
by
  sorry

end NUMINAMATH_GPT_value_of_m_l2019_201930


namespace NUMINAMATH_GPT_john_allowance_spent_l2019_201990

theorem john_allowance_spent (B t d : ℝ) (h1 : t = 0.25 * (B - d)) (h2 : d = 0.10 * (B - t)) :
  (t + d) / B = 0.31 := by
  sorry

end NUMINAMATH_GPT_john_allowance_spent_l2019_201990


namespace NUMINAMATH_GPT_sin_inv_tan_eq_l2019_201908

open Real

theorem sin_inv_tan_eq :
  let a := arcsin (4/5)
  let b := arctan 3
  sin (a + b) = (13 * sqrt 10) / 50 := 
by
  let a := arcsin (4/5)
  let b := arctan 3
  sorry

end NUMINAMATH_GPT_sin_inv_tan_eq_l2019_201908


namespace NUMINAMATH_GPT_sixty_percent_of_fifty_minus_forty_percent_of_thirty_l2019_201947

theorem sixty_percent_of_fifty_minus_forty_percent_of_thirty : 
  (0.6 * 50) - (0.4 * 30) = 18 :=
by
  sorry

end NUMINAMATH_GPT_sixty_percent_of_fifty_minus_forty_percent_of_thirty_l2019_201947


namespace NUMINAMATH_GPT_intersection_S_T_l2019_201995

def set_S : Set ℝ := { x | abs x < 5 }
def set_T : Set ℝ := { x | x^2 + 4*x - 21 < 0 }

theorem intersection_S_T :
  set_S ∩ set_T = { x | -5 < x ∧ x < 3 } :=
sorry

end NUMINAMATH_GPT_intersection_S_T_l2019_201995


namespace NUMINAMATH_GPT_num_students_left_l2019_201985

variable (Joe_weight : ℝ := 45)
variable (original_avg_weight : ℝ := 30)
variable (new_avg_weight : ℝ := 31)
variable (final_avg_weight : ℝ := 30)
variable (diff_avg_weight : ℝ := 7.5)

theorem num_students_left (n : ℕ) (x : ℕ) (W : ℝ := n * original_avg_weight)
  (new_W : ℝ := W + Joe_weight) (A : ℝ := Joe_weight - diff_avg_weight) : 
  new_W = (n + 1) * new_avg_weight →
  W + Joe_weight - x * A = (n + 1 - x) * final_avg_weight →
  x = 2 :=
by
  sorry

end NUMINAMATH_GPT_num_students_left_l2019_201985


namespace NUMINAMATH_GPT_cube_surface_area_with_holes_l2019_201958

theorem cube_surface_area_with_holes 
    (edge_length : ℝ) 
    (hole_side_length : ℝ) 
    (num_faces : ℕ) 
    (parallel_edges : Prop)
    (holes_centered : Prop)
    (h_edge : edge_length = 5)
    (h_hole : hole_side_length = 2)
    (h_faces : num_faces = 6)
    (h_inside_area : parallel_edges ∧ holes_centered)
    : (150 - 24 + 96 = 222) :=
by
    sorry

end NUMINAMATH_GPT_cube_surface_area_with_holes_l2019_201958


namespace NUMINAMATH_GPT_length_of_CD_l2019_201956

theorem length_of_CD (C D R S : ℝ) 
  (h1 : R = C + 3/8 * (D - C))
  (h2 : S = C + 4/11 * (D - C))
  (h3 : |S - R| = 3) :
  D - C = 264 := 
sorry

end NUMINAMATH_GPT_length_of_CD_l2019_201956


namespace NUMINAMATH_GPT_trains_speed_ratio_l2019_201983

-- Define the conditions
variables (V1 V2 L1 L2 : ℝ)
axiom time1 : L1 = 27 * V1
axiom time2 : L2 = 17 * V2
axiom timeTogether : L1 + L2 = 22 * (V1 + V2)

-- The theorem to prove the ratio of the speeds
theorem trains_speed_ratio : V1 / V2 = 7.8 :=
sorry

end NUMINAMATH_GPT_trains_speed_ratio_l2019_201983


namespace NUMINAMATH_GPT_sector_area_angle_1_sector_max_area_l2019_201974

-- The definition and conditions
variable (c : ℝ) (r l : ℝ)

-- 1) Proof that the area of the sector when the central angle is 1 radian is c^2 / 18
-- given 2r + l = c
theorem sector_area_angle_1 (h : 2 * r + l = c) (h1: l = r) :
  (1/2 * l * r = (c^2 / 18)) :=
by sorry

-- 2) Proof that the central angle that maximizes the area is 2 radians and the maximum area is c^2 / 16
-- given 2r + l = c
theorem sector_max_area (h : 2 * r + l = c) :
  ∃ l r, 2 * r = l ∧ 1/2 * l * r = (c^2 / 16) :=
by sorry

end NUMINAMATH_GPT_sector_area_angle_1_sector_max_area_l2019_201974


namespace NUMINAMATH_GPT_limes_given_l2019_201993

theorem limes_given (original_limes now_limes : ℕ) (h1 : original_limes = 9) (h2 : now_limes = 5) : (original_limes - now_limes = 4) := 
by
  sorry

end NUMINAMATH_GPT_limes_given_l2019_201993


namespace NUMINAMATH_GPT_volume_of_convex_polyhedron_l2019_201923

variables {S1 S2 S : ℝ} {h : ℝ}

theorem volume_of_convex_polyhedron (S1 S2 S h : ℝ) :
  (h > 0) → (S1 ≥ 0) → (S2 ≥ 0) → (S ≥ 0) →
  ∃ V, V = (h / 6) * (S1 + S2 + 4 * S) :=
by {
  sorry
}

end NUMINAMATH_GPT_volume_of_convex_polyhedron_l2019_201923


namespace NUMINAMATH_GPT_seventh_root_of_unity_problem_l2019_201919

theorem seventh_root_of_unity_problem (q : ℂ) (h : q^7 = 1) :
  (q = 1 → (q / (1 + q^2) + q^2 / (1 + q^4) + q^3 / (1 + q^6)) = 3 / 2) ∧ 
  (q ≠ 1 → (q / (1 + q^2) + q^2 / (1 + q^4) + q^3 / (1 + q^6)) = -2) :=
by
  sorry

end NUMINAMATH_GPT_seventh_root_of_unity_problem_l2019_201919


namespace NUMINAMATH_GPT_product_of_two_odd_numbers_not_always_composite_l2019_201905

theorem product_of_two_odd_numbers_not_always_composite :
  ∃ (m n : ℕ), (¬ (2 ∣ m) ∧ ¬ (2 ∣ n)) ∧ (∀ d : ℕ, d ∣ (m * n) → d = 1 ∨ d = m * n) :=
by
  sorry

end NUMINAMATH_GPT_product_of_two_odd_numbers_not_always_composite_l2019_201905


namespace NUMINAMATH_GPT_solve_f_zero_k_eq_2_find_k_range_has_two_zeros_sum_of_reciprocals_l2019_201970

-- Define the function f(x) based on the given conditions
def f (x k : ℝ) : ℝ := abs (x ^ 2 - 1) + x ^ 2 + k * x

-- Statement 1
theorem solve_f_zero_k_eq_2 :
  (∀ x : ℝ, f x 2 = 0 ↔ x = - (1 + Real.sqrt 3) / 2 ∨ x = -1 / 2) :=
sorry

-- Statement 2
theorem find_k_range_has_two_zeros (α β : ℝ) (hαβ : 0 < α ∧ α < β ∧ β < 2) :
  (∃ k : ℝ, f α k = 0 ∧ f β k = 0) ↔ - 7 / 2 < k ∧ k < -1 :=
sorry

-- Statement 3
theorem sum_of_reciprocals (α β : ℝ) (hαβ : 0 < α ∧ α < 1 ∧ 1 < β ∧ β < 2)
    (hα : f α (-1/α) = 0) (hβ : ∃ k : ℝ, f β k = 0) :
  (1 / α + 1 / β < 4) :=
sorry

end NUMINAMATH_GPT_solve_f_zero_k_eq_2_find_k_range_has_two_zeros_sum_of_reciprocals_l2019_201970


namespace NUMINAMATH_GPT_inequality_neg_reciprocal_l2019_201950

theorem inequality_neg_reciprocal (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 
  - (1 / a) < - (1 / b) :=
sorry

end NUMINAMATH_GPT_inequality_neg_reciprocal_l2019_201950


namespace NUMINAMATH_GPT_intersection_M_N_eq_02_l2019_201900

open Set

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {y | ∃ x ∈ M, y = 2 * x}

theorem intersection_M_N_eq_02 : M ∩ N = {0, 2} := 
by sorry

end NUMINAMATH_GPT_intersection_M_N_eq_02_l2019_201900


namespace NUMINAMATH_GPT_chipmunk_acorns_l2019_201966

-- Define the conditions and goal for the proof
theorem chipmunk_acorns :
  ∃ x : ℕ, (∀ h_c h_s : ℕ, h_c = h_s + 4 → 3 * h_c = x ∧ 4 * h_s = x) → x = 48 :=
by {
  -- We assume the problem conditions as given
  sorry
}

end NUMINAMATH_GPT_chipmunk_acorns_l2019_201966


namespace NUMINAMATH_GPT_johny_journey_distance_l2019_201967

def south_distance : ℕ := 40
def east_distance : ℕ := south_distance + 20
def north_distance : ℕ := 2 * east_distance
def total_distance : ℕ := south_distance + east_distance + north_distance

theorem johny_journey_distance :
  total_distance = 220 := by
  sorry

end NUMINAMATH_GPT_johny_journey_distance_l2019_201967


namespace NUMINAMATH_GPT_polynomial_horner_method_l2019_201981

theorem polynomial_horner_method :
  let a_4 := 3
  let a_3 := 0
  let a_2 := -1
  let a_1 := 2
  let a_0 := 1
  let x := 2
  let v_0 := 3
  let v_1 := v_0 * x + a_3
  let v_2 := v_1 * x + a_2
  let v_3 := v_2 * x + a_1
  v_3 = 22 :=
by 
  let a_4 := 3
  let a_3 := 0
  let a_2 := -1
  let a_1 := 2
  let a_0 := 1
  let x := 2
  let v_0 := a_4
  let v_1 := v_0 * x + a_3
  let v_2 := v_1 * x + a_2
  let v_3 := v_2 * x + a_1
  sorry

end NUMINAMATH_GPT_polynomial_horner_method_l2019_201981


namespace NUMINAMATH_GPT_towel_area_decrease_l2019_201921

theorem towel_area_decrease (L B : ℝ) (hL : 0 < L) (hB : 0 < B) :
  let A := L * B
  let L' := 0.80 * L
  let B' := 0.90 * B
  let A' := L' * B'
  A' = 0.72 * A →
  ((A - A') / A) * 100 = 28 :=
by
  intros _ _ _ _
  sorry

end NUMINAMATH_GPT_towel_area_decrease_l2019_201921


namespace NUMINAMATH_GPT_ab_equality_l2019_201986

theorem ab_equality (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_div : 4 * a * b - 1 ∣ (4 * a ^ 2 - 1) ^ 2) : a = b := sorry

end NUMINAMATH_GPT_ab_equality_l2019_201986


namespace NUMINAMATH_GPT_tanya_bought_six_plums_l2019_201992

theorem tanya_bought_six_plums (pears apples pineapples pieces_left : ℕ) 
  (h_pears : pears = 6) (h_apples : apples = 4) (h_pineapples : pineapples = 2) 
  (h_pieces_left : pieces_left = 9) (h_half_fell : pieces_left * 2 = total_fruit) :
  pears + apples + pineapples < total_fruit ∧ total_fruit - (pears + apples + pineapples) = 6 :=
by
  sorry

end NUMINAMATH_GPT_tanya_bought_six_plums_l2019_201992


namespace NUMINAMATH_GPT_product_has_trailing_zeros_l2019_201904

theorem product_has_trailing_zeros (a b : ℕ) (h1 : a = 350) (h2 : b = 60) :
  ∃ (n : ℕ), (10^n ∣ a * b) ∧ n = 3 :=
by
  sorry

end NUMINAMATH_GPT_product_has_trailing_zeros_l2019_201904


namespace NUMINAMATH_GPT_find_r_l2019_201946

-- Lean statement
theorem find_r (r : ℚ) (log_eq : Real.logb 81 (2 * r - 1) = -1 / 2) : r = 5 / 9 :=
by {
    sorry -- proof steps should not be included according to the requirements
}

end NUMINAMATH_GPT_find_r_l2019_201946


namespace NUMINAMATH_GPT_find_a_l2019_201973

noncomputable def f (a : ℝ) : ℝ → ℝ := fun x =>
  if x < 1 then 2^x + 1 else x^2 + a * x

theorem find_a (a : ℝ) (h : f a (f a 0) = 4 * a) : a = 2 := by
  sorry

end NUMINAMATH_GPT_find_a_l2019_201973


namespace NUMINAMATH_GPT_kelly_chickens_l2019_201942

theorem kelly_chickens
  (chicken_egg_rate : ℕ)
  (chickens : ℕ)
  (egg_price_per_dozen : ℕ)
  (total_money : ℕ)
  (weeks : ℕ)
  (days_per_week : ℕ)
  (dozen : ℕ)
  (total_eggs_sold : ℕ)
  (total_days : ℕ)
  (total_eggs_laid : ℕ) : 
  chicken_egg_rate = 3 →
  egg_price_per_dozen = 5 →
  total_money = 280 →
  weeks = 4 →
  days_per_week = 7 →
  dozen = 12 →
  total_eggs_sold = total_money / egg_price_per_dozen * dozen →
  total_days = weeks * days_per_week →
  total_eggs_laid = chickens * chicken_egg_rate * total_days →
  total_eggs_sold = total_eggs_laid →
  chickens = 8 :=
by
  intros
  sorry

end NUMINAMATH_GPT_kelly_chickens_l2019_201942


namespace NUMINAMATH_GPT_find_MT_square_l2019_201934

-- Definitions and conditions
variables (P Q R S L O M N T U : Type*)
variables (x : ℝ)
variables (PL PQ PS QR RS LO : finset ℝ)
variable (side_length_PQRS : ℝ) (area_PLQ area_QMTL area_SNUL area_RNMUT : ℝ)
variables (LO_MT_perpendicular LO_NU_perpendicular : Prop)

-- Stating the problem
theorem find_MT_square :
  (side_length_PQRS = 3) →
  (PL ⊆ PQ) →
  (PO ⊆ PS) →
  (PL = PO) →
  (PL = x) →
  (U ∈ LO) →
  (T ∈ LO) →
  (LO_MT_perpendicular) →
  (LO_NU_perpendicular) →
  (area_PLQ = 1) →
  (area_QMTL = 1) →
  (area_SNUL = 2) →
  (area_RNMUT = 2) →
  (x^2 / 2 = 1) → 
  (PL * LO = 1) →
  MT^2 = 1 / 2 :=
sorry

end NUMINAMATH_GPT_find_MT_square_l2019_201934


namespace NUMINAMATH_GPT_sum_of_reflected_midpoint_coords_l2019_201940

theorem sum_of_reflected_midpoint_coords (P R : ℝ × ℝ) 
  (hP : P = (2, 1)) (hR : R = (12, 15)) :
  let M := ((P.1 + R.1) / 2, (P.2 + R.2) / 2)
  let P' := (-P.1, P.2)
  let R' := (-R.1, R.2)
  let M' := ((P'.1 + R'.1) / 2, (P'.2 + R'.2) / 2)
  M'.1 + M'.2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_reflected_midpoint_coords_l2019_201940
