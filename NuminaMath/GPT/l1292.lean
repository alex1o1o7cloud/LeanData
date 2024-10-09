import Mathlib

namespace max_set_size_divisible_diff_l1292_129225

theorem max_set_size_divisible_diff (S : Finset ℕ) (h1 : ∀ x ∈ S, ∀ y ∈ S, x ≠ y → (5 ∣ (x - y) ∨ 25 ∣ (x - y))) : S.card ≤ 25 :=
sorry

end max_set_size_divisible_diff_l1292_129225


namespace faith_weekly_earnings_l1292_129270

theorem faith_weekly_earnings :
  let hourly_pay := 13.50
  let regular_hours_per_day := 8
  let workdays_per_week := 5
  let overtime_hours_per_day := 2
  let regular_pay_per_day := hourly_pay * regular_hours_per_day
  let regular_pay_per_week := regular_pay_per_day * workdays_per_week
  let overtime_pay_per_day := hourly_pay * overtime_hours_per_day
  let overtime_pay_per_week := overtime_pay_per_day * workdays_per_week
  let total_weekly_earnings := regular_pay_per_week + overtime_pay_per_week
  total_weekly_earnings = 675 := 
  by
    sorry

end faith_weekly_earnings_l1292_129270


namespace lcm_even_numbers_between_14_and_21_l1292_129256

-- Define the even numbers between 14 and 21
def evenNumbers := [14, 16, 18, 20]

-- Define a function to compute the LCM of a list of integers
def lcm_list (l : List ℕ) : ℕ :=
  l.foldr Nat.lcm 1

-- Theorem statement: the LCM of the even numbers between 14 and 21 equals 5040
theorem lcm_even_numbers_between_14_and_21 :
  lcm_list evenNumbers = 5040 :=
by
  sorry

end lcm_even_numbers_between_14_and_21_l1292_129256


namespace smallest_n_condition_smallest_n_value_l1292_129273

theorem smallest_n_condition :
  ∃ (n : ℕ), n < 1000 ∧ (99999 % n = 0) ∧ (9999 % (n + 7) = 0) ∧ 
  ∀ m, (m < 1000 ∧ (99999 % m = 0) ∧ (9999 % (m + 7) = 0)) → n ≤ m := 
sorry

theorem smallest_n_value :
  ∃ (n : ℕ), n = 266 ∧ n < 1000 ∧ (99999 % n = 0) ∧ (9999 % (n + 7) = 0) := 
sorry

end smallest_n_condition_smallest_n_value_l1292_129273


namespace clothing_loss_l1292_129235

theorem clothing_loss
  (a : ℝ)
  (h1 : ∃ x y : ℝ, x * 1.25 = a ∧ y * 0.75 = a ∧ x + y - 2 * a = -8) :
  a = 60 :=
sorry

end clothing_loss_l1292_129235


namespace fraction_zero_iff_x_is_four_l1292_129291

theorem fraction_zero_iff_x_is_four (x : ℝ) (h_ne_zero: x + 4 ≠ 0) :
  (16 - x^2) / (x + 4) = 0 ↔ x = 4 :=
sorry

end fraction_zero_iff_x_is_four_l1292_129291


namespace projections_on_hypotenuse_l1292_129201

variables {a b c p q : ℝ}
variables {ρa ρb : ℝ}

-- Given conditions
variable (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
variable (h2 : a < b)
variable (h3 : p = a * a / c)
variable (h4 : q = b * b / c)
variable (h5 : ρa = (a * (b + c - a)) / (a + b + c))
variable (h6 : ρb = (b * (a + c - b)) / (a + b + c))

-- Proof goal
theorem projections_on_hypotenuse 
  (h_right_triangle: a^2 + b^2 = c^2) : p < ρa ∧ q > ρb :=
by
  sorry

end projections_on_hypotenuse_l1292_129201


namespace problem_solution_l1292_129205

variables {R : Type} [LinearOrder R]

def M (x y : R) : R := max x y
def m (x y : R) : R := min x y

theorem problem_solution (p q r s t : R) (h : p < q) (h1 : q < r) (h2 : r < s) (h3 : s < t) :
  M (M p (m q r)) (m s (M p t)) = q :=
by
  sorry

end problem_solution_l1292_129205


namespace quadratic_even_coeff_l1292_129245

theorem quadratic_even_coeff (a b c : ℤ) (h₁ : a ≠ 0) (h₂ : ∃ r s : ℚ, r * s + b * r + c = 0) : (a % 2 = 0) ∨ (b % 2 = 0) ∨ (c % 2 = 0) := by
  sorry

end quadratic_even_coeff_l1292_129245


namespace largest_4_digit_number_divisible_by_1615_l1292_129278

theorem largest_4_digit_number_divisible_by_1615 (X : ℕ) (hX: 8640 = 1615 * X) (h1: 1000 ≤ 1615 * X ∧ 1615 * X ≤ 9999) : X = 5 :=
by
  sorry

end largest_4_digit_number_divisible_by_1615_l1292_129278


namespace find_four_consecutive_odd_numbers_l1292_129284

noncomputable def four_consecutive_odd_numbers (a b c d : ℤ) : Prop :=
  a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ d % 2 = 1 ∧
  (a = b + 2 ∨ a = b - 2) ∧ (b = c + 2 ∨ b = c - 2) ∧ (c = d + 2 ∨ c = d - 2)

def numbers_sum_to_26879 (a b c d : ℤ) : Prop :=
  1 + (a + b + c + d) +
  (a * b + a * c + a * d + b * c + b * d + c * d) +
  (a * b * c + a * b * d + a * c * d + b * c * d) +
  (a * b * c * d) = 26879

theorem find_four_consecutive_odd_numbers (a b c d : ℤ) :
  four_consecutive_odd_numbers a b c d ∧ numbers_sum_to_26879 a b c d →
  ((a, b, c, d) = (9, 11, 13, 15) ∨ (a, b, c, d) = (-17, -15, -13, -11)) :=
by {
  sorry
}

end find_four_consecutive_odd_numbers_l1292_129284


namespace diana_shopping_for_newborns_l1292_129206

-- Define the conditions
def num_toddlers : ℕ := 6
def num_teenagers : ℕ := 5 * num_toddlers
def total_children : ℕ := 40

-- Define the problem statement
theorem diana_shopping_for_newborns : (total_children - (num_toddlers + num_teenagers)) = 4 := by
  sorry

end diana_shopping_for_newborns_l1292_129206


namespace candy_distribution_l1292_129280

theorem candy_distribution (A B : ℕ) (h1 : 7 * A = B + 12) (h2 : 3 * A = B - 20) : A + B = 52 :=
by {
  -- proof goes here
  sorry
}

end candy_distribution_l1292_129280


namespace min_distance_parabola_midpoint_l1292_129239

theorem min_distance_parabola_midpoint 
  (a : ℝ) (m : ℝ) (h_pos_a : a > 0) :
  (m ≥ 1 / a → ∃ M_y : ℝ, M_y = (2 * m * a - 1) / (4 * a)) ∧ 
  (m < 1 / a → ∃ M_y : ℝ, M_y = a * m^2 / 4) := 
by 
  sorry

end min_distance_parabola_midpoint_l1292_129239


namespace building_shadow_length_l1292_129216

theorem building_shadow_length
  (flagpole_height : ℝ) (flagpole_shadow : ℝ) (building_height : ℝ)
  (h_flagpole : flagpole_height = 18) (s_flagpole : flagpole_shadow = 45) 
  (h_building : building_height = 26) :
  ∃ (building_shadow : ℝ), (building_height / building_shadow = flagpole_height / flagpole_shadow) ∧ building_shadow = 65 :=
by
  use 65
  sorry

end building_shadow_length_l1292_129216


namespace factor_theorem_solution_l1292_129283

theorem factor_theorem_solution (t : ℝ) :
  (6 * t ^ 2 - 17 * t - 7 = 0) ↔ 
  (t = (17 + Real.sqrt 457) / 12 ∨ t = (17 - Real.sqrt 457) / 12) :=
by sorry

end factor_theorem_solution_l1292_129283


namespace ln_n_lt_8m_l1292_129268

noncomputable def f (x : ℝ) (m : ℝ) (n : ℝ) : ℝ := 
  Real.log x - m * x^2 + 2 * n * x

theorem ln_n_lt_8m (m : ℝ) (n : ℝ) (h₀ : 0 < n) (h₁ : ∀ x > 0, f x m n ≤ f 1 m n) : 
  Real.log n < 8 * m := 
sorry

end ln_n_lt_8m_l1292_129268


namespace chemical_x_added_l1292_129202

theorem chemical_x_added (initial_volume : ℝ) (initial_percentage : ℝ) (final_percentage : ℝ) : 
  initial_volume = 80 → initial_percentage = 0.2 → final_percentage = 0.36 → 
  ∃ (a : ℝ), 0.20 * initial_volume + a = 0.36 * (initial_volume + a) ∧ a = 20 :=
by
  intros h1 h2 h3
  use 20
  sorry

end chemical_x_added_l1292_129202


namespace value_depletion_rate_l1292_129233

theorem value_depletion_rate (P F : ℝ) (t : ℝ) (r : ℝ) (h₁ : P = 1100) (h₂ : F = 891) (h₃ : t = 2) (decay_formula : F = P * (1 - r) ^ t) : r = 0.1 :=
by 
  sorry

end value_depletion_rate_l1292_129233


namespace ratio_a_to_c_l1292_129276

theorem ratio_a_to_c (a b c d : ℕ) 
  (h1 : a / b = 5 / 4) 
  (h2 : c / d = 4 / 3)
  (h3 : d / b = 1 / 5) : 
  a / c = 75 / 16 := 
  by 
    sorry

end ratio_a_to_c_l1292_129276


namespace sin_cos_sixth_power_sum_l1292_129289

theorem sin_cos_sixth_power_sum (θ : ℝ) (h : Real.sin (2 * θ) = Real.sqrt 2 / 2) : 
  (Real.sin θ)^6 + (Real.cos θ)^6 = 5 / 8 :=
by
  sorry

end sin_cos_sixth_power_sum_l1292_129289


namespace interior_sum_nine_l1292_129237

-- Defining the function for the sum of the interior numbers in the nth row of Pascal's Triangle
def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

-- Given conditions
axiom interior_sum_4 : interior_sum 4 = 6
axiom interior_sum_5 : interior_sum 5 = 14

-- Goal to prove
theorem interior_sum_nine : interior_sum 9 = 254 := by
  sorry

end interior_sum_nine_l1292_129237


namespace gcd_fact8_fact7_l1292_129238

noncomputable def fact8 : ℕ := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
noncomputable def fact7 : ℕ := 7 * 6 * 5 * 4 * 3 * 2 * 1

theorem gcd_fact8_fact7 : Nat.gcd fact8 fact7 = fact7 := by
  unfold fact8 fact7
  exact sorry

end gcd_fact8_fact7_l1292_129238


namespace range_of_c_l1292_129209

theorem range_of_c :
  (∃ (c : ℝ), ∀ (x y : ℝ), (x^2 + y^2 = 4) → ((12 * x - 5 * y + c) / 13 = 1))
  → (c > -13 ∧ c < 13) := 
sorry

end range_of_c_l1292_129209


namespace cos_sum_arithmetic_seq_l1292_129263

theorem cos_sum_arithmetic_seq (a : ℕ → ℝ) (d : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : a 1 + a 5 + a 9 = 5 * Real.pi) : 
  Real.cos (a 2 + a 8) = -1 / 2 :=
  sorry

end cos_sum_arithmetic_seq_l1292_129263


namespace total_doll_count_l1292_129298

noncomputable def sister_dolls : ℕ := 8
noncomputable def hannah_dolls : ℕ := 5 * sister_dolls
noncomputable def total_dolls : ℕ := hannah_dolls + sister_dolls

theorem total_doll_count : total_dolls = 48 := 
by 
  sorry

end total_doll_count_l1292_129298


namespace largest_expression_is_D_l1292_129204

-- Define each expression
def exprA : ℤ := 3 - 1 + 4 + 6
def exprB : ℤ := 3 - 1 * 4 + 6
def exprC : ℤ := 3 - (1 + 4) * 6
def exprD : ℤ := 3 - 1 + 4 * 6
def exprE : ℤ := 3 * (1 - 4) + 6

-- The theorem stating that exprD is the largest value among the given expressions.
theorem largest_expression_is_D : 
  exprD = 26 ∧ 
  exprD > exprA ∧ 
  exprD > exprB ∧ 
  exprD > exprC ∧ 
  exprD > exprE := 
by {
  sorry
}

end largest_expression_is_D_l1292_129204


namespace x_intercept_is_34_l1292_129294

-- Definitions of the initial line, rotation, and point.
def line_l (x y : ℝ) : Prop := 4 * x - 3 * y + 50 = 0

def rotation_angle : ℝ := 30
def rotation_center : ℝ × ℝ := (10, 10)

-- Define the slope of the line l
noncomputable def slope_of_l : ℝ := 4 / 3

-- Define the slope of the line m after rotating line l by 30 degrees counterclockwise
noncomputable def tan_30 : ℝ := 1 / Real.sqrt 3
noncomputable def slope_of_m : ℝ := (slope_of_l + tan_30) / (1 - slope_of_l * tan_30)

-- Assume line m goes through the point (rotation_center.x, rotation_center.y)
-- This defines line m
def line_m (x y : ℝ) : Prop := y - rotation_center.2 = slope_of_m * (x - rotation_center.1)

-- To find the x-intercept of line m, we set y = 0 and solve for x
noncomputable def x_intercept_of_m : ℝ := rotation_center.1 - rotation_center.2 / slope_of_m

-- Proof statement that the x-intercept of line m is 34
theorem x_intercept_is_34 : x_intercept_of_m = 34 :=
by
  -- This would be the proof, but for now we leave it as sorry
  sorry

end x_intercept_is_34_l1292_129294


namespace ladder_cost_l1292_129208

theorem ladder_cost (ladders1 ladders2 rung_count1 rung_count2 cost_per_rung : ℕ)
  (h1 : ladders1 = 10) (h2 : ladders2 = 20) (h3 : rung_count1 = 50) (h4 : rung_count2 = 60) (h5 : cost_per_rung = 2) :
  (ladders1 * rung_count1 + ladders2 * rung_count2) * cost_per_rung = 3400 :=
by 
  sorry

end ladder_cost_l1292_129208


namespace max_daily_sales_revenue_l1292_129277

noncomputable def p (t : ℕ) : ℝ :=
if 0 < t ∧ t < 25 then t + 20
else if 25 ≤ t ∧ t ≤ 30 then -t + 70
else 0

noncomputable def Q (t : ℕ) : ℝ :=
if 0 < t ∧ t ≤ 30 then -t + 40 else 0

theorem max_daily_sales_revenue :
  ∃ t : ℕ, 0 < t ∧ t ≤ 30 ∧ (p t) * (Q t) = 1125 ∧
  ∀ t' : ℕ, 0 < t' ∧ t' ≤ 30 → (p t') * (Q t') ≤ 1125 :=
sorry

end max_daily_sales_revenue_l1292_129277


namespace quadratic_root_condition_l1292_129232

theorem quadratic_root_condition (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Ici 10 ∪ Set.Iic (-10) :=
by 
  sorry

end quadratic_root_condition_l1292_129232


namespace complement_intersection_l1292_129295

open Set

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5}
noncomputable def A : Set ℕ := {1, 2, 3}
noncomputable def B : Set ℕ := {3, 4, 5}

theorem complement_intersection (U A B : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hA : A = {1, 2, 3}) (hB : B = {3, 4, 5}) :
  U \ (A ∩ B) = {1, 2, 4, 5} :=
by
  sorry

end complement_intersection_l1292_129295


namespace contingency_fund_allocation_l1292_129281

theorem contingency_fund_allocation :
  let donate := 240
  let community_pantry := donate * (1 / 3)
  let local_crisis := donate * (1 / 2)
  let remaining_after_two := donate - community_pantry - local_crisis
  let livelihood_project := remaining_after_two * (1 / 4)
  let contingency_fund := remaining_after_two - livelihood_project
  contingency_fund = 30 :=
by
  let donate := 240
  let community_pantry := donate * (1 / 3)
  let local_crisis := donate * (1 / 2)
  let remaining_after_two := donate - community_pantry - local_crisis
  let livelihood_project := remaining_after_two * (1 / 4)
  let contingency_fund := remaining_after_two - livelihood_project
  show contingency_fund = 30
  sorry

end contingency_fund_allocation_l1292_129281


namespace quadratic_specific_a_l1292_129211

noncomputable def quadratic_root_condition (a : ℝ) : Prop :=
  ∃ x : ℝ, (a + 2) * x^2 + 2 * a * x + 1 = 0

theorem quadratic_specific_a (a : ℝ) (h : quadratic_root_condition a) :
  a = 2 ∨ a = -1 :=
sorry

end quadratic_specific_a_l1292_129211


namespace find_a_circle_line_intersection_l1292_129228

theorem find_a_circle_line_intersection
  (h1 : ∀ x y : ℝ, x^2 + y^2 - 2 * a * x + 4 * y - 6 = 0)
  (h2 : ∀ x y : ℝ, x + 2 * y + 1 = 0) :
  a = 3 := 
sorry

end find_a_circle_line_intersection_l1292_129228


namespace odd_function_expression_l1292_129255

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 + 3 * x - 4 else - (x^2 - 3 * x - 4)

theorem odd_function_expression (x : ℝ) (h : x < 0) : 
  f x = -x^2 + 3 * x + 4 :=
by
  sorry

end odd_function_expression_l1292_129255


namespace alissa_presents_l1292_129285

def ethan_presents : ℝ := 31.0
def difference : ℝ := 22.0

theorem alissa_presents : ethan_presents - difference = 9.0 := by sorry

end alissa_presents_l1292_129285


namespace functions_are_equal_l1292_129226

-- Define the functions
def f (x : ℝ) : ℝ := |x|
def g (x : ℝ) : ℝ := (x^4)^(1/4)

-- Statement to be proven
theorem functions_are_equal : ∀ x : ℝ, f x = g x := by
  sorry

end functions_are_equal_l1292_129226


namespace rooks_placement_possible_l1292_129293

/-- 
  It is possible to place 8 rooks on a chessboard such that they do not attack each other
  and each rook stands on cells of different colors, given that the chessboard is divided 
  into 32 colors with exactly two cells of each color.
-/
theorem rooks_placement_possible :
  ∃ (placement : Fin 8 → Fin 8 × Fin 8),
    (∀ i j, i ≠ j → (placement i).fst ≠ (placement j).fst ∧ (placement i).snd ≠ (placement j).snd) ∧
    (∀ i j, i ≠ j → (placement i ≠ placement j)) ∧
    (∀ c : Fin 32, ∃! p1 p2, placement p1 = placement p2 ∧ (placement p1).fst ≠ (placement p2).fst 
                        ∧ (placement p1).snd ≠ (placement p2).snd) :=
by
  sorry

end rooks_placement_possible_l1292_129293


namespace luncheon_cost_l1292_129254

section LuncheonCosts

variables (s c p : ℝ)

/- Conditions -/
def eq1 : Prop := 2 * s + 5 * c + 2 * p = 6.25
def eq2 : Prop := 5 * s + 8 * c + 3 * p = 12.10

/- Goal -/
theorem luncheon_cost : eq1 s c p → eq2 s c p → s + c + p = 1.55 :=
by
  intro h1 h2
  sorry

end LuncheonCosts

end luncheon_cost_l1292_129254


namespace initial_walnut_trees_l1292_129222

/-- 
  Given there are 29 walnut trees in the park after cutting down 13 walnut trees, 
  prove that initially there were 42 walnut trees in the park.
-/
theorem initial_walnut_trees (cut_walnut_trees remaining_walnut_trees initial_walnut_trees : ℕ) 
  (h₁ : cut_walnut_trees = 13)
  (h₂ : remaining_walnut_trees = 29)
  (h₃ : initial_walnut_trees = cut_walnut_trees + remaining_walnut_trees) :
  initial_walnut_trees = 42 := 
sorry

end initial_walnut_trees_l1292_129222


namespace union_of_sets_l1292_129288

variable (A : Set ℤ) (B : Set ℤ)

theorem union_of_sets (hA : A = {0, 1, 2}) (hB : B = {-1, 0}) : A ∪ B = {-1, 0, 1, 2} :=
by
  sorry

end union_of_sets_l1292_129288


namespace tom_can_go_on_three_rides_l1292_129267

def rides_possible (total_tickets : ℕ) (spent_tickets : ℕ) (tickets_per_ride : ℕ) : ℕ :=
  (total_tickets - spent_tickets) / tickets_per_ride

theorem tom_can_go_on_three_rides :
  rides_possible 40 28 4 = 3 :=
by
  -- proof goes here
  sorry

end tom_can_go_on_three_rides_l1292_129267


namespace nth_equation_l1292_129299

theorem nth_equation (n : ℕ) : 2 * n * (2 * n + 2) + 1 = (2 * n + 1)^2 :=
by
  sorry

end nth_equation_l1292_129299


namespace trig_system_solution_l1292_129212

theorem trig_system_solution (x y : ℝ) (hx : 0 ≤ x ∧ x < 2 * Real.pi) (hy : 0 ≤ y ∧ y < 2 * Real.pi)
  (h1 : Real.sin x + Real.cos y = 0) (h2 : Real.cos x * Real.sin y = -1/2) :
    (x = Real.pi / 4 ∧ y = 5 * Real.pi / 4) ∨
    (x = 3 * Real.pi / 4 ∧ y = 3 * Real.pi / 4) ∨
    (x = 5 * Real.pi / 4 ∧ y = Real.pi / 4) ∨
    (x = 7 * Real.pi / 4 ∧ y = 7 * Real.pi / 4) := by
  sorry

end trig_system_solution_l1292_129212


namespace opposite_of_neg_2023_l1292_129260

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by
  sorry

end opposite_of_neg_2023_l1292_129260


namespace probability_of_D_given_T_l1292_129203

-- Definitions based on the conditions given in the problem.
def pr_D : ℚ := 1 / 400
def pr_Dc : ℚ := 399 / 400
def pr_T_given_D : ℚ := 1
def pr_T_given_Dc : ℚ := 0.05
def pr_T : ℚ := pr_T_given_D * pr_D + pr_T_given_Dc * pr_Dc

-- Statement to prove 
theorem probability_of_D_given_T : pr_T ≠ 0 → (pr_T_given_D * pr_D) / pr_T = 20 / 419 :=
by
  intros h1
  unfold pr_T pr_D pr_Dc pr_T_given_D pr_T_given_Dc
  -- Mathematical steps are skipped in Lean by inserting sorry
  sorry

-- Check that the statement can be built successfully
example : pr_D = 1 / 400 := by rfl
example : pr_Dc = 399 / 400 := by rfl
example : pr_T_given_D = 1 := by rfl
example : pr_T_given_Dc = 0.05 := by rfl
example : pr_T = (1 * (1 / 400) + 0.05 * (399 / 400)) := by rfl

end probability_of_D_given_T_l1292_129203


namespace no_primes_sum_to_53_l1292_129218

open Nat

def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem no_primes_sum_to_53 :
  ¬ ∃ (p q : Nat), p + q = 53 ∧ isPrime p ∧ isPrime q ∧ (p < 30 ∨ q < 30) :=
by
  sorry

end no_primes_sum_to_53_l1292_129218


namespace verify_statements_l1292_129282

theorem verify_statements (S : Set ℝ) (m l : ℝ) (hS : ∀ x, x ∈ S → x^2 ∈ S) :
  (m = 1 → S = {1}) ∧
  (m = -1/2 → (1/4 ≤ l ∧ l ≤ 1)) ∧
  (l = 1/2 → -Real.sqrt 2 / 2 ≤ m ∧ m ≤ 0) ∧
  (l = 1 → -1 ≤ m ∧ m ≤ 1) :=
  sorry

end verify_statements_l1292_129282


namespace find_BC_line_eq_l1292_129236

def line1_altitude : Prop := ∃ x y : ℝ, 2*x - 3*y + 1 = 0
def line2_altitude : Prop := ∃ x y : ℝ, x + y = 0
def vertex_A : Prop := ∃ a1 a2 : ℝ, a1 = 1 ∧ a2 = 2
def side_BC_equation : Prop := ∃ b c d : ℝ, b = 2 ∧ c = 3 ∧ d = 7

theorem find_BC_line_eq (H1 : line1_altitude) (H2 : line2_altitude) (H3 : vertex_A) : side_BC_equation :=
sorry

end find_BC_line_eq_l1292_129236


namespace sally_eats_sandwiches_l1292_129220

theorem sally_eats_sandwiches
  (saturday_sandwiches : ℕ)
  (bread_per_sandwich : ℕ)
  (total_bread : ℕ)
  (one_sandwich_on_sunday : ℕ)
  (saturday_bread : saturday_sandwiches * bread_per_sandwich = 4)
  (total_bread_consumed : total_bread = 6)
  (bread_on_sundy : bread_per_sandwich = 2) :
  (total_bread - saturday_sandwiches * bread_per_sandwich) / bread_per_sandwich = one_sandwich_on_sunday :=
sorry

end sally_eats_sandwiches_l1292_129220


namespace all_tell_truth_at_same_time_l1292_129214

-- Define the probabilities of each person telling the truth.
def prob_Alice := 0.7
def prob_Bob := 0.6
def prob_Carol := 0.8
def prob_David := 0.5

-- Prove that the probability that all four tell the truth at the same time is 0.168.
theorem all_tell_truth_at_same_time :
  prob_Alice * prob_Bob * prob_Carol * prob_David = 0.168 :=
by
  sorry

end all_tell_truth_at_same_time_l1292_129214


namespace allen_total_blocks_l1292_129296

/-- 
  If there are 7 blocks for every color of paint used and Shiela used 7 colors, 
  then the total number of blocks Allen has is 49.
-/
theorem allen_total_blocks
  (blocks_per_color : ℕ) 
  (number_of_colors : ℕ)
  (h1 : blocks_per_color = 7) 
  (h2 : number_of_colors = 7) : 
  blocks_per_color * number_of_colors = 49 := 
by 
  sorry

end allen_total_blocks_l1292_129296


namespace OneEmptyBox_NoBoxEmptyNoCompleteMatch_AtLeastTwoMatches_l1292_129269

def combination (n k : ℕ) : ℕ := Nat.choose n k
def arrangement (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem OneEmptyBox (n : ℕ) (hn : n = 5) : (combination 5 2) * (arrangement 5 5) = 1200 := by
  sorry

theorem NoBoxEmptyNoCompleteMatch (n : ℕ) (hn : n = 5) : (arrangement 5 5) - 1 = 119 := by
  sorry

theorem AtLeastTwoMatches (n : ℕ) (hn : n = 5) : (arrangement 5 5) - (combination 5 1 * 9 + 44) = 31 := by
  sorry

end OneEmptyBox_NoBoxEmptyNoCompleteMatch_AtLeastTwoMatches_l1292_129269


namespace range_of_m_l1292_129272

noncomputable def f (x : ℝ) := Real.exp x * (x - 1)
noncomputable def g (m x : ℝ) := m * x

theorem range_of_m :
  (∀ x₁ ∈ Set.Icc (-2 : ℝ) 2, ∃ x₂ ∈ Set.Icc (1 : ℝ) 2, f x₁ > g m x₂) ↔ m ∈ Set.Iio (-1/2 : ℝ) :=
sorry

end range_of_m_l1292_129272


namespace prime_divisor_property_l1292_129210

open Classical

theorem prime_divisor_property (p n q : ℕ) (hp : Nat.Prime p) (hn : 0 < n) (hq : q ∣ (n + 1)^p - n^p) : p ∣ q - 1 :=
by
  sorry

end prime_divisor_property_l1292_129210


namespace index_card_area_reduction_index_card_area_when_other_side_shortened_l1292_129279

-- Conditions
def original_length := 4
def original_width := 6
def shortened_length := 2
def target_area := 12
def shortened_other_width := 5

-- Theorems to prove
theorem index_card_area_reduction :
  (original_length - 2) * original_width = target_area := by
  sorry

theorem index_card_area_when_other_side_shortened :
  (original_length) * (original_width - 1) = 20 := by
  sorry

end index_card_area_reduction_index_card_area_when_other_side_shortened_l1292_129279


namespace g_h_of_2_eq_869_l1292_129200

-- Define the functions g and h
def g (x : ℝ) : ℝ := 3 * x^2 + 2
def h (x : ℝ) : ℝ := -2 * x^3 - 1

-- State the theorem we need to prove
theorem g_h_of_2_eq_869 : g (h 2) = 869 := by
  sorry

end g_h_of_2_eq_869_l1292_129200


namespace tina_final_balance_l1292_129257

noncomputable def monthlyIncome : ℝ := 1000
noncomputable def juneBonusRate : ℝ := 0.1
noncomputable def investmentReturnRate : ℝ := 0.05
noncomputable def taxRate : ℝ := 0.1

-- Savings rates
noncomputable def juneSavingsRate : ℝ := 0.25
noncomputable def julySavingsRate : ℝ := 0.20
noncomputable def augustSavingsRate : ℝ := 0.30

-- Expenses
noncomputable def juneRent : ℝ := 200
noncomputable def juneGroceries : ℝ := 100
noncomputable def juneBookRate : ℝ := 0.05

noncomputable def julyRent : ℝ := 250
noncomputable def julyGroceries : ℝ := 150
noncomputable def julyShoesRate : ℝ := 0.15

noncomputable def augustRent : ℝ := 300
noncomputable def augustGroceries : ℝ := 175
noncomputable def augustMiscellaneousRate : ℝ := 0.1

theorem tina_final_balance :
  let juneIncome := monthlyIncome * (1 + juneBonusRate)
  let juneSavings := juneIncome * juneSavingsRate
  let juneExpenses := juneRent + juneGroceries + juneIncome * juneBookRate
  let juneRemaining := juneIncome - juneSavings - juneExpenses

  let julyIncome := monthlyIncome
  let julyInvestmentReturn := juneSavings * investmentReturnRate
  let julyTotalIncome := julyIncome + julyInvestmentReturn
  let julySavings := julyTotalIncome * julySavingsRate
  let julyExpenses := julyRent + julyGroceries + julyIncome * julyShoesRate
  let julyRemaining := julyTotalIncome - julySavings - julyExpenses

  let augustIncome := monthlyIncome
  let augustInvestmentReturn := julySavings * investmentReturnRate
  let augustTotalIncome := augustIncome + augustInvestmentReturn
  let augustSavings := augustTotalIncome * augustSavingsRate
  let augustExpenses := augustRent + augustGroceries + augustIncome * augustMiscellaneousRate
  let augustRemaining := augustTotalIncome - augustSavings - augustExpenses

  let totalInvestmentReturn := julyInvestmentReturn + augustInvestmentReturn
  let totalTaxOnInvestment := totalInvestmentReturn * taxRate

  let finalBalance := juneRemaining + julyRemaining + augustRemaining - totalTaxOnInvestment

  finalBalance = 860.7075 := by
  sorry

end tina_final_balance_l1292_129257


namespace find_part_of_number_l1292_129207

theorem find_part_of_number (x y : ℕ) (h₁ : x = 1925) (h₂ : x / 7 = y + 100) : y = 175 :=
sorry

end find_part_of_number_l1292_129207


namespace reduced_population_l1292_129275

theorem reduced_population (initial_population : ℕ)
  (percentage_died : ℝ)
  (percentage_left : ℝ)
  (h_initial : initial_population = 8515)
  (h_died : percentage_died = 0.10)
  (h_left : percentage_left = 0.15) :
  ((initial_population - (⌊percentage_died * initial_population⌋₊ : ℕ)) - 
   (⌊percentage_left * (initial_population - (⌊percentage_died * initial_population⌋₊ : ℕ))⌋₊ : ℕ)) = 6515 :=
by
  sorry

end reduced_population_l1292_129275


namespace completing_the_square_l1292_129292

theorem completing_the_square (x : ℝ) : x^2 - 8 * x + 1 = 0 → (x - 4)^2 = 15 :=
by
  sorry

end completing_the_square_l1292_129292


namespace a_2017_value_l1292_129286

theorem a_2017_value (S : ℕ → ℤ) (a : ℕ → ℤ) (hS : ∀ n : ℕ, S (n + 1) = 2 * (n + 1) - 1) :
  a 2017 = 2 :=
by
  sorry

end a_2017_value_l1292_129286


namespace pirate_coins_total_l1292_129223

theorem pirate_coins_total (x : ℕ) (hx : x ≠ 0) (h_paul : ∃ k : ℕ, k = x / 2) (h_pete : ∃ m : ℕ, m = 5 * (x / 2)) 
  (h_ratio : (m : ℝ) = (k : ℝ) * 5) : (x = 4) → 
  ∃ total : ℕ, total = k + m ∧ total = 12 :=
by {
  sorry
}

end pirate_coins_total_l1292_129223


namespace part1_part2_l1292_129243

-- Given conditions
variable {f : ℝ → ℝ}
variable (h_odd : ∀ x : ℝ, f (-x) = -f x)

-- Proof statements to be demonstrated
theorem part1 (a : ℝ) : a = 1 := sorry

theorem part2 (f_inv : ℝ → ℝ) : 
  (∀ x : ℝ, x > -1 ∧ x < 1 → f (f_inv x) = x ∧ f_inv (f x) = x) :=
sorry

end part1_part2_l1292_129243


namespace silver_tokens_at_end_l1292_129264

theorem silver_tokens_at_end {R B S : ℕ} (x y : ℕ) 
  (hR_init : R = 60) (hB_init : B = 90) 
  (hR_final : R = 60 - 3 * x + y) 
  (hB_final : B = 90 + 2 * x - 4 * y) 
  (h_end_conditions : 0 ≤ R ∧ R < 3 ∧ 0 ≤ B ∧ B < 4) : 
  S = x + y → 
  S = 23 :=
sorry

end silver_tokens_at_end_l1292_129264


namespace probability_AB_together_l1292_129253

theorem probability_AB_together : 
  let total_events := 6
  let ab_together_events := 4
  let probability := ab_together_events / total_events
  probability = 2 / 3 :=
by
  sorry

end probability_AB_together_l1292_129253


namespace packaging_combinations_l1292_129262

-- Conditions
def wrapping_paper_choices : ℕ := 10
def ribbon_colors : ℕ := 5
def gift_tag_styles : ℕ := 6

-- Question and proof
theorem packaging_combinations : wrapping_paper_choices * ribbon_colors * gift_tag_styles = 300 := by
  sorry

end packaging_combinations_l1292_129262


namespace area_ratio_of_squares_l1292_129246

-- Definition of squares, and their perimeters' relationship
def perimeter (side_length : ℝ) := 4 * side_length

theorem area_ratio_of_squares (a b : ℝ) (h : perimeter a = 4 * perimeter b) : (a * a) = 16 * (b * b) :=
by
  -- We assume the given condition
  have ha : a = 4 * b := sorry
  -- We then prove the area ratio
  sorry

end area_ratio_of_squares_l1292_129246


namespace infinite_sum_equals_one_fourth_l1292_129259

theorem infinite_sum_equals_one_fourth :
  ∑' n : ℕ, (3^n / (1 + 3^n + 3^(n + 1) + 3^(2 * n + 1))) = 1 / 4 :=
sorry

end infinite_sum_equals_one_fourth_l1292_129259


namespace marbles_remaining_l1292_129251

def original_marbles : Nat := 64
def given_marbles : Nat := 14
def remaining_marbles : Nat := original_marbles - given_marbles

theorem marbles_remaining : remaining_marbles = 50 :=
  by
    sorry

end marbles_remaining_l1292_129251


namespace find_angle_A_find_perimeter_l1292_129215

noncomputable def cos_rule (b c a : ℝ) (h : b^2 + c^2 - a^2 = b * c) : ℝ :=
(b^2 + c^2 - a^2) / (2 * b * c)

theorem find_angle_A (A B C : ℝ) (a b c : ℝ)
  (h1 : b^2 + c^2 - a^2 = b * c) (hA : cos_rule b c a h1 = 1 / 2) :
  A = Real.arccos (1 / 2) :=
by sorry

theorem find_perimeter (a b c : ℝ)
  (h_a : a = Real.sqrt 2) (hA : Real.sin (Real.arccos (1 / 2))^2 = (Real.sqrt 3 / 2)^2)
  (hBC : Real.sin (Real.arccos (1 / 2))^2 = Real.sin (Real.arccos (1 / 2)) * Real.sin (Real.arccos (1 / 2)))
  (h_bc : b * c = 2)
  (h_bc_eq : b^2 + c^2 - a^2 = b * c) :
  a + b + c = 3 * Real.sqrt 2 :=
by sorry

end find_angle_A_find_perimeter_l1292_129215


namespace rectangular_field_diagonal_length_l1292_129252

noncomputable def diagonal_length_of_rectangular_field (a : ℝ) (A : ℝ) : ℝ :=
  let b := A / a
  let d := Real.sqrt (a^2 + b^2)
  d

theorem rectangular_field_diagonal_length :
  let a : ℝ := 14
  let A : ℝ := 135.01111065390137
  abs (diagonal_length_of_rectangular_field a A - 17.002) < 0.001 := by
    sorry

end rectangular_field_diagonal_length_l1292_129252


namespace length_of_FD_l1292_129287

theorem length_of_FD (a b c d f e : ℝ) (x : ℝ) :
  a = 0 ∧ b = 8 ∧ c = 8 ∧ d = 0 ∧ 
  e = 8 * (2 / 3) ∧ 
  (8 - x)^2 = x^2 + (8 / 3)^2 ∧ 
  a = d → c = b → 
  d = 8 → 
  x = 32 / 9 :=
by
  sorry

end length_of_FD_l1292_129287


namespace cubic_eq_roots_l1292_129213

theorem cubic_eq_roots (x1 x2 x3 : ℕ) (P : ℕ) 
  (h1 : x1 + x2 + x3 = 10) 
  (h2 : x1 * x2 * x3 = 30) 
  (h3 : x1 * x2 + x2 * x3 + x3 * x1 = P) : 
  P = 31 := by
  sorry

end cubic_eq_roots_l1292_129213


namespace pound_of_rice_cost_l1292_129231

theorem pound_of_rice_cost 
(E R K : ℕ) (h1: E = R) (h2: K = 4 * (E / 12)) (h3: K = 11) : R = 33 := by
  sorry

end pound_of_rice_cost_l1292_129231


namespace remaining_oak_trees_l1292_129274

def initial_oak_trees : ℕ := 9
def cut_down_oak_trees : ℕ := 2

theorem remaining_oak_trees : initial_oak_trees - cut_down_oak_trees = 7 := 
by 
  sorry

end remaining_oak_trees_l1292_129274


namespace knitting_time_is_correct_l1292_129297

-- Definitions of the conditions
def time_per_hat : ℕ := 2
def time_per_scarf : ℕ := 3
def time_per_mitten : ℕ := 1
def time_per_sock : ℕ := 3 / 2 -- fractional time in hours
def time_per_sweater : ℕ := 6
def number_of_grandchildren : ℕ := 3

-- Compute total time for one complete outfit
def time_per_outfit : ℕ := time_per_hat + time_per_scarf + (time_per_mitten * 2) + (time_per_sock * 2) + time_per_sweater

-- Compute total time for all outfits
def total_knitting_time : ℕ := number_of_grandchildren * time_per_outfit

-- Prove that total knitting time is 48 hours
theorem knitting_time_is_correct : total_knitting_time = 48 := by
  unfold total_knitting_time time_per_outfit
  norm_num
  sorry

end knitting_time_is_correct_l1292_129297


namespace initial_birds_on_fence_l1292_129234

theorem initial_birds_on_fence (B S : ℕ) (S_val : S = 2) (total : B + 5 + S = 10) : B = 3 :=
by
  sorry

end initial_birds_on_fence_l1292_129234


namespace notebooks_last_days_l1292_129244

theorem notebooks_last_days (n p u : Nat) (total_pages days : Nat) 
  (h1 : n = 5)
  (h2 : p = 40)
  (h3 : u = 4)
  (h_total : total_pages = n * p)
  (h_days  : days = total_pages / u) :
  days = 50 := 
by
  sorry

end notebooks_last_days_l1292_129244


namespace calculate_result_l1292_129261

theorem calculate_result :
  (-24) * ((5 / 6 : ℚ) - (4 / 3) + (5 / 8)) = -3 := 
by
  sorry

end calculate_result_l1292_129261


namespace problem_1956_Tokyo_Tech_l1292_129248

theorem problem_1956_Tokyo_Tech (a b c : ℝ) (ha : 0 < a) (ha_lt_one : a < 1) (hb : 0 < b) 
(hb_lt_one : b < 1) (hc : 0 < c) (hc_lt_one : c < 1) : a + b + c - a * b * c < 2 := 
sorry

end problem_1956_Tokyo_Tech_l1292_129248


namespace average_wx_l1292_129227

theorem average_wx (w x a b : ℝ) (i : ℂ) (h_i : i * i = -1)
  (h1 : 6 / w + 6 / x = 6 / (a + b * i))
  (h2 : w * x = a + b * i) :
  (w + x) / 2 = 1 / 2 :=
by
  sorry

end average_wx_l1292_129227


namespace final_coordinates_of_F_l1292_129290

-- Define the points D, E, F
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the initial points D, E, F
def D : Point := ⟨3, -4⟩
def E : Point := ⟨5, -1⟩
def F : Point := ⟨-2, -3⟩

-- Define the reflection over the y-axis
def reflect_over_y (p : Point) : Point := ⟨-p.x, p.y⟩

-- Define the reflection over the x-axis
def reflect_over_x (p : Point) : Point := ⟨p.x, -p.y⟩

-- First reflection over the y-axis
def F' : Point := reflect_over_y F

-- Second reflection over the x-axis
def F'' : Point := reflect_over_x F'

-- The proof problem
theorem final_coordinates_of_F'' :
  F'' = ⟨2, 3⟩ := 
sorry

end final_coordinates_of_F_l1292_129290


namespace combined_CD_length_l1292_129265

def CD1 := 1.5
def CD2 := 1.5
def CD3 := 2 * CD1

theorem combined_CD_length : CD1 + CD2 + CD3 = 6 := 
by
  sorry

end combined_CD_length_l1292_129265


namespace documentaries_count_l1292_129240

def number_of_documents
  (novels comics albums crates capacity : ℕ)
  (total_items := crates * capacity)
  (known_items := novels + comics + albums)
  (documentaries := total_items - known_items) : ℕ :=
  documentaries

theorem documentaries_count
  : number_of_documents 145 271 209 116 9 = 419 :=
by
  sorry

end documentaries_count_l1292_129240


namespace john_investment_in_bankA_l1292_129271

-- Definitions to set up the conditions
def total_investment : ℝ := 1500
def bankA_rate : ℝ := 0.04
def bankB_rate : ℝ := 0.06
def final_amount : ℝ := 1575

-- Definition of the question to be proved
theorem john_investment_in_bankA (x : ℝ) (h : 0 ≤ x ∧ x ≤ total_investment) :
  (x * (1 + bankA_rate) + (total_investment - x) * (1 + bankB_rate) = final_amount) -> x = 750 := sorry


end john_investment_in_bankA_l1292_129271


namespace digit_D_is_five_l1292_129249

variable (A B C D : Nat)
variable (h1 : (B * A) % 10 = A % 10)
variable (h2 : ∀ (C : Nat), B - A = B % 10 ∧ C ≤ A)

theorem digit_D_is_five : D = 5 :=
by
  sorry

end digit_D_is_five_l1292_129249


namespace sum_of_perpendiculars_eq_altitude_l1292_129258

variables {A B C P A' B' C' : Type*}
variables (AB AC BC PA' PB' PC' h : ℝ)

-- Conditions
def is_isosceles_triangle (AB AC BC : ℝ) : Prop :=
  AB = AC

def point_inside_triangle (P A B C : Type*) : Prop :=
  true -- Assume point P is inside the triangle

def is_perpendiculars_dropped (PA' PB' PC' : ℝ) : Prop :=
  true -- Assume PA', PB', PC' are the lengths of the perpendiculars from P to the sides BC, CA, AB

def base_of_triangle (BC : ℝ) : Prop :=
  true -- Assume BC is the base of triangle

-- Theorem statement
theorem sum_of_perpendiculars_eq_altitude
  (h : ℝ) (AB AC BC PA' PB' PC' : ℝ)
  (isosceles : is_isosceles_triangle AB AC BC)
  (point_inside_triangle' : point_inside_triangle P A B C)
  (perpendiculars_dropped : is_perpendiculars_dropped PA' PB' PC')
  (base_of_triangle' : base_of_triangle BC) : 
  PA' + PB' + PC' = h := 
sorry

end sum_of_perpendiculars_eq_altitude_l1292_129258


namespace find_middle_number_l1292_129224

theorem find_middle_number (a b c d e : ℝ) (h1 : (a + b + c + d + e) / 5 = 12.5)
  (h2 : a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e)
  (h3 : (a + b + c) / 3 = 11.6)
  (h4 : (c + d + e) / 3 = 13.5) : c = 12.8 :=
sorry

end find_middle_number_l1292_129224


namespace find_marks_in_english_l1292_129219

theorem find_marks_in_english 
    (avg : ℕ) (math_marks : ℕ) (physics_marks : ℕ) (chemistry_marks : ℕ) (biology_marks : ℕ) (total_subjects : ℕ)
    (avg_eq : avg = 78) 
    (math_eq : math_marks = 65) 
    (physics_eq : physics_marks = 82) 
    (chemistry_eq : chemistry_marks = 67) 
    (biology_eq : biology_marks = 85) 
    (subjects_eq : total_subjects = 5) : 
    math_marks + physics_marks + chemistry_marks + biology_marks + E = 78 * 5 → 
    E = 91 :=
by sorry

end find_marks_in_english_l1292_129219


namespace sum_of_cubes_pattern_l1292_129217

theorem sum_of_cubes_pattern :
  (1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 = 21^2) :=
by
  sorry

end sum_of_cubes_pattern_l1292_129217


namespace find_a_range_l1292_129229

noncomputable
def f (x : ℝ) (a : ℝ) : ℝ :=
  if x ≤ 0 then x * Real.exp x else a * x ^ 2 - 2 * x

theorem find_a_range (a : ℝ) : (∀ x : ℝ, -1 / Real.exp 1 ≤ f x a) → a ∈ Set.Ici (Real.exp 1) :=
  sorry

end find_a_range_l1292_129229


namespace players_per_group_l1292_129247

-- Definitions for given conditions
def num_new_players : Nat := 48
def num_returning_players : Nat := 6
def num_groups : Nat := 9

-- Proof that the number of players in each group is 6
theorem players_per_group :
  let total_players := num_new_players + num_returning_players
  total_players / num_groups = 6 := by
  sorry

end players_per_group_l1292_129247


namespace sheets_in_height_l1292_129250

theorem sheets_in_height (sheets_per_ream : ℕ) (thickness_per_ream : ℝ) (target_thickness : ℝ) 
  (h₀ : sheets_per_ream = 500) (h₁ : thickness_per_ream = 5.0) (h₂ : target_thickness = 7.5) :
  target_thickness / (thickness_per_ream / sheets_per_ream) = 750 :=
by sorry

end sheets_in_height_l1292_129250


namespace length_proof_l1292_129230

noncomputable def length_of_plot 
  (b : ℝ) -- breadth in meters
  (fence_cost_flat : ℝ) -- cost of fencing per meter on flat ground
  (height_rise : ℝ) -- total height rise in meters
  (total_cost: ℝ) -- total cost of fencing
  (length_increase : ℝ) -- length increase in meters more than breadth
  (cost_increase_rate : ℝ) -- percentage increase in cost per meter rise in height
  (breadth_cost_increase_factor : ℝ) -- scaling factor for cost increase on breadth
  (increased_breadth_cost_rate : ℝ) -- actual increased cost rate per meter for breadth
: ℝ :=
2 * (b + length_increase) * fence_cost_flat + 
2 * b * (fence_cost_flat + fence_cost_flat * (height_rise * cost_increase_rate))

theorem length_proof
  (b : ℝ) -- breadth in meters
  (fence_cost_flat : ℝ := 26.50) -- cost of fencing per meter on flat ground
  (height_rise : ℝ := 5) -- total height rise in meters
  (total_cost: ℝ := 5300) -- total cost of fencing
  (length_increase : ℝ := 20) -- length increase in meters more than breadth
  (cost_increase_rate : ℝ := 0.10) -- percentage increase in cost per meter rise in height
  (breadth_cost_increase_factor : ℝ := fence_cost_flat * 0.5) -- increased cost factor
  (increased_breadth_cost_rate : ℝ := 39.75) -- recalculated cost rate per meter for breadth
  (length: ℝ := b + length_increase)
  (proof_step : total_cost = length_of_plot b fence_cost_flat height_rise total_cost length_increase cost_increase_rate breadth_cost_increase_factor increased_breadth_cost_rate)
: length = 52 :=
by
  sorry -- Proof omitted

end length_proof_l1292_129230


namespace purchasing_plans_and_optimal_plan_l1292_129241

def company_time := 10
def model_A_cost := 60000
def model_B_cost := 40000
def model_A_production := 15
def model_B_production := 10
def budget := 440000
def production_capacity := 102

theorem purchasing_plans_and_optimal_plan (x y : ℕ) (h1 : x + y = company_time) (h2 : model_A_cost * x + model_B_cost * y ≤ budget) :
  (x = 0 ∧ y = 10) ∨ (x = 1 ∧ y = 9) ∨ (x = 2 ∧ y = 8) ∧ (x = 1 ∧ y = 9) :=
by 
  sorry

end purchasing_plans_and_optimal_plan_l1292_129241


namespace sum_in_range_l1292_129242

def a : ℚ := 4 + 1/4
def b : ℚ := 2 + 3/4
def c : ℚ := 7 + 1/8

theorem sum_in_range : 14 < a + b + c ∧ a + b + c < 15 := by
  sorry

end sum_in_range_l1292_129242


namespace viable_combinations_l1292_129221

-- Given conditions
def totalHerbs : Nat := 4
def totalCrystals : Nat := 6
def incompatibleComb1 : Nat := 2
def incompatibleComb2 : Nat := 1

-- Theorem statement proving the number of viable combinations
theorem viable_combinations : totalHerbs * totalCrystals - (incompatibleComb1 + incompatibleComb2) = 21 := by
  sorry

end viable_combinations_l1292_129221


namespace find_principal_sum_l1292_129266

theorem find_principal_sum 
  (CI SI P : ℝ) 
  (R : ℝ) 
  (T : ℝ) 
  (hCI : CI = 11730) 
  (hSI : SI = 10200) 
  (hT : T = 2) 
  (hCI_formula : CI = P * ((1 + R / 100)^T - 1)) 
  (hSI_formula : SI = (P * R * T) / 100) 
  (h_diff : CI - SI = 1530) :
  P = 34000 := 
by 
  sorry

end find_principal_sum_l1292_129266
