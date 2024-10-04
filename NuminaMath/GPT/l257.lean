import Mathlib

namespace find_M_l257_257453

theorem find_M 
  (M : ℕ)
  (h : 997 + 999 + 1001 + 1003 + 1005 = 5100 - M) :
  M = 95 :=
by
  sorry

end find_M_l257_257453


namespace cut_problem_l257_257046

theorem cut_problem (n : ℕ) : (1 / 2 : ℝ) ^ n = 1 / 64 ↔ n = 6 :=
by
  sorry

end cut_problem_l257_257046


namespace arccos_one_over_sqrt_two_l257_257710

theorem arccos_one_over_sqrt_two : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l257_257710


namespace sum_a_b_c_l257_257478

theorem sum_a_b_c (a b c : ℕ) (h : a = 5 ∧ b = 10 ∧ c = 14) : a + b + c = 29 :=
by
  sorry

end sum_a_b_c_l257_257478


namespace inequality_problem_l257_257118

theorem inequality_problem (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c :=
by
  sorry

end inequality_problem_l257_257118


namespace angle_B_shape_triangle_l257_257998

variable {a b c R : ℝ} 

theorem angle_B_shape_triangle 
  (h1 : c > a ∧ c > b)
  (h2 : b = Real.sqrt 3 * R)
  (h3 : b * Real.sin (Real.arcsin (b / (2 * R))) = (a + c) * Real.sin (Real.arcsin (a / (2 * R)))) :
  (Real.arcsin (b / (2 * R)) = Real.pi / 3 ∧ a = c / 2 ∧ Real.arcsin (a / (2 * R)) = Real.pi / 6 ∧ Real.arcsin (c / (2 * R)) = Real.pi / 2) :=
by
  sorry

end angle_B_shape_triangle_l257_257998


namespace find_n_l257_257861

-- Define that Amy bought and sold 15n avocados.
def bought_sold_avocados (n : ℕ) := 15 * n

-- Define the profit function.
def calculate_profit (n : ℕ) : ℤ := 
  let total_cost := 10 * n
  let total_earnings := 12 * n
  total_earnings - total_cost

theorem find_n (n : ℕ) (profit : ℤ) (h1 : profit = 100) (h2 : profit = calculate_profit n) : n = 50 := 
by 
  sorry

end find_n_l257_257861


namespace slices_remaining_is_correct_l257_257286

def slices_per_pizza : ℕ := 8
def pizzas_ordered : ℕ := 2
def slices_eaten : ℕ := 7
def total_slices : ℕ := slices_per_pizza * pizzas_ordered
def slices_remaining : ℕ := total_slices - slices_eaten

theorem slices_remaining_is_correct : slices_remaining = 9 := by
  sorry

end slices_remaining_is_correct_l257_257286


namespace number_of_freshmen_l257_257468

theorem number_of_freshmen (n : ℕ) : n < 450 ∧ n % 19 = 18 ∧ n % 17 = 10 → n = 265 := by
  sorry

end number_of_freshmen_l257_257468


namespace dart_lands_in_center_hexagon_l257_257374

noncomputable def area_regular_hexagon (s : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2) * s^2

theorem dart_lands_in_center_hexagon {s : ℝ} (h : s > 0) :
  let A_outer := area_regular_hexagon s
  let A_inner := area_regular_hexagon (s / 2)
  (A_inner / A_outer) = 1 / 4 :=
by
  let A_outer := area_regular_hexagon s
  let A_inner := area_regular_hexagon (s / 2)
  sorry

end dart_lands_in_center_hexagon_l257_257374


namespace arccos_one_over_sqrt_two_l257_257744

theorem arccos_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l257_257744


namespace second_term_is_4_l257_257145

-- Define the arithmetic sequence conditions
variables (a d : ℝ) -- first term a, common difference d

-- The condition given in the problem
def sum_first_and_third_term (a d : ℝ) : Prop :=
  a + (a + 2 * d) = 8

-- What we need to prove
theorem second_term_is_4 (a d : ℝ) (h : sum_first_and_third_term a d) : a + d = 4 :=
sorry

end second_term_is_4_l257_257145


namespace cos_double_angle_trig_identity_l257_257413

theorem cos_double_angle_trig_identity
  (α : ℝ) 
  (h : Real.sin (α - Real.pi / 3) = 4 / 5) : 
  Real.cos (2 * α + Real.pi / 3) = 7 / 25 :=
by
  sorry

end cos_double_angle_trig_identity_l257_257413


namespace erica_total_earnings_l257_257918

def fishPrice : Nat := 20
def pastCatch : Nat := 80
def todayCatch : Nat := 2 * pastCatch
def pastEarnings := pastCatch * fishPrice
def todayEarnings := todayCatch * fishPrice
def totalEarnings := pastEarnings + todayEarnings

theorem erica_total_earnings : totalEarnings = 4800 := by
  sorry

end erica_total_earnings_l257_257918


namespace number_of_solutions_l257_257254

def f (x : ℝ) : ℝ := |1 - 2 * x|

theorem number_of_solutions :
  (∃ n : ℕ, n = 8 ∧ ∀ x ∈ [0,1], f (f (f x)) = (1 / 2) * x) :=
sorry

end number_of_solutions_l257_257254


namespace sphere_surface_area_l257_257645

theorem sphere_surface_area (edge_length : ℝ) (diameter_eq_edge_length : (diameter : ℝ) = edge_length) :
  (edge_length = 2) → (diameter = 2) → (surface_area : ℝ) = 8 * Real.pi :=
by
  sorry

end sphere_surface_area_l257_257645


namespace group_8_extracted_number_is_72_l257_257640

-- Definitions related to the problem setup
def individ_to_group (n : ℕ) : ℕ := n / 10 + 1
def unit_digit (n : ℕ) : ℕ := n % 10
def extraction_rule (k m : ℕ) : ℕ := (k + m - 1) % 10

-- Given condition: total individuals split into sequential groups and m = 5
def total_individuals : ℕ := 100
def total_groups : ℕ := 10
def m : ℕ := 5
def k_8 : ℕ := 8

-- The final theorem statement
theorem group_8_extracted_number_is_72 : ∃ n : ℕ, individ_to_group n = k_8 ∧ unit_digit n = extraction_rule k_8 m := by
  sorry

end group_8_extracted_number_is_72_l257_257640


namespace intersection_M_N_l257_257137

def M : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
def N : Set ℝ := { y | ∃ x, y = x^2 ∧ -1 ≤ x ∧ x ≤ 1 }

theorem intersection_M_N :
  M ∩ N = { z | 0 ≤ z ∧ z ≤ 1 } := by
  sorry

end intersection_M_N_l257_257137


namespace arithmetic_seq_a7_a8_l257_257992

theorem arithmetic_seq_a7_a8 (a : ℕ → ℤ) (d : ℤ) (h₁ : a 1 + a 2 = 4) (h₂ : d = 2) :
  a 7 + a 8 = 28 := by
  sorry

end arithmetic_seq_a7_a8_l257_257992


namespace express_set_A_l257_257283

def A := {x : ℤ | -1 < abs (x - 1) ∧ abs (x - 1) < 2}

theorem express_set_A : A = {0, 1, 2} := 
by
  sorry

end express_set_A_l257_257283


namespace sequence_formula_l257_257791

theorem sequence_formula (a : ℕ → ℤ) (h1 : a 1 = 1)
  (h2 : ∀ n: ℕ, a (n + 1) = 2 * a n + n * (1 + 2^n)) :
  ∀ n : ℕ, a n = 2^(n - 2) * (n^2 - n + 6) - n - 1 :=
by intro n; sorry

end sequence_formula_l257_257791


namespace perfect_square_trinomial_l257_257277

theorem perfect_square_trinomial (a b : ℝ) :
  (∃ c : ℝ, 4 * (c^2) = 9 ∧ 4 * c = a - b) → 2 * a - 2 * b = 24 ∨ 2 * a - 2 * b = -24 :=
by
  sorry

end perfect_square_trinomial_l257_257277


namespace correct_statements_eq_3_l257_257787

def class (k : ℤ) : Set ℤ := { n | ∃ m : ℤ, n = 5 * m + k }

def statement_1 : Prop := 2013 ∈ class 3

def statement_2 : Prop := -2 ∈ class 2

def statement_3 : Prop := ( ⋃ k in {0, 1, 2, 3, 4}, class k ) = Set.univ

def statement_4 : Prop := ∀ a b : ℤ, (a - b) % 5 = 0 ↔ (a ∈ class 0 ∧ b ∈ class 0)

def correct_statements : ℕ :=
  [statement_1, statement_2, statement_3, statement_4].count true

theorem correct_statements_eq_3 : correct_statements = 3 := by
  sorry

end correct_statements_eq_3_l257_257787


namespace shortest_part_is_15_l257_257211

namespace ProofProblem

def rope_length : ℕ := 60
def ratio_part1 : ℕ := 3
def ratio_part2 : ℕ := 4
def ratio_part3 : ℕ := 5

def total_parts := ratio_part1 + ratio_part2 + ratio_part3
def length_per_part := rope_length / total_parts
def shortest_part_length := ratio_part1 * length_per_part

theorem shortest_part_is_15 :
  shortest_part_length = 15 := by
  sorry

end ProofProblem

end shortest_part_is_15_l257_257211


namespace problem_solution_l257_257119

theorem problem_solution (a b : ℝ) (h1 : b > a) (h2 : a > 0) :
  a^2 < b^2 ∧ ab < b^2 :=
sorry

end problem_solution_l257_257119


namespace seeds_in_big_garden_is_correct_l257_257915

def total_seeds : ℕ := 41
def small_gardens : ℕ := 3
def seeds_per_small_garden : ℕ := 4

def seeds_in_small_gardens : ℕ := small_gardens * seeds_per_small_garden
def seeds_in_big_garden : ℕ := total_seeds - seeds_in_small_gardens

theorem seeds_in_big_garden_is_correct : seeds_in_big_garden = 29 := by
  -- proof goes here
  sorry

end seeds_in_big_garden_is_correct_l257_257915


namespace solve_for_x_l257_257823

theorem solve_for_x (x : ℝ) : x^2 + 6 * x + 8 = -(x + 4) * (x + 6) ↔ x = -4 :=
by {
  sorry
}

end solve_for_x_l257_257823


namespace triple_angle_l257_257015

theorem triple_angle (α : ℝ) : 3 * α = α + α + α := 
by sorry

end triple_angle_l257_257015


namespace bogatyrs_truthful_count_l257_257901

noncomputable def number_of_truthful_warriors (total_warriors: ℕ) (sword_yes: ℕ) (spear_yes: ℕ) (axe_yes: ℕ) (bow_yes: ℕ) : ℕ :=
  let total_yes := sword_yes + spear_yes + axe_yes + bow_yes
  let lying_warriors := (total_yes - total_warriors) / 2
  total_warriors - lying_warriors

theorem bogatyrs_truthful_count :
  number_of_truthful_warriors 33 13 15 20 27 = 12 := by
  sorry

end bogatyrs_truthful_count_l257_257901


namespace correct_propositions_l257_257598

-- Definitions based on the propositions
def prop1 := 
"Sampling every 20 minutes from a uniformly moving production line is stratified sampling."

def prop2 := 
"The stronger the correlation between two random variables, the closer the absolute value of the correlation coefficient is to 1."

def prop3 := 
"In the regression line equation hat_y = 0.2 * x + 12, the forecasted variable hat_y increases by 0.2 units on average for each unit increase in the explanatory variable x."

def prop4 := 
"For categorical variables X and Y, the smaller the observed value k of their statistic K², the greater the certainty of the relationship between X and Y."

-- Mathematical statements for propositions
def p1 : Prop := false -- Proposition ① is incorrect
def p2 : Prop := true  -- Proposition ② is correct
def p3 : Prop := true  -- Proposition ③ is correct
def p4 : Prop := false -- Proposition ④ is incorrect

-- The theorem we need to prove
theorem correct_propositions : (p2 = true) ∧ (p3 = true) :=
by 
  -- Details of the proof here
  sorry

end correct_propositions_l257_257598


namespace Seokjin_total_fish_l257_257627

-- Define the conditions
def fish_yesterday := 10
def cost_yesterday := 3000
def additional_cost := 6000
def price_per_fish := cost_yesterday / fish_yesterday
def total_cost_today := cost_yesterday + additional_cost
def fish_today := total_cost_today / price_per_fish

-- Define the goal
theorem Seokjin_total_fish (h1 : fish_yesterday = 10)
                           (h2 : cost_yesterday = 3000)
                           (h3 : additional_cost = 6000)
                           (h4 : price_per_fish = cost_yesterday / fish_yesterday)
                           (h5 : total_cost_today = cost_yesterday + additional_cost)
                           (h6 : fish_today = total_cost_today / price_per_fish) :
  fish_yesterday + fish_today = 40 :=
by
  sorry

end Seokjin_total_fish_l257_257627


namespace trigonometric_identity_l257_257412

theorem trigonometric_identity
  (α : ℝ)
  (h : Real.sin (π / 6 - α) = 1 / 3) :
  2 * Real.cos (π / 6 + α / 2) ^ 2 - 1 = 1 / 3 := by
  sorry

end trigonometric_identity_l257_257412


namespace Erica_Ice_Cream_Spend_l257_257250

theorem Erica_Ice_Cream_Spend :
  (6 * ((3 * 2.00) + (2 * 1.50) + (2 * 3.00))) = 90 := sorry

end Erica_Ice_Cream_Spend_l257_257250


namespace find_x_l257_257946

open Nat

theorem find_x (n : ℕ) (x : ℕ) 
  (h1 : x = 2^n - 32)
  (h2 : (factors x).nodup)
  (h3 : (factors x).length = 3)
  (h4 : 3 ∈ factors x) :
  x = 480 ∨ x = 2016 := 
sorry

end find_x_l257_257946


namespace daughter_and_child_weight_l257_257004

variables (M D C : ℝ)

-- Conditions
def condition1 : Prop := M + D + C = 160
def condition2 : Prop := D = 40
def condition3 : Prop := C = (1/5) * M

-- Goal (Question)
def goal : Prop := D + C = 60

theorem daughter_and_child_weight
  (h1 : condition1 M D C)
  (h2 : condition2 D)
  (h3 : condition3 M C) : goal D C :=
by
  sorry

end daughter_and_child_weight_l257_257004


namespace smallest_special_number_gt_3429_l257_257100

open Set

def is_special_number (n : ℕ) : Prop :=
  (fintype.card (fintype.of_finset (finset.of_digits (nat.digits 10 n)) nat.digits_dec_eq)) = 4

theorem smallest_special_number_gt_3429 :
  ∃ n : ℕ, n > 3429 ∧ is_special_number n ∧ (∀ m : ℕ, m > 3429 ∧ is_special_number m → n ≤ m) :=
exists.intro 3450 (and.intro (by decide) (and.intro (by decide) (by decide)))

end smallest_special_number_gt_3429_l257_257100


namespace max_value_of_expression_l257_257282

theorem max_value_of_expression (x y : ℝ) 
  (h : (x - 4)^2 / 4 + y^2 / 9 = 1) : 
  (x^2 / 4 + y^2 / 9 ≤ 9) ∧ ∃ x y, (x - 4)^2 / 4 + y^2 / 9 = 1 ∧ x^2 / 4 + y^2 / 9 = 9 :=
by
  sorry

end max_value_of_expression_l257_257282


namespace enterprise_b_pays_more_in_2015_l257_257862

variable (a b x y : ℝ)
variable (ha2x : a + 2 * x = b)
variable (ha1y : a * (1+y)^2 = b)

theorem enterprise_b_pays_more_in_2015 : b * (1 + y) > b + x := by
  sorry

end enterprise_b_pays_more_in_2015_l257_257862


namespace largest_integer_satisfying_inequality_l257_257752

theorem largest_integer_satisfying_inequality :
  ∃ n : ℤ, n = 4 ∧ (1 / 4 + n / 8 < 7 / 8) ∧ ∀ m : ℤ, m > 4 → ¬(1 / 4 + m / 8 < 7 / 8) :=
by
  sorry

end largest_integer_satisfying_inequality_l257_257752


namespace opposite_event_of_hitting_at_least_once_is_missing_both_times_l257_257859

theorem opposite_event_of_hitting_at_least_once_is_missing_both_times
  (A B : Prop) :
  ¬(A ∨ B) ↔ (¬A ∧ ¬B) :=
by
  sorry

end opposite_event_of_hitting_at_least_once_is_missing_both_times_l257_257859


namespace Petya_can_verify_coins_l257_257808

theorem Petya_can_verify_coins :
  ∃ (c₁ c₂ c₃ c₅ : ℕ), 
  (c₁ = 1 ∧ c₂ = 2 ∧ c₃ = 3 ∧ c₅ = 5) ∧
  (∃ (w : ℕ), w = 9) ∧
  (∃ (cond : ℕ → Prop), 
    cond 1 ∧ cond 2 ∧ cond 3 ∧ cond 5) := sorry

end Petya_can_verify_coins_l257_257808


namespace value_of_y_l257_257271

theorem value_of_y (x y : ℝ) (h₁ : 1.5 * x = 0.75 * y) (h₂ : x = 20) : y = 40 :=
sorry

end value_of_y_l257_257271


namespace truthful_warriors_count_l257_257910

-- Noncomputable theory is not necessary here.
-- We define the number of warriors and their responses.
def warriors_count : ℕ := 33
def yes_sword : ℕ := 13
def yes_spear : ℕ := 15
def yes_axe : ℕ := 20
def yes_bow : ℕ := 27

-- The equation from the solution steps where x is the number of truthful warriors.
def total_yes := yes_sword + yes_spear + yes_axe + yes_bow

theorem truthful_warriors_count :
  ∃ x : ℕ, x + 3 * (warriors_count - x) = total_yes ∧ x = 12 :=
by {
  -- We state that there exists an x such that its value satisfies the total "yes" responses equation.
  -- Here, we assert that x equals 12.
  use 12,
  split,
  {
    -- Prove the equation x + 3 * (warriors_count - x) = total_yes
    rw [warriors_count],
    rw [total_yes],
    norm_num,
  },
  {
    -- State that x = 12
    refl,
  },
}

end truthful_warriors_count_l257_257910


namespace min_of_x_squared_y_squared_z_squared_l257_257451

theorem min_of_x_squared_y_squared_z_squared (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  x^2 + y^2 + z^2 ≥ 4 :=
by sorry

end min_of_x_squared_y_squared_z_squared_l257_257451


namespace sum_of_cubes_eq_neg_27_l257_257798

theorem sum_of_cubes_eq_neg_27
  (a b c : ℝ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_condition : (a^3 + 9) / a = (b^3 + 9) / b ∧ (b^3 + 9) / b = (c^3 + 9) / c) :
  a^3 + b^3 + c^3 = -27 :=
sorry

end sum_of_cubes_eq_neg_27_l257_257798


namespace number_of_truthful_warriors_l257_257882

theorem number_of_truthful_warriors (total_warriors : ℕ) 
  (sword_yes : ℕ) (spear_yes : ℕ) (axe_yes : ℕ) (bow_yes : ℕ) 
  (always_tells_truth : ℕ → Prop)
  (always_lies : ℕ → Prop)
  (hv1 : total_warriors = 33)
  (hv2 : sword_yes = 13)
  (hv3 : spear_yes = 15)
  (hv4 : axe_yes = 20)
  (hv5 : bow_yes = 27) :
  ∃ truthful_warriors, truthful_warriors = 12 := 
by {
  sorry
}

end number_of_truthful_warriors_l257_257882


namespace convert_base_8_to_base_10_l257_257083

def to_base_10 (n : ℕ) (b : ℕ) (digits : List ℕ) : ℕ :=
  digits.foldr (λ digit acc => acc * b + digit) 0

theorem convert_base_8_to_base_10 : 
  to_base_10 10 8 [6, 4, 2] = 166 := by
  sorry

end convert_base_8_to_base_10_l257_257083


namespace discounted_price_l257_257644

variable (marked_price : ℝ) (discount_rate : ℝ)
variable (marked_price_def : marked_price = 150)
variable (discount_rate_def : discount_rate = 20)

theorem discounted_price (hmp : marked_price = 150) (hdr : discount_rate = 20) : 
  marked_price - (discount_rate / 100) * marked_price = 120 := by
  rw [hmp, hdr]
  sorry

end discounted_price_l257_257644


namespace smallest_special_gt_3429_l257_257105

def is_special (n : ℕ) : Prop :=
  (10^3 ≤ n ∧ n < 10^4) ∧ (List.length (n.digits 10).eraseDup = 4)

theorem smallest_special_gt_3429 : 
  ∃ n : ℕ, n > 3429 ∧ is_special n ∧ ∀ m : ℕ, m > 3429 ∧ is_special m → n ≤ m := 
begin
  use 3450,
  split,
  { exact nat.succ_lt_succ (nat.s succ_lt_succ (nat.succ_lt_succ (nat.succ_lt_succ (nat.lt_succ_self 3430)))) },
  split,
  { unfold is_special,
    split,
    { split,
        { linarith },
    { linarith } },
    { unfold List.eraseDup,
    unfold List.redLength,
    exactly simp } },
  { intros m hm1 hm2,
    interval_cases m,
    sorry },
end

end smallest_special_gt_3429_l257_257105


namespace three_digit_numbers_l257_257267

theorem three_digit_numbers (n : ℕ) :
  n = 4 ↔ ∃ (x y : ℕ), 
  (100 ≤ 101 * x + 10 * y ∧ 101 * x + 10 * y < 1000) ∧ 
  (x ≠ 0 ∧ x ≠ 5) ∧ 
  (2 * x + y = 15) ∧ 
  (y < 10) :=
by { sorry }

end three_digit_numbers_l257_257267


namespace represent_2021_as_squares_l257_257206

theorem represent_2021_as_squares :
  ∃ n : ℕ, n = 505 → 2021 = (n + 1)^2 - (n - 1)^2 + 1^2 :=
by
  sorry

end represent_2021_as_squares_l257_257206


namespace num_packages_l257_257490

-- Defining the given conditions
def packages_count_per_package := 6
def total_tshirts := 426

-- The statement to be proved
theorem num_packages : (total_tshirts / packages_count_per_package) = 71 :=
by sorry

end num_packages_l257_257490


namespace fraction_collectors_edition_is_correct_l257_257391

-- Let's define the necessary conditions
variable (DinaDolls IvyDolls CollectorsEditionDolls : ℕ)
variable (FractionCollectorsEdition : ℚ)

-- Given conditions
axiom DinaHas60Dolls : DinaDolls = 60
axiom DinaHasTwiceAsManyDollsAsIvy : DinaDolls = 2 * IvyDolls
axiom IvyHas20CollectorsEditionDolls : CollectorsEditionDolls = 20

-- The statement to prove
theorem fraction_collectors_edition_is_correct :
  FractionCollectorsEdition = (CollectorsEditionDolls : ℚ) / (IvyDolls : ℚ) ∧
  DinaDolls = 60 →
  DinaDolls = 2 * IvyDolls →
  CollectorsEditionDolls = 20 →
  FractionCollectorsEdition = 2 / 3 := 
by
  sorry

end fraction_collectors_edition_is_correct_l257_257391


namespace peyton_juice_boxes_needed_l257_257584

def juice_boxes_needed
  (john_juice_per_day : ℕ)
  (samantha_juice_per_day : ℕ)
  (heather_juice_mon_wed : ℕ)
  (heather_juice_tue_thu : ℕ)
  (heather_juice_fri : ℕ)
  (john_weeks : ℕ)
  (samantha_weeks : ℕ)
  (heather_weeks : ℕ)
  : ℕ :=
  let john_juice_per_week := john_juice_per_day * 5
  let samantha_juice_per_week := samantha_juice_per_day * 5
  let heather_juice_per_week := heather_juice_mon_wed * 2 + heather_juice_tue_thu * 2 + heather_juice_fri
  let john_total_juice := john_juice_per_week * john_weeks
  let samantha_total_juice := samantha_juice_per_week * samantha_weeks
  let heather_total_juice := heather_juice_per_week * heather_weeks
  john_total_juice + samantha_total_juice + heather_total_juice

theorem peyton_juice_boxes_needed :
  juice_boxes_needed 2 1 3 2 1 25 20 25 = 625 :=
by
  sorry

end peyton_juice_boxes_needed_l257_257584


namespace second_term_is_4_l257_257146

-- Define the arithmetic sequence conditions
variables (a d : ℝ) -- first term a, common difference d

-- The condition given in the problem
def sum_first_and_third_term (a d : ℝ) : Prop :=
  a + (a + 2 * d) = 8

-- What we need to prove
theorem second_term_is_4 (a d : ℝ) (h : sum_first_and_third_term a d) : a + d = 4 :=
sorry

end second_term_is_4_l257_257146


namespace solution_set_inequality_l257_257547

theorem solution_set_inequality (m : ℤ) (h₁ : (∃! x : ℤ, |2 * x - m| ≤ 1 ∧ x = 2)) :
  {x : ℝ | |x - 1| + |x - 3| ≥ m} = {x : ℝ | x ≤ 0} ∪ {x : ℝ | x ≥ 4} :=
by
  -- The detailed proof would be added here.
  sorry

end solution_set_inequality_l257_257547


namespace range_of_x_l257_257143

theorem range_of_x (x : ℝ) (h : 2 * x - 4 ≥ 0) : x ≥ 2 :=
sorry

end range_of_x_l257_257143


namespace incorrect_conclusion_l257_257962

-- Define the given parabola.
def parabola (x : ℝ) : ℝ := (x - 2)^2 + 1

-- Define the conditions for the parabola.
def parabola_opens_upwards : Prop := ∀ x y : ℝ, parabola (x + y) = (x + y - 2)^2 + 1
def axis_of_symmetry : Prop := ∀ x : ℝ, parabola x = parabola (4 - x)
def vertex_coordinates : Prop := parabola 2 = 1 ∧ (parabola 2, 2) = (1, 2)
def behavior_when_x_less_than_2 : Prop := ∀ x : ℝ, x < 2 → parabola x < parabola (x + 1)

-- The statement that needs to be proven in Lean 4.
theorem incorrect_conclusion : ¬ behavior_when_x_less_than_2 :=
  by
  sorry

end incorrect_conclusion_l257_257962


namespace Harvard_attendance_l257_257994

theorem Harvard_attendance:
  (total_applicants : ℕ) (acceptance_rate : ℝ) (attendance_rate : ℝ) 
  (h1 : total_applicants = 20000) 
  (h2 : acceptance_rate = 0.05) 
  (h3 : attendance_rate = 0.9) :
  ∃ (number_attending : ℕ), number_attending = 900 := 
by 
  sorry

end Harvard_attendance_l257_257994


namespace truthfulness_count_l257_257885

-- Define variables to represent the number of warriors and their response counts
def num_warriors : Nat := 33
def yes_sword : Nat := 13
def yes_spear : Nat := 15
def yes_axe : Nat := 20
def yes_bow : Nat := 27

-- Define the total number of "Yes" answers
def total_yes_answers : Nat := yes_sword + yes_spear + yes_axe + yes_bow

theorem truthfulness_count :
  ∃ x : Nat, x + 3 * (num_warriors - x) = total_yes_answers ∧ x = 12 :=
by
  sorry

end truthfulness_count_l257_257885


namespace log_sqrt_defined_l257_257941

open Real

-- Define the conditions for the logarithm and square root arguments
def log_condition (x : ℝ) : Prop := 4 * x - 7 > 0
def sqrt_condition (x : ℝ) : Prop := 2 * x - 3 ≥ 0

-- Define the combined condition
def combined_condition (x : ℝ) : Prop := x > 7 / 4

-- The proof statement
theorem log_sqrt_defined (x : ℝ) : combined_condition x ↔ log_condition x ∧ sqrt_condition x :=
by
  -- Work through the equivalence and proof steps
  sorry

end log_sqrt_defined_l257_257941


namespace foreign_students_next_semester_l257_257519

theorem foreign_students_next_semester (total_students : ℕ) (percent_foreign : ℝ) (new_foreign_students : ℕ) 
  (h_total : total_students = 1800) (h_percent : percent_foreign = 0.30) (h_new : new_foreign_students = 200) : 
  (0.30 * 1800 + 200 : ℝ) = 740 := by
  sorry

end foreign_students_next_semester_l257_257519


namespace days_to_finish_by_b_l257_257844

theorem days_to_finish_by_b (A B C : ℚ) 
  (h1 : A + B + C = 1 / 5) 
  (h2 : A = 1 / 9) 
  (h3 : A + C = 1 / 7) : 
  1 / B = 12.115 :=
by
  sorry

end days_to_finish_by_b_l257_257844


namespace pyramid_volume_l257_257329

noncomputable def volume_of_pyramid (l : ℝ) : ℝ :=
  (l^3 / 24) * (Real.sqrt (Real.sqrt 2 + 1))

theorem pyramid_volume (l : ℝ) (α β : ℝ)
  (hα : α = π / 8)
  (hβ : β = π / 4)
  (hl : l = 6) :
  volume_of_pyramid l = 9 * Real.sqrt (Real.sqrt 2 + 1) := by
  sorry

end pyramid_volume_l257_257329


namespace smaller_number_l257_257596

theorem smaller_number (x y : ℝ) (h1 : x - y = 1650) (h2 : 0.075 * x = 0.125 * y) : y = 2475 := 
sorry

end smaller_number_l257_257596


namespace f_divisible_by_13_l257_257441

def f : ℕ → ℤ := sorry

theorem f_divisible_by_13 :
  (f 0 = 0) ∧ (f 1 = 0) ∧
  (∀ n, f (n + 2) = 4 ^ (n + 2) * f (n + 1) - 16 ^ (n + 1) * f n + n * 2 ^ (n ^ 2)) →
  (f 1989 % 13 = 0) ∧ (f 1990 % 13 = 0) ∧ (f 1991 % 13 = 0) :=
by
  intros h
  sorry

end f_divisible_by_13_l257_257441


namespace find_cost_price_l257_257647

theorem find_cost_price (C S : ℝ) (h1 : S = 1.35 * C) (h2 : S - 25 = 0.98 * C) : C = 25 / 0.37 :=
by
  sorry

end find_cost_price_l257_257647


namespace arccos_of_one_over_sqrt_two_l257_257721

theorem arccos_of_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
sorry

end arccos_of_one_over_sqrt_two_l257_257721


namespace investor_difference_l257_257658

/-
Scheme A yields 30% of the capital within a year.
Scheme B yields 50% of the capital within a year.
Investor invested $300 in scheme A.
Investor invested $200 in scheme B.
We need to prove that the difference in total money between scheme A and scheme B after a year is $90.
-/

def schemeA_yield_rate : ℝ := 0.30
def schemeB_yield_rate : ℝ := 0.50
def schemeA_investment : ℝ := 300
def schemeB_investment : ℝ := 200

def total_after_year (investment : ℝ) (yield_rate : ℝ) : ℝ :=
  investment * (1 + yield_rate)

theorem investor_difference :
  total_after_year schemeA_investment schemeA_yield_rate - total_after_year schemeB_investment schemeB_yield_rate = 90 := by
  sorry

end investor_difference_l257_257658


namespace probability_enemy_plane_hit_l257_257641

noncomputable def P_A : ℝ := 0.6
noncomputable def P_B : ℝ := 0.4

theorem probability_enemy_plane_hit : 1 - ((1 - P_A) * (1 - P_B)) = 0.76 :=
by
  sorry

end probability_enemy_plane_hit_l257_257641


namespace large_monkey_doll_cost_l257_257033

theorem large_monkey_doll_cost :
  ∃ (L : ℝ), (300 / L - 300 / (L - 2) = 25) ∧ L > 0 := by
  sorry

end large_monkey_doll_cost_l257_257033


namespace sandy_final_position_and_distance_l257_257636

-- Define the conditions as statements
def walked_south (distance : ℕ) := distance = 20
def turned_left_facing_east := true
def walked_east (distance : ℕ) := distance = 20
def turned_left_facing_north := true
def walked_north (distance : ℕ) := distance = 20
def turned_right_facing_east := true
def walked_east_again (distance : ℕ) := distance = 20

-- Final position computation as a proof statement
theorem sandy_final_position_and_distance :
  ∃ (d : ℕ) (dir : String), walked_south 20 → turned_left_facing_east → walked_east 20 →
  turned_left_facing_north → walked_north 20 →
  turned_right_facing_east → walked_east_again 20 ∧ d = 40 ∧ dir = "east" :=
by
  sorry

end sandy_final_position_and_distance_l257_257636


namespace difference_in_interest_rates_l257_257229

-- Definitions
def Principal : ℝ := 2300
def Time : ℝ := 3
def ExtraInterest : ℝ := 69

-- The difference in rates
theorem difference_in_interest_rates (R dR : ℝ) (h : (Principal * (R + dR) * Time) / 100 =
    (Principal * R * Time) / 100 + ExtraInterest) : dR = 1 :=
  sorry

end difference_in_interest_rates_l257_257229


namespace sum_lent_correct_l257_257042

def P : ℝ := 1000

-- Definitions based on conditions
def r : ℝ := 0.05
def t : ℝ := 5
def I : ℝ := P - 750

-- The simple interest calculation
def simple_interest (P r t : ℝ) : ℝ := P * r * t

theorem sum_lent_correct (P : ℝ) (r : ℝ) (t : ℝ) :
  (simple_interest P r t = P - 750) → P = 1000 :=
by
  intro h
  -- Proof omitted
  sorry

end sum_lent_correct_l257_257042


namespace find_x_l257_257945

open Nat

theorem find_x (n : ℕ) (x : ℕ) (h1 : x = 2^n - 32)
  (h2 : (3 : ℕ) ∣ x)
  (h3 : (factors x).length = 3) :
  x = 480 ∨ x = 2016 := by
  sorry

end find_x_l257_257945


namespace total_people_in_club_after_5_years_l257_257373

noncomputable def club_initial_people := 18
noncomputable def executives_per_year := 6
noncomputable def initial_regular_members := club_initial_people - executives_per_year

-- Define the function for regular members growth
noncomputable def regular_members_after_n_years (n : ℕ) : ℕ := initial_regular_members * 2 ^ n

-- Total people in the club after 5 years
theorem total_people_in_club_after_5_years : 
  club_initial_people + regular_members_after_n_years 5 - initial_regular_members = 390 :=
by
  sorry

end total_people_in_club_after_5_years_l257_257373


namespace angle_BAC_is_105_or_35_l257_257514

-- Definitions based on conditions
def arcAB : ℝ := 110
def arcAC : ℝ := 40
def arcBC_major : ℝ := 360 - (arcAB + arcAC)
def arcBC_minor : ℝ := arcAB - arcAC

-- The conjecture: proving that the inscribed angle ∠BAC is 105° or 35° given the conditions.
theorem angle_BAC_is_105_or_35
  (h1 : 0 < arcAB ∧ arcAB < 360)
  (h2 : 0 < arcAC ∧ arcAC < 360)
  (h3 : arcAB + arcAC < 360) :
  (arcBC_major / 2 = 105) ∨ (arcBC_minor / 2 = 35) :=
  sorry

end angle_BAC_is_105_or_35_l257_257514


namespace limit_calculation_l257_257866

open Real
open Complex

theorem limit_calculation :
  ∃ L : ℝ, 
  (tendsto (λ x, (2 - 3^(arctan (sqrt x))^2) ^ (2 / sin x)) (nhds 0) (nhds L)) ∧ L = 1 / 9 :=
sorry

end limit_calculation_l257_257866


namespace price_difference_is_7_42_l257_257327

def total_cost : ℝ := 80.34
def shirt_price : ℝ := 36.46
def sweater_price : ℝ := total_cost - shirt_price
def price_difference : ℝ := sweater_price - shirt_price

theorem price_difference_is_7_42 : price_difference = 7.42 :=
  by
    sorry

end price_difference_is_7_42_l257_257327


namespace range_of_f_l257_257760

noncomputable def f (x : ℝ) : ℝ := (3 * x + 8) / (x - 4)

theorem range_of_f : set.range f = set.univ \ {3} :=
by
  sorry

end range_of_f_l257_257760


namespace cyclist_speed_l257_257489

theorem cyclist_speed 
  (course_length : ℝ)
  (second_cyclist_speed : ℝ)
  (meeting_time : ℝ)
  (total_distance : ℝ)
  (condition1 : course_length = 45)
  (condition2 : second_cyclist_speed = 16)
  (condition3 : meeting_time = 1.5)
  (condition4 : total_distance = meeting_time * (second_cyclist_speed + 14))
  : (meeting_time * 14 + meeting_time * second_cyclist_speed = course_length) :=
by
  sorry

end cyclist_speed_l257_257489


namespace problem_1_problem_2_l257_257415

noncomputable section

variables {A B C : ℝ} {a b c : ℝ}

-- Condition definitions
def triangle_conditions (A B : ℝ) (a b c : ℝ) : Prop :=
  b = 3 ∧ c = 1 ∧ A = 2 * B

-- Problem 1: Prove that a = 2 * sqrt(3)
theorem problem_1 {A B C a : ℝ} (h : triangle_conditions A B a b c) : a = 2 * Real.sqrt 3 := sorry

-- Problem 2: Prove the value of cos(2A + π/6)
theorem problem_2 {A B C a : ℝ} (h : triangle_conditions A B a b c) : 
  Real.cos (2 * A + Real.pi / 6) = (4 * Real.sqrt 2 - 7 * Real.sqrt 3) / 18 := sorry

end problem_1_problem_2_l257_257415


namespace expand_product_l257_257243

theorem expand_product (y : ℝ) : 5 * (y - 3) * (y + 10) = 5 * y^2 + 35 * y - 150 :=
by 
  sorry

end expand_product_l257_257243


namespace sale_price_lower_by_2_5_percent_l257_257836

open Real

theorem sale_price_lower_by_2_5_percent (x : ℝ) : 
  let increased_price := 1.30 * x
  let sale_price := 0.75 * increased_price 
  sale_price = 0.975 * x :=
by
  let increased_price := 1.30 * x
  let sale_price := 0.75 * increased_price 
  show sale_price = 0.975 * x
  sorry

end sale_price_lower_by_2_5_percent_l257_257836


namespace probability_order_black_red_white_l257_257987

noncomputable def probability_black_red_white (total_balls : ℕ) (black : ℕ) (red : ℕ) (white : ℕ) : ℚ :=
  (black / total_balls) * ((red) / (total_balls - 1)) * ((white) / (total_balls - 2))

theorem probability_order_black_red_white :
  let total_balls := 15 in
  let black := 6 in
  let red := 5 in
  let white := 4 in
  probability_black_red_white total_balls black red white = 4 / 91 :=
by 
  sorry

end probability_order_black_red_white_l257_257987


namespace bonnets_difference_thursday_monday_l257_257580

variable (Bm Bt Bf : ℕ)

-- Conditions
axiom monday_bonnets_made : Bm = 10
axiom tuesday_wednesday_bonnets_made : Bm + (2 * Bm) = 30
axiom bonnets_sent_to_orphanages : (Bm + Bt + (Bt - 5) + Bm + (2 * Bm)) / 5 = 11
axiom friday_bonnets_made : Bf = Bt - 5

theorem bonnets_difference_thursday_monday :
  Bt - Bm = 5 :=
sorry

end bonnets_difference_thursday_monday_l257_257580


namespace second_term_arithmetic_seq_l257_257152

variable (a d : ℝ)

theorem second_term_arithmetic_seq (h : a + (a + 2 * d) = 8) : a + d = 4 := by
  sorry

end second_term_arithmetic_seq_l257_257152


namespace quadratic_transformation_l257_257652

theorem quadratic_transformation (a b c : ℝ) (h : a * (x - 1)^2 + b * (x - 1) + c = 2 * x^2 - 3 * x - 1) : 
  a = 2 ∧ b = 1 ∧ c = -2 := by
sorry

end quadratic_transformation_l257_257652


namespace arccos_one_over_sqrt_two_eq_pi_four_l257_257741

theorem arccos_one_over_sqrt_two_eq_pi_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_four_l257_257741


namespace amount_saved_per_person_l257_257805

-- Definitions based on the conditions
def original_price := 60
def discounted_price := 48
def number_of_people := 3
def discount := original_price - discounted_price

-- Proving that each person paid 4 dollars less.
theorem amount_saved_per_person : discount / number_of_people = 4 :=
by
  sorry

end amount_saved_per_person_l257_257805


namespace arccos_proof_l257_257681

noncomputable def arccos_identity : Prop := 
  ∃ x : ℝ, x = 1 / Real.sqrt 2 ∧ Real.arccos x = Real.pi / 4

theorem arccos_proof : arccos_identity :=
by
  sorry

end arccos_proof_l257_257681


namespace find_a_b_and_range_of_c_l257_257955

noncomputable def f (x a b c : ℝ) : ℝ := x^3 - a * x^2 + b * x + c

theorem find_a_b_and_range_of_c (c : ℝ) (h1 : ∀ x, 3 * x^2 - 2 * 3 * x - 9 = 0 → x = -1 ∨ x = 3)
    (h2 : ∀ x, x ∈ Set.Icc (-2 : ℝ) 6 → f x 3 (-9) c < c^2 + 4 * c) : 
    (a = 3 ∧ b = -9) ∧ (c > 6 ∨ c < -9) := by
  sorry

end find_a_b_and_range_of_c_l257_257955


namespace Yuna_place_l257_257757

theorem Yuna_place (Eunji_place : ℕ) (distance : ℕ) (Yuna_place : ℕ) 
  (h1 : Eunji_place = 100) 
  (h2 : distance = 11) 
  (h3 : Yuna_place = Eunji_place + distance) : 
  Yuna_place = 111 := 
sorry

end Yuna_place_l257_257757


namespace total_participants_l257_257835

theorem total_participants
  (F M : ℕ) 
  (half_female_democrats : F / 2 = 125)
  (one_third_democrats : (F + M) / 3 = (125 + M / 4))
  : F + M = 1750 :=
by
  sorry

end total_participants_l257_257835


namespace smallest_special_number_l257_257095

-- A natural number is "special" if it uses exactly four distinct digits
def is_special (n : ℕ) : Prop :=
  let digits := (n.digits 10).erase_dup in
  digits.length = 4

-- Define the smallest special number greater than 3429
def smallest_special_gt_3429 : ℕ :=
  3450

-- The theorem we want to prove
theorem smallest_special_number (h : ∀ n : ℕ, n > 3429 → is_special n → n ≥ smallest_special_gt_3429) :
  smallest_special_gt_3429 = 3450 :=
by
  sorry

end smallest_special_number_l257_257095


namespace right_triangle_median_l257_257174

variable (A B C M N : Type) [LinearOrder B] [LinearOrder C] [LinearOrder A] [LinearOrder M] [LinearOrder N]
variable (AC BC AM BN AB : ℝ)
variable (right_triangle : AC * AC + BC * BC = AB * AB)
variable (median_A : AC * AC + (1 / 4) * BC * BC = 81)
variable (median_B : BC * BC + (1 / 4) * AC * AC = 99)

theorem right_triangle_median :
  ∀ (AC BC AB : ℝ),
  (AC * AC + BC * BC = 144) → (AC * AC + BC * BC = AB * AB) → AB = 12 :=
by
  intros
  sorry

end right_triangle_median_l257_257174


namespace greatest_perimeter_of_triangle_l257_257562

theorem greatest_perimeter_of_triangle :
  ∃ (x : ℕ), (3 * x) + 15 = 57 ∧ 
  (x > 5 ∧ x < 15) ∧ 
  2 * x + x > 15 ∧ 
  x + 15 > 2 * x ∧ 
  2 * x + 15 > x := 
sorry

end greatest_perimeter_of_triangle_l257_257562


namespace two_pipes_fill_time_l257_257368

theorem two_pipes_fill_time (R : ℝ) (h1 : (3 : ℝ) * R * (8 : ℝ) = 1) : (2 : ℝ) * R * (12 : ℝ) = 1 :=
by 
  have hR : R = 1 / 24 := by linarith
  rw [hR]
  sorry

end two_pipes_fill_time_l257_257368


namespace base8_to_base10_conversion_l257_257062

def base8_to_base10 (n : Nat) : Nat := 
  match n with
  | 246 => 2 * 8^2 + 4 * 8^1 + 6 * 8^0
  | _ => 0  -- We define this only for the number 246_8

theorem base8_to_base10_conversion : base8_to_base10 246 = 166 := by 
  sorry

end base8_to_base10_conversion_l257_257062


namespace warriors_truth_tellers_l257_257892

/-- There are 33 warriors. Each warrior is either a truth-teller or a liar, 
    with only one favorite weapon: a sword, a spear, an axe, or a bow. 
    They were asked four questions, and the number of "Yes" answers to the 
    questions are 13, 15, 20, and 27 respectively. Prove that the number of 
    warriors who always tell the truth is 12. -/
theorem warriors_truth_tellers
  (warriors : ℕ) (truth_tellers : ℕ)
  (yes_to_sword : ℕ) (yes_to_spear : ℕ)
  (yes_to_axe : ℕ) (yes_to_bow : ℕ)
  (h1 : warriors = 33)
  (h2 : yes_to_sword = 13)
  (h3 : yes_to_spear = 15)
  (h4 : yes_to_axe = 20)
  (h5 : yes_to_bow = 27)
  (h6 : yes_to_sword + yes_to_spear + yes_to_axe + yes_to_bow = 75) :
  truth_tellers = 12 := by
  -- Proof will be here
  sorry

end warriors_truth_tellers_l257_257892


namespace goods_train_passes_man_in_10_seconds_l257_257221

def goods_train_pass_time (man_speed_kmph goods_speed_kmph goods_length_m : ℕ) : ℕ :=
  let relative_speed_mps := (man_speed_kmph + goods_speed_kmph) * 1000 / 3600
  goods_length_m / relative_speed_mps

theorem goods_train_passes_man_in_10_seconds :
  goods_train_pass_time 55 60 320 = 10 := sorry

end goods_train_passes_man_in_10_seconds_l257_257221


namespace arccos_identity_l257_257733

theorem arccos_identity : 
  ∀ x : ℝ, x = 1 / real.sqrt 2 → real.arccos x = real.pi / 4 :=
begin
  intros x hx,
  have h1 : real.cos (real.pi / 4) = 1 / real.sqrt 2 := by sorry,
  have h2 : real.arccos (1 / real.sqrt 2) = real.pi / 4,
  { rw ← h1,
    rw real.arccos_cos,
    exact_mod_cast hx },
  exact h2,
end

end arccos_identity_l257_257733


namespace shaded_area_calc_l257_257195

theorem shaded_area_calc (r1_area r2_area overlap_area circle_area : ℝ)
  (h_r1_area : r1_area = 36)
  (h_r2_area : r2_area = 28)
  (h_overlap_area : overlap_area = 21)
  (h_circle_area : circle_area = Real.pi) : 
  (r1_area + r2_area - overlap_area - circle_area) = 64 - Real.pi :=
by
  sorry

end shaded_area_calc_l257_257195


namespace sum_cubes_eq_neg_27_l257_257794

theorem sum_cubes_eq_neg_27 (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
 (h_eq : (a^3 + 9) / a = (b^3 + 9) / b ∧ (b^3 + 9) / b = (c^3 + 9) / c) :
 a^3 + b^3 + c^3 = -27 :=
by
  sorry

end sum_cubes_eq_neg_27_l257_257794


namespace provisions_last_days_l257_257431

def num_soldiers_initial : ℕ := 1200
def daily_consumption_initial : ℝ := 3
def initial_duration : ℝ := 30
def extra_soldiers : ℕ := 528
def daily_consumption_new : ℝ := 2.5

noncomputable def total_provisions : ℝ := num_soldiers_initial * daily_consumption_initial * initial_duration
noncomputable def total_soldiers_after_joining : ℕ := num_soldiers_initial + extra_soldiers
noncomputable def new_daily_consumption : ℝ := total_soldiers_after_joining * daily_consumption_new

theorem provisions_last_days : (total_provisions / new_daily_consumption) = 25 := by
  sorry

end provisions_last_days_l257_257431


namespace square_of_negative_is_positive_l257_257281

-- Define P as a negative integer
variable (P : ℤ) (hP : P < 0)

-- Theorem statement that P² is always positive.
theorem square_of_negative_is_positive : P^2 > 0 :=
sorry

end square_of_negative_is_positive_l257_257281


namespace concurrency_of_lines_l257_257999

open EuclideanGeometry

-- Define the main problem given the conditions and the statement to be proven
theorem concurrency_of_lines
  (ABC : Triangle)
  (I : Point)
  (A1 B1 C1 K L : Point)
  (incircle_tangent : is_tangent_incircle ABC I A1 B1 C1)
  (circumcircle_O1 : is_circumcircle (Triangle.mk B C1 B1) K)
  (circumcircle_O2 : is_circumcircle (Triangle.mk C B1 C1) L)
  (K_on_BC : on_line_segment BC K)
  (L_on_BC : on_line_segment BC L) :
  concurrent (Line.mk C1 L) (Line.mk B1 K) (Line.mk A1 I) :=
begin
  sorry
end

end concurrency_of_lines_l257_257999


namespace quadratic_polynomial_half_coefficient_l257_257433

theorem quadratic_polynomial_half_coefficient :
  ∃ b c : ℚ, ∀ x : ℤ, ∃ k : ℤ, (1/2 : ℚ) * (x^2 : ℚ) + b * (x : ℚ) + c = (k : ℚ) :=
by
  sorry

end quadratic_polynomial_half_coefficient_l257_257433


namespace distance_from_A_to_B_l257_257045

-- Definitions of the conditions
def avg_speed : ℝ := 25
def distance_AB (D : ℝ) : Prop := ∃ T : ℝ, D / (4 * T) = avg_speed ∧ D = 3 * (T * avg_speed)∧ (D / 2) = (T * avg_speed)

theorem distance_from_A_to_B : ∃ D : ℝ, distance_AB D ∧ D = 100 / 3 :=
by
  sorry

end distance_from_A_to_B_l257_257045


namespace truthful_warriors_count_l257_257909

-- Noncomputable theory is not necessary here.
-- We define the number of warriors and their responses.
def warriors_count : ℕ := 33
def yes_sword : ℕ := 13
def yes_spear : ℕ := 15
def yes_axe : ℕ := 20
def yes_bow : ℕ := 27

-- The equation from the solution steps where x is the number of truthful warriors.
def total_yes := yes_sword + yes_spear + yes_axe + yes_bow

theorem truthful_warriors_count :
  ∃ x : ℕ, x + 3 * (warriors_count - x) = total_yes ∧ x = 12 :=
by {
  -- We state that there exists an x such that its value satisfies the total "yes" responses equation.
  -- Here, we assert that x equals 12.
  use 12,
  split,
  {
    -- Prove the equation x + 3 * (warriors_count - x) = total_yes
    rw [warriors_count],
    rw [total_yes],
    norm_num,
  },
  {
    -- State that x = 12
    refl,
  },
}

end truthful_warriors_count_l257_257909


namespace difference_of_two_numbers_l257_257428

theorem difference_of_two_numbers 
(x y : ℝ) 
(h1 : x + y = 20) 
(h2 : x^2 - y^2 = 160) : 
  x - y = 8 := 
by 
  sorry

end difference_of_two_numbers_l257_257428


namespace determine_c_for_inverse_l257_257317

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := 1 / (3 * x + c)
noncomputable def f_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

theorem determine_c_for_inverse :
  (∀ x : ℝ, x ≠ 0 → f (f_inv x) c = x) ↔ c = 1 :=
sorry

end determine_c_for_inverse_l257_257317


namespace average_weight_of_a_and_b_l257_257825

-- Define the parameters in the conditions
variables (A B C : ℝ)

-- Conditions given in the problem
theorem average_weight_of_a_and_b (h1 : (A + B + C) / 3 = 45) 
                                 (h2 : (B + C) / 2 = 43) 
                                 (h3 : B = 33) : (A + B) / 2 = 41 := 
sorry

end average_weight_of_a_and_b_l257_257825


namespace joey_total_study_time_l257_257436

def hours_weekdays (hours_per_night : Nat) (nights_per_week : Nat) : Nat :=
  hours_per_night * nights_per_week

def hours_weekends (hours_per_day : Nat) (days_per_weekend : Nat) : Nat :=
  hours_per_day * days_per_weekend

def total_weekly_study_time (weekday_hours : Nat) (weekend_hours : Nat) : Nat :=
  weekday_hours + weekend_hours

def total_study_time_in_weeks (weekly_hours : Nat) (weeks : Nat) : Nat :=
  weekly_hours * weeks

theorem joey_total_study_time :
  let hours_per_night := 2
  let nights_per_week := 5
  let hours_per_day := 3
  let days_per_weekend := 2
  let weeks := 6
  hours_weekdays hours_per_night nights_per_week +
  hours_weekends hours_per_day days_per_weekend = 16 →
  total_study_time_in_weeks 16 weeks = 96 :=
by 
  intros h1 h2 h3 h4 h5
  have weekday_hours := hours_weekdays h1 h2
  have weekend_hours := hours_weekends h3 h4
  have total_weekly := total_weekly_study_time weekday_hours weekend_hours
  sorry

end joey_total_study_time_l257_257436


namespace three_digit_numbers_not_divisible_by_three_l257_257115

theorem three_digit_numbers_not_divisible_by_three :
  let digits := {1, 2, 3, 4, 5}
  let combinations := (Comb n 3).to_finset
  let valid_combinations := combinations.filter (λ comb, (sum comb) % 3 ≠ 0)
  let permutations_of_comb : Finset (Finset (List ℕ)) := valid_combinations.bind (λ comb, (Multichoose ℕ).permutations)
  permutations_of_comb.card = 18 := sorry

end three_digit_numbers_not_divisible_by_three_l257_257115


namespace empty_seats_in_theater_l257_257639

theorem empty_seats_in_theater :
  let total_seats := 750
  let occupied_seats := 532
  total_seats - occupied_seats = 218 :=
by
  sorry

end empty_seats_in_theater_l257_257639


namespace find_square_side_length_l257_257458

/-- Define the side lengths of the rectangle and the square --/
def rectangle_side_lengths (k : ℕ) (n : ℕ) : Prop := 
  k ≥ 7 ∧ n = 12 ∧ k * (k - 7) = n * n

theorem find_square_side_length (k n : ℕ) : rectangle_side_lengths k n → n = 12 :=
by
  intros
  sorry

end find_square_side_length_l257_257458


namespace purchase_price_l257_257587

noncomputable def cost_price_after_discount (P : ℝ) : ℝ :=
  0.8 * P + 375

theorem purchase_price {P : ℝ} (h : 1.15 * P = 18400) : cost_price_after_discount P = 13175 := by
  sorry

end purchase_price_l257_257587


namespace arccos_sqrt2_l257_257694

def arccos_eq (x : ℝ) := arccos x
def range_arccos := ∀ (x : ℝ), 0 ≤ arccos x ∧ arccos x ≤ π
def cos_pi_div_four : Real.cos (π / 4) = 1 / Real.sqrt 2 := by sorry

theorem arccos_sqrt2 :
  arccos_eq (1 / Real.sqrt 2) = π / 4 :=
by
  have h1 : Real.cos (π / 4) = 1 / Real.sqrt 2, from cos_pi_div_four
  sorry

end arccos_sqrt2_l257_257694


namespace graph_of_function_does_not_pass_through_first_quadrant_l257_257599

theorem graph_of_function_does_not_pass_through_first_quadrant (k : ℝ) (h : k < 0) : 
  ¬(∃ x y : ℝ, y = k * (x - k) ∧ x > 0 ∧ y > 0) :=
sorry

end graph_of_function_does_not_pass_through_first_quadrant_l257_257599


namespace minute_hand_coincides_hour_hand_11_times_l257_257843

noncomputable def number_of_coincidences : ℕ := 11

theorem minute_hand_coincides_hour_hand_11_times :
  ∀ (t : ℝ), (0 < t ∧ t < 12) → ∃(n : ℕ), (1 ≤ n ∧ n ≤ 11) ∧ t = (n * 1 + n * (5 / 11)) :=
sorry

end minute_hand_coincides_hour_hand_11_times_l257_257843


namespace proof_F_4_f_5_l257_257771

def f (a : ℤ) : ℤ := a - 2

def F (a b : ℤ) : ℤ := a * b + b^2

theorem proof_F_4_f_5 :
  F 4 (f 5) = 21 := by
  sorry

end proof_F_4_f_5_l257_257771


namespace power_division_l257_257667

theorem power_division : (19^11 / 19^6 = 247609) := sorry

end power_division_l257_257667


namespace remainder_division_l257_257247

def polynomial (x : ℝ) : ℝ := x^4 - 4 * x^2 + 7 * x - 8

theorem remainder_division : polynomial 3 = 58 :=
by
  sorry

end remainder_division_l257_257247


namespace moment_of_inertia_of_thin_spherical_shell_l257_257379

variable (R : ℝ) (M : ℝ) (μ : ℝ)

-- Definitions for the conditions
def spherical_shell_surface_density (μ : ℝ) : Prop := true
def spherical_shell_radius (R : ℝ) : Prop := true
def spherical_shell_mass (M : ℝ) (R : ℝ) (μ : ℝ) : Prop := M = 4 * π * R^2 * μ

-- The statement of the proof problem
theorem moment_of_inertia_of_thin_spherical_shell 
  (h1 : spherical_shell_surface_density μ)
  (h2 : spherical_shell_radius R)
  (h3 : spherical_shell_mass M R μ) :
  ∃ Θ, Θ = (2 / 3) * M * R^2 :=
sorry

end moment_of_inertia_of_thin_spherical_shell_l257_257379


namespace six_people_rolling_dice_prob_l257_257182

theorem six_people_rolling_dice_prob :
  (let 
    total_combinations := 6^6,
    acceptable_combinations := 6 * 5^4 * 4,
    prob := (acceptable_combinations : ℚ) / total_combinations
  in prob) = 625 / 1944 := 
by 
  -- Given the complexity of the problem solved above
  sorry

end six_people_rolling_dice_prob_l257_257182


namespace arccos_one_over_sqrt_two_l257_257750

theorem arccos_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l257_257750


namespace double_root_equation_correct_statements_l257_257424

theorem double_root_equation_correct_statements
  (a b c : ℝ) (r₁ r₂ : ℝ)
  (h1 : a ≠ 0)
  (h2 : r₁ = 2 * r₂)
  (h3 : r₁ ≠ r₂)
  (h4 : a * r₁ ^ 2 + b * r₁ + c = 0)
  (h5 : a * r₂ ^ 2 + b * r₂ + c = 0) :
  (∀ (m n : ℝ), (∀ (r : ℝ), r = 2 → (x - r) * (m * x + n) = 0 → 4 * m ^ 2 + 5 * m * n + n ^ 2 = 0)) ∧
  (∀ (p q : ℝ), p * q = 2 → ∃ x, p * x ^ 2 + 3 * x + q = 0 ∧
    (∃ x₁ x₂ : ℝ, x₁ = -1 / p ∧ x₂ = -q ∧ x₁ = 2 * x₂)) ∧
  (2 * b ^ 2 = 9 * a * c) :=
by
  sorry

end double_root_equation_correct_statements_l257_257424


namespace simplify_expression_l257_257817

theorem simplify_expression (y : ℝ) : 2 - (2 - (2 - (2 - (2 - y)))) = 4 - y :=
by
  sorry

end simplify_expression_l257_257817


namespace avg_b_c_is_45_l257_257824

-- Define the weights of a, b, and c
variables (a b c : ℝ)

-- Conditions given in the problem
def avg_a_b_c (a b c : ℝ) := (a + b + c) / 3 = 45
def avg_a_b (a b : ℝ) := (a + b) / 2 = 40
def weight_b (b : ℝ) := b = 35

-- Theorem statement
theorem avg_b_c_is_45 (a b c : ℝ) (h1 : avg_a_b_c a b c) (h2 : avg_a_b a b) (h3 : weight_b b) :
  (b + c) / 2 = 45 := by
  -- Proof omitted for brevity
  sorry

end avg_b_c_is_45_l257_257824


namespace hyperbola_foci_on_x_axis_l257_257475

theorem hyperbola_foci_on_x_axis (a : ℝ) 
  (h1 : 1 - a < 0)
  (h2 : a - 3 > 0)
  (h3 : ∀ c, c = 2 → 2 * c = 4) : 
  a = 4 := 
sorry

end hyperbola_foci_on_x_axis_l257_257475


namespace area_of_triangle_is_18_l257_257335

-- Define the vertices of the triangle
def point1 : ℝ × ℝ := (1, 4)
def point2 : ℝ × ℝ := (7, 4)
def point3 : ℝ × ℝ := (1, 10)

-- Define a function to calculate the area of a triangle given three vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * (B.1 - A.1) * (C.2 - A.2)

-- Statement of the problem
theorem area_of_triangle_is_18 :
  triangle_area point1 point2 point3 = 18 :=
by
  -- skipping the proof
  sorry

end area_of_triangle_is_18_l257_257335


namespace erica_earnings_l257_257917

def price_per_kg : ℝ := 20
def past_catch : ℝ := 80
def catch_today := 2 * past_catch
def total_catch := past_catch + catch_today
def total_earnings := total_catch * price_per_kg

theorem erica_earnings : total_earnings = 4800 := by
  sorry

end erica_earnings_l257_257917


namespace probability_both_selected_l257_257637

def probability_selection_ram : ℚ := 4 / 7
def probability_selection_ravi : ℚ := 1 / 5

theorem probability_both_selected : probability_selection_ram * probability_selection_ravi = 4 / 35 := 
by 
  -- Proof goes here
  sorry

end probability_both_selected_l257_257637


namespace find_y_values_l257_257577

theorem find_y_values (x : ℝ) (h1 : x^2 + 4 * ( (x + 1) / (x - 3) )^2 = 50)
  (y := ( (x - 3)^2 * (x + 4) ) / (2 * x - 4)) :
  y = -32 / 7 ∨ y = 2 :=
sorry

end find_y_values_l257_257577


namespace arccos_proof_l257_257679

noncomputable def arccos_identity : Prop := 
  ∃ x : ℝ, x = 1 / Real.sqrt 2 ∧ Real.arccos x = Real.pi / 4

theorem arccos_proof : arccos_identity :=
by
  sorry

end arccos_proof_l257_257679


namespace right_triangle_hypotenuse_consecutive_even_l257_257985

theorem right_triangle_hypotenuse_consecutive_even (x : ℕ) (h : x ≠ 0) :
  ∃ (a b c : ℕ), a^2 + b^2 = c^2 ∧ ((a, b, c) = (x - 2, x, x + 2) ∨ (a, b, c) = (x, x - 2, x + 2) ∨ (a, b, c) = (x + 2, x, x - 2)) ∧ c = 10 := 
by
  sorry

end right_triangle_hypotenuse_consecutive_even_l257_257985


namespace Bobby_ate_5_pancakes_l257_257056

theorem Bobby_ate_5_pancakes
  (total_pancakes : ℕ := 21)
  (dog_eaten : ℕ := 7)
  (leftover : ℕ := 9) :
  (total_pancakes - dog_eaten - leftover = 5) := by
  sorry

end Bobby_ate_5_pancakes_l257_257056


namespace simplify_and_evaluate_l257_257463

noncomputable def expr (x : ℝ) : ℝ :=
  ((x^2 + x - 2) / (x - 2) - x - 2) / ((x^2 + 4 * x + 4) / x)

theorem simplify_and_evaluate : expr 1 = -1 / 3 :=
by
  sorry

end simplify_and_evaluate_l257_257463


namespace number_of_truthful_warriors_l257_257879

theorem number_of_truthful_warriors (total_warriors : ℕ) 
  (sword_yes : ℕ) (spear_yes : ℕ) (axe_yes : ℕ) (bow_yes : ℕ) 
  (always_tells_truth : ℕ → Prop)
  (always_lies : ℕ → Prop)
  (hv1 : total_warriors = 33)
  (hv2 : sword_yes = 13)
  (hv3 : spear_yes = 15)
  (hv4 : axe_yes = 20)
  (hv5 : bow_yes = 27) :
  ∃ truthful_warriors, truthful_warriors = 12 := 
by {
  sorry
}

end number_of_truthful_warriors_l257_257879


namespace arccos_of_one_over_sqrt_two_l257_257724

theorem arccos_of_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
sorry

end arccos_of_one_over_sqrt_two_l257_257724


namespace gasoline_price_increase_l257_257429

theorem gasoline_price_increase :
  let P_initial := 29.90
  let P_final := 149.70
  (P_final - P_initial) / P_initial * 100 = 400 :=
by
  let P_initial := 29.90
  let P_final := 149.70
  sorry

end gasoline_price_increase_l257_257429


namespace value_of_y_l257_257272

theorem value_of_y (x y : ℝ) (h₁ : 1.5 * x = 0.75 * y) (h₂ : x = 20) : y = 40 :=
sorry

end value_of_y_l257_257272


namespace linear_function_above_x_axis_l257_257187

theorem linear_function_above_x_axis (a : ℝ) :
  (-1 < a ∧ a < 2 ∧ a ≠ 0) ↔
  (∀ x, -2 ≤ x ∧ x ≤ 1 → ax + a + 2 > 0) :=
sorry

end linear_function_above_x_axis_l257_257187


namespace arithmetic_sequence_second_term_l257_257148

theorem arithmetic_sequence_second_term (a d : ℝ) (h : a + (a + 2 * d) = 8) : a + d = 4 :=
sorry

end arithmetic_sequence_second_term_l257_257148


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l257_257678

theorem arccos_one_over_sqrt_two_eq_pi_over_four : 
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l257_257678


namespace equal_number_of_boys_and_girls_l257_257193

theorem equal_number_of_boys_and_girls
  (m d M D : ℕ)
  (h1 : (M / m) ≠ (D / d))
  (h2 : (M / m + D / d) / 2 = (M + D) / (m + d)) : m = d :=
sorry

end equal_number_of_boys_and_girls_l257_257193


namespace rationalize_denominator_l257_257811

theorem rationalize_denominator :
  let A := 9
  let B := 7
  let C := -18
  let D := 0
  let S := 2
  let F := 111
  (A + B + C + D + S + F = 111) ∧ 
  (
    (1 / (Real.sqrt 5 + Real.sqrt 6 + 2 * Real.sqrt 2)) * 
    ((Real.sqrt 5 + Real.sqrt 6) - 2 * Real.sqrt 2) * 
    (3 - 2 * Real.sqrt 30) / 
    (3^2 - (2 * Real.sqrt 30)^2) = 
    (9 * Real.sqrt 5 + 7 * Real.sqrt 6 - 18 * Real.sqrt 2) / 111
  ) := by
  sorry

end rationalize_denominator_l257_257811


namespace average_sales_is_104_l257_257318

-- Define the sales data for the months January to May
def january_sales : ℕ := 150
def february_sales : ℕ := 90
def march_sales : ℕ := 60
def april_sales : ℕ := 140
def may_sales : ℕ := 100
def may_discount : ℕ := 20

-- Define the adjusted sales for May after applying the discount
def adjusted_may_sales : ℕ := may_sales - (may_sales * may_discount / 100)

-- Define the total sales from January to May
def total_sales : ℕ := january_sales + february_sales + march_sales + april_sales + adjusted_may_sales

-- Define the number of months
def number_of_months : ℕ := 5

-- Define the average sales per month
def average_sales_per_month : ℕ := total_sales / number_of_months

-- Prove that the average sales per month is equal to 104
theorem average_sales_is_104 : average_sales_per_month = 104 := by
  -- Here, we'd write the proof, but we'll leave it as 'sorry' for now
  sorry

end average_sales_is_104_l257_257318


namespace find_x_squared_l257_257200

variable (a b x p q : ℝ)

theorem find_x_squared (h1 : a ≠ b) (h2 : b ≠ 0) (h3 : q ≠ p) (h4 : (a^2 + x^2) / (b^2 + x^2) = p / q) : 
  x^2 = (b^2 * p - a^2 * q) / (q - p) := 
by 
  sorry

end find_x_squared_l257_257200


namespace jade_more_transactions_l257_257177

theorem jade_more_transactions (mabel_transactions : ℕ) (anthony_percentage : ℕ) (cal_fraction_numerator : ℕ) 
  (cal_fraction_denominator : ℕ) (jade_transactions : ℕ) (h1 : mabel_transactions = 90) 
  (h2 : anthony_percentage = 10) (h3 : cal_fraction_numerator = 2) (h4 : cal_fraction_denominator = 3) 
  (h5 : jade_transactions = 83) :
  jade_transactions - (2 * (90 + (90 * 10 / 100)) / 3) = 17 := 
by
  sorry

end jade_more_transactions_l257_257177


namespace warriors_truth_tellers_l257_257891

/-- There are 33 warriors. Each warrior is either a truth-teller or a liar, 
    with only one favorite weapon: a sword, a spear, an axe, or a bow. 
    They were asked four questions, and the number of "Yes" answers to the 
    questions are 13, 15, 20, and 27 respectively. Prove that the number of 
    warriors who always tell the truth is 12. -/
theorem warriors_truth_tellers
  (warriors : ℕ) (truth_tellers : ℕ)
  (yes_to_sword : ℕ) (yes_to_spear : ℕ)
  (yes_to_axe : ℕ) (yes_to_bow : ℕ)
  (h1 : warriors = 33)
  (h2 : yes_to_sword = 13)
  (h3 : yes_to_spear = 15)
  (h4 : yes_to_axe = 20)
  (h5 : yes_to_bow = 27)
  (h6 : yes_to_sword + yes_to_spear + yes_to_axe + yes_to_bow = 75) :
  truth_tellers = 12 := by
  -- Proof will be here
  sorry

end warriors_truth_tellers_l257_257891


namespace arccos_proof_l257_257682

noncomputable def arccos_identity : Prop := 
  ∃ x : ℝ, x = 1 / Real.sqrt 2 ∧ Real.arccos x = Real.pi / 4

theorem arccos_proof : arccos_identity :=
by
  sorry

end arccos_proof_l257_257682


namespace y_intercept_of_line_l257_257753

theorem y_intercept_of_line (x y : ℝ) (h : 5 * x - 3 * y = 15) : (0, -5) = (0, (-5 : ℝ)) :=
by
  sorry

end y_intercept_of_line_l257_257753


namespace water_spilled_l257_257037

theorem water_spilled (x s : ℕ) (h1 : s = x + 7) : s = 8 := by
  -- The proof would go here
  sorry

end water_spilled_l257_257037


namespace find_f2_l257_257826

theorem find_f2 (f : ℝ → ℝ)
  (h : ∀ x : ℝ, f x + 3 * f (1 - x) = 2 * x ^ 2) :
  f 2 = -1 / 4 :=
by
  sorry

end find_f2_l257_257826


namespace arithmetic_sequence_second_term_l257_257149

theorem arithmetic_sequence_second_term (a d : ℝ) (h : a + (a + 2 * d) = 8) : a + d = 4 :=
sorry

end arithmetic_sequence_second_term_l257_257149


namespace johnson_and_carter_tie_in_september_l257_257597

def monthly_home_runs_johnson : List ℕ := [3, 14, 18, 13, 10, 16, 14, 5]
def monthly_home_runs_carter : List ℕ := [5, 9, 22, 11, 15, 17, 9, 9]

def cumulative_home_runs (runs : List ℕ) (up_to : ℕ) : ℕ :=
  (runs.take up_to).sum

theorem johnson_and_carter_tie_in_september :
  cumulative_home_runs monthly_home_runs_johnson 7 = cumulative_home_runs monthly_home_runs_carter 7 :=
by
  sorry

end johnson_and_carter_tie_in_september_l257_257597


namespace find_smallest_result_l257_257059

namespace small_result

def num_set : Set Int := { -10, -4, 0, 2, 7 }

def all_results : Set Int := 
  { z | ∃ x ∈ num_set, ∃ y ∈ num_set, z = x * y ∨ z = x + y }

def smallest_result := -70

theorem find_smallest_result : ∃ z ∈ all_results, z = smallest_result :=
by
  sorry

end small_result

end find_smallest_result_l257_257059


namespace unique_n_l257_257526

theorem unique_n (n : ℕ) (h_pos : 0 < n) :
  (∀ x y : ℕ, (xy + 1) % n = 0 → (x + y) % n = 0) ↔ n = 2 :=
by
  sorry

end unique_n_l257_257526


namespace nine_digit_number_l257_257376

-- Conditions as definitions
def highest_digit (n : ℕ) : Prop :=
  (n / 100000000) = 6

def million_place (n : ℕ) : Prop :=
  (n / 1000000) % 10 = 1

def hundred_place (n : ℕ) : Prop :=
  n % 1000 / 100 = 1

def rest_digits_zero (n : ℕ) : Prop :=
  (n % 1000000 / 1000) % 10 = 0 ∧ 
  (n % 1000000 / 10000) % 10 = 0 ∧ 
  (n % 1000000 / 100000) % 10 = 0 ∧ 
  (n % 100000000 / 10000000) % 10 = 0 ∧ 
  (n % 100000000 / 100000000) % 10 = 0 ∧ 
  (n % 1000000000 / 100000000) % 10 = 6

-- The nine-digit number
def given_number : ℕ := 6001000100

-- Prove number == 60,010,001,00 and approximate to 6 billion
theorem nine_digit_number :
  ∃ n : ℕ, highest_digit n ∧ million_place n ∧ hundred_place n ∧ rest_digits_zero n ∧ n = 6001000100 ∧ (n / 1000000000) = 6 :=
sorry

end nine_digit_number_l257_257376


namespace sum_cubes_eq_neg_27_l257_257793

theorem sum_cubes_eq_neg_27 (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
 (h_eq : (a^3 + 9) / a = (b^3 + 9) / b ∧ (b^3 + 9) / b = (c^3 + 9) / c) :
 a^3 + b^3 + c^3 = -27 :=
by
  sorry

end sum_cubes_eq_neg_27_l257_257793


namespace smallest_special_number_l257_257086

def is_special (n : ℕ) : Prop :=
  (∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
   n = a * 1000 + b * 100 + c * 10 + d)

theorem smallest_special_number (n : ℕ) (h1 : n > 3429) (h2 : is_special n) : n = 3450 :=
sorry

end smallest_special_number_l257_257086


namespace david_more_push_ups_than_zachary_l257_257630

def zachary_push_ups : ℕ := 53
def zachary_crunches : ℕ := 14
def zachary_total : ℕ := 67
def david_crunches : ℕ := zachary_crunches - 10
def david_push_ups : ℕ := zachary_total - david_crunches

theorem david_more_push_ups_than_zachary : david_push_ups - zachary_push_ups = 10 := by
  sorry  -- Proof is not required as per instructions

end david_more_push_ups_than_zachary_l257_257630


namespace correct_card_assignment_l257_257328

theorem correct_card_assignment :
  ∃ (cards : Fin 4 → Fin 4), 
    (¬ (cards 1 = 3 ∨ cards 2 = 3) ∧
     ¬ (cards 0 = 2 ∨ cards 2 = 2) ∧
     ¬ (cards 0 = 1) ∧
     ¬ (cards 0 = 3)) →
    (cards 0 = 4 ∧ cards 1 = 2 ∧ cards 2 = 1 ∧ cards 3 = 3) := 
by {
  sorry
}

end correct_card_assignment_l257_257328


namespace find_k_l257_257768

theorem find_k (k : ℝ) (h : (3 : ℝ)^2 - k * (3 : ℝ) - 6 = 0) : k = 1 :=
by
  sorry

end find_k_l257_257768


namespace question_1_question_2_l257_257432

variable (a b c : ℝ × ℝ)
variable (k : ℝ)

def vect_a : ℝ × ℝ := (3, 2)
def vect_b : ℝ × ℝ := (-1, 2)
def vect_c : ℝ × ℝ := (4, 1)

theorem question_1 :
  3 • vect_a + vect_b - 2 • vect_c = (0, 6) := 
by
  sorry

theorem question_2 (k : ℝ) : 
  let lhs := (3 + 4 * k) * 2
  let rhs := -5 * (2 + k)
  (lhs = rhs) → k = -16 / 13 := 
by
  sorry

end question_1_question_2_l257_257432


namespace ratio_of_scores_l257_257783

theorem ratio_of_scores (Lizzie Nathalie Aimee teammates : ℕ) (combinedLN : ℕ)
    (team_total : ℕ) (m : ℕ) :
    Lizzie = 4 →
    Nathalie = Lizzie + 3 →
    combinedLN = Lizzie + Nathalie →
    Aimee = m * combinedLN →
    teammates = 17 →
    team_total = Lizzie + Nathalie + Aimee + teammates →
    team_total = 50 →
    (Aimee / combinedLN) = 2 :=
by 
    sorry

end ratio_of_scores_l257_257783


namespace number_of_primes_l257_257039

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem number_of_primes (p : ℕ)
  (H_prime : is_prime p)
  (H_square : is_perfect_square (1 + p + p^2 + p^3 + p^4)) :
  p = 3 :=
sorry

end number_of_primes_l257_257039


namespace minimum_value_of_function_l257_257982

theorem minimum_value_of_function (x : ℝ) (h : x * Real.log 2 / Real.log 3 ≥ 1) : 
  ∃ t : ℝ, t = 2^x ∧ t ≥ 3 ∧ ∀ y : ℝ, y = t^2 - 2*t - 3 → y = (t-1)^2 - 4 := 
sorry

end minimum_value_of_function_l257_257982


namespace greatest_possible_value_y_l257_257184

theorem greatest_possible_value_y
  (x y : ℤ)
  (h : x * y + 3 * x + 2 * y = -6) : 
  y ≤ 3 :=
by sorry

end greatest_possible_value_y_l257_257184


namespace range_of_m_l257_257968

def set_A : Set ℝ := { x : ℝ | -2 ≤ x ∧ x ≤ 5 }
def set_B (m : ℝ) : Set ℝ := { x : ℝ | (2 * m - 1) ≤ x ∧ x ≤ (2 * m + 1) }

theorem range_of_m (m : ℝ) : (set_B m ⊆ set_A) ↔ (-1 / 2 ≤ m ∧ m ≤ 2) :=
by
  sorry

end range_of_m_l257_257968


namespace intersection_A_B_l257_257969

open Set

def universal_set : Set ℤ := {x | 1 ≤ x ∧ x ≤ 5}
def A : Set ℤ := {1, 2, 3}
def complement_B : Set ℤ := {1, 2}
def B : Set ℤ := universal_set \ complement_B

theorem intersection_A_B : A ∩ B = {3} :=
by
  sorry

end intersection_A_B_l257_257969


namespace area_ratio_l257_257561

theorem area_ratio (A B C D E : Type*)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
  (AB BC AC AD AE : ℝ) (ADE_ratio : ℝ) :
  AB = 25 ∧ BC = 39 ∧ AC = 42 ∧ AD = 19 ∧ AE = 14 →
  ADE_ratio = 19 / 56 :=
by sorry

end area_ratio_l257_257561


namespace satisfies_natural_solution_l257_257285

theorem satisfies_natural_solution (m : ℤ) :
  (∃ x : ℕ, x = 6 / (m - 1)) → (m = 2 ∨ m = 3 ∨ m = 4 ∨ m = 7) :=
by
  sorry

end satisfies_natural_solution_l257_257285


namespace least_number_l257_257935

theorem least_number (n : ℕ) (h1 : n % 38 = 1) (h2 : n % 3 = 1) : n = 115 :=
sorry

end least_number_l257_257935


namespace arccos_sqrt2_l257_257688

def arccos_eq (x : ℝ) := arccos x
def range_arccos := ∀ (x : ℝ), 0 ≤ arccos x ∧ arccos x ≤ π
def cos_pi_div_four : Real.cos (π / 4) = 1 / Real.sqrt 2 := by sorry

theorem arccos_sqrt2 :
  arccos_eq (1 / Real.sqrt 2) = π / 4 :=
by
  have h1 : Real.cos (π / 4) = 1 / Real.sqrt 2, from cos_pi_div_four
  sorry

end arccos_sqrt2_l257_257688


namespace integer_solutions_l257_257932

theorem integer_solutions (a b c : ℤ) (h₁ : 1 < a) 
    (h₂ : a < b) (h₃ : b < c) 
    (h₄ : (a-1) * (b-1) * (c-1) ∣ a * b * c - 1) :
    (a = 3 ∧ b = 5 ∧ c = 15) 
    ∨ (a = 2 ∧ b = 4 ∧ c = 8) :=
by sorry

end integer_solutions_l257_257932


namespace playground_dimensions_l257_257226

theorem playground_dimensions 
  (a b : ℕ) 
  (h1 : (a - 2) * (b - 2) = 4) : a * b = 2 * a + 2 * b :=
by
  sorry

end playground_dimensions_l257_257226


namespace probability_divisible_by_25_is_zero_l257_257529

-- Definitions of spinner outcomes and the function to generate four-digit numbers
def is_valid_spinner_outcome (n : ℕ) : Prop := n = 1 ∨ n = 2 ∨ n = 3

def generate_four_digit_number (spin1 spin2 spin3 spin4 : ℕ) : ℕ :=
  spin1 * 1000 + spin2 * 100 + spin3 * 10 + spin4

-- Condition stating that all outcomes of each spin are equally probable among {1, 2, 3}
def valid_outcome_condition (spin1 spin2 spin3 spin4 : ℕ) : Prop :=
  is_valid_spinner_outcome spin1 ∧ is_valid_spinner_outcome spin2 ∧
  is_valid_spinner_outcome spin3 ∧ is_valid_spinner_outcome spin4

-- Probability condition for the number being divisible by 25
def is_divisible_by_25 (n : ℕ) : Prop := n % 25 = 0

-- Main theorem: proving the probability is 0
theorem probability_divisible_by_25_is_zero :
  ∀ spin1 spin2 spin3 spin4,
    valid_outcome_condition spin1 spin2 spin3 spin4 →
    ¬ is_divisible_by_25 (generate_four_digit_number spin1 spin2 spin3 spin4) :=
by
  intros spin1 spin2 spin3 spin4 h
  -- Sorry for the proof details
  sorry

end probability_divisible_by_25_is_zero_l257_257529


namespace power_of_54_l257_257300

theorem power_of_54 (a b : ℕ) (h_a_pos : a > 0) (h_b_pos : b > 0) 
(h_eq : 54^a = a^b) : ∃ k : ℕ, a = 54^k := by
  sorry

end power_of_54_l257_257300


namespace largest_multiple_of_7_negation_gt_neg150_l257_257348

theorem largest_multiple_of_7_negation_gt_neg150 : 
  ∃ (k : ℤ), (k % 7 = 0 ∧ -k > -150 ∧ ∀ (m : ℤ), (m % 7 = 0 ∧ -m > -150 → m ≤ k)) :=
sorry

end largest_multiple_of_7_negation_gt_neg150_l257_257348


namespace power_sums_equal_l257_257388

theorem power_sums_equal (x y a b : ℝ)
  (h1 : x + y = a + b)
  (h2 : x^2 + y^2 = a^2 + b^2) :
  ∀ n : ℕ, x^n + y^n = a^n + b^n :=
by
  sorry

end power_sums_equal_l257_257388


namespace cost_price_per_meter_l257_257202

theorem cost_price_per_meter
  (total_meters : ℕ)
  (selling_price : ℕ)
  (loss_per_meter : ℕ)
  (total_cost_price : ℕ)
  (cost_price_per_meter : ℕ)
  (h1 : total_meters = 400)
  (h2 : selling_price = 18000)
  (h3 : loss_per_meter = 5)
  (h4 : total_cost_price = selling_price + total_meters * loss_per_meter)
  (h5 : cost_price_per_meter = total_cost_price / total_meters) :
  cost_price_per_meter = 50 :=
by
  sorry

end cost_price_per_meter_l257_257202


namespace arg_cubed_eq_pi_l257_257129

open Complex

theorem arg_cubed_eq_pi (z1 z2 : ℂ) (h1 : abs z1 = 3) (h2 : abs z2 = 5) (h3 : abs (z1 + z2) = 7) : 
  arg (z2 / z1) ^ 3 = π :=
by
  sorry

end arg_cubed_eq_pi_l257_257129


namespace sam_seashells_l257_257814

def seashells_problem := 
  let mary_seashells := 47
  let total_seashells := 65
  (total_seashells - mary_seashells) = 18

theorem sam_seashells :
  seashells_problem :=
by
  sorry

end sam_seashells_l257_257814


namespace sin_inequality_l257_257426

theorem sin_inequality (x y : ℝ) (hx : 0 ≤ x) (hx' : x ≤ π) (hy : 0 ≤ y) (hy' : y ≤ π) : 
  sin (x + y) ≤ sin x + sin y :=
sorry

end sin_inequality_l257_257426


namespace Faye_age_l257_257754

theorem Faye_age (D E C F : ℕ) (h1 : D = E - 5) (h2 : E = C + 3) (h3 : F = C + 2) (hD : D = 18) : F = 22 :=
by
  sorry

end Faye_age_l257_257754


namespace number_of_valid_numbers_l257_257971

def is_valid_number (N : ℕ) : Prop :=
  N ≥ 1000 ∧ N < 10000 ∧ ∃ a x : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ x < 1000 ∧ 
  N = 1000 * a + x ∧ x = N / 9

theorem number_of_valid_numbers : ∃ (n : ℕ), n = 7 ∧ ∀ N, is_valid_number N → N < 1000 * (n + 2) := 
sorry

end number_of_valid_numbers_l257_257971


namespace Vovochka_correct_pairs_count_l257_257780

def no_carry_pairs_count (digit_sum: ℕ → ℕ → Prop) : ℕ :=
  let count_pairs (lim: ℕ) : ℕ := (lim * (lim + 1)) / 2
  let digit_valid_pairs : ℕ := count_pairs 9
  (digit_valid_pairs * digit_valid_pairs) * 81

def digit_sum (x y: ℕ) : Prop := (x + y ≤ 9)

theorem Vovochka_correct_pairs_count :
  no_carry_pairs_count digit_sum = 244620 := by
  sorry

end Vovochka_correct_pairs_count_l257_257780


namespace value_of_expression_l257_257253

theorem value_of_expression (x1 x2 : ℝ) 
  (h1 : x1 ^ 2 - 3 * x1 - 4 = 0) 
  (h2 : x2 ^ 2 - 3 * x2 - 4 = 0)
  (h3 : x1 + x2 = 3) 
  (h4 : x1 * x2 = -4) : 
  x1 ^ 2 - 4 * x1 - x2 + 2 * x1 * x2 = -7 := by
  sorry

end value_of_expression_l257_257253


namespace auntie_em_parking_probability_l257_257648

theorem auntie_em_parking_probability :
  let total_spaces := 20
  let cars := 15
  let empty_spaces := total_spaces - cars
  let possible_configurations := Nat.choose total_spaces cars
  let unfavourable_configurations := Nat.choose (empty_spaces - 8 + 5) (empty_spaces - 8)
  let favourable_probability := 1 - ((unfavourable_configurations : ℚ) / (possible_configurations : ℚ))
  (favourable_probability = 1839 / 1938) :=
by
  -- sorry to skip the actual proof
  sorry

end auntie_em_parking_probability_l257_257648


namespace geom_seq_common_ratio_q_l257_257414

-- Define the geometric sequence
def geom_seq (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * q ^ (n - 1)

-- State the theorem
theorem geom_seq_common_ratio_q {a₁ q : ℝ} :
  (a₁ = 2) → (geom_seq a₁ q 4 = 16) → (q = 2) :=
by
  intros h₁ h₂
  sorry

end geom_seq_common_ratio_q_l257_257414


namespace arccos_one_over_sqrt_two_l257_257709

theorem arccos_one_over_sqrt_two : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l257_257709


namespace sum_of_y_coordinates_of_other_vertices_l257_257554

theorem sum_of_y_coordinates_of_other_vertices
  (A B : ℝ × ℝ)
  (C D : ℝ × ℝ)
  (hA : A = (2, 15))
  (hB : B = (8, -2))
  (h_mid : midpoint ℝ A B = midpoint ℝ C D) :
  C.snd + D.snd = 13 := 
sorry

end sum_of_y_coordinates_of_other_vertices_l257_257554


namespace unique_positive_integers_abc_l257_257394

def coprime (a b : ℕ) := Nat.gcd a b = 1

def allPrimeDivisorsNotCongruentTo1Mod7 (n : ℕ) := 
  ∀ p : ℕ, Nat.Prime p → p ∣ n → p % 7 ≠ 1

theorem unique_positive_integers_abc :
  ∀ a b c : ℕ,
    (1 ≤ a) →
    (1 ≤ b) →
    (1 ≤ c) →
    coprime a b →
    coprime b c →
    coprime c a →
    (a * a + b) ∣ (b * b + c) →
    (b * b + c) ∣ (c * c + a) →
    allPrimeDivisorsNotCongruentTo1Mod7 (a * a + b) →
  a = 1 ∧ b = 1 ∧ c = 1 := by
  sorry

end unique_positive_integers_abc_l257_257394


namespace parakeets_per_cage_l257_257224

theorem parakeets_per_cage 
  (num_cages : ℕ) 
  (parrots_per_cage : ℕ) 
  (total_birds : ℕ) 
  (hcages : num_cages = 6) 
  (hparrots : parrots_per_cage = 6) 
  (htotal : total_birds = 48) :
  (total_birds - num_cages * parrots_per_cage) / num_cages = 2 := 
  by
  sorry

end parakeets_per_cage_l257_257224


namespace solve_quadratic_l257_257314

theorem solve_quadratic : 
  ∃ x1 x2 : ℝ, (x1 = 2 + Real.sqrt 11) ∧ (x2 = 2 - (Real.sqrt 11)) ∧ 
  (∀ x : ℝ, x^2 - 4*x - 7 = 0 ↔ x = x1 ∨ x = x2) := 
sorry

end solve_quadratic_l257_257314


namespace arccos_sqrt_half_l257_257718

theorem arccos_sqrt_half : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := by
  sorry

end arccos_sqrt_half_l257_257718


namespace largest_perimeter_triangle_l257_257860

theorem largest_perimeter_triangle :
  ∃ (y : ℤ), 4 < y ∧ y < 20 ∧ 8 + 12 + y = 39 :=
by {
  -- we'll skip the proof steps
  sorry 
}

end largest_perimeter_triangle_l257_257860


namespace train_speed_kmh_l257_257048

theorem train_speed_kmh (T P: ℝ) (L: ℝ):
  (T = L + 320) ∧ (L = 18 * P) ->
  P = 20 -> 
  P * 3.6 = 72 := 
by
  sorry

end train_speed_kmh_l257_257048


namespace parabola_circle_intersection_l257_257112

theorem parabola_circle_intersection (a : ℝ) : 
  a ≤ Real.sqrt 2 + 1 / 4 → 
  ∃ (b x y : ℝ), y = x^2 + a ∧ x^2 + y^2 + 2 * b^2 = 2 * b * (x - y) + 1 :=
by
  sorry

end parabola_circle_intersection_l257_257112


namespace simplify_expression_l257_257311

theorem simplify_expression (p : ℝ) : 
  (2 * (3 * p + 4) - 5 * p * 2)^2 + (6 - 2 / 2) * (9 * p - 12) = 16 * p^2 - 19 * p + 4 := 
by 
  sorry

end simplify_expression_l257_257311


namespace frustum_shortest_distance_l257_257192

open Real

noncomputable def shortest_distance (R1 R2 : ℝ) (AB : ℝ) (string_from_midpoint : Bool) : ℝ :=
  if R1 = 5 ∧ R2 = 10 ∧ AB = 20 ∧ string_from_midpoint = true then 4 else 0

theorem frustum_shortest_distance : 
  shortest_distance 5 10 20 true = 4 :=
by sorry

end frustum_shortest_distance_l257_257192


namespace correct_M_l257_257621

-- Definition of the function M for calculating the position number
def M (k : ℕ) : ℕ :=
  if k % 2 = 1 then
    4 * k^2 - 4 * k + 2
  else
    4 * k^2 - 2 * k + 2

-- Theorem stating the correctness of the function M
theorem correct_M (k : ℕ) : M k = if k % 2 = 1 then 4 * k^2 - 4 * k + 2 else 4 * k^2 - 2 * k + 2 := 
by
  -- The proof is to be done later.
  -- sorry is used to indicate a placeholder.
  sorry

end correct_M_l257_257621


namespace intersection_eq_l257_257578

theorem intersection_eq {A : Set ℕ} {B : Set ℕ} 
  (hA : A = {0, 1, 2, 3, 4, 5, 6}) 
  (hB : B = {x | ∃ n ∈ A, x = 2 * n}) : 
  A ∩ B = {0, 2, 4, 6} := by
  sorry

end intersection_eq_l257_257578


namespace solve_for_y_l257_257276

noncomputable def x : ℝ := 20
noncomputable def y : ℝ := 40

theorem solve_for_y 
  (h₁ : 1.5 * x = 0.75 * y) 
  (h₂ : x = 20) : 
  y = 40 :=
by
  sorry

end solve_for_y_l257_257276


namespace difference_in_balances_l257_257659

/-- Define the parameters for Angela's and Bob's accounts --/
def P_A : ℕ := 5000  -- Angela's principal
def r_A : ℚ := 0.05  -- Angela's annual interest rate
def n_A : ℕ := 2  -- Compounding frequency for Angela
def t : ℕ := 15  -- Time in years

def P_B : ℕ := 7000  -- Bob's principal
def r_B : ℚ := 0.04  -- Bob's annual interest rate

/-- Computing the final amounts for Angela and Bob after 15 years --/
noncomputable def A_A : ℚ := P_A * ((1 + (r_A / n_A)) ^ (n_A * t))  -- Angela's final amount
noncomputable def A_B : ℚ := P_B * (1 + r_B * t)  -- Bob's final amount

/-- Proof statement: The difference in account balances to the nearest dollar --/
theorem difference_in_balances : abs (A_A - A_B) = 726 := by
  sorry

end difference_in_balances_l257_257659


namespace largest_multiple_of_7_negated_gt_neg_150_l257_257351

theorem largest_multiple_of_7_negated_gt_neg_150 :
  ∃ (n : ℕ), (negate (n * 7) > -150) ∧ (∀ m : ℕ, (negate (m * 7) > -150) → m ≤ n) ∧ (n * 7 = 147) :=
sorry

end largest_multiple_of_7_negated_gt_neg_150_l257_257351


namespace original_price_l257_257298

theorem original_price (sale_price : ℝ) (discount : ℝ) : 
  sale_price = 55 → discount = 0.45 → 
  ∃ (P : ℝ), 0.55 * P = sale_price ∧ P = 100 :=
by
  sorry

end original_price_l257_257298


namespace regular_polygon_sides_l257_257510

theorem regular_polygon_sides (perimeter side_length : ℝ) (h1 : perimeter = 180) (h2 : side_length = 15) :
  perimeter / side_length = 12 :=
by sorry

end regular_polygon_sides_l257_257510


namespace roots_of_polynomial_l257_257397

noncomputable def polynomial : Polynomial ℤ := Polynomial.X^3 - 4 * Polynomial.X^2 - Polynomial.X + 4

theorem roots_of_polynomial :
  (Polynomial.X - 1) * (Polynomial.X + 1) * (Polynomial.X - 4) = polynomial :=
by
  sorry

end roots_of_polynomial_l257_257397


namespace largest_multiple_of_7_negation_greater_than_neg_150_l257_257338

theorem largest_multiple_of_7_negation_greater_than_neg_150 : 
  ∃ k : ℤ, k * 7 = 147 ∧ ∀ n : ℤ, (k < n → n * 7 ≤ 150) :=
by
  use 21
  sorry

end largest_multiple_of_7_negation_greater_than_neg_150_l257_257338


namespace sin_alpha_cos_2beta_l257_257976

theorem sin_alpha_cos_2beta :
  ∀ α β : ℝ, 3 * Real.sin α - Real.sin β = Real.sqrt 10 ∧ α + β = Real.pi / 2 →
  Real.sin α = 3 * Real.sqrt 10 / 10 ∧ Real.cos (2 * β) = 4 / 5 :=
by
  intros α β h
  sorry

end sin_alpha_cos_2beta_l257_257976


namespace no_real_solution_l257_257183

-- Given conditions as definitions in Lean 4
def eq1 (x : ℝ) : Prop := x^5 + 3 * x^4 + 5 * x^3 + 5 * x^2 + 6 * x + 2 = 0
def eq2 (x : ℝ) : Prop := x^3 + 3 * x^2 + 4 * x + 1 = 0

-- The theorem to prove
theorem no_real_solution : ¬ ∃ x : ℝ, eq1 x ∧ eq2 x :=
by sorry

end no_real_solution_l257_257183


namespace different_dispatch_plans_l257_257377

theorem different_dispatch_plans :
  let teachers := {A, B, C, D, E, F, G, H} -- Constant set of teachers
  let dispatches :=  _ -- Function to calculate all valid dispatch plans according to conditions
  dispatches (teachers) = 600 :=
by
  sorry -- Proof omitted

end different_dispatch_plans_l257_257377


namespace fibonacci_polynomial_property_l257_257790

-- Define the Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n+2) => fibonacci n + fibonacci (n+1)

-- Define the polynomial P(x) of degree 990
noncomputable def P : ℕ → ℕ :=
  sorry  -- To be defined as a polynomial with specified properties

-- Statement of the problem (theorem)
theorem fibonacci_polynomial_property (P : ℕ → ℕ) (hP : ∀ k, 992 ≤ k → k ≤ 1982 → P k = fibonacci k) :
  P 1983 = fibonacci 1983 - 1 :=
sorry  -- Proof omitted

end fibonacci_polynomial_property_l257_257790


namespace smallest_special_greater_than_3429_l257_257088

def is_special (n : ℕ) : Prop := (nat.digits 10 n).nodup ∧ (nat.digits 10 n).length = 4

theorem smallest_special_greater_than_3429 : ∃ n, n > 3429 ∧ is_special n ∧ 
  ∀ m, m > 3429 ∧ is_special m → n ≤ m :=
by
  use 3450
  sorry

end smallest_special_greater_than_3429_l257_257088


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l257_257695

theorem arccos_one_over_sqrt_two_eq_pi_over_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l257_257695


namespace smallest_b_base_45b_perfect_square_l257_257363

theorem smallest_b_base_45b_perfect_square : ∃ b : ℕ, b > 3 ∧ (∃ n : ℕ, n^2 = 4 * b + 5) ∧ ∀ b' : ℕ, b' > 3 ∧ (∃ n' : ℕ, n'^2 = 4 * b' + 5) → b ≤ b' := 
sorry

end smallest_b_base_45b_perfect_square_l257_257363


namespace largest_n_for_ap_interior_angles_l257_257471

theorem largest_n_for_ap_interior_angles (n : ℕ) (d : ℤ) (a : ℤ) :
  (∀ i ∈ Finset.range n, a + i * d < 180) → 720 = d * (n - 1) * n → n ≤ 27 :=
by
  sorry

end largest_n_for_ap_interior_angles_l257_257471


namespace largest_five_digit_product_l257_257337

theorem largest_five_digit_product
  (digs : List ℕ)
  (h_digit_count : digs.length = 5)
  (h_product : (digs.foldr (· * ·) 1) = 9 * 8 * 7 * 6 * 5) :
  (digs.foldr (λ a b => if a > b then 10 * a + b else 10 * b + a) 0) = 98765 :=
sorry

end largest_five_digit_product_l257_257337


namespace complex_fraction_sum_l257_257392

theorem complex_fraction_sum :
  let a := (1 : ℂ)
  let b := (0 : ℂ)
  (a + b) = 1 :=
by
  sorry

end complex_fraction_sum_l257_257392


namespace students_attending_Harvard_l257_257993

theorem students_attending_Harvard (total_applicants : ℕ) (perc_accepted : ℝ) (perc_attending : ℝ)
    (h1 : total_applicants = 20000)
    (h2 : perc_accepted = 0.05)
    (h3 : perc_attending = 0.9) :
    total_applicants * perc_accepted * perc_attending = 900 := 
by
    sorry

end students_attending_Harvard_l257_257993


namespace probability_of_all_same_color_l257_257371

open Finset

-- Define the conditions of the problem
def num_red : ℕ := 3
def num_white : ℕ := 6
def num_blue : ℕ := 9
def total_marbles : ℕ := num_red + num_white + num_blue
def draw_count : ℕ := 4

-- Calculate the probabilities using combination
def P_all_red : ℚ := if draw_count ≤ num_red then 1 else 0
def P_all_white : ℚ := (Nat.choose num_white draw_count : ℚ) / (Nat.choose total_marbles draw_count : ℚ)
def P_all_blue : ℚ := (Nat.choose num_blue draw_count : ℚ) / (Nat.choose total_marbles draw_count : ℚ)

-- Define the total probability of drawing four marbles of the same color
def P_all_same_color : ℚ := P_all_red + P_all_white + P_all_blue

-- The goal is to prove that the total probability is equal to the correct answer
theorem probability_of_all_same_color :
  P_all_same_color = 9 / 170 := 
by 
  sorry

end probability_of_all_same_color_l257_257371


namespace largest_multiple_of_7_gt_neg_150_l257_257356

theorem largest_multiple_of_7_gt_neg_150 : ∃ (x : ℕ), (x % 7 = 0) ∧ ((- (x : ℤ)) > -150) ∧ ∀ y : ℕ, (y % 7 = 0 ∧ (- (y : ℤ)) > -150) → y ≤ x :=
by
  sorry

end largest_multiple_of_7_gt_neg_150_l257_257356


namespace waffle_bowl_more_scoops_l257_257524

-- Definitions based on conditions
def single_cone_scoops : ℕ := 1
def banana_split_scoops : ℕ := 3 * single_cone_scoops
def double_cone_scoops : ℕ := 2 * single_cone_scoops
def total_scoops : ℕ := 10
def remaining_scoops : ℕ := total_scoops - (banana_split_scoops + single_cone_scoops + double_cone_scoops)

-- Question: Prove that the waffle bowl has 1 more scoop than the banana split
theorem waffle_bowl_more_scoops : remaining_scoops - banana_split_scoops = 1 := by
  have h1 : single_cone_scoops = 1 := rfl
  have h2 : banana_split_scoops = 3 * single_cone_scoops := rfl
  have h3 : double_cone_scoops = 2 * single_cone_scoops := rfl
  have h4 : total_scoops = 10 := rfl
  have h5 : remaining_scoops = total_scoops - (banana_split_scoops + single_cone_scoops + double_cone_scoops) := rfl
  sorry

end waffle_bowl_more_scoops_l257_257524


namespace square_side_length_exists_l257_257456

theorem square_side_length_exists
    (k : ℕ)
    (n : ℕ)
    (h_side_length_condition : n * n = k * (k - 7))
    (h_grid_lines : k > 7) :
    n = 12 ∨ n = 24 :=
by sorry

end square_side_length_exists_l257_257456


namespace johns_actual_marks_l257_257570

def actual_marks (T : ℝ) (x : ℝ) (incorrect : ℝ) (students : ℕ) (avg_increase : ℝ) :=
  (incorrect = 82) ∧ (students = 80) ∧ (avg_increase = 1/2) ∧
  ((T + incorrect) / students = (T + x) / students + avg_increase)

theorem johns_actual_marks (T : ℝ) :
  ∃ x : ℝ, actual_marks T x 82 80 (1/2) ∧ x = 42 :=
by
  sorry

end johns_actual_marks_l257_257570


namespace smallest_special_number_l257_257096

-- A natural number is "special" if it uses exactly four distinct digits
def is_special (n : ℕ) : Prop :=
  let digits := (n.digits 10).erase_dup in
  digits.length = 4

-- Define the smallest special number greater than 3429
def smallest_special_gt_3429 : ℕ :=
  3450

-- The theorem we want to prove
theorem smallest_special_number (h : ∀ n : ℕ, n > 3429 → is_special n → n ≥ smallest_special_gt_3429) :
  smallest_special_gt_3429 = 3450 :=
by
  sorry

end smallest_special_number_l257_257096


namespace find_p_l257_257638

noncomputable def p (x1 x2 x3 x4 n : ℝ) :=
  (x1 + x3) * (x2 + x3) + (x1 + x4) * (x2 + x4)

theorem find_p (x1 x2 x3 x4 n : ℝ) (h1 : x1 ≠ x2)
(h2 : (x1 + x3) * (x1 + x4) = n - 10)
(h3 : (x2 + x3) * (x2 + x4) = n - 10) :
  p x1 x2 x3 x4 n = n - 20 :=
sorry

end find_p_l257_257638


namespace max_parts_by_rectangles_l257_257294

theorem max_parts_by_rectangles (n : ℕ) : 
  ∃ S : ℕ, S = 2 * n^2 - 2 * n + 2 :=
by
  sorry

end max_parts_by_rectangles_l257_257294


namespace no_real_roots_range_l257_257263

theorem no_real_roots_range (a : ℝ) : (¬ ∃ x : ℝ, x^2 + a * x - 4 * a = 0) ↔ (-16 < a ∧ a < 0) := by
  sorry

end no_real_roots_range_l257_257263


namespace speed_of_first_car_l257_257016

-- Define the conditions
def t : ℝ := 3.5
def v : ℝ := sorry -- (To be solved in the proof)
def speed_second_car : ℝ := 58
def total_distance : ℝ := 385

-- The distance each car travels after t hours
def distance_first_car : ℝ := v * t
def distance_second_car : ℝ := speed_second_car * t

-- The equation representing the total distance between the two cars after 3.5 hours
def equation := distance_first_car + distance_second_car = total_distance

-- The main theorem stating the speed of the first car
theorem speed_of_first_car : v = 52 :=
by
  -- The important proof steps would go here solving the equation "equation".
  sorry

end speed_of_first_car_l257_257016


namespace maximum_surface_area_of_cuboid_l257_257769

noncomputable def max_surface_area_of_inscribed_cuboid (R : ℝ) :=
  let (a, b, c) := (R, R, R) -- assuming cube dimensions where a=b=c
  2 * a * b + 2 * a * c + 2 * b * c

theorem maximum_surface_area_of_cuboid (R : ℝ) (h : ∃ a b c : ℝ, a^2 + b^2 + c^2 = 4 * R^2) :
  max_surface_area_of_inscribed_cuboid R = 8 * R^2 :=
sorry

end maximum_surface_area_of_cuboid_l257_257769


namespace part1_part2_l257_257260

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + (1 + a) * Real.exp (-x)

theorem part1 (a : ℝ) : (∀ x : ℝ, f x a = f (-x) a) ↔ a = 0 := by
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, 0 < x → f x a ≥ a + 1) → a ≤ 3 := by
  sorry

end part1_part2_l257_257260


namespace find_range_of_a_l257_257127

def p (a : ℝ) : Prop := 
  a = 0 ∨ (a > 0 ∧ a^2 - 4 * a < 0)

def q (a : ℝ) : Prop := 
  a^2 - 2 * a - 3 < 0

theorem find_range_of_a (a : ℝ) 
  (h1 : p a ∨ q a) 
  (h2 : ¬(p a ∧ q a)) : 
  (-1 < a ∧ a < 0) ∨ (3 ≤ a ∧ a < 4) := 
sorry

end find_range_of_a_l257_257127


namespace greatest_divisor_lemma_l257_257035

theorem greatest_divisor_lemma : ∃ (d : ℕ), d = Nat.gcd 1636 1852 ∧ d = 4 := by
  sorry

end greatest_divisor_lemma_l257_257035


namespace triangle_angles_l257_257591

theorem triangle_angles (α β : ℝ) (A B C : ℝ) (hA : A = 2) (hB : B = 3) (hC : C = 4) :
  2 * α + 3 * β = 180 :=
sorry

end triangle_angles_l257_257591


namespace value_of_x_squared_plus_reciprocal_l257_257422

theorem value_of_x_squared_plus_reciprocal (x : ℝ) (h : 47 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = 7 :=
by
  sorry

end value_of_x_squared_plus_reciprocal_l257_257422


namespace circle_tangent_problem_solution_l257_257387

noncomputable def circle_tangent_problem
(radius : ℝ)
(center : ℝ × ℝ)
(point_A : ℝ × ℝ)
(distance_OA : ℝ)
(segment_BC : ℝ) : ℝ :=
  let r := radius
  let O := center
  let A := point_A
  let OA := distance_OA
  let BC := segment_BC
  let AT := Real.sqrt (OA^2 - r^2)
  2 * AT - BC

-- Definitions for the conditions
def radius : ℝ := 8
def center : ℝ × ℝ := (0, 0)
def point_A : ℝ × ℝ := (17, 0)
def distance_OA : ℝ := 17
def segment_BC : ℝ := 12

-- Statement of the problem as an example theorem
theorem circle_tangent_problem_solution :
  circle_tangent_problem radius center point_A distance_OA segment_BC = 18 :=
by
  -- We would provide the proof here. The proof steps are not required as per the instructions.
  sorry

end circle_tangent_problem_solution_l257_257387


namespace fraction_defined_l257_257984

theorem fraction_defined (x : ℝ) : (1 - 2 * x ≠ 0) ↔ (x ≠ 1 / 2) :=
by sorry

end fraction_defined_l257_257984


namespace skill_position_players_wait_l257_257852

theorem skill_position_players_wait
  (num_linemen : ℕ) (drink_per_linemen : ℕ) 
  (num_skill_players : ℕ) (drink_per_skill_player : ℕ) 
  (total_water : ℕ) : ℕ :=
  (num_linemen = 12) →
  (drink_per_linemen = 8) →
  (num_skill_players = 10) →
  (drink_per_skill_player = 6) →
  (total_water = 126) →
  num_skill_players - (total_water - num_linemen * drink_per_linemen) / drink_per_skill_player = 5 := sorry

end skill_position_players_wait_l257_257852


namespace arnolds_total_protein_l257_257660

theorem arnolds_total_protein (collagen_protein_per_two_scoops : ℕ) (protein_per_scoop : ℕ) 
    (steak_protein : ℕ) (scoops_of_collagen : ℕ) (scoops_of_protein : ℕ) :
    collagen_protein_per_two_scoops = 18 →
    protein_per_scoop = 21 →
    steak_protein = 56 →
    scoops_of_collagen = 1 →
    scoops_of_protein = 1 →
    (collagen_protein_per_two_scoops / 2 * scoops_of_collagen + protein_per_scoop * scoops_of_protein + steak_protein = 86) :=
by
  intros hc p s sc sp
  sorry

end arnolds_total_protein_l257_257660


namespace find_exponent_M_l257_257027

theorem find_exponent_M (M : ℕ) : (32^4) * (4^6) = 2^M → M = 32 := by
  sorry

end find_exponent_M_l257_257027


namespace warriors_truth_tellers_l257_257895

/-- There are 33 warriors. Each warrior is either a truth-teller or a liar, 
    with only one favorite weapon: a sword, a spear, an axe, or a bow. 
    They were asked four questions, and the number of "Yes" answers to the 
    questions are 13, 15, 20, and 27 respectively. Prove that the number of 
    warriors who always tell the truth is 12. -/
theorem warriors_truth_tellers
  (warriors : ℕ) (truth_tellers : ℕ)
  (yes_to_sword : ℕ) (yes_to_spear : ℕ)
  (yes_to_axe : ℕ) (yes_to_bow : ℕ)
  (h1 : warriors = 33)
  (h2 : yes_to_sword = 13)
  (h3 : yes_to_spear = 15)
  (h4 : yes_to_axe = 20)
  (h5 : yes_to_bow = 27)
  (h6 : yes_to_sword + yes_to_spear + yes_to_axe + yes_to_bow = 75) :
  truth_tellers = 12 := by
  -- Proof will be here
  sorry

end warriors_truth_tellers_l257_257895


namespace domain_f_l257_257320

noncomputable def f (x : ℝ) := Real.sqrt (3 - x) + Real.log (x - 1)

theorem domain_f : { x : ℝ | 1 < x ∧ x ≤ 3 } = { x : ℝ | True } ∩ { x : ℝ | x ≤ 3 } ∩ { x : ℝ | x > 1 } :=
by
  sorry

end domain_f_l257_257320


namespace mode_of_dataset_with_average_is_l257_257126

theorem mode_of_dataset_with_average_is 
  (x : ℤ) 
  (h_avg : (1 + 0 + (-3) + 5 + x + 2 + (-3)) / 7 = 1) : 
  multiset.mode ({1, 0, -3, 5, x, 2, -3} : multiset ℤ) = { -3, 5 } := 
by 
  sorry

end mode_of_dataset_with_average_is_l257_257126


namespace number_of_sequences_less_than_1969_l257_257402

theorem number_of_sequences_less_than_1969 :
  (∃ S : ℕ → ℕ, (∀ n : ℕ, S (n + 1) > (S n) * (S n)) ∧ S 1969 = 1969) →
  ∃ N : ℕ, N < 1969 :=
sorry

end number_of_sequences_less_than_1969_l257_257402


namespace Winnie_keeps_lollipops_l257_257491

-- Definitions based on the conditions provided
def total_lollipops : ℕ := 60 + 135 + 5 + 250
def number_of_friends : ℕ := 12

-- The theorem statement we need to prove
theorem Winnie_keeps_lollipops : total_lollipops % number_of_friends = 6 :=
by
  -- proof omitted as instructed
  sorry

end Winnie_keeps_lollipops_l257_257491


namespace resistance_per_band_is_10_l257_257171

noncomputable def resistance_per_band := 10
def total_squat_weight := 30
def dumbbell_weight := 10
def number_of_bands := 2

theorem resistance_per_band_is_10 :
  (total_squat_weight - dumbbell_weight) / number_of_bands = resistance_per_band := 
by
  sorry

end resistance_per_band_is_10_l257_257171


namespace union_M_N_eq_M_l257_257548

-- Define set M
def M : Set ℝ := { y | ∃ x : ℝ, y = 2^x }

-- Define set N
def N : Set ℝ := { y | ∃ x : ℝ, y = Real.log (x - 1) }

-- Statement to prove that M ∪ N = M
theorem union_M_N_eq_M : M ∪ N = M := by
  sorry

end union_M_N_eq_M_l257_257548


namespace max_d_77733e_divisible_by_33_l257_257923

open Int

theorem max_d_77733e_divisible_by_33 : ∃ d e : ℕ, 
  (7 * 100000 + d * 10000 + 7 * 1000 + 3 * 100 + 3 * 10 + e) % 33 = 0 ∧ 
  (d ≤ 9) ∧ (e ≤ 9) ∧ 
  (∀ d' e', ((7 * 100000 + d' * 10000 + 7 * 1000 + 3 * 100 + 3 * 10 + e') % 33 = 0 ∧ d' ≤ 9 ∧ e' ≤ 9 → d' ≤ d)) 
  := ⟨6, 0, by sorry⟩

end max_d_77733e_divisible_by_33_l257_257923


namespace bicycle_distance_l257_257650

def distance : ℝ := 15

theorem bicycle_distance :
  ∀ (x y : ℝ),
  (x + 6) * (y - 5 / 60) = x * y →
  (x - 5) * (y + 6 / 60) = x * y →
  x * y = distance :=
by
  intros x y h1 h2
  sorry

end bicycle_distance_l257_257650


namespace transform_quadratic_l257_257331

theorem transform_quadratic (x m n : ℝ) 
  (h : x^2 - 6 * x - 1 = 0) : 
  (x + m)^2 = n ↔ (m = 3 ∧ n = 10) :=
by sorry

end transform_quadratic_l257_257331


namespace arccos_sqrt_half_l257_257717

theorem arccos_sqrt_half : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := by
  sorry

end arccos_sqrt_half_l257_257717


namespace members_in_both_sets_are_23_l257_257012

variable (U A B : Finset ℕ)
variable (count_U count_A count_B count_neither count_both : ℕ)

theorem members_in_both_sets_are_23 (hU : count_U = 192)
    (hA : count_A = 107) (hB : count_B = 49) (hNeither : count_neither = 59) :
    count_both = 23 :=
by
  sorry

end members_in_both_sets_are_23_l257_257012


namespace solve_for_x_l257_257821

theorem solve_for_x (x : ℝ) : x^2 + 6 * x + 8 = -(x + 4) * (x + 6) ↔ x = -4 :=
by {
  sorry
}

end solve_for_x_l257_257821


namespace chess_tournament_participants_l257_257983

theorem chess_tournament_participants (n : ℕ) (h : n * (n - 1) / 2 = 105) : n = 15 := by
  sorry

end chess_tournament_participants_l257_257983


namespace yola_past_weight_l257_257620

-- Definitions based on the conditions
def current_weight_yola : ℕ := 220
def weight_difference_current (D : ℕ) : ℕ := 30
def weight_difference_past (D : ℕ) : ℕ := D

-- Main statement
theorem yola_past_weight (D : ℕ) :
  (250 - D) = (current_weight_yola + weight_difference_current D - weight_difference_past D) :=
by
  sorry

end yola_past_weight_l257_257620


namespace ratio_of_distances_l257_257257

noncomputable def focus_parabola := (1, 0 : ℝ)  -- Focus F

def parabola := {p : ℝ × ℝ | p.snd ^ 2 = 4 * p.fst}

def line_through_focus (m : ℝ) := {p : ℝ × ℝ | p.snd = m * (p.fst - 1)}

def intersect_parabola_line (m : ℝ) := 
  {p : ℝ × ℝ | p ∈ parabola ∧ p ∈ line_through_focus m}

theorem ratio_of_distances (m : ℝ) (h_m_sqrt3 : m = real.sqrt 3)
  (A B : ℝ × ℝ) 
  (hA : A ∈ intersect_parabola_line m)
  (hB : B ∈ intersect_parabola_line m)
  (h_FA_gt_FB : dist focus_parabola A > dist focus_parabola B) : 
  dist focus_parabola A / dist focus_parabola B = 3 := 
sorry

end ratio_of_distances_l257_257257


namespace arccos_sqrt_half_l257_257712

theorem arccos_sqrt_half : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := by
  sorry

end arccos_sqrt_half_l257_257712


namespace sum_of_cubes_eq_neg_27_l257_257800

theorem sum_of_cubes_eq_neg_27
  (a b c : ℝ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_condition : (a^3 + 9) / a = (b^3 + 9) / b ∧ (b^3 + 9) / b = (c^3 + 9) / c) :
  a^3 + b^3 + c^3 = -27 :=
sorry

end sum_of_cubes_eq_neg_27_l257_257800


namespace roots_of_quadratic_eq_l257_257191

theorem roots_of_quadratic_eq (x : ℝ) : (x + 1) ^ 2 = 0 → x = -1 := by
  sorry

end roots_of_quadratic_eq_l257_257191


namespace smallest_special_number_gt_3429_l257_257098

-- Define what it means for a number to be special
def is_special (n : ℕ) : Prop :=
  (List.toFinset (Nat.digits 10 n)).card = 4

-- Define the problem statement in Lean
theorem smallest_special_number_gt_3429 : ∃ n : ℕ, 3429 < n ∧ is_special n ∧ ∀ m : ℕ, 3429 < m ∧ is_special m → n ≤ m := 
  by
  let smallest_n := 3450
  have hn : 3429 < smallest_n := by decide
  have hs : is_special smallest_n := by
    -- digits of 3450 are [3, 4, 5, 0], which are four different digits
    sorry 
  have minimal : ∀ m, 3429 < m ∧ is_special m → smallest_n ≤ m :=
    by
    -- This needs to show that no special number exists between 3429 and 3450
    sorry
  exact ⟨smallest_n, hn, hs, minimal⟩

end smallest_special_number_gt_3429_l257_257098


namespace arnold_total_protein_l257_257663

-- Definitions and conditions
def collagen_protein_per_scoop : ℕ := 18 / 2
def protein_powder_protein_per_scoop : ℕ := 21
def steak_protein : ℕ := 56

def collagen_scoops : ℕ := 1
def protein_powder_scoops : ℕ := 1
def steaks : ℕ := 1

-- Statement of the theorem/problem
theorem arnold_total_protein : 
  (collagen_protein_per_scoop * collagen_scoops) + 
  (protein_powder_protein_per_scoop * protein_powder_scoops) + 
  (steak_protein * steaks) = 86 :=
by
  sorry

end arnold_total_protein_l257_257663


namespace arccos_sqrt2_l257_257689

def arccos_eq (x : ℝ) := arccos x
def range_arccos := ∀ (x : ℝ), 0 ≤ arccos x ∧ arccos x ≤ π
def cos_pi_div_four : Real.cos (π / 4) = 1 / Real.sqrt 2 := by sorry

theorem arccos_sqrt2 :
  arccos_eq (1 / Real.sqrt 2) = π / 4 :=
by
  have h1 : Real.cos (π / 4) = 1 / Real.sqrt 2, from cos_pi_div_four
  sorry

end arccos_sqrt2_l257_257689


namespace largest_multiple_of_7_negated_gt_neg_150_l257_257350

theorem largest_multiple_of_7_negated_gt_neg_150 :
  ∃ (n : ℕ), (negate (n * 7) > -150) ∧ (∀ m : ℕ, (negate (m * 7) > -150) → m ≤ n) ∧ (n * 7 = 147) :=
sorry

end largest_multiple_of_7_negated_gt_neg_150_l257_257350


namespace triangle_is_isosceles_l257_257776

theorem triangle_is_isosceles
  (A B C : ℝ)
  (h_sum_angles : A + B + C = π)
  (h_condition : (Real.sin A + Real.sin B) * (Real.cos A + Real.cos B) = 2 * Real.sin C) :
  A = B :=
sorry

end triangle_is_isosceles_l257_257776


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l257_257696

theorem arccos_one_over_sqrt_two_eq_pi_over_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l257_257696


namespace Patel_family_theme_park_expenses_l257_257114

def regular_ticket_price : ℝ := 12.5
def senior_discount : ℝ := 0.8
def child_discount : ℝ := 0.6
def senior_ticket_price := senior_discount * regular_ticket_price
def child_ticket_price := child_discount * regular_ticket_price

theorem Patel_family_theme_park_expenses :
  (2 * senior_ticket_price + 2 * child_ticket_price + 4 * regular_ticket_price) = 85 := by
  sorry

end Patel_family_theme_park_expenses_l257_257114


namespace purely_imaginary_subtraction_l257_257131

-- Definition of the complex number z.
def z : ℂ := Complex.mk 2 (-1)

-- Statement to prove
theorem purely_imaginary_subtraction (h: z = Complex.mk 2 (-1)) : ∃ (b : ℝ), z - 2 = Complex.im b :=
by {
    sorry
}

end purely_imaginary_subtraction_l257_257131


namespace third_divisor_l257_257249

theorem third_divisor (x : ℕ) (h12 : 12 ∣ (x + 3)) (h15 : 15 ∣ (x + 3)) (h40 : 40 ∣ (x + 3)) :
  ∃ d : ℕ, d ≠ 12 ∧ d ≠ 15 ∧ d ≠ 40 ∧ d ∣ (x + 3) ∧ d = 2 :=
by
  sorry

end third_divisor_l257_257249


namespace mandatory_state_tax_rate_l257_257299

theorem mandatory_state_tax_rate 
  (MSRP : ℝ) (total_paid : ℝ) (insurance_rate : ℝ) (tax_rate : ℝ) 
  (insurance_cost : ℝ := insurance_rate * MSRP)
  (cost_before_tax : ℝ := MSRP + insurance_cost)
  (tax_amount : ℝ := total_paid - cost_before_tax) :
  MSRP = 30 → total_paid = 54 → insurance_rate = 0.2 → 
  tax_amount / cost_before_tax * 100 = tax_rate →
  tax_rate = 50 :=
by
  intros MSRP_val paid_val ins_rate_val comp_tax_rate
  sorry

end mandatory_state_tax_rate_l257_257299


namespace range_of_f_l257_257761

def f (x : ℝ) : ℝ := (3 * x + 8) / (x - 4)

theorem range_of_f :
  set.range f = set.Iio 3 ∪ set.Ioi 3 :=
sorry

end range_of_f_l257_257761


namespace neg_abs_neg_three_l257_257604

theorem neg_abs_neg_three : -|(-3)| = -3 := 
by
  sorry

end neg_abs_neg_three_l257_257604


namespace square_area_from_inscribed_circle_l257_257507

theorem square_area_from_inscribed_circle (r : ℝ) (π_pos : 0 < Real.pi) (circle_area : Real.pi * r^2 = 9 * Real.pi) : 
  (2 * r)^2 = 36 :=
by
  -- Proof goes here
  sorry

end square_area_from_inscribed_circle_l257_257507


namespace vovochka_correct_sum_combinations_l257_257781

theorem vovochka_correct_sum_combinations : 
  let digit_pairs := finset.filter (λ p : ℕ × ℕ, (p.fst + p.snd) < 10) ((finset.range 10).product (finset.range 10))
  let no_carry_combinations := finset.card digit_pairs
  no_carry_combinations ^ 3 * 81 = 244620 := 
by
  sorry

end vovochka_correct_sum_combinations_l257_257781


namespace geometric_solution_l257_257336

theorem geometric_solution (x y : ℝ) (h : x^2 + 2 * y^2 - 10 * x + 12 * y + 43 = 0) : x = 5 ∧ y = -3 := 
  by sorry

end geometric_solution_l257_257336


namespace length_width_difference_l257_257605

theorem length_width_difference
  (w l : ℝ)
  (h1 : l = 4 * w)
  (h2 : l * w = 768) :
  l - w = 24 * Real.sqrt 3 :=
by
  sorry

end length_width_difference_l257_257605


namespace polar_eq_parabola_l257_257878

/-- Prove that the curve defined by the polar equation is a parabola. -/
theorem polar_eq_parabola :
  ∀ (r θ : ℝ), r = 1 / (2 * Real.sin θ + Real.cos θ) →
    ∃ (x y : ℝ), (x = r * Real.cos θ) ∧ (y = r * Real.sin θ) ∧ (x + 2 * y = r^2) :=
by 
  sorry

end polar_eq_parabola_l257_257878


namespace ratio_of_pants_to_shirts_l257_257567

noncomputable def cost_shirt : ℝ := 6
noncomputable def cost_pants : ℝ := 8
noncomputable def num_shirts : ℝ := 10
noncomputable def total_cost : ℝ := 100

noncomputable def num_pants : ℝ :=
  (total_cost - (num_shirts * cost_shirt)) / cost_pants

theorem ratio_of_pants_to_shirts : num_pants / num_shirts = 1 / 2 := by
  sorry

end ratio_of_pants_to_shirts_l257_257567


namespace suraj_average_after_9th_innings_l257_257467

theorem suraj_average_after_9th_innings (A : ℕ) 
  (h1 : 8 * A + 90 = 9 * (A + 6)) : 
  (A + 6) = 42 :=
by
  sorry

end suraj_average_after_9th_innings_l257_257467


namespace athlete_speed_200m_in_24s_is_30kmh_l257_257052

noncomputable def speed_in_kmh (distance_meters : ℝ) (time_seconds : ℝ) : ℝ :=
  (distance_meters / 1000) / (time_seconds / 3600)

theorem athlete_speed_200m_in_24s_is_30kmh :
  speed_in_kmh 200 24 = 30 := by
  sorry

end athlete_speed_200m_in_24s_is_30kmh_l257_257052


namespace arccos_identity_l257_257730

theorem arccos_identity : 
  ∀ x : ℝ, x = 1 / real.sqrt 2 → real.arccos x = real.pi / 4 :=
begin
  intros x hx,
  have h1 : real.cos (real.pi / 4) = 1 / real.sqrt 2 := by sorry,
  have h2 : real.arccos (1 / real.sqrt 2) = real.pi / 4,
  { rw ← h1,
    rw real.arccos_cos,
    exact_mod_cast hx },
  exact h2,
end

end arccos_identity_l257_257730


namespace binom_prod_l257_257525

theorem binom_prod : (Nat.choose 10 3) * (Nat.choose 8 3) * 2 = 13440 := by
  sorry

end binom_prod_l257_257525


namespace pats_and_mats_numbers_l257_257807

theorem pats_and_mats_numbers (x y : ℕ) (hxy : x ≠ y) (hx_gt_hy : x > y) 
    (h_sum : (x + y) + (x - y) + x * y + (x / y) = 98) : x = 12 ∧ y = 6 :=
by
  sorry

end pats_and_mats_numbers_l257_257807


namespace remainder_polynomial_division_l257_257248

theorem remainder_polynomial_division :
  let f : ℝ → ℝ := λ x, x^4 - 4 * x^2 + 7 * x - 8 in
  f 3 = 58 :=
by
  intro f
  sorry

end remainder_polynomial_division_l257_257248


namespace probability_not_all_same_l257_257623

-- Define the conditions
def fair_six_sided_die := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

-- Define the event where all five dice show the same number
def all_dice_the_same (d1 d2 d3 d4 d5 : fair_six_sided_die) : Prop :=
  d1 = d2 ∧ d2 = d3 ∧ d3 = d4 ∧ d4 = d5

-- Define the total number of outcomes when rolling five dice
def total_outcomes : ℕ := 6^5

-- Define the number of outcomes where all dice show the same number
def same_number_outcomes : ℕ := 6

-- Define the probability that all dice show the same number
def prob_same_number : ℚ := same_number_outcomes / total_outcomes

-- Define the probability that not all dice show the same number
def prob_not_same_number : ℚ := 1 - prob_same_number

-- State the theorem
theorem probability_not_all_same : prob_not_same_number = 1295 / 1296 :=
by
  -- The proof will follow from the definitions, and using sorry to skip the internals.
  sorry

end probability_not_all_same_l257_257623


namespace unique_trivial_solution_of_linear_system_l257_257447

variable {R : Type*} [Field R]

theorem unique_trivial_solution_of_linear_system (a b c x y z : R)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_system : x + a * y + a^2 * z = 0 ∧ x + b * y + b^2 * z = 0 ∧ x + c * y + c^2 * z = 0) :
  x = 0 ∧ y = 0 ∧ z = 0 := sorry

end unique_trivial_solution_of_linear_system_l257_257447


namespace incorrect_conclusion_l257_257961

-- Define the given parabola.
def parabola (x : ℝ) : ℝ := (x - 2)^2 + 1

-- Define the conditions for the parabola.
def parabola_opens_upwards : Prop := ∀ x y : ℝ, parabola (x + y) = (x + y - 2)^2 + 1
def axis_of_symmetry : Prop := ∀ x : ℝ, parabola x = parabola (4 - x)
def vertex_coordinates : Prop := parabola 2 = 1 ∧ (parabola 2, 2) = (1, 2)
def behavior_when_x_less_than_2 : Prop := ∀ x : ℝ, x < 2 → parabola x < parabola (x + 1)

-- The statement that needs to be proven in Lean 4.
theorem incorrect_conclusion : ¬ behavior_when_x_less_than_2 :=
  by
  sorry

end incorrect_conclusion_l257_257961


namespace base8_to_base10_l257_257078

theorem base8_to_base10 (n : ℕ) : of_digits 8 [2, 4, 6] = 166 := by
  sorry

end base8_to_base10_l257_257078


namespace smallest_special_number_l257_257094

-- A natural number is "special" if it uses exactly four distinct digits
def is_special (n : ℕ) : Prop :=
  let digits := (n.digits 10).erase_dup in
  digits.length = 4

-- Define the smallest special number greater than 3429
def smallest_special_gt_3429 : ℕ :=
  3450

-- The theorem we want to prove
theorem smallest_special_number (h : ∀ n : ℕ, n > 3429 → is_special n → n ≥ smallest_special_gt_3429) :
  smallest_special_gt_3429 = 3450 :=
by
  sorry

end smallest_special_number_l257_257094


namespace arccos_one_over_sqrt_two_l257_257748

theorem arccos_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l257_257748


namespace graph_of_transformed_function_l257_257132

theorem graph_of_transformed_function
  (f : ℝ → ℝ)
  (hf : f⁻¹ 1 = 0) :
  f (1 - 1) = 1 :=
by
  sorry

end graph_of_transformed_function_l257_257132


namespace shaded_region_area_l257_257017

noncomputable def area_of_shaded_region (r : ℝ) (oa : ℝ) (ab_length : ℝ) : ℝ :=
  18 * (Real.sqrt 2) - 9 - (9 * Real.pi / 4)

theorem shaded_region_area (r : ℝ) (oa : ℝ) (ab_length : ℝ) : 
  r = 3 ∧ oa = 3 * Real.sqrt 2 ∧ ab_length = 6 * Real.sqrt 2 → 
  area_of_shaded_region r oa ab_length = 18 * (Real.sqrt 2) - 9 - (9 * Real.pi / 4) :=
by
  intro h
  obtain ⟨hr, hoa, hab⟩ := h
  rw [hr, hoa, hab]
  exact rfl

end shaded_region_area_l257_257017


namespace minimize_total_price_l257_257831

noncomputable def total_price (a : ℝ) (m x : ℝ) : ℝ :=
  a * ((m / 2 + x)^2 + (m / 2 - x)^2)

theorem minimize_total_price (a m : ℝ) : 
  ∃ y : ℝ, (∀ x, total_price a m x ≥ y) ∧ y = total_price a m 0 :=
by
  sorry

end minimize_total_price_l257_257831


namespace number_of_heaps_is_5_l257_257864

variable (bundles : ℕ) (bunches : ℕ) (heaps : ℕ) (total_removed : ℕ)
variable (sheets_per_bunch : ℕ) (sheets_per_bundle : ℕ) (sheets_per_heap : ℕ)

def number_of_heaps (bundles : ℕ) (sheets_per_bundle : ℕ)
                    (bunches : ℕ) (sheets_per_bunch : ℕ)
                    (total_removed : ℕ) (sheets_per_heap : ℕ) :=
  (total_removed - (bundles * sheets_per_bundle + bunches * sheets_per_bunch)) / sheets_per_heap

theorem number_of_heaps_is_5 :
  number_of_heaps 3 2 2 4 114 20 = 5 :=
by
  unfold number_of_heaps
  sorry

end number_of_heaps_is_5_l257_257864


namespace rahul_meena_work_together_l257_257459

theorem rahul_meena_work_together (days_rahul : ℚ) (days_meena : ℚ) (combined_days : ℚ) :
  days_rahul = 5 ∧ days_meena = 10 → combined_days = 10 / 3 :=
by
  intros h
  sorry

end rahul_meena_work_together_l257_257459


namespace find_vidya_age_l257_257019

theorem find_vidya_age (V M : ℕ) (h1: M = 3 * V + 5) (h2: M = 44) : V = 13 :=
by {
  sorry
}

end find_vidya_age_l257_257019


namespace cuboid_height_l257_257611

theorem cuboid_height
  (volume : ℝ)
  (width : ℝ)
  (length : ℝ)
  (height : ℝ)
  (h_volume : volume = 315)
  (h_width : width = 9)
  (h_length : length = 7)
  (h_volume_eq : volume = length * width * height) :
  height = 5 :=
by
  sorry

end cuboid_height_l257_257611


namespace find_prime_pairs_l257_257933

open Nat

def divides (a b : ℕ) : Prop := ∃ k, b = a * k

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def valid_prime_pairs (p q : ℕ): Prop :=
  is_prime p ∧ is_prime q ∧ divides p (30 * q - 1) ∧ divides q (30 * p - 1)

theorem find_prime_pairs :
  { (p, q) | valid_prime_pairs p q } = { (7, 11), (11, 7), (59, 61), (61, 59) } :=
sorry

end find_prime_pairs_l257_257933


namespace convert_246_octal_to_decimal_l257_257074

theorem convert_246_octal_to_decimal : 2 * (8^2) + 4 * (8^1) + 6 * (8^0) = 166 := 
by
  -- We skip the proof part as it is not required in the task
  sorry

end convert_246_octal_to_decimal_l257_257074


namespace largest_multiple_of_7_negation_greater_than_neg_150_l257_257341

theorem largest_multiple_of_7_negation_greater_than_neg_150 : 
  ∃ k : ℤ, k * 7 = 147 ∧ ∀ n : ℤ, (k < n → n * 7 ≤ 150) :=
by
  use 21
  sorry

end largest_multiple_of_7_negation_greater_than_neg_150_l257_257341


namespace simplify_expression_1_simplify_expression_2_l257_257523

-- Statement for the first problem
theorem simplify_expression_1 (a : ℝ) : 2 * a * (a - 3) - a^2 = a^2 - 6 * a := 
by sorry

-- Statement for the second problem
theorem simplify_expression_2 (x : ℝ) : (x - 1) * (x + 2) - x * (x + 1) = -2 := 
by sorry

end simplify_expression_1_simplify_expression_2_l257_257523


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l257_257699

theorem arccos_one_over_sqrt_two_eq_pi_over_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l257_257699


namespace hyperbola_foci_distance_l257_257572

-- Definitions based on the problem conditions
def hyperbola (x y : ℝ) : Prop := x^2 - (y^2 / 9) = 1

def foci_distance (PF1 : ℝ) : Prop := PF1 = 5

-- Main theorem stating the problem and expected outcome
theorem hyperbola_foci_distance (x y PF2 : ℝ) 
  (P_on_hyperbola : hyperbola x y) 
  (PF1_dist : foci_distance (dist (x, y) (some_focal_point_x1, 0))) :
  dist (x, y) (some_focal_point_x2, 0) = 7 ∨ dist (x, y) (some_focal_point_x2, 0) = 3 :=
sorry

end hyperbola_foci_distance_l257_257572


namespace maximum_value_is_one_div_sqrt_two_l257_257303

noncomputable def maximum_value_2ab_root2_plus_2ac_plus_2bc (a b c : ℝ) : ℝ :=
  2 * a * b * Real.sqrt 2 + 2 * a * c + 2 * b * c

theorem maximum_value_is_one_div_sqrt_two (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h : a^2 + b^2 + c^2 = 1) :
  maximum_value_2ab_root2_plus_2ac_plus_2bc a b c ≤ 1 / Real.sqrt 2 :=
by
  sorry

end maximum_value_is_one_div_sqrt_two_l257_257303


namespace number_of_boys_l257_257013

theorem number_of_boys 
  (B G : ℕ) 
  (h1 : B + G = 650) 
  (h2 : G = B + 106) :
  B = 272 :=
sorry

end number_of_boys_l257_257013


namespace infinitely_many_solutions_implies_b_eq_neg6_l257_257107

theorem infinitely_many_solutions_implies_b_eq_neg6 (b : ℤ) :
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 8)) → b = -6 :=
  sorry

end infinitely_many_solutions_implies_b_eq_neg6_l257_257107


namespace sum_of_cubes_eq_neg_27_l257_257799

theorem sum_of_cubes_eq_neg_27
  (a b c : ℝ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_condition : (a^3 + 9) / a = (b^3 + 9) / b ∧ (b^3 + 9) / b = (c^3 + 9) / c) :
  a^3 + b^3 + c^3 = -27 :=
sorry

end sum_of_cubes_eq_neg_27_l257_257799


namespace find_number_l257_257497

theorem find_number (x : ℝ) (h : 0.7 * x = 48 + 22) : x = 100 :=
by
  sorry

end find_number_l257_257497


namespace trig_identity_proof_l257_257977

noncomputable def check_trig_identities (α β : ℝ) : Prop :=
  3 * Real.sin α - Real.sin β = Real.sqrt 10 ∧ α + β = Real.pi / 2

theorem trig_identity_proof (α β : ℝ) (h : check_trig_identities α β) :
  Real.sin α = 3 * Real.sqrt 10 / 10 ∧ Real.cos (2 * β) = 4 / 5 := by
  sorry

end trig_identity_proof_l257_257977


namespace solve_equation_l257_257593

theorem solve_equation :
  ∃ (a b c d : ℚ), 
  (a^2 + b^2 + c^2 + d^2 - a * b - b * c - c * d - d + 2 / 5 = 0) ∧ 
  (a = 1 / 5 ∧ b = 2 / 5 ∧ c = 3 / 5 ∧ d = 4 / 5) := sorry

end solve_equation_l257_257593


namespace arnold_total_protein_l257_257662

-- Definitions and conditions
def collagen_protein_per_scoop : ℕ := 18 / 2
def protein_powder_protein_per_scoop : ℕ := 21
def steak_protein : ℕ := 56

def collagen_scoops : ℕ := 1
def protein_powder_scoops : ℕ := 1
def steaks : ℕ := 1

-- Statement of the theorem/problem
theorem arnold_total_protein : 
  (collagen_protein_per_scoop * collagen_scoops) + 
  (protein_powder_protein_per_scoop * protein_powder_scoops) + 
  (steak_protein * steaks) = 86 :=
by
  sorry

end arnold_total_protein_l257_257662


namespace intersection_of_A_and_B_l257_257404

-- Definitions of sets A and B
def A : Set ℤ := {1, 0, 3}
def B : Set ℤ := {-1, 1, 2, 3}

-- Statement of the theorem
theorem intersection_of_A_and_B : A ∩ B = {1, 3} :=
  sorry

end intersection_of_A_and_B_l257_257404


namespace walking_rate_on_escalator_l257_257381

theorem walking_rate_on_escalator (v : ℝ)
  (escalator_speed : ℝ := 12)
  (escalator_length : ℝ := 196)
  (travel_time : ℝ := 14)
  (effective_speed : ℝ := v + escalator_speed)
  (distance_eq : effective_speed * travel_time = escalator_length) :
  v = 2 := by
  sorry

end walking_rate_on_escalator_l257_257381


namespace probability_hare_killed_l257_257014

theorem probability_hare_killed (P_hit_1 P_hit_2 P_hit_3 : ℝ)
  (h1 : P_hit_1 = 3 / 5) (h2 : P_hit_2 = 3 / 10) (h3 : P_hit_3 = 1 / 10) :
  (1 - ((1 - P_hit_1) * (1 - P_hit_2) * (1 - P_hit_3))) = 0.748 :=
by
  sorry

end probability_hare_killed_l257_257014


namespace equal_candy_distribution_l257_257297

theorem equal_candy_distribution :
  ∀ (candies friends : ℕ), candies = 30 → friends = 4 → candies % friends = 2 :=
by
  sorry

end equal_candy_distribution_l257_257297


namespace polynomial_roots_unique_b_c_l257_257875

theorem polynomial_roots_unique_b_c :
    ∀ (r : ℝ), (r ^ 2 - 2 * r - 1 = 0) → (r ^ 5 - 29 * r - 12 = 0) :=
by
    sorry

end polynomial_roots_unique_b_c_l257_257875


namespace vovochka_add_correct_pairs_vovochka_min_diff_incorrect_l257_257782

-- Vovochka's addition method: adding two digits without carrying over
def vovochka_add (a b : ℕ) : ℕ := (a % 10 + b % 10) + ((a / 10 % 10 + b / 10 % 10) * 10) + ((a / 100 + b / 100) * 100)

-- Part (a): number of pairs producing correct result with Vovochka’s method
def correct_vovochka_pairs_count : ℕ := 244620

-- Part (b): smallest possible difference when Vovochka’s method is incorrect
def min_diff_vovochka_method : ℕ := 1800

-- Proving the number of correct cases equals 244620
theorem vovochka_add_correct_pairs :
  let count := ∑ (a b : ℕ) in finset.Icc 100 999, if vovochka_add a b = a + b then 1 else 0
  in count = correct_vovochka_pairs_count := sorry

-- Proving the smallest possible difference when Vovochka’s method is incorrect
theorem vovochka_min_diff_incorrect :
  let min_diff := min (∑ (a b : ℕ) in finset.Icc 100 999, if vovochka_add a b ≠ a + b then nat.abs (vovochka_add a b - (a + b)) else ⊤)
  in min_diff = min_diff_vovochka_method := sorry

end vovochka_add_correct_pairs_vovochka_min_diff_incorrect_l257_257782


namespace stationary_train_length_l257_257655

noncomputable def speed_train_kmh : ℝ := 144
noncomputable def speed_train_ms : ℝ := (speed_train_kmh * 1000) / 3600
noncomputable def time_to_pass_pole : ℝ := 8
noncomputable def time_to_pass_stationary : ℝ := 18
noncomputable def length_moving_train : ℝ := speed_train_ms * time_to_pass_pole
noncomputable def total_distance : ℝ := speed_train_ms * time_to_pass_stationary
noncomputable def length_stationary_train : ℝ := total_distance - length_moving_train

theorem stationary_train_length :
  length_stationary_train = 400 := by
  sorry

end stationary_train_length_l257_257655


namespace number_of_subsets_of_intersection_l257_257559

open Finset

theorem number_of_subsets_of_intersection (A B : Finset ℕ) (hA : A = {1, 2, 3}) (hB : B = {1, 3, 4}) :
  card (powerset (A ∩ B)) = 4 :=
by
  sorry

end number_of_subsets_of_intersection_l257_257559


namespace plane_split_into_8_regions_l257_257872

-- Define the conditions as separate lines in the plane.
def line1 (x y : ℝ) : Prop := y = 2 * x
def line2 (x y : ℝ) : Prop := y = (1 / 2) * x
def line3 (x y : ℝ) : Prop := x = y

-- Define a theorem stating that these lines together split the plane into 8 regions.
theorem plane_split_into_8_regions :
  (∀ (x y : ℝ), line1 x y ∨ line2 x y ∨ line3 x y) →
  -- The plane is split into exactly 8 regions by these lines
  ∃ (regions : ℕ), regions = 8 :=
sorry

end plane_split_into_8_regions_l257_257872


namespace determine_signs_l257_257806

theorem determine_signs (a b c : ℝ) (h1 : a != 0 ∧ b != 0 ∧ c == 0)
  (h2 : a > 0 ∨ (b + c) > 0) : a > 0 ∧ b < 0 ∧ c = 0 :=
by
  sorry

end determine_signs_l257_257806


namespace cos_sq_sub_sin_sq_pi_div_12_l257_257939

theorem cos_sq_sub_sin_sq_pi_div_12 : 
  (Real.cos (π / 12))^2 - (Real.sin (π / 12))^2 = Real.cos (π / 6) :=
by
  sorry

end cos_sq_sub_sin_sq_pi_div_12_l257_257939


namespace foreign_students_next_sem_eq_740_l257_257518

def total_students : ℕ := 1800
def percentage_foreign : ℕ := 30
def new_foreign_students : ℕ := 200

def initial_foreign_students : ℕ := total_students * percentage_foreign / 100
def total_foreign_students_next_semester : ℕ :=
  initial_foreign_students + new_foreign_students

theorem foreign_students_next_sem_eq_740 :
  total_foreign_students_next_semester = 740 :=
by
  sorry

end foreign_students_next_sem_eq_740_l257_257518


namespace count_three_digit_powers_of_two_l257_257421

theorem count_three_digit_powers_of_two : 
  ∃ n : ℕ, (n = 3) ∧ (finset.filter (λ n, 100 ≤ 2^n ∧ 2^n < 1000) (finset.range 16)).card = n :=
by
  sorry

end count_three_digit_powers_of_two_l257_257421


namespace truthfulness_count_l257_257886

-- Define variables to represent the number of warriors and their response counts
def num_warriors : Nat := 33
def yes_sword : Nat := 13
def yes_spear : Nat := 15
def yes_axe : Nat := 20
def yes_bow : Nat := 27

-- Define the total number of "Yes" answers
def total_yes_answers : Nat := yes_sword + yes_spear + yes_axe + yes_bow

theorem truthfulness_count :
  ∃ x : Nat, x + 3 * (num_warriors - x) = total_yes_answers ∧ x = 12 :=
by
  sorry

end truthfulness_count_l257_257886


namespace average_of_remaining_two_numbers_l257_257845

theorem average_of_remaining_two_numbers (a b c d e f : ℝ) 
  (h1 : (a + b + c + d + e + f) / 6 = 8) 
  (h2 : (a + b + c + d) / 4 = 5) : 
  (e + f) / 2 = 14 := 
by  
  sorry

end average_of_remaining_two_numbers_l257_257845


namespace problem_arith_seq_l257_257953

variables {a : ℕ → ℝ} (d : ℝ)
def is_arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem problem_arith_seq (h_arith : is_arithmetic_sequence a) 
  (h_condition : a 1 + a 6 + a 11 = 3) 
  : a 3 + a 9 = 2 :=
sorry

end problem_arith_seq_l257_257953


namespace carl_max_value_l257_257868

-- Definitions based on problem conditions.
def value_of_six_pound_rock : ℕ := 20
def weight_of_six_pound_rock : ℕ := 6
def value_of_three_pound_rock : ℕ := 9
def weight_of_three_pound_rock : ℕ := 3
def value_of_two_pound_rock : ℕ := 4
def weight_of_two_pound_rock : ℕ := 2
def max_weight_carl_can_carry : ℕ := 24

/-- Proves that Carl can carry rocks worth maximum 80 dollars given the conditions. -/
theorem carl_max_value : ∃ (n m k : ℕ),
    n * weight_of_six_pound_rock + m * weight_of_three_pound_rock + k * weight_of_two_pound_rock ≤ max_weight_carl_can_carry ∧
    n * value_of_six_pound_rock + m * value_of_three_pound_rock + k * value_of_two_pound_rock = 80 :=
by
  sorry

end carl_max_value_l257_257868


namespace fraction_red_knights_magical_l257_257601

theorem fraction_red_knights_magical (total_knights : ℕ) (fraction_red fraction_magical : ℚ)
  (fraction_red_twice_fraction_blue : ℚ) 
  (h_total_knights : total_knights > 0)
  (h_fraction_red : fraction_red = 2 / 7)
  (h_fraction_magical : fraction_magical = 1 / 6)
  (h_relation : fraction_red_twice_fraction_blue = 2)
  (h_magic_eq : (total_knights : ℚ) * fraction_magical = 
    total_knights * fraction_red * fraction_red_twice_fraction_blue * fraction_magical / 2 + 
    total_knights * (1 - fraction_red) * fraction_magical / 2) :
  total_knights * (fraction_red * fraction_red_twice_fraction_blue / (fraction_red * fraction_red_twice_fraction_blue + (1 - fraction_red) / 2)) = 
  total_knights * 7 / 27 := 
sorry

end fraction_red_knights_magical_l257_257601


namespace max_kings_l257_257194

theorem max_kings (initial_kings : ℕ) (kings_attacking_each_other : initial_kings = 21) 
  (no_two_kings_attack : ∀ kings_remaining, kings_remaining ≤ 16) : 
  ∃ kings_remaining, kings_remaining = 16 :=
by
  sorry

end max_kings_l257_257194


namespace value_of_x_l257_257981

theorem value_of_x :
  ∀ (x : ℕ), 
    x = 225 + 2 * 15 * 9 + 81 → 
    x = 576 := 
by
  intro x h
  sorry

end value_of_x_l257_257981


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l257_257702

theorem arccos_one_over_sqrt_two_eq_pi_over_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l257_257702


namespace shaded_region_area_l257_257565

theorem shaded_region_area (r : ℝ) (n : ℕ) (shaded_area : ℝ) (h_r : r = 3) (h_n : n = 6) :
  shaded_area = 27 * Real.pi - 54 := by
  sorry

end shaded_region_area_l257_257565


namespace pow_div_mul_pow_eq_l257_257865

theorem pow_div_mul_pow_eq (a b c d : ℕ) (h_a : a = 8) (h_b : b = 5) (h_c : c = 2) (h_d : d = 6) :
  (a^b / a^c) * (4^6) = 2^21 := by
  sorry

end pow_div_mul_pow_eq_l257_257865


namespace construct_triangle_condition_l257_257751

theorem construct_triangle_condition (m_a f_a s_a : ℝ) : 
  (m_a < f_a) ∧ (f_a < s_a) ↔ (exists A B C : Type, true) :=
sorry

end construct_triangle_condition_l257_257751


namespace second_term_arithmetic_seq_l257_257151

variable (a d : ℝ)

theorem second_term_arithmetic_seq (h : a + (a + 2 * d) = 8) : a + d = 4 := by
  sorry

end second_term_arithmetic_seq_l257_257151


namespace right_isosceles_triangle_acute_angle_45_l257_257989

theorem right_isosceles_triangle_acute_angle_45
    (a : ℝ)
    (h_leg_conditions : ∀ b : ℝ, a = b)
    (h_hypotenuse_condition : ∀ c : ℝ, c^2 = 2 * (a * a)) :
    ∃ θ : ℝ, θ = 45 :=
by
    sorry

end right_isosceles_triangle_acute_angle_45_l257_257989


namespace aang_caught_7_fish_l257_257656

theorem aang_caught_7_fish (A : ℕ) (h_avg : (A + 5 + 12) / 3 = 8) : A = 7 :=
by
  sorry

end aang_caught_7_fish_l257_257656


namespace max_value_of_expression_l257_257803

theorem max_value_of_expression (a b c : ℝ) (h1: 0 ≤ a) (h2: 0 ≤ b) (h3: 0 ≤ c) 
    (h_sum: a + b + c = 3) :
    (ab / (a + b) + ac / (a + c) + bc / (b + c) ≤ 3 / 2) :=
by
  sorry

end max_value_of_expression_l257_257803


namespace simplify_nested_fraction_l257_257522

theorem simplify_nested_fraction :
  (1 : ℚ) / (1 + (1 / (3 + (1 / 4)))) = 13 / 17 :=
by
  sorry

end simplify_nested_fraction_l257_257522


namespace largest_multiple_of_7_negated_gt_neg_150_l257_257353

theorem largest_multiple_of_7_negated_gt_neg_150 :
  ∃ (n : ℕ), (negate (n * 7) > -150) ∧ (∀ m : ℕ, (negate (m * 7) > -150) → m ≤ n) ∧ (n * 7 = 147) :=
sorry

end largest_multiple_of_7_negated_gt_neg_150_l257_257353


namespace sum_of_cubes_eq_neg_27_l257_257796

variable {a b c : ℝ}

-- Define the condition that k is the same for a, b, and c
def same_k (a b c k : ℝ) : Prop :=
  k = (a^3 + 9) / a ∧ k = (b^3 + 9) / b ∧ k = (c^3 + 9) / c

-- Theorem: Given the conditions, a^3 + b^3 + c^3 = -27
theorem sum_of_cubes_eq_neg_27 (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_same_k : ∃ k, same_k a b c k) :
  a^3 + b^3 + c^3 = -27 :=
sorry

end sum_of_cubes_eq_neg_27_l257_257796


namespace number_of_truth_tellers_is_twelve_l257_257906
noncomputable theory

section
variables (x : ℕ)
variables (y : ℕ)
variables (a b c d : ℕ)

-- Given conditions
def total_warriors : ℕ := 33
def yes_sword : ℕ := 13
def yes_spear : ℕ := 15
def yes_axe : ℕ := 20
def yes_bow : ℕ := 27

-- Calculate total yes answers (condition 6)
def total_yes_answers : ℕ := yes_sword + yes_spear + yes_axe + yes_bow

-- Truth-tellers say "yes" to only one question, liers say "yes" to three questions
def truth_yes : ℕ := x * 1
def lie_yes : ℕ := (total_warriors - x) * 3

theorem number_of_truth_tellers_is_twelve
  (h1 : total_warriors = 33)
  (h2 : yes_sword = 13)
  (h3 : yes_spear = 15)
  (h4 : yes_axe = 20)
  (h5 : yes_bow = 27)
  (h6 : total_yes_answers = 75)
  (h7 : total_yes_answers = truth_yes + lie_yes) :
    x = 12 :=
  by sorry

end

end number_of_truth_tellers_is_twelve_l257_257906


namespace rationalize_fraction_l257_257179

open BigOperators

theorem rationalize_fraction :
  (3 : ℝ) / (Real.sqrt 50 + 2) = (15 * Real.sqrt 2 - 6) / 46 :=
by
  -- Our proof intention will be inserted here.
  sorry

end rationalize_fraction_l257_257179


namespace integer_solutions_l257_257931

theorem integer_solutions (a b c : ℤ) (h₁ : 1 < a) 
    (h₂ : a < b) (h₃ : b < c) 
    (h₄ : (a-1) * (b-1) * (c-1) ∣ a * b * c - 1) :
    (a = 3 ∧ b = 5 ∧ c = 15) 
    ∨ (a = 2 ∧ b = 4 ∧ c = 8) :=
by sorry

end integer_solutions_l257_257931


namespace sum_of_divisors_of_252_l257_257026

theorem sum_of_divisors_of_252 :
  ∑ (d : ℕ) in (finset.filter (λ x, 252 % x = 0) (finset.range (252 + 1))), d = 728 :=
by
  sorry

end sum_of_divisors_of_252_l257_257026


namespace max_areas_in_disk_l257_257850

noncomputable def max_non_overlapping_areas (n : ℕ) : ℕ := 5 * n + 1

theorem max_areas_in_disk (n : ℕ) : 
  let disk_divided_by_2n_radii_and_two_secant_lines_areas  := (5 * n + 1)
  disk_divided_by_2n_radii_and_two_secant_lines_areas = max_non_overlapping_areas n := by sorry

end max_areas_in_disk_l257_257850


namespace neg_abs_neg_three_l257_257603

theorem neg_abs_neg_three : -|(-3)| = -3 := 
by
  sorry

end neg_abs_neg_three_l257_257603


namespace dataset_mode_l257_257122

noncomputable def find_mode_of_dataset (s : List ℤ) (mean : ℤ) : List ℤ :=
  let x := (mean * s.length) - (s.sum - x)
  let new_set := s.map (λ n => if n = x then 5 else n)
  let grouped := new_set.groupBy id
  let mode_elements := grouped.foldl
    (λ acc lst => if lst.length > acc.length then lst else acc) []
  mode_elements

theorem dataset_mode :
  find_mode_of_dataset [1, 0, -3, 5, 5, 2, -3] 1 = [-3, 5] :=
by
  sorry

end dataset_mode_l257_257122


namespace arccos_of_one_over_sqrt_two_l257_257722

theorem arccos_of_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
sorry

end arccos_of_one_over_sqrt_two_l257_257722


namespace arccos_proof_l257_257686

noncomputable def arccos_identity : Prop := 
  ∃ x : ℝ, x = 1 / Real.sqrt 2 ∧ Real.arccos x = Real.pi / 4

theorem arccos_proof : arccos_identity :=
by
  sorry

end arccos_proof_l257_257686


namespace solve_fraction_equation_l257_257465

theorem solve_fraction_equation (x : ℚ) (h : x ≠ -1) : 
  (x / (x + 1) = 2 * x / (3 * x + 3) - 1) → x = -3 / 4 :=
by
  sorry

end solve_fraction_equation_l257_257465


namespace geometric_sequence_a6_a8_sum_l257_257996

theorem geometric_sequence_a6_a8_sum 
  (a : ℕ → ℕ) (q : ℕ) 
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h1 : a 1 + a 3 = 5)
  (h2 : a 2 + a 4 = 10) : 
  a 6 + a 8 = 160 := 
sorry

end geometric_sequence_a6_a8_sum_l257_257996


namespace no_3_digit_number_with_digit_sum_27_and_even_l257_257417

-- Define what it means for a number to be 3-digit
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

-- Define the digit-sum function
def digitSum (n : ℕ) : ℕ :=
  (n / 100) + (n % 100 / 10) + (n % 10)

-- Define what it means for a number to be even
def isEven (n : ℕ) : Prop := n % 2 = 0

-- State the proof problem
theorem no_3_digit_number_with_digit_sum_27_and_even :
  ∀ n : ℕ, isThreeDigit n → digitSum n = 27 → isEven n → false :=
by
  -- Proof should go here
  sorry

end no_3_digit_number_with_digit_sum_27_and_even_l257_257417


namespace warriors_truth_tellers_l257_257894

/-- There are 33 warriors. Each warrior is either a truth-teller or a liar, 
    with only one favorite weapon: a sword, a spear, an axe, or a bow. 
    They were asked four questions, and the number of "Yes" answers to the 
    questions are 13, 15, 20, and 27 respectively. Prove that the number of 
    warriors who always tell the truth is 12. -/
theorem warriors_truth_tellers
  (warriors : ℕ) (truth_tellers : ℕ)
  (yes_to_sword : ℕ) (yes_to_spear : ℕ)
  (yes_to_axe : ℕ) (yes_to_bow : ℕ)
  (h1 : warriors = 33)
  (h2 : yes_to_sword = 13)
  (h3 : yes_to_spear = 15)
  (h4 : yes_to_axe = 20)
  (h5 : yes_to_bow = 27)
  (h6 : yes_to_sword + yes_to_spear + yes_to_axe + yes_to_bow = 75) :
  truth_tellers = 12 := by
  -- Proof will be here
  sorry

end warriors_truth_tellers_l257_257894


namespace product_of_roots_of_polynomial_l257_257235

theorem product_of_roots_of_polynomial : 
  ∀ x : ℝ, (x + 3) * (x - 4) = 22 → ∃ a b : ℝ, (x^2 - x - 34 = 0) ∧ (a * b = -34) :=
by
  sorry

end product_of_roots_of_polynomial_l257_257235


namespace equation_of_line_perpendicular_to_l_l257_257943

open Real

theorem equation_of_line_perpendicular_to_l
  (a : ℝ) (h_pos : a > 0)
  (h_chord : √(a^2 - 1) = 2)
  (h_center : (3 : ℝ) = a)
  :
  ∃ m : ℝ, (3 + 0 + m = 0) ∧ (∀ b : ℝ, ∃ x y : ℝ, l x y → x + y + m = 0) :=
by
  sorry

end equation_of_line_perpendicular_to_l_l257_257943


namespace find_number_being_divided_l257_257454

theorem find_number_being_divided (divisor quotient remainder : ℕ) (h1: divisor = 15) (h2: quotient = 9) (h3: remainder = 1) : 
  divisor * quotient + remainder = 136 :=
by
  -- Simplification and computation would follow here
  sorry

end find_number_being_divided_l257_257454


namespace base8_to_base10_conversion_l257_257061

def base8_to_base10 (n : Nat) : Nat := 
  match n with
  | 246 => 2 * 8^2 + 4 * 8^1 + 6 * 8^0
  | _ => 0  -- We define this only for the number 246_8

theorem base8_to_base10_conversion : base8_to_base10 246 = 166 := by 
  sorry

end base8_to_base10_conversion_l257_257061


namespace julia_total_balls_l257_257163

theorem julia_total_balls :
  (3 * 19) + (10 * 19) + (8 * 19) = 399 :=
by
  -- proof goes here
  sorry

end julia_total_balls_l257_257163


namespace find_unit_prices_minimize_cost_l257_257564

-- Definitions for the given prices and conditions
def cypress_price := 200
def pine_price := 150

def cost_eq1 (x y : ℕ) : Prop := 2 * x + 3 * y = 850
def cost_eq2 (x y : ℕ) : Prop := 3 * x + 2 * y = 900

-- Proving the unit prices of cypress and pine trees
theorem find_unit_prices (x y : ℕ) (h1 : cost_eq1 x y) (h2 : cost_eq2 x y) :
  x = cypress_price ∧ y = pine_price :=
sorry

-- Definitions for the number of trees and their costs
def total_trees := 80
def cypress_min (a : ℕ) : Prop := a ≥ 2 * (total_trees - a)
def total_cost (a : ℕ) : ℕ := 200 * a + 150 * (total_trees - a)

-- Conditions given for minimizing the cost
theorem minimize_cost (a : ℕ) (h1 : cypress_min a) : 
  a = 54 ∧ (total_trees - a) = 26 ∧ total_cost a = 14700 :=
sorry

end find_unit_prices_minimize_cost_l257_257564


namespace min_value_of_squared_sums_l257_257448

theorem min_value_of_squared_sums (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : 
  ∃ B, (B = x^2 + y^2 + z^2) ∧ (B ≥ 4) := 
by {
  sorry -- Proof will be provided here.
}

end min_value_of_squared_sums_l257_257448


namespace curve_is_parabola_l257_257401

theorem curve_is_parabola (t : ℝ) : 
  ∃ (x y : ℝ), (x = 3^t - 2) ∧ (y = 9^t - 4 * 3^t + 2 * t - 4) ∧ (∃ a b c : ℝ, y = a * x^2 + b * x + c) :=
by sorry

end curve_is_parabola_l257_257401


namespace greater_number_l257_257481

theorem greater_number (a b : ℝ) (h1 : a + b = 40) (h2 : a - b = 12) : a = 26 :=
by
  have h3 : 2 * a = 52 := by linarith
  have h4 : a = 26 := by linarith
  exact h4

end greater_number_l257_257481


namespace bogatyrs_truthful_count_l257_257902

noncomputable def number_of_truthful_warriors (total_warriors: ℕ) (sword_yes: ℕ) (spear_yes: ℕ) (axe_yes: ℕ) (bow_yes: ℕ) : ℕ :=
  let total_yes := sword_yes + spear_yes + axe_yes + bow_yes
  let lying_warriors := (total_yes - total_warriors) / 2
  total_warriors - lying_warriors

theorem bogatyrs_truthful_count :
  number_of_truthful_warriors 33 13 15 20 27 = 12 := by
  sorry

end bogatyrs_truthful_count_l257_257902


namespace bogatyrs_truthful_count_l257_257900

noncomputable def number_of_truthful_warriors (total_warriors: ℕ) (sword_yes: ℕ) (spear_yes: ℕ) (axe_yes: ℕ) (bow_yes: ℕ) : ℕ :=
  let total_yes := sword_yes + spear_yes + axe_yes + bow_yes
  let lying_warriors := (total_yes - total_warriors) / 2
  total_warriors - lying_warriors

theorem bogatyrs_truthful_count :
  number_of_truthful_warriors 33 13 15 20 27 = 12 := by
  sorry

end bogatyrs_truthful_count_l257_257900


namespace polynomial_expansion_abs_sum_l257_257553

theorem polynomial_expansion_abs_sum :
  let a_0 := 1
  let a_1 := -8
  let a_2 := 24
  let a_3 := -32
  let a_4 := 16
  |a_0| + |a_1| + |a_2| + |a_3| + |a_4| = 81 :=
by
  sorry

end polynomial_expansion_abs_sum_l257_257553


namespace salary_reduction_l257_257008

variable (S R : ℝ) (P : ℝ)
variable (h1 : R = S * (1 - P/100))
variable (h2 : S = R * (1 + 53.84615384615385 / 100))

theorem salary_reduction : P = 35 :=
by sorry

end salary_reduction_l257_257008


namespace quadratic_has_real_root_l257_257619

theorem quadratic_has_real_root (a b : ℝ) : 
  (¬(∀ x : ℝ, x^2 + a * x + b ≠ 0)) → (∃ x : ℝ, x^2 + a * x + b = 0) := 
by
  intro h
  sorry

end quadratic_has_real_root_l257_257619


namespace smallest_special_greater_than_3429_l257_257089

def is_special (n : ℕ) : Prop := (nat.digits 10 n).nodup ∧ (nat.digits 10 n).length = 4

theorem smallest_special_greater_than_3429 : ∃ n, n > 3429 ∧ is_special n ∧ 
  ∀ m, m > 3429 ∧ is_special m → n ≤ m :=
by
  use 3450
  sorry

end smallest_special_greater_than_3429_l257_257089


namespace triangle_area_l257_257788

theorem triangle_area (a b : ℝ) (sinC sinA : ℝ) 
  (h1 : a = Real.sqrt 5) 
  (h2 : b = 3) 
  (h3 : sinC = 2 * sinA) : 
  ∃ (area : ℝ), area = 3 := 
by 
  sorry

end triangle_area_l257_257788


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l257_257701

theorem arccos_one_over_sqrt_two_eq_pi_over_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l257_257701


namespace evaluate_g_at_neg_one_l257_257773

def g (x : ℝ) : ℝ := 5 * x^3 - 7 * x^2 - 3 * x + 9

theorem evaluate_g_at_neg_one : g (-1) = 7 :=
by 
  -- lean proof here
  sorry

end evaluate_g_at_neg_one_l257_257773


namespace fourth_term_of_geometric_progression_l257_257408

theorem fourth_term_of_geometric_progression (x : ℝ) (r : ℝ) 
  (h1 : (2 * x + 5) = r * x) 
  (h2 : (3 * x + 10) = r * (2 * x + 5)) : 
  (3 * x + 10) * r = -5 :=
by
  sorry

end fourth_term_of_geometric_progression_l257_257408


namespace convert_246_octal_to_decimal_l257_257071

theorem convert_246_octal_to_decimal : 2 * (8^2) + 4 * (8^1) + 6 * (8^0) = 166 := 
by
  -- We skip the proof part as it is not required in the task
  sorry

end convert_246_octal_to_decimal_l257_257071


namespace number_of_truthful_warriors_l257_257883

theorem number_of_truthful_warriors (total_warriors : ℕ) 
  (sword_yes : ℕ) (spear_yes : ℕ) (axe_yes : ℕ) (bow_yes : ℕ) 
  (always_tells_truth : ℕ → Prop)
  (always_lies : ℕ → Prop)
  (hv1 : total_warriors = 33)
  (hv2 : sword_yes = 13)
  (hv3 : spear_yes = 15)
  (hv4 : axe_yes = 20)
  (hv5 : bow_yes = 27) :
  ∃ truthful_warriors, truthful_warriors = 12 := 
by {
  sorry
}

end number_of_truthful_warriors_l257_257883


namespace pens_given_to_sharon_l257_257629

def initial_pens : Nat := 20
def mikes_pens : Nat := 22
def final_pens : Nat := 65

def total_pens_after_mike : Nat := initial_pens + mikes_pens
def total_pens_after_cindy : Nat := total_pens_after_mike * 2

theorem pens_given_to_sharon :
  total_pens_after_cindy - final_pens = 19 :=
by
  sorry

end pens_given_to_sharon_l257_257629


namespace smallest_enclosing_sphere_radius_l257_257239

-- Define the radius of each small sphere and the center set
def radius (r : ℝ) : Prop := r = 2

def center_set (C : Set (ℝ × ℝ × ℝ)) : Prop :=
  ∀ c ∈ C, ∃ x y z : ℝ, 
    (x = 2 ∨ x = -2) ∧ 
    (y = 2 ∨ y = -2) ∧ 
    (z = 2 ∨ z = -2) ∧
    (c = (x, y, z))

-- Prove the radius of the smallest enclosing sphere is 2√3 + 2
theorem smallest_enclosing_sphere_radius (r : ℝ) (C : Set (ℝ × ℝ × ℝ)) 
  (h_radius : radius r) (h_center_set : center_set C) :
  ∃ R : ℝ, R = 2 * Real.sqrt 3 + 2 :=
sorry

end smallest_enclosing_sphere_radius_l257_257239


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l257_257672

theorem arccos_one_over_sqrt_two_eq_pi_over_four : 
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l257_257672


namespace quadratic_other_root_l257_257427

theorem quadratic_other_root (m : ℝ) :
  (2 * 1^2 - m * 1 + 6 = 0) →
  ∃ y : ℝ, y ≠ 1 ∧ (2 * y^2 - m * y + 6 = 0) ∧ (1 * y = 3) :=
by
  intros h
  -- using sorry to skip the actual proof
  sorry

end quadratic_other_root_l257_257427


namespace min_xy_solution_l257_257252

theorem min_xy_solution (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 2 * x + 8 * y) :
  (x = 16 ∧ y = 4) :=
by
  sorry

end min_xy_solution_l257_257252


namespace remaining_pie_after_carlos_and_maria_l257_257668

theorem remaining_pie_after_carlos_and_maria (C M R : ℝ) (hC : C = 0.60) (hM : M = 0.25 * (1 - C)) : R = 1 - C - M → R = 0.30 :=
by
  intro hR
  simp only [hC, hM] at hR
  sorry

end remaining_pie_after_carlos_and_maria_l257_257668


namespace prob_snow_both_days_l257_257832

-- Definitions for the conditions
def prob_snow_monday : ℚ := 40 / 100
def prob_snow_tuesday : ℚ := 30 / 100

def independent_events (A B : Prop) : Prop := true  -- A placeholder definition of independence

-- The proof problem: 
theorem prob_snow_both_days : 
  independent_events (prob_snow_monday = 0.40) (prob_snow_tuesday = 0.30) →
  prob_snow_monday * prob_snow_tuesday = 0.12 := 
by 
  sorry

end prob_snow_both_days_l257_257832


namespace probability_abs_difference_gt_one_l257_257588

open Classical

noncomputable def dice_probability (x y : ℝ) : ℝ :=
if (x >= 0 ∧ x <= 2) ∧ (y >= 0 ∧ y <= 2) then
  if x = y then 1/8
  else if x = 0 ∧ y = 2 then 3/64
  else if x = 2 ∧ y = 0 then 3/64
  else if x = 0 ∧ y = 1 ∨ x = 1 ∧ y = 0 then 1/16
  else if x = 1 ∧ y = 2 ∨ x = 2 ∧ y = 1 then 1/16
else 0

theorem probability_abs_difference_gt_one :
  ∀ x y : ℝ, (0 ≤ x ∧ x ≤ 2) ∧ (0 ≤ y ∧ y ≤ 2) → (P (|x - y| > 1)) = 3/64 :=
begin
  intros x y,
  rw probability_eq_sum,
  exact sorry,
end

end probability_abs_difference_gt_one_l257_257588


namespace largest_multiple_of_7_neg_greater_than_neg_150_l257_257345

theorem largest_multiple_of_7_neg_greater_than_neg_150 : 
  ∃ (k : ℤ), k % 7 = 0 ∧ -k > -150 ∧ (∀ (m : ℤ), m % 7 = 0 ∧ -m > -150 → k ≥ m) ∧ k = 147 :=
by
  sorry

end largest_multiple_of_7_neg_greater_than_neg_150_l257_257345


namespace three_digit_powers_of_two_count_l257_257420

theorem three_digit_powers_of_two_count : 
  ∃ n_count : ℕ, (∀ n : ℕ, (100 ≤ 2^n ∧ 2^n < 1000) ↔ (n = 7 ∨ n = 8 ∨ n = 9)) ∧ n_count = 3 :=
by
  sorry

end three_digit_powers_of_two_count_l257_257420


namespace division_result_l257_257622

theorem division_result : 203515 / 2015 = 101 := 
by sorry

end division_result_l257_257622


namespace find_middle_integer_l257_257543

theorem find_middle_integer (a b c : ℕ) (h1 : a^2 = 97344) (h2 : c^2 = 98596) (h3 : c = a + 2) : b = a + 1 ∧ b = 313 :=
by
  sorry

end find_middle_integer_l257_257543


namespace min_largest_value_in_set_l257_257974

theorem min_largest_value_in_set (a b : ℕ) (h1 : 0 < a) (h2 : a < b) (h3 : (8:ℚ) / 19 * a * b ≤ (a - 1) * a / 2): a ≥ 13 :=
by
  sorry

end min_largest_value_in_set_l257_257974


namespace total_chairs_calculation_l257_257215

-- Definitions of the conditions
def numIndoorTables : Nat := 9
def numOutdoorTables : Nat := 11
def chairsPerIndoorTable : Nat := 10
def chairsPerOutdoorTable : Nat := 3

-- The proposition we want to prove
theorem total_chairs_calculation :
  numIndoorTables * chairsPerIndoorTable + numOutdoorTables * chairsPerOutdoorTable = 123 := by
sorry

end total_chairs_calculation_l257_257215


namespace exponent_equality_l257_257029

theorem exponent_equality (M : ℕ) : 32^4 * 4^6 = 2^M → M = 32 :=
by
  sorry

end exponent_equality_l257_257029


namespace infinite_chain_resistance_l257_257876

noncomputable def resistance_of_infinite_chain (R₀ : ℝ) : ℝ :=
  (R₀ * (1 + Real.sqrt 5)) / 2

theorem infinite_chain_resistance : resistance_of_infinite_chain 10 = 5 + 5 * Real.sqrt 5 :=
by
  sorry

end infinite_chain_resistance_l257_257876


namespace arithmetic_sequence_sum_l257_257607

theorem arithmetic_sequence_sum :
  ∃ (c d e : ℕ), 
  c = 15 + (9 - 3) ∧ 
  d = c + (9 - 3) ∧ 
  e = d + (9 - 3) ∧ 
  c + d + e = 81 :=
by 
  sorry

end arithmetic_sequence_sum_l257_257607


namespace solution_set_l257_257877

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ := x^2 + 3 * x - 4

-- Define the inequality
def inequality (x : ℝ) : Prop := quadratic_expr x > 0

-- State the theorem
theorem solution_set : ∀ x : ℝ, inequality x ↔ (x > 1 ∨ x < -4) :=
by
  sorry

end solution_set_l257_257877


namespace birthday_money_l257_257308

theorem birthday_money (x : ℤ) (h₀ : 16 + x - 25 = 19) : x = 28 :=
by
  sorry

end birthday_money_l257_257308


namespace ratio_of_shorts_to_pants_is_half_l257_257863

-- Define the parameters
def shirts := 4
def pants := 2 * shirts
def total_clothes := 16

-- Define the number of shorts
def shorts := total_clothes - (shirts + pants)

-- Define the ratio
def ratio := shorts / pants

-- Prove the ratio is 1/2
theorem ratio_of_shorts_to_pants_is_half : ratio = 1 / 2 :=
by
  -- Start the proof, but leave it as sorry
  sorry

end ratio_of_shorts_to_pants_is_half_l257_257863


namespace binom_10_0_equals_1_l257_257869

-- Define the factorial function
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Theorem to prove that binom 10 0 = 1
theorem binom_10_0_equals_1 :
  binom 10 0 = 1 := by
  sorry

end binom_10_0_equals_1_l257_257869


namespace gcd_lcm_sum_l257_257838

variable (a b : ℕ)

-- Definition for gcd
def gcdOf (a b : ℕ) : ℕ := Nat.gcd a b

-- Definition for lcm
def lcmOf (a b : ℕ) : ℕ := Nat.lcm a b

-- Statement of the problem
theorem gcd_lcm_sum (h1 : a = 8) (h2 : b = 12) : gcdOf a b + lcmOf a b = 28 := by
  sorry

end gcd_lcm_sum_l257_257838


namespace minimum_value_expression_l257_257399

theorem minimum_value_expression (a b c : ℝ) (h1 : b > c) (h2 : c > a) (h3 : b ≠ 0) :
  (∃ a b c : ℝ, (b > c ∧ c > a) ∧ b ≠ 0 ∧ (a + b) = b - c ∧ (b - c) = c - a ∧ (a - c) = 0 ∧
   ∀ x y z : ℝ, (x = a + b ∧ y = b - c ∧ z = c - a) → 
    (x^2 + y^2 + z^2) / b^2 = 4/3) :=
  sorry

end minimum_value_expression_l257_257399


namespace selling_price_increase_solution_maximum_profit_solution_l257_257501

-- Conditions
def purchase_price : ℝ := 30
def original_price : ℝ := 40
def monthly_sales : ℝ := 300
def sales_decrease_per_yuan : ℝ := 10

-- Questions
def selling_price_increase (x : ℝ) : Prop :=
  (x + 10) * (monthly_sales - sales_decrease_per_yuan * x) = 3360

def maximum_profit (x : ℝ) : Prop :=
  ∃ x : ℝ, 
    let M := -10 * x^2 + 200 * x + 3000 in
    M = 4000 ∧ x = 10

theorem selling_price_increase_solution : ∃ x : ℝ, selling_price_increase x := sorry

theorem maximum_profit_solution : ∃ x : ℝ, maximum_profit x := sorry

end selling_price_increase_solution_maximum_profit_solution_l257_257501


namespace complex_root_problem_l257_257937

theorem complex_root_problem (z : ℂ) :
  z^2 - 3*z = 10 - 6*Complex.I ↔
  z = 5.5 - 0.75 * Complex.I ∨
  z = -2.5 + 0.75 * Complex.I ∨
  z = 3.5 - 1.5 * Complex.I ∨
  z = -0.5 + 1.5 * Complex.I :=
sorry

end complex_root_problem_l257_257937


namespace butcher_net_loss_l257_257851

noncomputable def dishonest_butcher (advertised_price actual_price : ℝ) (quantity_sold : ℕ) (fine : ℝ) : ℝ :=
  let dishonest_gain_per_kg := actual_price - advertised_price
  let total_dishonest_gain := dishonest_gain_per_kg * quantity_sold
  fine - total_dishonest_gain

theorem butcher_net_loss 
  (advertised_price : ℝ) 
  (actual_price : ℝ) 
  (quantity_sold : ℕ) 
  (fine : ℝ)
  (h_advertised_price : advertised_price = 3.79)
  (h_actual_price : actual_price = 4.00)
  (h_quantity_sold : quantity_sold = 1800)
  (h_fine : fine = 500) :
  dishonest_butcher advertised_price actual_price quantity_sold fine = 122 := 
by
  simp [dishonest_butcher, h_advertised_price, h_actual_price, h_quantity_sold, h_fine]
  sorry

end butcher_net_loss_l257_257851


namespace mutual_fund_share_increase_l257_257666

theorem mutual_fund_share_increase (P : ℝ) (h1 : (P * 1.20) = 1.20 * P) (h2 : (1.20 * P) * (1 / 3) = 0.40 * P) :
  ((1.60 * P) = (P * 1.60)) :=
by
  sorry

end mutual_fund_share_increase_l257_257666


namespace village_population_l257_257020

variable (Px : ℕ) (t : ℕ) (dX dY : ℕ)
variable (Py : ℕ := 42000) (rateX : ℕ := 1200) (rateY : ℕ := 800) (timeYears : ℕ := 15)

theorem village_population : (Px - rateX * timeYears = Py + rateY * timeYears) → Px = 72000 :=
by
  sorry

end village_population_l257_257020


namespace expand_product_l257_257241

theorem expand_product (y : ℝ) : 5 * (y - 3) * (y + 10) = 5 * y^2 + 35 * y - 150 := 
  sorry

end expand_product_l257_257241


namespace sum_of_coeffs_l257_257403

theorem sum_of_coeffs (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) :
  (∃ a_0 a_1 a_2 a_3 a_4 a_5, (2 - x)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5)
  → (a_0 = 32 ∧ 1 = a_0 + a_1 + a_2 + a_3 + a_4 + a_5)
  → a_1 + a_2 + a_3 + a_4 + a_5 = -31 :=
by
  sorry

end sum_of_coeffs_l257_257403


namespace chinese_team_wins_gold_l257_257469

noncomputable def prob_player_a_wins : ℚ := 3 / 7
noncomputable def prob_player_b_wins : ℚ := 1 / 4

theorem chinese_team_wins_gold : prob_player_a_wins + prob_player_b_wins = 19 / 28 := by
  sorry

end chinese_team_wins_gold_l257_257469


namespace students_end_year_10_l257_257291

def students_at_end_of_year (initial_students : ℕ) (left_students : ℕ) (increase_percent : ℕ) : ℕ :=
  let remaining_students := initial_students - left_students
  let increased_students := (remaining_students * increase_percent) / 100
  remaining_students + increased_students

theorem students_end_year_10 : 
  students_at_end_of_year 10 4 70 = 10 := by 
  sorry

end students_end_year_10_l257_257291


namespace largest_multiple_of_18_with_8_and_0_digits_l257_257600

theorem largest_multiple_of_18_with_8_and_0_digits :
  ∃ m : ℕ, (∀ d ∈ (m.digits 10), d = 8 ∨ d = 0) ∧ (m % 18 = 0) ∧ (m = 8888888880) ∧ (m / 18 = 493826048) :=
by sorry

end largest_multiple_of_18_with_8_and_0_digits_l257_257600


namespace find_angle_A_min_perimeter_l257_257542

theorem find_angle_A (a b c : ℝ) (A B C : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) 
  (h₄ : a > 0 ∧ b > 0 ∧ c > 0) (h5 : b + c * Real.cos A = c + a * Real.cos C) 
  (hTriangle : A + B + C = Real.pi)
  (hSineLaw : Real.sin B = Real.sin C * Real.cos A + Real.sin A * Real.cos C) :
  A = Real.pi / 3 := 
by 
  sorry

theorem min_perimeter (a b c : ℝ) (A : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) 
  (h4 : a > 0 ∧ b > 0 ∧ c > 0 ∧ A = Real.pi / 3)
  (h_area : 1 / 2 * b * c * Real.sin A = Real.sqrt 3)
  (h_cosine : a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A) :
  a + b + c = 6 :=
by 
  sorry

end find_angle_A_min_perimeter_l257_257542


namespace base8_to_base10_conversion_l257_257064

def base8_to_base10 (n : Nat) : Nat := 
  match n with
  | 246 => 2 * 8^2 + 4 * 8^1 + 6 * 8^0
  | _ => 0  -- We define this only for the number 246_8

theorem base8_to_base10_conversion : base8_to_base10 246 = 166 := by 
  sorry

end base8_to_base10_conversion_l257_257064


namespace largest_multiple_of_7_gt_neg_150_l257_257355

theorem largest_multiple_of_7_gt_neg_150 : ∃ (x : ℕ), (x % 7 = 0) ∧ ((- (x : ℤ)) > -150) ∧ ∀ y : ℕ, (y % 7 = 0 ∧ (- (y : ℤ)) > -150) → y ≤ x :=
by
  sorry

end largest_multiple_of_7_gt_neg_150_l257_257355


namespace angle_in_fourth_quadrant_l257_257208

theorem angle_in_fourth_quadrant (θ : ℝ) (h : θ = -1445) : (θ % 360) > 270 ∧ (θ % 360) < 360 :=
by
  sorry

end angle_in_fourth_quadrant_l257_257208


namespace two_equal_sum_partition_three_equal_sum_partition_l257_257576

-- Definition 1: Sum of the set X_n
def sum_X_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Definition 2: Equivalences for partitioning X_n into two equal sum parts
def partition_two_equal_sum (n : ℕ) : Prop :=
  (n % 4 = 0 ∨ n % 4 = 3) ↔ ∃ (A B : Finset ℕ), A ∪ B = Finset.range n ∧ A ∩ B = ∅ ∧ A.sum id = B.sum id

-- Definition 3: Equivalences for partitioning X_n into three equal sum parts
def partition_three_equal_sum (n : ℕ) : Prop :=
  (n % 3 ≠ 1) ↔ ∃ (A B C : Finset ℕ), A ∪ B ∪ C = Finset.range n ∧ (A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅) ∧ A.sum id = B.sum id ∧ B.sum id = C.sum id

-- Main theorem statements
theorem two_equal_sum_partition (n : ℕ) : partition_two_equal_sum n :=
  sorry

theorem three_equal_sum_partition (n : ℕ) : partition_three_equal_sum n :=
  sorry

end two_equal_sum_partition_three_equal_sum_partition_l257_257576


namespace xyz_value_l257_257411

theorem xyz_value
  (x y z : ℝ)
  (h1 : (x + y + z) * (xy + xz + yz) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9)
  (h3 : x + y + z = 3)
  : xyz = 5 :=
by
  sorry

end xyz_value_l257_257411


namespace area_of_rectangular_plot_l257_257322

theorem area_of_rectangular_plot (B L : ℕ) (h1 : L = 3 * B) (h2 : B = 18) : L * B = 972 := by
  sorry

end area_of_rectangular_plot_l257_257322


namespace num_dogs_l257_257612

-- Define the conditions
def total_animals := 11
def ducks := 6
def total_legs := 32
def legs_per_duck := 2
def legs_per_dog := 4

-- Calculate intermediate values based on conditions
def duck_legs := ducks * legs_per_duck
def remaining_legs := total_legs - duck_legs

-- The proof statement
theorem num_dogs : ∃ D : ℕ, D = remaining_legs / legs_per_dog ∧ D + ducks = total_animals :=
by
  sorry

end num_dogs_l257_257612


namespace smallest_special_number_gt_3429_l257_257101

open Set

def is_special_number (n : ℕ) : Prop :=
  (fintype.card (fintype.of_finset (finset.of_digits (nat.digits 10 n)) nat.digits_dec_eq)) = 4

theorem smallest_special_number_gt_3429 :
  ∃ n : ℕ, n > 3429 ∧ is_special_number n ∧ (∀ m : ℕ, m > 3429 ∧ is_special_number m → n ≤ m) :=
exists.intro 3450 (and.intro (by decide) (and.intro (by decide) (by decide)))

end smallest_special_number_gt_3429_l257_257101


namespace find_S6_l257_257833

variable {a : ℕ → ℝ} 
variable {S : ℕ → ℝ}

/-- sum_of_first_n_terms_of_geometric_sequence -/
def sum_of_first_n_terms_of_geometric_sequence (S : ℕ → ℝ) : Prop :=
  ∃ a1 r, ∀ n, S n = a1 * (1 - r^(n+1)) / (1 - r)

-- Given conditions
axiom geom_seq_positive_terms : ∀ n, a n > 0
axiom sum_S2 : S 2 = 3
axiom sum_S4 : S 4 = 15

theorem find_S6 : S 6 = 63 := by
  sorry

end find_S6_l257_257833


namespace simplify_expression_l257_257181

theorem simplify_expression (z : ℝ) : (3 - 5*z^2) - (4 + 3*z^2) = -1 - 8*z^2 :=
by
  sorry

end simplify_expression_l257_257181


namespace min_distance_circle_to_line_l257_257476

noncomputable def circle : set (ℝ × ℝ) := 
  {p | (p.1 - 1)^2 + (p.2 - 1)^2 = 1}

def line := {p : ℝ × ℝ | 3 * p.1 + 4 * p.2 + 8 = 0}

def point_to_line_distance (point: ℝ × ℝ) (line: ℝ × ℝ → Prop) : ℝ :=
  abs (3 * point.1 + 4 * point.2 + 8) / real.sqrt (3^2 + 4^2)

theorem min_distance_circle_to_line : 
  (∀ (p : ℝ × ℝ), p ∈ circle → point_to_line_distance p line ≥ 2) := 
by sorry

end min_distance_circle_to_line_l257_257476


namespace prime_divisors_of_1320_l257_257138

theorem prime_divisors_of_1320 : 
  ∃ (S : Finset ℕ), (S = {2, 3, 5, 11}) ∧ S.card = 4 := 
by
  sorry

end prime_divisors_of_1320_l257_257138


namespace incorrect_conclusion_D_l257_257964

def parabola (x : ℝ) : ℝ := (x - 2) ^ 2 + 1

theorem incorrect_conclusion_D :
  ∀ x : ℝ, x < 2 → ∃ y1 y2 : ℝ, y1 = parabola x ∧ y2 = parabola (x + 1) ∧ y1 > y2 :=
by
  sorry

end incorrect_conclusion_D_l257_257964


namespace vovochka_correct_sum_cases_vovochka_min_difference_l257_257778

-- Part (a)
theorem vovochka_correct_sum_cases : 
  (∑ (a : ℕ) in finset.range 10, (∑ (b : ℕ) in finset.range (10 - a), 1)) ^ 3 = 244620 :=
sorry

-- Part (b)
theorem vovochka_min_difference : 
  ∃ (a b c x y z : ℕ), (a * 100 + b * 10 + c + x * 100 + y * 10 + z) - (a + x) * 100 - (b + y) * 10 - (c + z) = 1800 :=
sorry

end vovochka_correct_sum_cases_vovochka_min_difference_l257_257778


namespace profit_without_discount_l257_257654

theorem profit_without_discount (CP SP MP : ℝ) (discountRate profitRate : ℝ)
  (h1 : CP = 100)
  (h2 : discountRate = 0.05)
  (h3 : profitRate = 0.235)
  (h4 : SP = CP * (1 + profitRate))
  (h5 : MP = SP / (1 - discountRate)) :
  (((MP - CP) / CP) * 100) = 30 := 
sorry

end profit_without_discount_l257_257654


namespace inv_prop_x_y_l257_257000

theorem inv_prop_x_y (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : x = 4) (h3 : y = 2) (h4 : y = 10) : x = 4 / 5 :=
by
  sorry

end inv_prop_x_y_l257_257000


namespace sum_three_consecutive_divisible_by_three_l257_257614

theorem sum_three_consecutive_divisible_by_three (n : ℤ) : 3 ∣ ((n - 1) + n + (n + 1)) :=
by
  sorry  -- Proof goes here

end sum_three_consecutive_divisible_by_three_l257_257614


namespace auston_height_l257_257232

noncomputable def auston_height_in_meters (height_in_inches : ℝ) : ℝ :=
  let height_in_cm := height_in_inches * 2.54
  height_in_cm / 100

theorem auston_height : auston_height_in_meters 65 = 1.65 :=
by
  sorry

end auston_height_l257_257232


namespace smallest_special_number_gt_3429_l257_257097

-- Define what it means for a number to be special
def is_special (n : ℕ) : Prop :=
  (List.toFinset (Nat.digits 10 n)).card = 4

-- Define the problem statement in Lean
theorem smallest_special_number_gt_3429 : ∃ n : ℕ, 3429 < n ∧ is_special n ∧ ∀ m : ℕ, 3429 < m ∧ is_special m → n ≤ m := 
  by
  let smallest_n := 3450
  have hn : 3429 < smallest_n := by decide
  have hs : is_special smallest_n := by
    -- digits of 3450 are [3, 4, 5, 0], which are four different digits
    sorry 
  have minimal : ∀ m, 3429 < m ∧ is_special m → smallest_n ≤ m :=
    by
    -- This needs to show that no special number exists between 3429 and 3450
    sorry
  exact ⟨smallest_n, hn, hs, minimal⟩

end smallest_special_number_gt_3429_l257_257097


namespace probability_of_matching_pair_l257_257032
-- Import the necessary library for probability and combinatorics

def probability_matching_pair (pairs : ℕ) (total_shoes : ℕ) : ℚ :=
  if total_shoes = 2 * pairs then
    (pairs : ℚ) / ((total_shoes * (total_shoes - 1) / 2) : ℚ)
  else 0

theorem probability_of_matching_pair (pairs := 6) (total_shoes := 12) : 
  probability_matching_pair pairs total_shoes = 1 / 11 := 
by
  sorry

end probability_of_matching_pair_l257_257032


namespace line_equation_parallel_to_x_axis_through_point_l257_257186

-- Define the point (3, -2)
def point : ℝ × ℝ := (3, -2)

-- Define a predicate for a line being parallel to the X-axis
def is_parallel_to_x_axis (line : ℝ → ℝ → Prop) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, line x k

-- Define the equation of the line passing through the given point
def equation_of_line_through_point (p : ℝ × ℝ) (line : ℝ → ℝ → Prop) : Prop :=
  line p.1 p.2

-- State the theorem to be proved
theorem line_equation_parallel_to_x_axis_through_point :
  ∀ (line : ℝ → ℝ → Prop), 
    (equation_of_line_through_point point line) → (is_parallel_to_x_axis line) → (∀ x, line x (-2)) :=
by
  sorry

end line_equation_parallel_to_x_axis_through_point_l257_257186


namespace polynomial_evaluation_l257_257758

theorem polynomial_evaluation (x : ℝ) (h1 : x^2 - 4 * x - 12 = 0) (h2 : 0 < x) : x^3 - 4 * x^2 - 12 * x + 16 = 16 := 
by
  sorry

end polynomial_evaluation_l257_257758


namespace value_of_y_l257_257268

theorem value_of_y (x y : ℝ) (cond1 : 1.5 * x = 0.75 * y) (cond2 : x = 20) : y = 40 :=
by
  sorry

end value_of_y_l257_257268


namespace ratio_eq_thirteen_fifths_l257_257279

theorem ratio_eq_thirteen_fifths
  (a b c : ℝ)
  (h₁ : b / a = 4)
  (h₂ : c / b = 2) :
  (a + b + c) / (a + b) = 13 / 5 :=
sorry

end ratio_eq_thirteen_fifths_l257_257279


namespace ratio_of_vegetables_to_beef_l257_257162

variable (amountBeefInitial : ℕ) (amountBeefUnused : ℕ) (amountVegetables : ℕ)

def amount_beef_used (initial unused : ℕ) : ℕ := initial - unused
def ratio_vegetables_beef (vegetables beef : ℕ) : ℚ := vegetables / beef

theorem ratio_of_vegetables_to_beef 
  (h1 : amountBeefInitial = 4)
  (h2 : amountBeefUnused = 1)
  (h3 : amountVegetables = 6) :
  ratio_vegetables_beef amountVegetables (amount_beef_used amountBeefInitial amountBeefUnused) = 2 :=
by
  sorry

end ratio_of_vegetables_to_beef_l257_257162


namespace product_of_D_coordinates_l257_257951

theorem product_of_D_coordinates 
  (M D : ℝ × ℝ)
  (C : ℝ × ℝ)
  (hC : C = (5, 3))
  (hM : M = (3, 7))
  (h_midpoint : M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) : 
  D.1 * D.2 = 11 :=
by
  sorry

end product_of_D_coordinates_l257_257951


namespace compute_ζ7_sum_l257_257442

noncomputable def ζ_power_sum (ζ1 ζ2 ζ3 : ℂ) : Prop :=
  (ζ1 + ζ2 + ζ3 = 2) ∧
  (ζ1^2 + ζ2^2 + ζ3^2 = 6) ∧
  (ζ1^3 + ζ2^3 + ζ3^3 = 8) →
  ζ1^7 + ζ2^7 + ζ3^7 = 58

theorem compute_ζ7_sum (ζ1 ζ2 ζ3 : ℂ) (h : ζ_power_sum ζ1 ζ2 ζ3) : ζ1^7 + ζ2^7 + ζ3^7 = 58 :=
by
  -- proof goes here
  sorry

end compute_ζ7_sum_l257_257442


namespace truthfulness_count_l257_257887

-- Define variables to represent the number of warriors and their response counts
def num_warriors : Nat := 33
def yes_sword : Nat := 13
def yes_spear : Nat := 15
def yes_axe : Nat := 20
def yes_bow : Nat := 27

-- Define the total number of "Yes" answers
def total_yes_answers : Nat := yes_sword + yes_spear + yes_axe + yes_bow

theorem truthfulness_count :
  ∃ x : Nat, x + 3 * (num_warriors - x) = total_yes_answers ∧ x = 12 :=
by
  sorry

end truthfulness_count_l257_257887


namespace problem1_problem2_l257_257261

-- Definitions used directly from conditions
def inequality (m x : ℝ) : Prop := m * x ^ 2 - 2 * m * x - 1 < 0

-- Proof problem (1)
theorem problem1 (m : ℝ) (h : ∀ x : ℝ, inequality m x) : -1 < m ∧ m ≤ 0 :=
sorry

-- Proof problem (2)
theorem problem2 (x : ℝ) (h : ∀ m : ℝ, |m| ≤ 1 → inequality m x) :
  (1 - Real.sqrt 2 < x ∧ x < 1) ∨ (1 < x ∧ x < 1 + Real.sqrt 2) :=
sorry

end problem1_problem2_l257_257261


namespace combined_list_correct_l257_257161

def james_friends : ℕ := 75
def john_friends : ℕ := 3 * james_friends
def shared_friends : ℕ := 25
def combined_list : ℕ := james_friends + john_friends - shared_friends

theorem combined_list_correct :
  combined_list = 275 :=
by
  sorry

end combined_list_correct_l257_257161


namespace parallel_line_plane_no_common_points_l257_257954

noncomputable def line := Type
noncomputable def plane := Type

variable {l : line}
variable {α : plane}

-- Definitions for parallel lines and planes, and relations between lines and planes
def parallel_to_plane (l : line) (α : plane) : Prop := sorry -- Definition of line parallel to plane
def within_plane (m : line) (α : plane) : Prop := sorry -- Definition of line within plane
def no_common_points (l m : line) : Prop := sorry -- Definition of no common points between lines

theorem parallel_line_plane_no_common_points
  (h₁ : parallel_to_plane l α)
  (l2 : line)
  (h₂ : within_plane l2 α) :
  no_common_points l l2 :=
sorry

end parallel_line_plane_no_common_points_l257_257954


namespace total_chairs_l257_257214

def numIndoorTables := 9
def numOutdoorTables := 11
def chairsPerIndoorTable := 10
def chairsPerOutdoorTable := 3

theorem total_chairs :
  numIndoorTables * chairsPerIndoorTable + numOutdoorTables * chairsPerOutdoorTable = 123 :=
by
  sorry

end total_chairs_l257_257214


namespace son_age_is_eight_l257_257515

theorem son_age_is_eight (F S : ℕ) (h1 : F + 6 + S + 6 = 68) (h2 : F = 6 * S) : S = 8 :=
by
  sorry

end son_age_is_eight_l257_257515


namespace angle_measure_l257_257409

theorem angle_measure (α : ℝ) 
  (h1 : 90 - α + (180 - α) = 180) : 
  α = 45 := 
by 
  sorry

end angle_measure_l257_257409


namespace value_of_y_l257_257273

theorem value_of_y (x y : ℝ) (h₁ : 1.5 * x = 0.75 * y) (h₂ : x = 20) : y = 40 :=
sorry

end value_of_y_l257_257273


namespace wendy_packages_chocolates_l257_257022

variable (packages_per_5min : Nat := 2)
variable (dozen_size : Nat := 12)
variable (minutes_in_hour : Nat := 60)
variable (hours : Nat := 4)

theorem wendy_packages_chocolates (h1 : packages_per_5min = 2) 
                                 (h2 : dozen_size = 12) 
                                 (h3 : minutes_in_hour = 60) 
                                 (h4 : hours = 4) : 
    let chocolates_per_5min := packages_per_5min * dozen_size
    let intervals_per_hour := minutes_in_hour / 5
    let chocolates_per_hour := chocolates_per_5min * intervals_per_hour
    let chocolates_in_4hours := chocolates_per_hour * hours
    chocolates_in_4hours = 1152 := 
by
  let chocolates_per_5min := packages_per_5min * dozen_size
  let intervals_per_hour := minutes_in_hour / 5
  let chocolates_per_hour := chocolates_per_5min * intervals_per_hour
  let chocolates_in_4hours := chocolates_per_hour * hours
  sorry

end wendy_packages_chocolates_l257_257022


namespace arccos_of_one_over_sqrt_two_l257_257725

theorem arccos_of_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
sorry

end arccos_of_one_over_sqrt_two_l257_257725


namespace number_of_truthful_warriors_l257_257880

theorem number_of_truthful_warriors (total_warriors : ℕ) 
  (sword_yes : ℕ) (spear_yes : ℕ) (axe_yes : ℕ) (bow_yes : ℕ) 
  (always_tells_truth : ℕ → Prop)
  (always_lies : ℕ → Prop)
  (hv1 : total_warriors = 33)
  (hv2 : sword_yes = 13)
  (hv3 : spear_yes = 15)
  (hv4 : axe_yes = 20)
  (hv5 : bow_yes = 27) :
  ∃ truthful_warriors, truthful_warriors = 12 := 
by {
  sorry
}

end number_of_truthful_warriors_l257_257880


namespace magnitude_of_T_l257_257575

open Complex

noncomputable def i : ℂ := Complex.I

noncomputable def T : ℂ := (1 + i)^19 - (1 - i)^19

theorem magnitude_of_T : Complex.abs T = 1024 := by
  sorry

end magnitude_of_T_l257_257575


namespace smallest_special_number_gt_3429_l257_257102

open Set

def is_special_number (n : ℕ) : Prop :=
  (fintype.card (fintype.of_finset (finset.of_digits (nat.digits 10 n)) nat.digits_dec_eq)) = 4

theorem smallest_special_number_gt_3429 :
  ∃ n : ℕ, n > 3429 ∧ is_special_number n ∧ (∀ m : ℕ, m > 3429 ∧ is_special_number m → n ≤ m) :=
exists.intro 3450 (and.intro (by decide) (and.intro (by decide) (by decide)))

end smallest_special_number_gt_3429_l257_257102


namespace find_principal_l257_257222

theorem find_principal (R : ℝ) (T : ℝ) (I : ℝ) (hR : R = 0.12) (hT : T = 1) (hI : I = 1500) :
  ∃ P : ℝ, I = P * R * T ∧ P = 12500 := 
by
  use 12500
  rw [hR, hT, hI]
  norm_num
  sorry

end find_principal_l257_257222


namespace number_of_decks_bought_l257_257615

theorem number_of_decks_bought :
  ∃ T : ℕ, (8 * T + 5 * 8 = 64) ∧ T = 3 :=
by
  sorry

end number_of_decks_bought_l257_257615


namespace arccos_of_one_over_sqrt_two_l257_257719

theorem arccos_of_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
sorry

end arccos_of_one_over_sqrt_two_l257_257719


namespace searchlight_probability_l257_257857

theorem searchlight_probability (revolutions_per_minute : ℕ) (D : ℝ) (prob : ℝ)
  (h1 : revolutions_per_minute = 4)
  (h2 : prob = 0.6666666666666667) :
  D = (2 / 3) * (60 / revolutions_per_minute) :=
by
  -- To complete the proof, we will use the conditions given.
  sorry

end searchlight_probability_l257_257857


namespace sin_alpha_cos_2beta_l257_257975

theorem sin_alpha_cos_2beta :
  ∀ α β : ℝ, 3 * Real.sin α - Real.sin β = Real.sqrt 10 ∧ α + β = Real.pi / 2 →
  Real.sin α = 3 * Real.sqrt 10 / 10 ∧ Real.cos (2 * β) = 4 / 5 :=
by
  intros α β h
  sorry

end sin_alpha_cos_2beta_l257_257975


namespace arccos_sqrt2_l257_257687

def arccos_eq (x : ℝ) := arccos x
def range_arccos := ∀ (x : ℝ), 0 ≤ arccos x ∧ arccos x ≤ π
def cos_pi_div_four : Real.cos (π / 4) = 1 / Real.sqrt 2 := by sorry

theorem arccos_sqrt2 :
  arccos_eq (1 / Real.sqrt 2) = π / 4 :=
by
  have h1 : Real.cos (π / 4) = 1 / Real.sqrt 2, from cos_pi_div_four
  sorry

end arccos_sqrt2_l257_257687


namespace total_chairs_l257_257213

def numIndoorTables := 9
def numOutdoorTables := 11
def chairsPerIndoorTable := 10
def chairsPerOutdoorTable := 3

theorem total_chairs :
  numIndoorTables * chairsPerIndoorTable + numOutdoorTables * chairsPerOutdoorTable = 123 :=
by
  sorry

end total_chairs_l257_257213


namespace arccos_one_over_sqrt_two_l257_257747

theorem arccos_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l257_257747


namespace arccos_sqrt_half_l257_257711

theorem arccos_sqrt_half : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := by
  sorry

end arccos_sqrt_half_l257_257711


namespace range_of_m_l257_257287

variable (a b : ℝ)

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 1, x^3 - m ≤ a * x + b ∧ a * x + b ≤ x^3 + m) ↔ m ∈ Set.Ici (Real.sqrt 3 / 9) :=
by
  sorry

end range_of_m_l257_257287


namespace non_equilateral_triangle_combinations_l257_257205

theorem non_equilateral_triangle_combinations :
  ∀ (n : ℕ) (h : n = 6), 
  let total_combinations := nat.choose n 3 in
  let equilateral_combinations := 2 in
  total_combinations - equilateral_combinations = 18 :=
begin
  intros n h,
  have H1 : n.choose 3 = 20, by {
    rw h,
    exact nat.choose_eq_factorial_div_factorial (nat.choose_pos _ _) dec_trivial,
  },
  have H2 : 20 - 2 = 18, by {
    norm_num,
  },
  rw H1,
  exact H2,
end

end non_equilateral_triangle_combinations_l257_257205


namespace geometric_common_ratio_eq_three_l257_257157

theorem geometric_common_ratio_eq_three 
  (a : ℕ → ℤ) 
  (d : ℤ) 
  (h_arithmetic_seq : ∀ n, a (n + 1) = a n + d)
  (h_nonzero_d : d ≠ 0) 
  (h_geom_seq : (a 2 + 2 * d) ^ 2 = (a 2 + d) * (a 2 + 5 * d)) : 
  (a 3) / (a 2) = 3 :=
by 
  sorry

end geometric_common_ratio_eq_three_l257_257157


namespace opposite_of_neg_2_l257_257829

theorem opposite_of_neg_2 : ∃ y : ℝ, -2 + y = 0 ∧ y = 2 := by
  sorry

end opposite_of_neg_2_l257_257829


namespace profit_condition_maximize_profit_l257_257506

noncomputable def profit (x : ℕ) : ℕ := 
  (x + 10) * (300 - 10 * x)

theorem profit_condition (x : ℕ) : profit x = 3360 ↔ x = 2 ∨ x = 18 := by
  sorry

theorem maximize_profit : ∃ x, x = 10 ∧ profit x = 4000 := by
  sorry

end profit_condition_maximize_profit_l257_257506


namespace son_daughter_eggs_per_morning_l257_257589

-- Define the given conditions in Lean 4
def trays_per_week : Nat := 2
def eggs_per_tray : Nat := 24
def eggs_per_night_rhea_husband : Nat := 4
def nights_per_week : Nat := 7
def uneaten_eggs_per_week : Nat := 6

-- Define the total eggs bought per week
def total_eggs_per_week : Nat := trays_per_week * eggs_per_tray

-- Define the eggs eaten per week by Rhea and her husband
def eggs_eaten_per_week_rhea_husband : Nat := eggs_per_night_rhea_husband * nights_per_week

-- Prove the number of eggs eaten by son and daughter every morning
theorem son_daughter_eggs_per_morning :
  (total_eggs_per_week - eggs_eaten_per_week_rhea_husband - uneaten_eggs_per_week) = 14 :=
sorry

end son_daughter_eggs_per_morning_l257_257589


namespace quadratic_solution_eq_l257_257169

noncomputable def p : ℝ :=
  (8 + Real.sqrt 364) / 10

noncomputable def q : ℝ :=
  (8 - Real.sqrt 364) / 10

theorem quadratic_solution_eq (p q : ℝ) (h₁ : 5 * p^2 - 8 * p - 15 = 0) (h₂ : 5 * q^2 - 8 * q - 15 = 0) : 
  (p - q) ^ 2 = 14.5924 :=
sorry

end quadratic_solution_eq_l257_257169


namespace possible_value_of_n_l257_257139

theorem possible_value_of_n :
  ∃ (n : ℕ), (345564 - n) % (13 * 17 * 19) = 0 ∧ 0 < n ∧ n < 1000 ∧ n = 98 :=
sorry

end possible_value_of_n_l257_257139


namespace num_green_balls_l257_257212

theorem num_green_balls (G : ℕ) (h : (3 * 2 : ℚ) / ((5 + G) * (4 + G)) = 1/12) : G = 4 :=
by
  sorry

end num_green_balls_l257_257212


namespace fractions_equal_l257_257295

theorem fractions_equal (a b c d : ℚ) (h1 : a = 2/7) (h2 : b = 3) (h3 : c = 3/7) (h4 : d = 2) :
  a * b = c * d := 
sorry

end fractions_equal_l257_257295


namespace G_is_odd_l257_257406

noncomputable def G (F : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ :=
  F x * (1 / (a^x - 1) + 1 / 2)

theorem G_is_odd (F : ℝ → ℝ) (a : ℝ) (h : a > 0) (h₁ : a ≠ 1) (h₂ : ∀ x : ℝ, F (-x) = - F x) :
  ∀ x : ℝ, G F a (-x) = - G F a x :=
by 
  sorry

end G_is_odd_l257_257406


namespace find_k_all_reals_l257_257393

theorem find_k_all_reals (a b c : ℝ) : 
  (a + b) * (b + c) * (c + a) = (a + b + c) * (a * b + b * c + c * a) - a * b * c :=
sorry

end find_k_all_reals_l257_257393


namespace foreign_students_next_semester_l257_257520

theorem foreign_students_next_semester (total_students : ℕ) (percent_foreign : ℝ) (new_foreign_students : ℕ) 
  (h_total : total_students = 1800) (h_percent : percent_foreign = 0.30) (h_new : new_foreign_students = 200) : 
  (0.30 * 1800 + 200 : ℝ) = 740 := by
  sorry

end foreign_students_next_semester_l257_257520


namespace find_exponent_M_l257_257028

theorem find_exponent_M (M : ℕ) : (32^4) * (4^6) = 2^M → M = 32 := by
  sorry

end find_exponent_M_l257_257028


namespace variance_transformed_data_l257_257948

variables {X : Type*} [fintype X] {f : X → ℝ}

-- Assume the variance of the original data
noncomputable def DX : ℝ := (∑ x, (f x)^2) / (fintype.card X) - ((∑ x, f x) / (fintype.card X))^2

-- Assume the variance is given as 1/2
axiom var_f : DX = 1 / 2

-- Consider the transformed data
noncomputable def g (x : X) : ℝ := 2 * f x - 5

-- Define the variance of the transformed data
noncomputable def DY : ℝ := (∑ x, (g x)^2) / (fintype.card X) - ((∑ x, g x) / (fintype.card X))^2

-- State the theorem to prove
theorem variance_transformed_data : DY = 2 :=
by
  sorry

end variance_transformed_data_l257_257948


namespace unique_prime_sum_diff_l257_257245

theorem unique_prime_sum_diff :
  ∀ p : ℕ, Prime p ∧ (∃ p1 p2 p3 : ℕ, Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ (p = p1 + 2) ∧ (p = p3 - 2)) → p = 5 :=
by
  sorry

end unique_prime_sum_diff_l257_257245


namespace arccos_one_over_sqrt_two_eq_pi_four_l257_257740

theorem arccos_one_over_sqrt_two_eq_pi_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_four_l257_257740


namespace equilateral_triangle_distances_l257_257585

-- Defining the necessary conditions
variables {h x y z : ℝ}
variables (hx : 0 < h) (hx_cond : x + y + z = h)
variables (triangle_ineqs : x + y > z ∧ y + z > x ∧ z + x > y)

-- Lean 4 statement to express the proof problem
theorem equilateral_triangle_distances (hx : 0 < h) (hx_cond : x + y + z = h) (triangle_ineqs : x + y > z ∧ y + z > x ∧ z + x > y) : 
  x < h / 2 ∧ y < h / 2 ∧ z < h / 2 :=
sorry

end equilateral_triangle_distances_l257_257585


namespace robot_possible_path_lengths_l257_257047

theorem robot_possible_path_lengths (n : ℕ) (valid_path: ∀ (i : ℕ), i < n → (i % 4 = 0 ∨ i % 4 = 1 ∨ i % 4 = 2 ∨ i % 4 = 3)) :
  (n % 4 = 0) :=
by
  sorry

end robot_possible_path_lengths_l257_257047


namespace at_most_one_zero_l257_257571

-- Definition of the polynomial f(x)
def f (n : ℤ) (x : ℝ) : ℝ :=
  x^4 - 1994 * x^3 + (1993 + n) * x^2 - 11 * x + n

-- The target theorem statement
theorem at_most_one_zero (n : ℤ) : ∃! x : ℝ, f n x = 0 :=
by
  sorry

end at_most_one_zero_l257_257571


namespace arccos_sqrt2_l257_257691

def arccos_eq (x : ℝ) := arccos x
def range_arccos := ∀ (x : ℝ), 0 ≤ arccos x ∧ arccos x ≤ π
def cos_pi_div_four : Real.cos (π / 4) = 1 / Real.sqrt 2 := by sorry

theorem arccos_sqrt2 :
  arccos_eq (1 / Real.sqrt 2) = π / 4 :=
by
  have h1 : Real.cos (π / 4) = 1 / Real.sqrt 2, from cos_pi_div_four
  sorry

end arccos_sqrt2_l257_257691


namespace number_of_truthful_warriors_l257_257884

theorem number_of_truthful_warriors (total_warriors : ℕ) 
  (sword_yes : ℕ) (spear_yes : ℕ) (axe_yes : ℕ) (bow_yes : ℕ) 
  (always_tells_truth : ℕ → Prop)
  (always_lies : ℕ → Prop)
  (hv1 : total_warriors = 33)
  (hv2 : sword_yes = 13)
  (hv3 : spear_yes = 15)
  (hv4 : axe_yes = 20)
  (hv5 : bow_yes = 27) :
  ∃ truthful_warriors, truthful_warriors = 12 := 
by {
  sorry
}

end number_of_truthful_warriors_l257_257884


namespace convert_246_octal_to_decimal_l257_257072

theorem convert_246_octal_to_decimal : 2 * (8^2) + 4 * (8^1) + 6 * (8^0) = 166 := 
by
  -- We skip the proof part as it is not required in the task
  sorry

end convert_246_octal_to_decimal_l257_257072


namespace rational_sum_abs_ratios_l257_257538

theorem rational_sum_abs_ratios (a b c : ℚ) (h : |a * b * c| / (a * b * c) = 1) : (|a| / a + |b| / b + |c| / c = 3) ∨ (|a| / a + |b| / b + |c| / c = -1) := 
sorry

end rational_sum_abs_ratios_l257_257538


namespace prime_factors_of_x_l257_257947

theorem prime_factors_of_x (n : ℕ) (h1 : 2^n - 32 = x) (h2 : (nat.prime_factors x).length = 3) (h3 : 3 ∈ nat.prime_factors x) :
  x = 480 ∨ x = 2016 :=
sorry

end prime_factors_of_x_l257_257947


namespace major_premise_is_wrong_l257_257651

-- Definitions of the conditions
def line_parallel_to_plane (l : Type) (p : Type) : Prop := sorry
def line_contained_in_plane (l : Type) (p : Type) : Prop := sorry

-- Stating the main problem: the major premise is wrong
theorem major_premise_is_wrong :
  ∀ (a b : Type) (α : Type), line_contained_in_plane a α → line_parallel_to_plane b α → ¬ (line_parallel_to_plane b a) := 
by 
  intros a b α h1 h2
  sorry

end major_premise_is_wrong_l257_257651


namespace number_of_truth_tellers_is_twelve_l257_257907
noncomputable theory

section
variables (x : ℕ)
variables (y : ℕ)
variables (a b c d : ℕ)

-- Given conditions
def total_warriors : ℕ := 33
def yes_sword : ℕ := 13
def yes_spear : ℕ := 15
def yes_axe : ℕ := 20
def yes_bow : ℕ := 27

-- Calculate total yes answers (condition 6)
def total_yes_answers : ℕ := yes_sword + yes_spear + yes_axe + yes_bow

-- Truth-tellers say "yes" to only one question, liers say "yes" to three questions
def truth_yes : ℕ := x * 1
def lie_yes : ℕ := (total_warriors - x) * 3

theorem number_of_truth_tellers_is_twelve
  (h1 : total_warriors = 33)
  (h2 : yes_sword = 13)
  (h3 : yes_spear = 15)
  (h4 : yes_axe = 20)
  (h5 : yes_bow = 27)
  (h6 : total_yes_answers = 75)
  (h7 : total_yes_answers = truth_yes + lie_yes) :
    x = 12 :=
  by sorry

end

end number_of_truth_tellers_is_twelve_l257_257907


namespace partner_q_investment_time_l257_257330

theorem partner_q_investment_time 
  (P Q R : ℝ)
  (Profit_p Profit_q Profit_r : ℝ)
  (Tp Tq Tr : ℝ)
  (h1 : P / Q = 7 / 5)
  (h2 : Q / R = 5 / 3)
  (h3 : Profit_p / Profit_q = 7 / 14)
  (h4 : Profit_q / Profit_r = 14 / 9)
  (h5 : Tp = 5)
  (h6 : Tr = 9) :
  Tq = 14 :=
by
  sorry

end partner_q_investment_time_l257_257330


namespace xy_zero_l257_257618

theorem xy_zero (x y : ℝ) (h1 : x + y = 4) (h2 : x^3 - y^3 = 64) : x * y = 0 := by
  sorry

end xy_zero_l257_257618


namespace sum_of_integers_is_18_l257_257007

theorem sum_of_integers_is_18 (a b : ℕ) (h1 : b = 2 * a) (h2 : a * b + a + b = 156) (h3 : Nat.gcd a b = 1) (h4 : a < 25) : a + b = 18 :=
by
  sorry

end sum_of_integers_is_18_l257_257007


namespace find_values_l257_257396

def isInInterval (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 2

theorem find_values 
  (a b c d e : ℝ)
  (ha : isInInterval a) 
  (hb : isInInterval b) 
  (hc : isInInterval c) 
  (hd : isInInterval d)
  (he : isInInterval e)
  (h1 : a + b + c + d + e = 0)
  (h2 : a^3 + b^3 + c^3 + d^3 + e^3 = 0)
  (h3 : a^5 + b^5 + c^5 + d^5 + e^5 = 10) : 
  (a = 2 ∧ b = (Real.sqrt 5 - 1) / 2 ∧ c = (Real.sqrt 5 - 1) / 2 ∧ d = - (1 + Real.sqrt 5) / 2 ∧ e = - (1 + Real.sqrt 5) / 2) ∨
  (a = (Real.sqrt 5 - 1) / 2 ∧ b = 2 ∧ c = (Real.sqrt 5 - 1) / 2 ∧ d = - (1 + Real.sqrt 5) / 2 ∧ e = - (1 + Real.sqrt 5) / 2) ∨
  (a = (Real.sqrt 5 - 1) / 2 ∧ b = (Real.sqrt 5 - 1) / 2 ∧ c = 2 ∧ d = - (1 + Real.sqrt 5) / 2 ∧ e = - (1 + Real.sqrt 5) / 2) ∨
  (a = (Real.sqrt 5 - 1) / 2 ∧ b = (Real.sqrt 5 - 1) / 2 ∧ c = - (1 + Real.sqrt 5) / 2 ∧ d = 2 ∧ e = - (1 + Real.sqrt 5) / 2) ∨
  (a = (Real.sqrt 5 - 1) / 2 ∧ b = (Real.sqrt 5 - 1) / 2 ∧ c = - (1 + Real.sqrt 5) / 2 ∧ d = - (1 + Real.sqrt 5) / 2 ∧ e = 2) :=
sorry

end find_values_l257_257396


namespace base8_to_base10_conversion_l257_257060

def base8_to_base10 (n : Nat) : Nat := 
  match n with
  | 246 => 2 * 8^2 + 4 * 8^1 + 6 * 8^0
  | _ => 0  -- We define this only for the number 246_8

theorem base8_to_base10_conversion : base8_to_base10 246 = 166 := by 
  sorry

end base8_to_base10_conversion_l257_257060


namespace parabolas_intersect_diff_l257_257006

theorem parabolas_intersect_diff (a b c d : ℝ) (h1 : c ≥ a)
  (h2 : b = 3 * a^2 - 6 * a + 3)
  (h3 : d = 3 * c^2 - 6 * c + 3)
  (h4 : b = -2 * a^2 - 4 * a + 6)
  (h5 : d = -2 * c^2 - 4 * c + 6) :
  c - a = 1.6 :=
sorry

end parabolas_intersect_diff_l257_257006


namespace max_value_l257_257258

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def is_increasing (f : ℝ → ℝ) := ∀ {a b}, a < b → f a < f b

theorem max_value (f : ℝ → ℝ) (x y : ℝ)
  (h_odd : is_odd f)
  (h_increasing : is_increasing f)
  (h_eq : f (x^2 - 2 * x) + f y = 0) :
  2 * x + y ≤ 4 :=
sorry

end max_value_l257_257258


namespace dot_product_is_five_l257_257534

-- Define the vectors a and b
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-1, 3)

-- Define the condition that involves a and b
def condition : Prop := 2 • a - b = (3, 1)

-- Prove that the dot product of a and b equals 5 given the condition
theorem dot_product_is_five : condition → (a.1 * b.1 + a.2 * b.2) = 5 :=
by
  sorry

end dot_product_is_five_l257_257534


namespace number_of_truth_tellers_is_twelve_l257_257903
noncomputable theory

section
variables (x : ℕ)
variables (y : ℕ)
variables (a b c d : ℕ)

-- Given conditions
def total_warriors : ℕ := 33
def yes_sword : ℕ := 13
def yes_spear : ℕ := 15
def yes_axe : ℕ := 20
def yes_bow : ℕ := 27

-- Calculate total yes answers (condition 6)
def total_yes_answers : ℕ := yes_sword + yes_spear + yes_axe + yes_bow

-- Truth-tellers say "yes" to only one question, liers say "yes" to three questions
def truth_yes : ℕ := x * 1
def lie_yes : ℕ := (total_warriors - x) * 3

theorem number_of_truth_tellers_is_twelve
  (h1 : total_warriors = 33)
  (h2 : yes_sword = 13)
  (h3 : yes_spear = 15)
  (h4 : yes_axe = 20)
  (h5 : yes_bow = 27)
  (h6 : total_yes_answers = 75)
  (h7 : total_yes_answers = truth_yes + lie_yes) :
    x = 12 :=
  by sorry

end

end number_of_truth_tellers_is_twelve_l257_257903


namespace distribute_dogs_l257_257053

theorem distribute_dogs :
  ∃ (comb : ℕ), comb = (Nat.choose 11 3) * (Nat.choose 7 4) ∧ comb = 5775 :=
by
  use (Nat.choose 11 3) * (Nat.choose 7 4)
  split
  <;> sorry

end distribute_dogs_l257_257053


namespace mouse_seed_hiding_l257_257290

theorem mouse_seed_hiding : 
  ∀ (h_m h_r x : ℕ), 
  4 * h_m = x →
  7 * h_r = x →
  h_m = h_r + 3 →
  x = 28 :=
by
  intros h_m h_r x H1 H2 H3
  sorry

end mouse_seed_hiding_l257_257290


namespace perfect_square_trinomial_k_l257_257470

theorem perfect_square_trinomial_k (k : ℤ) :
  (∃ m : ℤ, 49 * m^2 + k * m + 1 = (7 * m + 1)^2) ∨
  (∃ m : ℤ, 49 * m^2 + k * m + 1 = (7 * m - 1)^2) ↔
  k = 14 ∨ k = -14 :=
sorry

end perfect_square_trinomial_k_l257_257470


namespace algebraic_expression_opposite_l257_257365

theorem algebraic_expression_opposite (a b x : ℝ) (h : b^2 * x^2 + |a| = -(b^2 * x^2 + |a|)) : a * b = 0 :=
by 
  sorry

end algebraic_expression_opposite_l257_257365


namespace min_and_max_f_l257_257003

noncomputable def f (x : ℝ) : ℝ := -2 * x + 1

theorem min_and_max_f :
  (∀ x, 0 ≤ x ∧ x ≤ 5 → f x ≥ -9) ∧ (∀ x, 0 ≤ x ∧ x ≤ 5 → f x ≤ 1) :=
by
  sorry

end min_and_max_f_l257_257003


namespace milk_leftover_l257_257385

variable {v : ℕ} -- 'v' is the number of sets of milkshakes in the 2:1 ratio.
variables {milk vanilla_chocolate : ℕ} -- spoon amounts per milkshake types
variables {total_milk total_vanilla_ice_cream total_chocolate_ice_cream : ℕ} -- total amount constraints
variables {milk_left : ℕ} -- amount of milk left after

-- Definitions based on the conditions
def milk_per_vanilla := 4
def milk_per_chocolate := 5
def ice_vanilla_per_milkshake := 12
def ice_chocolate_per_milkshake := 10
def initial_milk := 72
def initial_vanilla_ice_cream := 96
def initial_chocolate_ice_cream := 96

-- Constraints
def max_milkshakes := 16
def milk_needed (v : ℕ) := (4 * 2 * v) + (5 * v)
def vanilla_needed (v : ℕ) := 12 * 2 * v
def chocolate_needed (v : ℕ) := 10 * v 

-- Inequalities
lemma milk_constraint (v : ℕ) : milk_needed v ≤ initial_milk := sorry

lemma vanilla_constraint (v : ℕ) : vanilla_needed v ≤ initial_vanilla_ice_cream := sorry

lemma chocolate_constraint (v : ℕ) : chocolate_needed v ≤ initial_chocolate_ice_cream := sorry

lemma total_milkshakes_constraint (v : ℕ) : 3 * v ≤ max_milkshakes := sorry

-- Conclusion
theorem milk_leftover : milk_left = initial_milk - milk_needed 5 := sorry

end milk_leftover_l257_257385


namespace largest_multiple_of_7_negation_gt_neg150_l257_257347

theorem largest_multiple_of_7_negation_gt_neg150 : 
  ∃ (k : ℤ), (k % 7 = 0 ∧ -k > -150 ∧ ∀ (m : ℤ), (m % 7 = 0 ∧ -m > -150 → m ≤ k)) :=
sorry

end largest_multiple_of_7_negation_gt_neg150_l257_257347


namespace option_d_is_deductive_l257_257841

theorem option_d_is_deductive :
  (∀ (r : ℝ), S_r = Real.pi * r^2) → (S_1 = Real.pi) :=
by
  sorry

end option_d_is_deductive_l257_257841


namespace projection_of_3_neg2_onto_v_l257_257326

noncomputable def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product (a b : ℝ × ℝ) : ℝ := (a.1 * b.1 + a.2 * b.2)
  let scalar := (dot_product u v) / (dot_product v v)
  (scalar * v.1, scalar * v.2)

def v : ℝ × ℝ := (2, -8)

theorem projection_of_3_neg2_onto_v :
  projection (3, -2) v = (11/17, -44/17) :=
by sorry

end projection_of_3_neg2_onto_v_l257_257326


namespace crossnumber_unique_solution_l257_257009

-- Definition of two-digit numbers
def two_digit_numbers (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

-- Definition of prime
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Definition of square
def is_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- The given conditions reformulated
def crossnumber_problem : Prop :=
  ∃ (one_across one_down two_down three_across : ℕ),
    two_digit_numbers one_across ∧ is_prime one_across ∧
    two_digit_numbers one_down ∧ is_square one_down ∧
    two_digit_numbers two_down ∧ is_square two_down ∧
    two_digit_numbers three_across ∧ is_square three_across ∧
    one_across = 83 ∧ one_down = 81 ∧ two_down = 16 ∧ three_across = 16

theorem crossnumber_unique_solution : crossnumber_problem :=
by
  sorry

end crossnumber_unique_solution_l257_257009


namespace four_digit_numbers_count_l257_257973

def valid_middle_digit_pairs : ℕ :=
  (list.product [2, 3, 4, 5, 6, 7, 8, 9] [2, 3, 4, 5, 6, 7, 8, 9]).count (λ p, p.1 * p.2 > 10)

theorem four_digit_numbers_count :
  let first_digit_choices := 7
  let valid_pairs := valid_middle_digit_pairs
  let last_digit_choices := 10
  ∑ x in range first_digit_choices, 
  ∑ y in range valid_pairs, 
  ∑ z in range last_digit_choices, 1 = 3990 := 
sorry

end four_digit_numbers_count_l257_257973


namespace slower_train_speed_l257_257197

-- Conditions
variables (L : ℕ) -- Length of each train (in meters)
variables (v_f : ℕ) -- Speed of the faster train (in km/hr)
variables (t : ℕ) -- Time taken by the faster train to pass the slower one (in seconds)
variables (v_s : ℕ) -- Speed of the slower train (in km/hr)

-- Assumptions based on conditions of the problem
axiom length_eq : L = 30
axiom fast_speed : v_f = 42
axiom passing_time : t = 36

-- Conversion for km/hr to m/s
def km_per_hr_to_m_per_s (v : ℕ) : ℕ := (v * 5) / 18

-- Problem statement
theorem slower_train_speed : v_s = 36 :=
by
  let rel_speed := km_per_hr_to_m_per_s (v_f - v_s)
  have rel_speed_def : rel_speed = (42 - v_s) * 5 / 18 := by sorry
  have distance : 60 = rel_speed * t := by sorry
  have equation : 60 = (42 - v_s) * 10 := by sorry
  have solve_v_s : v_s = 36 := by sorry
  exact solve_v_s

end slower_train_speed_l257_257197


namespace trig_identity_proof_l257_257978

noncomputable def check_trig_identities (α β : ℝ) : Prop :=
  3 * Real.sin α - Real.sin β = Real.sqrt 10 ∧ α + β = Real.pi / 2

theorem trig_identity_proof (α β : ℝ) (h : check_trig_identities α β) :
  Real.sin α = 3 * Real.sqrt 10 / 10 ∧ Real.cos (2 * β) = 4 / 5 := by
  sorry

end trig_identity_proof_l257_257978


namespace domino_path_count_l257_257804

-- Definitions of coordinates for A and B
def A := (0, 4)
def B := (6, 0)

-- Conditions of the movements from A to B
def movements (right down : ℕ) : List (ℕ × ℕ) :=
  List.replicate right (1, 0) ++ List.replicate down (0, 1)

-- Function to count combinations
noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.descFactorial n k / (Nat.factorial k)

-- Main theorem statement
theorem domino_path_count : 
  ∃ (n m : ℕ), n = 6 ∧ m = 4 ∧ 
  (∀ (A B : ℕ × ℕ), A = (0, 4) ∧ B = (6, 0) → 
  binomial (n + m) m = 210) :=
by
  -- Introduce integers n=6, m=4 that represent grid dimensions.
  let n := 6
  let m := 4

  -- Sessions to start the path from A to B
  intros A B hA hB

  -- Calculate the binomial coefficient for the path count
  have : binomial (n + m) m = 210, from sorry

  -- Return the result ensuring conditions are met
  existsi n, existsi m
  split, refl, split, refl, assumption

end domino_path_count_l257_257804


namespace sum_of_roots_abs_eqn_zero_l257_257313

theorem sum_of_roots_abs_eqn_zero (x : ℝ) (hx : |x|^2 - 4*|x| - 5 = 0) : (5 + (-5) = 0) :=
  sorry

end sum_of_roots_abs_eqn_zero_l257_257313


namespace find_greater_number_l257_257479

theorem find_greater_number (x y : ℕ) 
  (h1 : x + y = 40)
  (h2 : x - y = 12) : x = 26 :=
by
  sorry

end find_greater_number_l257_257479


namespace pyramid_angle_problem_l257_257997

theorem pyramid_angle_problem :
  let A := (0, 0, 0)
  let B := (Real.sqrt 3, 0, 0)
  let C := (Real.sqrt 3, 1, 0)
  let D := (0, 1, 0)
  let P := (0, 0, 2)
  let E := (0, 1/2, 1)
  let AC := (Real.sqrt 3, 1, 0)
  let PB := (Real.sqrt 3, 0, -2)
  let dot_product := (AC.1 * PB.1 + AC.2 * PB.2 + AC.3 * PB.3)
  cos_angle_AC_PB : Real := dot_product / (Real.sqrt ((AC.1) ^ 2 + (AC.2) ^ 2 + (AC.3) ^ 2) * Real.sqrt ((PB.1) ^ 2 + (PB.2) ^ 2 + (PB.3) ^ 2)) = 3 * Real.sqrt 7 / 14
  ∧ ∃ (N_x : Real) (N_z : Real), 
    let N := (N_x, 0, N_z)
    let NE := (N.1 - E.1, N.2 - E.2, N.3 - E.3)
    NE.1 * PA.1 + NE.2 * PA.2 + NE.3 * PA.3 = 0
    ∧ NE.1 * AC.1 + NE.2 * AC.2 + NE.3 * AC.3 = 0
    ∧ N_x = Real.sqrt 3 / 6
    ∧ N_z = 1
    ∧ dist_A (A.1, A.2, A.3) N = 1
    ∧ dist_P (A.1, A.2, A.3) N = Real.sqrt 3 / 6 :=
by
  sorry

end pyramid_angle_problem_l257_257997


namespace equal_expressions_l257_257050

theorem equal_expressions : (-2)^3 = -(2^3) :=
by sorry

end equal_expressions_l257_257050


namespace pythagorean_triples_l257_257492

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem pythagorean_triples :
  is_pythagorean_triple 3 4 5 ∧ is_pythagorean_triple 6 8 10 :=
by
  sorry

end pythagorean_triples_l257_257492


namespace arccos_one_over_sqrt_two_l257_257745

theorem arccos_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l257_257745


namespace remainder_when_13_add_x_div_31_eq_22_l257_257301

open BigOperators

theorem remainder_when_13_add_x_div_31_eq_22
  (x : ℕ) (hx : x > 0) (hmod : 7 * x ≡ 1 [MOD 31]) :
  (13 + x) % 31 = 22 := 
  sorry

end remainder_when_13_add_x_div_31_eq_22_l257_257301


namespace arccos_sqrt_half_l257_257713

theorem arccos_sqrt_half : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := by
  sorry

end arccos_sqrt_half_l257_257713


namespace largest_multiple_of_7_neg_greater_than_neg_150_l257_257342

theorem largest_multiple_of_7_neg_greater_than_neg_150 : 
  ∃ (k : ℤ), k % 7 = 0 ∧ -k > -150 ∧ (∀ (m : ℤ), m % 7 = 0 ∧ -m > -150 → k ≥ m) ∧ k = 147 :=
by
  sorry

end largest_multiple_of_7_neg_greater_than_neg_150_l257_257342


namespace probability_blue_is_4_over_13_l257_257643

def num_red : ℕ := 5
def num_green : ℕ := 6
def num_yellow : ℕ := 7
def num_blue : ℕ := 8
def total_jelly_beans : ℕ := num_red + num_green + num_yellow + num_blue

def probability_blue : ℚ := num_blue / total_jelly_beans

theorem probability_blue_is_4_over_13
  (h_num_red : num_red = 5)
  (h_num_green : num_green = 6)
  (h_num_yellow : num_yellow = 7)
  (h_num_blue : num_blue = 8) :
  probability_blue = 4 / 13 :=
by
  sorry

end probability_blue_is_4_over_13_l257_257643


namespace arccos_one_over_sqrt_two_l257_257708

theorem arccos_one_over_sqrt_two : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l257_257708


namespace ferris_wheel_seats_l257_257594

-- Define the total number of seats S as a variable
variables (S : ℕ)

-- Define the conditions
def seat_capacity : ℕ := 15

def broken_seats : ℕ := 10

def max_riders : ℕ := 120

-- The theorem statement
theorem ferris_wheel_seats :
  ((S - broken_seats) * seat_capacity = max_riders) → S = 18 :=
by
  sorry

end ferris_wheel_seats_l257_257594


namespace max_value_of_d_l257_257928

theorem max_value_of_d : ∀ (d e : ℕ), (∃ (n : ℕ), n = 70733 + 10^4 * d + e ∧ (∃ (k3 k11 : ℤ), n = 3 * k3 ∧ n = 11 * k11) ∧ d = e ∧ d ≤ 9) → d = 2 :=
by 
  -- Given conditions and goals:
  -- 1. The number has the form 7d7,33e which in numerical form is: n = 70733 + 10^4 * d + e
  -- 2. The number n is divisible by 3 and 11.
  -- 3. d and e are digits (0 ≤ d, e ≤ 9).
  -- 4. To maximize the value of d, ensure that the given conditions hold.
  -- Problem: Prove that the maximum value of d for which this holds is 2.
  sorry

end max_value_of_d_l257_257928


namespace subset_M_N_l257_257264

def is_element_of_M (x : ℝ) : Prop :=
  ∃ k : ℤ, x = (k * Real.pi / 4) + (Real.pi / 4)

def is_element_of_N (x : ℝ) : Prop :=
  ∃ k : ℤ, x = (k * Real.pi / 8) - (Real.pi / 4)

theorem subset_M_N : ∀ x, is_element_of_M x → is_element_of_N x :=
by
  sorry

end subset_M_N_l257_257264


namespace problem_solution_l257_257847

open Real

noncomputable def length_and_slope_MP 
    (length_MN : ℝ) 
    (slope_MN : ℝ) 
    (length_NP : ℝ) 
    (slope_NP : ℝ) 
    : (ℝ × ℝ) := sorry

theorem problem_solution :
  length_and_slope_MP 6 14 7 8 = (5.55, 25.9) :=
  sorry

end problem_solution_l257_257847


namespace Danny_more_than_Larry_l257_257164

/-- Keith scored 3 points. --/
def Keith_marks : Nat := 3

/-- Larry scored 3 times as many marks as Keith. --/
def Larry_marks : Nat := 3 * Keith_marks

/-- The total marks scored by Keith, Larry, and Danny is 26. --/
def total_marks (D : Nat) : Prop := Keith_marks + Larry_marks + D = 26

/-- Prove the number of more marks Danny scored than Larry is 5. --/
theorem Danny_more_than_Larry (D : Nat) (h : total_marks D) : D - Larry_marks = 5 :=
sorry

end Danny_more_than_Larry_l257_257164


namespace increasing_function_on_interval_l257_257657

noncomputable def f_A (x : ℝ) : ℝ := 3 - x
noncomputable def f_B (x : ℝ) : ℝ := x^2 - 3 * x
noncomputable def f_C (x : ℝ) : ℝ := - (1 / (x + 1))
noncomputable def f_D (x : ℝ) : ℝ := -|x|

theorem increasing_function_on_interval (h0 : ∀ x : ℝ, x > 0):
  (∀ x y : ℝ, 0 < x -> x < y -> f_C x < f_C y) ∧ 
  (∀ (g : ℝ → ℝ), (g ≠ f_C) → (∀ x y : ℝ, 0 < x -> x < y -> g x ≥ g y)) :=
by sorry

end increasing_function_on_interval_l257_257657


namespace find_angles_l257_257382

theorem find_angles (A B : ℝ) (h1 : A + B = 90) (h2 : A = 4 * B) : A = 72 ∧ B = 18 :=
by {
  sorry
}

end find_angles_l257_257382


namespace num_positive_terms_arithmetic_seq_l257_257537

theorem num_positive_terms_arithmetic_seq :
  (∃ k : ℕ+, (∀ n : ℕ, n ≤ k → (90 - 2 * n) > 0)) → (k = 44) :=
sorry

end num_positive_terms_arithmetic_seq_l257_257537


namespace pizzas_served_dinner_eq_6_l257_257856

-- Definitions based on the conditions
def pizzas_served_lunch : Nat := 9
def pizzas_served_today : Nat := 15

-- The theorem to prove the number of pizzas served during dinner
theorem pizzas_served_dinner_eq_6 : pizzas_served_today - pizzas_served_lunch = 6 := by
  sorry

end pizzas_served_dinner_eq_6_l257_257856


namespace sum_of_divisors_252_l257_257024

theorem sum_of_divisors_252 :
  let n := 252
  let prime_factors := [2, 2, 3, 3, 7]
  sum_of_divisors n = 728 :=
by
  sorry

end sum_of_divisors_252_l257_257024


namespace value_of_y_l257_257270

theorem value_of_y (x y : ℝ) (cond1 : 1.5 * x = 0.75 * y) (cond2 : x = 20) : y = 40 :=
by
  sorry

end value_of_y_l257_257270


namespace success_vowel_last_l257_257266

open Finset

noncomputable def count_vowel_last_arrangements : ℕ :=
  (factorial 2) * (factorial 5 / ((factorial 3) * (factorial 2)))

theorem success_vowel_last : count_vowel_last_arrangements = 20 :=
by
  sorry

end success_vowel_last_l257_257266


namespace arccos_one_over_sqrt_two_l257_257746

theorem arccos_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l257_257746


namespace students_attending_harvard_is_900_l257_257995

noncomputable def students_attending_harvard : ℕ :=
  let total_applicants := 20000
  let acceptance_rate := 0.05
  let yield_rate := 0.90
  let accepted_students := acceptance_rate * total_applicants
  let attending_students := yield_rate * accepted_students
  attending_students.to_nat

theorem students_attending_harvard_is_900 :
  students_attending_harvard = 900 :=
by
  -- proof will go here
  sorry

end students_attending_harvard_is_900_l257_257995


namespace side_length_of_S2_l257_257812

theorem side_length_of_S2 :
  ∀ (r s : ℕ), 
    (2 * r + s = 2000) → 
    (2 * r + 5 * s = 3030) → 
    s = 258 :=
by
  intros r s h1 h2
  sorry

end side_length_of_S2_l257_257812


namespace find_triple_abc_l257_257398

theorem find_triple_abc (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
    (h_sum : a + b + c = 3)
    (h2 : a^2 - a ≥ 1 - b * c)
    (h3 : b^2 - b ≥ 1 - a * c)
    (h4 : c^2 - c ≥ 1 - a * b) :
    a = 1 ∧ b = 1 ∧ c = 1 :=
by
  sorry

end find_triple_abc_l257_257398


namespace bobby_initial_pieces_l257_257055

-- Definitions based on the conditions
def pieces_eaten_1 := 17
def pieces_eaten_2 := 15
def pieces_left := 4

-- Definition based on the question and answer
def initial_pieces (pieces_eaten_1 pieces_eaten_2 pieces_left : ℕ) : ℕ :=
  pieces_eaten_1 + pieces_eaten_2 + pieces_left

-- Theorem stating the problem and the expected answer
theorem bobby_initial_pieces : 
  initial_pieces pieces_eaten_1 pieces_eaten_2 pieces_left = 36 :=
by 
  sorry

end bobby_initial_pieces_l257_257055


namespace circle_radius_l257_257595

theorem circle_radius
  (area_sector : ℝ)
  (arc_length : ℝ)
  (h_area : area_sector = 8.75)
  (h_arc : arc_length = 3.5) :
  ∃ r : ℝ, r = 5 :=
by
  let r := 5
  use r
  sorry

end circle_radius_l257_257595


namespace train_late_average_speed_l257_257513

theorem train_late_average_speed 
  (distance : ℝ) (on_time_speed : ℝ) (late_time_additional : ℝ) 
  (on_time : distance / on_time_speed = 1.75) 
  (late : distance / (on_time_speed * 2/2.5) = 2) :
  distance / 2 = 35 :=
by
  sorry

end train_late_average_speed_l257_257513


namespace smallest_n_l257_257625

theorem smallest_n (n : ℕ) (hn : 0 < n) (h : 253 * n % 15 = 989 * n % 15) : n = 15 := by
  sorry

end smallest_n_l257_257625


namespace correct_average_of_ten_numbers_l257_257034

theorem correct_average_of_ten_numbers :
  let incorrect_average := 20 
  let num_values := 10 
  let incorrect_number := 26
  let correct_number := 86 
  let incorrect_total_sum := incorrect_average * num_values
  let correct_total_sum := incorrect_total_sum - incorrect_number + correct_number 
  (correct_total_sum / num_values) = 26 := 
by
  sorry

end correct_average_of_ten_numbers_l257_257034


namespace arccos_sqrt_half_l257_257715

theorem arccos_sqrt_half : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := by
  sorry

end arccos_sqrt_half_l257_257715


namespace find_pairs_l257_257774

theorem find_pairs (a b : ℕ) :
  (1111 * a) % (11 * b) = 11 * (a - b) →
  140 ≤ (1111 * a) / (11 * b) ∧ (1111 * a) / (11 * b) ≤ 160 →
  (a, b) = (3, 2) ∨ (a, b) = (6, 4) ∨ (a, b) = (7, 5) ∨ (a, b) = (9, 6) :=
by
  sorry

end find_pairs_l257_257774


namespace ivan_total_money_l257_257434

-- Define values of the coins
def penny_value : ℝ := 0.01
def dime_value : ℝ := 0.1
def nickel_value : ℝ := 0.05
def quarter_value : ℝ := 0.25

-- Define number of each type of coin in each piggy bank
def first_piggybank_pennies := 100
def first_piggybank_dimes := 50
def first_piggybank_nickels := 20
def first_piggybank_quarters := 10

def second_piggybank_pennies := 150
def second_piggybank_dimes := 30
def second_piggybank_nickels := 40
def second_piggybank_quarters := 15

def third_piggybank_pennies := 200
def third_piggybank_dimes := 60
def third_piggybank_nickels := 10
def third_piggybank_quarters := 20

-- Calculate the total value of each piggy bank
def first_piggybank_value : ℝ :=
  (first_piggybank_pennies * penny_value) +
  (first_piggybank_dimes * dime_value) +
  (first_piggybank_nickels * nickel_value) +
  (first_piggybank_quarters * quarter_value)

def second_piggybank_value : ℝ :=
  (second_piggybank_pennies * penny_value) +
  (second_piggybank_dimes * dime_value) +
  (second_piggybank_nickels * nickel_value) +
  (second_piggybank_quarters * quarter_value)

def third_piggybank_value : ℝ :=
  (third_piggybank_pennies * penny_value) +
  (third_piggybank_dimes * dime_value) +
  (third_piggybank_nickels * nickel_value) +
  (third_piggybank_quarters * quarter_value)

-- Calculate the total amount of money Ivan has
def total_value : ℝ :=
  first_piggybank_value + second_piggybank_value + third_piggybank_value

-- The theorem to prove
theorem ivan_total_money :
  total_value = 33.25 :=
by
  sorry

end ivan_total_money_l257_257434


namespace solve_x_l257_257820

theorem solve_x (x : ℝ) (h : x^2 + 6 * x + 8 = -(x + 4) * (x + 6)) : 
  x = -4 := 
by
  sorry

end solve_x_l257_257820


namespace largest_unrepresentable_n_l257_257440

theorem largest_unrepresentable_n (a b : ℕ) (ha : 1 < a) (hb : 1 < b) : ∃ n, ¬ ∃ x y : ℕ, n = 7 * a + 5 * b ∧ n = 47 :=
  sorry

end largest_unrepresentable_n_l257_257440


namespace total_cost_backpacks_l257_257550

theorem total_cost_backpacks:
  let original_price := 20.00
  let discount := 0.20
  let monogram_cost := 12.00
  let coupon := 5.00
  let state_tax : List Real := [0.06, 0.08, 0.055, 0.0725, 0.04]
  let discounted_price := original_price * (1 - discount)
  let pre_tax_cost := discounted_price + monogram_cost
  let final_costs := state_tax.map (λ tax_rate => pre_tax_cost * (1 + tax_rate))
  let total_cost_before_coupon := final_costs.sum
  total_cost_before_coupon - coupon = 143.61 := by
    sorry

end total_cost_backpacks_l257_257550


namespace arccos_proof_l257_257685

noncomputable def arccos_identity : Prop := 
  ∃ x : ℝ, x = 1 / Real.sqrt 2 ∧ Real.arccos x = Real.pi / 4

theorem arccos_proof : arccos_identity :=
by
  sorry

end arccos_proof_l257_257685


namespace find_positive_difference_l257_257767

theorem find_positive_difference 
  (p1 p2 : ℝ × ℝ) (q1 q2 : ℝ × ℝ) 
  (h_p1 : p1 = (0, 8)) (h_p2 : p2 = (4, 0))
  (h_q1 : q1 = (0, 5)) (h_q2 : q2 = (10, 0))
  (y : ℝ) (hy : y = 20) :
  let m_p := (p2.2 - p1.2) / (p2.1 - p1.1)
  let b_p := p1.2 - m_p * p1.1
  let x_p := (y - b_p) / m_p
  let m_q := (q2.2 - q1.2) / (q2.1 - q1.1)
  let b_q := q1.2 - m_q * q1.1
  let x_q := (y - b_q) / m_q
  abs (x_p - x_q) = 24 :=
by
  sorry

end find_positive_difference_l257_257767


namespace monotonicity_of_g_l257_257474

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a ^ x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (a - 2) / (x ^ 2)

theorem monotonicity_of_g (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  (∀ x : ℝ, x > 0 → (g a x) < (g a (x + 1))) ∧ (∀ x : ℝ, x < 0 → (g a x) > (g a (x - 1))) :=
  sorry

end monotonicity_of_g_l257_257474


namespace max_digit_d_of_form_7d733e_multiple_of_33_l257_257920

theorem max_digit_d_of_form_7d733e_multiple_of_33 
  (d e : ℕ) (d_digit : d < 10) (e_digit : e < 10) 
  (multiple_of_33: ∃ k : ℕ, 7 * 10^5 + d * 10^4 + 7 * 10^3 + 33 * 10 + e = 33 * k) 
  : d ≤ 6 := 
sorry

end max_digit_d_of_form_7d733e_multiple_of_33_l257_257920


namespace vacation_cost_division_l257_257011

theorem vacation_cost_division (total_cost : ℕ) (cost_per_person3 different_cost : ℤ) (n : ℕ)
  (h1 : total_cost = 375)
  (h2 : cost_per_person3 = total_cost / 3)
  (h3 : different_cost = cost_per_person3 - 50)
  (h4 : different_cost = total_cost / n) :
  n = 5 :=
  sorry

end vacation_cost_division_l257_257011


namespace cookie_baking_l257_257566

/-- It takes 7 minutes to bake 1 pan of cookies. In 28 minutes, you can bake 4 pans of cookies. -/
theorem cookie_baking (bake_time_per_pan : ℕ) (total_time : ℕ) (num_pans : ℕ) 
  (h1 : bake_time_per_pan = 7)
  (h2 : total_time = 28) : 
  num_pans = 4 := 
by
  sorry

end cookie_baking_l257_257566


namespace bogatyrs_truthful_count_l257_257898

noncomputable def number_of_truthful_warriors (total_warriors: ℕ) (sword_yes: ℕ) (spear_yes: ℕ) (axe_yes: ℕ) (bow_yes: ℕ) : ℕ :=
  let total_yes := sword_yes + spear_yes + axe_yes + bow_yes
  let lying_warriors := (total_yes - total_warriors) / 2
  total_warriors - lying_warriors

theorem bogatyrs_truthful_count :
  number_of_truthful_warriors 33 13 15 20 27 = 12 := by
  sorry

end bogatyrs_truthful_count_l257_257898


namespace parabolas_intersect_at_points_l257_257058

theorem parabolas_intersect_at_points :
  ∀ (x y : ℝ), (y = 3 * x^2 - 12 * x - 9) ↔ (y = 2 * x^2 - 8 * x + 5) →
  (x, y) = (2 + 3 * Real.sqrt 2, 66 - 36 * Real.sqrt 2) ∨ (x, y) = (2 - 3 * Real.sqrt 2, 66 + 36 * Real.sqrt 2) :=
by
  sorry

end parabolas_intersect_at_points_l257_257058


namespace composite_integer_expression_l257_257220

theorem composite_integer_expression (n : ℕ) (h : n > 1) (hn : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b) :
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ n = x * y + x * z + y * z + 1 :=
by
  sorry

end composite_integer_expression_l257_257220


namespace arccos_one_over_sqrt_two_l257_257704

theorem arccos_one_over_sqrt_two : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l257_257704


namespace fifth_roll_six_probability_l257_257670
noncomputable def probability_fifth_roll_six : ℚ := sorry

theorem fifth_roll_six_probability :
  let fair_die_prob : ℚ := (1/6)^4
  let biased_die_6_prob : ℚ := (2/3)^3 * (1/15)
  let biased_die_3_prob : ℚ := (1/10)^3 * (1/2)
  let total_prob := (1/3) * fair_die_prob + (1/3) * biased_die_6_prob + (1/3) * biased_die_3_prob
  let normalized_biased_6_prob := (1/3) * biased_die_6_prob / total_prob
  let prob_of_fifth_six := normalized_biased_6_prob * (2/3)
  probability_fifth_roll_six = prob_of_fifth_six :=
sorry

end fifth_roll_six_probability_l257_257670


namespace arccos_proof_l257_257684

noncomputable def arccos_identity : Prop := 
  ∃ x : ℝ, x = 1 / Real.sqrt 2 ∧ Real.arccos x = Real.pi / 4

theorem arccos_proof : arccos_identity :=
by
  sorry

end arccos_proof_l257_257684


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l257_257673

theorem arccos_one_over_sqrt_two_eq_pi_over_four : 
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l257_257673


namespace find_abc_integers_l257_257929

theorem find_abc_integers (a b c : ℕ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) 
(h4 : (a - 1) * (b - 1) * (c - 1) ∣ a * b * c - 1) : (a = 3 ∧ b = 5 ∧ c = 15) ∨ 
(a = 2 ∧ b = 4 ∧ c = 8) :=
sorry

end find_abc_integers_l257_257929


namespace smallest_special_greater_than_3429_l257_257090

def is_special (n : ℕ) : Prop := (nat.digits 10 n).nodup ∧ (nat.digits 10 n).length = 4

theorem smallest_special_greater_than_3429 : ∃ n, n > 3429 ∧ is_special n ∧ 
  ∀ m, m > 3429 ∧ is_special m → n ≤ m :=
by
  use 3450
  sorry

end smallest_special_greater_than_3429_l257_257090


namespace debby_remaining_pictures_l257_257038

variable (zoo_pictures : ℕ) (museum_pictures : ℕ) (deleted_pictures : ℕ)

def initial_pictures (zoo_pictures museum_pictures : ℕ) : ℕ :=
  zoo_pictures + museum_pictures

def remaining_pictures (zoo_pictures museum_pictures deleted_pictures : ℕ) : ℕ :=
  (initial_pictures zoo_pictures museum_pictures) - deleted_pictures

theorem debby_remaining_pictures :
  remaining_pictures 24 12 14 = 22 :=
by
  sorry

end debby_remaining_pictures_l257_257038


namespace sequence_general_formula_l257_257443

open Nat

noncomputable def seq (a : ℕ → ℝ) : Prop :=
∀ (n : ℕ), n > 0 → (n+1) * (a (n + 1))^2 - n * (a n)^2 + (a (n + 1)) * (a n) = 0

theorem sequence_general_formula :
  ∃ (a : ℕ → ℝ), seq a ∧ (a 1 = 1) ∧ (∀ (n : ℕ), n > 0 → a n = 1 / n) :=
by
  sorry

end sequence_general_formula_l257_257443


namespace range_of_a_l257_257967

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 2 * x^2 + (a - 1) * x + 1/2 > 0) ↔ (-1 < a ∧ a < 3) := 
sorry

end range_of_a_l257_257967


namespace max_integer_k_l257_257407

-- First, define the sequence a_n
def a (n : ℕ) : ℕ := n + 5

-- Define the sequence b_n given the recurrence relation and initial condition
def b (n : ℕ) : ℕ := 3 * n + 2

-- Define the sequence c_n
def c (n : ℕ) : ℚ := 3 / ((2 * a n - 11) * (2 * b n - 1))

-- Define the sum T_n of the first n terms of the sequence c_n
def T (n : ℕ) : ℚ := (1 / 2) * (1 - (1 / (2 * n + 1)))

-- The theorem to prove
theorem max_integer_k :
  ∃ k : ℕ, ∀ n : ℕ, n > 0 → T n > (k : ℚ) / 57 ∧ k = 18 :=
by
  sorry

end max_integer_k_l257_257407


namespace hexagon_side_lengths_l257_257874

theorem hexagon_side_lengths (n : ℕ) (h1 : n ≥ 0) (h2 : n ≤ 6) (h3 : 10 * n + 8 * (6 - n) = 56) : n = 4 :=
sorry

end hexagon_side_lengths_l257_257874


namespace proof_problem_l257_257117

variable {a b c : ℝ}

theorem proof_problem (h1 : ∀ x : ℝ, 4 * x^2 - 3 * x + 1 = a * (x - 1)^2 + b * (x - 1) + c) : 
  (4 * a + 2 * b + c = 28) := by
  -- The proof goes here. The goal statement is what we need.
  sorry

end proof_problem_l257_257117


namespace second_term_is_4_l257_257147

-- Define the arithmetic sequence conditions
variables (a d : ℝ) -- first term a, common difference d

-- The condition given in the problem
def sum_first_and_third_term (a d : ℝ) : Prop :=
  a + (a + 2 * d) = 8

-- What we need to prove
theorem second_term_is_4 (a d : ℝ) (h : sum_first_and_third_term a d) : a + d = 4 :=
sorry

end second_term_is_4_l257_257147


namespace sara_basketball_loss_l257_257816

theorem sara_basketball_loss (total_games : ℕ) (games_won : ℕ) (games_lost : ℕ) 
  (h1 : total_games = 16) 
  (h2 : games_won = 12) 
  (h3 : games_lost = total_games - games_won) : 
  games_lost = 4 :=
by
  sorry

end sara_basketball_loss_l257_257816


namespace arccos_one_over_sqrt_two_eq_pi_four_l257_257742

theorem arccos_one_over_sqrt_two_eq_pi_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_four_l257_257742


namespace erica_earnings_l257_257916

def price_per_kg : ℝ := 20
def past_catch : ℝ := 80
def catch_today := 2 * past_catch
def total_catch := past_catch + catch_today
def total_earnings := total_catch * price_per_kg

theorem erica_earnings : total_earnings = 4800 := by
  sorry

end erica_earnings_l257_257916


namespace solve_for_y_l257_257275

noncomputable def x : ℝ := 20
noncomputable def y : ℝ := 40

theorem solve_for_y 
  (h₁ : 1.5 * x = 0.75 * y) 
  (h₂ : x = 20) : 
  y = 40 :=
by
  sorry

end solve_for_y_l257_257275


namespace largest_multiple_of_7_neg_greater_than_neg_150_l257_257343

theorem largest_multiple_of_7_neg_greater_than_neg_150 : 
  ∃ (k : ℤ), k % 7 = 0 ∧ -k > -150 ∧ (∀ (m : ℤ), m % 7 = 0 ∧ -m > -150 → k ≥ m) ∧ k = 147 :=
by
  sorry

end largest_multiple_of_7_neg_greater_than_neg_150_l257_257343


namespace janet_extra_flowers_l257_257296

-- Define the number of flowers Janet picked for each type
def tulips : ℕ := 5
def roses : ℕ := 10
def daisies : ℕ := 8
def lilies : ℕ := 4

-- Define the number of flowers Janet used
def used : ℕ := 19

-- Calculate the total number of flowers Janet picked
def total_picked : ℕ := tulips + roses + daisies + lilies

-- Calculate the number of extra flowers
def extra_flowers : ℕ := total_picked - used

-- The theorem to be proven
theorem janet_extra_flowers : extra_flowers = 8 :=
by
  -- You would provide the proof here, but it's not required as per instructions
  sorry

end janet_extra_flowers_l257_257296


namespace current_tree_height_in_inches_l257_257165

-- Constants
def initial_height_ft : ℝ := 10
def growth_percentage : ℝ := 0.50
def feet_to_inches : ℝ := 12

-- Conditions
def growth_ft : ℝ := growth_percentage * initial_height_ft
def current_height_ft : ℝ := initial_height_ft + growth_ft

-- Question/Answer equivalence
theorem current_tree_height_in_inches :
  (current_height_ft * feet_to_inches) = 180 :=
by 
  sorry

end current_tree_height_in_inches_l257_257165


namespace arccos_proof_l257_257680

noncomputable def arccos_identity : Prop := 
  ∃ x : ℝ, x = 1 / Real.sqrt 2 ∧ Real.arccos x = Real.pi / 4

theorem arccos_proof : arccos_identity :=
by
  sorry

end arccos_proof_l257_257680


namespace train_tunnel_length_l257_257230

theorem train_tunnel_length 
  (train_length : ℝ) 
  (train_speed : ℝ) 
  (time_for_tail_to_exit : ℝ) 
  (h_train_length : train_length = 2) 
  (h_train_speed : train_speed = 90) 
  (h_time_for_tail_to_exit : time_for_tail_to_exit = 2 / 60) :
  ∃ tunnel_length : ℝ, tunnel_length = 1 := 
by
  sorry

end train_tunnel_length_l257_257230


namespace sum_of_cubes_l257_257980

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 11) (h2 : x * y = 12) : x^3 + y^3 = 935 := by
  sorry

end sum_of_cubes_l257_257980


namespace inequality_b_2pow_a_a_2pow_neg_b_l257_257446

theorem inequality_b_2pow_a_a_2pow_neg_b (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : 
  b * 2^a + a * 2^(-b) ≥ a + b :=
sorry

end inequality_b_2pow_a_a_2pow_neg_b_l257_257446


namespace sum_of_cubes_eq_neg_27_l257_257795

variable {a b c : ℝ}

-- Define the condition that k is the same for a, b, and c
def same_k (a b c k : ℝ) : Prop :=
  k = (a^3 + 9) / a ∧ k = (b^3 + 9) / b ∧ k = (c^3 + 9) / c

-- Theorem: Given the conditions, a^3 + b^3 + c^3 = -27
theorem sum_of_cubes_eq_neg_27 (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_same_k : ∃ k, same_k a b c k) :
  a^3 + b^3 + c^3 = -27 :=
sorry

end sum_of_cubes_eq_neg_27_l257_257795


namespace deluxe_stereo_time_fraction_l257_257203

theorem deluxe_stereo_time_fraction (S : ℕ) (B : ℝ)
  (H1 : 2 / 3 > 0)
  (H2 : 1.6 > 0) :
  (1.6 / 3 * S * B) / (1.2 * S * B) = 4 / 9 :=
by
  sorry

end deluxe_stereo_time_fraction_l257_257203


namespace three_digit_powers_of_two_count_l257_257419

theorem three_digit_powers_of_two_count : 
  (finset.range (10)).filter (λ n, 100 ≤ 2^n ∧ 2^n ≤ 999) = {7, 8, 9} := by
    sorry

end three_digit_powers_of_two_count_l257_257419


namespace largest_common_term_up_to_150_l257_257389

theorem largest_common_term_up_to_150 :
  ∃ a : ℕ, a ≤ 150 ∧ (∃ n : ℕ, a = 2 + 8 * n) ∧ (∃ m : ℕ, a = 3 + 9 * m) ∧ (∀ b : ℕ, b ≤ 150 → (∃ n' : ℕ, b = 2 + 8 * n') → (∃ m' : ℕ, b = 3 + 9 * m') → b ≤ a) := 
sorry

end largest_common_term_up_to_150_l257_257389


namespace arccos_of_one_over_sqrt_two_l257_257720

theorem arccos_of_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
sorry

end arccos_of_one_over_sqrt_two_l257_257720


namespace probability_hardcover_liberal_arts_probability_liberal_arts_then_hardcover_l257_257483

-- Definitions based on the conditions provided
def total_books : ℕ := 100
def liberal_arts_books : ℕ := 40
def hardcover_books : ℕ := 70
def softcover_science_books : ℕ := 20
def hardcover_liberal_arts_books : ℕ := 30
def softcover_liberal_arts_books : ℕ := liberal_arts_books - hardcover_liberal_arts_books
def total_events_2 : ℕ := total_books * total_books

-- Statement part 1: Probability of selecting a hardcover liberal arts book
theorem probability_hardcover_liberal_arts :
  (hardcover_liberal_arts_books : ℝ) / total_books = 0.3 :=
sorry

-- Statement part 2: Probability of selecting a liberal arts book then a hardcover book (with replacement)
theorem probability_liberal_arts_then_hardcover :
  ((liberal_arts_books : ℝ) / total_books) * ((hardcover_books : ℝ) / total_books) = 0.28 :=
sorry

end probability_hardcover_liberal_arts_probability_liberal_arts_then_hardcover_l257_257483


namespace range_of_z_minus_x_z_minus_y_l257_257410

theorem range_of_z_minus_x_z_minus_y (x y z : ℝ) (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y) (h_nonneg_z : 0 ≤ z) (h_sum : x + y + z = 1) :
  -1 / 8 ≤ (z - x) * (z - y) ∧ (z - x) * (z - y) ≤ 1 := by
  sorry

end range_of_z_minus_x_z_minus_y_l257_257410


namespace mats_in_10_days_of_all_weavers_l257_257156

theorem mats_in_10_days_of_all_weavers :
  let rate_a := 4 / 6 : ℚ
  let rate_b := 5 / 7 : ℚ
  let rate_c := 3 / 4 : ℚ
  let rate_d := 6 / 9 : ℚ
  let total_rate := rate_a + rate_b + rate_c + rate_d
  let total_mats := total_rate * 10
  ⌊total_mats⌋ = 28 :=
by
  sorry

end mats_in_10_days_of_all_weavers_l257_257156


namespace dave_bought_packs_l257_257237

def packs_of_white_shirts (bought_total : ℕ) (white_per_pack : ℕ) (blue_packs : ℕ) (blue_per_pack : ℕ) : ℕ :=
  (bought_total - blue_packs * blue_per_pack) / white_per_pack

theorem dave_bought_packs : packs_of_white_shirts 26 6 2 4 = 3 :=
by
  sorry

end dave_bought_packs_l257_257237


namespace arccos_of_one_over_sqrt_two_l257_257723

theorem arccos_of_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
sorry

end arccos_of_one_over_sqrt_two_l257_257723


namespace cliff_shiny_igneous_l257_257288

variables (I S : ℕ)

theorem cliff_shiny_igneous :
  I = S / 2 ∧ I + S = 270 → I / 3 = 30 := 
by
  intro h
  sorry

end cliff_shiny_igneous_l257_257288


namespace incorrect_option_D_l257_257959

theorem incorrect_option_D (x y : ℝ) : y = (x - 2) ^ 2 + 1 → ¬ (∀ (x : ℝ), x < 2 → y < (x - 1) ^ 2 + 1) :=
by
  intro h
  sorry

end incorrect_option_D_l257_257959


namespace min_value_of_squared_sums_l257_257449

theorem min_value_of_squared_sums (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : 
  ∃ B, (B = x^2 + y^2 + z^2) ∧ (B ≥ 4) := 
by {
  sorry -- Proof will be provided here.
}

end min_value_of_squared_sums_l257_257449


namespace find_number_of_clerks_l257_257846

-- Define the conditions 
def avg_salary_per_head_staff : ℝ := 90
def avg_salary_officers : ℝ := 600
def avg_salary_clerks : ℝ := 84
def number_of_officers : ℕ := 2

-- Define the variable C (number of clerks)
def number_of_clerks : ℕ := sorry   -- We will prove that this is 170

-- Define the total salary equations based on the conditions
def total_salary_officers := number_of_officers * avg_salary_officers
def total_salary_clerks := number_of_clerks * avg_salary_clerks
def total_number_of_staff := number_of_officers + number_of_clerks
def total_salary := total_salary_officers + total_salary_clerks

-- Define the average salary per head equation 
def avg_salary_eq : Prop := avg_salary_per_head_staff = total_salary / total_number_of_staff

theorem find_number_of_clerks (h : avg_salary_eq) : number_of_clerks = 170 :=
sorry

end find_number_of_clerks_l257_257846


namespace intersection_of_M_and_N_l257_257265

def M : Set ℝ := {x | -3 < x ∧ x < 1}
def N : Set ℝ := {-3, -2, -1, 0, 1}

theorem intersection_of_M_and_N : M ∩ N = {-2, -1, 0} := sorry

end intersection_of_M_and_N_l257_257265


namespace sum_faces_edges_vertices_triangular_prism_l257_257364

-- Given conditions for triangular prism:
def triangular_prism_faces : Nat := 2 + 3  -- 2 triangular faces and 3 rectangular faces
def triangular_prism_edges : Nat := 3 + 3 + 3  -- 3 top edges, 3 bottom edges, 3 connecting edges
def triangular_prism_vertices : Nat := 3 + 3  -- 3 vertices on the top base, 3 on the bottom base

-- Proof statement for the sum of the faces, edges, and vertices of a triangular prism
theorem sum_faces_edges_vertices_triangular_prism : 
  triangular_prism_faces + triangular_prism_edges + triangular_prism_vertices = 20 := by
  sorry

end sum_faces_edges_vertices_triangular_prism_l257_257364


namespace car_y_start_time_l257_257867

theorem car_y_start_time : 
  ∀ (t m : ℝ), 
  (35 * (t + m) = 294) ∧ (40 * t = 294) → 
  t = 7.35 ∧ m = 1.05 → 
  m * 60 = 63 :=
by
  intros t m h1 h2
  sorry

end car_y_start_time_l257_257867


namespace neg_int_solution_l257_257477

theorem neg_int_solution (x : ℤ) : -2 * x < 4 ↔ x = -1 :=
by
  sorry

end neg_int_solution_l257_257477


namespace prime_divides_factorial_plus_one_non_prime_not_divides_factorial_plus_one_factorial_mod_non_prime_is_zero_l257_257180

-- Show that if \( p \) is a prime number, then \( p \) divides \( (p-1)! + 1 \).
theorem prime_divides_factorial_plus_one (p : ℕ) (hp : Nat.Prime p) : p ∣ (Nat.factorial (p - 1) + 1) :=
sorry

-- Show that if \( n \) is not a prime number, then \( n \) does not divide \( (n-1)! + 1 \).
theorem non_prime_not_divides_factorial_plus_one (n : ℕ) (hn : ¬Nat.Prime n) : ¬(n ∣ (Nat.factorial (n - 1) + 1)) :=
sorry

-- Calculate the remainder of the division of \((n-1)!\) by \( n \).
theorem factorial_mod_non_prime_is_zero (n : ℕ) (hn : ¬Nat.Prime n) : (Nat.factorial (n - 1)) % n = 0 :=
sorry

end prime_divides_factorial_plus_one_non_prime_not_divides_factorial_plus_one_factorial_mod_non_prime_is_zero_l257_257180


namespace tetrahedron_fourth_face_possibilities_l257_257558

theorem tetrahedron_fourth_face_possibilities :
  ∃ (S : Set String), S = {"right-angled triangle", "acute-angled triangle", "isosceles triangle", "isosceles right-angled triangle", "equilateral triangle"} :=
sorry

end tetrahedron_fourth_face_possibilities_l257_257558


namespace sum_even_numbered_terms_l257_257956

variable (n : ℕ)

def a_n (n : ℕ) : ℕ := 2 * 3^(n-1)

def new_sequence (n : ℕ) : ℕ := a_n (2 * n)

def Sn (n : ℕ) : ℕ := (6 * (1 - 9^n)) / (1 - 9)

theorem sum_even_numbered_terms (n : ℕ) : Sn n = 3 * (9^n - 1) / 4 :=
by sorry

end sum_even_numbered_terms_l257_257956


namespace arccos_one_over_sqrt_two_eq_pi_four_l257_257739

theorem arccos_one_over_sqrt_two_eq_pi_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_four_l257_257739


namespace edmonton_to_red_deer_distance_l257_257109

noncomputable def distance_from_Edmonton_to_Calgary (speed time: ℝ) : ℝ :=
  speed * time

theorem edmonton_to_red_deer_distance :
  let speed := 110
  let time := 3
  let distance_Calgary_RedDeer := 110
  let distance_Edmonton_Calgary := distance_from_Edmonton_to_Calgary speed time
  let distance_Edmonton_RedDeer := distance_Edmonton_Calgary - distance_Calgary_RedDeer
  distance_Edmonton_RedDeer = 220 :=
by
  sorry

end edmonton_to_red_deer_distance_l257_257109


namespace problem_l257_257134

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem problem (a : ℝ) (h : f a = 2) : f (-a) = 0 := 
  sorry

end problem_l257_257134


namespace sum_sequence_eq_l257_257940

noncomputable def S (n : ℕ) : ℝ := Real.log (1 + n) / Real.log 0.1

theorem sum_sequence_eq :
  (S 99 - S 9) = -1 := by
  sorry

end sum_sequence_eq_l257_257940


namespace trigonometric_values_l257_257540

-- Define cos and sin terms
def cos (x : ℝ) : ℝ := sorry
def sin (x : ℝ) : ℝ := sorry

-- Define the condition given in the problem statement
def condition (x : ℝ) : Prop := cos x - 4 * sin x = 1

-- Define the result we need to prove
def result (x : ℝ) : Prop := sin x + 4 * cos x = 4 ∨ sin x + 4 * cos x = -4

-- The main statement in Lean 4 to be proved
theorem trigonometric_values (x : ℝ) : condition x → result x := by
  sorry

end trigonometric_values_l257_257540


namespace original_deck_size_l257_257043

-- Define the conditions
def boys_kept_away (remaining_cards kept_away_cards : ℕ) : Prop :=
  remaining_cards + kept_away_cards = 52

-- Define the problem
theorem original_deck_size (remaining_cards : ℕ) (kept_away_cards := 2) :
  boys_kept_away remaining_cards kept_away_cards → remaining_cards + kept_away_cards = 52 :=
by
  intro h
  exact h

end original_deck_size_l257_257043


namespace additional_distance_l257_257141

theorem additional_distance (distance_speed_10 : ℝ) (speed1 speed2 time1 time2 distance actual_distance additional_distance : ℝ)
  (h1 : actual_distance = distance_speed_10)
  (h2 : time1 = distance_speed_10 / speed1)
  (h3 : time1 = 5)
  (h4 : speed1 = 10)
  (h5 : time2 = actual_distance / speed2)
  (h6 : speed2 = 14)
  (h7 : distance = speed2 * time1)
  (h8 : distance = 70)
  : additional_distance = distance - actual_distance
  := by
  sorry

end additional_distance_l257_257141


namespace value_of_x_squared_plus_reciprocal_l257_257423

theorem value_of_x_squared_plus_reciprocal (x : ℝ) (h : 47 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = 7 :=
by
  sorry

end value_of_x_squared_plus_reciprocal_l257_257423


namespace ratio_of_c_and_d_l257_257560

theorem ratio_of_c_and_d 
  (x y c d : ℝ)
  (h₁ : 4 * x - 2 * y = c)
  (h₂ : 6 * y - 12 * x = d) 
  (h₃ : d ≠ 0) : 
  c / d = -1 / 3 :=
by
  sorry

end ratio_of_c_and_d_l257_257560


namespace total_chairs_calculation_l257_257216

-- Definitions of the conditions
def numIndoorTables : Nat := 9
def numOutdoorTables : Nat := 11
def chairsPerIndoorTable : Nat := 10
def chairsPerOutdoorTable : Nat := 3

-- The proposition we want to prove
theorem total_chairs_calculation :
  numIndoorTables * chairsPerIndoorTable + numOutdoorTables * chairsPerOutdoorTable = 123 := by
sorry

end total_chairs_calculation_l257_257216


namespace max_value_of_d_l257_257927

theorem max_value_of_d : ∀ (d e : ℕ), (∃ (n : ℕ), n = 70733 + 10^4 * d + e ∧ (∃ (k3 k11 : ℤ), n = 3 * k3 ∧ n = 11 * k11) ∧ d = e ∧ d ≤ 9) → d = 2 :=
by 
  -- Given conditions and goals:
  -- 1. The number has the form 7d7,33e which in numerical form is: n = 70733 + 10^4 * d + e
  -- 2. The number n is divisible by 3 and 11.
  -- 3. d and e are digits (0 ≤ d, e ≤ 9).
  -- 4. To maximize the value of d, ensure that the given conditions hold.
  -- Problem: Prove that the maximum value of d for which this holds is 2.
  sorry

end max_value_of_d_l257_257927


namespace acute_triangle_properties_l257_257563

theorem acute_triangle_properties (A B C : ℝ) (AC BC : ℝ)
  (h_acute : ∀ {x : ℝ}, x = A ∨ x = B ∨ x = C → x < π / 2)
  (h_BC : BC = 1)
  (h_B_eq_2A : B = 2 * A) :
  (AC / Real.cos A = 2) ∧ (Real.sqrt 2 < AC ∧ AC < Real.sqrt 3) :=
by
  sorry

end acute_triangle_properties_l257_257563


namespace convert_246_octal_to_decimal_l257_257070

theorem convert_246_octal_to_decimal : 2 * (8^2) + 4 * (8^1) + 6 * (8^0) = 166 := 
by
  -- We skip the proof part as it is not required in the task
  sorry

end convert_246_octal_to_decimal_l257_257070


namespace total_number_of_notes_l257_257493

-- The total amount of money in Rs.
def total_amount : ℕ := 400

-- The number of each type of note is equal.
variable (n : ℕ)

-- The total value equation given the number of each type of note.
def total_value : ℕ := n * 1 + n * 5 + n * 10

-- Prove that if the total value equals 400, the total number of notes is 75.
theorem total_number_of_notes : total_value n = total_amount → 3 * n = 75 :=
by
  sorry

end total_number_of_notes_l257_257493


namespace largest_multiple_of_7_negation_gt_neg150_l257_257346

theorem largest_multiple_of_7_negation_gt_neg150 : 
  ∃ (k : ℤ), (k % 7 = 0 ∧ -k > -150 ∧ ∀ (m : ℤ), (m % 7 = 0 ∧ -m > -150 → m ≤ k)) :=
sorry

end largest_multiple_of_7_negation_gt_neg150_l257_257346


namespace arccos_one_over_sqrt_two_l257_257707

theorem arccos_one_over_sqrt_two : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l257_257707


namespace square_side_length_exists_l257_257455

theorem square_side_length_exists
    (k : ℕ)
    (n : ℕ)
    (h_side_length_condition : n * n = k * (k - 7))
    (h_grid_lines : k > 7) :
    n = 12 ∨ n = 24 :=
by sorry

end square_side_length_exists_l257_257455


namespace union_sets_l257_257950

def set_A : Set ℝ := {x | x^3 - 3 * x^2 - x + 3 < 0}
def set_B : Set ℝ := {x | |x + 1 / 2| ≥ 1}

theorem union_sets :
  set_A ∪ set_B = ( {x : ℝ | x < -1} ∪ {x : ℝ | x ≥ 1 / 2} ) :=
by
  sorry

end union_sets_l257_257950


namespace number_of_valid_four_digit_numbers_l257_257972

-- Definition for the problem
def is_valid_four_digit_number (n : ℕ) : Prop :=
  2999 < n ∧ n <= 9999 ∧
  (let d1 := n / 1000,
       d2 := (n / 100) % 10,
       d3 := (n / 10) % 10,
       d4 := n % 10 in
   3 <= d1 ∧ d1 <= 9 ∧
   0 <= d4 ∧ d4 <= 9 ∧
   d2 * d3 > 10)

-- Statement of the problem
theorem number_of_valid_four_digit_numbers : 
  (Finset.range 10000).filter is_valid_four_digit_number).card = 4830 :=
sorry

end number_of_valid_four_digit_numbers_l257_257972


namespace girls_joined_l257_257990

theorem girls_joined (initial_girls : ℕ) (boys : ℕ) (girls_more_than_boys : ℕ) (G : ℕ) :
  initial_girls = 632 →
  boys = 410 →
  girls_more_than_boys = 687 →
  initial_girls + G = boys + girls_more_than_boys →
  G = 465 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  linarith

end girls_joined_l257_257990


namespace total_cakes_served_l257_257511

def L : Nat := 5
def D : Nat := 6
def Y : Nat := 3
def T : Nat := L + D + Y

theorem total_cakes_served : T = 14 := by
  sorry

end total_cakes_served_l257_257511


namespace number_of_grandchildren_l257_257551

/- Definitions based on conditions -/
def price_before_discount := 20.0
def discount_rate := 0.20
def monogram_cost := 12.0
def total_expenditure := 140.0

/- Definition based on discount calculation -/
def price_after_discount := price_before_discount * (1.0 - discount_rate)

/- Final theorem statement -/
theorem number_of_grandchildren : 
  total_expenditure / (price_after_discount + monogram_cost) = 5 := by
  sorry

end number_of_grandchildren_l257_257551


namespace no_integer_root_quadratic_trinomials_l257_257238

theorem no_integer_root_quadratic_trinomials :
  ¬ ∃ (a b c : ℤ),
    (∃ r1 r2 : ℤ, a * r1^2 + b * r1 + c = 0 ∧ a * r2^2 + b * r2 + c = 0 ∧ r1 ≠ r2) ∧
    (∃ s1 s2 : ℤ, (a + 1) * s1^2 + (b + 1) * s1 + (c + 1) = 0 ∧ (a + 1) * s2^2 + (b + 1) * s2 + (c + 1) = 0 ∧ s1 ≠ s2) :=
by
  sorry

end no_integer_root_quadratic_trinomials_l257_257238


namespace second_player_cannot_prevent_l257_257617

noncomputable section

structure Player where
  id : ℕ

def first_player : Player := ⟨1⟩
def second_player : Player := ⟨2⟩

structure Dot where
  color : String
  position : (ℝ × ℝ)

def is_equilateral_triangle (p1 p2 p3 : Dot) : Prop :=
  let is_dist_equal (a b c d e f) := (a - c)^2 + (b - d)^2 = (a - e)^2 + (b - f)^2 ∧ (a - c)^2 + (b - d)^2 = (c - e)^2 + (d - f)^2
  is_dist_equal p1.position.1 p1.position.2 p2.position.1 p2.position.2 p3.position.1 p3.position.2

def game_condition (red_dots blue_dots : List Dot) : Prop :=
  ∀ (three_reds : List (Dot)), three_reds.length = 3 → ¬is_equilateral_triangle three_reds.head three_reds.tail.head three_reds.tail.tail.head

theorem second_player_cannot_prevent (red_dots : List Dot) (blue_dots : List Dot) (H1 : ∀ i, i ∈ red_dots → i.color = "red")
  (H2 : ∀ i, i ∈ blue_dots → i.color = "blue") (H3 : (red_dots.length + 12) = 13 ∧ blue_dots.length = 120) : 
  ¬game_condition red_dots blue_dots :=
  sorry

end second_player_cannot_prevent_l257_257617


namespace warriors_truth_tellers_l257_257896

/-- There are 33 warriors. Each warrior is either a truth-teller or a liar, 
    with only one favorite weapon: a sword, a spear, an axe, or a bow. 
    They were asked four questions, and the number of "Yes" answers to the 
    questions are 13, 15, 20, and 27 respectively. Prove that the number of 
    warriors who always tell the truth is 12. -/
theorem warriors_truth_tellers
  (warriors : ℕ) (truth_tellers : ℕ)
  (yes_to_sword : ℕ) (yes_to_spear : ℕ)
  (yes_to_axe : ℕ) (yes_to_bow : ℕ)
  (h1 : warriors = 33)
  (h2 : yes_to_sword = 13)
  (h3 : yes_to_spear = 15)
  (h4 : yes_to_axe = 20)
  (h5 : yes_to_bow = 27)
  (h6 : yes_to_sword + yes_to_spear + yes_to_axe + yes_to_bow = 75) :
  truth_tellers = 12 := by
  -- Proof will be here
  sorry

end warriors_truth_tellers_l257_257896


namespace exists_x_y_mod_p_l257_257302

theorem exists_x_y_mod_p (p : ℕ) (hp : Nat.Prime p) (a : ℤ) : ∃ x y : ℤ, (x^2 + y^3) % p = a % p :=
by
  sorry

end exists_x_y_mod_p_l257_257302


namespace perfect_square_trinomial_l257_257527

theorem perfect_square_trinomial (c : ℝ) : (∃ a : ℝ, (x : ℝ) → x^2 + 150 * x + c = (x + a)^2) → c = 5625 :=
sorry

end perfect_square_trinomial_l257_257527


namespace Vishal_investment_percentage_more_than_Trishul_l257_257333

-- Definitions from the conditions
def R : ℚ := 2400
def T : ℚ := 0.90 * R
def total_investments : ℚ := 6936

-- Mathematically equivalent statement to prove
theorem Vishal_investment_percentage_more_than_Trishul :
  ∃ V : ℚ, V + T + R = total_investments ∧ (V - T) / T * 100 = 10 := 
by
  sorry

end Vishal_investment_percentage_more_than_Trishul_l257_257333


namespace find_greater_number_l257_257480

theorem find_greater_number (x y : ℕ) 
  (h1 : x + y = 40)
  (h2 : x - y = 12) : x = 26 :=
by
  sorry

end find_greater_number_l257_257480


namespace arccos_one_over_sqrt_two_l257_257706

theorem arccos_one_over_sqrt_two : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l257_257706


namespace range_of_a_l257_257168

noncomputable def f (a : ℝ) (x : ℝ) := Real.exp x / (1 + a * x^2)

theorem range_of_a (a : ℝ) (ha : a > 0)
  (h_monotone : ∀ x y, x ≤ y → f a x ≤ f a y) : 0 < a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l257_257168


namespace binom_10_0_eq_1_l257_257871

theorem binom_10_0_eq_1 :
  (Nat.choose 10 0) = 1 :=
by
  sorry

end binom_10_0_eq_1_l257_257871


namespace rashmi_bus_stop_distance_l257_257460

theorem rashmi_bus_stop_distance
  (T D : ℝ)
  (h1 : 5 * (T + 10/60) = D)
  (h2 : 6 * (T - 10/60) = D) :
  D = 5 :=
by
  sorry

end rashmi_bus_stop_distance_l257_257460


namespace profit_23_percent_of_cost_price_l257_257649

-- Define the conditions
variable (C : ℝ) -- Cost price of the turtleneck sweaters
variable (C_nonneg : 0 ≤ C) -- Ensure cost price is non-negative

-- Definitions based on conditions
def SP1 (C : ℝ) : ℝ := 1.20 * C
def SP2 (SP1 : ℝ) : ℝ := 1.25 * SP1
def SPF (SP2 : ℝ) : ℝ := 0.82 * SP2

-- Define the profit calculation
def Profit (C : ℝ) : ℝ := (SPF (SP2 (SP1 C))) - C

-- Statement of the theorem
theorem profit_23_percent_of_cost_price (C : ℝ) (C_nonneg : 0 ≤ C):
  Profit C = 0.23 * C :=
by
  -- The actual proof would go here
  sorry

end profit_23_percent_of_cost_price_l257_257649


namespace machines_in_first_scenario_l257_257142

theorem machines_in_first_scenario :
  ∃ M : ℕ, (∀ (units1 units2 : ℕ) (hours1 hours2 : ℕ),
    units1 = 20 ∧ hours1 = 10 ∧ units2 = 200 ∧ hours2 = 25 ∧
    (M * units1 / hours1 = 20 * units2 / hours2)) → M = 5 :=
by
  sorry

end machines_in_first_scenario_l257_257142


namespace arccos_identity_l257_257734

theorem arccos_identity : 
  ∀ x : ℝ, x = 1 / real.sqrt 2 → real.arccos x = real.pi / 4 :=
begin
  intros x hx,
  have h1 : real.cos (real.pi / 4) = 1 / real.sqrt 2 := by sorry,
  have h2 : real.arccos (1 / real.sqrt 2) = real.pi / 4,
  { rw ← h1,
    rw real.arccos_cos,
    exact_mod_cast hx },
  exact h2,
end

end arccos_identity_l257_257734


namespace z_has_purely_imaginary_difference_l257_257130

theorem z_has_purely_imaginary_difference
  (z : ℂ) (h : z = 2 - complex.i) : z - 2 = -complex.i := 
sorry

end z_has_purely_imaginary_difference_l257_257130


namespace arccos_identity_l257_257727

theorem arccos_identity : 
  ∀ x : ℝ, x = 1 / real.sqrt 2 → real.arccos x = real.pi / 4 :=
begin
  intros x hx,
  have h1 : real.cos (real.pi / 4) = 1 / real.sqrt 2 := by sorry,
  have h2 : real.arccos (1 / real.sqrt 2) = real.pi / 4,
  { rw ← h1,
    rw real.arccos_cos,
    exact_mod_cast hx },
  exact h2,
end

end arccos_identity_l257_257727


namespace per_capita_income_growth_l257_257430

theorem per_capita_income_growth (x : ℝ) : 
  (250 : ℝ) * (1 + x) ^ 20 ≥ 800 →
  (250 : ℝ) * (1 + x) ^ 40 ≥ 2560 := 
by
  intros h
  -- Proof is not required, so we skip it with sorry
  sorry

end per_capita_income_growth_l257_257430


namespace machine_Y_produces_more_widgets_l257_257173

-- Definitions for the rates and widgets produced
def W_x := 18 -- widgets per hour by machine X
def total_widgets := 1080

-- Calculations for time taken by each machine
def T_x := total_widgets / W_x -- time taken by machine X
def T_y := T_x - 10 -- machine Y takes 10 hours less

-- Rate at which machine Y produces widgets
def W_y := total_widgets / T_y

-- Calculation of percentage increase
def percentage_increase := (W_y - W_x) / W_x * 100

-- The final theorem to prove
theorem machine_Y_produces_more_widgets : percentage_increase = 20 := by
  sorry

end machine_Y_produces_more_widgets_l257_257173


namespace gifts_from_Pedro_l257_257590

theorem gifts_from_Pedro (gifts_from_Emilio gifts_from_Jorge total_gifts : ℕ)
  (h1 : gifts_from_Emilio = 11)
  (h2 : gifts_from_Jorge = 6)
  (h3 : total_gifts = 21) :
  total_gifts - (gifts_from_Emilio + gifts_from_Jorge) = 4 := by
  sorry

end gifts_from_Pedro_l257_257590


namespace find_starting_number_l257_257484

theorem find_starting_number : 
  ∃ x : ℕ, (∃ n : ℕ, n = 21 ∧ (forall k, 1 ≤ k ∧ k ≤ n → x + k*19 ≤ 500) ∧ 
  (forall k, 1 ≤ k ∧ k < n → x + k*19 > 0)) ∧ x = 113 := by {
  sorry
}

end find_starting_number_l257_257484


namespace vovochkas_no_carry_pairs_eq_correct_pairs_vovochkas_smallest_difference_l257_257779

theorem vovochkas_no_carry_pairs_eq_correct_pairs :
  let valid_digit_pairs := Nat.range 10.map (λ n, 10 - n)
  let sum_valid_digit_pairs := valid_digit_pairs.sum
  sum_valid_digit_pairs = 55 →
  let no_carry_combinations := 81 * 55 * 55
  no_carry_combinations = 244620 :=
by
  let valid_digit_pairs := List.map (λ n, 10 - n) (List.range 10)
  have h_valid_digit_sum : valid_digit_pairs.sum = 55 := by sorry
  let no_carry_combinations := 81 * 55 * 55
  have h_no_carry : no_carry_combinations = 244620 := by sorry
  exact h_no_carry

theorem vovochkas_smallest_difference :
  let incorrect_cases := [1800]
  incorrect_cases.minimum = 1800 :=
by
  let differences := [900, 90, 990]
  have h_min_diff : List.minimum differences = some 90 := by sorry
  let incorrect_cases := List.map (λ diff, 20 * diff) differences
  have h_min_incorrect : incorrect_cases.minimum = some 1800 := by sorry
  exact h_min_incorrect

end vovochkas_no_carry_pairs_eq_correct_pairs_vovochkas_smallest_difference_l257_257779


namespace simplest_form_expression_l257_257405

variable {b : ℝ}

theorem simplest_form_expression (h : b ≠ 1) :
  1 - (1 / (2 + (b / (1 - b)))) = 1 / (2 - b) :=
by
  sorry

end simplest_form_expression_l257_257405


namespace total_fish_approximation_l257_257289

variable (TotalA TotalB TotalC : ℕ)

-- Given conditions
def conditions :=
  let prop1 := 180 = 90 + 60 + 30
  let prop2 := 100 = 45 + 35 + 20
  -- tagged fish proportions
  let proportionA := 4 / 45
  let proportionB := 3 / 35
  let proportionC := 1 / 20
  -- equations representing the total number of fish in the pond
  let eqA := (90 / TotalA : ℚ) = proportionA
  let eqB := (60 / TotalB : ℚ) = proportionB
  let eqC := (30 / TotalC : ℚ) = proportionC
  prop1 ∧ prop2 ∧ eqA ∧ eqB ∧ eqC

theorem total_fish_approximation (h : conditions TotalA TotalB TotalC) :
  TotalA + TotalB + TotalC = 2313 :=
sorry

end total_fish_approximation_l257_257289


namespace parrots_per_cage_l257_257225

theorem parrots_per_cage (P : ℕ) (parakeets_per_cage : ℕ) (cages : ℕ) (total_birds : ℕ) 
    (h1 : parakeets_per_cage = 7) (h2 : cages = 8) (h3 : total_birds = 72) 
    (h4 : total_birds = cages * P + cages * parakeets_per_cage) : 
    P = 2 :=
by
  sorry

end parrots_per_cage_l257_257225


namespace arccos_one_over_sqrt_two_eq_pi_four_l257_257735

theorem arccos_one_over_sqrt_two_eq_pi_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_four_l257_257735


namespace arith_general_formula_geom_general_formula_geom_sum_formula_l257_257369

-- Arithmetic Sequence Conditions
def arith_seq (a₈ a₁₀ : ℕ → ℝ) := a₈ = 6 ∧ a₁₀ = 0

-- General formula for arithmetic sequence
theorem arith_general_formula (a₁ : ℝ) (d : ℝ) (h₈ : 6 = a₁ + 7 * d) (h₁₀ : 0 = a₁ + 9 * d) :
  ∀ n : ℕ, aₙ = 30 - 3 * (n - 1) :=
sorry

-- General formula for geometric sequence
def geom_seq (a₁ a₄ : ℕ → ℝ) := a₁ = 1/2 ∧ a₄ = 4

theorem geom_general_formula (a₁ : ℝ) (q : ℝ) (h₁ : a₁ = 1 / 2) (h₄ : 4 = a₁ * q ^ 3) :
  ∀ n : ℕ, aₙ = 2^(n-2) :=
sorry

-- Sum of the first n terms of geometric sequence
theorem geom_sum_formula (a₁ : ℝ) (q : ℝ) (h₁ : a₁ = 1 / 2) (h₄ : 4 = a₁ * q ^ 3) :
  ∀ n : ℕ, Sₙ = 2^(n-1) - 1 / 2 :=
sorry

end arith_general_formula_geom_general_formula_geom_sum_formula_l257_257369


namespace max_d_77733e_divisible_by_33_l257_257924

open Int

theorem max_d_77733e_divisible_by_33 : ∃ d e : ℕ, 
  (7 * 100000 + d * 10000 + 7 * 1000 + 3 * 100 + 3 * 10 + e) % 33 = 0 ∧ 
  (d ≤ 9) ∧ (e ≤ 9) ∧ 
  (∀ d' e', ((7 * 100000 + d' * 10000 + 7 * 1000 + 3 * 100 + 3 * 10 + e') % 33 = 0 ∧ d' ≤ 9 ∧ e' ≤ 9 → d' ≤ d)) 
  := ⟨6, 0, by sorry⟩

end max_d_77733e_divisible_by_33_l257_257924


namespace value_of_4k_minus_1_l257_257549

theorem value_of_4k_minus_1 (k x y : ℝ)
  (h1 : x + y - 5 * k = 0)
  (h2 : x - y - 9 * k = 0)
  (h3 : 2 * x + 3 * y = 6) :
  4 * k - 1 = 2 :=
  sorry

end value_of_4k_minus_1_l257_257549


namespace non_negative_integer_solutions_l257_257005

theorem non_negative_integer_solutions (x : ℕ) : 3 * x - 2 < 7 ↔ x = 0 ∨ x = 1 ∨ x = 2 := by
  sorry

end non_negative_integer_solutions_l257_257005


namespace johns_final_weight_is_200_l257_257437

-- Define the initial weight, percentage of weight loss, and weight gain
def initial_weight : ℝ := 220
def weight_loss_percentage : ℝ := 0.10
def weight_gain : ℝ := 2

-- Define a function to calculate the final weight
def final_weight (initial_weight : ℝ) (weight_loss_percentage : ℝ) (weight_gain : ℝ) : ℝ := 
  let weight_lost := initial_weight * weight_loss_percentage
  let weight_after_loss := initial_weight - weight_lost
  weight_after_loss + weight_gain

-- The proof problem is to show that the final weight is 200 pounds
theorem johns_final_weight_is_200 :
  final_weight initial_weight weight_loss_percentage weight_gain = 200 := 
by
  sorry

end johns_final_weight_is_200_l257_257437


namespace at_least_one_zero_of_product_zero_l257_257586

theorem at_least_one_zero_of_product_zero (a b c : ℝ) (h : a * b * c = 0) : a = 0 ∨ b = 0 ∨ c = 0 := by
  sorry

end at_least_one_zero_of_product_zero_l257_257586


namespace incorrect_conclusion_D_l257_257965

def parabola (x : ℝ) : ℝ := (x - 2) ^ 2 + 1

theorem incorrect_conclusion_D :
  ∀ x : ℝ, x < 2 → ∃ y1 y2 : ℝ, y1 = parabola x ∧ y2 = parabola (x + 1) ∧ y1 > y2 :=
by
  sorry

end incorrect_conclusion_D_l257_257965


namespace how_many_years_younger_is_C_compared_to_A_l257_257610

variables (a b c d : ℕ)

def condition1 : Prop := a + b = b + c + 13
def condition2 : Prop := b + d = c + d + 7
def condition3 : Prop := a + d = 2 * c - 12

theorem how_many_years_younger_is_C_compared_to_A
  (h1 : condition1 a b c)
  (h2 : condition2 b c d)
  (h3 : condition3 a c d) : a = c + 13 :=
sorry

end how_many_years_younger_is_C_compared_to_A_l257_257610


namespace largest_multiple_of_7_negation_greater_than_neg_150_l257_257339

theorem largest_multiple_of_7_negation_greater_than_neg_150 : 
  ∃ k : ℤ, k * 7 = 147 ∧ ∀ n : ℤ, (k < n → n * 7 ≤ 150) :=
by
  use 21
  sorry

end largest_multiple_of_7_negation_greater_than_neg_150_l257_257339


namespace count_birds_l257_257108

theorem count_birds (b m c : ℕ) (h1 : b + m + c = 300) (h2 : 2 * b + 4 * m + 3 * c = 708) : b = 192 := 
sorry

end count_birds_l257_257108


namespace sum_three_numbers_l257_257189

noncomputable def sum_of_three_numbers (a b c : ℝ) : ℝ :=
  a + b + c

theorem sum_three_numbers 
  (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = a + 20) 
  (h2 : (a + b + c) / 3 = c - 30) 
  (h3 : b = 10) :
  sum_of_three_numbers a b c = 60 :=
by
  sorry

end sum_three_numbers_l257_257189


namespace hexagon_side_length_l257_257324

theorem hexagon_side_length (p : ℕ) (s : ℕ) (h₁ : p = 24) (h₂ : s = 6) : p / s = 4 := by
  sorry

end hexagon_side_length_l257_257324


namespace similar_right_triangles_hypotenuse_relation_similar_right_triangles_reciprocal_relation_l257_257438

variable {a b c m_c a' b' c' m_c' : ℝ}

/- The first proof problem -/
theorem similar_right_triangles_hypotenuse_relation (h_sim : (a = k * a') ∧ (b = k * b') ∧ (c = k * c')) :
  a * a' + b * b' = c * c' := by
  sorry

/- The second proof problem -/
theorem similar_right_triangles_reciprocal_relation (h_sim : (a = k * a') ∧ (b = k * b') ∧ (c = k * c') ∧ (m_c = k * m_c')) :
  (1 / (a * a') + 1 / (b * b')) = 1 / (m_c * m_c') := by
  sorry

end similar_right_triangles_hypotenuse_relation_similar_right_triangles_reciprocal_relation_l257_257438


namespace find_square_side_length_l257_257457

/-- Define the side lengths of the rectangle and the square --/
def rectangle_side_lengths (k : ℕ) (n : ℕ) : Prop := 
  k ≥ 7 ∧ n = 12 ∧ k * (k - 7) = n * n

theorem find_square_side_length (k n : ℕ) : rectangle_side_lengths k n → n = 12 :=
by
  intros
  sorry

end find_square_side_length_l257_257457


namespace sum_cubes_eq_neg_27_l257_257792

theorem sum_cubes_eq_neg_27 (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
 (h_eq : (a^3 + 9) / a = (b^3 + 9) / b ∧ (b^3 + 9) / b = (c^3 + 9) / c) :
 a^3 + b^3 + c^3 = -27 :=
by
  sorry

end sum_cubes_eq_neg_27_l257_257792


namespace max_digit_d_of_form_7d733e_multiple_of_33_l257_257921

theorem max_digit_d_of_form_7d733e_multiple_of_33 
  (d e : ℕ) (d_digit : d < 10) (e_digit : e < 10) 
  (multiple_of_33: ∃ k : ℕ, 7 * 10^5 + d * 10^4 + 7 * 10^3 + 33 * 10 + e = 33 * k) 
  : d ≤ 6 := 
sorry

end max_digit_d_of_form_7d733e_multiple_of_33_l257_257921


namespace convert_base_8_to_base_10_l257_257081

def to_base_10 (n : ℕ) (b : ℕ) (digits : List ℕ) : ℕ :=
  digits.foldr (λ digit acc => acc * b + digit) 0

theorem convert_base_8_to_base_10 : 
  to_base_10 10 8 [6, 4, 2] = 166 := by
  sorry

end convert_base_8_to_base_10_l257_257081


namespace arccos_one_over_sqrt_two_eq_pi_four_l257_257736

theorem arccos_one_over_sqrt_two_eq_pi_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_four_l257_257736


namespace base8_to_base10_l257_257076

theorem base8_to_base10 (n : ℕ) : of_digits 8 [2, 4, 6] = 166 := by
  sorry

end base8_to_base10_l257_257076


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l257_257700

theorem arccos_one_over_sqrt_two_eq_pi_over_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l257_257700


namespace arith_seq_ratio_l257_257316

-- Definitions related to arithmetic sequence and sum
def arithmetic_seq (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_arith_seq (S a : ℕ → ℝ) := ∀ n : ℕ, S n = (n : ℝ) / 2 * (a 1 + a n)

-- Given condition
def condition (a : ℕ → ℝ) := a 8 / a 7 = 13 / 5

-- Prove statement
theorem arith_seq_ratio (a S : ℕ → ℝ)
  (h_arith : arithmetic_seq a)
  (h_sum : sum_of_arith_seq S a)
  (h_cond : condition a) :
  S 15 / S 13 = 3 := 
sorry

end arith_seq_ratio_l257_257316


namespace bus_children_problem_l257_257370

theorem bus_children_problem :
  ∃ X, 5 - 63 + X = 14 ∧ X - 63 = 9 :=
by 
  sorry

end bus_children_problem_l257_257370


namespace b_arithmetic_b_formula_T_formula_l257_257293

variable {a : ℕ → ℝ}
variable {b : ℕ → ℝ}
variable {c : ℕ → ℝ}

-- Define the sequence {a_n}
axiom a1 : a 1 = 2
axiom a_recurrence : ∀ n : ℕ, a n * a (n + 1) - 2 * a n + 1 = 0

-- Define the sequence {b_n}
def b (n : ℕ) := 2 / (a n - 1)

-- Question 1: Prove that {b_n} is an arithmetic sequence
theorem b_arithmetic : ∀ n : ℕ, b (n + 1) - b n = 2 := sorry

-- Proof that b_n = 2n
theorem b_formula : ∀ n : ℕ, b n = 2 * n := sorry

-- Define the sequence {c_n}
def c (n : ℕ) := if n = 1 then b 1 else 2 * 3^(n - 1)

-- Define the sum of nc_n
def T (n : ℕ) := ∑ i in Finset.range n, i * c (i + 1)

-- Question 2: Prove the formula for T_n
theorem T_formula : ∀ n : ℕ, T n = (n - 1/2) * 3^n + 1/2 := sorry

end b_arithmetic_b_formula_T_formula_l257_257293


namespace smallest_special_gt_3429_l257_257091

def is_special (n : ℕ) : Prop :=
  let digits := (n.digits 10).erase_dup
  digits.length = 4

theorem smallest_special_gt_3429 : ∃ n : ℕ, n > 3429 ∧ is_special n ∧ ∀ m : ℕ, m > 3429 ∧ is_special m → n ≤ m :=
by
  use 3450
  constructor
  . exact nat.lt_succ_self 3429  -- 3450 > 3429
  constructor
  . unfold is_special
    dsimp
    norm_num
  . intro m
    intro h
    intro hspec
    sorry

end smallest_special_gt_3429_l257_257091


namespace trigonometric_identity_l257_257464

theorem trigonometric_identity (x : ℝ) : 
  x = Real.pi / 4 → (1 + Real.sin (x + Real.pi / 4) - Real.cos (x + Real.pi / 4)) / 
                          (1 + Real.sin (x + Real.pi / 4) + Real.cos (x + Real.pi / 4)) = 1 :=
by 
  sorry

end trigonometric_identity_l257_257464


namespace ratio_proof_l257_257472

variables {d l e : ℕ} -- Define variables representing the number of doctors, lawyers, and engineers
variables (hd : ℕ → ℕ) (hl : ℕ → ℕ) (he : ℕ → ℕ) (ho : ℕ → ℕ)

-- Condition: Average ages
def avg_age_doctors := 40 * d
def avg_age_lawyers := 55 * l
def avg_age_engineers := 35 * e

-- Condition: Overall average age is 45 years
def overall_avg_age := (40 * d + 55 * l + 35 * e) / (d + l + e)

theorem ratio_proof (h1 : 40 * d + 55 * l + 35 * e = 45 * (d + l + e)) : 
  d = l ∧ e = 2 * l :=
by
  sorry

end ratio_proof_l257_257472


namespace frog_reaches_C_in_n_jumps_frog_reaches_C_no_D_in_n_jumps_frog_alive_probability_frog_average_lifespan_l257_257496

-- Part (a): Prove the number of ways to reach vertex C from A in n jumps when n is even
theorem frog_reaches_C_in_n_jumps (n : ℕ) (h_even : n % 2 = 0) : 
    ∃ c : ℕ, c = (4^n/2 - 1) / 3 := by sorry

-- Part (b): Prove the number of ways to reach vertex C from A in n jumps without jumping to D when n is even
theorem frog_reaches_C_no_D_in_n_jumps (n : ℕ) (h_even : n % 2 = 0) : 
    ∃ c : ℕ, c = 3^(n/2 - 1) := by sorry

-- Part (c): Prove the probability the frog is alive after n jumps with a mine at D
theorem frog_alive_probability (n : ℕ) (k : ℕ) (h_n : n = 2*k - 1 ∨ n = 2*k) : 
    ∃ p : ℝ, p = (3/4)^(k-1) := by sorry

-- Part (d): Prove the average lifespan of the frog in the presence of a mine at D
theorem frog_average_lifespan : 
    ∃ t : ℝ, t = 9 := by sorry

end frog_reaches_C_in_n_jumps_frog_reaches_C_no_D_in_n_jumps_frog_alive_probability_frog_average_lifespan_l257_257496


namespace original_price_of_good_l257_257632

theorem original_price_of_good (P : ℝ) (h1 : 0.684 * P = 6840) : P = 10000 :=
sorry

end original_price_of_good_l257_257632


namespace binom_10_0_eq_1_l257_257870

-- Definition of factorial
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Definition of binomial coefficient
def binom (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Theorem stating the binomial coefficient we need to prove
theorem binom_10_0_eq_1 : binom 10 0 = 1 :=
by
  -- This is where the proof would go, but we'll use sorry to skip it
  sorry

end binom_10_0_eq_1_l257_257870


namespace cello_viola_pairs_are_70_l257_257219

-- Given conditions
def cellos : ℕ := 800
def violas : ℕ := 600
def pair_probability : ℝ := 0.00014583333333333335

-- Theorem statement translating the mathematical problem
theorem cello_viola_pairs_are_70 (n : ℕ) (h1 : cellos = 800) (h2 : violas = 600) (h3 : pair_probability = 0.00014583333333333335) :
  n = 70 :=
sorry

end cello_viola_pairs_are_70_l257_257219


namespace alice_zoe_difference_l257_257309

-- Definitions of the conditions
def AliceApples := 8
def ZoeApples := 2

-- Theorem statement to prove the difference in apples eaten
theorem alice_zoe_difference : AliceApples - ZoeApples = 6 := by
  -- Proof
  sorry

end alice_zoe_difference_l257_257309


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l257_257697

theorem arccos_one_over_sqrt_two_eq_pi_over_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l257_257697


namespace larger_square_side_length_l257_257196

theorem larger_square_side_length (s1 s2 : ℝ) (h1 : s1 = 5) (h2 : s2 = s1 * 3) (a1 a2 : ℝ) (h3 : a1 = s1^2) (h4 : a2 = s2^2) : s2 = 15 := 
by
  sorry

end larger_square_side_length_l257_257196


namespace solve_for_x_l257_257979

variable (x y : ℝ)
variable (hx : x > 0)
variable (hy : y > 0)
variable (h : 3 * x^2 + 9 * x * y = x^3 + 3 * x^2 * y)

theorem solve_for_x : x = 3 :=
by
  sorry

end solve_for_x_l257_257979


namespace vovochka_no_carry_correct_cases_vovochka_minimum_difference_l257_257777

-- Part (a)
theorem vovochka_no_carry_correct_cases :
  let digit_pairs_without_carry := ∑ k in finset.range 10, (k + 1)
  let three_digit_pairs_without_carry := 9 * digit_pairs_without_carry
  let total_valid_combinations := 81 * three_digit_pairs_without_carry * three_digit_pairs_without_carry
  total_valid_combinations = 244620 :=
by {
  -- Definitions used in conditions
  let digit_pairs_without_carry := ∑ k in finset.range 10, (k + 1)
  let three_digit_pairs_without_carry := 9 * digit_pairs_without_carry
  let total_valid_combinations := 81 * three_digit_pairs_without_carry * three_digit_pairs_without_carry

  -- Assert the correct answer
  have correct_total_cases : total_valid_combinations = 244620 := 
    -- solution provided proof here
    sorry,

  exact correct_total_cases
}

-- Part (b)
theorem vovochka_minimum_difference :
  let smallest_difference := 1800
  ∃ a b c x y z : ℕ, (a, b, c) < (x, y, z) ∧ a + x ≥ 1 ∧ b + y ≥ 1 ∧ c + z ≥ 1 ∧
  correct_sum a b c x y z - vovochka_sum a b c x y z = smallest_difference :=
by {
  -- Definitions used in conditions
  let correct_sum := λ (a b c x y z : ℕ), 100 * (a + x) + 10 * (b + y) + (c + z)
  let vovochka_sum := λ (a b c x y z : ℕ), 100 * (a + x) + 10 * (b + y) + (c + z) % 10

  -- Assert the correct answer
  have smallest_diff_exists : 
    ∃ a b c x y z : ℕ, (a, b, c) < (x, y, z) ∧ a + x ≥ 1 ∧ b + y ≥ 1 ∧ c + z ≥ 1 ∧
    correct_sum a b c x y z - vovochka_sum a b c x y z = 1800 := 
    -- solution provided proof here
    sorry,

  exact smallest_diff_exists
}

end vovochka_no_carry_correct_cases_vovochka_minimum_difference_l257_257777


namespace initial_percentage_of_water_l257_257040

theorem initial_percentage_of_water (C V final_volume : ℝ) (P : ℝ) 
  (hC : C = 80)
  (hV : V = 36)
  (h_final_volume : final_volume = (3/4) * C)
  (h_initial_equation: (P / 100) * C + V = final_volume) : 
  P = 30 :=
by
  sorry

end initial_percentage_of_water_l257_257040


namespace average_visitors_per_day_l257_257646

theorem average_visitors_per_day (avg_visitors_Sunday : ℕ) (avg_visitors_other_days : ℕ) (total_days : ℕ) (starts_on_Sunday : Bool) :
  avg_visitors_Sunday = 500 → 
  avg_visitors_other_days = 140 → 
  total_days = 30 → 
  starts_on_Sunday = true → 
  (4 * avg_visitors_Sunday + 26 * avg_visitors_other_days) / total_days = 188 :=
by
  intros h1 h2 h3 h4
  sorry

end average_visitors_per_day_l257_257646


namespace total_fruits_in_baskets_l257_257508

def total_fruits (apples1 oranges1 bananas1 apples2 oranges2 bananas2 : ℕ) :=
  apples1 + oranges1 + bananas1 + apples2 + oranges2 + bananas2

theorem total_fruits_in_baskets :
  total_fruits 9 15 14 (9 - 2) (15 - 2) (14 - 2) = 70 :=
by
  sorry

end total_fruits_in_baskets_l257_257508


namespace truthfulness_count_l257_257889

-- Define variables to represent the number of warriors and their response counts
def num_warriors : Nat := 33
def yes_sword : Nat := 13
def yes_spear : Nat := 15
def yes_axe : Nat := 20
def yes_bow : Nat := 27

-- Define the total number of "Yes" answers
def total_yes_answers : Nat := yes_sword + yes_spear + yes_axe + yes_bow

theorem truthfulness_count :
  ∃ x : Nat, x + 3 * (num_warriors - x) = total_yes_answers ∧ x = 12 :=
by
  sorry

end truthfulness_count_l257_257889


namespace equal_charges_at_4_hours_l257_257306

-- Define the charges for both companies
def PaulsPlumbingCharge (h : ℝ) : ℝ := 55 + 35 * h
def ReliablePlumbingCharge (h : ℝ) : ℝ := 75 + 30 * h

-- Prove that for 4 hours of labor, the charges are equal
theorem equal_charges_at_4_hours : PaulsPlumbingCharge 4 = ReliablePlumbingCharge 4 :=
by
  sorry

end equal_charges_at_4_hours_l257_257306


namespace skill_position_players_waiting_l257_257853

def linemen_drink : ℕ := 8
def skill_position_player_drink : ℕ := 6
def num_linemen : ℕ := 12
def num_skill_position_players : ℕ := 10
def cooler_capacity : ℕ := 126

theorem skill_position_players_waiting :
  num_skill_position_players - (cooler_capacity - num_linemen * linemen_drink) / skill_position_player_drink = 5 :=
by
  -- Calculation is needed to be filled in here
  sorry

end skill_position_players_waiting_l257_257853


namespace problem_statement_l257_257556

variable (x : ℝ)

theorem problem_statement (h : x^2 - x - 1 = 0) : 1995 + 2 * x - x^3 = 1994 := by
  sorry

end problem_statement_l257_257556


namespace pieces_per_box_l257_257372

theorem pieces_per_box (total_pieces : ℕ) (boxes : ℕ) (h_total : total_pieces = 3000) (h_boxes : boxes = 6) :
  total_pieces / boxes = 500 := by
  sorry

end pieces_per_box_l257_257372


namespace length_of_bridge_l257_257188

theorem length_of_bridge
  (length_train : ℕ) (speed_train_kmhr : ℕ) (crossing_time : ℕ)
  (speed_conversion_factor : ℝ) (m_per_s_kmhr : ℝ) 
  (speed_train_ms : ℝ) (total_distance : ℝ) (length_bridge : ℝ)
  (h1 : length_train = 155)
  (h2 : speed_train_kmhr = 45)
  (h3 : crossing_time = 30)
  (h4 : speed_conversion_factor = 1000 / 3600)
  (h5 : m_per_s_kmhr = speed_train_kmhr * speed_conversion_factor)
  (h6 : speed_train_ms = 45 * (5 / 18))
  (h7 : total_distance = speed_train_ms * crossing_time)
  (h8 : length_bridge = total_distance - length_train):
  length_bridge = 220 :=
by
  sorry

end length_of_bridge_l257_257188


namespace truthful_warriors_count_l257_257914

-- Noncomputable theory is not necessary here.
-- We define the number of warriors and their responses.
def warriors_count : ℕ := 33
def yes_sword : ℕ := 13
def yes_spear : ℕ := 15
def yes_axe : ℕ := 20
def yes_bow : ℕ := 27

-- The equation from the solution steps where x is the number of truthful warriors.
def total_yes := yes_sword + yes_spear + yes_axe + yes_bow

theorem truthful_warriors_count :
  ∃ x : ℕ, x + 3 * (warriors_count - x) = total_yes ∧ x = 12 :=
by {
  -- We state that there exists an x such that its value satisfies the total "yes" responses equation.
  -- Here, we assert that x equals 12.
  use 12,
  split,
  {
    -- Prove the equation x + 3 * (warriors_count - x) = total_yes
    rw [warriors_count],
    rw [total_yes],
    norm_num,
  },
  {
    -- State that x = 12
    refl,
  },
}

end truthful_warriors_count_l257_257914


namespace value_of_m_l257_257766

theorem value_of_m (m : ℤ) (h1 : abs m = 2) (h2 : m - 2 ≠ 0) : m = -2 :=
by
  sorry

end value_of_m_l257_257766


namespace apples_problem_l257_257461

theorem apples_problem :
  ∃ (jackie rebecca : ℕ), (rebecca = 2 * jackie) ∧ (∃ (adam : ℕ), (adam = jackie + 3) ∧ (adam = 9) ∧ jackie = 6 ∧ rebecca = 12) :=
by
  sorry

end apples_problem_l257_257461


namespace inequality_bound_l257_257840

theorem inequality_bound (a x : ℝ) (h1 : |a| ≤ 1) (h2 : |x| ≤ 1) :
  |x^2 - ax - a^2| ≤ 5 / 4 :=
sorry

end inequality_bound_l257_257840


namespace fermat_1000_units_digit_l257_257582

-- Define Fermat numbers
def FermatNumber (n : ℕ) : ℕ := 2 ^ (2 ^ n) + 1

-- Define a function to extract the units digit
def units_digit (n : ℕ) : ℕ := n % 10

-- The theorem to be proven
theorem fermat_1000_units_digit : units_digit (FermatNumber 1000) = 7 := 
by sorry

end fermat_1000_units_digit_l257_257582


namespace expand_expression_l257_257111

theorem expand_expression (x y : ℤ) : (x + 12) * (3 * y + 8) = 3 * x * y + 8 * x + 36 * y + 96 := 
by
  sorry

end expand_expression_l257_257111


namespace incorrect_conclusion_D_l257_257963

def parabola (x : ℝ) : ℝ := (x - 2) ^ 2 + 1

theorem incorrect_conclusion_D :
  ∀ x : ℝ, x < 2 → ∃ y1 y2 : ℝ, y1 = parabola x ∧ y2 = parabola (x + 1) ∧ y1 > y2 :=
by
  sorry

end incorrect_conclusion_D_l257_257963


namespace base8_246_is_166_in_base10_l257_257068

def convert_base8_to_base10 (n : ℕ) : ℕ :=
  let d0 := n % 10;
  let d1 := (n / 10) % 10;
  let d2 := (n / 100) % 10;
  d2 * 8^2 + d1 * 8^1 + d0 * 8^0

theorem base8_246_is_166_in_base10 : convert_base8_to_base10 246 = 166 :=
  sorry

end base8_246_is_166_in_base10_l257_257068


namespace larry_expression_correct_l257_257172

theorem larry_expression_correct (a b c d e : ℤ) (h₁ : a = 3) (h₂ : b = 6) (h₃ : c = 2) (h₄ : d = 5) :
  (a - b + c - d + e = a - (b + (c - (d - e)))) → e = 3 :=
by
  sorry

end larry_expression_correct_l257_257172


namespace truthful_warriors_count_l257_257913

-- Noncomputable theory is not necessary here.
-- We define the number of warriors and their responses.
def warriors_count : ℕ := 33
def yes_sword : ℕ := 13
def yes_spear : ℕ := 15
def yes_axe : ℕ := 20
def yes_bow : ℕ := 27

-- The equation from the solution steps where x is the number of truthful warriors.
def total_yes := yes_sword + yes_spear + yes_axe + yes_bow

theorem truthful_warriors_count :
  ∃ x : ℕ, x + 3 * (warriors_count - x) = total_yes ∧ x = 12 :=
by {
  -- We state that there exists an x such that its value satisfies the total "yes" responses equation.
  -- Here, we assert that x equals 12.
  use 12,
  split,
  {
    -- Prove the equation x + 3 * (warriors_count - x) = total_yes
    rw [warriors_count],
    rw [total_yes],
    norm_num,
  },
  {
    -- State that x = 12
    refl,
  },
}

end truthful_warriors_count_l257_257913


namespace second_term_arithmetic_seq_l257_257153

variable (a d : ℝ)

theorem second_term_arithmetic_seq (h : a + (a + 2 * d) = 8) : a + d = 4 := by
  sorry

end second_term_arithmetic_seq_l257_257153


namespace girls_on_playground_l257_257485

variable (total_children : ℕ) (boys : ℕ) (girls : ℕ)

theorem girls_on_playground (h1 : total_children = 117) (h2 : boys = 40) (h3 : girls = total_children - boys) : girls = 77 :=
by
  sorry

end girls_on_playground_l257_257485


namespace fraction_of_fliers_sent_out_l257_257842

-- Definitions based on the conditions
def total_fliers : ℕ := 2500
def fliers_next_day : ℕ := 1500

-- Defining the fraction sent in the morning as x
variable (x : ℚ)

-- The remaining fliers after morning
def remaining_fliers_morning := (1 - x) * total_fliers

-- The remaining fliers after afternoon
def remaining_fliers_afternoon := remaining_fliers_morning - (1/4) * remaining_fliers_morning

-- The theorem statement
theorem fraction_of_fliers_sent_out :
  remaining_fliers_afternoon = fliers_next_day → x = 1/5 :=
sorry

end fraction_of_fliers_sent_out_l257_257842


namespace prove_m_range_l257_257775

theorem prove_m_range (m : ℝ) :
  (∀ x : ℝ, (2 * x + 5) / 3 - 1 ≤ 2 - x → 3 * (x - 1) + 5 > 5 * x + 2 * (m + x)) → m < -3 / 5 := by
  sorry

end prove_m_range_l257_257775


namespace avg_marks_calculation_l257_257634

theorem avg_marks_calculation (max_score : ℕ)
    (gibi_percent jigi_percent mike_percent lizzy_percent : ℚ)
    (hg : gibi_percent = 0.59) (hj : jigi_percent = 0.55) 
    (hm : mike_percent = 0.99) (hl : lizzy_percent = 0.67)
    (hmax : max_score = 700) :
    ((gibi_percent * max_score + jigi_percent * max_score +
      mike_percent * max_score + lizzy_percent * max_score) / 4 = 490) :=
by
  sorry

end avg_marks_calculation_l257_257634


namespace john_money_left_l257_257789

def cost_of_drink (q : ℝ) : ℝ := q
def cost_of_small_pizza (q : ℝ) : ℝ := cost_of_drink q
def cost_of_large_pizza (q : ℝ) : ℝ := 4 * cost_of_drink q
def total_cost (q : ℝ) : ℝ := 2 * cost_of_drink q + 2 * cost_of_small_pizza q + cost_of_large_pizza q
def initial_money : ℝ := 50
def remaining_money (q : ℝ) : ℝ := initial_money - total_cost q

theorem john_money_left (q : ℝ) : remaining_money q = 50 - 8 * q :=
by
  sorry

end john_money_left_l257_257789


namespace total_students_l257_257175

-- Define the conditions based on the problem
def valentines_have : ℝ := 58.0
def valentines_needed : ℝ := 16.0

-- Theorem stating that the total number of students (which is equal to the total number of Valentines required)
theorem total_students : valentines_have + valentines_needed = 74.0 :=
by
  sorry

end total_students_l257_257175


namespace system_of_equations_solution_l257_257466

theorem system_of_equations_solution :
  ∃ (X Y: ℝ), 
    (X^2 * Y^2 + X * Y^2 + X^2 * Y + X * Y + X + Y + 3 = 0) ∧ 
    (X^2 * Y + X * Y + 1 = 0) ∧ 
    (X = -2) ∧ (Y = -1/2) :=
by
  sorry

end system_of_equations_solution_l257_257466


namespace part_a_part_b_l257_257209

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

theorem part_a :
  ¬∃ (f g : ℝ → ℝ), (∀ x, f (g x) = x^2) ∧ (∀ x, g (f x) = x^3) :=
sorry

theorem part_b :
  ∃ (f g : ℝ → ℝ), (∀ x, f (g x) = x^2) ∧ (∀ x, g (f x) = x^4) :=
sorry

end part_a_part_b_l257_257209


namespace arccos_one_over_sqrt_two_l257_257703

theorem arccos_one_over_sqrt_two : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l257_257703


namespace paper_thickness_after_folding_five_times_l257_257531

-- Definitions of initial conditions
def initial_thickness : ℝ := 0.1
def num_folds : ℕ := 5

-- Target thickness after folding
def final_thickness (init_thickness : ℝ) (folds : ℕ) : ℝ :=
  (2 ^ folds) * init_thickness

-- Statement of the theorem
theorem paper_thickness_after_folding_five_times :
  final_thickness initial_thickness num_folds = 3.2 :=
by
  -- The proof (the implementation is replaced with sorry)
  sorry

end paper_thickness_after_folding_five_times_l257_257531


namespace smallest_n_l257_257023

theorem smallest_n (n : ℕ) (h1 : 1826 % 26 = 6) (h2 : 5 * n % 26 = 6) : n = 20 :=
sorry

end smallest_n_l257_257023


namespace find_x_l257_257944

open Nat

theorem find_x (n : ℕ) (x : ℕ) (h1 : x = 2^n - 32)
  (h2 : Nat.coprime 2 3)
  (h3 : (∀ p : ℕ, Prime p → p ∣ x → p = 2 ∨ p = 3))
  (h4 : Nat.count (λ p, Prime p ∧ p ∣ x) = 3)
  : x = 480 ∨ x = 2016 :=
by sorry

end find_x_l257_257944


namespace largest_neg_multiple_of_7_greater_than_neg_150_l257_257361

theorem largest_neg_multiple_of_7_greater_than_neg_150 : 
  ∃ (n : ℤ), (n % 7 = 0) ∧ (-n > -150) ∧ (∀ m : ℤ, (m % 7 = 0) ∧ (-m > -150) → m ≤ n) :=
begin
  use 147,
  split,
  { norm_num }, -- Verifies that 147 is a multiple of 7
  split,
  { norm_num }, -- Verifies that -147 > -150
  { intros m h,
    obtain ⟨k, rfl⟩ := (zmod.int_coe_zmod_eq_zero_iff_dvd m 7).mp h.1,
    suffices : k ≤ 21, { rwa [int.nat_abs_of_nonneg (by norm_num : (7 : ℤ) ≥ 0), ←abs_eq_nat_abs, int.abs_eq_nat_abs, nat.abs_of_nonneg (zero_le 21), ← int.le_nat_abs_iff_coe_nat_le] at this },
    have : -m > -150 := h.2,
    rwa [int.lt_neg, neg_le_neg_iff] at this,
    norm_cast at this,
    exact this
  }
end

end largest_neg_multiple_of_7_greater_than_neg_150_l257_257361


namespace largest_multiple_of_7_neg_greater_than_neg_150_l257_257344

theorem largest_multiple_of_7_neg_greater_than_neg_150 : 
  ∃ (k : ℤ), k % 7 = 0 ∧ -k > -150 ∧ (∀ (m : ℤ), m % 7 = 0 ∧ -m > -150 → k ≥ m) ∧ k = 147 :=
by
  sorry

end largest_multiple_of_7_neg_greater_than_neg_150_l257_257344


namespace convert_base_8_to_base_10_l257_257082

def to_base_10 (n : ℕ) (b : ℕ) (digits : List ℕ) : ℕ :=
  digits.foldr (λ digit acc => acc * b + digit) 0

theorem convert_base_8_to_base_10 : 
  to_base_10 10 8 [6, 4, 2] = 166 := by
  sorry

end convert_base_8_to_base_10_l257_257082


namespace quotient_division_l257_257626

/-- Definition of the condition that when 14 is divided by 3, the remainder is 2 --/
def division_property : Prop :=
  14 = 3 * (14 / 3) + 2

/-- Statement for finding the quotient when 14 is divided by 3 --/
theorem quotient_division (A : ℕ) (h : 14 = 3 * A + 2) : A = 4 :=
by
  have rem_2 := division_property
  sorry

end quotient_division_l257_257626


namespace rectangle_area_problem_l257_257375

theorem rectangle_area_problem (l w l1 l2 w1 w2 : ℝ) (h1 : l = l1 + l2) (h2 : w = w1 + w2) 
  (h3 : l1 * w1 = 12) (h4 : l2 * w1 = 15) (h5 : l1 * w2 = 12) 
  (h6 : l2 * w2 = 8) (h7 : w1 * l2 = 18) (h8 : l1 * w2 = 20) :
  l2 * w1 = 18 :=
sorry

end rectangle_area_problem_l257_257375


namespace distinct_integers_sum_l257_257535

theorem distinct_integers_sum (n : ℕ) (h : n > 3) (a : Fin n → ℤ)
  (h1 : ∀ i, 1 ≤ a i) (h2 : ∀ i j, i < j → a i < a j) (h3 : ∀ i, a i ≤ 2 * n - 3) :
  ∃ (i j k l m : Fin n), i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m ∧ j ≠ k ∧ j ≠ l ∧ j ≠ m ∧ 
  k ≠ l ∧ k ≠ m ∧ l ≠ m ∧ a i + a j = a k + a l ∧ a k + a l = a m :=
by
  sorry

end distinct_integers_sum_l257_257535


namespace mode_of_data_set_l257_257123

def avg (s : List ℚ) : ℚ := s.sum / s.length

theorem mode_of_data_set :
  ∃ (x : ℚ), avg [1, 0, -3, 5, x, 2, -3] = 1 ∧
  (∀ s : List ℚ, s = [1, 0, -3, 5, x, 2, -3] →
  mode s = [(-3 : ℚ), (5 : ℚ)]) :=
by
  sorry

end mode_of_data_set_l257_257123


namespace power_expression_result_l257_257190

theorem power_expression_result : (-2)^2004 + (-2)^2005 = -2^2004 :=
by
  sorry

end power_expression_result_l257_257190


namespace boys_at_beginning_is_15_l257_257383

noncomputable def number_of_boys_at_beginning (B : ℝ) : Prop :=
  let girls_start := 1.20 * B
  let girls_end := 2 * girls_start
  let total_students := B + girls_end
  total_students = 51 

theorem boys_at_beginning_is_15 : number_of_boys_at_beginning 15 := 
  by
  -- Sorry is added to skip the proof
  sorry

end boys_at_beginning_is_15_l257_257383


namespace greater_number_l257_257482

theorem greater_number (a b : ℝ) (h1 : a + b = 40) (h2 : a - b = 12) : a = 26 :=
by
  have h3 : 2 * a = 52 := by linarith
  have h4 : a = 26 := by linarith
  exact h4

end greater_number_l257_257482


namespace min_of_x_squared_y_squared_z_squared_l257_257450

theorem min_of_x_squared_y_squared_z_squared (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  x^2 + y^2 + z^2 ≥ 4 :=
by sorry

end min_of_x_squared_y_squared_z_squared_l257_257450


namespace find_a1_l257_257444

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (s : ℕ → ℝ) :=
∀ n : ℕ, s n = (n * (a 1 + a n)) / 2

theorem find_a1 
  (a : ℕ → ℝ) (s : ℕ → ℝ)
  (d : ℝ)
  (h_seq : arithmetic_sequence a d)
  (h_sum : sum_first_n_terms a s)
  (h_S10_eq_S11 : s 10 = s 11) : 
  a 1 = 20 := 
sorry

end find_a1_l257_257444


namespace smallest_multiple_of_40_gt_100_l257_257199

theorem smallest_multiple_of_40_gt_100 :
  ∃ x : ℕ, 0 < x ∧ 40 * x > 100 ∧ ∀ y : ℕ, 0 < y ∧ 40 * y > 100 → x ≤ y → 40 * x = 120 :=
by
  sorry

end smallest_multiple_of_40_gt_100_l257_257199


namespace exponent_equality_l257_257030

theorem exponent_equality (M : ℕ) : 32^4 * 4^6 = 2^M → M = 32 :=
by
  sorry

end exponent_equality_l257_257030


namespace PP1_length_l257_257160

open Real

theorem PP1_length (AB AC : ℝ) (h₁ : AB = 5) (h₂ : AC = 3)
  (h₃ : ∃ γ : ℝ, γ = 90)  -- a right angle at A
  (BC : ℝ) (h₄ : BC = sqrt (AB^2 - AC^2))
  (A1B : ℝ) (A1C : ℝ) (h₅ : BC = A1B + A1C)
  (h₆ : A1B / A1C = AB / AC)
  (PQ : ℝ) (h₇ : PQ = A1B)
  (PR : ℝ) (h₈ : PR = A1C)
  (PP1 : ℝ) :
  PP1 = (3 * sqrt 5) / 4 :=
sorry

end PP1_length_l257_257160


namespace find_m_l257_257416

theorem find_m (a b c m x : ℂ) :
  ( (2 * m + 1) * (x^2 - (b + 1) * x) = (2 * m - 3) * (2 * a * x - c) )
  →
  (x = (b + 1)) 
  →
  m = 1.5 := by
  sorry

end find_m_l257_257416


namespace unattainable_value_l257_257765

theorem unattainable_value : ∀ x : ℝ, x ≠ -4/3 → (y = (2 - x) / (3 * x + 4) → y ≠ -1/3) :=
by
  intro x hx h
  rw [eq_comm] at h
  sorry

end unattainable_value_l257_257765


namespace arccos_one_over_sqrt_two_l257_257749

theorem arccos_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l257_257749


namespace find_n_l257_257830

open Nat

theorem find_n (d : ℕ → ℕ) (n : ℕ) (h1 : ∀ j, d (j + 1) > d j) (h2 : n = d 13 + d 14 + d 15) (h3 : (d 5 + 1)^3 = d 15 + 1) : 
  n = 1998 :=
by
  sorry

end find_n_l257_257830


namespace gcd_8885_4514_5246_l257_257233

theorem gcd_8885_4514_5246 : Nat.gcd (Nat.gcd 8885 4514) 5246 = 1 :=
sorry

end gcd_8885_4514_5246_l257_257233


namespace investments_interest_yielded_l257_257653

def total_investment : ℝ := 15000
def part_one_investment : ℝ := 8200
def rate_one : ℝ := 0.06
def rate_two : ℝ := 0.075

def part_two_investment : ℝ := total_investment - part_one_investment

def interest_one : ℝ := part_one_investment * rate_one * 1
def interest_two : ℝ := part_two_investment * rate_two * 1

def total_interest : ℝ := interest_one + interest_two

theorem investments_interest_yielded : total_interest = 1002 := by
  sorry

end investments_interest_yielded_l257_257653


namespace profit_condition_maximize_profit_l257_257505

noncomputable def profit (x : ℕ) : ℕ := 
  (x + 10) * (300 - 10 * x)

theorem profit_condition (x : ℕ) : profit x = 3360 ↔ x = 2 ∨ x = 18 := by
  sorry

theorem maximize_profit : ∃ x, x = 10 ∧ profit x = 4000 := by
  sorry

end profit_condition_maximize_profit_l257_257505


namespace correct_quadratic_eq_l257_257158

-- Define the given conditions
def first_student_sum (b : ℝ) : Prop := 5 + 3 = -b
def second_student_product (c : ℝ) : Prop := (-12) * (-4) = c

-- Define the proof statement
theorem correct_quadratic_eq (b c : ℝ) (h1 : first_student_sum b) (h2 : second_student_product c) :
    b = -8 ∧ c = 48 ∧ (∀ x : ℝ, x^2 + b * x + c = 0 → (x=5 ∨ x=3 ∨ x=-12 ∨ x=-4)) :=
by
  sorry

end correct_quadratic_eq_l257_257158


namespace max_positive_integer_value_l257_257952

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n: ℕ, ∃ q: ℝ, a (n + 1) = a n * q

theorem max_positive_integer_value
  (a : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : ∀ n, a n > 0)
  (h3 : a 2 * a 4 = 4)
  (h4 : a 1 + a 2 + a 3 = 14) : 
  ∃ n, n ≤ 4 ∧ a n * a (n+1) * a (n+2) > 1 / 9 :=
sorry

end max_positive_integer_value_l257_257952


namespace Paul_lost_161_crayons_l257_257583

def total_crayons : Nat := 589
def crayons_given : Nat := 571
def extra_crayons_given : Nat := 410

theorem Paul_lost_161_crayons : ∃ L : Nat, crayons_given = L + extra_crayons_given ∧ L = 161 := by
  sorry

end Paul_lost_161_crayons_l257_257583


namespace value_of_y_l257_257269

theorem value_of_y (x y : ℝ) (cond1 : 1.5 * x = 0.75 * y) (cond2 : x = 20) : y = 40 :=
by
  sorry

end value_of_y_l257_257269


namespace total_charts_16_l257_257665

def total_charts_brought (number_of_associate_professors : Int) (number_of_assistant_professors : Int) : Int :=
  number_of_associate_professors * 1 + number_of_assistant_professors * 2

theorem total_charts_16 (A B : Int)
  (h1 : 2 * A + B = 11)
  (h2 : A + B = 9) :
  total_charts_brought A B = 16 :=
by {
  -- the proof will go here
  sorry
}

end total_charts_16_l257_257665


namespace find_n_divides_polynomial_l257_257395

theorem find_n_divides_polynomial :
  ∀ (n : ℕ), 0 < n → (n + 2) ∣ (n^3 + 3 * n + 29) ↔ (n = 1 ∨ n = 3 ∨ n = 13) :=
by
  sorry

end find_n_divides_polynomial_l257_257395


namespace arccos_of_one_over_sqrt_two_l257_257726

theorem arccos_of_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
sorry

end arccos_of_one_over_sqrt_two_l257_257726


namespace provider_assignment_ways_l257_257756

theorem provider_assignment_ways (total_providers : ℕ) (children : ℕ) (h1 : total_providers = 15) (h2 : children = 4) : 
  (Finset.range total_providers).card.factorial / (Finset.range (total_providers - children)).card.factorial = 32760 :=
by
  rw [h1, h2]
  norm_num
  sorry

end provider_assignment_ways_l257_257756


namespace number_of_articles_l257_257284

-- Conditions
variables (C S : ℚ)
-- Given that the cost price of 50 articles is equal to the selling price of some number of articles N.
variables (N : ℚ) (h1 : 50 * C = N * S)
-- Given that the gain is 11.11111111111111 percent.
variables (gain : ℚ := 1/9) (h2 : S = C * (1 + gain))

-- Prove that N = 45
theorem number_of_articles (C S : ℚ) (N : ℚ) (h1 : 50 * C = N * S)
    (gain : ℚ := 1/9) (h2 : S = C * (1 + gain)) : N = 45 :=
by
  sorry

end number_of_articles_l257_257284


namespace find_total_sales_l257_257494

theorem find_total_sales
  (S : ℝ)
  (h_comm1 : ∀ x, x ≤ 5000 → S = 0.9 * x → S = 16666.67 → false)
  (h_comm2 : S > 5000 → S - (500 + 0.05 * (S - 5000)) = 15000):
  S = 16052.63 :=
by
  sorry

end find_total_sales_l257_257494


namespace batches_of_engines_l257_257425

variable (total_engines : ℕ) (not_defective_engines : ℕ := 300) (engines_per_batch : ℕ := 80)

theorem batches_of_engines (h1 : 3 * total_engines / 4 = not_defective_engines) :
  total_engines / engines_per_batch = 5 := by
sorry

end batches_of_engines_l257_257425


namespace number_of_truth_tellers_is_twelve_l257_257908
noncomputable theory

section
variables (x : ℕ)
variables (y : ℕ)
variables (a b c d : ℕ)

-- Given conditions
def total_warriors : ℕ := 33
def yes_sword : ℕ := 13
def yes_spear : ℕ := 15
def yes_axe : ℕ := 20
def yes_bow : ℕ := 27

-- Calculate total yes answers (condition 6)
def total_yes_answers : ℕ := yes_sword + yes_spear + yes_axe + yes_bow

-- Truth-tellers say "yes" to only one question, liers say "yes" to three questions
def truth_yes : ℕ := x * 1
def lie_yes : ℕ := (total_warriors - x) * 3

theorem number_of_truth_tellers_is_twelve
  (h1 : total_warriors = 33)
  (h2 : yes_sword = 13)
  (h3 : yes_spear = 15)
  (h4 : yes_axe = 20)
  (h5 : yes_bow = 27)
  (h6 : total_yes_answers = 75)
  (h7 : total_yes_answers = truth_yes + lie_yes) :
    x = 12 :=
  by sorry

end

end number_of_truth_tellers_is_twelve_l257_257908


namespace monthly_average_growth_rate_optimal_selling_price_for_desired_profit_l257_257010

/- Part 1: Monthly Average Growth Rate -/
theorem monthly_average_growth_rate (m : ℝ) (sale_april sale_june : ℝ) (h_apr_val : sale_april = 256) (h_june_val : sale_june = 400) :
  256 * (1 + m) ^ 2 = 400 → m = 0.25 :=
sorry

/- Part 2: Optimal Selling Price for Desired Profit -/
theorem optimal_selling_price_for_desired_profit (y : ℝ) (initial_price selling_price : ℝ) (sale_june : ℝ) (h_june_sale : sale_june = 400) (profit : ℝ) (h_profit : profit = 8400) :
  (y - 35) * (1560 - 20 * y) = 8400 → y = 50 :=
sorry

end monthly_average_growth_rate_optimal_selling_price_for_desired_profit_l257_257010


namespace c_share_l257_257495

theorem c_share (a b c d e : ℝ) (k : ℝ)
  (h1 : a + b + c + d + e = 1010)
  (h2 : a - 25 = 4 * k)
  (h3 : b - 10 = 3 * k)
  (h4 : c - 15 = 6 * k)
  (h5 : d - 20 = 2 * k)
  (h6 : e - 30 = 5 * k) :
  c = 288 :=
by
  -- proof with necessary steps
  sorry

end c_share_l257_257495


namespace divisor_of_a_l257_257201

theorem divisor_of_a (a b : ℕ) (hx : a % x = 3) (hb : b % 6 = 5) (hab : (a * b) % 48 = 15) : x = 48 :=
by sorry

end divisor_of_a_l257_257201


namespace find_symmetric_curve_equation_l257_257759

def equation_of_curve_symmetric_to_line : Prop :=
  ∀ (x y : ℝ), (5 * x^2 + 12 * x * y - 22 * x - 12 * y - 19 = 0 ∧ x - y + 2 = 0) →
  12 * x * y + 5 * y^2 - 78 * y + 45 = 0

theorem find_symmetric_curve_equation : equation_of_curve_symmetric_to_line :=
sorry

end find_symmetric_curve_equation_l257_257759


namespace find_salary_l257_257631

def salary_remaining (S : ℝ) (food : ℝ) (house_rent : ℝ) (clothes : ℝ) (remaining : ℝ) : Prop :=
  S - food * S - house_rent * S - clothes * S = remaining

theorem find_salary :
  ∀ S : ℝ, 
  salary_remaining S (1/5) (1/10) (3/5) 15000 → 
  S = 150000 :=
by
  intros S h
  sorry

end find_salary_l257_257631


namespace find_x_l257_257528

theorem find_x (x : ℝ) (h : (4 + x) / (6 + x) = (2 + x) / (3 + x)) : x = 0 :=
by
  sorry

end find_x_l257_257528


namespace coin_problem_exists_l257_257499

theorem coin_problem_exists (n : ℕ) : 
  (∃ n, n % 8 = 6 ∧ n % 7 = 5 ∧ (∀ m, (m % 8 = 6 ∧ m % 7 = 5) → n ≤ m)) →
  (∃ n, (n % 8 = 6) ∧ (n % 7 = 5) ∧ (n % 9 = 0)) :=
by
  sorry

end coin_problem_exists_l257_257499


namespace gcd_problem_l257_257001

theorem gcd_problem :
  ∃ n : ℕ, (80 ≤ n) ∧ (n ≤ 100) ∧ (n % 9 = 0) ∧ (Nat.gcd n 27 = 9) ∧ (n = 90) :=
by sorry

end gcd_problem_l257_257001


namespace smallest_n_divisible_31_l257_257452

theorem smallest_n_divisible_31 (n : ℕ) : 31 ∣ (5 ^ n + n) → n = 30 :=
by
  sorry

end smallest_n_divisible_31_l257_257452


namespace expand_product_l257_257242

theorem expand_product (y : ℝ) : 5 * (y - 3) * (y + 10) = 5 * y^2 + 35 * y - 150 :=
by 
  sorry

end expand_product_l257_257242


namespace exactly_three_assertions_l257_257049

theorem exactly_three_assertions (x : ℕ) : 
  10 ≤ x ∧ x < 100 ∧
  ((x % 3 = 0) ∧ (x % 5 = 0) ∧ (x % 9 ≠ 0) ∧ (x % 15 = 0) ∧ (x % 25 ≠ 0) ∧ (x % 45 ≠ 0)) ↔
  (x = 15 ∨ x = 30 ∨ x = 60) :=
by
  sorry

end exactly_three_assertions_l257_257049


namespace range_of_f_l257_257762

noncomputable def f (x : ℝ) : ℝ := (3 * x + 8) / (x - 4)

theorem range_of_f : Set.range f = {y : ℝ | y ≠ 3} :=
sorry

end range_of_f_l257_257762


namespace interval_for_x_l257_257280

theorem interval_for_x (x : ℝ) 
  (hx1 : 1/x < 2) 
  (hx2 : 1/x > -3) : 
  x > 1/2 ∨ x < -1/3 :=
  sorry

end interval_for_x_l257_257280


namespace incorrect_conclusion_l257_257960

-- Define the given parabola.
def parabola (x : ℝ) : ℝ := (x - 2)^2 + 1

-- Define the conditions for the parabola.
def parabola_opens_upwards : Prop := ∀ x y : ℝ, parabola (x + y) = (x + y - 2)^2 + 1
def axis_of_symmetry : Prop := ∀ x : ℝ, parabola x = parabola (4 - x)
def vertex_coordinates : Prop := parabola 2 = 1 ∧ (parabola 2, 2) = (1, 2)
def behavior_when_x_less_than_2 : Prop := ∀ x : ℝ, x < 2 → parabola x < parabola (x + 1)

-- The statement that needs to be proven in Lean 4.
theorem incorrect_conclusion : ¬ behavior_when_x_less_than_2 :=
  by
  sorry

end incorrect_conclusion_l257_257960


namespace person_died_at_33_l257_257210

-- Define the conditions and constants
def start_age : ℕ := 25
def insurance_payment : ℕ := 10000
def premium : ℕ := 450
def loss : ℕ := 1000
def annual_interest_rate : ℝ := 0.05
def half_year_factor : ℝ := 1.025 -- half-yearly compounded interest factor

-- Calculate the number of premium periods (as an integer)
def n := 16 -- (derived from the calculations in the given solution)

-- Define the final age based on the number of premium periods
def final_age : ℕ := start_age + (n / 2)

-- The proof statement
theorem person_died_at_33 : final_age = 33 := by
  sorry

end person_died_at_33_l257_257210


namespace sin_monotonically_decreasing_l257_257321

open Real

theorem sin_monotonically_decreasing (f : ℝ → ℝ) (x : ℝ) :
  (∀ x, f x = sin (2 * x + π / 3)) →
  (0 ≤ x ∧ x ≤ π) →
  (∀ x, (π / 12) ≤ x ∧ x ≤ (7 * π / 12)) →
  ∀ x y, (x < y → f y ≤ f x) := by
  sorry

end sin_monotonically_decreasing_l257_257321


namespace normal_cost_of_car_wash_l257_257435

-- Conditions
variables (C : ℝ) (H1 : 20 * C > 0) (H2 : 0.60 * (20 * C) = 180)

-- Theorem to be proved
theorem normal_cost_of_car_wash (C : ℝ) (H1 : 20 * C > 0) (H2 : 0.60 * (20 * C) = 180) : C = 15 :=
by
  -- proof omitted
  sorry

end normal_cost_of_car_wash_l257_257435


namespace convert_base_8_to_base_10_l257_257084

def to_base_10 (n : ℕ) (b : ℕ) (digits : List ℕ) : ℕ :=
  digits.foldr (λ digit acc => acc * b + digit) 0

theorem convert_base_8_to_base_10 : 
  to_base_10 10 8 [6, 4, 2] = 166 := by
  sorry

end convert_base_8_to_base_10_l257_257084


namespace solve_quadratic_eq_l257_257315

theorem solve_quadratic_eq {x : ℝ} : (x^2 - 4 * x - 7 = 0) ↔ (x = 2 + sqrt 11 ∨ x = 2 - sqrt 11) :=
sorry

end solve_quadratic_eq_l257_257315


namespace value_of_Priyanka_l257_257772

-- Defining the context with the conditions
variables (X : ℕ) (Neha : ℕ) (Sonali Priyanka Sadaf Tanu : ℕ)
-- The conditions given in the problem
axiom h1 : Neha = X
axiom h2 : Sonali = 15
axiom h3 : Priyanka = 15
axiom h4 : Sadaf = Neha
axiom h5 : Tanu = Neha

-- Stating the theorem we need to prove
theorem value_of_Priyanka : Priyanka = 15 :=
by
  sorry

end value_of_Priyanka_l257_257772


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l257_257676

theorem arccos_one_over_sqrt_two_eq_pi_over_four : 
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l257_257676


namespace largest_multiple_of_7_gt_neg_150_l257_257357

theorem largest_multiple_of_7_gt_neg_150 : ∃ (x : ℕ), (x % 7 = 0) ∧ ((- (x : ℤ)) > -150) ∧ ∀ y : ℕ, (y % 7 = 0 ∧ (- (y : ℤ)) > -150) → y ≤ x :=
by
  sorry

end largest_multiple_of_7_gt_neg_150_l257_257357


namespace mode_of_data_set_l257_257121

variable (x : ℤ)
variable (data_set : List ℤ)
variable (average : ℚ)

-- Conditions
def initial_data_set := [1, 0, -3, 5, x, 2, -3]
def avg_condition := (1 + 0 + (-3) + 5 + x + 2 + (-3) : ℚ) / 7 = 1

-- Statement
theorem mode_of_data_set (h_avg : avg_condition x) : Multiset.mode (initial_data_set x) = { -3, 5 } := sorry

end mode_of_data_set_l257_257121


namespace gcd_ab_a2b2_eq_1_or_2_l257_257581

theorem gcd_ab_a2b2_eq_1_or_2
  (a b : Nat)
  (h_coprime : Nat.gcd a b = 1) :
  Nat.gcd (a + b) (a^2 + b^2) = 1 ∨ Nat.gcd (a + b) (a^2 + b^2) = 2 :=
by {
  sorry
}

end gcd_ab_a2b2_eq_1_or_2_l257_257581


namespace problem1_problem2_l257_257545

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin x * (Real.sqrt 3 * Real.cos x + Real.sin x) - 2

-- Problem 1
theorem problem1 (α : ℝ) (hα1 : Real.sin α = -1 / 2) (hα2 : Real.cos α = Real.sqrt 3 / 2) :
  f α = -3 := sorry

-- Problem 2
theorem problem2 (h0 : ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ -2) :
  ∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = -2 := sorry

end problem1_problem2_l257_257545


namespace smallest_special_gt_3429_l257_257092

def is_special (n : ℕ) : Prop :=
  let digits := (n.digits 10).erase_dup
  digits.length = 4

theorem smallest_special_gt_3429 : ∃ n : ℕ, n > 3429 ∧ is_special n ∧ ∀ m : ℕ, m > 3429 ∧ is_special m → n ≤ m :=
by
  use 3450
  constructor
  . exact nat.lt_succ_self 3429  -- 3450 > 3429
  constructor
  . unfold is_special
    dsimp
    norm_num
  . intro m
    intro h
    intro hspec
    sorry

end smallest_special_gt_3429_l257_257092


namespace distribution_scheme_count_l257_257755

-- Definitions based on conditions
variable (village1 village2 village3 village4 : Type)
variables (quota1 quota2 quota3 quota4 : ℕ)

-- Conditions as given in the problem
def valid_distribution (v1 v2 v3 v4 : ℕ) : Prop :=
  v1 = 1 ∧ v2 = 2 ∧ v3 = 3 ∧ v4 = 4

-- The goal is to prove the number of permutations is equal to 24
theorem distribution_scheme_count :
  (∃ v1 v2 v3 v4 : ℕ, valid_distribution v1 v2 v3 v4) → 
  (4 * 3 * 2 * 1 = 24) :=
by 
  sorry

end distribution_scheme_count_l257_257755


namespace max_value_of_expressions_l257_257128

theorem max_value_of_expressions (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  b > 1/2 ∧ b > 2 * a * b ∧ b > a^2 + b^2 :=
by
  sorry

end max_value_of_expressions_l257_257128


namespace ellipse_equation_no_match_l257_257609

-- Definitions based on conditions in a)
def a : ℝ := 6
def c : ℝ := 1

-- Calculation for b² based on solution steps
def b_squared := a^2 - c^2

-- Standard forms of ellipse equations
def standard_ellipse_eq1 (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b_squared) = 1
def standard_ellipse_eq2 (x y : ℝ) : Prop := (y^2 / a^2) + (x^2 / b_squared) = 1

-- The proof problem statement
theorem ellipse_equation_no_match : 
  ∀ (x y : ℝ), ¬(standard_ellipse_eq1 x y) ∧ ¬(standard_ellipse_eq2 x y) := 
sorry

end ellipse_equation_no_match_l257_257609


namespace find_solutions_l257_257934

theorem find_solutions (x : ℝ) : (x = -9 ∨ x = -3 ∨ x = 3) →
  (1 / (x^2 + 12 * x - 9) + 1 / (x^2 + 3 * x - 9) + 1 / (x^2 - 14 * x - 9) = 0) :=
by {
  sorry
}

end find_solutions_l257_257934


namespace problem_1_problem_2_l257_257255

-- Define the function f(x) = |x + a| + |x|
def f (x : ℝ) (a : ℝ) : ℝ := abs (x + a) + abs x

-- (Ⅰ) Prove that for a = 1, the solution set for f(x) ≥ 2 is (-∞, -1/2] ∪ [3/2, +∞)
theorem problem_1 : 
  ∀ (x : ℝ), f x 1 ≥ 2 ↔ (x ≤ -1/2 ∨ x ≥ 3/2) :=
by
  intro x
  sorry

-- (Ⅱ) Prove that if there exists x ∈ ℝ such that f(x) < 2, then -2 < a < 2
theorem problem_2 :
  (∃ (x : ℝ), f x a < 2) → -2 < a ∧ a < 2 :=
by
  intro h
  sorry

end problem_1_problem_2_l257_257255


namespace find_other_number_l257_257557

theorem find_other_number (w : ℕ) (x : ℕ) 
    (h1 : w = 468)
    (h2 : x * w = 2^4 * 3^3 * 13^3) 
    : x = 2028 :=
by
  sorry

end find_other_number_l257_257557


namespace number_of_truth_tellers_is_twelve_l257_257904
noncomputable theory

section
variables (x : ℕ)
variables (y : ℕ)
variables (a b c d : ℕ)

-- Given conditions
def total_warriors : ℕ := 33
def yes_sword : ℕ := 13
def yes_spear : ℕ := 15
def yes_axe : ℕ := 20
def yes_bow : ℕ := 27

-- Calculate total yes answers (condition 6)
def total_yes_answers : ℕ := yes_sword + yes_spear + yes_axe + yes_bow

-- Truth-tellers say "yes" to only one question, liers say "yes" to three questions
def truth_yes : ℕ := x * 1
def lie_yes : ℕ := (total_warriors - x) * 3

theorem number_of_truth_tellers_is_twelve
  (h1 : total_warriors = 33)
  (h2 : yes_sword = 13)
  (h3 : yes_spear = 15)
  (h4 : yes_axe = 20)
  (h5 : yes_bow = 27)
  (h6 : total_yes_answers = 75)
  (h7 : total_yes_answers = truth_yes + lie_yes) :
    x = 12 :=
  by sorry

end

end number_of_truth_tellers_is_twelve_l257_257904


namespace Rachel_homework_difference_l257_257810

theorem Rachel_homework_difference (m r : ℕ) (hm : m = 8) (hr : r = 14) : r - m = 6 := 
by 
  sorry

end Rachel_homework_difference_l257_257810


namespace playground_perimeter_l257_257602

-- Defining the conditions
def length : ℕ := 100
def breadth : ℕ := 500
def perimeter (L B : ℕ) : ℕ := 2 * (L + B)

-- The theorem to prove
theorem playground_perimeter : perimeter length breadth = 1200 := 
by
  -- The actual proof will be filled later
  sorry

end playground_perimeter_l257_257602


namespace find_first_number_l257_257319

theorem find_first_number (x : ℝ) :
  (20 + 40 + 60) / 3 = (x + 70 + 13) / 3 + 9 → x = 10 :=
by
  sorry

end find_first_number_l257_257319


namespace solve_for_x_l257_257822

theorem solve_for_x (x : ℝ) : x^2 + 6 * x + 8 = -(x + 4) * (x + 6) ↔ x = -4 :=
by {
  sorry
}

end solve_for_x_l257_257822


namespace min_abs_sum_l257_257473

theorem min_abs_sum (x : ℝ) : ∃ x : ℝ, (∀ y, abs (y + 3) + abs (y - 2) ≥ abs (x + 3) + abs (x - 2)) ∧ (abs (x + 3) + abs (x - 2) = 5) := sorry

end min_abs_sum_l257_257473


namespace shopkeeper_loss_amount_l257_257858

theorem shopkeeper_loss_amount (total_stock_worth : ℝ)
                               (portion_sold_at_profit : ℝ)
                               (portion_sold_at_loss : ℝ)
                               (profit_percentage : ℝ)
                               (loss_percentage : ℝ) :
  total_stock_wworth = 14999.999999999996 →
  portion_sold_at_profit = 0.2 →
  portion_sold_at_loss = 0.8 →
  profit_percentage = 0.10 →
  loss_percentage = 0.05 →
  (total_stock_worth - ((portion_sold_at_profit * total_stock_worth * (1 + profit_percentage)) + 
                        (portion_sold_at_loss * total_stock_worth * (1 - loss_percentage)))) = 300 := 
by 
  sorry

end shopkeeper_loss_amount_l257_257858


namespace bogatyrs_truthful_count_l257_257897

noncomputable def number_of_truthful_warriors (total_warriors: ℕ) (sword_yes: ℕ) (spear_yes: ℕ) (axe_yes: ℕ) (bow_yes: ℕ) : ℕ :=
  let total_yes := sword_yes + spear_yes + axe_yes + bow_yes
  let lying_warriors := (total_yes - total_warriors) / 2
  total_warriors - lying_warriors

theorem bogatyrs_truthful_count :
  number_of_truthful_warriors 33 13 15 20 27 = 12 := by
  sorry

end bogatyrs_truthful_count_l257_257897


namespace gcd_lcm_sum_l257_257839

variable (a b : ℕ)

-- Definition for gcd
def gcdOf (a b : ℕ) : ℕ := Nat.gcd a b

-- Definition for lcm
def lcmOf (a b : ℕ) : ℕ := Nat.lcm a b

-- Statement of the problem
theorem gcd_lcm_sum (h1 : a = 8) (h2 : b = 12) : gcdOf a b + lcmOf a b = 28 := by
  sorry

end gcd_lcm_sum_l257_257839


namespace find_x_value_l257_257120

noncomputable def log (a b: ℝ): ℝ := Real.log a / Real.log b

theorem find_x_value (a n : ℝ) (t y: ℝ):
  1 < a →
  1 < t →
  y = 8 →
  log n (a^t) - 3 * log a (a^t) * log y 8 = 3 →
  x = a^t →
  x = a^2 :=
by
  sorry

end find_x_value_l257_257120


namespace books_sold_to_used_bookstore_l257_257579

-- Conditions
def initial_books := 72
def books_from_club := 1 * 12
def books_from_bookstore := 5
def books_from_yardsales := 2
def books_from_daughter := 1
def books_from_mother := 4
def books_donated := 12
def books_end_of_year := 81

-- Proof problem
theorem books_sold_to_used_bookstore :
  initial_books
  + books_from_club
  + books_from_bookstore
  + books_from_yardsales
  + books_from_daughter
  + books_from_mother
  - books_donated
  - books_end_of_year
  = 3 := by
  -- calculation omitted
  sorry

end books_sold_to_used_bookstore_l257_257579


namespace canoes_more_than_kayaks_l257_257334

noncomputable def canoes_difference (C K : ℕ) : Prop :=
  15 * C + 18 * K = 405 ∧ 2 * C = 3 * K → C - K = 5

theorem canoes_more_than_kayaks (C K : ℕ) : canoes_difference C K :=
by
  sorry

end canoes_more_than_kayaks_l257_257334


namespace largest_multiple_of_7_negation_greater_than_neg_150_l257_257340

theorem largest_multiple_of_7_negation_greater_than_neg_150 : 
  ∃ k : ℤ, k * 7 = 147 ∧ ∀ n : ℤ, (k < n → n * 7 ≤ 150) :=
by
  use 21
  sorry

end largest_multiple_of_7_negation_greater_than_neg_150_l257_257340


namespace smallest_special_gt_3429_l257_257103

def is_special (n : ℕ) : Prop :=
  (10^3 ≤ n ∧ n < 10^4) ∧ (List.length (n.digits 10).eraseDup = 4)

theorem smallest_special_gt_3429 : 
  ∃ n : ℕ, n > 3429 ∧ is_special n ∧ ∀ m : ℕ, m > 3429 ∧ is_special m → n ≤ m := 
begin
  use 3450,
  split,
  { exact nat.succ_lt_succ (nat.s succ_lt_succ (nat.succ_lt_succ (nat.succ_lt_succ (nat.lt_succ_self 3430)))) },
  split,
  { unfold is_special,
    split,
    { split,
        { linarith },
    { linarith } },
    { unfold List.eraseDup,
    unfold List.redLength,
    exactly simp } },
  { intros m hm1 hm2,
    interval_cases m,
    sorry },
end

end smallest_special_gt_3429_l257_257103


namespace base8_to_base10_l257_257075

theorem base8_to_base10 (n : ℕ) : of_digits 8 [2, 4, 6] = 166 := by
  sorry

end base8_to_base10_l257_257075


namespace largest_multiple_of_7_gt_neg_150_l257_257354

theorem largest_multiple_of_7_gt_neg_150 : ∃ (x : ℕ), (x % 7 = 0) ∧ ((- (x : ℤ)) > -150) ∧ ∀ y : ℕ, (y % 7 = 0 ∧ (- (y : ℤ)) > -150) → y ≤ x :=
by
  sorry

end largest_multiple_of_7_gt_neg_150_l257_257354


namespace largest_multiple_of_7_negation_gt_neg150_l257_257349

theorem largest_multiple_of_7_negation_gt_neg150 : 
  ∃ (k : ℤ), (k % 7 = 0 ∧ -k > -150 ∧ ∀ (m : ℤ), (m % 7 = 0 ∧ -m > -150 → m ≤ k)) :=
sorry

end largest_multiple_of_7_negation_gt_neg150_l257_257349


namespace product_of_possible_values_of_b_l257_257002

theorem product_of_possible_values_of_b :
  let y₁ := -1
  let y₂ := 4
  let x₁ := 1
  let side_length := y₂ - y₁ -- Since this is 5 units
  let b₁ := x₁ - side_length -- This should be -4
  let b₂ := x₁ + side_length -- This should be 6
  let product := b₁ * b₂ -- So, (-4) * 6
  product = -24 :=
by
  sorry

end product_of_possible_values_of_b_l257_257002


namespace base8_246_is_166_in_base10_l257_257069

def convert_base8_to_base10 (n : ℕ) : ℕ :=
  let d0 := n % 10;
  let d1 := (n / 10) % 10;
  let d2 := (n / 100) % 10;
  d2 * 8^2 + d1 * 8^1 + d0 * 8^0

theorem base8_246_is_166_in_base10 : convert_base8_to_base10 246 = 166 :=
  sorry

end base8_246_is_166_in_base10_l257_257069


namespace arccos_one_over_sqrt_two_l257_257705

theorem arccos_one_over_sqrt_two : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l257_257705


namespace smallest_special_gt_3429_l257_257093

def is_special (n : ℕ) : Prop :=
  let digits := (n.digits 10).erase_dup
  digits.length = 4

theorem smallest_special_gt_3429 : ∃ n : ℕ, n > 3429 ∧ is_special n ∧ ∀ m : ℕ, m > 3429 ∧ is_special m → n ≤ m :=
by
  use 3450
  constructor
  . exact nat.lt_succ_self 3429  -- 3450 > 3429
  constructor
  . unfold is_special
    dsimp
    norm_num
  . intro m
    intro h
    intro hspec
    sorry

end smallest_special_gt_3429_l257_257093


namespace find_number_l257_257633

theorem find_number (x : ℤ) (h : x + x^2 = 342) : x = 18 ∨ x = -19 :=
sorry

end find_number_l257_257633


namespace truthful_warriors_count_l257_257911

-- Noncomputable theory is not necessary here.
-- We define the number of warriors and their responses.
def warriors_count : ℕ := 33
def yes_sword : ℕ := 13
def yes_spear : ℕ := 15
def yes_axe : ℕ := 20
def yes_bow : ℕ := 27

-- The equation from the solution steps where x is the number of truthful warriors.
def total_yes := yes_sword + yes_spear + yes_axe + yes_bow

theorem truthful_warriors_count :
  ∃ x : ℕ, x + 3 * (warriors_count - x) = total_yes ∧ x = 12 :=
by {
  -- We state that there exists an x such that its value satisfies the total "yes" responses equation.
  -- Here, we assert that x equals 12.
  use 12,
  split,
  {
    -- Prove the equation x + 3 * (warriors_count - x) = total_yes
    rw [warriors_count],
    rw [total_yes],
    norm_num,
  },
  {
    -- State that x = 12
    refl,
  },
}

end truthful_warriors_count_l257_257911


namespace arccos_identity_l257_257731

theorem arccos_identity : 
  ∀ x : ℝ, x = 1 / real.sqrt 2 → real.arccos x = real.pi / 4 :=
begin
  intros x hx,
  have h1 : real.cos (real.pi / 4) = 1 / real.sqrt 2 := by sorry,
  have h2 : real.arccos (1 / real.sqrt 2) = real.pi / 4,
  { rw ← h1,
    rw real.arccos_cos,
    exact_mod_cast hx },
  exact h2,
end

end arccos_identity_l257_257731


namespace intersection_of_M_and_N_l257_257770

def M : Set ℝ := {x | 0 < x ∧ x < 3}
def N : Set ℝ := {x | x > 2 ∨ x < -2}
def expected_intersection : Set ℝ := {x | 2 < x ∧ x < 3}

theorem intersection_of_M_and_N : M ∩ N = expected_intersection := by
  sorry

end intersection_of_M_and_N_l257_257770


namespace find_abc_integers_l257_257930

theorem find_abc_integers (a b c : ℕ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) 
(h4 : (a - 1) * (b - 1) * (c - 1) ∣ a * b * c - 1) : (a = 3 ∧ b = 5 ∧ c = 15) ∨ 
(a = 2 ∧ b = 4 ∧ c = 8) :=
sorry

end find_abc_integers_l257_257930


namespace evelyn_lost_bottle_caps_l257_257110

-- Definitions from the conditions
def initial_amount : ℝ := 63.0
def final_amount : ℝ := 45.0
def lost_amount : ℝ := 18.0

-- Statement to be proved
theorem evelyn_lost_bottle_caps : initial_amount - final_amount = lost_amount := 
by 
  sorry

end evelyn_lost_bottle_caps_l257_257110


namespace abcd_inequality_l257_257970

theorem abcd_inequality (a b c d : ℝ) :
  (a * c + b * d)^2 ≤ (a^2 + b^2) * (c^2 + d^2) :=
sorry

end abcd_inequality_l257_257970


namespace max_food_per_guest_l257_257828

theorem max_food_per_guest (total_food : ℕ) (min_guests : ℕ)
    (H1 : total_food = 406) (H2 : min_guests = 163) :
    2 ≤ total_food / min_guests ∧ total_food / min_guests < 3 := by
  sorry

end max_food_per_guest_l257_257828


namespace dot_product_in_triangle_l257_257154

noncomputable def ab := 3
noncomputable def ac := 2
noncomputable def bc := Real.sqrt 10

theorem dot_product_in_triangle : 
  let AB := ab
  let AC := ac
  let BC := bc
  (AB = 3) → (AC = 2) → (BC = Real.sqrt 10) → 
  ∃ cosA, (cosA = (AB^2 + AC^2 - BC^2) / (2 * AB * AC)) →
  ∃ dot_product, (dot_product = AB * AC * cosA) ∧ dot_product = 3 / 2 :=
by
  sorry

end dot_product_in_triangle_l257_257154


namespace contest_score_difference_l257_257155

theorem contest_score_difference :
  let percent_50 := 0.05
  let percent_60 := 0.20
  let percent_70 := 0.25
  let percent_80 := 0.30
  let percent_90 := 1 - (percent_50 + percent_60 + percent_70 + percent_80)
  let mean := (percent_50 * 50) + (percent_60 * 60) + (percent_70 * 70) + (percent_80 * 80) + (percent_90 * 90)
  let median := 70
  median - mean = -4 :=
by
  sorry

end contest_score_difference_l257_257155


namespace least_pennies_l257_257031

theorem least_pennies : 
  ∃ (a : ℕ), a % 5 = 1 ∧ a % 3 = 2 ∧ a = 11 :=
by
  sorry

end least_pennies_l257_257031


namespace sum_of_cubes_eq_neg_27_l257_257797

variable {a b c : ℝ}

-- Define the condition that k is the same for a, b, and c
def same_k (a b c k : ℝ) : Prop :=
  k = (a^3 + 9) / a ∧ k = (b^3 + 9) / b ∧ k = (c^3 + 9) / c

-- Theorem: Given the conditions, a^3 + b^3 + c^3 = -27
theorem sum_of_cubes_eq_neg_27 (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_same_k : ∃ k, same_k a b c k) :
  a^3 + b^3 + c^3 = -27 :=
sorry

end sum_of_cubes_eq_neg_27_l257_257797


namespace base8_246_is_166_in_base10_l257_257067

def convert_base8_to_base10 (n : ℕ) : ℕ :=
  let d0 := n % 10;
  let d1 := (n / 10) % 10;
  let d2 := (n / 100) % 10;
  d2 * 8^2 + d1 * 8^1 + d0 * 8^0

theorem base8_246_is_166_in_base10 : convert_base8_to_base10 246 = 166 :=
  sorry

end base8_246_is_166_in_base10_l257_257067


namespace algebraic_expression_eval_l257_257170

theorem algebraic_expression_eval (a b : ℝ) 
  (h_eq : ∀ (x : ℝ), ¬(x ≠ 0 ∧ x ≠ 1 ∧ (x / (x - 1) + (x - 1) / x = (a + b * x) / (x^2 - x)))) :
  8 * a + 4 * b - 5 = 27 := 
sorry

end algebraic_expression_eval_l257_257170


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l257_257674

theorem arccos_one_over_sqrt_two_eq_pi_over_four : 
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l257_257674


namespace log9_log11_lt_one_l257_257057

theorem log9_log11_lt_one (log9_pos : 0 < Real.log 9) (log11_pos : 0 < Real.log 11) : 
  Real.log 9 * Real.log 11 < 1 :=
by
  sorry

end log9_log11_lt_one_l257_257057


namespace common_ratio_l257_257541

variable {a : ℕ → ℝ} -- Define a as a sequence of real numbers

-- Define the conditions as hypotheses
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = q * a n

variables (q : ℝ) (h1 : a 2 = 2) (h2 : a 5 = 1 / 4)

-- Define the theorem to prove the common ratio
theorem common_ratio (h_geom : is_geometric_sequence a q) : q = 1 / 2 :=
  sorry

end common_ratio_l257_257541


namespace max_bishops_on_chessboard_l257_257198

theorem max_bishops_on_chessboard : ∃ n : ℕ, n = 14 ∧ (∃ k : ℕ, n * n = k^2) := 
by {
  sorry
}

end max_bishops_on_chessboard_l257_257198


namespace intersection_of_A_and_B_l257_257539

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | x ≥ 2}

theorem intersection_of_A_and_B :
  (A ∩ B) = {2} := 
by {
  sorry
}

end intersection_of_A_and_B_l257_257539


namespace distance_to_conference_l257_257439

theorem distance_to_conference (t d : ℝ) 
  (h1 : d = 40 * (t + 0.75))
  (h2 : d - 40 = 60 * (t - 1.25)) :
  d = 160 :=
by
  sorry

end distance_to_conference_l257_257439


namespace base8_to_base10_l257_257077

theorem base8_to_base10 (n : ℕ) : of_digits 8 [2, 4, 6] = 166 := by
  sorry

end base8_to_base10_l257_257077


namespace value_of_expression_l257_257966

variable (p q r s : ℝ)

-- Given condition in a)
def polynomial_function (x : ℝ) := p * x^3 + q * x^2 + r * x + s
def passes_through_point := polynomial_function p q r s (-1) = 4

-- Proof statement in c)
theorem value_of_expression (h : passes_through_point p q r s) : 6 * p - 3 * q + r - 2 * s = -24 := by
  sorry

end value_of_expression_l257_257966


namespace arccos_sqrt_half_l257_257716

theorem arccos_sqrt_half : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := by
  sorry

end arccos_sqrt_half_l257_257716


namespace minimum_period_l257_257259

noncomputable def f (ω x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 6)

theorem minimum_period (ω : ℝ) (hω : ω > 0) 
  (h : ∀ x1 x2 : ℝ, |f ω x1 - f ω x2| = 2 → |x1 - x2| = Real.pi / 2) :
  ∃ T > 0, ∀ x : ℝ, f ω (x + T) = f ω x ∧ T = Real.pi := sorry

end minimum_period_l257_257259


namespace slope_of_line_l257_257532

theorem slope_of_line (s x y : ℝ) (h1 : 2 * x + 3 * y = 8 * s + 5) (h2 : x + 2 * y = 3 * s + 2) :
  ∃ m c : ℝ, ∀ x y, x = m * y + c ∧ m = -7/2 :=
by
  sorry

end slope_of_line_l257_257532


namespace largest_neg_multiple_of_7_greater_than_neg_150_l257_257358

theorem largest_neg_multiple_of_7_greater_than_neg_150 : 
  ∃ (n : ℤ), (n % 7 = 0) ∧ (-n > -150) ∧ (∀ m : ℤ, (m % 7 = 0) ∧ (-m > -150) → m ≤ n) :=
begin
  use 147,
  split,
  { norm_num }, -- Verifies that 147 is a multiple of 7
  split,
  { norm_num }, -- Verifies that -147 > -150
  { intros m h,
    obtain ⟨k, rfl⟩ := (zmod.int_coe_zmod_eq_zero_iff_dvd m 7).mp h.1,
    suffices : k ≤ 21, { rwa [int.nat_abs_of_nonneg (by norm_num : (7 : ℤ) ≥ 0), ←abs_eq_nat_abs, int.abs_eq_nat_abs, nat.abs_of_nonneg (zero_le 21), ← int.le_nat_abs_iff_coe_nat_le] at this },
    have : -m > -150 := h.2,
    rwa [int.lt_neg, neg_le_neg_iff] at this,
    norm_cast at this,
    exact this
  }
end

end largest_neg_multiple_of_7_greater_than_neg_150_l257_257358


namespace sprinter_speed_l257_257378

theorem sprinter_speed
  (distance : ℝ)
  (time : ℝ)
  (H1 : distance = 100)
  (H2 : time = 10) :
    (distance / time = 10) ∧
    ((distance / time) * 60 = 600) ∧
    (((distance / time) * 60 * 60) / 1000 = 36) :=
by
  sorry

end sprinter_speed_l257_257378


namespace unique_solution_l257_257246

theorem unique_solution:
  ∃! (x y z : ℕ), 2^x + 9 * 7^y = z^3 ∧ x = 0 ∧ y = 1 ∧ z = 4 :=
by
  sorry

end unique_solution_l257_257246


namespace point_in_third_quadrant_l257_257159

theorem point_in_third_quadrant (x y : ℤ) (hx : x = -8) (hy : y = -3) : (x < 0) ∧ (y < 0) :=
by
  have hx_neg : x < 0 := by rw [hx]; norm_num
  have hy_neg : y < 0 := by rw [hy]; norm_num
  exact ⟨hx_neg, hy_neg⟩

end point_in_third_quadrant_l257_257159


namespace smallest_special_number_l257_257087

def is_special (n : ℕ) : Prop :=
  (∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
   n = a * 1000 + b * 100 + c * 10 + d)

theorem smallest_special_number (n : ℕ) (h1 : n > 3429) (h2 : is_special n) : n = 3450 :=
sorry

end smallest_special_number_l257_257087


namespace warriors_truth_tellers_l257_257893

/-- There are 33 warriors. Each warrior is either a truth-teller or a liar, 
    with only one favorite weapon: a sword, a spear, an axe, or a bow. 
    They were asked four questions, and the number of "Yes" answers to the 
    questions are 13, 15, 20, and 27 respectively. Prove that the number of 
    warriors who always tell the truth is 12. -/
theorem warriors_truth_tellers
  (warriors : ℕ) (truth_tellers : ℕ)
  (yes_to_sword : ℕ) (yes_to_spear : ℕ)
  (yes_to_axe : ℕ) (yes_to_bow : ℕ)
  (h1 : warriors = 33)
  (h2 : yes_to_sword = 13)
  (h3 : yes_to_spear = 15)
  (h4 : yes_to_axe = 20)
  (h5 : yes_to_bow = 27)
  (h6 : yes_to_sword + yes_to_spear + yes_to_axe + yes_to_bow = 75) :
  truth_tellers = 12 := by
  -- Proof will be here
  sorry

end warriors_truth_tellers_l257_257893


namespace arccos_sqrt_half_l257_257714

theorem arccos_sqrt_half : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := by
  sorry

end arccos_sqrt_half_l257_257714


namespace min_value_eq_six_l257_257278

theorem min_value_eq_six
    (α β : ℝ)
    (k : ℝ)
    (h1 : α^2 + 2 * (k + 3) * α + (k^2 + 3) = 0)
    (h2 : β^2 + 2 * (k + 3) * β + (k^2 + 3) = 0)
    (h3 : (2 * (k + 3))^2 - 4 * (k^2 + 3) ≥ 0) :
    ( (α - 1)^2 + (β - 1)^2 = 6 ) := 
sorry

end min_value_eq_six_l257_257278


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l257_257675

theorem arccos_one_over_sqrt_two_eq_pi_over_four : 
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l257_257675


namespace selling_price_increase_solution_maximum_profit_solution_l257_257502

-- Conditions
def purchase_price : ℝ := 30
def original_price : ℝ := 40
def monthly_sales : ℝ := 300
def sales_decrease_per_yuan : ℝ := 10

-- Questions
def selling_price_increase (x : ℝ) : Prop :=
  (x + 10) * (monthly_sales - sales_decrease_per_yuan * x) = 3360

def maximum_profit (x : ℝ) : Prop :=
  ∃ x : ℝ, 
    let M := -10 * x^2 + 200 * x + 3000 in
    M = 4000 ∧ x = 10

theorem selling_price_increase_solution : ∃ x : ℝ, selling_price_increase x := sorry

theorem maximum_profit_solution : ∃ x : ℝ, maximum_profit x := sorry

end selling_price_increase_solution_maximum_profit_solution_l257_257502


namespace expected_area_projection_l257_257936

noncomputable def expectedProjectionArea (cube_edge : ℝ) : ℝ := 
  3 / 2

theorem expected_area_projection (h : ∀ (c : ℝ), c = 1) : expectedProjectionArea 1 = 3 / 2 :=
by 
  rw expectedProjectionArea 
  simp
  sorry

end expected_area_projection_l257_257936


namespace range_of_m_l257_257544

theorem range_of_m (f : ℝ → ℝ) 
  (Hmono : ∀ x y, -2 ≤ x → x ≤ 2 → -2 ≤ y → y ≤ 2 → x ≤ y → f x ≤ f y)
  (Hineq : ∀ m, f (Real.log m / Real.log 2) < f (Real.log (m + 2) / Real.log 4))
  : ∀ m, (1 / 4 : ℝ) ≤ m ∧ m < 2 :=
sorry

end range_of_m_l257_257544


namespace only_composite_positive_integer_with_divisors_form_l257_257244

theorem only_composite_positive_integer_with_divisors_form (n : ℕ) (composite : ¬Nat.Prime n ∧ 1 < n)
  (H : ∀ d ∈ Nat.divisors n, ∃ (a r : ℕ), a ≥ 0 ∧ r ≥ 2 ∧ d = a^r + 1) : n = 10 :=
by
  sorry

end only_composite_positive_integer_with_divisors_form_l257_257244


namespace arccos_one_over_sqrt_two_l257_257743

theorem arccos_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l257_257743


namespace union_of_sets_l257_257949

noncomputable def setA : Set ℝ := { x : ℝ | x^3 - 3 * x^2 - x + 3 < 0 }
noncomputable def setB : Set ℝ := { x : ℝ | abs (x + 1/2) >= 1 }

theorem union_of_sets:
  setA ∪ setB = (Iio (-1) ∪ Ici (1/2)) :=
by
  sorry

end union_of_sets_l257_257949


namespace find_acute_angles_of_alex_triangle_l257_257516

theorem find_acute_angles_of_alex_triangle (α : ℝ) (h1 : α > 0) (h2 : α < 90) :
  let condition1 := «Alex drew a geometric picture by tracing his plastic right triangle four times»
  let condition2 := «Each time aligning the shorter leg with the hypotenuse and matching the vertex of the acute angle with the vertex of the right angle»
  let condition3 := «The "closing" fifth triangle was isosceles»
  α = 90 / 11 :=
sorry

end find_acute_angles_of_alex_triangle_l257_257516


namespace quarterly_production_growth_l257_257500

theorem quarterly_production_growth (P_A P_Q2 : ℕ) (x : ℝ)
  (hA : P_A = 500000)
  (hQ2 : P_Q2 = 1820000) :
  50 + 50 * (1 + x) + 50 * (1 + x)^2 = 182 :=
by 
  sorry

end quarterly_production_growth_l257_257500


namespace solve_x_l257_257818

theorem solve_x (x : ℝ) (h : x^2 + 6 * x + 8 = -(x + 4) * (x + 6)) : 
  x = -4 := 
by
  sorry

end solve_x_l257_257818


namespace number_of_truthful_warriors_l257_257881

theorem number_of_truthful_warriors (total_warriors : ℕ) 
  (sword_yes : ℕ) (spear_yes : ℕ) (axe_yes : ℕ) (bow_yes : ℕ) 
  (always_tells_truth : ℕ → Prop)
  (always_lies : ℕ → Prop)
  (hv1 : total_warriors = 33)
  (hv2 : sword_yes = 13)
  (hv3 : spear_yes = 15)
  (hv4 : axe_yes = 20)
  (hv5 : bow_yes = 27) :
  ∃ truthful_warriors, truthful_warriors = 12 := 
by {
  sorry
}

end number_of_truthful_warriors_l257_257881


namespace largest_neg_multiple_of_7_greater_than_neg_150_l257_257359

theorem largest_neg_multiple_of_7_greater_than_neg_150 : 
  ∃ (n : ℤ), (n % 7 = 0) ∧ (-n > -150) ∧ (∀ m : ℤ, (m % 7 = 0) ∧ (-m > -150) → m ≤ n) :=
begin
  use 147,
  split,
  { norm_num }, -- Verifies that 147 is a multiple of 7
  split,
  { norm_num }, -- Verifies that -147 > -150
  { intros m h,
    obtain ⟨k, rfl⟩ := (zmod.int_coe_zmod_eq_zero_iff_dvd m 7).mp h.1,
    suffices : k ≤ 21, { rwa [int.nat_abs_of_nonneg (by norm_num : (7 : ℤ) ≥ 0), ←abs_eq_nat_abs, int.abs_eq_nat_abs, nat.abs_of_nonneg (zero_le 21), ← int.le_nat_abs_iff_coe_nat_le] at this },
    have : -m > -150 := h.2,
    rwa [int.lt_neg, neg_le_neg_iff] at this,
    norm_cast at this,
    exact this
  }
end

end largest_neg_multiple_of_7_greater_than_neg_150_l257_257359


namespace second_discount_is_5_percent_l257_257606

noncomputable def salePriceSecondDiscount (initialPrice finalPrice priceAfterFirstDiscount: ℝ) : ℝ :=
  (initialPrice - priceAfterFirstDiscount) + (priceAfterFirstDiscount - finalPrice)

noncomputable def secondDiscountPercentage (initialPrice finalPrice priceAfterFirstDiscount: ℝ) : ℝ :=
  (priceAfterFirstDiscount - finalPrice) / priceAfterFirstDiscount * 100

theorem second_discount_is_5_percent :
  ∀ (initialPrice finalPrice priceAfterFirstDiscount: ℝ),
    initialPrice = 600 ∧
    finalPrice = 456 ∧
    priceAfterFirstDiscount = initialPrice * 0.80 →
    secondDiscountPercentage initialPrice finalPrice priceAfterFirstDiscount = 5 :=
by
  intros
  sorry

end second_discount_is_5_percent_l257_257606


namespace apple_cost_price_orange_cost_price_banana_cost_price_l257_257569

theorem apple_cost_price (A : ℚ) : 15 = A - (1/6 * A) → A = 18 := by
  intro h
  sorry

theorem orange_cost_price (O : ℚ) : 20 = O + (1/5 * O) → O = 100/6 := by
  intro h
  sorry

theorem banana_cost_price (B : ℚ) : 10 = B → B = 10 := by
  intro h
  sorry

end apple_cost_price_orange_cost_price_banana_cost_price_l257_257569


namespace total_bricks_calculation_l257_257837

def bricks_in_row : Nat := 30
def rows_in_wall : Nat := 50
def number_of_walls : Nat := 2
def total_bricks_for_both_walls : Nat := 3000

theorem total_bricks_calculation (h1 : bricks_in_row = 30) 
                                      (h2 : rows_in_wall = 50) 
                                      (h3 : number_of_walls = 2) : 
                                      bricks_in_row * rows_in_wall * number_of_walls = total_bricks_for_both_walls :=
by
  sorry

end total_bricks_calculation_l257_257837


namespace probability_one_instrument_l257_257204

-- Definitions based on conditions
def total_people : Nat := 800
def play_at_least_one : Nat := total_people / 5
def play_two_or_more : Nat := 32
def play_exactly_one : Nat := play_at_least_one - play_two_or_more

-- Target statement to prove the equivalence
theorem probability_one_instrument: (play_exactly_one : ℝ) / (total_people : ℝ) = 0.16 := by
  sorry

end probability_one_instrument_l257_257204


namespace correct_subtraction_l257_257628

theorem correct_subtraction (x : ℕ) (h : x - 63 = 8) : x - 36 = 35 :=
by sorry

end correct_subtraction_l257_257628


namespace friends_with_Ron_l257_257813

-- Ron is eating pizza with his friends 
def total_slices : Nat := 12
def slices_per_person : Nat := 4
def total_people := total_slices / slices_per_person
def ron_included := 1

theorem friends_with_Ron : total_people - ron_included = 2 := by
  sorry

end friends_with_Ron_l257_257813


namespace value_of_m_plus_n_l257_257136

noncomputable def exponential_function (a x m n : ℝ) : ℝ :=
  a^(x - m) + n - 3

theorem value_of_m_plus_n (a x m n y : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1)
  (h₃ : exponential_function a 3 m n = 2) : m + n = 7 :=
by
  sorry

end value_of_m_plus_n_l257_257136


namespace minValue_Proof_l257_257574

noncomputable def minValue (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 2) : Prop :=
  ∃ m : ℝ, m = 4.5 ∧ (∀ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 2 → (1/a + 1/b + 1/c) ≥ 9/2)

theorem minValue_Proof :
  ∀ (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 2), 
    minValue x y z h1 h2 h3 h4 := by
  sorry

end minValue_Proof_l257_257574


namespace sum_of_roots_combined_eq_five_l257_257938

noncomputable def sum_of_roots_poly1 : ℝ :=
-(-9/3)

noncomputable def sum_of_roots_poly2 : ℝ :=
-(-8/4)

theorem sum_of_roots_combined_eq_five :
  sum_of_roots_poly1 + sum_of_roots_poly2 = 5 :=
by
  sorry

end sum_of_roots_combined_eq_five_l257_257938


namespace cycle_original_cost_l257_257041

theorem cycle_original_cost (SP : ℝ) (gain : ℝ) (CP : ℝ) (h₁ : SP = 2000) (h₂ : gain = 1) (h₃ : SP = CP * (1 + gain)) : CP = 1000 :=
by
  sorry

end cycle_original_cost_l257_257041


namespace arccos_sqrt2_l257_257690

def arccos_eq (x : ℝ) := arccos x
def range_arccos := ∀ (x : ℝ), 0 ≤ arccos x ∧ arccos x ≤ π
def cos_pi_div_four : Real.cos (π / 4) = 1 / Real.sqrt 2 := by sorry

theorem arccos_sqrt2 :
  arccos_eq (1 / Real.sqrt 2) = π / 4 :=
by
  have h1 : Real.cos (π / 4) = 1 / Real.sqrt 2, from cos_pi_div_four
  sorry

end arccos_sqrt2_l257_257690


namespace sum_third_largest_and_smallest_l257_257367

-- Condition 1: Digits available are 7, 4, 0, 3, and 5
def digits : List ℕ := [7, 4, 0, 3, 5]

-- Condition 2 and 3: Form two-digit numbers with distinct digits, and 0 cannot be the first digit.
def valid_two_digit (a b : ℕ) : Bool :=
  a ≠ b ∧ a ≠ 0 

-- A function to generate all valid two-digit numbers from the given digits
def two_digit_numbers : List ℕ :=
  (digits.product digits).filter (λ pair => valid_two_digit pair.fst pair.snd = true).map (λ pair => 10 * pair.fst + pair.snd)

-- All valid two-digit numbers (as per the constraints)
def sorted_desc := two_digit_numbers.qsort (λ x y => x > y)
def sorted_asc := two_digit_numbers.qsort (λ x y => x < y)

-- Define the third largest and third smallest values
def third_largest := sorted_desc.nth 2
def third_smallest := sorted_asc.nth 2

-- The proof statement
theorem sum_third_largest_and_smallest : third_largest + third_smallest = 108 :=
by
  -- Placeholder for proof
  sorry

end sum_third_largest_and_smallest_l257_257367


namespace arccos_identity_l257_257729

theorem arccos_identity : 
  ∀ x : ℝ, x = 1 / real.sqrt 2 → real.arccos x = real.pi / 4 :=
begin
  intros x hx,
  have h1 : real.cos (real.pi / 4) = 1 / real.sqrt 2 := by sorry,
  have h2 : real.arccos (1 / real.sqrt 2) = real.pi / 4,
  { rw ← h1,
    rw real.arccos_cos,
    exact_mod_cast hx },
  exact h2,
end

end arccos_identity_l257_257729


namespace g_1000_is_1820_l257_257827

-- Definitions and conditions from the problem
def g (n : ℕ) : ℕ := sorry -- exact definition is unknown, we will assume conditions

-- Conditions as given
axiom g_g (n : ℕ) : g (g n) = 3 * n
axiom g_3n_plus_1 (n : ℕ) : g (3 * n + 1) = 3 * n + 2

-- Statement to prove
theorem g_1000_is_1820 : g 1000 = 1820 :=
by
  sorry

end g_1000_is_1820_l257_257827


namespace max_sequence_value_l257_257113

theorem max_sequence_value : 
  ∃ n ∈ (Set.univ : Set ℤ), (∀ m ∈ (Set.univ : Set ℤ), -m^2 + 15 * m + 3 ≤ -n^2 + 15 * n + 3) ∧ (-n^2 + 15 * n + 3 = 59) :=
by
  sorry

end max_sequence_value_l257_257113


namespace max_value_of_d_l257_257926

theorem max_value_of_d : ∀ (d e : ℕ), (∃ (n : ℕ), n = 70733 + 10^4 * d + e ∧ (∃ (k3 k11 : ℤ), n = 3 * k3 ∧ n = 11 * k11) ∧ d = e ∧ d ≤ 9) → d = 2 :=
by 
  -- Given conditions and goals:
  -- 1. The number has the form 7d7,33e which in numerical form is: n = 70733 + 10^4 * d + e
  -- 2. The number n is divisible by 3 and 11.
  -- 3. d and e are digits (0 ≤ d, e ≤ 9).
  -- 4. To maximize the value of d, ensure that the given conditions hold.
  -- Problem: Prove that the maximum value of d for which this holds is 2.
  sorry

end max_value_of_d_l257_257926


namespace arccos_sqrt2_l257_257693

def arccos_eq (x : ℝ) := arccos x
def range_arccos := ∀ (x : ℝ), 0 ≤ arccos x ∧ arccos x ≤ π
def cos_pi_div_four : Real.cos (π / 4) = 1 / Real.sqrt 2 := by sorry

theorem arccos_sqrt2 :
  arccos_eq (1 / Real.sqrt 2) = π / 4 :=
by
  have h1 : Real.cos (π / 4) = 1 / Real.sqrt 2, from cos_pi_div_four
  sorry

end arccos_sqrt2_l257_257693


namespace point_distance_l257_257509

theorem point_distance (x : ℤ) : abs x = 2021 → (x = 2021 ∨ x = -2021) := 
sorry

end point_distance_l257_257509


namespace t_shirt_price_increase_t_shirt_max_profit_l257_257504

theorem t_shirt_price_increase (x : ℝ) : (x + 10) * (300 - 10 * x) = 3360 → x = 2 := 
by 
  sorry

theorem t_shirt_max_profit (x : ℝ) : (-10 * x^2 + 200 * x + 3000) = 4000 ↔ x = 10 := 
by 
  sorry

end t_shirt_price_increase_t_shirt_max_profit_l257_257504


namespace max_digit_d_of_form_7d733e_multiple_of_33_l257_257922

theorem max_digit_d_of_form_7d733e_multiple_of_33 
  (d e : ℕ) (d_digit : d < 10) (e_digit : e < 10) 
  (multiple_of_33: ∃ k : ℕ, 7 * 10^5 + d * 10^4 + 7 * 10^3 + 33 * 10 + e = 33 * k) 
  : d ≤ 6 := 
sorry

end max_digit_d_of_form_7d733e_multiple_of_33_l257_257922


namespace solution_set_of_inequality_l257_257608

theorem solution_set_of_inequality (x : ℝ) (hx : x ≠ 0) :
  (x + 1) / x ≤ 3 ↔ x ∈ Set.Iio 0 ∪ Set.Ici 0.5 :=
by sorry

end solution_set_of_inequality_l257_257608


namespace trapezoid_longer_side_length_l257_257512

theorem trapezoid_longer_side_length (x : ℝ) (h₁ : 4 = 2*2) (h₂ : ∃ AP DQ O : ℝ, ∀ (S : ℝ), 
  S = (1/2) * (x + 2) * 1 → S = 2) : 
  x = 2 :=
by sorry

end trapezoid_longer_side_length_l257_257512


namespace incorrect_option_D_l257_257957

theorem incorrect_option_D (x y : ℝ) : y = (x - 2) ^ 2 + 1 → ¬ (∀ (x : ℝ), x < 2 → y < (x - 1) ^ 2 + 1) :=
by
  intro h
  sorry

end incorrect_option_D_l257_257957


namespace incorrect_option_D_l257_257958

theorem incorrect_option_D (x y : ℝ) : y = (x - 2) ^ 2 + 1 → ¬ (∀ (x : ℝ), x < 2 → y < (x - 1) ^ 2 + 1) :=
by
  intro h
  sorry

end incorrect_option_D_l257_257958


namespace calculate_x_l257_257521

theorem calculate_x :
  529 + 2 * 23 * 11 + 121 = 1156 :=
by
  -- Begin the proof (which we won't complete here)
  -- The proof steps would go here
  sorry  -- placeholder for the actual proof steps

end calculate_x_l257_257521


namespace calculate_expression_l257_257386

theorem calculate_expression : ((18^18 / 18^17)^3 * 8^3) / 2^9 = 5832 := by
  sorry

end calculate_expression_l257_257386


namespace find_line_equation_l257_257262

theorem find_line_equation (k x y x₁ y₁ x₂ y₂ : ℝ) (h_parabola : y ^ 2 = 2 * x) 
  (h_line_ny_eq : y = k * x + 2) (h_intersect_1 : (y₁ - (k * x₁ + 2)) = 0)
  (h_intersect_2 : (y₂ - (k * x₂ + 2)) = 0) 
  (h_y_intercept : (0,2) = (x,y))-- the line has y-intercept 2 
  (h_origin : (0,0) = (x, y)) -- origin 
  (h_orthogonal : x₁ * x₂ + y₁ * y₂ = 0): 
  y = -x + 2 :=
by {
  sorry
}

end find_line_equation_l257_257262


namespace scientific_notation_of_120000_l257_257176

theorem scientific_notation_of_120000 : 
  (120000 : ℝ) = 1.2 * 10^5 := 
by 
  sorry

end scientific_notation_of_120000_l257_257176


namespace geometric_sequence_sum_first_five_terms_l257_257292

theorem geometric_sequence_sum_first_five_terms
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : a 1 + a 3 = 10)
  (h2 : a 2 + a 4 = 30)
  (h_geom : ∀ n, a (n + 1) = a n * q) :
  (a 1 + a 2 + a 3 + a 4 + a 5) = 121 :=
sorry

end geometric_sequence_sum_first_five_terms_l257_257292


namespace deposit_paid_l257_257498

variable (P : ℝ) (Deposit Remaining : ℝ)

-- Define the conditions
def deposit_condition : Prop := Deposit = 0.10 * P
def remaining_condition : Prop := Remaining = 0.90 * P
def remaining_amount_given : Prop := Remaining = 1170

-- The goal to prove: the deposit paid is $130
theorem deposit_paid (h₁ : deposit_condition P Deposit) (h₂ : remaining_condition P Remaining) (h₃ : remaining_amount_given Remaining) : 
  Deposit = 130 :=
  sorry

end deposit_paid_l257_257498


namespace sum_of_divisors_252_l257_257025

open BigOperators

-- Definition of the sum of divisors for a given number n
def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ d in Nat.divisors n, d

-- Statement of the problem
theorem sum_of_divisors_252 : sum_of_divisors 252 = 728 := 
sorry

end sum_of_divisors_252_l257_257025


namespace solve_for_n_l257_257312

theorem solve_for_n (n : ℕ) (h : 3^n * 9^n = 81^(n - 12)) : n = 48 :=
sorry

end solve_for_n_l257_257312


namespace bogatyrs_truthful_count_l257_257899

noncomputable def number_of_truthful_warriors (total_warriors: ℕ) (sword_yes: ℕ) (spear_yes: ℕ) (axe_yes: ℕ) (bow_yes: ℕ) : ℕ :=
  let total_yes := sword_yes + spear_yes + axe_yes + bow_yes
  let lying_warriors := (total_yes - total_warriors) / 2
  total_warriors - lying_warriors

theorem bogatyrs_truthful_count :
  number_of_truthful_warriors 33 13 15 20 27 = 12 := by
  sorry

end bogatyrs_truthful_count_l257_257899


namespace largest_neg_multiple_of_7_greater_than_neg_150_l257_257360

theorem largest_neg_multiple_of_7_greater_than_neg_150 : 
  ∃ (n : ℤ), (n % 7 = 0) ∧ (-n > -150) ∧ (∀ m : ℤ, (m % 7 = 0) ∧ (-m > -150) → m ≤ n) :=
begin
  use 147,
  split,
  { norm_num }, -- Verifies that 147 is a multiple of 7
  split,
  { norm_num }, -- Verifies that -147 > -150
  { intros m h,
    obtain ⟨k, rfl⟩ := (zmod.int_coe_zmod_eq_zero_iff_dvd m 7).mp h.1,
    suffices : k ≤ 21, { rwa [int.nat_abs_of_nonneg (by norm_num : (7 : ℤ) ≥ 0), ←abs_eq_nat_abs, int.abs_eq_nat_abs, nat.abs_of_nonneg (zero_le 21), ← int.le_nat_abs_iff_coe_nat_le] at this },
    have : -m > -150 := h.2,
    rwa [int.lt_neg, neg_le_neg_iff] at this,
    norm_cast at this,
    exact this
  }
end

end largest_neg_multiple_of_7_greater_than_neg_150_l257_257360


namespace minimize_material_l257_257848

theorem minimize_material (π V R h : ℝ) (hV : V > 0) (h_cond : π * R^2 * h = V) :
  R = h / 2 :=
sorry

end minimize_material_l257_257848


namespace unattainable_y_l257_257764

theorem unattainable_y (x : ℚ) (h : x ≠ -4 / 3) : (∀ y : ℚ, (y = (2 - x) / (3 * x + 4)) → y ≠ -1 / 3) :=
by
  intro y
  intro h1
  have h2 : 3 * (-1 / 3) + 1 = 0 := by norm_num
  have h3 : 3 * x + 4 ≠ 0 := by
    intro h4
    have h5 : x = -4 / 3 := by linarith
    exact h h5
  rw h2 at h1
  exact h3 h1

end unattainable_y_l257_257764


namespace sequence_property_l257_257304

variable (a : ℕ → ℕ)

theorem sequence_property
  (h_bij : Function.Bijective a) (n : ℕ) :
  ∃ k, k < n ∧ a (n - k) < a n ∧ a n < a (n + k) :=
sorry

end sequence_property_l257_257304


namespace total_girls_in_circle_l257_257044

theorem total_girls_in_circle (girls : Nat) 
  (h1 : (4 + 7) = girls + 2) : girls = 11 := 
by
  sorry

end total_girls_in_circle_l257_257044


namespace proposition_b_proposition_d_l257_257366

-- Proposition B: For a > 0 and b > 0, if ab = 2, then the minimum value of a + 2b is 4
theorem proposition_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 2) : a + 2 * b ≥ 4 :=
  sorry

-- Proposition D: For a > 0 and b > 0, if a² + b² = 1, then the maximum value of a + b is sqrt(2).
theorem proposition_d (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^2 + b^2 = 1) : a + b ≤ Real.sqrt 2 :=
  sorry

end proposition_b_proposition_d_l257_257366


namespace erica_total_earnings_l257_257919

def fishPrice : Nat := 20
def pastCatch : Nat := 80
def todayCatch : Nat := 2 * pastCatch
def pastEarnings := pastCatch * fishPrice
def todayEarnings := todayCatch * fishPrice
def totalEarnings := pastEarnings + todayEarnings

theorem erica_total_earnings : totalEarnings = 4800 := by
  sorry

end erica_total_earnings_l257_257919


namespace arccos_identity_l257_257732

theorem arccos_identity : 
  ∀ x : ℝ, x = 1 / real.sqrt 2 → real.arccos x = real.pi / 4 :=
begin
  intros x hx,
  have h1 : real.cos (real.pi / 4) = 1 / real.sqrt 2 := by sorry,
  have h2 : real.arccos (1 / real.sqrt 2) = real.pi / 4,
  { rw ← h1,
    rw real.arccos_cos,
    exact_mod_cast hx },
  exact h2,
end

end arccos_identity_l257_257732


namespace total_population_of_towns_l257_257991

theorem total_population_of_towns :
  let num_towns := 25
  let avg_pop_min := 3600
  let avg_pop_max := 4000
  let estimated_avg_pop := (avg_pop_min + avg_pop_max) / 2
  num_towns * estimated_avg_pop = 95000 :=
by
  let num_towns := 25
  let avg_pop_min := 3600
  let avg_pop_max := 4000
  let estimated_avg_pop := (avg_pop_min + avg_pop_max) / 2
  show num_towns * estimated_avg_pop = 95000
  sorry

end total_population_of_towns_l257_257991


namespace foreign_students_next_sem_eq_740_l257_257517

def total_students : ℕ := 1800
def percentage_foreign : ℕ := 30
def new_foreign_students : ℕ := 200

def initial_foreign_students : ℕ := total_students * percentage_foreign / 100
def total_foreign_students_next_semester : ℕ :=
  initial_foreign_students + new_foreign_students

theorem foreign_students_next_sem_eq_740 :
  total_foreign_students_next_semester = 740 :=
by
  sorry

end foreign_students_next_sem_eq_740_l257_257517


namespace smallest_special_number_gt_3429_l257_257099

-- Define what it means for a number to be special
def is_special (n : ℕ) : Prop :=
  (List.toFinset (Nat.digits 10 n)).card = 4

-- Define the problem statement in Lean
theorem smallest_special_number_gt_3429 : ∃ n : ℕ, 3429 < n ∧ is_special n ∧ ∀ m : ℕ, 3429 < m ∧ is_special m → n ≤ m := 
  by
  let smallest_n := 3450
  have hn : 3429 < smallest_n := by decide
  have hs : is_special smallest_n := by
    -- digits of 3450 are [3, 4, 5, 0], which are four different digits
    sorry 
  have minimal : ∀ m, 3429 < m ∧ is_special m → smallest_n ≤ m :=
    by
    -- This needs to show that no special number exists between 3429 and 3450
    sorry
  exact ⟨smallest_n, hn, hs, minimal⟩

end smallest_special_number_gt_3429_l257_257099


namespace non_congruent_triangles_count_l257_257988

-- Let there be 15 equally spaced points on a circle,
-- and considering triangles formed by connecting 3 of these points.
def num_non_congruent_triangles (n : Nat) : Nat :=
  (if n = 15 then 19 else 0)

theorem non_congruent_triangles_count :
  num_non_congruent_triangles 15 = 19 :=
by
  sorry

end non_congruent_triangles_count_l257_257988


namespace initial_provisions_last_l257_257854

theorem initial_provisions_last (x : ℕ) (h : 2000 * (x - 20) = 4000 * 10) : x = 40 :=
by sorry

end initial_provisions_last_l257_257854


namespace arccos_one_over_sqrt_two_eq_pi_four_l257_257738

theorem arccos_one_over_sqrt_two_eq_pi_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_four_l257_257738


namespace casey_pumping_time_l257_257669

structure PlantRow :=
  (rows : ℕ) (plants_per_row : ℕ) (water_per_plant : ℚ)

structure Animal :=
  (count : ℕ) (water_per_animal : ℚ)

def morning_pump_rate := 3 -- gallons per minute
def afternoon_pump_rate := 5 -- gallons per minute

def corn := PlantRow.mk 4 15 0.5
def pumpkin := PlantRow.mk 3 10 0.8
def pigs := Animal.mk 10 4
def ducks := Animal.mk 20 0.25
def cows := Animal.mk 5 8

def total_water_needed_for_plants (corn pumpkin : PlantRow) : ℚ :=
  (corn.rows * corn.plants_per_row * corn.water_per_plant) +
  (pumpkin.rows * pumpkin.plants_per_row * pumpkin.water_per_plant)

def total_water_needed_for_animals (pigs ducks cows : Animal) : ℚ :=
  (pigs.count * pigs.water_per_animal) +
  (ducks.count * ducks.water_per_animal) +
  (cows.count * cows.water_per_animal)

def time_to_pump (total_water pump_rate : ℚ) : ℚ :=
  total_water / pump_rate

theorem casey_pumping_time :
  let total_water_plants := total_water_needed_for_plants corn pumpkin
  let total_water_animals := total_water_needed_for_animals pigs ducks cows
  let time_morning := time_to_pump total_water_plants morning_pump_rate
  let time_afternoon := time_to_pump total_water_animals afternoon_pump_rate
  time_morning + time_afternoon = 35 := by
sorry

end casey_pumping_time_l257_257669


namespace value_of_b_l257_257021

theorem value_of_b (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - a = 1) : b = 1 := by
  sorry

end value_of_b_l257_257021


namespace value_of_fraction_l257_257533

theorem value_of_fraction (a b : ℚ) (h : b / a = 1 / 2) : (a + b) / a = 3 / 2 :=
sorry

end value_of_fraction_l257_257533


namespace cats_remaining_proof_l257_257223

def initial_siamese : ℕ := 38
def initial_house : ℕ := 25
def sold_cats : ℕ := 45

def total_cats (s : ℕ) (h : ℕ) : ℕ := s + h
def remaining_cats (total : ℕ) (sold : ℕ) : ℕ := total - sold

theorem cats_remaining_proof : remaining_cats (total_cats initial_siamese initial_house) sold_cats = 18 :=
by
  sorry

end cats_remaining_proof_l257_257223


namespace mode_of_data_set_l257_257125

theorem mode_of_data_set :
  ∃ (x : ℝ), x = 5 ∧
    let data_set := [1, 0, -3, 5, x, 2, -3] in
    (1 + 0 - 3 + 5 + x + 2 - 3) / (data_set.length : ℝ) = 1 ∧
    {y : ℝ | ∃ (n : ℕ), ∀ (z : ℝ), z ∈ data_set → data_set.count z = n → n = 2} = {-3, 5} :=
begin
  sorry
end

end mode_of_data_set_l257_257125


namespace arccos_one_over_sqrt_two_eq_pi_four_l257_257737

theorem arccos_one_over_sqrt_two_eq_pi_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_four_l257_257737


namespace girls_select_same_color_l257_257642

def select_marbles (bag : ℕ) : Prop :=
  bag = 8 ∧ ∃ (white black : ℕ), white = 4 ∧ black = 4

def probability_same_color (prob : ℚ) : Prop :=
  prob = 1 / 35

theorem girls_select_same_color (bag : ℕ) (prob : ℚ) (h_bag : select_marbles bag) : 
  probability_same_color prob :=
sorry

end girls_select_same_color_l257_257642


namespace r_n_m_smallest_m_for_r_2006_l257_257166

def euler_totient (n : ℕ) : ℕ := 
  n * (1 - (1 / 2)) * (1 - (1 / 17)) * (1 - (1 / 59))

def r (n m : ℕ) : ℕ :=
  m * euler_totient n

theorem r_n_m (n m : ℕ) : r n m = m * euler_totient n := 
  by sorry

theorem smallest_m_for_r_2006 (n m : ℕ) (h : n = 2006) (h2 : r n m = 841 * 928) : 
  ∃ m, r n m = 841^2 := 
  by sorry

end r_n_m_smallest_m_for_r_2006_l257_257166


namespace no_integers_satisfy_equation_l257_257809

theorem no_integers_satisfy_equation :
  ∀ (a b c : ℤ), a^2 + b^2 - 8 * c ≠ 6 := by
  sorry

end no_integers_satisfy_equation_l257_257809


namespace base8_to_base10_l257_257079

theorem base8_to_base10 (n : ℕ) : of_digits 8 [2, 4, 6] = 166 := by
  sorry

end base8_to_base10_l257_257079


namespace largest_multiple_of_7_negated_gt_neg_150_l257_257352

theorem largest_multiple_of_7_negated_gt_neg_150 :
  ∃ (n : ℕ), (negate (n * 7) > -150) ∧ (∀ m : ℕ, (negate (m * 7) > -150) → m ≤ n) ∧ (n * 7 = 147) :=
sorry

end largest_multiple_of_7_negated_gt_neg_150_l257_257352


namespace gcd_cube_sum_condition_l257_257400

theorem gcd_cube_sum_condition (n : ℕ) (hn : n > 32) : Nat.gcd (n^3 + 125) (n + 5) = 1 := 
  by 
  sorry

end gcd_cube_sum_condition_l257_257400


namespace area_percentage_change_l257_257323

variable (a b : ℝ)

def initial_area : ℝ := a * b

def new_length (a : ℝ) : ℝ := a * 1.35

def new_width (b : ℝ) : ℝ := b * 0.86

def new_area (a b : ℝ) : ℝ := (new_length a) * (new_width b)

theorem area_percentage_change :
    ((new_area a b) / (initial_area a b)) = 1.161 :=
by
  sorry

end area_percentage_change_l257_257323


namespace t_shirt_price_increase_t_shirt_max_profit_l257_257503

theorem t_shirt_price_increase (x : ℝ) : (x + 10) * (300 - 10 * x) = 3360 → x = 2 := 
by 
  sorry

theorem t_shirt_max_profit (x : ℝ) : (-10 * x^2 + 200 * x + 3000) = 4000 ↔ x = 10 := 
by 
  sorry

end t_shirt_price_increase_t_shirt_max_profit_l257_257503


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l257_257677

theorem arccos_one_over_sqrt_two_eq_pi_over_four : 
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l257_257677


namespace base8_246_is_166_in_base10_l257_257066

def convert_base8_to_base10 (n : ℕ) : ℕ :=
  let d0 := n % 10;
  let d1 := (n / 10) % 10;
  let d2 := (n / 100) % 10;
  d2 * 8^2 + d1 * 8^1 + d0 * 8^0

theorem base8_246_is_166_in_base10 : convert_base8_to_base10 246 = 166 :=
  sorry

end base8_246_is_166_in_base10_l257_257066


namespace probability_l257_257986

def total_chips : ℕ := 15
def blue_chips : ℕ := 6
def red_chips : ℕ := 5
def yellow_chips : ℕ := 4

def probability_of_different_colors : ℚ :=
  (blue_chips / total_chips) * ((red_chips + yellow_chips) / total_chips) +
  (red_chips / total_chips) * ((blue_chips + yellow_chips) / total_chips) +
  (yellow_chips / total_chips) * ((blue_chips + red_chips) / total_chips)

theorem probability : probability_of_different_colors = 148 / 225 :=
by
  unfold probability_of_different_colors
  sorry

end probability_l257_257986


namespace inequality_l257_257207

theorem inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 3) : 
  (1 / (8 * a^2 - 18 * a + 11)) + (1 / (8 * b^2 - 18 * b + 11)) + (1 / (8 * c^2 - 18 * c + 11)) ≤ 3 := 
sorry

end inequality_l257_257207


namespace solution_l257_257664

noncomputable def x1 : ℝ := sorry
noncomputable def x2 : ℝ := sorry
noncomputable def x3 : ℝ := sorry
noncomputable def x4 : ℝ := sorry
noncomputable def x5 : ℝ := sorry
noncomputable def x6 : ℝ := sorry
noncomputable def x7 : ℝ := sorry
noncomputable def x8 : ℝ := sorry

axiom cond1 : x1 + 4 * x2 + 9 * x3 + 16 * x4 + 25 * x5 + 36 * x6 + 49 * x7 + 64 * x8 = 10
axiom cond2 : 4 * x1 + 9 * x2 + 16 * x3 + 25 * x4 + 36 * x5 + 49 * x6 + 64 * x7 + 81 * x8 = 40
axiom cond3 : 9 * x1 + 16 * x2 + 25 * x3 + 36 * x4 + 49 * x5 + 64 * x6 + 81 * x7 + 100 * x8 = 170

theorem solution : 16 * x1 + 25 * x2 + 36 * x3 + 49 * x4 + 64 * x5 + 81 * x6 + 100 * x7 + 121 * x8 = 400 := 
by
  sorry

end solution_l257_257664


namespace red_pigment_contribution_l257_257218

theorem red_pigment_contribution :
  ∀ (G : ℝ), (2 * G + G + 3 * G = 24) →
  (0.6 * (2 * G) + 0.5 * (3 * G) = 10.8) :=
by
  intro G
  intro h1
  sorry

end red_pigment_contribution_l257_257218


namespace digital_root_8_pow_n_l257_257178

-- Define the conditions
def n : ℕ := 1989

-- Define the simplified problem
def digital_root (x : ℕ) : ℕ := if x % 9 = 0 then 9 else x % 9

-- Statement of the problem
theorem digital_root_8_pow_n : digital_root (8 ^ n) = 8 := by
  have mod_nine_eq : 8^n % 9 = 8 := by
    sorry
  simp [digital_root, mod_nine_eq]

end digital_root_8_pow_n_l257_257178


namespace sams_seashells_l257_257815

theorem sams_seashells (mary_seashells : ℕ) (total_seashells : ℕ) (h_mary : mary_seashells = 47) (h_total : total_seashells = 65) : (total_seashells - mary_seashells) = 18 :=
by
  simp [h_mary, h_total]
  sorry

end sams_seashells_l257_257815


namespace truthfulness_count_l257_257890

-- Define variables to represent the number of warriors and their response counts
def num_warriors : Nat := 33
def yes_sword : Nat := 13
def yes_spear : Nat := 15
def yes_axe : Nat := 20
def yes_bow : Nat := 27

-- Define the total number of "Yes" answers
def total_yes_answers : Nat := yes_sword + yes_spear + yes_axe + yes_bow

theorem truthfulness_count :
  ∃ x : Nat, x + 3 * (num_warriors - x) = total_yes_answers ∧ x = 12 :=
by
  sorry

end truthfulness_count_l257_257890


namespace relationships_with_correlation_l257_257486

-- Definitions for each of the relationships as conditions
def person_age_wealth := true -- placeholder definition 
def curve_points_coordinates := true -- placeholder definition
def apple_production_climate := true -- placeholder definition
def tree_diameter_height := true -- placeholder definition
def student_school := true -- placeholder definition

-- Statement to prove which relationships involve correlation
theorem relationships_with_correlation :
  person_age_wealth ∧ apple_production_climate ∧ tree_diameter_height :=
by
  sorry

end relationships_with_correlation_l257_257486


namespace base8_to_base10_conversion_l257_257063

def base8_to_base10 (n : Nat) : Nat := 
  match n with
  | 246 => 2 * 8^2 + 4 * 8^1 + 6 * 8^0
  | _ => 0  -- We define this only for the number 246_8

theorem base8_to_base10_conversion : base8_to_base10 246 = 166 := by 
  sorry

end base8_to_base10_conversion_l257_257063


namespace truthfulness_count_l257_257888

-- Define variables to represent the number of warriors and their response counts
def num_warriors : Nat := 33
def yes_sword : Nat := 13
def yes_spear : Nat := 15
def yes_axe : Nat := 20
def yes_bow : Nat := 27

-- Define the total number of "Yes" answers
def total_yes_answers : Nat := yes_sword + yes_spear + yes_axe + yes_bow

theorem truthfulness_count :
  ∃ x : Nat, x + 3 * (num_warriors - x) = total_yes_answers ∧ x = 12 :=
by
  sorry

end truthfulness_count_l257_257888


namespace inequality_of_sum_of_squares_l257_257167

theorem inequality_of_sum_of_squares (a b c : ℝ) (h : a * b + b * c + a * c = 1) : (a + b + c) ^ 2 ≥ 3 :=
sorry

end inequality_of_sum_of_squares_l257_257167


namespace fewer_mpg_in_city_l257_257217

theorem fewer_mpg_in_city
  (highway_miles : ℕ)
  (city_miles : ℕ)
  (city_mpg : ℕ)
  (highway_mpg : ℕ)
  (tank_size : ℝ) :
  highway_miles = 462 →
  city_miles = 336 →
  city_mpg = 32 →
  tank_size = 336 / 32 →
  highway_mpg = 462 / tank_size →
  (highway_mpg - city_mpg) = 12 :=
by
  intros h_highway_miles h_city_miles h_city_mpg h_tank_size h_highway_mpg
  sorry

end fewer_mpg_in_city_l257_257217


namespace solve_for_y_l257_257274

noncomputable def x : ℝ := 20
noncomputable def y : ℝ := 40

theorem solve_for_y 
  (h₁ : 1.5 * x = 0.75 * y) 
  (h₂ : x = 20) : 
  y = 40 :=
by
  sorry

end solve_for_y_l257_257274


namespace fraction_simplification_l257_257234

theorem fraction_simplification :
  (3 / (2 - (3 / 4))) = 12 / 5 := 
by
  sorry

end fraction_simplification_l257_257234


namespace max_sum_terms_arithmetic_seq_l257_257786

theorem max_sum_terms_arithmetic_seq (a1 d : ℝ) (h1 : a1 > 0) 
  (h2 : 3 * (2 * a1 + 2 * d) = 11 * (2 * a1 + 10 * d)) :
  ∃ (n : ℕ),  (∀ k, 1 ≤ k ∧ k ≤ n → a1 + (k - 1) * d > 0) ∧  a1 + n * d ≤ 0 ∧ n = 7 :=
by
  sorry

end max_sum_terms_arithmetic_seq_l257_257786


namespace number_of_truth_tellers_is_twelve_l257_257905
noncomputable theory

section
variables (x : ℕ)
variables (y : ℕ)
variables (a b c d : ℕ)

-- Given conditions
def total_warriors : ℕ := 33
def yes_sword : ℕ := 13
def yes_spear : ℕ := 15
def yes_axe : ℕ := 20
def yes_bow : ℕ := 27

-- Calculate total yes answers (condition 6)
def total_yes_answers : ℕ := yes_sword + yes_spear + yes_axe + yes_bow

-- Truth-tellers say "yes" to only one question, liers say "yes" to three questions
def truth_yes : ℕ := x * 1
def lie_yes : ℕ := (total_warriors - x) * 3

theorem number_of_truth_tellers_is_twelve
  (h1 : total_warriors = 33)
  (h2 : yes_sword = 13)
  (h3 : yes_spear = 15)
  (h4 : yes_axe = 20)
  (h5 : yes_bow = 27)
  (h6 : total_yes_answers = 75)
  (h7 : total_yes_answers = truth_yes + lie_yes) :
    x = 12 :=
  by sorry

end

end number_of_truth_tellers_is_twelve_l257_257905


namespace number_of_baskets_l257_257834

-- Define the conditions
def total_peaches : Nat := 10
def red_peaches_per_basket : Nat := 4
def green_peaches_per_basket : Nat := 6
def peaches_per_basket : Nat := red_peaches_per_basket + green_peaches_per_basket

-- The goal is to prove that the number of baskets is 1 given the conditions

theorem number_of_baskets (h1 : total_peaches = 10)
                           (h2 : peaches_per_basket = red_peaches_per_basket + green_peaches_per_basket)
                           (h3 : red_peaches_per_basket = 4)
                           (h4 : green_peaches_per_basket = 6) : 
                           total_peaches / peaches_per_basket = 1 := by
                            sorry

end number_of_baskets_l257_257834


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l257_257671

theorem arccos_one_over_sqrt_two_eq_pi_over_four : 
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l257_257671


namespace sufficient_but_not_necessary_necessary_but_not_sufficient_l257_257942

def M (x : ℝ) : Prop := (x + 3) * (x - 5) > 0
def P (x : ℝ) (a : ℝ) : Prop := x^2 + (a - 8)*x - 8*a ≤ 0
def I : Set ℝ := {x | 5 < x ∧ x ≤ 8}

theorem sufficient_but_not_necessary (a : ℝ) :
  (∀ x, M x ∧ P x a ↔ x ∈ I) → a = 0 :=
sorry

theorem necessary_but_not_sufficient (a : ℝ) :
  (∀ x, (M x ∧ P x a → x ∈ I) ∧ (∀ x, x ∈ I → M x ∧ P x a)) → a ≤ 3 :=
sorry

end sufficient_but_not_necessary_necessary_but_not_sufficient_l257_257942


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l257_257698

theorem arccos_one_over_sqrt_two_eq_pi_over_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l257_257698


namespace arccos_identity_l257_257728

theorem arccos_identity : 
  ∀ x : ℝ, x = 1 / real.sqrt 2 → real.arccos x = real.pi / 4 :=
begin
  intros x hx,
  have h1 : real.cos (real.pi / 4) = 1 / real.sqrt 2 := by sorry,
  have h2 : real.arccos (1 / real.sqrt 2) = real.pi / 4,
  { rw ← h1,
    rw real.arccos_cos,
    exact_mod_cast hx },
  exact h2,
end

end arccos_identity_l257_257728


namespace expectation_inverse_quadratic_form_exists_l257_257802

open MeasureTheory ProbabilityTheory

noncomputable def gaussian_vector (d: ℕ) : Type := sorry -- Gaussian vector in ℝ^d with unit covariance matrix

axiom symmetric_positive_definite (d: ℕ) (B: Matrix (Fin d) (Fin d) ℝ) : Prop := 
B.isSymmetric ∧ ∀ z : Fin d → ℝ, z ≠ 0 → (z ⬝ᵥ (B.mul_vec z)) > 0

-- Main statement
theorem expectation_inverse_quadratic_form_exists (d: ℕ) (ξ : gaussian_vector d) (B: Matrix (Fin d) (Fin d) ℝ)
  (hB : symmetric_positive_definite d B) :
  ∃ (E : ℝ), E = 𝔼[(ξᵀ ⬝ᵥ (B.mul_vec ξ))⁻¹] ↔ d > 2 :=
sorry

end expectation_inverse_quadratic_form_exists_l257_257802


namespace shem_earnings_l257_257310

theorem shem_earnings (kem_hourly: ℝ) (ratio: ℝ) (workday_hours: ℝ) (shem_hourly: ℝ) (shem_daily: ℝ) :
  kem_hourly = 4 →
  ratio = 2.5 →
  shem_hourly = kem_hourly * ratio →
  workday_hours = 8 →
  shem_daily = shem_hourly * workday_hours →
  shem_daily = 80 :=
by
  -- Proof omitted
  sorry

end shem_earnings_l257_257310


namespace total_quarters_l257_257568

-- Definitions from conditions
def initial_quarters : ℕ := 49
def quarters_given_by_dad : ℕ := 25

-- Theorem to prove the total quarters is 74
theorem total_quarters : initial_quarters + quarters_given_by_dad = 74 :=
by sorry

end total_quarters_l257_257568


namespace at_least_two_even_degree_l257_257784

noncomputable theory

-- Define the scenario as an undirected graph with 44 vertices
def studentGraph : Type := simple_graph (fin 44)

-- State the theorem
theorem at_least_two_even_degree (G : studentGraph) : 
  ∃ (v1 v2 : fin 44), G.degree v1 % 2 = 0 ∧ G.degree v2 % 2 = 0 := 
sorry

end at_least_two_even_degree_l257_257784


namespace base8_246_is_166_in_base10_l257_257065

def convert_base8_to_base10 (n : ℕ) : ℕ :=
  let d0 := n % 10;
  let d1 := (n / 10) % 10;
  let d2 := (n / 100) % 10;
  d2 * 8^2 + d1 * 8^1 + d0 * 8^0

theorem base8_246_is_166_in_base10 : convert_base8_to_base10 246 = 166 :=
  sorry

end base8_246_is_166_in_base10_l257_257065


namespace abs_sub_self_nonneg_l257_257445

theorem abs_sub_self_nonneg (m : ℚ) : |m| - m ≥ 0 := 
sorry

end abs_sub_self_nonneg_l257_257445


namespace solve_for_a_l257_257555

theorem solve_for_a (x a : ℝ) (h : x = 5) (h_eq : a * x - 8 = 10 + 4 * a) : a = 18 :=
by
  sorry

end solve_for_a_l257_257555


namespace average_of_B_and_C_l257_257785

theorem average_of_B_and_C (x : ℚ) (A B C : ℚ)
  (h1 : A = 4 * x) (h2 : B = 6 * x) (h3 : C = 9 * x) (h4 : A = 50) :
  (B + C) / 2 = 93.75 := 
sorry

end average_of_B_and_C_l257_257785


namespace total_amount_lent_l257_257332

theorem total_amount_lent (A T : ℝ) (hA : A = 15008) (hInterest : 0.08 * A + 0.10 * (T - A) = 850) : 
  T = 11501.6 :=
by
  sorry

end total_amount_lent_l257_257332


namespace f_increasing_on_Ioo_l257_257546

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

theorem f_increasing_on_Ioo : ∀ x y : ℝ, x < y → f x < f y :=
sorry

end f_increasing_on_Ioo_l257_257546


namespace smallest_b_greater_than_three_l257_257362

theorem smallest_b_greater_than_three (b : ℕ) (h : b > 3) : 
  (∃ b, b = 5 ∧ (∃ n : ℕ, 4 * b + 5 = n^2)) :=
by
  use 5
  constructor
  · rfl
  · use 5
  sorry

end smallest_b_greater_than_three_l257_257362


namespace total_nephews_proof_l257_257231

-- We declare the current number of nephews as unknown variables
variable (Alden_current Vihaan Shruti Nikhil : ℕ)

-- State the conditions as hypotheses
theorem total_nephews_proof
  (h1 : 70 = (1 / 3 : ℚ) * Alden_current)
  (h2 : Vihaan = Alden_current + 120)
  (h3 : Shruti = 2 * Vihaan)
  (h4 : Nikhil = Alden_current + Shruti - 40) :
  Alden_current + Vihaan + Shruti + Nikhil = 2030 := 
by
  sorry

end total_nephews_proof_l257_257231


namespace tax_percentage_l257_257307

-- Definitions
def salary_before_taxes := 5000
def rent_expense_per_month := 1350
def total_late_rent_payments := 2 * rent_expense_per_month
def fraction_of_next_salary_after_taxes := (3 / 5 : ℚ)

-- Main statement to prove
theorem tax_percentage (T : ℚ) : 
  fraction_of_next_salary_after_taxes * (salary_before_taxes - (T / 100) * salary_before_taxes) = total_late_rent_payments → 
  T = 10 :=
by
  sorry

end tax_percentage_l257_257307


namespace smallest_special_number_l257_257085

def is_special (n : ℕ) : Prop :=
  (∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
   n = a * 1000 + b * 100 + c * 10 + d)

theorem smallest_special_number (n : ℕ) (h1 : n > 3429) (h2 : is_special n) : n = 3450 :=
sorry

end smallest_special_number_l257_257085


namespace second_person_more_heads_probability_l257_257616

noncomputable def coin_flip_probability (n m : ℕ) : ℚ :=
  if n < m then 1 / 2 else 0

theorem second_person_more_heads_probability :
  coin_flip_probability 10 11 = 1 / 2 :=
by
  sorry

end second_person_more_heads_probability_l257_257616


namespace no_real_solution_l257_257592

theorem no_real_solution :
  ¬ ∃ x : ℝ, 7 * (4 * x + 3) - 4 = -3 * (2 - 9 * x^2) :=
by
  sorry

end no_real_solution_l257_257592


namespace num_true_statements_l257_257133

theorem num_true_statements :
  (∀ x y a, a ≠ 0 → (a^2 * x > a^2 * y → x > y)) ∧
  (∀ x y a, a ≠ 0 → (a^2 * x ≥ a^2 * y → x ≥ y)) ∧
  (∀ x y a, a ≠ 0 → (x / a^2 ≥ y / a^2 → x ≥ y)) ∧
  (∀ x y a, a ≠ 0 → (x ≥ y → x / a^2 ≥ y / a^2)) →
  ((∀ x y a, a ≠ 0 → (a^2 * x > a^2 * y → x > y)) →
   (∀ x y a, a ≠ 0 → (x / a^2 ≥ y / a^2 → x ≥ y))) :=
sorry

end num_true_statements_l257_257133


namespace value_of_a_sum_l257_257116

theorem value_of_a_sum (a_7 a_6 a_5 a_4 a_3 a_2 a_1 a : ℝ) :
  (∀ x : ℝ, (3 * x - 1)^7 = a_7 * x^7 + a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a) →
  a + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = 128 := 
by
  sorry

end value_of_a_sum_l257_257116


namespace least_plates_to_ensure_matching_pair_l257_257036

theorem least_plates_to_ensure_matching_pair
  (white_plates : ℕ)
  (green_plates : ℕ)
  (red_plates : ℕ)
  (pink_plates : ℕ)
  (purple_plates : ℕ)
  (h_white : white_plates = 2)
  (h_green : green_plates = 6)
  (h_red : red_plates = 8)
  (h_pink : pink_plates = 4)
  (h_purple : purple_plates = 10) :
  ∃ n, n = 6 :=
by
  sorry

end least_plates_to_ensure_matching_pair_l257_257036


namespace actual_revenue_percentage_of_projected_l257_257635

theorem actual_revenue_percentage_of_projected (R : ℝ) (hR : R > 0) :
  (0.75 * R) / (1.2 * R) * 100 = 62.5 := 
by
  sorry

end actual_revenue_percentage_of_projected_l257_257635


namespace star_polygon_internal_angles_sum_l257_257228

-- Define the core aspects of the problem using type defintions and axioms.
def n_star_polygon_total_internal_angle_sum (n : ℕ) : ℝ :=
  180 * (n - 4)

theorem star_polygon_internal_angles_sum (n : ℕ) (h : n ≥ 6) :
  n_star_polygon_total_internal_angle_sum n = 180 * (n - 4) :=
by
  -- This step would involve the formal proof using Lean
  sorry

end star_polygon_internal_angles_sum_l257_257228


namespace probability_same_group_l257_257613

noncomputable def calcProbability : ℚ := 
  let totalOutcomes := 18 * 17
  let favorableCase1 := 6 * 5
  let favorableCase2 := 4 * 3
  let totalFavorableOutcomes := favorableCase1 + favorableCase2
  totalFavorableOutcomes / totalOutcomes

theorem probability_same_group (cards : Finset ℕ) (draws : Finset ℕ) (number1 number2 : ℕ) (condition_cardinality : cards.card = 20) 
  (condition_draws : draws.card = 4) (condition_numbers : number1 = 5 ∧ number2 = 14 ∧ number1 ∈ cards ∧ number2 ∈ cards) 
  : calcProbability = 7 / 51 :=
sorry

end probability_same_group_l257_257613


namespace truthful_warriors_count_l257_257912

-- Noncomputable theory is not necessary here.
-- We define the number of warriors and their responses.
def warriors_count : ℕ := 33
def yes_sword : ℕ := 13
def yes_spear : ℕ := 15
def yes_axe : ℕ := 20
def yes_bow : ℕ := 27

-- The equation from the solution steps where x is the number of truthful warriors.
def total_yes := yes_sword + yes_spear + yes_axe + yes_bow

theorem truthful_warriors_count :
  ∃ x : ℕ, x + 3 * (warriors_count - x) = total_yes ∧ x = 12 :=
by {
  -- We state that there exists an x such that its value satisfies the total "yes" responses equation.
  -- Here, we assert that x equals 12.
  use 12,
  split,
  {
    -- Prove the equation x + 3 * (warriors_count - x) = total_yes
    rw [warriors_count],
    rw [total_yes],
    norm_num,
  },
  {
    -- State that x = 12
    refl,
  },
}

end truthful_warriors_count_l257_257912


namespace parking_cost_savings_l257_257855

theorem parking_cost_savings
  (weekly_rate : ℕ := 10)
  (monthly_rate : ℕ := 24)
  (weeks_in_year : ℕ := 52)
  (months_in_year : ℕ := 12) :
  (weekly_rate * weeks_in_year) - (monthly_rate * months_in_year) = 232 :=
by
  sorry

end parking_cost_savings_l257_257855


namespace expand_product_l257_257240

theorem expand_product (y : ℝ) : 5 * (y - 3) * (y + 10) = 5 * y^2 + 35 * y - 150 := 
  sorry

end expand_product_l257_257240


namespace convert_base_8_to_base_10_l257_257080

def to_base_10 (n : ℕ) (b : ℕ) (digits : List ℕ) : ℕ :=
  digits.foldr (λ digit acc => acc * b + digit) 0

theorem convert_base_8_to_base_10 : 
  to_base_10 10 8 [6, 4, 2] = 166 := by
  sorry

end convert_base_8_to_base_10_l257_257080


namespace road_trip_ratio_l257_257462

-- Problem Definitions
variable (x d3 total grand_total : ℕ)
variable (hx1 : total = x + 2 * x + d3 + 2 * (x + 2 * x + d3))
variable (hx2 : d3 = 40)
variable (hx3 : total = 560)
variable (hx4 : grand_total = d3 / x)

-- Proof Statement
theorem road_trip_ratio (hx1 : total = x + 2 * x + d3 + 2 * (x + 2 * x + d3)) 
  (hx2 : d3 = 40) (hx3 : total = 560) : grand_total = 9 / 11 := by
  sorry

end road_trip_ratio_l257_257462


namespace maximum_area_of_sector_l257_257325

theorem maximum_area_of_sector (r l : ℝ) (h₁ : 2 * r + l = 10) : 
  (1 / 2 * l * r) ≤ 25 / 4 := 
sorry

end maximum_area_of_sector_l257_257325


namespace arnolds_total_protein_l257_257661

theorem arnolds_total_protein (collagen_protein_per_two_scoops : ℕ) (protein_per_scoop : ℕ) 
    (steak_protein : ℕ) (scoops_of_collagen : ℕ) (scoops_of_protein : ℕ) :
    collagen_protein_per_two_scoops = 18 →
    protein_per_scoop = 21 →
    steak_protein = 56 →
    scoops_of_collagen = 1 →
    scoops_of_protein = 1 →
    (collagen_protein_per_two_scoops / 2 * scoops_of_collagen + protein_per_scoop * scoops_of_protein + steak_protein = 86) :=
by
  intros hc p s sc sp
  sorry

end arnolds_total_protein_l257_257661


namespace solution_set_a_eq_1_find_a_min_value_3_l257_257135

open Real

noncomputable def f (x a : ℝ) := 2 * abs (x + 1) + abs (x - a)

-- The statement for the first question
theorem solution_set_a_eq_1 (x : ℝ) : f x 1 ≥ 5 ↔ x ≤ -2 ∨ x ≥ (4 / 3) := 
by sorry

-- The statement for the second question
theorem find_a_min_value_3 (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 3) ∧ (∃ x : ℝ, f x a = 3) ↔ a = 2 ∨ a = -4 := 
by sorry

end solution_set_a_eq_1_find_a_min_value_3_l257_257135


namespace remainder_of_x_plus_2_power_2008_l257_257624

-- Given: x^3 ≡ 1 (mod x^2 + x + 1)
def given_condition : Prop := ∀ x : ℤ, (x^3 - 1) % (x^2 + x + 1) = 0

-- To prove: The remainder when (x + 2)^2008 is divided by x^2 + x + 1 is 1
theorem remainder_of_x_plus_2_power_2008 (x : ℤ) (h : given_condition) :
  ((x + 2) ^ 2008) % (x^2 + x + 1) = 1 := by
  sorry

end remainder_of_x_plus_2_power_2008_l257_257624


namespace solve_x_l257_257819

theorem solve_x (x : ℝ) (h : x^2 + 6 * x + 8 = -(x + 4) * (x + 6)) : 
  x = -4 := 
by
  sorry

end solve_x_l257_257819


namespace find_multiple_l257_257305
-- Importing Mathlib to access any necessary math definitions.

-- Define the constants based on the given conditions.
def Darwin_money : ℝ := 45
def Mia_money : ℝ := 110
def additional_amount : ℝ := 20

-- The Lean theorem which encapsulates the proof problem.
theorem find_multiple (x : ℝ) : 
  Mia_money = x * Darwin_money + additional_amount → x = 2 :=
by
  sorry

end find_multiple_l257_257305


namespace z_in_fourth_quadrant_l257_257256

-- Given complex numbers z1 and z2
def z1 : ℂ := 3 - 2 * Complex.I
def z2 : ℂ := 1 + Complex.I

-- Define the multiplication of z1 and z2
def z : ℂ := z1 * z2

-- Prove that z is located in the fourth quadrant
theorem z_in_fourth_quadrant : z.re > 0 ∧ z.im < 0 :=
by
  -- Construction and calculations skipped for the math proof,
  -- the result should satisfy the conditions for being in the fourth quadrant
  sorry

end z_in_fourth_quadrant_l257_257256


namespace area_of_circular_flower_bed_l257_257185

theorem area_of_circular_flower_bed (C : ℝ) (hC : C = 62.8) : ∃ (A : ℝ), A = 314 :=
by
  sorry

end area_of_circular_flower_bed_l257_257185


namespace Carla_pays_more_than_Bob_l257_257487

theorem Carla_pays_more_than_Bob
  (slices : ℕ := 12)
  (veg_slices : ℕ := slices / 2)
  (non_veg_slices : ℕ := slices / 2)
  (base_cost : ℝ := 10)
  (extra_cost : ℝ := 3)
  (total_cost : ℝ := base_cost + extra_cost)
  (per_slice_cost : ℝ := total_cost / slices)
  (carla_slices : ℕ := veg_slices + 2)
  (bob_slices : ℕ := 3)
  (carla_payment : ℝ := carla_slices * per_slice_cost)
  (bob_payment : ℝ := bob_slices * per_slice_cost) :
  (carla_payment - bob_payment) = 5.41665 :=
sorry

end Carla_pays_more_than_Bob_l257_257487


namespace grain_milling_l257_257418

theorem grain_milling (A : ℚ) (h1 : 0.9 * A = 100) : A = 111 + 1 / 9 :=
by
  sorry

end grain_milling_l257_257418


namespace intersecting_x_value_l257_257018

theorem intersecting_x_value : 
  (∃ x y : ℝ, y = 3 * x - 17 ∧ 3 * x + y = 103) → 
  (∃ x : ℝ, x = 20) :=
by
  sorry

end intersecting_x_value_l257_257018


namespace g_13_equals_236_l257_257140

def g (n : ℕ) : ℕ := n^2 + 2 * n + 41

theorem g_13_equals_236 : g 13 = 236 := sorry

end g_13_equals_236_l257_257140


namespace arccos_proof_l257_257683

noncomputable def arccos_identity : Prop := 
  ∃ x : ℝ, x = 1 / Real.sqrt 2 ∧ Real.arccos x = Real.pi / 4

theorem arccos_proof : arccos_identity :=
by
  sorry

end arccos_proof_l257_257683


namespace triangular_stack_log_count_l257_257380

theorem triangular_stack_log_count : 
  ∀ (a₁ aₙ d : ℤ) (n : ℤ), a₁ = 15 → aₙ = 1 → d = -2 → 
  (a₁ - aₙ) / (-d) + 1 = n → 
  (n * (a₁ + aₙ)) / 2 = 64 :=
by
  intros a₁ aₙ d n h₁ hₙ hd hn
  sorry

end triangular_stack_log_count_l257_257380


namespace range_f_l257_257763

variable {α : Type} [LinearOrder α] [Field α]

noncomputable def f (x : α) : α := (3 * x + 8) / (x - 4)

theorem range_f : set.range f = { y : α | y ≠ 3 } :=
by
  sorry  -- Proof is omitted, according to the instructions

end range_f_l257_257763


namespace unique_solution_qx2_minus_16x_plus_8_eq_0_l257_257530

theorem unique_solution_qx2_minus_16x_plus_8_eq_0 (q : ℝ) (hq : q ≠ 0) :
  (∀ x : ℝ, q * x^2 - 16 * x + 8 = 0 → (256 - 32 * q = 0)) → q = 8 :=
by
  sorry

end unique_solution_qx2_minus_16x_plus_8_eq_0_l257_257530


namespace pushups_difference_l257_257390

theorem pushups_difference :
  let David_pushups := 44
  let Zachary_pushups := 35
  David_pushups - Zachary_pushups = 9 :=
by
  -- Here we define the push-ups counts
  let David_pushups := 44
  let Zachary_pushups := 35
  -- We need to show that David did 9 more push-ups than Zachary.
  show David_pushups - Zachary_pushups = 9
  sorry

end pushups_difference_l257_257390


namespace de_morgan_implication_l257_257801

variables (p q : Prop)

theorem de_morgan_implication (h : ¬(p ∧ q)) : ¬p ∨ ¬q :=
sorry

end de_morgan_implication_l257_257801


namespace max_value_2ab_2bc_root_3_l257_257573

theorem max_value_2ab_2bc_root_3 (a b c : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h_sum : a^2 + b^2 + c^2 = 3) :
  2 * a * b + 2 * b * c * Real.sqrt 3 ≤ 6 := by
sorry

end max_value_2ab_2bc_root_3_l257_257573


namespace probability_of_first_joker_second_king_is_correct_l257_257488

open Probability

noncomputable def probability_first_joker_second_king : ℚ :=
  let deck_size := 54
  let num_jokers := 2
  let num_kings := 4
  let prob_first_joker := (num_jokers : ℚ) / deck_size
  let prob_second_king_given_first_joker := (num_kings : ℚ) / (deck_size - 1)
  let prob_first_king := (num_kings : ℚ) / deck_size
  let prob_second_joker_given_first_king := (num_jokers : ℚ) / (deck_size - 1)
  (prob_first_joker * prob_second_king_given_first_joker) + (prob_first_king * prob_second_joker_given_first_king)

theorem probability_of_first_joker_second_king_is_correct :
  probability_first_joker_second_king = 8 / 1431 := sorry

end probability_of_first_joker_second_king_is_correct_l257_257488


namespace convert_246_octal_to_decimal_l257_257073

theorem convert_246_octal_to_decimal : 2 * (8^2) + 4 * (8^1) + 6 * (8^0) = 166 := 
by
  -- We skip the proof part as it is not required in the task
  sorry

end convert_246_octal_to_decimal_l257_257073


namespace bob_fencing_needed_l257_257054

-- Problem conditions
def length : ℕ := 225
def width : ℕ := 125
def small_gate : ℕ := 3
def large_gate : ℕ := 10

-- Definition of perimeter
def perimeter (l w : ℕ) : ℕ := 2 * (l + w)

-- Total width of the gates
def total_gate_width (g1 g2 : ℕ) : ℕ := g1 + g2

-- Amount of fencing needed
def fencing_needed (p gw : ℕ) : ℕ := p - gw

-- Theorem statement
theorem bob_fencing_needed :
  fencing_needed (perimeter length width) (total_gate_width small_gate large_gate) = 687 :=
by 
  sorry

end bob_fencing_needed_l257_257054


namespace quadratic_standard_form_l257_257873

theorem quadratic_standard_form :
  ∀ x : ℝ, (2 * x - 1) ^ 2 = (x + 1) * (3 * x + 4) →
  (∃ a b c : ℝ, a = 1 ∧ b = -11 ∧ c = -3 ∧ a * x^2 + b * x + c = 0) :=
by
  intro x
  intro h
  sorry

end quadratic_standard_form_l257_257873


namespace mode_of_data_set_l257_257124

noncomputable def data_set : List ℝ := [1, 0, -3, 5, 5, 2, -3]

theorem mode_of_data_set
  (x : ℝ)
  (h_avg : (1 + 0 - 3 + 5 + x + 2 - 3) / 7 = 1)
  (h_x : x = 5) :
  ({-3, 5} : Set ℝ) = {y : ℝ | data_set.count y = 2} :=
by
  -- Proof would go here
  sorry

end mode_of_data_set_l257_257124


namespace inequality_holds_for_all_x_iff_m_eq_1_l257_257106

theorem inequality_holds_for_all_x_iff_m_eq_1 (m : ℝ) (h_m : m ≠ 0) :
  (∀ x > 0, x^2 - 2 * m * Real.log x ≥ 1) ↔ m = 1 :=
by
  sorry

end inequality_holds_for_all_x_iff_m_eq_1_l257_257106


namespace max_area_of_triangle_l257_257536

open Real

theorem max_area_of_triangle (a b c : ℝ) 
  (ha : 9 ≥ a) 
  (ha1 : a ≥ 8) 
  (hb : 8 ≥ b) 
  (hb1 : b ≥ 4) 
  (hc : 4 ≥ c) 
  (hc1 : c ≥ 3) : 
  ∃ A : ℝ, ∃ S : ℝ, S ≤ 16 ∧ S = max (1/2 * b * c * sin A) 16 := 
sorry

end max_area_of_triangle_l257_257536


namespace number_of_voters_in_election_l257_257051

theorem number_of_voters_in_election
  (total_membership : ℕ)
  (votes_cast : ℕ)
  (winning_percentage_cast : ℚ)
  (percentage_of_total : ℚ)
  (h_total : total_membership = 1600)
  (h_winning_percentage : winning_percentage_cast = 0.60)
  (h_percentage_of_total : percentage_of_total = 0.196875)
  (h_votes : winning_percentage_cast * votes_cast = percentage_of_total * total_membership) :
  votes_cast = 525 :=
by
  sorry

end number_of_voters_in_election_l257_257051


namespace base9_num_digits_2500_l257_257552

theorem base9_num_digits_2500 : 
  ∀ (n : ℕ), (9^1 = 9) → (9^2 = 81) → (9^3 = 729) → (9^4 = 6561) → n = 4 := by
  sorry

end base9_num_digits_2500_l257_257552


namespace total_votes_cast_l257_257849

theorem total_votes_cast (V : ℕ) (C R : ℕ) 
  (hC : C = 30 * V / 100) 
  (hR1 : R = C + 4000) 
  (hR2 : R = 70 * V / 100) : 
  V = 10000 :=
by
  sorry

end total_votes_cast_l257_257849


namespace minimum_reciprocal_sum_l257_257144

noncomputable def circle_center : ℝ × ℝ := (-1, 2)

theorem minimum_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
    (h3 : a + 2 * b = 1) : (1 / a) + (1 / b) = 3 + 2 * Real.sqrt 2 := 
sorry

end minimum_reciprocal_sum_l257_257144


namespace arccos_sqrt2_l257_257692

def arccos_eq (x : ℝ) := arccos x
def range_arccos := ∀ (x : ℝ), 0 ≤ arccos x ∧ arccos x ≤ π
def cos_pi_div_four : Real.cos (π / 4) = 1 / Real.sqrt 2 := by sorry

theorem arccos_sqrt2 :
  arccos_eq (1 / Real.sqrt 2) = π / 4 :=
by
  have h1 : Real.cos (π / 4) = 1 / Real.sqrt 2, from cos_pi_div_four
  sorry

end arccos_sqrt2_l257_257692


namespace smallest_special_gt_3429_l257_257104

def is_special (n : ℕ) : Prop :=
  (10^3 ≤ n ∧ n < 10^4) ∧ (List.length (n.digits 10).eraseDup = 4)

theorem smallest_special_gt_3429 : 
  ∃ n : ℕ, n > 3429 ∧ is_special n ∧ ∀ m : ℕ, m > 3429 ∧ is_special m → n ≤ m := 
begin
  use 3450,
  split,
  { exact nat.succ_lt_succ (nat.s succ_lt_succ (nat.succ_lt_succ (nat.succ_lt_succ (nat.lt_succ_self 3430)))) },
  split,
  { unfold is_special,
    split,
    { split,
        { linarith },
    { linarith } },
    { unfold List.eraseDup,
    unfold List.redLength,
    exactly simp } },
  { intros m hm1 hm2,
    interval_cases m,
    sorry },
end

end smallest_special_gt_3429_l257_257104


namespace arithmetic_sequence_second_term_l257_257150

theorem arithmetic_sequence_second_term (a d : ℝ) (h : a + (a + 2 * d) = 8) : a + d = 4 :=
sorry

end arithmetic_sequence_second_term_l257_257150


namespace value_of_x_l257_257251

theorem value_of_x (x y : ℕ) (h1 : x / y = 8 / 3) (h2 : y = 27) : x = 72 :=
by
  sorry

end value_of_x_l257_257251


namespace correct_equation_by_moving_digit_l257_257236

theorem correct_equation_by_moving_digit :
  (10^2 - 1 = 99) → (101 = 102 - 1) :=
by
  intro h
  sorry

end correct_equation_by_moving_digit_l257_257236


namespace sum_of_lengths_of_edges_l257_257227

theorem sum_of_lengths_of_edges (s h : ℝ) 
(volume_eq : s^2 * h = 576) 
(surface_area_eq : 4 * s * h = 384) : 
8 * s + 4 * h = 112 := 
by
  sorry

end sum_of_lengths_of_edges_l257_257227


namespace max_d_77733e_divisible_by_33_l257_257925

open Int

theorem max_d_77733e_divisible_by_33 : ∃ d e : ℕ, 
  (7 * 100000 + d * 10000 + 7 * 1000 + 3 * 100 + 3 * 10 + e) % 33 = 0 ∧ 
  (d ≤ 9) ∧ (e ≤ 9) ∧ 
  (∀ d' e', ((7 * 100000 + d' * 10000 + 7 * 1000 + 3 * 100 + 3 * 10 + e') % 33 = 0 ∧ d' ≤ 9 ∧ e' ≤ 9 → d' ≤ d)) 
  := ⟨6, 0, by sorry⟩

end max_d_77733e_divisible_by_33_l257_257925


namespace bill_can_buy_donuts_in_35_ways_l257_257384

def different_ways_to_buy_donuts : ℕ :=
  5 + 20 + 10  -- Number of ways to satisfy the conditions

theorem bill_can_buy_donuts_in_35_ways :
  different_ways_to_buy_donuts = 35 :=
by
  -- Proof steps
  -- The problem statement and the solution show the calculation to be correct.
  sorry

end bill_can_buy_donuts_in_35_ways_l257_257384
