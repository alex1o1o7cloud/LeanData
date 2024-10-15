import Mathlib

namespace NUMINAMATH_GPT_purchasing_plan_exists_l1318_131828

-- Define the structure for our purchasing plan
structure PurchasingPlan where
  n3 : ℕ
  n6 : ℕ
  n9 : ℕ
  n12 : ℕ
  n15 : ℕ
  n19 : ℕ
  n21 : ℕ
  n30 : ℕ

-- Define the length function to sum up the total length of the purchasing plan
def length (p : PurchasingPlan) : ℕ :=
  3 * p.n3 + 6 * p.n6 + 9 * p.n9 + 12 * p.n12 + 15 * p.n15 + 19 * p.n19 + 21 * p.n21 + 30 * p.n30

-- Define the purchasing options
def options : List ℕ := [3, 6, 9, 12, 15, 19, 21, 30]

-- Define the requirement
def requiredLength : ℕ := 50

-- State the theorem that there exists a purchasing plan that sums up to the required length
theorem purchasing_plan_exists : ∃ p : PurchasingPlan, length p = requiredLength :=
  sorry

end NUMINAMATH_GPT_purchasing_plan_exists_l1318_131828


namespace NUMINAMATH_GPT_smallest_positive_integer_l1318_131890

theorem smallest_positive_integer :
  ∃ x : ℤ, 0 < x ∧ (x % 5 = 4) ∧ (x % 7 = 6) ∧ (x % 11 = 10) ∧ x = 384 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_l1318_131890


namespace NUMINAMATH_GPT_work_completion_time_l1318_131866

theorem work_completion_time :
  let work_rate_A := 1 / 8
  let work_rate_B := 1 / 6
  let work_rate_C := 1 / 4.8
  (work_rate_A + work_rate_B + work_rate_C) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_work_completion_time_l1318_131866


namespace NUMINAMATH_GPT_percentage_increase_is_50_l1318_131803

def initialNumber := 80
def finalNumber := 120

theorem percentage_increase_is_50 : ((finalNumber - initialNumber) / initialNumber : ℝ) * 100 = 50 := 
by 
  sorry

end NUMINAMATH_GPT_percentage_increase_is_50_l1318_131803


namespace NUMINAMATH_GPT_find_a8_l1318_131883

variable {a : ℕ → ℝ} -- Assuming the sequence is real-valued for generality

-- Defining the necessary properties and conditions of the arithmetic sequence.
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a n = a 0 + n * (a 1 - a 0)

-- Given conditions as hypothesis
variable (h_seq : arithmetic_sequence a) 
variable (h_sum : a 3 + a 6 + a 10 + a 13 = 32)

-- The proof statement
theorem find_a8 : a 8 = 8 :=
by
  sorry -- The proof itself

end NUMINAMATH_GPT_find_a8_l1318_131883


namespace NUMINAMATH_GPT_total_balloons_l1318_131861

theorem total_balloons (joan_balloons : ℕ) (melanie_balloons : ℕ) (h₁ : joan_balloons = 40) (h₂ : melanie_balloons = 41) : 
  joan_balloons + melanie_balloons = 81 := 
by
  sorry

end NUMINAMATH_GPT_total_balloons_l1318_131861


namespace NUMINAMATH_GPT_ab_bc_ca_leq_zero_l1318_131870

theorem ab_bc_ca_leq_zero (a b c : ℝ) (h : a + b + c = 0) : ab + bc + ca ≤ 0 :=
sorry

end NUMINAMATH_GPT_ab_bc_ca_leq_zero_l1318_131870


namespace NUMINAMATH_GPT_percent_of_dollar_in_pocket_l1318_131801

theorem percent_of_dollar_in_pocket :
  let nickel := 5
  let dime := 10
  let quarter := 25
  let half_dollar := 50
  (nickel + 2 * dime + quarter + half_dollar = 100) →
  (100 / 100 * 100 = 100) :=
by
  intros
  sorry

end NUMINAMATH_GPT_percent_of_dollar_in_pocket_l1318_131801


namespace NUMINAMATH_GPT_negation_of_exists_geq_prop_l1318_131878

open Classical

variable (P : Prop) (Q : Prop)

-- Original proposition:
def exists_geq_prop : Prop := 
  ∃ x : ℝ, x^2 + x + 1 ≥ 0

-- Its negation:
def forall_lt_neg : Prop :=
  ∀ x : ℝ, x^2 + x + 1 < 0

-- The theorem to prove:
theorem negation_of_exists_geq_prop : ¬ exists_geq_prop ↔ forall_lt_neg := 
by 
  -- The proof steps will be filled in here
  sorry

end NUMINAMATH_GPT_negation_of_exists_geq_prop_l1318_131878


namespace NUMINAMATH_GPT_even_function_periodicity_l1318_131887

noncomputable def f : ℝ → ℝ :=
sorry -- The actual function definition is not provided here but assumed to exist.

theorem even_function_periodicity (x : ℝ) (h1 : 1 ≤ x ∧ x ≤ 2)
  (h2 : f (x + 2) = f x)
  (hf_even : ∀ x, f x = f (-x))
  (hf_segment : ∀ x, 1 ≤ x ∧ x ≤ 2 → f x = x^2 + 2*x - 1) :
  ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x^2 - 6*x + 7 :=
sorry

end NUMINAMATH_GPT_even_function_periodicity_l1318_131887


namespace NUMINAMATH_GPT_usual_time_to_office_l1318_131810

theorem usual_time_to_office (P : ℝ) (T : ℝ) (h1 : T = (3 / 4) * (T + 20)) : T = 60 :=
by
  sorry

end NUMINAMATH_GPT_usual_time_to_office_l1318_131810


namespace NUMINAMATH_GPT_given_tan_alpha_eq_3_then_expression_eq_8_7_l1318_131813

theorem given_tan_alpha_eq_3_then_expression_eq_8_7 (α : ℝ) (h : Real.tan α = 3) :
  (6 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 8 / 7 := 
by
  sorry

end NUMINAMATH_GPT_given_tan_alpha_eq_3_then_expression_eq_8_7_l1318_131813


namespace NUMINAMATH_GPT_probability_of_both_qualified_bottles_expected_number_of_days_with_unqualified_milk_l1318_131811

noncomputable def qualification_rate : ℝ := 0.8
def probability_both_qualified (rate : ℝ) : ℝ := rate * rate
def unqualified_rate (rate : ℝ) : ℝ := 1 - rate
def expected_days (n : ℕ) (p : ℝ) : ℝ := n * p

theorem probability_of_both_qualified_bottles : 
  probability_both_qualified qualification_rate = 0.64 :=
by sorry

theorem expected_number_of_days_with_unqualified_milk :
  expected_days 3 (unqualified_rate qualification_rate) = 1.08 :=
by sorry

end NUMINAMATH_GPT_probability_of_both_qualified_bottles_expected_number_of_days_with_unqualified_milk_l1318_131811


namespace NUMINAMATH_GPT_range_of_k_l1318_131830

/-- If the function y = (k + 1) * x is decreasing on the entire real line, then k < -1. -/
theorem range_of_k (k : ℝ) (h : ∀ x y : ℝ, x < y → (k + 1) * x > (k + 1) * y) : k < -1 :=
sorry

end NUMINAMATH_GPT_range_of_k_l1318_131830


namespace NUMINAMATH_GPT_kenny_cost_per_book_l1318_131855

theorem kenny_cost_per_book (B : ℕ) :
  let lawn_charge := 15
  let mowed_lawns := 35
  let video_game_cost := 45
  let video_games := 5
  let total_earnings := lawn_charge * mowed_lawns
  let spent_on_video_games := video_game_cost * video_games
  let remaining_money := total_earnings - spent_on_video_games
  remaining_money / B = 300 / B :=
by
  sorry

end NUMINAMATH_GPT_kenny_cost_per_book_l1318_131855


namespace NUMINAMATH_GPT_reconstruct_point_A_l1318_131804

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A E' F' G' H' : V)

theorem reconstruct_point_A (E F G H : V) (p q r s : ℝ)
  (hE' : E' = 2 • F - E)
  (hF' : F' = 2 • G - F)
  (hG' : G' = 2 • H - G)
  (hH' : H' = 2 • E - H)
  : p = 1/4 ∧ q = 1/4  ∧ r = 1/4  ∧ s = 1/4  :=
by
  sorry

end NUMINAMATH_GPT_reconstruct_point_A_l1318_131804


namespace NUMINAMATH_GPT_fraction_zero_implies_x_eq_two_l1318_131881

theorem fraction_zero_implies_x_eq_two (x : ℝ) (h : (x^2 - 4) / (x + 2) = 0) : x = 2 :=
sorry

end NUMINAMATH_GPT_fraction_zero_implies_x_eq_two_l1318_131881


namespace NUMINAMATH_GPT_sum_square_divisors_positive_l1318_131864

theorem sum_square_divisors_positive (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b * c < 0) : 
  (a^2 + b^2) / c + (b^2 + c^2) / a + (c^2 + a^2) / b > 0 := 
by 
  sorry

end NUMINAMATH_GPT_sum_square_divisors_positive_l1318_131864


namespace NUMINAMATH_GPT_no_perfect_square_abc_sum_l1318_131842

theorem no_perfect_square_abc_sum (a b c : ℕ) (ha : 1 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) (hc : 0 ≤ c ∧ c ≤ 9) :
  ¬ ∃ m : ℕ, m * m = (100 * a + 10 * b + c) + (100 * b + 10 * c + a) + (100 * c + 10 * a + b) :=
by
  sorry

end NUMINAMATH_GPT_no_perfect_square_abc_sum_l1318_131842


namespace NUMINAMATH_GPT_tangent_product_l1318_131876

noncomputable section

open Real

theorem tangent_product 
  (x y k1 k2 : ℝ) :
  (x / 2) ^ 2 + y ^ 2 = 1 ∧ 
  (x, y) = (-3, -3) ∧ 
  k1 + k2 = 18 / 5 ∧
  k1 * k2 = 8 / 5 → 
  (3 * k1 - 3) * (3 * k2 - 3) = 9 := 
by
  intros 
  sorry

end NUMINAMATH_GPT_tangent_product_l1318_131876


namespace NUMINAMATH_GPT_lcm_of_36_48_75_l1318_131889

-- Definitions of the numbers and their factorizations
def num1 := 36
def num2 := 48
def num3 := 75

def factor_36 := (2^2, 3^2)
def factor_48 := (2^4, 3^1)
def factor_75 := (3^1, 5^2)

def highest_power_2 := 2^4
def highest_power_3 := 3^2
def highest_power_5 := 5^2

def lcm_36_48_75 := highest_power_2 * highest_power_3 * highest_power_5

-- The theorem statement
theorem lcm_of_36_48_75 : lcm_36_48_75 = 3600 := by
  sorry

end NUMINAMATH_GPT_lcm_of_36_48_75_l1318_131889


namespace NUMINAMATH_GPT_find_other_number_l1318_131865

-- Defining the two numbers and their properties
def sum_is_84 (a b : ℕ) : Prop := a + b = 84
def one_is_36 (a b : ℕ) : Prop := a = 36 ∨ b = 36
def other_is_48 (a b : ℕ) : Prop := a = 48 ∨ b = 48

-- The theorem statement
theorem find_other_number (a b : ℕ) (h1 : sum_is_84 a b) (h2 : one_is_36 a b) : other_is_48 a b :=
by {
  sorry
}

end NUMINAMATH_GPT_find_other_number_l1318_131865


namespace NUMINAMATH_GPT_expected_worth_is_1_33_l1318_131885

noncomputable def expected_worth_of_coin_flip : ℝ :=
  let prob_heads := 2 / 3
  let profit_heads := 5
  let prob_tails := 1 / 3
  let loss_tails := -6
  (prob_heads * profit_heads + prob_tails * loss_tails)

theorem expected_worth_is_1_33 : expected_worth_of_coin_flip = 1.33 := by
  sorry

end NUMINAMATH_GPT_expected_worth_is_1_33_l1318_131885


namespace NUMINAMATH_GPT_highland_baseball_club_members_l1318_131823

-- Define the given costs and expenditures.
def socks_cost : ℕ := 6
def tshirt_cost : ℕ := socks_cost + 7
def cap_cost : ℕ := socks_cost
def total_expenditure : ℕ := 5112
def home_game_cost : ℕ := socks_cost + tshirt_cost
def away_game_cost : ℕ := socks_cost + tshirt_cost + cap_cost
def cost_per_member : ℕ := home_game_cost + away_game_cost

theorem highland_baseball_club_members :
  total_expenditure / cost_per_member = 116 :=
by
  sorry

end NUMINAMATH_GPT_highland_baseball_club_members_l1318_131823


namespace NUMINAMATH_GPT_jimmy_bought_3_pens_l1318_131849

def cost_of_notebooks (num_notebooks : ℕ) (price_per_notebook : ℕ) : ℕ := num_notebooks * price_per_notebook
def cost_of_folders (num_folders : ℕ) (price_per_folder : ℕ) : ℕ := num_folders * price_per_folder
def total_cost (cost_notebooks cost_folders : ℕ) : ℕ := cost_notebooks + cost_folders
def total_spent (initial_money change : ℕ) : ℕ := initial_money - change
def cost_of_pens (total_spent amount_for_items : ℕ) : ℕ := total_spent - amount_for_items
def num_pens (cost_pens price_per_pen : ℕ) : ℕ := cost_pens / price_per_pen

theorem jimmy_bought_3_pens :
  let pen_price := 1
  let notebook_price := 3
  let num_notebooks := 4
  let folder_price := 5
  let num_folders := 2
  let initial_money := 50
  let change := 25
  let cost_notebooks := cost_of_notebooks num_notebooks notebook_price
  let cost_folders := cost_of_folders num_folders folder_price
  let total_items_cost := total_cost cost_notebooks cost_folders
  let amount_spent := total_spent initial_money change
  let pen_cost := cost_of_pens amount_spent total_items_cost
  num_pens pen_cost pen_price = 3 :=
by
  sorry

end NUMINAMATH_GPT_jimmy_bought_3_pens_l1318_131849


namespace NUMINAMATH_GPT_complete_square_transform_l1318_131874

theorem complete_square_transform (x : ℝ) : 
  x^2 - 2 * x = 9 ↔ (x - 1)^2 = 10 :=
by
  sorry

end NUMINAMATH_GPT_complete_square_transform_l1318_131874


namespace NUMINAMATH_GPT_trains_crossing_time_l1318_131806

noncomputable def timeToCross (L1 L2 : ℕ) (v1 v2 : ℕ) : ℝ :=
  let total_distance := (L1 + L2 : ℝ)
  let relative_speed := ((v1 + v2) * 1000 / 3600 : ℝ) -- converting km/hr to m/s
  total_distance / relative_speed

theorem trains_crossing_time :
  timeToCross 140 160 60 40 = 10.8 := 
  by 
    sorry

end NUMINAMATH_GPT_trains_crossing_time_l1318_131806


namespace NUMINAMATH_GPT_cake_heavier_than_bread_l1318_131834

-- Definitions
def weight_of_7_cakes_eq_1950_grams (C : ℝ) := 7 * C = 1950
def weight_of_5_cakes_12_breads_eq_2750_grams (C B : ℝ) := 5 * C + 12 * B = 2750

-- Statement
theorem cake_heavier_than_bread (C B : ℝ)
  (h1 : weight_of_7_cakes_eq_1950_grams C)
  (h2 : weight_of_5_cakes_12_breads_eq_2750_grams C B) :
  C - B = 165.47 :=
by {
  sorry
}

end NUMINAMATH_GPT_cake_heavier_than_bread_l1318_131834


namespace NUMINAMATH_GPT_arcsin_eq_pi_div_two_solve_l1318_131868

theorem arcsin_eq_pi_div_two_solve :
  ∀ (x : ℝ), (Real.arcsin x + Real.arcsin (3 * x) = Real.pi / 2) → x = Real.sqrt 10 / 10 :=
by
  intro x h
  sorry -- Proof is omitted as per instructions

end NUMINAMATH_GPT_arcsin_eq_pi_div_two_solve_l1318_131868


namespace NUMINAMATH_GPT_inequality_solution_empty_l1318_131852

theorem inequality_solution_empty {a : ℝ} :
  (∀ x : ℝ, ¬ (|x+2| + |x-1| < a)) ↔ a ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_empty_l1318_131852


namespace NUMINAMATH_GPT_total_cats_l1318_131858

-- Define the conditions as constants
def asleep_cats : ℕ := 92
def awake_cats : ℕ := 6

-- State the theorem that proves the total number of cats
theorem total_cats : asleep_cats + awake_cats = 98 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_total_cats_l1318_131858


namespace NUMINAMATH_GPT_find_a1_l1318_131837

-- Define the arithmetic sequence and the given conditions
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def geometric_mean (x y z : ℝ) : Prop :=
  y^2 = x * z

def problem_statement (a : ℕ → ℝ) (d : ℝ) : Prop :=
  (arithmetic_sequence a d) ∧ (geometric_mean (a 1) (a 2) (a 4))

theorem find_a1 (a : ℕ → ℝ) (d : ℝ) (h : problem_statement a d) : a 1 = 1 := by
  have h_seq : arithmetic_sequence a d := h.1
  have h_geom : geometric_mean (a 1) (a 2) (a 4) := h.2
  sorry

end NUMINAMATH_GPT_find_a1_l1318_131837


namespace NUMINAMATH_GPT_net_profit_from_plant_sales_l1318_131833

noncomputable def calculate_net_profit : ℝ :=
  let cost_basil := 2.00
  let cost_mint := 3.00
  let cost_zinnia := 7.00
  let cost_soil := 15.00
  let total_cost := cost_basil + cost_mint + cost_zinnia + cost_soil
  let basil_germinated := 20 * 0.80
  let mint_germinated := 15 * 0.75
  let zinnia_germinated := 10 * 0.70
  let revenue_healthy_basil := 12 * 5.00
  let revenue_small_basil := 8 * 3.00
  let revenue_healthy_mint := 10 * 6.00
  let revenue_small_mint := 4 * 4.00
  let revenue_healthy_zinnia := 5 * 10.00
  let revenue_small_zinnia := 2 * 7.00
  let total_revenue := revenue_healthy_basil + revenue_small_basil + revenue_healthy_mint + revenue_small_mint + revenue_healthy_zinnia + revenue_small_zinnia
  total_revenue - total_cost

theorem net_profit_from_plant_sales : calculate_net_profit = 197.00 := by
  sorry

end NUMINAMATH_GPT_net_profit_from_plant_sales_l1318_131833


namespace NUMINAMATH_GPT_total_items_count_l1318_131802

theorem total_items_count :
  let old_women  := 7
  let mules      := 7
  let bags       := 7
  let loaves     := 7
  let knives     := 7
  let sheaths    := 7
  let sheaths_per_loaf := knives * sheaths
  let sheaths_per_bag := loaves * sheaths_per_loaf
  let sheaths_per_mule := bags * sheaths_per_bag
  let sheaths_per_old_woman := mules * sheaths_per_mule
  let total_sheaths := old_women * sheaths_per_old_woman

  let loaves_per_bag := loaves
  let loaves_per_mule := bags * loaves_per_bag
  let loaves_per_old_woman := mules * loaves_per_mule
  let total_loaves := old_women * loaves_per_old_woman

  let knives_per_loaf := knives
  let knives_per_bag := loaves * knives_per_loaf
  let knives_per_mule := bags * knives_per_bag
  let knives_per_old_woman := mules * knives_per_mule
  let total_knives := old_women * knives_per_old_woman

  let total_bags := old_women * mules * bags

  let total_mules := old_women * mules

  let total_items := total_sheaths + total_loaves + total_knives + total_bags + total_mules + old_women

  total_items = 137256 :=
by
  sorry

end NUMINAMATH_GPT_total_items_count_l1318_131802


namespace NUMINAMATH_GPT_solve_inequalities_l1318_131851

theorem solve_inequalities (x : ℝ) (h1 : |4 - x| < 5) (h2 : x^2 < 36) : (-1 < x) ∧ (x < 6) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequalities_l1318_131851


namespace NUMINAMATH_GPT_prime_related_divisors_circle_l1318_131820

variables (n : ℕ)

-- Definitions of prime-related and conditions for n
def is_prime (p: ℕ): Prop := p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p
def prime_related (a b : ℕ) : Prop := 
  ∃ p : ℕ, is_prime p ∧ (a = p * b ∨ b = p * a)

-- The main statement to be proven
theorem prime_related_divisors_circle (n : ℕ) : 
  (n ≥ 3) ∧ (∀ a b, a ≠ b → (a ∣ n ∧ b ∣ n) → prime_related a b) ↔ ¬ (
    ∃ (p : ℕ) (k : ℕ), is_prime p ∧ (n = p ^ k) ∨ 
    ∃ (m : ℕ), n = m ^ 2 ) :=
sorry

end NUMINAMATH_GPT_prime_related_divisors_circle_l1318_131820


namespace NUMINAMATH_GPT_complement_of_union_is_singleton_five_l1318_131895

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set ℕ := {1, 2}

-- Define the set N
def N : Set ℕ := {3, 4}

-- Define the union of M and N
def M_union_N : Set ℕ := M ∪ N

-- Define the complement of union of M and N in U
def complement_U_M_union_N : Set ℕ := U \ M_union_N

-- State the theorem to be proved
theorem complement_of_union_is_singleton_five :
  complement_U_M_union_N = {5} :=
sorry

end NUMINAMATH_GPT_complement_of_union_is_singleton_five_l1318_131895


namespace NUMINAMATH_GPT_units_digit_17_pow_2024_l1318_131888

theorem units_digit_17_pow_2024 : (17 ^ 2024) % 10 = 1 := 
by
  sorry

end NUMINAMATH_GPT_units_digit_17_pow_2024_l1318_131888


namespace NUMINAMATH_GPT_count_valid_n_l1318_131871

theorem count_valid_n : ∃ (count : ℕ), count = 6 ∧ ∀ n : ℕ,
  0 < n ∧ n < 42 → (∃ m : ℕ, m > 0 ∧ n = 42 * m / (m + 1)) :=
by
  sorry

end NUMINAMATH_GPT_count_valid_n_l1318_131871


namespace NUMINAMATH_GPT_determine_colors_l1318_131817

-- Define the colors
inductive Color
| white
| red
| blue

open Color

-- Define the friends
inductive Friend
| Tamara 
| Valya
| Lida

open Friend

-- Define a function from Friend to their dress color and shoes color
def Dress : Friend → Color := sorry
def Shoes : Friend → Color := sorry

-- The problem conditions
axiom cond1 : Dress Tamara = Shoes Tamara
axiom cond2 : Shoes Valya = white
axiom cond3 : Dress Lida ≠ red
axiom cond4 : Shoes Lida ≠ red

-- The proof goal
theorem determine_colors :
  Dress Tamara = red ∧ Shoes Tamara = red ∧
  Dress Valya = blue ∧ Shoes Valya = white ∧
  Dress Lida = white ∧ Shoes Lida = blue :=
sorry

end NUMINAMATH_GPT_determine_colors_l1318_131817


namespace NUMINAMATH_GPT_shaded_area_l1318_131894

theorem shaded_area (x1 y1 x2 y2 x3 y3 : ℝ) 
  (vA vB vC vD vE vF : ℝ × ℝ)
  (h1 : vA = (0, 0))
  (h2 : vB = (0, 12))
  (h3 : vC = (12, 12))
  (h4 : vD = (12, 0))
  (h5 : vE = (24, 0))
  (h6 : vF = (18, 12))
  (h_base : 32 - 12 = 20)
  (h_height : 12 = 12) :
  (1 / 2 : ℝ) * 20 * 12 = 120 :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_l1318_131894


namespace NUMINAMATH_GPT_problem_l1318_131853

open Set

def M : Set ℝ := { x | -1 ≤ x ∧ x < 3 }
def N : Set ℝ := { x | x < 0 }
def complement_N : Set ℝ := { x | x ≥ 0 }

theorem problem : M ∩ complement_N = { x | 0 ≤ x ∧ x < 3 } :=
by
  sorry

end NUMINAMATH_GPT_problem_l1318_131853


namespace NUMINAMATH_GPT_sqrt_expression_eq_l1318_131826

theorem sqrt_expression_eq : 
  (Real.sqrt 18 / Real.sqrt 6 - Real.sqrt 12 + Real.sqrt 48 * Real.sqrt (1/3)) = -Real.sqrt 3 + 4 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_expression_eq_l1318_131826


namespace NUMINAMATH_GPT_division_of_powers_l1318_131843

variable {a : ℝ}

theorem division_of_powers (ha : a ≠ 0) : a^5 / a^3 = a^2 :=
by sorry

end NUMINAMATH_GPT_division_of_powers_l1318_131843


namespace NUMINAMATH_GPT_simplify_expression_l1318_131862

theorem simplify_expression : 4 * (15 / 7) * (21 / -45) = -4 :=
by 
    -- Lean's type system will verify the correctness of arithmetic simplifications.
    sorry

end NUMINAMATH_GPT_simplify_expression_l1318_131862


namespace NUMINAMATH_GPT_max_xy_l1318_131844

theorem max_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + 3 * y = 6) : xy ≤ 3 / 2 := sorry

end NUMINAMATH_GPT_max_xy_l1318_131844


namespace NUMINAMATH_GPT_largest_divisor_36_l1318_131856

theorem largest_divisor_36 (n : ℕ) (h : n > 0) (h_div : 36 ∣ n^3) : 6 ∣ n := 
sorry

end NUMINAMATH_GPT_largest_divisor_36_l1318_131856


namespace NUMINAMATH_GPT_smallest_solution_neg_two_l1318_131877

-- We set up the expressions and then state the smallest solution
def smallest_solution (x : ℝ) : Prop :=
  x * abs x = 3 * x + 2

theorem smallest_solution_neg_two :
  ∃ x : ℝ, smallest_solution x ∧ (∀ y : ℝ, smallest_solution y → y ≥ x) ∧ x = -2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_solution_neg_two_l1318_131877


namespace NUMINAMATH_GPT_value_of_expression_l1318_131818

theorem value_of_expression (x y : ℚ) (hx : x = 2/3) (hy : y = 5/8) : 
  (6 * x + 8 * y) / (48 * x * y) = 9 / 20 := 
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1318_131818


namespace NUMINAMATH_GPT_trapezium_other_side_length_l1318_131850

theorem trapezium_other_side_length (x : ℝ) : 
  (1 / 2) * (20 + x) * 13 = 247 → x = 18 :=
by
  sorry

end NUMINAMATH_GPT_trapezium_other_side_length_l1318_131850


namespace NUMINAMATH_GPT_incenter_divides_angle_bisector_2_1_l1318_131821

def is_incenter_divide_angle_bisector (AB BC AC : ℝ) (O : ℝ) : Prop :=
  AB = 15 ∧ BC = 12 ∧ AC = 18 → O = 2 / 1

theorem incenter_divides_angle_bisector_2_1 :
  is_incenter_divide_angle_bisector 15 12 18 (2 / 1) :=
by
  sorry

end NUMINAMATH_GPT_incenter_divides_angle_bisector_2_1_l1318_131821


namespace NUMINAMATH_GPT_tan_theta_eq_sqrt3_div_3_l1318_131816

theorem tan_theta_eq_sqrt3_div_3
  (θ : ℝ)
  (h : (Real.cos θ * Real.sqrt 3 + Real.sin θ) = 2) :
  Real.tan θ = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_GPT_tan_theta_eq_sqrt3_div_3_l1318_131816


namespace NUMINAMATH_GPT_finding_a_of_geometric_sequence_l1318_131880
noncomputable def geometric_sequence_a : Prop :=
  ∃ a : ℝ, (1, a, 2) = (1, a, 2) ∧ a^2 = 2

theorem finding_a_of_geometric_sequence :
  ∃ a : ℝ, (1, a, 2) = (1, a, 2) → a = Real.sqrt 2 ∨ a = -Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_finding_a_of_geometric_sequence_l1318_131880


namespace NUMINAMATH_GPT_negation_of_P_l1318_131822

open Real

theorem negation_of_P :
  (¬ (∀ x : ℝ, x > sin x)) ↔ (∃ x : ℝ, x ≤ sin x) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_P_l1318_131822


namespace NUMINAMATH_GPT_slope_of_parallel_line_l1318_131872

theorem slope_of_parallel_line (a b c : ℝ) (h : 3 * a - 6 * b = 12) :
  ∃ m, m = 1 / 2 := 
sorry

end NUMINAMATH_GPT_slope_of_parallel_line_l1318_131872


namespace NUMINAMATH_GPT_spent_on_video_games_l1318_131882

-- Defining the given amounts
def initial_amount : ℕ := 84
def grocery_spending : ℕ := 21
def final_amount : ℕ := 39

-- The proof statement: Proving Lenny spent $24 on video games.
theorem spent_on_video_games : initial_amount - final_amount - grocery_spending = 24 :=
by
  sorry

end NUMINAMATH_GPT_spent_on_video_games_l1318_131882


namespace NUMINAMATH_GPT_sum_of_ages_53_l1318_131899

variable (B D : ℕ)

def Ben_3_years_younger_than_Dan := B + 3 = D
def Ben_is_25 := B = 25
def sum_of_their_ages (B D : ℕ) := B + D

theorem sum_of_ages_53 : ∀ (B D : ℕ), Ben_3_years_younger_than_Dan B D → Ben_is_25 B → sum_of_their_ages B D = 53 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_ages_53_l1318_131899


namespace NUMINAMATH_GPT_production_statistics_relation_l1318_131846

noncomputable def a : ℚ := (10 + 12 + 14 + 14 + 15 + 15 + 16 + 17 + 17 + 17) / 10
noncomputable def b : ℚ := (15 + 15) / 2
noncomputable def c : ℤ := 17

theorem production_statistics_relation : c > a ∧ a > b :=
by
  sorry

end NUMINAMATH_GPT_production_statistics_relation_l1318_131846


namespace NUMINAMATH_GPT_find_four_digit_number_l1318_131807

def digits_sum (n : ℕ) : ℕ := (n / 1000) + (n / 100 % 10) + (n / 10 % 10) + (n % 10)
def digits_product (n : ℕ) : ℕ := (n / 1000) * (n / 100 % 10) * (n / 10 % 10) * (n % 10)

theorem find_four_digit_number :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (digits_sum n) * (digits_product n) = 3990 :=
by
  -- The proof is omitted as instructed.
  sorry

end NUMINAMATH_GPT_find_four_digit_number_l1318_131807


namespace NUMINAMATH_GPT_gcd_lcm_product_l1318_131859

theorem gcd_lcm_product (a b : ℕ) (h₀ : a = 15) (h₁ : b = 45) :
  Nat.gcd a b * Nat.lcm a b = 675 :=
by
  sorry

end NUMINAMATH_GPT_gcd_lcm_product_l1318_131859


namespace NUMINAMATH_GPT_find_principal_l1318_131812

theorem find_principal
  (R : ℝ) (hR : R = 0.05)
  (I : ℝ) (hI : I = 0.02)
  (A : ℝ) (hA : A = 1120)
  (n : ℕ) (hn : n = 6)
  (R' : ℝ) (hR' : R' = ((1 + R) / (1 + I)) - 1) :
  P = 938.14 :=
by
  have compound_interest_formula := A / (1 + R')^n
  sorry

end NUMINAMATH_GPT_find_principal_l1318_131812


namespace NUMINAMATH_GPT_solve_x_eq_10000_l1318_131845

theorem solve_x_eq_10000 (x : ℝ) (h : 5 * x^(1/4 : ℝ) - 3 * (x / x^(3/4 : ℝ)) = 10 + x^(1/4 : ℝ)) : x = 10000 :=
by
  sorry

end NUMINAMATH_GPT_solve_x_eq_10000_l1318_131845


namespace NUMINAMATH_GPT_mirror_area_l1318_131879

-- Defining the conditions as Lean functions and values
def frame_height : ℕ := 100
def frame_width : ℕ := 140
def frame_border : ℕ := 15

-- Statement to prove the area of the mirror
theorem mirror_area :
  let mirror_width := frame_width - 2 * frame_border
  let mirror_height := frame_height - 2 * frame_border
  mirror_width * mirror_height = 7700 :=
by
  sorry

end NUMINAMATH_GPT_mirror_area_l1318_131879


namespace NUMINAMATH_GPT_picnic_total_cost_is_correct_l1318_131898

-- Define the conditions given in the problem
def number_of_people : Nat := 4
def cost_per_sandwich : Nat := 5
def cost_per_fruit_salad : Nat := 3
def sodas_per_person : Nat := 2
def cost_per_soda : Nat := 2
def number_of_snack_bags : Nat := 3
def cost_per_snack_bag : Nat := 4

-- Calculate the total cost based on the given conditions
def total_cost_sandwiches : Nat := number_of_people * cost_per_sandwich
def total_cost_fruit_salads : Nat := number_of_people * cost_per_fruit_salad
def total_cost_sodas : Nat := number_of_people * sodas_per_person * cost_per_soda
def total_cost_snack_bags : Nat := number_of_snack_bags * cost_per_snack_bag

def total_spent : Nat := total_cost_sandwiches + total_cost_fruit_salads + total_cost_sodas + total_cost_snack_bags

-- The statement we want to prove
theorem picnic_total_cost_is_correct : total_spent = 60 :=
by
  -- Proof would be written here
  sorry

end NUMINAMATH_GPT_picnic_total_cost_is_correct_l1318_131898


namespace NUMINAMATH_GPT_water_level_in_cubic_tank_is_one_l1318_131838

def cubic_tank : Type := {s : ℝ // s > 0}

def water_volume (s : cubic_tank) : ℝ := 
  let ⟨side, _⟩ := s 
  side^3

def water_level (s : cubic_tank) (volume : ℝ) (fill_ratio : ℝ) : ℝ := 
  let ⟨side, _⟩ := s 
  fill_ratio * side

theorem water_level_in_cubic_tank_is_one
  (s : cubic_tank)
  (h1 : water_volume s = 64)
  (h2 : water_volume s / 4 = 16)
  (h3 : 0 < 0.25 ∧ 0.25 ≤ 1) :
  water_level s 16 0.25 = 1 :=
by 
  sorry

end NUMINAMATH_GPT_water_level_in_cubic_tank_is_one_l1318_131838


namespace NUMINAMATH_GPT_integer_solutions_count_l1318_131808

theorem integer_solutions_count :
  (∃ (n : ℕ), ∀ (x y : ℤ), x^2 + y^2 = 6 * x + 2 * y + 15 → n = 12) :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_count_l1318_131808


namespace NUMINAMATH_GPT_intersection_complement_l1318_131840

open Set

variable (U A B : Set ℕ)

-- Given conditions:
def universal_set : Set ℕ := {1, 2, 3, 4, 5, 6}
def set_A : Set ℕ := {1, 3, 5}
def set_B : Set ℕ := {2, 3}

theorem intersection_complement (U A B : Set ℕ) : 
  U = universal_set → A = set_A → B = set_B → (A ∩ (U \ B)) = {1, 5} := by
  sorry

end NUMINAMATH_GPT_intersection_complement_l1318_131840


namespace NUMINAMATH_GPT_find_x_for_salt_solution_l1318_131814

theorem find_x_for_salt_solution : ∀ (x : ℝ),
  (1 + x) * 0.10 = (x * 0.50) →
  x = 0.25 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_find_x_for_salt_solution_l1318_131814


namespace NUMINAMATH_GPT_square_area_ratio_l1318_131886

theorem square_area_ratio (s₁ s₂ d₂ : ℝ)
  (h1 : s₁ = 2 * d₂)
  (h2 : d₂ = s₂ * Real.sqrt 2) :
  (s₁^2) / (s₂^2) = 8 :=
by
  sorry

end NUMINAMATH_GPT_square_area_ratio_l1318_131886


namespace NUMINAMATH_GPT_physics_marks_l1318_131836

theorem physics_marks (P C M : ℕ) (h1 : P + C + M = 180) (h2 : P + M = 180) (h3 : P + C = 140) : P = 140 :=
by
  sorry

end NUMINAMATH_GPT_physics_marks_l1318_131836


namespace NUMINAMATH_GPT_car_sales_decrease_l1318_131809

theorem car_sales_decrease (P N : ℝ) (h1 : 1.30 * P / (N * (1 - D / 100)) = 1.8571 * (P / N)) : D = 30 :=
by
  sorry

end NUMINAMATH_GPT_car_sales_decrease_l1318_131809


namespace NUMINAMATH_GPT_total_amount_divided_l1318_131854

theorem total_amount_divided (A B C : ℝ) (h1 : A / B = 3 / 4) (h2 : B / C = 5 / 6) (h3 : A = 29491.525423728814) :
  A + B + C = 116000 := 
sorry

end NUMINAMATH_GPT_total_amount_divided_l1318_131854


namespace NUMINAMATH_GPT_problem_demo_l1318_131896

open Set

def U : Set ℕ := {x | 0 < x ∧ x ≤ 8}
def S : Set ℕ := {1, 2, 4, 5}
def T : Set ℕ := {3, 5, 7}

theorem problem_demo : S ∩ (U \ T) = {1, 2, 4} :=
by
  sorry

end NUMINAMATH_GPT_problem_demo_l1318_131896


namespace NUMINAMATH_GPT_prob_AB_diff_homes_l1318_131800

-- Define the volunteers
inductive Volunteer : Type
| A | B | C | D | E

open Volunteer

-- Define the homes
inductive Home : Type
| home1 | home2

open Home

-- Total number of ways to distribute the volunteers
def total_ways : ℕ := 2^5  -- Each volunteer has independently 2 choices

-- Number of ways in which A and B are in different homes
def diff_ways : ℕ := 2 * 4 * 2^3  -- Split the problem down by cases for simplicity

-- Calculate the probability
def probability : ℚ := diff_ways / total_ways

-- The final statement to prove
theorem prob_AB_diff_homes : probability = 8 / 15 := sorry

end NUMINAMATH_GPT_prob_AB_diff_homes_l1318_131800


namespace NUMINAMATH_GPT_ticket_sales_total_l1318_131869

variable (price_adult : ℕ) (price_child : ℕ) (total_tickets : ℕ) (child_tickets : ℕ)

def total_money_collected (price_adult : ℕ) (price_child : ℕ) (total_tickets : ℕ) (child_tickets : ℕ) : ℕ :=
  let adult_tickets := total_tickets - child_tickets
  let total_child := child_tickets * price_child
  let total_adult := adult_tickets * price_adult
  total_child + total_adult

theorem ticket_sales_total :
  price_adult = 6 →
  price_child = 4 →
  total_tickets = 21 →
  child_tickets = 11 →
  total_money_collected price_adult price_child total_tickets child_tickets = 104 :=
by
  intros
  unfold total_money_collected
  simp
  sorry

end NUMINAMATH_GPT_ticket_sales_total_l1318_131869


namespace NUMINAMATH_GPT_factorization_identity_l1318_131835

theorem factorization_identity (a b : ℝ) : (a^2 + b^2)^2 - 4 * a^2 * b^2 = (a + b)^2 * (a - b)^2 :=
by
  sorry

end NUMINAMATH_GPT_factorization_identity_l1318_131835


namespace NUMINAMATH_GPT_total_slices_at_picnic_l1318_131819

def danny_watermelons : ℕ := 3
def danny_slices_per_watermelon : ℕ := 10
def sister_watermelons : ℕ := 1
def sister_slices_per_watermelon : ℕ := 15

def total_danny_slices : ℕ := danny_watermelons * danny_slices_per_watermelon
def total_sister_slices : ℕ := sister_watermelons * sister_slices_per_watermelon
def total_slices : ℕ := total_danny_slices + total_sister_slices

theorem total_slices_at_picnic : total_slices = 45 :=
by
  sorry

end NUMINAMATH_GPT_total_slices_at_picnic_l1318_131819


namespace NUMINAMATH_GPT_val_total_money_l1318_131847

theorem val_total_money : 
  ∀ (nickels_initial dimes_initial nickels_found : ℕ),
    nickels_initial = 20 →
    dimes_initial = 3 * nickels_initial →
    nickels_found = 2 * nickels_initial →
    (nickels_initial * 5 + dimes_initial * 10 + nickels_found * 5) / 100 = 9 :=
by
  intros nickels_initial dimes_initial nickels_found h1 h2 h3
  sorry

end NUMINAMATH_GPT_val_total_money_l1318_131847


namespace NUMINAMATH_GPT_symmetric_line_eq_l1318_131860

-- Define points A and B
def A (a : ℝ) : ℝ × ℝ := (a-1, a+1)
def B (a : ℝ) : ℝ × ℝ := (a, a)

-- We want to prove the equation of the line L about which points A and B are symmetric is "x - y + 1 = 0".
theorem symmetric_line_eq (a : ℝ) : 
  ∃ m b, (m = 1) ∧ (b = 1) ∧ (∀ x y, (y = m * x + b) ↔ (x - y + 1 = 0)) :=
sorry

end NUMINAMATH_GPT_symmetric_line_eq_l1318_131860


namespace NUMINAMATH_GPT_difference_in_cm_l1318_131805

def line_length : ℝ := 80  -- The length of the line is 80.0 centimeters
def diff_length_factor : ℝ := 0.35  -- The difference factor in the terms of the line's length

theorem difference_in_cm (l : ℝ) (d : ℝ) (h₀ : l = 80) (h₁ : d = 0.35 * l) : d = 28 :=
by
  sorry

end NUMINAMATH_GPT_difference_in_cm_l1318_131805


namespace NUMINAMATH_GPT_expression_value_l1318_131873

theorem expression_value (a b c d : ℤ) (h_a : a = 15) (h_b : b = 19) (h_c : c = 3) (h_d : d = 2) :
  (a - (b - c)) - ((a - b) - c + d) = 4 := 
by
  rw [h_a, h_b, h_c, h_d]
  sorry

end NUMINAMATH_GPT_expression_value_l1318_131873


namespace NUMINAMATH_GPT_max_lambda_leq_64_div_27_l1318_131867

theorem max_lambda_leq_64_div_27 (a b c : ℝ) (ha : 0 < a ∧ a ≤ 1) (hb : 0 < b ∧ b ≤ 1) (hc : 0 < c ∧ c ≤ 1) :
  (1:ℝ) + (64:ℝ) / (27:ℝ) * (1 - a) * (1 - b) * (1 - c) ≤ Real.sqrt 3 / Real.sqrt (a + b + c) := 
sorry

end NUMINAMATH_GPT_max_lambda_leq_64_div_27_l1318_131867


namespace NUMINAMATH_GPT_inequality_solution_l1318_131841

theorem inequality_solution (x : ℝ) : (1 - 3 * (x - 1) < x) ↔ (x > 1) :=
by sorry

end NUMINAMATH_GPT_inequality_solution_l1318_131841


namespace NUMINAMATH_GPT_canteen_needs_bananas_l1318_131893

-- Define the given conditions
def total_bananas := 9828
def weeks := 9
def days_in_week := 7
def bananas_in_dozen := 12

-- Calculate the required value and prove the equivalence
theorem canteen_needs_bananas : 
  (total_bananas / (weeks * days_in_week)) / bananas_in_dozen = 13 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_canteen_needs_bananas_l1318_131893


namespace NUMINAMATH_GPT_point_not_in_second_quadrant_l1318_131875

-- Define the point P and the condition
def point_is_in_second_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 < 0 ∧ P.2 > 0

def point (m : ℝ) : ℝ × ℝ :=
  (m + 1, m)

-- The main theorem stating that P cannot be in the second quadrant
theorem point_not_in_second_quadrant (m : ℝ) : ¬ point_is_in_second_quadrant (point m) :=
by
  sorry

end NUMINAMATH_GPT_point_not_in_second_quadrant_l1318_131875


namespace NUMINAMATH_GPT_tight_sequence_from_sum_of_terms_range_of_q_for_tight_sequences_l1318_131824

def tight_sequence (a : ℕ → ℚ) : Prop :=
∀ n : ℕ, n > 0 → (1/2 : ℚ) ≤ a (n + 1) / a n ∧ a (n + 1) / a n ≤ 2

def sum_of_first_n_terms (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
∀ n : ℕ, n > 0 → S n = (1 / 4) * (n^2 + 3 * n)

noncomputable def geometric_sequence (a : ℕ → ℚ) (q : ℚ) : Prop :=
∀ n : ℕ, n > 0 → a n = a 1 * q ^ (n - 1)

theorem tight_sequence_from_sum_of_terms (S : ℕ → ℚ) (a : ℕ → ℚ) : 
  (∀ n : ℕ, n > 0 → S n = (1 / 4) * (n^2 + 3 * n)) →
  (∀ n : ℕ, n > 0 → a n = S n - S (n - 1)) →
  tight_sequence a :=
sorry

theorem range_of_q_for_tight_sequences (a : ℕ → ℚ) (S : ℕ → ℚ) (q : ℚ) :
  geometric_sequence a q →
  tight_sequence a →
  tight_sequence S →
  (1 / 2 : ℚ) ≤ q ∧ q < 1 :=
sorry

end NUMINAMATH_GPT_tight_sequence_from_sum_of_terms_range_of_q_for_tight_sequences_l1318_131824


namespace NUMINAMATH_GPT_find_p_l1318_131825

-- Assume the parametric equations and conditions specified in the problem.
noncomputable def parabola_eqns (p t : ℝ) (M E F : ℝ × ℝ) :=
  ∃ m : ℝ,
    (M = (6, m)) ∧
    (E = (-p / 2, m)) ∧
    (F = (p / 2, 0)) ∧
    (m^2 = 6 * p) ∧
    (|E.1 - F.1|^2 + |E.2 - F.2|^2 = |F.1 - M.1|^2 + |F.2 - M.2|^2) ∧
    (|F.1 - M.1|^2 + |F.2 - M.2|^2 = (F.1 + p / 2)^2 + (F.2 - m)^2)

theorem find_p {p t : ℝ} {M E F : ℝ × ℝ} (h : parabola_eqns p t M E F) : p = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_p_l1318_131825


namespace NUMINAMATH_GPT_exists_x_given_y_l1318_131831

theorem exists_x_given_y (y : ℝ) : ∃ x : ℝ, x^2 + y^2 = 10 ∧ x^2 - x * y - 3 * y + 12 = 0 := 
sorry

end NUMINAMATH_GPT_exists_x_given_y_l1318_131831


namespace NUMINAMATH_GPT_inverse_of_f_at_10_l1318_131848

noncomputable def f (x : ℝ) : ℝ := 1 + 3^(-x)

theorem inverse_of_f_at_10 :
  f⁻¹ 10 = -2 :=
sorry

end NUMINAMATH_GPT_inverse_of_f_at_10_l1318_131848


namespace NUMINAMATH_GPT_cantaloupe_total_l1318_131815

theorem cantaloupe_total (Fred Tim Alicia : ℝ) 
  (hFred : Fred = 38.5) 
  (hTim : Tim = 44.2)
  (hAlicia : Alicia = 29.7) : 
  Fred + Tim + Alicia = 112.4 :=
by
  sorry

end NUMINAMATH_GPT_cantaloupe_total_l1318_131815


namespace NUMINAMATH_GPT_triangle_XYZ_r_s_max_sum_l1318_131884

theorem triangle_XYZ_r_s_max_sum
  (r s : ℝ)
  (h_area : 1/2 * abs (r * (15 - 18) + 10 * (18 - s) + 20 * (s - 15)) = 90)
  (h_slope : s = -3 * r + 61.5) :
  r + s ≤ 42.91 :=
sorry

end NUMINAMATH_GPT_triangle_XYZ_r_s_max_sum_l1318_131884


namespace NUMINAMATH_GPT_numbers_sum_prod_l1318_131863

theorem numbers_sum_prod (x y : ℝ) (h_sum : x + y = 10) (h_prod : x * y = 24) : (x = 4 ∧ y = 6) ∨ (x = 6 ∧ y = 4) :=
by
  sorry

end NUMINAMATH_GPT_numbers_sum_prod_l1318_131863


namespace NUMINAMATH_GPT_find_values_of_a_l1318_131832

def P : Set ℝ := { x | x^2 + x - 6 = 0 }
def S (a : ℝ) : Set ℝ := { x | a * x + 1 = 0 }

theorem find_values_of_a (a : ℝ) : (S a ⊆ P) ↔ (a = 0 ∨ a = 1/3 ∨ a = -1/2) := by
  sorry

end NUMINAMATH_GPT_find_values_of_a_l1318_131832


namespace NUMINAMATH_GPT_solve_inequality_l1318_131829

theorem solve_inequality (a x : ℝ) :
  (a > 0 → (a - 1) / a < x ∧ x < 1) ∧ 
  (a = 0 → x < 1) ∧ 
  (a < 0 → x > (a - 1) / a ∨ x < 1) ↔ 
  (ax / (x - 1) < (a - 1) / (x - 1)) :=
sorry

end NUMINAMATH_GPT_solve_inequality_l1318_131829


namespace NUMINAMATH_GPT_cost_per_ticket_l1318_131897

/-- Adam bought 13 tickets and after riding the ferris wheel, he had 4 tickets left.
    He spent 81 dollars riding the ferris wheel, and we want to determine how much each ticket cost. -/
theorem cost_per_ticket (initial_tickets : ℕ) (tickets_left : ℕ) (total_cost : ℕ) (used_tickets : ℕ) 
    (ticket_cost : ℕ) (h1 : initial_tickets = 13) 
    (h2 : tickets_left = 4) 
    (h3 : total_cost = 81) 
    (h4 : used_tickets = initial_tickets - tickets_left) 
    (h5 : ticket_cost = total_cost / used_tickets) : ticket_cost = 9 :=
by {
    sorry
}

end NUMINAMATH_GPT_cost_per_ticket_l1318_131897


namespace NUMINAMATH_GPT_roots_real_and_equal_l1318_131857

theorem roots_real_and_equal :
  ∀ x : ℝ,
  (x^2 - 4 * x * Real.sqrt 5 + 20 = 0) →
  (Real.sqrt ((-4 * Real.sqrt 5)^2 - 4 * 1 * 20) = 0) →
  (∃ r : ℝ, x = r ∧ x = r) :=
by
  intro x h_eq h_discriminant
  sorry

end NUMINAMATH_GPT_roots_real_and_equal_l1318_131857


namespace NUMINAMATH_GPT_percentage_enclosed_by_pentagons_l1318_131827

-- Define the condition for the large square and smaller squares.
def large_square_area (b : ℝ) : ℝ := (4 * b) ^ 2

-- Define the condition for the number of smaller squares forming pentagons.
def pentagon_small_squares : ℝ := 10

-- Define the total number of smaller squares within a large square.
def total_small_squares : ℝ := 16

-- Prove that the percentage of the plane enclosed by pentagons is 62.5%.
theorem percentage_enclosed_by_pentagons :
  (pentagon_small_squares / total_small_squares) * 100 = 62.5 :=
by 
  -- The proof is left as an exercise.
  sorry

end NUMINAMATH_GPT_percentage_enclosed_by_pentagons_l1318_131827


namespace NUMINAMATH_GPT_probability_of_drawing_2_red_1_white_l1318_131892

def total_balls : ℕ := 7
def red_balls : ℕ := 4
def white_balls : ℕ := 3
def draws : ℕ := 3

def combinations (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_of_drawing_2_red_1_white :
  (combinations red_balls 2) * (combinations white_balls 1) / (combinations total_balls draws) = 18 / 35 := by
  sorry

end NUMINAMATH_GPT_probability_of_drawing_2_red_1_white_l1318_131892


namespace NUMINAMATH_GPT_pumps_work_hours_l1318_131839

theorem pumps_work_hours (d : ℕ) (h_d_pos : d > 0) : 6 * (8 / d) * d = 48 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_pumps_work_hours_l1318_131839


namespace NUMINAMATH_GPT_stacy_days_to_complete_paper_l1318_131891

-- Conditions as definitions
def total_pages : ℕ := 63
def pages_per_day : ℕ := 9

-- The problem statement
theorem stacy_days_to_complete_paper : total_pages / pages_per_day = 7 :=
by
  sorry

end NUMINAMATH_GPT_stacy_days_to_complete_paper_l1318_131891
