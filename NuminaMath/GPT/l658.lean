import Mathlib

namespace NUMINAMATH_GPT_world_grain_demand_l658_65858

theorem world_grain_demand (S D : ℝ) (h1 : S = 1800000) (h2 : S = 0.75 * D) : D = 2400000 := by
  sorry

end NUMINAMATH_GPT_world_grain_demand_l658_65858


namespace NUMINAMATH_GPT_intersection_complement_eq_l658_65866

noncomputable def U : Set Int := {-3, -2, -1, 0, 1, 2, 3}
noncomputable def A : Set Int := {-1, 0, 1, 2}
noncomputable def B : Set Int := {-3, 0, 2, 3}

-- Complement of B with respect to U
noncomputable def U_complement_B : Set Int := U \ B

-- The statement we need to prove
theorem intersection_complement_eq :
  A ∩ U_complement_B = {-1, 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_eq_l658_65866


namespace NUMINAMATH_GPT_ten_term_sequence_l658_65860
open Real

theorem ten_term_sequence (a b : ℝ) 
    (h₁ : a + b = 1)
    (h₂ : a^2 + b^2 = 3)
    (h₃ : a^3 + b^3 = 4)
    (h₄ : a^4 + b^4 = 7)
    (h₅ : a^5 + b^5 = 11) :
    a^10 + b^10 = 123 :=
  sorry

end NUMINAMATH_GPT_ten_term_sequence_l658_65860


namespace NUMINAMATH_GPT_smallest_k_l658_65840

theorem smallest_k (a b c d e k : ℕ) (h1 : a + 2 * b + 3 * c + 4 * d + 5 * e = k)
  (h2 : 5 * a = 4 * b) (h3 : 4 * b = 3 * c) (h4 : 3 * c = 2 * d) (h5 : 2 * d = e) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) : k = 522 :=
sorry

end NUMINAMATH_GPT_smallest_k_l658_65840


namespace NUMINAMATH_GPT_music_class_uncool_parents_l658_65837

theorem music_class_uncool_parents:
  ∀ (total students coolDads coolMoms bothCool : ℕ),
  total = 40 →
  coolDads = 25 →
  coolMoms = 19 →
  bothCool = 8 →
  (total - (bothCool + (coolDads - bothCool) + (coolMoms - bothCool))) = 4 :=
by
  intros total coolDads coolMoms bothCool h_total h_dads h_moms h_both
  sorry

end NUMINAMATH_GPT_music_class_uncool_parents_l658_65837


namespace NUMINAMATH_GPT_bella_started_with_136_candies_l658_65814

/-
Theorem:
Bella started with 136 candies.
-/

-- define the initial number of candies
variable (x : ℝ)

-- define the conditions
def condition1 : Prop := (x / 2 - 3 / 4) - 5 = 9
def condition2 : Prop := x = 136

-- structure the proof statement 
theorem bella_started_with_136_candies : condition1 x -> condition2 x :=
by
  sorry

end NUMINAMATH_GPT_bella_started_with_136_candies_l658_65814


namespace NUMINAMATH_GPT_contrapositive_x_squared_l658_65887

theorem contrapositive_x_squared :
  (∀ x : ℝ, x^2 < 1 → -1 < x ∧ x < 1) ↔ (∀ x : ℝ, x ≤ -1 ∨ x ≥ 1 → x^2 ≥ 1) := 
sorry

end NUMINAMATH_GPT_contrapositive_x_squared_l658_65887


namespace NUMINAMATH_GPT_sulfuric_acid_moles_used_l658_65876

-- Definitions and conditions
def iron_moles : ℕ := 2
def iron_ii_sulfate_moles_produced : ℕ := 2
def sulfuric_acid_to_iron_ratio : ℕ := 1

-- Proof statement
theorem sulfuric_acid_moles_used {H2SO4_moles : ℕ} 
  (h_fe_reacts : H2SO4_moles = iron_moles * sulfuric_acid_to_iron_ratio) 
  (h_fe produces: iron_ii_sulfate_moles_produced = iron_moles) : H2SO4_moles = 2 :=
by
  sorry

end NUMINAMATH_GPT_sulfuric_acid_moles_used_l658_65876


namespace NUMINAMATH_GPT_prove_x_minus_y_squared_l658_65826

variable (x y : ℝ)
variable (h1 : (x + y)^2 = 64)
variable (h2 : x * y = 12)

theorem prove_x_minus_y_squared : (x - y)^2 = 16 :=
by
  sorry

end NUMINAMATH_GPT_prove_x_minus_y_squared_l658_65826


namespace NUMINAMATH_GPT_solutions_eq_l658_65881

theorem solutions_eq :
  { (a, b, c) : ℕ × ℕ × ℕ | a * b + b * c + c * a = 2 * (a + b + c) } =
  { (2, 2, 2),
    (1, 2, 4), (1, 4, 2), 
    (2, 1, 4), (2, 4, 1),
    (4, 1, 2), (4, 2, 1) } :=
by sorry

end NUMINAMATH_GPT_solutions_eq_l658_65881


namespace NUMINAMATH_GPT_inequality_condition_sufficient_l658_65829

theorem inequality_condition_sufficient (A B C : ℝ) (x y z : ℝ) 
  (hA : 0 ≤ A) 
  (hB : 0 ≤ B) 
  (hC : 0 ≤ C) 
  (hABC : A^2 + B^2 + C^2 ≤ 2 * (A * B + A * C + B * C)) :
  A * (x - y) * (x - z) + B * (y - z) * (y - x) + C * (z - x) * (z - y) ≥ 0 :=
sorry

end NUMINAMATH_GPT_inequality_condition_sufficient_l658_65829


namespace NUMINAMATH_GPT_harold_grocery_expense_l658_65897

theorem harold_grocery_expense:
  ∀ (income rent car_payment savings utilities remaining groceries : ℝ),
    income = 2500 →
    rent = 700 →
    car_payment = 300 →
    utilities = 0.5 * car_payment →
    remaining = income - rent - car_payment - utilities →
    savings = 0.5 * remaining →
    (remaining - savings) = 650 →
    groceries = (remaining - 650) →
    groceries = 50 :=
by
  intros income rent car_payment savings utilities remaining groceries
  intro h_income
  intro h_rent
  intro h_car_payment
  intro h_utilities
  intro h_remaining
  intro h_savings
  intro h_final_remaining
  intro h_groceries
  sorry

end NUMINAMATH_GPT_harold_grocery_expense_l658_65897


namespace NUMINAMATH_GPT_max_area_dog_roam_l658_65885

theorem max_area_dog_roam (r : ℝ) (s : ℝ) (half_s : ℝ) (midpoint : Prop) :
  r = 10 → s = 20 → half_s = s / 2 → midpoint → 
  r > half_s → 
  π * r^2 = 100 * π :=
by 
  intros hr hs h_half_s h_midpoint h_rope_length
  sorry

end NUMINAMATH_GPT_max_area_dog_roam_l658_65885


namespace NUMINAMATH_GPT_inequality_f_l658_65893

-- Definitions of the given conditions
def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a*x + b

-- Theorem statement
theorem inequality_f (a b : ℝ) : 
  abs (f 1 a b) + 2 * abs (f 2 a b) + abs (f 3 a b) ≥ 2 :=
by sorry

end NUMINAMATH_GPT_inequality_f_l658_65893


namespace NUMINAMATH_GPT_arithmetic_sequence_property_l658_65879

variable {a : ℕ → ℝ} -- Let a be an arithmetic sequence
variable {S : ℕ → ℝ} -- Let S be the sum of the first n terms of the sequence

-- Conditions
axiom sum_of_first_n_terms (n : ℕ) : S n = n * (a 1 + (n - 1) * (a 2 - a 1) / 2)
axiom a_5 : a 5 = 3
axiom S_13 : S 13 = 91

-- Question to prove
theorem arithmetic_sequence_property : a 1 + a 11 = 10 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_property_l658_65879


namespace NUMINAMATH_GPT_intersection_A_B_eq_B_l658_65830

-- Define set A
def setA : Set ℝ := { x : ℝ | x > -3 }

-- Define set B
def setB : Set ℝ := { x : ℝ | x ≥ 2 }

-- Theorem statement of proving the intersection of setA and setB is setB itself
theorem intersection_A_B_eq_B : setA ∩ setB = setB :=
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_intersection_A_B_eq_B_l658_65830


namespace NUMINAMATH_GPT_minimum_value_expression_l658_65804

theorem minimum_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (4 * z / (2 * x + y)) + (4 * x / (y + 2 * z)) + (y / (x + z)) ≥ 3 :=
by 
  sorry

end NUMINAMATH_GPT_minimum_value_expression_l658_65804


namespace NUMINAMATH_GPT_find_y_l658_65846

theorem find_y (x y : ℤ) (h1 : x + y = 260) (h2 : x - y = 200) : y = 30 :=
sorry

end NUMINAMATH_GPT_find_y_l658_65846


namespace NUMINAMATH_GPT_clock_hand_positions_l658_65818

theorem clock_hand_positions : ∃ n : ℕ, n = 143 ∧ 
  (∀ t : ℝ, let hour_pos := t / 12
            let min_pos := t
            let switched_hour_pos := t
            let switched_min_pos := t / 12
            hour_pos = switched_min_pos ∧ min_pos = switched_hour_pos ↔
            ∃ k : ℤ, t = k / 11) :=
by sorry

end NUMINAMATH_GPT_clock_hand_positions_l658_65818


namespace NUMINAMATH_GPT_cuboid_height_l658_65848

-- Define the necessary constants
def width : ℕ := 30
def length : ℕ := 22
def sum_edges : ℕ := 224

-- Theorem stating the height of the cuboid
theorem cuboid_height (h : ℕ) : 4 * length + 4 * width + 4 * h = sum_edges → h = 4 := by
  sorry

end NUMINAMATH_GPT_cuboid_height_l658_65848


namespace NUMINAMATH_GPT_fiona_pairs_l658_65899

theorem fiona_pairs : Nat.choose 12 2 = 66 := by
  sorry

end NUMINAMATH_GPT_fiona_pairs_l658_65899


namespace NUMINAMATH_GPT_trajectory_eq_l658_65823

theorem trajectory_eq {x y : ℝ} (h₁ : (x-2)^2 + y^2 = 1) (h₂ : ∃ r, (x+1)^2 = (x-2)^2 + y^2 - r^2) :
  y^2 = 6 * x - 3 :=
by
  sorry

end NUMINAMATH_GPT_trajectory_eq_l658_65823


namespace NUMINAMATH_GPT_net_amount_spent_correct_l658_65832

def trumpet_cost : ℝ := 145.16
def song_book_revenue : ℝ := 5.84
def net_amount_spent : ℝ := 139.32

theorem net_amount_spent_correct : trumpet_cost - song_book_revenue = net_amount_spent :=
by
  sorry

end NUMINAMATH_GPT_net_amount_spent_correct_l658_65832


namespace NUMINAMATH_GPT_factor_polynomial_l658_65833

noncomputable def polynomial (x y n : ℤ) : ℤ := x^2 + 4 * x * y + 2 * x + n * y - n

theorem factor_polynomial (n : ℤ) :
  (∃ A B C D E F : ℤ, polynomial A B C = (A * x + B * y + C) * (D * x + E * y + F)) ↔ n = 0 :=
sorry

end NUMINAMATH_GPT_factor_polynomial_l658_65833


namespace NUMINAMATH_GPT_percentage_increase_in_consumption_l658_65882

theorem percentage_increase_in_consumption 
  (T C : ℝ) 
  (h1 : 0.8 * T * C * (1 + P / 100) = 0.88 * T * C)
  : P = 10 := 
by 
  sorry

end NUMINAMATH_GPT_percentage_increase_in_consumption_l658_65882


namespace NUMINAMATH_GPT_f_1982_value_l658_65874

noncomputable def f (n : ℕ) : ℕ := sorry  -- placeholder for the function definition

axiom f_condition_2 : f 2 = 0
axiom f_condition_3 : f 3 > 0
axiom f_condition_9999 : f 9999 = 3333
axiom f_add_condition (m n : ℕ) : f (m+n) - f m - f n = 0 ∨ f (m+n) - f m - f n = 1

open Nat

theorem f_1982_value : f 1982 = 660 :=
by
  sorry  -- proof goes here

end NUMINAMATH_GPT_f_1982_value_l658_65874


namespace NUMINAMATH_GPT_sum_sequence_formula_l658_65898

theorem sum_sequence_formula (a : ℕ → ℚ) (S : ℕ → ℚ) :
  (∀ n : ℕ, n > 0 → S n = n^2 * a n) ∧ a 1 = 1 →
  ∀ n : ℕ, n > 0 → S n = 2 * n / (n + 1) :=
by sorry

end NUMINAMATH_GPT_sum_sequence_formula_l658_65898


namespace NUMINAMATH_GPT_expr_simplified_l658_65828

theorem expr_simplified : |2 - Real.sqrt 2| - Real.sqrt (1 / 12) * Real.sqrt 27 + Real.sqrt 12 / Real.sqrt 6 = 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_expr_simplified_l658_65828


namespace NUMINAMATH_GPT_new_paint_intensity_l658_65815

def red_paint_intensity (initial_intensity replacement_intensity : ℝ) (replacement_fraction : ℝ) : ℝ :=
  (1 - replacement_fraction) * initial_intensity + replacement_fraction * replacement_intensity

theorem new_paint_intensity :
  red_paint_intensity 0.1 0.2 0.5 = 0.15 :=
by sorry

end NUMINAMATH_GPT_new_paint_intensity_l658_65815


namespace NUMINAMATH_GPT_sqrt_difference_inequality_l658_65842

noncomputable def sqrt10 := Real.sqrt 10
noncomputable def sqrt6 := Real.sqrt 6
noncomputable def sqrt7 := Real.sqrt 7
noncomputable def sqrt3 := Real.sqrt 3

theorem sqrt_difference_inequality : sqrt10 - sqrt6 < sqrt7 - sqrt3 :=
by 
  sorry

end NUMINAMATH_GPT_sqrt_difference_inequality_l658_65842


namespace NUMINAMATH_GPT_no_such_pairs_exist_l658_65801

theorem no_such_pairs_exist : ¬ ∃ (n m : ℕ), n > 1 ∧ (∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n) ∧ 
                                    (∀ d : ℕ, d ≠ n → d ∣ n → d + 1 ∣ m ∧ d + 1 ≠ m ∧ d + 1 ≠ 1) :=
by
  sorry

end NUMINAMATH_GPT_no_such_pairs_exist_l658_65801


namespace NUMINAMATH_GPT_total_investment_is_correct_l658_65825

def Raghu_investment : ℕ := 2300
def Trishul_investment (Raghu_investment : ℕ) : ℕ := Raghu_investment - (Raghu_investment / 10)
def Vishal_investment (Trishul_investment : ℕ) : ℕ := Trishul_investment + (Trishul_investment / 10)

theorem total_investment_is_correct :
    let Raghu_inv := Raghu_investment;
    let Trishul_inv := Trishul_investment Raghu_inv;
    let Vishal_inv := Vishal_investment Trishul_inv;
    Raghu_inv + Trishul_inv + Vishal_inv = 6647 :=
by
    sorry

end NUMINAMATH_GPT_total_investment_is_correct_l658_65825


namespace NUMINAMATH_GPT_find_r_from_tan_cosine_tangent_l658_65867

theorem find_r_from_tan_cosine_tangent 
  (θ : ℝ) 
  (r : ℝ) 
  (htan : Real.tan θ = -7 / 24) 
  (hquadrant : π / 2 < θ ∧ θ < π) 
  (hr : 100 * Real.cos θ = r) : 
  r = -96 := 
sorry

end NUMINAMATH_GPT_find_r_from_tan_cosine_tangent_l658_65867


namespace NUMINAMATH_GPT_reciprocal_of_neg_eight_l658_65807

theorem reciprocal_of_neg_eight : -8 * (-1/8) = 1 := 
by
  sorry

end NUMINAMATH_GPT_reciprocal_of_neg_eight_l658_65807


namespace NUMINAMATH_GPT_determine_a_l658_65892

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + 2 * x

theorem determine_a (a : ℝ) : (∀ x : ℝ, f a (-x) = -f a x) → a = 0 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_determine_a_l658_65892


namespace NUMINAMATH_GPT_exponent_multiplication_l658_65819

theorem exponent_multiplication :
  (10 ^ 10000) * (10 ^ 8000) = 10 ^ 18000 :=
by
  sorry

end NUMINAMATH_GPT_exponent_multiplication_l658_65819


namespace NUMINAMATH_GPT_find_percentage_l658_65888

theorem find_percentage (x p : ℝ) (h₀ : x = 780) (h₁ : 0.25 * x = (p / 100) * 1500 - 30) : p = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_percentage_l658_65888


namespace NUMINAMATH_GPT_count_pairs_l658_65855

theorem count_pairs (a b : ℤ) (ha : 1 ≤ a ∧ a ≤ 42) (hb : 1 ≤ b ∧ b ≤ 42) (h : a^9 % 43 = b^7 % 43) : (∃ (n : ℕ), n = 42) :=
  sorry

end NUMINAMATH_GPT_count_pairs_l658_65855


namespace NUMINAMATH_GPT_icosagon_diagonals_l658_65811

-- Definitions for the number of sides and the diagonal formula
def sides_icosagon : ℕ := 20

def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Statement:
theorem icosagon_diagonals : diagonals sides_icosagon = 170 := by
  apply sorry

end NUMINAMATH_GPT_icosagon_diagonals_l658_65811


namespace NUMINAMATH_GPT_min_value_l658_65844

theorem min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) : 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ (1 / x + 4 / y) ≥ 9) :=
by
  sorry

end NUMINAMATH_GPT_min_value_l658_65844


namespace NUMINAMATH_GPT_approx_sum_l658_65820

-- Definitions of the costs
def cost_bicycle : ℕ := 389
def cost_fan : ℕ := 189

-- Definition of the approximations
def approx_bicycle : ℕ := 400
def approx_fan : ℕ := 200

-- The statement to prove
theorem approx_sum (h₁ : cost_bicycle = 389) (h₂ : cost_fan = 189) : 
  approx_bicycle + approx_fan = 600 := 
by 
  sorry

end NUMINAMATH_GPT_approx_sum_l658_65820


namespace NUMINAMATH_GPT_solution_of_inequality_system_l658_65802

theorem solution_of_inequality_system (x : ℝ) : 
  (x - 1 < 0) ∧ (x + 1 > 0) ↔ (-1 < x) ∧ (x < 1) := 
by sorry

end NUMINAMATH_GPT_solution_of_inequality_system_l658_65802


namespace NUMINAMATH_GPT_metropolis_hospital_babies_l658_65889

theorem metropolis_hospital_babies 
    (a b d : ℕ) 
    (h1 : a = 3 * b) 
    (h2 : b = 2 * d) 
    (h3 : 2 * a + 3 * b + 5 * d = 1200) : 
    5 * d = 260 := 
sorry

end NUMINAMATH_GPT_metropolis_hospital_babies_l658_65889


namespace NUMINAMATH_GPT_rainy_days_l658_65809

namespace Mo

def drinks (R NR n : ℕ) :=
  -- Condition 3: Total number of days in the week equation
  R + NR = 7 ∧
  -- Condition 1-2: Total cups of drinks equation
  n * R + 3 * NR = 26 ∧
  -- Condition 4: Difference in cups of tea and hot chocolate equation
  3 * NR - n * R = 10

theorem rainy_days (R NR n : ℕ) (h: drinks R NR n) : 
  R = 1 := sorry

end Mo

end NUMINAMATH_GPT_rainy_days_l658_65809


namespace NUMINAMATH_GPT_daily_chicken_loss_l658_65822

/--
A small poultry farm has initially 300 chickens, 200 turkeys, and 80 guinea fowls. Every day, the farm loses some chickens, 8 turkeys, and 5 guinea fowls. After one week (7 days), there are 349 birds left in the farm. Prove the number of chickens the farmer loses daily.
-/
theorem daily_chicken_loss (initial_chickens initial_turkeys initial_guinea_fowls : ℕ)
  (daily_turkey_loss daily_guinea_fowl_loss days total_birds_left : ℕ)
  (h1 : initial_chickens = 300)
  (h2 : initial_turkeys = 200)
  (h3 : initial_guinea_fowls = 80)
  (h4 : daily_turkey_loss = 8)
  (h5 : daily_guinea_fowl_loss = 5)
  (h6 : days = 7)
  (h7 : total_birds_left = 349)
  (h8 : initial_chickens + initial_turkeys + initial_guinea_fowls
       - (daily_turkey_loss * days + daily_guinea_fowl_loss * days + (initial_chickens - total_birds_left)) = total_birds_left) :
  initial_chickens - (total_birds_left + daily_turkey_loss * days + daily_guinea_fowl_loss * days) / days = 20 :=
by {
    -- Proof goes here
    sorry
}

end NUMINAMATH_GPT_daily_chicken_loss_l658_65822


namespace NUMINAMATH_GPT_find_number_of_girls_l658_65849

-- Definitions for the number of candidates
variables (B G : ℕ)
variable (total_candidates : B + G = 2000)

-- Definitions for the percentages of passed candidates
variable (pass_rate_boys : ℝ := 0.34)
variable (pass_rate_girls : ℝ := 0.32)
variable (pass_rate_total : ℝ := 0.331)

-- Hypotheses based on the conditions
variables (P_B P_G : ℝ)
variable (pass_boys : P_B = pass_rate_boys * B)
variable (pass_girls : P_G = pass_rate_girls * G)
variable (pass_total_eq : P_B + P_G = pass_rate_total * 2000)

-- Goal: Prove that the number of girls (G) is 1800
theorem find_number_of_girls (B G : ℕ)
  (total_candidates : B + G = 2000)
  (pass_rate_boys : ℝ := 0.34)
  (pass_rate_girls : ℝ := 0.32)
  (pass_rate_total : ℝ := 0.331)
  (P_B P_G : ℝ)
  (pass_boys : P_B = pass_rate_boys * (B : ℝ))
  (pass_girls : P_G = pass_rate_girls * (G : ℝ))
  (pass_total_eq : P_B + P_G = pass_rate_total * 2000) : G = 1800 :=
sorry

end NUMINAMATH_GPT_find_number_of_girls_l658_65849


namespace NUMINAMATH_GPT_total_revenue_calculation_l658_65878

-- Define the total number of etchings sold
def total_etchings : ℕ := 16

-- Define the number of etchings sold at $35 each
def etchings_sold_35 : ℕ := 9

-- Define the price per etching sold at $35
def price_per_etching_35 : ℕ := 35

-- Define the price per etching sold at $45
def price_per_etching_45 : ℕ := 45

-- Define the total revenue calculation
def total_revenue : ℕ :=
  let revenue_35 := etchings_sold_35 * price_per_etching_35
  let etchings_sold_45 := total_etchings - etchings_sold_35
  let revenue_45 := etchings_sold_45 * price_per_etching_45
  revenue_35 + revenue_45

-- Theorem stating the total revenue is $630
theorem total_revenue_calculation : total_revenue = 630 := by
  sorry

end NUMINAMATH_GPT_total_revenue_calculation_l658_65878


namespace NUMINAMATH_GPT_totalWeightAlF3_is_correct_l658_65854

-- Define the atomic weights of Aluminum and Fluorine
def atomicWeightAl : ℝ := 26.98
def atomicWeightF : ℝ := 19.00

-- Define the number of atoms of Fluorine in Aluminum Fluoride (AlF3)
def numFluorineAtoms : ℕ := 3

-- Define the number of moles of Aluminum Fluoride
def numMolesAlF3 : ℕ := 7

-- Calculate the molecular weight of Aluminum Fluoride (AlF3)
noncomputable def molecularWeightAlF3 : ℝ :=
  atomicWeightAl + (numFluorineAtoms * atomicWeightF)

-- Calculate the total weight of the given moles of AlF3
noncomputable def totalWeight : ℝ :=
  molecularWeightAlF3 * numMolesAlF3

-- Theorem stating the total weight of 7 moles of AlF3
theorem totalWeightAlF3_is_correct : totalWeight = 587.86 := sorry

end NUMINAMATH_GPT_totalWeightAlF3_is_correct_l658_65854


namespace NUMINAMATH_GPT_unique_solution_l658_65868

theorem unique_solution (x y z : ℕ) (h_x : x > 1) (h_y : y > 1) (h_z : z > 1) :
  (x + 1)^y - x^z = 1 → x = 2 ∧ y = 2 ∧ z = 3 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_l658_65868


namespace NUMINAMATH_GPT_abs_diff_l658_65861

theorem abs_diff (a b : ℝ) (h_ab : a < b) (h_a : abs a = 6) (h_b : abs b = 3) :
  a - b = -9 ∨ a - b = 9 :=
by
  sorry

end NUMINAMATH_GPT_abs_diff_l658_65861


namespace NUMINAMATH_GPT_difference_of_two_smallest_integers_l658_65813

/--
The difference between the two smallest integers greater than 1 which, when divided by any integer 
\( k \) in the range from \( 3 \leq k \leq 13 \), leave a remainder of \( 2 \), is \( 360360 \).
-/
theorem difference_of_two_smallest_integers (n m : ℕ) (h_n : ∀ k : ℕ, 3 ≤ k ∧ k ≤ 13 → n % k = 2) (h_m : ∀ k : ℕ, 3 ≤ k ∧ k ≤ 13 → m % k = 2) (h_smallest : m > n) :
  m - n = 360360 :=
sorry

end NUMINAMATH_GPT_difference_of_two_smallest_integers_l658_65813


namespace NUMINAMATH_GPT_no_six_consecutive_nat_num_sum_eq_2015_l658_65884

theorem no_six_consecutive_nat_num_sum_eq_2015 :
  ∀ (a b c d e f : ℕ),
  a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e ∧ e + 1 = f →
  a * b * c + d * e * f ≠ 2015 :=
by
  intros a b c d e f h
  sorry

end NUMINAMATH_GPT_no_six_consecutive_nat_num_sum_eq_2015_l658_65884


namespace NUMINAMATH_GPT_rectangle_ratio_l658_65870

theorem rectangle_ratio 
  (s : ℝ) -- side length of the inner square
  (x y : ℝ) -- longer side and shorter side of the rectangle
  (h_inner_area : s^2 = (inner_square_area : ℝ))
  (h_outer_area : 9 * inner_square_area = outer_square_area)
  (h_outer_side_eq : (s + 2 * y)^2 = outer_square_area)
  (h_longer_side_eq : x + y = 3 * s) :
  x / y = 2 :=
by sorry

end NUMINAMATH_GPT_rectangle_ratio_l658_65870


namespace NUMINAMATH_GPT_fx_fixed_point_l658_65831

theorem fx_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ x y, (x = -1) ∧ (y = 3) ∧ (a * (x + 1) + 2 = y) :=
by
  sorry

end NUMINAMATH_GPT_fx_fixed_point_l658_65831


namespace NUMINAMATH_GPT_minimal_length_AX_XB_l658_65817

theorem minimal_length_AX_XB 
  (AA' BB' : ℕ) (A'B' : ℕ) 
  (h1 : AA' = 680) (h2 : BB' = 2000) (h3 : A'B' = 2010) 
  : ∃ X : ℕ, AX + XB = 3350 := 
sorry

end NUMINAMATH_GPT_minimal_length_AX_XB_l658_65817


namespace NUMINAMATH_GPT_jason_initial_cards_l658_65836

theorem jason_initial_cards (cards_given_away cards_left : ℕ) (h1 : cards_given_away = 9) (h2 : cards_left = 4) :
  cards_given_away + cards_left = 13 :=
sorry

end NUMINAMATH_GPT_jason_initial_cards_l658_65836


namespace NUMINAMATH_GPT_conference_session_time_l658_65839

def conference_duration_hours : ℕ := 8
def conference_duration_minutes : ℕ := 45
def break_time : ℕ := 30

theorem conference_session_time :
  (conference_duration_hours * 60 + conference_duration_minutes) - break_time = 495 :=
by sorry

end NUMINAMATH_GPT_conference_session_time_l658_65839


namespace NUMINAMATH_GPT_min_blocks_to_remove_l658_65856

theorem min_blocks_to_remove (n : ℕ) (h : n = 59) : 
  ∃ (k : ℕ), k = 32 ∧ (∃ m, n = m^3 + k ∧ m^3 ≤ n) :=
by {
  sorry
}

end NUMINAMATH_GPT_min_blocks_to_remove_l658_65856


namespace NUMINAMATH_GPT_min_expression_value_l658_65894

open Real

-- Define the conditions given in the problem: x, y, z are positive reals and their product is 32
variables {x y z : ℝ} (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) (h₄ : x * y * z = 32)

-- Define the expression that we want to find the minimum for: x^2 + 4xy + 4y^2 + 2z^2
def expression (x y z : ℝ) : ℝ := x^2 + 4 * x * y + 4 * y^2 + 2 * z^2

-- State the theorem: proving that the minimum value of the expression given the conditions is 96
theorem min_expression_value : ∃ (x y z : ℝ), 0 < x ∧ 0 < y ∧ 0 < z ∧ x * y * z = 32 ∧ expression x y z = 96 :=
sorry

end NUMINAMATH_GPT_min_expression_value_l658_65894


namespace NUMINAMATH_GPT_little_john_remaining_money_l658_65803

noncomputable def initial_amount: ℝ := 8.50
noncomputable def spent_on_sweets: ℝ := 1.25
noncomputable def given_to_each_friend: ℝ := 1.20
noncomputable def number_of_friends: ℝ := 2

theorem little_john_remaining_money : 
  initial_amount - (spent_on_sweets + given_to_each_friend * number_of_friends) = 4.85 :=
by
  sorry

end NUMINAMATH_GPT_little_john_remaining_money_l658_65803


namespace NUMINAMATH_GPT_M_subset_N_iff_l658_65821

section
variables {a x : ℝ}

-- Definitions based on conditions in the problem
def M (a : ℝ) : Set ℝ := { x | x^2 - a * x - x < 0 }
def N : Set ℝ := { x | x^2 - 2 * x - 3 ≤ 0 }

theorem M_subset_N_iff (a : ℝ) : M a ⊆ N ↔ -2 ≤ a ∧ a ≤ 2 :=
by
  sorry
end

end NUMINAMATH_GPT_M_subset_N_iff_l658_65821


namespace NUMINAMATH_GPT_divisible_by_4_l658_65800

theorem divisible_by_4 (n m : ℕ) (h1 : n > 0) (h2 : m > 0) (h3 : n^3 + (n + 1)^3 + (n + 2)^3 = m^3) : 4 ∣ n + 1 :=
sorry

end NUMINAMATH_GPT_divisible_by_4_l658_65800


namespace NUMINAMATH_GPT_geometric_series_sum_l658_65853

theorem geometric_series_sum (a r : ℝ) (h : |r| < 1) (h_a : a = 2 / 3) (h_r : r = 2 / 3) :
  ∑' i : ℕ, (a * r^i) = 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l658_65853


namespace NUMINAMATH_GPT_segment_equality_l658_65857

variables {Point : Type} [AddGroup Point]

-- Define the points A, B, C, D, E, F
variables (A B C D E F : Point)

-- Given conditions
variables (AC CE BD DF AD CF : Point)
variable (h1 : AC = CE)
variable (h2 : BD = DF)
variable (h3 : AD = CF)

-- Theorem statement
theorem segment_equality (h1 : A - C = C - E)
                         (h2 : B - D = D - F)
                         (h3 : A - D = C - F) :
  (C - D) = (A - B) ∧ (C - D) = (E - F) :=
by
  sorry

end NUMINAMATH_GPT_segment_equality_l658_65857


namespace NUMINAMATH_GPT_find_some_number_l658_65847

theorem find_some_number :
  ∃ (x : ℝ), abs (x - 0.004) < 0.0001 ∧ 9.237333333333334 = (69.28 * x) / 0.03 := by
  sorry

end NUMINAMATH_GPT_find_some_number_l658_65847


namespace NUMINAMATH_GPT_find_total_cows_l658_65806

-- Define the conditions given in the problem
def ducks_legs (D : ℕ) : ℕ := 2 * D
def cows_legs (C : ℕ) : ℕ := 4 * C
def total_legs (D C : ℕ) : ℕ := ducks_legs D + cows_legs C
def total_heads (D C : ℕ) : ℕ := D + C

-- State the problem in Lean 4
theorem find_total_cows (D C : ℕ) (h : total_legs D C = 2 * total_heads D C + 32) : C = 16 :=
sorry

end NUMINAMATH_GPT_find_total_cows_l658_65806


namespace NUMINAMATH_GPT_number_of_workers_l658_65810

-- Definitions corresponding to problem conditions
def total_contribution := 300000
def extra_total_contribution := 325000
def extra_amount := 50

-- Main statement to prove the number of workers
theorem number_of_workers : ∃ W C : ℕ, W * C = total_contribution ∧ W * (C + extra_amount) = extra_total_contribution ∧ W = 500 := by
  sorry

end NUMINAMATH_GPT_number_of_workers_l658_65810


namespace NUMINAMATH_GPT_growth_rate_l658_65834

variable (x : ℝ)

def initial_investment : ℝ := 500
def expected_investment : ℝ := 720

theorem growth_rate (x : ℝ) (h : 500 * (1 + x)^2 = 720) : x = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_growth_rate_l658_65834


namespace NUMINAMATH_GPT_total_land_l658_65841

variable (land_house : ℕ) (land_expansion : ℕ) (land_cattle : ℕ) (land_crop : ℕ)

theorem total_land (h1 : land_house = 25) 
                   (h2 : land_expansion = 15) 
                   (h3 : land_cattle = 40) 
                   (h4 : land_crop = 70) : 
  land_house + land_expansion + land_cattle + land_crop = 150 := 
by 
  sorry

end NUMINAMATH_GPT_total_land_l658_65841


namespace NUMINAMATH_GPT_circle_radii_order_l658_65808

theorem circle_radii_order (r_A r_B r_C : ℝ) 
  (h1 : r_A = Real.sqrt 10) 
  (h2 : 2 * Real.pi * r_B = 10 * Real.pi)
  (h3 : Real.pi * r_C^2 = 16 * Real.pi) : 
  r_C < r_A ∧ r_A < r_B := 
  sorry

end NUMINAMATH_GPT_circle_radii_order_l658_65808


namespace NUMINAMATH_GPT_local_minimum_at_minus_one_l658_65843

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem local_minimum_at_minus_one :
  (∃ δ > 0, ∀ x : ℝ, (x < -1 + δ ∧ x > -1 - δ) → f x ≥ f (-1)) :=
by
  sorry

end NUMINAMATH_GPT_local_minimum_at_minus_one_l658_65843


namespace NUMINAMATH_GPT_sin_identity_alpha_l658_65845

theorem sin_identity_alpha (α : ℝ) (hα : α = Real.pi / 7) : 
  1 / Real.sin α = 1 / Real.sin (2 * α) + 1 / Real.sin (3 * α) := 
by 
  sorry

end NUMINAMATH_GPT_sin_identity_alpha_l658_65845


namespace NUMINAMATH_GPT_correct_adjacent_book_left_l658_65812

-- Define the parameters
variable (prices : ℕ → ℕ)
variable (n : ℕ)
variable (step : ℕ)

-- Given conditions
axiom h1 : n = 31
axiom h2 : step = 2
axiom h3 : ∀ k : ℕ, 0 ≤ k ∧ k < n - 1 → prices (k + 1) = prices k + step
axiom h4 : prices 30 = prices 15 + prices 14

-- We need to show that the adjacent book referred to is at the left of the middle book.
theorem correct_adjacent_book_left (h : n = 31) (prices_step : ∀ k : ℕ, 0 ≤ k ∧ k < n - 1 → prices (k + 1) = prices k + step) : prices 30 = prices 15 + prices 14 := by
  sorry

end NUMINAMATH_GPT_correct_adjacent_book_left_l658_65812


namespace NUMINAMATH_GPT_length_of_bridge_l658_65886

theorem length_of_bridge (train_length : ℕ) (train_speed : ℕ) (cross_time : ℕ) 
  (h1 : train_length = 150) 
  (h2 : train_speed = 45) 
  (h3 : cross_time = 30) : 
  ∃ bridge_length : ℕ, bridge_length = 225 := sorry

end NUMINAMATH_GPT_length_of_bridge_l658_65886


namespace NUMINAMATH_GPT_add_decimals_l658_65850

theorem add_decimals :
  5.623 + 4.76 = 10.383 :=
by sorry

end NUMINAMATH_GPT_add_decimals_l658_65850


namespace NUMINAMATH_GPT_complex_addition_l658_65896

def c : ℂ := 3 - 2 * Complex.I
def d : ℂ := 1 + 3 * Complex.I

theorem complex_addition : 3 * c + 4 * d = 13 + 6 * Complex.I := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_complex_addition_l658_65896


namespace NUMINAMATH_GPT_common_difference_l658_65869

theorem common_difference (a : ℕ → ℤ) (d : ℤ) 
    (h1 : ∀ n : ℕ, a (n + 1) = a 1 + n * d)
    (h2 : a 1 + a 3 + a 5 = 15)
    (h3 : a 4 = 3) : 
    d = -2 := 
sorry

end NUMINAMATH_GPT_common_difference_l658_65869


namespace NUMINAMATH_GPT_amy_hours_per_week_school_year_l658_65851

variable (hours_per_week_summer : ℕ)
variable (weeks_summer : ℕ)
variable (earnings_summer : ℕ)
variable (additional_earnings_needed : ℕ)
variable (weeks_school_year : ℕ)
variable (hourly_wage : ℝ := earnings_summer / (hours_per_week_summer * weeks_summer))

theorem amy_hours_per_week_school_year :
  hours_per_week_school_year = (additional_earnings_needed / hourly_wage) / weeks_school_year :=
by 
  -- Using the hourly wage and total income needed, calculate the hours.
  let total_hours_needed := additional_earnings_needed / hourly_wage
  have h1 : hours_per_week_school_year = total_hours_needed / weeks_school_year := sorry
  exact h1

end NUMINAMATH_GPT_amy_hours_per_week_school_year_l658_65851


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l658_65824

-- Problem Definitions
def V_s : ℕ := 16
def t : ℕ := sorry -- t is arbitrary positive value
def V_b : ℕ := 48

-- Conditions
def upstream_time := 2 * t
def downstream_time := t
def upstream_distance := (V_b - V_s) * upstream_time
def downstream_distance := (V_b + V_s) * downstream_time

-- Proof Problem
theorem boat_speed_in_still_water :
  upstream_distance = downstream_distance → V_b = 48 :=
by sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l658_65824


namespace NUMINAMATH_GPT_partial_fraction_product_zero_l658_65864

theorem partial_fraction_product_zero
  (A B C : ℚ)
  (partial_fraction_eq : ∀ x : ℚ,
    x^2 - 25 = A * (x + 3) * (x - 5) + B * (x - 3) * (x - 5) + C * (x - 3) * (x + 3))
  (fact_3 : C = 0)
  (fact_neg3 : B = 1/3)
  (fact_5 : A = 0) :
  A * B * C = 0 := 
sorry

end NUMINAMATH_GPT_partial_fraction_product_zero_l658_65864


namespace NUMINAMATH_GPT_planting_trees_equation_l658_65890

theorem planting_trees_equation (x : ℝ) (h1 : x > 0) : 
  20 / x - 20 / ((1 + 0.1) * x) = 4 :=
sorry

end NUMINAMATH_GPT_planting_trees_equation_l658_65890


namespace NUMINAMATH_GPT_polynomial_expansion_l658_65883

theorem polynomial_expansion (x : ℝ) : 
  (1 + x^2) * (1 - x^3) = 1 + x^2 - x^3 - x^5 :=
by sorry

end NUMINAMATH_GPT_polynomial_expansion_l658_65883


namespace NUMINAMATH_GPT_increase_in_average_weight_l658_65895

theorem increase_in_average_weight :
  let initial_group_size := 6
  let initial_weight := 65
  let new_weight := 74
  let initial_avg_weight := A
  (new_weight - initial_weight) / initial_group_size = 1.5 := by
    sorry

end NUMINAMATH_GPT_increase_in_average_weight_l658_65895


namespace NUMINAMATH_GPT_percentage_increase_of_bill_l658_65880

theorem percentage_increase_of_bill 
  (original_bill : ℝ) 
  (increased_bill : ℝ)
  (h1 : original_bill = 60)
  (h2 : increased_bill = 78) : 
  ((increased_bill - original_bill) / original_bill * 100) = 30 := 
by 
  rw [h1, h2]
  -- The following steps show the intended logic:
  -- calc 
  --   [(78 - 60) / 60 * 100]
  --   = [(18) / 60 * 100]
  --   = [0.3 * 100]
  --   = 30
  sorry

end NUMINAMATH_GPT_percentage_increase_of_bill_l658_65880


namespace NUMINAMATH_GPT_term_value_in_sequence_l658_65859

theorem term_value_in_sequence (a : ℕ → ℕ) (n : ℕ) (h : ∀ n, a n = n * (n + 2) / 2) (h_val : a n = 220) : n = 20 :=
  sorry

end NUMINAMATH_GPT_term_value_in_sequence_l658_65859


namespace NUMINAMATH_GPT_gift_sequences_count_l658_65891

def num_students : ℕ := 11
def num_meetings : ℕ := 4
def sequences : ℕ := num_students ^ num_meetings

theorem gift_sequences_count : sequences = 14641 := by
  sorry

end NUMINAMATH_GPT_gift_sequences_count_l658_65891


namespace NUMINAMATH_GPT_quadratic_root_relation_l658_65872

theorem quadratic_root_relation (m n p q : ℝ) (s₁ s₂ : ℝ) 
  (h1 : s₁ + s₂ = -p) 
  (h2 : s₁ * s₂ = q) 
  (h3 : 3 * s₁ + 3 * s₂ = -m) 
  (h4 : 9 * s₁ * s₂ = n) 
  (h_m : m ≠ 0) 
  (h_n : n ≠ 0) 
  (h_p : p ≠ 0) 
  (h_q : q ≠ 0) :
  n = 9 * q :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_relation_l658_65872


namespace NUMINAMATH_GPT_rahul_matches_l658_65873

variable (m : ℕ)

/-- Rahul's current batting average is 51, and if he scores 78 runs in today's match,
    his new batting average will become 54. Prove that the number of matches he had played
    in this season before today's match is 8. -/
theorem rahul_matches (h1 : (51 * m) / m = 51)
                      (h2 : (51 * m + 78) / (m + 1) = 54) : m = 8 := by
  sorry

end NUMINAMATH_GPT_rahul_matches_l658_65873


namespace NUMINAMATH_GPT_f_even_l658_65816

-- Let g(x) = x^3 - x
def g (x : ℝ) : ℝ := x^3 - x

-- Let f(x) = |g(x^2)|
def f (x : ℝ) : ℝ := abs (g (x^2))

-- Prove that f(x) is even, i.e., f(-x) = f(x) for all x
theorem f_even : ∀ x : ℝ, f (-x) = f x := by
  sorry

end NUMINAMATH_GPT_f_even_l658_65816


namespace NUMINAMATH_GPT_matrix_non_invertible_at_36_31_l658_65835

-- Define the matrix A
def A (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2 * x, 9], ![4 - x, 11]]

-- State the theorem
theorem matrix_non_invertible_at_36_31 :
  ∃ x : ℝ, (A x).det = 0 ∧ x = 36 / 31 :=
by {
  sorry
}

end NUMINAMATH_GPT_matrix_non_invertible_at_36_31_l658_65835


namespace NUMINAMATH_GPT_andrei_cannot_ensure_victory_l658_65871

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

end NUMINAMATH_GPT_andrei_cannot_ensure_victory_l658_65871


namespace NUMINAMATH_GPT_mutually_exclusive_not_opposed_l658_65865

-- Define the types for cards and people
inductive Card
| red : Card
| white : Card
| black : Card

inductive Person
| A : Person
| B : Person
| C : Person

-- Define the event that a person receives a specific card
def receives (p : Person) (c : Card) : Prop := sorry

-- Conditions
axiom A_receives_red : receives Person.A Card.red → ¬ receives Person.B Card.red
axiom B_receives_red : receives Person.B Card.red → ¬ receives Person.A Card.red

-- The proof problem statement
theorem mutually_exclusive_not_opposed :
  (receives Person.A Card.red → ¬ receives Person.B Card.red) ∧
  (¬(receives Person.A Card.red ∧ receives Person.B Card.red)) ∧
  (¬∀ p : Person, receives p Card.red) :=
sorry

end NUMINAMATH_GPT_mutually_exclusive_not_opposed_l658_65865


namespace NUMINAMATH_GPT_mary_books_end_of_year_l658_65875

def total_books_end_of_year (books_start : ℕ) (book_club : ℕ) (lent_to_jane : ℕ) 
 (returned_by_alice : ℕ) (bought_5th_month : ℕ) (bought_yard_sales : ℕ) 
 (birthday_daughter : ℕ) (birthday_mother : ℕ) (received_sister : ℕ)
 (buy_one_get_one : ℕ) (donated_charity : ℕ) (borrowed_neighbor : ℕ)
 (sold_used_store : ℕ) : ℕ :=
  books_start + book_club - lent_to_jane + returned_by_alice + bought_5th_month + bought_yard_sales +
  birthday_daughter + birthday_mother + received_sister + buy_one_get_one - donated_charity - borrowed_neighbor - sold_used_store

theorem mary_books_end_of_year : total_books_end_of_year 200 (2 * 12) 10 5 15 8 1 8 6 4 30 5 7 = 219 := by
  sorry

end NUMINAMATH_GPT_mary_books_end_of_year_l658_65875


namespace NUMINAMATH_GPT_ratio_A_to_B_investment_l658_65863

variable (A B C : Type) [Field A] [Field B] [Field C]
variable (investA investB investC profit total_profit : A) 

-- Conditions
axiom A_invests_some_times_as_B : ∃ n : A, investA = n * investB
axiom B_invests_two_thirds_of_C : investB = (2/3) * investC
axiom total_profit_statement : total_profit = 3300
axiom B_share_statement : profit = 600

-- Theorem: Ratio of A's investment to B's investment is 3:1
theorem ratio_A_to_B_investment : ∃ n : A, investA = 3 * investB :=
sorry

end NUMINAMATH_GPT_ratio_A_to_B_investment_l658_65863


namespace NUMINAMATH_GPT_sarah_initial_trucks_l658_65838

theorem sarah_initial_trucks (trucks_given : ℕ) (trucks_left : ℕ) (initial_trucks : ℕ) :
  trucks_given = 13 → trucks_left = 38 → initial_trucks = trucks_left + trucks_given → initial_trucks = 51 := 
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_sarah_initial_trucks_l658_65838


namespace NUMINAMATH_GPT_raj_house_area_l658_65862

theorem raj_house_area :
  let bedroom_area := 11 * 11
  let bedrooms_total := bedroom_area * 4
  let bathroom_area := 6 * 8
  let bathrooms_total := bathroom_area * 2
  let kitchen_area := 265
  let living_area := kitchen_area
  bedrooms_total + bathrooms_total + kitchen_area + living_area = 1110 :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_raj_house_area_l658_65862


namespace NUMINAMATH_GPT_ella_dog_food_ratio_l658_65805

variable (ella_food_per_day : ℕ) (total_food_10days : ℕ) (x : ℕ)

theorem ella_dog_food_ratio
  (h1 : ella_food_per_day = 20)
  (h2 : total_food_10days = 1000) :
  (x : ℕ) = 4 :=
by
  sorry

end NUMINAMATH_GPT_ella_dog_food_ratio_l658_65805


namespace NUMINAMATH_GPT_cos_beta_value_l658_65827

variable (α β : ℝ)
variable (h₁ : 0 < α ∧ α < π)
variable (h₂ : 0 < β ∧ β < π)
variable (h₃ : Real.sin (α + β) = 5 / 13)
variable (h₄ : Real.tan (α / 2) = 1 / 2)

theorem cos_beta_value : Real.cos β = -16 / 65 := by
  sorry

end NUMINAMATH_GPT_cos_beta_value_l658_65827


namespace NUMINAMATH_GPT_initial_bacteria_count_l658_65852

theorem initial_bacteria_count :
  ∀ (n : ℕ), (n * 5^8 = 1953125) → n = 5 :=
by
  intro n
  intro h
  sorry

end NUMINAMATH_GPT_initial_bacteria_count_l658_65852


namespace NUMINAMATH_GPT_passed_boys_count_l658_65877

theorem passed_boys_count (P F : ℕ) 
  (h1 : P + F = 120) 
  (h2 : 37 * 120 = 39 * P + 15 * F) : 
  P = 110 :=
sorry

end NUMINAMATH_GPT_passed_boys_count_l658_65877
