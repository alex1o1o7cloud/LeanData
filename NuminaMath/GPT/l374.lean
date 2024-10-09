import Mathlib

namespace honey_production_l374_37443

-- Define the conditions:
def bees : ℕ := 60
def days : ℕ := 60
def honey_per_bee : ℕ := 1

-- Statement to prove:
theorem honey_production (bees_eq : 60 = bees) (days_eq : 60 = days) (honey_per_bee_eq : 1 = honey_per_bee) :
  bees * honey_per_bee = 60 := by
  sorry

end honey_production_l374_37443


namespace complex_solutions_x2_eq_neg4_l374_37468

-- Lean statement for the proof problem
theorem complex_solutions_x2_eq_neg4 (x : ℂ) (hx : x^2 = -4) : x = 2 * Complex.I ∨ x = -2 * Complex.I :=
by 
  sorry

end complex_solutions_x2_eq_neg4_l374_37468


namespace jacks_speed_l374_37457

-- Define the initial distance between Jack and Christina.
def initial_distance : ℝ := 360

-- Define Christina's speed.
def christina_speed : ℝ := 7

-- Define Lindy's speed.
def lindy_speed : ℝ := 12

-- Define the total distance Lindy travels.
def lindy_total_distance : ℝ := 360

-- Prove Jack's speed given the conditions.
theorem jacks_speed : ∃ v : ℝ, (initial_distance - christina_speed * (lindy_total_distance / lindy_speed)) / (lindy_total_distance / lindy_speed) = v ∧ v = 5 :=
by {
  sorry
}

end jacks_speed_l374_37457


namespace evaluate_expression_l374_37448

theorem evaluate_expression : 
  70 + (5 * 12) / (180 / 3) = 71 :=
  by
  sorry

end evaluate_expression_l374_37448


namespace shopkeeper_intended_profit_l374_37482

noncomputable def intended_profit_percentage (C L S : ℝ) : ℝ :=
  (L / C) - 1

theorem shopkeeper_intended_profit (C L S : ℝ) (h1 : L = C * (1 + intended_profit_percentage C L S))
  (h2 : S = 0.90 * L) (h3 : S = 1.35 * C) : intended_profit_percentage C L S = 0.5 :=
by
  -- We indicate that the proof is skipped
  sorry

end shopkeeper_intended_profit_l374_37482


namespace unaccounted_bottles_l374_37402

theorem unaccounted_bottles :
  let total_bottles := 254
  let football_bottles := 11 * 6
  let soccer_bottles := 53
  let lacrosse_bottles := football_bottles + 12
  let rugby_bottles := 49
  let team_bottles := football_bottles + soccer_bottles + lacrosse_bottles + rugby_bottles
  total_bottles - team_bottles = 8 :=
by
  rfl

end unaccounted_bottles_l374_37402


namespace b4_minus_a4_l374_37401

-- Given quadratic equation and specified root, prove the difference of fourth powers.
theorem b4_minus_a4 (a b : ℝ) (h_root : (a^2 - b^2)^2 = x) (h_equation : x^2 + 4 * a^2 * b^2 * x = 4) : b^4 - a^4 = 2 ∨ b^4 - a^4 = -2 :=
sorry

end b4_minus_a4_l374_37401


namespace solutions_diff_l374_37476

theorem solutions_diff (a b : ℝ) (h1: (a-5)*(a+5) = 26*a - 130) (h2: (b-5)*(b+5) = 26*b - 130) (h3 : a ≠ b) (h4: a > b) : a - b = 16 := 
by
  sorry 

end solutions_diff_l374_37476


namespace find_cost_price_l374_37422

-- Define the known data
def cost_price_80kg (C : ℝ) := 80 * C
def cost_price_20kg := 20 * 20
def selling_price_mixed := 2000
def total_cost_price_mixed (C : ℝ) := cost_price_80kg C + cost_price_20kg

-- Using the condition for 25% profit
def selling_price_of_mixed (C : ℝ) := 1.25 * total_cost_price_mixed C

-- The main theorem
theorem find_cost_price (C : ℝ) : selling_price_of_mixed C = selling_price_mixed → C = 15 :=
by
  sorry

end find_cost_price_l374_37422


namespace tournament_key_player_l374_37427

theorem tournament_key_player (n : ℕ) (plays : Fin n → Fin n → Bool) (wins : ∀ i j, plays i j → ¬plays j i) :
  ∃ X, ∀ (Y : Fin n), Y ≠ X → (plays X Y ∨ ∃ Z, plays X Z ∧ plays Z Y) :=
by
  sorry

end tournament_key_player_l374_37427


namespace rectangular_prism_sum_l374_37451

-- Definitions based on conditions
def edges := 12
def corners := 8
def faces := 6

-- Lean statement to prove question == answer given conditions.
theorem rectangular_prism_sum : edges + corners + faces = 26 := by
  sorry

end rectangular_prism_sum_l374_37451


namespace combine_material_points_l374_37473

variables {K K₁ K₂ : Type} {m m₁ m₂ : ℝ}

-- Assume some properties and operations for type K
noncomputable def add_material_points (K₁ K₂ : K × ℝ) : K × ℝ :=
(K₁.1, K₁.2 + K₂.2)

theorem combine_material_points (K₁ K₂ : K × ℝ) :
  (add_material_points K₁ K₂) = (K₁.1, K₁.2 + K₂.2) :=
sorry

end combine_material_points_l374_37473


namespace no_determinable_cost_of_2_pans_l374_37483

def pots_and_pans_problem : Prop :=
  ∀ (P Q : ℕ), 3 * P + 4 * Q = 100 → ¬∃ Q_cost : ℕ, Q_cost = 2 * Q

theorem no_determinable_cost_of_2_pans : pots_and_pans_problem :=
by
  sorry

end no_determinable_cost_of_2_pans_l374_37483


namespace find_a_l374_37400

noncomputable def polynomial (a : ℝ) : ℝ → ℝ := λ x => a * x^2 + (a - 3) * x + 1

-- This is a statement without the actual computation or proof.
theorem find_a (a : ℝ) :
  (∀ x : ℝ, polynomial a x = 0 → (∃! x, polynomial a x = 0)) ↔ a = 0 ∨ a = 1 ∨ a = 9 :=
sorry

end find_a_l374_37400


namespace total_dog_food_amount_l374_37447

def initial_dog_food : ℝ := 15
def first_purchase : ℝ := 15
def second_purchase : ℝ := 10

theorem total_dog_food_amount : initial_dog_food + first_purchase + second_purchase = 40 := 
by 
  sorry

end total_dog_food_amount_l374_37447


namespace base_number_pow_k_eq_4_pow_2k_plus_2_eq_64_l374_37489

theorem base_number_pow_k_eq_4_pow_2k_plus_2_eq_64 (x k : ℝ) (h1 : x^k = 4) (h2 : x^(2 * k + 2) = 64) : x = 2 :=
sorry

end base_number_pow_k_eq_4_pow_2k_plus_2_eq_64_l374_37489


namespace mean_equals_d_l374_37465

noncomputable def sqrt (x : ℝ) : ℝ :=
  Real.sqrt x

theorem mean_equals_d
  (a b c d e : ℝ)
  (h_a : a = sqrt 2)
  (h_b : b = sqrt 18)
  (h_c : c = sqrt 200)
  (h_d : d = sqrt 32)
  (h_e : e = sqrt 8) :
  d = (a + b + c + e) / 4 := by
  -- We insert proof steps here normally
  sorry

end mean_equals_d_l374_37465


namespace translation_m_n_l374_37431

theorem translation_m_n (m n : ℤ) (P Q : ℤ × ℤ) (hP : P = (-1, -3)) (hQ : Q = (-2, 0))
(hx : P.1 - m = Q.1) (hy : P.2 + n = Q.2) :
  m + n = 4 :=
by
  sorry

end translation_m_n_l374_37431


namespace maximize_box_volume_l374_37499

noncomputable def volume (x : ℝ) := (16 - 2 * x) * (10 - 2 * x) * x

theorem maximize_box_volume :
  (∃ x : ℝ, volume x = 144 ∧ ∀ y : ℝ, 0 < y ∧ y < 5 → volume y ≤ volume 2) := 
by
  sorry

end maximize_box_volume_l374_37499


namespace even_property_of_f_when_a_zero_non_even_odd_property_of_f_when_a_nonzero_minimum_value_of_f_l374_37437

open Real

def f (a x : ℝ) : ℝ := x^2 + |x - a| - 1

theorem even_property_of_f_when_a_zero : 
  ∀ x : ℝ, f 0 x = f 0 (-x) :=
by sorry

theorem non_even_odd_property_of_f_when_a_nonzero : 
  ∀ (a x : ℝ), a ≠ 0 → (f a x ≠ f a (-x) ∧ f a x ≠ -f a (-x)) :=
by sorry

theorem minimum_value_of_f :
  ∀ (a : ℝ), 
    (a ≤ -1/2 → ∃ x : ℝ, f a x = -a - 5/4) ∧ 
    (-1/2 < a ∧ a ≤ 1/2 → ∃ x : ℝ, f a x = a^2 - 1) ∧ 
    (a > 1/2 → ∃ x : ℝ, f a x = a - 5/4) :=
by sorry

end even_property_of_f_when_a_zero_non_even_odd_property_of_f_when_a_nonzero_minimum_value_of_f_l374_37437


namespace geometry_problem_l374_37429

/-- Given:
  DC = 5
  CB = 9
  AB = 1/3 * AD
  ED = 2/3 * AD
  Prove: FC = 10.6667 -/
theorem geometry_problem
  (DC CB AD FC : ℝ) (hDC : DC = 5) (hCB : CB = 9) (hAB : AB = 1 / 3 * AD) (hED : ED = 2 / 3 * AD)
  (AB ED: ℝ):
  FC = 10.6667 :=
by
  sorry

end geometry_problem_l374_37429


namespace price_difference_eq_l374_37403

-- Define the problem conditions
variable (P : ℝ) -- Original price
variable (H1 : P - 0.15 * P = 61.2) -- Condition 1: 15% discount results in $61.2
variable (H2 : P * (1 - 0.15) = 61.2) -- Another way to represent Condition 1 (if needed)
variable (H3 : 61.2 * 1.25 = 76.5) -- Condition 4: Price raises by 25% after the 15% discount
variable (H4 : 76.5 * 0.9 = 68.85) -- Condition 5: Additional 10% discount after raise
variable (H5 : P = 72) -- Calculated original price

-- Define the theorem to prove
theorem price_difference_eq :
  (P - 68.85 = 3.15) := 
by
  sorry

end price_difference_eq_l374_37403


namespace product_value_4_l374_37464

noncomputable def product_of_values (x y : ℝ) (h : x * y - x / y^3 - y / x^3 = 5) : ℝ :=
(x - 1) * (y - 1)

theorem product_value_4 (x y : ℝ) (h : x * y - x / y^3 - y / x^3 = 5) : ∃ v : ℝ, product_of_values x y h = v ∧ v = 4 :=
sorry

end product_value_4_l374_37464


namespace students_in_all_sections_is_six_l374_37491

-- Define the number of students in each section and the total.
variable (total_students : ℕ := 30)
variable (music_students : ℕ := 15)
variable (drama_students : ℕ := 18)
variable (dance_students : ℕ := 12)
variable (at_least_two_sections : ℕ := 14)

-- Define the number of students in all three sections.
def students_in_all_three_sections (total_students music_students drama_students dance_students at_least_two_sections : ℕ) : ℕ :=
  let a := 6 -- the result we want to prove
  a

-- The theorem proving that the number of students in all three sections is 6.
theorem students_in_all_sections_is_six :
  students_in_all_three_sections total_students music_students drama_students dance_students at_least_two_sections = 6 :=
by 
  sorry -- Proof is omitted

end students_in_all_sections_is_six_l374_37491


namespace find_sum_of_squares_l374_37497

theorem find_sum_of_squares (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (h1 : x * y + x + y = 119) (h2 : x^2 * y + x * y^2 = 1680) :
  x^2 + y^2 = 1057 := 
by
  sorry

end find_sum_of_squares_l374_37497


namespace three_b_minus_a_eq_neg_five_l374_37481

theorem three_b_minus_a_eq_neg_five (a b : ℤ) (h : |a - 2| + (b + 1)^2 = 0) : 3 * b - a = -5 :=
sorry

end three_b_minus_a_eq_neg_five_l374_37481


namespace solve_x_l374_37466

theorem solve_x : ∀ (x y : ℝ), (3 * x - y = 7) ∧ (x + 3 * y = 6) → x = 27 / 10 :=
by
  intros x y h
  sorry

end solve_x_l374_37466


namespace sum_of_first_n_terms_geom_sequence_l374_37460

theorem sum_of_first_n_terms_geom_sequence (a₁ q : ℚ) (S : ℕ → ℚ)
  (h : ∀ n, S n = a₁ * (1 - q^n) / (1 - q))
  (h_ratio : S 4 / S 2 = 3) :
  S 6 / S 4 = 7 / 3 :=
by
  sorry

end sum_of_first_n_terms_geom_sequence_l374_37460


namespace sum_of_squares_l374_37495

theorem sum_of_squares :
  (2^2 + 1^2 + 0^2 + (-1)^2 + (-2)^2 = 10) :=
by
  sorry

end sum_of_squares_l374_37495


namespace reflection_line_slope_l374_37488

theorem reflection_line_slope (m b : ℝ)
  (h_reflection : ∀ (x1 y1 x2 y2 : ℝ), 
    x1 = 2 ∧ y1 = 3 ∧ x2 = 10 ∧ y2 = 7 → 
    (x1 + x2) / 2 = (10 - 2) / 2 ∧ (y1 + y2) / 2 = (7 - 3) / 2 ∧ 
    y1 = m * x1 + b ∧ y2 = m * x2 + b) :
  m + b = 15 :=
sorry

end reflection_line_slope_l374_37488


namespace h_odd_l374_37472

variable (f g : ℝ → ℝ)

-- f is odd and g is even
axiom f_odd : ∀ x, -2 ≤ x ∧ x ≤ 2 → f (-x) = -f x
axiom g_even : ∀ x, -2 ≤ x ∧ x ≤ 2 → g (-x) = g x

-- Prove that h(x) = f(x) * g(x) is odd
theorem h_odd : ∀ x, -2 ≤ x ∧ x ≤ 2 → (f x) * (g x) = (f (-x)) * (g (-x)) := by
  sorry

end h_odd_l374_37472


namespace range_of_m_l374_37475

noncomputable def f (x : ℝ) : ℝ := -x^2
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := 2^x - m

theorem range_of_m (m : ℝ) :
  (∀ x1 ∈ Set.Icc (-1 : ℝ) 3, ∃ x2 ∈ Set.Icc (0 : ℝ) 2, f x1 ≥ g x2 m) ↔ m ≥ 10 := 
by
  sorry

end range_of_m_l374_37475


namespace binomial_coefficients_sum_l374_37404

theorem binomial_coefficients_sum :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ),
  (1 - 2 * 0)^5 = a_0 + a_1 * (1 + 0) + a_2 * (1 + 0)^2 + a_3 * (1 + 0)^3 + a_4 * (1 + 0)^4 + a_5 * (1 + 0)^5 →
  (1 - 2 * 1)^5 = (-1)^5 * a_5 →
  a_0 + a_1 + a_2 + a_3 + a_4 = 33 :=
by sorry

end binomial_coefficients_sum_l374_37404


namespace circle_area_l374_37434

-- Define the conditions of the problem
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 4*y + 9 = 0

-- State the proof problem
theorem circle_area : (∀ (x y : ℝ), circle_equation x y) → (∀ r : ℝ, r = 2 → π * r^2 = 4 * π) :=
by
  sorry

end circle_area_l374_37434


namespace moles_of_NaCl_formed_l374_37459

-- Given conditions
def sodium_bisulfite_moles : ℕ := 2
def hydrochloric_acid_moles : ℕ := 2
def balanced_reaction : Prop :=
  ∀ (NaHSO3 HCl NaCl H2O SO2 : ℕ), 
    NaHSO3 + HCl = NaCl + H2O + SO2

-- Target to prove:
theorem moles_of_NaCl_formed :
  balanced_reaction → sodium_bisulfite_moles = hydrochloric_acid_moles → 
  sodium_bisulfite_moles = 2 := 
sorry

end moles_of_NaCl_formed_l374_37459


namespace necessary_but_not_sufficient_l374_37405

variables (A B : Prop)

theorem necessary_but_not_sufficient 
  (h1 : ¬ B → ¬ A)  -- Condition: ¬ B → ¬ A is true
  (h2 : ¬ (¬ A → ¬ B))  -- Condition: ¬ A → ¬ B is false
  : (A → B) ∧ ¬ (B → A) := -- Conclusion: A → B and not (B → A)
by
  -- Proof is not required, so we place sorry
  sorry

end necessary_but_not_sufficient_l374_37405


namespace polynomial_coeff_sum_eq_neg_two_l374_37435

/-- If (1 - 2 * x) ^ 9 = a₉ * x ^ 9 + a₈ * x ^ 8 + ... + a₂ * x ^ 2 + a₁ * x + a₀, 
then a₁ + a₂ + ... + a₈ + a₉ = -2. -/
theorem polynomial_coeff_sum_eq_neg_two 
  (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℤ) 
  (h : (1 - 2 * x) ^ 9 = a₉ * x ^ 9 + a₈ * x ^ 8 + a₇ * x ^ 7 + a₆ * x ^ 6 + a₅ * x ^ 5 + a₄ * x ^ 4 + a₃ * x ^ 3 + a₂ * x ^ 2 + a₁ * x + a₀) : 
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = -2 :=
by sorry

end polynomial_coeff_sum_eq_neg_two_l374_37435


namespace cube_root_neg_frac_l374_37484

theorem cube_root_neg_frac : (-(1/3 : ℝ))^3 = - 1 / 27 := by
  sorry

end cube_root_neg_frac_l374_37484


namespace BC_total_750_l374_37479

theorem BC_total_750 (A B C : ℤ) 
  (h1 : A + B + C = 900) 
  (h2 : A + C = 400) 
  (h3 : C = 250) : 
  B + C = 750 := 
by 
  sorry

end BC_total_750_l374_37479


namespace combined_weight_of_emma_and_henry_l374_37413

variables (e f g h : ℕ)

theorem combined_weight_of_emma_and_henry 
  (h1 : e + f = 310)
  (h2 : f + g = 265)
  (h3 : g + h = 280) : e + h = 325 :=
by
  sorry

end combined_weight_of_emma_and_henry_l374_37413


namespace volume_solid_correct_l374_37474

noncomputable def volume_of_solid : ℝ := 
  let area_rhombus := 1250 -- Area of the rhombus calculated from the bounded region
  let height := 10 -- Given height of the solid
  area_rhombus * height -- Volume of the solid

theorem volume_solid_correct (height: ℝ := 10) :
  volume_of_solid = 12500 := by
  sorry

end volume_solid_correct_l374_37474


namespace speed_second_half_l374_37462

theorem speed_second_half (H : ℝ) (S1 S2 : ℝ) (T : ℝ) : T = 11 → S1 = 30 → S1 * T1 = 150 → S1 * T1 + S2 * T2 = 300 → S2 = 25 :=
by
  intro hT hS1 hD1 hTotal
  sorry

end speed_second_half_l374_37462


namespace incorrect_statement_A_l374_37498

-- conditions as stated in the table
def spring_length (x : ℕ) : ℝ :=
  if x = 0 then 20
  else if x = 1 then 20.5
  else if x = 2 then 21
  else if x = 3 then 21.5
  else if x = 4 then 22
  else if x = 5 then 22.5
  else 0 -- assuming 0 for out of range for simplicity

-- questions with answers
-- Prove that statement A is incorrect
theorem incorrect_statement_A : ¬ (spring_length 0 = 20) := by
  sorry

end incorrect_statement_A_l374_37498


namespace no_set_of_9_numbers_l374_37450

theorem no_set_of_9_numbers (numbers : Finset ℕ) (median : ℕ) (max_value : ℕ) (mean : ℕ) :
  numbers.card = 9 → 
  median = 2 →
  max_value = 13 →
  mean = 7 →
  (∀ x ∈ numbers, x ≤ max_value) →
  (∃ m ∈ numbers, x ≤ median) →
  False :=
by
  sorry

end no_set_of_9_numbers_l374_37450


namespace households_used_both_brands_l374_37408

/-- 
A marketing firm determined that, of 160 households surveyed, 80 used neither brand A nor brand B soap.
60 used only brand A soap and for every household that used both brands of soap, 3 used only brand B soap.
--/
theorem households_used_both_brands (X: ℕ) (H: 4*X + 140 = 160): X = 5 :=
by
  sorry

end households_used_both_brands_l374_37408


namespace speed_in_still_water_l374_37428

theorem speed_in_still_water (upstream downstream : ℝ) 
  (h_up : upstream = 25) 
  (h_down : downstream = 45) : 
  (upstream + downstream) / 2 = 35 := 
by 
  -- Proof will go here
  sorry

end speed_in_still_water_l374_37428


namespace additional_machines_needed_l374_37461

theorem additional_machines_needed
  (machines : ℕ)
  (days : ℕ)
  (one_fourth_less_days : ℕ)
  (machine_days_total : ℕ)
  (machines_needed : ℕ)
  (additional_machines : ℕ) 
  (h1 : machines = 15) 
  (h2 : days = 36)
  (h3 : one_fourth_less_days = 27)
  (h4 : machine_days_total = machines * days)
  (h5 : machines_needed = machine_days_total / one_fourth_less_days) :
  additional_machines = machines_needed - machines → additional_machines = 5 :=
by
  admit -- sorry

end additional_machines_needed_l374_37461


namespace a_2018_value_l374_37423

theorem a_2018_value (S a : ℕ -> ℕ) (h₁ : S 1 = a 1) (h₂ : a 1 = 1) (h₃ : ∀ n : ℕ, n > 0 -> S (n + 1) = 3 * S n) :
  a 2018 = 2 * 3 ^ 2016 :=
sorry

end a_2018_value_l374_37423


namespace equal_intercepts_lines_area_two_lines_l374_37439

-- Defining the general equation of the line l with parameter a
def line_eq (a : ℝ) (x y : ℝ) : Prop := y = -(a + 1) * x + 2 - a

-- Problem statement for equal intercepts condition
theorem equal_intercepts_lines (a : ℝ) : 
  (∃ x y : ℝ, line_eq a x y ∧ x ≠ 0 ∧ y ≠ 0 ∧ (x = y ∨ x + y = 2*a + 2)) →
  (a = 2 ∨ a = 0) → 
  (line_eq a 1 (-3) ∨ line_eq a 1 1) :=
sorry

-- Problem statement for triangle area condition
theorem area_two_lines (a : ℝ) : 
  (∃ x y : ℝ, line_eq a x y ∧ x ≠ 0 ∧ y ≠ 0 ∧ (1 / 2 * |x| * |y| = 2)) →
  (a = 8 ∨ a = 0) → 
  (line_eq a 1 (-9) ∨ line_eq a 1 1) :=
sorry

end equal_intercepts_lines_area_two_lines_l374_37439


namespace ellipse_condition_necessary_but_not_sufficient_l374_37424

-- Define the conditions and proof statement in Lean 4
theorem ellipse_condition (m : ℝ) (h₁ : 2 < m) (h₂ : m < 6) : 
  (6 - m ≠ m - 2) -> 
  (∃ x y : ℝ, (x^2) / (m - 2) + (y^2) / (6 - m)= 1) :=
by
  sorry

theorem necessary_but_not_sufficient : (2 < m ∧ m < 6) ↔ (2 < m ∧ m < 6 ∧ m ≠ 4) :=
by
  sorry

end ellipse_condition_necessary_but_not_sufficient_l374_37424


namespace sodium_bicarbonate_moles_combined_l374_37480

theorem sodium_bicarbonate_moles_combined (HCl NaCl NaHCO3 : ℝ) (reaction : HCl + NaHCO3 = NaCl) 
  (HCl_eq_one : HCl = 1) (NaCl_eq_one : NaCl = 1) : 
  NaHCO3 = 1 := 
by 
  -- Placeholder for the proof
  sorry

end sodium_bicarbonate_moles_combined_l374_37480


namespace infinite_series_sum_l374_37454

theorem infinite_series_sum :
  (∑' n : ℕ, (n + 1) / 4^(n + 1)) + (∑' n : ℕ, 1 / 2^(n + 1)) = 13 / 9 := 
sorry

end infinite_series_sum_l374_37454


namespace last_three_digits_of_11_pow_210_l374_37411

theorem last_three_digits_of_11_pow_210 : (11 ^ 210) % 1000 = 601 :=
by sorry

end last_three_digits_of_11_pow_210_l374_37411


namespace triangle_inequality_l374_37487

theorem triangle_inequality
  (a b c : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_ineq : 5 * (a^2 + b^2 + c^2) < 6 * (a * b + b * c + c * a)) :
  a + b > c ∧ b + c > a ∧ c + a > b :=
by
  sorry

end triangle_inequality_l374_37487


namespace max_value_of_largest_integer_l374_37432

theorem max_value_of_largest_integer (a1 a2 a3 a4 a5 a6 a7 : ℕ) (h1 : a1 + a2 + a3 + a4 + a5 + a6 + a7 = 560) (h2 : a7 - a1 = 20) : a7 ≤ 21 :=
sorry

end max_value_of_largest_integer_l374_37432


namespace container_capacity_l374_37492

/-- Given a container where 8 liters is 20% of its capacity, calculate the total capacity of 
    40 such containers filled with water. -/
theorem container_capacity (c : ℝ) (h : 8 = 0.20 * c) : 
    40 * c * 40 = 1600 := 
by
  sorry

end container_capacity_l374_37492


namespace no_real_solutions_l374_37444

theorem no_real_solutions (x : ℝ) (h_nonzero : x ≠ 0) (h_pos : 0 < x):
  (x^10 + 1) * (x^8 + x^6 + x^4 + x^2 + 1) ≠ 12 * x^9 :=
by
-- Proof will go here.
sorry

end no_real_solutions_l374_37444


namespace boat_speed_in_still_water_l374_37440

variable (B S : ℝ)

theorem boat_speed_in_still_water :
  (B + S = 38) ∧ (B - S = 16) → B = 27 :=
by
  sorry

end boat_speed_in_still_water_l374_37440


namespace percentage_decrease_is_17_point_14_l374_37416

-- Define the conditions given in the problem
variable (S : ℝ) -- original salary
variable (D : ℝ) -- percentage decrease

-- Given conditions
def given_conditions : Prop :=
  1.40 * S - (D / 100) * 1.40 * S = 1.16 * S

-- The required proof problem, where we assert D = 17.14
theorem percentage_decrease_is_17_point_14 (S : ℝ) (h : given_conditions S D) : D = 17.14 := 
  sorry

end percentage_decrease_is_17_point_14_l374_37416


namespace problem_statement_l374_37415

theorem problem_statement (x : ℕ) (h : x = 2016) : (x^2 - x) - (x^2 - 2 * x + 1) = 2015 := by
  sorry

end problem_statement_l374_37415


namespace total_revenue_calculation_l374_37467

variables (a b : ℕ) -- Assuming a and b are natural numbers representing the number of newspapers

-- Define the prices
def purchase_price_per_copy : ℝ := 0.4
def selling_price_per_copy : ℝ := 0.5
def return_price_per_copy : ℝ := 0.2

-- Define the revenue and cost calculations
def revenue_from_selling (b : ℕ) : ℝ := selling_price_per_copy * b
def revenue_from_returning (a b : ℕ) : ℝ := return_price_per_copy * (a - b)
def cost_of_purchasing (a : ℕ) : ℝ := purchase_price_per_copy * a

-- Define the total revenue
def total_revenue (a b : ℕ) : ℝ :=
  revenue_from_selling b + revenue_from_returning a b - cost_of_purchasing a

-- The theorem we need to prove
theorem total_revenue_calculation (a b : ℕ) :
  total_revenue a b = 0.3 * b - 0.2 * a :=
by
  sorry

end total_revenue_calculation_l374_37467


namespace Danny_caps_vs_wrappers_l374_37478

def park_caps : ℕ := 58
def park_wrappers : ℕ := 25
def beach_caps : ℕ := 34
def beach_wrappers : ℕ := 15
def forest_caps : ℕ := 21
def forest_wrappers : ℕ := 32
def before_caps : ℕ := 12
def before_wrappers : ℕ := 11

noncomputable def total_caps : ℕ := park_caps + beach_caps + forest_caps + before_caps
noncomputable def total_wrappers : ℕ := park_wrappers + beach_wrappers + forest_wrappers + before_wrappers

theorem Danny_caps_vs_wrappers : total_caps - total_wrappers = 42 := by
  sorry

end Danny_caps_vs_wrappers_l374_37478


namespace result_of_y_minus_3x_l374_37446

theorem result_of_y_minus_3x (x y : ℝ) (h1 : x + y = 8) (h2 : y - x = 7.5) : y - 3 * x = 7 :=
sorry

end result_of_y_minus_3x_l374_37446


namespace determine_a_and_theta_l374_37419

noncomputable def f (a θ : ℝ) (x : ℝ) : ℝ := 2 * a * Real.sin (2 * x + θ)

theorem determine_a_and_theta :
  (∃ a θ : ℝ, 0 < θ ∧ θ < π ∧ a ≠ 0 ∧ (∀ x ∈ Set.Icc (-2 : ℝ) (2 : ℝ), f a θ x ∈ Set.Icc (-2 : ℝ) 2) ∧ 
  (∀ (x1 x2 : ℝ), x1 ∈ Set.Icc (-5 * π / 12) (π / 12) → x2 ∈ Set.Icc (-5 * π / 12) (π / 12) → x1 < x2 → f a θ x1 > f a θ x2)) →
  (a = -1) ∧ (θ = π / 3) :=
sorry

end determine_a_and_theta_l374_37419


namespace range_of_a_l374_37414

def is_odd_function (f : ℝ → ℝ) := 
  ∀ x : ℝ, f (-x) = - f x

noncomputable def f (x : ℝ) :=
  if x ≥ 0 then x^2 + 2*x else -(x^2 + 2*(-x))

theorem range_of_a (a : ℝ) (h_odd : is_odd_function f) 
(hf_pos : ∀ x : ℝ, x ≥ 0 → f x = x^2 + 2*x) : 
  f (2 - a^2) > f a → -2 < a ∧ a < 1 :=
sorry

end range_of_a_l374_37414


namespace digit_product_equality_l374_37463

theorem digit_product_equality (x y z : ℕ) (hx : x = 3) (hy : y = 7) (hz : z = 1) :
  x * (10 * x + y) = 111 * z :=
by
  -- Using hx, hy, and hz, the proof can proceed from here
  sorry

end digit_product_equality_l374_37463


namespace area_of_border_correct_l374_37453

def height_of_photograph : ℕ := 12
def width_of_photograph : ℕ := 16
def border_width : ℕ := 3
def lining_width : ℕ := 1

def area_of_photograph : ℕ := height_of_photograph * width_of_photograph

def total_height : ℕ := height_of_photograph + 2 * (lining_width + border_width)
def total_width : ℕ := width_of_photograph + 2 * (lining_width + border_width)

def area_of_framed_area : ℕ := total_height * total_width

def area_of_border_including_lining : ℕ := area_of_framed_area - area_of_photograph

theorem area_of_border_correct : area_of_border_including_lining = 288 := by
  sorry

end area_of_border_correct_l374_37453


namespace algebra_expression_value_l374_37412

theorem algebra_expression_value (a b : ℝ) (h : (30^3) * a + 30 * b - 7 = 9) :
  (-30^3) * a + (-30) * b + 2 = -14 := 
by
  sorry

end algebra_expression_value_l374_37412


namespace Bomi_change_l374_37425

def candy_cost : ℕ := 350
def chocolate_cost : ℕ := 500
def total_paid : ℕ := 1000
def total_cost := candy_cost + chocolate_cost
def change := total_paid - total_cost

theorem Bomi_change : change = 150 :=
by
  -- Here we would normally provide the proof steps.
  sorry

end Bomi_change_l374_37425


namespace marble_count_l374_37421

-- Definitions from conditions
variable (M P : ℕ)
def condition1 : Prop := M = 26 * P
def condition2 : Prop := M = 28 * (P - 1)

-- Theorem to be proved
theorem marble_count (h1 : condition1 M P) (h2 : condition2 M P) : M = 364 := 
by
  sorry

end marble_count_l374_37421


namespace number_of_exclusive_students_l374_37426

-- Definitions from the conditions
def S_both : ℕ := 16
def S_alg : ℕ := 36
def S_geo_only : ℕ := 15

-- Theorem to prove the number of students taking algebra or geometry but not both
theorem number_of_exclusive_students : (S_alg - S_both) + S_geo_only = 35 :=
by
  sorry

end number_of_exclusive_students_l374_37426


namespace set_D_forms_triangle_l374_37458

theorem set_D_forms_triangle (a b c : ℕ) (h1 : a = 4) (h2 : b = 5) (h3 : c = 6) : a + b > c ∧ a + c > b ∧ b + c > a := by
  rw [h1, h2, h3]
  show 4 + 5 > 6 ∧ 4 + 6 > 5 ∧ 5 + 6 > 4
  sorry

end set_D_forms_triangle_l374_37458


namespace combined_cost_price_is_250_l374_37452

axiom store_selling_conditions :
  ∃ (CP_A CP_B CP_C : ℝ),
    (CP_A = (110 + 70) / 2) ∧
    (CP_B = (90 + 30) / 2) ∧
    (CP_C = (150 + 50) / 2) ∧
    (CP_A + CP_B + CP_C = 250)

theorem combined_cost_price_is_250 : ∃ (CP_A CP_B CP_C : ℝ), CP_A + CP_B + CP_C = 250 :=
by sorry

end combined_cost_price_is_250_l374_37452


namespace max_entanglements_l374_37449

theorem max_entanglements (a b : ℕ) (h1 : a < b) (h2 : a < 1000) (h3 : b < 1000) :
  ∃ n ≤ 9, ∀ k, k ≤ n → ∃ a' b' : ℕ, (b' - a' = b - a - 2^k) :=
by sorry

end max_entanglements_l374_37449


namespace find_m_n_l374_37471

theorem find_m_n (m n : ℕ) (h : 26019 * m - 649 * n = 118) : m = 2 ∧ n = 80 :=
by 
  sorry

end find_m_n_l374_37471


namespace words_count_correct_l374_37485

def number_of_words (n : ℕ) : ℕ :=
if n % 2 = 0 then
  8 * 3^(n / 2 - 1)
else
  14 * 3^((n - 1) / 2)

theorem words_count_correct (n : ℕ) :
  number_of_words n = if n % 2 = 0 then 8 * 3^(n / 2 - 1) else 14 * 3^((n - 1) / 2) :=
by
  sorry

end words_count_correct_l374_37485


namespace g_eval_1000_l374_37438

def g (n : ℕ) : ℕ := sorry
axiom g_comp (n : ℕ) : g (g n) = 2 * n
axiom g_form (n : ℕ) : g (3 * n + 1) = 3 * n + 2

theorem g_eval_1000 : g 1000 = 1008 :=
by
  sorry

end g_eval_1000_l374_37438


namespace store_loss_l374_37470

theorem store_loss (x y : ℝ) (hx : x + 0.25 * x = 135) (hy : y - 0.25 * y = 135) : 
  (135 * 2) - (x + y) = -18 := 
by
  sorry

end store_loss_l374_37470


namespace percentage_increase_of_kim_l374_37441

variables (S P K : ℝ)
variables (h1 : S = 0.80 * P) (h2 : S + P = 1.80) (h3 : K = 1.12)

theorem percentage_increase_of_kim (hK : K = 1.12) (hS : S = 0.80 * P) (hSP : S + P = 1.80) :
  ((K - S) / S * 100) = 40 :=
sorry

end percentage_increase_of_kim_l374_37441


namespace sweater_cost_l374_37442

theorem sweater_cost (S : ℚ) (M : ℚ) (C : ℚ) (h1 : S = 80) (h2 : M = 3 / 4 * 80) (h3 : C = S - M) : C = 20 := by
  sorry

end sweater_cost_l374_37442


namespace right_triangle_exists_with_area_ab_l374_37410

theorem right_triangle_exists_with_area_ab (a b c d : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (d_pos : 0 < d)
    (h1 : a * b = c * d) (h2 : a + b = c - d) :
    ∃ (x y z : ℕ), x^2 + y^2 = z^2 ∧ (x * y / 2 = a * b) := sorry

end right_triangle_exists_with_area_ab_l374_37410


namespace find_x_l374_37477

theorem find_x (x : ℝ) (h : 0.25 * x = 200 - 30) : x = 680 := 
by
  sorry

end find_x_l374_37477


namespace plane_through_intersection_l374_37407

def plane1 (x y z : ℝ) : Prop := x + y + 5 * z - 1 = 0
def plane2 (x y z : ℝ) : Prop := 2 * x + 3 * y - z + 2 = 0
def pointM (x y z : ℝ) : Prop := (x, y, z) = (3, 2, 1)

theorem plane_through_intersection (x y z : ℝ) :
  plane1 x y z ∧ plane2 x y z ∧ pointM x y z → 5 * x + 14 * y - 74 * z + 31 = 0 := by
  intro h
  sorry

end plane_through_intersection_l374_37407


namespace stone_breadth_5_l374_37445

theorem stone_breadth_5 (hall_length_m hall_breadth_m stone_length_dm num_stones b₁ b₂ : ℝ) 
  (h1 : hall_length_m = 36) 
  (h2 : hall_breadth_m = 15) 
  (h3 : stone_length_dm = 3) 
  (h4 : num_stones = 3600)
  (h5 : hall_length_m * 10 * hall_breadth_m * 10 = 54000)
  (h6 : stone_length_dm * b₁ * num_stones = hall_length_m * 10 * hall_breadth_m * 10) :
  b₂ = 5 := 
  sorry

end stone_breadth_5_l374_37445


namespace smaller_cuboid_length_l374_37455

-- Definitions based on conditions
def original_cuboid_volume : ℝ := 18 * 15 * 2
def smaller_cuboid_volume (L : ℝ) : ℝ := 4 * 3 * L
def smaller_cuboids_total_volume (L : ℝ) : ℝ := 7.5 * smaller_cuboid_volume L

-- Theorem statement
theorem smaller_cuboid_length :
  ∃ L : ℝ, smaller_cuboids_total_volume L = original_cuboid_volume ∧ L = 6 := 
by
  sorry

end smaller_cuboid_length_l374_37455


namespace problem_1_problem_2_l374_37409

open Set

variables {U : Type*} [TopologicalSpace U] (a x : ℝ)

def M : Set ℝ := { x | -2 ≤ x ∧ x ≤ 5 }
def N (a : ℝ) : Set ℝ := { x | a + 1 ≤ x ∧ x ≤ 2 * a + 1 }

noncomputable def complement_N (a : ℝ) : Set ℝ := { x | x < a + 1 ∨ 2 * a + 1 < x }

theorem problem_1 (h : a = 2) :
  M ∩ (complement_N a) = { x | -2 ≤ x ∧ x < 3 } :=
sorry

theorem problem_2 (h : M ∪ N a = M) :
  a ≤ 2 :=
sorry

end problem_1_problem_2_l374_37409


namespace probability_of_first_three_red_cards_l374_37436

def total_cards : ℕ := 104
def suits : ℕ := 4
def cards_per_suit : ℕ := 26
def red_suits : ℕ := 2
def black_suits : ℕ := 2
def total_red_cards : ℕ := 52
def total_black_cards : ℕ := 52

noncomputable def probability_first_three_red : ℚ :=
  (total_red_cards / total_cards) * ((total_red_cards - 1) / (total_cards - 1)) * ((total_red_cards - 2) / (total_cards - 2))

theorem probability_of_first_three_red_cards :
  probability_first_three_red = 425 / 3502 :=
sorry

end probability_of_first_three_red_cards_l374_37436


namespace value_of_expression_l374_37420

theorem value_of_expression : 1 + 2 / (1 + 2 / (2 * 2)) = 7 / 3 := 
by 
  -- proof to be filled in
  sorry

end value_of_expression_l374_37420


namespace slope_OA_l374_37433

-- Definitions for the given conditions
def ellipse (a b : ℝ) := {P : ℝ × ℝ | (P.1^2) / a^2 + (P.2^2) / b^2 = 1}

def C1 := ellipse 2 1  -- ∑(x^2 / 4 + y^2 = 1)
def C2 := ellipse 2 4  -- ∑(y^2 / 16 + x^2 / 4 = 1)

variable {P₁ P₂ : ℝ × ℝ}  -- Points A and B
variable (h1 : P₁ ∈ C1)
variable (h2 : P₂ ∈ C2)
variable (h_rel : P₂.1 = 2 * P₁.1 ∧ P₂.2 = 2 * P₁.2)  -- ∑(x₂ = 2x₁, y₂ = 2y₁)

-- Proof that the slope of ray OA is ±1
theorem slope_OA : ∃ (m : ℝ), (m = 1 ∨ m = -1) :=
sorry

end slope_OA_l374_37433


namespace ellipse_chord_line_eq_l374_37430

noncomputable def chord_line (x y : ℝ) : ℝ := 2 * x + 4 * y - 3

theorem ellipse_chord_line_eq :
  ∀ (x y : ℝ),
    (x ^ 2 / 2 + y ^ 2 = 1) ∧ (x + y = 1) → (chord_line x y = 0) :=
by
  intros x y h
  sorry

end ellipse_chord_line_eq_l374_37430


namespace range_of_a_l374_37486

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x ≤ 1 → x * |x - a| - 2 < 0) ↔ (-1 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l374_37486


namespace seq_nonzero_l374_37456

def seq (a : ℕ → ℤ) : Prop :=
  (a 1 = 1) ∧ (a 2 = 2) ∧ (∀ n, n ≥ 3 → 
    (if (a (n - 2) * a (n - 1)) % 2 = 0 
     then a n = 5 * a (n - 1) - 3 * a (n - 2) 
     else a n = a (n - 1) - a (n - 2)))

theorem seq_nonzero (a : ℕ → ℤ) (h : seq a) : ∀ n, n > 0 → a n ≠ 0 :=
  sorry

end seq_nonzero_l374_37456


namespace new_average_score_after_drop_l374_37493

theorem new_average_score_after_drop
  (avg_score : ℝ) (num_students : ℕ) (drop_score : ℝ) (remaining_students : ℕ) :
  avg_score = 62.5 →
  num_students = 16 →
  drop_score = 70 →
  remaining_students = 15 →
  (num_students * avg_score - drop_score) / remaining_students = 62 :=
by
  intros h_avg h_num h_drop h_remain
  rw [h_avg, h_num, h_drop, h_remain]
  norm_num

end new_average_score_after_drop_l374_37493


namespace total_cost_of_books_l374_37417

-- Conditions from the problem
def C1 : ℝ := 350
def loss_percent : ℝ := 0.15
def gain_percent : ℝ := 0.19
def SP1 : ℝ := C1 - (loss_percent * C1) -- Selling price of the book sold at a loss
def SP2 : ℝ := SP1 -- Selling price of the book sold at a gain

-- Statement to prove the total cost
theorem total_cost_of_books : C1 + (SP2 / (1 + gain_percent)) = 600 := by
  sorry

end total_cost_of_books_l374_37417


namespace exact_time_is_3_07_27_l374_37490

theorem exact_time_is_3_07_27 (t : ℝ) (H1 : t > 0) (H2 : t < 60) 
(H3 : 6 * (t + 8) = 89 + 0.5 * t) : t = 7 + 27/60 :=
by
  sorry

end exact_time_is_3_07_27_l374_37490


namespace find_p_l374_37406

theorem find_p (p : ℝ) : 
  (Nat.choose 5 3) * p^3 = 80 → p = 2 :=
by
  intro h
  sorry

end find_p_l374_37406


namespace min_value_of_reciprocal_sum_l374_37469

theorem min_value_of_reciprocal_sum (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a + 2 * b = 1) (h2 : c + 2 * d = 1) :
  16 ≤ (1 / a) + 1 / (b * c * d) :=
by
  sorry

end min_value_of_reciprocal_sum_l374_37469


namespace geometric_progression_fourth_term_l374_37496

theorem geometric_progression_fourth_term (a b c : ℝ) (r : ℝ) 
  (h1 : a = 2) (h2 : b = 2 * Real.sqrt 2) (h3 : c = 4) (h4 : r = Real.sqrt 2)
  (h5 : b = a * r) (h6 : c = b * r) :
  c * r = 4 * Real.sqrt 2 := 
sorry

end geometric_progression_fourth_term_l374_37496


namespace largest_prime_factor_8250_l374_37494

-- Define a function to check if a number is prime (using an existing library function)
def is_prime (n: ℕ) : Prop := Nat.Prime n

-- Define the given problem statement as a Lean theorem
theorem largest_prime_factor_8250 :
  ∃ p, is_prime p ∧ p ∣ 8250 ∧ 
    ∀ q, is_prime q ∧ q ∣ 8250 → q ≤ p :=
sorry -- The proof will be filled in later

end largest_prime_factor_8250_l374_37494


namespace subdivide_tetrahedron_l374_37418

/-- A regular tetrahedron with edge length 1 can be divided into smaller regular tetrahedrons and octahedrons,
    such that the edge lengths of the resulting tetrahedrons and octahedrons are less than 1 / 100 after a 
    finite number of subdivisions. -/
theorem subdivide_tetrahedron (edge_len : ℝ) (h : edge_len = 1) :
  ∃ (k : ℕ), (1 / (2^k : ℝ) < 1 / 100) :=
by sorry

end subdivide_tetrahedron_l374_37418
