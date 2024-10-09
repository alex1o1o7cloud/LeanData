import Mathlib

namespace inequality_holds_l1398_139865

theorem inequality_holds (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) : a^2 + b^2 ≥ 2 :=
sorry

end inequality_holds_l1398_139865


namespace valid_addends_l1398_139895

noncomputable def is_valid_addend (n : ℕ) : Prop :=
  ∃ (X Y : ℕ), (100 * 9 + 10 * X + 4) = n ∧ (30 + Y) ∈ [36, 30, 20, 10]

theorem valid_addends :
  ∀ (n : ℕ),
  is_valid_addend n ↔ (n = 964 ∨ n = 974 ∨ n = 984 ∨ n = 994) :=
by
  sorry

end valid_addends_l1398_139895


namespace intersection_M_N_l1398_139837

def M : Set ℝ := { x | x / (x - 1) ≥ 0 }
def N : Set ℝ := { y | ∃ x : ℝ, y = 3 * x^2 + 1 }

theorem intersection_M_N :
  { x | x / (x - 1) ≥ 0 } ∩ { y | ∃ x : ℝ, y = 3 * x^2 + 1 } = { x | x > 1 } :=
sorry

end intersection_M_N_l1398_139837


namespace area_difference_8_7_area_difference_9_8_l1398_139839

-- Define the side lengths of the tablets
def side_length_7 : ℕ := 7
def side_length_8 : ℕ := 8
def side_length_9 : ℕ := 9

-- Define the areas of the tablets
def area_7 := side_length_7 * side_length_7
def area_8 := side_length_8 * side_length_8
def area_9 := side_length_9 * side_length_9

-- Prove the differences in area
theorem area_difference_8_7 : area_8 - area_7 = 15 := by sorry
theorem area_difference_9_8 : area_9 - area_8 = 17 := by sorry

end area_difference_8_7_area_difference_9_8_l1398_139839


namespace possible_distances_AG_l1398_139842

theorem possible_distances_AG (A B V G : ℝ) (AB VG : ℝ) (x AG : ℝ) :
  (AB = 600) →
  (VG = 600) →
  (AG = 3 * x) →
  (AG = 900 ∨ AG = 1800) :=
by
  intros h1 h2 h3
  sorry

end possible_distances_AG_l1398_139842


namespace max_property_l1398_139858

noncomputable def f : ℚ → ℚ := sorry

axiom f_zero : f 0 = 0
axiom f_pos_of_nonzero : ∀ α : ℚ, α ≠ 0 → f α > 0
axiom f_mul : ∀ α β : ℚ, f (α * β) = f α * f β
axiom f_add : ∀ α β : ℚ, f (α + β) ≤ f α + f β
axiom f_bounded_by_1989 : ∀ m : ℤ, f m ≤ 1989

theorem max_property (α β : ℚ) (h : f α ≠ f β) : f (α + β) = max (f α) (f β) := sorry

end max_property_l1398_139858


namespace swimming_speed_in_still_water_l1398_139854

theorem swimming_speed_in_still_water 
  (speed_of_water : ℝ) (distance : ℝ) (time : ℝ) (v : ℝ) 
  (h_water_speed : speed_of_water = 2) 
  (h_time_distance : time = 4 ∧ distance = 8) :
  v = 4 :=
by
  sorry

end swimming_speed_in_still_water_l1398_139854


namespace complement_intersection_l1398_139850

open Set

variable (U : Set ℤ) (A B : Set ℤ)

theorem complement_intersection (hU : U = univ)
                               (hA : A = {3, 4})
                               (h_union : A ∪ B = {1, 2, 3, 4}) :
  (U \ A) ∩ B = {1, 2} :=
by
  sorry

end complement_intersection_l1398_139850


namespace children_tickets_sold_l1398_139898

theorem children_tickets_sold {A C : ℕ} (h1 : 6 * A + 4 * C = 104) (h2 : A + C = 21) : C = 11 :=
by
  sorry

end children_tickets_sold_l1398_139898


namespace a_eq_b_if_fraction_is_integer_l1398_139834

theorem a_eq_b_if_fraction_is_integer (a b : ℕ) (h_pos_a : 1 ≤ a) (h_pos_b : 1 ≤ b) :
  ∃ k : ℕ, (a^4 + a^3 + 1) = k * (a^2 * b^2 + a * b^2 + 1) -> a = b :=
by
  sorry

end a_eq_b_if_fraction_is_integer_l1398_139834


namespace negation_necessary_not_sufficient_l1398_139855

theorem negation_necessary_not_sufficient (p q : Prop) : 
  ((¬ p) → ¬ (p ∨ q)) := 
sorry

end negation_necessary_not_sufficient_l1398_139855


namespace polynomial_product_equals_expected_result_l1398_139863

-- Define the polynomials
def polynomial_product (x : ℝ) : ℝ := (x + 1) * (x^2 - x + 1)

-- Define the expected result of the product
def expected_result (x : ℝ) : ℝ := x^3 + 1

-- The main theorem to prove
theorem polynomial_product_equals_expected_result (x : ℝ) : polynomial_product x = expected_result x :=
by
  -- Placeholder for the proof
  sorry

end polynomial_product_equals_expected_result_l1398_139863


namespace y_A_functional_relationship_y_B_functional_relationship_cost_effective_B_cost_effective_equal_cost_effective_A_l1398_139852

-- Definitions of cost functions for travel agencies
def full_ticket_price : ℕ := 240

def y_A (x : ℕ) : ℕ := 120 * x + 240
def y_B (x : ℕ) : ℕ := 144 * x + 144

-- Prove functional relationships for y_A and y_B
theorem y_A_functional_relationship (x : ℕ) : y_A x = 120 * x + 240 :=
by sorry

theorem y_B_functional_relationship (x : ℕ) : y_B x = 144 * x + 144 :=
by sorry

-- Prove conditions for cost-effectiveness
theorem cost_effective_B (x : ℕ) : x < 4 → y_A x > y_B x :=
by sorry

theorem cost_effective_equal (x : ℕ) : x = 4 → y_A x = y_B x :=
by sorry

theorem cost_effective_A (x : ℕ) : x > 4 → y_A x < y_B x :=
by sorry

end y_A_functional_relationship_y_B_functional_relationship_cost_effective_B_cost_effective_equal_cost_effective_A_l1398_139852


namespace remainder_of_power_mod_l1398_139809

theorem remainder_of_power_mod (a n p : ℕ) (h_prime : Nat.Prime p) (h_a : a < p) :
  (3 : ℕ)^2024 % 17 = 13 :=
by
  sorry

end remainder_of_power_mod_l1398_139809


namespace conditional_probability_chinese_fail_l1398_139803

theorem conditional_probability_chinese_fail :
  let P_math := 0.16
  let P_chinese := 0.07
  let P_both := 0.04
  P_both / P_chinese = (4 / 7) := by
  let P_math := 0.16
  let P_chinese := 0.07
  let P_both := 0.04
  sorry

end conditional_probability_chinese_fail_l1398_139803


namespace intersection_of_A_and_B_l1398_139890

def I := {x : ℝ | true}
def A := {x : ℝ | x * (x - 1) ≥ 0}
def B := {x : ℝ | x > 1}
def C := {x : ℝ | x > 1}

theorem intersection_of_A_and_B : A ∩ B = C := by
  sorry

end intersection_of_A_and_B_l1398_139890


namespace percentage_of_profits_l1398_139889

variable (R P : ℝ) -- Let R be the revenues and P be the profits in the previous year
variable (H1 : (P/R) * 100 = 10) -- The condition we want to prove
variable (H2 : 0.95 * R) -- Revenues in 2009 are 0.95R
variable (H3 : 0.1 * 0.95 * R) -- Profits in 2009 are 0.1 * 0.95R = 0.095R
variable (H4 : 0.095 * R = 0.95 * P) -- The given relation between profits in 2009 and previous year

theorem percentage_of_profits (H1 : (P/R) * 100 = 10) 
  (H2 : ∀ (R : ℝ),  ∃ ρ, ρ = 0.95 * R)
  (H3 : ∀ (R : ℝ),  ∃ π, π = 0.10 * (0.95 * R))
  (H4 : ∀ (R P : ℝ), 0.095 * R = 0.95 * P) :
  ∀ (P R : ℝ), (P/R) * 100 = 10 := 
by
  sorry

end percentage_of_profits_l1398_139889


namespace sq_diff_eq_binom_identity_l1398_139828

variable (a b : ℝ)

theorem sq_diff_eq_binom_identity : (a - b) ^ 2 = a ^ 2 - 2 * a * b + b ^ 2 :=
by
  sorry

end sq_diff_eq_binom_identity_l1398_139828


namespace nalani_fraction_sold_is_3_over_8_l1398_139835

-- Definitions of conditions
def num_dogs : ℕ := 2
def puppies_per_dog : ℕ := 10
def total_amount_received : ℕ := 3000
def price_per_puppy : ℕ := 200

-- Calculation of total puppies and sold puppies
def total_puppies : ℕ := num_dogs * puppies_per_dog
def puppies_sold : ℕ := total_amount_received / price_per_puppy

-- Fraction of puppies sold
def fraction_sold : ℚ := puppies_sold / total_puppies

theorem nalani_fraction_sold_is_3_over_8 :
  fraction_sold = 3 / 8 :=
sorry

end nalani_fraction_sold_is_3_over_8_l1398_139835


namespace radius_of_circle_of_roots_l1398_139886

theorem radius_of_circle_of_roots (z : ℂ)
  (h : (z + 2)^6 = 64 * z^6) :
  ∃ r : ℝ, r = 4 / 3 ∧ ∀ z, (z + 2)^6 = 64 * z^6 →
  abs (z + 2) = (4 / 3 : ℝ) * abs z :=
by
  sorry

end radius_of_circle_of_roots_l1398_139886


namespace slices_with_both_toppings_l1398_139812

theorem slices_with_both_toppings :
  ∀ (h p b : ℕ),
  (h + b = 9) ∧ (p + b = 12) ∧ (h + p + b = 15) → b = 6 :=
by
  sorry

end slices_with_both_toppings_l1398_139812


namespace total_stickers_l1398_139849

theorem total_stickers (r s t : ℕ) (h1 : r = 30) (h2 : s = 3 * r) (h3 : t = s + 20) : r + s + t = 230 :=
by sorry

end total_stickers_l1398_139849


namespace area_of_region_enclosed_by_graph_l1398_139862

noncomputable def area_of_enclosed_region : ℝ :=
  let x1 := 41.67
  let x2 := 62.5
  let y1 := 8.33
  let y2 := -8.33
  0.5 * (x2 - x1) * (y1 - y2)

theorem area_of_region_enclosed_by_graph :
  area_of_enclosed_region = 173.28 :=
sorry

end area_of_region_enclosed_by_graph_l1398_139862


namespace contrapositive_example_l1398_139823

variable {a : ℕ → ℝ}

theorem contrapositive_example 
  (h₁ : ∀ n : ℕ, n > 0 → (a n + a (n + 2)) / 2 < a (n + 1)) :
  (∀ n : ℕ, n > 0 → a n ≤ a (n + 1)) → ∀ n : ℕ, n > 0 → (a n + a (n + 2)) / 2 ≥ a (n + 1) :=
by
  sorry

end contrapositive_example_l1398_139823


namespace find_natural_numbers_l1398_139867

-- Problem statement: Find all natural numbers x, y, z such that 3^x + 4^y = 5^z
theorem find_natural_numbers (x y z : ℕ) (h : 3^x + 4^y = 5^z) : x = 2 ∧ y = 2 ∧ z = 2 :=
sorry

end find_natural_numbers_l1398_139867


namespace min_elements_in_AS_l1398_139844

theorem min_elements_in_AS (n : ℕ) (h : n ≥ 2) (S : Finset ℝ) (h_card : S.card = n) :
  ∃ (A_S : Finset ℝ), ∀ T : Finset ℝ, (∀ a b : ℝ, a ≠ b → a ∈ S → b ∈ S → (a + b) / 2 ∈ T) → 
  T.card ≥ 2 * n - 3 :=
sorry

end min_elements_in_AS_l1398_139844


namespace sum_of_coordinates_of_intersection_l1398_139815

def h : ℝ → ℝ := -- Define h(x). This would be specific to the function provided; we abstract it here for the proof.
sorry

theorem sum_of_coordinates_of_intersection (a b : ℝ) (h_eq: h a = h (a - 5)) : a + b = 6 :=
by
  -- We need a [step from the problem conditions], hence introducing the given conditions
  have : b = h a := sorry
  have : b = h (a - 5) := sorry
  exact sorry

end sum_of_coordinates_of_intersection_l1398_139815


namespace union_of_P_and_Q_l1398_139800

def P : Set ℝ := { x | |x| ≥ 3 }
def Q : Set ℝ := { y | ∃ x, y = 2^x - 1 }

theorem union_of_P_and_Q : P ∪ Q = { y | y ≤ -3 ∨ y > -1 } := by
  sorry

end union_of_P_and_Q_l1398_139800


namespace jake_sister_weight_ratio_l1398_139841

theorem jake_sister_weight_ratio
  (jake_present_weight : ℕ)
  (total_weight : ℕ)
  (weight_lost : ℕ)
  (sister_weight : ℕ)
  (jake_weight_after_loss : ℕ)
  (ratio : ℕ) :
  jake_present_weight = 188 →
  total_weight = 278 →
  weight_lost = 8 →
  jake_weight_after_loss = jake_present_weight - weight_lost →
  sister_weight = total_weight - jake_present_weight →
  ratio = jake_weight_after_loss / sister_weight →
  ratio = 2 := by
  sorry

end jake_sister_weight_ratio_l1398_139841


namespace g_of_minus_1_eq_9_l1398_139873

-- defining f(x) and g(f(x)), and stating the objective to prove g(-1)=9
def f (x : ℝ) : ℝ := 4 * x - 9
def g (x : ℝ) : ℝ := 3 * x ^ 2 - 4 * x + 5

theorem g_of_minus_1_eq_9 : g (-1) = 9 :=
  sorry

end g_of_minus_1_eq_9_l1398_139873


namespace savings_after_expense_increase_l1398_139847

-- Define the conditions
def monthly_salary : ℝ := 6500
def initial_savings_percentage : ℝ := 0.20
def increase_expenses_percentage : ℝ := 0.20

-- Define the statement we want to prove
theorem savings_after_expense_increase :
  (monthly_salary - (monthly_salary - (initial_savings_percentage * monthly_salary) + (increase_expenses_percentage * (monthly_salary - (initial_savings_percentage * monthly_salary))))) = 260 :=
sorry

end savings_after_expense_increase_l1398_139847


namespace problem_statement_l1398_139845

noncomputable def sequence_def (a : ℝ) (S : ℕ → ℝ) (n : ℕ) : Prop :=
  (a ≠ 0) ∧
  (S 1 = a) ∧
  (S 2 = 2 / S 1) ∧
  (∀ n, n ≥ 3 → S n = 2 / S (n - 1))

theorem problem_statement (a : ℝ) (S : ℕ → ℝ) (h : sequence_def a S 2018) : 
  S 2018 = 2 / a := 
by 
  sorry

end problem_statement_l1398_139845


namespace boxes_left_l1398_139861

-- Define the initial number of boxes
def initial_boxes : ℕ := 10

-- Define the number of boxes sold
def boxes_sold : ℕ := 5

-- Define a theorem stating that the number of boxes left is 5
theorem boxes_left : initial_boxes - boxes_sold = 5 :=
by
  sorry

end boxes_left_l1398_139861


namespace inequality_system_no_solution_l1398_139859

theorem inequality_system_no_solution (k x : ℝ) (h₁ : 1 < x ∧ x ≤ 2) (h₂ : x > k) : k ≥ 2 :=
sorry

end inequality_system_no_solution_l1398_139859


namespace cheaper_module_cost_l1398_139884

theorem cheaper_module_cost (x : ℝ) :
  (21 * x + 10 = 62.50) → (x = 2.50) :=
by
  intro h
  sorry

end cheaper_module_cost_l1398_139884


namespace number_of_numbers_l1398_139887

theorem number_of_numbers (n : ℕ) (S : ℕ) 
  (h1 : (S + 26) / n = 16) 
  (h2 : (S + 46) / n = 18) : 
  n = 10 := 
by 
  -- placeholder for the proof
  sorry

end number_of_numbers_l1398_139887


namespace hexagon_diagonals_sum_correct_l1398_139882

noncomputable def hexagon_diagonals_sum : ℝ :=
  let AB := 40
  let S := 100
  let AC := 140
  let AD := 240
  let AE := 340
  AC + AD + AE

theorem hexagon_diagonals_sum_correct : hexagon_diagonals_sum = 720 :=
  by
  show hexagon_diagonals_sum = 720
  sorry

end hexagon_diagonals_sum_correct_l1398_139882


namespace largest_side_l1398_139891

-- Definitions of conditions from part (a)
def perimeter_eq (l w : ℝ) : Prop := 2 * l + 2 * w = 240
def area_eq (l w : ℝ) : Prop := l * w = 2880

-- The main proof statement
theorem largest_side (l w : ℝ) (h1 : perimeter_eq l w) (h2 : area_eq l w) : l = 72 ∨ w = 72 :=
by
  sorry

end largest_side_l1398_139891


namespace unit_cubes_fill_box_l1398_139816

theorem unit_cubes_fill_box (p : ℕ) (hp : Nat.Prime p) :
  let length := p
  let width := 2 * p
  let height := 3 * p
  length * width * height = 6 * p^3 :=
by
  -- Proof here
  sorry

end unit_cubes_fill_box_l1398_139816


namespace a5_eq_11_l1398_139820

variable (a : ℕ → ℚ) (S : ℕ → ℚ)
variable (n : ℕ) (d : ℚ) (a1 : ℚ)

-- The definitions as given in the conditions
def arithmetic_sequence (a : ℕ → ℚ) (a1 : ℚ) (d : ℚ) : Prop :=
  ∀ n, a n = a1 + (n - 1) * d

def sum_of_terms (S : ℕ → ℚ) (a1 : ℚ) (d : ℚ) : Prop :=
  ∀ n, S n = n / 2 * (2 * a1 + (n - 1) * d)

-- Given conditions
def cond1 (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  a 3 + S 3 = 22

def cond2 (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  a 4 - S 4 = -15

-- The statement to prove
theorem a5_eq_11 (a : ℕ → ℚ) (S : ℕ → ℚ) (a1 : ℚ) (d : ℚ)
  (h_arith : arithmetic_sequence a a1 d)
  (h_sum : sum_of_terms S a1 d)
  (h1 : cond1 a S)
  (h2 : cond2 a S) : a 5 = 11 := by
  sorry

end a5_eq_11_l1398_139820


namespace exists_N_with_N_and_N2_ending_same_l1398_139838

theorem exists_N_with_N_and_N2_ending_same : 
  ∃ (N : ℕ), (N > 0) ∧ (N % 100000 = (N*N) % 100000) ∧ (N / 10000 ≠ 0) := sorry

end exists_N_with_N_and_N2_ending_same_l1398_139838


namespace find_R_when_S_is_five_l1398_139811

theorem find_R_when_S_is_five (g : ℚ) :
  (∀ (S : ℚ), R = g * S^2 - 5) →
  (R = 25 ∧ S = 3) →
  R = (250 / 3) - 5 :=
by 
  sorry

end find_R_when_S_is_five_l1398_139811


namespace base7_to_base10_conversion_l1398_139802

def convert_base_7_to_10 := 243

namespace Base7toBase10

theorem base7_to_base10_conversion :
  2 * 7^2 + 4 * 7^1 + 3 * 7^0 = 129 := by
  -- The original number 243 in base 7 is expanded and evaluated to base 10.
  sorry

end Base7toBase10

end base7_to_base10_conversion_l1398_139802


namespace remainder_7_pow_150_mod_4_l1398_139804

theorem remainder_7_pow_150_mod_4 : (7 ^ 150) % 4 = 1 :=
by
  sorry

end remainder_7_pow_150_mod_4_l1398_139804


namespace probability_of_rolling_five_l1398_139826

theorem probability_of_rolling_five (total_outcomes : ℕ) (favorable_outcomes : ℕ) 
  (h1 : total_outcomes = 6) (h2 : favorable_outcomes = 1) : 
  favorable_outcomes / total_outcomes = (1 / 6 : ℚ) :=
by
  sorry

end probability_of_rolling_five_l1398_139826


namespace only_valid_set_is_b_l1398_139866

def can_form_triangle (a b c : Nat) : Prop :=
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b)

theorem only_valid_set_is_b :
  can_form_triangle 2 3 4 ∧ 
  ¬ can_form_triangle 1 2 3 ∧
  ¬ can_form_triangle 3 4 9 ∧
  ¬ can_form_triangle 2 2 4 := by
  sorry

end only_valid_set_is_b_l1398_139866


namespace Sravan_travel_time_l1398_139868

theorem Sravan_travel_time :
  let total_distance := 540
  let first_half_distance := total_distance / 2
  let second_half_distance := total_distance / 2
  let speed_first_half := 45
  let speed_second_half := 30
  let time_first_half := first_half_distance / speed_first_half
  let time_second_half := second_half_distance / speed_second_half
  let total_time := time_first_half + time_second_half
  total_time = 15 :=
by
  let total_distance := 540
  let first_half_distance := total_distance / 2
  let second_half_distance := total_distance / 2
  let speed_first_half := 45
  let speed_second_half := 30
  let time_first_half := first_half_distance / speed_first_half
  let time_second_half := second_half_distance / speed_second_half
  let total_time := time_first_half + time_second_half
  sorry

end Sravan_travel_time_l1398_139868


namespace proof_numbers_exist_l1398_139827

noncomputable def exists_numbers : Prop :=
  ∃ a b c : ℕ, a > 10^10 ∧ b > 10^10 ∧ c > 10^10 ∧
  (a * b % (a + 2012) = 0) ∧
  (a * c % (a + 2012) = 0) ∧
  (b * c % (b + 2012) = 0) ∧
  (a * b * c % (b + 2012) = 0) ∧
  (a * b * c % (c + 2012) = 0)

theorem proof_numbers_exist : exists_numbers :=
  sorry

end proof_numbers_exist_l1398_139827


namespace octagon_area_half_l1398_139825

theorem octagon_area_half (parallelogram : ℝ) (h_parallelogram : parallelogram = 1) : 
  (octagon_area : ℝ) =
  1 / 2 := 
  sorry

end octagon_area_half_l1398_139825


namespace number_of_lockers_l1398_139878

-- Problem Conditions
def locker_numbers_consecutive_from_one := ∀ (n : ℕ), n ≥ 1
def cost_per_digit := 0.02
def total_cost := 137.94

-- Theorem Statement
theorem number_of_lockers (h1 : locker_numbers_consecutive_from_one) (h2 : cost_per_digit = 0.02) (h3 : total_cost = 137.94) : ∃ n : ℕ, n = 2001 :=
sorry

end number_of_lockers_l1398_139878


namespace quadratic_has_distinct_real_roots_expression_value_l1398_139821

variable (x m : ℝ)

-- Condition: Quadratic equation
def quadratic_eq := (x^2 - 2 * (m - 1) * x - m * (m + 2) = 0)

-- Prove that the quadratic equation always has two distinct real roots
theorem quadratic_has_distinct_real_roots (m : ℝ) : 
  ∃ a b : ℝ, a ≠ b ∧ quadratic_eq a m ∧ quadratic_eq b m :=
by
  sorry

-- Given that x = -2 is a root, prove that 2018 - 3(m-1)^2 = 2015
theorem expression_value (m : ℝ) (h : quadratic_eq (-2) m) : 
  2018 - 3 * (m - 1)^2 = 2015 :=
by
  sorry

end quadratic_has_distinct_real_roots_expression_value_l1398_139821


namespace circles_intersect_l1398_139860

def circle1_eq (x y : ℝ) : Prop := x^2 + y^2 - 2 * x = 0
def circle2_eq (x y : ℝ) : Prop := x^2 + y^2 + 4 * y = 0

theorem circles_intersect :
  ∃ x y : ℝ, circle1_eq x y ∧ circle2_eq x y := by
  sorry

end circles_intersect_l1398_139860


namespace initial_velocity_is_three_l1398_139872

-- Define the displacement function s(t)
def s (t : ℝ) : ℝ := 3 * t - t ^ 2

-- Define the initial time condition
def initial_time : ℝ := 0

-- State the main theorem about the initial velocity
theorem initial_velocity_is_three : (deriv s) initial_time = 3 :=
by
  sorry

end initial_velocity_is_three_l1398_139872


namespace comic_books_left_l1398_139848

theorem comic_books_left (total : ℕ) (sold : ℕ) (left : ℕ) (h1 : total = 90) (h2 : sold = 65) :
  left = total - sold → left = 25 := by
  sorry

end comic_books_left_l1398_139848


namespace find_ending_number_l1398_139894

def ending_number (n : ℕ) : Prop :=
  18 < n ∧ n % 7 = 0 ∧ ((21 + n) / 2 : ℝ) = 38.5

theorem find_ending_number : ending_number 56 :=
by
  unfold ending_number
  sorry

end find_ending_number_l1398_139894


namespace find_a_and_b_l1398_139813

theorem find_a_and_b (a b c : ℝ) (h1 : a = 6 - b) (h2 : c^2 = a * b - 9) : a = 3 ∧ b = 3 :=
by
  sorry

end find_a_and_b_l1398_139813


namespace train_length_proof_l1398_139832

noncomputable def train_length (speed_kmph : ℕ) (time_seconds : ℕ) : ℝ :=
  (speed_kmph * 1000 / 3600) * time_seconds

theorem train_length_proof : train_length 100 18 = 500.04 :=
  sorry

end train_length_proof_l1398_139832


namespace fundraiser_successful_l1398_139819

-- Defining the conditions
def num_students_bringing_brownies := 30
def brownies_per_student := 12
def num_students_bringing_cookies := 20
def cookies_per_student := 24
def num_students_bringing_donuts := 15
def donuts_per_student := 12
def price_per_treat := 2

-- Calculating the total number of each type of treat
def total_brownies := num_students_bringing_brownies * brownies_per_student
def total_cookies := num_students_bringing_cookies * cookies_per_student
def total_donuts := num_students_bringing_donuts * donuts_per_student

-- Calculating the total number of treats
def total_treats := total_brownies + total_cookies + total_donuts

-- Calculating the total money raised
def total_money_raised := total_treats * price_per_treat

theorem fundraiser_successful : total_money_raised = 2040 := by
    -- We introduce a sorry here because we are not providing the proof steps.
    sorry

end fundraiser_successful_l1398_139819


namespace rectangle_dimensions_l1398_139853

theorem rectangle_dimensions (l w : ℝ) (h1 : l = 2 * w) (h2 : 2 * (l + w) = 3 * (l * w)) : 
  w = 1 ∧ l = 2 := by
  sorry

end rectangle_dimensions_l1398_139853


namespace sum_of_three_integers_eq_57_l1398_139851

theorem sum_of_three_integers_eq_57
  (a b c : ℕ) (h1: a * b * c = 7^3) (h2: a ≠ b) (h3: b ≠ c) (h4: a ≠ c) :
  a + b + c = 57 :=
sorry

end sum_of_three_integers_eq_57_l1398_139851


namespace correct_mean_l1398_139801

-- Definitions of conditions
def n : ℕ := 30
def mean_incorrect : ℚ := 140
def value_correct : ℕ := 145
def value_incorrect : ℕ := 135

-- The statement to be proved
theorem correct_mean : 
  let S_incorrect := mean_incorrect * n
  let Difference := value_correct - value_incorrect
  let S_correct := S_incorrect + Difference
  let mean_correct := S_correct / n
  mean_correct = 140.33 := 
by
  sorry

end correct_mean_l1398_139801


namespace catch_up_time_l1398_139857

theorem catch_up_time (x : ℕ) : 240 * x = 150 * x + 12 * 150 := by
  sorry

end catch_up_time_l1398_139857


namespace exponent_fraction_law_l1398_139864

theorem exponent_fraction_law :
  (2 ^ 2017 + 2 ^ 2013) / (2 ^ 2017 - 2 ^ 2013) = 17 / 15 :=
  sorry

end exponent_fraction_law_l1398_139864


namespace find_n_l1398_139822

theorem find_n (n : ℕ)
  (h1 : ∃ k : ℕ, k = n^3) -- the cube is cut into n^3 unit cubes
  (h2 : ∃ r : ℕ, r = 4 * n^2) -- 4 faces are painted, each with area n^2
  (h3 : 1 / 3 = r / (6 * k)) -- one-third of the total number of faces are red
  : n = 2 :=
by
  sorry

end find_n_l1398_139822


namespace find_n_tan_eq_l1398_139880

theorem find_n_tan_eq (n : ℤ) (h1 : -90 < n ∧ n < 90) (h2 : ∀ k : ℤ, 225 - 180 * k = 45) : n = 45 := by
  sorry

end find_n_tan_eq_l1398_139880


namespace num_neg_values_of_x_l1398_139830

theorem num_neg_values_of_x 
  (n : ℕ) 
  (xn_pos_int : ∃ k, n = k ∧ k > 0) 
  (sqrt_x_169_pos_int : ∀ x, ∃ m, x + 169 = m^2 ∧ m > 0) :
  ∃ count, count = 12 := 
by
  sorry

end num_neg_values_of_x_l1398_139830


namespace polynomial_evaluation_l1398_139806

def p (x : ℝ) (a b c d : ℝ) := x^4 + a * x^3 + b * x^2 + c * x + d

theorem polynomial_evaluation
  (a b c d : ℝ)
  (h1 : p 1 a b c d = 1993)
  (h2 : p 2 a b c d = 3986)
  (h3 : p 3 a b c d = 5979) :
  (1 / 4 : ℝ) * (p 11 a b c d + p (-7) a b c d) = 5233 := by
  sorry

end polynomial_evaluation_l1398_139806


namespace opposite_of_2023_l1398_139876

theorem opposite_of_2023 : -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l1398_139876


namespace range_of_a_union_B_eq_A_range_of_a_inter_B_eq_empty_l1398_139807

open Set

noncomputable def A (a : ℝ) : Set ℝ := { x : ℝ | a - 1 < x ∧ x < 2 * a + 1 }
def B : Set ℝ := { x : ℝ | 0 < x ∧ x < 1 }

theorem range_of_a_union_B_eq_A (a : ℝ) :
  (A a ∪ B) = A a ↔ (0 ≤ a ∧ a ≤ 1) := by
  sorry

theorem range_of_a_inter_B_eq_empty (a : ℝ) :
  (A a ∩ B) = ∅ ↔ (a ≤ - 1 / 2 ∨ 2 ≤ a) := by
  sorry

end range_of_a_union_B_eq_A_range_of_a_inter_B_eq_empty_l1398_139807


namespace distance_origin_to_point_l1398_139888

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_origin_to_point :
  distance (0, 0) (-15, 8) = 17 :=
by 
  sorry

end distance_origin_to_point_l1398_139888


namespace find_f_neg_8point5_l1398_139846

def f (x : ℝ) : ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom periodicity : ∀ x : ℝ, f (x + 2) = -f x
axiom initial_condition : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x

theorem find_f_neg_8point5 : f (-8.5) = -0.5 :=
by
  -- Expect this proof to follow the outlined logic
  sorry

end find_f_neg_8point5_l1398_139846


namespace vector_dot_product_example_l1398_139810

noncomputable def vector_dot_product (e1 e2 : ℝ) : ℝ :=
  let c := e1 * (-3 * e1)
  let d := (e1 * (2 * e2))
  let e := (e2 * (2 * e2))
  c + d + e

theorem vector_dot_product_example (e1 e2 : ℝ) (unit_vectors : e1^2 = 1 ∧ e2^2 = 1) :
  (e1 - e2) * (e1 - e2) = 1 ∧ (e1 * e2 = 1 / 2) → 
  vector_dot_product e1 e2 = -5 / 2 := by {
  sorry
}

end vector_dot_product_example_l1398_139810


namespace arithmetic_sequence_third_term_l1398_139883

theorem arithmetic_sequence_third_term (a d : ℤ) (h : a + (a + 4 * d) = 14) : a + 2 * d = 7 := by
  -- We assume the sum of the first and fifth term is 14 and prove that the third term is 7.
  sorry

end arithmetic_sequence_third_term_l1398_139883


namespace total_students_l1398_139877

-- Define the conditions
def chocolates_distributed (y z : ℕ) : ℕ :=
  y * y + z * z

-- Define the main theorem to be proved
theorem total_students (y z : ℕ) (h : z = y + 3) (chocolates_left: ℕ) (initial_chocolates: ℕ)
  (h_chocolates: chocolates_distributed y z = initial_chocolates - chocolates_left) : 
  y + z = 33 :=
by
  sorry

end total_students_l1398_139877


namespace find_numbers_l1398_139874

-- Define the conditions
def geometric_mean_condition (a b : ℝ) : Prop :=
  a * b = 3

def harmonic_mean_condition (a b : ℝ) : Prop :=
  2 / (1 / a + 1 / b) = 3 / 2

-- State the theorem to be proven
theorem find_numbers (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  geometric_mean_condition a b ∧ harmonic_mean_condition a b → (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 1) := 
by 
  sorry

end find_numbers_l1398_139874


namespace ab_bc_ca_plus_one_pos_l1398_139881

variable (a b c : ℝ)
variable (h₁ : |a| < 1)
variable (h₂ : |b| < 1)
variable (h₃ : |c| < 1)

theorem ab_bc_ca_plus_one_pos :
  ab + bc + ca + 1 > 0 := sorry

end ab_bc_ca_plus_one_pos_l1398_139881


namespace units_digit_of_product_l1398_139871

theorem units_digit_of_product (a b c : ℕ) (n m p : ℕ) (units_a : a ≡ 4 [MOD 10])
  (units_b : b ≡ 9 [MOD 10]) (units_c : c ≡ 16 [MOD 10])
  (exp_a : n = 150) (exp_b : m = 151) (exp_c : p = 152) :
  (a^n * b^m * c^p) % 10 = 4 :=
by
  sorry

end units_digit_of_product_l1398_139871


namespace reciprocal_of_neg_eight_l1398_139875

theorem reciprocal_of_neg_eight : (1 / (-8 : ℝ)) = -1 / 8 := sorry

end reciprocal_of_neg_eight_l1398_139875


namespace children_gift_distribution_l1398_139805

theorem children_gift_distribution (N : ℕ) (hN : N > 1) :
  (∀ n : ℕ, n < N → (∃ k : ℕ, k < N ∧ k ≠ n)) →
  (∃ m : ℕ, (N - 1) = 2 * m) :=
by
  sorry

end children_gift_distribution_l1398_139805


namespace amy_total_tickets_l1398_139899

theorem amy_total_tickets (initial_tickets additional_tickets : ℕ) (h_initial : initial_tickets = 33) (h_additional : additional_tickets = 21) : 
  initial_tickets + additional_tickets = 54 := 
by 
  sorry

end amy_total_tickets_l1398_139899


namespace calc_expression_correct_l1398_139831

noncomputable def calc_expression : Real :=
  Real.sqrt 8 - (1 / 3)⁻¹ / Real.sqrt 3 + (1 - Real.sqrt 2)^2

theorem calc_expression_correct :
  calc_expression = 3 - Real.sqrt 3 :=
sorry

end calc_expression_correct_l1398_139831


namespace number_of_correct_propositions_is_one_l1398_139879

def obtuse_angle_is_second_quadrant (θ : ℝ) : Prop :=
  θ > 90 ∧ θ < 180

def acute_angle (θ : ℝ) : Prop :=
  θ < 90

def first_quadrant_not_negative (θ : ℝ) : Prop :=
  θ > 0 ∧ θ < 90

def second_quadrant_greater_first (θ₁ θ₂ : ℝ) : Prop :=
  (θ₁ > 90 ∧ θ₁ < 180) → (θ₂ > 0 ∧ θ₂ < 90) → θ₁ > θ₂

theorem number_of_correct_propositions_is_one :
  (¬ ∀ θ, obtuse_angle_is_second_quadrant θ) ∧
  (∀ θ, acute_angle θ → θ < 90) ∧
  (¬ ∀ θ, first_quadrant_not_negative θ) ∧
  (¬ ∀ θ₁ θ₂, second_quadrant_greater_first θ₁ θ₂) →
  1 = 1 :=
by
  sorry

end number_of_correct_propositions_is_one_l1398_139879


namespace trust_meteorologist_l1398_139843

noncomputable def problem_statement : Prop :=
  let r := 0.74
  let p := 0.5
  let senators_forecast := (1 - 1.5 * p) * p^2 * r
  let meteorologist_forecast := 1.5 * p * (1 - p)^2 * (1 - r)
  meteorologist_forecast > senators_forecast

theorem trust_meteorologist : problem_statement :=
  sorry

end trust_meteorologist_l1398_139843


namespace camille_total_birds_l1398_139897

theorem camille_total_birds :
  let cardinals := 3
  let robins := 4 * cardinals
  let blue_jays := 2 * cardinals
  let sparrows := 3 * cardinals + 1
  let pigeons := 3 * blue_jays
  let finches := robins / 2
  cardinals + robins + blue_jays + sparrows + pigeons + finches = 55 :=
by
  let cardinals := 3
  let robins := 4 * cardinals
  let blue_jays := 2 * cardinals
  let sparrows := 3 * cardinals + 1
  let pigeons := 3 * blue_jays
  let finches := robins / 2
  show cardinals + robins + blue_jays + sparrows + pigeons + finches = 55
  sorry

end camille_total_birds_l1398_139897


namespace find_triples_l1398_139896

theorem find_triples (x y z : ℝ) :
  (x + 1)^2 = x + y + 2 ∧
  (y + 1)^2 = y + z + 2 ∧
  (z + 1)^2 = z + x + 2 ↔ (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -1 ∧ y = -1 ∧ z = -1) :=
by
  sorry

end find_triples_l1398_139896


namespace keesha_total_cost_is_correct_l1398_139836

noncomputable def hair_cost : ℝ := 
  let cost := 50.0 
  let discount := cost * 0.10 
  let discounted_cost := cost - discount 
  let tip := discounted_cost * 0.20 
  discounted_cost + tip

noncomputable def nails_cost : ℝ := 
  let manicure_cost := 30.0 
  let pedicure_cost := 35.0 * 0.50 
  let total_without_tip := manicure_cost + pedicure_cost 
  let tip := total_without_tip * 0.20 
  total_without_tip + tip

noncomputable def makeup_cost : ℝ := 
  let cost := 40.0 
  let tax := cost * 0.07 
  let total_without_tip := cost + tax 
  let tip := total_without_tip * 0.20 
  total_without_tip + tip

noncomputable def facial_cost : ℝ := 
  let cost := 60.0 
  let discount := cost * 0.15 
  let discounted_cost := cost - discount 
  let tip := discounted_cost * 0.20 
  discounted_cost + tip

noncomputable def total_cost : ℝ := 
  hair_cost + nails_cost + makeup_cost + facial_cost

theorem keesha_total_cost_is_correct : total_cost = 223.56 := by
  sorry

end keesha_total_cost_is_correct_l1398_139836


namespace area_of_triangle_OPF_l1398_139892

theorem area_of_triangle_OPF (O : ℝ × ℝ) (F : ℝ × ℝ) (P : ℝ × ℝ)
  (hO : O = (0, 0)) (hF : F = (1, 0)) (hP_on_parabola : P.2 ^ 2 = 4 * P.1)
  (hPF : dist P F = 3) : Real.sqrt 2 = 1 / 2 * abs (F.1 - O.1) * (2 * Real.sqrt 2) := 
sorry

end area_of_triangle_OPF_l1398_139892


namespace probability_distance_greater_than_2_l1398_139824

theorem probability_distance_greater_than_2 :
  let D := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}
  let area_square := 9
  let area_sector := Real.pi
  let area_shaded := area_square - area_sector
  let P := area_shaded / area_square
  P = (9 - Real.pi) / 9 :=
by
  sorry

end probability_distance_greater_than_2_l1398_139824


namespace find_age_of_mother_l1398_139885

def Grace_age := 60
def ratio_GM_Grace := 3 / 8
def ratio_GM_Mother := 2

theorem find_age_of_mother (G M GM : ℕ) (h1 : G = ratio_GM_Grace * GM) 
                           (h2 : GM = ratio_GM_Mother * M) (h3 : G = Grace_age) : 
  M = 80 :=
by
  sorry

end find_age_of_mother_l1398_139885


namespace max_ants_collisions_l1398_139808

theorem max_ants_collisions (n : ℕ) (hpos : 0 < n) :
  ∃ (ants : Fin n → ℝ) (speeds: Fin n → ℝ) (finite_collisions : Prop)
    (collisions_bound : ℕ),
  (∀ i : Fin n, speeds i ≠ 0) →
  finite_collisions →
  collisions_bound = (n * (n - 1)) / 2 :=
by
  sorry

end max_ants_collisions_l1398_139808


namespace expression_evaluation_l1398_139817

theorem expression_evaluation :
  1 - (2 - (3 - 4 - (5 - 6))) = -1 :=
sorry

end expression_evaluation_l1398_139817


namespace perimeter_of_fence_l1398_139814

noncomputable def n : ℕ := 18
noncomputable def w : ℝ := 0.5
noncomputable def d : ℝ := 4

theorem perimeter_of_fence : 3 * ((n / 3 - 1) * d + n / 3 * w) = 69 := by
  sorry

end perimeter_of_fence_l1398_139814


namespace vector_combination_l1398_139840

-- Define the vectors and the conditions
def vec_a : ℝ × ℝ := (1, -2)
def vec_b (m : ℝ) : ℝ × ℝ := (2, m)
def parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ v = (k * u.1, k * u.2)

-- The main theorem to be proved
theorem vector_combination (m : ℝ) (h_parallel : parallel vec_a (vec_b m)) : 3 * vec_a + 2 * vec_b m = (7, -14) := by
  sorry

end vector_combination_l1398_139840


namespace city_mpg_l1398_139870

-- Definitions
def total_distance := 256.2 -- total distance in miles
def total_gallons := 21.0 -- total gallons of gasoline

-- Theorem statement
theorem city_mpg : total_distance / total_gallons = 12.2 :=
by sorry

end city_mpg_l1398_139870


namespace max_ab_value_l1398_139818

noncomputable def max_ab (a b : ℝ) : ℝ :=
  if (a > 0 ∧ b > 0 ∧ 2 * a + b = 1) then a * b else 0

theorem max_ab_value (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : 2 * a + b = 1) :
  max_ab a b = 1 / 8 := sorry

end max_ab_value_l1398_139818


namespace find_fx_l1398_139829

theorem find_fx (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x^2 + 1) = 2 * x^2 + 1) : ∀ x : ℝ, f x = 2 * x - 1 := 
sorry

end find_fx_l1398_139829


namespace copper_percentage_l1398_139833

theorem copper_percentage (copperFirst copperSecond totalWeight1 totalWeight2: ℝ) 
    (h1 : copperFirst = 0.25)
    (h2 : copperSecond = 0.50) 
    (h3 : totalWeight1 = 200) 
    (h4 : totalWeight2 = 800) : 
    (copperFirst * totalWeight1 + copperSecond * totalWeight2) / (totalWeight1 + totalWeight2) * 100 = 45 := 
by 
  sorry

end copper_percentage_l1398_139833


namespace probability_top_card_is_king_or_queen_l1398_139856

-- Defining the basic entities of the problem
def standard_deck_size := 52
def ranks := 13
def suits := 4
def number_of_kings := 4
def number_of_queens := 4
def number_of_kings_and_queens := number_of_kings + number_of_queens

-- Statement: Calculating the probability that the top card is either a King or a Queen
theorem probability_top_card_is_king_or_queen :
  (number_of_kings_and_queens : ℚ) / standard_deck_size = 2 / 13 := by
  -- Skipping the proof for now
  sorry

end probability_top_card_is_king_or_queen_l1398_139856


namespace sum_of_three_numbers_l1398_139893

theorem sum_of_three_numbers (S F T : ℕ) (h1 : S = 150) (h2 : F = 2 * S) (h3 : T = F / 3) :
  F + S + T = 550 :=
by
  sorry

end sum_of_three_numbers_l1398_139893


namespace tangent_line_parabola_l1398_139869

theorem tangent_line_parabola (k : ℝ) 
  (h : ∀ (x y : ℝ), 4 * x + 6 * y + k = 0 → y^2 = 32 * x) : k = 72 := 
sorry

end tangent_line_parabola_l1398_139869
