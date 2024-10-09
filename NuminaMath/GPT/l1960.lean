import Mathlib

namespace total_milk_bottles_l1960_196090

theorem total_milk_bottles (marcus_bottles : ℕ) (john_bottles : ℕ) (h1 : marcus_bottles = 25) (h2 : john_bottles = 20) : marcus_bottles + john_bottles = 45 := by
  sorry

end total_milk_bottles_l1960_196090


namespace sin_neg_pi_over_three_l1960_196071

theorem sin_neg_pi_over_three : Real.sin (-Real.pi / 3) = -Real.sqrt 3 / 2 :=
by
  sorry

end sin_neg_pi_over_three_l1960_196071


namespace line_perpendicular_value_of_a_l1960_196070

theorem line_perpendicular_value_of_a :
  ∀ (a : ℝ),
    (∃ (l1 l2 : ℝ → ℝ),
      (∀ x, l1 x = (-a / (1 - a)) * x + 3 / (1 - a)) ∧
      (∀ x, l2 x = (-(a - 1) / (2 * a + 3)) * x + 2 / (2 * a + 3)) ∧
      (∀ x y, l1 x ≠ l2 y) ∧ 
      (-a / (1 - a)) * (-(a - 1) / (2 * a + 3)) = -1) →
    a = -3 := sorry

end line_perpendicular_value_of_a_l1960_196070


namespace goldfish_remaining_to_catch_l1960_196019

-- Define the number of total goldfish in the aquarium
def total_goldfish : ℕ := 100

-- Define the number of goldfish Maggie is allowed to take home (half of total goldfish)
def allowed_to_take_home := total_goldfish / 2

-- Define the number of goldfish Maggie caught (3/5 of allowed_to_take_home)
def caught := (3 * allowed_to_take_home) / 5

-- Prove the number of goldfish Maggie remains with to catch
theorem goldfish_remaining_to_catch : allowed_to_take_home - caught = 20 := by
  -- Sorry is used to skip the proof
  sorry

end goldfish_remaining_to_catch_l1960_196019


namespace dot_product_eq_one_l1960_196097

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (-1, 2)

theorem dot_product_eq_one : (a.1 * b.1 + a.2 * b.2) = 1 := by
  sorry

end dot_product_eq_one_l1960_196097


namespace circle_condition_l1960_196065

theorem circle_condition (k : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 - 4 * x + 2 * y + 5 * k = 0) ↔ k < 1 := 
sorry

end circle_condition_l1960_196065


namespace twelve_sided_figure_area_is_13_cm2_l1960_196067

def twelve_sided_figure_area_cm2 : ℝ :=
  let unit_square := 1
  let full_squares := 9
  let triangle_pairs := 4
  full_squares * unit_square + triangle_pairs * unit_square

theorem twelve_sided_figure_area_is_13_cm2 :
  twelve_sided_figure_area_cm2 = 13 := 
by
  sorry

end twelve_sided_figure_area_is_13_cm2_l1960_196067


namespace geometric_sequence_general_term_and_sum_l1960_196039

theorem geometric_sequence_general_term_and_sum (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ) 
  (h₁ : ∀ n, a n = 2 ^ n)
  (h₂ : ∀ n, b n = 2 * n - 1)
  : (∀ n, T n = 6 + (2 * n - 3) * 2 ^ (n + 1)) :=
by {
  sorry
}

end geometric_sequence_general_term_and_sum_l1960_196039


namespace initial_amount_liquid_A_l1960_196087

-- Definitions and conditions
def initial_ratio (a : ℕ) (b : ℕ) := a = 4 * b
def replaced_mixture_ratio (a : ℕ) (b : ℕ) (r₀ r₁ : ℕ) := 4 * r₀ = 2 * (r₁ + 20)

-- Theorem to prove the initial amount of liquid A
theorem initial_amount_liquid_A (a b r₀ r₁ : ℕ) :
  initial_ratio a b → replaced_mixture_ratio a b r₀ r₁ → a = 16 := 
by
  sorry

end initial_amount_liquid_A_l1960_196087


namespace inscribed_circle_radius_eq_four_l1960_196026

theorem inscribed_circle_radius_eq_four
  (A p s r : ℝ)
  (hA : A = 2 * p)
  (hp : p = 2 * s)
  (hArea : A = r * s) :
  r = 4 :=
by
  -- Proof would go here.
  sorry

end inscribed_circle_radius_eq_four_l1960_196026


namespace zebras_total_games_l1960_196018

theorem zebras_total_games 
  (x y : ℝ)
  (h1 : x = 0.40 * y)
  (h2 : (x + 8) / (y + 11) = 0.55) 
  : y + 11 = 24 :=
sorry

end zebras_total_games_l1960_196018


namespace joan_original_seashells_l1960_196079

-- Definitions based on the conditions
def seashells_left : ℕ := 27
def seashells_given_away : ℕ := 43

-- Theorem statement
theorem joan_original_seashells : 
  seashells_left + seashells_given_away = 70 := 
by
  sorry

end joan_original_seashells_l1960_196079


namespace find_b_l1960_196083

theorem find_b (p : ℕ) (hp : Nat.Prime p) :
  (∃ b : ℕ, b = (p + 1) ^ 2 ∨ b = 4 * p ∧ ∀ (x1 x2 : ℤ), x1 * x2 = p * b ∧ x1 + x2 = b) → 
  (∃ b : ℕ, b = (p + 1) ^ 2 ∨ b = 4 * p) :=
by
  sorry

end find_b_l1960_196083


namespace avg_marks_l1960_196047

theorem avg_marks (P C M B E H G : ℝ) 
  (h1 : C = P + 75)
  (h2 : M = P + 105)
  (h3 : B = P - 15)
  (h4 : E = P - 25)
  (h5 : H = P - 25)
  (h6 : G = P - 25)
  (h7 : P + C + M + B + E + H + G = P + 520) :
  (M + B + H + G) / 4 = 82 :=
by 
  sorry

end avg_marks_l1960_196047


namespace pq_inequality_l1960_196075

theorem pq_inequality (p : ℝ) (q : ℝ) (hp : 0 ≤ p) (hp2 : p < 2) (hq : q > 0) :
  4 * (p * q^2 + 2 * p^2 * q + 4 * q^2 + 5 * p * q) / (p + q) > 3 * p^2 * q :=
by {
  sorry
}

end pq_inequality_l1960_196075


namespace find_set_of_x_l1960_196068

noncomputable def exponential_inequality_solution (x : ℝ) : Prop :=
  1 < Real.exp x ∧ Real.exp x < 2

theorem find_set_of_x (x : ℝ) :
  exponential_inequality_solution x ↔ 0 < x ∧ x < Real.log 2 :=
by
  sorry

end find_set_of_x_l1960_196068


namespace calculate_final_price_l1960_196029

noncomputable def final_price (j_init p_init : ℝ) (j_inc p_inc : ℝ) (tax discount : ℝ) (j_quantity p_quantity : ℕ) : ℝ :=
  let j_new := j_init + j_inc
  let p_new := p_init * (1 + p_inc)
  let total_price := (j_new * j_quantity) + (p_new * p_quantity)
  let tax_amount := total_price * tax
  let price_with_tax := total_price + tax_amount
  let final_price := if j_quantity > 1 ∧ p_quantity >= 3 then price_with_tax * (1 - discount) else price_with_tax
  final_price

theorem calculate_final_price :
  final_price 30 100 10 (0.20) (0.07) (0.10) 2 5 = 654.84 :=
by
  sorry

end calculate_final_price_l1960_196029


namespace percentage_difference_j_p_l1960_196080

theorem percentage_difference_j_p (j p t : ℝ) (h1 : j = t * 80 / 100) 
  (h2 : t = p * (100 - t) / 100) (h3 : t = 6.25) : 
  ((p - j) / p) * 100 = 25 := 
by
  sorry

end percentage_difference_j_p_l1960_196080


namespace group_c_right_angled_triangle_l1960_196084

theorem group_c_right_angled_triangle :
  (3^2 + 4^2 = 5^2) := by
  sorry

end group_c_right_angled_triangle_l1960_196084


namespace maximize_binom_term_l1960_196043

theorem maximize_binom_term :
  ∃ k, k ∈ Finset.range (207) ∧
  (∀ m ∈ Finset.range (207), (Nat.choose 206 k * (Real.sqrt 5)^k) ≥ (Nat.choose 206 m * (Real.sqrt 5)^m)) ∧ k = 143 :=
sorry

end maximize_binom_term_l1960_196043


namespace A_inter_CUB_eq_l1960_196023

noncomputable def U := Set.univ (ℝ)

noncomputable def A := { x : ℝ | 0 ≤ x ∧ x ≤ 2 }

noncomputable def B := { y : ℝ | ∃ x : ℝ, x ∈ A ∧ y = x + 1 }

noncomputable def C_U (s : Set ℝ) := { x : ℝ | x ∉ s }

noncomputable def A_inter_CUB := A ∩ C_U B

theorem A_inter_CUB_eq : A_inter_CUB = { x : ℝ | 0 ≤ x ∧ x < 1 } :=
  by sorry

end A_inter_CUB_eq_l1960_196023


namespace simplify_expression_correct_l1960_196053

def simplify_expression : ℚ :=
  15 * (7 / 10) * (1 / 9)

theorem simplify_expression_correct : simplify_expression = 7 / 6 :=
by
  unfold simplify_expression
  sorry

end simplify_expression_correct_l1960_196053


namespace mateo_orange_bottles_is_1_l1960_196098

def number_of_orange_bottles_mateo_has (mateo_orange : ℕ) : Prop :=
  let julios_orange_bottles := 4
  let julios_grape_bottles := 7
  let mateos_grape_bottles := 3
  let liters_per_bottle := 2
  let julios_total_liters := (julios_orange_bottles + julios_grape_bottles) * liters_per_bottle
  let mateos_grape_liters := mateos_grape_bottles * liters_per_bottle
  let mateos_total_liters := (mateo_orange * liters_per_bottle) + mateos_grape_liters
  let additional_liters_to_julio := 14
  julios_total_liters = mateos_total_liters + additional_liters_to_julio

/-
Prove that Mateo has exactly 1 bottle of orange soda (assuming the problem above)
-/
theorem mateo_orange_bottles_is_1 : number_of_orange_bottles_mateo_has 1 :=
sorry

end mateo_orange_bottles_is_1_l1960_196098


namespace land_for_crop_production_l1960_196035

-- Conditions as Lean definitions
def total_land : ℕ := 150
def house_and_machinery : ℕ := 25
def future_expansion : ℕ := 15
def cattle_rearing : ℕ := 40

-- Proof statement defining the goal
theorem land_for_crop_production : 
  total_land - (house_and_machinery + future_expansion + cattle_rearing) = 70 := 
by
  sorry

end land_for_crop_production_l1960_196035


namespace total_cost_all_children_l1960_196054

-- Defining the constants and conditions
def regular_tuition : ℕ := 45
def early_bird_discount : ℕ := 15
def first_sibling_discount : ℕ := 15
def additional_sibling_discount : ℕ := 10
def weekend_class_extra_cost : ℕ := 20
def multi_instrument_discount : ℕ := 10

def Ali_cost : ℕ := regular_tuition - early_bird_discount
def Matt_cost : ℕ := regular_tuition - first_sibling_discount
def Jane_cost : ℕ := regular_tuition - additional_sibling_discount + weekend_class_extra_cost - multi_instrument_discount
def Sarah_cost : ℕ := regular_tuition - additional_sibling_discount + weekend_class_extra_cost - multi_instrument_discount

-- Proof statement
theorem total_cost_all_children : Ali_cost + Matt_cost + Jane_cost + Sarah_cost = 150 := by
  sorry

end total_cost_all_children_l1960_196054


namespace total_amount_shared_l1960_196016

theorem total_amount_shared (x y z : ℝ) (h1 : x = 1.25 * y) (h2 : y = 1.20 * z) (hz : z = 100) :
  x + y + z = 370 := by
  sorry

end total_amount_shared_l1960_196016


namespace range_of_a_l1960_196044

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then 
  a * x + 1 - 4 * a 
else 
  x ^ 2 - 3 * a * x

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = f a x2) → 
  a ∈ (Set.Ioi (2/3) ∪ Set.Iic 0) :=
sorry

end range_of_a_l1960_196044


namespace problem1_expr_eval_l1960_196034

theorem problem1_expr_eval : 
  (1:ℤ) - (1:ℤ)^(2022:ℕ) - (3 * (2/3:ℚ)^2 - (8/3:ℚ) / ((-2)^3:ℤ)) = -8/3 :=
by
  sorry

end problem1_expr_eval_l1960_196034


namespace bowling_tournament_orders_l1960_196057

theorem bowling_tournament_orders :
  let choices := 2
  let number_of_games := 5
  let total_orders := choices ^ number_of_games
  total_orders = 32 :=
by
  let choices := 2
  let number_of_games := 5
  let total_orders := choices ^ number_of_games
  show total_orders = 32
  sorry

end bowling_tournament_orders_l1960_196057


namespace solve_inequality_system_l1960_196022

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := by
  sorry

end solve_inequality_system_l1960_196022


namespace Laura_bought_one_kg_of_potatoes_l1960_196073

theorem Laura_bought_one_kg_of_potatoes :
  let price_salad : ℝ := 3
  let price_beef_per_kg : ℝ := 2 * price_salad
  let price_potato_per_kg : ℝ := price_salad * (1 / 3)
  let price_juice_per_liter : ℝ := 1.5
  let total_cost : ℝ := 22
  let num_salads : ℝ := 2
  let num_beef_kg : ℝ := 2
  let num_juice_liters : ℝ := 2
  let cost_salads := num_salads * price_salad
  let cost_beef := num_beef_kg * price_beef_per_kg
  let cost_juice := num_juice_liters * price_juice_per_liter
  (total_cost - (cost_salads + cost_beef + cost_juice)) / price_potato_per_kg = 1 :=
sorry

end Laura_bought_one_kg_of_potatoes_l1960_196073


namespace y_intercept_of_line_is_minus_one_l1960_196066

theorem y_intercept_of_line_is_minus_one : 
  (∀ x y : ℝ, y = 2 * x - 1 → y = -1) :=
by
  sorry

end y_intercept_of_line_is_minus_one_l1960_196066


namespace lcm_hcf_product_l1960_196096

theorem lcm_hcf_product (A B : ℕ) (h_prod : A * B = 18000) (h_hcf : Nat.gcd A B = 30) : Nat.lcm A B = 600 :=
sorry

end lcm_hcf_product_l1960_196096


namespace living_room_size_l1960_196020

theorem living_room_size :
  let length := 16
  let width := 10
  let total_rooms := 6
  let total_area := length * width
  let unit_size := total_area / total_rooms
  let living_room_size := 3 * unit_size
  living_room_size = 80 := by
    sorry

end living_room_size_l1960_196020


namespace Jina_mascots_total_l1960_196045

theorem Jina_mascots_total :
  let teddies := 5
  let bunnies := 3 * teddies
  let koala := 1
  let additional_teddies := 2 * bunnies
  teddies + bunnies + koala + additional_teddies = 51 :=
by
  let teddies := 5
  let bunnies := 3 * teddies
  let koala := 1
  let additional_teddies := 2 * bunnies
  show teddies + bunnies + koala + additional_teddies = 51
  sorry

end Jina_mascots_total_l1960_196045


namespace smallest_integer_x_divisibility_l1960_196074

theorem smallest_integer_x_divisibility :
  ∃ x : ℤ, (2 * x + 2) % 33 = 0 ∧ (2 * x + 2) % 44 = 0 ∧ (2 * x + 2) % 55 = 0 ∧ (2 * x + 2) % 666 = 0 ∧ x = 36629 := 
sorry

end smallest_integer_x_divisibility_l1960_196074


namespace age_of_female_employee_when_hired_l1960_196064

-- Defining the conditions
def hired_year : ℕ := 1989
def retirement_year : ℕ := 2008
def sum_age_employment : ℕ := 70

-- Given the conditions we found that years of employment (Y):
def years_of_employment : ℕ := retirement_year - hired_year -- 19

-- Defining the age when hired (A)
def age_when_hired : ℕ := sum_age_employment - years_of_employment -- 51

-- Now we need to prove
theorem age_of_female_employee_when_hired : age_when_hired = 51 :=
by
  -- Here should be the proof steps, but we use sorry for now
  sorry

end age_of_female_employee_when_hired_l1960_196064


namespace ratio_problem_l1960_196061

/-
  Given the ratio A : B : C = 3 : 2 : 5, we need to prove that 
  (2 * A + 3 * B) / (5 * C - 2 * A) = 12 / 19.
-/

theorem ratio_problem
  (A B C : ℚ)
  (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (2 * A + 3 * B) / (5 * C - 2 * A) = 12 / 19 :=
by sorry

end ratio_problem_l1960_196061


namespace equal_real_roots_value_of_m_l1960_196059

theorem equal_real_roots_value_of_m (m : ℝ) (h : (x^2 - 4*x + m = 0)) 
  (discriminant_zero : (16 - 4*m) = 0) : m = 4 :=
sorry

end equal_real_roots_value_of_m_l1960_196059


namespace r_fourth_power_sum_l1960_196031

theorem r_fourth_power_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_fourth_power_sum_l1960_196031


namespace fraction_of_students_getting_A_l1960_196041

theorem fraction_of_students_getting_A
    (frac_B : ℚ := 1/2)
    (frac_C : ℚ := 1/8)
    (frac_D : ℚ := 1/12)
    (frac_F : ℚ := 1/24)
    (passing_grade_frac: ℚ := 0.875) :
    (1 - (frac_B + frac_C + frac_D + frac_F) = 1/8) :=
by
  sorry

end fraction_of_students_getting_A_l1960_196041


namespace enrico_earnings_l1960_196021

def roosterPrice (weight: ℕ) : ℝ :=
  if weight < 20 then weight * 0.80
  else if weight ≤ 35 then weight * 0.65
  else weight * 0.50

theorem enrico_earnings :
  roosterPrice 15 + roosterPrice 30 + roosterPrice 40 + roosterPrice 50 = 76.50 := 
by
  sorry

end enrico_earnings_l1960_196021


namespace edward_money_left_l1960_196037

noncomputable def toy_cost : ℝ := 0.95

noncomputable def toy_quantity : ℕ := 4

noncomputable def toy_discount : ℝ := 0.15

noncomputable def race_track_cost : ℝ := 6.00

noncomputable def race_track_tax : ℝ := 0.08

noncomputable def initial_amount : ℝ := 17.80

noncomputable def total_toy_cost_before_discount : ℝ := toy_quantity * toy_cost

noncomputable def discount_amount : ℝ := toy_discount * total_toy_cost_before_discount

noncomputable def total_toy_cost_after_discount : ℝ := total_toy_cost_before_discount - discount_amount

noncomputable def race_track_tax_amount : ℝ := race_track_tax * race_track_cost

noncomputable def total_race_track_cost_after_tax : ℝ := race_track_cost + race_track_tax_amount

noncomputable def total_amount_spent : ℝ := total_toy_cost_after_discount + total_race_track_cost_after_tax

noncomputable def money_left : ℝ := initial_amount - total_amount_spent

theorem edward_money_left : money_left = 8.09 := by
  -- proof goes here
  sorry

end edward_money_left_l1960_196037


namespace perimeter_right_triangle_l1960_196038

-- Given conditions
def area : ℝ := 200
def b : ℝ := 20

-- Mathematical problem
theorem perimeter_right_triangle :
  ∀ (x c : ℝ), 
  (1 / 2) * b * x = area →
  c^2 = x^2 + b^2 →
  x + b + c = 40 + 20 * Real.sqrt 2 := 
  by
  sorry

end perimeter_right_triangle_l1960_196038


namespace prepaid_card_cost_correct_l1960_196040

noncomputable def prepaid_phone_card_cost
    (cost_per_minute : ℝ) (call_minutes : ℝ) (remaining_credit : ℝ) : ℝ :=
  remaining_credit + (call_minutes * cost_per_minute)

theorem prepaid_card_cost_correct :
  let cost_per_minute := 0.16
  let call_minutes := 22
  let remaining_credit := 26.48
  prepaid_phone_card_cost cost_per_minute call_minutes remaining_credit = 30.00 := by
  sorry

end prepaid_card_cost_correct_l1960_196040


namespace digits_interchanged_l1960_196006

theorem digits_interchanged (a b k : ℤ) (h : 10 * a + b = k * (a + b) + 2) :
  10 * b + a = (k + 9) * (a + b) + 2 :=
by
  sorry

end digits_interchanged_l1960_196006


namespace range_of_k_l1960_196002

theorem range_of_k (x y k : ℝ) (h1 : x - y = k - 1) (h2 : 3 * x + 2 * y = 4 * k + 5) (hk : 2 * x + 3 * y > 7) : k > 1 / 3 := 
sorry

end range_of_k_l1960_196002


namespace square_area_increase_l1960_196042

theorem square_area_increase (s : ℕ) (h : (s = 5) ∨ (s = 10) ∨ (s = 15)) :
  (1.35^2 - 1) * 100 = 82.25 :=
by
  sorry

end square_area_increase_l1960_196042


namespace quadratic_no_real_roots_iff_l1960_196015

theorem quadratic_no_real_roots_iff (m : ℝ) : (∀ x : ℝ, x^2 + 3 * x + m ≠ 0) ↔ m > 9 / 4 :=
by
  sorry

end quadratic_no_real_roots_iff_l1960_196015


namespace possible_values_of_ratio_l1960_196010

theorem possible_values_of_ratio (a d : ℝ) (h : a ≠ 0) (h_eq : a^2 - 6 * a * d + 8 * d^2 = 0) : 
  ∃ x : ℝ, (x = 1/2 ∨ x = 1/4) ∧ x = d/a :=
by
  sorry

end possible_values_of_ratio_l1960_196010


namespace solve_equation_1_solve_equation_2_l1960_196007

theorem solve_equation_1 :
  ∀ x : ℝ, (2 * x - 1) ^ 2 = 9 ↔ (x = 2 ∨ x = -1) :=
by
  sorry

theorem solve_equation_2 :
  ∀ x : ℝ, x ^ 2 - 4 * x - 12 = 0 ↔ (x = 6 ∨ x = -2) :=
by
  sorry

end solve_equation_1_solve_equation_2_l1960_196007


namespace population_hypothetical_town_l1960_196027

theorem population_hypothetical_town :
  ∃ (a b c : ℕ), a^2 + 150 = b^2 + 1 ∧ b^2 + 1 + 150 = c^2 ∧ a^2 = 5476 :=
by {
  sorry
}

end population_hypothetical_town_l1960_196027


namespace expression_value_l1960_196077

theorem expression_value (x : ℝ) (h : x = 4) :
  (x^2 - 2*x - 15) / (x - 5) = 7 :=
sorry

end expression_value_l1960_196077


namespace church_full_capacity_l1960_196052

theorem church_full_capacity
  (chairs_per_row : ℕ)
  (rows : ℕ)
  (people_per_chair : ℕ)
  (h1 : chairs_per_row = 6)
  (h2 : rows = 20)
  (h3 : people_per_chair = 5) :
  (chairs_per_row * rows * people_per_chair) = 600 := by
  sorry

end church_full_capacity_l1960_196052


namespace min_value_inequality_l1960_196095

theorem min_value_inequality (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + 3 * b = 1) : 
  (2 / a + 3 / b) ≥ 14 :=
sorry

end min_value_inequality_l1960_196095


namespace connie_correct_answer_l1960_196099

theorem connie_correct_answer (y : ℕ) (h1 : y - 8 = 32) : y + 8 = 48 := by
  sorry

end connie_correct_answer_l1960_196099


namespace new_credit_card_balance_l1960_196013

theorem new_credit_card_balance (i g x r n : ℝ)
    (h_i : i = 126)
    (h_g : g = 60)
    (h_x : x = g / 2)
    (h_r : r = 45)
    (h_n : n = (i + g + x) - r) :
    n = 171 :=
sorry

end new_credit_card_balance_l1960_196013


namespace intersection_of_A_and_B_l1960_196008

open Set

def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2 < x ∧ x ≤ 5}

theorem intersection_of_A_and_B : A ∩ B = {x | 2 < x ∧ x < 3} := 
by
  sorry

end intersection_of_A_and_B_l1960_196008


namespace required_fraction_l1960_196014

theorem required_fraction
  (total_members : ℝ)
  (top_10_lists : ℝ) :
  total_members = 775 →
  top_10_lists = 193.75 →
  top_10_lists / total_members = 0.25 :=
by
  sorry

end required_fraction_l1960_196014


namespace max_x_add_inv_x_l1960_196063

variable (x : ℝ) (y : Fin 2022 → ℝ)

-- Conditions
def sum_condition : Prop := x + (Finset.univ.sum y) = 2024
def reciprocal_sum_condition : Prop := (1/x) + (Finset.univ.sum (λ i => 1 / (y i))) = 2024

-- The statement we need to prove
theorem max_x_add_inv_x (h_sum : sum_condition x y) (h_rec_sum : reciprocal_sum_condition x y) : 
  x + (1/x) ≤ 2 := by
  sorry

end max_x_add_inv_x_l1960_196063


namespace Cindy_walking_speed_l1960_196093

noncomputable def walking_speed (total_time : ℕ) (running_speed : ℕ) (running_distance : ℚ) (walking_distance : ℚ) : ℚ := 
  let time_to_run := running_distance / running_speed
  let walking_time := total_time - (time_to_run * 60)
  walking_distance / (walking_time / 60)

theorem Cindy_walking_speed : walking_speed 40 3 0.5 0.5 = 1 := 
  sorry

end Cindy_walking_speed_l1960_196093


namespace geometric_series_sum_150_terms_l1960_196072

theorem geometric_series_sum_150_terms (a : ℕ) (r : ℝ)
  (h₁ : a = 250)
  (h₂ : (a - a * r ^ 50) / (1 - r) = 625)
  (h₃ : (a - a * r ^ 100) / (1 - r) = 1225) :
  (a - a * r ^ 150) / (1 - r) = 1801 := by
  sorry

end geometric_series_sum_150_terms_l1960_196072


namespace total_trees_after_planting_l1960_196004

-- Define the initial counts of the trees
def initial_maple_trees : ℕ := 2
def initial_poplar_trees : ℕ := 5
def initial_oak_trees : ℕ := 4

-- Define the planting rules
def maple_trees_planted (initial_maple : ℕ) : ℕ := 3 * initial_maple
def poplar_trees_planted (initial_poplar : ℕ) : ℕ := 3 * initial_poplar

-- Calculate the total number of each type of tree after planting
def total_maple_trees (initial_maple : ℕ) : ℕ :=
  initial_maple + maple_trees_planted initial_maple

def total_poplar_trees (initial_poplar : ℕ) : ℕ :=
  initial_poplar + poplar_trees_planted initial_poplar

def total_oak_trees (initial_oak : ℕ) : ℕ := initial_oak

-- Calculate the total number of trees in the park
def total_trees (initial_maple initial_poplar initial_oak : ℕ) : ℕ :=
  total_maple_trees initial_maple + total_poplar_trees initial_poplar + total_oak_trees initial_oak

-- The proof statement
theorem total_trees_after_planting :
  total_trees initial_maple_trees initial_poplar_trees initial_oak_trees = 32 := 
by
  -- Proof placeholder
  sorry

end total_trees_after_planting_l1960_196004


namespace determine_a_l1960_196030

lemma even_exponent (a : ℤ) : (a^2 - 4*a) % 2 = 0 :=
sorry

lemma decreasing_function (a : ℤ) : a^2 - 4*a < 0 :=
sorry

theorem determine_a (a : ℤ) (h1 : (a^2 - 4*a) % 2 = 0) (h2 : a^2 - 4*a < 0) : a = 2 :=
sorry

end determine_a_l1960_196030


namespace kibble_consumption_rate_l1960_196017

-- Kira fills her cat's bowl with 3 pounds of kibble before going to work.
def initial_kibble : ℚ := 3

-- There is still 1 pound left when she returns.
def remaining_kibble : ℚ := 1

-- Kira was away from home for 8 hours.
def time_away : ℚ := 8

-- Calculate the amount of kibble eaten
def kibble_eaten : ℚ := initial_kibble - remaining_kibble

-- Calculate the rate of consumption (hours per pound)
def rate_of_consumption (time: ℚ) (kibble: ℚ) : ℚ := time / kibble

-- Theorem statement: It takes 4 hours for Kira's cat to eat a pound of kibble.
theorem kibble_consumption_rate : rate_of_consumption time_away kibble_eaten = 4 := by
  sorry

end kibble_consumption_rate_l1960_196017


namespace roots_magnitude_order_l1960_196058

theorem roots_magnitude_order (m : ℝ) (a b c d : ℝ)
  (h1 : m > 0)
  (h2 : a ^ 2 - m * a - 1 = 0)
  (h3 : b ^ 2 - m * b - 1 = 0)
  (h4 : c ^ 2 + m * c - 1 = 0)
  (h5 : d ^ 2 + m * d - 1 = 0)
  (ha_pos : a > 0) (hb_neg : b < 0)
  (hc_pos : c > 0) (hd_neg : d < 0) :
  |a| > |c| ∧ |c| > |b| ∧ |b| > |d| :=
sorry

end roots_magnitude_order_l1960_196058


namespace mom_prepared_pieces_l1960_196049

-- Define the conditions
def jane_pieces : ℕ := 4
def total_eaters : ℕ := 3

-- Define the hypothesis that each of the eaters ate an equal number of pieces
def each_ate_equal (pieces : ℕ) : Prop := pieces = jane_pieces

-- The number of pieces Jane's mom prepared
theorem mom_prepared_pieces : total_eaters * jane_pieces = 12 :=
by
  -- Placeholder for actual proof
  sorry

end mom_prepared_pieces_l1960_196049


namespace trapezoid_ratio_l1960_196025

-- Define the isosceles trapezoid properties and the point inside it
noncomputable def isosceles_trapezoid (r s : ℝ) (hr : r > s) (triangle_areas : List ℝ) : Prop :=
  triangle_areas = [2, 3, 4, 5]

-- Define the problem statement
theorem trapezoid_ratio (r s : ℝ) (hr : r > s) (areas : List ℝ) (hareas : isosceles_trapezoid r s hr areas) :
  r / s = 2 + Real.sqrt 2 := sorry

end trapezoid_ratio_l1960_196025


namespace rider_distance_traveled_l1960_196094

noncomputable def caravan_speed := 1  -- km/h
noncomputable def rider_speed := 1 + Real.sqrt 2  -- km/h

theorem rider_distance_traveled : 
  (1 / (rider_speed - 1) + 1 / (rider_speed + 1)) = 1 :=
by
  sorry

end rider_distance_traveled_l1960_196094


namespace avg_xy_36_l1960_196056

-- Given condition: The average of the numbers 2, 6, 10, x, and y is 18
def avg_condition (x y : ℝ) : Prop :=
  (2 + 6 + 10 + x + y) / 5 = 18

-- Goal: To prove that the average of x and y is 36
theorem avg_xy_36 (x y : ℝ) (h : avg_condition x y) : (x + y) / 2 = 36 :=
by
  sorry

end avg_xy_36_l1960_196056


namespace smallest_four_digit_mod_8_l1960_196011

theorem smallest_four_digit_mod_8 :
  ∃ x : ℕ, x ≥ 1000 ∧ x < 10000 ∧ x % 8 = 5 ∧ (∀ y : ℕ, y >= 1000 ∧ y < 10000 ∧ y % 8 = 5 → x ≤ y) :=
sorry

end smallest_four_digit_mod_8_l1960_196011


namespace negation_of_proposition_l1960_196092

theorem negation_of_proposition :
  (¬ ∀ (a b : ℝ), a > b → a^2 > b^2) ↔ ∃ (a b : ℝ), a ≤ b ∧ a^2 ≤ b^2 :=
sorry

end negation_of_proposition_l1960_196092


namespace oliver_shirts_problem_l1960_196028

-- Defining the quantities of short sleeve shirts, long sleeve shirts, and washed shirts.
def shortSleeveShirts := 39
def longSleeveShirts  := 47
def shirtsWashed := 20

-- Stating the problem formally.
theorem oliver_shirts_problem :
  shortSleeveShirts + longSleeveShirts - shirtsWashed = 66 :=
by
  -- Proof goes here.
  sorry

end oliver_shirts_problem_l1960_196028


namespace triangle_area_example_l1960_196032

-- Define the right triangle DEF with angle at D being 45 degrees and DE = 8 units
noncomputable def area_of_45_45_90_triangle (DE : ℝ) (angle_d : ℝ) (h_angle : angle_d = 45) (h_DE : DE = 8) : ℝ :=
  1 / 2 * DE * DE

-- State the theorem to prove the area
theorem triangle_area_example {DE : ℝ} {angle_d : ℝ} (h_angle : angle_d = 45) (h_DE : DE = 8) :
  area_of_45_45_90_triangle DE angle_d h_angle h_DE = 32 := 
sorry

end triangle_area_example_l1960_196032


namespace not_all_terms_positive_l1960_196076

variable (a b c d : ℝ)
variable (e f g h : ℝ)

theorem not_all_terms_positive
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (he : e < 0) (hf : f < 0) (hg : g < 0) (hh : h < 0) :
  ¬ ((a * e + b * c > 0) ∧ (e * f + c * g > 0) ∧ (f * d + g * h > 0) ∧ (d * a + h * b > 0)) :=
sorry

end not_all_terms_positive_l1960_196076


namespace cards_problem_l1960_196069

-- Define the conditions and goal
theorem cards_problem 
    (L R : ℕ) 
    (h1 : L + 6 = 3 * (R - 6))
    (h2 : R + 2 = 2 * (L - 2)) : 
    L = 66 := 
by 
  -- proof goes here
  sorry

end cards_problem_l1960_196069


namespace area_of_triangle_ABC_l1960_196086

def A : ℝ × ℝ := (4, -3)
def B : ℝ × ℝ := (-1, 2)
def C : ℝ × ℝ := (2, -7)

theorem area_of_triangle_ABC : 
  let v := (A.1 - C.1, A.2 - C.2)
  let w := (B.1 - C.1, B.2 - C.2)
  let parallelogram_area := |v.1 * w.2 - v.2 * w.1|
  let triangle_area := parallelogram_area / 2
  triangle_area = 15 :=
by
  sorry

end area_of_triangle_ABC_l1960_196086


namespace perfect_square_solutions_l1960_196060

theorem perfect_square_solutions (a b : ℕ) (ha : a > b) (ha_pos : 0 < a) (hb_pos : 0 < b) (hA : ∃ k : ℕ, a^2 + 4 * b + 1 = k^2) (hB : ∃ l : ℕ, b^2 + 4 * a + 1 = l^2) :
  a = 8 ∧ b = 4 ∧ (a^2 + 4 * b + 1 = (a+1)^2) ∧ (b^2 + 4 * a + 1 = (b + 3)^2) :=
by
  sorry

end perfect_square_solutions_l1960_196060


namespace find_c_minus_d_l1960_196082

variable (g : ℝ → ℝ)
variable (c d : ℝ)
variable (invertible_g : Function.Injective g)
variable (g_at_c : g c = d)
variable (g_at_d : g d = 5)

theorem find_c_minus_d : c - d = -3 := by
  sorry

end find_c_minus_d_l1960_196082


namespace average_production_l1960_196003

theorem average_production (n : ℕ) :
  let total_past_production := 50 * n
  let total_production_including_today := 100 + total_past_production
  let average_production := total_production_including_today / (n + 1)
  average_production = 55
  -> n = 9 :=
by
  sorry

end average_production_l1960_196003


namespace cards_received_at_home_l1960_196036

-- Definitions based on the conditions
def initial_cards := 403
def total_cards := 690

-- The theorem to prove the number of cards received at home
theorem cards_received_at_home : total_cards - initial_cards = 287 :=
by
  -- Proof goes here, but we use sorry as a placeholder.
  sorry

end cards_received_at_home_l1960_196036


namespace complement_intersection_l1960_196062

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | 3 ≤ x}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5}

theorem complement_intersection (x : ℝ) : x ∈ (U \ A ∩ B) ↔ (0 ≤ x ∧ x < 3) :=
by {
  sorry
}

end complement_intersection_l1960_196062


namespace montoya_family_budget_on_food_l1960_196005

def spending_on_groceries : ℝ := 0.6
def spending_on_eating_out : ℝ := 0.2

theorem montoya_family_budget_on_food :
  spending_on_groceries + spending_on_eating_out = 0.8 :=
  by
  sorry

end montoya_family_budget_on_food_l1960_196005


namespace symmetric_point_coordinates_l1960_196001

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def symmetric_about_x_axis (P : Point3D) : Point3D :=
  { x := P.x, y := -P.y, z := -P.z }

theorem symmetric_point_coordinates :
  symmetric_about_x_axis {x := 1, y := 3, z := 6} = {x := 1, y := -3, z := -6} :=
by
  sorry

end symmetric_point_coordinates_l1960_196001


namespace problem_solution_l1960_196024

theorem problem_solution (m : ℝ) (h : (m - 2023)^2 + (2024 - m)^2 = 2025) :
  (m - 2023) * (2024 - m) = -1012 :=
sorry

end problem_solution_l1960_196024


namespace find_x_l1960_196051

theorem find_x (x : ℚ) (h : 2 / 5 = (4 / 3) / x) : x = 10 / 3 :=
by
sorry

end find_x_l1960_196051


namespace line_does_not_pass_through_third_quadrant_l1960_196078

def line (x : ℝ) : ℝ := -x + 1

-- A line passes through the point (1, 0) and has a slope of -1
def passes_through_point (P : ℝ × ℝ) : Prop :=
  ∃ m b, m = -1 ∧ P.2 = m * P.1 + b ∧ line P.1 = P.2

def third_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 < 0

theorem line_does_not_pass_through_third_quadrant :
  ¬ ∃ p : ℝ × ℝ, passes_through_point p ∧ third_quadrant p :=
sorry

end line_does_not_pass_through_third_quadrant_l1960_196078


namespace sum_squares_of_six_consecutive_even_eq_1420_l1960_196085

theorem sum_squares_of_six_consecutive_even_eq_1420 
  (n : ℤ) 
  (h : n + (n + 2) + (n + 4) + (n + 6) + (n + 8) + (n + 10) = 90) :
  n^2 + (n + 2)^2 + (n + 4)^2 + (n + 6)^2 + (n + 8)^2 + (n + 10)^2 = 1420 :=
by
  sorry

end sum_squares_of_six_consecutive_even_eq_1420_l1960_196085


namespace find_a_b_sum_l1960_196048

-- Conditions
def f (x a b : ℝ) : ℝ := x^3 - a * x^2 - b * x + a^2

def f_prime (x a b : ℝ) : ℝ := 3 * x^2 - 2 * a * x - b

theorem find_a_b_sum (a b : ℝ) (h1 : f_prime 1 a b = 0) (h2 : f 1 a b = 10) : a + b = 7 := 
sorry

end find_a_b_sum_l1960_196048


namespace inequality_abc_l1960_196091

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  1 / (1 + 2 * a) + 1 / (1 + 2 * b) + 1 / (1 + 2 * c) ≥ 1 :=
by
  sorry

end inequality_abc_l1960_196091


namespace find_a_of_inequality_solution_set_l1960_196033

theorem find_a_of_inequality_solution_set :
  (∃ (a : ℝ), (∀ (x : ℝ), |2*x - a| + a ≤ 4 ↔ -1 ≤ x ∧ x ≤ 2) ∧ a = 1) :=
by sorry

end find_a_of_inequality_solution_set_l1960_196033


namespace freezer_temp_is_correct_l1960_196050

def freezer_temp (temp: ℤ) := temp

theorem freezer_temp_is_correct (temp: ℤ)
  (freezer_below_zero: temp = -18): freezer_temp temp = -18 := 
by
  -- since freezer_below_zero state that temperature is -18
  exact freezer_below_zero

end freezer_temp_is_correct_l1960_196050


namespace polygon_sides_eq_n_l1960_196012

theorem polygon_sides_eq_n
  (sum_except_two_angles : ℝ)
  (angle_equal : ℝ)
  (h1 : sum_except_two_angles = 2970)
  (h2 : angle_equal * 2 < 180)
  : ∃ n : ℕ, 180 * (n - 2) = 2970 + 2 * angle_equal ∧ n = 19 :=
by
  sorry

end polygon_sides_eq_n_l1960_196012


namespace no_such_n_l1960_196009

theorem no_such_n (n : ℕ) (h_positive : n > 0) : 
  ¬ ∃ k : ℕ, (n^2 + 1) = k * (Nat.floor (Real.sqrt n))^2 + 2 := by
  sorry

end no_such_n_l1960_196009


namespace scientific_notation_of_0_00003_l1960_196089

theorem scientific_notation_of_0_00003 :
  0.00003 = 3 * 10^(-5) :=
sorry

end scientific_notation_of_0_00003_l1960_196089


namespace calories_in_300g_lemonade_l1960_196081

def lemonade_calories (lemon_juice_in_g : Nat) (sugar_in_g : Nat) (water_in_g : Nat) (lemon_juice_cal : Nat) (sugar_cal : Nat) : Nat :=
  (lemon_juice_in_g * lemon_juice_cal / 100) + (sugar_in_g * sugar_cal / 100)

def total_weight (lemon_juice_in_g : Nat) (sugar_in_g : Nat) (water_in_g : Nat) : Nat :=
  lemon_juice_in_g + sugar_in_g + water_in_g

theorem calories_in_300g_lemonade :
  (lemonade_calories 500 200 1000 30 400) * 300 / (total_weight 500 200 1000) = 168 := 
  by
    sorry

end calories_in_300g_lemonade_l1960_196081


namespace reciprocal_of_neg_2023_l1960_196000

theorem reciprocal_of_neg_2023 : ∃ b : ℚ, -2023 * b = 1 ∧ b = -1 / 2023 :=
by
  sorry

end reciprocal_of_neg_2023_l1960_196000


namespace rectangle_sides_l1960_196088

theorem rectangle_sides (x y : ℕ) (h_diff : x ≠ y) (h_eq : x * y = 2 * x + 2 * y) : 
  (x = 3 ∧ y = 6) ∨ (x = 6 ∧ y = 3) :=
sorry

end rectangle_sides_l1960_196088


namespace first_worker_time_l1960_196055

def productivity (x y z : ℝ) : Prop :=
  x + y + z = 20 ∧
  (20 / x) > 3 ∧
  (20 / x) + (60 / (y + z)) = 8

theorem first_worker_time (x y z : ℝ) (h : productivity x y z) : 
  (80 / x) = 16 :=
  sorry

end first_worker_time_l1960_196055


namespace unique_solution_x_ln3_plus_x_ln4_eq_x_ln5_l1960_196046

theorem unique_solution_x_ln3_plus_x_ln4_eq_x_ln5 :
  ∃! x : ℝ, 0 < x ∧ x^(Real.log 3) + x^(Real.log 4) = x^(Real.log 5) := sorry

end unique_solution_x_ln3_plus_x_ln4_eq_x_ln5_l1960_196046
