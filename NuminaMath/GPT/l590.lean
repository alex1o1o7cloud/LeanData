import Mathlib

namespace relation_of_exponents_l590_59005

theorem relation_of_exponents
  (a b c d : ℝ)
  (x y p z : ℝ)
  (h1 : a^x = c)
  (h2 : b^p = c)
  (h3 : b^y = d)
  (h4 : a^z = d) :
  py = xz :=
sorry

end relation_of_exponents_l590_59005


namespace packs_of_sugar_l590_59007

theorem packs_of_sugar (cost_apples_per_kg cost_walnuts_per_kg cost_apples total : ℝ) (weight_apples weight_walnuts : ℝ) (less_sugar_by_1 : ℝ) (packs : ℕ) :
  cost_apples_per_kg = 2 →
  cost_walnuts_per_kg = 6 →
  cost_apples = weight_apples * cost_apples_per_kg →
  weight_apples = 5 →
  weight_walnuts = 0.5 →
  less_sugar_by_1 = 1 →
  total = 16 →
  packs = (total - (weight_apples * cost_apples_per_kg + weight_walnuts * cost_walnuts_per_kg)) / (cost_apples_per_kg - less_sugar_by_1) →
  packs = 3 :=
by
  sorry

end packs_of_sugar_l590_59007


namespace probability_of_selecting_one_defective_l590_59047

-- Definitions based on conditions from the problem
def items : List ℕ := [0, 1, 2, 3]  -- 0 represents defective, 1, 2, 3 represent genuine

def sample_space : List (ℕ × ℕ) := 
  [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

def event_A : List (ℕ × ℕ) := 
  [(0, 1), (0, 2), (0, 3)]

-- The probability of event A, calculated based on the classical method
def probability_event_A : ℚ := event_A.length / sample_space.length

theorem probability_of_selecting_one_defective : 
  probability_event_A = 1 / 2 := by
  sorry

end probability_of_selecting_one_defective_l590_59047


namespace rectangle_perimeter_l590_59003

variables (L W : ℕ)

-- conditions
def conditions : Prop :=
  L - 4 = W + 3 ∧
  (L - 4) * (W + 3) = L * W

-- prove the solution
theorem rectangle_perimeter (h : conditions L W) : 2 * L + 2 * W = 50 := sorry

end rectangle_perimeter_l590_59003


namespace total_miles_driven_l590_59062

-- Conditions
def miles_darius : ℕ := 679
def miles_julia : ℕ := 998

-- Proof statement
theorem total_miles_driven : miles_darius + miles_julia = 1677 := 
by
  -- placeholder for the proof steps
  sorry

end total_miles_driven_l590_59062


namespace rotated_square_height_l590_59051

noncomputable def height_of_B (side_length : ℝ) (rotation_angle : ℝ) : ℝ :=
  let diagonal := side_length * Real.sqrt 2
  let vertical_component := diagonal * Real.sin rotation_angle
  vertical_component

theorem rotated_square_height :
  height_of_B 1 (Real.pi / 6) = Real.sqrt 2 / 2 :=
by
  sorry

end rotated_square_height_l590_59051


namespace sum_first_five_arithmetic_l590_59010

theorem sum_first_five_arithmetic (a : ℕ → ℝ) (h₁ : ∀ n, a n = a 0 + n * (a 1 - a 0)) (h₂ : a 1 = -1) (h₃ : a 3 = -5) :
  (a 0 + a 1 + a 2 + a 3 + a 4) = -15 :=
by
  sorry

end sum_first_five_arithmetic_l590_59010


namespace partnership_profit_l590_59024

noncomputable def total_profit
  (P : ℝ)
  (mary_investment : ℝ := 700)
  (harry_investment : ℝ := 300)
  (effort_share := P / 3 / 2)
  (remaining_share := 2 / 3 * P)
  (total_investment := mary_investment + harry_investment)
  (mary_share_remaining := (mary_investment / total_investment) * remaining_share)
  (harry_share_remaining := (harry_investment / total_investment) * remaining_share) : Prop :=
  (effort_share + mary_share_remaining) - (effort_share + harry_share_remaining) = 800

theorem partnership_profit : ∃ P : ℝ, total_profit P ∧ P = 3000 :=
  sorry

end partnership_profit_l590_59024


namespace ratio_of_ages_l590_59006

theorem ratio_of_ages (sandy_future_age : ℕ) (sandy_years_future : ℕ) (molly_current_age : ℕ)
  (h1 : sandy_future_age = 42) (h2 : sandy_years_future = 6) (h3 : molly_current_age = 27) :
  (sandy_future_age - sandy_years_future) / gcd (sandy_future_age - sandy_years_future) molly_current_age = 
    4 / 3 :=
by
  sorry

end ratio_of_ages_l590_59006


namespace find_total_money_l590_59029

theorem find_total_money
  (d x T : ℝ)
  (h1 : d = 5 / 17)
  (h2 : x = 35)
  (h3 : d * T = x) :
  T = 119 :=
by sorry

end find_total_money_l590_59029


namespace minimize_quadratic_expression_l590_59049

theorem minimize_quadratic_expression:
  ∀ x : ℝ, (∃ a b c : ℝ, a = 1 ∧ b = -8 ∧ c = 15 ∧ x^2 + b * x + c ≥ (4 - 4)^2 - 1) :=
by
  sorry

end minimize_quadratic_expression_l590_59049


namespace dividend_is_176_l590_59020

theorem dividend_is_176 (divisor quotient remainder : ℕ) (h1 : divisor = 19) (h2 : quotient = 9) (h3 : remainder = 5) :
  divisor * quotient + remainder = 176 := by
  sorry

end dividend_is_176_l590_59020


namespace dividend_is_correct_l590_59098

theorem dividend_is_correct :
  ∃ (R D Q V: ℕ), R = 6 ∧ D = 5 * Q ∧ D = 3 * R + 2 ∧ V = D * Q + R ∧ V = 86 :=
by
  sorry

end dividend_is_correct_l590_59098


namespace milk_savings_l590_59038

theorem milk_savings :
  let cost_for_two_packs : ℝ := 2.50
  let cost_per_pack_individual : ℝ := 1.30
  let num_packs_per_set := 2
  let num_sets := 10
  let cost_per_pack_set := cost_for_two_packs / num_packs_per_set
  let savings_per_pack := cost_per_pack_individual - cost_per_pack_set
  let total_packs := num_sets * num_packs_per_set
  let total_savings := savings_per_pack * total_packs
  total_savings = 1 :=
by
  sorry

end milk_savings_l590_59038


namespace no_valid_digit_replacement_l590_59022

theorem no_valid_digit_replacement :
  ¬ ∃ (A B C D E M X : ℕ),
    (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ M ∧ A ≠ X ∧
     B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ M ∧ B ≠ X ∧
     C ≠ D ∧ C ≠ E ∧ C ≠ M ∧ C ≠ X ∧
     D ≠ E ∧ D ≠ M ∧ D ≠ X ∧
     E ≠ M ∧ E ≠ X ∧
     M ≠ X ∧
     0 ≤ A ∧ A < 10 ∧
     0 ≤ B ∧ B < 10 ∧
     0 ≤ C ∧ C < 10 ∧
     0 ≤ D ∧ D < 10 ∧
     0 ≤ E ∧ E < 10 ∧
     0 ≤ M ∧ M < 10 ∧
     0 ≤ X ∧ X < 10 ∧
     A * B * C * D + 1 = C * E * M * X) :=
sorry

end no_valid_digit_replacement_l590_59022


namespace cylinder_radius_l590_59001

theorem cylinder_radius
  (r h : ℝ) (S : ℝ) (h_cylinder : h = 8) (S_surface : S = 130 * Real.pi)
  (surface_area_eq : S = 2 * Real.pi * r^2 + 2 * Real.pi * r * h) :
  r = 5 :=
by
  sorry

end cylinder_radius_l590_59001


namespace quadratic_has_two_real_roots_l590_59021

theorem quadratic_has_two_real_roots (a b c : ℝ) (h : a * c < 0) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * (x1^2) + b * x1 + c = 0 ∧ a * (x2^2) + b * x2 + c = 0) :=
by
  sorry

end quadratic_has_two_real_roots_l590_59021


namespace arccos_half_eq_pi_div_three_l590_59080

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
by
  sorry

end arccos_half_eq_pi_div_three_l590_59080


namespace daily_serving_size_l590_59077

-- Definitions based on problem conditions
def days : ℕ := 180
def capsules_per_bottle : ℕ := 60
def bottles : ℕ := 6
def total_capsules : ℕ := bottles * capsules_per_bottle

-- Theorem statement to prove the daily serving size
theorem daily_serving_size :
  total_capsules / days = 2 := by
  sorry

end daily_serving_size_l590_59077


namespace geometric_sequence_sum_first_five_terms_l590_59093

theorem geometric_sequence_sum_first_five_terms
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : a 1 + a 3 = 10)
  (h2 : a 2 + a 4 = 30)
  (h_geom : ∀ n, a (n + 1) = a n * q) :
  (a 1 + a 2 + a 3 + a 4 + a 5) = 121 :=
sorry

end geometric_sequence_sum_first_five_terms_l590_59093


namespace airfare_price_for_BD_l590_59014

theorem airfare_price_for_BD (AB AC AD CD BC : ℝ) (hAB : AB = 2000) (hAC : AC = 1600) (hAD : AD = 2500) 
    (hCD : CD = 900) (hBC : BC = 1200) (proportional_pricing : ∀ x y : ℝ, x * (y / x) = y) : 
    ∃ BD : ℝ, BD = 1500 :=
by
  sorry

end airfare_price_for_BD_l590_59014


namespace probability_sum_18_l590_59099

def total_outcomes := 100

def successful_pairs := [(8, 10), (9, 9), (10, 8)]

def num_successful_outcomes := successful_pairs.length

theorem probability_sum_18 : (num_successful_outcomes / total_outcomes : ℚ) = 3 / 100 := 
by
  -- The actual proof should go here
  sorry

end probability_sum_18_l590_59099


namespace cameron_list_length_l590_59016

-- Definitions of multiples
def smallest_multiple_perfect_square := 900
def smallest_multiple_perfect_cube := 27000
def multiple_of_30 (n : ℕ) : Prop := n % 30 = 0

-- Problem statement
theorem cameron_list_length :
  ∀ n, 900 ≤ n ∧ n ≤ 27000 ∧ multiple_of_30 n ->
  (871 = (900 - 30 + 1)) :=
sorry

end cameron_list_length_l590_59016


namespace ratio_of_doctors_to_lawyers_l590_59037

/--
Given the average age of a group consisting of doctors and lawyers is 47,
the average age of doctors is 45,
and the average age of lawyers is 55,
prove that the ratio of the number of doctors to the number of lawyers is 4:1.
-/
theorem ratio_of_doctors_to_lawyers
  (d l : ℕ) -- numbers of doctors and lawyers
  (avg_group_age : ℝ := 47)
  (avg_doctors_age : ℝ := 45)
  (avg_lawyers_age : ℝ := 55)
  (h : (45 * d + 55 * l) / (d + l) = 47) :
  d = 4 * l :=
by
  sorry

end ratio_of_doctors_to_lawyers_l590_59037


namespace f_2006_eq_1_l590_59035

noncomputable def f : ℤ → ℤ := sorry
axiom odd_function : ∀ x : ℤ, f (-x) = -f x
axiom period_3 : ∀ x : ℤ, f (3 * (x + 1)) = f (3 * x + 1)
axiom f_at_1 : f 1 = -1

theorem f_2006_eq_1 : f 2006 = 1 := by
  sorry

end f_2006_eq_1_l590_59035


namespace find_certain_number_l590_59060

-- Definitions of conditions from the problem
def greatest_number : ℕ := 10
def divided_1442_by_greatest_number_leaves_remainder := (1442 % greatest_number = 12)
def certain_number_mod_greatest_number (x : ℕ) := (x % greatest_number = 6)

-- Theorem statement
theorem find_certain_number (x : ℕ) (h1 : greatest_number = 10)
  (h2 : 1442 % greatest_number = 12)
  (h3 : certain_number_mod_greatest_number x) : x = 1446 :=
sorry

end find_certain_number_l590_59060


namespace final_percentage_acid_l590_59064

theorem final_percentage_acid (initial_volume : ℝ) (initial_percentage : ℝ)
(removal_volume : ℝ) (final_volume : ℝ) (final_percentage : ℝ) :
  initial_volume = 12 → 
  initial_percentage = 0.40 → 
  removal_volume = 4 →
  final_volume = initial_volume - removal_volume →
  final_percentage = (initial_percentage * initial_volume) / final_volume * 100 →
  final_percentage = 60 := by
  intros h1 h2 h3 h4 h5
  sorry

end final_percentage_acid_l590_59064


namespace probability_at_least_one_multiple_of_4_l590_59012

/-- Definition for the total number of integers in the range -/
def total_numbers : ℕ := 60

/-- Definition for the number of multiples of 4 within the range -/
def multiples_of_4 : ℕ := 15

/-- Probability that a single number chosen is not a multiple of 4 -/
def prob_not_multiple_of_4 : ℚ := (total_numbers - multiples_of_4) / total_numbers

/-- Probability that none of the three chosen numbers is a multiple of 4 -/
def prob_none_multiple_of_4 : ℚ := prob_not_multiple_of_4 ^ 3

/-- Given condition that Linda choose three times -/
axiom linda_chooses_thrice (x y z : ℕ) : 
1 ≤ x ∧ x ≤ 60 ∧ 
1 ≤ y ∧ y ≤ 60 ∧ 
1 ≤ z ∧ z ≤ 60

/-- Theorem stating the desired probability -/
theorem probability_at_least_one_multiple_of_4 : 
1 - prob_none_multiple_of_4 = 37 / 64 := by
  sorry

end probability_at_least_one_multiple_of_4_l590_59012


namespace sequence_formula_l590_59017

open Nat

noncomputable def S : ℕ → ℤ
| n => n^2 - 2 * n + 2

noncomputable def a : ℕ → ℤ
| 0 => 1  -- note that in Lean, sequence indexing starts from 0
| (n+1) => 2*(n+1) - 3

theorem sequence_formula (n : ℕ) : 
  a n = if n = 0 then 1 else 2*n - 3 := by
  sorry

end sequence_formula_l590_59017


namespace select_terms_from_sequence_l590_59063

theorem select_terms_from_sequence (k : ℕ) (hk : k ≥ 3) :
  ∃ (terms : Fin k → ℚ), (∀ i j : Fin k, i < j → (terms j - terms i) = (j.val - i.val) / k!) ∧
  (∀ i : Fin k, terms i ∈ {x : ℚ | ∃ n : ℕ, x = 1 / (n : ℚ)}) :=
by
  sorry

end select_terms_from_sequence_l590_59063


namespace cos_2x_eq_cos_2y_l590_59050

theorem cos_2x_eq_cos_2y (x y : ℝ) 
  (h1 : Real.sin x + Real.cos y = 1) 
  (h2 : Real.cos x + Real.sin y = -1) : 
  Real.cos (2 * x) = Real.cos (2 * y) := by
  sorry

end cos_2x_eq_cos_2y_l590_59050


namespace num_factors_of_2310_with_more_than_three_factors_l590_59023

theorem num_factors_of_2310_with_more_than_three_factors : 
  (∃ n : ℕ, n > 0 ∧ ∀ d : ℕ, d ∣ 2310 → (∀ f : ℕ, f ∣ d → f = 1 ∨ f = d ∨ f ∣ d) → 26 = n) := sorry

end num_factors_of_2310_with_more_than_three_factors_l590_59023


namespace valid_n_values_l590_59011

theorem valid_n_values :
  {n : ℕ | ∀ a : ℕ, a^(n+1) ≡ a [MOD n]} = {1, 2, 6, 42, 1806} :=
sorry

end valid_n_values_l590_59011


namespace sphere_radius_l590_59089

theorem sphere_radius (r_A r_B : ℝ) (h₁ : r_A = 40) (h₂ : (4 * π * r_A^2) / (4 * π * r_B^2) = 16) : r_B = 20 :=
  sorry

end sphere_radius_l590_59089


namespace parametric_equation_solution_l590_59053

noncomputable def solve_parametric_equation (a b : ℝ) (ha2b : a ≠ 2 * b) (ha3b : a ≠ -3 * b) : ℝ :=
  (5 / (a - 2 * b))

theorem parametric_equation_solution (a b x : ℝ) (ha2b : a ≠ 2 * b) (ha3b : a ≠ -3 * b) 
  (h : (a * x - 3) / (b * x + 1) = 2) : 
  x = solve_parametric_equation a b ha2b ha3b :=
sorry

end parametric_equation_solution_l590_59053


namespace tan_half_angle_second_quadrant_l590_59055

variables (θ : ℝ) (k : ℤ)
open Real

theorem tan_half_angle_second_quadrant (h : (π / 2) + 2 * k * π < θ ∧ θ < π + 2 * k * π) : 
  tan (θ / 2) > 1 := 
sorry

end tan_half_angle_second_quadrant_l590_59055


namespace train_speed_in_kmh_l590_59059

def length_of_train : ℝ := 600
def length_of_overbridge : ℝ := 100
def time_to_cross_overbridge : ℝ := 70

theorem train_speed_in_kmh :
  (length_of_train + length_of_overbridge) / time_to_cross_overbridge * 3.6 = 36 := 
by 
  sorry

end train_speed_in_kmh_l590_59059


namespace sum_of_a_and_b_l590_59092

noncomputable def f (x : Real) : Real := (1 + Real.sin (2 * x)) / 2
noncomputable def a : Real := f (Real.log 5)
noncomputable def b : Real := f (Real.log (1 / 5))

theorem sum_of_a_and_b : a + b = 1 := by
  -- proof to be provided
  sorry

end sum_of_a_and_b_l590_59092


namespace complement_intersection_l590_59068

def U : Set ℝ := fun x => True
def A : Set ℝ := fun x => x < 0
def B : Set ℝ := fun x => x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2

theorem complement_intersection (hU : ∀ x : ℝ, U x) :
  ((compl A) ∩ B) = {0, 1, 2} :=
by {
  sorry
}

end complement_intersection_l590_59068


namespace john_total_cost_l590_59033

-- The total cost John incurs to rent a car, buy gas, and drive 320 miles
def total_cost (rental_cost gas_cost_per_gallon cost_per_mile miles driven_gallons : ℝ): ℝ :=
  rental_cost + (gas_cost_per_gallon * driven_gallons) + (cost_per_mile * miles)

theorem john_total_cost :
  let rental_cost := 150
  let gallons := 8
  let gas_cost_per_gallon := 3.50
  let cost_per_mile := 0.50
  let miles := 320
  total_cost rental_cost gas_cost_per_gallon cost_per_mile miles gallons = 338 := 
by
  -- The detailed proof is skipped here
  sorry

end john_total_cost_l590_59033


namespace customer_paid_l590_59013

def cost_price : ℝ := 7999.999999999999
def percentage_markup : ℝ := 0.10
def selling_price (cp : ℝ) (markup : ℝ) := cp + cp * markup

theorem customer_paid :
  selling_price cost_price percentage_markup = 8800 :=
by
  sorry

end customer_paid_l590_59013


namespace crowdfunding_highest_level_backing_l590_59057

-- Definitions according to the conditions
def lowest_level_backing : ℕ := 50
def second_level_backing : ℕ := 10 * lowest_level_backing
def highest_level_backing : ℕ := 100 * lowest_level_backing
def total_raised : ℕ := (2 * highest_level_backing) + (3 * second_level_backing) + (10 * lowest_level_backing)

-- Statement of the problem
theorem crowdfunding_highest_level_backing (h: total_raised = 12000) :
  highest_level_backing = 5000 :=
sorry

end crowdfunding_highest_level_backing_l590_59057


namespace inequality_l590_59054

theorem inequality (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h_sum : a + b + c + d = 1) : 
  b * c * d / (1 - a)^2 + a * c * d / (1 - b)^2 + a * b * d / (1 - c)^2 + a * b * c / (1 - d)^2 ≤ 1 / 9 :=
sorry

end inequality_l590_59054


namespace dolphins_scored_15_l590_59076

theorem dolphins_scored_15 (s d : ℤ) 
  (h1 : s + d = 48) 
  (h2 : s - d = 18) : 
  d = 15 := 
sorry

end dolphins_scored_15_l590_59076


namespace sum_of_final_two_numbers_l590_59094

noncomputable def final_sum (X m n : ℚ) : ℚ :=
  3 * m + 3 * n - 14

theorem sum_of_final_two_numbers (X m n : ℚ) 
  (h1 : m + n = X) :
  final_sum X m n = 3 * X - 14 :=
  sorry

end sum_of_final_two_numbers_l590_59094


namespace meal_cost_per_person_l590_59096

/-
Problem Statement:
Prove that the cost per meal is $3 given the conditions:
- There are 2 adults and 5 children.
- The total bill is $21.
-/

theorem meal_cost_per_person (total_adults : ℕ) (total_children : ℕ) (total_bill : ℝ) 
(total_people : ℕ) (cost_per_meal : ℝ) : 
total_adults = 2 → total_children = 5 → total_bill = 21 → total_people = total_adults + total_children →
cost_per_meal = total_bill / total_people → 
cost_per_meal = 3 :=
by
  intros h1 h2 h3 h4 h5
  simp [h1, h2, h3, h4, h5]
  sorry

end meal_cost_per_person_l590_59096


namespace original_number_is_144_l590_59082

theorem original_number_is_144 (A B C : ℕ) (A_digit : A < 10) (B_digit : B < 10) (C_digit : C < 10)
  (h1 : 100 * A + 10 * B + B = 144)
  (h2 : A * B * B = 10 * A + C)
  (h3 : (10 * A + C) % 10 = C) : 100 * A + 10 * B + B = 144 := 
sorry

end original_number_is_144_l590_59082


namespace oil_to_water_ratio_in_bottle_D_l590_59034

noncomputable def bottle_oil_water_ratio (CA : ℝ) (CB : ℝ) (CC : ℝ) (CD : ℝ) : ℝ :=
  let oil_A := (1 / 2) * CA
  let water_A := (1 / 2) * CA
  let oil_B := (1 / 4) * CB
  let water_B := (1 / 4) * CB
  let total_water_B := CB - oil_B - water_B
  let oil_C := (1 / 3) * CC
  let water_C := 0.4 * CC
  let total_water_C := CC - oil_C - water_C
  let total_capacity_D := CD
  let total_oil_D := oil_A + oil_B + oil_C
  let total_water_D := water_A + total_water_B + water_C + total_water_C
  total_oil_D / total_water_D

theorem oil_to_water_ratio_in_bottle_D (CA : ℝ) :
  let CB := 2 * CA
  let CC := 3 * CA
  let CD := CA + CC
  bottle_oil_water_ratio CA CB CC CD = (2 / 3.7) :=
by 
  sorry

end oil_to_water_ratio_in_bottle_D_l590_59034


namespace add_100ml_water_l590_59056

theorem add_100ml_water 
    (current_volume : ℕ) 
    (current_water_percentage : ℝ) 
    (desired_water_percentage : ℝ) 
    (current_water_volume : ℝ) 
    (x : ℝ) :
    current_volume = 300 →
    current_water_percentage = 0.60 →
    desired_water_percentage = 0.70 →
    current_water_volume = 0.60 * 300 →
    180 + x = 0.70 * (300 + x) →
    x = 100 := 
sorry

end add_100ml_water_l590_59056


namespace terminating_decimals_l590_59079

theorem terminating_decimals (n : ℤ) (h : 1 ≤ n ∧ n ≤ 180) :
  (∃ k : ℤ, k = 20 ∧ ∀ n, 1 ≤ n ∧ n ≤ 180 → (∃ m, m * 180 = n * (2^2 * 5))) := by
  sorry

end terminating_decimals_l590_59079


namespace final_position_l590_59026

structure Position where
  base : ℝ × ℝ
  stem : ℝ × ℝ

def rotate180 (pos : Position) : Position :=
  { base := (-pos.base.1, -pos.base.2),
    stem := (-pos.stem.1, -pos.stem.2) }

def reflectX (pos : Position) : Position :=
  { base := (pos.base.1, -pos.base.2),
    stem := (pos.stem.1, -pos.stem.2) }

def rotateHalfTurn (pos : Position) : Position :=
  { base := (-pos.base.1, -pos.base.2),
    stem := (-pos.stem.1, -pos.stem.2) }

def reflectY (pos : Position) : Position :=
  { base := (-pos.base.1, pos.base.2),
    stem := (-pos.stem.1, pos.stem.2) }

theorem final_position : 
  let initial_pos := Position.mk (1, 0) (0, 1)
  let pos1 := rotate180 initial_pos
  let pos2 := reflectX pos1
  let pos3 := rotateHalfTurn pos2
  let final_pos := reflectY pos3
  final_pos = { base := (-1, 0), stem := (0, -1) } :=
by
  sorry

end final_position_l590_59026


namespace product_N_l590_59087

theorem product_N (A D D1 A1 : ℤ) (N : ℤ) 
  (h1 : D = A - N)
  (h2 : D1 = D + 7)
  (h3 : A1 = A - 2)
  (h4 : |D1 - A1| = 8) : 
  N = 1 → N = 17 → N * 17 = 17 :=
by
  sorry

end product_N_l590_59087


namespace max_bus_capacity_l590_59009

-- Definitions and conditions
def left_side_regular_seats := 12
def left_side_priority_seats := 3
def right_side_regular_seats := 9
def right_side_priority_seats := 2
def right_side_wheelchair_space := 1
def regular_seat_capacity := 3
def priority_seat_capacity := 2
def back_row_seat_capacity := 7
def standing_capacity := 14

-- Definition of total bus capacity
def total_bus_capacity : ℕ :=
  (left_side_regular_seats * regular_seat_capacity) + 
  (left_side_priority_seats * priority_seat_capacity) + 
  (right_side_regular_seats * regular_seat_capacity) + 
  (right_side_priority_seats * priority_seat_capacity) + 
  back_row_seat_capacity + 
  standing_capacity

-- Theorem to prove
theorem max_bus_capacity : total_bus_capacity = 94 := by
  -- skipping the proof
  sorry

end max_bus_capacity_l590_59009


namespace two_x_plus_y_equals_7_l590_59084

noncomputable def proof_problem (x y A : ℝ) : ℝ :=
  if (2 * x + y = A ∧ x + 2 * y = 8 ∧ (x + y) / 3 = 1.6666666666666667) then A else 0

theorem two_x_plus_y_equals_7 (x y : ℝ) : 
  (2 * x + y = proof_problem x y 7) ↔
  (2 * x + y = 7 ∧ x + 2 * y = 8 ∧ (x + y) / 3 = 1.6666666666666667) :=
by sorry

end two_x_plus_y_equals_7_l590_59084


namespace calc_val_l590_59086

theorem calc_val : 
  (3 + 5 + 7) / (2 + 4 + 6) * (4 + 8 + 12) / (1 + 3 + 5) = 10 / 3 :=
by 
  -- Calculation proof
  sorry

end calc_val_l590_59086


namespace members_play_both_l590_59004

-- Define the conditions
variables (N B T neither : ℕ)
variables (B_union_T B_and_T : ℕ)

-- Assume the given conditions
axiom hN : N = 42
axiom hB : B = 20
axiom hT : T = 23
axiom hNeither : neither = 6
axiom hB_union_T : B_union_T = N - neither

-- State the problem: Prove that B_and_T = 7
theorem members_play_both (N B T neither B_union_T B_and_T : ℕ) 
  (hN : N = 42) 
  (hB : B = 20) 
  (hT : T = 23) 
  (hNeither : neither = 6) 
  (hB_union_T : B_union_T = N - neither) 
  (hInclusionExclusion : B_union_T = B + T - B_and_T) :
  B_and_T = 7 := sorry

end members_play_both_l590_59004


namespace ratio_of_roses_l590_59000

theorem ratio_of_roses (total_flowers tulips carnations roses : ℕ) 
  (h1 : total_flowers = 40) 
  (h2 : tulips = 10) 
  (h3 : carnations = 14) 
  (h4 : roses = total_flowers - (tulips + carnations)) :
  roses / total_flowers = 2 / 5 :=
by
  sorry

end ratio_of_roses_l590_59000


namespace fraction_division_problem_l590_59036

theorem fraction_division_problem :
  (-1/42 : ℚ) / (1/6 - 3/14 + 2/3 - 2/7) = -1/14 :=
by
  -- Skipping the proof step as per the instructions
  sorry

end fraction_division_problem_l590_59036


namespace find_k_eq_3_l590_59048

theorem find_k_eq_3 (k : ℝ) (h : k ≠ 0) :
  ∀ x : ℝ, (x^2 - k) * (x + k) = x^3 - k * (x^2 + x + 3) → k = 3 :=
by sorry

end find_k_eq_3_l590_59048


namespace Deepak_age_l590_59019

variable (R D : ℕ)

theorem Deepak_age 
  (h1 : R / D = 4 / 3)
  (h2 : R + 6 = 26) : D = 15 := 
sorry

end Deepak_age_l590_59019


namespace ellipse_intersects_x_axis_at_four_l590_59041

theorem ellipse_intersects_x_axis_at_four
    (f1 f2 : ℝ × ℝ)
    (h1 : f1 = (0, 0))
    (h2 : f2 = (4, 0))
    (h3 : ∃ P : ℝ × ℝ, P = (1, 0) ∧ (dist P f1 + dist P f2 = 4)) :
  ∃ Q : ℝ × ℝ, Q = (4, 0) ∧ (dist Q f1 + dist Q f2 = 4) :=
sorry

end ellipse_intersects_x_axis_at_four_l590_59041


namespace find_x_l590_59043

theorem find_x : ∃ x : ℕ, x + 1 = 5 ∧ x = 4 :=
by
  sorry

end find_x_l590_59043


namespace parabola_x_coordinate_l590_59031

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (p, 0)

theorem parabola_x_coordinate
  (M : ℝ × ℝ)
  (h_parabola : (M.2)^2 = 4 * M.1)
  (h_distance : dist M (parabola_focus 2) = 3) :
  M.1 = 1 :=
by
  sorry

end parabola_x_coordinate_l590_59031


namespace cab_driver_income_day3_l590_59097

theorem cab_driver_income_day3 :
  let income1 := 200
  let income2 := 150
  let income4 := 400
  let income5 := 500
  let avg_income := 400
  let total_income := avg_income * 5 
  total_income - (income1 + income2 + income4 + income5) = 750 := by
  sorry

end cab_driver_income_day3_l590_59097


namespace distinct_integers_no_perfect_square_product_l590_59027

theorem distinct_integers_no_perfect_square_product
  (k : ℕ) (hk : 0 < k) :
  ∀ a b : ℕ, k^2 < a ∧ a < (k+1)^2 → k^2 < b ∧ b < (k+1)^2 → a ≠ b → ¬∃ m : ℕ, a * b = m^2 :=
by sorry

end distinct_integers_no_perfect_square_product_l590_59027


namespace min_positive_period_f_max_value_f_decreasing_intervals_g_l590_59083

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (x + 7 * Real.pi / 4) + Real.cos (x - 3 * Real.pi / 4)

theorem min_positive_period_f : 
  ∃ (p : ℝ), p > 0 ∧ (∀ x : ℝ, f (x + 2*Real.pi) = f x) :=
sorry

theorem max_value_f : 
  ∃ (M : ℝ), (∀ x : ℝ, f x ≤ M) ∧ (∃ x : ℝ, f x = M) ∧ M = 2 :=
sorry

noncomputable def g (x : ℝ) : ℝ := f (-x)

theorem decreasing_intervals_g :
  ∀ (k : ℤ), ∀ x : ℝ, (5 * Real.pi / 4 + 2 * ↑k * Real.pi ≤ x ∧ x ≤ 9 * Real.pi / 4 + 2 * ↑k * Real.pi) →
  ∀ (h : x ≤ Real.pi * 2 * (↑k+1)), g x ≥ g (x + Real.pi) :=
sorry

end min_positive_period_f_max_value_f_decreasing_intervals_g_l590_59083


namespace eddie_age_l590_59039

theorem eddie_age (Becky_age Irene_age Eddie_age : ℕ)
  (h1 : Becky_age * 2 = Irene_age)
  (h2 : Irene_age = 46)
  (h3 : Eddie_age = 4 * Becky_age) :
  Eddie_age = 92 := by
  sorry

end eddie_age_l590_59039


namespace company_profit_is_correct_l590_59072

structure CompanyInfo where
  num_employees : ℕ
  shirts_per_employee_per_day : ℕ
  hours_per_shift : ℕ
  wage_per_hour : ℕ
  bonus_per_shirt : ℕ
  price_per_shirt : ℕ
  nonemployee_expenses_per_day : ℕ

def daily_profit (info : CompanyInfo) : ℤ :=
  let total_shirts_per_day := info.num_employees * info.shirts_per_employee_per_day
  let total_revenue := total_shirts_per_day * info.price_per_shirt
  let daily_wage_per_employee := info.wage_per_hour * info.hours_per_shift
  let total_daily_wage := daily_wage_per_employee * info.num_employees
  let daily_bonus_per_employee := info.bonus_per_shirt * info.shirts_per_employee_per_day
  let total_daily_bonus := daily_bonus_per_employee * info.num_employees
  let total_labor_cost := total_daily_wage + total_daily_bonus
  let total_expenses := total_labor_cost + info.nonemployee_expenses_per_day
  total_revenue - total_expenses

theorem company_profit_is_correct (info : CompanyInfo) (h : 
  info.num_employees = 20 ∧
  info.shirts_per_employee_per_day = 20 ∧
  info.hours_per_shift = 8 ∧
  info.wage_per_hour = 12 ∧
  info.bonus_per_shirt = 5 ∧
  info.price_per_shirt = 35 ∧
  info.nonemployee_expenses_per_day = 1000
) : daily_profit info = 9080 := 
by
  sorry

end company_profit_is_correct_l590_59072


namespace max_product_of_two_integers_sum_2000_l590_59071

theorem max_product_of_two_integers_sum_2000 : 
  ∃ (x : ℕ), (2000 * x - x^2 = 1000000 ∧ 0 ≤ x ∧ x ≤ 2000) := 
by
  sorry

end max_product_of_two_integers_sum_2000_l590_59071


namespace S21_sum_is_4641_l590_59070

-- Define the conditions and the sum of the nth group
def first_number_in_group (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2 + 1

def last_number_in_group (n : ℕ) : ℕ :=
  first_number_in_group n + (n - 1)

def sum_of_group (n : ℕ) : ℕ :=
  n * (first_number_in_group n + last_number_in_group n) / 2

-- The theorem to prove
theorem S21_sum_is_4641 : sum_of_group 21 = 4641 := by
  sorry

end S21_sum_is_4641_l590_59070


namespace difference_of_square_of_non_divisible_by_3_l590_59067

theorem difference_of_square_of_non_divisible_by_3 (n : ℕ) (h : ¬ (n % 3 = 0)) : (n^2 - 1) % 3 = 0 :=
sorry

end difference_of_square_of_non_divisible_by_3_l590_59067


namespace more_cats_than_spinsters_l590_59085

theorem more_cats_than_spinsters :
  ∀ (S C : ℕ), (S = 18) → (2 * C = 9 * S) → (C - S = 63) :=
by
  intros S C hS hRatio
  sorry

end more_cats_than_spinsters_l590_59085


namespace pre_images_of_one_l590_59069

def f (x : ℝ) := x^3 - x + 1

theorem pre_images_of_one : {x : ℝ | f x = 1} = {-1, 0, 1} :=
by {
  sorry
}

end pre_images_of_one_l590_59069


namespace square_side_length_l590_59044

theorem square_side_length (d : ℝ) (h : d = 2 * Real.sqrt 2) : ∃ s : ℝ, s = 2 :=
by 
  sorry

end square_side_length_l590_59044


namespace monotonic_decreasing_interval_l590_59042

noncomputable def xlnx (x : ℝ) : ℝ := x * Real.log x

theorem monotonic_decreasing_interval : 
  ∀ x, (0 < x) ∧ (x < 5) → (Real.log x + 1 < 0) ↔ (0 < x) ∧ (x < 1 / Real.exp 1) := 
by
  sorry

end monotonic_decreasing_interval_l590_59042


namespace parabola_directrix_l590_59030

theorem parabola_directrix {x y : ℝ} (h : y^2 = 6 * x) : x = -3 / 2 := 
sorry

end parabola_directrix_l590_59030


namespace round_robin_teams_l590_59074

theorem round_robin_teams (x : ℕ) (h : x ≠ 0) :
  (x * (x - 1)) / 2 = 15 → ∃ n : ℕ, x = n :=
by
  sorry

end round_robin_teams_l590_59074


namespace ali_peter_fish_ratio_l590_59040

theorem ali_peter_fish_ratio (P J A : ℕ) (h1 : J = P + 1) (h2 : A = 12) (h3 : A + P + J = 25) : A / P = 2 :=
by
  -- Step-by-step simplifications will follow here in the actual proof.
  sorry

end ali_peter_fish_ratio_l590_59040


namespace correct_calculation_l590_59065

theorem correct_calculation (a : ℝ) : a^4 / a = a^3 :=
by {
  sorry
}

end correct_calculation_l590_59065


namespace right_triangle_legs_sum_l590_59028

theorem right_triangle_legs_sum (x : ℕ) (hx1 : x * x + (x + 1) * (x + 1) = 41 * 41) : x + (x + 1) = 59 :=
by sorry

end right_triangle_legs_sum_l590_59028


namespace function_graph_intersection_l590_59088

theorem function_graph_intersection (f : ℝ → ℝ) :
  (∃ y : ℝ, f 1 = y) → (∃! y : ℝ, f 1 = y) :=
by
  sorry

end function_graph_intersection_l590_59088


namespace odd_function_value_l590_59073

noncomputable def f (x : ℝ) : ℝ := sorry -- Placeholder for the function definition

-- Prove that f(-1/2) = -1/2 given the conditions
theorem odd_function_value :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, 0 ≤ x ∧ x < 1 → f x = x) →
  f (-1/2) = -1/2 :=
by
  sorry

end odd_function_value_l590_59073


namespace minimum_employees_needed_l590_59095

def min_new_employees (water_pollution: ℕ) (air_pollution: ℕ) (both: ℕ) : ℕ :=
  119 + 34

theorem minimum_employees_needed : min_new_employees 98 89 34 = 153 := 
  by
  sorry

end minimum_employees_needed_l590_59095


namespace proof_problem_l590_59061

variable (x y : ℝ)

theorem proof_problem :
  ¬ (x^2 + x^2 = x^4) ∧
  ¬ ((x - y)^2 = x^2 - y^2) ∧
  ¬ ((x^2 * y)^3 = x^6 * y) ∧
  ((-x)^2 * x^3 = x^5) :=
by
  sorry

end proof_problem_l590_59061


namespace fifth_grade_total_students_l590_59081

-- Define the conditions given in the problem
def total_boys : ℕ := 350
def total_playing_soccer : ℕ := 250
def percentage_boys_playing_soccer : ℝ := 0.86
def girls_not_playing_soccer : ℕ := 115

-- Define the total number of students
def total_students : ℕ := 500

-- Prove that the total number of students is 500
theorem fifth_grade_total_students 
  (H1 : total_boys = 350) 
  (H2 : total_playing_soccer = 250) 
  (H3 : percentage_boys_playing_soccer = 0.86) 
  (H4 : girls_not_playing_soccer = 115) :
  total_students = 500 := 
sorry

end fifth_grade_total_students_l590_59081


namespace roots_of_unity_polynomial_l590_59091

theorem roots_of_unity_polynomial (c d : ℤ) (z : ℂ) (hz : z^3 = 1) :
  (z^3 + c * z + d = 0) → (z = 1) :=
sorry

end roots_of_unity_polynomial_l590_59091


namespace length_of_AP_l590_59052

noncomputable def square_side_length : ℝ := 8
noncomputable def rect_width : ℝ := 12
noncomputable def rect_height : ℝ := 8

axiom AD_perpendicular_WX : true
axiom shaded_area_half_WXYZ : true

theorem length_of_AP (AP : ℝ) (shaded_area : ℝ)
  (h1 : shaded_area = (rect_width * rect_height) / 2)
  (h2 : shaded_area = (square_side_length - AP) * square_side_length)
  : AP = 2 := by
  sorry

end length_of_AP_l590_59052


namespace molecular_weight_correct_l590_59058

-- Define atomic weights
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

-- Define the number of atoms
def num_N : ℕ := 2
def num_O : ℕ := 3

-- Define the expected molecular weight
def expected_molecular_weight : ℝ := 76.02

-- The theorem to prove
theorem molecular_weight_correct :
  (num_N * atomic_weight_N + num_O * atomic_weight_O) = expected_molecular_weight := 
by
  sorry

end molecular_weight_correct_l590_59058


namespace pumpkin_weight_difference_l590_59046

theorem pumpkin_weight_difference (Brad: ℕ) (Jessica: ℕ) (Betty: ℕ) 
    (h1 : Brad = 54) 
    (h2 : Jessica = Brad / 2) 
    (h3 : Betty = Jessica * 4) 
    : (Betty - Jessica) = 81 := 
by
  sorry

end pumpkin_weight_difference_l590_59046


namespace perfect_squares_l590_59018

theorem perfect_squares (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) : 
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 :=
sorry

end perfect_squares_l590_59018


namespace simplify_expression_l590_59015

theorem simplify_expression (a b c x : ℝ) (h₁ : a ≠ b) (h₂ : a ≠ c) (h₃ : b ≠ c) :
  ( ( (x + a)^4 ) / ( (a - b) * (a - c) ) 
  + ( (x + b)^4 ) / ( (b - a) * (b - c) ) 
  + ( (x + c)^4 ) / ( (c - a) * (c - b) ) ) = a + b + c + 4 * x := 
by
  sorry

end simplify_expression_l590_59015


namespace a_gt_b_l590_59002

theorem a_gt_b (n : ℕ) (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (hn_ge_two : n ≥ 2)
  (ha_eq : a^n = a + 1) (hb_eq : b^(2*n) = b + 3 * a) : a > b :=
by
  sorry

end a_gt_b_l590_59002


namespace inequality_proof_l590_59008

theorem inequality_proof {x y z : ℝ}
  (h1 : x + 2 * y + 4 * z ≥ 3)
  (h2 : y - 3 * x + 2 * z ≥ 5) :
  y - x + 2 * z ≥ 3 :=
by
  sorry

end inequality_proof_l590_59008


namespace new_box_volume_eq_5_76_m3_l590_59025

-- Given conditions:
def original_width_cm := 80
def original_length_cm := 75
def original_height_cm := 120
def conversion_factor_cm3_to_m3 := 1000000

-- New dimensions after doubling
def new_width_cm := 2 * original_width_cm
def new_length_cm := 2 * original_length_cm
def new_height_cm := 2 * original_height_cm

-- Statement of the problem
theorem new_box_volume_eq_5_76_m3 :
  (new_width_cm * new_length_cm * new_height_cm : ℝ) / conversion_factor_cm3_to_m3 = 5.76 := 
  sorry

end new_box_volume_eq_5_76_m3_l590_59025


namespace group_members_count_l590_59045

theorem group_members_count (n: ℕ) (total_paise: ℕ) (condition1: total_paise = 3249) :
  (n * n = total_paise) → n = 57 :=
by
  sorry

end group_members_count_l590_59045


namespace angle_BCA_measure_l590_59078

theorem angle_BCA_measure
  (A B C : Type)
  (angle_ABC : ℝ)
  (angle_BCA : ℝ)
  (angle_BAC : ℝ)
  (h1 : angle_ABC = 90)
  (h2 : angle_BAC = 2 * angle_BCA) :
  angle_BCA = 30 :=
by
  sorry

end angle_BCA_measure_l590_59078


namespace z_is_negative_y_intercept_l590_59075

-- Define the objective function as an assumption or condition
def objective_function (x y z : ℝ) : Prop := z = 3 * x - y

-- Define what we need to prove: z is the negative of the y-intercept 
def negative_y_intercept (x y z : ℝ) : Prop := ∃ m b, (y = m * x + b) ∧ m = 3 ∧ b = -z

-- The theorem we need to prove
theorem z_is_negative_y_intercept (x y z : ℝ) (h : objective_function x y z) : negative_y_intercept x y z :=
  sorry

end z_is_negative_y_intercept_l590_59075


namespace distance_halfway_along_orbit_l590_59032

-- Define the conditions
variables (perihelion aphelion : ℝ) (perihelion_dist : perihelion = 3) (aphelion_dist : aphelion = 15)

-- State the theorem
theorem distance_halfway_along_orbit : 
  ∃ d, d = (perihelion + aphelion) / 2 ∧ d = 9 :=
by
  sorry

end distance_halfway_along_orbit_l590_59032


namespace recurring_decimals_sum_correct_l590_59090

noncomputable def recurring_decimals_sum : ℚ :=
  let x := (2:ℚ) / 3
  let y := (4:ℚ) / 9
  x + y

theorem recurring_decimals_sum_correct :
  recurring_decimals_sum = 10 / 9 := 
  sorry

end recurring_decimals_sum_correct_l590_59090


namespace total_toys_is_correct_l590_59066

-- Define the given conditions
def toy_cars : ℕ := 20
def toy_soldiers : ℕ := 2 * toy_cars
def total_toys : ℕ := toy_cars + toy_soldiers

-- Prove the expected total number of toys
theorem total_toys_is_correct : total_toys = 60 :=
by
  sorry

end total_toys_is_correct_l590_59066
