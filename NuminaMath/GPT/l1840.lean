import Mathlib

namespace total_bill_for_group_is_129_l1840_184019

theorem total_bill_for_group_is_129 :
  let num_adults := 6
  let num_teenagers := 3
  let num_children := 1
  let cost_adult_meal := 9
  let cost_teenager_meal := 7
  let cost_child_meal := 5
  let cost_soda := 2.50
  let num_sodas := 10
  let cost_dessert := 4
  let num_desserts := 3
  let cost_appetizer := 6
  let num_appetizers := 2
  let total_bill := 
    (num_adults * cost_adult_meal) +
    (num_teenagers * cost_teenager_meal) +
    (num_children * cost_child_meal) +
    (num_sodas * cost_soda) +
    (num_desserts * cost_dessert) +
    (num_appetizers * cost_appetizer)
  total_bill = 129 := by
sorry

end total_bill_for_group_is_129_l1840_184019


namespace extreme_values_range_of_a_inequality_of_zeros_l1840_184035

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := -2 * (Real.log x) - a / (x ^ 2) + 1

theorem extreme_values (a : ℝ) (h : a = 1) :
  (∀ x ∈ Set.Icc (1 / 2 : ℝ) 2, f x a ≤ 0) ∧
  (∃ x ∈ Set.Icc (1 / 2 : ℝ) 2, f x a = 0) ∧
  (∀ x ∈ Set.Icc (1 / 2 : ℝ) 2, f x a ≥ -3 + 2 * (Real.log 2)) ∧
  (∃ x ∈ Set.Icc (1 / 2 : ℝ) 2, f x a = -3 + 2 * (Real.log 2)) :=
sorry

theorem range_of_a :
  (∀ a : ℝ, (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ↔ 0 < a ∧ a < 1) :=
sorry

theorem inequality_of_zeros (a : ℝ) (h : 0 < a) (h1 : a < 1) (x1 x2 : ℝ) (hx1 : f x1 a = 0) (hx2 : f x2 a = 0) (hx1x2 : x1 ≠ x2) :
  1 / (x1 ^ 2) + 1 / (x2 ^ 2) > 2 / a :=
sorry

end extreme_values_range_of_a_inequality_of_zeros_l1840_184035


namespace scientific_notation_conversion_l1840_184028

theorem scientific_notation_conversion :
  (6.1 * 10^9 = (6.1 : ℝ) * 10^8) :=
sorry

end scientific_notation_conversion_l1840_184028


namespace S_11_eq_zero_l1840_184040

noncomputable def S (n : ℕ) : ℝ := sorry
variable (a_n : ℕ → ℝ) (d : ℝ)
variable (h1 : ∀ n, a_n (n+1) = a_n n + d) -- common difference d ≠ 0
variable (h2 : S 5 = S 6)

theorem S_11_eq_zero (h_nonzero : d ≠ 0) : S 11 = 0 := by
  sorry

end S_11_eq_zero_l1840_184040


namespace simplify_and_evaluate_expression_l1840_184015

theorem simplify_and_evaluate_expression (a : ℕ) (h : a = 2023) :
  a * (1 - 2 * a) + 2 * (a + 1) * (a - 1) = 2021 :=
by
  rw [h]
  sorry

end simplify_and_evaluate_expression_l1840_184015


namespace mom_tshirts_count_l1840_184088

def packages : ℕ := 71
def tshirts_per_package : ℕ := 6

theorem mom_tshirts_count : packages * tshirts_per_package = 426 := by
  sorry

end mom_tshirts_count_l1840_184088


namespace problem_statement_l1840_184086

noncomputable def sgn (x : ℝ) : ℝ :=
if x < 0 then -1 else if x = 0 then 0 else 1

noncomputable def a : ℝ :=
1 / Real.logb (1 / 4) (1 / 2015) + 1 / Real.logb (1 / 504) (1 / 2015)

def b : ℝ := 2017

theorem problem_statement :
  (a + b + (a - b) * sgn (a - b)) / 2 = 2017 :=
sorry

end problem_statement_l1840_184086


namespace beth_sells_half_of_coins_l1840_184069

theorem beth_sells_half_of_coins (x y : ℕ) (h₁ : x = 125) (h₂ : y = 35) : (x + y) / 2 = 80 :=
by
  sorry

end beth_sells_half_of_coins_l1840_184069


namespace range_of_a_l1840_184038

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 9 ^ x - 2 * 3 ^ x + a - 3 > 0) → a > 4 :=
by
  sorry

end range_of_a_l1840_184038


namespace tangent_lines_to_curve_at_l1840_184043

noncomputable
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

noncomputable
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + (a - 2) * x

noncomputable
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * a * x + (a - 2)

theorem tangent_lines_to_curve_at (a : ℝ) :
  is_even_function (f' a) →
  (∀ x, f a x = - 2 → (2*x + (- f a x) = 0 ∨ 19*x - 4*(- f a x) - 27 = 0)) :=
by
  sorry

end tangent_lines_to_curve_at_l1840_184043


namespace total_amount_paid_correct_l1840_184099

/--
Given:
1. The marked price of each article is $17.5.
2. A discount of 30% was applied to the total marked price of the pair of articles.

Prove:
The total amount paid for the pair of articles is $24.5.
-/
def total_amount_paid (marked_price_each : ℝ) (discount_rate : ℝ) : ℝ :=
  let marked_price_pair := marked_price_each * 2
  let discount := discount_rate * marked_price_pair
  marked_price_pair - discount

theorem total_amount_paid_correct :
  total_amount_paid 17.5 0.30 = 24.5 :=
by
  sorry

end total_amount_paid_correct_l1840_184099


namespace abs_eq_sqrt_five_l1840_184008

theorem abs_eq_sqrt_five (x : ℝ) (h : |x| = Real.sqrt 5) : x = Real.sqrt 5 ∨ x = -Real.sqrt 5 := 
sorry

end abs_eq_sqrt_five_l1840_184008


namespace nontrivial_power_of_nat_l1840_184087

theorem nontrivial_power_of_nat (n : ℕ) :
  (∃ A p : ℕ, 2^n + 1 = A^p ∧ p > 1) → n = 3 :=
by
  sorry

end nontrivial_power_of_nat_l1840_184087


namespace find_a_l1840_184031

theorem find_a (a : ℝ) (h : 2 * a + 3 = -3) : a = -3 := 
by 
  sorry

end find_a_l1840_184031


namespace distance_from_circle_to_line_l1840_184082

def polar_circle (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ
def polar_line (θ : ℝ) : Prop := θ = Real.pi / 6

theorem distance_from_circle_to_line : 
  ∃ d : ℝ, polar_circle ρ θ ∧ polar_line θ → d = Real.sqrt 3 := 
by
  sorry

end distance_from_circle_to_line_l1840_184082


namespace inverse_is_correct_l1840_184010

-- Definitions
def original_proposition (n : ℤ) : Prop := n < 0 → n ^ 2 > 0
def inverse_proposition (n : ℤ) : Prop := n ^ 2 > 0 → n < 0

-- Theorem stating the inverse
theorem inverse_is_correct : 
  (∀ n : ℤ, original_proposition n) → (∀ n : ℤ, inverse_proposition n) :=
by
  sorry

end inverse_is_correct_l1840_184010


namespace sad_girls_count_l1840_184079

variables (total_children happy_children sad_children neither_children : ℕ)
variables (total_boys total_girls happy_boys sad_children total_sad_boys : ℕ)

theorem sad_girls_count :
  total_children = 60 ∧ 
  happy_children = 30 ∧ 
  sad_children = 10 ∧ 
  neither_children = 20 ∧ 
  total_boys = 17 ∧ 
  total_girls = 43 ∧ 
  happy_boys = 6 ∧ 
  neither_boys = 5 ∧ 
  sad_children = total_sad_boys + (sad_children - total_sad_boys) ∧ 
  total_sad_boys = total_boys - happy_boys - neither_boys → 
  (sad_children - total_sad_boys = 4) := 
by
  intros h
  sorry

end sad_girls_count_l1840_184079


namespace last_passenger_probability_l1840_184068

noncomputable def probability_last_passenger_sits_correctly (n : ℕ) : ℝ :=
if n = 0 then 0 else 1 / 2

theorem last_passenger_probability (n : ℕ) :
  (probability_last_passenger_sits_correctly n) = 1 / 2 :=
by {
  sorry
}

end last_passenger_probability_l1840_184068


namespace positive_difference_l1840_184097

theorem positive_difference (x y : ℝ) 
  (h1 : x + y = 30) 
  (h2 : 3 * y - 4 * x = 9) : 
  abs (y - x) = 129 / 7 - (30 - 129 / 7) := 
by {
  sorry
}

end positive_difference_l1840_184097


namespace perfume_price_l1840_184039

variable (P : ℝ)

theorem perfume_price (h_increase : 1.10 * P = P + 0.10 * P)
    (h_decrease : 0.935 * P = 1.10 * P - 0.15 * 1.10 * P)
    (h_final_price : P - 0.935 * P = 78) : P = 1200 := 
by
  sorry

end perfume_price_l1840_184039


namespace flu_infection_equation_l1840_184096

theorem flu_infection_equation (x : ℕ) (h : 1 + x + x^2 = 36) : 1 + x + x^2 = 36 :=
by
  sorry

end flu_infection_equation_l1840_184096


namespace smallest_w_l1840_184063

theorem smallest_w (w : ℕ) (hw : w > 0) (h1 : ∃ k1, 936 * w = 2^5 * k1) (h2 : ∃ k2, 936 * w = 3^3 * k2) (h3 : ∃ k3, 936 * w = 10^2 * k3) : 
  w = 300 :=
by
  sorry

end smallest_w_l1840_184063


namespace fraction_of_full_fare_half_ticket_l1840_184051

theorem fraction_of_full_fare_half_ticket (F R : ℝ) 
  (h1 : F + R = 216) 
  (h2 : F + (1/2)*F + 2*R = 327) : 
  (1/2) = 1/2 :=
by
  sorry

end fraction_of_full_fare_half_ticket_l1840_184051


namespace islander_distances_l1840_184042

theorem islander_distances (A B C D : ℕ) (k1 : A = 1 ∨ A = 2)
  (k2 : B = 2)
  (C_liar : C = 1) (is_knight : C ≠ 1) :
  C = 1 ∨ C = 3 ∨ C = 4 ∧ D = 2 :=
by {
  sorry
}

end islander_distances_l1840_184042


namespace students_know_mothers_birthday_l1840_184081

-- Defining the given conditions
def total_students : ℕ := 40
def A : ℕ := 10
def B : ℕ := 12
def C : ℕ := 22
def D : ℕ := 26

-- Statement to prove
theorem students_know_mothers_birthday : (B + C) = 22 :=
by
  sorry

end students_know_mothers_birthday_l1840_184081


namespace symmetric_point_reflection_y_axis_l1840_184045

theorem symmetric_point_reflection_y_axis (x y : ℝ) (h : (x, y) = (-2, 3)) :
  (-x, y) = (2, 3) :=
sorry

end symmetric_point_reflection_y_axis_l1840_184045


namespace correct_proposition_l1840_184059

-- Definitions
def p (x : ℝ) : Prop := x > 2 → x > 1 ∧ ¬ (x > 1 → x > 2)

def q (a b : ℝ) : Prop := a > b → 1 / a < 1 / b

-- Propositions
def p_and_q (x a b : ℝ) := p x ∧ q a b
def not_p_or_q (x a b : ℝ) := ¬ (p x) ∨ q a b
def p_and_not_q (x a b : ℝ) := p x ∧ ¬ (q a b)
def not_p_and_not_q (x a b : ℝ) := ¬ (p x) ∧ ¬ (q a b)

-- Main theorem
theorem correct_proposition (x a b : ℝ) (h_p : p x) (h_q : ¬ (q a b)) :
  (p_and_q x a b = false) ∧
  (not_p_or_q x a b = false) ∧
  (p_and_not_q x a b = true) ∧
  (not_p_and_not_q x a b = false) :=
by
  sorry

end correct_proposition_l1840_184059


namespace legendre_symbol_two_l1840_184078

theorem legendre_symbol_two (m : ℕ) [Fact (Nat.Prime m)] (hm : Odd m) :
  (legendreSym 2 m) = (-1 : ℤ) ^ ((m^2 - 1) / 8) :=
sorry

end legendre_symbol_two_l1840_184078


namespace youngest_child_age_l1840_184006

theorem youngest_child_age (x : ℝ) (h : x + (x + 1) + (x + 2) + (x + 3) = 12) : x = 1.5 :=
by sorry

end youngest_child_age_l1840_184006


namespace total_travel_options_l1840_184052

theorem total_travel_options (trains_A_to_B : ℕ) (ferries_B_to_C : ℕ) (flights_A_to_C : ℕ) 
  (h1 : trains_A_to_B = 3) (h2 : ferries_B_to_C = 2) (h3 : flights_A_to_C = 2) :
  (trains_A_to_B * ferries_B_to_C + flights_A_to_C = 8) :=
by
  sorry

end total_travel_options_l1840_184052


namespace number_of_blocks_needed_l1840_184007

-- Define the dimensions of the fort
def fort_length : ℕ := 20
def fort_width : ℕ := 15
def fort_height : ℕ := 8

-- Define the thickness of the walls and the floor
def wall_thickness : ℕ := 2
def floor_thickness : ℕ := 1

-- Define the original volume of the fort
def V_original : ℕ := fort_length * fort_width * fort_height

-- Define the interior dimensions of the fort considering the thickness of the walls and floor
def interior_length : ℕ := fort_length - 2 * wall_thickness
def interior_width : ℕ := fort_width - 2 * wall_thickness
def interior_height : ℕ := fort_height - floor_thickness

-- Define the volume of the interior space
def V_interior : ℕ := interior_length * interior_width * interior_height

-- Statement to prove: number of blocks needed equals 1168
theorem number_of_blocks_needed : V_original - V_interior = 1168 := 
by 
  sorry

end number_of_blocks_needed_l1840_184007


namespace profit_condition_maximize_profit_l1840_184065

noncomputable def profit (x : ℕ) : ℕ := 
  (x + 10) * (300 - 10 * x)

theorem profit_condition (x : ℕ) : profit x = 3360 ↔ x = 2 ∨ x = 18 := by
  sorry

theorem maximize_profit : ∃ x, x = 10 ∧ profit x = 4000 := by
  sorry

end profit_condition_maximize_profit_l1840_184065


namespace calculate_savings_l1840_184093

def income : ℕ := 5 * (45000 + 35000 + 7000 + 10000 + 13000)
def expenses : ℕ := 5 * (30000 + 10000 + 5000 + 4500 + 9000)
def initial_savings : ℕ := 849400
def total_savings : ℕ := initial_savings + income - expenses

theorem calculate_savings : total_savings = 1106900 := by
  -- proof to be filled in
  sorry

end calculate_savings_l1840_184093


namespace intersection_A_B_l1840_184049

-- Conditions
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := { y | ∃ x ∈ A, y = 3 * x - 2 }

-- Question and proof statement
theorem intersection_A_B :
  A ∩ B = {1, 4} := by
  sorry

end intersection_A_B_l1840_184049


namespace birthday_candles_l1840_184076

theorem birthday_candles :
  ∀ (candles_Ambika : ℕ) (candles_Aniyah : ℕ),
  candles_Ambika = 4 →
  candles_Aniyah = 6 * candles_Ambika →
  (candles_Ambika + candles_Aniyah) / 2 = 14 :=
by
  intro candles_Ambika candles_Aniyah h1 h2
  rw [h1, h2]
  sorry

end birthday_candles_l1840_184076


namespace mass_percentage_of_calcium_in_calcium_oxide_l1840_184023

theorem mass_percentage_of_calcium_in_calcium_oxide
  (Ca_molar_mass : ℝ)
  (O_molar_mass : ℝ)
  (Ca_mass : Ca_molar_mass = 40.08)
  (O_mass : O_molar_mass = 16.00) :
  ((Ca_molar_mass / (Ca_molar_mass + O_molar_mass)) * 100) = 71.45 :=
by
  sorry

end mass_percentage_of_calcium_in_calcium_oxide_l1840_184023


namespace sum_series_1_to_60_l1840_184047

-- Define what it means to be the sum of the first n natural numbers
def sum_n (n : Nat) : Nat := n * (n + 1) / 2

theorem sum_series_1_to_60 : sum_n 60 = 1830 :=
by
  sorry

end sum_series_1_to_60_l1840_184047


namespace pears_left_l1840_184044

theorem pears_left (keith_initial : ℕ) (keith_given : ℕ) (mike_initial : ℕ) 
  (hk : keith_initial = 47) (hg : keith_given = 46) (hm : mike_initial = 12) :
  (keith_initial - keith_given) + mike_initial = 13 := by
  sorry

end pears_left_l1840_184044


namespace cows_number_l1840_184061

theorem cows_number (D C : ℕ) (L H : ℕ) 
  (h1 : L = 2 * D + 4 * C)
  (h2 : H = D + C)
  (h3 : L = 2 * H + 12) 
  : C = 6 := 
by
  sorry

end cows_number_l1840_184061


namespace perpendicular_vectors_l1840_184098

/-- Given vectors a and b which are perpendicular, find the value of m -/
theorem perpendicular_vectors (m : ℝ) (a b : ℝ × ℝ)
  (h1 : a = (2 * m, 1))
  (h2 : b = (1, m - 3))
  (h3 : (a.1 * b.1 + a.2 * b.2) = 0) : m = 1 :=
by
  sorry

end perpendicular_vectors_l1840_184098


namespace rectangle_maximized_area_side_length_l1840_184012

theorem rectangle_maximized_area_side_length
  (x y : ℝ)
  (h_perimeter : 2 * x + 2 * y = 40)
  (h_max_area : x * y = 100) :
  x = 10 :=
by
  sorry

end rectangle_maximized_area_side_length_l1840_184012


namespace Tim_weekly_water_intake_l1840_184030

variable (daily_bottle_intake : ℚ)
variable (additional_intake : ℚ)
variable (quart_to_ounces : ℚ)
variable (days_in_week : ℕ := 7)

theorem Tim_weekly_water_intake (H1 : daily_bottle_intake = 2 * 1.5)
                              (H2 : additional_intake = 20)
                              (H3 : quart_to_ounces = 32) :
  (daily_bottle_intake * quart_to_ounces + additional_intake) * days_in_week = 812 := by
  sorry

end Tim_weekly_water_intake_l1840_184030


namespace vinegar_evaporation_rate_l1840_184017

def percentage_vinegar_evaporates_each_year (x : ℕ) : Prop :=
  let initial_vinegar : ℕ := 100
  let vinegar_left_after_first_year : ℕ := initial_vinegar - x
  let vinegar_left_after_two_years : ℕ := vinegar_left_after_first_year * (100 - x) / 100
  vinegar_left_after_two_years = 64

theorem vinegar_evaporation_rate :
  ∃ x : ℕ, percentage_vinegar_evaporates_each_year x ∧ x = 20 :=
by
  sorry

end vinegar_evaporation_rate_l1840_184017


namespace scheduling_competitions_l1840_184056

-- Define the problem conditions
def scheduling_conditions (gyms : ℕ) (sports : ℕ) (max_sports_per_gym : ℕ) : Prop :=
  gyms = 4 ∧ sports = 3 ∧ max_sports_per_gym = 2

-- Define the main statement
theorem scheduling_competitions :
  scheduling_conditions 4 3 2 →
  (number_of_arrangements = 60) :=
by
  sorry

end scheduling_competitions_l1840_184056


namespace quadratic_roots_expression_l1840_184036

theorem quadratic_roots_expression {m n : ℝ}
  (h₁ : m^2 + m - 12 = 0)
  (h₂ : n^2 + n - 12 = 0)
  (h₃ : m + n = -1) :
  m^2 + 2 * m + n = 11 :=
by {
  sorry
}

end quadratic_roots_expression_l1840_184036


namespace tagged_fish_in_second_catch_l1840_184013

theorem tagged_fish_in_second_catch :
  ∀ (T : ℕ),
    (40 > 0) →
    (800 > 0) →
    (T / 40 = 40 / 800) →
    T = 2 := 
by
  intros T h1 h2 h3
  sorry

end tagged_fish_in_second_catch_l1840_184013


namespace common_ratio_of_geometric_sequence_l1840_184050

theorem common_ratio_of_geometric_sequence (a_1 q : ℝ) (hq : q ≠ 1) 
  (S : ℕ → ℝ) (hS: ∀ n, S n = a_1 * (1 - q^n) / (1 - q))
  (arithmetic_seq : 2 * S 7 = S 8 + S 9) :
  q = -2 :=
by sorry

end common_ratio_of_geometric_sequence_l1840_184050


namespace pizza_consumption_order_l1840_184020

theorem pizza_consumption_order :
  let total_slices := 168
  let alex_slices := (1/6) * total_slices
  let beth_slices := (2/7) * total_slices
  let cyril_slices := (1/3) * total_slices
  let eve_slices_initial := (1/8) * total_slices
  let dan_slices_initial := total_slices - (alex_slices + beth_slices + cyril_slices + eve_slices_initial)
  let eve_slices := eve_slices_initial + 2
  let dan_slices := dan_slices_initial - 2
  (cyril_slices > beth_slices ∧ beth_slices > eve_slices ∧ eve_slices > alex_slices ∧ alex_slices > dan_slices) :=
  sorry

end pizza_consumption_order_l1840_184020


namespace case_D_has_two_solutions_l1840_184005

-- Definitions for the conditions of each case
structure CaseA :=
(b : ℝ) (A : ℝ) (B : ℝ)

structure CaseB :=
(a : ℝ) (c : ℝ) (B : ℝ)

structure CaseC :=
(a : ℝ) (b : ℝ) (A : ℝ)

structure CaseD :=
(a : ℝ) (b : ℝ) (A : ℝ)

-- Setting the values based on the given conditions
def caseA := CaseA.mk 10 45 70
def caseB := CaseB.mk 60 48 100
def caseC := CaseC.mk 14 16 45
def caseD := CaseD.mk 7 5 80

-- Define a function that checks if a case has two solutions
def has_two_solutions (a b c : ℝ) (A B : ℝ) : Prop := sorry

-- The theorem to prove that out of the given cases, only Case D has two solutions
theorem case_D_has_two_solutions :
  has_two_solutions caseA.b caseB.B caseC.a caseC.b caseC.A = false →
  has_two_solutions caseB.a caseB.c caseB.B caseC.b caseC.A = false →
  has_two_solutions caseC.a caseC.b caseC.A caseD.a caseD.b = false →
  has_two_solutions caseD.a caseD.b caseD.A caseA.b caseA.A = true :=
sorry

end case_D_has_two_solutions_l1840_184005


namespace num_turtles_on_sand_l1840_184092

def num_turtles_total : ℕ := 42
def num_turtles_swept : ℕ := num_turtles_total / 3
def num_turtles_sand : ℕ := num_turtles_total - num_turtles_swept

theorem num_turtles_on_sand : num_turtles_sand = 28 := by
  sorry

end num_turtles_on_sand_l1840_184092


namespace combine_like_terms_l1840_184055

theorem combine_like_terms (x y : ℝ) : 3 * x * y - 6 * x * y + (-2) * x * y = -5 * x * y := by
  sorry

end combine_like_terms_l1840_184055


namespace find_percentage_of_alcohol_in_second_solution_l1840_184091

def alcohol_content_second_solution (V2: ℕ) (p1 p2 p_final: ℕ) (V1 V_final: ℕ) : ℕ :=
  ((V_final * p_final) - (V1 * p1)) * 100 / V2

def percentage_correct : Prop :=
  alcohol_content_second_solution 125 20 12 15 75 200 = 12

theorem find_percentage_of_alcohol_in_second_solution : percentage_correct :=
by
  sorry

end find_percentage_of_alcohol_in_second_solution_l1840_184091


namespace fewest_students_possible_l1840_184027

theorem fewest_students_possible :
  ∃ n : ℕ, n ≡ 2 [MOD 5] ∧ n ≡ 4 [MOD 6] ∧ n ≡ 6 [MOD 8] ∧ n = 22 :=
sorry

end fewest_students_possible_l1840_184027


namespace question_correctness_l1840_184025

theorem question_correctness (a b : ℝ) (h : a < b) : a - 1 < b - 1 :=
by sorry

end question_correctness_l1840_184025


namespace intersection_eq_l1840_184060

def setA : Set ℝ := {x | -1 < x ∧ x < 1}
def setB : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem intersection_eq : setA ∩ setB = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_eq_l1840_184060


namespace cube_side_length_l1840_184089

theorem cube_side_length (s : ℝ) (h : s^3 = 6 * s^2) (hs : s ≠ 0) : s = 6 :=
by sorry

end cube_side_length_l1840_184089


namespace units_of_Product_C_sold_l1840_184073

-- Definitions of commission rates
def commission_rate_A : ℝ := 0.05
def commission_rate_B : ℝ := 0.07
def commission_rate_C : ℝ := 0.10

-- Definitions of revenues per unit
def revenue_A : ℝ := 1500
def revenue_B : ℝ := 2000
def revenue_C : ℝ := 3500

-- Definition of units sold
def units_A : ℕ := 5
def units_B : ℕ := 3

-- Commission calculations for Product A and B
def commission_A : ℝ := commission_rate_A * revenue_A * units_A
def commission_B : ℝ := commission_rate_B * revenue_B * units_B

-- Previous average commission and new average commission
def previous_avg_commission : ℝ := 100
def new_avg_commission : ℝ := 250

-- The main proof statement
theorem units_of_Product_C_sold (x : ℝ) (h1 : new_avg_commission = previous_avg_commission + 150)
  (h2 : total_units = units_A + units_B + x)
  (h3 : total_new_commission = commission_A + commission_B + (commission_rate_C * revenue_C * x))
  : x = 12 :=
by
  sorry

end units_of_Product_C_sold_l1840_184073


namespace conor_total_vegetables_l1840_184058

-- Definitions for each day of the week
def vegetables_per_day_mon_wed : Nat := 12 + 9 + 8 + 15 + 7
def vegetables_per_day_thu_sat : Nat := 7 + 5 + 4 + 10 + 4
def total_vegetables : Nat := 3 * vegetables_per_day_mon_wed + 3 * vegetables_per_day_thu_sat

-- Lean statement for the proof problem
theorem conor_total_vegetables : total_vegetables = 243 := by
  sorry

end conor_total_vegetables_l1840_184058


namespace remaining_money_l1840_184004

def potato_cost : ℕ := 6 * 2
def tomato_cost : ℕ := 9 * 3
def cucumber_cost : ℕ := 5 * 4
def banana_cost : ℕ := 3 * 5
def total_cost : ℕ := potato_cost + tomato_cost + cucumber_cost + banana_cost
def initial_money : ℕ := 500

theorem remaining_money : initial_money - total_cost = 426 :=
by
  sorry

end remaining_money_l1840_184004


namespace average_sales_l1840_184002

/-- The sales for the first five months -/
def sales_first_five_months := [5435, 5927, 5855, 6230, 5562]

/-- The sale for the sixth month -/
def sale_sixth_month := 3991

/-- The correct average sale to be achieved -/
def correct_average_sale := 5500

theorem average_sales :
  (sales_first_five_months.sum + sale_sixth_month) / 6 = correct_average_sale :=
by
  sorry

end average_sales_l1840_184002


namespace computation_one_computation_two_l1840_184001

-- Proof problem (1)
theorem computation_one :
  (-2)^3 + |(-3)| - Real.tan (Real.pi / 4) = -6 := by
  sorry

-- Proof problem (2)
theorem computation_two (a : ℝ) :
  (a + 2)^2 - a * (a - 4) = 8 * a + 4 := by
  sorry

end computation_one_computation_two_l1840_184001


namespace max_p_plus_q_l1840_184014

theorem max_p_plus_q (p q : ℝ) (h : ∀ x : ℝ, |x| ≤ 1 → 2 * p * x^2 + q * x - p + 1 ≥ 0) : p + q ≤ 2 :=
sorry

end max_p_plus_q_l1840_184014


namespace Willey_Farm_Available_Capital_l1840_184011

theorem Willey_Farm_Available_Capital 
  (total_acres : ℕ)
  (cost_per_acre_corn : ℕ)
  (cost_per_acre_wheat : ℕ)
  (acres_wheat : ℕ)
  (available_capital : ℕ) :
  total_acres = 4500 →
  cost_per_acre_corn = 42 →
  cost_per_acre_wheat = 35 →
  acres_wheat = 3400 →
  available_capital = (acres_wheat * cost_per_acre_wheat) + 
                      ((total_acres - acres_wheat) * cost_per_acre_corn) →
  available_capital = 165200 := sorry

end Willey_Farm_Available_Capital_l1840_184011


namespace local_minimum_interval_l1840_184064

-- Definitions of the function and its derivative
def y (x a : ℝ) : ℝ := x^3 - 2 * a * x + a
def y_prime (x a : ℝ) : ℝ := 3 * x^2 - 2 * a

-- The proof problem statement
theorem local_minimum_interval (a : ℝ) : 
  (0 < a ∧ a < 3 / 2) ↔ ∃ (x : ℝ), (0 < x ∧ x < 1) ∧ y_prime x a = 0 :=
sorry

end local_minimum_interval_l1840_184064


namespace number_of_houses_around_square_l1840_184085

namespace HouseCounting

-- Definitions for the conditions
def M (k : ℕ) : ℕ := k
def J (k : ℕ) : ℕ := k

-- The main theorem stating the solution
theorem number_of_houses_around_square (n : ℕ)
  (h1 : M 5 % n = J 12 % n)
  (h2 : J 5 % n = M 30 % n) : n = 32 :=
sorry

end HouseCounting

end number_of_houses_around_square_l1840_184085


namespace claudia_total_earnings_l1840_184034

-- Definition of the problem conditions
def class_fee : ℕ := 10
def kids_saturday : ℕ := 20
def kids_sunday : ℕ := kids_saturday / 2

-- Theorem stating that Claudia makes $300.00 for the weekend
theorem claudia_total_earnings : (kids_saturday * class_fee) + (kids_sunday * class_fee) = 300 := 
by
  sorry

end claudia_total_earnings_l1840_184034


namespace existence_of_E_l1840_184018

def ellipse_eq (x y : ℝ) : Prop := (x^2 / 6) + (y^2 / 2) = 1

def point_on_x_axis (E : ℝ × ℝ) : Prop := E.snd = 0

def ea_dot_eb_constant (E A B : ℝ × ℝ) : ℝ :=
  let ea := (A.fst - E.fst, A.snd)
  let eb := (B.fst - E.fst, B.snd)
  ea.fst * eb.fst + ea.snd * eb.snd

noncomputable def E : ℝ × ℝ := (7/3, 0)

noncomputable def const_value : ℝ := (-5/9)

theorem existence_of_E :
  (∃ E, point_on_x_axis E ∧
        (∀ A B, ellipse_eq A.fst A.snd ∧ ellipse_eq B.fst B.snd →
                  ea_dot_eb_constant E A B = const_value)) :=
  sorry

end existence_of_E_l1840_184018


namespace infinite_integer_and_noninteger_terms_l1840_184032

theorem infinite_integer_and_noninteger_terms (m : Nat) (h_m : m > 0) :
  ∃ (infinite_int_terms : Nat → Prop) (infinite_nonint_terms : Nat → Prop),
  (∀ n, ∃ k, infinite_int_terms k ∧ ∀ k, infinite_int_terms k → ∃ N, k = n + N + 1) ∧
  (∀ n, ∃ k, infinite_nonint_terms k ∧ ∀ k, infinite_nonint_terms k → ∃ N, k = n + N + 1) :=
sorry

end infinite_integer_and_noninteger_terms_l1840_184032


namespace area_isosceles_right_triangle_l1840_184048

theorem area_isosceles_right_triangle 
( a : ℝ × ℝ )
( b : ℝ × ℝ )
( h_a : a = (Real.cos (2 / 3 * Real.pi), Real.sin (2 / 3 * Real.pi)) )
( is_isosceles_right_triangle : (a + b).fst * (a - b).fst + (a + b).snd * (a - b).snd = 0 
                                ∧ (a + b).fst * (a + b).fst + (a + b).snd * (a + b).snd 
                                = (a - b).fst * (a - b).fst + (a - b).snd * (a - b).snd ):
  1 / 2 * Real.sqrt ((1 - 1 / 2)^2 + (Real.sqrt 3 / 2 - -1 / 2)^2 )
 * Real.sqrt ((1 - -1 / 2)^2 + (Real.sqrt 3 / 2 - -1 / 2 )^2 ) = 1 :=
by
  sorry

end area_isosceles_right_triangle_l1840_184048


namespace max_smaller_rectangles_l1840_184033

theorem max_smaller_rectangles (a : ℕ) (d : ℕ) (n : ℕ) 
    (ha : a = 100) (hd : d = 2) (hn : n = 50) : 
    n + 1 * (n + 1) = 2601 :=
by
  rw [hn]
  norm_num
  sorry

end max_smaller_rectangles_l1840_184033


namespace hiking_hours_l1840_184000

theorem hiking_hours
  (violet_water_per_hour : ℕ := 800)
  (dog_water_per_hour : ℕ := 400)
  (total_water : ℕ := 4800) :
  (total_water / (violet_water_per_hour + dog_water_per_hour) = 4) :=
by
  sorry

end hiking_hours_l1840_184000


namespace greatest_air_conditioning_but_no_racing_stripes_l1840_184026

variable (total_cars : ℕ) (no_air_conditioning_cars : ℕ) (at_least_racing_stripes_cars : ℕ)
variable (total_cars_eq : total_cars = 100)
variable (no_air_conditioning_cars_eq : no_air_conditioning_cars = 37)
variable (at_least_racing_stripes_cars_ge : at_least_racing_stripes_cars ≥ 51)

theorem greatest_air_conditioning_but_no_racing_stripes
  (total_cars_eq : total_cars = 100)
  (no_air_conditioning_cars_eq : no_air_conditioning_cars = 37)
  (at_least_racing_stripes_cars_ge : at_least_racing_stripes_cars ≥ 51) :
  ∃ max_air_conditioning_no_racing_stripes : ℕ, max_air_conditioning_no_racing_stripes = 12 :=
by
  sorry

end greatest_air_conditioning_but_no_racing_stripes_l1840_184026


namespace blithe_initial_toys_l1840_184094

-- Define the conditions as given in the problem
def lost_toys : ℤ := 6
def found_toys : ℤ := 9
def final_toys : ℤ := 43

-- Define the problem statement to prove the initial number of toys
theorem blithe_initial_toys (T : ℤ) (h : T - lost_toys + found_toys = final_toys) : T = 40 :=
sorry

end blithe_initial_toys_l1840_184094


namespace find_x_in_interval_l1840_184090

theorem find_x_in_interval (x : ℝ) 
  (h₁ : 4 ≤ (x + 1) / (3 * x - 7)) 
  (h₂ : (x + 1) / (3 * x - 7) < 9) : 
  x ∈ Set.Ioc (32 / 13) (29 / 11) := 
sorry

end find_x_in_interval_l1840_184090


namespace combination_30_2_l1840_184024

theorem combination_30_2 : Nat.choose 30 2 = 435 := by
  sorry

end combination_30_2_l1840_184024


namespace solve_system_of_equations_l1840_184083

theorem solve_system_of_equations (u v w : ℝ) (h₀ : u ≠ 0) (h₁ : v ≠ 0) (h₂ : w ≠ 0) :
  (3 / (u * v) + 15 / (v * w) = 2) ∧
  (15 / (v * w) + 5 / (w * u) = 2) ∧
  (5 / (w * u) + 3 / (u * v) = 2) →
  (u = 1 ∧ v = 3 ∧ w = 5) ∨
  (u = -1 ∧ v = -3 ∧ w = -5) :=
by
  sorry

end solve_system_of_equations_l1840_184083


namespace cube_bug_probability_l1840_184067

theorem cube_bug_probability :
  ∃ n : ℕ, (∃ p : ℚ, p = 547/2187) ∧ (p = n/6561) ∧ n = 1641 :=
by
  sorry

end cube_bug_probability_l1840_184067


namespace bowling_ball_weight_l1840_184053

theorem bowling_ball_weight (b c : ℝ) 
  (h1 : 5 * b = 4 * c) 
  (h2 : 2 * c = 80) : 
  b = 32 :=
by
  sorry

end bowling_ball_weight_l1840_184053


namespace fourth_number_of_expression_l1840_184054

theorem fourth_number_of_expression (x : ℝ) (h : 0.3 * 0.8 + 0.1 * x = 0.29) : x = 0.5 :=
by
  sorry

end fourth_number_of_expression_l1840_184054


namespace value_of_expression_l1840_184074

theorem value_of_expression (x y z : ℝ) (hz : z ≠ 0) 
    (h1 : 2 * x - 3 * y - z = 0) 
    (h2 : x + 3 * y - 14 * z = 0) : 
    (x^2 + 3 * x * y) / (y^2 + z^2) = 7 := 
by 
  sorry

end value_of_expression_l1840_184074


namespace distance_between_points_l1840_184016

theorem distance_between_points :
  let A : ℝ × ℝ × ℝ := (1, -2, 3)
  let B : ℝ × ℝ × ℝ := (1, 2, 3)
  let C : ℝ × ℝ × ℝ := (1, 2, -3)
  dist B C = 6 :=
by
  sorry

end distance_between_points_l1840_184016


namespace quadratic_real_roots_l1840_184009

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_real_roots (k : ℝ) :
  discriminant (k - 1) 4 2 ≥ 0 ↔ k ≤ 3 ∧ k ≠ 1 :=
by
  sorry

end quadratic_real_roots_l1840_184009


namespace conic_section_type_l1840_184080

theorem conic_section_type (x y : ℝ) : 
  9 * x^2 - 36 * y^2 = 36 → 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1) :=
by
  sorry

end conic_section_type_l1840_184080


namespace sarah_friends_apples_l1840_184041

-- Definitions of initial conditions
def initial_apples : ℕ := 25
def left_apples : ℕ := 3
def apples_given_teachers : ℕ := 16
def apples_eaten : ℕ := 1

-- Theorem that states the number of friends who received apples
theorem sarah_friends_apples :
  (initial_apples - left_apples - apples_given_teachers - apples_eaten = 5) :=
by
  sorry

end sarah_friends_apples_l1840_184041


namespace digit_sum_square_l1840_184022

theorem digit_sum_square (n : ℕ) (hn : 0 < n) :
  let A := (4 * (10 ^ (2 * n) - 1)) / 9
  let B := (8 * (10 ^ n - 1)) / 9
  ∃ k : ℕ, A + 2 * B + 4 = k ^ 2 := 
by
  sorry

end digit_sum_square_l1840_184022


namespace andy_remaining_demerits_l1840_184077

-- Definitions based on conditions
def max_demerits : ℕ := 50
def demerits_per_late_instance : ℕ := 2
def late_instances : ℕ := 6
def joke_demerits : ℕ := 15

-- Calculation of total demerits for the month
def total_demerits : ℕ := (demerits_per_late_instance * late_instances) + joke_demerits

-- Proof statement: Andy can receive 23 more demerits without being fired
theorem andy_remaining_demerits : max_demerits - total_demerits = 23 :=
by
  -- Placeholder for proof
  sorry

end andy_remaining_demerits_l1840_184077


namespace karen_nuts_l1840_184003

/-- Karen added 0.25 cup of walnuts to a batch of trail mix.
Later, she added 0.25 cup of almonds.
In all, Karen put 0.5 cups of nuts in the trail mix. -/
theorem karen_nuts (walnuts almonds : ℝ) 
  (h_walnuts : walnuts = 0.25) 
  (h_almonds : almonds = 0.25) : 
  walnuts + almonds = 0.5 := 
by
  sorry

end karen_nuts_l1840_184003


namespace coincide_green_square_pairs_l1840_184075

structure Figure :=
  (green_squares : ℕ)
  (red_triangles : ℕ)
  (blue_triangles : ℕ)

theorem coincide_green_square_pairs (f : Figure) (hs : f.green_squares = 4)
  (rt : f.red_triangles = 3) (bt : f.blue_triangles = 6)
  (gs_coincide : ∀ n, n ≤ f.green_squares ⟶ n = f.green_squares) 
  (rt_coincide : ∃ n, n = 2) (bt_coincide : ∃ n, n = 2) 
  (red_blue_pairs : ∃ n, n = 3) : 
  ∃ pairs, pairs = 4 :=
by 
  sorry

end coincide_green_square_pairs_l1840_184075


namespace find_natural_number_n_l1840_184062

def is_terminating_decimal (x : ℚ) : Prop :=
  ∃ (a b : ℕ), x = (a / b) ∧ (∃ (m n : ℕ), b = 2 ^ m * 5 ^ n)

theorem find_natural_number_n (n : ℕ) (h₁ : is_terminating_decimal (1 / n)) (h₂ : is_terminating_decimal (1 / (n + 1))) : n = 4 :=
by sorry

end find_natural_number_n_l1840_184062


namespace evaluate_operations_l1840_184046

def spadesuit (x y : ℝ) := (x + y) * (x - y)
def heartsuit (x y : ℝ) := x ^ 2 - y ^ 2

theorem evaluate_operations : spadesuit 5 (heartsuit 3 2) = 0 :=
by
  sorry

end evaluate_operations_l1840_184046


namespace find_initial_period_l1840_184071

theorem find_initial_period (P : ℝ) (T : ℝ) 
  (h1 : 1680 = (P * 4 * T) / 100)
  (h2 : 1680 = (P * 5 * 4) / 100) 
  : T = 5 := 
by 
  sorry

end find_initial_period_l1840_184071


namespace jacks_paycheck_l1840_184095

theorem jacks_paycheck (P : ℝ) (h1 : 0.2 * 0.8 * P = 20) : P = 125 :=
sorry

end jacks_paycheck_l1840_184095


namespace simplify_1_simplify_2_l1840_184021

theorem simplify_1 (a b : ℤ) : 2 * a - (a + b) = a - b :=
by
  sorry

theorem simplify_2 (x y : ℤ) : (x^2 - 2 * y^2) - 2 * (3 * y^2 - 2 * x^2) = 5 * x^2 - 8 * y^2 :=
by
  sorry

end simplify_1_simplify_2_l1840_184021


namespace parabola_line_intersect_at_one_point_l1840_184057

theorem parabola_line_intersect_at_one_point :
  ∃ a : ℝ, (∀ x : ℝ, (ax^2 + 5 * x + 2 = -2 * x + 1)) ↔ a = 49 / 4 :=
by sorry

end parabola_line_intersect_at_one_point_l1840_184057


namespace katerina_weight_correct_l1840_184029

-- We define the conditions
def total_weight : ℕ := 95
def alexa_weight : ℕ := 46

-- Define the proposition to prove: Katerina's weight is the total weight minus Alexa's weight, which should be 49.
theorem katerina_weight_correct : (total_weight - alexa_weight = 49) :=
by
  -- We use sorry to skip the proof.
  sorry

end katerina_weight_correct_l1840_184029


namespace numbers_whose_triples_plus_1_are_primes_l1840_184084

def is_prime (n : ℕ) : Prop := Nat.Prime n

def in_prime_range (n : ℕ) : Prop := 
  is_prime n ∧ 70 ≤ n ∧ n ≤ 110

def transformed_by_3_and_1 (x : ℕ) : ℕ := 3 * x + 1

theorem numbers_whose_triples_plus_1_are_primes :
  { x : ℕ | in_prime_range (transformed_by_3_and_1 x) } = {24, 26, 32, 34, 36} :=
by
  sorry

end numbers_whose_triples_plus_1_are_primes_l1840_184084


namespace problem1_problem2_problem3_problem4_problem5_problem6_problem7_problem8_l1840_184070

theorem problem1 : 0 - (-22) = 22 := 
by 
  sorry

theorem problem2 : 8.5 - (-1.5) = 10 := 
by 
  sorry

theorem problem3 : (-13 : ℚ) - (4/7) - (-13 : ℚ) - (5/7) = 1/7 := 
by 
  sorry

theorem problem4 : (-1/2 : ℚ) - (1/4 : ℚ) = -3/4 := 
by 
  sorry

theorem problem5 : -51 + 12 + (-7) + (-11) + 36 = -21 := 
by 
  sorry

theorem problem6 : (5/6 : ℚ) + (-2/3) + 1 + (1/6) + (-1/3) = 1 := 
by 
  sorry

theorem problem7 : -13 + (-7) - 20 - (-40) + 16 = 16 := 
by 
  sorry

theorem problem8 : 4.7 - (-8.9) - 7.5 + (-6) = 0.1 := 
by 
  sorry

end problem1_problem2_problem3_problem4_problem5_problem6_problem7_problem8_l1840_184070


namespace joan_payment_l1840_184072

theorem joan_payment (cat_toy_cost cage_cost change_received : ℝ) 
  (h1 : cat_toy_cost = 8.77) 
  (h2 : cage_cost = 10.97) 
  (h3 : change_received = 0.26) : 
  cat_toy_cost + cage_cost - change_received = 19.48 := 
by 
  sorry

end joan_payment_l1840_184072


namespace am_gm_hm_inequality_l1840_184066

theorem am_gm_hm_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a ≠ b) (h5 : b ≠ c) (h6 : a ≠ c) : 
  (a + b + c) / 3 > (a * b * c) ^ (1 / 3) ∧ (a * b * c) ^ (1 / 3) > 3 * a * b * c / (a * b + b * c + c * a) :=
by
  sorry

end am_gm_hm_inequality_l1840_184066


namespace P_finishes_in_15_minutes_more_l1840_184037

variable (P Q : ℝ)

def rate_p := 1 / 4
def rate_q := 1 / 15
def time_together := 3
def total_job := 1

theorem P_finishes_in_15_minutes_more :
  let combined_rate := rate_p + rate_q
  let completed_job_in_3_hours := combined_rate * time_together
  let remaining_job := total_job - completed_job_in_3_hours
  let time_for_P_to_finish := remaining_job / rate_p
  let minutes_needed := time_for_P_to_finish * 60
  minutes_needed = 15 :=
by
  -- Proof steps go here
  sorry

end P_finishes_in_15_minutes_more_l1840_184037
