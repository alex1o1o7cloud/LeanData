import Mathlib

namespace athlete_speed_l360_36034

theorem athlete_speed (distance time : ℝ) (h1 : distance = 200) (h2 : time = 25) :
  (distance / time) = 8 := by
  sorry

end athlete_speed_l360_36034


namespace area_of_R_sum_m_n_l360_36002

theorem area_of_R_sum_m_n  (s : ℕ) 
  (square_area : ℕ) 
  (rectangle1_area : ℕ)
  (rectangle2_area : ℕ) :
  square_area = 4 → rectangle1_area = 8 → rectangle2_area = 2 → s = 6 → 
  36 - (square_area + rectangle1_area + rectangle2_area) = 22 :=
by
  intros
  sorry

end area_of_R_sum_m_n_l360_36002


namespace difference_of_squares_l360_36019

theorem difference_of_squares (a b : ℕ) (h₁ : a + b = 60) (h₂ : a - b = 14) : a^2 - b^2 = 840 := by
  sorry

end difference_of_squares_l360_36019


namespace floor_sqrt_30_squared_eq_25_l360_36063

theorem floor_sqrt_30_squared_eq_25 (h1 : 5 < Real.sqrt 30) (h2 : Real.sqrt 30 < 6) : Int.floor (Real.sqrt 30) ^ 2 = 25 := 
by
  sorry

end floor_sqrt_30_squared_eq_25_l360_36063


namespace multiplication_identity_l360_36057

theorem multiplication_identity (x y : ℝ) : 
  (2*x^3 - 5*y^2) * (4*x^6 + 10*x^3*y^2 + 25*y^4) = 8*x^9 - 125*y^6 := 
by
  sorry

end multiplication_identity_l360_36057


namespace goats_more_than_pigs_l360_36045

-- Defining the number of goats
def number_of_goats : ℕ := 66

-- Condition: there are twice as many chickens as goats
def number_of_chickens : ℕ := 2 * number_of_goats

-- Calculating the total number of goats and chickens
def total_goats_and_chickens : ℕ := number_of_goats + number_of_chickens

-- Condition: the number of ducks is half of the total number of goats and chickens
def number_of_ducks : ℕ := total_goats_and_chickens / 2

-- Condition: the number of pigs is a third of the number of ducks
def number_of_pigs : ℕ := number_of_ducks / 3

-- The statement we need to prove
theorem goats_more_than_pigs : number_of_goats - number_of_pigs = 33 := by
  -- The proof is omitted as instructed
  sorry

end goats_more_than_pigs_l360_36045


namespace inequality_f_solution_minimum_g_greater_than_f_l360_36076

noncomputable def f (x : ℝ) := abs (x - 2) - abs (x + 1)

theorem inequality_f_solution : {x : ℝ | f x > 1} = {x | x < 0} :=
sorry

noncomputable def g (a x : ℝ) := (a * x^2 - x + 1) / x

theorem minimum_g_greater_than_f (a : ℝ) (h : 0 < a) : 
  (∀ x : ℝ, 0 < x → g a x > f x) ↔ 1 ≤ a :=
sorry

end inequality_f_solution_minimum_g_greater_than_f_l360_36076


namespace part1_part2_l360_36026

def f (x : ℝ) : ℝ := x^2 - 1

theorem part1 (m x : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) (ineq : 4 * m^2 * |f x| + 4 * f m ≤ |f (x-1)|) : 
    -1/2 ≤ m ∧ m ≤ 1/2 := 
sorry

theorem part2 (x1 : ℝ) (hx1 : 1 ≤ x1 ∧ x1 ≤ 2) : 
    (∃ x2 : ℝ, 1 ≤ x2 ∧ x2 ≤ 2 ∧ f x1 = |2 * f x2 - a * x2|) →
    (0 ≤ a ∧ a ≤ 3/2 ∨ a = 3) := 
sorry

end part1_part2_l360_36026


namespace total_pokemon_cards_l360_36074

-- Definitions based on conditions
def dozen := 12
def amount_per_person := 9 * dozen
def num_people := 4

-- Proposition to prove
theorem total_pokemon_cards :
  num_people * amount_per_person = 432 :=
by sorry

end total_pokemon_cards_l360_36074


namespace sarah_photos_l360_36061

theorem sarah_photos (photos_Cristina photos_John photos_Clarissa total_slots : ℕ)
  (hCristina : photos_Cristina = 7)
  (hJohn : photos_John = 10)
  (hClarissa : photos_Clarissa = 14)
  (hTotal : total_slots = 40) :
  ∃ photos_Sarah, photos_Sarah = total_slots - (photos_Cristina + photos_John + photos_Clarissa) ∧ photos_Sarah = 9 :=
by
  sorry

end sarah_photos_l360_36061


namespace A_investment_is_correct_l360_36073

-- Definitions based on the given conditions
def B_investment : ℝ := 8000
def C_investment : ℝ := 10000
def P_B : ℝ := 1000
def diff_P_A_P_C : ℝ := 500

-- Main statement we need to prove
theorem A_investment_is_correct (A_investment : ℝ) 
  (h1 : B_investment = 8000) 
  (h2 : C_investment = 10000)
  (h3 : P_B = 1000)
  (h4 : diff_P_A_P_C = 500)
  (h5 : A_investment = B_investment * (P_B / 1000) * 1.5) :
  A_investment = 12000 :=
sorry

end A_investment_is_correct_l360_36073


namespace find_triangle_areas_l360_36090

variables (A B C D : Point)
variables (S_ABC S_ACD S_ABD S_BCD : ℝ)

def quadrilateral_area (S_ABC S_ACD S_ABD S_BCD : ℝ) : Prop :=
  S_ABC + S_ACD + S_ABD + S_BCD = 25

def conditions (S_ABC S_ACD S_ABD S_BCD : ℝ) : Prop :=
  (S_ABC = 2 * S_BCD) ∧ (S_ABD = 3 * S_ACD)

theorem find_triangle_areas
  (S_ABC S_ACD S_ABD S_BCD : ℝ) :
  quadrilateral_area S_ABC S_ACD S_ABD S_BCD →
  conditions S_ABC S_ACD S_ABD S_BCD →
  S_ABC = 10 ∧ S_ACD = 5 ∧ S_ABD = 15 ∧ S_BCD = 10 :=
by
  sorry

end find_triangle_areas_l360_36090


namespace sliced_meat_cost_per_type_with_rush_shipping_l360_36007

theorem sliced_meat_cost_per_type_with_rush_shipping:
  let original_cost := 40.0
  let rush_delivery_percentage := 0.3
  let num_types := 4
  let rush_delivery_cost := rush_delivery_percentage * original_cost
  let total_cost := original_cost + rush_delivery_cost
  let cost_per_type := total_cost / num_types
  cost_per_type = 13.0 :=
by
  sorry

end sliced_meat_cost_per_type_with_rush_shipping_l360_36007


namespace primes_sum_product_composite_l360_36059

theorem primes_sum_product_composite {p q r : ℕ} (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hdistinct_pq : p ≠ q) (hdistinct_pr : p ≠ r) (hdistinct_qr : q ≠ r) :
  ¬ Nat.Prime (p + q + r + p * q * r) :=
by
  sorry

end primes_sum_product_composite_l360_36059


namespace son_distance_from_father_is_correct_l360_36030

noncomputable def distance_between_son_and_father 
  (L F S d : ℝ) 
  (h_L : L = 6) 
  (h_F : F = 1.8) 
  (h_S : S = 0.9) 
  (h_d : d = 2.1) 
  (incident_point_condition : F / d = L / (d + x) ∧ S / x = F / (d + x)) : ℝ :=
  4.9

theorem son_distance_from_father_is_correct (L F S d : ℝ) 
  (h_L : L = 6) 
  (h_F : F = 1.8) 
  (h_S : S = 0.9) 
  (h_d : d = 2.1)
  (incident_point_condition : F / d = L / (d + 4.9) ∧ S / 4.9 = F / (d + 4.9)) : 
  distance_between_son_and_father L F S d h_L h_F h_S h_d incident_point_condition = 4.9 :=
sorry

end son_distance_from_father_is_correct_l360_36030


namespace percentage_of_amount_l360_36054

theorem percentage_of_amount :
  (0.25 * 300) = 75 :=
by
  sorry

end percentage_of_amount_l360_36054


namespace alex_total_earnings_l360_36077

def total_earnings (hours_w1 hours_w2 wage : ℕ) : ℕ :=
  (hours_w1 + hours_w2) * wage

theorem alex_total_earnings
  (hours_w1 hours_w2 wage : ℕ)
  (h1 : hours_w1 = 28)
  (h2 : hours_w2 = hours_w1 - 10)
  (h3 : wage * 10 = 80) :
  total_earnings hours_w1 hours_w2 wage = 368 :=
by
  sorry

end alex_total_earnings_l360_36077


namespace widgets_unloaded_l360_36003
-- We import the necessary Lean library for general mathematical purposes.

-- We begin the lean statement for our problem.
theorem widgets_unloaded (n_doo n_geegaw n_widget n_yamyam : ℕ) :
  (2^n_doo) * (11^n_geegaw) * (5^n_widget) * (7^n_yamyam) = 104350400 →
  n_widget = 2 := by
  -- Placeholder for proof
  sorry

end widgets_unloaded_l360_36003


namespace mike_travel_time_l360_36052

-- Definitions of conditions
def dave_steps_per_min : ℕ := 85
def dave_step_length_cm : ℕ := 70
def dave_time_min : ℕ := 20
def mike_steps_per_min : ℕ := 95
def mike_step_length_cm : ℕ := 65

-- Calculate Dave's speed in cm/min
def dave_speed_cm_per_min := dave_steps_per_min * dave_step_length_cm

-- Calculate the distance to school in cm
def school_distance_cm := dave_speed_cm_per_min * dave_time_min

-- Calculate Mike's speed in cm/min
def mike_speed_cm_per_min := mike_steps_per_min * mike_step_length_cm

-- Calculate the time for Mike to get to school in minutes as a rational number
def mike_time_min := (school_distance_cm : ℚ) / mike_speed_cm_per_min

-- The proof problem statement
theorem mike_travel_time :
  mike_time_min = 19 + 2 / 7 :=
sorry

end mike_travel_time_l360_36052


namespace total_books_on_shelves_l360_36079

def num_shelves : ℕ := 520
def books_per_shelf : ℝ := 37.5

theorem total_books_on_shelves : num_shelves * books_per_shelf = 19500 :=
by
  sorry

end total_books_on_shelves_l360_36079


namespace range_of_a_l360_36091

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + 2*a*x + a > 0) → 0 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l360_36091


namespace main_theorem_l360_36029

def d_digits (d : ℕ) : Prop :=
  ∃ (d_1 d_2 d_3 d_4 d_5 d_6 d_7 d_8 d_9 : ℕ),
    d = d_1 * 10^8 + d_2 * 10^7 + d_3 * 10^6 + d_4 * 10^5 + d_5 * 10^4 + d_6 * 10^3 + d_7 * 10^2 + d_8 * 10 + d_9

noncomputable def condition1 (d e : ℕ) (i : ℕ) : Prop :=
  (e - (d / 10^(8 - i) % 10)) * 10^(8 - i) + d ≡ 0 [MOD 7]

noncomputable def condition2 (e f : ℕ) (i : ℕ) : Prop :=
  (f - (e / 10^(8 - i) % 10)) * 10^(8 - i) + e ≡ 0 [MOD 7]

theorem main_theorem
  (d e f : ℕ)
  (h1 : d_digits d)
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ 9 → condition1 d e i)
  (h3 : ∀ i, 1 ≤ i ∧ i ≤ 9 → condition2 e f i) :
  ∀ i, 1 ≤ i ∧ i ≤ 9 → (d / 10^(8 - i) % 10) ≡ (f / 10^(8 - i) % 10) [MOD 7] := sorry

end main_theorem_l360_36029


namespace number_of_solutions_l360_36042

theorem number_of_solutions :
  (∃ (xs : List ℤ), (∀ x ∈ xs, |3 * x + 4| ≤ 10) ∧ xs.length = 7) := sorry

end number_of_solutions_l360_36042


namespace smallest_n_condition_l360_36095

theorem smallest_n_condition (n : ℕ) : (4 * n) ∣ (n^2) ∧ (5 * n) ∣ (u^3) → n = 100 :=
by
  sorry

end smallest_n_condition_l360_36095


namespace completing_square_l360_36015

theorem completing_square (x : ℝ) : (x^2 - 2 * x = 2) → ((x - 1)^2 = 3) :=
by
  sorry

end completing_square_l360_36015


namespace mary_money_left_l360_36071

theorem mary_money_left (p : ℝ) : 50 - (4 * p + 2 * p + 4 * p) = 50 - 10 * p := 
by 
  sorry

end mary_money_left_l360_36071


namespace tagged_fish_proportion_l360_36038

def total_fish_in_pond : ℕ := 750
def tagged_fish_first_catch : ℕ := 30
def fish_second_catch : ℕ := 50
def tagged_fish_second_catch := 2

theorem tagged_fish_proportion :
  (tagged_fish_second_catch : ℤ) * (total_fish_in_pond : ℤ) = (tagged_fish_first_catch : ℤ) * (fish_second_catch : ℤ) :=
by
  -- The statement should reflect the given proportion:
  -- T * 750 = 30 * 50
  -- Given T = 2
  sorry

end tagged_fish_proportion_l360_36038


namespace extra_men_needed_l360_36086

theorem extra_men_needed
  (total_length : ℕ) (total_days : ℕ) (initial_men : ℕ)
  (completed_days : ℕ) (completed_work : ℕ) (remaining_work : ℕ)
  (remaining_days : ℕ) (total_man_days_needed : ℕ)
  (number_of_men_needed : ℕ) (extra_men_needed : ℕ)
  (h1 : total_length = 10)
  (h2 : total_days = 60)
  (h3 : initial_men = 30)
  (h4 : completed_days = 20)
  (h5 : completed_work = 2)
  (h6 : remaining_work = total_length - completed_work)
  (h7 : remaining_days = total_days - completed_days)
  (h8 : total_man_days_needed = remaining_work * (completed_days * initial_men) / completed_work)
  (h9 : number_of_men_needed = total_man_days_needed / remaining_days)
  (h10 : extra_men_needed = number_of_men_needed - initial_men)
  : extra_men_needed = 30 :=
by sorry

end extra_men_needed_l360_36086


namespace area_excluding_hole_l360_36064

theorem area_excluding_hole (x : ℝ) : 
  (2 * x + 8) * (x + 6) - (2 * x - 2) * (x - 1) = 24 * x + 46 :=
by
  sorry

end area_excluding_hole_l360_36064


namespace earning_80_yuan_represents_l360_36060

-- Defining the context of the problem
def spending (n : Int) : Int := -n
def earning (n : Int) : Int := n

-- The problem statement as a Lean theorem
theorem earning_80_yuan_represents (x : Int) (hx : earning x = 80) : x = 80 := 
by
  sorry

end earning_80_yuan_represents_l360_36060


namespace find_fourth_number_l360_36023

variable (a : ℕ → ℕ)

theorem find_fourth_number (h₁ : a 7 = 42) (h₂ : a 9 = 110)
    (h₃ : ∀ n, n ≥ 3 → a n = a (n-1) + a (n-2)) : a 4 = 10 :=
by
  sorry

end find_fourth_number_l360_36023


namespace max_value_of_a_l360_36041

variable {R : Type*} [LinearOrderedField R]

def det (a b c d : R) : R := a * d - b * c

theorem max_value_of_a (a : R) :
  (∀ x : R, det (x - 1) (a - 2) (a + 1) x ≥ 1) → a ≤ (3 / 2 : R) :=
by
  sorry

end max_value_of_a_l360_36041


namespace evaluate_expression_l360_36065

theorem evaluate_expression:
  (-2)^2002 + (-1)^2003 + 2^2004 + (-1)^2005 = 3 * 2^2002 - 2 :=
by
  sorry

end evaluate_expression_l360_36065


namespace calc_a8_l360_36085

variable {a : ℕ+ → ℕ}

-- Conditions
axiom recur_relation : ∀ (p q : ℕ+), a (p + q) = a p * a q
axiom initial_condition : a 2 = 2

-- Proof statement
theorem calc_a8 : a 8 = 16 := by
  sorry

end calc_a8_l360_36085


namespace inequality_proof_l360_36040

-- Define the conditions and the theorem statement
variables {a b c d : ℝ}

theorem inequality_proof (h1 : c < d) (h2 : a > b) (h3 : b > 0) : a - c > b - d :=
by
  sorry

end inequality_proof_l360_36040


namespace lucia_hiphop_classes_l360_36022

def cost_hiphop_class : Int := 10
def cost_ballet_class : Int := 12
def cost_jazz_class : Int := 8
def num_ballet_classes : Int := 2
def num_jazz_classes : Int := 1
def total_cost : Int := 52

def num_hiphop_classes : Int := (total_cost - (num_ballet_classes * cost_ballet_class + num_jazz_classes * cost_jazz_class)) / cost_hiphop_class

theorem lucia_hiphop_classes : num_hiphop_classes = 2 := by
  sorry

end lucia_hiphop_classes_l360_36022


namespace triangle_area_l360_36099

/-
A triangle with side lengths in the ratio 4:5:6 is inscribed in a circle of radius 5.
We need to prove that the area of the triangle is 250/9.
-/

theorem triangle_area (x : ℝ) (r : ℝ) (h_r : r = 5) (h_ratio : 6 * x = 2 * r) :
  (1 / 2) * (4 * x) * (5 * x) = 250 / 9 := by 
  -- Proof goes here.
  sorry

end triangle_area_l360_36099


namespace coordinates_of_A_after_move_l360_36018

noncomputable def moved_coordinates (a : ℝ) : ℝ × ℝ :=
  let x := 2 * a - 9 + 5
  let y := 1 - 2 * a
  (x, y)

theorem coordinates_of_A_after_move (a : ℝ) (h : moved_coordinates a = (0, 1 - 2 * a)) :
  moved_coordinates 2 = (-5, -3) :=
by
  -- Proof omitted
  sorry

end coordinates_of_A_after_move_l360_36018


namespace recurring_fraction_division_l360_36011

-- Define the values
def x : ℚ := 8 / 11
def y : ℚ := 20 / 11

-- The theorem statement function to prove x / y = 2 / 5
theorem recurring_fraction_division :
  (x / y = (2 : ℚ) / 5) :=
by 
  -- Skip the proof
  sorry

end recurring_fraction_division_l360_36011


namespace intersection_A_B_l360_36037

def A : Set ℝ := {x | abs x <= 1}

def B : Set ℝ := {y | ∃ x : ℝ, y = x^2}

theorem intersection_A_B :
  (A ∩ B) = {x | 0 ≤ x ∧ x ≤ 1} := sorry

end intersection_A_B_l360_36037


namespace solve_quartic_equation_l360_36081

theorem solve_quartic_equation :
  (∃ x : ℝ, x > 0 ∧ 
    (1 / 3) * (4 * x ^ 2 - 3) = (x ^ 2 - 60 * x - 12) * (x ^ 2 + 30 * x + 6) ∧ 
    ∃ y1 y2 : ℝ, y1 + y2 = 60 ∧ (x^2 - 60 * x - 12 = 0)) → 
    x = 30 + Real.sqrt 912 :=
sorry

end solve_quartic_equation_l360_36081


namespace part1_part2_l360_36010
open Real

noncomputable def f (x : ℝ) (m : ℝ) := x^2 - m * log x
noncomputable def h (x : ℝ) (a : ℝ) := x^2 - x + a
noncomputable def k (x : ℝ) (a : ℝ) := x - 2 * log x - a

theorem part1 (x : ℝ) (m : ℝ) (h_pos_x : 1 < x) : 
  (f x m) - (h x 0) ≥ 0 → m ≤ exp 1 :=
sorry

theorem part2 (a : ℝ) :
  (∀ x, 1 ≤ x ∧ x < 2 → k x a < 0) ∧ 
  (k 2 a < 0) ∧ 
  (∀ x, 2 < x ∧ x ≤ 3 → k x a > 0) →
  2 - 2 * log 2 < a ∧ a ≤ 3 - 2 * log 3 :=
sorry

end part1_part2_l360_36010


namespace intersection_point_is_correct_l360_36008

def line1 (x y : ℝ) := x - 2 * y + 7 = 0
def line2 (x y : ℝ) := 2 * x + y - 1 = 0

theorem intersection_point_is_correct : line1 (-1) 3 ∧ line2 (-1) 3 :=
by
  sorry

end intersection_point_is_correct_l360_36008


namespace point_on_x_axis_l360_36016

theorem point_on_x_axis (A B C D : ℝ × ℝ) : B = (3,0) → B.2 = 0 :=
by
  intros h
  subst h
  exact rfl

end point_on_x_axis_l360_36016


namespace new_person_weight_l360_36012

theorem new_person_weight
  (initial_avg_weight : ℝ := 57)
  (num_people : ℕ := 8)
  (weight_to_replace : ℝ := 55)
  (weight_increase_first : ℝ := 1.5)
  (weight_increase_second : ℝ := 2)
  (weight_increase_third : ℝ := 2.5)
  (weight_increase_fourth : ℝ := 3)
  (weight_increase_fifth : ℝ := 3.5)
  (weight_increase_sixth : ℝ := 4)
  (weight_increase_seventh : ℝ := 4.5) :
  ∃ x : ℝ, x = 67 :=
by
  sorry

end new_person_weight_l360_36012


namespace tank_fill_time_with_leak_l360_36096

theorem tank_fill_time_with_leak 
  (pump_fill_time : ℕ) (leak_empty_time : ℕ) (effective_fill_time : ℕ)
  (hp : pump_fill_time = 5)
  (hl : leak_empty_time = 10)
  (he : effective_fill_time = 10) : effective_fill_time = 10 :=
by
  sorry

end tank_fill_time_with_leak_l360_36096


namespace divisibility_by_30_l360_36087

theorem divisibility_by_30 (p : ℕ) (hp_prime : Nat.Prime p) (hp_ge_3 : p ≥ 3) : 30 ∣ (p^3 - 1) ↔ p % 15 = 1 := 
  sorry

end divisibility_by_30_l360_36087


namespace set_S_infinite_l360_36001

-- Definition of a power
def is_power (n : ℕ) : Prop := 
  ∃ (a k : ℕ), a > 0 ∧ k ≥ 2 ∧ n = a^k

-- Definition of the set S, those integers which cannot be expressed as the sum of two powers
def in_S (n : ℕ) : Prop := 
  ¬ ∃ (a b k m : ℕ), a > 0 ∧ b > 0 ∧ k ≥ 2 ∧ m ≥ 2 ∧ n = a^k + b^m

-- The theorem statement asserting that S is infinite
theorem set_S_infinite : Infinite {n : ℕ | in_S n} :=
sorry

end set_S_infinite_l360_36001


namespace simplify_and_evaluate_l360_36027

theorem simplify_and_evaluate (x y : ℤ) (hx : x = -1) (hy : y = 2) : 
  x^2 - 2 * (3 * y^2 - x * y) + (y^2 - 2 * x * y) = -19 := 
by
  -- Proof will go here, but it's omitted as per instructions
  sorry

end simplify_and_evaluate_l360_36027


namespace cosine_of_angle_in_third_quadrant_l360_36049

theorem cosine_of_angle_in_third_quadrant (B : ℝ) (hB : B ∈ Set.Ioo (π : ℝ) (3 * π / 2)) (hSinB : Real.sin B = -5 / 13) :
  Real.cos B = -12 / 13 :=
sorry

end cosine_of_angle_in_third_quadrant_l360_36049


namespace quadratic_coefficient_a_l360_36097

theorem quadratic_coefficient_a (a b c : ℝ) :
  (2 = 9 * a - 3 * b + c) ∧
  (2 = 9 * a + 3 * b + c) ∧
  (-6 = 4 * a + 2 * b + c) →
  a = 8 / 5 :=
by
  sorry

end quadratic_coefficient_a_l360_36097


namespace triangle_area_eq_l360_36044

/--
Given:
1. The base of the triangle is 4 meters.
2. The height of the triangle is 5 meters.

Prove:
The area of the triangle is 10 square meters.
-/
theorem triangle_area_eq (base height : ℝ) (h_base : base = 4) (h_height : height = 5) : 
  (base * height / 2) = 10 := by
  sorry

end triangle_area_eq_l360_36044


namespace probability_sin_cos_in_range_l360_36075

noncomputable def probability_sin_cos_interval : ℝ :=
  let interval_length := (Real.pi / 2 + Real.pi / 6)
  let valid_length := (Real.pi / 2 - 0)
  valid_length / interval_length

theorem probability_sin_cos_in_range :
  probability_sin_cos_interval = 3 / 4 :=
sorry

end probability_sin_cos_in_range_l360_36075


namespace find_special_integers_l360_36004

theorem find_special_integers (n : ℕ) (h : n > 1) :
  (∀ d, d ∣ n ∧ d > 1 → ∃ a r, a > 0 ∧ r > 1 ∧ d = a^r + 1) ↔ (n = 10 ∨ ∃ a, a > 0 ∧ n = a^2 + 1) :=
by
  sorry

end find_special_integers_l360_36004


namespace weight_of_b_l360_36039

theorem weight_of_b (A B C : ℝ)
  (h1 : A + B + C = 135)
  (h2 : A + B = 80)
  (h3 : B + C = 94) : 
  B = 39 := 
by 
  sorry

end weight_of_b_l360_36039


namespace quadratic_roots_real_distinct_l360_36069

theorem quadratic_roots_real_distinct (k : ℝ) (h : k < 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + x1 + k - 1 = 0) ∧ (x2^2 + x2 + k - 1 = 0) :=
by
  sorry

end quadratic_roots_real_distinct_l360_36069


namespace cos_alpha_add_beta_div2_l360_36021

open Real 

theorem cos_alpha_add_beta_div2 (α β : ℝ) 
  (h_range : -π/2 < β ∧ β < 0 ∧ 0 < α ∧ α < π/2)
  (h_cos1 : cos (π/4 + α) = 1/3)
  (h_cos2 : cos (π/4 - β/2) = sqrt 3 / 3) :
  cos (α + β/2) = 5 * sqrt 3 / 9 :=
sorry

end cos_alpha_add_beta_div2_l360_36021


namespace kanul_total_amount_l360_36032

variable (T : ℝ)
variable (H1 : 3000 + 2000 + 0.10 * T = T)

theorem kanul_total_amount : T = 5555.56 := 
by 
  /- with the conditions given, 
     we can proceed to prove T = 5555.56 -/
  sorry

end kanul_total_amount_l360_36032


namespace slope_of_line_l360_36043

theorem slope_of_line : ∀ (x y : ℝ), (x / 4 - y / 3 = 1) → ((3 * x / 4) - 3) = 0 → (y = (3 / 4) * x - 3) :=
by 
  intros x y h_eq h_slope 
  sorry

end slope_of_line_l360_36043


namespace twelve_percent_greater_than_80_l360_36092

theorem twelve_percent_greater_than_80 (x : ℝ) (h : x = 80 + 0.12 * 80) : x = 89.6 :=
by
  sorry

end twelve_percent_greater_than_80_l360_36092


namespace find_x_and_y_l360_36082

variables (x y : ℝ)

def arithmetic_mean_condition : Prop := (8 + 15 + x + y + 22 + 30) / 6 = 15
def relationship_condition : Prop := y = x + 6

theorem find_x_and_y (h1 : arithmetic_mean_condition x y) (h2 : relationship_condition x y) : 
  x = 4.5 ∧ y = 10.5 :=
by
  sorry

end find_x_and_y_l360_36082


namespace area_of_shaded_region_l360_36046

theorem area_of_shaded_region :
  let inner_square_side_length := 3
  let triangle_base := 2
  let triangle_height := 1
  let number_of_triangles := 8
  let area_inner_square := inner_square_side_length * inner_square_side_length
  let area_one_triangle := (1/2) * triangle_base * triangle_height
  let total_area_triangles := number_of_triangles * area_one_triangle
  let total_area_shaded := area_inner_square + total_area_triangles
  total_area_shaded = 17 :=
sorry

end area_of_shaded_region_l360_36046


namespace f_identically_zero_l360_36089

open Real

-- Define the function f and its properties
noncomputable def f : ℝ → ℝ := sorry

-- Given conditions
axiom func_eqn (a b : ℝ) : f (a * b) = a * f b + b * f a 
axiom func_bounded (x : ℝ) : |f x| ≤ 1

-- Goal: Prove that f is identically zero
theorem f_identically_zero : ∀ x : ℝ, f x = 0 := 
by
  sorry

end f_identically_zero_l360_36089


namespace percentage_multiplication_l360_36055

theorem percentage_multiplication :
  (0.15 * 0.20 * 0.25) * 100 = 0.75 := 
by
  sorry

end percentage_multiplication_l360_36055


namespace expression_value_l360_36051

theorem expression_value (a b : ℝ) (h : a^2 * b^2 / (a^4 - 2 * b^4) = 1) : 
  (a^2 - b^2) / (a^2 + b^2) = 1 / 3 := 
by 
  sorry

end expression_value_l360_36051


namespace problem_solution_l360_36056

def equal_group_B : Prop :=
  (-2)^3 = -(2^3)

theorem problem_solution : equal_group_B := by
  sorry

end problem_solution_l360_36056


namespace quadratic_distinct_real_roots_l360_36080

theorem quadratic_distinct_real_roots (c : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + c = 0 ∧ y^2 - 2*y + c = 0) ↔ c < 1 :=
by
  sorry

end quadratic_distinct_real_roots_l360_36080


namespace log_equation_positive_x_l360_36098

theorem log_equation_positive_x (x : ℝ) (hx : 0 < x) (hx1 : x ≠ 1) : 
  (Real.log x / Real.log 2) * (Real.log 9 / Real.log x) = Real.log 9 / Real.log 2 :=
by sorry

end log_equation_positive_x_l360_36098


namespace max_pasture_area_maximization_l360_36093

noncomputable def max_side_length (fence_cost_per_foot : ℕ) (total_cost : ℕ) : ℕ :=
  let total_length := total_cost / fence_cost_per_foot
  let x := total_length / 4
  2 * x

theorem max_pasture_area_maximization :
  max_side_length 8 1920 = 120 :=
by
  sorry

end max_pasture_area_maximization_l360_36093


namespace hyperbola_properties_l360_36013

-- Definitions from the conditions
def line_l (x y : ℝ) : Prop := 4 * x - 3 * y + 20 = 0
def asymptote_l (x y : ℝ) : Prop := 4 * x - 3 * y = 0
def foci_on_x_axis (x y : ℝ) : Prop := y = 0

-- Standard equation of the hyperbola
def hyperbola_equation (x y : ℝ) : Prop := (x^2 / 9) - (y^2 / 16) = 1

-- Define eccentricity
def eccentricity := 5 / 3

-- Proof statement
theorem hyperbola_properties :
  (∃ x y : ℝ, line_l x y ∧ foci_on_x_axis x y) →
  (∃ x y : ℝ, asymptote_l x y) →
  ∃ x y : ℝ, hyperbola_equation x y ∧ eccentricity = 5 / 3 :=
by
  sorry

end hyperbola_properties_l360_36013


namespace sum_first_nine_terms_arithmetic_sequence_l360_36017

theorem sum_first_nine_terms_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : ∀ n, a (n + 1) - a n = (a 2 - a 1))
  (h1 : a 1 + a 4 + a 7 = 45)
  (h2 : a 3 + a 6 + a 9 = 27) :
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9) = 108 := 
sorry

end sum_first_nine_terms_arithmetic_sequence_l360_36017


namespace no_such_function_l360_36072

theorem no_such_function :
  ¬ (∃ f : ℕ → ℕ, ∀ n ≥ 2, f (f (n - 1)) = f (n + 1) - f (n)) :=
sorry

end no_such_function_l360_36072


namespace probability_A_and_B_selected_l360_36058

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l360_36058


namespace polynomial_expansion_identity_l360_36083

theorem polynomial_expansion_identity
  (a a1 a3 a4 a5 : ℝ)
  (h : (a - x)^5 = a + a1 * x + 80 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5) :
  a + a1 + 80 + a3 + a4 + a5 = 1 := 
sorry

end polynomial_expansion_identity_l360_36083


namespace vector_decomposition_l360_36024

noncomputable def x : ℝ × ℝ × ℝ := (5, 15, 0)
noncomputable def p : ℝ × ℝ × ℝ := (1, 0, 5)
noncomputable def q : ℝ × ℝ × ℝ := (-1, 3, 2)
noncomputable def r : ℝ × ℝ × ℝ := (0, -1, 1)

theorem vector_decomposition : x = (4 : ℝ) • p + (-1 : ℝ) • q + (-18 : ℝ) • r :=
by
  sorry

end vector_decomposition_l360_36024


namespace solve_for_q_l360_36050

theorem solve_for_q (n m q: ℚ)
  (h1 : 3 / 4 = n / 88)
  (h2 : 3 / 4 = (m + n) / 100)
  (h3 : 3 / 4 = (q - m) / 150) :
  q = 121.5 :=
sorry

end solve_for_q_l360_36050


namespace correct_algorithm_description_l360_36028

def conditions_about_algorithms (desc : String) : Prop :=
  (desc = "A" → false) ∧
  (desc = "B" → false) ∧
  (desc = "C" → true) ∧
  (desc = "D" → false)

theorem correct_algorithm_description : ∃ desc : String, 
  conditions_about_algorithms desc :=
by
  use "C"
  unfold conditions_about_algorithms
  simp
  sorry

end correct_algorithm_description_l360_36028


namespace snow_at_mrs_hilts_house_l360_36025

theorem snow_at_mrs_hilts_house
    (snow_at_school : ℕ)
    (extra_snow_at_house : ℕ) 
    (school_snow_amount : snow_at_school = 17) 
    (extra_snow_amount : extra_snow_at_house = 12) :
  snow_at_school + extra_snow_at_house = 29 := 
by
  sorry

end snow_at_mrs_hilts_house_l360_36025


namespace greater_number_is_18_l360_36033

theorem greater_number_is_18 (x y : ℕ) (h₁ : x + y = 30) (h₂ : x - y = 6) : x = 18 :=
by
  sorry

end greater_number_is_18_l360_36033


namespace arithmetic_sequence_eleven_term_l360_36020

theorem arithmetic_sequence_eleven_term (a1 d a11 : ℕ) (h_sum7 : 7 * (2 * a1 + 6 * d) = 154) (h_a1 : a1 = 5) :
  a11 = a1 + 10 * d → a11 = 25 :=
by
  sorry

end arithmetic_sequence_eleven_term_l360_36020


namespace unique_real_solution_k_l360_36094

theorem unique_real_solution_k (k : ℝ) :
  ∃! x : ℝ, (3 * x + 8) * (x - 6) = -62 + k * x ↔ k = -10 + 12 * Real.sqrt 1.5 ∨ k = -10 - 12 * Real.sqrt 1.5 := by
  sorry

end unique_real_solution_k_l360_36094


namespace force_of_water_pressure_on_plate_l360_36088

noncomputable def force_on_plate_under_water (γ : ℝ) (g : ℝ) (a b : ℝ) : ℝ :=
  γ * g * (b^2 - a^2) / 2

theorem force_of_water_pressure_on_plate :
  let γ : ℝ := 1000 -- kg/m^3
  let g : ℝ := 9.81  -- m/s^2
  let a : ℝ := 0.5   -- top depth
  let b : ℝ := 2.5   -- bottom depth
  force_on_plate_under_water γ g a b = 29430 := sorry

end force_of_water_pressure_on_plate_l360_36088


namespace find_pairs_l360_36062

theorem find_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  (3 * a + 1 ∣ 4 * b - 1) ∧ (2 * b + 1 ∣ 3 * a - 1) ↔ (a = 2 ∧ b = 2) := 
by 
  sorry

end find_pairs_l360_36062


namespace sum_of_square_areas_l360_36006

theorem sum_of_square_areas (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 1) : a^2 + b^2 = 7 :=
sorry

end sum_of_square_areas_l360_36006


namespace intersection_eq_l360_36031

def A : Set ℤ := {x | x ∈ Set.Icc (-2 : ℤ) 2}
def B : Set ℝ := {y | y ≤ 1}

theorem intersection_eq : A ∩ {y | y ∈ Set.Icc (-2 : ℤ) 1} = {-2, -1, 0, 1} := by
  sorry

end intersection_eq_l360_36031


namespace password_count_correct_l360_36047

-- Defining variables
def n_letters := 26
def n_digits := 10

-- The number of permutations for selecting 2 different letters
def perm_letters := n_letters * (n_letters - 1)
-- The number of permutations for selecting 2 different numbers
def perm_digits := n_digits * (n_digits - 1)

-- The total number of possible passwords
def total_permutations := perm_letters * perm_digits

-- The theorem we need to prove
theorem password_count_correct :
  total_permutations = (n_letters * (n_letters - 1)) * (n_digits * (n_digits - 1)) :=
by
  -- The proof goes here
  sorry

end password_count_correct_l360_36047


namespace carpet_covering_cost_l360_36009

noncomputable def carpet_cost (floor_length floor_width carpet_length carpet_width carpet_cost_per_square : ℕ) : ℕ :=
  let floor_area := floor_length * floor_width
  let carpet_area := carpet_length * carpet_width
  let num_of_squares := floor_area / carpet_area
  num_of_squares * carpet_cost_per_square

theorem carpet_covering_cost :
  carpet_cost 6 10 2 2 15 = 225 :=
by
  sorry

end carpet_covering_cost_l360_36009


namespace calculate_fixed_payment_calculate_variable_payment_compare_plans_for_x_eq_30_l360_36014

noncomputable def cost_plan1_fixed (num_suits num_ties : ℕ) : ℕ :=
  if num_ties > num_suits then 200 * num_suits + 40 * (num_ties - num_suits)
  else 200 * num_suits

noncomputable def cost_plan2_fixed (num_suits num_ties : ℕ) : ℕ :=
  (200 * num_suits + 40 * num_ties) * 9 / 10

noncomputable def cost_plan1_variable (num_suits : ℕ) (x : ℕ) : ℕ :=
  200 * num_suits + 40 * (x - num_suits)

noncomputable def cost_plan2_variable (num_suits : ℕ) (x : ℕ) : ℕ :=
  (200 * num_suits + 40 * x) * 9 / 10

theorem calculate_fixed_payment :
  cost_plan1_fixed 20 22 = 4080 ∧ cost_plan2_fixed 20 22 = 4392 :=
by sorry

theorem calculate_variable_payment (x : ℕ) (hx : x > 20) :
  cost_plan1_variable 20 x = 40 * x + 3200 ∧ cost_plan2_variable 20 x = 36 * x + 3600 :=
by sorry

theorem compare_plans_for_x_eq_30 :
  cost_plan1_variable 20 30 < cost_plan2_variable 20 30 :=
by sorry


end calculate_fixed_payment_calculate_variable_payment_compare_plans_for_x_eq_30_l360_36014


namespace right_triangle_hypotenuse_l360_36067

noncomputable def triangle_hypotenuse (a b c : ℝ) : Prop :=
(a + b + c = 40) ∧
(a * b = 48) ∧
(a^2 + b^2 = c^2) ∧
(c = 18.8)

theorem right_triangle_hypotenuse :
  ∃ (a b c : ℝ), triangle_hypotenuse a b c :=
by
  sorry

end right_triangle_hypotenuse_l360_36067


namespace number_of_dimes_l360_36035

-- Definitions based on conditions
def total_coins : Nat := 28
def nickels : Nat := 4

-- Definition of the number of dimes.
def dimes : Nat := total_coins - nickels

-- Theorem statement with the expected answer
theorem number_of_dimes : dimes = 24 := by
  -- Proof is skipped with sorry
  sorry

end number_of_dimes_l360_36035


namespace triangle_is_acute_l360_36066

-- Define the condition that the angles have a ratio of 2:3:4
def angle_ratio_cond (a b c : ℝ) : Prop :=
  a / b = 2 / 3 ∧ b / c = 3 / 4

-- Define the sum of the angles in a triangle
def angle_sum_cond (a b c : ℝ) : Prop :=
  a + b + c = 180

-- The proof problem stating that triangle with angles in ratio 2:3:4 is acute
theorem triangle_is_acute (a b c : ℝ) (h_ratio : angle_ratio_cond a b c) (h_sum : angle_sum_cond a b c) : 
  a < 90 ∧ b < 90 ∧ c < 90 := 
by
  sorry

end triangle_is_acute_l360_36066


namespace geo_seq_sum_neg_six_l360_36053

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ (a₁ q : ℝ), q ≠ 0 ∧ ∀ n, a n = a₁ * q^n

theorem geo_seq_sum_neg_six
  (a : ℕ → ℝ)
  (hgeom : geometric_sequence a)
  (ha_neg : a 1 < 0)
  (h_condition : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 36) :
  a 3 + a 5 = -6 :=
  sorry

end geo_seq_sum_neg_six_l360_36053


namespace range_of_a_in_circle_l360_36048

theorem range_of_a_in_circle (a : ℝ) : 
  ((1 - a)^2 + (1 + a)^2 < 4) ↔ (-1 < a ∧ a < 1) :=
by
  sorry

end range_of_a_in_circle_l360_36048


namespace exp_addition_property_l360_36005

theorem exp_addition_property (x y : ℝ) : (Real.exp (x + y)) = (Real.exp x) * (Real.exp y) := 
sorry

end exp_addition_property_l360_36005


namespace keystone_arch_larger_angle_l360_36068

def isosceles_trapezoid_larger_angle (n : ℕ) : Prop :=
  n = 10 → ∃ (x : ℝ), x = 99

theorem keystone_arch_larger_angle :
  isosceles_trapezoid_larger_angle 10 :=
by
  sorry

end keystone_arch_larger_angle_l360_36068


namespace sqrt_product_l360_36000

theorem sqrt_product (a b : ℝ) (ha : a = 20) (hb : b = 1/5) : Real.sqrt a * Real.sqrt b = 2 := 
by
  sorry

end sqrt_product_l360_36000


namespace ratio_arithmetic_sequence_last_digit_l360_36084

def is_ratio_arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) :=
  ∀ n : ℕ, n > 0 → (a (n + 2) * a n) = (a (n + 1) ^ 2) * d

theorem ratio_arithmetic_sequence_last_digit :
  ∃ a : ℕ → ℕ, is_ratio_arithmetic_sequence a 1 ∧ a 1 = 1 ∧ a 2 = 1 ∧ a 3 = 2 ∧
  (a 2009 / a 2006) % 10 = 6 :=
sorry

end ratio_arithmetic_sequence_last_digit_l360_36084


namespace right_triangle_acute_angles_l360_36036

theorem right_triangle_acute_angles (α β : ℝ) 
  (h1 : α + β = 90)
  (h2 : ∀ (δ1 δ2 ε1 ε2 : ℝ), δ1 + ε1 = 135 ∧ δ1 / ε1 = 13 / 17 
                       ∧ ε2 = 180 - ε1 ∧ δ2 = 180 - δ1) :
  α = 63 ∧ β = 27 := 
  sorry

end right_triangle_acute_angles_l360_36036


namespace number_and_its_square_root_l360_36078

theorem number_and_its_square_root (x : ℝ) (h : x + 10 * Real.sqrt x = 39) : x = 9 :=
sorry

end number_and_its_square_root_l360_36078


namespace cone_height_l360_36070

theorem cone_height (h : ℝ) (r : ℝ) 
  (volume_eq : (1/3) * π * r^2 * h = 19683 * π) 
  (isosceles_right_triangle : h = r) : 
  h = 39.0 :=
by
  -- The proof will go here
  sorry

end cone_height_l360_36070
