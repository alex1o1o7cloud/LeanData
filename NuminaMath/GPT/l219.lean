import Mathlib

namespace students_selected_milk_l219_219802

noncomputable def selected_soda_percent : ℚ := 50 / 100
noncomputable def selected_milk_percent : ℚ := 30 / 100
noncomputable def selected_soda_count : ℕ := 90
noncomputable def selected_milk_count := selected_milk_percent / selected_soda_percent * selected_soda_count

theorem students_selected_milk :
    selected_milk_count = 54 :=
by
  sorry

end students_selected_milk_l219_219802


namespace least_sum_of_variables_l219_219631

theorem least_sum_of_variables (x y z w : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0)  
  (h : 2 * x^2 = 5 * y^3 ∧ 5 * y^3 = 8 * z^4 ∧ 8 * z^4 = 3 * w) : x + y + z + w = 54 := 
sorry

end least_sum_of_variables_l219_219631


namespace total_shaded_area_l219_219218

theorem total_shaded_area (S T : ℕ) 
  (h1 : 12 / S = 4)
  (h2 : S / T = 3) :
  (S * S) + 8 * (T * T) = 17 :=
by
  sorry

end total_shaded_area_l219_219218


namespace speed_of_boat_in_still_water_l219_219952

theorem speed_of_boat_in_still_water
    (speed_stream : ℝ)
    (distance_downstream : ℝ)
    (distance_upstream : ℝ)
    (t : ℝ)
    (x : ℝ)
    (h1 : speed_stream = 10)
    (h2 : distance_downstream = 80)
    (h3 : distance_upstream = 40)
    (h4 : t = distance_downstream / (x + speed_stream))
    (h5 : t = distance_upstream / (x - speed_stream)) :
  x = 30 :=
by sorry

end speed_of_boat_in_still_water_l219_219952


namespace numberOfBoysInClass_l219_219035

-- Define the problem condition: students sit in a circle and boy at 5th position is opposite to boy at 20th position
def studentsInCircle (n : ℕ) : Prop :=
  (n > 5) ∧ (n > 20) ∧ ((20 - 5) * 2 + 2 = n)

-- The main theorem: Given the conditions, prove the total number of boys equals 32
theorem numberOfBoysInClass : ∀ n : ℕ, studentsInCircle n → n = 32 :=
by
  intros n hn
  sorry

end numberOfBoysInClass_l219_219035


namespace quadratic_inequality_solution_l219_219960

-- Definition of the given conditions and the theorem to prove
theorem quadratic_inequality_solution 
  (a b c : ℝ)
  (h1 : a < 0)
  (h2 : ∀ x, ax^2 + bx + c < 0 ↔ x < -2 ∨ x > -1/2) :
  ∀ x, ax^2 - bx + c > 0 ↔ 1/2 < x ∧ x < 2 :=
by
  sorry

end quadratic_inequality_solution_l219_219960


namespace intersection_eq_l219_219652

def A : Set ℝ := {x | -2 < x ∧ x ≤ 1}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 3}

theorem intersection_eq : A ∩ B = {x | -1 < x ∧ x ≤ 1} :=
by
  sorry

end intersection_eq_l219_219652


namespace machines_used_l219_219828

variable (R S : ℕ)

/-- 
  A company has two types of machines, type R and type S. 
  Operating at a constant rate, a machine of type R does a certain job in 36 hours, 
  and a machine of type S does the job in 9 hours. 
  If the company used the same number of each type of machine to do the job in 12 hours, 
  then the company used 15 machines of type R.
-/
theorem machines_used (hR : ∀ ⦃n⦄, n * (1 / 36) + n * (1 / 9) = (1 / 12)) :
  R = 15 := 
by 
  sorry

end machines_used_l219_219828


namespace intersection_A_B_l219_219749

-- Define the sets A and B
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {x | x > 0}

-- The theorem we want to prove
theorem intersection_A_B : A ∩ B = {1} := 
by {
  sorry
}

end intersection_A_B_l219_219749


namespace odd_positive_int_divides_3pow_n_plus_1_l219_219270

theorem odd_positive_int_divides_3pow_n_plus_1 (n : ℕ) (hn_odd : n % 2 = 1) (hn_pos : n > 0) : 
  n ∣ (3^n + 1) ↔ n = 1 := 
by
  sorry

end odd_positive_int_divides_3pow_n_plus_1_l219_219270


namespace dot_product_a_b_l219_219826

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 1)

theorem dot_product_a_b : (a.1 * b.1 + a.2 * b.2) = -1 := by
  sorry

end dot_product_a_b_l219_219826


namespace sum_PS_TV_l219_219348

theorem sum_PS_TV 
  (P V : ℝ) 
  (hP : P = 3) 
  (hV : V = 33)
  (n : ℕ) 
  (hn : n = 6) 
  (Q R S T U : ℝ) 
  (hPR : P < Q ∧ Q < R ∧ R < S ∧ S < T ∧ T < U ∧ U < V)
  (h_divide : ∀ i : ℕ, i ≤ n → P + i * (V - P) / n = P + i * 5) :
  (P, V, Q, R, S, T, U) = (3, 33, 8, 13, 18, 23, 28) → (S - P) + (V - T) = 25 :=
by {
  sorry
}

end sum_PS_TV_l219_219348


namespace students_taking_art_l219_219102

theorem students_taking_art :
  ∀ (total_students music_students both_students neither_students : ℕ),
  total_students = 500 →
  music_students = 30 →
  both_students = 10 →
  neither_students = 460 →
  music_students + both_students + neither_students = total_students →
  ((total_students - neither_students) - (music_students - both_students) + both_students = 20) :=
by
  intros total_students music_students both_students neither_students 
  intro h_total h_music h_both h_neither h_sum 
  sorry

end students_taking_art_l219_219102


namespace larger_number_solution_l219_219009

theorem larger_number_solution (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 10) : x = 25 :=
by
  sorry

end larger_number_solution_l219_219009


namespace maximum_additional_payment_expected_value_difference_l219_219815

-- Add the conditions as definitions
def a1 : ℕ := 1298
def a2 : ℕ := 1347
def a3 : ℕ := 1337
def b1 : ℕ := 1402
def b2 : ℕ := 1310
def b3 : ℕ := 1298

-- Prices in rubles per kilowatt-hour
def peak_price : ℝ := 4.03
def night_price : ℝ := 1.01
def semi_peak_price : ℝ := 3.39

-- Actual consumptions in kilowatt-hour
def ΔP : ℝ := 104
def ΔN : ℝ := 37
def ΔSP : ℝ := 39

-- Correct payment calculated by the company
def correct_payment : ℝ := 660.72

-- Statements to prove
theorem maximum_additional_payment : 397.34 = (104 * 4.03 + 39 * 3.39 + 37 * 1.01 - 660.72) :=
by
  sorry

theorem expected_value_difference : 19.3 = ((5 * 1402 + 3 * 1347 + 1337 - 1298 - 3 * 1270 - 5 * 1214) / 15 * 8.43 - 660.72) :=
by
  sorry

end maximum_additional_payment_expected_value_difference_l219_219815


namespace points_symmetric_about_x_axis_l219_219651

def point := ℝ × ℝ

def symmetric_x_axis (A B : point) : Prop :=
  A.1 = B.1 ∧ A.2 = -B.2

theorem points_symmetric_about_x_axis : symmetric_x_axis (-1, 3) (-1, -3) :=
by
  sorry

end points_symmetric_about_x_axis_l219_219651


namespace ninety_eight_squared_l219_219152

theorem ninety_eight_squared : 98^2 = 9604 :=
by 
  -- The proof steps are omitted and replaced with 'sorry'
  sorry

end ninety_eight_squared_l219_219152


namespace solve_for_b_l219_219503

theorem solve_for_b (b : ℚ) (h : b - b / 4 = 5 / 2) : b = 10 / 3 :=
by 
  sorry

end solve_for_b_l219_219503


namespace number_of_integers_l219_219448

theorem number_of_integers (n : ℤ) : (200 < n ∧ n < 300 ∧ ∃ r : ℤ, n % 7 = r ∧ n % 9 = r) ↔ 
  n = 252 ∨ n = 253 ∨ n = 254 ∨ n = 255 ∨ n = 256 ∨ n = 257 ∨ n = 258 :=
by {
  sorry
}

end number_of_integers_l219_219448


namespace union_of_A_and_B_l219_219200

open Set

-- Define the sets A and B based on given conditions
def A (x : ℤ) : Set ℤ := {y | y = x^2 ∨ y = 2 * x - 1 ∨ y = -4}
def B (x : ℤ) : Set ℤ := {y | y = x - 5 ∨ y = 1 - x ∨ y = 9}

-- Specific condition given in the problem
def A_intersect_B_condition (x : ℤ) : Prop :=
  A x ∩ B x = {9}

-- Prove problem statement that describes the union of A and B
theorem union_of_A_and_B (x : ℤ) (h : A_intersect_B_condition x) : A x ∪ B x = {-8, -7, -4, 4, 9} :=
sorry

end union_of_A_and_B_l219_219200


namespace pyramid_volume_l219_219922

noncomputable def volume_of_pyramid (EF FG QE : ℝ) : ℝ :=
  (1 / 3) * (EF * FG) * QE

theorem pyramid_volume
  (EF FG QE : ℝ)
  (h1 : EF = 10)
  (h2 : FG = 5)
  (h3 : QE = 9) :
  volume_of_pyramid EF FG QE = 150 :=
by
  simp [volume_of_pyramid, h1, h2, h3]
  sorry

end pyramid_volume_l219_219922


namespace total_number_of_coins_l219_219548

theorem total_number_of_coins (x : ℕ) (h : 1 * x + 5 * x + 10 * x + 50 * x + 100 * x = 332) : 5 * x = 10 :=
by {
  sorry
}

end total_number_of_coins_l219_219548


namespace jerry_books_vs_action_figures_l219_219491

-- Define the initial conditions as constants
def initial_books : ℕ := 7
def initial_action_figures : ℕ := 3
def added_action_figures : ℕ := 2

-- Define the total number of action figures after adding
def total_action_figures : ℕ := initial_action_figures + added_action_figures

-- The theorem we need to prove
theorem jerry_books_vs_action_figures : initial_books - total_action_figures = 2 :=
by
  -- Proof placeholder
  sorry

end jerry_books_vs_action_figures_l219_219491


namespace transportation_degrees_l219_219162

theorem transportation_degrees
  (salaries : ℕ) (r_and_d : ℕ) (utilities : ℕ) (equipment : ℕ) (supplies : ℕ) (total_degrees : ℕ)
  (h_salaries : salaries = 60)
  (h_r_and_d : r_and_d = 9)
  (h_utilities : utilities = 5)
  (h_equipment : equipment = 4)
  (h_supplies : supplies = 2)
  (h_total_degrees : total_degrees = 360) :
  (total_degrees * (100 - (salaries + r_and_d + utilities + equipment + supplies)) / 100 = 72) :=
by {
  sorry
}

end transportation_degrees_l219_219162


namespace area_of_given_triangle_is_8_l219_219709

-- Define the vertices of the triangle
def x1 := 2
def y1 := -3
def x2 := -1
def y2 := 6
def x3 := 4
def y3 := -5

-- Define the determinant formula for the area of the triangle
def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℤ) : ℤ :=
  abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2

theorem area_of_given_triangle_is_8 :
  area_of_triangle x1 y1 x2 y2 x3 y3 = 8 := by
  sorry

end area_of_given_triangle_is_8_l219_219709


namespace function_bounds_l219_219954

theorem function_bounds {a : ℝ} :
  (∀ x : ℝ, x > 0 → 4 - x^2 + a * Real.log x ≤ 3) → a = 2 :=
by
  sorry

end function_bounds_l219_219954


namespace gain_percent_is_33_33_l219_219266
noncomputable def gain_percent_calculation (C S : ℝ) := ((S - C) / C) * 100

theorem gain_percent_is_33_33
  (C S : ℝ)
  (h : 75 * C = 56.25 * S) :
  gain_percent_calculation C S = 33.33 := by
  sorry

end gain_percent_is_33_33_l219_219266


namespace problem_l219_219124

def f (x : ℝ) : ℝ := x^3 + 2 * x

theorem problem : f 5 + f (-5) = 0 := by
  sorry

end problem_l219_219124


namespace intersection_M_N_l219_219511

-- Definitions of sets M and N
def M : Set ℝ := {x | x < 2}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- The statement to prove
theorem intersection_M_N : M ∩ N = {x | 1 ≤ x ∧ x < 2} := 
by 
  sorry

end intersection_M_N_l219_219511


namespace greatest_divisor_of_976543_and_897623_l219_219105

theorem greatest_divisor_of_976543_and_897623 :
  Nat.gcd (976543 - 7) (897623 - 11) = 4 := by
  sorry

end greatest_divisor_of_976543_and_897623_l219_219105


namespace f_one_eq_zero_l219_219320

-- Define the function f on ℝ
variable (f : ℝ → ℝ)

-- Conditions for the problem
axiom odd_function : ∀ x : ℝ, f (-x) = -f (x)
axiom periodic_function : ∀ x : ℝ, f (x + 2) = f (x)

-- Goal: Prove that f(1) = 0
theorem f_one_eq_zero : f 1 = 0 :=
by
  sorry

end f_one_eq_zero_l219_219320


namespace jellybeans_to_buy_l219_219529

-- Define the conditions: a minimum of 150 jellybeans and a remainder of 15 when divided by 17.
def condition (n : ℕ) : Prop :=
  n ≥ 150 ∧ n % 17 = 15

-- Define the main statement to prove: if condition holds, then n is 151
theorem jellybeans_to_buy (n : ℕ) (h : condition n) : n = 151 :=
by
  -- Proof is skipped with sorry
  sorry

end jellybeans_to_buy_l219_219529


namespace product_of_two_digit_numbers_5488_has_smaller_number_56_l219_219796

theorem product_of_two_digit_numbers_5488_has_smaller_number_56 (a b : ℕ) (h_a2 : 10 ≤ a) (h_a3 : a < 100) (h_b2 : 10 ≤ b) (h_b3 : b < 100) (h_prod : a * b = 5488) : a = 56 ∨ b = 56 :=
by {
  sorry
}

end product_of_two_digit_numbers_5488_has_smaller_number_56_l219_219796


namespace usual_time_to_office_l219_219917

theorem usual_time_to_office (S T : ℝ) (h : T = 4 / 3 * (T + 8)) : T = 24 :=
by
  sorry

end usual_time_to_office_l219_219917


namespace theta_quadrant_l219_219181

theorem theta_quadrant (θ : ℝ) (h : Real.sin (2 * θ) < 0) : 
  (Real.sin θ < 0 ∧ Real.cos θ > 0) ∨ (Real.sin θ > 0 ∧ Real.cos θ < 0) :=
sorry

end theta_quadrant_l219_219181


namespace sum_of_digits_of_N_l219_219036

theorem sum_of_digits_of_N (N : ℕ) (hN : N * (N + 1) / 2 = 3003) :
  (Nat.digits 10 N).sum = 14 :=
sorry

end sum_of_digits_of_N_l219_219036


namespace gcd_204_85_l219_219432

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end gcd_204_85_l219_219432


namespace max_a9_l219_219725

theorem max_a9 (a : Fin 18 → ℕ) (h_pos: ∀ i, 1 ≤ a i) (h_incr: ∀ i j, i < j → a i < a j) (h_sum: (Finset.univ : Finset (Fin 18)).sum a = 2001) : a 8 ≤ 192 :=
by
  -- Proof goes here
  sorry

end max_a9_l219_219725


namespace mean_of_sequence_starting_at_3_l219_219567

def arithmetic_sequence (start : ℕ) (n : ℕ) : List ℕ :=
List.range n |>.map (λ i => start + i)

def arithmetic_mean (seq : List ℕ) : ℚ := (seq.sum : ℚ) / seq.length

theorem mean_of_sequence_starting_at_3 : 
  ∀ (seq : List ℕ),
  seq = arithmetic_sequence 3 60 → 
  arithmetic_mean seq = 32.5 := 
by
  intros seq h
  rw [h]
  sorry

end mean_of_sequence_starting_at_3_l219_219567


namespace question_proof_l219_219149

open Set

variable (U : Set ℝ := univ)
variable (M : Set ℝ := {x | x < 1})
variable (N : Set ℝ := {x | -1 < x ∧ x < 2})

theorem question_proof : {x | x ≥ 2} = compl (M ∪ N) :=
by
  sorry

end question_proof_l219_219149


namespace find_a_l219_219647

theorem find_a (a : ℝ) (x : ℝ) (h : x^2 + a * x + 4 = (x + 2)^2) : a = 4 :=
sorry

end find_a_l219_219647


namespace find_rate_per_kg_mangoes_l219_219678

-- Definitions based on the conditions
def rate_per_kg_grapes : ℕ := 70
def quantity_grapes : ℕ := 8
def total_payment : ℕ := 1000
def quantity_mangoes : ℕ := 8

-- Proposition stating what we want to prove
theorem find_rate_per_kg_mangoes (r : ℕ) (H : total_payment = (rate_per_kg_grapes * quantity_grapes) + (r * quantity_mangoes)) : r = 55 := sorry

end find_rate_per_kg_mangoes_l219_219678


namespace sixth_grade_students_total_l219_219232

noncomputable def total_students (x y : ℕ) : ℕ := x + y

theorem sixth_grade_students_total (x y : ℕ) 
(h1 : x + (1 / 3) * y = 105) 
(h2 : y + (1 / 2) * x = 105) 
: total_students x y = 147 := 
by
  sorry

end sixth_grade_students_total_l219_219232


namespace savings_same_l219_219693

theorem savings_same (A_salary B_salary total_salary : ℝ)
  (A_spend_perc B_spend_perc : ℝ)
  (h_total : A_salary + B_salary = total_salary)
  (h_A_salary : A_salary = 4500)
  (h_A_spend_perc : A_spend_perc = 0.95)
  (h_B_spend_perc : B_spend_perc = 0.85)
  (h_total_salary : total_salary = 6000) :
  ((1 - A_spend_perc) * A_salary) = ((1 - B_spend_perc) * B_salary) :=
by
  sorry

end savings_same_l219_219693


namespace repeating_decimal_fraction_l219_219573

theorem repeating_decimal_fraction :
  ∃ (a b : ℕ), (0 ≤ a) ∧ (0 < b) ∧ (Nat.gcd a b = 1) ∧ (a + b = 134) ∧ 
  ((a : ℚ) / b = 35 / 99) :=
by {
  sorry
}

end repeating_decimal_fraction_l219_219573


namespace trig_sum_identity_l219_219278

theorem trig_sum_identity :
  Real.sin (20 * Real.pi / 180) + Real.sin (40 * Real.pi / 180) +
  Real.sin (60 * Real.pi / 180) - Real.sin (80 * Real.pi / 180) = Real.sqrt 3 / 2 := 
sorry

end trig_sum_identity_l219_219278


namespace min_value_expression_l219_219365

theorem min_value_expression (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_cond : x * y * z = 1/2) :
  x^3 + 4 * x * y + 16 * y^3 + 8 * y * z + 3 * z^3 ≥ 18 :=
sorry

end min_value_expression_l219_219365


namespace complement_of_A_is_negatives_l219_219312

theorem complement_of_A_is_negatives :
  let U := Set.univ (α := ℝ)
  let A := {x : ℝ | x ≥ 0}
  (U \ A) = {x : ℝ | x < 0} :=
by
  sorry

end complement_of_A_is_negatives_l219_219312


namespace revenue_change_l219_219882

theorem revenue_change (T C : ℝ) (T_new C_new : ℝ)
  (h1 : T_new = 0.81 * T)
  (h2 : C_new = 1.15 * C)
  (R : ℝ := T * C) : 
  ((T_new * C_new - R) / R) * 100 = -6.85 :=
by
  sorry

end revenue_change_l219_219882


namespace triangle_is_equilateral_l219_219986

theorem triangle_is_equilateral (a b c : ℝ) (h : a^2 + b^2 + c^2 = ab + ac + bc) : a = b ∧ b = c :=
by
  sorry

end triangle_is_equilateral_l219_219986


namespace cubic_sum_l219_219295

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by
  sorry

end cubic_sum_l219_219295


namespace female_lion_weight_l219_219747

theorem female_lion_weight (male_weight : ℚ) (weight_difference : ℚ) (female_weight : ℚ) : 
  male_weight = 145/4 → 
  weight_difference = 47/10 → 
  male_weight = female_weight + weight_difference → 
  female_weight = 631/20 :=
by
  intros h₁ h₂ h₃
  sorry

end female_lion_weight_l219_219747


namespace star_is_addition_l219_219081

theorem star_is_addition (star : ℝ → ℝ → ℝ) 
  (H : ∀ a b c : ℝ, star (star a b) c = a + b + c) : 
  ∀ a b : ℝ, star a b = a + b :=
by
  sorry

end star_is_addition_l219_219081


namespace suff_but_not_nec_l219_219031

-- Definition of proposition p
def p (m : ℝ) : Prop := m = -1

-- Definition of proposition q
def q (m : ℝ) : Prop := 
  let line1 := fun (x y : ℝ) => x - y = 0
  let line2 := fun (x y : ℝ) => x + (m^2) * y = 0
  ∀ (x1 y1 x2 y2 : ℝ), line1 x1 y1 → line2 x2 y2 → (x1 = x2 → y1 = -y2)

-- The proof problem
theorem suff_but_not_nec (m : ℝ) : p m → q m ∧ (q m → m = -1 ∨ m = 1) :=
sorry

end suff_but_not_nec_l219_219031


namespace sum_of_interior_angles_octagon_l219_219785

theorem sum_of_interior_angles_octagon : (8 - 2) * 180 = 1080 :=
by
  sorry

end sum_of_interior_angles_octagon_l219_219785


namespace money_left_after_expenses_l219_219895

theorem money_left_after_expenses : 
  let salary := 150000.00000000003
  let food := salary * (1 / 5)
  let house_rent := salary * (1 / 10)
  let clothes := salary * (3 / 5)
  let total_spent := food + house_rent + clothes
  let money_left := salary - total_spent
  money_left = 15000.00000000000 :=
by
  sorry

end money_left_after_expenses_l219_219895


namespace carlson_max_jars_l219_219144

theorem carlson_max_jars (n a : ℕ) (h1 : 13 * n - a = 8 * (n + a)) : 
  ∃ (k : ℕ), a = 5 * k ∧ n = 9 * k ∧ 13 * n = 117 * k ∧ 23 ≤ 13 * k := by {
  sorry
}

end carlson_max_jars_l219_219144


namespace distance_between_planes_l219_219679

open Real

def plane1 (x y z : ℝ) : Prop := 3 * x - y + 2 * z - 3 = 0
def plane2 (x y z : ℝ) : Prop := 6 * x - 2 * y + 4 * z + 4 = 0

theorem distance_between_planes :
  ∀ (x y z : ℝ), plane1 x y z →
  6 * x - 2 * y + 4 * z + 4 ≠ 0 →
  (∃ d : ℝ, d = abs (6 * x - 2 * y + 4 * z + 4) / sqrt (6^2 + (-2)^2 + 4^2) ∧ d = 5 * sqrt 14 / 14) :=
by
  intros x y z p1 p2
  sorry

end distance_between_planes_l219_219679


namespace distance_focus_directrix_l219_219066

theorem distance_focus_directrix (p : ℝ) (x_1 : ℝ) (h1 : 0 < p) (h2 : x_1^2 = 2 * p)
  (h3 : 1 + p / 2 = 3) : p = 4 :=
by
  sorry

end distance_focus_directrix_l219_219066


namespace domain_of_fx_l219_219931

theorem domain_of_fx :
  {x : ℝ | x ≥ 1 ∧ x^2 < 2} = {x : ℝ | 1 ≤ x ∧ x < Real.sqrt 2} := by
sorry

end domain_of_fx_l219_219931


namespace original_price_four_pack_l219_219901

theorem original_price_four_pack (price_with_rush: ℝ) (increase_rate: ℝ) (num_packs: ℕ):
  price_with_rush = 13 → increase_rate = 0.30 → num_packs = 4 → num_packs * (price_with_rush / (1 + increase_rate)) = 40 :=
by
  intros h_price h_rate h_packs
  rw [h_price, h_rate, h_packs]
  sorry

end original_price_four_pack_l219_219901


namespace length_of_interval_l219_219076

theorem length_of_interval (a b : ℝ) (h : 10 = (b - a) / 2) : b - a = 20 :=
by 
  sorry

end length_of_interval_l219_219076


namespace john_paint_area_l219_219849

noncomputable def area_to_paint (length width height openings : ℝ) : ℝ :=
  let wall_area := 2 * (length * height) + 2 * (width * height)
  let ceiling_area := length * width
  let total_area := wall_area + ceiling_area
  total_area - openings

theorem john_paint_area :
  let length := 15
  let width := 12
  let height := 10
  let openings := 70
  let bedrooms := 2
  2 * (area_to_paint length width height openings) = 1300 :=
by
  let length := 15
  let width := 12
  let height := 10
  let openings := 70
  let bedrooms := 2
  sorry

end john_paint_area_l219_219849


namespace arithmetic_geometric_sum_l219_219271

theorem arithmetic_geometric_sum (a : ℕ → ℚ) (S : ℕ → ℚ) (d : ℚ) (n : ℕ) (h1 : a 1 = 2)
  (h2 : a 3 = a 1 + 2 * d) (h3 : a 5 = a 1 + 4 * d) (h4 : (a 3) ^ 2 = a 1 * a 5)
  (h5 : d ≠ 0) : S n = (n^2 + 7 * n) / 4 := sorry

end arithmetic_geometric_sum_l219_219271


namespace ellipse_min_area_contains_circles_l219_219137

-- Define the ellipse and circles
def ellipse (x y : ℝ) := (x^2 / 16) + (y^2 / 9) = 1
def circle1 (x y : ℝ) := ((x - 2)^2 + y^2 = 4)
def circle2 (x y : ℝ) := ((x + 2)^2 + y^2 = 4)

-- Proof statement: The smallest possible area of the ellipse containing the circles
theorem ellipse_min_area_contains_circles : 
  ∃ (k : ℝ), 
  (∀ (x y : ℝ), 
    (circle1 x y → ellipse x y) ∧ 
    (circle2 x y → ellipse x y)) ∧
  (k = 12) := 
sorry

end ellipse_min_area_contains_circles_l219_219137


namespace find_hourly_rate_l219_219353

-- Defining the conditions
def hours_worked : ℝ := 7.5
def overtime_factor : ℝ := 1.5
def total_hours_worked : ℝ := 10.5
def total_earnings : ℝ := 48

-- Proving the hourly rate
theorem find_hourly_rate (R : ℝ) (h : 7.5 * R + (10.5 - 7.5) * 1.5 * R = 48) : R = 4 := by
  sorry

end find_hourly_rate_l219_219353


namespace admittedApplicants_l219_219577

-- Definitions for the conditions in the problem
def totalApplicants : ℕ := 70
def task1Applicants : ℕ := 35
def task2Applicants : ℕ := 48
def task3Applicants : ℕ := 64
def task4Applicants : ℕ := 63

-- The proof statement
theorem admittedApplicants : 
  ∀ (totalApplicants task3Applicants task4Applicants : ℕ),
  totalApplicants = 70 →
  task3Applicants = 64 →
  task4Applicants = 63 →
  ∃ (interApplicants : ℕ), interApplicants = 57 :=
by
  intros totalApplicants task3Applicants task4Applicants
  intros h_totalApps h_task3Apps h_task4Apps
  sorry

end admittedApplicants_l219_219577


namespace factor_expression_l219_219043

variable (x : ℝ)

theorem factor_expression : 
  (10 * x^3 + 50 * x^2 - 5) - (-5 * x^3 + 15 * x^2 - 5) = 5 * x^2 * (3 * x + 7) := 
by 
  sorry

end factor_expression_l219_219043


namespace rectangle_side_difference_l219_219362

theorem rectangle_side_difference (p d x y : ℝ) (h1 : 2 * x + 2 * y = p)
                                   (h2 : x^2 + y^2 = d^2)
                                   (h3 : x = 2 * y) :
    x - y = p / 6 := 
sorry

end rectangle_side_difference_l219_219362


namespace garden_perimeter_is_64_l219_219303

-- Define the playground dimensions and its area 
def playground_length := 16
def playground_width := 12
def playground_area := playground_length * playground_width

-- Define the garden width and its area being the same as the playground's area
def garden_width := 8
def garden_area := playground_area

-- Calculate the garden's length
def garden_length := garden_area / garden_width

-- Calculate the perimeter of the garden
def garden_perimeter := 2 * (garden_length + garden_width)

theorem garden_perimeter_is_64 :
  garden_perimeter = 64 := 
sorry

end garden_perimeter_is_64_l219_219303


namespace ratio_of_wealth_l219_219315

theorem ratio_of_wealth (W P : ℝ) 
  (h1 : 0 < P) (h2 : 0 < W) 
  (pop_X : ℝ := 0.4 * P) 
  (wealth_X : ℝ := 0.6 * W) 
  (top50_pop_X : ℝ := 0.5 * pop_X) 
  (top50_wealth_X : ℝ := 0.8 * wealth_X) 
  (pop_Y : ℝ := 0.2 * P) 
  (wealth_Y : ℝ := 0.3 * W) 
  (avg_wealth_top50_X : ℝ := top50_wealth_X / top50_pop_X) 
  (avg_wealth_Y : ℝ := wealth_Y / pop_Y) : 
  avg_wealth_top50_X / avg_wealth_Y = 1.6 := 
by sorry

end ratio_of_wealth_l219_219315


namespace triangle_condition_l219_219798

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin x) * (Real.cos x) + (Real.sqrt 3) * (Real.cos x) ^ 2 - (Real.sqrt 3) / 2

theorem triangle_condition (a b c : ℝ) (h : b^2 + c^2 = a^2 + Real.sqrt 3 * b * c) : 
  f (Real.pi / 6) = Real.sqrt 3 / 2 := by
  sorry

end triangle_condition_l219_219798


namespace inequality_transform_l219_219255

theorem inequality_transform {a b : ℝ} (h : a < b) : -2 + 2 * a < -2 + 2 * b :=
sorry

end inequality_transform_l219_219255


namespace journey_time_approx_24_hours_l219_219625

noncomputable def journey_time_in_hours : ℝ :=
  let t1 := 70 / 60  -- time for destination 1
  let t2 := 50 / 35  -- time for destination 2
  let t3 := 20 / 60 + 20 / 30  -- time for destination 3
  let t4 := 30 / 40 + 60 / 70  -- time for destination 4
  let t5 := 60 / 35  -- time for destination 5
  let return_distance := 70 + 50 + 40 + 90 + 60 + 100  -- total return distance
  let return_time := return_distance / 55  -- time for return journey
  let stay_time := 1 + 3 + 2 + 2.5 + 0.75  -- total stay time
  t1 + t2 + t3 + t4 + t5 + return_time + stay_time  -- total journey time

theorem journey_time_approx_24_hours : abs (journey_time_in_hours - 24) < 1 :=
by
  sorry

end journey_time_approx_24_hours_l219_219625


namespace verify_first_rope_length_l219_219919

def length_first_rope : ℝ :=
  let rope1_len := 20
  let rope2_len := 2
  let rope3_len := 2
  let rope4_len := 2
  let rope5_len := 7
  let knots := 4
  let knot_loss := 1.2
  let total_len := 35
  rope1_len

theorem verify_first_rope_length : length_first_rope = 20 := by
  sorry

end verify_first_rope_length_l219_219919


namespace centroids_coincide_l219_219764

noncomputable def centroid (A B C : ℂ) : ℂ :=
  (A + B + C) / 3

theorem centroids_coincide (A B C : ℂ) (k : ℝ) (C1 A1 B1 : ℂ)
  (h1 : C1 = k * (B - A) + A)
  (h2 : A1 = k * (C - B) + B)
  (h3 : B1 = k * (A - C) + C) :
  centroid A1 B1 C1 = centroid A B C := by
  sorry

end centroids_coincide_l219_219764


namespace simplify_and_evaluate_evaluate_at_zero_l219_219790

noncomputable def simplified_expression (x : ℤ) : ℚ :=
  (1 - 1/(x-1)) / ((x^2 - 4*x + 4) / (x^2 - 1))

theorem simplify_and_evaluate (x : ℤ) (h : x ≠ 1 ∧ x ≠ 2 ∧ x ≠ -1) : 
  simplified_expression x = (x+1)/(x-2) :=
by
  sorry

theorem evaluate_at_zero : simplified_expression 0 = -1/2 :=
by
  sorry

end simplify_and_evaluate_evaluate_at_zero_l219_219790


namespace prime_number_conditions_l219_219472

theorem prime_number_conditions :
  ∃ p n : ℕ, Prime p ∧ p = n^2 + 9 ∧ p = (n+1)^2 - 8 :=
by
  sorry

end prime_number_conditions_l219_219472


namespace quadratic_parabola_equation_l219_219971

theorem quadratic_parabola_equation :
  ∃ (a b c : ℝ), 
    (∀ x y, y = 3 * x^2 - 6 * x + 5 → (x - 1)*(x - 1) = (x - 1)^2) ∧ -- Original vertex condition and standard form
    (∀ x y, y = -x - 2 → a = 2) ∧ -- Given intersection point condition
    (∀ x y, y = -3 * (x - 1)^2 + 2 → y = -3 * (x - 1)^2 + b ∧ y = -4) → -- Vertex unchanged and direction reversed
    (a, b, c) = (-3, 6, -4) := -- Resulting equation coefficients
sorry

end quadratic_parabola_equation_l219_219971


namespace diameter_percentage_l219_219058

theorem diameter_percentage (d_R d_S : ℝ) (h : π * (d_R / 2)^2 = 0.25 * π * (d_S / 2)^2) : 
  d_R = 0.5 * d_S :=
by 
  sorry

end diameter_percentage_l219_219058


namespace molecular_weight_6_moles_C4H8O2_is_528_624_l219_219962

-- Define the atomic weights of Carbon, Hydrogen, and Oxygen.
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Define the molecular formula of C4H8O2.
def num_C_atoms : ℕ := 4
def num_H_atoms : ℕ := 8
def num_O_atoms : ℕ := 2

-- Define the number of moles of C4H8O2.
def num_moles_C4H8O2 : ℝ := 6

-- Define the molecular weight of one mole of C4H8O2.
def molecular_weight_C4H8O2 : ℝ :=
  (num_C_atoms * atomic_weight_C) +
  (num_H_atoms * atomic_weight_H) +
  (num_O_atoms * atomic_weight_O)

-- The total weight of 6 moles of C4H8O2.
def total_weight_6_moles_C4H8O2 : ℝ :=
  num_moles_C4H8O2 * molecular_weight_C4H8O2

-- Theorem stating that the molecular weight of 6 moles of C4H8O2 is 528.624 grams.
theorem molecular_weight_6_moles_C4H8O2_is_528_624 :
  total_weight_6_moles_C4H8O2 = 528.624 :=
by
  -- Proof is omitted.
  sorry

end molecular_weight_6_moles_C4H8O2_is_528_624_l219_219962


namespace repetend_of_five_over_eleven_l219_219176

noncomputable def repetend_of_decimal_expansion (n d : ℕ) : ℕ := sorry

theorem repetend_of_five_over_eleven : repetend_of_decimal_expansion 5 11 = 45 :=
by sorry

end repetend_of_five_over_eleven_l219_219176


namespace trig_intersection_identity_l219_219545

theorem trig_intersection_identity (x0 : ℝ) (hx0 : x0 ≠ 0) (htan : -x0 = Real.tan x0) :
  (x0^2 + 1) * (1 + Real.cos (2 * x0)) = 2 := 
sorry

end trig_intersection_identity_l219_219545


namespace fraction_classification_l219_219325

theorem fraction_classification (x y : ℤ) :
  (∃ a b : ℤ, a/b = x/(x+1)) ∧ ¬(∃ a b : ℤ, a/b = x/2 + 1) ∧ ¬(∃ a b : ℤ, a/b = x/2) ∧ ¬(∃ a b : ℤ, a/b = xy/3) :=
by sorry

end fraction_classification_l219_219325


namespace rectangle_divided_into_13_squares_l219_219911

theorem rectangle_divided_into_13_squares (s a b : ℕ) (h₁ : a * b = 13 * s^2)
  (h₂ : ∃ k l : ℕ, a = k * s ∧ b = l * s ∧ k * l = 13) :
  (a = s ∧ b = 13 * s) ∨ (a = 13 * s ∧ b = s) :=
by
sorry

end rectangle_divided_into_13_squares_l219_219911


namespace initial_ratio_of_milk_to_water_l219_219872

-- Define the capacity of the can, the amount of milk added, and the ratio when full.
def capacity : ℕ := 72
def additionalMilk : ℕ := 8
def fullRatioNumerator : ℕ := 2
def fullRatioDenominator : ℕ := 1

-- Define the initial amounts of milk and water in the can.
variables (M W : ℕ)

-- Define the conditions given in the problem.
def conditions : Prop :=
  M + W + additionalMilk = capacity ∧
  (M + additionalMilk) * fullRatioDenominator = fullRatioNumerator * W

-- Define the expected result, the initial ratio of milk to water in the can.
def expected_ratio : ℕ × ℕ :=
  (5, 3)

-- The theorem to prove the initial ratio of milk to water given the conditions.
theorem initial_ratio_of_milk_to_water (M W : ℕ) (h : conditions M W) :
  (M / Nat.gcd M W, W / Nat.gcd M W) = expected_ratio :=
sorry

end initial_ratio_of_milk_to_water_l219_219872


namespace probability_first_spade_second_ace_l219_219000

theorem probability_first_spade_second_ace :
  let n : ℕ := 52
  let spades : ℕ := 13
  let aces : ℕ := 4
  let ace_of_spades : ℕ := 1
  let non_ace_spades : ℕ := spades - ace_of_spades
  (non_ace_spades / n : ℚ) * (aces / (n - 1) : ℚ) +
  (ace_of_spades / n : ℚ) * ((aces - 1) / (n - 1) : ℚ) =
  (1 / n : ℚ) :=
by {
  -- proof goes here
  sorry
}

end probability_first_spade_second_ace_l219_219000


namespace max_profit_l219_219662

theorem max_profit : ∃ v p : ℝ, 
  v + p ≤ 5 ∧
  v + 3 * p ≤ 12 ∧
  100000 * v + 200000 * p = 850000 :=
by
  sorry

end max_profit_l219_219662


namespace part_a_part_b_part_c_part_d_part_e_part_f_l219_219603

-- Part (a)
theorem part_a (n : ℤ) (h : ¬ ∃ k : ℤ, n = 5 * k) : ∃ k : ℤ, n^2 = 5 * k + 1 ∨ n^2 = 5 * k - 1 := 
sorry

-- Part (b)
theorem part_b (n : ℤ) (h : ¬ ∃ k : ℤ, n = 5 * k) : ∃ k : ℤ, n^4 - 1 = 5 * k := 
sorry

-- Part (c)
theorem part_c (n : ℤ) : n^5 % 10 = n % 10 := 
sorry

-- Part (d)
theorem part_d (n : ℤ) : ∃ k : ℤ, n^5 - n = 30 * k := 
sorry

-- Part (e)
theorem part_e (k n : ℤ) (h1 : ¬ ∃ j : ℤ, k = 5 * j) (h2 : ¬ ∃ j : ℤ, n = 5 * j) : ∃ j : ℤ, k^4 - n^4 = 5 * j := 
sorry

-- Part (f)
theorem part_f (k m n : ℤ) (h : k^2 + m^2 = n^2) : ∃ j : ℤ, k = 5 * j ∨ ∃ r : ℤ, m = 5 * r ∨ ∃ s : ℤ, n = 5 * s := 
sorry

end part_a_part_b_part_c_part_d_part_e_part_f_l219_219603


namespace total_books_l219_219539

variable (M K G : ℕ)

-- Conditions
def Megan_books := 32
def Kelcie_books := Megan_books / 4
def Greg_books := 2 * Kelcie_books + 9

-- Theorem to prove
theorem total_books : Megan_books + Kelcie_books + Greg_books = 65 := by
  unfold Megan_books Kelcie_books Greg_books
  sorry

end total_books_l219_219539


namespace parabola_behavior_l219_219187

-- Definitions for the conditions
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2

-- The proof statement
theorem parabola_behavior (a : ℝ) (x : ℝ) (ha : 0 < a) : 
  (0 < a ∧ a < 1 → parabola a x < x^2) ∧
  (a > 1 → parabola a x > x^2) ∧
  (∀ ε > 0, ∃ δ > 0, δ ≤ a → |parabola a x - 0| < ε) := 
sorry

end parabola_behavior_l219_219187


namespace teams_same_matches_l219_219321

theorem teams_same_matches (n : ℕ) (h : n = 30) : ∃ (i j : ℕ), i ≠ j ∧ ∀ (m : ℕ), m ≤ n - 1 → (some_number : ℕ) = (some_number : ℕ) :=
by {
  sorry
}

end teams_same_matches_l219_219321


namespace total_volume_of_mixed_solutions_l219_219454

theorem total_volume_of_mixed_solutions :
  let v1 := 3.6
  let v2 := 1.4
  v1 + v2 = 5.0 := by
  sorry

end total_volume_of_mixed_solutions_l219_219454


namespace solve_for_x_l219_219375

def delta (x : ℝ) : ℝ := 5 * x + 9
def phi (x : ℝ) : ℝ := 7 * x + 6

theorem solve_for_x (x : ℝ) (h : delta (phi x) = -4) : x = -43 / 35 :=
by
  sorry

end solve_for_x_l219_219375


namespace find_sum_x_y_l219_219174

theorem find_sum_x_y (x y : ℝ) 
  (h1 : x^3 - 3 * x^2 + 2026 * x = 2023)
  (h2 : y^3 + 6 * y^2 + 2035 * y = -4053) : 
  x + y = -1 := 
sorry

end find_sum_x_y_l219_219174


namespace largest_angle_in_ratio_triangle_l219_219813

theorem largest_angle_in_ratio_triangle (x : ℝ) (h1 : 3 * x + 4 * x + 5 * x = 180) : 
  5 * (180 / (3 + 4 + 5)) = 75 := by
  sorry

end largest_angle_in_ratio_triangle_l219_219813


namespace total_difference_is_correct_l219_219738

-- Define the harvest rates
def valencia_weekday_ripe := 90
def valencia_weekday_unripe := 38
def navel_weekday_ripe := 125
def navel_weekday_unripe := 65
def blood_weekday_ripe := 60
def blood_weekday_unripe := 42

def valencia_weekend_ripe := 75
def valencia_weekend_unripe := 33
def navel_weekend_ripe := 100
def navel_weekend_unripe := 57
def blood_weekend_ripe := 45
def blood_weekend_unripe := 36

-- Define the number of weekdays and weekend days
def weekdays := 5
def weekend_days := 2

-- Calculate the total harvests
def total_valencia_ripe := valencia_weekday_ripe * weekdays + valencia_weekend_ripe * weekend_days
def total_valencia_unripe := valencia_weekday_unripe * weekdays + valencia_weekend_unripe * weekend_days
def total_navel_ripe := navel_weekday_ripe * weekdays + navel_weekend_ripe * weekend_days
def total_navel_unripe := navel_weekday_unripe * weekdays + navel_weekend_unripe * weekend_days
def total_blood_ripe := blood_weekday_ripe * weekdays + blood_weekend_ripe * weekend_days
def total_blood_unripe := blood_weekday_unripe * weekdays + blood_weekend_unripe * weekend_days

-- Calculate the total differences
def valencia_difference := total_valencia_ripe - total_valencia_unripe
def navel_difference := total_navel_ripe - total_navel_unripe
def blood_difference := total_blood_ripe - total_blood_unripe

-- Define the total difference
def total_difference := valencia_difference + navel_difference + blood_difference

-- Theorem statement
theorem total_difference_is_correct :
  total_difference = 838 := by
  sorry

end total_difference_is_correct_l219_219738


namespace positive_integers_condition_l219_219578

theorem positive_integers_condition : ∃ n : ℕ, (n > 0) ∧ (n < 50) ∧ (∃ k : ℕ, n = k * (50 - n)) :=
sorry

end positive_integers_condition_l219_219578


namespace l_shape_area_l219_219257

theorem l_shape_area (large_length large_width small_length small_width : ℕ)
  (large_rect_area : large_length = 10 ∧ large_width = 7)
  (small_rect_area : small_length = 3 ∧ small_width = 2) :
  (large_length * large_width) - 2 * (small_length * small_width) = 58 :=
by 
  sorry

end l_shape_area_l219_219257


namespace number_of_students_selected_from_school2_l219_219159

-- Definitions from conditions
def total_students : ℕ := 360
def students_school1 : ℕ := 123
def students_school2 : ℕ := 123
def students_school3 : ℕ := 114
def selected_students : ℕ := 60
def initial_selected_from_school1 : ℕ := 1 -- Student 002 is already selected

-- Proportion calculation
def remaining_selected_students : ℕ := selected_students - initial_selected_from_school1
def remaining_students : ℕ := total_students - initial_selected_from_school1

-- Placeholder for calculation used in the proof
def students_selected_from_school2 : ℕ := 20

-- The Lean proof statement
theorem number_of_students_selected_from_school2 :
  students_selected_from_school2 =
  Nat.ceil ((students_school2 * remaining_selected_students : ℚ) / remaining_students) :=
sorry

end number_of_students_selected_from_school2_l219_219159


namespace area_of_triangle_MAB_l219_219862

noncomputable def triangle_area (A B M : ℝ × ℝ) : ℝ :=
  0.5 * ((B.1 - A.1) * (M.2 - A.2) - (M.1 - A.1) * (B.2 - A.2))

theorem area_of_triangle_MAB :
  let C1 (p : ℝ × ℝ) := p.1^2 - p.2^2 = 2
  let C2 (p : ℝ × ℝ) := ∃ θ, p.1 = 2 + 2 * Real.cos θ ∧ p.2 = 2 * Real.sin θ
  let M := (3.0, 0.0)
  let A := (2, 2 * Real.sin (Real.pi / 6))
  let B := (2 * Real.sqrt 3, 2 * Real.sin (Real.pi / 6))
  triangle_area A B M = (3 * Real.sqrt 3 - 3) / 2 :=
by
  sorry

end area_of_triangle_MAB_l219_219862


namespace sales_on_same_days_l219_219859

-- Definitions representing the conditions
def bookstore_sales_days : List ℕ := [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
def toy_store_sales_days : List ℕ := [2, 9, 16, 23, 30]

-- Lean statement to prove the number of common sale days
theorem sales_on_same_days : (bookstore_sales_days ∩ toy_store_sales_days).length = 2 :=
by sorry

end sales_on_same_days_l219_219859


namespace side_length_of_square_l219_219475

-- Define the areas of the triangles AOR, BOP, and CRQ
def S1 := 1
def S2 := 3
def S3 := 1

-- Prove that the side length of the square OPQR is 2
theorem side_length_of_square (side_length : ℝ) : 
  S1 = 1 ∧ S2 = 3 ∧ S3 = 1 → side_length = 2 :=
by
  intros h
  sorry

end side_length_of_square_l219_219475


namespace Cinderella_solves_l219_219974

/--
There are three bags labeled as "Poppy", "Millet", and "Mixture". Each label is incorrect.
By inspecting one grain from the bag labeled as "Mixture", Cinderella can determine the exact contents of all three bags.
-/
theorem Cinderella_solves (bag_contents : String → String) (examined_grain : String) :
  (bag_contents "Mixture" = "Poppy" ∨ bag_contents "Mixture" = "Millet") →
  (∀ l, bag_contents l ≠ l) →
  (examined_grain = "Poppy" ∨ examined_grain = "Millet") →
  examined_grain = bag_contents "Mixture" →
  ∃ poppy_bag millet_bag mixture_bag : String,
    poppy_bag ≠ "Poppy" ∧ millet_bag ≠ "Millet" ∧ mixture_bag ≠ "Mixture" ∧
    bag_contents poppy_bag = "Poppy" ∧
    bag_contents millet_bag = "Millet" ∧
    bag_contents mixture_bag = "Mixture" :=
sorry

end Cinderella_solves_l219_219974


namespace factor_complete_polynomial_l219_219107

theorem factor_complete_polynomial :
  5 * (x + 3) * (x + 7) * (x + 11) * (x + 13) - 4 * x^2 =
  (5 * x^2 + 94 * x + 385) * (x^2 - 20 * x + 77) :=
sorry

end factor_complete_polynomial_l219_219107


namespace apples_total_l219_219308

theorem apples_total (Benny_picked Dan_picked : ℕ) (hB : Benny_picked = 2) (hD : Dan_picked = 9) : Benny_picked + Dan_picked = 11 :=
by
  -- Definitions
  sorry

end apples_total_l219_219308


namespace factorial_square_gt_power_l219_219358

theorem factorial_square_gt_power {n : ℕ} (h : n > 2) : (n! * n!) > n^n :=
sorry

end factorial_square_gt_power_l219_219358


namespace total_hiking_distance_l219_219841

def saturday_distance : ℝ := 8.2
def sunday_distance : ℝ := 1.6
def total_distance (saturday_distance sunday_distance : ℝ) : ℝ := saturday_distance + sunday_distance

theorem total_hiking_distance :
  total_distance saturday_distance sunday_distance = 9.8 :=
by
  -- The proof is omitted
  sorry

end total_hiking_distance_l219_219841


namespace price_of_cheaper_book_l219_219343

theorem price_of_cheaper_book
    (total_cost : ℕ)
    (sets : ℕ)
    (price_more_expensive_book_increase : ℕ)
    (h1 : total_cost = 21000)
    (h2 : sets = 3)
    (h3 : price_more_expensive_book_increase = 300) :
  ∃ x : ℕ, 3 * ((x + (x + price_more_expensive_book_increase))) = total_cost ∧ x = 3350 :=
by
  sorry

end price_of_cheaper_book_l219_219343


namespace range_of_a_l219_219571

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 5 → a * x^2 - x - 4 > 0) → a > 5 :=
by
  sorry

end range_of_a_l219_219571


namespace age_ratio_l219_219086

noncomputable def rahul_present_age (future_age : ℕ) (years_passed : ℕ) : ℕ := future_age - years_passed

theorem age_ratio (future_rahul_age : ℕ) (years_passed : ℕ) (deepak_age : ℕ) :
  future_rahul_age = 26 →
  years_passed = 6 →
  deepak_age = 15 →
  rahul_present_age future_rahul_age years_passed / deepak_age = 4 / 3 :=
by
  intros
  have h1 : rahul_present_age 26 6 = 20 := rfl
  sorry

end age_ratio_l219_219086


namespace train_length_proof_l219_219992

-- Defining the conditions
def speed_kmph : ℕ := 72
def platform_length : ℕ := 250  -- in meters
def time_seconds : ℕ := 26

-- Conversion factor from kmph to m/s
def kmph_to_mps (v : ℕ) : ℕ := (v * 1000) / 3600

-- The main goal: the length of the train
def train_length (speed_kmph : ℕ) (platform_length : ℕ) (time_seconds : ℕ) : ℕ :=
  let speed_mps := kmph_to_mps speed_kmph
  let total_distance := speed_mps * time_seconds
  total_distance - platform_length

theorem train_length_proof : train_length speed_kmph platform_length time_seconds = 270 := 
by 
  unfold train_length kmph_to_mps
  sorry

end train_length_proof_l219_219992


namespace largest_k_sum_of_consecutive_odds_l219_219012

theorem largest_k_sum_of_consecutive_odds (k m : ℕ) (h1 : k * (2 * m + k) = 2^15) : k ≤ 128 :=
by {
  sorry
}

end largest_k_sum_of_consecutive_odds_l219_219012


namespace max_gold_coins_l219_219189

theorem max_gold_coins (n k : ℕ) (h1 : n = 13 * k + 3) (h2 : n < 100) : n = 94 := by
  sorry

end max_gold_coins_l219_219189


namespace total_animals_l219_219155

theorem total_animals (initial_elephants initial_hippos : ℕ) 
  (ratio_female_hippos : ℚ)
  (births_per_female_hippo : ℕ)
  (newborn_elephants_diff : ℕ)
  (he : initial_elephants = 20)
  (hh : initial_hippos = 35)
  (rfh : ratio_female_hippos = 5 / 7)
  (bpfh : births_per_female_hippo = 5)
  (ned : newborn_elephants_diff = 10) :
  ∃ (total_animals : ℕ), total_animals = 315 :=
by sorry

end total_animals_l219_219155


namespace greatest_integer_gcd_30_is_125_l219_219447

theorem greatest_integer_gcd_30_is_125 : ∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ ∀ k : ℕ, k < 150 ∧ Nat.gcd k 30 = 5 → k ≤ n := 
sorry

end greatest_integer_gcd_30_is_125_l219_219447


namespace min_value_fraction_l219_219885

variable {a b : ℝ}

theorem min_value_fraction (h₁ : a + b = 1) (ha : a > 0) (hb : b > 0) : 
  (1 / a + 4 / b) ≥ 9 :=
sorry

end min_value_fraction_l219_219885


namespace triangle_side_eq_nine_l219_219106

theorem triangle_side_eq_nine (a b c : ℕ) 
  (h_tri_ineq : a + b > c ∧ a + c > b ∧ b + c > a)
  (h_sqrt_eq : (Nat.sqrt (a - 9)) + (b - 2)^2 = 0)
  (h_c_odd : c % 2 = 1) :
  c = 9 :=
sorry

end triangle_side_eq_nine_l219_219106


namespace parabola_vertex_on_x_axis_l219_219193

theorem parabola_vertex_on_x_axis (c : ℝ) : 
    (∃ h k, h = -3 ∧ k = 0 ∧ ∀ x, x^2 + 6 * x + c = x^2 + 6 * x + (c - (h^2)/4)) → c = 9 :=
by
    sorry

end parabola_vertex_on_x_axis_l219_219193


namespace correct_operation_l219_219832

theorem correct_operation (x y : ℝ) : (x^3 * y^2 - y^2 * x^3 = 0) :=
by sorry

end correct_operation_l219_219832


namespace divisible_by_12_l219_219242

theorem divisible_by_12 (n : ℕ) (h1 : (5140 + n) % 4 = 0) (h2 : (5 + 1 + 4 + n) % 3 = 0) : n = 8 :=
by
  sorry

end divisible_by_12_l219_219242


namespace stratified_sampling_group_C_l219_219430

theorem stratified_sampling_group_C
  (total_cities : ℕ)
  (cities_group_A : ℕ)
  (cities_group_B : ℕ)
  (cities_group_C : ℕ)
  (total_selected : ℕ)
  (C_subset_correct: total_cities = cities_group_A + cities_group_B + cities_group_C)
  (total_cities_correct: total_cities = 48)
  (cities_group_A_correct: cities_group_A = 8)
  (cities_group_B_correct: cities_group_B = 24)
  (total_selected_correct: total_selected = 12)
  : (total_selected * cities_group_C) / total_cities = 4 :=
by 
  sorry

end stratified_sampling_group_C_l219_219430


namespace ways_to_divide_day_l219_219092

theorem ways_to_divide_day (n m : ℕ) (h : n * m = 86400) : 
  (∃ k : ℕ, k = 96) :=
  sorry

end ways_to_divide_day_l219_219092


namespace parameter_a_solution_exists_l219_219095

theorem parameter_a_solution_exists (a : ℝ) : 
  (a < -2 / 3 ∨ a > 0) → ∃ b x y : ℝ, 
  x = 6 / a - abs (y - a) ∧ x^2 + y^2 + b^2 + 63 = 2 * (b * y - 8 * x) :=
by
  intro h
  sorry

end parameter_a_solution_exists_l219_219095


namespace range_of_m_l219_219020

variable (x m : ℝ)

theorem range_of_m (h1 : ∀ x : ℝ, 2 * x^2 - 2 * m * x + m < 0) 
    (h2 : ∃ a b : ℤ, a ≠ b ∧ ∀ x : ℝ, (a < x ∧ x < b) → 2 * x^2 - 2 * m * x + m < 0): 
    -8 / 5 ≤ m ∧ m < -2 / 3 ∨ 8 / 3 < m ∧ m ≤ 18 / 5 :=
sorry

end range_of_m_l219_219020


namespace complement_of_alpha_l219_219728

-- Define that the angle α is given as 44 degrees 36 minutes
def alpha : ℚ := 44 + 36 / 60  -- using rational numbers to represent the degrees and minutes

-- Define the complement function
def complement (angle : ℚ) : ℚ := 90 - angle

-- State the proposition to prove
theorem complement_of_alpha : complement alpha = 45 + 24 / 60 := 
by
  sorry

end complement_of_alpha_l219_219728


namespace red_star_team_wins_l219_219115

theorem red_star_team_wins (x y : ℕ) (h1 : x + y = 9) (h2 : 3 * x + y = 23) : x = 7 := by
  sorry

end red_star_team_wins_l219_219115


namespace fraction_B_A_C_l219_219072

theorem fraction_B_A_C (A B C : ℕ) (x : ℚ) 
  (h1 : A = (1 / 3) * (B + C)) 
  (h2 : A = B + 10) 
  (h3 : A + B + C = 360) : 
  x = 2 / 7 ∧ B = x * (A + C) :=
by
  sorry -- The proof steps can be filled in

end fraction_B_A_C_l219_219072


namespace compute_xy_l219_219240

theorem compute_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h1 : x ^ (Real.sqrt y) = 27) (h2 : (Real.sqrt x) ^ y = 9) :
  x * y = 12 * Real.sqrt 3 :=
sorry

end compute_xy_l219_219240


namespace probability_both_selected_l219_219204

def probability_selection_ram : ℚ := 4 / 7
def probability_selection_ravi : ℚ := 1 / 5

theorem probability_both_selected : probability_selection_ram * probability_selection_ravi = 4 / 35 := 
by 
  -- Proof goes here
  sorry

end probability_both_selected_l219_219204


namespace y_squared_plus_three_y_is_perfect_square_l219_219720

theorem y_squared_plus_three_y_is_perfect_square (y : ℕ) :
  (∃ x : ℕ, y^2 + 3^y = x^2) ↔ y = 1 ∨ y = 3 := 
by
  sorry

end y_squared_plus_three_y_is_perfect_square_l219_219720


namespace translate_upwards_l219_219263

theorem translate_upwards (x : ℝ) : (2 * x^2) + 2 = 2 * x^2 + 2 := by
  sorry

end translate_upwards_l219_219263


namespace apples_vs_cherries_l219_219907

def pies_per_day : Nat := 12
def apple_days_per_week : Nat := 3
def cherry_days_per_week : Nat := 2

theorem apples_vs_cherries :
  (apple_days_per_week * pies_per_day) - (cherry_days_per_week * pies_per_day) = 12 := by
  sorry

end apples_vs_cherries_l219_219907


namespace total_hours_A_ascending_and_descending_l219_219670

theorem total_hours_A_ascending_and_descending
  (ascending_speed_A ascending_speed_B descending_speed_A descending_speed_B distance summit_distance : ℝ)
  (h1 : descending_speed_A = 1.5 * ascending_speed_A)
  (h2 : descending_speed_B = 1.5 * ascending_speed_B)
  (h3 : ascending_speed_A > ascending_speed_B)
  (h4 : 1/ascending_speed_A + 1/ascending_speed_B = 1/hour - 600/summit_distance)
  (h5 : 0.5 * summit_distance/ascending_speed_A = (summit_distance - 600)/ascending_speed_B) :
  (summit_distance / ascending_speed_A) + (summit_distance / descending_speed_A) = 1.5 := 
sorry

end total_hours_A_ascending_and_descending_l219_219670


namespace chef_served_173_guests_l219_219182

noncomputable def total_guests_served : ℕ :=
  let adults := 58
  let children := adults - 35
  let seniors := 2 * children
  let teenagers := seniors - 15
  let toddlers := teenagers / 2
  adults + children + seniors + teenagers + toddlers

theorem chef_served_173_guests : total_guests_served = 173 :=
  by
    -- Proof will be provided here.
    sorry

end chef_served_173_guests_l219_219182


namespace smallest_n_for_gn_gt_20_l219_219600

def g (n : ℕ) : ℕ := sorry -- definition of the sum of the digits to the right of the decimal of 1 / 3^n

theorem smallest_n_for_gn_gt_20 : ∃ n : ℕ, n > 0 ∧ g n > 20 ∧ ∀ m, 0 < m ∧ m < n -> g m ≤ 20 :=
by
  -- here should be the proof
  sorry

end smallest_n_for_gn_gt_20_l219_219600


namespace sum_of_d_and_e_l219_219977

-- Define the original numbers and their sum
def original_first := 3742586
def original_second := 4829430
def correct_sum := 8572016

-- The given incorrect addition result
def given_sum := 72120116

-- Define the digits d and e
def d := 2
def e := 8

-- Define the correct adjusted sum if we replace d with e
def adjusted_first := 3782586
def adjusted_second := 4889430
def adjusted_sum := 8672016

-- State the final theorem
theorem sum_of_d_and_e : 
  (given_sum != correct_sum) → 
  (original_first + original_second = correct_sum) → 
  (adjusted_first + adjusted_second = adjusted_sum) → 
  (d + e = 10) :=
by
  sorry

end sum_of_d_and_e_l219_219977


namespace bahs_from_yahs_l219_219468

theorem bahs_from_yahs (b r y : ℝ) 
  (h1 : 18 * b = 30 * r) 
  (h2 : 10 * r = 25 * y) : 
  1250 * y = 300 * b := 
by
  sorry

end bahs_from_yahs_l219_219468


namespace coterminal_angle_equivalence_l219_219606

theorem coterminal_angle_equivalence (k : ℤ) : ∃ n : ℤ, -463 % 360 = (k * 360 + 257) % 360 :=
by
  sorry

end coterminal_angle_equivalence_l219_219606


namespace function_even_l219_219845

theorem function_even (n : ℤ) (h : 30 ∣ n)
    (h_prop: (1 : ℝ)^n^2 + (-1: ℝ)^n^2 = 2 * ((1: ℝ)^n + (-1: ℝ)^n - 1)) :
    ∀ x : ℝ, (x^n = (-x)^n) :=
by
    sorry

end function_even_l219_219845


namespace henry_time_around_track_l219_219378

theorem henry_time_around_track (H : ℕ) : 
  (∀ (M := 12), lcm M H = 84) → H = 7 :=
by
  sorry

end henry_time_around_track_l219_219378


namespace probability_red_blue_green_l219_219518

def total_marbles : ℕ := 5 + 4 + 3 + 6
def favorable_marbles : ℕ := 5 + 4 + 3

theorem probability_red_blue_green : 
  (favorable_marbles : ℚ) / total_marbles = 2 / 3 := 
by 
  sorry

end probability_red_blue_green_l219_219518


namespace work_together_days_l219_219298

theorem work_together_days (ravi_days prakash_days : ℕ) (hr : ravi_days = 50) (hp : prakash_days = 75) : 
  (ravi_days * prakash_days) / (ravi_days + prakash_days) = 30 :=
sorry

end work_together_days_l219_219298


namespace additional_laps_needed_l219_219077

-- Definitions of problem conditions
def total_required_distance : ℕ := 2400
def lap_length : ℕ := 150
def madison_laps : ℕ := 6
def gigi_laps : ℕ := 6

-- Target statement to prove the number of additional laps needed
theorem additional_laps_needed : (total_required_distance - (madison_laps + gigi_laps) * lap_length) / lap_length = 4 := by
  sorry

end additional_laps_needed_l219_219077


namespace remaining_customers_is_13_l219_219730

-- Given conditions
def initial_customers : ℕ := 36
def half_left_customers : ℕ := initial_customers / 2  -- 50% of customers leaving
def remaining_customers_after_half_left : ℕ := initial_customers - half_left_customers

def thirty_percent_of_remaining : ℚ := remaining_customers_after_half_left * 0.30 
def thirty_percent_of_remaining_rounded : ℕ := thirty_percent_of_remaining.floor.toNat  -- rounding down

def final_remaining_customers : ℕ := remaining_customers_after_half_left - thirty_percent_of_remaining_rounded

-- Proof statement without proof
theorem remaining_customers_is_13 : final_remaining_customers = 13 := by
  sorry

end remaining_customers_is_13_l219_219730


namespace total_cost_of_backpack_and_pencil_case_l219_219767

-- Definitions based on the given conditions
def pencil_case_price : ℕ := 8
def backpack_price : ℕ := 5 * pencil_case_price

-- Statement of the proof problem
theorem total_cost_of_backpack_and_pencil_case : 
  pencil_case_price + backpack_price = 48 :=
by
  -- Skip the proof
  sorry

end total_cost_of_backpack_and_pencil_case_l219_219767


namespace star_comm_star_distrib_over_add_star_special_case_star_no_identity_star_not_assoc_l219_219812

def star (x y : ℤ) := (x + 2) * (y + 2) - 2

-- Statement A: commutativity
theorem star_comm : ∀ x y : ℤ, star x y = star y x := 
by sorry

-- Statement B: distributivity over addition
theorem star_distrib_over_add : ¬(∀ x y z : ℤ, star x (y + z) = star x y + star x z) :=
by sorry

-- Statement C: special case
theorem star_special_case : ¬(∀ x : ℤ, star (x - 2) (x + 2) = star x x - 2) :=
by sorry

-- Statement D: identity element
theorem star_no_identity : ¬(∃ e : ℤ, ∀ x : ℤ, star x e = x ∧ star e x = x) :=
by sorry

-- Statement E: associativity
theorem star_not_assoc : ¬(∀ x y z : ℤ, star (star x y) z = star x (star y z)) :=
by sorry

end star_comm_star_distrib_over_add_star_special_case_star_no_identity_star_not_assoc_l219_219812


namespace convert_spherical_to_rectangular_l219_219596

noncomputable def spherical_to_rectangular (rho theta phi : ℝ) : ℝ × ℝ × ℝ :=
  (rho * Real.sin phi * Real.cos theta, rho * Real.sin phi * Real.sin theta, rho * Real.cos phi)

theorem convert_spherical_to_rectangular :
  spherical_to_rectangular 10 (4 * Real.pi / 3) (Real.pi / 3) = (-5 * Real.sqrt 3, -15 / 2, 5) :=
by 
  sorry

end convert_spherical_to_rectangular_l219_219596


namespace tan_five_pi_over_four_l219_219277

-- Define the question to prove
theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
by
  sorry

end tan_five_pi_over_four_l219_219277


namespace point_outside_circle_l219_219119

theorem point_outside_circle (a b : ℝ) (h : ∃ x y : ℝ, (a * x + b * y = 1) ∧ (x^2 + y^2 = 1)) :
  a^2 + b^2 > 1 := by
  sorry

end point_outside_circle_l219_219119


namespace distinct_real_roots_of_quadratic_l219_219542

variable (m : ℝ)

theorem distinct_real_roots_of_quadratic (h1 : 4 + 4 * m > 0) (h2 : m ≠ 0) : m = 1 :=
by
  sorry

end distinct_real_roots_of_quadratic_l219_219542


namespace least_number_subtracted_l219_219950

theorem least_number_subtracted {
  x : ℕ
} : 
  (∀ (m : ℕ), m ∈ [5, 9, 11] → (997 - x) % m = 3) → x = 4 :=
by
  sorry

end least_number_subtracted_l219_219950


namespace geom_seq_sum_l219_219029

theorem geom_seq_sum (a : ℕ → ℝ) (r : ℝ)
  (h1 : a 1 + a 2 = 16) 
  (h2 : a 3 + a 4 = 24) 
  (h_geom : ∀ n, a (n+1) = r * a n):
  a 7 + a 8 = 54 :=
sorry

end geom_seq_sum_l219_219029


namespace work_completion_l219_219382

theorem work_completion (W : ℝ) (a b : ℝ) (ha : a = W / 12) (hb : b = W / 6) :
  W / (a + b) = 4 :=
by {
  sorry
}

end work_completion_l219_219382


namespace conditional_prob_correct_l219_219368

/-- Define the events A and B as per the problem -/
def event_A (x y : ℕ) : Prop := (x + y) % 2 = 0

def event_B (x y : ℕ) : Prop := (x % 2 = 0 ∨ y % 2 = 0) ∧ x ≠ y

/-- Define the probability of event A -/
def prob_A : ℚ := 1 / 2

/-- Define the combined probability of both events A and B occurring -/
def prob_A_and_B : ℚ := 1 / 6

/-- Calculate the conditional probability P(B | A) -/
def conditional_prob : ℚ := prob_A_and_B / prob_A

theorem conditional_prob_correct : conditional_prob = 1 / 3 := by
  -- This is where you would provide the proof if required
  sorry

end conditional_prob_correct_l219_219368


namespace least_five_digit_congruent_6_mod_17_l219_219337

theorem least_five_digit_congruent_6_mod_17 : ∃ n: ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 17 = 6 ∧ ∀ m: ℕ, 10000 ≤ m ∧ m < 100000 ∧ m % 17 = 6 → n ≤ m :=
sorry

end least_five_digit_congruent_6_mod_17_l219_219337


namespace quadratic_sum_of_squares_l219_219878

theorem quadratic_sum_of_squares (α β : ℝ) (h1 : α * β = 3) (h2 : α + β = 7) : α^2 + β^2 = 43 := 
by
  sorry

end quadratic_sum_of_squares_l219_219878


namespace factor_polynomial_l219_219100

theorem factor_polynomial :
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) = 
  (x^2 + 6 * x + 7) * (x^2 + 6 * x + 8) :=
sorry

end factor_polynomial_l219_219100


namespace boat_speed_still_water_l219_219206

theorem boat_speed_still_water (V_b V_c : ℝ) (h1 : 45 / (V_b - V_c) = t) (h2 : V_b = 12)
(h3 : V_b + V_c = 15):
  V_b = 12 :=
by
  sorry

end boat_speed_still_water_l219_219206


namespace roots_sum_powers_l219_219656

theorem roots_sum_powers (t : ℕ → ℝ) (b d f : ℝ)
  (ht0 : t 0 = 3)
  (ht1 : t 1 = 6)
  (ht2 : t 2 = 11)
  (hrec : ∀ k ≥ 2, t (k + 1) = b * t k + d * t (k - 1) + f * t (k - 2))
  (hpoly : ∀ x : ℝ, x^3 - 6 * x^2 + 11 * x - 6 = 0) :
  b + d + f = 13 :=
sorry

end roots_sum_powers_l219_219656


namespace horner_method_operations_l219_219417

-- Define the polynomial
def poly (x : ℤ) : ℤ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x + 1

-- Define Horner's method evaluation for the specific polynomial at x = 2
def horners_method_evaluated (x : ℤ) : ℤ :=
  (((((5 * x + 4) * x + 3) * x + 2) * x + 1) * x + 1)

-- Count multiplication and addition operations
def count_mul_ops : ℕ := 5
def count_add_ops : ℕ := 5

-- Proof statement
theorem horner_method_operations :
  ∀ (x : ℤ), x = 2 → 
  (count_mul_ops = 5) ∧ (count_add_ops = 5) :=
by
  intros x h
  sorry

end horner_method_operations_l219_219417


namespace number_of_people_in_first_group_l219_219268

-- Define variables representing the work done by one person in one day (W) and the number of people in the first group (P)
variable (W : ℕ) (P : ℕ)

-- Conditions from the problem
-- Some people can do 3 times a particular work in 3 days
def condition1 : Prop := P * 3 * W = 3 * W

-- It takes 6 people 3 days to do 6 times of that particular work
def condition2 : Prop := 6 * 3 * W = 6 * W

-- The statement to prove
theorem number_of_people_in_first_group 
  (h1 : condition1 W P) 
  (h2 : condition2 W) : P = 3 :=
by
  sorry

end number_of_people_in_first_group_l219_219268


namespace min_diff_proof_l219_219360

noncomputable def triangleMinDiff : ℕ :=
  let PQ := 666
  let QR := 667
  let PR := 2010 - PQ - QR
  if (PQ < QR ∧ QR < PR ∧ PQ + QR > PR ∧ PQ + PR > QR ∧ PR + QR > PQ) then QR - PQ else 0

theorem min_diff_proof :
  ∃ PQ QR PR : ℕ, PQ + QR + PR = 2010 ∧ PQ < QR ∧ QR < PR ∧ (PQ + QR > PR) ∧ (PQ + PR > QR) ∧ (PR + QR > PQ) ∧ (QR - PQ = triangleMinDiff) := sorry

end min_diff_proof_l219_219360


namespace repeating_decimal_product_l219_219171

noncomputable def x : ℚ := 1 / 33
noncomputable def y : ℚ := 1 / 3

theorem repeating_decimal_product :
  (x * y) = 1 / 99 :=
by
  -- Definitions of x and y
  sorry

end repeating_decimal_product_l219_219171


namespace shooting_accuracy_l219_219004

theorem shooting_accuracy 
  (P_A : ℚ) 
  (P_AB : ℚ) 
  (h1 : P_A = 9 / 10) 
  (h2 : P_AB = 1 / 2) 
  : P_AB / P_A = 5 / 9 := 
by
  sorry

end shooting_accuracy_l219_219004


namespace calculate_expression_l219_219359

theorem calculate_expression :
  36 + (150 / 15) + (12 ^ 2 * 5) - 300 - (270 / 9) = 436 := by
  sorry

end calculate_expression_l219_219359


namespace target_has_more_tools_l219_219394

-- Define the number of tools in the Walmart multitool
def walmart_screwdriver : ℕ := 1
def walmart_knives : ℕ := 3
def walmart_other_tools : ℕ := 2
def walmart_total_tools : ℕ := walmart_screwdriver + walmart_knives + walmart_other_tools

-- Define the number of tools in the Target multitool
def target_screwdriver : ℕ := 1
def target_knives : ℕ := 2 * walmart_knives
def target_files_scissors : ℕ := 3 + 1
def target_total_tools : ℕ := target_screwdriver + target_knives + target_files_scissors

-- The theorem stating the difference in the number of tools
theorem target_has_more_tools : (target_total_tools - walmart_total_tools) = 6 := by
  sorry

end target_has_more_tools_l219_219394


namespace cages_used_l219_219810

theorem cages_used (total_puppies sold_puppies puppies_per_cage remaining_puppies needed_cages additional_cage total_cages: ℕ) 
  (h1 : total_puppies = 36) 
  (h2 : sold_puppies = 7) 
  (h3 : puppies_per_cage = 4) 
  (h4 : remaining_puppies = total_puppies - sold_puppies) 
  (h5 : needed_cages = remaining_puppies / puppies_per_cage) 
  (h6 : additional_cage = if (remaining_puppies % puppies_per_cage = 0) then 0 else 1) 
  (h7 : total_cages = needed_cages + additional_cage) : 
  total_cages = 8 := 
by 
  sorry

end cages_used_l219_219810


namespace q_is_false_l219_219639

variable {p q : Prop}

theorem q_is_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : ¬q := by
  sorry

end q_is_false_l219_219639


namespace range_of_m_l219_219039

theorem range_of_m (x y m : ℝ) (h1 : x - 2 * y = 1) (h2 : 2 * x + y = 4 * m) (h3 : x + 3 * y < 6) : m < 7 / 4 :=
sorry

end range_of_m_l219_219039


namespace dark_more_than_light_l219_219099

-- Define the board size
def board_size : ℕ := 9

-- Define the number of dark squares in odd rows
def dark_in_odd_row : ℕ := 5

-- Define the number of light squares in odd rows
def light_in_odd_row : ℕ := 4

-- Define the number of dark squares in even rows
def dark_in_even_row : ℕ := 4

-- Define the number of light squares in even rows
def light_in_even_row : ℕ := 5

-- Calculate the total number of dark squares
def total_dark_squares : ℕ := (dark_in_odd_row * ((board_size + 1) / 2)) + (dark_in_even_row * (board_size / 2))

-- Calculate the total number of light squares
def total_light_squares : ℕ := (light_in_odd_row * ((board_size + 1) / 2)) + (light_in_even_row * (board_size / 2))

-- Define the main theorem
theorem dark_more_than_light : total_dark_squares - total_light_squares = 1 := by
  sorry

end dark_more_than_light_l219_219099


namespace original_price_lamp_l219_219959

theorem original_price_lamp
  (P : ℝ)
  (discount_rate : ℝ)
  (discounted_price : ℝ)
  (discount_is_20_perc : discount_rate = 0.20)
  (new_price_is_96 : discounted_price = 96)
  (price_after_discount : discounted_price = P * (1 - discount_rate)) :
  P = 120 :=
by
  sorry

end original_price_lamp_l219_219959


namespace max_arithmetic_subsequences_l219_219013

-- Definition of an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ (d c : ℤ), ∀ n : ℕ, a n = d * n + c

-- Condition that the sum of the indices is even
def sum_indices_even (n m : ℕ) : Prop :=
  (n % 2 = 0 ∧ m % 2 = 0) ∨ (n % 2 = 1 ∧ m % 2 = 1)

-- Maximum count of 3-term arithmetic sequences in a sequence of 20 terms
theorem max_arithmetic_subsequences (a : ℕ → ℤ) (h_arith : arithmetic_sequence a) :
  ∃ n : ℕ, n = 180 :=
by
  sorry

end max_arithmetic_subsequences_l219_219013


namespace enjoyable_gameplay_time_l219_219868

def total_gameplay_time_base : ℝ := 150
def enjoyable_fraction_base : ℝ := 0.30
def total_gameplay_time_expansion : ℝ := 50
def load_screen_fraction_expansion : ℝ := 0.25
def inventory_management_fraction_expansion : ℝ := 0.25
def mod_skip_fraction : ℝ := 0.15

def enjoyable_time_base : ℝ := total_gameplay_time_base * enjoyable_fraction_base
def not_load_screen_time_expansion : ℝ := total_gameplay_time_expansion * (1 - load_screen_fraction_expansion)
def not_inventory_management_time_expansion : ℝ := not_load_screen_time_expansion * (1 - inventory_management_fraction_expansion)

def tedious_time_base : ℝ := total_gameplay_time_base * (1 - enjoyable_fraction_base)
def tedious_time_expansion : ℝ := total_gameplay_time_expansion - not_inventory_management_time_expansion
def total_tedious_time : ℝ := tedious_time_base + tedious_time_expansion

def time_skipped_by_mod : ℝ := total_tedious_time * mod_skip_fraction

def total_enjoyable_time : ℝ := enjoyable_time_base + not_inventory_management_time_expansion + time_skipped_by_mod

theorem enjoyable_gameplay_time :
  total_enjoyable_time = 92.16 :=     by     simp [total_enjoyable_time, enjoyable_time_base, not_inventory_management_time_expansion, time_skipped_by_mod]; sorry

end enjoyable_gameplay_time_l219_219868


namespace max_value_of_S_n_divided_l219_219034

noncomputable def arithmetic_sequence (a₁ d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

noncomputable def S_n (a₁ d n : ℕ) : ℕ :=
  n * (n + 4)

theorem max_value_of_S_n_divided (a₁ d : ℕ) (h₁ : ∀ n, a₁ + (2 * n - 1) * d = 2 * (a₁ + (n - 1) * d) - 3)
  (h₂ : (a₁ + 5 * d)^2 = a₁ * (a₁ + 20 * d)) :
  ∃ n, 2 * S_n a₁ d n / 2^n = 6 := 
sorry

end max_value_of_S_n_divided_l219_219034


namespace expression_divisible_by_7_l219_219005

theorem expression_divisible_by_7 (k : ℕ) : 
  (∀ n : ℕ, n > 0 → ∃ m : ℤ, 3^(6*n-1) - k * 2^(3*n-2) + 1 = 7 * m) ↔ ∃ m' : ℤ, k = 7 * m' + 3 := 
by
  sorry

end expression_divisible_by_7_l219_219005


namespace equation_correct_l219_219591

variable (x y : ℝ)

-- Define the conditions
def condition1 : Prop := (x + y) / 3 = 1.888888888888889
def condition2 : Prop := 2 * x + y = 7

-- Prove the required equation under given conditions
theorem equation_correct : condition1 x y → condition2 x y → (x + y) = 5.666666666666667 := by
  intros _ _
  sorry

end equation_correct_l219_219591


namespace sequence_geometric_progression_l219_219617

theorem sequence_geometric_progression (p : ℝ) (a : ℕ → ℝ) (h1 : a 1 = 2)
  (h2 : ∀ n : ℕ, 0 < n → a (n + 1) = p * a n + 2^n)
  (h3 : ∀ n : ℕ, 0 < n → a (n + 1)^2 = a n * a (n + 2)): 
  ∃ p : ℝ, ∀ n : ℕ, a n = 2^n :=
by
  sorry

end sequence_geometric_progression_l219_219617


namespace simplify_expression_l219_219640

variable (x : ℝ)

theorem simplify_expression :
  (2 * x + 25) + (150 * x + 35) + (50 * x + 10) = 202 * x + 70 :=
sorry

end simplify_expression_l219_219640


namespace nth_equation_l219_219008

theorem nth_equation (n : ℕ) : (2 * n + 1)^2 - 1 = 4 * n * (n + 1) := 
by
  sorry

end nth_equation_l219_219008


namespace complex_number_sum_zero_l219_219599

theorem complex_number_sum_zero (a b : ℝ) (i : ℂ) (h : a + b * i = 1 - i) : a + b = 0 := 
by sorry

end complex_number_sum_zero_l219_219599


namespace polygon_sides_given_ratio_l219_219721

theorem polygon_sides_given_ratio (n : ℕ) 
  (h : (n - 2) * 180 / 360 = 9 / 2) : n = 11 :=
sorry

end polygon_sides_given_ratio_l219_219721


namespace batsman_average_after_11th_inning_l219_219556

variable (x : ℝ) -- The average before the 11th inning
variable (new_average : ℝ) -- The average after the 11th inning
variable (total_runs : ℝ) -- Total runs scored after 11 innings

-- Given conditions
def condition1 := total_runs = 11 * (x + 5)
def condition2 := total_runs = 10 * x + 110

theorem batsman_average_after_11th_inning : 
  ∀ (x : ℝ), 
    (x = 55) → (x + 5 = 60) :=
by
  intros
  sorry

end batsman_average_after_11th_inning_l219_219556


namespace min_total_cost_l219_219434

-- Defining the variables involved
variables (x y z : ℝ)
variables (h : ℝ := 1) (V : ℝ := 4)
def base_cost (x y : ℝ) : ℝ := 200 * (x * y)
def side_cost (x y : ℝ) (h : ℝ) : ℝ := 100 * (2 * (x + y)) * h
def total_cost (x y h : ℝ) : ℝ := base_cost x y + side_cost x y h

-- The condition that volume is 4 m^3
theorem min_total_cost : 
  (∀ x y, x * y = V) → 
  ∃ x y, total_cost x y h = 1600 :=
by
  sorry

end min_total_cost_l219_219434


namespace least_number_to_add_l219_219999

theorem least_number_to_add (a : ℕ) (p q r : ℕ) (h : a = 1076) (hp : p = 41) (hq : q = 59) (hr : r = 67) :
  ∃ k : ℕ, k = 171011 ∧ (a + k) % (lcm p (lcm q r)) = 0 :=
sorry

end least_number_to_add_l219_219999


namespace benny_picked_proof_l219_219644

-- Define the number of apples Dan picked
def dan_picked: ℕ := 9

-- Define the total number of apples picked
def total_apples: ℕ := 11

-- Define the number of apples Benny picked
def benny_picked (dan_picked total_apples: ℕ): ℕ :=
  total_apples - dan_picked

-- The theorem we need to prove
theorem benny_picked_proof: benny_picked dan_picked total_apples = 2 :=
by
  -- We calculate the number of apples Benny picked
  sorry

end benny_picked_proof_l219_219644


namespace point_B_is_4_l219_219211

def point_A : ℤ := -3
def units_to_move : ℤ := 7
def point_B : ℤ := point_A + units_to_move

theorem point_B_is_4 : point_B = 4 :=
by
  sorry

end point_B_is_4_l219_219211


namespace yellow_balloons_ratio_l219_219465

theorem yellow_balloons_ratio 
  (total_balloons : ℕ) 
  (colors : ℕ) 
  (yellow_balloons_taken : ℕ) 
  (h_total_balloons : total_balloons = 672)
  (h_colors : colors = 4)
  (h_yellow_balloons_taken : yellow_balloons_taken = 84) :
  yellow_balloons_taken / (total_balloons / colors) = 1 / 2 :=
sorry

end yellow_balloons_ratio_l219_219465


namespace sheena_sewing_weeks_l219_219840

theorem sheena_sewing_weeks (sew_time : ℕ) (bridesmaids : ℕ) (sewing_per_week : ℕ) 
    (h_sew_time : sew_time = 12) (h_bridesmaids : bridesmaids = 5) (h_sewing_per_week : sewing_per_week = 4) : 
    (bridesmaids * sew_time) / sewing_per_week = 15 := 
  by sorry

end sheena_sewing_weeks_l219_219840


namespace find_x_l219_219390

def binary_operation (a b c d : Int) : Int × Int := (a - c, b + d)

theorem find_x (x y : Int)
  (H1 : binary_operation 6 5 2 3 = (4, 8))
  (H2 : binary_operation x y 5 4 = (4, 8)) :
  x = 9 :=
by
  -- Necessary conditions and hypotheses are provided
  sorry -- Proof not required

end find_x_l219_219390


namespace initial_lychees_count_l219_219355

theorem initial_lychees_count (L : ℕ) (h1 : L / 2 = 2 * 100 * 5 / 5 * 5) : L = 500 :=
by sorry

end initial_lychees_count_l219_219355


namespace selection_ways_l219_219366

namespace CulturalPerformance

-- Define basic conditions
def num_students : ℕ := 6
def can_sing : ℕ := 3
def can_dance : ℕ := 2
def both_sing_and_dance : ℕ := 1

-- Define the proof statement
theorem selection_ways :
  ∃ (ways : ℕ), ways = 15 := by
  sorry

end CulturalPerformance

end selection_ways_l219_219366


namespace lottery_win_amount_l219_219558

theorem lottery_win_amount (total_tax : ℝ) (federal_tax_rate : ℝ) (local_tax_rate : ℝ) (tax_paid : ℝ) :
  total_tax = tax_paid →
  federal_tax_rate = 0.25 →
  local_tax_rate = 0.15 →
  tax_paid = 18000 →
  ∃ x : ℝ, x = 49655 :=
by
  intros h1 h2 h3 h4
  use (tax_paid / (federal_tax_rate + local_tax_rate * (1 - federal_tax_rate))), by
    norm_num at h1 h2 h3 h4
    sorry

end lottery_win_amount_l219_219558


namespace fraction_ratio_equivalence_l219_219017

theorem fraction_ratio_equivalence :
  ∃ (d : ℚ), d = 240 / 1547 ∧ ((2 / 13) / d) = ((5 / 34) / (7 / 48)) := 
by
  sorry

end fraction_ratio_equivalence_l219_219017


namespace zero_points_C_exist_l219_219942

theorem zero_points_C_exist (A B C : ℝ × ℝ) (hAB_dist : dist A B = 12) (h_perimeter : dist A B + dist A C + dist B C = 52)
    (h_area : abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2 = 100) : 
    false :=
by
  sorry

end zero_points_C_exist_l219_219942


namespace num_ways_books_distribution_l219_219070

-- Given conditions
def num_copies_type1 : ℕ := 8
def num_copies_type2 : ℕ := 4
def min_books_in_library_type1 : ℕ := 1
def max_books_in_library_type1 : ℕ := 7
def min_books_in_library_type2 : ℕ := 1
def max_books_in_library_type2 : ℕ := 3

-- The proof problem statement
theorem num_ways_books_distribution : 
  (max_books_in_library_type1 - min_books_in_library_type1 + 1) * 
  (max_books_in_library_type2 - min_books_in_library_type2 + 1) = 21 := by
    sorry

end num_ways_books_distribution_l219_219070


namespace barbara_total_candies_l219_219953

-- Condition: Barbara originally has 9 candies.
def C1 := 9

-- Condition: Barbara buys 18 more candies.
def C2 := 18

-- Question (proof problem): Prove that the total number of candies Barbara has is 27.
theorem barbara_total_candies : C1 + C2 = 27 := by
  -- Proof steps are not required, hence using sorry.
  sorry

end barbara_total_candies_l219_219953


namespace find_valid_pairs_l219_219761

noncomputable def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

def distinct_two_digit_primes : List (ℕ × ℕ) :=
  [(13, 53), (19, 47), (23, 43), (29, 37)]

def average (p q : ℕ) : ℕ := (p + q) / 2

def number1 (p q : ℕ) : ℕ := 100 * p + q
def number2 (p q : ℕ) : ℕ := 100 * q + p

theorem find_valid_pairs (p q : ℕ)
  (hp : is_prime p) (hq : is_prime q)
  (hpq : p ≠ q)
  (havg : average p q ∣ number1 p q ∧ average p q ∣ number2 p q) :
  (p, q) ∈ distinct_two_digit_primes ∨ (q, p) ∈ distinct_two_digit_primes :=
sorry

end find_valid_pairs_l219_219761


namespace calc_result_l219_219865

theorem calc_result : 
  let a := 82 + 3/5
  let b := 1/15
  let c := 3
  let d := 42 + 7/10
  (a / b) * c - d = 3674.3 :=
by
  sorry

end calc_result_l219_219865


namespace angle_in_gradians_l219_219239

noncomputable def gradians_in_full_circle : ℝ := 600
noncomputable def degrees_in_full_circle : ℝ := 360
noncomputable def angle_in_degrees : ℝ := 45

theorem angle_in_gradians :
  angle_in_degrees / degrees_in_full_circle * gradians_in_full_circle = 75 := 
by
  sorry

end angle_in_gradians_l219_219239


namespace directrix_of_parabola_l219_219936

theorem directrix_of_parabola (x y : ℝ) : 
  (x^2 = - (1/8) * y) → (y = 1/32) :=
sorry

end directrix_of_parabola_l219_219936


namespace positive_X_solution_l219_219705

def boxtimes (X Y : ℤ) : ℤ := X^2 - 2 * X + Y^2

theorem positive_X_solution (X : ℤ) (h : boxtimes X 7 = 164) : X = 13 :=
by
  sorry

end positive_X_solution_l219_219705


namespace simplify_expression_l219_219623

def E (x : ℝ) : ℝ :=
  6 * x^2 + 4 * x + 9 - (7 - 5 * x - 9 * x^3 + 8 * x^2)

theorem simplify_expression (x : ℝ) : E x = 9 * x^3 - 2 * x^2 + 9 * x + 2 :=
by
  sorry

end simplify_expression_l219_219623


namespace exponential_fraction_l219_219042

theorem exponential_fraction :
  (2^2014 + 2^2012) / (2^2014 - 2^2012) = 5 / 3 := 
by
  sorry

end exponential_fraction_l219_219042


namespace valid_n_values_l219_219525

variables (n x y : ℕ)

theorem valid_n_values :
  (n * (x - 3) = y + 3) ∧ (x + n = 3 * (y - n)) →
  (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 7) :=
by
  sorry

end valid_n_values_l219_219525


namespace f_g_of_4_eq_18_sqrt_21_div_7_l219_219397

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt x + 12 / Real.sqrt x

def g (x : ℝ) : ℝ := 2 * x ^ 2 - 2 * x - 3

theorem f_g_of_4_eq_18_sqrt_21_div_7 : f (g 4) = (18 * Real.sqrt 21) / 7 := by
  sorry

end f_g_of_4_eq_18_sqrt_21_div_7_l219_219397


namespace equiangular_polygon_angle_solution_l219_219680

-- Given two equiangular polygons P_1 and P_2 with different numbers of sides
-- Each angle of P_1 is x degrees
-- Each angle of P_2 is k * x degrees where k is an integer greater than 1
-- Prove that the number of valid pairs (x, k) is exactly 1

theorem equiangular_polygon_angle_solution : ∃ x k : ℕ, ( ∀ n m : ℕ, x = 180 - 360 / n ∧ k * x = 180 - 360 / m → (k > 1) → x = 60 ∧ k = 2) := sorry

end equiangular_polygon_angle_solution_l219_219680


namespace olivia_used_pieces_l219_219377

-- Definition of initial pieces of paper and remaining pieces of paper
def initial_pieces : ℕ := 81
def remaining_pieces : ℕ := 25

-- Prove that Olivia used 56 pieces of paper
theorem olivia_used_pieces : (initial_pieces - remaining_pieces) = 56 :=
by
  -- Proof steps can be filled here
  sorry

end olivia_used_pieces_l219_219377


namespace total_weight_of_remaining_macaroons_l219_219227

def total_weight_remaining_macaroons (total_macaroons : ℕ) (weight_per_macaroon : ℕ) (bags : ℕ) (bags_eaten : ℕ) : ℕ :=
  let macaroons_per_bag := total_macaroons / bags
  let remaining_macaroons := total_macaroons - macaroons_per_bag * bags_eaten
  remaining_macaroons * weight_per_macaroon

theorem total_weight_of_remaining_macaroons
  (total_macaroons : ℕ)
  (weight_per_macaroon : ℕ)
  (bags : ℕ)
  (bags_eaten : ℕ)
  (h1 : total_macaroons = 12)
  (h2 : weight_per_macaroon = 5)
  (h3 : bags = 4)
  (h4 : bags_eaten = 1)
  : total_weight_remaining_macaroons total_macaroons weight_per_macaroon bags bags_eaten = 45 := by
  sorry

end total_weight_of_remaining_macaroons_l219_219227


namespace jane_baking_time_l219_219038

-- Definitions based on the conditions
variables (J : ℝ) (J_time : J > 0) -- J is the time it takes Jane to bake cakes individually
variables (Roy_time : 5 > 0) -- Roy can bake cakes in 5 hours
variables (together_time : 2 > 0) -- They work together for 2 hours
variables (remaining_time : 0.4 > 0) -- Jane completes the remaining task in 0.4 hours alone

-- Lean statement to prove Jane's individual baking time
theorem jane_baking_time : 
  (2 * (1 / J + 1 / 5) + 0.4 * (1 / J) = 1) → 
  J = 4 :=
by 
  sorry

end jane_baking_time_l219_219038


namespace christine_wander_time_l219_219715

noncomputable def distance : ℝ := 80
noncomputable def speed : ℝ := 20
noncomputable def time : ℝ := distance / speed

theorem christine_wander_time : time = 4 := 
by
  sorry

end christine_wander_time_l219_219715


namespace total_amount_correct_l219_219018

-- Define the prices of jeans and tees
def price_jean : ℕ := 11
def price_tee : ℕ := 8

-- Define the quantities sold
def quantity_jeans_sold : ℕ := 4
def quantity_tees_sold : ℕ := 7

-- Calculate the total amount earned
def total_amount : ℕ := (price_jean * quantity_jeans_sold) + (price_tee * quantity_tees_sold)

-- Now, we state and prove the theorem
theorem total_amount_correct : total_amount = 100 :=
by
  -- Here we assert the correctness of the calculation
  sorry

end total_amount_correct_l219_219018


namespace number_of_marbles_in_Ellen_box_l219_219592

-- Defining the conditions given in the problem
def Dan_box_volume : ℕ := 216
def Ellen_side_multiplier : ℕ := 3
def marble_size_consistent_between_boxes : Prop := True -- Placeholder for the consistency condition

-- Main theorem statement
theorem number_of_marbles_in_Ellen_box :
  ∃ number_of_marbles_in_Ellen_box : ℕ,
  (∀ s : ℕ, s^3 = Dan_box_volume → (Ellen_side_multiplier * s)^3 / s^3 = 27 → 
  number_of_marbles_in_Ellen_box = 27 * Dan_box_volume) :=
by
  sorry

end number_of_marbles_in_Ellen_box_l219_219592


namespace find_x_of_perpendicular_l219_219799

-- Definitions based on the conditions in a)
def a (x : ℝ) : ℝ × ℝ := (x, x + 1)
def b : ℝ × ℝ := (1, 2)

-- The mathematical proof problem in Lean 4 statement: prove that the dot product is zero implies x = -2/3
theorem find_x_of_perpendicular (x : ℝ) (h : (a x).fst * b.fst + (a x).snd * b.snd = 0) : x = -2 / 3 := 
by
  sorry

end find_x_of_perpendicular_l219_219799


namespace max_n_value_l219_219143

theorem max_n_value (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  (1 / (a - b)) + (1 / (b - c)) + (1 / (c - d)) ≥ (9 / (a - d)) :=
by
  sorry

end max_n_value_l219_219143


namespace like_terms_exponents_l219_219958

theorem like_terms_exponents (m n : ℤ) 
  (h1 : m - 1 = 1) 
  (h2 : m + n = 3) : 
  m = 2 ∧ n = 1 :=
by 
  sorry

end like_terms_exponents_l219_219958


namespace white_area_is_69_l219_219888

def area_of_sign : ℕ := 6 * 20

def area_of_M : ℕ := 2 * (6 * 1) + 2 * 2

def area_of_A : ℕ := 2 * 4 + 1 * 2

def area_of_T : ℕ := 1 * 4 + 6 * 1

def area_of_H : ℕ := 2 * (6 * 1) + 1 * 3

def total_black_area : ℕ := area_of_M + area_of_A + area_of_T + area_of_H

def white_area (sign_area black_area : ℕ) : ℕ := sign_area - black_area

theorem white_area_is_69 : white_area area_of_sign total_black_area = 69 := by
  sorry

end white_area_is_69_l219_219888


namespace min_product_of_positive_numbers_l219_219660

theorem min_product_of_positive_numbers {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (h : a * b = a + b) : a * b = 4 :=
sorry

end min_product_of_positive_numbers_l219_219660


namespace line_tangent_to_parabola_l219_219113

theorem line_tangent_to_parabola (c : ℝ) : (∀ (x y : ℝ), 2 * x - y + c = 0 ∧ x^2 = 4 * y) → c = -4 := by
  sorry

end line_tangent_to_parabola_l219_219113


namespace cos_180_eq_neg1_sin_180_eq_0_l219_219836

theorem cos_180_eq_neg1 : Real.cos (180 * Real.pi / 180) = -1 := sorry
theorem sin_180_eq_0 : Real.sin (180 * Real.pi / 180) = 0 := sorry

end cos_180_eq_neg1_sin_180_eq_0_l219_219836


namespace total_votes_l219_219406

variable (T S R F V : ℝ)

-- Conditions
axiom h1 : T = S + 0.15 * V
axiom h2 : S = R + 0.05 * V
axiom h3 : R = F + 0.07 * V
axiom h4 : T + S + R + F = V
axiom h5 : T - 2500 - 2000 = S + 2500
axiom h6 : S + 2500 = R + 2000 + 0.05 * V

theorem total_votes : V = 30000 :=
sorry

end total_votes_l219_219406


namespace average_increase_by_3_l219_219296

def initial_average_before_inning_17 (A : ℝ) : Prop :=
  16 * A + 85 = 17 * 37

theorem average_increase_by_3 (A : ℝ) (h : initial_average_before_inning_17 A) :
  37 - A = 3 :=
by
  sorry

end average_increase_by_3_l219_219296


namespace fuel_for_empty_plane_per_mile_l219_219180

theorem fuel_for_empty_plane_per_mile :
  let F := 106000 / 400 - (35 * 3 + 70 * 2)
  F = 20 := 
by
  sorry

end fuel_for_empty_plane_per_mile_l219_219180


namespace triangle_area_is_9_point_5_l219_219027

def Point : Type := (ℝ × ℝ)

def A : Point := (0, 1)
def B : Point := (4, 0)
def C : Point := (3, 5)

noncomputable def areaOfTriangle (A B C : Point) : ℝ :=
  0.5 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

theorem triangle_area_is_9_point_5 :
  areaOfTriangle A B C = 9.5 :=
by
  sorry

end triangle_area_is_9_point_5_l219_219027


namespace negation_statement_l219_219661

variable (x y : ℝ)

theorem negation_statement :
  ¬ (x > 1 ∧ y > 2) ↔ (x ≤ 1 ∨ y ≤ 2) :=
by
  sorry

end negation_statement_l219_219661


namespace net_pay_rate_per_hour_l219_219085

-- Defining the given conditions
def travel_hours : ℕ := 3
def speed_mph : ℕ := 50
def fuel_efficiency : ℕ := 25 -- miles per gallon
def pay_rate_per_mile : ℚ := 0.60 -- dollars per mile
def gas_cost_per_gallon : ℚ := 2.50 -- dollars per gallon

-- Define the statement we want to prove
theorem net_pay_rate_per_hour : 
  (travel_hours * speed_mph * pay_rate_per_mile - 
  (travel_hours * speed_mph / fuel_efficiency) * gas_cost_per_gallon) / 
  travel_hours = 25 :=
by
  repeat {sorry}

end net_pay_rate_per_hour_l219_219085


namespace gain_percent_correct_l219_219127

variable (CP SP Gain : ℝ)
variable (H₁ : CP = 900)
variable (H₂ : SP = 1125)
variable (H₃ : Gain = SP - CP)

theorem gain_percent_correct : (Gain / CP) * 100 = 25 :=
by
  sorry

end gain_percent_correct_l219_219127


namespace equivalence_of_statements_l219_219022

variable (S M : Prop)

theorem equivalence_of_statements : 
  (S → M) ↔ ((¬M → ¬S) ∧ (¬S ∨ M)) :=
by
  sorry

end equivalence_of_statements_l219_219022


namespace can_form_triangle_l219_219037

theorem can_form_triangle (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_condition : c^2 ≤ 4 * a * b) : 
  a + b > c ∧ a + c > b ∧ b + c > a := 
sorry

end can_form_triangle_l219_219037


namespace annie_weeks_off_sick_l219_219822

-- Define the conditions and the question
def weekly_hours_chess : ℕ := 2
def weekly_hours_drama : ℕ := 8
def weekly_hours_glee : ℕ := 3
def semester_weeks : ℕ := 12
def total_hours_before_midterms : ℕ := 52

-- Define the proof problem
theorem annie_weeks_off_sick :
  let total_weekly_hours := weekly_hours_chess + weekly_hours_drama + weekly_hours_glee
  let attended_weeks := total_hours_before_midterms / total_weekly_hours
  semester_weeks - attended_weeks = 8 :=
by
  -- Automatically prove by computation of above assumptions.
  sorry

end annie_weeks_off_sick_l219_219822


namespace cube_root_of_x_sqrt_x_eq_x_half_l219_219249

variable (x : ℝ) (h : 0 < x)

theorem cube_root_of_x_sqrt_x_eq_x_half : (x * Real.sqrt x) ^ (1/3) = x ^ (1/2) := by
  sorry

end cube_root_of_x_sqrt_x_eq_x_half_l219_219249


namespace minimum_value_quadratic_l219_219803

noncomputable def quadratic (x : ℝ) : ℝ := x^2 - 4 * x + 5

theorem minimum_value_quadratic :
  ∀ x : ℝ, quadratic x ≥ 1 :=
by
  sorry

end minimum_value_quadratic_l219_219803


namespace probability_two_face_cards_l219_219428

def cardDeck : ℕ := 52
def totalFaceCards : ℕ := 12

-- Probability of selecting one face card as the first card
def probabilityFirstFaceCard : ℚ := totalFaceCards / cardDeck

-- Probability of selecting another face card as the second card
def probabilitySecondFaceCard (cardsLeft : ℕ) : ℚ := (totalFaceCards - 1) / cardsLeft

-- Combined probability of selecting two face cards
theorem probability_two_face_cards :
  let combined_probability := probabilityFirstFaceCard * probabilitySecondFaceCard (cardDeck - 1)
  combined_probability = 22 / 442 := 
  by
    sorry

end probability_two_face_cards_l219_219428


namespace solve_for_y_l219_219801

theorem solve_for_y : ∀ (y : ℚ), 2 * y + 3 * y = 500 - (4 * y + 6 * y) → y = 100 / 3 := by
  intros y h
  sorry

end solve_for_y_l219_219801


namespace find_first_two_solutions_l219_219139

theorem find_first_two_solutions :
  ∃ (n1 n2 : ℕ), 
    (n1 ≡ 3 [MOD 7]) ∧ (n1 ≡ 4 [MOD 9]) ∧ 
    (n2 ≡ 3 [MOD 7]) ∧ (n2 ≡ 4 [MOD 9]) ∧ 
    n1 < n2 ∧ 
    n1 = 31 ∧ n2 = 94 := 
by 
  sorry

end find_first_two_solutions_l219_219139


namespace book_distribution_ways_l219_219291

theorem book_distribution_ways : 
  ∃ n : ℕ, n = 7 ∧ ∀ k : ℕ, 1 ≤ k ∧ k ≤ 7 →
  ∃ l : ℕ, l + (8 - l) = 8 ∧ 1 ≤ l ∧ 1 ≤ 8 - l :=
by
  -- We will provide a proof here.
  sorry

end book_distribution_ways_l219_219291


namespace ratio_n_over_p_l219_219668

-- Definitions and conditions from the problem
variables {m n p : ℝ}

-- The quadratic equation x^2 + mx + n = 0 has roots that are thrice those of x^2 + px + m = 0.
-- None of m, n, and p is zero.

-- Prove that n / p = 27 given these conditions.
theorem ratio_n_over_p (hmn0 : m ≠ 0) (hn : n = 9 * m) (hp : p = m / 3):
  n / p = 27 :=
  by
    sorry -- Formal proof will go here.

end ratio_n_over_p_l219_219668


namespace puja_runs_distance_in_meters_l219_219696

noncomputable def puja_distance (time_in_seconds : ℝ) (speed_kmph : ℝ) : ℝ :=
  let time_in_hours := time_in_seconds / 3600
  let distance_km := speed_kmph * time_in_hours
  distance_km * 1000

theorem puja_runs_distance_in_meters :
  abs (puja_distance 59.995200383969284 30 - 499.96) < 0.01 :=
by
  sorry

end puja_runs_distance_in_meters_l219_219696


namespace rachel_assembly_time_l219_219756

theorem rachel_assembly_time :
  let chairs := 20
  let tables := 8
  let bookshelves := 5
  let time_per_chair := 6
  let time_per_table := 8
  let time_per_bookshelf := 12
  let total_chairs_time := chairs * time_per_chair
  let total_tables_time := tables * time_per_table
  let total_bookshelves_time := bookshelves * time_per_bookshelf
  total_chairs_time + total_tables_time + total_bookshelves_time = 244 := by
  sorry

end rachel_assembly_time_l219_219756


namespace trapezoid_area_correct_l219_219425

-- Given sides of the trapezoid
def sides : List ℚ := [4, 6, 8, 10]

-- Definition of the function to calculate the sum of all possible areas.
noncomputable def sumOfAllPossibleAreas (sides : List ℚ) : ℚ :=
  -- Assuming configurations and calculations are correct by problem statement
  let r4 := 21
  let r5 := 7
  let r6 := 0
  let n4 := 3
  let n5 := 15
  r4 + r5 + r6 + n4 + n5

-- Check that the given sides lead to sum of areas equal to 46
theorem trapezoid_area_correct : sumOfAllPossibleAreas sides = 46 := by
  sorry

end trapezoid_area_correct_l219_219425


namespace general_admission_tickets_l219_219965

variable (x y : ℕ)

theorem general_admission_tickets (h1 : x + y = 525) (h2 : 4 * x + 6 * y = 2876) : y = 388 := by
  sorry

end general_admission_tickets_l219_219965


namespace dan_money_left_l219_219175

theorem dan_money_left
  (initial_amount : ℝ := 45)
  (cost_per_candy_bar : ℝ := 4)
  (num_candy_bars : ℕ := 4)
  (price_toy_car : ℝ := 15)
  (discount_rate_toy_car : ℝ := 0.10)
  (sales_tax_rate : ℝ := 0.05) :
  initial_amount - ((num_candy_bars * cost_per_candy_bar) + ((price_toy_car - (price_toy_car * discount_rate_toy_car)) * (1 + sales_tax_rate))) = 14.02 :=
by
  sorry

end dan_money_left_l219_219175


namespace find_other_number_l219_219590

theorem find_other_number (HCF LCM a b : ℕ) (h1 : HCF = 108) (h2 : LCM = 27720) (h3 : a = 216) (h4 : HCF * LCM = a * b) : b = 64 :=
  sorry

end find_other_number_l219_219590


namespace train_length_490_l219_219121

noncomputable def train_length (speed_kmh : ℕ) (time_sec : ℕ) : ℕ :=
  let speed_ms := speed_kmh * 1000 / 3600
  speed_ms * time_sec

theorem train_length_490 :
  train_length 63 28 = 490 := by
  -- Proof goes here
  sorry

end train_length_490_l219_219121


namespace arithmetic_sequence_sum_l219_219504

theorem arithmetic_sequence_sum (a d : ℚ) (a1 : a = 1 / 2) 
(S : ℕ → ℚ) (Sn : ∀ n, S n = n * a + (n * (n - 1) / 2) * d) 
(S2_eq_a3 : S 2 = a + 2 * d) :
  ∀ n, S n = (1 / 4 : ℚ) * n^2 + (1 / 4 : ℚ) * n :=
by
  intros n
  sorry

end arithmetic_sequence_sum_l219_219504


namespace car_time_interval_l219_219621

-- Define the conditions
def road_length := 3 -- in miles
def total_time := 10 -- in hours
def number_of_cars := 30

-- Define the conversion factor and the problem to prove
def hours_to_minutes (hours: ℕ) : ℕ := hours * 60
def time_interval_per_car (total_time_minutes: ℕ) (number_of_cars: ℕ) : ℕ := total_time_minutes / number_of_cars

-- The Lean 4 statement for the proof problem
theorem car_time_interval :
  time_interval_per_car (hours_to_minutes total_time) number_of_cars = 20 :=
by
  sorry

end car_time_interval_l219_219621


namespace seventh_root_of_unity_sum_l219_219489

theorem seventh_root_of_unity_sum (z : ℂ) (h1 : z^7 = 1) (h2 : z ≠ 1) :
  z + z^2 + z^4 = (-1 + Complex.I * Real.sqrt 11) / 2 ∨ z + z^2 + z^4 = (-1 - Complex.I * Real.sqrt 11) / 2 := 
by sorry

end seventh_root_of_unity_sum_l219_219489


namespace like_terms_proof_l219_219219

theorem like_terms_proof (m n : ℤ) 
  (h1 : m + 10 = 3 * n - m) 
  (h2 : 7 - n = n - m) :
  m^2 - 2 * m * n + n^2 = 9 := by
  sorry

end like_terms_proof_l219_219219


namespace train_length_l219_219553

theorem train_length (L V : ℝ) 
  (h1 : L = V * 18) 
  (h2 : L + 175 = V * 39) : 
  L = 150 := 
by 
  -- proof omitted 
  sorry

end train_length_l219_219553


namespace none_of_these_l219_219574

theorem none_of_these (a T : ℝ) : 
  ¬(∀ (x y : ℝ), 4 * T * x + 2 * a^2 * y + 4 * a * T = 0) ∧ 
  ¬(∀ (x y : ℝ), 4 * T * x - 2 * a^2 * y + 4 * a * T = 0) ∧ 
  ¬(∀ (x y : ℝ), 4 * T * x + 2 * a^2 * y - 4 * a * T = 0) ∧ 
  ¬(∀ (x y : ℝ), 4 * T * x - 2 * a^2 * y - 4 * a * T = 0) :=
sorry

end none_of_these_l219_219574


namespace gcd_lcm_product_24_36_l219_219843

-- Definitions for gcd, lcm, and product for given numbers, skipping proof with sorry
theorem gcd_lcm_product_24_36 : Nat.gcd 24 36 * Nat.lcm 24 36 = 864 := by
  -- Sorry used to skip proof
  sorry

end gcd_lcm_product_24_36_l219_219843


namespace max_value_of_square_diff_max_value_of_square_diff_achieved_l219_219153

theorem max_value_of_square_diff (a b : ℝ) (h : a^2 + b^2 = 4) : (a - b)^2 ≤ 8 :=
sorry

theorem max_value_of_square_diff_achieved (a b : ℝ) (h : a^2 + b^2 = 4) : ∃ a b : ℝ, (a - b)^2 = 8 :=
sorry

end max_value_of_square_diff_max_value_of_square_diff_achieved_l219_219153


namespace trent_walks_to_bus_stop_l219_219850

theorem trent_walks_to_bus_stop (x : ℕ) (h1 : 2 * (x + 7) = 22) : x = 4 :=
sorry

end trent_walks_to_bus_stop_l219_219850


namespace product_of_four_consecutive_integers_divisible_by_12_l219_219755

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l219_219755


namespace abs_inequality_solution_l219_219160

theorem abs_inequality_solution (x : ℝ) :
  3 ≤ |x + 2| ∧ |x + 2| ≤ 7 ↔ (1 ≤ x ∧ x ≤ 5) ∨ (-9 ≤ x ∧ x ≤ -5) :=
by
  sorry

end abs_inequality_solution_l219_219160


namespace simplify_and_evaluate_expression_l219_219852

variable (x y : ℝ)
variable (h1 : x = 1)
variable (h2 : y = Real.sqrt 2)

theorem simplify_and_evaluate_expression : 
  (x + 2 * y) ^ 2 - x * (x + 4 * y) + (1 - y) * (1 + y) = 7 := by
  sorry

end simplify_and_evaluate_expression_l219_219852


namespace possible_values_of_a_and_b_l219_219405

theorem possible_values_of_a_and_b (a b : ℕ) : 
  (a = 22 ∨ a = 33 ∨ a = 40 ∨ a = 42) ∧ 
  (b = 21 ∨ b = 10 ∨ b = 3 ∨ b = 1) ∧ 
  (a % (b + 1) = 0) ∧ (43 % (a + b) = 0) :=
sorry

end possible_values_of_a_and_b_l219_219405


namespace gold_coins_distribution_l219_219252

theorem gold_coins_distribution (x y : ℝ) (h₁ : x + y = 25) (h₂ : x ≠ y)
  (h₃ : (x^2 - y^2) = k * (x - y)) : k = 25 :=
sorry

end gold_coins_distribution_l219_219252


namespace complex_expression_equals_zero_l219_219649

def i : ℂ := Complex.I

theorem complex_expression_equals_zero : 2 * i^5 + (1 - i)^2 = 0 := 
by
  sorry

end complex_expression_equals_zero_l219_219649


namespace apples_to_cucumbers_l219_219260

theorem apples_to_cucumbers (a b c : ℕ) 
    (h₁ : 10 * a = 5 * b) 
    (h₂ : 3 * b = 4 * c) : 
    (24 * a) = 16 * c := 
by
  sorry

end apples_to_cucumbers_l219_219260


namespace brownie_cost_l219_219403

theorem brownie_cost (total_money : ℕ) (num_pans : ℕ) (pieces_per_pan : ℕ) (cost_per_piece : ℕ) :
  total_money = 32 → num_pans = 2 → pieces_per_pan = 8 → cost_per_piece = total_money / (num_pans * pieces_per_pan) → 
  cost_per_piece = 2 :=
by
  intros h1 h2 h3 h4
  sorry

end brownie_cost_l219_219403


namespace triangle_inequality_third_side_l219_219722

theorem triangle_inequality_third_side (a b x : ℝ) (h₁ : a = 3) (h₂ : b = 5) (h₃ : 0 < x) (h₄ : x < a + b) (h₅ : a < b + x) (h₆ : b < a + x) :
  ¬(x = 9) := by
  sorry

end triangle_inequality_third_side_l219_219722


namespace part2_proof_l219_219401

noncomputable def f (x : ℝ) : ℝ := Real.exp (Real.log x) - Real.exp 1 * x

theorem part2_proof (x : ℝ) (h : 0 < x) :
  x * f x - Real.exp x + 2 * Real.exp 1 * x ≤ 0 := 
sorry

end part2_proof_l219_219401


namespace cos_double_angle_l219_219819

theorem cos_double_angle 
  (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1/3) 
  (h2 : Real.cos α * Real.sin β = 1/6) :
  Real.cos (2 * α + 2 * β) = 1/9 :=
sorry

end cos_double_angle_l219_219819


namespace chord_intersection_probability_l219_219659

noncomputable def probability_chord_intersection : ℚ :=
1 / 3

theorem chord_intersection_probability 
    (A B C D : ℕ) 
    (total_points : ℕ) 
    (adjacent : A + 1 = B ∨ A = B + 1)
    (distinct : ∀ (A B C D : ℕ), A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
    (points_on_circle : total_points = 2023) :
    ∃ p : ℚ, p = probability_chord_intersection :=
by sorry

end chord_intersection_probability_l219_219659


namespace range_f_l219_219916

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3) * Real.sin x - Real.cos x

theorem range_f : Set.range f = Set.Icc (-2 : ℝ) 2 := 
by
  sorry

end range_f_l219_219916


namespace otimes_property_l219_219951

def otimes (a b : ℚ) : ℚ := (a^3) / b

theorem otimes_property : otimes (otimes 2 3) 4 - otimes 2 (otimes 3 4) = 80 / 27 := by
  sorry

end otimes_property_l219_219951


namespace solve_for_x_l219_219389

theorem solve_for_x (x : ℝ) (h : (2 / (x + 3) + 3 * x / (x + 3) - 4 / (x + 3) = 4)) : x = -14 := 
by 
  sorry

end solve_for_x_l219_219389


namespace inequality_solution_l219_219508

theorem inequality_solution (x : ℝ) (h1 : 2 * x + 1 > x + 3) (h2 : 2 * x - 4 < x) : 2 < x ∧ x < 4 := sorry

end inequality_solution_l219_219508


namespace rectangle_ratio_l219_219635

theorem rectangle_ratio (a b : ℝ) (side : ℝ) (M N : ℝ → ℝ) (P Q : ℝ → ℝ)
  (h_side : side = 4)
  (h_M : M 0 = 4 / 3 ∧ M 4 = 8 / 3)
  (h_N : N 0 = 4 / 3 ∧ N 4 = 8 / 3)
  (h_perpendicular : P 0 = Q 0 ∧ P 4 = Q 4)
  (h_area : side * side = 16) :
  let UV := 6 / 5
  let VW := 40 / 3
  UV / VW = 9 / 100 :=
sorry

end rectangle_ratio_l219_219635


namespace median_length_range_l219_219664

/-- Define the structure of the triangle -/
structure Triangle :=
  (A B C : ℝ) -- vertices of the triangle
  (AD AE AF : ℝ) -- lengths of altitude, angle bisector, and median
  (angleA : AngleType) -- type of angle A (acute, orthogonal, obtuse)

-- Define the angle type as a custom type
inductive AngleType
| acute
| orthogonal
| obtuse

def m_range (t : Triangle) : Set ℝ :=
  match t.angleA with
  | AngleType.acute => {m : ℝ | 13 < m ∧ m < (2028 / 119)}
  | AngleType.orthogonal => {m : ℝ | m = (2028 / 119)}
  | AngleType.obtuse => {m : ℝ | (2028 / 119) < m}

-- Lean statement for proving the problem
theorem median_length_range (t : Triangle)
  (hAD : t.AD = 12)
  (hAE : t.AE = 13) : t.AF ∈ m_range t :=
by
  sorry

end median_length_range_l219_219664


namespace D_72_eq_81_l219_219128

-- Definition of the function for the number of decompositions
def D (n : Nat) : Nat :=
  -- D(n) would ideally be implemented here as per the given conditions
  sorry

-- Prime factorization of 72
def prime_factorization_72 : List Nat :=
  [2, 2, 2, 3, 3]

-- Statement to prove
theorem D_72_eq_81 : D 72 = 81 :=
by
  -- Placeholder for actual proof
  sorry

end D_72_eq_81_l219_219128


namespace present_ages_ratio_l219_219244

noncomputable def ratio_of_ages (F S : ℕ) : ℚ :=
  F / S

theorem present_ages_ratio (F S : ℕ) (h1 : F + S = 220) (h2 : (F + 10) * 3 = (S + 10) * 5) :
  ratio_of_ages F S = 7 / 4 :=
by
  sorry

end present_ages_ratio_l219_219244


namespace megan_pages_left_l219_219482

theorem megan_pages_left (total_problems completed_problems problems_per_page : ℕ)
    (h_total : total_problems = 40)
    (h_completed : completed_problems = 26)
    (h_problems_per_page : problems_per_page = 7) :
    (total_problems - completed_problems) / problems_per_page = 2 :=
by
  sorry

end megan_pages_left_l219_219482


namespace smallest_two_digit_palindrome_l219_219097

def is_palindrome {α : Type} [DecidableEq α] (xs : List α) : Prop :=
  xs = xs.reverse

-- A number is a two-digit palindrome in base 5 if it has the form ab5 where a and b are digits 0-4
def two_digit_palindrome_base5 (n : ℕ) : Prop :=
  ∃ a b : ℕ, a < 5 ∧ b < 5 ∧ a ≠ 0 ∧ n = a * 5 + b ∧ is_palindrome [a, b]

-- A number is a three-digit palindrome in base 2 if it has the form abc2 where a = c and b can vary (0-1)
def three_digit_palindrome_base2 (n : ℕ) : Prop :=
  ∃ a b c : ℕ, a < 2 ∧ b < 2 ∧ c < 2 ∧ a = c ∧ n = a * 4 + b * 2 + c ∧ is_palindrome [a, b, c]

theorem smallest_two_digit_palindrome :
  ∃ n, two_digit_palindrome_base5 n ∧ three_digit_palindrome_base2 n ∧
       (∀ m, two_digit_palindrome_base5 m ∧ three_digit_palindrome_base2 m → n ≤ m) :=
sorry

end smallest_two_digit_palindrome_l219_219097


namespace tomato_plants_count_l219_219138

theorem tomato_plants_count :
  ∀ (sunflowers corn tomatoes total_rows plants_per_row : ℕ),
  sunflowers = 45 →
  corn = 81 →
  plants_per_row = 9 →
  total_rows = (sunflowers / plants_per_row) + (corn / plants_per_row) →
  tomatoes = total_rows * plants_per_row →
  tomatoes = 126 :=
by
  intros sunflowers corn tomatoes total_rows plants_per_row Hs Hc Hp Ht Hm
  rw [Hs, Hc, Hp] at *
  -- Additional calculation steps could go here to prove the theorem if needed
  sorry

end tomato_plants_count_l219_219138


namespace max_elevation_reached_l219_219675

theorem max_elevation_reached 
  (t : ℝ) 
  (s : ℝ) 
  (h : s = 200 * t - 20 * t^2) : 
  ∃ t_max : ℝ, ∃ s_max : ℝ, t_max = 5 ∧ s_max = 500 ∧ s_max = 200 * t_max - 20 * t_max^2 := sorry

end max_elevation_reached_l219_219675


namespace allocation_schemes_correct_l219_219054

noncomputable def allocation_schemes : Nat :=
  let C (n k : Nat) : Nat := Nat.choose n k
  -- Calculate category 1: one school gets 1 professor, two get 2 professors each
  let category1 := C 3 1 * C 5 1 * C 4 2 * C 2 2 / 2
  -- Calculate category 2: one school gets 3 professors, two get 1 professor each
  let category2 := C 3 1 * C 5 3 * C 2 1 * C 1 1 / 2
  -- Total allocation ways
  let totalWays := 6 * (category1 + category2)
  totalWays

theorem allocation_schemes_correct : allocation_schemes = 900 := by
  sorry

end allocation_schemes_correct_l219_219054


namespace inequality_subtract_l219_219851

-- Definitions of the main variables and conditions
variables {a b : ℝ}
-- Condition that should hold
axiom h : a > b

-- Expected conclusion
theorem inequality_subtract : a - 1 > b - 2 :=
by
  sorry

end inequality_subtract_l219_219851


namespace total_brownies_l219_219456

theorem total_brownies (brought_to_school left_at_home : ℕ) (h1 : brought_to_school = 16) (h2 : left_at_home = 24) : 
  brought_to_school + left_at_home = 40 := 
by 
  sorry

end total_brownies_l219_219456


namespace ratio_girls_to_boys_l219_219040

theorem ratio_girls_to_boys (g b : ℕ) (h1 : g = b + 4) (h2 : g + b = 28) :
  g / gcd g b = 4 ∧ b / gcd g b = 3 :=
by
  sorry

end ratio_girls_to_boys_l219_219040


namespace quadratic_product_fact_l219_219742

def quadratic_factors_product : Prop :=
  let integer_pairs := [(-1, 24), (-2, 12), (-3, 8), (-4, 6), (-6, 4), (-8, 3), (-12, 2), (-24, 1)]
  let t_values := integer_pairs.map (fun (c, d) => c + d)
  let product_t := t_values.foldl (fun acc t => acc * t) 1
  product_t = -5290000

theorem quadratic_product_fact : quadratic_factors_product :=
by sorry

end quadratic_product_fact_l219_219742


namespace mark_total_flowers_l219_219267

theorem mark_total_flowers (yellow purple green total : ℕ) 
  (hyellow : yellow = 10)
  (hpurple : purple = yellow + (yellow * 80 / 100))
  (hgreen : green = (yellow + purple) * 25 / 100)
  (htotal : total = yellow + purple + green) : 
  total = 35 :=
by
  sorry

end mark_total_flowers_l219_219267


namespace sin_x_sin_y_eq_sin_beta_sin_gamma_l219_219707

theorem sin_x_sin_y_eq_sin_beta_sin_gamma
  (A B C M : Type)
  (AM BM CM : ℝ)
  (alpha beta gamma x y : ℝ)
  (h1 : AM * AM = BM * CM)
  (h2 : BM ≠ 0)
  (h3 : CM ≠ 0)
  (hx : AM / BM = Real.sin beta / Real.sin x)
  (hy : AM / CM = Real.sin gamma / Real.sin y) :
  Real.sin x * Real.sin y = Real.sin beta * Real.sin gamma := 
sorry

end sin_x_sin_y_eq_sin_beta_sin_gamma_l219_219707


namespace total_outfits_l219_219682

def numRedShirts : ℕ := 7
def numGreenShirts : ℕ := 5
def numPants : ℕ := 6
def numRedHats : ℕ := 7
def numGreenHats : ℕ := 9

theorem total_outfits : 
  ((numRedShirts * numPants * numGreenHats) + 
   (numGreenShirts * numPants * numRedHats) + 
   ((numRedShirts * numRedHats + numGreenShirts * numGreenHats) * numPants)
  ) = 1152 := 
by
  sorry

end total_outfits_l219_219682


namespace cube_diagonal_length_l219_219557

theorem cube_diagonal_length (V A : ℝ) (hV : V = 384) (hA : A = 384) : 
  ∃ d : ℝ, d = 8 * Real.sqrt 3 :=
by
  sorry

end cube_diagonal_length_l219_219557


namespace find_mode_l219_219453

def scores : List ℕ :=
  [105, 107, 111, 111, 112, 112, 115, 118, 123, 124, 124, 126, 127, 129, 129, 129, 130, 130, 130, 130, 131, 140, 140, 140, 140]

def mode (ls : List ℕ) : ℕ :=
  ls.foldl (λmodeScore score => if ls.count score > ls.count modeScore then score else modeScore) 0

theorem find_mode :
  mode scores = 130 :=
by
  sorry

end find_mode_l219_219453


namespace angle_same_terminal_side_l219_219469

theorem angle_same_terminal_side (α θ : ℝ) (hα : α = 1690) (hθ : 0 < θ) (hθ2 : θ < 360) (h_terminal_side : ∃ k : ℤ, α = k * 360 + θ) : θ = 250 :=
by
  sorry

end angle_same_terminal_side_l219_219469


namespace sheryll_paid_total_l219_219002

-- Variables/conditions
variables (cost_per_book : ℝ) (num_books : ℕ) (discount_per_book : ℝ)

-- Given conditions
def assumption1 : cost_per_book = 5 := by sorry
def assumption2 : num_books = 10 := by sorry
def assumption3 : discount_per_book = 0.5 := by sorry

-- Theorem statement
theorem sheryll_paid_total : cost_per_book = 5 → num_books = 10 → discount_per_book = 0.5 → 
  (cost_per_book - discount_per_book) * num_books = 45 := by
  sorry

end sheryll_paid_total_l219_219002


namespace unique_ordered_pairs_satisfying_equation_l219_219820

theorem unique_ordered_pairs_satisfying_equation :
  ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x^6 * y^6 - 19 * x^3 * y^3 + 18 = 0 ↔ (x, y) = (1, 1) ∧
  (∀ x y : ℕ, 0 < x ∧ 0 < y ∧ x^6 * y^6 - 19 * x^3 * y^3 + 18 = 0 → (x, y) = (1, 1)) :=
by
  sorry

end unique_ordered_pairs_satisfying_equation_l219_219820


namespace intersection_of_A_and_B_l219_219989

-- Definitions of sets A and B
def A : Set ℤ := {1, 0, 3}
def B : Set ℤ := {-1, 1, 2, 3}

-- Statement of the theorem
theorem intersection_of_A_and_B : A ∩ B = {1, 3} :=
  sorry

end intersection_of_A_and_B_l219_219989


namespace positive_difference_of_two_numbers_l219_219324

theorem positive_difference_of_two_numbers :
  ∃ (x y : ℤ), (x + y = 40) ∧ (3 * y - 2 * x = 8) ∧ (|y - x| = 4) :=
by
  sorry

end positive_difference_of_two_numbers_l219_219324


namespace meaningful_fraction_l219_219473

theorem meaningful_fraction (x : ℝ) : (x + 1 ≠ 0) ↔ (x ≠ -1) :=
by
  sorry

end meaningful_fraction_l219_219473


namespace apples_remaining_l219_219497

-- Define the initial conditions
def number_of_trees := 52
def apples_on_tree_before := 9
def apples_picked := 2

-- Define the target proof: the number of apples remaining on the tree
def apples_on_tree_after := apples_on_tree_before - apples_picked

-- The statement we aim to prove
theorem apples_remaining : apples_on_tree_after = 7 := sorry

end apples_remaining_l219_219497


namespace sum_arithmetic_series_l219_219006

theorem sum_arithmetic_series :
  let a1 := 1000
  let an := 5000
  let d := 4
  let n := (an - a1) / d + 1
  let Sn := n * (a1 + an) / 2
  Sn = 3003000 := by
    sorry

end sum_arithmetic_series_l219_219006


namespace total_revenue_correct_l219_219870

def small_slices_price := 150
def large_slices_price := 250
def total_slices_sold := 5000
def small_slices_sold := 2000

def large_slices_sold := total_slices_sold - small_slices_sold

def revenue_from_small_slices := small_slices_sold * small_slices_price
def revenue_from_large_slices := large_slices_sold * large_slices_price
def total_revenue := revenue_from_small_slices + revenue_from_large_slices

theorem total_revenue_correct : total_revenue = 1050000 := by
  sorry

end total_revenue_correct_l219_219870


namespace B_finishes_in_10_days_l219_219654

noncomputable def B_remaining_work_days (A_work_days : ℕ := 15) (A_initial_days_worked : ℕ := 5) (B_work_days : ℝ := 14.999999999999996) : ℝ :=
  let A_rate := 1 / A_work_days
  let B_rate := 1 / B_work_days
  let remaining_work := 1 - (A_rate * A_initial_days_worked)
  let days_for_B := remaining_work / B_rate
  days_for_B

theorem B_finishes_in_10_days :
  B_remaining_work_days 15 5 14.999999999999996 = 10 :=
by
  sorry

end B_finishes_in_10_days_l219_219654


namespace heather_bicycling_time_l219_219866

theorem heather_bicycling_time (distance speed : ℕ) (h1 : distance = 96) (h2 : speed = 6) : 
(distance / speed) = 16 := by
  sorry

end heather_bicycling_time_l219_219866


namespace max_value_q_l219_219024

noncomputable def q (A M C : ℕ) : ℕ :=
  A * M * C + A * M + M * C + C * A + A + M + C

theorem max_value_q : ∀ A M C : ℕ, A + M + C = 15 → q A M C ≤ 215 :=
by 
  sorry

end max_value_q_l219_219024


namespace teorema_dos_bicos_white_gray_eq_angle_x_l219_219314

-- Define the problem statement
theorem teorema_dos_bicos_white_gray_eq
    (n : ℕ)
    (AB CD : ℝ)
    (peaks : Fin n → ℝ)
    (white_angles gray_angles : Fin n → ℝ)
    (h_parallel : AB = CD)
    (h_white_angles : ∀ i, white_angles i = peaks i)
    (h_gray_angles : ∀ i, gray_angles i = peaks i):
    (Finset.univ.sum white_angles) = (Finset.univ.sum gray_angles) := sorry

theorem angle_x
    (AB CD : ℝ)
    (x : ℝ)
    (h_parallel : AB = CD):
    x = 32 := sorry

end teorema_dos_bicos_white_gray_eq_angle_x_l219_219314


namespace number_of_monomials_l219_219735

-- Define the degree of a monomial
def degree (x_deg y_deg z_deg : ℕ) : ℕ := x_deg + y_deg + z_deg

-- Define a condition for the coefficient of the monomial
def monomial_coefficient (coeff : ℤ) : Prop := coeff = -3

-- Define a condition for the presence of the variables x, y, z
def contains_vars (x_deg y_deg z_deg : ℕ) : Prop := x_deg ≥ 1 ∧ y_deg ≥ 1 ∧ z_deg ≥ 1

-- Define the proof for the number of such monomials
theorem number_of_monomials :
  ∃ (x_deg y_deg z_deg : ℕ), contains_vars x_deg y_deg z_deg ∧ monomial_coefficient (-3) ∧ degree x_deg y_deg z_deg = 5 ∧ (6 = 6) :=
by
  sorry

end number_of_monomials_l219_219735


namespace girl_buys_roses_l219_219104

theorem girl_buys_roses 
  (x y : ℤ)
  (h1 : y = 1)
  (h2 : x > 0)
  (h3 : (200 : ℤ) / (x + 10) < (100 : ℤ) / x)
  (h4 : (80 : ℤ) / 12 = ((100 : ℤ) / x) - ((200 : ℤ) / (x + 10))) :
  x = 5 ∧ y = 1 :=
by
  sorry

end girl_buys_roses_l219_219104


namespace running_speed_proof_l219_219445

-- Definitions used in the conditions
def num_people : ℕ := 4
def stretch_km : ℕ := 300
def bike_speed_kmph : ℕ := 50
def total_time_hours : ℚ := 19 + (1/3)

-- The running speed to be proven
def running_speed_kmph : ℚ := 15.52

-- The main statement
theorem running_speed_proof
  (num_people_eq : num_people = 4)
  (stretch_eq : stretch_km = 300)
  (bike_speed_eq : bike_speed_kmph = 50)
  (total_time_eq : total_time_hours = 19.333333333333332) :
  running_speed_kmph = 15.52 :=
sorry

end running_speed_proof_l219_219445


namespace repeating_decimals_sum_l219_219306

def repeating_decimal1 : ℚ := (2 : ℚ) / 9  -- 0.\overline{2}
def repeating_decimal2 : ℚ := (3 : ℚ) / 99 -- 0.\overline{03}

theorem repeating_decimals_sum : repeating_decimal1 + repeating_decimal2 = (25 : ℚ) / 99 :=
by
  sorry

end repeating_decimals_sum_l219_219306


namespace equal_distribution_l219_219033

def earnings : List ℕ := [30, 35, 45, 55, 65]

def total_earnings : ℕ := earnings.sum

def equal_share (total: ℕ) : ℕ := total / earnings.length

def redistribution_amount (earner: ℕ) (equal: ℕ) : ℕ := earner - equal

theorem equal_distribution :
  redistribution_amount 65 (equal_share total_earnings) = 19 :=
by
  sorry

end equal_distribution_l219_219033


namespace value_expression_l219_219050

theorem value_expression (p q : ℚ) (h : p / q = 4 / 5) : 18 / 7 + (2 * q - p) / (2 * q + p) = 3 := by 
  sorry

end value_expression_l219_219050


namespace cost_of_shoes_l219_219301

   theorem cost_of_shoes (initial_budget remaining_budget : ℝ) (H_initial : initial_budget = 999) (H_remaining : remaining_budget = 834) : 
   initial_budget - remaining_budget = 165 := by
     sorry
   
end cost_of_shoes_l219_219301


namespace union_of_A_and_B_l219_219881

def A : Set Int := {-1, 1, 2}
def B : Set Int := {-2, -1, 0}

theorem union_of_A_and_B : A ∪ B = {-2, -1, 0, 1, 2} :=
by
  sorry

end union_of_A_and_B_l219_219881


namespace karen_bonus_problem_l219_219795

theorem karen_bonus_problem (n already_graded last_two target : ℕ) (h_already_graded : already_graded = 8)
  (h_last_two : last_two = 290) (h_target : target = 600) (max_score : ℕ)
  (h_max_score : max_score = 150) (required_avg : ℕ) (h_required_avg : required_avg = 75) :
  ∃ A : ℕ, (A = 70) ∧ (target = 600) ∧ (last_two = 290) ∧ (already_graded = 8) ∧
  (required_avg = 75) := by
  sorry

end karen_bonus_problem_l219_219795


namespace solve_equation_l219_219874

theorem solve_equation (x : ℝ) : (x - 1) * (x + 1) = x - 1 → (x = 0 ∨ x = 1) :=
by
  intro h
  sorry

end solve_equation_l219_219874


namespace hexagon_transformation_l219_219164

-- Define a shape composed of 36 identical small equilateral triangles
def Shape := { s : ℕ // s = 36 }

-- Define the number of triangles needed to form a hexagon
def TrianglesNeededForHexagon : ℕ := 18

-- Proof statement: Given a shape of 36 small triangles, we need 18 more triangles to form a hexagon
theorem hexagon_transformation (shape : Shape) : TrianglesNeededForHexagon = 18 :=
by
  -- This is our formalization of the problem statement which asserts
  -- that the transformation to a hexagon needs exactly 18 additional triangles.
  sorry

end hexagon_transformation_l219_219164


namespace evaluate_expression_l219_219581

theorem evaluate_expression (x y z : ℚ) 
    (hx : x = 1 / 4) 
    (hy : y = 1 / 3) 
    (hz : z = -6) : 
    x^2 * y^3 * z^2 = 1 / 12 :=
by
  sorry

end evaluate_expression_l219_219581


namespace tan_11pi_over_6_l219_219700

theorem tan_11pi_over_6 : Real.tan (11 * Real.pi / 6) = - (Real.sqrt 3 / 3) :=
by
  sorry

end tan_11pi_over_6_l219_219700


namespace complete_work_in_days_l219_219593

def rate_x : ℚ := 1 / 10
def rate_y : ℚ := 1 / 15
def rate_z : ℚ := 1 / 20

def combined_rate : ℚ := rate_x + rate_y + rate_z

theorem complete_work_in_days :
  1 / combined_rate = 60 / 13 :=
by
  -- Proof will go here
  sorry

end complete_work_in_days_l219_219593


namespace no_adjacent_same_roll_probability_l219_219791

noncomputable def probability_no_adjacent_same_roll : ℚ :=
  (1331 / 1728)

theorem no_adjacent_same_roll_probability :
  (probability_no_adjacent_same_roll = (1331 / 1728)) :=
by
  sorry

end no_adjacent_same_roll_probability_l219_219791


namespace two_buttons_diff_size_color_l219_219311

variables (box : Type) 
variable [Finite box]
variables (Big Small White Black : box → Prop)

axiom big_ex : ∃ x, Big x
axiom small_ex : ∃ x, Small x
axiom white_ex : ∃ x, White x
axiom black_ex : ∃ x, Black x
axiom size : ∀ x, Big x ∨ Small x
axiom color : ∀ x, White x ∨ Black x

theorem two_buttons_diff_size_color : 
  ∃ x y, x ≠ y ∧ (Big x ∧ Small y ∨ Small x ∧ Big y) ∧ (White x ∧ Black y ∨ Black x ∧ White y) := 
by
  sorry

end two_buttons_diff_size_color_l219_219311


namespace min_adj_white_pairs_l219_219131

theorem min_adj_white_pairs (black_cells : Finset (Fin 64)) (h_black_count : black_cells.card = 20) : 
  ∃ rem_white_pairs, rem_white_pairs = 34 := 
sorry

end min_adj_white_pairs_l219_219131


namespace fred_baseball_cards_l219_219370

variable (initial_cards : ℕ)
variable (bought_cards : ℕ)

theorem fred_baseball_cards (h1 : initial_cards = 5) (h2 : bought_cards = 3) : initial_cards - bought_cards = 2 := by
  sorry

end fred_baseball_cards_l219_219370


namespace fraction_simplification_l219_219789

theorem fraction_simplification :
  8 * (15 / 11) * (-25 / 40) = -15 / 11 :=
by
  sorry

end fraction_simplification_l219_219789


namespace max_non_overlapping_squares_l219_219773

theorem max_non_overlapping_squares (m n : ℕ) : 
  ∃ max_squares : ℕ, max_squares = m :=
by
  sorry

end max_non_overlapping_squares_l219_219773


namespace largest_x_l219_219059

theorem largest_x (x : ℝ) (h : ⌊x⌋ / x = 7 / 8) : x = 48 / 7 := 
sorry

end largest_x_l219_219059


namespace volume_rotation_l219_219858

theorem volume_rotation
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (a b : ℝ)
  (h₁ : ∀ (x : ℝ), f x = x^3)
  (h₂ : ∀ (x : ℝ), g x = x^(1/2))
  (h₃ : a = 0)
  (h₄ : b = 1):
  ∫ x in a..b, π * ((g x)^2 - (f x)^2) = 5 * π / 14 :=
by
  sorry

end volume_rotation_l219_219858


namespace total_books_l219_219183

def shelves : ℕ := 150
def books_per_shelf : ℕ := 15

theorem total_books (shelves books_per_shelf : ℕ) : shelves * books_per_shelf = 2250 := by
  sorry

end total_books_l219_219183


namespace salt_solution_concentration_l219_219372

theorem salt_solution_concentration :
  ∀ (C : ℝ),
  (∀ (mix_vol : ℝ) (pure_water : ℝ) (salt_solution_vol : ℝ),
    mix_vol = 1.5 →
    pure_water = 1 →
    salt_solution_vol = 0.5 →
    1.5 * 0.15 = 0.5 * (C / 100) →
    C = 45) :=
by
  intros C mix_vol pure_water salt_solution_vol h_mix h_pure h_salt h_eq
  sorry

end salt_solution_concentration_l219_219372


namespace cos_alpha_l219_219386

theorem cos_alpha (α : ℝ) (h1 : 0 < α ∧ α < π/2) 
  (h2 : Real.sin (α - π / 6) = 1 / 3) : 
  Real.cos α = (2 * Real.sqrt 6 - 1) / 6 :=
by
  sorry

end cos_alpha_l219_219386


namespace unique_intersection_y_eq_bx2_5x_2_y_eq_neg2x_neg2_iff_b_eq_49_div_16_l219_219393

theorem unique_intersection_y_eq_bx2_5x_2_y_eq_neg2x_neg2_iff_b_eq_49_div_16 
  (b : ℝ) : 
  (∃ (x : ℝ), bx^2 + 7*x + 4 = 0 ∧ ∀ (x' : ℝ), bx^2 + 7*x' + 4 ≠ 0) ↔ b = 49 / 16 :=
by
  sorry

end unique_intersection_y_eq_bx2_5x_2_y_eq_neg2x_neg2_iff_b_eq_49_div_16_l219_219393


namespace floor_sufficient_but_not_necessary_l219_219534

theorem floor_sufficient_but_not_necessary {x y : ℝ} : 
  (∀ x y : ℝ, (⌊x⌋₊ = ⌊y⌋₊) → abs (x - y) < 1) ∧ 
  ¬ (∀ x y : ℝ, abs (x - y) < 1 → (⌊x⌋₊ = ⌊y⌋₊)) :=
by
  sorry

end floor_sufficient_but_not_necessary_l219_219534


namespace find_b_l219_219264

theorem find_b (b : ℝ) : (∃ c : ℝ, (16 : ℝ) * x^2 + 40 * x + b = (4 * x + c)^2) → b = 25 :=
by
  sorry

end find_b_l219_219264


namespace tallest_building_height_l219_219165

theorem tallest_building_height :
  ∃ H : ℝ, H + (1/2) * H + (1/4) * H + (1/20) * H = 180 ∧ H = 100 := by
  sorry

end tallest_building_height_l219_219165


namespace relationship_between_y1_y2_l219_219474

theorem relationship_between_y1_y2 (b y1 y2 : ℝ) 
  (h1 : y1 = -(-2) + b) 
  (h2 : y2 = -(3) + b) : 
  y1 > y2 := 
by {
  sorry
}

end relationship_between_y1_y2_l219_219474


namespace max_ab_l219_219088

theorem max_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : ab ≤ 1 / 4 :=
sorry

end max_ab_l219_219088


namespace abs_eq_solution_l219_219923

theorem abs_eq_solution (x : ℝ) (h : abs (x - 3) = abs (x + 2)) : x = 1 / 2 :=
sorry

end abs_eq_solution_l219_219923


namespace units_digit_3968_805_l219_219318

theorem units_digit_3968_805 : 
  (3968 ^ 805) % 10 = 8 := 
by
  -- Proof goes here
  sorry

end units_digit_3968_805_l219_219318


namespace sum_of_dimensions_l219_219108

noncomputable def rectangular_prism_dimensions (A B C : ℝ) : Prop :=
  (A * B = 30) ∧ (A * C = 40) ∧ (B * C = 60)

theorem sum_of_dimensions (A B C : ℝ) (h : rectangular_prism_dimensions A B C) : A + B + C = 9 * Real.sqrt 5 :=
by
  sorry

end sum_of_dimensions_l219_219108


namespace find_k_l219_219103

theorem find_k (k : ℝ) : 
  (1 : ℝ)^2 + k * 1 - 3 = 0 → k = 2 :=
by
  intro h
  sorry

end find_k_l219_219103


namespace tylers_age_l219_219419

theorem tylers_age (B T : ℕ) 
  (h1 : T = B - 3) 
  (h2 : T + B = 11) : 
  T = 4 :=
sorry

end tylers_age_l219_219419


namespace example_proof_l219_219357

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom axiom1 (x y : ℝ) : f (x - y) = f x * g y - g x * f y
axiom axiom2 (x : ℝ) : f x ≠ 0
axiom axiom3 : f 1 = f 2

theorem example_proof : g (-1) + g 1 = 1 := by
  sorry

end example_proof_l219_219357


namespace ellipsoid_volume_div_pi_l219_219793

noncomputable def ellipsoid_projection_min_area : ℝ := 9 * Real.pi
noncomputable def ellipsoid_projection_max_area : ℝ := 25 * Real.pi
noncomputable def ellipsoid_circle_projection_area : ℝ := 16 * Real.pi
noncomputable def ellipsoid_volume (a b c : ℝ) : ℝ := (4/3) * Real.pi * a * b * c

theorem ellipsoid_volume_div_pi (a b c : ℝ)
  (h_min : (a * b = 9))
  (h_max : (b * c = 25))
  (h_circle : (b = 4)) :
  ellipsoid_volume a b c / Real.pi = 75 := 
  by
    sorry

end ellipsoid_volume_div_pi_l219_219793


namespace inverse_function_point_l219_219568

noncomputable def f (a : ℝ) (x : ℝ) := a^(x + 1)

theorem inverse_function_point (a : ℝ) (h_pos : 0 < a) (h_annoylem : f a (-1) = 1) :
  ∃ g : ℝ → ℝ, (∀ y, f a (g y) = y ∧ g (f a y) = y) ∧ g 1 = -1 :=
by
  sorry

end inverse_function_point_l219_219568


namespace length_AB_of_parabola_l219_219837

theorem length_AB_of_parabola (x1 x2 : ℝ)
  (h : x1 + x2 = 6) :
  abs (x1 + x2 + 2) = 8 := by
  sorry

end length_AB_of_parabola_l219_219837


namespace delores_remaining_money_l219_219285

variable (delores_money : ℕ := 450)
variable (computer_price : ℕ := 1000)
variable (computer_discount : ℝ := 0.30)
variable (printer_price : ℕ := 100)
variable (printer_tax_rate : ℝ := 0.15)
variable (table_price_euros : ℕ := 200)
variable (exchange_rate : ℝ := 1.2)

def computer_sale_price : ℝ := computer_price * (1 - computer_discount)
def printer_total_cost : ℝ := printer_price * (1 + printer_tax_rate)
def table_cost_dollars : ℝ := table_price_euros * exchange_rate
def total_cost : ℝ := computer_sale_price + printer_total_cost + table_cost_dollars
def remaining_money : ℝ := delores_money - total_cost

theorem delores_remaining_money : remaining_money = -605 := by
  sorry

end delores_remaining_money_l219_219285


namespace combined_average_score_l219_219704

theorem combined_average_score (M E : ℕ) (m e : ℕ) (h1 : M = 82) (h2 : E = 68) (h3 : m = 5 * e / 7) :
  ((m * M) + (e * E)) / (m + e) = 72 :=
by
  -- Placeholder for the proof
  sorry

end combined_average_score_l219_219704


namespace problem_statement_l219_219671

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
  if x ≥ 0 then 2^x + 2*x + b else -(2^(-x) + 2*(-x) + b)

theorem problem_statement (b : ℝ) (hb : 2^0 + 2*0 + b = 0) : f (-1) b = -3 :=
by
  sorry

end problem_statement_l219_219671


namespace additional_interest_due_to_higher_rate_l219_219118

def principal : ℝ := 2500
def rate1 : ℝ := 0.18
def rate2 : ℝ := 0.12
def time : ℝ := 2

def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t

theorem additional_interest_due_to_higher_rate :
  simple_interest principal rate1 time - simple_interest principal rate2 time = 300 :=
by
  sorry

end additional_interest_due_to_higher_rate_l219_219118


namespace regular_octagon_angle_ABG_l219_219342

-- Definition of a regular octagon
structure RegularOctagon (V : Type) :=
(vertices : Fin 8 → V)

def angleABG (O : RegularOctagon ℝ) : ℝ :=
  22.5

-- The statement: In a regular octagon ABCDEFGH, the measure of ∠ABG is 22.5°
theorem regular_octagon_angle_ABG (O : RegularOctagon ℝ) : angleABG O = 22.5 :=
  sorry

end regular_octagon_angle_ABG_l219_219342


namespace rectangle_area_l219_219585

theorem rectangle_area (P W : ℝ) (hP : P = 52) (hW : W = 11) :
  ∃ A L : ℝ, (2 * L + 2 * W = P) ∧ (A = L * W) ∧ (A = 165) :=
by
  sorry

end rectangle_area_l219_219585


namespace smallest_b_for_q_ge_half_l219_219322

open Nat

def binomial (n k : ℕ) : ℕ := if h : k ≤ n then n.choose k else 0

def q (b : ℕ) : ℚ := (binomial (32 - b) 2 + binomial (b - 1) 2) / (binomial 38 2 : ℕ)

theorem smallest_b_for_q_ge_half : ∃ (b : ℕ), b = 18 ∧ q b ≥ 1 / 2 :=
by
  -- Prove and find the smallest b such that q(b) ≥ 1/2
  sorry

end smallest_b_for_q_ge_half_l219_219322


namespace value_of_t_plus_one_over_t_l219_219692

theorem value_of_t_plus_one_over_t
  (t : ℝ)
  (h1 : t^2 - 3 * t + 1 = 0)
  (h2 : t ≠ 0) :
  t + 1 / t = 3 :=
by
  sorry

end value_of_t_plus_one_over_t_l219_219692


namespace simplify_expression_l219_219723

theorem simplify_expression (x y : ℤ) (h1 : x = -1) (h2 : y = 2) :
  (3 * x^2 * y - 2 * x * y^2) - (x * y^2 - 2 * x^2 * y) - 2 * (-3 * x^2 * y - x * y^2) = 26 :=
by
  rw [h1, h2]
  sorry

end simplify_expression_l219_219723


namespace evaluate_expression_l219_219898

theorem evaluate_expression {x y : ℕ} (h₁ : 144 = 2^x * 3^y) (hx : x = 4) (hy : y = 2) : (1 / 7) ^ (y - x) = 49 := 
by
  sorry

end evaluate_expression_l219_219898


namespace quadratic_solution_unique_l219_219515

theorem quadratic_solution_unique (b : ℝ) (hb : b ≠ 0) (hdisc : 30 * 30 - 4 * b * 10 = 0) :
  ∃ x : ℝ, bx ^ 2 + 30 * x + 10 = 0 ∧ x = -2 / 3 :=
by
  sorry

end quadratic_solution_unique_l219_219515


namespace cubic_roots_l219_219241

theorem cubic_roots (a b x₃ : ℤ)
  (h1 : (2^3 + a * 2^2 + b * 2 + 6 = 0))
  (h2 : (3^3 + a * 3^2 + b * 3 + 6 = 0))
  (h3 : 2 * 3 * x₃ = -6) :
  a = -4 ∧ b = 1 ∧ x₃ = -1 :=
by {
  sorry
}

end cubic_roots_l219_219241


namespace line_divides_circle_l219_219274

theorem line_divides_circle (k m : ℝ) :
  (∀ x y : ℝ, y = x - 1 → x^2 + y^2 + k*x + m*y - 4 = 0 → m - k = 2) :=
sorry

end line_divides_circle_l219_219274


namespace length_of_fourth_side_in_cyclic_quadrilateral_l219_219524

theorem length_of_fourth_side_in_cyclic_quadrilateral :
  ∀ (r a b c : ℝ), r = 300 ∧ a = 300 ∧ b = 300 ∧ c = 150 * Real.sqrt 2 →
  ∃ d : ℝ, d = 450 :=
by
  sorry

end length_of_fourth_side_in_cyclic_quadrilateral_l219_219524


namespace num_seven_digit_palindromes_l219_219053

theorem num_seven_digit_palindromes : 
  let a_choices := 9
  let b_choices := 10
  let c_choices := 10
  let d_choices := 10
  a_choices * b_choices * c_choices * d_choices = 9000 :=
by
  sorry

end num_seven_digit_palindromes_l219_219053


namespace quadratic_y_at_x_5_l219_219869

-- Define the quadratic function
noncomputable def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the conditions and question as part of a theorem
theorem quadratic_y_at_x_5 (a b c : ℝ) 
  (h1 : ∀ x, quadratic a b c x ≤ 10) -- Maximum value condition (The maximum value is 10)
  (h2 : (quadratic a b c (-2)) = 10) -- y = 10 when x = -2 (maximum point)
  (h3 : quadratic a b c 0 = -8) -- The first point (0, -8)
  (h4 : quadratic a b c 1 = 0) -- The second point (1, 0)
  : quadratic a b c 5 = -400 / 9 :=
sorry

end quadratic_y_at_x_5_l219_219869


namespace probability_of_one_triplet_without_any_pairs_l219_219003

noncomputable def probability_one_triplet_no_pairs : ℚ :=
  let total_outcomes := 6^5
  let choices_for_triplet := 6
  let ways_to_choose_triplet_dice := Nat.choose 5 3
  let choices_for_remaining_dice := 5 * 4
  let successful_outcomes := choices_for_triplet * ways_to_choose_triplet_dice * choices_for_remaining_dice
  successful_outcomes / total_outcomes

theorem probability_of_one_triplet_without_any_pairs :
  probability_one_triplet_no_pairs = 25 / 129 := by
  sorry

end probability_of_one_triplet_without_any_pairs_l219_219003


namespace kim_probability_same_color_l219_219096

noncomputable def probability_same_color (total_shoes : ℕ) (pairs_of_shoes : ℕ) : ℚ :=
  let total_selections := (total_shoes * (total_shoes - 1)) / 2
  let successful_selections := pairs_of_shoes
  successful_selections / total_selections

theorem kim_probability_same_color :
  probability_same_color 10 5 = 1 / 9 :=
by
  unfold probability_same_color
  have h_total : (10 * 9) / 2 = 45 := by norm_num
  have h_success : 5 = 5 := by norm_num
  rw [h_total, h_success]
  norm_num
  done

end kim_probability_same_color_l219_219096


namespace min_value_frac_sum_l219_219706

variable {a b c : ℝ}

theorem min_value_frac_sum (h1 : 0 < a ∧ a < 1) (h2 : 0 < b ∧ b < 1) (h3 : 0 < c ∧ c < 1) (h4 : a * b + b * c + c * a = 1) : 
  (1 / (1 - a)) + (1 / (1 - b)) + (1 / (1 - c)) = (9 + 3 * Real.sqrt 3) / 2 :=
  sorry

end min_value_frac_sum_l219_219706


namespace seq_arithmetic_l219_219928

theorem seq_arithmetic (a : ℕ → ℕ) (h : ∀ p q : ℕ, a p + a q = a (p + q)) (h1 : a 1 = 2) :
  ∀ n : ℕ, a n = 2 * n :=
by
  sorry

end seq_arithmetic_l219_219928


namespace minimum_distance_on_circle_l219_219871

open Complex

noncomputable def minimum_distance (z : ℂ) : ℝ :=
  abs (z - (1 + 2*I))

theorem minimum_distance_on_circle :
  ∀ z : ℂ, abs (z + 2 - 2*I) = 1 → minimum_distance z = 2 :=
by
  intros z hz
  -- Proof is omitted
  sorry

end minimum_distance_on_circle_l219_219871


namespace scientific_notation_15_7_trillion_l219_219896

theorem scientific_notation_15_7_trillion :
  ∃ n : ℝ, n = 15.7 * 10^12 ∧ n = 1.57 * 10^13 :=
by
  sorry

end scientific_notation_15_7_trillion_l219_219896


namespace none_of_these_l219_219304

theorem none_of_these (s x y : ℝ) (hs : s > 1) (hx2y_ne_zero : x^2 * y ≠ 0) (hineq : x * s^2 > y * s^2) :
  ¬ (-x > -y) ∧ ¬ (-x > y) ∧ ¬ (1 > -y / x) ∧ ¬ (1 < y / x) :=
by
  sorry

end none_of_these_l219_219304


namespace find_r_floor_r_add_r_eq_18point2_l219_219880

theorem find_r_floor_r_add_r_eq_18point2 (r : ℝ) (h : ⌊r⌋ + r = 18.2) : r = 9.2 := 
sorry

end find_r_floor_r_add_r_eq_18point2_l219_219880


namespace largest_n_unique_k_l219_219409

theorem largest_n_unique_k : ∃ n : ℕ, (∀ k : ℤ, (8 / 15 : ℚ) < (n : ℚ) / (n + k) ∧ (n : ℚ) / (n + k) < (7 / 13 : ℚ) → k = unique_k) ∧ n = 112 :=
sorry

end largest_n_unique_k_l219_219409


namespace total_travel_time_l219_219759

-- Defining the conditions
def car_travel_180_miles_in_4_hours : Prop :=
  180 / 4 = 45

def car_travel_135_miles_additional_time : Prop :=
  135 / 45 = 3

-- The main statement to be proved
theorem total_travel_time : car_travel_180_miles_in_4_hours ∧ car_travel_135_miles_additional_time → 4 + 3 = 7 := by
  sorry

end total_travel_time_l219_219759


namespace y_comparison_l219_219339

theorem y_comparison :
  let y1 := (-1)^2 - 2*(-1) + 3
  let y2 := (-2)^2 - 2*(-2) + 3
  y2 > y1 := by
  sorry

end y_comparison_l219_219339


namespace beginning_of_spring_period_and_day_l219_219744

noncomputable def daysBetween : Nat := 46 -- Total days: Dec 21, 2004 to Feb 4, 2005

theorem beginning_of_spring_period_and_day :
  let total_days := daysBetween
  let segment := total_days / 9
  let day_within_segment := total_days % 9
  segment = 5 ∧ day_within_segment = 1 := by
sorry

end beginning_of_spring_period_and_day_l219_219744


namespace smallest_N_l219_219292

-- Definitions for the problem conditions
def is_rectangular_block (a b c : ℕ) (N : ℕ) : Prop :=
  N = a * b * c ∧ 143 = (a - 1) * (b - 1) * (c - 1)

-- Theorem to prove the smallest possible value of N
theorem smallest_N : ∃ a b c : ℕ, is_rectangular_block a b c 336 :=
by
  sorry

end smallest_N_l219_219292


namespace points_on_circle_l219_219718

theorem points_on_circle (t : ℝ) : 
  let x := (2 - t^2) / (2 + t^2)
  let y := (3 * t) / (2 + t^2)
  x^2 + y^2 = 1 := 
by 
  let x := (2 - t^2) / (2 + t^2)
  let y := (3 * t) / (2 + t^2)
  sorry

end points_on_circle_l219_219718


namespace total_length_of_ropes_l219_219294

theorem total_length_of_ropes (L : ℝ) 
  (h1 : (L - 12 = 4 * (L - 42))) : 
  2 * L = 104 := 
by
  sorry

end total_length_of_ropes_l219_219294


namespace ratio_sum_of_squares_l219_219157

theorem ratio_sum_of_squares (a b c : ℕ) (h : a = 6 ∧ b = 1 ∧ c = 7 ∧ 72 / 98 = (a * (b.sqrt^2)).sqrt / c) : a + b + c = 14 := by 
  sorry

end ratio_sum_of_squares_l219_219157


namespace shop_owner_profitable_l219_219007

noncomputable def shop_owner_profit (CP_SP_difference_percentage: ℚ) (CP: ℚ) (buy_cheat_percentage: ℚ) (sell_cheat_percentage: ℚ) (buy_discount_percentage: ℚ) (sell_markup_percentage: ℚ) : ℚ := 
  CP_SP_difference_percentage * 100

theorem shop_owner_profitable :
  shop_owner_profit ((114 * (110 / 80 / 100) - 90) / 90) 1 0.14 0.20 0.10 0.10 = 74.17 := 
by
  sorry

end shop_owner_profitable_l219_219007


namespace maximize_S_n_l219_219350

variable (a_1 d : ℝ)
noncomputable def S (n : ℕ) := n * a_1 + (n * (n - 1) / 2) * d

theorem maximize_S_n {n : ℕ} (h1 : S 17 > 0) (h2 : S 18 < 0) : n = 9 := sorry

end maximize_S_n_l219_219350


namespace smallest_solution_is_9_l219_219136

noncomputable def smallest_positive_solution (x : ℝ) : Prop :=
  (3*x / (x - 3) + (3*x^2 - 45) / (x + 3) = 14) ∧ (x > 3) ∧ (∀ y : ℝ, (3*y / (y - 3) + (3*y^2 - 45) / (y + 3) = 14) → (y > 3) → (y ≥ 9))

theorem smallest_solution_is_9 : ∃ x : ℝ, smallest_positive_solution x ∧ x = 9 :=
by
  exists 9
  have : smallest_positive_solution 9 := sorry
  exact ⟨this, rfl⟩

end smallest_solution_is_9_l219_219136


namespace previous_spider_weight_l219_219565

noncomputable def giant_spider_weight (prev_spider_weight : ℝ) : ℝ :=
  2.5 * prev_spider_weight

noncomputable def leg_cross_sectional_area : ℝ := 0.5
noncomputable def leg_pressure : ℝ := 4
noncomputable def legs : ℕ := 8

noncomputable def force_per_leg : ℝ := leg_pressure * leg_cross_sectional_area
noncomputable def total_weight : ℝ := force_per_leg * (legs : ℝ)

theorem previous_spider_weight (prev_spider_weight : ℝ) (h_giant : giant_spider_weight prev_spider_weight = total_weight) : prev_spider_weight = 6.4 :=
by
  sorry

end previous_spider_weight_l219_219565


namespace dessert_probability_l219_219169

noncomputable def P (e : Prop) : ℝ := sorry

variables (D C : Prop)

theorem dessert_probability 
  (P_D : P D = 0.6)
  (P_D_and_not_C : P (D ∧ ¬C) = 0.12) :
  P (¬ D) = 0.4 :=
by
  -- Proof is skipped using sorry, as instructed.
  sorry

end dessert_probability_l219_219169


namespace cost_price_of_radio_l219_219302

-- Definitions for conditions
def selling_price := 1245
def loss_percentage := 17

-- Prove that the cost price is Rs. 1500 given the conditions
theorem cost_price_of_radio : 
  ∃ C, (C - 1245) * 100 / C = 17 ∧ C = 1500 := 
sorry

end cost_price_of_radio_l219_219302


namespace trapezoid_area_is_correct_l219_219808

noncomputable def trapezoid_area (base_short : ℝ) (angle_adj : ℝ) (angle_diag : ℝ) : ℝ :=
  let width := 2 * base_short -- calculated width from angle_adj
  let height := base_short / Real.tan (angle_adj / 2 * Real.pi / 180)
  (base_short + base_short + width) * height / 2

theorem trapezoid_area_is_correct :
  trapezoid_area 2 135 150 = 2 :=
by
  sorry

end trapezoid_area_is_correct_l219_219808


namespace pipe_R_fill_time_l219_219167

theorem pipe_R_fill_time (P_rate Q_rate combined_rate : ℝ) (hP : P_rate = 1 / 2) (hQ : Q_rate = 1 / 4)
  (h_combined : combined_rate = 1 / 1.2) : (∃ R_rate : ℝ, R_rate = 1 / 12) :=
by
  sorry

end pipe_R_fill_time_l219_219167


namespace largest_whole_number_lt_150_l219_219197

theorem largest_whole_number_lt_150 : ∃ (x : ℕ), (x <= 16 ∧ ∀ y : ℕ, y < 17 → 9 * y < 150) :=
by
  sorry

end largest_whole_number_lt_150_l219_219197


namespace percent_runs_by_running_between_wickets_l219_219230

theorem percent_runs_by_running_between_wickets :
  (132 - (12 * 4 + 2 * 6)) / 132 * 100 = 54.54545454545455 :=
by
  sorry

end percent_runs_by_running_between_wickets_l219_219230


namespace evaluate_x_squared_minus_y_squared_l219_219509

theorem evaluate_x_squared_minus_y_squared
  (x y : ℝ)
  (h1 : x + y = 12)
  (h2 : 3 * x + y = 18) :
  x^2 - y^2 = -72 := 
sorry

end evaluate_x_squared_minus_y_squared_l219_219509


namespace sufficient_and_necessary_cond_l219_219980

theorem sufficient_and_necessary_cond (x : ℝ) : |x| > 2 ↔ (x > 2) :=
sorry

end sufficient_and_necessary_cond_l219_219980


namespace necessarily_positive_l219_219437

theorem necessarily_positive (a b c : ℝ) (ha : 0 < a ∧ a < 2) (hb : -2 < b ∧ b < 0) (hc : 0 < c ∧ c < 3) :
  (b + c) > 0 :=
sorry

end necessarily_positive_l219_219437


namespace andre_tuesday_ladybugs_l219_219188

theorem andre_tuesday_ladybugs (M T : ℕ) (dots_per_ladybug total_dots monday_dots tuesday_dots : ℕ)
  (h1 : M = 8)
  (h2 : dots_per_ladybug = 6)
  (h3 : total_dots = 78)
  (h4 : monday_dots = M * dots_per_ladybug)
  (h5 : tuesday_dots = total_dots - monday_dots)
  (h6 : tuesday_dots = T * dots_per_ladybug) :
  T = 5 :=
sorry

end andre_tuesday_ladybugs_l219_219188


namespace math_equivalent_problem_l219_219415

noncomputable def correct_difference (A B C D : ℕ) (incorrect_difference : ℕ) : ℕ :=
  if (B = 3) ∧ (D = 2) ∧ (C = 5) ∧ (incorrect_difference = 60) then
    ((A * 10 + B) - 52)
  else
    0

theorem math_equivalent_problem (A : ℕ) : correct_difference A 3 5 2 60 = 31 :=
by
  sorry

end math_equivalent_problem_l219_219415


namespace electronics_weight_l219_219056

variable (B C E : ℝ)

-- Conditions
def initial_ratio : Prop := B / 5 = C / 4 ∧ C / 4 = E / 2
def removed_clothes : Prop := B / 10 = (C - 9) / 4

-- Proof statement
theorem electronics_weight (h1 : initial_ratio B C E) (h2 : removed_clothes B C) : E = 9 := 
by
  sorry

end electronics_weight_l219_219056


namespace remainder_when_divided_by_eleven_l219_219741

-- Definitions from the conditions
def two_pow_five_mod_eleven : ℕ := 10
def two_pow_ten_mod_eleven : ℕ := 1
def ten_mod_eleven : ℕ := 10
def ten_square_mod_eleven : ℕ := 1

-- Proposition we want to prove
theorem remainder_when_divided_by_eleven :
  (7 * 10^20 + 2^20) % 11 = 8 := 
by 
  -- Proof goes here
  sorry

end remainder_when_divided_by_eleven_l219_219741


namespace evaluate_expression_l219_219282

theorem evaluate_expression : ((3 ^ 2) ^ 3) - ((2 ^ 3) ^ 2) = 665 := by
  sorry

end evaluate_expression_l219_219282


namespace find_number_l219_219777

theorem find_number {x : ℤ} (h : x + 5 = 6) : x = 1 :=
sorry

end find_number_l219_219777


namespace sellingPrice_is_459_l219_219857

-- Definitions based on conditions
def costPrice : ℝ := 540
def markupPercentage : ℝ := 0.15
def discountPercentage : ℝ := 0.2608695652173913

-- Calculating the marked price based on the given conditions
def markedPrice (cp : ℝ) (markup : ℝ) : ℝ := cp + (markup * cp)

-- Calculating the discount amount based on the marked price and the discount percentage
def discount (mp : ℝ) (discountPct : ℝ) : ℝ := discountPct * mp

-- Calculating the selling price
def sellingPrice (mp : ℝ) (discountAmt : ℝ) : ℝ := mp - discountAmt

-- Stating the final proof problem
theorem sellingPrice_is_459 :
  sellingPrice (markedPrice costPrice markupPercentage) (discount (markedPrice costPrice markupPercentage) discountPercentage) = 459 :=
by
  sorry

end sellingPrice_is_459_l219_219857


namespace find_g_of_2_l219_219839

open Real

noncomputable def g : ℝ → ℝ := sorry

theorem find_g_of_2
  (H: ∀ x : ℝ, g (2 ^ x) + x * g (2 ^ (-x)) + x = 1) : g 2 = -1 :=
by
  sorry

end find_g_of_2_l219_219839


namespace intersection_eq_l219_219620

def M (x : ℝ) : Prop := (x + 3) * (x - 2) < 0

def N (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 3

def intersection (x : ℝ) : Prop := M x ∧ N x

theorem intersection_eq : ∀ x, intersection x ↔ (1 ≤ x ∧ x < 2) :=
by sorry

end intersection_eq_l219_219620


namespace pqr_problem_l219_219490

noncomputable def pqr_sums_to_44 (p q r : ℝ) : Prop :=
  (p < q) ∧ (∀ x, (x < -6 ∨ |x - 20| ≤ 2) ↔ ( (x - p) * (x - q) / (x - r) ≥ 0 ))

theorem pqr_problem (p q r : ℝ) (h : pqr_sums_to_44 p q r) : p + 2*q + 3*r = 44 :=
sorry

end pqr_problem_l219_219490


namespace probability_no_shaded_square_l219_219147

theorem probability_no_shaded_square : 
  let n : ℕ := 502 * 1004
  let m : ℕ := 502^2
  let total_rectangles := 3 * n
  let rectangles_with_shaded := 3 * m
  let probability_includes_shaded := rectangles_with_shaded / total_rectangles
  1 - probability_includes_shaded = (1 : ℚ) / 2 := 
by 
  sorry

end probability_no_shaded_square_l219_219147


namespace find_width_l219_219804

variable (L W : ℕ)

def perimeter (L W : ℕ) : ℕ := 2 * L + 2 * W

theorem find_width (h1 : perimeter L W = 46) (h2 : W = L + 7) : W = 15 :=
sorry

end find_width_l219_219804


namespace fraction_order_l219_219876

theorem fraction_order :
  (25 / 19 : ℚ) < (21 / 16 : ℚ) ∧ (21 / 16 : ℚ) < (23 / 17 : ℚ) := by
  sorry

end fraction_order_l219_219876


namespace trigonometric_identity_l219_219538

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 4) :
  (Real.sin θ + Real.cos θ) / (17 * Real.sin θ) + (Real.sin θ ^ 2) / 4 = 21 / 68 := 
sorry

end trigonometric_identity_l219_219538


namespace part_I_part_II_l219_219809

open Set

-- Define the sets A and B
def A : Set ℝ := { x | 1 < x ∧ x < 2 }
def B (a : ℝ) : Set ℝ := { x | 2 * a - 1 < x ∧ x < 2 * a + 1 }

-- Part (Ⅰ): Given A ⊆ B, prove that 1/2 ≤ a ≤ 1
theorem part_I (a : ℝ) : A ⊆ B a → (1 / 2 ≤ a ∧ a ≤ 1) :=
by sorry

-- Part (Ⅱ): Given A ∩ B = ∅, prove that a ≥ 3/2 or a ≤ 0
theorem part_II (a : ℝ) : A ∩ B a = ∅ → (a ≥ 3 / 2 ∨ a ≤ 0) :=
by sorry

end part_I_part_II_l219_219809


namespace silk_dyed_amount_l219_219440

-- Define the conditions
def yards_green : ℕ := 61921
def yards_pink : ℕ := 49500

-- Define the total calculation
def total_yards : ℕ := yards_green + yards_pink

-- State what needs to be proven: that the total yards is 111421
theorem silk_dyed_amount : total_yards = 111421 := by
  sorry

end silk_dyed_amount_l219_219440


namespace arithmetic_sequence_a4_l219_219752

theorem arithmetic_sequence_a4 (a1 : ℤ) (S3 : ℤ) (h1 : a1 = 3) (h2 : S3 = 15) : 
  ∃ (a4 : ℤ), a4 = 9 :=
by
  sorry

end arithmetic_sequence_a4_l219_219752


namespace half_angle_quadrants_l219_219572

variable (k : ℤ) (α : ℝ)

-- Conditions
def is_second_quadrant (α : ℝ) (k : ℤ) : Prop :=
  2 * k * Real.pi + Real.pi / 2 < α ∧ α < 2 * k * Real.pi + Real.pi

-- Question: Determine the quadrant(s) in which α / 2 lies under the given condition.
theorem half_angle_quadrants (α : ℝ) (k : ℤ) 
  (h : is_second_quadrant α k) : 
  ((k * Real.pi + Real.pi / 4 < α / 2) ∧ (α / 2 < k * Real.pi + Real.pi / 2)) ↔ 
  (∃ (m : ℤ), (2 * m * Real.pi < α / 2 ∧ α / 2 < 2 * m * Real.pi + Real.pi)) ∨ ( ∃ (m : ℤ), (2 * m * Real.pi + Real.pi < α / 2 ∧ α / 2 < 2 * m * Real.pi + 2 * Real.pi)) := 
sorry

end half_angle_quadrants_l219_219572


namespace triple_solutions_l219_219332

theorem triple_solutions (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 2 ∧ b = 2 ∧ c = 2) ↔ a! + b! = 2 ^ c! :=
by
  sorry

end triple_solutions_l219_219332


namespace trajectory_of_P_l219_219269

-- Definitions for points and distance
structure Point where
  x : ℝ
  y : ℝ

noncomputable def dist (P Q : Point) : ℝ :=
  Real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)

-- Fixed points F1 and F2
variable (F1 F2 : Point)
-- Distance condition
axiom dist_F1F2 : dist F1 F2 = 8

-- Moving point P satisfying the condition
variable (P : Point)
axiom dist_PF1_PF2 : dist P F1 + dist P F2 = 8

-- Proof goal: P lies on the line segment F1F2
theorem trajectory_of_P : ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ P = ⟨(1 - t) * F1.x + t * F2.x, (1 - t) * F1.y + t * F2.y⟩ :=
  sorry

end trajectory_of_P_l219_219269


namespace percentage_books_not_sold_is_60_percent_l219_219533

def initial_stock : ℕ := 700
def sold_monday : ℕ := 50
def sold_tuesday : ℕ := 82
def sold_wednesday : ℕ := 60
def sold_thursday : ℕ := 48
def sold_friday : ℕ := 40

def total_sold : ℕ := sold_monday + sold_tuesday + sold_wednesday + sold_thursday + sold_friday
def books_not_sold : ℕ := initial_stock - total_sold
def percentage_not_sold : ℚ := (books_not_sold * 100) / initial_stock

theorem percentage_books_not_sold_is_60_percent : percentage_not_sold = 60 := by
  sorry

end percentage_books_not_sold_is_60_percent_l219_219533


namespace smallest_n_for_cookies_l219_219892

theorem smallest_n_for_cookies :
  ∃ n : ℕ, 15 * n - 1 % 11 = 0 ∧ (∀ m : ℕ, 15 * m - 1 % 11 = 0 → n ≤ m) :=
sorry

end smallest_n_for_cookies_l219_219892


namespace price_of_each_cupcake_l219_219983

variable (x : ℝ)

theorem price_of_each_cupcake (h : 50 * x + 40 * 0.5 = 2 * 40 + 20 * 2) : x = 2 := 
by 
  sorry

end price_of_each_cupcake_l219_219983


namespace brian_video_watching_time_l219_219702

/--
Brian watches a 4-minute video of cats.
Then he watches a video twice as long as the cat video involving dogs.
Finally, he watches a video on gorillas that's twice as long as the combined duration of the first two videos.
Prove that Brian spends a total of 36 minutes watching animal videos.
-/
theorem brian_video_watching_time (cat_video dog_video gorilla_video : ℕ) 
  (h₁ : cat_video = 4) 
  (h₂ : dog_video = 2 * cat_video) 
  (h₃ : gorilla_video = 2 * (cat_video + dog_video)) : 
  cat_video + dog_video + gorilla_video = 36 := by
  sorry

end brian_video_watching_time_l219_219702


namespace measure_of_angle_x_l219_219945

-- Defining the conditions
def angle_ABC : ℝ := 108
def angle_ABD : ℝ := 180 - angle_ABC
def angle_in_triangle_ABD_1 : ℝ := 26
def sum_of_angles_in_triangle (a b c : ℝ) : Prop := a + b + c = 180

-- The theorem to prove
theorem measure_of_angle_x (h1 : angle_ABD = 72)
                           (h2 : angle_in_triangle_ABD_1 = 26)
                           (h3 : sum_of_angles_in_triangle angle_ABD angle_in_triangle_ABD_1 x) :
  x = 82 :=
by {
  -- Since this is a formal statement, we leave the proof as an exercise 
  sorry
}

end measure_of_angle_x_l219_219945


namespace shanghai_population_scientific_notation_l219_219429

theorem shanghai_population_scientific_notation :
  16.3 * 10^6 = 1.63 * 10^7 :=
sorry

end shanghai_population_scientific_notation_l219_219429


namespace solution_l219_219943

theorem solution (y : ℝ) (h : 6 * y^2 + 7 = 2 * y + 12) : (12 * y - 4)^2 = 128 :=
sorry

end solution_l219_219943


namespace games_attended_l219_219794

theorem games_attended (games_this_month games_last_month games_next_month total_games : ℕ) 
  (h1 : games_this_month = 11) 
  (h2 : games_last_month = 17) 
  (h3 : games_next_month = 16) : 
  total_games = games_this_month + games_last_month + games_next_month → 
  total_games = 44 :=
by
  sorry

end games_attended_l219_219794


namespace value_of_f_l219_219904

noncomputable
def f (k l m x : ℝ) : ℝ := k + m / (x - l)

theorem value_of_f (k l m : ℝ) (hk : k = -2) (hl : l = 2.5) (hm : m = 12) :
  f k l m (k + l + m) = -4 / 5 :=
by
  sorry

end value_of_f_l219_219904


namespace one_third_sugar_l219_219605

theorem one_third_sugar (s : ℚ) (h : s = 23 / 4) : (1 / 3) * s = 1 + 11 / 12 :=
by {
  sorry
}

end one_third_sugar_l219_219605


namespace gcd_lcm_product_l219_219300

theorem gcd_lcm_product (a b : ℕ) (ha : a = 24) (hb : b = 36) : 
  Nat.gcd a b * Nat.lcm a b = 864 :=
by
  rw [ha, hb]
  -- This theorem proves that the product of the GCD and LCM of 24 and 36 equals 864.

  sorry -- Proof will go here

end gcd_lcm_product_l219_219300


namespace positive_number_is_49_l219_219526

theorem positive_number_is_49 (a : ℝ) (x : ℝ) (h₁ : (3 - a) * (3 - a) = x) (h₂ : (2 * a + 1) * (2 * a + 1) = x) :
  x = 49 :=
sorry

end positive_number_is_49_l219_219526


namespace trapezoid_area_l219_219855

theorem trapezoid_area (EF GH EG FH : ℝ) (h : ℝ)
  (h_EF : EF = 60) (h_GH : GH = 30) (h_EG : EG = 25) (h_FH : FH = 18) (h_alt : h = 15) :
  (1 / 2 * (EF + GH) * h) = 675 :=
by
  rw [h_EF, h_GH, h_alt]
  sorry

end trapezoid_area_l219_219855


namespace min_value_expression_l219_219331

theorem min_value_expression (n : ℕ) (h : 0 < n) : 
  ∃ (m : ℕ), (m = n) ∧ (∀ k > 0, (k = n) -> (n / 3 + 27 / n) = 6) := 
sorry

end min_value_expression_l219_219331


namespace part1_solution_set_part2_range_of_a_l219_219672

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1_solution_set :
  ∀ x : ℝ, f x 1 ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
by
  intro x
  unfold f
  sorry

theorem part2_range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, f x a > -a) ↔ (a > -3 / 2) :=
by
  intro a
  unfold f
  sorry

end part1_solution_set_part2_range_of_a_l219_219672


namespace limit_of_power_seq_l219_219083

-- Define the problem and its conditions
theorem limit_of_power_seq (a : ℝ) (h : 0 < a ∨ 1 < a) :
  (0 < a ∧ a < 1 → ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, a^n < ε) ∧ 
  (1 < a → ∀ N > 0, ∃ n : ℕ, a^n > N) :=
by
  sorry

end limit_of_power_seq_l219_219083


namespace tomatoes_cheaper_than_cucumbers_percentage_l219_219727

noncomputable def P_c := 5
noncomputable def two_T_three_P_c := 23
noncomputable def T := (two_T_three_P_c - 3 * P_c) / 2
noncomputable def percentage_by_which_tomatoes_cheaper_than_cucumbers := ((P_c - T) / P_c) * 100

theorem tomatoes_cheaper_than_cucumbers_percentage : 
  P_c = 5 → 
  (2 * T + 3 * P_c = 23) →
  T < P_c →
  percentage_by_which_tomatoes_cheaper_than_cucumbers = 20 :=
by
  intros
  sorry

end tomatoes_cheaper_than_cucumbers_percentage_l219_219727


namespace expression_value_l219_219726

/-- The value of the expression 1.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) is 1200. -/
theorem expression_value : 
  1.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 1200 :=
by
  sorry

end expression_value_l219_219726


namespace sum_series_eq_l219_219446

open BigOperators

theorem sum_series_eq : 
  ∑ n in Finset.range 256, (1 : ℝ) / ((2 * (n + 1 : ℕ) - 3) * (2 * (n + 1 : ℕ) + 1)) = -257 / 513 := 
by 
  sorry

end sum_series_eq_l219_219446


namespace set_A_is_2_3_l219_219827

noncomputable def A : Set ℤ := { x : ℤ | 3 / (x - 1) > 1 }

theorem set_A_is_2_3 : A = {2, 3} :=
by
  sorry

end set_A_is_2_3_l219_219827


namespace area_sum_of_three_circles_l219_219217

theorem area_sum_of_three_circles (R d : ℝ) (x y z : ℝ) 
    (hxyz : x^2 + y^2 + z^2 = d^2) :
    (π * ((R^2 - x^2) + (R^2 - y^2) + (R^2 - z^2))) = π * (3 * R^2 - d^2) :=
by
  sorry

end area_sum_of_three_circles_l219_219217


namespace unfair_coin_probability_l219_219835

theorem unfair_coin_probability (P : ℕ → ℝ) :
  let heads := 3/4
  let initial_condition := P 0 = 1
  let recurrence_relation := ∀n, P (n + 1) = 3 / 4 * (1 - P n) + 1 / 4 * P n
  recurrence_relation →
  initial_condition →
  P 40 = 1 / 2 * (1 + (1 / 2) ^ 40) :=
by
  sorry

end unfair_coin_probability_l219_219835


namespace count_valid_pairs_l219_219674

theorem count_valid_pairs : 
  ∃! n : ℕ, 
  n = 2 ∧ 
  (∀ (a b : ℕ), (0 < a ∧ 0 < b) → 
    (a * b + 97 = 18 * Nat.lcm a b + 14 * Nat.gcd a b) → 
    n = 2)
:= sorry

end count_valid_pairs_l219_219674


namespace income_difference_l219_219935

theorem income_difference
  (D W : ℝ)
  (hD : 0.08 * D = 800)
  (hW : 0.08 * W = 840) :
  (W + 840) - (D + 800) = 540 := 
  sorry

end income_difference_l219_219935


namespace sufficient_but_not_necessary_l219_219995

-- Define what it means for α to be of the form (π/6 + 2kπ) where k ∈ ℤ
def is_pi_six_plus_two_k_pi (α : ℝ) : Prop :=
  ∃ k : ℤ, α = Real.pi / 6 + 2 * k * Real.pi

-- Define the condition sin α = 1 / 2
def sin_is_half (α : ℝ) : Prop :=
  Real.sin α = 1 / 2

-- The theorem stating that the given condition is a sufficient but not necessary condition
theorem sufficient_but_not_necessary (α : ℝ) :
  is_pi_six_plus_two_k_pi α → sin_is_half α ∧ ¬ (sin_is_half α → is_pi_six_plus_two_k_pi α) :=
by
  sorry

end sufficient_but_not_necessary_l219_219995


namespace original_wire_length_l219_219559

theorem original_wire_length 
(L : ℝ) 
(h1 : L / 2 - 3 / 2 > 0) 
(h2 : L / 2 - 3 > 0) 
(h3 : L / 4 - 11.5 > 0)
(h4 : L / 4 - 6.5 = 7) : 
L = 54 := 
sorry

end original_wire_length_l219_219559


namespace b_days_to_complete_work_l219_219226

theorem b_days_to_complete_work (x : ℕ) 
  (A : ℝ := 1 / 30) 
  (B : ℝ := 1 / x) 
  (C : ℝ := 1 / 40)
  (work_eq : 8 * (A + B + C) + 4 * (A + B) = 1) 
  (x_ne_0 : x ≠ 0) : 
  x = 30 := 
by
  sorry

end b_days_to_complete_work_l219_219226


namespace find_wrong_number_read_l219_219634

theorem find_wrong_number_read (avg_initial avg_correct num_total wrong_num : ℕ) 
    (h1 : avg_initial = 15)
    (h2 : avg_correct = 16)
    (h3 : num_total = 10)
    (h4 : wrong_num = 36) 
    : wrong_num - (avg_correct * num_total - avg_initial * num_total) = 26 := 
by
  -- This is where the proof would go.
  sorry

end find_wrong_number_read_l219_219634


namespace probability_is_3888_over_7533_l219_219132

noncomputable def probability_odd_sum_given_even_product : ℚ := 
  let total_outcomes := 6^5
  let all_odd_outcomes := 3^5
  let at_least_one_even_outcomes := total_outcomes - all_odd_outcomes
  let favorable_outcomes := 5 * 3^4 + 10 * 3^4 + 3^5
  favorable_outcomes / at_least_one_even_outcomes

theorem probability_is_3888_over_7533 :
  probability_odd_sum_given_even_product = 3888 / 7533 := 
sorry

end probability_is_3888_over_7533_l219_219132


namespace mirror_area_l219_219934

theorem mirror_area (frame_length frame_width frame_border_length : ℕ) (mirror_area : ℕ)
  (h_frame_length : frame_length = 100)
  (h_frame_width : frame_width = 130)
  (h_frame_border_length : frame_border_length = 15)
  (h_mirror_area : mirror_area = (frame_length - 2 * frame_border_length) * (frame_width - 2 * frame_border_length)) :
  mirror_area = 7000 := by 
    sorry

end mirror_area_l219_219934


namespace smallest_n_divisible_by_24_and_864_l219_219016

theorem smallest_n_divisible_by_24_and_864 :
  ∃ n : ℕ, (0 < n) ∧ (24 ∣ n^2) ∧ (864 ∣ n^3) ∧ (∀ m : ℕ, (0 < m) → (24 ∣ m^2) → (864 ∣ m^3) → (n ≤ m)) :=
sorry

end smallest_n_divisible_by_24_and_864_l219_219016


namespace power_sum_eq_nine_l219_219877

theorem power_sum_eq_nine {m n p q : ℕ} (h : ∀ x > 0, (x + 1)^m / x^n - 1 = (x + 1)^p / x^q) :
  (m^2 + 2 * n + p)^(2 * q) = 9 :=
sorry

end power_sum_eq_nine_l219_219877


namespace tangent_condition_l219_219946

theorem tangent_condition (a b : ℝ) : 
    a = b → 
    (∀ x y : ℝ, (y = x + 2 → (x - a)^2 + (y - b)^2 = 2 → y = x + 2)) :=
by
  sorry

end tangent_condition_l219_219946


namespace total_distance_traveled_l219_219586

/-- The total distance traveled by Mr. and Mrs. Hugo over three days. -/
theorem total_distance_traveled :
  let first_day := 200
  let second_day := (3/4 : ℚ) * first_day
  let third_day := (1/2 : ℚ) * (first_day + second_day)
  first_day + second_day + third_day = 525 := by
  let first_day := 200
  let second_day := (3/4 : ℚ) * first_day
  let third_day := (1/2 : ℚ) * (first_day + second_day)
  have h1 : first_day + second_day + third_day = 525 := by
    sorry
  exact h1

end total_distance_traveled_l219_219586


namespace largest_c_such_that_neg5_in_range_l219_219807

theorem largest_c_such_that_neg5_in_range :
  ∃ (c : ℝ), (∀ x : ℝ, x^2 + 5 * x + c = -5) → c = 5 / 4 :=
sorry

end largest_c_such_that_neg5_in_range_l219_219807


namespace monotonic_increasing_f_C_l219_219480

noncomputable def f_A (x : ℝ) : ℝ := -Real.log x
noncomputable def f_B (x : ℝ) : ℝ := 1 / (2^x)
noncomputable def f_C (x : ℝ) : ℝ := -(1 / x)
noncomputable def f_D (x : ℝ) : ℝ := 3^(abs (x - 1))

theorem monotonic_increasing_f_C : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f_C x < f_C y :=
sorry

end monotonic_increasing_f_C_l219_219480


namespace olivia_bags_count_l219_219816

def cans_per_bag : ℕ := 5
def total_cans : ℕ := 20

theorem olivia_bags_count : total_cans / cans_per_bag = 4 := by
  sorry

end olivia_bags_count_l219_219816


namespace cube_traversal_count_l219_219547

-- Defining the cube traversal problem
def cube_traversal (num_faces : ℕ) (adj_faces : ℕ) (visits : ℕ) : ℕ :=
  if (num_faces = 6 ∧ adj_faces = 4) then
    4 * 2
  else
    0

-- Theorem statement
theorem cube_traversal_count : 
  cube_traversal 6 4 1 = 8 :=
by
  -- Skipping the proof with sorry for now
  sorry

end cube_traversal_count_l219_219547


namespace div_by_10_l219_219202

theorem div_by_10 (n : ℕ) (hn : 10 ∣ (3^n + 1)) : 10 ∣ (3^(n+4) + 1) :=
by
  sorry

end div_by_10_l219_219202


namespace sqrt_product_simplification_l219_219305

variable (p : ℝ)

theorem sqrt_product_simplification (hp : 0 ≤ p) :
  (Real.sqrt (42 * p) * Real.sqrt (7 * p) * Real.sqrt (14 * p)) = 42 * p * Real.sqrt (7 * p) :=
sorry

end sqrt_product_simplification_l219_219305


namespace trajectory_equation_l219_219051

theorem trajectory_equation (m x y : ℝ) (a b : ℝ × ℝ)
  (ha : a = (m * x, y + 1))
  (hb : b = (x, y - 1))
  (h_perp : a.1 * b.1 + a.2 * b.2 = 0) :
  m * x^2 + y^2 = 1 :=
sorry

end trajectory_equation_l219_219051


namespace local_odd_function_range_of_a_l219_219893

variable (f : ℝ → ℝ)
variable (a : ℝ)

def local_odd_function (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f (-x₀) = -f x₀

theorem local_odd_function_range_of_a (hf : ∀ x, f x = -a * (2^x) - 4) :
  local_odd_function f → (-4 ≤ a ∧ a < 0) :=
by
  sorry

end local_odd_function_range_of_a_l219_219893


namespace distinct_balls_boxes_l219_219589

def count_distinct_distributions (balls : ℕ) (boxes : ℕ) : ℕ :=
  if balls = 7 ∧ boxes = 3 then 8 else 0

theorem distinct_balls_boxes :
  count_distinct_distributions 7 3 = 8 :=
by sorry

end distinct_balls_boxes_l219_219589


namespace pastries_made_correct_l219_219645

-- Definitions based on conditions
def cakes_made := 14
def cakes_sold := 97
def pastries_sold := 8
def cakes_more_than_pastries := 89

-- Definition of the function to compute pastries made
def pastries_made (cakes_made cakes_sold pastries_sold cakes_more_than_pastries : ℕ) : ℕ :=
  cakes_sold - cakes_more_than_pastries

-- The statement to prove
theorem pastries_made_correct : pastries_made cakes_made cakes_sold pastries_sold cakes_more_than_pastries = 8 := by
  unfold pastries_made
  norm_num
  sorry

end pastries_made_correct_l219_219645


namespace dice_sum_is_4_l219_219748

-- Defining the sum of points obtained from two dice rolls
def sum_of_dice (a b : ℕ) : ℕ := a + b

-- The main theorem stating the condition we need to prove
theorem dice_sum_is_4 (a b : ℕ) (h : sum_of_dice a b = 4) :
  (a = 3 ∧ b = 1) ∨ (a = 1 ∧ b = 3) ∨ (a = 2 ∧ b = 2) :=
sorry

end dice_sum_is_4_l219_219748


namespace trams_to_add_l219_219734

theorem trams_to_add (initial_trams : ℕ) (initial_interval new_interval : ℤ)
  (reduce_by_fraction : ℤ) (total_distance : ℤ)
  (h1 : initial_trams = 12)
  (h2 : initial_interval = total_distance / initial_trams)
  (h3 : reduce_by_fraction = 5)
  (h4 : new_interval = initial_interval - initial_interval / reduce_by_fraction) :
  initial_trams + (total_distance / new_interval - initial_trams) = 15 :=
by
  sorry

end trams_to_add_l219_219734


namespace length_of_field_l219_219921

-- Define the conditions and given facts.
def double_length (w l : ℝ) : Prop := l = 2 * w
def pond_area (l w : ℝ) : Prop := 49 = 1/8 * (l * w)

-- Define the main statement that incorporates the given conditions and expected result.
theorem length_of_field (w l : ℝ) (h1 : double_length w l) (h2 : pond_area l w) : l = 28 := by
  sorry

end length_of_field_l219_219921


namespace function_solution_l219_219910

theorem function_solution (f : ℝ → ℝ) (H : ∀ x y : ℝ, 1 < x → 1 < y → f x - f y = (y - x) * f (x * y)) :
  ∃ k : ℝ, ∀ x : ℝ, 1 < x → f x = k / x :=
by
  sorry

end function_solution_l219_219910


namespace determine_xyz_l219_219903

theorem determine_xyz (x y z : ℝ) 
    (h1 : (x + y + z) * (x * y + x * z + y * z) = 12) 
    (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 16) : 
  x * y * z = -4 / 3 := 
sorry

end determine_xyz_l219_219903


namespace intersection_M_N_l219_219663

-- Definitions of the sets M and N
def M : Set ℝ := { -1, 0, 1 }
def N : Set ℝ := { x | x^2 ≤ x }

-- The theorem to be proven
theorem intersection_M_N : M ∩ N = { 0, 1 } :=
by
  sorry

end intersection_M_N_l219_219663


namespace fraction_inequality_l219_219191

theorem fraction_inequality (a b m : ℝ) (h1 : b > a) (h2 : m > 0) : 
  (b / a) > ((b + m) / (a + m)) :=
sorry

end fraction_inequality_l219_219191


namespace sequence_geometric_l219_219947

noncomputable def S (n : ℕ) (a : ℕ → ℝ) : ℝ :=
  3 * a n - 2

theorem sequence_geometric (a : ℕ → ℝ) (h : ∀ n, S n a = 3 * a n - 2) :
  ∀ n, a n = (3/2)^(n-1) :=
by
  intro n
  sorry

end sequence_geometric_l219_219947


namespace hotel_R_greater_than_G_l219_219823

variables (R G P : ℝ)

def hotel_charges_conditions :=
  P = 0.50 * R ∧ P = 0.80 * G

theorem hotel_R_greater_than_G :
  hotel_charges_conditions R G P → R = 1.60 * G :=
by
  sorry

end hotel_R_greater_than_G_l219_219823


namespace find_prime_pairs_l219_219847

def is_prime (n : ℕ) := n ≥ 2 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def has_prime_root (m n : ℕ) : Prop :=
  ∃ (p: ℕ), is_prime p ∧ (p * p - m * p - n = 0)

theorem find_prime_pairs :
  ∀ (m n : ℕ), (is_prime m ∧ is_prime n) → has_prime_root m n → (m, n) = (2, 3) :=
by sorry

end find_prime_pairs_l219_219847


namespace imaginary_unit_power_l219_219956

-- Definition of the imaginary unit i
def imaginary_unit_i : ℂ := Complex.I

theorem imaginary_unit_power :
  (imaginary_unit_i ^ 2015) = -imaginary_unit_i := by
  sorry

end imaginary_unit_power_l219_219956


namespace remainder_of_6_pow_50_mod_215_l219_219972

theorem remainder_of_6_pow_50_mod_215 :
  (6 ^ 50) % 215 = 36 := 
sorry

end remainder_of_6_pow_50_mod_215_l219_219972


namespace flowers_remaining_along_path_after_events_l219_219912

def total_flowers : ℕ := 30
def total_peonies : ℕ := 15
def total_tulips : ℕ := 15
def unwatered_flowers : ℕ := 10
def tulips_watered_by_sineglazka : ℕ := 10
def tulips_picked_by_neznaika : ℕ := 6
def remaining_flowers : ℕ := 19

theorem flowers_remaining_along_path_after_events :
  total_peonies + total_tulips = total_flowers →
  tulips_watered_by_sineglazka + unwatered_flowers = total_flowers →
  tulips_picked_by_neznaika ≤ total_tulips →
  remaining_flowers = 19 := sorry

end flowers_remaining_along_path_after_events_l219_219912


namespace regionA_regionC_area_ratio_l219_219381

-- Definitions for regions A and B
def regionA (l w : ℝ) : Prop := 2 * (l + w) = 16 ∧ l = 2 * w
def regionB (l w : ℝ) : Prop := 2 * (l + w) = 20 ∧ l = 2 * w
def area (l w : ℝ) : ℝ := l * w

theorem regionA_regionC_area_ratio {lA wA lB wB lC wC : ℝ} :
  regionA lA wA → regionB lB wB → (lC = lB ∧ wC = wB) → 
  (area lC wC ≠ 0) → 
  (area lA wA / area lC wC = 16 / 25) :=
by
  intros hA hB hC hC_area_ne_zero
  sorry

end regionA_regionC_area_ratio_l219_219381


namespace tuples_satisfy_equation_l219_219186

theorem tuples_satisfy_equation (a b c : ℤ) :
  (a - b)^3 * (a + b)^2 = c^2 + 2 * (a - b) + 1 ↔ (a = 1 ∧ b = 0 ∧ c = 0) ∨ (a = -1 ∧ b = 0 ∧ c = 0) :=
sorry

end tuples_satisfy_equation_l219_219186


namespace same_days_to_dig_scenario_l219_219949

def volume (depth length breadth : ℝ) : ℝ :=
  depth * length * breadth

def days_to_dig (depth length breadth days : ℝ) : Prop :=
  ∃ (labors : ℝ), 
    (volume depth length breadth) * days = (volume 100 25 30) * 12

theorem same_days_to_dig_scenario :
  days_to_dig 75 20 50 12 :=
sorry

end same_days_to_dig_scenario_l219_219949


namespace bob_hair_length_l219_219091

theorem bob_hair_length (h_0 : ℝ) (r : ℝ) (t : ℝ) (months_per_year : ℝ) (h : ℝ) :
  h_0 = 6 ∧ r = 0.5 ∧ t = 5 ∧ months_per_year = 12 → h = h_0 + r * months_per_year * t :=
sorry

end bob_hair_length_l219_219091


namespace rake_yard_alone_time_l219_219814

-- Definitions for the conditions
def brother_time := 45 -- Brother takes 45 minutes
def together_time := 18 -- Together it takes 18 minutes

-- Define and prove the time it takes you to rake the yard alone based on given conditions
theorem rake_yard_alone_time : 
  ∃ (x : ℕ), (1 / (x : ℚ) + 1 / (brother_time : ℚ) = 1 / (together_time : ℚ)) ∧ x = 30 :=
by
  sorry

end rake_yard_alone_time_l219_219814


namespace solution_exists_unique_n_l219_219082

theorem solution_exists_unique_n (n : ℕ) : 
  (∀ m : ℕ, (10 * m > 120) ∨ ∃ k1 k2 k3 : ℕ, 10 * k1 + n * k2 + (n + 1) * k3 = 120) = false → 
  n = 16 := by sorry

end solution_exists_unique_n_l219_219082


namespace not_all_same_probability_l219_219632

-- Definition of the total number of outcomes when rolling 5 8-sided dice
def total_outcomes : ℕ := 8^5

-- Definition of the number of outcomes where all five dice show the same number
def same_number_outcomes : ℕ := 8

-- Definition to find the probability that not all 5 dice show the same number
def probability_not_all_same : ℚ := 1 - (same_number_outcomes / total_outcomes)

-- Statement of the main theorem
theorem not_all_same_probability : probability_not_all_same = (4095 : ℚ) / 4096 :=
by
  rw [probability_not_all_same, same_number_outcomes, total_outcomes]
  -- Simplification steps would go here, but we use sorry to skip the proof
  sorry

end not_all_same_probability_l219_219632


namespace real_m_of_complex_product_l219_219842

-- Define the conditions that m is a real number and (m^2 + i)(1 - mi) is a real number
def is_real (z : ℂ) : Prop := z.im = 0
def cplx_eq (m : ℝ) : ℂ := (⟨m^2, 1⟩ : ℂ) * (⟨1, -m⟩ : ℂ)

theorem real_m_of_complex_product (m : ℝ) : is_real (cplx_eq m) ↔ m = 1 :=
by
  sorry

end real_m_of_complex_product_l219_219842


namespace quadratic_real_roots_l219_219683

theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2 * x + m = 0) ↔ m ≤ 1 :=
by
  sorry

end quadratic_real_roots_l219_219683


namespace min_value_of_F_l219_219414

variable (x1 x2 : ℝ)

def constraints :=
  2 - 2 * x1 - x2 ≥ 0 ∧
  2 - x1 + x2 ≥ 0 ∧
  5 - x1 - x2 ≥ 0 ∧
  0 ≤ x1 ∧
  0 ≤ x2

noncomputable def F := x2 - x1

theorem min_value_of_F : constraints x1 x2 → ∃ (minF : ℝ), minF = -2 :=
by
  sorry

end min_value_of_F_l219_219414


namespace average_speed_l219_219643

   theorem average_speed (x : ℝ) : 
     let s1 := 40
     let s2 := 20
     let d1 := x
     let d2 := 2 * x
     let total_distance := d1 + d2
     let time1 := d1 / s1
     let time2 := d2 / s2
     let total_time := time1 + time2
     total_distance / total_time = 24 :=
   by
     sorry
   
end average_speed_l219_219643


namespace find_f_l219_219216

noncomputable def f (f'₁ : ℝ) (x : ℝ) : ℝ := f'₁ * Real.exp x - x ^ 2

theorem find_f'₁ (f'₁ : ℝ) (h : f f'₁ = λ x => f'₁ * Real.exp x - x ^ 2) :
  f'₁ = 2 * Real.exp 1 / (Real.exp 1 - 1) := by
  sorry

end find_f_l219_219216


namespace sequence_gcd_equality_l219_219392

theorem sequence_gcd_equality (a : ℕ → ℕ) 
  (h : ∀ (i j : ℕ), i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j) : 
  ∀ i, a i = i := 
sorry

end sequence_gcd_equality_l219_219392


namespace intersection_A_B_when_m_eq_2_range_of_m_for_p_implies_q_l219_219028

noncomputable def A := {x : ℝ | -4 < x ∧ x < 2}
noncomputable def B (m : ℝ) := {x : ℝ | 1 - m ≤ x ∧ x ≤ 1 + m}

theorem intersection_A_B_when_m_eq_2 : (A ∩ B 2) = {x : ℝ | -1 ≤ x ∧ x < 2} :=
by
  sorry

theorem range_of_m_for_p_implies_q : {m : ℝ | m ≥ 5} = {m : ℝ | ∀ x, ((x^2 + 2 * x - 8 < 0) → ((x - 1 + m) * (x - 1 - m) ≤ 0)) ∧ ¬((x - 1 + m) * (x - 1 - m) ≤ 0 → (x^2 + 2 * x - 8 < 0))} :=
by
  sorry

end intersection_A_B_when_m_eq_2_range_of_m_for_p_implies_q_l219_219028


namespace player_B_questions_l219_219201

theorem player_B_questions :
  ∀ (a b : ℕ → ℕ), (∀ i j, i ≠ j → a i + b j = a j + b i) →
  ∃ k, k = 11 := sorry

end player_B_questions_l219_219201


namespace g_minus_6_eq_neg_20_l219_219435

noncomputable def g : ℤ → ℤ := sorry

axiom condition1 : g 1 - 1 > 0
axiom condition2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom condition3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

theorem g_minus_6_eq_neg_20 : g (-6) = -20 := sorry

end g_minus_6_eq_neg_20_l219_219435


namespace speed_of_current_eq_l219_219883

theorem speed_of_current_eq :
  ∃ (m c : ℝ), (m + c = 15) ∧ (m - c = 8.6) ∧ (c = 3.2) :=
by
  sorry

end speed_of_current_eq_l219_219883


namespace total_cost_bicycle_helmet_l219_219626

-- Let h represent the cost of the helmet
def helmet_cost := 40

-- Let b represent the cost of the bicycle
def bicycle_cost := 5 * helmet_cost

-- We need to prove that the total cost (bicycle + helmet) is equal to 240
theorem total_cost_bicycle_helmet : bicycle_cost + helmet_cost = 240 := 
by
  -- This will skip the proof, we only need the statement
  sorry

end total_cost_bicycle_helmet_l219_219626


namespace james_net_profit_l219_219523

def totalCandyBarsSold (boxes : Nat) (candyBarsPerBox : Nat) : Nat :=
  boxes * candyBarsPerBox

def revenue30CandyBars (pricePerCandyBar : Real) : Real :=
  30 * pricePerCandyBar

def revenue20CandyBars (pricePerCandyBar : Real) : Real :=
  20 * pricePerCandyBar

def totalRevenue (revenue1 : Real) (revenue2 : Real) : Real :=
  revenue1 + revenue2

def costNonDiscountedBoxes (candyBars : Nat) (pricePerCandyBar : Real) : Real :=
  candyBars * pricePerCandyBar

def costDiscountedBoxes (candyBars : Nat) (pricePerCandyBar : Real) : Real :=
  candyBars * pricePerCandyBar

def totalCost (cost1 : Real) (cost2 : Real) : Real :=
  cost1 + cost2

def salesTax (totalRevenue : Real) (taxRate : Real) : Real :=
  totalRevenue * taxRate

def totalExpenses (cost : Real) (salesTax : Real) (fixedExpense : Real) : Real :=
  cost + salesTax + fixedExpense

def netProfit (totalRevenue : Real) (totalExpenses : Real) : Real :=
  totalRevenue - totalExpenses

theorem james_net_profit :
  let boxes := 5
  let candyBarsPerBox := 10
  let totalCandyBars := totalCandyBarsSold boxes candyBarsPerBox

  let priceFirst30 := 1.50
  let priceNext20 := 1.30
  let priceSubsequent := 1.10

  let revenueFirst30 := revenue30CandyBars priceFirst30
  let revenueNext20 := revenue20CandyBars priceNext20
  let totalRevenue := totalRevenue revenueFirst30 revenueNext20

  let priceNonDiscounted := 1.00
  let candyBarsNonDiscounted := 20
  let costNonDiscounted := costNonDiscountedBoxes candyBarsNonDiscounted priceNonDiscounted

  let priceDiscounted := 0.80
  let candyBarsDiscounted := 30
  let costDiscounted := costDiscountedBoxes candyBarsDiscounted priceDiscounted

  let totalCost := totalCost costNonDiscounted costDiscounted

  let taxRate := 0.07
  let salesTax := salesTax totalRevenue taxRate

  let fixedExpense := 15.0
  let totalExpenses := totalExpenses totalCost salesTax fixedExpense

  netProfit totalRevenue totalExpenses = 7.03 :=
by
  sorry

end james_net_profit_l219_219523


namespace melissa_total_commission_l219_219356

def sale_price_coupe : ℝ := 30000
def sale_price_suv : ℝ := 2 * sale_price_coupe
def sale_price_luxury_sedan : ℝ := 80000

def commission_rate_coupe_and_suv : ℝ := 0.02
def commission_rate_luxury_sedan : ℝ := 0.03

def commission (rate : ℝ) (price : ℝ) : ℝ := rate * price

def total_commission : ℝ :=
  commission commission_rate_coupe_and_suv sale_price_coupe +
  commission commission_rate_coupe_and_suv sale_price_suv +
  commission commission_rate_luxury_sedan sale_price_luxury_sedan

theorem melissa_total_commission :
  total_commission = 4200 := by
  sorry

end melissa_total_commission_l219_219356


namespace certain_number_l219_219667

theorem certain_number (x : ℝ) (h : 7125 / x = 5700) : x = 1.25 := 
sorry

end certain_number_l219_219667


namespace money_distribution_l219_219251

theorem money_distribution :
  ∀ (A B C : ℕ), 
  A + B + C = 900 → 
  B + C = 750 → 
  C = 250 → 
  A + C = 400 := 
by
  intros A B C h1 h2 h3
  sorry

end money_distribution_l219_219251


namespace min_C2_minus_D2_is_36_l219_219861

noncomputable def find_min_C2_minus_D2 (x y z : ℝ) : ℝ :=
  (Real.sqrt (x + 3) + Real.sqrt (y + 6) + Real.sqrt (z + 11))^2 -
  (Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2))^2

theorem min_C2_minus_D2_is_36 : ∀ (x y z : ℝ), 0 ≤ x → 0 ≤ y → 0 ≤ z → 
  find_min_C2_minus_D2 x y z ≥ 36 :=
by
  intros x y z hx hy hz
  sorry

end min_C2_minus_D2_is_36_l219_219861


namespace sum_infinite_series_l219_219166

theorem sum_infinite_series : ∑' k : ℕ, (k^2 : ℝ) / (3^k) = 7 / 8 :=
sorry

end sum_infinite_series_l219_219166


namespace distinct_flavors_count_l219_219341

theorem distinct_flavors_count (red_candies : ℕ) (green_candies : ℕ)
  (h_red : red_candies = 0 ∨ red_candies = 1 ∨ red_candies = 2 ∨ red_candies = 3 ∨ red_candies = 4 ∨ red_candies = 5 ∨ red_candies = 6)
  (h_green : green_candies = 0 ∨ green_candies = 1 ∨ green_candies = 2 ∨ green_candies = 3 ∨ green_candies = 4 ∨ green_candies = 5) :
  ∃ unique_flavors : Finset (ℚ), unique_flavors.card = 25 :=
by
  sorry

end distinct_flavors_count_l219_219341


namespace fraction_is_three_eights_l219_219714

-- The given number
def number := 48

-- The fraction 'x' by which the number exceeds by 30
noncomputable def fraction (x : ℝ) : Prop :=
number = number * x + 30

-- Our goal is to prove that the fraction is 3/8
theorem fraction_is_three_eights : fraction (3 / 8) :=
by
  -- We reduced the goal proof to a simpler form for illustration, you can solve it rigorously
  sorry

end fraction_is_three_eights_l219_219714


namespace bigger_number_l219_219830

theorem bigger_number (yoongi : ℕ) (jungkook : ℕ) (h1 : yoongi = 4) (h2 : jungkook = 6 + 3) : jungkook > yoongi :=
by
  sorry

end bigger_number_l219_219830


namespace sin_over_cos_inequality_l219_219595

-- Define the main theorem and condition
theorem sin_over_cos_inequality (t : ℝ) (h₁ : 0 < t) (h₂ : t ≤ Real.pi / 2) : 
  (Real.sin t / t)^3 > Real.cos t := 
sorry

end sin_over_cos_inequality_l219_219595


namespace line_of_intersection_l219_219969

theorem line_of_intersection :
  ∀ (x y z : ℝ),
    (3 * x + 4 * y - 2 * z + 1 = 0) ∧ (2 * x - 4 * y + 3 * z + 4 = 0) →
    (∃ t : ℝ, x = -1 + 4 * t ∧ y = 1 / 2 - 13 * t ∧ z = -20 * t) :=
by
  intro x y z
  intro h
  cases h
  sorry

end line_of_intersection_l219_219969


namespace problem1_l219_219055

theorem problem1 :
  (2021 - Real.pi)^0 + (Real.sqrt 3 - 1) - 2 + (2 * Real.sqrt 3) = 3 * Real.sqrt 3 - 2 :=
by
  sorry

end problem1_l219_219055


namespace Dima_impossible_cut_l219_219978

theorem Dima_impossible_cut (n : ℕ) 
  (h1 : n % 5 = 0) 
  (h2 : n % 7 = 0) 
  (h3 : n ≤ 200) : ¬(n % 6 = 0) :=
sorry

end Dima_impossible_cut_l219_219978


namespace distance_between_parallel_lines_correct_l219_219616

open Real

noncomputable def distance_between_parallel_lines : ℝ :=
  let a := (3, 1)
  let b := (2, 4)
  let d := (4, -6)
  let v := (b.1 - a.1, b.2 - a.2)
  let d_perp := (6, 4) -- a vector perpendicular to d
  let v_dot_d_perp := v.1 * d_perp.1 + v.2 * d_perp.2
  let d_perp_dot_d_perp := d_perp.1 * d_perp.1 + d_perp.2 * d_perp.2
  let proj_v_onto_d_perp := (v_dot_d_perp / d_perp_dot_d_perp * d_perp.1, v_dot_d_perp / d_perp_dot_d_perp * d_perp.2)
  sqrt (proj_v_onto_d_perp.1 * proj_v_onto_d_perp.1 + proj_v_onto_d_perp.2 * proj_v_onto_d_perp.2)

theorem distance_between_parallel_lines_correct :
  distance_between_parallel_lines = (3 * sqrt 13) / 13 := by
  sorry

end distance_between_parallel_lines_correct_l219_219616


namespace gcd_1755_1242_l219_219319

theorem gcd_1755_1242 : Nat.gcd 1755 1242 = 27 := 
by
  sorry

end gcd_1755_1242_l219_219319


namespace sphere_circumscribed_around_cone_radius_l219_219580

-- Definitions of the given conditions
variable (r h : ℝ)

-- Theorem statement (without the proof)
theorem sphere_circumscribed_around_cone_radius :
  ∃ R : ℝ, R = (Real.sqrt (r^2 + h^2)) / 2 :=
sorry

end sphere_circumscribed_around_cone_radius_l219_219580


namespace units_digit_of_30_factorial_is_0_l219_219431

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_30_factorial_is_0 : units_digit (factorial 30) = 0 := by
  sorry

end units_digit_of_30_factorial_is_0_l219_219431


namespace find_4_digit_number_l219_219688

theorem find_4_digit_number (a b c d : ℕ) (h1 : 1000 * a + 100 * b + 10 * c + d = 1000 * d + 100 * c + 10 * b + a - 7182) :
  1000 * a + 100 * b + 10 * c + d = 1909 :=
by {
  sorry
}

end find_4_digit_number_l219_219688


namespace inverse_proportion_points_l219_219760

theorem inverse_proportion_points (x1 x2 x3 : ℝ) :
  (10 / x1 = -5) →
  (10 / x2 = 2) →
  (10 / x3 = 5) →
  x1 < x3 ∧ x3 < x2 :=
by sorry

end inverse_proportion_points_l219_219760


namespace narrow_black_stripes_are_eight_l219_219391

variable (w n b : ℕ)

-- Given conditions as definitions in Lean
def white_stripes_eq : Prop := b = w + 7
def total_black_eq_total_white_plus_one : Prop := w + n = b + 1

theorem narrow_black_stripes_are_eight (h₁ : white_stripes_eq w b) (h₂ : total_black_eq_total_white_plus_one w n b) : n = 8 := by
  -- Use the assumptions to derive n = 8
  sorry

end narrow_black_stripes_are_eight_l219_219391


namespace distance_between_stations_is_correct_l219_219464

noncomputable def distance_between_stations : ℕ := 200

theorem distance_between_stations_is_correct 
  (start_hour_p : ℕ := 7) 
  (speed_p : ℕ := 20) 
  (start_hour_q : ℕ := 8) 
  (speed_q : ℕ := 25) 
  (meeting_hour : ℕ := 12)
  (time_travel_p := meeting_hour - start_hour_p) -- Time traveled by train from P
  (time_travel_q := meeting_hour - start_hour_q) -- Time traveled by train from Q 
  (distance_travel_p := speed_p * time_travel_p) 
  (distance_travel_q := speed_q * time_travel_q) : 
  distance_travel_p + distance_travel_q = distance_between_stations :=
by 
  sorry

end distance_between_stations_is_correct_l219_219464


namespace shaded_region_area_l219_219860

/-- A rectangle measuring 12cm by 8cm has four semicircles drawn with their diameters as the sides
of the rectangle. Prove that the area of the shaded region inside the rectangle but outside
the semicircles is equal to 96 - 52π (cm²). --/
theorem shaded_region_area (A : ℝ) (π : ℝ) (hA : A = 96 - 52 * π) : 
  ∀ (length width r1 r2 : ℝ) (hl : length = 12) (hw : width = 8) 
  (hr1 : r1 = length / 2) (hr2 : r2 = width / 2),
  (length * width) - (2 * (1/2 * π * r1^2 + 1/2 * π * r2^2)) = A := 
by 
  sorry

end shaded_region_area_l219_219860


namespace range_of_k_l219_219703

theorem range_of_k (k : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, 0 ≤ k * x^2 + k * x + 3) :
  0 ≤ k ∧ k ≤ 12 :=
sorry

end range_of_k_l219_219703


namespace outfit_count_l219_219173

def num_shirts := 8
def num_hats := 8
def num_pants := 4

def shirt_colors := 6
def hat_colors := 6
def pants_colors := 4

def total_possible_outfits := num_shirts * num_hats * num_pants

def same_color_restricted_outfits := 4 * 8 * 7

def num_valid_outfits := total_possible_outfits - same_color_restricted_outfits

theorem outfit_count (h1 : num_shirts = 8) (h2 : num_hats = 8) (h3 : num_pants = 4)
                     (h4 : shirt_colors = 6) (h5 : hat_colors = 6) (h6 : pants_colors = 4)
                     (h7 : total_possible_outfits = 256) (h8 : same_color_restricted_outfits = 224) :
  num_valid_outfits = 32 :=
by
  sorry

end outfit_count_l219_219173


namespace right_triangle_primes_l219_219492

-- Define what it means for a number to be a prime
def is_prime (n : ℕ) : Prop := ∃ m : ℕ, m > 1 ∧ m < n ∧ n % m = 0

-- State the problem
theorem right_triangle_primes
  (a b : ℕ)
  (ha : is_prime a)
  (hb : is_prime b)
  (a_gt_b : a > b)
  (a_plus_b : a + b = 90)
  (a_minus_b_prime : is_prime (a - b)) :
  b = 17 :=
sorry

end right_triangle_primes_l219_219492


namespace inequality_proof_l219_219416

open Real

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (sqrt (a^2 + 8 * b * c) / a + sqrt (b^2 + 8 * a * c) / b + sqrt (c^2 + 8 * a * b) / c) ≥ 9 :=
by 
  sorry

end inequality_proof_l219_219416


namespace sum_and_product_l219_219506

theorem sum_and_product (c d : ℝ) (h1 : 2 * c = -8) (h2 : c^2 - d = 4) : c + d = 8 := by
  sorry

end sum_and_product_l219_219506


namespace simplify_proof_l219_219289

noncomputable def simplify_expression (x : ℝ) (hx : x ≠ 0) (hx1 : x ≠ 1) (hx_1 : x ≠ -1) : ℝ :=
  (1 - 1/x) / ((1 - x^2) / x)

theorem simplify_proof (x : ℝ) (hx : x ≠ 0) (hx1 : x ≠ 1) (hx_1 : x ≠ -1) : 
  simplify_expression x hx hx1 hx_1 = -1 / (1 + x) := by 
  sorry

end simplify_proof_l219_219289


namespace problem_statement_l219_219064

theorem problem_statement (A B : ℝ) (hA : A = 10 * π / 180) (hB : B = 35 * π / 180) :
  (1 + Real.tan A) * (1 + Real.sin B) = 
  1 + Real.tan A + (Real.sqrt 2 / 2) * (Real.cos A - Real.sin A) + Real.tan A * (Real.sqrt 2 / 2) * (Real.cos A - Real.sin A) :=
by
  sorry

end problem_statement_l219_219064


namespace find_number_l219_219481

theorem find_number (x : ℝ) (h : (((x + 45) / 2) / 2) + 45 = 85) : x = 115 :=
by
  sorry

end find_number_l219_219481


namespace on_real_axis_in_first_quadrant_on_line_l219_219163

theorem on_real_axis (m : ℝ) : 
  (m = -3 ∨ m = 5) ↔ (m^2 - 2 * m - 15 = 0) := 
sorry

theorem in_first_quadrant (m : ℝ) : 
  (m < -3 ∨ m > 5) ↔ ((m^2 + 5 * m + 6 > 0) ∧ (m^2 - 2 * m - 15 > 0)) := 
sorry

theorem on_line (m : ℝ) : 
  (m = 1 ∨ m = -5 / 2) ↔ ((m^2 + 5 * m + 6) + (m^2 - 2 * m - 15) + 5 = 0) := 
sorry

end on_real_axis_in_first_quadrant_on_line_l219_219163


namespace solve_special_sequence_l219_219449

noncomputable def special_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1010 ∧ a 2 = 1015 ∧ ∀ n ≥ 1, a n + a (n + 1) + a (n + 2) = 2 * n + 1

theorem solve_special_sequence :
  ∃ a : ℕ → ℕ, special_sequence a ∧ a 1000 = 1676 :=
by
  sorry

end solve_special_sequence_l219_219449


namespace expression_divisible_by_25_l219_219148

theorem expression_divisible_by_25 (n : ℕ) : 
    (2^(n+2) * 3^n + 5 * n - 4) % 25 = 0 :=
by {
  sorry
}

end expression_divisible_by_25_l219_219148


namespace probability_not_finishing_on_time_l219_219940

-- Definitions based on the conditions
def P_finishing_on_time : ℚ := 5 / 8

-- Theorem to prove the required probability
theorem probability_not_finishing_on_time :
  (1 - P_finishing_on_time) = 3 / 8 := by
  sorry

end probability_not_finishing_on_time_l219_219940


namespace sum_of_faces_edges_vertices_l219_219245

def cube_faces : ℕ := 6
def cube_edges : ℕ := 12
def cube_vertices : ℕ := 8

theorem sum_of_faces_edges_vertices :
  cube_faces + cube_edges + cube_vertices = 26 := by
  sorry

end sum_of_faces_edges_vertices_l219_219245


namespace find_fraction_result_l219_219283

open Complex

theorem find_fraction_result (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
    (h1 : x + y + z = 30)
    (h2 : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2 * x * y * z) :
    (x^3 + y^3 + z^3) / (x * y * z) = 33 := 
    sorry

end find_fraction_result_l219_219283


namespace rectangle_division_impossible_l219_219733

theorem rectangle_division_impossible :
  ¬ ∃ n m : ℕ, n * 5 = 55 ∧ m * 11 = 39 :=
by
  sorry

end rectangle_division_impossible_l219_219733


namespace cost_of_article_l219_219609

theorem cost_of_article (C : ℝ) (H1 : 350 - C = G + 0.05 * G) (H2 : 345 - C = G) : C = 245 :=
by
  sorry

end cost_of_article_l219_219609


namespace triangle_inequality_l219_219060

variable (a b c : ℝ)

theorem triangle_inequality (h₁ : a + b + c = 1) (h₂ : a > 0) (h₃ : b > 0) (h₄ : c > 0) (h₅ : a + b > c) (h₆ : b + c > a) (h₇ : c + a > b) : 
  a^2 + b^2 + c^2 + 4 * a * b * c < 1 / 2 := 
sorry

end triangle_inequality_l219_219060


namespace slices_of_pizza_left_l219_219133

theorem slices_of_pizza_left (initial_slices: ℕ) 
  (breakfast_slices: ℕ) (lunch_slices: ℕ) (snack_slices: ℕ) (dinner_slices: ℕ) :
  initial_slices = 15 →
  breakfast_slices = 4 →
  lunch_slices = 2 →
  snack_slices = 2 →
  dinner_slices = 5 →
  (initial_slices - breakfast_slices - lunch_slices - snack_slices - dinner_slices) = 2 :=
by
  intros
  repeat { sorry }

end slices_of_pizza_left_l219_219133


namespace luis_bought_6_pairs_of_blue_socks_l219_219583

open Nat

-- Conditions
def total_pairs_red := 4
def total_cost_red := 3
def total_cost := 42
def blue_socks_cost := 5

-- Deduce the spent amount on red socks, and from there calculate the number of blue socks bought.
theorem luis_bought_6_pairs_of_blue_socks :
  (yes : ℕ) -> yes * blue_socks_cost = total_cost - total_pairs_red * total_cost_red → yes = 6 :=
sorry

end luis_bought_6_pairs_of_blue_socks_l219_219583


namespace distinct_real_roots_iff_l219_219154

theorem distinct_real_roots_iff (a : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (∀ x, x^2 + 3 * x - a = 0 → (x = x₁ ∨ x = x₂))) ↔ a > - (9 : ℝ) / 4 :=
sorry

end distinct_real_roots_iff_l219_219154


namespace sequence_tenth_term_l219_219739

theorem sequence_tenth_term :
  ∃ (a : ℕ → ℚ), a 1 = 1 ∧ (∀ n : ℕ, n > 0 → a (n + 1) = a n / (1 + a n)) ∧ a 10 = 1 / 10 :=
sorry

end sequence_tenth_term_l219_219739


namespace inequality_2n_1_lt_n_plus_1_sq_l219_219387

theorem inequality_2n_1_lt_n_plus_1_sq (n : ℕ) (h : 0 < n) : 2 * n - 1 < (n + 1) ^ 2 := 
by 
  sorry

end inequality_2n_1_lt_n_plus_1_sq_l219_219387


namespace root_not_less_than_a_l219_219484

noncomputable def f (x : ℝ) : ℝ := (1/2)^x - x^3

theorem root_not_less_than_a (a b c x0 : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < c)
  (h4 : f a * f b * f c < 0) (hx : f x0 = 0) : ¬ (x0 < a) :=
sorry

end root_not_less_than_a_l219_219484


namespace rhombus_shorter_diagonal_l219_219349

theorem rhombus_shorter_diagonal (d1 d2 : ℝ) (area : ℝ) (h1 : d2 = 20) (h2 : area = 120) (h3 : area = (d1 * d2) / 2) : d1 = 12 :=
by 
  sorry

end rhombus_shorter_diagonal_l219_219349


namespace max_min_values_of_f_l219_219025

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem max_min_values_of_f :
  (∀ x ∈ Set.Icc (-3 : ℝ) (0 : ℝ), f x ≤ 2) ∧ 
  (∃ x ∈ Set.Icc (-3 : ℝ) (0 : ℝ), f x = 2) ∧
  (∀ x ∈ Set.Icc (-3 : ℝ) (0 : ℝ), f x ≥ -18) ∧ 
  (∃ x ∈ Set.Icc (-3 : ℝ) (0 : ℝ), f x = -18)
:= by
  sorry  -- To be replaced with the actual proof

end max_min_values_of_f_l219_219025


namespace gcd_f100_f101_l219_219540

-- Define the function f
def f (x : ℤ) : ℤ := x^2 - x + 2010

-- A statement asserting the greatest common divisor of f(100) and f(101) is 10
theorem gcd_f100_f101 : Int.gcd (f 100) (f 101) = 10 := by
  sorry

end gcd_f100_f101_l219_219540


namespace num_positive_integers_l219_219520

theorem num_positive_integers (m : ℕ) : 
  (∃ n, m^2 - 2 = n ∧ n ∣ 2002) ↔ (m = 2 ∨ m = 3 ∨ m = 4) :=
by
  sorry

end num_positive_integers_l219_219520


namespace max_average_hours_l219_219570

theorem max_average_hours :
  let hours_Wednesday := 2
  let hours_Thursday := 2
  let hours_Friday := hours_Wednesday + 3
  let total_hours := hours_Wednesday + hours_Thursday + hours_Friday
  let average_hours := total_hours / 3
  average_hours = 3 :=
by
  sorry

end max_average_hours_l219_219570


namespace horse_problem_l219_219766

-- Definitions based on conditions:
def total_horses : ℕ := 100
def tiles_pulled_by_big_horse (x : ℕ) : ℕ := 3 * x
def tiles_pulled_by_small_horses (x : ℕ) : ℕ := (100 - x) / 3

-- The statement to prove:
theorem horse_problem (x : ℕ) : 
    tiles_pulled_by_big_horse x + tiles_pulled_by_small_horses x = 100 :=
sorry

end horse_problem_l219_219766


namespace remainder_of_polynomial_division_l219_219228

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := 7 * x^4 - 16 * x^3 + 3 * x^2 - 5 * x - 20

-- Define the divisor D(x)
def D (x : ℝ) : ℝ := 2 * x - 4

-- The remainder theorem sets x to 2 and evaluates P(x)
theorem remainder_of_polynomial_division : P 2 = -34 :=
by
  -- We will substitute x=2 directly into P(x)
  sorry

end remainder_of_polynomial_division_l219_219228


namespace eval_g_at_8_l219_219532

def g (x : ℚ) : ℚ := (3 * x + 2) / (x - 2)

theorem eval_g_at_8 : g 8 = 13 / 3 := by
  sorry

end eval_g_at_8_l219_219532


namespace rowing_distance_l219_219146

theorem rowing_distance :
  let row_speed := 4 -- kmph
  let river_speed := 2 -- kmph
  let total_time := 1.5 -- hours
  ∃ d, 
    let downstream_speed := row_speed + river_speed
    let upstream_speed := row_speed - river_speed
    let downstream_time := d / downstream_speed
    let upstream_time := d / upstream_speed
    downstream_time + upstream_time = total_time ∧ d = 2.25 :=
by
  sorry

end rowing_distance_l219_219146


namespace spider_paths_l219_219466

-- Define the grid points and the binomial coefficient calculation.
def grid_paths (n m : ℕ) : ℕ := Nat.choose (n + m) n

-- The problem statement
theorem spider_paths : grid_paths 4 3 = 35 := by
  sorry

end spider_paths_l219_219466


namespace smallest_even_n_sum_eq_l219_219658
  
theorem smallest_even_n_sum_eq (n : ℕ) (h_pos : n > 0) (h_even : n % 2 = 0) :
  n = 12 ↔ 
  let s₁ := n / 2 * (2 * 5 + (n - 1) * 6)
  let s₂ := n / 2 * (2 * 13 + (n - 1) * 3)
  s₁ = s₂ :=
by
  sorry

end smallest_even_n_sum_eq_l219_219658


namespace simplify_expression_calculate_expression_l219_219080

-- Problem 1
theorem simplify_expression (x : ℝ) : 
  (x + 1) * (x + 1) - x * (x + 1) = x + 1 := by
  sorry

-- Problem 2
theorem calculate_expression : 
  (-1 : ℝ) ^ 2023 + 2 ^ (-2 : ℝ) + 4 * (Real.cos (Real.pi / 6))^2 = 9 / 4 := by
  sorry

end simplify_expression_calculate_expression_l219_219080


namespace initial_flowers_per_bunch_l219_219505

theorem initial_flowers_per_bunch (x : ℕ) (h₁: 8 * x = 72) : x = 9 :=
  by
  sorry

end initial_flowers_per_bunch_l219_219505


namespace sum_of_coefficients_l219_219335

noncomputable def polynomial (x : ℝ) : ℝ := x^3 + 3*x^2 - 4*x - 12
noncomputable def simplified_polynomial (x : ℝ) (A B C : ℝ) : ℝ := A*x^2 + B*x + C

theorem sum_of_coefficients : 
  ∃ (A B C D : ℝ), 
    (∀ x ≠ D, simplified_polynomial x A B C = (polynomial x) / (x + 3)) ∧ 
    (A + B + C + D = -6) :=
by
  sorry

end sum_of_coefficients_l219_219335


namespace amy_total_soups_l219_219549

def chicken_soup := 6
def tomato_soup := 3
def vegetable_soup := 4
def clam_chowder := 2
def french_onion_soup := 1
def minestrone_soup := 5

theorem amy_total_soups : (chicken_soup + tomato_soup + vegetable_soup + clam_chowder + french_onion_soup + minestrone_soup) = 21 := by
  sorry

end amy_total_soups_l219_219549


namespace harris_carrot_expense_l219_219410

theorem harris_carrot_expense
  (carrots_per_day : ℕ)
  (days_per_year : ℕ)
  (carrots_per_bag : ℕ)
  (cost_per_bag : ℝ)
  (total_expense : ℝ) :
  carrots_per_day = 1 →
  days_per_year = 365 →
  carrots_per_bag = 5 →
  cost_per_bag = 2 →
  total_expense = 146 :=
by
  intros h1 h2 h3 h4
  sorry

end harris_carrot_expense_l219_219410


namespace m_minus_n_is_perfect_square_l219_219463

theorem m_minus_n_is_perfect_square (m n : ℕ) (h : 0 < m) (h1 : 0 < n) (h2 : 2001 * m^2 + m = 2002 * n^2 + n) : ∃ k : ℕ, m = n + k^2 :=
by
    sorry

end m_minus_n_is_perfect_square_l219_219463


namespace op_4_3_equals_23_l219_219061

def op (a b : ℕ) : ℕ := a ^ 2 + a * b + a - b ^ 2

theorem op_4_3_equals_23 : op 4 3 = 23 := by
  -- Proof steps would go here
  sorry

end op_4_3_equals_23_l219_219061


namespace fraction_computation_l219_219140

theorem fraction_computation :
  ((11^4 + 324) * (23^4 + 324) * (35^4 + 324) * (47^4 + 324) * (59^4 + 324)) / 
  ((5^4 + 324) * (17^4 + 324) * (29^4 + 324) * (41^4 + 324) * (53^4 + 324)) = 295.615 := 
sorry

end fraction_computation_l219_219140


namespace a_plus_b_l219_219966

theorem a_plus_b (a b : ℝ) (h : a * (a - 4) = 12) (h2 : b * (b - 4) = 12) (h3 : a ≠ b) : a + b = 4 :=
sorry

end a_plus_b_l219_219966


namespace find_value_of_4_minus_2a_l219_219093

theorem find_value_of_4_minus_2a (a b : ℚ) (h1 : 4 + 2 * a = 5 - b) (h2 : 5 + b = 9 + 3 * a) : 4 - 2 * a = 26 / 5 := 
by
  sorry

end find_value_of_4_minus_2a_l219_219093


namespace term_15_of_sequence_l219_219930

theorem term_15_of_sequence : 
  ∃ (a : ℕ → ℝ), a 1 = 3 ∧ a 2 = 7 ∧ (∀ n, a (n + 1) = 21 / a n) ∧ a 15 = 3 :=
sorry

end term_15_of_sequence_l219_219930


namespace find_a_min_value_of_f_l219_219615

theorem find_a (a : ℕ) (h1 : 3 / 2 < 2 + a) (h2 : 1 / 2 ≥ 2 - a) : a = 1 := by
  sorry

theorem min_value_of_f (a x : ℝ) (hx : -1 ≤ x ∧ x ≤ 2) : 
    (a = 1) → ∃ m : ℝ, m = 3 ∧ ∀ x : ℝ, |x + a| + |x - 2| ≥ m := by
  sorry

end find_a_min_value_of_f_l219_219615


namespace lines_parallel_l219_219231

noncomputable def line1 (a : ℝ) : ℝ × ℝ × ℝ := (a, 2, 6)
noncomputable def line2 (a : ℝ) : ℝ × ℝ × ℝ := (1, a-1, a^2-1)

def are_parallel (line1 line2 : ℝ × ℝ × ℝ) : Prop :=
  let ⟨a1, b1, _⟩ := line1
  let ⟨a2, b2, _⟩ := line2
  a1 * b2 = a2 * b1

theorem lines_parallel (a : ℝ) :
  are_parallel (line1 a) (line2 a) → a = -1 :=
sorry

end lines_parallel_l219_219231


namespace bacteria_colony_growth_l219_219924

theorem bacteria_colony_growth (n : ℕ) : 
  (∀ m: ℕ, 4 * 3^m ≤ 500 → m < n) → n = 5 :=
by
  sorry

end bacteria_colony_growth_l219_219924


namespace tan_alpha_sub_beta_l219_219476

theorem tan_alpha_sub_beta (α β : ℝ) (h₁ : Real.tan α = 9) (h₂ : Real.tan β = 6) : Real.tan (α - β) = 3 / 55 := 
sorry

end tan_alpha_sub_beta_l219_219476


namespace time_for_worker_C_l219_219408

theorem time_for_worker_C (time_A time_B time_total : ℝ) (time_A_pos : 0 < time_A) (time_B_pos : 0 < time_B) (time_total_pos : 0 < time_total) 
  (hA : time_A = 12) (hB : time_B = 15) (hTotal : time_total = 6) : 
  (1 / (1 / time_total - 1 / time_A - 1 / time_B) = 60) :=
by 
  sorry

end time_for_worker_C_l219_219408


namespace catherine_pencils_per_friend_l219_219459

theorem catherine_pencils_per_friend :
  ∀ (pencils pens given_pens : ℕ), 
  pencils = pens ∧ pens = 60 ∧ given_pens = 8 ∧ 
  (∃ remaining_items : ℕ, remaining_items = 22 ∧ 
    ∀ friends : ℕ, friends = 7 → 
    remaining_items = (pens - (given_pens * friends)) + (pencils - (given_pens * friends * (pencils / pens)))) →
  ((pencils - (given_pens * friends * (pencils / pens))) / friends) = 6 :=
by 
  sorry

end catherine_pencils_per_friend_l219_219459


namespace nancy_seeds_in_big_garden_l219_219212

theorem nancy_seeds_in_big_garden :
  let total_seeds := 52
  let small_gardens := 6
  let seeds_per_small_garden := 4
  let total_seeds_small_gardens := small_gardens * seeds_per_small_garden
  let seeds_in_big_garden := total_seeds - total_seeds_small_gardens
  seeds_in_big_garden = 28 := by
  let total_seeds := 52
  let small_gardens := 6
  let seeds_per_small_garden := 4
  let total_seeds_small_gardens := small_gardens * seeds_per_small_garden
  let seeds_in_big_garden := total_seeds - total_seeds_small_gardens
  sorry

end nancy_seeds_in_big_garden_l219_219212


namespace odd_c_perfect_square_no_even_c_infinitely_many_solutions_l219_219873

open Nat

/-- Problem (1): prove that if c is an odd number, then c is a perfect square given 
    c(a c + 1)^2 = (5c + 2b)(2c + b) -/
theorem odd_c_perfect_square (a b c : ℕ) (h_eq : c * (a * c + 1)^2 = (5 * c + 2 * b) * (2 * c + b)) (h_odd : Odd c) : ∃ k : ℕ, c = k^2 :=
  sorry

/-- Problem (2): prove that there does not exist an even number c that satisfies 
    c(a c + 1)^2 = (5c + 2b)(2c + b) for some a and b -/
theorem no_even_c (a b : ℕ) : ∀ c : ℕ, Even c → ¬ (c * (a * c + 1)^2 = (5 * c + 2 * b) * (2 * c + b)) :=
  sorry

/-- Problem (3): prove that there are infinitely many solutions of positive integers 
    (a, b, c) that satisfy c(a c + 1)^2 = (5c + 2b)(2c + b) -/
theorem infinitely_many_solutions (n : ℕ) : ∃ a b c : ℕ, 0 < a ∧ 0 < b ∧ 0 < c ∧
  c * (a * c + 1)^2 = (5 * c + 2 * b) * (2 * c + b) :=
  sorry

end odd_c_perfect_square_no_even_c_infinitely_many_solutions_l219_219873


namespace ounces_per_container_l219_219998

def weight_pounds : ℝ := 3.75
def num_containers : ℕ := 4
def pound_to_ounces : ℕ := 16

theorem ounces_per_container :
  (weight_pounds * pound_to_ounces) / num_containers = 15 :=
by
  sorry

end ounces_per_container_l219_219998


namespace storks_equal_other_birds_l219_219973

-- Definitions of initial numbers of birds
def initial_sparrows := 2
def initial_crows := 1
def initial_storks := 3
def initial_egrets := 0

-- Birds arriving initially
def sparrows_arrived := 1
def crows_arrived := 3
def storks_arrived := 6
def egrets_arrived := 4

-- Birds leaving after 15 minutes
def sparrows_left := 2
def crows_left := 0
def storks_left := 0
def egrets_left := 1

-- Additional birds arriving after 30 minutes
def additional_sparrows := 0
def additional_crows := 4
def additional_storks := 3
def additional_egrets := 0

-- Final counts
def final_sparrows := initial_sparrows + sparrows_arrived - sparrows_left + additional_sparrows
def final_crows := initial_crows + crows_arrived - crows_left + additional_crows
def final_storks := initial_storks + storks_arrived - storks_left + additional_storks
def final_egrets := initial_egrets + egrets_arrived - egrets_left + additional_egrets

def total_other_birds := final_sparrows + final_crows + final_egrets

-- Theorem statement
theorem storks_equal_other_birds : final_storks - total_other_birds = 0 := by
  sorry

end storks_equal_other_birds_l219_219973


namespace Chloe_total_points_l219_219677

-- Define the points scored in each round
def first_round_points : ℕ := 40
def second_round_points : ℕ := 50
def last_round_points : ℤ := -4

-- Define total points calculation
def total_points := first_round_points + second_round_points + last_round_points

-- The final statement to prove
theorem Chloe_total_points : total_points = 86 := by
  -- This proof is to be completed
  sorry

end Chloe_total_points_l219_219677


namespace total_pixels_correct_l219_219284

-- Define the monitor's dimensions and pixel density as given conditions
def width_inches : ℕ := 21
def height_inches : ℕ := 12
def pixels_per_inch : ℕ := 100

-- Define the width and height in pixels based on the given conditions
def width_pixels : ℕ := width_inches * pixels_per_inch
def height_pixels : ℕ := height_inches * pixels_per_inch

-- State the objective: proving the total number of pixels on the monitor
theorem total_pixels_correct : width_pixels * height_pixels = 2520000 := by
  sorry

end total_pixels_correct_l219_219284


namespace max_gold_coins_l219_219330

theorem max_gold_coins (n : ℤ) (h₁ : ∃ k : ℤ, n = 13 * k + 3) (h₂ : n < 150) : n ≤ 146 :=
by {
  sorry -- Proof not required as per instructions
}

end max_gold_coins_l219_219330


namespace function_identity_l219_219494

variable (f : ℕ+ → ℕ+)

theorem function_identity (h : ∀ n : ℕ+, f (n + 1) > f (f n)) : ∀ n : ℕ+, f n = n := sorry

end function_identity_l219_219494


namespace find_x_l219_219207

theorem find_x (x : ℝ) (h : 0.45 * x = (1 / 3) * x + 110) : x = 942.857 :=
by
  sorry

end find_x_l219_219207


namespace number_whose_multiples_are_considered_for_calculating_the_average_l219_219209

theorem number_whose_multiples_are_considered_for_calculating_the_average
  (x : ℕ)
  (n : ℕ)
  (a : ℕ)
  (b : ℕ)
  (h1 : n = 10)
  (h2 : a = (x + 2*x + 3*x + 4*x + 5*x + 6*x + 7*x) / 7)
  (h3 : b = 2*n)
  (h4 : a^2 - b^2 = 0) :
  x = 5 := 
sorry

end number_whose_multiples_are_considered_for_calculating_the_average_l219_219209


namespace polynomial_multiplication_l219_219134

theorem polynomial_multiplication (x a : ℝ) : (x - a) * (x^2 + a * x + a^2) = x^3 - a^3 :=
by
  sorry

end polynomial_multiplication_l219_219134


namespace smallest_solution_eq_l219_219041

theorem smallest_solution_eq :
  (∀ x : ℝ, x ≠ 3 →
  (3 * x / (x - 3) + (3 * x^2 - 27) / x = 15) → 
  x = 1 - Real.sqrt 10 ∨ (∃ y : ℝ, y ≤ 1 - Real.sqrt 10 ∧ y ≠ 3 ∧ 3 * y / (y - 3) + (3 * y^2 - 27) / y = 15)) :=
sorry

end smallest_solution_eq_l219_219041


namespace does_not_pass_through_second_quadrant_l219_219120

def line_eq (x y : ℝ) : Prop := x - y - 1 = 0

theorem does_not_pass_through_second_quadrant :
  ¬ ∃ (x y : ℝ), line_eq x y ∧ x < 0 ∧ y > 0 :=
sorry

end does_not_pass_through_second_quadrant_l219_219120


namespace odd_base_divisibility_by_2_base_divisibility_by_m_l219_219775

-- Part (a)
theorem odd_base_divisibility_by_2 (q : ℕ) :
  (∀ a : ℕ, (a * q) % 2 = 0 ↔ a % 2 = 0) → q % 2 = 1 := 
sorry

-- Part (b)
theorem base_divisibility_by_m (q m : ℕ) (h1 : m > 1) :
  (∀ a : ℕ, (a * q) % m = 0 ↔ a % m = 0) → ∃ k : ℕ, q = 1 + m * k ∧ k ≥ 1 :=
sorry

end odd_base_divisibility_by_2_base_divisibility_by_m_l219_219775


namespace exists_lcm_lt_l219_219001

theorem exists_lcm_lt (p q : ℕ) (hpq_coprime : Nat.gcd p q = 1) (hp_gt_one : p > 1) (hq_gt_one : q > 1) (hpq_diff_gt_one : (p < q ∧ q - p > 1) ∨ (p > q ∧ p - q > 1)) :
  ∃ n : ℕ, Nat.lcm (p + n) (q + n) < Nat.lcm p q := by
  sorry

end exists_lcm_lt_l219_219001


namespace tank_capacity_l219_219026

theorem tank_capacity (C : ℝ) (h_leak : ∀ t, t = 6 -> C / 6 = C / t)
    (h_inlet : ∀ r, r = 240 -> r = 4 * 60)
    (h_net : ∀ t, t = 8 -> 240 - C / 6 = C / 8) :
    C = 5760 / 7 := 
by 
  sorry

end tank_capacity_l219_219026


namespace determinant_of_A_l219_219554

-- Define the 2x2 matrix A
def A : Matrix (Fin 2) (Fin 2) ℝ := ![![7, -2], ![-3, 6]]

-- The statement to be proved
theorem determinant_of_A : Matrix.det A = 36 := 
  by sorry

end determinant_of_A_l219_219554


namespace journey_distance_l219_219510

theorem journey_distance :
  ∃ D : ℝ, (D / 42 + D / 48 = 10) ∧ D = 224 :=
by
  sorry

end journey_distance_l219_219510


namespace omega_eq_six_l219_219613

theorem omega_eq_six (A ω : ℝ) (φ : ℝ) (f : ℝ → ℝ) (h1 : A ≠ 0) (h2 : ω > 0)
  (h3 : -π / 2 < φ ∧ φ < π / 2) (h4 : ∀ x, f x = A * Real.sin (ω * x + φ))
  (h5 : ∀ x, f (-x) = -f x) 
  (h6 : ∀ x, f (x + π / 6) = -f (x - π / 6)) :
  ω = 6 :=
sorry

end omega_eq_six_l219_219613


namespace ratio_B_C_l219_219724

def total_money := 595
def A_share := 420
def B_share := 105
def C_share := 70

-- The main theorem stating the expected ratio
theorem ratio_B_C : (B_share / C_share : ℚ) = 3 / 2 := by
  sorry

end ratio_B_C_l219_219724


namespace probability_slope_le_one_l219_219279

noncomputable def point := (ℝ × ℝ)

def Q_in_unit_square (Q : point) : Prop :=
  0 ≤ Q.1 ∧ Q.1 ≤ 1 ∧ 0 ≤ Q.2 ∧ Q.2 ≤ 1

def slope_le_one (Q : point) : Prop :=
  (Q.2 - (1/4)) / (Q.1 - (3/4)) ≤ 1

theorem probability_slope_le_one :
  ∃ p q : ℕ, Q_in_unit_square Q → slope_le_one Q →
  p.gcd q = 1 ∧ (p + q = 11) :=
sorry

end probability_slope_le_one_l219_219279


namespace general_term_of_sequence_l219_219913

theorem general_term_of_sequence 
  (a : ℕ → ℝ)
  (log_a : ℕ → ℝ)
  (h1 : ∀ n, log_a n = Real.log (a n)) 
  (h2 : ∃ d, ∀ n, log_a (n + 1) - log_a n = d)
  (h3 : d = Real.log 3)
  (h4 : log_a 0 + log_a 1 + log_a 2 = 6 * Real.log 3) : 
  ∀ n, a n = 3 ^ n :=
by
  sorry

end general_term_of_sequence_l219_219913


namespace calculate_xy_l219_219710

theorem calculate_xy (x y : ℝ) (h1 : x + y = 12) (h2 : 3 * x + y = 20) : x * y = 32 :=
by
  sorry

end calculate_xy_l219_219710


namespace relayRaceOrders_l219_219582

def countRelayOrders (s1 s2 s3 s4 : String) : Nat :=
  if s1 = "Laura" then
    (if s2 ≠ "Laura" ∧ s3 ≠ "Laura" ∧ s4 ≠ "Laura" then
      if (s2 = "Alice" ∨ s2 = "Bob" ∨ s2 = "Cindy") ∧ 
         (s3 = "Alice" ∨ s3 = "Bob" ∨ s3 = "Cindy") ∧ 
         (s4 = "Alice" ∨ s4 = "Bob" ∨ s4 = "Cindy") then
        if s2 ≠ s3 ∧ s3 ≠ s4 ∧ s2 ≠ s4 then 6 else 0
      else 0
    else 0)
  else 0

theorem relayRaceOrders : countRelayOrders "Laura" "Alice" "Bob" "Cindy" = 6 := 
by sorry

end relayRaceOrders_l219_219582


namespace min_value_of_x_l219_219258

-- Definitions for the conditions given in the problem
def men := 4
def women (x : ℕ) := x
def min_x := 594

-- Definition of the probability p
def C (n k : ℕ) : ℕ := sorry -- Define the binomial coefficient properly

def probability (x : ℕ) : ℚ :=
  (2 * (C (x+1) 2) + (x + 1)) /
  (C (x + 1) 3 + 3 * (C (x + 1) 2) + (x + 1))

-- The theorem statement to prove
theorem min_value_of_x (x : ℕ) : probability x ≤ 1 / 100 →  x = min_x := 
by
  sorry

end min_value_of_x_l219_219258


namespace fraction_condition_l219_219172

theorem fraction_condition (x : ℝ) (h₁ : x > 1) (h₂ : 1 / x < 1) : false :=
sorry

end fraction_condition_l219_219172


namespace sum_of_fractions_l219_219487

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := 
by 
  sorry

end sum_of_fractions_l219_219487


namespace total_wrappers_l219_219246

theorem total_wrappers (a m : ℕ) (ha : a = 34) (hm : m = 15) : a + m = 49 :=
by
  sorry

end total_wrappers_l219_219246


namespace total_number_of_squares_is_13_l219_219528

-- Define the vertices of the region
def region_condition (x y : ℕ) : Prop :=
  y ≤ x ∧ y ≤ 4 ∧ x ≤ 4

-- Define the type of squares whose vertices have integer coordinates
def square (n : ℕ) (x y : ℕ) : Prop :=
  region_condition x y ∧ region_condition (x - n) y ∧ 
  region_condition x (y - n) ∧ region_condition (x - n) (y - n)

-- Count the number of squares of each size within the region
def number_of_squares (size : ℕ) : ℕ :=
  match size with
  | 1 => 10 -- number of 1x1 squares
  | 2 => 3  -- number of 2x2 squares
  | _ => 0  -- there are no larger squares in this context

-- Prove the total number of squares is 13
theorem total_number_of_squares_is_13 : number_of_squares 1 + number_of_squares 2 = 13 :=
by
  sorry

end total_number_of_squares_is_13_l219_219528


namespace intersection_eq_l219_219384

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_eq : A ∩ B = ({-2, 2} : Set ℤ) :=
by
  sorry

end intersection_eq_l219_219384


namespace ticket_sales_l219_219363

-- Definitions of the conditions
theorem ticket_sales (adult_cost child_cost total_people child_count : ℕ)
  (h1 : adult_cost = 8)
  (h2 : child_cost = 1)
  (h3 : total_people = 22)
  (h4 : child_count = 18) :
  (child_count * child_cost + (total_people - child_count) * adult_cost = 50) := by
  sorry

end ticket_sales_l219_219363


namespace total_sales_calculation_l219_219618

def average_price_per_pair : ℝ := 9.8
def number_of_pairs_sold : ℕ := 70
def total_amount : ℝ := 686

theorem total_sales_calculation :
  average_price_per_pair * (number_of_pairs_sold : ℝ) = total_amount :=
by
  -- proof goes here
  sorry

end total_sales_calculation_l219_219618


namespace foci_distance_l219_219048

variable (x y : ℝ)

def ellipse_eq : Prop := (x^2 / 45) + (y^2 / 5) = 9

theorem foci_distance : ellipse_eq x y → (distance_between_foci : ℝ) = 12 * Real.sqrt 10 :=
by
  sorry

end foci_distance_l219_219048


namespace solve_fraction_eq_zero_l219_219110

theorem solve_fraction_eq_zero (x : ℝ) (h₁ : 3 - x = 0) (h₂ : 4 + 2 * x ≠ 0) : x = 3 :=
by sorry

end solve_fraction_eq_zero_l219_219110


namespace minimum_p_l219_219310

-- Define the problem constants and conditions
noncomputable def problem_statement :=
  ∃ p q : ℕ, 
    0 < p ∧ 0 < q ∧ 
    (2008 / 2009 < p / (q : ℚ)) ∧ (p / (q : ℚ) < 2009 / 2010) ∧ 
    (∀ p' q' : ℕ, (0 < p' ∧ 0 < q' ∧ (2008 / 2009 < p' / (q' : ℚ)) ∧ (p' / (q' : ℚ) < 2009 / 2010)) → p ≤ p') 

-- The proof
theorem minimum_p (h : problem_statement) :
  ∃ p q : ℕ, 
    0 < p ∧ 0 < q ∧ 
    (2008 / 2009 < p / (q : ℚ)) ∧ (p / (q : ℚ) < 2009 / 2010) ∧
    p = 4017 :=
sorry

end minimum_p_l219_219310


namespace duration_of_each_movie_l219_219800

-- define the conditions
def num_screens : ℕ := 6
def hours_open : ℕ := 8
def num_movies : ℕ := 24

-- define the total screening time
def total_screening_time : ℕ := num_screens * hours_open

-- define the expected duration of each movie
def movie_duration : ℕ := total_screening_time / num_movies

-- state the theorem
theorem duration_of_each_movie : movie_duration = 2 := by sorry

end duration_of_each_movie_l219_219800


namespace jogging_days_in_second_week_l219_219388

theorem jogging_days_in_second_week
  (daily_jogging_time : ℕ) (first_week_days : ℕ) (total_jogging_time : ℕ) :
  daily_jogging_time = 30 →
  first_week_days = 3 →
  total_jogging_time = 240 →
  ∃ second_week_days : ℕ, second_week_days = 5 :=
by
  intros
  -- Conditions
  have h1 := daily_jogging_time = 30
  have h2 := first_week_days = 3
  have h3 := total_jogging_time = 240
  -- Calculations
  have first_week_time := first_week_days * daily_jogging_time
  have second_week_time := total_jogging_time - first_week_time
  have second_week_days := second_week_time / daily_jogging_time
  -- Conclusion
  use second_week_days
  sorry

end jogging_days_in_second_week_l219_219388


namespace maximize_profit_price_l219_219695

-- Definitions from the conditions
def initial_price : ℝ := 80
def initial_sales : ℝ := 200
def price_reduction_per_unit : ℝ := 1
def sales_increase_per_unit : ℝ := 20
def cost_price_per_helmet : ℝ := 50

-- Profit function
def profit (x : ℝ) : ℝ :=
  (x - cost_price_per_helmet) * (initial_sales + (initial_price - x) * sales_increase_per_unit)

-- The theorem statement
theorem maximize_profit_price : 
  ∃ x, (x = 70) ∧ (∀ y, profit y ≤ profit x) :=
sorry

end maximize_profit_price_l219_219695


namespace A_and_B_together_finish_in_ten_days_l219_219762

-- Definitions of conditions
def B_daily_work := 1 / 15
def A_daily_work := B_daily_work / 2
def combined_daily_work := A_daily_work + B_daily_work

-- The theorem to be proved
theorem A_and_B_together_finish_in_ten_days : 1 / combined_daily_work = 10 := 
  by 
    sorry

end A_and_B_together_finish_in_ten_days_l219_219762


namespace asymptotes_of_hyperbola_l219_219774

theorem asymptotes_of_hyperbola : 
  ∀ (x y : ℝ), 9 * y^2 - 25 * x^2 = 169 → (y = (5/3) * x ∨ y = -(5/3) * x) :=
by 
  sorry

end asymptotes_of_hyperbola_l219_219774


namespace f_of_f_of_f_of_3_l219_219471

def f (x : ℕ) : ℕ := 
  if x > 9 then x - 1 
  else x ^ 3

theorem f_of_f_of_f_of_3 : f (f (f 3)) = 25 :=
by sorry

end f_of_f_of_f_of_3_l219_219471


namespace perfect_squares_less_than_500_ending_in_4_l219_219938

theorem perfect_squares_less_than_500_ending_in_4 : 
  (∃ (squares : Finset ℕ), (∀ n ∈ squares, n < 500 ∧ (n % 10 = 4)) ∧ squares.card = 5) :=
by
  sorry

end perfect_squares_less_than_500_ending_in_4_l219_219938


namespace carol_extra_invitations_l219_219624

theorem carol_extra_invitations : 
  let invitations_per_pack := 3
  let packs_bought := 2
  let friends_to_invite := 9
  packs_bought * invitations_per_pack < friends_to_invite → 
  friends_to_invite - (packs_bought * invitations_per_pack) = 3 :=
by 
  intros _  -- Introduce the condition
  exact sorry  -- Placeholder for the proof

end carol_extra_invitations_l219_219624


namespace find_missing_number_l219_219122

theorem find_missing_number (x : ℚ) (h : (476 + 424) * 2 - x * 476 * 424 = 2704) : 
  x = -1 / 223 :=
by
  sorry

end find_missing_number_l219_219122


namespace coin_flip_difference_l219_219932

/-- The positive difference between the probability of a fair coin landing heads up
exactly 4 times out of 5 flips and the probability of a fair coin landing heads up
5 times out of 5 flips is 1/8. -/
theorem coin_flip_difference :
  (5 * (1 / 2) ^ 5) - ((1 / 2) ^ 5) = (1 / 8) :=
by
  sorry

end coin_flip_difference_l219_219932


namespace remainder_when_dividing_l219_219094

theorem remainder_when_dividing (a : ℕ) (h1 : a = 432 * 44) : a % 38 = 8 :=
by
  -- Proof goes here
  sorry

end remainder_when_dividing_l219_219094


namespace tickets_sold_correctly_l219_219371

theorem tickets_sold_correctly :
  let total := 620
  let cost_per_ticket := 4
  let tickets_sold := 155
  total / cost_per_ticket = tickets_sold :=
by
  sorry

end tickets_sold_correctly_l219_219371


namespace john_trip_time_l219_219551

theorem john_trip_time (normal_distance : ℕ) (normal_time : ℕ) (extra_distance : ℕ) 
  (double_extra_distance : ℕ) (same_speed : ℕ) 
  (h1: normal_distance = 150) 
  (h2: normal_time = 3) 
  (h3: extra_distance = 50)
  (h4: double_extra_distance = 2 * extra_distance)
  (h5: same_speed = normal_distance / normal_time) : 
  normal_time + double_extra_distance / same_speed = 5 :=
by 
  sorry

end john_trip_time_l219_219551


namespace action_figure_total_l219_219787

variable (initial_figures : ℕ) (added_figures : ℕ)

theorem action_figure_total (h₁ : initial_figures = 8) (h₂ : added_figures = 2) : (initial_figures + added_figures) = 10 := by
  sorry

end action_figure_total_l219_219787


namespace missing_digit_in_decimal_representation_of_power_of_two_l219_219145

theorem missing_digit_in_decimal_representation_of_power_of_two :
  (∃ m : ℕ, m < 10 ∧
   ∀ (n : ℕ), (0 ≤ n ∧ n < 10 → n ≠ m) →
     (45 - m) % 9 = (2^29) % 9) :=
sorry

end missing_digit_in_decimal_representation_of_power_of_two_l219_219145


namespace initial_stops_eq_l219_219963

-- Define the total number of stops S
def total_stops : ℕ := 7

-- Define the number of stops made after the initial deliveries
def additional_stops : ℕ := 4

-- Define the number of initial stops as a proof problem
theorem initial_stops_eq : total_stops - additional_stops = 3 :=
by
sorry

end initial_stops_eq_l219_219963


namespace find_remainder_of_n_l219_219427

theorem find_remainder_of_n (n k d : ℕ) (hn_pos : n > 0) (hk_pos : k > 0) (hd_pos_digits : d < 10^k) 
  (h : n * 10^k + d = n * (n + 1) / 2) : n % 9 = 1 :=
sorry

end find_remainder_of_n_l219_219427


namespace count_dracula_is_alive_l219_219768

variable (P Q : Prop)
variable (h1 : P)          -- I am human
variable (h2 : P → Q)      -- If I am human, then Count Dracula is alive

theorem count_dracula_is_alive : Q :=
by
  sorry

end count_dracula_is_alive_l219_219768


namespace biology_class_grades_l219_219611

theorem biology_class_grades (total_students : ℕ)
  (PA PB PC PD : ℕ)
  (h1 : PA = 12 * PB / 10)
  (h2 : PC = PB)
  (h3 : PD = 5 * PB / 10)
  (h4 : PA + PB + PC + PD = total_students) :
  total_students = 40 → PB = 11 := 
by
  sorry

end biology_class_grades_l219_219611


namespace find_set_A_l219_219198

open Set

variable (U : Set ℕ) (A B : Set ℕ)
variable (hU : U = {0, 1, 2, 3, 4, 5})
variable (h1 : (U \ A) ∩ B = {0, 4})
variable (h2 : (U \ A) ∩ (U \ B) = {3, 5})

theorem find_set_A :
  A = {1, 2} :=
by
  sorry

end find_set_A_l219_219198


namespace max_value_of_expression_l219_219170

theorem max_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 * x^2 - 2 * x * y + y^2 = 6) :
  ∃ (a b c d : ℕ), (a + b * Real.sqrt c) / d = 9 + 3 * Real.sqrt 3 ∧ a + b + c + d = 16 :=
by
  sorry

end max_value_of_expression_l219_219170


namespace find_certain_number_l219_219411

theorem find_certain_number (x : ℝ) : 
  ((2 * (x + 5)) / 5 - 5 = 22) → x = 62.5 :=
by
  intro h
  -- Proof goes here
  sorry

end find_certain_number_l219_219411


namespace meeting_time_l219_219686

-- Definitions for the problem conditions.
def track_length : ℕ := 1800
def speed_A_kmph : ℕ := 36
def speed_B_kmph : ℕ := 54

-- Conversion factor from kmph to mps.
def kmph_to_mps (speed_kmph : ℕ) : ℕ := (speed_kmph * 1000) / 3600

-- Calculate the speeds in mps.
def speed_A_mps : ℕ := kmph_to_mps speed_A_kmph
def speed_B_mps : ℕ := kmph_to_mps speed_B_kmph

-- Calculate the time to complete one lap for A and B.
def time_lap_A : ℕ := track_length / speed_A_mps
def time_lap_B : ℕ := track_length / speed_B_mps

-- Prove the time to meet at the starting point.
theorem meeting_time : (Nat.lcm time_lap_A time_lap_B) = 360 := by
  -- Skipping the proof with sorry placeholder
  sorry

end meeting_time_l219_219686


namespace triangle_angle_distance_l219_219527

noncomputable def triangle_properties (ABC P Q R: Type) (angle : ABC → ABC → ABC → ℝ) (dist : ABC → ABC → ℝ) : Prop :=
  ∀ (A B C P Q R : ABC),
    angle B P C = 45 ∧
    angle Q A C = 45 ∧
    angle B C P = 30 ∧
    angle A C Q = 30 ∧
    angle A B R = 15 ∧
    angle B A R = 15 →
    angle P R Q = 90 ∧
    dist Q R = dist P R

theorem triangle_angle_distance (ABC P Q R: Type) (angle : ABC → ABC → ABC → ℝ) (dist : ABC → ABC → ℝ) :
  triangle_properties ABC P Q R angle dist →
  ∀ (A B C P Q R : ABC),
    angle B P C = 45 ∧
    angle Q A C = 45 ∧
    angle B C P = 30 ∧
    angle A C Q = 30 ∧
    angle A B R = 15 ∧
    angle B A R = 15 →
    angle P R Q = 90 ∧
    dist Q R = dist P R :=
by intros; sorry

end triangle_angle_distance_l219_219527


namespace blanch_breakfast_slices_l219_219075

-- Define the initial number of slices
def initial_slices : ℕ := 15

-- Define the slices eaten at different times
def lunch_slices : ℕ := 2
def snack_slices : ℕ := 2
def dinner_slices : ℕ := 5

-- Define the number of slices left
def slices_left : ℕ := 2

-- Calculate the total slices eaten during lunch, snack, and dinner
def total_eaten_ex_breakfast : ℕ := lunch_slices + snack_slices + dinner_slices

-- Define the slices eaten during breakfast
def breakfast_slices : ℕ := initial_slices - total_eaten_ex_breakfast - slices_left

-- The theorem to prove
theorem blanch_breakfast_slices : breakfast_slices = 4 := by
  sorry

end blanch_breakfast_slices_l219_219075


namespace fruit_basket_cost_is_28_l219_219499

def basket_total_cost : ℕ := 4 * 1 + 3 * 2 + (24 / 12) * 4 + 2 * 3 + 2 * 2

theorem fruit_basket_cost_is_28 : basket_total_cost = 28 := by
  sorry

end fruit_basket_cost_is_28_l219_219499


namespace quad_completion_l219_219970

theorem quad_completion (a b c : ℤ) 
    (h : ∀ x : ℤ, 8 * x^2 - 48 * x - 128 = a * (x + b)^2 + c) : 
    a + b + c = -195 := 
by
  sorry

end quad_completion_l219_219970


namespace triangle_is_right_triangle_l219_219323

theorem triangle_is_right_triangle
  (a b c : ℝ)
  (h : a^2 + b^2 + c^2 - 10 * a - 6 * b - 8 * c + 50 = 0) :
  a^2 = b^2 + c^2 ∨ b^2 = a^2 + c^2 ∨ c^2 = a^2 + b^2 :=
sorry

end triangle_is_right_triangle_l219_219323


namespace quadratic_function_conditions_l219_219665

noncomputable def quadratic_function_example (x : ℝ) : ℝ :=
  -2.5 * x^2 + 15 * x - 12.5

theorem quadratic_function_conditions :
  quadratic_function_example 1 = 0 ∧
  quadratic_function_example 5 = 0 ∧
  quadratic_function_example 3 = 10 :=
by
  sorry

end quadratic_function_conditions_l219_219665


namespace max_value_ratio_l219_219968

/-- Define the conditions on function f and variables x and y. -/
def conditions (f : ℝ → ℝ) (x y : ℝ) :=
  (∀ x, f (-x) + f x = 0) ∧
  (∀ x1 x2, x1 < x2 → f x1 < f x2) ∧
  f (x^2 - 6 * x) + f (y^2 - 4 * y + 12) ≤ 0

/-- The maximum value of (y - 2) / x under the given conditions. -/
theorem max_value_ratio (f : ℝ → ℝ) (x y : ℝ) (cond : conditions f x y) :
  (y - 2) / x ≤ (Real.sqrt 2) / 4 :=
sorry

end max_value_ratio_l219_219968


namespace find_x_when_y_is_20_l219_219374

-- Definition of the problem conditions.
def constant_ratio (x y : ℝ) : Prop := ∃ k, (3 * x - 4) = k * (y + 7)

-- Main theorem statement.
theorem find_x_when_y_is_20 :
  (constant_ratio x 5 → constant_ratio 3 5) → 
  (constant_ratio x 20 → x = 5.0833) :=
  by sorry

end find_x_when_y_is_20_l219_219374


namespace decreasing_population_density_l219_219653

def Population (t : Type) : Type := t

variable (stable_period: Prop)
variable (infertility: Prop)
variable (death_rate_exceeds_birth_rate: Prop)
variable (complex_structure: Prop)

theorem decreasing_population_density :
  death_rate_exceeds_birth_rate → true := sorry

end decreasing_population_density_l219_219653


namespace simplify_powers_l219_219776

-- Defining the multiplicative rule for powers
def power_mul (x : ℕ) (a b : ℕ) : ℕ := x^(a+b)

-- Proving that x^5 * x^6 = x^11
theorem simplify_powers (x : ℕ) : x^5 * x^6 = x^11 :=
by
  change x^5 * x^6 = x^(5 + 6)
  sorry

end simplify_powers_l219_219776


namespace length_of_AB_l219_219657

variables (AB CD : ℝ)

-- Given conditions
def area_ratio (h : ℝ) : Prop := (1/2 * AB * h) / (1/2 * CD * h) = 4
def sum_condition : Prop := AB + CD = 200

-- The proof problem: proving the length of AB
theorem length_of_AB (h : ℝ) (h_area_ratio : area_ratio AB CD h) 
  (h_sum_condition : sum_condition AB CD) : AB = 160 :=
sorry

end length_of_AB_l219_219657


namespace each_person_gets_9_wings_l219_219982

noncomputable def chicken_wings_per_person (initial_wings : ℕ) (additional_wings : ℕ) (friends : ℕ) : ℕ :=
  (initial_wings + additional_wings) / friends

theorem each_person_gets_9_wings :
  chicken_wings_per_person 20 25 5 = 9 :=
by
  sorry

end each_person_gets_9_wings_l219_219982


namespace min_score_needed_l219_219543

-- Definitions of the conditions
def current_scores : List ℤ := [88, 92, 75, 81, 68, 70]
def desired_increase : ℤ := 5
def number_of_tests := current_scores.length
def current_total : ℤ := current_scores.sum
def current_average : ℤ := current_total / number_of_tests
def desired_average : ℤ := current_average + desired_increase 
def new_number_of_tests : ℤ := number_of_tests + 1
def total_required_score : ℤ := desired_average * new_number_of_tests

-- Lean 4 statement (theorem) to prove
theorem min_score_needed : total_required_score - current_total = 114 := by
  sorry

end min_score_needed_l219_219543


namespace remaining_area_is_344_l219_219771

def garden_length : ℕ := 20
def garden_width : ℕ := 18
def shed_side : ℕ := 4

def area_rectangle : ℕ := garden_length * garden_width
def area_shed : ℕ := shed_side * shed_side

def remaining_garden_area : ℕ := area_rectangle - area_shed

theorem remaining_area_is_344 : remaining_garden_area = 344 := by
  sorry

end remaining_area_is_344_l219_219771


namespace find_m_l219_219765

def vector_parallel (a b : ℝ × ℝ) : Prop := 
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem find_m
  (m : ℝ)
  (a : ℝ × ℝ := (m, 1))
  (b : ℝ × ℝ := (2, -1))
  (h : vector_parallel a (b.1 - a.1, b.2 - a.2)) :
  m = -2 :=
by
  sorry

end find_m_l219_219765


namespace complement_of_M_is_correct_l219_219444

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 3}

-- Define the complement of M in U
def complement_M_in_U : Set ℝ := {x : ℝ | x < -1 ∨ x > 3}

-- State the theorem
theorem complement_of_M_is_correct : (U \ M) = complement_M_in_U := by sorry

end complement_of_M_is_correct_l219_219444


namespace largest_root_of_equation_l219_219772

theorem largest_root_of_equation : ∃ (x : ℝ), (x - 37)^2 - 169 = 0 ∧ ∀ y, (y - 37)^2 - 169 = 0 → y ≤ x :=
by
  sorry

end largest_root_of_equation_l219_219772


namespace ratio_of_area_of_small_triangle_to_square_l219_219402

theorem ratio_of_area_of_small_triangle_to_square
  (n : ℕ)
  (square_area : ℝ)
  (A1 : square_area > 0)
  (ADF_area : ℝ)
  (H1 : ADF_area = n * square_area)
  (FEC_area : ℝ)
  (H2 : FEC_area = 1 / (4 * n)) :
  FEC_area / square_area = 1 / (4 * n) :=
by
  sorry

end ratio_of_area_of_small_triangle_to_square_l219_219402


namespace Eunji_score_equals_56_l219_219326

theorem Eunji_score_equals_56 (Minyoung_score Yuna_score : ℕ) (Eunji_score : ℕ) 
  (h1 : Minyoung_score = 55) (h2 : Yuna_score = 57)
  (h3 : Eunji_score > Minyoung_score) (h4 : Eunji_score < Yuna_score) : Eunji_score = 56 := by
  -- Given the hypothesis, it is a fact that Eunji's score is 56.
  sorry

end Eunji_score_equals_56_l219_219326


namespace max_ratio_square_l219_219361

variables {a b c x y : ℝ}
-- Assume a, b, c are positive real numbers
variable (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
-- Assume the order of a, b, c: a ≥ b ≥ c
variable (h_order : a ≥ b ∧ b ≥ c)
-- Define the system of equations
variable (h_system : a^2 + y^2 = c^2 + x^2 ∧ c^2 + x^2 = (a - x)^2 + (c - y)^2)
-- Assume the constraints on x and y
variable (h_constraints : 0 ≤ x ∧ x < a ∧ 0 ≤ y ∧ y < c)

theorem max_ratio_square :
  ∃ (ρ : ℝ), ρ = (a / c) ∧ ρ^2 = 4 / 3 :=
sorry

end max_ratio_square_l219_219361


namespace opposite_of_lime_is_black_l219_219933

-- Given colors of the six faces
inductive Color
| Purple | Cyan | Magenta | Silver | Lime | Black

-- Hinged squares forming a cube
structure Cube :=
(top : Color) (bottom : Color) (front : Color) (back : Color) (left : Color) (right : Color)

-- Condition: Magenta is on the top
def magenta_top (c : Cube) : Prop := c.top = Color.Magenta

-- Problem statement: Prove the color opposite to Lime is Black
theorem opposite_of_lime_is_black (c : Cube) (HM : magenta_top c) (HL : c.front = Color.Lime)
    (HBackFace : c.back = Color.Black) : c.back = Color.Black := 
sorry

end opposite_of_lime_is_black_l219_219933


namespace no_attention_prob_l219_219737

noncomputable def prob_no_attention (p1 p2 p3 : ℝ) : ℝ :=
  (1 - p1) * (1 - p2) * (1 - p3)

theorem no_attention_prob :
  let p1 := 0.9
  let p2 := 0.8
  let p3 := 0.6
  prob_no_attention p1 p2 p3 = 0.008 :=
by
  unfold prob_no_attention
  sorry

end no_attention_prob_l219_219737


namespace cos_180_eq_neg_one_l219_219575

theorem cos_180_eq_neg_one : Real.cos (180 * Real.pi / 180) = -1 := by
  sorry

end cos_180_eq_neg_one_l219_219575


namespace evaluate_expression_l219_219233

-- Definition of the given condition.
def sixty_four_eq_sixteen_squared : Prop := 64 = 16^2

-- The statement to prove that the given expression equals the answer.
theorem evaluate_expression (h : sixty_four_eq_sixteen_squared) : 
  (16^24) / (64^8) = 16^8 :=
by 
  -- h contains the condition that 64 = 16^2, but we provide a proof step later with sorry
  sorry

end evaluate_expression_l219_219233


namespace xyz_value_l219_219976

noncomputable def find_xyz (x y z : ℝ) 
  (h₁ : (x + y + z) * (x * y + x * z + y * z) = 45)
  (h₂ : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 14)
  (h₃ : (x + y + z)^2 = 25) : ℝ :=
  if (x * y * z = 31 / 3) then 31 / 3 else 0  -- This should hold with the given conditions

theorem xyz_value (x y z : ℝ)
  (h₁ : (x + y + z) * (x * y + x * z + y * z) = 45)
  (h₂ : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 14)
  (h₃ : (x + y + z)^2 = 25) :
  find_xyz x y z h₁ h₂ h₃ = 31 / 3 :=
by 
  sorry  -- The proof should demonstrate that find_xyz equals 31 / 3 given the conditions

end xyz_value_l219_219976


namespace initial_ratio_of_milk_to_water_l219_219376

variable (M W : ℕ)
noncomputable def M_initial := 45 - W
noncomputable def W_new := W + 9

theorem initial_ratio_of_milk_to_water :
  M_initial = 36 ∧ W = 9 →
  M_initial / (W + 9) = 2 ↔ 4 = M_initial / W := 
sorry

end initial_ratio_of_milk_to_water_l219_219376


namespace extrema_range_of_m_l219_219711

def has_extrema (f : ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, (∀ z : ℝ, z ≤ x → f z ≤ f x) ∧ (∀ z : ℝ, z ≥ y → f z ≤ f y)

noncomputable def f (m x : ℝ) : ℝ :=
  x^3 + m * x^2 + (m + 6) * x + 1

theorem extrema_range_of_m (m : ℝ) :
  has_extrema (f m) ↔ (m ∈ Set.Iic (-3) ∪ Set.Ici 6) :=
by
  sorry

end extrema_range_of_m_l219_219711


namespace julia_more_kids_on_monday_l219_219856

-- Definition of the problem statement
def playedWithOnMonday : ℕ := 6
def playedWithOnTuesday : ℕ := 5
def difference := playedWithOnMonday - playedWithOnTuesday

theorem julia_more_kids_on_monday : difference = 1 :=
by
  -- Proof can be filled out here.
  sorry

end julia_more_kids_on_monday_l219_219856


namespace remainder_n_sq_plus_3n_5_mod_25_l219_219745

theorem remainder_n_sq_plus_3n_5_mod_25 (k : ℤ) (n : ℤ) (h : n = 25 * k - 1) : 
  (n^2 + 3 * n + 5) % 25 = 3 := 
by
  sorry

end remainder_n_sq_plus_3n_5_mod_25_l219_219745


namespace chess_tournament_l219_219537

def number_of_players := 30

def total_games (n : ℕ) : ℕ := n * (n - 1) / 2

theorem chess_tournament : total_games number_of_players = 435 := by
  sorry

end chess_tournament_l219_219537


namespace arcsin_sqrt2_over_2_eq_pi_over_4_l219_219369

theorem arcsin_sqrt2_over_2_eq_pi_over_4 :
  Real.arcsin (Real.sqrt 2 / 2) = Real.pi / 4 :=
sorry

end arcsin_sqrt2_over_2_eq_pi_over_4_l219_219369


namespace time_lent_to_C_eq_l219_219713

variable (principal_B : ℝ := 5000)
variable (time_B : ℕ := 2)
variable (principal_C : ℝ := 3000)
variable (total_interest : ℝ := 1980)
variable (rate_of_interest_per_annum : ℝ := 0.09)

theorem time_lent_to_C_eq (n : ℝ) (H : principal_B * rate_of_interest_per_annum * time_B + principal_C * rate_of_interest_per_annum * n = total_interest) : 
  n = 2 / 3 :=
by
  sorry

end time_lent_to_C_eq_l219_219713


namespace work_completion_l219_219347

theorem work_completion (d : ℝ) :
  (9 * (1 / d) + 8 * (1 / 20) = 1) ↔ (d = 15) :=
by
  sorry

end work_completion_l219_219347


namespace football_banquet_total_food_l219_219687

-- Definitions representing the conditions
def individual_max_food (n : Nat) := n ≤ 2
def min_guests (g : Nat) := g ≥ 160

-- The proof problem statement
theorem football_banquet_total_food : 
  ∀ (n g : Nat), (∀ i, i ≤ g → individual_max_food n) ∧ min_guests g → g * n = 320 := 
by
  intros n g h
  sorry

end football_banquet_total_food_l219_219687


namespace green_function_solution_l219_219831

noncomputable def G (x ξ : ℝ) (α : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ ξ then α + Real.log ξ else if ξ ≤ x ∧ x ≤ 1 then α + Real.log x else 0

theorem green_function_solution (x ξ α : ℝ) (hα : α ≠ 0) (hx_bound : 0 < x ∧ x ≤ 1) :
  ( G x ξ α = if 0 < x ∧ x ≤ ξ then α + Real.log ξ else if ξ ≤ x ∧ x ≤ 1 then α + Real.log x else 0 ) :=
sorry

end green_function_solution_l219_219831


namespace nonneg_integer_solutions_l219_219646

theorem nonneg_integer_solutions :
  { x : ℕ | 5 * x + 3 < 3 * (2 + x) } = {0, 1} :=
by
  sorry

end nonneg_integer_solutions_l219_219646


namespace determine_value_of_x_l219_219478

theorem determine_value_of_x (x y z : ℤ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxy : x ≥ y) (hyz : y ≥ z)
  (h1 : x^2 - y^2 - z^2 + x * y = 4033) 
  (h2 : x^2 + 4 * y^2 + 4 * z^2 - 4 * x * y - 3 * x * z - 3 * y * z = -3995) : 
  x = 69 := sorry

end determine_value_of_x_l219_219478


namespace jakes_weight_l219_219961

theorem jakes_weight
  (J K : ℝ)
  (h1 : J - 8 = 2 * K)
  (h2 : J + K = 290) :
  J = 196 :=
by
  sorry

end jakes_weight_l219_219961


namespace Nell_has_123_more_baseball_cards_than_Ace_cards_l219_219905

def Nell_cards_diff (baseball_cards_new : ℕ) (ace_cards_new : ℕ) : ℕ :=
  baseball_cards_new - ace_cards_new

theorem Nell_has_123_more_baseball_cards_than_Ace_cards:
  (Nell_cards_diff 178 55) = 123 :=
by
  -- proof here
  sorry

end Nell_has_123_more_baseball_cards_than_Ace_cards_l219_219905


namespace acute_triangle_properties_l219_219622

theorem acute_triangle_properties (A B C : ℝ) (AC BC : ℝ)
  (h_acute : ∀ {x : ℝ}, x = A ∨ x = B ∨ x = C → x < π / 2)
  (h_BC : BC = 1)
  (h_B_eq_2A : B = 2 * A) :
  (AC / Real.cos A = 2) ∧ (Real.sqrt 2 < AC ∧ AC < Real.sqrt 3) :=
by
  sorry

end acute_triangle_properties_l219_219622


namespace ratio_of_raspberries_l219_219422

theorem ratio_of_raspberries (B R K L : ℕ) (h1 : B = 42) (h2 : L = 7) (h3 : K = B / 3) (h4 : B = R + K + L) :
  R / Nat.gcd R B = 1 ∧ B / Nat.gcd R B = 2 :=
by
  sorry

end ratio_of_raspberries_l219_219422


namespace tangent_at_5_eqn_l219_219400

noncomputable def f : ℝ → ℝ := sorry

axiom f_even : ∀ x : ℝ, f (-x) = f x
axiom f_period : ∀ x : ℝ, f (x + 2) = f (2 - x)
axiom tangent_at_neg1 : ∀ x y : ℝ, x - y + 3 = 0 → x = -1 → y = f x

theorem tangent_at_5_eqn : 
  ∀ x y : ℝ, x = 5 → y = f x → x + y - 7 = 0 :=
sorry

end tangent_at_5_eqn_l219_219400


namespace trig_identity_example_l219_219046

theorem trig_identity_example (α : ℝ) (h : Real.tan α = 1 / 2) : 
  (4 * Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 2 / 3 :=
by
  sorry

end trig_identity_example_l219_219046


namespace fraction_to_percentage_l219_219521

theorem fraction_to_percentage (y : ℝ) (h : y > 0) : ((7 * y) / 20 + (3 * y) / 10) = 0.65 * y :=
by
  -- the proof steps will go here
  sorry

end fraction_to_percentage_l219_219521


namespace parabola_equation_conditions_l219_219602

def focus_on_x_axis (focus : ℝ × ℝ) := (∃ x : ℝ, focus = (x, 0))
def foot_of_perpendicular (line : ℝ × ℝ → Prop) (focus : ℝ × ℝ) :=
  (∃ point : ℝ × ℝ, point = (2, 1) ∧ line focus ∧ line point ∧ line (0, 0))

theorem parabola_equation_conditions (focus : ℝ × ℝ) (line : ℝ × ℝ → Prop) :
  focus_on_x_axis focus →
  foot_of_perpendicular line focus →
  ∃ a : ℝ, ∀ x y : ℝ, y^2 = a * x ↔ y^2 = 10 * x :=
by
  intros h1 h2
  use 10
  sorry

end parabola_equation_conditions_l219_219602


namespace slices_per_large_pizza_l219_219135

structure PizzaData where
  total_pizzas : Nat
  small_pizzas : Nat
  medium_pizzas : Nat
  slices_per_small : Nat
  slices_per_medium : Nat
  total_slices : Nat

def large_slices (data : PizzaData) : Nat := (data.total_slices - (data.small_pizzas * data.slices_per_small + data.medium_pizzas * data.slices_per_medium)) / (data.total_pizzas - data.small_pizzas - data.medium_pizzas)

def PizzaSlicingConditions := {data : PizzaData // 
  data.total_pizzas = 15 ∧
  data.small_pizzas = 4 ∧
  data.medium_pizzas = 5 ∧
  data.slices_per_small = 6 ∧
  data.slices_per_medium = 8 ∧
  data.total_slices = 136}

theorem slices_per_large_pizza (data : PizzaSlicingConditions) : large_slices data.val = 12 :=
by
  sorry

end slices_per_large_pizza_l219_219135


namespace second_quadrant_necessary_not_sufficient_l219_219287

open Classical

-- Definitions
def isSecondQuadrant (α : ℝ) : Prop := 90 < α ∧ α < 180
def isObtuseAngle (α : ℝ) : Prop := 90 < α ∧ α < 180 ∨ 180 < α ∧ α < 270

-- The theorem statement
theorem second_quadrant_necessary_not_sufficient (α : ℝ) :
  (isSecondQuadrant α → isObtuseAngle α) ∧ ¬(isSecondQuadrant α ↔ isObtuseAngle α) :=
by
  sorry

end second_quadrant_necessary_not_sufficient_l219_219287


namespace inequality_solution_set_l219_219638

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def f' (x : ℝ) : ℝ := sorry

axiom deriv_cond : ∀ (x : ℝ), x ≠ 0 → f' x < (2 * f x) / x
axiom zero_points : f (-2) = 0 ∧ f 1 = 0

theorem inequality_solution_set :
  {x : ℝ | x * f x < 0} = { x : ℝ | (-2 ≤ x ∧ x < 0) ∨ (0 < x ∧ x ≤ 1) } :=
sorry

end inequality_solution_set_l219_219638


namespace problem1_problem2_l219_219045

section
variable {α : Real}
variable (tan_α : Real)
variable (sin_α cos_α : Real)

def trigonometric_identities (tan_α sin_α cos_α : Real) : Prop :=
  tan_α = 2 ∧ sin_α = tan_α * cos_α

theorem problem1 (h : trigonometric_identities tan_α sin_α cos_α) :
  (4 * sin_α - 2 * cos_α) / (5 * cos_α + 3 * sin_α) = 6 / 11 := by
  sorry

theorem problem2 (h : trigonometric_identities tan_α sin_α cos_α) :
  (1 / 4 * sin_α^2 + 1 / 3 * sin_α * cos_α + 1 / 2 * cos_α^2) = 13 / 30 := by
  sorry
end

end problem1_problem2_l219_219045


namespace infinite_geometric_series_l219_219011

theorem infinite_geometric_series
  (p q r : ℝ)
  (h_series : ∑' n : ℕ, p / q^(n+1) = 9) :
  (∑' n : ℕ, p / (p + r)^(n+1)) = (9 * (q - 1)) / (9 * q + r - 10) :=
by 
  sorry

end infinite_geometric_series_l219_219011


namespace sum_x_coordinates_eq_3_l219_219902

def f : ℝ → ℝ := sorry -- definition of the function f as given by the five line segments

theorem sum_x_coordinates_eq_3 :
  (∃ x1 x2 x3 : ℝ, (f x1 = x1 + 1 ∧ f x2 = x2 + 1 ∧ f x3 = x3 + 1) ∧ (x1 + x2 + x3 = 3)) :=
sorry

end sum_x_coordinates_eq_3_l219_219902


namespace excess_calories_l219_219690

theorem excess_calories (bags : ℕ) (ounces_per_bag : ℕ) (calories_per_ounce : ℕ)
  (run_minutes : ℕ) (calories_per_minute : ℕ)
  (h_bags : bags = 3) (h_ounces_per_bag : ounces_per_bag = 2)
  (h_calories_per_ounce : calories_per_ounce = 150)
  (h_run_minutes : run_minutes = 40)
  (h_calories_per_minute : calories_per_minute = 12) :
  (bags * ounces_per_bag * calories_per_ounce) - (run_minutes * calories_per_minute) = 420 := by
  sorry

end excess_calories_l219_219690


namespace rt_triangle_case1_rt_triangle_case2_rt_triangle_case3_l219_219729

-- Case 1
theorem rt_triangle_case1
  (a : ℝ) (b : ℝ) (c : ℝ) (A B C : ℝ)
  (h : A = 30) (h_bc : B + C = 90) (h_ac : A + C = 90)
  (ha : a = 4) (hb : b = 4 * Real.sqrt 3) (hc : c = 8)
  : b = 4 * Real.sqrt 3 ∧ c = 8 := by
  sorry

-- Case 2
theorem rt_triangle_case2
  (a : ℝ) (b : ℝ) (c : ℝ) (A B C : ℝ)
  (h : B = 60) (h_bc : B + C = 90) (h_ac : A + C = 90)
  (ha : a = Real.sqrt 3 - 1) (hb : b = 3 - Real.sqrt 3) 
  (ha_b: A = 30)
  (h_c: c = 2 * Real.sqrt 3 - 2)
  : B = 60 ∧ A = 30 ∧ c = 2 * Real.sqrt 3 - 2 := by
  sorry

-- Case 3
theorem rt_triangle_case3
  (a : ℝ) (b : ℝ) (c : ℝ) (A B C : ℝ)
  (h : A = 60) (h_bc : B + C = 90) (h_ac : A + C = 90)
  (hc : c = 2 + Real.sqrt 3)
  (ha : a = Real.sqrt 3 + 3/2) 
  (hb: b = (2 + Real.sqrt 3) / 2)
  : a = Real.sqrt 3 + 3/2 ∧ b = (2 + Real.sqrt 3) / 2 := by
  sorry

end rt_triangle_case1_rt_triangle_case2_rt_triangle_case3_l219_219729


namespace sparrows_among_non_robins_percentage_l219_219891

-- Define percentages of different birds
def finches_percentage : ℝ := 0.40
def sparrows_percentage : ℝ := 0.20
def owls_percentage : ℝ := 0.15
def robins_percentage : ℝ := 0.25

-- Define the statement to prove 
theorem sparrows_among_non_robins_percentage :
  ((sparrows_percentage / (1 - robins_percentage)) * 100) = 26.67 := by
  -- This is where the proof would go, but it's omitted as per instructions
  sorry

end sparrows_among_non_robins_percentage_l219_219891


namespace line_passes_point_a_ne_zero_l219_219939

theorem line_passes_point_a_ne_zero (a : ℝ) (h1 : ∀ (x y : ℝ), (y = 5 * x + a) → (x = a ∧ y = a^2)) (h2 : a ≠ 0) : a = 6 :=
sorry

end line_passes_point_a_ne_zero_l219_219939


namespace apples_in_basket_l219_219313

theorem apples_in_basket (x : ℕ) (h1 : 22 * x = (x + 45) * 13) : 22 * x = 1430 :=
by
  sorry

end apples_in_basket_l219_219313


namespace minimum_value_expression_l219_219213

theorem minimum_value_expression (a b c d e f : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f) 
(h_sum : a + b + c + d + e + f = 7) : 
  ∃ min_val : ℝ, min_val = 63 ∧ 
  (∀ a b c d e f : ℝ, 0 < a → 0 < b → 0 < c → 0 < d → 0 < e → 0 < f → a + b + c + d + e + f = 7 → 
  (1 / a + 4 / b + 9 / c + 16 / d + 25 / e + 36 / f) ≥ min_val) := 
sorry

end minimum_value_expression_l219_219213


namespace find_x_l219_219423

theorem find_x (p q : ℕ) (h1 : 1 < p) (h2 : 1 < q) (h3 : 17 * (p + 1) = (14 * (q + 1))) (h4 : p + q = 40) : 
    x = 14 := 
by
  sorry

end find_x_l219_219423


namespace television_final_price_l219_219461

theorem television_final_price :
  let original_price := 1200
  let discount_percent := 0.30
  let tax_percent := 0.08
  let rebate := 50
  let discount := discount_percent * original_price
  let sale_price := original_price - discount
  let tax := tax_percent * sale_price
  let price_including_tax := sale_price + tax
  let final_amount := price_including_tax - rebate
  final_amount = 857.2 :=
by
{
  -- The proof would go here, but it's omitted as per instructions.
  sorry
}

end television_final_price_l219_219461


namespace dice_probability_l219_219316

noncomputable def probability_each_number_appears_at_least_once : ℝ :=
  1 - (6 * (5/6)^10 - 15 * (4/6)^10 + 20 * (3/6)^10 - 15 * (2/6)^10 + 6 * (1/6)^10)

theorem dice_probability : probability_each_number_appears_at_least_once = 0.272 :=
by
  sorry

end dice_probability_l219_219316


namespace functional_equation_solution_l219_219208

theorem functional_equation_solution (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x y : ℝ, (x + y) * (f x - f y) = (x - y) * f (x + y)) →
  ∀ x : ℝ, f x = a * x^2 + b * x :=
by
  intro h
  intro x
  have : ∀ x y : ℝ, (x + y) * (f x - f y) = (x - y) * f (x + y) := h
  sorry

end functional_equation_solution_l219_219208


namespace billy_apples_ratio_l219_219486

theorem billy_apples_ratio :
  let monday := 2
  let tuesday := 2 * monday
  let wednesday := 9
  let friday := monday / 2
  let total_apples := 20
  let thursday := total_apples - (monday + tuesday + wednesday + friday)
  thursday / friday = 4 := 
by
  let monday := 2
  let tuesday := 2 * monday
  let wednesday := 9
  let friday := monday / 2
  let total_apples := 20
  let thursday := total_apples - (monday + tuesday + wednesday + friday)
  sorry

end billy_apples_ratio_l219_219486


namespace Karen_packs_piece_of_cake_days_l219_219546

theorem Karen_packs_piece_of_cake_days 
(Total Ham_Days : ℕ) (Ham_probability Cake_probability : ℝ) 
  (H_Total : Total = 5) 
  (H_Ham_Days : Ham_Days = 3) 
  (H_Ham_probability : Ham_probability = (3 / 5)) 
  (H_Cake_probability : Ham_probability * (Cake_probability / 5) = 0.12) : 
  Cake_probability = 1 := 
by
  sorry

end Karen_packs_piece_of_cake_days_l219_219546


namespace piesEatenWithForksPercentage_l219_219288

def totalPies : ℕ := 2000
def notEatenWithForks : ℕ := 640
def eatenWithForks : ℕ := totalPies - notEatenWithForks

def percentageEatenWithForks := (eatenWithForks : ℚ) / totalPies * 100

theorem piesEatenWithForksPercentage : percentageEatenWithForks = 68 := by
  sorry

end piesEatenWithForksPercentage_l219_219288


namespace six_hundred_sixes_not_square_l219_219806

theorem six_hundred_sixes_not_square : 
  ∀ (n : ℕ), (n = 66666666666666666666666666666666666666666666666666666666666 -- continued 600 times
  ∨ n = 66666666666666666666666666666666666666666666666666666666666 -- continued with some zeros
  ) → ¬ (∃ k : ℕ, k * k = n) := 
by
  sorry

end six_hundred_sixes_not_square_l219_219806


namespace zero_of_f_l219_219451

noncomputable def f (x : ℝ) : ℝ := Real.logb 5 (x - 1)

theorem zero_of_f :
  ∃ x : ℝ, f x = 0 ∧ x = 2 :=
by
  use 2
  unfold f
  sorry -- Skip the proof steps, as instructed.

end zero_of_f_l219_219451


namespace compare_a_b_l219_219436

theorem compare_a_b (a b : ℝ) (h : 5 * (a - 1) = b + a ^ 2) : a > b :=
sorry

end compare_a_b_l219_219436


namespace number_of_zeros_l219_219915

noncomputable def f (x : ℝ) : ℝ := |2^x - 1| - 3^x

theorem number_of_zeros : ∃! x : ℝ, f x = 0 := sorry

end number_of_zeros_l219_219915


namespace total_yield_UncleLi_yield_difference_l219_219906

-- Define the conditions related to Uncle Li and Aunt Lin
def UncleLiAcres : ℕ := 12
def UncleLiYieldPerAcre : ℕ := 660
def AuntLinAcres : ℕ := UncleLiAcres - 2
def AuntLinTotalYield : ℕ := UncleLiYieldPerAcre * UncleLiAcres - 420

-- Prove the total yield of Uncle Li's rice
theorem total_yield_UncleLi : UncleLiYieldPerAcre * UncleLiAcres = 7920 := by
  sorry

-- Prove how much less the yield per acre of Uncle Li's rice is compared to Aunt Lin's
theorem yield_difference :
  UncleLiYieldPerAcre - AuntLinTotalYield / AuntLinAcres = 90 := by
  sorry

end total_yield_UncleLi_yield_difference_l219_219906


namespace angle_B_in_right_triangle_in_degrees_l219_219886

def angleSum (A B C: ℝ) : Prop := A + B + C = 180

theorem angle_B_in_right_triangle_in_degrees (A B C : ℝ) (h1 : C = 90) (h2 : A = 35.5) (h3 : angleSum A B C) : B = 54.5 := 
by
  sorry

end angle_B_in_right_triangle_in_degrees_l219_219886


namespace equivalence_l219_219846

theorem equivalence (a b c : ℝ) (h : a + c = 2 * b) : a^2 + 8 * b * c = (2 * b + c)^2 := 
by 
  sorry

end equivalence_l219_219846


namespace cost_of_seven_books_l219_219897

theorem cost_of_seven_books (h : 3 * 12 = 36) : 7 * 12 = 84 :=
sorry

end cost_of_seven_books_l219_219897


namespace surface_dots_sum_l219_219442

-- Define the sum of dots on opposite faces of a standard die
axiom sum_opposite_faces (x y : ℕ) : x + y = 7

-- Define the large cube dimensions
def large_cube_dimension : ℕ := 3

-- Define the total number of small cubes
def num_small_cubes : ℕ := large_cube_dimension ^ 3

-- Calculate the number of faces on the surface of the large cube
def num_surface_faces : ℕ := 6 * large_cube_dimension ^ 2

-- Given the sum of opposite faces, compute the total number of dots on the surface
theorem surface_dots_sum : num_surface_faces / 2 * 7 = 189 := by
  sorry

end surface_dots_sum_l219_219442


namespace find_r_in_geometric_sum_l219_219927

theorem find_r_in_geometric_sum (S_n : ℕ → ℕ) (r : ℤ)
  (hSn : ∀ n : ℕ, S_n n = 2 * 3^n + r)
  (hgeo : ∀ n : ℕ, n ≥ 2 → S_n n - S_n (n - 1) = 4 * 3^(n - 1))
  (hn1 : S_n 1 = 6 + r) :
  r = -2 :=
by
  sorry

end find_r_in_geometric_sum_l219_219927


namespace intersecting_parabolas_circle_radius_sq_l219_219681

theorem intersecting_parabolas_circle_radius_sq:
  (∀ (x y : ℝ), (y = (x + 1)^2 ∧ x + 4 = (y - 3)^2) → 
  ((x + 1/2)^2 + (y - 7/2)^2 = 13/2)) := sorry

end intersecting_parabolas_circle_radius_sq_l219_219681


namespace problem_l219_219522

noncomputable def x : ℝ := 123.75
noncomputable def y : ℝ := 137.5
noncomputable def original_value : ℝ := 125

theorem problem (y_more : y = original_value + 0.1 * original_value) (x_less : x = y * 0.9) : y = 137.5 :=
by
  sorry

end problem_l219_219522


namespace indistinguishable_distributions_l219_219243

def ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) : ℕ :=
  if boxes = 2 && balls = 6 then 4 else 0

theorem indistinguishable_distributions : ways_to_distribute_balls 6 2 = 4 :=
by sorry

end indistinguishable_distributions_l219_219243


namespace f_odd_f_periodic_f_def_on_interval_problem_solution_l219_219779

noncomputable def f : ℝ → ℝ := 
sorry

theorem f_odd (x : ℝ) : f (-x) = -f x := 
sorry

theorem f_periodic (x : ℝ) : f (x + 4) = f x := 
sorry

theorem f_def_on_interval (x : ℝ) (h : -2 < x ∧ x < 0) : f x = 2 ^ x :=
sorry

theorem problem_solution : f 2015 - f 2014 = 1 / 2 :=
sorry

end f_odd_f_periodic_f_def_on_interval_problem_solution_l219_219779


namespace divisor_is_seventeen_l219_219261

theorem divisor_is_seventeen (D x : ℕ) (h1 : D = 7 * x) (h2 : D + x = 136) : x = 17 :=
by
  sorry

end divisor_is_seventeen_l219_219261


namespace cooking_time_remaining_l219_219792

def time_to_cook_remaining (n_total n_cooked t_per : ℕ) : ℕ := (n_total - n_cooked) * t_per

theorem cooking_time_remaining :
  ∀ (n_total n_cooked t_per : ℕ), n_total = 13 → n_cooked = 5 → t_per = 6 → time_to_cook_remaining n_total n_cooked t_per = 48 :=
by
  intros n_total n_cooked t_per h1 h2 h3
  rw [h1, h2, h3]
  exact rfl

end cooking_time_remaining_l219_219792


namespace solve_equation_l219_219098

theorem solve_equation (x : ℝ) : 
  (x - 1) / 2 - (2 * x + 3) / 3 = 1 ↔ 3 * (x - 1) - 2 * (2 * x + 3) = 6 := 
sorry

end solve_equation_l219_219098


namespace right_pyramid_sum_edges_l219_219955

theorem right_pyramid_sum_edges (a h : ℝ) (base_side slant_height : ℝ) :
  base_side = 12 ∧ slant_height = 15 ∧ ∀ x : ℝ, a = 117 :=
by
  sorry

end right_pyramid_sum_edges_l219_219955


namespace meter_to_leap_l219_219684

theorem meter_to_leap
  (strides leaps bounds meters : ℝ)
  (h1 : 3 * strides = 4 * leaps)
  (h2 : 5 * bounds = 7 * strides)
  (h3 : 2 * bounds = 9 * meters) :
  1 * meters = (56 / 135) * leaps :=
by
  sorry

end meter_to_leap_l219_219684


namespace person_time_to_walk_without_walkway_l219_219716

def time_to_walk_without_walkway 
  (walkway_length : ℝ) 
  (time_with_walkway : ℝ) 
  (time_against_walkway : ℝ) 
  (correct_time : ℝ) : Prop :=
  ∃ (vp vw : ℝ), 
    ((vp + vw) * time_with_walkway = walkway_length) ∧ 
    ((vp - vw) * time_against_walkway = walkway_length) ∧ 
     correct_time = walkway_length / vp

theorem person_time_to_walk_without_walkway : 
  time_to_walk_without_walkway 120 40 160 64 :=
sorry

end person_time_to_walk_without_walkway_l219_219716


namespace unique_real_solution_bound_l219_219276

theorem unique_real_solution_bound (b : ℝ) :
  (∀ x : ℝ, x^3 - b*x^2 - 4*b*x + b^2 - 4 = 0 → ∃! y : ℝ, y = x) → b < 1 :=
by
  sorry

end unique_real_solution_bound_l219_219276


namespace paintings_after_30_days_l219_219562

theorem paintings_after_30_days (paintings_per_day : ℕ) (initial_paintings : ℕ) (days : ℕ)
    (h1 : paintings_per_day = 2)
    (h2 : initial_paintings = 20)
    (h3 : days = 30) :
    initial_paintings + paintings_per_day * days = 80 := by
  sorry

end paintings_after_30_days_l219_219562


namespace find_S20_l219_219250

theorem find_S20 (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : ∀ n, n ≥ 2 → S n = 1 + 2 * a n)
  (h2 : a 1 = 2) : 
  S 20 = 2^19 + 1 := 
sorry

end find_S20_l219_219250


namespace Harkamal_total_payment_l219_219044

theorem Harkamal_total_payment :
  let cost_grapes := 10 * 70
  let cost_mangoes := 9 * 55
  let cost_apples := 12 * 80
  let cost_papayas := 7 * 45
  let cost_oranges := 15 * 30
  let cost_bananas := 5 * 25
  cost_grapes + cost_mangoes + cost_apples + cost_papayas + cost_oranges + cost_bananas = 3045 := by
  sorry

end Harkamal_total_payment_l219_219044


namespace tangent_line_equation_l219_219069

theorem tangent_line_equation 
  (A : ℝ × ℝ)
  (hA : A = (-1, 2))
  (parabola : ℝ → ℝ)
  (h_parabola : ∀ x, parabola x = 2 * x ^ 2) 
  (tangent : ℝ × ℝ → ℝ)
  (h_tangent : ∀ P, tangent P = -4 * P.1 + 4 * (-1) + 2) : 
  tangent A = 4 * (-1) + 2 :=
by
  sorry

end tangent_line_equation_l219_219069


namespace loan_duration_in_years_l219_219383

-- Define the conditions as constants
def carPrice : ℝ := 20000
def downPayment : ℝ := 5000
def monthlyPayment : ℝ := 250

-- Define the goal
theorem loan_duration_in_years :
  (carPrice - downPayment) / monthlyPayment / 12 = 5 := 
sorry

end loan_duration_in_years_l219_219383


namespace set_expression_l219_219125

def is_natural_number (x : ℚ) : Prop :=
  ∃ n : ℕ, x = n

theorem set_expression :
  {x : ℕ | is_natural_number (6 / (5 - x) : ℚ)} = {2, 3, 4} :=
sorry

end set_expression_l219_219125


namespace min_expression_n_12_l219_219438

theorem min_expression_n_12 : ∃ n : ℕ, (n > 0) ∧ (∀ m : ℕ, m > 0 → (n = 12 → (n / 3 + 50 / n ≤ 
                        m / 3 + 50 / m))) :=
by
  sorry

end min_expression_n_12_l219_219438


namespace find_vector_v_l219_219544

def vector3 := ℝ × ℝ × ℝ

def cross_product (u v : vector3) : vector3 :=
  (u.2.1 * v.2.2 - u.2.2 * v.2.1,
   u.2.2 * v.1  - u.1   * v.2.2,
   u.1   * v.2.1 - u.2.1 * v.1)

def a : vector3 := (1, 2, 1)
def b : vector3 := (2, 0, -1)
def v : vector3 := (3, 2, 0)
def b_cross_a : vector3 := (2, 3, 4)
def a_cross_b : vector3 := (-2, 3, -4)

theorem find_vector_v :
  cross_product v a = b_cross_a ∧ cross_product v b = a_cross_b :=
sorry

end find_vector_v_l219_219544


namespace linda_savings_fraction_l219_219443

theorem linda_savings_fraction (savings tv_cost : ℝ) (h1 : savings = 960) (h2 : tv_cost = 240) : (savings - tv_cost) / savings = 3 / 4 :=
by
  intros
  sorry

end linda_savings_fraction_l219_219443


namespace cookies_left_after_three_days_l219_219203

theorem cookies_left_after_three_days
  (initial_cookies : ℕ)
  (first_day_fraction_eaten : ℚ)
  (second_day_fraction_eaten : ℚ)
  (initial_value : initial_cookies = 64)
  (first_day_fraction : first_day_fraction_eaten = 3/4)
  (second_day_fraction : second_day_fraction_eaten = 1/2) :
  initial_cookies - (first_day_fraction_eaten * 64) - (second_day_fraction_eaten * ((1 - first_day_fraction_eaten) * 64)) = 8 :=
by
  sorry

end cookies_left_after_three_days_l219_219203


namespace tangent_line_at_1_l219_219367

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Define the derivative f'
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

-- Define the point of tangency
def point_of_tangency : ℝ × ℝ := (1, f 1)

-- Define the slope of the tangent line at x=1
def slope_at_1 : ℝ := f' 1

-- Define the tangent line equation at x=1
def tangent_line (x y : ℝ) : Prop := 2 * x - y + 1 = 0

-- Theorem that the tangent line to f at x=1 is 2x - y + 1 = 0
theorem tangent_line_at_1 :
  tangent_line 1 (f 1) :=
by
  sorry

end tangent_line_at_1_l219_219367


namespace am_gm_inequality_even_sum_l219_219073

theorem am_gm_inequality_even_sum (a b : ℤ) (ha : 0 < a) (hb : 0 < b) (h_even : (a + b) % 2 = 0) :
  (a + b : ℚ) / 2 ≥ Real.sqrt (a * b) :=
sorry

end am_gm_inequality_even_sum_l219_219073


namespace ribbon_cost_comparison_l219_219235

theorem ribbon_cost_comparison 
  (A : Type)
  (yellow_ribbon_cost blue_ribbon_cost : ℕ)
  (h1 : yellow_ribbon_cost = 24)
  (h2 : blue_ribbon_cost = 36) :
  (∃ n : ℕ, n > 0 ∧ yellow_ribbon_cost / n < blue_ribbon_cost / n) ∨
  (∃ n : ℕ, n > 0 ∧ yellow_ribbon_cost / n > blue_ribbon_cost / n) ∨
  (∃ n : ℕ, n > 0 ∧ yellow_ribbon_cost / n = blue_ribbon_cost / n) :=
sorry

end ribbon_cost_comparison_l219_219235


namespace value_of_a_plus_b_l219_219655

open Set Real

def setA : Set ℝ := {x | x^2 - 2 * x - 3 > 0}
def setB (a b : ℝ) : Set ℝ := {x | x^2 + a * x + b ≤ 0}
def universalSet : Set ℝ := univ

theorem value_of_a_plus_b (a b : ℝ) :
  (setA ∪ setB a b = universalSet) ∧ (setA ∩ setB a b = {x : ℝ | 3 < x ∧ x ≤ 4}) → a + b = -7 :=
by
  sorry

end value_of_a_plus_b_l219_219655


namespace neg_of_p_l219_219199

variable (x : ℝ)

def p : Prop := ∀ x ≥ 0, 2^x = 3

theorem neg_of_p : ¬p ↔ ∃ x ≥ 0, 2^x ≠ 3 :=
by
  sorry

end neg_of_p_l219_219199


namespace Thomas_speed_greater_than_Jeremiah_l219_219354

-- Define constants
def Thomas_passes_kilometers_per_hour := 5
def Jeremiah_passes_kilometers_per_hour := 6

-- Define speeds (in meters per hour)
def Thomas_speed := Thomas_passes_kilometers_per_hour * 1000
def Jeremiah_speed := Jeremiah_passes_kilometers_per_hour * 1000

-- Define hypothetical additional distances
def Thomas_hypothetical_additional_distance := 600 * 2
def Jeremiah_hypothetical_additional_distance := 50 * 2

-- Define effective distances traveled
def Thomas_effective_distance := Thomas_speed + Thomas_hypothetical_additional_distance
def Jeremiah_effective_distance := Jeremiah_speed + Jeremiah_hypothetical_additional_distance

-- Theorem to prove
theorem Thomas_speed_greater_than_Jeremiah : Thomas_effective_distance > Jeremiah_effective_distance := by
  -- Placeholder for the proof
  sorry

end Thomas_speed_greater_than_Jeremiah_l219_219354


namespace problem_f_2016_eq_l219_219116

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.sin x + b * x^3 + 4
noncomputable def f' (a b : ℝ) (x : ℝ) : ℝ := a * Real.cos x + 3 * b * x^2

theorem problem_f_2016_eq :
  ∀ (a b : ℝ),
  f a b 2016 + f a b (-2016) + f' a b 2017 - f' a b (-2017) = 8 + 2 * b * 2016^3 :=
by
  intro a b
  sorry

end problem_f_2016_eq_l219_219116


namespace symmetric_parabola_l219_219421

def parabola1 (x : ℝ) : ℝ := (x - 2)^2 + 3
def parabola2 (x : ℝ) : ℝ := -(x + 2)^2 - 3

theorem symmetric_parabola : ∀ x y : ℝ,
  y = parabola1 x ↔ 
  (-y) = parabola2 (-x) ∧ y = -(x + 2)^2 - 3 :=
sorry

end symmetric_parabola_l219_219421


namespace points_below_line_l219_219604

theorem points_below_line (d q x1 x2 y1 y2 : ℝ) 
  (h1 : 2 = 1 + 3 * d)
  (h2 : x1 = 1 + d)
  (h3 : x2 = x1 + d)
  (h4 : 2 = q ^ 3)
  (h5 : y1 = q)
  (h6 : y2 = q ^ 2) :
  x1 > y1 ∧ x2 > y2 :=
by {
  sorry
}

end points_below_line_l219_219604


namespace number_of_trousers_given_l219_219669

-- Define the conditions
def shirts_given : Nat := 589
def total_clothing_given : Nat := 934

-- Define the expected answer
def expected_trousers_given : Nat := 345

-- The theorem statement to prove the number of trousers given
theorem number_of_trousers_given : total_clothing_given - shirts_given = expected_trousers_given :=
by
  sorry

end number_of_trousers_given_l219_219669


namespace tangent_lines_l219_219890

noncomputable def curve1 (x : ℝ) : ℝ := 2 * x ^ 2 - 5
noncomputable def curve2 (x : ℝ) : ℝ := x ^ 2 - 3 * x + 5

theorem tangent_lines :
  (∃ (m₁ b₁ m₂ b₂ : ℝ), 
    (∀ x y, y = -20 * x - 55 ∨ y = -13 * x - 20 ∨ y = 8 * x - 13 ∨ y = x + 1) ∧ 
    (
      (m₁ = 4 * 2 ∧ b₁ = 3) ∨ 
      (m₁ = 2 * -5 - 3 ∧ b₁ = 45) ∨
      (m₂ = 4 * -5 ∧ b₂ = 45) ∨
      (m₂ = 2 * 2 - 3 ∧ b₂ = 3)
    )) :=
sorry

end tangent_lines_l219_219890


namespace inequality_and_equality_hold_l219_219399

theorem inequality_and_equality_hold (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a * b)) ∧ (1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a * b) ↔ a = b) :=
sorry

end inequality_and_equality_hold_l219_219399


namespace sum_mod_nine_l219_219237

def a : ℕ := 1234
def b : ℕ := 1235
def c : ℕ := 1236
def d : ℕ := 1237
def e : ℕ := 1238
def modulus : ℕ := 9

theorem sum_mod_nine : (a + b + c + d + e) % modulus = 6 :=
by
  sorry

end sum_mod_nine_l219_219237


namespace total_amount_correct_l219_219338

def num_2won_bills : ℕ := 8
def value_2won_bills : ℕ := 2
def num_1won_bills : ℕ := 2
def value_1won_bills : ℕ := 1

theorem total_amount_correct :
  (num_2won_bills * value_2won_bills) + (num_1won_bills * value_1won_bills) = 18 :=
by
  sorry

end total_amount_correct_l219_219338


namespace inverse_function_of_f_l219_219676

noncomputable def f (x : ℝ) : ℝ := (x - 1) ^ 2

noncomputable def f_inv (y : ℝ) : ℝ := 1 - Real.sqrt y

theorem inverse_function_of_f :
  ∀ x, x ≤ 1 → f_inv (f x) = x ∧ ∀ y, 0 ≤ y → f (f_inv y) = y :=
by
  intros
  sorry

end inverse_function_of_f_l219_219676


namespace factorize_expression_l219_219179

theorem factorize_expression : (x^2 + 9)^2 - 36*x^2 = (x + 3)^2 * (x - 3)^2 := 
by 
  sorry

end factorize_expression_l219_219179


namespace distance_to_focus_F2_l219_219633

noncomputable def ellipse_foci_distance
  (x y : ℝ)
  (a b : ℝ) 
  (h_ellipse : (x^2 / a^2) + (y^2 / b^2) = 1) 
  (a2 : a^2 = 9) 
  (b2 : b^2 = 2) 
  (F1 P : ℝ) 
  (h_P_on_ellipse : F1 = 3) 
  (h_PF1 : F1 = 4) 
: ℝ :=
  2

-- theorem to prove the problem statement
theorem distance_to_focus_F2
  (x y : ℝ)
  (a b : ℝ)
  (h_ellipse : (x^2 / a^2) + (y^2 / b^2) = 1)
  (a2 : a^2 = 9)
  (b2 : b^2 = 2)
  (F1 P : ℝ)
  (h_P_on_ellipse : F1 = 3)
  (h_PF1 : F1 = 4)
: F2 = 2 :=
by
  sorry

end distance_to_focus_F2_l219_219633


namespace sequence_an_expression_l219_219985

theorem sequence_an_expression (a : ℕ → ℕ) : 
  a 1 = 1 ∧ (∀ n : ℕ, n ≥ 1 → (a n / n - a (n - 1) / (n - 1)) = 2) → (∀ n : ℕ, a n = 2 * n * n - n) :=
by
  sorry

end sequence_an_expression_l219_219985


namespace odd_function_f_l219_219328

noncomputable def f : ℝ → ℝ
| x => if hx : x ≥ 0 then x * (1 - x) else x * (1 + x)

theorem odd_function_f {f : ℝ → ℝ}
  (h_odd : ∀ x : ℝ, f (-x) = - f x)
  (h_pos : ∀ x : ℝ, 0 ≤ x → f x = x * (1 - x)) :
  ∀ x : ℝ, x ≤ 0 → f x = x * (1 + x) := by
  intro x hx
  sorry

end odd_function_f_l219_219328


namespace arithmetic_value_l219_219848

theorem arithmetic_value : (8 * 4) + 3 = 35 := by
  sorry

end arithmetic_value_l219_219848


namespace find_n_squares_l219_219629

theorem find_n_squares (n : ℤ) : 
  (∃ a : ℤ, n^2 + 6 * n + 24 = a^2) ↔ n = 4 ∨ n = -2 ∨ n = -4 ∨ n = -10 :=
by
  sorry

end find_n_squares_l219_219629


namespace energy_loss_per_bounce_l219_219988

theorem energy_loss_per_bounce
  (h : ℝ) (t : ℝ) (g : ℝ) (y : ℝ)
  (h_conds : h = 0.2)
  (t_conds : t = 18)
  (g_conds : g = 10)
  (model : t = Real.sqrt (2 * h / g) + 2 * (Real.sqrt (2 * h * y / g)) / (1 - Real.sqrt y)) :
  1 - y = 0.36 :=
by
  sorry

end energy_loss_per_bounce_l219_219988


namespace verify_graphical_method_l219_219929

variable {R : Type} [LinearOrderedField R]

/-- Statement of the mentioned conditions -/
def poly (a b c d x : R) : R := a * x^3 + b * x^2 + c * x + d

/-- The main theorem stating the graphical method validity -/
theorem verify_graphical_method (a b c d x0 EJ : R) (h : 0 < a) (h1 : 0 < b) (h2 : 0 < c) (h3 : 0 < d) (h4 : 0 < x0) (h5 : x0 < 1)
: EJ = poly a b c d x0 := by sorry

end verify_graphical_method_l219_219929


namespace tank_capacity_l219_219797

theorem tank_capacity (x : ℝ) (h : 0.50 * x = 75) : x = 150 :=
by sorry

end tank_capacity_l219_219797


namespace division_and_multiplication_result_l219_219918

theorem division_and_multiplication_result :
  let num : ℝ := 6.5
  let divisor : ℝ := 6
  let multiplier : ℝ := 12
  num / divisor * multiplier = 13 :=
by
  sorry

end division_and_multiplication_result_l219_219918


namespace find_a_l219_219238

theorem find_a :
  let p1 := (⟨-3, 7⟩ : ℝ × ℝ)
  let p2 := (⟨2, -1⟩ : ℝ × ℝ)
  let direction := (5, -8)
  let target_direction := (a, -2)
  a = (direction.1 * -2) / (direction.2) := by
  sorry

end find_a_l219_219238


namespace expr_simplification_l219_219569

noncomputable def simplify_sqrt_expr : ℝ :=
  Real.sqrt 3 - Real.sqrt 12 + Real.sqrt 27

theorem expr_simplification : simplify_sqrt_expr = 2 * Real.sqrt 3 := by
  sorry

end expr_simplification_l219_219569


namespace madeline_biked_more_l219_219763

def madeline_speed : ℕ := 12
def madeline_time : ℕ := 3
def max_speed : ℕ := 15
def max_time : ℕ := 2

theorem madeline_biked_more : (madeline_speed * madeline_time) - (max_speed * max_time) = 6 := 
by 
  sorry

end madeline_biked_more_l219_219763


namespace area_ADC_proof_l219_219648

-- Definitions for the given conditions and question
variables (BD DC : ℝ) (ABD_area ADC_area : ℝ)

-- Conditions
def ratio_condition := BD / DC = 3 / 2
def ABD_area_condition := ABD_area = 30

-- Question rewritten as proof problem
theorem area_ADC_proof (h1 : ratio_condition BD DC) (h2 : ABD_area_condition ABD_area) :
  ADC_area = 20 :=
sorry

end area_ADC_proof_l219_219648


namespace problem1_problem2_l219_219123

-- Problem 1 Statement
theorem problem1 : (3 * Real.sqrt 48 - 2 * Real.sqrt 27) / Real.sqrt 3 = 6 :=
by sorry

-- Problem 2 Statement
theorem problem2 : 
  (Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) - Real.sqrt ((-3)^2) + 1 / (2 - Real.sqrt 5) = -3 - Real.sqrt 5 :=
by sorry

end problem1_problem2_l219_219123


namespace number_of_small_jars_l219_219205

theorem number_of_small_jars (S L : ℕ) (h1 : S + L = 100) (h2 : 3 * S + 5 * L = 376) : S = 62 := 
sorry

end number_of_small_jars_l219_219205


namespace trig_expression_value_l219_219637

theorem trig_expression_value : 
  (2 * (Real.sin (25 * Real.pi / 180))^2 - 1) / 
  (Real.sin (20 * Real.pi / 180) * Real.cos (20 * Real.pi / 180)) = -2 := 
by
  -- Proof goes here
  sorry

end trig_expression_value_l219_219637


namespace solve_x2_y2_eq_3z2_in_integers_l219_219248

theorem solve_x2_y2_eq_3z2_in_integers (x y z : ℤ) : x^2 + y^2 = 3 * z^2 → x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end solve_x2_y2_eq_3z2_in_integers_l219_219248


namespace cannot_cut_out_rect_l219_219049

noncomputable def square_area : ℝ := 400
noncomputable def rect_area : ℝ := 300
noncomputable def length_to_width_ratio : ℝ × ℝ := (3, 2)

theorem cannot_cut_out_rect (h1: square_area = 400) (h2: rect_area = 300) (h3: length_to_width_ratio = (3, 2)) : 
  false := sorry

end cannot_cut_out_rect_l219_219049


namespace proposition_a_sufficient_not_necessary_negation_of_proposition_b_incorrect_proposition_c_not_necessary_proposition_d_necessary_not_sufficient_final_answer_correct_l219_219691

theorem proposition_a_sufficient_not_necessary (a : ℝ) : (a > 1 → 1 / a < 1) ∧ (1 / a < 1 → a > 1 ∨ a < 1) :=
sorry

theorem negation_of_proposition_b_incorrect (x : ℝ) : ¬(∀ x < 1, x^2 < 1) ↔ ∃ x < 1, x^2 ≥ 1 :=
sorry

theorem proposition_c_not_necessary (x y : ℝ) : (x ≥ 2 ∧ y ≥ 2 → x^2 + y^2 ≥ 8) ∧ (x^2 + y^2 ≥ 4 → ¬(x ≥ 2 ∧ y ≥ 2)) :=
sorry

theorem proposition_d_necessary_not_sufficient (a b : ℝ) : (a ≠ 0 → ab ≠ 0) ∧ (ab ≠ 0 → a ≠ 0 ∨ b ≠ 0) :=
sorry

theorem final_answer_correct :
  let proposition_A := (∃ (a : ℝ), a > 1 ∧ 1 / a < 1 ∧ (1 / a < 1 → a > 1 ∨ a < 1))
  let proposition_B := (¬(∀ (x : ℝ), x < 1 → x^2 < 1) ↔ ∃ (x : ℝ), x < 1 ∧ x^2 ≥ 1)
  let proposition_C := (∃ (x y : ℝ), (x ≥ 2 ∧ y ≥ 2 → x^2 + y^2 ≥ 8) ∧ (x^2 + y^2 ≥ 4 → ¬(x ≥ 2 ∧ y ≥ 2)))
  let proposition_D := (∃ (a b : ℝ), a ≠ 0 ∧ ab ≠ 0 ∧ (ab ≠ 0 → a ≠ 0 ∨ b ≠ 0))
  proposition_A ∧ proposition_D
:= 
sorry

end proposition_a_sufficient_not_necessary_negation_of_proposition_b_incorrect_proposition_c_not_necessary_proposition_d_necessary_not_sufficient_final_answer_correct_l219_219691


namespace hyperbola_focus_to_asymptote_distance_l219_219345

theorem hyperbola_focus_to_asymptote_distance :
  ∀ (x y : ℝ), (x ^ 2 - y ^ 2 = 1) →
  ∃ c : ℝ, (c = 1) :=
by
  sorry

end hyperbola_focus_to_asymptote_distance_l219_219345


namespace shirts_needed_for_vacation_l219_219479

def vacation_days := 7
def same_shirt_days := 2
def different_shirts_per_day := 2
def different_shirt_days := vacation_days - same_shirt_days

theorem shirts_needed_for_vacation : different_shirt_days * different_shirts_per_day + same_shirt_days = 11 := by
  sorry

end shirts_needed_for_vacation_l219_219479


namespace correct_operation_l219_219329

theorem correct_operation (a b : ℝ) : (a^2 * b)^2 = a^4 * b^2 := by
  sorry

end correct_operation_l219_219329


namespace kamal_marks_in_english_l219_219641

theorem kamal_marks_in_english :
  ∀ (E Math Physics Chemistry Biology Average : ℕ), 
    Math = 65 → 
    Physics = 82 → 
    Chemistry = 67 → 
    Biology = 85 → 
    Average = 79 → 
    (Math + Physics + Chemistry + Biology + E) / 5 = Average → 
    E = 96 :=
by
  intros E Math Physics Chemistry Biology Average
  intros hMath hPhysics hChemistry hBiology hAverage hTotal
  sorry

end kamal_marks_in_english_l219_219641


namespace food_left_after_bbqs_l219_219948

noncomputable def mushrooms_bought : ℕ := 15
noncomputable def chicken_bought : ℕ := 20
noncomputable def beef_bought : ℕ := 10

noncomputable def mushrooms_consumed : ℕ := 5 * 3
noncomputable def chicken_consumed : ℕ := 4 * 2
noncomputable def beef_consumed : ℕ := 2 * 1

noncomputable def mushrooms_left : ℕ := mushrooms_bought - mushrooms_consumed
noncomputable def chicken_left : ℕ := chicken_bought - chicken_consumed
noncomputable def beef_left : ℕ := beef_bought - beef_consumed

noncomputable def total_food_left : ℕ := mushrooms_left + chicken_left + beef_left

theorem food_left_after_bbqs : total_food_left = 20 :=
  by
    unfold total_food_left mushrooms_left chicken_left beef_left
    unfold mushrooms_consumed chicken_consumed beef_consumed
    unfold mushrooms_bought chicken_bought beef_bought
    sorry

end food_left_after_bbqs_l219_219948


namespace bathroom_area_l219_219753

-- Definitions based on conditions
def totalHouseArea : ℝ := 1110
def numBedrooms : ℕ := 4
def bedroomArea : ℝ := 11 * 11
def kitchenArea : ℝ := 265
def numBathrooms : ℕ := 2

-- Mathematically equivalent proof problem
theorem bathroom_area :
  let livingArea := kitchenArea  -- living area is equal to kitchen area
  let totalRoomArea := numBedrooms * bedroomArea + kitchenArea + livingArea
  let remainingArea := totalHouseArea - totalRoomArea
  let bathroomArea := remainingArea / numBathrooms
  bathroomArea = 48 :=
by
  repeat { sorry }

end bathroom_area_l219_219753


namespace arithmetic_sequence_solution_l219_219757

variable {a : ℕ → ℤ}  -- assuming our sequence is integer-valued for simplicity

-- a is an arithmetic sequence if there exists a common difference d such that 
-- ∀ n, a_{n+1} = a_n + d
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- sum of the terms from a₁ to a₁₀₁₇ is equal to zero
def sum_condition (a : ℕ → ℤ) : Prop :=
  (Finset.range 2017).sum a = 0

theorem arithmetic_sequence_solution (a : ℕ → ℤ) (h_arith : is_arithmetic_sequence a) (h_sum : sum_condition a) :
  a 3 + a 2013 = 0 :=
sorry

end arithmetic_sequence_solution_l219_219757


namespace integers_abs_le_3_l219_219498

theorem integers_abs_le_3 :
  {x : ℤ | |x| ≤ 3} = { -3, -2, -1, 0, 1, 2, 3 } :=
by
  sorry

end integers_abs_le_3_l219_219498


namespace sum_of_sequence_l219_219079

noncomputable def sequence_sum (n : ℕ) : ℤ :=
  6 * 2^n - (n + 6)

theorem sum_of_sequence (a S : ℕ → ℤ) (n : ℕ) :
  a 1 = 5 →
  (∀ n : ℕ, 1 ≤ n → S (n + 1) = 2 * S n + n + 5) →
  S n = sequence_sum n :=
by sorry

end sum_of_sequence_l219_219079


namespace isosceles_triangle_base_vertex_trajectory_l219_219493

theorem isosceles_triangle_base_vertex_trajectory :
  ∀ (x y : ℝ), 
  (∀ (A : ℝ × ℝ) (B : ℝ × ℝ), 
    A = (2, 4) ∧ B = (2, 8) ∧ 
    ((x-2)^2 + (y-4)^2 = 16)) → 
  ((x ≠ 2) ∧ (y ≠ 8) → (x-2)^2 + (y-4)^2 = 16) :=
sorry

end isosceles_triangle_base_vertex_trajectory_l219_219493


namespace balloons_problem_l219_219272

theorem balloons_problem :
  ∃ (b y : ℕ), y = 3414 ∧ b + y = 8590 ∧ b - y = 1762 := 
by
  sorry

end balloons_problem_l219_219272


namespace gcd_10010_15015_l219_219914

theorem gcd_10010_15015 :
  Int.gcd 10010 15015 = 5005 :=
by 
  sorry

end gcd_10010_15015_l219_219914


namespace smallest_positive_x_l219_219770

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem smallest_positive_x : ∃ x : ℕ, x > 0 ∧ is_palindrome (x + 6789) ∧ x = 218 := by
  sorry

end smallest_positive_x_l219_219770


namespace dynaco_shares_sold_l219_219458

-- Define the conditions
def MicrotronPrice : ℝ := 36
def DynacoPrice : ℝ := 44
def TotalShares : ℕ := 300
def AvgPrice : ℝ := 40
def TotalValue : ℝ := TotalShares * AvgPrice

-- Define unknown variables
variables (M D : ℕ)

-- Express conditions in Lean
def total_shares_eq : Prop := M + D = TotalShares
def total_value_eq : Prop := MicrotronPrice * M + DynacoPrice * D = TotalValue

-- Define the problem statement
theorem dynaco_shares_sold : ∃ D : ℕ, 
  (∃ M : ℕ, total_shares_eq M D ∧ total_value_eq M D) ∧ D = 150 :=
by
  sorry

end dynaco_shares_sold_l219_219458


namespace total_loaves_served_l219_219967

-- Definitions based on the conditions provided
def wheat_bread_loaf : ℝ := 0.2
def white_bread_loaf : ℝ := 0.4

-- Statement that needs to be proven
theorem total_loaves_served : wheat_bread_loaf + white_bread_loaf = 0.6 := 
by
  sorry

end total_loaves_served_l219_219967


namespace shorter_leg_of_right_triangle_l219_219265

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : c = 65) (h₃ : a ≤ b) : a = 25 :=
sorry

end shorter_leg_of_right_triangle_l219_219265


namespace caffeine_per_energy_drink_l219_219884

variable (amount_of_caffeine_per_drink : ℕ)

def maximum_safe_caffeine_per_day := 500
def drinks_per_day := 4
def additional_safe_amount := 20

theorem caffeine_per_energy_drink :
  4 * amount_of_caffeine_per_drink + additional_safe_amount = maximum_safe_caffeine_per_day →
  amount_of_caffeine_per_drink = 120 :=
by
  sorry

end caffeine_per_energy_drink_l219_219884


namespace tan_product_l219_219307

theorem tan_product :
  (Real.tan (Real.pi / 8)) * (Real.tan (3 * Real.pi / 8)) * (Real.tan (5 * Real.pi / 8)) = 1 :=
sorry

end tan_product_l219_219307


namespace negation_of_proposition_l219_219619

theorem negation_of_proposition :
  (¬ ∃ m : ℝ, 1 / (m^2 + m - 6) > 0) ↔ (∀ m : ℝ, (1 / (m^2 + m - 6) < 0) ∨ (m^2 + m - 6 = 0)) :=
by
  sorry

end negation_of_proposition_l219_219619


namespace total_volume_correct_l219_219531

-- Definitions based on the conditions
def box_length := 30 -- in cm
def box_width := 1 -- in cm
def box_height := 1 -- in cm
def horizontal_rows := 7
def vertical_rows := 5
def floors := 3

-- The volume of a single box
def box_volume : Int := box_length * box_width * box_height

-- The total number of boxes is the product of rows and floors
def total_boxes : Int := horizontal_rows * vertical_rows * floors

-- The total volume of all the boxes
def total_volume : Int := box_volume * total_boxes

-- The statement to prove
theorem total_volume_correct : total_volume = 3150 := 
by 
  simp [box_volume, total_boxes, total_volume]
  sorry

end total_volume_correct_l219_219531


namespace no_intersection_curves_l219_219829

theorem no_intersection_curves (k : ℕ) (hn : k > 0) 
  (h_intersection : ∀ x y : ℝ, ¬(x^2 + y^2 = k^2 ∧ x * y = k)) : 
  k = 1 := 
sorry

end no_intersection_curves_l219_219829


namespace circumference_of_circle_l219_219597

theorem circumference_of_circle (R : ℝ) : 
  (C = 2 * Real.pi * R) :=
sorry

end circumference_of_circle_l219_219597


namespace general_term_formula_sum_of_sequence_l219_219863

-- Define the arithmetic sequence {a_n}
def a (n : ℕ) : ℤ := n - 1

-- Conditions: a_5 = 4, a_3 + a_8 = 9
def cond1 : Prop := a 5 = 4
def cond2 : Prop := a 3 + a 8 = 9

theorem general_term_formula (h1 : cond1) (h2 : cond2) : ∀ n : ℕ, a n = n - 1 :=
by
  -- Place holder for proof
  sorry

-- Define the sequence {b_n}
def b (n : ℕ) : ℤ := 2 * a n - 1

-- Sum of the first n terms of b_n
def S (n : ℕ) : ℤ := n * (n - 2)

theorem sum_of_sequence (h1 : cond1) (h2 : cond2) : ∀ n : ℕ, (Finset.range (n + 1)).sum b = S n :=
by
  -- Place holder for proof
  sorry

end general_term_formula_sum_of_sequence_l219_219863


namespace smallest_of_seven_even_numbers_l219_219811

theorem smallest_of_seven_even_numbers (a b c d e f g : ℕ) 
  (h1 : a % 2 = 0) 
  (h2 : b = a + 2) 
  (h3 : c = a + 4) 
  (h4 : d = a + 6) 
  (h5 : e = a + 8) 
  (h6 : f = a + 10) 
  (h7 : g = a + 12) 
  (h_sum : a + b + c + d + e + f + g = 700) : 
  a = 94 :=
by sorry

end smallest_of_seven_even_numbers_l219_219811


namespace circle_equation_l219_219158

theorem circle_equation
  (a b r : ℝ) 
  (h1 : a^2 + b^2 = r^2) 
  (h2 : (a - 2)^2 + b^2 = r^2) 
  (h3 : b / (a - 2) = 1) : 
  (x - 1)^2 + (y + 1)^2 = 2 := 
by
  sorry

end circle_equation_l219_219158


namespace product_of_consecutive_integers_sqrt_73_l219_219987

theorem product_of_consecutive_integers_sqrt_73 : 
  ∃ (m n : ℕ), (m < n) ∧ ∃ (j k : ℕ), (j = 8) ∧ (k = 9) ∧ (m = j) ∧ (n = k) ∧ (m * n = 72) := by
  sorry

end product_of_consecutive_integers_sqrt_73_l219_219987


namespace find_ordered_pair_l219_219627

theorem find_ordered_pair (x y : ℝ) :
  (2 * x + 3 * y = (6 - x) + (6 - 3 * y)) ∧ (x - 2 * y = (x - 2) - (y + 2)) ↔ (x = -4) ∧ (y = 4) := by
  sorry

end find_ordered_pair_l219_219627


namespace cost_per_liter_of_gas_today_l219_219032

-- Definition of the conditions
def oil_price_rollback : ℝ := 0.4
def liters_today : ℝ := 10
def liters_friday : ℝ := 25
def total_liters := liters_today + liters_friday
def total_cost : ℝ := 39

-- The theorem to prove
theorem cost_per_liter_of_gas_today (C : ℝ) :
  (liters_today * C) + (liters_friday * (C - oil_price_rollback)) = total_cost →
  C = 1.4 := 
by 
  sorry

end cost_per_liter_of_gas_today_l219_219032


namespace find_a_l219_219758

theorem find_a (a b c : ℂ) (ha : a.im = 0)
  (h1 : a + b + c = 5)
  (h2 : a * b + b * c + c * a = 8)
  (h3 : a * b * c = 4) :
  a = 1 ∨ a = 2 :=
sorry

end find_a_l219_219758


namespace minimum_value_of_expression_l219_219275

variable (a b c d : ℝ)

-- The given conditions:
def cond1 : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0
def cond2 : Prop := a^2 + b^2 = 4
def cond3 : Prop := c * d = 1

-- The minimum value:
def expression_value : ℝ := (a^2 * c^2 + b^2 * d^2) * (b^2 * c^2 + a^2 * d^2)

theorem minimum_value_of_expression :
  cond1 a b c d → cond2 a b → cond3 c d → expression_value a b c d ≥ 16 :=
by
  sorry

end minimum_value_of_expression_l219_219275


namespace compare_neg_rational_numbers_l219_219178

theorem compare_neg_rational_numbers :
  - (3 / 2) > - (5 / 3) := 
sorry

end compare_neg_rational_numbers_l219_219178


namespace solve_inequality_l219_219993

theorem solve_inequality (a x : ℝ) (ha : a ≠ 0) :
  (a > 0 → (x^2 - 5 * a * x + 6 * a^2 > 0 ↔ (x < 2 * a ∨ x > 3 * a))) ∧
  (a < 0 → (x^2 - 5 * a * x + 6 * a^2 > 0 ↔ (x < 3 * a ∨ x > 2 * a))) :=
by
  sorry

end solve_inequality_l219_219993


namespace math_problem_l219_219223

theorem math_problem :
  (10^2 + 6^2) / 2 = 68 :=
by
  sorry

end math_problem_l219_219223


namespace colbert_planks_needed_to_buy_l219_219222

variables (total_planks : ℕ) (planks_from_storage : ℕ) 
          (planks_from_parents : ℕ) (planks_from_friends : ℕ)

def planks_needed_from_store := 
  total_planks - (planks_from_storage + planks_from_parents + planks_from_friends)

theorem colbert_planks_needed_to_buy : 
  total_planks = 200 → planks_from_storage = total_planks / 4 → 
  planks_from_parents = total_planks / 2 → planks_from_friends = 20 → 
  planks_needed_from_store total_planks planks_from_storage planks_from_parents planks_from_friends = 30 :=
by
  -- proof steps here
  sorry

end colbert_planks_needed_to_buy_l219_219222


namespace cover_faces_with_strips_l219_219407

theorem cover_faces_with_strips (a b c : ℕ) :
  (∃ f g h : ℕ, a = 5 * f ∨ b = 5 * g ∨ c = 5 * h) ↔
  (∃ u v : ℕ, (a = 5 * u ∧ b = 5 * v) ∨ (a = 5 * u ∧ c = 5 * v) ∨ (b = 5 * u ∧ c = 5 * v)) := 
sorry

end cover_faces_with_strips_l219_219407


namespace arccos_neg_one_eq_pi_l219_219541

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
  sorry

end arccos_neg_one_eq_pi_l219_219541


namespace december_25_is_thursday_l219_219194

theorem december_25_is_thursday (thanksgiving : ℕ) (h : thanksgiving = 27) :
  (∀ n, n % 7 = 0 → n + thanksgiving = 25 → n / 7 = 4) :=
by
  sorry

end december_25_is_thursday_l219_219194


namespace find_line_equation_l219_219457
noncomputable def line_equation (l : ℝ → ℝ → Prop) : Prop :=
    (∀ x y : ℝ, l x y ↔ (2 * x + y - 4 = 0) ∨ (x + y - 3 = 0))

theorem find_line_equation (l : ℝ → ℝ → Prop) :
  (l 1 2) →
  (∃ x1 : ℝ, x1 > 0 ∧ ∃ y1 : ℝ, y1 > 0 ∧ l x1 0 ∧ l 0 y1) ∧
  (∃ x2 : ℝ, x2 < 0 ∧ ∃ y2 : ℝ, y2 > 0 ∧ l x2 0 ∧ l 0 y2) ∧
  (∃ x4 : ℝ, x4 > 0 ∧ ∃ y4 : ℝ, y4 < 0 ∧ l x4 0 ∧ l 0 y4) ∧
  (∃ x_int y_int : ℝ, l x_int 0 ∧ l 0 y_int ∧ x_int + y_int = 6) →
  (line_equation l) :=
by
  sorry

end find_line_equation_l219_219457


namespace Aren_listening_time_l219_219561

/--
Aren’s flight from New York to Hawaii will take 11 hours 20 minutes. He spends 2 hours reading, 
4 hours watching two movies, 30 minutes eating his dinner, some time listening to the radio, 
and 1 hour 10 minutes playing games. He has 3 hours left to take a nap. 
Prove that he spends 40 minutes listening to the radio.
-/
theorem Aren_listening_time 
  (total_flight_time : ℝ := 11 * 60 + 20)
  (reading_time : ℝ := 2 * 60)
  (watching_movies_time : ℝ := 4 * 60)
  (eating_dinner_time : ℝ := 30)
  (playing_games_time : ℝ := 1 * 60 + 10)
  (nap_time : ℝ := 3 * 60) :
  total_flight_time - (reading_time + watching_movies_time + eating_dinner_time + playing_games_time + nap_time) = 40 :=
by sorry

end Aren_listening_time_l219_219561


namespace jenn_money_left_over_l219_219395

-- Definitions based on problem conditions
def num_jars : ℕ := 5
def quarters_per_jar : ℕ := 160
def value_per_quarter : ℚ := 0.25   -- Rational number to represent $0.25
def cost_of_bike : ℚ := 180         -- Rational number to represent $180

-- Statement to prove that Jenn will have $20 left after buying the bike
theorem jenn_money_left_over : 
  (num_jars * quarters_per_jar * value_per_quarter) - cost_of_bike = 20 :=
by
  sorry

end jenn_money_left_over_l219_219395


namespace find_common_difference_l219_219225

noncomputable def common_difference (a₁ d : ℤ) : Prop :=
  let a₂ := a₁ + d
  let a₃ := a₁ + 2 * d
  let S₅ := 5 * a₁ + 10 * d
  a₂ + a₃ = 8 ∧ S₅ = 25 → d = 2

-- Statement of the proof problem
theorem find_common_difference (a₁ d : ℤ) (h : common_difference a₁ d) : d = 2 :=
by sorry

end find_common_difference_l219_219225


namespace find_m_of_equation_has_positive_root_l219_219195

theorem find_m_of_equation_has_positive_root :
  (∃ x : ℝ, 0 < x ∧ (x - 1) / (x - 5) = (m * x) / (10 - 2 * x)) → m = -8 / 5 :=
by
  sorry

end find_m_of_equation_has_positive_root_l219_219195


namespace find_k_l219_219161

theorem find_k (k : ℝ) (h : ∀ x: ℝ, (x = -2) → (1 + k / (x - 1) = 0)) : k = 3 :=
by
  sorry

end find_k_l219_219161


namespace integer_solutions_count_l219_219293

theorem integer_solutions_count :
  ∃ (s : Finset ℤ), s.card = 6 ∧ ∀ x ∈ s, 4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 6 :=
by
  sorry

end integer_solutions_count_l219_219293


namespace integer_pairs_prime_P_l219_219156

theorem integer_pairs_prime_P (P : ℕ) (hP_prime : Prime P) 
  (h_condition : ∃ a b : ℤ, |a + b| + (a - b)^2 = P) : 
  P = 2 ∧ ((∃ a b : ℤ, |a + b| = 2 ∧ a - b = 0) ∨ 
           (∃ a b : ℤ, |a + b| = 1 ∧ (a - b = 1 ∨ a - b = -1))) :=
by
  sorry

end integer_pairs_prime_P_l219_219156


namespace max_horizontal_distance_domino_l219_219256

theorem max_horizontal_distance_domino (n : ℕ) : 
    (n > 0) → ∃ d, d = 2 * Real.log n := 
by {
    sorry
}

end max_horizontal_distance_domino_l219_219256


namespace commission_percentage_l219_219579

theorem commission_percentage (fixed_salary second_base_salary sales_amount earning: ℝ) (commission: ℝ) 
  (h1 : fixed_salary = 1800)
  (h2 : second_base_salary = 1600)
  (h3 : sales_amount = 5000)
  (h4 : earning = 1800) :
  fixed_salary = second_base_salary + (sales_amount * commission) → 
  commission * 100 = 4 :=
by
  -- proof goes here
  sorry

end commission_percentage_l219_219579


namespace calories_in_300g_lemonade_proof_l219_219719

def g_lemon := 150
def g_sugar := 200
def g_water := 450

def c_lemon_per_100g := 30
def c_sugar_per_100g := 400
def c_water := 0

def total_calories :=
  g_lemon * c_lemon_per_100g / 100 +
  g_sugar * c_sugar_per_100g / 100 +
  g_water * c_water

def total_weight := g_lemon + g_sugar + g_water

def caloric_density := total_calories / total_weight

def calories_in_300g_lemonade := 300 * caloric_density

theorem calories_in_300g_lemonade_proof : calories_in_300g_lemonade = 317 := by
  sorry

end calories_in_300g_lemonade_proof_l219_219719


namespace smallest_root_of_unity_l219_219404

open Complex

theorem smallest_root_of_unity (z : ℂ) (h : z^6 - z^3 + 1 = 0) : ∃ k : ℕ, k < 18 ∧ z = exp (2 * pi * I * k / 18) :=
by
  sorry

end smallest_root_of_unity_l219_219404


namespace measurable_length_l219_219941

-- Definitions of lines, rays, and line segments

-- A line is infinitely long with no endpoints.
def isLine (l : Type) : Prop := ∀ x y : l, (x ≠ y)

-- A line segment has two endpoints and a finite length.
def isLineSegment (ls : Type) : Prop := ∃ a b : ls, a ≠ b ∧ ∃ d : ℝ, d > 0

-- A ray has one endpoint and is infinitely long.
def isRay (r : Type) : Prop := ∃ e : r, ∀ x : r, x ≠ e

-- Problem statement
theorem measurable_length (x : Type) : isLineSegment x → (∃ d : ℝ, d > 0) :=
by
  -- Proof is not required
  sorry

end measurable_length_l219_219941


namespace complex_number_in_fourth_quadrant_l219_219979

variable {a b : ℝ}

theorem complex_number_in_fourth_quadrant (a b : ℝ): 
  (a^2 + 1 > 0) ∧ (-b^2 - 1 < 0) → 
  ((a^2 + 1, -b^2 - 1).fst > 0 ∧ (a^2 + 1, -b^2 - 1).snd < 0) :=
by
  intro h
  exact h

#check complex_number_in_fourth_quadrant

end complex_number_in_fourth_quadrant_l219_219979


namespace statement1_statement2_statement3_l219_219887

variable (P_W P_Z : ℝ)

/-- The conditions of the problem: -/
def conditions : Prop :=
  P_W = 0.4 ∧ P_Z = 0.2

/-- Proof of the first statement -/
theorem statement1 (h : conditions P_W P_Z) : 
  P_W * P_Z = 0.08 := 
by sorry

/-- Proof of the second statement -/
theorem statement2 (h : conditions P_W P_Z) :
  P_W * (1 - P_Z) + (1 - P_W) * P_Z = 0.44 := 
by sorry

/-- Proof of the third statement -/
theorem statement3 (h : conditions P_W P_Z) :
  1 - P_W * P_Z = 0.92 := 
by sorry

end statement1_statement2_statement3_l219_219887


namespace find_x_l219_219997

theorem find_x 
  (b : ℤ) (h_b : b = 0) 
  (a z y x w : ℤ)
  (h1 : z + a = 1)
  (h2 : y + z + a = 0)
  (h3 : x + y + z = a)
  (h4 : w + x + y = z)
  :
  x = 2 :=
by {
    sorry
}    

end find_x_l219_219997


namespace find_angle4_l219_219084

theorem find_angle4
  (angle1 angle2 angle3 angle4 : ℝ)
  (h1 : angle1 = 70)
  (h2 : angle2 = 110)
  (h3 : angle3 = 40)
  (h4 : angle2 + angle3 + angle4 = 180) :
  angle4 = 30 := 
  sorry

end find_angle4_l219_219084


namespace average_speed_l219_219698

theorem average_speed (s₁ s₂ s₃ s₄ s₅ : ℝ) (h₁ : s₁ = 85) (h₂ : s₂ = 45) (h₃ : s₃ = 60) (h₄ : s₄ = 75) (h₅ : s₅ = 50) : 
  (s₁ + s₂ + s₃ + s₄ + s₅) / 5 = 63 := 
by 
  sorry

end average_speed_l219_219698


namespace regression_coeff_nonzero_l219_219177

theorem regression_coeff_nonzero (a b r : ℝ) (h : b = 0 → r = 0) : b ≠ 0 :=
sorry

end regression_coeff_nonzero_l219_219177


namespace am_gm_inequality_l219_219825

theorem am_gm_inequality (a1 a2 a3 : ℝ) (h₀ : 0 < a1) (h₁ : 0 < a2) (h₂ : 0 < a3) (h₃ : a1 + a2 + a3 = 1) : 
  1 / a1 + 1 / a2 + 1 / a3 ≥ 9 :=
by
  sorry

end am_gm_inequality_l219_219825


namespace fraction_of_red_marbles_after_tripling_blue_l219_219717

theorem fraction_of_red_marbles_after_tripling_blue (x : ℕ) (h₁ : ∃ y, y = (4 * x) / 7) (h₂ : ∃ z, z = (3 * x) / 7) :
  (3 * x / 7) / (((12 * x) / 7) + ((3 * x) / 7)) = 1 / 5 :=
by
  sorry

end fraction_of_red_marbles_after_tripling_blue_l219_219717


namespace find_x_value_l219_219666

theorem find_x_value (PQ_is_straight_line : True) 
  (angles_on_line : List ℕ) (h : angles_on_line = [x, x, x, x, x])
  (sum_of_angles : angles_on_line.sum = 180) :
  x = 36 :=
by
  sorry

end find_x_value_l219_219666


namespace pies_count_l219_219926

-- Definitions based on the conditions given in the problem
def strawberries_per_pie := 3
def christine_strawberries := 10
def rachel_strawberries := 2 * christine_strawberries

-- The theorem to prove
theorem pies_count : (christine_strawberries + rachel_strawberries) / strawberries_per_pie = 10 := by
  sorry

end pies_count_l219_219926


namespace raisin_addition_l219_219512

theorem raisin_addition : 
  let yellow_raisins := 0.3
  let black_raisins := 0.4
  yellow_raisins + black_raisins = 0.7 := 
by
  sorry

end raisin_addition_l219_219512


namespace unique_function_property_l219_219783

def f (n : Nat) : Nat := sorry

theorem unique_function_property :
  (∀ x y : ℕ+, x < y → f x < f y) ∧
  (∀ y x : ℕ+, f (y * f x) = x^2 * f (x * y)) →
  ∀ n : ℕ+, f n = n^2 :=
by
  intros h
  sorry

end unique_function_property_l219_219783


namespace triangle_side_cube_l219_219455

theorem triangle_side_cube 
  (a b c : ℕ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_gcd : Nat.gcd a (Nat.gcd b c) = 1)
  (angle_condition : ∃ A B : ℝ, A = 3 * B) 
  : ∃ n m : ℕ, (a = n ^ 3 ∨ b = n ^ 3 ∨ c = n ^ 3) :=
sorry

end triangle_side_cube_l219_219455


namespace calculate_expression_l219_219778

variable (f g : ℝ → ℝ)

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)
def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g x = - g (-x)

theorem calculate_expression 
  (hf : is_even_function f)
  (hg : is_odd_function g)
  (hfg : ∀ x : ℝ, f x - g x = x ^ 3 + x ^ 2 + 1) :
  f 1 + g 1 = 1 :=
  sorry

end calculate_expression_l219_219778


namespace cos_sum_simplified_l219_219564

theorem cos_sum_simplified :
  (Real.cos (2 * Real.pi / 17) + Real.cos (6 * Real.pi / 17) + Real.cos (8 * Real.pi / 17)) = ((Real.sqrt 13 - 1) / 4) :=
by
  sorry

end cos_sum_simplified_l219_219564


namespace part1_part2_l219_219297

-- Part (I)
theorem part1 (a : ℝ) :
  (∀ x : ℝ, 3 * x - abs (-2 * x + 1) ≥ a ↔ 2 ≤ x) → a = 3 :=
by
  sorry

-- Part (II)
theorem part2 (a : ℝ) :
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → (x - abs (x - a) ≤ 1)) → (a ≤ 1 ∨ 3 ≤ a) :=
by
  sorry

end part1_part2_l219_219297


namespace frustum_slant_height_l219_219221

theorem frustum_slant_height (r1 r2 V : ℝ) (h l : ℝ) 
    (H1 : r1 = 2) (H2 : r2 = 6) (H3 : V = 104 * π)
    (H4 : V = (1/3) * π * h * (r1^2 + r2^2 + r1 * r2)) 
    (H5 : h = 6)
    (H6 : l = Real.sqrt (h^2 + (r2 - r1)^2)) :
    l = 2 * Real.sqrt 13 :=
by sorry

end frustum_slant_height_l219_219221


namespace batch_preparation_l219_219514

theorem batch_preparation (total_students cupcakes_per_student cupcakes_per_batch percent_not_attending : ℕ)
    (hlt1 : total_students = 150)
    (hlt2 : cupcakes_per_student = 3)
    (hlt3 : cupcakes_per_batch = 20)
    (hlt4 : percent_not_attending = 20)
    : (total_students * (80 / 100) * cupcakes_per_student) / cupcakes_per_batch = 18 := by
  sorry

end batch_preparation_l219_219514


namespace converse_proposition_l219_219685

-- Define the predicate variables p and q
variables (p q : Prop)

-- State the theorem about the converse of the proposition
theorem converse_proposition (hpq : p → q) : q → p :=
sorry

end converse_proposition_l219_219685


namespace digit_205_of_14_div_360_l219_219990

noncomputable def decimal_expansion_of_fraction (n d : ℕ) : ℕ → ℕ := sorry

theorem digit_205_of_14_div_360 : 
  decimal_expansion_of_fraction 14 360 205 = 8 :=
sorry

end digit_205_of_14_div_360_l219_219990


namespace circle_center_l219_219607

theorem circle_center (x y : ℝ) (h : x^2 + 8*x + y^2 - 4*y = 16) : (x, y) = (-4, 2) :=
by 
  sorry

end circle_center_l219_219607


namespace sum_abs_frac_geq_frac_l219_219628

theorem sum_abs_frac_geq_frac (n : ℕ) (h1 : n ≥ 3) (a : Fin n → ℝ) (hnz : ∀ i : Fin n, a i ≠ 0) 
(hsum : (Finset.univ.sum a) = S) : 
  (Finset.univ.sum (fun i => |(S - a i) / a i|)) ≥ (n - 1) / (n - 2) :=
sorry

end sum_abs_frac_geq_frac_l219_219628


namespace sum_n_k_eq_eight_l219_219485

theorem sum_n_k_eq_eight (n k : Nat) (h1 : 4 * k = n - 3) (h2 : 8 * k + 13 = 3 * n) : n + k = 8 :=
by
  sorry

end sum_n_k_eq_eight_l219_219485


namespace this_week_usage_less_next_week_usage_less_l219_219697

def last_week_usage : ℕ := 91

def usage_this_week : ℕ := (4 * 8) + (3 * 10)

def usage_next_week : ℕ := (5 * 5) + (2 * 12)

theorem this_week_usage_less : last_week_usage - usage_this_week = 29 := by
  -- proof goes here
  sorry

theorem next_week_usage_less : last_week_usage - usage_next_week = 42 := by
  -- proof goes here
  sorry

end this_week_usage_less_next_week_usage_less_l219_219697


namespace chocolates_problem_l219_219555

theorem chocolates_problem (C S : ℝ) (n : ℕ) 
  (h1 : 24 * C = n * S)
  (h2 : (S - C) / C = 0.5) : 
  n = 16 :=
by 
  sorry

end chocolates_problem_l219_219555


namespace find_integer_k_l219_219346

theorem find_integer_k (k : ℤ) : (∃ k : ℤ, (k = 6) ∨ (k = 2) ∨ (k = 0) ∨ (k = -4)) ↔ (∃ k : ℤ, (2 * k^2 + k - 8) % (k - 1) = 0) :=
by
  sorry

end find_integer_k_l219_219346


namespace find_y_l219_219336

open Real

variable {x y : ℝ}

theorem find_y (h1 : x * y = 25) (h2 : x / y = 36) (hx : 0 < x) (hy : 0 < y) :
  y = 5 / 6 :=
by
  sorry

end find_y_l219_219336


namespace odd_function_increasing_ln_x_condition_l219_219273

theorem odd_function_increasing_ln_x_condition 
  {f : ℝ → ℝ} 
  (h_odd : ∀ x, f (-x) = -f x) 
  (h_increasing : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y) 
  {x : ℝ} 
  (h_f_ln_x : f (Real.log x) < 0) : 
  0 < x ∧ x < 1 := 
sorry

end odd_function_increasing_ln_x_condition_l219_219273


namespace polynomial_constant_l219_219920

theorem polynomial_constant
  (P : Polynomial ℤ)
  (h : ∀ Q F G : Polynomial ℤ, P.comp Q = F * G → F.degree = 0 ∨ G.degree = 0) :
  P.degree = 0 :=
by sorry

end polynomial_constant_l219_219920


namespace number_of_sides_of_polygon_l219_219014

theorem number_of_sides_of_polygon :
  ∃ n : ℕ, (n * (n - 3)) / 2 = 2 * n + 7 ∧ n = 8 := 
by
  sorry

end number_of_sides_of_polygon_l219_219014


namespace min_value_frac_l219_219824

theorem min_value_frac (x y : ℝ) (h1 : x > -1) (h2 : y > 0) (h3 : x + 2 * y = 2) : 
  ∃ L, (L = 3) ∧ (∀ (x y : ℝ), x > -1 → y > 0 → x + 2*y = 2 → 
  (∃ L, (L = 3) ∧ (∀ (x y : ℝ), x > -1 → y > 0 → x + 2*y = 2 → 
  ∀ (f : ℝ), f = (1 / (x + 1) + 2 / y) → f ≥ L))) :=
sorry

end min_value_frac_l219_219824


namespace find_A_l219_219708

theorem find_A (A B C D E F G H I J : ℕ)
  (h1 : A > B ∧ B > C)
  (h2 : D > E ∧ E > F)
  (h3 : G > H ∧ H > I ∧ I > J)
  (h4 : (D = E + 2) ∧ (E = F + 2))
  (h5 : (G = H + 2) ∧ (H = I + 2) ∧ (I = J + 2))
  (h6 : A + B + C = 10) : A = 6 :=
sorry

end find_A_l219_219708


namespace divisibility_condition_l219_219732

theorem divisibility_condition
  (a p q : ℕ) (hpq : p ≤ q) (hp_pos : 0 < p) (hq_pos : 0 < q) (ha_pos : 0 < a) :
  (p ∣ a^p ∨ p ∣ a^q) → (p ∣ a^p ∧ p ∣ a^q) :=
by
  sorry

end divisibility_condition_l219_219732


namespace find_two_digit_numbers_l219_219610

theorem find_two_digit_numbers (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h : 2 * (a + b) = a * b) : 
  10 * a + b = 63 ∨ 10 * a + b = 44 ∨ 10 * a + b = 36 :=
by sorry

end find_two_digit_numbers_l219_219610


namespace fill_missing_digits_l219_219701

noncomputable def first_number (a : ℕ) : ℕ := a * 1000 + 2 * 100 + 5 * 10 + 7
noncomputable def second_number (b c : ℕ) : ℕ := 2 * 1000 + b * 100 + 9 * 10 + c

theorem fill_missing_digits (a b c : ℕ) : a = 1 ∧ b = 5 ∧ c = 6 → first_number a + second_number b c = 5842 :=
by
  intros
  sorry

end fill_missing_digits_l219_219701


namespace circle_equation_l219_219507

theorem circle_equation : ∃ (x y : ℝ), (x - 2)^2 + y^2 = 2 :=
by
  sorry

end circle_equation_l219_219507


namespace JordanRectangleWidth_l219_219754

/-- Given that Carol's rectangle measures 15 inches by 24 inches,
and Jordan's rectangle is 8 inches long with equal area as Carol's rectangle,
prove that Jordan's rectangle is 45 inches wide. -/
theorem JordanRectangleWidth :
  ∃ W : ℝ, (15 * 24 = 8 * W) → W = 45 := by
  sorry

end JordanRectangleWidth_l219_219754


namespace percentage_difference_l219_219280

theorem percentage_difference : (70 / 100 : ℝ) * 100 - (60 / 100 : ℝ) * 80 = 22 := by
  sorry

end percentage_difference_l219_219280


namespace find_principal_l219_219462

theorem find_principal 
  (SI : ℝ) 
  (R : ℝ) 
  (T : ℝ) 
  (h_SI : SI = 4052.25) 
  (h_R : R = 9) 
  (h_T : T = 5) : 
  (SI * 100) / (R * T) = 9005 := 
by 
  rw [h_SI, h_R, h_T]
  sorry

end find_principal_l219_219462


namespace find_original_number_l219_219185

theorem find_original_number (x : ℝ) (h : 1.125 * x - 0.75 * x = 30) : x = 80 :=
by
  sorry

end find_original_number_l219_219185


namespace difference_between_waiter_and_twenty_less_l219_219996

-- Definitions for the given conditions
def total_slices : ℕ := 78
def ratio_buzz : ℕ := 5
def ratio_waiter : ℕ := 8
def total_ratio : ℕ := ratio_buzz + ratio_waiter
def slices_per_part : ℕ := total_slices / total_ratio
def buzz_share : ℕ := ratio_buzz * slices_per_part
def waiter_share : ℕ := ratio_waiter * slices_per_part
def twenty_less_waiter : ℕ := waiter_share - 20

-- The proof statement
theorem difference_between_waiter_and_twenty_less : 
  waiter_share - twenty_less_waiter = 20 :=
by sorry

end difference_between_waiter_and_twenty_less_l219_219996


namespace stratified_sampling_third_grade_students_l219_219184

variable (total_students : ℕ) (second_year_female_probability : ℚ) (sample_size : ℕ)

theorem stratified_sampling_third_grade_students
  (h_total : total_students = 2000)
  (h_probability : second_year_female_probability = 0.19)
  (h_sample_size : sample_size = 64) :
  let sampling_fraction := 64 / 2000
  let third_grade_students := 2000 * sampling_fraction
  third_grade_students = 16 :=
by
  -- the proof would go here, but we're skipping it per instructions
  sorry

end stratified_sampling_third_grade_students_l219_219184


namespace find_fourth_vertex_l219_219650

open Complex

theorem find_fourth_vertex (A B C: ℂ) (hA: A = 2 + 3 * Complex.I) 
                            (hB: B = -3 + 2 * Complex.I) 
                            (hC: C = -2 - 3 * Complex.I) : 
                            ∃ D : ℂ, D = 2.5 + 0.5 * Complex.I :=
by 
  sorry

end find_fourth_vertex_l219_219650


namespace sphere_surface_area_l219_219433

variable (x y z : ℝ)

theorem sphere_surface_area :
  (x^2 + y^2 + z^2 = 1) → (4 * Real.pi) = 4 * Real.pi :=
by
  intro h
  -- The proof will be inserted here
  sorry

end sphere_surface_area_l219_219433


namespace capital_at_end_of_2014_year_capital_exceeds_32dot5_billion_l219_219168

noncomputable def company_capital (n : ℕ) : ℝ :=
  if n = 0 then 1000
  else 2 * company_capital (n - 1) - 500

theorem capital_at_end_of_2014 : company_capital 4 = 8500 :=
by sorry

theorem year_capital_exceeds_32dot5_billion : ∀ n : ℕ, company_capital n > 32500 → n ≥ 7 :=
by sorry

end capital_at_end_of_2014_year_capital_exceeds_32dot5_billion_l219_219168


namespace total_spent_by_mrs_hilt_l219_219396

-- Define the cost per set of tickets for kids.
def cost_per_set_kids : ℕ := 1
-- Define the number of tickets in a set for kids.
def tickets_per_set_kids : ℕ := 4

-- Define the cost per set of tickets for adults.
def cost_per_set_adults : ℕ := 2
-- Define the number of tickets in a set for adults.
def tickets_per_set_adults : ℕ := 3

-- Define the total number of kids' tickets purchased.
def total_kids_tickets : ℕ := 12
-- Define the total number of adults' tickets purchased.
def total_adults_tickets : ℕ := 9

-- Prove that the total amount spent by Mrs. Hilt is $9.
theorem total_spent_by_mrs_hilt :
  (total_kids_tickets / tickets_per_set_kids * cost_per_set_kids) + 
  (total_adults_tickets / tickets_per_set_adults * cost_per_set_adults) = 9 :=
by sorry

end total_spent_by_mrs_hilt_l219_219396


namespace initial_number_of_red_balls_l219_219894

theorem initial_number_of_red_balls 
  (num_white_balls num_red_balls : ℕ)
  (h1 : num_red_balls = 4 * num_white_balls + 3)
  (num_actions : ℕ)
  (h2 : 4 + 5 * num_actions = num_white_balls)
  (h3 : 34 + 17 * num_actions = num_red_balls) : 
  num_red_balls = 119 := 
by
  sorry

end initial_number_of_red_balls_l219_219894


namespace find_a_of_exponential_passing_point_l219_219220

theorem find_a_of_exponential_passing_point (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) (h_point : a^2 = 4) : a = 2 :=
by
  -- Proof will be filled in here
  sorry

end find_a_of_exponential_passing_point_l219_219220


namespace concert_attendance_difference_l219_219550

theorem concert_attendance_difference :
  let first_concert := 65899
  let second_concert := 66018
  second_concert - first_concert = 119 :=
by
  sorry

end concert_attendance_difference_l219_219550


namespace initial_number_is_nine_l219_219519

theorem initial_number_is_nine (x : ℝ) (h : 3 * (2 * x + 13) = 93) : x = 9 :=
sorry

end initial_number_is_nine_l219_219519


namespace area_and_cost_of_path_l219_219598

-- Define the dimensions of the grass field
def length_field : ℝ := 85
def width_field : ℝ := 55

-- Define the width of the path around the field
def width_path : ℝ := 2.5

-- Define the cost per square meter of constructing the path
def cost_per_sqm : ℝ := 2

-- Define new dimensions including the path
def new_length : ℝ := length_field + 2 * width_path
def new_width : ℝ := width_field + 2 * width_path

-- Define the area of the entire field including the path
def area_with_path : ℝ := new_length * new_width

-- Define the area of the grass field without the path
def area_field : ℝ := length_field * width_field

-- Define the area of the path alone
def area_path : ℝ := area_with_path - area_field

-- Define the cost of constructing the path
def cost_constructing_path : ℝ := area_path * cost_per_sqm

-- Theorem to prove the area of the path and cost of constructing it
theorem area_and_cost_of_path :
  area_path = 725 ∧ cost_constructing_path = 1450 :=
by
  -- Skipping the proof as instructed
  sorry

end area_and_cost_of_path_l219_219598


namespace li_bai_initial_wine_l219_219344

theorem li_bai_initial_wine (x : ℕ) 
  (h : (((((x * 2 - 2) * 2 - 2) * 2 - 2) * 2 - 2) = 2)) : 
  x = 2 :=
by
  sorry

end li_bai_initial_wine_l219_219344


namespace total_cost_of_square_park_l219_219991

-- Define the cost per side and number of sides
def cost_per_side : ℕ := 56
def sides_of_square : ℕ := 4

-- The total cost of fencing the park
def total_cost_of_fencing (cost_per_side : ℕ) (sides_of_square : ℕ) : ℕ := cost_per_side * sides_of_square

-- The statement we need to prove
theorem total_cost_of_square_park : total_cost_of_fencing cost_per_side sides_of_square = 224 :=
by sorry

end total_cost_of_square_park_l219_219991


namespace monthly_installment_amount_l219_219875

theorem monthly_installment_amount (total_cost : ℝ) (down_payment_percentage : ℝ) (additional_down_payment : ℝ) 
  (balance_after_months : ℝ) (months : ℕ) (monthly_installment : ℝ) : 
    total_cost = 1000 → 
    down_payment_percentage = 0.20 → 
    additional_down_payment = 20 → 
    balance_after_months = 520 → 
    months = 4 → 
    monthly_installment = 65 :=
by
  intros
  sorry

end monthly_installment_amount_l219_219875


namespace six_digit_phone_number_count_l219_219071

def six_digit_to_seven_digit_count (six_digit : ℕ) (h : 100000 ≤ six_digit ∧ six_digit < 1000000) : ℕ :=
  let num_positions := 7
  let num_digits := 10
  num_positions * num_digits

theorem six_digit_phone_number_count (six_digit : ℕ) (h : 100000 ≤ six_digit ∧ six_digit < 1000000) :
  six_digit_to_seven_digit_count six_digit h = 70 := by
  -- Proof goes here
  sorry

end six_digit_phone_number_count_l219_219071


namespace probability_of_two_queens_or_at_least_one_king_l219_219867

def probability_two_queens_or_at_least_one_king : ℚ := 2 / 13

theorem probability_of_two_queens_or_at_least_one_king :
  let probability_two_queens := (4/52) * (3/51)
  let probability_exactly_one_king := (2 * (4/52) * (48/51))
  let probability_two_kings := (4/52) * (3/51)
  let probability_at_least_one_king := probability_exactly_one_king + probability_two_kings
  let total_probability := probability_two_queens + probability_at_least_one_king
  total_probability = probability_two_queens_or_at_least_one_king := 
by
  sorry

end probability_of_two_queens_or_at_least_one_king_l219_219867


namespace banana_pieces_l219_219584

theorem banana_pieces (B G P : ℕ) 
  (h1 : P = 4 * G)
  (h2 : G = B + 5)
  (h3 : P = 192) : B = 43 := 
by
  sorry

end banana_pieces_l219_219584


namespace joan_games_last_year_l219_219089

theorem joan_games_last_year (games_this_year : ℕ) (total_games : ℕ) (games_last_year : ℕ) 
  (h1 : games_this_year = 4) 
  (h2 : total_games = 9) 
  (h3 : total_games = games_this_year + games_last_year) : 
  games_last_year = 5 := 
by
  sorry

end joan_games_last_year_l219_219089


namespace length_difference_squares_l219_219516

theorem length_difference_squares (A B : ℝ) (hA : A^2 = 25) (hB : B^2 = 81) : B - A = 4 :=
by
  sorry

end length_difference_squares_l219_219516


namespace find_g_expression_l219_219424

theorem find_g_expression (g f : ℝ → ℝ) (h_sym : ∀ x y, g x = y ↔ g (2 - x) = 4 - y)
  (h_f : ∀ x, f x = 3 * x - 1) :
  ∀ x, g x = 3 * x - 1 :=
by
  sorry

end find_g_expression_l219_219424


namespace axis_of_symmetry_l219_219560

noncomputable def f (x : ℝ) := x^2 - 2 * x + Real.cos (x - 1)

theorem axis_of_symmetry :
  ∀ x : ℝ, f (1 + x) = f (1 - x) :=
by 
  sorry

end axis_of_symmetry_l219_219560


namespace part_i_part_ii_l219_219900

open Real -- Open the Real number space

-- (i) Prove that for any real number x, there exist two points of the same color that are at a distance of x from each other
theorem part_i (color : Real × Real → Bool) :
  ∀ x : ℝ, ∃ p1 p2 : Real × Real, color p1 = color p2 ∧ dist p1 p2 = x :=
by
  sorry

-- (ii) Prove that there exists a color such that for every real number x, 
-- we can find two points of that color that are at a distance of x from each other
theorem part_ii (color : Real × Real → Bool) :
  ∃ c : Bool, ∀ x : ℝ, ∃ p1 p2 : Real × Real, color p1 = c ∧ color p2 = c ∧ dist p1 p2 = x :=
by
  sorry

end part_i_part_ii_l219_219900


namespace total_gallons_in_tanks_l219_219210

theorem total_gallons_in_tanks (
  tank1_cap : ℕ := 7000) (tank2_cap : ℕ := 5000) (tank3_cap : ℕ := 3000)
  (fill1_fraction : ℚ := 3/4) (fill2_fraction : ℚ := 4/5) (fill3_fraction : ℚ := 1/2)
  : tank1_cap * fill1_fraction + tank2_cap * fill2_fraction + tank3_cap * fill3_fraction = 10750 := by
  sorry

end total_gallons_in_tanks_l219_219210


namespace bottles_in_cups_l219_219488

-- Defining the given conditions
variables (BOTTLE GLASS CUP JUG : ℕ)

axiom h1 : JUG = BOTTLE + GLASS
axiom h2 : 2 * JUG = 7 * GLASS
axiom h3 : BOTTLE = CUP + 2 * GLASS

theorem bottles_in_cups : BOTTLE = 5 * CUP :=
sorry

end bottles_in_cups_l219_219488


namespace membership_relation_l219_219630

-- Definitions of M and N
def M (x : ℝ) : Prop := abs (x + 1) < 4
def N (x : ℝ) : Prop := x / (x - 3) < 0

theorem membership_relation (a : ℝ) (h : M a) : N a → M a := by
  sorry

end membership_relation_l219_219630


namespace Bennett_has_6_brothers_l219_219190

theorem Bennett_has_6_brothers (num_aaron_brothers : ℕ) (num_bennett_brothers : ℕ) 
  (h1 : num_aaron_brothers = 4) 
  (h2 : num_bennett_brothers = 2 * num_aaron_brothers - 2) : 
  num_bennett_brothers = 6 := by
  sorry

end Bennett_has_6_brothers_l219_219190


namespace non_congruent_triangles_with_perimeter_11_l219_219925

theorem non_congruent_triangles_with_perimeter_11 : 
  ∀ (a b c : ℕ), a + b + c = 11 → a < b + c → b < a + c → c < a + b → 
  ∃! (a b c : ℕ), (a, b, c) = (2, 4, 5) ∨ (a, b, c) = (3, 4, 4) := 
by sorry

end non_congruent_triangles_with_perimeter_11_l219_219925


namespace tammy_speed_second_day_l219_219576

theorem tammy_speed_second_day:
  ∃ (v t: ℝ), 
    t + (t - 2) = 14 ∧
    v * t + (v + 0.5) * (t - 2) = 52 ∧
    (v + 0.5) = 4 := sorry

end tammy_speed_second_day_l219_219576


namespace percentage_error_in_calculated_area_l219_219536

theorem percentage_error_in_calculated_area
  (a : ℝ)
  (measured_side_length : ℝ := 1.025 * a) :
  (measured_side_length ^ 2 - a ^ 2) / (a ^ 2) * 100 = 5.0625 :=
by 
  sorry

end percentage_error_in_calculated_area_l219_219536


namespace restore_original_salary_l219_219517

theorem restore_original_salary (orig_salary : ℝ) (reducing_percent : ℝ) (increasing_percent : ℝ) :
  reducing_percent = 20 → increasing_percent = 25 →
  (orig_salary * (1 - reducing_percent / 100)) * (1 + increasing_percent / 100 / (1 - reducing_percent / 100)) = orig_salary
:= by
  intros
  sorry

end restore_original_salary_l219_219517


namespace max_value_l219_219467

theorem max_value (a b c : ℕ) (h1 : a = 2^35) (h2 : b = 26) (h3 : c = 1) : max a (max b c) = 2^35 :=
by
  -- This is where the proof would go
  sorry

end max_value_l219_219467


namespace find_a_b_l219_219334

theorem find_a_b (a b : ℝ) (z : ℂ) (hz : z = 1 + Complex.I) 
  (h : (z^2 + a*z + b) / (z^2 - z + 1) = 1 - Complex.I) : a = -1 ∧ b = 2 :=
by
  sorry

end find_a_b_l219_219334


namespace cubics_of_sum_and_product_l219_219151

theorem cubics_of_sum_and_product (x y : ℝ) (h₁ : x + y = 10) (h₂ : x * y = 11) : 
  x^3 + y^3 = 670 :=
by
  sorry

end cubics_of_sum_and_product_l219_219151


namespace nested_fraction_value_l219_219224

theorem nested_fraction_value :
  1 + (1 / (1 + (1 / (2 + (2 / 3))))) = 19 / 11 :=
by sorry

end nested_fraction_value_l219_219224


namespace estimate_students_in_range_l219_219068

noncomputable def n_students := 3000
noncomputable def score_range_low := 70
noncomputable def score_range_high := 80
noncomputable def est_students_in_range := 408

theorem estimate_students_in_range : ∀ (n : ℕ) (k : ℕ), n = n_students →
  k = est_students_in_range →
  normal_distribution :=
sorry

end estimate_students_in_range_l219_219068


namespace positive_integers_count_l219_219379

theorem positive_integers_count (n : ℕ) : 
  ∃ m : ℕ, (m ≤ n / 2014 ∧ m ≤ n / 2016 ∧ (m + 1) * 2014 > n ∧ (m + 1) * 2016 > n) ↔
  (n = 1015056) :=
by
  sorry

end positive_integers_count_l219_219379


namespace div_z_x_l219_219889

variables (x y z : ℚ)

theorem div_z_x (h1 : x / y = 3) (h2 : y / z = 5 / 2) : z / x = 2 / 15 :=
sorry

end div_z_x_l219_219889


namespace percent_relation_l219_219259

theorem percent_relation (x y z w : ℝ) (h1 : x = 1.25 * y) (h2 : y = 0.40 * z) (h3 : z = 1.10 * w) :
  (x / w) * 100 = 55 := by sorry

end percent_relation_l219_219259


namespace condition1_condition2_condition3_condition4_l219_219021

-- Proof for the equivalence of conditions and point descriptions

theorem condition1 (x y : ℝ) : 
  (x >= -2) ↔ ∃ y : ℝ, x = -2 ∨ x > -2 := 
by
  sorry

theorem condition2 (x y : ℝ) : 
  (-2 < x ∧ x < 2) ↔ ∃ y : ℝ, -2 < x ∧ x < 2 := 
by
  sorry

theorem condition3 (x y : ℝ) : 
  (|x| < 2) ↔ -2 < x ∧ x < 2 :=
by
  sorry

theorem condition4 (x y : ℝ) : 
  (|x| ≥ 2) ↔ (x ≤ -2 ∨ x ≥ 2) :=
by 
  sorry

end condition1_condition2_condition3_condition4_l219_219021


namespace future_ages_equation_l219_219101

-- Defining the ages of Joe and James with given conditions
def joe_current_age : ℕ := 22
def james_current_age : ℕ := 12

-- Defining the condition that Joe is 10 years older than James
lemma joe_older_than_james : joe_current_age = james_current_age + 10 := by
  unfold joe_current_age james_current_age
  simp

-- Defining the future age condition equation and the target years y.
theorem future_ages_equation (y : ℕ) :
  2 * (joe_current_age + y) = 3 * (james_current_age + y) → y = 8 := by
  unfold joe_current_age james_current_age
  intro h
  linarith

end future_ages_equation_l219_219101


namespace square_area_l219_219234

theorem square_area (perimeter : ℝ) (h_perimeter : perimeter = 40) : 
  ∃ (area : ℝ), area = 100 := by
  sorry

end square_area_l219_219234


namespace age_of_15th_student_l219_219594

theorem age_of_15th_student 
  (avg_age_all : ℕ → ℕ → ℕ)
  (avg_age : avg_age_all 15 15 = 15)
  (avg_age_4 : avg_age_all 4 14 = 14)
  (avg_age_10 : avg_age_all 10 16 = 16) : 
  ∃ age15 : ℕ, age15 = 9 := 
by
  sorry

end age_of_15th_student_l219_219594


namespace simplify_fraction_l219_219090

theorem simplify_fraction : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1/2 :=
by sorry

end simplify_fraction_l219_219090


namespace cos_double_angle_given_tan_l219_219062

theorem cos_double_angle_given_tan (x : ℝ) (h : Real.tan x = 2) : Real.cos (2 * x) = -3 / 5 :=
by sorry

end cos_double_angle_given_tan_l219_219062


namespace find_ages_of_son_daughter_and_niece_l219_219413

theorem find_ages_of_son_daughter_and_niece
  (S : ℕ) (D : ℕ) (N : ℕ)
  (h1 : ∀ (M : ℕ), M = S + 24) 
  (h2 : ∀ (M : ℕ), 2 * (S + 2) = M + 2)
  (h3 : D = S / 2)
  (h4 : 2 * (D + 6) = 2 * S * 2 / 3)
  (h5 : N = S - 3)
  (h6 : 5 * N = 4 * S) :
  S = 22 ∧ D = 11 ∧ N = 19 := 
by 
  sorry

end find_ages_of_son_daughter_and_niece_l219_219413


namespace express_in_scientific_notation_l219_219496

theorem express_in_scientific_notation :
  ∃ (a : ℝ) (b : ℤ), 159600 = a * 10 ^ b ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.596 ∧ b = 5 :=
by
  sorry

end express_in_scientific_notation_l219_219496


namespace no_pos_integers_exist_l219_219309

theorem no_pos_integers_exist (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  ¬ (3 * (a * b + b * c + c * a) ∣ a^2 + b^2 + c^2) :=
sorry

end no_pos_integers_exist_l219_219309


namespace f_leq_2x_l219_219352

noncomputable def f : ℝ → ℝ := sorry
axiom f_nonneg {x : ℝ} (hx : 0 ≤ x ∧ x ≤ 1) : 0 ≤ f x
axiom f_one : f 1 = 1
axiom f_superadditive {x y : ℝ} (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hxy : x + y ≤ 1) : f (x + y) ≥ f x + f y

-- The theorem statement to be proved
theorem f_leq_2x {x : ℝ} (hx : 0 ≤ x ∧ x ≤ 1) : f x ≤ 2 * x := sorry

end f_leq_2x_l219_219352


namespace range_abs_plus_one_l219_219015

 theorem range_abs_plus_one : 
   ∀ y : ℝ, (∃ x : ℝ, y = |x| + 1) ↔ y ≥ 1 := 
 by
   sorry
 
end range_abs_plus_one_l219_219015


namespace incorrect_statement_is_C_l219_219588

theorem incorrect_statement_is_C (b h s a x : ℝ) (hb : b > 0) (hh : h > 0) (hs : s > 0) (hx : x < 0) :
  ¬ (9 * s^2 = 4 * (3 * s)^2) :=
by
  sorry

end incorrect_statement_is_C_l219_219588


namespace positivity_of_fraction_l219_219112

theorem positivity_of_fraction
  (a b c d x1 x2 x3 x4 : ℝ)
  (h_neg_a : a < 0)
  (h_neg_b : b < 0)
  (h_neg_c : c < 0)
  (h_neg_d : d < 0)
  (h_abs : |x1 - a| + |x2 + b| + |x3 - c| + |x4 + d| = 0) :
  (x1 * x2 / (x3 * x4) > 0) := by
  sorry

end positivity_of_fraction_l219_219112


namespace perimeter_ABCDEFG_l219_219439

variables {Point : Type}
variables {dist : Point → Point → ℝ}  -- Distance function

-- Definitions for midpoint and equilateral triangles
def is_midpoint (M A B : Point) : Prop := dist A M = dist M B ∧ dist A B = 2 * dist A M
def is_equilateral (A B C : Point) : Prop := dist A B = dist B C ∧ dist B C = dist C A

variables {A B C D E F G : Point}  -- Points in the plane
variables (h_eq_triangle_ABC : is_equilateral A B C)
variables (h_eq_triangle_ADE : is_equilateral A D E)
variables (h_eq_triangle_EFG : is_equilateral E F G)
variables (h_midpoint_D : is_midpoint D A C)
variables (h_midpoint_G : is_midpoint G A E)
variables (h_midpoint_F : is_midpoint F D E)
variables (h_AB_length : dist A B = 6)

theorem perimeter_ABCDEFG : 
  dist A B + dist B C + dist C D + dist D E + dist E F + dist F G + dist G A = 24 :=
sorry

end perimeter_ABCDEFG_l219_219439


namespace f_2_eq_4_l219_219364

def f (n : ℕ) : ℕ := (List.range (n + 1)).sum + (List.range n).sum

theorem f_2_eq_4 : f 2 = 4 := by
  sorry

end f_2_eq_4_l219_219364


namespace hawks_score_l219_219614

theorem hawks_score (a b : ℕ) (h1 : a + b = 58) (h2 : a - b = 12) : b = 23 :=
by
  sorry

end hawks_score_l219_219614


namespace abs_neg_six_l219_219957

theorem abs_neg_six : abs (-6) = 6 := by
  sorry

end abs_neg_six_l219_219957


namespace min_sum_of_dimensions_l219_219694

theorem min_sum_of_dimensions (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 3003) :
  a + b + c = 45 := sorry

end min_sum_of_dimensions_l219_219694


namespace root_of_quadratic_eq_l219_219065

theorem root_of_quadratic_eq (a b : ℝ) (h : a + b - 3 = 0) : a + b = 3 :=
sorry

end root_of_quadratic_eq_l219_219065


namespace larger_value_3a_plus_1_l219_219552

theorem larger_value_3a_plus_1 {a : ℝ} (h : 8 * a^2 + 6 * a + 2 = 0) : 3 * a + 1 ≤ 3 * (-1/4 : ℝ) + 1 := 
sorry

end larger_value_3a_plus_1_l219_219552


namespace hotpot_total_cost_l219_219642

def table_cost : ℝ := 280
def table_limit : ℕ := 8
def extra_person_cost : ℝ := 29.9
def total_people : ℕ := 12

theorem hotpot_total_cost : 
  total_people > table_limit →
  table_cost + (total_people - table_limit) * extra_person_cost = 369.7 := 
by 
  sorry

end hotpot_total_cost_l219_219642


namespace equal_playing_time_l219_219229

def number_of_players : ℕ := 10
def players_on_field : ℕ := 8
def match_duration : ℕ := 45

theorem equal_playing_time :
  (players_on_field * match_duration) / number_of_players = 36 :=
by
  sorry

end equal_playing_time_l219_219229


namespace emily_euros_contribution_l219_219192

-- Declare the conditions as a definition
def conditions : Prop :=
  ∃ (cost_of_pie : ℝ) (emily_usd : ℝ) (berengere_euros : ℝ) (exchange_rate : ℝ),
    cost_of_pie = 15 ∧
    emily_usd = 10 ∧
    berengere_euros = 3 ∧
    exchange_rate = 1.1

-- Define the proof problem based on the conditions and required contribution
theorem emily_euros_contribution : conditions → (∃ emily_euros_more : ℝ, emily_euros_more = 3) :=
by
  intro h
  sorry

end emily_euros_contribution_l219_219192


namespace smallest_sum_of_50_consecutive_integers_with_product_zero_and_positive_sum_l219_219299

theorem smallest_sum_of_50_consecutive_integers_with_product_zero_and_positive_sum :
  ∃ (a : ℤ), (∃ (l : List ℤ), l.length = 50 ∧ List.prod l = 0 ∧ 0 < List.sum l ∧ List.sum l = 25) :=
by
  sorry

end smallest_sum_of_50_consecutive_integers_with_product_zero_and_positive_sum_l219_219299


namespace total_pears_picked_l219_219854

def pears_Alyssa : ℕ := 42
def pears_Nancy : ℕ := 17

theorem total_pears_picked : pears_Alyssa + pears_Nancy = 59 :=
by sorry

end total_pears_picked_l219_219854


namespace typhoon_probabilities_l219_219078

-- Defining the conditions
def probAtLeastOneHit : ℝ := 0.36

-- Defining the events and probabilities
def probOfHit (p : ℝ) := p
def probBothHit (p : ℝ) := p^2

def probAtLeastOne (p : ℝ) : ℝ := p^2 + 2 * p * (1 - p)

-- Defining the variable X as the number of cities hit by the typhoon
def P_X_0 (p : ℝ) : ℝ := (1 - p)^2
def P_X_1 (p : ℝ) : ℝ := 2 * p * (1 - p)
def E_X (p : ℝ) : ℝ := 2 * p

-- Main theorem
theorem typhoon_probabilities :
  ∀ (p : ℝ),
    probAtLeastOne p = probAtLeastOneHit → 
    p = 0.2 ∧ P_X_0 p = 0.64 ∧ P_X_1 p = 0.32 ∧ E_X p = 0.4 :=
by
  intros p h
  sorry

end typhoon_probabilities_l219_219078


namespace geometric_sequence_common_ratio_l219_219712

theorem geometric_sequence_common_ratio (a1 a2 a3 a4 : ℝ)
  (h₁ : a1 = 32) (h₂ : a2 = -48) (h₃ : a3 = 72) (h₄ : a4 = -108)
  (h_geom : ∃ r, a2 = r * a1 ∧ a3 = r * a2 ∧ a4 = r * a3) :
  ∃ r, r = -3/2 :=
by
  sorry

end geometric_sequence_common_ratio_l219_219712


namespace age_difference_l219_219340

variable (A B C : ℕ)

theorem age_difference (h₁ : C = A - 20) : (A + B) = (B + C) + 20 := 
sorry

end age_difference_l219_219340


namespace ratio_volume_sphere_to_hemisphere_l219_219817

noncomputable def volume_sphere (r : ℝ) : ℝ :=
  (4/3) * Real.pi * r^3

noncomputable def volume_hemisphere (r : ℝ) : ℝ :=
  (1/2) * volume_sphere r

theorem ratio_volume_sphere_to_hemisphere (p : ℝ) (hp : 0 < p) :
  (volume_sphere p) / (volume_hemisphere (2 * p)) = 1 / 4 :=
by
  sorry

end ratio_volume_sphere_to_hemisphere_l219_219817


namespace closest_ratio_one_l219_219426

theorem closest_ratio_one (a c : ℕ) (h1 : 30 * a + 15 * c = 2700) (h2 : a ≥ 1) (h3 : c ≥ 1) :
  a = c :=
by sorry

end closest_ratio_one_l219_219426


namespace find_c_l219_219844

noncomputable def cubic_function (x : ℝ) (c : ℝ) : ℝ :=
  x^3 - 3 * x + c

theorem find_c (c : ℝ) :
  (∃ x₁ x₂ : ℝ, cubic_function x₁ c = 0 ∧ cubic_function x₂ c = 0 ∧ x₁ ≠ x₂) →
  (c = -2 ∨ c = 2) :=
by
  sorry

end find_c_l219_219844


namespace books_remainder_l219_219601

theorem books_remainder (total_books new_books_per_section sections : ℕ) 
  (h1 : total_books = 1521) 
  (h2 : new_books_per_section = 45) 
  (h3 : sections = 41) : 
  (total_books * sections) % new_books_per_section = 36 :=
by
  sorry

end books_remainder_l219_219601


namespace perpendicular_line_eq_l219_219412

theorem perpendicular_line_eq (x y : ℝ) : 
  (∃ m : ℝ, (m * y + 2 * x = -5 / 2) ∧ (x - 2 * y + 3 = 0)) →
  ∃ a b c : ℝ, (a * x + b * y + c = 0) ∧ (2 * a + b = 0) ∧ c = 1 := sorry

end perpendicular_line_eq_l219_219412


namespace number_of_performance_orders_l219_219290

-- Define the options for the programs
def programs : List String := ["A", "B", "C", "D", "E", "F", "G", "H"]

-- Define a function to count valid performance orders under given conditions
def countPerformanceOrders (progs : List String) : ℕ :=
  sorry  -- This is where the logic to count performance orders goes

-- The theorem to assert the total number of performance orders
theorem number_of_performance_orders : countPerformanceOrders programs = 2860 :=
by
  sorry  -- Proof of the theorem

end number_of_performance_orders_l219_219290


namespace average_marks_second_class_l219_219109

variable (average_marks_first_class : ℝ) (students_first_class : ℕ)
variable (students_second_class : ℕ) (combined_average_marks : ℝ)

theorem average_marks_second_class (H1 : average_marks_first_class = 60)
  (H2 : students_first_class = 55) (H3 : students_second_class = 48)
  (H4 : combined_average_marks = 59.067961165048544) :
  48 * 57.92 = 103 * 59.067961165048544 - 3300 := by
  sorry

end average_marks_second_class_l219_219109


namespace quadratic_has_real_root_l219_219964

theorem quadratic_has_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ≤ -10 ∨ b ≥ 10) := sorry

end quadratic_has_real_root_l219_219964


namespace ratio_of_autobiographies_to_fiction_l219_219731

theorem ratio_of_autobiographies_to_fiction (total_books fiction_books non_fiction_books picture_books autobiographies: ℕ) 
  (h1 : total_books = 35) 
  (h2 : fiction_books = 5) 
  (h3 : non_fiction_books = fiction_books + 4) 
  (h4 : picture_books = 11) 
  (h5 : autobiographies = total_books - (fiction_books + non_fiction_books + picture_books)) :
  autobiographies / fiction_books = 2 :=
by sorry

end ratio_of_autobiographies_to_fiction_l219_219731


namespace gcd_three_numbers_l219_219750

theorem gcd_three_numbers (a b c : ℕ) (h1 : a = 72) (h2 : b = 120) (h3 : c = 168) :
  Nat.gcd (Nat.gcd a b) c = 24 :=
by
  rw [h1, h2, h3]
  exact sorry

end gcd_three_numbers_l219_219750


namespace find_expression_value_l219_219746

-- We declare our variables x and y
variables (x y : ℝ)

-- We state our conditions as hypotheses
def h1 : 3 * x + y = 5 := sorry
def h2 : x + 3 * y = 8 := sorry

-- We prove the given mathematical expression
theorem find_expression_value (h1 : 3 * x + y = 5) (h2 : x + 3 * y = 8) : 10 * x^2 + 19 * x * y + 10 * y^2 = 153 := 
by
  -- We intentionally skip the proof
  sorry

end find_expression_value_l219_219746


namespace max_value_of_g_l219_219111

noncomputable def f1 (x : ℝ) : ℝ := 3 * x + 3
noncomputable def f2 (x : ℝ) : ℝ := (1/3) * x + 2
noncomputable def f3 (x : ℝ) : ℝ := -x + 8

noncomputable def g (x : ℝ) : ℝ := min (min (f1 x) (f2 x)) (f3 x)

theorem max_value_of_g : ∃ x : ℝ, g x = 3.5 :=
by
  sorry

end max_value_of_g_l219_219111


namespace find_l_l219_219909

variables (a b c l : ℤ)
def g (x : ℤ) : ℤ := a * x^2 + b * x + c

theorem find_l :
  g a b c 2 = 0 →
  60 < g a b c 6 ∧ g a b c 6 < 70 →
  80 < g a b c 9 ∧ g a b c 9 < 90 →
  6000 * l < g a b c 100 ∧ g a b c 100 < 6000 * (l + 1) →
  l = 5 :=
sorry

end find_l_l219_219909


namespace dan_must_exceed_speed_to_arrive_before_cara_l219_219129

noncomputable def minimum_speed_for_dan (distance : ℕ) (cara_speed : ℕ) (dan_delay : ℕ) : ℕ :=
  (distance / (distance / cara_speed - dan_delay)) + 1

theorem dan_must_exceed_speed_to_arrive_before_cara
  (distance : ℕ) (cara_speed : ℕ) (dan_delay : ℕ) :
  distance = 180 →
  cara_speed = 30 →
  dan_delay = 1 →
  minimum_speed_for_dan distance cara_speed dan_delay > 36 :=
by
  sorry

end dan_must_exceed_speed_to_arrive_before_cara_l219_219129


namespace point_on_ellipse_l219_219908

noncomputable def ellipse_condition (P F1 F2 : ℝ × ℝ) : Prop :=
  let x := P.1
  let y := P.2
  let d1 := ((x - F1.1)^2 + (y - F1.2)^2).sqrt
  let d2 := ((x - F2.1)^2 + (y - F2.2)^2).sqrt
  x^2 + 4 * y^2 = 16 ∧ d1 = 7

theorem point_on_ellipse (P F1 F2 : ℝ × ℝ)
  (h : ellipse_condition P F1 F2) : 
  let x := P.1
  let y := P.2
  let d2 := ((x - F2.1)^2 + (y - F2.2)^2).sqrt
  d2 = 1 :=
sorry

end point_on_ellipse_l219_219908


namespace necessary_but_not_sufficient_condition_holds_l219_219899

-- Let m be a real number
variable (m : ℝ)

-- Define the conditions
def condition_1 : Prop := (m + 3) * (2 * m + 1) < 0
def condition_2 : Prop := -(2 * m - 1) > m + 2
def condition_3 : Prop := m + 2 > 0

-- Define necessary but not sufficient condition
def necessary_but_not_sufficient : Prop :=
  -2 < m ∧ m < -1 / 3

-- Problem statement
theorem necessary_but_not_sufficient_condition_holds 
  (h1 : condition_1 m) 
  (h2 : condition_2 m) 
  (h3 : condition_3 m) : necessary_but_not_sufficient m :=
sorry

end necessary_but_not_sufficient_condition_holds_l219_219899


namespace reptile_house_animal_multiple_l219_219818

theorem reptile_house_animal_multiple (R F x : ℕ) (hR : R = 16) (hF : F = 7) (hCond : R = x * F - 5) : x = 3 := by
  sorry

end reptile_house_animal_multiple_l219_219818


namespace symmetric_point_line_eq_l219_219254

theorem symmetric_point_line_eq (A B : ℝ × ℝ) (l : ℝ → ℝ) (x1 y1 x2 y2 : ℝ)
  (hA : A = (4, 5))
  (hB : B = (-2, 7))
  (hSymmetric : ∀ x y, B = (2 * l x - A.1, 2 * l y - A.2)) :
  ∀ x y, l x = 3 * x - 5 ∧ l y = 3 * y + 6 :=
by
  sorry

end symmetric_point_line_eq_l219_219254


namespace number_of_mismatching_socks_l219_219452

def SteveTotalSocks := 48
def StevePairsMatchingSocks := 11

theorem number_of_mismatching_socks :
  SteveTotalSocks - (StevePairsMatchingSocks * 2) = 26 := by
  sorry

end number_of_mismatching_socks_l219_219452


namespace minimum_value_x2_minus_x1_range_of_a_l219_219130

noncomputable def f (x : ℝ) := Real.sin x + Real.exp x
noncomputable def g (x : ℝ) (a : ℝ) := a * x
noncomputable def F (x : ℝ) (a : ℝ) := f x - g x a

-- Question (I)
theorem minimum_value_x2_minus_x1 : ∃ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ 0 ≤ x₂ ∧ a = 1 / 3 ∧ f x₁ = g x₂ a → x₂ - x₁ = 3 := 
sorry

-- Question (II)
theorem range_of_a (a : ℝ) : (∀ x ≥ 0, F x a ≥ F (-x) a) ↔ a ≤ 2 :=
sorry

end minimum_value_x2_minus_x1_range_of_a_l219_219130


namespace value_of_Y_l219_219317

-- Definitions for the conditions in part a)
def M := 2021 / 3
def N := M / 4
def Y := M + N

-- The theorem stating the question and its correct answer
theorem value_of_Y : Y = 843 := by
  sorry

end value_of_Y_l219_219317


namespace pentagon_area_l219_219502

theorem pentagon_area (a b c d e : ℤ) (O : 31 * 25 = 775) (H : 12^2 + 5^2 = 13^2) 
  (rect_side_lengths : (a, b, c, d, e) = (13, 19, 20, 25, 31)) :
  775 - 1/2 * 12 * 5 = 745 := 
by
  sorry

end pentagon_area_l219_219502


namespace value_of_y_l219_219833

theorem value_of_y (y : ℝ) (h : (y / 5) / 3 = 5 / (y / 3)) : y = 15 ∨ y = -15 :=
by
  sorry

end value_of_y_l219_219833


namespace triangle_area_is_14_l219_219067

def vector : Type := (ℝ × ℝ)
def a : vector := (4, -1)
def b : vector := (2 * 2, 2 * 3)

noncomputable def parallelogram_area (u v : vector) : ℝ :=
  let (ux, uy) := u
  let (vx, vy) := v
  abs (ux * vy - uy * vx)

noncomputable def triangle_area (u v : vector) : ℝ :=
  (parallelogram_area u v) / 2

theorem triangle_area_is_14 : triangle_area a b = 14 :=
by
  unfold a b triangle_area parallelogram_area
  sorry

end triangle_area_is_14_l219_219067


namespace simplify_expr_l219_219566

theorem simplify_expr (a : ℝ) (h : a > 1) : (1 - a) * (1 / (a - 1)).sqrt = -(a - 1).sqrt :=
sorry

end simplify_expr_l219_219566


namespace max_passengers_l219_219441

theorem max_passengers (total_stops : ℕ) (bus_capacity : ℕ)
  (h_total_stops : total_stops = 12) 
  (h_bus_capacity : bus_capacity = 20) 
  (h_no_same_stop : ∀ (a b : ℕ), a ≠ b → (a < total_stops) → (b < total_stops) → 
    ∃ x y : ℕ, x ≠ y ∧ x < total_stops ∧ y < total_stops ∧ 
    ((x = a ∧ y ≠ a) ∨ (x ≠ b ∧ y = b))) :
  ∃ max_passengers : ℕ, max_passengers = 50 :=
  sorry

end max_passengers_l219_219441


namespace find_f_zero_l219_219769

variable (f : ℝ → ℝ)

def odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x + 1) = -g (-x + 1)

def even_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x - 1) = g (-x - 1)

theorem find_f_zero
  (H1 : odd_function f)
  (H2 : even_function f)
  (H3 : f 4 = 6) :
  f 0 = -6 := by
  sorry

end find_f_zero_l219_219769


namespace find_cost_of_jersey_l219_219864

def cost_of_jersey (J : ℝ) : Prop := 
  let shorts_cost := 15.20
  let socks_cost := 6.80
  let total_players := 16
  let total_cost := 752
  total_players * (J + shorts_cost + socks_cost) = total_cost

theorem find_cost_of_jersey : cost_of_jersey 25 :=
  sorry

end find_cost_of_jersey_l219_219864


namespace g_of_5_l219_219114

theorem g_of_5 (g : ℝ → ℝ) (h : ∀ x ≠ 0, 4 * g x - 3 * g (1 / x) = 2 * x) :
  g 5 = 402 / 70 := 
sorry

end g_of_5_l219_219114


namespace hawks_total_points_l219_219373

def touchdowns : ℕ := 3
def points_per_touchdown : ℕ := 7
def total_points (t : ℕ) (p : ℕ) : ℕ := t * p

theorem hawks_total_points : total_points touchdowns points_per_touchdown = 21 := 
by 
  sorry

end hawks_total_points_l219_219373


namespace greatest_multiple_5_7_less_than_700_l219_219420

theorem greatest_multiple_5_7_less_than_700 :
  ∃ n, n < 700 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ (∀ m, m < 700 ∧ m % 5 = 0 ∧ m % 7 = 0 → m ≤ n) → n = 665 :=
by
  sorry

end greatest_multiple_5_7_less_than_700_l219_219420


namespace value_of_e_l219_219286

theorem value_of_e
  (a b c d e : ℤ)
  (h1 : b = a + 2)
  (h2 : c = a + 4)
  (h3 : d = a + 6)
  (h4 : e = a + 8)
  (h5 : a + c = 146) :
  e = 79 :=
  by sorry

end value_of_e_l219_219286


namespace present_age_of_eldest_is_45_l219_219821

theorem present_age_of_eldest_is_45 (x : ℕ) 
  (h1 : (5 * x - 10) + (7 * x - 10) + (8 * x - 10) + (9 * x - 10) = 107) :
  9 * x = 45 :=
sorry

end present_age_of_eldest_is_45_l219_219821


namespace negation_of_proposition_p_l219_219834

def has_real_root (m : ℝ) : Prop := ∃ x : ℝ, x^2 + m * x + 1 = 0

def negation_of_p : Prop := ∀ m : ℝ, ¬ has_real_root m

theorem negation_of_proposition_p : negation_of_p :=
by sorry

end negation_of_proposition_p_l219_219834


namespace triangle_inequality_of_three_l219_219563

theorem triangle_inequality_of_three (x y z : ℝ) :
  |x| + |y| + |z| ≤ |x + y - z| + |x - y + z| + |-x + y + z| := 
sorry

end triangle_inequality_of_three_l219_219563


namespace minimum_value_f_l219_219612

theorem minimum_value_f (x y : ℝ) (h : 2 * x^2 + 3 * x * y + 2 * y^2 = 1) : (∃ x y : ℝ, (x + y + x * y) ≥ - (9 / 8) ∧ 2 * x^2 + 3 * x * y + 2 * y^2 = 1) :=
sorry

end minimum_value_f_l219_219612


namespace students_in_section_B_l219_219786

variable (x : ℕ)

/-- There are 30 students in section A and the number of students in section B is x. The 
    average weight of section A is 40 kg, and the average weight of section B is 35 kg. 
    The average weight of the whole class is 38 kg. Prove that the number of students in
    section B is 20. -/
theorem students_in_section_B (h : 30 * 40 + x * 35 = 38 * (30 + x)) : x = 20 :=
  sorry

end students_in_section_B_l219_219786


namespace bales_stacked_correct_l219_219281

-- Given conditions
def initial_bales : ℕ := 28
def final_bales : ℕ := 82

-- Define the stacking function
def bales_stacked (initial final : ℕ) : ℕ := final - initial

-- Theorem statement we need to prove
theorem bales_stacked_correct : bales_stacked initial_bales final_bales = 54 := by
  sorry

end bales_stacked_correct_l219_219281


namespace width_decrease_l219_219057

-- Given conditions and known values
variable (L W : ℝ) -- original length and width
variable (P : ℝ)   -- percentage decrease in width

-- The known condition for the area comparison
axiom area_condition : 1.4 * (L * (W * (1 - P / 100))) = 1.1199999999999999 * (L * W)

-- The property we want to prove
theorem width_decrease (L W: ℝ) (h : L > 0) (h1 : W > 0) :
  P = 20 := 
by
  sorry

end width_decrease_l219_219057


namespace sum_of_invalid_domain_of_g_l219_219879

noncomputable def g (x : ℝ) : ℝ := 1 / (2 + (1 / (3 + (1 / x))))

theorem sum_of_invalid_domain_of_g : 
  (0 : ℝ) + (-1 / 3) + (-2 / 7) = -13 / 21 :=
by
  sorry

end sum_of_invalid_domain_of_g_l219_219879


namespace Asya_Petya_l219_219975

theorem Asya_Petya (a b : ℕ) (ha : 100 ≤ a ∧ a < 1000) (hb : 100 ≤ b ∧ b < 1000) 
  (h : 1000 * a + b = 7 * a * b) : a = 143 ∧ b = 143 :=
by
  sorry

end Asya_Petya_l219_219975


namespace fraction_problem_l219_219984

theorem fraction_problem :
  (1 / 4 + 3 / 8) - 1 / 8 = 1 / 2 :=
by
  -- The proof steps are skipped
  sorry

end fraction_problem_l219_219984


namespace square_of_real_is_positive_or_zero_l219_219530

def p (x : ℝ) : Prop := x^2 > 0
def q (x : ℝ) : Prop := x^2 = 0

theorem square_of_real_is_positive_or_zero (x : ℝ) : (p x ∨ q x) :=
by
  sorry

end square_of_real_is_positive_or_zero_l219_219530


namespace maximum_value_l219_219608

variables (a b c : ℝ)
variables (a_vec b_vec c_vec : EuclideanSpace ℝ (Fin 3))

axiom norm_a : ‖a_vec‖ = 2
axiom norm_b : ‖b_vec‖ = 3
axiom norm_c : ‖c_vec‖ = 4

theorem maximum_value : 
  (‖(a_vec - (3:ℝ) • b_vec)‖^2 + ‖(b_vec - (3:ℝ) • c_vec)‖^2 + ‖(c_vec - (3:ℝ) • a_vec)‖^2) ≤ 377 :=
by
  sorry

end maximum_value_l219_219608


namespace solve_for_y_l219_219699

theorem solve_for_y (y : ℝ) (h : (4/7) * (1/5) * y - 2 = 14) : y = 140 := 
sorry

end solve_for_y_l219_219699


namespace three_solutions_exists_l219_219981

theorem three_solutions_exists (n : ℕ) (h_pos : 0 < n) (h_sol : ∃ x y : ℤ, x^3 - 3 * x * y^2 + y^3 = n) :
  ∃ x1 y1 x2 y2 x3 y3 : ℤ, (x1^3 - 3 * x1 * y1^2 + y1^3 = n) ∧ (x2^3 - 3 * x2 * y2^2 + y2^3 = n) ∧ (x3^3 - 3 * x3 * y3^2 + y3^3 = n) ∧ (x1, y1) ≠ (x2, y2) ∧ (x2, y2) ≠ (x3, y3) ∧ (x1, y1) ≠ (x3, y3) :=
by
  sorry

end three_solutions_exists_l219_219981


namespace geometric_sequence_sum_inv_l219_219010

theorem geometric_sequence_sum_inv
  (a : ℕ → ℝ)
  (h1 : a 1 = 2)
  (h2 : a 1 + a 3 + a 5 = 14) :
  (1 / a 1) + (1 / a 3) + (1 / a 5) = 7 / 8 :=
by
  sorry

end geometric_sequence_sum_inv_l219_219010


namespace parabola_focus_coords_l219_219023

theorem parabola_focus_coords :
  ∀ (x y : ℝ), y^2 = -4 * x → (x, y) = (-1, 0) :=
by
  intros x y h
  sorry

end parabola_focus_coords_l219_219023


namespace amoeba_reproduction_time_l219_219074

/--
An amoeba reproduces by fission, splitting itself into two separate amoebae. 
It takes 8 days for one amoeba to divide into 16 amoebae. 

Prove that it takes 2 days for an amoeba to reproduce.
-/
theorem amoeba_reproduction_time (day_per_cycle : ℕ) (n_cycles : ℕ) 
  (h1 : n_cycles * day_per_cycle = 8)
  (h2 : 2^n_cycles = 16) : 
  day_per_cycle = 2 :=
by
  sorry

end amoeba_reproduction_time_l219_219074


namespace dog_farthest_distance_l219_219385

/-- 
Given a dog tied to a post at the point (3,4), a 15 meter long rope, and a wall from (5,4) to (5,9), 
prove that the farthest distance the dog can travel from the origin (0,0) is 20 meters.
-/
theorem dog_farthest_distance (post : ℝ × ℝ) (rope_length : ℝ) (wall_start wall_end origin : ℝ × ℝ)
  (h_post : post = (3,4))
  (h_rope_length : rope_length = 15)
  (h_wall_start : wall_start = (5,4))
  (h_wall_end : wall_end = (5,9))
  (h_origin : origin = (0,0)) :
  ∃ farthest_distance : ℝ, farthest_distance = 20 :=
by
  sorry

end dog_farthest_distance_l219_219385


namespace expand_expression_l219_219087

theorem expand_expression (y : ℚ) : 5 * (4 * y^3 - 3 * y^2 + 2 * y - 6) = 20 * y^3 - 15 * y^2 + 10 * y - 30 := by
  sorry

end expand_expression_l219_219087


namespace meeting_percentage_l219_219418

theorem meeting_percentage
    (workday_hours : ℕ)
    (first_meeting_minutes : ℕ)
    (second_meeting_factor : ℕ)
    (hp_workday_hours : workday_hours = 10)
    (hp_first_meeting_minutes : first_meeting_minutes = 60)
    (hp_second_meeting_factor : second_meeting_factor = 2) 
    : (first_meeting_minutes + first_meeting_minutes * second_meeting_factor : ℚ) 
    / (workday_hours * 60) * 100 = 30 := 
by
  have workday_minutes := workday_hours * 60
  have second_meeting_minutes := first_meeting_minutes * second_meeting_factor
  have total_meeting_minutes := first_meeting_minutes + second_meeting_minutes
  have percentage := (total_meeting_minutes : ℚ) / workday_minutes * 100
  sorry

end meeting_percentage_l219_219418


namespace time_to_fill_pool_l219_219483

theorem time_to_fill_pool :
  ∀ (total_volume : ℝ) (filling_rate : ℝ) (leaking_rate : ℝ),
  total_volume = 60 →
  filling_rate = 1.6 →
  leaking_rate = 0.1 →
  (total_volume / (filling_rate - leaking_rate)) = 40 :=
by
  intros total_volume filling_rate leaking_rate hv hf hl
  rw [hv, hf, hl]
  sorry

end time_to_fill_pool_l219_219483


namespace joe_speed_first_part_l219_219994

theorem joe_speed_first_part (v : ℝ) :
  let d1 := 420 -- distance of the first part in miles
  let d2 := 120 -- distance of the second part in miles
  let v2 := 40  -- speed during the second part in miles per hour
  let d_total := d1 + d2 -- total distance
  let avg_speed := 54 -- average speed in miles per hour
  let t1 := d1 / v -- time for the first part
  let t2 := d2 / v2 -- time for the second part
  let t_total := t1 + t2 -- total time
  (d_total / t_total) = avg_speed -> v = 60 :=
by
  intros
  sorry

end joe_speed_first_part_l219_219994


namespace three_digit_number_base_10_l219_219047

theorem three_digit_number_base_10 (A B C : ℕ) (x : ℕ)
  (h1 : x = 100 * A + 10 * B + 6)
  (h2 : x = 82 * C + 36)
  (hA : 1 ≤ A ∧ A ≤ 9)
  (hB : 0 ≤ B ∧ B ≤ 9)
  (hC : 0 ≤ C ∧ C ≤ 8) :
  x = 446 := by
  sorry

end three_digit_number_base_10_l219_219047


namespace problem_solution_l219_219460

noncomputable def sqrt_3_simplest : Prop :=
  let A := Real.sqrt 3
  let B := Real.sqrt 0.5
  let C := Real.sqrt 8
  let D := Real.sqrt (1 / 3)
  ∀ (x : ℝ), x = A ∨ x = B ∨ x = C ∨ x = D → x = A → 
    (x = Real.sqrt 0.5 ∨ x = Real.sqrt 8 ∨ x = Real.sqrt (1 / 3)) ∧ 
    ¬(x = Real.sqrt 0.5 ∨ x = 2 * Real.sqrt 2 ∨ x = Real.sqrt (1 / 3))

theorem problem_solution : sqrt_3_simplest :=
by
  sorry

end problem_solution_l219_219460


namespace opposite_sides_line_l219_219142

theorem opposite_sides_line (a : ℝ) : (0 + 0 - a) * (1 + 1 - a) < 0 → 0 < a ∧ a < 2 := by
  sorry

end opposite_sides_line_l219_219142


namespace number_of_grandchildren_l219_219398

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

end number_of_grandchildren_l219_219398


namespace greatest_number_of_consecutive_integers_whose_sum_is_36_l219_219805

/-- 
Given that the sum of N consecutive integers starting from a is 36, 
prove that the greatest possible value of N is 72.
-/
theorem greatest_number_of_consecutive_integers_whose_sum_is_36 :
  ∀ (N a : ℤ), (N > 0) → (N * (2 * a + N - 1)) = 72 → N ≤ 72 := 
by
  intros N a hN h
  sorry

end greatest_number_of_consecutive_integers_whose_sum_is_36_l219_219805


namespace correct_answer_l219_219150

def M : Set ℕ := {1}
def N : Set ℕ := {1, 2, 3}

theorem correct_answer : M ⊆ N := by
  sorry

end correct_answer_l219_219150


namespace find_real_medal_min_weighings_l219_219944

axiom has_9_medals : Prop
axiom one_real_medal : Prop
axiom real_medal_heavier : Prop
axiom has_balance_scale : Prop

theorem find_real_medal_min_weighings
  (h1 : has_9_medals)
  (h2 : one_real_medal)
  (h3 : real_medal_heavier)
  (h4 : has_balance_scale) :
  ∃ (minimum_weighings : ℕ), minimum_weighings = 2 := 
  sorry

end find_real_medal_min_weighings_l219_219944


namespace trajectory_of_P_is_right_branch_of_hyperbola_l219_219333

-- Definitions of the given points F1 and F2
def F1 : ℝ × ℝ := (-5, 0)
def F2 : ℝ × ℝ := (5, 0)

-- Definition of the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2).sqrt

-- Definition of point P satisfying the condition
def P (x y : ℝ) : Prop :=
  abs (distance (x, y) F1 - distance (x, y) F2) = 8

-- Trajectory of point P is the right branch of the hyperbola
theorem trajectory_of_P_is_right_branch_of_hyperbola :
  ∀ (x y : ℝ), P x y → True := -- Trajectory is hyperbola (right branch)
by
  sorry

end trajectory_of_P_is_right_branch_of_hyperbola_l219_219333


namespace sequence_x_sequence_y_sequence_z_sequence_t_l219_219450

theorem sequence_x (n : ℕ) : 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 → 
  (if n = 1 then (n^2 + n = 2) else 
   if n = 2 then (n^2 + n = 6) else 
   if n = 3 then (n^2 + n = 12) else 
   if n = 4 then (n^2 + n = 20) else true) := 
by sorry

theorem sequence_y (n : ℕ) : 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 → 
  (if n = 1 then (2 * n^2 = 2) else 
   if n = 2 then (2 * n^2 = 8) else 
   if n = 3 then (2 * n^2 = 18) else 
   if n = 4 then (2 * n^2 = 32) else true) := 
by sorry

theorem sequence_z (n : ℕ) : 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 → 
  (if n = 1 then (n^3 = 1) else 
   if n = 2 then (n^3 = 8) else 
   if n = 3 then (n^3 = 27) else 
   if n = 4 then (n^3 = 64) else true) := 
by sorry

theorem sequence_t (n : ℕ) : 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 → 
  (if n = 1 then (2^n = 2) else 
   if n = 2 then (2^n = 4) else 
   if n = 3 then (2^n = 8) else 
   if n = 4 then (2^n = 16) else true) := 
by sorry

end sequence_x_sequence_y_sequence_z_sequence_t_l219_219450


namespace smallest_positive_integer_satisfying_condition_l219_219937

-- Define the condition
def isConditionSatisfied (n : ℕ) : Prop :=
  (Real.sqrt n - Real.sqrt (n - 1) < 0.01) ∧ n > 0

-- State the theorem
theorem smallest_positive_integer_satisfying_condition :
  ∃ n : ℕ, isConditionSatisfied n ∧ (∀ m : ℕ, isConditionSatisfied m → n ≤ m) ∧ n = 2501 :=
by
  sorry

end smallest_positive_integer_satisfying_condition_l219_219937


namespace age_sum_is_ninety_l219_219636

theorem age_sum_is_ninety (a b c : ℕ)
  (h1 : a = 20 + b + c)
  (h2 : a^2 = 1800 + (b + c)^2) :
  a + b + c = 90 := 
sorry

end age_sum_is_ninety_l219_219636


namespace infinite_primes_dividing_expression_l219_219751

theorem infinite_primes_dividing_expression (k : ℕ) (hk : k > 0) : 
  ∃ᶠ p in Filter.atTop, ∃ n : ℕ, p ∣ (2017^n + k) :=
sorry

end infinite_primes_dividing_expression_l219_219751


namespace apples_picked_l219_219788

theorem apples_picked (n_a : ℕ) (k_a : ℕ) (total : ℕ) (m_a : ℕ) (h_n : n_a = 3) (h_k : k_a = 6) (h_t : total = 16) :
  m_a = total - (n_a + k_a) →
  m_a = 7 :=
by
  sorry

end apples_picked_l219_219788


namespace max_product_sum_1976_l219_219673

theorem max_product_sum_1976 (a : ℕ) (P : ℕ → ℕ) (h : ∀ n, P n > 0 → a = 1976) :
  ∃ (k l : ℕ), (2 * k + 3 * l = 1976) ∧ (P 1976 = 2 * 3 ^ 658) := sorry

end max_product_sum_1976_l219_219673


namespace number_of_men_in_first_group_l219_219052

theorem number_of_men_in_first_group (M : ℕ) : (M * 15 = 25 * 18) → M = 30 :=
by
  sorry

end number_of_men_in_first_group_l219_219052


namespace face_value_is_100_l219_219736

-- Definitions based on conditions
def faceValue (F : ℝ) : Prop :=
  let discountedPrice := 0.92 * F
  let brokerageFee := 0.002 * discountedPrice
  let totalCostPrice := discountedPrice + brokerageFee
  totalCostPrice = 92.2

-- The proof statement in Lean
theorem face_value_is_100 : ∃ F : ℝ, faceValue F ∧ F = 100 :=
by
  use 100
  unfold faceValue
  simp
  norm_num
  sorry

end face_value_is_100_l219_219736


namespace P_intersection_complement_Q_l219_219587

-- Define sets P and Q
def P : Set ℝ := { x | 0 ≤ x ∧ x ≤ 3 }
def Q : Set ℝ := { x | x^2 ≥ 4 }

-- Prove the required intersection
theorem P_intersection_complement_Q : P ∩ (Set.univ \ Q) = { x | 0 ≤ x ∧ x < 2 } :=
by
  -- Proof will be inserted here
  sorry

end P_intersection_complement_Q_l219_219587


namespace graph_transform_l219_219063

-- Define the quadratic function y1 as y = -2x^2 + 4x + 1
def y1 (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 1

-- Define the quadratic function y2 as y = -2x^2
def y2 (x : ℝ) : ℝ := -2 * x^2

-- Define the transformation function for moving 1 unit to the left and 3 units down
def transform (y : ℝ → ℝ) (x : ℝ) : ℝ := y (x + 1) - 3

-- Statement to prove
theorem graph_transform : ∀ x : ℝ, transform y1 x = y2 x :=
by
  intros x
  sorry

end graph_transform_l219_219063


namespace expression_value_at_neg1_l219_219380

theorem expression_value_at_neg1
  (p q : ℤ)
  (h1 : p + q = 2016) :
  p * (-1)^3 + q * (-1) - 10 = -2026 := by
  sorry

end expression_value_at_neg1_l219_219380


namespace sufficient_and_necessary_condition_l219_219782

theorem sufficient_and_necessary_condition (x : ℝ) :
  (x - 2) * (x + 2) > 0 ↔ x > 2 ∨ x < -2 :=
by sorry

end sufficient_and_necessary_condition_l219_219782


namespace train_passing_time_l219_219689

def train_distance_km : ℝ := 10
def train_time_min : ℝ := 15
def train_length_m : ℝ := 111.11111111111111

theorem train_passing_time : 
  let time_to_pass_signal_post := train_length_m / ((train_distance_km * 1000) / (train_time_min * 60))
  time_to_pass_signal_post = 10 :=
by
  sorry

end train_passing_time_l219_219689


namespace student_age_is_24_l219_219236

/-- A man is 26 years older than his student. In two years, his age will be twice the age of his student.
    Prove that the present age of the student is 24 years old. -/
theorem student_age_is_24 (S M : ℕ) (h1 : M = S + 26) (h2 : M + 2 = 2 * (S + 2)) : S = 24 :=
by
  sorry

end student_age_is_24_l219_219236


namespace sets_equal_l219_219743

theorem sets_equal :
  let M := {x | x^2 - 2 * x + 1 = 0}
  let N := {1}
  M = N :=
by
  sorry

end sets_equal_l219_219743


namespace solution_to_ball_problem_l219_219214

noncomputable def probability_of_arithmetic_progression : Nat :=
  let p := 3
  let q := 9464
  p + q

theorem solution_to_ball_problem : probability_of_arithmetic_progression = 9467 := by
  sorry

end solution_to_ball_problem_l219_219214


namespace eval_expression_pow_i_l219_219501

theorem eval_expression_pow_i :
  i^(12345 : ℤ) + i^(12346 : ℤ) + i^(12347 : ℤ) + i^(12348 : ℤ) = (0 : ℂ) :=
by
  -- Since this statement doesn't need the full proof, we use sorry to leave it open 
  sorry

end eval_expression_pow_i_l219_219501


namespace cheeseburger_cost_l219_219019

-- Definitions for given conditions
def milkshake_price : ℝ := 5
def cheese_fries_price : ℝ := 8
def jim_money : ℝ := 20
def cousin_money : ℝ := 10
def combined_money := jim_money + cousin_money
def spending_percentage : ℝ := 0.80
def total_spent := spending_percentage * combined_money
def number_of_milkshakes : ℝ := 2
def number_of_cheeseburgers : ℝ := 2

-- Prove the cost of one cheeseburger
theorem cheeseburger_cost : (total_spent - (number_of_milkshakes * milkshake_price) - cheese_fries_price) / number_of_cheeseburgers = 3 :=
by
  sorry

end cheeseburger_cost_l219_219019


namespace union_of_sets_l219_219500

-- Define the sets and conditions
variables (a b : ℝ)
variables (A : Set ℝ) (B : Set ℝ)
variables (log2 : ℝ → ℝ)

-- State the assumptions and final proof goal
theorem union_of_sets (h_inter : A ∩ B = {2}) 
                      (h_A : A = {3, log2 a}) 
                      (h_B : B = {a, b}) 
                      (h_log2 : log2 4 = 2) :
  A ∪ B = {2, 3, 4} :=
by {
    sorry
}

end union_of_sets_l219_219500


namespace find_a_b_l219_219141

noncomputable def f (a b x : ℝ) := b * a^x

def passes_through (a b : ℝ) : Prop :=
  f a b 1 = 27 ∧ f a b (-1) = 3

theorem find_a_b (a b : ℝ) (h : passes_through a b) : 
  a = 3 ∧ b = 9 :=
  sorry

end find_a_b_l219_219141


namespace maxwell_distance_when_meeting_l219_219838

variable (total_distance : ℝ := 50)
variable (maxwell_speed : ℝ := 4)
variable (brad_speed : ℝ := 6)
variable (t : ℝ := total_distance / (maxwell_speed + brad_speed))

theorem maxwell_distance_when_meeting :
  (maxwell_speed * t = 20) :=
by
  sorry

end maxwell_distance_when_meeting_l219_219838


namespace function_domain_l219_219495

noncomputable def domain_function : Set ℝ :=
  {x : ℝ | x ≠ 8}

theorem function_domain :
  ∀ x, x ∈ domain_function ↔ x ∈ (Set.Iio 8 ∪ Set.Ioi 8) := by
  intro x
  sorry

end function_domain_l219_219495


namespace material_for_one_pillowcase_l219_219740

def material_in_first_bale (x : ℝ) : Prop :=
  4 * x + 1100 = 5000

def material_in_third_bale : ℝ := 0.22 * 5000

def total_material_used_for_producing_items (x y : ℝ) : Prop :=
  150 * (y + 3.25) + 240 * y = x

theorem material_for_one_pillowcase :
  ∀ (x y : ℝ), 
    material_in_first_bale x → 
    material_in_third_bale = 1100 → 
    (x = 975) → 
    total_material_used_for_producing_items x y →
    y = 1.25 :=
by
  intro x y h1 h2 h3 h4
  rw [h3] at h4
  have : 150 * (y + 3.25) + 240 * y = 975 := h4
  sorry

end material_for_one_pillowcase_l219_219740


namespace count_lattice_right_triangles_with_incenter_l219_219215

def is_lattice_point (p : ℤ × ℤ) : Prop := ∃ x y : ℤ, p = (x, y)

def is_right_triangle (O A B : ℤ × ℤ) : Prop :=
  O = (0, 0) ∧ (O.1 = A.1 ∨ O.2 = A.2) ∧ (O.1 = B.1 ∨ O.2 = B.2) ∧
  (A.1 * B.2 - A.2 * B.1 ≠ 0) -- Ensure A and B are not collinear with O

def incenter (O A B : ℤ × ℤ) : ℤ × ℤ :=
  ((A.1 + B.1 - O.1) / 2, (A.2 + B.2 - O.2) / 2)

theorem count_lattice_right_triangles_with_incenter :
  let I := (2015, 7 * 2015)
  ∃ (O A B : ℤ × ℤ), is_right_triangle O A B ∧ incenter O A B = I :=
sorry

end count_lattice_right_triangles_with_incenter_l219_219215


namespace num_positive_integers_l219_219247

theorem num_positive_integers (N : ℕ) (h : N > 3) : (∃ (k : ℕ) (h_div : 48 % k = 0), k = N - 3) → (∃ (c : ℕ), c = 8) := sorry

end num_positive_integers_l219_219247


namespace first_player_winning_strategy_l219_219784

-- Defining the type for the positions on the chessboard
structure Position where
  x : Nat
  y : Nat
  deriving DecidableEq

-- Initial position C1
def C1 : Position := ⟨3, 1⟩

-- Winning position H8
def H8 : Position := ⟨8, 8⟩

-- Function to check if a position is a winning position
-- the target winning position is H8
def isWinningPosition (p : Position) : Bool :=
  p = H8

-- Function to determine the next possible positions
-- from the current position based on the allowed moves
def nextPositions (p : Position) : List Position :=
  (List.range (8 - p.x)).map (λ dx => ⟨p.x + dx + 1, p.y⟩) ++
  (List.range (8 - p.y)).map (λ dy => ⟨p.x, p.y + dy + 1⟩) ++
  (List.range (min (8 - p.x) (8 - p.y))).map (λ d => ⟨p.x + d + 1, p.y + d + 1⟩)

-- Statement of the problem: First player has a winning strategy from C1
theorem first_player_winning_strategy : 
  ∃ move : Position, move ∈ nextPositions C1 ∧
  ∀ next_move : Position, next_move ∈ nextPositions move → isWinningPosition next_move :=
sorry

end first_player_winning_strategy_l219_219784


namespace select_p_elements_with_integer_mean_l219_219477

theorem select_p_elements_with_integer_mean {p : ℕ} (hp : Nat.Prime p) (p_odd : p % 2 = 1) :
  ∃ (M : Finset ℕ), (M.card = (p^2 + 1) / 2) ∧ ∃ (S : Finset ℕ), (S.card = p) ∧ ((S.sum id) % p = 0) :=
by
  -- sorry to skip the proof
  sorry

end select_p_elements_with_integer_mean_l219_219477


namespace license_plate_count_l219_219470

/-- Number of vowels available for the license plate -/
def num_vowels := 6

/-- Number of consonants available for the license plate -/
def num_consonants := 20

/-- Number of possible digits for the license plate -/
def num_digits := 10

/-- Number of special characters available for the license plate -/
def num_special_chars := 2

/-- Calculate the total number of possible license plates -/
def total_license_plates : Nat :=
  num_vowels * num_consonants * num_digits * num_consonants * num_special_chars

/- Prove that the total number of possible license plates is 48000 -/
theorem license_plate_count : total_license_plates = 48000 :=
  by
    unfold total_license_plates
    sorry

end license_plate_count_l219_219470


namespace inequality_solution_set_l219_219535

theorem inequality_solution_set (m : ℝ) : 
  (∀ (x : ℝ), m * x^2 - (1 - m) * x + m ≥ 0) ↔ m ≥ 1/3 := 
sorry

end inequality_solution_set_l219_219535


namespace find_x_l219_219262

theorem find_x (x : ℝ) : (x / 18) * (36 / 72) = 1 → x = 36 :=
by
  intro h
  sorry

end find_x_l219_219262


namespace height_of_spruce_tree_l219_219351

theorem height_of_spruce_tree (t : ℚ) (h1 : t = 25 / 64) :
  (∃ s : ℚ, s = 3 / (1 - t) ∧ s = 64 / 13) :=
by
  sorry

end height_of_spruce_tree_l219_219351


namespace cubic_roots_geometric_progression_l219_219780

theorem cubic_roots_geometric_progression 
  (a r : ℝ)
  (h_roots: 27 * a^3 * r^3 - 81 * a^2 * r^2 + 63 * a * r - 14 = 0)
  (h_sum: a + a * r + a * r^2 = 3)
  (h_product: a^3 * r^3 = 14 / 27)
  : (max (a^2) ((a * r^2)^2) - min (a^2) ((a * r^2)^2) = 5 / 3) := 
sorry

end cubic_roots_geometric_progression_l219_219780


namespace fred_total_cards_l219_219126

theorem fred_total_cards 
  (initial_cards : ℕ := 26) 
  (cards_given_to_mary : ℕ := 18) 
  (unopened_box_cards : ℕ := 40) : 
  initial_cards - cards_given_to_mary + unopened_box_cards = 48 := 
by 
  sorry

end fred_total_cards_l219_219126


namespace like_terms_m_eq_2_l219_219327

theorem like_terms_m_eq_2 (m : ℕ) :
  (∀ (x y : ℝ), 3 * x^m * y^3 = 3 * x^2 * y^3) -> m = 2 :=
by
  intros _
  sorry

end like_terms_m_eq_2_l219_219327


namespace regular_hexagon_interior_angle_measure_l219_219781

theorem regular_hexagon_interior_angle_measure :
  let n := 6
  let sum_of_angles := (n - 2) * 180
  let measure_of_each_angle := sum_of_angles / n
  measure_of_each_angle = 120 :=
by
  sorry

end regular_hexagon_interior_angle_measure_l219_219781


namespace retailer_markup_percentage_l219_219196

-- Definitions of initial conditions
def CP : ℝ := 100
def intended_profit_percentage : ℝ := 0.25
def discount_percentage : ℝ := 0.25
def actual_profit_percentage : ℝ := 0.2375

-- Proving the retailer marked his goods at 65% above the cost price
theorem retailer_markup_percentage : ∃ (MP : ℝ), ((0.75 * MP - CP) / CP) * 100 = actual_profit_percentage * 100 ∧ ((MP - CP) / CP) * 100 = 65 := 
by
  -- The mathematical proof steps mean to be filled here  
  sorry

end retailer_markup_percentage_l219_219196


namespace intersection_condition_sufficient_but_not_necessary_l219_219853

theorem intersection_condition_sufficient_but_not_necessary (k : ℝ) :
  (-Real.sqrt 3 / 3 < k ∧ k < Real.sqrt 3 / 3) →
  ((∃ x : ℝ, (k^2 + 1) * x^2 + (2 * k^2 - 2) * x + k^2 = 0) ∧ 
   ¬ (∃ k, (∃ x : ℝ, (k^2 + 1) * x^2 + (2 * k^2 - 2) * x + k^2 = 0) → 
   (¬ (-Real.sqrt 3 / 3 < k ∧ k < Real.sqrt 3 / 3)))) :=
sorry

end intersection_condition_sufficient_but_not_necessary_l219_219853


namespace sum_of_coefficients_l219_219030

theorem sum_of_coefficients (d : ℤ) (h : d ≠ 0) :
    let a := 3 + 2
    let b := 17 + 2
    let c := 10 + 5
    let e := 16 + 4
    a + b + c + e = 59 :=
by
  let a := 3 + 2
  let b := 17 + 2
  let c := 10 + 5
  let e := 16 + 4
  sorry

end sum_of_coefficients_l219_219030


namespace geom_arith_sequence_l219_219513

theorem geom_arith_sequence (a b c m n : ℝ) 
  (h1 : b^2 = a * c) 
  (h2 : m = (a + b) / 2) 
  (h3 : n = (b + c) / 2) : 
  a / m + c / n = 2 := 
by 
  sorry

end geom_arith_sequence_l219_219513


namespace stock_price_after_two_years_l219_219253

theorem stock_price_after_two_years 
    (p0 : ℝ) (r1 r2 : ℝ) (p1 p2 : ℝ) 
    (h0 : p0 = 100) (h1 : r1 = 0.50) 
    (h2 : r2 = 0.30) 
    (h3 : p1 = p0 * (1 + r1)) 
    (h4 : p2 = p1 * (1 - r2)) : 
    p2 = 105 :=
by sorry

end stock_price_after_two_years_l219_219253


namespace jia_jia_clover_count_l219_219117

theorem jia_jia_clover_count : ∃ x : ℕ, 3 * x + 4 = 100 ∧ x = 32 := by
  sorry

end jia_jia_clover_count_l219_219117
