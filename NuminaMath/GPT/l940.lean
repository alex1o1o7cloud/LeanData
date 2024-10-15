import Mathlib

namespace NUMINAMATH_GPT_parallel_lines_of_equation_l940_94088

theorem parallel_lines_of_equation (y : Real) :
  (y - 2) * (y + 3) = 0 → (y = 2 ∨ y = -3) :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_of_equation_l940_94088


namespace NUMINAMATH_GPT_f_19_eq_2017_l940_94066

noncomputable def f : ℤ → ℤ := sorry

axiom f_condition1 : f 1 = 1
axiom f_condition2 : ∀ m n : ℤ, f (m + n) = f m + f n + 3 * (4 * m * n - 1)

theorem f_19_eq_2017 : f 19 = 2017 := by
  sorry

end NUMINAMATH_GPT_f_19_eq_2017_l940_94066


namespace NUMINAMATH_GPT_max_u_plus_2v_l940_94080

theorem max_u_plus_2v (u v : ℝ) (h1 : 2 * u + 3 * v ≤ 10) (h2 : 4 * u + v ≤ 9) : u + 2 * v ≤ 6.1 :=
sorry

end NUMINAMATH_GPT_max_u_plus_2v_l940_94080


namespace NUMINAMATH_GPT_more_red_than_yellow_l940_94006

-- Define the number of bouncy balls per pack
def bouncy_balls_per_pack : ℕ := 18

-- Define the number of packs Jill bought
def packs_red : ℕ := 5
def packs_yellow : ℕ := 4

-- Define the total number of bouncy balls purchased for each color
def total_red : ℕ := bouncy_balls_per_pack * packs_red
def total_yellow : ℕ := bouncy_balls_per_pack * packs_yellow

-- The theorem statement indicating how many more red bouncy balls than yellow bouncy balls Jill bought
theorem more_red_than_yellow : total_red - total_yellow = 18 := by
  sorry

end NUMINAMATH_GPT_more_red_than_yellow_l940_94006


namespace NUMINAMATH_GPT_number_of_boxes_in_each_case_l940_94005

theorem number_of_boxes_in_each_case (a b : ℕ) :
    a + b = 2 → 9 = a * 8 + b :=
by
    intro h
    sorry

end NUMINAMATH_GPT_number_of_boxes_in_each_case_l940_94005


namespace NUMINAMATH_GPT_train_crossing_time_l940_94038

-- Define the length of the train
def train_length : ℝ := 120

-- Define the speed of the train
def train_speed : ℝ := 15

-- Define the target time to cross the man
def target_time : ℝ := 8

-- Proposition to prove
theorem train_crossing_time :
  target_time = train_length / train_speed :=
by
  sorry

end NUMINAMATH_GPT_train_crossing_time_l940_94038


namespace NUMINAMATH_GPT_derek_dogs_l940_94056

theorem derek_dogs (d c : ℕ) (h1 : d = 90) 
  (h2 : c = d / 3) 
  (h3 : c + 210 = 2 * (d + 120 - d)) : 
  d + 120 - d = 120 :=
by
  sorry

end NUMINAMATH_GPT_derek_dogs_l940_94056


namespace NUMINAMATH_GPT_relationship_among_f_l940_94039

theorem relationship_among_f (
  f : ℝ → ℝ
) (h_even : ∀ x, f x = f (-x))
  (h_periodic : ∀ x, f (x - 1) = f (x + 1))
  (h_increasing : ∀ a b, (0 ≤ a ∧ a < b ∧ b ≤ 1) → f a < f b) :
  f 2 < f (-5.5) ∧ f (-5.5) < f (-1) :=
by
  sorry

end NUMINAMATH_GPT_relationship_among_f_l940_94039


namespace NUMINAMATH_GPT_abs_neg_two_eq_two_l940_94087

theorem abs_neg_two_eq_two : abs (-2) = 2 :=
sorry

end NUMINAMATH_GPT_abs_neg_two_eq_two_l940_94087


namespace NUMINAMATH_GPT_chairs_per_row_l940_94073

-- Definition of the given conditions
def rows : ℕ := 20
def people_per_chair : ℕ := 5
def total_people : ℕ := 600

-- The statement to be proven
theorem chairs_per_row (x : ℕ) (h : rows * (x * people_per_chair) = total_people) : x = 6 := 
by sorry

end NUMINAMATH_GPT_chairs_per_row_l940_94073


namespace NUMINAMATH_GPT_sum_of_solutions_l940_94037

-- Define the polynomial equation and the condition
def equation (x : ℝ) : Prop := 3 = (x^3 - 3 * x^2 - 12 * x) / (x + 3)

-- Sum of solutions for the given polynomial equation under the constraint
theorem sum_of_solutions :
  (∀ x : ℝ, equation x → x ≠ -3) →
  ∃ (a b : ℝ), equation a ∧ equation b ∧ a + b = 4 := 
by
  intros h
  sorry

end NUMINAMATH_GPT_sum_of_solutions_l940_94037


namespace NUMINAMATH_GPT_money_conditions_l940_94094

theorem money_conditions (a b : ℝ) (h1 : 4 * a - b > 32) (h2 : 2 * a + b = 26) : 
  a > 9.67 ∧ b < 6.66 := 
sorry

end NUMINAMATH_GPT_money_conditions_l940_94094


namespace NUMINAMATH_GPT_johnny_tables_l940_94017

theorem johnny_tables :
  ∀ (T : ℕ),
  (∀ (T : ℕ), 4 * T + 5 * T = 45) →
  T = 5 :=
  sorry

end NUMINAMATH_GPT_johnny_tables_l940_94017


namespace NUMINAMATH_GPT_proof_problem_l940_94019

def f (x : ℤ) : ℤ := 3 * x + 5
def g (x : ℤ) : ℤ := 4 * x - 3

theorem proof_problem : 
  (f (g (f (g 3)))) / (g (f (g (f 3)))) = (380 / 653) := 
  by 
    sorry

end NUMINAMATH_GPT_proof_problem_l940_94019


namespace NUMINAMATH_GPT_probability_heads_odd_l940_94004

theorem probability_heads_odd (n : ℕ) (p : ℚ) (Q : ℕ → ℚ) (h : p = 3/4) (h_rec : ∀ n, Q (n + 1) = p * (1 - Q n) + (1 - p) * Q n) :
  Q 40 = 1/2 * (1 - 1/4^40) := 
sorry

end NUMINAMATH_GPT_probability_heads_odd_l940_94004


namespace NUMINAMATH_GPT_milk_needed_for_cookies_l940_94029

-- Definition of the problem conditions
def cookies_per_milk_usage := 24
def milk_in_liters := 5
def liters_to_milliliters := 1000
def milk_for_6_cookies := 1250

-- Prove that 1250 milliliters of milk are needed to bake 6 cookies
theorem milk_needed_for_cookies
  (h1 : cookies_per_milk_usage = 24)
  (h2 : milk_in_liters = 5)
  (h3 : liters_to_milliliters = 1000) :
  milk_for_6_cookies = 1250 :=
by
  -- Proof is omitted with sorry
  sorry

end NUMINAMATH_GPT_milk_needed_for_cookies_l940_94029


namespace NUMINAMATH_GPT_solve_fractional_eq_l940_94028

theorem solve_fractional_eq (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 0) : 
  (2 / (x - 1) = 1 / x) ↔ (x = -1) :=
by 
  sorry

end NUMINAMATH_GPT_solve_fractional_eq_l940_94028


namespace NUMINAMATH_GPT_aras_current_height_l940_94021

-- Define the variables and conditions
variables (x : ℝ) (sheas_original_height : ℝ := x) (ars_original_height : ℝ := x)
variables (sheas_growth_factor : ℝ := 0.30) (sheas_current_height : ℝ := 65)
variables (sheas_growth : ℝ := sheas_current_height - sheas_original_height)
variables (aras_growth : ℝ := sheas_growth / 3)

-- Define a theorem for Ara's current height
theorem aras_current_height (h1 : sheas_current_height = (1 + sheas_growth_factor) * sheas_original_height)
                           (h2 : sheas_original_height = ars_original_height) :
                           aras_growth + ars_original_height = 55 :=
by
  sorry

end NUMINAMATH_GPT_aras_current_height_l940_94021


namespace NUMINAMATH_GPT_distance_from_P_to_y_axis_l940_94057

theorem distance_from_P_to_y_axis 
  (x y : ℝ)
  (h1 : (x^2 / 16) + (y^2 / 25) = 1)
  (F1 : ℝ × ℝ := (0, -3))
  (F2 : ℝ × ℝ := (0, 3))
  (h2 : (F1.1 - x)^2 + (F1.2 - y)^2 = 9 ∨ (F2.1 - x)^2 + (F2.2 - y)^2 = 9 
          ∨ (F1.1 - x)^2 + (F1.2 - y)^2 + (F2.1 - x)^2 + (F2.2 - y)^2 = (F1.1 - F2.1)^2 + (F1.2 - F2.2)^2) :
  |x| = 16 / 5 :=
by
  sorry

end NUMINAMATH_GPT_distance_from_P_to_y_axis_l940_94057


namespace NUMINAMATH_GPT_solve_equation_l940_94011

theorem solve_equation (x : ℝ) (h : x + 3 ≠ 0) : (2 / (x + 3) = 1) → (x = -1) :=
by
  intro h1
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_solve_equation_l940_94011


namespace NUMINAMATH_GPT_total_jellybeans_l940_94034

theorem total_jellybeans (G : ℕ) (H1 : G = 8 + 2) (H2 : ∀ O : ℕ, O = G - 1) : 
  8 + G + (G - 1) = 27 := 
by 
  sorry

end NUMINAMATH_GPT_total_jellybeans_l940_94034


namespace NUMINAMATH_GPT_find_g_2_l940_94042

noncomputable def g (x : ℝ) : ℝ := sorry

axiom functional_eq (x : ℝ) (hx : x ≠ 0) : 2 * g x - 3 * g (1 / x) = x ^ 2

theorem find_g_2 : g 2 = 8.25 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_g_2_l940_94042


namespace NUMINAMATH_GPT_Maggie_age_l940_94041

theorem Maggie_age (Kate Maggie Sue : ℕ) (h1 : Kate + Maggie + Sue = 48) (h2 : Kate = 19) (h3 : Sue = 12) : Maggie = 17 := by
  sorry

end NUMINAMATH_GPT_Maggie_age_l940_94041


namespace NUMINAMATH_GPT_executive_board_elections_l940_94059

noncomputable def num_candidates : ℕ := 18
noncomputable def num_positions : ℕ := 6
noncomputable def num_former_board_members : ℕ := 8

noncomputable def total_selections := Nat.choose num_candidates num_positions
noncomputable def no_former_board_members_selections := Nat.choose (num_candidates - num_former_board_members) num_positions

noncomputable def valid_selections := total_selections - no_former_board_members_selections

theorem executive_board_elections : valid_selections = 18354 :=
by sorry

end NUMINAMATH_GPT_executive_board_elections_l940_94059


namespace NUMINAMATH_GPT_clara_age_l940_94067

theorem clara_age (x : ℕ) (n m : ℕ) (h1 : x - 2 = n^2) (h2 : x + 3 = m^3) : x = 123 :=
by sorry

end NUMINAMATH_GPT_clara_age_l940_94067


namespace NUMINAMATH_GPT_stratified_sampling_third_grade_l940_94036

theorem stratified_sampling_third_grade 
  (N : ℕ) (N3 : ℕ) (S : ℕ) (x : ℕ)
  (h1 : N = 1600)
  (h2 : N3 = 400)
  (h3 : S = 80)
  (h4 : N3 / N = x / S) :
  x = 20 := 
by {
  sorry
}

end NUMINAMATH_GPT_stratified_sampling_third_grade_l940_94036


namespace NUMINAMATH_GPT_solution_set_l940_94085

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ := x^2 + 3 * x - 4

-- Define the inequality
def inequality (x : ℝ) : Prop := quadratic_expr x > 0

-- State the theorem
theorem solution_set : ∀ x : ℝ, inequality x ↔ (x > 1 ∨ x < -4) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_l940_94085


namespace NUMINAMATH_GPT_candidate_total_score_l940_94055

theorem candidate_total_score (written_score : ℝ) (interview_score : ℝ) (written_weight : ℝ) (interview_weight : ℝ) :
    written_score = 90 → interview_score = 80 → written_weight = 0.70 → interview_weight = 0.30 →
    written_score * written_weight + interview_score * interview_weight = 87 :=
by
  intros
  sorry

end NUMINAMATH_GPT_candidate_total_score_l940_94055


namespace NUMINAMATH_GPT_problem_C_l940_94099

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def is_obtuse_triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  dot_product AB BC > 0 → ∃ (u v : ℝ × ℝ), dot_product u v < 0

theorem problem_C (A B C : ℝ × ℝ) :
  let AB := (B.1 - A.1, B.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  dot_product AB BC > 0 → ∃ (u v : ℝ × ℝ), dot_product u v < 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_C_l940_94099


namespace NUMINAMATH_GPT_jar_a_marbles_l940_94074

theorem jar_a_marbles : ∃ A : ℕ, (∃ B : ℕ, B = A + 12) ∧ (∃ C : ℕ, C = 2 * (A + 12)) ∧ (A + (A + 12) + 2 * (A + 12) = 148) ∧ (A = 28) :=
by
sorry

end NUMINAMATH_GPT_jar_a_marbles_l940_94074


namespace NUMINAMATH_GPT_complete_the_square_l940_94025

theorem complete_the_square (x : ℝ) :
  (x^2 + 14*x + 60) = ((x + 7) ^ 2 + 11) :=
by
  sorry

end NUMINAMATH_GPT_complete_the_square_l940_94025


namespace NUMINAMATH_GPT_smallest_positive_phi_l940_94076

open Real

theorem smallest_positive_phi :
  (∃ k : ℤ, (2 * φ + π / 4 = π / 2 + k * π)) →
  (∀ k, φ = π / 8 + k * π / 2) → 
  0 < φ → 
  φ = π / 8 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_phi_l940_94076


namespace NUMINAMATH_GPT_find_n_l940_94048

theorem find_n (n : ℕ) (h_lcm : Nat.lcm n 14 = 56) (h_gcf : Nat.gcd n 14 = 12) : n = 48 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l940_94048


namespace NUMINAMATH_GPT_gcd_168_54_264_l940_94070

theorem gcd_168_54_264 : Nat.gcd (Nat.gcd 168 54) 264 = 6 :=
by
  -- proof goes here and ends with sorry for now
  sorry

end NUMINAMATH_GPT_gcd_168_54_264_l940_94070


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l940_94016

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := 
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l940_94016


namespace NUMINAMATH_GPT_mass_of_man_l940_94032

variable (L : ℝ) (B : ℝ) (h : ℝ) (ρ : ℝ)

-- Given conditions
def boatLength := L = 3
def boatBreadth := B = 2
def sinkingDepth := h = 0.018
def waterDensity := ρ = 1000

-- The mass of the man
theorem mass_of_man (L B h ρ : ℝ) (H1 : boatLength L) (H2 : boatBreadth B) (H3 : sinkingDepth h) (H4 : waterDensity ρ) : 
  ρ * L * B * h = 108 := by
  sorry

end NUMINAMATH_GPT_mass_of_man_l940_94032


namespace NUMINAMATH_GPT_oil_consumption_relation_l940_94033

noncomputable def initial_oil : ℝ := 62

noncomputable def remaining_oil (x : ℝ) : ℝ :=
  if x = 100 then 50
  else if x = 200 then 38
  else if x = 300 then 26
  else if x = 400 then 14
  else 62 - 0.12 * x

theorem oil_consumption_relation (x : ℝ) :
  remaining_oil x = 62 - 0.12 * x := by
  sorry

end NUMINAMATH_GPT_oil_consumption_relation_l940_94033


namespace NUMINAMATH_GPT_purchase_in_april_l940_94061

namespace FamilySavings

def monthly_income : ℕ := 150000
def monthly_expenses : ℕ := 115000
def initial_savings : ℕ := 45000
def furniture_cost : ℕ := 127000

noncomputable def monthly_savings : ℕ := monthly_income - monthly_expenses
noncomputable def additional_amount_needed : ℕ := furniture_cost - initial_savings
noncomputable def months_required : ℕ := (additional_amount_needed + monthly_savings - 1) / monthly_savings  -- ceiling division

theorem purchase_in_april : months_required = 3 :=
by
  -- Proof goes here
  sorry

end FamilySavings

end NUMINAMATH_GPT_purchase_in_april_l940_94061


namespace NUMINAMATH_GPT_solution_set_of_inequality_l940_94023

noncomputable def f : ℝ → ℝ := sorry 

axiom f_cond : ∀ x : ℝ, f x + deriv f x > 1
axiom f_at_zero : f 0 = 4

theorem solution_set_of_inequality : {x : ℝ | f x > 3 / Real.exp x + 1} = { x : ℝ | x > 0 } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l940_94023


namespace NUMINAMATH_GPT_wet_surface_area_is_correct_l940_94018

-- Define the dimensions of the cistern
def cistern_length : ℝ := 6  -- in meters
def cistern_width  : ℝ := 4  -- in meters
def water_depth    : ℝ := 1.25  -- in meters

-- Compute areas for each surface in contact with water
def bottom_area : ℝ := cistern_length * cistern_width
def long_sides_area : ℝ := 2 * (cistern_length * water_depth)
def short_sides_area : ℝ := 2 * (cistern_width * water_depth)

-- Calculate the total area of the wet surface
def total_wet_surface_area : ℝ := bottom_area + long_sides_area + short_sides_area

-- Statement to prove
theorem wet_surface_area_is_correct : total_wet_surface_area = 49 := by
  sorry

end NUMINAMATH_GPT_wet_surface_area_is_correct_l940_94018


namespace NUMINAMATH_GPT_probability_xiaoming_l940_94031

variable (win_probability : ℚ) 
          (xiaoming_goal : ℕ)
          (xiaojie_goal : ℕ)
          (rounds_needed_xiaoming : ℕ)
          (rounds_needed_xiaojie : ℕ)

def probability_xiaoming_wins_2_consecutive_rounds
   (win_probability : ℚ) 
   (rounds_needed_xiaoming : ℕ) : ℚ :=
  (win_probability ^ 2) + 
  2 * win_probability ^ 3 * (1 - win_probability) + 
  win_probability ^ 4

theorem probability_xiaoming :
    win_probability = (1/2) ∧ 
    rounds_needed_xiaoming = 2 ∧
    rounds_needed_xiaojie = 3 →
    probability_xiaoming_wins_2_consecutive_rounds (1 / 2) 2 = 7 / 16 :=
by
  -- Proof steps placeholder
  sorry

end NUMINAMATH_GPT_probability_xiaoming_l940_94031


namespace NUMINAMATH_GPT_quadratic_equation_solution_unique_l940_94098

noncomputable def b_solution := (-3 + 3 * Real.sqrt 21) / 2
noncomputable def c_solution := (33 - 3 * Real.sqrt 21) / 2

theorem quadratic_equation_solution_unique :
  (∃ (b c : ℝ), 
     (∀ (x : ℝ), 3 * x^2 + b * x + c = 0 → x = b_solution) ∧ 
     b + c = 15 ∧ 3 * c = b^2 ∧
     b = b_solution ∧ c = c_solution) :=
by { sorry }

end NUMINAMATH_GPT_quadratic_equation_solution_unique_l940_94098


namespace NUMINAMATH_GPT_puppies_per_dog_l940_94044

/--
Chuck breeds dogs. He has 3 pregnant dogs.
They each give birth to some puppies. Each puppy needs 2 shots and each shot costs $5.
The total cost of the shots is $120. Prove that each pregnant dog gives birth to 4 puppies.
-/
theorem puppies_per_dog :
  let num_dogs := 3
  let cost_per_shot := 5
  let shots_per_puppy := 2
  let total_cost := 120
  let cost_per_puppy := shots_per_puppy * cost_per_shot
  let total_puppies := total_cost / cost_per_puppy
  (total_puppies / num_dogs) = 4 := by
  sorry

end NUMINAMATH_GPT_puppies_per_dog_l940_94044


namespace NUMINAMATH_GPT_min_value_expression_l940_94065

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
    9 ≤ (5 * z / (2 * x + y) + 5 * x / (y + 2 * z) + 2 * y / (x + z) + (x + y + z) / (x * y + y * z + z * x)) :=
sorry

end NUMINAMATH_GPT_min_value_expression_l940_94065


namespace NUMINAMATH_GPT_base_conversion_and_addition_l940_94040

def C : ℕ := 12

def base9_to_nat (d2 d1 d0 : ℕ) : ℕ := (d2 * 9^2) + (d1 * 9^1) + (d0 * 9^0)

def base13_to_nat (d2 d1 d0 : ℕ) : ℕ := (d2 * 13^2) + (d1 * 13^1) + (d0 * 13^0)

def num1 := base9_to_nat 7 5 2
def num2 := base13_to_nat 6 C 3

theorem base_conversion_and_addition :
  num1 + num2 = 1787 :=
by
  sorry

end NUMINAMATH_GPT_base_conversion_and_addition_l940_94040


namespace NUMINAMATH_GPT_smaller_triangle_perimeter_l940_94075

theorem smaller_triangle_perimeter (p : ℕ) (h : p * 3 = 120) : p = 40 :=
sorry

end NUMINAMATH_GPT_smaller_triangle_perimeter_l940_94075


namespace NUMINAMATH_GPT_convert_spherical_to_cartesian_l940_94047

theorem convert_spherical_to_cartesian :
  let ρ := 5
  let θ₁ := 3 * Real.pi / 4
  let φ₁ := 9 * Real.pi / 5
  let φ' := 2 * Real.pi - φ₁
  let θ' := θ₁ + Real.pi
  ∃ (θ : ℝ) (φ : ℝ),
    0 ≤ θ ∧ θ < 2 * Real.pi ∧
    0 ≤ φ ∧ φ ≤ Real.pi ∧
    (∃ (x y z : ℝ),
      x = ρ * Real.sin φ' * Real.cos θ' ∧
      y = ρ * Real.sin φ' * Real.sin θ' ∧
      z = ρ * Real.cos φ') ∧
    θ = θ' ∧ φ = φ' :=
by
  sorry

end NUMINAMATH_GPT_convert_spherical_to_cartesian_l940_94047


namespace NUMINAMATH_GPT_g_odd_l940_94091

def g (x : ℝ) : ℝ := x^3 - 2*x

theorem g_odd : ∀ x : ℝ, g (-x) = -g x := by
  sorry

end NUMINAMATH_GPT_g_odd_l940_94091


namespace NUMINAMATH_GPT_scale_division_remainder_l940_94071

theorem scale_division_remainder (a b c r : ℕ) (h1 : a = b * c + r) (h2 : 0 ≤ r) (h3 : r < b) :
  (3 * a) % (3 * b) = 3 * r :=
sorry

end NUMINAMATH_GPT_scale_division_remainder_l940_94071


namespace NUMINAMATH_GPT_shaded_area_triangle_l940_94097

theorem shaded_area_triangle (a b : ℝ) (h1 : a = 5) (h2 : b = 15) :
  let area_shaded : ℝ := (5^2) - (1/2 * ((15 / 4) * 5))
  area_shaded = 175 / 8 := 
by
  sorry

end NUMINAMATH_GPT_shaded_area_triangle_l940_94097


namespace NUMINAMATH_GPT_no_possible_values_of_k_l940_94069

theorem no_possible_values_of_k :
  ¬(∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p + q = 65) :=
by
  sorry

end NUMINAMATH_GPT_no_possible_values_of_k_l940_94069


namespace NUMINAMATH_GPT_square_line_product_l940_94089

theorem square_line_product (b : ℝ) 
  (h1 : ∃ y1 y2, y1 = -1 ∧ y2 = 4) 
  (h2 : ∃ x1, x1 = 3) 
  (h3 : (4 - (-1)) = (5 : ℝ)) 
  (h4 : ((∃ b1, b1 = 3 + 5 ∨ b1 = 3 - 5) → b = b1)) :
  b = -2 ∨ b = 8 → b * 8 = -16 :=
by sorry

end NUMINAMATH_GPT_square_line_product_l940_94089


namespace NUMINAMATH_GPT_ratio_65_13_l940_94010

theorem ratio_65_13 : 65 / 13 = 5 := 
by
  sorry

end NUMINAMATH_GPT_ratio_65_13_l940_94010


namespace NUMINAMATH_GPT_arc_length_of_sector_l940_94015

theorem arc_length_of_sector (θ r : ℝ) (h1 : θ = 120) (h2 : r = 2) : 
  (θ / 360) * (2 * Real.pi * r) = (4 * Real.pi) / 3 :=
by
  sorry

end NUMINAMATH_GPT_arc_length_of_sector_l940_94015


namespace NUMINAMATH_GPT_odd_function_product_nonpositive_l940_94060

noncomputable def is_odd_function (f : ℝ → ℝ) := 
  ∀ x : ℝ, f (-x) = -f x

theorem odd_function_product_nonpositive (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) : 
  ∀ x : ℝ, f x * f (-x) ≤ 0 :=
by 
  sorry

end NUMINAMATH_GPT_odd_function_product_nonpositive_l940_94060


namespace NUMINAMATH_GPT_range_of_a_l940_94084

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) → a ≥ 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_a_l940_94084


namespace NUMINAMATH_GPT_billiard_ball_radius_unique_l940_94003

noncomputable def radius_of_billiard_balls (r : ℝ) : Prop :=
  let side_length := 292
  let lhs := (8 + 2 * Real.sqrt 3) * r
  lhs = side_length

theorem billiard_ball_radius_unique (r : ℝ) : radius_of_billiard_balls r → r = (146 / 13) * (4 - Real.sqrt 3 / 3) :=
by
  intro h1
  sorry

end NUMINAMATH_GPT_billiard_ball_radius_unique_l940_94003


namespace NUMINAMATH_GPT_replace_asterisks_l940_94026

theorem replace_asterisks (x : ℝ) (h : (x / 20) * (x / 80) = 1) : x = 40 :=
sorry

end NUMINAMATH_GPT_replace_asterisks_l940_94026


namespace NUMINAMATH_GPT_interest_rate_l940_94045

noncomputable def simple_interest (P r t: ℝ) : ℝ := P * r * t / 100

noncomputable def compound_interest (P r t: ℝ) : ℝ := P * (1 + r / 100) ^ t - P

theorem interest_rate (P r: ℝ) (h1: simple_interest P r 2 = 50) (h2: compound_interest P r 2 = 51.25) : r = 5 :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_l940_94045


namespace NUMINAMATH_GPT_solve_for_r_l940_94014

theorem solve_for_r (r : ℚ) (h : 4 * (r - 10) = 3 * (3 - 3 * r) + 9) : r = 58 / 13 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_r_l940_94014


namespace NUMINAMATH_GPT_inequalities_consistent_l940_94002

theorem inequalities_consistent (x y : ℝ) (h1 : y ≥ 0) (h2 : y * (y + 1) ≤ (x + 1) ^ 2) (h3 : y * (y - 1) ≤ x ^ 2) : true := 
by 
  sorry

end NUMINAMATH_GPT_inequalities_consistent_l940_94002


namespace NUMINAMATH_GPT_quadratic_has_real_roots_l940_94078

theorem quadratic_has_real_roots (m : ℝ) : (∃ x : ℝ, x^2 + x - 4 * m = 0) ↔ m ≥ -1 / 16 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_real_roots_l940_94078


namespace NUMINAMATH_GPT_parabola_vertex_n_l940_94000

theorem parabola_vertex_n (x y : ℝ) (h : y = -3 * x^2 - 24 * x - 72) : ∃ m n : ℝ, (m, n) = (-4, -24) :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_n_l940_94000


namespace NUMINAMATH_GPT_recurring_decimals_sum_l940_94022

theorem recurring_decimals_sum :
  (0.333333333333 : ℚ) + (0.040404040404 : ℚ) + (0.005005005005 : ℚ) = 42 / 111 :=
by
  sorry

end NUMINAMATH_GPT_recurring_decimals_sum_l940_94022


namespace NUMINAMATH_GPT_negation_proof_l940_94092

theorem negation_proof :
  ¬(∀ x : ℝ, x > 0 → Real.exp x > x + 1) ↔ ∃ x : ℝ, x > 0 ∧ Real.exp x ≤ x + 1 :=
by sorry

end NUMINAMATH_GPT_negation_proof_l940_94092


namespace NUMINAMATH_GPT_determine_irrational_option_l940_94096

def is_irrational (x : ℝ) : Prop := ¬ ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

def option_A : ℝ := 7
def option_B : ℝ := 0.5
def option_C : ℝ := abs (3 / 20 : ℚ)
def option_D : ℝ := 0.5151151115 -- Assume notation describes the stated behavior

theorem determine_irrational_option :
  is_irrational option_D ∧
  ¬ is_irrational option_A ∧
  ¬ is_irrational option_B ∧
  ¬ is_irrational option_C := 
by
  sorry

end NUMINAMATH_GPT_determine_irrational_option_l940_94096


namespace NUMINAMATH_GPT_non_negative_integer_solutions_l940_94064

theorem non_negative_integer_solutions (x : ℕ) : 3 * x - 2 < 7 ↔ x = 0 ∨ x = 1 ∨ x = 2 := by
  sorry

end NUMINAMATH_GPT_non_negative_integer_solutions_l940_94064


namespace NUMINAMATH_GPT_product_of_four_consecutive_integers_l940_94001

theorem product_of_four_consecutive_integers (n : ℤ) : ∃ k : ℤ, k^2 = (n-1) * n * (n+1) * (n+2) + 1 :=
by
  sorry

end NUMINAMATH_GPT_product_of_four_consecutive_integers_l940_94001


namespace NUMINAMATH_GPT_problem_statement_l940_94054
open Real

noncomputable def l1 (α : ℝ) : ℝ → ℝ → Prop := fun x y => (sin α) * x - (cos α) * y + 1 = 0
noncomputable def l2 (α : ℝ) : ℝ → ℝ → Prop := fun x y => (sin α) * x + (cos α) * y + 1 = 0
noncomputable def l3 (α : ℝ) : ℝ → ℝ → Prop := fun x y => (cos α) * x - (sin α) * y + 1 = 0
noncomputable def l4 (α : ℝ) : ℝ → ℝ → Prop := fun x y => (cos α) * x + (sin α) * y + 1 = 0

theorem problem_statement:
  (∃ (α : ℝ), ∀ (x y : ℝ), l1 α x y → l2 α x y) ∧
  (∀ (α : ℝ), ∀ (x y : ℝ), l1 α x y → (sin α) * (cos α) + (-cos α) * (sin α) = 0) ∧
  (∃ (p : ℝ × ℝ), ∀ (α : ℝ), abs ((sin α) * p.1 - (cos α) * p.2 + 1) / sqrt ((sin α)^2 + (cos α)^2) = 1 ∧
                        abs ((sin α) * p.1 + (cos α) * p.2 + 1) / sqrt ((sin α)^2 + (cos α)^2) = 1 ∧
                        abs ((cos α) * p.1 - (sin α) * p.2 + 1) / sqrt ((cos α)^2 + (sin α)^2) = 1 ∧
                        abs ((cos α) * p.1 + (sin α) * p.2 + 1) / sqrt ((cos α)^2 + (sin α)^2) = 1) :=
sorry

end NUMINAMATH_GPT_problem_statement_l940_94054


namespace NUMINAMATH_GPT_acute_triangle_l940_94093

theorem acute_triangle (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
                       (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a)
                       (h7 : a^3 + b^3 = c^3) :
                       c^2 < a^2 + b^2 :=
by {
  sorry
}

end NUMINAMATH_GPT_acute_triangle_l940_94093


namespace NUMINAMATH_GPT_g_10_plus_g_neg10_eq_6_l940_94043

variable (a b c : ℝ)
noncomputable def g : ℝ → ℝ := λ x => a * x ^ 8 + b * x ^ 6 - c * x ^ 4 + 5

theorem g_10_plus_g_neg10_eq_6 (h : g a b c 10 = 3) : g a b c 10 + g a b c (-10) = 6 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_g_10_plus_g_neg10_eq_6_l940_94043


namespace NUMINAMATH_GPT_greatest_divisor_540_180_under_60_l940_94020

theorem greatest_divisor_540_180_under_60 : ∃ d, d ∣ 540 ∧ d ∣ 180 ∧ d < 60 ∧ ∀ k, k ∣ 540 → k ∣ 180 → k < 60 → k ≤ d :=
by
  sorry

end NUMINAMATH_GPT_greatest_divisor_540_180_under_60_l940_94020


namespace NUMINAMATH_GPT_power_multiplication_l940_94035

theorem power_multiplication (x : ℝ) : (-4 * x^3)^2 = 16 * x^6 := 
by 
  sorry

end NUMINAMATH_GPT_power_multiplication_l940_94035


namespace NUMINAMATH_GPT_cos_of_largest_angle_is_neg_half_l940_94030

-- Lean does not allow forward references to elements yet to be declared, 
-- hence we keep a strict order for declarations
namespace TriangleCosine

open Real

-- Define the side lengths of the triangle as constants
def a : ℝ := 3
def b : ℝ := 5
def c : ℝ := 7

-- Define the expression using cosine rule to find cos C
noncomputable def cos_largest_angle : ℝ := (a^2 + b^2 - c^2) / (2 * a * b)

-- Declare the theorem statement
theorem cos_of_largest_angle_is_neg_half : cos_largest_angle = -1 / 2 := 
by 
  sorry

end TriangleCosine

end NUMINAMATH_GPT_cos_of_largest_angle_is_neg_half_l940_94030


namespace NUMINAMATH_GPT_find_a_l940_94077

theorem find_a (a : ℝ) : (∃ x y : ℝ, y = 4 - 3 * x ∧ y = 2 * x - 1 ∧ y = a * x + 7) → a = 6 := 
by
  sorry

end NUMINAMATH_GPT_find_a_l940_94077


namespace NUMINAMATH_GPT_train_crosses_bridge_in_approximately_21_seconds_l940_94079

noncomputable def length_of_train : ℝ := 110  -- meters
noncomputable def speed_of_train_kmph : ℝ := 60  -- kilometers per hour
noncomputable def length_of_bridge : ℝ := 240  -- meters

noncomputable def speed_of_train_mps : ℝ := (speed_of_train_kmph * 1000) / 3600

noncomputable def total_distance : ℝ := length_of_train + length_of_bridge

noncomputable def required_time : ℝ := total_distance / speed_of_train_mps

theorem train_crosses_bridge_in_approximately_21_seconds :
  |required_time - 21| < 1 :=
by sorry

end NUMINAMATH_GPT_train_crosses_bridge_in_approximately_21_seconds_l940_94079


namespace NUMINAMATH_GPT_arithmetic_progression_infinite_kth_powers_l940_94049

theorem arithmetic_progression_infinite_kth_powers {a d k : ℕ} (ha : a > 0) (hd : d > 0) (hk : k > 0) :
  (∀ n : ℕ, ¬ ∃ b : ℕ, a + n * d = b ^ k) ∨ (∀ b : ℕ, ∃ n : ℕ, a + n * d = b ^ k) :=
sorry

end NUMINAMATH_GPT_arithmetic_progression_infinite_kth_powers_l940_94049


namespace NUMINAMATH_GPT_complex_fraction_l940_94072

open Complex

/-- The given complex fraction \(\frac{5 - i}{1 - i}\) evaluates to \(3 + 2i\). -/
theorem complex_fraction : (⟨5, -1⟩ : ℂ) / (⟨1, -1⟩ : ℂ) = ⟨3, 2⟩ :=
  by
  sorry

end NUMINAMATH_GPT_complex_fraction_l940_94072


namespace NUMINAMATH_GPT_satisfy_equation_l940_94012

theorem satisfy_equation (a b c : ℤ) (h1 : a = c) (h2 : b - 1 = a) : a * (a - b) + b * (b - c) + c * (c - a) = 2 :=
by
  sorry

end NUMINAMATH_GPT_satisfy_equation_l940_94012


namespace NUMINAMATH_GPT_smallest_prime_factor_of_2939_l940_94009

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def smallest_prime_factor (n : ℕ) (p : ℕ) : Prop :=
  is_prime p ∧ p ∣ n ∧ ∀ q : ℕ, is_prime q ∧ q ∣ n → p ≤ q

theorem smallest_prime_factor_of_2939 : smallest_prime_factor 2939 13 :=
by
  sorry

end NUMINAMATH_GPT_smallest_prime_factor_of_2939_l940_94009


namespace NUMINAMATH_GPT_probability_crisp_stops_on_dime_l940_94027

noncomputable def crisp_stops_on_dime_probability : ℚ :=
  let a := (2/3 : ℚ)
  let b := (1/3 : ℚ)
  let a1 := (15/31 : ℚ)
  let b1 := (30/31 : ℚ)
  (2 / 3) * a1 + (1 / 3) * b1

theorem probability_crisp_stops_on_dime :
  crisp_stops_on_dime_probability = 20 / 31 :=
by
  sorry

end NUMINAMATH_GPT_probability_crisp_stops_on_dime_l940_94027


namespace NUMINAMATH_GPT_trigonometric_relationship_l940_94050

-- Given conditions
variables (x : ℝ) (a b c : ℝ)

-- Required conditions
variables (h1 : π / 4 < x) (h2 : x < π / 2)
variables (ha : a = Real.sin x)
variables (hb : b = Real.cos x)
variables (hc : c = Real.tan x)

-- Proof goal
theorem trigonometric_relationship : b < a ∧ a < c :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_trigonometric_relationship_l940_94050


namespace NUMINAMATH_GPT_gcd_max_value_l940_94007

theorem gcd_max_value (x y : ℤ) (h_posx : x > 0) (h_posy : y > 0) (h_sum : x + y = 780) :
  gcd x y ≤ 390 ∧ ∃ x' y', x' > 0 ∧ y' > 0 ∧ x' + y' = 780 ∧ gcd x' y' = 390 := by
  sorry

end NUMINAMATH_GPT_gcd_max_value_l940_94007


namespace NUMINAMATH_GPT_tg_pi_over_12_eq_exists_two_nums_l940_94058

noncomputable def tg (x : ℝ) := Real.tan x

theorem tg_pi_over_12_eq : tg (Real.pi / 12) = Real.sqrt ((2 - Real.sqrt 3) / (2 + Real.sqrt 3)) :=
sorry

theorem exists_two_nums (a : Fin 13 → ℝ) (h_diff : Function.Injective a) :
  ∃ x y, 0 < (x - y) / (1 + x * y) ∧ (x - y) / (1 + x * y) < Real.sqrt ((2 - Real.sqrt 3) / (2 + Real.sqrt 3)) :=
sorry

end NUMINAMATH_GPT_tg_pi_over_12_eq_exists_two_nums_l940_94058


namespace NUMINAMATH_GPT_tan_a_values_l940_94062

theorem tan_a_values (a : ℝ) (h : Real.sin (2 * a) = 2 - 2 * Real.cos (2 * a)) :
  Real.tan a = 0 ∨ Real.tan a = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_tan_a_values_l940_94062


namespace NUMINAMATH_GPT_sequence_formula_l940_94046

-- Defining the sequence and the conditions
def bounded_seq (a : ℕ → ℝ) : Prop :=
  ∃ C > 0, ∀ n, |a n| ≤ C

-- Statement of the problem in Lean
theorem sequence_formula (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) = 3 * a n - 4) →
  bounded_seq a →
  ∀ n : ℕ, a n = 2 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_sequence_formula_l940_94046


namespace NUMINAMATH_GPT_negation_of_exists_l940_94063

theorem negation_of_exists (h : ¬ (∃ x_0 : ℝ, x_0^3 - x_0^2 + 1 ≤ 0)) : ∀ x : ℝ, x^3 - x^2 + 1 > 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_exists_l940_94063


namespace NUMINAMATH_GPT_initial_investment_l940_94095

noncomputable def doubling_period (r : ℝ) : ℝ := 70 / r
noncomputable def investment_after_doubling (P : ℝ) (n : ℝ) : ℝ := P * (2 ^ n)

theorem initial_investment (total_amount : ℝ) (years : ℝ) (rate : ℝ) (initial : ℝ) :
  rate = 8 → total_amount = 28000 → years = 18 → 
  initial = total_amount / (2 ^ (years / (doubling_period rate))) :=
by
  intros hrate htotal hyears
  simp [doubling_period, investment_after_doubling] at *
  rw [hrate, htotal, hyears]
  norm_num
  sorry

end NUMINAMATH_GPT_initial_investment_l940_94095


namespace NUMINAMATH_GPT_locus_of_orthocenter_l940_94008

theorem locus_of_orthocenter (A_x A_y : ℝ) (h_A : A_x = 0 ∧ A_y = 2)
    (c_r : ℝ) (h_c : c_r = 2) 
    (M_x M_y Q_x Q_y : ℝ)
    (h_circle : Q_x^2 + Q_y^2 = c_r^2)
    (h_tangent : M_x ≠ 0 ∧ (M_y - 2) / M_x = -Q_x / Q_y)
    (h_M_on_tangent : M_x^2 + (M_y - 2)^2 = 4 ∧ M_x ≠ 0)
    (H_x H_y : ℝ)
    (h_orthocenter : (H_x - A_x)^2 + (H_y - A_y + 2)^2 = 4) :
    (H_x^2 + (H_y - 2)^2 = 4) ∧ (H_x ≠ 0) := 
sorry

end NUMINAMATH_GPT_locus_of_orthocenter_l940_94008


namespace NUMINAMATH_GPT_motorcycle_tire_max_distance_l940_94086

theorem motorcycle_tire_max_distance :
  let wear_front := (1 : ℝ) / 25000
  let wear_rear := (1 : ℝ) / 15000
  let s := 18750
  wear_front * (s / 2) + wear_rear * (s / 2) = 1 :=
by 
  let wear_front := (1 : ℝ) / 25000
  let wear_rear := (1 : ℝ) / 15000
  sorry

end NUMINAMATH_GPT_motorcycle_tire_max_distance_l940_94086


namespace NUMINAMATH_GPT_solve_for_q_l940_94083

theorem solve_for_q :
  ∀ (q : ℕ), 16^15 = 4^q → q = 30 :=
by
  intro q
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_q_l940_94083


namespace NUMINAMATH_GPT_maximize_triangle_area_l940_94053

theorem maximize_triangle_area (m : ℝ) (l : ∀ x y, x + y + m = 0) (C : ∀ x y, x^2 + y^2 + 4 * y = 0) :
  m = 0 ∨ m = 4 :=
sorry

end NUMINAMATH_GPT_maximize_triangle_area_l940_94053


namespace NUMINAMATH_GPT_equation_of_directrix_l940_94013

theorem equation_of_directrix (x y : ℝ) (h : y^2 = 2 * x) : 
  x = - (1/2) :=
sorry

end NUMINAMATH_GPT_equation_of_directrix_l940_94013


namespace NUMINAMATH_GPT_range_of_f_l940_94052

def f (x : ℤ) : ℤ := (x - 1)^2 - 1

theorem range_of_f :
  Set.image f {-1, 0, 1, 2, 3} = {-1, 0, 3} :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_l940_94052


namespace NUMINAMATH_GPT_eccentricity_of_hyperbola_l940_94081

noncomputable def hyperbola_eccentricity (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : ℝ :=
  let c := 2 * b
  let e := c / a
  e

theorem eccentricity_of_hyperbola (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) 
  (h_cond : hyperbola_eccentricity a b h_a h_b = 2 * (b / a)) :
  hyperbola_eccentricity a b h_a h_b = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_eccentricity_of_hyperbola_l940_94081


namespace NUMINAMATH_GPT_stock_yield_percentage_l940_94024

def annualDividend (parValue : ℕ) (rate : ℕ) : ℕ :=
  (parValue * rate) / 100

def yieldPercentage (dividend : ℕ) (marketPrice : ℕ) : ℕ :=
  (dividend * 100) / marketPrice

theorem stock_yield_percentage :
  let par_value := 100
  let rate := 8
  let market_price := 80
  yieldPercentage (annualDividend par_value rate) market_price = 10 :=
by
  sorry

end NUMINAMATH_GPT_stock_yield_percentage_l940_94024


namespace NUMINAMATH_GPT_integer_b_if_integer_a_l940_94051

theorem integer_b_if_integer_a (a b : ℤ) (h : 2 * a + a^2 = 2 * b + b^2) : (∃ a' : ℤ, a = a') → ∃ b' : ℤ, b = b' :=
by
-- proof will be filled in here
sorry

end NUMINAMATH_GPT_integer_b_if_integer_a_l940_94051


namespace NUMINAMATH_GPT_significant_digits_of_side_length_l940_94082

noncomputable def num_significant_digits (n : Float) : Nat :=
  -- This is a placeholder function to determine the number of significant digits
  sorry

theorem significant_digits_of_side_length :
  ∀ (A : Float), A = 3.2400 → num_significant_digits (Float.sqrt A) = 5 :=
by
  intro A h
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_significant_digits_of_side_length_l940_94082


namespace NUMINAMATH_GPT_polyhedron_edges_faces_vertices_l940_94068

theorem polyhedron_edges_faces_vertices
  (E F V n m : ℕ)
  (h1 : n * F = 2 * E)
  (h2 : m * V = 2 * E)
  (h3 : V + F = E + 2) :
  ¬(m * F = 2 * E) :=
sorry

end NUMINAMATH_GPT_polyhedron_edges_faces_vertices_l940_94068


namespace NUMINAMATH_GPT_number_of_chocolate_boxes_l940_94090

theorem number_of_chocolate_boxes
  (x y p : ℕ)
  (pieces_per_box : ℕ)
  (total_candies : ℕ)
  (h_y : y = 4)
  (h_pieces : pieces_per_box = 9)
  (h_total : total_candies = 90) :
  x = 6 :=
by
  -- Definitions of the conditions
  let caramel_candies := y * pieces_per_box
  let total_chocolate_candies := total_candies - caramel_candies
  let x := total_chocolate_candies / pieces_per_box
  
  -- Main theorem statement: x = 6
  sorry

end NUMINAMATH_GPT_number_of_chocolate_boxes_l940_94090
