import Mathlib

namespace cherry_sodas_correct_l40_40714

/-
A cooler is filled with 24 cans of cherry soda and orange pop. 
There are twice as many cans of orange pop as there are of cherry soda. 
Prove that the number of cherry sodas is 8.
-/
def num_cherry_sodas (C O : ℕ) : Prop :=
  O = 2 * C ∧ C + O = 24 → C = 8

theorem cherry_sodas_correct (C O : ℕ) (h : O = 2 * C ∧ C + O = 24) : C = 8 :=
by
  sorry

end cherry_sodas_correct_l40_40714


namespace evaluate_expression_m_4_evaluate_expression_m_negative_4_l40_40838

variables (a b c d m : ℝ)

theorem evaluate_expression_m_4 (h1 : a + b = 0) (h2 : c * d = 1) (h3 : |m| = 4) (h_m_4 : m = 4) :
  (a + b) / (3 * m) + m^2 - 5 * (c * d) + 6 * m = 35 :=
by sorry

theorem evaluate_expression_m_negative_4 (h1 : a + b = 0) (h2 : c * d = 1) (h3 : |m| = 4) (h_m_negative_4 : m = -4) :
  (a + b) / (3 * m) + m^2 - 5 * (c * d) + 6 * m = -13 :=
by sorry

end evaluate_expression_m_4_evaluate_expression_m_negative_4_l40_40838


namespace hyperbola_eccentricity_l40_40278

theorem hyperbola_eccentricity (x y : ℝ) :
  (∃ a b : ℝ, a^2 = 4 ∧ b^2 = 12 ∧ (∃ c : ℝ, c^2 = a^2 + b^2 ∧ c = 4) ∧
  ∃ e : ℝ, e = c / a ∧ e = 2) :=
sorry

end hyperbola_eccentricity_l40_40278


namespace inequalities_in_quadrants_l40_40030

theorem inequalities_in_quadrants (x y : ℝ) :
  (y > - (1 / 2) * x + 6) ∧ (y > 3 * x - 4) → (x > 0) ∧ (y > 0) :=
  sorry

end inequalities_in_quadrants_l40_40030


namespace a_less_than_2_l40_40782

-- Define the quadratic function f(x)
noncomputable def f (x : ℝ) : ℝ := x^2 - 6 * x + 2

-- Define the condition that the inequality f(x) - a > 0 has solutions in the interval [0,5]
def inequality_holds (a : ℝ) : Prop := ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 5 → f x - a > 0

-- Theorem stating that a must be less than 2 to satisfy the above condition
theorem a_less_than_2 : ∀ (a : ℝ), (∃ (x : ℝ), 0 ≤ x ∧ x ≤ 5 ∧ f x - a > 0) → a < 2 := 
sorry

end a_less_than_2_l40_40782


namespace min_positive_announcements_l40_40449

theorem min_positive_announcements (x y : ℕ) 
  (h1 : x * (x - 1) = 90)
  (h2 : y * (y - 1) + (x - y) * (x - y - 1) = 48) 
  : y = 3 :=
sorry

end min_positive_announcements_l40_40449


namespace line_through_point_with_equal_intercepts_l40_40948

/-- A line passing through point (-2, 3) and having equal intercepts
on the coordinate axes can have the equation y = -3/2 * x or x + y = 1. -/
theorem line_through_point_with_equal_intercepts (x y : Real) :
  (∃ (m : Real), (y = m * x) ∧ (y - m * (-2) = 3 ∧ y - m * 0 = 0))
  ∨ (∃ (a : Real), (x + y = a) ∧ (a = 1 ∧ (-2) + 3 = a)) :=
sorry

end line_through_point_with_equal_intercepts_l40_40948


namespace height_of_block_l40_40968

theorem height_of_block (h : ℝ) : 
  ((∃ (side : ℝ), ∃ (n : ℕ), side = 15 ∧ n = 10 ∧ 15 * 30 * h = n * side^3) → h = 75) := 
by
  intros
  sorry

end height_of_block_l40_40968


namespace medal_ratio_l40_40484

theorem medal_ratio (total_medals : ℕ) (track_medals : ℕ) (badminton_medals : ℕ) (swimming_medals : ℕ) 
  (h1 : total_medals = 20) 
  (h2 : track_medals = 5) 
  (h3 : badminton_medals = 5) 
  (h4 : swimming_medals = total_medals - track_medals - badminton_medals) : 
  swimming_medals / track_medals = 2 := 
by 
  sorry

end medal_ratio_l40_40484


namespace ratio_men_to_women_l40_40107

theorem ratio_men_to_women (M W : ℕ) (h1 : W = M + 4) (h2 : M + W = 18) : M = 7 ∧ W = 11 :=
by
  sorry

end ratio_men_to_women_l40_40107


namespace pairs_nat_eq_l40_40999

theorem pairs_nat_eq (n k : ℕ) :
  (n + 1) ^ k = n! + 1 ↔ (n, k) = (1, 1) ∨ (n, k) = (2, 1) ∨ (n, k) = (4, 2) :=
by
  sorry

end pairs_nat_eq_l40_40999


namespace value_of_f_at_5_l40_40666

def f (x : ℤ) : ℤ := x^3 - x^2 + x

theorem value_of_f_at_5 : f 5 = 105 := by
  sorry

end value_of_f_at_5_l40_40666


namespace sum_of_number_and_its_square_is_20_l40_40962

theorem sum_of_number_and_its_square_is_20 (n : ℕ) (h : n = 4) : n + n^2 = 20 :=
by
  sorry

end sum_of_number_and_its_square_is_20_l40_40962


namespace calculate_S_value_l40_40374

def operation_S (a b : ℕ) : ℕ := 4 * a + 7 * b

theorem calculate_S_value : operation_S 8 3 = 53 :=
by
  -- proof goes here
  sorry

end calculate_S_value_l40_40374


namespace min_value_of_m_n_squared_l40_40133

theorem min_value_of_m_n_squared 
  (a b c : ℝ)
  (triangle_cond : a^2 + b^2 = c^2)
  (m n : ℝ)
  (line_cond : a * m + b * n + 3 * c = 0) 
  : m^2 + n^2 = 9 := 
by
  sorry

end min_value_of_m_n_squared_l40_40133


namespace total_rats_l40_40683

theorem total_rats (Elodie_rats Hunter_rats Kenia_rats : ℕ) 
  (h1 : Elodie_rats = 30) 
  (h2 : Elodie_rats = Hunter_rats + 10)
  (h3 : Kenia_rats = 3 * (Elodie_rats + Hunter_rats)) :
  Elodie_rats + Hunter_rats + Kenia_rats = 200 :=
by
  sorry

end total_rats_l40_40683


namespace appropriate_sampling_method_is_stratified_l40_40805

-- Definition of the problem conditions
def total_students := 500 + 500
def male_students := 500
def female_students := 500
def survey_sample_size := 100

-- The goal is to show that given these conditions, the appropriate sampling method is Stratified sampling method.
theorem appropriate_sampling_method_is_stratified :
  total_students = 1000 ∧
  male_students = 500 ∧
  female_students = 500 ∧
  survey_sample_size = 100 →
  sampling_method = "Stratified" :=
by
  intros h
  sorry

end appropriate_sampling_method_is_stratified_l40_40805


namespace school_C_paintings_l40_40826

theorem school_C_paintings
  (A B C : ℕ)
  (h1 : B + C = 41)
  (h2 : A + C = 38)
  (h3 : A + B = 43) : 
  C = 18 :=
by
  sorry

end school_C_paintings_l40_40826


namespace quadratic_inequality_solution_l40_40752

theorem quadratic_inequality_solution (m : ℝ) :
  (∀ x : ℝ, m * x^2 + m * x + 1 > 0) ↔ (0 ≤ m ∧ m < 4) :=
by
  sorry

end quadratic_inequality_solution_l40_40752


namespace problem_A_problem_C_l40_40426

section
variables {a b : ℝ}

-- A: If a and b are positive real numbers, and a > b, then a^3 + b^3 > a^2 * b + a * b^2.
theorem problem_A (ha : 0 < a) (hb : 0 < b) (h : a > b) : a^3 + b^3 > a^2 * b + a * b^2 := sorry

end

section
variables {a b : ℝ}

-- C: If a and b are real numbers, then "a > b > 0" is a sufficient but not necessary condition for "1/a < 1/b".
theorem problem_C (ha : 0 < a) (hb : 0 < b) (h : a > b) : 1/a < 1/b := sorry

end

end problem_A_problem_C_l40_40426


namespace smallest_n_condition_l40_40790

theorem smallest_n_condition :
  ∃ n ≥ 2, ∃ (a : Fin n → ℤ), (Finset.sum Finset.univ a = 1990) ∧ (Finset.univ.prod a = 1990) ∧ (n = 5) :=
by
  sorry

end smallest_n_condition_l40_40790


namespace ratio_of_cube_sides_l40_40591

theorem ratio_of_cube_sides 
  (a b : ℝ) 
  (h : (6 * a^2) / (6 * b^2) = 49) :
  a / b = 7 :=
by
  sorry

end ratio_of_cube_sides_l40_40591


namespace number_of_initial_cards_l40_40459

theorem number_of_initial_cards (x : ℝ) (h1 : x + 276.0 = 580) : x = 304 :=
by
  sorry

end number_of_initial_cards_l40_40459


namespace contemporaries_probability_l40_40547

theorem contemporaries_probability:
  (∀ (x y : ℝ),
    0 ≤ x ∧ x ≤ 400 ∧
    0 ≤ y ∧ y ≤ 400 ∧
    (x < y + 80) ∧ (y < x + 80)) →
    (∃ p : ℝ, p = 9 / 25) :=
by sorry

end contemporaries_probability_l40_40547


namespace number_of_students_taking_art_l40_40309

noncomputable def total_students : ℕ := 500
noncomputable def students_taking_music : ℕ := 50
noncomputable def students_taking_both : ℕ := 10
noncomputable def students_taking_neither : ℕ := 440

theorem number_of_students_taking_art (A : ℕ) (h1: total_students = 500) (h2: students_taking_music = 50) 
  (h3: students_taking_both = 10) (h4: students_taking_neither = 440) : A = 20 :=
by 
  have h5 : total_students = students_taking_music - students_taking_both + A - students_taking_both + 
    students_taking_both + students_taking_neither := sorry
  have h6 : 500 = 40 + A - 10 + 10 + 440 := sorry
  have h7 : 500 = A + 480 := sorry
  have h8 : A = 20 := by linarith 
  exact h8

end number_of_students_taking_art_l40_40309


namespace arthur_walks_distance_l40_40122

variables (blocks_east blocks_north : ℕ) 
variable (distance_per_block : ℝ)
variable (total_blocks : ℕ)
def total_distance (blocks : ℕ) (distance_per_block : ℝ) : ℝ :=
  blocks * distance_per_block

theorem arthur_walks_distance (h_east : blocks_east = 8) (h_north : blocks_north = 10) 
    (h_total_blocks : total_blocks = blocks_east + blocks_north)
    (h_distance_per_block : distance_per_block = 1 / 4) :
  total_distance total_blocks distance_per_block = 4.5 :=
by {
  -- Here we specify the proof, but as required, we use sorry to skip it.
  sorry
}

end arthur_walks_distance_l40_40122


namespace beth_coins_sold_l40_40099

def initial_coins : ℕ := 250
def additional_coins : ℕ := 75
def percentage_sold : ℚ := 60 / 100
def total_coins : ℕ := initial_coins + additional_coins
def coins_sold : ℚ := percentage_sold * total_coins

theorem beth_coins_sold : coins_sold = 195 :=
by
  -- Sorry is used to skip the proof as requested
  sorry

end beth_coins_sold_l40_40099


namespace part_a_l40_40510

theorem part_a 
  (x y u v : ℝ) 
  (h1 : x + y = u + v) 
  (h2 : x^2 + y^2 = u^2 + v^2) : 
  ∀ n : ℕ, x^n + y^n = u^n + v^n := 
by sorry

end part_a_l40_40510


namespace solve_for_x_l40_40800

theorem solve_for_x (x y : ℝ) (h1 : y = 1 / (4 * x + 2)) (h2 : y = 2) : x = -3 / 8 := by 
  sorry

end solve_for_x_l40_40800


namespace cube_of_square_of_third_smallest_prime_l40_40763

-- Define the third smallest prime number
def third_smallest_prime : ℕ := 5

-- Theorem to prove the cube of the square of the third smallest prime number
theorem cube_of_square_of_third_smallest_prime :
  (third_smallest_prime^2)^3 = 15625 := by
  sorry

end cube_of_square_of_third_smallest_prime_l40_40763


namespace sum_of_f1_possible_values_l40_40269

noncomputable def f : ℝ → ℝ := sorry

theorem sum_of_f1_possible_values :
  (∀ (x y : ℝ), f (f (x - y)) = f x * f y - f x + f y - 2 * x * y) →
  (f 1 = -1) := sorry

end sum_of_f1_possible_values_l40_40269


namespace inscribed_circle_radius_l40_40575

theorem inscribed_circle_radius (a r : ℝ) (unit_square : a = 1)
  (touches_arc_AC : ∀ (x : ℝ × ℝ), x.1^2 + x.2^2 = (a - r)^2)
  (touches_arc_BD : ∀ (y : ℝ × ℝ), y.1^2 + y.2^2 = (a - r)^2)
  (touches_side_AB : ∀ (z : ℝ × ℝ), z.1 = r ∨ z.2 = r) :
  r = 3 / 8 := by sorry

end inscribed_circle_radius_l40_40575


namespace dante_eggs_l40_40986

theorem dante_eggs (E F : ℝ) (h1 : F = E / 2) (h2 : F + E = 90) : E = 60 :=
by
  sorry

end dante_eggs_l40_40986


namespace miranda_saved_per_month_l40_40735

-- Definition of the conditions and calculation in the problem
def total_cost : ℕ := 260
def sister_contribution : ℕ := 50
def months : ℕ := 3
def miranda_savings : ℕ := total_cost - sister_contribution
def saved_per_month : ℕ := miranda_savings / months

-- Theorem statement with the expected answer
theorem miranda_saved_per_month : saved_per_month = 70 :=
by
  sorry

end miranda_saved_per_month_l40_40735


namespace units_digit_of_power_l40_40488

theorem units_digit_of_power (a b : ℕ) : (a % 10 = 7) → (b % 4 = 0) → ((a^b) % 10 = 1) :=
by
  intros
  sorry

end units_digit_of_power_l40_40488


namespace solve_for_x_l40_40483

theorem solve_for_x : ∀ (x : ℝ), (-3 * x - 8 = 5 * x + 4) → (x = -3 / 2) := by
  intro x
  intro h
  sorry

end solve_for_x_l40_40483


namespace root_interval_k_l40_40925

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x - 1

theorem root_interval_k (k : ℤ) (h : ∃ ξ : ℝ, k < ξ ∧ ξ < k+1 ∧ f ξ = 0) : k = 0 :=
by
  sorry

end root_interval_k_l40_40925


namespace find_a_solve_inequality_l40_40807

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (2^x + a) / (2^x - 1)

theorem find_a (h : ∀ x : ℝ, f x a = -f (-x) a) : a = 1 := sorry

theorem solve_inequality (x : ℝ) (hx : 0 < x ∧ x < 1) : f x 1 > 3 := sorry

end find_a_solve_inequality_l40_40807


namespace equilateral_cannot_be_obtuse_l40_40689

-- Additional definitions for clarity and mathematical rigor.
def is_equilateral (a b c : ℝ) : Prop := a = b ∧ b = c ∧ c = a
def is_obtuse (A B C : ℝ) : Prop := 
    (A > 90 ∧ B < 90 ∧ C < 90) ∨ 
    (B > 90 ∧ A < 90 ∧ C < 90) ∨
    (C > 90 ∧ A < 90 ∧ B < 90)

-- Theorem statement
theorem equilateral_cannot_be_obtuse (a b c : ℝ) (A B C : ℝ) :
  is_equilateral a b c → 
  (A + B + C = 180) → 
  (A = B ∧ B = C) → 
  ¬ is_obtuse A B C :=
by { sorry } -- Proof is not necessary as per instruction.

end equilateral_cannot_be_obtuse_l40_40689


namespace largest_value_is_E_l40_40365

-- Define the given values
def A := 1 - 0.1
def B := 1 - 0.01
def C := 1 - 0.001
def D := 1 - 0.0001
def E := 1 - 0.00001

-- Main theorem statement
theorem largest_value_is_E : E > A ∧ E > B ∧ E > C ∧ E > D :=
by
  sorry

end largest_value_is_E_l40_40365


namespace frac_val_of_x_y_l40_40945

theorem frac_val_of_x_y (x y : ℝ) (h: (4 : ℝ) < (2 * x - 3 * y) / (2 * x + 3 * y) ∧ (2 * x - 3 * y) / (2 * x + 3 * y) < 8) (ht: ∃ t : ℤ, x = t * y) : x / y = -2 := 
by
  sorry

end frac_val_of_x_y_l40_40945


namespace range_of_m_l40_40051

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, mx^2 - 4*x + 1 = 0 ∧ ∀ y : ℝ, mx^2 - 4*x + 1 = 0 → y = x) → m ≤ 4 :=
sorry

end range_of_m_l40_40051


namespace sum_of_first_6_terms_l40_40965

-- Definitions based on given conditions
def arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ := a1 + d * (n - 1)

-- The conditions provided in the problem
def condition_1 (a1 d : ℤ) : Prop := arithmetic_sequence a1 d 2 + arithmetic_sequence a1 d 4 = 4
def condition_2 (a1 d : ℤ) : Prop := arithmetic_sequence a1 d 3 + arithmetic_sequence a1 d 5 = 10

-- The sum of the first 6 terms of the arithmetic sequence
def sum_first_6_terms (a1 d : ℤ) : ℤ := 6 * a1 + 15 * d

-- The theorem to prove
theorem sum_of_first_6_terms (a1 d : ℤ) 
  (h1 : condition_1 a1 d)
  (h2 : condition_2 a1 d) :
  sum_first_6_terms a1 d = 21 := sorry

end sum_of_first_6_terms_l40_40965


namespace value_of_some_number_l40_40650

theorem value_of_some_number (a : ℤ) (h : a = 105) :
  (a ^ 3 = 3 * (5 ^ 3) * (3 ^ 2) * (7 ^ 2)) :=
by {
  sorry
}

end value_of_some_number_l40_40650


namespace solve_system_l40_40416

theorem solve_system :
  (∀ x y : ℝ, 
    (x^2 * y - x * y^2 - 3 * x + 3 * y + 1 = 0 ∧
     x^3 * y - x * y^3 - 3 * x^2 + 3 * y^2 + 3 = 0) → (x, y) = (2, 1)) :=
by simp [← solve_system]; sorry

end solve_system_l40_40416


namespace last_two_digits_of_7_pow_2017_l40_40109

theorem last_two_digits_of_7_pow_2017 :
  (7 ^ 2017) % 100 = 7 :=
sorry

end last_two_digits_of_7_pow_2017_l40_40109


namespace symmetric_circle_equation_l40_40614

noncomputable def equation_of_symmetric_circle (C₁ : ℝ → ℝ → Prop) (l : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, C₁ x y ↔ x^2 + y^2 - 4 * x - 8 * y + 19 = 0

theorem symmetric_circle_equation :
  ∀ (C₁ : ℝ → ℝ → Prop) (l : ℝ → ℝ → Prop),
  equation_of_symmetric_circle C₁ l →
  (∀ x y, l x y ↔ x + 2 * y - 5 = 0) →
  ∃ C₂ : ℝ → ℝ → Prop, (∀ x y, C₂ x y ↔ x^2 + y^2 = 1) :=
by
  intros C₁ l hC₁ hₗ
  sorry

end symmetric_circle_equation_l40_40614


namespace sum_of_largest_three_consecutive_numbers_l40_40308

theorem sum_of_largest_three_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 := 
by
  sorry

end sum_of_largest_three_consecutive_numbers_l40_40308


namespace time_for_Dawson_l40_40301

variable (D : ℝ)
variable (Henry_time : ℝ := 7)
variable (avg_time : ℝ := 22.5)

theorem time_for_Dawson (h : avg_time = (D + Henry_time) / 2) : D = 38 := 
by 
  sorry

end time_for_Dawson_l40_40301


namespace ratio_of_areas_of_circles_l40_40901

-- Given conditions
variables (R_C R_D : ℝ) -- Radii of circles C and D respectively
variables (L : ℝ) -- Common length of the arcs

-- Equivalent arc condition
def arc_length_condition : Prop :=
  (60 / 360) * (2 * Real.pi * R_C) = L ∧ (40 / 360) * (2 * Real.pi * R_D) = L

-- Goal: ratio of areas
def area_ratio : Prop :=
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = (9 / 4)

-- Problem statement
theorem ratio_of_areas_of_circles (R_C R_D L : ℝ) (hc : arc_length_condition R_C R_D L) :
  area_ratio R_C R_D :=
by
  sorry

end ratio_of_areas_of_circles_l40_40901


namespace air_conditioned_rooms_fraction_l40_40472

theorem air_conditioned_rooms_fraction (R A : ℝ) (h1 : 3/4 * R = 3/4 * R - 1/4 * R)
                                        (h2 : 2/3 * A = 2/3 * A - 1/3 * A)
                                        (h3 : 1/3 * A = 0.8 * 1/4 * R) :
    A / R = 3 / 5 :=
by
  -- Proof content goes here
  sorry

end air_conditioned_rooms_fraction_l40_40472


namespace total_weight_proof_l40_40873

-- Define molar masses
def molar_mass_C : ℝ := 12.01
def molar_mass_H : ℝ := 1.008

-- Define moles of elements in each compound
def moles_C4H10 : ℕ := 8
def moles_C3H8 : ℕ := 5
def moles_CH4 : ℕ := 3

-- Define the molar masses of each compound
def molar_mass_C4H10 : ℝ := 4 * molar_mass_C + 10 * molar_mass_H
def molar_mass_C3H8 : ℝ := 3 * molar_mass_C + 8 * molar_mass_H
def molar_mass_CH4 : ℝ := 1 * molar_mass_C + 4 * molar_mass_H

-- Define the total weight
def total_weight : ℝ :=
  moles_C4H10 * molar_mass_C4H10 +
  moles_C3H8 * molar_mass_C3H8 +
  moles_CH4 * molar_mass_CH4

theorem total_weight_proof :
  total_weight = 733.556 := by
  sorry

end total_weight_proof_l40_40873


namespace greatest_integer_b_for_no_real_roots_l40_40076

theorem greatest_integer_b_for_no_real_roots (b : ℤ) :
  (∀ x : ℝ, x^2 + (b:ℝ)*x + 10 ≠ 0) ↔ b ≤ 6 :=
sorry

end greatest_integer_b_for_no_real_roots_l40_40076


namespace percentage_increase_visitors_l40_40040

theorem percentage_increase_visitors
  (original_visitors : ℕ)
  (original_fee : ℝ := 1)
  (fee_reduction : ℝ := 0.25)
  (visitors_increase : ℝ := 0.20) :
  ((original_visitors + (visitors_increase * original_visitors)) / original_visitors - 1) * 100 = 20 := by
  sorry

end percentage_increase_visitors_l40_40040


namespace daisies_per_bouquet_is_7_l40_40158

/-
Each bouquet of roses contains 12 roses.
Each bouquet of daisies contains an equal number of daisies.
The flower shop sells 20 bouquets today.
10 of the bouquets are rose bouquets and 10 are daisy bouquets.
The flower shop sold 190 flowers in total today.
-/

def num_daisies_per_bouquet (roses_per_bouquet daisies_sold bouquets_sold total_roses_sold total_flowers_sold : ℕ) : ℕ :=
  (total_flowers_sold - total_roses_sold) / bouquets_sold 

theorem daisies_per_bouquet_is_7 :
  ∀ (roses_per_bouquet daisies_sold bouquets_sold total_roses_sold total_flowers_sold : ℕ),
  (roses_per_bouquet = 12) →
  (bouquets_sold = 10) →
  (total_roses_sold = bouquets_sold * roses_per_bouquet) →
  (total_flowers_sold = 190) →
  num_daisies_per_bouquet roses_per_bouquet daisies_sold bouquets_sold total_roses_sold total_flowers_sold = 7 :=
by
  intros
  -- Placeholder for the actual proof
  sorry

end daisies_per_bouquet_is_7_l40_40158


namespace bales_stored_in_barn_l40_40734

-- Defining the conditions
def bales_initial : Nat := 28
def bales_stacked : Nat := 28
def bales_already_there : Nat := 54

-- Formulate the proof statement
theorem bales_stored_in_barn : bales_already_there + bales_stacked = 82 := by
  sorry

end bales_stored_in_barn_l40_40734


namespace value_b_minus_a_l40_40013

theorem value_b_minus_a (a b : ℝ) (h₁ : a + b = 507) (h₂ : (a - b) / b = 1 / 7) : b - a = -34.428571 :=
by
  sorry

end value_b_minus_a_l40_40013


namespace trapezoid_perimeter_is_correct_l40_40263

noncomputable def trapezoid_perimeter_proof : ℝ :=
  let EF := 60
  let θ := Real.pi / 4 -- 45 degrees in radians
  let h := 30 * Real.sqrt 2
  let GH := EF + 2 * h / Real.tan θ
  let EG := h / Real.tan θ
  EF + GH + 2 * EG -- Perimeter calculation

theorem trapezoid_perimeter_is_correct :
  trapezoid_perimeter_proof = 180 + 60 * Real.sqrt 2 := 
by
  sorry

end trapezoid_perimeter_is_correct_l40_40263


namespace rose_spent_on_food_l40_40095

theorem rose_spent_on_food (T : ℝ) 
  (h_clothing : 0.5 * T = 0.5 * T)
  (h_other_items : 0.3 * T = 0.3 * T)
  (h_total_tax : 0.044 * T = 0.044 * T)
  (h_tax_clothing : 0.04 * 0.5 * T = 0.02 * T)
  (h_tax_other_items : 0.08 * 0.3 * T = 0.024 * T) :
  (0.2 * T = T - (0.5 * T + 0.3 * T)) :=
by sorry

end rose_spent_on_food_l40_40095


namespace area_of_park_l40_40536

-- Definitions of conditions
def ratio_length_breadth (L B : ℝ) : Prop := L / B = 1 / 3
def cycling_time_distance (speed time perimeter : ℝ) : Prop := perimeter = speed * time

theorem area_of_park :
  ∃ (L B : ℝ),
    ratio_length_breadth L B ∧
    cycling_time_distance 12 (8 / 60) (2 * (L + B)) ∧
    L * B = 120000 := by
  sorry

end area_of_park_l40_40536


namespace initial_bacteria_count_l40_40228

theorem initial_bacteria_count (d: ℕ) (t_final: ℕ) (N_final: ℕ) 
    (h1: t_final = 4 * 60)  -- 4 minutes equals 240 seconds
    (h2: d = 15)            -- Doubling interval is 15 seconds
    (h3: N_final = 2097152) -- Final bacteria count is 2,097,152
    :
    ∃ n: ℕ, N_final = n * 2^((t_final / d)) ∧ n = 32 :=
by
  sorry

end initial_bacteria_count_l40_40228


namespace violet_needs_water_l40_40964

/-- Violet needs 800 ml of water per hour hiked, her dog needs 400 ml of water per hour,
    and they can hike for 4 hours. We need to prove that Violet needs 4.8 liters of water
    for the hike. -/
theorem violet_needs_water (hiking_hours : ℝ)
  (violet_water_per_hour : ℝ)
  (dog_water_per_hour : ℝ)
  (violet_water_needed : ℝ)
  (dog_water_needed : ℝ)
  (total_water_needed_ml : ℝ)
  (total_water_needed_liters : ℝ) :
  hiking_hours = 4 ∧
  violet_water_per_hour = 800 ∧
  dog_water_per_hour = 400 ∧
  violet_water_needed = 3200 ∧
  dog_water_needed = 1600 ∧
  total_water_needed_ml = 4800 ∧
  total_water_needed_liters = 4.8 →
  total_water_needed_liters = 4.8 :=
by sorry

end violet_needs_water_l40_40964


namespace sum_of_roots_eq_14_l40_40367

theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, (x - 7)^2 = 16 → (x = 11 ∨ x = 3)) → 11 + 3 = 14 :=
by
  intro h
  have x1 : 11 = 11 := rfl
  have x2 : 3 = 3 := rfl
  exact rfl

end sum_of_roots_eq_14_l40_40367


namespace carpet_area_proof_l40_40760

noncomputable def carpet_area (main_room_length_ft : ℕ) (main_room_width_ft : ℕ)
  (corridor_length_ft : ℕ) (corridor_width_ft : ℕ) (feet_per_yard : ℕ) : ℚ :=
  let main_room_length_yd := main_room_length_ft / feet_per_yard
  let main_room_width_yd := main_room_width_ft / feet_per_yard
  let corridor_length_yd := corridor_length_ft / feet_per_yard
  let corridor_width_yd := corridor_width_ft / feet_per_yard
  let main_room_area_yd2 := main_room_length_yd * main_room_width_yd
  let corridor_area_yd2 := corridor_length_yd * corridor_width_yd
  main_room_area_yd2 + corridor_area_yd2

theorem carpet_area_proof : carpet_area 15 12 10 3 3 = 23.33 :=
by
  -- Proof steps go here
  sorry

end carpet_area_proof_l40_40760


namespace geometric_sequence_k_squared_l40_40678

theorem geometric_sequence_k_squared (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a (n+1) = a n * r) (h5 : a 5 * a 8 * a 11 = k) : 
  k^2 = a 5 * a 6 * a 7 * a 9 * a 10 * a 11 := by
  sorry

end geometric_sequence_k_squared_l40_40678


namespace Andy_earnings_l40_40552

/-- Andy's total earnings during an 8-hour shift. --/
theorem Andy_earnings (hours : ℕ) (hourly_wage : ℕ) (num_racquets : ℕ) (pay_per_racquet : ℕ)
  (num_grommets : ℕ) (pay_per_grommet : ℕ) (num_stencils : ℕ) (pay_per_stencil : ℕ)
  (h_shift : hours = 8) (h_hourly : hourly_wage = 9) (h_racquets : num_racquets = 7)
  (h_pay_racquets : pay_per_racquet = 15) (h_grommets : num_grommets = 2)
  (h_pay_grommets : pay_per_grommet = 10) (h_stencils : num_stencils = 5)
  (h_pay_stencils : pay_per_stencil = 1) :
  (hours * hourly_wage + num_racquets * pay_per_racquet + num_grommets * pay_per_grommet +
  num_stencils * pay_per_stencil) = 202 :=
by
  sorry

end Andy_earnings_l40_40552


namespace min_y_value_l40_40415

open Real

noncomputable def f (x : ℝ) : ℝ := (x^2 + 7*x + 10) / (x + 1)

theorem min_y_value : ∀ x > -1, f x ≥ 9 :=
by sorry

end min_y_value_l40_40415


namespace ed_pets_count_l40_40715

theorem ed_pets_count : 
  let dogs := 2 
  let cats := 3 
  let fish := 2 * (cats + dogs) 
  let birds := dogs * cats 
  dogs + cats + fish + birds = 21 := 
by
  sorry

end ed_pets_count_l40_40715


namespace real_solutions_of_polynomial_l40_40995

theorem real_solutions_of_polynomial (b : ℝ) :
  b < -4 → ∃! x : ℝ, x^3 - b * x^2 - 4 * b * x + b^2 - 4 = 0 :=
by
  sorry

end real_solutions_of_polynomial_l40_40995


namespace pool_one_quarter_capacity_at_6_l40_40048

-- Variables and parameters
variables (volume : ℕ → ℝ) (T : ℕ)

-- Conditions
def doubles_every_hour : Prop :=
  ∀ t, volume (t + 1) = 2 * volume t

def full_capacity_at_8 : Prop :=
  volume 8 = T

def one_quarter_capacity (t : ℕ) : Prop :=
  volume t = T / 4

-- Theorem to prove
theorem pool_one_quarter_capacity_at_6 (h1 : doubles_every_hour volume) (h2 : full_capacity_at_8 volume T) : one_quarter_capacity volume T 6 :=
sorry

end pool_one_quarter_capacity_at_6_l40_40048


namespace initial_cd_count_l40_40630

variable (X : ℕ)

theorem initial_cd_count (h1 : (2 / 3 : ℝ) * X + 8 = 22) : X = 21 :=
by
  sorry

end initial_cd_count_l40_40630


namespace suresh_borrowed_amount_l40_40083

theorem suresh_borrowed_amount 
  (P: ℝ)
  (i1 i2 i3: ℝ)
  (t1 t2 t3: ℝ)
  (total_interest: ℝ)
  (h1 : i1 = 0.12) 
  (h2 : t1 = 3)
  (h3 : i2 = 0.09)
  (h4 : t2 = 5)
  (h5 : i3 = 0.13)
  (h6 : t3 = 3)
  (h_total : total_interest = 8160) 
  (h_interest_eq : total_interest = P * i1 * t1 + P * i2 * t2 + P * i3 * t3)
  : P = 6800 :=
by
  sorry

end suresh_borrowed_amount_l40_40083


namespace convert_fraction_to_decimal_l40_40350

noncomputable def fraction_to_decimal (num : ℕ) (den : ℕ) : ℝ :=
  (num : ℝ) / (den : ℝ)

theorem convert_fraction_to_decimal :
  fraction_to_decimal 5 16 = 0.3125 :=
by
  sorry

end convert_fraction_to_decimal_l40_40350


namespace population_net_increase_l40_40613

-- Define the birth rate and death rate conditions
def birth_rate := 4 / 2 -- people per second
def death_rate := 2 / 2 -- people per second
def net_increase_per_sec := birth_rate - death_rate -- people per second

-- Define the duration of one day in seconds
def seconds_in_a_day := 24 * 3600 -- seconds

-- Define the problem to prove
theorem population_net_increase :
  net_increase_per_sec * seconds_in_a_day = 86400 :=
by
  sorry

end population_net_increase_l40_40613


namespace diagonal_length_of_regular_hexagon_l40_40223

theorem diagonal_length_of_regular_hexagon (
  side_length : ℝ
) (h_side_length : side_length = 12) : 
  ∃ DA, DA = 12 * Real.sqrt 3 :=
by 
  sorry

end diagonal_length_of_regular_hexagon_l40_40223


namespace original_number_of_laborers_l40_40526

theorem original_number_of_laborers (L : ℕ) 
  (h : L * 9 = (L - 6) * 15) : L = 15 :=
sorry

end original_number_of_laborers_l40_40526


namespace geese_ratio_l40_40052

/-- Define the problem conditions --/

def lily_ducks := 20
def lily_geese := 10

def rayden_ducks : ℕ := 3 * lily_ducks
def total_lily_animals := lily_ducks + lily_geese
def total_rayden_animals := total_lily_animals + 70
def rayden_geese := total_rayden_animals - rayden_ducks

/-- Prove the desired ratio of the number of geese Rayden bought to the number of geese Lily bought --/
theorem geese_ratio : rayden_geese / lily_geese = 4 :=
sorry

end geese_ratio_l40_40052


namespace football_team_total_members_l40_40521

-- Definitions from the problem conditions
def initialMembers : ℕ := 42
def newMembers : ℕ := 17

-- Mathematical equivalent proof problem
theorem football_team_total_members : initialMembers + newMembers = 59 := by
  sorry

end football_team_total_members_l40_40521


namespace average_temps_l40_40267

-- Define the temperature lists
def temps_C : List ℚ := [
  37.3, 37.2, 36.9, -- Sunday
  36.6, 36.9, 37.1, -- Monday
  37.1, 37.3, 37.2, -- Tuesday
  36.8, 37.3, 37.5, -- Wednesday
  37.1, 37.7, 37.3, -- Thursday
  37.5, 37.4, 36.9, -- Friday
  36.9, 37.0, 37.1  -- Saturday
]

def temps_K : List ℚ := [
  310.4, 310.3, 310.0, -- Sunday
  309.8, 310.0, 310.2, -- Monday
  310.2, 310.4, 310.3, -- Tuesday
  309.9, 310.4, 310.6, -- Wednesday
  310.2, 310.8, 310.4, -- Thursday
  310.6, 310.5, 310.0, -- Friday
  310.0, 310.1, 310.2  -- Saturday
]

def temps_R : List ℚ := [
  558.7, 558.6, 558.1, -- Sunday
  557.7, 558.1, 558.3, -- Monday
  558.3, 558.7, 558.6, -- Tuesday
  558.0, 558.7, 559.1, -- Wednesday
  558.3, 559.4, 558.7, -- Thursday
  559.1, 558.9, 558.1, -- Friday
  558.1, 558.2, 558.3  -- Saturday
]

-- Calculate the average of a list of temperatures
def average (temps : List ℚ) : ℚ :=
  temps.sum / temps.length

-- Define the average temperatures
def avg_C := average temps_C
def avg_K := average temps_K
def avg_R := average temps_R

-- State that the computed averages are equal to the provided values
theorem average_temps :
  avg_C = 37.1143 ∧
  avg_K = 310.1619 ∧
  avg_R = 558.2524 :=
by
  -- Proof can be completed here
  sorry

end average_temps_l40_40267


namespace solve_fractional_equation_l40_40872

theorem solve_fractional_equation (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) :
  (3 / (x - 1) = 5 + 3 * x / (1 - x)) → (x = 4) :=
by 
  intro h
  -- proof steps would go here
  sorry

end solve_fractional_equation_l40_40872


namespace no_such_functions_l40_40816

open Real

theorem no_such_functions : ¬ ∃ (f g : ℝ → ℝ), (∀ x y : ℝ, f (x^2 + g y) - f (x^2) + g (y) - g (x) ≤ 2 * y) ∧ (∀ x : ℝ, f (x) ≥ x^2) := by
  sorry

end no_such_functions_l40_40816


namespace rectangle_area_l40_40994

theorem rectangle_area (l w : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 250) : l * w = 2500 :=
  sorry

end rectangle_area_l40_40994


namespace children_on_bus_after_stops_l40_40467

-- Define the initial number of children and changes at each stop
def initial_children := 128
def first_stop_addition := 67
def second_stop_subtraction := 34
def third_stop_addition := 54

-- Prove that the number of children on the bus after all the stops is 215
theorem children_on_bus_after_stops :
  initial_children + first_stop_addition - second_stop_subtraction + third_stop_addition = 215 := by
  -- The proof is omitted
  sorry

end children_on_bus_after_stops_l40_40467


namespace find_x_l40_40171

-- Define the problem conditions.
def workers := ℕ
def gadgets := ℕ
def gizmos := ℕ
def hours := ℕ

-- Given conditions
def condition1 (g h : ℝ) := (1 / g = 2) ∧ (1 / h = 3)
def condition2 (g h : ℝ) := (100 * 3 / g = 900) ∧ (100 * 3 / h = 600)
def condition3 (x : ℕ) (g h : ℝ) := (40 * 4 / g = x) ∧ (40 * 4 / h = 480)

-- Proof problem statement
theorem find_x (g h : ℝ) (x : ℕ) : 
  condition1 g h → condition2 g h → condition3 x g h → x = 320 :=
by 
  intros h1 h2 h3
  sorry

end find_x_l40_40171


namespace max_a_l40_40268

variable {a x : ℝ}

theorem max_a (h : x^2 - 2 * x - 3 > 0 → x < a ∧ ¬ (x < a → x^2 - 2 * x - 3 > 0)) : a = 3 :=
sorry

end max_a_l40_40268


namespace crayons_difference_l40_40190

def initial_crayons : ℕ := 8597
def crayons_given : ℕ := 7255
def crayons_lost : ℕ := 3689

theorem crayons_difference : crayons_given - crayons_lost = 3566 := by
  sorry

end crayons_difference_l40_40190


namespace mean_equality_l40_40570

-- Define the mean calculation
def mean (a b c : ℕ) : ℚ := (a + b + c) / 3

-- The given conditions
theorem mean_equality (z : ℕ) (y : ℕ) (hz : z = 24) :
  mean 8 15 21 = mean 16 z y → y = 4 :=
by
  sorry

end mean_equality_l40_40570


namespace angle_bisector_slope_l40_40254

theorem angle_bisector_slope
  (m₁ m₂ : ℝ) (h₁ : m₁ = 2) (h₂ : m₂ = -1) (k : ℝ)
  (h_k : k = (m₁ + m₂ + Real.sqrt ((m₁ - m₂)^2 + 4)) / 2) :
  k = (1 + Real.sqrt 13) / 2 :=
by
  rw [h₁, h₂] at h_k
  sorry

end angle_bisector_slope_l40_40254


namespace percentage_increase_soda_price_l40_40648

theorem percentage_increase_soda_price
  (C_new : ℝ) (S_new : ℝ) (C_increase : ℝ) (C_total_before : ℝ)
  (h1 : C_new = 20)
  (h2: S_new = 6)
  (h3: C_increase = 0.25)
  (h4: C_new * (1 - C_increase) + S_new * (1 + (S_new / (S_new * (1 + (S_new / (S_new * 0.5)))))) = C_total_before) : 
  (S_new - S_new * (1 - C_increase) * 100 / (S_new * (1 + 0.5)) * C_total_before) = 50 := 
by 
  -- This is where the proof would go.
  sorry

end percentage_increase_soda_price_l40_40648


namespace units_digit_7_pow_3_pow_4_l40_40909

theorem units_digit_7_pow_3_pow_4 :
  (7 ^ (3 ^ 4)) % 10 = 7 :=
by
  -- Here's the proof placeholder
  sorry

end units_digit_7_pow_3_pow_4_l40_40909


namespace math_problem_l40_40116

theorem math_problem 
  (a b c d : ℝ) 
  (h1 : a ≥ b) 
  (h2 : b ≥ c) 
  (h3 : c ≥ d) 
  (h4 : d > 0) 
  (h5 : a + b + c + d = 1) : 
  (a + 2*b + 3*c + 4*d) * a^a * b^b * c^c * d^d < 1 := 
sorry

end math_problem_l40_40116


namespace solve_system_l40_40920

theorem solve_system (x y : ℝ) (h1 : x + 2 * y = 8) (h2 : 2 * x + y = 7) : x + y = 5 :=
by
  sorry

end solve_system_l40_40920


namespace remainder_2365947_div_8_l40_40602

theorem remainder_2365947_div_8 : (2365947 % 8) = 3 :=
by
  sorry

end remainder_2365947_div_8_l40_40602


namespace total_number_of_cows_l40_40203

theorem total_number_of_cows (n : ℕ) 
  (h1 : n > 0) 
  (h2 : (1/3) * n + (1/6) * n + (1/8) * n + 9 = n) : n = 216 :=
sorry

end total_number_of_cows_l40_40203


namespace quadratic_inequality_solution_range_l40_40353

theorem quadratic_inequality_solution_range (a : ℝ) :
  (∃ x, 1 < x ∧ x < 4 ∧ x^2 - 4 * x - 2 - a > 0) → a < -2 :=
sorry

end quadratic_inequality_solution_range_l40_40353


namespace no_polynomial_exists_l40_40823

open Polynomial

theorem no_polynomial_exists (a b c : ℤ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) :
  ¬ ∃ (P : ℤ[X]), P.eval a = b ∧ P.eval b = c ∧ P.eval c = a :=
sorry

end no_polynomial_exists_l40_40823


namespace taller_pot_shadow_length_l40_40458

theorem taller_pot_shadow_length
  (height1 shadow1 height2 : ℝ)
  (h1 : height1 = 20)
  (h2 : shadow1 = 10)
  (h3 : height2 = 40) :
  ∃ shadow2 : ℝ, height2 / shadow2 = height1 / shadow1 ∧ shadow2 = 20 :=
by
  -- Since Lean requires proofs for existential statements,
  -- we add "sorry" to skip the proof.
  sorry

end taller_pot_shadow_length_l40_40458


namespace students_still_in_school_l40_40111

theorem students_still_in_school
  (total_students : ℕ)
  (half_trip : total_students / 2 > 0)
  (half_remaining_sent_home : (total_students / 2) / 2 > 0)
  (total_students_val : total_students = 1000)
  :
  let students_still_in_school := total_students - (total_students / 2) - ((total_students - (total_students / 2)) / 2)
  students_still_in_school = 250 :=
by
  sorry

end students_still_in_school_l40_40111


namespace find_f_on_interval_l40_40022

/-- Representation of periodic and even functions along with specific interval definition -/
noncomputable def f (x : ℝ) : ℝ := 
if 2 ≤ x ∧ x ≤ 3 then -2*(x-3)^2 + 4 else 0 -- Define f(x) on [2,3], otherwise undefined

/-- Main proof statement -/
theorem find_f_on_interval :
  (∀ x, f x = f (x + 2)) ∧  -- f(x) is periodic with period 2
  (∀ x, f x = f (-x)) ∧   -- f(x) is even
  (∀ x, 2 ≤ x ∧ x ≤ 3 → f x = -2*(x-3)^2 + 4) →
  (∀ x, 0 ≤ x ∧ x ≤ 2 → f x = -2*(x-1)^2 + 4) :=
sorry

end find_f_on_interval_l40_40022


namespace findSolutions_l40_40517

-- Define the given mathematical problem
def originalEquation (x : ℝ) : Prop :=
  ((x - 3) * (x - 4) * (x - 5) * (x - 6) * (x - 5) * (x - 4) * (x - 3)) / ((x - 4) * (x - 6) * (x - 4)) = 1

-- Define the conditions where the equation is valid
def validCondition (x : ℝ) : Prop :=
  x ≠ 4 ∧ x ≠ 6

-- Define the set of solutions
def solutions (x : ℝ) : Prop :=
  x = 4 + Real.sqrt 2 ∨ x = 4 - Real.sqrt 2

-- The theorem stating the correct set of solutions
theorem findSolutions (x : ℝ) : originalEquation x ∧ validCondition x ↔ solutions x :=
by sorry

end findSolutions_l40_40517


namespace binary_mult_div_to_decimal_l40_40304

theorem binary_mult_div_to_decimal:
  let n1 := 2 ^ 5 + 2 ^ 4 + 2 ^ 2 + 2 ^ 1 -- This represents 101110_2
  let n2 := 2 ^ 6 + 2 ^ 4 + 2 ^ 2         -- This represents 1010100_2
  let d := 2 ^ 2                          -- This represents 100_2
  n1 * n2 / d = 2995 := 
by
  sorry

end binary_mult_div_to_decimal_l40_40304


namespace number_doubled_is_12_l40_40606

theorem number_doubled_is_12 (A B C D E : ℝ) (h1 : (A + B + C + D + E) / 5 = 6.8)
  (X : ℝ) (h2 : ((A + B + C + D + E - X) + 2 * X) / 5 = 9.2) : X = 12 :=
by
  sorry

end number_doubled_is_12_l40_40606


namespace interest_rate_determination_l40_40992

-- Problem statement
theorem interest_rate_determination (P r : ℝ) :
  (50 = P * r * 2) ∧ (51.25 = P * ((1 + r) ^ 2 - 1)) → r = 0.05 :=
by
  intros h
  sorry

end interest_rate_determination_l40_40992


namespace remainder_when_four_times_n_minus_9_divided_by_11_l40_40471

theorem remainder_when_four_times_n_minus_9_divided_by_11 
  (n : ℤ) (h : n % 11 = 4) : (4 * n - 9) % 11 = 7 := by
  sorry

end remainder_when_four_times_n_minus_9_divided_by_11_l40_40471


namespace Megan_deleted_files_l40_40509

theorem Megan_deleted_files (initial_files folders files_per_folder deleted_files : ℕ) 
    (h1 : initial_files = 93) 
    (h2 : folders = 9)
    (h3 : files_per_folder = 8) 
    (h4 : deleted_files = initial_files - folders * files_per_folder) : 
  deleted_files = 21 :=
by
  sorry

end Megan_deleted_files_l40_40509


namespace disproof_of_Alitta_l40_40494

-- Definition: A prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition: A number is odd
def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

-- The value is a specific set of odd primes including 11
def contains (p : ℕ) : Prop :=
  p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 11

-- Main statement: There exists an odd prime p in the given options such that p^2 - 2 is not a prime
theorem disproof_of_Alitta :
  ∃ p : ℕ, contains p ∧ is_prime p ∧ is_odd p ∧ ¬ is_prime (p^2 - 2) :=
by
  sorry

end disproof_of_Alitta_l40_40494


namespace operation_4_3_is_5_l40_40172

def custom_operation (m n : ℕ) : ℕ := n ^ 2 - m

theorem operation_4_3_is_5 : custom_operation 4 3 = 5 :=
by
  -- Proof goes here
  sorry

end operation_4_3_is_5_l40_40172


namespace rectangle_length_width_l40_40387

theorem rectangle_length_width (x y : ℝ) 
  (h1 : 2 * x + 2 * y = 16) 
  (h2 : x - y = 1) : 
  x = 4.5 ∧ y = 3.5 :=
by {
  sorry
}

end rectangle_length_width_l40_40387


namespace horse_distance_traveled_l40_40610

theorem horse_distance_traveled :
  let r2 := 12
  let n2 := 120
  let D2 := n2 * 2 * Real.pi * r2
  D2 = 2880 * Real.pi :=
by
  sorry

end horse_distance_traveled_l40_40610


namespace min_val_xy_l40_40738

theorem min_val_xy (x y : ℝ) 
  (h : 2 * (Real.cos (x + y - 1))^2 = ((x + 1)^2 + (y - 1)^2 - 2 * x * y) / (x - y + 1)) : 
  xy ≥ (1 / 4) :=
sorry

end min_val_xy_l40_40738


namespace maximum_value_of_z_l40_40739

theorem maximum_value_of_z :
  ∃ x y : ℝ, (x - y ≥ 0) ∧ (x + y ≤ 2) ∧ (y ≥ 0) ∧ (∀ u v : ℝ, (u - v ≥ 0) ∧ (u + v ≤ 2) ∧ (v ≥ 0) → 3 * u - v ≤ 6) :=
by
  sorry

end maximum_value_of_z_l40_40739


namespace find_special_numbers_l40_40448

def is_digit_sum_equal (n m : Nat) : Prop := 
  (n.digits 10).sum = (m.digits 10).sum

def is_valid_number (n : Nat) : Prop := 
  100 ≤ n ∧ n ≤ 999 ∧ is_digit_sum_equal n (6 * n)

theorem find_special_numbers :
  {n : Nat | is_valid_number n} = {117, 135} :=
sorry

end find_special_numbers_l40_40448


namespace distance_from_star_l40_40783

def speed_of_light : ℝ := 3 * 10^5 -- km/s
def time_years : ℝ := 4 -- years
def seconds_per_year : ℝ := 3 * 10^7 -- s

theorem distance_from_star :
  let distance := speed_of_light * (time_years * seconds_per_year)
  distance = 3.6 * 10^13 :=
by
  sorry

end distance_from_star_l40_40783


namespace find_x_l40_40404

theorem find_x (x : ℝ) (h1 : 0 < x) (h2 : ⌈x⌉₊ * x = 198) : x = 13.2 :=
by
  sorry

end find_x_l40_40404


namespace sun_city_population_greater_than_twice_roseville_l40_40108

-- Conditions
def willowdale_population : ℕ := 2000
def roseville_population : ℕ := 3 * willowdale_population - 500
def sun_city_population : ℕ := 12000

-- Theorem
theorem sun_city_population_greater_than_twice_roseville :
  sun_city_population = 2 * roseville_population + 1000 :=
by
  -- The proof is omitted as per the problem statement
  sorry

end sun_city_population_greater_than_twice_roseville_l40_40108


namespace problem_l40_40732

variable {a b c : ℝ}

theorem problem (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  a + 4 / b ≤ -4 ∨ b + 4 / c ≤ -4 ∨ c + 4 / a ≤ -4 := 
sorry

end problem_l40_40732


namespace compare_powers_l40_40955

def n1 := 22^44
def n2 := 33^33
def n3 := 44^22

theorem compare_powers : n1 > n2 ∧ n2 > n3 := by
  sorry

end compare_powers_l40_40955


namespace loan_proof_l40_40170

-- Definition of the conditions
def interest_rate_year_1 : ℝ := 0.10
def interest_rate_year_2 : ℝ := 0.12
def interest_rate_year_3 : ℝ := 0.14
def total_interest_paid : ℝ := 5400

-- Theorem proving the results
theorem loan_proof (P : ℝ) 
                   (annual_repayment : ℝ)
                   (remaining_principal : ℝ) :
  (interest_rate_year_1 * P) + 
  (interest_rate_year_2 * P) + 
  (interest_rate_year_3 * P) = total_interest_paid →
  3 * annual_repayment = total_interest_paid →
  remaining_principal = P →
  P = 15000 ∧ 
  annual_repayment = 1800 ∧ 
  remaining_principal = 15000 :=
by
  intros h1 h2 h3
  sorry

end loan_proof_l40_40170


namespace parametric_line_segment_computation_l40_40400

theorem parametric_line_segment_computation :
  ∃ (a b c d : ℝ), 
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
   (-3, 10) = (a * t + b, c * t + d) ∧
   (4, 16) = (a * 1 + b, c * 1 + d)) ∧
  (b = -3) ∧ (d = 10) ∧ 
  (a + b = 4) ∧ (c + d = 16) ∧ 
  (a^2 + b^2 + c^2 + d^2 = 194) :=
sorry

end parametric_line_segment_computation_l40_40400


namespace number_of_girls_l40_40762

theorem number_of_girls (boys girls : ℕ) (h1 : boys = 337) (h2 : girls = boys + 402) : girls = 739 := by
  sorry

end number_of_girls_l40_40762


namespace inequality_of_triangle_tangents_l40_40897

theorem inequality_of_triangle_tangents
  (a b c x y z : ℝ)
  (h1 : a = y + z)
  (h2 : b = x + z)
  (h3 : c = x + y)
  (h_order : a ≥ b ∧ b ≥ c)
  (h_tangents : z ≥ y ∧ y ≥ x) :
  (a * z + b * y + c * x ≥ (a^2 + b^2 + c^2) / 2) ∧
  ((a^2 + b^2 + c^2) / 2 ≥ a * x + b * y + c * z) :=
sorry

end inequality_of_triangle_tangents_l40_40897


namespace converse_statement_l40_40731

theorem converse_statement (a : ℝ) : (a > 2018 → a > 2017) ↔ (a > 2017 → a > 2018) :=
by
  sorry

end converse_statement_l40_40731


namespace line_parabola_one_intersection_l40_40445

theorem line_parabola_one_intersection (k : ℝ) : 
  ((∃ (x y : ℝ), y = k * x - 1 ∧ y^2 = 4 * x ∧ (∀ u v : ℝ, u ≠ x → v = k * u - 1 → v^2 ≠ 4 * u)) ↔ (k = 0 ∨ k = 1)) := 
sorry

end line_parabola_one_intersection_l40_40445


namespace focal_length_of_curve_l40_40511

theorem focal_length_of_curve : 
  (∀ θ : ℝ, ∃ x y : ℝ, x = 2 * Real.cos θ ∧ y = Real.sin θ) →
  ∃ f : ℝ, f = 2 * Real.sqrt 3 :=
by sorry

end focal_length_of_curve_l40_40511


namespace dinner_cost_per_kid_l40_40621

theorem dinner_cost_per_kid
  (row_ears : ℕ)
  (seeds_bag : ℕ)
  (seeds_ear : ℕ)
  (pay_row : ℝ)
  (bags_used : ℕ)
  (dinner_fraction : ℝ)
  (h1 : row_ears = 70)
  (h2 : seeds_bag = 48)
  (h3 : seeds_ear = 2)
  (h4 : pay_row = 1.5)
  (h5 : bags_used = 140)
  (h6 : dinner_fraction = 0.5) :
  ∃ (dinner_cost : ℝ), dinner_cost = 36 :=
by
  sorry

end dinner_cost_per_kid_l40_40621


namespace largest_integer_x_l40_40079

theorem largest_integer_x (x : ℤ) :
  (x ^ 2 - 11 * x + 28 < 0) → x ≤ 6 := sorry

end largest_integer_x_l40_40079


namespace number_of_attendees_choosing_water_l40_40892

variables {total_attendees : ℕ} (juice_percent water_percent : ℚ)

-- Conditions
def attendees_juice (total_attendees : ℕ) : ℚ := 0.7 * total_attendees
def attendees_water (total_attendees : ℕ) : ℚ := 0.3 * total_attendees
def attendees_juice_given := (attendees_juice total_attendees) = 140

-- Theorem statement
theorem number_of_attendees_choosing_water 
  (h1 : juice_percent = 0.7) 
  (h2 : water_percent = 0.3) 
  (h3 : attendees_juice total_attendees = 140) : 
  attendees_water total_attendees = 60 :=
sorry

end number_of_attendees_choosing_water_l40_40892


namespace container_capacity_l40_40474

theorem container_capacity (C : ℝ) 
  (h1 : (0.30 * C : ℝ) + 27 = 0.75 * C) : C = 60 :=
sorry

end container_capacity_l40_40474


namespace number_divisible_by_20p_l40_40335

noncomputable def floor_expr (p : ℕ) : ℤ :=
  Int.floor ((2 + Real.sqrt 5) ^ p - 2 ^ (p + 1))

theorem number_divisible_by_20p (p : ℕ) (hp : Nat.Prime p ∧ p % 2 = 1) :
  ∃ k : ℤ, floor_expr p = k * 20 * p :=
by
  sorry

end number_divisible_by_20p_l40_40335


namespace compare_neg_third_and_neg_point_three_l40_40104

/-- Compare two numbers -1/3 and -0.3 -/
theorem compare_neg_third_and_neg_point_three : (-1 / 3 : ℝ) < -0.3 :=
sorry

end compare_neg_third_and_neg_point_three_l40_40104


namespace train_length_is_correct_l40_40153

variable (speed_km_hr : Float) (time_sec : Float)

def speed_m_s (speed_km_hr : Float) : Float := speed_km_hr * (1000 / 3600)

def length_of_train (speed_km_hr : Float) (time_sec : Float) : Float :=
  speed_m_s speed_km_hr * time_sec

theorem train_length_is_correct :
  length_of_train 60 12 = 200.04 := 
sorry

end train_length_is_correct_l40_40153


namespace find_tax_percentage_l40_40555

-- Definitions based on given conditions
def income_total : ℝ := 58000
def income_threshold : ℝ := 40000
def tax_above_threshold_percentage : ℝ := 0.2
def total_tax : ℝ := 8000

-- Let P be the percentage taxed on the first $40,000
variable (P : ℝ)

-- Formulate the problem as a proof goal
theorem find_tax_percentage (h : total_tax = 8000) :
  P = ((total_tax - (tax_above_threshold_percentage * (income_total - income_threshold))) / income_threshold) * 100 :=
by sorry

end find_tax_percentage_l40_40555


namespace deepak_present_age_l40_40718

-- Let R be Rahul's current age and D be Deepak's current age
variables (R D : ℕ)

-- Given conditions
def ratio_condition : Prop := (4 : ℚ) / 3 = (R : ℚ) / D
def rahul_future_age_condition : Prop := R + 6 = 50

-- Prove Deepak's present age D is 33 years
theorem deepak_present_age : ratio_condition R D ∧ rahul_future_age_condition R → D = 33 := 
sorry

end deepak_present_age_l40_40718


namespace largest_angle_is_75_l40_40002

-- Let the measures of the angles be represented as 3x, 4x, and 5x for some value x
variable (x : ℝ)

-- Define the angles based on the given ratio
def angle1 := 3 * x
def angle2 := 4 * x
def angle3 := 5 * x

-- The sum of the angles in a triangle is 180 degrees
axiom angle_sum : angle1 + angle2 + angle3 = 180

-- Prove that the largest angle is 75 degrees
theorem largest_angle_is_75 : 5 * (180 / 12) = 75 :=
by
  -- Proof is not required as per the instructions
  sorry

end largest_angle_is_75_l40_40002


namespace coordinates_of_P_l40_40677

-- Define the conditions and the question as a Lean theorem
theorem coordinates_of_P (m : ℝ) (P : ℝ × ℝ) (h1 : P = (m + 3, m + 1)) (h2 : P.2 = 0) :
  P = (2, 0) := 
sorry

end coordinates_of_P_l40_40677


namespace temperature_difference_l40_40027

variable (highest_temp : ℤ)
variable (lowest_temp : ℤ)

theorem temperature_difference : 
  highest_temp = 2 ∧ lowest_temp = -8 → (highest_temp - lowest_temp = 10) := by
  sorry

end temperature_difference_l40_40027


namespace new_boarders_joined_l40_40721

theorem new_boarders_joined (initial_boarders new_boarders initial_day_students total_boarders total_day_students: ℕ)
  (h1: initial_boarders = 60)
  (h2: initial_day_students = 150)
  (h3: total_boarders = initial_boarders + new_boarders)
  (h4: total_day_students = initial_day_students)
  (h5: 2 * initial_day_students = 5 * initial_boarders)
  (h6: 2 * total_boarders = total_day_students) :
  new_boarders = 15 :=
by
  sorry

end new_boarders_joined_l40_40721


namespace pure_imaginary_iff_real_part_zero_l40_40281

theorem pure_imaginary_iff_real_part_zero (a b : ℝ) : (∃ z : ℂ, z = a + bi ∧ z.im ≠ 0) ↔ (a = 0 ∧ b ≠ 0) :=
sorry

end pure_imaginary_iff_real_part_zero_l40_40281


namespace supplement_twice_angle_l40_40918

theorem supplement_twice_angle (α : ℝ) (h : 180 - α = 2 * α) : α = 60 := by
  admit -- This is a placeholder for the actual proof

end supplement_twice_angle_l40_40918


namespace total_tosses_correct_l40_40345

def num_heads : Nat := 3
def num_tails : Nat := 7
def total_tosses : Nat := num_heads + num_tails

theorem total_tosses_correct : total_tosses = 10 := by
  sorry

end total_tosses_correct_l40_40345


namespace train_cable_car_distance_and_speeds_l40_40681
-- Import necessary libraries

-- Defining the variables and conditions
variables (s v1 v2 : ℝ)
variables (half_hour_sym_dist additional_distance quarter_hour_meet : ℝ)

-- Defining the conditions
def conditions :=
  (half_hour_sym_dist = v1 * (1 / 2) + v2 * (1 / 2)) ∧
  (additional_distance = 2 / v2) ∧
  (quarter_hour_meet = 1 / 4) ∧
  (v1 + v2 = 2 * s) ∧
  (v2 * (additional_distance + half_hour_sym_dist) = (v1 * (additional_distance + half_hour_sym_dist) - s)) ∧
  ((v1 + v2) * (half_hour_sym_dist + additional_distance + quarter_hour_meet) = 2 * s)

-- Proving the statement
theorem train_cable_car_distance_and_speeds
  (h : conditions s v1 v2 half_hour_sym_dist additional_distance quarter_hour_meet) :
  s = 24 ∧ v1 = 40 ∧ v2 = 8 := sorry

end train_cable_car_distance_and_speeds_l40_40681


namespace gervais_km_correct_henri_km_correct_madeleine_km_correct_total_km_correct_henri_drove_farthest_l40_40419

def gervais_distance_miles_per_day : Real := 315
def gervais_days : Real := 3
def gervais_km_per_mile : Real := 1.60934

def henri_total_miles : Real := 1250
def madeleine_distance_miles_per_day : Real := 100
def madeleine_days : Real := 5

def gervais_total_km := gervais_distance_miles_per_day * gervais_days * gervais_km_per_mile
def henri_total_km := henri_total_miles * gervais_km_per_mile
def madeleine_total_km := madeleine_distance_miles_per_day * madeleine_days * gervais_km_per_mile

def combined_total_km := gervais_total_km + henri_total_km + madeleine_total_km

theorem gervais_km_correct : gervais_total_km = 1520.82405 := sorry
theorem henri_km_correct : henri_total_km = 2011.675 := sorry
theorem madeleine_km_correct : madeleine_total_km = 804.67 := sorry
theorem total_km_correct : combined_total_km = 4337.16905 := sorry
theorem henri_drove_farthest : henri_total_km = 2011.675 := sorry

end gervais_km_correct_henri_km_correct_madeleine_km_correct_total_km_correct_henri_drove_farthest_l40_40419


namespace solve_for_x_l40_40336

theorem solve_for_x (x : ℝ) (h : (3 / 4) - (1 / 2) = 1 / x) : x = 4 :=
sorry

end solve_for_x_l40_40336


namespace find_a1_for_geometric_sequence_l40_40804

noncomputable def geometric_sequence := ℕ → ℝ

def is_geometric_sequence (a : geometric_sequence) : Prop :=
  ∃ q, ∀ n, a (n + 1) = a n * q

theorem find_a1_for_geometric_sequence (a : geometric_sequence)
  (h_geom : is_geometric_sequence a)
  (h1 : a 2 * a 5 = 2 * a 3)
  (h2 : (a 4 + a 6) / 2 = 5 / 4) :
  a 1 = 16 ∨ a 1 = -16 :=
sorry

end find_a1_for_geometric_sequence_l40_40804


namespace find_b_squared_l40_40646

theorem find_b_squared :
  let ellipse_eq := ∀ x y : ℝ, x^2 / 25 + y^2 / b^2 = 1
  let hyperbola_eq := ∀ x y : ℝ, x^2 / 225 - y^2 / 144 = 1 / 36
  let coinciding_foci := 
    let c_ellipse := Real.sqrt (25 - b^2)
    let c_hyperbola := Real.sqrt ((225 / 36) + (144 / 36))
    c_ellipse = c_hyperbola
  ellipse_eq ∧ hyperbola_eq ∧ coinciding_foci → b^2 = 14.75
:= by sorry

end find_b_squared_l40_40646


namespace cosine_sine_difference_identity_l40_40615

theorem cosine_sine_difference_identity :
  (Real.cos (75 * Real.pi / 180) * Real.cos (15 * Real.pi / 180)
  - Real.sin (255 * Real.pi / 180) * Real.sin (165 * Real.pi / 180)) = 1 / 2 := by
  -- Proof goes here
  sorry

end cosine_sine_difference_identity_l40_40615


namespace determine_ab_l40_40871

noncomputable def f (a b : ℕ) (x : ℝ) : ℝ := x^2 + 2 * a * x + b * 2^x

theorem determine_ab (a b : ℕ) (h : ∀ x : ℝ, f a b x = 0 ↔ f a b (f a b x) = 0) :
  (a, b) = (0, 0) ∨ (a, b) = (1, 0) :=
by
  sorry

end determine_ab_l40_40871


namespace emily_small_gardens_l40_40043

theorem emily_small_gardens (total_seeds planted_big_garden seeds_per_small_garden : ℕ) 
  (h1 : total_seeds = 41) 
  (h2 : planted_big_garden = 29) 
  (h3 : seeds_per_small_garden = 4) : 
  (total_seeds - planted_big_garden) / seeds_per_small_garden = 3 := 
by
  sorry

end emily_small_gardens_l40_40043


namespace intersections_line_segment_l40_40529

def intersects_count (a b : ℕ) (x y : ℕ) : ℕ :=
  let steps := gcd x y
  2 * (steps + 1)

theorem intersections_line_segment (x y : ℕ) (h_x : x = 501) (h_y : y = 201) :
  intersects_count 1 1 x y = 336 := by
  sorry

end intersections_line_segment_l40_40529


namespace sin_pi_minus_a_l40_40386

theorem sin_pi_minus_a (a : ℝ) (h_cos_a : Real.cos a = Real.sqrt 5 / 3) (h_range_a : a ∈ Set.Ioo (-Real.pi / 2) 0) : 
  Real.sin (Real.pi - a) = -2 / 3 :=
by sorry

end sin_pi_minus_a_l40_40386


namespace find_number_l40_40000

theorem find_number (x : ℤ) (h : x + 2 - 3 = 7) : x = 8 :=
sorry

end find_number_l40_40000


namespace xy_inequality_l40_40601

theorem xy_inequality (x y θ : ℝ) 
    (h : (x * Real.cos θ + y * Real.sin θ)^2 + x * Real.sin θ - y * Real.cos θ = 1) : 
    x^2 + y^2 ≥ 3/4 :=
sorry

end xy_inequality_l40_40601


namespace lemmings_distance_average_l40_40232

noncomputable def diagonal_length (side: ℝ) : ℝ :=
  Real.sqrt (side^2 + side^2)

noncomputable def fraction_traveled (side: ℝ) (distance: ℝ) : ℝ :=
  distance / (Real.sqrt 2 * side)

noncomputable def final_coordinates (side: ℝ) (distance1: ℝ) (angle: ℝ) (distance2: ℝ) : (ℝ × ℝ) :=
  let frac := fraction_traveled side distance1
  let initial_pos := (frac * side, frac * side)
  let move_dist := distance2 * (Real.sqrt 2 / 2)
  (initial_pos.1 + move_dist, initial_pos.2 + move_dist)

noncomputable def average_shortest_distances (side: ℝ) (coords: ℝ × ℝ) : ℝ :=
  let x_dist := min coords.1 (side - coords.1)
  let y_dist := min coords.2 (side - coords.2)
  (x_dist + (side - x_dist) + y_dist + (side - y_dist)) / 4

theorem lemmings_distance_average :
  let side := 15
  let distance1 := 9.3
  let angle := 45 / 180 * Real.pi -- convert to radians
  let distance2 := 3
  let coords := final_coordinates side distance1 angle distance2
  average_shortest_distances side coords = 7.5 :=
by
  sorry

end lemmings_distance_average_l40_40232


namespace solve_eq_l40_40314

theorem solve_eq (x : ℝ) (h : 2 - 1 / (2 - x) = 1 / (2 - x)) : x = 1 := 
sorry

end solve_eq_l40_40314


namespace whisky_replacement_l40_40015

variable (V : ℝ) (x : ℝ)

theorem whisky_replacement (h_condition : 0.40 * V - 0.40 * x + 0.19 * x = 0.26 * V) : 
  x = (2 / 3) * V := 
sorry

end whisky_replacement_l40_40015


namespace min_eccentricity_sum_l40_40707

def circle_O1 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 16
def circle_O2 (x y r : ℝ) : Prop := x^2 + y^2 = r^2 ∧ 0 < r ∧ r < 2

def moving_circle_tangent (e1 e2 : ℝ) (r : ℝ) : Prop :=
  e1 = 2 / (4 - r) ∧ e2 = 2 / (4 + r)

theorem min_eccentricity_sum : ∃ (e1 e2 : ℝ) (r : ℝ), 
  circle_O1 x y ∧ circle_O2 x y r ∧ moving_circle_tangent e1 e2 r ∧
    e1 > e2 ∧ (e1 + 2 * e2) = (3 + 2 * Real.sqrt 2) / 4 :=
sorry

end min_eccentricity_sum_l40_40707


namespace probability_win_more_than_5000_l40_40973

def boxes : Finset ℕ := {5, 500, 5000}
def keys : Finset (Finset ℕ) := { {5}, {500}, {5000} }

noncomputable def probability_correct_key (box : ℕ) : ℚ :=
  if box = 5000 then 1 / 3 else if box = 500 then 1 / 2 else 1

theorem probability_win_more_than_5000 :
    (probability_correct_key 5000) * (probability_correct_key 500) = 1 / 6 :=
by
  -- Proof is omitted
  sorry

end probability_win_more_than_5000_l40_40973


namespace no_rational_xyz_satisfies_l40_40135

theorem no_rational_xyz_satisfies:
  ¬ ∃ (x y z : ℚ), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ 
  (1 / (x - y) ^ 2 + 1 / (y - z) ^ 2 + 1 / (z - x) ^ 2 = 2014) :=
by
  -- The proof will go here
  sorry

end no_rational_xyz_satisfies_l40_40135


namespace function_domain_l40_40159

noncomputable def sqrt_domain : Set ℝ :=
  {x | x + 1 ≥ 0 ∧ 2 - x > 0 ∧ 2 - x ≠ 1}

theorem function_domain :
  sqrt_domain = {x | -1 ≤ x ∧ x < 1} ∪ {x | 1 < x ∧ x < 2} :=
by
  sorry

end function_domain_l40_40159


namespace area_of_quadrilateral_ABDE_l40_40217

-- Definitions for the given problem
variable (AB CE AC DE : ℝ)
variable (parABCE parACDE : Prop)
variable (areaCOD : ℝ)

-- Lean 4 statement for the proof problem
theorem area_of_quadrilateral_ABDE
  (h1 : parABCE)
  (h2 : parACDE)
  (h3 : AB = 5)
  (h4 : AC = 5)
  (h5 : CE = 10)
  (h6 : DE = 10)
  (h7 : areaCOD = 10)
  : (AB + AC + CE + DE) / 2 + areaCOD = 52.5 := 
sorry

end area_of_quadrilateral_ABDE_l40_40217


namespace find_k_l40_40911

theorem find_k (a b : ℤ × ℤ) (k : ℤ) 
  (h₁ : a = (2, 1)) 
  (h₂ : a.1 + b.1 = 1 ∧ a.2 + b.2 = k)
  (h₃ : a.1 * b.1 + a.2 * b.2 = 0) : k = 3 :=
sorry

end find_k_l40_40911


namespace students_not_yes_for_either_subject_l40_40578

variable (total_students yes_m no_m unsure_m yes_r no_r unsure_r yes_only_m : ℕ)

theorem students_not_yes_for_either_subject :
  total_students = 800 →
  yes_m = 500 →
  no_m = 200 →
  unsure_m = 100 →
  yes_r = 400 →
  no_r = 100 →
  unsure_r = 300 →
  yes_only_m = 150 →
  ∃ students_not_yes, students_not_yes = total_students - (yes_only_m + (yes_m - yes_only_m) + (yes_r - (yes_m - yes_only_m))) ∧ students_not_yes = 400 :=
by
  intros ht yt1 nnm um ypr ynr ur yom
  sorry

end students_not_yes_for_either_subject_l40_40578


namespace red_candies_l40_40322

theorem red_candies (R Y B : ℕ) 
  (h1 : Y = 3 * R - 20)
  (h2 : B = Y / 2)
  (h3 : R + B = 90) :
  R = 40 :=
by
  sorry

end red_candies_l40_40322


namespace solution_l40_40697

-- Define M and N according to the given conditions
def M : Set ℝ := {x | x < 0 ∨ x > 2}
def N : Set ℝ := {x | x ≥ 1}

-- Define the complement of M in Real numbers
def complementM : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Define the union of the complement of M and N
def problem_statement : Set ℝ := complementM ∪ N

-- State the theorem
theorem solution :
  problem_statement = { x | x ≥ 0 } :=
by
  sorry

end solution_l40_40697


namespace truck_sand_amount_l40_40246

theorem truck_sand_amount (initial_sand loss_sand final_sand : ℝ) (h1 : initial_sand = 4.1) (h2 : loss_sand = 2.4) :
  initial_sand - loss_sand = final_sand ↔ final_sand = 1.7 := 
by
  sorry

end truck_sand_amount_l40_40246


namespace curved_surface_area_cone_l40_40465

theorem curved_surface_area_cone :
  let r := 8  -- base radius in cm
  let l := 19  -- lateral edge length in cm
  let π := Real.pi
  let CSA := π * r * l
  477.5 < CSA ∧ CSA < 478 := by
  sorry

end curved_surface_area_cone_l40_40465


namespace principal_amount_invested_l40_40184

noncomputable def calculate_principal : ℕ := sorry

theorem principal_amount_invested (P : ℝ) (y : ℝ) 
    (h1 : 300 = P * y * 2 / 100) -- Condition for simple interest
    (h2 : 307.50 = P * ((1 + y/100)^2 - 1)) -- Condition for compound interest
    : P = 73.53 := 
sorry

end principal_amount_invested_l40_40184


namespace common_difference_arithmetic_sequence_l40_40780

theorem common_difference_arithmetic_sequence
  (a : ℕ) (d : ℚ) (n : ℕ) (a_n : ℕ) (S_n : ℕ)
  (h1 : a = 2)
  (h2 : a_n = 20)
  (h3 : S_n = 132)
  (h4 : a_n = a + (n - 1) * d)
  (h5 : S_n = n * (a + a_n) / 2) :
  d = 18 / 11 := sorry

end common_difference_arithmetic_sequence_l40_40780


namespace magnitude_of_T_l40_40787

open Complex

noncomputable def i : ℂ := Complex.I

noncomputable def T : ℂ := (1 + i)^19 - (1 - i)^19

theorem magnitude_of_T : Complex.abs T = 1024 := by
  sorry

end magnitude_of_T_l40_40787


namespace customers_in_other_countries_l40_40180

-- Define the given conditions

def total_customers : ℕ := 7422
def customers_us : ℕ := 723

theorem customers_in_other_countries : total_customers - customers_us = 6699 :=
by
  -- This part will contain the proof, which is not required for this task.
  sorry

end customers_in_other_countries_l40_40180


namespace reduce_entanglement_l40_40796

/- 
Define a graph structure and required operations as per the given conditions. 
-/
structure Graph (V : Type) :=
  (E : V -> V -> Prop)

def remove_odd_degree_verts (G : Graph V) : Graph V :=
  sorry -- Placeholder for graph reduction logic

def duplicate_graph (G : Graph V) : Graph V :=
  sorry -- Placeholder for graph duplication logic

/--
  Prove that any graph where each vertex can be part of multiple entanglements 
  can be reduced to a state where no two vertices are connected using the given operations.
-/
theorem reduce_entanglement (G : Graph V) : ∃ G', 
  G' = remove_odd_degree_verts (duplicate_graph G) ∧
  (∀ (v1 v2 : V), ¬ G'.E v1 v2) :=
  by
  sorry

end reduce_entanglement_l40_40796


namespace ferris_wheel_time_10_seconds_l40_40491

noncomputable def time_to_reach_height (R : ℝ) (T : ℝ) (h : ℝ) : ℝ :=
  let ω := 2 * Real.pi / T
  let t := (Real.arcsin (h / R - 1)) / ω
  t

theorem ferris_wheel_time_10_seconds :
  time_to_reach_height 30 120 15 = 10 :=
by
  sorry

end ferris_wheel_time_10_seconds_l40_40491


namespace sum_smallest_largest_consecutive_even_integers_l40_40146

theorem sum_smallest_largest_consecutive_even_integers
  (n : ℕ) (a y : ℤ) 
  (hn_even : Even n) 
  (h_mean : y = (a + (a + 2 * (n - 1))) / 2) :
  2 * y = (a + (a + 2 * (n - 1))) :=
by
  sorry

end sum_smallest_largest_consecutive_even_integers_l40_40146


namespace cone_height_l40_40019

theorem cone_height 
  (sector_radius : ℝ) 
  (central_angle : ℝ) 
  (sector_radius_eq : sector_radius = 3) 
  (central_angle_eq : central_angle = 2 * π / 3) : 
  ∃ h : ℝ, h = 2 * Real.sqrt 2 :=
by
  -- Formalize conditions
  let r := 1
  let l := sector_radius
  let θ := central_angle

  -- Combine conditions
  have r_eq : r = 1 := by sorry

  -- Calculate height using Pythagorean theorem
  let h := (l^2 - r^2).sqrt

  use h
  have h_eq : h = 2 * Real.sqrt 2 := by sorry
  exact h_eq

end cone_height_l40_40019


namespace part1_part2_l40_40253

noncomputable def f (x a : ℝ) := 5 - |x + a| - |x - 2|

theorem part1 : 
  (∀ x, f x 1 ≥ 0 ↔ -2 ≤ x ∧ x ≤ 3) :=
sorry

theorem part2 :
  (∀ a, (∀ x, f x a ≤ 1) ↔ (a ≤ -6 ∨ a ≥ 2)) :=
sorry

end part1_part2_l40_40253


namespace otimes_calculation_l40_40815

def otimes (x y : ℝ) : ℝ := x^2 - 2*y

theorem otimes_calculation (k : ℝ) : otimes k (otimes k k) = -k^2 + 4*k :=
by
  sorry

end otimes_calculation_l40_40815


namespace find_value_l40_40758

theorem find_value (a b : ℝ) (h : a^2 + b^2 - 2 * a + 6 * b + 10 = 0) : 2 * a^100 - 3 * b⁻¹ = 3 :=
by sorry

end find_value_l40_40758


namespace water_added_l40_40679

theorem water_added (initial_fullness : ℚ) (final_fullness : ℚ) (capacity : ℚ)
  (h1 : initial_fullness = 0.40) (h2 : final_fullness = 3 / 4) (h3 : capacity = 80) :
  (final_fullness * capacity - initial_fullness * capacity) = 28 := by
  sorry

end water_added_l40_40679


namespace solve_arctan_equation_l40_40759

noncomputable def f (x : ℝ) : ℝ :=
  Real.arctan (1 / x) + Real.arctan (1 / (x^3))

theorem solve_arctan_equation (x : ℝ) (hx : x = (1 + Real.sqrt 5) / 2) :
  f x = Real.pi / 4 :=
by
  rw [hx]
  sorry

end solve_arctan_equation_l40_40759


namespace least_number_of_tiles_l40_40670

-- Definitions for classroom dimensions
def classroom_length : ℕ := 624 -- in cm
def classroom_width : ℕ := 432 -- in cm

-- Definitions for tile dimensions
def rectangular_tile_length : ℕ := 60
def rectangular_tile_width : ℕ := 80
def triangular_tile_base : ℕ := 40
def triangular_tile_height : ℕ := 40

-- Definition for the area calculation
def area (length width : ℕ) : ℕ := length * width
def area_triangular_tile (base height : ℕ) : ℕ := (base * height) / 2

-- Define the area of the classroom and tiles
def classroom_area : ℕ := area classroom_length classroom_width
def rectangular_tile_area : ℕ := area rectangular_tile_length rectangular_tile_width
def triangular_tile_area : ℕ := area_triangular_tile triangular_tile_base triangular_tile_height

-- Define the number of tiles required
def number_of_rectangular_tiles : ℕ := (classroom_area + rectangular_tile_area - 1) / rectangular_tile_area -- ceiling division in lean
def number_of_triangular_tiles : ℕ := (classroom_area + triangular_tile_area - 1) / triangular_tile_area -- ceiling division in lean

-- Define the minimum number of tiles required
def minimum_number_of_tiles : ℕ := min number_of_rectangular_tiles number_of_triangular_tiles

-- The main theorem establishing the least number of tiles required
theorem least_number_of_tiles : minimum_number_of_tiles = 57 := by
    sorry

end least_number_of_tiles_l40_40670


namespace hectors_sibling_product_l40_40539

theorem hectors_sibling_product (sisters : Nat) (brothers : Nat) (helen : Nat -> Prop): 
  (helen 4) → (helen 7) → (helen 5) → (helen 6) →
  (sisters + 1 = 5) → (brothers + 1 = 7) → ((sisters * brothers) = 30) :=
by
  sorry

end hectors_sibling_product_l40_40539


namespace find_fraction_identity_l40_40245

variable (x y z : ℝ)

theorem find_fraction_identity
 (h1 : 16 * y^2 = 15 * x * z)
 (h2 : y = 2 * x * z / (x + z)) :
 x / z + z / x = 34 / 15 := by
-- proof skipped
sorry

end find_fraction_identity_l40_40245


namespace trapezoid_DC_length_l40_40532

theorem trapezoid_DC_length 
  (AB DC: ℝ) (BC: ℝ) 
  (angle_BCD angle_CDA: ℝ)
  (h1: AB = 8)
  (h2: BC = 4 * Real.sqrt 3)
  (h3: angle_BCD = 60)
  (h4: angle_CDA = 45)
  (h5: AB = DC):
  DC = 14 + 4 * Real.sqrt 2 :=
sorry

end trapezoid_DC_length_l40_40532


namespace distance_from_origin_to_point_l40_40178

def point : ℝ × ℝ := (12, -16)
def origin : ℝ × ℝ := (0, 0)

theorem distance_from_origin_to_point : 
  let (x1, y1) := origin
  let (x2, y2) := point 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 20 :=
by
  sorry

end distance_from_origin_to_point_l40_40178


namespace om_4_2_eq_18_l40_40393

def om (a b : ℝ) : ℝ := 2 * a + 5 * b

theorem om_4_2_eq_18 : om 4 2 = 18 :=
by
  sorry

end om_4_2_eq_18_l40_40393


namespace cost_of_each_muffin_l40_40207

-- Define the cost of juice
def juice_cost : ℝ := 1.45

-- Define the total cost paid by Kevin
def total_cost : ℝ := 3.70

-- Assume the cost of each muffin
def muffin_cost (M : ℝ) : Prop := 
  3 * M + juice_cost = total_cost

-- The theorem we aim to prove
theorem cost_of_each_muffin : muffin_cost 0.75 :=
by
  -- Here the proof would go
  sorry

end cost_of_each_muffin_l40_40207


namespace find_number_with_10_questions_l40_40039

theorem find_number_with_10_questions (n : ℕ) (h : n ≤ 1000) : n = 300 :=
by
  sorry

end find_number_with_10_questions_l40_40039


namespace max_bottles_drunk_l40_40120

theorem max_bottles_drunk (e b : ℕ) (h1 : e = 16) (h2 : b = 4) : 
  ∃ n : ℕ, n = 5 :=
by
  sorry

end max_bottles_drunk_l40_40120


namespace area_of_sector_one_radian_l40_40139

theorem area_of_sector_one_radian (r θ : ℝ) (hθ : θ = 1) (hr : r = 1) : 
  (1/2 * (r * θ) * r) = 1/2 :=
by
  sorry

end area_of_sector_one_radian_l40_40139


namespace alyssa_bought_224_new_cards_l40_40830

theorem alyssa_bought_224_new_cards
  (initial_cards : ℕ)
  (after_purchase_cards : ℕ)
  (h1 : initial_cards = 676)
  (h2 : after_purchase_cards = 900) :
  after_purchase_cards - initial_cards = 224 :=
by
  -- Placeholder to avoid proof since it's explicitly not required 
  sorry

end alyssa_bought_224_new_cards_l40_40830


namespace inequality_proof_l40_40518

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d):
    1/a + 1/b + 4/c + 16/d ≥ 64/(a + b + c + d) :=
by
  sorry

end inequality_proof_l40_40518


namespace present_age_of_son_l40_40260

variable (S M : ℕ)

theorem present_age_of_son :
  (M = S + 30) ∧ (M + 2 = 2 * (S + 2)) → S = 28 :=
by
  sorry

end present_age_of_son_l40_40260


namespace fraction_addition_l40_40231

theorem fraction_addition (a b : ℕ) (hb : b ≠ 0) (h : a / (b : ℚ) = 3 / 5) : (a + b) / (b : ℚ) = 8 / 5 := 
by
sorry

end fraction_addition_l40_40231


namespace Shara_will_owe_money_l40_40915

theorem Shara_will_owe_money
    (B : ℕ)
    (h1 : 6 * 10 = 60)
    (h2 : B / 2 = 60)
    (h3 : 4 * 10 = 40)
    (h4 : 60 + 40 = 100) :
  B - 100 = 20 :=
sorry

end Shara_will_owe_money_l40_40915


namespace proof_problem_l40_40118

theorem proof_problem (a b c d x : ℝ)
  (h1 : c = 6 * d)
  (h2 : 2 * a = 1 / (-b))
  (h3 : abs x = 9) :
  (2 * a * b - 6 * d + c - x / 3 = -4) ∨ (2 * a * b - 6 * d + c - x / 3 = 2) :=
by
  sorry

end proof_problem_l40_40118


namespace find_integer_values_of_a_l40_40953

theorem find_integer_values_of_a
  (x a b c : ℤ)
  (h : (x - a) * (x - 10) + 5 = (x + b) * (x + c)) :
  a = 4 ∨ a = 16 := by
    sorry

end find_integer_values_of_a_l40_40953


namespace tanA_over_tanB_l40_40230

noncomputable def tan_ratios (A B C a b c : ℝ) : Prop :=
  a * Real.cos B - b * Real.cos A + 2 * c = 0

theorem tanA_over_tanB {A B C a b c : ℝ} (h : tan_ratios A B C a b c) : 
  Real.tan A / Real.tan B = -1 / 3 :=
by
  sorry

end tanA_over_tanB_l40_40230


namespace question1_question2_application_l40_40317

theorem question1: (-4)^2 - (-3) * (-5) = 1 := by
  sorry

theorem question2 (a : ℝ) (h : a = -4) : a^2 - (a + 1) * (a - 1) = 1 := by
  sorry

theorem application (a : ℝ) (h : a = 1.35) : a * (a - 1) * 2 * a - a^3 - a * (a - 1)^2 = -1.35 := by
  sorry

end question1_question2_application_l40_40317


namespace largest_x_fraction_l40_40682

theorem largest_x_fraction (x : ℝ) (h : (⌊x⌋ : ℝ) / x = 11 / 12) : x ≤ 120 / 11 := by
  sorry

end largest_x_fraction_l40_40682


namespace original_price_calc_l40_40096

theorem original_price_calc (h : 1.08 * x = 2) : x = 100 / 54 := by
  sorry

end original_price_calc_l40_40096


namespace minValue_is_9_minValue_achieves_9_l40_40867

noncomputable def minValue (x y : ℝ) : ℝ :=
  (x^2 + 1/(y^2)) * (1/(x^2) + 4 * y^2)

theorem minValue_is_9 (x y : ℝ) (hx : x > 0) (hy : y > 0) : minValue x y ≥ 9 :=
  sorry

theorem minValue_achieves_9 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = 1/2) : minValue x y = 9 :=
  sorry

end minValue_is_9_minValue_achieves_9_l40_40867


namespace problem_a_problem_b_l40_40313
-- Import the entire math library to ensure all necessary functionality is included

-- Define the problem context
variables {x y z : ℝ}

-- State the conditions as definitions
def conditions (x y z : ℝ) : Prop :=
  (x ≤ y) ∧ (y ≤ z) ∧ (x + y + z = 12) ∧ (x^2 + y^2 + z^2 = 54)

-- State the formal proof problems
theorem problem_a (h : conditions x y z) : x ≤ 3 ∧ 5 ≤ z :=
sorry

theorem problem_b (h : conditions x y z) : 
  9 ≤ x * y ∧ x * y ≤ 25 ∧
  9 ≤ y * z ∧ y * z ≤ 25 ∧
  9 ≤ z * x ∧ z * x ≤ 25 :=
sorry

end problem_a_problem_b_l40_40313


namespace complement_set_example_l40_40910

open Set

variable (U M : Set ℕ)

def complement (U M : Set ℕ) := U \ M

theorem complement_set_example :
  (U = {1, 2, 3, 4, 5, 6}) → 
  (M = {1, 3, 5}) → 
  (complement U M = {2, 4, 6}) := by
  intros hU hM
  rw [complement, hU, hM]
  sorry

end complement_set_example_l40_40910


namespace min_top_block_sum_l40_40066

theorem min_top_block_sum : 
  ∀ (assign_numbers : ℕ → ℕ) 
  (layer_1 : Fin 16 → ℕ) (layer_2 : Fin 9 → ℕ) (layer_3 : Fin 4 → ℕ) (top_block : ℕ),
  (∀ i, layer_3 i = layer_2 (i / 2) + layer_2 ((i / 2) + 1) + layer_2 ((i / 2) + 3) + layer_2 ((i / 2) + 4)) →
  (∀ i, layer_2 i = layer_1 (i / 2) + layer_1 ((i / 2) + 1) + layer_1 ((i / 2) + 3) + layer_1 ((i / 2) + 4)) →
  (top_block = layer_3 0 + layer_3 1 + layer_3 2 + layer_3 3) →
  top_block = 40 :=
sorry

end min_top_block_sum_l40_40066


namespace total_wet_surface_area_eq_l40_40296

-- Definitions based on given conditions
def length_cistern : ℝ := 10
def width_cistern : ℝ := 6
def height_water : ℝ := 1.35

-- Problem statement: Prove the total wet surface area is as calculated
theorem total_wet_surface_area_eq :
  let area_bottom : ℝ := length_cistern * width_cistern
  let area_longer_sides : ℝ := 2 * (length_cistern * height_water)
  let area_shorter_sides : ℝ := 2 * (width_cistern * height_water)
  let total_wet_surface_area : ℝ := area_bottom + area_longer_sides + area_shorter_sides
  total_wet_surface_area = 103.2 :=
by
  -- Since we do not need the proof, we use sorry here
  sorry

end total_wet_surface_area_eq_l40_40296


namespace Robinson_age_l40_40774

theorem Robinson_age (R : ℕ)
    (brother : ℕ := R + 2)
    (sister : ℕ := R + 6)
    (mother : ℕ := R + 20)
    (avg_age_yesterday : ℕ := 39)
    (total_age_yesterday : ℕ := 156)
    (eq : (R - 1) + (brother - 1) + (sister - 1) + (mother - 1) = total_age_yesterday) :
  R = 33 :=
by
  sorry

end Robinson_age_l40_40774


namespace systematic_sampling_interval_l40_40046

def population_size : ℕ := 2000
def sample_size : ℕ := 50
def interval (N n : ℕ) : ℕ := N / n

theorem systematic_sampling_interval :
  interval population_size sample_size = 40 := by
  sorry

end systematic_sampling_interval_l40_40046


namespace pressure_increases_when_block_submerged_l40_40062

theorem pressure_increases_when_block_submerged 
  (P0 : ℝ) (ρ : ℝ) (g : ℝ) (h0 h1 : ℝ) :
  h1 > h0 → 
  (P0 + ρ * g * h1) > (P0 + ρ * g * h0) :=
by
  intros h1_gt_h0
  sorry

end pressure_increases_when_block_submerged_l40_40062


namespace convert_to_cylindrical_l40_40727

noncomputable def cylindricalCoordinates (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.arccos (x / r)
  if y / r < 0 then (r, 2 * Real.pi - θ, z) else (r, θ, z)

theorem convert_to_cylindrical :
  cylindricalCoordinates 3 (-3 * Real.sqrt 3) 4 = (6, 5 * Real.pi / 3, 4) :=
by
  sorry

end convert_to_cylindrical_l40_40727


namespace inequality_proof_l40_40016

-- Let x and y be real numbers such that x > y
variables {x y : ℝ} (hx : x > y)

-- We need to prove -2x < -2y
theorem inequality_proof (hx : x > y) : -2 * x < -2 * y :=
sorry

end inequality_proof_l40_40016


namespace maximize_GDP_investment_l40_40757

def invest_A_B_max_GDP : Prop :=
  ∃ (A B : ℝ), 
  A + B ≤ 30 ∧
  20000 * A + 40000 * B ≤ 1000000 ∧
  24 * A + 32 * B ≥ 800 ∧
  A = 20 ∧ B = 10

theorem maximize_GDP_investment : invest_A_B_max_GDP :=
by
  sorry

end maximize_GDP_investment_l40_40757


namespace prob_of_king_or_queen_top_l40_40528

/-- A standard deck comprises 52 cards, with 13 ranks and 4 suits, each rank having one card per suit. -/
def standard_deck : Set (String × String) :=
Set.prod { "Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King" }
          { "Hearts", "Diamonds", "Clubs", "Spades" }

/-- There are four cards of rank King and four of rank Queen in the standard deck. -/
def count_kings_and_queens : Nat := 
4 + 4

/-- The total number of cards in a standard deck is 52. -/
def total_cards : Nat := 52

/-- The probability that the top card is either a King or a Queen is 2/13. -/
theorem prob_of_king_or_queen_top :
  (count_kings_and_queens / total_cards : ℚ) = (2 / 13 : ℚ) :=
sorry

end prob_of_king_or_queen_top_l40_40528


namespace sum_of_possible_values_l40_40237

variable {S : ℝ} (h : S ≠ 0)

theorem sum_of_possible_values (h : S ≠ 0) : ∃ N : ℝ, N ≠ 0 ∧ 6 * N + 2 / N = S → ∀ N1 N2 : ℝ, (6 * N1 + 2 / N1 = S ∧ 6 * N2 + 2 / N2 = S) → (N1 + N2) = S / 6 :=
by
  sorry

end sum_of_possible_values_l40_40237


namespace four_P_plus_five_square_of_nat_l40_40085

theorem four_P_plus_five_square_of_nat 
  (a b : ℕ)
  (P : ℕ)
  (hP : P = (Nat.lcm a b) / (a + 1) + (Nat.lcm a b) / (b + 1))
  (h_prime : Nat.Prime P) : 
  ∃ n : ℕ, 4 * P + 5 = (2 * n + 1) ^ 2 :=
by
  sorry

end four_P_plus_five_square_of_nat_l40_40085


namespace heaviest_tv_l40_40863

theorem heaviest_tv :
  let area (width height : ℝ) := width * height
  let weight (area : ℝ) := area * 4
  let weight_in_pounds (weight : ℝ) := weight / 16
  let bill_area := area 48 100
  let bob_area := area 70 60
  let steve_area := area 84 92
  let bill_weight := weight bill_area
  let bob_weight := weight bob_area
  let steve_weight := weight steve_area
  let bill_weight_pounds := weight_in_pounds (weight bill_area)
  let bob_weight_pounds := weight_in_pounds (weight bob_area)
  let steve_weight_pounds := weight_in_pounds (weight steve_area)
  bob_weight_pounds + bill_weight_pounds < steve_weight_pounds
  ∧ abs ((steve_weight_pounds) - (bill_weight_pounds + bob_weight_pounds)) = 318 :=
by
  sorry

end heaviest_tv_l40_40863


namespace question1_question2_l40_40275

-- Condition: p
def p (x : ℝ) : Prop := 4 * x^2 + 12 * x - 7 ≤ 0

-- Condition: q
def q (a : ℝ) (x : ℝ) : Prop := a - 3 ≤ x ∧ x ≤ a + 3

-- Question 1 statement: Given p is true and q is false when a = 0, find range of x
theorem question1 (x : ℝ) (h : p x ∧ ¬q 0 x) : -7/2 ≤ x ∧ x < -3 :=
sorry

-- Question 2 statement: If p is a sufficient condition for q, find range of a
theorem question2 (a : ℝ) (h : ∀ x, p x → q a x) : -5/2 ≤ a ∧ a ≤ 1/2 :=
sorry

end question1_question2_l40_40275


namespace min_x_plus_y_l40_40092

theorem min_x_plus_y (x y : ℝ) (h1 : x * y = 2 * x + y + 2) (h2 : x > 1) :
  x + y ≥ 7 :=
sorry

end min_x_plus_y_l40_40092


namespace difference_of_percentages_l40_40255

variable (x y : ℝ)

theorem difference_of_percentages :
  (0.60 * (50 + x)) - (0.45 * (30 + y)) = 16.5 + 0.60 * x - 0.45 * y := 
sorry

end difference_of_percentages_l40_40255


namespace value_of_g_at_2_l40_40777

def g (x : ℝ) : ℝ := x^2 - 4

theorem value_of_g_at_2 : g 2 = 0 :=
by
  -- proof goes here
  sorry

end value_of_g_at_2_l40_40777


namespace binary_101011_is_43_l40_40919

def binary_to_decimal_conversion (b : Nat) : Nat := 
  match b with
  | 101011 => 43
  | _ => 0

theorem binary_101011_is_43 : binary_to_decimal_conversion 101011 = 43 := by
  sorry

end binary_101011_is_43_l40_40919


namespace problem_solution_l40_40990

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
if x < 0 then 3^x else m - x^2

def p (m : ℝ) : Prop :=
∃ x, f m x = 0

def q (m : ℝ) : Prop :=
m = 1 / 9 → f m (f m (-1)) = 0

theorem problem_solution :
  ¬ (∃ m, m < 0 ∧ p m) ∧ q (1 / 9) :=
by 
  sorry

end problem_solution_l40_40990


namespace cans_to_collect_l40_40409

theorem cans_to_collect
  (martha_cans : ℕ)
  (diego_half_plus_ten : ℕ)
  (total_cans_required : ℕ)
  (martha_cans_collected : martha_cans = 90)
  (diego_collected : diego_half_plus_ten = (martha_cans / 2) + 10)
  (goal_cans : total_cans_required = 150) :
  total_cans_required - (martha_cans + diego_half_plus_ten) = 5 :=
by
  sorry

end cans_to_collect_l40_40409


namespace average_distance_per_day_l40_40533

def distance_Monday : ℝ := 4.2
def distance_Tuesday : ℝ := 3.8
def distance_Wednesday : ℝ := 3.6
def distance_Thursday : ℝ := 4.4

def total_distance : ℝ := distance_Monday + distance_Tuesday + distance_Wednesday + distance_Thursday

def number_of_days : ℕ := 4

theorem average_distance_per_day : total_distance / number_of_days = 4 := by
  sorry

end average_distance_per_day_l40_40533


namespace gcd_polynomial_l40_40351

theorem gcd_polynomial (b : ℤ) (h : ∃ k : ℤ, b = 2 * 997 * k) : 
  Int.gcd (3 * b^2 + 34 * b + 102) (b + 21) = 21 := 
by
  -- Proof would go here, but is omitted as instructed
  sorry

end gcd_polynomial_l40_40351


namespace subset_singleton_natural_l40_40943

/-
  Problem Statement:
  Prove that the set {2} is a subset of the set of natural numbers.
-/

open Set

theorem subset_singleton_natural :
  {2} ⊆ (Set.univ : Set ℕ) :=
by
  sorry

end subset_singleton_natural_l40_40943


namespace unique_solution_of_diophantine_l40_40414

theorem unique_solution_of_diophantine (m n : ℕ) (hm_pos : m > 0) (hn_pos: n > 0) :
  m^2 = Int.sqrt n + Int.sqrt (2 * n + 1) → (m = 13 ∧ n = 4900) :=
by
  sorry

end unique_solution_of_diophantine_l40_40414


namespace range_of_a_l40_40808

-- Define the function g(x) = x^3 - 3ax - a
def g (a x : ℝ) : ℝ := x^3 - 3*a*x - a

-- Define the derivative of g(x) which is g'(x) = 3x^2 - 3a
def g' (a x : ℝ) : ℝ := 3*x^2 - 3*a

theorem range_of_a (a : ℝ) : g a 0 * g a 1 < 0 → 0 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l40_40808


namespace standard_deviation_distance_l40_40391

-- Definitions and assumptions based on the identified conditions
def mean : ℝ := 12
def std_dev : ℝ := 1.2
def value : ℝ := 9.6

-- Statement to prove
theorem standard_deviation_distance : (value - mean) / std_dev = -2 :=
by sorry

end standard_deviation_distance_l40_40391


namespace megawheel_seat_capacity_l40_40840

theorem megawheel_seat_capacity (seats people : ℕ) (h1 : seats = 15) (h2 : people = 75) : people / seats = 5 := by
  sorry

end megawheel_seat_capacity_l40_40840


namespace remainder_division_l40_40551

theorem remainder_division :
  ∃ N R1 Q2, N = 44 * 432 + R1 ∧ N = 30 * Q2 + 18 ∧ R1 < 44 ∧ 18 = R1 :=
by
  sorry

end remainder_division_l40_40551


namespace line_through_point_equal_intercepts_l40_40878

theorem line_through_point_equal_intercepts (a b : ℝ) : 
  ((∃ (k : ℝ), k ≠ 0 ∧ (3 = 2 * k) ∧ b = k) ∨ ((a ≠ 0) ∧ (5/a = 1))) → 
  (a = 1 ∧ b = 1) ∨ (3 * a - 2 * b = 0) := 
by 
  sorry

end line_through_point_equal_intercepts_l40_40878


namespace power_simplification_l40_40874

theorem power_simplification :
  (1 / ((-5) ^ 4) ^ 2) * (-5) ^ 9 = -5 :=
by 
  sorry

end power_simplification_l40_40874


namespace sqrt_t4_plus_t2_eq_abs_t_sqrt_t2_plus_1_l40_40952

theorem sqrt_t4_plus_t2_eq_abs_t_sqrt_t2_plus_1 (t : ℝ) : 
  Real.sqrt (t^4 + t^2) = |t| * Real.sqrt (t^2 + 1) :=
sorry

end sqrt_t4_plus_t2_eq_abs_t_sqrt_t2_plus_1_l40_40952


namespace worker_total_pay_l40_40987

def regular_rate : ℕ := 10
def total_surveys : ℕ := 50
def cellphone_surveys : ℕ := 35
def non_cellphone_surveys := total_surveys - cellphone_surveys
def higher_rate := regular_rate + (30 * regular_rate / 100)

def pay_non_cellphone_surveys := non_cellphone_surveys * regular_rate
def pay_cellphone_surveys := cellphone_surveys * higher_rate

def total_pay := pay_non_cellphone_surveys + pay_cellphone_surveys

theorem worker_total_pay : total_pay = 605 := by
  sorry

end worker_total_pay_l40_40987


namespace original_movie_length_l40_40205

theorem original_movie_length (final_length cut_scene original_length : ℕ) 
    (h1 : cut_scene = 3) (h2 : final_length = 57) (h3 : final_length + cut_scene = original_length) : 
  original_length = 60 := 
by 
  -- Proof omitted
  sorry

end original_movie_length_l40_40205


namespace hyperbola_center_is_equidistant_l40_40342

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem hyperbola_center_is_equidistant (F1 F2 C : ℝ × ℝ) 
  (hF1 : F1 = (3, -2)) 
  (hF2 : F2 = (11, 6))
  (hC : C = ((F1.1 + F2.1) / 2, (F1.2 + F2.2) / 2)) :
  C = (7, 2) ∧ distance C F1 = distance C F2 :=
by
  -- Fill in with the appropriate proofs
  sorry

end hyperbola_center_is_equidistant_l40_40342


namespace simplify_expression_l40_40870

theorem simplify_expression (x : ℝ) : 
  ( ( (x^(16/8))^(1/4) )^3 * ( (x^(16/4))^(1/8) )^5 ) = x^4 := 
by 
  sorry

end simplify_expression_l40_40870


namespace total_vessels_proof_l40_40315

def cruise_ships : Nat := 4
def cargo_ships : Nat := cruise_ships * 2
def sailboats : Nat := cargo_ships + 6
def fishing_boats : Nat := sailboats / 7
def total_vessels : Nat := cruise_ships + cargo_ships + sailboats + fishing_boats

theorem total_vessels_proof : total_vessels = 28 := by
  sorry

end total_vessels_proof_l40_40315


namespace ratio_of_wire_lengths_l40_40402

theorem ratio_of_wire_lengths 
  (bonnie_wire_length : ℕ := 80)
  (roark_wire_length : ℕ := 12000) :
  bonnie_wire_length / roark_wire_length = 1 / 150 :=
by
  sorry

end ratio_of_wire_lengths_l40_40402


namespace nesbitt_inequality_l40_40596

variable (a b c d : ℝ)

-- Assume a, b, c, d are positive real numbers
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom pos_d : 0 < d

theorem nesbitt_inequality :
  a / (b + c) + b / (c + d) + c / (d + a) + d / (a + b) ≥ 2 := by
  sorry

end nesbitt_inequality_l40_40596


namespace smallest_n_for_divisibility_l40_40070

noncomputable def geometric_sequence (a₁ r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n-1)

theorem smallest_n_for_divisibility (h₁ : ∀ n : ℕ, geometric_sequence (1/2 : ℚ) 60 n = (1/2 : ℚ) * 60^(n-1))
    (h₂ : (60 : ℚ) * (1 / 2) = 30)
    (n : ℕ) :
  (∃ n : ℕ, n ≥ 1 ∧ (geometric_sequence (1/2 : ℚ) 60 n) ≥ 10^6) ↔ n = 7 :=
by
  sorry

end smallest_n_for_divisibility_l40_40070


namespace original_proposition_converse_proposition_false_inverse_proposition_false_contrapositive_proposition_true_l40_40649

-- Definition of the quadrilateral being a rhombus
def is_rhombus (quad : Type) : Prop := 
-- A quadrilateral is a rhombus if and only if all its sides are equal in length
sorry

-- Definition of the diagonals of quadrilateral being perpendicular
def diagonals_are_perpendicular (quad : Type) : Prop := 
-- The diagonals of a quadrilateral are perpendicular
sorry

-- Original proposition: If a quadrilateral is a rhombus, then its diagonals are perpendicular to each other
theorem original_proposition (quad : Type) : is_rhombus quad → diagonals_are_perpendicular quad :=
sorry

-- Converse proposition: If the diagonals of a quadrilateral are perpendicular to each other, then it is a rhombus, which is False
theorem converse_proposition_false (quad : Type) : diagonals_are_perpendicular quad → ¬ is_rhombus quad :=
sorry

-- Inverse proposition: If a quadrilateral is not a rhombus, then its diagonals are not perpendicular, which is False
theorem inverse_proposition_false (quad : Type) : ¬ is_rhombus quad → ¬ diagonals_are_perpendicular quad :=
sorry

-- Contrapositive proposition: If the diagonals of a quadrilateral are not perpendicular, then it is not a rhombus, which is True
theorem contrapositive_proposition_true (quad : Type) : ¬ diagonals_are_perpendicular quad → ¬ is_rhombus quad :=
sorry

end original_proposition_converse_proposition_false_inverse_proposition_false_contrapositive_proposition_true_l40_40649


namespace sum_and_product_of_roots_l40_40064

theorem sum_and_product_of_roots (a b : ℝ) (h1 : a * a * a - 4 * a * a - a + 4 = 0)
  (h2 : b * b * b - 4 * b * b - b + 4 = 0) :
  a + b + a * b = -1 :=
sorry

end sum_and_product_of_roots_l40_40064


namespace total_conference_games_scheduled_l40_40619

-- Definitions of the conditions
def num_divisions : ℕ := 2
def teams_per_division : ℕ := 6
def intradivision_games_per_pair : ℕ := 3
def interdivision_games_per_pair : ℕ := 2

-- The statement to prove the total number of conference games
theorem total_conference_games_scheduled : 
  (num_divisions * (teams_per_division * (teams_per_division - 1) * intradivision_games_per_pair) / 2) 
  + (teams_per_division * teams_per_division * interdivision_games_per_pair) = 162 := 
by
  sorry

end total_conference_games_scheduled_l40_40619


namespace ratio_of_buttons_to_magnets_per_earring_l40_40084

-- Definitions related to the problem statement
def gemstones_per_button : ℕ := 3
def magnets_per_earring : ℕ := 2
def sets_of_earrings : ℕ := 4
def required_gemstones : ℕ := 24

-- Problem statement translation into Lean 4
theorem ratio_of_buttons_to_magnets_per_earring :
  (required_gemstones / gemstones_per_button / (sets_of_earrings * 2)) = 1 / 2 := by
  sorry

end ratio_of_buttons_to_magnets_per_earring_l40_40084


namespace min_absolute_difference_l40_40609

open Int

theorem min_absolute_difference (x y : ℤ) (hx : 0 < x) (hy : 0 < y) (h : x * y - 4 * x + 3 * y = 215) : |x - y| = 15 :=
sorry

end min_absolute_difference_l40_40609


namespace arithmetic_sequence_sum_l40_40163

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S_n : ℕ → ℝ) (d : ℝ)
  (h1 : ∀ n, a n = a 0 + n * d) 
  (h2 : ∀ n, S_n n = n * (a 0 + a n) / 2) 
  (h3 : 2 * a 6 = 5 + a 8) :
  S_n 9 = 45 := 
by 
  sorry

end arithmetic_sequence_sum_l40_40163


namespace EH_length_l40_40520

structure Rectangle :=
(AB BC CD DA : ℝ)
(horiz: AB=CD)
(verti: BC=DA)
(diag_eq: (AB^2 + BC^2) = (CD^2 + DA^2))

structure Point :=
(x y : ℝ)

noncomputable def H_distance (E D : Point)
    (AB BC : ℝ) : ℝ :=
    (E.y - D.y) -- if we consider D at origin (0,0)

theorem EH_length
    (AB BC : ℝ)
    (H_dist : ℝ)
    (E : Point)
    (rectangle : Rectangle) :
    AB = 50 →
    BC = 60 →
    E.x^2 + BC^2 = 30^2 + 60^2 →
    E.y = 40 →
    H_dist = E.y - CD →
    H_dist = 7.08 :=
by
    sorry

end EH_length_l40_40520


namespace remainder_divided_by_82_l40_40310

theorem remainder_divided_by_82 (x : ℤ) : 
  (∃ k : ℤ, x = 82 * k + 5) ↔ (∃ m : ℤ, x + 13 = 41 * m + 18) :=
by
  sorry

end remainder_divided_by_82_l40_40310


namespace prime_base_values_l40_40348

theorem prime_base_values :
  ∀ p : ℕ, Prime p →
    (2 * p^3 + p^2 + 6 + 4 * p^2 + p + 4 + 2 * p^2 + p + 5 + 2 * p^2 + 2 * p + 2 + 9 =
     4 * p^2 + 3 * p + 3 + 5 * p^2 + 7 * p + 2 + 3 * p^2 + 2 * p + 1) →
    false :=
by {
  sorry
}

end prime_base_values_l40_40348


namespace smallest_number_of_rectangles_l40_40597

-- Defining the given problem conditions
def rectangle_area : ℕ := 3 * 4
def smallest_square_side_length : ℕ := 12

-- Lean 4 statement to prove the problem
theorem smallest_number_of_rectangles 
    (h : ∃ n : ℕ, n * n = smallest_square_side_length * smallest_square_side_length)
    (h1 : ∃ m : ℕ, m * rectangle_area = smallest_square_side_length * smallest_square_side_length) :
    m = 9 :=
by
  sorry

end smallest_number_of_rectangles_l40_40597


namespace total_cookies_sold_l40_40388

/-- Clara's cookie sales -/
def numCookies (type1_box : Nat) (type1_cookies_per_box : Nat)
               (type2_box : Nat) (type2_cookies_per_box : Nat)
               (type3_box : Nat) (type3_cookies_per_box : Nat) : Nat :=
  (type1_box * type1_cookies_per_box) +
  (type2_box * type2_cookies_per_box) +
  (type3_box * type3_cookies_per_box)

theorem total_cookies_sold :
  numCookies 50 12 80 20 70 16 = 3320 := by
  sorry

end total_cookies_sold_l40_40388


namespace solve_arithmetic_sequence_l40_40659

theorem solve_arithmetic_sequence (x : ℝ) 
  (term1 term2 term3 : ℝ)
  (h1 : term1 = 3 / 4)
  (h2 : term2 = 2 * x - 3)
  (h3 : term3 = 7 * x) 
  (h_arith : term2 - term1 = term3 - term2) :
  x = -9 / 4 :=
by
  sorry

end solve_arithmetic_sequence_l40_40659


namespace no_hexagon_cross_section_l40_40075

-- Define the shape of the cross-section resulting from cutting a triangular prism with a plane
inductive Shape
| triangle
| quadrilateral
| pentagon
| hexagon

-- Define the condition of cutting a triangular prism
structure TriangularPrism where
  cut : Shape

-- The theorem stating that cutting a triangular prism with a plane cannot result in a hexagon
theorem no_hexagon_cross_section (P : TriangularPrism) : P.cut ≠ Shape.hexagon :=
by
  sorry

end no_hexagon_cross_section_l40_40075


namespace solve_for_x_l40_40220

theorem solve_for_x (x : ℝ) (h : 3 * x - 4 * x + 7 * x = 120) : x = 20 := 
by
  sorry

end solve_for_x_l40_40220


namespace find_n_l40_40137

theorem find_n (n : ℚ) (h : (1 / (n + 2) + 3 / (n + 2) + n / (n + 2) = 4)) : n = -4 / 3 :=
by
  sorry

end find_n_l40_40137


namespace probability_shaded_region_l40_40154

def triangle_game :=
  let total_regions := 6
  let shaded_regions := 3
  shaded_regions / total_regions

theorem probability_shaded_region:
  triangle_game = 1 / 2 := by
  sorry

end probability_shaded_region_l40_40154


namespace max_regions_divided_by_lines_l40_40032

theorem max_regions_divided_by_lines (m n : ℕ) (hm : m ≠ 0) (hn : n ≠ 0) :
  ∃ r : ℕ, r = m * n + 2 * m + 2 * n - 1 :=
by
  sorry

end max_regions_divided_by_lines_l40_40032


namespace find_x_l40_40427

namespace ProofProblem

def δ (x : ℚ) : ℚ := 5 * x + 6
def φ (x : ℚ) : ℚ := 9 * x + 4

theorem find_x (x : ℚ) : (δ (φ x) = 14) ↔ (x = -4 / 15) :=
by
  sorry

end ProofProblem

end find_x_l40_40427


namespace max_expr_under_condition_l40_40584

-- Define the conditions and variables
variable {x : ℝ}

-- State the theorem about the maximum value of the given expression under the given condition
theorem max_expr_under_condition (h : x < -3) : 
  ∃ M, M = -2 * Real.sqrt 2 - 3 ∧ ∀ y, y < -3 → y + 2 / (y + 3) ≤ M :=
sorry

end max_expr_under_condition_l40_40584


namespace solve_for_a_l40_40971

theorem solve_for_a (a : ℝ) (h : a / 0.3 = 0.6) : a = 0.18 :=
by sorry

end solve_for_a_l40_40971


namespace elena_butter_l40_40288

theorem elena_butter (cups_flour butter : ℕ) (h1 : cups_flour * 4 = 28) (h2 : butter * 4 = 12) : butter = 3 := 
by
  sorry

end elena_butter_l40_40288


namespace find_number_l40_40204

noncomputable def least_common_multiple (a b : ℕ) : ℕ := Nat.lcm a b

theorem find_number (n : ℕ) (h1 : least_common_multiple (least_common_multiple n 16) (least_common_multiple 18 24) = 144) : n = 9 :=
sorry

end find_number_l40_40204


namespace find_m_l40_40803

theorem find_m (S : ℕ → ℕ) (a : ℕ → ℕ) (m : ℕ) :
  (∀ n, S n = (n * (3 * n - 1)) / 2) →
  (a 1 = 1) →
  (∀ n ≥ 2, a n = S n - S (n - 1)) →
  (a m = 3 * m - 2) →
  (a 4 * a 4 = a 1 * a m) →
  m = 34 :=
by
  intro hS h1 ha1 ha2 hgeom
  sorry

end find_m_l40_40803


namespace exinscribed_sphere_inequality_l40_40572

variable (r r_A r_B r_C r_D : ℝ)

theorem exinscribed_sphere_inequality 
  (hr : 0 < r) 
  (hrA : 0 < r_A) 
  (hrB : 0 < r_B) 
  (hrC : 0 < r_C) 
  (hrD : 0 < r_D) :
  1 / Real.sqrt (r_A^2 - r_A * r_B + r_B^2) +
  1 / Real.sqrt (r_B^2 - r_B * r_C + r_C^2) +
  1 / Real.sqrt (r_C^2 - r_C * r_D + r_D^2) +
  1 / Real.sqrt (r_D^2 - r_D * r_A + r_A^2) ≤
  2 / r := by
  sorry

end exinscribed_sphere_inequality_l40_40572


namespace dorothy_age_relation_l40_40858

theorem dorothy_age_relation (D S : ℕ) (h1: S = 5) (h2: D + 5 = 2 * (S + 5)) : D = 3 * S :=
by
  -- implement the proof here
  sorry

end dorothy_age_relation_l40_40858


namespace range_of_m_l40_40770

theorem range_of_m {m : ℝ} : 
  (¬ ∃ x : ℝ, (1 / 2 ≤ x ∧ x ≤ 2 ∧ x^2 - 2 * x - m ≤ 0)) → m < -1 :=
by
  sorry

end range_of_m_l40_40770


namespace system_of_equations_solution_exists_l40_40160

theorem system_of_equations_solution_exists :
  ∃ (x y z : ℤ), 
    (x + y - 2018 = (y - 2019) * x) ∧
    (y + z - 2017 = (y - 2019) * z) ∧
    (x + z + 5 = x * z) ∧
    (x = 3 ∧ y = 2021 ∧ z = 4 ∨ 
    x = -1 ∧ y = 2019 ∧ z = -2) := 
sorry

end system_of_equations_solution_exists_l40_40160


namespace necessary_condition_real_roots_l40_40354

theorem necessary_condition_real_roots (a : ℝ) :
  (a >= 1 ∨ a <= -2) → (∃ x : ℝ, x^2 - a * x + 1 = 0) :=
by
  sorry

end necessary_condition_real_roots_l40_40354


namespace part1_part2_l40_40845

def f (x : ℝ) : ℝ := abs (x - 2)

theorem part1 (x : ℝ) : f x > 4 - abs (x + 1) ↔ x < -3 / 2 ∨ x > 5 / 2 := 
sorry

theorem part2 (a b : ℝ) (ha : 0 < a ∧ a < 1/2) (hb : 0 < b ∧ b < 1/2)
  (h : f (1 / a) + f (2 / b) = 10) : a + b / 2 ≥ 2 / 7 := 
sorry

end part1_part2_l40_40845


namespace determine_n_eq_1_l40_40286

theorem determine_n_eq_1 :
  ∃ n : ℝ, (∀ x : ℝ, (x = 2 → (x^3 - 3*x^2 + n = 2*x^3 - 6*x^2 + 5*n))) → n = 1 :=
by
  sorry

end determine_n_eq_1_l40_40286


namespace segments_do_not_intersect_l40_40366

noncomputable def check_intersection (AP PB BQ QC CR RD DS SA : ℚ) : Bool :=
  (AP / PB) * (BQ / QC) * (CR / RD) * (DS / SA) = 1

theorem segments_do_not_intersect :
  let AP := (3 : ℚ)
  let PB := (6 : ℚ)
  let BQ := (2 : ℚ)
  let QC := (4 : ℚ)
  let CR := (1 : ℚ)
  let RD := (5 : ℚ)
  let DS := (4 : ℚ)
  let SA := (6 : ℚ)
  ¬ check_intersection AP PB BQ QC CR RD DS SA :=
by sorry

end segments_do_not_intersect_l40_40366


namespace find_percentage_l40_40356

theorem find_percentage : 
  ∀ (P : ℕ), 
  (50 - 47 = (P / 100) * 15) →
  P = 20 := 
by
  intro P h
  sorry

end find_percentage_l40_40356


namespace mean_score_is_76_l40_40771

noncomputable def mean_stddev_problem := 
  ∃ (M SD : ℝ), (M - 2 * SD = 60) ∧ (M + 3 * SD = 100) ∧ (M = 76)

theorem mean_score_is_76 : mean_stddev_problem :=
sorry

end mean_score_is_76_l40_40771


namespace probability_prime_sum_is_1_9_l40_40242

def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_prime_sum (a b : ℕ) : Bool :=
  Nat.Prime (a + b)

def valid_prime_pairs : List (ℕ × ℕ) :=
  [(2, 3), (2, 5), (2, 11), (2, 17), (2, 29)]

def total_pairs : ℕ :=
  45

def valid_pairs_count : ℕ :=
  valid_prime_pairs.length

def probability_prime_sum : ℚ :=
  valid_pairs_count / total_pairs

theorem probability_prime_sum_is_1_9 :
  probability_prime_sum = 1 / 9 := 
by
  sorry

end probability_prime_sum_is_1_9_l40_40242


namespace dhoni_savings_percent_l40_40305

variable (E : ℝ) -- Assuming E is Dhoni's last month's earnings

-- Condition 1: Dhoni spent 25% of his earnings on rent
def spent_on_rent (E : ℝ) : ℝ := 0.25 * E

-- Condition 2: Dhoni spent 10% less than what he spent on rent on a new dishwasher
def spent_on_dishwasher (E : ℝ) : ℝ := 0.225 * E

-- Prove the percentage of last month's earnings Dhoni had left over
theorem dhoni_savings_percent (E : ℝ) : 
    52.5 / 100 * E = E - (spent_on_rent E + spent_on_dishwasher E) :=
by
  sorry

end dhoni_savings_percent_l40_40305


namespace fixed_point_l40_40773

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 1) + 2

theorem fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a (-1) = 3 :=
by
  unfold f
  sorry

end fixed_point_l40_40773


namespace find_t_l40_40115

noncomputable def f (x t k : ℝ): ℝ := (1/3) * x^3 - (t/2) * x^2 + k * x

theorem find_t (a b t k : ℝ) (h1 : t > 0) (h2 : k > 0) 
  (h3 : a + b = t) (h4 : a * b = k)
  (h5 : 2 * a = b - 2)
  (h6 : (-2) ^ 2 = a * b) : 
  t = 5 :=
by 
  sorry

end find_t_l40_40115


namespace temperature_range_l40_40078

-- Define the highest and lowest temperature conditions
variable (t : ℝ)
def highest_temp := t ≤ 30
def lowest_temp := 20 ≤ t

-- The theorem to prove the range of temperature change
theorem temperature_range (t : ℝ) (h_high : highest_temp t) (h_low : lowest_temp t) : 20 ≤ t ∧ t ≤ 30 :=
by 
  -- Insert the proof or leave as sorry for now
  sorry

end temperature_range_l40_40078


namespace cost_of_dinner_l40_40997

theorem cost_of_dinner (x : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (total_cost : ℝ) : 
  tax_rate = 0.09 → tip_rate = 0.18 → total_cost = 36.90 → 
  1.27 * x = 36.90 → x = 29 :=
by
  intros htr htt htc heq
  rw [←heq] at htc
  sorry

end cost_of_dinner_l40_40997


namespace vlad_taller_than_sister_l40_40719

def height_vlad_meters : ℝ := 1.905
def height_sister_cm : ℝ := 86.36

theorem vlad_taller_than_sister :
  (height_vlad_meters * 100 - height_sister_cm = 104.14) :=
by 
  sorry

end vlad_taller_than_sister_l40_40719


namespace inequality_proof_l40_40589

theorem inequality_proof (a b c : ℝ) (h1 : b < c) (h2 : 1 < a) (h3 : a < b + c) (h4 : b + c < a + 1) : b < a :=
by
  sorry

end inequality_proof_l40_40589


namespace sequence_sum_l40_40060

theorem sequence_sum :
  1 - 4 + 7 - 10 + 13 - 16 + 19 - 22 + 25 - 28 + 31 - 34 + 37 - 40 + 43 - 46 + 49 - 52 + 55 = 28 :=
by
  sorry

end sequence_sum_l40_40060


namespace find_rate_percent_l40_40240

-- Definitions based on the given conditions
def principal : ℕ := 800
def time : ℕ := 4
def simple_interest : ℕ := 192
def si_formula (P R T : ℕ) : ℕ := P * R * T / 100

-- Statement: prove that the rate percent (R) is 6%
theorem find_rate_percent (R : ℕ) (h : simple_interest = si_formula principal R time) : R = 6 :=
sorry

end find_rate_percent_l40_40240


namespace number_of_cupcakes_l40_40320

theorem number_of_cupcakes (total gluten_free vegan gluten_free_vegan non_vegan : ℕ) 
    (h1 : gluten_free = total / 2)
    (h2 : vegan = 24)
    (h3 : gluten_free_vegan = vegan / 2)
    (h4 : non_vegan = 28)
    (h5 : gluten_free_vegan = gluten_free / 2) :
    total = 52 :=
by
  sorry

end number_of_cupcakes_l40_40320


namespace pizza_order_l40_40112

theorem pizza_order (couple_want: ℕ) (child_want: ℕ) (num_couples: ℕ) (num_children: ℕ) (slices_per_pizza: ℕ)
  (hcouple: couple_want = 3) (hchild: child_want = 1) (hnumc: num_couples = 1) (hnumch: num_children = 6) (hsp: slices_per_pizza = 4) :
  (couple_want * 2 * num_couples + child_want * num_children) / slices_per_pizza = 3 := 
by
  -- Proof here
  sorry

end pizza_order_l40_40112


namespace add_fractions_l40_40319

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = (5 / a) :=
by sorry

end add_fractions_l40_40319


namespace total_distance_hopped_l40_40832

def distance_hopped (rate: ℕ) (time: ℕ) : ℕ := rate * time

def spotted_rabbit_distance (time: ℕ) : ℕ :=
  let pattern := [8, 11, 16, 20, 9]
  let full_cycles := time / pattern.length
  let remaining_minutes := time % pattern.length
  let full_cycle_distance := full_cycles * pattern.sum
  let remaining_distance := (List.take remaining_minutes pattern).sum
  full_cycle_distance + remaining_distance

theorem total_distance_hopped :
  distance_hopped 15 12 + distance_hopped 12 12 + distance_hopped 18 12 + distance_hopped 10 12 + spotted_rabbit_distance 12 = 807 :=
by
  sorry

end total_distance_hopped_l40_40832


namespace total_votes_l40_40326

/-- Let V be the total number of votes. Define the votes received by the candidate and rival. -/
def votes_cast (V : ℕ) (votes_candidate : ℕ) (votes_rival : ℕ) : Prop :=
  votes_candidate = 40 * V / 100 ∧ votes_rival = votes_candidate + 2000 ∧ votes_candidate + votes_rival = V

/-- Prove that the total number of votes is 10000 given the conditions. -/
theorem total_votes (V : ℕ) (votes_candidate : ℕ) (votes_rival : ℕ) :
  votes_cast V votes_candidate votes_rival → V = 10000 :=
by
  sorry

end total_votes_l40_40326


namespace largest_even_number_l40_40395

theorem largest_even_number (x : ℕ) (h : x + (x+2) + (x+4) = 1194) : x + 4 = 400 :=
by
  have : 3*x + 6 = 1194 := by linarith
  have : 3*x = 1188 := by linarith
  have : x = 396 := by linarith
  linarith

end largest_even_number_l40_40395


namespace problem_value_l40_40352

theorem problem_value (x : ℤ) (h : x = -2023) : 
  abs (abs (abs x - x) - abs x) - x = 4046 :=
by
  sorry

end problem_value_l40_40352


namespace original_price_is_correct_l40_40276

-- Given conditions as Lean definitions
def reduced_price : ℝ := 2468
def reduction_amount : ℝ := 161.46

-- To find the original price including the sales tax
def original_price_including_tax (P : ℝ) : Prop :=
  P - reduction_amount = reduced_price

-- The proof statement to show the price is 2629.46
theorem original_price_is_correct : original_price_including_tax 2629.46 :=
by
  sorry

end original_price_is_correct_l40_40276


namespace solve_for_a_l40_40422

theorem solve_for_a (a : Real) (h_pos : a > 0) (h_eq : (fun x => x^2 + 4) ((fun x => x^2 - 2) a) = 18) : 
  a = Real.sqrt (Real.sqrt 14 + 2) := by 
  sorry

end solve_for_a_l40_40422


namespace apples_fraction_of_pears_l40_40290

variables (A O P : ℕ)

-- Conditions
def oranges_condition := O = 3 * A
def pears_condition := P = 4 * O

-- Statement we need to prove
theorem apples_fraction_of_pears (A O P : ℕ) (h1 : O = 3 * A) (h2 : P = 4 * O) : (A : ℚ) / P = 1 / 12 :=
by
  sorry

end apples_fraction_of_pears_l40_40290


namespace saline_drip_duration_l40_40150

theorem saline_drip_duration (rate_drops_per_minute : ℕ) (drop_to_ml_rate : ℕ → ℕ → Prop)
  (ml_received : ℕ) (time_hours : ℕ) :
  rate_drops_per_minute = 20 ->
  drop_to_ml_rate 100 5 ->
  ml_received = 120 ->
  time_hours = 2 :=
by {
  sorry
}

end saline_drip_duration_l40_40150


namespace shirt_original_price_l40_40372

theorem shirt_original_price (original_price final_price : ℝ) (h1 : final_price = 0.5625 * original_price) 
  (h2 : final_price = 19) : original_price = 33.78 :=
by
  sorry

end shirt_original_price_l40_40372


namespace combined_total_score_is_correct_l40_40921

-- Definitions of point values
def touchdown_points := 6
def extra_point_points := 1
def field_goal_points := 3

-- Hawks' Scores
def hawks_touchdowns := 4
def hawks_successful_extra_points := 2
def hawks_field_goals := 2

-- Eagles' Scores
def eagles_touchdowns := 3
def eagles_successful_extra_points := 3
def eagles_field_goals := 3

-- Calculations
def hawks_total_points := hawks_touchdowns * touchdown_points +
                          hawks_successful_extra_points * extra_point_points +
                          hawks_field_goals * field_goal_points

def eagles_total_points := eagles_touchdowns * touchdown_points +
                           eagles_successful_extra_points * extra_point_points +
                           eagles_field_goals * field_goal_points

def combined_total_score := hawks_total_points + eagles_total_points

-- The theorem that needs to be proved
theorem combined_total_score_is_correct : combined_total_score = 62 :=
by
  -- proof would go here
  sorry

end combined_total_score_is_correct_l40_40921


namespace hulk_jump_geometric_sequence_l40_40435

theorem hulk_jump_geometric_sequence (n : ℕ) (a_n : ℕ) : 
  (a_n = 3 * 2^(n - 1)) → (a_n > 3000) → n = 11 :=
by
  sorry

end hulk_jump_geometric_sequence_l40_40435


namespace harkamal_purchase_mangoes_l40_40514

variable (m : ℕ)

def cost_of_grapes (cost_per_kg grapes_weight : ℕ) : ℕ := cost_per_kg * grapes_weight
def cost_of_mangoes (cost_per_kg mangoes_weight : ℕ) : ℕ := cost_per_kg * mangoes_weight

theorem harkamal_purchase_mangoes :
  (cost_of_grapes 70 10 + cost_of_mangoes 55 m = 1195) → m = 9 :=
by
  sorry

end harkamal_purchase_mangoes_l40_40514


namespace find_second_speed_l40_40549

theorem find_second_speed (d t_b : ℝ) (v1 : ℝ) (t_m t_a : ℤ): 
  d = 13.5 ∧ v1 = 5 ∧ t_m = 12 ∧ t_a = 15 →
  (t_b = (d / v1) - (t_m / 60)) →
  (t2 = t_b - (t_a / 60)) →
  v = d / t2 →
  v = 6 :=
by
  sorry

end find_second_speed_l40_40549


namespace train_speed_in_km_per_hour_l40_40456

-- Definitions based on the conditions
def train_length : ℝ := 240  -- The length of the train in meters.
def time_to_pass_tree : ℝ := 8  -- The time to pass the tree in seconds.
def meters_per_second_to_kilometers_per_hour : ℝ := 3.6  -- Conversion factor from meters/second to kilometers/hour.

-- Statement based on the question and the correct answer
theorem train_speed_in_km_per_hour : (train_length / time_to_pass_tree) * meters_per_second_to_kilometers_per_hour = 108 :=
by
  sorry

end train_speed_in_km_per_hour_l40_40456


namespace probability_three_non_red_purple_balls_l40_40894

def total_balls : ℕ := 150
def prob_white : ℝ := 0.15
def prob_green : ℝ := 0.20
def prob_yellow : ℝ := 0.30
def prob_red : ℝ := 0.30
def prob_purple : ℝ := 0.05
def prob_not_red_purple : ℝ := 1 - (prob_red + prob_purple)

theorem probability_three_non_red_purple_balls :
  (prob_not_red_purple * prob_not_red_purple * prob_not_red_purple) = 0.274625 :=
by
  sorry

end probability_three_non_red_purple_balls_l40_40894


namespace students_like_both_l40_40741

theorem students_like_both {total students_apple_pie students_chocolate_cake students_none students_at_least_one students_both : ℕ} 
  (h_total : total = 50)
  (h_apple : students_apple_pie = 22)
  (h_chocolate : students_chocolate_cake = 20)
  (h_none : students_none = 17)
  (h_least_one : students_at_least_one = total - students_none)
  (h_union : students_at_least_one = students_apple_pie + students_chocolate_cake - students_both) :
  students_both = 9 :=
by
  sorry

end students_like_both_l40_40741


namespace problem_inequality_l40_40229

theorem problem_inequality (a b : ℝ) (hab : 1 / a + 1 / b = 1) : 
  ∀ n : ℕ, (a + b)^n - a^n - b^n ≥ 2^(2 * n) - 2^(n + 1) := 
by
  sorry

end problem_inequality_l40_40229


namespace complex_fraction_value_l40_40728

theorem complex_fraction_value (a b : ℝ) (h : (i - 2) / (1 + i) = a + b * i) : a + b = 1 :=
by
  sorry

end complex_fraction_value_l40_40728


namespace integral_of_2x_minus_1_over_x_sq_l40_40754

theorem integral_of_2x_minus_1_over_x_sq:
  ∫ x in (1 : ℝ)..3, (2 * x - (1 / x^2)) = 26 / 3 := by
  sorry

end integral_of_2x_minus_1_over_x_sq_l40_40754


namespace total_bill_amount_l40_40470

theorem total_bill_amount (n : ℕ) (cost_per_meal : ℕ) (gratuity_rate : ℚ) (total_bill_with_gratuity : ℚ)
  (h1 : n = 7) (h2 : cost_per_meal = 100) (h3 : gratuity_rate = 20 / 100) :
  total_bill_with_gratuity = (n * cost_per_meal : ℕ) * (1 + gratuity_rate) :=
sorry

end total_bill_amount_l40_40470


namespace files_deleted_l40_40753

theorem files_deleted 
  (orig_files : ℕ) (final_files : ℕ) (deleted_files : ℕ) 
  (h_orig : orig_files = 24) 
  (h_final : final_files = 21) : 
  deleted_files = orig_files - final_files :=
by
  rw [h_orig, h_final]
  sorry

end files_deleted_l40_40753


namespace remainder_783245_div_7_l40_40694

theorem remainder_783245_div_7 :
  783245 % 7 = 1 :=
sorry

end remainder_783245_div_7_l40_40694


namespace length_of_base_AD_l40_40701

-- Definitions based on the conditions
def isosceles_trapezoid (A B C D : Type) : Prop := sorry -- Implementation of an isosceles trapezoid
def length_of_lateral_side (A B C D : Type) : ℝ := 40 -- The lateral side is 40 cm
def angle_BAC (A B C D : Type) : ℝ := 45 -- The angle ∠BAC is 45 degrees
def bisector_O_center (O A B D M : Type) : Prop := sorry -- Implementation that O is the center of circumscribed circle and lies on bisector

-- Main theorem based on the derived problem statement
theorem length_of_base_AD (A B C D O M : Type) 
  (h_iso_trapezoid : isosceles_trapezoid A B C D)
  (h_length_lateral : length_of_lateral_side A B C D = 40)
  (h_angle_BAC : angle_BAC A B C D = 45)
  (h_O_center_bisector : bisector_O_center O A B D M)
  : ℝ :=
  20 * (Real.sqrt 6 + Real.sqrt 2)

end length_of_base_AD_l40_40701


namespace compute_difference_of_reciprocals_l40_40134

theorem compute_difference_of_reciprocals
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x - y = x / y) :
  (1 / x) - (1 / y) = - (1 / y^2) :=
by
  sorry

end compute_difference_of_reciprocals_l40_40134


namespace wall_area_160_l40_40643

noncomputable def wall_area (small_tile_area : ℝ) (fraction_small : ℝ) : ℝ :=
  small_tile_area / fraction_small

theorem wall_area_160 (small_tile_area : ℝ) (fraction_small : ℝ) (h1 : small_tile_area = 80) (h2 : fraction_small = 1 / 2) :
  wall_area small_tile_area fraction_small = 160 :=
by
  rw [wall_area, h1, h2]
  norm_num

end wall_area_160_l40_40643


namespace girls_attended_festival_l40_40357

variable (g b : ℕ)

theorem girls_attended_festival :
  g + b = 1500 ∧ (2 / 3) * g + (1 / 2) * b = 900 → (2 / 3) * g = 600 := by
  sorry

end girls_attended_festival_l40_40357


namespace john_hourly_wage_l40_40086

theorem john_hourly_wage (days_off: ℕ) (hours_per_day: ℕ) (weekly_wage: ℕ) 
  (days_off_eq: days_off = 3) (hours_per_day_eq: hours_per_day = 4) (weekly_wage_eq: weekly_wage = 160):
  (weekly_wage / ((7 - days_off) * hours_per_day) = 10) :=
by
  /-
  Given:
  days_off = 3
  hours_per_day = 4
  weekly_wage = 160

  To prove:
  weekly_wage / ((7 - days_off) * hours_per_day) = 10
  -/
  sorry

end john_hourly_wage_l40_40086


namespace max_water_bottles_one_athlete_l40_40736

-- Define variables and key conditions
variable (total_bottles : Nat := 40)
variable (total_athletes : Nat := 25)
variable (at_least_one : ∀ i, i < total_athletes → Nat.succ i ≥ 1)

-- Define the problem as a theorem
theorem max_water_bottles_one_athlete (h_distribution : total_bottles = 40) :
  ∃ max_bottles, max_bottles = 16 :=
by
  sorry

end max_water_bottles_one_athlete_l40_40736


namespace value_of_m_sub_n_l40_40651

theorem value_of_m_sub_n (m n : ℤ) (h1 : |m| = 5) (h2 : n^2 = 36) (h3 : m * n < 0) : m - n = 11 ∨ m - n = -11 := 
by 
  sorry

end value_of_m_sub_n_l40_40651


namespace maximum_a_value_condition_l40_40730

theorem maximum_a_value_condition (x a : ℝ) :
  (∀ x, (x^2 - 2 * x - 3 > 0 → x < a)) ↔ a ≤ -1 :=
by sorry

end maximum_a_value_condition_l40_40730


namespace pat_moved_chairs_l40_40432

theorem pat_moved_chairs (total_chairs : ℕ) (carey_moved : ℕ) (left_to_move : ℕ) (pat_moved : ℕ) :
  total_chairs = 74 →
  carey_moved = 28 →
  left_to_move = 17 →
  pat_moved = total_chairs - left_to_move - carey_moved →
  pat_moved = 29 :=
by
  intros h_total h_carey h_left h_equation
  rw [h_total, h_carey, h_left] at h_equation
  exact h_equation

end pat_moved_chairs_l40_40432


namespace contrapositive_example_l40_40323

theorem contrapositive_example :
  (∀ x : ℝ, x^2 < 4 → -2 < x ∧ x < 2) ↔ (∀ x : ℝ, (x ≥ 2 ∨ x ≤ -2) → x^2 ≥ 4) :=
by
  sorry

end contrapositive_example_l40_40323


namespace order_fractions_l40_40626

theorem order_fractions : (16/13 : ℚ) < 21/17 ∧ 21/17 < 20/15 :=
by {
  -- use cross-multiplication:
  -- 16*17 < 21*13 -> 272 < 273 -> true
  -- 16*15 < 20*13 -> 240 < 260 -> true
  -- 21*15 < 20*17 -> 315 < 340 -> true
  sorry
}

end order_fractions_l40_40626


namespace probability_A_mc_and_B_tf_probability_at_least_one_mc_l40_40028

-- Define the total number of questions
def total_questions : ℕ := 5

-- Define the number of multiple choice questions and true or false questions
def multiple_choice_questions : ℕ := 3
def true_false_questions : ℕ := 2

-- First proof problem: Probability that A draws a multiple-choice question and B draws a true or false question
theorem probability_A_mc_and_B_tf :
  (multiple_choice_questions * true_false_questions : ℚ) / (total_questions * (total_questions - 1)) = 3 / 10 :=
by
  sorry

-- Second proof problem: Probability that at least one of A and B draws a multiple-choice question
theorem probability_at_least_one_mc :
  1 - (true_false_questions * (true_false_questions - 1) : ℚ) / (total_questions * (total_questions - 1)) = 9 / 10 :=
by
  sorry

end probability_A_mc_and_B_tf_probability_at_least_one_mc_l40_40028


namespace allocation_ways_l40_40687

theorem allocation_ways (programs : Finset ℕ) (grades : Finset ℕ) (h_programs : programs.card = 6) (h_grades : grades.card = 4) : 
  ∃ ways : ℕ, ways = 1080 := 
by 
  sorry

end allocation_ways_l40_40687


namespace problem_statement_l40_40869

theorem problem_statement (n : ℕ) (h1 : 0 < n) (h2 : ∃ k : ℤ, (1/2 + 1/3 + 1/11 + 1/n : ℚ) = k) : ¬ (n > 66) := 
sorry

end problem_statement_l40_40869


namespace num_pigs_on_farm_l40_40709

variables (P : ℕ)
def cows := 2 * P - 3
def goats := (2 * P - 3) + 6
def total_animals := P + cows P + goats P

theorem num_pigs_on_farm (h : total_animals P = 50) : P = 10 :=
sorry

end num_pigs_on_farm_l40_40709


namespace store_price_reduction_l40_40667

theorem store_price_reduction 
    (initial_price : ℝ) (initial_sales : ℕ) (price_reduction : ℝ)
    (sales_increase_factor : ℝ) (target_profit : ℝ)
    (x : ℝ) : (initial_price, initial_price - price_reduction, x) = (80, 50, 12) →
    sales_increase_factor = 20 →
    target_profit = 7920 →
    (30 - x) * (200 + sales_increase_factor * x / 2) = 7920 →
    x = 12 ∧ (initial_price - x) = 68 :=
by 
    intros h₁ h₂ h₃ h₄
    sorry

end store_price_reduction_l40_40667


namespace simple_interest_true_discount_l40_40431

theorem simple_interest_true_discount (P R T : ℝ) 
  (h1 : 85 = (P * R * T) / 100)
  (h2 : 80 = (85 * P) / (P + 85)) : P = 1360 :=
sorry

end simple_interest_true_discount_l40_40431


namespace one_of_a_b_c_is_zero_l40_40717

theorem one_of_a_b_c_is_zero
  (a b c : ℝ)
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^9 + b^9) * (b^9 + c^9) * (c^9 + a^9) = (a * b * c)^9) :
  a = 0 ∨ b = 0 ∨ c = 0 :=
by
  sorry

end one_of_a_b_c_is_zero_l40_40717


namespace ratio_problem_l40_40334

-- Define the conditions and the required proof
theorem ratio_problem (p q n : ℝ) (h1 : p / q = 5 / n) (h2 : 2 * p + q = 14) : n = 1 :=
by
  sorry

end ratio_problem_l40_40334


namespace find_side_length_of_square_l40_40209

theorem find_side_length_of_square (n k : ℕ) (hk : k ≥ 1) (h : (n + k) * (n + k) - n * n = 47) : n = 23 :=
  sorry

end find_side_length_of_square_l40_40209


namespace new_bottles_from_recycling_l40_40595

theorem new_bottles_from_recycling (initial_bottles : ℕ) (required_bottles : ℕ) (h : initial_bottles = 125) (r : required_bottles = 5) : 
∃ new_bottles : ℕ, new_bottles = (initial_bottles / required_bottles ^ 2 + initial_bottles / (required_bottles * required_bottles / required_bottles) + initial_bottles / (required_bottles * required_bottles * required_bottles / required_bottles * required_bottles * required_bottles)) :=
  sorry

end new_bottles_from_recycling_l40_40595


namespace factorization_example_l40_40192

theorem factorization_example (C D : ℤ) (h : 20 * y^2 - 122 * y + 72 = (C * y - 8) * (D * y - 9)) : C * D + C = 25 := by
  sorry

end factorization_example_l40_40192


namespace function_decreasing_on_interval_l40_40460

noncomputable def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x - 3

theorem function_decreasing_on_interval : ∀ x₁ x₂ : ℝ, 1 ≤ x₁ → 1 ≤ x₂ → x₁ ≤ x₂ → f x₁ ≥ f x₂ := by
  sorry

end function_decreasing_on_interval_l40_40460


namespace evaluate_expression_l40_40981

theorem evaluate_expression :
  let a := 17
  let b := 19
  let c := 23
  let numerator1 := 136 * (1 / b - 1 / c) + 361 * (1 / c - 1 / a) + 529 * (1 / a - 1 / b)
  let denominator := a * (1 / b - 1 / c) + b * (1 / c - 1 / a) + c * (1 / a - 1 / b)
  let numerator2 := 144 * (1 / b - 1 / c) + 400 * (1 / c - 1 / a) + 576 * (1 / a - 1 / b)
  (numerator1 / denominator) * (numerator2 / denominator) = 3481 := by
  sorry

end evaluate_expression_l40_40981


namespace maximum_value_of_m_l40_40959

theorem maximum_value_of_m (x y : ℝ) (hx : x > 1 / 2) (hy : y > 1) : 
    (4 * x^2 / (y - 1) + y^2 / (2 * x - 1)) ≥ 8 := 
sorry

end maximum_value_of_m_l40_40959


namespace total_hooligans_l40_40842

def hooligans_problem (X Y : ℕ) : Prop :=
  (X * Y = 365) ∧ (X + Y = 78 ∨ X + Y = 366)

theorem total_hooligans (X Y : ℕ) (h : hooligans_problem X Y) : X + Y = 78 ∨ X + Y = 366 :=
  sorry

end total_hooligans_l40_40842


namespace carla_needs_30_leaves_l40_40716

-- Definitions of the conditions
def items_per_day : Nat := 5
def total_days : Nat := 10
def total_bugs : Nat := 20

-- Maths problem to be proved
theorem carla_needs_30_leaves :
  let total_items := items_per_day * total_days
  let required_leaves := total_items - total_bugs
  required_leaves = 30 :=
by
  sorry

end carla_needs_30_leaves_l40_40716


namespace triangular_weight_l40_40665

theorem triangular_weight (c t : ℝ) (h1 : c + t = 3 * c) (h2 : 4 * c + t = t + c + 90) : t = 60 := 
by sorry

end triangular_weight_l40_40665


namespace quadratic_real_roots_m_range_l40_40574

theorem quadratic_real_roots_m_range :
  ∀ (m : ℝ), (∃ x : ℝ, x^2 + 4*x + m + 5 = 0) ↔ m ≤ -1 :=
by sorry

end quadratic_real_roots_m_range_l40_40574


namespace expression_as_polynomial_l40_40148

theorem expression_as_polynomial (x : ℝ) :
  (3 * x^3 + 2 * x^2 + 5 * x + 9) * (x - 2) -
  (x - 2) * (2 * x^3 + 5 * x^2 - 74) +
  (4 * x - 17) * (x - 2) * (x + 4) = 
  x^4 + 2 * x^3 - 5 * x^2 + 9 * x - 30 :=
sorry

end expression_as_polynomial_l40_40148


namespace simplify_abs_expression_l40_40272

theorem simplify_abs_expression
  (a b : ℝ)
  (h1 : a < 0)
  (h2 : a * b < 0)
  : |a - b - 3| - |4 + b - a| = -1 := by
  sorry

end simplify_abs_expression_l40_40272


namespace sum_of_B_and_C_in_base_6_l40_40566

def digit_base_6 (n: Nat) : Prop :=
  n > 0 ∧ n < 6

theorem sum_of_B_and_C_in_base_6
  (A B C : Nat)
  (hA : digit_base_6 A)
  (hB : digit_base_6 B)
  (hC : digit_base_6 C)
  (hDistinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (hSum : 43 * (A + B + C) = 216 * A) :
  B + C = 5 := by
  sorry

end sum_of_B_and_C_in_base_6_l40_40566


namespace power_function_constant_l40_40706

theorem power_function_constant (k α : ℝ)
  (h : (1 / 2 : ℝ) ^ α * k = (Real.sqrt 2 / 2)) : k + α = 3 / 2 := by
  sorry

end power_function_constant_l40_40706


namespace field_area_l40_40550

theorem field_area (L W : ℝ) (h1: L = 20) (h2 : 2 * W + L = 41) : L * W = 210 :=
by
  sorry

end field_area_l40_40550


namespace product_xy_min_value_x_plus_y_min_value_attained_l40_40283

theorem product_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 8 / y = 1) : x * y = 64 := 
sorry

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 8 / y = 1) : 
  x + y = 18 := 
sorry

-- Additional theorem to prove that the minimum value is attained when x = 6 and y = 12
theorem min_value_attained (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 8 / y = 1) :
  x = 6 ∧ y = 12 := 
sorry

end product_xy_min_value_x_plus_y_min_value_attained_l40_40283


namespace least_common_multiple_135_195_l40_40145

def leastCommonMultiple (a b : ℕ) : ℕ :=
  Nat.lcm a b

theorem least_common_multiple_135_195 : leastCommonMultiple 135 195 = 1755 := by
  sorry

end least_common_multiple_135_195_l40_40145


namespace total_fare_for_100_miles_l40_40073

theorem total_fare_for_100_miles (b c : ℝ) (h₁ : 200 = b + 80 * c) : 240 = b + 100 * c :=
sorry

end total_fare_for_100_miles_l40_40073


namespace evaluate_expression_l40_40206

theorem evaluate_expression : 
  let e := 3 + 2 * Real.sqrt 3 + 1 / (3 + 2 * Real.sqrt 3) + 1 / (2 * Real.sqrt 3 - 3)
  e = 3 + 10 * Real.sqrt 3 / 3 :=
by
  let e := 3 + 2 * Real.sqrt 3 + 1 / (3 + 2 * Real.sqrt 3) + 1 / (2 * Real.sqrt 3 - 3)
  have h : e = 3 + 10 * Real.sqrt 3 / 3 := sorry
  exact h

end evaluate_expression_l40_40206


namespace find_number_l40_40604

theorem find_number (x : ℝ) (h : 0.95 * x - 12 = 178) : x = 200 :=
sorry

end find_number_l40_40604


namespace product_abc_l40_40984

theorem product_abc (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h : a * b^3 = 180) : a * b * c = 60 * c := 
sorry

end product_abc_l40_40984


namespace number_of_sides_is_15_l40_40065

variable {n : ℕ} -- n is the number of sides

-- Define the conditions
def sum_of_all_but_one_angle (n : ℕ) : Prop :=
  180 * (n - 2) - 2190 > 0 ∧ 180 * (n - 2) - 2190 < 180

-- State the theorem to be proven
theorem number_of_sides_is_15 (n : ℕ) (h : sum_of_all_but_one_angle n) : n = 15 :=
sorry

end number_of_sides_is_15_l40_40065


namespace pirates_share_l40_40970

def initial_coins (N : ℕ) := N ≥ 3000 ∧ N ≤ 4000

def first_pirate (N : ℕ) := N - (2 + (N - 2) / 4)
def second_pirate (remaining : ℕ) := remaining - (2 + (remaining - 2) / 4)
def third_pirate (remaining : ℕ) := remaining - (2 + (remaining - 2) / 4)
def fourth_pirate (remaining : ℕ) := remaining - (2 + (remaining - 2) / 4)

def final_remaining (N : ℕ) :=
  let step1 := first_pirate N
  let step2 := second_pirate step1
  let step3 := third_pirate step2
  let step4 := fourth_pirate step3
  step4

theorem pirates_share (N : ℕ) (h : initial_coins N) :
  final_remaining N / 4 = 660 :=
by
  sorry

end pirates_share_l40_40970


namespace unique_root_a_b_values_l40_40814

theorem unique_root_a_b_values {a b : ℝ} (h1 : ∀ x, x^2 + a * x + b = 0 ↔ x = 1) : a = -2 ∧ b = 1 := by
  sorry

end unique_root_a_b_values_l40_40814


namespace min_value_u_l40_40938

theorem min_value_u (x y : ℝ) (h₀ : x ≥ 0) (h₁ : y ≥ 0)
  (h₂ : 2 * x + y = 6) : 
  ∀u, u = 4 * x ^ 2 + 3 * x * y + y ^ 2 - 6 * x - 3 * y -> 
  u ≥ 27 / 2 := sorry

end min_value_u_l40_40938


namespace investor_wait_time_l40_40161

noncomputable def compound_interest_time (P A r : ℝ) (n : ℕ) : ℝ :=
  (Real.log (A / P)) / (n * Real.log (1 + r / n))

theorem investor_wait_time :
  compound_interest_time 600 661.5 0.10 2 = 1 := 
sorry

end investor_wait_time_l40_40161


namespace two_sectors_area_l40_40234

theorem two_sectors_area {r : ℝ} {θ : ℝ} (h_radius : r = 15) (h_angle : θ = 45) : 
  2 * (θ / 360) * (π * r^2) = 56.25 * π := 
by
  rw [h_radius, h_angle]
  norm_num
  sorry

end two_sectors_area_l40_40234


namespace work_rate_ab_together_l40_40411

-- Define A, B, and C as the work rates of individuals
variables (A B C : ℝ)

-- We are given the following conditions:
-- 1. a, b, and c together can finish the job in 11 days
-- 2. c alone can finish the job in 41.25 days

-- Given these conditions, we aim to prove that a and b together can finish the job in 15 days
theorem work_rate_ab_together
  (h1 : A + B + C = 1 / 11)
  (h2 : C = 1 / 41.25) :
  1 / (A + B) = 15 :=
by
  sorry

end work_rate_ab_together_l40_40411


namespace q_evaluation_l40_40271

def q (x y : ℤ) : ℤ :=
if x >= 0 ∧ y >= 0 then x - y
else if x < 0 ∧ y < 0 then x + 3 * y
else 2 * x + 2 * y

theorem q_evaluation : q (q 1 (-1)) (q (-2) (-3)) = -22 := by
sorry

end q_evaluation_l40_40271


namespace solve_xy_l40_40693

theorem solve_xy : ∃ (x y : ℝ), x = 1 / 3 ∧ y = 2 / 3 ∧ x^2 + (1 - y)^2 + (x - y)^2 = 1 / 3 :=
by
  use 1 / 3, 2 / 3
  sorry

end solve_xy_l40_40693


namespace complex_calculation_l40_40857

def complex_add (a b : ℂ) : ℂ := a + b
def complex_mul (a b : ℂ) : ℂ := a * b

theorem complex_calculation :
  let z1 := (⟨2, -3⟩ : ℂ)
  let z2 := (⟨4, 6⟩ : ℂ)
  let z3 := (⟨-1, 2⟩ : ℂ)
  complex_mul (complex_add z1 z2) z3 = (⟨-12, 9⟩ : ℂ) :=
by 
  sorry

end complex_calculation_l40_40857


namespace largest_number_l40_40337

theorem largest_number 
  (a b c : ℝ) (h1 : a = 0.8) (h2 : b = 1/2) (h3 : c = 0.9) (h4 : a ≤ 2) (h5 : b ≤ 2) (h6 : c ≤ 2) :
  max (max a b) c = 0.9 :=
by
  sorry

end largest_number_l40_40337


namespace value_of_expr_l40_40639

theorem value_of_expr : (365^2 - 349^2) / 16 = 714 := by
  sorry

end value_of_expr_l40_40639


namespace fg_eq_gf_condition_l40_40586

theorem fg_eq_gf_condition (m n p q : ℝ) (f g : ℝ → ℝ)
  (hf : ∀ x, f x = m * x + n) 
  (hg : ∀ x, g x = p * x + q) : 
  (∀ x, f (g x) = g (f x)) ↔ n * (1 - p) = q * (1 - m) := 
sorry

end fg_eq_gf_condition_l40_40586


namespace product_of_consecutive_multiples_of_4_divisible_by_192_l40_40340

theorem product_of_consecutive_multiples_of_4_divisible_by_192 :
  ∀ (n : ℤ), 192 ∣ (4 * n) * (4 * (n + 1)) * (4 * (n + 2)) :=
by
  intro n
  sorry

end product_of_consecutive_multiples_of_4_divisible_by_192_l40_40340


namespace adam_and_simon_distance_l40_40991

theorem adam_and_simon_distance :
  ∀ (t : ℝ), (10 * t)^2 + (12 * t)^2 = 16900 → t = 65 / Real.sqrt 61 :=
by
  sorry

end adam_and_simon_distance_l40_40991


namespace number_satisfying_condition_l40_40454

-- The sum of the digits function
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Main theorem
theorem number_satisfying_condition : ∃ n : ℕ, n * sum_of_digits n = 2008 ∧ n = 251 :=
by
  sorry

end number_satisfying_condition_l40_40454


namespace cone_radius_l40_40710

open Real

theorem cone_radius
  (l : ℝ) (L : ℝ) (h_l : l = 5) (h_L : L = 15 * π) :
  ∃ r : ℝ, L = π * r * l ∧ r = 3 :=
by
  sorry

end cone_radius_l40_40710


namespace division_result_l40_40542

theorem division_result : (8900 / 6) / 4 = 370.8333 :=
by sorry

end division_result_l40_40542


namespace defective_percentage_is_0_05_l40_40612

-- Define the problem conditions as Lean definitions
def total_meters : ℕ := 4000
def defective_meters : ℕ := 2

-- Define the percentage calculation function
def percentage_defective (defective total : ℕ) : ℚ :=
  (defective : ℚ) / (total : ℚ) * 100

-- Rewrite the proof statement using these definitions
theorem defective_percentage_is_0_05 :
  percentage_defective defective_meters total_meters = 0.05 :=
by
  sorry

end defective_percentage_is_0_05_l40_40612


namespace min_value_l40_40632

-- Defining the conditions
variables {x y z : ℝ}

-- Problem statement translating the conditions
theorem min_value (h₁ : x > 0) (h₂ : y > 0) (h₃ : z > 0) (h₄ : x + y + z = 5) : 
  ∃ (minval : ℝ), minval = 36/5 ∧ ∀ w, w = (1/x + 4/y + 9/z) → w ≥ minval :=
by
  sorry

end min_value_l40_40632


namespace divide_into_parts_l40_40684

theorem divide_into_parts (x y : ℚ) (h_sum : x + y = 10) (h_diff : y - x = 5) : 
  x = 5 / 2 ∧ y = 15 / 2 := 
sorry

end divide_into_parts_l40_40684


namespace value_of_m_l40_40333

theorem value_of_m (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (3 / (x - 2) + (x + m) / (2 - x) = 1)) → m = 1 :=
by
  sorry

end value_of_m_l40_40333


namespace ways_to_insert_plus_l40_40210

-- Definition of the problem conditions
def num_ones : ℕ := 15
def target_sum : ℕ := 0 

-- Binomial coefficient calculation
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- The theorem to be proven
theorem ways_to_insert_plus :
  binomial 14 9 = 2002 :=
by
  sorry

end ways_to_insert_plus_l40_40210


namespace probability_three_even_dice_l40_40377

theorem probability_three_even_dice :
  let p_even := 1 / 2
  let combo := Nat.choose 5 3
  let probability := combo * (p_even ^ 3) * ((1 - p_even) ^ 2)
  probability = 5 / 16 := 
by
  sorry

end probability_three_even_dice_l40_40377


namespace incorrect_transformation_l40_40056

-- Definitions based on conditions
variable (a b c : ℝ)

-- Conditions
axiom eq_add_six (h : a = b) : a + 6 = b + 6
axiom eq_div_nine (h : a = b) : a / 9 = b / 9
axiom eq_mul_c (h : a / c = b / c) (hc : c ≠ 0) : a = b
axiom eq_div_neg_two (h : -2 * a = -2 * b) : a = b

-- Proving the incorrect transformation statement
theorem incorrect_transformation : ¬ (a = -b) ∧ (-2 * a = -2 * b → a = b) := by
  sorry

end incorrect_transformation_l40_40056


namespace total_baseball_cards_l40_40912

-- Define the number of baseball cards each person has
def mary_cards : ℕ := 15
def sam_cards : ℕ := 15
def keith_cards : ℕ := 15
def alyssa_cards : ℕ := 15
def john_cards : ℕ := 12
def sarah_cards : ℕ := 18
def emma_cards : ℕ := 10

-- The total number of baseball cards they have
theorem total_baseball_cards :
  mary_cards + sam_cards + keith_cards + alyssa_cards + john_cards + sarah_cards + emma_cards = 100 :=
by
  sorry

end total_baseball_cards_l40_40912


namespace initial_students_count_l40_40327

theorem initial_students_count (n : ℕ) (W : ℝ) 
  (h1 : W = n * 28) 
  (h2 : W + 1 = (n + 1) * 27.1) : 
  n = 29 := by
  sorry

end initial_students_count_l40_40327


namespace marts_income_percentage_l40_40746

variable (J T M : ℝ)

theorem marts_income_percentage (h1 : M = 1.40 * T) (h2 : T = 0.60 * J) : M = 0.84 * J :=
by
  sorry

end marts_income_percentage_l40_40746


namespace ferries_are_divisible_by_4_l40_40151

theorem ferries_are_divisible_by_4 (t T : ℕ) (H : ∃ n : ℕ, T = n * t) :
  ∃ N : ℕ, N = 4 * (T / t) ∧ N % 4 = 0 :=
by
  sorry

end ferries_are_divisible_by_4_l40_40151


namespace train_speed_solution_l40_40972

def train_speed_problem (L v : ℝ) (man_time platform_time : ℝ) (platform_length : ℝ) :=
  man_time = 12 ∧
  platform_time = 30 ∧
  platform_length = 180 ∧
  L = v * man_time ∧
  (L + platform_length) = v * platform_time

theorem train_speed_solution (L v : ℝ) (h : train_speed_problem L v 12 30 180) :
  v * 3.6 = 36 :=
by
  sorry

end train_speed_solution_l40_40972


namespace sum_of_consecutive_integers_l40_40250

theorem sum_of_consecutive_integers (S : ℕ) (hS : S = 560):
  ∃ (N : ℕ), N = 11 ∧ 
  ∀ n (k : ℕ), 2 ≤ n → (n * (2 * k + n - 1)) = 1120 → N = 11 :=
by
  sorry

end sum_of_consecutive_integers_l40_40250


namespace fraction_product_is_one_l40_40023

theorem fraction_product_is_one : 
  (1 / 4) * (1 / 5) * (1 / 6) * 120 = 1 :=
by 
  sorry

end fraction_product_is_one_l40_40023


namespace carrots_thrown_out_l40_40903

def initial_carrots := 19
def additional_carrots := 46
def total_current_carrots := 61

def total_picked := initial_carrots + additional_carrots

theorem carrots_thrown_out : total_picked - total_current_carrots = 4 := by
  sorry

end carrots_thrown_out_l40_40903


namespace ratio_a_d_l40_40854

theorem ratio_a_d (a b c d : ℕ) 
  (hab : a * 4 = b * 3) 
  (hbc : b * 9 = c * 7) 
  (hcd : c * 7 = d * 5) : 
  a * 12 = d :=
sorry

end ratio_a_d_l40_40854


namespace smallest_n_l40_40868

theorem smallest_n (o y v : ℕ) (h1 : 18 * o = 21 * y) (h2 : 21 * y = 10 * v) (h3 : 10 * v = 30 * n) : 
  n = 21 := by
  sorry

end smallest_n_l40_40868


namespace find_number_l40_40055

theorem find_number (x : ℝ) (h : 3034 - (x / 20.04) = 2984) : x = 1002 :=
by
  sorry

end find_number_l40_40055


namespace values_of_x_defined_l40_40166

noncomputable def problem_statement (x : ℝ) : Prop :=
  (2 * x - 3 > 0) ∧ (5 - 2 * x > 0)

theorem values_of_x_defined (x : ℝ) :
  problem_statement x ↔ (3 / 2 < x ∧ x < 5 / 2) :=
by sorry

end values_of_x_defined_l40_40166


namespace weeks_of_exercise_l40_40094

def hours_per_day : ℕ := 1
def days_per_week : ℕ := 5
def total_hours : ℕ := 40

def weekly_hours : ℕ := hours_per_day * days_per_week

theorem weeks_of_exercise (W : ℕ) (h : total_hours = weekly_hours * W) : W = 8 :=
by
  sorry

end weeks_of_exercise_l40_40094


namespace arccos_one_over_sqrt_two_eq_pi_four_l40_40226

theorem arccos_one_over_sqrt_two_eq_pi_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := 
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_four_l40_40226


namespace find_x_l40_40346

theorem find_x (x : ℝ) (h : (4 + x) / (6 + x) = (2 + x) / (3 + x)) : x = 0 :=
by
  sorry

end find_x_l40_40346


namespace mouse_lives_correct_l40_40593

def cat_lives : ℕ := 9
def dog_lives : ℕ := cat_lives - 3
def mouse_lives : ℕ := dog_lives + 7

theorem mouse_lives_correct : mouse_lives = 13 :=
by
  sorry

end mouse_lives_correct_l40_40593


namespace math_problem_l40_40266

variable {x a b : ℝ}

theorem math_problem (h1 : x < a) (h2 : a < 0) (h3 : b = -a) : x^2 > b^2 ∧ b^2 > 0 :=
by {
  sorry
}

end math_problem_l40_40266


namespace max_value_ab_ac_bc_l40_40477

open Real

theorem max_value_ab_ac_bc {a b c : ℝ} (h : a + 3 * b + c = 6) : 
  ab + ac + bc ≤ 4 :=
sorry

end max_value_ab_ac_bc_l40_40477


namespace exist_six_subsets_of_six_elements_l40_40902

theorem exist_six_subsets_of_six_elements (n m : ℕ) (X : Finset ℕ) (A : Fin m → Finset ℕ) :
    n > 6 →
    X.card = n →
    (∀ i, (A i).card = 5 ∧ (A i ⊆ X)) →
    m > (n * (n-1) * (n-2) * (n-3) * (4*n-15)) / 600 →
    ∃ i1 i2 i3 i4 i5 i6 : Fin m,
      i1 < i2 ∧ i2 < i3 ∧ i3 < i4 ∧ i4 < i5 ∧ i5 < i6 ∧
      (A i1 ∪ A i2 ∪ A i3 ∪ A i4 ∪ A i5 ∪ A i6).card = 6 := 
sorry

end exist_six_subsets_of_six_elements_l40_40902


namespace jack_jill_same_speed_l40_40866

-- Definitions for Jack and Jill's conditions
def jacks_speed (x : ℝ) : ℝ := x^2 - 13*x - 48
def jills_distance (x : ℝ) : ℝ := x^2 - 5*x - 84
def jills_time (x : ℝ) : ℝ := x + 8

-- Theorem stating the same walking speed given the conditions
theorem jack_jill_same_speed (x : ℝ) (h : jacks_speed x = jills_distance x / jills_time x) : 
  jacks_speed x = 6 :=
by
  sorry

end jack_jill_same_speed_l40_40866


namespace intersection_point_l40_40349

theorem intersection_point (x y : ℝ) (h1 : y = x + 1) (h2 : y = -x + 1) : (x = 0) ∧ (y = 1) := 
by
  sorry

end intersection_point_l40_40349


namespace ratio_qp_l40_40468

theorem ratio_qp (P Q : ℤ)
  (h : ∀ x : ℝ, x ≠ -3 → x ≠ 0 → x ≠ 6 → 
    P / (x + 3) + Q / (x * (x - 6)) = (x^2 - 4 * x + 15) / (x * (x + 3) * (x - 6))) : 
  Q / P = 5 := 
sorry

end ratio_qp_l40_40468


namespace sum_of_consecutive_integers_product_is_negative_336_l40_40361

theorem sum_of_consecutive_integers_product_is_negative_336 :
  ∃ (n : ℤ), (n - 1) * n * (n + 1) = -336 ∧ (n - 1) + n + (n + 1) = -21 :=
by
  sorry

end sum_of_consecutive_integers_product_is_negative_336_l40_40361


namespace ball_bounce_height_l40_40625

theorem ball_bounce_height (b : ℕ) (h₀: ℝ) (r: ℝ) (h_final: ℝ) :
  h₀ = 200 ∧ r = 3 / 4 ∧ h_final = 25 →
  200 * (3 / 4) ^ b < 25 ↔ b ≥ 25 := by
  sorry

end ball_bounce_height_l40_40625


namespace dave_apps_left_l40_40599

def initial_apps : ℕ := 24
def initial_files : ℕ := 9
def files_left : ℕ := 5
def apps_left (files_left: ℕ) : ℕ := files_left + 7

theorem dave_apps_left :
  apps_left files_left = 12 :=
by
  sorry

end dave_apps_left_l40_40599


namespace inequality_abc_l40_40836

theorem inequality_abc (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + b*c + c*a :=
by
  sorry

end inequality_abc_l40_40836


namespace percentage_of_a_l40_40618

theorem percentage_of_a (a : ℕ) (x : ℕ) (h1 : a = 190) (h2 : (x * a) / 100 = 95) : x = 50 := by
  sorry

end percentage_of_a_l40_40618


namespace points_on_x_eq_3_is_vertical_line_points_with_x_lt_3_points_with_x_gt_3_points_on_y_eq_2_is_horizontal_line_points_with_y_gt_2_l40_40927

open Set

-- Define the point in the coordinate plane as a product of real numbers
def Point := ℝ × ℝ

-- Prove points with x = 3 form a vertical line
theorem points_on_x_eq_3_is_vertical_line : {p : Point | p.1 = 3} = {p : Point | ∀ y : ℝ, (3, y) = p} := sorry

-- Prove points with x < 3 lie to the left of x = 3
theorem points_with_x_lt_3 : {p : Point | p.1 < 3} = {p : Point | ∀ x y : ℝ, x < 3 → p = (x, y)} := sorry

-- Prove points with x > 3 lie to the right of x = 3
theorem points_with_x_gt_3 : {p : Point | p.1 > 3} = {p : Point | ∀ x y : ℝ, x > 3 → p = (x, y)} := sorry

-- Prove points with y = 2 form a horizontal line
theorem points_on_y_eq_2_is_horizontal_line : {p : Point | p.2 = 2} = {p : Point | ∀ x : ℝ, (x, 2) = p} := sorry

-- Prove points with y > 2 lie above y = 2
theorem points_with_y_gt_2 : {p : Point | p.2 > 2} = {p : Point | ∀ x y : ℝ, y > 2 → p = (x, y)} := sorry

end points_on_x_eq_3_is_vertical_line_points_with_x_lt_3_points_with_x_gt_3_points_on_y_eq_2_is_horizontal_line_points_with_y_gt_2_l40_40927


namespace problem_statement_l40_40512

variable {a x y : ℝ}

theorem problem_statement (hx : 0 < a) (ha : a < 1) (h : a^x < a^y) : x^3 > y^3 :=
sorry

end problem_statement_l40_40512


namespace foxes_hunt_duration_l40_40176

variable (initial_weasels : ℕ) (initial_rabbits : ℕ) (remaining_rodents : ℕ)
variable (foxes : ℕ) (weasels_per_week : ℕ) (rabbits_per_week : ℕ)

def total_rodents_per_week (weasels_per_week rabbits_per_week foxes : ℕ) : ℕ :=
  foxes * (weasels_per_week + rabbits_per_week)

def initial_rodents (initial_weasels initial_rabbits : ℕ) : ℕ :=
  initial_weasels + initial_rabbits

def total_rodents_caught (initial_rodents remaining_rodents : ℕ) : ℕ :=
  initial_rodents - remaining_rodents

def weeks_hunted (total_rodents_caught total_rodents_per_week : ℕ) : ℕ :=
  total_rodents_caught / total_rodents_per_week

theorem foxes_hunt_duration
  (initial_weasels := 100) (initial_rabbits := 50) (remaining_rodents := 96)
  (foxes := 3) (weasels_per_week := 4) (rabbits_per_week := 2) :
  weeks_hunted (total_rodents_caught (initial_rodents initial_weasels initial_rabbits) remaining_rodents) 
                 (total_rodents_per_week weasels_per_week rabbits_per_week foxes) = 3 :=
by
  sorry

end foxes_hunt_duration_l40_40176


namespace least_positive_integer_l40_40750
  
theorem least_positive_integer 
  (x : ℕ) (d n : ℕ) (p : ℕ) 
  (h_eq : x = 10^p * d + n) 
  (h_ratio : n = x / 17) 
  (h_cond1 : 1 ≤ d) 
  (h_cond2 : d ≤ 9)
  (h_nonzero : n > 0) : 
  x = 10625 :=
by
  sorry

end least_positive_integer_l40_40750


namespace max_value_2x_minus_y_l40_40307

theorem max_value_2x_minus_y (x y : ℝ) (h : x^2 / 4 + y^2 / 9 = 1) : 2 * x - y ≤ 5 :=
sorry

end max_value_2x_minus_y_l40_40307


namespace fraction_meaningful_range_l40_40898

variable (x : ℝ)

theorem fraction_meaningful_range (h : x - 2 ≠ 0) : x ≠ 2 :=
by
  sorry

end fraction_meaningful_range_l40_40898


namespace find_x_unique_l40_40405

def productOfDigits (x : ℕ) : ℕ :=
  -- Assuming the implementation of product of digits function
  sorry

def sumOfDigits (x : ℕ) : ℕ :=
  -- Assuming the implementation of sum of digits function
  sorry

theorem find_x_unique : ∀ x : ℕ, (productOfDigits x = 44 * x - 86868 ∧ ∃ n : ℕ, sumOfDigits x = n^3) -> x = 1989 :=
by
  intros x h
  sorry

end find_x_unique_l40_40405


namespace triangle_inequality_l40_40446

theorem triangle_inequality 
  (A B C : ℝ) -- angle measures
  (a b c : ℝ) -- side lengths
  (h1 : a = b * (Real.cos C) + c * (Real.cos B)) 
  (cos_half_C_pos : 0 < Real.cos (C/2)) 
  (cos_half_C_lt_one : Real.cos (C/2) < 1)
  (cos_half_B_pos : 0 < Real.cos (B/2)) 
  (cos_half_B_lt_one : Real.cos (B/2) < 1) :
  2 * b * Real.cos (C / 2) + 2 * c * Real.cos (B / 2) > a + b + c :=
by
  sorry

end triangle_inequality_l40_40446


namespace right_triangle_hypotenuse_length_l40_40369

theorem right_triangle_hypotenuse_length (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) : 
  ∃ c : ℝ, c^2 = a^2 + b^2 ∧ c = 10 :=
by
  sorry

end right_triangle_hypotenuse_length_l40_40369


namespace part1_part2_l40_40067

def A (x : ℤ) := ∃ m n : ℤ, x = m^2 - n^2
def B (x : ℤ) := ∃ k : ℤ, x = 2 * k + 1

theorem part1 (h1: A 8) (h2: A 9) (h3: ¬ A 10) : 
  (A 8) ∧ (A 9) ∧ (¬ A 10) :=
by {
  sorry
}

theorem part2 (x : ℤ) (h : A x) : B x :=
by {
  sorry
}

end part1_part2_l40_40067


namespace smallest_number_l40_40582

/-
  Let's declare each number in its base form as variables,
  convert them to their decimal equivalents, and assert that the decimal
  value of $(31)_4$ is the smallest among the given numbers.

  Note: We're not providing the proof steps, just the statement.
-/

noncomputable def A_base7_to_dec : ℕ := 2 * 7^1 + 0 * 7^0
noncomputable def B_base5_to_dec : ℕ := 3 * 5^1 + 0 * 5^0
noncomputable def C_base6_to_dec : ℕ := 2 * 6^1 + 3 * 6^0
noncomputable def D_base4_to_dec : ℕ := 3 * 4^1 + 1 * 4^0

theorem smallest_number : D_base4_to_dec < A_base7_to_dec ∧ D_base4_to_dec < B_base5_to_dec ∧ D_base4_to_dec < C_base6_to_dec := by
  sorry

end smallest_number_l40_40582


namespace floor_neg_seven_fourths_l40_40493

theorem floor_neg_seven_fourths : Int.floor (-7 / 4 : ℚ) = -2 := 
by 
  sorry

end floor_neg_seven_fourths_l40_40493


namespace lcm_gcd_product_l40_40501

theorem lcm_gcd_product (a b : ℕ) (ha : a = 12) (hb : b = 9) :
  Nat.lcm a b * Nat.gcd a b = 108 := by
  rw [ha, hb]
  -- Replace with Nat library functions and calculate
  sorry

end lcm_gcd_product_l40_40501


namespace circular_seat_coloring_l40_40634

def count_colorings (n : ℕ) : ℕ :=
  sorry

theorem circular_seat_coloring :
  count_colorings 6 = 66 :=
by
  sorry

end circular_seat_coloring_l40_40634


namespace customers_left_is_31_l40_40341

-- Define the initial number of customers
def initial_customers : ℕ := 33

-- Define the number of additional customers
def additional_customers : ℕ := 26

-- Define the final number of customers after some left and new ones came
def final_customers : ℕ := 28

-- Define the number of customers who left 
def customers_left (x : ℕ) : Prop :=
  (initial_customers - x) + additional_customers = final_customers

-- The proof statement that we aim to prove
theorem customers_left_is_31 : ∃ x : ℕ, customers_left x ∧ x = 31 :=
by
  use 31
  unfold customers_left
  sorry

end customers_left_is_31_l40_40341


namespace average_weight_of_a_and_b_l40_40843

-- Given conditions as Lean definitions
variable (A B C : ℝ)
variable (h1 : (A + B + C) / 3 = 45)
variable (h2 : (B + C) / 2 = 46)
variable (hB : B = 37)

-- The statement we want to prove
theorem average_weight_of_a_and_b : (A + B) / 2 = 40 := by
  sorry

end average_weight_of_a_and_b_l40_40843


namespace larger_number_is_400_l40_40641

def problem_statement : Prop :=
  ∃ (a b hcf lcm num1 num2 : ℕ),
  hcf = 25 ∧
  a = 14 ∧
  b = 16 ∧
  lcm = hcf * a * b ∧
  num1 = hcf * a ∧
  num2 = hcf * b ∧
  num1 < num2 ∧
  num2 = 400

theorem larger_number_is_400 : problem_statement :=
  sorry

end larger_number_is_400_l40_40641


namespace problem1_solution_set_problem2_proof_l40_40088

-- Define the function f(x) with a given value of a.
def f (x : ℝ) (a : ℝ) : ℝ := |x + a|

-- Problem 1: Solve the inequality f(x) ≥ 5 - |x - 2| when a = 1.
theorem problem1_solution_set (x : ℝ) :
  f x 1 ≥ 5 - |x - 2| ↔ x ∈ (Set.Iic (-2) ∪ Set.Ici 3) :=
sorry

-- Problem 2: Given the solution set of f(x) ≤ 5 is [-9, 1] and the equation 1/m + 1/(2n) = a, prove m + 2n ≥ 1
theorem problem2_proof (a m n : ℝ) (hma : a = 4) (hmpos : m > 0) (hnpos : n > 0) :
  (1 / m + 1 / (2 * n) = a) → m + 2 * n ≥ 1 :=
sorry

end problem1_solution_set_problem2_proof_l40_40088


namespace evaluate_expression_l40_40093

theorem evaluate_expression (a x : ℝ) (h : x = 2 * a + 6) : 2 * (x - a + 5) = 2 * a + 22 := by
  sorry

end evaluate_expression_l40_40093


namespace cost_of_bananas_l40_40705

theorem cost_of_bananas (A B : ℕ) (h1 : 2 * A + B = 7) (h2 : A + B = 5) : B = 3 :=
by
  sorry

end cost_of_bananas_l40_40705


namespace avg_mpg_sum_l40_40605

def first_car_gallons : ℕ := 25
def second_car_gallons : ℕ := 35
def total_miles : ℕ := 2275
def first_car_mpg : ℕ := 40

noncomputable def sum_of_avg_mpg_of_two_cars : ℝ := 76.43

theorem avg_mpg_sum :
  let first_car_miles := (first_car_gallons * first_car_mpg : ℕ)
  let second_car_miles := total_miles - first_car_miles
  let second_car_mpg := (second_car_miles : ℝ) / second_car_gallons
  let sum_avg_mpg := (first_car_mpg : ℝ) + second_car_mpg
  sum_avg_mpg = sum_of_avg_mpg_of_two_cars :=
by
  sorry

end avg_mpg_sum_l40_40605


namespace fraction_evaluation_l40_40045

theorem fraction_evaluation :
  (2 + 3 * 6) / (23 + 6) = 20 / 29 := by
  -- Proof can be filled in here
  sorry

end fraction_evaluation_l40_40045


namespace sum_of_reciprocals_of_squares_roots_eq_14_3125_l40_40946

theorem sum_of_reciprocals_of_squares_roots_eq_14_3125
  (α β γ : ℝ)
  (h1 : α + β + γ = 15)
  (h2 : α * β + β * γ + γ * α = 26)
  (h3 : α * β * γ = -8) :
  (1 / α^2) + (1 / β^2) + (1 / γ^2) = 14.3125 := 
by
  sorry

end sum_of_reciprocals_of_squares_roots_eq_14_3125_l40_40946


namespace multiplicative_inverse_l40_40152

theorem multiplicative_inverse (a b n : ℤ) (h₁ : a = 208) (h₂ : b = 240) (h₃ : n = 307) : 
  (a * b) % n = 1 :=
by
  rw [h₁, h₂, h₃]
  norm_num
  sorry

end multiplicative_inverse_l40_40152


namespace solution_l40_40545

-- Define the vectors and their conditions
variables {u v : ℝ}

def vec1 := (3, -2)
def vec2 := (9, -7)
def vec3 := (-1, 2)
def vec4 := (-3, 4)

-- Condition: The linear combination of vec1 and u*vec2 equals the linear combination of vec3 and v*vec4.
axiom H : (3 + 9 * u, -2 - 7 * u) = (-1 - 3 * v, 2 + 4 * v)

-- Statement of the proof problem:
theorem solution : u = -4/15 ∧ v = -8/15 :=
by {
  sorry
}

end solution_l40_40545


namespace edric_hourly_rate_l40_40977

-- Define conditions
def edric_monthly_salary : ℝ := 576
def edric_weekly_hours : ℝ := 8 * 6 -- 48 hours
def average_weeks_per_month : ℝ := 4.33
def edric_monthly_hours : ℝ := edric_weekly_hours * average_weeks_per_month -- Approx 207.84 hours

-- Define the expected result
def edric_expected_hourly_rate : ℝ := 2.77

-- Proof statement
theorem edric_hourly_rate :
  edric_monthly_salary / edric_monthly_hours = edric_expected_hourly_rate :=
by
  sorry

end edric_hourly_rate_l40_40977


namespace gym_distance_diff_l40_40942

theorem gym_distance_diff (D G : ℕ) (hD : D = 10) (hG : G = 7) : G - D / 2 = 2 := by
  sorry

end gym_distance_diff_l40_40942


namespace geometric_sequence_sum_l40_40725

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) (h1 : ∀ n, a (n + 1) = a n * r)
    (h2 : r = 2) (h3 : a 1 * 2 + a 3 * 8 + a 5 * 32 = 3) :
    a 4 * 16 + a 6 * 64 + a 8 * 256 = 24 :=
sorry

end geometric_sequence_sum_l40_40725


namespace jovial_frogs_not_green_l40_40324

variables {Frog : Type} (jovial green can_jump can_swim : Frog → Prop)

theorem jovial_frogs_not_green :
  (∀ frog, jovial frog → can_swim frog) →
  (∀ frog, green frog → ¬ can_jump frog) →
  (∀ frog, ¬ can_jump frog → ¬ can_swim frog) →
  (∀ frog, jovial frog → ¬ green frog) :=
by
  intros h1 h2 h3 frog hj
  sorry

end jovial_frogs_not_green_l40_40324


namespace sophomores_in_program_l40_40117

-- Define variables
variable (P S : ℕ)

-- Conditions for the problem
def total_students (P S : ℕ) : Prop := P + S = 36
def percent_sophomores_club (P S : ℕ) (x : ℕ) : Prop := x = 3 * P / 10
def percent_seniors_club (P S : ℕ) (y : ℕ) : Prop := y = S / 4
def equal_club_members (x y : ℕ) : Prop := x = y

-- Theorem stating the problem and proof goal
theorem sophomores_in_program
  (x y : ℕ)
  (h1 : total_students P S)
  (h2 : percent_sophomores_club P S x)
  (h3 : percent_seniors_club P S y)
  (h4 : equal_club_members x y) :
  P = 15 := 
sorry

end sophomores_in_program_l40_40117


namespace area_not_covered_correct_l40_40745

-- Define the dimensions of the rectangle
def rectangle_length : ℕ := 10
def rectangle_width : ℕ := 8

-- Define the side length of the square
def square_side_length : ℕ := 5

-- The area of the rectangle
def rectangle_area : ℕ := rectangle_length * rectangle_width

-- The area of the square
def square_area : ℕ := square_side_length * square_side_length

-- The area of the region not covered by the square
def area_not_covered : ℕ := rectangle_area - square_area

-- The theorem statement asserting the required area
theorem area_not_covered_correct : area_not_covered = 55 :=
by
  -- Proof is omitted
  sorry

end area_not_covered_correct_l40_40745


namespace consecutive_nums_sum_as_product_l40_40530

theorem consecutive_nums_sum_as_product {n : ℕ} (h : 100 < n) :
  ∃ (a b c : ℕ), (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ (2 ≤ a) ∧ (2 ≤ b) ∧ (2 ≤ c) ∧ 
  ((n + (n+1) + (n+2) = a * b * c) ∨ ((n+1) + (n+2) + (n+3) = a * b * c)) :=
by
  sorry

end consecutive_nums_sum_as_product_l40_40530


namespace solve_for_x_l40_40102

-- declare an existential quantifier to encapsulate the condition and the answer.
theorem solve_for_x : ∃ x : ℝ, x + (x + 2) + (x + 4) = 24 ∧ x = 6 := 
by 
  -- begin sorry to skip the proof part
  sorry

end solve_for_x_l40_40102


namespace smallest_x_l40_40036

theorem smallest_x (x : ℕ) :
  (x % 6 = 5) ∧ (x % 7 = 6) ∧ (x % 8 = 7) → x = 167 :=
by
  sorry

end smallest_x_l40_40036


namespace max_value_proof_l40_40129

noncomputable def max_value (x y z : ℝ) : ℝ :=
  1 / x + 2 / y + 3 / z

theorem max_value_proof (x y z : ℝ) (h1 : 2 / 5 ≤ z ∧ z ≤ min x y)
    (h2 : x * z ≥ 4 / 15) (h3 : y * z ≥ 1 / 5) : max_value x y z ≤ 13 := 
by
  sorry

end max_value_proof_l40_40129


namespace hcf_of_given_numbers_l40_40311

def hcf (x y : ℕ) : ℕ := Nat.gcd x y

theorem hcf_of_given_numbers :
  ∃ (A B : ℕ), A = 33 ∧ A * B = 363 ∧ hcf A B = 11 := 
by
  sorry

end hcf_of_given_numbers_l40_40311


namespace volume_of_inscribed_tetrahedron_l40_40198

theorem volume_of_inscribed_tetrahedron (r h : ℝ) (V : ℝ) (tetrahedron_inscribed : Prop) 
  (cylinder_condition : π * r^2 * h = 1) 
  (inscribed : tetrahedron_inscribed → True) : 
  V ≤ 2 / (3 * π) :=
sorry

end volume_of_inscribed_tetrahedron_l40_40198


namespace find_value_of_x_l40_40110

theorem find_value_of_x (x y z : ℤ) (h1 : x > y) (h2 : y > z) (h3 : z = 3)
  (h4 : 2 * x + 3 * y + 3 * z = 5 * y + 11) (h5 : (x = y + 1) ∧ (y = z + 1)) :
  x = 5 := 
sorry

end find_value_of_x_l40_40110


namespace min_n_Sn_l40_40155

/--
Given an arithmetic sequence {a_n}, let S_n denote the sum of its first n terms.
If S_4 = -2, S_5 = 0, and S_6 = 3, then the minimum value of n * S_n is -9.
-/
theorem min_n_Sn (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h₁ : S 4 = -2)
  (h₂ : S 5 = 0)
  (h₃ : S 6 = 3)
  (h₄ : ∀ n, S n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2)
  : ∃ n : ℕ, n * S n = -9 := 
sorry

end min_n_Sn_l40_40155


namespace georgina_teaches_2_phrases_per_week_l40_40450

theorem georgina_teaches_2_phrases_per_week
    (total_phrases : ℕ) 
    (initial_phrases : ℕ) 
    (days_owned : ℕ)
    (phrases_per_week : ℕ):
    total_phrases = 17 → 
    initial_phrases = 3 → 
    days_owned = 49 → 
    phrases_per_week = (total_phrases - initial_phrases) / (days_owned / 7) → 
    phrases_per_week = 2 := 
by
  intros h_total h_initial h_days h_calc
  rw [h_total, h_initial, h_days] at h_calc
  sorry  -- Proof to be filled

end georgina_teaches_2_phrases_per_week_l40_40450


namespace sin_810_eq_one_l40_40259

theorem sin_810_eq_one : Real.sin (810 * Real.pi / 180) = 1 :=
by
  -- You can add the proof here
  sorry

end sin_810_eq_one_l40_40259


namespace factor_1_factor_2_l40_40531

theorem factor_1 {x : ℝ} : x^2 - 4*x + 3 = (x - 1) * (x - 3) :=
sorry

theorem factor_2 {x : ℝ} : 4*x^2 + 12*x - 7 = (2*x + 7) * (2*x - 1) :=
sorry

end factor_1_factor_2_l40_40531


namespace smallest_positive_even_integer_l40_40082

noncomputable def smallest_even_integer (n : ℕ) : ℕ := 
  if 2 * n > 0 ∧ (3^(n * (n + 1) / 8)) > 500 then n else 0

theorem smallest_positive_even_integer :
  smallest_even_integer 6 = 6 :=
by
  -- Skipping the proofs
  sorry

end smallest_positive_even_integer_l40_40082


namespace fraction_evaluation_l40_40478

def number_of_primes_between_10_and_30 : ℕ := 6

theorem fraction_evaluation : (number_of_primes_between_10_and_30^2 - 4) / (number_of_primes_between_10_and_30 + 2) = 4 := by
  sorry

end fraction_evaluation_l40_40478


namespace greatest_large_chips_l40_40862

theorem greatest_large_chips :
  ∃ (l : ℕ), (∃ (s : ℕ), ∃ (p : ℕ), s + l = 70 ∧ s = l + p ∧ Nat.Prime p) ∧ 
  (∀ (l' : ℕ), (∃ (s' : ℕ), ∃ (p' : ℕ), s' + l' = 70 ∧ s' = l' + p' ∧ Nat.Prime p') → l' ≤ 34) :=
sorry

end greatest_large_chips_l40_40862


namespace proof_of_problem_l40_40522

noncomputable def f : ℝ → ℝ := sorry  -- define f as a function in ℝ to ℝ

theorem proof_of_problem 
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_f1 : f 1 = 1)
  (h_periodic : ∀ x : ℝ, f (x + 6) = f x + f 3) :
  f 2015 + f 2016 = -1 := 
sorry

end proof_of_problem_l40_40522


namespace negation_equiv_l40_40453

open Classical

-- Proposition p
def p : Prop := ∃ x : ℝ, x^2 - x + 1 = 0

-- Negation of proposition p
def neg_p : Prop := ∀ x : ℝ, x^2 - x + 1 ≠ 0

-- Statement to prove the equivalence of the negation of p and neg_p
theorem negation_equiv :
  ¬p ↔ neg_p := 
sorry

end negation_equiv_l40_40453


namespace locus_centers_of_tangent_circles_l40_40603

theorem locus_centers_of_tangent_circles (a b : ℝ) :
  (x^2 + y^2 = 1) ∧ ((x - 1)^2 + (y -1)^2 = 81) →
  (a^2 + b^2 - (2 * a * b) / 63 - (66 * a) / 63 - (66 * b) / 63 + 17 = 0) :=
by
  sorry

end locus_centers_of_tangent_circles_l40_40603


namespace container_volume_ratio_l40_40026

theorem container_volume_ratio
  (A B C : ℚ)  -- A is the volume of the first container, B is the volume of the second container, C is the volume of the third container
  (h1 : (8 / 9) * A = (7 / 9) * B)  -- Condition: First container was 8/9 full and second container gets filled to 7/9 after transfer.
  (h2 : (7 / 9) * B + (1 / 2) * C = C)  -- Condition: Mixing contents from second and third containers completely fill third container.
  : A / C = 63 / 112 := sorry  -- We need to prove this.

end container_volume_ratio_l40_40026


namespace indira_cricket_minutes_l40_40885

def totalMinutesSeanPlayed (sean_minutes_per_day : ℕ) (days : ℕ) : ℕ :=
  sean_minutes_per_day * days

def totalMinutesIndiraPlayed (total_minutes_together : ℕ) (total_minutes_sean : ℕ) : ℕ :=
  total_minutes_together - total_minutes_sean

theorem indira_cricket_minutes :
  totalMinutesIndiraPlayed 1512 (totalMinutesSeanPlayed 50 14) = 812 :=
by
  sorry

end indira_cricket_minutes_l40_40885


namespace Mary_is_10_years_younger_l40_40476

theorem Mary_is_10_years_younger
  (betty_age : ℕ)
  (albert_age : ℕ)
  (mary_age : ℕ)
  (h1 : albert_age = 2 * mary_age)
  (h2 : albert_age = 4 * betty_age)
  (h_betty : betty_age = 5) :
  (albert_age - mary_age) = 10 :=
  by
  sorry

end Mary_is_10_years_younger_l40_40476


namespace find_initial_investment_l40_40490

open Real

noncomputable def initial_investment (x : ℝ) (years : ℕ) (final_value : ℝ) : ℝ := 
  final_value / (3 ^ (years / (112 / x)))

theorem find_initial_investment :
  let x := 8
  let years := 28
  let final_value := 31500
  initial_investment x years final_value = 3500 := 
by 
  sorry

end find_initial_investment_l40_40490


namespace div_condition_nat_l40_40544

theorem div_condition_nat (n : ℕ) : (n + 1) ∣ (n^2 + 1) ↔ n = 0 ∨ n = 1 :=
by
  sorry

end div_condition_nat_l40_40544


namespace fraction_of_products_inspected_jane_l40_40399

theorem fraction_of_products_inspected_jane 
  (P : ℝ) 
  (J : ℝ) 
  (John_rejection_rate : ℝ) 
  (Jane_rejection_rate : ℝ)
  (Total_rejection_rate : ℝ) 
  (hJohn : John_rejection_rate = 0.005) 
  (hJane : Jane_rejection_rate = 0.008) 
  (hTotal : Total_rejection_rate = 0.0075) 
  : J = 5 / 6 := by
{
  sorry
}

end fraction_of_products_inspected_jane_l40_40399


namespace fabric_area_l40_40444

theorem fabric_area (length width : ℝ) (h_length : length = 8) (h_width : width = 3) : 
  length * width = 24 := 
by
  rw [h_length, h_width]
  norm_num

end fabric_area_l40_40444


namespace satisfactory_fraction_is_28_over_31_l40_40408

-- Define the number of students for each grade
def students_with_grade_A := 8
def students_with_grade_B := 7
def students_with_grade_C := 6
def students_with_grade_D := 4
def students_with_grade_E := 3
def students_with_grade_F := 3

-- Calculate the total number of students with satisfactory grades
def satisfactory_grades := students_with_grade_A + students_with_grade_B + students_with_grade_C + students_with_grade_D + students_with_grade_E

-- Calculate the total number of students
def total_students := satisfactory_grades + students_with_grade_F

-- Define the fraction of satisfactory grades
def satisfactory_fraction : ℚ := satisfactory_grades / total_students

-- The main proposition that the satisfactory fraction is 28/31
theorem satisfactory_fraction_is_28_over_31 : satisfactory_fraction = 28 / 31 := by {
  sorry
}

end satisfactory_fraction_is_28_over_31_l40_40408


namespace sum_arithmetic_sequence_l40_40563

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
    ∃ d, ∀ n, a (n+1) = a n + d

-- The conditions
def condition_1 (a : ℕ → ℝ) : Prop :=
    (a 1 + a 2 + a 3 = 6)

def condition_2 (a : ℕ → ℝ) : Prop :=
    (a 10 + a 11 + a 12 = 9)

-- The Theorem statement
theorem sum_arithmetic_sequence :
    is_arithmetic_sequence a →
    condition_1 a →
    condition_2 a →
    (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 + a 12 = 30) :=
by
  intro h1 h2 h3
  sorry

end sum_arithmetic_sequence_l40_40563


namespace germination_percentage_l40_40686

theorem germination_percentage :
  ∀ (seeds_plot1 seeds_plot2 germination_rate1 germination_rate2 : ℝ),
    seeds_plot1 = 300 →
    seeds_plot2 = 200 →
    germination_rate1 = 0.30 →
    germination_rate2 = 0.35 →
    ((germination_rate1 * seeds_plot1 + germination_rate2 * seeds_plot2) / (seeds_plot1 + seeds_plot2)) * 100 = 32 :=
by
  intros seeds_plot1 seeds_plot2 germination_rate1 germination_rate2 h1 h2 h3 h4
  sorry

end germination_percentage_l40_40686


namespace line_through_intersections_l40_40497

def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4
def circle2 (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 9

theorem line_through_intersections (x y : ℝ) :
  circle1 x y → circle2 x y → 2 * x - 3 * y = 0 := 
sorry

end line_through_intersections_l40_40497


namespace maximize_xyz_l40_40187

theorem maximize_xyz (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_sum : x + y + z = 60) :
    (x, y, z) = (20, 40 / 3, 80 / 3) → x^3 * y^2 * z^4 ≤ 20^3 * (40 / 3)^2 * (80 / 3)^4 :=
by
  sorry

end maximize_xyz_l40_40187


namespace largest_d_for_g_of_minus5_l40_40302

theorem largest_d_for_g_of_minus5 (d : ℝ) : 
  (∃ x : ℝ, x^2 + 2 * x + d = -5) → d ≤ -4 :=
by
-- Proof steps will be inserted here
sorry

end largest_d_for_g_of_minus5_l40_40302


namespace kyle_origami_stars_l40_40156

/-- Kyle bought 2 glass bottles, each can hold 15 origami stars,
    then bought another 3 identical glass bottles.
    Prove that the total number of origami stars needed to fill them is 75. -/
theorem kyle_origami_stars : (2 * 15) + (3 * 15) = 75 := by
  sorry

end kyle_origami_stars_l40_40156


namespace min_value_of_a_and_b_l40_40886

theorem min_value_of_a_and_b (a b : ℝ) (h : a ^ 2 + 2 * b ^ 2 = 6) : 
  ∃ (m : ℝ), (∀ (x y : ℝ), x ^ 2 + 2 * y ^ 2 = 6 → x + y ≥ m) ∧ (a + b = m) :=
sorry

end min_value_of_a_and_b_l40_40886


namespace chapters_ratio_l40_40647

theorem chapters_ratio
  (c1 : ℕ) (c2 : ℕ) (total : ℕ) (x : ℕ)
  (h1 : c1 = 20)
  (h2 : c2 = 15)
  (h3 : total = 75)
  (h4 : x = (c1 + 2 * c2) / 2)
  (h5 : c1 + 2 * c2 + x = total) :
  (x : ℚ) / (c1 + 2 * c2 : ℚ) = 1 / 2 :=
by
  sorry

end chapters_ratio_l40_40647


namespace initial_cost_of_smartphone_l40_40900

theorem initial_cost_of_smartphone 
(C : ℝ) 
(h : 0.85 * C = 255) : 
C = 300 := 
sorry

end initial_cost_of_smartphone_l40_40900


namespace isabella_exchange_l40_40436

theorem isabella_exchange (d : ℚ) : 
  (8 * d / 5 - 72 = 4 * d) → d = -30 :=
by
  sorry

end isabella_exchange_l40_40436


namespace monomial_sum_mn_l40_40014

-- Define the conditions as Lean definitions
def is_monomial_sum (x y : ℕ) (m n : ℕ) : Prop :=
  ∃ k : ℕ, (x ^ 2) * (y ^ m) + (x ^ n) * (y ^ 3) = x ^ k

-- State our main theorem
theorem monomial_sum_mn (x y : ℕ) (m n : ℕ) (h : is_monomial_sum x y m n) : m + n = 5 :=
sorry  -- Completion of the proof is not required

end monomial_sum_mn_l40_40014


namespace transport_cost_l40_40208

theorem transport_cost (weight_g : ℕ) (cost_per_kg : ℕ) (weight_kg : ℕ) (total_cost : ℕ)
  (h1 : weight_g = 2000)
  (h2 : cost_per_kg = 15000)
  (h3 : weight_kg = weight_g / 1000)
  (h4 : total_cost = weight_kg * cost_per_kg) :
  total_cost = 30000 :=
by
  sorry

end transport_cost_l40_40208


namespace volume_original_cone_l40_40980

-- Given conditions
def V_cylinder : ℝ := 21
def V_truncated_cone : ℝ := 91

-- To prove: The volume of the original cone is 94.5
theorem volume_original_cone : 
    (∃ (H R h r : ℝ), (π * r^2 * h = V_cylinder) ∧ (1 / 3 * π * (R^2 + R * r + r^2) * (H - h) = V_truncated_cone)) →
    (1 / 3 * π * R^2 * H = 94.5) :=
by
  sorry

end volume_original_cone_l40_40980


namespace coin_value_l40_40418

variables (n d q : ℕ)  -- Number of nickels, dimes, and quarters
variable (total_coins : n + d + q = 30)  -- Total coins condition

-- Original value in cents
def original_value : ℕ := 5 * n + 10 * d + 25 * q

-- Swapped values in cents
def swapped_value : ℕ := 10 * n + 25 * d + 5 * q

-- Condition given about the value difference
variable (value_difference : swapped_value = original_value + 150)

-- Prove the total value of coins is $5.00 (500 cents)
theorem coin_value : original_value = 500 :=
by
  sorry

end coin_value_l40_40418


namespace female_students_proportion_and_count_l40_40174

noncomputable def num_students : ℕ := 30
noncomputable def num_male_students : ℕ := 8
noncomputable def overall_avg_score : ℚ := 90
noncomputable def male_avg_scores : (ℚ × ℚ × ℚ) := (87, 95, 89)
noncomputable def female_avg_scores : (ℚ × ℚ × ℚ) := (92, 94, 91)
noncomputable def avg_attendance_alg_geom : ℚ := 0.85
noncomputable def avg_attendance_calc : ℚ := 0.89

theorem female_students_proportion_and_count :
  ∃ (F : ℕ), F = num_students - num_male_students ∧ (F / num_students : ℚ) = 11 / 15 :=
by
  sorry

end female_students_proportion_and_count_l40_40174


namespace feet_heads_difference_l40_40495

theorem feet_heads_difference :
  let hens := 60
  let goats := 35
  let camels := 6
  let keepers := 10
  let heads := hens + goats + camels + keepers
  let feet := (2 * hens) + (4 * goats) + (4 * camels) + (2 * keepers)
  feet - heads = 193 :=
by
  sorry

end feet_heads_difference_l40_40495


namespace solve_for_F_l40_40473

variable (S W F : ℝ)

def condition1 (S W : ℝ) : Prop := S = W / 3
def condition2 (W F : ℝ) : Prop := W = F + 60
def condition3 (S W F : ℝ) : Prop := S + W + F = 150

theorem solve_for_F (S W F : ℝ) (h1 : condition1 S W) (h2 : condition2 W F) (h3 : condition3 S W F) : F = 52.5 :=
sorry

end solve_for_F_l40_40473


namespace Luke_piles_of_quarters_l40_40505

theorem Luke_piles_of_quarters (Q : ℕ) (h : 6 * Q = 30) : Q = 5 :=
by
  sorry

end Luke_piles_of_quarters_l40_40505


namespace this_year_sales_l40_40397

def last_year_sales : ℝ := 320 -- in millions
def percent_increase : ℝ := 0.5 -- 50%

theorem this_year_sales : (last_year_sales * (1 + percent_increase)) = 480 := by
  sorry

end this_year_sales_l40_40397


namespace circle_center_radius_l40_40199

theorem circle_center_radius (x y : ℝ) :
  x^2 - 6*x + y^2 + 2*y - 9 = 0 ↔ (x-3)^2 + (y+1)^2 = 19 :=
sorry

end circle_center_radius_l40_40199


namespace regular_triangular_pyramid_volume_l40_40293

theorem regular_triangular_pyramid_volume (a γ : ℝ) : 
  ∃ V, V = (a^3 * Real.sin (γ / 2)^2) / (12 * Real.sqrt (1 - (Real.sin (γ / 2))^2)) := 
sorry

end regular_triangular_pyramid_volume_l40_40293


namespace parabola_relationship_l40_40775

theorem parabola_relationship (a : ℝ) (h : a < 0) :
  let y1 := 24 * a - 7
  let y2 := -7
  let y3 := 3 * a - 7
  y1 < y3 ∧ y3 < y2 :=
by
  let y1 := 24 * a - 7
  let y2 := -7
  let y3 := 3 * a - 7
  sorry

end parabola_relationship_l40_40775


namespace ice_cream_children_count_ice_cream_girls_count_l40_40664

-- Proof Problem for part (a)
theorem ice_cream_children_count (n : ℕ) (h : 3 * n = 24) : n = 8 := sorry

-- Proof Problem for part (b)
theorem ice_cream_girls_count (x y : ℕ) (h : x + y = 8) 
  (hx_even : x % 2 = 0) (hy_even : y % 2 = 0) (hx_pos : x > 0) (hxy : x < y) : y = 6 := sorry

end ice_cream_children_count_ice_cream_girls_count_l40_40664


namespace cost_jam_l40_40430

noncomputable def cost_of_jam (N B J : ℕ) : ℝ :=
  N * J * 5 / 100

theorem cost_jam (N B J : ℕ) (h₁ : N > 1) (h₂ : 4 * N + 20 = 414) :
  cost_of_jam N B J = 2.25 := by
  sorry

end cost_jam_l40_40430


namespace math_problem_l40_40202

theorem math_problem :
  (Int.ceil ((16 / 5 : ℚ) * (-34 / 4 : ℚ)) - Int.floor ((16 / 5 : ℚ) * Int.floor (-34 / 4 : ℚ))) = 2 :=
by
  sorry

end math_problem_l40_40202


namespace distinctKeyArrangements_l40_40381

-- Given conditions as definitions in Lean.
def houseNextToCar : Prop := sorry
def officeNextToBike : Prop := sorry
def noDifferenceByRotationOrReflection (arr1 arr2 : List ℕ) : Prop := sorry

-- Main statement to be proven
theorem distinctKeyArrangements : 
  houseNextToCar ∧ officeNextToBike ∧ (∀ (arr1 arr2 : List ℕ), noDifferenceByRotationOrReflection arr1 arr2 ↔ arr1 = arr2) 
  → ∃ n : ℕ, n = 16 :=
by sorry

end distinctKeyArrangements_l40_40381


namespace jam_consumption_l40_40380

theorem jam_consumption (x y t : ℝ) :
  x + y = 100 →
  t = 45 * x / y →
  t = 20 * y / x →
  x = 40 ∧ y = 60 ∧ 
  (y / 45 = 4 / 3) ∧ 
  (x / 20 = 2) := by
  sorry

end jam_consumption_l40_40380


namespace solve_complex_addition_l40_40996

noncomputable def complex_addition : Prop :=
  let i := Complex.I
  let z1 := 3 - 5 * i
  let z2 := -1 + 12 * i
  let result := 2 + 7 * i
  z1 + z2 = result

theorem solve_complex_addition :
  complex_addition :=
by
  sorry

end solve_complex_addition_l40_40996


namespace largest_consecutive_positive_elements_l40_40704

theorem largest_consecutive_positive_elements (a : ℕ → ℝ)
  (h₁ : ∀ n ≥ 2, a n = a (n-1) + a (n+2)) :
  ∃ m, m = 5 ∧ ∀ k < m, a k > 0 :=
sorry

end largest_consecutive_positive_elements_l40_40704


namespace price_increase_percentage_l40_40194

-- Define the problem conditions
def lowest_price := 12
def highest_price := 21

-- Formulate the goal as a theorem
theorem price_increase_percentage :
  ((highest_price - lowest_price) / lowest_price : ℚ) * 100 = 75 := by
  sorry

end price_increase_percentage_l40_40194


namespace volume_of_box_is_correct_l40_40819

def metallic_sheet_initial_length : ℕ := 48
def metallic_sheet_initial_width : ℕ := 36
def square_cut_side_length : ℕ := 8

def box_length : ℕ := metallic_sheet_initial_length - 2 * square_cut_side_length
def box_width : ℕ := metallic_sheet_initial_width - 2 * square_cut_side_length
def box_height : ℕ := square_cut_side_length

def box_volume : ℕ := box_length * box_width * box_height

theorem volume_of_box_is_correct : box_volume = 5120 := by
  sorry

end volume_of_box_is_correct_l40_40819


namespace find_abc_l40_40012

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

axiom condition1 : a * b = 45 * (3 : ℝ)^(1/3)
axiom condition2 : a * c = 75 * (3 : ℝ)^(1/3)
axiom condition3 : b * c = 30 * (3 : ℝ)^(1/3)

theorem find_abc : a * b * c = 75 * (2 : ℝ)^(1/2) := sorry

end find_abc_l40_40012


namespace three_numbers_lcm_ratio_l40_40576

theorem three_numbers_lcm_ratio
  (x : ℕ)
  (h1 : 3 * x.gcd 4 = 1)
  (h2 : (3 * x * 4 * x) / x.gcd (3 * x) = 180)
  (h3 : ∃ y : ℕ, y = 5 * (3 * x))
  : (3 * x = 45 ∧ 4 * x = 60 ∧ 5 * (3 * x) = 225) ∧
      lcm (lcm (3 * x) (4 * x)) (5 * (3 * x)) = 900 :=
by
  sorry

end three_numbers_lcm_ratio_l40_40576


namespace r_at_5_l40_40623

def r (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3) * (x - 4) + x^2 - 1

theorem r_at_5 :
  r 5 = 48 := by
  sorry

end r_at_5_l40_40623


namespace find_number_l40_40713

theorem find_number : ∀ (x : ℝ), (0.15 * 0.30 * 0.50 * x = 99) → (x = 4400) :=
by
  intro x
  intro h
  sorry

end find_number_l40_40713


namespace power_sum_eq_l40_40457

theorem power_sum_eq (n : ℕ) : (-2)^2009 + (-2)^2010 = 2^2009 := by
  sorry

end power_sum_eq_l40_40457


namespace a_n_divisible_by_11_l40_40861

-- Define the sequence
def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ 
  a 2 = 3 ∧ 
  ∀ n, a (n + 2) = (n + 3) * a (n + 1) - (n + 2) * a n

-- Main statement
theorem a_n_divisible_by_11 (a : ℕ → ℤ) (h : seq a) :
  ∀ n, ∃ k : ℕ, a n % 11 = 0 ↔ n = 4 + 11 * k :=
sorry

end a_n_divisible_by_11_l40_40861


namespace train_crossing_time_l40_40485

theorem train_crossing_time
  (train_length : ℝ)
  (platform_length : ℝ)
  (time_to_cross_platform : ℝ)
  (train_speed : ℝ := (train_length + platform_length) / time_to_cross_platform)
  (time_to_cross_signal_pole : ℝ := train_length / train_speed) :
  train_length = 300 ∧ platform_length = 1000 ∧ time_to_cross_platform = 39 → time_to_cross_signal_pole = 9 := by
  intro h
  cases h
  sorry

end train_crossing_time_l40_40485


namespace difference_of_numbers_l40_40853

theorem difference_of_numbers (x y : ℕ) (h1 : x + y = 64) (h2 : y = 26) : x - y = 12 :=
sorry

end difference_of_numbers_l40_40853


namespace edge_length_of_divided_cube_l40_40802

theorem edge_length_of_divided_cube (volume_original_cube : ℕ) (num_divisions : ℕ) (volume_of_one_smaller_cube : ℕ) (edge_length : ℕ) :
  volume_original_cube = 1000 →
  num_divisions = 8 →
  volume_of_one_smaller_cube = volume_original_cube / num_divisions →
  volume_of_one_smaller_cube = edge_length ^ 3 →
  edge_length = 5 :=
by
  sorry

end edge_length_of_divided_cube_l40_40802


namespace assignment_problem_l40_40969

theorem assignment_problem (a b c : ℕ) (h1 : a = 10) (h2 : b = 20) (h3 : c = 30) :
  let a := b
  let b := c
  let c := a
  a = 20 ∧ b = 30 ∧ c = 20 :=
by
  sorry

end assignment_problem_l40_40969


namespace ab_not_divisible_by_5_then_neither_divisible_l40_40688

theorem ab_not_divisible_by_5_then_neither_divisible (a b : ℕ) : ¬(¬(5 ∣ a) ∧ ¬(5 ∣ b)) → ¬(5 ∣ (a * b)) :=
by
  -- Mathematical statement for proof by contradiction:
  have H1: ¬(¬(5 ∣ a) ∧ ¬(5 ∣ b)) := sorry
  -- Rest of the proof would go here  
  sorry

end ab_not_divisible_by_5_then_neither_divisible_l40_40688


namespace mitch_earns_correctly_l40_40248

noncomputable def mitch_weekly_earnings : ℝ :=
  let earnings_mw := 3 * (3 * 5 : ℝ) -- Monday to Wednesday
  let earnings_tf := 2 * (6 * 4 : ℝ) -- Thursday and Friday
  let earnings_sat := 4 * 6         -- Saturday
  let earnings_sun := 5 * 8         -- Sunday
  let total_earnings := earnings_mw + earnings_tf + earnings_sat + earnings_sun
  let after_expenses := total_earnings - 25
  let after_tax := after_expenses - 0.10 * after_expenses
  after_tax

theorem mitch_earns_correctly : mitch_weekly_earnings = 118.80 := by
  sorry

end mitch_earns_correctly_l40_40248


namespace sum_of_a_and_b_l40_40797

theorem sum_of_a_and_b {a b : ℝ} (h : a^2 + b^2 + (a*b)^2 = 4*a*b - 1) : a + b = 2 ∨ a + b = -2 :=
sorry

end sum_of_a_and_b_l40_40797


namespace kamal_marks_physics_correct_l40_40401

-- Definition of the conditions
def kamal_marks_english : ℕ := 76
def kamal_marks_mathematics : ℕ := 60
def kamal_marks_chemistry : ℕ := 67
def kamal_marks_biology : ℕ := 85
def kamal_average_marks : ℕ := 74
def kamal_num_subjects : ℕ := 5

-- Definition of the total marks
def kamal_total_marks : ℕ := kamal_average_marks * kamal_num_subjects

-- Sum of known marks
def kamal_known_marks : ℕ := kamal_marks_english + kamal_marks_mathematics + kamal_marks_chemistry + kamal_marks_biology

-- The expected result for Physics
def kamal_marks_physics : ℕ := 82

-- Proof statement
theorem kamal_marks_physics_correct :
  kamal_total_marks - kamal_known_marks = kamal_marks_physics :=
by
  simp [kamal_total_marks, kamal_known_marks, kamal_marks_physics]
  sorry

end kamal_marks_physics_correct_l40_40401


namespace trigonometric_identity_proof_l40_40820

theorem trigonometric_identity_proof 
  (α : ℝ) 
  (h1 : Real.tan (2 * α) = 3 / 4) 
  (h2 : α ∈ Set.Ioo (-(Real.pi / 2)) (Real.pi / 2))
  (h3 : ∃ x : ℝ, (Real.sin (x + 2) + Real.sin (α - x) - 2 * Real.sin α) = 0) : 
  Real.cos (2 * α) = -4 / 5 ∧ Real.tan (α / 2) = (1 - Real.sqrt 10) / 3 := 
sorry

end trigonometric_identity_proof_l40_40820


namespace find_integer_l40_40940

noncomputable def least_possible_sum (x y z k : ℕ) : Prop :=
  2 * x = 5 * y ∧ 5 * y = 6 * z ∧ x + k + z = 26

theorem find_integer (x y z : ℕ) (h : least_possible_sum x y z 6) :
  6 = (26 - x - z) :=
  by {
    sorry
  }

end find_integer_l40_40940


namespace cos_angles_difference_cos_angles_sum_l40_40068

-- Part (a)
theorem cos_angles_difference: 
  (Real.cos (36 * Real.pi / 180) - Real.cos (72 * Real.pi / 180) = 1 / 2) :=
sorry

-- Part (b)
theorem cos_angles_sum: 
  (Real.cos (Real.pi / 7) - Real.cos (2 * Real.pi / 7) + Real.cos (3 * Real.pi / 7) = 1 / 2) :=
sorry

end cos_angles_difference_cos_angles_sum_l40_40068


namespace max_volume_of_prism_l40_40235

theorem max_volume_of_prism (a b c s : ℝ) (h : a + b + c = 3 * s) : a * b * c ≤ s^3 :=
by {
    -- placeholder for the proof
    sorry
}

end max_volume_of_prism_l40_40235


namespace subtract_largest_unit_fraction_l40_40723

theorem subtract_largest_unit_fraction
  (a b n : ℕ) (ha : a > 0) (hb : b > a) (hn : 1 ≤ b * n ∧ b * n <= a * n + b): 
  (a * n - b < a) := by
  sorry

end subtract_largest_unit_fraction_l40_40723


namespace cauliflower_sales_l40_40410

noncomputable def broccoli_sales : ℝ := 57
noncomputable def carrot_sales : ℝ := 2 * broccoli_sales
noncomputable def spinach_sales : ℝ := 16 + (1 / 2 * carrot_sales)
noncomputable def total_sales : ℝ := 380
noncomputable def other_sales : ℝ := broccoli_sales + carrot_sales + spinach_sales

theorem cauliflower_sales :
  total_sales - other_sales = 136 :=
by
  -- proof skipped
  sorry

end cauliflower_sales_l40_40410


namespace green_hats_count_l40_40975

theorem green_hats_count : ∃ G B : ℕ, B + G = 85 ∧ 6 * B + 7 * G = 540 ∧ G = 30 :=
by
  sorry

end green_hats_count_l40_40975


namespace circle_radius_five_iff_l40_40749

noncomputable def circle_eq_radius (x y : ℝ) (k : ℝ) : Prop :=
  x^2 + 8*x + y^2 + 4*y - k = 0

def is_circle_with_radius (r : ℝ) (x y : ℝ) (k : ℝ) : Prop :=
  circle_eq_radius x y k ↔ r = 5 ∧ k = 5

theorem circle_radius_five_iff (k : ℝ) :
  (∃ x y : ℝ, circle_eq_radius x y k) ↔ k = 5 :=
sorry

end circle_radius_five_iff_l40_40749


namespace naomi_regular_bikes_l40_40786
-- Import necessary libraries

-- Define the condition and the proof problem
theorem naomi_regular_bikes (R C : ℕ) (h1 : C = 11) 
  (h2 : 2 * R + 4 * C = 58) : R = 7 := 
  by 
  -- Include all necessary conditions as assumptions
  have hC : C = 11 := h1
  have htotal : 2 * R + 4 * C = 58 := h2
  -- Skip the proof itself
  sorry

end naomi_regular_bikes_l40_40786


namespace symmetric_point_about_origin_l40_40833

theorem symmetric_point_about_origin (P Q : ℤ × ℤ) (h : P = (-2, -3)) : Q = (2, 3) :=
by
  sorry

end symmetric_point_about_origin_l40_40833


namespace k_valid_iff_l40_40558

open Nat

theorem k_valid_iff (k : ℕ) :
  (∃ m n : ℕ, m * (m + k) = n * (n + 1)) ↔ k ≠ 2 ∧ k ≠ 3 :=
by
  sorry

end k_valid_iff_l40_40558


namespace bicycle_speed_B_l40_40864

theorem bicycle_speed_B (v_A v_B : ℝ) (d : ℝ) (h1 : d = 12) (h2 : v_A = 1.2 * v_B) (h3 : d / v_B - d / v_A = 1 / 6) : v_B = 12 :=
by
  sorry

end bicycle_speed_B_l40_40864


namespace problem_statement_l40_40751

def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}
def T : Set ℝ := {x | x < 2}

def set_otimes (A B : Set ℝ) : Set ℝ := {x | x ∈ (A ∪ B) ∧ x ∉ (A ∩ B)}

theorem problem_statement : set_otimes M T = {x | x < -1 ∨ (2 ≤ x ∧ x ≤ 4)} :=
by sorry

end problem_statement_l40_40751


namespace calculate_expression_l40_40935

theorem calculate_expression :
  (3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3)) = (3 + 2 * Real.sqrt 3 / 3) :=
by
  sorry

end calculate_expression_l40_40935


namespace max_f_angle_A_of_triangle_l40_40631

noncomputable def f (x : ℝ) : ℝ := (Real.cos (2 * x - 4 * Real.pi / 3)) + 2 * (Real.cos x)^2

theorem max_f : ∃ x : ℝ, f x = 2 := sorry

theorem angle_A_of_triangle (A B C : ℝ) (h : A + B + C = Real.pi)
  (h2 : f (B + C) = 3 / 2) : A = Real.pi / 3 := sorry

end max_f_angle_A_of_triangle_l40_40631


namespace sum_of_cubes_of_consecutive_even_integers_l40_40698

theorem sum_of_cubes_of_consecutive_even_integers 
    (x y z : ℕ) 
    (h1 : x % 2 = 0) 
    (h2 : y % 2 = 0) 
    (h3 : z % 2 = 0) 
    (h4 : y = x + 2) 
    (h5 : z = y + 2) 
    (h6 : x * y * z = 12 * (x + y + z)) : 
  x^3 + y^3 + z^3 = 8568 := 
by
  -- Proof goes here
  sorry

end sum_of_cubes_of_consecutive_even_integers_l40_40698


namespace alice_current_age_l40_40851

def alice_age_twice_eve (a b : Nat) : Prop := a = 2 * b

def eve_age_after_10_years (a b : Nat) : Prop := a = b + 10

theorem alice_current_age (a b : Nat) (h1 : alice_age_twice_eve a b) (h2 : eve_age_after_10_years a b) : a = 20 := by
  sorry

end alice_current_age_l40_40851


namespace seashells_ratio_l40_40487

theorem seashells_ratio (s_1 s_2 S t s3 : ℕ) (hs1 : s_1 = 5) (hs2 : s_2 = 7) (hS : S = 36)
  (ht : t = s_1 + s_2) (hs3 : s3 = S - t) :
  s3 / t = 2 :=
by
  rw [hs1, hs2] at ht
  simp at ht
  rw [hS, ht] at hs3
  simp at hs3
  sorry

end seashells_ratio_l40_40487


namespace dot_product_AB_BC_l40_40924

theorem dot_product_AB_BC 
  (a b c : ℝ) 
  (h1 : b^2 = a * c)
  (h2 : a + c = 3)
  (cosB : ℝ)
  (h3 : cosB = 3 / 4) : 
  (a * c * (-cosB) = -3/2) :=
by 
  -- Given conditions
  sorry

end dot_product_AB_BC_l40_40924


namespace circle_center_l40_40917

theorem circle_center {x y : ℝ} :
  4 * x^2 - 8 * x + 4 * y^2 - 16 * y + 20 = 0 → (x, y) = (1, 2) :=
by
  sorry

end circle_center_l40_40917


namespace min_value_func_y_l40_40211

noncomputable def geometric_sum (t : ℝ) (n : ℕ) : ℝ :=
  t * 3^(n-1) - (1 / 3)

noncomputable def func_y (x t : ℝ) : ℝ :=
  (x + 2) * (x + 10) / (x + t)

theorem min_value_func_y :
  ∀ (t : ℝ), (∀ n : ℕ, geometric_sum t n = (1) → (∀ x > 0, func_y x t ≥ 16)) :=
  sorry

end min_value_func_y_l40_40211


namespace simplify_expr_l40_40784

def expr (y : ℝ) := y * (4 * y^2 - 3) - 6 * (y^2 - 3 * y + 8)

theorem simplify_expr (y : ℝ) : expr y = 4 * y^3 - 6 * y^2 + 15 * y - 48 :=
by
  sorry

end simplify_expr_l40_40784


namespace teamA_fraction_and_sum_l40_40785

def time_to_minutes (t : ℝ) : ℝ := t * 60

def fraction_teamA_worked (m n : ℕ) (h_coprime : Nat.gcd m n = 1) (h_fraction : m = 1 ∧ n = 5) : Prop :=
  (90 - 60) / 150 = m / n

theorem teamA_fraction_and_sum (m n : ℕ) (h_coprime : Nat.gcd m n = 1) (h_fraction : m = 1 ∧ n = 5) :
  90 / 150 = 1 / 5 → m + n = 6 :=
by
  sorry

end teamA_fraction_and_sum_l40_40785


namespace find_f_2_l40_40005

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_2 (h1 : ∀ x1 x2 : ℝ, f (x1 * x2) = f x1 + f x2) (h2 : f 8 = 3) : f 2 = 1 :=
by
  sorry

end find_f_2_l40_40005


namespace unique_y_for_star_eq_9_l40_40403

def star (x y : ℝ) : ℝ := 3 * x - 2 * y + x^2 * y

theorem unique_y_for_star_eq_9 : ∃! y : ℝ, star 2 y = 9 := by
  sorry

end unique_y_for_star_eq_9_l40_40403


namespace problem_statement_l40_40934

-- Define the necessary and sufficient conditions
def necessary_but_not_sufficient (P Q : Prop) : Prop :=
  (Q → P) ∧ (¬ (P → Q))

-- Specific propositions in this scenario
def x_conditions (x : ℝ) : Prop := x^2 - 2 * x - 3 = 0
def x_equals_3 (x : ℝ) : Prop := x = 3

-- Prove the given problem statement
theorem problem_statement (x : ℝ) : necessary_but_not_sufficient (x_conditions x) (x_equals_3 x) :=
  sorry

end problem_statement_l40_40934


namespace profit_percent_l40_40183

theorem profit_percent (P C : ℝ) (h : (2 / 3) * P = 0.88 * C) : P - C = 0.32 * C → (P - C) / C * 100 = 32 := by
  sorry

end profit_percent_l40_40183


namespace inequality_ABC_l40_40373

theorem inequality_ABC (x y z : ℝ) : 
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) + 
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) + 
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
sorry

end inequality_ABC_l40_40373


namespace range_of_a_l40_40222

noncomputable def p (x : ℝ) : Prop := abs (3 * x - 4) > 2
noncomputable def q (x : ℝ) : Prop := 1 / (x^2 - x - 2) > 0
noncomputable def r (x a : ℝ) : Prop := (x - a) * (x - a - 1) < 0

theorem range_of_a {a : ℝ} :
  (∀ x : ℝ, ¬ r x a → ¬ p x) → (a ≥ 2 ∨ a ≤ -1/3) :=
sorry

end range_of_a_l40_40222


namespace intersection_of_sets_l40_40794

theorem intersection_of_sets (A B : Set ℕ) (hA : A = {0, 1, 2}) (hB : B = {1, 2, 3, 4}) :
  A ∩ B = {1, 2} :=
by
  sorry

end intersection_of_sets_l40_40794


namespace compound_interest_second_year_l40_40332

theorem compound_interest_second_year
  (P: ℝ) (r: ℝ) (CI_3 : ℝ) (CI_2 : ℝ)
  (h1 : r = 0.06)
  (h2 : CI_3 = 1272)
  (h3 : CI_3 = CI_2 * (1 + r)) :
  CI_2 = 1200 :=
by
  sorry

end compound_interest_second_year_l40_40332


namespace length_of_AE_l40_40106

theorem length_of_AE (AF CE ED : ℝ) (ABCD_area : ℝ) (hAF : AF = 30) (hCE : CE = 40) (hED : ED = 50) (hABCD_area : ABCD_area = 7200) : ∃ AE : ℝ, AE = 322.5 := sorry

end length_of_AE_l40_40106


namespace cube_side_length_l40_40703

theorem cube_side_length (n : ℕ) (h : (6 * n^2) / (6 * n^3) = 1 / 3) : n = 3 :=
sorry

end cube_side_length_l40_40703


namespace intersection_complement_A_B_l40_40930

open Set

theorem intersection_complement_A_B :
  let A := {x : ℝ | x + 1 > 0}
  let B := {-2, -1, 0, 1}
  (compl A ∩ B : Set ℝ) = {-2, -1} :=
by
  sorry

end intersection_complement_A_B_l40_40930


namespace determine_right_triangle_l40_40978

-- Definitions based on conditions
def condition_A (A B C : ℝ) : Prop := A^2 + B^2 = C^2
def condition_B (A B C : ℝ) : Prop := A^2 - B^2 = C^2
def condition_C (A B C : ℝ) : Prop := A + B = C
def condition_D (A B C : ℝ) : Prop := A / B = 3 / 4 ∧ B / C = 4 / 5

-- Problem statement: D cannot determine that triangle ABC is a right triangle
theorem determine_right_triangle (A B C : ℝ) : ¬ condition_D A B C :=
by sorry

end determine_right_triangle_l40_40978


namespace ounces_per_gallon_l40_40090

-- conditions
def gallons_of_milk (james : Type) : ℕ := 3
def ounces_drank (james : Type) : ℕ := 13
def ounces_left (james : Type) : ℕ := 371

-- question
def ounces_in_gallon (james : Type) : ℕ := 128

-- proof statement
theorem ounces_per_gallon (james : Type) :
  (gallons_of_milk james) * (ounces_in_gallon james) = (ounces_left james + ounces_drank james) :=
sorry

end ounces_per_gallon_l40_40090


namespace last_digit_2019_digit_number_l40_40655

theorem last_digit_2019_digit_number :
  ∃ n : ℕ → ℕ,  
    (∀ k, 0 ≤ k → k < 2018 → (n k * 10 + n (k + 1)) % 13 = 0) ∧ 
    n 0 = 6 ∧ 
    n 2018 = 2 :=
sorry

end last_digit_2019_digit_number_l40_40655


namespace appropriate_sampling_method_l40_40385

-- Defining the sizes of the boxes
def size_large : ℕ := 120
def size_medium : ℕ := 60
def size_small : ℕ := 20

-- Define a sample size
def sample_size : ℕ := 25

-- Define the concept of appropriate sampling method as being equivalent to stratified sampling in this context
theorem appropriate_sampling_method : 3 > 0 → sample_size > 0 → size_large = 120 ∧ size_medium = 60 ∧ size_small = 20 → 
("stratified sampling" = "stratified sampling") :=
by 
  sorry

end appropriate_sampling_method_l40_40385


namespace volume_surface_ratio_l40_40825

-- Define the structure of the shape
structure Shape where
  center_cube : unit
  surrounding_cubes : Fin 6 -> unit
  top_cube : unit

-- Define the properties for the calculation
def volume (s : Shape) : ℕ := 8
def surface_area (s : Shape) : ℕ := 28
def ratio_volume_surface_area (s : Shape) : ℚ := volume s / surface_area s

-- Main theorem statement
theorem volume_surface_ratio (s : Shape) : ratio_volume_surface_area s = 2 / 7 := sorry

end volume_surface_ratio_l40_40825


namespace problem_l40_40772

noncomputable def a := Real.log 2 / Real.log 3
noncomputable def b := Real.log (1/8) / Real.log 2
noncomputable def c := Real.sqrt 2

theorem problem : c > a ∧ a > b := 
by
  sorry

end problem_l40_40772


namespace race_distance_l40_40958

theorem race_distance (D : ℝ)
  (A_speed : ℝ := D / 20)
  (B_speed : ℝ := D / 25)
  (A_beats_B_by : ℝ := 18)
  (h1 : A_speed * 25 = D + A_beats_B_by)
  : D = 72 := 
by
  sorry

end race_distance_l40_40958


namespace conditions_iff_positive_l40_40390

theorem conditions_iff_positive (a b : ℝ) (h₁ : a + b > 0) (h₂ : ab > 0) : 
  (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ ab > 0) :=
sorry

end conditions_iff_positive_l40_40390


namespace additional_savings_l40_40025

def initial_price : Float := 30
def discount1 : Float := 5
def discount2_percent : Float := 0.25

def price_after_discount1_then_discount2 : Float := 
  (initial_price - discount1) * (1 - discount2_percent)

def price_after_discount2_then_discount1 : Float := 
  initial_price * (1 - discount2_percent) - discount1

theorem additional_savings :
  price_after_discount1_then_discount2 - price_after_discount2_then_discount1 = 1.25 := by
  sorry

end additional_savings_l40_40025


namespace no_int_a_divisible_289_l40_40433

theorem no_int_a_divisible_289 : ¬ ∃ a : ℤ, ∃ k : ℤ, a^2 - 3 * a - 19 = 289 * k :=
by
  sorry

end no_int_a_divisible_289_l40_40433


namespace probability_at_least_three_aces_l40_40131

open Nat

noncomputable def combination (n k : ℕ) : ℕ :=
  n.choose k

theorem probability_at_least_three_aces :
  (combination 4 3 * combination 48 2 + combination 4 4 * combination 48 1) / combination 52 5 = (combination 4 3 * combination 48 2 + combination 4 4 * combination 48 1 : ℚ) / combination 52 5 :=
by
  sorry

end probability_at_least_three_aces_l40_40131


namespace container_capacity_l40_40580

theorem container_capacity (C : ℝ) (h1 : 0.30 * C + 36 = 0.75 * C) : C = 80 :=
by
  sorry

end container_capacity_l40_40580


namespace B_monthly_income_is_correct_l40_40855

variable (A_m B_m C_m : ℝ)
variable (A_annual C_m_value : ℝ)
variable (ratio_A_to_B : ℝ)

-- Given conditions
def conditions :=
  A_annual = 537600 ∧
  C_m_value = 16000 ∧
  ratio_A_to_B = 5 / 2 ∧
  A_m = A_annual / 12 ∧
  B_m = (2 / 5) * A_m ∧
  B_m = 1.12 * C_m ∧
  C_m = C_m_value

-- Prove that B's monthly income is Rs. 17920
theorem B_monthly_income_is_correct (h : conditions A_m B_m C_m A_annual C_m_value ratio_A_to_B) : 
  B_m = 17920 :=
by 
  sorry

end B_monthly_income_is_correct_l40_40855


namespace expected_value_of_difference_l40_40756

noncomputable def expected_difference (num_days : ℕ) : ℝ :=
  let p_prime := 3 / 4
  let p_composite := 1 / 4
  let p_no_reroll := 2 / 3
  let expected_unsweetened_days := p_prime * p_no_reroll * num_days
  let expected_sweetened_days := p_composite * p_no_reroll * num_days
  expected_unsweetened_days - expected_sweetened_days

theorem expected_value_of_difference :
  expected_difference 365 = 121.667 := by
  sorry

end expected_value_of_difference_l40_40756


namespace correct_factorization_from_left_to_right_l40_40412

theorem correct_factorization_from_left_to_right 
  (x a b c m n : ℝ) : 
  (2 * a * b - 2 * a * c = 2 * a * (b - c)) :=
sorry

end correct_factorization_from_left_to_right_l40_40412


namespace quadratic_inequality_solution_l40_40765

theorem quadratic_inequality_solution (a b : ℝ)
  (h : ∀ x : ℝ, -1/2 < x ∧ x < 1/3 → ax^2 + bx + 2 > 0) :
  a + b = -14 :=
sorry

end quadratic_inequality_solution_l40_40765


namespace remove_terms_yield_desired_sum_l40_40113

-- Define the original sum and the terms to be removed
def originalSum : ℚ := 1/3 + 1/6 + 1/9 + 1/12 + 1/15 + 1/18
def termsToRemove : List ℚ := [1/9, 1/12, 1/15, 1/18]

-- Definition of the desired remaining sum
def desiredSum : ℚ := 1/2

noncomputable def sumRemainingTerms : ℚ :=
originalSum - List.sum termsToRemove

-- Lean theorem to prove
theorem remove_terms_yield_desired_sum : sumRemainingTerms = desiredSum :=
by 
  sorry

end remove_terms_yield_desired_sum_l40_40113


namespace minimum_sum_dimensions_l40_40755

def is_product (a b c : ℕ) (v : ℕ) : Prop :=
  a * b * c = v

def sum (a b c : ℕ) : ℕ :=
  a + b + c

theorem minimum_sum_dimensions : 
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ is_product a b c 3003 ∧ sum a b c = 45 :=
by
  sorry

end minimum_sum_dimensions_l40_40755


namespace solution_existence_l40_40657

theorem solution_existence (m : ℤ) :
  (∀ x y : ℤ, 2 * x + (m - 1) * y = 3 ∧ (m + 1) * x + 4 * y = -3) ↔
  (m = -3 ∨ m = 3 → 
    (m = -3 → ∃ x y : ℤ, 2 * x + (m - 1) * y = 3 ∧ (m + 1) * x + 4 * y = -3) ∧
    (m = 3 → ¬∃ x y : ℤ, 2 * x + (m - 1) * y = 3 ∧ (m + 1) * x + 4 * y = -3)) := by
  sorry

end solution_existence_l40_40657


namespace problem_proof_l40_40680

variable {a1 a2 b1 b2 b3 : ℝ}

theorem problem_proof 
  (h1 : ∃ d, -7 + d = a1 ∧ a1 + d = a2 ∧ a2 + d = -1)
  (h2 : ∃ r, -4 * r = b1 ∧ b1 * r = b2 ∧ b2 * r = b3 ∧ b3 * r = -1)
  (ha : a2 - a1 = 2)
  (hb : b2 = -2) :
  (a2 - a1) / b2 = -1 :=
by
  sorry

end problem_proof_l40_40680


namespace function_is_odd_and_increasing_l40_40077

-- Define the function y = x^(3/5)
def f (x : ℝ) : ℝ := x ^ (3 / 5)

-- Define what it means for the function to be odd
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define what it means for the function to be increasing in its domain
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- The proposition to prove
theorem function_is_odd_and_increasing :
  is_odd f ∧ is_increasing f :=
by
  sorry

end function_is_odd_and_increasing_l40_40077


namespace derivative_of_f_l40_40496

noncomputable def f (x : ℝ) : ℝ := x * Real.cos x - Real.sin x

theorem derivative_of_f : ∀ x : ℝ, deriv f x = -x * Real.sin x := by
  sorry

end derivative_of_f_l40_40496


namespace handshake_problem_l40_40001

theorem handshake_problem :
  let companies := 5
  let representatives_per_company := 5
  let total_people := companies * representatives_per_company
  let possible_handshakes_per_person := total_people - representatives_per_company - 1
  let total_possible_handshakes := total_people * possible_handshakes_per_person
  let unique_handshakes := total_possible_handshakes / 2
  unique_handshakes = 250 :=
by 
  let companies := 5
  let representatives_per_company := 5
  let total_people := companies * representatives_per_company
  let possible_handshakes_per_person := total_people - representatives_per_company - 1
  let total_possible_handshakes := total_people * possible_handshakes_per_person
  let unique_handshakes := total_possible_handshakes / 2
  sorry

end handshake_problem_l40_40001


namespace jason_initial_cards_l40_40562

theorem jason_initial_cards (cards_sold : Nat) (cards_after_selling : Nat) (initial_cards : Nat) 
  (h1 : cards_sold = 224) 
  (h2 : cards_after_selling = 452) 
  (h3 : initial_cards = cards_after_selling + cards_sold) : 
  initial_cards = 676 := 
sorry

end jason_initial_cards_l40_40562


namespace percent_of_x_is_z_l40_40515

def condition1 (z y : ℝ) : Prop := 0.45 * z = 0.72 * y
def condition2 (y x : ℝ) : Prop := y = 0.75 * x
def condition3 (w z : ℝ) : Prop := w = 0.60 * z^2
def condition4 (z w : ℝ) : Prop := z = 0.30 * w^(1/3)

theorem percent_of_x_is_z (x y z w : ℝ) 
  (h1 : condition1 z y) 
  (h2 : condition2 y x)
  (h3 : condition3 w z)
  (h4 : condition4 z w) : 
  z / x = 1.2 :=
sorry

end percent_of_x_is_z_l40_40515


namespace max_C_usage_l40_40729

-- Definition of variables (concentration percentages and weights)
def A_conc := 3 / 100
def B_conc := 8 / 100
def C_conc := 11 / 100

def target_conc := 7 / 100
def total_weight := 100

def max_A := 50
def max_B := 70
def max_C := 60

-- Equation to satisfy
def conc_equation (x y : ℝ) : Prop :=
  C_conc * x + B_conc * y + A_conc * (total_weight - x - y) = target_conc * total_weight

-- Definition with given constraints
def within_constraints (x y : ℝ) : Prop :=
  x ≤ max_C ∧ y ≤ max_B ∧ (total_weight - x - y) ≤ max_A

-- The theorem that needs to be proved
theorem max_C_usage (x y : ℝ) : within_constraints x y ∧ conc_equation x y → x ≤ 50 :=
by
  sorry

end max_C_usage_l40_40729


namespace fraction_equals_repeating_decimal_l40_40182

noncomputable def repeating_decimal_fraction : ℚ :=
  let a : ℚ := 46 / 100
  let r : ℚ := 1 / 100
  (a / (1 - r))

theorem fraction_equals_repeating_decimal :
  repeating_decimal_fraction = 46 / 99 :=
by
  sorry

end fraction_equals_repeating_decimal_l40_40182


namespace domain_of_f_l40_40164

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x - 1)

theorem domain_of_f :
  { x : ℝ | 2 * x - 1 > 0 } = { x : ℝ | x > 1 / 2 } :=
by
  sorry

end domain_of_f_l40_40164


namespace externally_tangent_intersect_two_points_l40_40565

def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0
def circle2 (x y r : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = r^2 ∧ r > 0

theorem externally_tangent (r : ℝ) : 
  (∃ x y : ℝ, circle2 x y r) →
  (∃ x y : ℝ, circle1 x y) → 
  (dist (1, 1) (4, 5) = r + 1) → 
  r = 4 := 
sorry

theorem intersect_two_points (r : ℝ) : 
  (∃ x y : ℝ, circle2 x y r) → 
  (∃ x y : ℝ, circle1 x y) → 
  (|r - 1| < dist (1, 1) (4, 5) ∧ dist (1, 1) (4, 5) < r + 1) → 
  4 < r ∧ r < 6 :=
sorry

end externally_tangent_intersect_two_points_l40_40565


namespace stormi_needs_more_money_l40_40674

theorem stormi_needs_more_money
  (cars_washed : ℕ) (price_per_car : ℕ)
  (lawns_mowed : ℕ) (price_per_lawn : ℕ)
  (bike_cost : ℕ)
  (h1 : cars_washed = 3)
  (h2 : price_per_car = 10)
  (h3 : lawns_mowed = 2)
  (h4 : price_per_lawn = 13)
  (h5 : bike_cost = 80) : 
  bike_cost - (cars_washed * price_per_car + lawns_mowed * price_per_lawn) = 24 := by
  sorry

end stormi_needs_more_money_l40_40674


namespace arithmetic_sequence_difference_l40_40439

theorem arithmetic_sequence_difference (a d : ℕ) (n m : ℕ) (hnm : m > n) (h_a : a = 3) (h_d : d = 7) (h_n : n = 1001) (h_m : m = 1004) :
  (a + (m - 1) * d) - (a + (n - 1) * d) = 21 :=
by
  sorry

end arithmetic_sequence_difference_l40_40439


namespace range_of_a_l40_40024

def my_Op (a b : ℝ) : ℝ := a - 2 * b

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, my_Op x 3 > 0 → my_Op x a > a) ↔ (∀ x : ℝ, x > 6 → x > 3 * a) → a ≤ 2 :=
by sorry

end range_of_a_l40_40024


namespace expression_remainder_l40_40378

theorem expression_remainder (n : ℤ) (h : n % 5 = 3) : (n + 1) % 5 = 4 :=
by
  sorry

end expression_remainder_l40_40378


namespace f_eq_zero_range_x_l40_40330

-- Definition of the function f on domain ℝ*
def f (x : ℝ) : ℝ := sorry

-- Conditions
axiom f_domain : ∀ x : ℝ, x ≠ 0 → f x = f x
axiom f_4 : f 4 = 1
axiom f_multiplicative : ∀ x1 x2 : ℝ, x1 ≠ 0 → x2 ≠ 0 → f (x1 * x2) = f x1 + f x2
axiom f_increasing : ∀ x y : ℝ, x < y → f x < f y

-- Problem (1): Prove f(1) = 0
theorem f_eq_zero : f 1 = 0 :=
sorry

-- Problem (2): Prove range 3 < x ≤ 5 given the inequality condition
theorem range_x (x : ℝ) : f (3 * x + 1) + f (2 * x - 6) ≤ 3 → 3 < x ∧ x ≤ 5 :=
sorry

end f_eq_zero_range_x_l40_40330


namespace total_spokes_in_garage_l40_40049

theorem total_spokes_in_garage :
  let bicycle1_spokes := 12 + 10
  let bicycle2_spokes := 14 + 12
  let bicycle3_spokes := 10 + 14
  let tricycle_spokes := 14 + 12 + 16
  bicycle1_spokes + bicycle2_spokes + bicycle3_spokes + tricycle_spokes = 114 :=
by
  let bicycle1_spokes := 12 + 10
  let bicycle2_spokes := 14 + 12
  let bicycle3_spokes := 10 + 14
  let tricycle_spokes := 14 + 12 + 16
  show bicycle1_spokes + bicycle2_spokes + bicycle3_spokes + tricycle_spokes = 114
  sorry

end total_spokes_in_garage_l40_40049


namespace gel_pen_price_relation_b_l40_40105

variable (x y b g T : ℝ)

def actual_amount_paid : ℝ := x * b + y * g

axiom gel_pen_cost_condition : (x + y) * g = 4 * actual_amount_paid x y b g
axiom ballpoint_pen_cost_condition : (x + y) * b = (1/2) * actual_amount_paid x y b g

theorem gel_pen_price_relation_b :
   (∀ x y b g : ℝ, (actual_amount_paid x y b g = x * b + y * g) 
    ∧ ((x + y) * g = 4 * actual_amount_paid x y b g)
    ∧ ((x + y) * b = (1/2) * actual_amount_paid x y b g))
    → g = 8 * b := 
sorry

end gel_pen_price_relation_b_l40_40105


namespace sqrt_of_sqrt_81_l40_40069

theorem sqrt_of_sqrt_81 : Real.sqrt 81 = 9 := 
  by
  sorry

end sqrt_of_sqrt_81_l40_40069


namespace solve_inequality_l40_40568

theorem solve_inequality (a : ℝ) :
  (a = 0 → {x : ℝ | x ≤ -1} = {x : ℝ | ax^2 - 2 ≥ 2 * x - a * x}) ∧
  (a > 0 → {x : ℝ | x ≥ 2 / a} ∪ {x : ℝ | x ≤ -1} = {x : ℝ | ax^2 - 2 ≥ 2 * x - a * x}) ∧
  (-2 < a ∧ a < 0 → {x : ℝ | 2 / a ≤ x ∧ x ≤ -1} = {x : ℝ | ax^2 - 2 ≥ 2 * x - a * x}) ∧
  (a = -2 → {x : ℝ | x = -1} = {x : ℝ | ax^2 - 2 ≥ 2 * x - a * x}) ∧
  (a < -2 → {x : ℝ | -1 ≤ x ∧ x ≤ 2 / a} = {x : ℝ | ax^2 - 2 ≥ 2 * x - a * x}) :=
by 
  sorry

end solve_inequality_l40_40568


namespace number_of_divisible_permutations_l40_40831

def digit_list := [1, 3, 1, 1, 5, 2, 1, 5, 2]
def count_permutations (d : List ℕ) (n : ℕ) : ℕ :=
  let fact := Nat.factorial
  let number := fact 8 / (fact 3 * fact 2 * fact 1)
  number

theorem number_of_divisible_permutations : count_permutations digit_list 2 = 3360 := by
  sorry

end number_of_divisible_permutations_l40_40831


namespace gcd_204_85_l40_40162

theorem gcd_204_85: Nat.gcd 204 85 = 17 := by
  sorry

end gcd_204_85_l40_40162


namespace g_minus_one_eq_zero_l40_40480

def g (x r : ℝ) : ℝ := 3 * x^3 - 2 * x^2 + 4 * x - 5 + r

theorem g_minus_one_eq_zero (r : ℝ) : g (-1) r = 0 → r = 14 := by
  sorry

end g_minus_one_eq_zero_l40_40480


namespace line_through_intersection_parallel_to_y_axis_l40_40611

theorem line_through_intersection_parallel_to_y_axis:
  ∃ x, (∃ y, 3 * x + 2 * y - 5 = 0 ∧ x - 3 * y + 2 = 0) ∧
       (x = 1) :=
sorry

end line_through_intersection_parallel_to_y_axis_l40_40611


namespace range_of_m_l40_40020

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ 4) → 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ |m - 1|) → 
  -3 ≤ m ∧ m ≤ 5 := 
by 
  sorry

end range_of_m_l40_40020


namespace derivative_at_1_l40_40008

def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem derivative_at_1 : deriv f 1 = 0 :=
by
  -- Proof to be provided
  sorry

end derivative_at_1_l40_40008


namespace nails_remaining_proof_l40_40091

noncomputable
def remaining_nails (initial_nails kitchen_percent fence_percent : ℕ) : ℕ :=
  let kitchen_used := initial_nails * kitchen_percent / 100
  let remaining_after_kitchen := initial_nails - kitchen_used
  let fence_used := remaining_after_kitchen * fence_percent / 100
  let final_remaining := remaining_after_kitchen - fence_used
  final_remaining

theorem nails_remaining_proof :
  remaining_nails 400 30 70 = 84 := by
  sorry

end nails_remaining_proof_l40_40091


namespace megan_folders_count_l40_40050

theorem megan_folders_count (init_files deleted_files files_per_folder : ℕ) (h₁ : init_files = 93) (h₂ : deleted_files = 21) (h₃ : files_per_folder = 8) :
  (init_files - deleted_files) / files_per_folder = 9 :=
by
  sorry

end megan_folders_count_l40_40050


namespace dragon_legs_l40_40363

variable {x y n : ℤ}

theorem dragon_legs :
  (x = 40) ∧
  (y = 9) ∧
  (220 = 40 * x + n * y) →
  n = 4 :=
by
  sorry

end dragon_legs_l40_40363


namespace ring_stack_distance_l40_40300

noncomputable def vertical_distance (rings : Nat) : Nat :=
  let diameters := List.range rings |>.map (λ i => 15 - 2 * i)
  let thickness := 1 * rings
  thickness

theorem ring_stack_distance :
  vertical_distance 7 = 58 :=
by 
  sorry

end ring_stack_distance_l40_40300


namespace find_values_f_l40_40828

open Real

noncomputable def f (ω A x : ℝ) : ℝ := 2 * sin (ω * x) * cos (ω * x) + 2 * A * (cos (ω * x))^2 - A

theorem find_values_f (θ : ℝ) (A : ℝ) (ω : ℝ) (hA : A > 0) (hω : ω = 1)
  (h1 : π / 6 < θ) (h2 : θ < π / 3) (h3 : f ω A θ = 2 / 3) :
  f ω A (π / 3 - θ) = (1 + 2 * sqrt 6) / 3 :=
  sorry

end find_values_f_l40_40828


namespace percentage_increase_l40_40236

variable (A B C : ℝ)
variable (h1 : A = 0.71 * C)
variable (h2 : A = 0.05 * B)

theorem percentage_increase (A B C : ℝ) (h1 : A = 0.71 * C) (h2 : A = 0.05 * B) : (B - C) / C = 13.2 :=
by
  sorry

end percentage_increase_l40_40236


namespace total_people_large_seats_is_84_l40_40561

-- Definition of the number of large seats
def large_seats : Nat := 7

-- Definition of the number of people each large seat can hold
def people_per_large_seat : Nat := 12

-- Definition of the total number of people that can ride on large seats
def total_people_large_seats : Nat := large_seats * people_per_large_seat

-- Statement that we need to prove
theorem total_people_large_seats_is_84 : total_people_large_seats = 84 := by
  sorry

end total_people_large_seats_is_84_l40_40561


namespace range_of_values_for_a_l40_40573

theorem range_of_values_for_a (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 > 0 ∧ x2 < 0 ∧ x1^2 + a * x1 + a^2 - 1 = 0 ∧ x2^2 + a * x2 + a^2 - 1 = 0) → (-1 < a ∧ a < 1) :=
by
  sorry

end range_of_values_for_a_l40_40573


namespace polynomial_simplification_l40_40882

variable (x : ℝ)

theorem polynomial_simplification :
  (3*x^3 + 4*x^2 + 12)*(x + 1) - (x + 1)*(2*x^3 + 6*x^2 - 42) + (6*x^2 - 28)*(x + 1)*(x - 2) = 
  7*x^4 - 7*x^3 - 42*x^2 + 82*x + 110 :=
by
  sorry

end polynomial_simplification_l40_40882


namespace solution_set_inequality_l40_40475

theorem solution_set_inequality (x : ℝ) : 
  (abs (x + 3) - abs (x - 2) ≥ 3) ↔ (x ≥ 1) := 
by {
  sorry
}

end solution_set_inequality_l40_40475


namespace extended_hexagon_area_l40_40359

theorem extended_hexagon_area (original_area : ℝ) (side_length_extension : ℝ)
  (original_side_length : ℝ) (new_side_length : ℝ) :
  original_area = 18 ∧ side_length_extension = 1 ∧ original_side_length = 2 
  ∧ new_side_length = original_side_length + 2 * side_length_extension →
  36 = original_area + 6 * (0.5 * side_length_extension * (original_side_length + 1)) := 
sorry

end extended_hexagon_area_l40_40359


namespace output_of_code_snippet_is_six_l40_40502

-- Define the variables and the condition
def a : ℕ := 3
def y : ℕ := if a < 10 then 2 * a else a * a 

-- The statement to be proved
theorem output_of_code_snippet_is_six :
  y = 6 :=
by
  sorry

end output_of_code_snippet_is_six_l40_40502


namespace nap_duration_is_two_hours_l40_40126

-- Conditions as definitions in Lean
def naps_per_week : ℕ := 3
def days : ℕ := 70
def total_nap_hours : ℕ := 60

-- Calculate the duration of each nap
theorem nap_duration_is_two_hours :
  ∃ (nap_duration : ℕ), nap_duration = 2 ∧
  (days / 7) * naps_per_week * nap_duration = total_nap_hours :=
by
  sorry

end nap_duration_is_two_hours_l40_40126


namespace tshirts_per_package_l40_40021

-- Definitions based on the conditions
def total_tshirts : ℕ := 70
def num_packages : ℕ := 14

-- Theorem to prove the number of t-shirts per package
theorem tshirts_per_package : total_tshirts / num_packages = 5 := by
  -- The proof is omitted, only the statement is provided as required.
  sorry

end tshirts_per_package_l40_40021


namespace problem_statement_l40_40633

-- Define the constants and variables
variables (x y z a b c : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := x / a + y / b + z / c = 4
def condition2 : Prop := a / x + b / y + c / z = 1

-- State the theorem that proves the question equals the correct answer
theorem problem_statement (h1 : condition1 x y z a b c) (h2 : condition2 x y z a b c) :
    x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 12 :=
sorry

end problem_statement_l40_40633


namespace valid_integer_pairs_l40_40879

theorem valid_integer_pairs :
  ∀ a b : ℕ, 1 ≤ a → 1 ≤ b → a ^ (b ^ 2) = b ^ a → (a, b) = (1, 1) ∨ (a, b) = (16, 2) ∨ (a, b) = (27, 3) :=
by
  sorry

end valid_integer_pairs_l40_40879


namespace inequality_transformation_l40_40916

theorem inequality_transformation (a b : ℝ) (h : a > b) : -2 * a < -2 * b :=
by {
  sorry
}

end inequality_transformation_l40_40916


namespace find_breadth_of_rectangular_plot_l40_40185

-- Define the conditions
def length_is_thrice_breadth (b l : ℕ) : Prop := l = 3 * b
def area_is_363 (b l : ℕ) : Prop := l * b = 363

-- State the theorem
theorem find_breadth_of_rectangular_plot : ∃ b : ℕ, ∀ l : ℕ, length_is_thrice_breadth b l ∧ area_is_363 b l → b = 11 := 
by
  sorry

end find_breadth_of_rectangular_plot_l40_40185


namespace problem_a_problem_b_problem_c_l40_40837

variables {x y z t : ℝ}

-- Variables are positive
axiom pos_x : 0 < x
axiom pos_y : 0 < y
axiom pos_z : 0 < z
axiom pos_t : 0 < t

-- Problem a)
theorem problem_a : x^4 * y^2 * z + y^4 * x^2 * z + y^4 * z^2 * x + z^4 * y^2 * x + x^4 * z^2 * y + z^4 * x^2 * y
  ≥ 2 * (x^3 * y^2 * z^2 + x^2 * y^3 * z^2 + x^2 * y^2 * z^3) :=
sorry

-- Problem b)
theorem problem_b : x^5 + y^5 + z^5 ≥ x^2 * y^2 * z + x^2 * y * z^2 + x * y^2 * z^2 :=
sorry

-- Problem c)
theorem problem_c : x^3 + y^3 + z^3 + t^3 ≥ x * y * z + x * y * t + x * z * t + y * z * t :=
sorry

end problem_a_problem_b_problem_c_l40_40837


namespace billy_music_book_songs_l40_40889

theorem billy_music_book_songs (can_play : ℕ) (needs_to_learn : ℕ) (total_songs : ℕ) 
  (h1 : can_play = 24) (h2 : needs_to_learn = 28) : 
  total_songs = can_play + needs_to_learn ↔ total_songs = 52 :=
by
  sorry

end billy_music_book_songs_l40_40889


namespace tyrone_gave_marbles_to_eric_l40_40244

theorem tyrone_gave_marbles_to_eric (initial_tyrone_marbles : ℕ) (initial_eric_marbles : ℕ) (marbles_given : ℕ) :
  initial_tyrone_marbles = 150 ∧ initial_eric_marbles = 30 ∧ (initial_tyrone_marbles - marbles_given = 3 * initial_eric_marbles) → marbles_given = 60 :=
by
  sorry

end tyrone_gave_marbles_to_eric_l40_40244


namespace simplify_and_rationalize_l40_40398

noncomputable def expression := 
  (Real.sqrt 8 / Real.sqrt 3) * 
  (Real.sqrt 25 / Real.sqrt 30) * 
  (Real.sqrt 16 / Real.sqrt 21)

theorem simplify_and_rationalize :
  expression = 4 * Real.sqrt 14 / 63 :=
by
  sorry

end simplify_and_rationalize_l40_40398


namespace distinct_right_angles_l40_40712

theorem distinct_right_angles (n : ℕ) (h : n > 0) : 
  ∃ (a b c d : ℕ), (a + b + c + d ≥ 4 * (Int.sqrt n)) ∧ (a * c ≥ n) ∧ (b * d ≥ n) :=
by sorry

end distinct_right_angles_l40_40712


namespace square_ratio_l40_40029

theorem square_ratio (x y : ℝ) (hx : x = 60 / 17) (hy : y = 780 / 169) : 
  x / y = 169 / 220 :=
by
  sorry

end square_ratio_l40_40029


namespace magnitude_of_z_l40_40379

open Complex

theorem magnitude_of_z (z : ℂ) (h : z + I = (2 + I) / I) : abs z = Real.sqrt 10 := by
  sorry

end magnitude_of_z_l40_40379


namespace binomial_square_solution_l40_40486

variable (t u b : ℝ)

theorem binomial_square_solution (h1 : 2 * t * u = 12) (h2 : u^2 = 9) : b = t^2 → b = 4 :=
by
  sorry

end binomial_square_solution_l40_40486


namespace Rick_received_amount_l40_40523

theorem Rick_received_amount :
  let total_promised := 400
  let sally_owes := 35
  let amy_owes := 30
  let derek_owes := amy_owes / 2
  let carl_owes := 35
  let total_owed := sally_owes + amy_owes + derek_owes + carl_owes
  total_promised - total_owed = 285 :=
by
  sorry

end Rick_received_amount_l40_40523


namespace fruit_seller_loss_percentage_l40_40535

theorem fruit_seller_loss_percentage :
  ∃ (C : ℝ), 
    (5 : ℝ) = C - (6.25 - C * (1 + 0.05)) → 
    (C = 6.25) → 
    (C - 5 = 1.25) → 
    (1.25 / 6.25 * 100 = 20) :=
by 
  sorry

end fruit_seller_loss_percentage_l40_40535


namespace inner_rectangle_length_l40_40963

theorem inner_rectangle_length 
  (a b c : ℝ)
  (h1 : ∃ a1 a2 a3 : ℝ, a2 - a1 = a3 - a2)
  (w_inner : ℝ)
  (width_inner : w_inner = 2)
  (w_shaded : ℝ)
  (width_shaded : w_shaded = 1.5)
  (ar_prog : a = 2 * w_inner ∧ b = 3 * w_inner + 15 ∧ c = 3 * w_inner + 33)
  : ∀ x : ℝ, 2 * x = a → 3 * x + 15 = b → 3 * x + 33 = c → x = 3 :=
by
  sorry

end inner_rectangle_length_l40_40963


namespace journey_time_calculation_l40_40875

theorem journey_time_calculation (dist totalDistance : ℝ) (rate1 rate2 : ℝ)
  (firstHalfDistance secondHalfDistance : ℝ) (time1 time2 totalTime : ℝ) :
  totalDistance = 224 ∧ rate1 = 21 ∧ rate2 = 24 ∧
  firstHalfDistance = totalDistance / 2 ∧ secondHalfDistance = totalDistance / 2 ∧
  time1 = firstHalfDistance / rate1 ∧ time2 = secondHalfDistance / rate2 ∧
  totalTime = time1 + time2 →
  totalTime = 10 :=
sorry

end journey_time_calculation_l40_40875


namespace number_of_possible_values_l40_40329

theorem number_of_possible_values (a b c : ℕ) (h : a + 11 * b + 111 * c = 1050) :
  ∃ (n : ℕ), 6 ≤ n ∧ n ≤ 1050 ∧ (n % 9 = 6) ∧ (n = a + 2 * b + 3 * c) :=
sorry

end number_of_possible_values_l40_40329


namespace system_equations_sum_14_l40_40571

theorem system_equations_sum_14 (a b c d : ℝ) 
  (h1 : a + c = 4) 
  (h2 : a * d + b * c = 5) 
  (h3 : a * c + b + d = 8) 
  (h4 : b * d = 1) :
  a + b + c + d = 7 ∨ a + b + c + d = 7 → (a + b + c + d) * 2 = 14 := 
by {
  sorry
}

end system_equations_sum_14_l40_40571


namespace work_completion_l40_40121

variable (A B : Type)

/-- A can do half of the work in 70 days and B can do one third of the work in 35 days.
Together, A and B can complete the work in 60 days. -/
theorem work_completion (hA : (1 : ℚ) / 2 / 70 = (1 : ℚ) / a) 
                      (hB : (1 : ℚ) / 3 / 35 = (1 : ℚ) / b) :
                      (1 / 140 + 1 / 105) = 1 / 60 :=
  sorry

end work_completion_l40_40121


namespace arithmetic_sequence_fraction_zero_l40_40262

noncomputable def arithmetic_sequence_term (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

theorem arithmetic_sequence_fraction_zero (a1 d : ℝ) 
    (h1 : a1 ≠ 0) (h9 : arithmetic_sequence_term a1 d 9 = 0) :
  (arithmetic_sequence_term a1 d 1 + 
   arithmetic_sequence_term a1 d 8 + 
   arithmetic_sequence_term a1 d 11 + 
   arithmetic_sequence_term a1 d 16) / 
  (arithmetic_sequence_term a1 d 7 + 
   arithmetic_sequence_term a1 d 8 + 
   arithmetic_sequence_term a1 d 14) = 0 :=
by
  sorry

end arithmetic_sequence_fraction_zero_l40_40262


namespace scenario_a_scenario_b_scenario_c_scenario_d_scenario_e_scenario_f_l40_40264

open Nat

-- Definitions for combinations and permutations
def binomial (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))
def variations (n k : ℕ) : ℕ := n.factorial / (n - k).factorial

-- Scenario a: Each path can be used by at most one person and at most once
theorem scenario_a : binomial 5 2 * binomial 3 2 = 30 := by sorry

-- Scenario b: Each path can be used twice but only in different directions
theorem scenario_b : binomial 5 2 * binomial 5 2 = 100 := by sorry

-- Scenario c: No restrictions
theorem scenario_c : (5 * 5) * (5 * 5) = 625 := by sorry

-- Scenario d: Same as a) with two people distinguished
theorem scenario_d : variations 5 2 * variations 3 2 = 120 := by sorry

-- Scenario e: Same as b) with two people distinguished
theorem scenario_e : variations 5 2 * variations 5 2 = 400 := by sorry

-- Scenario f: Same as c) with two people distinguished
theorem scenario_f : (5 * 5) * (5 * 5) = 625 := by sorry

end scenario_a_scenario_b_scenario_c_scenario_d_scenario_e_scenario_f_l40_40264


namespace geometric_sequence_first_term_l40_40644

theorem geometric_sequence_first_term 
  (T : ℕ → ℝ) 
  (h1 : T 5 = 243) 
  (h2 : T 6 = 729) 
  (hr : ∃ r : ℝ, ∀ n : ℕ, T n = T 1 * r^(n - 1)) :
  T 1 = 3 :=
by
  sorry

end geometric_sequence_first_term_l40_40644


namespace james_proof_l40_40033

def james_pages_per_hour 
  (writes_some_pages_an_hour : ℕ)
  (writes_5_pages_to_2_people_each_day : ℕ)
  (hours_spent_writing_per_week : ℕ) 
  (writes_total_pages_per_day : ℕ)
  (writes_total_pages_per_week : ℕ) 
  (pages_per_hour : ℕ) 
: Prop :=
  writes_some_pages_an_hour = writes_5_pages_to_2_people_each_day / hours_spent_writing_per_week

theorem james_proof
  (writes_some_pages_an_hour : ℕ := 10)
  (writes_5_pages_to_2_people_each_day : ℕ := 5 * 2)
  (hours_spent_writing_per_week : ℕ := 7)
  (writes_total_pages_per_day : ℕ := writes_5_pages_to_2_people_each_day)
  (writes_total_pages_per_week : ℕ := writes_total_pages_per_day * 7)
  (pages_per_hour : ℕ := writes_total_pages_per_week / hours_spent_writing_per_week)
: writes_some_pages_an_hour = pages_per_hour :=
by {
  sorry 
}

end james_proof_l40_40033


namespace find_x_l40_40622

-- Definition of logarithm in Lean
noncomputable def log (b a: ℝ) : ℝ := Real.log a / Real.log b

-- Problem statement in Lean
theorem find_x (x : ℝ) (h : log 64 4 = 1 / 3) : log x 8 = 1 / 3 → x = 512 :=
by sorry

end find_x_l40_40622


namespace tangent_slope_at_1_0_l40_40196

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_slope_at_1_0 : (deriv f 1) = 3 := by
  sorry

end tangent_slope_at_1_0_l40_40196


namespace option_B_valid_l40_40252

-- Definitions derived from conditions
def at_least_one_black (balls : List Bool) : Prop :=
  ∃ b ∈ balls, b = true

def both_black (balls : List Bool) : Prop :=
  balls = [true, true]

def exactly_one_black (balls : List Bool) : Prop :=
  balls.count true = 1

def exactly_two_black (balls : List Bool) : Prop :=
  balls.count true = 2

def mutually_exclusive (P Q : Prop) : Prop :=
  P ∧ Q → False

def non_complementary (P Q : Prop) : Prop :=
  ¬(P → ¬Q) ∧ ¬(¬P → Q)

-- Balls: true represents a black ball, false represents a red ball.
def all_draws := [[true, true], [true, false], [false, true], [false, false]]

-- Proof statement
theorem option_B_valid :
  (mutually_exclusive (exactly_one_black [true, false]) (exactly_two_black [true, true])) ∧ 
  (non_complementary (exactly_one_black [true, false]) (exactly_two_black [true, true])) :=
  sorry

end option_B_valid_l40_40252


namespace equation_represents_hyperbola_l40_40383

theorem equation_represents_hyperbola (x y : ℝ) :
  x^2 - 4*y^2 - 2*x + 8*y - 8 = 0 → ∃ a b h k : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ (a * (x - h)^2 - b * (y - k)^2 = 1) := 
sorry

end equation_represents_hyperbola_l40_40383


namespace polynomial_unique_l40_40600

noncomputable def p (x : ℝ) : ℝ := x^2 + 1

theorem polynomial_unique (p : ℝ → ℝ) 
  (h1 : p 2 = 5) 
  (h2 : ∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 2) : 
  ∀ x : ℝ, p x = x^2 + 1 :=
by
  sorry

end polynomial_unique_l40_40600


namespace rabbit_weight_l40_40974

theorem rabbit_weight (a b c : ℕ) (h1 : a + b + c = 30) (h2 : a + c = 2 * b) (h3 : a + b = c) :
  a = 5 := by
  sorry

end rabbit_weight_l40_40974


namespace domain_of_function_l40_40282

theorem domain_of_function (x : ℝ) :
  (x^2 - 5*x + 6 ≥ 0) → (x ≠ 2) → (x < 2 ∨ x ≥ 3) :=
by
  intros h1 h2
  sorry

end domain_of_function_l40_40282


namespace find_first_number_l40_40074

/-- The Least Common Multiple (LCM) of two numbers A and B is 2310,
    and their Highest Common Factor (HCF) is 30.
    Given one of the numbers B is 180, find the other number A. -/
theorem find_first_number (A B : ℕ) (LCM HCF : ℕ) (h1 : LCM = 2310) (h2 : HCF = 30) (h3 : B = 180) (h4 : A * B = LCM * HCF) :
  A = 385 :=
by sorry

end find_first_number_l40_40074


namespace problem_inequality_l40_40149

variable {a b c : ℝ}

theorem problem_inequality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_not_all_equal : ¬ (a = b ∧ b = c)) : 
  2 * (a^3 + b^3 + c^3) > a^2 * (b + c) + b^2 * (a + c) + c^2 * (a + b) :=
by sorry

end problem_inequality_l40_40149


namespace john_subtracts_79_l40_40318

theorem john_subtracts_79 :
  let a := 40
  let b := 1
  let n := (a - b) * (a - b)
  n = a * a - 79
:= by
  sorry

end john_subtracts_79_l40_40318


namespace sam_and_erica_money_total_l40_40810

def sam_money : ℕ := 38
def erica_money : ℕ := 53

theorem sam_and_erica_money_total : sam_money + erica_money = 91 :=
by
  -- the proof is not required; hence we skip it
  sorry

end sam_and_erica_money_total_l40_40810


namespace solve_inequality_l40_40463

theorem solve_inequality :
  {x : ℝ | (x - 1) * (2 * x + 1) ≤ 0} = { x : ℝ | -1/2 ≤ x ∧ x ≤ 1 } :=
by
  sorry

end solve_inequality_l40_40463


namespace solution_set_f_neg_x_l40_40822

noncomputable def f (a b x : Real) : Real := (a * x - 1) * (x - b)

theorem solution_set_f_neg_x (a b : Real) (h : ∀ x, f a b x > 0 ↔ -1 < x ∧ x < 3) : 
  ∀ x, f a b (-x) < 0 ↔ x < -3 ∨ x > 1 := 
by
  sorry

end solution_set_f_neg_x_l40_40822


namespace find_m_interval_l40_40540

-- Define the sequence recursively
def sequence_recursive (x : ℕ → ℝ) (n : ℕ) : Prop :=
  x 0 = 5 ∧ ∀ n, x (n + 1) = (x n ^ 2 + 5 * x n + 4) / (x n + 6)

-- The left-hand side of the inequality
noncomputable def target_value : ℝ := 4 + 1 / (2 ^ 20)

-- The condition that the sequence element must satisfy
def condition (x : ℕ → ℝ) (m : ℕ) : Prop :=
  x m ≤ target_value

-- The proof problem statement, m lies within the given interval
theorem find_m_interval (x : ℕ → ℝ) (m : ℕ) :
  sequence_recursive x n →
  condition x m →
  81 ≤ m ∧ m ≤ 242 :=
sorry

end find_m_interval_l40_40540


namespace ratio_a_d_l40_40538

theorem ratio_a_d 
  (a b c d : ℕ) 
  (h1 : a / b = 1 / 4) 
  (h2 : b / c = 13 / 9) 
  (h3 : c / d = 5 / 13) : 
  a / d = 5 / 36 :=
sorry

end ratio_a_d_l40_40538


namespace tangent_line_eq_monotonic_intervals_extremes_f_l40_40277

variables {a x : ℝ}

noncomputable def f (a x : ℝ) : ℝ := -1/3 * x^3 + 2 * a * x^2 - 3 * a^2 * x
noncomputable def f' (a x : ℝ) : ℝ := -x^2 + 4 * a * x - 3 * a^2

theorem tangent_line_eq {a : ℝ} (h : a = -1) : (∃ y, y = f (-1) (-2) ∧ 3 * x - 3 * y + 8 = 0) := sorry

theorem monotonic_intervals_extremes {a : ℝ} (h : 0 < a) :
  (∀ x, (a < x ∧ x < 3 * a → 0 < f' a x) ∧ 
        (x < a ∨ 3 * a < x → f' a x < 0) ∧ 
        (f a (3 * a) = 0 ∧ f a a = -4/3 * a^3)) := sorry

theorem f'_inequality_range (h1 : ∀ x, 2 * a ≤ x ∧ x ≤ 2 * a + 2 → |f' a x| ≤ 3 * a) :
  (1 ≤ a ∧ a ≤ 3) := sorry

end tangent_line_eq_monotonic_intervals_extremes_f_l40_40277


namespace factorization_problem_l40_40407

theorem factorization_problem :
  ∃ a b : ℤ, (∀ y : ℤ, 4 * y^2 - 3 * y - 28 = (4 * y + a) * (y + b))
    ∧ (a - b = 11) := by
  sorry

end factorization_problem_l40_40407


namespace simplify_fraction_l40_40287

theorem simplify_fraction (k : ℤ) : 
  (∃ (a b : ℤ), a = 1 ∧ b = 2 ∧ (6 * k + 12) / 6 = a * k + b) → (1 / 2 : ℚ) = (1 / 2 : ℚ) := 
by
  intro h
  sorry

end simplify_fraction_l40_40287


namespace problem_statement_l40_40598

open Set

variable (U P Q : Set ℕ)
variables (hU : U = {1, 2, 3, 4, 5, 6}) (hP : P = {1, 2, 3, 4}) (hQ : Q = {3, 4, 5})

theorem problem_statement : P ∩ (U \ Q) = {1, 2} :=
by
  sorry

end problem_statement_l40_40598


namespace monkeys_more_than_giraffes_l40_40513

theorem monkeys_more_than_giraffes :
  let zebras := 12
  let camels := zebras / 2
  let monkeys := 4 * camels
  let giraffes := 2
  monkeys - giraffes = 22
:= by
  intros
  let zebras := 12
  let camels := zebras / 2
  let monkeys := 4 * camels
  let giraffes := 2
  have h := monkeys - giraffes
  exact sorry

end monkeys_more_than_giraffes_l40_40513


namespace tan_beta_l40_40932

noncomputable def tan_eq_2 (α β : ℝ) (h1 : Real.tan (α + β) = 3) (h2 : Real.tan (α + π / 4) = 2) : Real :=
2

theorem tan_beta (α β : ℝ) (h1 : Real.tan (α + β) = 3) (h2 : Real.tan (α + π / 4) = 2) :
  Real.tan β = tan_eq_2 α β h1 h2 := by
  sorry

end tan_beta_l40_40932


namespace solve_abs_eq_l40_40818

theorem solve_abs_eq (x : ℝ) : (|x + 2| = 3*x - 6) → x = 4 :=
by
  intro h
  sorry

end solve_abs_eq_l40_40818


namespace cars_on_river_road_l40_40567

variable (B C M : ℕ)

theorem cars_on_river_road
  (h1 : ∃ B C : ℕ, B / C = 1 / 3) -- ratio of buses to cars is 1:3
  (h2 : ∀ B C : ℕ, C = B + 40) -- 40 fewer buses than cars
  (h3 : ∃ B C M : ℕ, B + C + M = 720) -- total number of vehicles is 720
  : C = 60 :=
sorry

end cars_on_river_road_l40_40567


namespace tetrahedron_a_exists_tetrahedron_b_not_exists_l40_40835

/-- Part (a): There exists a tetrahedron with two edges shorter than 1 cm,
    and the other four edges longer than 1 km. -/
theorem tetrahedron_a_exists : 
  ∃ (a b c d : ℝ), 
    ((a < 1 ∧ b < 1 ∧ 1000 < c ∧ 1000 < d ∧ 1000 < (a + c) ∧ 1000 < (b + d)) ∧ 
     a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a) := 
sorry

/-- Part (b): There does not exist a tetrahedron with four edges shorter than 1 cm,
    and the other two edges longer than 1 km. -/
theorem tetrahedron_b_not_exists : 
  ¬ ∃ (a b c d : ℝ), 
    ((a < 1 ∧ b < 1 ∧ c < 1 ∧ d < 1 ∧ 1000 < (a + c) ∧ 1000 < (b + d)) ∧ 
     a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ) := 
sorry

end tetrahedron_a_exists_tetrahedron_b_not_exists_l40_40835


namespace find_c_l40_40817

/-
Given:
1. c and d are integers.
2. x^2 - x - 1 is a factor of cx^{18} + dx^{17} + x^2 + 1.
Show that c = -1597 under these conditions.

Assume we have the following Fibonacci number definitions:
F_16 = 987,
F_17 = 1597,
F_18 = 2584,
then:
Proof that c = -1597.
-/

noncomputable def fib (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else fib (n - 1) + fib (n - 2)

theorem find_c (c d : ℤ) (h1 : c * 2584 + d * 1597 + 1 = 0) (h2 : c * 1597 + d * 987 + 2 = 0) :
  c = -1597 :=
by
  sorry

end find_c_l40_40817


namespace possible_slopes_of_line_intersecting_ellipse_l40_40292

theorem possible_slopes_of_line_intersecting_ellipse (m : ℝ) :
  (∃ x y : ℝ, y = m * x + 3 ∧ 4 * x^2 + 25 * y^2 = 100) ↔ m ∈ (Set.Iic (-2/5) ∪ Set.Ici (2/5)) :=
by
  sorry

end possible_slopes_of_line_intersecting_ellipse_l40_40292


namespace find_k_l40_40181

def total_balls (k : ℕ) : ℕ := 7 + k

def probability_green (k : ℕ) : ℚ := 7 / (total_balls k)
def probability_purple (k : ℕ) : ℚ := k / (total_balls k)

def expected_value (k : ℕ) : ℚ :=
  (probability_green k) * 3 + (probability_purple k) * (-1)

theorem find_k (k : ℕ) (h_pos : k > 0) (h_exp_value : expected_value k = 1) : k = 7 :=
sorry

end find_k_l40_40181


namespace Q_value_l40_40914

theorem Q_value (a b c P Q : ℝ) (h1 : a + b + c = 0)
    (h2 : (a^2 / (2 * a^2 + b * c)) + (b^2 / (2 * b^2 + a * c)) + (c^2 / (2 * c^2 + a * b)) = P - 3 * Q) : 
    Q = 8 := 
sorry

end Q_value_l40_40914


namespace f_f_n_plus_n_eq_n_plus_1_l40_40038

-- Define the function f : ℕ+ → ℕ+ satisfying the given condition
axiom f : ℕ+ → ℕ+

-- Define that for all positive integers n, f satisfies the condition f(f(n)) + f(n+1) = n + 2
axiom f_condition : ∀ n : ℕ+, f (f n) + f (n + 1) = n + 2

-- State that we want to prove that f(f(n) + n) = n + 1 for all positive integers n
theorem f_f_n_plus_n_eq_n_plus_1 : ∀ n : ℕ+, f (f n + n) = n + 1 := 
by sorry

end f_f_n_plus_n_eq_n_plus_1_l40_40038


namespace complement_union_correct_l40_40844

open Set

variable (U A B : Set ℕ)
variable (hU : U = {0, 1, 2, 3, 4})
variable (hA : A = {0, 1, 2})
variable (hB : B = {2, 3})

theorem complement_union_correct :
  (U \ A) ∪ B = {2, 3, 4} := by
  sorry

end complement_union_correct_l40_40844


namespace men_in_first_group_l40_40637

theorem men_in_first_group
  (M : ℕ) -- number of men in the first group
  (h1 : M * 8 * 24 = 12 * 8 * 16) : M = 8 :=
sorry

end men_in_first_group_l40_40637


namespace cheryl_distance_walked_l40_40617

theorem cheryl_distance_walked :
  let s1 := 2  -- speed during the first segment in miles per hour
  let t1 := 3  -- time during the first segment in hours
  let s2 := 4  -- speed during the second segment in miles per hour
  let t2 := 2  -- time during the second segment in hours
  let s3 := 1  -- speed during the third segment in miles per hour
  let t3 := 3  -- time during the third segment in hours
  let s4 := 3  -- speed during the fourth segment in miles per hour
  let t4 := 5  -- time during the fourth segment in hours
  let d1 := s1 * t1  -- distance for the first segment
  let d2 := s2 * t2  -- distance for the second segment
  let d3 := s3 * t3  -- distance for the third segment
  let d4 := s4 * t4  -- distance for the fourth segment
  d1 + d2 + d3 + d4 = 32 :=
by
  sorry

end cheryl_distance_walked_l40_40617


namespace min_white_surface_area_is_five_over_ninety_six_l40_40503

noncomputable def fraction_white_surface_area (total_surface_area white_surface_area : ℕ) :=
  (white_surface_area : ℚ) / (total_surface_area : ℚ)

theorem min_white_surface_area_is_five_over_ninety_six :
  let total_surface_area := 96
  let white_surface_area := 5
  fraction_white_surface_area total_surface_area white_surface_area = 5 / 96 :=
by
  sorry

end min_white_surface_area_is_five_over_ninety_six_l40_40503


namespace fraction_of_girls_in_debate_l40_40316

theorem fraction_of_girls_in_debate (g b : ℕ) (h : g = b) :
  ((2 / 3) * g) / ((2 / 3) * g + (3 / 5) * b) = 30 / 57 :=
by
  sorry

end fraction_of_girls_in_debate_l40_40316


namespace distance_ratio_l40_40956

theorem distance_ratio (D90 D180 : ℝ) 
  (h1 : D90 + D180 = 3600) 
  (h2 : D90 / 90 + D180 / 180 = 30) : 
  D90 / D180 = 1 := 
by 
  sorry

end distance_ratio_l40_40956


namespace log_comparison_l40_40949

theorem log_comparison 
  (a : ℝ := 1 / 6 * Real.log 8)
  (b : ℝ := 1 / 2 * Real.log 5)
  (c : ℝ := Real.log (Real.sqrt 6) - Real.log (Real.sqrt 2)) :
  a < c ∧ c < b := 
by
  sorry

end log_comparison_l40_40949


namespace hydrochloric_acid_moles_l40_40859

theorem hydrochloric_acid_moles (amyl_alcohol moles_required : ℕ) 
  (h_ratio : amyl_alcohol = moles_required) 
  (h_balanced : amyl_alcohol = 3) :
  moles_required = 3 :=
by
  sorry

end hydrochloric_acid_moles_l40_40859


namespace final_temp_fahrenheit_correct_l40_40417

noncomputable def initial_temp_celsius : ℝ := 50
noncomputable def conversion_c_to_f (c: ℝ) : ℝ := (c * 9 / 5) + 32
noncomputable def final_temp_celsius := initial_temp_celsius / 2

theorem final_temp_fahrenheit_correct : conversion_c_to_f final_temp_celsius = 77 :=
  by sorry

end final_temp_fahrenheit_correct_l40_40417


namespace percentage_difference_l40_40224

theorem percentage_difference :
    let A := (40 / 100) * ((50 / 100) * 60)
    let B := (50 / 100) * ((60 / 100) * 70)
    (B - A) = 9 :=
by
    sorry

end percentage_difference_l40_40224


namespace SquareArea_l40_40284

theorem SquareArea (s : ℝ) (θ : ℝ) (h1 : s = 3) (h2 : θ = π / 4) : s * s = 9 := 
by 
  sorry

end SquareArea_l40_40284


namespace scientific_notation_of_384_000_000_l40_40928

theorem scientific_notation_of_384_000_000 :
  384000000 = 3.84 * 10^8 :=
sorry

end scientific_notation_of_384_000_000_l40_40928


namespace salt_added_correct_l40_40392

theorem salt_added_correct (x : ℝ)
  (hx : x = 119.99999999999996)
  (initial_salt : ℝ := 0.20 * x)
  (evaporation_volume : ℝ := x - (1/4) * x)
  (additional_water : ℝ := 8)
  (final_volume : ℝ := evaporation_volume + additional_water)
  (final_concentration : ℝ := 1 / 3)
  (final_salt : ℝ := final_concentration * final_volume)
  (salt_added : ℝ := final_salt - initial_salt) :
  salt_added = 8.67 :=
sorry

end salt_added_correct_l40_40392


namespace bankers_discount_l40_40834

theorem bankers_discount {TD S BD : ℝ} (hTD : TD = 66) (hS : S = 429) :
  BD = (TD * S) / (S - TD) → BD = 78 :=
by
  intros h
  rw [hTD, hS] at h
  sorry

end bankers_discount_l40_40834


namespace speed_of_man_in_still_water_l40_40695

theorem speed_of_man_in_still_water 
  (V_m V_s : ℝ)
  (h1 : 6 = V_m + V_s)
  (h2 : 4 = V_m - V_s) : 
  V_m = 5 := 
by 
  sorry

end speed_of_man_in_still_water_l40_40695


namespace incorrect_median_l40_40179

def data_set : List ℕ := [7, 11, 10, 11, 6, 14, 11, 10, 11, 9]

noncomputable def median (l : List ℕ) : ℚ := 
  let sorted := l.toArray.qsort (· ≤ ·) 
  if sorted.size % 2 = 0 then
    (sorted.get! (sorted.size / 2 - 1) + sorted.get! (sorted.size / 2)) / 2
  else
    sorted.get! (sorted.size / 2)

theorem incorrect_median :
  median data_set ≠ 10 := by
  sorry

end incorrect_median_l40_40179


namespace pasha_wins_9_games_l40_40010

theorem pasha_wins_9_games :
  ∃ w l : ℕ, (w + l = 12) ∧ (2^w * (2^l - 1) - (2^l - 1) * 2^(w - 1) = 2023) ∧ (w = 9) :=
by
  sorry

end pasha_wins_9_games_l40_40010


namespace ratio_of_puzzle_times_l40_40635

def total_time := 70
def warmup_time := 10
def remaining_puzzles := 60 / 2

theorem ratio_of_puzzle_times : (remaining_puzzles / warmup_time) = 3 := by
  -- Given Conditions
  have H1 : 70 = 10 + 2 * (60 / 2) := by sorry
  -- Simplification and Calculation
  have H2 : (remaining_puzzles = 30) := by sorry
  -- Ratio Calculation
  have ratio_calculation: (30 / 10) = 3 := by sorry
  exact ratio_calculation

end ratio_of_puzzle_times_l40_40635


namespace line_intersects_ellipse_l40_40138

theorem line_intersects_ellipse (k : ℝ) (m : ℝ) : 
  (∀ x y : ℝ, y = k * x + 1 → (x^2 / 5) + (y^2 / m) = 1 → True) ↔ (1 < m ∧ m < 5) ∨ (5 < m) :=
by
  sorry

end line_intersects_ellipse_l40_40138


namespace min_dot_product_l40_40904

-- Define the conditions of the ellipse and focal points
variables (P : ℝ × ℝ)
def ellipse (x y : ℝ) := (x^2 / 4) + (y^2 / 3) = 1
def O : ℝ × ℝ := (0, 0)
def F : ℝ × ℝ := (-1, 0)

-- Define vectors
def OP (P : ℝ × ℝ) : ℝ × ℝ := P
def FP (P : ℝ × ℝ) : ℝ × ℝ := (P.1 + 1, P.2)

-- Define the dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Prove that the minimum value of the dot product is 2
theorem min_dot_product (hP : ellipse P.1 P.2) : 
  ∃ (P : ℝ × ℝ), dot_product (OP P) (FP P) = 2 := sorry

end min_dot_product_l40_40904


namespace scientific_notation_correct_l40_40537

noncomputable def scientific_notation_139000 : Prop :=
  139000 = 1.39 * 10^5

theorem scientific_notation_correct : scientific_notation_139000 :=
by
  -- The proof would be included here, but we add sorry to skip it
  sorry

end scientific_notation_correct_l40_40537


namespace at_least_one_is_one_l40_40543

theorem at_least_one_is_one (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0):
  (1/x + 1/y + 1/z = 1) → (1/(x + y + z) = 1) → (x = 1 ∨ y = 1 ∨ z = 1) :=
by
  sorry

end at_least_one_is_one_l40_40543


namespace total_earnings_correct_l40_40280

-- Define the earnings of each individual
def SalvadorEarnings := 1956
def SantoEarnings := SalvadorEarnings / 2
def MariaEarnings := 3 * SantoEarnings
def PedroEarnings := SantoEarnings + MariaEarnings

-- Define the total earnings calculation
def TotalEarnings := SalvadorEarnings + SantoEarnings + MariaEarnings + PedroEarnings

-- State the theorem to prove
theorem total_earnings_correct :
  TotalEarnings = 9780 :=
sorry

end total_earnings_correct_l40_40280


namespace Johnny_is_8_l40_40429

-- Define Johnny's current age
def johnnys_age (x : ℕ) : Prop :=
  x + 2 = 2 * (x - 3)

theorem Johnny_is_8 (x : ℕ) (h : johnnys_age x) : x = 8 :=
sorry

end Johnny_is_8_l40_40429


namespace min_value_l40_40481

theorem min_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 1) : 
  ∃ min_val, min_val = 5 + 2 * Real.sqrt 6 ∧ (∀ x, (x = 5 + 2 * Real.sqrt 6) → x ≥ min_val) :=
by
  sorry

end min_value_l40_40481


namespace work_completion_days_l40_40455

theorem work_completion_days (A B : ℕ) (hA : A = 20) (hB : B = 20) : A + B / (A + B) / 2 = 10 :=
by 
  rw [hA, hB]
  -- Proof omitted
  sorry

end work_completion_days_l40_40455


namespace line_through_longest_chord_l40_40009

-- Define the point M and the circle equation
def M : ℝ × ℝ := (3, -1)
def circle_eqn (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + y - 2 = 0

-- Define the standard form of the circle equation
def standard_circle_eqn (x y : ℝ) : Prop := (x - 2)^2 + (y + 1/2)^2 = 25/4

-- Define the line equation
def line_eqn (x y : ℝ) : Prop := x + 2 * y - 2 = 0

-- Theorem: Equation of the line containing the longest chord passing through M
theorem line_through_longest_chord : 
  (circle_eqn 3 (-1)) → 
  ∀ (x y : ℝ), standard_circle_eqn x y → ∃ (k b : ℝ), line_eqn x y :=
by
  -- Proof goes here
  intro h1 x y h2
  sorry

end line_through_longest_chord_l40_40009


namespace Jolene_cars_washed_proof_l40_40123

-- Definitions for conditions
def number_of_families : ℕ := 4
def babysitting_rate : ℕ := 30 -- in dollars
def car_wash_rate : ℕ := 12 -- in dollars
def total_money_raised : ℕ := 180 -- in dollars

-- Mathematical representation of the problem:
def babysitting_earnings : ℕ := number_of_families * babysitting_rate
def earnings_from_cars : ℕ := total_money_raised - babysitting_earnings
def number_of_cars_washed : ℕ := earnings_from_cars / car_wash_rate

-- The proof statement
theorem Jolene_cars_washed_proof : number_of_cars_washed = 5 := 
sorry

end Jolene_cars_washed_proof_l40_40123


namespace find_x_in_sequence_l40_40933

theorem find_x_in_sequence :
  (∀ a b c d : ℕ, a * b * c * d = 120) →
  (a = 2) →
  (b = 4) →
  (d = 3) →
  ∃ x : ℕ, 2 * 4 * x * 3 = 120 ∧ x = 5 :=
sorry

end find_x_in_sequence_l40_40933


namespace complement_U_A_l40_40546

def U : Finset ℤ := {-2, -1, 0, 1, 2}
def A : Finset ℤ := {-2, -1, 1, 2}

theorem complement_U_A : (U \ A) = {0} := by
  sorry

end complement_U_A_l40_40546


namespace count_positive_integers_x_satisfying_inequality_l40_40849

theorem count_positive_integers_x_satisfying_inequality :
  ∃ n : ℕ, n = 6 ∧ (∀ x : ℕ, (144 ≤ x^2 ∧ x^2 ≤ 289) → (x = 12 ∨ x = 13 ∨ x = 14 ∨ x = 15 ∨ x = 16 ∨ x = 17)) :=
sorry

end count_positive_integers_x_satisfying_inequality_l40_40849


namespace num_pens_multiple_of_16_l40_40590

theorem num_pens_multiple_of_16 (Pencils Students : ℕ) (h1 : Pencils = 928) (h2 : Students = 16)
  (h3 : ∃ (Pn : ℕ), Pencils = Pn * Students) :
  ∃ (k : ℕ), ∃ (Pens : ℕ), Pens = 16 * k :=
by
  sorry

end num_pens_multiple_of_16_l40_40590


namespace combine_like_terms_l40_40289

theorem combine_like_terms (a b : ℝ) : -3 * a^2 * b + 2 * a^2 * b = -a^2 * b := 
  sorry

end combine_like_terms_l40_40289


namespace reciprocals_of_each_other_l40_40793

theorem reciprocals_of_each_other (a b : ℝ) (h : (a + b)^2 - (a - b)^2 = 4) : a * b = 1 :=
by 
  sorry

end reciprocals_of_each_other_l40_40793


namespace arithmetic_mean_of_reciprocals_is_correct_l40_40299

/-- The first four prime numbers -/
def first_four_primes : List ℕ := [2, 3, 5, 7]

/-- Taking reciprocals and summing them up  -/
def reciprocals_sum : ℚ :=
  (1/2) + (1/3) + (1/5) + (1/7)

/-- The arithmetic mean of the reciprocals  -/
def arithmetic_mean_of_reciprocals :=
  reciprocals_sum / 4

/-- The result of the arithmetic mean of the reciprocals  -/
theorem arithmetic_mean_of_reciprocals_is_correct :
  arithmetic_mean_of_reciprocals = 247/840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_is_correct_l40_40299


namespace num_words_at_least_one_vowel_l40_40788

-- Definitions based on conditions.
def letters : List Char := ['A', 'B', 'E', 'G', 'H']
def vowels : List Char := ['A', 'E']
def consonants : List Char := ['B', 'G', 'H']

-- The main statement posing the question and answer.
theorem num_words_at_least_one_vowel :
  let total_words := (letters.length) ^ 5
  let consonant_words := (consonants.length) ^ 5
  let result := total_words - consonant_words
  result = 2882 :=
by {
  let total_words := 5 ^ 5
  let consonant_words := 3 ^ 5
  let result := total_words - consonant_words
  have : result = 2882 := by sorry
  exact this
}

end num_words_at_least_one_vowel_l40_40788


namespace angle_ABC_is_83_degrees_l40_40806

theorem angle_ABC_is_83_degrees (A B C D K : Type)
  (angle_BAC : Real) (angle_CAD : Real) (angle_ACD : Real)
  (AB AC AD : Real) (angle_ABC : Real) :
  angle_BAC = 60 ∧ angle_CAD = 60 ∧ angle_ACD = 23 ∧ AB + AD = AC → 
  angle_ABC = 83 :=
by
  sorry

end angle_ABC_is_83_degrees_l40_40806


namespace volume_of_cuboid_l40_40624

-- Define the edges of the cuboid
def edge1 : ℕ := 6
def edge2 : ℕ := 5
def edge3 : ℕ := 6

-- Define the volume formula for a cuboid
def volume (a b c : ℕ) : ℕ := a * b * c

-- State the theorem
theorem volume_of_cuboid : volume edge1 edge2 edge3 = 180 := by
  sorry

end volume_of_cuboid_l40_40624


namespace roots_sum_roots_product_algebraic_expression_l40_40256

theorem roots_sum (x1 x2 : ℝ) (h : x1^2 - x1 - 1 = 0 ∧ x2^2 - x2 - 1 = 0) :
  x1 + x2 = 1 :=
sorry

theorem roots_product (x1 x2 : ℝ) (h : x1^2 - x1 - 1 = 0 ∧ x2^2 - x2 - 1 = 0) :
  x1 * x2 = -1 :=
sorry

theorem algebraic_expression (x1 x2 : ℝ) (h : x1^2 - x1 - 1 = 0 ∧ x2^2 - x2 - 1 = 0) :
  x1^2 + x2^2 = 3 :=
sorry

end roots_sum_roots_product_algebraic_expression_l40_40256


namespace max_cube_side_length_max_rect_parallelepiped_dimensions_l40_40638

-- Part (a)
theorem max_cube_side_length (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ s : ℝ, s = a * b * c / (a * b + b * c + a * c) :=
sorry

-- Part (b)
theorem max_rect_parallelepiped_dimensions (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ x y z : ℝ, x = a / 3 ∧ y = b / 3 ∧ z = c / 3 :=
sorry

end max_cube_side_length_max_rect_parallelepiped_dimensions_l40_40638


namespace kibble_left_l40_40375

-- Define the initial amount of kibble
def initial_kibble := 3

-- Define the rate at which the cat eats kibble
def kibble_rate := 1 / 4

-- Define the time Kira was away
def time_away := 8

-- Define the amount of kibble eaten by the cat during the time away
def kibble_eaten := (time_away * kibble_rate)

-- Define the remaining kibble in the bowl
def remaining_kibble := initial_kibble - kibble_eaten

-- State and prove that the remaining amount of kibble is 1 pound
theorem kibble_left : remaining_kibble = 1 := by
  sorry

end kibble_left_l40_40375


namespace Ali_winning_strategy_l40_40469

def Ali_and_Mohammad_game (m n : ℕ) (a : Fin m → ℕ) : Prop :=
∃ (k l : ℕ), k > 0 ∧ l > 0 ∧ (∃ p : ℕ, Nat.Prime p ∧ m = p^k ∧ n = p^l)

theorem Ali_winning_strategy (m n : ℕ) (a : Fin m → ℕ) :
  Ali_and_Mohammad_game m n a :=
sorry

end Ali_winning_strategy_l40_40469


namespace basic_cable_cost_l40_40041

variable (B M S : ℝ)

def CostOfMovieChannels (B : ℝ) : ℝ := B + 12
def CostOfSportsChannels (M : ℝ) : ℝ := M - 3

theorem basic_cable_cost :
  let M := CostOfMovieChannels B
  let S := CostOfSportsChannels M
  B + M + S = 36 → B = 5 :=
by
  intro h
  let M := CostOfMovieChannels B
  let S := CostOfSportsChannels M
  sorry

end basic_cable_cost_l40_40041


namespace range_of_m_l40_40114

open Set

variable {α : Type}

noncomputable def A (m : ℝ) : Set ℝ := {x | m+1 ≤ x ∧ x ≤ 2*m-1}
noncomputable def B : Set ℝ := {x | x^2 - 2*x - 15 ≤ 0}

theorem range_of_m (m : ℝ) (hA : A m ⊆ B) (hA_nonempty : A m ≠ ∅) : 1 ≤ m ∧ m ≤ 3 := by
  sorry

end range_of_m_l40_40114


namespace total_distance_of_ship_l40_40500

-- Define the conditions
def first_day_distance : ℕ := 100
def second_day_distance := 3 * first_day_distance
def third_day_distance := second_day_distance + 110
def total_distance := first_day_distance + second_day_distance + third_day_distance

-- Theorem stating that given the conditions the total distance traveled is 810 miles
theorem total_distance_of_ship :
  total_distance = 810 := by
  sorry

end total_distance_of_ship_l40_40500


namespace weave_mats_l40_40421

theorem weave_mats (m n p q : ℕ) (h1 : m * n = p * q) (h2 : ∀ k, k = n → n * 2 = k * 2) :
  (8 * 2 = 16) :=
by
  -- This is where we would traditionally include the proof steps.
  sorry

end weave_mats_l40_40421


namespace maximizing_sum_of_arithmetic_sequence_l40_40865

theorem maximizing_sum_of_arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h_decreasing : ∀ n, a n > a (n + 1))
  (h_sum : S 5 = S 10) :
  (S 7 >= S n ∧ S 8 >= S n) := sorry

end maximizing_sum_of_arithmetic_sequence_l40_40865


namespace simplify_fraction_l40_40524

open Real

theorem simplify_fraction (x : ℝ) : (3 + 2 * sin x + 2 * cos x) / (3 + 2 * sin x - 2 * cos x) = 3 / 5 + (2 / 5) * cos x :=
by
  sorry

end simplify_fraction_l40_40524


namespace top_card_is_red_l40_40054

noncomputable def standard_deck (ranks : ℕ) (suits : ℕ) : ℕ := ranks * suits

def red_cards_in_deck (hearts : ℕ) (diamonds : ℕ) : ℕ := hearts + diamonds

noncomputable def probability_red_card (red_cards : ℕ) (total_cards : ℕ) : ℚ := red_cards / total_cards

theorem top_card_is_red (hearts diamonds spades clubs : ℕ) (deck_size : ℕ)
  (H1 : hearts = 13) (H2 : diamonds = 13) (H3 : spades = 13) (H4 : clubs = 13) (H5 : deck_size = 52):
  probability_red_card (red_cards_in_deck hearts diamonds) deck_size = 1/2 :=
by 
  sorry

end top_card_is_red_l40_40054


namespace johns_donation_l40_40764

theorem johns_donation (A : ℝ) (T : ℝ) (J : ℝ) (h1 : A + 0.5 * A = 75) (h2 : T = 3 * A) 
                       (h3 : (T + J) / 4 = 75) : J = 150 := by
  sorry

end johns_donation_l40_40764


namespace find_roots_l40_40960

theorem find_roots (x : ℝ) : x^2 - 2 * x - 2 / x + 1 / x^2 - 13 = 0 ↔ 
  (x = (-3 + Real.sqrt 5) / 2 ∨ x = (-3 - Real.sqrt 5) / 2 ∨ x = (5 + Real.sqrt 21) / 2 ∨ x = (5 - Real.sqrt 21) / 2) := by
  sorry

end find_roots_l40_40960


namespace find_x_l40_40640

theorem find_x (x : ℝ) (h : (3 * x - 7) / 4 = 14) : x = 21 :=
sorry

end find_x_l40_40640


namespace find_k_l40_40031

-- Define the sum of even integers from 2 to 2k
def sum_even_integers (k : ℕ) : ℕ :=
  2 * (k * (k + 1)) / 2

-- Define the condition that this sum equals 132
def sum_condition (t : ℕ) (k : ℕ) : Prop :=
  sum_even_integers k = t

theorem find_k (k : ℕ) (t : ℕ) (h₁ : t = 132) (h₂ : sum_condition t k) : k = 11 := by
  sorry

end find_k_l40_40031


namespace max_books_john_can_buy_l40_40587

-- Define the key variables and conditions
def johns_money : ℕ := 3745
def book_cost : ℕ := 285
def sales_tax_rate : ℚ := 0.05

-- Define the total cost per book including tax
def total_cost_per_book : ℝ := book_cost + book_cost * sales_tax_rate

-- Define the inequality problem
theorem max_books_john_can_buy : ∃ (x : ℕ), 300 * x ≤ johns_money ∧ 300 * (x + 1) > johns_money :=
by
  sorry

end max_books_john_can_buy_l40_40587


namespace second_projection_at_given_distance_l40_40696

structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

structure Line :=
  (point : Point)
  (direction : Point) -- Assume direction is given as a vector

def is_parallel (line1 line2 : Line) : Prop :=
  -- Function to check if two lines are parallel
  sorry

def distance (point1 point2 : Point) : ℝ := 
  -- Function to compute the distance between two points
  sorry

def first_projection_exists (M : Point) (a : Line) : Prop :=
  -- Check the projection outside the line a
  sorry

noncomputable def second_projection
  (M : Point)
  (a : Line)
  (d : ℝ)
  (h_parallel : is_parallel a (Line.mk ⟨0, 0, 0⟩ ⟨1, 0, 0⟩))
  (h_projection : first_projection_exists M a) :
  Point :=
  sorry

theorem second_projection_at_given_distance
  (M : Point)
  (a : Line)
  (d : ℝ)
  (h_parallel : is_parallel a (Line.mk ⟨0, 0, 0⟩ ⟨1, 0, 0⟩))
  (h_projection : first_projection_exists M a) :
  distance (second_projection M a d h_parallel h_projection) a.point = d :=
  sorry

end second_projection_at_given_distance_l40_40696


namespace no_sum_14_l40_40554

theorem no_sum_14 (x y : ℤ) (h : x * y + 4 = 40) : x + y ≠ 14 :=
by sorry

end no_sum_14_l40_40554


namespace solve_equation_x_squared_eq_16x_l40_40556

theorem solve_equation_x_squared_eq_16x :
  ∀ x : ℝ, x^2 = 16 * x ↔ (x = 0 ∨ x = 16) :=
by 
  intro x
  -- Complete proof here
  sorry

end solve_equation_x_squared_eq_16x_l40_40556


namespace range_of_m_l40_40347

noncomputable def p (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 10
noncomputable def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0 ∧ m > 0

theorem range_of_m (m : ℝ) (h₀ : m > 0) (h₁ : ∀ x : ℝ, q x m → p x) : m ≥ 9 :=
sorry

end range_of_m_l40_40347


namespace retirement_year_l40_40499

-- Define the basic conditions
def rule_of_70 (age: ℕ) (years_of_employment: ℕ) : Prop :=
  age + years_of_employment ≥ 70

def age_in_hiring_year : ℕ := 32
def hiring_year : ℕ := 1987

theorem retirement_year : ∃ y: ℕ, rule_of_70 (age_in_hiring_year + y) y ∧ (hiring_year + y = 2006) :=
  sorry

end retirement_year_l40_40499


namespace total_kids_in_lawrence_county_l40_40035

theorem total_kids_in_lawrence_county :
  ∀ (h c T : ℕ), h = 274865 → c = 38608 → T = h + c → T = 313473 :=
by
  intros h c T h_eq c_eq T_eq
  rw [h_eq, c_eq] at T_eq
  exact T_eq

end total_kids_in_lawrence_county_l40_40035


namespace find_balloons_given_to_Fred_l40_40896

variable (x : ℝ)
variable (Sam_initial_balance : ℝ := 46.0)
variable (Dan_balance : ℝ := 16.0)
variable (total_balance : ℝ := 52.0)

theorem find_balloons_given_to_Fred
  (h : Sam_initial_balance - x + Dan_balance = total_balance) :
  x = 10.0 :=
by
  sorry

end find_balloons_given_to_Fred_l40_40896


namespace find_s_t_l40_40673

theorem find_s_t 
  (FG GH EH : ℝ)
  (angleE angleF : ℝ)
  (h1 : FG = 10)
  (h2 : GH = 15)
  (h3 : EH = 12)
  (h4 : angleE = 45)
  (h5 : angleF = 45)
  (s t : ℕ)
  (h6 : 12 + 7.5 * Real.sqrt 2 = s + Real.sqrt t) :
  s + t = 5637 :=
sorry

end find_s_t_l40_40673


namespace jerry_feathers_count_l40_40768

noncomputable def hawk_feathers : ℕ := 6
noncomputable def eagle_feathers : ℕ := 17 * hawk_feathers
noncomputable def total_feathers : ℕ := hawk_feathers + eagle_feathers
noncomputable def remaining_feathers_after_sister : ℕ := total_feathers - 10
noncomputable def jerry_feathers_left : ℕ := remaining_feathers_after_sister / 2

theorem jerry_feathers_count : jerry_feathers_left = 49 :=
  by
  sorry

end jerry_feathers_count_l40_40768


namespace range_of_a_if_p_true_l40_40303

theorem range_of_a_if_p_true : 
  (∃ (x : ℝ), 1 ≤ x ∧ x ≤ 9 ∧ x^2 - a * x + 36 ≤ 0) → a ≥ 12 :=
sorry

end range_of_a_if_p_true_l40_40303


namespace range_of_k_for_distinct_roots_l40_40370
-- Import necessary libraries

-- Define the quadratic equation and conditions
noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Define the property of having distinct real roots
def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  discriminant a b c > 0

-- Define the specific problem instance and range condition
theorem range_of_k_for_distinct_roots (k : ℝ) :
  has_two_distinct_real_roots 1 2 k ↔ k < 1 :=
by
  sorry

end range_of_k_for_distinct_roots_l40_40370


namespace kids_have_equal_eyes_l40_40343

theorem kids_have_equal_eyes (mom_eyes dad_eyes kids_num total_eyes kids_eyes : ℕ) 
  (h_mom_eyes : mom_eyes = 1) 
  (h_dad_eyes : dad_eyes = 3) 
  (h_kids_num : kids_num = 3) 
  (h_total_eyes : total_eyes = 16) 
  (h_family_eyes : mom_eyes + dad_eyes + kids_num * kids_eyes = total_eyes) :
  kids_eyes = 4 :=
by
  sorry

end kids_have_equal_eyes_l40_40343


namespace unique_seq_l40_40291

def seq (a : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j

theorem unique_seq (a : ℕ → ℕ) (h1 : ∀ n, 0 < a n) : 
  seq a ↔ (∀ n, a n = n) := 
by
  intros
  sorry

end unique_seq_l40_40291


namespace joes_mean_score_is_88_83_l40_40658

def joesQuizScores : List ℕ := [88, 92, 95, 81, 90, 87]

noncomputable def mean (lst : List ℕ) : ℝ := (lst.sum : ℝ) / lst.length

theorem joes_mean_score_is_88_83 :
  mean joesQuizScores = 88.83 := 
sorry

end joes_mean_score_is_88_83_l40_40658


namespace solve_trig_problem_l40_40125

-- Definition of the given problem for trigonometric identities
def problem_statement : Prop :=
  (1 - Real.tan (Real.pi / 12)) / (1 + Real.tan (Real.pi / 12)) = Real.sqrt 3 / 3

theorem solve_trig_problem : problem_statement :=
  by
  sorry -- No proof is needed here

end solve_trig_problem_l40_40125


namespace total_spent_l40_40950

-- Constants representing the conditions from the problem
def cost_per_deck : ℕ := 8
def tom_decks : ℕ := 3
def friend_decks : ℕ := 5

-- Theorem stating the total amount spent by Tom and his friend
theorem total_spent : tom_decks * cost_per_deck + friend_decks * cost_per_deck = 64 := by
  sorry

end total_spent_l40_40950


namespace circle_condition_l40_40189

def represents_circle (m : ℝ) : Prop :=
  ∀ x y : ℝ, (x + 1/2)^2 + (y + m)^2 = 5/4 - m

theorem circle_condition (m : ℝ) : represents_circle m ↔ m < 5/4 :=
by sorry

end circle_condition_l40_40189


namespace mean_total_sample_variance_total_sample_expected_final_score_l40_40061

section SeagrassStatistics

variables (m n : ℕ) (mean_x mean_y: ℝ) (var_x var_y: ℝ) (A_win_A B_win_A : ℝ)

-- Assumptions from the conditions
variable (hp1 : m = 12)
variable (hp2 : mean_x = 18)
variable (hp3 : var_x = 19)
variable (hp4 : n = 18)
variable (hp5 : mean_y = 36)
variable (hp6 : var_y = 70)
variable (hp7 : A_win_A = 3 / 5)
variable (hp8 : B_win_A = 1 / 2)

-- Statements to prove
theorem mean_total_sample (m n : ℕ) (mean_x mean_y : ℝ) : 
  m * mean_x + n * mean_y = (m + n) * 28.8 := sorry

theorem variance_total_sample (m n : ℕ) (mean_x mean_y var_x var_y : ℝ) :
  m * (var_x + (mean_x - 28.8)^2) + n * (var_y + (mean_y - 28.8)^2) = (m + n) * 127.36 := sorry

theorem expected_final_score (A_win_A B_win_A : ℝ) :
  2 * ((6/25) * 1 + (15/25) * 2 + (4/25) * 0) = 36 / 25 := sorry

end SeagrassStatistics

end mean_total_sample_variance_total_sample_expected_final_score_l40_40061


namespace find_initial_candies_l40_40258

-- Definitions for the conditions
def initial_candies (x : ℕ) : Prop :=
  (3 * x) % 4 = 0 ∧
  (x % 2) = 0 ∧
  ∃ (k : ℕ), 2 ≤ k ∧ k ≤ 6 ∧ (1 * x) / 2 - 20 - k = 4

-- Theorems we need to prove
theorem find_initial_candies (x : ℕ) (h : initial_candies x) : x = 52 ∨ x = 56 ∨ x = 60 :=
sorry

end find_initial_candies_l40_40258


namespace height_of_C_l40_40251

noncomputable def height_A_B_C (h_A h_B h_C : ℝ) : Prop := 
  (h_A + h_B + h_C) / 3 = 143 ∧ 
  h_A + 4.5 = (h_B + h_C) / 2 ∧ 
  h_B = h_C + 3

theorem height_of_C (h_A h_B h_C : ℝ) (h : height_A_B_C h_A h_B h_C) : h_C = 143 :=
  sorry

end height_of_C_l40_40251


namespace iceberg_submersion_l40_40362

theorem iceberg_submersion (V_total V_immersed S_total S_submerged : ℝ) :
  convex_polyhedron ∧ floating_on_sea ∧
  V_total > 0 ∧ V_immersed > 0 ∧ S_total > 0 ∧ S_submerged > 0 ∧
  (V_immersed / V_total >= 0.90) ∧ ((S_total - S_submerged) / S_total >= 0.50) :=
sorry

end iceberg_submersion_l40_40362


namespace mary_age_l40_40922

theorem mary_age (M F : ℕ) (h1 : F = 4 * M) (h2 : F - 3 = 5 * (M - 3)) : M = 12 :=
by
  sorry

end mary_age_l40_40922


namespace inequality_solution_l40_40368

theorem inequality_solution (x : ℝ) (h : 3 * x + 2 ≠ 0) : 
  3 - 2/(3 * x + 2) < 5 ↔ x > -2/3 := 
sorry

end inequality_solution_l40_40368


namespace find_sample_size_l40_40672

theorem find_sample_size
  (teachers : ℕ := 200)
  (male_students : ℕ := 1200)
  (female_students : ℕ := 1000)
  (sampled_females : ℕ := 80)
  (total_people := teachers + male_students + female_students)
  (ratio : sampled_females / female_students = n / total_people)
  : n = 192 := 
by
  sorry

end find_sample_size_l40_40672


namespace cherries_used_l40_40371

theorem cherries_used (initial remaining used : ℕ) (h_initial : initial = 77) (h_remaining : remaining = 17) (h_used : used = initial - remaining) : used = 60 :=
by
  rw [h_initial, h_remaining] at h_used
  simp at h_used
  exact h_used

end cherries_used_l40_40371


namespace combined_molecular_weight_l40_40947

theorem combined_molecular_weight 
  (atomic_weight_N : ℝ)
  (atomic_weight_O : ℝ)
  (atomic_weight_H : ℝ)
  (atomic_weight_C : ℝ)
  (moles_N2O3 : ℝ)
  (moles_H2O : ℝ)
  (moles_CO2 : ℝ)
  (molecular_weight_N2O3 : ℝ)
  (molecular_weight_H2O : ℝ)
  (molecular_weight_CO2 : ℝ)
  (weight_N2O3 : ℝ)
  (weight_H2O : ℝ)
  (weight_CO2 : ℝ)
  : 
  moles_N2O3 = 4 →
  moles_H2O = 3.5 →
  moles_CO2 = 2 →
  atomic_weight_N = 14.01 →
  atomic_weight_O = 16.00 →
  atomic_weight_H = 1.01 →
  atomic_weight_C = 12.01 →
  molecular_weight_N2O3 = (2 * atomic_weight_N) + (3 * atomic_weight_O) →
  molecular_weight_H2O = (2 * atomic_weight_H) + atomic_weight_O →
  molecular_weight_CO2 = atomic_weight_C + (2 * atomic_weight_O) →
  weight_N2O3 = moles_N2O3 * molecular_weight_N2O3 →
  weight_H2O = moles_H2O * molecular_weight_H2O →
  weight_CO2 = moles_CO2 * molecular_weight_CO2 →
  weight_N2O3 + weight_H2O + weight_CO2 = 455.17 :=
by 
  intros;
  sorry

end combined_molecular_weight_l40_40947


namespace bacon_cost_l40_40306

namespace PancakeBreakfast

def cost_of_stack_pancakes : ℝ := 4.0
def stacks_sold : ℕ := 60
def slices_bacon_sold : ℕ := 90
def total_revenue : ℝ := 420.0

theorem bacon_cost (B : ℝ) 
  (h1 : stacks_sold * cost_of_stack_pancakes + slices_bacon_sold * B = total_revenue) : 
  B = 2 :=
  by {
    sorry
  }

end PancakeBreakfast

end bacon_cost_l40_40306


namespace circle_center_x_coordinate_eq_l40_40004

theorem circle_center_x_coordinate_eq (a : ℝ) (h : (∃ k : ℝ, ∀ x y : ℝ, x^2 + y^2 - a * x = k) ∧ (1 = a / 2)) : a = 2 :=
sorry

end circle_center_x_coordinate_eq_l40_40004


namespace profit_percentage_is_ten_l40_40200

-- Define the cost price (CP) and selling price (SP) as constants
def CP : ℝ := 90.91
def SP : ℝ := 100

-- Define a theorem to prove the profit percentage is 10%
theorem profit_percentage_is_ten : ((SP - CP) / CP) * 100 = 10 := 
by 
  -- Skip the proof.
  sorry

end profit_percentage_is_ten_l40_40200


namespace jennie_total_rental_cost_l40_40941

-- Definition of the conditions in the problem
def daily_rate : ℕ := 30
def weekly_rate : ℕ := 190
def days_rented : ℕ := 11
def first_week_days : ℕ := 7

-- Proof statement which translates the problem to Lean
theorem jennie_total_rental_cost : (weekly_rate + (days_rented - first_week_days) * daily_rate) = 310 := by
  sorry

end jennie_total_rental_cost_l40_40941


namespace probability_target_hit_l40_40384

theorem probability_target_hit {P_A P_B : ℚ}
  (hA : P_A = 1 / 2) 
  (hB : P_B = 1 / 3) 
  : (1 - (1 - P_A) * (1 - P_B)) = 2 / 3 := 
by
  sorry

end probability_target_hit_l40_40384


namespace moles_of_HCl_formed_l40_40798

-- Define the reaction
def balancedReaction (CH4 Cl2 CH3Cl HCl : ℕ) : Prop :=
  CH4 + Cl2 = CH3Cl + HCl

-- Number of moles given
def molesCH4 := 2
def molesCl2 := 4

-- Theorem statement
theorem moles_of_HCl_formed :
  ∀ CH4 Cl2 CH3Cl HCl : ℕ, balancedReaction CH4 Cl2 CH3Cl HCl →
  CH4 = molesCH4 →
  Cl2 = molesCl2 →
  HCl = 2 := sorry

end moles_of_HCl_formed_l40_40798


namespace bus_stop_time_l40_40761

noncomputable def time_stopped_per_hour (excl_speed incl_speed : ℕ) : ℕ :=
  60 * (excl_speed - incl_speed) / excl_speed

theorem bus_stop_time:
  time_stopped_per_hour 54 36 = 20 :=
by
  sorry

end bus_stop_time_l40_40761


namespace ratio_of_sizes_l40_40724

-- Defining Anna's size
def anna_size : ℕ := 2

-- Defining Becky's size as three times Anna's size
def becky_size : ℕ := 3 * anna_size

-- Defining Ginger's size
def ginger_size : ℕ := 8

-- Defining the goal statement
theorem ratio_of_sizes : (ginger_size : ℕ) / (becky_size : ℕ) = 4 / 3 :=
by
  sorry

end ratio_of_sizes_l40_40724


namespace minimal_range_of_observations_l40_40295

variable {x1 x2 x3 x4 x5 : ℝ}

def arithmetic_mean (x1 x2 x3 x4 x5 : ℝ) : Prop :=
  (x1 + x2 + x3 + x4 + x5) / 5 = 8

def median (x1 x2 x3 x4 x5 : ℝ) : Prop :=
  x3 = 10 ∧ x1 ≤ x2 ∧ x2 ≤ x3 ∧ x3 ≤ x4 ∧ x4 ≤ x5

theorem minimal_range_of_observations 
  (h_mean : arithmetic_mean x1 x2 x3 x4 x5)
  (h_median : median x1 x2 x3 x4 x5) : 
  ∃ x1 x2 x3 x4 x5 : ℝ, (x1 + x2 + x3 + x4 + x5) = 40 ∧ x3 = 10 ∧ x1 ≤ x2 ∧ x2 ≤ x3 ∧ x3 ≤ x4 ∧ x4 ≤ x5 ∧ (x5 - x1) = 5 :=
by 
  sorry

end minimal_range_of_observations_l40_40295


namespace number_of_customers_per_month_l40_40883

-- Define the constants and conditions
def price_lettuce_per_head : ℝ := 1
def price_tomato_per_piece : ℝ := 0.5
def num_lettuce_per_customer : ℕ := 2
def num_tomato_per_customer : ℕ := 4
def monthly_sales : ℝ := 2000

-- Calculate the cost per customer
def cost_per_customer : ℝ := 
  (num_lettuce_per_customer * price_lettuce_per_head) + 
  (num_tomato_per_customer * price_tomato_per_piece)

-- Prove the number of customers per month
theorem number_of_customers_per_month : monthly_sales / cost_per_customer = 500 :=
  by
    -- Here, we would write the proof steps
    sorry

end number_of_customers_per_month_l40_40883


namespace polynomial_bound_l40_40452

noncomputable def P (x : ℝ) (a b c d : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem polynomial_bound (a b c d : ℝ) (hP : ∀ x : ℝ, |x| < 1 → |P x a b c d| ≤ 1) : 
  |a| + |b| + |c| + |d| ≤ 7 := 
sorry

end polynomial_bound_l40_40452


namespace moles_C2H6_for_HCl_l40_40899

theorem moles_C2H6_for_HCl 
  (form_HCl : ℕ)
  (moles_Cl2 : ℕ)
  (reaction : ℕ) : 
  (6 * (reaction * moles_Cl2)) = form_HCl * (6 * reaction) :=
by
  -- The necessary proof steps will go here
  sorry

end moles_C2H6_for_HCl_l40_40899


namespace smallest_integer_N_l40_40142

theorem smallest_integer_N : ∃ (N : ℕ), 
  (∀ (a : ℕ → ℕ), ((∀ (i : ℕ), i < 125 -> a i > 0 ∧ a i ≤ N) ∧
  (∀ (i : ℕ), 1 ≤ i ∧ i < 124 → a i > (a (i - 1) + a (i + 1)) / 2) ∧
  (∀ (i j : ℕ), i < 125 ∧ j < 125 ∧ i ≠ j → a i ≠ a j)) → N = 2016) :=
sorry

end smallest_integer_N_l40_40142


namespace gcd_of_sum_and_product_l40_40233

theorem gcd_of_sum_and_product (x y : ℕ) (h1 : x + y = 1130) (h2 : x * y = 100000) : Int.gcd x y = 2 := 
sorry

end gcd_of_sum_and_product_l40_40233


namespace joan_needs_more_flour_l40_40954

-- Definitions for the conditions
def total_flour : ℕ := 7
def flour_added : ℕ := 3

-- The theorem stating the proof problem
theorem joan_needs_more_flour : total_flour - flour_added = 4 :=
by
  sorry

end joan_needs_more_flour_l40_40954


namespace negative_half_power_zero_l40_40216

theorem negative_half_power_zero : (- (1 / 2)) ^ 0 = 1 :=
by
  sorry

end negative_half_power_zero_l40_40216


namespace fraction_value_l40_40893

def op_at (a b : ℤ) : ℤ := a * b - b ^ 2
def op_sharp (a b : ℤ) : ℤ := a + b - a * b ^ 2

theorem fraction_value : (op_at 7 3) / (op_sharp 7 3) = -12 / 53 :=
by
  sorry

end fraction_value_l40_40893


namespace triangle_shape_area_l40_40620

theorem triangle_shape_area (a b : ℕ) (area_small area_middle area_large : ℕ) :
  a = 2 →
  b = 2 →
  area_small = (1 / 2) * a * b →
  area_middle = 2 * area_small →
  area_large = 2 * area_middle →
  area_small + area_middle + area_large = 14 :=
by
  intros
  sorry

end triangle_shape_area_l40_40620


namespace range_of_a_l40_40344

def quadratic_function (a x : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x < y → quadratic_function a x ≥ quadratic_function a y ∧ y ≤ 4) →
  a ≤ -5 :=
by sorry

end range_of_a_l40_40344


namespace original_weight_calculation_l40_40722

-- Conditions
variable (postProcessingWeight : ℝ) (originalWeight : ℝ)
variable (lostPercentage : ℝ)

-- Problem Statement
theorem original_weight_calculation
  (h1 : postProcessingWeight = 240)
  (h2 : lostPercentage = 0.40) :
  originalWeight = 400 :=
sorry

end original_weight_calculation_l40_40722


namespace total_kids_played_l40_40926

-- Definitions based on conditions
def kidsMonday : Nat := 17
def kidsTuesday : Nat := 15
def kidsWednesday : Nat := 2

-- Total kids calculation
def totalKids : Nat := kidsMonday + kidsTuesday + kidsWednesday

-- Theorem to prove
theorem total_kids_played (Julia : Prop) : totalKids = 34 :=
by
  -- Using sorry to skip the proof
  sorry

end total_kids_played_l40_40926


namespace multiplication_problem_l40_40607

theorem multiplication_problem :
  250 * 24.98 * 2.498 * 1250 = 19484012.5 := by
  sorry

end multiplication_problem_l40_40607


namespace math_problem_proof_l40_40792

noncomputable def ellipse_equation : Prop := 
  let e := (Real.sqrt 2) / 2
  ∃ (a b : ℝ), 0 < a ∧ a > b ∧ e = (Real.sqrt 2) / 2 ∧ 
    (∀ x y, (x^2) / (a^2) + (y^2) / (b^2) = 1 ↔ x^2 / 2 + y^2 = 1)

noncomputable def fixed_point_exist : Prop :=
  let S := (0, 1/3) 
  ∀ k : ℝ, ∃ A B : ℝ × ℝ, 
    let M := (0, 1)
    ( 
        (A.1, A.2) ∈ {P : ℝ × ℝ | (P.1^2) / 2 + P.2^2 = 1} ∧ 
        (B.1, B.2) ∈ {P : ℝ × ℝ | (P.1^2) / 2 + P.2^2 = 1} ∧ 
        (S.2 = k * S.1 - 1 / 3) ∧ 
        ((A.1 - M.1)^2 + (A.2 - M.2)^2) + ((B.1 - M.1)^2 + (B.2 - M.2)^2) = ((A.1 - B.1)^2 + (A.2 - M.2)^2) / 2)

theorem math_problem_proof : ellipse_equation ∧ fixed_point_exist := sorry

end math_problem_proof_l40_40792


namespace log_roots_equivalence_l40_40406

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 5 / Real.log 3
noncomputable def c : ℝ := Real.log 2 / Real.log 5

theorem log_roots_equivalence :
  (x : ℝ) → (x = a ∨ x = b ∨ x = c) ↔ (x^3 - (a + b + c)*x^2 + (a*b + b*c + c*a)*x - a*b*c = 0) := by
  sorry

end log_roots_equivalence_l40_40406


namespace roots_of_equation_l40_40168

theorem roots_of_equation :
  ∀ x : ℝ, (21 / (x^2 - 9) - 3 / (x - 3) = 1) ↔ (x = 3 ∨ x = -7) :=
by
  sorry

end roots_of_equation_l40_40168


namespace find_e_value_l40_40860

theorem find_e_value : (14 ^ 2) * (5 ^ 3) * 568 = 13916000 := by
  sorry

end find_e_value_l40_40860


namespace complement_in_U_l40_40579

def A : Set ℝ := { x : ℝ | |x - 1| > 3 }
def U : Set ℝ := Set.univ

theorem complement_in_U :
  (U \ A) = { x : ℝ | -2 ≤ x ∧ x ≤ 4 } :=
by
  sorry

end complement_in_U_l40_40579


namespace spadesuit_evaluation_l40_40766

-- Define the operation
def spadesuit (x y : ℚ) : ℚ := x - (1 / y)

-- Prove the main statement
theorem spadesuit_evaluation : spadesuit 3 (spadesuit 3 (3 / 2)) = 18 / 7 :=
by
  sorry

end spadesuit_evaluation_l40_40766


namespace find_angle_C_l40_40887

theorem find_angle_C (A B C : ℝ) (h1 : A = 88) (h2 : B - C = 20) (angle_sum : A + B + C = 180) : C = 36 :=
by
  sorry

end find_angle_C_l40_40887


namespace peter_total_dogs_l40_40072

def num_german_shepherds_sam : ℕ := 3
def num_french_bulldogs_sam : ℕ := 4
def num_german_shepherds_peter := 3 * num_german_shepherds_sam
def num_french_bulldogs_peter := 2 * num_french_bulldogs_sam

theorem peter_total_dogs : num_german_shepherds_peter + num_french_bulldogs_peter = 17 :=
by {
  -- adding proofs later
  sorry
}

end peter_total_dogs_l40_40072


namespace solution_set_l40_40219

-- Define the system of equations
def system_of_equations (x y : ℤ) : Prop :=
  4 * x^2 = y^2 + 2 * y + 4 ∧
  (2 * x)^2 - (y + 1)^2 = 3 ∧
  (2 * x - (y + 1)) * (2 * x + (y + 1)) = 3

-- Prove that the solutions to the system are the set we expect
theorem solution_set : 
  { (x, y) : ℤ × ℤ | system_of_equations x y } = { (1, 0), (1, -2), (-1, 0), (-1, -2) } := 
by 
  -- Proof omitted
  sorry

end solution_set_l40_40219


namespace family_trip_eggs_l40_40906

theorem family_trip_eggs (adults girls boys : ℕ)
  (eggs_per_adult : ℕ) (eggs_per_girl : ℕ) (extra_eggs_for_boy : ℕ) :
  adults = 3 →
  eggs_per_adult = 3 →
  girls = 7 →
  eggs_per_girl = 1 →
  boys = 10 →
  extra_eggs_for_boy = 1 →
  (adults * eggs_per_adult + girls * eggs_per_girl + boys * (eggs_per_girl + extra_eggs_for_boy)) = 36 :=
by
  intros
  sorry

end family_trip_eggs_l40_40906


namespace gardening_project_cost_l40_40221

def cost_rose_bushes (number_of_bushes: ℕ) (cost_per_bush: ℕ) : ℕ := number_of_bushes * cost_per_bush
def cost_gardener (hourly_rate: ℕ) (hours_per_day: ℕ) (days: ℕ) : ℕ := hourly_rate * hours_per_day * days
def cost_soil (cubic_feet: ℕ) (cost_per_cubic_foot: ℕ) : ℕ := cubic_feet * cost_per_cubic_foot

theorem gardening_project_cost :
  cost_rose_bushes 20 150 + cost_gardener 30 5 4 + cost_soil 100 5 = 4100 :=
by
  sorry

end gardening_project_cost_l40_40221


namespace days_needed_to_wash_all_towels_l40_40516

def towels_per_hour : ℕ := 7
def hours_per_day : ℕ := 2
def total_towels : ℕ := 98

theorem days_needed_to_wash_all_towels :
  (total_towels / (towels_per_hour * hours_per_day)) = 7 :=
by
  sorry

end days_needed_to_wash_all_towels_l40_40516


namespace tan_390_correct_l40_40119

-- We assume basic trigonometric functions and their properties
noncomputable def tan_390_equals_sqrt3_div3 : Prop :=
  Real.tan (390 * Real.pi / 180) = Real.sqrt 3 / 3

theorem tan_390_correct : tan_390_equals_sqrt3_div3 :=
  by
  -- Proof is omitted
  sorry

end tan_390_correct_l40_40119


namespace triangle_AC_length_l40_40059

open Real

theorem triangle_AC_length (A : ℝ) (AB AC S : ℝ) (h1 : A = π / 3) (h2 : AB = 2) (h3 : S = sqrt 3 / 2) : AC = 1 :=
by
  sorry

end triangle_AC_length_l40_40059


namespace liters_conversion_hours_to_days_cubic_meters_to_cubic_cm_l40_40128

-- Define the conversions and the corresponding proofs
theorem liters_conversion : 8.32 = 8 + 320 / 1000 := sorry

theorem hours_to_days : 6 = 1 / 4 * 24 := sorry

theorem cubic_meters_to_cubic_cm : 0.75 * 10^6 = 750000 := sorry

end liters_conversion_hours_to_days_cubic_meters_to_cubic_cm_l40_40128


namespace find_MN_sum_l40_40726

noncomputable def M : ℝ := sorry -- Placeholder for the actual non-zero solution M
noncomputable def N : ℝ := M ^ 2

theorem find_MN_sum :
  (M^2 = N) ∧ (Real.log N / Real.log M = Real.log M / Real.log N) ∧ (M ≠ N) ∧ (M ≠ 1) ∧ (N ≠ 1) → (M + N = 6) :=
by
  intros h
  exact sorry -- Will be replaced by the actual proof


end find_MN_sum_l40_40726


namespace green_ish_count_l40_40097

theorem green_ish_count (total : ℕ) (blue_ish : ℕ) (both : ℕ) (neither : ℕ) (green_ish : ℕ) :
  total = 150 ∧ blue_ish = 90 ∧ both = 40 ∧ neither = 30 → green_ish = 70 :=
by
  sorry

end green_ish_count_l40_40097


namespace factorial_square_ge_power_l40_40089

theorem factorial_square_ge_power (n : ℕ) (h : n ≥ 1) : (n!)^2 ≥ n^n := 
by sorry

end factorial_square_ge_power_l40_40089


namespace slope_of_parallel_line_l40_40100

theorem slope_of_parallel_line (a b c : ℝ) (h: 3*a + 6*b = -24) :
  ∃ m : ℝ, (a * 3 + b * 6 = c) → m = -1/2 :=
by
  sorry

end slope_of_parallel_line_l40_40100


namespace series_sum_eq_neg_one_l40_40360

   noncomputable def sum_series : ℝ :=
     ∑' k : ℕ, if k = 0 then 0 else (12 ^ k) / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1)))

   theorem series_sum_eq_neg_one : sum_series = -1 :=
   sorry
   
end series_sum_eq_neg_one_l40_40360


namespace printing_company_proportion_l40_40951

theorem printing_company_proportion (x y : ℕ) :
  (28*x + 42*y) / (28*x) = 5/3 → x / y = 9 / 4 := by
  sorry

end printing_company_proportion_l40_40951


namespace num_two_digit_multiples_5_and_7_l40_40913

/-- 
    Theorem: There are exactly 2 positive two-digit integers that are multiples of both 5 and 7.
-/
theorem num_two_digit_multiples_5_and_7 : 
  ∃ (count : ℕ), count = 2 ∧ ∀ (n : ℕ), (10 ≤ n ∧ n ≤ 99) → 
    (n % 5 = 0 ∧ n % 7 = 0) ↔ (n = 35 ∨ n = 70) := 
by
  sorry

end num_two_digit_multiples_5_and_7_l40_40913


namespace average_of_remaining_two_numbers_l40_40312

theorem average_of_remaining_two_numbers (S S3 : ℝ) (h_avg5 : S / 5 = 8) (h_avg3 : S3 / 3 = 4) : S / 5 = 8 ∧ S3 / 3 = 4 → (S - S3) / 2 = 14 :=
by 
  sorry

end average_of_remaining_two_numbers_l40_40312


namespace selection_ways_l40_40608

/-- 
A math interest group in a vocational school consists of 4 boys and 3 girls. 
If 3 students are randomly selected from these 7 students to participate in a math competition, 
and the selection must include both boys and girls, then the number of different ways to select the 
students is 30.
-/
theorem selection_ways (B G : ℕ) (students : ℕ) (selections : ℕ) (condition_boys_girls : B = 4 ∧ G = 3)
  (condition_students : students = B + G) (condition_selections : selections = 3) :
  (B = 4 ∧ G = 3 ∧ students = 7 ∧ selections = 3) → 
  ∃ (res : ℕ), res = 30 :=
by
  sorry

end selection_ways_l40_40608


namespace smallest_number_divisible_l40_40338

theorem smallest_number_divisible (x y : ℕ) (h : x + y = 4728) 
  (h1 : (x + y) % 27 = 0) 
  (h2 : (x + y) % 35 = 0) 
  (h3 : (x + y) % 25 = 0) 
  (h4 : (x + y) % 21 = 0) : 
  x = 4725 := by 
  sorry

end smallest_number_divisible_l40_40338


namespace probability_of_multiple_of_3_is_1_5_l40_40856

-- Definition of the problem conditions
def digits : List ℕ := [1, 2, 3, 4, 5]

-- Function to calculate the probability
noncomputable def probability_of_multiple_of_3 : ℚ := 
  let total_permutations := (Nat.factorial 5) / (Nat.factorial (5 - 4))  -- i.e., 120
  let valid_permutations := Nat.factorial 4  -- i.e., 24, for the valid combination
  valid_permutations / total_permutations 

-- Statement to be proved
theorem probability_of_multiple_of_3_is_1_5 :
  probability_of_multiple_of_3 = 1 / 5 := 
by
  -- Skeleton for the proof
  sorry

end probability_of_multiple_of_3_is_1_5_l40_40856


namespace cos_alpha_third_quadrant_l40_40136

theorem cos_alpha_third_quadrant (α : ℝ) (h1 : Real.sin α = -5 / 13) (h2 : Real.tan α > 0) : Real.cos α = -12 / 13 := 
sorry

end cos_alpha_third_quadrant_l40_40136


namespace possible_values_l40_40908

def expression (m n : ℕ) : ℤ :=
  (m^2 + m * n + n^2) / (m * n - 1)

theorem possible_values (m n : ℕ) (h : m * n ≠ 1) : 
  ∃ (N : ℤ), N = expression m n → N = 0 ∨ N = 4 ∨ N = 7 :=
by
  sorry

end possible_values_l40_40908


namespace tan_15_deg_product_l40_40225

theorem tan_15_deg_product : (1 + Real.tan 15) * (1 + Real.tan 15) = 2.1433 := by
  sorry

end tan_15_deg_product_l40_40225


namespace jackson_money_proof_l40_40173

noncomputable def jackson_money (W : ℝ) := 7 * W
noncomputable def lucy_money (W : ℝ) := 3 * W
noncomputable def ethan_money (W : ℝ) := 3 * W + 20

theorem jackson_money_proof : ∀ (W : ℝ), (W + 7 * W + 3 * W + (3 * W + 20) = 600) → jackson_money W = 290.01 :=
by 
  intros W h
  have total_eq := h
  sorry

end jackson_money_proof_l40_40173


namespace combined_average_speed_l40_40929

theorem combined_average_speed 
    (dA tA dB tB dC tC : ℝ)
    (mile_feet : ℝ)
    (hA : dA = 300) (hTA : tA = 6)
    (hB : dB = 400) (hTB : tB = 8)
    (hC : dC = 500) (hTC : tC = 10)
    (hMileFeet : mile_feet = 5280) :
    (1200 / 5280) / (24 / 3600) = 34.09 := 
by
  sorry

end combined_average_speed_l40_40929


namespace number_of_solutions_l40_40699

theorem number_of_solutions (n : ℕ) (hn : 0 < n) :
  ∃ k : ℕ, k = 2 + 4 * n ∧ (∃ (x y : ℤ), x ^ 2 + 2016 * y ^ 2 = 2017 ^ n) :=
by
  sorry

end number_of_solutions_l40_40699


namespace solution_of_system_l40_40382

theorem solution_of_system 
  (k : ℝ) (x y : ℝ)
  (h1 : (1 : ℝ) = 2 * 1 - 1)
  (h2 : (1 : ℝ) = k * 1)
  (h3 : k ≠ 0)
  (h4 : 2 * x - y = 1)
  (h5 : k * x - y = 0) : 
  x = 1 ∧ y = 1 :=
by
  sorry

end solution_of_system_l40_40382


namespace length_of_garden_side_l40_40175

theorem length_of_garden_side (perimeter : ℝ) (side_length : ℝ) (h1 : perimeter = 112) (h2 : perimeter = 4 * side_length) : 
  side_length = 28 :=
by
  sorry

end length_of_garden_side_l40_40175


namespace correct_option_is_C_l40_40884

-- Define the polynomial expressions and their expected values as functions
def optionA (x : ℝ) : Prop := (x + 2) * (x - 5) = x^2 - 2 * x - 3
def optionB (x : ℝ) : Prop := (x + 3) * (x - 1 / 3) = x^2 + x - 1
def optionC (x : ℝ) : Prop := (x - 2 / 3) * (x + 1 / 2) = x^2 - 1 / 6 * x - 1 / 3
def optionD (x : ℝ) : Prop := (x - 2) * (-x - 2) = x^2 - 4

-- Problem Statement: Verify that the polynomial multiplication in Option C is correct
theorem correct_option_is_C (x : ℝ) : optionC x :=
by
  -- Statement indicating the proof goes here
  sorry

end correct_option_is_C_l40_40884


namespace book_price_l40_40037

theorem book_price (x : ℕ) : 
  9 * x ≤ 1100 ∧ 13 * x ≤ 1500 → x = 123 :=
sorry

end book_price_l40_40037


namespace perimeter_of_equilateral_triangle_l40_40923

-- Defining the conditions
def area_eq_twice_side (s : ℝ) : Prop :=
  (s^2 * Real.sqrt 3) / 4 = 2 * s

-- Defining the proof problem
theorem perimeter_of_equilateral_triangle (s : ℝ) (h : area_eq_twice_side s) : 
  3 * s = 8 * Real.sqrt 3 :=
sorry

end perimeter_of_equilateral_triangle_l40_40923


namespace intersection_of_complements_l40_40003

theorem intersection_of_complements {U S T : Set ℕ}
  (hU : U = {1, 2, 3, 4, 5, 6, 7, 8})
  (hS : S = {1, 3, 5})
  (hT : T = {3, 6}) :
  (U \ S) ∩ (U \ T) = {2, 4, 7, 8} :=
by
  sorry

end intersection_of_complements_l40_40003


namespace nathan_tomato_plants_l40_40645

theorem nathan_tomato_plants (T: ℕ) : 
  5 * 14 + T * 16 = 186 * 7 / 6 + 9 * 10 :=
  sorry

end nathan_tomato_plants_l40_40645


namespace revenue_change_l40_40700

theorem revenue_change (x : ℝ) 
  (increase_in_1996 : ∀ R : ℝ, R * (1 + x/100) > R) 
  (decrease_in_1997 : ∀ R : ℝ, R * (1 + x/100) * (1 - x/100) < R * (1 + x/100)) 
  (decrease_from_1995_to_1997 : ∀ R : ℝ, R * (1 + x/100) * (1 - x/100) = R * 0.96): 
  x = 20 :=
by
  sorry

end revenue_change_l40_40700


namespace complex_magnitude_comparison_l40_40821

open Complex

theorem complex_magnitude_comparison :
  let z1 := (5 : ℂ) + (3 : ℂ) * I
  let z2 := (5 : ℂ) + (4 : ℂ) * I
  abs z1 < abs z2 :=
by 
  let z1 := (5 : ℂ) + (3 : ℂ) * I
  let z2 := (5 : ℂ) + (4 : ℂ) * I
  sorry

end complex_magnitude_comparison_l40_40821


namespace sum_is_integer_l40_40101

theorem sum_is_integer (x y z : ℝ) 
  (h1 : x^2 = y + 2) 
  (h2 : y^2 = z + 2) 
  (h3 : z^2 = x + 2) : 
  x + y + z = 0 :=
  sorry

end sum_is_integer_l40_40101


namespace probability_product_positive_is_5_div_9_l40_40812

noncomputable def probability_positive_product : ℚ :=
  let interval := Set.Icc (-30 : ℝ) 15
  let length_interval := 45
  let length_neg := 30
  let length_pos := 15
  let prob_neg := (length_neg : ℚ) / length_interval
  let prob_pos := (length_pos : ℚ) / length_interval
  let prob_product_pos := prob_neg^2 + prob_pos^2
  prob_product_pos

theorem probability_product_positive_is_5_div_9 :
  probability_positive_product = 5 / 9 :=
by
  sorry

end probability_product_positive_is_5_div_9_l40_40812


namespace large_green_curlers_l40_40130

-- Define the number of total curlers
def total_curlers : ℕ := 16

-- Define the fraction for pink curlers
def pink_fraction : ℕ := 1 / 4

-- Define the number of pink curlers
def pink_curlers : ℕ := pink_fraction * total_curlers

-- Define the number of blue curlers
def blue_curlers : ℕ := 2 * pink_curlers

-- Define the total number of pink and blue curlers
def pink_and_blue_curlers : ℕ := pink_curlers + blue_curlers

-- Define the number of green curlers
def green_curlers : ℕ := total_curlers - pink_and_blue_curlers

-- Theorem stating the number of green curlers is 4
theorem large_green_curlers : green_curlers = 4 := by
  -- Proof would go here
  sorry

end large_green_curlers_l40_40130


namespace tan_alpha_l40_40585

theorem tan_alpha (α : ℝ) (hα1 : α > π / 2) (hα2 : α < π) (h_sin : Real.sin α = 4 / 5) : Real.tan α = - (4 / 3) :=
by 
  sorry

end tan_alpha_l40_40585


namespace triangle_is_right_triangle_l40_40461

theorem triangle_is_right_triangle 
  (A B C : ℝ)
  (h1 : 0 < A ∧ 0 < B ∧ 0 < C)
  (h2 : A + B + C = 180)
  (h3 : A / B = 2 / 3)
  (h4 : A / C = 2 / 5) : 
  A = 36 ∧ B = 54 ∧ C = 90 := 
sorry

end triangle_is_right_triangle_l40_40461


namespace completing_the_square_solution_correct_l40_40744

theorem completing_the_square_solution_correct (x : ℝ) :
  (x^2 + 8 * x + 9 = 0) ↔ ((x + 4)^2 = 7) :=
by
  sorry

end completing_the_square_solution_correct_l40_40744


namespace opposite_of_neg_nine_is_nine_l40_40848

-- Define the predicate for the opposite of a number
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- State the theorem that needs to be proved
theorem opposite_of_neg_nine_is_nine : ∃ x, is_opposite (-9) x ∧ x = 9 := 
by {
  -- Use "sorry" to indicate the proof is not provided
  sorry
}

end opposite_of_neg_nine_is_nine_l40_40848


namespace cards_with_1_count_l40_40285

theorem cards_with_1_count (m k : ℕ) 
  (h1 : k = m + 100) 
  (sum_of_products : (m * (m - 1) / 2) + (k * (k - 1) / 2) - m * k = 1000) : 
  m = 3950 :=
by
  sorry

end cards_with_1_count_l40_40285


namespace prove_a2_a3_a4_sum_l40_40423

theorem prove_a2_a3_a4_sum (a1 a2 a3 a4 a5 : ℝ) (h : ∀ x : ℝ, a1 * (x-1)^4 + a2 * (x-1)^3 + a3 * (x-1)^2 + a4 * (x-1) + a5 = x^4) :
  a2 + a3 + a4 = 14 :=
sorry

end prove_a2_a3_a4_sum_l40_40423


namespace compare_neg_fractions_l40_40691

theorem compare_neg_fractions : - (4 / 3 : ℚ) < - (5 / 4 : ℚ) := 
by sorry

end compare_neg_fractions_l40_40691


namespace gravel_cost_correct_l40_40274

-- Definitions from the conditions
def lawn_length : ℕ := 80
def lawn_breadth : ℕ := 60
def road_width : ℕ := 15
def gravel_cost_per_sq_m : ℕ := 3

-- Calculate areas of the roads
def area_road_length : ℕ := lawn_length * road_width
def area_road_breadth : ℕ := (lawn_breadth - road_width) * road_width

-- Total area to be graveled
def total_area : ℕ := area_road_length + area_road_breadth

-- Total cost
def total_cost : ℕ := total_area * gravel_cost_per_sq_m

-- Prove the total cost is 5625 Rs
theorem gravel_cost_correct : total_cost = 5625 := by
  sorry

end gravel_cost_correct_l40_40274


namespace tetrahedron_parallelepiped_areas_tetrahedron_heights_distances_l40_40434

-- Definition for Part (a)
theorem tetrahedron_parallelepiped_areas 
  (S1 S2 S3 S4 P1 P2 P3 : ℝ)
  (h1 : true)
  (h2 : true) :
  S1^2 + S2^2 + S3^2 + S4^2 = P1^2 + P2^2 + P3^2 := 
sorry

-- Definition for Part (b)
theorem tetrahedron_heights_distances 
  (h1 h2 h3 h4 d1 d2 d3 : ℝ)
  (h : true) :
  (1/(h1^2)) + (1/(h2^2)) + (1/(h3^2)) + (1/(h4^2)) = (1/(d1^2)) + (1/(d2^2)) + (1/(d3^2)) := 
sorry

end tetrahedron_parallelepiped_areas_tetrahedron_heights_distances_l40_40434


namespace square_side_4_FP_length_l40_40668

theorem square_side_4_FP_length (EF GH EP FP GP : ℝ) :
  EF = 4 ∧ GH = 4 ∧ EP = 4 ∧ GP = 4 ∧
  (1 / 2) * EP * 2 = 4 → FP = 2 * Real.sqrt 5 :=
by
  intro h
  sorry

end square_side_4_FP_length_l40_40668


namespace repeating_decimal_to_fraction_l40_40047

theorem repeating_decimal_to_fraction : (0.7 + 23 / 99 / 10) = (62519 / 66000) := by
  sorry

end repeating_decimal_to_fraction_l40_40047


namespace average_probable_weight_l40_40188

-- Define the conditions
def Arun_opinion (w : ℝ) : Prop := 64 < w ∧ w < 72
def Brother_opinion (w : ℝ) : Prop := 60 < w ∧ w < 70
def Mother_opinion (w : ℝ) : Prop := w ≤ 67

-- The proof problem statement
theorem average_probable_weight :
  ∃ (w : ℝ), Arun_opinion w ∧ Brother_opinion w ∧ Mother_opinion w →
  (64 + 67) / 2 = 65.5 :=
by
  sorry

end average_probable_weight_l40_40188


namespace no_integer_solution_l40_40847

theorem no_integer_solution (x y : ℤ) : ¬(x^4 + y^2 = 4 * y + 4) :=
by
  sorry

end no_integer_solution_l40_40847


namespace congruence_equivalence_l40_40140

theorem congruence_equivalence (m n a b : ℤ) (h_coprime : Int.gcd m n = 1) :
  a ≡ b [ZMOD m * n] ↔ (a ≡ b [ZMOD m] ∧ a ≡ b [ZMOD n]) :=
sorry

end congruence_equivalence_l40_40140


namespace tan_80_l40_40985

theorem tan_80 (m : ℝ) (h : Real.cos (100 * Real.pi / 180) = m) :
    Real.tan (80 * Real.pi / 180) = Real.sqrt (1 - m^2) / -m :=
by
  sorry

end tan_80_l40_40985


namespace solution_set_f_neg_x_l40_40071

noncomputable def f (a b x : ℝ) : ℝ := (a * x - 1) * (x + b)

theorem solution_set_f_neg_x (a b : ℝ) (h : ∀ x, f a b x > 0 ↔ -1 < x ∧ x < 3) :
  ∀ x, f a b (-x) < 0 ↔ (x < -3 ∨ x > 1) :=
by
  intro x
  specialize h (-x)
  sorry

end solution_set_f_neg_x_l40_40071


namespace sequence_general_term_l40_40339

noncomputable def b_n (n : ℕ) : ℚ := 2 * n - 1
noncomputable def c_n (n : ℕ) : ℚ := n / (2 * n + 1)

theorem sequence_general_term (n : ℕ) : 
  b_n n + c_n n = (4 * n^2 + n - 1) / (2 * n + 1) :=
by sorry

end sequence_general_term_l40_40339


namespace percent_students_at_trip_l40_40740

variable (total_students : ℕ)
variable (students_taking_more_than_100 : ℕ := (14 * total_students) / 100)
variable (students_not_taking_more_than_100 : ℕ := (75 * total_students) / 100)
variable (students_who_went_to_trip := (students_taking_more_than_100 * 100) / 25)

/--
  If 14 percent of the students at a school went to a camping trip and took more than $100,
  and 75 percent of the students who went to the camping trip did not take more than $100,
  then 56 percent of the students at the school went to the camping trip.
-/
theorem percent_students_at_trip :
    (students_who_went_to_trip * 100) / total_students = 56 :=
sorry

end percent_students_at_trip_l40_40740


namespace factors_and_multiple_of_20_l40_40396

-- Define the relevant numbers
def a := 20
def b := 5
def c := 4

-- Given condition: the equation 20 / 5 = 4
def condition : Prop := a / b = c

-- Factors and multiples relationships to prove
def are_factors : Prop := a % b = 0 ∧ a % c = 0
def is_multiple : Prop := b * c = a

-- The main statement combining everything
theorem factors_and_multiple_of_20 (h : condition) : are_factors ∧ is_multiple :=
sorry

end factors_and_multiple_of_20_l40_40396


namespace max_x5_l40_40663

theorem max_x5 (x1 x2 x3 x4 x5 : ℕ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) (h4 : 0 < x4) (h5 : 0 < x5) 
  (h : x1 + x2 + x3 + x4 + x5 ≤ x1 * x2 * x3 * x4 * x5) : x5 ≤ 5 :=
  sorry

end max_x5_l40_40663


namespace ryan_distance_correct_l40_40006

-- Definitions of the conditions
def billy_distance : ℝ := 30
def madison_distance : ℝ := billy_distance * 1.2
def ryan_distance : ℝ := madison_distance * 0.5

-- Statement to prove
theorem ryan_distance_correct : ryan_distance = 18 := by
  sorry

end ryan_distance_correct_l40_40006


namespace exists_nat_m_inequality_for_large_n_l40_40247

section sequence_problem

-- Define the sequence
noncomputable def a (n : ℕ) : ℚ :=
if n = 7 then 16 / 3 else
if n < 7 then 0 else -- hands off values before a7 that are not needed
3 * a (n - 1) / (7 - a (n - 1) + 4)

-- Define the properties to be proven
theorem exists_nat_m {m : ℕ} :
  (∀ n, n > m → a n < 2) ∧ (∀ n, n ≤ m → a n > 2) :=
sorry

theorem inequality_for_large_n (n : ℕ) (hn : n ≥ 10) :
  (a (n - 1) + a n + 1) / 2 < a n :=
sorry

end sequence_problem

end exists_nat_m_inequality_for_large_n_l40_40247


namespace ones_digit_of_6_pow_52_l40_40720

theorem ones_digit_of_6_pow_52 : (6 ^ 52) % 10 = 6 := by
  -- we'll put the proof here
  sorry

end ones_digit_of_6_pow_52_l40_40720


namespace least_number_1056_div_26_l40_40519

/-- Define the given values and the divisibility condition -/
def least_number_to_add (n : ℕ) (d : ℕ) : ℕ :=
  let remainder := n % d
  d - remainder

/-- State the theorem to prove that the least number to add to 1056 to make it divisible by 26 is 10. -/
theorem least_number_1056_div_26 : least_number_to_add 1056 26 = 10 :=
by
  sorry -- Proof is omitted as per the instruction

end least_number_1056_div_26_l40_40519


namespace value_of_expression_l40_40936

def a : ℕ := 7
def b : ℕ := 5

theorem value_of_expression : (a^2 - b^2)^4 = 331776 := by
  sorry

end value_of_expression_l40_40936


namespace part1_part2_l40_40294

-- Definitions for the conditions
def A : Set ℝ := {x : ℝ | 2 * x - 4 < 0}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 5}
def U : Set ℝ := Set.univ

-- The questions translated as Lean theorems
theorem part1 : A ∩ B = {x : ℝ | 0 < x ∧ x < 2} := by
  sorry

theorem part2 : (U \ A) ∩ B = {x : ℝ | 2 ≤ x ∧ x < 5} := by
  sorry

end part1_part2_l40_40294


namespace number_of_zeros_of_f_l40_40506

def f (x : ℝ) : ℝ := 2 * x - 3 * x

theorem number_of_zeros_of_f :
  ∃ (n : ℕ), n = 2 ∧ (∀ x, f x = 0 → x ∈ {x | f x = 0}) :=
by {
  sorry
}

end number_of_zeros_of_f_l40_40506


namespace sum_of_coeffs_l40_40811

theorem sum_of_coeffs 
  (a b c d e x : ℝ)
  (h : (729 * x ^ 3 + 8) = (a * x + b) * (c * x ^ 2 + d * x + e)) :
  a + b + c + d + e = 78 :=
sorry

end sum_of_coeffs_l40_40811


namespace find_particular_number_l40_40895

theorem find_particular_number (x : ℤ) (h : x - 7 = 2) : x = 9 :=
by {
  -- The proof will be written here.
  sorry
}

end find_particular_number_l40_40895


namespace perfect_square_n_l40_40708

theorem perfect_square_n (n : ℤ) (h1 : n > 0) (h2 : ∃ k : ℤ, n^2 + 19 * n + 48 = k^2) : n = 33 :=
sorry

end perfect_square_n_l40_40708


namespace distance_between_centers_of_circles_l40_40442

theorem distance_between_centers_of_circles (C_1 C_2 : ℝ) : 
  (∀ a : ℝ, (C_1 = a ∧ C_2 = a ∧ (4- a)^2 + (1 - a)^2 = a^2)) → 
  |C_1 - C_2| = 8 :=
by
  sorry

end distance_between_centers_of_circles_l40_40442


namespace quadratic_csq_l40_40979

theorem quadratic_csq (x q t : ℝ) (h : 9 * x^2 - 36 * x - 81 = 0) (hq : q = -2) (ht : t = 13) :
  q + t = 11 :=
by
  sorry

end quadratic_csq_l40_40979


namespace bowling_ball_weight_l40_40098

theorem bowling_ball_weight (b c : ℝ) (h1 : 9 * b = 6 * c) (h2 : 4 * c = 120) : b = 20 :=
sorry

end bowling_ball_weight_l40_40098


namespace cube_edge_length_l40_40437

theorem cube_edge_length (sum_edges length_edge : ℝ) (cube_has_12_edges : 12 * length_edge = sum_edges) (sum_edges_eq_144 : sum_edges = 144) : length_edge = 12 :=
by
  sorry

end cube_edge_length_l40_40437


namespace sqrt_of_16_eq_4_sqrt_of_364_eq_pm19_opposite_of_2_sub_sqrt6_eq_sqrt6_sub_2_l40_40801

theorem sqrt_of_16_eq_4 : Real.sqrt 16 = 4 := 
by sorry

theorem sqrt_of_364_eq_pm19 : Real.sqrt 364 = 19 ∨ Real.sqrt 364 = -19 := 
by sorry

theorem opposite_of_2_sub_sqrt6_eq_sqrt6_sub_2 : -(2 - Real.sqrt 6) = Real.sqrt 6 - 2 := 
by sorry

end sqrt_of_16_eq_4_sqrt_of_364_eq_pm19_opposite_of_2_sub_sqrt6_eq_sqrt6_sub_2_l40_40801


namespace final_expression_simplified_l40_40144

variable (a : ℝ)

theorem final_expression_simplified : 
  (2 * a + 6 - 3 * a) / 2 = -a / 2 + 3 := 
by 
sorry

end final_expression_simplified_l40_40144


namespace bricks_per_course_l40_40669

theorem bricks_per_course : 
  ∃ B : ℕ, (let initial_courses := 3
            let additional_courses := 2
            let total_courses := initial_courses + additional_courses
            let last_course_half_removed := B / 2
            let total_bricks := B * total_courses - last_course_half_removed
            total_bricks = 1800) ↔ B = 400 :=
by {sorry}

end bricks_per_course_l40_40669


namespace ratio_of_saturday_to_friday_customers_l40_40577

def tips_per_customer : ℝ := 2.0
def customers_friday : ℕ := 28
def customers_sunday : ℕ := 36
def total_tips : ℝ := 296

theorem ratio_of_saturday_to_friday_customers :
  let tips_friday := customers_friday * tips_per_customer
  let tips_sunday := customers_sunday * tips_per_customer
  let tips_friday_and_sunday := tips_friday + tips_sunday
  let tips_saturday := total_tips - tips_friday_and_sunday
  let customers_saturday := tips_saturday / tips_per_customer
  (customers_saturday / customers_friday : ℝ) = 3 := 
by
  sorry

end ratio_of_saturday_to_friday_customers_l40_40577


namespace eval_7_star_3_l40_40907

def operation (a b : ℕ) : ℕ := (4 * a + 5 * b - a * b)

theorem eval_7_star_3 : operation 7 3 = 22 :=
  by {
    -- substitution and calculation steps
    sorry
  }

end eval_7_star_3_l40_40907


namespace frac_m_q_eq_one_l40_40213

theorem frac_m_q_eq_one (m n p q : ℕ) 
  (h1 : m = 40 * n)
  (h2 : p = 5 * n)
  (h3 : p = q / 8) : (m / q = 1) :=
by
  sorry

end frac_m_q_eq_one_l40_40213


namespace expression_equivalence_l40_40489

theorem expression_equivalence : (2 / 20) + (3 / 30) + (4 / 40) + (5 / 50) = 0.4 := by
  sorry

end expression_equivalence_l40_40489


namespace fermats_little_theorem_l40_40654

theorem fermats_little_theorem (n p : ℕ) [hp : Fact p.Prime] : p ∣ (n^p - n) :=
sorry

end fermats_little_theorem_l40_40654


namespace root_in_interval_2_3_l40_40769

noncomputable def f (x : ℝ) : ℝ := -|x - 5| + 2^(x - 1)

theorem root_in_interval_2_3 :
  (f 2) * (f 3) < 0 → ∃ c, 2 < c ∧ c < 3 ∧ f c = 0 := by sorry

end root_in_interval_2_3_l40_40769


namespace prime_1011_n_l40_40779

theorem prime_1011_n (n : ℕ) (h : n ≥ 2) : 
  n = 2 ∨ n = 3 ∨ (∀ m : ℕ, m ∣ (n^3 + n + 1) → m = 1 ∨ m = n^3 + n + 1) :=
by sorry

end prime_1011_n_l40_40779


namespace calculate_stripes_l40_40212

theorem calculate_stripes :
  let olga_stripes_per_shoe := 3
  let rick_stripes_per_shoe := olga_stripes_per_shoe - 1
  let hortense_stripes_per_shoe := olga_stripes_per_shoe * 2
  let ethan_stripes_per_shoe := hortense_stripes_per_shoe + 2
  (olga_stripes_per_shoe * 2 + rick_stripes_per_shoe * 2 + hortense_stripes_per_shoe * 2 + ethan_stripes_per_shoe * 2) / 2 = 19 := 
by
  sorry

end calculate_stripes_l40_40212


namespace sum_of_coordinates_of_center_l40_40789

theorem sum_of_coordinates_of_center (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (7, -6)) (h2 : (x2, y2) = (-1, 4)) :
  let center_x := (x1 + x2) / 2
  let center_y := (y1 + y2) / 2
  center_x + center_y = 2 := by
  sorry

end sum_of_coordinates_of_center_l40_40789


namespace bananas_bought_l40_40492

theorem bananas_bought (O P B : Nat) (x : Nat) 
  (h1 : P - O = B)
  (h2 : O + P = 120)
  (h3 : P = 90)
  (h4 : 60 * x + 30 * (2 * x) = 24000) : 
  x = 200 := by
  sorry

end bananas_bought_l40_40492


namespace addition_problem_l40_40813

theorem addition_problem (F I V N E : ℕ) (h1: F = 8) (h2: I % 2 = 0) 
  (h3: 1 ≤ F ∧ F ≤ 9) (h4: 1 ≤ I ∧ I ≤ 9) (h5: 1 ≤ V ∧ V ≤ 9) 
  (h6: 1 ≤ N ∧ N ≤ 9) (h7: 1 ≤ E ∧ E ≤ 9) 
  (h8: F ≠ I ∧ F ≠ V ∧ F ≠ N ∧ F ≠ E) 
  (h9: I ≠ V ∧ I ≠ N ∧ I ≠ E ∧ V ≠ N ∧ V ≠ E ∧ N ≠ E)
  (h10: 2 * F + 2 * I + 2 * V = 1000 * N + 100 * I + 10 * N + E):
  V = 5 :=
sorry

end addition_problem_l40_40813


namespace triangle_area_from_altitudes_l40_40850

noncomputable def triangleArea (altitude1 altitude2 altitude3 : ℝ) : ℝ :=
  sorry

theorem triangle_area_from_altitudes
  (h1 : altitude1 = 15)
  (h2 : altitude2 = 21)
  (h3 : altitude3 = 35) :
  triangleArea 15 21 35 = 245 * Real.sqrt 3 :=
sorry

end triangle_area_from_altitudes_l40_40850


namespace no_primes_in_sequence_l40_40656

def P : ℕ := 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47 * 53 * 59 * 61

theorem no_primes_in_sequence :
  ∀ n : ℕ, 2 ≤ n ∧ n ≤ 59 → ¬ Nat.Prime (P + n) :=
by
  sorry

end no_primes_in_sequence_l40_40656


namespace combinatorial_solution_l40_40462

theorem combinatorial_solution (x : ℕ) (h1 : 0 ≤ x) (h2 : x ≤ 14)
  (h3 : 0 ≤ 2 * x - 4) (h4 : 2 * x - 4 ≤ 14) : x = 4 ∨ x = 6 := by
  sorry

end combinatorial_solution_l40_40462


namespace polygon_sides_arithmetic_progression_l40_40191

theorem polygon_sides_arithmetic_progression
  (angles_in_arithmetic_progression : ∃ (a d : ℝ) (angles : ℕ → ℝ), ∀ (k : ℕ), angles k = a + k * d)
  (common_difference : ∃ (d : ℝ), d = 3)
  (largest_angle : ∃ (n : ℕ) (angles : ℕ → ℝ), angles n = 150) :
  ∃ (n : ℕ), n = 15 :=
sorry

end polygon_sides_arithmetic_progression_l40_40191


namespace roots_equation_l40_40279

theorem roots_equation (α β : ℝ) (h1 : α^2 - 4 * α - 1 = 0) (h2 : β^2 - 4 * β - 1 = 0) :
  3 * α^3 + 4 * β^2 = 80 + 35 * α :=
by
  sorry

end roots_equation_l40_40279


namespace evaluate_expression_l40_40846

-- Definitions for a and b
def a : Int := 1
def b : Int := -1

theorem evaluate_expression : 
  5 * (3 * a ^ 2 * b - a * b ^ 2) - (a * b ^ 2 + 3 * a ^ 2 * b) + 1 = -17 := by
  -- Simplification steps skipped
  sorry

end evaluate_expression_l40_40846


namespace center_cell_value_l40_40018

open Matrix Finset

def table := Matrix (Fin 3) (Fin 3) ℝ

def row_products (T : table) : Prop :=
  (T 0 0 * T 0 1 * T 0 2 = 1) ∧ 
  (T 1 0 * T 1 1 * T 1 2 = 1) ∧ 
  (T 2 0 * T 2 1 * T 2 2 = 1)

def col_products (T : table) : Prop :=
  (T 0 0 * T 1 0 * T 2 0 = 1) ∧ 
  (T 0 1 * T 1 1 * T 2 1 = 1) ∧ 
  (T 0 2 * T 1 2 * T 2 2 = 1)

def square_products (T : table) : Prop :=
  (T 0 0 * T 0 1 * T 1 0 * T 1 1 = 2) ∧ 
  (T 0 1 * T 0 2 * T 1 1 * T 1 2 = 2) ∧ 
  (T 1 0 * T 1 1 * T 2 0 * T 2 1 = 2) ∧ 
  (T 1 1 * T 1 2 * T 2 1 * T 2 2 = 2)

theorem center_cell_value (T : table) 
  (h_row : row_products T) 
  (h_col : col_products T) 
  (h_square : square_products T) : 
  T 1 1 = 1 :=
by
  sorry

end center_cell_value_l40_40018


namespace average_speed_l40_40413

theorem average_speed (d d1 d2 s1 s2 : ℝ)
    (h1 : d = 100)
    (h2 : d1 = 50)
    (h3 : d2 = 50)
    (h4 : s1 = 20)
    (h5 : s2 = 50) :
    d / ((d1 / s1) + (d2 / s2)) = 28.57 :=
by
  sorry

end average_speed_l40_40413


namespace cost_per_mile_proof_l40_40273

noncomputable def daily_rental_cost : ℝ := 50
noncomputable def daily_budget : ℝ := 88
noncomputable def max_miles : ℝ := 190.0

theorem cost_per_mile_proof : 
  (daily_budget - daily_rental_cost) / max_miles = 0.20 := 
by
  sorry

end cost_per_mile_proof_l40_40273


namespace sum_fiftieth_powers_100_gon_l40_40420

noncomputable def sum_fiftieth_powers_all_sides_and_diagonals (n : ℕ) (R : ℝ) : ℝ := sorry
-- Define the sum of 50-th powers of all the sides and diagonals for a general n-gon inscribed in a circle of radius R

theorem sum_fiftieth_powers_100_gon (R : ℝ) : 
  sum_fiftieth_powers_all_sides_and_diagonals 100 R = sorry := sorry

end sum_fiftieth_powers_100_gon_l40_40420


namespace arithmetic_sequence_problem_l40_40143

theorem arithmetic_sequence_problem 
  (a_n b_n : ℕ → ℕ) 
  (S_n T_n : ℕ → ℕ) 
  (h1: ∀ n, S_n n = (n * (a_n n + a_n (n-1))) / 2)
  (h2: ∀ n, T_n n = (n * (b_n n + b_n (n-1))) / 2)
  (h3: ∀ n, (S_n n) / (T_n n) = (7 * n + 2) / (n + 3)):
  (a_n 4) / (b_n 4) = 51 / 10 := 
sorry

end arithmetic_sequence_problem_l40_40143


namespace max_value_of_m_l40_40998

theorem max_value_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - 8 > 0 → x < m) → m = -2 :=
by
  sorry

end max_value_of_m_l40_40998


namespace max_area_difference_160_perimeter_rectangles_l40_40132

theorem max_area_difference_160_perimeter_rectangles : 
  ∃ (l1 w1 l2 w2 : ℕ), (2 * l1 + 2 * w1 = 160) ∧ (2 * l2 + 2 * w2 = 160) ∧ 
  (l1 * w1 - l2 * w2 = 1521) := sorry

end max_area_difference_160_perimeter_rectangles_l40_40132


namespace permits_increase_l40_40438

theorem permits_increase :
  let old_permits := 26^2 * 10^3
  let new_permits := 26^4 * 10^4
  new_permits = 67600 * old_permits :=
by
  let old_permits := 26^2 * 10^3
  let new_permits := 26^4 * 10^4
  exact sorry

end permits_increase_l40_40438


namespace problem1_l40_40103

theorem problem1 (x y : ℝ) (h : x + y > 2) : x > 1 ∨ y > 1 :=
sorry

end problem1_l40_40103


namespace sequences_meet_at_2017_l40_40711

-- Define the sequences for Paul and Penny
def paul_sequence (n : ℕ) : ℕ := 3 * n - 2
def penny_sequence (m : ℕ) : ℕ := 2022 - 5 * m

-- Statement to be proven
theorem sequences_meet_at_2017 : ∃ n m : ℕ, paul_sequence n = 2017 ∧ penny_sequence m = 2017 := by
  sorry

end sequences_meet_at_2017_l40_40711


namespace vanessa_total_earnings_l40_40939

theorem vanessa_total_earnings :
  let num_dresses := 7
  let num_shirts := 4
  let price_per_dress := 7
  let price_per_shirt := 5
  (num_dresses * price_per_dress + num_shirts * price_per_shirt) = 69 :=
by
  sorry

end vanessa_total_earnings_l40_40939


namespace daily_profit_9080_l40_40799

theorem daily_profit_9080 (num_employees : Nat) (shirts_per_employee_per_day : Nat) (hours_per_shift : Nat) (wage_per_hour : Nat) (bonus_per_shirt : Nat) (shirt_sale_price : Nat) (nonemployee_expenses : Nat) :
  num_employees = 20 →
  shirts_per_employee_per_day = 20 →
  hours_per_shift = 8 →
  wage_per_hour = 12 →
  bonus_per_shirt = 5 →
  shirt_sale_price = 35 →
  nonemployee_expenses = 1000 →
  (num_employees * shirts_per_employee_per_day * shirt_sale_price) - ((num_employees * (hours_per_shift * wage_per_hour + shirts_per_employee_per_day * bonus_per_shirt)) + nonemployee_expenses) = 9080 := 
by
  intros
  sorry

end daily_profit_9080_l40_40799


namespace quadratic_two_distinct_real_roots_l40_40776

theorem quadratic_two_distinct_real_roots (k : ℝ) : 
  (k - 1 ≠ 0 ∧ 8 - 4 * k > 0) ↔ (k < 2 ∧ k ≠ 1) := 
by
  sorry

end quadratic_two_distinct_real_roots_l40_40776


namespace length_of_train_l40_40560

variable (L : ℕ)

def speed_tree (L : ℕ) : ℚ := L / 120

def speed_platform (L : ℕ) : ℚ := (L + 500) / 160

theorem length_of_train
    (h1 : speed_tree L = speed_platform L)
    : L = 1500 :=
sorry

end length_of_train_l40_40560


namespace perimeter_of_regular_polygon_l40_40702

theorem perimeter_of_regular_polygon
  (side_length : ℕ)
  (exterior_angle : ℕ)
  (h1 : exterior_angle = 90)
  (h2 : side_length = 7) :
  4 * side_length = 28 :=
by
  sorry

end perimeter_of_regular_polygon_l40_40702


namespace vector_magnitude_proof_l40_40829

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ := 
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_magnitude_proof
  (a b c : ℝ × ℝ)
  (h_a : a = (-2, 1))
  (h_b : b = (-2, 3))
  (h_c : ∃ m : ℝ, c = (m, -1) ∧ (m * b.1 + (-1) * b.2 = 0)) :
  vector_magnitude (a.1 - c.1, a.2 - c.2) = Real.sqrt 17 / 2 :=
by
  sorry

end vector_magnitude_proof_l40_40829


namespace negation_of_prop1_equiv_l40_40852

-- Given proposition: if x > 1 then x > 0
def prop1 (x : ℝ) : Prop := x > 1 → x > 0

-- Negation of the given proposition: if x ≤ 1 then x ≤ 0
def neg_prop1 (x : ℝ) : Prop := x ≤ 1 → x ≤ 0

-- The theorem to prove that the negation of the proposition "If x > 1, then x > 0" 
-- is "If x ≤ 1, then x ≤ 0"
theorem negation_of_prop1_equiv (x : ℝ) : ¬(prop1 x) ↔ neg_prop1 x :=
by
  sorry

end negation_of_prop1_equiv_l40_40852


namespace log_fraction_identity_l40_40642

theorem log_fraction_identity (a b : ℝ) (h2 : Real.log 2 = a) (h3 : Real.log 3 = b) :
  (Real.log 12 / Real.log 15) = (2 * a + b) / (1 - a + b) := 
  sorry

end log_fraction_identity_l40_40642


namespace largest_three_digit_number_divisible_by_8_l40_40662

-- Define the properties of a number being a three-digit number
def isThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

-- Define the property of a number being divisible by 8
def isDivisibleBy8 (n : ℕ) : Prop := n % 8 = 0

-- The theorem we want to prove: the largest three-digit number divisible by 8 is 992
theorem largest_three_digit_number_divisible_by_8 : ∃ n, isThreeDigitNumber n ∧ isDivisibleBy8 n ∧ (∀ m, isThreeDigitNumber m ∧ isDivisibleBy8 m → m ≤ 992) :=
  sorry

end largest_three_digit_number_divisible_by_8_l40_40662


namespace lena_more_candy_bars_than_nicole_l40_40331

theorem lena_more_candy_bars_than_nicole
  (Lena Kevin Nicole : ℕ)
  (h1 : Lena = 16)
  (h2 : Lena + 5 = 3 * Kevin)
  (h3 : Kevin + 4 = Nicole) :
  Lena - Nicole = 5 :=
by
  sorry

end lena_more_candy_bars_than_nicole_l40_40331


namespace smallest_digit_divisible_by_11_l40_40257

theorem smallest_digit_divisible_by_11 : ∃ d : ℕ, (0 ≤ d ∧ d ≤ 9) ∧ d = 6 ∧ (d + 7 - (4 + 3 + 6)) % 11 = 0 := by
  sorry

end smallest_digit_divisible_by_11_l40_40257


namespace find_x_in_interval_l40_40265

noncomputable def a : ℝ := Real.sqrt 2014 - Real.sqrt 2013

theorem find_x_in_interval :
  ∀ x : ℝ, (0 < x) → (x < Real.pi) →
  (a^(Real.tan x ^ 2) + (Real.sqrt 2014 + Real.sqrt 2013)^(-Real.tan x ^ 2) = 2 * a^3) →
  (x = Real.pi / 3 ∨ x = 2 * Real.pi / 3) := by
  -- add proof here
  sorry

end find_x_in_interval_l40_40265


namespace apples_per_hour_l40_40498

def total_apples : ℕ := 15
def hours : ℕ := 3

theorem apples_per_hour : total_apples / hours = 5 := by
  sorry

end apples_per_hour_l40_40498


namespace glue_needed_l40_40905

-- Definitions based on conditions
def num_friends : ℕ := 7
def clippings_per_friend : ℕ := 3
def drops_per_clipping : ℕ := 6

-- Calculation
def total_clippings : ℕ := num_friends * clippings_per_friend
def total_drops_of_glue : ℕ := drops_per_clipping * total_clippings

-- Theorem statement
theorem glue_needed : total_drops_of_glue = 126 := by
  sorry

end glue_needed_l40_40905


namespace negation_of_proposition_l40_40588

open Real

theorem negation_of_proposition (P : ∀ x : ℝ, sin x ≥ 1) :
  ∃ x : ℝ, sin x < 1 :=
sorry

end negation_of_proposition_l40_40588


namespace intersection_of_A_and_B_l40_40877

-- Definitions of sets A and B
def A : Set ℝ := { x | x ≥ 1 }
def B : Set ℝ := { x | -1 < x ∧ x < 2 }

-- The theorem we want to prove
theorem intersection_of_A_and_B : A ∩ B = { x | 1 ≤ x ∧ x < 2 } :=
by
  sorry

end intersection_of_A_and_B_l40_40877


namespace max_has_two_nickels_l40_40214

theorem max_has_two_nickels (n : ℕ) (nickels : ℕ) (coins_value_total : ℕ) :
  (coins_value_total = 15 * n) -> (coins_value_total + 10 = 16 * (n + 1)) -> 
  coins_value_total - nickels * 5 + nickels + 25 = 90 -> 
  n = 6 -> 
  2 = nickels := 
by 
  sorry

end max_has_two_nickels_l40_40214


namespace line_passes_through_fixed_point_l40_40534

theorem line_passes_through_fixed_point (k : ℝ) : ∀ x y : ℝ, (y - 1 = k * (x + 2)) → (x = -2 ∧ y = 1) :=
by
  intro x y h
  sorry

end line_passes_through_fixed_point_l40_40534


namespace quadratic_must_have_m_eq_neg2_l40_40321

theorem quadratic_must_have_m_eq_neg2 (m : ℝ) (h : (m - 2) * x^|m| - 3 * x - 4 = 0) :
  (|m| = 2) ∧ (m ≠ 2) → m = -2 :=
by
  sorry

end quadratic_must_have_m_eq_neg2_l40_40321


namespace sum_of_primes_less_than_20_l40_40675

theorem sum_of_primes_less_than_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 = 77) :=
by
  sorry

end sum_of_primes_less_than_20_l40_40675


namespace partition_natural_numbers_l40_40215

theorem partition_natural_numbers :
  ∃ (f : ℕ → ℕ), (∀ n, 1 ≤ f n ∧ f n ≤ 100) ∧
  (∀ a b c, a + 99 * b = c → f a = f c ∨ f a = f b ∨ f b = f c) :=
sorry

end partition_natural_numbers_l40_40215


namespace abs_h_eq_2_l40_40482

-- Definitions based on the given conditions
def sum_of_squares_of_roots (h : ℝ) : Prop :=
  let a := 1
  let b := -4 * h
  let c := -8
  let sum_of_roots := -b / a
  let prod_of_roots := c / a
  let sum_of_squares := sum_of_roots^2 - 2 * prod_of_roots
  sum_of_squares = 80

-- Theorem to prove the absolute value of h is 2
theorem abs_h_eq_2 (h : ℝ) (h_condition : sum_of_squares_of_roots h) : |h| = 2 :=
by
  sorry

end abs_h_eq_2_l40_40482


namespace three_million_times_three_million_l40_40880

theorem three_million_times_three_million : 
  (3 * 10^6) * (3 * 10^6) = 9 * 10^12 := 
by
  sorry

end three_million_times_three_million_l40_40880


namespace length_of_BC_l40_40063

-- Define the given conditions and the theorem using Lean
theorem length_of_BC 
  (A B C : ℝ × ℝ) 
  (hA : A = (0, 0)) 
  (hB : ∃ b : ℝ, B = (-b, -b^2)) 
  (hC : ∃ b : ℝ, C = (b, -b^2)) 
  (hBC_parallel_x_axis : ∀ b : ℝ, C.2 = B.2)
  (hArea : ∀ b : ℝ, b^3 = 72) 
  : ∀ b : ℝ, (BC : ℝ) = 2 * b := 
by
  sorry

end length_of_BC_l40_40063


namespace correct_mark_l40_40197

theorem correct_mark
  (n : ℕ)
  (initial_avg : ℝ)
  (wrong_mark : ℝ)
  (correct_avg : ℝ)
  (correct_total_marks : ℝ)
  (actual_total_marks : ℝ)
  (final_mark : ℝ) :
  n = 25 →
  initial_avg = 100 →
  wrong_mark = 60 →
  correct_avg = 98 →
  correct_total_marks = (n * correct_avg) →
  actual_total_marks = (n * initial_avg - wrong_mark + final_mark) →
  correct_total_marks = actual_total_marks →
  final_mark = 10 :=
by
  intros h_n h_initial_avg h_wrong_mark h_correct_avg h_correct_total_marks h_actual_total_marks h_eq
  sorry

end correct_mark_l40_40197


namespace lamp_count_and_profit_l40_40057

-- Define the parameters given in the problem
def total_lamps : ℕ := 50
def total_cost : ℕ := 2500
def cost_A : ℕ := 40
def cost_B : ℕ := 65
def marked_A : ℕ := 60
def marked_B : ℕ := 100
def discount_A : ℕ := 10 -- percent
def discount_B : ℕ := 30 -- percent

-- Derived definitions from the solution
def lamps_A : ℕ := 30
def lamps_B : ℕ := 20
def selling_price_A : ℕ := marked_A * (100 - discount_A) / 100
def selling_price_B : ℕ := marked_B * (100 - discount_B) / 100
def profit_A : ℕ := selling_price_A - cost_A
def profit_B : ℕ := selling_price_B - cost_B
def total_profit : ℕ := (profit_A * lamps_A) + (profit_B * lamps_B)

-- Lean statement
theorem lamp_count_and_profit :
  lamps_A + lamps_B = total_lamps ∧
  (cost_A * lamps_A + cost_B * lamps_B) = total_cost ∧
  total_profit = 520 := by
  -- proofs will go here
  sorry

end lamp_count_and_profit_l40_40057


namespace warehouse_box_storage_l40_40508

theorem warehouse_box_storage (S : ℝ) (h1 : (3 - 1/4) * S = 55000) : (1/4) * S = 5000 :=
by
  sorry

end warehouse_box_storage_l40_40508


namespace euler_totient_divisibility_l40_40616

def euler_totient (n : ℕ) : ℕ := sorry

theorem euler_totient_divisibility (n : ℕ) (hn : 0 < n) : 2^(n * (n + 1)) ∣ 32 * euler_totient (2^(2^n) - 1) := 
sorry

end euler_totient_divisibility_l40_40616


namespace hyperbola_real_axis_length_l40_40053

theorem hyperbola_real_axis_length
    (a b : ℝ) 
    (h_pos_a : a > 0) 
    (h_pos_b : b > 0) 
    (h_eccentricity : a * Real.sqrt 5 = Real.sqrt (a^2 + b^2))
    (h_distance : b * a * Real.sqrt 5 / Real.sqrt (a^2 + b^2) = 8) :
    2 * a = 8 :=
sorry

end hyperbola_real_axis_length_l40_40053


namespace math_problem_l40_40564

variables (a b c d m : ℝ)

theorem math_problem 
  (h1 : a = -b)            -- condition 1: a and b are opposite numbers
  (h2 : c * d = 1)         -- condition 2: c and d are reciprocal numbers
  (h3 : |m| = 1) :         -- condition 3: absolute value of m is 1
  (a + b) * c * d - 2009 * m = -2009 ∨ (a + b) * c * d - 2009 * m = 2009 :=
sorry

end math_problem_l40_40564


namespace solve_for_x_l40_40541

theorem solve_for_x : ∀ x : ℝ, (x - 27) / 3 = (3 * x + 6) / 8 → x = -234 :=
by
  intro x
  intro h
  sorry

end solve_for_x_l40_40541


namespace son_l40_40466

theorem son's_age (S M : ℕ) 
  (h1 : M = S + 24) 
  (h2 : M + 2 = 2 * (S + 2)) : S = 22 := 
by 
  sorry

end son_l40_40466


namespace inequality_solution_set_l40_40355

theorem inequality_solution_set (x : ℝ) : (x - 1) * abs (x + 2) ≥ 0 ↔ (x ≥ 1 ∨ x = -2) :=
by
  sorry

end inequality_solution_set_l40_40355


namespace angle_B_is_60_l40_40364

noncomputable def triangle_with_centroid (a b c : ℝ) (GA GB GC : ℝ) : Prop :=
  56 * a * GA + 40 * b * GB + 35 * c * GC = 0

theorem angle_B_is_60 {a b c GA GB GC : ℝ} (h : 56 * a * GA + 40 * b * GB + 35 * c * GC = 0) :
  ∃ B : ℝ, B = 60 :=
sorry

end angle_B_is_60_l40_40364


namespace calculate_expression_l40_40957

theorem calculate_expression (x : ℝ) : 2 * x^3 * (-3 * x)^2 = 18 * x^5 :=
by
  sorry

end calculate_expression_l40_40957


namespace cube_cut_problem_l40_40425

theorem cube_cut_problem (n s : ℕ) (h1 : n^3 - s^3 = 152) (h2 : ∀ i, i = 1 ∨ i = s)
  (h3 : s * s * s ≤ n * n * n) (h4 : n > 1) : n = 6 :=
by sorry

end cube_cut_problem_l40_40425


namespace independence_events_exactly_one_passing_l40_40479

-- Part 1: Independence of Events

def event_A (die1 : ℕ) : Prop :=
  die1 % 2 = 1

def event_B (die1 die2 : ℕ) : Prop :=
  (die1 + die2) % 3 = 0

def P_event_A : ℚ :=
  1 / 2

def P_event_B : ℚ :=
  1 / 3

def P_event_AB : ℚ :=
  1 / 6

theorem independence_events : P_event_AB = P_event_A * P_event_B :=
by
  sorry

-- Part 2: Probability of Exactly One Passing the Assessment

def probability_of_hitting (p : ℝ) : ℝ :=
  1 - (1 - p)^2

def P_A_hitting : ℝ :=
  0.7

def P_B_hitting : ℝ :=
  0.6

def probability_one_passing : ℝ :=
  (probability_of_hitting P_A_hitting) * (1 - probability_of_hitting P_B_hitting) + (1 - probability_of_hitting P_A_hitting) * (probability_of_hitting P_B_hitting)

theorem exactly_one_passing : probability_one_passing = 0.2212 :=
by
  sorry

end independence_events_exactly_one_passing_l40_40479


namespace derivative_at_pi_over_4_l40_40177

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem derivative_at_pi_over_4 : 
  deriv f (Real.pi / 4) = Real.sqrt 2 / 2 + Real.sqrt 2 * Real.pi / 8 :=
by
  -- Proof goes here
  sorry

end derivative_at_pi_over_4_l40_40177


namespace player_matches_average_increase_l40_40557

theorem player_matches_average_increase 
  (n T : ℕ) 
  (h1 : T = 32 * n) 
  (h2 : (T + 76) / (n + 1) = 36) : 
  n = 10 := 
by 
  sorry

end player_matches_average_increase_l40_40557


namespace radius_of_third_circle_l40_40157

noncomputable def circle_radius {r1 r2 : ℝ} (h1 : r1 = 15) (h2 : r2 = 25) : ℝ :=
  let A_shaded := (25^2 * Real.pi) - (15^2 * Real.pi)
  let r := Real.sqrt (A_shaded / Real.pi)
  r

theorem radius_of_third_circle (r1 r2 r3 : ℝ) (h1 : r1 = 15) (h2 : r2 = 25) :
  circle_radius h1 h2 = 20 :=
by 
  sorry

end radius_of_third_circle_l40_40157


namespace real_part_of_complex_div_l40_40890

noncomputable def complexDiv (c1 c2 : ℂ) := c1 / c2

theorem real_part_of_complex_div (i_unit : ℂ) (h_i : i_unit = Complex.I) :
  (Complex.re (complexDiv (2 * i_unit) (1 + i_unit)) = 1) :=
by
  sorry

end real_part_of_complex_div_l40_40890


namespace range_of_m_l40_40592

noncomputable def f (x m : ℝ) : ℝ := -x^2 + m * x

theorem range_of_m {m : ℝ} : (∀ x y : ℝ, x ≤ y → x ≤ 1 → y ≤ 1 → f x m ≤ f y m) ↔ 2 ≤ m := 
sorry

end range_of_m_l40_40592


namespace evaluate_sum_l40_40447

theorem evaluate_sum : (-1:ℤ) ^ 2010 + (-1:ℤ) ^ 2011 + (1:ℤ) ^ 2012 - (1:ℤ) ^ 2013 + (-1:ℤ) ^ 2014 = 0 := by
  sorry

end evaluate_sum_l40_40447


namespace jerry_average_increase_l40_40007

-- Definitions of conditions
def first_three_tests_average (avg : ℕ) : Prop := avg = 85
def fourth_test_score (score : ℕ) : Prop := score = 97
def desired_average_increase (increase : ℕ) : Prop := increase = 3

-- The theorem to prove
theorem jerry_average_increase
  (first_avg first_avg_value : ℕ)
  (fourth_score fourth_score_value : ℕ)
  (increase_points : ℕ)
  (h1 : first_three_tests_average first_avg)
  (h2 : fourth_test_score fourth_score)
  (h3 : desired_average_increase increase_points) :
  fourth_score = 97 → (first_avg + fourth_score) / 4 = 88 → increase_points = 3 :=
by
  intros _ _
  sorry

end jerry_average_increase_l40_40007


namespace cost_per_use_correct_l40_40781

-- Definitions based on conditions in the problem
def total_cost : ℕ := 30
def uses_per_week : ℕ := 3
def number_of_weeks : ℕ := 2
def total_uses : ℕ := uses_per_week * number_of_weeks

-- Statement based on the question and correct answer
theorem cost_per_use_correct : (total_cost / total_uses) = 5 := sorry

end cost_per_use_correct_l40_40781


namespace trapezoid_ABCD_BCE_area_l40_40841

noncomputable def triangle_area (a b c : ℝ) (angle_abc : ℝ) : ℝ :=
  1 / 2 * a * b * Real.sin angle_abc

noncomputable def area_of_triangle_BCE (AB DC AD : ℝ) (angle_DAB : ℝ) (area_triangle_DCB : ℝ) : ℝ :=
  let ratio := AB / DC
  (ratio / (1 + ratio)) * area_triangle_DCB

theorem trapezoid_ABCD_BCE_area :
  ∀ (AB DC AD : ℝ) (angle_DAB : ℝ) (area_triangle_DCB : ℝ),
    AB = 30 →
    DC = 24 →
    AD = 3 →
    angle_DAB = Real.pi / 3 →
    area_triangle_DCB = 18 * Real.sqrt 3 →
    area_of_triangle_BCE AB DC AD angle_DAB area_triangle_DCB = 10 * Real.sqrt 3 := 
by
  intros
  sorry

end trapezoid_ABCD_BCE_area_l40_40841


namespace rem_frac_l40_40218

def rem (x y : ℚ) : ℚ := x - y * (⌊x / y⌋ : ℤ)

theorem rem_frac : rem (7 / 12) (-3 / 4) = -1 / 6 :=
by
  sorry

end rem_frac_l40_40218


namespace walking_distance_l40_40809

theorem walking_distance (a b : ℝ) (h1 : 10 * a + 45 * b = a * 15)
(h2 : x * (a + 9 * b) = 10 * a + 45 * b) : x = 13.5 :=
by
  sorry

end walking_distance_l40_40809


namespace bus_children_l40_40742

theorem bus_children (X : ℕ) (initial_children : ℕ) (got_on : ℕ) (total_children_after : ℕ) 
  (h1 : initial_children = 28) 
  (h2 : got_on = 82) 
  (h3 : total_children_after = 30) 
  (h4 : initial_children + got_on - X = total_children_after) : 
  got_on - X = 2 :=
by 
  -- h1, h2, h3, and h4 are conditions from the problem
  sorry

end bus_children_l40_40742


namespace age_of_17th_student_l40_40165

theorem age_of_17th_student (avg_age_17 : ℕ) (total_students : ℕ) (avg_age_5 : ℕ) (students_5 : ℕ) (avg_age_9 : ℕ) (students_9 : ℕ)
  (h1 : avg_age_17 = 17) (h2 : total_students = 17) (h3 : avg_age_5 = 14) (h4 : students_5 = 5) (h5 : avg_age_9 = 16) (h6 : students_9 = 9) :
  ∃ age_17th_student : ℕ, age_17th_student = 75 :=
by
  sorry

end age_of_17th_student_l40_40165


namespace minimize_f_minimize_f_exact_l40_40080

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 + 14 * x - 20

-- State the theorem that x = -7 minimizes the function f(x)
theorem minimize_f : ∀ x : ℝ, f x ≥ f (-7) :=
by
  intro x
  unfold f
  sorry

-- An alternative statement could include the exact condition for the minimum value
theorem minimize_f_exact : ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y ∧ x = -7 :=
by
  use -7
  intro y
  unfold f
  sorry

end minimize_f_minimize_f_exact_l40_40080


namespace number_of_ways_to_form_team_l40_40147

noncomputable def binomial : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n + 1, k + 1 => binomial n k + binomial n (k + 1)

theorem number_of_ways_to_form_team :
  let total_selections := binomial 11 5
  let all_boys_selections := binomial 8 5
  total_selections - all_boys_selections = 406 :=
by 
  sorry

end number_of_ways_to_form_team_l40_40147


namespace three_digit_numbers_divide_26_l40_40628

def divides (d n : ℕ) : Prop := ∃ k, n = d * k

theorem three_digit_numbers_divide_26 (a b c : ℕ) (h₀ : a ≠ 0) (h₁ : a < 10) (h₂ : b < 10) (h₃ : c < 10) :
  (divides 26 (a^2 + b^2 + c^2)) ↔ 
    ((a = 1 ∧ b = 0 ∧ c = 0) ∨
     (a = 1 ∧ b = 1 ∧ c = 0) ∨
     (a = 3 ∧ b = 2 ∧ c = 0) ∨
     (a = 5 ∧ b = 1 ∧ c = 0) ∨
     (a = 4 ∧ b = 3 ∧ c = 1)) :=
by 
  sorry

end three_digit_numbers_divide_26_l40_40628


namespace banana_production_total_l40_40827

def banana_production (nearby_island_production : ℕ) (jakies_multiplier : ℕ) : ℕ :=
  nearby_island_production + (jakies_multiplier * nearby_island_production)

theorem banana_production_total
  (nearby_island_production : ℕ)
  (jakies_multiplier : ℕ)
  (h1 : nearby_island_production = 9000)
  (h2 : jakies_multiplier = 10)
  : banana_production nearby_island_production jakies_multiplier = 99000 :=
by
  sorry

end banana_production_total_l40_40827


namespace find_peaches_l40_40737

theorem find_peaches (A P : ℕ) (h1 : A + P = 15) (h2 : 1000 * A + 2000 * P = 22000) : P = 7 := sorry

end find_peaches_l40_40737


namespace mutually_exclusive_pairs_l40_40748

/-- Define the events for shooting rings and drawing balls. -/
inductive ShootEvent
| hits_7th_ring : ShootEvent
| hits_8th_ring : ShootEvent

inductive PersonEvent
| at_least_one_hits : PersonEvent
| A_hits_B_does_not : PersonEvent

inductive BallEvent
| at_least_one_black : BallEvent
| both_red : BallEvent
| no_black : BallEvent
| one_red : BallEvent

/-- Define mutually exclusive events. -/
def mutually_exclusive (e1 e2 : Prop) : Prop := e1 ∧ e2 → False

/-- Prove the pairs of events that are mutually exclusive. -/
theorem mutually_exclusive_pairs :
  mutually_exclusive (ShootEvent.hits_7th_ring = ShootEvent.hits_7th_ring) (ShootEvent.hits_8th_ring = ShootEvent.hits_8th_ring) ∧
  ¬mutually_exclusive (PersonEvent.at_least_one_hits = PersonEvent.at_least_one_hits) (PersonEvent.A_hits_B_does_not = PersonEvent.A_hits_B_does_not) ∧
  mutually_exclusive (BallEvent.at_least_one_black = BallEvent.at_least_one_black) (BallEvent.both_red = BallEvent.both_red) ∧
  mutually_exclusive (BallEvent.no_black = BallEvent.no_black) (BallEvent.one_red = BallEvent.one_red) :=
by {
  sorry
}

end mutually_exclusive_pairs_l40_40748


namespace friends_carrying_bananas_l40_40087

theorem friends_carrying_bananas :
  let total_friends := 35
  let friends_with_pears := 14
  let friends_with_oranges := 8
  let friends_with_apples := 5
  total_friends - (friends_with_pears + friends_with_oranges + friends_with_apples) = 8 := 
by
  sorry

end friends_carrying_bananas_l40_40087


namespace ratio_of_area_to_breadth_l40_40124

theorem ratio_of_area_to_breadth (b l A : ℝ) (h₁ : b = 10) (h₂ : l - b = 10) (h₃ : A = l * b) : A / b = 20 := 
by
  sorry

end ratio_of_area_to_breadth_l40_40124


namespace possible_values_of_N_l40_40676

def is_valid_N (N : ℕ) : Prop :=
  (N > 22) ∧ (N ≤ 25)

theorem possible_values_of_N :
  {N : ℕ | is_valid_N N} = {23, 24, 25} :=
by
  sorry

end possible_values_of_N_l40_40676


namespace ab_sum_l40_40328

open Set

noncomputable def U : Set ℝ := univ
def A : Set ℝ := { x | -1 < x ∧ x < 5 }
def B : Set ℝ := { x | 2 < x ∧ x < 8 }
def C (a : ℝ) : Set ℝ := { x | a + 1 ≤ x ∧ x ≤ 2 * a - 2 }
def complement_A : Set ℝ := { x | x ≤ -1 ∨ x ≥ 5 }
def complement_B : Set ℝ := { x | x ≤ 2 ∨ x ≥ 8 }
def complement_A_and_C (a b : ℝ) : Set ℝ := { x | 6 ≤ x ∧ x ≤ b }

theorem ab_sum (a b: ℝ) (h: (complement_A ∩ C a) = complement_A_and_C a b) : a + b = 13 :=
by
  sorry

end ab_sum_l40_40328


namespace great_eighteen_hockey_league_games_l40_40238

theorem great_eighteen_hockey_league_games :
  (let teams_per_division := 9
   let games_intra_division_per_team := 8 * 3
   let games_inter_division_per_team := teams_per_division * 2
   let total_games_per_team := games_intra_division_per_team + games_inter_division_per_team
   let total_game_instances := 18 * total_games_per_team
   let unique_games := total_game_instances / 2
   unique_games = 378) :=
by
  sorry

end great_eighteen_hockey_league_games_l40_40238


namespace div_by_19_l40_40167

theorem div_by_19 (n : ℕ) : 19 ∣ (26^n - 7^n) :=
sorry

end div_by_19_l40_40167


namespace find_Y_value_l40_40193

theorem find_Y_value : ∃ Y : ℤ, 80 - (Y - (6 + 2 * (7 - 8 - 5))) = 89 ∧ Y = -15 := by
  sorry

end find_Y_value_l40_40193


namespace scientific_notation_35100_l40_40504

theorem scientific_notation_35100 : 35100 = 3.51 * 10^4 :=
by
  sorry

end scientific_notation_35100_l40_40504


namespace tangerine_count_l40_40195

-- Definitions based directly on the conditions
def initial_oranges : ℕ := 5
def remaining_oranges : ℕ := initial_oranges - 2
def remaining_tangerines (T : ℕ) : ℕ := T - 10
def condition1 (T : ℕ) : Prop := remaining_tangerines T = remaining_oranges + 4

-- Theorem to prove the number of tangerines in the bag
theorem tangerine_count (T : ℕ) (h : condition1 T) : T = 17 :=
by
  sorry

end tangerine_count_l40_40195


namespace compute_expression_value_l40_40525

-- Define the expression
def expression : ℤ := 1013^2 - 1009^2 - 1011^2 + 997^2

-- State the theorem with the required conditions and conclusions
theorem compute_expression_value : expression = -19924 := 
by 
  -- The proof steps would go here.
  sorry

end compute_expression_value_l40_40525


namespace domain_of_sqrt_l40_40081

theorem domain_of_sqrt (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 :=
by sorry

end domain_of_sqrt_l40_40081


namespace smallest_b_factors_l40_40581

theorem smallest_b_factors (b p q : ℤ) (H : p * q = 2016) : 
  (∀ k₁ k₂ : ℤ, k₁ * k₂ = 2016 → k₁ + k₂ ≥ p + q) → 
  b = 90 :=
by
  -- Here, we assume the premises stated for integers p, q such that their product is 2016.
  -- We need to fill in the proof steps which will involve checking all appropriate (p, q) pairs.
  sorry

end smallest_b_factors_l40_40581


namespace stickers_per_page_l40_40451

theorem stickers_per_page (total_pages total_stickers : ℕ) (h1 : total_pages = 22) (h2 : total_stickers = 220) : (total_stickers / total_pages) = 10 :=
by
  sorry

end stickers_per_page_l40_40451


namespace solution_set_l40_40967
  
noncomputable def f (x : ℝ) : ℝ :=
  Real.log (Real.exp (2 * x) + 1) - x

theorem solution_set (x : ℝ) :
  f (x + 2) > f (2 * x - 3) ↔ (1 / 3 < x ∧ x < 5) :=
by
  sorry

end solution_set_l40_40967


namespace min_value_of_a_l40_40424

noncomputable def smallest_root_sum : ℕ := 78

theorem min_value_of_a (r s t : ℕ) (h1 : r * s * t = 2310) (h2 : r > 0) (h3 : s > 0) (h4 : t > 0) :
  r + s + t = smallest_root_sum :=
sorry

end min_value_of_a_l40_40424


namespace simplify_expression_l40_40988

theorem simplify_expression (x y : ℝ) :  3 * x + 5 * x + 7 * x + 2 * y = 15 * x + 2 * y := 
by 
  sorry

end simplify_expression_l40_40988


namespace find_a_l40_40201

theorem find_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : (a^2 + a = 6)) : a = 2 :=
sorry

end find_a_l40_40201


namespace average_marks_all_students_proof_l40_40376

-- Definitions based on the given conditions
def class1_student_count : ℕ := 35
def class2_student_count : ℕ := 45
def class1_average_marks : ℕ := 40
def class2_average_marks : ℕ := 60

-- Total marks calculations
def class1_total_marks : ℕ := class1_student_count * class1_average_marks
def class2_total_marks : ℕ := class2_student_count * class2_average_marks
def total_marks : ℕ := class1_total_marks + class2_total_marks

-- Total student count
def total_student_count : ℕ := class1_student_count + class2_student_count

-- Average marks of all students
noncomputable def average_marks_all_students : ℚ := total_marks / total_student_count

-- Lean statement to prove
theorem average_marks_all_students_proof
  (h1 : class1_student_count = 35)
  (h2 : class2_student_count = 45)
  (h3 : class1_average_marks = 40)
  (h4 : class2_average_marks = 60) :
  average_marks_all_students = 51.25 := by
  sorry

end average_marks_all_students_proof_l40_40376


namespace proposition_incorrect_l40_40747

theorem proposition_incorrect :
  ¬(∀ x : ℝ, x^2 + 3 * x + 1 > 0) :=
by
  sorry

end proposition_incorrect_l40_40747


namespace sum_of_products_of_two_at_a_time_l40_40983

-- Given conditions
variables (a b c : ℝ)
axiom sum_of_squares : a^2 + b^2 + c^2 = 252
axiom sum_of_numbers : a + b + c = 22

-- The goal
theorem sum_of_products_of_two_at_a_time : a * b + b * c + c * a = 116 :=
sorry

end sum_of_products_of_two_at_a_time_l40_40983


namespace unique_solution_l40_40743

theorem unique_solution : ∀ (x y z : ℕ), 
  x > 0 → y > 0 → z > 0 → 
  x^2 = 2 * (y + z) → 
  x^6 = y^6 + z^6 + 31 * (y^2 + z^2) → 
  (x, y, z) = (2, 1, 1) :=
by sorry

end unique_solution_l40_40743


namespace Loisa_saves_70_l40_40548

-- Define the conditions
def tablet_cost_cash := 450
def down_payment := 100
def payment_first_4_months := 40 * 4
def payment_next_4_months := 35 * 4
def payment_last_4_months := 30 * 4

-- Define the total installment payment
def total_installment_payment := down_payment + payment_first_4_months + payment_next_4_months + payment_last_4_months

-- Define the amount saved by paying cash instead of on installment
def amount_saved := total_installment_payment - tablet_cost_cash

-- The theorem to prove the savings amount
theorem Loisa_saves_70 : amount_saved = 70 := by
  -- Direct calculation or further proof steps here
  sorry

end Loisa_saves_70_l40_40548


namespace opposite_of_neg_two_thirds_l40_40888

theorem opposite_of_neg_two_thirds : - (- (2 / 3) : ℚ) = (2 / 3 : ℚ) :=
by
  sorry

end opposite_of_neg_two_thirds_l40_40888


namespace coefficient_x3_l40_40243

noncomputable def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem coefficient_x3 (n k : ℕ) (x : ℤ) :
  let expTerm : ℤ := 1 - x + (1 / x^2017)
  let expansion := fun (k : ℕ) => binomial n k • ((1 - x)^(n - k) * (1 / x^2017)^k)
  (n = 9) → (k = 3) →
  (expansion k) = -84 :=
  by
    intros
    sorry

end coefficient_x3_l40_40243


namespace product_of_three_consecutive_integers_is_multiple_of_6_l40_40270

theorem product_of_three_consecutive_integers_is_multiple_of_6 (n : ℕ) (h : n > 0) :
    ∃ k : ℕ, n * (n + 1) * (n + 2) = 6 * k :=
by
  sorry

end product_of_three_consecutive_integers_is_multiple_of_6_l40_40270


namespace ratio_of_sums_eq_neg_sqrt_2_l40_40169

open Real

theorem ratio_of_sums_eq_neg_sqrt_2
    (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 6) :
    (x + y) / (x - y) = -Real.sqrt 2 :=
by sorry

end ratio_of_sums_eq_neg_sqrt_2_l40_40169


namespace ellipse_condition_range_k_l40_40594

theorem ellipse_condition_range_k (k : ℝ) : 
  (2 - k > 0) ∧ (3 + k > 0) ∧ (2 - k ≠ 3 + k) → -3 < k ∧ k < 2 := 
by 
  sorry

end ellipse_condition_range_k_l40_40594


namespace positive_difference_is_zero_l40_40652

-- Definitions based on conditions
def jo_sum (n : ℕ) : ℕ := (n * (n + 1)) / 2

def rounded_to_nearest_5 (x : ℕ) : ℕ :=
  if x % 5 = 0 then x
  else (x / 5) * 5 + (if x % 5 >= 3 then 5 else 0)

def alan_sum (n : ℕ) : ℕ :=
  (List.range (n + 1)).map rounded_to_nearest_5 |>.sum

-- Theorem based on question and correct answer
theorem positive_difference_is_zero :
  jo_sum 120 - alan_sum 120 = 0 := sorry

end positive_difference_is_zero_l40_40652


namespace path_area_and_cost_correct_l40_40661

-- Define the given conditions
def length_field : ℝ := 75
def width_field : ℝ := 55
def path_width : ℝ := 2.5
def cost_per_sq_meter : ℝ := 7

-- Calculate new dimensions including the path
def length_including_path : ℝ := length_field + 2 * path_width
def width_including_path : ℝ := width_field + 2 * path_width

-- Calculate areas
def area_entire_field : ℝ := length_including_path * width_including_path
def area_grass_field : ℝ := length_field * width_field
def area_path : ℝ := area_entire_field - area_grass_field

-- Calculate cost
def cost_of_path : ℝ := area_path * cost_per_sq_meter

theorem path_area_and_cost_correct : 
  area_path = 675 ∧ cost_of_path = 4725 :=
by
  sorry

end path_area_and_cost_correct_l40_40661


namespace ratio_of_altitude_to_radius_l40_40227

theorem ratio_of_altitude_to_radius (r R h : ℝ)
  (hR : R = 2 * r)
  (hV : (1/3) * π * R^2 * h = (1/3) * (4/3) * π * r^3) :
  h / R = 1 / 6 := by
  sorry

end ratio_of_altitude_to_radius_l40_40227


namespace ratio_area_rectangle_triangle_l40_40297

-- Define the lengths L and W as positive real numbers
variables {L W : ℝ} (hL : L > 0) (hW : W > 0)

-- Define the area of the rectangle
noncomputable def area_rectangle (L W : ℝ) : ℝ := L * W

-- Define the area of the triangle with base L and height W
noncomputable def area_triangle (L W : ℝ) : ℝ := (1 / 2) * L * W

-- Define the ratio between the area of the rectangle and the area of the triangle
noncomputable def area_ratio (L W : ℝ) : ℝ := area_rectangle L W / area_triangle L W

-- Prove that this ratio is equal to 2
theorem ratio_area_rectangle_triangle : area_ratio L W = 2 := by sorry

end ratio_area_rectangle_triangle_l40_40297


namespace range_of_y_l40_40671

theorem range_of_y (y : ℝ) (hy : y < 0) (h : ⌈y⌉ * ⌊y⌋ = 132) : -12 < y ∧ y < -11 := 
by 
  sorry

end range_of_y_l40_40671


namespace rowing_time_to_place_and_back_l40_40034

def speed_man_still_water : ℝ := 8 -- km/h
def speed_river : ℝ := 2 -- km/h
def total_distance : ℝ := 7.5 -- km

theorem rowing_time_to_place_and_back :
  let V_m := speed_man_still_water
  let V_r := speed_river
  let D := total_distance / 2
  let V_up := V_m - V_r
  let V_down := V_m + V_r
  let T_up := D / V_up
  let T_down := D / V_down
  T_up + T_down = 1 :=
by
  sorry

end rowing_time_to_place_and_back_l40_40034


namespace elevator_max_weight_capacity_l40_40636

theorem elevator_max_weight_capacity 
  (num_adults : ℕ)
  (weight_adult : ℕ)
  (num_children : ℕ)
  (weight_child : ℕ)
  (max_next_person_weight : ℕ) 
  (H_adults : num_adults = 3)
  (H_weight_adult : weight_adult = 140)
  (H_children : num_children = 2)
  (H_weight_child : weight_child = 64)
  (H_max_next : max_next_person_weight = 52) : 
  num_adults * weight_adult + num_children * weight_child + max_next_person_weight = 600 := 
by
  sorry

end elevator_max_weight_capacity_l40_40636


namespace VIP_ticket_price_l40_40394

variable (total_savings : ℕ) 
variable (num_VIP_tickets : ℕ)
variable (num_regular_tickets : ℕ)
variable (price_per_regular_ticket : ℕ)
variable (remaining_savings : ℕ)

theorem VIP_ticket_price 
  (h1 : total_savings = 500)
  (h2 : num_VIP_tickets = 2)
  (h3 : num_regular_tickets = 3)
  (h4 : price_per_regular_ticket = 50)
  (h5 : remaining_savings = 150) :
  (total_savings - remaining_savings) - (num_regular_tickets * price_per_regular_ticket) = num_VIP_tickets * 100 := 
by
  sorry

end VIP_ticket_price_l40_40394


namespace sequence_problem_proof_l40_40891

-- Define the sequence terms, using given conditions
def a_1 : ℕ := 1
def a_2 : ℕ := 2
def a_3 : ℕ := a_1 + a_2
def a_4 : ℕ := a_2 + a_3
def x : ℕ := a_3 + a_4

-- Prove that x = 8
theorem sequence_problem_proof : x = 8 := 
by
  sorry

end sequence_problem_proof_l40_40891


namespace number_of_white_balls_l40_40839

theorem number_of_white_balls (x : ℕ) (h1 : 3 + x ≠ 0) (h2 : (3 : ℚ) / (3 + x) = 1 / 5) : x = 12 :=
sorry

end number_of_white_balls_l40_40839


namespace squares_in_rectangle_l40_40795

theorem squares_in_rectangle (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a ≤ 1) (h5 : b ≤ 1) (h6 : c ≤ 1) (h7 : a + b + c = 2)  : 
  a + b + c ≤ 2 := sorry

end squares_in_rectangle_l40_40795


namespace ball_highest_point_at_l40_40989

noncomputable def h (a b t : ℝ) : ℝ := a * t^2 + b * t

theorem ball_highest_point_at (a b : ℝ) :
  (h a b 3 = h a b 7) →
  t = 4.9 :=
by
  sorry

end ball_highest_point_at_l40_40989


namespace intersection_points_eq_2_l40_40944

def eq1 (x y : ℝ) : Prop := (x - 2 * y + 3) * (4 * x + y - 5) = 0
def eq2 (x y : ℝ) : Prop := (x + 2 * y - 3) * (3 * x - 4 * y + 6) = 0

theorem intersection_points_eq_2 : ∃ (points : Finset (ℝ × ℝ)), 
  (∀ p ∈ points, eq1 p.1 p.2 ∧ eq2 p.1 p.2) ∧ points.card = 2 := 
sorry

end intersection_points_eq_2_l40_40944


namespace original_remainder_when_dividing_by_44_is_zero_l40_40583

theorem original_remainder_when_dividing_by_44_is_zero 
  (N R : ℕ) 
  (Q : ℕ) 
  (h1 : N = 44 * 432 + R) 
  (h2 : N = 34 * Q + 2) 
  : R = 0 := 
sorry

end original_remainder_when_dividing_by_44_is_zero_l40_40583


namespace difference_between_numbers_l40_40241

theorem difference_between_numbers :
  ∃ S : ℝ, L = 1650 ∧ L = 6 * S + 15 ∧ L - S = 1377.5 :=
sorry

end difference_between_numbers_l40_40241


namespace hot_dog_remainder_l40_40261

theorem hot_dog_remainder : 35252983 % 6 = 1 :=
by
  sorry

end hot_dog_remainder_l40_40261


namespace no_such_rectangle_exists_l40_40976

theorem no_such_rectangle_exists :
  ¬(∃ (x y : ℝ), (∃ a b c d : ℕ, x = a + b * Real.sqrt 3 ∧ y = c + d * Real.sqrt 3) ∧ 
                (x * y = (3 * Real.sqrt 3) / 2 + n * (Real.sqrt 3 / 2))) :=
sorry

end no_such_rectangle_exists_l40_40976


namespace polynomial_has_no_real_roots_l40_40298

theorem polynomial_has_no_real_roots :
  ∀ x : ℝ, x^8 - x^7 + 2*x^6 - 2*x^5 + 3*x^4 - 3*x^3 + 4*x^2 - 4*x + 5/2 ≠ 0 :=
by
  sorry

end polynomial_has_no_real_roots_l40_40298


namespace number_of_possible_values_of_s_l40_40876

noncomputable def s := {s : ℚ | ∃ w x y z : ℕ, s = w / 1000 + x / 10000 + y / 100000 + z / 1000000 ∧ w < 10 ∧ x < 10 ∧ y < 10 ∧ z < 10}

theorem number_of_possible_values_of_s (s_approx : s → ℚ → Prop) (h_s_approx : ∀ s, s_approx s (3 / 11)) :
  ∃ n : ℕ, n = 266 :=
by
  sorry

end number_of_possible_values_of_s_l40_40876


namespace avg_weight_BC_l40_40358

variable (A B C : ℝ)

def totalWeight_ABC := 3 * 45
def totalWeight_AB := 2 * 40
def weight_B := 31

theorem avg_weight_BC : ((B + C) / 2) = 43 :=
  by
    have totalWeight_ABC_eq : A + B + C = totalWeight_ABC := by sorry
    have totalWeight_AB_eq : A + B = totalWeight_AB := by sorry
    have weight_B_eq : B = weight_B := by sorry
    sorry

end avg_weight_BC_l40_40358


namespace sum_angles_of_two_triangles_l40_40961

theorem sum_angles_of_two_triangles (a1 a3 a5 a2 a4 a6 : ℝ) 
  (hABC : a1 + a3 + a5 = 180) (hDEF : a2 + a4 + a6 = 180) : 
  a1 + a2 + a3 + a4 + a5 + a6 = 360 :=
by
  sorry

end sum_angles_of_two_triangles_l40_40961


namespace positive_integer_solutions_condition_l40_40627

theorem positive_integer_solutions_condition (a : ℕ) (A B : ℝ) :
  (∃ (x y z : ℕ), x^2 + y^2 + z^2 = (13 * a)^2 ∧
  x^2 * (A * x^2 + B * y^2) + y^2 * (A * y^2 + B * z^2) + z^2 * (A * z^2 + B * x^2) = (1/4) * (2 * A + B) * (13 * a)^4)
  ↔ A = (1 / 2) * B := 
sorry

end positive_integer_solutions_condition_l40_40627


namespace poodle_barks_count_l40_40629

-- Define the conditions as hypothesis
variables (poodle_barks terrier_barks terrier_hushes : ℕ)

-- Define the conditions
def condition1 : Prop :=
  poodle_barks = 2 * terrier_barks

def condition2 : Prop :=
  terrier_hushes = terrier_barks / 2

def condition3 : Prop :=
  terrier_hushes = 6

-- The theorem we need to prove
theorem poodle_barks_count (poodle_barks terrier_barks terrier_hushes : ℕ)
  (h1 : condition1 poodle_barks terrier_barks)
  (h2 : condition2 terrier_barks terrier_hushes)
  (h3 : condition3 terrier_hushes) :
  poodle_barks = 24 :=
by
  -- Proof is not required as per instructions
  sorry

end poodle_barks_count_l40_40629


namespace industrial_park_investment_l40_40044

noncomputable def investment_in_projects : Prop :=
  ∃ (x : ℝ), 0.054 * x + 0.0828 * (2000 - x) = 122.4 ∧ x = 1500 ∧ (2000 - x) = 500

theorem industrial_park_investment :
  investment_in_projects :=
by
  have h : ∃ (x : ℝ), 0.054 * x + 0.0828 * (2000 - x) = 122.4 ∧ x = 1500 ∧ (2000 - x) = 500 := 
    sorry
  exact h

end industrial_park_investment_l40_40044


namespace max_mn_l40_40791

theorem max_mn (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (m n : ℝ)
  (h₂ : 2 * m + n = 2) : m * n ≤ 4 / 9 :=
by
  sorry

end max_mn_l40_40791


namespace erica_pie_fraction_as_percentage_l40_40042

theorem erica_pie_fraction_as_percentage (apple_pie_fraction : ℚ) (cherry_pie_fraction : ℚ) 
  (h1 : apple_pie_fraction = 1 / 5) 
  (h2 : cherry_pie_fraction = 3 / 4) 
  (common_denominator : ℚ := 20) : 
  (apple_pie_fraction + cherry_pie_fraction) * 100 = 95 :=
by
  sorry

end erica_pie_fraction_as_percentage_l40_40042


namespace janice_remaining_hours_l40_40127

def homework_time : ℕ := 30
def clean_room_time : ℕ := homework_time / 2
def walk_dog_time : ℕ := homework_time + 5
def trash_time : ℕ := homework_time / 6
def total_task_time : ℕ := homework_time + clean_room_time + walk_dog_time + trash_time
def remaining_minutes : ℕ := 35

theorem janice_remaining_hours : (remaining_minutes : ℚ) / 60 = (7 / 12 : ℚ) :=
by
  sorry

end janice_remaining_hours_l40_40127


namespace chris_pennies_count_l40_40660

theorem chris_pennies_count (a c : ℤ) 
  (h1 : c + 2 = 4 * (a - 2)) 
  (h2 : c - 2 = 3 * (a + 2)) : 
  c = 62 := 
by 
  -- The actual proof is omitted
  sorry

end chris_pennies_count_l40_40660


namespace set_difference_NM_l40_40464

open Set

def setDifference (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem set_difference_NM :
  let M := {1, 2, 3, 4, 5}
  let N := {1, 2, 3, 7}
  setDifference N M = {7} :=
by
  sorry

end set_difference_NM_l40_40464


namespace sum_of_g1_values_l40_40058

noncomputable def g : Polynomial ℝ := sorry

theorem sum_of_g1_values :
  (∀ x : ℝ, x ≠ 0 → g.eval (x-1) + g.eval x + g.eval (x+1) = (g.eval x)^2 / (4036 * x)) →
  g.degree ≠ 0 →
  g.eval 1 = 12108 :=
by
  sorry

end sum_of_g1_values_l40_40058


namespace marbles_left_l40_40966

def initial_marbles : ℕ := 100
def percent_t_to_Theresa : ℕ := 25
def percent_t_to_Elliot : ℕ := 10

theorem marbles_left (w t e : ℕ) (h_w : w = initial_marbles)
                                 (h_t : t = percent_t_to_Theresa)
                                 (h_e : e = percent_t_to_Elliot) : w - ((t * w) / 100 + (e * w) / 100) = 65 :=
by
  rw [h_w, h_t, h_e]
  sorry

end marbles_left_l40_40966


namespace division_then_multiplication_l40_40239

theorem division_then_multiplication : (180 / 6) * 3 = 90 := 
by
  have step1 : 180 / 6 = 30 := sorry
  have step2 : 30 * 3 = 90 := sorry
  sorry

end division_then_multiplication_l40_40239


namespace div_operation_example_l40_40881

theorem div_operation_example : ((180 / 6) / 3) = 10 := by
  sorry

end div_operation_example_l40_40881


namespace part1_part2_l40_40993

def f (x : ℝ) : ℝ :=
  abs (x - 1) + 2 * abs (x + 5)

theorem part1 : ∀ x, f x < 10 ↔ (x > -19 / 3 ∧ x ≤ -5) ∨ (-5 < x ∧ x < -1) :=
  sorry

theorem part2 (a b x : ℝ) (ha : abs a < 3) (hb : abs b < 3) :
  abs (a + b) + abs (a - b) < f x :=
  sorry

end part1_part2_l40_40993


namespace jaya_rank_from_bottom_l40_40569

theorem jaya_rank_from_bottom (n t : ℕ) (h_n : n = 53) (h_t : t = 5) : n - t + 1 = 50 := by
  sorry

end jaya_rank_from_bottom_l40_40569


namespace trajectory_range_k_l40_40653

-- Condition Definitions
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)
def N (x : ℝ) : ℝ × ℝ := (x, 0)
def vector_MN (x y : ℝ) : ℝ × ℝ := (0, -y)
def vector_AN (x : ℝ) : ℝ × ℝ := (x + 1, 0)
def vector_BN (x : ℝ) : ℝ × ℝ := (x - 1, 0)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Problem 1: Prove the trajectory equation
theorem trajectory (x y : ℝ) (h : (vector_MN x y).1^2 + (vector_MN x y).2^2 = dot_product (vector_AN x) (vector_BN x)) :
  x^2 - y^2 = 1 :=
sorry

-- Problem 2: Prove the range of k
theorem range_k (k : ℝ) :
  (∃ x y : ℝ, y = k * x - 1 ∧ x^2 - y^2 = 1) ↔ -Real.sqrt 2 ≤ k ∧ k ≤ Real.sqrt 2 :=
sorry

end trajectory_range_k_l40_40653


namespace complement_of_A_in_U_l40_40443

def U : Set ℝ := Set.univ
def A : Set ℝ := { x | |x - 1| > 2 }

theorem complement_of_A_in_U : 
  ∀ x, x ∈ U → x ∈ U \ A ↔ -1 ≤ x ∧ x ≤ 3 := by
  sorry

end complement_of_A_in_U_l40_40443


namespace lucy_age_l40_40249

theorem lucy_age (Inez_age : ℕ) (Zack_age : ℕ) (Jose_age : ℕ) (Lucy_age : ℕ) 
  (h1 : Inez_age = 18) 
  (h2 : Zack_age = Inez_age + 4) 
  (h3 : Jose_age = Zack_age - 6) 
  (h4 : Lucy_age = Jose_age + 2) : 
  Lucy_age = 18 := by
sorry

end lucy_age_l40_40249


namespace find_a_l40_40011

theorem find_a (a x : ℝ) (h1 : 3 * x + 5 = 11) (h2 : 6 * x + 3 * a = 22) : a = 10 / 3 :=
by
  -- the proof will go here
  sorry

end find_a_l40_40011


namespace binomial_square_evaluation_l40_40553

theorem binomial_square_evaluation : 15^2 + 2 * 15 * 3 + 3^2 = 324 := by
  sorry

end binomial_square_evaluation_l40_40553


namespace LukaLemonadeSolution_l40_40017

def LukaLemonadeProblem : Prop :=
  ∃ (L S W : ℕ), 
    (S = 3 * L) ∧
    (W = 3 * S) ∧
    (L = 4) ∧
    (W = 36)

theorem LukaLemonadeSolution : LukaLemonadeProblem :=
  by sorry

end LukaLemonadeSolution_l40_40017


namespace no_solution_iff_n_eq_neg2_l40_40428

noncomputable def has_no_solution (n : ℝ) : Prop :=
  ∀ x y z : ℝ, ¬ (n * x + y + z = 2 ∧ 
                  x + n * y + z = 2 ∧ 
                  x + y + n * z = 2)

theorem no_solution_iff_n_eq_neg2 (n : ℝ) : has_no_solution n ↔ n = -2 := by
  sorry

end no_solution_iff_n_eq_neg2_l40_40428


namespace math_problem_l40_40389

noncomputable def x : ℝ := (Real.sqrt 5 + 1) / 2
noncomputable def y : ℝ := (Real.sqrt 5 - 1) / 2

theorem math_problem :
    x^3 * y + 2 * x^2 * y^2 + x * y^3 = 5 := 
by
  sorry

end math_problem_l40_40389


namespace doors_per_apartment_l40_40507

def num_buildings : ℕ := 2
def num_floors_per_building : ℕ := 12
def num_apt_per_floor : ℕ := 6
def total_num_doors : ℕ := 1008

theorem doors_per_apartment : total_num_doors / (num_buildings * num_floors_per_building * num_apt_per_floor) = 7 :=
by
  sorry

end doors_per_apartment_l40_40507


namespace election_threshold_l40_40733

theorem election_threshold (total_votes geoff_percent_more_votes : ℕ) (geoff_vote_percent : ℚ) (geoff_votes_needed extra_votes_needed : ℕ) (threshold_percent : ℚ) :
  total_votes = 6000 → 
  geoff_vote_percent = 0.5 → 
  geoff_votes_needed = (geoff_vote_percent / 100) * total_votes →
  extra_votes_needed = 3000 → 
  (geoff_votes_needed + extra_votes_needed) / total_votes * 100 = threshold_percent →
  threshold_percent = 50.5 := 
by
  intros total_votes_eq geoff_vote_percent_eq geoff_votes_needed_eq extra_votes_needed_eq threshold_eq
  sorry

end election_threshold_l40_40733


namespace smallest_triangle_perimeter_l40_40982

theorem smallest_triangle_perimeter :
  ∃ (y : ℕ), (y % 2 = 0) ∧ (y < 17) ∧ (y > 3) ∧ (7 + 10 + y = 21) :=
by
  sorry

end smallest_triangle_perimeter_l40_40982


namespace mom_age_when_jayson_born_l40_40685

theorem mom_age_when_jayson_born (jayson_age dad_age mom_age : ℕ) 
  (h1 : jayson_age = 10) 
  (h2 : dad_age = 4 * jayson_age)
  (h3 : mom_age = dad_age - 2) :
  mom_age - jayson_age = 28 :=
by
  sorry

end mom_age_when_jayson_born_l40_40685


namespace albert_runs_track_l40_40325

theorem albert_runs_track (x : ℕ) (track_distance : ℕ) (total_distance : ℕ) (additional_laps : ℕ) 
(h1 : track_distance = 9)
(h2 : total_distance = 99)
(h3 : additional_laps = 5)
(h4 : total_distance = track_distance * x + track_distance * additional_laps) :
x = 6 :=
by
  sorry

end albert_runs_track_l40_40325


namespace member_number_property_l40_40186

theorem member_number_property :
  ∃ (country : Fin 6) (member_number : Fin 1978),
    (∀ (i j : Fin 1978), i ≠ j → member_number ≠ i + j) ∨
    (∀ (k : Fin 1978), member_number ≠ 2 * k) :=
by
  sorry

end member_number_property_l40_40186


namespace range_of_a_if_f_increasing_l40_40440

theorem range_of_a_if_f_increasing (a : ℝ) :
  (∀ x : ℝ, 3*x^2 + 3*a ≥ 0) → (a ≥ 0) :=
sorry

end range_of_a_if_f_increasing_l40_40440


namespace options_necessarily_positive_l40_40141

variable (x y z : ℝ)

theorem options_necessarily_positive (h₁ : -1 < x) (h₂ : x < 0) (h₃ : 0 < y) (h₄ : y < 1) (h₅ : 2 < z) (h₆ : z < 3) :
  y + x^2 * z > 0 ∧
  y + x^2 > 0 ∧
  y + y^2 > 0 ∧
  y + 2 * z > 0 := 
  sorry

end options_necessarily_positive_l40_40141


namespace fraction_is_one_fifth_l40_40527

theorem fraction_is_one_fifth
  (x a b : ℤ)
  (hx : x^2 = 25)
  (h2x : 2 * x = a * x / b + 9) :
  a = 1 ∧ b = 5 :=
by
  sorry

end fraction_is_one_fifth_l40_40527


namespace largest_multiple_of_15_less_than_500_l40_40692

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), (n < 500) ∧ (15 ∣ n) ∧ (∀ m : ℕ, (m < 500) ∧ (15 ∣ m) → m ≤ n) ∧ n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l40_40692


namespace geometric_sum_S12_l40_40441

theorem geometric_sum_S12 (a r : ℝ) (h₁ : r ≠ 1) (S4_eq : a * (1 - r^4) / (1 - r) = 24) (S8_eq : a * (1 - r^8) / (1 - r) = 36) : a * (1 - r^12) / (1 - r) = 42 := 
sorry

end geometric_sum_S12_l40_40441


namespace product_of_consecutive_triangular_not_square_infinite_larger_triangular_numbers_square_product_l40_40824

section TriangularNumbers

-- Define triangular numbers
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

-- Statement 1: The product of two consecutive triangular numbers is not a perfect square
theorem product_of_consecutive_triangular_not_square (n : ℕ) (hn : n > 0) :
  ¬ ∃ m : ℕ, triangular (n - 1) * triangular n = m * m := by
  sorry

-- Statement 2: There exist infinitely many larger triangular numbers such that the product with t_n is a perfect square
theorem infinite_larger_triangular_numbers_square_product (n : ℕ) :
  ∃ᶠ m in at_top, ∃ k : ℕ, triangular n * triangular m = k * k := by
  sorry

end TriangularNumbers

end product_of_consecutive_triangular_not_square_infinite_larger_triangular_numbers_square_product_l40_40824


namespace max_value_cos2_sin_l40_40937

noncomputable def max_cos2_sin (x : Real) : Real := 
  (Real.cos x) ^ 2 + Real.sin x

theorem max_value_cos2_sin : 
  ∃ x : Real, (-1 ≤ Real.sin x) ∧ (Real.sin x ≤ 1) ∧ 
    max_cos2_sin x = 5 / 4 :=
sorry

end max_value_cos2_sin_l40_40937


namespace range_of_theta_l40_40767

theorem range_of_theta (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi)
    (h_ineq : 3 * (Real.sin θ ^ 5 + Real.cos (2 * θ) ^ 5) > 5 * (Real.sin θ ^ 3 + Real.cos (2 * θ) ^ 3)) :
    θ ∈ Set.Ico (7 * Real.pi / 6) (11 * Real.pi / 6) :=
sorry

end range_of_theta_l40_40767


namespace sum_mnp_is_405_l40_40690

theorem sum_mnp_is_405 :
  let C1_radius := 4
  let C2_radius := 10
  let C3_radius := C1_radius + C2_radius
  let chord_length := (8 * Real.sqrt 390) / 7
  ∃ m n p : ℕ,
    m * Real.sqrt n / p = chord_length ∧
    m.gcd p = 1 ∧
    (∀ k : ℕ, k^2 ∣ n → k = 1) ∧
    m + n + p = 405 :=
by
  sorry

end sum_mnp_is_405_l40_40690


namespace weight_of_fourth_dog_l40_40778

theorem weight_of_fourth_dog (y x : ℝ) : 
  (25 + 31 + 35 + x) / 4 = (25 + 31 + 35 + x + y) / 5 → 
  x = -91 - 5 * y :=
by
  sorry

end weight_of_fourth_dog_l40_40778


namespace smallest_multiple_6_15_l40_40931

theorem smallest_multiple_6_15 (b : ℕ) (hb1 : b % 6 = 0) (hb2 : b % 15 = 0) :
  ∃ (b : ℕ), (b > 0) ∧ (b % 6 = 0) ∧ (b % 15 = 0) ∧ (∀ x : ℕ, (x > 0) ∧ (x % 6 = 0) ∧ (x % 15 = 0) → x ≥ b) :=
sorry

end smallest_multiple_6_15_l40_40931


namespace functional_equation_solution_l40_40559

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x^2 * y) = f (x * y) + y * f (f x + y)) →
  (∀ y : ℝ, f y = 0) :=
by
  intro h
  sorry

end functional_equation_solution_l40_40559
