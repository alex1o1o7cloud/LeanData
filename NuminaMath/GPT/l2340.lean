import Mathlib

namespace inscribed_square_properties_l2340_234061

theorem inscribed_square_properties (r : ℝ) (s : ℝ) (d : ℝ) (A_circle : ℝ) (A_square : ℝ) (total_diagonals : ℝ) (hA_circle : A_circle = 324 * Real.pi) (hr : r = Real.sqrt 324) (hd : d = 2 * r) (hs : s = d / Real.sqrt 2) (hA_square : A_square = s ^ 2) (htotal_diagonals : total_diagonals = 2 * d) :
  A_square = 648 ∧ total_diagonals = 72 :=
by sorry

end inscribed_square_properties_l2340_234061


namespace polynomial_value_at_4_l2340_234090

def f (x : ℝ) : ℝ := 9 + 15 * x - 8 * x^2 - 20 * x^3 + 6 * x^4 + 3 * x^5

theorem polynomial_value_at_4 :
  f 4 = 3269 :=
by
  sorry

end polynomial_value_at_4_l2340_234090


namespace new_tax_rate_is_correct_l2340_234010

noncomputable def new_tax_rate (old_rate : ℝ) (income : ℝ) (savings : ℝ) : ℝ := 
  let old_tax := old_rate * income / 100
  let new_tax := (income - savings) / income * old_tax
  let rate := new_tax / income * 100
  rate

theorem new_tax_rate_is_correct :
  ∀ (income : ℝ) (old_rate : ℝ) (savings : ℝ),
    old_rate = 42 →
    income = 34500 →
    savings = 4830 →
    new_tax_rate old_rate income savings = 28 := 
by
  intros income old_rate savings h1 h2 h3
  sorry

end new_tax_rate_is_correct_l2340_234010


namespace evaluate_expression_l2340_234087

theorem evaluate_expression (x : ℕ) (h : x = 3) : x + x^2 * (x^(x^2)) = 177150 := by
  sorry

end evaluate_expression_l2340_234087


namespace brad_age_proof_l2340_234014

theorem brad_age_proof :
  ∀ (Shara_age Jaymee_age Average_age Brad_age : ℕ),
  Jaymee_age = 2 * Shara_age + 2 →
  Average_age = (Shara_age + Jaymee_age) / 2 →
  Brad_age = Average_age - 3 →
  Shara_age = 10 →
  Brad_age = 13 :=
by
  intros Shara_age Jaymee_age Average_age Brad_age
  intro h1 h2 h3 h4
  sorry

end brad_age_proof_l2340_234014


namespace positive_integers_congruent_to_2_mod_7_lt_500_count_l2340_234024

theorem positive_integers_congruent_to_2_mod_7_lt_500_count : 
  ∃ n : ℕ, n = 72 ∧ ∀ k : ℕ, (k < n → (∃ m : ℕ, (m < 500 ∧ m % 7 = 2) ∧ m = 2 + 7 * k)) := 
by
  sorry

end positive_integers_congruent_to_2_mod_7_lt_500_count_l2340_234024


namespace trees_in_one_row_l2340_234066

variable (total_trees_cleaned : ℕ)
variable (trees_per_row : ℕ)

theorem trees_in_one_row (h1 : total_trees_cleaned = 20) (h2 : trees_per_row = 5) :
  (total_trees_cleaned / trees_per_row) = 4 :=
by
  sorry

end trees_in_one_row_l2340_234066


namespace A_finish_time_l2340_234064

theorem A_finish_time {A_work B_work C_work : ℝ} 
  (h1 : A_work + B_work + C_work = 1/4)
  (h2 : B_work = 1/24)
  (h3 : C_work = 1/8) :
  1 / A_work = 12 := by
  sorry

end A_finish_time_l2340_234064


namespace math_problem_l2340_234052

theorem math_problem : 8 / 4 - 3 - 10 + 3 * 7 = 10 := by
  sorry

end math_problem_l2340_234052


namespace chocolate_bars_in_small_box_l2340_234044

-- Given conditions
def num_small_boxes : ℕ := 21
def total_chocolate_bars : ℕ := 525

-- Statement to prove
theorem chocolate_bars_in_small_box : total_chocolate_bars / num_small_boxes = 25 := by
  sorry

end chocolate_bars_in_small_box_l2340_234044


namespace tom_needs_noodle_packages_l2340_234017

def beef_pounds : ℕ := 10
def noodle_multiplier : ℕ := 2
def initial_noodles : ℕ := 4
def package_weight : ℕ := 2

theorem tom_needs_noodle_packages :
  (noodle_multiplier * beef_pounds - initial_noodles) / package_weight = 8 := 
by 
  -- Faithfully skipping the solution steps
  sorry

end tom_needs_noodle_packages_l2340_234017


namespace distinct_positive_integers_criteria_l2340_234091

theorem distinct_positive_integers_criteria (x y z : ℕ) (hx : x ≠ y) (hy : y ≠ z) (hz : z ≠ x)
  (hxyz_div : x * y * z ∣ (x * y - 1) * (y * z - 1) * (z * x - 1)) :
  (x, y, z) = (2, 3, 5) ∨ (x, y, z) = (2, 5, 3) ∨ (x, y, z) = (3, 2, 5) ∨
  (x, y, z) = (3, 5, 2) ∨ (x, y, z) = (5, 2, 3) ∨ (x, y, z) = (5, 3, 2) :=
by sorry

end distinct_positive_integers_criteria_l2340_234091


namespace optimal_floor_optimal_floor_achieved_at_three_l2340_234038

theorem optimal_floor : ∀ (n : ℕ), n > 0 → (n + 9 / n : ℝ) ≥ 6 := sorry

theorem optimal_floor_achieved_at_three : ∃ n : ℕ, (n > 0 ∧ (n + 9 / n : ℝ) = 6) := sorry

end optimal_floor_optimal_floor_achieved_at_three_l2340_234038


namespace son_l2340_234054

variable (M S : ℕ)

theorem son's_age (h1 : M = 4 * S) (h2 : (M - 3) + (S - 3) = 49) : S = 11 :=
by
  sorry

end son_l2340_234054


namespace base8_base6_positive_integer_l2340_234034

theorem base8_base6_positive_integer (C D N : ℕ)
  (base8: N = 8 * C + D)
  (base6: N = 6 * D + C)
  (valid_C_base8: C < 8)
  (valid_D_base6: D < 6)
  (valid_C_D: 7 * C = 5 * D)
: N = 43 := by
  sorry

end base8_base6_positive_integer_l2340_234034


namespace goose_eggs_count_l2340_234074

theorem goose_eggs_count (E : ℝ) (h1 : 1 / 4 * E = (1 / 4) * E)
  (h2 : 4 / 5 * (1 / 4) * E = (4 / 5) * (1 / 4) * E)
  (h3 : 3 / 5 * (4 / 5) * (1 / 4) * E = 120)
  (h4 : 120 = 120)
  : E = 800 :=
by
  sorry

end goose_eggs_count_l2340_234074


namespace trains_crossing_time_l2340_234098

noncomputable def TrainA_length := 200  -- meters
noncomputable def TrainA_time := 15  -- seconds
noncomputable def TrainB_length := 300  -- meters
noncomputable def TrainB_time := 25  -- seconds

noncomputable def Speed (length : ℕ) (time : ℕ) := (length : ℝ) / (time : ℝ)

noncomputable def TrainA_speed := Speed TrainA_length TrainA_time
noncomputable def TrainB_speed := Speed TrainB_length TrainB_time

noncomputable def relative_speed := TrainA_speed + TrainB_speed
noncomputable def total_distance := (TrainA_length : ℝ) + (TrainB_length : ℝ)

noncomputable def crossing_time := total_distance / relative_speed

theorem trains_crossing_time :
  (crossing_time : ℝ) = 500 / 25.33 :=
sorry

end trains_crossing_time_l2340_234098


namespace volume_of_released_gas_l2340_234083

def mol_co2 : ℝ := 2.4
def molar_volume : ℝ := 22.4

theorem volume_of_released_gas : mol_co2 * molar_volume = 53.76 := by
  sorry -- proof to be filled in

end volume_of_released_gas_l2340_234083


namespace number_of_pizzas_ordered_l2340_234012

-- Define the total number of people
def total_people : ℕ := 6

-- Define the number of slices per pizza
def slices_per_pizza : ℕ := 8

-- Define the number of slices each person ate
def slices_per_person : ℕ := 4

-- Define the total number of slices eaten
def total_slices_eaten : ℕ := total_people * slices_per_person

-- Prove that the number of pizzas needed is 3
theorem number_of_pizzas_ordered : total_slices_eaten / slices_per_pizza = 3 := by
  sorry

end number_of_pizzas_ordered_l2340_234012


namespace simplify_fraction_part1_simplify_fraction_part2_l2340_234060

-- Part 1
theorem simplify_fraction_part1 (x : ℝ) (h1 : x ≠ -2) :
  (x^2 / (x + 2)) + ((4 * x + 4) / (x + 2)) = x + 2 :=
sorry

-- Part 2
theorem simplify_fraction_part2 (x : ℝ) (h1 : x ≠ 1) :
  (x^2 / ((x - 1)^2)) / ((1 - 2 * x) / (x - 1) - (x - 1)) = -1 / (x - 1) :=
sorry

end simplify_fraction_part1_simplify_fraction_part2_l2340_234060


namespace calculate_fraction_product_l2340_234057

noncomputable def b8 := 2 * (8^2) + 6 * (8^1) + 2 * (8^0) -- 262_8 in base 10
noncomputable def b4 := 1 * (4^1) + 3 * (4^0) -- 13_4 in base 10
noncomputable def b7 := 1 * (7^2) + 4 * (7^1) + 4 * (7^0) -- 144_7 in base 10
noncomputable def b5 := 2 * (5^1) + 4 * (5^0) -- 24_5 in base 10

theorem calculate_fraction_product : 
  ((b8 : ℕ) / (b4 : ℕ)) * ((b7 : ℕ) / (b5 : ℕ)) = 147 :=
by
  sorry

end calculate_fraction_product_l2340_234057


namespace recipe_total_cups_l2340_234032

noncomputable def total_cups (butter_ratio flour_ratio sugar_ratio sugar_cups : ℕ) : ℕ :=
  let part := sugar_cups / sugar_ratio
  let butter_cups := butter_ratio * part
  let flour_cups := flour_ratio * part
  butter_cups + flour_cups + sugar_cups

theorem recipe_total_cups : 
  total_cups 2 7 5 10 = 28 :=
by
  sorry

end recipe_total_cups_l2340_234032


namespace prob1_prob2_l2340_234009

-- Definitions and conditions for Problem 1
def U : Set ℝ := {x | x ≤ 4}
def A : Set ℝ := {x | x^2 - x - 6 < 0}
def B : Set ℝ := {x | -3 < x ∧ x ≤ 3}

-- Proof Problem 1: Equivalent Lean proof statement
theorem prob1 : (U \ A) ∩ B = {-3, -2, 3} := by
  sorry

-- Definitions and conditions for Problem 2
def tan_alpha_eq_3 (α : ℝ) : Prop := Real.tan α = 3

-- Proof Problem 2: Equivalent Lean proof statement
theorem prob2 (α : ℝ) (h : tan_alpha_eq_3 α) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2 ∧
  Real.cos α ^ 2 - 3 * Real.sin α * Real.cos α = -4 / 5 := by
  sorry

end prob1_prob2_l2340_234009


namespace determine_n_l2340_234000

theorem determine_n (n : ℕ) (x : ℤ) (h : x^n + (2 + x)^n + (2 - x)^n = 0) : n = 1 :=
sorry

end determine_n_l2340_234000


namespace third_chapter_is_24_pages_l2340_234059

-- Define the total number of pages in the book
def total_pages : ℕ := 125

-- Define the number of pages in the first chapter
def first_chapter_pages : ℕ := 66

-- Define the number of pages in the second chapter
def second_chapter_pages : ℕ := 35

-- Define the number of pages in the third chapter
def third_chapter_pages : ℕ := total_pages - (first_chapter_pages + second_chapter_pages)

-- Prove that the number of pages in the third chapter is 24
theorem third_chapter_is_24_pages : third_chapter_pages = 24 := by
  sorry

end third_chapter_is_24_pages_l2340_234059


namespace linear_eq_find_m_l2340_234089

theorem linear_eq_find_m (m : ℤ) (x : ℝ) 
  (h : (m - 5) * x^(|m| - 4) + 5 = 0) 
  (h_linear : |m| - 4 = 1) 
  (h_nonzero : m - 5 ≠ 0) : m = -5 :=
by
  sorry

end linear_eq_find_m_l2340_234089


namespace find_k1_over_k2_plus_k2_over_k1_l2340_234023

theorem find_k1_over_k2_plus_k2_over_k1 (p q k k1 k2 : ℚ)
  (h1 : k * (p^2) - (2 * k - 3) * p + 7 = 0)
  (h2 : k * (q^2) - (2 * k - 3) * q + 7 = 0)
  (h3 : p ≠ 0)
  (h4 : q ≠ 0)
  (h5 : k ≠ 0)
  (h6 : k1 ≠ 0)
  (h7 : k2 ≠ 0)
  (h8 : p / q + q / p = 6 / 7)
  (h9 : (p + q) = (2 * k - 3) / k)
  (h10 : p * q = 7 / k)
  (h11 : k1 + k2 = 6)
  (h12 : k1 * k2 = 9 / 4) :
  (k1 / k2 + k2 / k1 = 14) :=
  sorry

end find_k1_over_k2_plus_k2_over_k1_l2340_234023


namespace vector_equation_solution_l2340_234040

open Real

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_equation_solution (a b x : V) (h : 3 • a + 4 • (b - x) = 0) : 
  x = (3 / 4) • a + b := 
sorry

end vector_equation_solution_l2340_234040


namespace sum_ages_l2340_234015

variable (Bob_age Carol_age : ℕ)

theorem sum_ages (h1 : Bob_age = 16) (h2 : Carol_age = 50) (h3 : Carol_age = 3 * Bob_age + 2) :
  Bob_age + Carol_age = 66 :=
by
  sorry

end sum_ages_l2340_234015


namespace pencils_ordered_l2340_234047

theorem pencils_ordered (pencils_per_student : ℕ) (number_of_students : ℕ) (total_pencils : ℕ) :
  pencils_per_student = 3 →
  number_of_students = 65 →
  total_pencils = pencils_per_student * number_of_students →
  total_pencils = 195 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end pencils_ordered_l2340_234047


namespace dave_deleted_apps_l2340_234008

theorem dave_deleted_apps :
  ∃ d : ℕ, d = 150 - 65 :=
sorry

end dave_deleted_apps_l2340_234008


namespace original_length_wire_l2340_234071

-- Define the conditions.
def length_cut_off_parts : ℕ := 10
def remaining_length_relation (L_remaining : ℕ) : Prop :=
  L_remaining = 4 * (2 * length_cut_off_parts) + 10

-- Define the theorem to prove the original length of the wire.
theorem original_length_wire (L_remaining : ℕ) (H : remaining_length_relation L_remaining) : 
  L_remaining + 2 * length_cut_off_parts = 110 :=
by 
  -- Use the given conditions
  unfold remaining_length_relation at H
  -- The proof would show that the equation holds true.
  sorry

end original_length_wire_l2340_234071


namespace find_num_trumpet_players_l2340_234033

namespace OprahWinfreyHighSchoolMarchingBand

def num_trumpet_players (total_weight : ℕ) 
  (num_clarinet : ℕ) (num_trombone : ℕ) 
  (num_tuba : ℕ) (num_drum : ℕ) : ℕ :=
(total_weight - 
  ((num_clarinet * 5) + 
  (num_trombone * 10) + 
  (num_tuba * 20) + 
  (num_drum * 15)))
  / 5

theorem find_num_trumpet_players :
  num_trumpet_players 245 9 8 3 2 = 6 :=
by
  -- calculation and reasoning steps would go here
  sorry

end OprahWinfreyHighSchoolMarchingBand

end find_num_trumpet_players_l2340_234033


namespace range_of_a_l2340_234035

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + 2 * x + 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a (f a x) ≥ 0) ↔ a ≥ (Real.sqrt 5 - 1) / 2 :=
sorry

end range_of_a_l2340_234035


namespace collinear_example_l2340_234092

structure Vector2D where
  x : ℝ
  y : ℝ

def collinear (u v : Vector2D) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ v.x = k * u.x ∧ v.y = k * u.y

def a : Vector2D := ⟨1, 2⟩
def b : Vector2D := ⟨2, 4⟩

theorem collinear_example :
  collinear a b :=
by
  sorry

end collinear_example_l2340_234092


namespace first_dig_site_date_difference_l2340_234082

-- Definitions for the conditions
def F : Int := sorry  -- The age of the first dig site
def S : Int := sorry  -- The age of the second dig site
def T : Int := sorry  -- The age of the third dig site
def Fo : Int := 8400  -- The age of the fourth dig site
def x : Int := (S - F)

-- The conditions
axiom condition1 : F = S + x
axiom condition2 : T = F + 3700
axiom condition3 : Fo = 2 * T
axiom condition4 : S = 852
axiom condition5 : S > F  -- Ensuring S is older than F for meaningfulness

-- The theorem to prove
theorem first_dig_site_date_difference : x = 352 :=
by
  -- Proof goes here
  sorry

end first_dig_site_date_difference_l2340_234082


namespace ellipse_standard_equation_l2340_234062

theorem ellipse_standard_equation (a c : ℝ) (h1 : a^2 = 13) (h2 : c^2 = 12) :
  (∃ b : ℝ, b^2 = a^2 - c^2 ∧ 
    ((∀ x y : ℝ, (x^2 / 13 + y^2 = 1)) ∨ (∀ x y : ℝ, (x^2 + y^2 / 13 = 1)))) :=
by
  sorry

end ellipse_standard_equation_l2340_234062


namespace find_r_of_tangential_cones_l2340_234072

theorem find_r_of_tangential_cones (r : ℝ) : 
  (∃ (r1 r2 r3 R : ℝ), r1 = 2 * r ∧ r2 = 3 * r ∧ r3 = 10 * r ∧ R = 15 ∧
  -- Additional conditions to ensure the three cones touch and share a slant height
  -- with the truncated cone of radius R
  true) → r = 29 :=
by
  intro h
  sorry

end find_r_of_tangential_cones_l2340_234072


namespace range_of_a_l2340_234045

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (a < x ∧ x < a + 1) → (-2 ≤ x ∧ x ≤ 2)) ↔ -2 ≤ a ∧ a ≤ 1 :=
by 
  sorry

end range_of_a_l2340_234045


namespace dad_steps_l2340_234018

theorem dad_steps (dad_steps_per_masha_steps : ℕ) (masha_steps_per_dad_steps : ℕ) (masha_steps_per_yasha_steps : ℕ) (yasha_steps_per_masha_steps : ℕ) (masha_yasha_total_steps : ℕ) (dad_step_rate : dad_steps_per_masha_steps = 3) (masha_step_rate : masha_steps_per_dad_steps = 5) (masha_step_rate_yasha : masha_steps_per_yasha_steps = 3) (yasha_step_rate_masha : yasha_steps_per_masha_steps = 5) (total_steps : masha_yasha_total_steps = 400) : 
∃ dad_steps : ℕ, dad_steps = 90 :=
by 
  sorry

end dad_steps_l2340_234018


namespace cups_per_girl_l2340_234093

noncomputable def numStudents := 30
noncomputable def numBoys := 10
noncomputable def numCupsByBoys := numBoys * 5
noncomputable def totalCups := 90
noncomputable def numGirls := 2 * numBoys
noncomputable def numCupsByGirls := totalCups - numCupsByBoys

theorem cups_per_girl : (numCupsByGirls / numGirls) = 2 := by
  sorry

end cups_per_girl_l2340_234093


namespace quadratic_roots_algebraic_expression_value_l2340_234049

-- Part 1: Proof statement for the roots of the quadratic equation
theorem quadratic_roots : (∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 7 ∧ x₂ = 2 - Real.sqrt 7 ∧ (∀ x : ℝ, x^2 - 4 * x - 3 = 0 → x = x₁ ∨ x = x₂)) :=
by
  sorry

-- Part 2: Proof statement for the algebraic expression value
theorem algebraic_expression_value (a : ℝ) (h : a^2 = 3 * a + 10) :
  (a + 4) * (a - 4) - 3 * (a - 1) = -3 :=
by
  sorry

end quadratic_roots_algebraic_expression_value_l2340_234049


namespace ellipse_semi_minor_axis_l2340_234001

theorem ellipse_semi_minor_axis (b : ℝ) 
    (h1 : 0 < b) 
    (h2 : b < 5)
    (h_ellipse : ∀ x y : ℝ, x^2 / 25 + y^2 / b^2 = 1) 
    (h_eccentricity : 4 / 5 = 4 / 5) : b = 3 := 
sorry

end ellipse_semi_minor_axis_l2340_234001


namespace ratio_of_waist_to_hem_l2340_234055

theorem ratio_of_waist_to_hem
  (cuffs_length : ℕ)
  (hem_length : ℕ)
  (ruffles_length : ℕ)
  (num_ruffles : ℕ)
  (lace_cost_per_meter : ℕ)
  (total_spent : ℕ)
  (waist_length : ℕ) :
  cuffs_length = 50 →
  hem_length = 300 →
  ruffles_length = 20 →
  num_ruffles = 5 →
  lace_cost_per_meter = 6 →
  total_spent = 36 →
  waist_length = (total_spent / lace_cost_per_meter * 100) -
                (2 * cuffs_length + hem_length + num_ruffles * ruffles_length) →
  waist_length / hem_length = 1 / 3 :=
by
  sorry

end ratio_of_waist_to_hem_l2340_234055


namespace average_speed_difference_l2340_234013

noncomputable def v_R : Float := 56.44102863722254
noncomputable def distance : Float := 750
noncomputable def t_R : Float := distance / v_R
noncomputable def t_P : Float := t_R - 2
noncomputable def v_P : Float := distance / t_P

theorem average_speed_difference : v_P - v_R = 10 := by
  sorry

end average_speed_difference_l2340_234013


namespace population_total_l2340_234094

variable (x y : ℕ)

theorem population_total (h1 : 20 * y = 12 * y * (x + y)) : x + y = 240 :=
  by
  -- Proceed with solving the provided conditions.
  sorry

end population_total_l2340_234094


namespace sequence_diff_ge_abs_m_l2340_234086

-- Define the conditions and theorem in Lean

theorem sequence_diff_ge_abs_m
    (m : ℤ) (h_m : |m| ≥ 2)
    (a : ℕ → ℤ)
    (h_seq_not_zero : ¬ (a 1 = 0 ∧ a 2 = 0))
    (h_rec : ∀ n : ℕ, n ≥ 1 → a (n + 2) = a (n + 1) - m * a n)
    (r s : ℕ) (h_r : r > s) (h_s : s ≥ 2)
    (h_equal : a r = a 1 ∧ a s = a 1) :
    r - s ≥ |m| :=
by
  sorry

end sequence_diff_ge_abs_m_l2340_234086


namespace inequality_proof_l2340_234016

theorem inequality_proof (a b m n p : ℝ) (h1 : a > b) (h2 : m > n) (h3 : p > 0) : n - a * p < m - b * p :=
sorry

end inequality_proof_l2340_234016


namespace greatest_num_consecutive_integers_l2340_234078

theorem greatest_num_consecutive_integers (N a : ℤ) (h : (N * (2*a + N - 1) = 210)) :
  ∃ N, N = 210 :=
sorry

end greatest_num_consecutive_integers_l2340_234078


namespace next_consecutive_time_l2340_234019

theorem next_consecutive_time (current_hour : ℕ) (current_minute : ℕ) 
  (valid_minutes : 0 ≤ current_minute ∧ current_minute < 60) 
  (valid_hours : 0 ≤ current_hour ∧ current_hour < 24) : 
  current_hour = 4 ∧ current_minute = 56 →
  ∃ next_hour next_minute : ℕ, 
    (0 ≤ next_minute ∧ next_minute < 60) ∧ 
    (0 ≤ next_hour ∧ next_hour < 24) ∧
    (next_hour, next_minute) = (12, 34) ∧ 
    (next_hour * 60 + next_minute) - (current_hour * 60 + current_minute) = 458 := 
by sorry

end next_consecutive_time_l2340_234019


namespace problem_conditions_l2340_234076

noncomputable def f (a b c x : ℝ) := 3 * a * x^2 + 2 * b * x + c

theorem problem_conditions (a b c : ℝ) (h0 : a + b + c = 0)
  (h1 : f a b c 0 > 0) (h2 : f a b c 1 > 0) :
    (a > 0 ∧ -2 < b / a ∧ b / a < -1) ∧
    (∃ z1 z2 : ℝ, 0 < z1 ∧ z1 < 1 ∧ 0 < z2 ∧ z2 < 1 ∧ z1 ≠ z2 ∧ f a b c z1 = 0 ∧ f a b c z2 = 0) :=
by
  sorry

end problem_conditions_l2340_234076


namespace gcd_2952_1386_l2340_234081

theorem gcd_2952_1386 : Nat.gcd 2952 1386 = 18 := by
  sorry

end gcd_2952_1386_l2340_234081


namespace sum_of_fractions_equals_three_l2340_234070

-- Definitions according to the conditions
def proper_fraction (a b : ℕ) := 1 ≤ a ∧ a < b
def improper_fraction (a b : ℕ) := a ≥ b
def mixed_number (a b c : ℕ) := a + b / c

-- Constants according to the given problem
def n := 8
def d := 9
def improper_n := 9

-- Values for elements in the conditions
def largest_proper_fraction := n / d
def smallest_improper_fraction := improper_n / d
def smallest_mixed_number := 1 + 1 / d

-- Theorem statement with the correct answer
theorem sum_of_fractions_equals_three :
  largest_proper_fraction + smallest_improper_fraction + smallest_mixed_number = 3 :=
sorry

end sum_of_fractions_equals_three_l2340_234070


namespace range_of_a_l2340_234085

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x : ℝ, |x - a| + |x + 1| ≤ 2) ↔ (a < -3 ∨ a > 1) :=
    sorry

end range_of_a_l2340_234085


namespace average_A_B_l2340_234039

variables (A B C : ℝ)

def conditions (A B C : ℝ) : Prop :=
  (A + B + C) / 3 = 45 ∧
  (B + C) / 2 = 43 ∧
  B = 31

theorem average_A_B (A B C : ℝ) (h : conditions A B C) : (A + B) / 2 = 40 :=
by
  sorry

end average_A_B_l2340_234039


namespace longest_side_of_rectangle_l2340_234031

theorem longest_side_of_rectangle 
    (l w : ℝ) 
    (h1 : 2 * l + 2 * w = 240) 
    (h2 : l * w = 2400) : 
    max l w = 80 :=
by sorry

end longest_side_of_rectangle_l2340_234031


namespace rank_from_right_l2340_234084

theorem rank_from_right (n total rank_left : ℕ) (h1 : rank_left = 5) (h2 : total = 21) : n = total - (rank_left - 1) :=
by {
  sorry
}

end rank_from_right_l2340_234084


namespace find_m_l2340_234029

noncomputable def f (x : ℝ) : ℝ := 1 / x
noncomputable def f' (x : ℝ) : ℝ := -1 / (x^2)
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := m * x

theorem find_m (m : ℝ) :
  g 2 m = 1 / (f' 2) →
  m = -2 :=
by
  sorry

end find_m_l2340_234029


namespace store_discount_difference_l2340_234069

theorem store_discount_difference 
  (p : ℝ) -- original price
  (p1 : ℝ := p * 0.60) -- price after initial discount
  (p2 : ℝ := p1 * 0.90) -- price after additional discount
  (claimed_discount : ℝ := 0.55) -- store's claimed discount
  (true_discount : ℝ := (p - p2) / p) -- calculated true discount
  (difference : ℝ := claimed_discount - true_discount)
  : difference = 0.09 :=
sorry

end store_discount_difference_l2340_234069


namespace solution_set_of_inequality_l2340_234063

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 + 2 * x + 3 > 0} = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end solution_set_of_inequality_l2340_234063


namespace length_more_than_breadth_l2340_234053

theorem length_more_than_breadth (length cost_per_metre total_cost : ℝ) (breadth : ℝ) :
  length = 60 → cost_per_metre = 26.50 → total_cost = 5300 → 
  (total_cost = (2 * length + 2 * breadth) * cost_per_metre) → length - breadth = 20 :=
by
  intros hlength hcost_per_metre htotal_cost hperimeter_cost
  rw [hlength, hcost_per_metre] at hperimeter_cost
  sorry

end length_more_than_breadth_l2340_234053


namespace find_X_in_rectangle_diagram_l2340_234043

theorem find_X_in_rectangle_diagram :
  ∀ (X : ℝ),
  (1 + 1 + 1 + 2 + X = 1 + 2 + 1 + 6) → X = 5 :=
by
  intros X h
  sorry

end find_X_in_rectangle_diagram_l2340_234043


namespace eggs_in_nests_l2340_234096

theorem eggs_in_nests (x : ℕ) (h1 : 2 * x + 3 + 4 = 17) : x = 5 :=
by
  /- This is where the proof would go, but the problem only requires the statement -/
  sorry

end eggs_in_nests_l2340_234096


namespace lowest_score_l2340_234058

-- Define the conditions
def test_scores (s1 s2 s3 : ℕ) := s1 = 86 ∧ s2 = 112 ∧ s3 = 91
def max_score := 120
def target_average := 95
def num_tests := 5
def total_points_needed := target_average * num_tests

-- Define the proof statement
theorem lowest_score 
  (s1 s2 s3 : ℕ)
  (condition1 : test_scores s1 s2 s3)
  (max_pts : ℕ := max_score) 
  (target_avg : ℕ := target_average) 
  (num_tests : ℕ := num_tests)
  (total_needed : ℕ := total_points_needed) :
  ∃ s4 s5 : ℕ, s4 ≤ max_pts ∧ s5 ≤ max_pts ∧ s4 + s5 + s1 + s2 + s3 = total_needed ∧ (s4 = 66 ∨ s5 = 66) :=
by
  sorry

end lowest_score_l2340_234058


namespace twice_abs_difference_of_squares_is_4000_l2340_234077

theorem twice_abs_difference_of_squares_is_4000 :
  2 * |(105:ℤ)^2 - (95:ℤ)^2| = 4000 :=
by sorry

end twice_abs_difference_of_squares_is_4000_l2340_234077


namespace truck_speed_kmph_l2340_234030

theorem truck_speed_kmph (d : ℕ) (t : ℕ) (km_m : ℕ) (hr_s : ℕ) 
  (h1 : d = 600) (h2 : t = 20) (h3 : km_m = 1000) (h4 : hr_s = 3600) : 
  (d / t) * (hr_s / km_m) = 108 := by
  sorry

end truck_speed_kmph_l2340_234030


namespace probability_all_same_color_l2340_234005

open scoped Classical

noncomputable def num_black : ℕ := 5
noncomputable def num_red : ℕ := 4
noncomputable def num_green : ℕ := 6
noncomputable def num_blue : ℕ := 3
noncomputable def num_yellow : ℕ := 2

noncomputable def total_marbles : ℕ :=
  num_black + num_red + num_green + num_blue + num_yellow

noncomputable def prob_all_same_color : ℚ :=
  let p_black := if num_black >= 4 then 
      (num_black / total_marbles) * ((num_black - 1) / (total_marbles - 1)) *
      ((num_black - 2) / (total_marbles - 2)) * ((num_black - 3) / (total_marbles - 3)) else 0
  let p_green := if num_green >= 4 then 
      (num_green / total_marbles) * ((num_green - 1) / (total_marbles - 1)) *
      ((num_green - 2) / (total_marbles - 2)) * ((num_green - 3) / (total_marbles - 3)) else 0
  p_black + p_green

theorem probability_all_same_color :
  prob_all_same_color = 0.004128 :=
sorry

end probability_all_same_color_l2340_234005


namespace minimum_reciprocal_sum_l2340_234065

theorem minimum_reciprocal_sum (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  4 ≤ (1 / a) + (1 / b) :=
sorry

end minimum_reciprocal_sum_l2340_234065


namespace time_to_finish_furniture_l2340_234036

-- Define the problem's conditions
def chairs : ℕ := 7
def tables : ℕ := 3
def minutes_per_piece : ℕ := 4

-- Define total furniture
def total_furniture : ℕ := chairs + tables

-- Define the function to calculate total time
def total_time (pieces : ℕ) (time_per_piece: ℕ) : ℕ :=
  pieces * time_per_piece

-- Theorem statement to be proven
theorem time_to_finish_furniture : total_time total_furniture minutes_per_piece = 40 := 
by
  -- Provide a placeholder for the proof
  sorry

end time_to_finish_furniture_l2340_234036


namespace unique_solution_of_pair_of_equations_l2340_234079

-- Definitions and conditions
def pair_of_equations (x k : ℝ) : Prop :=
  (x^2 + 1 = 4 * x + k)

-- Theorem to prove
theorem unique_solution_of_pair_of_equations :
  ∃ k : ℝ, (∀ x : ℝ, pair_of_equations x k -> x = 2) ∧ k = 0 :=
by
  -- Proof omitted
  sorry

end unique_solution_of_pair_of_equations_l2340_234079


namespace number_of_students_in_class_l2340_234051

theorem number_of_students_in_class (S : ℕ) 
  (h1 : ∀ n : ℕ, 4 * n ≠ 0 → S % 4 = 0) -- S is divisible by 4
  (h2 : ∀ G : ℕ, 3 * G ≠ 0 → (S * 3) % 4 = G) -- Number of students who went to the playground (3/4 * S) is integer
  (h3 : ∀ B : ℕ, G - B ≠ 0 → (G * 2) / 3 = 10) -- Number of girls on the playground
  : S = 20 := sorry

end number_of_students_in_class_l2340_234051


namespace half_lake_covered_day_l2340_234095

theorem half_lake_covered_day
  (N : ℕ) -- the total number of flowers needed to cover the entire lake
  (flowers_on_day : ℕ → ℕ) -- a function that gives the number of flowers on a specific day
  (h1 : flowers_on_day 20 = N) -- on the 20th day, the number of flowers is N
  (h2 : ∀ d, flowers_on_day (d + 1) = 2 * flowers_on_day d) -- the number of flowers doubles each day
  : flowers_on_day 19 = N / 2 :=
by
  sorry

end half_lake_covered_day_l2340_234095


namespace minimum_n_required_l2340_234027

def A_0 : (ℝ × ℝ) := (0, 0)

def is_on_x_axis (A : ℝ × ℝ) : Prop := A.snd = 0
def is_on_y_equals_x_squared (B : ℝ × ℝ) : Prop := B.snd = B.fst ^ 2
def is_equilateral_triangle (A B C : ℝ × ℝ) : Prop := sorry

def A_n (n : ℕ) : ℝ × ℝ := sorry
def B_n (n : ℕ) : ℝ × ℝ := sorry

def euclidean_distance (P Q : ℝ × ℝ) : ℝ :=
  ((Q.fst - P.fst) ^ 2 + (Q.snd - P.snd) ^ 2) ^ (1/2)

theorem minimum_n_required (n : ℕ) (h1 : ∀ n, is_on_x_axis (A_n n))
    (h2 : ∀ n, is_on_y_equals_x_squared (B_n n))
    (h3 : ∀ n, is_equilateral_triangle (A_n (n-1)) (B_n n) (A_n n)) :
    (euclidean_distance A_0 (A_n n) ≥ 50) → n ≥ 17 :=
by sorry

end minimum_n_required_l2340_234027


namespace mod_inverse_17_1200_l2340_234075

theorem mod_inverse_17_1200 : ∃ x : ℕ, x < 1200 ∧ 17 * x % 1200 = 1 := 
by
  use 353
  sorry

end mod_inverse_17_1200_l2340_234075


namespace lcm_of_2_4_5_6_l2340_234020

theorem lcm_of_2_4_5_6 : Nat.lcm (Nat.lcm (Nat.lcm 2 4) 5) 6 = 60 :=
by
  sorry

end lcm_of_2_4_5_6_l2340_234020


namespace windows_ways_l2340_234003

theorem windows_ways (n : ℕ) (h : n = 8) : (n * (n - 1)) = 56 :=
by
  sorry

end windows_ways_l2340_234003


namespace length_of_platform_l2340_234046

variables (t L T_p T_s : ℝ)
def train_length := 200  -- length of the train in meters
def platform_cross_time := 50  -- time in seconds to cross the platform
def pole_cross_time := 42  -- time in seconds to cross the signal pole

theorem length_of_platform :
  T_p = platform_cross_time ->
  T_s = pole_cross_time ->
  t = train_length ->
  (L = 38) :=
by
  intros hp hsp ht
  sorry  -- proof goes here

end length_of_platform_l2340_234046


namespace correct_statements_count_l2340_234007

-- Definitions
def proper_fraction (x : ℚ) : Prop := (0 < x) ∧ (x < 1)
def improper_fraction (x : ℚ) : Prop := (x ≥ 1)

-- Statements as conditions
def statement1 (a b : ℚ) : Prop := proper_fraction a ∧ proper_fraction b → proper_fraction (a + b)
def statement2 (a b : ℚ) : Prop := proper_fraction a ∧ proper_fraction b → proper_fraction (a * b)
def statement3 (a b : ℚ) : Prop := proper_fraction a ∧ improper_fraction b → improper_fraction (a + b)
def statement4 (a b : ℚ) : Prop := proper_fraction a ∧ improper_fraction b → improper_fraction (a * b)

-- The main theorem stating the correct answer
theorem correct_statements_count : 
  (¬ (∀ a b, statement1 a b)) ∧ 
  (∀ a b, statement2 a b) ∧ 
  (∀ a b, statement3 a b) ∧ 
  (¬ (∀ a b, statement4 a b)) → 
  (2 = 2)
:= by sorry

end correct_statements_count_l2340_234007


namespace largest_divisor_of_expression_l2340_234022

theorem largest_divisor_of_expression
  (x : ℤ) (h_odd : x % 2 = 1) : 
  ∃ k : ℤ, k = 40 ∧ 40 ∣ (12 * x + 2) * (8 * x + 14) * (10 * x + 10) :=
by
  sorry

end largest_divisor_of_expression_l2340_234022


namespace original_amount_l2340_234041

variable (M : ℕ)

def initialAmountAfterFirstLoss := M - M / 3
def amountAfterFirstWin := initialAmountAfterFirstLoss M + 10
def amountAfterSecondLoss := amountAfterFirstWin M - (amountAfterFirstWin M) / 3
def amountAfterSecondWin := amountAfterSecondLoss M + 20
def finalAmount := amountAfterSecondWin M - (amountAfterSecondWin M) / 4

theorem original_amount : finalAmount M = M → M = 30 :=
by
  sorry

end original_amount_l2340_234041


namespace unique_rhombus_property_not_in_rectangle_l2340_234097

-- Definitions of properties for a rhombus and a rectangle
def is_rhombus (sides_equal : Prop) (opposite_sides_parallel : Prop) (opposite_angles_equal : Prop)
  (diagonals_perpendicular_and_bisect : Prop) : Prop :=
  sides_equal ∧ opposite_sides_parallel ∧ opposite_angles_equal ∧ diagonals_perpendicular_and_bisect

def is_rectangle (opposite_sides_equal_and_parallel : Prop) (all_angles_right : Prop)
  (diagonals_equal_and_bisect : Prop) : Prop :=
  opposite_sides_equal_and_parallel ∧ all_angles_right ∧ diagonals_equal_and_bisect

-- Proof objective: Prove that the unique property of a rhombus is the perpendicular and bisecting nature of its diagonals
theorem unique_rhombus_property_not_in_rectangle :
  ∀ (sides_equal opposite_sides_parallel opposite_angles_equal
      diagonals_perpendicular_and_bisect opposite_sides_equal_and_parallel
      all_angles_right diagonals_equal_and_bisect : Prop),
  is_rhombus sides_equal opposite_sides_parallel opposite_angles_equal diagonals_perpendicular_and_bisect →
  is_rectangle opposite_sides_equal_and_parallel all_angles_right diagonals_equal_and_bisect →
  diagonals_perpendicular_and_bisect ∧ ¬diagonals_equal_and_bisect :=
by
  sorry

end unique_rhombus_property_not_in_rectangle_l2340_234097


namespace f_is_32x5_l2340_234021

-- Define the function f
def f (x : ℝ) : ℝ := (2 * x + 1) ^ 5 - 5 * (2 * x + 1) ^ 4 + 10 * (2 * x + 1) ^ 3 - 10 * (2 * x + 1) ^ 2 + 5 * (2 * x + 1) - 1

-- State the theorem to be proved
theorem f_is_32x5 (x : ℝ) : f x = 32 * x ^ 5 := 
by
  sorry

end f_is_32x5_l2340_234021


namespace sum_of_possible_values_l2340_234068

theorem sum_of_possible_values (N : ℝ) (h : N * (N - 4) = -21) : (∃ x y : ℝ, x * (x - 4) = -21 ∧ y * (y - 4) = -21 ∧ x + y = 4) :=
sorry

end sum_of_possible_values_l2340_234068


namespace total_travel_time_is_correct_l2340_234056

-- Conditions as definitions
def total_distance : ℕ := 200
def initial_fraction : ℚ := 1 / 4
def initial_time : ℚ := 1 -- in hours
def lunch_time : ℚ := 1 -- in hours
def remaining_fraction : ℚ := 1 / 2
def pit_stop_time : ℚ := 0.5 -- in hours
def speed_increase : ℚ := 10

-- Derived/Calculated values needed for the problem statement
def initial_distance : ℚ := initial_fraction * total_distance
def initial_speed : ℚ := initial_distance / initial_time
def remaining_distance : ℚ := total_distance - initial_distance
def half_remaining_distance : ℚ := remaining_fraction * remaining_distance
def second_drive_time : ℚ := half_remaining_distance / initial_speed
def last_distance : ℚ := remaining_distance - half_remaining_distance
def last_speed : ℚ := initial_speed + speed_increase
def last_drive_time : ℚ := last_distance / last_speed

-- Total time calculation
def total_time : ℚ :=
  initial_time + lunch_time + second_drive_time + pit_stop_time + last_drive_time

-- Lean theorem statement
theorem total_travel_time_is_correct : total_time = 5.25 :=
  sorry

end total_travel_time_is_correct_l2340_234056


namespace geometric_sequence_iff_arithmetic_sequence_l2340_234004

/-
  Suppose that {a_n} is an infinite geometric sequence with common ratio q, where q^2 ≠ 1.
  Also suppose that {b_n} is a sequence of positive natural numbers (ℕ).
  Prove that {a_{b_n}} forms a geometric sequence if and only if {b_n} forms an arithmetic sequence.
-/

theorem geometric_sequence_iff_arithmetic_sequence
  (a : ℕ → ℕ) (b : ℕ → ℕ) (q : ℝ)
  (h_geom_a : ∃ a1, ∀ n, a n = a1 * q ^ (n - 1))
  (h_q_squared_ne_one : q^2 ≠ 1)
  (h_bn_positive : ∀ n, 0 < b n) :
  (∃ a1, ∃ q', ∀ n, a (b n) = a1 * q' ^ n) ↔ (∃ d, ∀ n, b (n + 1) - b n = d) := 
sorry

end geometric_sequence_iff_arithmetic_sequence_l2340_234004


namespace problem_statement_l2340_234073

def op (x y : ℝ) : ℝ := (x + 3) * (y - 1)

theorem problem_statement (a : ℝ) : (∀ x : ℝ, op (x - a) (x + a) > -16) ↔ -2 < a ∧ a < 6 :=
by
  sorry

end problem_statement_l2340_234073


namespace find_number_with_21_multiples_of_4_l2340_234002

theorem find_number_with_21_multiples_of_4 (n : ℕ) (h₁ : ∀ k : ℕ, n + k * 4 ≤ 92 → k < 21) : n = 80 :=
sorry

end find_number_with_21_multiples_of_4_l2340_234002


namespace graph_not_pass_through_second_quadrant_l2340_234025

theorem graph_not_pass_through_second_quadrant :
  ¬ ∃ x y : ℝ, y = 2 * x - 3 ∧ x < 0 ∧ y > 0 :=
by sorry

end graph_not_pass_through_second_quadrant_l2340_234025


namespace range_of_a_l2340_234037

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 + 2 * x + a ≤ 0) ↔ a ≤ 1 :=
by
  sorry

end range_of_a_l2340_234037


namespace find_distance_between_sides_l2340_234067

-- Define the given conditions
def length_side1 : ℝ := 20
def length_side2 : ℝ := 18
def area_trapezium : ℝ := 247

-- Define the distance h between parallel sides
def distance_between_sides (h : ℝ) : Prop :=
  area_trapezium = (1 / 2) * (length_side1 + length_side2) * h

-- Define the theorem we want to prove
theorem find_distance_between_sides : ∃ h : ℝ, distance_between_sides h ∧ h = 13 := by
  sorry

end find_distance_between_sides_l2340_234067


namespace unique_solutions_xy_l2340_234080

theorem unique_solutions_xy (x y : ℝ) : 
  x^3 + y^3 = 1 ∧ x^4 + y^4 = 1 ↔ (x = 0 ∧ y = 1) ∨ (x = 1 ∧ y = 0) :=
by sorry

end unique_solutions_xy_l2340_234080


namespace area_ratio_l2340_234048

-- Definitions corresponding to the conditions
variable {A B C P Q R : Type}
variable (t : ℝ)
variable (h_pos : 0 < t) (h_lt_one : t < 1)

-- Define the areas in terms of provided conditions
noncomputable def area_AP : ℝ := sorry
noncomputable def area_BQ : ℝ := sorry
noncomputable def area_CR : ℝ := sorry
noncomputable def K : ℝ := area_AP * area_BQ * area_CR
noncomputable def L : ℝ := sorry -- Area of triangle ABC

-- The statement to be proved
theorem area_ratio (h_pos : 0 < t) (h_lt_one : t < 1) :
  (K / L) = (1 - t + t^2)^2 :=
sorry

end area_ratio_l2340_234048


namespace smallest_value_z_minus_x_l2340_234028

theorem smallest_value_z_minus_x 
  (x y z : ℕ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (hmul : x * y * z = 5040) 
  (hxy : x < y) 
  (hyz : y < z) : 
  z - x = 9 := 
  sorry

end smallest_value_z_minus_x_l2340_234028


namespace product_telescope_l2340_234099

theorem product_telescope : ((1 + (1 / 1)) * 
                             (1 + (1 / 2)) * 
                             (1 + (1 / 3)) * 
                             (1 + (1 / 4)) * 
                             (1 + (1 / 5)) * 
                             (1 + (1 / 6)) * 
                             (1 + (1 / 7)) * 
                             (1 + (1 / 8)) * 
                             (1 + (1 / 9)) * 
                             (1 + (1 / 10))) = 11 := 
by
  sorry

end product_telescope_l2340_234099


namespace problem1_problem2_l2340_234050

-- Problem 1
theorem problem1 : (-2) ^ 2 + (Real.sqrt 2 - 1) ^ 0 - 1 = 4 := by
  sorry

-- Problem 2
theorem problem2 (a : ℝ) (A : ℝ) (B : ℝ) (h1 : A = a - 1) (h2 : B = -a + 3) (h3 : A > B) : a > 2 := by
  sorry

end problem1_problem2_l2340_234050


namespace sets_are_equal_l2340_234088

def int : Type := ℤ  -- Redefine integer as ℤ for clarity

def SetA : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}
def SetB : Set ℤ := {x | ∃ k : ℤ, x = 2 * k - 1}

theorem sets_are_equal : SetA = SetB := by
  -- implement the proof here
  sorry

end sets_are_equal_l2340_234088


namespace flour_needed_l2340_234026

-- Definitions
def cups_per_loaf := 2.5
def loaves := 2

-- Statement we want to prove
theorem flour_needed {cups_per_loaf loaves : ℝ} (h : cups_per_loaf = 2.5) (l : loaves = 2) : 
  cups_per_loaf * loaves = 5 :=
sorry

end flour_needed_l2340_234026


namespace equivalence_statement_l2340_234006

open Complex

noncomputable def distinct_complex (a b c d : ℂ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem equivalence_statement (a b c d : ℂ) (h : distinct_complex a b c d) :
  (∀ (z : ℂ), (abs (z - a) + abs (z - b) ≥ abs (z - c) + abs (z - d)))
  ↔ (∃ (t : ℝ), 0 < t ∧ t < 1 ∧ c = t * a + (1 - t) * b ∧ d = (1 - t) * a + t * b) :=
sorry

end equivalence_statement_l2340_234006


namespace fraction_of_selected_films_in_color_l2340_234011

variables (x y : ℕ)

theorem fraction_of_selected_films_in_color (B C : ℕ) (e : ℚ)
  (h1 : B = 20 * x)
  (h2 : C = 6 * y)
  (h3 : e = (6 * y : ℚ) / (((y / 5 : ℚ) + 6 * y))) :
  e = 30 / 31 :=
by {
  sorry
}

end fraction_of_selected_films_in_color_l2340_234011


namespace determine_m_for_value_range_l2340_234042

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2 * x + m

theorem determine_m_for_value_range :
  ∀ m : ℝ, (∀ x : ℝ, f m x ≥ 0) ↔ m = 1 :=
by
  sorry

end determine_m_for_value_range_l2340_234042
