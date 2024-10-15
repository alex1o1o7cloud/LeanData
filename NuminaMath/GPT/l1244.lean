import Mathlib

namespace NUMINAMATH_GPT_legs_sum_of_right_triangle_with_hypotenuse_41_l1244_124408

noncomputable def right_triangle_legs_sum (x : ℕ) : ℕ := x + (x + 1)

theorem legs_sum_of_right_triangle_with_hypotenuse_41 :
  ∃ x : ℕ, (x * x + (x + 1) * (x + 1) = 41 * 41) ∧ right_triangle_legs_sum x = 57 := by
sorry

end NUMINAMATH_GPT_legs_sum_of_right_triangle_with_hypotenuse_41_l1244_124408


namespace NUMINAMATH_GPT_uniform_heights_l1244_124490

theorem uniform_heights (varA varB : ℝ) (hA : varA = 0.56) (hB : varB = 2.1) : varA < varB := by
  rw [hA, hB]
  exact (by norm_num)

end NUMINAMATH_GPT_uniform_heights_l1244_124490


namespace NUMINAMATH_GPT_ms_hatcher_students_l1244_124406

-- Define the number of third-graders
def third_graders : ℕ := 20

-- Condition: The number of fourth-graders is twice the number of third-graders
def fourth_graders : ℕ := 2 * third_graders

-- Condition: The number of fifth-graders is half the number of third-graders
def fifth_graders : ℕ := third_graders / 2

-- The total number of students Ms. Hatcher teaches in a day
def total_students : ℕ := third_graders + fourth_graders + fifth_graders

-- The statement to prove
theorem ms_hatcher_students : total_students = 70 := by
  sorry

end NUMINAMATH_GPT_ms_hatcher_students_l1244_124406


namespace NUMINAMATH_GPT_hexagon_diagonal_length_is_twice_side_l1244_124456

noncomputable def regular_hexagon_side_length : ℝ := 12

def diagonal_length_in_regular_hexagon (s : ℝ) : ℝ :=
2 * s

theorem hexagon_diagonal_length_is_twice_side :
  diagonal_length_in_regular_hexagon regular_hexagon_side_length = 2 * regular_hexagon_side_length :=
by 
  -- Simplify and check the computation according to the understanding of the properties of the hexagon
  sorry

end NUMINAMATH_GPT_hexagon_diagonal_length_is_twice_side_l1244_124456


namespace NUMINAMATH_GPT_third_side_length_is_six_l1244_124424

theorem third_side_length_is_six
  (a b : ℝ) (c : ℤ)
  (h1 : a = 6.31) 
  (h2 : b = 0.82) 
  (h3 : (a + b > c) ∧ ((b : ℝ) + (c : ℝ) > a) ∧ (c + a > b)) 
  (h4 : 5.49 < (c : ℝ)) 
  (h5 : (c : ℝ) < 7.13) : 
  c = 6 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_third_side_length_is_six_l1244_124424


namespace NUMINAMATH_GPT_horner_first_calculation_at_3_l1244_124413

def f (x : ℝ) : ℝ :=
  0.5 * x ^ 6 + 4 * x ^ 5 - x ^ 4 + 3 * x ^ 3 - 5 * x

def horner_first_step (x : ℝ) : ℝ :=
  0.5 * x + 4

theorem horner_first_calculation_at_3 :
  horner_first_step 3 = 5.5 := by
  sorry

end NUMINAMATH_GPT_horner_first_calculation_at_3_l1244_124413


namespace NUMINAMATH_GPT_perpendicular_lines_b_value_l1244_124447

theorem perpendicular_lines_b_value :
  ( ∀ x y : ℝ, 2 * x + 3 * y + 4 = 0)  →
  ( ∀ x y : ℝ, b * x + 3 * y - 1 = 0) →
  ( - (2 : ℝ) / (3 : ℝ) * - b / (3 : ℝ) = -1 ) →
  b = - (9 : ℝ) / (2 : ℝ) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_perpendicular_lines_b_value_l1244_124447


namespace NUMINAMATH_GPT_ratio_of_typing_speeds_l1244_124492

-- Defining Tim's and Tom's typing speeds
variables (T M : ℝ)

-- Conditions given in the problem
def condition1 : Prop := T + M = 15
def condition2 : Prop := T + 1.6 * M = 18

-- Conclusion to be proved: the ratio of M to T is 1:2
theorem ratio_of_typing_speeds (h1 : condition1 T M) (h2 : condition2 T M) :
  M / T = 1 / 2 :=
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_ratio_of_typing_speeds_l1244_124492


namespace NUMINAMATH_GPT_proof_problem_l1244_124481

theorem proof_problem (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a * b ∣ c * (c ^ 2 - c + 1))
  (h5 : (c ^ 2 + 1) ∣ (a + b)) :
  (a = c ∧ b = c ^ 2 - c + 1) ∨ (a = c ^ 2 - c + 1 ∧ b = c) :=
sorry

end NUMINAMATH_GPT_proof_problem_l1244_124481


namespace NUMINAMATH_GPT_incenter_circumcenter_identity_l1244_124488

noncomputable def triangle : Type := sorry
noncomputable def incenter (t : triangle) : Type := sorry
noncomputable def circumcenter (t : triangle) : Type := sorry
noncomputable def inradius (t : triangle) : ℝ := sorry
noncomputable def circumradius (t : triangle) : ℝ := sorry
noncomputable def distance (A B : Type) : ℝ := sorry

theorem incenter_circumcenter_identity (t : triangle) (I O : Type)
  (hI : I = incenter t) (hO : O = circumcenter t)
  (r : ℝ) (h_r : r = inradius t)
  (R : ℝ) (h_R : R = circumradius t) :
  distance I O ^ 2 = R ^ 2 - 2 * R * r :=
sorry

end NUMINAMATH_GPT_incenter_circumcenter_identity_l1244_124488


namespace NUMINAMATH_GPT_calculate_expression_l1244_124452

theorem calculate_expression : 
  (0.25 ^ 16) * ((-4) ^ 17) = -4 := 
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1244_124452


namespace NUMINAMATH_GPT_digit_difference_l1244_124484

theorem digit_difference (X Y : ℕ) (h1 : 10 * X + Y - (10 * Y + X) = 36) : X - Y = 4 := by
  sorry

end NUMINAMATH_GPT_digit_difference_l1244_124484


namespace NUMINAMATH_GPT_first_discount_percentage_l1244_124420

theorem first_discount_percentage :
  ∃ x : ℝ, (9649.12 * (1 - x / 100) * 0.9 * 0.95 = 6600) ∧ (19.64 ≤ x ∧ x ≤ 19.66) :=
sorry

end NUMINAMATH_GPT_first_discount_percentage_l1244_124420


namespace NUMINAMATH_GPT_age_ratio_l1244_124465

theorem age_ratio 
  (a b c : ℕ)
  (h1 : a = b + 2)
  (h2 : a + b + c = 32)
  (h3 : b = 12) :
  b = 2 * c :=
by
  sorry

end NUMINAMATH_GPT_age_ratio_l1244_124465


namespace NUMINAMATH_GPT_line_problems_l1244_124445

noncomputable def l1 : (ℝ → ℝ) := λ x => x - 1
noncomputable def l2 (k : ℝ) : (ℝ → ℝ) := λ x => -(k + 1) / k * x - 1

theorem line_problems (k : ℝ) :
  ∃ k, k = 0 → (l2 k 1) = 90 →      -- A
  (∀ k, (l1 1 = l2 k 1 → True)) →   -- B
  (∀ k, (l1 1 ≠ l2 k 1 → True)) →   -- C (negated conclusion from False in C)
  (∀ k, (l1 1 * l2 k 1 ≠ -1))       -- D
:=
sorry

end NUMINAMATH_GPT_line_problems_l1244_124445


namespace NUMINAMATH_GPT_find_c_l1244_124473

theorem find_c (a b c : ℚ) (h_eqn : ∀ y, a * y^2 + b * y + c = y^2 / 12 + 5 * y / 6 + 145 / 12)
  (h_vertex : ∀ x, x = a * (-5)^2 + b * (-5) + c)
  (h_pass : a * (-1 + 5)^2 + 1 = 4) :
  c = 145 / 12 := by
sorry

end NUMINAMATH_GPT_find_c_l1244_124473


namespace NUMINAMATH_GPT_cookie_distribution_l1244_124462

theorem cookie_distribution:
  ∀ (initial_boxes brother_cookies sister_cookies leftover after_siblings leftover_sonny : ℕ),
    initial_boxes = 45 →
    brother_cookies = 12 →
    sister_cookies = 9 →
    after_siblings = initial_boxes - brother_cookies - sister_cookies →
    leftover_sonny = 17 →
    leftover = after_siblings - leftover_sonny →
    leftover = 7 :=
by
  intros initial_boxes brother_cookies sister_cookies leftover after_siblings leftover_sonny
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_cookie_distribution_l1244_124462


namespace NUMINAMATH_GPT_milk_distribution_l1244_124414

theorem milk_distribution 
  (x y z : ℕ)
  (h_total : x + y + z = 780)
  (h_equiv : 3 * x / 4 = 4 * y / 5 ∧ 3 * x / 4 = 4 * z / 7) :
  x = 240 ∧ y = 225 ∧ z = 315 := 
sorry

end NUMINAMATH_GPT_milk_distribution_l1244_124414


namespace NUMINAMATH_GPT_units_digit_multiplication_l1244_124421

-- Define a function to find the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- Statement of the problem: Given the product 27 * 36, prove that the units digit is 2.
theorem units_digit_multiplication (a b : ℕ) (h1 : units_digit 27 = 7) (h2 : units_digit 36 = 6) :
  units_digit (27 * 36) = 2 :=
by
  have h3 : units_digit (7 * 6) = 2 := by sorry
  exact h3

end NUMINAMATH_GPT_units_digit_multiplication_l1244_124421


namespace NUMINAMATH_GPT_complement_M_l1244_124497

open Set

-- Define the universal set U as the set of all real numbers
def U := ℝ

-- Define the set M as {x | |x| > 2}
def M : Set ℝ := {x | |x| > 2}

-- State that the complement of M (in the universal set U) is [-2, 2]
theorem complement_M : Mᶜ = {x | -2 ≤ x ∧ x ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_complement_M_l1244_124497


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1244_124443

theorem geometric_sequence_sum :
  let a := (1/2 : ℚ)
  let r := (1/3 : ℚ)
  let n := 6
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 364 / 243 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1244_124443


namespace NUMINAMATH_GPT_water_added_to_mixture_is_11_l1244_124441

noncomputable def initial_mixture_volume : ℕ := 45
noncomputable def initial_milk_ratio : ℚ := 4
noncomputable def initial_water_ratio : ℚ := 1
noncomputable def final_milk_ratio : ℚ := 9
noncomputable def final_water_ratio : ℚ := 5

theorem water_added_to_mixture_is_11 :
  ∃ x : ℚ, (initial_milk_ratio * initial_mixture_volume / 
            (initial_water_ratio * initial_mixture_volume + x)) = (final_milk_ratio / final_water_ratio)
  ∧ x = 11 :=
by
  -- Proof here
  sorry

end NUMINAMATH_GPT_water_added_to_mixture_is_11_l1244_124441


namespace NUMINAMATH_GPT_mother_younger_than_father_l1244_124499

variable (total_age : ℕ) (father_age : ℕ) (brother_age : ℕ) (sister_age : ℕ) (kaydence_age : ℕ) (mother_age : ℕ)

noncomputable def family_data : Prop :=
  total_age = 200 ∧
  father_age = 60 ∧
  brother_age = father_age / 2 ∧
  sister_age = 40 ∧
  kaydence_age = 12 ∧
  mother_age = total_age - (father_age + brother_age + sister_age + kaydence_age)

theorem mother_younger_than_father :
  family_data total_age father_age brother_age sister_age kaydence_age mother_age →
  father_age - mother_age = 2 :=
sorry

end NUMINAMATH_GPT_mother_younger_than_father_l1244_124499


namespace NUMINAMATH_GPT_number_of_samples_from_retired_l1244_124425

def ratio_of_forms (retired current students : ℕ) : Prop :=
retired = 3 ∧ current = 7 ∧ students = 40

def total_sampled_forms := 300

theorem number_of_samples_from_retired :
  ∃ (xr : ℕ), ratio_of_forms 3 7 40 → xr = (300 / (3 + 7 + 40)) * 3 :=
sorry

end NUMINAMATH_GPT_number_of_samples_from_retired_l1244_124425


namespace NUMINAMATH_GPT_range_of_f_l1244_124409

-- Define the function f(x) = 4 sin^3(x) + sin^2(x) - 4 sin(x) + 8
noncomputable def f (x : ℝ) : ℝ :=
  4 * (Real.sin x) ^ 3 + (Real.sin x) ^ 2 - 4 * (Real.sin x) + 8

-- Statement to prove the range of f(x)
theorem range_of_f :
  ∀ x : ℝ, 6 + 3 / 4 ≤ f x ∧ f x ≤ 9 + 25 / 27 :=
sorry

end NUMINAMATH_GPT_range_of_f_l1244_124409


namespace NUMINAMATH_GPT_find_px_l1244_124494

theorem find_px (p : ℕ → ℚ) (h1 : p 1 = 1) (h2 : p 2 = 1 / 4) (h3 : p 3 = 1 / 9) 
  (h4 : p 4 = 1 / 16) (h5 : p 5 = 1 / 25) : p 6 = 1 / 18 :=
sorry

end NUMINAMATH_GPT_find_px_l1244_124494


namespace NUMINAMATH_GPT_hall_area_l1244_124429

theorem hall_area {L W : ℝ} (h₁ : W = 0.5 * L) (h₂ : L - W = 20) : L * W = 800 := by
  sorry

end NUMINAMATH_GPT_hall_area_l1244_124429


namespace NUMINAMATH_GPT_impossible_even_sum_l1244_124467

theorem impossible_even_sum (n m : ℤ) (h : (n^2 + m^2) % 2 = 1) : (n + m) % 2 ≠ 0 :=
sorry

end NUMINAMATH_GPT_impossible_even_sum_l1244_124467


namespace NUMINAMATH_GPT_arithmetic_problem_l1244_124415

theorem arithmetic_problem :
  12.1212 + 17.0005 - 9.1103 = 20.0114 :=
sorry

end NUMINAMATH_GPT_arithmetic_problem_l1244_124415


namespace NUMINAMATH_GPT_marked_vertices_coincide_l1244_124474

theorem marked_vertices_coincide :
  ∀ (P Q : Fin 16 → Prop),
  (∃ A B C D E F G : Fin 16, P A ∧ P B ∧ P C ∧ P D ∧ P E ∧ P F ∧ P G) →
  (∃ A' B' C' D' E' F' G' : Fin 16, Q A' ∧ Q B' ∧ Q C' ∧ Q D' ∧ Q E' ∧ Q F' ∧ Q G') →
  ∃ (r : Fin 16), ∃ (A B C D : Fin 16), 
  (Q ((A + r) % 16) ∧ Q ((B + r) % 16) ∧ Q ((C + r) % 16) ∧ Q ((D + r) % 16)) :=
by
  sorry

end NUMINAMATH_GPT_marked_vertices_coincide_l1244_124474


namespace NUMINAMATH_GPT_find_multiplier_l1244_124463

theorem find_multiplier (x : ℤ) : 
  30 * x - 138 = 102 ↔ x = 8 := 
by
  sorry

end NUMINAMATH_GPT_find_multiplier_l1244_124463


namespace NUMINAMATH_GPT_find_a_conditions_l1244_124435

theorem find_a_conditions (a : ℝ) : 
    (∃ m : ℤ, a = m + 1/2) ∨ (∃ m : ℤ, a = m + 1/3) ∨ (∃ m : ℤ, a = m - 1/3) ↔ 
    (∃ n : ℤ, a = n + 1/2 ∨ a = n + 1/3 ∨ a = n - 1/3) :=
by
  sorry

end NUMINAMATH_GPT_find_a_conditions_l1244_124435


namespace NUMINAMATH_GPT_no_common_real_solution_l1244_124485

theorem no_common_real_solution :
  ¬ ∃ (x y : ℝ), (x^2 - 6 * x + y + 9 = 0) ∧ (x^2 + 4 * y + 5 = 0) :=
by
  sorry

end NUMINAMATH_GPT_no_common_real_solution_l1244_124485


namespace NUMINAMATH_GPT_sum_of_first_15_even_positive_integers_l1244_124483

theorem sum_of_first_15_even_positive_integers :
  let a := 2
  let l := 30
  let n := 15
  let S := (a + l) / 2 * n
  S = 240 := by
  sorry

end NUMINAMATH_GPT_sum_of_first_15_even_positive_integers_l1244_124483


namespace NUMINAMATH_GPT_circle_area_from_circumference_l1244_124457

theorem circle_area_from_circumference
  (c : ℝ)    -- the circumference
  (hc : c = 36)    -- condition: circumference is 36 cm
  : 
  ∃ A : ℝ,   -- there exists an area A
    A = 324 / π :=   -- conclusion: area is 324/π
by
  sorry   -- proof goes here

end NUMINAMATH_GPT_circle_area_from_circumference_l1244_124457


namespace NUMINAMATH_GPT_total_profit_Q2_is_correct_l1244_124493

-- Conditions as definitions
def profit_Q1_A := 1500
def profit_Q1_B := 2000
def profit_Q1_C := 1000

def profit_Q2_A := 2500
def profit_Q2_B := 3000
def profit_Q2_C := 1500

def profit_Q3_A := 3000
def profit_Q3_B := 2500
def profit_Q3_C := 3500

def profit_Q4_A := 2000
def profit_Q4_B := 3000
def profit_Q4_C := 2000

-- The total profit calculation for the second quarter
def total_profit_Q2 := profit_Q2_A + profit_Q2_B + profit_Q2_C

-- Proof statement
theorem total_profit_Q2_is_correct : total_profit_Q2 = 7000 := by
  sorry

end NUMINAMATH_GPT_total_profit_Q2_is_correct_l1244_124493


namespace NUMINAMATH_GPT_bead_necklaces_sold_l1244_124411

def cost_per_necklace : ℕ := 7
def total_earnings : ℕ := 70
def gemstone_necklaces_sold : ℕ := 7

theorem bead_necklaces_sold (B : ℕ) 
  (h1 : total_earnings = cost_per_necklace * (B + gemstone_necklaces_sold))  :
  B = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_bead_necklaces_sold_l1244_124411


namespace NUMINAMATH_GPT_mao_li_total_cards_l1244_124442

theorem mao_li_total_cards : (23 : ℕ) + (20 : ℕ) = 43 := by
  sorry

end NUMINAMATH_GPT_mao_li_total_cards_l1244_124442


namespace NUMINAMATH_GPT_eval_expr_l1244_124482

theorem eval_expr : (2/5) + (3/8) - (1/10) = 27/40 :=
by
  sorry

end NUMINAMATH_GPT_eval_expr_l1244_124482


namespace NUMINAMATH_GPT_lines_coinicide_l1244_124496

open Real

theorem lines_coinicide (k m n : ℝ) :
  (∃ (x y : ℝ), y = k * x + m ∧ y = m * x + n ∧ y = n * x + k) →
  k = m ∧ m = n :=
by
  sorry

end NUMINAMATH_GPT_lines_coinicide_l1244_124496


namespace NUMINAMATH_GPT_evaluate_expression_l1244_124460

theorem evaluate_expression : (5 * 3 ^ 4 + 6 * 4 ^ 3 = 789) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1244_124460


namespace NUMINAMATH_GPT_min_value_sq_sum_l1244_124446

theorem min_value_sq_sum (x1 x2 : ℝ) (h : x1 * x2 = 2013) : (x1 + x2)^2 ≥ 8052 :=
by
  sorry

end NUMINAMATH_GPT_min_value_sq_sum_l1244_124446


namespace NUMINAMATH_GPT_system_of_equations_solutions_l1244_124422

theorem system_of_equations_solutions (x1 x2 x3 : ℝ) :
  (2 * x1^2 / (1 + x1^2) = x2) ∧ (2 * x2^2 / (1 + x2^2) = x3) ∧ (2 * x3^2 / (1 + x3^2) = x1)
  → (x1 = 0 ∧ x2 = 0 ∧ x3 = 0) ∨ (x1 = 1 ∧ x2 = 1 ∧ x3 = 1) :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_solutions_l1244_124422


namespace NUMINAMATH_GPT_mean_and_variance_l1244_124461

def scores_A : List ℝ := [8, 9, 14, 15, 15, 16, 21, 22]
def scores_B : List ℝ := [7, 8, 13, 15, 15, 17, 22, 23]

noncomputable def mean (l : List ℝ) : ℝ := (l.sum) / (l.length)
noncomputable def variance (l : List ℝ) : ℝ := mean (l.map (λ x => (x - (mean l)) ^ 2))

theorem mean_and_variance :
  (mean scores_A = mean scores_B) ∧ (variance scores_A < variance scores_B) :=
by
  sorry

end NUMINAMATH_GPT_mean_and_variance_l1244_124461


namespace NUMINAMATH_GPT_hair_cut_length_l1244_124466

theorem hair_cut_length (original_length after_haircut : ℕ) (h1 : original_length = 18) (h2 : after_haircut = 9) :
  original_length - after_haircut = 9 :=
by
  sorry

end NUMINAMATH_GPT_hair_cut_length_l1244_124466


namespace NUMINAMATH_GPT_identify_parrots_l1244_124436

-- Definitions of parrots
inductive Parrot
| gosha : Parrot
| kesha : Parrot
| roma : Parrot

open Parrot

-- Properties of each parrot
def always_honest (p : Parrot) : Prop :=
  p = gosha

def always_liar (p : Parrot) : Prop :=
  p = kesha

def sometimes_honest (p : Parrot) : Prop :=
  p = roma

-- Statements given by each parrot
def Gosha_statement : Prop :=
  always_liar kesha

def Kesha_statement : Prop :=
  sometimes_honest kesha

def Roma_statement : Prop :=
  always_honest kesha

-- Final statement to prove the identities
theorem identify_parrots (p : Parrot) :
  Gosha_statement ∧ Kesha_statement ∧ Roma_statement → (always_liar Parrot.kesha ∧ sometimes_honest Parrot.roma) :=
by
  intro h
  exact sorry

end NUMINAMATH_GPT_identify_parrots_l1244_124436


namespace NUMINAMATH_GPT_smallest_possible_N_l1244_124489

theorem smallest_possible_N (N : ℕ) (h : ∀ m : ℕ, m ≤ 60 → m % 3 = 0 → ∃ i : ℕ, i < 20 ∧ m = 3 * i + 1 ∧ N = 20) :
    N = 20 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_possible_N_l1244_124489


namespace NUMINAMATH_GPT_luke_total_points_l1244_124451

/-- Luke gained 327 points in each round of a trivia game. 
    He played 193 rounds of the game. 
    How many points did he score in total? -/
theorem luke_total_points : 327 * 193 = 63111 :=
by
  sorry

end NUMINAMATH_GPT_luke_total_points_l1244_124451


namespace NUMINAMATH_GPT_birthday_pizza_problem_l1244_124410

theorem birthday_pizza_problem (m : ℕ) (h1 : m > 11) (h2 : 55 % m = 0) : 10 + 55 / m = 13 := by
  sorry

end NUMINAMATH_GPT_birthday_pizza_problem_l1244_124410


namespace NUMINAMATH_GPT_range_of_m_l1244_124464

-- Conditions:
def is_opposite_sides_of_line (p1 p2 : ℝ × ℝ) (a b m : ℝ) : Prop :=
  let l1 := a * p1.1 + b * p1.2 + m
  let l2 := a * p2.1 + b * p2.2 + m
  l1 * l2 < 0

-- Point definitions:
def point1 : ℝ × ℝ := (1, 3)
def point2 : ℝ × ℝ := (-4, -2)

-- Line definition with coefficients
def a : ℝ := 2
def b : ℝ := 1

-- Proof Goal:
theorem range_of_m (m : ℝ) : is_opposite_sides_of_line point1 point2 a b m ↔ -5 < m ∧ m < 10 :=
by sorry

end NUMINAMATH_GPT_range_of_m_l1244_124464


namespace NUMINAMATH_GPT_simplify_radical_expression_l1244_124453

theorem simplify_radical_expression :
  (Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7) :=
by
  sorry

end NUMINAMATH_GPT_simplify_radical_expression_l1244_124453


namespace NUMINAMATH_GPT_boys_passed_l1244_124438

theorem boys_passed (total_boys : ℕ) (avg_marks : ℕ) (avg_passed : ℕ) (avg_failed : ℕ) (P : ℕ) 
    (h1 : total_boys = 120) (h2 : avg_marks = 36) (h3 : avg_passed = 39) (h4 : avg_failed = 15)
    (h5 : P + (total_boys - P) = 120) 
    (h6 : P * avg_passed + (total_boys - P) * avg_failed = total_boys * avg_marks) :
    P = 105 := 
sorry

end NUMINAMATH_GPT_boys_passed_l1244_124438


namespace NUMINAMATH_GPT_no_exp_function_satisfies_P_d_decreasing_nonnegative_exists_c_infinitely_many_l1244_124417

-- Define the context for real numbers and the main property P.
def property_P (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x + f (x + 2) ≤ 2 * f (x + 1)

-- For part (1)
theorem no_exp_function_satisfies_P (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) :
  ¬ ∃ (f : ℝ → ℝ), (∀ x : ℝ, f x = a^x) ∧ property_P f :=
sorry

-- Define the context for natural numbers, d(x), and main properties related to P.
def d (f : ℕ → ℕ) (x : ℕ) : ℕ := f (x + 1) - f x

-- For part (2)(i)
theorem d_decreasing_nonnegative (f : ℕ → ℕ) (P : ∀ x : ℕ, f x + f (x + 2) ≤ 2 * f (x + 1)) :
  ∀ x : ℕ, d f (x + 1) ≤ d f x ∧ d f x ≥ 0 :=
sorry

-- For part (2)(ii)
theorem exists_c_infinitely_many (f : ℕ → ℕ) (P : ∀ x : ℕ, f x + f (x + 2) ≤ 2 * f (x + 1)) :
  ∃ c : ℕ, 0 ≤ c ∧ c ≤ d f 1 ∧ ∀ N : ℕ, ∃ n : ℕ, n > N ∧ d f n = c :=
sorry

end NUMINAMATH_GPT_no_exp_function_satisfies_P_d_decreasing_nonnegative_exists_c_infinitely_many_l1244_124417


namespace NUMINAMATH_GPT_james_marbles_left_l1244_124472

def total_initial_marbles : Nat := 28
def marbles_in_bag_A : Nat := 4
def marbles_in_bag_B : Nat := 6
def marbles_in_bag_C : Nat := 2
def marbles_in_bag_D : Nat := 8
def marbles_in_bag_E : Nat := 4
def marbles_in_bag_F : Nat := 4

theorem james_marbles_left : total_initial_marbles - marbles_in_bag_D = 20 := by
  -- James has 28 marbles initially.
  -- He gives away Bag D which has 8 marbles.
  -- 28 - 8 = 20
  sorry

end NUMINAMATH_GPT_james_marbles_left_l1244_124472


namespace NUMINAMATH_GPT_compound_interest_rate_l1244_124426

theorem compound_interest_rate (P r : ℝ) (h1 : 17640 = P * (1 + r / 100)^8)
                                (h2 : 21168 = P * (1 + r / 100)^12) :
  4 * (r / 100) = 18.6 :=
by
  sorry

end NUMINAMATH_GPT_compound_interest_rate_l1244_124426


namespace NUMINAMATH_GPT_direct_proportion_increases_inverse_proportion_increases_l1244_124405

-- Question 1: Prove y=2x increases as x increases.
theorem direct_proportion_increases (x1 x2 : ℝ) (h : x1 < x2) : 
  2 * x1 < 2 * x2 := by sorry

-- Question 2: Prove y=-2/x increases as x increases when x > 0.
theorem inverse_proportion_increases (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) : 
  - (2 / x1) < - (2 / x2) := by sorry

end NUMINAMATH_GPT_direct_proportion_increases_inverse_proportion_increases_l1244_124405


namespace NUMINAMATH_GPT_sum_of_common_ratios_eq_three_l1244_124455

variable (k p r a2 a3 b2 b3 : ℝ)

-- Conditions on the sequences:
variable (h_nz_k : k ≠ 0)  -- k is nonzero as it is scaling factor
variable (h_seq1 : a2 = k * p)
variable (h_seq2 : a3 = k * p^2)
variable (h_seq3 : b2 = k * r)
variable (h_seq4 : b3 = k * r^2)
variable (h_diff_ratios : p ≠ r)

-- The given equation:
variable (h_eq : a3^2 - b3^2 = 3 * (a2^2 - b2^2))

-- The theorem statement
theorem sum_of_common_ratios_eq_three :
  p^2 + r^2 = 3 :=
by
  -- Introduce the assumptions
  sorry

end NUMINAMATH_GPT_sum_of_common_ratios_eq_three_l1244_124455


namespace NUMINAMATH_GPT_max_shirt_price_l1244_124468

theorem max_shirt_price (total_budget : ℝ) (entrance_fee : ℝ) (num_shirts : ℝ) 
  (discount_rate : ℝ) (tax_rate : ℝ) (max_price : ℝ) 
  (budget_after_fee : total_budget - entrance_fee = 195)
  (shirt_discount : num_shirts > 15 → discounted_price = num_shirts * max_price * (1 - discount_rate))
  (price_with_tax : discounted_price * (1 + tax_rate) ≤ 195) : 
  max_price ≤ 10 := 
sorry

end NUMINAMATH_GPT_max_shirt_price_l1244_124468


namespace NUMINAMATH_GPT_minimum_distance_to_recover_cost_l1244_124476

theorem minimum_distance_to_recover_cost 
  (initial_consumption : ℝ) (modification_cost : ℝ) (modified_consumption : ℝ) (gas_cost : ℝ) : 
  22000 < (modification_cost / gas_cost) / (initial_consumption - modified_consumption) * 100 ∧ 
  (modification_cost / gas_cost) / (initial_consumption - modified_consumption) * 100 < 26000 :=
by
  let initial_consumption := 8.4
  let modified_consumption := 6.3
  let modification_cost := 400.0
  let gas_cost := 0.80
  sorry

end NUMINAMATH_GPT_minimum_distance_to_recover_cost_l1244_124476


namespace NUMINAMATH_GPT_same_sign_abc_l1244_124454
open Classical

theorem same_sign_abc (a b c : ℝ) (h1 : (b / a) * (c / a) > 1) (h2 : (b / a) + (c / a) ≥ -2) : 
  (a > 0 ∧ b > 0 ∧ c > 0) ∨ (a < 0 ∧ b < 0 ∧ c < 0) :=
sorry

end NUMINAMATH_GPT_same_sign_abc_l1244_124454


namespace NUMINAMATH_GPT_transformation_correctness_l1244_124470

theorem transformation_correctness :
  (∀ x : ℝ, 3 * x = -4 → x = -4 / 3) ∧
  (∀ x : ℝ, 5 = 2 - x → x = -3) ∧
  (∀ x : ℝ, (x - 1) / 6 - (2 * x + 3) / 8 = 1 → 4 * (x - 1) - 3 * (2 * x + 3) = 24) ∧
  (∀ x : ℝ, 3 * x - (2 - 4 * x) = 5 → 3 * x + 4 * x - 2 = 5) :=
by
  -- Prove the given conditions
  sorry

end NUMINAMATH_GPT_transformation_correctness_l1244_124470


namespace NUMINAMATH_GPT_complement_intersection_l1244_124440

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

#check (Set.compl B) ∩ A = {1}

theorem complement_intersection (hU : U = {1, 2, 3, 4, 5}) (hA : A = {1, 5}) (hB : B = {2, 3, 5}) :
   (U \ B) ∩ A = {1} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l1244_124440


namespace NUMINAMATH_GPT_households_neither_car_nor_bike_l1244_124403

-- Define the given conditions
def total_households : ℕ := 90
def car_and_bike : ℕ := 18
def households_with_car : ℕ := 44
def bike_only : ℕ := 35

-- Prove the number of households with neither car nor bike
theorem households_neither_car_nor_bike :
  (total_households - ((households_with_car + bike_only) - car_and_bike)) = 11 :=
by
  sorry

end NUMINAMATH_GPT_households_neither_car_nor_bike_l1244_124403


namespace NUMINAMATH_GPT_min_omega_l1244_124479

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem min_omega (ω φ T : ℝ) (hω : ω > 0) (hφ1 : 0 < φ) (hφ2 : φ < Real.pi / 2)
  (hT : f ω φ T = Real.sqrt 3 / 2)
  (hx : f ω φ (Real.pi / 6) = 0) :
  ω = 4 := by
  sorry

end NUMINAMATH_GPT_min_omega_l1244_124479


namespace NUMINAMATH_GPT_numbers_at_distance_1_from_neg2_l1244_124407

theorem numbers_at_distance_1_from_neg2 : 
  ∃ x : ℤ, (|x + 2| = 1) ∧ (x = -1 ∨ x = -3) :=
by
  sorry

end NUMINAMATH_GPT_numbers_at_distance_1_from_neg2_l1244_124407


namespace NUMINAMATH_GPT_equilateral_triangle_fixed_area_equilateral_triangle_max_area_l1244_124449

theorem equilateral_triangle_fixed_area (a b c : ℝ) (Δ : ℝ) (s : ℝ) (R : ℝ) (is_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  (s = (a + b + c) / 2 ∧ Δ = Real.sqrt (s * (s - a) * (s - b) * (s - c))) →
  (a * b * c = minimized ∨ a + b + c = minimized ∨ a^2 + b^2 + c^2 = minimized ∨ R = minimized) →
    (a = b ∧ b = c) :=
by
  sorry

theorem equilateral_triangle_max_area (a b c : ℝ) (Δ : ℝ) (s : ℝ) (R : ℝ) (is_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  (s = (a + b + c) / 2 ∧ Δ = Real.sqrt (s * (s - a) * (s - b) * (s - c))) →
  (a * b * c = fixed ∨ a + b + c = fixed ∨ a^2 + b^2 + c^2 = fixed ∨ R = fixed) →
  (Δ = maximized) →
    (a = b ∧ b = c) :=
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_fixed_area_equilateral_triangle_max_area_l1244_124449


namespace NUMINAMATH_GPT_y_days_worked_l1244_124430

theorem y_days_worked 
  ( W : ℝ )
  ( x_rate : ℝ := W / 21 )
  ( y_rate : ℝ := W / 15 )
  ( d : ℝ )
  ( y_work_done : ℝ := d * y_rate )
  ( x_work_done_after_y_leaves : ℝ := 14 * x_rate )
  ( total_work_done : y_work_done + x_work_done_after_y_leaves = W ) :
  d = 5 := 
sorry

end NUMINAMATH_GPT_y_days_worked_l1244_124430


namespace NUMINAMATH_GPT_ben_heads_probability_l1244_124400

def coin_flip_probability : ℚ :=
  let total_ways := 2^10
  let ways_exactly_five_heads := Nat.choose 10 5
  let probability_exactly_five_heads := ways_exactly_five_heads / total_ways
  let remaining_probability := 1 - probability_exactly_five_heads
  let probability_more_heads := remaining_probability / 2
  probability_more_heads

theorem ben_heads_probability :
  coin_flip_probability = 193 / 512 := by
  sorry

end NUMINAMATH_GPT_ben_heads_probability_l1244_124400


namespace NUMINAMATH_GPT_minimum_a_plus_b_l1244_124498

theorem minimum_a_plus_b (a b : ℤ) (h : a * b = 144) : a + b ≥ -24 :=
by sorry

end NUMINAMATH_GPT_minimum_a_plus_b_l1244_124498


namespace NUMINAMATH_GPT_prove_divisibility_l1244_124431

-- Given the conditions:
variables (a b r s : ℕ)
variables (pos_a : a > 0) (pos_b : b > 0) (pos_r : r > 0) (pos_s : s > 0)
variables (a_le_two : a ≤ 2)
variables (no_common_prime_factor : (gcd a b) = 1)
variables (divisibility_condition : (a ^ s + b ^ s) ∣ (a ^ r + b ^ r))

-- We aim to prove that:
theorem prove_divisibility : s ∣ r := 
sorry

end NUMINAMATH_GPT_prove_divisibility_l1244_124431


namespace NUMINAMATH_GPT_proportional_parts_l1244_124434

theorem proportional_parts (A B C D : ℕ) (number : ℕ) (h1 : A = 5 * x) (h2 : B = 7 * x) (h3 : C = 4 * x) (h4 : D = 8 * x) (h5 : C = 60) : number = 360 := by
  sorry

end NUMINAMATH_GPT_proportional_parts_l1244_124434


namespace NUMINAMATH_GPT_total_weight_of_ripe_fruits_correct_l1244_124459

-- Definitions based on conditions
def total_apples : ℕ := 14
def total_pears : ℕ := 10
def total_lemons : ℕ := 5

def ripe_apple_weight : ℕ := 150
def ripe_pear_weight : ℕ := 200
def ripe_lemon_weight : ℕ := 100

def unripe_apples : ℕ := 6
def unripe_pears : ℕ := 4
def unripe_lemons : ℕ := 2

def total_weight_of_ripe_fruits : ℕ :=
  (total_apples - unripe_apples) * ripe_apple_weight +
  (total_pears - unripe_pears) * ripe_pear_weight +
  (total_lemons - unripe_lemons) * ripe_lemon_weight

theorem total_weight_of_ripe_fruits_correct :
  total_weight_of_ripe_fruits = 2700 :=
by
  -- proof goes here (use sorry to skip the actual proof)
  sorry

end NUMINAMATH_GPT_total_weight_of_ripe_fruits_correct_l1244_124459


namespace NUMINAMATH_GPT_line_equation_l1244_124418

theorem line_equation
  (P : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) (hP : P = (-4, 6))
  (hxA : A.2 = 0) (hyB : B.1 = 0)
  (hMidpoint : P = ((A.1 + B.1)/2, (A.2 + B.2)/2)):
  3 * A.1 - 2 * B.2 + 24 = 0 :=
by
  -- Define point P
  let P := (-4, 6)
  -- Define points A and B, knowing P is the midpoint of AB and using conditions from the problem
  let A := (-8, 0)
  let B := (0, 12)
  sorry

end NUMINAMATH_GPT_line_equation_l1244_124418


namespace NUMINAMATH_GPT_identity_1_over_n_n_plus_1_sum_series_1_over_k_k_plus_1_sum_series_1_over_3k_minus_2_3k_plus_1_l1244_124491

-- Question 1: Prove the given identity for 1/(n(n+1))
theorem identity_1_over_n_n_plus_1 (n : ℕ) (hn : n ≠ 0) : 
  (1 : ℚ) / (n * (n + 1)) = (1 : ℚ) / n - (1 : ℚ) / (n + 1) :=
by
  sorry

-- Question 2: Prove the sum of series 1/k(k+1) from k=1 to k=2021
theorem sum_series_1_over_k_k_plus_1 : 
  (Finset.range 2021).sum (λ k => (1 : ℚ) / (k+1) / (k+2)) = 2021 / 2022 :=
by
  sorry

-- Question 3: Prove the sum of series 1/(3k-2)(3k+1) from k=1 to k=673
theorem sum_series_1_over_3k_minus_2_3k_plus_1 : 
  (Finset.range 673).sum (λ k => (1 : ℚ) / ((3 * k + 1 - 2) * (3 * k + 1))) = 674 / 2023 :=
by
  sorry

end NUMINAMATH_GPT_identity_1_over_n_n_plus_1_sum_series_1_over_k_k_plus_1_sum_series_1_over_3k_minus_2_3k_plus_1_l1244_124491


namespace NUMINAMATH_GPT_remainder_of_470521_div_5_l1244_124495

theorem remainder_of_470521_div_5 : 470521 % 5 = 1 := 
by sorry

end NUMINAMATH_GPT_remainder_of_470521_div_5_l1244_124495


namespace NUMINAMATH_GPT_intersection_at_7_m_l1244_124486

def f (x : Int) (d : Int) : Int := 4 * x + d

theorem intersection_at_7_m (d m : Int) (h₁ : f 7 d = m) (h₂ : 7 = f m d) : m = 7 := by
  sorry

end NUMINAMATH_GPT_intersection_at_7_m_l1244_124486


namespace NUMINAMATH_GPT_MMobile_cheaper_l1244_124419

-- Define the given conditions
def TMobile_base_cost : ℕ := 50
def TMobile_additional_cost : ℕ := 16
def MMobile_base_cost : ℕ := 45
def MMobile_additional_cost : ℕ := 14
def additional_lines : ℕ := 3

-- Define functions to calculate total costs
def TMobile_total_cost : ℕ := TMobile_base_cost + TMobile_additional_cost * additional_lines
def MMobile_total_cost : ℕ := MMobile_base_cost + MMobile_additional_cost * additional_lines

-- Statement to be proved
theorem MMobile_cheaper : TMobile_total_cost - MMobile_total_cost = 11 := by
  sorry

end NUMINAMATH_GPT_MMobile_cheaper_l1244_124419


namespace NUMINAMATH_GPT_point_on_x_axis_coordinates_l1244_124428

theorem point_on_x_axis_coordinates (a : ℝ) (P : ℝ × ℝ) (h : P = (a - 1, a + 2)) (hx : P.2 = 0) : P = (-3, 0) :=
by
  -- Replace this with the full proof
  sorry

end NUMINAMATH_GPT_point_on_x_axis_coordinates_l1244_124428


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1244_124487

theorem arithmetic_sequence_sum (a₁ d S : ℤ)
  (ha : 10 * a₁ + 24 * d = 37) :
  19 * (a₁ + 2 * d) + (a₁ + 10 * d) = 74 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1244_124487


namespace NUMINAMATH_GPT_geometric_series_sum_l1244_124458

noncomputable def T (r : ℝ) := 15 / (1 - r)

theorem geometric_series_sum (b : ℝ) (hb1 : -1 < b) (hb2 : b < 1) (H : T b * T (-b) = 3240) : T b + T (-b) = 432 := 
by sorry

end NUMINAMATH_GPT_geometric_series_sum_l1244_124458


namespace NUMINAMATH_GPT_food_last_after_join_l1244_124423

-- Define the conditions
def initial_men := 760
def additional_men := 2280
def initial_days := 22
def days_before_join := 2
def initial_food := initial_men * initial_days
def remaining_food := initial_food - (initial_men * days_before_join)
def total_men := initial_men + additional_men

-- Define the goal to prove
theorem food_last_after_join :
  (remaining_food / total_men) = 5 :=
by
  sorry

end NUMINAMATH_GPT_food_last_after_join_l1244_124423


namespace NUMINAMATH_GPT_probability_exact_n_points_l1244_124477

open Classical

noncomputable def probability_of_n_points (n : ℕ) : ℚ :=
  1/3 * (2 + (-1/2)^n)

theorem probability_exact_n_points (n : ℕ) :
  ∀ n : ℕ, probability_of_n_points n = 1/3 * (2 + (-1/2)^n) :=
sorry

end NUMINAMATH_GPT_probability_exact_n_points_l1244_124477


namespace NUMINAMATH_GPT_y1_gt_y2_l1244_124475

theorem y1_gt_y2 (k : ℝ) (y1 y2 : ℝ) (hA : y1 = k * (-3) + 3) (hB : y2 = k * 1 + 3) (hK : k < 0) : y1 > y2 :=
by 
  sorry

end NUMINAMATH_GPT_y1_gt_y2_l1244_124475


namespace NUMINAMATH_GPT_distinct_solutions_abs_eq_l1244_124402

theorem distinct_solutions_abs_eq (x : ℝ) : (|x - 5| = |x + 3|) → x = 1 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_distinct_solutions_abs_eq_l1244_124402


namespace NUMINAMATH_GPT_find_f_of_monotonic_and_condition_l1244_124444

noncomputable def monotonic (f : ℝ → ℝ) :=
  ∀ {a b : ℝ}, a < b → f a ≤ f b

theorem find_f_of_monotonic_and_condition (f : ℝ → ℝ) (h_mono : monotonic f) (h_cond : ∀ x : ℝ, 0 < x → f (f x - x^2) = 6) : f 2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_f_of_monotonic_and_condition_l1244_124444


namespace NUMINAMATH_GPT_product_of_xy_l1244_124437

-- Define the problem conditions
variables (x y : ℝ)
-- Define the condition that |x-3| and |y+1| are opposite numbers
def opposite_abs_values := |x - 3| = - |y + 1|

-- State the theorem
theorem product_of_xy (h : opposite_abs_values x y) : x * y = -3 :=
sorry -- Proof is omitted

end NUMINAMATH_GPT_product_of_xy_l1244_124437


namespace NUMINAMATH_GPT_half_angle_quadrant_l1244_124412

variables {α : ℝ} {k : ℤ} {n : ℤ}

theorem half_angle_quadrant (h : ∃ k : ℤ, k * 360 + 180 < α ∧ α < k * 360 + 270) :
  ∃ (n : ℤ), (n * 360 + 90 < α / 2 ∧ α / 2 < n * 360 + 135) ∨ 
      (n * 360 + 270 < α / 2 ∧ α / 2 < n * 360 + 315) :=
by sorry

end NUMINAMATH_GPT_half_angle_quadrant_l1244_124412


namespace NUMINAMATH_GPT_find_circle_equation_l1244_124404

-- Define the intersection point of the lines x + y + 1 = 0 and x - y - 1 = 0
def center : ℝ × ℝ := (0, -1)

-- Define the chord length AB
def chord_length : ℝ := 6

-- Line equation that intersects the circle
def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y - 11 = 0

-- Circle equation to be proven
def circle_eq (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 18

-- Main theorem: Prove that the given circle equation is correct under the conditions
theorem find_circle_equation (x y : ℝ) (hc : x + y + 1 = 0) (hc' : x - y - 1 = 0) 
  (hl : line_eq x y) : circle_eq x y :=
sorry

end NUMINAMATH_GPT_find_circle_equation_l1244_124404


namespace NUMINAMATH_GPT_count_numbers_with_digit_sum_10_l1244_124480

theorem count_numbers_with_digit_sum_10 : 
  ∃ n : ℕ, 
  (n = 66) ∧ ∀ (a b c : ℕ), 
  0 ≤ a ∧ a ≤ 9 ∧ 
  0 ≤ b ∧ b ≤ 9 ∧ 
  0 ≤ c ∧ c ≤ 9 ∧ 
  a + b + c = 10 → 
  true :=
by
  sorry

end NUMINAMATH_GPT_count_numbers_with_digit_sum_10_l1244_124480


namespace NUMINAMATH_GPT_arrange_natural_numbers_divisors_l1244_124471

theorem arrange_natural_numbers_divisors :
  ∃ (seq : List ℕ), seq = [7, 1, 8, 4, 10, 6, 9, 3, 2, 5] ∧ 
  seq.length = 10 ∧
  ∀ n (h : n < seq.length), seq[n] ∣ (List.take n seq).sum := 
by
  sorry

end NUMINAMATH_GPT_arrange_natural_numbers_divisors_l1244_124471


namespace NUMINAMATH_GPT_expected_value_is_150_l1244_124427

noncomputable def expected_value_of_winnings : ℝ :=
  let p := (1:ℝ)/8
  let winnings := [0, 2, 3, 5, 7]
  let losses := [4, 6]
  let extra := 5
  let win_sum := (winnings.sum : ℝ)
  let loss_sum := (losses.sum : ℝ)
  let E := p * 0 + p * win_sum - p * loss_sum + p * extra
  E

theorem expected_value_is_150 : expected_value_of_winnings = 1.5 := 
by sorry

end NUMINAMATH_GPT_expected_value_is_150_l1244_124427


namespace NUMINAMATH_GPT_tan_theta_sub_9pi_l1244_124401

theorem tan_theta_sub_9pi (θ : ℝ) (h : Real.cos (Real.pi + θ) = -1 / 2) : 
  Real.tan (θ - 9 * Real.pi) = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_theta_sub_9pi_l1244_124401


namespace NUMINAMATH_GPT_bar_graph_proportion_correct_l1244_124478

def white : ℚ := 1/2
def black : ℚ := 1/4
def gray : ℚ := 1/8
def light_gray : ℚ := 1/16

theorem bar_graph_proportion_correct :
  (white = 1 / 2) ∧
  (black = white / 2) ∧
  (gray = black / 2) ∧
  (light_gray = gray / 2) →
  (white = 1 / 2) ∧
  (black = 1 / 4) ∧
  (gray = 1 / 8) ∧
  (light_gray = 1 / 16) :=
by
  intros
  sorry

end NUMINAMATH_GPT_bar_graph_proportion_correct_l1244_124478


namespace NUMINAMATH_GPT_condition1_condition2_condition3_l1244_124416

noncomputable def Z (m : ℝ) : ℂ := (m^2 - 4 * m) + (m^2 - m - 6) * Complex.I

-- Condition 1: Point Z is in the third quadrant
theorem condition1 (m : ℝ) (h_quad3 : (m^2 - 4 * m) < 0 ∧ (m^2 - m - 6) < 0) : 0 < m ∧ m < 3 :=
sorry

-- Condition 2: Point Z is on the imaginary axis
theorem condition2 (m : ℝ) (h_imaginary : (m^2 - 4 * m) = 0 ∧ (m^2 - m - 6) ≠ 0) : m = 0 ∨ m = 4 :=
sorry

-- Condition 3: Point Z is on the line x - y + 3 = 0
theorem condition3 (m : ℝ) (h_line : (m^2 - 4 * m) - (m^2 - m - 6) + 3 = 0) : m = 3 :=
sorry

end NUMINAMATH_GPT_condition1_condition2_condition3_l1244_124416


namespace NUMINAMATH_GPT_max_profit_l1244_124433

noncomputable def C (x : ℝ) : ℝ :=
  if x < 80 then (1/2) * x^2 + 40 * x
  else 101 * x + 8100 / x - 2180

noncomputable def profit (x : ℝ) : ℝ :=
  if x < 80 then 100 * x - C x - 500
  else 100 * x - C x - 500

theorem max_profit :
  (∀ x, (0 < x ∧ x < 80) → profit x = - (1/2) * x^2 + 60 * x - 500) ∧
  (∀ x, (80 ≤ x) → profit x = 1680 - (x + 8100 / x)) ∧
  (∃ x, x = 90 ∧ profit x = 1500) :=
by {
  -- Proof here
  sorry
}

end NUMINAMATH_GPT_max_profit_l1244_124433


namespace NUMINAMATH_GPT_possible_values_f_zero_l1244_124469

theorem possible_values_f_zero (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + y) = 2 * f x * f y) :
    f 0 = 0 ∨ f 0 = 1 / 2 := 
sorry

end NUMINAMATH_GPT_possible_values_f_zero_l1244_124469


namespace NUMINAMATH_GPT_solve_for_t_l1244_124448

variable (S₁ S₂ u t : ℝ)

theorem solve_for_t 
  (h₀ : u ≠ 0) 
  (h₁ : u = (S₁ - S₂) / (t - 1)) :
  t = (S₁ - S₂ + u) / u :=
by
  sorry

end NUMINAMATH_GPT_solve_for_t_l1244_124448


namespace NUMINAMATH_GPT_derivative_y_l1244_124450

noncomputable def y (x : ℝ) : ℝ := Real.sin x - Real.exp (x * Real.log 2)

theorem derivative_y (x : ℝ) : 
  deriv y x = Real.cos x - Real.exp (x * Real.log 2) * Real.log 2 := 
by 
  sorry

end NUMINAMATH_GPT_derivative_y_l1244_124450


namespace NUMINAMATH_GPT_largest_4digit_congruent_17_mod_28_l1244_124439

theorem largest_4digit_congruent_17_mod_28 :
  ∃ n, n < 10000 ∧ n % 28 = 17 ∧ ∀ m, m < 10000 ∧ m % 28 = 17 → m ≤ 9982 :=
by
  sorry

end NUMINAMATH_GPT_largest_4digit_congruent_17_mod_28_l1244_124439


namespace NUMINAMATH_GPT_smallest_three_digit_multiple_of_three_with_odd_hundreds_l1244_124432

theorem smallest_three_digit_multiple_of_three_with_odd_hundreds :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (∃ a b c : ℕ, n = 100 * a + 10 * b + c ∧ a % 2 = 1 ∧ n % 3 = 0 ∧ n = 102) :=
by
  sorry

end NUMINAMATH_GPT_smallest_three_digit_multiple_of_three_with_odd_hundreds_l1244_124432
