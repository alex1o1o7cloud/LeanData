import Mathlib

namespace temp_interpretation_l838_83855

theorem temp_interpretation (below_zero : ℤ) (above_zero : ℤ) (h : below_zero = -2):
  above_zero = 3 → 3 = 0 := by
  intro h2
  have : above_zero = 3 := h2
  sorry

end temp_interpretation_l838_83855


namespace calculate_expression_l838_83810

theorem calculate_expression : -1^2021 + 1^2022 = 0 := by
  sorry

end calculate_expression_l838_83810


namespace kittens_per_bunny_l838_83861

-- Conditions
def total_initial_bunnies : ℕ := 30
def fraction_given_to_friend : ℚ := 2 / 5
def total_bunnies_after_birth : ℕ := 54

-- Determine the number of kittens each bunny gave birth to
theorem kittens_per_bunny (initial_bunnies given_fraction total_bunnies_after : ℕ) 
  (h1 : initial_bunnies = total_initial_bunnies)
  (h2 : given_fraction = fraction_given_to_friend)
  (h3 : total_bunnies_after = total_bunnies_after_birth) :
  (total_bunnies_after - (total_initial_bunnies - (total_initial_bunnies * fraction_given_to_friend))) / 
    (total_initial_bunnies * (1 - fraction_given_to_friend)) = 2 :=
by
  sorry

end kittens_per_bunny_l838_83861


namespace largest_number_not_sum_of_two_composites_l838_83893

-- Define what it means to be a composite number
def isComposite (n : ℕ) : Prop :=
∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

-- Define the problem statement
theorem largest_number_not_sum_of_two_composites :
  ∃ n : ℕ, (¬∃ a b : ℕ, isComposite a ∧ isComposite b ∧ n = a + b) ∧
           ∀ m : ℕ, (¬∃ x y : ℕ, isComposite x ∧ isComposite y ∧ m = x + y) → m ≥ n :=
  sorry

end largest_number_not_sum_of_two_composites_l838_83893


namespace king_middle_school_teachers_l838_83875

theorem king_middle_school_teachers 
    (students : ℕ)
    (classes_per_student : ℕ)
    (normal_class_size : ℕ)
    (special_classes : ℕ)
    (special_class_size : ℕ)
    (classes_per_teacher : ℕ)
    (H1 : students = 1500)
    (H2 : classes_per_student = 5)
    (H3 : normal_class_size = 30)
    (H4 : special_classes = 10)
    (H5 : special_class_size = 15)
    (H6 : classes_per_teacher = 3) : 
    ∃ teachers : ℕ, teachers = 85 :=
by
  sorry

end king_middle_school_teachers_l838_83875


namespace parabola_tangent_xaxis_at_p2_parabola_vertex_yaxis_at_p0_parabolas_symmetric_m_point_parabola_familiy_point_through_l838_83881

noncomputable def parabola (p x : ℝ) : ℝ := (p-1) * x^2 + 2 * p * x + 4

-- 1. Prove that if \( p = 2 \), the parabola \( g_p \) is tangent to the \( x \)-axis.
theorem parabola_tangent_xaxis_at_p2 : ∀ x, parabola 2 x = (x + 2)^2 := 
by 
  intro x
  sorry

-- 2. Prove that if \( p = 0 \), the vertex of the parabola \( g_p \) lies on the \( y \)-axis.
theorem parabola_vertex_yaxis_at_p0 : ∃ x, parabola 0 x = 4 := 
by 
  sorry

-- 3. Prove the parabolas for \( p = 2 \) and \( p = 0 \) are symmetric with respect to \( M(-1, 2) \).
theorem parabolas_symmetric_m_point : ∀ x, 
  (parabola 2 x = (x + 2)^2) → 
  (parabola 0 x = -x^2 + 4) → 
  (-1, 2) = (-1, 2) := 
by 
  sorry

-- 4. Prove that the points \( (0, 4) \) and \( (-2, 0) \) lie on the curve for all \( p \).
theorem parabola_familiy_point_through : ∀ p, 
  parabola p 0 = 4 ∧ 
  parabola p (-2) = 0 :=
by 
  sorry

end parabola_tangent_xaxis_at_p2_parabola_vertex_yaxis_at_p0_parabolas_symmetric_m_point_parabola_familiy_point_through_l838_83881


namespace cucumber_kinds_l838_83838

theorem cucumber_kinds (x : ℕ) :
  (3 * 5) + (4 * x) + 30 + 85 = 150 → x = 5 :=
by
  intros h
  -- h : 15 + 4 * x + 30 + 85 = 150 

  -- Proof would go here
  sorry

end cucumber_kinds_l838_83838


namespace ζ_sum_8_l838_83899

open Complex

def ζ1 : ℂ := sorry
def ζ2 : ℂ := sorry
def ζ3 : ℂ := sorry

def e1 := ζ1 + ζ2 + ζ3
def e2 := ζ1 * ζ2 + ζ2 * ζ3 + ζ3 * ζ1
def e3 := ζ1 * ζ2 * ζ3

axiom h1 : e1 = 2
axiom h2 : e1^2 - 2 * e2 = 8
axiom h3 : (e1^2 - 2 * e2)^2 - 2 * (e2^2 - 2 * e1 * e3) = 26

theorem ζ_sum_8 : ζ1^8 + ζ2^8 + ζ3^8 = 219 :=
by {
  -- The proof goes here, omitting solution steps as instructed.
  sorry
}

end ζ_sum_8_l838_83899


namespace floor_neg_seven_fourths_l838_83813

theorem floor_neg_seven_fourths : Int.floor (-7 / 4) = -2 := 
by
  sorry

end floor_neg_seven_fourths_l838_83813


namespace find_b_l838_83887

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
  2 * x / (x^2 + b * x + 1)

noncomputable def f_inverse (y : ℝ) : ℝ :=
  (1 - y) / y

theorem find_b (b : ℝ) (h : ∀ x, f_inverse (f x b) = x) : b = 4 :=
sorry

end find_b_l838_83887


namespace gcd_1995_228_eval_f_at_2_l838_83852

-- Euclidean Algorithm Problem
theorem gcd_1995_228 : Nat.gcd 1995 228 = 57 :=
by
  sorry

-- Horner's Method Problem
def f (x : ℝ) : ℝ := 3 * x ^ 5 + 2 * x ^ 3 - 8 * x + 5

theorem eval_f_at_2 : f 2 = 101 :=
by
  sorry

end gcd_1995_228_eval_f_at_2_l838_83852


namespace exponent_m_n_add_l838_83824

variable (a : ℝ) (m n : ℕ)

theorem exponent_m_n_add (h1 : a ^ m = 2) (h2 : a ^ n = 3) : a ^ (m + n) = 6 := by
  sorry

end exponent_m_n_add_l838_83824


namespace find_g_at_6_l838_83841

def g (x : ℝ) : ℝ := 3 * x ^ 4 - 20 * x ^ 3 + 37 * x ^ 2 - 18 * x - 80

theorem find_g_at_6 : g 6 = 712 := by
  -- We apply the remainder theorem to determine the value of g(6).
  sorry

end find_g_at_6_l838_83841


namespace math_problem_equivalence_l838_83888

section

variable (x y z : ℝ) (w : String)

theorem math_problem_equivalence (h₀ : x / 15 = 4 / 5) (h₁ : y = 80) (h₂ : z = 0.8) (h₃ : w = "八折"):
  x = 12 ∧ y = 80 ∧ z = 0.8 ∧ w = "八折" :=
by
  sorry

end

end math_problem_equivalence_l838_83888


namespace remainder_of_N_mod_103_l838_83857

noncomputable def N : ℕ :=
  sorry -- This will capture the mathematical calculation of N using the conditions stated.

theorem remainder_of_N_mod_103 : (N % 103) = 43 :=
  sorry

end remainder_of_N_mod_103_l838_83857


namespace Kelly_baking_powder_difference_l838_83847

theorem Kelly_baking_powder_difference : 0.4 - 0.3 = 0.1 :=
by 
  -- sorry is a placeholder for a proof
  sorry

end Kelly_baking_powder_difference_l838_83847


namespace union_A_B_l838_83858

-- Define them as sets
def A : Set ℝ := {x | -3 < x ∧ x ≤ 2}
def B : Set ℝ := {x | -2 < x ∧ x ≤ 3}

-- Statement of the theorem
theorem union_A_B : A ∪ B = {x | -3 < x ∧ x ≤ 3} :=
by
  sorry

end union_A_B_l838_83858


namespace distance_between_points_l838_83896

theorem distance_between_points :
  let x1 := 1
  let y1 := 16
  let x2 := 9
  let y2 := 3
  let distance := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  distance = Real.sqrt 233 :=
by
  sorry

end distance_between_points_l838_83896


namespace max_area_guaranteed_l838_83844

noncomputable def max_rectangle_area (board_size : ℕ) (removed_cells : ℕ) : ℕ :=
  if board_size = 8 ∧ removed_cells = 8 then 8 else 0

theorem max_area_guaranteed :
  max_rectangle_area 8 8 = 8 :=
by
  -- Proof logic goes here
  sorry

end max_area_guaranteed_l838_83844


namespace number_of_kg_of_mangoes_l838_83876

variable {m : ℕ}
def cost_apples := 8 * 70
def cost_mangoes (m : ℕ) := 75 * m
def total_cost := 1235

theorem number_of_kg_of_mangoes (h : cost_apples + cost_mangoes m = total_cost) : m = 9 :=
by
  sorry

end number_of_kg_of_mangoes_l838_83876


namespace IncorrectOption_l838_83869

namespace Experiment

def OptionA : Prop := 
  ∃ method : String, method = "sampling detection"

def OptionB : Prop := 
  ¬(∃ experiment : String, experiment = "does not need a control group, nor repeated experiments")

def OptionC : Prop := 
  ∃ action : String, action = "test tube should be gently shaken"

def OptionD : Prop := 
  ∃ condition : String, condition = "field of view should not be too bright"

theorem IncorrectOption : OptionB :=
  sorry

end Experiment

end IncorrectOption_l838_83869


namespace equal_distribution_l838_83884

variables (Emani Howard : ℕ)

-- Emani has $30 more than Howard
axiom emani_condition : Emani = Howard + 30

-- Emani has $150
axiom emani_has_money : Emani = 150

theorem equal_distribution : (Emani + Howard) / 2 = 135 :=
by
  sorry

end equal_distribution_l838_83884


namespace range_of_a_l838_83889

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - 2 * x - a

theorem range_of_a (a : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ↔ a > 2 - 2 * Real.log 2 :=
by
  sorry

end range_of_a_l838_83889


namespace complement_in_N_l838_83897

variable (M : Set ℕ) (N : Set ℕ)
def complement_N (M N : Set ℕ) : Set ℕ := { x ∈ N | x ∉ M }

theorem complement_in_N (M : Set ℕ) (N : Set ℕ) : 
  M = {2, 3, 4} → N = {0, 2, 3, 4, 5} → complement_N M N = {0, 5} :=
by
  intro hM hN
  subst hM
  subst hN 
  -- sorry is used to skip the proof
  sorry

end complement_in_N_l838_83897


namespace kate_money_left_l838_83867

def kate_savings_march := 27
def kate_savings_april := 13
def kate_savings_may := 28
def kate_expenditure_keyboard := 49
def kate_expenditure_mouse := 5

def total_savings := kate_savings_march + kate_savings_april + kate_savings_may
def total_expenditure := kate_expenditure_keyboard + kate_expenditure_mouse
def money_left := total_savings - total_expenditure

-- Prove that Kate has $14 left
theorem kate_money_left : money_left = 14 := 
by 
  sorry

end kate_money_left_l838_83867


namespace range_of_a_l838_83850

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := if x ≤ 1 then (a - 3) * x - 3 else Real.log x / Real.log a

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ 3 < a ∧ a ≤ 6 :=
by
  sorry

end range_of_a_l838_83850


namespace designated_time_to_B_l838_83882

theorem designated_time_to_B (s v : ℝ) (x : ℝ) (V' : ℝ)
  (h1 : s / 2 = (x + 2) * V')
  (h2 : s / (2 * V') + 1 + s / (2 * (V' + v)) = x) :
  x = (v + Real.sqrt (9 * v ^ 2 + 6 * v * s)) / v :=
by
  sorry

end designated_time_to_B_l838_83882


namespace problem1_problem2_l838_83835

-- Sub-problem 1
theorem problem1 (x y : ℝ) (h1 : 9 * x + 10 * y = 1810) (h2 : 11 * x + 8 * y = 1790) : 
  x - y = -10 := 
sorry

-- Sub-problem 2
theorem problem2 (x y : ℝ) (h1 : 2 * x + 2.5 * y = 1200) (h2 : 1000 * x + 900 * y = 530000) :
  x = 350 ∧ y = 200 := 
sorry

end problem1_problem2_l838_83835


namespace students_play_football_l838_83800

theorem students_play_football (total_students : ℕ) (C : ℕ) (B : ℕ) (neither : ℕ) (F : ℕ)
  (h1 : total_students = 460)
  (h2 : C = 175)
  (h3 : B = 90)
  (h4 : neither = 50)
  (h5 : total_students = neither + F + C - B) : 
  F = 325 :=
by 
  sorry

end students_play_football_l838_83800


namespace correct_number_of_statements_l838_83845

-- Definitions based on the problem's conditions
def condition_1 : Prop :=
  ∀ (n : ℕ) (a b c d e : ℚ), n = 5 ∧ ∃ x y z, (x = a ∧ y = b ∧ z = c) ∧ (x < 0 ∧ y < 0 ∧ z < 0 ∧ d ≥ 0 ∧ e ≥ 0) →
  (a * b * c * d * e < 0 ∨ a * b * c * d * e = 0)

def condition_2 : Prop := 
  ∀ m : ℝ, |m| + m = 0 → m ≤ 0

def condition_3 : Prop := 
  ∀ a b : ℝ, (1 / a < 1 / b) → ¬ (a < b ∨ b < a)

def condition_4 : Prop := 
  ∀ a : ℝ, ∃ max_val, max_val = 5 ∧ 5 - |a - 5| ≤ max_val

-- Main theorem to state the correct number of true statements
theorem correct_number_of_statements : 
  (condition_2 ∧ condition_4) ∧
  ¬condition_1 ∧ 
  ¬condition_3 :=
by
  sorry

end correct_number_of_statements_l838_83845


namespace teams_have_equal_people_l838_83840

-- Definitions capturing the conditions
def managers : Nat := 3
def employees : Nat := 3
def teams : Nat := 3

-- The total number of people
def total_people : Nat := managers + employees

-- The proof statement
theorem teams_have_equal_people : total_people / teams = 2 := by
  sorry

end teams_have_equal_people_l838_83840


namespace integer_roots_of_polynomial_l838_83866

theorem integer_roots_of_polynomial : 
  {x : ℤ | x^3 - 6 * x^2 - 4 * x + 24 = 0} = {2, -2} :=
by
  sorry

end integer_roots_of_polynomial_l838_83866


namespace days_C_alone_l838_83892

theorem days_C_alone (r_A r_B r_C : ℝ) (h1 : r_A + r_B = 1 / 3) (h2 : r_B + r_C = 1 / 6) (h3 : r_A + r_C = 5 / 18) : 
  1 / r_C = 18 := 
  sorry

end days_C_alone_l838_83892


namespace proof_problem_l838_83808

variable (A B C : ℕ)

-- Defining the conditions
def condition1 : Prop := A + B + C = 700
def condition2 : Prop := B + C = 600
def condition3 : Prop := C = 200

-- Stating the proof problem
theorem proof_problem (h1 : condition1 A B C) (h2 : condition2 B C) (h3 : condition3 C) : A + C = 300 :=
sorry

end proof_problem_l838_83808


namespace find_min_value_l838_83842

noncomputable def expression (x : ℝ) : ℝ :=
  (Real.sin x ^ 8 + Real.cos x ^ 8 + 2) / (Real.sin x ^ 6 + Real.cos x ^ 6 + 2)

theorem find_min_value : ∃ x : ℝ, expression x = 5 / 4 :=
sorry

end find_min_value_l838_83842


namespace moles_of_NaOH_combined_l838_83816

-- Define the reaction conditions
variable (moles_NH4NO3 : ℕ) (moles_NaNO3 : ℕ)

-- Define a proof problem that asserts the number of moles of NaOH combined
theorem moles_of_NaOH_combined
  (h1 : moles_NH4NO3 = 3)  -- 3 moles of NH4NO3 are combined
  (h2 : moles_NaNO3 = 3)  -- 3 moles of NaNO3 are formed
  : ∃ moles_NaOH : ℕ, moles_NaOH = 3 :=
by {
  -- Proof skeleton to be filled
  sorry
}

end moles_of_NaOH_combined_l838_83816


namespace union_A_B_l838_83854

def setA : Set ℝ := { x | Real.log x / Real.log (1/2) > -1 }
def setB : Set ℝ := { x | 2^x > Real.sqrt 2 }

theorem union_A_B : setA ∪ setB = { x | 0 < x } := by
  sorry

end union_A_B_l838_83854


namespace range_of_a_for_decreasing_function_l838_83880

variable (a : ℝ)

noncomputable def f (x : ℝ) : ℝ := -x^2 - 2 * (a - 1) * x + 5

noncomputable def f' (x : ℝ) : ℝ := -2 * x - 2 * (a - 1)

theorem range_of_a_for_decreasing_function :
  (∀ x : ℝ, -1 ≤ x → f' a x ≤ 0) → 2 ≤ a := sorry

end range_of_a_for_decreasing_function_l838_83880


namespace tweets_when_hungry_l838_83843

theorem tweets_when_hungry (H : ℕ) : 
  (18 * 20) + (H * 20) + (45 * 20) = 1340 → H = 4 := by
  sorry

end tweets_when_hungry_l838_83843


namespace four_digit_integer_l838_83883

theorem four_digit_integer (a b c d : ℕ) 
    (h1 : a + b + c + d = 16) 
    (h2 : b + c = 10) 
    (h3 : a - d = 2) 
    (h4 : (a - b + c - d) % 11 = 0) : 
    a = 4 ∧ b = 4 ∧ c = 6 ∧ d = 2 :=
sorry

end four_digit_integer_l838_83883


namespace evaluate_expression_l838_83898

theorem evaluate_expression : (1 - 1 / (1 - 1 / (1 + 2))) = (-1 / 2) :=
by sorry

end evaluate_expression_l838_83898


namespace find_x_squared_plus_y_squared_l838_83825

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 20) (h2 : x * y = 16) : x^2 + y^2 = 432 :=
sorry

end find_x_squared_plus_y_squared_l838_83825


namespace average_score_l838_83805

theorem average_score (N : ℕ) (p3 p2 p1 p0 : ℕ) (n : ℕ) 
  (H1 : N = 3)
  (H2 : p3 = 30)
  (H3 : p2 = 50)
  (H4 : p1 = 10)
  (H5 : p0 = 10)
  (H6 : n = 20)
  (H7 : p3 + p2 + p1 + p0 = 100) :
  (3 * (p3 * n / 100) + 2 * (p2 * n / 100) + 1 * (p1 * n / 100) + 0 * (p0 * n / 100)) / n = 2 :=
by 
  sorry

end average_score_l838_83805


namespace discount_percentage_l838_83817

theorem discount_percentage (cost_price marked_price : ℝ) (profit_percentage : ℝ) 
  (h_cost_price : cost_price = 47.50)
  (h_marked_price : marked_price = 65)
  (h_profit_percentage : profit_percentage = 0.30) :
  ((marked_price - (cost_price + (profit_percentage * cost_price))) / marked_price) * 100 = 5 :=
by
  sorry

end discount_percentage_l838_83817


namespace number_of_pairs_of_shoes_size_40_to_42_200_pairs_l838_83807

theorem number_of_pairs_of_shoes_size_40_to_42_200_pairs 
  (total_pairs_sample : ℕ)
  (freq_3rd_group : ℝ)
  (freq_1st_group : ℕ)
  (freq_2nd_group : ℕ)
  (freq_4th_group : ℕ)
  (total_pairs_200 : ℕ)
  (scaled_pairs_size_40_42 : ℕ)
: total_pairs_sample = 40 ∧ freq_3rd_group = 0.25 ∧ freq_1st_group = 6 ∧ freq_2nd_group = 7 ∧ freq_4th_group = 9 ∧ total_pairs_200 = 200 ∧ scaled_pairs_size_40_42 = 40 :=
sorry

end number_of_pairs_of_shoes_size_40_to_42_200_pairs_l838_83807


namespace no_equal_differences_between_products_l838_83836

theorem no_equal_differences_between_products (a b c d : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
    (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
    (h_order : a < b ∧ b < c ∧ c < d) :
    ¬ (∃ k : ℕ, ac - ab = k ∧ ad - ac = k ∧ bc - ad = k ∧ bd - bc = k ∧ cd - bd = k) :=
by
  sorry

end no_equal_differences_between_products_l838_83836


namespace northern_village_population_l838_83874

theorem northern_village_population
    (x : ℕ) -- Northern village population
    (western_village_population : ℕ := 400)
    (southern_village_population : ℕ := 200)
    (total_conscripted : ℕ := 60)
    (northern_village_conscripted : ℕ := 10)
    (h : (northern_village_conscripted : ℚ) / total_conscripted = (x : ℚ) / (x + western_village_population + southern_village_population)) : 
    x = 120 :=
    sorry

end northern_village_population_l838_83874


namespace restore_arithmetic_operations_l838_83819

/--
Given the placeholders \(A, B, C, D, E\) for operations in the equations:
1. \(4 A 2 = 2\)
2. \(8 = 4 C 2\)
3. \(2 D 3 = 5\)
4. \(4 = 5 E 1\)

Prove that:
(a) \(A = ÷\)
(b) \(B = =\)
(c) \(C = ×\)
(d) \(D = +\)
(e) \(E = -\)
-/
theorem restore_arithmetic_operations {A B C D E : String} (h1 : B = "=") 
    (h2 : "4" ++ A  ++ "2" ++ B ++ "2" = "4" ++ "÷" ++ "2" ++ "=" ++ "2")
    (h3 : "8" ++ "=" ++ "4" ++ C ++ "2" = "8" ++ "=" ++ "4" ++ "×" ++ "2")
    (h4 : "2" ++ D ++ "3" ++ "=" ++ "5" = "2" ++ "+" ++ "3" ++ "=" ++ "5")
    (h5 : "4" ++ "=" ++ "5" ++ E ++ "1" = "4" ++ "=" ++ "5" ++ "-" ++ "1") :
  (A = "÷") ∧ (B = "=") ∧ (C = "×") ∧ (D = "+") ∧ (E = "-") := by
    sorry

end restore_arithmetic_operations_l838_83819


namespace simplify_expression_l838_83872

theorem simplify_expression (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 2 :=
by
  sorry

end simplify_expression_l838_83872


namespace fill_in_square_l838_83877

variable {α : Type*} [CommRing α]

theorem fill_in_square (a b : α) (square : α) (h : square * 3 * a * b = 3 * a^2 * b) : square = a :=
sorry

end fill_in_square_l838_83877


namespace division_remainder_l838_83871

theorem division_remainder (dividend quotient divisor remainder : ℕ) 
  (h_dividend : dividend = 12401) 
  (h_quotient : quotient = 76) 
  (h_divisor : divisor = 163) 
  (h_remainder : dividend = quotient * divisor + remainder) : 
  remainder = 13 := 
by
  sorry

end division_remainder_l838_83871


namespace sequence_contradiction_l838_83812

open Classical

variable {α : Type} (a : ℕ → α) [PartialOrder α]

theorem sequence_contradiction {a : ℕ → ℝ} :
  (∀ n, a n < 2) ↔ ¬ ∃ k, a k ≥ 2 := 
by sorry

end sequence_contradiction_l838_83812


namespace double_root_possible_values_l838_83851

theorem double_root_possible_values (b_3 b_2 b_1 : ℤ) (s : ℤ)
  (h : (Polynomial.X - Polynomial.C s) ^ 2 ∣
    Polynomial.C 24 + Polynomial.C b_1 * Polynomial.X + Polynomial.C b_2 * Polynomial.X ^ 2 + Polynomial.C b_3 * Polynomial.X ^ 3 + Polynomial.X ^ 4) :
  s = -2 ∨ s = -1 ∨ s = 1 ∨ s = 2 :=
sorry

end double_root_possible_values_l838_83851


namespace fraction_solution_l838_83894

theorem fraction_solution (a : ℕ) (h : a > 0) (h_eq : (a : ℚ) / (a + 45) = 0.75) : a = 135 :=
sorry

end fraction_solution_l838_83894


namespace lemonade_glasses_from_fruit_l838_83829

noncomputable def lemons_per_glass : ℕ := 2
noncomputable def oranges_per_glass : ℕ := 1
noncomputable def total_lemons : ℕ := 18
noncomputable def total_oranges : ℕ := 10
noncomputable def grapefruits : ℕ := 6
noncomputable def lemons_per_grapefruit : ℕ := 2
noncomputable def oranges_per_grapefruit : ℕ := 1

theorem lemonade_glasses_from_fruit :
  (total_lemons / lemons_per_glass) = 9 →
  (total_oranges / oranges_per_glass) = 10 →
  min (total_lemons / lemons_per_glass) (total_oranges / oranges_per_glass) = 9 →
  (grapefruits * lemons_per_grapefruit = 12) →
  (grapefruits * oranges_per_grapefruit = 6) →
  (9 + grapefruits) = 15 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end lemonade_glasses_from_fruit_l838_83829


namespace six_digit_numbers_with_zero_l838_83823

theorem six_digit_numbers_with_zero : 
  let total := 9 * 10^5
  let without_zero := 9^6 
  total - without_zero = 368559 := 
by
  let total := 9 * 10^5
  let without_zero := 9^6 
  show total - without_zero = 368559
  sorry

end six_digit_numbers_with_zero_l838_83823


namespace max_possible_value_l838_83890

theorem max_possible_value (P Q : ℤ) (hP : P * P ≤ 729 ∧ 729 ≤ -P * P * P)
  (hQ : Q * Q ≤ 729 ∧ 729 ≤ -Q * Q * Q) :
  10 * (P - Q) = 180 :=
by
  sorry

end max_possible_value_l838_83890


namespace min_distance_origin_to_intersections_l838_83864

theorem min_distance_origin_to_intersections (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (hline : (1 : ℝ)/a + 4/b = 1) :
  |(0 : ℝ) - a| + |(0 : ℝ) - b| = 9 :=
sorry

end min_distance_origin_to_intersections_l838_83864


namespace identity_proof_l838_83822

theorem identity_proof (A B C A1 B1 C1 : ℝ) :
  (A^2 + B^2 + C^2) * (A1^2 + B1^2 + C1^2) - (A * A1 + B * B1 + C * C1)^2 =
    (A * B1 + A1 * B)^2 + (A * C1 + A1 * C)^2 + (B * C1 + B1 * C)^2 :=
by
  sorry

end identity_proof_l838_83822


namespace diophantine_eq_unique_solutions_l838_83804

theorem diophantine_eq_unique_solutions (x y : ℕ) (hx_positive : x > 0) (hy_positive : y > 0) :
  x^y = y^x + 1 ↔ (x = 2 ∧ y = 1) ∨ (x = 3 ∧ y = 2) :=
by
  sorry

end diophantine_eq_unique_solutions_l838_83804


namespace find_r_l838_83826

theorem find_r (f g : ℝ → ℝ) (monic_f : ∀x, f x = (x - r - 2) * (x - r - 8) * (x - a))
  (monic_g : ∀x, g x = (x - r - 4) * (x - r - 10) * (x - b)) (h : ∀ x, f x - g x = r):
  r = 32 :=
by
  sorry

end find_r_l838_83826


namespace solve_inequality_system_l838_83853

theorem solve_inequality_system (x : ℝ) :
  (x - 1 < 2 * x + 1) ∧ ((2 * x - 5) / 3 ≤ 1) → (-2 < x ∧ x ≤ 4) :=
by
  intro cond
  sorry

end solve_inequality_system_l838_83853


namespace greatest_remainder_when_dividing_by_10_l838_83879

theorem greatest_remainder_when_dividing_by_10 (x : ℕ) : 
  ∃ r : ℕ, r < 10 ∧ r = x % 10 ∧ r = 9 :=
by
  sorry

end greatest_remainder_when_dividing_by_10_l838_83879


namespace eighth_group_number_correct_stratified_sampling_below_30_correct_l838_83802

noncomputable def systematic_sampling_eighth_group_number 
  (total_employees : ℕ) (sample_size : ℕ) (groups : ℕ) (fifth_group_number : ℕ) : ℕ :=
  let interval := total_employees / groups
  let initial_number := fifth_group_number - 4 * interval
  initial_number + 7 * interval

theorem eighth_group_number_correct :
  systematic_sampling_eighth_group_number 200 40 40 22 = 37 :=
  sorry

noncomputable def stratified_sampling_below_30_persons 
  (total_employees : ℕ) (sample_size : ℕ) (percent_below_30 : ℕ) : ℕ :=
  (percent_below_30 * sample_size) / 100

theorem stratified_sampling_below_30_correct :
  stratified_sampling_below_30_persons 200 40 40 = 16 :=
  sorry

end eighth_group_number_correct_stratified_sampling_below_30_correct_l838_83802


namespace proposition_correctness_l838_83859

theorem proposition_correctness :
  (∀ x : ℝ, (|x-1| < 2) → (x < 3)) ∧
  (∀ (P Q : Prop), (Q → ¬ P) → (P → ¬ Q)) :=
by 
sorry

end proposition_correctness_l838_83859


namespace necessary_and_sufficient_condition_l838_83860

theorem necessary_and_sufficient_condition (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 > 2 ∧ x2 < 2 ∧ x1^2 - m * x1 - 1 = 0 ∧ x2^2 - m * x2 - 1 = 0) ↔ m > 1.5 :=
by
  sorry

end necessary_and_sufficient_condition_l838_83860


namespace single_shot_percentage_decrease_l838_83811

theorem single_shot_percentage_decrease
  (initial_salary : ℝ)
  (final_salary : ℝ := initial_salary * 0.95 * 0.90 * 0.85) :
  ((1 - final_salary / initial_salary) * 100) = 27.325 := by
  sorry

end single_shot_percentage_decrease_l838_83811


namespace length_of_ribbon_l838_83849

theorem length_of_ribbon (perimeter : ℝ) (sides : ℕ) (h1 : perimeter = 42) (h2 : sides = 6) : (perimeter / sides) = 7 :=
by {
  sorry
}

end length_of_ribbon_l838_83849


namespace remaining_amount_to_be_paid_l838_83868

theorem remaining_amount_to_be_paid (part_payment : ℝ) (percentage : ℝ) (h : part_payment = 650 ∧ percentage = 0.15) :
    (part_payment / percentage - part_payment) = 3683.33 := by
  cases h with
  | intro h1 h2 =>
    sorry

end remaining_amount_to_be_paid_l838_83868


namespace gift_equation_l838_83834

theorem gift_equation (x : ℝ) : 15 * (x + 40) = 900 := 
by
  sorry

end gift_equation_l838_83834


namespace frog_reaches_C_in_n_jumps_frog_reaches_C_no_D_in_n_jumps_frog_alive_probability_frog_average_lifespan_l838_83821

-- Part (a): Prove the number of ways to reach vertex C from A in n jumps when n is even
theorem frog_reaches_C_in_n_jumps (n : ℕ) (h_even : n % 2 = 0) : 
    ∃ c : ℕ, c = (4^n/2 - 1) / 3 := by sorry

-- Part (b): Prove the number of ways to reach vertex C from A in n jumps without jumping to D when n is even
theorem frog_reaches_C_no_D_in_n_jumps (n : ℕ) (h_even : n % 2 = 0) : 
    ∃ c : ℕ, c = 3^(n/2 - 1) := by sorry

-- Part (c): Prove the probability the frog is alive after n jumps with a mine at D
theorem frog_alive_probability (n : ℕ) (k : ℕ) (h_n : n = 2*k - 1 ∨ n = 2*k) : 
    ∃ p : ℝ, p = (3/4)^(k-1) := by sorry

-- Part (d): Prove the average lifespan of the frog in the presence of a mine at D
theorem frog_average_lifespan : 
    ∃ t : ℝ, t = 9 := by sorry

end frog_reaches_C_in_n_jumps_frog_reaches_C_no_D_in_n_jumps_frog_alive_probability_frog_average_lifespan_l838_83821


namespace sequence_properties_l838_83803

-- Definitions from conditions
def S (n : ℕ) := n^2 - n
def a (n : ℕ) := if n = 1 then 0 else 2 * (n - 1)
def b (n : ℕ) := 2^(n - 1)
def c (n : ℕ) := a n * b n
def T (n : ℕ) := (n - 2) * 2^(n + 1) + 4

-- Theorem statement proving the required identities
theorem sequence_properties {n : ℕ} (hn : n ≠ 0) :
  (a n = (if n = 1 then 0 else 2 * (n - 1))) ∧ 
  (b 2 = a 2) ∧ 
  (b 4 = a 5) ∧ 
  (T n = (n - 2) * 2^(n + 1) + 4) := by
  sorry

end sequence_properties_l838_83803


namespace remainder_when_divided_by_6_l838_83891

theorem remainder_when_divided_by_6 (a : ℕ) (h1 : a % 2 = 1) (h2 : a % 3 = 2) : a % 6 = 5 :=
sorry

end remainder_when_divided_by_6_l838_83891


namespace length_of_AB_l838_83820

noncomputable def ratio3to5 (AP PB : ℝ) : Prop := AP / PB = 3 / 5
noncomputable def ratio4to5 (AQ QB : ℝ) : Prop := AQ / QB = 4 / 5
noncomputable def pointDistances (P Q : ℝ) : Prop := P - Q = 3

theorem length_of_AB (A B P Q : ℝ) (P_on_AB : P > A ∧ P < B) (Q_on_AB : Q > A ∧ Q < B)
  (middle_side : P < (A + B) / 2 ∧ Q < (A + B) / 2)
  (h1 : ratio3to5 (P - A) (B - P))
  (h2 : ratio4to5 (Q - A) (B - Q))
  (h3 : pointDistances P Q) : B - A = 43.2 := 
sorry

end length_of_AB_l838_83820


namespace sixth_root_of_unity_l838_83801

/- Constants and Variables -/
variable (p q r s t k : ℂ)
variable (nz_p : p ≠ 0) (nz_q : q ≠ 0) (nz_r : r ≠ 0) (nz_s : s ≠ 0) (nz_t : t ≠ 0)
variable (hk1 : p * k^5 + q * k^4 + r * k^3 + s * k^2 + t * k + p = 0)
variable (hk2 : q * k^5 + r * k^4 + s * k^3 + t * k^2 + p * k + q = 0)

/- Theorem to prove -/
theorem sixth_root_of_unity : k^6 = 1 :=
by sorry

end sixth_root_of_unity_l838_83801


namespace number_of_permutations_l838_83833

theorem number_of_permutations (readers : Fin 8 → Type) : ∃! (n : ℕ), n = 40320 :=
by
  sorry

end number_of_permutations_l838_83833


namespace log_identity_l838_83862

theorem log_identity
  (x : ℝ)
  (h1 : x < 1)
  (h2 : (Real.log x / Real.log 10)^2 - Real.log (x^4) / Real.log 10 = 100) :
  (Real.log x / Real.log 10)^3 - Real.log (x^5) / Real.log 10 = -114 + Real.sqrt 104 := 
by
  sorry

end log_identity_l838_83862


namespace arithmetic_sequence_sum_l838_83870

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Define the arithmetic property of the sequence
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum
  (h1 : is_arithmetic_sequence a d)
  (h2 : a 2 + a 4 + a 7 + a 11 = 44) :
  a 3 + a 5 + a 10 = 33 := 
sorry

end arithmetic_sequence_sum_l838_83870


namespace proof_allison_brian_noah_l838_83839

-- Definitions based on the problem conditions

-- Definition for the cubes
def allison_cube := [6, 6, 6, 6, 6, 6]
def brian_cube := [1, 2, 2, 3, 3, 4]
def noah_cube := [3, 3, 3, 3, 5, 5]

-- Helper function to calculate the probability of succeeding conditions
def probability_succeeding (A B C : List ℕ) : ℚ :=
  if (A.all (λ x => x = 6)) ∧ (B.all (λ x => x ≤ 5)) ∧ (C.all (λ x => x ≤ 5)) then 1 else 0

-- Define the proof statement for the given problem
theorem proof_allison_brian_noah :
  probability_succeeding allison_cube brian_cube noah_cube = 1 :=
by
  -- Since all conditions fulfill the requirement, we'll use sorry to skip the proof for now
  sorry

end proof_allison_brian_noah_l838_83839


namespace total_divisions_is_48_l838_83809

-- Definitions based on the conditions
def initial_cells := 1
def final_cells := 1993
def cells_added_division_42 := 41
def cells_added_division_44 := 43

-- The main statement we want to prove
theorem total_divisions_is_48 (a b : ℕ) 
  (h1 : cells_added_division_42 = 41)
  (h2 : cells_added_division_44 = 43)
  (h3 : cells_added_division_42 * a + cells_added_division_44 * b = final_cells - initial_cells) :
  a + b = 48 := 
sorry

end total_divisions_is_48_l838_83809


namespace tan_theta_eq_1_over_3_l838_83814

noncomputable def unit_circle_point (θ : ℝ) : Prop :=
  let x := Real.cos θ
  let y := Real.sin θ
  (x^2 + y^2 = 1) ∧ (θ = Real.arccos ((4*x + 3*y) / 5))

theorem tan_theta_eq_1_over_3 (θ : ℝ) (h : unit_circle_point θ) : Real.tan θ = 1 / 3 := 
by
  sorry

end tan_theta_eq_1_over_3_l838_83814


namespace cos_double_angle_l838_83848

open Real

theorem cos_double_angle (α : ℝ) (h : tan α = 3) : cos (2 * α) = -4 / 5 :=
sorry

end cos_double_angle_l838_83848


namespace focus_of_parabola_l838_83865

theorem focus_of_parabola (x y : ℝ) : (y^2 + 4 * x = 0) → (x = -1 ∧ y = 0) :=
by sorry

end focus_of_parabola_l838_83865


namespace right_angled_triangle_k_values_l838_83831

def i : ℝ × ℝ := (1, 0)
def j : ℝ × ℝ := (0, 1)

def AB : ℝ × ℝ := (2, 1)
def AC (k : ℝ) : ℝ × ℝ := (3, k)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def BC (k : ℝ) : ℝ × ℝ := (1, k - 1)

theorem right_angled_triangle_k_values (k : ℝ) :
  (dot_product AB (AC k) = 0 ∨ dot_product AB (BC k) = 0 ∨ dot_product (BC k) (AC k) = 0) ↔ (k = -6 ∨ k = -1) :=
sorry

end right_angled_triangle_k_values_l838_83831


namespace mean_eq_value_of_z_l838_83873

theorem mean_eq_value_of_z (z : ℤ) : 
  ((6 + 15 + 9 + 20) / 4 : ℚ) = ((13 + z) / 2 : ℚ) → (z = 12) := by
  sorry

end mean_eq_value_of_z_l838_83873


namespace intersection_when_a_eq_1_range_for_A_inter_B_eq_A_l838_83895

noncomputable def f (x : ℝ) : ℝ := Real.log (3 - abs (x - 1))

def setA : Set ℝ := { x | 3 - abs (x - 1) > 0 }

def setB (a : ℝ) : Set ℝ := { x | x^2 - (a + 5) * x + 5 * a < 0 }

theorem intersection_when_a_eq_1 : (setA ∩ setB 1) = { x | 1 < x ∧ x < 4 } :=
by
  sorry

theorem range_for_A_inter_B_eq_A : { a | (setA ∩ setB a) = setA } = { a | a ≤ -2 } :=
by
  sorry

end intersection_when_a_eq_1_range_for_A_inter_B_eq_A_l838_83895


namespace yoongi_has_5_carrots_l838_83863

def yoongis_carrots (initial_carrots sister_gave: ℕ) : ℕ :=
  initial_carrots + sister_gave

theorem yoongi_has_5_carrots : yoongis_carrots 3 2 = 5 := by 
  sorry

end yoongi_has_5_carrots_l838_83863


namespace teams_played_same_matches_l838_83832

theorem teams_played_same_matches (n : ℕ) (h : n = 30)
  (matches_played : Fin n → ℕ) :
  ∃ (i j : Fin n), i ≠ j ∧ matches_played i = matches_played j :=
by
  sorry

end teams_played_same_matches_l838_83832


namespace misha_current_dollars_l838_83830

variable (x : ℕ)

def misha_needs_more : ℕ := 13
def total_amount : ℕ := 47

theorem misha_current_dollars : x = total_amount - misha_needs_more → x = 34 :=
by
  sorry

end misha_current_dollars_l838_83830


namespace sector_area_maximized_l838_83815

noncomputable def maximize_sector_area (r θ : ℝ) : Prop :=
  2 * r + θ * r = 20 ∧
  (r > 0 ∧ θ > 0) ∧
  ∀ (r' θ' : ℝ), (2 * r' + θ' * r' = 20 ∧ r' > 0 ∧ θ' > 0) → (1/2 * θ' * r'^2 ≤ 1/2 * θ * r^2)

theorem sector_area_maximized : maximize_sector_area 5 2 :=
by
  sorry

end sector_area_maximized_l838_83815


namespace point_A_equidistant_l838_83878

/-
This statement defines the problem of finding the coordinates of point A that is equidistant from points B and C.
-/
theorem point_A_equidistant (x : ℝ) :
  (dist (x, 0, 0) (3, 5, 6)) = (dist (x, 0, 0) (1, 2, 3)) ↔ x = 14 :=
by {
  sorry
}

end point_A_equidistant_l838_83878


namespace batsman_average_after_12th_innings_l838_83886

noncomputable def batsman_average (runs_in_12th_innings : ℕ) (average_increase : ℕ) (initial_average_after_11_innings : ℕ) : ℕ :=
initial_average_after_11_innings + average_increase

theorem batsman_average_after_12th_innings
(score_in_12th_innings : ℕ)
(average_increase : ℕ)
(initial_average_after_11_innings : ℕ)
(total_runs_after_11_innings := 11 * initial_average_after_11_innings)
(total_runs_after_12_innings := total_runs_after_11_innings + score_in_12th_innings)
(new_average_after_12_innings := total_runs_after_12_innings / 12)
:
score_in_12th_innings = 80 ∧ average_increase = 3 ∧ initial_average_after_11_innings = 44 → 
batsman_average score_in_12th_innings average_increase initial_average_after_11_innings = 47 := 
by
  -- skipping the actual proof for now
  sorry

end batsman_average_after_12th_innings_l838_83886


namespace sinks_per_house_l838_83837

theorem sinks_per_house (total_sinks : ℕ) (houses : ℕ) (h_total_sinks : total_sinks = 266) (h_houses : houses = 44) :
  total_sinks / houses = 6 :=
by {
  sorry
}

end sinks_per_house_l838_83837


namespace prove_value_l838_83846

variable (m n : ℤ)

-- Conditions from the problem
def condition1 : Prop := m^2 + 2 * m * n = 384
def condition2 : Prop := 3 * m * n + 2 * n^2 = 560

-- Proposition to be proved
theorem prove_value (h1 : condition1 m n) (h2 : condition2 m n) : 2 * m^2 + 13 * m * n + 6 * n^2 - 444 = 2004 := by
  sorry

end prove_value_l838_83846


namespace correct_option_l838_83885

theorem correct_option :
  (3 * a^2 - a^2 = 2 * a^2) ∧
  (¬ (a^2 * a^3 = a^6)) ∧
  (¬ ((3 * a)^2 = 6 * a^2)) ∧
  (¬ (a^6 / a^3 = a^2)) :=
by
  -- We only need to state the theorem; the proof details are omitted per the instructions.
  sorry

end correct_option_l838_83885


namespace smallest_three_digit_number_l838_83818

theorem smallest_three_digit_number :
  ∃ x : ℕ, 100 ≤ x ∧ x < 1000 ∧
  (x % 2 = 0) ∧
  ((x + 1) % 3 = 0) ∧
  ((x + 2) % 4 = 0) ∧
  ((x + 3) % 5 = 0) ∧
  ((x + 4) % 6 = 0) ∧
  x = 122 :=
by
  sorry

end smallest_three_digit_number_l838_83818


namespace seahawks_final_score_l838_83827

def num_touchdowns : ℕ := 4
def num_field_goals : ℕ := 3
def points_per_touchdown : ℕ := 7
def points_per_fieldgoal : ℕ := 3

theorem seahawks_final_score : (num_touchdowns * points_per_touchdown) + (num_field_goals * points_per_fieldgoal) = 37 := by
  sorry

end seahawks_final_score_l838_83827


namespace square_inequality_l838_83828

theorem square_inequality (a b : ℝ) (h1 : a + b < 0) (h2 : b > 0) : a^2 > b^2 :=
by
  sorry

end square_inequality_l838_83828


namespace pauline_bought_2_pounds_of_meat_l838_83806

theorem pauline_bought_2_pounds_of_meat :
  ∀ (cost_taco_shells cost_bell_pepper cost_meat_per_pound total_spent : ℝ) 
    (num_bell_peppers : ℕ),
  cost_taco_shells = 5 →
  cost_bell_pepper = 1.5 →
  cost_meat_per_pound = 3 →
  total_spent = 17 →
  num_bell_peppers = 4 →
  (total_spent - (cost_taco_shells + (num_bell_peppers * cost_bell_pepper))) / cost_meat_per_pound = 2 :=
by
  intros cost_taco_shells cost_bell_pepper cost_meat_per_pound total_spent num_bell_peppers 
         h1 h2 h3 h4 h5
  sorry

end pauline_bought_2_pounds_of_meat_l838_83806


namespace common_root_equations_l838_83856

theorem common_root_equations (a b : ℝ) 
  (h : ∃ x₀ : ℝ, (x₀ ^ 2 + a * x₀ + b = 0) ∧ (x₀ ^ 2 + b * x₀ + a = 0)) 
  (hc : ∀ x₁ x₂ : ℝ, (x₁ ^ 2 + a * x₁ + b = 0 ∧ x₂ ^ 2 + bx₀ + a = 0) → x₁ = x₂) :
  a + b = -1 :=
sorry

end common_root_equations_l838_83856
