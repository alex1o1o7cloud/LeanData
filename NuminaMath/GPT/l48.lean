import Mathlib

namespace estimate_students_spending_more_than_60_l48_48905

-- Definition of the problem
def students_surveyed : ℕ := 50
def students_inclined_to_subscribe : ℕ := 8
def total_students : ℕ := 1000
def estimated_students : ℕ := 600

-- Define the proof task
theorem estimate_students_spending_more_than_60 :
  (students_inclined_to_subscribe : ℝ) / (students_surveyed : ℝ) * (total_students : ℝ) = estimated_students :=
by
  sorry

end estimate_students_spending_more_than_60_l48_48905


namespace geometric_series_sum_l48_48952

theorem geometric_series_sum :
  let a := 1 / 4
  let r := 1 / 4
  let n := 5
  (a + a * r + a * r^2 + a * r^3 + a * r^4) = 341 / 1024 :=
by
  let a := 1 / 4
  let r := 1 / 4
  let n := 5
  -- Proof steps go here
  sorry

end geometric_series_sum_l48_48952


namespace compare_fractions_l48_48479

-- Define the fractions
def frac1 : ℚ := -2/3
def frac2 : ℚ := -3/4

-- Prove that -2/3 > -3/4
theorem compare_fractions : frac1 > frac2 :=
by {
  sorry
}

end compare_fractions_l48_48479


namespace johnny_marbles_l48_48581

def num_ways_to_choose_marbles (n k : ℕ) : ℕ := Nat.choose n k

theorem johnny_marbles :
  num_ways_to_choose_marbles 7 3 = 35 :=
by
  sorry

end johnny_marbles_l48_48581


namespace solve_equation_l48_48619

theorem solve_equation : ∃ x : ℝ, 2 * x + 1 = 0 ∧ x = -1 / 2 := by
  sorry

end solve_equation_l48_48619


namespace number_of_towers_l48_48053

noncomputable def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def multinomial (n : ℕ) (k1 k2 k3 : ℕ) : ℕ :=
  factorial n / (factorial k1 * factorial k2 * factorial k3)

theorem number_of_towers :
  (multinomial 10 3 3 4 = 4200) :=
by
  sorry

end number_of_towers_l48_48053


namespace find_initial_solution_liters_l48_48977

-- Define the conditions
def percentage_initial_solution_alcohol := 0.26
def added_water := 5
def percentage_new_mixture_alcohol := 0.195

-- Define the initial amount of the solution
def initial_solution_liters (x : ℝ) : Prop :=
  0.26 * x = 0.195 * (x + 5)

-- State the proof problem
theorem find_initial_solution_liters : initial_solution_liters 15 :=
by
  sorry

end find_initial_solution_liters_l48_48977


namespace relation_between_abc_l48_48976

theorem relation_between_abc (a b c : ℕ) (h₁ : a = 3 ^ 44) (h₂ : b = 4 ^ 33) (h₃ : c = 5 ^ 22) : a > b ∧ b > c :=
by
  -- Proof goes here
  sorry

end relation_between_abc_l48_48976


namespace num_A_is_9_l48_48524

-- Define the total number of animals
def total_animals : ℕ := 17

-- Define the number of animal B
def num_B : ℕ := 8

-- Define the number of animal A
def num_A : ℕ := total_animals - num_B

-- Statement to prove
theorem num_A_is_9 : num_A = 9 :=
by
  sorry

end num_A_is_9_l48_48524


namespace tom_average_score_increase_l48_48707

def initial_scores : List ℕ := [72, 78, 81]
def fourth_exam_score : ℕ := 90

theorem tom_average_score_increase :
  let initial_avg := (initial_scores.sum : ℚ) / (initial_scores.length : ℚ)
  let total_score_after_fourth := initial_scores.sum + fourth_exam_score
  let new_avg := (total_score_after_fourth : ℚ) / (initial_scores.length + 1 : ℚ)
  new_avg - initial_avg = 3.25 := by 
  -- Proof goes here
  sorry

end tom_average_score_increase_l48_48707


namespace max_items_with_discount_l48_48997

theorem max_items_with_discount (total_money items original_price discount : ℕ) 
  (h_orig: original_price = 30)
  (h_discount: discount = 24) 
  (h_limit: items > 5 → (total_money <= 270)) : items ≤ 10 :=
by
  sorry

end max_items_with_discount_l48_48997


namespace sum_of_digits_of_m_eq_nine_l48_48417

theorem sum_of_digits_of_m_eq_nine
  (m : ℕ)
  (h1 : m * 3 / 2 - 72 = m) :
  1 + (m / 10 % 10) + (m % 10) = 9 :=
by
  sorry

end sum_of_digits_of_m_eq_nine_l48_48417


namespace range_of_a_l48_48679

open Real

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x ^ 2 + a * x - 1 < 0) ↔ -4 < a ∧ a ≤ 0 := 
by
  sorry

end range_of_a_l48_48679


namespace range_of_b_l48_48050

open Real

theorem range_of_b (b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 4 → abs (y - (x + b)) = 1) ↔ -sqrt 2 < b ∧ b < sqrt 2 := 
by sorry

end range_of_b_l48_48050


namespace donuts_per_box_l48_48725

-- Define the conditions and the theorem
theorem donuts_per_box :
  (10 * 12 - 12 - 8) / 10 = 10 :=
by
  sorry

end donuts_per_box_l48_48725


namespace grandfather_age_correct_l48_48557

-- Let's define the conditions
def xiaowen_age : ℕ := 13
def grandfather_age : ℕ := 5 * xiaowen_age + 8

-- The statement to prove
theorem grandfather_age_correct : grandfather_age = 73 := by
  sorry

end grandfather_age_correct_l48_48557


namespace widgets_production_l48_48536

variables (A B C : ℝ)
variables (P : ℝ)

-- Conditions provided
def condition1 : Prop := 7 * A + 11 * B = 305
def condition2 : Prop := 8 * A + 22 * C = P

-- The question we need to answer
def question : Prop :=
  ∃ Q : ℝ, Q = 8 * (A + B + C)

theorem widgets_production (h1 : condition1 A B) (h2 : condition2 A C P) :
  question A B C :=
sorry

end widgets_production_l48_48536


namespace convex_polyhedron_P_T_V_sum_eq_34_l48_48453

theorem convex_polyhedron_P_T_V_sum_eq_34
  (F : ℕ) (V : ℕ) (E : ℕ) (T : ℕ) (P : ℕ) 
  (hF : F = 32)
  (hT1 : 3 * T + 5 * P = 960)
  (hT2 : 2 * E = V * (T + P))
  (hT3 : T + P - 2 = 60)
  (hT4 : F + V - E = 2) :
  P + T + V = 34 := by
  sorry

end convex_polyhedron_P_T_V_sum_eq_34_l48_48453


namespace technicians_count_l48_48576

theorem technicians_count {T R : ℕ} (h1 : T + R = 12) (h2 : 2 * T + R = 18) : T = 6 :=
sorry

end technicians_count_l48_48576


namespace fraction_n_m_l48_48722

noncomputable def a (k : ℝ) := 2*k + 1
noncomputable def b (k : ℝ) := 3*k + 2
noncomputable def c (k : ℝ) := 3 - 4*k
noncomputable def S (k : ℝ) := a k + 2*(b k) + 3*(c k)

theorem fraction_n_m : 
  (∀ (k : ℝ), -1/2 ≤ k ∧ k ≤ 3/4 → (S (3/4) = 11 ∧ S (-1/2) = 16)) → 
  11/16 = 11 / 16 :=
by
  sorry

end fraction_n_m_l48_48722


namespace cos_of_sum_eq_one_l48_48879

theorem cos_of_sum_eq_one
  (x y : ℝ)
  (a : ℝ)
  (h1 : x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4))
  (h2 : y ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4))
  (h3 : x^3 + Real.sin x - 2 * a = 0)
  (h4 : 4 * y^3 + Real.sin y * Real.cos y + a = 0) :
  Real.cos (x + 2 * y) = 1 := 
by
  sorry

end cos_of_sum_eq_one_l48_48879


namespace no_real_roots_x2_plus_4_l48_48904

theorem no_real_roots_x2_plus_4 : ¬ ∃ x : ℝ, x^2 + 4 = 0 := by
  sorry

end no_real_roots_x2_plus_4_l48_48904


namespace necessary_but_not_sufficient_condition_l48_48418

theorem necessary_but_not_sufficient_condition (a b c d : ℝ) : 
  (a + b < c + d) → (a < c ∨ b < d) :=
sorry

end necessary_but_not_sufficient_condition_l48_48418


namespace least_multiplier_l48_48843

theorem least_multiplier (x: ℕ) (h1: 72 * x % 112 = 0) (h2: ∀ y, 72 * y % 112 = 0 → x ≤ y) : x = 14 :=
sorry

end least_multiplier_l48_48843


namespace thickness_of_layer_l48_48363

noncomputable def volume_of_sphere (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

noncomputable def volume_of_cylinder (r h : ℝ) : ℝ := Real.pi * r^2 * h

theorem thickness_of_layer (radius_sphere radius_cylinder : ℝ) (volume_sphere volume_cylinder : ℝ) (h : ℝ) : 
  radius_sphere = 3 → 
  radius_cylinder = 10 →
  volume_sphere = volume_of_sphere radius_sphere →
  volume_cylinder = volume_of_cylinder radius_cylinder h →
  volume_sphere = volume_cylinder → 
  h = 9 / 25 :=
by
  intros
  sorry

end thickness_of_layer_l48_48363


namespace min_value_x2_sub_xy_add_y2_l48_48710

/-- Given positive real numbers x and y such that x^2 + xy + 3y^2 = 10, 
prove that the minimum value of x^2 - xy + y^2 is 2. -/
theorem min_value_x2_sub_xy_add_y2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^2 + x * y + 3 * y^2 = 10) : 
  ∃ (value : ℝ), value = x^2 - x * y + y^2 ∧ value = 2 := 
by 
  sorry

end min_value_x2_sub_xy_add_y2_l48_48710


namespace fraction_of_Bs_l48_48225

theorem fraction_of_Bs 
  (num_students : ℕ)
  (As_fraction : ℚ)
  (Cs_fraction : ℚ)
  (Ds_number : ℕ)
  (total_students : ℕ) 
  (h1 : As_fraction = 1 / 5) 
  (h2 : Cs_fraction = 1 / 2) 
  (h3 : Ds_number = 40) 
  (h4 : total_students = 800) : 
  num_students / total_students = 1 / 4 :=
by
sorry

end fraction_of_Bs_l48_48225


namespace find_a₁_l48_48166

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q ^ n

noncomputable def sequence_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q ^ n) / (1 - q)

variables (a₁ q : ℝ)
-- Condition: The common ratio should not be 1.
axiom hq : q ≠ 1
-- Condition: Second term of the sequence a₂ = 1
axiom ha₂ : geometric_sequence a₁ q 1 = 1
-- Condition: 9S₃ = S₆
axiom hsum : 9 * sequence_sum a₁ q 3 = sequence_sum a₁ q 6

theorem find_a₁ : a₁ = 1 / 2 :=
  sorry

end find_a₁_l48_48166


namespace geometric_sequence_sum_l48_48394

variables (a : ℕ → ℤ) (q : ℤ)

-- assumption that the sequence is geometric
def geometric_sequence (a : ℕ → ℤ) (q : ℤ) : Prop := 
  ∀ n, a (n + 1) = a n * q

noncomputable def a2 := a 2
noncomputable def a3 := a 3
noncomputable def a4 := a 4
noncomputable def a5 := a 5
noncomputable def a6 := a 6
noncomputable def a7 := a 7

theorem geometric_sequence_sum
  (h_geom : geometric_sequence a q)
  (h1 : a2 + a3 = 1)
  (h2 : a3 + a4 = -2) :
  a5 + a6 + a7 = 24 :=
sorry

end geometric_sequence_sum_l48_48394


namespace frog_eats_per_day_l48_48635

-- Definition of the constants
def flies_morning : ℕ := 5
def flies_afternoon : ℕ := 6
def escaped_flies : ℕ := 1
def weekly_required_flies : ℕ := 14
def days_in_week : ℕ := 7

-- Prove that the frog eats 2 flies per day
theorem frog_eats_per_day : (flies_morning + flies_afternoon - escaped_flies) * days_in_week + 4 = 14 → (14 / days_in_week = 2) :=
by
  sorry

end frog_eats_per_day_l48_48635


namespace cost_price_of_cloth_l48_48964

theorem cost_price_of_cloth:
  ∀ (meters_sold profit_per_meter : ℕ) (selling_price : ℕ),
  meters_sold = 45 →
  profit_per_meter = 12 →
  selling_price = 4500 →
  (selling_price - (profit_per_meter * meters_sold)) / meters_sold = 88 :=
by
  intros meters_sold profit_per_meter selling_price h1 h2 h3
  sorry

end cost_price_of_cloth_l48_48964


namespace maria_workday_end_l48_48385

def time_in_minutes (h : ℕ) (m : ℕ) : ℕ := h * 60 + m

def start_time : ℕ := time_in_minutes 7 25
def lunch_break : ℕ := 45
def noon : ℕ := time_in_minutes 12 0
def work_hours : ℕ := 8 * 60
def end_time : ℕ := time_in_minutes 16 10

theorem maria_workday_end : start_time + (noon - start_time) + lunch_break + (work_hours - (noon - start_time)) = end_time := by
  sorry

end maria_workday_end_l48_48385


namespace remainder_of_a_squared_l48_48102

theorem remainder_of_a_squared (n : ℕ) (a : ℤ) (h : a % n * a % n % n = 1) : (a * a) % n = 1 := by
  sorry

end remainder_of_a_squared_l48_48102


namespace tangent_line_at_2_m_range_for_three_roots_l48_48284

def f (x : ℝ) : ℝ := 2 * x ^ 3 - 3 * x ^ 2 + 3

theorem tangent_line_at_2 :
  ∃ k b, k = 12 ∧ b = -17 ∧ (∀ x, 12 * x - (k * (x - 2) + f 2) = b) :=
by
  sorry

theorem m_range_for_three_roots :
  {m : ℝ | ∃ x₀ x₁ x₂, x₀ < x₁ ∧ x₁ < x₂ ∧ f x₀ + m = 0 ∧ f x₁ + m = 0 ∧ f x₂ + m = 0} = 
  {m : ℝ | -3 < m ∧ m < -2} :=
by
  sorry

end tangent_line_at_2_m_range_for_three_roots_l48_48284


namespace purchase_costs_10_l48_48825

def total_cost (a b c d e : ℝ) := a + b + c + d + e
def cost_dates (a : ℝ) := 3 * a
def cost_cantaloupe (a b : ℝ) := a - b
def cost_eggs (b c : ℝ) := b + c

theorem purchase_costs_10 (a b c d e : ℝ) 
  (h_total_cost : total_cost a b c d e = 30)
  (h_cost_dates : d = cost_dates a)
  (h_cost_cantaloupe : c = cost_cantaloupe a b)
  (h_cost_eggs : e = cost_eggs b c) :
  b + c + e = 10 :=
by
  have := h_total_cost
  have := h_cost_dates
  have := h_cost_cantaloupe
  have := h_cost_eggs
  sorry

end purchase_costs_10_l48_48825


namespace quadratic_min_value_unique_l48_48588

theorem quadratic_min_value_unique {a b c : ℝ} (h : a > 0) :
  (∀ x : ℝ, 3 * x^2 - 8 * x + 7 ≥ 3 * (4 / 3)^2 - 8 * (4 / 3) + 7) → 
  ∃ x : ℝ, x = 4 / 3 :=
by
  sorry

end quadratic_min_value_unique_l48_48588


namespace inserted_number_sq_property_l48_48925

noncomputable def inserted_number (n : ℕ) : ℕ :=
  (5 * 10^n - 1) * 10^(n+1) + 1

theorem inserted_number_sq_property (n : ℕ) : (inserted_number n)^2 = (10^(n+1) - 1)^2 :=
by sorry

end inserted_number_sq_property_l48_48925


namespace sophia_collection_value_l48_48177

-- Define the conditions
def stamps_count : ℕ := 24
def partial_stamps_count : ℕ := 8
def partial_value : ℤ := 40
def stamp_value_per_each : ℤ := partial_value / partial_stamps_count
def total_value : ℤ := stamps_count * stamp_value_per_each

-- Statement of the conclusion that needs proving
theorem sophia_collection_value :
  total_value = 120 := by
  sorry

end sophia_collection_value_l48_48177


namespace cheerleaders_uniforms_l48_48449

theorem cheerleaders_uniforms (total_cheerleaders : ℕ) (size_6_cheerleaders : ℕ) (half_size_6_cheerleaders : ℕ) (size_2_cheerleaders : ℕ) : 
  total_cheerleaders = 19 →
  size_6_cheerleaders = 10 →
  half_size_6_cheerleaders = size_6_cheerleaders / 2 →
  size_2_cheerleaders = total_cheerleaders - (size_6_cheerleaders + half_size_6_cheerleaders) →
  size_2_cheerleaders = 4 :=
by
  intros
  sorry

end cheerleaders_uniforms_l48_48449


namespace arrangement_proof_l48_48809

/-- The Happy Valley Zoo houses 5 chickens, 3 dogs, and 6 cats in a large exhibit area
    with separate but adjacent enclosures. We need to find the number of ways to place
    the 14 animals in a row of 14 enclosures, ensuring all animals of each type are together,
    and that chickens are always placed before cats, but with no restrictions regarding the
    placement of dogs. -/
def number_of_arrangements : ℕ :=
  let chickens := 5
  let dogs := 3
  let cats := 6
  let chicken_permutations := Nat.factorial chickens
  let dog_permutations := Nat.factorial dogs
  let cat_permutations := Nat.factorial cats
  let group_arrangements := 3 -- Chickens-Dogs-Cats, Dogs-Chickens-Cats, Chickens-Cats-Dogs
  group_arrangements * chicken_permutations * dog_permutations * cat_permutations

theorem arrangement_proof : number_of_arrangements = 1555200 :=
by 
  sorry

end arrangement_proof_l48_48809


namespace max_angle_C_l48_48010

-- Define the necessary context and conditions
variable {a b c : ℝ}

-- Condition that a^2 + b^2 = 2c^2 in a triangle
axiom triangle_condition : a^2 + b^2 = 2 * c^2

-- Theorem statement
theorem max_angle_C (h : a^2 + b^2 = 2 * c^2) : ∃ C : ℝ, C = Real.pi / 3 := sorry

end max_angle_C_l48_48010


namespace not_quadratic_eq3_l48_48522

-- Define the equations as functions or premises
def eq1 (x : ℝ) := 9 * x^2 = 7 * x
def eq2 (y : ℝ) := abs (y^2) = 8
def eq3 (y : ℝ) := 3 * y * (y - 1) = y * (3 * y + 1)
def eq4 (x : ℝ) := abs 2 * (x^2 + 1) = abs 10

-- Define what it means to be a quadratic equation
def is_quadratic (eq : ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, eq x = (a * x^2 + b * x + c = 0)

-- Prove that eq3 is not a quadratic equation
theorem not_quadratic_eq3 : ¬ is_quadratic eq3 :=
sorry

end not_quadratic_eq3_l48_48522


namespace percentage_increase_sale_l48_48713

theorem percentage_increase_sale (P S : ℝ) (hP : P > 0) (hS : S > 0) 
  (h1 : ∀ P S : ℝ, 0.7 * P * S * (1 + X / 100) = 1.26 * P * S) : 
  X = 80 := 
by
  sorry

end percentage_increase_sale_l48_48713


namespace arithmetic_seq_num_terms_l48_48143

theorem arithmetic_seq_num_terms (a1 : ℕ := 1) (S_odd S_even : ℕ) (n : ℕ) 
  (h1 : S_odd = 341) (h2 : S_even = 682) : 2 * n = 10 :=
by
  sorry

end arithmetic_seq_num_terms_l48_48143


namespace correct_statement_about_residuals_l48_48016

-- Define the properties and characteristics of residuals as per the definition
axiom residuals_definition : Prop
axiom residuals_usefulness : residuals_definition → Prop

-- The theorem to prove that the correct statement about residuals is that they can be used to assess the effectiveness of model fitting
theorem correct_statement_about_residuals (h : residuals_definition) : residuals_usefulness h :=
sorry

end correct_statement_about_residuals_l48_48016


namespace horizontal_asymptote_condition_l48_48554

open Polynomial

def polynomial_deg_with_horiz_asymp (p : Polynomial ℝ) : Prop :=
  degree p ≤ 4

theorem horizontal_asymptote_condition (p : Polynomial ℝ) :
  polynomial_deg_with_horiz_asymp p :=
sorry

end horizontal_asymptote_condition_l48_48554


namespace solve_x_values_l48_48477

theorem solve_x_values (x : ℝ) :
  (5 + x) / (7 + x) = (2 + x^2) / (4 + x) ↔ x = 1 ∨ x = -2 ∨ x = -3 := 
sorry

end solve_x_values_l48_48477


namespace range_of_x_for_f_lt_0_l48_48323

noncomputable def f (x : ℝ) : ℝ := x^2 - x^(1/2)

theorem range_of_x_for_f_lt_0 :
  {x : ℝ | f x < 0} = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

end range_of_x_for_f_lt_0_l48_48323


namespace number_of_kids_stayed_home_is_668278_l48_48795

  def number_of_kids_who_stayed_home : Prop :=
    ∃ X : ℕ, X + 150780 = 819058 ∧ X = 668278

  theorem number_of_kids_stayed_home_is_668278 : number_of_kids_who_stayed_home :=
    sorry
  
end number_of_kids_stayed_home_is_668278_l48_48795


namespace sum_of_three_consecutive_cubes_divisible_by_9_l48_48580

theorem sum_of_three_consecutive_cubes_divisible_by_9 (n : ℕ) : 
  (n^3 + (n + 1)^3 + (n + 2)^3) % 9 = 0 := 
by
  sorry

end sum_of_three_consecutive_cubes_divisible_by_9_l48_48580


namespace math_problem_l48_48111

theorem math_problem : 3 * 3^4 + 9^60 / 9^59 - 27^3 = -19431 := by
  sorry

end math_problem_l48_48111


namespace permutation_problem_l48_48341

noncomputable def permutation (n r : ℕ) : ℕ := (n.factorial) / ( (n - r).factorial)

theorem permutation_problem : 5 * permutation 5 3 + 4 * permutation 4 2 = 348 := by
  sorry

end permutation_problem_l48_48341


namespace rosa_initial_flowers_l48_48998

-- Definitions derived from conditions
def initial_flowers (total_flowers : ℕ) (given_flowers : ℕ) : ℕ :=
  total_flowers - given_flowers

-- The theorem stating the proof problem
theorem rosa_initial_flowers : initial_flowers 90 23 = 67 :=
by
  -- The proof goes here
  sorry

end rosa_initial_flowers_l48_48998


namespace coefficient_x18_is_zero_coefficient_x17_is_3420_l48_48913

open Polynomial

noncomputable def P : Polynomial ℚ := (1 + X^5 + X^7)^20

theorem coefficient_x18_is_zero : coeff P 18 = 0 :=
sorry

theorem coefficient_x17_is_3420 : coeff P 17 = 3420 :=
sorry

end coefficient_x18_is_zero_coefficient_x17_is_3420_l48_48913


namespace scientific_notation_of_105000_l48_48880

theorem scientific_notation_of_105000 : (105000 : ℝ) = 1.05 * 10^5 := 
by {
  sorry
}

end scientific_notation_of_105000_l48_48880


namespace find_y_l48_48432

theorem find_y (y : ℝ) (h : (y + 10 + (5 * y) + 4 + (3 * y) + 12) / 3 = 6 * y - 8) :
  y = 50 / 9 := by
  sorry

end find_y_l48_48432


namespace car_speed_constant_l48_48279

theorem car_speed_constant (v : ℝ) : 
  (1 / (v / 3600) - 1 / (80 / 3600) = 2) → v = 3600 / 47 := 
by
  sorry

end car_speed_constant_l48_48279


namespace sum_four_digit_integers_l48_48454

def a := 1000
def l := 9999
def n := 9999 - 1000 + 1
def S (n : ℕ) (a : ℕ) (l : ℕ) := n / 2 * (a + l)

theorem sum_four_digit_integers : S n a l = 49495500 :=
by
  sorry

end sum_four_digit_integers_l48_48454


namespace solve_quadratic_solve_inequalities_l48_48073
open Classical

-- Define the equation for Part 1
theorem solve_quadratic (x : ℝ) : x^2 - 6 * x + 5 = 0 → (x = 1 ∨ x = 5) :=
by
  sorry

-- Define the inequalities for Part 2
theorem solve_inequalities (x : ℝ) : (x + 3 > 0) ∧ (2 * (x - 1) < 4) → (-3 < x ∧ x < 3) :=
by
  sorry

end solve_quadratic_solve_inequalities_l48_48073


namespace greatest_x_lcm_l48_48312

theorem greatest_x_lcm (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x ≤ 105 ∧ ∃ y, y = 105 ∧ x = y := 
sorry

end greatest_x_lcm_l48_48312


namespace courtyard_width_is_14_l48_48920

-- Given conditions
def length_courtyard := 24   -- 24 meters
def num_bricks := 8960       -- Total number of bricks

@[simp]
def brick_length_m : ℝ := 0.25  -- 25 cm in meters
@[simp]
def brick_width_m : ℝ := 0.15   -- 15 cm in meters

-- Correct answer
def width_courtyard : ℝ := 14

-- Prove that the width of the courtyard is 14 meters
theorem courtyard_width_is_14 : 
  (length_courtyard * width_courtyard) = (num_bricks * (brick_length_m * brick_width_m)) :=
by
  -- Lean proof will go here
  sorry

end courtyard_width_is_14_l48_48920


namespace sumata_family_total_miles_l48_48800

theorem sumata_family_total_miles
  (days : ℝ) (miles_per_day : ℝ)
  (h1 : days = 5.0)
  (h2 : miles_per_day = 250) : 
  miles_per_day * days = 1250 := 
by
  sorry

end sumata_family_total_miles_l48_48800


namespace problem_l48_48083

theorem problem (k : ℕ) (h1 : 30^k ∣ 929260) : 3^k - k^3 = 2 :=
sorry

end problem_l48_48083


namespace major_airlines_wifi_l48_48616

-- Definitions based on conditions
def percentage (x : ℝ) := 0 ≤ x ∧ x ≤ 100

variables (W S B : ℝ)

-- Assume the conditions
axiom H1 : S = 70
axiom H2 : B = 45
axiom H3 : B ≤ S

-- The final proof problem that W = 45
theorem major_airlines_wifi : W = B :=
by
  sorry

end major_airlines_wifi_l48_48616


namespace arithmetic_sequence_sum_l48_48924

theorem arithmetic_sequence_sum (a : ℕ → Int) (a1 a2017 : Int)
  (h1 : a 1 = a1) 
  (h2017 : a 2017 = a2017)
  (roots_eq : ∀ x, x^2 - 10 * x + 16 = 0 → (x = a1 ∨ x = a2017))
  (arith_seq : ∀ n m, a (n + 1) - a n = a (m + 1) - a m) :
  a 2 + a 1009 + a 2016 = 15 :=
by
  sorry

end arithmetic_sequence_sum_l48_48924


namespace least_positive_linear_combination_24_18_l48_48747

theorem least_positive_linear_combination_24_18 (x y : ℤ) :
  ∃ (a : ℤ) (b : ℤ), 24 * a + 18 * b = 6 :=
by
  use 1
  use -1
  sorry

end least_positive_linear_combination_24_18_l48_48747


namespace curve_not_parabola_l48_48841

theorem curve_not_parabola (k : ℝ) : ¬ (∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c = 1 ∧ a * x^2 + b * y = c) :=
sorry

end curve_not_parabola_l48_48841


namespace height_of_highest_wave_l48_48092

theorem height_of_highest_wave 
  (h_austin : ℝ) -- Austin's height
  (h_high : ℝ) -- Highest wave's height
  (h_short : ℝ) -- Shortest wave's height 
  (height_relation1 : h_high = 4 * h_austin + 2)
  (height_relation2 : h_short = h_austin + 4)
  (surfboard : ℝ) (surfboard_len : surfboard = 7)
  (short_wave_len : h_short = surfboard + 3) :
  h_high = 26 :=
by
  -- Define local variables with the values from given conditions
  let austin_height := 6        -- as per calculation: 10 - 4 = 6
  let highest_wave_height := 26 -- as per calculation: (6 * 4) + 2 = 26
  sorry

end height_of_highest_wave_l48_48092


namespace alice_average_speed_l48_48653

/-- Alice cycled 40 miles at 8 miles per hour and 20 miles at 40 miles per hour. 
    The average speed for the entire trip --/
theorem alice_average_speed :
  let distance1 := 40
  let speed1 := 8
  let distance2 := 20
  let speed2 := 40
  let total_distance := distance1 + distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_time := time1 + time2
  (total_distance / total_time) = (120 / 11) := 
by
  sorry -- proof steps would go here

end alice_average_speed_l48_48653


namespace ratio_sum_l48_48193

theorem ratio_sum {x y : ℚ} (h : x / y = 3 / 4) : (x + y) / y = 7 / 4 :=
sorry

end ratio_sum_l48_48193


namespace smallest_prime_with_digit_sum_23_l48_48960

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ p > 0, Prime p ∧ digits_sum p = 23 ∧ ∀ q, Prime q ∧ digits_sum q = 23 → q ≥ p :=
by
  exists 887
  sorry -- The detailed proof steps will be provided here.

end smallest_prime_with_digit_sum_23_l48_48960


namespace inequality_solution_set_l48_48878

theorem inequality_solution_set (a : ℝ) : (-16 < a ∧ a ≤ 0) ↔ (∀ x : ℝ, a * x^2 + a * x - 4 < 0) :=
by
  sorry

end inequality_solution_set_l48_48878


namespace sum_first_10_terms_l48_48720

noncomputable def a (n : ℕ) := 1 / (4 * (n + 1) ^ 2 - 1)

theorem sum_first_10_terms : (Finset.range 10).sum a = 10 / 21 :=
by
  sorry

end sum_first_10_terms_l48_48720


namespace trains_meet_distance_l48_48805

noncomputable def time_difference : ℝ :=
  5 -- Time difference between two departures in hours

noncomputable def speed_train_a : ℝ :=
  30 -- Speed of Train A in km/h

noncomputable def speed_train_b : ℝ :=
  40 -- Speed of Train B in km/h

noncomputable def distance_train_a : ℝ :=
  speed_train_a * time_difference -- Distance covered by Train A before Train B starts

noncomputable def relative_speed : ℝ :=
  speed_train_b - speed_train_a -- Relative speed of Train B with respect to Train A

noncomputable def catch_up_time : ℝ :=
  distance_train_a / relative_speed -- Time taken for Train B to catch up with Train A

noncomputable def distance_from_delhi : ℝ :=
  speed_train_b * catch_up_time -- Distance from Delhi where the two trains will meet

theorem trains_meet_distance :
  distance_from_delhi = 600 := by
  sorry

end trains_meet_distance_l48_48805


namespace contrapositive_example_l48_48062

theorem contrapositive_example (x : ℝ) : (x > 2 → x^2 > 4) → (x^2 ≤ 4 → x ≤ 2) :=
by
  sorry

end contrapositive_example_l48_48062


namespace distribution_of_books_l48_48520

theorem distribution_of_books :
  let A := 2 -- number of identical art albums (type A)
  let B := 3 -- number of identical stamp albums (type B)
  let friends := 4 -- number of friends
  let total_ways := 5 -- total number of ways to distribute books 
  (A + B) = friends + 1 →
  total_ways = 5 := 
by
  intros A B friends total_ways h
  sorry

end distribution_of_books_l48_48520


namespace inequality_holds_for_all_xyz_in_unit_interval_l48_48571

theorem inequality_holds_for_all_xyz_in_unit_interval :
  ∀ (x y z : ℝ), (0 ≤ x ∧ x ≤ 1) → (0 ≤ y ∧ y ≤ 1) → (0 ≤ z ∧ z ≤ 1) → 
  (x / (y + z + 1) + y / (z + x + 1) + z / (x + y + 1) ≤ 1 - (1 - x) * (1 - y) * (1 - z)) :=
by
  intros x y z hx hy hz
  sorry

end inequality_holds_for_all_xyz_in_unit_interval_l48_48571


namespace total_amount_received_l48_48714
noncomputable section

variables (B : ℕ) (H1 : (1 / 3 : ℝ) * B = 50)
theorem total_amount_received (H2 : (2 / 3 : ℝ) * B = 100) (H3 : ∀ (x : ℕ), x = 5): 
  100 * 5 = 500 := 
by
  sorry

end total_amount_received_l48_48714


namespace ac_bd_bound_l48_48894

variables {a b c d : ℝ}

theorem ac_bd_bound (h1 : a^2 + b^2 = 1) (h2 : c^2 + d^2 = 4) : |a * c + b * d| ≤ 2 := 
sorry

end ac_bd_bound_l48_48894


namespace deductive_vs_inductive_l48_48888

def is_inductive_reasoning (stmt : String) : Prop :=
  match stmt with
  | "C" => True
  | _ => False

theorem deductive_vs_inductive (A B C D : String) 
  (hA : A = "All trigonometric functions are periodic functions, sin(x) is a trigonometric function, therefore sin(x) is a periodic function.")
  (hB : B = "All odd numbers cannot be divided by 2, 525 is an odd number, therefore 525 cannot be divided by 2.")
  (hC : C = "From 1=1^2, 1+3=2^2, 1+3+5=3^2, it follows that 1+3+…+(2n-1)=n^2 (n ∈ ℕ*)")
  (hD : D = "If two lines are parallel, the corresponding angles are equal. If ∠A and ∠B are corresponding angles of two parallel lines, then ∠A = ∠B.") :
  is_inductive_reasoning C :=
by
  sorry

end deductive_vs_inductive_l48_48888


namespace count_three_digit_numbers_with_identical_digits_l48_48325

/-!
# Problem Statement:
Prove that the number of three-digit numbers with at least two identical digits is 252,
given that three-digit numbers range from 100 to 999.

## Definitions:
- Three-digit numbers are those in the range 100 to 999.

## Theorem:
The number of three-digit numbers with at least two identical digits is 252.
-/
theorem count_three_digit_numbers_with_identical_digits : 
    (∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ 
    ∃ (d1 d2 d3 : ℕ), n = d1 * 100 + d2 * 10 + d3 ∧ (d1 = d2 ∨ d1 = d3 ∨ d2 = d3)) :=
sorry

end count_three_digit_numbers_with_identical_digits_l48_48325


namespace range_of_a_l48_48155

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + 2 * x - Real.exp x

theorem range_of_a (h : ∀ m n : ℝ, 0 < m → 0 < n → m > n → (f a m - f a n) / (m - n) < 2) :
  a ≤ Real.exp 1 / (2 * 1) := 
sorry

end range_of_a_l48_48155


namespace odd_function_neg_value_l48_48928

theorem odd_function_neg_value (f : ℝ → ℝ) (h_odd : ∀ x : ℝ, f (-x) = -f x) (h_value : f 1 = 1) : f (-1) = -1 :=
by
  sorry

end odd_function_neg_value_l48_48928


namespace sum_squares_mod_divisor_l48_48552

-- Define the sum of the squares from 1 to 10
def sum_squares := (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2 + 8^2 + 9^2 + 10^2)

-- Define the divisor
def divisor := 11

-- Prove that the remainder of sum_squares when divided by divisor is 0
theorem sum_squares_mod_divisor : sum_squares % divisor = 0 :=
by
  sorry

end sum_squares_mod_divisor_l48_48552


namespace power_function_k_values_l48_48826

theorem power_function_k_values (k : ℝ) :
  (∃ (a : ℝ), (k^2 - k - 5) = a ∧ (∀ x : ℝ, (k^2 - k - 5) * x^3 = a * x^3)) →
  (k = 3 ∨ k = -2) :=
by
  intro h
  sorry

end power_function_k_values_l48_48826


namespace marks_in_mathematics_l48_48665

-- Definitions for the given conditions in the problem
def marks_in_english : ℝ := 86
def marks_in_physics : ℝ := 82
def marks_in_chemistry : ℝ := 87
def marks_in_biology : ℝ := 81
def average_marks : ℝ := 85
def number_of_subjects : ℕ := 5

-- Defining the total marks based on the provided conditions
def total_marks : ℝ := average_marks * number_of_subjects

-- Proving that the marks in mathematics are 89
theorem marks_in_mathematics : total_marks - (marks_in_english + marks_in_physics + marks_in_chemistry + marks_in_biology) = 89 :=
by
  sorry

end marks_in_mathematics_l48_48665


namespace similar_triangles_area_ratio_l48_48181

theorem similar_triangles_area_ratio (ratio_angles : ℕ) (area_larger : ℕ) (h_ratio : ratio_angles = 3) (h_area_larger : area_larger = 400) :
  ∃ area_smaller : ℕ, area_smaller = 36 :=
by
  sorry

end similar_triangles_area_ratio_l48_48181


namespace sum_of_valid_single_digit_z_l48_48585

theorem sum_of_valid_single_digit_z :
  let valid_z (z : ℕ) := z < 10 ∧ (16 + z) % 3 = 0
  let sum_z := (Finset.filter valid_z (Finset.range 10)).sum id
  sum_z = 15 :=
by
  -- Proof steps are omitted
  sorry

end sum_of_valid_single_digit_z_l48_48585


namespace pentagon_arithmetic_progression_angle_l48_48923

theorem pentagon_arithmetic_progression_angle (a n : ℝ) 
  (h1 : a + (a + n) + (a + 2 * n) + (a + 3 * n) + (a + 4 * n) = 540) :
  a + 2 * n = 108 :=
by
  sorry

end pentagon_arithmetic_progression_angle_l48_48923


namespace common_difference_is_7_l48_48433

-- Define the arithmetic sequence with common difference d
def arithmetic_seq (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

-- Define the conditions
variables (a1 d : ℕ)

-- Define the conditions provided in the problem
def condition1 := (arithmetic_seq a1 d 3) + (arithmetic_seq a1 d 6) = 11
def condition2 := (arithmetic_seq a1 d 5) + (arithmetic_seq a1 d 8) = 39

-- Prove that the common difference d is 7
theorem common_difference_is_7 : condition1 a1 d → condition2 a1 d → d = 7 :=
by
  intros cond1 cond2
  sorry

end common_difference_is_7_l48_48433


namespace room_width_l48_48280

theorem room_width (length : ℝ) (total_cost : ℝ) (rate_per_sqm : ℝ) (width : ℝ)
  (h_length : length = 5.5)
  (h_total_cost : total_cost = 15400)
  (h_rate_per_sqm : rate_per_sqm = 700)
  (h_area : total_cost = rate_per_sqm * (length * width)) :
  width = 4 := 
sorry

end room_width_l48_48280


namespace prime_square_mod_30_l48_48876

theorem prime_square_mod_30 (p : ℕ) (hp : Nat.Prime p) (hp_gt_5 : p > 5) : 
  p^2 % 30 = 1 ∨ p^2 % 30 = 19 := 
sorry

end prime_square_mod_30_l48_48876


namespace chef_used_apples_l48_48465

theorem chef_used_apples (initial_apples remaining_apples used_apples : ℕ) 
  (h1 : initial_apples = 40) 
  (h2 : remaining_apples = 39) 
  (h3 : used_apples = initial_apples - remaining_apples) : 
  used_apples = 1 := 
  sorry

end chef_used_apples_l48_48465


namespace smallest_n_terminating_decimal_l48_48313

theorem smallest_n_terminating_decimal : ∃ n : ℕ, (∀ m : ℕ, m < n → (∀ k : ℕ, (n = 103 + k) → (∃ a b : ℕ, k = 2^a * 5^b)) → (k ≠ 0 → k = 125)) ∧ n = 22 := 
sorry

end smallest_n_terminating_decimal_l48_48313


namespace value_of_double_operation_l48_48559

def op1 (x : ℝ) : ℝ := 9 - x
def op2 (x : ℝ) : ℝ := x - 9

theorem value_of_double_operation :
  op2 (op1 10) = -10 := 
by 
  sorry

end value_of_double_operation_l48_48559


namespace percentage_of_ducks_among_non_heron_l48_48298

def birds_percentage (geese swans herons ducks total_birds : ℕ) : ℕ :=
  let non_heron_birds := total_birds - herons
  let duck_percentage := (ducks * 100) / non_heron_birds
  duck_percentage

theorem percentage_of_ducks_among_non_heron : 
  birds_percentage 28 20 15 32 100 = 37 :=   /- 37 approximates 37.6 -/
sorry

end percentage_of_ducks_among_non_heron_l48_48298


namespace trumpet_cost_l48_48983

variable (total_amount : ℝ) (book_cost : ℝ)

theorem trumpet_cost (h1 : total_amount = 151) (h2 : book_cost = 5.84) :
  (total_amount - book_cost = 145.16) :=
by
  sorry

end trumpet_cost_l48_48983


namespace circle_equation_center_xaxis_radius_2_l48_48736

theorem circle_equation_center_xaxis_radius_2 (a x y : ℝ) :
  (0:ℝ) < 2 ∧ (a - 1)^2 + 2^2 = 4 -> (x - 1)^2 + y^2 = 4 :=
by
  sorry

end circle_equation_center_xaxis_radius_2_l48_48736


namespace all_numbers_non_positive_l48_48768

theorem all_numbers_non_positive 
  (a : ℕ → ℝ) 
  (n : ℕ) 
  (h0 : a 0 = 0) 
  (hn : a n = 0) 
  (h : ∀ k, 1 ≤ k → k ≤ n - 1 → (a (k - 1) - 2 * a k + a (k + 1) ≥ 0)) : 
  ∀ k, 0 ≤ k → k ≤ n → a k ≤ 0 := 
by 
  sorry

end all_numbers_non_positive_l48_48768


namespace intersection_l48_48245

def A : Set ℝ := { x | -2 < x ∧ x < 3 }
def B : Set ℝ := { x | x > -1 }

theorem intersection (x : ℝ) : x ∈ (A ∩ B) ↔ -1 < x ∧ x < 3 := by
  sorry

end intersection_l48_48245


namespace abs_fraction_lt_one_l48_48583

theorem abs_fraction_lt_one (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) : 
  |(x - y) / (1 - x * y)| < 1 := 
sorry

end abs_fraction_lt_one_l48_48583


namespace juice_difference_proof_l48_48309

def barrel_initial_A := 10
def barrel_initial_B := 8
def transfer_amount := 3

def barrel_final_A := barrel_initial_A + transfer_amount
def barrel_final_B := barrel_initial_B - transfer_amount

def juice_difference := barrel_final_A - barrel_final_B

theorem juice_difference_proof : juice_difference = 8 := by
  sorry

end juice_difference_proof_l48_48309


namespace area_of_square_is_25_l48_48683

-- Define side length of the square
def sideLength : ℝ := 5

-- Define the area of the square
def area_of_square (side : ℝ) : ℝ := side * side

-- Prove the area of the square with side length 5 is 25 square meters
theorem area_of_square_is_25 : area_of_square sideLength = 25 := by
  sorry

end area_of_square_is_25_l48_48683


namespace distance_from_origin_to_line_l48_48832

theorem distance_from_origin_to_line : 
  let A := 1
  let B := 2
  let C := -5
  let x_0 := 0
  let y_0 := 0
  let distance := |A * x_0 + B * y_0 + C| / (Real.sqrt (A ^ 2 + B ^ 2))
  distance = Real.sqrt 5 :=
by
  sorry

end distance_from_origin_to_line_l48_48832


namespace find_k_l48_48686

theorem find_k
  (angle_C : ℝ)
  (AB : ℝ × ℝ)
  (AC : ℝ × ℝ)
  (h1 : angle_C = 90)
  (h2 : AB = (k, 1))
  (h3 : AC = (2, 3)) :
  k = 5 := by
  sorry

end find_k_l48_48686


namespace find_a_plus_b_l48_48692

theorem find_a_plus_b (a b : ℕ) 
  (h1 : 2^(2 * a) + 2^b + 5 = k^2) : a + b = 4 ∨ a + b = 5 :=
sorry

end find_a_plus_b_l48_48692


namespace sqrt_eq_two_or_neg_two_l48_48379

theorem sqrt_eq_two_or_neg_two (x : ℝ) (h : x^2 = 4) : x = 2 ∨ x = -2 :=
sorry

end sqrt_eq_two_or_neg_two_l48_48379


namespace average_score_of_juniors_l48_48355

theorem average_score_of_juniors :
  ∀ (N : ℕ) (junior_percent senior_percent overall_avg senior_avg : ℚ),
  junior_percent = 0.20 →
  senior_percent = 0.80 →
  overall_avg = 86 →
  senior_avg = 85 →
  (N * overall_avg - (N * senior_percent * senior_avg)) / (N * junior_percent) = 90 := 
by
  intros N junior_percent senior_percent overall_avg senior_avg
  intros h1 h2 h3 h4
  sorry

end average_score_of_juniors_l48_48355


namespace trigonometric_simplification_l48_48982

theorem trigonometric_simplification (α : ℝ) :
  (2 * Real.cos α ^ 2 - 1) /
  (2 * Real.tan (π / 4 - α) * Real.sin (π / 4 + α) ^ 2) = 1 :=
sorry

end trigonometric_simplification_l48_48982


namespace intersection_A_B_l48_48025

def is_log2 (y x : ℝ) : Prop := y = Real.log x / Real.log 2

def set_A (y : ℝ) : Set ℝ := { x | ∃ y, is_log2 y x}
def set_B : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }

theorem intersection_A_B : (set_A 1) ∩ set_B = { x | 0 < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_A_B_l48_48025


namespace star_points_number_l48_48984

-- Let n be the number of points in the star
def n : ℕ := sorry

-- Let A and B be the angles at the star points, with the condition that A_i = B_i - 20
def A (i : ℕ) : ℝ := sorry
def B (i : ℕ) : ℝ := sorry

-- Condition: For all i, A_i = B_i - 20
axiom angle_condition : ∀ i, A i = B i - 20

-- Total sum of angle differences equal to 360 degrees
axiom angle_sum_condition : n * 20 = 360

-- Theorem to prove
theorem star_points_number : n = 18 := by
  sorry

end star_points_number_l48_48984


namespace geometric_sequence_common_ratio_l48_48135

variables {a_n : ℕ → ℝ} {S_n q : ℝ}

axiom a1_eq : a_n 1 = 2
axiom an_eq : ∀ n, a_n n = if n > 0 then 2 * q^(n-1) else 0
axiom Sn_eq : ∀ n, a_n n = -64 → S_n = -42 → q = -2

theorem geometric_sequence_common_ratio (q : ℝ) :
  (∀ n, a_n n = if n > 0 then 2 * q^(n-1) else 0) →
  a_n 1 = 2 →
  (∀ n, a_n n = -64 → S_n = -42 → q = -2) :=
by intros _ _ _; sorry

end geometric_sequence_common_ratio_l48_48135


namespace find_number_l48_48497

theorem find_number (x : ℝ) (h : 2994 / x = 173) : x = 17.3 := 
sorry

end find_number_l48_48497


namespace actual_area_of_park_l48_48846

-- Definitions of given conditions
def map_scale : ℕ := 250 -- scale: 1 inch = 250 miles
def map_length : ℕ := 6 -- length on map in inches
def map_width : ℕ := 4 -- width on map in inches

-- Definition of actual lengths
def actual_length : ℕ := map_length * map_scale -- actual length in miles
def actual_width : ℕ := map_width * map_scale -- actual width in miles

-- Theorem to prove the actual area
theorem actual_area_of_park : actual_length * actual_width = 1500000 := by
  -- By the conditions provided, the actual length and width in miles can be calculated directly:
  -- actual_length = 6 * 250 = 1500
  -- actual_width = 4 * 250 = 1000
  -- actual_area = 1500 * 1000 = 1500000
  sorry

end actual_area_of_park_l48_48846


namespace relationship_between_x_t_G_D_and_x_l48_48988

-- Definitions
variables {G D : ℝ → ℝ}
variables {t : ℝ}
noncomputable def number_of_boys (x : ℝ) : ℝ := 9000 / x
noncomputable def total_population (x : ℝ) (x_t : ℝ) : Prop := x_t = 15000 / x

-- The proof problem
theorem relationship_between_x_t_G_D_and_x
  (G D : ℝ → ℝ)
  (x : ℝ) (t : ℝ) (x_t : ℝ)
  (h1 : 90 = x / 100 * number_of_boys x)
  (h2 : 0.60 * x_t = number_of_boys x)
  (h3 : 0.40 * x_t > 0)
  (h4 : true) :       -- Placeholder for some condition not used directly
  total_population x x_t :=
by
  -- Proof would go here
  sorry

end relationship_between_x_t_G_D_and_x_l48_48988


namespace lamp_turn_off_ways_l48_48210

theorem lamp_turn_off_ways : 
  ∃ (ways : ℕ), ways = 10 ∧
  (∃ (n : ℕ) (m : ℕ), 
    n = 6 ∧  -- 6 lamps in a row
    m = 2 ∧  -- turn off 2 of them
    ways = Nat.choose (n - m + 1) m) := -- 2 adjacent lamps cannot be turned off
by
  -- Proof will be provided here.
  sorry

end lamp_turn_off_ways_l48_48210


namespace complement_of_M_in_U_is_14_l48_48115

def U : Set ℕ := {x | x < 5 ∧ x > 0}

def M : Set ℕ := {x | x^2 - 5 * x + 6 = 0}

theorem complement_of_M_in_U_is_14 : 
  {x | x ∈ U ∧ x ∉ M} = {1, 4} :=
by
  sorry

end complement_of_M_in_U_is_14_l48_48115


namespace union_A_B_equiv_l48_48347

def A : Set ℝ := {x : ℝ | x > 2}
def B : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}

theorem union_A_B_equiv : A ∪ B = {x : ℝ | x ≥ 1} :=
by
  sorry

end union_A_B_equiv_l48_48347


namespace hoseok_position_l48_48509

variable (total_people : ℕ) (pos_from_back : ℕ)

theorem hoseok_position (h₁ : total_people = 9) (h₂ : pos_from_back = 5) :
  (total_people - pos_from_back + 1) = 5 :=
by
  sorry

end hoseok_position_l48_48509


namespace contradiction_assumption_l48_48837

-- Proposition P: "Among a, b, c, d, at least one is negative"
def P (a b c d : ℝ) : Prop :=
  a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0

-- Correct assumption when using contradiction: all are non-negative
def notP (a b c d : ℝ) : Prop :=
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0

-- Proof problem statement: assuming notP leads to contradiction to prove P
theorem contradiction_assumption (a b c d : ℝ) (h : ¬ P a b c d) : notP a b c d :=
by
  sorry

end contradiction_assumption_l48_48837


namespace right_triangle_set_l48_48620

theorem right_triangle_set:
  (1^2 + 2^2 = (Real.sqrt 5)^2) ∧
  ¬ (6^2 + 8^2 = 9^2) ∧
  ¬ ((Real.sqrt 3)^2 + (Real.sqrt 2)^2 = 5^2) ∧
  ¬ ((3^2)^2 + (4^2)^2 = (5^2)^2)  :=
by
  sorry

end right_triangle_set_l48_48620


namespace box_volume_l48_48835

theorem box_volume (initial_length initial_width cut_length : ℕ)
  (length_condition : initial_length = 13) (width_condition : initial_width = 9)
  (cut_condition : cut_length = 2) : 
  (initial_length - 2 * cut_length) * (initial_width - 2 * cut_length) * cut_length = 90 := 
by
  sorry

end box_volume_l48_48835


namespace max_value_of_a_l48_48492

theorem max_value_of_a 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^3 - a * x) 
  (h2 : ∀ x ≥ 1, ∀ y ≥ 1, x ≤ y → f x ≤ f y) : 
  a ≤ 3 :=
sorry

end max_value_of_a_l48_48492


namespace hemisphere_surface_area_l48_48856

theorem hemisphere_surface_area (r : ℝ) (h : r = 10) : 
  (4 * Real.pi * r^2) / 2 + (Real.pi * r^2) = 300 * Real.pi := by
  sorry

end hemisphere_surface_area_l48_48856


namespace famous_figures_mathematicians_l48_48236

-- List of figures encoded as integers for simplicity
def Bill_Gates := 1
def Gauss := 2
def Liu_Xiang := 3
def Nobel := 4
def Chen_Jingrun := 5
def Chen_Xingshen := 6
def Gorky := 7
def Einstein := 8

-- Set of mathematicians encoded as a set of integers
def mathematicians : Set ℕ := {2, 5, 6}

-- Correct answer set
def correct_answer_set : Set ℕ := {2, 5, 6}

-- The statement to prove
theorem famous_figures_mathematicians:
  mathematicians = correct_answer_set :=
by sorry

end famous_figures_mathematicians_l48_48236


namespace find_number_of_students_l48_48750

theorem find_number_of_students (N : ℕ) (h1 : T = 80 * N) (h2 : (T - 350) / (N - 5) = 90) 
: N = 10 := 
by 
  -- Proof steps would go here. Omitted as per the instruction.
  sorry

end find_number_of_students_l48_48750


namespace sum_of_equal_numbers_l48_48639

theorem sum_of_equal_numbers (a b : ℝ) (h1 : (12 + 25 + 18 + a + b) / 5 = 20) (h2 : a = b) : a + b = 45 :=
sorry

end sum_of_equal_numbers_l48_48639


namespace binom_26_6_l48_48260

theorem binom_26_6 (h₁ : Nat.choose 25 5 = 53130) (h₂ : Nat.choose 25 6 = 177100) :
  Nat.choose 26 6 = 230230 :=
by
  sorry

end binom_26_6_l48_48260


namespace parallel_lines_condition_l48_48068

theorem parallel_lines_condition (a : ℝ) : 
  (a = -2) ↔ (∀ x y : ℝ, ax + 2 * y = 0 → y = 1 + x) := 
sorry

end parallel_lines_condition_l48_48068


namespace relationship_among_a_b_c_l48_48019

theorem relationship_among_a_b_c :
  let a := (1/6) ^ (1/2)
  let b := Real.log (1/3) / Real.log 6
  let c := Real.log (1/7) / Real.log (1/6)
  c > a ∧ a > b :=
by
  sorry

end relationship_among_a_b_c_l48_48019


namespace unique_solution_c_min_l48_48979

theorem unique_solution_c_min (x y : ℝ) (c : ℝ)
  (h1 : 2 * (x+7)^2 + (y-4)^2 = c)
  (h2 : (x+4)^2 + 2 * (y-7)^2 = c) :
  c = 6 :=
sorry

end unique_solution_c_min_l48_48979


namespace hours_per_day_l48_48360

theorem hours_per_day 
  (H : ℕ)
  (h1 : 6 * 8 * H = 48 * H)
  (h2 : 4 * 3 * 8 = 96)
  (h3 : (48 * H) / 75 = 96 / 30) : 
  H = 5 :=
by
  sorry

end hours_per_day_l48_48360


namespace smallest_m_plus_n_l48_48152

theorem smallest_m_plus_n (m n : ℕ) (hmn : m > n) (hid : (2012^m : ℕ) % 1000 = (2012^n) % 1000) : m + n = 104 :=
sorry

end smallest_m_plus_n_l48_48152


namespace youngest_brother_age_l48_48955

theorem youngest_brother_age (x : ℕ) (h : x + (x + 1) + (x + 2) = 96) : x = 31 :=
sorry

end youngest_brother_age_l48_48955


namespace sum_of_sequences_l48_48287

-- Define the sequences and their type
def seq1 : List ℕ := [2, 12, 22, 32, 42]
def seq2 : List ℕ := [10, 20, 30, 40, 50]

-- The property we wish to prove
theorem sum_of_sequences : seq1.sum + seq2.sum = 260 :=
by
  sorry

end sum_of_sequences_l48_48287


namespace negation_of_even_sum_l48_48556

variables (a b : Int)

def is_even (n : Int) : Prop := ∃ k : Int, n = 2 * k

theorem negation_of_even_sum (h : ¬(is_even a ∧ is_even b)) : ¬is_even (a + b) :=
sorry

end negation_of_even_sum_l48_48556


namespace Bing_max_games_l48_48858

/-- 
  Jia, Yi, and Bing play table tennis with the following rules: each game is played between two 
  people, and the loser gives way to the third person. If Jia played 10 games and Yi played 
  7 games, then Bing can play at most 13 games; and can win at most 10 games.
-/
theorem Bing_max_games 
  (games_played_Jia : ℕ)
  (games_played_Yi : ℕ)
  (games_played_Bing : ℕ)
  (games_won_Bing  : ℕ)
  (hJia : games_played_Jia = 10)
  (hYi : games_played_Yi = 7) :
  (games_played_Bing ≤ 13) ∧ (games_won_Bing ≤ 10) := 
sorry

end Bing_max_games_l48_48858


namespace coordinate_of_point_A_l48_48779

theorem coordinate_of_point_A (a b : ℝ) 
    (h1 : |b| = 3) 
    (h2 : |a| = 4) 
    (h3 : a > b) : 
    (a, b) = (4, 3) ∨ (a, b) = (4, -3) :=
by
    sorry

end coordinate_of_point_A_l48_48779


namespace new_student_weight_l48_48153

theorem new_student_weight (avg_weight : ℝ) (x : ℝ) :
  (avg_weight * 10 - 120) = ((avg_weight - 6) * 10 + x) → x = 60 :=
by
  intro h
  -- The proof would go here, but it's skipped.
  sorry

end new_student_weight_l48_48153


namespace sum_is_correct_l48_48368

theorem sum_is_correct (a b c d : ℤ) 
  (h : a + 1 = b + 2 ∧ b + 2 = c + 3 ∧ c + 3 = d + 4 ∧ d + 4 = a + b + c + d + 7) : 
  a + b + c + d = -6 := 
by 
  sorry

end sum_is_correct_l48_48368


namespace total_marbles_l48_48740

theorem total_marbles
  (R B Y : ℕ)  -- Red, Blue, and Yellow marbles as natural numbers
  (h_ratio : 2 * (R + B + Y) = 9 * Y)  -- The ratio condition translated
  (h_yellow : Y = 36)  -- The number of yellow marbles condition
  : R + B + Y = 81 :=  -- Statement that the total number of marbles is 81
sorry

end total_marbles_l48_48740


namespace effective_percentage_change_l48_48709

def original_price (P : ℝ) : ℝ := P
def annual_sale_discount (P : ℝ) : ℝ := 0.70 * P
def clearance_event_discount (P : ℝ) : ℝ := 0.80 * (annual_sale_discount P)
def sales_tax (P : ℝ) : ℝ := 1.10 * (clearance_event_discount P)

theorem effective_percentage_change (P : ℝ) :
  (sales_tax P) = 0.616 * P := by
  sorry

end effective_percentage_change_l48_48709


namespace solve_system_of_equations_l48_48081

theorem solve_system_of_equations (n : ℕ) (hn : n ≥ 3) (x : ℕ → ℝ) :
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n →
    x i ^ 3 = (x ((i % n) + 1) + x ((i % n) + 2) + 1)) →
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n →
    (x i = -1 ∨ x i = (1 + Real.sqrt 5) / 2 ∨ x i = (1 - Real.sqrt 5) / 2)) :=
sorry

end solve_system_of_equations_l48_48081


namespace determine_a_l48_48573

theorem determine_a :
  ∃ (a b c d : ℕ), 
  (18 ^ a) * (9 ^ (4 * a - 1)) * (27 ^ c) = (2 ^ 6) * (3 ^ b) * (7 ^ d) ∧ 
  a * c = 4 / (2 * b + d) ∧ 
  b^2 - 4 * a * c = d ∧ 
  a = 6 := 
by
  sorry

end determine_a_l48_48573


namespace mike_falls_short_l48_48362

theorem mike_falls_short : 
  ∀ (max_marks mike_score : ℕ) (pass_percentage : ℚ),
  pass_percentage = 0.30 → 
  max_marks = 800 → 
  mike_score = 212 → 
  (pass_percentage * max_marks - mike_score) = 28 :=
by
  intros max_marks mike_score pass_percentage h1 h2 h3
  sorry

end mike_falls_short_l48_48362


namespace vector_equation_l48_48387

noncomputable def vec_a : (ℝ × ℝ) := (1, -1)
noncomputable def vec_b : (ℝ × ℝ) := (2, 1)
noncomputable def vec_c : (ℝ × ℝ) := (-2, 1)

theorem vector_equation (x y : ℝ) 
  (h : vec_c = (x * vec_a.1 + y * vec_b.1, x * vec_a.2 + y * vec_b.2)) : 
  x - y = -1 := 
by { sorry }

end vector_equation_l48_48387


namespace series_sum_eq_4_over_9_l48_48488

noncomputable def sum_series : ℝ := ∑' (k : ℕ), (k+1) / 4^(k+1)

theorem series_sum_eq_4_over_9 : sum_series = 4 / 9 := 
sorry

end series_sum_eq_4_over_9_l48_48488


namespace train_passing_time_l48_48829

theorem train_passing_time (L : ℕ) (v_kmph : ℕ) (v_mps : ℕ) (time : ℕ)
  (h1 : L = 90)
  (h2 : v_kmph = 36)
  (h3 : v_mps = v_kmph * (1000 / 3600))
  (h4 : v_mps = 10)
  (h5 : time = L / v_mps) :
  time = 9 := by
  sorry

end train_passing_time_l48_48829


namespace surface_area_of_resulting_solid_l48_48870

-- Define the original cube dimensions
def original_cube_surface_area (s : ℕ) := 6 * s * s

-- Define the smaller cube dimensions to be cut
def small_cube_surface_area (s : ℕ) := 3 * s * s

-- Define the proof problem
theorem surface_area_of_resulting_solid :
  original_cube_surface_area 3 - small_cube_surface_area 1 - small_cube_surface_area 2 + (3 * 1 + 3 * 4) = 54 :=
by
  -- The actual proof is to be filled in here
  sorry

end surface_area_of_resulting_solid_l48_48870


namespace percent_decrease_is_80_l48_48338

-- Definitions based on the conditions
def original_price := 100
def sale_price := 20

-- Theorem statement to prove the percent decrease
theorem percent_decrease_is_80 :
  ((original_price - sale_price) / original_price * 100) = 80 := 
by
  sorry

end percent_decrease_is_80_l48_48338


namespace geometric_progression_condition_l48_48721

theorem geometric_progression_condition {b : ℕ → ℝ} (b1_ne_b2 : b 1 ≠ b 2) (h : ∀ n, b (n + 2) = b n / b (n + 1)) :
  (∀ n, b (n+1) / b n = b 2 / b 1) ↔ b 1 = b 2^3 := sorry

end geometric_progression_condition_l48_48721


namespace value_of_b_l48_48915

-- Define the variables and conditions
variables (a b c : ℚ)
axiom h1 : a + b + c = 150
axiom h2 : a + 10 = b - 3
axiom h3 : b - 3 = 4 * c 

-- The statement we want to prove
theorem value_of_b : b = 655 / 9 := 
by 
  -- We start with assumptions h1, h2, and h3
  sorry

end value_of_b_l48_48915


namespace infinite_power_tower_equation_l48_48000

noncomputable def infinite_power_tower (x : ℝ) : ℝ :=
  x ^ x ^ x ^ x ^ x -- continues infinitely

theorem infinite_power_tower_equation (x : ℝ) (h_pos : 0 < x) (h_eq : infinite_power_tower x = 2) : x = Real.sqrt 2 :=
  sorry

end infinite_power_tower_equation_l48_48000


namespace watermelon_heavier_than_pineapple_l48_48042

noncomputable def watermelon_weight : ℕ := 1 * 1000 + 300 -- Weight of one watermelon in grams
noncomputable def pineapple_weight : ℕ := 450 -- Weight of one pineapple in grams

theorem watermelon_heavier_than_pineapple :
    (4 * watermelon_weight = 5 * 1000 + 200) →
    (3 * watermelon_weight + 4 * pineapple_weight = 5 * 1000 + 700) →
    watermelon_weight - pineapple_weight = 850 :=
by
    intros h1 h2
    sorry

end watermelon_heavier_than_pineapple_l48_48042


namespace terminating_decimal_expansion_l48_48989

theorem terminating_decimal_expansion : (15 / 625 : ℝ) = 0.024 :=
by
  -- Lean requires a justification for non-trivial facts
  -- Provide math reasoning here if necessary
  sorry

end terminating_decimal_expansion_l48_48989


namespace vasya_numbers_l48_48330

-- Define the conditions
def sum_eq_product (x y : ℝ) : Prop := x + y = x * y
def product_eq_quotient (x y : ℝ) : Prop := x * y = x / y

-- State the proof problem
theorem vasya_numbers : 
  ∃ x y : ℝ, sum_eq_product x y ∧ product_eq_quotient x y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end vasya_numbers_l48_48330


namespace inequality_condition_l48_48234

theorem inequality_condition (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*a*x + a > 0) ↔ (0 < a ∧ a < 1) ∨ (False) := 
sorry

end inequality_condition_l48_48234


namespace find_m_l48_48415

variable (m : ℝ)

theorem find_m (h1 : 3 * (-7.5) - y = m) (h2 : -0.4 * (-7.5) + y = 3) : m = -22.5 :=
by
  sorry

end find_m_l48_48415


namespace percentage_of_second_division_l48_48015

theorem percentage_of_second_division
  (total_students : ℕ)
  (students_first_division : ℕ)
  (students_just_passed : ℕ)
  (h1: total_students = 300)
  (h2: students_first_division = 75)
  (h3: students_just_passed = 63) :
  (total_students - (students_first_division + students_just_passed)) * 100 / total_students = 54 := 
by
  -- Proof will be added later
  sorry

end percentage_of_second_division_l48_48015


namespace g_at_2_l48_48369

-- Assuming g is a function from ℝ to ℝ such that it satisfies the given condition.
def g : ℝ → ℝ := sorry

-- Condition of the problem
axiom g_condition : ∀ x : ℝ, g (2 ^ x) + x * g (2 ^ (-x)) = 2

-- The statement we want to prove
theorem g_at_2 : g (2) = 0 :=
by
  sorry

end g_at_2_l48_48369


namespace find_root_power_117_l48_48676

noncomputable def problem (a b c : ℝ) (x1 x2 : ℝ) :=
  (3 * a - b) / c * x1^2 + c * (3 * a + b) / (3 * a - b) = 0 ∧
  (3 * a - b) / c * x2^2 + c * (3 * a + b) / (3 * a - b) = 0 ∧
  x1 + x2 = 0

theorem find_root_power_117 (a b c : ℝ) (x1 x2 : ℝ) (h : problem a b c x1 x2) : 
  x1 ^ 117 + x2 ^ 117 = 0 :=
sorry

end find_root_power_117_l48_48676


namespace integral_equality_l48_48207

theorem integral_equality :
  ∫ x in (-1 : ℝ)..(1 : ℝ), (Real.tan x) ^ 11 + (Real.cos x) ^ 21
  = 2 * ∫ x in (0 : ℝ)..(1 : ℝ), (Real.cos x) ^ 21 :=
by
  sorry

end integral_equality_l48_48207


namespace ratio_of_ticket_prices_l48_48392

-- Given conditions
def num_adults := 400
def num_children := 200
def adult_ticket_price : ℕ := 32
def total_amount : ℕ := 16000
def child_ticket_price (C : ℕ) : Prop := num_adults * adult_ticket_price + num_children * C = total_amount

theorem ratio_of_ticket_prices (C : ℕ) (hC : child_ticket_price C) :
  adult_ticket_price / C = 2 :=
by
  sorry

end ratio_of_ticket_prices_l48_48392


namespace length_second_train_is_125_l48_48940

noncomputable def length_second_train (speed_faster speed_slower distance1 : ℕ) (time_minutes : ℝ) : ℝ :=
  let relative_speed_m_per_minute := (speed_faster - speed_slower) * 1000 / 60
  let total_distance_covered := relative_speed_m_per_minute * time_minutes
  total_distance_covered - distance1

theorem length_second_train_is_125 :
  length_second_train 50 40 125 1.5 = 125 :=
  by sorry

end length_second_train_is_125_l48_48940


namespace amc_inequality_l48_48792

variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)

theorem amc_inequality : (a / (b + c) + b / (a + c) + c / (a + b)) ≥ 3 / 2 :=
sorry

end amc_inequality_l48_48792


namespace slope_of_perpendicular_line_l48_48007

theorem slope_of_perpendicular_line (x y : ℝ) (h : 5 * x - 4 * y = 20) : 
  ∃ m : ℝ, m = -4 / 5 :=
sorry

end slope_of_perpendicular_line_l48_48007


namespace fraction_to_terminating_decimal_l48_48821

theorem fraction_to_terminating_decimal :
  (45 : ℚ) / 64 = (703125 : ℚ) / 1000000 := by
  sorry

end fraction_to_terminating_decimal_l48_48821


namespace existence_of_b_l48_48307

theorem existence_of_b's (n m : ℕ) (h1 : 1 < n) (h2 : 1 < m) 
  (a : Fin m → ℕ) (h3 : ∀ i, 0 < a i ∧ a i ≤ n^m) :
  ∃ b : Fin m → ℕ, (∀ i, 0 < b i ∧ b i ≤ n) ∧ (∀ i, a i + b i < n) :=
by
  sorry

end existence_of_b_l48_48307


namespace average_people_added_each_year_l48_48974

-- a) Identifying questions and conditions
-- Question: What is the average number of people added each year?
-- Conditions: In 2000, about 450,000 people lived in Maryville. In 2005, about 467,000 people lived in Maryville.

-- c) Mathematically equivalent proof problem
-- Mathematically equivalent proof problem: Prove that the average number of people added each year is 3400 given the conditions.

-- d) Lean 4 statement
theorem average_people_added_each_year :
  let population_2000 := 450000
  let population_2005 := 467000
  let years_passed := 2005 - 2000
  let total_increase := population_2005 - population_2000
  total_increase / years_passed = 3400 := by
    sorry

end average_people_added_each_year_l48_48974


namespace arithmetic_seq_term_298_eq_100_l48_48538

-- Define the arithmetic sequence
def arithmetic_seq (a₁ d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Define the specific sequence given in the problem
def a_n (n : ℕ) : ℕ := arithmetic_seq 1 3 n

-- State the theorem
theorem arithmetic_seq_term_298_eq_100 : a_n 100 = 298 :=
by
  -- Proof will be filled in
  sorry

end arithmetic_seq_term_298_eq_100_l48_48538


namespace trajectory_of_midpoint_l48_48281

theorem trajectory_of_midpoint {x y : ℝ} :
  (∃ Mx My : ℝ, (Mx + 3)^2 + My^2 = 4 ∧ (2 * x - 3 = Mx) ∧ (2 * y = My)) →
  x^2 + y^2 = 1 :=
by
  intro h
  sorry

end trajectory_of_midpoint_l48_48281


namespace cube_side_length_l48_48708

theorem cube_side_length (n : ℕ) (h1 : 6 * n^2 = (6 * n^3) / 3) : n = 3 :=
sorry

end cube_side_length_l48_48708


namespace sums_correct_l48_48403

theorem sums_correct (x : ℕ) (h : x + 2 * x = 48) : x = 16 :=
by
  sorry

end sums_correct_l48_48403


namespace binary_to_decimal_and_octal_l48_48973

theorem binary_to_decimal_and_octal (binary_input : Nat) (h : binary_input = 0b101101110) :
    binary_input == 366 ∧ (366 : Nat) == 0o66 :=
by
  sorry

end binary_to_decimal_and_octal_l48_48973


namespace probability_segments_length_l48_48437

theorem probability_segments_length (x y : ℝ) : 
    80 ≥ x ∧ x ≥ 20 ∧ 80 ≥ y ∧ y ≥ 20 ∧ 80 ≥ 80 - x - y ∧ 80 - x - y ≥ 20 → 
    (∃ (s : ℝ), s = (200 / 3200) ∧ s = (1 / 16)) :=
by
  intros h
  sorry

end probability_segments_length_l48_48437


namespace find_weight_of_first_new_player_l48_48993

variable (weight_of_first_new_player : ℕ)
variable (weight_of_second_new_player : ℕ := 60) -- Second new player's weight is a given constant
variable (num_of_original_players : ℕ := 7)
variable (avg_weight_of_original_players : ℕ := 121)
variable (new_avg_weight : ℕ := 113)
variable (num_of_new_players : ℕ := 2)

def total_weight_of_original_players : ℕ := 
  num_of_original_players * avg_weight_of_original_players

def total_weight_of_new_players : ℕ :=
  num_of_new_players * new_avg_weight

def combined_weight_without_first_new_player : ℕ := 
  total_weight_of_original_players + weight_of_second_new_player

def weight_of_first_new_player_proven : Prop :=
  total_weight_of_new_players - combined_weight_without_first_new_player = weight_of_first_new_player

theorem find_weight_of_first_new_player : weight_of_first_new_player = 110 :=
by 
  sorry

end find_weight_of_first_new_player_l48_48993


namespace ratio_of_number_halving_l48_48839

theorem ratio_of_number_halving (x y : ℕ) (h1 : y = x / 2) (h2 : y = 9) : x / y = 2 :=
by
  sorry

end ratio_of_number_halving_l48_48839


namespace swimming_speed_still_water_l48_48140

theorem swimming_speed_still_water 
  (v t : ℝ) 
  (h1 : 3 = (v + 3) * t / (v - 3)) 
  (h2 : t ≠ 0) :
  v = 9 :=
by
  sorry

end swimming_speed_still_water_l48_48140


namespace problem1_problem2_l48_48691

theorem problem1 : (-(3 / 4) - (5 / 8) + (9 / 12)) * (-24) = 15 := by
  sorry

theorem problem2 : (-1 ^ 6 + |(-2) ^ 3 - 10| - (-3) / (-1) ^ 2023) = 14 := by
  sorry

end problem1_problem2_l48_48691


namespace exponent_problem_proof_l48_48173

theorem exponent_problem_proof :
  3 * 3^4 - 27^60 / 27^58 = -486 :=
by
  sorry

end exponent_problem_proof_l48_48173


namespace drum_capacity_ratio_l48_48753

variable {C_X C_Y : ℝ}

theorem drum_capacity_ratio (h1 : C_X / 2 + C_Y / 2 = 3 * C_Y / 4) : C_Y / C_X = 2 :=
by
  have h2: C_X / 2 = C_Y / 4 := by
    sorry
  have h3: C_X = C_Y / 2 := by
    sorry
  rw [h3]
  have h4: C_Y / (C_Y / 2) = 2 := by
    sorry
  exact h4

end drum_capacity_ratio_l48_48753


namespace number_of_rectangles_required_l48_48845

theorem number_of_rectangles_required
  (width : ℝ) (area : ℝ) (total_length : ℝ) (length : ℝ)
  (H1 : width = 42) (H2 : area = 1638) (H3 : total_length = 390) (H4 : length = area / width)
  : (total_length / length) = 10 := 
sorry

end number_of_rectangles_required_l48_48845


namespace contractor_engaged_days_l48_48496

theorem contractor_engaged_days
  (earnings_per_day : ℤ)
  (fine_per_day : ℤ)
  (total_earnings : ℤ)
  (absent_days : ℤ)
  (days_worked : ℤ) 
  (h1 : earnings_per_day = 25)
  (h2 : fine_per_day = 15 / 2)
  (h3 : total_earnings = 620)
  (h4 : absent_days = 4)
  (h5 : total_earnings = earnings_per_day * days_worked - fine_per_day * absent_days) :
  days_worked = 26 := 
by {
  -- Proof goes here
  sorry
}

end contractor_engaged_days_l48_48496


namespace cos_product_identity_l48_48370

noncomputable def L : ℝ := 3.418 * (Real.cos (2 * Real.pi / 31)) *
                               (Real.cos (4 * Real.pi / 31)) *
                               (Real.cos (8 * Real.pi / 31)) *
                               (Real.cos (16 * Real.pi / 31)) *
                               (Real.cos (32 * Real.pi / 31))

theorem cos_product_identity : L = 1 / 32 := by
  sorry

end cos_product_identity_l48_48370


namespace inequality_l48_48075

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.log 3 / Real.log (Real.pi)
noncomputable def c : ℝ := Real.log 0.5 / Real.log 2

theorem inequality (h1: a = Real.sqrt 2) (h2: b = Real.log 3 / Real.log Real.pi) (h3: c = Real.log 0.5 / Real.log 2) : a > b ∧ b > c := 
by 
  sorry

end inequality_l48_48075


namespace linda_total_profit_is_50_l48_48052

def total_loaves : ℕ := 60
def loaves_sold_morning (total_loaves : ℕ) : ℕ := total_loaves / 3
def loaves_sold_afternoon (loaves_left_morning : ℕ) : ℕ := loaves_left_morning / 2
def loaves_sold_evening (loaves_left_afternoon : ℕ) : ℕ := loaves_left_afternoon

def price_per_loaf_morning : ℕ := 3
def price_per_loaf_afternoon : ℕ := 150 / 100 -- Representing $1.50 as 150 cents to use integer arithmetic
def price_per_loaf_evening : ℕ := 1

def cost_per_loaf : ℕ := 1

def calculate_profit (total_loaves loaves_sold_morning loaves_sold_afternoon loaves_sold_evening price_per_loaf_morning price_per_loaf_afternoon price_per_loaf_evening cost_per_loaf : ℕ) : ℕ := 
  let revenue_morning := loaves_sold_morning * price_per_loaf_morning
  let loaves_left_morning := total_loaves - loaves_sold_morning
  let revenue_afternoon := loaves_sold_afternoon * price_per_loaf_afternoon
  let loaves_left_afternoon := loaves_left_morning - loaves_sold_afternoon
  let revenue_evening := loaves_sold_evening * price_per_loaf_evening
  let total_revenue := revenue_morning + revenue_afternoon + revenue_evening
  let total_cost := total_loaves * cost_per_loaf
  total_revenue - total_cost

theorem linda_total_profit_is_50 : calculate_profit total_loaves (loaves_sold_morning total_loaves) (loaves_sold_afternoon (total_loaves - loaves_sold_morning total_loaves)) (total_loaves - loaves_sold_morning total_loaves - loaves_sold_afternoon (total_loaves - loaves_sold_morning total_loaves)) price_per_loaf_morning price_per_loaf_afternoon price_per_loaf_evening cost_per_loaf = 50 := 
  by 
    sorry

end linda_total_profit_is_50_l48_48052


namespace abs_eq_abs_implies_l48_48003

theorem abs_eq_abs_implies (x : ℝ) (h : |x - 3| = |x + 1|) : x = 1 := 
sorry

end abs_eq_abs_implies_l48_48003


namespace total_distance_traveled_l48_48769

def speed := 60  -- Jace drives 60 miles per hour
def first_leg_time := 4  -- Jace drives for 4 hours straight
def break_time := 0.5  -- Jace takes a 30-minute break (0.5 hours)
def second_leg_time := 9  -- Jace drives for another 9 hours straight

def distance (speed : ℕ) (time : ℕ) : ℕ := speed * time  -- Distance formula

theorem total_distance_traveled : 
  distance speed first_leg_time + distance speed second_leg_time = 780 := by
-- Sorry allows us to skip the proof, since only the statement is required.
sorry

end total_distance_traveled_l48_48769


namespace ratios_of_PQR_and_XYZ_l48_48054

-- Define triangle sides
def sides_PQR : ℕ × ℕ × ℕ := (7, 24, 25)
def sides_XYZ : ℕ × ℕ × ℕ := (9, 40, 41)

-- Perimeter calculation functions
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- Area calculation functions for right triangles
def area (a b : ℕ) : ℕ := (a * b) / 2

-- Required proof statement
theorem ratios_of_PQR_and_XYZ :
  let (a₁, b₁, c₁) := sides_PQR
  let (a₂, b₂, c₂) := sides_XYZ
  area a₁ b₁ * 15 = 7 * area a₂ b₂ ∧ perimeter a₁ b₁ c₁ * 45 = 28 * perimeter a₂ b₂ c₂ :=
sorry

end ratios_of_PQR_and_XYZ_l48_48054


namespace simone_finishes_task_at_1115_l48_48926

noncomputable def simone_finish_time
  (start_time: Nat) -- Start time in minutes past midnight
  (task_1_duration: Nat) -- Duration of the first task in minutes
  (task_2_duration: Nat) -- Duration of the second task in minutes
  (break_duration: Nat) -- Duration of the break in minutes
  (task_3_duration: Nat) -- Duration of the third task in minutes
  (end_time: Nat) := -- End time to be proven
  start_time + task_1_duration + task_2_duration + break_duration + task_3_duration = end_time

theorem simone_finishes_task_at_1115 :
  simone_finish_time 480 45 45 15 90 675 := -- 480 minutes is 8:00 AM; 675 minutes is 11:15 AM
  by sorry

end simone_finishes_task_at_1115_l48_48926


namespace problem_solution_l48_48238

theorem problem_solution (x : ℝ) (h : x^2 - 8*x - 3 = 0) : (x - 1) * (x - 3) * (x - 5) * (x - 7) = 180 :=
by sorry

end problem_solution_l48_48238


namespace probability_rain_weekend_l48_48824

theorem probability_rain_weekend :
  let p_rain_saturday := 0.30
  let p_rain_sunday := 0.60
  let p_rain_sunday_given_rain_saturday := 0.40
  let p_no_rain_saturday := 1 - p_rain_saturday
  let p_no_rain_sunday_given_no_rain_saturday := 1 - p_rain_sunday
  let p_no_rain_both_days := p_no_rain_saturday * p_no_rain_sunday_given_no_rain_saturday
  let p_rain_sunday_given_rain_saturday := 1 - p_rain_sunday_given_rain_saturday
  let p_no_rain_sunday_given_rain_saturday := p_rain_saturday * p_rain_sunday_given_rain_saturday
  let p_no_rain_all_scenarios := p_no_rain_both_days + p_no_rain_sunday_given_rain_saturday
  let p_rain_weekend := 1 - p_no_rain_all_scenarios
  p_rain_weekend = 0.54 :=
sorry

end probability_rain_weekend_l48_48824


namespace triangle_equilateral_if_condition_l48_48950

-- Define the given conditions
variables {A B C : ℝ} -- Angles
variables {a b c : ℝ} -- Opposite sides

-- Assume the condition that a/ cos(A) = b/ cos(B) = c/ cos(C)
def triangle_condition (A B C a b c : ℝ) : Prop :=
  a / Real.cos A = b / Real.cos B ∧ b / Real.cos B = c / Real.cos C

-- The theorem to prove under these conditions
theorem triangle_equilateral_if_condition (A B C a b c : ℝ) 
  (h : triangle_condition A B C a b c) : 
  A = B ∧ B = C :=
sorry

end triangle_equilateral_if_condition_l48_48950


namespace mindy_mork_earnings_ratio_l48_48169

theorem mindy_mork_earnings_ratio (M K : ℝ) (h1 : 0.20 * M + 0.30 * K = 0.225 * (M + K)) : M / K = 3 :=
by
  sorry

end mindy_mork_earnings_ratio_l48_48169


namespace unique_zero_point_mn_l48_48373

noncomputable def f (a : ℝ) (x : ℝ) := a * (x^2 + 2 / x) - Real.log x

theorem unique_zero_point_mn (a : ℝ) (m n x₀ : ℝ) (hmn : m + 1 = n) (a_pos : 0 < a) (f_zero : f a x₀ = 0) (x0_in_range : m < x₀ ∧ x₀ < n) : m + n = 5 := by
  sorry

end unique_zero_point_mn_l48_48373


namespace farmer_field_m_value_l48_48903

theorem farmer_field_m_value (m : ℝ) 
    (h_length : ∀ m, m > -4 → 2 * m + 9 > 0) 
    (h_breadth : ∀ m, m > -4 → m - 4 > 0)
    (h_area : (2 * m + 9) * (m - 4) = 88) : 
    m = 7.5 :=
by
  sorry

end farmer_field_m_value_l48_48903


namespace percentage_reduction_price_increase_l48_48780

-- Part 1: Prove the percentage reduction 
theorem percentage_reduction (P0 P1 : ℝ) (r : ℝ) (hp0 : P0 = 50) (hp1 : P1 = 32) :
  P1 = P0 * (1 - r) ^ 2 → r = 1 - 2 * Real.sqrt 2 / 5 :=
by
  intro h
  rw [hp0, hp1] at h
  sorry

-- Part 2: Prove the required price increase
theorem price_increase (G p0 V0 y : ℝ) (hp0 : p0 = 10) (hV0 : V0 = 500) (hG : G = 6000) (hy_range : 0 < y ∧ y ≤ 8):
  G = (p0 + y) * (V0 - 20 * y) → y = 5 :=
by
  intro h
  rw [hp0, hV0, hG] at h
  sorry

end percentage_reduction_price_increase_l48_48780


namespace octahedron_tetrahedron_volume_ratio_l48_48999

theorem octahedron_tetrahedron_volume_ratio (s : ℝ) :
  let V_T := (s^3 * Real.sqrt 2) / 12
  let a := s / 2
  let V_O := (a^3 * Real.sqrt 2) / 3
  V_O / V_T = 1 / 2 :=
by
  sorry

end octahedron_tetrahedron_volume_ratio_l48_48999


namespace intersection_of_M_and_N_l48_48481

noncomputable def M : Set ℝ := { y : ℝ | ∃ x : ℝ, y = x^2 }
noncomputable def N : Set ℝ := { y : ℝ | ∃ x : ℝ, x^2 + y^2 = 1 }

theorem intersection_of_M_and_N : M ∩ N = { y : ℝ | 0 ≤ y ∧ y ≤ 1 } :=
by
  sorry

end intersection_of_M_and_N_l48_48481


namespace find_f_2_l48_48349

noncomputable def f (a b x : ℝ) : ℝ := a * x^5 + b * x^3 - x + 2

theorem find_f_2 (a b : ℝ)
  (h : f a b (-2) = 5) : f a b 2 = -1 :=
by 
  sorry

end find_f_2_l48_48349


namespace part_a_part_b_l48_48537

theorem part_a (a : ℕ) : ¬ (∃ k : ℕ, k^2 = ( ((a^2 - 3)^3 + 1)^a - 1)) :=
sorry

theorem part_b (a : ℕ) : ¬ (∃ k : ℕ, k^2 = ( ((a^2 - 3)^3 + 1)^(a + 1) - 1)) :=
sorry

end part_a_part_b_l48_48537


namespace max_value_of_xyz_l48_48457

theorem max_value_of_xyz (x y z : ℝ) (h : x + 3 * y + z = 5) : xy + xz + yz ≤ 125 / 4 := 
sorry

end max_value_of_xyz_l48_48457


namespace john_has_dollars_left_l48_48480

-- Definitions based on the conditions
def john_savings_octal : ℕ := 5273
def rental_car_cost_decimal : ℕ := 1500

-- Define the function to convert octal to decimal
def octal_to_decimal (n : ℕ) : ℕ := -- Conversion logic
sorry

-- Statements for the conversion and subtraction
def john_savings_decimal : ℕ := octal_to_decimal john_savings_octal
def amount_left_for_gas_and_accommodations : ℕ :=
  john_savings_decimal - rental_car_cost_decimal

-- Theorem statement equivalent to the correct answer
theorem john_has_dollars_left :
  amount_left_for_gas_and_accommodations = 1247 :=
by sorry

end john_has_dollars_left_l48_48480


namespace complement_of_B_in_A_l48_48300

def complement (A B : Set Int) := { x ∈ A | x ∉ B }

theorem complement_of_B_in_A (A B : Set Int) (a : Int) (h1 : A = {2, 3, 4}) (h2 : B = {a + 2, a}) (h3 : A ∩ B = B)
: complement A B = {3} :=
  sorry

end complement_of_B_in_A_l48_48300


namespace general_term_defines_sequence_l48_48094

/-- Sequence definition -/
def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 2 ∧ ∀ n, a (n + 1) = (2 * a n + 6) / (a n + 1)

/-- General term formula -/
def general_term (n : ℕ) : ℚ :=
  (3 * 4 ^ n + 2 * (-1) ^ n) / (4 ^ n - (-1) ^ n)

/-- Theorem stating that the general term formula defines the sequence -/
theorem general_term_defines_sequence : ∀ (a : ℕ → ℚ), seq a → ∀ n, a n = general_term n :=
by
  intros a h_seq n
  sorry

end general_term_defines_sequence_l48_48094


namespace remainder_zero_l48_48029

theorem remainder_zero {n : ℕ} (h : n > 0) : 
  (2013^n - 1803^n - 1781^n + 1774^n) % 203 = 0 :=
by {
  sorry
}

end remainder_zero_l48_48029


namespace total_water_bottles_needed_l48_48064

def number_of_people : ℕ := 4
def travel_time_one_way : ℕ := 8
def number_of_way : ℕ := 2
def water_consumption_per_hour : ℚ := 1 / 2

theorem total_water_bottles_needed : (number_of_people * (travel_time_one_way * number_of_way) * water_consumption_per_hour) = 32 := by
  sorry

end total_water_bottles_needed_l48_48064


namespace correct_statement_l48_48799

theorem correct_statement :
  (∃ (A : Prop), A = (2 * x^3 - 4 * x - 3 ≠ 3)) ∧
  (∃ (B : Prop), B = ((2 + 3) ≠ 6)) ∧
  (∃ (C : Prop), C = (-4 * x^2 * y = -4)) ∧
  (∃ (D : Prop), D = (1 = 1 ∧ 1 = 1 / 8)) →
  (C) :=
by sorry

end correct_statement_l48_48799


namespace student_failed_by_l48_48034

theorem student_failed_by :
  ∀ (total_marks obtained_marks passing_percentage : ℕ),
  total_marks = 700 →
  obtained_marks = 175 →
  passing_percentage = 33 →
  (passing_percentage * total_marks) / 100 - obtained_marks = 56 :=
by
  intros total_marks obtained_marks passing_percentage h1 h2 h3
  sorry

end student_failed_by_l48_48034


namespace find_x_solutions_l48_48948

theorem find_x_solutions :
  ∀ {x : ℝ}, (x = (1/x) + (-x)^2 + 3) → (x = -1 ∨ x = 1) :=
by
  sorry

end find_x_solutions_l48_48948


namespace exists_pairwise_coprime_product_of_two_consecutive_integers_l48_48994

theorem exists_pairwise_coprime_product_of_two_consecutive_integers (n : ℕ) (h : 0 < n) :
  ∃ (a : Fin n → ℕ), (∀ i, 2 ≤ a i) ∧ (Pairwise (IsCoprime on fun i => a i)) ∧ (∃ k : ℕ, (Finset.univ.prod a) - 1 = k * (k + 1)) := 
sorry

end exists_pairwise_coprime_product_of_two_consecutive_integers_l48_48994


namespace students_chemistry_or_physics_not_both_l48_48318

variables (total_chemistry total_both total_physics_only : ℕ)

theorem students_chemistry_or_physics_not_both
  (h1 : total_chemistry = 30)
  (h2 : total_both = 15)
  (h3 : total_physics_only = 18) :
  total_chemistry - total_both + total_physics_only = 33 :=
by
  sorry

end students_chemistry_or_physics_not_both_l48_48318


namespace inequality_one_inequality_system_l48_48378

theorem inequality_one (x : ℝ) : 2 * x + 3 ≤ 5 * x ↔ x ≥ 1 := sorry

theorem inequality_system (x : ℝ) : 
  (5 * x - 1 ≤ 3 * (x + 1)) ∧ 
  ((2 * x - 1) / 2 - (5 * x - 1) / 4 < 1) ↔ 
  (-5 < x ∧ x ≤ 2) := sorry

end inequality_one_inequality_system_l48_48378


namespace pumps_time_to_empty_pool_l48_48274

theorem pumps_time_to_empty_pool :
  (1 / (1 / 6 + 1 / 9) * 60) = 216 :=
by
  norm_num
  sorry

end pumps_time_to_empty_pool_l48_48274


namespace curve_intersects_every_plane_l48_48603

theorem curve_intersects_every_plane (A B C D : ℝ) (h : A ≠ 0 ∨ B ≠ 0 ∨ C ≠ 0) :
  ∃ t : ℝ, A * t + B * t^3 + C * t^5 + D = 0 :=
by
  sorry

end curve_intersects_every_plane_l48_48603


namespace ellipse_and_line_properties_l48_48678

theorem ellipse_and_line_properties :
  (∃ a b : ℝ, a > b ∧ b > 0 ∧ a * a = 4 ∧ b * b = 3 ∧
  ∀ x y : ℝ, (x, y) = (1, 3/2) → x^2 / a^2 + y^2 / b^2 = 1) ∧
  (∃ k : ℝ, k = 1 / 2 ∧ ∀ x y : ℝ, (x, y) = (2, 1) →
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧
  (x1 - 2) * (x2 - 2) + (k * (x1 - 2) + 1 - 1) * (k * (x2 - 2) + 1 - 1) = 5 / 4) :=
sorry

end ellipse_and_line_properties_l48_48678


namespace goldfish_below_surface_l48_48618

theorem goldfish_below_surface (Toby_counts_at_surface : ℕ) (percentage_at_surface : ℝ) (total_goldfish : ℕ) (below_surface : ℕ) :
    (Toby_counts_at_surface = 15 ∧ percentage_at_surface = 0.25 ∧ Toby_counts_at_surface = percentage_at_surface * total_goldfish ∧ below_surface = total_goldfish - Toby_counts_at_surface) →
    below_surface = 45 :=
by
  sorry

end goldfish_below_surface_l48_48618


namespace cylindrical_coordinates_cone_shape_l48_48400

def cylindrical_coordinates := Type

def shape_description (r θ z : ℝ) : Prop :=
θ = 2 * z

theorem cylindrical_coordinates_cone_shape (r θ z : ℝ) :
  shape_description r θ z → θ = 2 * z → Prop := sorry

end cylindrical_coordinates_cone_shape_l48_48400


namespace teddy_has_8_cats_l48_48124

theorem teddy_has_8_cats (dogs_teddy : ℕ) (cats_teddy : ℕ) (dogs_total : ℕ) (pets_total : ℕ)
  (h1 : dogs_teddy = 7)
  (h2 : dogs_total = dogs_teddy + (dogs_teddy + 9) + (dogs_teddy - 5))
  (h3 : pets_total = dogs_total + cats_teddy + (cats_teddy + 13))
  (h4 : pets_total = 54) :
  cats_teddy = 8 := by
  sorry

end teddy_has_8_cats_l48_48124


namespace circle_center_sum_l48_48627

theorem circle_center_sum (x y : ℝ) (h : (x - 2)^2 + (y + 1)^2 = 15) : x + y = 1 :=
sorry

end circle_center_sum_l48_48627


namespace polynomial_sum_l48_48813

theorem polynomial_sum :
  let f := (x^3 + 9*x^2 + 26*x + 24) 
  let g := (x + 3)
  let A := 1
  let B := 6
  let C := 8
  let D := -3
  (y = f/g) → (A + B + C + D = 12) :=
by 
  sorry

end polynomial_sum_l48_48813


namespace solve_m_n_l48_48305

theorem solve_m_n (m n : ℤ) :
  (m * 1 + n * 1 = 6) ∧ (m * 2 + n * -1 = 6) → (m = 4) ∧ (n = 2) := by
  sorry

end solve_m_n_l48_48305


namespace Isabel_subtasks_remaining_l48_48643

-- Definition of the known quantities
def Total_problems : ℕ := 72
def Completed_problems : ℕ := 32
def Subtasks_per_problem : ℕ := 5

-- Definition of the calculations
def Total_subtasks : ℕ := Total_problems * Subtasks_per_problem
def Completed_subtasks : ℕ := Completed_problems * Subtasks_per_problem
def Remaining_subtasks : ℕ := Total_subtasks - Completed_subtasks

-- The theorem we need to prove
theorem Isabel_subtasks_remaining : Remaining_subtasks = 200 := by
  -- Proof would go here, but we'll use sorry to indicate it's omitted
  sorry

end Isabel_subtasks_remaining_l48_48643


namespace local_minimum_at_2_l48_48434

noncomputable def f (x : ℝ) : ℝ := (2 / x) + Real.log x

theorem local_minimum_at_2 : ∃ δ > 0, ∀ y, abs (y - 2) < δ → f y ≥ f 2 := by
  sorry

end local_minimum_at_2_l48_48434


namespace maximum_value_expression_l48_48724

theorem maximum_value_expression (a b : ℝ) (h : a^2 + b^2 = 9) : 
  ∃ x, x = 5 ∧ ∀ y, y = ab - b + a → y ≤ x :=
by
  sorry

end maximum_value_expression_l48_48724


namespace initial_number_of_students_l48_48666

/-- 
Theorem: If the average mark of the students of a class in an exam is 90, and 2 students whose average mark is 45 are excluded, resulting in the average mark of the remaining students being 95, then the initial number of students is 20.
-/
theorem initial_number_of_students (N : ℕ) (T : ℕ)
  (h1 : T = N * 90)
  (h2 : (T - 90) / (N - 2) = 95) : 
  N = 20 :=
sorry

end initial_number_of_students_l48_48666


namespace age_difference_is_16_l48_48781

-- Variables
variables (y : ℕ) -- y represents the present age of the younger person

-- Conditions from the problem
def elder_present_age := 30
def elder_age_6_years_ago := elder_present_age - 6
def younger_age_6_years_ago := y - 6

-- Given condition 6 years ago:
def condition_6_years_ago := elder_age_6_years_ago = 3 * younger_age_6_years_ago

-- The theorem to prove the difference in ages is 16 years
theorem age_difference_is_16
  (h1 : elder_present_age = 30)
  (h2 : condition_6_years_ago) :
  elder_present_age - y = 16 :=
by sorry

end age_difference_is_16_l48_48781


namespace students_answered_both_correct_l48_48046

theorem students_answered_both_correct (total_students : ℕ)
  (answered_sets_correctly : ℕ) (answered_functions_correctly : ℕ)
  (both_wrong : ℕ) (total : total_students = 50)
  (sets_correct : answered_sets_correctly = 40)
  (functions_correct : answered_functions_correctly = 31)
  (wrong_both : both_wrong = 4) :
  (40 + 31 - (total_students - 4) + both_wrong = 50) → total_students - (40 + 31 - (total_students - 4)) = 29 :=
by
  sorry

end students_answered_both_correct_l48_48046


namespace set_intersection_l48_48159

theorem set_intersection (M N : Set ℝ) 
  (hM : M = {x | 2 * x - 3 < 1}) 
  (hN : N = {x | -1 < x ∧ x < 3}) : 
  (M ∩ N) = {x | -1 < x ∧ x < 2} := 
by 
  sorry

end set_intersection_l48_48159


namespace kaashish_problem_l48_48167

theorem kaashish_problem (x y : ℤ) (h : 2 * x + 3 * y = 100) (k : x = 28 ∨ y = 28) : x = 8 ∨ y = 8 :=
by
  sorry

end kaashish_problem_l48_48167


namespace sum_of_first_10_terms_is_350_l48_48514

-- Define the terms and conditions for the arithmetic sequence
variables (a d : ℤ)

-- Define the 4th and 8th terms of the sequence
def fourth_term := a + 3*d
def eighth_term := a + 7*d

-- Given conditions
axiom h1 : fourth_term a d = 23
axiom h2 : eighth_term a d = 55

-- Sum of the first 10 terms of the sequence
def sum_first_10_terms := 10 / 2 * (2*a + (10 - 1)*d)

-- Theorem to prove
theorem sum_of_first_10_terms_is_350 : sum_first_10_terms a d = 350 :=
by sorry

end sum_of_first_10_terms_is_350_l48_48514


namespace value_of_xyz_l48_48197

open Real

theorem value_of_xyz (x y z : ℝ)
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 37)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 11) :
  x * y * z = 26 / 3 := 
  sorry

end value_of_xyz_l48_48197


namespace solve_equation_real_l48_48827

theorem solve_equation_real (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) (h3 : x ≠ 4) :
  (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 2) * (x - 1) / ((x - 4) * (x - 2) * (x - 1)) = 1 ↔
  x = (9 + Real.sqrt 5) / 2 ∨ x = (9 - Real.sqrt 5) / 2 :=
by  
  sorry

end solve_equation_real_l48_48827


namespace base_7_minus_base_8_l48_48375

def convert_base_7 (n : ℕ) : ℕ :=
  match n with
  | 543210 => 5 * 7^5 + 4 * 7^4 + 3 * 7^3 + 2 * 7^2 + 1 * 7^1 + 0 * 7^0
  | _ => 0

def convert_base_8 (n : ℕ) : ℕ :=
  match n with
  | 45321 => 4 * 8^4 + 5 * 8^3 + 3 * 8^2 + 2 * 8^1 + 1 * 8^0
  | _ => 0

theorem base_7_minus_base_8 : convert_base_7 543210 - convert_base_8 45321 = 75620 := by
  sorry

end base_7_minus_base_8_l48_48375


namespace least_area_in_rectangle_l48_48458

theorem least_area_in_rectangle
  (x y : ℤ)
  (h1 : 2 * (x + y) = 150)
  (h2 : x > 0)
  (h3 : y > 0) :
  ∃ x y : ℤ, (2 * (x + y) = 150) ∧ (x * y = 74) := by
  sorry

end least_area_in_rectangle_l48_48458


namespace area_enclosed_by_equation_is_96_l48_48395

-- Definitions based on the conditions
def equation (x y : ℝ) : Prop := |x| + |3 * y| = 12

-- The theorem to prove the area enclosed by the graph is 96 square units
theorem area_enclosed_by_equation_is_96 :
  (∃ x y : ℝ, equation x y) → ∃ A : ℝ, A = 96 :=
sorry

end area_enclosed_by_equation_is_96_l48_48395


namespace solve_for_y_l48_48367

theorem solve_for_y {y : ℕ} (h : (1000 : ℝ) = (10 : ℝ)^3) : (1000 : ℝ)^4 = (10 : ℝ)^y ↔ y = 12 :=
by
  sorry

end solve_for_y_l48_48367


namespace max_lambda_inequality_l48_48132

theorem max_lambda_inequality (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  (1 / Real.sqrt (20 * a + 23 * b) + 1 / Real.sqrt (23 * a + 20 * b)) ≥ (2 / Real.sqrt 43 / Real.sqrt (a + b)) :=
by
  sorry

end max_lambda_inequality_l48_48132


namespace any_integer_amount_purchasable_amount_over_mn_minus_two_payable_l48_48782
open Int

variable (m n : ℕ) (h : Nat.gcd m n = 1)

theorem any_integer_amount_purchasable (x : ℤ) : 
  ∃ (a b : ℤ), a * n + b * m = x :=
by sorry

theorem amount_over_mn_minus_two_payable (k : ℤ) (hk : k > m * n - 2) : 
  ∃ (a b : ℤ), a * n + b * m = k :=
by sorry

end any_integer_amount_purchasable_amount_over_mn_minus_two_payable_l48_48782


namespace power_function_value_l48_48731

noncomputable def f (x : ℝ) : ℝ := x ^ (1/2)

-- Given the condition
axiom passes_through_point : f 3 = Real.sqrt 3

-- Prove that f(9) = 3
theorem power_function_value : f 9 = 3 := by
  sorry

end power_function_value_l48_48731


namespace Sn_divisible_by_3_Sn_divisible_by_9_iff_even_l48_48726

-- Define \( S_n \) following the given conditions
def S (n : ℕ) : ℕ :=
  let a := 2^n + 1 -- first term
  let b := 2^(n+1) - 1 -- last term
  let m := b - a + 1 -- number of terms
  (m * (a + b)) / 2 -- sum of the arithmetic series

-- The first part: Prove that \( S_n \) is divisible by 3 for all positive integers \( n \)
theorem Sn_divisible_by_3 (n : ℕ) (hn : 0 < n) : 3 ∣ S n := sorry

-- The second part: Prove that \( S_n \) is divisible by 9 if and only if \( n \) is even
theorem Sn_divisible_by_9_iff_even (n : ℕ) (hn : 0 < n) : 9 ∣ S n ↔ Even n := sorry

end Sn_divisible_by_3_Sn_divisible_by_9_iff_even_l48_48726


namespace johns_total_amount_l48_48237

def amount_from_grandpa : ℕ := 30
def multiplier : ℕ := 3
def amount_from_grandma : ℕ := amount_from_grandpa * multiplier
def total_amount : ℕ := amount_from_grandpa + amount_from_grandma

theorem johns_total_amount :
  total_amount = 120 :=
by
  sorry

end johns_total_amount_l48_48237


namespace value_of_a_plus_b_l48_48228

noncomputable def f (x : ℝ) := abs (Real.log (x + 1))

theorem value_of_a_plus_b (a b : ℝ) (h1 : a < b) 
  (h2 : f a = f (- (b + 1) / (b + 2))) 
  (h3 : f (10 * a + 6 * b + 21) = 4 * Real.log 2) : 
  a + b = -11 / 15 := 
by 
  sorry

end value_of_a_plus_b_l48_48228


namespace spaceship_travel_distance_l48_48886

-- Define each leg of the journey
def distance1 := 0.5
def distance2 := 0.1
def distance3 := 0.1

-- Define the total distance traveled
def total_distance := distance1 + distance2 + distance3

-- The statement to prove
theorem spaceship_travel_distance : total_distance = 0.7 := sorry

end spaceship_travel_distance_l48_48886


namespace tangent_line_equation_l48_48951

noncomputable def y (x : ℝ) := (2 * x - 1) / (x + 2)
def point : ℝ × ℝ := (-1, -3)
def tangent_eq (x y : ℝ) : Prop := 5 * x - y + 2 = 0

theorem tangent_line_equation :
  tangent_eq (-1) (-3) := 
sorry

end tangent_line_equation_l48_48951


namespace complement_U_A_l48_48430

def U : Set ℝ := { x | x^2 ≤ 4 }
def A : Set ℝ := { x | abs (x + 1) ≤ 1 }

theorem complement_U_A :
  (U \ A) = { x | 0 < x ∧ x ≤ 2 } :=
by
  sorry

end complement_U_A_l48_48430


namespace evaluate_five_applications_of_f_l48_48224

def f (x : ℤ) : ℤ :=
  if x ≥ 0 then x + 5 else -x^2 - 3

theorem evaluate_five_applications_of_f :
  f (f (f (f (f (-1))))) = -17554795004 :=
by
  sorry

end evaluate_five_applications_of_f_l48_48224


namespace problem_l48_48569

variable (a b c : ℝ)

def a_def : a = Real.log (1 / 2) := sorry
def b_def : b = Real.exp (1 / Real.exp 1) := sorry
def c_def : c = Real.exp (-2) := sorry

theorem problem (ha : a = Real.log (1 / 2)) 
               (hb : b = Real.exp (1 / Real.exp 1)) 
               (hc : c = Real.exp (-2)) : 
               a < c ∧ c < b := 
by
  rw [ha, hb, hc]
  sorry

end problem_l48_48569


namespace minimum_value_expression_l48_48539

theorem minimum_value_expression {a b c : ℤ} (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  4 * (a^2 + b^2 + c^2) - (a + b + c)^2 = 8 := 
sorry

end minimum_value_expression_l48_48539


namespace sum_of_factors_coefficients_l48_48518

theorem sum_of_factors_coefficients (a b c d e f g h i j k l m n o p : ℤ) :
  (81 * x^8 - 256 * y^8 = (a * x + b * y) *
                        (c * x^2 + d * x * y + e * y^2) *
                        (f * x^3 + g * x * y^2 + h * y^3) *
                        (i * x + j * y) *
                        (k * x^2 + l * x * y + m * y^2) *
                        (n * x^3 + o * x * y^2 + p * y^3)) →
  a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p = 40 :=
by
  sorry

end sum_of_factors_coefficients_l48_48518


namespace tens_digit_of_19_pow_2023_l48_48168

theorem tens_digit_of_19_pow_2023 :
  (19^2023 % 100) / 10 % 10 = 5 := 
sorry

end tens_digit_of_19_pow_2023_l48_48168


namespace roots_reciprocal_l48_48231

theorem roots_reciprocal (a b c r s : ℝ) (h_eqn : a ≠ 0) (h_roots : a * r^2 + b * r + c = 0 ∧ a * s^2 + b * s + c = 0) (h_cond : b^2 = 4 * a * c) : r * s = 1 :=
by
  -- Proof goes here
  sorry

end roots_reciprocal_l48_48231


namespace find_amount_l48_48969

-- Let A be the certain amount.
variable (A x : ℝ)

-- Given conditions
def condition1 (x : ℝ) := 0.65 * x = 0.20 * A
def condition2 (x : ℝ) := x = 150

-- Goal
theorem find_amount (A x : ℝ) (h1 : condition1 A x) (h2 : condition2 x) : A = 487.5 := 
by 
  sorry

end find_amount_l48_48969


namespace inequality_1_inequality_2_l48_48797

theorem inequality_1 (x : ℝ) : 4 * x + 5 ≤ 2 * (x + 1) → x ≤ -3/2 :=
by
  sorry

theorem inequality_2 (x : ℝ) : (2 * x - 1) / 3 - (9 * x + 2) / 6 ≤ 1 → x ≥ -2 :=
by
  sorry

end inequality_1_inequality_2_l48_48797


namespace greatest_odd_factors_l48_48553

theorem greatest_odd_factors (n : ℕ) (h1 : n < 1000) (h2 : ∀ k : ℕ, k * k = n → (k < 32)) :
  n = 31 * 31 :=
by
  sorry

end greatest_odd_factors_l48_48553


namespace natural_pairs_l48_48515

theorem natural_pairs (x y : ℕ) : 2^(2 * x + 1) + 2^x + 1 = y^2 ↔ (x = 0 ∧ y = 2) ∨ (x = 4 ∧ y = 23) :=
by sorry

end natural_pairs_l48_48515


namespace girls_additional_laps_l48_48174

def distance_per_lap : ℚ := 1 / 6
def boys_laps : ℕ := 34
def boys_distance : ℚ := boys_laps * distance_per_lap
def girls_distance : ℚ := 9
def additional_distance : ℚ := girls_distance - boys_distance
def additional_laps (distance : ℚ) (lap_distance : ℚ) : ℚ := distance / lap_distance

theorem girls_additional_laps :
  additional_laps additional_distance distance_per_lap = 20 := 
by
  sorry

end girls_additional_laps_l48_48174


namespace value_of_expression_l48_48671

variable {a : ℝ}

theorem value_of_expression (h : a^2 + 2 * a - 1 = 0) : 2 * a^2 + 4 * a - 2024 = -2022 :=
by
  sorry

end value_of_expression_l48_48671


namespace garden_wall_additional_courses_l48_48916

theorem garden_wall_additional_courses (initial_courses additional_courses : ℕ) (bricks_per_course total_bricks bricks_removed : ℕ) 
  (h1 : bricks_per_course = 400) 
  (h2 : initial_courses = 3) 
  (h3 : bricks_removed = bricks_per_course / 2) 
  (h4 : total_bricks = 1800) 
  (h5 : total_bricks = initial_courses * bricks_per_course + additional_courses * bricks_per_course - bricks_removed) : 
  additional_courses = 2 :=
by
  sorry

end garden_wall_additional_courses_l48_48916


namespace find_a_given_solution_l48_48311

theorem find_a_given_solution (a : ℝ) (x : ℝ) (h : x = 1) (eqn : a * (x + 1) = 2 * (2 * x - a)) : a = 1 := 
by
  sorry

end find_a_given_solution_l48_48311


namespace combined_garden_area_l48_48211

def garden_area (length width : ℕ) : ℕ :=
  length * width

def total_area (count length width : ℕ) : ℕ :=
  count * garden_area length width

theorem combined_garden_area :
  let M_length := 16
  let M_width := 5
  let M_count := 3
  let Ma_length := 8
  let Ma_width := 4
  let Ma_count := 2
  total_area M_count M_length M_width + total_area Ma_count Ma_length Ma_width = 304 :=
by
  sorry

end combined_garden_area_l48_48211


namespace right_triangle_hypotenuse_enlargement_l48_48933

theorem right_triangle_hypotenuse_enlargement
  (a b c : ℝ)
  (h : c^2 = a^2 + b^2) :
  ((5 * a)^2 + (5 * b)^2 = (5 * c)^2) :=
by sorry

end right_triangle_hypotenuse_enlargement_l48_48933


namespace milk_butterfat_mixture_l48_48049

theorem milk_butterfat_mixture (x gallons_50 gall_10_perc final_gall mixture_perc: ℝ)
    (H1 : gall_10_perc = 24) 
    (H2 : mixture_perc = 0.20 * (x + gall_10_perc))
    (H3 : 0.50 * x + 0.10 * gall_10_perc = 0.20 * (x + gall_10_perc)) 
    (H4 : final_gall = 20) :
    x = 8 :=
sorry

end milk_butterfat_mixture_l48_48049


namespace impossible_tiling_conditions_l48_48205

theorem impossible_tiling_conditions (m n : ℕ) :
  ¬ (∃ (a b : ℕ), (a - 1) * 4 + (b + 1) * 4 = m * n ∧ a * 4 % 4 = 2 ∧ b * 4 % 4 = 0) :=
sorry

end impossible_tiling_conditions_l48_48205


namespace minimum_basketballs_sold_l48_48631

theorem minimum_basketballs_sold :
  ∃ (F B K : ℕ), F + B + K = 180 ∧ 3 * F + 5 * B + 10 * K = 800 ∧ F > B ∧ B > K ∧ K = 2 :=
by
  sorry

end minimum_basketballs_sold_l48_48631


namespace ducks_cows_problem_l48_48247

theorem ducks_cows_problem (D C : ℕ) (h : 2 * D + 4 * C = 2 * (D + C) + 24) : C = 12 := 
  sorry

end ducks_cows_problem_l48_48247


namespace derivative_at_pi_over_3_l48_48382

noncomputable def f (x : ℝ) : ℝ := Real.cos x + Real.sqrt 3 * Real.sin x

theorem derivative_at_pi_over_3 : 
  (deriv f) (Real.pi / 3) = 0 := 
by 
  sorry

end derivative_at_pi_over_3_l48_48382


namespace travel_cost_from_B_to_C_l48_48734

noncomputable def distance (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

noncomputable def travel_cost_by_air (distance : ℝ) (booking_fee : ℝ) (per_km_cost : ℝ) : ℝ :=
  booking_fee + (distance * per_km_cost)

theorem travel_cost_from_B_to_C :
  let AC := 4000
  let AB := 4500
  let BC := Real.sqrt (AB^2 - AC^2)
  let booking_fee := 120
  let per_km_cost := 0.12
  travel_cost_by_air BC booking_fee per_km_cost = 367.39 := by
  sorry

end travel_cost_from_B_to_C_l48_48734


namespace total_movie_hours_l48_48443

-- Definitions
def JoyceMovie : ℕ := 12 -- Joyce's favorite movie duration in hours
def MichaelMovie : ℕ := 10 -- Michael's favorite movie duration in hours
def NikkiMovie : ℕ := 30 -- Nikki's favorite movie duration in hours
def RynMovie : ℕ := 24 -- Ryn's favorite movie duration in hours

-- Condition translations
def Joyce_movie_condition : Prop := JoyceMovie = MichaelMovie + 2
def Nikki_movie_condition : Prop := NikkiMovie = 3 * MichaelMovie
def Ryn_movie_condition : Prop := RynMovie = (4 * NikkiMovie) / 5
def Nikki_movie_given : Prop := NikkiMovie = 30

-- The theorem to prove
theorem total_movie_hours : Joyce_movie_condition ∧ Nikki_movie_condition ∧ Ryn_movie_condition ∧ Nikki_movie_given → 
  (JoyceMovie + MichaelMovie + NikkiMovie + RynMovie = 76) :=
by
  intros h
  sorry

end total_movie_hours_l48_48443


namespace measure_diagonal_without_pythagorean_theorem_l48_48435

variables (a b c : ℝ)

-- Definition of the function to measure the diagonal distance
def diagonal_method (a b c : ℝ) : ℝ :=
  -- by calculating the hypotenuse scaled by sqrt(3), we ignore using the Pythagorean theorem directly
  sorry

-- Calculate distance by arranging bricks
theorem measure_diagonal_without_pythagorean_theorem (distance_extreme_corners : ℝ) :
  distance_extreme_corners = (diagonal_method a b c) :=
  sorry

end measure_diagonal_without_pythagorean_theorem_l48_48435


namespace find_jamals_grade_l48_48364

noncomputable def jamals_grade (n_students : ℕ) (absent_students : ℕ) (test_avg_28_students : ℕ) (new_total_avg_30_students : ℕ) (taqeesha_score : ℕ) : ℕ :=
  let total_28_students := 28 * test_avg_28_students
  let total_30_students := 30 * new_total_avg_30_students
  let combined_score := total_30_students - total_28_students
  combined_score - taqeesha_score

theorem find_jamals_grade :
  jamals_grade 30 2 85 86 92 = 108 :=
by
  sorry

end find_jamals_grade_l48_48364


namespace todd_money_after_repay_l48_48742

-- Definitions as conditions
def borrowed_amount : ℤ := 100
def repay_amount : ℤ := 110
def ingredients_cost : ℤ := 75
def snow_cones_sold : ℤ := 200
def price_per_snow_cone : ℚ := 0.75

-- Function to calculate money left after transactions
def money_left_after_repay (borrowed : ℤ) (repay : ℤ) (cost : ℤ) (sold : ℤ) (price : ℚ) : ℚ :=
  let left_after_cost := borrowed - cost
  let earnings := sold * price
  let total_before_repay := left_after_cost + earnings
  total_before_repay - repay

-- The theorem stating the problem
theorem todd_money_after_repay :
  money_left_after_repay borrowed_amount repay_amount ingredients_cost snow_cones_sold price_per_snow_cone = 65 := 
by
  -- This is a placeholder for the actual proof
  sorry

end todd_money_after_repay_l48_48742


namespace not_perfect_square_9n_squared_minus_9n_plus_9_l48_48044

theorem not_perfect_square_9n_squared_minus_9n_plus_9
  (n : ℕ) (h : n > 1) : ¬ (∃ k : ℕ, 9 * n^2 - 9 * n + 9 = k * k) := sorry

end not_perfect_square_9n_squared_minus_9n_plus_9_l48_48044


namespace professional_pay_per_hour_l48_48703

def professionals : ℕ := 2
def hours_per_day : ℕ := 6
def days : ℕ := 7
def total_cost : ℕ := 1260

theorem professional_pay_per_hour :
  (total_cost / (professionals * hours_per_day * days) = 15) :=
by
  sorry

end professional_pay_per_hour_l48_48703


namespace sum_of_number_and_reverse_l48_48219

theorem sum_of_number_and_reverse (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) :
  (10 * a + b) + (10 * b + a) = 99 :=
by
  sorry

end sum_of_number_and_reverse_l48_48219


namespace part1_part2_l48_48128

noncomputable def f (x : ℝ) (k : ℝ) : ℝ :=
  k - |x - 3|

theorem part1 (k : ℝ) (h : ∀ x, f (x + 3) k ≥ 0 ↔ x ∈ [-1, 1]) : k = 1 :=
sorry

variable (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c)

theorem part2 (h : (1 / a) + (1 / (2 * b)) + (1 / (3 * c)) = 1) : 
  (1 / 9) * a + (2 / 9) * b + (3 / 9) * c ≥ 1 :=
sorry

end part1_part2_l48_48128


namespace F_2457_find_Q_l48_48470

-- Define the properties of a "rising number"
def is_rising_number (m : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    m = 1000 * a + 100 * b + 10 * c + d ∧
    a < b ∧ b < c ∧ c < d ∧
    a + d = b + c

-- Define F(m) as specified
def F (m : ℕ) : ℤ :=
  let a := m / 1000
  let b := (m / 100) % 10
  let c := (m / 10) % 10
  let d := m % 10
  let m' := 1000 * c + 100 * b + 10 * a + d
  (m' - m) / 99

-- Problem statement for F(2457)
theorem F_2457 : F 2457 = 30 := sorry

-- Properties given in the problem statement for P and Q
def is_specific_rising_number (P Q : ℕ) : Prop :=
  ∃ (x y z t : ℕ),
    P = 1000 + 100 * x + 10 * y + z ∧
    Q = 1000 * x + 100 * t + 60 + z ∧
    1 < x ∧ x < t ∧ t < 6 ∧ 6 < z ∧
    1 + z = x + y ∧
    x + z = t + 6 ∧
    F P + F Q % 7 = 0

-- Problem statement to find the value of Q
theorem find_Q (Q : ℕ) : 
  ∃ (P : ℕ), is_specific_rising_number P Q ∧ Q = 3467 := sorry

end F_2457_find_Q_l48_48470


namespace f_divisible_by_8_l48_48088

-- Define the function f
def f (n : ℕ) : ℕ := 5^n + 2 * 3^(n-1) + 1

-- Theorem statement
theorem f_divisible_by_8 (n : ℕ) (hn : n > 0) : 8 ∣ f n := sorry

end f_divisible_by_8_l48_48088


namespace sum_of_fractions_is_correct_l48_48366

-- Definitions from the conditions
def half_of_third := (1 : ℚ) / 2 * (1 : ℚ) / 3
def third_of_quarter := (1 : ℚ) / 3 * (1 : ℚ) / 4
def quarter_of_fifth := (1 : ℚ) / 4 * (1 : ℚ) / 5
def sum_fractions := half_of_third + third_of_quarter + quarter_of_fifth

-- The theorem to prove
theorem sum_of_fractions_is_correct : sum_fractions = (3 : ℚ) / 10 := by
  sorry

end sum_of_fractions_is_correct_l48_48366


namespace problem_l48_48429

theorem problem (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1 - x) = x^2 + 1) : f (-1) = 5 := 
  sorry

end problem_l48_48429


namespace response_rate_increase_approx_l48_48269

theorem response_rate_increase_approx :
  let original_customers := 80
  let original_respondents := 7
  let redesigned_customers := 63
  let redesigned_respondents := 9
  let original_response_rate := (original_respondents : ℝ) / original_customers * 100
  let redesigned_response_rate := (redesigned_respondents : ℝ) / redesigned_customers * 100
  let percentage_increase := (redesigned_response_rate - original_response_rate) / original_response_rate * 100
  abs (percentage_increase - 63.24) < 0.01 := by
  sorry

end response_rate_increase_approx_l48_48269


namespace negation_of_neither_even_l48_48667

variable (a b : Nat)

def is_even (n : Nat) : Prop :=
  n % 2 = 0

theorem negation_of_neither_even 
  (H : ¬ (¬ is_even a ∧ ¬ is_even b)) : is_even a ∨ is_even b :=
sorry

end negation_of_neither_even_l48_48667


namespace time_comparison_l48_48291

variable (s : ℝ) (h_pos : s > 0)

noncomputable def t1 : ℝ := 120 / s
noncomputable def t2 : ℝ := 480 / (4 * s)

theorem time_comparison : t1 s = t2 s := by
  rw [t1, t2]
  field_simp [h_pos]
  norm_num
  sorry

end time_comparison_l48_48291


namespace points_symmetric_about_y_eq_x_l48_48697

theorem points_symmetric_about_y_eq_x (x y r : ℝ) :
  (x^2 + y^2 ≤ r^2 ∧ x + y > 0) →
  (∃ p q : ℝ, (q = p ∧ p + q = 0) ∨ (p = q ∧ q = -p)) :=
sorry

end points_symmetric_about_y_eq_x_l48_48697


namespace woman_work_time_l48_48331

theorem woman_work_time :
  ∀ (M W B : ℝ), (M = 1/6) → (B = 1/12) → (M + W + B = 1/3) → (W = 1/12) → (1 / W = 12) :=
by
  intros M W B hM hB h_combined hW
  sorry

end woman_work_time_l48_48331


namespace sector_area_l48_48562

theorem sector_area (arc_length : ℝ) (central_angle : ℝ) (r : ℝ) (area : ℝ) : 
  arc_length = π / 3 ∧ central_angle = π / 6 → arc_length = central_angle * r → area = 1 / 2 * central_angle * r^2 → area = π / 3 :=
by
  sorry

end sector_area_l48_48562


namespace intersection_complement_N_l48_48602

def is_universal_set (R : Set ℝ) : Prop := ∀ x : ℝ, x ∈ R

def is_complement (U S C : Set ℝ) : Prop := 
  ∀ x : ℝ, x ∈ C ↔ x ∈ U ∧ x ∉ S

theorem intersection_complement_N 
  (U M N C : Set ℝ)
  (h_universal : is_universal_set U)
  (hM : M = {x : ℝ | -2 ≤ x ∧ x ≤ 2})
  (hN : N = {x : ℝ | x < 1})
  (h_compl : is_complement U M C) :
  (C ∩ N) = {x : ℝ | x < -2} := 
by 
  sorry

end intersection_complement_N_l48_48602


namespace meeting_point_2015_is_C_l48_48212

-- Given definitions based on conditions
variable (x y : ℝ) -- Speeds of the motorcycle and the cyclist
variable (A B C D : Point) -- Points on segment AB
variable (meetings : ℕ → Point) -- Function representing the meeting point sequence

-- Conditions stating the alternating meeting pattern
axiom start_at_A (n : ℕ) : meetings (2 * n + 1) = C
axiom start_at_B (n : ℕ) : meetings (2 * n + 2) = D

-- The theorem statement to be proved
theorem meeting_point_2015_is_C : meetings 2015 = C := sorry

end meeting_point_2015_is_C_l48_48212


namespace parakeets_per_cage_l48_48961

-- Define total number of cages
def num_cages: Nat := 6

-- Define number of parrots per cage
def parrots_per_cage: Nat := 2

-- Define total number of birds in the store
def total_birds: Nat := 54

-- Theorem statement: prove the number of parakeets per cage
theorem parakeets_per_cage : (total_birds - num_cages * parrots_per_cage) / num_cages = 7 :=
by
  sorry

end parakeets_per_cage_l48_48961


namespace A_and_C_work_together_in_2_hours_l48_48216

theorem A_and_C_work_together_in_2_hours
  (A_rate : ℚ)
  (B_rate : ℚ)
  (C_rate : ℚ)
  (A_4_hours : A_rate = 1 / 4)
  (B_12_hours : B_rate = 1 / 12)
  (B_and_C_3_hours : B_rate + C_rate = 1 / 3) :
  (A_rate + C_rate = 1 / 2) :=
by
  sorry

end A_and_C_work_together_in_2_hours_l48_48216


namespace line_intersects_circle_l48_48626

noncomputable def line_eqn (a : ℝ) (x y : ℝ) : ℝ := a * x - y - a + 3
noncomputable def circle_eqn (x y : ℝ) : ℝ := x^2 + y^2 - 4 * x - 2 * y - 4

-- Given the line l passes through M(1, 3)
def passes_through_M (a : ℝ) : Prop := line_eqn a 1 3 = 0

-- Given M(1, 3) is inside the circle
def M_inside_circle : Prop := circle_eqn 1 3 < 0

-- To prove the line intersects the circle
theorem line_intersects_circle (a : ℝ) (h1 : passes_through_M a) (h2 : M_inside_circle) : 
  ∃ p : ℝ × ℝ, line_eqn a p.1 p.2 = 0 ∧ circle_eqn p.1 p.2 = 0 :=
sorry

end line_intersects_circle_l48_48626


namespace last_two_digits_2005_power_1989_l48_48278

theorem last_two_digits_2005_power_1989 : (2005 ^ 1989) % 100 = 25 :=
by
  sorry

end last_two_digits_2005_power_1989_l48_48278


namespace sum_of_cubes_equals_square_l48_48292

theorem sum_of_cubes_equals_square :
  1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 = 21^2 := 
by 
  sorry

end sum_of_cubes_equals_square_l48_48292


namespace minimum_amount_spent_on_boxes_l48_48271

theorem minimum_amount_spent_on_boxes
  (box_length : ℕ) (box_width : ℕ) (box_height : ℕ) 
  (cost_per_box : ℝ) (total_volume_needed : ℕ) :
  box_length = 20 →
  box_width = 20 →
  box_height = 12 →
  cost_per_box = 0.50 →
  total_volume_needed = 2400000 →
  (total_volume_needed / (box_length * box_width * box_height) * cost_per_box) = 250 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end minimum_amount_spent_on_boxes_l48_48271


namespace root_sum_product_eq_l48_48354

theorem root_sum_product_eq (p q : ℝ) (h1 : p / 3 = 9) (h2 : q / 3 = 14) :
  p + q = 69 :=
by 
  sorry

end root_sum_product_eq_l48_48354


namespace cookies_in_jar_l48_48568

theorem cookies_in_jar (C : ℕ) (h : C - 1 = (C + 5) / 2) : C = 7 :=
by
  -- Proof goes here
  sorry

end cookies_in_jar_l48_48568


namespace distance_swam_against_current_l48_48101

def swimming_speed_in_still_water : ℝ := 4
def speed_of_current : ℝ := 2
def time_taken_against_current : ℝ := 5

theorem distance_swam_against_current : ∀ distance : ℝ,
  (distance = (swimming_speed_in_still_water - speed_of_current) * time_taken_against_current) → distance = 10 :=
by
  intros distance h
  sorry

end distance_swam_against_current_l48_48101


namespace minimum_value_of_f_l48_48662

-- Define the function y = f(x)
def f (x : ℝ) : ℝ := x^2 + 8 * x + 25

-- We need to prove that the minimum value of f(x) is 9
theorem minimum_value_of_f : ∃ x : ℝ, f x = 9 ∧ ∀ y : ℝ, f y ≥ 9 :=
by
  sorry

end minimum_value_of_f_l48_48662


namespace lemonade_problem_l48_48987

theorem lemonade_problem (L S W : ℕ) (h1 : W = 4 * S) (h2 : S = 2 * L) (h3 : L = 3) : L + S + W = 24 :=
by
  sorry

end lemonade_problem_l48_48987


namespace num_unique_seven_digit_integers_l48_48160

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

noncomputable def unique_seven_digit_integers : ℕ :=
  factorial 7 / (factorial 2 * factorial 2 * factorial 2)

theorem num_unique_seven_digit_integers : unique_seven_digit_integers = 630 := by
  sorry

end num_unique_seven_digit_integers_l48_48160


namespace complex_expression_equality_l48_48257

-- Define the basic complex number properties and operations.
def i : ℂ := Complex.I -- Define the imaginary unit

theorem complex_expression_equality (a b : ℤ) :
  (3 - 4 * i) * ((-4 + 2 * i) ^ 2) = -28 - 96 * i :=
by
  -- Syntactical proof placeholders
  sorry

end complex_expression_equality_l48_48257


namespace stickers_distribution_l48_48188

-- Definitions for initial sticker quantities and stickers given to first four friends
def initial_space_stickers : ℕ := 120
def initial_cat_stickers : ℕ := 80
def initial_dinosaur_stickers : ℕ := 150
def initial_superhero_stickers : ℕ := 45

def given_space_stickers : ℕ := 25
def given_cat_stickers : ℕ := 13
def given_dinosaur_stickers : ℕ := 33
def given_superhero_stickers : ℕ := 29

-- Definitions for remaining stickers calculation
def remaining_space_stickers : ℕ := initial_space_stickers - given_space_stickers
def remaining_cat_stickers : ℕ := initial_cat_stickers - given_cat_stickers
def remaining_dinosaur_stickers : ℕ := initial_dinosaur_stickers - given_dinosaur_stickers
def remaining_superhero_stickers : ℕ := initial_superhero_stickers - given_superhero_stickers

def total_remaining_stickers : ℕ := remaining_space_stickers + remaining_cat_stickers + remaining_dinosaur_stickers + remaining_superhero_stickers

-- Definition for number of each type of new sticker
def each_new_type_stickers : ℕ := total_remaining_stickers / 4
def remainder_stickers : ℕ := total_remaining_stickers % 4

-- Statement to be proved
theorem stickers_distribution :
  ∃ X : ℕ, X = 3 ∧ each_new_type_stickers = 73 :=
by
  sorry

end stickers_distribution_l48_48188


namespace OC_eq_l48_48992

variable {V : Type} [AddCommGroup V]

-- Given vectors a and b
variables (a b : V)

-- Conditions given in the problem
def OA := a + b
def AB := 3 • (a - b)
def CB := 2 • a + b

-- Prove that OC = 2a - 3b
theorem OC_eq : (a + b) + (3 • (a - b)) + (- (2 • a + b)) = 2 • a - 3 • b :=
by
  -- write your proof here
  sorry

end OC_eq_l48_48992


namespace total_seats_round_table_l48_48767

theorem total_seats_round_table (n : ℕ) (h : n = 38)
  (ka_position : ℕ) (sl_position : ℕ) 
  (h1 : ka_position = 10) 
  (h2 : sl_position = 29) 
  (h3 : (ka_position + n/2) % n = sl_position) : 
  n = 38 :=
by
  -- All steps and solution proof
  sorry

end total_seats_round_table_l48_48767


namespace sum_p_q_r_l48_48328

theorem sum_p_q_r :
  ∃ (p q r : ℤ), 
    (∀ x : ℤ, x ^ 2 + 20 * x + 96 = (x + p) * (x + q)) ∧ 
    (∀ x : ℤ, x ^ 2 - 22 * x + 120 = (x - q) * (x - r)) ∧ 
    p + q + r = 30 :=
by 
  sorry

end sum_p_q_r_l48_48328


namespace volume_of_rectangular_prism_l48_48694

theorem volume_of_rectangular_prism (l w h : ℕ) (x : ℕ) 
  (h_ratio : l = 3 * x ∧ w = 2 * x ∧ h = x)
  (h_edges : 4 * l + 4 * w + 4 * h = 72) : 
  l * w * h = 162 := 
by
  sorry

end volume_of_rectangular_prism_l48_48694


namespace geometric_seq_ratio_l48_48450

theorem geometric_seq_ratio : 
  ∀ (a : ℕ → ℝ) (q : ℝ), 
    (∀ n, a (n+1) = a n * q) → 
    q > 1 → 
    a 1 + a 6 = 8 → 
    a 3 * a 4 = 12 → 
    a 2018 / a 2013 = 3 :=
by
  intros a q h_geom h_q_pos h_sum_eq h_product_eq
  sorry

end geometric_seq_ratio_l48_48450


namespace buying_ways_l48_48302

theorem buying_ways (students : ℕ) (choices : ℕ) (at_least_one_pencil : ℕ) : 
  students = 4 ∧ choices = 2 ∧ at_least_one_pencil = 1 → 
  (choices^students - 1) = 15 :=
by
  sorry

end buying_ways_l48_48302


namespace opposite_neg_one_half_l48_48543

def opposite (x : ℚ) : ℚ := -x

theorem opposite_neg_one_half :
  opposite (- 1 / 2) = 1 / 2 := by
  sorry

end opposite_neg_one_half_l48_48543


namespace min_f_x_gt_2_solve_inequality_l48_48386

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 / (x + b)

theorem min_f_x_gt_2 (a b : ℝ) (h1 : ∀ x, f a b x = 2 * x + 3 → x = -2 ∨ x = 3) :
∃ c, ∀ x > 2, f a b x ≥ c ∧ (∀ y, y > 2 → f a b y = c → y = 4 ∧ c = 8) :=
sorry

theorem solve_inequality (a b k : ℝ) (x : ℝ) (h1 : ∀ x, f a b x = 2 * x + 3 → x = -2 ∨ x = 3) :
  f a b x < (k * (x - 1) + 1 - x^2) / (2 - x) ↔ 
  (x < 2 ∧ k = 0) ∨ 
  (-1 < k ∧ k < 0 ∧ 1 - 1 / k < x ∧ x < 2) ∨ 
  ((k > 0 ∨ k < -1) ∧ (1 - 1 / k < x ∧ x < 2) ∨ x > 2) ∨ 
  (k = -1 ∧ x ≠ 2) :=
sorry

end min_f_x_gt_2_solve_inequality_l48_48386


namespace intersection_of_A_and_B_l48_48914

def A : Set ℝ := {x | x ≤ 1}
def B : Set ℝ := {x | -Real.sqrt 3 < x ∧ x < Real.sqrt 3}

theorem intersection_of_A_and_B : (A ∩ B) = {x | -Real.sqrt 3 < x ∧ x ≤ 1} := by
  sorry

end intersection_of_A_and_B_l48_48914


namespace Matt_income_from_plantation_l48_48406

noncomputable def plantation_income :=
  let plantation_area := 500 * 500  -- square feet
  let grams_peanuts_per_sq_ft := 50 -- grams
  let grams_peanut_butter_per_20g_peanuts := 5  -- grams
  let price_per_kg_peanut_butter := 10 -- $

  -- Total revenue calculation
  plantation_area * grams_peanuts_per_sq_ft * grams_peanut_butter_per_20g_peanuts /
  20 / 1000 * price_per_kg_peanut_butter

theorem Matt_income_from_plantation :
  plantation_income = 31250 := sorry

end Matt_income_from_plantation_l48_48406


namespace ellipse_foci_distance_l48_48849

theorem ellipse_foci_distance 
  (a b : ℝ) 
  (h_a : a = 8) 
  (h_b : b = 3) : 
  2 * (Real.sqrt (a^2 - b^2)) = 2 * Real.sqrt 55 := 
by
  rw [h_a, h_b]
  sorry

end ellipse_foci_distance_l48_48849


namespace water_spilled_l48_48756

theorem water_spilled (x s : ℕ) (h1 : s = x + 7) : s = 8 := by
  -- The proof would go here
  sorry

end water_spilled_l48_48756


namespace minimum_value_l48_48956

def minimum_value_problem (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_sum : a + b = 2) : Prop :=
  ∃ c : ℝ, c = (1 / (a + 1) + 4 / (b + 1)) ∧ c = 9 / 4

theorem minimum_value (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_sum : a + b = 2) : 
  (1 / (a + 1) + 4 / (b + 1)) = 9 / 4 :=
by 
  -- Proof goes here
  sorry

end minimum_value_l48_48956


namespace captain_age_l48_48444

noncomputable def whole_team_age : ℕ := 253
noncomputable def remaining_players_age : ℕ := 198
noncomputable def captain_and_wicket_keeper_age : ℕ := whole_team_age - remaining_players_age
noncomputable def wicket_keeper_age (C : ℕ) : ℕ := C + 3

theorem captain_age (C : ℕ) (whole_team : whole_team_age = 11 * 23) (remaining_players : remaining_players_age = 9 * 22) 
    (sum_ages : captain_and_wicket_keeper_age = 55) (wicket_keeper : wicket_keeper_age C = C + 3) : C = 26 := 
  sorry

end captain_age_l48_48444


namespace green_block_weight_l48_48401

theorem green_block_weight (y g : ℝ) (h1 : y = 0.6) (h2 : y = g + 0.2) : g = 0.4 :=
by
  sorry

end green_block_weight_l48_48401


namespace find_tangent_line_l48_48975

theorem find_tangent_line (k : ℝ) :
  (∃ k : ℝ, ∀ (x y : ℝ), y = k * (x - 1) + 3 ∧ k^2 + 1 = 1) →
  (∃ k : ℝ, k = 4 / 3 ∧ (k * x - y + 3 - k = 0) ∨ (x = 1)) :=
sorry

end find_tangent_line_l48_48975


namespace percentage_difference_l48_48344

theorem percentage_difference (x y : ℝ) (h : x = 12 * y) : (1 - y / x) * 100 = 91.67 :=
by {
  sorry
}

end percentage_difference_l48_48344


namespace trig_identity_l48_48893

theorem trig_identity (α : ℝ) (h : 3 * Real.sin α + Real.cos α = 0) : 
  (1 / (Real.cos α ^ 2 + Real.sin (2 * α))) = 10 / 3 := 
by 
  sorry

end trig_identity_l48_48893


namespace height_percentage_l48_48086

theorem height_percentage (a b c : ℝ) 
  (h1 : a = 0.6 * b) 
  (h2 : c = 1.25 * a) : 
  (b - a) / a * 100 = 66.67 ∧ (c - a) / a * 100 = 25 := 
by 
  sorry

end height_percentage_l48_48086


namespace pentagon_largest_angle_l48_48459

variable (F G H I J : ℝ)

-- Define the conditions given in the problem
axiom angle_sum : F + G + H + I + J = 540
axiom angle_F : F = 80
axiom angle_G : G = 100
axiom angle_HI : H = I
axiom angle_J : J = 2 * H + 20

-- Statement that the largest angle in the pentagon is 190°
theorem pentagon_largest_angle : max F (max G (max H (max I J))) = 190 :=
sorry

end pentagon_largest_angle_l48_48459


namespace S_is_multiples_of_six_l48_48690

-- Defining the problem.
def S : Set ℝ :=
  { t | ∃ n : ℤ, t = 6 * n }

-- We are given that S is non-empty
axiom S_non_empty : ∃ x, x ∈ S

-- Condition: For any x, y ∈ S, both x + y ∈ S and x - y ∈ S.
axiom S_closed_add_sub : ∀ x y, x ∈ S → y ∈ S → (x + y ∈ S ∧ x - y ∈ S)

-- The smallest positive number in S is 6.
axiom S_smallest : ∀ ε, ε > 0 → ∃ x, x ∈ S ∧ x = 6

-- The goal is to prove that S is exactly the set of all multiples of 6.
theorem S_is_multiples_of_six : ∀ t, t ∈ S ↔ ∃ n : ℤ, t = 6 * n :=
by
  sorry

end S_is_multiples_of_six_l48_48690


namespace train_speed_proof_l48_48715

noncomputable def train_speed (L : ℕ) (t : ℝ) (v_m : ℝ) : ℝ :=
  let v_m_m_s := v_m * (1000 / 3600)
  let v_rel := L / t
  v_rel + v_m_m_s

theorem train_speed_proof
  (L : ℕ)
  (t : ℝ)
  (v_m : ℝ)
  (hL : L = 900)
  (ht : t = 53.99568034557235)
  (hv_m : v_m = 3)
  : train_speed L t v_m = 63.0036 :=
  by sorry

end train_speed_proof_l48_48715


namespace jack_jill_same_speed_l48_48105

theorem jack_jill_same_speed (x : ℝ) (h : x^2 - 8*x - 10 = 0) :
  (x^2 - 7*x - 18) = 2 := 
sorry

end jack_jill_same_speed_l48_48105


namespace find_m_range_l48_48198

-- Defining the function and conditions
variable {f : ℝ → ℝ}
variable {m : ℝ}

-- Prove if given the conditions, then the range of m is as specified
theorem find_m_range (h1 : ∀ x, f (-x) = -f x) 
                     (h2 : ∀ x, -2 < x ∧ x < 2 → f (x) > f (x+1)) 
                     (h3 : -2 < m - 1 ∧ m - 1 < 2) 
                     (h4 : -2 < 2 * m - 1 ∧ 2 * m - 1 < 2) 
                     (h5 : f (m - 1) + f (2 * m - 1) > 0) :
  -1/2 < m ∧ m < 2/3 :=
sorry

end find_m_range_l48_48198


namespace grilled_cheese_sandwiches_l48_48404

-- Define the number of ham sandwiches Joan makes
def ham_sandwiches := 8

-- Define the cheese requirements for each type of sandwich
def cheddar_for_ham := 1
def swiss_for_ham := 1
def cheddar_for_grilled := 2
def gouda_for_grilled := 1

-- Total cheese used
def total_cheddar := 40
def total_swiss := 20
def total_gouda := 30

-- Prove the number of grilled cheese sandwiches Joan makes
theorem grilled_cheese_sandwiches (ham_sandwiches : ℕ) (cheddar_for_ham : ℕ) (swiss_for_ham : ℕ)
                                  (cheddar_for_grilled : ℕ) (gouda_for_grilled : ℕ)
                                  (total_cheddar : ℕ) (total_swiss : ℕ) (total_gouda : ℕ) :
    (total_cheddar - ham_sandwiches * cheddar_for_ham) / cheddar_for_grilled = 16 :=
by
  sorry

end grilled_cheese_sandwiches_l48_48404


namespace program_output_l48_48345

theorem program_output (x : ℤ) : 
  (if x < 0 then -1 else if x = 0 then 0 else 1) = 1 ↔ x = 3 :=
by
  sorry

end program_output_l48_48345


namespace range_of_m_l48_48348

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 5| < m^2 - m) ↔ m < -1 ∨ m > 2 := 
by
  sorry

end range_of_m_l48_48348


namespace parallel_lines_condition_l48_48761

theorem parallel_lines_condition {a : ℝ} :
  (∀ x y : ℝ, a * x + 2 * y + 3 * a = 0) ∧ (∀ x y : ℝ, 3 * x + (a - 1) * y = a - 7) ↔ a = 3 :=
by
  sorry

end parallel_lines_condition_l48_48761


namespace jessies_original_weight_l48_48380

theorem jessies_original_weight (current_weight weight_lost original_weight : ℕ) 
  (h_current: current_weight = 27) (h_lost: weight_lost = 101) 
  (h_original: original_weight = current_weight + weight_lost) : 
  original_weight = 128 :=
by
  rw [h_current, h_lost] at h_original
  exact h_original

end jessies_original_weight_l48_48380


namespace factorize_polynomial_l48_48463

theorem factorize_polynomial (x y : ℝ) : 
  (2 * x^2 * y - 4 * x * y^2 + 2 * y^3) = 2 * y * (x - y)^2 :=
sorry

end factorize_polynomial_l48_48463


namespace smallest_repeating_block_fraction_3_over_11_l48_48766

theorem smallest_repeating_block_fraction_3_over_11 :
  ∃ n : ℕ, (∃ d : ℕ, (3/11 : ℚ) = d / 10^n) ∧ n = 2 :=
sorry

end smallest_repeating_block_fraction_3_over_11_l48_48766


namespace parabolic_arch_properties_l48_48402

noncomputable def parabolic_arch_height (x : ℝ) : ℝ :=
  let a : ℝ := -4 / 125
  let k : ℝ := 20
  a * x^2 + k

theorem parabolic_arch_properties :
  (parabolic_arch_height 10 = 16.8) ∧ (parabolic_arch_height 10 = parabolic_arch_height 10 → (10 = 10 ∨ 10 = -10)) :=
by
  have h1 : parabolic_arch_height 10 = 16.8 :=
    sorry
  have h2 : parabolic_arch_height 10 = parabolic_arch_height 10 → (10 = 10 ∨ 10 = -10) :=
    sorry
  exact ⟨h1, h2⟩

end parabolic_arch_properties_l48_48402


namespace theresa_hours_l48_48232

theorem theresa_hours (h1 h2 h3 h4 h5 h6 : ℕ) (avg : ℕ) (x : ℕ) 
  (H_cond : h1 = 10 ∧ h2 = 8 ∧ h3 = 9 ∧ h4 = 11 ∧ h5 = 6 ∧ h6 = 8)
  (H_avg : avg = 9) : 
  (h1 + h2 + h3 + h4 + h5 + h6 + x) / 7 = avg ↔ x = 11 :=
by
  sorry

end theresa_hours_l48_48232


namespace vet_appointments_cost_l48_48187

variable (x : ℝ)

def JohnVetAppointments (x : ℝ) : Prop := 
  (x + 0.20 * x + 0.20 * x + 100 = 660)

theorem vet_appointments_cost :
  (∃ x : ℝ, JohnVetAppointments x) → x = 400 :=
by
  intro h
  obtain ⟨x, hx⟩ := h
  simp [JohnVetAppointments] at hx
  sorry

end vet_appointments_cost_l48_48187


namespace remainder_equiv_l48_48586

theorem remainder_equiv (x : ℤ) (h : ∃ k : ℤ, x = 95 * k + 31) : ∃ m : ℤ, x = 19 * m + 12 := 
sorry

end remainder_equiv_l48_48586


namespace tree_height_at_2_years_l48_48607

theorem tree_height_at_2_years (h : ℕ → ℚ) (h5 : h 5 = 243) 
  (h_rec : ∀ n, h (n - 1) = h n / 3) : h 2 = 9 :=
  sorry

end tree_height_at_2_years_l48_48607


namespace alcohol_solution_l48_48594

/-- 
A 40-liter solution of alcohol and water is 5 percent alcohol. If 3.5 liters of alcohol and 6.5 liters of water are added to this solution, 
what percent of the solution produced is alcohol? 
-/
theorem alcohol_solution (original_volume : ℝ) (original_percent_alcohol : ℝ)
                        (added_alcohol : ℝ) (added_water : ℝ) :
  original_volume = 40 →
  original_percent_alcohol = 5 →
  added_alcohol = 3.5 →
  added_water = 6.5 →
  (100 * (original_volume * original_percent_alcohol / 100 + added_alcohol) / (original_volume + added_alcohol + added_water)) = 11 := 
by 
  intros h1 h2 h3 h4
  sorry

end alcohol_solution_l48_48594


namespace sampling_correct_l48_48633

def systematic_sampling (total_students : Nat) (num_selected : Nat) (interval : Nat) (start : Nat) : List Nat :=
  (List.range num_selected).map (λ i => start + i * interval)

theorem sampling_correct :
  systematic_sampling 60 6 10 3 = [3, 13, 23, 33, 43, 53] :=
by
  sorry

end sampling_correct_l48_48633


namespace sin_2012_eq_neg_sin_32_l48_48803

theorem sin_2012_eq_neg_sin_32 : Real.sin (2012 * Real.pi / 180) = - Real.sin (32 * Real.pi / 180) :=
by
  sorry

end sin_2012_eq_neg_sin_32_l48_48803


namespace compare_variables_l48_48149

theorem compare_variables (a b c : ℝ) (h1 : a = 2 ^ (1 / 2)) (h2 : b = Real.log 3 / Real.log π) (h3 : c = Real.log (1 / 3) / Real.log 2) : 
  a > b ∧ b > c :=
by
  sorry

end compare_variables_l48_48149


namespace candy_per_smaller_bag_l48_48613

-- Define the variables and parameters
def george_candy : ℕ := 648
def friends : ℕ := 3
def total_people : ℕ := friends + 1
def smaller_bags : ℕ := 8

-- Define the theorem
theorem candy_per_smaller_bag : (george_candy / total_people) / smaller_bags = 20 :=
by
  -- Assume the proof steps, not required to actually complete
  sorry

end candy_per_smaller_bag_l48_48613


namespace remainder_5_pow_2023_mod_11_l48_48374

theorem remainder_5_pow_2023_mod_11 : (5^2023) % 11 = 4 :=
by
  have h1 : 5^2 % 11 = 25 % 11 := sorry
  have h2 : 25 % 11 = 3 := sorry
  have h3 : (3^5) % 11 = 1 := sorry
  have h4 : 3^1011 % 11 = ((3^5)^202 * 3) % 11 := sorry
  have h5 : ((3^5)^202 * 3) % 11 = (1^202 * 3) % 11 := sorry
  have h6 : (1^202 * 3) % 11 = 3 % 11 := sorry
  have h7 : (5^2023) % 11 = (3 * 5) % 11 := sorry
  have h8 : (3 * 5) % 11 = 15 % 11 := sorry
  have h9 : 15 % 11 = 4 := sorry
  exact h9

end remainder_5_pow_2023_mod_11_l48_48374


namespace no_simultaneous_inequalities_l48_48907

theorem no_simultaneous_inequalities (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  ¬ ((a + b < c + d) ∧ ((a + b) * (c + d) < a * b + c * d) ∧ ((a + b) * c * d < a * b * (c + d))) :=
by
  sorry

end no_simultaneous_inequalities_l48_48907


namespace intersection_eq_l48_48802

def S : Set ℝ := { x | x > -2 }
def T : Set ℝ := { x | -4 ≤ x ∧ x ≤ 1 }

theorem intersection_eq : S ∩ T = { x | -2 < x ∧ x ≤ 1 } :=
by
  simp [S, T]
  sorry

end intersection_eq_l48_48802


namespace probability_three_specific_cards_l48_48093

theorem probability_three_specific_cards :
  let deck_size := 52
  let diamonds := 13
  let spades := 13
  let hearts := 13
  let p1 := diamonds / deck_size
  let p2 := spades / (deck_size - 1)
  let p3 := hearts / (deck_size - 2)
  p1 * p2 * p3 = 169 / 5100 :=
by
  sorry

end probability_three_specific_cards_l48_48093


namespace correct_statement_D_l48_48544

def is_correct_option (n : ℕ) := n = 4

theorem correct_statement_D : is_correct_option 4 :=
  sorry

end correct_statement_D_l48_48544


namespace percent_of_x_is_y_l48_48294

variable (x y : ℝ)

theorem percent_of_x_is_y
  (h : 0.5 * (x - y) = 0.2 * (x + y)) :
  y = 0.4286 * x := by
  sorry

end percent_of_x_is_y_l48_48294


namespace derivative_at_x1_is_12_l48_48123

theorem derivative_at_x1_is_12 : 
  (deriv (fun x : ℝ => (2 * x + 1) ^ 2) 1) = 12 :=
by
  sorry

end derivative_at_x1_is_12_l48_48123


namespace tan_difference_l48_48286

theorem tan_difference (x y : ℝ) (hx : Real.tan x = 3) (hy : Real.tan y = 2) :
  Real.tan (x - y) = 1 / 7 := 
  sorry

end tan_difference_l48_48286


namespace water_volume_per_minute_l48_48226

theorem water_volume_per_minute 
  (depth : ℝ) (width : ℝ) (flow_kmph : ℝ)
  (h_depth : depth = 8) (h_width : width = 25) (h_flow_rate : flow_kmph = 8) :
  (width * depth * (flow_kmph * 1000 / 60)) = 26666.67 :=
by 
  have flow_m_per_min := flow_kmph * 1000 / 60
  have area := width * depth
  have volume_per_minute := area * flow_m_per_min
  sorry

end water_volume_per_minute_l48_48226


namespace max_min_diff_half_dollars_l48_48884

-- Definitions based only on conditions
variables (a c d : ℕ)

-- Conditions:
def condition1 : Prop := a + c + d = 60
def condition2 : Prop := 5 * a + 25 * c + 50 * d = 1000

-- The mathematically equivalent proof statement
theorem max_min_diff_half_dollars : condition1 a c d → condition2 a c d → (∃ d_min d_max : ℕ, d_min = 0 ∧ d_max = 15 ∧ d_max - d_min = 15) :=
by
  intros
  sorry

end max_min_diff_half_dollars_l48_48884


namespace mary_needs_6_cups_l48_48233
-- We import the whole Mathlib library first.

-- We define the conditions and the question.
def total_cups : ℕ := 8
def cups_added : ℕ := 2
def cups_needed : ℕ := total_cups - cups_added

-- We state the theorem we need to prove.
theorem mary_needs_6_cups : cups_needed = 6 :=
by
  -- We use a placeholder for the proof.
  sorry

end mary_needs_6_cups_l48_48233


namespace solve_quintic_equation_l48_48361

theorem solve_quintic_equation :
  {x : ℝ | x * (x - 3)^2 * (5 + x) * (x^2 - 1) = 0} = {0, 3, -5, 1, -1} :=
by
  sorry

end solve_quintic_equation_l48_48361


namespace simplify_expression_l48_48033

variable (x y : ℝ)

theorem simplify_expression : (3 * x + 4 * x + 5 * y + 2 * y) = 7 * x + 7 * y :=
by
  sorry

end simplify_expression_l48_48033


namespace x_minus_y_possible_values_l48_48158

theorem x_minus_y_possible_values (x y : ℝ) (hx : x^2 = 9) (hy : |y| = 4) (hxy : x < y) : x - y = -1 ∨ x - y = -7 := 
sorry

end x_minus_y_possible_values_l48_48158


namespace Alina_messages_comparison_l48_48099

theorem Alina_messages_comparison 
  (lucia_day1 : ℕ) (alina_day1 : ℕ) (lucia_day2 : ℕ) (alina_day2 : ℕ) (lucia_day3 : ℕ) (alina_day3 : ℕ)
  (h1 : lucia_day1 = 120)
  (h2 : alina_day1 = lucia_day1 - 20)
  (h3 : lucia_day2 = lucia_day1 / 3)
  (h4 : lucia_day3 = lucia_day1)
  (h5 : alina_day3 = alina_day1)
  (h6 : lucia_day1 + lucia_day2 + lucia_day3 + alina_day1 + alina_day2 + alina_day3 = 680) :
  alina_day2 = alina_day1 + 100 :=
sorry

end Alina_messages_comparison_l48_48099


namespace graph_passes_fixed_point_l48_48502

-- Mathematical conditions
variables (a : ℝ)

-- Real numbers and conditions
def is_fixed_point (a : ℝ) : Prop :=
  a > 0 ∧ a ≠ 1 ∧ ∃ x y, (x, y) = (2, 2) ∧ y = a^(x-2) + 1

-- Lean statement for the problem
theorem graph_passes_fixed_point : is_fixed_point a :=
  sorry

end graph_passes_fixed_point_l48_48502


namespace second_man_speed_l48_48889

/-- A formal statement of the problem -/
theorem second_man_speed (v : ℝ) 
  (start_same_place : ∀ t : ℝ, t ≥ 0 → 2 * t = (10 - v) * 1) : 
  v = 8 :=
by
  sorry

end second_man_speed_l48_48889


namespace episodes_first_season_l48_48510

theorem episodes_first_season :
  ∃ (E : ℕ), (100000 * E + 200000 * (3 / 2) * E + 200000 * (3 / 2)^2 * E + 200000 * (3 / 2)^3 * E + 200000 * 24 = 16800000) ∧ E = 8 := 
by {
  sorry
}

end episodes_first_season_l48_48510


namespace part1_part2_l48_48495

-- Part 1: Determining the number of toys A and ornaments B wholesaled
theorem part1 (x y : ℕ) (h₁ : x + y = 100) (h₂ : 60 * x + 50 * y = 5650) : 
  x = 65 ∧ y = 35 := by
  sorry

-- Part 2: Determining the minimum number of toys A to wholesale for a 1400元 profit
theorem part2 (m : ℕ) (h₁ : m ≤ 100) (h₂ : (80 - 60) * m + (60 - 50) * (100 - m) ≥ 1400) : 
  m ≥ 40 := by
  sorry

end part1_part2_l48_48495


namespace option_B_is_linear_inequality_with_one_var_l48_48478

noncomputable def is_linear_inequality_with_one_var (in_eq : String) : Prop :=
  match in_eq with
  | "3x^2 > 45 - 9x" => false
  | "3x - 2 < 4" => true
  | "1 / x < 2" => false
  | "4x - 3 < 2y - 7" => false
  | _ => false

theorem option_B_is_linear_inequality_with_one_var :
  is_linear_inequality_with_one_var "3x - 2 < 4" = true :=
by
  -- Add proof steps here
  sorry

end option_B_is_linear_inequality_with_one_var_l48_48478


namespace harry_books_l48_48657

theorem harry_books : ∀ (H : ℝ), 
  (H + 2 * H + H / 2 = 175) → 
  H = 50 :=
by
  intros H h_sum
  sorry

end harry_books_l48_48657


namespace sum_of_coefficients_l48_48738

theorem sum_of_coefficients :
  (∃ a b c d e : ℤ, 512 * x ^ 3 + 27 = a * x * (c * x ^ 2 + d * x + e) + b * (c * x ^ 2 + d * x + e)) →
  (a = 8 ∧ b = 3 ∧ c = 64 ∧ d = -24 ∧ e = 9) →
  a + b + c + d + e = 60 :=
by
  intro h1 h2
  sorry

end sum_of_coefficients_l48_48738


namespace shirt_discount_l48_48541

theorem shirt_discount (original_price discounted_price : ℕ) 
  (h1 : original_price = 22) 
  (h2 : discounted_price = 16) : 
  original_price - discounted_price = 6 := 
by
  sorry

end shirt_discount_l48_48541


namespace suraya_picked_more_apples_l48_48654

theorem suraya_picked_more_apples (suraya caleb kayla : ℕ) 
  (h1 : suraya = caleb + 12)
  (h2 : caleb = kayla - 5)
  (h3 : kayla = 20) : suraya - kayla = 7 := by
  sorry

end suraya_picked_more_apples_l48_48654


namespace minimum_g7_l48_48790

def is_tenuous (g : ℕ → ℤ) : Prop :=
∀ x y : ℕ, 0 < x → 0 < y → g x + g y > x^2

noncomputable def min_possible_value_g7 (g : ℕ → ℤ) (h : is_tenuous g) 
  (h_sum : (g 1 + g 2 + g 3 + g 4 + g 5 + g 6 + g 7 + g 8 + g 9 + g 10) = 
             -29) : ℤ :=
g 7

theorem minimum_g7 (g : ℕ → ℤ) (h : is_tenuous g)
  (h_sum : (g 1 + g 2 + g 3 + g 4 + g 5 + g 6 + g 7 + g 8 + g 9 + g 10) = 
             -29) :
  min_possible_value_g7 g h h_sum = 49 :=
sorry

end minimum_g7_l48_48790


namespace det_M_pow_three_eq_twenty_seven_l48_48333

-- Define a matrix M
variables (M : Matrix (Fin n) (Fin n) ℝ)

-- Given condition: det M = 3
axiom det_M_eq_3 : Matrix.det M = 3

-- State the theorem we aim to prove
theorem det_M_pow_three_eq_twenty_seven : Matrix.det (M^3) = 27 :=
by
  sorry

end det_M_pow_three_eq_twenty_seven_l48_48333


namespace find_150th_letter_l48_48317

theorem find_150th_letter (n : ℕ) (pattern : ℕ → Char) (h : ∀ m, pattern (m % 3) = if m % 3 = 0 then 'A' else if m % 3 = 1 then 'B' else 'C') :
  pattern 149 = 'C' :=
by
  sorry

end find_150th_letter_l48_48317


namespace bianca_total_pictures_l48_48500

def album1_pictures : Nat := 27
def album2_3_4_pictures : Nat := 3 * 2

theorem bianca_total_pictures : album1_pictures + album2_3_4_pictures = 33 := by
  sorry

end bianca_total_pictures_l48_48500


namespace area_triangle_DEF_l48_48971

noncomputable def triangleDEF (DE EF DF : ℝ) (angleDEF : ℝ) : ℝ :=
  if angleDEF = 60 ∧ DF = 3 ∧ EF = 6 / Real.sqrt 3 then
    1 / 2 * DE * EF * Real.sin (Real.pi / 3)
  else
    0

theorem area_triangle_DEF :
  triangleDEF (Real.sqrt 3) (6 / Real.sqrt 3) 3 60 = 3 * Real.sqrt 3 / 2 :=
by
  sorry

end area_triangle_DEF_l48_48971


namespace trapezium_other_side_length_l48_48259

theorem trapezium_other_side_length 
  (side1 : ℝ) (perpendicular_distance : ℝ) (area : ℝ) (side1_val : side1 = 5) 
  (perpendicular_distance_val : perpendicular_distance = 6) (area_val : area = 27) : 
  ∃ other_side : ℝ, other_side = 4 :=
by
  sorry

end trapezium_other_side_length_l48_48259


namespace train_length_is_correct_l48_48242

noncomputable def length_of_train (time_in_seconds : ℝ) (relative_speed : ℝ) : ℝ :=
  relative_speed * time_in_seconds

noncomputable def relative_speed_in_mps (speed_of_train_kmph : ℝ) (speed_of_man_kmph : ℝ) : ℝ :=
  (speed_of_train_kmph + speed_of_man_kmph) * (1000 / 3600)

theorem train_length_is_correct :
  let speed_of_train_kmph := 65.99424046076315
  let speed_of_man_kmph := 6
  let time_in_seconds := 6
  length_of_train time_in_seconds (relative_speed_in_mps speed_of_train_kmph speed_of_man_kmph) = 119.9904 := by
  sorry

end train_length_is_correct_l48_48242


namespace solve_for_s_l48_48170

theorem solve_for_s (k s : ℝ) 
  (h1 : 7 = k * 3^s) 
  (h2 : 126 = k * 9^s) : 
  s = 2 + Real.log 2 / Real.log 3 := by
  sorry

end solve_for_s_l48_48170


namespace train_speed_l48_48778

theorem train_speed (length : ℝ) (time : ℝ)
  (length_pos : length = 160) (time_pos : time = 8) : 
  (length / time) * 3.6 = 72 :=
by
  sorry

end train_speed_l48_48778


namespace range_of_a_range_of_m_l48_48732

def f (x : ℝ) : ℝ := |2 * x + 1| + |2 * x - 3|

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, f x < |1 - 2 * a|) ↔ a ∈ (Set.Iic (-3/2) ∪ Set.Ici (5/2)) := by sorry

theorem range_of_m (m : ℝ) : 
  (∃ t : ℝ, t^2 - 2 * Real.sqrt 6 * t + f m = 0) ↔ m ∈ (Set.Icc (-1) 2) := by sorry

end range_of_a_range_of_m_l48_48732


namespace madeline_flower_count_l48_48525

theorem madeline_flower_count 
    (r w : ℕ) 
    (b_percent : ℝ) 
    (total : ℕ) 
    (h_r : r = 4)
    (h_w : w = 2)
    (h_b_percent : b_percent = 0.40)
    (h_total : r + w + (b_percent * total) = total) : 
    total = 10 :=
by 
    sorry

end madeline_flower_count_l48_48525


namespace tina_assignment_time_l48_48939

theorem tina_assignment_time (total_time clean_time_per_key remaining_keys assignment_time : ℕ) 
  (h1 : total_time = 52) 
  (h2 : clean_time_per_key = 3) 
  (h3 : remaining_keys = 14) 
  (h4 : assignment_time = total_time - remaining_keys * clean_time_per_key) :
  assignment_time = 10 :=
by
  rw [h1, h2, h3] at h4
  assumption

end tina_assignment_time_l48_48939


namespace hyperbola_foci_property_l48_48134

noncomputable def hyperbola (x y b : ℝ) : Prop :=
  (x^2 / 9) - (y^2 / b^2) = 1

theorem hyperbola_foci_property (x y b : ℝ) (h : hyperbola x y b) (b_pos : b > 0) (PF1 : ℝ) (PF2 : ℝ) (hPF1 : PF1 = 5) :
  PF2 = 11 :=
by
  sorry

end hyperbola_foci_property_l48_48134


namespace sum_of_midpoint_coordinates_l48_48752

theorem sum_of_midpoint_coordinates :
  let x1 := 3
  let y1 := 4
  let x2 := 9
  let y2 := 18
  let midpoint := ((x1 + x2) / 2, (y1 + y2) / 2)
  let sum_of_coordinates := midpoint.fst + midpoint.snd
  sum_of_coordinates = 17 :=
by
  let x1 := 3
  let y1 := 4
  let x2 := 9
  let y2 := 18
  let midpoint := ((x1 + x2) / 2, (y1 + y2) / 2)
  let sum_of_coordinates := midpoint.fst + midpoint.snd
  show sum_of_coordinates = 17
  sorry

end sum_of_midpoint_coordinates_l48_48752


namespace joan_gave_sam_seashells_l48_48350

theorem joan_gave_sam_seashells (original_seashells : ℕ) (left_seashells : ℕ) (given_seashells : ℕ) 
  (h1 : original_seashells = 70) (h2 : left_seashells = 27) : given_seashells = 43 :=
by
  have h3 : given_seashells = original_seashells - left_seashells := sorry
  rw [h1, h2] at h3
  exact h3

end joan_gave_sam_seashells_l48_48350


namespace simplify_expression_l48_48504

variable (b : ℤ)

theorem simplify_expression :
  (3 * b + 6 - 6 * b) / 3 = -b + 2 :=
sorry

end simplify_expression_l48_48504


namespace min_value_2013_Quanzhou_simulation_l48_48946

theorem min_value_2013_Quanzhou_simulation:
  ∃ (x y : ℝ), (x - y - 1 = 0) ∧ (x - 2) ^ 2 + (y - 2) ^ 2 = 1 / 2 :=
by
  use 2
  use 3
  sorry

end min_value_2013_Quanzhou_simulation_l48_48946


namespace FatherCandyCount_l48_48936

variables (a b c d e : ℕ)

-- Conditions
def BillyInitial := 6
def CalebInitial := 11
def AndyInitial := 9
def BillyReceived := 8
def CalebReceived := 11
def AndyHasMore := 4

-- Define number of candies Andy has now based on Caleb's candies
def AndyTotal (b c : ℕ) : ℕ := c + AndyHasMore

-- Define number of candies received by Andy
def AndyReceived (a b c d e : ℕ) : ℕ := (AndyTotal b c) - AndyInitial

-- Define total candies bought by father
def FatherBoughtCandies (d e f : ℕ) : ℕ := d + e + f

theorem FatherCandyCount : FatherBoughtCandies BillyReceived CalebReceived (AndyReceived BillyInitial CalebInitial AndyInitial BillyReceived CalebReceived)  = 36 :=
by
  sorry

end FatherCandyCount_l48_48936


namespace jill_peaches_l48_48448

variable (S J : ℕ)

theorem jill_peaches (h1 : S = 19) (h2 : S = J + 13) : J = 6 :=
by
  sorry

end jill_peaches_l48_48448


namespace cheaper_candy_price_l48_48038

theorem cheaper_candy_price
    (mix_total_weight : ℝ) (mix_price_per_pound : ℝ)
    (cheap_weight : ℝ) (expensive_weight : ℝ) (expensive_price_per_pound : ℝ)
    (cheap_total_value : ℝ) (expensive_total_value : ℝ) (total_mix_value : ℝ) :
    mix_total_weight = 80 →
    mix_price_per_pound = 2.20 →
    cheap_weight = 64 →
    expensive_weight = mix_total_weight - cheap_weight →
    expensive_price_per_pound = 3.00 →
    cheap_total_value = cheap_weight * x →
    expensive_total_value = expensive_weight * expensive_price_per_pound →
    total_mix_value = mix_total_weight * mix_price_per_pound →
    total_mix_value = cheap_total_value + expensive_total_value →
    x = 2 := 
sorry

end cheaper_candy_price_l48_48038


namespace parallel_slope_l48_48184

theorem parallel_slope {x1 y1 x2 y2 : ℝ} (h : x1 = 3 ∧ y1 = -2 ∧ x2 = 1 ∧ y2 = 5) :
    let slope := (y2 - y1) / (x2 - x1)
    slope = -7 / 2 := 
by 
    sorry

end parallel_slope_l48_48184


namespace train_cross_time_l48_48059

-- Definitions from the conditions
def length_of_train : ℤ := 600
def speed_of_man_kmh : ℤ := 2
def speed_of_train_kmh : ℤ := 56

-- Conversion factors and speed conversion
def kmh_to_mph_factor : ℤ := 1000 / 3600 -- 1 km/hr = 0.27778 m/s approximately

def speed_of_man_ms : ℤ := speed_of_man_kmh * kmh_to_mph_factor -- Convert speed of man to m/s
def speed_of_train_ms : ℤ := speed_of_train_kmh * kmh_to_mph_factor -- Convert speed of train to m/s

-- Calculating relative speed
def relative_speed_ms : ℤ := speed_of_train_ms - speed_of_man_ms

-- Calculating the time taken to cross
def time_to_cross : ℤ := length_of_train / relative_speed_ms 

-- The theorem to prove
theorem train_cross_time : time_to_cross = 40 := 
by sorry

end train_cross_time_l48_48059


namespace count_integer_triangles_with_perimeter_12_l48_48076

theorem count_integer_triangles_with_perimeter_12 : 
  ∃! (sides : ℕ × ℕ × ℕ), sides.1 + sides.2.1 + sides.2.2 = 12 ∧ sides.1 + sides.2.1 > sides.2.2 ∧ sides.1 + sides.2.2 > sides.2.1 ∧ sides.2.1 + sides.2.2 > sides.1 ∧
  (sides = (2, 5, 5) ∨ sides = (3, 4, 5) ∨ sides = (4, 4, 4)) :=
by 
  exists 3
  sorry

end count_integer_triangles_with_perimeter_12_l48_48076


namespace james_needs_to_sell_12_coins_l48_48416

theorem james_needs_to_sell_12_coins:
  ∀ (num_coins : ℕ) (initial_price new_price : ℝ),
  num_coins = 20 ∧ initial_price = 15 ∧ new_price = initial_price + (2 / 3) * initial_price →
  (num_coins * initial_price) / new_price = 12 :=
by
  intros num_coins initial_price new_price h
  obtain ⟨hc1, hc2, hc3⟩ := h
  sorry

end james_needs_to_sell_12_coins_l48_48416


namespace find_x_from_triangle_area_l48_48947

theorem find_x_from_triangle_area :
  ∀ (x : ℝ), x > 0 ∧ (1 / 2) * x * 3 * x = 96 → x = 8 :=
by
  intros x hx
  -- The proof goes here
  sorry

end find_x_from_triangle_area_l48_48947


namespace min_value_l48_48794

theorem min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = a + b) : a + 4 * b ≥ 9 :=
sorry

end min_value_l48_48794


namespace allocation_of_fabric_l48_48390

theorem allocation_of_fabric (x : ℝ) (y : ℝ) 
  (fabric_for_top : 3 * x = 2 * x)
  (fabric_for_pants : 3 * y = 3 * (600 - x))
  (total_fabric : x + y = 600)
  (sets_match : (x / 3) * 2 = (y / 3) * 3) : 
  x = 360 ∧ y = 240 := 
by
  sorry

end allocation_of_fabric_l48_48390


namespace number_multiplied_by_any_integer_results_in_itself_l48_48551

theorem number_multiplied_by_any_integer_results_in_itself (N : ℤ) (h : ∀ (x : ℤ), N * x = N) : N = 0 :=
  sorry

end number_multiplied_by_any_integer_results_in_itself_l48_48551


namespace solution_set_of_f_neg_2x_l48_48482

def f (a b x : ℝ) : ℝ := (a * x - 1) * (x + b)

theorem solution_set_of_f_neg_2x (a b : ℝ) (hf_sol : ∀ x : ℝ, (a * x - 1) * (x + b) > 0 ↔ -1 < x ∧ x < 3) :
  ∀ x : ℝ, f a b (-2 * x) < 0 ↔ (x < -3/2 ∨ x > 1/2) :=
by
  sorry

end solution_set_of_f_neg_2x_l48_48482


namespace chromium_alloy_l48_48194

theorem chromium_alloy (x : ℝ) (h1 : 0.12 * x + 0.10 * 35 = 0.106 * (x + 35)) : x = 15 := 
by 
  -- statement only, no proof required.
  sorry

end chromium_alloy_l48_48194


namespace problem_l48_48881

def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

def sum_arithmetic_sequence (a d n : ℕ) : ℕ := n * a + (n * (n - 1) * d) / 2

theorem problem (a1 S3 : ℕ) (a1_eq : a1 = 2) (S3_eq : S3 = 12) : 
  ∃ a6 : ℕ, a6 = 12 := by
  let a2 := (S3 - a1) / 2
  let d := a2 - a1
  let a6 := a1 + 5 * d
  use a6
  sorry

end problem_l48_48881


namespace Sophia_fraction_finished_l48_48776

/--
Sophia finished a fraction of a book.
She calculated that she finished 90 more pages than she has yet to read.
Her book is 270.00000000000006 pages long.
Prove that the fraction of the book she finished is 2/3.
-/
theorem Sophia_fraction_finished :
  let total_pages : ℝ := 270.00000000000006
  let yet_to_read : ℝ := (total_pages - 90) / 2
  let finished_pages : ℝ := yet_to_read + 90
  finished_pages / total_pages = 2 / 3 :=
by
  sorry

end Sophia_fraction_finished_l48_48776


namespace suitable_comprehensive_survey_l48_48106

-- Definitions based on conditions

def heights_of_students (n : Nat) : Prop := n = 45
def disease_rate_wheat (area : Type) : Prop := True
def love_for_chrysanthemums (population : Type) : Prop := True
def food_safety_hotel (time : Type) : Prop := True

-- The theorem to prove

theorem suitable_comprehensive_survey : 
  (heights_of_students 45 → True) ∧ 
  (disease_rate_wheat ℕ → False) ∧ 
  (love_for_chrysanthemums ℕ → False) ∧ 
  (food_safety_hotel ℕ → False) →
  heights_of_students 45 :=
by
  intros
  sorry

end suitable_comprehensive_survey_l48_48106


namespace parabola_line_intersection_l48_48005

theorem parabola_line_intersection (x1 x2 : ℝ) (h1 : x1 * x2 = 1) (h2 : x1 + 1 = 4) : x2 + 1 = 4 / 3 :=
by
  sorry

end parabola_line_intersection_l48_48005


namespace log_m_n_iff_m_minus_1_n_minus_1_l48_48717

theorem log_m_n_iff_m_minus_1_n_minus_1 (m n : ℝ) (h1 : m > 0) (h2 : m ≠ 1) (h3 : n > 0) :
  (Real.log n / Real.log m < 0) ↔ ((m - 1) * (n - 1) < 0) :=
sorry

end log_m_n_iff_m_minus_1_n_minus_1_l48_48717


namespace round_to_nearest_tenth_l48_48296

theorem round_to_nearest_tenth : 
  let x := 36.89753 
  let tenth_place := 8
  let hundredth_place := 9
  (hundredth_place > 5) → (Float.round (10 * x) / 10 = 36.9) := 
by
  intros x tenth_place hundredth_place h
  sorry

end round_to_nearest_tenth_l48_48296


namespace equal_numbers_product_l48_48372

theorem equal_numbers_product :
  ∀ (a b c d : ℕ), 
  (a + b + c + d = 80) → 
  (a = 12) → 
  (b = 22) → 
  (c = d) → 
  (c * d = 529) :=
by
  intros a b c d hsum ha hb hcd
  -- proof skipped
  sorry

end equal_numbers_product_l48_48372


namespace polynomial_remainder_l48_48131

theorem polynomial_remainder (p q r : Polynomial ℝ) (h1 : p.eval 2 = 6) (h2 : p.eval 4 = 14)
  (r_deg : r.degree < 2) :
  p = q * (X - 2) * (X - 4) + r → r = 4 * X - 2 :=
by
  sorry

end polynomial_remainder_l48_48131


namespace area_of_triangle_l48_48534

theorem area_of_triangle :
  let A := (10, 1)
  let B := (15, 8)
  let C := (10, 8)
  ∃ (area : ℝ), 
  area = 17.5 ∧ 
  area = 1 / 2 * (abs (B.1 - C.1)) * (abs (C.2 - A.2)) :=
by
  sorry

end area_of_triangle_l48_48534


namespace value_of_f_2011_l48_48546

noncomputable def f (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 7

theorem value_of_f_2011 (a b c : ℝ) (h : f a b c (-2011) = -17) : f a b c 2011 = 31 :=
by {
  sorry
}

end value_of_f_2011_l48_48546


namespace jake_sold_tuesday_correct_l48_48901

def jake_initial_pieces : ℕ := 80
def jake_sold_monday : ℕ := 15
def jake_remaining_wednesday : ℕ := 7

def pieces_sold_tuesday (initial : ℕ) (sold_monday : ℕ) (remaining_wednesday : ℕ) : ℕ :=
  initial - sold_monday - remaining_wednesday

theorem jake_sold_tuesday_correct :
  pieces_sold_tuesday jake_initial_pieces jake_sold_monday jake_remaining_wednesday = 58 :=
by
  unfold pieces_sold_tuesday
  norm_num
  sorry

end jake_sold_tuesday_correct_l48_48901


namespace find_a_l48_48675

noncomputable def f (x : ℝ) : ℝ := x^2 + 10

noncomputable def g (x : ℝ) : ℝ := x^2 - 6

theorem find_a (a : ℝ) (h₀ : a > 0) (h₁ : f (g a) = 12) :
    a = Real.sqrt (6 + Real.sqrt 2) ∨ a = Real.sqrt (6 - Real.sqrt 2) :=
sorry

end find_a_l48_48675


namespace complement_of_A_eq_l48_48820

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x > 1}

theorem complement_of_A_eq {U : Set ℝ} (U_eq : U = Set.univ) {A : Set ℝ} (A_eq : A = {x | x > 1}) :
    U \ A = {x | x ≤ 1} :=
by
  sorry

end complement_of_A_eq_l48_48820


namespace circle_radius_c_eq_32_l48_48208

theorem circle_radius_c_eq_32 :
  ∃ c : ℝ, (∀ x y : ℝ, x^2 - 8*x + y^2 + 10*y + c = 0 ↔ (x-4)^2 + (y+5)^2 = 9) :=
by
  use 32
  sorry

end circle_radius_c_eq_32_l48_48208


namespace vectors_parallel_l48_48048

theorem vectors_parallel (m n : ℝ) (k : ℝ) (h1 : 2 = k * 1) (h2 : -1 = k * m) (h3 : 2 = k * n) : 
  m + n = 1 / 2 := 
by
  sorry

end vectors_parallel_l48_48048


namespace rate_of_mixed_oil_l48_48673

/--
If 10 litres of an oil at Rs. 50 per litre is mixed with 5 litres of another oil at Rs. 68 per litre, 
8 litres of a third oil at Rs. 42 per litre, and 7 litres of a fourth oil at Rs. 62 per litre, 
then the rate of the mixed oil per litre is Rs. 53.67.
-/
theorem rate_of_mixed_oil :
  let cost1 := 10 * 50
  let cost2 := 5 * 68
  let cost3 := 8 * 42
  let cost4 := 7 * 62
  let total_cost := cost1 + cost2 + cost3 + cost4
  let total_volume := 10 + 5 + 8 + 7
  let rate_per_litre := total_cost / total_volume
  rate_per_litre = 53.67 :=
by
  intros
  sorry

end rate_of_mixed_oil_l48_48673


namespace solve_equation_l48_48728

noncomputable def equation (x : ℝ) : ℝ :=
(13 * x - x^2) / (x + 1) * (x + (13 - x) / (x + 1))

theorem solve_equation :
  equation 1 = 42 ∧ equation 6 = 42 ∧ equation (3 + Real.sqrt 2) = 42 ∧ equation (3 - Real.sqrt 2) = 42 :=
by
  sorry

end solve_equation_l48_48728


namespace min_value_f_l48_48774

open Real

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (exp x - 1) + x

theorem min_value_f {x0 : ℝ} (hx0 : 0 < x0) (hx0_min : ∀ x > 0, f x ≥ f x0) :
  f x0 = x0 + 1 ∧ f x0 < 3 :=
by sorry

end min_value_f_l48_48774


namespace factorize_expression_l48_48004

theorem factorize_expression (x : ℝ) : 9 * x^3 - 18 * x^2 + 9 * x = 9 * x * (x - 1)^2 := 
by 
    sorry

end factorize_expression_l48_48004


namespace entrance_exit_ways_equal_49_l48_48023

-- Define the number of gates on each side
def south_gates : ℕ := 4
def north_gates : ℕ := 3

-- Define the total number of gates
def total_gates : ℕ := south_gates + north_gates

-- State the theorem and provide the expected proof structure
theorem entrance_exit_ways_equal_49 : (total_gates * total_gates) = 49 := 
by {
  sorry
}

end entrance_exit_ways_equal_49_l48_48023


namespace belongs_to_one_progression_l48_48376

-- Define the arithmetic progression and membership property
def is_arith_prog (P : ℕ → Prop) : Prop :=
  ∃ a d, ∀ n, P (a + n * d)

-- Define the given conditions
def condition (P1 P2 P3 : ℕ → Prop) : Prop :=
  is_arith_prog P1 ∧ is_arith_prog P2 ∧ is_arith_prog P3 ∧
  (P1 1 ∨ P2 1 ∨ P3 1) ∧
  (P1 2 ∨ P2 2 ∨ P3 2) ∧
  (P1 3 ∨ P2 3 ∨ P3 3) ∧
  (P1 4 ∨ P2 4 ∨ P3 4) ∧
  (P1 5 ∨ P2 5 ∨ P3 5) ∧
  (P1 6 ∨ P2 6 ∨ P3 6) ∧
  (P1 7 ∨ P2 7 ∨ P3 7) ∧
  (P1 8 ∨ P2 8 ∨ P3 8)

-- Statement to prove
theorem belongs_to_one_progression (P1 P2 P3 : ℕ → Prop) (h : condition P1 P2 P3) : 
  P1 1980 ∨ P2 1980 ∨ P3 1980 := 
by
sorry

end belongs_to_one_progression_l48_48376


namespace standard_lamp_probability_l48_48223

-- Define the given probabilities
def P_A1 : ℝ := 0.45
def P_A2 : ℝ := 0.40
def P_A3 : ℝ := 0.15

def P_B_given_A1 : ℝ := 0.70
def P_B_given_A2 : ℝ := 0.80
def P_B_given_A3 : ℝ := 0.81

-- Define the calculation for the total probability of B
def P_B : ℝ := P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2 + P_A3 * P_B_given_A3

-- The statement to prove
theorem standard_lamp_probability : P_B = 0.7565 := by sorry

end standard_lamp_probability_l48_48223


namespace inequality_holds_iff_m_eq_n_l48_48917

theorem inequality_holds_iff_m_eq_n (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (∀ (α β : ℝ), 
    ⌊(m + n) * α⌋ + ⌊(m + n) * β⌋ ≥ 
    ⌊m * α⌋ + ⌊m * β⌋ + ⌊n * (α + β)⌋) ↔ m = n :=
by
  sorry

end inequality_holds_iff_m_eq_n_l48_48917


namespace evaluate_expression_l48_48164

theorem evaluate_expression : (125^(1/3 : ℝ)) * (81^(-1/4 : ℝ)) * (32^(1/5 : ℝ)) = (10 / 3 : ℝ) :=
by
  sorry

end evaluate_expression_l48_48164


namespace lastNumberIsOneOverSeven_l48_48414

-- Definitions and conditions
def seq (a : ℕ → ℝ) : Prop :=
  ∀ k : ℕ, 2 ≤ k ∧ k ≤ 99 → a k = a (k - 1) * a (k + 1)

def nonZeroSeq (a : ℕ → ℝ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 100 → a k ≠ 0

def firstSeq7 (a : ℕ → ℝ) : Prop :=
  a 1 = 7

-- Theorem statement
theorem lastNumberIsOneOverSeven (a : ℕ → ℝ) :
  seq a → nonZeroSeq a → firstSeq7 a → a 100 = 1 / 7 :=
by
  sorry

end lastNumberIsOneOverSeven_l48_48414


namespace fraction_integer_condition_special_integers_l48_48634

theorem fraction_integer_condition (p : ℕ) (h : (p + 2) % (p + 1) = 0) : p = 2 :=
by
  sorry

theorem special_integers (N : ℕ) (h1 : ∀ q : ℕ, N = 2 ^ p * 3 ^ q ∧ (2 * p + 1) * (2 * q + 1) = 3 * (p + 1) * (q + 1)) : 
  N = 144 ∨ N = 324 :=
by
  sorry

end fraction_integer_condition_special_integers_l48_48634


namespace deny_evenness_l48_48814

-- We need to define the natural numbers and their parity.
variables {a b c : ℕ}

-- Define what it means for a number to be odd and even.
def is_odd (n : ℕ) := ∃ k, n = 2 * k + 1
def is_even (n : ℕ) := ∃ k, n = 2 * k

-- The Lean theorem statement translating the given problem.
theorem deny_evenness :
  (is_odd a ∧ is_odd b ∧ is_odd c) → ¬(is_even a ∨ is_even b ∨ is_even c) :=
by sorry

end deny_evenness_l48_48814


namespace probability_interval_l48_48804

theorem probability_interval (P_A P_B : ℚ) (h1 : P_A = 5/6) (h2 : P_B = 3/4) :
  ∃ p : ℚ, (5/12 ≤ p ∧ p ≤ 3/4) :=
sorry

end probability_interval_l48_48804


namespace triangle_lengths_ce_l48_48243

theorem triangle_lengths_ce (AE BE CE : ℝ) (angle_AEB angle_BEC angle_CED : ℝ) (h1 : angle_AEB = 30)
  (h2 : angle_BEC = 45) (h3 : angle_CED = 45) (h4 : AE = 30) (h5 : BE = AE / 2) (h6 : CE = BE) : CE = 15 :=
by sorry

end triangle_lengths_ce_l48_48243


namespace quadrilateral_with_exactly_two_axes_of_symmetry_is_either_rectangle_or_rhombus_l48_48117

structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

def has_exactly_two_axes_of_symmetry (q : Quadrilateral) : Prop :=
  -- Definition to be developed further based on symmetry conditions
  sorry

def is_rectangle (q : Quadrilateral) : Prop :=
  -- Definition to be developed further based on properties of rectangle
  sorry

def is_rhombus (q : Quadrilateral) : Prop :=
  -- Definition to be developed further based on properties of rhombus
  sorry

theorem quadrilateral_with_exactly_two_axes_of_symmetry_is_either_rectangle_or_rhombus
  (q : Quadrilateral)
  (h : has_exactly_two_axes_of_symmetry q) :
  is_rectangle q ∨ is_rhombus q := by
  sorry

end quadrilateral_with_exactly_two_axes_of_symmetry_is_either_rectangle_or_rhombus_l48_48117


namespace odd_function_k_eq_neg_one_f_x_greater_2_neg_x_k_gt_zero_l48_48337

noncomputable def f (x : ℝ) (k : ℝ) := 2^x + k * 2^(-x)

-- Prove that if f(x) is an odd function, then k = -1.
theorem odd_function_k_eq_neg_one {k : ℝ} (h : ∀ x, f x k = -f (-x) k) : k = -1 :=
by sorry

-- Prove that if for all x in [0, +∞), f(x) > 2^(-x), then k > 0.
theorem f_x_greater_2_neg_x_k_gt_zero {k : ℝ} (h : ∀ x, 0 ≤ x → f x k > 2^(-x)) : k > 0 :=
by sorry

end odd_function_k_eq_neg_one_f_x_greater_2_neg_x_k_gt_zero_l48_48337


namespace investment_time_R_l48_48206

theorem investment_time_R (x t : ℝ) 
  (h1 : 7 * 5 * x / (5 * 7 * x) = 7 / 9)
  (h2 : 3 * t * x / (5 * 7 * x) = 4 / 9) : 
  t = 140 / 27 :=
by
  -- Placeholder for the proof, which is not required in this step.
  sorry

end investment_time_R_l48_48206


namespace polynomial_irreducible_if_not_divisible_by_5_l48_48176

theorem polynomial_irreducible_if_not_divisible_by_5 (k : ℤ) (h1 : ¬ ∃ m : ℤ, k = 5 * m) :
    ¬ ∃ (f g : Polynomial ℤ), (f.degree < 5) ∧ (f * g = x^5 - x + Polynomial.C k) :=
  sorry

end polynomial_irreducible_if_not_divisible_by_5_l48_48176


namespace part1_l48_48834

theorem part1 : 2 * Real.tan (60 * Real.pi / 180) * Real.cos (30 * Real.pi / 180) - (Real.sin (45 * Real.pi / 180)) ^ 2 = 5 / 2 := 
sorry

end part1_l48_48834


namespace marble_solid_color_percentage_l48_48301

theorem marble_solid_color_percentage (a b : ℕ) (h1 : a = 5) (h2 : b = 85) : a + b = 90 := 
by
  sorry

end marble_solid_color_percentage_l48_48301


namespace possible_ratios_of_distances_l48_48255

theorem possible_ratios_of_distances (a b : ℝ) (h : a > b) (h1 : ∃ points : Fin 4 → ℝ × ℝ, 
  ∀ (i j : Fin 4), i ≠ j → 
  (dist (points i) (points j) = a ∨ dist (points i) (points j) = b )) :
  a / b = Real.sqrt 2 ∨ 
  a / b = (1 + Real.sqrt 5) / 2 ∨ 
  a / b = Real.sqrt 3 ∨ 
  a / b = Real.sqrt (2 + Real.sqrt 3) :=
by 
  sorry

end possible_ratios_of_distances_l48_48255


namespace range_of_a_l48_48408

noncomputable def log_a (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → x < 1 → log_a a (2 - a * x) < log_a a (2 - a * (x / 2))) →
  1 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l48_48408


namespace difference_between_x_and_y_l48_48200

theorem difference_between_x_and_y 
  (x y : ℕ) 
  (h1 : 3 ^ x * 4 ^ y = 531441) 
  (h2 : x = 12) : x - y = 12 := 
by 
  sorry

end difference_between_x_and_y_l48_48200


namespace avg_ticket_cost_per_person_l48_48777

-- Define the conditions
def full_price : ℤ := 150
def half_price : ℤ := full_price / 2
def num_full_price_tickets : ℤ := 2
def num_half_price_tickets : ℤ := 2
def free_tickets : ℤ := 1
def total_people : ℤ := 5

-- Prove that the average cost of tickets per person is 90 yuan
theorem avg_ticket_cost_per_person : ((num_full_price_tickets * full_price + num_half_price_tickets * half_price) / total_people) = 90 := 
by 
  sorry

end avg_ticket_cost_per_person_l48_48777


namespace angle_between_line_and_plane_l48_48658

variables (α β : ℝ) -- angles in radians
-- Definitions to capture the provided conditions
def dihedral_angle (α : ℝ) : Prop := true -- The angle between the planes γ₁ and γ₂
def angle_with_edge (β : ℝ) : Prop := true -- The angle between line AB and edge l

-- The angle between line AB and the plane γ₂
theorem angle_between_line_and_plane (α β : ℝ) (h1 : dihedral_angle α) (h2 : angle_with_edge β) : 
  ∃ θ : ℝ, θ = Real.arcsin (Real.sin α * Real.sin β) :=
by
  sorry

end angle_between_line_and_plane_l48_48658


namespace tournament_player_count_l48_48425

theorem tournament_player_count (n : ℕ) :
  (∃ points_per_game : ℕ, points_per_game = (n * (n - 1)) / 2) →
  (∃ T : ℕ, T = 90) →
  (n * (n - 1)) / 4 = 90 →
  n = 19 :=
by
  intros h1 h2 h3
  sorry

end tournament_player_count_l48_48425


namespace simplify_exponentiation_l48_48716

theorem simplify_exponentiation (x : ℕ) :
  (x^5 * x^3)^2 = x^16 := 
by {
  sorry -- proof will go here
}

end simplify_exponentiation_l48_48716


namespace total_movies_seen_l48_48687

theorem total_movies_seen (d h a c : ℕ) (hd : d = 7) (hh : h = 12) (ha : a = 15) (hc : c = 2) :
  (c + (d - c) + (h - c) + (a - c)) = 30 :=
by
  sorry

end total_movies_seen_l48_48687


namespace missing_jar_size_l48_48119

theorem missing_jar_size (x : ℕ) (h₁ : 3 * 16 + 3 * x + 3 * 40 = 252) 
                          (h₂ : 3 + 3 + 3 = 9) : x = 28 := 
by 
  sorry

end missing_jar_size_l48_48119


namespace find_two_digit_number_l48_48148

theorem find_two_digit_number : ∃ (y : ℕ), (10 ≤ y ∧ y < 100) ∧ (∃ x : ℕ, x = (y / 10) + (y % 10) ∧ x^3 = y^2) ∧ y = 27 := 
by
  sorry

end find_two_digit_number_l48_48148


namespace geom_seq_sum_2016_2017_l48_48579

noncomputable def geom_seq (n : ℕ) (a1 q : ℝ) : ℝ := a1 * q ^ (n - 1)

noncomputable def sum_geometric_seq (a1 q : ℝ) (n : ℕ) : ℝ :=
if q = 1 then
  a1 * n
else
  a1 * (1 - q ^ n) / (1 - q)

theorem geom_seq_sum_2016_2017 :
  (a1 = 2) →
  (geom_seq 2 a1 q + geom_seq 5 a1 q = 0) →
  sum_geometric_seq a1 q 2016 + sum_geometric_seq a1 q 2017 = 2 :=
by
  sorry

end geom_seq_sum_2016_2017_l48_48579


namespace combined_height_difference_is_correct_l48_48527

-- Define the initial conditions
def uncle_height : ℕ := 72
def james_initial_height : ℕ := (2 * uncle_height) / 3
def sarah_initial_height : ℕ := (3 * james_initial_height) / 4

-- Define the growth spurts
def james_growth_spurt : ℕ := 10
def sarah_growth_spurt : ℕ := 12

-- Define their heights after growth spurts
def james_final_height : ℕ := james_initial_height + james_growth_spurt
def sarah_final_height : ℕ := sarah_initial_height + sarah_growth_spurt

-- Define the combined height of James and Sarah after growth spurts
def combined_height : ℕ := james_final_height + sarah_final_height

-- Define the combined height difference between uncle and both James and Sarah now
def combined_height_difference : ℕ := combined_height - uncle_height

-- Lean statement to prove the combined height difference
theorem combined_height_difference_is_correct : combined_height_difference = 34 := by
  -- proof omitted
  sorry

end combined_height_difference_is_correct_l48_48527


namespace sum_of_reciprocals_is_five_l48_48365

theorem sum_of_reciprocals_is_five (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x - y = 3 * x * y) : 
  (1 / x) + (1 / y) = 5 :=
sorry

end sum_of_reciprocals_is_five_l48_48365


namespace arithmetic_seq_solution_l48_48036

theorem arithmetic_seq_solution (a : ℕ → ℝ) (n : ℕ) (S : ℕ → ℝ) 
  (h_nonzero : ∀ n, a n ≠ 0) 
  (h_arith : ∀ n ≥ 2, a (n+1) - a n ^ 2 + a (n-1) = 0) 
  (h_sum : ∀ k, S k = (k * (a 1 + a k)) / 2) :
  S (2 * n - 1) - 4 * n = -2 := 
sorry

end arithmetic_seq_solution_l48_48036


namespace largest_divisible_by_3_power_l48_48407

theorem largest_divisible_by_3_power :
  ∃ n : ℕ, (∀ k : ℕ, 1 ≤ k ∧ k ≤ 100 → ∃ m : ℕ, (3^m ∣ (2*k - 1)) → n = 49) :=
sorry

end largest_divisible_by_3_power_l48_48407


namespace rhombus_area_l48_48517

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 20) :
  (d1 * d2) / 2 = 120 :=
by
  sorry

end rhombus_area_l48_48517


namespace number_of_possible_lists_l48_48163

/-- 
Define the basic conditions: 
- 18 balls, numbered 1 through 18
- Selection process is repeated 4 times 
- Each selection is independent
- After each selection, the ball is replaced 
- We need to prove the total number of possible lists of four numbers 
--/
def number_of_balls : ℕ := 18
def selections : ℕ := 4

theorem number_of_possible_lists : (number_of_balls ^ selections) = 104976 := by
  sorry

end number_of_possible_lists_l48_48163


namespace range_of_a_l48_48791

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x^2 + 2*a*x + a) > 0) → (0 < a ∧ a < 1) :=
sorry

end range_of_a_l48_48791


namespace probability_X_l48_48927

theorem probability_X (P : ℕ → ℚ) (h1 : P 1 = 1/10) (h2 : P 2 = 2/10) (h3 : P 3 = 3/10) (h4 : P 4 = 4/10) :
  P 2 + P 3 = 1/2 :=
by
  sorry

end probability_X_l48_48927


namespace leaf_raking_earnings_l48_48098

variable {S M L P : ℕ}

theorem leaf_raking_earnings (h1 : 5 * 4 + 7 * 2 + 10 * 1 + 3 * 1 = 47)
                             (h2 : 5 * 2 + 3 * 1 + 7 * 1 + 10 * 2 = 40)
                             (h3 : 163 - 87 = 76) :
  5 * S + 7 * M + 10 * L + 3 * P = 76 :=
by
  sorry

end leaf_raking_earnings_l48_48098


namespace correct_substitution_l48_48027

theorem correct_substitution (x y : ℝ) 
  (h1 : y = 1 - x) 
  (h2 : x - 2 * y = 4) : x - 2 + 2 * x = 4 :=
by
  sorry

end correct_substitution_l48_48027


namespace Bethany_total_riding_hours_l48_48575

-- Define daily riding hours
def Monday_hours : Nat := 1
def Wednesday_hours : Nat := 1
def Friday_hours : Nat := 1
def Tuesday_hours : Nat := 1 / 2
def Thursday_hours : Nat := 1 / 2
def Saturday_hours : Nat := 2

-- Define total weekly hours
def weekly_hours : Nat :=
  Monday_hours + Wednesday_hours + Friday_hours + (Tuesday_hours + Thursday_hours) + Saturday_hours

-- Definition to account for the 2-week period
def total_hours (weeks : Nat) : Nat := weeks * weekly_hours

-- Prove that Bethany rode 12 hours over 2 weeks
theorem Bethany_total_riding_hours : total_hours 2 = 12 := by
  sorry

end Bethany_total_riding_hours_l48_48575


namespace choose_lines_intersect_l48_48985

-- We need to define the proof problem
theorem choose_lines_intersect : 
  ∃ (lines : ℕ → ℝ × ℝ → ℝ), 
    (∀ i j, i < 100 ∧ j < 100 ∧ i ≠ j → (lines i = lines j) → ∃ (p : ℕ), p = 2022) :=
sorry

end choose_lines_intersect_l48_48985


namespace chess_tournament_participants_l48_48114

open Int

theorem chess_tournament_participants (n : ℕ) (h_games: n * (n - 1) / 2 = 190) : n = 20 :=
by
  sorry

end chess_tournament_participants_l48_48114


namespace janet_extra_flowers_l48_48892

-- Define the number of flowers Janet picked for each type
def tulips : ℕ := 5
def roses : ℕ := 10
def daisies : ℕ := 8
def lilies : ℕ := 4

-- Define the number of flowers Janet used
def used : ℕ := 19

-- Calculate the total number of flowers Janet picked
def total_picked : ℕ := tulips + roses + daisies + lilies

-- Calculate the number of extra flowers
def extra_flowers : ℕ := total_picked - used

-- The theorem to be proven
theorem janet_extra_flowers : extra_flowers = 8 :=
by
  -- You would provide the proof here, but it's not required as per instructions
  sorry

end janet_extra_flowers_l48_48892


namespace arithmetic_sequence_sum_l48_48628

theorem arithmetic_sequence_sum (S : ℕ → ℕ)
  (h₁ : S 3 = 9)
  (h₂ : S 6 = 36) :
  S 9 - S 6 = 45 :=
by
  sorry

end arithmetic_sequence_sum_l48_48628


namespace find_shift_b_l48_48056

-- Define the periodic function f
variable (f : ℝ → ℝ)
-- Define the condition on f
axiom f_periodic : ∀ x, f (x - 30) = f x

-- The theorem we want to prove
theorem find_shift_b : ∃ b > 0, (∀ x, f ((x - b) / 3) = f (x / 3)) ∧ b = 90 := 
by
  sorry

end find_shift_b_l48_48056


namespace patch_area_difference_l48_48582

theorem patch_area_difference :
  let alan_length := 30
  let alan_width := 50
  let betty_length := 35
  let betty_width := 40
  let alan_area := alan_length * alan_width
  let betty_area := betty_length * betty_width
  let area_difference := alan_area - betty_area
  area_difference = 100 := 
by
  -- Definitions
  let alan_length := 30
  let alan_width := 50
  let betty_length := 35
  let betty_width := 40
  let alan_area := alan_length * alan_width
  let betty_area := betty_length * betty_width
  let area_difference := alan_area - betty_area
  -- Proof (intentionally left as sorry)
  -- Lean should be able to use the initial definitions to verify the theorem statement.
  sorry

end patch_area_difference_l48_48582


namespace dad_borrowed_nickels_l48_48316

-- Definitions for the initial and remaining nickels
def initial_nickels : ℕ := 31
def remaining_nickels : ℕ := 11

-- Statement of the problem in Lean
theorem dad_borrowed_nickels : initial_nickels - remaining_nickels = 20 := by
  -- Proof goes here
  sorry

end dad_borrowed_nickels_l48_48316


namespace caitlin_bracelets_l48_48828

-- Define the conditions
def twice_as_many_small_beads (x y : Nat) : Prop :=
  y = 2 * x

def total_large_small_beads (total large small : Nat) : Prop :=
  total = large + small ∧ large = small

def bracelet_beads (large_beads_per_bracelet small_beads_per_bracelet large_per_bracelet : Nat) : Prop :=
  small_beads_per_bracelet = 2 * large_per_bracelet

def total_bracelets (total_large_beads large_per_bracelet bracelets : Nat) : Prop :=
  bracelets = total_large_beads / large_per_bracelet

-- The theorem to be proved
theorem caitlin_bracelets (total_beads large_per_bracelet small_per_bracelet : Nat) (bracelets : Nat) :
    total_beads = 528 ∧
    large_per_bracelet = 12 ∧
    twice_as_many_small_beads large_per_bracelet small_per_bracelet ∧
    total_large_small_beads total_beads 264 264 ∧
    bracelet_beads large_per_bracelet small_per_bracelet 12 ∧
    total_bracelets 264 12 bracelets
  → bracelets = 22 := by
  sorry

end caitlin_bracelets_l48_48828


namespace melony_profit_l48_48945

theorem melony_profit (profit_3_shirts : ℝ)
  (profit_2_sandals : ℝ)
  (h1 : profit_3_shirts = 21)
  (h2 : profit_2_sandals = 4 * 21) : profit_3_shirts / 3 * 7 + profit_2_sandals / 2 * 3 = 175 := 
by 
  sorry

end melony_profit_l48_48945


namespace find_symmetric_point_l48_48484

structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

def M : Point := ⟨3, -3, -1⟩

def line (x y z : ℝ) : Prop := 
  (x - 6) / 5 = (y - 3.5) / 4 ∧ (x - 6) / 5 = (z + 0.5) / 0

theorem find_symmetric_point (M' : Point) :
  (line M.x M.y M.z) →
  M' = ⟨-1, 2, 0⟩ := by
  sorry

end find_symmetric_point_l48_48484


namespace line_circle_no_intersection_l48_48798

theorem line_circle_no_intersection : 
  ∀ (x y : ℝ), 3 * x + 4 * y ≠ 12 ∧ x^2 + y^2 = 4 :=
by
  sorry

end line_circle_no_intersection_l48_48798


namespace gcd_m_n_l48_48737

noncomputable def m := 55555555
noncomputable def n := 111111111

theorem gcd_m_n : Int.gcd m n = 1 := by
  sorry

end gcd_m_n_l48_48737


namespace lesser_number_l48_48498

theorem lesser_number (x y : ℕ) (h1 : x + y = 60) (h2 : x - y = 10) : y = 25 :=
by
  have h3 : x = 35 := sorry
  exact sorry

end lesser_number_l48_48498


namespace bernoulli_inequality_l48_48104

theorem bernoulli_inequality (n : ℕ) (x : ℝ) (h : x ≥ -1) : 1 + n * x ≤ (1 + x) ^ n :=
sorry

end bernoulli_inequality_l48_48104


namespace area_increase_is_50_l48_48705

def length := 13
def width := 10
def length_new := length + 2
def width_new := width + 2
def area_original := length * width
def area_new := length_new * width_new
def area_increase := area_new - area_original

theorem area_increase_is_50 : area_increase = 50 :=
by
  -- Here we will include the steps to prove the theorem if required
  sorry

end area_increase_is_50_l48_48705


namespace rate_per_meter_eq_2_5_l48_48865

-- Definitions of the conditions
def diameter : ℝ := 14
def total_cost : ℝ := 109.96

-- The theorem to be proven
theorem rate_per_meter_eq_2_5 (π : ℝ) (hπ : π = 3.14159) : 
  diameter = 14 ∧ total_cost = 109.96 → (109.96 / (π * 14)) = 2.5 :=
by
  sorry

end rate_per_meter_eq_2_5_l48_48865


namespace average_of_scores_l48_48911

theorem average_of_scores :
  let scores := [50, 60, 70, 80, 80]
  let total := 340
  let num_subjects := 5
  let average := total / num_subjects
  average = 68 :=
by
  sorry

end average_of_scores_l48_48911


namespace scientific_notation_of_investment_l48_48934

theorem scientific_notation_of_investment : 41800000000 = 4.18 * 10^10 := 
by
  sorry

end scientific_notation_of_investment_l48_48934


namespace initial_amount_l48_48587

theorem initial_amount (M : ℝ) (h1 : M * 2 - 50 > 0) (h2 : (M * 2 - 50) * 2 - 60 > 0) 
(h3 : ((M * 2 - 50) * 2 - 60) * 2 - 70 > 0) 
(h4 : (((M * 2 - 50) * 2 - 60) * 2 - 70) * 2 - 80 = 0) : M = 53.75 := 
sorry

end initial_amount_l48_48587


namespace part1_min_value_part2_min_value_l48_48109

def f (x : ℝ) : ℝ := |x - 1| + 2 * |x + 1|

theorem part1_min_value :
  ∃ (m : ℝ), m = 2 ∧ (∀ (x : ℝ), f x ≥ m) :=
sorry

theorem part2_min_value (a b : ℝ) (h : a^2 + b^2 = 2) :
  ∃ (y : ℝ), y = (1 / (a^2 + 1) + 4 / (b^2 + 1)) ∧ y = 9 / 4 :=
sorry

end part1_min_value_part2_min_value_l48_48109


namespace exponential_inequality_l48_48055

theorem exponential_inequality (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  (Real.exp a * Real.exp c > Real.exp b * Real.exp d) :=
by sorry

end exponential_inequality_l48_48055


namespace emily_height_in_cm_l48_48130

theorem emily_height_in_cm 
  (inches_in_foot : ℝ) (cm_in_foot : ℝ) (emily_height_in_inches : ℝ)
  (h_if : inches_in_foot = 12) (h_cf : cm_in_foot = 30.5) (h_ehi : emily_height_in_inches = 62) :
  emily_height_in_inches * (cm_in_foot / inches_in_foot) = 157.6 :=
by
  sorry

end emily_height_in_cm_l48_48130


namespace find_triangle_side1_l48_48397

def triangle_side1 (Perimeter Side2 Side3 Side1 : ℕ) : Prop :=
  Perimeter = Side1 + Side2 + Side3

theorem find_triangle_side1 :
  ∀ (Perimeter Side2 Side3 Side1 : ℕ), 
    (Perimeter = 160) → (Side2 = 50) → (Side3 = 70) → triangle_side1 Perimeter Side2 Side3 Side1 → Side1 = 40 :=
by
  intros Perimeter Side2 Side3 Side1 h1 h2 h3 h4
  sorry

end find_triangle_side1_l48_48397


namespace annie_initial_money_l48_48501

theorem annie_initial_money
  (hamburger_price : ℕ := 4)
  (milkshake_price : ℕ := 3)
  (num_hamburgers : ℕ := 8)
  (num_milkshakes : ℕ := 6)
  (money_left : ℕ := 70)
  (total_cost_hamburgers : ℕ := num_hamburgers * hamburger_price)
  (total_cost_milkshakes : ℕ := num_milkshakes * milkshake_price)
  (total_cost : ℕ := total_cost_hamburgers + total_cost_milkshakes)
  : num_hamburgers * hamburger_price + num_milkshakes * milkshake_price + money_left = 120 :=
by
  -- proof part skipped
  sorry

end annie_initial_money_l48_48501


namespace sum_of_squares_of_roots_l48_48953

theorem sum_of_squares_of_roots :
  let a := 10
  let b := 16
  let c := -18
  let sum_of_roots := -b / a
  let product_of_roots := c / a
  sum_of_roots ^ 2 - 2 * product_of_roots = 244 / 25 := by
  sorry

end sum_of_squares_of_roots_l48_48953


namespace crates_lost_l48_48079

theorem crates_lost (total_crates : ℕ) (total_cost : ℕ) (desired_profit_percent : ℕ) 
(lost_crates remaining_crates : ℕ) (price_per_crate : ℕ) 
(h1 : total_crates = 10) (h2 : total_cost = 160) (h3 : desired_profit_percent = 25) 
(h4 : price_per_crate = 25) (h5 : remaining_crates = total_crates - lost_crates)
(h6 : price_per_crate * remaining_crates = total_cost + total_cost * desired_profit_percent / 100) :
  lost_crates = 2 :=
by
  sorry

end crates_lost_l48_48079


namespace shorter_leg_of_right_triangle_l48_48089

theorem shorter_leg_of_right_triangle (a b : ℕ) (h : a^2 + b^2 = 65^2) : min a b = 16 :=
sorry

end shorter_leg_of_right_triangle_l48_48089


namespace find_two_primes_l48_48172

-- Define what it means for a number to be prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 0 ∧ m ≠ n → n % m ≠ 0

-- Prove the existence of two specific prime numbers with the desired properties
theorem find_two_primes :
  ∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ p = 2 ∧ q = 5 ∧ is_prime (p + q) ∧ is_prime (q - p) :=
by
  exists 2
  exists 5
  repeat {split}
  sorry

end find_two_primes_l48_48172


namespace bc_sum_condition_l48_48371

-- Define the conditions as Lean definitions
def is_positive_integer (n : ℕ) : Prop := n > 0
def not_equal_to (x y : ℕ) : Prop := x ≠ y
def less_than_or_equal_to_nine (n : ℕ) : Prop := n ≤ 9

-- Main proof statement
theorem bc_sum_condition (a b c : ℕ) (h_pos_a : is_positive_integer a) (h_pos_b : is_positive_integer b) (h_pos_c : is_positive_integer c)
  (h_a_not_1 : a ≠ 1) (h_b_not_c : b ≠ c) (h_b_le_9 : less_than_or_equal_to_nine b) (h_c_le_9 : less_than_or_equal_to_nine c)
  (h_eq : (10 * a + b) * (10 * a + c) = 100 * a * a + 110 * a + b * c) :
  b + c = 11 := by
  sorry

end bc_sum_condition_l48_48371


namespace simplify_expression_l48_48240

theorem simplify_expression (b c : ℝ) : 
  (2 * 3 * b * 4 * b^2 * 5 * b^3 * 6 * b^4 * 7 * c^2 = 5040 * b^10 * c^2) :=
by sorry

end simplify_expression_l48_48240


namespace sum_of_coordinates_eq_nine_halves_l48_48507

theorem sum_of_coordinates_eq_nine_halves {f : ℝ → ℝ} 
  (h₁ : 2 = (f 1) / 2) :
  (4 + (1 / 2) = 9 / 2) :=
by 
  sorry

end sum_of_coordinates_eq_nine_halves_l48_48507


namespace claire_balance_after_week_l48_48830

theorem claire_balance_after_week :
  ∀ (gift_card : ℝ) (latte_cost croissant_cost : ℝ) (days : ℕ) (cookie_cost : ℝ) (cookies : ℕ),
  gift_card = 100 ∧
  latte_cost = 3.75 ∧
  croissant_cost = 3.50 ∧
  days = 7 ∧
  cookie_cost = 1.25 ∧
  cookies = 5 →
  (gift_card - (days * (latte_cost + croissant_cost) + cookie_cost * cookies) = 43) :=
by
  -- Skipping proof details with sorry
  sorry

end claire_balance_after_week_l48_48830


namespace literature_books_cost_more_l48_48413

theorem literature_books_cost_more :
  let num_books := 45
  let literature_cost_per_book := 7
  let technology_cost_per_book := 5
  (num_books * literature_cost_per_book) - (num_books * technology_cost_per_book) = 90 :=
by
  sorry

end literature_books_cost_more_l48_48413


namespace number_value_proof_l48_48908

theorem number_value_proof (x y : ℝ) (h1 : 0.5 * x = y + 20) (h2 : x - 2 * y = 40) : x = 40 := 
by
  sorry

end number_value_proof_l48_48908


namespace handshake_max_l48_48272

theorem handshake_max (N : ℕ) (hN : N > 4) (pN pNm1 : ℕ) 
    (hpN : pN ≠ pNm1) (h1 : ∃ p1, pN ≠ p1) (h2 : ∃ p2, pNm1 ≠ p2) :
    ∀ (i : ℕ), i ≤ N - 2 → i ≤ N - 2 :=
sorry

end handshake_max_l48_48272


namespace monotonic_power_function_l48_48221

noncomputable def power_function (a : ℝ) (x : ℝ) : ℝ := (a^2 - 2 * a - 2) * x^a

theorem monotonic_power_function (a : ℝ) (h1 : ∀ x : ℝ, ( ∀ x1 x2 : ℝ, x1 < x2 → power_function a x1 < power_function a x2 ) )
  (h2 : a^2 - 2 * a - 2 = 1) (h3 : a > 0) : a = 3 :=
by
  sorry

end monotonic_power_function_l48_48221


namespace find_x_in_terms_of_a_b_l48_48032

variable (a b x : ℝ)
variable (ha : a > 0) (hb : b > 0) (hx : x > 0) (r : ℝ)
variable (h1 : r = (4 * a)^(3 * b))
variable (h2 : r = a ^ b * x ^ b)

theorem find_x_in_terms_of_a_b 
  (ha : a > 0) (hb : b > 0) (hx : x > 0)
  (h1 : (4 * a)^(3 * b) = r)
  (h2 : r = a^b * x^b) :
  x = 64 * a^2 :=
by
  sorry

end find_x_in_terms_of_a_b_l48_48032


namespace triangle_area_rational_l48_48609

theorem triangle_area_rational
  (x1 y1 x2 y2 x3 y3 : ℤ)
  (h : y1 = y2) :
  ∃ (k : ℚ), 
    k = abs ((x2 - x1) * y3) / 2 := sorry

end triangle_area_rational_l48_48609


namespace gcd_of_78_and_36_l48_48919

theorem gcd_of_78_and_36 :
  Nat.gcd 78 36 = 6 :=
by
  sorry

end gcd_of_78_and_36_l48_48919


namespace B_catches_up_with_A_l48_48996

-- Define the conditions
def speed_A : ℝ := 10 -- A's speed in kmph
def speed_B : ℝ := 20 -- B's speed in kmph
def delay : ℝ := 6 -- Delay in hours after A's start

-- Define the total distance where B catches up with A
def distance_catch_up : ℝ := 120

-- Statement to prove B catches up with A at 120 km from the start
theorem B_catches_up_with_A :
  (speed_A * delay + speed_A * (distance_catch_up / speed_B - delay)) = distance_catch_up :=
by
  sorry

end B_catches_up_with_A_l48_48996


namespace twenty_seven_cubes_volume_l48_48842

def volume_surface_relation (x V S : ℝ) : Prop :=
  V = x^3 ∧ S = 6 * x^2 ∧ V + S = (4 / 3) * (12 * x)

theorem twenty_seven_cubes_volume (x : ℝ) (hx : volume_surface_relation x (x^3) (6 * x^2)) : 
  27 * (x^3) = 216 :=
by
  sorry

end twenty_seven_cubes_volume_l48_48842


namespace number_of_terms_in_arithmetic_sequence_is_39_l48_48353

theorem number_of_terms_in_arithmetic_sequence_is_39 :
  ∀ (a d l : ℤ), 
  d ≠ 0 → 
  a = 128 → 
  d = -3 → 
  l = 14 → 
  ∃ n : ℕ, (a + (↑n - 1) * d = l) ∧ (n = 39) :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_is_39_l48_48353


namespace int_n_satisfying_conditions_l48_48383

theorem int_n_satisfying_conditions : 
  (∃! (n : ℤ), ∃ (k : ℤ), (n + 3 = k^2 * (23 - n)) ∧ n ≠ 23) :=
by
  use 2
  -- Provide a proof for this statement here
  sorry

end int_n_satisfying_conditions_l48_48383


namespace max_length_sequence_l48_48965

def seq_term (n : ℕ) (y : ℤ) : ℤ :=
  match n with
  | 0 => 2000
  | 1 => y
  | k + 2 => seq_term (k + 1) y - seq_term k y

theorem max_length_sequence (y : ℤ) :
  1200 < y ∧ y < 1334 ∧ (∀ n, seq_term n y ≥ 0 ∨ seq_term (n + 1) y < 0) ↔ y = 1333 :=
by
  sorry

end max_length_sequence_l48_48965


namespace complex_div_imaginary_unit_eq_l48_48608

theorem complex_div_imaginary_unit_eq :
  (∀ i : ℂ, i^2 = -1 → (1 / (1 + i)) = ((1 - i) / 2)) :=
by
  intro i
  intro hi
  /- The proof will be inserted here -/
  sorry

end complex_div_imaginary_unit_eq_l48_48608


namespace bucket_full_weight_l48_48058

variable (p q r : ℚ)
variable (x y : ℚ)

-- Define the conditions
def condition1 : Prop := p = r + (3 / 4) * y
def condition2 : Prop := q = r + (1 / 3) * y
def condition3 : Prop := x = r

-- Define the conclusion
def conclusion : Prop := x + y = (4 * p - r) / 3

-- The theorem stating that the conclusion follows from the conditions
theorem bucket_full_weight (h1 : condition1 p r y) (h2 : condition2 q r y) (h3 : condition3 x r) : conclusion x y p r :=
by
  sorry

end bucket_full_weight_l48_48058


namespace debate_team_selections_l48_48648

theorem debate_team_selections
  (A_selected C_selected B_selected E_selected : Prop)
  (h1: A_selected ∨ C_selected)
  (h2: B_selected ∨ E_selected)
  (h3: ¬ (B_selected ∧ E_selected) ∧ ¬ (C_selected ∧ E_selected))
  (not_B_selected : ¬ B_selected) :
  A_selected ∧ E_selected :=
by
  sorry

end debate_team_selections_l48_48648


namespace a_le_neg4_l48_48590

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - 2 * x
noncomputable def g (a x : ℝ) : ℝ := a * Real.log x

noncomputable def h (a x : ℝ) : ℝ := f x - g a x

-- Theorem
theorem a_le_neg4 (a : ℝ) : 
  (∀ (x1 x2 : ℝ), x1 ≠ x2 → x1 > 0 → x2 > 0 → (h a x1 - h a x2) / (x1 - x2) > 2) →
  a ≤ -4 :=
by
  sorry

end a_le_neg4_l48_48590


namespace quadratic_inequality_real_solutions_l48_48610

-- Definitions and conditions
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- The main statement
theorem quadratic_inequality_real_solutions (c : ℝ) (h_pos : 0 < c) : 
  (∀ x : ℝ, x^2 - 8 * x + c < 0) ↔ (c < 16) :=
by 
  sorry

end quadratic_inequality_real_solutions_l48_48610


namespace winner_collected_l48_48095

variable (M : ℕ)
variable (last_year_rate this_year_rate : ℝ)
variable (extra_miles : ℕ)
variable (money_collected_last_year money_collected_this_year : ℝ)

axiom rate_last_year : last_year_rate = 4
axiom rate_this_year : this_year_rate = 2.75
axiom extra_miles_eq : extra_miles = 5

noncomputable def money_eq (M : ℕ) : ℝ :=
  last_year_rate * M

theorem winner_collected :
  ∃ M : ℕ, money_eq M = 44 :=
by
  sorry

end winner_collected_l48_48095


namespace f_2023_l48_48890

noncomputable def f : ℕ → ℤ := sorry

axiom f_defined_for_all : ∀ x : ℕ, f x ≠ 0 → (x ≥ 0)
axiom f_one : f 1 = 1
axiom f_functional_eq : ∀ a b : ℕ, f (a + b) = f a + f b - 3 * f (a * b)

theorem f_2023 : f 2023 = -(2^2022 - 1) := sorry

end f_2023_l48_48890


namespace trig_inequality_l48_48851
open Real

theorem trig_inequality (α β γ x y z : ℝ) 
  (h1 : α + β + γ = π)
  (h2 : x + y + z = 0) :
  y * z * (sin α)^2 + z * x * (sin β)^2 + x * y * (sin γ)^2 ≤ 0 := 
sorry

end trig_inequality_l48_48851


namespace fraction_is_three_halves_l48_48882

theorem fraction_is_three_halves (a b : ℝ) (hb : b ≠ 0) (h : 2 * a = 3 * b) : a / b = 3 / 2 :=
sorry

end fraction_is_three_halves_l48_48882


namespace remainder_of_2n_div_11_l48_48511

theorem remainder_of_2n_div_11 (n k : ℤ) (h : n = 22 * k + 12) : (2 * n) % 11 = 2 :=
by
  sorry

end remainder_of_2n_div_11_l48_48511


namespace ratio_of_ages_in_two_years_l48_48695

-- Define the constants
def son_age : ℕ := 24
def age_difference : ℕ := 26

-- Define the equations based on conditions
def man_age := son_age + age_difference
def son_future_age := son_age + 2
def man_future_age := man_age + 2

-- State the theorem for the required ratio
theorem ratio_of_ages_in_two_years : man_future_age / son_future_age = 2 := by
  sorry

end ratio_of_ages_in_two_years_l48_48695


namespace factor_sum_l48_48863

variable (x y : ℝ)

theorem factor_sum :
  let a := 1
  let b := -2
  let c := 1
  let d := 2
  let e := 4
  let f := 1
  let g := 2
  let h := 1
  let j := -2
  let k := 4
  (27 * x^9 - 512 * y^9) = ((a * x + b * y) * (c * x^3 + d * x * y^2 + e * y^3) * 
  (f * x + g * y) * (h * x^3 + j * x * y^2 + k * y^3)) → 
  (a + b + c + d + e + f + g + h + j + k = 12) :=
by
  sorry

end factor_sum_l48_48863


namespace smallest_four_digit_multiple_of_37_l48_48508

theorem smallest_four_digit_multiple_of_37 : ∃ n : ℕ, n ≥ 1000 ∧ n ≤ 9999 ∧ 37 ∣ n ∧ (∀ m : ℕ, m ≥ 1000 ∧ m ≤ 9999 ∧ 37 ∣ m → n ≤ m) ∧ n = 1036 :=
by
  sorry

end smallest_four_digit_multiple_of_37_l48_48508


namespace hot_dogs_leftover_l48_48342

theorem hot_dogs_leftover :
  36159782 % 6 = 2 :=
by
  sorry

end hot_dogs_leftover_l48_48342


namespace silvia_percentage_shorter_l48_48533

theorem silvia_percentage_shorter :
  let j := (2 : ℝ) + 1
  let s := Real.sqrt ((2 : ℝ) ^ 2 + (1 : ℝ) ^ 2)
  (abs (( (j - s) / j) * 100 - 25) < 1) :=
by
  let j := (2 : ℝ) + 1
  let s := Real.sqrt ((2 : ℝ) ^ 2 + (1 : ℝ) ^ 2)
  show (abs (( (j - s) / j) * 100 - 25) < 1)
  sorry

end silvia_percentage_shorter_l48_48533


namespace min_m_n_sum_l48_48968

theorem min_m_n_sum (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : 108 * m = n^3) : m + n = 8 :=
  sorry

end min_m_n_sum_l48_48968


namespace avg_visitors_per_day_l48_48213

theorem avg_visitors_per_day 
  (avg_visitors_sundays : ℕ) 
  (avg_visitors_other_days : ℕ) 
  (total_days : ℕ) 
  (sundays : ℕ) 
  (other_days : ℕ)
  (hs : avg_visitors_sundays = 630)
  (ho : avg_visitors_other_days = 240)
  (td : total_days = 30)
  (sd : sundays = 4)
  (od : other_days = 26)
  : (4 * avg_visitors_sundays + 26 * avg_visitors_other_days) / 30 = 292 := 
by
  sorry

end avg_visitors_per_day_l48_48213


namespace max_possible_median_l48_48356

theorem max_possible_median (total_cups : ℕ) (total_customers : ℕ) (min_cups_per_customer : ℕ)
  (h1 : total_cups = 310) (h2 : total_customers = 120) (h3 : min_cups_per_customer = 1) :
  ∃ median : ℕ, median = 4 :=
by {
  sorry
}

end max_possible_median_l48_48356


namespace min_value_a_b_l48_48672

theorem min_value_a_b (x y a b : ℝ) (h1 : 2 * x - y + 2 ≥ 0) (h2 : 8 * x - y - 4 ≤ 0) 
  (h3 : x ≥ 0) (h4 : y ≥ 0) (h5 : a > 0) (h6 : b > 0) (h7 : a * x + y = 8) : 
  a + b ≥ 4 :=
sorry

end min_value_a_b_l48_48672


namespace subtraction_example_l48_48421

theorem subtraction_example :
  145.23 - 0.07 = 145.16 :=
sorry

end subtraction_example_l48_48421


namespace radius_of_circle_centered_at_l48_48810

def center : ℝ × ℝ := (3, 4)

def intersects_axes_at_three_points (A : ℝ × ℝ) (r : ℝ) : Prop :=
  (A.1 - r = 0 ∨ A.1 + r = 0) ∧ (A.2 - r = 0 ∨ A.2 + r = 0)

theorem radius_of_circle_centered_at (A : ℝ × ℝ) : 
  (intersects_axes_at_three_points A 4) ∨ (intersects_axes_at_three_points A 5) :=
by
  sorry

end radius_of_circle_centered_at_l48_48810


namespace amount_after_two_years_l48_48090

noncomputable def amountAfterYears (presentValue : ℝ) (rate : ℝ) (n : ℕ) : ℝ :=
  presentValue * (1 + rate) ^ n

theorem amount_after_two_years 
  (presentValue : ℝ := 62000) 
  (rate : ℝ := 1 / 8) 
  (n : ℕ := 2) : 
  amountAfterYears presentValue rate n = 78468.75 := 
  sorry

end amount_after_two_years_l48_48090


namespace chess_piece_problem_l48_48668

theorem chess_piece_problem
  (a b c : ℕ)
  (h1 : b = b * 2 - a)
  (h2 : c = c * 2)
  (h3 : a = a * 2 - b)
  (h4 : c = c * 2 - a + b)
  (h5 : a * 2 = 16)
  (h6 : b * 2 = 16)
  (h7 : c * 2 = 16) : 
  a = 26 ∧ b = 14 ∧ c = 8 := 
sorry

end chess_piece_problem_l48_48668


namespace smallest_k_condition_l48_48966

theorem smallest_k_condition (n k : ℕ) (h_n : n ≥ 2) (h_k : k = 2 * n) :
  ∀ (f : Fin n → Fin n → Fin k), (∀ i j, f i j < k) →
  (∃ a b c d : Fin n, a ≠ c ∧ b ≠ d ∧ f a b ≠ f a d ∧ f a b ≠ f c b ∧ f a b ≠ f c d ∧ f a d ≠ f c b ∧ f a d ≠ f c d ∧ f c b ≠ f c d) :=
sorry

end smallest_k_condition_l48_48966


namespace prime_pairs_solution_l48_48203

def is_prime (n : ℕ) : Prop := Nat.Prime n

def conditions (p q : ℕ) : Prop := 
  p^2 ∣ q^3 + 1 ∧ q^2 ∣ p^6 - 1

theorem prime_pairs_solution :
  ({(p, q) | is_prime p ∧ is_prime q ∧ conditions p q} = {(3, 2), (2, 3)}) :=
by
  sorry

end prime_pairs_solution_l48_48203


namespace find_xyz_l48_48310

theorem find_xyz
  (x y z : ℝ)
  (h1 : x + y + z = 38)
  (h2 : x * y * z = 2002)
  (h3 : 0 < x ∧ x ≤ 11)
  (h4 : z ≥ 14) :
  x = 11 ∧ y = 13 ∧ z = 14 :=
sorry

end find_xyz_l48_48310


namespace rotation_150_positions_l48_48468

/-
Define the initial positions and the shapes involved.
-/
noncomputable def initial_positions := ["A", "B", "C", "D"]
noncomputable def initial_order := ["triangle", "smaller_circle", "square", "pentagon"]

def rotate_clockwise_150 (pos : List String) : List String :=
  -- 1 full position and two-thirds into the next position
  [pos.get! 1, pos.get! 2, pos.get! 3, pos.get! 0]

theorem rotation_150_positions :
  rotate_clockwise_150 initial_positions = ["Triangle between B and C", 
                                            "Smaller circle between C and D", 
                                            "Square between D and A", 
                                            "Pentagon between A and B"] :=
by sorry

end rotation_150_positions_l48_48468


namespace lcm_of_a_c_l48_48506

theorem lcm_of_a_c (a b c : ℕ) (h1 : Nat.lcm a b = 20) (h2 : Nat.lcm b c = 24) : Nat.lcm a c = 30 := by
  sorry

end lcm_of_a_c_l48_48506


namespace equivalent_angle_l48_48162

theorem equivalent_angle (θ : ℝ) : 
  (∃ k : ℤ, θ = k * 360 + 257) ↔ θ = -463 ∨ (∃ k : ℤ, θ = k * 360 + 257) :=
by
  sorry

end equivalent_angle_l48_48162


namespace quadratic_has_real_solution_l48_48343

theorem quadratic_has_real_solution (a b c : ℝ) : 
  ∃ x : ℝ, x^2 + (a - b) * x + (b - c) = 0 ∨ 
           x^2 + (b - c) * x + (c - a) = 0 ∨ 
           x^2 + (c - a) * x + (a - b) = 0 :=
  sorry

end quadratic_has_real_solution_l48_48343


namespace charlie_contribution_l48_48957

theorem charlie_contribution (a b c : ℝ) (h₁ : a + b + c = 72) (h₂ : a = 1/4 * (b + c)) (h₃ : b = 1/5 * (a + c)) :
  c = 49 :=
by sorry

end charlie_contribution_l48_48957


namespace regular_admission_ticket_price_l48_48253

theorem regular_admission_ticket_price
  (n : ℕ) (t : ℕ) (p : ℕ)
  (n_r n_s r : ℕ)
  (H1 : n_r = 3 * n_s)
  (H2 : n_s + n_r = n)
  (H3 : n_r * r + n_s * p = t)
  (H4 : n = 3240)
  (H5 : t = 22680)
  (H6 : p = 4) : 
  r = 8 :=
by sorry

end regular_admission_ticket_price_l48_48253


namespace percentage_increase_l48_48897

theorem percentage_increase (N P : ℕ) (h1 : N = 40)
       (h2 : (N + (P / 100) * N) - (N - (30 / 100) * N) = 22) : P = 25 :=
by 
  have p1 := h1
  have p2 := h2
  sorry

end percentage_increase_l48_48897


namespace pamela_skittles_l48_48133

variable (initial_skittles : Nat) (given_to_karen : Nat)

def skittles_after_giving (initial_skittles given_to_karen : Nat) : Nat :=
  initial_skittles - given_to_karen

theorem pamela_skittles (h1 : initial_skittles = 50) (h2 : given_to_karen = 7) :
  skittles_after_giving initial_skittles given_to_karen = 43 := by
  sorry

end pamela_skittles_l48_48133


namespace verify_quadratic_solution_l48_48854

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def quadratic_roots : Prop :=
  ∃ (p q : ℕ) (x1 x2 : ℤ), is_prime p ∧ is_prime q ∧ 
  (x1 + x2 = -(p : ℤ)) ∧ (x1 * x2 = (3 * q : ℤ)) ∧ x1 < 0 ∧ x2 < 0 ∧ 
  ((p = 7 ∧ q = 2) ∨ (p = 5 ∧ q = 2))

theorem verify_quadratic_solution : quadratic_roots :=
  by {
    sorry
  }

end verify_quadratic_solution_l48_48854


namespace math_problem_l48_48018

variables {A B : Type} [Fintype A] [Fintype B]
          (p1 p2 : ℝ) (h1 : 1/2 < p1) (h2 : p1 < p2) (h3 : p2 < 1)
          (nA : ℕ) (hA : nA = 3) (nB : ℕ) (hB : nB = 3)

noncomputable def E_X : ℝ := nA * p1
noncomputable def E_Y : ℝ := nB * p2

noncomputable def D_X : ℝ := nA * p1 * (1 - p1)
noncomputable def D_Y : ℝ := nB * p2 * (1 - p2)

theorem math_problem :
  E_X p1 nA = 3 * p1 →
  E_Y p2 nB = 3 * p2 →
  D_X p1 nA = 3 * p1 * (1 - p1) →
  D_Y p2 nB = 3 * p2 * (1 - p2) →
  E_X p1 nA < E_Y p2 nB ∧ D_X p1 nA > D_Y p2 nB :=
by
  sorry

end math_problem_l48_48018


namespace no_solution_to_system_l48_48175

theorem no_solution_to_system :
  ∀ x y : ℝ, ¬ (3 * x - 4 * y = 12 ∧ 9 * x - 12 * y = 15) :=
by
  sorry

end no_solution_to_system_l48_48175


namespace pens_difference_proof_l48_48204

variables (A B M N X Y : ℕ)

-- Initial number of pens for Alex and Jane
def Alex_initial (A : ℕ) := A
def Jane_initial (B : ℕ) := B

-- Weekly multiplication factors for Alex and Jane
def Alex_weekly_growth (X : ℕ) := X
def Jane_weekly_growth (Y : ℕ) := Y

-- Number of pens after 4 weeks
def Alex_after_4_weeks (A X : ℕ) := A * X^4
def Jane_after_4_weeks (B Y : ℕ) := B * Y^4

-- Proving the difference in the number of pens
theorem pens_difference_proof (hM : M = A * X^4) (hN : N = B * Y^4) :
  M - N = (A * X^4) - (B * Y^4) :=
by sorry

end pens_difference_proof_l48_48204


namespace find_ellipse_equation_l48_48656

-- Definitions based on conditions
def ellipse_centered_at_origin (x y : ℝ) (m n : ℝ) := m * x ^ 2 + n * y ^ 2 = 1

def passes_through_points_A_and_B (m n : ℝ) := 
  (ellipse_centered_at_origin 0 (-2) m n) ∧ (ellipse_centered_at_origin (3 / 2) (-1) m n)

-- Statement to be proved
theorem find_ellipse_equation : 
  ∃ (m n : ℝ), (m > 0) ∧ (n > 0) ∧ (m ≠ n) ∧ 
  passes_through_points_A_and_B m n ∧ 
  m = 1 / 3 ∧ n = 1 / 4 :=
by sorry

end find_ellipse_equation_l48_48656


namespace shaded_area_of_rectangle_l48_48008

theorem shaded_area_of_rectangle :
  let length := 5   -- Length of the rectangle in cm
  let width := 12   -- Width of the rectangle in cm
  let base := 2     -- Base of each triangle in cm
  let height := 5   -- Height of each triangle in cm
  let rect_area := length * width
  let triangle_area := (1 / 2) * base * height
  let unshaded_area := 2 * triangle_area
  let shaded_area := rect_area - unshaded_area
  shaded_area = 50 :=
by
  -- Calculation follows solution steps.
  sorry

end shaded_area_of_rectangle_l48_48008


namespace sum_of_A_and_B_l48_48772

theorem sum_of_A_and_B (A B : ℕ) (h1 : A ≠ B) (h2 : A < 10) (h3 : B < 10) :
  (10 * A + B) * 6 = 111 * B → A + B = 11 :=
by
  intros h
  sorry

end sum_of_A_and_B_l48_48772


namespace average_income_Q_and_R_l48_48972

variable (P Q R: ℝ)

theorem average_income_Q_and_R:
  (P + Q) / 2 = 5050 →
  (P + R) / 2 = 5200 →
  P = 4000 →
  (Q + R) / 2 = 6250 :=
by
  sorry

end average_income_Q_and_R_l48_48972


namespace area_of_ABCD_is_196_l48_48954

-- Define the shorter side length of the smaller rectangles
def shorter_side : ℕ := 7

-- Define the longer side length of the smaller rectangles
def longer_side : ℕ := 2 * shorter_side

-- Define the width of rectangle ABCD
def width_ABCD : ℕ := 2 * shorter_side

-- Define the length of rectangle ABCD
def length_ABCD : ℕ := longer_side

-- Define the area of rectangle ABCD
def area_ABCD : ℕ := length_ABCD * width_ABCD

-- Statement of the problem
theorem area_of_ABCD_is_196 : area_ABCD = 196 :=
by
  -- insert proof here
  sorry

end area_of_ABCD_is_196_l48_48954


namespace calculate_exponent_product_l48_48995

theorem calculate_exponent_product : (3^3 * 5^3) * (3^8 * 5^8) = 15^11 := by
  sorry

end calculate_exponent_product_l48_48995


namespace common_solutions_y_values_l48_48108

theorem common_solutions_y_values :
  (∃ x : ℝ, x^2 + y^2 - 16 = 0 ∧ x^2 - 3*y - 12 = 0) ↔ (y = -4 ∨ y = 1) :=
by {
  sorry
}

end common_solutions_y_values_l48_48108


namespace jenny_stamps_l48_48862

theorem jenny_stamps :
  let num_books := 8
  let pages_per_book := 42
  let stamps_per_page := 6
  let new_stamps_per_page := 10
  let complete_books_in_new_system := 4
  let pages_in_fifth_book := 33
  (num_books * pages_per_book * stamps_per_page) % new_stamps_per_page = 6 :=
by
  sorry

end jenny_stamps_l48_48862


namespace surface_area_of_solid_l48_48578

-- Define a unit cube and the number of cubes
def unitCube : Type := { faces : ℕ // faces = 6 }
def numCubes : ℕ := 10

-- Define the surface area contribution from different orientations
def surfaceAreaFacingUs (cubes : ℕ) : ℕ := 2 * cubes -- faces towards and away
def verticalSidesArea (heightCubes : ℕ) : ℕ := 2 * heightCubes -- left and right vertical sides
def horizontalSidesArea (widthCubes : ℕ) : ℕ := 2 * widthCubes -- top and bottom horizontal sides

-- Define the surface area for the given configuration of 10 cubes
def totalSurfaceArea (cubes : ℕ) (height : ℕ) (width : ℕ) : ℕ :=
  (surfaceAreaFacingUs cubes) + (verticalSidesArea height) + (horizontalSidesArea width)

-- Assumptions based on problem description
def heightCubes : ℕ := 3
def widthCubes : ℕ := 4

-- The theorem we want to prove
theorem surface_area_of_solid : totalSurfaceArea numCubes heightCubes widthCubes = 34 := by
  sorry

end surface_area_of_solid_l48_48578


namespace max_a_no_lattice_points_l48_48922

theorem max_a_no_lattice_points :
  ∀ (m : ℝ), (1 / 3) < m → m < (17 / 51) →
  ¬ ∃ (x : ℕ) (y : ℕ), 0 < x ∧ x ≤ 50 ∧ y = m * x + 3 := 
by
  sorry

end max_a_no_lattice_points_l48_48922


namespace problem1_problem2_l48_48773

-- Problem 1
theorem problem1 : 40 + ((1 / 6) - (2 / 3) + (3 / 4)) * 12 = 43 :=
by
  sorry

-- Problem 2
theorem problem2 : (-1) ^ 2 * (-5) + ((-3) ^ 2 + 2 * (-5)) = 4 :=
by
  sorry

end problem1_problem2_l48_48773


namespace range_of_inclination_angle_l48_48723

theorem range_of_inclination_angle (α : ℝ) (A : ℝ × ℝ) (hA : A = (-2, 0))
  (ellipse_eq : ∀ (x y : ℝ), (x^2 / 2) + y^2 = 1) :
    (0 ≤ α ∧ α < Real.arcsin (Real.sqrt 3 / 3)) ∨ 
    (π - Real.arcsin (Real.sqrt 3 / 3) < α ∧ α < π) :=
sorry

end range_of_inclination_angle_l48_48723


namespace positive_reals_power_equality_l48_48629

open Real

theorem positive_reals_power_equality (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^b = b^a) (h4 : a < 1) : a = b := 
  by
  sorry

end positive_reals_power_equality_l48_48629


namespace time_between_rings_is_288_minutes_l48_48930

def intervals_between_rings (total_rings : ℕ) (total_minutes : ℕ) : ℕ := 
  let intervals := total_rings - 1
  total_minutes / intervals

theorem time_between_rings_is_288_minutes (total_minutes_in_day total_rings : ℕ) 
  (h1 : total_minutes_in_day = 1440) (h2 : total_rings = 6) : 
  intervals_between_rings total_rings total_minutes_in_day = 288 := 
by 
  sorry

end time_between_rings_is_288_minutes_l48_48930


namespace find_y_l48_48469

theorem find_y (y : ℚ) : (3 / y - (3 / y) * (y / 5) = 1.2) → y = 5 / 3 :=
sorry

end find_y_l48_48469


namespace percentage_increase_is_50_l48_48584

-- Definition of the given values
def original_time : ℕ := 30
def new_time : ℕ := 45

-- Assertion stating that the percentage increase is 50%
theorem percentage_increase_is_50 :
  (new_time - original_time) * 100 / original_time = 50 := 
sorry

end percentage_increase_is_50_l48_48584


namespace football_cost_correct_l48_48786

def cost_marble : ℝ := 9.05
def cost_baseball : ℝ := 6.52
def total_cost : ℝ := 20.52
def cost_football : ℝ := total_cost - cost_marble - cost_baseball

theorem football_cost_correct : cost_football = 4.95 := 
by
  -- The proof is omitted, as per instructions.
  sorry

end football_cost_correct_l48_48786


namespace park_maple_trees_total_l48_48762

theorem park_maple_trees_total (current_maples planted_maples : ℕ) 
    (h1 : current_maples = 2) (h2 : planted_maples = 9) 
    : current_maples + planted_maples = 11 := 
by
  sorry

end park_maple_trees_total_l48_48762


namespace zoo_individuals_remaining_l48_48937

noncomputable def initial_students_class1 := 10
noncomputable def initial_students_class2 := 10
noncomputable def chaperones := 5
noncomputable def teachers := 2
noncomputable def students_left := 10
noncomputable def chaperones_left := 2

theorem zoo_individuals_remaining :
  let total_initial_individuals := initial_students_class1 + initial_students_class2 + chaperones + teachers
  let total_left := students_left + chaperones_left
  total_initial_individuals - total_left = 15 := by
  sorry

end zoo_individuals_remaining_l48_48937


namespace stream_speed_l48_48445

theorem stream_speed (v : ℝ) (boat_speed : ℝ) (distance : ℝ) (time : ℝ) 
    (h1 : boat_speed = 10) 
    (h2 : distance = 54) 
    (h3 : time = 3) 
    (h4 : distance = (boat_speed + v) * time) : 
    v = 8 :=
by
  sorry

end stream_speed_l48_48445


namespace intersection_A_B_l48_48327

-- Define the sets A and B
def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {x | 3 - 2 * x > 0}

-- Prove the intersection of A and B
theorem intersection_A_B :
  (A ∩ B) = {x | x < (3 / 2)} := sorry

end intersection_A_B_l48_48327


namespace parabola_properties_l48_48475

open Real 

theorem parabola_properties 
  (a : ℝ) 
  (h₀ : a ≠ 0)
  (h₁ : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (x₁^2 + (1 - 2 * a) * x₁ + a^2 = 0) ∧ (x₂^2 + (1 - 2 * a) * x₂ + a^2 = 0)) :
  (a < 1 / 4 ∧ ∀ x₁ x₂, (x₁^2 + (1 - 2 * a) * x₁ + a^2 = 0) ∧ (x₂^2 + (1 - 2 * a) * x₂ + a^2 = 0) → x₁ < 0 ∧ x₂ < 0) ∧
  (∀ (x₁ x₂ C : ℝ), (x₁^2 + (1 - 2 * a) * x₁ + a^2 = 0) ∧ (x₂^2 + (1 - 2 * a) * x₂ + a^2 = 0) 
   ∧ (C = a^2) ∧ (-x₁ - x₂ = C - 2) → a = -3) :=
by
  sorry

end parabola_properties_l48_48475


namespace tractor_brigades_l48_48452
noncomputable def brigade_plowing : Prop :=
∃ x y : ℝ,
  x * y = 240 ∧
  (x + 3) * (y + 2) = 324 ∧
  x > 20 ∧
  (x + 3) > 20 ∧
  x = 24 ∧
  (x + 3) = 27

theorem tractor_brigades:
  brigade_plowing :=
sorry

end tractor_brigades_l48_48452


namespace range_of_a_l48_48540

def inequality_system_has_solution (a : ℝ) : Prop :=
  ∃ x : ℝ, (x + a ≥ 0) ∧ (1 - 2 * x > x - 2)

theorem range_of_a (a : ℝ) : inequality_system_has_solution a ↔ a > -1 :=
by
  sorry

end range_of_a_l48_48540


namespace coterminal_angle_l48_48788

theorem coterminal_angle :
  ∀ θ : ℤ, (θ - 60) % 360 = 0 → θ = -300 ∨ θ = -60 ∨ θ = 600 ∨ θ = 1380 :=
by
  sorry

end coterminal_angle_l48_48788


namespace symmetrical_shapes_congruent_l48_48811

theorem symmetrical_shapes_congruent
  (shapes : Type)
  (is_symmetrical : shapes → shapes → Prop)
  (congruent : shapes → shapes → Prop)
  (symmetrical_implies_equal_segments : ∀ (s1 s2 : shapes), is_symmetrical s1 s2 → ∀ (segment : ℝ), segment_s1 = segment_s2)
  (symmetrical_implies_equal_angles : ∀ (s1 s2 : shapes), is_symmetrical s1 s2 → ∀ (angle : ℝ), angle_s1 = angle_s2) :
  ∀ (s1 s2 : shapes), is_symmetrical s1 s2 → congruent s1 s2 :=
by
  sorry

end symmetrical_shapes_congruent_l48_48811


namespace area_of_plot_is_correct_l48_48393

-- Define the side length of the square plot
def side_length : ℝ := 50.5

-- Define the area of the square plot
def area_of_square (s : ℝ) : ℝ := s * s

-- Theorem stating that the area of a square plot with side length 50.5 m is 2550.25 m²
theorem area_of_plot_is_correct : area_of_square side_length = 2550.25 := by
  sorry

end area_of_plot_is_correct_l48_48393


namespace two_digit_numbers_with_5_as_second_last_digit_l48_48254

theorem two_digit_numbers_with_5_as_second_last_digit:
  ∀ N : ℕ, (10 ≤ N ∧ N ≤ 99) → (∃ k : ℤ, (N * k) % 100 / 10 = 5) ↔ ¬(N % 20 = 0) :=
by
  sorry

end two_digit_numbers_with_5_as_second_last_digit_l48_48254


namespace lucien_balls_count_l48_48612

theorem lucien_balls_count (lucca_balls : ℕ) (lucca_percent_basketballs : ℝ) (lucien_percent_basketballs : ℝ) (total_basketballs : ℕ)
  (h1 : lucca_balls = 100)
  (h2 : lucca_percent_basketballs = 0.10)
  (h3 : lucien_percent_basketballs = 0.20)
  (h4 : total_basketballs = 50) :
  ∃ lucien_balls : ℕ, lucien_balls = 200 :=
by
  sorry

end lucien_balls_count_l48_48612


namespace inclination_angle_of_line_l48_48491

theorem inclination_angle_of_line (α : ℝ) (h_eq : ∀ x y, x - y + 1 = 0 ↔ y = x + 1) (h_range : 0 < α ∧ α < 180) :
  α = 45 :=
by
  -- α is the inclination angle satisfying tan α = 1 and 0 < α < 180
  sorry

end inclination_angle_of_line_l48_48491


namespace solve_eq1_solve_eq2_l48_48494

-- Define the first equation
def eq1 (x : ℝ) : Prop := x^2 - 2 * x - 1 = 0

-- Define the second equation
def eq2 (x : ℝ) : Prop := (x - 2)^2 = 2 * x - 4

-- State the first theorem
theorem solve_eq1 (x : ℝ) : eq1 x ↔ (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2) :=
by sorry

-- State the second theorem
theorem solve_eq2 (x : ℝ) : eq2 x ↔ (x = 2 ∨ x = 4) :=
by sorry

end solve_eq1_solve_eq2_l48_48494


namespace larger_solution_quadratic_l48_48201

theorem larger_solution_quadratic :
  (∃ a b : ℝ, a ≠ b ∧ (a = 9) ∧ (b = -2) ∧
              (∀ x : ℝ, x^2 - 7 * x - 18 = 0 → (x = a ∨ x = b))) →
  9 = max a b :=
by
  sorry

end larger_solution_quadratic_l48_48201


namespace smallest_of_product_and_sum_l48_48649

theorem smallest_of_product_and_sum (a b c : ℤ) 
  (h1 : a * b * c = 32) 
  (h2 : a + b + c = 3) : 
  a = -4 ∨ b = -4 ∨ c = -4 :=
sorry

end smallest_of_product_and_sum_l48_48649


namespace moles_CO2_formed_l48_48251

-- Define the conditions based on the problem statement
def moles_HCl := 1
def moles_NaHCO3 := 1

-- Define the reaction equation in equivalence terms
def chemical_equation (hcl : Nat) (nahco3 : Nat) : Nat :=
  if hcl = 1 ∧ nahco3 = 1 then 1 else 0

-- State the proof problem
theorem moles_CO2_formed : chemical_equation moles_HCl moles_NaHCO3 = 1 :=
by
  -- The proof goes here
  sorry

end moles_CO2_formed_l48_48251


namespace quadratic_transformation_l48_48733

theorem quadratic_transformation (a b c : ℝ) (h : a * x^2 + b * x + c = 5 * (x + 2)^2 - 7) :
  ∃ (n m g : ℝ), 2 * a * x^2 + 2 * b * x + 2 * c = n * (x - g)^2 + m ∧ g = -2 :=
by
  sorry

end quadratic_transformation_l48_48733


namespace smith_family_seating_problem_l48_48144

theorem smith_family_seating_problem :
  let total_children := 8
  let boys := 4
  let girls := 4
  (total_children.factorial - (boys.factorial * girls.factorial)) = 39744 :=
by
  sorry

end smith_family_seating_problem_l48_48144


namespace count_three_letter_sets_l48_48398

-- Define the set of letters
def letters := Finset.range 10  -- representing letters A (0) to J (9)

-- Define the condition that J (represented by 9) cannot be the first initial
def valid_first_initials := letters.erase 9  -- remove 9 (J) from 0 to 9

-- Calculate the number of valid three-letter sets of initials
theorem count_three_letter_sets : 
  let first_initials := valid_first_initials
  let second_initials := letters
  let third_initials := letters
  first_initials.card * second_initials.card * third_initials.card = 900 := by
  sorry

end count_three_letter_sets_l48_48398


namespace problem_1_problem_2_l48_48202

-- Definition f
def f (a x : ℝ) : ℝ := a * x + 3 - abs (2 * x - 1)

-- Problem 1: If a = 1, prove ∀ x, f(1, x) ≤ 2
theorem problem_1 : (∀ x : ℝ, f 1 x ≤ 2) :=
sorry

-- Problem 2: The range of a for which f has a maximum value is -2 ≤ a ≤ 2
theorem problem_2 : (∀ a : ℝ, (∀ x : ℝ, (2 * x - 1 > 0 -> (f a x) ≤ (f a ((4 - a) / (2 * (4 - a))))) 
                        ∧ (2 * x - 1 ≤ 0 -> (f a x) ≤ (f a (1 - 2 / (1 - a))))) 
                        ↔ -2 ≤ a ∧ a ≤ 2) :=
sorry

end problem_1_problem_2_l48_48202


namespace temp_below_zero_negative_l48_48615

theorem temp_below_zero_negative (temp_below_zero : ℤ) : temp_below_zero = -3 ↔ temp_below_zero < 0 := by
  sorry

end temp_below_zero_negative_l48_48615


namespace triangular_weight_l48_48142

noncomputable def rectangular_weight := 90
variables {C T : ℕ}

-- Conditions
axiom cond1 : C + T = 3 * C
axiom cond2 : 4 * C + T = T + C + rectangular_weight

-- Question: How much does the triangular weight weigh?
theorem triangular_weight : T = 60 :=
sorry

end triangular_weight_l48_48142


namespace total_six_letter_words_l48_48483

def num_vowels := 6
def vowel_count := 5
def word_length := 6

theorem total_six_letter_words : (num_vowels ^ word_length) = 46656 :=
by sorry

end total_six_letter_words_l48_48483


namespace symmetric_circle_eq_l48_48760

theorem symmetric_circle_eq :
  ∀ (x y : ℝ),
  ((x + 2)^2 + y^2 = 5) →
  (x - y + 1 = 0) →
  (∃ (a b : ℝ), ((a + 1)^2 + (b + 1)^2 = 5)) := 
by
  intros x y h_circle h_line
  -- skip the proof
  sorry

end symmetric_circle_eq_l48_48760


namespace hyperbola_m_value_l48_48670

theorem hyperbola_m_value (m k : ℝ) (h₀ : k > 0) (h₁ : 0 < -m) 
  (h₂ : 2 * k = Real.sqrt (1 + m)) : 
  m = -3 := 
by {
  sorry
}

end hyperbola_m_value_l48_48670


namespace gold_silver_weight_problem_l48_48535

theorem gold_silver_weight_problem (x y : ℕ) (h1 : 9 * x = 11 * y) (h2 : (10 * y + x) - (8 * x + y) = 13) :
  9 * x = 11 * y ∧ (10 * y + x) - (8 * x + y) = 13 :=
by
  refine ⟨h1, h2⟩

end gold_silver_weight_problem_l48_48535


namespace age_of_person_A_l48_48264

-- Definitions corresponding to the conditions
variables (x y z : ℕ)
axiom sum_of_ages : x + y = 70
axiom age_difference_A_B : x - z = y
axiom age_difference_B_A_half : y - z = x / 2

-- The proof statement that needs to be proved
theorem age_of_person_A : x = 42 := by 
  -- This is where the proof would go
  sorry

end age_of_person_A_l48_48264


namespace smallest_sum_a_b_l48_48297

theorem smallest_sum_a_b (a b: ℕ) (h₀: 0 < a) (h₁: 0 < b) (h₂: a ≠ b) (h₃: 1 / (a: ℝ) + 1 / (b: ℝ) = 1 / 15) : a + b = 64 :=
sorry

end smallest_sum_a_b_l48_48297


namespace cos_value_of_tan_third_quadrant_l48_48157

theorem cos_value_of_tan_third_quadrant (x : ℝ) (h1 : Real.tan x = 4 / 3) (h2 : π < x ∧ x < 3 * π / 2) : 
  Real.cos x = -3 / 5 := 
sorry

end cos_value_of_tan_third_quadrant_l48_48157


namespace batsman_total_score_l48_48045

-- We establish our variables and conditions first
variables (T : ℕ) -- total score
variables (boundaries : ℕ := 3) -- number of boundaries
variables (sixes : ℕ := 8) -- number of sixes
variables (boundary_runs_per : ℕ := 4) -- runs per boundary
variables (six_runs_per : ℕ := 6) -- runs per six
variables (running_percentage : ℕ := 50) -- percentage of runs made by running

-- Define the amounts of runs from boundaries and sixes
def runs_from_boundaries := boundaries * boundary_runs_per
def runs_from_sixes := sixes * six_runs_per

-- Main theorem to prove
theorem batsman_total_score :
  T = runs_from_boundaries + runs_from_sixes + T / 2 → T = 120 :=
by
  sorry

end batsman_total_score_l48_48045


namespace chairs_left_l48_48704

-- Conditions
def red_chairs : Nat := 4
def yellow_chairs : Nat := 2 * red_chairs
def blue_chairs : Nat := yellow_chairs - 2
def lisa_borrows : Nat := 3

-- Theorem
theorem chairs_left (chairs_left : Nat) : chairs_left = red_chairs + yellow_chairs + blue_chairs - lisa_borrows :=
by
  sorry

end chairs_left_l48_48704


namespace min_minutes_for_B_cheaper_l48_48427

-- Define the relevant constants and costs associated with each plan
def cost_A (x : ℕ) : ℕ := 12 * x
def cost_B (x : ℕ) : ℕ := 2500 + 6 * x
def cost_C (x : ℕ) : ℕ := 9 * x

-- Lean statement for the proof problem
theorem min_minutes_for_B_cheaper : ∃ (x : ℕ), x = 834 ∧ cost_B x < cost_A x ∧ cost_B x < cost_C x := 
sorry

end min_minutes_for_B_cheaper_l48_48427


namespace bob_cleaning_time_is_correct_l48_48071

-- Definitions for conditions
def timeAliceTakes : ℕ := 32
def bobTimeFactor : ℚ := 3 / 4

-- Theorem to prove
theorem bob_cleaning_time_is_correct : (bobTimeFactor * timeAliceTakes : ℚ) = 24 := 
by
  sorry

end bob_cleaning_time_is_correct_l48_48071


namespace probability_diamond_or_ace_l48_48021

theorem probability_diamond_or_ace (total_cards : ℕ) (diamonds : ℕ) (aces : ℕ) (jokers : ℕ)
  (not_diamonds_nor_aces : ℕ) (p_not_diamond_nor_ace : ℚ) (p_both_not_diamond_nor_ace : ℚ) : 
  total_cards = 54 →
  diamonds = 13 →
  aces = 4 →
  jokers = 2 →
  not_diamonds_nor_aces = 38 →
  p_not_diamond_nor_ace = 19 / 27 →
  p_both_not_diamond_nor_ace = (19 / 27) ^ 2 →
  1 - p_both_not_diamond_nor_ace = 368 / 729 :=
by 
  intros
  sorry

end probability_diamond_or_ace_l48_48021


namespace ratio_brown_eyes_l48_48084

theorem ratio_brown_eyes (total_people : ℕ) (blue_eyes : ℕ) (black_eyes : ℕ) (green_eyes : ℕ) (brown_eyes : ℕ) 
    (h1 : total_people = 100) 
    (h2 : blue_eyes = 19) 
    (h3 : black_eyes = total_people / 4) 
    (h4 : green_eyes = 6) 
    (h5 : brown_eyes = total_people - (blue_eyes + black_eyes + green_eyes)) : 
    brown_eyes / total_people = 1 / 2 :=
by sorry

end ratio_brown_eyes_l48_48084


namespace brick_fence_depth_l48_48645

theorem brick_fence_depth (length height total_bricks : ℕ) 
    (h1 : length = 20) 
    (h2 : height = 5) 
    (h3 : total_bricks = 800) : 
    (total_bricks / (4 * length * height) = 2) := 
by
  sorry

end brick_fence_depth_l48_48645


namespace sum_a1_a11_l48_48848

theorem sum_a1_a11 
  (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 : ℤ) 
  (h1 : a_0 = -512) 
  (h2 : -2 = a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 + a_11) 
  : a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 + a_11 = 510 :=
sorry

end sum_a1_a11_l48_48848


namespace clea_ride_time_l48_48941

noncomputable def walk_down_stopped (x y : ℝ) : Prop := 90 * x = y
noncomputable def walk_down_moving (x y k : ℝ) : Prop := 30 * (x + k) = y
noncomputable def ride_time (y k t : ℝ) : Prop := t = y / k

theorem clea_ride_time (x y k t : ℝ) (h1 : walk_down_stopped x y) (h2 : walk_down_moving x y k) :
  ride_time y k t → t = 45 :=
sorry

end clea_ride_time_l48_48941


namespace triangle_height_l48_48066

def width := 10
def length := 2 * width
def area_rectangle := width * length
def base_triangle := width

theorem triangle_height (h : ℝ) : (1 / 2) * base_triangle * h = area_rectangle → h = 40 :=
by
  sorry

end triangle_height_l48_48066


namespace log5_6_identity_l48_48426

noncomputable def a : ℝ := Real.log 2
noncomputable def b : ℝ := Real.log 10 / Real.log 3

theorem log5_6_identity :
  Real.log 6 / Real.log 5 = ((a * b) + 1) / (b - (a * b)) :=
by sorry

end log5_6_identity_l48_48426


namespace price_reduction_equation_l48_48867

variable (x : ℝ)

theorem price_reduction_equation :
    (58 * (1 - x)^2 = 43) :=
sorry

end price_reduction_equation_l48_48867


namespace rate_of_interest_per_annum_l48_48958

theorem rate_of_interest_per_annum (R : ℝ) : 
  (5000 * R * 2 / 100) + (3000 * R * 4 / 100) = 1540 → 
  R = 7 := 
by {
  sorry
}

end rate_of_interest_per_annum_l48_48958


namespace minimize_average_comprehensive_cost_l48_48868

theorem minimize_average_comprehensive_cost :
  ∀ (f : ℕ → ℝ), (∀ (x : ℕ), x ≥ 10 → f x = 560 + 48 * x + 10800 / x) →
  ∃ x : ℕ, x = 15 ∧ ( ∀ y : ℕ, y ≥ 10 → f y ≥ f 15 ) :=
by
  sorry

end minimize_average_comprehensive_cost_l48_48868


namespace longest_side_range_l48_48024

-- Definitions and conditions
def is_triangle (x y z : ℝ) : Prop := 
  x + y > z ∧ x + z > y ∧ y + z > x

-- Problem statement
theorem longest_side_range (l x y z : ℝ) 
  (h_triangle: is_triangle x y z) 
  (h_perimeter: x + y + z = l / 2) 
  (h_longest: x ≥ y ∧ x ≥ z) : 
  l / 6 ≤ x ∧ x < l / 4 :=
by
  sorry

end longest_side_range_l48_48024


namespace not_possible_to_fill_grid_l48_48423

theorem not_possible_to_fill_grid :
  ¬ ∃ (f : Fin 7 → Fin 7 → ℝ), ∀ i j : Fin 7,
    ((if j > 0 then f i (j - 1) else 0) +
     (if j < 6 then f i (j + 1) else 0) +
     (if i > 0 then f (i - 1) j else 0) +
     (if i < 6 then f (i + 1) j else 0)) = 1 :=
by
  sorry

end not_possible_to_fill_grid_l48_48423


namespace travel_time_difference_l48_48070

theorem travel_time_difference :
  (160 / 40) - (280 / 40) = 3 := by
  sorry

end travel_time_difference_l48_48070


namespace percent_of_g_is_h_l48_48632

variable (a b c d e f g h : ℝ)

-- Conditions
def cond1a : f = 0.60 * a := sorry
def cond1b : f = 0.45 * b := sorry
def cond2a : g = 0.70 * b := sorry
def cond2b : g = 0.30 * c := sorry
def cond3a : h = 0.80 * c := sorry
def cond3b : h = 0.10 * f := sorry
def cond4a : c = 0.30 * a := sorry
def cond4b : c = 0.25 * b := sorry
def cond5a : d = 0.40 * a := sorry
def cond5b : d = 0.35 * b := sorry
def cond6a : e = 0.50 * b := sorry
def cond6b : e = 0.20 * c := sorry

-- Theorem to prove
theorem percent_of_g_is_h (h_percent_g : ℝ) 
  (h_formula : h = h_percent_g * g) : 
  h = 0.285714 * g :=
by
  sorry

end percent_of_g_is_h_l48_48632


namespace factorize_expression_l48_48218

theorem factorize_expression (m x : ℝ) : 
  m^3 * (x - 2) - m * (x - 2) = m * (x - 2) * (m + 1) * (m - 1) := 
by sorry

end factorize_expression_l48_48218


namespace max_min_y_l48_48293

noncomputable def y (x : ℝ) : ℝ := (Real.sin x)^(2:ℝ) + 2 * (Real.sin x) * (Real.cos x) + 3 * (Real.cos x)^(2:ℝ)

theorem max_min_y : 
  ∀ x : ℝ, 
  2 - Real.sqrt 2 ≤ y x ∧ y x ≤ 2 + Real.sqrt 2 :=
by sorry

end max_min_y_l48_48293


namespace pieces_to_same_point_l48_48077

theorem pieces_to_same_point :
  ∀ (x y z : ℤ), (∃ (final_pos : ℤ), (x = final_pos ∧ y = final_pos ∧ z = final_pos)) ↔ 
  (x, y, z) = (1, 2009, 2010) ∨ 
  (x, y, z) = (0, 2009, 2010) ∨ 
  (x, y, z) = (2, 2009, 2010) ∨ 
  (x, y, z) = (3, 2009, 2010) := 
by {
  sorry
}

end pieces_to_same_point_l48_48077


namespace last_term_arithmetic_progression_eq_62_l48_48646

theorem last_term_arithmetic_progression_eq_62
  (a : ℕ) (d : ℕ) (n : ℕ) 
  (h_a : a = 2)
  (h_d : d = 2)
  (h_n : n = 31) : 
  a + (n - 1) * d = 62 :=
by
  sorry

end last_term_arithmetic_progression_eq_62_l48_48646


namespace sequence_sum_l48_48136

-- Definitions representing the given conditions
variables (A H M O X : ℕ)

-- Assuming the conditions as hypotheses
theorem sequence_sum (h₁ : A + 9 + H = 19) (h₂ : 9 + H + M = 19) (h₃ : H + M + O = 19)
  (h₄ : M + O + X = 19) : A + H + M + O = 26 :=
sorry

end sequence_sum_l48_48136


namespace sandra_age_l48_48113

theorem sandra_age (S : ℕ) (h1 : ∀ x : ℕ, x = 14) (h2 : S - 3 = 3 * (14 - 3)) : S = 36 :=
by sorry

end sandra_age_l48_48113


namespace problem_expression_eq_zero_l48_48647

variable {x y : ℝ}

theorem problem_expression_eq_zero (h : x * y ≠ 0) : 
    ( ( (x^2 - 1) / x ) * ( (y^2 - 1) / y ) ) - 
    ( ( (x^2 - 1) / y ) * ( (y^2 - 1) / x ) ) = 0 :=
by
  sorry

end problem_expression_eq_zero_l48_48647


namespace min_value_arithmetic_sequence_l48_48306

theorem min_value_arithmetic_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) (h_arith_seq : a n = 1 + (n - 1) * 1)
  (h_sum : S n = n * (1 + n) / 2) :
  ∃ n, (S n + 8) / a n = 9 / 2 :=
by
  sorry

end min_value_arithmetic_sequence_l48_48306


namespace exist_equilateral_triangle_on_parallel_lines_l48_48246

-- Define the concept of lines and points in a relation to them
def Line := ℝ → ℝ -- For simplicity, let's assume lines are functions

-- Define the points A1, A2, A3
structure Point :=
(x : ℝ)
(y : ℝ)

-- Define the concept of parallel lines
def parallel (D1 D2 : Line) : Prop :=
  ∀ x y, D1 x - D2 x = D1 y - D2 y

axiom D1 : Line
axiom D2 : Line
axiom D3 : Line

-- Ensure the lines are parallel
axiom parallel_D1_D2 : parallel D1 D2
axiom parallel_D2_D3 : parallel D2 D3

-- Main statement to prove
theorem exist_equilateral_triangle_on_parallel_lines :
  ∃ (A1 A2 A3 : Point), 
    (A1.y = D1 A1.x) ∧ 
    (A2.y = D2 A2.x) ∧ 
    (A3.y = D3 A3.x) ∧ 
    ((A1.x - A2.x)^2 + (A1.y - A2.y)^2 = (A2.x - A3.x)^2 + (A2.y - A3.y)^2) ∧ 
    ((A2.x - A3.x)^2 + (A2.y - A3.y)^2 = (A3.x - A1.x)^2 + (A3.y - A1.y)^2) := sorry

end exist_equilateral_triangle_on_parallel_lines_l48_48246


namespace find_f_l48_48022

noncomputable def f (x : ℕ) : ℚ := (1/4) * x * (x + 1) * (2 * x + 1)

lemma f_initial_condition : f 1 = 3 / 2 := by
  sorry

lemma f_functional_equation (x y : ℕ) :
  f (x + y) = (1 + y / (x + 1)) * f x + (1 + x / (y + 1)) * f y + x^2 * y + x * y + x * y^2 := by
  sorry

theorem find_f (x : ℕ) : f x = (1 / 4) * x * (x + 1) * (2 * x + 1) := by
  sorry

end find_f_l48_48022


namespace selena_taco_packages_l48_48178

-- Define the problem conditions
def tacos_per_package : ℕ := 4
def shells_per_package : ℕ := 6
def min_tacos : ℕ := 60
def min_shells : ℕ := 60

-- Lean statement to prove the smallest number of taco packages needed
theorem selena_taco_packages :
  ∃ n : ℕ, (n * tacos_per_package ≥ min_tacos) ∧ (∃ m : ℕ, (m * shells_per_package ≥ min_shells) ∧ (n * tacos_per_package = m * shells_per_package) ∧ n = 15) := 
by {
  sorry
}

end selena_taco_packages_l48_48178


namespace total_sum_lent_l48_48614

theorem total_sum_lent (x : ℚ) (second_part : ℚ) (total_sum : ℚ) (h : second_part = 1688) 
  (h_interest : x * 3/100 * 8 = second_part * 5/100 * 3) : total_sum = 2743 :=
by
  sorry

end total_sum_lent_l48_48614


namespace fraction_simplification_l48_48621

theorem fraction_simplification : (3 : ℚ) / (2 - (3 / 4)) = 12 / 5 := by
  sorry

end fraction_simplification_l48_48621


namespace seventh_numbers_sum_l48_48833

def first_row_seq (n : ℕ) : ℕ := n^2 + n - 1

def second_row_seq (n : ℕ) : ℕ := n * (n + 1) / 2

theorem seventh_numbers_sum :
  first_row_seq 7 + second_row_seq 7 = 83 :=
by
  -- Skipping the proof
  sorry

end seventh_numbers_sum_l48_48833


namespace coefficient_of_x_l48_48085

theorem coefficient_of_x :
  let expr := (5 * (x - 6)) + (6 * (9 - 3 * x ^ 2 + 3 * x)) - (9 * (5 * x - 4))
  (expr : ℝ) → 
  let expr' := 5 * x - 30 + 54 - 18 * x ^ 2 + 18 * x - 45 * x + 36
  (expr' : ℝ) → 
  let coeff_x := 5 + 18 - 45
  coeff_x = -22 :=
by
  sorry

end coefficient_of_x_l48_48085


namespace greatest_integer_with_gcf_5_l48_48191

theorem greatest_integer_with_gcf_5 :
  ∃ n, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
by
  sorry

end greatest_integer_with_gcf_5_l48_48191


namespace min_value_inequality_l48_48644

theorem min_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  ( (x^2 + 4*x + 2) * (y^2 + 5*y + 3) * (z^2 + 6*z + 4) ) / (x * y * z) ≥ 336 := 
by
  sorry

end min_value_inequality_l48_48644


namespace rebus_solution_l48_48970

theorem rebus_solution :
  ∃ (A B C : ℕ), 
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ 
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ 
    (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ∧ 
    A = 4 ∧ B = 7 ∧ C = 6 :=
by
  sorry

end rebus_solution_l48_48970


namespace quadrilateral_area_proof_l48_48899

-- Assume we have a rectangle with area 24 cm^2 and two triangles with total area 7.5 cm^2.
-- We want to prove the area of the quadrilateral ABCD is 16.5 cm^2 inside this rectangle.

def rectangle_area : ℝ := 24
def triangles_area : ℝ := 7.5
def quadrilateral_area : ℝ := rectangle_area - triangles_area

theorem quadrilateral_area_proof : quadrilateral_area = 16.5 := 
by
  exact sorry

end quadrilateral_area_proof_l48_48899


namespace least_ab_value_l48_48041

theorem least_ab_value (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h : (1 : ℚ)/a + (1 : ℚ)/(3 * b) = 1 / 6) : a * b = 98 :=
by
  sorry

end least_ab_value_l48_48041


namespace mixture_cost_in_july_l48_48315

theorem mixture_cost_in_july :
  (∀ C : ℝ, C > 0 → 
    (cost_green_tea_july : ℝ) = 0.1 → 
    (cost_green_tea_july = 0.1 * C) →
    (equal_quantities_mixture:  ℝ) = 1.5 →
    (cost_coffee_july: ℝ) = 2 * C →
    (total_mixture_cost: ℝ) = equal_quantities_mixture * cost_green_tea_july + equal_quantities_mixture * cost_coffee_july →
    total_mixture_cost = 3.15) :=
by
  sorry

end mixture_cost_in_july_l48_48315


namespace log_ordering_correct_l48_48126

noncomputable def log_ordering : Prop :=
  let a := 20.3
  let b := 0.32
  let c := Real.log b
  (0 < b ∧ b < 1) ∧ (c < 0) ∧ (c < b ∧ b < a)

theorem log_ordering_correct : log_ordering :=
by
  -- skipped proof
  sorry

end log_ordering_correct_l48_48126


namespace number_of_possible_third_side_lengths_l48_48512

theorem number_of_possible_third_side_lengths (a b : ℕ) (ha : a = 8) (hb : b = 11) : 
  ∃ n : ℕ, n = 15 ∧ ∀ x : ℕ, (a + b > x) ∧ (a + x > b) ∧ (b + x > a) ↔ (4 ≤ x ∧ x ≤ 18) :=
by
  sorry

end number_of_possible_third_side_lengths_l48_48512


namespace a_sufficient_but_not_necessary_l48_48165

theorem a_sufficient_but_not_necessary (a : ℝ) : 
  (a = 1 → |a| = 1) ∧ (¬ (|a| = 1 → a = 1)) :=
by 
  sorry

end a_sufficient_but_not_necessary_l48_48165


namespace evaluate_expression_l48_48107

theorem evaluate_expression (x : ℝ) (h1 : x^5 + 1 ≠ 0) (h2 : x^5 - 1 ≠ 0) :
  ( ((x^2 - 2*x + 2)^2 * (x^3 - x^2 + 1)^2 / (x^5 + 1)^2)^2 *
    ((x^2 + 2*x + 2)^2 * (x^3 + x^2 + 1)^2 / (x^5 - 1)^2)^2 )
  = 1 := 
by 
  sorry

end evaluate_expression_l48_48107


namespace solve_y_minus_x_l48_48885

theorem solve_y_minus_x (x y : ℝ) (h1 : x + y = 399) (h2 : x / y = 0.9) : y - x = 21 :=
sorry

end solve_y_minus_x_l48_48885


namespace f_of_5_l48_48419

/- The function f(x) is defined by f(x) = x^2 - x. Prove that f(5) = 20. -/
def f (x : ℤ) : ℤ := x^2 - x

theorem f_of_5 : f 5 = 20 := by
  sorry

end f_of_5_l48_48419


namespace area_triangle_ABC_l48_48230

theorem area_triangle_ABC (AB CD height : ℝ) 
  (h_parallel : AB + CD = 20)
  (h_ratio : CD = 3 * AB)
  (h_height : height = (2 * 20) / (AB + CD)) :
  (1 / 2) * AB * height = 5 := sorry

end area_triangle_ABC_l48_48230


namespace roger_toys_l48_48060

theorem roger_toys (initial_money spent_money toy_cost remaining_money toys : ℕ) 
  (h1 : initial_money = 63) 
  (h2 : spent_money = 48) 
  (h3 : toy_cost = 3) 
  (h4 : remaining_money = initial_money - spent_money) 
  (h5 : toys = remaining_money / toy_cost) : 
  toys = 5 := 
by 
  sorry

end roger_toys_l48_48060


namespace parts_per_day_l48_48074

noncomputable def total_parts : ℕ := 400
noncomputable def unfinished_parts_after_3_days : ℕ := 60
noncomputable def excess_parts_after_3_days : ℕ := 20

variables (x y : ℕ)

noncomputable def condition1 : Prop := (3 * x + 2 * y = total_parts - unfinished_parts_after_3_days)
noncomputable def condition2 : Prop := (3 * x + 3 * y = total_parts + excess_parts_after_3_days)

theorem parts_per_day (x y : ℕ) (h1 : condition1 x y) (h2 : condition2 x y) : x = 60 ∧ y = 80 :=
by {
  sorry
}

end parts_per_day_l48_48074


namespace range_of_m_l48_48138

theorem range_of_m (α : ℝ) (m : ℝ) (h : (α > π ∧ α < 3 * π / 2) ∨ (α > 3 * π / 2 ∧ α < 2 * π)) :
  -1 < (Real.sin α) ∧ (Real.sin α) < 0 ∧ (Real.sin α) = (2 * m - 3) / (4 - m) → 
  m ∈ Set.Ioo (-1 : ℝ) (3 / 2 : ℝ) :=
by
  sorry

end range_of_m_l48_48138


namespace total_charge_for_2_hours_l48_48189

theorem total_charge_for_2_hours (A F : ℕ) (h1 : F = A + 35) (h2 : F + 4 * A = 350) : 
  F + A = 161 := 
by 
  sorry

end total_charge_for_2_hours_l48_48189


namespace pests_eaten_by_frogs_in_week_l48_48262

-- Definitions
def pests_per_day_per_frog : ℕ := 80
def days_per_week : ℕ := 7
def number_of_frogs : ℕ := 5

-- Proposition to prove
theorem pests_eaten_by_frogs_in_week : (pests_per_day_per_frog * days_per_week * number_of_frogs) = 2800 := 
by sorry

end pests_eaten_by_frogs_in_week_l48_48262


namespace inequality_proof_l48_48276

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
(h : Real.sqrt a + Real.sqrt b + Real.sqrt c = 3) :
  (a + b) / (2 + a + b) + (b + c) / (2 + b + c) + (c + a) / (2 + c + a) ≥ 3 / 2 := 
sorry

end inequality_proof_l48_48276


namespace no_positive_integer_solutions_l48_48640

theorem no_positive_integer_solutions :
  ¬ ∃ (x1 x2 : ℕ), 903 * x1 + 731 * x2 = 1106 := by
  sorry

end no_positive_integer_solutions_l48_48640


namespace largest_lcm_value_l48_48910

open Nat

theorem largest_lcm_value :
  max (max (max (max (max (Nat.lcm 15 3) (Nat.lcm 15 5)) (Nat.lcm 15 6)) (Nat.lcm 15 9)) (Nat.lcm 15 10)) (Nat.lcm 15 15) = 45 :=
by
  sorry

end largest_lcm_value_l48_48910


namespace center_of_circle_l48_48295

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y - 2 = 0

-- Define the condition for the center of the circle
def is_center_of_circle (a b : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - a)^2 + (y - b)^2 = 4

-- The main theorem to be proved
theorem center_of_circle : is_center_of_circle 1 (-1) :=
by
  sorry

end center_of_circle_l48_48295


namespace cubic_sum_identity_l48_48011

theorem cubic_sum_identity
  (x y z : ℝ)
  (h1 : x + y + z = 8)
  (h2 : x * y + x * z + y * z = 17)
  (h3 : x * y * z = -14) :
  x^3 + y^3 + z^3 = 62 :=
sorry

end cubic_sum_identity_l48_48011


namespace train_speed_120_kmph_l48_48179

theorem train_speed_120_kmph (t : ℝ) (d : ℝ) (h_t : t = 9) (h_d : d = 300) : 
    (d / t) * 3.6 = 120 :=
by
  sorry

end train_speed_120_kmph_l48_48179


namespace remainder_when_divided_by_5_l48_48978

theorem remainder_when_divided_by_5 (k : ℕ) 
  (h1 : k % 6 = 5) 
  (h2 : k % 7 = 3)
  (h3 : k < 41) : k % 5 = 2 :=
sorry

end remainder_when_divided_by_5_l48_48978


namespace annulus_area_of_tangent_segments_l48_48593

theorem annulus_area_of_tangent_segments (r : ℝ) (l : ℝ) (region_area : ℝ) 
  (h_rad : r = 3) (h_len : l = 6) : region_area = 9 * Real.pi :=
sorry

end annulus_area_of_tangent_segments_l48_48593


namespace largest_four_digit_number_mod_l48_48873

theorem largest_four_digit_number_mod (n : ℕ) : 
  (n < 10000) → 
  (n % 11 = 2) → 
  (n % 7 = 4) → 
  n ≤ 9973 :=
by
  sorry

end largest_four_digit_number_mod_l48_48873


namespace evaluate_expression_l48_48462

theorem evaluate_expression (x : ℤ) (h : x = 3) : x^6 - 6 * x^2 + 7 * x = 696 :=
by
  have hx : x = 3 := h
  sorry

end evaluate_expression_l48_48462


namespace linear_function_of_additivity_l48_48037

theorem linear_function_of_additivity (f : ℝ → ℝ) 
  (h_add : ∀ x y : ℝ, f (x + y) = f x + f y) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
sorry

end linear_function_of_additivity_l48_48037


namespace train_passing_tree_time_l48_48962

theorem train_passing_tree_time
  (train_length : ℝ) (train_speed_kmhr : ℝ) (conversion_factor : ℝ)
  (train_speed_ms : train_speed_ms = train_speed_kmhr * conversion_factor) :
  train_length = 500 → train_speed_kmhr = 72 → conversion_factor = 5 / 18 →
  500 / (72 * (5 / 18)) = 25 := 
by
  intros h1 h2 h3
  sorry

end train_passing_tree_time_l48_48962


namespace no_solution_for_equation_l48_48749

theorem no_solution_for_equation (x : ℝ) (hx : x ≠ -1) :
  (5 * x + 2) / (x^2 + x) ≠ 3 / (x + 1) := 
sorry

end no_solution_for_equation_l48_48749


namespace find_f_8_l48_48542

def f (n : ℕ) : ℕ := n^2 - 3 * n + 20

theorem find_f_8 : f 8 = 60 := 
by 
sorry

end find_f_8_l48_48542


namespace find_A_l48_48836

variable {a b : ℝ}

theorem find_A (h : (5 * a + 3 * b)^2 = (5 * a - 3 * b)^2 + A) : A = 60 * a * b :=
sorry

end find_A_l48_48836


namespace calculate_value_l48_48664

theorem calculate_value : 2 * (75 * 1313 - 25 * 1313) = 131300 := 
by 
  sorry

end calculate_value_l48_48664


namespace points_lie_on_line_l48_48006

theorem points_lie_on_line (t : ℝ) (ht : t ≠ 0) :
    let x := (t + 2) / t
    let y := (t - 2) / t
    x + y = 2 :=
by
  let x := (t + 2) / t
  let y := (t - 2) / t
  sorry

end points_lie_on_line_l48_48006


namespace smallest_positive_period_of_f_max_min_values_of_f_in_interval_l48_48847

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (Real.cos x, -1 / 2)
noncomputable def vec_b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.cos (2 * x))
noncomputable def f (x : ℝ) : ℝ := (vec_a x).1 * (vec_b x).1 + (vec_a x).2 * (vec_b x).2

theorem smallest_positive_period_of_f :
  (∀ x : ℝ, f (x + Real.pi) = f x) ∧ (∀ T > 0, (∀ x : ℝ, f (x + T) = f x) → T ≥ Real.pi) :=
by sorry

theorem max_min_values_of_f_in_interval :
  ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ 1 ∧ f x ≥ -1 / 2 :=
by sorry

end smallest_positive_period_of_f_max_min_values_of_f_in_interval_l48_48847


namespace fraction_spent_on_food_l48_48103

theorem fraction_spent_on_food (r c f : ℝ) (l s : ℝ)
  (hr : r = 1/10)
  (hc : c = 3/5)
  (hl : l = 16000)
  (hs : s = 160000)
  (heq : f * s + r * s + c * s + l = s) :
  f = 1/5 :=
by
  sorry

end fraction_spent_on_food_l48_48103


namespace Q_difference_l48_48935

def Q (x n : ℕ) : ℕ :=
  (Finset.range (10^n)).sum (λ k => x / (k + 1))

theorem Q_difference (n : ℕ) : 
  Q (10^n) n - Q (10^n - 1) n = (n + 1)^2 :=
by
  sorry

end Q_difference_l48_48935


namespace min_sum_ab_72_l48_48638

theorem min_sum_ab_72 (a b : ℤ) (h : a * b = 72) : a + b ≥ -17 := sorry

end min_sum_ab_72_l48_48638


namespace initial_quantity_l48_48682

variables {A : ℝ} -- initial quantity of acidic liquid
variables {W : ℝ} -- quantity of water removed

theorem initial_quantity (h1: A * 0.6 = W + 25) (h2: W = 9) : A = 27 :=
by
  sorry

end initial_quantity_l48_48682


namespace water_speed_l48_48129

theorem water_speed (swimmer_speed still_water : ℝ) (distance time : ℝ) (h1 : swimmer_speed = 12) (h2 : distance = 12) (h3 : time = 6) :
  ∃ v : ℝ, v = 10 ∧ distance = (swimmer_speed - v) * time :=
by { sorry }

end water_speed_l48_48129


namespace crackers_initial_count_l48_48270

theorem crackers_initial_count (friends : ℕ) (crackers_per_friend : ℕ) (total_crackers : ℕ) :
  (friends = 4) → (crackers_per_friend = 2) → (total_crackers = friends * crackers_per_friend) → total_crackers = 8 :=
by intros h_friends h_crackers_per_friend h_total_crackers
   rw [h_friends, h_crackers_per_friend] at h_total_crackers
   exact h_total_crackers

end crackers_initial_count_l48_48270


namespace investor_pieces_impossible_to_be_2002_l48_48949

theorem investor_pieces_impossible_to_be_2002 : 
  ¬ ∃ k : ℕ, 1 + 7 * k = 2002 := 
by
  sorry

end investor_pieces_impossible_to_be_2002_l48_48949


namespace greatest_divisor_of_arithmetic_sequence_l48_48352

theorem greatest_divisor_of_arithmetic_sequence (x c : ℕ) : ∃ d, d = 15 ∧ ∀ S, S = 15 * x + 105 * c → d ∣ S :=
by
  sorry

end greatest_divisor_of_arithmetic_sequence_l48_48352


namespace typing_time_l48_48121

theorem typing_time (typing_speed : ℕ) (words_per_page : ℕ) (number_of_pages : ℕ) (h1 : typing_speed = 90) (h2 : words_per_page = 450) (h3 : number_of_pages = 10) : 
  (words_per_page / typing_speed) * number_of_pages = 50 := 
by
  sorry

end typing_time_l48_48121


namespace boy_run_time_l48_48735

section
variables {d1 d2 d3 d4 : ℝ} -- distances
variables {v1 v2 v3 v4 : ℝ} -- velocities
variables {t : ℝ} -- time

-- Define conditions
def distances_and_velocities (d1 d2 d3 d4 v1 v2 v3 v4 : ℝ) :=
  d1 = 25 ∧ d2 = 30 ∧ d3 = 40 ∧ d4 = 35 ∧
  v1 = 3.33 ∧ v2 = 3.33 ∧ v3 = 2.78 ∧ v4 = 2.22

-- Problem statement
theorem boy_run_time
  (h : distances_and_velocities d1 d2 d3 d4 v1 v2 v3 v4) :
  t = (d1 / v1) + (d2 / v2) + (d3 / v3) + (d4 / v4) := 
sorry
end

end boy_run_time_l48_48735


namespace complex_roots_eqn_l48_48151

open Complex

theorem complex_roots_eqn (a b c d e k : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (he : e ≠ 0) 
  (hk1 : a * k^3 + b * k^2 + c * k + d = e)
  (hk2 : b * k^3 + c * k^2 + d * k + e = a) :
  k = 1 ∨ k = -1 ∨ k = Complex.I ∨ k = -Complex.I := 
sorry

end complex_roots_eqn_l48_48151


namespace katherine_savings_multiple_l48_48391

variable (A K : ℕ)

theorem katherine_savings_multiple
  (h1 : A + K = 750)
  (h2 : A - 150 = 1 / 3 * K) :
  2 * K / A = 3 :=
sorry

end katherine_savings_multiple_l48_48391


namespace scrooge_mcduck_max_box_l48_48637

-- Define Fibonacci numbers
def fib : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fib (n+1) + fib n

-- The problem statement: for a given positive integer k (number of coins initially),
-- the maximum box index n into which Scrooge McDuck can place a coin
-- is F_{k+2} - 1.
theorem scrooge_mcduck_max_box (k : ℕ) (h_pos : 0 < k) :
  ∃ n, n = fib (k + 2) - 1 :=
sorry

end scrooge_mcduck_max_box_l48_48637


namespace joe_lowest_dropped_score_l48_48521

theorem joe_lowest_dropped_score (A B C D : ℕ) 
  (hmean_before : (A + B + C + D) / 4 = 35)
  (hmean_after : (A + B + C) / 3 = 40)
  (hdrop : D = min A (min B (min C D))) :
  D = 20 :=
by sorry

end joe_lowest_dropped_score_l48_48521


namespace circle_range_of_m_l48_48039

theorem circle_range_of_m (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + m * x + 2 * m * y + 2 * m^2 + m - 1 = 0 → (2 * m^2 + m - 1 = 0)) → (-2 < m) ∧ (m < 2/3) :=
by
  sorry

end circle_range_of_m_l48_48039


namespace garden_perimeter_l48_48775

theorem garden_perimeter (L B : ℕ) (hL : L = 100) (hB : B = 200) : 
  2 * (L + B) = 600 := by
sorry

end garden_perimeter_l48_48775


namespace symbols_invariance_l48_48486

def final_symbol_invariant (symbols : List Char) : Prop :=
  ∀ (erase : List Char → List Char), 
  (∀ (l : List Char), 
    (erase l = List.cons '+' (List.tail (List.tail l)) ∨ 
    erase l = List.cons '-' (List.tail (List.tail l))) → 
    erase (erase l) = List.cons '+' (List.tail (List.tail (erase l))) ∨ 
    erase (erase l) = List.cons '-' (List.tail (List.tail (erase l)))) →
  (symbols = []) ∨ (symbols = ['+']) ∨ (symbols = ['-'])

theorem symbols_invariance (symbols : List Char) (h : final_symbol_invariant symbols) : 
  ∃ (s : Char), s = '+' ∨ s = '-' :=
  sorry

end symbols_invariance_l48_48486


namespace bicycle_has_four_wheels_l48_48840

variables (Car : Type) (Bicycle : Car) (FourWheeled : Car → Prop)
axiom car_four_wheels : ∀ (c : Car), FourWheeled c

theorem bicycle_has_four_wheels : FourWheeled Bicycle :=
by {
  apply car_four_wheels
}

end bicycle_has_four_wheels_l48_48840


namespace circle_through_point_and_tangent_to_lines_l48_48266

theorem circle_through_point_and_tangent_to_lines :
  ∃ h k,
     ((h, k) = (4 / 5, 3 / 5) ∨ (h, k) = (4, -1)) ∧ 
     ((x - h)^2 + (y - k)^2 = 5) :=
by
  let P := (3, 1)
  let l1 := fun x y => x + 2 * y + 3 
  let l2 := fun x y => x + 2 * y - 7 
  sorry

end circle_through_point_and_tangent_to_lines_l48_48266


namespace cos_neg_pi_over_3_eq_one_half_sin_eq_sqrt3_over_2_solutions_l48_48784

noncomputable def cos_negative_pi_over_3 : Real :=
  Real.cos (-Real.pi / 3)

theorem cos_neg_pi_over_3_eq_one_half :
  cos_negative_pi_over_3 = 1 / 2 :=
  by
    sorry

noncomputable def solutions_sin_eq_sqrt3_over_2 (x : Real) : Prop :=
  Real.sin x = Real.sqrt 3 / 2 ∧ 0 ≤ x ∧ x < 2 * Real.pi

theorem sin_eq_sqrt3_over_2_solutions :
  {x : Real | solutions_sin_eq_sqrt3_over_2 x} = {Real.pi / 3, 2 * Real.pi / 3} :=
  by
    sorry

end cos_neg_pi_over_3_eq_one_half_sin_eq_sqrt3_over_2_solutions_l48_48784


namespace sin_double_angle_l48_48358

theorem sin_double_angle (x : ℝ) (h : Real.sin (π / 4 - x) = 3 / 5) : Real.sin (2 * x) = 7 / 25 := by
  sorry

end sin_double_angle_l48_48358


namespace area_of_garden_l48_48741

variable (w l : ℕ)
variable (h1 : l = 3 * w) 
variable (h2 : 2 * (l + w) = 72)

theorem area_of_garden : l * w = 243 := by
  sorry

end area_of_garden_l48_48741


namespace carla_wins_one_game_l48_48642

/-
We are given the conditions:
Alice, Bob, and Carla each play each other twice in a round-robin format.
Alice won 5 games and lost 3 games.
Bob won 6 games and lost 2 games.
Carla lost 5 games.
We need to prove that Carla won 1 game.
-/

theorem carla_wins_one_game (games_per_match : Nat) 
                            (total_players : Nat)
                            (alice_wins : Nat) 
                            (alice_losses : Nat) 
                            (bob_wins : Nat) 
                            (bob_losses : Nat) 
                            (carla_losses : Nat) :
  (games_per_match = 2) → 
  (total_players = 3) → 
  (alice_wins = 5) → 
  (alice_losses = 3) → 
  (bob_wins = 6) → 
  (bob_losses = 2) → 
  (carla_losses = 5) → 
  ∃ (carla_wins : Nat), 
  carla_wins = 1 := 
by
  intros 
    games_match_eq total_players_eq 
    alice_wins_eq alice_losses_eq 
    bob_wins_eq bob_losses_eq 
    carla_losses_eq
  sorry

end carla_wins_one_game_l48_48642


namespace distinct_integers_integer_expression_l48_48351

theorem distinct_integers_integer_expression 
  (x y z : ℤ) (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) (n : ℕ) : 
  ∃ k : ℤ, k = (x^n / ((x - y) * (x - z)) + y^n / ((y - x) * (y - z)) + z^n / ((z - x) * (z - y))) := 
sorry

end distinct_integers_integer_expression_l48_48351


namespace average_visitors_per_day_l48_48942

/-- The average number of visitors per day in a month of 30 days that begins with a Sunday is 188, 
given that the library has 500 visitors on Sundays and 140 visitors on other days. -/
theorem average_visitors_per_day (visitors_sunday visitors_other : ℕ) (days_in_month : ℕ) 
   (starts_on_sunday : Bool) (sundays : ℕ) 
   (visitors_sunday_eq_500 : visitors_sunday = 500)
   (visitors_other_eq_140 : visitors_other = 140)
   (days_in_month_eq_30 : days_in_month = 30)
   (starts_on_sunday_eq_true : starts_on_sunday = true)
   (sundays_eq_4 : sundays = 4) :
   (visitors_sunday * sundays + visitors_other * (days_in_month - sundays)) / days_in_month = 188 := 
by {
  sorry
}

end average_visitors_per_day_l48_48942


namespace find_m_l48_48871

theorem find_m (m : ℕ) (h1 : 0 ≤ m ∧ m ≤ 9) (h2 : (8 + 4 + 5 + 9) - (6 + m + 3 + 7) % 11 = 0) : m = 9 :=
by
  sorry

end find_m_l48_48871


namespace rebecca_groups_of_eggs_l48_48684

def eggs : Nat := 16
def group_size : Nat := 2

theorem rebecca_groups_of_eggs : (eggs / group_size) = 8 := by
  sorry

end rebecca_groups_of_eggs_l48_48684


namespace angle_ABC_measure_l48_48866

theorem angle_ABC_measure
  (angle_CBD : ℝ)
  (angle_sum_around_B : ℝ)
  (angle_ABD : ℝ)
  (h1 : angle_CBD = 90)
  (h2 : angle_sum_around_B = 200)
  (h3 : angle_ABD = 60) :
  ∃ angle_ABC : ℝ, angle_ABC = 50 :=
by
  sorry

end angle_ABC_measure_l48_48866


namespace estimate_number_of_trees_l48_48067

-- Definitions derived from the conditions
def forest_length : ℝ := 100
def forest_width : ℝ := 0.5
def plot_length : ℝ := 1
def plot_width : ℝ := 0.5
def tree_counts : List ℕ := [65110, 63200, 64600, 64700, 67300, 63300, 65100, 66600, 62800, 65500]

-- The main theorem stating the problem
theorem estimate_number_of_trees :
  let avg_trees_per_plot := tree_counts.sum / tree_counts.length
  let total_plots := (forest_length * forest_width) / (plot_length * plot_width)
  avg_trees_per_plot * total_plots = 6482100 :=
by
  sorry

end estimate_number_of_trees_l48_48067


namespace number_of_pupils_wrong_entry_l48_48674

theorem number_of_pupils_wrong_entry 
  (n : ℕ) (A : ℝ) 
  (h_wrong_entry : ∀ m, (m = 85 → n * (A + 1 / 2) = n * A + 52))
  (h_increase : ∀ m, (m = 33 → n * (A + 1 / 2) = n * A + 52)) 
  : n = 104 := 
sorry

end number_of_pupils_wrong_entry_l48_48674


namespace jordan_time_for_7_miles_l48_48547

noncomputable def time_for_7_miles (jordan_miles : ℕ) (jordan_time : ℤ) : ℤ :=
  jordan_miles * jordan_time 

theorem jordan_time_for_7_miles :
  ∃ jordan_time : ℤ, (time_for_7_miles 7 (16 / 3)) = 112 / 3 :=
by
  sorry

end jordan_time_for_7_miles_l48_48547


namespace eval_expression_l48_48807

theorem eval_expression : (3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (Real.sqrt 3 - 3)) = 3 + (2 * Real.sqrt 3 / 3)) :=
by sorry

end eval_expression_l48_48807


namespace total_students_in_halls_l48_48028

theorem total_students_in_halls :
  let S_g := 30
  let S_b := 2 * S_g
  let S_m := 3 / 5 * (S_g + S_b)
  S_g + S_b + S_m = 144 :=
by
  sorry

end total_students_in_halls_l48_48028


namespace find_g_l48_48440

noncomputable def g (x : ℝ) : ℝ := sorry

theorem find_g (g : ℝ → ℝ)
  (H : ∀ x y : ℝ, (g x * g y - g (x * y)) / 5 = x + y + 4) :
  g = fun x => x + 5 :=
by
  sorry

end find_g_l48_48440


namespace hyperbola_asymptote_l48_48377

theorem hyperbola_asymptote (a : ℝ) (h₀ : a > 0) 
  (h₁ : ∃ (x y : ℝ), (x, y) = (2, 1) ∧ 
       (y = (2 / a) * x ∨ y = -(2 / a) * x)) : a = 4 := by
  sorry

end hyperbola_asymptote_l48_48377


namespace range_of_a_l48_48389

noncomputable def f (x a : ℝ) : ℝ := -x^3 + 3 * x + a

theorem range_of_a (a : ℝ) :
  (∃ (m n p : ℝ), m ≠ n ∧ n ≠ p ∧ m ≠ p ∧ f m a = 2024 ∧ f n a = 2024 ∧ f p a = 2024) ↔
  2022 < a ∧ a < 2026 :=
sorry

end range_of_a_l48_48389


namespace gcd_2728_1575_l48_48823

theorem gcd_2728_1575 : Int.gcd 2728 1575 = 1 :=
by sorry

end gcd_2728_1575_l48_48823


namespace Kaleb_candies_l48_48912

theorem Kaleb_candies 
  (tickets_whack_a_mole : ℕ) 
  (tickets_skee_ball : ℕ) 
  (candy_cost : ℕ)
  (h1 : tickets_whack_a_mole = 8)
  (h2 : tickets_skee_ball = 7)
  (h3 : candy_cost = 5) : 
  (tickets_whack_a_mole + tickets_skee_ball) / candy_cost = 3 := 
by
  sorry

end Kaleb_candies_l48_48912


namespace total_frogs_in_ponds_l48_48597

def pondA_frogs := 32
def pondB_frogs := pondA_frogs / 2

theorem total_frogs_in_ponds : pondA_frogs + pondB_frogs = 48 := by
  sorry

end total_frogs_in_ponds_l48_48597


namespace exists_palindromic_product_l48_48472

open Nat

def is_palindrome (n : ℕ) : Prop :=
  let digits := toDigits 10 n
  digits = digits.reverse

theorem exists_palindromic_product (x : ℕ) (hx : ¬ (10 ∣ x)) : ∃ y : ℕ, is_palindrome (x * y) :=
by
  -- Prove that there exists a natural number y such that x * y is a palindromic number
  sorry

end exists_palindromic_product_l48_48472


namespace simplify_fraction_l48_48659

theorem simplify_fraction (x : ℝ) :
  ((x + 2) / 4) + ((3 - 4 * x) / 3) = (18 - 13 * x) / 12 := by
  sorry

end simplify_fraction_l48_48659


namespace cos_double_angle_l48_48545

variable {α : ℝ}

theorem cos_double_angle (h1 : (Real.tan α - (1 / Real.tan α) = 3 / 2)) (h2 : (α > π / 4) ∧ (α < π / 2)) :
  Real.cos (2 * α) = -3 / 5 := 
sorry

end cos_double_angle_l48_48545


namespace other_store_pools_l48_48156

variable (P A : ℕ)
variable (three_times : P = 3 * A)
variable (total_pools : P + A = 800)

theorem other_store_pools (three_times : P = 3 * A) (total_pools : P + A = 800) : A = 266 := 
by
  sorry

end other_store_pools_l48_48156


namespace initial_noodles_l48_48002

variable (d w e r : ℕ)

-- Conditions
def gave_to_william (w : ℕ) := w = 15
def gave_to_emily (e : ℕ) := e = 20
def remaining_noodles (r : ℕ) := r = 40

-- The statement to be proven
theorem initial_noodles (h1 : gave_to_william w) (h2 : gave_to_emily e) (h3 : remaining_noodles r) : d = w + e + r := by
  -- Proof will be filled in later.
  sorry

end initial_noodles_l48_48002


namespace russom_greatest_number_of_envelopes_l48_48831

theorem russom_greatest_number_of_envelopes :
  ∃ n, n > 0 ∧ 18 % n = 0 ∧ 12 % n = 0 ∧ ∀ m, m > 0 ∧ 18 % m = 0 ∧ 12 % m = 0 → m ≤ n :=
sorry

end russom_greatest_number_of_envelopes_l48_48831


namespace probability_two_white_balls_l48_48290

def bagA := [1, 1]
def bagB := [2, 1]

def total_outcomes := 6
def favorable_outcomes := 2

theorem probability_two_white_balls : (favorable_outcomes : ℚ) / total_outcomes = 1 / 3 :=
by
  sorry

end probability_two_white_balls_l48_48290


namespace lifespan_represents_sample_l48_48680

-- Definitions
def survey_population := 2500
def provinces_and_cities := 11

-- Theorem stating that the lifespan of the urban residents surveyed represents a sample
theorem lifespan_represents_sample
  (number_of_residents : ℕ) (num_provinces : ℕ) 
  (h₁ : number_of_residents = survey_population)
  (h₂ : num_provinces = provinces_and_cities) :
  "Sample" = "Sample" :=
by 
  -- Proof skipped
  sorry

end lifespan_represents_sample_l48_48680


namespace remainder_when_divided_by_4x_minus_8_l48_48065

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := 8 * x^3 - 20 * x^2 + 28 * x - 30

-- Define the divisor d(x)
def d (x : ℝ) : ℝ := 4 * x - 8

-- The specific value where the remainder theorem applies (root of d(x) = 0 is x = 2)
def x₀ : ℝ := 2

-- Prove the remainder when p(x) is divided by d(x) is 10
theorem remainder_when_divided_by_4x_minus_8 :
  (p x₀ = 10) :=
by
  -- The proof will be filled in here.
  sorry

end remainder_when_divided_by_4x_minus_8_l48_48065


namespace ice_cream_melt_l48_48487

theorem ice_cream_melt (r_sphere r_cylinder : ℝ) (h : ℝ)
  (V_sphere : ℝ := (4 / 3) * Real.pi * r_sphere^3)
  (V_cylinder : ℝ := Real.pi * r_cylinder^2 * h)
  (H_equal_volumes : V_sphere = V_cylinder) :
  h = 4 / 9 := by
  sorry

end ice_cream_melt_l48_48487


namespace rearrange_infinite_decimal_l48_48267

-- Define the set of digits
def Digit : Type := Fin 10

-- Define the classes of digits
def Class1 (d : Digit) (dec : ℕ → Digit) : Prop :=
  ∃ n : ℕ, ∀ m : ℕ, m > n → dec m ≠ d

def Class2 (d : Digit) (dec : ℕ → Digit) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, m > n ∧ dec m = d

-- The statement to prove
theorem rearrange_infinite_decimal (dec : ℕ → Digit) (h : ∃ d : Digit, ¬ Class1 d dec) :
  ∃ rearranged : ℕ → Digit, (Class1 d rearranged ∧ Class2 d rearranged) →
  ∃ r : ℚ, ∃ n : ℕ, ∀ m ≥ n, rearranged m = rearranged (m + n) :=
sorry

end rearrange_infinite_decimal_l48_48267


namespace simon_sand_dollars_l48_48258

theorem simon_sand_dollars (S G P : ℕ) (h1 : G = 3 * S) (h2 : P = 5 * G) (h3 : S + G + P = 190) : S = 10 := by
  sorry

end simon_sand_dollars_l48_48258


namespace cos_double_angle_l48_48967

theorem cos_double_angle (θ : ℝ) (h : Real.sin (Real.pi - θ) = 1 / 3) : 
  Real.cos (2 * θ) = 7 / 9 :=
by 
  sorry

end cos_double_angle_l48_48967


namespace tomatoes_picked_yesterday_l48_48852

/-
Given:
1. The farmer initially had 171 tomatoes.
2. The farmer picked some tomatoes yesterday (Y).
3. The farmer picked 30 tomatoes today.
4. The farmer will have 7 tomatoes left after today.

Prove:
The number of tomatoes the farmer picked yesterday (Y) is 134.
-/

theorem tomatoes_picked_yesterday (Y : ℕ) (h : 171 - Y - 30 = 7) : Y = 134 :=
sorry

end tomatoes_picked_yesterday_l48_48852


namespace positive_integer_expression_iff_l48_48505

theorem positive_integer_expression_iff (p : ℕ) : (0 < p) ∧ (∃ k : ℕ, 0 < k ∧ 4 * p + 35 = k * (3 * p - 8)) ↔ p = 3 :=
by
  sorry

end positive_integer_expression_iff_l48_48505


namespace sin_half_angle_product_lt_quarter_l48_48754

theorem sin_half_angle_product_lt_quarter (A B C : ℝ) (h : A + B + C = 180) :
    Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) < 1 / 4 := 
    sorry

end sin_half_angle_product_lt_quarter_l48_48754


namespace probability_not_late_probability_late_and_misses_bus_l48_48324

variable (P_Sam_late : ℚ)
variable (P_miss_bus_given_late : ℚ)

theorem probability_not_late (h1 : P_Sam_late = 5/9) :
  1 - P_Sam_late = 4/9 := by
  rw [h1]
  norm_num

theorem probability_late_and_misses_bus (h1 : P_Sam_late = 5/9) (h2 : P_miss_bus_given_late = 1/3) :
  P_Sam_late * P_miss_bus_given_late = 5/27 := by
  rw [h1, h2]
  norm_num

#check probability_not_late
#check probability_late_and_misses_bus

end probability_not_late_probability_late_and_misses_bus_l48_48324


namespace quadrilateral_area_BEIH_l48_48078

-- Define the necessary points in the problem
structure Point :=
(x : ℚ)
(y : ℚ)

-- Definitions of given points and midpoints
def B : Point := ⟨0, 0⟩
def E : Point := ⟨0, 1.5⟩
def F : Point := ⟨1.5, 0⟩

-- Definitions of line equations from points
def line_DE (p : Point) : Prop := p.y = - (1 / 2) * p.x + 1.5
def line_AF (p : Point) : Prop := p.y = -2 * p.x + 3

-- Intersection points
def I : Point := ⟨3 / 5, 9 / 5⟩
def H : Point := ⟨3 / 4, 3 / 4⟩

-- Function to calculate the area using the Shoelace Theorem
def shoelace_area (a b c d : Point) : ℚ :=
  (1 / 2) * ((a.x * b.y + b.x * c.y + c.x * d.y + d.x * a.y) - (a.y * b.x + b.y * c.x + c.y * d.x + d.y * a.x))

-- The proof statement
theorem quadrilateral_area_BEIH :
  shoelace_area B E I H = 9 / 16 :=
sorry

end quadrilateral_area_BEIH_l48_48078


namespace fish_swim_eastward_l48_48526

-- Define the conditions
variables (E : ℕ)
variable (total_fish_left : ℕ := 2870)
variable (fish_westward : ℕ := 1800)
variable (fish_north : ℕ := 500)
variable (fishwestward_not_caught : ℕ := fish_westward / 4)
variable (fishnorth_not_caught : ℕ := fish_north)
variable (fish_tobe_left_after_caught : ℕ := total_fish_left - fishwestward_not_caught - fishnorth_not_caught)

-- Define the theorem to prove
theorem fish_swim_eastward (h : 3 / 5 * E = fish_tobe_left_after_caught) : E = 3200 := 
by
  sorry

end fish_swim_eastward_l48_48526


namespace find_set_B_l48_48439

def A : Set ℕ := {1, 2}
def B : Set (Set ℕ) := { x | x ⊆ A }

theorem find_set_B : B = { ∅, {1}, {2}, {1, 2} } :=
by
  sorry

end find_set_B_l48_48439


namespace sum_of_roots_l48_48304

variable (x1 x2 k m : ℝ)
variable (h1 : x1 ≠ x2)
variable (h2 : 4 * x1^2 - k * x1 = m)
variable (h3 : 4 * x2^2 - k * x2 = m)

theorem sum_of_roots (x1 x2 k m : ℝ) (h1 : x1 ≠ x2)
  (h2 : 4 * x1 ^ 2 - k * x1 = m) (h3 : 4 * x2 ^ 2 - k * x2 = m) :
  x1 + x2 = k / 4 := sorry

end sum_of_roots_l48_48304


namespace solve_for_c_l48_48195

theorem solve_for_c (a b c : ℝ) (h : 1/a - 1/b = 2/c) : c = (a * b * (b - a)) / 2 := by
  sorry

end solve_for_c_l48_48195


namespace roots_polynomial_sum_l48_48528

theorem roots_polynomial_sum (p q : ℂ) (hp : p^2 - 6 * p + 10 = 0) (hq : q^2 - 6 * q + 10 = 0) :
  p^4 + p^5 * q^3 + p^3 * q^5 + q^4 = 16056 := by
  sorry

end roots_polynomial_sum_l48_48528


namespace ratio_of_sugar_to_flour_l48_48783

theorem ratio_of_sugar_to_flour
  (F B : ℕ)
  (h1 : F = 10 * B)
  (h2 : F = 8 * (B + 60))
  (sugar : ℕ)
  (hs : sugar = 2000) :
  sugar / F = 5 / 6 :=
by {
  sorry -- proof omitted
}

end ratio_of_sugar_to_flour_l48_48783


namespace vegetables_sold_ratio_l48_48764

def totalMassInstalled (carrots zucchini broccoli : ℕ) : ℕ := carrots + zucchini + broccoli

def massSold (soldMass : ℕ) : ℕ := soldMass

def vegetablesSoldRatio (carrots zucchini broccoli soldMass : ℕ) : ℚ :=
  soldMass / (carrots + zucchini + broccoli)

theorem vegetables_sold_ratio
  (carrots zucchini broccoli soldMass : ℕ)
  (h_carrots : carrots = 15)
  (h_zucchini : zucchini = 13)
  (h_broccoli : broccoli = 8)
  (h_soldMass : soldMass = 18) :
  vegetablesSoldRatio carrots zucchini broccoli soldMass = 1 / 2 := by
  sorry

end vegetables_sold_ratio_l48_48764


namespace tanks_fill_l48_48959

theorem tanks_fill
  (c : ℕ) -- capacity of each tank
  (h1 : 300 < c) -- first tank is filled with 300 liters, thus c > 300
  (h2 : 450 < c) -- second tank is filled with 450 liters, thus c > 450
  (h3 : (45 : ℝ) / 100 = (450 : ℝ) / c) -- second tank is 45% filled, thus 0.45 * c = 450
  (h4 : 300 + 450 < 2 * c) -- the two tanks have the same capacity, thus they must have enough capacity to be filled more than 750 liters
  : c - 300 + (c - 450) = 1250 :=
sorry

end tanks_fill_l48_48959


namespace amplitude_of_resultant_wave_l48_48182

noncomputable def y1 (t : ℝ) := 3 * Real.sqrt 2 * Real.sin (100 * Real.pi * t)
noncomputable def y2 (t : ℝ) := 3 * Real.sin (100 * Real.pi * t - Real.pi / 4)
noncomputable def y (t : ℝ) := y1 t + y2 t

theorem amplitude_of_resultant_wave :
  ∃ R : ℝ, R = 3 * Real.sqrt 5 ∧ ∀ t : ℝ, y t = R * Real.sin (100 * Real.pi * t - θ) :=
by
  let y_combined := y
  use 3 * Real.sqrt 5
  sorry

end amplitude_of_resultant_wave_l48_48182


namespace intersection_of_line_with_x_axis_l48_48461

theorem intersection_of_line_with_x_axis 
  (k : ℝ) 
  (h : ∀ x y : ℝ, y = k * x + 4 → (x = -1 ∧ y = 2)) 
  : ∃ x : ℝ, (2 : ℝ) * x + 4 = 0 ∧ x = -2 :=
by {
  sorry
}

end intersection_of_line_with_x_axis_l48_48461


namespace croissant_to_orange_ratio_l48_48275

-- Define the conditions as given in the problem
variables (c o : ℝ)
variable (emily_expenditure : ℝ)
variable (lucas_expenditure : ℝ)

-- Given conditions of expenditures
axiom emily_expenditure_is : emily_expenditure = 5 * c + 4 * o
axiom lucas_expenditure_is : lucas_expenditure = 3 * emily_expenditure
axiom lucas_expenditure_as_purchased : lucas_expenditure = 4 * c + 10 * o

-- Prove the ratio of the cost of a croissant to an orange
theorem croissant_to_orange_ratio : (c / o) = 2 / 11 :=
by sorry

end croissant_to_orange_ratio_l48_48275


namespace marble_game_solution_l48_48239

theorem marble_game_solution (B R : ℕ) (h1 : B + R = 21) (h2 : (B * (B - 1)) / (21 * 20) = 1 / 2) : B^2 + R^2 = 261 :=
by
  sorry

end marble_game_solution_l48_48239


namespace inverse_value_l48_48641

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 25 / (7 + 2 * x)

-- Define the goal of the proof
theorem inverse_value {g : ℝ → ℝ}
  (h : ∀ y, g (g⁻¹ y) = y) :
  ((g⁻¹ 5)⁻¹) = -1 :=
by
  sorry

end inverse_value_l48_48641


namespace percentage_problem_l48_48909

theorem percentage_problem (x : ℝ) (h : (3 / 8) * x = 141) : (round (0.3208 * x) = 121) :=
by
  sorry

end percentage_problem_l48_48909


namespace triangle_is_obtuse_l48_48565

-- Define the sides of the triangle with the given ratio
def a (x : ℝ) := 3 * x
def b (x : ℝ) := 4 * x
def c (x : ℝ) := 6 * x

-- The theorem statement
theorem triangle_is_obtuse (x : ℝ) (hx : 0 < x) : 
  (a x)^2 + (b x)^2 < (c x)^2 :=
by
  sorry

end triangle_is_obtuse_l48_48565


namespace union_of_sets_l48_48283

def A : Set ℕ := {2, 3}
def B : Set ℕ := {1, 2}

theorem union_of_sets : A ∪ B = {1, 2, 3} := by
  sorry

end union_of_sets_l48_48283


namespace circumradius_of_sector_l48_48796

noncomputable def R_circumradius (θ : ℝ) (r : ℝ) := r / (2 * Real.sin (θ / 2))

theorem circumradius_of_sector (r : ℝ) (θ : ℝ) (hθ : θ = 120) (hr : r = 8) :
  R_circumradius θ r = (8 * Real.sqrt 3) / 3 :=
by
  rw [hθ, hr, R_circumradius]
  sorry

end circumradius_of_sector_l48_48796


namespace tangent_line_at_point_l48_48685

theorem tangent_line_at_point (x y : ℝ) (h_curve : y = x^3 - 2 * x + 1) (h_point : (x, y) = (1, 0)) :
  y = x - 1 :=
sorry

end tangent_line_at_point_l48_48685


namespace sufficient_but_not_necessary_l48_48013

theorem sufficient_but_not_necessary {a b : ℝ} (h1 : a > 1) (h2 : b > 2) :
  (a + b > 3 ∧ a * b > 2) ∧ ¬((a + b > 3 ∧ a * b > 2) → (a > 1 ∧ b > 2)) :=
by
  sorry

end sufficient_but_not_necessary_l48_48013


namespace right_triangle_area_perimeter_l48_48574

theorem right_triangle_area_perimeter (a b : ℕ) (h₁ : a = 36) (h₂ : b = 48) : 
  (1/2) * (a * b) = 864 ∧ a + b + Nat.sqrt (a * a + b * b) = 144 := by
  sorry

end right_triangle_area_perimeter_l48_48574


namespace find_fg_minus_gf_l48_48577

def f (x : ℝ) : ℝ := 3 * x^2 + 4 * x - 5
def g (x : ℝ) : ℝ := 2 * x + 1

theorem find_fg_minus_gf (x : ℝ) : f (g x) - g (f x) = 6 * x^2 + 12 * x + 11 := 
by 
  sorry

end find_fg_minus_gf_l48_48577


namespace larry_result_is_correct_l48_48874

theorem larry_result_is_correct (a b c d e : ℤ) 
  (h1: a = 2) (h2: b = 4) (h3: c = 3) (h4: d = 5) (h5: e = -15) :
  a - (b - (c * (d + e))) = (-17 + e) :=
by 
  rw [h1, h2, h3, h4, h5]
  sorry

end larry_result_is_correct_l48_48874


namespace cubes_sum_l48_48599

theorem cubes_sum (a b c : ℝ) (h1 : a + b + c = 1) (h2 : ab + ac + bc = -4) (h3 : abc = -6) :
  a^3 + b^3 + c^3 = -5 :=
by
  sorry

end cubes_sum_l48_48599


namespace joe_first_lift_weight_l48_48428

variables (x y : ℕ)

theorem joe_first_lift_weight (h1 : x + y = 600) (h2 : 2 * x = y + 300) : x = 300 :=
by
  sorry

end joe_first_lift_weight_l48_48428


namespace sum_even_numbers_l48_48801

def is_even (n : ℕ) : Prop := n % 2 = 0

def largest_even_less_than_or_equal (n m : ℕ) : ℕ :=
if h : m % 2 = 0 ∧ m ≤ n then m else
if h : m % 2 = 1 ∧ (m - 1) ≤ n then m - 1 else 0

def smallest_even_less_than_or_equal (n : ℕ) : ℕ :=
if h : 2 ≤ n then 2 else 0

theorem sum_even_numbers (n : ℕ) (h : n = 49) :
  largest_even_less_than_or_equal n 48 + smallest_even_less_than_or_equal n = 50 :=
by sorry

end sum_even_numbers_l48_48801


namespace box_third_dimension_length_l48_48745

noncomputable def box_height (num_cubes : ℕ) (cube_volume : ℝ) (length : ℝ) (width : ℝ) : ℝ :=
  let total_volume := num_cubes * cube_volume
  total_volume / (length * width)

theorem box_third_dimension_length (num_cubes : ℕ) (cube_volume : ℝ) (length : ℝ) (width : ℝ)
  (h_num_cubes : num_cubes = 24)
  (h_cube_volume : cube_volume = 27)
  (h_length : length = 8)
  (h_width : width = 12) :
  box_height num_cubes cube_volume length width = 6.75 :=
by {
  -- proof skipped
  sorry
}

end box_third_dimension_length_l48_48745


namespace real_part_sum_l48_48320

-- Definitions of a and b as real numbers and i as the imaginary unit
variables (a b : ℝ)
def i := Complex.I

-- Condition given in the problem
def given_condition : Prop := (a + b * i) / (2 - i) = 3 + i

-- Statement to prove
theorem real_part_sum : given_condition a b → a + b = 20 := by
  sorry

end real_part_sum_l48_48320


namespace marble_problem_l48_48981

-- Defining the problem in Lean statement
theorem marble_problem 
  (m : ℕ) (n k : ℕ) (hx : m = 220) (hy : n = 20) : 
  (∀ x : ℕ, (k = n + x) → (m / n = 11) → (m / k = 10)) → (x = 2) :=
by {
  sorry
}

end marble_problem_l48_48981


namespace circle_properties_l48_48017

noncomputable def circle_center_and_radius (x y: ℝ) : Prop :=
  (x^2 + 8*x + y^2 - 10*y = 11)

theorem circle_properties :
  (∃ (a b r : ℝ), (a, b) = (-4, 5) ∧ r = 2 * Real.sqrt 13 ∧ circle_center_and_radius x y → a + b + r = 1 + 2 * Real.sqrt 13) :=
  sorry

end circle_properties_l48_48017


namespace rachel_colored_pictures_l48_48819

theorem rachel_colored_pictures :
  ∃ b1 b2 : ℕ, b1 = 23 ∧ b2 = 32 ∧ ∃ remaining: ℕ, remaining = 11 ∧ (b1 + b2) - remaining = 44 :=
by
  sorry

end rachel_colored_pictures_l48_48819


namespace complement_U_A_l48_48154

open Set

def U : Set ℤ := {-2, -1, 0, 1, 2}
def A : Set ℤ := {x | x^2 < 3}

theorem complement_U_A :
  (U \ A) = {-2, 2} :=
sorry

end complement_U_A_l48_48154


namespace largest_prime_factor_problem_l48_48222

def largest_prime_factor (n : ℕ) : ℕ :=
  -- This function calculates the largest prime factor of n
  sorry

theorem largest_prime_factor_problem :
  largest_prime_factor 57 = 19 ∧
  largest_prime_factor 133 = 19 ∧
  ∀ n, n = 63 ∨ n = 85 ∨ n = 143 → largest_prime_factor n < 19 :=
by
  sorry

end largest_prime_factor_problem_l48_48222


namespace length_of_train_is_110_l48_48273

-- Define the speeds and time as constants
def speed_train_kmh := 90
def speed_man_kmh := 9
def time_pass_seconds := 4

-- Define the conversion factor from km/h to m/s
def kmh_to_mps (kmh : ℕ) : ℚ := (kmh : ℚ) * (5 / 18)

-- Calculate relative speed in m/s
def relative_speed_mps : ℚ := kmh_to_mps (speed_train_kmh + speed_man_kmh)

-- Define the length of the train in meters
def length_of_train : ℚ := relative_speed_mps * time_pass_seconds

-- The theorem to prove: The length of the train is 110 meters
theorem length_of_train_is_110 : length_of_train = 110 := 
by sorry

end length_of_train_is_110_l48_48273


namespace most_suitable_candidate_l48_48787

-- Definitions for variances
def variance_A := 3.4
def variance_B := 2.1
def variance_C := 2.5
def variance_D := 2.7

-- We start the theorem to state the most suitable candidate based on given variances and average scores.
theorem most_suitable_candidate :
  (variance_A = 3.4) ∧ (variance_B = 2.1) ∧ (variance_C = 2.5) ∧ (variance_D = 2.7) →
  true := 
by
  sorry

end most_suitable_candidate_l48_48787


namespace find_sequence_term_l48_48424

noncomputable def sequence_sum (n : ℕ) : ℚ :=
  (2 / 3) * n^2 - (1 / 3) * n

def sequence_term (n : ℕ) : ℚ :=
  if n = 1 then (1 / 3) else (4 / 3) * n - 1

theorem find_sequence_term (n : ℕ) : sequence_term n = (sequence_sum n - sequence_sum (n - 1)) :=
by
  unfold sequence_sum
  unfold sequence_term
  sorry

end find_sequence_term_l48_48424


namespace arc_length_calc_l48_48455

-- Defining the conditions
def circle_radius := 12 -- radius OR
def angle_RIP := 30 -- angle in degrees

-- Defining the goal
noncomputable def arc_length_RP := 4 * Real.pi -- length of arc RP

-- The statement to prove
theorem arc_length_calc :
  arc_length_RP = 4 * Real.pi :=
sorry

end arc_length_calc_l48_48455


namespace profit_growth_equation_l48_48757

noncomputable def profitApril : ℝ := 250000
noncomputable def profitJune : ℝ := 360000
noncomputable def averageMonthlyGrowth (x : ℝ) : ℝ := 25 * (1 + x) * (1 + x)

theorem profit_growth_equation (x : ℝ) :
  averageMonthlyGrowth x = 36 * 10000 ↔ 25 * (1 + x)^2 = 36 :=
by
  sorry

end profit_growth_equation_l48_48757


namespace handshake_problem_l48_48591

noncomputable def number_of_handshakes (n : ℕ) : ℕ :=
  n.choose 2

theorem handshake_problem : number_of_handshakes 25 = 300 := 
  by
  sorry

end handshake_problem_l48_48591


namespace polynomial_divisibility_l48_48261

theorem polynomial_divisibility (a b c : ℝ) :
  (∀ x : ℝ, (x ^ 4 + a * x ^ 2 + b * x + c) = (x - 1) ^ 3 * (x + 1) →
  a = 0 ∧ b = 2 ∧ c = -1) :=
by
  intros x h
  sorry

end polynomial_divisibility_l48_48261


namespace point_on_angle_bisector_l48_48844

theorem point_on_angle_bisector (a b : ℝ) (h : (a, b) = (b, a)) : a = b ∨ a = -b := 
by
  sorry

end point_on_angle_bisector_l48_48844


namespace betty_needs_more_money_l48_48563

-- Define the variables and conditions
def wallet_cost : ℕ := 100
def parents_gift : ℕ := 15
def grandparents_gift : ℕ := parents_gift * 2
def initial_betty_savings : ℕ := wallet_cost / 2
def total_savings : ℕ := initial_betty_savings + parents_gift + grandparents_gift

-- Prove that Betty needs 5 more dollars to buy the wallet
theorem betty_needs_more_money : total_savings + 5 = wallet_cost :=
by
  sorry

end betty_needs_more_money_l48_48563


namespace Shinyoung_ate_most_of_cake_l48_48299

noncomputable def Shinyoung_portion := (1 : ℚ) / 3
noncomputable def Seokgi_portion := (1 : ℚ) / 4
noncomputable def Woong_portion := (1 : ℚ) / 5

theorem Shinyoung_ate_most_of_cake :
  Shinyoung_portion > Seokgi_portion ∧ Shinyoung_portion > Woong_portion := by
  sorry

end Shinyoung_ate_most_of_cake_l48_48299


namespace monotonicity_of_f_f_greater_than_2_ln_a_plus_3_div_2_l48_48513

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem monotonicity_of_f (a : ℝ) :
  (a ≤ 0 → ∀ x y, x < y → f x a > f y a) ∧
  (a > 0 →
    (∀ x, x < Real.log (1 / a) → f x a > f (Real.log (1 / a)) a) ∧
    (∀ x, x > Real.log (1 / a) → f x a > f (Real.log (1 / a)) a)) :=
sorry

theorem f_greater_than_2_ln_a_plus_3_div_2 (a : ℝ) (h : a > 0) (x : ℝ) :
  f x a > 2 * Real.log a + 3 / 2 :=
sorry

end monotonicity_of_f_f_greater_than_2_ln_a_plus_3_div_2_l48_48513


namespace original_number_is_500_l48_48906

theorem original_number_is_500 (x : ℝ) (h1 : x * 1.3 = 650) : x = 500 :=
sorry

end original_number_is_500_l48_48906


namespace volume_removed_percentage_l48_48217

noncomputable def volume_rect_prism (l w h : ℝ) : ℝ :=
  l * w * h

noncomputable def volume_cube (s : ℝ) : ℝ :=
  s * s * s

noncomputable def percent_removed (original_volume removed_volume : ℝ) : ℝ :=
  (removed_volume / original_volume) * 100

theorem volume_removed_percentage :
  let l := 18
  let w := 12
  let h := 10
  let cube_side := 4
  let num_cubes := 8
  let original_volume := volume_rect_prism l w h
  let removed_volume := num_cubes * volume_cube cube_side
  percent_removed original_volume removed_volume = 23.7 := 
sorry

end volume_removed_percentage_l48_48217


namespace final_volume_solution_l48_48531

variables (V2 V12 V_final : ℝ)

-- Given conditions
def V2_percent_solution (V2 : ℝ) := true
def V12_percent_solution (V12 : ℝ) := V12 = 18
def mixture_equation (V2 V12 V_final : ℝ) := 0.02 * V2 + 0.12 * V12 = 0.05 * V_final
def total_volume (V2 V12 V_final : ℝ) := V_final = V2 + V12

theorem final_volume_solution (V2 V_final : ℝ) (hV2: V2_percent_solution V2)
    (hV12 : V12_percent_solution V12) (h_mix : mixture_equation V2 V12 V_final)
    (h_total : total_volume V2 V12 V_final) : V_final = 60 :=
sorry

end final_volume_solution_l48_48531


namespace cost_of_each_item_number_of_purchasing_plans_l48_48604

-- Question 1: Cost of each item
theorem cost_of_each_item : 
  ∃ (x y : ℕ), 
    (10 * x + 5 * y = 2000) ∧ 
    (5 * x + 3 * y = 1050) ∧ 
    (x = 150) ∧ 
    (y = 100) :=
by
    sorry

-- Question 2: Number of different purchasing plans
theorem number_of_purchasing_plans : 
  (∀ (a b : ℕ), 
    (150 * a + 100 * b = 4000) → 
    (a ≥ 12) → 
    (b ≥ 12) → 
    (4 = 4)) :=
by
    sorry

end cost_of_each_item_number_of_purchasing_plans_l48_48604


namespace inscribed_circle_radius_in_quarter_circle_l48_48730

theorem inscribed_circle_radius_in_quarter_circle (R r : ℝ) (hR : R = 4) :
  (r + r * Real.sqrt 2 = R) ↔ r = 4 * Real.sqrt 2 - 4 := by
  sorry

end inscribed_circle_radius_in_quarter_circle_l48_48730


namespace greatest_number_dividing_1642_and_1856_l48_48503

theorem greatest_number_dividing_1642_and_1856 (a b r1 r2 k : ℤ) (h_intro : a = 1642) (h_intro2 : b = 1856) 
    (h_r1 : r1 = 6) (h_r2 : r2 = 4) (h_k1 : k = Int.gcd (a - r1) (b - r2)) :
    k = 4 :=
by
  sorry

end greatest_number_dividing_1642_and_1856_l48_48503


namespace a_c_sum_l48_48921

theorem a_c_sum (a b c d : ℕ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : d = a * b * c) (h5 : 233 % d = 79) : a + c = 13 :=
sorry

end a_c_sum_l48_48921


namespace f_sum_lt_zero_l48_48595

theorem f_sum_lt_zero {f : ℝ → ℝ} (h_odd : ∀ x, f (-x) = -f x) (h_monotone : ∀ x y, x < y → f y < f x)
  (α β γ : ℝ) (h1 : α + β > 0) (h2 : β + γ > 0) (h3 : γ + α > 0) :
  f α + f β + f γ < 0 :=
sorry

end f_sum_lt_zero_l48_48595


namespace geometric_series_first_term_l48_48617

theorem geometric_series_first_term (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
by
  sorry

end geometric_series_first_term_l48_48617


namespace find_a_range_l48_48696

theorem find_a_range (a : ℝ) (x : ℝ) (h1 : a * x < 6) (h2 : (3 * x - 6 * a) / 2 > a / 3 - 1) :
  a ≤ -3 / 2 :=
sorry

end find_a_range_l48_48696


namespace aiyanna_more_cookies_than_alyssa_l48_48711

-- Definitions of the conditions
def alyssa_cookies : ℕ := 129
def aiyanna_cookies : ℕ := 140

-- The proof problem statement
theorem aiyanna_more_cookies_than_alyssa : (aiyanna_cookies - alyssa_cookies) = 11 := sorry

end aiyanna_more_cookies_than_alyssa_l48_48711


namespace triangle_perimeter_l48_48308

/-- The lengths of two sides of a triangle are 3 and 5 respectively. The third side is a root of the equation x^2 - 7x + 12 = 0. Find the perimeter of the triangle. -/
theorem triangle_perimeter :
  let side1 := 3
  let side2 := 5
  let third_side1 := 3
  let third_side2 := 4
  (third_side1 * third_side1 - 7 * third_side1 + 12 = 0) ∧
  (third_side2 * third_side2 - 7 * third_side2 + 12 = 0) →
  (side1 + side2 + third_side1 = 11 ∨ side1 + side2 + third_side2 = 12) :=
by
  sorry

end triangle_perimeter_l48_48308


namespace find_k_l48_48120

noncomputable def proof_problem (x1 x2 x3 x4 : ℝ) (k : ℝ) : Prop :=
  (x1 + x2) / (x3 + x4) = k ∧
  (x3 + x4) / (x1 + x2) = k ∧
  (x1 + x3) / (x2 + x4) = k ∧
  (x2 + x4) / (x1 + x3) = k ∧
  (x1 + x4) / (x2 + x3) = k ∧
  (x2 + x3) / (x1 + x4) = k ∧
  x1 ≠ x2 ∨ x2 ≠ x3 ∨ x3 ≠ x4 ∨ x4 ≠ x1

theorem find_k (x1 x2 x3 x4 : ℝ) (h : proof_problem x1 x2 x3 x4 k) : k = -1 :=
  sorry

end find_k_l48_48120


namespace molecular_weight_C4H10_l48_48560

theorem molecular_weight_C4H10 (molecular_weight_six_moles : ℝ) (h : molecular_weight_six_moles = 390) :
  molecular_weight_six_moles / 6 = 65 :=
by
  -- proof to be filled in here
  sorry

end molecular_weight_C4H10_l48_48560


namespace evaluateExpression_at_3_l48_48116

noncomputable def evaluateExpression (x : ℚ) : ℚ :=
  (x - 1 + (2 - 2 * x) / (x + 1)) / ((x * x - x) / (x + 1))

theorem evaluateExpression_at_3 : evaluateExpression 3 = 2 / 3 := by
  sorry

end evaluateExpression_at_3_l48_48116


namespace steve_family_time_l48_48185

theorem steve_family_time :
  let hours_per_day := 24
  let hours_sleeping := hours_per_day * (1/3)
  let hours_school := hours_per_day * (1/6)
  let hours_assignments := hours_per_day * (1/12)
  hours_per_day - (hours_sleeping + hours_school + hours_assignments) = 10 :=
by
  let hours_per_day := 24
  let hours_sleeping := hours_per_day * (1/3)
  let hours_school := hours_per_day * (1/6)
  let hours_assignments := hours_per_day * (1/12)
  sorry

end steve_family_time_l48_48185


namespace unique_integer_triplet_solution_l48_48051

theorem unique_integer_triplet_solution (x y z : ℤ) (h : x^2 + y^2 + z^2 = 2 * x * y * z) : 
    (x = 0 ∧ y = 0 ∧ z = 0) :=
by
  sorry

end unique_integer_triplet_solution_l48_48051


namespace alice_zoe_difference_l48_48244

-- Definitions of the conditions
def AliceApples := 8
def ZoeApples := 2

-- Theorem statement to prove the difference in apples eaten
theorem alice_zoe_difference : AliceApples - ZoeApples = 6 := by
  -- Proof
  sorry

end alice_zoe_difference_l48_48244


namespace max_value_of_expression_l48_48069

noncomputable def max_value (x : ℝ) : ℝ :=
  x * (1 + x) * (3 - x)

theorem max_value_of_expression :
  ∃ x : ℝ, 0 < x ∧ max_value x = (70 + 26 * Real.sqrt 13) / 27 :=
sorry

end max_value_of_expression_l48_48069


namespace quadratic_roots_diff_square_l48_48150

theorem quadratic_roots_diff_square :
  ∀ (d e : ℝ), (∀ x : ℝ, 4 * x^2 + 8 * x - 48 = 0 → (x = d ∨ x = e)) → (d - e)^2 = 49 :=
by
  intros d e h
  sorry

end quadratic_roots_diff_square_l48_48150


namespace max_k_value_condition_l48_48252

theorem max_k_value_condition (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  ∃ k, k = 100 ∧ (∀ k < 100, ∃ (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c), 
   (k * a * b * c / (a + b + c) <= (a + b)^2 + (a + b + 4 * c)^2)) :=
sorry

end max_k_value_condition_l48_48252


namespace jim_total_weight_per_hour_l48_48663

theorem jim_total_weight_per_hour :
  let hours := 8
  let gold_chest := 100
  let gold_bag := 50
  let gold_extra := 30 + 20 + 10
  let silver := 30
  let bronze := 50
  let weight_gold := 10
  let weight_silver := 5
  let weight_bronze := 2
  let total_gold := gold_chest + 2 * gold_bag + gold_extra
  let total_weight := total_gold * weight_gold + silver * weight_silver + bronze * weight_bronze
  total_weight / hours = 356.25 := by
  sorry

end jim_total_weight_per_hour_l48_48663


namespace number_of_insects_l48_48850

theorem number_of_insects (total_legs : ℕ) (legs_per_insect : ℕ) (h1 : total_legs = 48) (h2 : legs_per_insect = 6) : (total_legs / legs_per_insect) = 8 := by
  sorry

end number_of_insects_l48_48850


namespace inequality_solution_l48_48853

theorem inequality_solution (x : ℝ) (h : 1 - x > x - 1) : x < 1 :=
sorry

end inequality_solution_l48_48853


namespace zero_points_product_l48_48087

noncomputable def f (a x : ℝ) : ℝ := abs (Real.log x / Real.log a) - (1 / 2) ^ x

theorem zero_points_product (a x1 x2 : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1)
  (hx1_zero : f a x1 = 0) (hx2_zero : f a x2 = 0) : 0 < x1 * x2 ∧ x1 * x2 < 1 :=
by
  sorry

end zero_points_product_l48_48087


namespace number_of_ways_to_choose_officers_l48_48209

open Nat

theorem number_of_ways_to_choose_officers (n : ℕ) (h : n = 8) : 
  n * (n - 1) * (n - 2) = 336 := by
  sorry

end number_of_ways_to_choose_officers_l48_48209


namespace solution_set_l48_48499

variable (x : ℝ)

def condition_1 : Prop := 2 * x - 4 ≤ 0
def condition_2 : Prop := -x + 1 < 0

theorem solution_set : (condition_1 x ∧ condition_2 x) ↔ (1 < x ∧ x ≤ 2) := by
sorry

end solution_set_l48_48499


namespace evaluate_expression_l48_48817

theorem evaluate_expression (x y : ℕ) (h₁ : x = 3) (h₂ : y = 4) : 
  (x^5 + 3*y^2 + 7) / (x + 4) = 298 / 7 := by
  sorry

end evaluate_expression_l48_48817


namespace boat_speed_in_still_water_l48_48592

-- Definitions and conditions
def Vs : ℕ := 5  -- Speed of the stream in km/hr
def distance : ℕ := 135  -- Distance traveled in km
def time : ℕ := 5  -- Time in hours

-- Statement to prove
theorem boat_speed_in_still_water : 
  ((distance = (Vb + Vs) * time) -> Vb = 22) :=
by
  sorry

end boat_speed_in_still_water_l48_48592


namespace original_price_of_sarees_l48_48700

theorem original_price_of_sarees 
  (P : ℝ) 
  (h1 : 0.72 * P = 144) : 
  P = 200 := 
sorry

end original_price_of_sarees_l48_48700


namespace maximize_S_n_at_24_l48_48548

noncomputable def a_n (n : ℕ) : ℝ := 142 + (n - 1) * (-2)
noncomputable def b_n (n : ℕ) : ℝ := 142 + (n - 1) * (-6)
noncomputable def S_n (n : ℕ) : ℝ := (n / 2.0) * (2 * 142 + (n - 1) * (-6))

theorem maximize_S_n_at_24 : ∀ (n : ℕ), S_n n ≤ S_n 24 :=
by sorry

end maximize_S_n_at_24_l48_48548


namespace reciprocal_of_sum_frac_is_correct_l48_48476

/-- The reciprocal of the sum of the fractions 1/4 and 1/6 is 12/5. -/
theorem reciprocal_of_sum_frac_is_correct:
  (1 / (1 / 4 + 1 / 6)) = (12 / 5) :=
by 
  sorry

end reciprocal_of_sum_frac_is_correct_l48_48476


namespace construct_quadratic_l48_48944

-- Definitions from the problem's conditions
def quadratic_has_zeros (f : ℝ → ℝ) (r1 r2 : ℝ) : Prop :=
  f r1 = 0 ∧ f r2 = 0

def quadratic_value_at (f : ℝ → ℝ) (x_val value : ℝ) : Prop :=
  f x_val = value

-- Construct the Lean theorem statement
theorem construct_quadratic :
  ∃ f : ℝ → ℝ, quadratic_has_zeros f 1 5 ∧ quadratic_value_at f 3 10 ∧
  ∀ x, f x = (-5/2 : ℝ) * x^2 + 15 * x - 25 / 2 :=
sorry

end construct_quadratic_l48_48944


namespace average_marks_l48_48601

theorem average_marks (avg1 avg2 : ℝ) (n1 n2 : ℕ) 
  (h_avg1 : avg1 = 40) 
  (h_avg2 : avg2 = 60) 
  (h_n1 : n1 = 25) 
  (h_n2 : n2 = 30) : 
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 50.91 := 
by
  sorry

end average_marks_l48_48601


namespace regular_pay_limit_l48_48729

theorem regular_pay_limit (x : ℝ) : 3 * x + 6 * 13 = 198 → x = 40 :=
by
  intro h
  -- proof skipped
  sorry

end regular_pay_limit_l48_48729


namespace line_circle_intersection_range_l48_48771

theorem line_circle_intersection_range (b : ℝ) :
    (2 - Real.sqrt 2) < b ∧ b < (2 + Real.sqrt 2) ↔
    ∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧ ((p1.1 - 2)^2 + p1.2^2 = 1) ∧ ((p2.1 - 2)^2 + p2.2^2 = 1) ∧ (p1.2 = p1.1 - b ∧ p2.2 = p2.1 - b) :=
by
  sorry

end line_circle_intersection_range_l48_48771


namespace probability_first_third_fifth_correct_probability_exactly_three_hits_correct_l48_48422

noncomputable def probability_first_third_fifth_hit : ℚ :=
  (3 / 5) * (2 / 5) * (3 / 5) * (2 / 5) * (3 / 5)

noncomputable def binomial_coefficient (n k : ℕ) : ℚ :=
  ↑(Nat.factorial n) / (↑(Nat.factorial k) * ↑(Nat.factorial (n - k)))

noncomputable def probability_exactly_three_hits : ℚ :=
  binomial_coefficient 5 3 * (3 / 5)^3 * (2 / 5)^2

theorem probability_first_third_fifth_correct :
  probability_first_third_fifth_hit = 108 / 3125 :=
by sorry

theorem probability_exactly_three_hits_correct :
  probability_exactly_three_hits = 216 / 625 :=
by sorry

end probability_first_third_fifth_correct_probability_exactly_three_hits_correct_l48_48422


namespace sum_of_cubes_of_roots_l48_48265

theorem sum_of_cubes_of_roots:
  (∀ r s t : ℝ, (r + s + t = 8) ∧ (r * s + s * t + t * r = 9) ∧ (r * s * t = 2) → r^3 + s^3 + t^3 = 344) :=
by
  intros r s t h
  have h1 := h.1
  have h2 := h.2.1
  have h3 := h.2.2
  sorry

end sum_of_cubes_of_roots_l48_48265


namespace soak_time_l48_48931

/-- 
Bill needs to soak his clothes for 4 minutes to get rid of each grass stain.
His clothes have 3 grass stains and 1 marinara stain.
The total soaking time is 19 minutes.
Prove that the number of minutes needed to soak for each marinara stain is 7.
-/
theorem soak_time (m : ℕ) (grass_stain_time : ℕ) (num_grass_stains : ℕ) (num_marinara_stains : ℕ) (total_time : ℕ)
  (h1 : grass_stain_time = 4)
  (h2 : num_grass_stains = 3)
  (h3 : num_marinara_stains = 1)
  (h4 : total_time = 19) :
  m = 7 :=
by sorry

end soak_time_l48_48931


namespace cost_of_horse_l48_48063

theorem cost_of_horse (H C : ℝ) 
  (h1 : 4 * H + 9 * C = 13400)
  (h2 : 0.4 * H + 1.8 * C = 1880) :
  H = 2000 :=
by
  sorry

end cost_of_horse_l48_48063


namespace min_value_expression_l48_48289

variable {a b c : ℝ}

theorem min_value_expression (h1 : a < b) (h2 : a > 0) (h3 : b^2 - 4 * a * c ≤ 0) : 
  ∃ m : ℝ, m = 3 ∧ (∀ x : ℝ, ((a + b + c) / (b - a)) ≥ m) := 
sorry

end min_value_expression_l48_48289


namespace bicycle_helmet_lock_costs_l48_48410

-- Given total cost, relationships between costs, and the specific costs
theorem bicycle_helmet_lock_costs (H : ℝ) (bicycle helmet lock : ℝ) 
  (h1 : bicycle = 5 * H) 
  (h2 : helmet = H) 
  (h3 : lock = H / 2)
  (total_cost : bicycle + helmet + lock = 360) :
  H = 55.38 ∧ bicycle = 276.90 ∧ lock = 27.72 :=
by 
  -- The proof would go here
  sorry

end bicycle_helmet_lock_costs_l48_48410


namespace pythagorean_triple_l48_48932

theorem pythagorean_triple {c a b : ℕ} (h1 : a = 24) (h2 : b = 7) (h3 : c = 25) : a^2 + b^2 = c^2 :=
by
  rw [h1, h2, h3]
  norm_num

end pythagorean_triple_l48_48932


namespace number_of_regions_on_sphere_l48_48420

theorem number_of_regions_on_sphere (n : ℕ) (h : ∀ {a b c: ℤ}, a ≠ b → b ≠ c → a ≠ c → True) : 
  ∃ a_n, a_n = n^2 - n + 2 := 
by
  sorry

end number_of_regions_on_sphere_l48_48420


namespace simplest_quadratic_radical_problem_l48_48623

/-- The simplest quadratic radical -/
def simplest_quadratic_radical (r : ℝ) : Prop :=
  ((∀ a b : ℝ, r = a * b → b = 1 ∧ a = r) ∧ (∀ a b : ℝ, r ≠ a / b))

theorem simplest_quadratic_radical_problem :
  (simplest_quadratic_radical (Real.sqrt 6)) ∧ 
  (¬ simplest_quadratic_radical (Real.sqrt 8)) ∧ 
  (¬ simplest_quadratic_radical (Real.sqrt (1/3))) ∧ 
  (¬ simplest_quadratic_radical (Real.sqrt 4)) :=
by
  sorry

end simplest_quadratic_radical_problem_l48_48623


namespace coin_loading_impossible_l48_48630

theorem coin_loading_impossible (p q : ℝ) (hp : p ≠ 1 - p) (hq : q ≠ 1 - q) :
  ¬ (p * q = 1/4 ∧ p * (1 - q) = 1/4 ∧ (1 - p) * q = 1/4 ∧ (1 - p) * (1 - q) = 1/4) :=
sorry

end coin_loading_impossible_l48_48630


namespace four_b_is_222_22_percent_of_a_l48_48339

-- noncomputable is necessary because Lean does not handle decimal numbers directly
noncomputable def a (b : ℝ) : ℝ := 1.8 * b
noncomputable def four_b (b : ℝ) : ℝ := 4 * b

theorem four_b_is_222_22_percent_of_a (b : ℝ) : four_b b = 2.2222 * a b := 
by
  sorry

end four_b_is_222_22_percent_of_a_l48_48339


namespace simplify_div_l48_48712

theorem simplify_div : (27 * 10^12) / (9 * 10^4) = 3 * 10^8 := 
by
  sorry

end simplify_div_l48_48712


namespace percentage_cut_l48_48561

def original_budget : ℝ := 840
def cut_amount : ℝ := 588

theorem percentage_cut : (cut_amount / original_budget) * 100 = 70 :=
by
  sorry

end percentage_cut_l48_48561


namespace counterexample_to_conjecture_l48_48689

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, 1 < m ∧ m < n → ¬(m ∣ n)

def is_power_of_two (k : ℕ) : Prop := ∃ m : ℕ, m > 0 ∧ k = 2 ^ m

theorem counterexample_to_conjecture :
  ∃ n : ℤ, n > 5 ∧ ¬ (3 ∣ n) ∧ ¬ (∃ p k : ℕ, is_prime p ∧ is_power_of_two k ∧ n = p + k) :=
sorry

end counterexample_to_conjecture_l48_48689


namespace quadratic_function_a_value_l48_48727

theorem quadratic_function_a_value (a : ℝ) (h₁ : a ≠ 1) :
  (∀ x : ℝ, ∃ c₀ c₁ c₂ : ℝ, (a-1) * x^(a^2 + 1) + 2 * x + 3 = c₂ * x^2 + c₁ * x + c₀) → a = -1 :=
by
  sorry

end quadratic_function_a_value_l48_48727


namespace trig_expression_value_l48_48624

theorem trig_expression_value (x : ℝ) (h : Real.tan (3 * Real.pi - x) = 2) : 
  (2 * Real.cos (x / 2) ^ 2 - Real.sin x - 1) / (Real.sin x + Real.cos x) = -3 :=
by 
  sorry

end trig_expression_value_l48_48624


namespace solution_set_ineq_l48_48564

theorem solution_set_ineq (x : ℝ) : (1 / x > 1) ↔ (0 < x ∧ x < 1) :=
by
  sorry

end solution_set_ineq_l48_48564


namespace jars_needed_l48_48523

-- Definitions based on the given conditions
def total_cherry_tomatoes : ℕ := 56
def cherry_tomatoes_per_jar : ℕ := 8

-- Lean theorem to prove the question
theorem jars_needed (total_cherry_tomatoes cherry_tomatoes_per_jar : ℕ) (h1 : total_cherry_tomatoes = 56) (h2 : cherry_tomatoes_per_jar = 8) : (total_cherry_tomatoes / cherry_tomatoes_per_jar) = 7 := by
  -- Proof omitted
  sorry

end jars_needed_l48_48523


namespace hikers_count_l48_48816

theorem hikers_count (B H K : ℕ) (h1 : H = B + 178) (h2 : K = B / 2) (h3 : H + B + K = 920) : H = 474 :=
by
  sorry

end hikers_count_l48_48816


namespace c_investment_l48_48180

theorem c_investment (x : ℝ) (h1 : 5000 / (5000 + 8000 + x) * 88000 = 36000) : 
  x = 20454.5 :=
by
  sorry

end c_investment_l48_48180


namespace range_of_j_l48_48227

def h (x: ℝ) : ℝ := 2 * x + 1
def j (x: ℝ) : ℝ := h (h (h (h (h x))))

theorem range_of_j :
  ∀ x, -1 ≤ x ∧ x ≤ 3 → -1 ≤ j x ∧ j x ≤ 127 :=
by 
  intros x hx
  sorry

end range_of_j_l48_48227


namespace estimate_students_less_than_2_hours_probability_one_male_one_female_l48_48743

-- Definitions from the conditions
def total_students_surveyed : ℕ := 40
def total_grade_ninth_students : ℕ := 400
def freq_0_1 : ℕ := 8
def freq_1_2 : ℕ := 20
def freq_2_3 : ℕ := 7
def freq_3_4 : ℕ := 5
def male_students_at_least_3_hours : ℕ := 2
def female_students_at_least_3_hours : ℕ := 3

-- Question 1 proof statement
theorem estimate_students_less_than_2_hours :
  total_grade_ninth_students * (freq_0_1 + freq_1_2) / total_students_surveyed = 280 :=
by sorry

-- Question 2 proof statement
theorem probability_one_male_one_female :
  (male_students_at_least_3_hours * female_students_at_least_3_hours) / (Nat.choose 5 2) = (3 / 5) :=
by sorry

end estimate_students_less_than_2_hours_probability_one_male_one_female_l48_48743


namespace number_of_students_l48_48718

theorem number_of_students (left_pos right_pos total_pos : ℕ) 
  (h₁ : left_pos = 5) 
  (h₂ : right_pos = 3) 
  (h₃ : total_pos = left_pos - 1 + 1 + (right_pos - 1)) : 
  total_pos = 7 :=
by
  rw [h₁, h₂] at h₃
  simp at h₃
  exact h₃

end number_of_students_l48_48718


namespace distinct_remainders_l48_48566

theorem distinct_remainders (n : ℕ) (hn : 0 < n) : 
  ∀ (i j : ℕ), (i < n) → (j < n) → (2 * i + 1 ≠ 2 * j + 1) → 
  ((2 * i + 1) ^ (2 * i + 1) % 2^n ≠ (2 * j + 1) ^ (2 * j + 1) % 2^n) :=
by
  sorry

end distinct_remainders_l48_48566


namespace broken_marbles_total_l48_48137

theorem broken_marbles_total :
  let broken_set_1 := 0.10 * 50
  let broken_set_2 := 0.20 * 60
  let broken_set_3 := 0.30 * 70
  let broken_set_4 := 0.15 * 80
  let total_broken := broken_set_1 + broken_set_2 + broken_set_3 + broken_set_4
  total_broken = 50 :=
by
  sorry


end broken_marbles_total_l48_48137


namespace average_of_six_numbers_l48_48918

theorem average_of_six_numbers :
  (∀ a b : ℝ, (a + b) / 2 = 6.2) →
  (∀ c d : ℝ, (c + d) / 2 = 6.1) →
  (∀ e f : ℝ, (e + f) / 2 = 6.9) →
  ((a + b + c + d + e + f) / 6 = 6.4) :=
by
  intros h1 h2 h3
  -- Proof goes here, but will be skipped with sorry.
  sorry

end average_of_six_numbers_l48_48918


namespace max_value_expression_l48_48838

noncomputable def max_expression (a b c : ℝ) : ℝ :=
  (a * b * c * (a + b + c)) / ((a + b)^2 * (b + c)^3)

theorem max_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  max_expression a b c ≤ 1 / 12 := 
sorry

end max_value_expression_l48_48838


namespace find_coordinates_of_B_find_equation_of_BC_l48_48214

-- Problem 1: Prove that the coordinates of B are (10, 5)
theorem find_coordinates_of_B (A B : ℝ × ℝ)
  (eq_med_C : ∀ (M : ℝ × ℝ), (M = ((B.1+3)/2, (B.2-1)/2) → 6 * M.1 + 10 * M.2 - 59 = 0))
  (eq_angle_bisector : B.1 - 4 * B.2 + 10 = 0) :
  B = (10, 5) :=
sorry

-- Problem 2: Prove that the equation of line BC is 2x + 9y - 65 = 0
theorem find_equation_of_BC (A B C : ℝ × ℝ)
  (eq_med_C : ∀ (M : ℝ × ℝ), (M = ((B.1+3)/2, (B.2-1)/2) → 6 * M.1 + 10 * M.2 - 59 = 0))
  (eq_angle_bisector : B.1 - 4 * B.2 + 10 = 0)
  (coordinates_B : B = (10, 5)) :
  ∃ k : ℝ, ∀ P : ℝ × ℝ, (P.1 - C.1) / (P.2 - C.2) = k → 2 * P.1 + 9 * P.2 - 65 = 0 :=
sorry

end find_coordinates_of_B_find_equation_of_BC_l48_48214


namespace man_monthly_salary_l48_48285

theorem man_monthly_salary (S E : ℝ) (h1 : 0.20 * S = S - 1.20 * E) (h2 : E = 0.80 * S) :
  S = 6000 :=
by
  sorry

end man_monthly_salary_l48_48285


namespace prob_8th_roll_last_l48_48652

-- Define the conditions as functions or constants
def prob_diff_rolls : ℚ := 5/6
def prob_same_roll : ℚ := 1/6

-- Define the theorem stating the probability of the 8th roll being the last roll
theorem prob_8th_roll_last : (1 : ℚ) * prob_diff_rolls^6 * prob_same_roll = 15625 / 279936 := 
sorry

end prob_8th_roll_last_l48_48652


namespace ratio_male_to_female_l48_48902

theorem ratio_male_to_female (total_members female_members : ℕ) (h_total : total_members = 18) (h_female : female_members = 6) :
  (total_members - female_members) / Nat.gcd (total_members - female_members) female_members = 2 ∧
  female_members / Nat.gcd (total_members - female_members) female_members = 1 :=
by
  sorry

end ratio_male_to_female_l48_48902


namespace amelia_wins_probability_l48_48396

def amelia_prob_heads : ℚ := 1 / 4
def blaine_prob_heads : ℚ := 3 / 7

def probability_blaine_wins_first_turn : ℚ := blaine_prob_heads

def probability_amelia_wins_first_turn : ℚ :=
  (1 - blaine_prob_heads) * amelia_prob_heads

def probability_amelia_wins_second_turn : ℚ :=
  (1 - blaine_prob_heads) * (1 - amelia_prob_heads) * (1 - blaine_prob_heads) * amelia_prob_heads

def probability_amelia_wins_third_turn : ℚ :=
  (1 - blaine_prob_heads) * (1 - amelia_prob_heads) * (1 - blaine_prob_heads) * 
  (1 - amelia_prob_heads) * (1 - blaine_prob_heads) * amelia_prob_heads

def probability_amelia_wins : ℚ :=
  probability_amelia_wins_first_turn + probability_amelia_wins_second_turn + probability_amelia_wins_third_turn

theorem amelia_wins_probability : probability_amelia_wins = 223 / 784 := by
  sorry

end amelia_wins_probability_l48_48396


namespace change_received_l48_48139

-- Define the given conditions
def num_apples : ℕ := 5
def cost_per_apple : ℝ := 0.75
def amount_paid : ℝ := 10.00

-- Prove the change is equal to $6.25
theorem change_received :
  amount_paid - (num_apples * cost_per_apple) = 6.25 :=
by
  sorry

end change_received_l48_48139


namespace intersection_point_lines_distance_point_to_line_l48_48473

-- Problem 1
theorem intersection_point_lines :
  ∃ (x y : ℝ), (x - y + 2 = 0) ∧ (x - 2 * y + 3 = 0) ∧ (x = -1) ∧ (y = 1) :=
sorry

-- Problem 2
theorem distance_point_to_line :
  ∀ (x y : ℝ), (x = 1) ∧ (y = -2) → ∃ d : ℝ, d = 3 ∧ (d = abs (3 * x + 4 * y - 10) / (Real.sqrt (3^2 + 4^2))) :=
sorry

end intersection_point_lines_distance_point_to_line_l48_48473


namespace remove_brackets_l48_48869

-- Define the variables a, b, and c
variables (a b c : ℝ)

-- State the theorem
theorem remove_brackets (a b c : ℝ) : a - (b - c) = a - b + c := 
sorry

end remove_brackets_l48_48869


namespace walking_speed_is_4_l48_48530

def distance : ℝ := 20
def total_time : ℝ := 3.75
def running_distance : ℝ := 10
def running_speed : ℝ := 8
def walking_distance : ℝ := 10

theorem walking_speed_is_4 (W : ℝ) 
  (H1 : running_distance + walking_distance = distance)
  (H2 : running_speed > 0)
  (H3 : walking_distance > 0)
  (H4 : W > 0)
  (H5 : walking_distance / W + running_distance / running_speed = total_time) :
  W = 4 :=
by sorry

end walking_speed_is_4_l48_48530


namespace find_m_range_l48_48516

theorem find_m_range (m : ℝ) (x : ℝ) (h : ∃ c d : ℝ, (c ≠ 0) ∧ (∀ x, (c * x + d)^2 = x^2 + (12 / 5) * x + (2 * m / 5))) : 3.5 ≤ m ∧ m ≤ 3.7 :=
by
  sorry

end find_m_range_l48_48516


namespace complete_square_transform_l48_48490

theorem complete_square_transform (x : ℝ) :
  x^2 - 2 * x - 5 = 0 → (x - 1)^2 = 6 :=
by
  intro h
  sorry

end complete_square_transform_l48_48490


namespace kite_initial_gain_percentage_l48_48127

noncomputable def initial_gain_percentage (MP CP : ℝ) : ℝ :=
  ((MP - CP) / CP) * 100

theorem kite_initial_gain_percentage :
  ∃ MP CP : ℝ,
    SP = 30 ∧
    SP = MP * 0.9 ∧
    1.035 * CP = SP ∧
    initial_gain_percentage MP CP = 15 :=
sorry

end kite_initial_gain_percentage_l48_48127


namespace Julia_watch_collection_l48_48460

section
variable (silver_watches : ℕ) (bronze_watches : ℕ) (gold_watches : ℕ) (total_watches : ℕ)

theorem Julia_watch_collection :
  silver_watches = 20 →
  bronze_watches = 3 * silver_watches →
  gold_watches = 10 * (silver_watches + bronze_watches) / 100 →
  total_watches = silver_watches + bronze_watches + gold_watches →
  total_watches = 88 :=
by
  intros
  sorry
end

end Julia_watch_collection_l48_48460


namespace not_prime_1001_base_l48_48451

theorem not_prime_1001_base (n : ℕ) (h : n ≥ 2) : ¬ Nat.Prime (n^3 + 1) :=
sorry

end not_prime_1001_base_l48_48451


namespace f_periodic_4_l48_48340

noncomputable def f : ℝ → ℝ := sorry -- f is some function ℝ → ℝ

theorem f_periodic_4 (h : ∀ x, f x = -f (x + 2)) : f 100 = f 4 := 
by
  sorry

end f_periodic_4_l48_48340


namespace coordinates_of_C_prime_l48_48900

-- Define the given vertices of the triangle
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (3, 2)

-- Define the similarity ratio
def similarity_ratio : ℝ := 2

-- Define the function for the similarity transformation
def similarity_transform (center : ℝ × ℝ) (ratio : ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := point
  (ratio * x, ratio * y)

-- Prove the coordinates of C'
theorem coordinates_of_C_prime :
  similarity_transform (0, 0) similarity_ratio C = (6, 4) ∨ 
  similarity_transform (0, 0) similarity_ratio C = (-6, -4) :=
by
  sorry

end coordinates_of_C_prime_l48_48900


namespace negation_of_universal_proposition_l48_48651

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, |x| + x^2 ≥ 0) ↔ ∃ x : ℝ, |x| + x^2 < 0 :=
by
  sorry

end negation_of_universal_proposition_l48_48651


namespace no_nonzero_ints_l48_48288

theorem no_nonzero_ints (A B : ℤ) (hA : A ≠ 0) (hB : B ≠ 0) :
  (A ∣ (A + B) ∨ B ∣ (A - B)) → false :=
sorry

end no_nonzero_ints_l48_48288


namespace min_races_required_to_determine_top_3_horses_l48_48759

def maxHorsesPerRace := 6
def totalHorses := 30
def possibleConditions := "track conditions and layouts change for each race"

noncomputable def minRacesToDetermineTop3 : Nat :=
  7

-- Problem Statement: Prove that given the conditions on track and race layout changes,
-- the minimum number of races needed to confidently determine the top 3 fastest horses is 7.
theorem min_races_required_to_determine_top_3_horses 
  (maxHorsesPerRace : Nat := 6) 
  (totalHorses : Nat := 30)
  (possibleConditions : String := "track conditions and layouts change for each race") :
  minRacesToDetermineTop3 = 7 :=
  sorry

end min_races_required_to_determine_top_3_horses_l48_48759


namespace shorter_leg_length_l48_48485

theorem shorter_leg_length (m h x : ℝ) (H1 : m = 15) (H2 : h = 3 * x) (H3 : m = 0.5 * h) : x = 10 :=
by
  sorry

end shorter_leg_length_l48_48485


namespace orange_juice_fraction_l48_48464

theorem orange_juice_fraction 
    (capacity1 capacity2 : ℕ)
    (orange_fraction1 orange_fraction2 : ℚ)
    (h_capacity1 : capacity1 = 800)
    (h_capacity2 : capacity2 = 700)
    (h_orange_fraction1 : orange_fraction1 = 1/4)
    (h_orange_fraction2 : orange_fraction2 = 1/3) :
    (capacity1 * orange_fraction1 + capacity2 * orange_fraction2) / (capacity1 + capacity2) = 433.33 / 1500 :=
by sorry

end orange_juice_fraction_l48_48464


namespace polynomial_divisible_l48_48943

theorem polynomial_divisible (n : ℕ) (hn : n > 0) :
  ∀ x : ℝ, (x-1)^3 ∣ x^(2*n+1) - (2*n+1)*x^(n+1) + (2*n+1)*x^n - 1 :=
by
  sorry

end polynomial_divisible_l48_48943


namespace train_platform_length_l48_48456

theorem train_platform_length 
  (speed_train_kmph : ℕ) 
  (time_cross_platform : ℕ) 
  (time_cross_man : ℕ) 
  (L_platform : ℕ) :
  speed_train_kmph = 72 ∧ 
  time_cross_platform = 34 ∧ 
  time_cross_man = 18 ∧ 
  L_platform = 320 :=
by
  sorry

end train_platform_length_l48_48456


namespace tom_gets_correct_share_l48_48661

def total_savings : ℝ := 18500.0
def natalie_share : ℝ := 0.35 * total_savings
def remaining_after_natalie : ℝ := total_savings - natalie_share
def rick_share : ℝ := 0.30 * remaining_after_natalie
def remaining_after_rick : ℝ := remaining_after_natalie - rick_share
def lucy_share : ℝ := 0.40 * remaining_after_rick
def remaining_after_lucy : ℝ := remaining_after_rick - lucy_share
def minimum_share : ℝ := 1000.0
def tom_share : ℝ := remaining_after_lucy

theorem tom_gets_correct_share :
  (natalie_share ≥ minimum_share) ∧ (rick_share ≥ minimum_share) ∧ (lucy_share ≥ minimum_share) →
  tom_share = 5050.50 :=
by
  sorry

end tom_gets_correct_share_l48_48661


namespace parabola_hyperbola_focus_l48_48883

theorem parabola_hyperbola_focus (p : ℝ) (h : p > 0) :
  (∀ x y : ℝ, (y ^ 2 = 2 * p * x) ∧ (x ^ 2 / 4 - y ^ 2 / 5 = 1) → p = 6) :=
by
  sorry

end parabola_hyperbola_focus_l48_48883


namespace eval_expression_l48_48147

theorem eval_expression :
  (2011 * (2012 * 10001) * (2013 * 100010001)) - (2013 * (2011 * 10001) * (2012 * 100010001)) =
  -2 * 2012 * 2013 * 10001 * 100010001 :=
by
  sorry

end eval_expression_l48_48147


namespace find_x_l48_48765

theorem find_x (x : ℝ) : 
  3.5 * ( (3.6 * 0.48 * 2.50) / (0.12 * 0.09 * x) ) = 2800.0000000000005 → x = 1.25 :=
by
  sorry

end find_x_l48_48765


namespace translation_correct_l48_48739

-- Define the points in the Cartesian coordinate system
structure Point where
  x : ℤ
  y : ℤ

-- Given points A and B
def A : Point := { x := -1, y := 0 }
def B : Point := { x := 1, y := 2 }

-- Translated point A' (A₁)
def A₁ : Point := { x := 2, y := -1 }

-- Define the translation applied to a point
def translate (p : Point) (v : Point) : Point :=
  { x := p.x + v.x, y := p.y + v.y }

-- Calculate the translation vector from A to A'
def translationVector : Point :=
  { x := A₁.x - A.x, y := A₁.y - A.y }

-- Define the expected point B' (B₁)
def B₁ : Point := { x := 4, y := 1 }

-- Theorem statement
theorem translation_correct :
  translate B translationVector = B₁ :=
by
  -- proof goes here
  sorry

end translation_correct_l48_48739


namespace inequality_always_negative_l48_48009

theorem inequality_always_negative (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + k * x - 3 / 4 < 0) ↔ (-3 < k ∧ k ≤ 0) :=
by
  -- Proof omitted
  sorry

end inequality_always_negative_l48_48009


namespace luke_number_of_rounds_l48_48605

variable (points_per_round total_points : ℕ)

theorem luke_number_of_rounds 
  (h1 : points_per_round = 3)
  (h2 : total_points = 78) : 
  total_points / points_per_round = 26 := 
by 
  sorry

end luke_number_of_rounds_l48_48605


namespace pedestrian_wait_probability_l48_48057

-- Define the duration of the red light
def red_light_duration := 45

-- Define the favorable time window for the pedestrian to wait at least 20 seconds
def favorable_window := 25

-- The probability that the pedestrian has to wait at least 20 seconds
def probability_wait_at_least_20 : ℚ := favorable_window / red_light_duration

theorem pedestrian_wait_probability : probability_wait_at_least_20 = 5 / 9 := by
  sorry

end pedestrian_wait_probability_l48_48057


namespace minimum_value_of_f_l48_48702

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 2)

theorem minimum_value_of_f :
  ∃ a > 2, (∀ x > 2, f x ≥ f a) ∧ a = 3 := by
sorry

end minimum_value_of_f_l48_48702


namespace greatest_value_of_x_l48_48381

theorem greatest_value_of_x
  (x : ℕ)
  (h1 : x % 4 = 0) -- x is a multiple of 4
  (h2 : x > 0) -- x is positive
  (h3 : x^3 < 2000) -- x^3 < 2000
  : x ≤ 12 :=
by
  sorry

end greatest_value_of_x_l48_48381


namespace diagonal_of_rectangle_l48_48411

theorem diagonal_of_rectangle (a b d : ℝ)
  (h_side : a = 15)
  (h_area : a * b = 120)
  (h_diag : a^2 + b^2 = d^2) :
  d = 17 :=
by
  sorry

end diagonal_of_rectangle_l48_48411


namespace string_length_l48_48183

theorem string_length (cylinder_circumference : ℝ)
  (total_loops : ℕ) (post_height : ℝ)
  (height_per_loop : ℝ := post_height / total_loops)
  (hypotenuse_per_loop : ℝ := Real.sqrt (height_per_loop ^ 2 + cylinder_circumference ^ 2))
  : total_loops = 5 → cylinder_circumference = 4 → post_height = 15 → hypotenuse_per_loop * total_loops = 25 :=
by 
  intros h1 h2 h3
  sorry

end string_length_l48_48183


namespace probability_of_distinct_divisors_l48_48171

theorem probability_of_distinct_divisors :
  ∃ (m n : ℕ), (m.gcd n = 1) ∧ (m / n) = 125 / 158081 :=
by
  sorry

end probability_of_distinct_divisors_l48_48171


namespace possible_values_of_sum_l48_48442

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + y^3 + 21 * x * y = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end possible_values_of_sum_l48_48442


namespace cos_sin_inequality_inequality_l48_48215

noncomputable def proof_cos_sin_inequality (a b : ℝ) (cos_x sin_x: ℝ) : Prop :=
  (cos_x ^ 2 = a) → (sin_x ^ 2 = b) → (a + b = 1) → (1 / 4 ≤ a ^ 3 + b ^ 3 ∧ a ^ 3 + b ^ 3 ≤ 1)

theorem cos_sin_inequality_inequality (a b : ℝ) (cos_x sin_x : ℝ) :
  proof_cos_sin_inequality a b cos_x sin_x :=
  by { sorry }

end cos_sin_inequality_inequality_l48_48215


namespace percentage_of_z_equals_39_percent_of_y_l48_48091

theorem percentage_of_z_equals_39_percent_of_y
    (x y z : ℝ)
    (h1 : y = 0.75 * x)
    (h2 : z = 0.65 * x)
    (P : ℝ)
    (h3 : (P / 100) * z = 0.39 * y) :
    P = 45 :=
by sorry

end percentage_of_z_equals_39_percent_of_y_l48_48091


namespace amoeba_count_after_ten_days_l48_48471

theorem amoeba_count_after_ten_days : 
  let initial_amoebas := 1
  let splits_per_day := 3
  let days := 10
  (initial_amoebas * splits_per_day ^ days) = 59049 := 
by 
  let initial_amoebas := 1
  let splits_per_day := 3
  let days := 10
  show (initial_amoebas * splits_per_day ^ days) = 59049
  sorry

end amoeba_count_after_ten_days_l48_48471


namespace length_of_bridge_l48_48855

theorem length_of_bridge
    (speed_kmh : Real)
    (time_minutes : Real)
    (speed_cond : speed_kmh = 5)
    (time_cond : time_minutes = 15) :
    let speed_mmin := speed_kmh * 1000 / 60
    let distance_m := speed_mmin * time_minutes
    distance_m = 1250 :=
by
    sorry

end length_of_bridge_l48_48855


namespace password_probability_l48_48622

theorem password_probability :
  let even_digits := [0, 2, 4, 6, 8]
  let vowels := ['A', 'E', 'I', 'O', 'U']
  let non_zero_digits := [1, 2, 3, 4, 5, 6, 7, 8, 9]
  (even_digits.length / 10) * (vowels.length / 26) * (non_zero_digits.length / 10) = 9 / 52 :=
by
  sorry

end password_probability_l48_48622


namespace max_rectangle_area_under_budget_l48_48014

/-- 
Let L and W be the length and width of a rectangle, respectively, where:
1. The length L is made of materials priced at 3 yuan per meter.
2. The width W is made of materials priced at 5 yuan per meter.
3. Both L and W are integers.
4. The total cost 3L + 5W does not exceed 100 yuan.

Prove that the maximum area of the rectangle that can be made under these constraints is 40 square meters.
--/
theorem max_rectangle_area_under_budget :
  ∃ (L W : ℤ), 3 * L + 5 * W ≤ 100 ∧ 0 ≤ L ∧ 0 ≤ W ∧ L * W = 40 :=
sorry

end max_rectangle_area_under_budget_l48_48014


namespace jodi_walks_days_l48_48719

section
variables {d : ℕ} -- d is the number of days Jodi walks per week

theorem jodi_walks_days (h : 1 * d + 2 * d + 3 * d + 4 * d = 60) : d = 6 := by
  sorry

end

end jodi_walks_days_l48_48719


namespace abs_eq_4_reciprocal_eq_self_l48_48035

namespace RationalProofs

-- Problem 1
theorem abs_eq_4 (x : ℚ) : |x| = 4 ↔ x = 4 ∨ x = -4 :=
by sorry

-- Problem 2
theorem reciprocal_eq_self (x : ℚ) : x ≠ 0 → x⁻¹ = x ↔ x = 1 ∨ x = -1 :=
by sorry

end RationalProofs

end abs_eq_4_reciprocal_eq_self_l48_48035


namespace emery_family_trip_l48_48100

theorem emery_family_trip 
  (first_part_distance : ℕ) (first_part_time : ℕ) (total_time : ℕ) (speed : ℕ) (second_part_time : ℕ) :
  first_part_distance = 100 ∧ first_part_time = 1 ∧ total_time = 4 ∧ speed = 100 ∧ second_part_time = 3 →
  second_part_time * speed = 300 :=
by 
  sorry

end emery_family_trip_l48_48100


namespace repeating_seventy_two_exceeds_seventy_two_l48_48277

noncomputable def repeating_decimal (n d : ℕ) : ℚ := n / d

theorem repeating_seventy_two_exceeds_seventy_two :
  repeating_decimal 72 99 - (72 / 100) = (2 / 275) := 
sorry

end repeating_seventy_two_exceeds_seventy_two_l48_48277


namespace bicycle_price_l48_48256

theorem bicycle_price (P : ℝ) (h : 0.2 * P = 200) : P = 1000 := 
by
  sorry

end bicycle_price_l48_48256


namespace sum_non_solutions_l48_48660

theorem sum_non_solutions (A B C : ℝ) (h : ∀ x, (x + B) * (A * x + 36) = 3 * (x + C) * (x + 9) → x ≠ -12) :
  -12 = -12 := 
sorry

end sum_non_solutions_l48_48660


namespace parabola_line_intersection_l48_48822

/-- 
Given the parabola y^2 = -x and the line l: y = k(x + 1) intersect at points A and B,
(Ⅰ) Find the range of values for k;
(Ⅱ) Let O be the vertex of the parabola, prove that OA ⟂ OB.
-/
theorem parabola_line_intersection (k : ℝ) (A B : ℝ × ℝ)
  (hA : A.2 ^ 2 = -A.1) (hB : B.2 ^ 2 = -B.1)
  (hlineA : A.2 = k * (A.1 + 1)) (hlineB : B.2 = k * (B.1 + 1)) :
  (k ≠ 0) ∧ ((A.2 * B.2 = -1) → A.1 * B.1 * (A.2 * B.2) = -1) :=
by
  sorry

end parabola_line_intersection_l48_48822


namespace unique_solution_l48_48806

noncomputable def f (a b x : ℝ) := 2 * (a + b) * Real.exp (2 * x) + 2 * a * b
noncomputable def g (a b x : ℝ) := 4 * Real.exp (2 * x) + a + b

theorem unique_solution (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  ∃! x, f a b x = ( (a^(1/3) + b^(1/3))/2 )^3 * g a b x :=
sorry

end unique_solution_l48_48806


namespace fewer_onions_than_tomatoes_and_corn_l48_48438

def tomatoes : ℕ := 2073
def corn : ℕ := 4112
def onions : ℕ := 985

theorem fewer_onions_than_tomatoes_and_corn :
  (tomatoes + corn - onions) = 5200 :=
by
  sorry

end fewer_onions_than_tomatoes_and_corn_l48_48438


namespace loss_percentage_is_30_l48_48555

theorem loss_percentage_is_30
  (cost_price : ℝ)
  (selling_price : ℝ)
  (h1 : cost_price = 1900)
  (h2 : selling_price = 1330) :
  (cost_price - selling_price) / cost_price * 100 = 30 :=
by
  -- This is a placeholder for the actual proof
  sorry

end loss_percentage_is_30_l48_48555


namespace largest_percentage_drop_l48_48611

theorem largest_percentage_drop (jan feb mar apr may jun : ℤ) 
  (h_jan : jan = -10)
  (h_feb : feb = 5)
  (h_mar : mar = -15)
  (h_apr : apr = 10)
  (h_may : may = -30)
  (h_jun : jun = 0) :
  may = -30 ∧ ∀ month, month ≠ may → month ≥ -30 :=
by
  sorry

end largest_percentage_drop_l48_48611


namespace hearty_total_beads_l48_48986

-- Definition of the problem conditions
def blue_beads_per_package (r : ℕ) : ℕ := 2 * r
def red_beads_per_package : ℕ := 40
def red_packages : ℕ := 5
def blue_packages : ℕ := 3

-- Define the total number of beads Hearty has
def total_beads (r : ℕ) (rp : ℕ) (bp : ℕ) : ℕ :=
  (rp * red_beads_per_package) + (bp * blue_beads_per_package red_beads_per_package)

-- The theorem to be proven
theorem hearty_total_beads : total_beads red_beads_per_package red_packages blue_packages = 440 := by
  sorry

end hearty_total_beads_l48_48986


namespace sum_of_coords_D_eq_eight_l48_48744

def point := (ℝ × ℝ)

def N : point := (4, 6)
def C : point := (10, 2)

def is_midpoint (M A B : point) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

theorem sum_of_coords_D_eq_eight
  (D : point)
  (h_midpoint : is_midpoint N C D) :
  D.1 + D.2 = 8 :=
by 
  sorry

end sum_of_coords_D_eq_eight_l48_48744


namespace fraction_value_l48_48991

theorem fraction_value (a b : ℚ) (h : a / b = 2 / 5) : a / (a + b) = 2 / 7 :=
by
  -- The proof goes here.
  sorry

end fraction_value_l48_48991


namespace red_peppers_weight_correct_l48_48026

def weight_of_red_peppers : Prop :=
  ∀ (T G : ℝ), (T = 0.66) ∧ (G = 0.33) → (T - G = 0.33)

theorem red_peppers_weight_correct : weight_of_red_peppers :=
  sorry

end red_peppers_weight_correct_l48_48026


namespace cost_combination_exists_l48_48431

/-!
Given:
- Nadine spent a total of $105.
- The table costs $34.
- The mirror costs $15.
- The lamp costs $6.
- The total cost of the 2 chairs and 3 decorative vases is $50.

Prove:
- There are multiple combinations of individual chair cost (C) and individual vase cost (V) such that 2 * C + 3 * V = 50.
-/

theorem cost_combination_exists :
  ∃ (C V : ℝ), 2 * C + 3 * V = 50 :=
by {
  sorry
}

end cost_combination_exists_l48_48431


namespace parallelogram_properties_l48_48334

noncomputable def length_adjacent_side_and_area (base height : ℝ) (angle : ℕ) : ℝ × ℝ :=
  let hypotenuse := height / Real.sin (angle * Real.pi / 180)
  let area := base * height
  (hypotenuse, area)

theorem parallelogram_properties :
  ∀ (base height : ℝ) (angle : ℕ),
  base = 12 → height = 6 → angle = 30 →
  length_adjacent_side_and_area base height angle = (12, 72) :=
by
  intros
  sorry

end parallelogram_properties_l48_48334


namespace prime_factors_identity_l48_48815

theorem prime_factors_identity (w x y z k : ℕ) 
    (h : 2^w * 3^x * 5^y * 7^z * 11^k = 900) : 
      2 * w + 3 * x + 5 * y + 7 * z + 11 * k = 20 :=
by
  sorry

end prime_factors_identity_l48_48815


namespace num_teams_is_seventeen_l48_48748

-- Each team faces all other teams 10 times and there are 1360 games in total.
def total_teams (n : ℕ) : Prop := 1360 = (n * (n - 1) * 10) / 2

theorem num_teams_is_seventeen : ∃ n : ℕ, total_teams n ∧ n = 17 := 
by 
  sorry

end num_teams_is_seventeen_l48_48748


namespace expand_fraction_product_l48_48268

-- Define the variable x and the condition that x ≠ 0
variable (x : ℝ) (h : x ≠ 0)

-- State the theorem
theorem expand_fraction_product (h : x ≠ 0) :
  3 / 7 * (7 / x^2 + 7 * x - 7 / x) = 3 / x^2 + 3 * x - 3 / x :=
sorry

end expand_fraction_product_l48_48268


namespace sin_half_angle_correct_l48_48598

noncomputable def sin_half_angle (theta : ℝ) (h1 : Real.sin theta = 3 / 5) (h2 : 5 * Real.pi / 2 < theta ∧ theta < 3 * Real.pi) : ℝ :=
  -3 * Real.sqrt 10 / 10

theorem sin_half_angle_correct (theta : ℝ) (h1 : Real.sin theta = 3 / 5) (h2 : 5 * Real.pi / 2 < theta ∧ theta < 3 * Real.pi) :
  sin_half_angle theta h1 h2 = Real.sin (theta / 2) :=
by
  sorry

end sin_half_angle_correct_l48_48598


namespace abs_expression_eq_five_l48_48072

theorem abs_expression_eq_five : 
  abs (abs (-abs (-2 + 3) - 2) + 2) = 5 :=
by
  sorry -- proof omitted

end abs_expression_eq_five_l48_48072


namespace inequality_am_gm_l48_48329

theorem inequality_am_gm (a b : ℝ) (h1 : a < 1) (h2 : b < 1) (h3 : a + b ≥ 1/2) :
  (1 - a) * (1 - b) ≤ 9 / 16 := 
by
  sorry

end inequality_am_gm_l48_48329


namespace gcd_g10_g13_l48_48357

-- Define the polynomial function g
def g (x : ℤ) : ℤ := x^3 - 3 * x^2 + x + 2050

-- State the theorem to prove that gcd(g(10), g(13)) is 1
theorem gcd_g10_g13 : Int.gcd (g 10) (g 13) = 1 := by
  sorry

end gcd_g10_g13_l48_48357


namespace min_value_expression_l48_48625

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) : 
  ∃ c, ∀ x y, 0 < x → 0 < y → x + y = 1 → c = 9 ∧ ((1 / x) + (4 / y)) ≥ 9 := 
sorry

end min_value_expression_l48_48625


namespace contractor_engaged_days_l48_48877

/-
  Prove that the contractor was engaged for 30 days given that:
  1. The contractor receives Rs. 25 for each day he works.
  2. The contractor is fined Rs. 7.50 for each day he is absent.
  3. The contractor gets Rs. 425 in total.
  4. The contractor was absent for 10 days.
-/

theorem contractor_engaged_days
  (wage_per_day : ℕ) (fine_per_absent_day : ℕ) (total_amount_received : ℕ) (absent_days : ℕ) 
  (total_days_engaged : ℕ) :
  wage_per_day = 25 →
  fine_per_absent_day = 7500 / 1000 → -- Rs. 7.50 = 7500 / 1000 (to avoid float, using integers)
  total_amount_received = 425 →
  absent_days = 10 →
  total_days_engaged = (total_amount_received + absent_days * fine_per_absent_day) / wage_per_day + absent_days →
  total_days_engaged = 30 :=
by
  sorry

end contractor_engaged_days_l48_48877


namespace f_13_eq_223_l48_48359

def f (n : ℕ) : ℕ := n^2 + n + 41

theorem f_13_eq_223 : f 13 = 223 :=
by
  sorry

end f_13_eq_223_l48_48359


namespace value_range_sin_neg_l48_48190

theorem value_range_sin_neg (x : ℝ) (h : x ∈ Set.Icc (Real.pi / 4) (5 * Real.pi / 4)) : 
  Set.Icc (-1) (Real.sqrt 2 / 2) ( - (Real.sin x) ) :=
sorry

end value_range_sin_neg_l48_48190


namespace usable_area_l48_48186

def garden_length : ℕ := 20
def garden_width : ℕ := 18
def pond_side : ℕ := 4

theorem usable_area :
  garden_length * garden_width - pond_side * pond_side = 344 :=
by
  sorry

end usable_area_l48_48186


namespace find_x_l48_48229

theorem find_x (x y : ℤ) (h1 : x + y = 4) (h2 : x - y = 36) : x = 20 :=
by
  sorry

end find_x_l48_48229


namespace distance_after_one_hour_l48_48860

-- Definitions representing the problem's conditions
def initial_distance : ℕ := 20
def speed_athos : ℕ := 4
def speed_aramis : ℕ := 5

-- The goal is to prove that the possible distances after one hour are among the specified values
theorem distance_after_one_hour :
  ∃ d : ℕ, d = 11 ∨ d = 29 ∨ d = 21 ∨ d = 19 :=
sorry -- proof not required as per the instructions

end distance_after_one_hour_l48_48860


namespace sign_of_ac_l48_48399

theorem sign_of_ac (a b c d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) (h : (a / b) + (c / d) = (a + c) / (b + d)) : a * c < 0 :=
by
  sorry

end sign_of_ac_l48_48399


namespace increase_in_average_weight_l48_48405

variable {A X : ℝ}

-- Given initial conditions
axiom average_initial_weight_8 : X = (8 * A - 62 + 90) / 8 - A

-- The goal to prove
theorem increase_in_average_weight : X = 3.5 :=
by
  sorry

end increase_in_average_weight_l48_48405


namespace feathers_already_have_l48_48895

-- Given conditions
def total_feathers : Nat := 900
def feathers_still_needed : Nat := 513

-- Prove that the number of feathers Charlie already has is 387
theorem feathers_already_have : (total_feathers - feathers_still_needed) = 387 := by
  sorry

end feathers_already_have_l48_48895


namespace min_value_x_plus_9_div_x_l48_48303

theorem min_value_x_plus_9_div_x (x : ℝ) (hx : x > 0) : x + 9 / x ≥ 6 := by
  -- sorry indicates that the proof is omitted.
  sorry

end min_value_x_plus_9_div_x_l48_48303


namespace initial_amount_l48_48891

theorem initial_amount (P R : ℝ) (h1 : 956 = P * (1 + (3 * R) / 100)) (h2 : 1052 = P * (1 + (3 * (R + 4)) / 100)) : P = 800 := 
by
  -- We would provide the proof steps here normally
  sorry

end initial_amount_l48_48891


namespace expenditure_ratio_l48_48012

variable {I : ℝ} -- Income in the first year

-- Conditions
def first_year_savings (I : ℝ) : ℝ := 0.5 * I
def first_year_expenditure (I : ℝ) : ℝ := I - first_year_savings I
def second_year_income (I : ℝ) : ℝ := 1.5 * I
def second_year_savings (I : ℝ) : ℝ := 2 * first_year_savings I
def second_year_expenditure (I : ℝ) : ℝ := second_year_income I - second_year_savings I

-- Condition statement in Lean
theorem expenditure_ratio (I : ℝ) : 
  let total_expenditure := first_year_expenditure I + second_year_expenditure I
  (total_expenditure / first_year_expenditure I) = 2 :=
  by 
    sorry

end expenditure_ratio_l48_48012


namespace problem1_problem2_l48_48031

-- Problem 1
theorem problem1 : 5*Real.sqrt 2 - (Real.sqrt 18 + Real.sqrt (1/2)) = (3/2) * Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem2 : (2*Real.sqrt 3 - 1)^2 + (Real.sqrt 24) / (Real.sqrt 2) = 13 - 2*Real.sqrt 3 := by
  sorry

end problem1_problem2_l48_48031


namespace simone_fraction_per_day_l48_48043

theorem simone_fraction_per_day 
  (x : ℚ) -- Define the fraction of an apple Simone ate each day as x.
  (h1 : 16 * x + 15 * (1/3) = 13) -- Condition: Simone and Lauri together ate 13 apples.
  : x = 1/2 := 
 by 
  sorry

end simone_fraction_per_day_l48_48043


namespace sum_of_digits_is_3_l48_48161

-- We introduce variables for the digits a and b, and the number
variables (a b : ℕ)

-- Conditions: a and b must be digits, and the number must satisfy the given equation
-- One half of (10a + b) exceeds its one fourth by 3
def valid_digits (a b : ℕ) : Prop := a < 10 ∧ b < 10
def equation_condition (a b : ℕ) : Prop := 2 * (10 * a + b) = (10 * a + b) + 12

-- The number is two digits number
def two_digits_number (a b : ℕ) : ℕ := 10 * a + b

-- Final statement combining all conditions and proving the desired sum of digits
theorem sum_of_digits_is_3 : 
  ∀ (a b : ℕ), valid_digits a b → equation_condition a b → a + b = 3 := 
by
  intros a b h1 h2
  sorry

end sum_of_digits_is_3_l48_48161


namespace brownies_cut_into_pieces_l48_48447

theorem brownies_cut_into_pieces (total_amount_made : ℕ) (pans : ℕ) (cost_per_brownie : ℕ) (brownies_sold : ℕ) 
  (h1 : total_amount_made = 32) (h2 : pans = 2) (h3 : cost_per_brownie = 2) (h4 : brownies_sold = total_amount_made / cost_per_brownie) :
  16 = brownies_sold :=
by
  sorry

end brownies_cut_into_pieces_l48_48447


namespace locus_of_centers_of_tangent_circles_l48_48980

noncomputable def C1 (x y : ℝ) : Prop := x^2 + y^2 = 1
noncomputable def C2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 25
noncomputable def locus (a b : ℝ) : Prop := 4 * a^2 + 4 * b^2 - 6 * a - 25 = 0

theorem locus_of_centers_of_tangent_circles :
  (∃ (a b r : ℝ), a^2 + b^2 = (r + 1)^2 ∧ (a - 3)^2 + b^2 = (5 - r)^2) →
  (∃ a b : ℝ, locus a b) :=
sorry

end locus_of_centers_of_tangent_circles_l48_48980


namespace jason_seashells_l48_48332

theorem jason_seashells (initial_seashells : ℕ) (given_seashells : ℕ) (remaining_seashells : ℕ) 
(h1 : initial_seashells = 49) (h2 : given_seashells = 13) :
remaining_seashells = initial_seashells - given_seashells := by
  sorry

end jason_seashells_l48_48332


namespace equivalent_proof_problem_l48_48688

def op (a b : ℝ) : ℝ := (a + b) ^ 2

theorem equivalent_proof_problem (x y : ℝ) : 
  op ((x + y) ^ 2) ((x - y) ^ 2) = 4 * (x ^ 2 + y ^ 2) ^ 2 := 
by 
  sorry

end equivalent_proof_problem_l48_48688


namespace prove_f_three_eq_neg_three_l48_48681

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * Real.sin (2 * x) + b * Real.tan x + 1

theorem prove_f_three_eq_neg_three (a b : ℝ) (h : f (-3) a b = 5) : f 3 a b = -3 := by
  sorry

end prove_f_three_eq_neg_three_l48_48681


namespace num_shoes_sold_sab_dane_sold_6_pairs_of_shoes_l48_48001

theorem num_shoes_sold (price_shoes : ℕ) (num_shirts : ℕ) (price_shirts : ℕ) (total_earn_per_person : ℕ) : ℕ :=
  let total_earnings_shirts := num_shirts * price_shirts
  let total_earnings := total_earn_per_person * 2
  let earnings_from_shoes := total_earnings - total_earnings_shirts
  let num_shoes_sold := earnings_from_shoes / price_shoes
  num_shoes_sold

theorem sab_dane_sold_6_pairs_of_shoes :
  num_shoes_sold 3 18 2 27 = 6 :=
by
  sorry

end num_shoes_sold_sab_dane_sold_6_pairs_of_shoes_l48_48001


namespace no_nat_solution_for_exp_eq_l48_48146

theorem no_nat_solution_for_exp_eq (n x y z : ℕ) (hn : n > 1) (hx : x ≤ n) (hy : y ≤ n) :
  ¬ (x^n + y^n = z^n) :=
by
  sorry

end no_nat_solution_for_exp_eq_l48_48146


namespace smallest_positive_period_max_min_values_l48_48963

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, -1 / 2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.cos (2 * x))
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem smallest_positive_period (x : ℝ) :
  ∃ T, T > 0 ∧ ∀ x, f (x + T) = f x ∧ ∀ T', T' > 0 ∧ ∀ x, f (x + T') = f x → T ≤ T' :=
  sorry

theorem max_min_values : ∃ max min : ℝ, (max = 1) ∧ (min = -1 / 2) ∧
  ∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 →
  min ≤ f x ∧ f x ≤ max :=
  sorry

end smallest_positive_period_max_min_values_l48_48963


namespace min_value_proven_l48_48699

open Real

noncomputable def min_value (x y : ℝ) (h1 : log x + log y = 1) : Prop :=
  2 * x + 5 * y ≥ 20 ∧ (2 * x + 5 * y = 20 ↔ 2 * x = 5 * y ∧ x * y = 10)

theorem min_value_proven (x y : ℝ) (h1 : log x + log y = 1) :
  min_value x y h1 :=
sorry

end min_value_proven_l48_48699


namespace common_fraction_proof_l48_48061

def expr_as_common_fraction : Prop :=
  let numerator := (3 / 6) + (4 / 5)
  let denominator := (5 / 12) + (1 / 4)
  (numerator / denominator) = (39 / 20)

theorem common_fraction_proof : expr_as_common_fraction :=
by
  sorry

end common_fraction_proof_l48_48061


namespace probability_of_team_with_2_girls_2_boys_l48_48020

open Nat

-- Define the combinatorics function for binomial coefficients
def binomial (n k : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem probability_of_team_with_2_girls_2_boys :
  let total_women := 8
  let total_men := 6
  let team_size := 4
  let ways_to_choose_2_girls := binomial total_women 2
  let ways_to_choose_2_boys := binomial total_men 2
  let total_ways_to_form_team := binomial (total_women + total_men) team_size
  let favorable_outcomes := ways_to_choose_2_girls * ways_to_choose_2_boys
  (favorable_outcomes : ℚ) / total_ways_to_form_team = 60 / 143 := 
by sorry

end probability_of_team_with_2_girls_2_boys_l48_48020


namespace smallest_value_l48_48693

theorem smallest_value (x : ℝ) (h : 3 * x^2 + 33 * x - 90 = x * (x + 18)) : x ≥ -10.5 :=
sorry

end smallest_value_l48_48693


namespace part1_part2_l48_48335

-- Problem Part 1
theorem part1 : (-((-8)^(1/3)) - |(3^(1/2) - 2)| + ((-3)^2)^(1/2) + -3^(1/2) = 3) :=
by {
  sorry
}

-- Problem Part 2
theorem part2 (x : ℤ) : (2 * x + 5 ≤ 3 * (x + 2) ∧ 2 * x - (1 + 3 * x) / 2 < 1) ↔ (x = 0 ∨ x = 1 ∨ x = 2) :=
by {
  sorry
}

end part1_part2_l48_48335


namespace distribution_centers_count_l48_48857

theorem distribution_centers_count (n : ℕ) (h : n = 5) : n + (n * (n - 1)) / 2 = 15 :=
by
  subst h -- replace n with 5
  show 5 + (5 * (5 - 1)) / 2 = 15
  have : (5 * 4) / 2 = 10 := by norm_num
  show 5 + 10 = 15
  norm_num

end distribution_centers_count_l48_48857


namespace projectile_height_reach_l48_48650

theorem projectile_height_reach (t : ℝ) (h : -16 * t^2 + 64 * t = 25) : t = 3.6 :=
by
  sorry

end projectile_height_reach_l48_48650


namespace sqrt_nine_over_four_l48_48570

theorem sqrt_nine_over_four (x : ℝ) : x = 3 / 2 ∨ x = - (3 / 2) ↔ x * x = 9 / 4 :=
by {
  sorry
}

end sqrt_nine_over_four_l48_48570


namespace remainder_b_91_mod_49_l48_48758

def b (n : ℕ) := 12^n + 14^n

theorem remainder_b_91_mod_49 : (b 91) % 49 = 38 := by
  sorry

end remainder_b_91_mod_49_l48_48758


namespace kanul_spent_on_raw_materials_eq_500_l48_48118

variable (total_amount : ℕ)
variable (machinery_cost : ℕ)
variable (cash_percentage : ℕ)

def amount_spent_on_raw_materials (total_amount machinery_cost cash_percentage : ℕ) : ℕ :=
  total_amount - machinery_cost - (total_amount * cash_percentage / 100)

theorem kanul_spent_on_raw_materials_eq_500 :
  total_amount = 1000 →
  machinery_cost = 400 →
  cash_percentage = 10 →
  amount_spent_on_raw_materials total_amount machinery_cost cash_percentage = 500 :=
by
  intros
  sorry

end kanul_spent_on_raw_materials_eq_500_l48_48118


namespace number_div_by_3_l48_48446

theorem number_div_by_3 (x : ℕ) (h : 54 = x - 39) : x / 3 = 31 :=
by
  sorry

end number_div_by_3_l48_48446


namespace domain_of_sqrt_function_l48_48192

theorem domain_of_sqrt_function :
  {x : ℝ | 0 ≤ x + 1} = {x : ℝ | -1 ≤ x} :=
by {
  sorry
}

end domain_of_sqrt_function_l48_48192


namespace total_area_of_map_l48_48145

def level1_area : ℕ := 40 * 20
def level2_area : ℕ := 15 * 15
def level3_area : ℕ := (25 * 12) / 2

def total_area : ℕ := level1_area + level2_area + level3_area

theorem total_area_of_map : total_area = 1175 := by
  -- Proof to be completed
  sorry

end total_area_of_map_l48_48145


namespace coeff_x3_in_expansion_l48_48550

theorem coeff_x3_in_expansion : (Polynomial.coeff ((Polynomial.C 1 - Polynomial.C 2 * Polynomial.X)^6) 3) = -160 := 
by 
  sorry

end coeff_x3_in_expansion_l48_48550


namespace angle_bisector_slope_l48_48346

/-
Given conditions:
1. line1: y = 2x
2. line2: y = 4x
Prove:
k = (sqrt(21) - 6) / 7
-/

theorem angle_bisector_slope :
  let m1 := 2
  let m2 := 4
  let k := (Real.sqrt 21 - 6) / 7
  (1 - m1 * m2) ≠ 0 →
  k = (m1 + m2 - Real.sqrt (1 + m1^2 + m2^2)) / (1 - m1 * m2)
:=
sorry

end angle_bisector_slope_l48_48346


namespace Norine_retire_age_l48_48872

theorem Norine_retire_age:
  ∀ (A W : ℕ),
    (A = 50) →
    (W = 19) →
    (A + W = 85) →
    (A = 50 + 8) :=
by
  intros A W hA hW hAW
  sorry

end Norine_retire_age_l48_48872


namespace lines_skew_iff_a_ne_20_l48_48572

variable {t u a : ℝ}
-- Definitions for the lines
def line1 (t : ℝ) (a : ℝ) := (2 + 3 * t, 3 + 4 * t, a + 5 * t)
def line2 (u : ℝ) := (3 + 6 * u, 2 + 5 * u, 1 + 2 * u)

-- Condition for lines to intersect
def lines_intersect (t u a : ℝ) :=
  2 + 3 * t = 3 + 6 * u ∧
  3 + 4 * t = 2 + 5 * u ∧
  a + 5 * t = 1 + 2 * u

-- The main theorem stating when lines are skew
theorem lines_skew_iff_a_ne_20 (a : ℝ) :
  (¬ ∃ t u : ℝ, lines_intersect t u a) ↔ a ≠ 20 := 
by 
  sorry

end lines_skew_iff_a_ne_20_l48_48572


namespace current_algae_plants_l48_48859

def original_algae_plants : ℕ := 809
def additional_algae_plants : ℕ := 2454

theorem current_algae_plants :
  original_algae_plants + additional_algae_plants = 3263 := by
  sorry

end current_algae_plants_l48_48859


namespace berries_ratio_l48_48235

theorem berries_ratio (total_berries : ℕ) (stacy_berries : ℕ) (ratio_stacy_steve : ℕ)
  (h_total : total_berries = 1100) (h_stacy : stacy_berries = 800)
  (h_ratio : stacy_berries = 4 * ratio_stacy_steve) :
  ratio_stacy_steve / (total_berries - stacy_berries - ratio_stacy_steve) = 2 :=
by {
  sorry
}

end berries_ratio_l48_48235


namespace problem1_problem2_l48_48655

theorem problem1 : 12 - (-18) + (-7) + (-15) = 8 :=
by sorry

theorem problem2 : (-1)^7 * 2 + (-3)^2 / 9 = -1 :=
by sorry

end problem1_problem2_l48_48655


namespace lock_combination_l48_48793

def valid_combination (T I D E b : ℕ) : Prop :=
  (T > 0) ∧ (I > 0) ∧ (D > 0) ∧ (E > 0) ∧
  (T ≠ I) ∧ (T ≠ D) ∧ (T ≠ E) ∧ (I ≠ D) ∧ (I ≠ E) ∧ (D ≠ E) ∧
  (T * b^3 + I * b^2 + D * b + E) + 
  (E * b^3 + D * b^2 + I * b + T) + 
  (T * b^3 + I * b^2 + D * b + E) = 
  (D * b^3 + I * b^2 + E * b + T)

theorem lock_combination : ∃ (T I D E b : ℕ), valid_combination T I D E b ∧ (T * 100 + I * 10 + D = 984) :=
sorry

end lock_combination_l48_48793


namespace entrance_exam_correct_answers_l48_48196

theorem entrance_exam_correct_answers (c w : ℕ) 
  (h1 : c + w = 70) 
  (h2 : 3 * c - w = 38) : 
  c = 27 := 
sorry

end entrance_exam_correct_answers_l48_48196


namespace tim_drinks_amount_l48_48097

theorem tim_drinks_amount (H : ℚ := 2/7) (T : ℚ := 5/8) : 
  (T * H) = 5/28 :=
by sorry

end tim_drinks_amount_l48_48097


namespace domain_of_f_intervals_of_monotonicity_extremal_values_l48_48677

noncomputable def f (x : ℝ) := (1 / 2) * x ^ 2 - 5 * x + 4 * Real.log x 

theorem domain_of_f : ∀ x, 0 < x → f x = (1 / 2) * x ^ 2 - 5 * x + 4 * Real.log x :=
by
  intro x hx
  exact rfl

theorem intervals_of_monotonicity :
  (∀ x, 0 < x ∧ x < 1 → f x < f 1) ∧
  (∀ x, 1 < x ∧ x < 4 → f x > f 1 ∧ f x < f 4) ∧
  (∀ x, 4 < x → f x > f 4) :=
sorry

theorem extremal_values :
  (f 1 = - (9 / 2)) ∧ 
  (f 4 = -12 + 4 * Real.log 4) :=
sorry

end domain_of_f_intervals_of_monotonicity_extremal_values_l48_48677


namespace find_multiple_l48_48263

-- Defining the conditions
variables (A B k : ℕ)

-- Given conditions
def sum_condition : Prop := A + B = 77
def bigger_number_condition : Prop := A = 42

-- Using the conditions and aiming to prove that k = 5
theorem find_multiple
  (h1 : sum_condition A B)
  (h2 : bigger_number_condition A) :
  6 * B = k * A → k = 5 :=
by
  sorry

end find_multiple_l48_48263


namespace eggs_remainder_l48_48896

def daniel_eggs := 53
def eliza_eggs := 68
def fiona_eggs := 26
def george_eggs := 47
def total_eggs := daniel_eggs + eliza_eggs + fiona_eggs + george_eggs

theorem eggs_remainder :
  total_eggs % 15 = 14 :=
by
  sorry

end eggs_remainder_l48_48896


namespace misread_number_l48_48529

theorem misread_number (X : ℕ) :
  (average_10_initial : ℕ) = 18 →
  (incorrect_read : ℕ) = 26 →
  (average_10_correct : ℕ) = 22 →
  (10 * 22 - 10 * 18 = X + 26 - 26) →
  X = 66 :=
by sorry

end misread_number_l48_48529


namespace last_number_remaining_l48_48319

theorem last_number_remaining :
  (∃ f : ℕ → ℕ, ∃ n : ℕ, (∀ k < n, f (2 * k) = 2 * k + 2 ∧
                         ∀ k < n, f (2 * k + 1) = 2 * k + 1 + 2^(k+1)) ∧ 
                         n = 200 ∧ f (2 * n) = 128) :=
sorry

end last_number_remaining_l48_48319


namespace find_m_minus_n_l48_48875

noncomputable def m_abs := 4
noncomputable def n_abs := 6

theorem find_m_minus_n (m n : ℝ) (h1 : |m| = m_abs) (h2 : |n| = n_abs) (h3 : |m + n| = m + n) : m - n = -2 ∨ m - n = -10 :=
sorry

end find_m_minus_n_l48_48875


namespace number_of_false_propositions_is_even_l48_48125

theorem number_of_false_propositions_is_even 
  (P Q : Prop) : 
  ∃ (n : ℕ), (P ∧ ¬P ∧ (¬Q → ¬P) ∧ (Q → P)) = false ∧ n % 2 = 0 := sorry

end number_of_false_propositions_is_even_l48_48125


namespace total_eggs_found_l48_48808

def eggs_club_house := 12
def eggs_park := 5
def eggs_town_hall_garden := 3

theorem total_eggs_found : eggs_club_house + eggs_park + eggs_town_hall_garden = 20 :=
by
  sorry

end total_eggs_found_l48_48808


namespace stream_current_rate_l48_48474

theorem stream_current_rate (r w : ℝ) : 
  (15 / (r + w) + 5 = 15 / (r - w)) → 
  (15 / (2 * r + w) + 1 = 15 / (2 * r - w)) →
  w = 2 := 
by
  sorry

end stream_current_rate_l48_48474


namespace value_of_f_at_112_5_l48_48755

noncomputable def f : ℝ → ℝ := sorry

lemma f_even_func (x : ℝ) : f x = f (-x) := sorry
lemma f_func_eq (x : ℝ) : f x + f (x + 1) = 4 := sorry
lemma f_interval : ∀ x, -3 ≤ x ∧ x ≤ -2 → f x = 4 * x + 12 := sorry

theorem value_of_f_at_112_5 : f 112.5 = 2 := sorry

end value_of_f_at_112_5_l48_48755


namespace minimal_rotations_triangle_l48_48388

/-- Given a triangle with angles α, β, γ at vertices 1, 2, 3 respectively.
    The triangle returns to its original position after 15 rotations around vertex 1 by α,
    and after 6 rotations around vertex 2 by β.
    We need to show that the minimal positive integer n such that the triangle returns
    to its original position after n rotations around vertex 3 by γ is 5. -/
theorem minimal_rotations_triangle :
  ∃ (α β γ : ℝ) (k m l n : ℤ), 
    (15 * α = 360 * k) ∧ 
    (6 * β = 360 * m) ∧ 
    (α + β + γ = 180) ∧ 
    (n * γ = 360 * l) ∧ 
    (∀ n' : ℤ, n' > 0 → (∃ k' m' l' : ℤ, 
      (15 * α = 360 * k') ∧ 
      (6 * β = 360 * m') ∧ 
      (α + β + γ = 180) ∧ 
      (n' * γ = 360 * l') → n <= n')) ∧ 
    n = 5 := by
  sorry

end minimal_rotations_triangle_l48_48388


namespace ben_apples_difference_l48_48040

theorem ben_apples_difference (B P T : ℕ) (h1 : P = 40) (h2 : T = 18) (h3 : (3 / 8) * B = T) :
  B - P = 8 :=
sorry

end ben_apples_difference_l48_48040


namespace ratio_of_teenagers_to_toddlers_l48_48220

theorem ratio_of_teenagers_to_toddlers
  (total_children : ℕ)
  (number_of_toddlers : ℕ)
  (number_of_newborns : ℕ)
  (h1 : total_children = 40)
  (h2 : number_of_toddlers = 6)
  (h3 : number_of_newborns = 4)
  : (total_children - number_of_toddlers - number_of_newborns) / number_of_toddlers = 5 :=
by
  sorry

end ratio_of_teenagers_to_toddlers_l48_48220


namespace cost_of_paving_l48_48558

-- Definitions based on the given conditions
def length : ℝ := 6.5
def width : ℝ := 2.75
def rate : ℝ := 600

-- Theorem statement to prove the cost of paving
theorem cost_of_paving : length * width * rate = 10725 := by
  -- Calculation steps would go here, but we omit them with sorry
  sorry

end cost_of_paving_l48_48558


namespace min_fencing_l48_48314

variable (w l : ℝ)

noncomputable def area := w * l

noncomputable def length := 2 * w

theorem min_fencing (h1 : area w l ≥ 500) (h2 : l = length w) : 
  w = 5 * Real.sqrt 10 ∧ l = 10 * Real.sqrt 10 :=
  sorry

end min_fencing_l48_48314


namespace calculate_expression_l48_48249

theorem calculate_expression:
  202.2 * 89.8 - 20.22 * 186 + 2.022 * 3570 - 0.2022 * 16900 = 18198 :=
by
  sorry

end calculate_expression_l48_48249


namespace cannot_fold_patternD_to_cube_l48_48706

def patternA : Prop :=
  -- 5 squares arranged in a cross shape
  let squares := 5
  let shape  := "cross"
  squares = 5 ∧ shape = "cross"

def patternB : Prop :=
  -- 4 squares in a straight line
  let squares := 4
  let shape  := "line"
  squares = 4 ∧ shape = "line"

def patternC : Prop :=
  -- 3 squares in an L shape, and 2 squares attached to one end of the L making a T shape
  let squares := 5
  let shape  := "T"
  squares = 5 ∧ shape = "T"

def patternD : Prop :=
  -- 6 squares in a "+" shape with one extra square
  let squares := 7
  let shape  := "plus"
  squares = 7 ∧ shape = "plus"

theorem cannot_fold_patternD_to_cube :
  patternD → ¬ (patternA ∨ patternB ∨ patternC) :=
by
  sorry

end cannot_fold_patternD_to_cube_l48_48706


namespace lion_cubs_per_month_l48_48770

theorem lion_cubs_per_month
  (initial_lions : ℕ)
  (final_lions : ℕ)
  (months : ℕ)
  (lions_dying_per_month : ℕ)
  (net_increase : ℕ)
  (x : ℕ) : 
  initial_lions = 100 → 
  final_lions = 148 → 
  months = 12 → 
  lions_dying_per_month = 1 → 
  net_increase = 48 → 
  12 * (x - 1) = net_increase → 
  x = 5 := by
  intros initial_lions_eq final_lions_eq months_eq lions_dying_eq net_increase_eq equation
  sorry

end lion_cubs_per_month_l48_48770


namespace length_of_side_of_pentagon_l48_48110

-- Assuming these conditions from the math problem:
-- 1. The perimeter of the regular polygon is 125.
-- 2. The polygon is a pentagon (5 sides).

-- Let's define the conditions:
def perimeter := 125
def sides := 5
def regular_polygon (perimeter : ℕ) (sides : ℕ) := (perimeter / sides : ℕ)

-- Statement to be proved:
theorem length_of_side_of_pentagon : regular_polygon perimeter sides = 25 := 
by sorry

end length_of_side_of_pentagon_l48_48110


namespace roshini_spent_on_sweets_l48_48250

theorem roshini_spent_on_sweets
  (initial_amount : Real)
  (amount_given_per_friend : Real)
  (num_friends : Nat)
  (total_amount_given : Real)
  (amount_spent_on_sweets : Real) :
  initial_amount = 10.50 →
  amount_given_per_friend = 3.40 →
  num_friends = 2 →
  total_amount_given = amount_given_per_friend * num_friends →
  amount_spent_on_sweets = initial_amount - total_amount_given →
  amount_spent_on_sweets = 3.70 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end roshini_spent_on_sweets_l48_48250


namespace average_of_five_numbers_l48_48589

noncomputable def average_of_two (x1 x2 : ℝ) := (x1 + x2) / 2
noncomputable def average_of_three (x3 x4 x5 : ℝ) := (x3 + x4 + x5) / 3
noncomputable def average_of_five (x1 x2 x3 x4 x5 : ℝ) := (x1 + x2 + x3 + x4 + x5) / 5

theorem average_of_five_numbers (x1 x2 x3 x4 x5 : ℝ)
    (h1 : average_of_two x1 x2 = 12)
    (h2 : average_of_three x3 x4 x5 = 7) :
    average_of_five x1 x2 x3 x4 x5 = 9 := by
  sorry

end average_of_five_numbers_l48_48589


namespace problem1_problem2_problem3_l48_48938

theorem problem1 : (3 * Real.sqrt 2 - Real.sqrt 12) * (Real.sqrt 18 + 2 * Real.sqrt 3) = 6 :=
by
  sorry

theorem problem2 : (Real.sqrt 5 - Real.sqrt 6) ^ 2 - (Real.sqrt 5 + Real.sqrt 6) ^ 2 = -4 * Real.sqrt 30 :=
by
  sorry

theorem problem3 : (2 * Real.sqrt (3 / 2) - Real.sqrt (1 / 2)) * (1 / 2 * Real.sqrt 8 + Real.sqrt (2 / 3)) = (5 / 3) * Real.sqrt 3 + 1 :=
by
  sorry

end problem1_problem2_problem3_l48_48938


namespace inequality_proof_l48_48122

variable (x y : ℝ)
variable (h1 : x ≥ 0)
variable (h2 : y ≥ 0)
variable (h3 : x + y ≤ 1)

theorem inequality_proof (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + y ≤ 1) : 
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) := 
by 
  sorry

end inequality_proof_l48_48122


namespace tangent_line_ln_l48_48436

theorem tangent_line_ln (x y : ℝ) (h_curve : y = Real.log (x + 1)) (h_point : (1, Real.log 2) = (1, y)) :
  x - 2 * y - 1 + 2 * Real.log 2 = 0 :=
by
  sorry

end tangent_line_ln_l48_48436


namespace rook_path_exists_l48_48818

theorem rook_path_exists :
  ∃ (path : Finset (Fin 8 × Fin 8)) (s1 s2 : Fin 8 × Fin 8),
  s1 ≠ s2 ∧
  s1.1 % 2 = s2.1 % 2 ∧ s1.2 % 2 = s2.2 % 2 ∧
  ∀ s : Fin 8 × Fin 8, s ∈ path ∧ s ≠ s2 :=
sorry

end rook_path_exists_l48_48818


namespace slope_interval_non_intersect_l48_48532

noncomputable def parabola (x : ℝ) : ℝ := x^2 + 5

def Q : ℝ × ℝ := (10, 10)

theorem slope_interval_non_intersect (r s : ℝ) (h : ∀ m : ℝ,
  ¬∃ x : ℝ, parabola x = m * (x - 10) + 10 ↔ r < m ∧ m < s) :
  r + s = 40 :=
sorry

end slope_interval_non_intersect_l48_48532


namespace ratio_of_m_l48_48606

theorem ratio_of_m (a b m m1 m2 : ℚ) 
  (h1 : a^2 - 2*a + (3/m) = 0)
  (h2 : a + b = 2 - 2/m)
  (h3 : a * b = 3/m)
  (h4 : (a/b) + (b/a) = 3/2) 
  (h5 : 8 * m^2 - 31 * m + 8 = 0)
  (h6 : m1 + m2 = 31/8)
  (h7 : m1 * m2 = 1) :
  (m1/m2) + (m2/m1) = 833/64 :=
sorry

end ratio_of_m_l48_48606


namespace feet_in_mile_l48_48887

theorem feet_in_mile (d t : ℝ) (speed_mph : ℝ) (speed_fps : ℝ) (miles_to_feet : ℝ) (hours_to_seconds : ℝ) :
  d = 200 → t = 4 → speed_mph = 34.09 → miles_to_feet = 5280 → hours_to_seconds = 3600 → 
  speed_fps = d / t → speed_fps = speed_mph * miles_to_feet / hours_to_seconds → 
  miles_to_feet = 5280 :=
by
  intros hd ht hspeed_mph hmiles_to_feet hhours_to_seconds hspeed_fps_eq hconversion
  -- You can add the proof steps here.
  sorry

end feet_in_mile_l48_48887


namespace number_of_crystals_in_container_l48_48596

-- Define the dimensions of the energy crystal
def length_crystal := 30
def width_crystal := 25
def height_crystal := 5

-- Define the dimensions of the cubic container
def side_container := 27

-- Volume of the cubic container
def volume_container := side_container ^ 3

-- Volume of the energy crystal
def volume_crystal := length_crystal * width_crystal * height_crystal

-- Proof statement
theorem number_of_crystals_in_container :
  volume_container / volume_crystal ≥ 5 :=
sorry

end number_of_crystals_in_container_l48_48596


namespace find_x_plus_y_l48_48322

theorem find_x_plus_y (x y : ℚ) (h1 : |x| + x + y = 12) (h2 : x + |y| - y = 10) : x + y = 26/5 :=
sorry

end find_x_plus_y_l48_48322


namespace line_intersects_x_axis_at_point_l48_48493

theorem line_intersects_x_axis_at_point :
  ∃ x, (4 * x - 2 * 0 = 6) ∧ (2 - 0 = 2 * (0 - x)) → x = 2 := 
by
  sorry

end line_intersects_x_axis_at_point_l48_48493


namespace abs_triangle_inequality_l48_48669

theorem abs_triangle_inequality (x y z : ℝ) : 
  |x| + |y| + |z| ≤ |x + y - z| + |x - y + z| + |-x + y + z| :=
by sorry

end abs_triangle_inequality_l48_48669


namespace sandy_books_cost_l48_48636

theorem sandy_books_cost :
  ∀ (x : ℕ),
  (1280 + 880) / (x + 55) = 18 → 
  x = 65 :=
by
  intros x h
  sorry

end sandy_books_cost_l48_48636


namespace find_fx_neg_l48_48409

variable (f : ℝ → ℝ)

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def f_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≥ 0 → f x = x^2 - 2*x

theorem find_fx_neg (h1 : odd_function f) (h2 : f_nonneg f) : 
  ∀ x : ℝ, x < 0 → f x = -x^2 - 2*x := 
by
  sorry

end find_fx_neg_l48_48409


namespace abes_age_after_x_years_l48_48864

-- Given conditions
def A : ℕ := 28
def sum_condition (x : ℕ) : Prop := (A + (A - x) = 35)

-- Proof statement
theorem abes_age_after_x_years
  (x : ℕ)
  (h : sum_condition x) :
  (A + x = 49) :=
  sorry

end abes_age_after_x_years_l48_48864


namespace sum_not_fourteen_l48_48384

theorem sum_not_fourteen (a b c d : ℕ) (ha : 1 ≤ a ∧ a ≤ 6) (hb : 1 ≤ b ∧ b ≤ 6) (hc : 1 ≤ c ∧ c ≤ 6) (hd : 1 ≤ d ∧ d ≤ 6) 
  (hprod : a * b * c * d = 120) : a + b + c + d ≠ 14 :=
sorry

end sum_not_fourteen_l48_48384


namespace students_playing_both_l48_48898

theorem students_playing_both
    (total_students baseball_team hockey_team : ℕ)
    (h1 : total_students = 36)
    (h2 : baseball_team = 25)
    (h3 : hockey_team = 19)
    (h4 : total_students = baseball_team + hockey_team - students_both) :
    students_both = 8 := by
  sorry

end students_playing_both_l48_48898


namespace cylinder_volume_increase_factor_l48_48929

theorem cylinder_volume_increase_factor
    (π : Real)
    (r h : Real)
    (V_original : Real := π * r^2 * h)
    (new_height : Real := 3 * h)
    (new_radius : Real := 4 * r)
    (V_new : Real := π * (new_radius)^2 * new_height) :
    V_new / V_original = 48 :=
by
  sorry

end cylinder_volume_increase_factor_l48_48929


namespace james_money_left_no_foreign_currency_needed_l48_48441

noncomputable def JameMoneyLeftAfterPurchase : ℝ :=
  let usd_bills := 50 + 20 + 5 + 1 + 20 + 10 -- USD bills and coins
  let euro_in_usd := 5 * 1.20               -- €5 bill to USD
  let pound_in_usd := 2 * 1.35 - 0.8 / 100 * (2 * 1.35) -- £2 coin to USD after fee
  let yen_in_usd := 100 * 0.009 - 1.5 / 100 * (100 * 0.009) -- ¥100 coin to USD after fee
  let franc_in_usd := 2 * 1.08 - 1 / 100 * (2 * 1.08) -- 2₣ coins to USD after fee
  let total_usd := usd_bills + euro_in_usd + pound_in_usd + yen_in_usd + franc_in_usd
  let present_cost_with_tax := 88 * 1.08   -- Present cost after 8% tax
  total_usd - present_cost_with_tax        -- Amount left after purchasing the present

theorem james_money_left :
  JameMoneyLeftAfterPurchase = 22.6633 :=
by
  sorry

theorem no_foreign_currency_needed :
  (0 : ℝ)  = 0 :=
by
  sorry

end james_money_left_no_foreign_currency_needed_l48_48441


namespace lending_rate_is_7_percent_l48_48701

-- Conditions
def principal : ℝ := 5000
def borrowing_rate : ℝ := 0.04  -- 4% p.a. simple interest
def time : ℕ := 2  -- 2 years
def gain_per_year : ℝ := 150

-- Proof of the final statement
theorem lending_rate_is_7_percent :
  let borrowing_interest := principal * borrowing_rate * time / 100
  let interest_per_year := borrowing_interest / 2
  let total_interest_earned_per_year := interest_per_year + gain_per_year
  (total_interest_earned_per_year * 100) / principal = 7 :=
by
  sorry

end lending_rate_is_7_percent_l48_48701


namespace product_of_two_numbers_l48_48990
noncomputable def find_product (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) : ℝ :=
x * y

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) : find_product x y h1 h2 = 200 :=
sorry

end product_of_two_numbers_l48_48990


namespace john_unanswered_questions_l48_48549

theorem john_unanswered_questions
  (x y z : ℕ)
  (h1 : 5 * x + 2 * z = 93)
  (h2 : 4 * x - y = 54)
  (h3 : x + y + z = 30) : z = 9 :=
sorry

end john_unanswered_questions_l48_48549


namespace katie_sold_4_bead_necklaces_l48_48199

theorem katie_sold_4_bead_necklaces :
  ∃ (B : ℕ), 
    (∃ (G : ℕ), G = 3) ∧ 
    (∃ (C : ℕ), C = 3) ∧ 
    (∃ (T : ℕ), T = 21) ∧ 
    B * 3 + 3 * 3 = 21 :=
sorry

end katie_sold_4_bead_necklaces_l48_48199


namespace apples_fell_out_l48_48080

theorem apples_fell_out (initial_apples stolen_apples remaining_apples : ℕ) 
  (h₁ : initial_apples = 79) 
  (h₂ : stolen_apples = 45) 
  (h₃ : remaining_apples = 8) 
  : initial_apples - stolen_apples - remaining_apples = 26 := by
  sorry

end apples_fell_out_l48_48080


namespace polynomial_complete_square_l48_48600

theorem polynomial_complete_square :
  ∃ a h k : ℝ, (∀ x : ℝ, 4 * x^2 - 12 * x + 1 = a * (x - h)^2 + k) ∧ a + h + k = -2.5 := by
  sorry

end polynomial_complete_square_l48_48600


namespace alice_total_cost_usd_is_correct_l48_48519

def tea_cost_yen : ℕ := 250
def sandwich_cost_yen : ℕ := 350
def conversion_rate : ℕ := 100
def total_cost_usd (tea_cost_yen sandwich_cost_yen conversion_rate : ℕ) : ℕ :=
  (tea_cost_yen + sandwich_cost_yen) / conversion_rate

theorem alice_total_cost_usd_is_correct :
  total_cost_usd tea_cost_yen sandwich_cost_yen conversion_rate = 6 := 
by
  sorry

end alice_total_cost_usd_is_correct_l48_48519


namespace linear_system_incorrect_statement_l48_48412

def is_determinant (a b c d : ℝ) := a * d - b * c

def is_solution_system (a1 b1 c1 a2 b2 c2 D Dx Dy : ℝ) :=
  D = is_determinant a1 b1 a2 b2 ∧
  Dx = is_determinant c1 b1 c2 b2 ∧
  Dy = is_determinant a1 c1 a2 c2

def is_solution_linear_system (a1 b1 c1 a2 b2 c2 x y : ℝ) :=
  a1 * x + b1 * y = c1 ∧ a2 * x + b2 * y = c2

theorem linear_system_incorrect_statement :
  ∀ (x y : ℝ),
    is_solution_system 3 (-1) 1 1 3 7 10 10 20 ∧
    is_solution_linear_system 3 (-1) 1 1 3 7 x y →
    x = 1 ∧ y = 2 ∧ ¬(20 = -20) := 
by sorry

end linear_system_incorrect_statement_l48_48412


namespace identify_counterfeit_bag_l48_48096

-- Definitions based on problem conditions
def num_bags := 10
def genuine_weight := 10
def counterfeit_weight := 11
def expected_total_weight := genuine_weight * ((num_bags * (num_bags + 1)) / 2 : ℕ)

-- Lean theorem for the above problem
theorem identify_counterfeit_bag (W : ℕ) (Δ := W - expected_total_weight) :
  ∃ i : ℕ, 1 ≤ i ∧ i ≤ num_bags ∧ Δ = i :=
by sorry

end identify_counterfeit_bag_l48_48096


namespace stickers_in_either_not_both_l48_48789

def stickers_shared := 12
def emily_total_stickers := 22
def mia_unique_stickers := 10

theorem stickers_in_either_not_both : 
  (emily_total_stickers - stickers_shared) + mia_unique_stickers = 20 :=
by
  sorry

end stickers_in_either_not_both_l48_48789


namespace function_has_one_zero_l48_48241

-- Define the function f
def f (x m : ℝ) : ℝ := (m - 1) * x^2 + 2 * (m + 1) * x - 1

-- State the theorem
theorem function_has_one_zero (m : ℝ) :
  (∃! x : ℝ, f x m = 0) ↔ m = 0 ∨ m = -3 := 
sorry

end function_has_one_zero_l48_48241


namespace buckets_required_l48_48321

variable (C : ℝ) (N : ℝ)

theorem buckets_required (h : N * C = 105 * (2 / 5) * C) : N = 42 := 
  sorry

end buckets_required_l48_48321


namespace intersection_of_A_and_B_l48_48248

def setA (x : Real) : Prop := -1 < x ∧ x < 3
def setB (x : Real) : Prop := -2 < x ∧ x < 2

theorem intersection_of_A_and_B : {x : Real | setA x} ∩ {x : Real | setB x} = {x : Real | -1 < x ∧ x < 2} := 
by
  sorry

end intersection_of_A_and_B_l48_48248


namespace paco_ate_more_cookies_l48_48467

-- Define the number of cookies Paco originally had
def original_cookies : ℕ := 25

-- Define the number of cookies Paco ate
def eaten_cookies : ℕ := 5

-- Define the number of cookies Paco bought
def bought_cookies : ℕ := 3

-- Define the number of more cookies Paco ate than bought
def more_cookies_eaten_than_bought : ℕ := eaten_cookies - bought_cookies

-- Prove that Paco ate 2 more cookies than he bought
theorem paco_ate_more_cookies : more_cookies_eaten_than_bought = 2 := by
  sorry

end paco_ate_more_cookies_l48_48467


namespace total_octopus_legs_l48_48785

-- Define the number of octopuses Carson saw
def num_octopuses : ℕ := 5

-- Define the number of legs per octopus
def legs_per_octopus : ℕ := 8

-- Define or state the theorem for total number of legs
theorem total_octopus_legs : num_octopuses * legs_per_octopus = 40 := by
  sorry

end total_octopus_legs_l48_48785


namespace polynomial_relation_l48_48326

def M (m : ℚ) : ℚ := 5 * m^2 - 8 * m + 1
def N (m : ℚ) : ℚ := 4 * m^2 - 8 * m - 1

theorem polynomial_relation (m : ℚ) : M m > N m := by
  sorry

end polynomial_relation_l48_48326


namespace fg_minus_gf_l48_48698

-- Definitions provided by the conditions
def f (x : ℝ) : ℝ := 4 * x + 8
def g (x : ℝ) : ℝ := 2 * x - 3

theorem fg_minus_gf (x : ℝ) : f (g x) - g (f x) = -17 := 
  sorry

end fg_minus_gf_l48_48698


namespace original_number_q_l48_48082

variables (q : ℝ) (a b c : ℝ)
 
theorem original_number_q : 
  (a = 1.125 * q) → (b = 0.75 * q) → (c = 30) → (a - b = c) → q = 80 :=
by
  sorry

end original_number_q_l48_48082


namespace total_pizzas_served_l48_48567

def lunch_pizzas : ℚ := 12.5
def dinner_pizzas : ℚ := 8.25

theorem total_pizzas_served : lunch_pizzas + dinner_pizzas = 20.75 := by
  sorry

end total_pizzas_served_l48_48567


namespace division_remainder_l48_48112

def remainder (x y : ℕ) : ℕ := x % y

theorem division_remainder (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : (x : ℚ) / y = 96.15) (h4 : y = 20) : remainder x y = 3 :=
by
  sorry

end division_remainder_l48_48112


namespace central_angle_of_sector_l48_48336

theorem central_angle_of_sector (r : ℝ) (θ : ℝ) (h_perimeter: 2 * r + θ * r = π * r / 2) : θ = π - 2 :=
sorry

end central_angle_of_sector_l48_48336


namespace polynomial_expansion_l48_48812

theorem polynomial_expansion (a_0 a_1 a_2 a_3 a_4 : ℤ)
  (h1 : a_0 + a_1 + a_2 + a_3 + a_4 = 5^4)
  (h2 : a_0 - a_1 + a_2 - a_3 + a_4 = 1) :
  (a_0 + a_2 + a_4)^2 - (a_1 + a_3)^2 = 625 :=
by
  sorry

end polynomial_expansion_l48_48812


namespace length_of_AE_l48_48047

noncomputable def AE_calculation (AB AC AD : ℝ) (h : ℝ) (AE : ℝ) : Prop :=
  AB = 3.6 ∧ AC = 3.6 ∧ AD = 1.2 ∧ 
  (0.5 * AC * h = 0.5 * AE * (1/3) * h) →
  AE = 10.8

theorem length_of_AE {h : ℝ} : AE_calculation 3.6 3.6 1.2 h 10.8 :=
sorry

end length_of_AE_l48_48047


namespace integral_sin_pi_over_2_to_pi_l48_48489

theorem integral_sin_pi_over_2_to_pi : ∫ x in (Real.pi / 2)..Real.pi, Real.sin x = 1 := by
  sorry

end integral_sin_pi_over_2_to_pi_l48_48489


namespace largest_number_by_replacement_l48_48746

theorem largest_number_by_replacement 
  (n : ℝ) (n_1 n_3 n_6 n_8 : ℝ)
  (h : n = -0.3168)
  (h1 : n_1 = -0.3468)
  (h3 : n_3 = -0.4168)
  (h6 : n_6 = -0.3148)
  (h8 : n_8 = -0.3164)
  : n_6 > n_1 ∧ n_6 > n_3 ∧ n_6 > n_8 := 
by {
  -- Proof goes here
  sorry
}

end largest_number_by_replacement_l48_48746


namespace sum_of_integers_l48_48466

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 10) (h2 : x * y = 80) (hx_pos : 0 < x) (hy_pos : 0 < y) : x + y = 20 := by
  sorry

end sum_of_integers_l48_48466


namespace race_length_l48_48861

theorem race_length (A_time : ℕ) (diff_distance diff_time : ℕ) (A_time_eq : A_time = 380)
  (diff_distance_eq : diff_distance = 50) (diff_time_eq : diff_time = 20) :
  let B_speed := diff_distance / diff_time
  let B_time := A_time + diff_time
  let race_length := B_speed * B_time
  race_length = 1000 := 
by
  sorry

end race_length_l48_48861


namespace smallest_composite_proof_l48_48282

noncomputable def smallest_composite_no_prime_factors_less_than_15 : ℕ :=
  289

theorem smallest_composite_proof :
  smallest_composite_no_prime_factors_less_than_15 = 289 :=
by
  sorry

end smallest_composite_proof_l48_48282


namespace num_ordered_pairs_l48_48751

theorem num_ordered_pairs (x y : ℕ) (hx : x > 0) (hy : y > 0) (h : x * y = 4410) : 
  ∃ (n : ℕ), n = 36 :=
sorry

end num_ordered_pairs_l48_48751


namespace new_average_mark_of_remaining_students_l48_48141

def new_average (total_students : ℕ) (excluded_students : ℕ) (avg_marks : ℕ) (excluded_avg_marks : ℕ) : ℕ :=
  ((total_students * avg_marks) - (excluded_students * excluded_avg_marks)) / (total_students - excluded_students)

theorem new_average_mark_of_remaining_students 
  (total_students : ℕ) (excluded_students : ℕ) (avg_marks : ℕ) (excluded_avg_marks : ℕ)
  (h1 : total_students = 33)
  (h2 : excluded_students = 3)
  (h3 : avg_marks = 90)
  (h4 : excluded_avg_marks = 40) : 
  new_average total_students excluded_students avg_marks excluded_avg_marks = 95 :=
by
  sorry

end new_average_mark_of_remaining_students_l48_48141


namespace fuel_tank_capacity_l48_48763

theorem fuel_tank_capacity (C : ℝ) (h1 : 0.12 * 106 + 0.16 * (C - 106) = 30) : C = 214 :=
by
  sorry

end fuel_tank_capacity_l48_48763


namespace third_median_length_l48_48030

noncomputable def triangle_median_length (m₁ m₂ : ℝ) (area : ℝ) : ℝ :=
  if m₁ = 5 ∧ m₂ = 4 ∧ area = 6 * Real.sqrt 5 then
    3 * Real.sqrt 7
  else
    0

theorem third_median_length (m₁ m₂ : ℝ) (area : ℝ)
  (h₁ : m₁ = 5) (h₂ : m₂ = 4) (h₃ : area = 6 * Real.sqrt 5) :
  triangle_median_length m₁ m₂ area = 3 * Real.sqrt 7 :=
by
  -- Proof is skipped
  sorry

end third_median_length_l48_48030
