import Mathlib

namespace NUMINAMATH_GPT_wait_time_probability_l116_11648

theorem wait_time_probability
  (P_B1_8_00 : ℚ)
  (P_B1_8_20 : ℚ)
  (P_B1_8_40 : ℚ)
  (P_B2_9_00 : ℚ)
  (P_B2_9_20 : ℚ)
  (P_B2_9_40 : ℚ)
  (h_independent : true)
  (h_employee_arrival : true)
  (h_P_B1 : P_B1_8_00 = 1/4 ∧ P_B1_8_20 = 1/2 ∧ P_B1_8_40 = 1/4)
  (h_P_B2 : P_B2_9_00 = 1/4 ∧ P_B2_9_20 = 1/2 ∧ P_B2_9_40 = 1/4) :
  (P_B1_8_00 * P_B2_9_20 + P_B1_8_00 * P_B2_9_40 = 3/16) :=
sorry

end NUMINAMATH_GPT_wait_time_probability_l116_11648


namespace NUMINAMATH_GPT_regular_polygon_sides_l116_11600

theorem regular_polygon_sides (h : ∀ n : ℕ, (120 * n) = 180 * (n - 2)) : 6 = 6 :=
by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l116_11600


namespace NUMINAMATH_GPT_Alyssa_puppies_l116_11620

theorem Alyssa_puppies (initial_puppies : ℕ) (given_puppies : ℕ)
  (h_initial : initial_puppies = 7) (h_given : given_puppies = 5) :
  initial_puppies - given_puppies = 2 :=
by
  sorry

end NUMINAMATH_GPT_Alyssa_puppies_l116_11620


namespace NUMINAMATH_GPT_square_difference_l116_11659

theorem square_difference (x : ℤ) (h : x^2 = 9801) : (x - 2) * (x + 2) = 9797 :=
by 
  have diff_squares : (x - 2) * (x + 2) = x^2 - 4 := by ring
  rw [diff_squares, h]
  norm_num

end NUMINAMATH_GPT_square_difference_l116_11659


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l116_11675

theorem arithmetic_sequence_sum (n : ℕ) (S : ℕ → ℕ) (h1 : S n = 54) (h2 : S (2 * n) = 72) :
  S (3 * n) = 78 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l116_11675


namespace NUMINAMATH_GPT_find_d_l116_11692

theorem find_d (c : ℝ) (d : ℝ) (α : ℝ) (β : ℝ) (γ : ℝ) (ω : ℝ)
  (h1 : α = c)
  (h2 : β = 43)
  (h3 : γ = 59)
  (h4 : ω = d)
  (h5 : c = 36) :
  d = 42 := 
sorry

end NUMINAMATH_GPT_find_d_l116_11692


namespace NUMINAMATH_GPT_complex_multiplication_quadrant_l116_11697

-- Given conditions
def complex_mul (z1 z2 : ℂ) : ℂ := z1 * z2

-- Proving point is in the fourth quadrant
theorem complex_multiplication_quadrant
  (a b : ℝ) (z : ℂ)
  (h1 : z = a + b * Complex.I)
  (h2 : z = complex_mul (1 + Complex.I) (3 - Complex.I)) :
  b < 0 ∧ a > 0 :=
by
  sorry

end NUMINAMATH_GPT_complex_multiplication_quadrant_l116_11697


namespace NUMINAMATH_GPT_lateral_surface_area_of_cylinder_l116_11633

theorem lateral_surface_area_of_cylinder :
  let r := 1
  let h := 2
  2 * Real.pi * r * h = 4 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_lateral_surface_area_of_cylinder_l116_11633


namespace NUMINAMATH_GPT_tickets_won_in_skee_ball_l116_11621

-- Define the conditions as Lean definitions
def tickets_from_whack_a_mole : ℕ := 8
def ticket_cost_per_candy : ℕ := 5
def candies_bought : ℕ := 3

-- We now state the conjecture (mathematical proof problem) 
-- Prove that the number of tickets won in skee ball is 7.
theorem tickets_won_in_skee_ball :
  (candies_bought * ticket_cost_per_candy) - tickets_from_whack_a_mole = 7 :=
by
  sorry

end NUMINAMATH_GPT_tickets_won_in_skee_ball_l116_11621


namespace NUMINAMATH_GPT_total_slices_l116_11678

theorem total_slices (pizzas : ℕ) (slices1 slices2 slices3 slices4 : ℕ)
  (h1 : pizzas = 4)
  (h2 : slices1 = 8)
  (h3 : slices2 = 8)
  (h4 : slices3 = 10)
  (h5 : slices4 = 12) :
  slices1 + slices2 + slices3 + slices4 = 38 := by
  sorry

end NUMINAMATH_GPT_total_slices_l116_11678


namespace NUMINAMATH_GPT_integer_solutions_system_ineq_l116_11642

theorem integer_solutions_system_ineq (x : ℤ) :
  (3 * x + 6 > x + 8 ∧ (x : ℚ) / 4 ≥ (x - 1) / 3) ↔ (x = 2 ∨ x = 3 ∨ x = 4) :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_system_ineq_l116_11642


namespace NUMINAMATH_GPT_alexandra_brianna_meeting_probability_l116_11626

noncomputable def probability_meeting (A B : ℕ × ℕ) : ℚ :=
if A = (0,0) ∧ B = (5,7) then 347 / 768 else 0

theorem alexandra_brianna_meeting_probability :
  probability_meeting (0,0) (5,7) = 347 / 768 := 
by sorry

end NUMINAMATH_GPT_alexandra_brianna_meeting_probability_l116_11626


namespace NUMINAMATH_GPT_initial_cats_l116_11638

-- Define the conditions as hypotheses
variables (total_cats now : ℕ) (cats_given : ℕ)

-- State the main theorem
theorem initial_cats:
  total_cats = 31 → cats_given = 14 → (total_cats - cats_given) = 17 :=
by sorry

end NUMINAMATH_GPT_initial_cats_l116_11638


namespace NUMINAMATH_GPT_possible_remainders_of_a2_l116_11613

theorem possible_remainders_of_a2 (p : ℕ) (k : ℕ) (hp : Nat.Prime p) (hk : 0 < k) 
  (hresidue : ∀ i : ℕ, i < p → ∃ j : ℕ, j < p ∧ ((j^k+j) % p = i)) :
  ∃ s : Finset ℕ, s = Finset.range p ∧ (2^k + 2) % p ∈ s := 
sorry

end NUMINAMATH_GPT_possible_remainders_of_a2_l116_11613


namespace NUMINAMATH_GPT_solve_quartic_eqn_l116_11682

noncomputable def solutionSet : Set ℂ :=
  {x | x^2 = 6 ∨ x^2 = -6}

theorem solve_quartic_eqn (x : ℂ) : (x^4 - 36 = 0) ↔ (x ∈ solutionSet) := 
sorry

end NUMINAMATH_GPT_solve_quartic_eqn_l116_11682


namespace NUMINAMATH_GPT_contrapositive_l116_11606

-- Definitions based on the conditions
def original_proposition (a b : ℝ) : Prop := a^2 + b^2 = 0 → a = 0 ∧ b = 0

-- The theorem to prove the contrapositive
theorem contrapositive (a b : ℝ) : original_proposition a b ↔ (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) :=
sorry

end NUMINAMATH_GPT_contrapositive_l116_11606


namespace NUMINAMATH_GPT_egg_sales_l116_11668

/-- Two vendors together sell 110 eggs and both have equal revenues.
    Given the conditions about changing the number of eggs and corresponding revenues,
    the first vendor sells 60 eggs and the second vendor sells 50 eggs. -/
theorem egg_sales (x y : ℝ) (h1 : x + (110 - x) = 110) (h2 : 110 * (y / x) = 5) (h3 : 110 * (y / (110 - x)) = 7.2) :
  x = 60 ∧ (110 - x) = 50 :=
by sorry

end NUMINAMATH_GPT_egg_sales_l116_11668


namespace NUMINAMATH_GPT_arithmetic_seq_sum_l116_11629

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0)
    (h_sum : a 0 + a 1 + a 2 + a 3 = 30) : a 1 + a 2 = 15 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_l116_11629


namespace NUMINAMATH_GPT_comic_books_ratio_l116_11601

variable (S : ℕ)

def initial_comics := 22
def remaining_comics := 17
def comics_bought := 6

theorem comic_books_ratio (h1 : initial_comics - S + comics_bought = remaining_comics) :
  (S : ℚ) / initial_comics = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_comic_books_ratio_l116_11601


namespace NUMINAMATH_GPT_evening_sales_l116_11609

theorem evening_sales
  (remy_bottles_morning : ℕ := 55)
  (nick_bottles_fewer : ℕ := 6)
  (price_per_bottle : ℚ := 0.50)
  (evening_sales_more : ℚ := 3) :
  let nick_bottles_morning := remy_bottles_morning - nick_bottles_fewer
  let remy_sales_morning := remy_bottles_morning * price_per_bottle
  let nick_sales_morning := nick_bottles_morning * price_per_bottle
  let total_morning_sales := remy_sales_morning + nick_sales_morning
  let total_evening_sales := total_morning_sales + evening_sales_more
  total_evening_sales = 55 :=
by
  sorry

end NUMINAMATH_GPT_evening_sales_l116_11609


namespace NUMINAMATH_GPT_smallest_number_of_students_l116_11624

theorem smallest_number_of_students
  (tenth_graders eighth_graders ninth_graders : ℕ)
  (ratio1 : 7 * eighth_graders = 4 * tenth_graders)
  (ratio2 : 9 * ninth_graders = 5 * tenth_graders) :
  (∀ n, (∃ a b c, a = 7 * b ∧ b = 4 * n ∧ a = 9 * c ∧ c = 5 * n) → n = 134) :=
by {
  -- We currently just assume the result for Lean to be syntactically correct
  sorry
}

end NUMINAMATH_GPT_smallest_number_of_students_l116_11624


namespace NUMINAMATH_GPT_complement_intersection_l116_11696

-- Define the universal set U and sets A, B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Define the intersection of A and B
def A_inter_B : Set ℕ := {x ∈ A | x ∈ B}

-- Define the complement of A_inter_B in U
def complement_U_A_inter_B : Set ℕ := {x ∈ U | x ∉ A_inter_B}

-- Prove that the complement of the intersection of A and B in U is {1, 4, 5}
theorem complement_intersection :
  complement_U_A_inter_B = {1, 4, 5} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l116_11696


namespace NUMINAMATH_GPT_factorization_example_l116_11607

theorem factorization_example :
  (4 : ℤ) * x^2 - 1 = (2 * x + 1) * (2 * x - 1) := 
by
  sorry

end NUMINAMATH_GPT_factorization_example_l116_11607


namespace NUMINAMATH_GPT_product_of_numbers_l116_11615

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 7) (h2 : x^2 + y^2 = 85) : x * y = 18 := by
  sorry

end NUMINAMATH_GPT_product_of_numbers_l116_11615


namespace NUMINAMATH_GPT_student_answers_all_correctly_l116_11695

/-- 
The exam tickets have 2 theoretical questions and 1 problem each. There are 28 tickets. 
A student is prepared for 50 theoretical questions out of 56 and 22 problems out of 28.
The probability that by drawing a ticket at random, and the student answers all questions 
correctly is 0.625.
-/
theorem student_answers_all_correctly :
  let total_theoretical := 56
  let total_problems := 28
  let prepared_theoretical := 50
  let prepared_problems := 22
  let p_correct_theoretical := (prepared_theoretical * (prepared_theoretical - 1)) / (total_theoretical * (total_theoretical - 1))
  let p_correct_problem := prepared_problems / total_problems
  let combined_probability := p_correct_theoretical * p_correct_problem
  combined_probability = 0.625 :=
  sorry

end NUMINAMATH_GPT_student_answers_all_correctly_l116_11695


namespace NUMINAMATH_GPT_no_two_perfect_cubes_l116_11658

theorem no_two_perfect_cubes (n : ℕ) : ¬ (∃ a b : ℕ, a^3 = n + 2 ∧ b^3 = n^2 + n + 1) := by
  sorry

end NUMINAMATH_GPT_no_two_perfect_cubes_l116_11658


namespace NUMINAMATH_GPT_divisor_of_first_division_l116_11688

theorem divisor_of_first_division (n d : ℕ) (hn_pos : 0 < n)
  (h₁ : (n + 1) % d = 4) (h₂ : n % 2 = 1) : 
  d = 6 :=
sorry

end NUMINAMATH_GPT_divisor_of_first_division_l116_11688


namespace NUMINAMATH_GPT_evaluate_expression_l116_11612

noncomputable def a := Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35 + 2
noncomputable def b := -Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35 + 2
noncomputable def c := Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35 + 2
noncomputable def d := -Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35 + 2

theorem evaluate_expression : (1 / a + 1 / b + 1 / c + 1 / d)^2 = 39 / 140 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l116_11612


namespace NUMINAMATH_GPT_min_colors_required_l116_11676

-- Define predicate for the conditions
def conditions (n : ℕ) (m : ℕ) (k : ℕ)(Paint : ℕ → Set ℕ) : Prop := 
  (∀ S : Finset ℕ, S.card = n → (∃ c ∈ ⋃ p ∈ S, Paint p, c ∈ S)) ∧ 
  (∀ c, ¬ (∀ i ∈ (Finset.range m).1, c ∈ Paint i))

-- The main theorem statement
theorem min_colors_required :
  ∀ (Paint : ℕ → Set ℕ), conditions 20 100 21 Paint → 
  ∃ k, conditions 20 100 k Paint ∧ k = 21 :=
sorry

end NUMINAMATH_GPT_min_colors_required_l116_11676


namespace NUMINAMATH_GPT_Helen_raisins_l116_11630

/-- Given that Helen baked 19 chocolate chip cookies yesterday, baked some raisin cookies and 237 chocolate chip cookies this morning,
    and baked 25 more chocolate chip cookies than raisin cookies in total,
    prove that the number of raisin cookies (R) she baked is 231. -/
theorem Helen_raisins (R : ℕ) (h1 : 25 + R = 256) : R = 231 :=
by
  sorry

end NUMINAMATH_GPT_Helen_raisins_l116_11630


namespace NUMINAMATH_GPT_tony_walking_speed_l116_11645

-- Define the conditions as hypotheses
def walking_speed_on_weekend (W : ℝ) : Prop := 
  let store_distance := 4 
  let run_speed := 10
  let day1_time := store_distance / W
  let day2_time := store_distance / run_speed
  let day3_time := store_distance / run_speed
  let avg_time := (day1_time + day2_time + day3_time) / 3
  avg_time = 56 / 60

-- State the theorem
theorem tony_walking_speed : ∃ W : ℝ, walking_speed_on_weekend W ∧ W = 2 := 
sorry

end NUMINAMATH_GPT_tony_walking_speed_l116_11645


namespace NUMINAMATH_GPT_correct_calculation_l116_11664

theorem correct_calculation (a b : ℝ) :
  ((ab)^3 = a^3 * b^3) ∧ 
  ¬(a + 2 * a^2 = 3 * a^3) ∧ 
  ¬(a * (-a)^4 = -a^5) ∧ 
  ¬((a^3)^2 = a^5) :=
  by
  sorry

end NUMINAMATH_GPT_correct_calculation_l116_11664


namespace NUMINAMATH_GPT_plan1_more_cost_effective_than_plan2_l116_11641

variable (x : ℝ)

def plan1_cost (x : ℝ) : ℝ :=
  36 + 0.1 * x

def plan2_cost (x : ℝ) : ℝ :=
  0.6 * x

theorem plan1_more_cost_effective_than_plan2 (h : x > 72) : 
  plan1_cost x < plan2_cost x :=
by
  sorry

end NUMINAMATH_GPT_plan1_more_cost_effective_than_plan2_l116_11641


namespace NUMINAMATH_GPT_age_of_person_l116_11654

theorem age_of_person (x : ℕ) (h : 3 * (x + 3) - 3 * (x - 3) = x) : x = 18 :=
  sorry

end NUMINAMATH_GPT_age_of_person_l116_11654


namespace NUMINAMATH_GPT_number_division_l116_11670

theorem number_division (n q r d : ℕ) (h1 : d = 18) (h2 : q = 11) (h3 : r = 1) (h4 : n = (d * q) + r) : n = 199 := 
by 
  sorry

end NUMINAMATH_GPT_number_division_l116_11670


namespace NUMINAMATH_GPT_pumpkins_eaten_l116_11635

theorem pumpkins_eaten (initial: ℕ) (left: ℕ) (eaten: ℕ) (h1 : initial = 43) (h2 : left = 20) : eaten = 23 :=
by {
  -- We are skipping the proof as per the requirement
  sorry
}

end NUMINAMATH_GPT_pumpkins_eaten_l116_11635


namespace NUMINAMATH_GPT_sum_of_A_H_l116_11623

theorem sum_of_A_H (A B C D E F G H : ℝ) (h1 : C = 10) 
  (h2 : A + B + C = 40) (h3 : B + C + D = 40) (h4 : C + D + E = 40) 
  (h5 : D + E + F = 40) (h6 : E + F + G = 40) (h7 : F + G + H = 40) :
  A + H = 30 := 
sorry

end NUMINAMATH_GPT_sum_of_A_H_l116_11623


namespace NUMINAMATH_GPT_angle_B_in_triangle_ABC_side_b_in_triangle_ABC_l116_11639

-- Conditions for (1): In ΔABC, A = 60°, a = 4√3, b = 4√2, prove B = 45°.
theorem angle_B_in_triangle_ABC
  (A : Real)
  (a b : Real)
  (hA : A = 60)
  (ha : a = 4 * Real.sqrt 3)
  (hb : b = 4 * Real.sqrt 2) :
  ∃ B : Real, B = 45 := by
  sorry

-- Conditions for (2): In ΔABC, a = 3√3, c = 2, B = 150°, prove b = 7.
theorem side_b_in_triangle_ABC
  (a c B : Real)
  (ha : a = 3 * Real.sqrt 3)
  (hc : c = 2)
  (hB : B = 150) :
  ∃ b : Real, b = 7 := by
  sorry

end NUMINAMATH_GPT_angle_B_in_triangle_ABC_side_b_in_triangle_ABC_l116_11639


namespace NUMINAMATH_GPT_part1_inequality_part2_inequality_case1_part2_inequality_case2_part2_inequality_case3_l116_11651

-- Part (1)
theorem part1_inequality (m : ℝ) : (∀ x : ℝ, (m^2 + 1)*x^2 - (2*m - 1)*x + 1 > 0) ↔ m > -3/4 := sorry

-- Part (2)
theorem part2_inequality_case1 (a : ℝ) (h : 0 < a ∧ a < 1) : 
  (∀ x : ℝ, (x - 1)*(a*x - 1) > 0 ↔ x < 1 ∨ x > 1/a) := sorry

theorem part2_inequality_case2 : 
  (∀ x : ℝ, (x - 1)*(0*x - 1) > 0 ↔ x < 1) := sorry

theorem part2_inequality_case3 (a : ℝ) (h : a < 0) : 
  (∀ x : ℝ, (x - 1)*(a*x - 1) > 0 ↔ 1/a < x ∧ x < 1) := sorry

end NUMINAMATH_GPT_part1_inequality_part2_inequality_case1_part2_inequality_case2_part2_inequality_case3_l116_11651


namespace NUMINAMATH_GPT_additional_matches_l116_11617

theorem additional_matches 
  (avg_runs_first_25 : ℕ → ℚ) 
  (avg_runs_additional : ℕ → ℚ) 
  (avg_runs_all : ℚ) 
  (total_matches_first_25 : ℕ) 
  (total_matches_all : ℕ) 
  (total_runs_first_25 : ℚ) 
  (total_runs_all : ℚ) 
  (x : ℕ)
  (h1 : avg_runs_first_25 25 = 45)
  (h2 : avg_runs_additional x = 15)
  (h3 : avg_runs_all = 38.4375)
  (h4 : total_matches_first_25 = 25)
  (h5 : total_matches_all = 32)
  (h6 : total_runs_first_25 = avg_runs_first_25 25 * 25)
  (h7 : total_runs_all = avg_runs_all * 32)
  (h8 : total_runs_first_25 + avg_runs_additional x * x = total_runs_all) :
  x = 7 :=
sorry

end NUMINAMATH_GPT_additional_matches_l116_11617


namespace NUMINAMATH_GPT_monomial_sum_l116_11673

theorem monomial_sum (m n : ℤ) (h1 : n - 1 = 4) (h2 : m - 1 = 2) : m - 2 * n = -7 := by
  sorry

end NUMINAMATH_GPT_monomial_sum_l116_11673


namespace NUMINAMATH_GPT_peter_savings_l116_11646

noncomputable def calc_discounted_price (original_price : ℝ) (discount_percentage : ℝ) : ℝ :=
    original_price * (1 - discount_percentage / 100)

noncomputable def calc_savings (original_price : ℝ) (external_price : ℝ) : ℝ :=
    original_price - external_price

noncomputable def total_savings : ℝ :=
    let math_original := 45.0
    let math_discount := 20.0
    let science_original := 60.0
    let science_discount := 25.0
    let literature_original := 35.0
    let literature_discount := 15.0
    let math_external := calc_discounted_price math_original math_discount
    let science_external := calc_discounted_price science_original science_discount
    let literature_external := calc_discounted_price literature_original literature_discount
    let math_savings := calc_savings math_original math_external
    let science_savings := calc_savings science_original science_external
    let literature_savings := calc_savings literature_original literature_external
    math_savings + science_savings + literature_savings

theorem peter_savings :
  total_savings = 29.25 :=
by
    sorry

end NUMINAMATH_GPT_peter_savings_l116_11646


namespace NUMINAMATH_GPT_modulus_of_z_l116_11690

open Complex -- Open the Complex number namespace

-- Define the given condition as a hypothesis
def condition (z : ℂ) : Prop := (1 + I) * z = 3 + I

-- Statement of the theorem
theorem modulus_of_z (z : ℂ) (h : condition z) : Complex.abs z = Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_modulus_of_z_l116_11690


namespace NUMINAMATH_GPT_vacation_hours_per_week_l116_11619

open Nat

theorem vacation_hours_per_week :
  let planned_hours_per_week := 25
  let total_weeks := 15
  let total_money_needed := 4500
  let sick_weeks := 3
  let hourly_rate := total_money_needed / (planned_hours_per_week * total_weeks)
  let remaining_weeks := total_weeks - sick_weeks
  let total_hours_needed := total_money_needed / hourly_rate
  let required_hours_per_week := total_hours_needed / remaining_weeks
  required_hours_per_week = 31.25 := by
sorry

end NUMINAMATH_GPT_vacation_hours_per_week_l116_11619


namespace NUMINAMATH_GPT_frank_peanuts_average_l116_11677

theorem frank_peanuts_average :
  let one_dollar := 7 * 1
  let five_dollar := 4 * 5
  let ten_dollar := 2 * 10
  let twenty_dollar := 1 * 20
  let total_money := one_dollar + five_dollar + ten_dollar + twenty_dollar
  let change := 4
  let money_spent := total_money - change
  let cost_per_pound := 3
  let total_pounds := money_spent / cost_per_pound
  let days := 7
  let average_per_day := total_pounds / days
  average_per_day = 3 :=
by
  sorry

end NUMINAMATH_GPT_frank_peanuts_average_l116_11677


namespace NUMINAMATH_GPT_fill_parentheses_l116_11604

variable (a b : ℝ)

theorem fill_parentheses :
  1 - a^2 + 2 * a * b - b^2 = 1 - (a^2 - 2 * a * b + b^2) :=
by
  sorry

end NUMINAMATH_GPT_fill_parentheses_l116_11604


namespace NUMINAMATH_GPT_log_base_function_inequalities_l116_11602

/-- 
Given the function y = log_(1/(sqrt(2))) (1/(x + 3)),
prove that:
1. for y > 0, x ∈ (-2, +∞)
2. for y < 0, x ∈ (-3, -2)
-/
theorem log_base_function_inequalities :
  let y (x : ℝ) := Real.logb (1 / Real.sqrt 2) (1 / (x + 3))
  ∀ x : ℝ, (y x > 0 ↔ x > -2) ∧ (y x < 0 ↔ -3 < x ∧ x < -2) :=
by
  intros
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_log_base_function_inequalities_l116_11602


namespace NUMINAMATH_GPT_fraction_is_one_third_l116_11614

theorem fraction_is_one_third :
  (3 + 9 - 27 + 81 + 243 - 729) / (9 + 27 - 81 + 243 + 729 - 2187) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_is_one_third_l116_11614


namespace NUMINAMATH_GPT_negation_of_p_l116_11687

def p := ∀ x : ℝ, Real.sin x ≤ 1

theorem negation_of_p : ¬p ↔ ∃ x : ℝ, Real.sin x > 1 := 
by 
  sorry

end NUMINAMATH_GPT_negation_of_p_l116_11687


namespace NUMINAMATH_GPT_sin_cos_power_sum_l116_11683

theorem sin_cos_power_sum (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 4) : 
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 61 / 64 := 
by
  sorry

end NUMINAMATH_GPT_sin_cos_power_sum_l116_11683


namespace NUMINAMATH_GPT_true_propositions_count_l116_11674

theorem true_propositions_count
  (a b c : ℝ)
  (h : a > b) :
  ( (a > b → a * c^2 > b * c^2) ∧
    (a * c^2 > b * c^2 → a > b) ∧
    (a ≤ b → a * c^2 ≤ b * c^2) ∧
    (a * c^2 ≤ b * c^2 → a ≤ b) 
  ) ∧ 
  (¬(a > b → a * c^2 > b * c^2) ∧
   ¬(a * c^2 ≤ b * c^2 → a ≤ b)) →
  (a * c^2 > b * c^2 → a > b) ∧
  (a ≤ b → a * c^2 ≤ b * c^2) ∨
  (a > b → a * c^2 > b * c^2) ∨
  (a * c^2 ≤ b * c^2 → a ≤ b) :=
sorry

end NUMINAMATH_GPT_true_propositions_count_l116_11674


namespace NUMINAMATH_GPT_range_of_magnitudes_l116_11699

theorem range_of_magnitudes (AB AC BC : ℝ) (hAB : AB = 8) (hAC : AC = 5) :
  3 ≤ BC ∧ BC ≤ 13 :=
by
  sorry

end NUMINAMATH_GPT_range_of_magnitudes_l116_11699


namespace NUMINAMATH_GPT_hannah_money_left_l116_11689

variable (initial_amount : ℕ) (amount_spent_rides : ℕ) (amount_spent_dessert : ℕ)
  (remaining_after_rides : ℕ) (remaining_money : ℕ)

theorem hannah_money_left :
  initial_amount = 30 →
  amount_spent_rides = initial_amount / 2 →
  remaining_after_rides = initial_amount - amount_spent_rides →
  amount_spent_dessert = 5 →
  remaining_money = remaining_after_rides - amount_spent_dessert →
  remaining_money = 10 := by
  sorry

end NUMINAMATH_GPT_hannah_money_left_l116_11689


namespace NUMINAMATH_GPT_second_range_is_18_l116_11694

variable (range1 range2 range3 : ℕ)

theorem second_range_is_18
  (h1 : range1 = 30)
  (h2 : range2 = 18)
  (h3 : range3 = 32) :
  range2 = 18 := by
  sorry

end NUMINAMATH_GPT_second_range_is_18_l116_11694


namespace NUMINAMATH_GPT_final_laptop_price_l116_11693

theorem final_laptop_price :
  let original_price := 1000.00
  let first_discounted_price := original_price * (1 - 0.10)
  let second_discounted_price := first_discounted_price * (1 - 0.25)
  let recycling_fee := second_discounted_price * 0.05
  let final_price := second_discounted_price + recycling_fee
  final_price = 708.75 :=
by
  sorry

end NUMINAMATH_GPT_final_laptop_price_l116_11693


namespace NUMINAMATH_GPT_intersection_points_eq_one_l116_11603

-- Definitions for the equations of the circles
def circle1 (x y : ℝ) : ℝ := x^2 + (y - 3)^2
def circle2 (x y : ℝ) : ℝ := x^2 + (y + 2)^2

-- The proof problem statement
theorem intersection_points_eq_one : 
  ∃ p : ℝ × ℝ, (circle1 p.1 p.2 = 9) ∧ (circle2 p.1 p.2 = 4) ∧
  (∀ q : ℝ × ℝ, (circle1 q.1 q.2 = 9) ∧ (circle2 q.1 q.2 = 4) → q = p) :=
sorry

end NUMINAMATH_GPT_intersection_points_eq_one_l116_11603


namespace NUMINAMATH_GPT_amare_needs_more_fabric_l116_11632

theorem amare_needs_more_fabric :
  let first_two_dresses_in_feet := 2 * 5.5 * 3
  let next_two_dresses_in_feet := 2 * 6 * 3
  let last_two_dresses_in_feet := 2 * 6.5 * 3
  let total_fabric_needed := first_two_dresses_in_feet + next_two_dresses_in_feet + last_two_dresses_in_feet
  let fabric_amare_has := 10
  total_fabric_needed - fabric_amare_has = 98 :=
by {
  sorry
}

end NUMINAMATH_GPT_amare_needs_more_fabric_l116_11632


namespace NUMINAMATH_GPT_find_t_eq_l116_11643

variable (a V V_0 S t : ℝ)

theorem find_t_eq (h1 : V = a * t + V_0) (h2 : S = (1/3) * a * t^3 + V_0 * t) : t = (V - V_0) / a :=
sorry

end NUMINAMATH_GPT_find_t_eq_l116_11643


namespace NUMINAMATH_GPT_initial_cupcakes_baked_l116_11649

variable (toddAte := 21)       -- Todd ate 21 cupcakes.
variable (packages := 6)       -- She could make 6 packages.
variable (cupcakesPerPackage := 3) -- Each package contains 3 cupcakes.
variable (cupcakesLeft := packages * cupcakesPerPackage) -- Cupcakes left after Todd ate some.

theorem initial_cupcakes_baked : cupcakesLeft + toddAte = 39 :=
by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_initial_cupcakes_baked_l116_11649


namespace NUMINAMATH_GPT_statement_A_solution_set_statement_B_insufficient_condition_statement_C_negation_statement_D_not_necessary_condition_l116_11637

-- Statement A: Proving the solution set of the inequality
theorem statement_A_solution_set (x : ℝ) : 
  (x + 2) / (2 * x + 1) > 1 ↔ (-1 / 2) < x ∧ x < 1 :=
sorry

-- Statement B: "ab > 1" is not a sufficient condition for "a > 1, b > 1"
theorem statement_B_insufficient_condition (a b : ℝ) :
  (a * b > 1) → ¬(a > 1 ∧ b > 1) :=
sorry

-- Statement C: The negation of p: ∀ x ∈ ℝ, x² > 0 is true
theorem statement_C_negation (x0 : ℝ) : 
  (∀ x : ℝ, x^2 > 0) → ¬ (∃ x0 : ℝ, x0^2 ≤ 0) :=
sorry

-- Statement D: "a < 2" is not a necessary condition for "a < 6"
theorem statement_D_not_necessary_condition (a : ℝ) :
  (a < 2) → ¬(a < 6) :=
sorry

end NUMINAMATH_GPT_statement_A_solution_set_statement_B_insufficient_condition_statement_C_negation_statement_D_not_necessary_condition_l116_11637


namespace NUMINAMATH_GPT_van_distance_covered_l116_11665

noncomputable def distance_covered (V : ℝ) := 
  let D := V * 6
  D

theorem van_distance_covered : ∃ (D : ℝ), ∀ (V : ℝ), 
  (D = 288) ∧ (D = distance_covered V) ∧ (D = 32 * 9) :=
by
  sorry

end NUMINAMATH_GPT_van_distance_covered_l116_11665


namespace NUMINAMATH_GPT_train_length_is_correct_l116_11650

noncomputable def speed_kmph_to_mps (speed : ℝ) : ℝ :=
  speed * 1000 / 3600

noncomputable def distance_crossed (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

noncomputable def train_length (speed_kmph crossing_time bridge_length : ℝ) : ℝ :=
  distance_crossed (speed_kmph_to_mps speed_kmph) crossing_time - bridge_length

theorem train_length_is_correct :
  ∀ (crossing_time bridge_length speed_kmph : ℝ),
    crossing_time = 26.997840172786177 →
    bridge_length = 150 →
    speed_kmph = 36 →
    train_length speed_kmph crossing_time bridge_length = 119.97840172786177 :=
by
  intros crossing_time bridge_length speed_kmph h1 h2 h3
  rw [h1, h2, h3]
  simp only [speed_kmph_to_mps, distance_crossed, train_length]
  sorry

end NUMINAMATH_GPT_train_length_is_correct_l116_11650


namespace NUMINAMATH_GPT_proportion_of_segments_l116_11622

theorem proportion_of_segments
  (a b c d : ℝ)
  (h1 : b = 3)
  (h2 : c = 4)
  (h3 : d = 6)
  (h4 : a / b = c / d) :
  a = 2 :=
by
  sorry

end NUMINAMATH_GPT_proportion_of_segments_l116_11622


namespace NUMINAMATH_GPT_gcd_of_128_144_480_450_l116_11652

theorem gcd_of_128_144_480_450 : Nat.gcd (Nat.gcd 128 144) (Nat.gcd 480 450) = 6 := 
by
  sorry

end NUMINAMATH_GPT_gcd_of_128_144_480_450_l116_11652


namespace NUMINAMATH_GPT_row_even_col_odd_contradiction_row_odd_col_even_contradiction_l116_11656

theorem row_even_col_odd_contradiction : 
  ¬ (∃ (M : Matrix (Fin 20) (Fin 15) ℕ), 
      (∀ r : Fin 20, ∃ i : Fin 15, M r i = 2) ∧ 
      (∀ c : Fin 15, ∀ j : Fin 20, M j c = 5)) := 
sorry

theorem row_odd_col_even_contradiction : 
  ¬ (∃ (M : Matrix (Fin 20) (Fin 15) ℕ), 
      (∀ r : Fin 20, ∀ i : Fin 15, M r i = 5) ∧ 
      (∀ c : Fin 15, ∃ j : Fin 20, M j c = 2)) := 
sorry

end NUMINAMATH_GPT_row_even_col_odd_contradiction_row_odd_col_even_contradiction_l116_11656


namespace NUMINAMATH_GPT_shale_mix_per_pound_is_5_l116_11679

noncomputable def cost_of_shale_mix_per_pound 
  (cost_limestone : ℝ) (cost_compound : ℝ) (weight_limestone : ℝ) (total_weight : ℝ) : ℝ :=
  let total_cost_limestone := weight_limestone * cost_limestone 
  let weight_shale := total_weight - weight_limestone
  let total_cost := total_weight * cost_compound
  let total_cost_shale := total_cost - total_cost_limestone
  total_cost_shale / weight_shale

theorem shale_mix_per_pound_is_5 :
  cost_of_shale_mix_per_pound 3 4.25 37.5 100 = 5 := 
by 
  sorry

end NUMINAMATH_GPT_shale_mix_per_pound_is_5_l116_11679


namespace NUMINAMATH_GPT_area_of_garden_l116_11640

-- Define the garden properties
variables {l w : ℕ}

-- Calculate length from the condition of walking length 30 times
def length_of_garden (total_distance : ℕ) (times : ℕ) := total_distance / times

-- Calculate perimeter from the condition of walking perimeter 12 times
def perimeter_of_garden (total_distance : ℕ) (times : ℕ) := total_distance / times

-- Define the proof statement
theorem area_of_garden (total_distance : ℕ) (times_length_walk : ℕ) (times_perimeter_walk : ℕ)
  (h1 : length_of_garden total_distance times_length_walk = l)
  (h2 : perimeter_of_garden total_distance (2 * times_perimeter_walk) = 2 * (l + w)) :
  l * w = 400 := 
sorry

end NUMINAMATH_GPT_area_of_garden_l116_11640


namespace NUMINAMATH_GPT_shifted_parabola_correct_l116_11662

-- Define original equation of parabola
def original_parabola (x : ℝ) : ℝ := 2 * x^2 - 1

-- Define shifted equation of parabola
def shifted_parabola (x : ℝ) : ℝ := 2 * (x + 1)^2 - 1

-- Proof statement: the expression of the new parabola after shifting 1 unit to the left
theorem shifted_parabola_correct :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x + 1) :=
by
  -- Proof is omitted, sorry
  sorry

end NUMINAMATH_GPT_shifted_parabola_correct_l116_11662


namespace NUMINAMATH_GPT_tim_coins_value_l116_11667

variable (d q : ℕ)

-- Given Conditions
def total_coins (d q : ℕ) : Prop := d + q = 18
def quarter_to_dime_relation (d q : ℕ) : Prop := q = d + 2

-- Prove the value of the coins
theorem tim_coins_value (d q : ℕ) (h1 : total_coins d q) (h2 : quarter_to_dime_relation d q) : 10 * d + 25 * q = 330 := by
  sorry

end NUMINAMATH_GPT_tim_coins_value_l116_11667


namespace NUMINAMATH_GPT_largest_prime_divisor_36_squared_plus_81_squared_l116_11636

-- Definitions of the key components in the problem
def a := 36
def b := 81
def expr := a^2 + b^2
def largest_prime_divisor (n : ℕ) : ℕ := sorry -- Assume this function can compute the largest prime divisor

-- Theorem stating the problem
theorem largest_prime_divisor_36_squared_plus_81_squared : largest_prime_divisor (36^2 + 81^2) = 53 := 
  sorry

end NUMINAMATH_GPT_largest_prime_divisor_36_squared_plus_81_squared_l116_11636


namespace NUMINAMATH_GPT_quadratic_single_intersection_l116_11653

theorem quadratic_single_intersection (m : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x + m = 0 → x^2 - 2 * x + m = (x-1)^2) :=
sorry

end NUMINAMATH_GPT_quadratic_single_intersection_l116_11653


namespace NUMINAMATH_GPT_john_initial_payment_l116_11631

-- Definitions based on the conditions from step a)
def cost_per_soda : ℕ := 2
def num_sodas : ℕ := 3
def change_received : ℕ := 14

-- Problem Statement: Prove that the total amount of money John paid initially is $20
theorem john_initial_payment :
  cost_per_soda * num_sodas + change_received = 20 := 
by
  sorry -- Proof steps are omitted as per instructions

end NUMINAMATH_GPT_john_initial_payment_l116_11631


namespace NUMINAMATH_GPT_ab_plus_2_l116_11669

theorem ab_plus_2 (a b : ℝ) (h : ∀ x : ℝ, (x - 3) * (3 * x + 7) = x^2 - 12 * x + 27 → x = a ∨ x = b) (ha : a ≠ b) :
  (a + 2) * (b + 2) = -30 :=
sorry

end NUMINAMATH_GPT_ab_plus_2_l116_11669


namespace NUMINAMATH_GPT_juan_european_stamps_total_cost_l116_11657

/-- Define the cost of European stamps collection for Juan -/
def total_cost_juan_stamps : ℝ := 
  -- Costs of stamps from the 1980s
  (15 * 0.07) + (11 * 0.06) + (14 * 0.08) +
  -- Costs of stamps from the 1990s
  (14 * 0.07) + (10 * 0.06) + (12 * 0.08)

/-- Prove that the total cost for European stamps from the 80s and 90s is $5.37 -/
theorem juan_european_stamps_total_cost : total_cost_juan_stamps = 5.37 :=
  by sorry

end NUMINAMATH_GPT_juan_european_stamps_total_cost_l116_11657


namespace NUMINAMATH_GPT_no_solution_range_has_solution_range_l116_11684

open Real

theorem no_solution_range (a : ℝ) : (∀ x, ¬ (|x - 4| + |3 - x| < a)) ↔ a ≤ 1 := 
sorry

theorem has_solution_range (a : ℝ) : (∃ x, |x - 4| + |3 - x| < a) ↔ 1 < a :=
sorry

end NUMINAMATH_GPT_no_solution_range_has_solution_range_l116_11684


namespace NUMINAMATH_GPT_proportional_increase_l116_11686

theorem proportional_increase (x y : ℝ) (h : 3 * x - 2 * y = 7) : y = (3 / 2) * x - 7 / 2 :=
by
  sorry

end NUMINAMATH_GPT_proportional_increase_l116_11686


namespace NUMINAMATH_GPT_large_bucket_capacity_l116_11698

variable (S L : ℕ)

theorem large_bucket_capacity (h1 : L = 2 * S + 3) (h2 : 2 * S + 5 * L = 63) : L = 11 :=
sorry

end NUMINAMATH_GPT_large_bucket_capacity_l116_11698


namespace NUMINAMATH_GPT_john_weekly_earnings_increase_l116_11605

theorem john_weekly_earnings_increase :
  let earnings_before := 60 + 100
  let earnings_after := 78 + 120
  let increase := earnings_after - earnings_before
  (increase / earnings_before : ℚ) * 100 = 23.75 :=
by
  -- Definitions
  let earnings_before := (60 : ℚ) + 100
  let earnings_after := (78 : ℚ) + 120
  let increase := earnings_after - earnings_before

  -- Calculation of percentage increase
  let percentage_increase : ℚ := (increase / earnings_before) * 100

  -- Expected result
  have expected_result : percentage_increase = 23.75 := by sorry
  exact expected_result

end NUMINAMATH_GPT_john_weekly_earnings_increase_l116_11605


namespace NUMINAMATH_GPT_average_of_remaining_five_l116_11625

open Nat Real

theorem average_of_remaining_five (avg9 avg4 : ℝ) (S S4 : ℝ) 
(h1 : avg9 = 18) (h2 : avg4 = 8) 
(h_sum9 : S = avg9 * 9) 
(h_sum4 : S4 = avg4 * 4) :
(S - S4) / 5 = 26 := by
  sorry

end NUMINAMATH_GPT_average_of_remaining_five_l116_11625


namespace NUMINAMATH_GPT_area_of_triangle_ABC_l116_11616

def point := (ℝ × ℝ)

def A : point := (0, 0)
def B : point := (1424233, 2848467)
def C : point := (1424234, 2848469)

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_triangle_ABC : triangle_area A B C = 0.50 := by
  sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_l116_11616


namespace NUMINAMATH_GPT_daily_rate_problem_l116_11618

noncomputable def daily_rate : ℝ := 126.19 -- Correct answer

theorem daily_rate_problem
  (days : ℕ := 14)
  (pet_fee : ℝ := 100)
  (service_fee_rate : ℝ := 0.20)
  (security_deposit : ℝ := 1110)
  (deposit_rate : ℝ := 0.50)
  (x : ℝ) : x = daily_rate :=
by
  have total_cost := days * x + pet_fee + service_fee_rate * (days * x)
  have total_cost_with_fees := days * x * (1 + service_fee_rate) + pet_fee
  have security_deposit_cost := deposit_rate * total_cost_with_fees
  have eq_security : security_deposit_cost = security_deposit := sorry
  sorry

end NUMINAMATH_GPT_daily_rate_problem_l116_11618


namespace NUMINAMATH_GPT_exists_ordering_no_arithmetic_progression_l116_11691

theorem exists_ordering_no_arithmetic_progression (m : ℕ) (hm : 0 < m) :
  ∃ (a : Fin (2^m) → ℕ), (∀ i j k : Fin (2^m), i < j → j < k → a j - a i ≠ a k - a j) := sorry

end NUMINAMATH_GPT_exists_ordering_no_arithmetic_progression_l116_11691


namespace NUMINAMATH_GPT_smallest_possible_N_l116_11644

theorem smallest_possible_N (p q r s t : ℕ) (h_pos : p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0 ∧ t > 0)
(h_sum : p + q + r + s + t = 2022) :
    ∃ N : ℕ, N = 506 ∧ N = max (p + q) (max (q + r) (max (r + s) (s + t))) :=
by
    sorry

end NUMINAMATH_GPT_smallest_possible_N_l116_11644


namespace NUMINAMATH_GPT_lexi_laps_l116_11627

theorem lexi_laps (total_distance lap_distance : ℝ) (h1 : total_distance = 3.25) (h2 : lap_distance = 0.25) :
  total_distance / lap_distance = 13 :=
by
  sorry

end NUMINAMATH_GPT_lexi_laps_l116_11627


namespace NUMINAMATH_GPT_third_generation_tail_length_l116_11608

theorem third_generation_tail_length (tail_length : ℕ → ℕ) (h0 : tail_length 0 = 16)
    (h_next : ∀ n, tail_length (n + 1) = tail_length n + (25 * tail_length n) / 100) :
    tail_length 2 = 25 :=
by
  sorry

end NUMINAMATH_GPT_third_generation_tail_length_l116_11608


namespace NUMINAMATH_GPT_number_of_students_scoring_above_90_l116_11663

theorem number_of_students_scoring_above_90
  (total_students : ℕ)
  (mean : ℝ)
  (variance : ℝ)
  (students_scoring_at_least_60 : ℕ)
  (h1 : total_students = 1200)
  (h2 : mean = 75)
  (h3 : ∃ (σ : ℝ), variance = σ^2)
  (h4 : students_scoring_at_least_60 = 960)
  : ∃ n, n = total_students - students_scoring_at_least_60 ∧ n = 240 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_students_scoring_above_90_l116_11663


namespace NUMINAMATH_GPT_no_real_solution_l116_11680

theorem no_real_solution (n : ℝ) : (∀ x : ℝ, (x+6)*(x-3) = n + 4*x → false) ↔ n < -73/4 := by
  sorry

end NUMINAMATH_GPT_no_real_solution_l116_11680


namespace NUMINAMATH_GPT_tenly_more_stuffed_animals_than_kenley_l116_11681

def mckenna_stuffed_animals := 34
def kenley_stuffed_animals := 2 * mckenna_stuffed_animals
def total_stuffed_animals_all := 175
def total_stuffed_animals_mckenna_kenley := mckenna_stuffed_animals + kenley_stuffed_animals
def tenly_stuffed_animals := total_stuffed_animals_all - total_stuffed_animals_mckenna_kenley
def stuffed_animals_difference := tenly_stuffed_animals - kenley_stuffed_animals

theorem tenly_more_stuffed_animals_than_kenley :
  stuffed_animals_difference = 5 := by
  sorry

end NUMINAMATH_GPT_tenly_more_stuffed_animals_than_kenley_l116_11681


namespace NUMINAMATH_GPT_simplify_expression_l116_11660

theorem simplify_expression (y : ℝ) : (3 * y^4)^5 = 243 * y^20 :=
sorry

end NUMINAMATH_GPT_simplify_expression_l116_11660


namespace NUMINAMATH_GPT_interest_rate_of_first_investment_l116_11661

theorem interest_rate_of_first_investment (x y : ℝ) (h1 : x + y = 2000) (h2 : y = 650) (h3 : 0.10 * x - 0.08 * y = 83) : (0.10 * x) / x = 0.10 := by
  sorry

end NUMINAMATH_GPT_interest_rate_of_first_investment_l116_11661


namespace NUMINAMATH_GPT_last_digit_of_1_div_3_pow_9_is_7_l116_11647

noncomputable def decimal_expansion_last_digit (n d : ℕ) : ℕ :=
  (n / d) % 10

theorem last_digit_of_1_div_3_pow_9_is_7 :
  decimal_expansion_last_digit 1 (3^9) = 7 :=
by
  sorry

end NUMINAMATH_GPT_last_digit_of_1_div_3_pow_9_is_7_l116_11647


namespace NUMINAMATH_GPT_solve_system_l116_11628

theorem solve_system :
  ∀ (x y z : ℝ),
  (x^2 - 23 * y - 25 * z = -681) →
  (y^2 - 21 * x - 21 * z = -419) →
  (z^2 - 19 * x - 21 * y = -313) →
  (x = 20 ∧ y = 22 ∧ z = 23) :=
by
  intros x y z h1 h2 h3
  sorry

end NUMINAMATH_GPT_solve_system_l116_11628


namespace NUMINAMATH_GPT_range_of_a_l116_11672

theorem range_of_a (a : ℝ) 
  (h1 : ∀ x : ℝ, (2 * a + 1) * x + a - 2 > (2 * a + 1) * 0 + a - 2)
  (h2 : a - 2 < 0) : -1 / 2 < a ∧ a < 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l116_11672


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_solution_l116_11655

theorem arithmetic_geometric_sequence_solution (u v : ℕ → ℝ) (a b u₀ : ℝ) :
  (∀ n, u (n + 1) = a * u n + b) ∧ (∀ n, v (n + 1) = a * v n + b) →
  u 0 = u₀ →
  v 0 = b / (1 - a) →
  ∀ n, u n = a ^ n * (u₀ - b / (1 - a)) + b / (1 - a) :=
by
  intros
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_solution_l116_11655


namespace NUMINAMATH_GPT_neg_p_sufficient_not_necessary_q_l116_11611

theorem neg_p_sufficient_not_necessary_q (p q : Prop) 
  (h₁ : p → ¬q) 
  (h₂ : ¬(¬q → p)) : (q → ¬p) ∧ ¬(¬p → q) :=
sorry

end NUMINAMATH_GPT_neg_p_sufficient_not_necessary_q_l116_11611


namespace NUMINAMATH_GPT_blue_more_than_white_l116_11685

theorem blue_more_than_white :
  ∃ (B R : ℕ), (B > 16) ∧ (R = 2 * B) ∧ (B + R + 16 = 100) ∧ (B - 16 = 12) :=
sorry

end NUMINAMATH_GPT_blue_more_than_white_l116_11685


namespace NUMINAMATH_GPT_evaluate_expression_l116_11671

theorem evaluate_expression :
  (2 ^ (-1 : ℤ) + 2 ^ (-2 : ℤ))⁻¹ = (4 / 3 : ℚ) := by
    sorry

end NUMINAMATH_GPT_evaluate_expression_l116_11671


namespace NUMINAMATH_GPT_final_expression_l116_11610

theorem final_expression (y : ℝ) : (3 * (1 / 2 * (12 * y + 3))) = 18 * y + 4.5 :=
by
  sorry

end NUMINAMATH_GPT_final_expression_l116_11610


namespace NUMINAMATH_GPT_bear_problem_l116_11666

-- Definitions of the variables
variables (W B Br : ℕ)

-- Given conditions
def condition1 : B = 2 * W := sorry
def condition2 : B = 60 := sorry
def condition3 : W + B + Br = 190 := sorry

-- The proof statement
theorem bear_problem : Br - B = 40 :=
by
  -- we would use the given conditions to prove this statement
  sorry

end NUMINAMATH_GPT_bear_problem_l116_11666


namespace NUMINAMATH_GPT_fourth_number_value_l116_11634

variable (A B C D E F : ℝ)

theorem fourth_number_value 
  (h1 : A + B + C + D + E + F = 180)
  (h2 : A + B + C + D = 100)
  (h3 : D + E + F = 105) : 
  D = 25 := 
by 
  sorry

end NUMINAMATH_GPT_fourth_number_value_l116_11634
