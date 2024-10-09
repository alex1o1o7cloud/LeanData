import Mathlib

namespace students_met_goal_l146_14628

def money_needed_per_student : ℕ := 450
def number_of_students : ℕ := 6
def collective_expenses : ℕ := 3000
def amount_raised_day1 : ℕ := 600
def amount_raised_day2 : ℕ := 900
def amount_raised_day3 : ℕ := 400
def days_remaining : ℕ := 4
def half_of_first_three_days : ℕ :=
  (amount_raised_day1 + amount_raised_day2 + amount_raised_day3) / 2

def total_needed : ℕ :=
  money_needed_per_student * number_of_students + collective_expenses
def total_raised : ℕ :=
  amount_raised_day1 + amount_raised_day2 + amount_raised_day3 + (half_of_first_three_days * days_remaining)

theorem students_met_goal : total_raised >= total_needed := by
  sorry

end students_met_goal_l146_14628


namespace football_team_matches_l146_14668

theorem football_team_matches (total_matches loses total_points: ℕ) 
  (points_win points_draw points_lose wins draws: ℕ)
  (h1: total_matches = 15)
  (h2: loses = 4)
  (h3: total_points = 29)
  (h4: points_win = 3)
  (h5: points_draw = 1)
  (h6: points_lose = 0)
  (h7: wins + draws + loses = total_matches)
  (h8: points_win * wins + points_draw * draws = total_points) :
  wins = 9 ∧ draws = 2 :=
sorry


end football_team_matches_l146_14668


namespace modular_inverse_expression_l146_14635

-- Definitions of the inverses as given in the conditions
def inv_7_mod_77 : ℤ := 11
def inv_13_mod_77 : ℤ := 6

-- The main theorem stating the equivalence
theorem modular_inverse_expression :
  (3 * inv_7_mod_77 + 9 * inv_13_mod_77) % 77 = 10 :=
by
  sorry

end modular_inverse_expression_l146_14635


namespace square_value_is_10000_l146_14642
noncomputable def squareValue : Real := 6400000 / 400 / 1.6

theorem square_value_is_10000 : squareValue = 10000 :=
  by
  -- The proof is based on the provided steps, which will be omitted here.
  sorry

end square_value_is_10000_l146_14642


namespace cos_pi_minus_alpha_l146_14687

theorem cos_pi_minus_alpha (α : ℝ) (hα : α > π ∧ α < 3 * π / 2) (h : Real.sin α = -5/13) :
  Real.cos (π - α) = 12 / 13 := 
by
  sorry

end cos_pi_minus_alpha_l146_14687


namespace isosceles_triangle_angle_l146_14614

theorem isosceles_triangle_angle {x : ℝ} (hx0 : 0 < x) (hx1 : x < 90) (hx2 : 2 * x = 180 / 7) : x = 180 / 7 :=
sorry

end isosceles_triangle_angle_l146_14614


namespace Jane_shopping_oranges_l146_14684

theorem Jane_shopping_oranges 
  (o a : ℕ)
  (h1 : a + o = 5)
  (h2 : 30 * a + 45 * o + 20 = n)
  (h3 : ∃ k : ℕ, n = 100 * k) : 
  o = 2 :=
by
  sorry

end Jane_shopping_oranges_l146_14684


namespace part1_part2_l146_14601

-- Define the function f(x) = |x - 1| + |x - 2|
def f (x : ℝ) : ℝ := abs (x - 1) + abs (x - 2)

-- Prove the statement about f(x) and the inequality
theorem part1 : { x : ℝ | (2 / 3) ≤ x ∧ x ≤ 4 } ⊆ { x : ℝ | f x ≤ x + 1 } :=
sorry

-- State k = 1 as the minimum value of f(x)
def k : ℝ := 1

-- Prove the non-existence of positive a and b satisfying the given conditions
theorem part2 : ¬ ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ 2 * a + b = k ∧ (1 / a + 2 / b = 4) :=
sorry

end part1_part2_l146_14601


namespace apples_harvested_l146_14624

theorem apples_harvested (weight_juice weight_restaurant weight_per_bag sales_price total_sales : ℤ) 
  (h1 : weight_juice = 90) 
  (h2 : weight_restaurant = 60) 
  (h3 : weight_per_bag = 5) 
  (h4 : sales_price = 8) 
  (h5 : total_sales = 408) : 
  (weight_juice + weight_restaurant + (total_sales / sales_price) * weight_per_bag = 405) :=
by
  sorry

end apples_harvested_l146_14624


namespace initial_oil_amounts_l146_14620

-- Definitions related to the problem
variables (A0 B0 C0 : ℝ)
variables (x : ℝ)

-- Conditions given in the problem
def bucketC_initial := C0 = 48
def transferA_to_B := x = 64 ∧ 64 = (2/3 * A0)
def transferB_to_C := x = 64 ∧ 64 = ((4/5 * (B0 + 1/3 * A0)) * (1/5 + 1))

-- Proof statement to show the solutions
theorem initial_oil_amounts (A0 B0 : ℝ) (C0 x : ℝ) 
  (h1 : bucketC_initial C0)
  (h2 : transferA_to_B A0 x)
  (h3 : transferB_to_C B0 A0 x) :
  A0 = 96 ∧ B0 = 48 :=
by 
  -- Placeholder for the proof
  sorry

end initial_oil_amounts_l146_14620


namespace conjugate_in_fourth_quadrant_l146_14697

def complex_conjugate (z : ℂ) : ℂ := ⟨z.re, -z.im⟩

-- Given complex number
def z : ℂ := ⟨5, 3⟩

-- Conjugate of z
def z_conjugate : ℂ := complex_conjugate z

-- Cartesian coordinates of the conjugate
def z_conjugate_coordinates : ℝ × ℝ := (z_conjugate.re, z_conjugate.im)

-- Definition of the Fourth Quadrant
def is_in_fourth_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

theorem conjugate_in_fourth_quadrant :
  is_in_fourth_quadrant z_conjugate_coordinates :=
by sorry

end conjugate_in_fourth_quadrant_l146_14697


namespace mr_william_land_percentage_l146_14690

-- Define the conditions
def farm_tax_percentage : ℝ := 0.5
def total_tax_collected : ℝ := 3840
def mr_william_tax : ℝ := 480

-- Theorem statement proving the question == answer
theorem mr_william_land_percentage : 
  (mr_william_tax / total_tax_collected) * 100 = 12.5 := 
by
  -- sorry is used to skip the proof
  sorry

end mr_william_land_percentage_l146_14690


namespace exists_integer_coordinates_l146_14670

theorem exists_integer_coordinates :
  ∃ (x y : ℤ), (x^2 + y^2) = 2 * 2017^2 + 2 * 2018^2 :=
by
  sorry

end exists_integer_coordinates_l146_14670


namespace total_sum_lent_l146_14643

noncomputable def interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem total_sum_lent 
  (x y : ℝ)
  (h1 : interest x (3 / 100) 5 = interest y (5 / 100) 3) 
  (h2 : y = 1332.5) : 
  x + y = 2665 :=
by
  -- We would continue the proof steps here.
  sorry

end total_sum_lent_l146_14643


namespace pieces_picked_by_olivia_l146_14618

-- Define the conditions
def picked_by_edward : ℕ := 3
def total_picked : ℕ := 19

-- Prove the number of pieces picked up by Olivia
theorem pieces_picked_by_olivia (O : ℕ) (h : O + picked_by_edward = total_picked) : O = 16 :=
by sorry

end pieces_picked_by_olivia_l146_14618


namespace penguin_permutations_correct_l146_14683

def num_permutations_of_multiset (total : ℕ) (freqs : List ℕ) : ℕ :=
  Nat.factorial total / (freqs.foldl (λ acc x => acc * Nat.factorial x) 1)

def penguin_permutations : ℕ := num_permutations_of_multiset 7 [2, 1, 1, 1, 1, 1]

theorem penguin_permutations_correct : penguin_permutations = 2520 := by
  sorry

end penguin_permutations_correct_l146_14683


namespace HCF_is_five_l146_14654

noncomputable def HCF_of_numbers (a b : ℕ) : ℕ := Nat.gcd a b

theorem HCF_is_five :
  ∃ (a b : ℕ),
    a + b = 55 ∧
    Nat.lcm a b = 120 ∧
    (1 / (a : ℝ) + 1 / (b : ℝ) = 0.09166666666666666) →
    HCF_of_numbers a b = 5 :=
by 
  sorry

end HCF_is_five_l146_14654


namespace part_a_part_b_l146_14691

-- Part (a): Prove that if 2^n - 1 divides m^2 + 9 for positive integers m and n, then n must be a power of 2.
theorem part_a (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : (2^n - 1) ∣ (m^2 + 9)) : ∃ k : ℕ, n = 2^k := 
sorry

-- Part (b): Prove that if n is a power of 2, then there exists a positive integer m such that 2^n - 1 divides m^2 + 9.
theorem part_b (n : ℕ) (hn : ∃ k : ℕ, n = 2^k) : ∃ m : ℕ, 0 < m ∧ (2^n - 1) ∣ (m^2 + 9) := 
sorry

end part_a_part_b_l146_14691


namespace min_value_exponential_sub_l146_14678

theorem min_value_exponential_sub (x y : ℝ) (hx : x > 0) (hy : y > 0)
    (h : x + 2 * y = x * y) : ∃ y₀ > 0, ∀ y > 1, e^y - 8 / x ≥ e :=
by
  sorry

end min_value_exponential_sub_l146_14678


namespace increasing_function_range_l146_14689

theorem increasing_function_range (a : ℝ) (f : ℝ → ℝ)
  (h : ∀ x : ℝ, f x = x^3 - a * x - 1) :
  (∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2) ↔ a ≤ 0 :=
sorry

end increasing_function_range_l146_14689


namespace problem_curves_l146_14666

theorem problem_curves (x y : ℝ) : 
  ((x * (x^2 + y^2 - 4) = 0 → (x = 0 ∨ x^2 + y^2 = 4)) ∧
  (x^2 + (x^2 + y^2 - 4)^2 = 0 → ((x = 0 ∧ y = -2) ∨ (x = 0 ∧ y = 2)))) :=
by
  sorry -- proof to be filled in later

end problem_curves_l146_14666


namespace sqrt_64_eq_pm_8_l146_14692

theorem sqrt_64_eq_pm_8 : ∃x : ℤ, x^2 = 64 ∧ (x = 8 ∨ x = -8) :=
by
  sorry

end sqrt_64_eq_pm_8_l146_14692


namespace eighth_term_of_arithmetic_sequence_l146_14632

theorem eighth_term_of_arithmetic_sequence :
  ∀ (a : ℕ → ℤ),
  (a 1 = 11) →
  (a 2 = 8) →
  (a 3 = 5) →
  (∃ (d : ℤ), ∀ n, a (n + 1) = a n + d) →
  a 8 = -10 :=
by
  intros a h1 h2 h3 arith
  sorry

end eighth_term_of_arithmetic_sequence_l146_14632


namespace goose_eggs_at_pond_l146_14608

noncomputable def total_goose_eggs (E : ℝ) : Prop :=
  (5 / 12) * (5 / 16) * (5 / 9) * (3 / 7) * E = 84

theorem goose_eggs_at_pond : 
  ∃ E : ℝ, total_goose_eggs E ∧ E = 678 :=
by
  use 678
  dsimp [total_goose_eggs]
  sorry

end goose_eggs_at_pond_l146_14608


namespace inversely_varies_y_l146_14639

theorem inversely_varies_y (x y : ℕ) (k : ℕ) (h₁ : 7 * y = k / x^3) (h₂ : y = 8) (h₃ : x = 2) : 
  y = 1 :=
by
  sorry

end inversely_varies_y_l146_14639


namespace second_cube_surface_area_l146_14652

theorem second_cube_surface_area (s : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) (sa : ℝ) :
  6 * s^2 = 54 →
  a = s →
  b = a * (1 / 2) →
  c * Real.sqrt 3 = 2 * b →
  sa = 6 * c^2 →
  sa = 18 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end second_cube_surface_area_l146_14652


namespace negation_of_proposition_l146_14671

noncomputable def negation_proposition (f : ℝ → Prop) : Prop :=
  ∃ x : ℝ, x ≥ 0 ∧ ¬ f x

theorem negation_of_proposition :
  (∀ x : ℝ, x ≥ 0 → x^2 + x - 1 > 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^2 + x - 1 ≤ 0) :=
by
  sorry

end negation_of_proposition_l146_14671


namespace necessary_and_sufficient_condition_for_absolute_inequality_l146_14645

theorem necessary_and_sufficient_condition_for_absolute_inequality (a : ℝ) :
  (a < 3) ↔ (∀ x : ℝ, |x + 2| + |x - 1| > a) :=
sorry

end necessary_and_sufficient_condition_for_absolute_inequality_l146_14645


namespace number_of_classes_l146_14685

theorem number_of_classes (x : ℕ) (h : x * (x - 1) / 2 = 45) : x = 10 :=
by {
  sorry -- Proof goes here
}

end number_of_classes_l146_14685


namespace min_reciprocal_sum_l146_14617

theorem min_reciprocal_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (hS : S 2019 = 4038) 
  (h_seq : ∀ n, S n = (n * (a 1 + a n)) / 2) :
  ∃ m, m = 4 ∧ (∀ i, i = 9 → ∀ j, j = 2011 → 
  a i + a j = 4 ∧ m = min (1 / a i + 9 / a j) 4) :=
by sorry

end min_reciprocal_sum_l146_14617


namespace probability_correct_l146_14686

noncomputable def probability_of_getting_number_greater_than_4 : ℚ :=
  let favorable_outcomes := 2
  let total_outcomes := 6
  favorable_outcomes / total_outcomes

theorem probability_correct :
  probability_of_getting_number_greater_than_4 = 1 / 3 := by sorry

end probability_correct_l146_14686


namespace right_triangle_third_side_l146_14693

theorem right_triangle_third_side (a b c : ℝ) (ha : a = 8) (hb : b = 6) (h_right_triangle : a^2 + b^2 = c^2) :
  c = 10 :=
by
  sorry

end right_triangle_third_side_l146_14693


namespace expression_in_scientific_notation_l146_14621

-- Conditions
def billion : ℝ := 10^9
def a : ℝ := 20.8

-- Statement
theorem expression_in_scientific_notation : a * billion = 2.08 * 10^10 := by
  sorry

end expression_in_scientific_notation_l146_14621


namespace percentage_failed_in_hindi_l146_14638

theorem percentage_failed_in_hindi (P_E : ℝ) (P_H_and_E : ℝ) (P_P : ℝ) (H : ℝ) : 
  P_E = 0.5 ∧ P_H_and_E = 0.25 ∧ P_P = 0.5 → H = 0.25 :=
by
  sorry

end percentage_failed_in_hindi_l146_14638


namespace pies_calculation_l146_14636

-- Definition: Number of ingredients per pie
def ingredients_per_pie (apples total_apples pies : ℤ) : ℤ := total_apples / pies

-- Definition: Number of pies that can be made with available ingredients 
def pies_from_ingredients (ingredient_amount per_pie : ℤ) : ℤ := ingredient_amount / per_pie

-- Hypothesis
theorem pies_calculation (apples_per_pie pears_per_pie apples pears pies : ℤ) 
  (h1: ingredients_per_pie apples 12 pies = 4)
  (h2: ingredients_per_pie apples 6 pies = 2)
  (h3: pies_from_ingredients 36 4 = 9)
  (h4: pies_from_ingredients 18 2 = 9): 
  pies = 9 := 
sorry

end pies_calculation_l146_14636


namespace find_initial_tomatoes_l146_14694

-- Define the initial number of tomatoes
def initial_tomatoes (T : ℕ) : Prop :=
  T + 77 - 172 = 80

-- Theorem statement to prove the initial number of tomatoes is 175
theorem find_initial_tomatoes : ∃ T : ℕ, initial_tomatoes T ∧ T = 175 :=
sorry

end find_initial_tomatoes_l146_14694


namespace min_value_expr_l146_14661

theorem min_value_expr (a : ℝ) (ha : a > 0) : 
  ∃ (x : ℝ), x = (a-1)*(4*a-1)/a ∧ ∀ (y : ℝ), y = (a-1)*(4*a-1)/a → y ≥ -1 :=
by sorry

end min_value_expr_l146_14661


namespace minimum_four_sum_multiple_of_four_l146_14660

theorem minimum_four_sum_multiple_of_four (n : ℕ) (h : n = 7) (s : Fin n → ℤ) :
  ∃ (a b c d : Fin n), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  (s a + s b + s c + s d) % 4 = 0 := 
by
  -- Proof goes here
  sorry

end minimum_four_sum_multiple_of_four_l146_14660


namespace determine_digits_l146_14619

theorem determine_digits :
  ∃ (A B C D : ℕ), 
    1000 ≤ 1000 * A + 100 * B + 10 * C + D ∧ 
    1000 * A + 100 * B + 10 * C + D ≤ 9999 ∧ 
    1000 ≤ 1000 * C + 100 * B + 10 * A + D ∧ 
    1000 * C + 100 * B + 10 * A + D ≤ 9999 ∧ 
    (1000 * A + 100 * B + 10 * C + D) * D = 1000 * C + 100 * B + 10 * A + D ∧ 
    A = 2 ∧ B = 1 ∧ C = 7 ∧ D = 8 :=
by
  sorry

end determine_digits_l146_14619


namespace sale_in_third_month_l146_14662

theorem sale_in_third_month (sale1 sale2 sale4 sale5 sale6 avg_sale : ℝ) (n_months : ℝ) (sale3 : ℝ):
  sale1 = 5400 →
  sale2 = 9000 →
  sale4 = 7200 →
  sale5 = 4500 →
  sale6 = 1200 →
  avg_sale = 5600 →
  n_months = 6 →
  (n_months * avg_sale) - (sale1 + sale2 + sale4 + sale5 + sale6) = sale3 →
  sale3 = 6300 :=
by
  intros
  sorry

end sale_in_third_month_l146_14662


namespace find_Δ_l146_14634

-- Define the constants and conditions
variables (Δ p : ℕ)
axiom condition1 : Δ + p = 84
axiom condition2 : (Δ + p) + p = 153

-- State the theorem
theorem find_Δ : Δ = 15 :=
by
  sorry

end find_Δ_l146_14634


namespace log10_two_bounds_l146_14616

theorem log10_two_bounds
  (h1 : 10 ^ 3 = 1000)
  (h2 : 10 ^ 4 = 10000)
  (h3 : 2 ^ 10 = 1024)
  (h4 : 2 ^ 12 = 4096) :
  1 / 4 < Real.log 2 / Real.log 10 ∧ Real.log 2 / Real.log 10 < 0.4 := 
sorry

end log10_two_bounds_l146_14616


namespace evaluate_F_of_4_and_f_of_5_l146_14658

def f (a : ℤ) : ℤ := 2 * a - 2
def F (a b : ℤ) : ℤ := b^2 + a + 1

theorem evaluate_F_of_4_and_f_of_5 : F 4 (f 5) = 69 := by
  -- Definitions and intermediate steps are not included in the statement, proof is omitted.
  sorry

end evaluate_F_of_4_and_f_of_5_l146_14658


namespace simplify_vectors_l146_14657

variable (α : Type*) [AddCommGroup α]

variables (CE AC DE AD : α)

theorem simplify_vectors : CE + AC - DE - AD = (0 : α) := 
by sorry

end simplify_vectors_l146_14657


namespace additional_cars_needed_to_make_multiple_of_8_l146_14613

theorem additional_cars_needed_to_make_multiple_of_8 (current_cars : ℕ) (rows_of_cars : ℕ) (next_multiple : ℕ)
  (h1 : current_cars = 37)
  (h2 : rows_of_cars = 8)
  (h3 : next_multiple = 40)
  (h4 : next_multiple ≥ current_cars)
  (h5 : next_multiple % rows_of_cars = 0) :
  (next_multiple - current_cars) = 3 :=
by { sorry }

end additional_cars_needed_to_make_multiple_of_8_l146_14613


namespace remainder_9_minus_n_plus_n_plus_5_mod_8_l146_14631

theorem remainder_9_minus_n_plus_n_plus_5_mod_8 (n : ℤ) : 
  ((9 - n) + (n + 5)) % 8 = 6 := by
  sorry

end remainder_9_minus_n_plus_n_plus_5_mod_8_l146_14631


namespace cos_double_angle_l146_14659

theorem cos_double_angle (a : ℝ) (h : Real.sin a = 1 / 3) : Real.cos (2 * a) = 7 / 9 :=
by
  sorry

end cos_double_angle_l146_14659


namespace ratio_first_to_second_l146_14611

theorem ratio_first_to_second (A B C : ℕ) (h1 : A + B + C = 98) (h2 : B = 30) (h3 : B / C = 5 / 8) : A / B = 2 / 3 :=
sorry

end ratio_first_to_second_l146_14611


namespace jose_land_division_l146_14605

/-- Let the total land Jose bought be 20000 square meters. Let Jose divide this land equally among himself and his four siblings. Prove that the land Jose will have after dividing it is 4000 square meters. -/
theorem jose_land_division : 
  let total_land := 20000
  let numberOfPeople := 5
  total_land / numberOfPeople = 4000 := by
sorry

end jose_land_division_l146_14605


namespace total_bricks_used_l146_14625

-- Definitions for conditions
def num_courses_per_wall : Nat := 10
def num_bricks_per_course : Nat := 20
def num_complete_walls : Nat := 5
def incomplete_wall_missing_courses : Nat := 3

-- Lean statement to prove the mathematically equivalent problem
theorem total_bricks_used : 
  (num_complete_walls * (num_courses_per_wall * num_bricks_per_course) + 
  ((num_courses_per_wall - incomplete_wall_missing_courses) * num_bricks_per_course)) = 1140 :=
by
  sorry

end total_bricks_used_l146_14625


namespace tire_cost_l146_14630

theorem tire_cost (total_cost : ℝ) (num_tires : ℕ)
    (h1 : num_tires = 8) (h2 : total_cost = 4) : 
    total_cost / num_tires = 0.50 := 
by
  sorry

end tire_cost_l146_14630


namespace arithmetic_sequence_common_difference_l146_14669

   variable (a_n : ℕ → ℝ)
   variable (a_5 : ℝ := 13)
   variable (S_5 : ℝ := 35)
   variable (d : ℝ)

   theorem arithmetic_sequence_common_difference {a_1 : ℝ} :
     (a_1 + 4 * d = a_5) ∧ (5 * a_1 + 10 * d = S_5) → d = 3 :=
   by
     sorry
   
end arithmetic_sequence_common_difference_l146_14669


namespace Christina_weekly_distance_l146_14604

/-- 
Prove that Christina covered 74 kilometers that week given the following conditions:
1. Christina walks 7km to school every day from Monday to Friday.
2. She returns home covering the same distance each day.
3. Last Friday, she had to pass by her friend, which is another 2km away from the school in the opposite direction from home.
-/
theorem Christina_weekly_distance : 
  let distance_to_school := 7
  let days_school := 5
  let extra_distance_Friday := 2
  let daily_distance := 2 * distance_to_school
  let total_distance_from_Monday_to_Thursday := 4 * daily_distance
  let distance_on_Friday := daily_distance + 2 * extra_distance_Friday
  total_distance_from_Monday_to_Thursday + distance_on_Friday = 74 := 
by
  sorry

end Christina_weekly_distance_l146_14604


namespace updated_mean_of_decremented_observations_l146_14699

theorem updated_mean_of_decremented_observations (mean : ℝ) (n : ℕ) (decrement : ℝ) 
  (h_mean : mean = 200) (h_n : n = 50) (h_decrement : decrement = 47) : 
  (mean * n - decrement * n) / n = 153 := 
by 
  sorry

end updated_mean_of_decremented_observations_l146_14699


namespace angle_ABD_l146_14664

theorem angle_ABD (A B C D E F : Type)
  (quadrilateral : Prop)
  (angle_ABC : ℝ)
  (angle_BDE : ℝ)
  (angle_BDF : ℝ)
  (h1 : quadrilateral)
  (h2 : angle_ABC = 120)
  (h3 : angle_BDE = 30)
  (h4 : angle_BDF = 28) :
  (180 - angle_ABC = 60) :=
by
  sorry

end angle_ABD_l146_14664


namespace melted_ice_cream_depth_l146_14696

theorem melted_ice_cream_depth
  (r_sphere : ℝ) (r_cylinder : ℝ) (V_sphere : ℝ) (V_cylinder : ℝ)
  (h : ℝ)
  (hr_sphere : r_sphere = 3)
  (hr_cylinder : r_cylinder = 10)
  (hV_sphere : V_sphere = 4 / 3 * Real.pi * r_sphere^3)
  (hV_cylinder : V_cylinder = Real.pi * r_cylinder^2 * h)
  (volume_conservation : V_sphere = V_cylinder) :
  h = 9 / 25 :=
by
  sorry

end melted_ice_cream_depth_l146_14696


namespace find_k_l146_14688

theorem find_k
  (AB AC : ℝ)
  (k : ℝ)
  (h1 : AB = AC)
  (h2 : AB = 8)
  (h3 : AC = 5 - k) : k = -3 :=
by
  sorry

end find_k_l146_14688


namespace sequence_sum_l146_14610

theorem sequence_sum (S : ℕ → ℤ) (a : ℕ → ℤ) (hS : ∀ n : ℕ, S (n + 1) = (n + 1) * (n + 1) - 1)
  (ha : ∀ n : ℕ, a (n + 1) = S (n + 1) - S n) :
  a 1 + a 3 + a 5 + a 7 + a 9 = 44 :=
by
  sorry

end sequence_sum_l146_14610


namespace sum_of_squares_of_medians_triangle_13_14_15_l146_14612

noncomputable def sum_of_squares_of_medians (a b c : ℝ) : ℝ :=
  (3 / 4) * (a^2 + b^2 + c^2)

theorem sum_of_squares_of_medians_triangle_13_14_15 :
  sum_of_squares_of_medians 13 14 15 = 442.5 :=
by
  -- By calculation using the definition of sum_of_squares_of_medians
  -- and substituting the given side lengths.
  -- Detailed proof steps are omitted
  sorry

end sum_of_squares_of_medians_triangle_13_14_15_l146_14612


namespace total_cages_used_l146_14606

def num_puppies : Nat := 45
def num_adult_dogs : Nat := 30
def num_kittens : Nat := 25

def puppies_sold : Nat := 39
def adult_dogs_sold : Nat := 15
def kittens_sold : Nat := 10

def cage_capacity_puppies : Nat := 3
def cage_capacity_adult_dogs : Nat := 2
def cage_capacity_kittens : Nat := 2

def remaining_puppies : Nat := num_puppies - puppies_sold
def remaining_adult_dogs : Nat := num_adult_dogs - adult_dogs_sold
def remaining_kittens : Nat := num_kittens - kittens_sold

def cages_for_puppies : Nat := (remaining_puppies + cage_capacity_puppies - 1) / cage_capacity_puppies
def cages_for_adult_dogs : Nat := (remaining_adult_dogs + cage_capacity_adult_dogs - 1) / cage_capacity_adult_dogs
def cages_for_kittens : Nat := (remaining_kittens + cage_capacity_kittens - 1) / cage_capacity_kittens

def total_cages : Nat := cages_for_puppies + cages_for_adult_dogs + cages_for_kittens

-- Theorem stating the final goal
theorem total_cages_used : total_cages = 18 := by
  sorry

end total_cages_used_l146_14606


namespace perfect_square_trinomial_m_l146_14698

theorem perfect_square_trinomial_m (m : ℤ) : (∃ (a : ℤ), (x : ℝ) → x^2 + m * x + 9 = (x + a)^2) → (m = 6 ∨ m = -6) :=
sorry

end perfect_square_trinomial_m_l146_14698


namespace dane_daughters_initial_flowers_l146_14672

theorem dane_daughters_initial_flowers :
  (exists (x y : ℕ), x = y ∧ 5 * 4 = 20 ∧ x + y = 30) →
  (exists f : ℕ, f = 5 ∧ 10 = 30 - 20 + 10 ∧ x = f * 2) :=
by
  -- Lean proof needs to go here
  sorry

end dane_daughters_initial_flowers_l146_14672


namespace enrique_commission_l146_14615

def commission_earned (suits_sold: ℕ) (suit_price: ℝ) (shirts_sold: ℕ) (shirt_price: ℝ) 
                      (loafers_sold: ℕ) (loafers_price: ℝ) (commission_rate: ℝ) : ℝ :=
  let total_sales := (suits_sold * suit_price) + (shirts_sold * shirt_price) + (loafers_sold * loafers_price)
  total_sales * commission_rate

theorem enrique_commission :
  commission_earned 2 700 6 50 2 150 0.15 = 300 := by
  sorry

end enrique_commission_l146_14615


namespace part_a_part_b_l146_14682

-- Part (a)
theorem part_a
  (initial_deposit : ℝ)
  (initial_exchange_rate : ℝ)
  (annual_return_rate : ℝ)
  (final_exchange_rate : ℝ)
  (conversion_fee_rate : ℝ)
  (broker_commission_rate : ℝ) :
  initial_deposit = 12000 →
  initial_exchange_rate = 60 →
  annual_return_rate = 0.12 →
  final_exchange_rate = 80 →
  conversion_fee_rate = 0.04 →
  broker_commission_rate = 0.25 →
  let deposit_in_dollars := 12000 / 60
  let profit_in_dollars := deposit_in_dollars * 0.12
  let total_in_dollars := deposit_in_dollars + profit_in_dollars
  let broker_commission := profit_in_dollars * 0.25
  let amount_before_conversion := total_in_dollars - broker_commission
  let amount_in_rubles := amount_before_conversion * 80
  let conversion_fee := amount_in_rubles * 0.04
  let final_amount := amount_in_rubles - conversion_fee
  final_amount = 16742.4 := sorry

-- Part (b)
theorem part_b
  (initial_deposit : ℝ)
  (final_amount : ℝ) :
  initial_deposit = 12000 →
  final_amount = 16742.4 →
  let effective_return := (16742.4 / 12000) - 1
  effective_return * 100 = 39.52 := sorry

end part_a_part_b_l146_14682


namespace g_inv_eq_l146_14603

def g (x : ℝ) : ℝ := 2 * x ^ 2 + 3 * x - 5

theorem g_inv_eq (x : ℝ) (g_inv : ℝ → ℝ) (h_inv : ∀ y, g (g_inv y) = y ∧ g_inv (g y) = y) :
  (x = ( -1 + Real.sqrt 11 ) / 2) ∨ (x = ( -1 - Real.sqrt 11 ) / 2) :=
by
  -- proof omitted
  sorry

end g_inv_eq_l146_14603


namespace circle_radius_one_l146_14629

-- Define the circle equation as a hypothesis
def circle_equation (x y : ℝ) : Prop :=
  16 * x^2 + 32 * x + 16 * y^2 - 48 * y + 68 = 0

-- The goal is to prove the radius of the circle defined above
theorem circle_radius_one :
  ∃ r : ℝ, r = 1 ∧ ∀ x y : ℝ, circle_equation x y → (x + 1)^2 + (y - 1.5)^2 = r^2 :=
by
  sorry

end circle_radius_one_l146_14629


namespace min_value_of_expression_l146_14675

theorem min_value_of_expression {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + b = 1) : 
  (1 / a) + (2 / b) >= 8 :=
by
  sorry

end min_value_of_expression_l146_14675


namespace frogs_moving_l146_14623

theorem frogs_moving (initial_frogs tadpoles mature_frogs pond_capacity frogs_to_move : ℕ)
  (h1 : initial_frogs = 5)
  (h2 : tadpoles = 3 * initial_frogs)
  (h3 : mature_frogs = (2 * tadpoles) / 3)
  (h4 : pond_capacity = 8)
  (h5 : frogs_to_move = (initial_frogs + mature_frogs) - pond_capacity) :
  frogs_to_move = 7 :=
by {
  sorry
}

end frogs_moving_l146_14623


namespace total_heads_is_46_l146_14609

noncomputable def total_heads (hens cows : ℕ) : ℕ :=
  hens + cows

def num_feet_hens (num_hens : ℕ) : ℕ :=
  2 * num_hens

def num_cows (total_feet feet_hens_per_cow feet_cow_per_cow : ℕ) : ℕ :=
  (total_feet - feet_hens_per_cow) / feet_cow_per_cow

theorem total_heads_is_46 (num_hens : ℕ) (total_feet : ℕ)
  (hen_feet cow_feet hen_head cow_head : ℕ)
  (num_heads : ℕ) :
  num_hens = 24 →
  total_feet = 136 →
  hen_feet = 2 →
  cow_feet = 4 →
  hen_head = 1 →
  cow_head = 1 →
  num_heads = total_heads num_hens (num_cows total_feet (num_feet_hens num_hens) cow_feet) →
  num_heads = 46 :=
by
  intros
  sorry

end total_heads_is_46_l146_14609


namespace expression_value_l146_14681

theorem expression_value (x y : ℝ) (h : x + y = -1) : x^4 + 5 * x^3 * y + x^2 * y + 8 * x^2 * y^2 + x * y^2 + 5 * x * y^3 + y^4 = 1 :=
by
  sorry

end expression_value_l146_14681


namespace seating_arrangement_count_l146_14665

-- Define the conditions.
def chairs : ℕ := 7
def people : ℕ := 5
def end_chairs : ℕ := 3

-- Define the main theorem to prove the number of arrangements.
theorem seating_arrangement_count :
  (end_chairs * 2) * (6 * 5 * 4 * 3) = 2160 := by
  sorry

end seating_arrangement_count_l146_14665


namespace afternoon_shells_eq_l146_14674

def morning_shells : ℕ := 292
def total_shells : ℕ := 616

theorem afternoon_shells_eq :
  total_shells - morning_shells = 324 := by
  sorry

end afternoon_shells_eq_l146_14674


namespace range_of_a3_plus_a9_l146_14667

variable {a_n : ℕ → ℝ}

-- Given condition: in a geometric sequence, a4 * a8 = 9
def geom_seq_condition (a_n : ℕ → ℝ) : Prop :=
  a_n 4 * a_n 8 = 9

-- Theorem statement
theorem range_of_a3_plus_a9 (a_n : ℕ → ℝ) (h : geom_seq_condition a_n) :
  ∃ x y, (x + y = a_n 3 + a_n 9) ∧ (x ≥ 0 ∧ y ≥ 0 ∧ x + y ≥ 6) ∨ (x ≤ 0 ∧ y ≤ 0 ∧ x + y ≤ -6) ∨ (x = 0 ∧ y = 0 ∧ a_n 3 + a_n 9 ∈ (Set.Ici 6 ∪ Set.Iic (-6))) :=
sorry

end range_of_a3_plus_a9_l146_14667


namespace find_third_number_l146_14649

theorem find_third_number :
  let total_sum := 121526
  let first_addend := 88888
  let second_addend := 1111
  (total_sum = first_addend + second_addend + 31527) :=
by
  sorry

end find_third_number_l146_14649


namespace walnuts_left_in_burrow_l146_14653

-- Definitions of conditions
def boy_gathers : ℕ := 15
def originally_in_burrow : ℕ := 25
def boy_drops : ℕ := 3
def boy_hides : ℕ := 5
def girl_brings : ℕ := 12
def girl_eats : ℕ := 4
def girl_gives_away : ℕ := 3
def girl_loses : ℕ := 2

-- Theorem statement
theorem walnuts_left_in_burrow : 
  originally_in_burrow + (boy_gathers - boy_drops - boy_hides) + 
  (girl_brings - girl_eats - girl_gives_away - girl_loses) = 35 := 
sorry

end walnuts_left_in_burrow_l146_14653


namespace imaginary_part_of_z_is_2_l146_14641

noncomputable def z : ℂ := (3 * Complex.I + 1) / (1 - Complex.I)

theorem imaginary_part_of_z_is_2 : z.im = 2 := 
by 
  -- proof goes here
  sorry

end imaginary_part_of_z_is_2_l146_14641


namespace boat_speed_in_still_water_l146_14655

theorem boat_speed_in_still_water
  (v c : ℝ)
  (h1 : v + c = 10)
  (h2 : v - c = 4) :
  v = 7 :=
by
  sorry

end boat_speed_in_still_water_l146_14655


namespace quadratic_inequality_solution_set_l146_14680

theorem quadratic_inequality_solution_set (a b : ℝ) (h : ∀ x, 1 < x ∧ x < 3 → x^2 < ax + b) : b^a = 81 :=
sorry

end quadratic_inequality_solution_set_l146_14680


namespace largest_possible_percent_error_l146_14646

theorem largest_possible_percent_error 
  (r : ℝ) (delta : ℝ) (h_r : r = 15) (h_delta : delta = 0.1) : 
  ∃(error : ℝ), error = 0.21 :=
by
  -- The proof would go here
  sorry

end largest_possible_percent_error_l146_14646


namespace border_area_l146_14676

theorem border_area (h_photo : ℕ) (w_photo : ℕ) (border : ℕ) (h : h_photo = 8) (w : w_photo = 10) (b : border = 2) :
  (2 * (border + h_photo) * (border + w_photo) - h_photo * w_photo) = 88 :=
by
  rw [h, w, b]
  sorry

end border_area_l146_14676


namespace dividend_calculation_l146_14651

theorem dividend_calculation (q d r x : ℝ) 
  (hq : q = -427.86) (hd : d = 52.7) (hr : r = -14.5)
  (hx : x = q * d + r) : 
  x = -22571.002 :=
by 
  sorry

end dividend_calculation_l146_14651


namespace students_paid_half_l146_14622

theorem students_paid_half (F H : ℕ) 
  (h1 : F + H = 25)
  (h2 : 50 * F + 25 * H = 1150) : 
  H = 4 := by
  sorry

end students_paid_half_l146_14622


namespace example_solution_l146_14673

variable (x y θ : Real)
variable (h1 : 0 < x) (h2 : 0 < y)
variable (h3 : θ ∈ Set.Ioo (Real.pi / 4) (Real.pi / 2))
variable (h4 : Real.sin θ / x = Real.cos θ / y)
variable (h5 : Real.cos θ ^ 2 / x ^ 2 + Real.sin θ ^ 2 / y ^ 2 = 10 / (3 * (x ^ 2 + y ^ 2)))

theorem example_solution : x / y = Real.sqrt 3 :=
by
  sorry

end example_solution_l146_14673


namespace man_rowing_speed_l146_14663

noncomputable def rowing_speed_in_still_water : ℝ :=
  let distance := 0.1   -- kilometers
  let time := 20 / 3600 -- hours
  let current_speed := 3 -- km/hr
  let downstream_speed := distance / time
  downstream_speed - current_speed

theorem man_rowing_speed :
  rowing_speed_in_still_water = 15 :=
  by
    -- Proof comes here
    sorry

end man_rowing_speed_l146_14663


namespace avg_of_xyz_l146_14627

-- Define the given condition
def given_condition (x y z : ℝ) := 
  (5 / 2) * (x + y + z) = 20

-- Define the question (and the proof target) using the given conditions.
theorem avg_of_xyz (x y z : ℝ) (h : given_condition x y z) : 
  (x + y + z) / 3 = 8 / 3 :=
sorry

end avg_of_xyz_l146_14627


namespace last_three_digits_of_7_pow_103_l146_14626

theorem last_three_digits_of_7_pow_103 : (7^103) % 1000 = 614 := by
  sorry

end last_three_digits_of_7_pow_103_l146_14626


namespace determine_b_eq_l146_14695

theorem determine_b_eq (b : ℝ) : (∃! (x : ℝ), |x^2 + 3 * b * x + 4 * b| ≤ 3) ↔ b = 4 / 3 ∨ b = 1 := 
by sorry

end determine_b_eq_l146_14695


namespace percentage_increase_l146_14644

theorem percentage_increase (initial final : ℝ) (h_initial : initial = 200) (h_final : final = 250) :
  ((final - initial) / initial) * 100 = 25 := 
sorry

end percentage_increase_l146_14644


namespace helens_mother_brought_101_l146_14650

-- Define the conditions
def total_hotdogs : ℕ := 480
def dylan_mother_hotdogs : ℕ := 379
def helens_mother_hotdogs := total_hotdogs - dylan_mother_hotdogs

-- Theorem statement: Prove that the number of hotdogs Helen's mother brought is 101
theorem helens_mother_brought_101 : helens_mother_hotdogs = 101 :=
by
  sorry

end helens_mother_brought_101_l146_14650


namespace exists_x_y_not_divisible_by_3_l146_14633

theorem exists_x_y_not_divisible_by_3 (k : ℕ) (hk : 0 < k) :
  ∃ (x y : ℤ), (x^2 + 2 * y^2 = 3^k) ∧ (¬ (x % 3 = 0)) ∧ (¬ (y % 3 = 0)) :=
sorry

end exists_x_y_not_divisible_by_3_l146_14633


namespace probability_same_color_is_correct_l146_14648

-- Define the total number of each color marbles
def red_marbles : ℕ := 5
def white_marbles : ℕ := 6
def blue_marbles : ℕ := 7
def green_marbles : ℕ := 4

-- Define the total number of marbles
def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles + green_marbles

-- Define the probability calculation function
def probability_all_same_color : ℚ :=
  (red_marbles * (red_marbles - 1) * (red_marbles - 2) * (red_marbles - 3) / (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3))) +
  (white_marbles * (white_marbles - 1) * (white_marbles - 2) * (white_marbles - 3) / (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3))) +
  (blue_marbles * (blue_marbles - 1) * (blue_marbles - 2) * (blue_marbles - 3) / (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3))) +
  (green_marbles * (green_marbles - 1) * (green_marbles - 2) * (green_marbles - 3) / (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3)))

-- Define the theorem to prove the computed probability
theorem probability_same_color_is_correct :
  probability_all_same_color = 106 / 109725 := sorry

end probability_same_color_is_correct_l146_14648


namespace sugar_percentage_after_additions_l146_14602

noncomputable def initial_solution_volume : ℝ := 440
noncomputable def initial_water_percentage : ℝ := 0.88
noncomputable def initial_kola_percentage : ℝ := 0.08
noncomputable def initial_sugar_percentage : ℝ := 1 - initial_water_percentage - initial_kola_percentage
noncomputable def sugar_added : ℝ := 3.2
noncomputable def water_added : ℝ := 10
noncomputable def kola_added : ℝ := 6.8

noncomputable def initial_sugar_amount := initial_sugar_percentage * initial_solution_volume
noncomputable def new_sugar_amount := initial_sugar_amount + sugar_added
noncomputable def new_solution_volume := initial_solution_volume + sugar_added + water_added + kola_added

noncomputable def final_sugar_percentage := (new_sugar_amount / new_solution_volume) * 100

theorem sugar_percentage_after_additions :
    final_sugar_percentage = 4.52 :=
by
    sorry

end sugar_percentage_after_additions_l146_14602


namespace decreasing_function_range_a_l146_14637

noncomputable def f (a x : ℝ) : ℝ := -x^3 + x^2 + a * x

theorem decreasing_function_range_a (a : ℝ) :
  (∀ x : ℝ, deriv (f a) x ≤ 0) ↔ a ≤ -(1/3) :=
by
  -- This is a placeholder for the proof.
  sorry

end decreasing_function_range_a_l146_14637


namespace necessary_but_not_sufficient_l146_14677

theorem necessary_but_not_sufficient (x : ℝ) :
  (x^2 - 5*x + 4 < 0) → (|x - 2| < 1) ∧ ¬( |x - 2| < 1 → x^2 - 5*x + 4 < 0) :=
by 
  sorry

end necessary_but_not_sufficient_l146_14677


namespace proof_A2_less_than_3A1_plus_n_l146_14656

-- Define the conditions in terms of n, A1, and A2.
variables (n : ℕ)

-- A1 and A2 are the numbers of selections to select two students
-- such that their weight difference is ≤ 1 kg and ≤ 2 kg respectively.
variables (A1 A2 : ℕ)

-- The main theorem needs to prove that A2 < 3 * A1 + n.
theorem proof_A2_less_than_3A1_plus_n (h : A2 < 3 * A1 + n) : A2 < 3 * A1 + n :=
by {
  sorry -- proof goes here, but it's not required for the Lean statement.
}

end proof_A2_less_than_3A1_plus_n_l146_14656


namespace tap_filling_time_l146_14679

theorem tap_filling_time (T : ℝ) (hT1 : T > 0) 
  (h_fill_with_one_tap : ∀ (t : ℝ), t = T → t > 0)
  (h_fill_with_second_tap : ∀ (s : ℝ), s = 60 → s > 0)
  (both_open_first_10_minutes : 10 * (1 / T + 1 / 60) + 20 * (1 / 60) = 1) :
    T = 20 := 
sorry

end tap_filling_time_l146_14679


namespace part_one_part_two_l146_14640

def f (x : ℝ) := |x + 2|

theorem part_one (x : ℝ) : 2 * f x < 4 - |x - 1| ↔ -7 / 3 < x ∧ x < -1 := sorry

theorem part_two (m n : ℝ) (x a : ℝ) (h : m > 0) (h : n > 0) (h : m + n = 1) :
  (|x - a| - f x ≤ 1/m + 1/n) ↔ (-6 ≤ a ∧ a ≤ 2) := sorry

end part_one_part_two_l146_14640


namespace sum_D_E_F_l146_14600

theorem sum_D_E_F (D E F : ℤ) (h : ∀ x, x^3 + D * x^2 + E * x + F = (x + 3) * x * (x - 4)) : 
  D + E + F = -13 :=
by
  sorry

end sum_D_E_F_l146_14600


namespace average_age_of_girls_l146_14607

theorem average_age_of_girls (total_students : ℕ) (avg_age_boys : ℕ) (num_girls : ℕ) (avg_age_school : ℚ) 
  (h1 : total_students = 604) 
  (h2 : avg_age_boys = 12) 
  (h3 : num_girls = 151) 
  (h4 : avg_age_school = 11.75) : 
  (total_age_of_girls / num_girls) = 11 :=
by
  -- Definitions
  let num_boys := total_students - num_girls
  let total_age := avg_age_school * total_students
  let total_age_boys := avg_age_boys * num_boys
  let total_age_girls := total_age - total_age_boys
  -- Proof goal
  have : total_age_of_girls = total_age_girls := sorry
  have : total_age_of_girls / num_girls = 11 := sorry
  sorry

end average_age_of_girls_l146_14607


namespace system_of_equations_solution_l146_14647

/-- Integer solutions to the system of equations:
    \begin{cases}
        xz - 2yt = 3 \\
        xt + yz = 1
    \end{cases}
-/
theorem system_of_equations_solution :
  ∃ (x y z t : ℤ), 
    x * z - 2 * y * t = 3 ∧ 
    x * t + y * z = 1 ∧
    ((x = 1 ∧ y = 0 ∧ z = 3 ∧ t = 1) ∨
     (x = -1 ∧ y = 0 ∧ z = -3 ∧ t = -1) ∨
     (x = 3 ∧ y = 1 ∧ z = 1 ∧ t = 0) ∨
     (x = -3 ∧ y = -1 ∧ z = -1 ∧ t = 0)) :=
by {
  sorry
}

end system_of_equations_solution_l146_14647
