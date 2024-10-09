import Mathlib

namespace championship_winner_is_902_l1953_195324

namespace BasketballMatch

inductive Class : Type
| c901
| c902
| c903
| c904

open Class

def A_said (champ third : Class) : Prop :=
  champ = c902 ∧ third = c904

def B_said (fourth runner_up : Class) : Prop :=
  fourth = c901 ∧ runner_up = c903

def C_said (third champ : Class) : Prop :=
  third = c903 ∧ champ = c904

def half_correct (P Q : Prop) : Prop := 
  (P ∧ ¬Q) ∨ (¬P ∧ Q)

theorem championship_winner_is_902 (A_third B_fourth B_runner_up C_third : Class) 
  (H_A : half_correct (A_said c902 A_third) (A_said A_third c902))
  (H_B : half_correct (B_said B_fourth B_runner_up) (B_said B_runner_up B_fourth))
  (H_C : half_correct (C_said C_third c904) (C_said c904 C_third)) :
  ∃ winner, winner = c902 :=
sorry

end BasketballMatch

end championship_winner_is_902_l1953_195324


namespace find_value_of_expression_l1953_195383

theorem find_value_of_expression (a : ℝ) (h : a^2 + 3 * a - 1 = 0) : 2 * a^2 + 6 * a + 2021 = 2023 := 
by
  sorry

end find_value_of_expression_l1953_195383


namespace y_affected_by_other_factors_l1953_195314

-- Given the linear regression model
def linear_regression_model (b a e x : ℝ) : ℝ := b * x + a + e

-- Theorem: Prove that the dependent variable \( y \) may be affected by factors other than the independent variable \( x \)
theorem y_affected_by_other_factors (b a e x : ℝ) :
  ∃ y, (y = linear_regression_model b a e x ∧ e ≠ 0) :=
sorry

end y_affected_by_other_factors_l1953_195314


namespace constant_S13_l1953_195397

theorem constant_S13 (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) (h_seq : ∀ n, a n = a 1 + (n - 1) * d) 
(h_sum : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2)
(h_constant : ∀ a1 d, (a 2 + a 8 + a 11 = 3 * a1 + 18 * d)) : (S 13 = 91 * d) :=
by
  sorry

end constant_S13_l1953_195397


namespace largest_integer_value_of_x_l1953_195331

theorem largest_integer_value_of_x (x : ℤ) (h : 8 - 5 * x > 22) : x ≤ -3 :=
sorry

end largest_integer_value_of_x_l1953_195331


namespace probability_same_color_is_correct_l1953_195305

/- Given that there are 5 balls in total, where 3 are white and 2 are black, and two balls are drawn randomly from the bag, we need to prove that the probability of drawing two balls of the same color is 2/5. -/

def total_balls : ℕ := 5
def white_balls : ℕ := 3
def black_balls : ℕ := 2

def total_ways (n r : ℕ) : ℕ := n.choose r
def white_ways : ℕ := total_ways white_balls 2
def black_ways : ℕ := total_ways black_balls 2
def same_color_ways : ℕ := white_ways + black_ways
def total_draws : ℕ := total_ways total_balls 2

def probability_same_color := ((same_color_ways : ℚ) / total_draws)
def expected_probability := (2 : ℚ) / 5

theorem probability_same_color_is_correct :
  probability_same_color = expected_probability :=
by
  sorry

end probability_same_color_is_correct_l1953_195305


namespace not_sufficient_not_necessary_l1953_195348

theorem not_sufficient_not_necessary (a : ℝ) :
  ¬ ((a^2 > 1) → (1/a > 0)) ∧ ¬ ((1/a > 0) → (a^2 > 1)) := sorry

end not_sufficient_not_necessary_l1953_195348


namespace ral_current_age_l1953_195385

theorem ral_current_age (Ral_age Suri_age : ℕ) (h1 : Ral_age = 2 * Suri_age) (h2 : Suri_age + 3 = 16) : Ral_age = 26 :=
by {
  -- Proof goes here
  sorry
}

end ral_current_age_l1953_195385


namespace tangent_line_to_circle_polar_l1953_195303

-- Definitions
def polar_circle_equation (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ
def point_polar_coordinates (ρ θ : ℝ) : Prop := ρ = 2 * Real.sqrt 2 ∧ θ = Real.pi / 4
def tangent_line_polar_equation (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 2

-- Theorem Statement
theorem tangent_line_to_circle_polar {ρ θ : ℝ} :
  (∃ ρ θ, polar_circle_equation ρ θ) →
  (∃ ρ θ, point_polar_coordinates ρ θ) →
  tangent_line_polar_equation ρ θ :=
sorry

end tangent_line_to_circle_polar_l1953_195303


namespace scientific_notation_of_600000_l1953_195333

theorem scientific_notation_of_600000 :
  600000 = 6 * 10^5 :=
sorry

end scientific_notation_of_600000_l1953_195333


namespace Diane_age_when_conditions_met_l1953_195372

variable (Diane_current : ℕ) (Alex_current : ℕ) (Allison_current : ℕ)
variable (D : ℕ)

axiom Diane_current_age : Diane_current = 16
axiom Alex_Allison_sum : Alex_current + Allison_current = 47
axiom Diane_half_Alex : D = (Alex_current + (D - 16)) / 2
axiom Diane_twice_Allison : D = 2 * (Allison_current + (D - 16))

theorem Diane_age_when_conditions_met : D = 78 :=
by
  sorry

end Diane_age_when_conditions_met_l1953_195372


namespace cookies_sold_by_Lucy_l1953_195310

theorem cookies_sold_by_Lucy :
  let cookies_first_round := 34
  let cookies_second_round := 27
  cookies_first_round + cookies_second_round = 61 := by
  sorry

end cookies_sold_by_Lucy_l1953_195310


namespace find_a_l1953_195308

theorem find_a (x a : ℕ) (h : (x + 4) + 4 = (5 * x + a + 38) / 5) : a = 2 :=
sorry

end find_a_l1953_195308


namespace lighter_dog_weight_l1953_195368

theorem lighter_dog_weight
  (x y z : ℕ)
  (h1 : x + y + z = 36)
  (h2 : y + z = 3 * x)
  (h3 : x + z = 2 * y) :
  x = 9 :=
by
  sorry

end lighter_dog_weight_l1953_195368


namespace prime_5p_plus_4p4_is_perfect_square_l1953_195394

theorem prime_5p_plus_4p4_is_perfect_square (p : ℕ) [Fact (Nat.Prime p)] :
  ∃ q : ℕ, 5^p + 4 * p^4 = q^2 ↔ p = 5 :=
by
  sorry

end prime_5p_plus_4p4_is_perfect_square_l1953_195394


namespace sundae_cost_l1953_195371

theorem sundae_cost (ice_cream_cost toppings_cost : ℕ) (num_toppings : ℕ) :
  ice_cream_cost = 200  →
  toppings_cost = 50 →
  num_toppings = 10 →
  ice_cream_cost + num_toppings * toppings_cost = 700 := by
  sorry

end sundae_cost_l1953_195371


namespace surface_area_of_cube_l1953_195374

-- Define the condition: volume of the cube is 1728 cubic centimeters
def volume_cube (s : ℝ) : ℝ := s^3
def given_volume : ℝ := 1728

-- Define the question: surface area of the cube
def surface_area_cube (s : ℝ) : ℝ := 6 * s^2

-- The statement that needs to be proved
theorem surface_area_of_cube :
  ∃ s : ℝ, volume_cube s = given_volume → surface_area_cube s = 864 :=
by
  sorry

end surface_area_of_cube_l1953_195374


namespace ratio_product_even_odd_composite_l1953_195321

theorem ratio_product_even_odd_composite :
  (4 * 6 * 8 * 10 * 12) / (9 * 15 * 21 * 25 * 27) = (2^10) / (3^6 * 5^2 * 7) :=
by
  sorry

end ratio_product_even_odd_composite_l1953_195321


namespace largest_n_with_triangle_property_l1953_195375

/-- Triangle property: For any subset {a, b, c} with a ≤ b ≤ c, a + b > c -/
def triangle_property (s : Finset ℕ) : Prop :=
  ∀ (a b c : ℕ), a ∈ s → b ∈ s → c ∈ s → a ≤ b → b ≤ c → a + b > c

/-- Definition of the set {3, 4, ..., n} -/
def consecutive_set (n : ℕ) : Finset ℕ :=
  Finset.range (n + 1) \ Finset.range 3

/-- The problem statement: The largest possible value of n where all eleven-element
 subsets of {3, 4, ..., n} have the triangle property -/
theorem largest_n_with_triangle_property : ∃ n, (∀ s ⊆ consecutive_set n, s.card = 11 → triangle_property s) ∧ n = 321 := sorry

end largest_n_with_triangle_property_l1953_195375


namespace other_acute_angle_measure_l1953_195378

-- Definitions based on the conditions
def right_triangle_sum (a b : ℝ) : Prop := a + b = 90
def is_right_triangle (a b : ℝ) : Prop := right_triangle_sum a b ∧ a = 20

-- The statement to prove
theorem other_acute_angle_measure {a b : ℝ} (h : is_right_triangle a b) : b = 70 :=
sorry

end other_acute_angle_measure_l1953_195378


namespace calculation_correct_l1953_195389

theorem calculation_correct (x y : ℝ) (hx0 : x ≠ 0) (hy0 : y ≠ 0) (hxy : x = 2 * y) : 
  (x - 2 / x) * (y + 2 / y) = 1 / 2 * (x^2 - 2 * x + 8 - 16 / x) := 
by 
  sorry

end calculation_correct_l1953_195389


namespace find_m_range_l1953_195379

noncomputable def proposition_p (x : ℝ) : Prop := (-2 : ℝ) ≤ x ∧ x ≤ 10
noncomputable def proposition_q (x : ℝ) (m : ℝ) : Prop := (1 - m ≤ x ∧ x ≤ 1 + m)

theorem find_m_range (m : ℝ) (h : m > 0) : (¬ ∃ x : ℝ, proposition_p x) → (¬ ∃ x : ℝ, proposition_q x m) → (¬ (¬ (¬ ∃ x : ℝ, proposition_q x m)) → ¬ (¬ ∃ x : ℝ, proposition_p x)) → m ≥ 9 := 
sorry

end find_m_range_l1953_195379


namespace odd_square_not_sum_of_five_odd_squares_l1953_195311

theorem odd_square_not_sum_of_five_odd_squares :
  ∀ (n : ℤ), (∃ k : ℤ, k^2 % 8 = n % 8 ∧ n % 8 = 1) →
             ¬(∃ a b c d e : ℤ, (a^2 % 8 = 1) ∧ (b^2 % 8 = 1) ∧ (c^2 % 8 = 1) ∧ (d^2 % 8 = 1) ∧ 
               (e^2 % 8 = 1) ∧ (n % 8 = (a^2 + b^2 + c^2 + d^2 + e^2) % 8)) :=
by
  sorry

end odd_square_not_sum_of_five_odd_squares_l1953_195311


namespace evaluate_f_at_2_l1953_195319

-- Define the polynomial function f(x)
def f (x : ℝ) : ℝ := 3 * x^6 - 2 * x^5 + x^3 + 1

theorem evaluate_f_at_2 : f 2 = 34 :=
by
  -- Insert proof here
  sorry

end evaluate_f_at_2_l1953_195319


namespace number_13_on_top_after_folds_l1953_195342

/-
A 5x5 grid of numbers from 1 to 25 with the following sequence of folds:
1. Fold along the diagonal from bottom-left to top-right
2. Fold the left half over the right half
3. Fold the top half over the bottom half
4. Fold the bottom half over the top half
Prove that the number 13 ends up on top after all folds.
-/

def grid := (⟨ 5, 5 ⟩ : Nat × Nat)

def initial_grid : ℕ → ℕ := λ n => if 1 ≤ n ∧ n ≤ 25 then n else 0

def fold_diagonal (g : ℕ → ℕ) : ℕ → ℕ := sorry -- implement function to represent step 1 fold

def fold_left_over_right (g : ℕ → ℕ) : ℕ → ℕ := sorry -- implement function to represent step 2 fold

def fold_top_over_bottom (g : ℕ → ℕ) : ℕ → ℕ := sorry -- implement function to represent step 3 fold

def fold_bottom_over_top (g : ℕ → ℕ) : ℕ → ℕ := sorry -- implement function to represent step 4 fold

theorem number_13_on_top_after_folds : (fold_bottom_over_top (fold_top_over_bottom (fold_left_over_right (fold_diagonal initial_grid)))) 13 = 13 :=
by {
  sorry
}

end number_13_on_top_after_folds_l1953_195342


namespace Bryan_deposit_amount_l1953_195313

theorem Bryan_deposit_amount (deposit_mark : ℕ) (deposit_bryan : ℕ)
  (h1 : deposit_mark = 88)
  (h2 : deposit_bryan = 5 * deposit_mark - 40) : 
  deposit_bryan = 400 := 
by
  sorry

end Bryan_deposit_amount_l1953_195313


namespace groups_needed_for_sampling_l1953_195365

def total_students : ℕ := 600
def sample_size : ℕ := 20

theorem groups_needed_for_sampling : (total_students / sample_size = 30) :=
by
  sorry

end groups_needed_for_sampling_l1953_195365


namespace sum_of_solutions_l1953_195364

theorem sum_of_solutions (x : ℝ) :
  (∀ x : ℝ, x^3 + x^2 - 6*x - 20 = 4*x + 24) →
  let polynomial := (x^3 + x^2 - 10*x - 44);
  (polynomial = 0) →
  let a := 1;
  let b := 1;
  -b/a = -1 :=
sorry

end sum_of_solutions_l1953_195364


namespace quadratic_discriminant_l1953_195344

theorem quadratic_discriminant : 
  let a := 4
  let b := -6
  let c := 9
  (b^2 - 4 * a * c = -108) := 
by
  sorry

end quadratic_discriminant_l1953_195344


namespace num_new_books_not_signed_l1953_195350

theorem num_new_books_not_signed (adventure_books mystery_books science_fiction_books non_fiction_books used_books signed_books : ℕ)
    (h1 : adventure_books = 13)
    (h2 : mystery_books = 17)
    (h3 : science_fiction_books = 25)
    (h4 : non_fiction_books = 10)
    (h5 : used_books = 42)
    (h6 : signed_books = 10) : 
    (adventure_books + mystery_books + science_fiction_books + non_fiction_books) - used_books - signed_books = 13 := 
by
  sorry

end num_new_books_not_signed_l1953_195350


namespace tan_subtraction_l1953_195304

theorem tan_subtraction (α β : ℝ) (h₁ : Real.tan α = 9) (h₂ : Real.tan β = 6) :
  Real.tan (α - β) = 3 / 55 :=
by
  sorry

end tan_subtraction_l1953_195304


namespace sale_in_fifth_month_l1953_195373

theorem sale_in_fifth_month (Sale1 Sale2 Sale3 Sale4 Sale6 AvgSale : ℤ) 
(h1 : Sale1 = 6435) (h2 : Sale2 = 6927) (h3 : Sale3 = 6855) (h4 : Sale4 = 7230) 
(h5 : Sale6 = 4991) (h6 : AvgSale = 6500) : (39000 - (Sale1 + Sale2 + Sale3 + Sale4 + Sale6)) = 6562 :=
by
  sorry

end sale_in_fifth_month_l1953_195373


namespace similar_right_triangles_hypotenuse_relation_similar_right_triangles_reciprocal_relation_l1953_195363

variable {a b c m_c a' b' c' m_c' : ℝ}

/- The first proof problem -/
theorem similar_right_triangles_hypotenuse_relation (h_sim : (a = k * a') ∧ (b = k * b') ∧ (c = k * c')) :
  a * a' + b * b' = c * c' := by
  sorry

/- The second proof problem -/
theorem similar_right_triangles_reciprocal_relation (h_sim : (a = k * a') ∧ (b = k * b') ∧ (c = k * c') ∧ (m_c = k * m_c')) :
  (1 / (a * a') + 1 / (b * b')) = 1 / (m_c * m_c') := by
  sorry

end similar_right_triangles_hypotenuse_relation_similar_right_triangles_reciprocal_relation_l1953_195363


namespace ellipse_area_quadrants_eq_zero_l1953_195367

theorem ellipse_area_quadrants_eq_zero 
(E : Type)
(x y : E → ℝ) 
(h_ellipse : ∀ (x y : ℝ), (x - 19)^2 / (19 * 1998) + (y - 98)^2 / (98 * 1998) = 1998) 
(R1 R2 R3 R4 : ℝ)
(H1 : ∀ (R1 R2 R3 R4 : ℝ), R1 = R_ellipse / 4 ∧ R2 = R_ellipse / 4 ∧ R3 = R_ellipse / 4 ∧ R4 = R_ellipse / 4)
: R1 - R2 + R3 - R4 = 0 := 
by 
sorry

end ellipse_area_quadrants_eq_zero_l1953_195367


namespace sally_nickels_count_l1953_195323

theorem sally_nickels_count (original_nickels dad_nickels mom_nickels : ℕ) 
    (h1: original_nickels = 7) 
    (h2: dad_nickels = 9) 
    (h3: mom_nickels = 2) 
    : original_nickels + dad_nickels + mom_nickels = 18 :=
by
  sorry

end sally_nickels_count_l1953_195323


namespace problem1_problem2_problem3_l1953_195302

noncomputable def f : ℝ → ℝ := sorry -- Define your function here satisfying the conditions

theorem problem1 (h1 : ∀ x : ℝ, f (2 - x) = f x)
                 (h2 : ∀ x : ℝ, x ≥ 1 → f x = Real.log (x + 1/x)) :
  f (-1) = 1 - Real.log 3 := sorry

theorem problem2 (h1 : ∀ x : ℝ, f (2 - x) = f x)
                 (h2 : ∀ x : ℝ, x ≥ 1 → f x = Real.log (x + 1/x)) :
  ∀ x : ℝ, f (2 - 2 * x) < f (x + 3) ↔ x ∈ Set.Ico (-1/3) 3 := sorry

theorem problem3 (h1 : ∀ x : ℝ, f (2 - x) = f x)
                 (h2 : ∀ x : ℝ, x ≥ 1 → f x = Real.log (x + 1/x))
                 (a : ℝ) :
  (∃ x : ℝ, 1 < x ∧ f x = Real.log (a / x + 2 * a)) ↔ a > 2/3 := sorry

end problem1_problem2_problem3_l1953_195302


namespace amy_l1953_195360

theorem amy's_speed (a b : ℝ) (s : ℝ) 
  (h1 : ∀ (major minor : ℝ), major = 2 * minor) 
  (h2 : ∀ (w : ℝ), w = 4) 
  (h3 : ∀ (t_diff : ℝ), t_diff = 48) 
  (h4 : 2 * a + 2 * Real.pi * Real.sqrt ((4 * b^2 + b^2) / 2) - (2 * a + 2 * Real.pi * Real.sqrt (((2 * b + 8)^2 + (b + 4)^2) / 2)) = 48 * s) :
  s = Real.pi / 2 := sorry

end amy_l1953_195360


namespace inequality_A_if_ab_pos_inequality_D_if_ab_pos_l1953_195307

variable (a b : ℝ)

theorem inequality_A_if_ab_pos (h : a * b > 0) : a^2 + b^2 ≥ 2 * a * b := 
sorry

theorem inequality_D_if_ab_pos (h : a * b > 0) : (b / a) + (a / b) ≥ 2 :=
sorry

end inequality_A_if_ab_pos_inequality_D_if_ab_pos_l1953_195307


namespace problem_l1953_195377

-- Define the matrix
def A : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![2, 5, 0], ![0, 2, 3], ![3, 0, 2]]

-- Define the condition that there exists a nonzero vector v such that A * v = k * v
def exists_eigenvector (k : ℝ) : Prop :=
  ∃ (v : Fin 3 → ℝ), v ≠ 0 ∧ A.mulVec v = k • v

theorem problem : ∀ (k : ℝ), exists_eigenvector k ↔ (k = 2 + (45)^(1/3)) :=
sorry

end problem_l1953_195377


namespace simplify_expression_l1953_195395

variable (a b : ℝ)

theorem simplify_expression :
  -2 * (a^3 - 3 * b^2) + 4 * (-b^2 + a^3) = 2 * a^3 + 2 * b^2 :=
by
  sorry

end simplify_expression_l1953_195395


namespace geometric_sequence_11th_term_l1953_195339

theorem geometric_sequence_11th_term (a r : ℝ) (h₁ : a * r ^ 4 = 8) (h₂ : a * r ^ 7 = 64) : 
  a * r ^ 10 = 512 :=
by sorry

end geometric_sequence_11th_term_l1953_195339


namespace calc_expr_l1953_195336

theorem calc_expr :
  (-1) * (-3) + 3^2 / (8 - 5) = 6 :=
by
  sorry

end calc_expr_l1953_195336


namespace find_n_l1953_195327

def x := 3
def y := 1
def n := x - 3 * y^(x - y) + 1

theorem find_n : n = 1 :=
by
  unfold n x y
  sorry

end find_n_l1953_195327


namespace initial_pennies_indeterminate_l1953_195317

-- Conditions
def initial_nickels : ℕ := 7
def dad_nickels : ℕ := 9
def mom_nickels : ℕ := 2
def total_nickels_now : ℕ := 18

-- Proof problem statement
theorem initial_pennies_indeterminate :
  ∀ (initial_nickels dad_nickels mom_nickels total_nickels_now : ℕ), 
  initial_nickels = 7 → dad_nickels = 9 → mom_nickels = 2 → total_nickels_now = 18 → 
  (∃ (initial_pennies : ℕ), true) → false :=
by
  sorry

end initial_pennies_indeterminate_l1953_195317


namespace total_molecular_weight_correct_l1953_195322

-- Defining the molecular weights of elements
def mol_weight_C : ℝ := 12.01
def mol_weight_H : ℝ := 1.01
def mol_weight_Cl : ℝ := 35.45
def mol_weight_O : ℝ := 16.00

-- Defining the number of moles of compounds
def moles_C2H5Cl : ℝ := 15
def moles_O2 : ℝ := 12

-- Calculating the molecular weights of compounds
def mol_weight_C2H5Cl : ℝ := (2 * mol_weight_C) + (5 * mol_weight_H) + mol_weight_Cl
def mol_weight_O2 : ℝ := 2 * mol_weight_O

-- Calculating the total weight of each compound
def total_weight_C2H5Cl : ℝ := moles_C2H5Cl * mol_weight_C2H5Cl
def total_weight_O2 : ℝ := moles_O2 * mol_weight_O2

-- Defining the final total weight
def total_weight : ℝ := total_weight_C2H5Cl + total_weight_O2

-- Statement to prove
theorem total_molecular_weight_correct :
  total_weight = 1351.8 := by
  sorry

end total_molecular_weight_correct_l1953_195322


namespace compute_value_l1953_195398

variable (p q : ℚ)
variable (h : ∀ x, 3 * x^2 - 7 * x - 6 = 0 → x = p ∨ x = q)

theorem compute_value (h_pq : p ≠ q) : (5 * p^3 - 5 * q^3) * (p - q)⁻¹ = 335 / 9 := by
  -- We assume p and q are the roots of the polynomial and p ≠ q.
  have sum_roots : p + q = 7 / 3 := sorry
  have prod_roots : p * q = -2 := sorry
  -- Additional steps to derive the required result (proof) are ignored here.
  sorry

end compute_value_l1953_195398


namespace radius_of_inscribed_sphere_l1953_195362

theorem radius_of_inscribed_sphere (a b c s : ℝ)
  (h1: 2 * (a * b + a * c + b * c) = 616)
  (h2: a + b + c = 40)
  : s = Real.sqrt 246 ↔ (2 * s) ^ 2 = a ^ 2 + b ^ 2 + c ^ 2 :=
by
  sorry

end radius_of_inscribed_sphere_l1953_195362


namespace shadow_length_of_flagpole_is_correct_l1953_195345

noncomputable def length_of_shadow_flagpole : ℕ :=
  let h_flagpole : ℕ := 18
  let shadow_building : ℕ := 60
  let h_building : ℕ := 24
  let similar_conditions : Prop := true
  45

theorem shadow_length_of_flagpole_is_correct :
  length_of_shadow_flagpole = 45 := by
  sorry

end shadow_length_of_flagpole_is_correct_l1953_195345


namespace sum_series_l1953_195359

theorem sum_series :
  ∑' n : ℕ, (2^n / (3^(2^n) + 1)) = 1 / 2 :=
sorry

end sum_series_l1953_195359


namespace smallest_n_exists_l1953_195334

theorem smallest_n_exists :
  ∃ (a1 a2 a3 a4 a5 : ℤ), a1 + a2 + a3 + a4 + a5 = 1990 ∧ a1 * a2 * a3 * a4 * a5 = 1990 :=
sorry

end smallest_n_exists_l1953_195334


namespace mixed_number_fraction_division_and_subtraction_l1953_195332

theorem mixed_number_fraction_division_and_subtraction :
  ( (11 / 6) / (11 / 4) ) - (1 / 2) = 1 / 6 := 
sorry

end mixed_number_fraction_division_and_subtraction_l1953_195332


namespace train_speed_l1953_195384

-- Define the conditions given in the problem
def train_length : ℝ := 160
def time_to_cross_man : ℝ := 4

-- Define the statement to be proved
theorem train_speed (H1 : train_length = 160) (H2 : time_to_cross_man = 4) : train_length / time_to_cross_man = 40 :=
by
  sorry

end train_speed_l1953_195384


namespace cost_price_per_metre_l1953_195388

theorem cost_price_per_metre (total_selling_price : ℕ) (total_metres : ℕ) (loss_per_metre : ℕ)
  (h1 : total_selling_price = 9000)
  (h2 : total_metres = 300)
  (h3 : loss_per_metre = 6) :
  (total_selling_price + (loss_per_metre * total_metres)) / total_metres = 36 :=
by
  sorry

end cost_price_per_metre_l1953_195388


namespace range_of_a_l1953_195356

theorem range_of_a :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 2 * x * (3 * x + a) < 1) → a < 1 :=
by
  sorry

end range_of_a_l1953_195356


namespace Hector_gumballs_l1953_195320

theorem Hector_gumballs :
  ∃ (total_gumballs : ℕ)
  (gumballs_Todd : ℕ) (gumballs_Alisha : ℕ) (gumballs_Bobby : ℕ) (gumballs_remaining : ℕ),
  gumballs_Todd = 4 ∧
  gumballs_Alisha = 2 * gumballs_Todd ∧
  gumballs_Bobby = 4 * gumballs_Alisha - 5 ∧
  gumballs_remaining = 6 ∧
  total_gumballs = gumballs_Todd + gumballs_Alisha + gumballs_Bobby + gumballs_remaining ∧
  total_gumballs = 45 :=
by
  sorry

end Hector_gumballs_l1953_195320


namespace a_plus_d_eq_five_l1953_195309

theorem a_plus_d_eq_five (a b c d k : ℝ) (hk : 0 < k) 
  (h1 : a + b = 11) 
  (h2 : b^2 + c^2 = k) 
  (h3 : b + c = 9) 
  (h4 : c + d = 3) : 
  a + d = 5 :=
by
  sorry

end a_plus_d_eq_five_l1953_195309


namespace factor_theorem_example_l1953_195340

theorem factor_theorem_example (t : ℚ) : (4 * t^3 + 6 * t^2 + 11 * t - 6 = 0) ↔ (t = 1/2) :=
by sorry

end factor_theorem_example_l1953_195340


namespace pow_mult_same_base_l1953_195316

theorem pow_mult_same_base (a b : ℕ) : 10^a * 10^b = 10^(a + b) := by 
  sorry

example : 10^655 * 10^652 = 10^1307 :=
  pow_mult_same_base 655 652

end pow_mult_same_base_l1953_195316


namespace cody_tickets_l1953_195341

theorem cody_tickets (initial_tickets spent_tickets won_tickets : ℕ) (h_initial : initial_tickets = 49) (h_spent : spent_tickets = 25) (h_won : won_tickets = 6) : initial_tickets - spent_tickets + won_tickets = 30 := 
by 
  sorry

end cody_tickets_l1953_195341


namespace paint_cost_l1953_195376

theorem paint_cost (l : ℝ) (b : ℝ) (rate : ℝ) (area : ℝ) (cost : ℝ) 
  (h1 : l = 3 * b) 
  (h2 : l = 18.9999683334125) 
  (h3 : rate = 3.00001) 
  (h4 : area = l * b) 
  (h5 : cost = area * rate) : 
  cost = 361.00 :=
by
  sorry

end paint_cost_l1953_195376


namespace find_p_l1953_195301

def parabola_def (p : ℝ) : Prop := p > 0 ∧ ∀ (m : ℝ), (2 - (-p/2) = 4)

theorem find_p (p : ℝ) (m : ℝ) (h₁ : parabola_def p) (h₂ : (m ^ 2) = 2 * p * 2) 
(h₃ : (m ^ 2) = 2 * p * 2 → dist (2, m) (p / 2, 0) = 4) :
p = 4 :=
by
  sorry

end find_p_l1953_195301


namespace pages_left_in_pad_l1953_195330

-- Definitions from conditions
def total_pages : ℕ := 120
def science_project_pages (total : ℕ) : ℕ := total * 25 / 100
def math_homework_pages : ℕ := 10

-- Proving the final number of pages left
theorem pages_left_in_pad :
  let remaining_pages_after_usage := total_pages - science_project_pages total_pages - math_homework_pages
  let pages_left_after_art_project := remaining_pages_after_usage / 2
  pages_left_after_art_project = 40 :=
by
  sorry

end pages_left_in_pad_l1953_195330


namespace incorrect_rational_number_statement_l1953_195306

theorem incorrect_rational_number_statement :
  ¬ (∀ x : ℚ, x > 0 ∨ x < 0) := by
sorry

end incorrect_rational_number_statement_l1953_195306


namespace max_prime_factors_of_c_l1953_195312

-- Definitions of conditions
variables (c d : ℕ)
variable (prime_factor_count : ℕ → ℕ)
variable (gcd : ℕ → ℕ → ℕ)
variable (lcm : ℕ → ℕ → ℕ)

-- Conditions
axiom gcd_condition : prime_factor_count (gcd c d) = 11
axiom lcm_condition : prime_factor_count (lcm c d) = 44
axiom fewer_prime_factors : prime_factor_count c < prime_factor_count d

-- Proof statement
theorem max_prime_factors_of_c : prime_factor_count c ≤ 27 := 
sorry

end max_prime_factors_of_c_l1953_195312


namespace quadratic_real_solutions_l1953_195382

theorem quadratic_real_solutions (x y : ℝ) :
  (∃ z : ℝ, 16 * z^2 + 4 * x * y * z + (y^2 - 3) = 0) ↔ (x ≤ -2 ∨ x ≥ 2) :=
by
  sorry

end quadratic_real_solutions_l1953_195382


namespace no_nat_number_divisible_by_1998_has_digit_sum_lt_27_l1953_195352

-- Definition of a natural number being divisible by another
def divisible (m n : ℕ) : Prop := ∃ k : ℕ, m = k * n

-- Definition of the sum of the digits of a natural number
def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

-- Statement of the problem
theorem no_nat_number_divisible_by_1998_has_digit_sum_lt_27 :
  ¬ ∃ n : ℕ, divisible n 1998 ∧ sum_of_digits n < 27 :=
by 
  sorry

end no_nat_number_divisible_by_1998_has_digit_sum_lt_27_l1953_195352


namespace flavored_drink_ratio_l1953_195355

theorem flavored_drink_ratio :
  ∃ (F C W: ℚ), F / C = 1 / 7.5 ∧ F / W = 1 / 56.25 ∧ C/W = 6/90 ∧ F / C / 3 = ((F / W) * 2)
:= sorry

end flavored_drink_ratio_l1953_195355


namespace future_value_option_B_correct_l1953_195393

noncomputable def future_value_option_B (p q : ℝ) : ℝ :=
  150 * (1 + p / 100 / 2) ^ 6 * (1 + q / 100 / 4) ^ 12

theorem future_value_option_B_correct (p q A₂ : ℝ) :
  A₂ = 150 * (1 + p / 100 / 2) ^ 6 * (1 + q / 100 / 4) ^ 12 →
  ∃ A₂, A₂ = future_value_option_B p q :=
by
  intro h
  use A₂
  exact h

end future_value_option_B_correct_l1953_195393


namespace relay_team_order_count_l1953_195357

theorem relay_team_order_count :
  ∃ (orders : ℕ), orders = 6 :=
by
  let team_members := 4
  let remaining_members := team_members - 1  -- Excluding Lisa
  let first_lap_choices := remaining_members.choose 3  -- Choices for the first lap
  let third_lap_choices := (remaining_members - 1).choose 2  -- Choices for the third lap
  let fourth_lap_choices := (remaining_members - 2).choose 1  -- The last remaining member choices
  have orders := first_lap_choices * third_lap_choices * fourth_lap_choices
  use orders
  sorry

end relay_team_order_count_l1953_195357


namespace counterexample_to_proposition_l1953_195386

theorem counterexample_to_proposition : ∃ (a : ℝ), a^2 > 0 ∧ a ≤ 0 :=
  sorry

end counterexample_to_proposition_l1953_195386


namespace q_value_l1953_195399

-- Define the problem conditions
def prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_multiple_of (a b : ℕ) : Prop := ∃ k, a = k * b

-- Statement of the problem
theorem q_value (p q : ℕ) (hp : prime p) (hq : prime q) (h1 : q = 13 * p + 2) (h2 : is_multiple_of (q - 1) 3) : q = 67 :=
sorry

end q_value_l1953_195399


namespace calculate_expression_l1953_195392

theorem calculate_expression :
  (1/4 * 6.16^2) - (4 * 1.04^2) = 5.16 :=
by
  sorry

end calculate_expression_l1953_195392


namespace find_varphi_l1953_195361

theorem find_varphi (ϕ : ℝ) (h0 : 0 < ϕ ∧ ϕ < π / 2) :
  (∀ x₁ x₂, |(2 * Real.cos (2 * x₁)) - (2 * Real.cos (2 * x₂ - 2 * ϕ))| = 4 → 
    ∃ (x₁ x₂ : ℝ), |x₁ - x₂| = π / 6 
  ) → ϕ = π / 3 :=
by
  sorry

end find_varphi_l1953_195361


namespace group_card_exchanges_l1953_195318

theorem group_card_exchanges (x : ℕ) (hx : x * (x - 1) = 90) : x = 10 :=
by { sorry }

end group_card_exchanges_l1953_195318


namespace maximum_area_of_rectangle_with_fixed_perimeter_l1953_195369

theorem maximum_area_of_rectangle_with_fixed_perimeter (x y : ℝ) 
  (h₁ : 2 * (x + y) = 40) 
  (h₂ : x = y) :
  x * y = 100 :=
by
  sorry

end maximum_area_of_rectangle_with_fixed_perimeter_l1953_195369


namespace gcd_72_108_l1953_195381

theorem gcd_72_108 : Nat.gcd 72 108 = 36 :=
by
  sorry

end gcd_72_108_l1953_195381


namespace coffee_consumption_l1953_195325

theorem coffee_consumption (h1 h2 g1 h3: ℕ) (k : ℕ) (g2 : ℕ) :
  (k = h1 * g1) → (h1 = 9) → (g1 = 2) → (h2 = 6) → (k / h2 = g2) → (g2 = 3) :=
by
  sorry

end coffee_consumption_l1953_195325


namespace find_a_l1953_195391

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^x / (1 + a * 2^x)

theorem find_a (a : ℝ) (f : ℝ → ℝ) (h_f_def : ∀ x, f x = 2^x / (1 + a * 2^x))
  (h_symm : ∀ x, f x + f (-x) = 1) : a = 1 :=
sorry

end find_a_l1953_195391


namespace eiffel_tower_model_height_l1953_195326

theorem eiffel_tower_model_height 
  (H1 : ℝ) (W1 : ℝ) (W2 : ℝ) (H2 : ℝ)
  (h1 : H1 = 324)
  (w1 : W1 = 8000000)  -- converted 8000 tons to 8000000 kg
  (w2 : W2 = 1)
  (h_eq : (H2 / H1)^3 = W2 / W1) : 
  H2 = 1.62 :=
by
  rw [h1, w1, w2] at h_eq
  sorry

end eiffel_tower_model_height_l1953_195326


namespace proportion1_proportion2_l1953_195380

theorem proportion1 (x : ℚ) : (x / (5 / 9) = (1 / 20) / (1 / 3)) → x = 1 / 12 :=
sorry

theorem proportion2 (x : ℚ) : (x / 0.25 = 0.5 / 0.1) → x = 1.25 :=
sorry

end proportion1_proportion2_l1953_195380


namespace compute_fraction_l1953_195338

theorem compute_fraction :
  ( (11^4 + 400) * (25^4 + 400) * (37^4 + 400) * (49^4 + 400) * (61^4 + 400) ) /
  ( (5^4 + 400) * (17^4 + 400) * (29^4 + 400) * (41^4 + 400) * (53^4 + 400) ) = 799 := 
by
  sorry

end compute_fraction_l1953_195338


namespace parabola_no_real_intersection_l1953_195343

theorem parabola_no_real_intersection (a b c : ℝ) (h₁ : a = 1) (h₂ : b = -4) (h₃ : c = 5) :
  ∀ (x : ℝ), ¬ (a * x^2 + b * x + c = 0) :=
by
  sorry

end parabola_no_real_intersection_l1953_195343


namespace set_different_l1953_195370

-- Definitions of the sets ①, ②, ③, and ④
def set1 : Set ℤ := {x | x = 1}
def set2 : Set ℤ := {y | (y - 1)^2 = 0}
def set3 : Set ℤ := {x | x = 1}
def set4 : Set ℤ := {1}

-- Lean statement to prove that set3 is different from the others
theorem set_different : set3 ≠ set1 ∧ set3 ≠ set2 ∧ set3 ≠ set4 :=
by
  -- Skipping the proof with sorry
  sorry

end set_different_l1953_195370


namespace explicit_formula_for_f_l1953_195358

def f (k : ℕ) : ℚ :=
  if k = 1 then 4 / 3
  else (5 ^ (k + 1) / 81) * (8 * 10 ^ (1 - k) - 9 * k + 1) + (2 ^ (k + 1)) / 3

theorem explicit_formula_for_f (k : ℕ) (hk : k ≥ 1) : 
  (f k = if k = 1 then 4 / 3 else (5 ^ (k + 1) / 81) * (8 * 10 ^ (1 - k) - 9 * k + 1) + (2 ^ (k + 1)) / 3) ∧ 
  ∀ k ≥ 2, 2 * f k = f (k - 1) - k * 5^k + 2^k :=
by {
  sorry
}

end explicit_formula_for_f_l1953_195358


namespace remainder_of_9_6_plus_8_7_plus_7_8_mod_7_l1953_195354

theorem remainder_of_9_6_plus_8_7_plus_7_8_mod_7 : (9^6 + 8^7 + 7^8) % 7 = 2 := 
by sorry

end remainder_of_9_6_plus_8_7_plus_7_8_mod_7_l1953_195354


namespace circle_center_sum_l1953_195300

theorem circle_center_sum {x y : ℝ} (h : x^2 + y^2 - 10*x + 4*y + 15 = 0) :
  (x, y) = (5, -2) ∧ x + y = 3 :=
by
  sorry

end circle_center_sum_l1953_195300


namespace number_of_teams_l1953_195335

theorem number_of_teams (n : ℕ) (h1 : ∀ k, k = 10) (h2 : n * 10 * (n - 1) / 2 = 1900) : n = 20 :=
by
  sorry

end number_of_teams_l1953_195335


namespace inequality_abc_l1953_195353

theorem inequality_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
    a^2 + b^2 + c^2 + 3 ≥ (1 / a) + (1 / b) + (1 / c) + a + b + c :=
sorry

end inequality_abc_l1953_195353


namespace boat_speed_in_still_water_l1953_195366

/-- Given a boat's speed along the stream and against the stream, prove its speed in still water. -/
theorem boat_speed_in_still_water (b s : ℝ) 
  (h1 : b + s = 11)
  (h2 : b - s = 5) : b = 8 :=
sorry

end boat_speed_in_still_water_l1953_195366


namespace rectangular_prism_diagonals_l1953_195329

theorem rectangular_prism_diagonals (length width height : ℕ) (length_eq : length = 4) (width_eq : width = 3) (height_eq : height = 2) : 
  ∃ (total_diagonals : ℕ), total_diagonals = 16 :=
by
  let face_diagonals := 12
  let space_diagonals := 4
  let total_diagonals := face_diagonals + space_diagonals
  use total_diagonals
  sorry

end rectangular_prism_diagonals_l1953_195329


namespace solution_set_of_inequality_l1953_195390

theorem solution_set_of_inequality (x : ℝ) : 
  (|x - 1| + |x - 2| ≥ 5) ↔ (x ≤ -1 ∨ x ≥ 4) :=
by sorry

end solution_set_of_inequality_l1953_195390


namespace largest_n_l1953_195328

noncomputable def a (n : ℕ) (x : ℤ) : ℤ := 2 + (n - 1) * x
noncomputable def b (n : ℕ) (y : ℤ) : ℤ := 3 + (n - 1) * y

theorem largest_n {n : ℕ} (x y : ℤ) :
  a 1 x = 2 ∧ b 1 y = 3 ∧ 3 * a 2 x < 2 * b 2 y ∧ a n x * b n y = 4032 →
  n = 367 :=
sorry

end largest_n_l1953_195328


namespace find_x_l1953_195315

theorem find_x (x : ℝ) (h1 : x ≠ 0) (h2 : x = (1 / x) * (-x) + 3) : x = 2 :=
by
  sorry

end find_x_l1953_195315


namespace smallest_divisor_of_2880_that_gives_perfect_square_is_5_l1953_195337

theorem smallest_divisor_of_2880_that_gives_perfect_square_is_5 :
  (∃ x : ℕ, x ≠ 0 ∧ 2880 % x = 0 ∧ (∃ y : ℕ, 2880 / x = y * y) ∧ x = 5) := by
  sorry

end smallest_divisor_of_2880_that_gives_perfect_square_is_5_l1953_195337


namespace num_digits_of_prime_started_numerals_l1953_195347

theorem num_digits_of_prime_started_numerals (n : ℕ) (h : 4 * 10^(n-1) = 400) : n = 3 := 
  sorry

end num_digits_of_prime_started_numerals_l1953_195347


namespace pool_filling_time_l1953_195351

theorem pool_filling_time :
  let pool_capacity := 12000 -- in cubic meters
  let first_valve_time := 120 -- in minutes
  let first_valve_rate := pool_capacity / first_valve_time -- in cubic meters per minute
  let second_valve_rate := first_valve_rate + 50 -- in cubic meters per minute
  let combined_rate := first_valve_rate + second_valve_rate -- in cubic meters per minute
  let time_to_fill := pool_capacity / combined_rate -- in minutes
  time_to_fill = 48 :=
by
  sorry

end pool_filling_time_l1953_195351


namespace positional_relationship_l1953_195387

theorem positional_relationship (r PO QO : ℝ) (h_r : r = 6) (h_PO : PO = 4) (h_QO : QO = 6) :
  (PO < r) ∧ (QO = r) :=
by
  sorry

end positional_relationship_l1953_195387


namespace exterior_angle_DEG_l1953_195396

-- Define the degree measures of angles in a square and a pentagon.
def square_interior_angle := 90
def pentagon_interior_angle := 108

-- Define the sum of the adjacent interior angles at D
def adjacent_interior_sum := square_interior_angle + pentagon_interior_angle

-- Statement to prove the exterior angle DEG
theorem exterior_angle_DEG :
  360 - adjacent_interior_sum = 162 := by
  sorry

end exterior_angle_DEG_l1953_195396


namespace a2018_is_4035_l1953_195346

noncomputable def f : ℝ → ℝ := sorry
def a (n : ℕ) : ℝ := sorry

axiom domain : ∀ x : ℝ, true 
axiom condition_2 : ∀ x : ℝ, x < 0 → f x > 1
axiom condition_3 : ∀ x y : ℝ, f x * f y = f (x + y)
axiom sequence_def : ∀ n : ℕ, n > 0 → a 1 = f 0 ∧ f (a (n + 1)) = 1 / f (-2 - a n)

theorem a2018_is_4035 : a 2018 = 4035 :=
sorry

end a2018_is_4035_l1953_195346


namespace total_amount_spent_on_cookies_l1953_195349

def days_in_april : ℕ := 30
def cookies_per_day : ℕ := 3
def cost_per_cookie : ℕ := 18

theorem total_amount_spent_on_cookies : days_in_april * cookies_per_day * cost_per_cookie = 1620 := by
  sorry

end total_amount_spent_on_cookies_l1953_195349
