import Mathlib

namespace desired_depth_l1128_112815

-- Define the given conditions
def men_hours_30m (d : ℕ) : ℕ := 18 * 8 * d
def men_hours_Dm (d1 : ℕ) (D : ℕ) : ℕ := 40 * 6 * d1

-- Define the proportion
def proportion (d d1 : ℕ) (D : ℕ) : Prop :=
  (men_hours_30m d) / 30 = (men_hours_Dm d1 D) / D

-- The main theorem to prove the desired depth
theorem desired_depth (d d1 : ℕ) (H : proportion d d1 50) : 50 = 50 :=
by sorry

end desired_depth_l1128_112815


namespace factorization_correct_l1128_112859

theorem factorization_correct (a x y : ℝ) : a * x - a * y = a * (x - y) := by sorry

end factorization_correct_l1128_112859


namespace tan_alpha_in_third_quadrant_l1128_112807

theorem tan_alpha_in_third_quadrant (α : Real) (h1 : Real.sin α = -5/13) (h2 : ∃ k : ℕ, π < α + k * 2 * π ∧ α + k * 2 * π < 3 * π) : 
  Real.tan α = 5/12 :=
sorry

end tan_alpha_in_third_quadrant_l1128_112807


namespace train_ticket_product_l1128_112811

theorem train_ticket_product
  (a b c d e : ℕ)
  (h1 : b = a + 1)
  (h2 : c = a + 2)
  (h3 : d = a + 3)
  (h4 : e = a + 4)
  (h_sum : a + b + c + d + e = 120) :
  a * b * c * d * e = 7893600 :=
sorry

end train_ticket_product_l1128_112811


namespace ingrid_tax_rate_proof_l1128_112868

namespace TaxProblem

-- Define the given conditions
def john_income : ℝ := 56000
def ingrid_income : ℝ := 72000
def combined_income := john_income + ingrid_income

def john_tax_rate : ℝ := 0.30
def combined_tax_rate : ℝ := 0.35625

-- Calculate John's tax
def john_tax := john_tax_rate * john_income

-- Calculate total tax paid
def total_tax_paid := combined_tax_rate * combined_income

-- Calculate Ingrid's tax
def ingrid_tax := total_tax_paid - john_tax

-- Prove Ingrid's tax rate
theorem ingrid_tax_rate_proof (r : ℝ) :
  (ingrid_tax / ingrid_income) * 100 = 40 :=
  by sorry

end TaxProblem

end ingrid_tax_rate_proof_l1128_112868


namespace relationship_among_a_b_c_l1128_112895

noncomputable def a : ℝ := (1 / 2) ^ (3 / 4)
noncomputable def b : ℝ := (3 / 4) ^ (1 / 2)
noncomputable def c : ℝ := Real.log 3 / Real.log 2

theorem relationship_among_a_b_c : a < b ∧ b < c := 
by
  -- Skipping the proof steps
  sorry

end relationship_among_a_b_c_l1128_112895


namespace initial_salt_percentage_l1128_112839

theorem initial_salt_percentage (initial_mass : ℝ) (added_salt_mass : ℝ) (final_solution_percentage : ℝ) (final_mass : ℝ) 
  (h1 : initial_mass = 100) 
  (h2 : added_salt_mass = 38.46153846153846) 
  (h3 : final_solution_percentage = 0.35) 
  (h4 : final_mass = 138.46153846153846) : 
  ((10 / 100) * 100) = 10 := 
sorry

end initial_salt_percentage_l1128_112839


namespace new_tax_rate_l1128_112825

theorem new_tax_rate
  (old_rate : ℝ) (income : ℝ) (savings : ℝ) (new_rate : ℝ)
  (h1 : old_rate = 0.46)
  (h2 : income = 36000)
  (h3 : savings = 5040)
  (h4 : new_rate = (income * old_rate - savings) / income) :
  new_rate = 0.32 :=
by {
  sorry
}

end new_tax_rate_l1128_112825


namespace functional_eq_solution_l1128_112843

-- Define the conditions
variables (f g : ℕ → ℕ)

-- Define the main theorem
theorem functional_eq_solution :
  (∀ n : ℕ, f n + f (n + g n) = f (n + 1)) →
  ( (∀ n, f n = 0) ∨ 
    (∃ (n₀ c : ℕ), 
      (∀ n < n₀, f n = 0) ∧ 
      (∀ n ≥ n₀, f n = c * 2^(n - n₀)) ∧
      (∀ n < n₀ - 1, ∃ ck : ℕ, g n = ck) ∧
      g (n₀ - 1) = 1 ∧
      ∀ n ≥ n₀, g n = 0 ) ) := 
by
  intro h
  /- Proof goes here -/
  sorry

end functional_eq_solution_l1128_112843


namespace savings_after_expense_increase_l1128_112823

-- Define constants and initial conditions
def salary : ℝ := 7272.727272727273
def savings_rate : ℝ := 0.10
def expense_increase_rate : ℝ := 0.05

-- Define initial savings, expenses, and new expenses
def initial_savings : ℝ := savings_rate * salary
def initial_expenses : ℝ := salary - initial_savings
def new_expenses : ℝ := initial_expenses * (1 + expense_increase_rate)
def new_savings : ℝ := salary - new_expenses

-- The theorem statement
theorem savings_after_expense_increase : new_savings = 400 := by
  sorry

end savings_after_expense_increase_l1128_112823


namespace pentagon_angles_l1128_112851

def is_point_in_convex_pentagon (O A B C D E : Point) : Prop := sorry
def angle (A B C : Point) : ℝ := sorry -- Assume definition of angle in radians

theorem pentagon_angles (O A B C D E: Point) (hO : is_point_in_convex_pentagon O A B C D E)
  (h1: angle A O B = angle B O C) (h2: angle B O C = angle C O D)
  (h3: angle C O D = angle D O E) (h4: angle D O E = angle E O A) :
  (angle E O A = angle A O B) ∨ (angle E O A + angle A O B = π) :=
sorry

end pentagon_angles_l1128_112851


namespace diminished_radius_10_percent_l1128_112829

theorem diminished_radius_10_percent
  (r r' : ℝ) 
  (h₁ : r > 0)
  (h₂ : r' > 0)
  (h₃ : (π * r'^2) = 0.8100000000000001 * (π * r^2)) :
  r' = 0.9 * r :=
by sorry

end diminished_radius_10_percent_l1128_112829


namespace scientific_notation_4040000_l1128_112870

theorem scientific_notation_4040000 :
  (4040000 : ℝ) = 4.04 * (10 : ℝ)^6 :=
by
  sorry

end scientific_notation_4040000_l1128_112870


namespace sum_of_roots_combined_eq_five_l1128_112858

noncomputable def sum_of_roots_poly1 : ℝ :=
-(-9/3)

noncomputable def sum_of_roots_poly2 : ℝ :=
-(-8/4)

theorem sum_of_roots_combined_eq_five :
  sum_of_roots_poly1 + sum_of_roots_poly2 = 5 :=
by
  sorry

end sum_of_roots_combined_eq_five_l1128_112858


namespace graph_of_equation_l1128_112864

theorem graph_of_equation (x y : ℝ) : 
  (x - y)^2 = x^2 + y^2 ↔ (x = 0 ∨ y = 0) := 
by 
  sorry

end graph_of_equation_l1128_112864


namespace greatest_n_le_5_value_ge_2525_l1128_112854

theorem greatest_n_le_5_value_ge_2525 (n : ℤ) (V : ℤ) 
  (h1 : 101 * n^2 ≤ V) 
  (h2 : ∀ k : ℤ, (101 * k^2 ≤ V) → (k ≤ 5)) : 
  V ≥ 2525 := 
sorry

end greatest_n_le_5_value_ge_2525_l1128_112854


namespace probability_of_event_l1128_112847

-- Definitions for the problem setup

-- Box C and its range
def boxC := {i : ℕ | 1 ≤ i ∧ i ≤ 30}

-- Box D and its range
def boxD := {i : ℕ | 21 ≤ i ∧ i ≤ 50}

-- Condition for a tile from box C being less than 20
def tile_from_C_less_than_20 (i : ℕ) : Prop := i ∈ boxC ∧ i < 20

-- Condition for a tile from box D being odd or greater than 45
def tile_from_D_odd_or_greater_than_45 (i : ℕ) : Prop := i ∈ boxD ∧ (i % 2 = 1 ∨ i > 45)

-- Main statement
theorem probability_of_event :
  (19 / 30 : ℚ) * (17 / 30 : ℚ) = (323 / 900 : ℚ) :=
by sorry

end probability_of_event_l1128_112847


namespace find_x_l1128_112828

theorem find_x (x y : ℤ) (h₁ : x / y = 12 / 5) (h₂ : y = 25) : x = 60 :=
by
  sorry

end find_x_l1128_112828


namespace calculate_expression_l1128_112876

theorem calculate_expression : 
  3 * 995 + 4 * 996 + 5 * 997 + 6 * 998 + 7 * 999 - 4985 * 3 = 9980 :=
by
  sorry

end calculate_expression_l1128_112876


namespace find_multiple_l1128_112898

theorem find_multiple (x m : ℤ) (hx : x = 13) (h : x + x + 2 * x + m * x = 104) : m = 4 :=
by
  -- Proof to be provided
  sorry

end find_multiple_l1128_112898


namespace range_of_w_l1128_112836

noncomputable def f (w x : ℝ) : ℝ := Real.sin (w * x) - Real.sqrt 3 * Real.cos (w * x)

theorem range_of_w (w : ℝ) (h_w : 0 < w) :
  (∀ f_zeros : Finset ℝ, ∀ x ∈ f_zeros, (0 < x ∧ x < Real.pi) → f w x = 0 → f_zeros.card = 2) ↔
  (4 / 3 < w ∧ w ≤ 7 / 3) :=
by sorry

end range_of_w_l1128_112836


namespace sum_of_roots_of_quadratic_l1128_112837

theorem sum_of_roots_of_quadratic :
  let a := 2
  let b := -8
  let c := 6
  let sum_of_roots := (-b / a)
  2 * (sum_of_roots) * sum_of_roots - 8 * sum_of_roots + 6 = 0 :=
by
  sorry

end sum_of_roots_of_quadratic_l1128_112837


namespace volume_of_one_pizza_piece_l1128_112831

theorem volume_of_one_pizza_piece
  (h : ℝ) (d : ℝ) (n : ℕ)
  (h_eq : h = 1 / 2)
  (d_eq : d = 16)
  (n_eq : n = 16) :
  ((π * (d / 2)^2 * h) / n) = 2 * π :=
by
  rw [h_eq, d_eq, n_eq]
  sorry

end volume_of_one_pizza_piece_l1128_112831


namespace hyperbola_equation_l1128_112841

noncomputable def focal_distance : ℝ := 10
noncomputable def c : ℝ := 5
noncomputable def point_P : (ℝ × ℝ) := (2, 1)
noncomputable def eq1 : Prop := ∀ (x y : ℝ), (x^2) / 20 - (y^2) / 5 = 1 ↔ c = 5 ∧ focal_distance = 10 ∧ point_P.1 = 2 ∧ point_P.2 = 1
noncomputable def eq2 : Prop := ∀ (x y : ℝ), (y^2) / 5 - (x^2) / 20 = 1 ↔ c = 5 ∧ focal_distance = 10 ∧ point_P.1 = 2 ∧ point_P.2 = 1

theorem hyperbola_equation :
  (∃ a b : ℝ, c = 5 ∧ focal_distance = 10 ∧ point_P.1 = 2 ∧ point_P.2 = 1 ∧ 
    (∀ x y : ℝ, (x^2) / a^2 - (y^2) / b^2 = 1) ∨ 
    (∃ a' b' : ℝ, c = 5 ∧ focal_distance = 10 ∧ point_P.1 = 2 ∧ point_P.2 = 1 ∧ 
      (∀ x y : ℝ, (y^2) / a'^2 - (x^2) / b'^2 = 1))) :=
by sorry

end hyperbola_equation_l1128_112841


namespace compare_cubic_terms_l1128_112856

theorem compare_cubic_terms (a b : ℝ) :
    (a ≥ b → a^3 - b^3 ≥ a * b^2 - a^2 * b) ∧
    (a < b → a^3 - b^3 ≤ a * b^2 - a^2 * b) :=
by sorry

end compare_cubic_terms_l1128_112856


namespace angles_sum_eq_l1128_112887

variables {a b c : ℝ} {A B C : ℝ}

theorem angles_sum_eq {a b c : ℝ} {A B C : ℝ}
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : A > 0) (h5 : B > 0) (h6 : C > 0)
  (h7 : A + B + C = π)
  (h8 : (a + c - b) * (a + c + b) = 3 * a * c) :
  A + C = 2 * π / 3 :=
sorry

end angles_sum_eq_l1128_112887


namespace find_number_l1128_112874

theorem find_number (some_number : ℤ) (h : some_number + 9 = 54) : some_number = 45 :=
sorry

end find_number_l1128_112874


namespace equivalent_problem_l1128_112881

noncomputable def problem_statement : Prop :=
  ∀ (a b c d : ℝ), a ≠ -1 → b ≠ -1 → c ≠ -1 → d ≠ -1 →
  ∀ (ω : ℂ), ω^4 = 1 → ω ≠ 1 →
  (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 4 / (1 + ω)) →
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 2)

#check problem_statement

-- Expected output for type checking without providing the proof
theorem equivalent_problem : problem_statement :=
  sorry

end equivalent_problem_l1128_112881


namespace x_add_y_eq_neg_one_l1128_112812

theorem x_add_y_eq_neg_one (x y : ℝ) (h : |x + 3| + (y - 2)^2 = 0) : x + y = -1 :=
by sorry

end x_add_y_eq_neg_one_l1128_112812


namespace find_g_l1128_112830

variable (x : ℝ)

-- Given condition
def given_condition (g : ℝ → ℝ) : Prop :=
  5 * x^5 + 3 * x^3 - 4 * x + 2 + g x = 7 * x^3 - 9 * x^2 + x + 5

-- Goal
def goal (g : ℝ → ℝ) : Prop :=
  g x = -5 * x^5 + 4 * x^3 - 9 * x^2 + 5 * x + 3

-- The statement combining given condition and goal to prove
theorem find_g (g : ℝ → ℝ) (h : given_condition x g) : goal x g :=
by
  sorry

end find_g_l1128_112830


namespace slope_of_line_l1128_112846

theorem slope_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) : (- (4 : ℝ) / 7) = -4 / 7 :=
by
  -- Sorry for the proof for completeness
  sorry

end slope_of_line_l1128_112846


namespace valid_word_combinations_l1128_112866

-- Definition of valid_combination based on given conditions
def valid_combination : ℕ :=
  26 * 5 * 26

-- Statement to prove the number of valid four-letter combinations is 3380
theorem valid_word_combinations : valid_combination = 3380 := by
  sorry

end valid_word_combinations_l1128_112866


namespace squirrel_spiral_path_height_l1128_112897

-- Define the conditions
def spiralPath (circumference rise totalDistance : ℝ) : Prop :=
  ∃ (numberOfCircuits : ℝ), numberOfCircuits = totalDistance / circumference ∧ numberOfCircuits * rise = totalDistance

-- Define the height of the post proof
theorem squirrel_spiral_path_height : 
  let circumference := 2 -- feet
  let rise := 4 -- feet
  let totalDistance := 8 -- feet
  let height := 16 -- feet
  spiralPath circumference rise totalDistance → height = (totalDistance / circumference) * rise :=
by
  intro h
  sorry

end squirrel_spiral_path_height_l1128_112897


namespace smallest_consecutive_sum_l1128_112845

theorem smallest_consecutive_sum (x : ℤ) (h : x + (x + 1) + (x + 2) = 90) : x = 29 :=
by 
  sorry

end smallest_consecutive_sum_l1128_112845


namespace union_setA_setB_l1128_112808

noncomputable def setA : Set ℝ := { x : ℝ | 2 / (x + 1) ≥ 1 }
noncomputable def setB : Set ℝ := { y : ℝ | ∃ x : ℝ, y = 2^x ∧ x < 0 }

theorem union_setA_setB : setA ∪ setB = { x : ℝ | -1 < x ∧ x ≤ 1 } :=
by
  sorry

end union_setA_setB_l1128_112808


namespace exists_fraction_x_only_and_f_of_1_is_0_l1128_112848

theorem exists_fraction_x_only_and_f_of_1_is_0 : ∃ f : ℚ → ℚ, (∀ x : ℚ, f x = (x - 1) / x) ∧ f 1 = 0 := 
by
  sorry

end exists_fraction_x_only_and_f_of_1_is_0_l1128_112848


namespace quadratic_expression_positive_intervals_l1128_112818

noncomputable def quadratic_expression (x : ℝ) : ℝ := (x + 3) * (x - 1)
def interval_1 (x : ℝ) : Prop := x < (1 - Real.sqrt 13) / 2
def interval_2 (x : ℝ) : Prop := x > (1 + Real.sqrt 13) / 2

theorem quadratic_expression_positive_intervals (x : ℝ) :
  quadratic_expression x > 0 ↔ interval_1 x ∨ interval_2 x :=
by {
  sorry
}

end quadratic_expression_positive_intervals_l1128_112818


namespace leftmost_three_nonzero_digits_of_arrangements_l1128_112880

-- Definitions based on the conditions
def num_rings := 10
def chosen_rings := 6
def num_fingers := 5

-- Calculate the possible arrangements
def arrangements : ℕ := Nat.choose num_rings chosen_rings * Nat.factorial chosen_rings * Nat.choose (chosen_rings + (num_fingers - 1)) (num_fingers - 1)

-- Find the leftmost three nonzero digits
def leftmost_three_nonzero_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  (digits.reverse.takeWhile (· > 0)).reverse.take 3
  |> List.foldl (· + · * 10) 0
  
-- The main theorem to prove
theorem leftmost_three_nonzero_digits_of_arrangements :
  leftmost_three_nonzero_digits arrangements = 317 :=
by
  sorry

end leftmost_three_nonzero_digits_of_arrangements_l1128_112880


namespace Betty_flies_caught_in_morning_l1128_112891

-- Definitions from the conditions
def total_flies_needed_in_a_week : ℕ := 14
def flies_eaten_per_day : ℕ := 2
def days_in_a_week : ℕ := 7
def flies_caught_in_morning (X : ℕ) : ℕ := X
def flies_caught_in_afternoon : ℕ := 6
def flies_escaped : ℕ := 1
def flies_short : ℕ := 4

-- Given statement in Lean 4
theorem Betty_flies_caught_in_morning (X : ℕ) 
  (h1 : flies_caught_in_morning X + flies_caught_in_afternoon - flies_escaped = total_flies_needed_in_a_week - flies_short) : 
  X = 5 :=
by
  sorry

end Betty_flies_caught_in_morning_l1128_112891


namespace find_fourth_number_l1128_112816

theorem find_fourth_number 
  (average : ℝ) 
  (a1 a2 a3 : ℝ) 
  (x : ℝ) 
  (n : ℝ) 
  (h1 : average = 20) 
  (h2 : a1 = 3) 
  (h3 : a2 = 16) 
  (h4 : a3 = 33) 
  (h5 : n = 27) 
  (h_avg : (a1 + a2 + a3 + x) / 4 = average) :
  x = n + 1 :=
by
  sorry

end find_fourth_number_l1128_112816


namespace walkways_area_l1128_112833

-- Define the conditions and prove the total walkway area is 416 square feet
theorem walkways_area (rows : ℕ) (columns : ℕ) (bed_width : ℝ) (bed_height : ℝ) (walkway_width : ℝ) 
  (h_rows : rows = 4) (h_columns : columns = 3) (h_bed_width : bed_width = 8) (h_bed_height : bed_height = 3) (h_walkway_width : walkway_width = 2) : 
  (rows * (bed_height + walkway_width) + walkway_width) * (columns * (bed_width + walkway_width) + walkway_width) - rows * columns * bed_width * bed_height = 416 := 
by 
  sorry

end walkways_area_l1128_112833


namespace range_of_a_l1128_112871

def solution_set_non_empty (a : ℝ) : Prop :=
  ∃ x : ℝ, |x - 3| + |x - 4| < a

theorem range_of_a (a : ℝ) : solution_set_non_empty a ↔ a > 1 := sorry

end range_of_a_l1128_112871


namespace vendor_sells_50_percent_on_first_day_l1128_112877

variables (A : ℝ) (S : ℝ)

theorem vendor_sells_50_percent_on_first_day 
  (h : 0.2 * A * (1 - S) + 0.5 * A * (1 - S) * 0.8 = 0.3 * A) : S = 0.5 :=
  sorry

end vendor_sells_50_percent_on_first_day_l1128_112877


namespace compare_abc_l1128_112893

noncomputable def a := Real.exp (Real.sqrt 2)
noncomputable def b := 2 + Real.sqrt 2
noncomputable def c := Real.log (12 + 6 * Real.sqrt 2)

theorem compare_abc : a > b ∧ b > c :=
by
  sorry

end compare_abc_l1128_112893


namespace range_of_m_l1128_112822

-- Definition of the quadratic function
def quadratic_function (m x : ℝ) : ℝ :=
  x^2 + (m - 1) * x + 1

-- Statement of the proof problem in Lean
theorem range_of_m (m : ℝ) : 
  (∀ x : ℤ, 0 ≤ x ∧ x ≤ 5 → quadratic_function m x ≥ quadratic_function m (x + 1)) ↔ m ≤ -8 :=
by
  sorry

end range_of_m_l1128_112822


namespace identify_incorrect_propositions_l1128_112878

-- Definitions for parallel lines and planes
def line := Type -- Define a line type
def plane := Type -- Define a plane type
def parallel_to (l1 l2 : line) : Prop := sorry -- Assume a definition for parallel lines
def parallel_to_plane (l : line) (pl : plane) : Prop := sorry -- Assume a definition for a line parallel to a plane
def contained_in (l : line) (pl : plane) : Prop := sorry -- Assume a definition for a line contained in a plane

theorem identify_incorrect_propositions (a b : line) (α : plane) :
  (parallel_to_plane a α ∧ parallel_to_plane b α → ¬parallel_to a b) ∧
  (parallel_to_plane a α ∧ contained_in b α → ¬parallel_to a b) ∧
  (parallel_to a b ∧ contained_in b α → ¬parallel_to_plane a α) ∧
  (parallel_to a b ∧ parallel_to_plane b α → ¬parallel_to_plane a α) :=
by
  sorry -- The proof is not required

end identify_incorrect_propositions_l1128_112878


namespace higher_selling_price_is_463_l1128_112863

-- Definitions and conditions
def cost_price : ℝ := 400
def selling_price_340 : ℝ := 340
def loss_340 : ℝ := selling_price_340 - cost_price
def gain_percent : ℝ := 0.05
def additional_gain : ℝ := gain_percent * -loss_340
def expected_gain := -loss_340 + additional_gain

-- Theorem to prove that the higher selling price is 463
theorem higher_selling_price_is_463 : ∃ P : ℝ, P = cost_price + expected_gain ∧ P = 463 :=
by
  sorry

end higher_selling_price_is_463_l1128_112863


namespace nine_pow_div_eighty_one_pow_l1128_112820

theorem nine_pow_div_eighty_one_pow (a b : ℕ) (h1 : a = 9^2) (h2 : b = a^4) :
  (9^10 / b = 81) := by
  sorry

end nine_pow_div_eighty_one_pow_l1128_112820


namespace total_waiting_time_difference_l1128_112860

theorem total_waiting_time_difference :
  let n_swings := 6
  let n_slide := 4 * n_swings
  let t_swings := 3.5 * 60
  let t_slide := 45
  let T_swings := n_swings * t_swings
  let T_slide := n_slide * t_slide
  let T_difference := T_swings - T_slide
  T_difference = 180 :=
by
  sorry

end total_waiting_time_difference_l1128_112860


namespace gcf_lcm_problem_l1128_112882

def GCF (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem gcf_lcm_problem :
  GCF (LCM 9 15) (LCM 10 21) = 15 := by
  sorry

end gcf_lcm_problem_l1128_112882


namespace james_missing_legos_l1128_112806

theorem james_missing_legos  (h1 : 500 > 0) (h2 : 500 % 2 = 0) (h3 : 245 < 500)  :
  let total_legos := 500
  let used_legos := total_legos / 2
  let leftover_legos := total_legos - used_legos
  let legos_in_box := 245
  leftover_legos - legos_in_box = 5 := by
{
  sorry
}

end james_missing_legos_l1128_112806


namespace polynomial_coeff_sum_l1128_112838

noncomputable def polynomial_expansion (x : ℝ) :=
  (2 * x + 3) * (4 * x^3 - 2 * x^2 + x - 7)

theorem polynomial_coeff_sum :
  let A := 8
  let B := 8
  let C := -4
  let D := -11
  let E := -21
  A + B + C + D + E = -20 :=
by
  -- The following proof steps are skipped
  sorry

end polynomial_coeff_sum_l1128_112838


namespace exist_pos_integers_m_n_l1128_112853

def d (n : ℕ) : ℕ :=
  -- Number of divisors of n
  sorry 

theorem exist_pos_integers_m_n :
  ∃ (m n : ℕ), (m > 0) ∧ (n > 0) ∧ (m = 24) ∧ 
  ((∃ (triples : Finset (ℕ × ℕ × ℕ)),
    (∀ (a b c : ℕ), (a, b, c) ∈ triples ↔ (0 < a) ∧ (a < b) ∧ (b < c) ∧ (c ≤ m) ∧ (d (n + a) * d (n + b) * d (n + c)) % (a * b * c) = 0) ∧ 
    (triples.card = 2024))) :=
sorry

end exist_pos_integers_m_n_l1128_112853


namespace inscribed_sphere_l1128_112801

theorem inscribed_sphere (r_base height : ℝ) (r_sphere b d : ℝ)
  (h_base : r_base = 15)
  (h_height : height = 20)
  (h_sphere : r_sphere = b * Real.sqrt d - b)
  (h_rsphere_eq : r_sphere = 120 / 11) : 
  b + d = 12 := 
sorry

end inscribed_sphere_l1128_112801


namespace rectangle_length_is_16_l1128_112826

noncomputable def rectangle_length (b : ℝ) (c : ℝ) : ℝ :=
  let pi := Real.pi
  let full_circle_circumference := 2 * c
  let radius := full_circle_circumference / (2 * pi)
  let diameter := 2 * radius
  let side_length_of_square := diameter
  let perimeter_of_square := 4 * side_length_of_square
  let perimeter_of_rectangle := perimeter_of_square
  let length_of_rectangle := (perimeter_of_rectangle / 2) - b
  length_of_rectangle

theorem rectangle_length_is_16 :
  rectangle_length 14 23.56 = 16 :=
by
  sorry

end rectangle_length_is_16_l1128_112826


namespace pictures_per_album_l1128_112896

-- Define the conditions
def uploaded_pics_phone : ℕ := 22
def uploaded_pics_camera : ℕ := 2
def num_albums : ℕ := 4

-- Define the total pictures uploaded
def total_pictures : ℕ := uploaded_pics_phone + uploaded_pics_camera

-- Define the target statement as the theorem
theorem pictures_per_album : (total_pictures / num_albums) = 6 := by
  sorry

end pictures_per_album_l1128_112896


namespace find_fraction_of_number_l1128_112886

theorem find_fraction_of_number (N : ℚ) (h : (3/10 : ℚ) * N - 8 = 12) :
  (1/5 : ℚ) * N = 40 / 3 :=
by
  sorry

end find_fraction_of_number_l1128_112886


namespace max_watched_hours_l1128_112803

-- Define the duration of one episode in minutes
def episode_duration : ℕ := 30

-- Define the number of weekdays Max watched the show
def weekdays_watched : ℕ := 4

-- Define the total minutes Max watched
def total_minutes_watched : ℕ := episode_duration * weekdays_watched

-- Define the conversion factor from minutes to hours
def minutes_to_hours_factor : ℕ := 60

-- Define the total hours watched
def total_hours_watched : ℕ := total_minutes_watched / minutes_to_hours_factor

-- Proof statement
theorem max_watched_hours : total_hours_watched = 2 :=
by
  sorry

end max_watched_hours_l1128_112803


namespace find_math_books_l1128_112894

theorem find_math_books 
  (M H : ℕ)
  (h1 : M + H = 80)
  (h2 : 4 * M + 5 * H = 390) : 
  M = 10 := 
by 
  sorry

end find_math_books_l1128_112894


namespace range_of_t_l1128_112805

theorem range_of_t (a b t : ℝ) (h1 : a * (-1)^2 + b * (-1) + 1 / 2 = 0)
    (h2 : (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ y = a * x^2 + b * x + 1 / 2))
    (h3 : t = 2 * a + b) : 
    -1 < t ∧ t < 1 / 2 :=
  sorry

end range_of_t_l1128_112805


namespace vec_op_not_comm_l1128_112852

open Real

-- Define the operation ⊙
def vec_op (a b: ℝ × ℝ) : ℝ :=
  (a.1 * b.2) - (a.2 * b.1)

-- Define a predicate to check if two vectors are collinear
def collinear (a b: ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

-- Define the proof theorem
theorem vec_op_not_comm (a b: ℝ × ℝ) : vec_op a b ≠ vec_op b a :=
by
  -- The contents of the proof will go here. Insert 'sorry' to skip.
  sorry

end vec_op_not_comm_l1128_112852


namespace toy_car_production_l1128_112862

theorem toy_car_production (yesterday today total : ℕ) 
  (hy : yesterday = 60)
  (ht : today = 2 * yesterday) :
  total = yesterday + today :=
by
  sorry

end toy_car_production_l1128_112862


namespace paint_cost_is_624_rs_l1128_112819

-- Given conditions:
-- Length of floor is 21.633307652783934 meters.
-- Length is 200% more than the breadth (i.e., length = 3 * breadth).
-- Cost to paint the floor is Rs. 4 per square meter.

noncomputable def length : ℝ := 21.633307652783934
noncomputable def cost_per_sq_meter : ℝ := 4
noncomputable def breadth : ℝ := length / 3
noncomputable def area : ℝ := length * breadth
noncomputable def total_cost : ℝ := area * cost_per_sq_meter

theorem paint_cost_is_624_rs : total_cost = 624 := by
  sorry

end paint_cost_is_624_rs_l1128_112819


namespace range_of_m_l1128_112821

variable (m : ℝ)

def A (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 3 * m - 1 }
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 10}

theorem range_of_m (h : A m ∪ B = B) : m ≤ 11 / 3 := by
  sorry

end range_of_m_l1128_112821


namespace sum_of_solutions_l1128_112892

theorem sum_of_solutions (s : Finset ℝ) :
  (∀ x ∈ s, |x^2 - 16 * x + 60| = 4) →
  s.sum id = 24 := 
by
  sorry

end sum_of_solutions_l1128_112892


namespace option_D_correct_l1128_112873

theorem option_D_correct (y : ℝ) : -9 * y^2 + 16 * y^2 = 7 * y^2 :=
by sorry

end option_D_correct_l1128_112873


namespace second_number_in_first_set_l1128_112879

theorem second_number_in_first_set :
  ∃ (x : ℝ), (20 + x + 60) / 3 = (10 + 80 + 15) / 3 + 5 ∧ x = 40 :=
by
  use 40
  sorry

end second_number_in_first_set_l1128_112879


namespace positive_difference_between_solutions_l1128_112849

theorem positive_difference_between_solutions : 
  let f (x : ℝ) := (5 - (x^2 / 3 : ℝ))^(1 / 3 : ℝ)
  let a := 4 * Real.sqrt 6
  let b := -4 * Real.sqrt 6
  |a - b| = 8 * Real.sqrt 6 := 
by 
  sorry

end positive_difference_between_solutions_l1128_112849


namespace all_selected_prob_l1128_112835

def probability_of_selection (P_ram P_ravi P_raj : ℚ) : ℚ :=
  P_ram * P_ravi * P_raj

theorem all_selected_prob :
  let P_ram := 2/7
  let P_ravi := 1/5
  let P_raj := 3/8
  probability_of_selection P_ram P_ravi P_raj = 3/140 := by
  sorry

end all_selected_prob_l1128_112835


namespace initial_candies_equal_twenty_l1128_112861

-- Definitions based on conditions
def friends : ℕ := 6
def candies_per_friend : ℕ := 4
def total_needed_candies : ℕ := friends * candies_per_friend
def additional_candies : ℕ := 4

-- Main statement
theorem initial_candies_equal_twenty :
  (total_needed_candies - additional_candies) = 20 := by
  sorry

end initial_candies_equal_twenty_l1128_112861


namespace problem1_problem2_l1128_112832

def f (x : ℝ) : ℝ := abs (x - 1) - abs (x + 3)

-- Proof Problem 1
theorem problem1 (x : ℝ) (h : f x > 2) : x < -2 := sorry

-- Proof Problem 2
theorem problem2 (k : ℝ) (h : ∀ x : ℝ, -3 ≤ x ∧ x ≤ -1 → f x ≤ k * x + 1) : k ≤ -1 := sorry

end problem1_problem2_l1128_112832


namespace rhombus_diagonal_l1128_112810

theorem rhombus_diagonal (d1 d2 : ℝ) (area_tri : ℝ) (h1 : d1 = 15) (h2 : area_tri = 75) :
  (d1 * d2) / 2 = 2 * area_tri → d2 = 20 :=
by
  sorry

end rhombus_diagonal_l1128_112810


namespace clive_can_correct_time_l1128_112800

def can_show_correct_time (hour_hand_angle minute_hand_angle : ℝ) :=
  ∃ θ : ℝ, θ ∈ [0, 360] ∧ hour_hand_angle + θ % 360 = minute_hand_angle + θ % 360

theorem clive_can_correct_time (hour_hand_angle minute_hand_angle : ℝ) :
  can_show_correct_time hour_hand_angle minute_hand_angle :=
sorry

end clive_can_correct_time_l1128_112800


namespace find_minuend_l1128_112875

variable (x y : ℕ)

-- Conditions
axiom h1 : x - y = 8008
axiom h2 : x - 10 * y = 88

-- Theorem statement
theorem find_minuend : x = 8888 :=
by
  sorry

end find_minuend_l1128_112875


namespace correct_propositions_l1128_112890

def line (P: Type) := P → P → Prop  -- A line is a relation between points in a plane

variables (plane1 plane2: Type) -- Define two types representing two planes
variables (P1 P2: plane1) -- Points in plane1
variables (Q1 Q2: plane2) -- Points in plane2

axiom perpendicular_planes : ¬∃ l1 : line plane1, ∀ l2 : line plane2, ¬ (∀ p1 p2, l1 p1 p2 ∧ ∀ q1 q2, l2 q1 q2)

theorem correct_propositions : 3 = 3 := by
  sorry

end correct_propositions_l1128_112890


namespace monotonic_increasing_intervals_max_min_values_l1128_112899

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (2 * x - Real.pi / 3)

theorem monotonic_increasing_intervals (k : ℤ) :
  ∃ (a b : ℝ), a = k * Real.pi - Real.pi / 12 ∧ b = k * Real.pi + 5 * Real.pi / 12 ∧
    ∀ x₁ x₂ : ℝ, a ≤ x₁ ∧ x₁ ≤ x₂ ∧ x₂ ≤ b → f x₁ ≤ f x₂ :=
sorry

theorem max_min_values : ∃ (xmin xmax : ℝ) (fmin fmax : ℝ),
  xmin = 0 ∧ fmin = f 0 ∧ fmin = - Real.sqrt 3 / 2 ∧
  xmax = 5 * Real.pi / 12 ∧ fmax = f (5 * Real.pi / 12) ∧ fmax = 1 ∧
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 →
    fmin ≤ f x ∧ f x ≤ fmax :=
sorry

end monotonic_increasing_intervals_max_min_values_l1128_112899


namespace money_problem_l1128_112827

-- Define the conditions and the required proof
theorem money_problem (B S : ℕ) 
  (h1 : B = 2 * S) -- Condition 1: Brother brought twice as much money as the sister
  (h2 : B - 180 = S - 30) -- Condition 3: Remaining money of brother and sister are equal
  : B = 300 ∧ S = 150 := -- Correct answer to prove
  
sorry -- Placeholder for proof

end money_problem_l1128_112827


namespace hire_charges_paid_by_B_l1128_112889

theorem hire_charges_paid_by_B (total_cost : ℝ) (hours_A hours_B hours_C : ℝ) (b_payment : ℝ) :
  total_cost = 720 ∧ hours_A = 9 ∧ hours_B = 10 ∧ hours_C = 13 ∧ b_payment = (total_cost / (hours_A + hours_B + hours_C)) * hours_B → b_payment = 225 :=
by
  sorry

end hire_charges_paid_by_B_l1128_112889


namespace ratio_of_adult_to_kid_charge_l1128_112867

variable (A : ℝ)  -- Charge for adults

-- Conditions
def kids_charge : ℝ := 3
def num_kids_per_day : ℝ := 8
def num_adults_per_day : ℝ := 10
def weekly_earnings : ℝ := 588
def days_per_week : ℝ := 7

-- Hypothesis for the relationship between charges and total weekly earnings
def total_weekly_earnings_eq : Prop :=
  days_per_week * (num_kids_per_day * kids_charge + num_adults_per_day * A) = weekly_earnings

-- Statement to be proved
theorem ratio_of_adult_to_kid_charge (h : total_weekly_earnings_eq A) : (A / kids_charge) = 2 := 
by 
  sorry

end ratio_of_adult_to_kid_charge_l1128_112867


namespace math_problem_proof_l1128_112817

-- Define the base conversion functions
def base11_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 2471 => 1 * 11^0 + 7 * 11^1 + 4 * 11^2 + 2 * 11^3
  | _    => 0

def base5_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 121 => 1 * 5^0 + 2 * 5^1 + 1 * 5^2
  | _   => 0

def base7_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 3654 => 4 * 7^0 + 5 * 7^1 + 6 * 7^2 + 3 * 7^3
  | _    => 0

def base8_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 5680 => 0 * 8^0 + 8 * 8^1 + 6 * 8^2 + 5 * 8^3
  | _    => 0

theorem math_problem_proof :
  let x := base11_to_base10 2471
  let y := base5_to_base10 121
  let z := base7_to_base10 3654
  let w := base8_to_base10 5680
  x / y - z + w = 1736 :=
by
  sorry

end math_problem_proof_l1128_112817


namespace ratio_of_ages_is_six_l1128_112834

-- Definitions of ages
def Cody_age : ℕ := 14
def Grandmother_age : ℕ := 84

-- The ratio we want to prove
def age_ratio : ℕ := Grandmother_age / Cody_age

-- The theorem stating the ratio is 6
theorem ratio_of_ages_is_six : age_ratio = 6 :=
by
  -- Placeholder for the proof
  sorry

end ratio_of_ages_is_six_l1128_112834


namespace find_exponent_l1128_112869

theorem find_exponent (n : ℝ) (hn: (3:ℝ)^n = Real.sqrt 3) : n = 1 / 2 :=
by sorry

end find_exponent_l1128_112869


namespace three_point_three_six_as_fraction_l1128_112814

theorem three_point_three_six_as_fraction : 3.36 = (84 : ℚ) / 25 := 
by
  sorry

end three_point_three_six_as_fraction_l1128_112814


namespace find_abc_l1128_112884

theorem find_abc (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : a^3 + b^3 + c^3 = 2001 → (a = 10 ∧ b = 10 ∧ c = 1) ∨ (a = 10 ∧ b = 1 ∧ c = 10) ∨ (a = 1 ∧ b = 10 ∧ c = 10) := 
sorry

end find_abc_l1128_112884


namespace find_number_l1128_112872

theorem find_number (x : ℝ) (h : 50 + 5 * 12 / (x / 3) = 51) : x = 180 := 
by 
  sorry

end find_number_l1128_112872


namespace fraction_eval_l1128_112865

theorem fraction_eval :
  (8 : ℝ) / (4 * 25) = (0.8 : ℝ) / (0.4 * 25) :=
sorry

end fraction_eval_l1128_112865


namespace number_of_ordered_triples_l1128_112844

theorem number_of_ordered_triples (a b c : ℤ) : 
  ∃ (n : ℕ), -31 <= a ∧ a <= 31 ∧ -31 <= b ∧ b <= 31 ∧ -31 <= c ∧ c <= 31 ∧ 
  (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ (a + b + c > 0) ∧ n = 117690 :=
by sorry

end number_of_ordered_triples_l1128_112844


namespace solve_natural_numbers_system_l1128_112885

theorem solve_natural_numbers_system :
  ∃ a b c : ℕ, (a^3 - b^3 - c^3 = 3 * a * b * c) ∧ (a^2 = 2 * (a + b + c)) ∧
  ((a = 4 ∧ b = 1 ∧ c = 3) ∨ (a = 4 ∧ b = 2 ∧ c = 2) ∨ (a = 4 ∧ b = 3 ∧ c = 1)) :=
by
  sorry

end solve_natural_numbers_system_l1128_112885


namespace ratio_of_Lev_to_Akeno_l1128_112855

theorem ratio_of_Lev_to_Akeno (L : ℤ) (A : ℤ) (Ambrocio : ℤ) :
  A = 2985 ∧ Ambrocio = L - 177 ∧ A = L + Ambrocio + 1172 → L / A = 1 / 3 :=
by
  intro h
  sorry

end ratio_of_Lev_to_Akeno_l1128_112855


namespace num_palindromes_is_correct_l1128_112813

section Palindromes

def num_alphanumeric_chars : ℕ := 10 + 26

def num_four_char_palindromes : ℕ := num_alphanumeric_chars * num_alphanumeric_chars

theorem num_palindromes_is_correct : num_four_char_palindromes = 1296 :=
by
  sorry

end Palindromes

end num_palindromes_is_correct_l1128_112813


namespace total_amount_spent_l1128_112857

def speakers : ℝ := 118.54
def new_tires : ℝ := 106.33
def window_tints : ℝ := 85.27
def seat_covers : ℝ := 79.99
def scheduled_maintenance : ℝ := 199.75
def steering_wheel_cover : ℝ := 15.63
def air_fresheners_set : ℝ := 12.96
def car_wash : ℝ := 25.0

theorem total_amount_spent :
  speakers + new_tires + window_tints + seat_covers + scheduled_maintenance + steering_wheel_cover + air_fresheners_set + car_wash = 643.47 :=
by
  sorry

end total_amount_spent_l1128_112857


namespace denote_below_warning_level_l1128_112802

-- Conditions
def warning_water_level : ℝ := 905.7
def exceed_by_10 : ℝ := 10
def below_by_5 : ℝ := -5

-- Problem statement
theorem denote_below_warning_level : below_by_5 = -5 := 
by
  sorry

end denote_below_warning_level_l1128_112802


namespace max_value_of_expression_l1128_112840

theorem max_value_of_expression (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_product : a * b * c = 16) : 
  a^b - b^c + c^a ≤ 263 :=
sorry

end max_value_of_expression_l1128_112840


namespace savings_account_final_amount_l1128_112809

noncomputable def final_amount (P R : ℝ) (t : ℕ) : ℝ :=
  P * (1 + R) ^ t

theorem savings_account_final_amount :
  final_amount 2500 0.06 21 = 8017.84 :=
by
  sorry

end savings_account_final_amount_l1128_112809


namespace find_train_probability_l1128_112804

-- Define the time range and parameters
def start_time : ℕ := 120
def end_time : ℕ := 240
def wait_time : ℕ := 30

-- Define the conditions
def is_in_range (t : ℕ) : Prop := start_time ≤ t ∧ t ≤ end_time

-- Define the probability function
def probability_of_finding_train : ℚ :=
  let area_triangle : ℚ := (1 / 2) * 30 * 30
  let area_parallelogram : ℚ := 90 * 30
  let shaded_area : ℚ := area_triangle + area_parallelogram
  let total_area : ℚ := (end_time - start_time) * (end_time - start_time)
  shaded_area / total_area

-- The theorem to prove
theorem find_train_probability :
  probability_of_finding_train = 7 / 32 :=
by
  sorry

end find_train_probability_l1128_112804


namespace amount_y_gets_each_rupee_x_gets_l1128_112842

-- Given conditions
variables (x y z a : ℝ)
variables (h_y_share : y = 36) (h_total : x + y + z = 156) (h_z : z = 0.50 * x)

-- Proof problem
theorem amount_y_gets_each_rupee_x_gets (h : 36 / x = a) : a = 9 / 20 :=
by {
  -- The proof is omitted and replaced with 'sorry'.
  sorry
}

end amount_y_gets_each_rupee_x_gets_l1128_112842


namespace maximize_sector_area_l1128_112883

noncomputable def max_area_sector_angle (r : ℝ) (l := 36 - 2 * r) (α := l / r) : ℝ :=
  α

theorem maximize_sector_area (h : ∀ r : ℝ, 2 * r + 36 - 2 * r = 36) :
  max_area_sector_angle 9 = 2 :=
by
  sorry

end maximize_sector_area_l1128_112883


namespace pow_mod_eq_l1128_112888

theorem pow_mod_eq : (17 ^ 2001) % 23 = 11 := 
by {
  sorry
}

end pow_mod_eq_l1128_112888


namespace house_painting_cost_l1128_112850

theorem house_painting_cost :
  let judson_contrib := 500.0
  let kenny_contrib_euros := judson_contrib * 1.2 / 1.1
  let camilo_contrib_pounds := (kenny_contrib_euros * 1.1 + 200.0) / 1.3
  let camilo_contrib_usd := camilo_contrib_pounds * 1.3
  judson_contrib + kenny_contrib_euros * 1.1 + camilo_contrib_usd = 2020.0 := 
by {
  sorry
}

end house_painting_cost_l1128_112850


namespace range_of_x_l1128_112824

theorem range_of_x (x y : ℝ) (h : 4 * x * y + 4 * y^2 + x + 6 = 0) : x ≤ -2 ∨ x ≥ 3 :=
sorry

end range_of_x_l1128_112824
