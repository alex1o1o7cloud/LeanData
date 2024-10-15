import Mathlib

namespace NUMINAMATH_GPT_overlap_length_l130_13088

-- Variables in the conditions
variables (tape_length overlap total_length : ℕ)

-- Conditions
def two_tapes_overlap := (tape_length + tape_length - overlap = total_length)

-- The proof statement we need to prove
theorem overlap_length (h : two_tapes_overlap 275 overlap 512) : overlap = 38 :=
by
  sorry

end NUMINAMATH_GPT_overlap_length_l130_13088


namespace NUMINAMATH_GPT_solve_quadratic_equation_solve_linear_factor_equation_l130_13070

theorem solve_quadratic_equation :
  ∀ (x : ℝ), x^2 - 6 * x + 1 = 0 → (x = 3 - 2 * Real.sqrt 2 ∨ x = 3 + 2 * Real.sqrt 2) :=
by
  intro x
  intro h
  sorry

theorem solve_linear_factor_equation :
  ∀ (x : ℝ), x * (2 * x - 1) = 2 * (2 * x - 1) → (x = 1 / 2 ∨ x = 2) :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solve_quadratic_equation_solve_linear_factor_equation_l130_13070


namespace NUMINAMATH_GPT_square_area_from_diagonal_l130_13074

theorem square_area_from_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 2) : (d / Real.sqrt 2) ^ 2 = 64 :=
by
  sorry

end NUMINAMATH_GPT_square_area_from_diagonal_l130_13074


namespace NUMINAMATH_GPT_general_formula_sum_and_min_value_l130_13062

variables {a : ℕ → ℤ} {S : ℕ → ℤ}

-- Given conditions
def a1 := (a 1 = -5)
def a_condition := (3 * a 3 + a 5 = 0)

-- Prove the general formula for an arithmetic sequence
theorem general_formula (a1 : a 1 = -5) (a_condition : 3 * a 3 + a 5 = 0) : 
  ∀ n, a n = 2 * n - 7 := 
by
  sorry

-- Using the general formula to find the sum Sn and its minimum value
theorem sum_and_min_value (a1 : a 1 = -5) (a_condition : 3 * a 3 + a 5 = 0)
  (h : ∀ n, a n = 2 * n - 7) : 
  ∀ n, S n = n^2 - 6 * n ∧ ∃ n, S n = -9 :=
by
  sorry

end NUMINAMATH_GPT_general_formula_sum_and_min_value_l130_13062


namespace NUMINAMATH_GPT_find_f_neg_half_l130_13042

def is_odd_function {α β : Type*} [AddGroup α] [Neg β] (f : α → β) : Prop :=
  ∀ x : α, f (-x) = -f x

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 2 else 0

theorem find_f_neg_half (f_odd : is_odd_function f) (f_pos : ∀ x > 0, f x = Real.log x / Real.log 2) :
  f (-1/2) = 1 := by
  sorry

end NUMINAMATH_GPT_find_f_neg_half_l130_13042


namespace NUMINAMATH_GPT_least_positive_integer_n_l130_13075

theorem least_positive_integer_n (n : ℕ) (h : (n > 0)) :
  (∃ m : ℕ, m > 0 ∧ (1 / (m : ℝ) - 1 / (m + 1 : ℝ) < 1 / 8) ∧ (∀ k : ℕ, k > 0 ∧ (1 / (k : ℝ) - 1 / (k + 1 : ℝ) < 1 / 8) → m ≤ k)) →
  n = 3 := by
  sorry

end NUMINAMATH_GPT_least_positive_integer_n_l130_13075


namespace NUMINAMATH_GPT_g_of_f_of_3_is_1852_l130_13068

def f (x : ℤ) : ℤ := x^3 - 2
def g (x : ℤ) : ℤ := 3 * x^2 - x + 2

theorem g_of_f_of_3_is_1852 : g (f 3) = 1852 := by
  sorry

end NUMINAMATH_GPT_g_of_f_of_3_is_1852_l130_13068


namespace NUMINAMATH_GPT_cost_price_of_bicycle_l130_13006

variables {CP_A SP_AB SP_BC : ℝ}

theorem cost_price_of_bicycle (h1 : SP_AB = CP_A * 1.2)
                             (h2 : SP_BC = SP_AB * 1.25)
                             (h3 : SP_BC = 225) :
                             CP_A = 150 :=
by sorry

end NUMINAMATH_GPT_cost_price_of_bicycle_l130_13006


namespace NUMINAMATH_GPT_complete_square_solution_l130_13041

theorem complete_square_solution :
  ∀ (x : ℝ), (x^2 + 8*x + 9 = 0) → ((x + 4)^2 = 7) :=
by
  intro x h_eq
  sorry

end NUMINAMATH_GPT_complete_square_solution_l130_13041


namespace NUMINAMATH_GPT_calculation_proof_l130_13054

theorem calculation_proof : (96 / 6) * 3 / 2 = 24 := by
  sorry

end NUMINAMATH_GPT_calculation_proof_l130_13054


namespace NUMINAMATH_GPT_original_proposition_contrapositive_converse_inverse_negation_false_l130_13017

variable {a b c : ℝ}

-- Original Proposition
theorem original_proposition (h : a < b) : a + c < b + c :=
sorry

-- Contrapositive
theorem contrapositive (h : a + c >= b + c) : a >= b :=
sorry

-- Converse
theorem converse (h : a + c < b + c) : a < b :=
sorry

-- Inverse
theorem inverse (h : a >= b) : a + c >= b + c :=
sorry

-- Negation is false
theorem negation_false (h : a < b) : ¬ (a + c >= b + c) :=
sorry

end NUMINAMATH_GPT_original_proposition_contrapositive_converse_inverse_negation_false_l130_13017


namespace NUMINAMATH_GPT_no_x_axis_intersection_iff_l130_13046

theorem no_x_axis_intersection_iff (m : ℝ) :
    (∀ x : ℝ, x^2 - x + m ≠ 0) ↔ m > 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_no_x_axis_intersection_iff_l130_13046


namespace NUMINAMATH_GPT_batteries_difference_is_correct_l130_13072

-- Define the number of batteries used in each item
def flashlights_batteries : ℝ := 3.5
def toys_batteries : ℝ := 15.75
def remote_controllers_batteries : ℝ := 7.25
def wall_clock_batteries : ℝ := 4.8
def wireless_mouse_batteries : ℝ := 3.4

-- Define the combined total of batteries used in the other items
def combined_total : ℝ := flashlights_batteries + remote_controllers_batteries + wall_clock_batteries + wireless_mouse_batteries

-- Define the difference between the total number of batteries used in toys and the combined total of other items
def batteries_difference : ℝ := toys_batteries - combined_total

theorem batteries_difference_is_correct : batteries_difference = -3.2 :=
by
  sorry

end NUMINAMATH_GPT_batteries_difference_is_correct_l130_13072


namespace NUMINAMATH_GPT_longest_side_of_triangle_l130_13064

theorem longest_side_of_triangle (a b c : ℕ) (h1 : a = 3) (h2 : b = 5) 
    (cond : a^2 + b^2 - 6 * a - 10 * b + 34 = 0) 
    (triangle_ineq1 : a + b > c)
    (triangle_ineq2 : a + c > b)
    (triangle_ineq3 : b + c > a)
    (hScalene: a ≠ b ∧ b ≠ c ∧ a ≠ c) : c = 6 ∨ c = 7 := 
by {
  sorry
}

end NUMINAMATH_GPT_longest_side_of_triangle_l130_13064


namespace NUMINAMATH_GPT_students_diff_l130_13099

-- Define the conditions
def M : ℕ := 457
def B : ℕ := 394

-- Prove the final answer
theorem students_diff : M - B = 63 := by
  -- The proof is omitted here with a sorry placeholder
  sorry

end NUMINAMATH_GPT_students_diff_l130_13099


namespace NUMINAMATH_GPT_power_mean_inequality_l130_13057

theorem power_mean_inequality
  (n : ℕ) (hn : 0 < n) (x1 x2 : ℝ) :
  (x1^n + x2^n)^(n+1) / (x1^(n-1) + x2^(n-1))^n ≤ (x1^(n+1) + x2^(n+1))^n / (x1^n + x2^n)^(n-1) :=
by
  sorry

end NUMINAMATH_GPT_power_mean_inequality_l130_13057


namespace NUMINAMATH_GPT_smallest_class_number_l130_13030

theorem smallest_class_number (x : ℕ)
  (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) + (x + 15) = 57) :
  x = 2 :=
by sorry

end NUMINAMATH_GPT_smallest_class_number_l130_13030


namespace NUMINAMATH_GPT_find_k_l130_13024

-- Define the vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

-- Define the condition for vectors to be parallel
def is_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- Translate the problem condition
def problem_condition (k : ℝ) : Prop :=
  let lhs := (k * a.1 + b.1, k * a.2 + b.2)
  let rhs := (a.1 - 3 * b.1, a.2 - 3 * b.2)
  is_parallel lhs rhs

-- The goal is to find k such that the condition holds
theorem find_k : problem_condition (-1/3) :=
by
  sorry

end NUMINAMATH_GPT_find_k_l130_13024


namespace NUMINAMATH_GPT_solve_problem_l130_13056

noncomputable def f : ℝ → ℝ
| x => if x > 0 then Real.logb 2 x else 3^x

theorem solve_problem : f (f (1 / 2)) = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_solve_problem_l130_13056


namespace NUMINAMATH_GPT_moles_of_HNO3_l130_13081

theorem moles_of_HNO3 (HNO3 NaHCO3 NaNO3 : ℝ)
  (h1 : NaHCO3 = 1) (h2 : NaNO3 = 1) :
  HNO3 = 1 :=
by sorry

end NUMINAMATH_GPT_moles_of_HNO3_l130_13081


namespace NUMINAMATH_GPT_exists_integers_not_all_zero_l130_13043

-- Given conditions
variables (a b c : ℝ)
variables (ab bc ca : ℚ)
variables (ha : a * b = ab) (hb : b * c = bc) (hc : c * a = ca)
variables (x y z : ℤ)

-- The theorem to prove
theorem exists_integers_not_all_zero (ha : a * b = ab) (hb : b * c = bc) (hc : c * a = ca):
  ∃ (x y z : ℤ), (¬ (x = 0 ∧ y = 0 ∧ z = 0)) ∧ (a * x + b * y + c * z = 0) :=
sorry

end NUMINAMATH_GPT_exists_integers_not_all_zero_l130_13043


namespace NUMINAMATH_GPT_count_valid_sequences_returning_rectangle_l130_13069

/-- The transformations that can be applied to the rectangle -/
inductive Transformation
| rot90   : Transformation
| rot180  : Transformation
| rot270  : Transformation
| reflYeqX  : Transformation
| reflYeqNegX : Transformation

/-- Apply a transformation to a point (x, y) -/
def apply_transformation (t : Transformation) (p : ℝ × ℝ) : ℝ × ℝ :=
match t with
| Transformation.rot90   => (-p.2,  p.1)
| Transformation.rot180  => (-p.1, -p.2)
| Transformation.rot270  => ( p.2, -p.1)
| Transformation.reflYeqX  => ( p.2,  p.1)
| Transformation.reflYeqNegX => (-p.2, -p.1)

/-- Apply a sequence of transformations to a list of points -/
def apply_sequence (seq : List Transformation) (points : List (ℝ × ℝ)) : List (ℝ × ℝ) :=
  seq.foldl (λ acc t => acc.map (apply_transformation t)) points

/-- Prove that there are exactly 12 valid sequences of three transformations that return the rectangle to its original position -/
theorem count_valid_sequences_returning_rectangle :
  let rectangle := [(0,0), (6,0), (6,2), (0,2)];
  let transformations := [Transformation.rot90, Transformation.rot180, Transformation.rot270, Transformation.reflYeqX, Transformation.reflYeqNegX];
  let seq_transformations := List.replicate 3 transformations;
  (seq_transformations.filter (λ seq => apply_sequence seq rectangle = rectangle)).length = 12 :=
sorry

end NUMINAMATH_GPT_count_valid_sequences_returning_rectangle_l130_13069


namespace NUMINAMATH_GPT_cups_added_l130_13091

/--
A bowl was half full of water. Some cups of water were then added to the bowl, filling the bowl to 70% of its capacity. There are now 14 cups of water in the bowl.
Prove that the number of cups of water added to the bowl is 4.
-/
theorem cups_added (C : ℚ) (h1 : C / 2 + 0.2 * C = 14) : 
  14 - C / 2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_cups_added_l130_13091


namespace NUMINAMATH_GPT_marigold_ratio_l130_13029

theorem marigold_ratio :
  ∃ x, 14 + 25 + x = 89 ∧ x / 25 = 2 := by
  sorry

end NUMINAMATH_GPT_marigold_ratio_l130_13029


namespace NUMINAMATH_GPT_mean_of_three_l130_13039

theorem mean_of_three (x y z a : ℝ)
  (h₁ : (x + y) / 2 = 5)
  (h₂ : (y + z) / 2 = 9)
  (h₃ : (z + x) / 2 = 10) :
  (x + y + z) / 3 = 8 :=
by
  sorry

end NUMINAMATH_GPT_mean_of_three_l130_13039


namespace NUMINAMATH_GPT_solution_to_equation_l130_13060

noncomputable def equation (x : ℝ) : ℝ := 
  (3 * x^2) / (x - 2) - (3 * x + 8) / 4 + (5 - 9 * x) / (x - 2) + 2

theorem solution_to_equation :
  equation 3.294 = 0 ∧ equation (-0.405) = 0 :=
by
  sorry

end NUMINAMATH_GPT_solution_to_equation_l130_13060


namespace NUMINAMATH_GPT_initial_students_count_l130_13095

variable (initial_students : ℕ)
variable (number_of_new_boys : ℕ := 5)
variable (initial_percentage_girls : ℝ := 0.40)
variable (new_percentage_girls : ℝ := 0.32)

theorem initial_students_count (h : initial_percentage_girls * initial_students = new_percentage_girls * (initial_students + number_of_new_boys)) : 
  initial_students = 20 := 
by 
  sorry

end NUMINAMATH_GPT_initial_students_count_l130_13095


namespace NUMINAMATH_GPT_people_per_team_l130_13077

theorem people_per_team 
  (managers : ℕ) (employees : ℕ) (teams : ℕ) 
  (h1 : managers = 23) (h2 : employees = 7) (h3 : teams = 6) :
  (managers + employees) / teams = 5 :=
by
  sorry

end NUMINAMATH_GPT_people_per_team_l130_13077


namespace NUMINAMATH_GPT_range_of_m_l130_13096

theorem range_of_m (m : ℝ) : (∀ x ∈ Set.Icc (-1 : ℝ) (1 : ℝ), x^2 - 4 * x - 2 * m + 1 ≤ 0) ↔ m ∈ Set.Ici (3 : ℝ) := 
sorry

end NUMINAMATH_GPT_range_of_m_l130_13096


namespace NUMINAMATH_GPT_symmetric_points_sum_l130_13015

theorem symmetric_points_sum {c e : ℤ} 
  (P : ℤ × ℤ × ℤ) 
  (sym_xoy : ℤ × ℤ × ℤ) 
  (sym_y : ℤ × ℤ × ℤ) 
  (hP : P = (-4, -2, 3)) 
  (h_sym_xoy : sym_xoy = (-4, -2, -3)) 
  (h_sym_y : sym_y = (4, -2, 3)) 
  (hc : c = -3) 
  (he : e = 4) : 
  c + e = 1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_symmetric_points_sum_l130_13015


namespace NUMINAMATH_GPT_problem_solution_l130_13066

-- Definitions and Assumptions
variable (f : ℝ → ℝ)
variable (h_diff : Differentiable ℝ f)
variable (h_condition : ∀ x : ℝ, f x - (deriv^[2]) f x > 0)

-- Statement to Prove
theorem problem_solution : e * f 2015 > f 2016 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l130_13066


namespace NUMINAMATH_GPT_part_I_part_II_l130_13048

-- Define the function f
def f (x: ℝ) : ℝ := abs (x - 1) - 2 * abs (x + 1)

-- The conditions and questions transformed into Lean statements
theorem part_I : ∃ m, (∀ x: ℝ, f x ≤ m) ∧ (m = f (-1)) ∧ (m = 2) := by
  sorry

theorem part_II (a b c : ℝ) (h₀ : 0 < a ∧ 0 < b ∧ 0 < c) (h₁ : a^2 + 3 * b^2 + 2 * c^2 = 2) : 
  ∃ n, (∀ a b c : ℝ, (0 < a ∧ 0 < b ∧ 0 < c) ∧ (a^2 + 3 * b^2 + 2 * c^2 = 2) → ab + 2 * bc ≤ n) ∧ (n = 1) := by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l130_13048


namespace NUMINAMATH_GPT_pencils_ratio_l130_13021

theorem pencils_ratio
  (Sarah_pencils : ℕ)
  (Tyrah_pencils : ℕ)
  (Tim_pencils : ℕ)
  (h1 : Tyrah_pencils = 12)
  (h2 : Tim_pencils = 16)
  (h3 : Tim_pencils = 8 * Sarah_pencils) :
  Tyrah_pencils / Sarah_pencils = 6 :=
by
  sorry

end NUMINAMATH_GPT_pencils_ratio_l130_13021


namespace NUMINAMATH_GPT_line_equation_l130_13003

theorem line_equation (b : ℝ) :
  (∃ b, (∀ x y, y = (3/4) * x + b) ∧ 
  (1/2) * |b| * |- (4/3) * b| = 6 →
  (3 * x - 4 * y + 12 = 0 ∨ 3 * x - 4 * y - 12 = 0)) := 
sorry

end NUMINAMATH_GPT_line_equation_l130_13003


namespace NUMINAMATH_GPT_grasshopper_jump_l130_13089

theorem grasshopper_jump (frog_jump grasshopper_jump : ℕ)
  (h1 : frog_jump = grasshopper_jump + 17)
  (h2 : frog_jump = 53) :
  grasshopper_jump = 36 :=
by
  sorry

end NUMINAMATH_GPT_grasshopper_jump_l130_13089


namespace NUMINAMATH_GPT_distinct_nat_numbers_l130_13001

theorem distinct_nat_numbers 
  (a b c : ℕ) (p q r : ℤ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_sum : a + b + c = 55) 
  (h_ab : a + b = p * p) 
  (h_bc : b + c = q * q) 
  (h_ca : c + a = r * r) : 
  a = 19 ∧ b = 6 ∧ c = 30 :=
sorry

end NUMINAMATH_GPT_distinct_nat_numbers_l130_13001


namespace NUMINAMATH_GPT_correct_ordering_of_powers_l130_13045

theorem correct_ordering_of_powers : 
  7^8 < 3^15 ∧ 3^15 < 4^12 ∧ 4^12 < 8^10 :=
  by
    sorry

end NUMINAMATH_GPT_correct_ordering_of_powers_l130_13045


namespace NUMINAMATH_GPT_simplify_fraction_l130_13093

theorem simplify_fraction (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a ≠ b) :
  (a^3 - b^3) / (a * b) - (a * b^2 - b^3) / (a * b - a^3) = (a^2 + a * b + b^2) / b :=
by {
  -- Proof skipped
  sorry
}

end NUMINAMATH_GPT_simplify_fraction_l130_13093


namespace NUMINAMATH_GPT_roy_cat_finishes_food_on_wednesday_l130_13012

-- Define the conditions
def morning_consumption := (1 : ℚ) / 5
def evening_consumption := (1 : ℚ) / 6
def total_cans := 10

-- Define the daily consumption calculation
def daily_consumption := morning_consumption + evening_consumption

-- Define the day calculation function
def day_cat_finishes_food : String :=
  let total_days := total_cans / daily_consumption
  if total_days ≤ 7 then "certain day within a week"
  else if total_days ≤ 14 then "Wednesday next week"
  else "later"

-- The main theorem to prove
theorem roy_cat_finishes_food_on_wednesday : day_cat_finishes_food = "Wednesday next week" := sorry

end NUMINAMATH_GPT_roy_cat_finishes_food_on_wednesday_l130_13012


namespace NUMINAMATH_GPT_how_many_grapes_l130_13037

-- Define the conditions given in the problem
def apples_to_grapes :=
  (3 / 4) * 12 = 6

-- Define the result to prove
def grapes_value :=
  (1 / 3) * 9 = 2

-- The statement combining the conditions and the problem to be proven
theorem how_many_grapes : apples_to_grapes → grapes_value :=
by
  intro h
  sorry

end NUMINAMATH_GPT_how_many_grapes_l130_13037


namespace NUMINAMATH_GPT_complex_magnitude_l130_13018

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the complex number z with the given condition
variable (z : ℂ) (h : z * (1 + i) = 2 * i)

-- Statement of the problem: Prove that |z + 2 * i| = √10
theorem complex_magnitude (z : ℂ) (h : z * (1 + i) = 2 * i) : Complex.abs (z + 2 * i) = Real.sqrt 10 := 
sorry

end NUMINAMATH_GPT_complex_magnitude_l130_13018


namespace NUMINAMATH_GPT_nate_matches_left_l130_13073

def initial_matches : ℕ := 70
def matches_dropped : ℕ := 10
def matches_eaten : ℕ := 2 * matches_dropped
def total_matches_lost : ℕ := matches_dropped + matches_eaten
def remaining_matches : ℕ := initial_matches - total_matches_lost

theorem nate_matches_left : remaining_matches = 40 := by
  sorry

end NUMINAMATH_GPT_nate_matches_left_l130_13073


namespace NUMINAMATH_GPT_quadrilateral_area_inequality_l130_13011

theorem quadrilateral_area_inequality 
  (T : ℝ) (a b c d e f : ℝ) (φ : ℝ) 
  (hT : T = (1/2) * e * f * Real.sin φ) 
  (hptolemy : e * f ≤ a * c + b * d) : 
  2 * T ≤ a * c + b * d := 
sorry

end NUMINAMATH_GPT_quadrilateral_area_inequality_l130_13011


namespace NUMINAMATH_GPT_moles_of_HCl_formed_l130_13044

-- Conditions: 1 mole of Methane (CH₄) and 2 moles of Chlorine (Cl₂)
def methane := 1 -- 1 mole of methane
def chlorine := 2 -- 2 moles of chlorine

-- Reaction: CH₄ + Cl₂ → CH₃Cl + HCl
-- We state that 1 mole of methane reacts with 1 mole of chlorine to form 1 mole of hydrochloric acid
def reaction (methane chlorine : ℕ) : ℕ := methane

-- Theorem: Prove 1 mole of hydrochloric acid (HCl) is formed
theorem moles_of_HCl_formed : reaction methane chlorine = 1 := by
  sorry

end NUMINAMATH_GPT_moles_of_HCl_formed_l130_13044


namespace NUMINAMATH_GPT_ticket_cost_is_nine_l130_13013

theorem ticket_cost_is_nine (bought_tickets : ℕ) (left_tickets : ℕ) (spent_dollars : ℕ) 
  (h1 : bought_tickets = 6) 
  (h2 : left_tickets = 3) 
  (h3 : spent_dollars = 27) : 
  spent_dollars / (bought_tickets - left_tickets) = 9 :=
by
  -- Using the imported library and the given conditions
  sorry

end NUMINAMATH_GPT_ticket_cost_is_nine_l130_13013


namespace NUMINAMATH_GPT_admission_cutoff_score_l130_13076

theorem admission_cutoff_score (n : ℕ) (x : ℚ) (admitted_average non_admitted_average total_average : ℚ)
    (h1 : admitted_average = x + 15)
    (h2 : non_admitted_average = x - 20)
    (h3 : total_average = 90)
    (h4 : (admitted_average * (2 / 5) + non_admitted_average * (3 / 5)) = total_average) : x = 96 := 
by
  sorry

end NUMINAMATH_GPT_admission_cutoff_score_l130_13076


namespace NUMINAMATH_GPT_men_complete_units_per_day_l130_13000

noncomputable def UnitsCompletedByMen (total_units : ℕ) (units_by_women : ℕ) : ℕ :=
  total_units - units_by_women

theorem men_complete_units_per_day :
  UnitsCompletedByMen 12 3 = 9 := by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_men_complete_units_per_day_l130_13000


namespace NUMINAMATH_GPT_inverse_undefined_at_one_l130_13078

noncomputable def f (x : ℝ) : ℝ := (x - 2) / (x - 5)

theorem inverse_undefined_at_one : ∀ (x : ℝ), (x = 1) → ¬∃ y : ℝ, f y = x :=
by
  sorry

end NUMINAMATH_GPT_inverse_undefined_at_one_l130_13078


namespace NUMINAMATH_GPT_problem_A_eq_7_problem_A_eq_2012_l130_13033

open Nat

-- Problem statement for A = 7
theorem problem_A_eq_7 (n k : ℕ) :
  (n! + 7 * n = n^k) ↔ ((n, k) = (2, 4) ∨ (n, k) = (3, 3)) :=
sorry

-- Problem statement for A = 2012
theorem problem_A_eq_2012 (n k : ℕ) :
  ¬ (n! + 2012 * n = n^k) :=
sorry

end NUMINAMATH_GPT_problem_A_eq_7_problem_A_eq_2012_l130_13033


namespace NUMINAMATH_GPT_rectangle_area_l130_13005

theorem rectangle_area (a b : ℝ) (h : 2 * a^2 - 11 * a + 5 = 0) (hb : 2 * b^2 - 11 * b + 5 = 0) : a * b = 5 / 2 :=
sorry

end NUMINAMATH_GPT_rectangle_area_l130_13005


namespace NUMINAMATH_GPT_find_x_in_interval_l130_13080

theorem find_x_in_interval (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 2) (h_eq : (2 - Real.sin (2 * x)) * Real.sin (x + π / 4) = 1) : x = π / 4 := 
sorry

end NUMINAMATH_GPT_find_x_in_interval_l130_13080


namespace NUMINAMATH_GPT_find_A_l130_13020

theorem find_A (J : ℤ := 15)
  (JAVA_pts : ℤ := 50)
  (AJAX_pts : ℤ := 53)
  (AXLE_pts : ℤ := 40)
  (L : ℤ := 12)
  (JAVA_eq : ∀ A V : ℤ, 2 * A + V + J = JAVA_pts)
  (AJAX_eq : ∀ A X : ℤ, 2 * A + X + J = AJAX_pts)
  (AXLE_eq : ∀ A X E : ℤ, A + X + L + E = AXLE_pts) : A = 21 :=
sorry

end NUMINAMATH_GPT_find_A_l130_13020


namespace NUMINAMATH_GPT_lcm_of_10_and_21_l130_13061

theorem lcm_of_10_and_21 : Nat.lcm 10 21 = 210 :=
by
  sorry

end NUMINAMATH_GPT_lcm_of_10_and_21_l130_13061


namespace NUMINAMATH_GPT_smallest_possible_z_l130_13085

theorem smallest_possible_z (w x y z : ℕ) (k : ℕ) (h1 : w = x - 1) (h2 : y = x + 1) (h3 : z = x + 2)
  (h4 : w ≠ x ∧ x ≠ y ∧ y ≠ z ∧ w ≠ y ∧ w ≠ z ∧ x ≠ z) (h5 : k = 2) (h6 : w^3 + x^3 + y^3 = k * z^3) : z = 6 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_z_l130_13085


namespace NUMINAMATH_GPT_no_such_function_exists_l130_13047

theorem no_such_function_exists (f : ℝ → ℝ) (Hf : ∀ x : ℝ, 2 * f (Real.cos x) = f (Real.sin x) + Real.sin x) : False :=
by
  sorry

end NUMINAMATH_GPT_no_such_function_exists_l130_13047


namespace NUMINAMATH_GPT_result_when_j_divided_by_26_l130_13094

noncomputable def j := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 10 11) (Nat.lcm 12 13)) (Nat.lcm 14 15))

theorem result_when_j_divided_by_26 : j / 26 = 2310 := by 
  sorry

end NUMINAMATH_GPT_result_when_j_divided_by_26_l130_13094


namespace NUMINAMATH_GPT_conference_games_scheduled_l130_13002

theorem conference_games_scheduled
  (divisions : ℕ)
  (teams_per_division : ℕ)
  (intra_games_per_pair : ℕ)
  (inter_games_per_pair : ℕ)
  (h_div : divisions = 3)
  (h_teams : teams_per_division = 4)
  (h_intra : intra_games_per_pair = 3)
  (h_inter : inter_games_per_pair = 2) :
  let intra_division_games := (teams_per_division * (teams_per_division - 1) / 2) * intra_games_per_pair
  let intra_division_total := intra_division_games * divisions
  let inter_division_games := teams_per_division * (teams_per_division * (divisions - 1)) * inter_games_per_pair
  let inter_division_total := inter_division_games * divisions / 2
  let total_games := intra_division_total + inter_division_total
  total_games = 150 :=
by
  sorry

end NUMINAMATH_GPT_conference_games_scheduled_l130_13002


namespace NUMINAMATH_GPT_frac_plus_a_ge_seven_l130_13010

theorem frac_plus_a_ge_seven (a : ℝ) (h : a > 3) : 4 / (a - 3) + a ≥ 7 := 
by
  sorry

end NUMINAMATH_GPT_frac_plus_a_ge_seven_l130_13010


namespace NUMINAMATH_GPT_max_value_expr_l130_13063

-- Define the expression
def expr (a b c d : ℝ) : ℝ :=
  a + b + c + d - a * b - b * c - c * d - d * a

-- The main theorem
theorem max_value_expr :
  (∀ (a b c d : ℝ), 0 ≤ a ∧ a ≤ 1 → 0 ≤ b ∧ b ≤ 1 → 0 ≤ c ∧ c ≤ 1 → 0 ≤ d ∧ d ≤ 1 → expr a b c d ≤ 2) ∧
  (∃ (a b c d : ℝ), 0 ≤ a ∧ a = 1 ∧ 0 ≤ b ∧ b = 0 ∧ 0 ≤ c ∧ c = 1 ∧ 0 ≤ d ∧ d = 0 ∧ expr a b c d = 2) :=
  by
  sorry

end NUMINAMATH_GPT_max_value_expr_l130_13063


namespace NUMINAMATH_GPT_probability_two_tails_two_heads_l130_13065

theorem probability_two_tails_two_heads :
  let num_coins := 4
  let num_tails_heads := 2
  let num_sequences := Nat.choose num_coins num_tails_heads
  let single_probability := (1 / 2) ^ num_coins
  let total_probability := num_sequences * single_probability
  total_probability = 3 / 8 :=
by
  let num_coins := 4
  let num_tails_heads := 2
  let num_sequences := Nat.choose num_coins num_tails_heads
  let single_probability := (1 / 2) ^ num_coins
  let total_probability := num_sequences * single_probability
  sorry

end NUMINAMATH_GPT_probability_two_tails_two_heads_l130_13065


namespace NUMINAMATH_GPT_no_int_solutions_l130_13040

theorem no_int_solutions (c x y : ℤ) (h1 : 0 < c) (h2 : c % 2 = 1) : x ^ 2 - y ^ 3 ≠ (2 * c) ^ 3 - 1 :=
sorry

end NUMINAMATH_GPT_no_int_solutions_l130_13040


namespace NUMINAMATH_GPT_largest_n_for_factoring_l130_13025

theorem largest_n_for_factoring :
  ∃ (n : ℤ), (∀ (A B : ℤ), (3 * A + B = n) → (3 * A * B = 90) → n = 271) :=
by sorry

end NUMINAMATH_GPT_largest_n_for_factoring_l130_13025


namespace NUMINAMATH_GPT_arithmetic_seq_third_term_l130_13097

theorem arithmetic_seq_third_term
  (a d : ℝ)
  (h : a + (a + 2 * d) = 10) :
  a + d = 5 := by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_third_term_l130_13097


namespace NUMINAMATH_GPT_range_of_k_l130_13084

theorem range_of_k 
  (h : ∀ x : ℝ, x = 1 → k^2 * x^2 - 6 * k * x + 8 ≥ 0) :
  k ≥ 4 ∨ k ≤ 2 := by
sorry

end NUMINAMATH_GPT_range_of_k_l130_13084


namespace NUMINAMATH_GPT_polynomial_simplification_l130_13038

theorem polynomial_simplification (x : ℝ) :
  (2 * x^4 + 3 * x^3 - 5 * x^2 + 9 * x - 8) + (-x^5 + x^4 - 2 * x^3 + 4 * x^2 - 6 * x + 14) = 
  -x^5 + 3 * x^4 + x^3 - x^2 + 3 * x + 6 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_simplification_l130_13038


namespace NUMINAMATH_GPT_Petya_tore_out_sheets_l130_13034

theorem Petya_tore_out_sheets (n m : ℕ) (h1 : n = 185) (h2 : m = 518)
  (h3 : m.digits = n.digits) : (m - n + 1) / 2 = 167 :=
by
  sorry

end NUMINAMATH_GPT_Petya_tore_out_sheets_l130_13034


namespace NUMINAMATH_GPT_factorial_expression_simplification_l130_13008

theorem factorial_expression_simplification : (3 * (Nat.factorial 5) + 15 * (Nat.factorial 4)) / (Nat.factorial 6) = 1 := by
  sorry

end NUMINAMATH_GPT_factorial_expression_simplification_l130_13008


namespace NUMINAMATH_GPT_symmetric_point_A_is_B_l130_13022

/-
  Define the symmetric point function for reflecting a point across the origin.
  Define the coordinate of point A.
  Assert that the symmetric point of A has coordinates (-2, 6).
-/

structure Point where
  x : ℤ
  y : ℤ

def symmetric_point (p : Point) : Point :=
  Point.mk (-p.x) (-p.y)

def A : Point := ⟨2, -6⟩

def B : Point := ⟨-2, 6⟩

theorem symmetric_point_A_is_B : symmetric_point A = B := by
  sorry

end NUMINAMATH_GPT_symmetric_point_A_is_B_l130_13022


namespace NUMINAMATH_GPT_hannah_dogs_food_total_l130_13019

def first_dog_food : ℝ := 1.5
def second_dog_food : ℝ := 2 * first_dog_food
def third_dog_food : ℝ := second_dog_food + 2.5

theorem hannah_dogs_food_total : first_dog_food + second_dog_food + third_dog_food = 10 := by
  sorry

end NUMINAMATH_GPT_hannah_dogs_food_total_l130_13019


namespace NUMINAMATH_GPT_max_abs_eq_one_vertices_l130_13026

theorem max_abs_eq_one_vertices (x y : ℝ) :
  (max (|x + y|) (|x - y|) = 1) ↔ (x = -1 ∧ y = 0) ∨ (x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = -1) ∨ (x = 0 ∧ y = 1) :=
sorry

end NUMINAMATH_GPT_max_abs_eq_one_vertices_l130_13026


namespace NUMINAMATH_GPT_sufficient_not_necessary_l130_13053

-- Define set A and set B
def setA (x : ℝ) := x > 5
def setB (x : ℝ) := x > 3

-- Statement:
theorem sufficient_not_necessary (x : ℝ) : setA x → setB x :=
by
  intro h
  exact sorry

end NUMINAMATH_GPT_sufficient_not_necessary_l130_13053


namespace NUMINAMATH_GPT_christmas_bonus_remainder_l130_13004

theorem christmas_bonus_remainder (P : ℕ) (h : P % 5 = 2) : (3 * P) % 5 = 1 :=
by
  sorry

end NUMINAMATH_GPT_christmas_bonus_remainder_l130_13004


namespace NUMINAMATH_GPT_problem_statement_l130_13086

namespace LeanProofExample

def not_divisible (n : ℕ) (p : ℕ) : Prop :=
  ¬(p ∣ n)

theorem problem_statement (x y : ℕ) 
  (hx : not_divisible x 59) 
  (hy : not_divisible y 59)
  (h : 3 * x + 28 * y ≡ 0 [MOD 59]) :
  ¬(5 * x + 16 * y ≡ 0 [MOD 59]) :=
  sorry

end LeanProofExample

end NUMINAMATH_GPT_problem_statement_l130_13086


namespace NUMINAMATH_GPT_problem1_problem2_l130_13098

variable (t : ℝ)

-- Problem 1
theorem problem1 (h : (4:ℝ) - 8 * t + 16 < 0) : t > 5 / 2 :=
sorry

-- Problem 2
theorem problem2 (hp: 4 - t > t - 2) (hq : t - 2 > 0) (hdisjoint : (∃ (p : Prop) (q : Prop), (p ∨ q) ∧ ¬(p ∧ q))):
  (2 < t ∧ t ≤ 5 / 2) ∨ (t ≥ 3) :=
sorry


end NUMINAMATH_GPT_problem1_problem2_l130_13098


namespace NUMINAMATH_GPT_number_of_throwers_l130_13090

theorem number_of_throwers (total_players throwers right_handed : ℕ) 
  (h1 : total_players = 64)
  (h2 : right_handed = 55) 
  (h3 : ∀ T N, T + N = total_players → 
  T + (2/3 : ℚ) * N = right_handed) : 
  throwers = 37 := 
sorry

end NUMINAMATH_GPT_number_of_throwers_l130_13090


namespace NUMINAMATH_GPT_correct_operation_l130_13067

noncomputable def check_operations : Prop :=
    ∀ (a : ℝ), ( a^6 / a^3 = a^3 ) ∧ 
               ¬( 3 * a^5 + a^5 = 4 * a^10 ) ∧
               ¬( (2 * a)^3 = 2 * a^3 ) ∧
               ¬( (a^2)^4 = a^6 )

theorem correct_operation : check_operations :=
by
  intro a
  have h1 : a^6 / a^3 = a^3 := by
    sorry
  have h2 : ¬(3 * a^5 + a^5 = 4 * a^10) := by
    sorry
  have h3 : ¬((2 * a)^3 = 2 * a^3) := by
    sorry
  have h4 : ¬((a^2)^4 = a^6) := by
    sorry
  exact ⟨h1, h2, h3, h4⟩

end NUMINAMATH_GPT_correct_operation_l130_13067


namespace NUMINAMATH_GPT_logarithmic_relationship_l130_13014

theorem logarithmic_relationship
  (a b c : ℝ) (m n r : ℝ)
  (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) (h4 : 1 < c)
  (h5 : m = Real.log c / Real.log a)
  (h6 : n = Real.log c / Real.log b)
  (h7 : r = a ^ c) :
  n < m ∧ m < r :=
sorry

end NUMINAMATH_GPT_logarithmic_relationship_l130_13014


namespace NUMINAMATH_GPT_geometric_sequence_sum_l130_13050

noncomputable def sum_of_first_n_terms (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum (q : ℝ) (h_pos : q > 0) (h_a1 : a_1 = 1) (h_a5 : a_5 = 16) :
  sum_of_first_n_terms 1 q 7 = 127 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l130_13050


namespace NUMINAMATH_GPT_minimum_red_pieces_l130_13055

theorem minimum_red_pieces (w b r : ℕ) 
  (h1 : b ≤ w / 2) 
  (h2 : r ≥ 3 * b) 
  (h3 : w + b ≥ 55) : r = 57 := 
sorry

end NUMINAMATH_GPT_minimum_red_pieces_l130_13055


namespace NUMINAMATH_GPT_max_k_condition_l130_13049

theorem max_k_condition (k : ℕ) (total_goods : ℕ) (num_platforms : ℕ) (platform_capacity : ℕ) :
  total_goods = 1500 ∧ num_platforms = 25 ∧ platform_capacity = 80 → 
  (∀ (c : ℕ), 1 ≤ c ∧ c ≤ k → c ∣ k) → 
  (∀ (total : ℕ), total ≤ num_platforms * platform_capacity → total ≥ total_goods) → 
  k ≤ 26 := 
sorry

end NUMINAMATH_GPT_max_k_condition_l130_13049


namespace NUMINAMATH_GPT_final_value_A_eq_B_pow_N_l130_13092

-- Definitions of conditions
def compute_A (A B : ℕ) (N : ℕ) : ℕ :=
    if N ≤ 0 then 
        1 
    else 
        let rec compute_loop (A' B' N' : ℕ) : ℕ :=
            if N' = 0 then A' 
            else 
                let B'' := B' * B'
                let N'' := N' / 2
                let A'' := if N' % 2 = 1 then A' * B' else A'
                compute_loop A'' B'' N'' 
        compute_loop A B N

-- Theorem statement
theorem final_value_A_eq_B_pow_N (A B N : ℕ) : compute_A A B N = B ^ N :=
    sorry

end NUMINAMATH_GPT_final_value_A_eq_B_pow_N_l130_13092


namespace NUMINAMATH_GPT_Marty_combinations_l130_13087

theorem Marty_combinations:
  let colors := ({blue, green, yellow, black, white} : Finset String)
  let tools := ({brush, roller, sponge, spray_gun} : Finset String)
  colors.card * tools.card = 20 := 
by
  sorry

end NUMINAMATH_GPT_Marty_combinations_l130_13087


namespace NUMINAMATH_GPT_net_pay_rate_is_26_dollars_per_hour_l130_13083

-- Defining the conditions
noncomputable def total_distance (time_hours : ℝ) (speed_mph : ℝ) : ℝ :=
  time_hours * speed_mph

noncomputable def adjusted_fuel_efficiency (original_efficiency : ℝ) (decrease_percentage : ℝ) : ℝ :=
  original_efficiency * (1 - decrease_percentage)

noncomputable def gasoline_used (distance : ℝ) (efficiency : ℝ) : ℝ :=
  distance / efficiency

noncomputable def earnings (rate_per_mile : ℝ) (distance : ℝ) : ℝ :=
  rate_per_mile * distance

noncomputable def updated_gasoline_price (original_price : ℝ) (increase_percentage : ℝ) : ℝ :=
  original_price * (1 + increase_percentage)

noncomputable def total_cost_gasoline (gasoline_price : ℝ) (gasoline_used : ℝ) : ℝ :=
  gasoline_price * gasoline_used

noncomputable def net_earnings (earnings : ℝ) (cost : ℝ) : ℝ :=
  earnings - cost

noncomputable def net_rate_of_pay (net_earnings : ℝ) (time_hours : ℝ) : ℝ :=
  net_earnings / time_hours

-- Given constants
def time_hours : ℝ := 3
def speed_mph : ℝ := 50
def original_efficiency : ℝ := 30
def decrease_percentage : ℝ := 0.10
def rate_per_mile : ℝ := 0.60
def original_gasoline_price : ℝ := 2.00
def increase_percentage : ℝ := 0.20

-- Proof problem statement
theorem net_pay_rate_is_26_dollars_per_hour :
  net_rate_of_pay 
    (net_earnings
       (earnings rate_per_mile (total_distance time_hours speed_mph))
       (total_cost_gasoline
          (updated_gasoline_price original_gasoline_price increase_percentage)
          (gasoline_used
            (total_distance time_hours speed_mph)
            (adjusted_fuel_efficiency original_efficiency decrease_percentage))))
    time_hours = 26 := 
  sorry

end NUMINAMATH_GPT_net_pay_rate_is_26_dollars_per_hour_l130_13083


namespace NUMINAMATH_GPT_initial_red_marbles_l130_13031

theorem initial_red_marbles (r g : ℕ) (h1 : r * 3 = 7 * g) (h2 : 4 * (r - 14) = g + 30) : r = 24 := 
sorry

end NUMINAMATH_GPT_initial_red_marbles_l130_13031


namespace NUMINAMATH_GPT_repeating_decimal_to_fraction_denominator_l130_13051

theorem repeating_decimal_to_fraction_denominator :
  ∀ (S : ℚ), (S = 0.27) → (∃ a b : ℤ, b ≠ 0 ∧ S = a / b ∧ Int.gcd a b = 1 ∧ b = 3) :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_to_fraction_denominator_l130_13051


namespace NUMINAMATH_GPT_max_value_a_l130_13059

noncomputable def setA (a : ℝ) : Set ℝ := { x | (x - 1) * (x - a) ≥ 0 }
noncomputable def setB (a : ℝ) : Set ℝ := { x | x ≥ a - 1 }

theorem max_value_a (a : ℝ) :
  (setA a ∪ setB a = Set.univ) → a ≤ 2 := by
  sorry

end NUMINAMATH_GPT_max_value_a_l130_13059


namespace NUMINAMATH_GPT_number_of_correct_statements_l130_13009

theorem number_of_correct_statements:
  (¬∀ (a : ℝ), -a < 0) ∧
  (∀ (x : ℝ), |x| = -x → x < 0) ∧
  (∀ (a : ℚ), (∀ (b : ℚ), |b| ≥ |a|) → a = 0) ∧
  (∀ (x y : ℝ), 5 * x^2 * y ≠ 0 → 2 + 1 = 3) →
  2 = 2 := sorry

end NUMINAMATH_GPT_number_of_correct_statements_l130_13009


namespace NUMINAMATH_GPT_cheaper_to_buy_more_l130_13036

def cost (n : ℕ) : ℕ :=
  if 1 ≤ n ∧ n ≤ 30 then 15 * n
  else if 31 ≤ n ∧ n ≤ 60 then 13 * n
  else if 61 ≤ n ∧ n ≤ 90 then 12 * n
  else if 91 ≤ n then 11 * n
  else 0

theorem cheaper_to_buy_more (n : ℕ) : 
  (∃ m, m < n ∧ cost (m + 1) < cost m) ↔ n = 9 := sorry

end NUMINAMATH_GPT_cheaper_to_buy_more_l130_13036


namespace NUMINAMATH_GPT_base_h_equation_l130_13007

theorem base_h_equation (h : ℕ) : 
  (5 * h^3 + 7 * h^2 + 3 * h + 4) + (6 * h^3 + 4 * h^2 + 2 * h + 1) = 
  1 * h^4 + 4 * h^3 + 1 * h^2 + 5 * h + 5 → 
  h = 10 := 
sorry

end NUMINAMATH_GPT_base_h_equation_l130_13007


namespace NUMINAMATH_GPT_propositions_alpha_and_beta_true_l130_13052

def even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = -f (-x)

def strictly_increasing_function (f : ℝ → ℝ) : Prop :=
∀ x y, x < y → f x < f y

def strictly_decreasing_function (f : ℝ → ℝ) : Prop :=
∀ x y, x < y → f x > f y

def alpha (f : ℝ → ℝ) : Prop :=
∀ x, ∃ g h : ℝ → ℝ, even_function g ∧ odd_function h ∧ f x = g x + h x

def beta (f : ℝ → ℝ) : Prop :=
∀ x, strictly_increasing_function f → ∃ p q : ℝ → ℝ, 
  strictly_increasing_function p ∧ strictly_decreasing_function q ∧ f x = p x + q x

theorem propositions_alpha_and_beta_true (f : ℝ → ℝ) :
  alpha f ∧ beta f :=
by
  sorry

end NUMINAMATH_GPT_propositions_alpha_and_beta_true_l130_13052


namespace NUMINAMATH_GPT_percentage_of_second_discount_is_correct_l130_13028

def car_original_price : ℝ := 12000
def first_discount : ℝ := 0.20
def final_price_after_discounts : ℝ := 7752
def third_discount : ℝ := 0.05

def solve_percentage_second_discount : Prop := 
  ∃ (second_discount : ℝ), 
    (car_original_price * (1 - first_discount) * (1 - second_discount) * (1 - third_discount) = final_price_after_discounts) ∧ 
    (second_discount * 100 = 15)

theorem percentage_of_second_discount_is_correct : solve_percentage_second_discount :=
  sorry

end NUMINAMATH_GPT_percentage_of_second_discount_is_correct_l130_13028


namespace NUMINAMATH_GPT_largest_stamps_per_page_l130_13071

theorem largest_stamps_per_page (h1 : Nat := 1050) (h2 : Nat := 1260) (h3 : Nat := 1470) :
  Nat.gcd h1 (Nat.gcd h2 h3) = 210 :=
by
  sorry

end NUMINAMATH_GPT_largest_stamps_per_page_l130_13071


namespace NUMINAMATH_GPT_mountaineering_team_problem_l130_13082

structure Climber :=
  (total_students : ℕ)
  (advanced_climbers : ℕ)
  (intermediate_climbers : ℕ)
  (beginners : ℕ)

structure Experience :=
  (advanced_points : ℕ)
  (intermediate_points : ℕ)
  (beginner_points : ℕ)

structure TeamComposition :=
  (advanced_needed : ℕ)
  (intermediate_needed : ℕ)
  (beginners_needed : ℕ)
  (max_experience : ℕ)

def team_count (students : Climber) (xp : Experience) (comp : TeamComposition) : ℕ :=
  let total_experience := comp.advanced_needed * xp.advanced_points +
                          comp.intermediate_needed * xp.intermediate_points +
                          comp.beginners_needed * xp.beginner_points
  let max_teams_from_advanced := students.advanced_climbers / comp.advanced_needed
  let max_teams_from_intermediate := students.intermediate_climbers / comp.intermediate_needed
  let max_teams_from_beginners := students.beginners / comp.beginners_needed
  if total_experience ≤ comp.max_experience then
    min (max_teams_from_advanced) $ min (max_teams_from_intermediate) (max_teams_from_beginners)
  else 0

def problem : Prop :=
  team_count
    ⟨172, 45, 70, 57⟩
    ⟨80, 50, 30⟩
    ⟨5, 8, 5, 1000⟩ = 8

-- Let's declare the theorem now:
theorem mountaineering_team_problem : problem := sorry

end NUMINAMATH_GPT_mountaineering_team_problem_l130_13082


namespace NUMINAMATH_GPT_set_contains_one_implies_values_l130_13027

theorem set_contains_one_implies_values (x : ℝ) (A : Set ℝ) (hA : A = {x, x^2}) (h1 : 1 ∈ A) : x = 1 ∨ x = -1 := by
  sorry

end NUMINAMATH_GPT_set_contains_one_implies_values_l130_13027


namespace NUMINAMATH_GPT_school_points_l130_13079

theorem school_points (a b c : ℕ) (h1 : a + b + c = 285)
  (h2 : ∃ x : ℕ, a - 8 = x ∧ b - 12 = x ∧ c - 7 = x) : a + c = 187 :=
sorry

end NUMINAMATH_GPT_school_points_l130_13079


namespace NUMINAMATH_GPT_equation_is_linear_in_one_variable_l130_13058

theorem equation_is_linear_in_one_variable (n : ℤ) :
  (∀ x : ℝ, (n - 2) * x ^ |n - 1| + 5 = 0 → False) → n = 0 := by
  sorry

end NUMINAMATH_GPT_equation_is_linear_in_one_variable_l130_13058


namespace NUMINAMATH_GPT_problem_min_x_plus_2y_l130_13032

theorem problem_min_x_plus_2y (x y : ℝ) (h : x^2 + 4 * y^2 - 2 * x + 8 * y + 1 = 0) : 
  x + 2 * y ≥ -2 * Real.sqrt 2 - 1 :=
sorry

end NUMINAMATH_GPT_problem_min_x_plus_2y_l130_13032


namespace NUMINAMATH_GPT_monthly_salary_l130_13035

variable (S : ℝ)
variable (Saves : ℝ)
variable (NewSaves : ℝ)

open Real

theorem monthly_salary (h1 : Saves = 0.30 * S)
                       (h2 : NewSaves = Saves - 0.25 * Saves)
                       (h3 : NewSaves = 400) :
    S = 1777.78 := by
    sorry

end NUMINAMATH_GPT_monthly_salary_l130_13035


namespace NUMINAMATH_GPT_problem_intersection_l130_13016

theorem problem_intersection (a b : ℝ) 
    (h1 : b = - 2 / a) 
    (h2 : b = a + 3) 
    : 1 / a - 1 / b = -3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_intersection_l130_13016


namespace NUMINAMATH_GPT_smallest_integer_is_17_l130_13023

theorem smallest_integer_is_17
  (a b c d : ℕ)
  (h1 : b = 33)
  (h2 : d = b + 3)
  (h3 : (a + b + c + d) = 120)
  (h4 : a ≤ b)
  (h5 : c > b)
  : a = 17 :=
sorry

end NUMINAMATH_GPT_smallest_integer_is_17_l130_13023
