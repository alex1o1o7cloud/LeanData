import Mathlib

namespace NUMINAMATH_GPT_teal_more_blue_l638_63828

def numSurveyed : ℕ := 150
def numGreen : ℕ := 90
def numBlue : ℕ := 50
def numBoth : ℕ := 40
def numNeither : ℕ := 20

theorem teal_more_blue : 40 + (numSurveyed - (numBoth + (numGreen - numBoth) + numNeither)) = 80 :=
by
  -- Here we simplify numerically until we get the required answer
  -- start with calculating the total accounted and remaining
  sorry

end NUMINAMATH_GPT_teal_more_blue_l638_63828


namespace NUMINAMATH_GPT_domain_lg_tan_minus_sqrt3_l638_63830

open Real

theorem domain_lg_tan_minus_sqrt3 :
  {x : ℝ | ∃ k : ℤ, k * π + π / 3 < x ∧ x < k * π + π / 2} =
    {x : ℝ | ∃ k : ℤ, k * π + π / 3 < x ∧ x < k * π + π / 2} :=
by
  sorry

end NUMINAMATH_GPT_domain_lg_tan_minus_sqrt3_l638_63830


namespace NUMINAMATH_GPT_sum_of_vars_l638_63841

variables (a b c d k p : ℝ)

theorem sum_of_vars (h1 : a^2 + b^2 + c^2 + d^2 = 390)
                    (h2 : ab + bc + ca + ad + bd + cd = 5)
                    (h3 : ad + bd + cd = k)
                    (h4 : (a * b * c * d)^2 = p) :
                    a + b + c + d = 20 :=
by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_sum_of_vars_l638_63841


namespace NUMINAMATH_GPT_men_seated_count_l638_63808

theorem men_seated_count (total_passengers : ℕ) (two_thirds_women : total_passengers * 2 / 3 = women)
                         (one_eighth_standing : total_passengers / 3 / 8 = standing_men) :
  total_passengers = 48 →
  women = 32 →
  standing_men = 2 →
  men_seated = (total_passengers - women) - standing_men →
  men_seated = 14 :=
by
  intros
  sorry

end NUMINAMATH_GPT_men_seated_count_l638_63808


namespace NUMINAMATH_GPT_find_concentration_of_second_mixture_l638_63845

noncomputable def concentration_of_second_mixture (total_volume : ℝ) (final_percent : ℝ) (pure_antifreeze : ℝ) (pure_antifreeze_amount : ℝ) : ℝ :=
  let remaining_volume := total_volume - pure_antifreeze_amount
  let final_pure_amount := final_percent * total_volume
  let required_pure_antifreeze := final_pure_amount - pure_antifreeze
  (required_pure_antifreeze / remaining_volume) * 100

theorem find_concentration_of_second_mixture :
  concentration_of_second_mixture 55 0.20 6.11 6.11 = 10 :=
by
  simp [concentration_of_second_mixture]
  sorry

end NUMINAMATH_GPT_find_concentration_of_second_mixture_l638_63845


namespace NUMINAMATH_GPT_false_proposition_l638_63839

-- Definitions based on conditions
def opposite_angles (α β : ℝ) : Prop := α = β
def perpendicular (l m : ℝ → ℝ) : Prop := ∀ x, l x * m x = -1
def parallel (l m : ℝ → ℝ) : Prop := ∃ c, ∀ x, l x = m x + c
def corresponding_angles (α β : ℝ) : Prop := α = β

-- Propositions from the problem
def proposition1 : Prop := ∀ α β, opposite_angles α β → α = β
def proposition2 : Prop := ∀ l m n, perpendicular l n → perpendicular m n → parallel l m
def proposition3 : Prop := ∀ α β, α = β → opposite_angles α β
def proposition4 : Prop := ∀ α β, corresponding_angles α β → α = β

-- Statement to prove proposition 3 is false under given conditions
theorem false_proposition : ¬ proposition3 := by
  -- By our analysis, if proposition 3 is false, then it means the given definition for proposition 3 holds under all circumstances.
  sorry

end NUMINAMATH_GPT_false_proposition_l638_63839


namespace NUMINAMATH_GPT_int_solutions_to_inequalities_l638_63891

theorem int_solutions_to_inequalities :
  { x : ℤ | -5 * x ≥ 3 * x + 15 } ∩
  { x : ℤ | -3 * x ≤ 9 } ∩
  { x : ℤ | 7 * x ≤ -14 } = { -3, -2 } :=
by {
  sorry
}

end NUMINAMATH_GPT_int_solutions_to_inequalities_l638_63891


namespace NUMINAMATH_GPT_describe_T_l638_63894

def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (common : ℝ), 
    (common = 5 ∧ p.1 + 3 = common ∧ p.2 - 6 ≤ common) ∨
    (common = 5 ∧ p.2 - 6 = common ∧ p.1 + 3 ≤ common) ∨
    (common = p.1 + 3 ∧ common = p.2 - 6 ∧ common ≤ 5)}

theorem describe_T :
  T = {(2, y) | y ≤ 11} ∪ { (x, 11) | x ≤ 2} ∪ { (x, x + 9) | x ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_describe_T_l638_63894


namespace NUMINAMATH_GPT_problem_solved_by_at_least_one_student_l638_63898

theorem problem_solved_by_at_least_one_student (P_A P_B : ℝ) 
  (hA : P_A = 0.8) 
  (hB : P_B = 0.9) :
  (1 - (1 - P_A) * (1 - P_B) = 0.98) :=
by
  have pAwrong := 1 - P_A
  have pBwrong := 1 - P_B
  have both_wrong := pAwrong * pBwrong
  have one_right := 1 - both_wrong
  sorry

end NUMINAMATH_GPT_problem_solved_by_at_least_one_student_l638_63898


namespace NUMINAMATH_GPT_solve_for_x_l638_63843

def equation (x : ℝ) (y : ℝ) : Prop := 5 * y^2 + y + 10 = 2 * (9 * x^2 + y + 6)

def y_condition (x : ℝ) : ℝ := 3 * x

theorem solve_for_x (x : ℝ) :
  equation x (y_condition x) ↔ (x = 1/3 ∨ x = -2/9) := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l638_63843


namespace NUMINAMATH_GPT_function_decreasing_interval_l638_63855

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

theorem function_decreasing_interval :
  ∃ I : Set ℝ, I = (Set.Ioo 0 2) ∧ ∀ x ∈ I, deriv f x < 0 :=
by
  sorry

end NUMINAMATH_GPT_function_decreasing_interval_l638_63855


namespace NUMINAMATH_GPT_find_ordered_pair_l638_63874

theorem find_ordered_pair:
  ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ 10 * m * n = 45 - 5 * m - 3 * n ∧ (m, n) = (1, 11) :=
by
  sorry

end NUMINAMATH_GPT_find_ordered_pair_l638_63874


namespace NUMINAMATH_GPT_prime_mod_30_not_composite_l638_63895

theorem prime_mod_30_not_composite (p : ℕ) (h_prime : Prime p) (h_gt_30 : p > 30) : 
  ¬ ∃ (x : ℕ), (x > 1 ∧ ∃ (a b : ℕ), x = a * b ∧ a > 1 ∧ b > 1) ∧ (0 < x ∧ x < 30 ∧ ∃ (k : ℕ), p = 30 * k + x) :=
by
  sorry

end NUMINAMATH_GPT_prime_mod_30_not_composite_l638_63895


namespace NUMINAMATH_GPT_range_of_a_l638_63861

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 1| + |x - 2| > a) → a < 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l638_63861


namespace NUMINAMATH_GPT_solve_for_a_l638_63810

variable (x y a : ℤ)
variable (hx : x = 1)
variable (hy : y = -3)
variable (eq : a * x - y = 1)
 
theorem solve_for_a : a = -2 := by
  -- Placeholder to satisfy the lean prover, no actual proof steps
  sorry

end NUMINAMATH_GPT_solve_for_a_l638_63810


namespace NUMINAMATH_GPT_if_a_eq_b_then_ac_eq_bc_l638_63859

theorem if_a_eq_b_then_ac_eq_bc (a b c : ℝ) : a = b → ac = bc :=
sorry

end NUMINAMATH_GPT_if_a_eq_b_then_ac_eq_bc_l638_63859


namespace NUMINAMATH_GPT_sum_of_first_nine_terms_l638_63800

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ (n : ℕ), a n = a 1 + d * (n - 1)

variables (a : ℕ → ℝ) (h_seq : arithmetic_sequence a)

-- Given condition: a₂ + a₃ + a₇ + a₈ = 20
def condition : Prop := a 2 + a 3 + a 7 + a 8 = 20

-- Statement: Prove that the sum of the first 9 terms is 45
theorem sum_of_first_nine_terms (h : condition a) : 
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9) = 45 :=
by sorry

end NUMINAMATH_GPT_sum_of_first_nine_terms_l638_63800


namespace NUMINAMATH_GPT_susan_total_distance_l638_63818

theorem susan_total_distance (a b : ℕ) (r : ℝ) (h1 : a = 15) (h2 : b = 25) (h3 : r = 3) :
  (r * ((a + b) / 60)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_susan_total_distance_l638_63818


namespace NUMINAMATH_GPT_minimum_value_of_quadratic_l638_63857

def quadratic_polynomial (x : ℝ) : ℝ := 2 * x^2 - 16 * x + 22

theorem minimum_value_of_quadratic : ∃ x : ℝ, quadratic_polynomial x = -10 :=
by 
  use 4
  { sorry }

end NUMINAMATH_GPT_minimum_value_of_quadratic_l638_63857


namespace NUMINAMATH_GPT_find_expression_value_l638_63856

theorem find_expression_value (x y : ℚ) (h₁ : 3 * x + y = 6) (h₂ : x + 3 * y = 8) :
  9 * x ^ 2 + 15 * x * y + 9 * y ^ 2 = 1629 / 16 := 
sorry

end NUMINAMATH_GPT_find_expression_value_l638_63856


namespace NUMINAMATH_GPT_compute_product_sum_l638_63807

theorem compute_product_sum (a b c : ℕ) (ha : a = 3) (hb : b = 4) (hc : c = 5) :
  (a * b * c) * ((1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c) = 47 :=
by
  sorry

end NUMINAMATH_GPT_compute_product_sum_l638_63807


namespace NUMINAMATH_GPT_correct_calculation_l638_63819

theorem correct_calculation (a : ℝ) : a^3 / a^2 = a := by
  sorry

end NUMINAMATH_GPT_correct_calculation_l638_63819


namespace NUMINAMATH_GPT_trajectory_of_circle_center_is_ellipse_l638_63813

theorem trajectory_of_circle_center_is_ellipse 
    (a b : ℝ) (θ : ℝ) 
    (h1 : a ≠ b)
    (h2 : 0 < a)
    (h3 : 0 < b)
    : ∃ (x y : ℝ), 
    (x, y) = (a * Real.cos θ, b * Real.sin θ) ∧ 
    (x^2 / a^2 + y^2 / b^2 = 1) :=
sorry

end NUMINAMATH_GPT_trajectory_of_circle_center_is_ellipse_l638_63813


namespace NUMINAMATH_GPT_decagon_diagonals_l638_63871

-- Define the number of sides of the polygon
def n : ℕ := 10

-- Calculate the number of diagonals in an n-sided polygon
def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- State the theorem that the number of diagonals in a decagon is 35
theorem decagon_diagonals : number_of_diagonals n = 35 := by
  sorry

end NUMINAMATH_GPT_decagon_diagonals_l638_63871


namespace NUMINAMATH_GPT_Sam_bought_cards_l638_63842

theorem Sam_bought_cards (original_cards current_cards : ℕ) 
  (h1 : original_cards = 87) (h2 : current_cards = 74) : 
  original_cards - current_cards = 13 :=
by
  -- The 'sorry' here means the proof is omitted.
  sorry

end NUMINAMATH_GPT_Sam_bought_cards_l638_63842


namespace NUMINAMATH_GPT_grandfather_age_l638_63890

theorem grandfather_age :
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ 10 * a + b = a + b^2 ∧ 10 * a + b = 89 :=
by
  sorry

end NUMINAMATH_GPT_grandfather_age_l638_63890


namespace NUMINAMATH_GPT_total_salaries_proof_l638_63873

def total_salaries (A_salary B_salary : ℝ) :=
  A_salary + B_salary

theorem total_salaries_proof : ∀ A_salary B_salary : ℝ,
  A_salary = 3000 →
  (0.05 * A_salary = 0.15 * B_salary) →
  total_salaries A_salary B_salary = 4000 :=
by
  intros A_salary B_salary h1 h2
  rw [h1] at h2
  sorry

end NUMINAMATH_GPT_total_salaries_proof_l638_63873


namespace NUMINAMATH_GPT_slices_per_person_l638_63881

theorem slices_per_person (total_slices : ℕ) (total_people : ℕ) (h_slices : total_slices = 12) (h_people : total_people = 3) :
  total_slices / total_people = 4 :=
by
  sorry

end NUMINAMATH_GPT_slices_per_person_l638_63881


namespace NUMINAMATH_GPT_oppose_estimation_l638_63837

-- Define the conditions
def survey_total : ℕ := 50
def favorable_attitude : ℕ := 15
def total_population : ℕ := 9600

-- Calculate the proportion opposed
def proportion_opposed : ℚ := (survey_total - favorable_attitude) / survey_total

-- Define the statement to be proved
theorem oppose_estimation : 
  proportion_opposed * total_population = 6720 := by
  sorry

end NUMINAMATH_GPT_oppose_estimation_l638_63837


namespace NUMINAMATH_GPT_number_of_sets_count_number_of_sets_l638_63884

theorem number_of_sets (P : Set ℕ) :
  ({1, 2} ⊆ P ∧ P ⊆ {1, 2, 3, 4}) → (P = {1, 2} ∨ P = {1, 2, 3} ∨ P = {1, 2, 4}) :=
sorry

theorem count_number_of_sets :
  ∃ (Ps : Finset (Set ℕ)), 
  (∀ P ∈ Ps, {1, 2} ⊆ P ∧ P ⊆ {1, 2, 3, 4}) ∧ Ps.card = 3 :=
sorry

end NUMINAMATH_GPT_number_of_sets_count_number_of_sets_l638_63884


namespace NUMINAMATH_GPT_geometric_series_evaluation_l638_63809

theorem geometric_series_evaluation (c d : ℝ) (h : (∑' n : ℕ, c / d^(n + 1)) = 3) :
  (∑' n : ℕ, c / (c + 2 * d)^(n + 1)) = (3 * d - 3) / (5 * d - 4) :=
sorry

end NUMINAMATH_GPT_geometric_series_evaluation_l638_63809


namespace NUMINAMATH_GPT_quadratic_cubic_expression_l638_63865

theorem quadratic_cubic_expression
  (r s : ℝ)
  (h_eq : ∀ x : ℝ, 3 * x^2 - 4 * x - 12 = 0 → x = r ∨ x = s) :
  (9 * r^3 - 9 * s^3) / (r - s) = 52 :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_cubic_expression_l638_63865


namespace NUMINAMATH_GPT_find_xyz_l638_63814

theorem find_xyz (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
  (h4 : x * (y + z) = 198) (h5 : y * (z + x) = 216) (h6 : z * (x + y) = 234) :
  x * y * z = 1080 :=
sorry

end NUMINAMATH_GPT_find_xyz_l638_63814


namespace NUMINAMATH_GPT_min_value_frac_function_l638_63838

theorem min_value_frac_function (x : ℝ) (h : x > -1) : (x^2 / (x + 1)) ≥ 0 :=
sorry

end NUMINAMATH_GPT_min_value_frac_function_l638_63838


namespace NUMINAMATH_GPT_find_value_of_a20_l638_63899

variable {α : Type*} [LinearOrder α] [Field α]

def arithmetic_sequence (a d : α) (n : ℕ) : α :=
  a + (n - 1) * d

def arithmetic_sum (a d : α) (n : ℕ) : α :=
  n * a + (n * (n - 1) / 2) * d

theorem find_value_of_a20 
  (a d : ℝ) 
  (h1 : arithmetic_sequence a d 3 + arithmetic_sequence a d 5 = 4)
  (h2 : arithmetic_sum a d 15 = 60) :
  arithmetic_sequence a d 20 = 10 := 
sorry

end NUMINAMATH_GPT_find_value_of_a20_l638_63899


namespace NUMINAMATH_GPT_find_s_l638_63883

theorem find_s (s : ℝ) (t : ℝ) (h1 : t = 4) (h2 : t = 12 * s^2 + 2 * s) : s = 0.5 ∨ s = -2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_s_l638_63883


namespace NUMINAMATH_GPT_problem_statement_l638_63812

theorem problem_statement
  (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : a + b + c = 0)
  (h5 : ab + ac + bc ≠ 0) :
  (a^7 + b^7 + c^7) / (abc * (ab + ac + bc)) = -7 :=
  sorry

end NUMINAMATH_GPT_problem_statement_l638_63812


namespace NUMINAMATH_GPT_even_function_f3_l638_63862

theorem even_function_f3 (a : ℝ) (h : ∀ x : ℝ, (x + 2) * (x - a) = (-x + 2) * (-x - a)) : (3 + 2) * (3 - a) = 5 := by
  sorry

end NUMINAMATH_GPT_even_function_f3_l638_63862


namespace NUMINAMATH_GPT_factor_expression_l638_63872

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l638_63872


namespace NUMINAMATH_GPT_union_of_A_and_B_l638_63836

section
variable {A B : Set ℝ}
variable (a b : ℝ)

def setA := {x : ℝ | x^2 - 3 * x + a = 0}
def setB := {x : ℝ | x^2 + b = 0}

theorem union_of_A_and_B:
  setA a ∩ setB b = {2} →
  setA a ∪ setB b = ({-2, 1, 2} : Set ℝ) := by
  sorry
end

end NUMINAMATH_GPT_union_of_A_and_B_l638_63836


namespace NUMINAMATH_GPT_roots_of_quadratic_l638_63849

theorem roots_of_quadratic (x1 x2 : ℝ) (h : ∀ x, x^2 - 3 * x - 2 = 0 → x = x1 ∨ x = x2) :
  x1 * x2 + x1 + x2 = 1 :=
sorry

end NUMINAMATH_GPT_roots_of_quadratic_l638_63849


namespace NUMINAMATH_GPT_solution_set_of_inequality_l638_63854

theorem solution_set_of_inequality : {x : ℝ | -2 < x ∧ x < 1} = {x : ℝ | -x^2 - x + 2 > 0} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l638_63854


namespace NUMINAMATH_GPT_milk_after_three_operations_l638_63815

-- Define the initial amount of milk and the proportion replaced each step
def initial_milk : ℝ := 100
def proportion_replaced : ℝ := 0.2

-- Define the amount of milk after each replacement operation
noncomputable def milk_after_n_operations (n : ℕ) (milk : ℝ) : ℝ :=
  if n = 0 then milk
  else (1 - proportion_replaced) * milk_after_n_operations (n - 1) milk

-- Define the statement about the amount of milk after three operations
theorem milk_after_three_operations : milk_after_n_operations 3 initial_milk = 51.2 :=
by
  sorry

end NUMINAMATH_GPT_milk_after_three_operations_l638_63815


namespace NUMINAMATH_GPT_polynomial_degrees_l638_63858

-- Define the degree requirement for the polynomial.
def polynomial_deg_condition (m n : ℕ) : Prop :=
  2 + m = 5 ∧ n - 2 = 0 ∧ 2 + 2 = 5

theorem polynomial_degrees (m n : ℕ) (h : polynomial_deg_condition m n) : m - n = 1 :=
by
  have h1 : 2 + m = 5 := h.1
  have h2 : n - 2 = 0 := h.2.1
  have h3 := h.2.2
  have : m = 3 := by linarith
  have : n = 2 := by linarith
  linarith

end NUMINAMATH_GPT_polynomial_degrees_l638_63858


namespace NUMINAMATH_GPT_find_x_l638_63864

def vector := (ℝ × ℝ)

def collinear (u v : vector) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2)

def a : vector := (1, 2)
def b (x : ℝ) : vector := (x, 1)
def a_minus_b (x : ℝ) : vector := ((1 - x), 1)

theorem find_x (x : ℝ) (h : collinear a (a_minus_b x)) : x = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l638_63864


namespace NUMINAMATH_GPT_remainders_identical_l638_63804

theorem remainders_identical (a b : ℕ) (h1 : a > b) :
  ∃ r₁ r₂ q₁ q₂ : ℕ, 
  a = (a - b) * q₁ + r₁ ∧ 
  b = (a - b) * q₂ + r₂ ∧ 
  r₁ = r₂ := by 
sorry

end NUMINAMATH_GPT_remainders_identical_l638_63804


namespace NUMINAMATH_GPT_gerald_remaining_pfennigs_l638_63876

-- Definitions of Gerald's initial money and the costs of items
def farthings : Nat := 54
def groats : Nat := 8
def florins : Nat := 17
def meat_pie_cost : Nat := 120
def sausage_roll_cost : Nat := 75

-- Conversion rates
def farthings_to_pfennigs (f : Nat) : Nat := f / 6
def groats_to_pfennigs (g : Nat) : Nat := g * 4
def florins_to_pfennigs (f : Nat) : Nat := f * 40

-- Total pfennigs Gerald has
def total_pfennigs : Nat :=
  farthings_to_pfennigs farthings + groats_to_pfennigs groats + florins_to_pfennigs florins

-- Total cost of both items
def total_cost : Nat := meat_pie_cost + sausage_roll_cost

-- Gerald's remaining pfennigs after purchase
def remaining_pfennigs : Nat := total_pfennigs - total_cost

theorem gerald_remaining_pfennigs :
  remaining_pfennigs = 526 :=
by
  sorry

end NUMINAMATH_GPT_gerald_remaining_pfennigs_l638_63876


namespace NUMINAMATH_GPT_radius_of_sector_l638_63852

theorem radius_of_sector (l : ℝ) (α : ℝ) (R : ℝ) (h1 : l = 2 * π / 3) (h2 : α = π / 3) : R = 2 := by
  have : l = |α| * R := by sorry
  rw [h1, h2] at this
  sorry

end NUMINAMATH_GPT_radius_of_sector_l638_63852


namespace NUMINAMATH_GPT_min_value_of_a_l638_63875

theorem min_value_of_a (a b c : ℝ) (h₁ : a > 0) (h₂ : ∃ p q : ℝ, 0 < p ∧ p < 2 ∧ 0 < q ∧ q < 2 ∧ 
  ∀ x, ax^2 + bx + c = a * (x - p) * (x - q)) (h₃ : 25 * a + 10 * b + 4 * c ≥ 4) (h₄ : c ≥ 1) : 
  a ≥ 16 / 25 :=
sorry

end NUMINAMATH_GPT_min_value_of_a_l638_63875


namespace NUMINAMATH_GPT_constant_S13_l638_63850

noncomputable def S (a d : ℝ) (n : ℕ) : ℝ := n * (2 * a + (n - 1) * d) / 2

theorem constant_S13 (a d p : ℝ) 
  (h : a + a + 3 * d + a + 7 * d = p) : 
  S a d 13 = 13 * p / 18 :=
by
  unfold S
  sorry

end NUMINAMATH_GPT_constant_S13_l638_63850


namespace NUMINAMATH_GPT_minimum_value_expr_l638_63863

theorem minimum_value_expr (a : ℝ) (h₀ : 0 < a) (h₁ : a < 2) : 
  ∃ (m : ℝ), m = (4 / a + 1 / (2 - a)) ∧ m = 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_expr_l638_63863


namespace NUMINAMATH_GPT_remainder_when_divided_by_63_l638_63848

theorem remainder_when_divided_by_63 (x : ℤ) (h1 : ∃ q : ℤ, x = 63 * q + r ∧ 0 ≤ r ∧ r < 63) (h2 : ∃ k : ℤ, x = 9 * k + 2) :
  ∃ r : ℤ, 0 ≤ r ∧ r < 63 ∧ r = 7 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_63_l638_63848


namespace NUMINAMATH_GPT_loci_of_square_view_l638_63882

-- Definitions based on the conditions in a)
def square (A B C D : Point) : Prop := -- Formalize what it means to be a square
sorry

def region1 (P : Point) (A B : Point) : Prop := -- Formalize the definition of region 1
sorry

def region2 (P : Point) (B C : Point) : Prop := -- Formalize the definition of region 2
sorry

-- Additional region definitions (3 through 9)
-- ...

def visible_side (P A B : Point) : Prop := -- Definition of a visible side from a point
sorry

def visible_diagonal (P A C : Point) : Prop := -- Definition of a visible diagonal from a point
sorry

def loci_of_angles (angle : ℝ) : Set Point := -- Definition of loci for a given angle
sorry

-- Main problem statement with the question and conditions as hypotheses
theorem loci_of_square_view (A B C D P : Point) (angle : ℝ) :
    square A B C D →
    (∀ P, (visible_side P A B ∨ visible_side P B C ∨ visible_side P C D ∨ visible_side P D A → 
             P ∈ loci_of_angles angle) ∧ 
         ((region1 P A B ∨ region2 P B C) → visible_diagonal P A C)) →
    -- Additional conditions here
    True :=
-- Prove that the loci is as described in the solution
sorry

end NUMINAMATH_GPT_loci_of_square_view_l638_63882


namespace NUMINAMATH_GPT_parabola_equation_l638_63825

theorem parabola_equation (m : ℝ) (focus : ℝ × ℝ) (M : ℝ × ℝ) 
  (h_vertex : (0, 0) = (0, 0))
  (h_focus : focus = (p, 0))
  (h_point : M = (1, m))
  (h_distance : dist M focus = 2) 
  : (forall x y : ℝ, y^2 = 4*x) :=
sorry

end NUMINAMATH_GPT_parabola_equation_l638_63825


namespace NUMINAMATH_GPT_option_A_correct_l638_63801

variable (f g : ℝ → ℝ)

-- Given conditions
axiom cond1 : ∀ x : ℝ, f x - g (4 - x) = 2
axiom cond2 : ∀ x : ℝ, deriv g x = deriv f (x - 2)
axiom cond3 : ∀ x : ℝ, f (x + 2) = - f (- x - 2)

theorem option_A_correct : ∀ x : ℝ, f (4 + x) + f (- x) = 0 :=
by
  -- Proving the theorem
  sorry

end NUMINAMATH_GPT_option_A_correct_l638_63801


namespace NUMINAMATH_GPT_washing_machine_capacity_l638_63820

-- Define the problem conditions
def families : Nat := 3
def people_per_family : Nat := 4
def days : Nat := 7
def towels_per_person_per_day : Nat := 1
def loads : Nat := 6

-- Define the statement to prove
theorem washing_machine_capacity :
  (families * people_per_family * days * towels_per_person_per_day) / loads = 14 := by
  sorry

end NUMINAMATH_GPT_washing_machine_capacity_l638_63820


namespace NUMINAMATH_GPT_janet_income_difference_l638_63817

def janet_current_job_income (hours_per_week : ℕ) (weeks_per_month : ℕ) (hourly_rate : ℝ) : ℝ :=
  hours_per_week * weeks_per_month * hourly_rate

def janet_freelance_income (hours_per_week : ℕ) (weeks_per_month : ℕ) (hourly_rate : ℝ) : ℝ :=
  hours_per_week * weeks_per_month * hourly_rate

def extra_fica_taxes (weekly_tax : ℝ) (weeks_per_month : ℕ) : ℝ :=
  weekly_tax * weeks_per_month

def healthcare_premiums (monthly_premium : ℝ) : ℝ :=
  monthly_premium

def janet_net_freelance_income (freelance_income : ℝ) (additional_costs : ℝ) : ℝ :=
  freelance_income - additional_costs

theorem janet_income_difference
  (hours_per_week : ℕ)
  (weeks_per_month : ℕ)
  (current_hourly_rate : ℝ)
  (freelance_hourly_rate : ℝ)
  (weekly_tax : ℝ)
  (monthly_premium : ℝ)
  (H_hours : hours_per_week = 40)
  (H_weeks : weeks_per_month = 4)
  (H_current_rate : current_hourly_rate = 30)
  (H_freelance_rate : freelance_hourly_rate = 40)
  (H_weekly_tax : weekly_tax = 25)
  (H_monthly_premium : monthly_premium = 400) :
  janet_net_freelance_income (janet_freelance_income 40 4 40) (extra_fica_taxes 25 4 + healthcare_premiums 400) 
  - janet_current_job_income 40 4 30 = 1100 := 
  by 
    sorry

end NUMINAMATH_GPT_janet_income_difference_l638_63817


namespace NUMINAMATH_GPT_smallest_n_satisfies_conditions_l638_63840

theorem smallest_n_satisfies_conditions :
  ∃ (n : ℕ), (∀ m : ℕ, (5 * m = 5 * n → m = n) ∧ (3 * m = 3 * n → m = n)) ∧
  (n = 45) :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_satisfies_conditions_l638_63840


namespace NUMINAMATH_GPT_find_f_2015_l638_63803

variables (f : ℝ → ℝ)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 6) = f x + f 3

theorem find_f_2015 (h1 : is_even_function f) (h2 : satisfies_condition f) (h3 : f 1 = 2) : f 2015 = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_f_2015_l638_63803


namespace NUMINAMATH_GPT_minimum_value_of_expression_l638_63821

noncomputable def min_value (a b : ℝ) : ℝ := 1 / a + 3 / b

theorem minimum_value_of_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 3 * b = 1) : min_value a b ≥ 16 := 
sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l638_63821


namespace NUMINAMATH_GPT_car_distances_equal_600_l638_63896

-- Define the variables
def time_R (t : ℝ) := t
def speed_R := 50
def time_P (t : ℝ) := t - 2
def speed_P := speed_R + 10
def distance (t : ℝ) := speed_R * time_R t

-- The Lean theorem statement
theorem car_distances_equal_600 (t : ℝ) (h : time_R t = t) (h1 : speed_R = 50) 
  (h2 : time_P t = t - 2) (h3 : speed_P = speed_R + 10) :
  distance t = 600 :=
by
  -- We would provide the proof here, but for now we use sorry to indicate the proof is omitted.
  sorry

end NUMINAMATH_GPT_car_distances_equal_600_l638_63896


namespace NUMINAMATH_GPT_arcsin_cos_eq_neg_pi_div_six_l638_63866

theorem arcsin_cos_eq_neg_pi_div_six :
  Real.arcsin (Real.cos (2 * Real.pi / 3)) = -Real.pi / 6 :=
by
  sorry

end NUMINAMATH_GPT_arcsin_cos_eq_neg_pi_div_six_l638_63866


namespace NUMINAMATH_GPT_log_expression_evaluation_l638_63823

theorem log_expression_evaluation (log2 log5 : ℝ) (h : log2 + log5 = 1) :
  log2 * (log5 + log10) + 2 * log5 - log5 * log20 = 1 := by
  sorry

end NUMINAMATH_GPT_log_expression_evaluation_l638_63823


namespace NUMINAMATH_GPT_cookies_prepared_l638_63860

theorem cookies_prepared (n_people : ℕ) (cookies_per_person : ℕ) (total_cookies : ℕ) 
  (h1 : n_people = 25) (h2 : cookies_per_person = 45) : total_cookies = 1125 :=
by
  sorry

end NUMINAMATH_GPT_cookies_prepared_l638_63860


namespace NUMINAMATH_GPT_trig_identity_l638_63827

open Real

theorem trig_identity (α : ℝ) (h : 2 * sin α + cos α = 0) : 
  2 * sin α ^ 2 - 3 * sin α * cos α - 5 * cos α ^ 2 = -12 / 5 :=
sorry

end NUMINAMATH_GPT_trig_identity_l638_63827


namespace NUMINAMATH_GPT_find_point_A_coordinates_l638_63822

theorem find_point_A_coordinates :
  ∃ (A : ℝ × ℝ), (A.2 = 0) ∧ 
  (dist A (-3, 2) = dist A (4, -5)) →
  A = (2, 0) :=
by
-- We'll provide the explicit exact proof later
-- Proof steps would go here
sorry 

end NUMINAMATH_GPT_find_point_A_coordinates_l638_63822


namespace NUMINAMATH_GPT_A_wins_if_N_is_perfect_square_l638_63888

noncomputable def player_A_can_always_win (N : ℕ) : Prop :=
  ∀ (B_moves : ℕ → ℕ), ∃ (A_moves : ℕ → ℕ), A_moves 0 = N ∧
  (∀ n, B_moves n = 0 ∨ (A_moves n ∣ B_moves (n + 1) ∨ B_moves (n + 1) ∣ A_moves n))

theorem A_wins_if_N_is_perfect_square :
  ∀ N : ℕ, player_A_can_always_win N ↔ ∃ n : ℕ, N = n * n := sorry

end NUMINAMATH_GPT_A_wins_if_N_is_perfect_square_l638_63888


namespace NUMINAMATH_GPT_max_value_S_n_l638_63811

theorem max_value_S_n 
  (a : ℕ → ℕ)
  (a1 : a 1 = 2)
  (S : ℕ → ℕ)
  (h : ∀ n, 6 * S n = 3 * a (n + 1) + 4 ^ n - 1) :
  ∃ n, S n = 10 := 
sorry

end NUMINAMATH_GPT_max_value_S_n_l638_63811


namespace NUMINAMATH_GPT_last_operation_ends_at_eleven_am_l638_63844

-- Definitions based on conditions
def operation_duration : ℕ := 45 -- duration of each operation in minutes
def start_time : ℕ := 8 * 60 -- start time of the first operation in minutes since midnight
def interval : ℕ := 15 -- interval between operations in minutes
def total_operations : ℕ := 10 -- total number of operations

-- Compute the start time of the last operation (10th operation)
def start_time_last_operation : ℕ := start_time + interval * (total_operations - 1)

-- Compute the end time of the last operation
def end_time_last_operation : ℕ := start_time_last_operation + operation_duration

-- End time of the last operation expected to be 11:00 a.m. in minutes since midnight
def expected_end_time : ℕ := 11 * 60 

theorem last_operation_ends_at_eleven_am : 
  end_time_last_operation = expected_end_time := by
  sorry

end NUMINAMATH_GPT_last_operation_ends_at_eleven_am_l638_63844


namespace NUMINAMATH_GPT_max_m_value_l638_63806

theorem max_m_value {m : ℝ} : 
  (∀ x : ℝ, (x^2 - 2 * x - 8 > 0 → x < m)) ∧ ¬(∀ x : ℝ, (x^2 - 2 * x - 8 > 0 ↔ x < m)) → m ≤ -2 :=
sorry

end NUMINAMATH_GPT_max_m_value_l638_63806


namespace NUMINAMATH_GPT_eval_at_d_eq_4_l638_63878

theorem eval_at_d_eq_4 : ((4: ℕ) ^ 4 - (4: ℕ) * ((4: ℕ) - 2) ^ 4) ^ 4 = 136048896 :=
by
  sorry

end NUMINAMATH_GPT_eval_at_d_eq_4_l638_63878


namespace NUMINAMATH_GPT_chastity_lollipops_l638_63868

theorem chastity_lollipops (initial_money lollipop_cost gummy_cost left_money total_gummies total_spent lollipops : ℝ)
  (h1 : initial_money = 15)
  (h2 : lollipop_cost = 1.50)
  (h3 : gummy_cost = 2)
  (h4 : left_money = 5)
  (h5 : total_gummies = 2)
  (h6 : total_spent = initial_money - left_money)
  (h7 : total_spent = 10)
  (h8 : total_gummies * gummy_cost = 4)
  (h9 : total_spent - (total_gummies * gummy_cost) = 6)
  (h10 : lollipops = (total_spent - (total_gummies * gummy_cost)) / lollipop_cost) :
  lollipops = 4 := 
sorry

end NUMINAMATH_GPT_chastity_lollipops_l638_63868


namespace NUMINAMATH_GPT_lab_techs_share_l638_63816

theorem lab_techs_share (u c t : ℕ) 
  (h1 : c = 6 * u)
  (h2 : t = u / 2)
  (h3 : u = 12) : 
  (c + u) / t = 14 := 
by 
  sorry

end NUMINAMATH_GPT_lab_techs_share_l638_63816


namespace NUMINAMATH_GPT_exponentiation_multiplication_l638_63802

theorem exponentiation_multiplication (a : ℝ) : a^6 * a^2 = a^8 :=
by sorry

end NUMINAMATH_GPT_exponentiation_multiplication_l638_63802


namespace NUMINAMATH_GPT_op_op_k_l638_63887

def op (x y : ℝ) : ℝ := x^3 + x - y

theorem op_op_k (k : ℝ) : op k (op k k) = k := sorry

end NUMINAMATH_GPT_op_op_k_l638_63887


namespace NUMINAMATH_GPT_solution_set_l638_63853

def solve_inequalities (x : ℝ) : Prop :=
  (3 * x - 2) / (x - 6) ≤ 1 ∧ 2 * x ^ 2 - x - 1 > 0

theorem solution_set : { x : ℝ | solve_inequalities x } = { x : ℝ | (-2 ≤ x ∧ x < 1/2) ∨ (1 < x ∧ x < 6) } :=
by sorry

end NUMINAMATH_GPT_solution_set_l638_63853


namespace NUMINAMATH_GPT_S2_side_length_656_l638_63879

noncomputable def S1_S2_S3_side_lengths (l1 l2 a b c : ℕ) (total_length : ℕ) : Prop :=
  l1 + l2 + a + b + c = total_length

theorem S2_side_length_656 :
  ∃ (l1 l2 a c : ℕ), S1_S2_S3_side_lengths l1 l2 a 656 c 3322 :=
by
  sorry

end NUMINAMATH_GPT_S2_side_length_656_l638_63879


namespace NUMINAMATH_GPT_Marie_speed_l638_63877

theorem Marie_speed (distance time : ℕ) (h1 : distance = 372) (h2 : time = 31) : distance / time = 12 :=
by
  have h3 : distance = 372 := h1
  have h4 : time = 31 := h2
  sorry

end NUMINAMATH_GPT_Marie_speed_l638_63877


namespace NUMINAMATH_GPT_balls_in_boxes_l638_63846

theorem balls_in_boxes : (2^7 = 128) := 
by
  -- number of balls
  let n : ℕ := 7
  -- number of boxes
  let b : ℕ := 2
  have h : b ^ n = 128 := by sorry
  exact h

end NUMINAMATH_GPT_balls_in_boxes_l638_63846


namespace NUMINAMATH_GPT_abc_inequality_l638_63824

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a + b + c ≤ (a^4 + b^4 + c^4) / (a * b * c) :=
sorry

end NUMINAMATH_GPT_abc_inequality_l638_63824


namespace NUMINAMATH_GPT_fraction_of_dark_tiles_is_correct_l638_63897

def num_tiles_in_block : ℕ := 64
def num_dark_tiles : ℕ := 18
def expected_fraction_dark_tiles : ℚ := 9 / 32

theorem fraction_of_dark_tiles_is_correct :
  (num_dark_tiles : ℚ) / num_tiles_in_block = expected_fraction_dark_tiles := by
sorry

end NUMINAMATH_GPT_fraction_of_dark_tiles_is_correct_l638_63897


namespace NUMINAMATH_GPT_exist_m_squared_plus_9_mod_2_pow_n_minus_1_l638_63892

theorem exist_m_squared_plus_9_mod_2_pow_n_minus_1 (n : ℕ) (hn : n > 0) :
  (∃ m : ℤ, (m^2 + 9) % (2^n - 1) = 0) ↔ ∃ k : ℕ, n = 2^k :=
by
  sorry

end NUMINAMATH_GPT_exist_m_squared_plus_9_mod_2_pow_n_minus_1_l638_63892


namespace NUMINAMATH_GPT_arun_weight_lower_limit_l638_63829

theorem arun_weight_lower_limit :
  ∃ (w : ℝ), w > 60 ∧ w <= 64 ∧ (∀ (a : ℝ), 60 < a ∧ a <= 64 → ((a + 64) / 2 = 63) → a = 62) :=
by
  sorry

end NUMINAMATH_GPT_arun_weight_lower_limit_l638_63829


namespace NUMINAMATH_GPT_total_students_class_is_63_l638_63867

def num_tables : ℕ := 6
def students_per_table : ℕ := 3
def girls_bathroom : ℕ := 4
def times_canteen : ℕ := 4
def group1_students : ℕ := 4
def group2_students : ℕ := 5
def group3_students : ℕ := 6
def germany_students : ℕ := 2
def france_students : ℕ := 4
def norway_students : ℕ := 3
def italy_students : ℕ := 1

def total_students_in_class : ℕ :=
  (num_tables * students_per_table) +
  girls_bathroom +
  (times_canteen * girls_bathroom) +
  (group1_students + group2_students + group3_students) +
  (germany_students + france_students + norway_students + italy_students)

theorem total_students_class_is_63 : total_students_in_class = 63 :=
  by
    sorry

end NUMINAMATH_GPT_total_students_class_is_63_l638_63867


namespace NUMINAMATH_GPT_total_cases_is_8_l638_63885

def num_blue_cards : Nat := 3
def num_yellow_cards : Nat := 5

def total_cases : Nat := num_blue_cards + num_yellow_cards

theorem total_cases_is_8 : total_cases = 8 := by
  sorry

end NUMINAMATH_GPT_total_cases_is_8_l638_63885


namespace NUMINAMATH_GPT_find_numbers_between_70_and_80_with_gcd_6_l638_63869

theorem find_numbers_between_70_and_80_with_gcd_6 :
  ∃ n, 70 ≤ n ∧ n ≤ 80 ∧ Nat.gcd n 30 = 6 ∧ (n = 72 ∨ n = 78) :=
by
  sorry

end NUMINAMATH_GPT_find_numbers_between_70_and_80_with_gcd_6_l638_63869


namespace NUMINAMATH_GPT_evaluate_expression_l638_63851

theorem evaluate_expression : 
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_l638_63851


namespace NUMINAMATH_GPT_least_positive_integer_condition_l638_63893

theorem least_positive_integer_condition (n : ℕ) :
  (∀ d ∈ [2, 3, 4, 5, 6, 7, 8, 9, 11], n % d = 1) → n = 10396 := 
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_condition_l638_63893


namespace NUMINAMATH_GPT_length_PR_in_triangle_l638_63847

/-- In any triangle PQR, given:
  PQ = 7, QR = 10, median PS = 5,
  the length of PR must be sqrt(149). -/
theorem length_PR_in_triangle (PQ QR PS : ℝ) (PQ_eq : PQ = 7) (QR_eq : QR = 10) (PS_eq : PS = 5) : 
  ∃ (PR : ℝ), PR = Real.sqrt 149 := 
sorry

end NUMINAMATH_GPT_length_PR_in_triangle_l638_63847


namespace NUMINAMATH_GPT_donut_combinations_l638_63886

-- Define the problem statement where Bill needs to purchase 10 donuts,
-- with at least one of each of the 5 kinds, and calculate the combinations.

def count_donut_combinations : ℕ :=
  Nat.choose 9 4

theorem donut_combinations :
  count_donut_combinations = 126 :=
by
  -- Proof can be filled in here
  sorry

end NUMINAMATH_GPT_donut_combinations_l638_63886


namespace NUMINAMATH_GPT_mean_greater_than_median_by_two_l638_63832

theorem mean_greater_than_median_by_two (x : ℕ) (h : x > 0) :
  ((x + (x + 2) + (x + 4) + (x + 7) + (x + 17)) / 5 - (x + 4)) = 2 :=
sorry

end NUMINAMATH_GPT_mean_greater_than_median_by_two_l638_63832


namespace NUMINAMATH_GPT_value_of_expression_l638_63834

noncomputable def a : ℝ := Real.log 3 / Real.log 4

theorem value_of_expression (h : a = Real.log 3 / Real.log 4) : 2^a + 2^(-a) = 4 * Real.sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l638_63834


namespace NUMINAMATH_GPT_part1_part2_l638_63805

noncomputable def f (x : ℝ) : ℝ := |x - 2|
noncomputable def g (x m : ℝ) : ℝ := -|x + 3| + m

def solution_set_ineq_1 (a : ℝ) : Set ℝ :=
  if a = 1 then {x | x < 2 ∨ x > 2}
  else if a > 1 then Set.univ
  else {x | x < 1 + a ∨ x > 3 - a}

theorem part1 (a : ℝ) : 
  ∃ S : Set ℝ, S = solution_set_ineq_1 a ∧ ∀ x : ℝ, (f x + a - 1 > 0) ↔ x ∈ S := sorry

theorem part2 (m : ℝ) : 
  (∀ x : ℝ, f x ≥ g x m) ↔ m < 5 := sorry

end NUMINAMATH_GPT_part1_part2_l638_63805


namespace NUMINAMATH_GPT_expected_value_unfair_die_l638_63835

theorem expected_value_unfair_die :
  let p8 := 3 / 8
  let p1_7 := (1 - p8) / 7
  let E := p1_7 * (1 + 2 + 3 + 4 + 5 + 6 + 7) + p8 * 8
  E = 5.5 := by
  sorry

end NUMINAMATH_GPT_expected_value_unfair_die_l638_63835


namespace NUMINAMATH_GPT_p_sufficient_not_necessary_for_q_l638_63831

open Real

def p (x : ℝ) : Prop := abs x < 1
def q (x : ℝ) : Prop := x^2 + x - 6 < 0

theorem p_sufficient_not_necessary_for_q : 
  (∀ x : ℝ, p x → q x) ∧ ¬(∀ x : ℝ, q x → p x) :=
by
  sorry

end NUMINAMATH_GPT_p_sufficient_not_necessary_for_q_l638_63831


namespace NUMINAMATH_GPT_equation_one_solution_equation_two_no_solution_l638_63889

theorem equation_one_solution (x : ℝ) (hx1 : x ≠ 3) : (2 * x + 9) / (3 - x) = (4 * x - 7) / (x - 3) ↔ x = -1 / 3 := 
by 
    sorry

theorem equation_two_no_solution (x : ℝ) (hx2 : x ≠ 1) (hx3 : x ≠ -1) : 
    (x + 1) / (x - 1) - 4 / (x ^ 2 - 1) = 1 → False := 
by 
    sorry

end NUMINAMATH_GPT_equation_one_solution_equation_two_no_solution_l638_63889


namespace NUMINAMATH_GPT_lowest_possible_price_l638_63870

-- Definitions based on the provided conditions
def regular_discount_range : Set Real := {x | 0.10 ≤ x ∧ x ≤ 0.30}
def additional_discount : Real := 0.20
def retail_price : Real := 35.00

-- Problem statement transformed into Lean
theorem lowest_possible_price :
  ∃ d ∈ regular_discount_range, (retail_price * (1 - d)) * (1 - additional_discount) = 19.60 :=
by
  sorry

end NUMINAMATH_GPT_lowest_possible_price_l638_63870


namespace NUMINAMATH_GPT_area_of_rectangle_l638_63880

theorem area_of_rectangle
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h_a : a = 16)
  (h_c : c = 17)
  (h_diag : a^2 + b^2 = c^2) :
  abs (a * b - 91.9136) < 0.0001 :=
by
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l638_63880


namespace NUMINAMATH_GPT_distributor_income_proof_l638_63826

noncomputable def income_2017 (a k x : ℝ) : ℝ :=
  (a + k / (x - 7)) * (x - 5)

theorem distributor_income_proof (a : ℝ) (x : ℝ) (h_range : 10 ≤ x ∧ x ≤ 14) (h_k : k = 3 * a):
  income_2017 a (3 * a) x = 12 * a ↔ x = 13 := by
  sorry

end NUMINAMATH_GPT_distributor_income_proof_l638_63826


namespace NUMINAMATH_GPT_max_books_borrowed_l638_63833

noncomputable def max_books_per_student : ℕ := 14

theorem max_books_borrowed (students_borrowed_0 : ℕ)
                           (students_borrowed_1 : ℕ)
                           (students_borrowed_2 : ℕ)
                           (total_students : ℕ)
                           (average_books : ℕ)
                           (remaining_students_borrowed_at_least_3 : ℕ)
                           (total_books : ℕ)
                           (max_books : ℕ) 
  (h1 : students_borrowed_0 = 2)
  (h2 : students_borrowed_1 = 10)
  (h3 : students_borrowed_2 = 5)
  (h4 : total_students = 20)
  (h5 : average_books = 2)
  (h6 : remaining_students_borrowed_at_least_3 = total_students - students_borrowed_0 - students_borrowed_1 - students_borrowed_2)
  (h7 : total_books = total_students * average_books)
  (h8 : total_books = (students_borrowed_1 * 1 + students_borrowed_2 * 2) + remaining_students_borrowed_at_least_3 * 3 + (max_books - 6))
  (h_max : max_books = max_books_per_student) :
  max_books ≤ max_books_per_student := 
sorry

end NUMINAMATH_GPT_max_books_borrowed_l638_63833
