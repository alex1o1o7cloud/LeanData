import Mathlib

namespace NUMINAMATH_GPT_find_b_from_conditions_l2254_225401

theorem find_b_from_conditions 
  (x y b : ℝ) 
  (h1 : 3 * x - 5 * y = b) 
  (h2 : x / (x + y) = 5 / 7) 
  (h3 : x - y = 3) : 
  b = 5 := 
by 
  sorry

end NUMINAMATH_GPT_find_b_from_conditions_l2254_225401


namespace NUMINAMATH_GPT_sin_300_eq_neg_sqrt3_div_2_l2254_225411

-- Defining the problem statement as a Lean theorem
theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_sin_300_eq_neg_sqrt3_div_2_l2254_225411


namespace NUMINAMATH_GPT_coprime_permutations_count_l2254_225452

noncomputable def count_coprime_permutations (l : List ℕ) : ℕ :=
if h : l = [1, 2, 3, 4, 5, 6, 7] ∨ l = [1, 2, 3, 7, 5, 6, 4] -- other permutations can be added as needed
then 864
else 0

theorem coprime_permutations_count :
  count_coprime_permutations [1, 2, 3, 4, 5, 6, 7] = 864 :=
sorry

end NUMINAMATH_GPT_coprime_permutations_count_l2254_225452


namespace NUMINAMATH_GPT_base_k_eq_26_l2254_225467

theorem base_k_eq_26 (k : ℕ) (h : 3 * k + 2 = 26) : k = 8 :=
by {
  -- The actual proof will go here.
  sorry
}

end NUMINAMATH_GPT_base_k_eq_26_l2254_225467


namespace NUMINAMATH_GPT_tangent_line_at_point_l2254_225445

theorem tangent_line_at_point (x y : ℝ) (h : y = x^2) (hx : x = 1) (hy : y = 1) : 
  2 * x - y - 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_point_l2254_225445


namespace NUMINAMATH_GPT_sum_of_root_and_square_of_other_root_eq_2007_l2254_225479

/-- If α and β are the two real roots of the equation x^2 - x - 2006 = 0,
    then the value of α + β^2 is 2007. --/
theorem sum_of_root_and_square_of_other_root_eq_2007
  (α β : ℝ)
  (hα : α^2 - α - 2006 = 0)
  (hβ : β^2 - β - 2006 = 0) :
  α + β^2 = 2007 := sorry

end NUMINAMATH_GPT_sum_of_root_and_square_of_other_root_eq_2007_l2254_225479


namespace NUMINAMATH_GPT_B_inter_A_complement_eq_one_l2254_225493

def I : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 3, 5}
def B : Set ℕ := {1, 3}
def A_complement : Set ℕ := I \ A

theorem B_inter_A_complement_eq_one : B ∩ A_complement = {1} := by
  sorry

end NUMINAMATH_GPT_B_inter_A_complement_eq_one_l2254_225493


namespace NUMINAMATH_GPT_fabian_initial_hours_l2254_225414

-- Define the conditions
def speed : ℕ := 5
def total_distance : ℕ := 30
def additional_time : ℕ := 3

-- The distance Fabian covers in the additional time
def additional_distance := speed * additional_time

-- The initial distance walked by Fabian
def initial_distance := total_distance - additional_distance

-- The initial hours Fabian walked
def initial_hours := initial_distance / speed

theorem fabian_initial_hours : initial_hours = 3 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_fabian_initial_hours_l2254_225414


namespace NUMINAMATH_GPT_harold_monthly_income_l2254_225468

variable (M : ℕ)

def rent : ℕ := 700
def car_payment : ℕ := 300
def utilities : ℕ := car_payment / 2
def groceries : ℕ := 50

def total_expenses : ℕ := rent + car_payment + utilities + groceries
def remaining_money_after_expenses : ℕ := M - total_expenses
def retirement_saving_target : ℕ := 650
def required_remaining_money_pre_saving : ℕ := 2 * retirement_saving_target

theorem harold_monthly_income :
  remaining_money_after_expenses = required_remaining_money_pre_saving → M = 2500 :=
by
  sorry

end NUMINAMATH_GPT_harold_monthly_income_l2254_225468


namespace NUMINAMATH_GPT_trader_sold_40_meters_of_cloth_l2254_225438

theorem trader_sold_40_meters_of_cloth 
  (total_profit_per_meter : ℕ) 
  (total_profit : ℕ) 
  (meters_sold : ℕ) 
  (h1 : total_profit_per_meter = 30) 
  (h2 : total_profit = 1200) 
  (h3 : total_profit = total_profit_per_meter * meters_sold) : 
  meters_sold = 40 := by
  sorry

end NUMINAMATH_GPT_trader_sold_40_meters_of_cloth_l2254_225438


namespace NUMINAMATH_GPT_smallest_positive_period_sin_cos_sin_l2254_225482

noncomputable def smallest_positive_period := 2 * Real.pi

theorem smallest_positive_period_sin_cos_sin :
  ∃ T > 0, (∀ x, (Real.sin x - 2 * Real.cos (2 * x) + 4 * Real.sin (4 * x)) = (Real.sin (x + T) - 2 * Real.cos (2 * (x + T)) + 4 * Real.sin (4 * (x + T)))) ∧ T = smallest_positive_period := by
sorry

end NUMINAMATH_GPT_smallest_positive_period_sin_cos_sin_l2254_225482


namespace NUMINAMATH_GPT_find_correct_grades_l2254_225458

structure StudentGrades := 
  (Volodya: ℕ) 
  (Sasha: ℕ) 
  (Petya: ℕ)

def isCorrectGrades (grades : StudentGrades) : Prop :=
  grades.Volodya = 5 ∧ grades.Sasha = 4 ∧ grades.Petya = 3

theorem find_correct_grades (grades : StudentGrades)
  (h1 : grades.Volodya = 5 ∨ grades.Volodya ≠ 5)
  (h2 : grades.Sasha = 3 ∨ grades.Sasha ≠ 3)
  (h3 : grades.Petya ≠ 5 ∨ grades.Petya = 5)
  (unique_h1: grades.Volodya = 5 ∨ grades.Sasha = 5 ∨ grades.Petya = 5) 
  (unique_h2: grades.Volodya = 4 ∨ grades.Sasha = 4 ∨ grades.Petya = 4)
  (unique_h3: grades.Volodya = 3 ∨ grades.Sasha = 3 ∨ grades.Petya = 3) 
  (lyingCount: (grades.Volodya ≠ 5 ∧ grades.Sasha ≠ 3 ∧ grades.Petya = 5)
              ∨ (grades.Volodya = 5 ∧ grades.Sasha ≠ 3 ∧ grades.Petya ≠ 5)
              ∨ (grades.Volodya ≠ 5 ∧ grades.Sasha = 3 ∧ grades.Petya ≠ 5)) :
  isCorrectGrades grades :=
sorry

end NUMINAMATH_GPT_find_correct_grades_l2254_225458


namespace NUMINAMATH_GPT_sarah_copies_total_pages_l2254_225488

noncomputable def total_pages_copied (people : ℕ) (pages_first : ℕ) (copies_first : ℕ) (pages_second : ℕ) (copies_second : ℕ) : ℕ :=
  (pages_first * (copies_first * people)) + (pages_second * (copies_second * people))

theorem sarah_copies_total_pages :
  total_pages_copied 20 30 3 45 2 = 3600 := by
  sorry

end NUMINAMATH_GPT_sarah_copies_total_pages_l2254_225488


namespace NUMINAMATH_GPT_total_pay_per_week_l2254_225403

variable (X Y : ℝ)
variable (hx : X = 1.2 * Y)
variable (hy : Y = 240)

theorem total_pay_per_week : X + Y = 528 := by
  sorry

end NUMINAMATH_GPT_total_pay_per_week_l2254_225403


namespace NUMINAMATH_GPT_trevor_coin_difference_l2254_225455

theorem trevor_coin_difference:
  ∀ (total_coins quarters: ℕ),
  total_coins = 77 →
  quarters = 29 →
  (total_coins - quarters = 48) := by
  intros total_coins quarters h1 h2
  sorry

end NUMINAMATH_GPT_trevor_coin_difference_l2254_225455


namespace NUMINAMATH_GPT_arccos_cos_7_l2254_225435

noncomputable def arccos_cos_7_eq_7_minus_2pi : Prop :=
  ∃ x : ℝ, x = 7 - 2 * Real.pi ∧ Real.arccos (Real.cos 7) = x

theorem arccos_cos_7 :
  arccos_cos_7_eq_7_minus_2pi :=
by
  sorry

end NUMINAMATH_GPT_arccos_cos_7_l2254_225435


namespace NUMINAMATH_GPT_polynomial_average_k_l2254_225427

theorem polynomial_average_k (h : ∀ x : ℕ, x * (36 / x) = 36 → (x + (36 / x) = 37 ∨ x + (36 / x) = 20 ∨ x + (36 / x) = 15 ∨ x + (36 / x) = 13 ∨ x + (36 / x) = 12)) :
  (37 + 20 + 15 + 13 + 12) / 5 = 19.4 := by
sorry

end NUMINAMATH_GPT_polynomial_average_k_l2254_225427


namespace NUMINAMATH_GPT_min_value_of_n_l2254_225471

/-!
    Given:
    - There are 53 students.
    - Each student must join one club and can join at most two clubs.
    - There are three clubs: Science, Culture, and Lifestyle.

    Prove:
    The minimum value of n, where n is the maximum number of people who join exactly the same set of clubs, is 9.
-/

def numStudents : ℕ := 53
def numClubs : ℕ := 3
def numSets : ℕ := 6

theorem min_value_of_n : ∃ n : ℕ, n = 9 ∧ 
  ∀ (students clubs sets : ℕ), students = numStudents → clubs = numClubs → sets = numSets →
  (students / sets + if students % sets = 0 then 0 else 1) = 9 :=
by
  sorry -- proof to be filled out

end NUMINAMATH_GPT_min_value_of_n_l2254_225471


namespace NUMINAMATH_GPT_imaginary_part_zero_iff_a_eq_neg1_l2254_225418

theorem imaginary_part_zero_iff_a_eq_neg1 (a : ℝ) (h : (Complex.I * (a + Complex.I) + a - 1).im = 0) : 
  a = -1 :=
sorry

end NUMINAMATH_GPT_imaginary_part_zero_iff_a_eq_neg1_l2254_225418


namespace NUMINAMATH_GPT_find_uncertain_mushrooms_l2254_225448

-- Definitions for the conditions based on the problem statement.
variable (totalMushrooms : ℕ)
variable (safeMushrooms : ℕ)
variable (poisonousMushrooms : ℕ)
variable (uncertainMushrooms : ℕ)

-- The conditions given in the problem
-- 1. Lillian found 32 mushrooms.
-- 2. She identified 9 mushrooms as safe to eat.
-- 3. The number of poisonous mushrooms is twice the number of safe mushrooms.
-- 4. The total number of mushrooms is the sum of safe, poisonous, and uncertain mushrooms.

axiom given_conditions : 
  totalMushrooms = 32 ∧
  safeMushrooms = 9 ∧
  poisonousMushrooms = 2 * safeMushrooms ∧
  totalMushrooms = safeMushrooms + poisonousMushrooms + uncertainMushrooms

-- The proof problem: Given the conditions, prove the number of uncertain mushrooms equals 5
theorem find_uncertain_mushrooms : 
  uncertainMushrooms = 5 :=
by sorry

end NUMINAMATH_GPT_find_uncertain_mushrooms_l2254_225448


namespace NUMINAMATH_GPT_triangle_area_l2254_225407

theorem triangle_area (a b c : ℕ) (h₁ : a = 7) (h₂ : b = 24) (h₃ : c = 25) (h₄ : a^2 + b^2 = c^2) : 
  ∃ A : ℕ, A = 84 ∧ A = (a * b) / 2 := by
  sorry

end NUMINAMATH_GPT_triangle_area_l2254_225407


namespace NUMINAMATH_GPT_number_of_therapy_hours_l2254_225439

theorem number_of_therapy_hours (A F H : ℝ) (h1 : F = A + 35) 
  (h2 : F + (H - 1) * A = 350) (h3 : F + A = 161) :
  H = 5 :=
by
  sorry

end NUMINAMATH_GPT_number_of_therapy_hours_l2254_225439


namespace NUMINAMATH_GPT_coefficient_comparison_expansion_l2254_225492

theorem coefficient_comparison_expansion (n : ℕ) (h₁ : 2 * n * (n - 1) = 14 * n) : n = 8 :=
by
  sorry

end NUMINAMATH_GPT_coefficient_comparison_expansion_l2254_225492


namespace NUMINAMATH_GPT_tan_alpha_eq_neg_one_l2254_225456

theorem tan_alpha_eq_neg_one (α : ℝ) (h : Real.sin (π / 6 - α) = Real.cos (π / 6 + α)) : Real.tan α = -1 :=
  sorry

end NUMINAMATH_GPT_tan_alpha_eq_neg_one_l2254_225456


namespace NUMINAMATH_GPT_least_product_ab_l2254_225461

theorem least_product_ab (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : (1 : ℚ) / a + 1 / (3 * b) = 1 / 6) : a * b ≥ 48 :=
by
  sorry

end NUMINAMATH_GPT_least_product_ab_l2254_225461


namespace NUMINAMATH_GPT_average_salary_correct_l2254_225474

def A_salary := 10000
def B_salary := 5000
def C_salary := 11000
def D_salary := 7000
def E_salary := 9000

def total_salary := A_salary + B_salary + C_salary + D_salary + E_salary
def num_individuals := 5

def average_salary := total_salary / num_individuals

theorem average_salary_correct : average_salary = 8600 := by
  sorry

end NUMINAMATH_GPT_average_salary_correct_l2254_225474


namespace NUMINAMATH_GPT_fraction_of_cost_due_to_high_octane_is_half_l2254_225454

theorem fraction_of_cost_due_to_high_octane_is_half :
  ∀ (cost_regular cost_high : ℝ) (units_high units_regular : ℕ),
    units_high * cost_high + units_regular * cost_regular ≠ 0 →
    cost_high = 3 * cost_regular →
    units_high = 1515 →
    units_regular = 4545 →
    (units_high * cost_high) / (units_high * cost_high + units_regular * cost_regular) = 1 / 2 :=
by
  intro cost_regular cost_high units_high units_regular h_total_cost_ne_zero h_cost_rel h_units_high h_units_regular
  -- skip the actual proof steps
  sorry

end NUMINAMATH_GPT_fraction_of_cost_due_to_high_octane_is_half_l2254_225454


namespace NUMINAMATH_GPT_intersection_a_b_l2254_225436

-- Definitions of sets A and B
def A : Set ℝ := {x | -2 < x ∧ x ≤ 2}
def B : Set ℝ := {-2, -1, 0}

-- The proof problem
theorem intersection_a_b : A ∩ B = {-1, 0} :=
by
  sorry

end NUMINAMATH_GPT_intersection_a_b_l2254_225436


namespace NUMINAMATH_GPT_percentage_of_diameter_l2254_225453

variable (d_R d_S r_R r_S : ℝ)
variable (A_R A_S : ℝ)
variable (pi : ℝ) (h1 : pi > 0)

theorem percentage_of_diameter 
(h_area : A_R = 0.64 * A_S) 
(h_radius_R : r_R = d_R / 2) 
(h_radius_S : r_S = d_S / 2)
(h_area_R : A_R = pi * r_R^2) 
(h_area_S : A_S = pi * r_S^2) 
: (d_R / d_S) * 100 = 80 := by
  sorry

end NUMINAMATH_GPT_percentage_of_diameter_l2254_225453


namespace NUMINAMATH_GPT_find_value_l2254_225442

noncomputable def a : ℝ := 5 - 2 * Real.sqrt 6

theorem find_value :
  a^2 - 10 * a + 1 = 0 :=
by
  -- Since we are only required to write the statement, add sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_find_value_l2254_225442


namespace NUMINAMATH_GPT_unique_solution_condition_l2254_225415

theorem unique_solution_condition (a b c : ℝ) : 
  (∃! x : ℝ, 4 * x - 7 + a = c * x + b) ↔ c ≠ 4 :=
sorry

end NUMINAMATH_GPT_unique_solution_condition_l2254_225415


namespace NUMINAMATH_GPT_min_value_expr_l2254_225495

theorem min_value_expr (a b : ℝ) (h₁ : 0 < b) (h₂ : b < a) :
  ∃ x : ℝ, x = a^2 + 1 / (b * (a - b)) ∧ x ≥ 4 :=
by sorry

end NUMINAMATH_GPT_min_value_expr_l2254_225495


namespace NUMINAMATH_GPT_polynomial_remainder_l2254_225470

noncomputable def remainder_div (p : Polynomial ℚ) (d1 d2 d3 : Polynomial ℚ) : Polynomial ℚ :=
  let d := d1 * d2 * d3 
  let q := p /ₘ d 
  let r := p %ₘ d 
  r

theorem polynomial_remainder :
  let p := (X^6 + 2 * X^4 - X^3 - 7 * X^2 + 3 * X + 1)
  let d1 := X - 2
  let d2 := X + 1
  let d3 := X - 3
  remainder_div p d1 d2 d3 = 29 * X^2 + 17 * X - 19 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_remainder_l2254_225470


namespace NUMINAMATH_GPT_total_cost_nancy_spends_l2254_225410

def price_crystal_beads : ℝ := 12
def price_metal_beads : ℝ := 15
def sets_crystal_beads : ℕ := 3
def sets_metal_beads : ℕ := 4
def discount_crystal : ℝ := 0.10
def tax_metal : ℝ := 0.05

theorem total_cost_nancy_spends :
  sets_crystal_beads * price_crystal_beads * (1 - discount_crystal) + 
  sets_metal_beads * price_metal_beads * (1 + tax_metal) = 95.40 := 
  by sorry

end NUMINAMATH_GPT_total_cost_nancy_spends_l2254_225410


namespace NUMINAMATH_GPT_f_2017_of_9_eq_8_l2254_225428

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def f (n : ℕ) : ℕ :=
  sum_of_digits (n^2 + 1)

def f_k (k n : ℕ) : ℕ :=
  if k = 0 then n else f (f_k (k-1) n)

theorem f_2017_of_9_eq_8 : f_k 2017 9 = 8 := by
  sorry

end NUMINAMATH_GPT_f_2017_of_9_eq_8_l2254_225428


namespace NUMINAMATH_GPT_minimum_value_of_f_l2254_225440

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt ((x + 2)^2 + 4^2)) + (Real.sqrt ((x + 1)^2 + 3^2))

theorem minimum_value_of_f : ∃ x : ℝ, f x = 5 * Real.sqrt 2 ∧ ∀ y : ℝ, f y ≥ f x :=
by
  use -3
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l2254_225440


namespace NUMINAMATH_GPT_max_min_value_d_l2254_225416

-- Definitions of the given conditions
def circle_eqn (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Definition of the distance squared from a point to a fixed point
def dist_sq (P Q : ℝ × ℝ) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Definition of the function d
def d (P : ℝ × ℝ) : ℝ := dist_sq P A + dist_sq P B

-- The main theorem that we need to prove
theorem max_min_value_d :
  (∀ P : ℝ × ℝ, circle_eqn P.1 P.2 → d P ≤ 74) ∧
  (∃ P : ℝ × ℝ, circle_eqn P.1 P.2 ∧ d P = 74) ∧
  (∀ P : ℝ × ℝ, circle_eqn P.1 P.2 → 34 ≤ d P) ∧
  (∃ P : ℝ × ℝ, circle_eqn P.1 P.2 ∧ d P = 34) :=
sorry

end NUMINAMATH_GPT_max_min_value_d_l2254_225416


namespace NUMINAMATH_GPT_solve_inequality_l2254_225472

theorem solve_inequality (x : ℝ) (h : 3 * x + 4 ≠ 0) :
  (3 - 1 / (3 * x + 4) < 5) ↔ (-4 / 3 < x) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l2254_225472


namespace NUMINAMATH_GPT_simplify_expr_for_a_neq_0_1_neg1_final_value_when_a_2_l2254_225409

theorem simplify_expr_for_a_neq_0_1_neg1 (a : ℝ) (h1 : a ≠ 1) (h0 : a ≠ 0) (h_neg1 : a ≠ -1) :
  ( (a - 1)^2 / ((a + 1) * (a - 1)) ) / (a - (2 * a / (a + 1))) = 1 / a := by
  sorry

theorem final_value_when_a_2 :
  ( (2 - 1)^2 / ((2 + 1) * (2 - 1)) ) / (2 - (2 * 2 / (2 + 1))) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_simplify_expr_for_a_neq_0_1_neg1_final_value_when_a_2_l2254_225409


namespace NUMINAMATH_GPT_find_third_integer_l2254_225457

theorem find_third_integer (a b c : ℕ) (h1 : a * b * c = 42) (h2 : a + b = 9) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : c = 3 :=
sorry

end NUMINAMATH_GPT_find_third_integer_l2254_225457


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_product_l2254_225450

theorem arithmetic_geometric_sequence_product :
  ∀ (a : ℕ → ℝ) (q : ℝ),
    a 1 = 3 →
    (a 1) + (a 1 * q^2) + (a 1 * q^4) = 21 →
    (a 2) * (a 6) = 72 :=
by 
  intros a q h1 h2 
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_product_l2254_225450


namespace NUMINAMATH_GPT_smallest_positive_multiple_of_37_l2254_225444

theorem smallest_positive_multiple_of_37 :
  ∃ n, n > 0 ∧ (∃ a, n = 37 * a) ∧ (∃ k, n = 76 * k + 7) ∧ n = 2405 := 
by
  sorry

end NUMINAMATH_GPT_smallest_positive_multiple_of_37_l2254_225444


namespace NUMINAMATH_GPT_total_balls_is_108_l2254_225477

theorem total_balls_is_108 (B : ℕ) (W : ℕ) (n : ℕ) (h1 : W = 8 * B) 
                           (h2 : n = B + W) 
                           (h3 : 100 ≤ n - W + 1) 
                           (h4 : 100 > B) : n = 108 := 
by sorry

end NUMINAMATH_GPT_total_balls_is_108_l2254_225477


namespace NUMINAMATH_GPT_parabola_point_coord_l2254_225437

theorem parabola_point_coord {x y : ℝ} (h₁ : y^2 = 4 * x) (h₂ : (x - 1)^2 + y^2 = 100) : x = 9 ∧ (y = 6 ∨ y = -6) :=
by 
  sorry

end NUMINAMATH_GPT_parabola_point_coord_l2254_225437


namespace NUMINAMATH_GPT_fewest_seats_occupied_l2254_225484

def min_seats_occupied (N : ℕ) : ℕ :=
  if h : N % 4 = 0 then (N / 2) else (N / 2) + 1

theorem fewest_seats_occupied (N : ℕ) (h : N = 150) : min_seats_occupied N = 74 := by
  sorry

end NUMINAMATH_GPT_fewest_seats_occupied_l2254_225484


namespace NUMINAMATH_GPT_find_fifth_number_l2254_225473

def avg_sum_9_numbers := 936
def sum_first_5_numbers := 495
def sum_last_5_numbers := 500

theorem find_fifth_number (A1 A2 A3 A4 A5 A6 A7 A8 A9 : ℝ)
  (h1 : A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8 + A9 = avg_sum_9_numbers)
  (h2 : A1 + A2 + A3 + A4 + A5 = sum_first_5_numbers)
  (h3 : A5 + A6 + A7 + A8 + A9 = sum_last_5_numbers) :
  A5 = 29.5 :=
sorry

end NUMINAMATH_GPT_find_fifth_number_l2254_225473


namespace NUMINAMATH_GPT_value_of_fraction_l2254_225446

theorem value_of_fraction (m n : ℝ) (h1 : m^2 - 2 * m - 1 = 0) (h2 : n^2 + 2 * n - 1 = 0) (h3 : m * n ≠ 1) : 
  (mn + n + 1) / n = 3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_fraction_l2254_225446


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_empty_l2254_225404

theorem quadratic_inequality_solution_set_empty
  (m : ℝ)
  (h : ∀ x : ℝ, mx^2 - mx - 1 < 0) :
  -4 < m ∧ m < 0 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_empty_l2254_225404


namespace NUMINAMATH_GPT_enchilada_cost_l2254_225478

theorem enchilada_cost : ∃ T E : ℝ, 2 * T + 3 * E = 7.80 ∧ 3 * T + 5 * E = 12.70 ∧ E = 2.00 :=
by
  sorry

end NUMINAMATH_GPT_enchilada_cost_l2254_225478


namespace NUMINAMATH_GPT_simplify_and_evaluate_l2254_225494

theorem simplify_and_evaluate 
  (x y : ℤ) 
  (h1 : |x| = 2) 
  (h2 : y = 1) 
  (h3 : x * y < 0) : 
  3 * x^2 * y - 2 * x^2 - (x * y)^2 - 3 * x^2 * y - 4 * (x * y)^2 = -18 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l2254_225494


namespace NUMINAMATH_GPT_fraction_power_mult_equality_l2254_225413

-- Define the fraction and the power
def fraction := (1 : ℚ) / 3
def power : ℚ := fraction ^ 4

-- Define the multiplication
def result := 8 * power

-- Prove the equality
theorem fraction_power_mult_equality : result = 8 / 81 := by
  sorry

end NUMINAMATH_GPT_fraction_power_mult_equality_l2254_225413


namespace NUMINAMATH_GPT_jim_makes_60_dollars_l2254_225469

-- Definitions based on the problem conditions
def average_weight_per_rock : ℝ := 1.5
def price_per_pound : ℝ := 4
def number_of_rocks : ℕ := 10

-- Problem statement
theorem jim_makes_60_dollars :
  (average_weight_per_rock * number_of_rocks) * price_per_pound = 60 := by
  sorry

end NUMINAMATH_GPT_jim_makes_60_dollars_l2254_225469


namespace NUMINAMATH_GPT_algebraic_expression_l2254_225497

theorem algebraic_expression (a b : Real) 
  (h : a * b = 2 * (a^2 + b^2)) : 2 * a * b - (a^2 + b^2) = 0 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_l2254_225497


namespace NUMINAMATH_GPT_price_for_two_bracelets_l2254_225400

theorem price_for_two_bracelets
    (total_bracelets : ℕ)
    (price_per_bracelet : ℕ)
    (total_earned_for_single : ℕ)
    (total_earned : ℕ)
    (bracelets_sold_single : ℕ)
    (bracelets_left : ℕ)
    (remaining_earned : ℕ)
    (pairs_sold : ℕ)
    (price_per_pair : ℕ) :
    total_bracelets = 30 →
    price_per_bracelet = 5 →
    total_earned_for_single = 60 →
    total_earned = 132 →
    bracelets_sold_single = total_earned_for_single / price_per_bracelet →
    bracelets_left = total_bracelets - bracelets_sold_single →
    remaining_earned = total_earned - total_earned_for_single →
    pairs_sold = bracelets_left / 2 →
    price_per_pair = remaining_earned / pairs_sold →
    price_per_pair = 8 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end NUMINAMATH_GPT_price_for_two_bracelets_l2254_225400


namespace NUMINAMATH_GPT_proof_of_inequality_l2254_225466

theorem proof_of_inequality (a : ℝ) (h : (∃ x : ℝ, x - 2 * a + 4 = 0 ∧ x < 0)) :
  (a - 3) * (a - 4) > 0 :=
by
  sorry

end NUMINAMATH_GPT_proof_of_inequality_l2254_225466


namespace NUMINAMATH_GPT_find_prime_q_l2254_225449

theorem find_prime_q (p q r : ℕ) 
  (prime_p : Nat.Prime p)
  (prime_q : Nat.Prime q)
  (prime_r : Nat.Prime r)
  (eq_r : q - p = r)
  (cond_p : 5 < p ∧ p < 15)
  (cond_q : q < 15) :
  q = 13 :=
sorry

end NUMINAMATH_GPT_find_prime_q_l2254_225449


namespace NUMINAMATH_GPT_find_x_l2254_225441

theorem find_x (x : ℝ) (h : 0.65 * x = 0.20 * 552.50) : x = 170 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2254_225441


namespace NUMINAMATH_GPT_determine_m_in_hexadecimal_conversion_l2254_225462

theorem determine_m_in_hexadecimal_conversion :
  ∃ m : ℕ, 1 * 6^5 + 3 * 6^4 + m * 6^3 + 5 * 6^2 + 0 * 6^1 + 2 * 6^0 = 12710 ∧ m = 4 :=
by
  sorry

end NUMINAMATH_GPT_determine_m_in_hexadecimal_conversion_l2254_225462


namespace NUMINAMATH_GPT_part1_part2_l2254_225408

noncomputable def f (x a : ℝ) := (x - 1) * Real.exp x + a * x + 1
noncomputable def g (x : ℝ) := x * Real.exp x

-- Problem Part 1: Prove the range of a for which f(x) has two extreme points
theorem part1 (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ a = f x₂ a) ↔ (0 < a ∧ a < (1 / Real.exp 1)) :=
sorry

-- Problem Part 2: Prove the range of a for which f(x) ≥ 2sin(x) for x ≥ 0
theorem part2 (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x → f x a ≥ 2 * Real.sin x) ↔ (2 ≤ a) :=
sorry

end NUMINAMATH_GPT_part1_part2_l2254_225408


namespace NUMINAMATH_GPT_boxes_given_away_l2254_225426

def total_boxes := 12
def pieces_per_box := 6
def remaining_pieces := 30

theorem boxes_given_away : (total_boxes * pieces_per_box - remaining_pieces) / pieces_per_box = 7 :=
by
  sorry

end NUMINAMATH_GPT_boxes_given_away_l2254_225426


namespace NUMINAMATH_GPT_roots_triangle_ineq_l2254_225480

variable {m : ℝ}

def roots_form_triangle (x1 x2 x3 : ℝ) : Prop :=
  x1 + x2 > x3 ∧ x1 + x3 > x2 ∧ x2 + x3 > x1

theorem roots_triangle_ineq (h : ∀ x, (x - 2) * (x^2 - 4*x + m) = 0) :
  3 < m ∧ m < 4 :=
by
  sorry

end NUMINAMATH_GPT_roots_triangle_ineq_l2254_225480


namespace NUMINAMATH_GPT_child_tickets_sold_l2254_225422

-- Define variables and types
variables (A C : ℕ)

-- Main theorem to prove
theorem child_tickets_sold : A + C = 80 ∧ 12 * A + 5 * C = 519 → C = 63 :=
by
  intros
  sorry

end NUMINAMATH_GPT_child_tickets_sold_l2254_225422


namespace NUMINAMATH_GPT_fourth_root_is_four_l2254_225499

-- Define the polynomial f
def f (x : ℝ) : ℝ := x^4 - 8 * x^3 - 7 * x^2 + 9 * x + 11

-- Conditions that must be true for the given problem
@[simp] def f_neg1_zero : f (-1) = 0 := by sorry
@[simp] def f_2_zero : f (2) = 0 := by sorry
@[simp] def f_neg3_zero : f (-3) = 0 := by sorry

-- The theorem stating the fourth root
theorem fourth_root_is_four (root4 : ℝ) (H : f root4 = 0) : root4 = 4 := by sorry

end NUMINAMATH_GPT_fourth_root_is_four_l2254_225499


namespace NUMINAMATH_GPT_price_decrease_l2254_225425

theorem price_decrease (P : ℝ) (h₁ : 1.25 * P = P * 1.25) (h₂ : 1.10 * P = P * 1.10) :
  1.25 * P * (1 - 12 / 100) = 1.10 * P :=
by
  sorry

end NUMINAMATH_GPT_price_decrease_l2254_225425


namespace NUMINAMATH_GPT_present_age_of_A_is_11_l2254_225464

-- Definitions for present ages
variables (A B C : ℕ)

-- Definitions for the given conditions
def sum_of_ages_present : Prop := A + B + C = 57
def age_ratio_three_years_ago (x : ℕ) : Prop := (A - 3 = x) ∧ (B - 3 = 2 * x) ∧ (C - 3 = 3 * x)

-- The proof statement
theorem present_age_of_A_is_11 (x : ℕ) (h1 : sum_of_ages_present A B C) (h2 : age_ratio_three_years_ago A B C x) : A = 11 := 
by
  sorry

end NUMINAMATH_GPT_present_age_of_A_is_11_l2254_225464


namespace NUMINAMATH_GPT_family_gathering_l2254_225489

theorem family_gathering (P : ℕ) 
  (h1 : (P / 2 = P - 10)) : P = 20 :=
sorry

end NUMINAMATH_GPT_family_gathering_l2254_225489


namespace NUMINAMATH_GPT_sector_perimeter_l2254_225498

-- Conditions:
def theta : ℝ := 54  -- central angle in degrees
def r : ℝ := 20      -- radius in cm

-- Translation of given conditions and expected result:
theorem sector_perimeter (theta_eq : theta = 54) (r_eq : r = 20) :
  let l := (θ * r) / 180 * Real.pi 
  let perim := l + 2 * r 
  perim = 6 * Real.pi + 40 := sorry

end NUMINAMATH_GPT_sector_perimeter_l2254_225498


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l2254_225402

theorem negation_of_universal_proposition (x : ℝ) :
  (¬ (∀ x : ℝ, |x| < 0)) ↔ (∃ x_0 : ℝ, |x_0| ≥ 0) := by
  sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l2254_225402


namespace NUMINAMATH_GPT_total_lives_correct_l2254_225433

namespace VideoGame

def num_friends : ℕ := 8
def lives_each : ℕ := 8

def total_lives (n : ℕ) (l : ℕ) : ℕ := n * l 

theorem total_lives_correct : total_lives num_friends lives_each = 64 := by
  sorry

end NUMINAMATH_GPT_total_lives_correct_l2254_225433


namespace NUMINAMATH_GPT_problem1_problem2_l2254_225465

-- Problem (Ⅰ)
theorem problem1 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  (1 + 1 / a) * (1 + 1 / b) ≥ 9 :=
sorry

-- Problem (Ⅱ)
theorem problem2 (a : ℝ) (h1 : ∀ (x : ℝ), x ≥ 1 ↔ |x + 3| - |x - a| ≥ 2) :
  a = 2 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l2254_225465


namespace NUMINAMATH_GPT_initial_number_of_bedbugs_l2254_225420

theorem initial_number_of_bedbugs (N : ℕ) 
  (h1 : ∃ N : ℕ, True)
  (h2 : ∀ (n : ℕ), (triples_daily : ℕ → ℕ) → triples_daily n = 3 * n)
  (h3 : ∀ (n : ℕ), (N * 3^4 = n) → n = 810) : 
  N = 10 :=
sorry

end NUMINAMATH_GPT_initial_number_of_bedbugs_l2254_225420


namespace NUMINAMATH_GPT_longest_side_of_enclosure_l2254_225431

theorem longest_side_of_enclosure (l w : ℝ)
  (h_perimeter : 2 * l + 2 * w = 240)
  (h_area : l * w = 8 * 240) :
  max l w = 80 :=
by
  sorry

end NUMINAMATH_GPT_longest_side_of_enclosure_l2254_225431


namespace NUMINAMATH_GPT_circle_equation_l2254_225485

theorem circle_equation (x y : ℝ) : 
  (∀ (a b : ℝ), (a - 1)^2 + (b - 1)^2 = 2 → (a, b) = (0, 0)) ∧
  ((0 - 1)^2 + (0 - 1)^2 = 2) → 
  (x - 1)^2 + (y - 1)^2 = 2 := 
by 
  sorry

end NUMINAMATH_GPT_circle_equation_l2254_225485


namespace NUMINAMATH_GPT_initial_fee_l2254_225406

theorem initial_fee (total_bowls : ℤ) (lost_bowls : ℤ) (broken_bowls : ℤ) (safe_fee : ℤ)
  (loss_fee : ℤ) (total_payment : ℤ) (paid_amount : ℤ) :
  total_bowls = 638 →
  lost_bowls = 12 →
  broken_bowls = 15 →
  safe_fee = 3 →
  loss_fee = 4 →
  total_payment = 1825 →
  paid_amount = total_payment - ((total_bowls - lost_bowls - broken_bowls) * safe_fee - (lost_bowls + broken_bowls) * loss_fee) →
  paid_amount = 100 :=
by
  intros _ _ _ _ _ _ _
  sorry

end NUMINAMATH_GPT_initial_fee_l2254_225406


namespace NUMINAMATH_GPT_algebraic_expression_value_l2254_225423

-- Define the given condition as a predicate
def condition (a : ℝ) := a^2 + a - 4 = 0

-- Then the goal to prove with the given condition
theorem algebraic_expression_value (a : ℝ) (h : condition a) : (a^2 - 3) * (a + 2) = -2 :=
sorry

end NUMINAMATH_GPT_algebraic_expression_value_l2254_225423


namespace NUMINAMATH_GPT_beau_age_calculation_l2254_225443

variable (sons_age : ℕ) (beau_age_today : ℕ) (beau_age_3_years_ago : ℕ)

def triplets := 3
def sons_today := 16
def sons_age_3_years_ago := sons_today - 3
def sum_of_sons_3_years_ago := triplets * sons_age_3_years_ago

theorem beau_age_calculation
  (h1 : sons_today = 16)
  (h2 : sum_of_sons_3_years_ago = beau_age_3_years_ago)
  (h3 : beau_age_today = beau_age_3_years_ago + 3) :
  beau_age_today = 42 :=
sorry

end NUMINAMATH_GPT_beau_age_calculation_l2254_225443


namespace NUMINAMATH_GPT_trapezium_second_side_length_l2254_225460

-- Define the problem in Lean
variables (a h A b : ℝ)

-- Define the conditions
def conditions : Prop :=
  a = 20 ∧ h = 25 ∧ A = 475

-- Prove the length of the second parallel side
theorem trapezium_second_side_length (h_cond : conditions a h A) : b = 18 :=
by
  sorry

end NUMINAMATH_GPT_trapezium_second_side_length_l2254_225460


namespace NUMINAMATH_GPT_simplify_scientific_notation_l2254_225451

theorem simplify_scientific_notation :
  (12 * 10^10) / (6 * 10^2) = 2 * 10^8 := 
sorry

end NUMINAMATH_GPT_simplify_scientific_notation_l2254_225451


namespace NUMINAMATH_GPT_mixture_ratios_equal_quantities_l2254_225463

-- Define the given conditions
def ratio_p_milk_water := (5, 4)
def ratio_q_milk_water := (2, 7)

-- Define what we're trying to prove: the ratio p : q such that the resulting mixture has equal milk and water
theorem mixture_ratios_equal_quantities 
  (P Q : ℝ) 
  (h1 : 5 * P + 2 * Q = 4 * P + 7 * Q) :
  P / Q = 5 :=
  sorry

end NUMINAMATH_GPT_mixture_ratios_equal_quantities_l2254_225463


namespace NUMINAMATH_GPT_remainders_of_65_powers_l2254_225421

theorem remainders_of_65_powers (n : ℕ) :
  (65 ^ (6 * n)) % 9 = 1 ∧
  (65 ^ (6 * n + 1)) % 9 = 2 ∧
  (65 ^ (6 * n + 2)) % 9 = 4 ∧
  (65 ^ (6 * n + 3)) % 9 = 8 :=
by
  sorry

end NUMINAMATH_GPT_remainders_of_65_powers_l2254_225421


namespace NUMINAMATH_GPT_sum_of_solutions_is_24_l2254_225487

theorem sum_of_solutions_is_24 (a : ℝ) (x1 x2 : ℝ) 
    (h1 : abs (x1 - a) = 100) (h2 : abs (x2 - a) = 100)
    (sum_eq : x1 + x2 = 24) : a = 12 :=
sorry

end NUMINAMATH_GPT_sum_of_solutions_is_24_l2254_225487


namespace NUMINAMATH_GPT_min_value_of_expression_l2254_225491

theorem min_value_of_expression (n : ℕ) (h : n > 0) : (n / 3 + 27 / n) ≥ 6 :=
by {
  -- Proof goes here but is not required in the statement
  sorry
}

end NUMINAMATH_GPT_min_value_of_expression_l2254_225491


namespace NUMINAMATH_GPT_quadratic_expression_positive_l2254_225475

theorem quadratic_expression_positive (k : ℝ) : 
  (∀ x : ℝ, x^2 - (k - 5) * x + k + 2 > 0) ↔ (7 - 4 * Real.sqrt 2 < k ∧ k < 7 + 4 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_expression_positive_l2254_225475


namespace NUMINAMATH_GPT_Olivia_money_left_l2254_225434

theorem Olivia_money_left (initial_amount spend_amount : ℕ) (h1 : initial_amount = 128) 
  (h2 : spend_amount = 38) : initial_amount - spend_amount = 90 := by
  sorry

end NUMINAMATH_GPT_Olivia_money_left_l2254_225434


namespace NUMINAMATH_GPT_regular_polygon_inscribed_circle_area_l2254_225459

theorem regular_polygon_inscribed_circle_area
  (n : ℕ) (R : ℝ) (hR : R ≠ 0) (h_area : (1 / 2 : ℝ) * n * R^2 * Real.sin (2 * Real.pi / n) = 4 * R^2) :
  n = 20 :=
by 
  sorry

end NUMINAMATH_GPT_regular_polygon_inscribed_circle_area_l2254_225459


namespace NUMINAMATH_GPT_roots_negative_and_bounds_find_possible_values_of_b_and_c_l2254_225424

theorem roots_negative_and_bounds
  (b c x₁ x₂ x₁' x₂' : ℤ) 
  (h1 : x₁ * x₂ > 0) 
  (h2 : x₁' * x₂' > 0)
  (h3 : x₁^2 + b * x₁ + c = 0) 
  (h4 : x₂^2 + b * x₂ + c = 0) 
  (h5 : x₁'^2 + c * x₁' + b = 0) 
  (h6 : x₂'^2 + c * x₂' + b = 0) :
  x₁ < 0 ∧ x₂ < 0 ∧ x₁' < 0 ∧ x₂' < 0 ∧ (b - 1 ≤ c ∧ c ≤ b + 1) :=
by
  sorry


theorem find_possible_values_of_b_and_c 
  (b c : ℤ) 
  (h's : ∃ x₁ x₂ x₁' x₂', 
    x₁ * x₂ > 0 ∧ 
    x₁' * x₂' > 0 ∧ 
    (x₁^2 + b * x₁ + c = 0) ∧ 
    (x₂^2 + b * x₂ + c = 0) ∧ 
    (x₁'^2 + c * x₁' + b = 0) ∧ 
    (x₂'^2 + c * x₂' + b = 0)) :
  (b = 4 ∧ c = 4) ∨ 
  (b = 5 ∧ c = 6) ∨ 
  (b = 6 ∧ c = 5) :=
by
  sorry

end NUMINAMATH_GPT_roots_negative_and_bounds_find_possible_values_of_b_and_c_l2254_225424


namespace NUMINAMATH_GPT_evaluate_expression_l2254_225496

theorem evaluate_expression :
  3 ^ (1 ^ (2 ^ 8)) + ((3 ^ 1) ^ 2) ^ 4 = 6564 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2254_225496


namespace NUMINAMATH_GPT_equilateral_triangle_intersections_l2254_225486

-- Define the main theorem based on the conditions

theorem equilateral_triangle_intersections :
  let a_1 := (6 - 1) * (7 - 1) / 2
  let a_2 := (6 - 2) * (7 - 2) / 2
  let a_3 := (6 - 3) * (7 - 3) / 2
  let a_4 := (6 - 4) * (7 - 4) / 2
  let a_5 := (6 - 5) * (7 - 5) / 2
  a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 = 70 := by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_intersections_l2254_225486


namespace NUMINAMATH_GPT_allocation_methods_count_l2254_225405

/-- The number of ways to allocate 24 quotas to 3 venues such that:
1. Each venue gets at least one quota.
2. Each venue gets a different number of quotas.
is equal to 222. -/
theorem allocation_methods_count : 
  ∃ n : ℕ, n = 222 ∧ 
  ∃ (a b c: ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  a + b + c = 24 ∧ 
  1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c := 
sorry

end NUMINAMATH_GPT_allocation_methods_count_l2254_225405


namespace NUMINAMATH_GPT_minimum_abs_sum_l2254_225476

def matrix_squared_condition (p q r s : ℤ) : Prop :=
  (p * p + q * r = 9) ∧ 
  (q * r + s * s = 9) ∧ 
  (p * q + q * s = 0) ∧ 
  (r * p + r * s = 0)

def abs_sum (p q r s : ℤ) : ℤ :=
  |p| + |q| + |r| + |s|

theorem minimum_abs_sum (p q r s : ℤ) (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0) (h4 : s ≠ 0) 
  (h5 : matrix_squared_condition p q r s) : abs_sum p q r s = 8 :=
by 
  sorry

end NUMINAMATH_GPT_minimum_abs_sum_l2254_225476


namespace NUMINAMATH_GPT_max_value_of_f_on_interval_l2254_225429

noncomputable def f (x : ℝ) : ℝ := (Real.sin (4 * x)) / (2 * Real.sin ((Real.pi / 2) - 2 * x))

theorem max_value_of_f_on_interval :
  ∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 6), f x = (Real.sqrt 3) / 2 := sorry

end NUMINAMATH_GPT_max_value_of_f_on_interval_l2254_225429


namespace NUMINAMATH_GPT_period1_period2_multiple_l2254_225432

theorem period1_period2_multiple
  (students_period1 : ℕ)
  (students_period2 : ℕ)
  (h_students_period1 : students_period1 = 11)
  (h_students_period2 : students_period2 = 8)
  (M : ℕ)
  (h_condition : students_period1 = M * students_period2 - 5) :
  M = 2 :=
by
  sorry

end NUMINAMATH_GPT_period1_period2_multiple_l2254_225432


namespace NUMINAMATH_GPT_no_integer_solutions_x2_plus_3xy_minus_2y2_eq_122_l2254_225447

theorem no_integer_solutions_x2_plus_3xy_minus_2y2_eq_122 :
  ¬ ∃ x y : ℤ, x^2 + 3 * x * y - 2 * y^2 = 122 := sorry

end NUMINAMATH_GPT_no_integer_solutions_x2_plus_3xy_minus_2y2_eq_122_l2254_225447


namespace NUMINAMATH_GPT_meaningful_fraction_l2254_225419

theorem meaningful_fraction {x : ℝ} : (x - 2) ≠ 0 ↔ x ≠ 2 :=
by
  sorry

end NUMINAMATH_GPT_meaningful_fraction_l2254_225419


namespace NUMINAMATH_GPT_total_goals_l2254_225490

def first_period_goals (k: ℕ) : ℕ :=
  k

def second_period_goals (k: ℕ) : ℕ :=
  2 * k

def spiders_first_period_goals (k: ℕ) : ℕ :=
  k / 2

def spiders_second_period_goals (s1: ℕ) : ℕ :=
  s1 * s1

def third_period_goals (k1 k2: ℕ) : ℕ :=
  2 * (k1 + k2)

def spiders_third_period_goals (s2: ℕ) : ℕ :=
  s2

def apply_bonus (goals: ℕ) (multiple: ℕ) : ℕ :=
  if goals % multiple = 0 then goals + 1 else goals

theorem total_goals (k1 k2 s1 s2 k3 s3 : ℕ) :
  first_period_goals 2 = k1 →
  second_period_goals k1 = k2 →
  spiders_first_period_goals k1 = s1 →
  spiders_second_period_goals s1 = s2 →
  third_period_goals k1 k2 = k3 →
  apply_bonus k3 3 = k3 + 1 →
  apply_bonus s2 2 = s2 →
  spiders_third_period_goals s2 = s3 →
  apply_bonus s3 2 = s3 →
  2 + k2 + (k3 + 1) + (s1 + s2 + s3) = 22 :=
by
  sorry

end NUMINAMATH_GPT_total_goals_l2254_225490


namespace NUMINAMATH_GPT_base7_to_base10_proof_l2254_225412

theorem base7_to_base10_proof (c d : ℕ) (h1 : 764 = 4 * 100 + c * 10 + d) : (c * d) / 20 = 6 / 5 :=
by
  sorry

end NUMINAMATH_GPT_base7_to_base10_proof_l2254_225412


namespace NUMINAMATH_GPT_crayons_given_l2254_225430

theorem crayons_given (initial lost left given : ℕ)
  (h1 : initial = 1453)
  (h2 : lost = 558)
  (h3 : left = 332)
  (h4 : given = initial - left - lost) :
  given = 563 :=
by
  rw [h1, h2, h3] at h4
  exact h4

end NUMINAMATH_GPT_crayons_given_l2254_225430


namespace NUMINAMATH_GPT_deliver_all_cargo_l2254_225483

theorem deliver_all_cargo (containers : ℕ) (cargo_mass : ℝ) (ships : ℕ) (ship_capacity : ℝ)
  (h1 : containers ≥ 35)
  (h2 : cargo_mass = 18)
  (h3 : ships = 7)
  (h4 : ship_capacity = 3)
  (h5 : ∀ t, (0 < t) → (t ≤ containers) → (t = 35)) :
  (ships * ship_capacity) ≥ cargo_mass :=
by
  sorry

end NUMINAMATH_GPT_deliver_all_cargo_l2254_225483


namespace NUMINAMATH_GPT_hazel_lemonade_total_l2254_225481

theorem hazel_lemonade_total 
  (total_lemonade: ℕ)
  (sold_construction: ℕ := total_lemonade / 2) 
  (sold_kids: ℕ := 18) 
  (gave_friends: ℕ := sold_kids / 2) 
  (drank_herself: ℕ := 1) :
  total_lemonade = 56 :=
  sorry

end NUMINAMATH_GPT_hazel_lemonade_total_l2254_225481


namespace NUMINAMATH_GPT_complex_quadratic_solution_l2254_225417

theorem complex_quadratic_solution (c d : ℤ) (h1 : 0 < c) (h2 : 0 < d) (h3 : (c + d * Complex.I) ^ 2 = 7 + 24 * Complex.I) :
  c + d * Complex.I = 4 + 3 * Complex.I :=
sorry

end NUMINAMATH_GPT_complex_quadratic_solution_l2254_225417
