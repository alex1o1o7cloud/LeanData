import Mathlib

namespace NUMINAMATH_GPT_tan_diff_pi_over_4_l956_95607

theorem tan_diff_pi_over_4 (α : ℝ) (hα1 : π < α) (hα2 : α < 3 / 2 * π) (hcos : Real.cos α = -4 / 5) :
  Real.tan (π / 4 - α) = 1 / 7 := by
  sorry

end NUMINAMATH_GPT_tan_diff_pi_over_4_l956_95607


namespace NUMINAMATH_GPT_gcd_1729_1337_l956_95642

theorem gcd_1729_1337 : Nat.gcd 1729 1337 = 7 := 
by
  sorry

end NUMINAMATH_GPT_gcd_1729_1337_l956_95642


namespace NUMINAMATH_GPT_min_arithmetic_series_sum_l956_95635

-- Definitions from the conditions
def arithmetic_sequence (a1 : ℤ) (d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d
def arithmetic_series_sum (a1 : ℤ) (d : ℤ) (n : ℕ) : ℤ := n * (a1 + (n-1) * d / 2)

-- Theorem statement
theorem min_arithmetic_series_sum (a2 a7 : ℤ) (h1 : a2 = -7) (h2 : a7 = 3) :
  ∃ n, (n * (a2 + (n - 1) * 2 / 2) = (n * n) - 10 * n) ∧
  (∀ m, n* (a2 + (m - 1) * 2 / 2) ≥ n * (n * n - 10 * n)) :=
sorry

end NUMINAMATH_GPT_min_arithmetic_series_sum_l956_95635


namespace NUMINAMATH_GPT_sticks_form_triangle_l956_95605

theorem sticks_form_triangle (a b c d e : ℝ) 
  (h1 : 2 < a) (h2 : a < 8)
  (h3 : 2 < b) (h4 : b < 8)
  (h5 : 2 < c) (h6 : c < 8)
  (h7 : 2 < d) (h8 : d < 8)
  (h9 : 2 < e) (h10 : e < 8) :
  ∃ x y z, 
    (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧
    (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) ∧
    (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) ∧
    x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
    x + y > z ∧ x + z > y ∧ y + z > x :=
by sorry

end NUMINAMATH_GPT_sticks_form_triangle_l956_95605


namespace NUMINAMATH_GPT_sin_alpha_l956_95638

noncomputable def f (x : ℝ) : ℝ := Real.cos (x / 2 - Real.pi / 4)

theorem sin_alpha (α : ℝ) (h : f α = 1 / 3) : Real.sin α = -7 / 9 :=
by 
  sorry

end NUMINAMATH_GPT_sin_alpha_l956_95638


namespace NUMINAMATH_GPT_batsman_percentage_running_between_wickets_l956_95633

def boundaries : Nat := 6
def runs_per_boundary : Nat := 4
def sixes : Nat := 4
def runs_per_six : Nat := 6
def no_balls : Nat := 8
def runs_per_no_ball : Nat := 1
def wide_balls : Nat := 5
def runs_per_wide_ball : Nat := 1
def leg_byes : Nat := 2
def runs_per_leg_bye : Nat := 1
def total_score : Nat := 150

def runs_from_boundaries : Nat := boundaries * runs_per_boundary
def runs_from_sixes : Nat := sixes * runs_per_six
def runs_not_off_bat : Nat := no_balls * runs_per_no_ball + wide_balls * runs_per_wide_ball + leg_byes * runs_per_leg_bye

def runs_running_between_wickets : Nat := total_score - runs_not_off_bat - runs_from_boundaries - runs_from_sixes

def percentage_runs_running_between_wickets : Float := 
  (runs_running_between_wickets.toFloat / total_score.toFloat) * 100

theorem batsman_percentage_running_between_wickets : percentage_runs_running_between_wickets = 58 := sorry

end NUMINAMATH_GPT_batsman_percentage_running_between_wickets_l956_95633


namespace NUMINAMATH_GPT_find_q_zero_l956_95617

-- Assuming the polynomials p, q, and r are defined, and their relevant conditions are satisfied.

def constant_term (f : ℕ → ℝ) : ℝ := f 0

theorem find_q_zero (p q r : ℕ → ℝ)
  (h : p * q = r)
  (h_p_const : constant_term p = 5)
  (h_r_const : constant_term r = -10) :
  q 0 = -2 :=
sorry

end NUMINAMATH_GPT_find_q_zero_l956_95617


namespace NUMINAMATH_GPT_arithmetic_sequence_ratio_l956_95677

theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 d : ℝ)
  (h1 : ∀ n, a n = a1 + (n - 1) * d) (h2 : ∀ n, S n = n * (2 * a1 + (n - 1) * d) / 2)
  (h_nonzero: ∀ n, a n ≠ 0):
  (S 5) / (a 3) = 5 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_ratio_l956_95677


namespace NUMINAMATH_GPT_Mrs_Brown_points_l956_95666

-- Conditions given
variables (points_William points_Adams points_Daniel points_mean: ℝ) (num_classes: ℕ)

-- Define the conditions
def Mrs_William_points := points_William = 50
def Mr_Adams_points := points_Adams = 57
def Mrs_Daniel_points := points_Daniel = 57
def mean_condition := points_mean = 53.3
def num_classes_condition := num_classes = 4

-- Define the problem to prove
theorem Mrs_Brown_points :
  Mrs_William_points points_William ∧ Mr_Adams_points points_Adams ∧ Mrs_Daniel_points points_Daniel ∧ mean_condition points_mean ∧ num_classes_condition num_classes →
  ∃ (points_Brown: ℝ), points_Brown = 49 :=
by
  sorry

end NUMINAMATH_GPT_Mrs_Brown_points_l956_95666


namespace NUMINAMATH_GPT_express_114_as_ones_and_threes_with_min_ten_ones_l956_95682

theorem express_114_as_ones_and_threes_with_min_ten_ones :
  ∃n: ℕ, n = 35 ∧ ∃ x y : ℕ, x + 3 * y = 114 ∧ x ≥ 10 := sorry

end NUMINAMATH_GPT_express_114_as_ones_and_threes_with_min_ten_ones_l956_95682


namespace NUMINAMATH_GPT_total_revenue_correct_l956_95695

def items : Type := ℕ × ℝ

def magazines : items := (425, 2.50)
def newspapers : items := (275, 1.50)
def books : items := (150, 5.00)
def pamphlets : items := (75, 0.50)

def revenue (item : items) : ℝ := item.1 * item.2

def total_revenue : ℝ :=
  revenue magazines +
  revenue newspapers +
  revenue books +
  revenue pamphlets

theorem total_revenue_correct : total_revenue = 2262.50 := by
  sorry

end NUMINAMATH_GPT_total_revenue_correct_l956_95695


namespace NUMINAMATH_GPT_solution_set_fraction_inequality_l956_95673

theorem solution_set_fraction_inequality : 
  { x : ℝ | 0 < x ∧ x < 1/3 } = { x : ℝ | 1/x > 3 } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_fraction_inequality_l956_95673


namespace NUMINAMATH_GPT_terminal_side_in_third_quadrant_l956_95693

open Real

theorem terminal_side_in_third_quadrant (θ : ℝ) (h1 : sin θ < 0) (h2 : cos θ < 0) : 
    θ ∈ Set.Ioo (π : ℝ) (3 * π / 2) := 
sorry

end NUMINAMATH_GPT_terminal_side_in_third_quadrant_l956_95693


namespace NUMINAMATH_GPT_total_grandchildren_l956_95613

-- Define the conditions 
def daughters := 5
def sons := 4
def children_per_daughter := 8 + 7
def children_per_son := 6 + 3

-- State the proof problem
theorem total_grandchildren : daughters * children_per_daughter + sons * children_per_son = 111 :=
by
  sorry

end NUMINAMATH_GPT_total_grandchildren_l956_95613


namespace NUMINAMATH_GPT_binary_arithmetic_l956_95614

theorem binary_arithmetic :
  let a := 0b1101
  let b := 0b0110
  let c := 0b1011
  let d := 0b1001
  a + b - c + d = 0b10001 := by
sorry

end NUMINAMATH_GPT_binary_arithmetic_l956_95614


namespace NUMINAMATH_GPT_minimum_value_S15_minus_S10_l956_95630

theorem minimum_value_S15_minus_S10 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_geom_seq : ∀ n, S (n + 1) = S n * a (n + 1))
  (h_pos_terms : ∀ n, a n > 0)
  (h_arith_seq : S 10 - 2 * S 5 = 3)
  (h_geom_sub_seq : (S 10 - S 5) * (S 10 - S 5) = S 5 * (S 15 - S 10)) :
  ∃ m, m = 12 ∧ (S 15 - S 10) ≥ m := sorry

end NUMINAMATH_GPT_minimum_value_S15_minus_S10_l956_95630


namespace NUMINAMATH_GPT_javier_needs_10_dozen_l956_95636

def javier_goal : ℝ := 96
def cost_per_dozen : ℝ := 2.40
def selling_price_per_donut : ℝ := 1

theorem javier_needs_10_dozen : (javier_goal / ((selling_price_per_donut - (cost_per_dozen / 12)) * 12)) = 10 :=
by
  sorry

end NUMINAMATH_GPT_javier_needs_10_dozen_l956_95636


namespace NUMINAMATH_GPT_triangle_inscribed_angle_l956_95698

theorem triangle_inscribed_angle 
  (y : ℝ)
  (arc_PQ arc_QR arc_RP : ℝ)
  (h1 : arc_PQ = 2 * y + 40)
  (h2 : arc_QR = 3 * y + 15)
  (h3 : arc_RP = 4 * y - 40)
  (h4 : arc_PQ + arc_QR + arc_RP = 360) :
  ∃ angle_P : ℝ, angle_P = 64.995 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_inscribed_angle_l956_95698


namespace NUMINAMATH_GPT_points_on_same_line_l956_95625

theorem points_on_same_line (k : ℤ) : 
  (∃ m b : ℤ, ∀ p : ℤ × ℤ, p = (1, 4) ∨ p = (3, -2) ∨ p = (6, k / 3) → p.2 = m * p.1 + b) ↔ k = -33 :=
by
  sorry

end NUMINAMATH_GPT_points_on_same_line_l956_95625


namespace NUMINAMATH_GPT_least_positive_integer_with_12_factors_is_972_l956_95621

theorem least_positive_integer_with_12_factors_is_972 : ∃ k : ℕ, (∀ n : ℕ, (∃ (d : ℕ), d * k = n) ↔ n = 12) ∧ k = 972 := sorry

end NUMINAMATH_GPT_least_positive_integer_with_12_factors_is_972_l956_95621


namespace NUMINAMATH_GPT_system_of_equations_inconsistent_l956_95653

theorem system_of_equations_inconsistent :
  ¬∃ (x1 x2 x3 x4 x5 : ℝ), 
    (x1 + 2 * x2 - x3 + 3 * x4 - x5 = 0) ∧ 
    (2 * x1 - x2 + 3 * x3 + x4 - x5 = -1) ∧
    (x1 - x2 + x3 + 2 * x4 = 2) ∧
    (4 * x1 + 3 * x3 + 6 * x4 - 2 * x5 = 5) := 
sorry

end NUMINAMATH_GPT_system_of_equations_inconsistent_l956_95653


namespace NUMINAMATH_GPT_full_tank_cost_l956_95602

-- Definitions from the conditions
def total_liters_given := 36
def total_cost_given := 18
def tank_capacity := 64

-- Hypothesis based on the conditions
def price_per_liter := total_cost_given / total_liters_given

-- Conclusion we need to prove
theorem full_tank_cost: price_per_liter * tank_capacity = 32 :=
  sorry

end NUMINAMATH_GPT_full_tank_cost_l956_95602


namespace NUMINAMATH_GPT_ratio_of_m_l956_95628

theorem ratio_of_m (a b m m1 m2 : ℝ)
  (h1 : a * m^2 + b * m + c = 0)
  (h2 : (a / b + b / a) = 3 / 7)
  (h3 : a + b = (3 * m - 2) / m)
  (h4 : a * b = 7 / m)
  (h5 : (a + b)^2 = ab / (m * (7/ m)) - 2) :
  (m1 + m2 = 21) ∧ (m1 * m2 = 4) → 
  (m1/m2 + m2/m1 = 108.25) := sorry

end NUMINAMATH_GPT_ratio_of_m_l956_95628


namespace NUMINAMATH_GPT_product_of_fractions_is_3_div_80_l956_95645

def product_fractions (a b c d e f : ℚ) : ℚ := (a / b) * (c / d) * (e / f)

theorem product_of_fractions_is_3_div_80 
  (h₁ : product_fractions 3 8 2 5 1 4 = 3 / 80) : True :=
by
  sorry

end NUMINAMATH_GPT_product_of_fractions_is_3_div_80_l956_95645


namespace NUMINAMATH_GPT_min_value_xyz_l956_95662

open Real

theorem min_value_xyz
  (x y z : ℝ)
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0)
  (h4 : 5 * x + 16 * y + 33 * z ≥ 136) :
  x^3 + y^3 + z^3 + x^2 + y^2 + z^2 ≥ 50 :=
sorry

end NUMINAMATH_GPT_min_value_xyz_l956_95662


namespace NUMINAMATH_GPT_nonnegative_difference_of_roots_l956_95641

theorem nonnegative_difference_of_roots :
  ∀ (x : ℝ), x^2 + 40 * x + 300 = -50 → (∃ a b : ℝ, x^2 + 40 * x + 350 = 0 ∧ x = a ∧ x = b ∧ |a - b| = 25) := 
by 
sorry

end NUMINAMATH_GPT_nonnegative_difference_of_roots_l956_95641


namespace NUMINAMATH_GPT_coordinates_of_P_l956_95660

theorem coordinates_of_P (a : ℝ) (h : 2 * a - 6 = 0) : (2 * a - 6, a + 1) = (0, 4) :=
by 
  have ha : a = 3 := by linarith
  rw [ha]
  sorry

end NUMINAMATH_GPT_coordinates_of_P_l956_95660


namespace NUMINAMATH_GPT_laborer_savings_l956_95631

theorem laborer_savings
  (monthly_expenditure_first6 : ℕ := 70)
  (monthly_expenditure_next4 : ℕ := 60)
  (monthly_income : ℕ := 69)
  (expenditure_first6 := 6 * monthly_expenditure_first6)
  (income_first6 := 6 * monthly_income)
  (debt : ℕ := expenditure_first6 - income_first6)
  (expenditure_next4 := 4 * monthly_expenditure_next4)
  (income_next4 := 4 * monthly_income)
  (savings : ℕ := income_next4 - (expenditure_next4 + debt)) :
  savings = 30 := 
by
  sorry

end NUMINAMATH_GPT_laborer_savings_l956_95631


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l956_95667

variables {a b : ℤ}

theorem necessary_but_not_sufficient_condition : (¬(a = 1) ∨ ¬(b = 2)) ↔ ¬(a + b = 3) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l956_95667


namespace NUMINAMATH_GPT_tetrahedron_mistaken_sum_l956_95646

theorem tetrahedron_mistaken_sum :
  let edges := 6
  let vertices := 4
  let faces := 4
  let joe_count := vertices + 1  -- Joe counts one vertex twice
  edges + joe_count + faces = 15 := by
  sorry

end NUMINAMATH_GPT_tetrahedron_mistaken_sum_l956_95646


namespace NUMINAMATH_GPT_cars_each_remaining_day_l956_95644

theorem cars_each_remaining_day (total_cars : ℕ) (monday_cars : ℕ) (tuesday_cars : ℕ)
  (wednesday_cars : ℕ) (thursday_cars : ℕ) (remaining_days : ℕ)
  (h_total : total_cars = 450)
  (h_mon : monday_cars = 50)
  (h_tue : tuesday_cars = 50)
  (h_wed : wednesday_cars = 2 * monday_cars)
  (h_thu : thursday_cars = 2 * monday_cars)
  (h_remaining : remaining_days = (total_cars - (monday_cars + tuesday_cars + wednesday_cars + thursday_cars)) / 3)
  :
  remaining_days = 50 := sorry

end NUMINAMATH_GPT_cars_each_remaining_day_l956_95644


namespace NUMINAMATH_GPT_total_students_is_45_l956_95639

theorem total_students_is_45
  (students_burgers : ℕ) 
  (total_students : ℕ) 
  (hb : students_burgers = 30) 
  (ht : total_students = 45) : 
  total_students = 45 :=
by
  sorry

end NUMINAMATH_GPT_total_students_is_45_l956_95639


namespace NUMINAMATH_GPT_percent_increase_first_quarter_l956_95620

theorem percent_increase_first_quarter (P : ℝ) (X : ℝ) (h1 : P > 0) 
  (end_of_second_quarter : P * 1.8 = P*(1 + X / 100) * 1.44) : 
  X = 25 :=
by
  sorry

end NUMINAMATH_GPT_percent_increase_first_quarter_l956_95620


namespace NUMINAMATH_GPT_problem_l956_95600

theorem problem (a b c k : ℝ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) (hk : k ≠ 0)
  (h1 : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / (k * (b - c)^2) + b / (k * (c - a)^2) + c / (k * (a - b)^2) = 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_l956_95600


namespace NUMINAMATH_GPT_conquering_Loulan_necessary_for_returning_home_l956_95699

theorem conquering_Loulan_necessary_for_returning_home : 
  ∀ (P Q : Prop), (¬ Q → ¬ P) → (P → Q) :=
by sorry

end NUMINAMATH_GPT_conquering_Loulan_necessary_for_returning_home_l956_95699


namespace NUMINAMATH_GPT_trapezoid_segment_length_l956_95658

theorem trapezoid_segment_length (a b : ℝ) : 
  ∃ x : ℝ, x = Real.sqrt ((a^2 + b^2) / 2) :=
sorry

end NUMINAMATH_GPT_trapezoid_segment_length_l956_95658


namespace NUMINAMATH_GPT_simplify_expression_l956_95619

def expression (x y : ℤ) : ℤ := 
  ((2 * x + y) * (2 * x - y) - (2 * x - 3 * y)^2) / (-2 * y)

theorem simplify_expression {x y : ℤ} (hx : x = 1) (hy : y = -2) :
  expression x y = -16 :=
by 
  -- This proof will involve algebraic manipulation and substitution.
  sorry

end NUMINAMATH_GPT_simplify_expression_l956_95619


namespace NUMINAMATH_GPT_neg_p_iff_exists_ge_zero_l956_95668

variable (x : ℝ)

def p : Prop := ∀ x : ℝ, x^2 + x + 1 < 0

theorem neg_p_iff_exists_ge_zero : ¬ p ↔ ∃ x : ℝ, x^2 + x + 1 ≥ 0 :=
by 
   sorry

end NUMINAMATH_GPT_neg_p_iff_exists_ge_zero_l956_95668


namespace NUMINAMATH_GPT_nonagon_isosceles_triangle_count_l956_95611

theorem nonagon_isosceles_triangle_count (N : ℕ) (hN : N = 9) : 
  ∃(k : ℕ), k = 30 := 
by 
  have h := hN
  sorry      -- Solution steps would go here if we were proving it

end NUMINAMATH_GPT_nonagon_isosceles_triangle_count_l956_95611


namespace NUMINAMATH_GPT_equation1_solution_equation2_solution_l956_95655

theorem equation1_solution (x : ℝ) : (x - 1) ^ 3 = 64 ↔ x = 5 := sorry

theorem equation2_solution (x : ℝ) : 25 * x ^ 2 + 3 = 12 ↔ x = 3 / 5 ∨ x = -3 / 5 := sorry

end NUMINAMATH_GPT_equation1_solution_equation2_solution_l956_95655


namespace NUMINAMATH_GPT_f_shift_l956_95680

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

-- Define the main theorem
theorem f_shift (x h : ℝ) : f (x + h) - f x = h * (6 * x + 3 * h - 4) :=
by
  sorry

end NUMINAMATH_GPT_f_shift_l956_95680


namespace NUMINAMATH_GPT_closest_to_sin_2016_deg_is_neg_half_l956_95610

/-- Given the value of \( \sin 2016^\circ \), show that the closest number from the given options is \( -\frac{1}{2} \).
Options:
A: \( \frac{11}{2} \)
B: \( -\frac{1}{2} \)
C: \( \frac{\sqrt{2}}{2} \)
D: \( -1 \)
-/
theorem closest_to_sin_2016_deg_is_neg_half :
  let sin_2016 := Real.sin (2016 * Real.pi / 180)
  |sin_2016 - (-1 / 2)| < |sin_2016 - 11 / 2| ∧
  |sin_2016 - (-1 / 2)| < |sin_2016 - Real.sqrt 2 / 2| ∧
  |sin_2016 - (-1 / 2)| < |sin_2016 - (-1)| :=
by
  sorry

end NUMINAMATH_GPT_closest_to_sin_2016_deg_is_neg_half_l956_95610


namespace NUMINAMATH_GPT_max_k_exists_l956_95696

noncomputable def max_possible_k (x y k : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ k > 0)
  (h_eq : 5 = k^3 * ((x^2 / y^2) + (y^2 / x^2)) + k^2 * ((x / y) + (y / x))) : ℝ :=
sorry

theorem max_k_exists (x y k : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ k > 0)
  (h_eq : 5 = k^3 * ((x^2 / y^2) + (y^2 / x^2)) + k^2 * ((x / y) + (y / x))) :
  ∃ k_max : ℝ, k_max = max_possible_k x y k h_pos h_eq :=
sorry

end NUMINAMATH_GPT_max_k_exists_l956_95696


namespace NUMINAMATH_GPT_largest_constant_C_l956_95618

theorem largest_constant_C (C : ℝ) : C = 2 / Real.sqrt 3 ↔ ∀ (x y z : ℝ), x^2 + y^2 + 2 * z^2 + 1 ≥ C * (x + y + z) :=
by
  sorry

end NUMINAMATH_GPT_largest_constant_C_l956_95618


namespace NUMINAMATH_GPT_coordinates_of_C_are_correct_l956_95615

noncomputable section 

def Point := (ℝ × ℝ)

def A : Point := (1, 3)
def B : Point := (13, 9)

def vector_AB (A B : Point) : Point :=
  (B.1 - A.1, B.2 - A.2)

def scalar_mult (s : ℝ) (v : Point) : Point :=
  (s * v.1, s * v.2)

def add_vectors (v1 v2 : Point) : Point :=
  (v1.1 + v2.1, v1.2 + v2.2)

def C : Point :=
  let AB := vector_AB A B
  add_vectors B (scalar_mult (1 / 2) AB)

theorem coordinates_of_C_are_correct : C = (19, 12) := by sorry

end NUMINAMATH_GPT_coordinates_of_C_are_correct_l956_95615


namespace NUMINAMATH_GPT_fraction_blue_balls_l956_95643

theorem fraction_blue_balls (total_balls : ℕ) (red_fraction : ℚ) (other_balls : ℕ) (remaining_blue_fraction : ℚ) 
  (h1 : total_balls = 360) 
  (h2 : red_fraction = 1/4) 
  (h3 : other_balls = 216) 
  (h4 : remaining_blue_fraction = 1/5) :
  (total_balls - (total_balls / 4) - other_balls) = total_balls * (5 * red_fraction / 270) := 
by
  sorry

end NUMINAMATH_GPT_fraction_blue_balls_l956_95643


namespace NUMINAMATH_GPT_basketball_scores_l956_95637

theorem basketball_scores (n : ℕ) (h : n = 7) : 
  ∃ (k : ℕ), k = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_basketball_scores_l956_95637


namespace NUMINAMATH_GPT_r_amount_l956_95622

-- Let p, q, and r be the amounts of money p, q, and r have, respectively
variables (p q r : ℝ)

-- Given conditions: p + q + r = 5000 and r = (2 / 3) * (p + q)
theorem r_amount (h1 : p + q + r = 5000) (h2 : r = (2 / 3) * (p + q)) :
  r = 2000 :=
sorry

end NUMINAMATH_GPT_r_amount_l956_95622


namespace NUMINAMATH_GPT_arithmetic_seq_finite_negative_terms_l956_95692

theorem arithmetic_seq_finite_negative_terms (a d : ℝ) :
  (∃ N : ℕ, ∀ n : ℕ, n > N → a + n * d ≥ 0) ↔ (a < 0 ∧ d > 0) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_finite_negative_terms_l956_95692


namespace NUMINAMATH_GPT_squats_day_after_tomorrow_l956_95690

theorem squats_day_after_tomorrow (initial_squats : ℕ) (daily_increase : ℕ) (today : ℕ) (tomorrow : ℕ) (day_after_tomorrow : ℕ)
  (h1 : initial_squats = 30)
  (h2 : daily_increase = 5)
  (h3 : today = initial_squats + daily_increase)
  (h4 : tomorrow = today + daily_increase)
  (h5 : day_after_tomorrow = tomorrow + daily_increase) : 
  day_after_tomorrow = 45 := 
sorry

end NUMINAMATH_GPT_squats_day_after_tomorrow_l956_95690


namespace NUMINAMATH_GPT_distance_AB_l956_95676

-- Definitions and conditions taken from part a)
variables (a b c : ℝ) (h_ac_gt_b : a + c > b) (h_a_ge_0 : a ≥ 0) (h_b_ge_0 : b ≥ 0) (h_c_ge_0 : c ≥ 0)

-- The main theorem statement
theorem distance_AB (a b c : ℝ) (h_ac_gt_b : a + c > b) (h_a_ge_0 : a ≥ 0) (h_b_ge_0 : b ≥ 0) (h_c_ge_0 : c ≥ 0) : 
  ∃ s : ℝ, s = Real.sqrt ((a * b * c) / (a + c - b)) := 
sorry

end NUMINAMATH_GPT_distance_AB_l956_95676


namespace NUMINAMATH_GPT_percent_water_evaporated_l956_95685

theorem percent_water_evaporated (W : ℝ) (E : ℝ) (T : ℝ) (hW : W = 10) (hE : E = 0.16) (hT : T = 75) : 
  ((min (E * T) W) / W) * 100 = 100 :=
by
  sorry

end NUMINAMATH_GPT_percent_water_evaporated_l956_95685


namespace NUMINAMATH_GPT_not_cube_of_sum_l956_95609

theorem not_cube_of_sum (a b : ℕ) : ¬ ∃ (k : ℤ), a^3 + b^3 + 4 = k^3 :=
by
  sorry

end NUMINAMATH_GPT_not_cube_of_sum_l956_95609


namespace NUMINAMATH_GPT_pentagon_angle_T_l956_95687

theorem pentagon_angle_T (P Q R S T : ℝ) 
  (hPRT: P = R ∧ R = T)
  (hQS: Q + S = 180): 
  T = 120 :=
by
  sorry

end NUMINAMATH_GPT_pentagon_angle_T_l956_95687


namespace NUMINAMATH_GPT_complex_expression_proof_l956_95649

open Complex

theorem complex_expression_proof {x y z : ℂ}
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : x + y + z = 15)
  (h2 : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2 * x * y * z) :
  (x^3 + y^3 + z^3) / (x * y * z) = 18 :=
by
  sorry

end NUMINAMATH_GPT_complex_expression_proof_l956_95649


namespace NUMINAMATH_GPT_inequality_problem_l956_95647

theorem inequality_problem
  (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) :
  (b^2 / a + a^2 / b) ≥ (a + b) :=
sorry

end NUMINAMATH_GPT_inequality_problem_l956_95647


namespace NUMINAMATH_GPT_calculate_f_at_pi_div_6_l956_95672

noncomputable def f (x : ℝ) (ω φ : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem calculate_f_at_pi_div_6 (ω φ : ℝ) 
  (h : ∀ x : ℝ, f (π / 3 + x) ω φ = f (-x) ω φ) :
  f (π / 6) ω φ = 2 ∨ f (π / 6) ω φ = -2 :=
sorry

end NUMINAMATH_GPT_calculate_f_at_pi_div_6_l956_95672


namespace NUMINAMATH_GPT_Nikka_stamp_collection_l956_95648

theorem Nikka_stamp_collection (S : ℝ) 
  (h1 : 0.35 * S ≥ 0) 
  (h2 : 0.2 * S ≥ 0) 
  (h3 : 0 < S) 
  (h4 : 0.45 * S = 45) : S = 100 :=
sorry

end NUMINAMATH_GPT_Nikka_stamp_collection_l956_95648


namespace NUMINAMATH_GPT_soccer_score_combinations_l956_95674

theorem soccer_score_combinations :
  ∃ (x y z : ℕ), x + y + z = 14 ∧ 3 * x + y = 19 ∧ x + y + z ≥ 0 ∧ 
    ({ (3, 10, 1), (4, 7, 3), (5, 4, 5), (6, 1, 7) } = 
      { (x, y, z) | x + y + z = 14 ∧ 3 * x + y = 19 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 }) :=
by 
  sorry

end NUMINAMATH_GPT_soccer_score_combinations_l956_95674


namespace NUMINAMATH_GPT_line_through_point_trangle_area_line_with_given_slope_l956_95640

theorem line_through_point_trangle_area (k : ℝ) (b : ℝ) : 
  (∃ k, (∀ x y, y = k * (x + 3) + 4 ∧ (1 / 2) * (abs (3 * k + 4) * abs (-4 / k - 3)) = 3)) → 
  (∃ k₁ k₂, k₁ = -2/3 ∧ k₂ = -8/3 ∧ 
    (∀ x y, y = k₁ * (x + 3) + 4 → 2 * x + 3 * y - 6 = 0) ∧ 
    (∀ x y, y = k₂ * (x + 3) + 4 → 8 * x + 3 * y + 12 = 0)) := 
sorry

theorem line_with_given_slope (b : ℝ) : 
  (∀ x y, y = (1 / 6) * x + b) → (1 / 2) * abs (6 * b * b) = 3 → 
  (b = 1 ∨ b = -1) → (∀ x y, (b = 1 → x - 6 * y + 6 = 0 ∧ b = -1 → x - 6 * y - 6 = 0)) := 
sorry

end NUMINAMATH_GPT_line_through_point_trangle_area_line_with_given_slope_l956_95640


namespace NUMINAMATH_GPT_parallel_vectors_perpendicular_vectors_l956_95671

/-- Given vectors a and b where a = (1, 2) and b = (x, 1),
    let u = a + b and v = a - b.
    Prove that if u is parallel to v, then x = 1/2. 
    Also, prove that if u is perpendicular to v, then x = 2 or x = -2. --/

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (x, 1)
noncomputable def vector_u (x : ℝ) : ℝ × ℝ := (1 + x, 3)
noncomputable def vector_v (x : ℝ) : ℝ × ℝ := (1 - x, 1)

theorem parallel_vectors (x : ℝ) :
  (vector_u x).fst / (vector_v x).fst = (vector_u x).snd / (vector_v x).snd ↔ x = 1 / 2 :=
by
  sorry

theorem perpendicular_vectors (x : ℝ) :
  (vector_u x).fst * (vector_v x).fst + (vector_u x).snd * (vector_v x).snd = 0 ↔ x = 2 ∨ x = -2 :=
by
  sorry

end NUMINAMATH_GPT_parallel_vectors_perpendicular_vectors_l956_95671


namespace NUMINAMATH_GPT_constant_term_of_binomial_expansion_l956_95603

noncomputable def constant_in_binomial_expansion (a : ℝ) : ℝ := 
  if h : a = ∫ (x : ℝ) in (0)..(1), 2 * x 
  then ((1 : ℝ) - (a : ℝ)^(-1 : ℝ))^6
  else 0

theorem constant_term_of_binomial_expansion : 
  ∃ a : ℝ, (a = ∫ (x : ℝ) in (0)..(1), 2 * x) → constant_in_binomial_expansion a = (15 : ℝ) := sorry

end NUMINAMATH_GPT_constant_term_of_binomial_expansion_l956_95603


namespace NUMINAMATH_GPT_M_geq_N_l956_95681

variable (x y : ℝ)
def M : ℝ := x^2 + y^2 + 1
def N : ℝ := x + y + x * y

theorem M_geq_N (x y : ℝ) : M x y ≥ N x y :=
by
sorry

end NUMINAMATH_GPT_M_geq_N_l956_95681


namespace NUMINAMATH_GPT_water_level_drop_l956_95608

theorem water_level_drop :
  (∀ x : ℝ, x > 0 → (x = 4) → (x > 0 → x = 4)) →
  ∃ y : ℝ, y < 0 ∧ (y = -1) :=
by
  sorry

end NUMINAMATH_GPT_water_level_drop_l956_95608


namespace NUMINAMATH_GPT_sale_on_day_five_l956_95665

def sale1 : ℕ := 435
def sale2 : ℕ := 927
def sale3 : ℕ := 855
def sale6 : ℕ := 741
def average_sale : ℕ := 625
def total_days : ℕ := 5

theorem sale_on_day_five : 
  average_sale * total_days - (sale1 + sale2 + sale3 + sale6) = 167 :=
by
  sorry

end NUMINAMATH_GPT_sale_on_day_five_l956_95665


namespace NUMINAMATH_GPT_rita_canoe_trip_distance_l956_95612

theorem rita_canoe_trip_distance 
  (D : ℝ)
  (h_upstream : ∃ t1, t1 = D / 3)
  (h_downstream : ∃ t2, t2 = D / 9)
  (h_total_time : ∃ t1 t2, t1 + t2 = 8) :
  D = 18 :=
by
  sorry

end NUMINAMATH_GPT_rita_canoe_trip_distance_l956_95612


namespace NUMINAMATH_GPT_tangent_vertical_y_axis_iff_a_gt_0_l956_95626

theorem tangent_vertical_y_axis_iff_a_gt_0 {a : ℝ} (f : ℝ → ℝ) 
    (hf : ∀ x > 0, f x = a * x^2 - Real.log x)
    (h_tangent_vertical : ∃ x > 0, (deriv f x) = 0) :
    a > 0 := 
sorry

end NUMINAMATH_GPT_tangent_vertical_y_axis_iff_a_gt_0_l956_95626


namespace NUMINAMATH_GPT_complex_quadrant_l956_95616

theorem complex_quadrant (z : ℂ) (h : z = (↑(1/2) : ℂ) + (↑(1/2) : ℂ) * I ) : 
  0 < z.re ∧ 0 < z.im :=
by {
sorry -- Proof goes here
}

end NUMINAMATH_GPT_complex_quadrant_l956_95616


namespace NUMINAMATH_GPT_share_of_y_l956_95669

-- Define the conditions as hypotheses
variables (n : ℝ) (x y z : ℝ)

-- The main theorem we need to prove
theorem share_of_y (h1 : x = n) 
                   (h2 : y = 0.45 * n) 
                   (h3 : z = 0.50 * n) 
                   (h4 : x + y + z = 78) : 
  y = 18 :=
by 
  -- insert proof here (not required as per instructions)
  sorry

end NUMINAMATH_GPT_share_of_y_l956_95669


namespace NUMINAMATH_GPT_gcd_of_1237_and_1957_is_one_l956_95659

noncomputable def gcd_1237_1957 : Nat := Nat.gcd 1237 1957

theorem gcd_of_1237_and_1957_is_one : gcd_1237_1957 = 1 :=
by
  unfold gcd_1237_1957
  have : Nat.gcd 1237 1957 = 1 := sorry
  exact this

end NUMINAMATH_GPT_gcd_of_1237_and_1957_is_one_l956_95659


namespace NUMINAMATH_GPT_sales_worth_l956_95664

theorem sales_worth (S: ℝ) : 
  (1300 + 0.025 * (S - 4000) = 0.05 * S + 600) → S = 24000 :=
by
  sorry

end NUMINAMATH_GPT_sales_worth_l956_95664


namespace NUMINAMATH_GPT_original_price_of_trouser_l956_95652

theorem original_price_of_trouser (sale_price : ℝ) (discount : ℝ) (original_price : ℝ) 
  (h1 : sale_price = 30) (h2 : discount = 0.70) : 
  original_price = 100 :=
by
  sorry

end NUMINAMATH_GPT_original_price_of_trouser_l956_95652


namespace NUMINAMATH_GPT_equation_of_perpendicular_line_l956_95679

theorem equation_of_perpendicular_line :
  ∃ c : ℝ, (∀ x y : ℝ, (2 * x + y + c = 0 ↔ (x = 1 ∧ y = 1))) → (c = -3) := 
by
  sorry

end NUMINAMATH_GPT_equation_of_perpendicular_line_l956_95679


namespace NUMINAMATH_GPT_geometric_sequence_terms_l956_95634

theorem geometric_sequence_terms
  (a_3 : ℝ) (a_4 : ℝ)
  (h1 : a_3 = 12)
  (h2 : a_4 = 18) :
  ∃ (a_1 a_2 : ℝ) (q: ℝ), 
    a_1 = 16 / 3 ∧ a_2 = 8 ∧ a_3 = a_1 * q^2 ∧ a_4 = a_1 * q^3 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_terms_l956_95634


namespace NUMINAMATH_GPT_average_weight_of_all_children_l956_95629

theorem average_weight_of_all_children 
    (boys_weight_avg : ℕ)
    (number_of_boys : ℕ)
    (girls_weight_avg : ℕ)
    (number_of_girls : ℕ)
    (tall_boy_weight : ℕ)
    (ht1 : boys_weight_avg = 155)
    (ht2 : number_of_boys = 8)
    (ht3 : girls_weight_avg = 130)
    (ht4 : number_of_girls = 6)
    (ht5 : tall_boy_weight = 175)
    : (boys_weight_avg * (number_of_boys - 1) + tall_boy_weight + girls_weight_avg * number_of_girls) / (number_of_boys + number_of_girls) = 146 :=
by
  sorry

end NUMINAMATH_GPT_average_weight_of_all_children_l956_95629


namespace NUMINAMATH_GPT_profit_per_unit_and_minimum_units_l956_95663

noncomputable def conditions (x y m : ℝ) : Prop :=
  2 * x + 7 * y = 41 ∧
  x + 3 * y = 18 ∧
  0.5 * m + 0.3 * (30 - m) ≥ 13.1

theorem profit_per_unit_and_minimum_units (x y m : ℝ) :
  conditions x y m → x = 3 ∧ y = 5 ∧ m ≥ 21 :=
by
  sorry

end NUMINAMATH_GPT_profit_per_unit_and_minimum_units_l956_95663


namespace NUMINAMATH_GPT_product_of_tangents_l956_95627

theorem product_of_tangents : 
  (Real.tan (Real.pi / 8) * Real.tan (3 * Real.pi / 8) * 
   Real.tan (5 * Real.pi / 8) * Real.tan (7 * Real.pi / 8) = -2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_product_of_tangents_l956_95627


namespace NUMINAMATH_GPT_find_orange_juice_amount_l956_95688

variable (s y t oj : ℝ)

theorem find_orange_juice_amount (h1 : s = 0.2) (h2 : y = 0.1) (h3 : t = 0.5) (h4 : oj = t - (s + y)) : oj = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_find_orange_juice_amount_l956_95688


namespace NUMINAMATH_GPT_pairs_equality_l956_95624

-- Define all the pairs as given in the problem.
def pairA_1 : ℤ := - (2^7)
def pairA_2 : ℤ := (-2)^7
def pairB_1 : ℤ := - (3^2)
def pairB_2 : ℤ := (-3)^2
def pairC_1 : ℤ := -3 * (2^3)
def pairC_2 : ℤ := - (3^2) * 2
def pairD_1 : ℤ := -((-3)^2)
def pairD_2 : ℤ := -((-2)^3)

-- The problem statement.
theorem pairs_equality :
  pairA_1 = pairA_2 ∧ ¬ (pairB_1 = pairB_2) ∧ ¬ (pairC_1 = pairC_2) ∧ ¬ (pairD_1 = pairD_2) := by
  sorry

end NUMINAMATH_GPT_pairs_equality_l956_95624


namespace NUMINAMATH_GPT_find_number_l956_95656

theorem find_number (x : ℕ) (h : (18 / 100) * x = 90) : x = 500 :=
sorry

end NUMINAMATH_GPT_find_number_l956_95656


namespace NUMINAMATH_GPT_range_of_sum_l956_95601

theorem range_of_sum (a b c d : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : d > 0) :
  1 < (a / (a + b + d) + b / (a + b + c) + c / (b + c + d) + d / (a + c + d)) ∧
  (a / (a + b + d) + b / (a + b + c) + c / (b + c + d) + d / (a + c + d)) < 2 :=
sorry

end NUMINAMATH_GPT_range_of_sum_l956_95601


namespace NUMINAMATH_GPT_largest_sum_ABC_l956_95661

theorem largest_sum_ABC (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C) (h4 : A * B * C = 3003) : 
  A + B + C ≤ 105 :=
sorry

end NUMINAMATH_GPT_largest_sum_ABC_l956_95661


namespace NUMINAMATH_GPT_perfect_squares_between_2_and_20_l956_95697

-- Defining the conditions and problem statement
theorem perfect_squares_between_2_and_20 : 
  ∃ n, n = 3 ∧ ∀ m, (2 < m ∧ m < 20 ∧ ∃ k, k * k = m) ↔ m = 4 ∨ m = 9 ∨ m = 16 :=
by {
  -- Start the proof process
  sorry -- Placeholder for the proof
}

end NUMINAMATH_GPT_perfect_squares_between_2_and_20_l956_95697


namespace NUMINAMATH_GPT_sally_out_of_pocket_cost_l956_95675

/-- Definitions of the given conditions -/
def given_money : Int := 320
def cost_per_book : Int := 15
def number_of_students : Int := 35

/-- Theorem to prove the amount Sally needs to pay out of pocket -/
theorem sally_out_of_pocket_cost : 
  let total_cost := number_of_students * cost_per_book
  let amount_given := given_money
  let out_of_pocket_cost := total_cost - amount_given
  out_of_pocket_cost = 205 := by
  sorry

end NUMINAMATH_GPT_sally_out_of_pocket_cost_l956_95675


namespace NUMINAMATH_GPT_symmetric_circle_eq_l956_95650

open Real

-- Define the original circle equation and the line of symmetry
def original_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def line_of_symmetry (x y : ℝ) : Prop := y = -x

-- Define the symmetry transformation with respect to the line y = -x
def symmetric_point (x y : ℝ) : ℝ × ℝ := (-y, -x)

-- Define the new circle that is symmetric to the original circle with respect to y = -x
def new_circle (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 1

-- The theorem to be proven
theorem symmetric_circle_eq :
  ∀ x y : ℝ, original_circle (-y) (-x) ↔ new_circle x y := 
by
  sorry

end NUMINAMATH_GPT_symmetric_circle_eq_l956_95650


namespace NUMINAMATH_GPT_relationship_among_abc_l956_95686

noncomputable def a : ℝ := (1/4)^(1/2)
noncomputable def b : ℝ := Real.log 3 / Real.log 2
noncomputable def c : ℝ := (1/3)^(1/2)

theorem relationship_among_abc : b > c ∧ c > a :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_relationship_among_abc_l956_95686


namespace NUMINAMATH_GPT_problem_3000_mod_1001_l956_95604

theorem problem_3000_mod_1001 : (300 ^ 3000 - 1) % 1001 = 0 := 
by
  have h1: (300 ^ 3000) % 7 = 1 := sorry
  have h2: (300 ^ 3000) % 11 = 1 := sorry
  have h3: (300 ^ 3000) % 13 = 1 := sorry
  sorry

end NUMINAMATH_GPT_problem_3000_mod_1001_l956_95604


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_l956_95691

theorem arithmetic_geometric_sequence
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (b : ℕ → ℕ)
  (T : ℕ → ℕ)
  (h1 : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1))
  (h2 : S 3 = 12)
  (h3 : (a 1 + (a 2 - a 1))^2 = a 1 * (a 1 + 2 * (a 2 - a 1) + 2))
  (h4 : ∀ n, b n = (3 ^ n) * a n) :
  (∀ n, a n = 2 * n) ∧ 
  (∀ n, T n = (2 * n - 1) * 3^(n + 1) / 2 + 3 / 2) :=
sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_l956_95691


namespace NUMINAMATH_GPT_find_triples_of_positive_integers_l956_95657

theorem find_triples_of_positive_integers (p q n : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hn_pos : 0 < n) 
  (equation : p * (p + 3) + q * (q + 3) = n * (n + 3)) : 
  (p = 3 ∧ q = 2 ∧ n = 4) :=
sorry

end NUMINAMATH_GPT_find_triples_of_positive_integers_l956_95657


namespace NUMINAMATH_GPT_angle_QPS_l956_95684

-- Definitions of the points and angles
variables (P Q R S : Point)
variables (angle : Point → Point → Point → ℝ)

-- Conditions about the isosceles triangles and angles
variables (isosceles_PQR : PQ = QR)
variables (isosceles_PRS : PR = RS)
variables (R_inside_PQS : ¬(R ∈ convex_hull ℝ {P, Q, S}))
variables (angle_PQR : angle P Q R = 50)
variables (angle_PRS : angle P R S = 120)

-- The theorem we want to prove
theorem angle_QPS : angle Q P S = 35 :=
sorry -- Proof goes here

end NUMINAMATH_GPT_angle_QPS_l956_95684


namespace NUMINAMATH_GPT_inequality_abc_l956_95651

variable (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)

theorem inequality_abc :
  (a * b / (a + b) + b * c / (b + c) + c * a / (c + a)) ≤ (3 * (a * b + b * c + c * a)) / (2 * (a + b + c)) :=
  sorry

end NUMINAMATH_GPT_inequality_abc_l956_95651


namespace NUMINAMATH_GPT_price_increase_eq_20_percent_l956_95683

theorem price_increase_eq_20_percent (a x : ℝ) (h : a * (1 + x) * (1 + x) = a * 1.44) : x = 0.2 :=
by {
  -- This part will contain the proof steps.
  sorry -- Placeholder
}

end NUMINAMATH_GPT_price_increase_eq_20_percent_l956_95683


namespace NUMINAMATH_GPT_smallest_positive_integer_in_form_l956_95689

theorem smallest_positive_integer_in_form (m n : ℤ) : 
  ∃ m n : ℤ, 3001 * m + 24567 * n = 1 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_in_form_l956_95689


namespace NUMINAMATH_GPT_range_of_a_l956_95670

def f (x : ℝ) : ℝ := 3 * x * |x|

theorem range_of_a : {a : ℝ | f (1 - a) + f (2 * a) < 0 } = {a : ℝ | a < -1} :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l956_95670


namespace NUMINAMATH_GPT_part1_solution_part2_solution_l956_95678

section Part1

noncomputable def f (x : ℝ) : ℝ := x^2 + x - 6

theorem part1_solution (x : ℝ) : f x > 0 ↔ x < -3 ∨ x > 2 :=
sorry

end Part1

section Part2

variables (a : ℝ) (ha : a < 0)
noncomputable def g (x : ℝ) : ℝ := a*x^2 + (3 - 2*a)*x - 6

theorem part2_solution (x : ℝ) :
  if h1 : a < -3/2 then g x < 0 ↔ x < -3/a ∨ x > 2
  else if h2 : a = -3/2 then g x < 0 ↔ x ≠ 2
  else -3/2 < a ∧ a < 0 → g x < 0 ↔ x < 2 ∨ x > -3/a :=
sorry

end Part2

end NUMINAMATH_GPT_part1_solution_part2_solution_l956_95678


namespace NUMINAMATH_GPT_make_tea_time_efficiently_l956_95632

theorem make_tea_time_efficiently (minutes_kettle minutes_boil minutes_teapot minutes_teacups minutes_tea_leaves total_estimate total_time : ℕ)
  (h1 : minutes_kettle = 1)
  (h2 : minutes_boil = 15)
  (h3 : minutes_teapot = 1)
  (h4 : minutes_teacups = 1)
  (h5 : minutes_tea_leaves = 2)
  (h6 : total_estimate = 20)
  (h_total_time : total_time = minutes_kettle + minutes_boil) :
  total_time = 16 :=
by
  sorry

end NUMINAMATH_GPT_make_tea_time_efficiently_l956_95632


namespace NUMINAMATH_GPT_f_2016_value_l956_95694

def f : ℝ → ℝ := sorry

axiom f_prop₁ : ∀ x : ℝ, (x + 6) + f x = 0
axiom f_symmetry : ∀ x : ℝ, f (-x) = -f x ∧ f 0 = 0

theorem f_2016_value : f 2016 = 0 :=
by
  sorry

end NUMINAMATH_GPT_f_2016_value_l956_95694


namespace NUMINAMATH_GPT_total_pizza_pieces_l956_95606

-- Definitions based on the conditions
def pieces_per_pizza : Nat := 6
def pizzas_per_student : Nat := 20
def number_of_students : Nat := 10

-- Statement of the theorem
theorem total_pizza_pieces :
  pieces_per_pizza * pizzas_per_student * number_of_students = 1200 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_total_pizza_pieces_l956_95606


namespace NUMINAMATH_GPT_pipe_filling_problem_l956_95623

theorem pipe_filling_problem (x : ℝ) (h : (2 / 15) * x + (1 / 20) * (10 - x) = 1) : x = 6 :=
sorry

end NUMINAMATH_GPT_pipe_filling_problem_l956_95623


namespace NUMINAMATH_GPT_selina_sells_5_shirts_l956_95654

theorem selina_sells_5_shirts
    (pants_price shorts_price shirts_price : ℕ)
    (pants_sold shorts_sold shirts_bought remaining_money : ℕ)
    (total_earnings : ℕ) :
  pants_price = 5 →
  shorts_price = 3 →
  shirts_price = 4 →
  pants_sold = 3 →
  shorts_sold = 5 →
  shirts_bought = 2 →
  remaining_money = 30 →
  total_earnings = remaining_money + shirts_bought * 10 →
  total_earnings = 50 →
  total_earnings = pants_sold * pants_price + shorts_sold * shorts_price + 20 →
  20 / shirts_price = 5 :=
by
  sorry

end NUMINAMATH_GPT_selina_sells_5_shirts_l956_95654
