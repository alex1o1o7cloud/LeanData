import Mathlib

namespace neg_sqrt_two_sq_l1724_172441

theorem neg_sqrt_two_sq : (- Real.sqrt 2) ^ 2 = 2 := 
by
  sorry

end neg_sqrt_two_sq_l1724_172441


namespace gcd_98_63_l1724_172427

theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by
  sorry

end gcd_98_63_l1724_172427


namespace smallest_number_with_2020_divisors_l1724_172458

-- Given a natural number n expressed in terms of its prime factors
def divisor_count (n : ℕ) (f : ℕ → ℕ) : ℕ :=
  f 2 + 1 * f 3 + 1 * f 5 + 1

-- The smallest number with exactly 2020 distinct natural divisors
theorem smallest_number_with_2020_divisors :
  ∃ n : ℕ, divisor_count n = 2020 ∧ 
           n = 2 ^ 100 * 3 ^ 4 * 5 ^ 1 :=
sorry

end smallest_number_with_2020_divisors_l1724_172458


namespace convert_kmph_to_mps_l1724_172423

theorem convert_kmph_to_mps (speed_kmph : ℕ) (one_kilometer_in_meters : ℕ) (one_hour_in_seconds : ℕ) :
  speed_kmph = 108 →
  one_kilometer_in_meters = 1000 →
  one_hour_in_seconds = 3600 →
  (speed_kmph * one_kilometer_in_meters) / one_hour_in_seconds = 30 := by
  intros h1 h2 h3
  sorry

end convert_kmph_to_mps_l1724_172423


namespace ellipse_eccentricity_l1724_172412

theorem ellipse_eccentricity (k : ℝ) : 
  (∀ x y : ℝ, x^2 / (k + 8) + y^2 / 9 = 1) ∧ (∃ e : ℝ, e = 1 / 2) → 
  (k = 4 ∨ k = -5 / 4) := sorry

end ellipse_eccentricity_l1724_172412


namespace spaghetti_tortellini_ratio_l1724_172414

theorem spaghetti_tortellini_ratio (students_surveyed : ℕ)
                                    (spaghetti_lovers : ℕ)
                                    (tortellini_lovers : ℕ)
                                    (h1 : students_surveyed = 850)
                                    (h2 : spaghetti_lovers = 300)
                                    (h3 : tortellini_lovers = 200) :
  spaghetti_lovers / tortellini_lovers = 3 / 2 :=
by
  sorry

end spaghetti_tortellini_ratio_l1724_172414


namespace solve_S20_minus_2S10_l1724_172486

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, n > 0 → a n > 0) ∧
  (∀ n : ℕ, n ≥ 2 → S n = (n / (n - 1 : ℝ)) * (a n ^ 2 - a 1 ^ 2))

theorem solve_S20_minus_2S10 :
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ),
    arithmetic_sequence a S →
    S 20 - 2 * S 10 = 50 :=
by
  intros
  sorry

end solve_S20_minus_2S10_l1724_172486


namespace fifth_term_is_2_11_over_60_l1724_172434

noncomputable def fifth_term_geo_prog (a₁ a₂ a₃ : ℝ) (r : ℝ) : ℝ :=
  a₃ * r^2

theorem fifth_term_is_2_11_over_60
  (a₁ a₂ a₃ : ℝ)
  (h₁ : a₁ = 2^(1/4))
  (h₂ : a₂ = 2^(1/5))
  (h₃ : a₃ = 2^(1/6))
  (r : ℝ)
  (common_ratio : r = a₂ / a₁) :
  fifth_term_geo_prog a₁ a₂ a₃ r = 2^(11/60) :=
by
  sorry

end fifth_term_is_2_11_over_60_l1724_172434


namespace calculate_expression_l1724_172419

variable (x : ℝ)

def quadratic_condition : Prop := x^2 + x - 1 = 0

theorem calculate_expression (h : quadratic_condition x) : 2*x^3 + 3*x^2 - x = 1 := by
  sorry

end calculate_expression_l1724_172419


namespace find_n_l1724_172460

theorem find_n : ∃ n : ℤ, 100 ≤ n ∧ n ≤ 280 ∧ Real.cos (n * Real.pi / 180) = Real.cos (317 * Real.pi / 180) ∧ n = 317 := 
by
  sorry

end find_n_l1724_172460


namespace geometric_sequence_sum_l1724_172493

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_pos : ∀ n, a n > 0)
  (h1 : a 1 + a 3 = 3)
  (h2 : a 4 + a 6 = 6):
  a 1 * a 3 + a 2 * a 4 + a 3 * a 5 + a 4 * a 6 + a 5 * a 7 = 62 :=
sorry

end geometric_sequence_sum_l1724_172493


namespace find_a_l1724_172465

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {1, 3, a}
def B (a : ℝ) : Set ℝ := {1, a^2}

-- Theorem to be proved
theorem find_a (a : ℝ) 
  (h1 : A a ∪ B a = {1, 3, a}) : a = 0 ∨ a = 1 ∨ a = -1 :=
by
  sorry

end find_a_l1724_172465


namespace necessary_and_sufficient_condition_l1724_172490

theorem necessary_and_sufficient_condition (x : ℝ) :
  (x - 2) * (x - 3) ≤ 0 ↔ |x - 2| + |x - 3| = 1 := sorry

end necessary_and_sufficient_condition_l1724_172490


namespace gcd_12m_18n_l1724_172402

theorem gcd_12m_18n (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_gcd_mn : m.gcd n = 10) : (12 * m).gcd (18 * n) = 60 := by
  sorry

end gcd_12m_18n_l1724_172402


namespace speed_of_X_l1724_172480

theorem speed_of_X (t1 t2 Vx : ℝ) (h1 : t2 - t1 = 3) 
  (h2 : 3 * Vx + Vx * t1 = 60 * t1 + 30)
  (h3 : 3 * Vx + Vx * t2 + 30 = 60 * t2) : Vx = 60 :=
by sorry

end speed_of_X_l1724_172480


namespace smallest_possible_AC_l1724_172487

theorem smallest_possible_AC 
    (AB AC CD : ℤ) 
    (BD_squared : ℕ) 
    (h_isosceles : AB = AC)
    (h_point_D : ∃ D : ℤ, D = CD)
    (h_perpendicular : BD_squared = 85) 
    (h_integers : ∃ x y : ℤ, AC = x ∧ CD = y) 
    : AC = 11 :=
by
  sorry

end smallest_possible_AC_l1724_172487


namespace smallest_two_digit_integer_l1724_172426

-- Define the problem parameters and condition
theorem smallest_two_digit_integer (n : ℕ) (a b : ℕ) 
  (h1 : n = 10 * a + b) 
  (h2 : 1 ≤ a) (h3 : a ≤ 9) (h4 : 0 ≤ b) (h5 : b ≤ 9) 
  (h6 : 19 * a = 8 * b + 3) : 
  n = 12 :=
sorry

end smallest_two_digit_integer_l1724_172426


namespace sum_of_squares_eq_ten_l1724_172416

noncomputable def x1 : ℝ := Real.sqrt 3 - Real.sqrt 2
noncomputable def x2 : ℝ := Real.sqrt 3 + Real.sqrt 2

theorem sum_of_squares_eq_ten : x1^2 + x2^2 = 10 := 
by
  sorry

end sum_of_squares_eq_ten_l1724_172416


namespace base_of_second_fraction_l1724_172479

theorem base_of_second_fraction (x k : ℝ) (h1 : (1/2)^18 * (1/x)^k = 1/18^18) (h2 : k = 9) : x = 9 :=
by
  sorry

end base_of_second_fraction_l1724_172479


namespace find_b_l1724_172422

-- Define the lines and the condition of parallelism
def line1 := ∀ (x y b : ℝ), 4 * y + 8 * b = 16 * x
def line2 := ∀ (x y b : ℝ), y - 2 = (b - 3) * x
def are_parallel (m1 m2 : ℝ) := m1 = m2

-- Translate the problem to a Lean statement
theorem find_b (b : ℝ) : (∀ x y, 4 * y + 8 * b = 16 * x) → (∀ x y, y - 2 = (b - 3) * x) → b = 7 :=
by
  sorry

end find_b_l1724_172422


namespace quadratic_nonnegative_quadratic_inv_nonnegative_l1724_172498

-- Problem Definitions and Proof Statements

variables {R : Type*} [LinearOrderedField R]

def f (a b c x : R) : R := a * x^2 + 2 * b * x + c

theorem quadratic_nonnegative {a b c : R} (ha : a ≠ 0) (h : ∀ x : R, f a b c x ≥ 0) : 
  a ≥ 0 ∧ c ≥ 0 ∧ a * c - b^2 ≥ 0 :=
sorry

theorem quadratic_inv_nonnegative {a b c : R} (ha : a ≥ 0) (hc : c ≥ 0) (hac : a * c - b^2 ≥ 0) :
  ∀ x : R, f a b c x ≥ 0 :=
sorry

end quadratic_nonnegative_quadratic_inv_nonnegative_l1724_172498


namespace original_investment_amount_l1724_172421

-- Definitions
def annual_interest_rate : ℝ := 0.04
def investment_period_years : ℝ := 0.25
def final_amount : ℝ := 10204

-- Statement to prove
theorem original_investment_amount :
  let P := final_amount / (1 + annual_interest_rate * investment_period_years)
  P = 10104 :=
by
  -- Placeholder for the proof
  sorry

end original_investment_amount_l1724_172421


namespace range_of_k_l1724_172415

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / x
noncomputable def g (x : ℝ) : ℝ := x / Real.exp x

theorem range_of_k (k : ℝ) (x1 x2 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) :
  (g x1 / k ≤ f x2 / (k + 1)) ↔ k ≥ 1 / (2 * Real.exp 1 - 1) := by
  sorry

end range_of_k_l1724_172415


namespace find_w_l1724_172447

noncomputable def roots_cubic_eq (x : ℝ) : ℝ := x^3 + 2 * x^2 + 5 * x - 8

def p : ℝ := sorry -- one root of x^3 + 2x^2 + 5x - 8 = 0
def q : ℝ := sorry -- another root of x^3 + 2x^2 + 5x - 8 = 0
def r : ℝ := sorry -- another root of x^3 + 2x^2 + 5x - 8 = 0

theorem find_w 
  (h1 : roots_cubic_eq p = 0)
  (h2 : roots_cubic_eq q = 0)
  (h3 : roots_cubic_eq r = 0)
  (h4 : p + q + r = -2): 
  ∃ w : ℝ, w = 18 := 
sorry

end find_w_l1724_172447


namespace arithmetic_sequence_problem_l1724_172471

noncomputable def a_n (n : ℕ) : ℝ := sorry  -- Define the arithmetic sequence

theorem arithmetic_sequence_problem
  (a_4 : ℝ) (a_9 : ℝ)
  (h_a4 : a_4 = 5)
  (h_a9 : a_9 = 17)
  (h_arithmetic : ∀ n : ℕ, a_n (n + 1) = a_n n + (a_n 2 - a_n 1)) :
  a_n 14 = 29 :=
by
  -- the proof will utilize the property of arithmetic sequence and substitutions
  sorry

end arithmetic_sequence_problem_l1724_172471


namespace largest_arithmetic_seq_3digit_l1724_172499

theorem largest_arithmetic_seq_3digit : 
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ (∃ a b c : ℕ, n = 100*a + 10*b + c ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a = 9 ∧ ∃ d, b = a - d ∧ c = a - 2*d) ∧ n = 963 :=
by sorry

end largest_arithmetic_seq_3digit_l1724_172499


namespace defective_pens_count_l1724_172467

theorem defective_pens_count (total_pens : ℕ) (prob_not_defective : ℚ) (D : ℕ) 
  (h1 : total_pens = 8) 
  (h2 : prob_not_defective = 0.5357142857142857) : 
  D = 2 := 
by
  sorry

end defective_pens_count_l1724_172467


namespace isosceles_triangle_perimeter_l1724_172495

/-
Problem:
Given an isosceles triangle with side lengths 5 and 6, prove that the perimeter of the triangle is either 16 or 17.
-/

theorem isosceles_triangle_perimeter (a b : ℕ) (h₁ : a = 5 ∨ a = 6) (h₂ : b = 5 ∨ b = 6) (h₃ : a ≠ b) : 
  (a + a + b = 16 ∨ a + a + b = 17) ∧ (b + b + a = 16 ∨ b + b + a = 17) :=
by
  sorry

end isosceles_triangle_perimeter_l1724_172495


namespace simplify_and_evaluate_l1724_172410

theorem simplify_and_evaluate (m : ℝ) (h : m = 5) :
  (m + 2 - (5 / (m - 2))) / ((3 * m - m^2) / (m - 2)) = - (8 / 5) :=
by
  sorry

end simplify_and_evaluate_l1724_172410


namespace find_h_l1724_172430

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ := 3 * x^2 + 9 * x + 20

-- Main statement
theorem find_h : 
  ∃ (a k h : ℝ), (∀ x : ℝ, quadratic_expr x = a * (x - h)^2 + k) ∧ h = -3/2 := 
by
  sorry

end find_h_l1724_172430


namespace time_difference_l1724_172478

-- Definitions for the problem conditions
def Zoe_speed : ℕ := 9 -- Zoe's speed in minutes per mile
def Henry_speed : ℕ := 7 -- Henry's speed in minutes per mile
def Race_length : ℕ := 12 -- Race length in miles

-- Theorem to prove the time difference
theorem time_difference : (Race_length * Zoe_speed) - (Race_length * Henry_speed) = 24 :=
by
  sorry

end time_difference_l1724_172478


namespace num_rectangles_in_5x5_grid_l1724_172497

theorem num_rectangles_in_5x5_grid : 
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  num_ways_choose_2 * num_ways_choose_2 = 100 :=
by
  -- Definitions based on conditions
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  
  -- Required proof (just showing the statement here)
  show num_ways_choose_2 * num_ways_choose_2 = 100
  sorry

end num_rectangles_in_5x5_grid_l1724_172497


namespace min_value_of_reciprocal_sum_l1724_172466

theorem min_value_of_reciprocal_sum {a b : ℝ} (h : a > 0 ∧ b > 0)
  (h_circle1 : ∀ x y : ℝ, x^2 + y^2 = 4)
  (h_circle2 : ∀ x y : ℝ, (x - 2)^2 + (y - 2)^2 = 4)
  (h_common_chord : a + b = 2) :
  (1 / a + 9 / b = 8) := 
sorry

end min_value_of_reciprocal_sum_l1724_172466


namespace correct_operation_l1724_172438

theorem correct_operation (x : ℝ) (hx : x ≠ 0) : x^2 / x^8 = 1 / x^6 :=
by
  sorry

end correct_operation_l1724_172438


namespace square_area_from_circle_l1724_172400

-- Define the conditions for the circle's equation
def circle_equation (x y : ℝ) : Prop :=
  2 * x^2 = -2 * y^2 + 8 * x - 8 * y + 28 

-- State the main theorem to prove the area of the square
theorem square_area_from_circle (x y : ℝ) (h : circle_equation x y) :
  ∃ s : ℝ, s^2 = 88 :=
sorry

end square_area_from_circle_l1724_172400


namespace original_equation_proof_l1724_172457

theorem original_equation_proof :
  ∃ (A O H M J : ℕ),
  A ≠ O ∧ A ≠ H ∧ A ≠ M ∧ A ≠ J ∧
  O ≠ H ∧ O ≠ M ∧ O ≠ J ∧
  H ≠ M ∧ H ≠ J ∧
  M ≠ J ∧
  A + 8 * (10 * O + H) = 10 * M + J ∧
  (O = 1) ∧ (H = 2) ∧ (M = 9) ∧ (J = 6) ∧ (A = 0) :=
by
  sorry

end original_equation_proof_l1724_172457


namespace rhombus_construction_possible_l1724_172442

-- Definitions for points, lines, and distances
variables {Point : Type} {Line : Type}
def is_parallel (l1 l2 : Line) : Prop := sorry
def distance_between (l1 l2 : Line) : ℝ := sorry
def point_on_line (p : Point) (l : Line) : Prop := sorry

-- Given parallel lines l₁ and l₂ and their distance a
variable {l1 l2 : Line}
variable (a : ℝ)
axiom parallel_lines : is_parallel l1 l2
axiom distance_eq_a : distance_between l1 l2 = a

-- Given points A and B
variable (A B : Point)

-- Definition of a rhombus that meets the criteria
noncomputable def construct_rhombus (A B : Point) (l1 l2 : Line) (a : ℝ) : Prop :=
  ∃ C1 C2 D1 D2 : Point, 
    point_on_line C1 l1 ∧ 
    point_on_line D1 l2 ∧ 
    point_on_line C2 l1 ∧ 
    point_on_line D2 l2 ∧ 
    sorry -- additional conditions ensuring sides passing through A and B and forming a rhombus

theorem rhombus_construction_possible : 
  construct_rhombus A B l1 l2 a :=
sorry

end rhombus_construction_possible_l1724_172442


namespace cut_scene_length_l1724_172451

theorem cut_scene_length
  (original_length final_length : ℕ)
  (h_original : original_length = 60)
  (h_final : final_length = 54) :
  original_length - final_length = 6 :=
by 
  sorry

end cut_scene_length_l1724_172451


namespace hexagon_chord_length_valid_l1724_172488

def hexagon_inscribed_chord_length : ℚ := 48 / 49

theorem hexagon_chord_length_valid : 
    ∃ (p q : ℕ), gcd p q = 1 ∧ hexagon_inscribed_chord_length = p / q ∧ p + q = 529 :=
sorry

end hexagon_chord_length_valid_l1724_172488


namespace part1_part2_l1724_172406

def f (x : ℝ) := |x + 2|

theorem part1 (x : ℝ) : 2 * f x < 4 - |x - 1| ↔ -7/3 < x ∧ x < -1 :=
by sorry

theorem part2 (a : ℝ) (m n : ℝ) (hmn : m + n = 1) (hm : 0 < m) (hn : 0 < n) :
  (∀ x, |x - a| - f x ≤ 1/m + 1/n) ↔ -6 ≤ a ∧ a ≤ 2 :=
by sorry

end part1_part2_l1724_172406


namespace dye_jobs_scheduled_l1724_172405

noncomputable def revenue_from_haircuts (n : ℕ) : ℕ := n * 30
noncomputable def revenue_from_perms (n : ℕ) : ℕ := n * 40
noncomputable def revenue_from_dye_jobs (n : ℕ) : ℕ := n * (60 - 10)
noncomputable def total_revenue (haircuts perms dye_jobs : ℕ) (tips : ℕ) : ℕ :=
  revenue_from_haircuts haircuts + revenue_from_perms perms + revenue_from_dye_jobs dye_jobs + tips

theorem dye_jobs_scheduled : 
  (total_revenue 4 1 dye_jobs 50 = 310) → (dye_jobs = 2) := 
by
  sorry

end dye_jobs_scheduled_l1724_172405


namespace line_passes_through_fixed_point_l1724_172411

theorem line_passes_through_fixed_point (m n : ℝ) (h : m + 2 * n - 1 = 0) :
  mx + 3 * y + n = 0 → (x, y) = (1/2, -1/6) :=
by
  sorry

end line_passes_through_fixed_point_l1724_172411


namespace problem_equivalent_proof_l1724_172431

theorem problem_equivalent_proof (a : ℝ) (h : a / 2 - 2 / a = 5) :
  (a^8 - 256) / (16 * a^4) * (2 * a / (a^2 + 4)) = 81 :=
sorry

end problem_equivalent_proof_l1724_172431


namespace sum_of_reciprocals_of_factors_of_12_l1724_172424

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l1724_172424


namespace problem1_problem2_l1724_172473

-- Define the conditions
variable (a x : ℝ)
variable (h_gt_zero : x > 0) (a_gt_zero : a > 0)

-- Problem 1: Prove that 0 < x ≤ 300
theorem problem1 (h: 12 * (500 - x) * (1 + 0.005 * x) ≥ 12 * 500) : 0 < x ∧ x ≤ 300 := 
sorry

-- Problem 2: Prove that 0 < a ≤ 5.5 given the conditions
theorem problem2 (h1 : 12 * (a - 13 / 1000 * x) * x ≤ 12 * (500 - x) * (1 + 0.005 * x))
                (h2 : x = 250) : 0 < a ∧ a ≤ 5.5 := 
sorry

end problem1_problem2_l1724_172473


namespace error_percent_in_area_l1724_172463

theorem error_percent_in_area 
    (L W : ℝ) 
    (measured_length : ℝ := 1.09 * L) 
    (measured_width : ℝ := 0.92 * W) 
    (correct_area : ℝ := L * W) 
    (incorrect_area : ℝ := measured_length * measured_width) :
    100 * (incorrect_area - correct_area) / correct_area = 0.28 :=
by
  sorry

end error_percent_in_area_l1724_172463


namespace compute_expr_l1724_172452

open Real

-- Define the polynomial and its roots.
def polynomial (x : ℝ) := 3 * x^2 - 5 * x - 2

-- Given conditions: p and q are roots of the polynomial.
def is_root (p q : ℝ) : Prop := 
  polynomial p = 0 ∧ polynomial q = 0

-- The main theorem.
theorem compute_expr (p q : ℝ) (h : is_root p q) : 
  ∃ k : ℝ, k = p - q ∧ (p ≠ q) → (9 * p^3 + 9 * q^3) / (p - q) = 215 / (3 * (p - q)) :=
sorry

end compute_expr_l1724_172452


namespace find_g2_l1724_172464

-- Given conditions:
variables (g : ℝ → ℝ) 
axiom cond1 : ∀ (x y : ℝ), x * g y = 2 * y * g x
axiom cond2 : g 10 = 5

-- Proof to show g(2) = 2
theorem find_g2 : g 2 = 2 := 
by
  -- Skipping the actual proof
  sorry

end find_g2_l1724_172464


namespace arrangements_TOOTH_l1724_172453
-- Import necessary libraries

-- Define the problem conditions
def word_length : Nat := 5
def count_T : Nat := 2
def count_O : Nat := 2

-- State the problem as a theorem
theorem arrangements_TOOTH : 
  (word_length.factorial / (count_T.factorial * count_O.factorial)) = 30 := by
  sorry

end arrangements_TOOTH_l1724_172453


namespace solution_l1724_172439

noncomputable def problem (a b c x y z : ℝ) :=
  11 * x + b * y + c * z = 0 ∧
  a * x + 19 * y + c * z = 0 ∧
  a * x + b * y + 37 * z = 0 ∧
  a ≠ 11 ∧
  x ≠ 0

theorem solution (a b c x y z : ℝ) (h : problem a b c x y z) :
  (a / (a - 11)) + (b / (b - 19)) + (c / (c - 37)) = 1 :=
sorry

end solution_l1724_172439


namespace rectangle_circle_area_ratio_l1724_172449

theorem rectangle_circle_area_ratio (w r: ℝ) (h1: 3 * w = π * r) (h2: 2 * w = l) :
  (l * w) / (π * r ^ 2) = 2 * π / 9 :=
by 
  sorry

end rectangle_circle_area_ratio_l1724_172449


namespace total_votes_l1724_172445

theorem total_votes (V : ℝ) (h1 : 0.35 * V + (0.35 * V + 1650) = V) : V = 5500 := 
by 
  sorry

end total_votes_l1724_172445


namespace find_rates_l1724_172468

theorem find_rates
  (d b p t_p t_b t_w: ℕ)
  (rp rb rw: ℚ)
  (h1: d = b + 10)
  (h2: b = 3 * p)
  (h3: p = 50)
  (h4: t_p = 4)
  (h5: t_b = 2)
  (h6: t_w = 5)
  (h7: rp = p / t_p)
  (h8: rb = b / t_b)
  (h9: rw = d / t_w):
  rp = 12.5 ∧ rb = 75 ∧ rw = 32 := by
  sorry

end find_rates_l1724_172468


namespace no_solution_eq_eight_diff_l1724_172443

theorem no_solution_eq_eight_diff (k : ℕ) (h1 : k > 0) (h2 : k ≤ 99) 
  (h3 : ∀ x y : ℕ, x^2 - k * y^2 ≠ 8) : 
  (99 - 3 = 96) := 
by 
  sorry

end no_solution_eq_eight_diff_l1724_172443


namespace max_value_of_expression_l1724_172425

theorem max_value_of_expression (m : ℝ) : 4 - |2 - m| ≤ 4 :=
by 
  sorry

end max_value_of_expression_l1724_172425


namespace exists_infinite_n_for_multiple_of_prime_l1724_172408

theorem exists_infinite_n_for_multiple_of_prime (p : ℕ) (hp : Nat.Prime p) :
  ∃ᶠ n in at_top, 2 ^ n - n ≡ 0 [MOD p] :=
by
  sorry

end exists_infinite_n_for_multiple_of_prime_l1724_172408


namespace decreasing_interval_of_even_function_l1724_172450

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k*x^2 + (k - 1)*x + 2

theorem decreasing_interval_of_even_function (k : ℝ) (h : ∀ x : ℝ, f k x = f k (-x)) :
  ∃ k : ℝ, k = 1 ∧ ∀ x : ℝ, (x < 0 → f k x > f k (-x)) := 
sorry

end decreasing_interval_of_even_function_l1724_172450


namespace highest_possible_value_l1724_172407

theorem highest_possible_value 
  (t q r1 r2 : ℝ)
  (h_eq : r1 + r2 = t)
  (h_cond : ∀ n : ℕ, n > 0 → r1^n + r2^n = t) :
  t = 2 → q = 1 → 
  r1 = 1 → r2 = 1 →
  (1 / r1^1004 + 1 / r2^1004 = 2) :=
by
  intros h_t h_q h_r1 h_r2
  rw [h_r1, h_r2]
  norm_num

end highest_possible_value_l1724_172407


namespace medium_pizza_promotion_price_l1724_172494

-- Define the conditions
def regular_price_medium_pizza : ℝ := 18
def total_savings : ℝ := 39
def number_of_medium_pizzas : ℝ := 3

-- Define the goal
theorem medium_pizza_promotion_price : 
  ∃ P : ℝ, 3 * regular_price_medium_pizza - 3 * P = total_savings ∧ P = 5 := 
by
  sorry

end medium_pizza_promotion_price_l1724_172494


namespace surface_area_of_sphere_l1724_172435

theorem surface_area_of_sphere (a : Real) (h : a = 2 * Real.sqrt 3) : 
  (4 * Real.pi * ((Real.sqrt 3 * a / 2) ^ 2)) = 36 * Real.pi :=
by
  sorry

end surface_area_of_sphere_l1724_172435


namespace initial_amount_l1724_172477

theorem initial_amount (P : ℝ) (h1 : ∀ x : ℝ, x * (9 / 8) * (9 / 8) = 81000) : P = 64000 :=
sorry

end initial_amount_l1724_172477


namespace ratio_of_b_to_a_l1724_172484

theorem ratio_of_b_to_a (a b c : ℕ) (x y : ℕ) 
  (h1 : a > 0) 
  (h2 : x = 100 * a + 10 * b + c)
  (h3 : y = 100 * 9 + 10 * 9 + 9 - 241) 
  (h4 : x = y) :
  b = 5 → a = 7 → (b / a : ℚ) = 5 / 7 := 
by
  intros
  subst_vars
  sorry

end ratio_of_b_to_a_l1724_172484


namespace standard_parabola_with_symmetry_axis_eq_1_l1724_172448

-- Define the condition that the axis of symmetry is x = 1
def axis_of_symmetry_x_eq_one (x : ℝ) : Prop :=
  x = 1

-- Define the standard equation of the parabola y^2 = -4x
def standard_parabola_eq (y x : ℝ) : Prop :=
  y^2 = -4 * x

-- Theorem: Prove that given the axis of symmetry of the parabola is x = 1,
-- the standard equation of the parabola is y^2 = -4x.
theorem standard_parabola_with_symmetry_axis_eq_1 : ∀ (x y : ℝ),
  axis_of_symmetry_x_eq_one x → standard_parabola_eq y x :=
by
  intros
  sorry

end standard_parabola_with_symmetry_axis_eq_1_l1724_172448


namespace find_power_y_l1724_172492

theorem find_power_y 
  (y : ℕ) 
  (h : (12 : ℝ)^y * (6 : ℝ)^3 / (432 : ℝ) = 72) : 
  y = 2 :=
by
  sorry

end find_power_y_l1724_172492


namespace number_of_even_multiples_of_3_l1724_172429

theorem number_of_even_multiples_of_3 :
  ∃ n, n = (198 - 6) / 6 + 1 := by
  sorry

end number_of_even_multiples_of_3_l1724_172429


namespace correct_units_l1724_172420

def units_time := ["hour", "minute", "second"]
def units_mass := ["gram", "kilogram", "ton"]
def units_length := ["millimeter", "centimeter", "decimeter", "meter", "kilometer"]

theorem correct_units :
  (units_time = ["hour", "minute", "second"]) ∧
  (units_mass = ["gram", "kilogram", "ton"]) ∧
  (units_length = ["millimeter", "centimeter", "decimeter", "meter", "kilometer"]) :=
by
  -- Please provide the proof here
  sorry

end correct_units_l1724_172420


namespace solve_for_x_l1724_172485

theorem solve_for_x (x : ℝ) 
  (h : 6 * x + 12 * x = 558 - 9 * (x - 4)) : 
  x = 22 := 
sorry

end solve_for_x_l1724_172485


namespace max_bag_weight_l1724_172472

-- Let's define the conditions first
def green_beans_weight := 4
def milk_weight := 6
def carrots_weight := 2 * green_beans_weight
def additional_capacity := 2

-- The total weight of groceries
def total_groceries_weight := green_beans_weight + milk_weight + carrots_weight

-- The maximum weight the bag can hold is the total weight of groceries plus the additional capacity
theorem max_bag_weight : (total_groceries_weight + additional_capacity) = 20 := by
  sorry

end max_bag_weight_l1724_172472


namespace net_progress_l1724_172461

def lost_yards : Int := 5
def gained_yards : Int := 7

theorem net_progress : gained_yards - lost_yards = 2 := 
by
  sorry

end net_progress_l1724_172461


namespace largest_prime_factor_1001_l1724_172446

theorem largest_prime_factor_1001 : ∃ p : ℕ, p = 13 ∧ Prime p ∧ (∀ q : ℕ, Prime q ∧ q ∣ 1001 → q ≤ 13) := sorry

end largest_prime_factor_1001_l1724_172446


namespace scientific_notation_141260_million_l1724_172403

theorem scientific_notation_141260_million :
  ∃ (a : ℝ) (n : ℤ), 141260 * 10^6 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.4126 ∧ n = 5 :=
by
  sorry

end scientific_notation_141260_million_l1724_172403


namespace geometric_sequence_eighth_term_l1724_172481

variable (a r : ℕ)
variable (h1 : a = 3)
variable (h2 : a * r^6 = 2187)
variable (h3 : a = 3)

theorem geometric_sequence_eighth_term (a r : ℕ) (h1 : a = 3) (h2 : a * r^6 = 2187) (h3 : a = 3) :
  a * r^7 = 6561 := by
  sorry

end geometric_sequence_eighth_term_l1724_172481


namespace find_angle_EFC_l1724_172436

-- Define the properties of the problem.
def is_isosceles (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist A C

def angle (A B C : ℝ × ℝ) : ℝ :=
  -- Compute the angle using the law of cosines or any other method
  sorry

def perpendicular_foot (P A B : ℝ × ℝ) : ℝ × ℝ :=
  -- Compute the foot of the perpendicular from point P to the line AB
  sorry

noncomputable def main_problem : Prop :=
  ∀ (A B C D E F : ℝ × ℝ),
    is_isosceles A B C →
    angle A B C = 22 →  -- Given angle BAC
    ∃ x : ℝ, dist B D = 2 * dist D C →  -- Point D such that BD = 2 * CD
    E = perpendicular_foot B A D →
    F = perpendicular_foot B A C →
    angle E F C = 33  -- required to prove

-- Statement of the main problem.
theorem find_angle_EFC : main_problem := sorry

end find_angle_EFC_l1724_172436


namespace center_of_circle_l1724_172476

theorem center_of_circle (ρ θ : ℝ) (h : ρ = 2 * Real.cos (θ - π / 4)) : (ρ, θ) = (1, π / 4) :=
sorry

end center_of_circle_l1724_172476


namespace area_of_square_l1724_172401

def side_length (x : ℕ) : ℕ := 3 * x - 12

def side_length_alt (x : ℕ) : ℕ := 18 - 2 * x

theorem area_of_square (x : ℕ) (h : 3 * x - 12 = 18 - 2 * x) : (side_length x) ^ 2 = 36 :=
by
  sorry

end area_of_square_l1724_172401


namespace percentage_problem_l1724_172462

theorem percentage_problem (x : ℝ) (h : 0.20 * x = 100) : 1.20 * x = 600 :=
sorry

end percentage_problem_l1724_172462


namespace initial_salt_percentage_is_10_l1724_172483

-- Declarations for terminology
def initial_volume : ℕ := 72
def added_water : ℕ := 18
def final_volume : ℕ := initial_volume + added_water
def final_salt_percentage : ℝ := 0.08

-- Amount of salt in the initial solution
def initial_salt_amount (P : ℝ) := initial_volume * P

-- Amount of salt in the final solution
def final_salt_amount : ℝ := final_volume * final_salt_percentage

-- Proof that the initial percentage of salt was 10%
theorem initial_salt_percentage_is_10 :
  ∃ P : ℝ, initial_salt_amount P = final_salt_amount ∧ P = 0.1 :=
by
  sorry

end initial_salt_percentage_is_10_l1724_172483


namespace positive_sum_minus_terms_gt_zero_l1724_172417

theorem positive_sum_minus_terms_gt_zero 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = 1) : 
  a^2 + a * b + b^2 - a - b > 0 := 
by
  sorry

end positive_sum_minus_terms_gt_zero_l1724_172417


namespace solve_eq_l1724_172456

theorem solve_eq (x : ℝ) : (x - 2)^2 = 9 * x^2 ↔ x = -1 ∨ x = 1 / 2 := by
  sorry

end solve_eq_l1724_172456


namespace range_of_a_l1724_172491

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then a + |x - 2|
  else x^2 - 2 * a * x + 2 * a

theorem range_of_a (a : ℝ) (x : ℝ) (h : ∀ x : ℝ, f a x ≥ 0) : -1 ≤ a ∧ a ≤ 2 :=
sorry

end range_of_a_l1724_172491


namespace ferris_wheel_small_seat_capacity_l1724_172454

def num_small_seats : Nat := 2
def capacity_per_small_seat : Nat := 14

theorem ferris_wheel_small_seat_capacity : num_small_seats * capacity_per_small_seat = 28 := by
  sorry

end ferris_wheel_small_seat_capacity_l1724_172454


namespace find_other_number_l1724_172428

theorem find_other_number (B : ℕ) (HCF : ℕ) (LCM : ℕ) (A : ℕ) 
  (h1 : A = 24) 
  (h2 : HCF = 16) 
  (h3 : LCM = 312) 
  (h4 : HCF * LCM = A * B) :
  B = 208 :=
by
  sorry

end find_other_number_l1724_172428


namespace parallel_lines_necessity_parallel_lines_not_sufficiency_l1724_172440

theorem parallel_lines_necessity (a b : ℝ) (h : 2 * b = a * 2) : ab = 4 :=
by sorry

theorem parallel_lines_not_sufficiency (a b : ℝ) (h : ab = 4) : 
  ¬ (2 * b = a * 2 ∧ (2 * a - 2 = 0 -> 2 * b - 2 = 0)) :=
by sorry

end parallel_lines_necessity_parallel_lines_not_sufficiency_l1724_172440


namespace increase_in_average_l1724_172469

theorem increase_in_average {a1 a2 a3 a4 : ℕ} 
                            (h1 : a1 = 92) 
                            (h2 : a2 = 89) 
                            (h3 : a3 = 91) 
                            (h4 : a4 = 93) : 
    ((a1 + a2 + a3 + a4 : ℚ) / 4) - ((a1 + a2 + a3 : ℚ) / 3) = 0.58 := 
by
  sorry

end increase_in_average_l1724_172469


namespace cube_div_identity_l1724_172459

theorem cube_div_identity (a b : ℕ) (h₁ : a = 6) (h₂ : b = 3) :
  (a^3 + b^3) / (a^2 - a * b + b^2) = 9 := by
  sorry

end cube_div_identity_l1724_172459


namespace calc_nabla_example_l1724_172489

-- Define the custom operation ∇
def op_nabla (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

-- State the proof problem
theorem calc_nabla_example : op_nabla (op_nabla 2 3) (op_nabla 4 5) = 49 / 56 := by
  sorry

end calc_nabla_example_l1724_172489


namespace sum_of_three_largest_consecutive_numbers_l1724_172413

theorem sum_of_three_largest_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  -- proof using Lean tactics to be added here
  sorry

end sum_of_three_largest_consecutive_numbers_l1724_172413


namespace girls_from_clay_is_30_l1724_172482

-- Definitions for the given conditions
def total_students : ℕ := 150
def total_boys : ℕ := 90
def total_girls : ℕ := 60
def students_jonas : ℕ := 50
def students_clay : ℕ := 70
def students_hart : ℕ := 30
def boys_jonas : ℕ := 25

-- Theorem to prove that the number of girls from Clay Middle School is 30
theorem girls_from_clay_is_30 
  (h1 : total_students = 150)
  (h2 : total_boys = 90)
  (h3 : total_girls = 60)
  (h4 : students_jonas = 50)
  (h5 : students_clay = 70)
  (h6 : students_hart = 30)
  (h7 : boys_jonas = 25) : 
  ∃ girls_clay : ℕ, girls_clay = 30 :=
by 
  sorry

end girls_from_clay_is_30_l1724_172482


namespace domain_sqrt_log_l1724_172418

def domain_condition1 (x : ℝ) : Prop := x + 1 ≥ 0
def domain_condition2 (x : ℝ) : Prop := 6 - 3 * x > 0

theorem domain_sqrt_log (x : ℝ) : domain_condition1 x ∧ domain_condition2 x ↔ -1 ≤ x ∧ x < 2 :=
  sorry

end domain_sqrt_log_l1724_172418


namespace no_intersection_l1724_172474

-- Definitions of the sets M1 and M2 based on parameters A, B, C and integer x
def M1 (A B : ℤ) : Set ℤ := {y | ∃ x : ℤ, y = x^2 + A * x + B}
def M2 (C : ℤ) : Set ℤ := {y | ∃ x : ℤ, y = 2 * x^2 + 2 * x + C}

-- The statement of the theorem
theorem no_intersection (A B : ℤ) : ∃ C : ℤ, M1 A B ∩ M2 C = ∅ :=
sorry

end no_intersection_l1724_172474


namespace milk_leftover_l1724_172475

variable {v : ℕ} -- 'v' is the number of sets of milkshakes in the 2:1 ratio.
variables {milk vanilla_chocolate : ℕ} -- spoon amounts per milkshake types
variables {total_milk total_vanilla_ice_cream total_chocolate_ice_cream : ℕ} -- total amount constraints
variables {milk_left : ℕ} -- amount of milk left after

-- Definitions based on the conditions
def milk_per_vanilla := 4
def milk_per_chocolate := 5
def ice_vanilla_per_milkshake := 12
def ice_chocolate_per_milkshake := 10
def initial_milk := 72
def initial_vanilla_ice_cream := 96
def initial_chocolate_ice_cream := 96

-- Constraints
def max_milkshakes := 16
def milk_needed (v : ℕ) := (4 * 2 * v) + (5 * v)
def vanilla_needed (v : ℕ) := 12 * 2 * v
def chocolate_needed (v : ℕ) := 10 * v 

-- Inequalities
lemma milk_constraint (v : ℕ) : milk_needed v ≤ initial_milk := sorry

lemma vanilla_constraint (v : ℕ) : vanilla_needed v ≤ initial_vanilla_ice_cream := sorry

lemma chocolate_constraint (v : ℕ) : chocolate_needed v ≤ initial_chocolate_ice_cream := sorry

lemma total_milkshakes_constraint (v : ℕ) : 3 * v ≤ max_milkshakes := sorry

-- Conclusion
theorem milk_leftover : milk_left = initial_milk - milk_needed 5 := sorry

end milk_leftover_l1724_172475


namespace intersects_x_axis_vertex_coordinates_l1724_172455

-- Definition of the quadratic function and conditions
def quadratic_function (a : ℝ) (x : ℝ) : ℝ :=
  x^2 - a * x - 2 * a^2

-- Condition: a ≠ 0
axiom a_nonzero (a : ℝ) : a ≠ 0

-- Statement for the first part of the problem
theorem intersects_x_axis (a : ℝ) (h : a ≠ 0) :
  ∃ x₁ x₂ : ℝ, quadratic_function a x₁ = 0 ∧ quadratic_function a x₂ = 0 ∧ x₁ * x₂ < 0 :=
by 
  sorry

-- Statement for the second part of the problem
theorem vertex_coordinates (a : ℝ) (h : a ≠ 0) (hy_intercept : quadratic_function a 0 = -2) :
  ∃ x_vertex : ℝ, quadratic_function a x_vertex = (if a = 1 then (1/2)^2 - 9/4 else (1/2)^2 - 9/4) :=
by 
  sorry


end intersects_x_axis_vertex_coordinates_l1724_172455


namespace remainder_when_sum_div_by_8_l1724_172433

theorem remainder_when_sum_div_by_8 (n : ℤ) : ((8 - n) + (n + 4)) % 8 = 4 := by
  sorry

end remainder_when_sum_div_by_8_l1724_172433


namespace ratio_of_cube_sides_l1724_172409

theorem ratio_of_cube_sides {a b : ℝ} (h : (6 * a^2) / (6 * b^2) = 16) : a / b = 4 :=
by
  sorry

end ratio_of_cube_sides_l1724_172409


namespace game_no_loser_l1724_172470

theorem game_no_loser (x : ℕ) (h_start : x = 2017) :
  ∀ y, (y = x ∨ ∀ n, (n = 2 * y ∨ n = y - 1000) → (n > 1000 ∧ n < 4000)) →
       (y > 1000 ∧ y < 4000) :=
sorry

end game_no_loser_l1724_172470


namespace avg_age_difference_l1724_172432

noncomputable def team_size : ℕ := 11
noncomputable def avg_age_team : ℝ := 26
noncomputable def wicket_keeper_extra_age : ℝ := 3
noncomputable def num_remaining_players : ℕ := 9
noncomputable def avg_age_remaining_players : ℝ := 23

theorem avg_age_difference :
  avg_age_team - avg_age_remaining_players = 0.33 := 
by
  sorry

end avg_age_difference_l1724_172432


namespace fourth_power_sqrt_eq_256_l1724_172404

theorem fourth_power_sqrt_eq_256 (x : ℝ) (h : (x^(1/2))^4 = 256) : x = 16 := by sorry

end fourth_power_sqrt_eq_256_l1724_172404


namespace ellipse_equation_1_ellipse_equation_2_l1724_172444

-- Proof Problem 1
theorem ellipse_equation_1 (x y : ℝ) 
  (foci_condition : (x+2) * (x+2) + y*y + (x-2) * (x-2) + y*y = 36) :
  x^2 / 9 + y^2 / 5 = 1 :=
sorry

-- Proof Problem 2
theorem ellipse_equation_2 (x y : ℝ)
  (foci_condition : (x^2 + (y+5)^2 = 0) ∧ (x^2 + (y-5)^2 = 0))
  (point_on_ellipse : 3^2 / 15 + 4^2 / (15 + 25) = 1) :
  y^2 / 40 + x^2 / 15 = 1 :=
sorry

end ellipse_equation_1_ellipse_equation_2_l1724_172444


namespace triangle_area_fraction_l1724_172437

-- Define the grid size
def grid_size : ℕ := 6

-- Define the vertices of the triangle
def vertex_A : (ℕ × ℕ) := (3, 3)
def vertex_B : (ℕ × ℕ) := (3, 5)
def vertex_C : (ℕ × ℕ) := (5, 5)

-- Define the area of the larger grid
def area_square := grid_size ^ 2

-- Compute the base and height of the triangle
def base_triangle := vertex_C.1 - vertex_B.1
def height_triangle := vertex_B.2 - vertex_A.2

-- Compute the area of the triangle
def area_triangle := (base_triangle * height_triangle) / 2

-- Define the fraction of the area of the larger square inside the triangle
def area_fraction := area_triangle / area_square

-- State the theorem
theorem triangle_area_fraction :
  area_fraction = 1 / 18 :=
by
  sorry

end triangle_area_fraction_l1724_172437


namespace scientific_notation_of_0_0000003_l1724_172496

theorem scientific_notation_of_0_0000003 :
  0.0000003 = 3 * 10^(-7) :=
sorry

end scientific_notation_of_0_0000003_l1724_172496
