import Mathlib

namespace NUMINAMATH_GPT_compare_negatives_l1556_155615

theorem compare_negatives : (-1.5 : ℝ) < (-1 + -1/5 : ℝ) :=
by 
  sorry

end NUMINAMATH_GPT_compare_negatives_l1556_155615


namespace NUMINAMATH_GPT_largest_tan_B_l1556_155693

-- The context of the problem involves a triangle with given side lengths
variables (ABC : Triangle) -- A triangle ABC

-- Define the lengths of sides AB and BC
variables (AB BC : ℝ) 
-- Define the value of tan B
variable (tanB : ℝ)

-- The given conditions
def condition_1 := AB = 25
def condition_2 := BC = 20

-- Define the actual statement we need to prove
theorem largest_tan_B (ABC : Triangle) (AB BC tanB : ℝ) : 
  AB = 25 → BC = 20 → tanB = 3 / 4 := sorry

end NUMINAMATH_GPT_largest_tan_B_l1556_155693


namespace NUMINAMATH_GPT_four_digit_number_divisible_by_11_l1556_155662

theorem four_digit_number_divisible_by_11 :
  ∃ (a b c d : ℕ), 
    a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ 
    a + b + c + d = 10 ∧ 
    (a + c) % 11 = (b + d) % 11 ∧
    (10 - a != 0 ∨ 10 - c != 0 ∨ 10 - b != 0 ∨ 10 - d != 0) := sorry

end NUMINAMATH_GPT_four_digit_number_divisible_by_11_l1556_155662


namespace NUMINAMATH_GPT_angle_measure_l1556_155684

theorem angle_measure : 
  ∃ (x : ℝ), (x + (3 * x + 3) = 90) ∧ x = 21.75 := by
  sorry

end NUMINAMATH_GPT_angle_measure_l1556_155684


namespace NUMINAMATH_GPT_min_value_a_plus_b_l1556_155614

theorem min_value_a_plus_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : Real.sqrt (3^a * 3^b) = 3^((a + b) / 2)) : a + b = 4 := by
  sorry

end NUMINAMATH_GPT_min_value_a_plus_b_l1556_155614


namespace NUMINAMATH_GPT_units_digit_sum_l1556_155624

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_sum :
  units_digit (24^3 + 17^3) = 7 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_sum_l1556_155624


namespace NUMINAMATH_GPT_intersecting_points_are_same_l1556_155677

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (3, -2)
def radius1 : ℝ := 5

def center2 : ℝ × ℝ := (3, 6)
def radius2 : ℝ := 3

-- Define the equations of the two circles
def circle1 (x y : ℝ) : Prop := (x - center1.1)^2 + (y + center1.2)^2 = radius1^2
def circle2 (x y : ℝ) : Prop := (x - center2.1)^2 + (y - center2.2)^2 = radius2^2

-- Prove that points C and D coincide
theorem intersecting_points_are_same : ∃ x y, circle1 x y ∧ circle2 x y → (0 = 0) :=
by
  sorry

end NUMINAMATH_GPT_intersecting_points_are_same_l1556_155677


namespace NUMINAMATH_GPT_max_writers_and_editors_l1556_155605

theorem max_writers_and_editors (total_people writers editors x : ℕ) (h_total_people : total_people = 100)
(h_writers : writers = 40) (h_editors : editors > 38) (h_both : 2 * x + (writers + editors - x) = total_people) :
x ≤ 21 := sorry

end NUMINAMATH_GPT_max_writers_and_editors_l1556_155605


namespace NUMINAMATH_GPT_P_started_following_J_l1556_155619

theorem P_started_following_J :
  ∀ (t : ℝ),
    (6 * 7.3 + 3 = 8 * (7.3 - t)) → t = 1.45 → t + 12 = 13.45 :=
by
  sorry

end NUMINAMATH_GPT_P_started_following_J_l1556_155619


namespace NUMINAMATH_GPT_two_std_dev_less_than_mean_l1556_155672

def mean : ℝ := 14.0
def std_dev : ℝ := 1.5

theorem two_std_dev_less_than_mean : (mean - 2 * std_dev) = 11.0 := 
by sorry

end NUMINAMATH_GPT_two_std_dev_less_than_mean_l1556_155672


namespace NUMINAMATH_GPT_area_of_rectangle_is_32_proof_l1556_155606

noncomputable def triangle_sides : ℝ := 7.3 + 5.4 + 11.3
def equality_of_perimeters (rectangle_length rectangle_width : ℝ) : Prop := 
  2 * (rectangle_length + rectangle_width) = triangle_sides

def rectangle_length (rectangle_width : ℝ) : ℝ := 2 * rectangle_width

def area_of_rectangle_is_32 (rectangle_width : ℝ) : Prop :=
  rectangle_length rectangle_width * rectangle_width = 32

theorem area_of_rectangle_is_32_proof : 
  ∃ (rectangle_width : ℝ), 
  equality_of_perimeters (rectangle_length rectangle_width) rectangle_width ∧ area_of_rectangle_is_32 rectangle_width :=
by
  sorry

end NUMINAMATH_GPT_area_of_rectangle_is_32_proof_l1556_155606


namespace NUMINAMATH_GPT_and_false_iff_not_both_true_l1556_155687

variable (p q : Prop)

theorem and_false_iff_not_both_true (h : ¬(p ∧ q)) : ¬p ∨ ¬q :=
by
    sorry

end NUMINAMATH_GPT_and_false_iff_not_both_true_l1556_155687


namespace NUMINAMATH_GPT_toy_truck_cost_is_correct_l1556_155695

-- Define the initial amount, amount spent on the pencil case, and the final amount
def initial_amount : ℝ := 10
def pencil_case_cost : ℝ := 2
def final_amount : ℝ := 5

-- Define the amount spent on the toy truck
def toy_truck_cost : ℝ := initial_amount - pencil_case_cost - final_amount

-- Prove that the amount spent on the toy truck is 3 dollars
theorem toy_truck_cost_is_correct : toy_truck_cost = 3 := by
  sorry

end NUMINAMATH_GPT_toy_truck_cost_is_correct_l1556_155695


namespace NUMINAMATH_GPT_find_m_of_parallelepiped_volume_l1556_155661

theorem find_m_of_parallelepiped_volume 
  {m : ℝ} 
  (h_pos : m > 0) 
  (h_vol : abs (3 * (m^2 - 9) - 2 * (4 * m - 15) + 2 * (12 - 5 * m)) = 20) : 
  m = (9 + Real.sqrt 249) / 6 :=
sorry

end NUMINAMATH_GPT_find_m_of_parallelepiped_volume_l1556_155661


namespace NUMINAMATH_GPT_intersection_is_empty_l1556_155635

-- Define the domain and range sets
def A : Set ℝ := {x | x < 0}
def B : Set ℝ := {x | 0 < x}

-- The Lean theorem to prove that the intersection of A and B is the empty set
theorem intersection_is_empty : A ∩ B = ∅ := by
  sorry

end NUMINAMATH_GPT_intersection_is_empty_l1556_155635


namespace NUMINAMATH_GPT_sum_of_logs_in_acute_triangle_l1556_155644

theorem sum_of_logs_in_acute_triangle (A B C : ℝ)
  (hA : 0 < A ∧ A < π / 2) (hB : 0 < B ∧ B < π / 2) (hC : 0 < C ∧ C < π / 2) 
  (h_triangle : A + B + C = π) :
  (Real.log (Real.sin B) / Real.log (Real.sin A)) +
  (Real.log (Real.sin C) / Real.log (Real.sin B)) +
  (Real.log (Real.sin A) / Real.log (Real.sin C)) ≥ 3 := by
  sorry

end NUMINAMATH_GPT_sum_of_logs_in_acute_triangle_l1556_155644


namespace NUMINAMATH_GPT_find_expression_value_l1556_155640

theorem find_expression_value (x : ℝ) (h : 4 * x^2 - 2 * x + 5 = 7) :
  2 * (x^2 - x) - (x - 1) + (2 * x + 3) = 5 := by
  sorry

end NUMINAMATH_GPT_find_expression_value_l1556_155640


namespace NUMINAMATH_GPT_range_of_m_l1556_155648

def proposition_p (m : ℝ) : Prop := (m^2 - 4 ≥ 0)
def proposition_q (m : ℝ) : Prop := (4 - 4 * m < 0)
def p_or_q (m : ℝ) : Prop := proposition_p m ∨ proposition_q m
def not_p (m : ℝ) : Prop := ¬ proposition_p m

theorem range_of_m (m : ℝ) (h1 : p_or_q m) (h2 : not_p m) : 1 < m ∧ m < 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1556_155648


namespace NUMINAMATH_GPT_extrema_f_unique_solution_F_l1556_155610

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (1 / 2) * x^2 - m * Real.log x
noncomputable def F (x : ℝ) (m : ℝ) : ℝ := - (1 / 2) * x^2 + (m + 1) * x - m * Real.log x

theorem extrema_f (m : ℝ) :
  (m ≤ 0 → ∀ x > 0, ∀ y > 0, x ≠ y → f x m ≠ f y m) ∧
  (m > 0 → ∃ x₀ > 0, ∀ x > 0, f x₀ m ≤ f x m) :=
sorry

theorem unique_solution_F (m : ℝ) (h : m ≥ 1) :
  ∃ x₀ > 0, ∀ x > 0, F x₀ m = 0 ∧ (F x m = 0 → x = x₀) :=
sorry

end NUMINAMATH_GPT_extrema_f_unique_solution_F_l1556_155610


namespace NUMINAMATH_GPT_soccer_team_selection_l1556_155618

-- Definitions of the problem
def total_members := 16
def utility_exclusion_cond := total_members - 1

-- Lean statement for the proof problem, using the conditions and answer:
theorem soccer_team_selection :
  (utility_exclusion_cond) * (total_members - 1) * (total_members - 2) * (total_members - 3) * (total_members - 4) = 409500 :=
by
  sorry

end NUMINAMATH_GPT_soccer_team_selection_l1556_155618


namespace NUMINAMATH_GPT_hyperbola_equation_l1556_155665

-- Define the conditions of the problem
def asymptotic_eq (C : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, C x y → (y = 2 * x ∨ y = -2 * x)

def passes_through_point (C : ℝ → ℝ → Prop) : Prop :=
  C 2 2

-- State the equation of the hyperbola
def is_equation_of_hyperbola (C : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, C x y ↔ x^2 / 3 - y^2 / 12 = 1

-- The theorem statement combining all conditions to prove the final equation
theorem hyperbola_equation {C : ℝ → ℝ → Prop} :
  asymptotic_eq C →
  passes_through_point C →
  is_equation_of_hyperbola C :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_equation_l1556_155665


namespace NUMINAMATH_GPT_original_square_side_length_l1556_155692

-- Defining the variables and conditions
variables (x : ℝ) (h₁ : 1.2 * x * (x - 2) = x * x)

-- Theorem statement to prove the side length of the original square is 12 cm
theorem original_square_side_length : x = 12 :=
by
  sorry

end NUMINAMATH_GPT_original_square_side_length_l1556_155692


namespace NUMINAMATH_GPT_initial_roses_l1556_155602

theorem initial_roses (x : ℕ) (h1 : x - 3 + 34 = 36) : x = 5 :=
by 
  sorry

end NUMINAMATH_GPT_initial_roses_l1556_155602


namespace NUMINAMATH_GPT_problem_statement_l1556_155660

theorem problem_statement
  (a b c : ℝ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_condition : a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0) :
  a / (b - c)^3 + b / (c - a)^3 + c / (a - b)^3 = 0 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1556_155660


namespace NUMINAMATH_GPT_sum_of_exponents_l1556_155688

-- Define the expression inside the radical
def radicand (a b c : ℝ) : ℝ := 40 * a^6 * b^3 * c^14

-- Define the simplified expression outside the radical
def simplified_expr (a b c : ℝ) : ℝ := (2 * a^2 * b * c^4)

-- State the theorem to prove the sum of the exponents of the variables outside the radical
theorem sum_of_exponents (a b c : ℝ) : 
  let exponents_sum := 2 + 1 + 4
  exponents_sum = 7 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_exponents_l1556_155688


namespace NUMINAMATH_GPT_correct_proposition_l1556_155630

theorem correct_proposition (a b : ℝ) (h : a > |b|) : a^2 > b^2 :=
sorry

end NUMINAMATH_GPT_correct_proposition_l1556_155630


namespace NUMINAMATH_GPT_find_equations_of_lines_l1556_155641

-- Define the given constants and conditions
def point_P := (2, 2)
def line_l1 (x y : ℝ) := 3 * x - 2 * y + 1 = 0
def line_l2 (x y : ℝ) := x + 3 * y + 4 = 0
def intersection_point := (-1, -1)
def slope_perpendicular_line := 3

-- The theorem that we need to prove
theorem find_equations_of_lines :
  (∀ k, k = 0 → line_l1 2 2 → (x = y ∨ x + y = 4)) ∧
  (line_l1 (-1) (-1) ∧ line_l2 (-1) (-1) →
   (3 * x - y + 2 = 0))
:=
sorry

end NUMINAMATH_GPT_find_equations_of_lines_l1556_155641


namespace NUMINAMATH_GPT_count_squares_containing_A_l1556_155612

-- Given conditions
def figure_with_squares : Prop := ∃ n : ℕ, n = 20

-- The goal is to prove that the number of squares containing A is 13
theorem count_squares_containing_A (h : figure_with_squares) : ∃ k : ℕ, k = 13 :=
by 
  sorry

end NUMINAMATH_GPT_count_squares_containing_A_l1556_155612


namespace NUMINAMATH_GPT_fixed_point_of_function_l1556_155679

variable (a : ℝ)

noncomputable def f (x : ℝ) : ℝ := a^(1 - x) - 2

theorem fixed_point_of_function (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) : f a 1 = -1 := by
  sorry

end NUMINAMATH_GPT_fixed_point_of_function_l1556_155679


namespace NUMINAMATH_GPT_contractor_initial_people_l1556_155690

theorem contractor_initial_people (P : ℕ) (days_total days_done : ℕ) 
  (percent_done : ℚ) (additional_people : ℕ) (T : ℕ) :
  days_total = 50 →
  days_done = 25 →
  percent_done = 0.4 →
  additional_people = 90 →
  T = P + additional_people →
  (P : ℚ) * 62.5 = (T : ℚ) * 50 →
  P = 360 :=
by
  intros h_days_total h_days_done h_percent_done h_additional_people h_T h_eq
  sorry

end NUMINAMATH_GPT_contractor_initial_people_l1556_155690


namespace NUMINAMATH_GPT_circle_area_isosceles_triangle_l1556_155601

theorem circle_area_isosceles_triangle (a b c : ℝ) (h₁ : a = 3) (h₂ : b = 3) (h₃ : c = 2) :
  ∃ R : ℝ, R = (81 / 32) * Real.pi :=
by sorry

end NUMINAMATH_GPT_circle_area_isosceles_triangle_l1556_155601


namespace NUMINAMATH_GPT_athletes_in_camp_hours_l1556_155669

theorem athletes_in_camp_hours (initial_athletes : ℕ) (left_rate : ℕ) (left_hours : ℕ) (arrived_rate : ℕ) 
  (difference : ℕ) (hours : ℕ) 
  (h_initial: initial_athletes = 300) 
  (h_left_rate: left_rate = 28) 
  (h_left_hours: left_hours = 4) 
  (h_arrived_rate: arrived_rate = 15) 
  (h_difference: difference = 7) 
  (h_left: left_rate * left_hours = 112) 
  (h_equation: initial_athletes - (left_rate * left_hours) + (arrived_rate * hours) = initial_athletes - difference) : 
  hours = 7 :=
by
  sorry

end NUMINAMATH_GPT_athletes_in_camp_hours_l1556_155669


namespace NUMINAMATH_GPT_expr_eval_l1556_155627

def expr : ℕ := 3 * 3^4 - 9^27 / 9^25

theorem expr_eval : expr = 162 := by
  -- Proof will be written here if needed
  sorry

end NUMINAMATH_GPT_expr_eval_l1556_155627


namespace NUMINAMATH_GPT_max_abs_x2_is_2_l1556_155626

noncomputable def max_abs_x2 {x₁ x₂ x₃ : ℝ} (h : x₁^2 + x₂^2 + x₃^2 + x₁ * x₂ + x₂ * x₃ = 2) : ℝ :=
2

theorem max_abs_x2_is_2 {x₁ x₂ x₃ : ℝ} (h : x₁^2 + x₂^2 + x₃^2 + x₁ * x₂ + x₂ * x₃ = 2) :
  max_abs_x2 h = 2 := 
sorry

end NUMINAMATH_GPT_max_abs_x2_is_2_l1556_155626


namespace NUMINAMATH_GPT_calculation_correct_l1556_155670

noncomputable def problem_calculation : ℝ :=
  4 * Real.sin (Real.pi / 3) - abs (-1) + (Real.sqrt 3 - 1)^0 + Real.sqrt 48

theorem calculation_correct : problem_calculation = 6 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_calculation_correct_l1556_155670


namespace NUMINAMATH_GPT_neg_and_eq_or_not_l1556_155617

theorem neg_and_eq_or_not (p q : Prop) : ¬(p ∧ q) ↔ ¬p ∨ ¬q :=
by sorry

end NUMINAMATH_GPT_neg_and_eq_or_not_l1556_155617


namespace NUMINAMATH_GPT_number_of_goats_l1556_155608

theorem number_of_goats (C G : ℕ) 
  (h1 : C = 2) 
  (h2 : ∀ G : ℕ, 460 * C + 60 * G = 1400) 
  (h3 : 460 = 460) 
  (h4 : 60 = 60) : 
  G = 8 :=
by
  sorry

end NUMINAMATH_GPT_number_of_goats_l1556_155608


namespace NUMINAMATH_GPT_geometric_progression_fraction_l1556_155637

theorem geometric_progression_fraction (a₁ a₂ a₃ a₄ : ℝ) (h1 : a₂ = 2 * a₁) (h2 : a₃ = 2 * a₂) (h3 : a₄ = 2 * a₃) : 
  (2 * a₁ + a₂) / (2 * a₃ + a₄) = 1 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_geometric_progression_fraction_l1556_155637


namespace NUMINAMATH_GPT_min_value_M_l1556_155657

theorem min_value_M (a b c : ℝ) (h1 : a < b) (h2 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0): 
  ∃ M : ℝ, M = 8 ∧ M = (a + 2 * b + 4 * c) / (b - a) :=
sorry

end NUMINAMATH_GPT_min_value_M_l1556_155657


namespace NUMINAMATH_GPT_concert_duration_l1556_155629

def duration_in_minutes (hours : Int) (extra_minutes : Int) : Int :=
  hours * 60 + extra_minutes

theorem concert_duration : duration_in_minutes 7 45 = 465 :=
by
  sorry

end NUMINAMATH_GPT_concert_duration_l1556_155629


namespace NUMINAMATH_GPT_avg_salary_rest_of_workers_l1556_155638

theorem avg_salary_rest_of_workers (avg_all : ℝ) (avg_technicians : ℝ) (total_workers : ℕ) (n_technicians : ℕ) (avg_rest : ℝ) :
  avg_all = 8000 ∧ avg_technicians = 20000 ∧ total_workers = 49 ∧ n_technicians = 7 →
  avg_rest = 6000 :=
by
  sorry

end NUMINAMATH_GPT_avg_salary_rest_of_workers_l1556_155638


namespace NUMINAMATH_GPT_simplify_expression_l1556_155651

def expr_initial (y : ℝ) := 3*y + 4*y^2 + 2 - (7 - 3*y - 4*y^2)
def expr_simplified (y : ℝ) := 8*y^2 + 6*y - 5

theorem simplify_expression (y : ℝ) : expr_initial y = expr_simplified y :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1556_155651


namespace NUMINAMATH_GPT_defective_and_shipped_percent_l1556_155659

def defective_percent : ℝ := 0.05
def shipped_percent : ℝ := 0.04

theorem defective_and_shipped_percent : (defective_percent * shipped_percent) * 100 = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_defective_and_shipped_percent_l1556_155659


namespace NUMINAMATH_GPT_simplify_expression_l1556_155647

theorem simplify_expression : ((1 + 2 + 3 + 4 + 5 + 6) / 3 + (3 * 5 + 12) / 4) = 13.75 :=
by
-- Proof steps would go here, but we replace them with 'sorry' for now.
sorry

end NUMINAMATH_GPT_simplify_expression_l1556_155647


namespace NUMINAMATH_GPT_distance_to_origin_l1556_155642

theorem distance_to_origin (x y : ℤ) (hx : x = -5) (hy : y = 12) :
  Real.sqrt (x^2 + y^2) = 13 := by
  rw [hx, hy]
  norm_num
  sorry

end NUMINAMATH_GPT_distance_to_origin_l1556_155642


namespace NUMINAMATH_GPT_number_of_integers_with_three_divisors_l1556_155628

def has_exactly_three_positive_divisors (n : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ n = p * p

theorem number_of_integers_with_three_divisors (n : ℕ) :
  n = 2012 → Nat.card { x : ℕ | x ≤ n ∧ has_exactly_three_positive_divisors x } = 14 :=
by
  sorry

end NUMINAMATH_GPT_number_of_integers_with_three_divisors_l1556_155628


namespace NUMINAMATH_GPT_series_sum_half_l1556_155671

theorem series_sum_half :
  ∑' n : ℕ, (2^n / (3^(2^n) + 1)) = 1/2 := 
sorry

end NUMINAMATH_GPT_series_sum_half_l1556_155671


namespace NUMINAMATH_GPT_range_of_y_div_x_l1556_155683

theorem range_of_y_div_x (x y : ℝ) (h : x^2 + (y-3)^2 = 1) : 
  (∃ k : ℝ, k = y / x ∧ (k ≤ -2 * Real.sqrt 2 ∨ k ≥ 2 * Real.sqrt 2)) :=
sorry

end NUMINAMATH_GPT_range_of_y_div_x_l1556_155683


namespace NUMINAMATH_GPT_intersection_A_B_l1556_155620

def A : Set ℝ := { x | -2 < x ∧ x < 2 }
def B : Set ℝ := { x | 1 < x ∧ x < 3 }

theorem intersection_A_B : A ∩ B = { x | 1 < x ∧ x < 2 } := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1556_155620


namespace NUMINAMATH_GPT_minimum_value_f_l1556_155636

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 1)

theorem minimum_value_f (x : ℝ) (h : x > 1) : (∃ y, (f y = 3) ∧ ∀ z, z > 1 → f z ≥ 3) :=
by sorry

end NUMINAMATH_GPT_minimum_value_f_l1556_155636


namespace NUMINAMATH_GPT_depth_of_water_l1556_155681

variable (RonHeight DepthOfWater : ℝ)

-- Definitions based on conditions
def RonStandingHeight := 12 -- Ron's height is 12 feet
def DepthOfWaterCalculation := 5 * RonStandingHeight -- Depth is 5 times Ron's height

-- Theorem statement to prove
theorem depth_of_water (hRon : RonHeight = RonStandingHeight) (hDepth : DepthOfWater = DepthOfWaterCalculation) :
  DepthOfWater = 60 := by
  sorry

end NUMINAMATH_GPT_depth_of_water_l1556_155681


namespace NUMINAMATH_GPT_find_other_root_l1556_155625

theorem find_other_root (a b c x : ℝ) (h₁ : a ≠ 0) 
  (h₂ : b ≠ 0) (h₃ : c ≠ 0)
  (h₄ : a * (b + 2 * c) * x^2 + b * (2 * c - a) * x + c * (2 * a - b) = 0)
  (h₅ : a * (b + 2 * c) - b * (2 * c - a) + c * (2 * a - b) = 0) :
  ∃ y : ℝ, y = - (c * (2 * a - b)) / (a * (b + 2 * c)) :=
sorry

end NUMINAMATH_GPT_find_other_root_l1556_155625


namespace NUMINAMATH_GPT_find_b_l1556_155658

theorem find_b (b : ℝ) (tangent_condition : ∀ x y : ℝ, y = -2 * x + b → y^2 = 8 * x) : b = -1 :=
sorry

end NUMINAMATH_GPT_find_b_l1556_155658


namespace NUMINAMATH_GPT_baseball_card_count_l1556_155667

-- Define initial conditions
def initial_cards := 15

-- Maria takes half of one more than the number of initial cards
def maria_takes := (initial_cards + 1) / 2

-- Remaining cards after Maria takes her share
def remaining_after_maria := initial_cards - maria_takes

-- You give Peter 1 card
def remaining_after_peter := remaining_after_maria - 1

-- Paul triples the remaining cards
def final_cards := remaining_after_peter * 3

-- Theorem statement to prove
theorem baseball_card_count :
  final_cards = 18 := by
sorry

end NUMINAMATH_GPT_baseball_card_count_l1556_155667


namespace NUMINAMATH_GPT_jen_age_proof_l1556_155656

-- Definitions
def son_age := 16
def son_present_age := son_age
def jen_present_age := 41

-- Conditions
axiom jen_older_25 (x : ℕ) : ∀ y : ℕ, x = y + 25 → y = son_present_age
axiom jen_age_formula (j s : ℕ) : j = 3 * s - 7 → j = son_present_age + 25

-- Proof problem statement
theorem jen_age_proof : jen_present_age = 41 :=
by
  -- Declare variables
  let j := jen_present_age
  let s := son_present_age
  -- Apply conditions (in Lean, sorry will skip the proof)
  sorry

end NUMINAMATH_GPT_jen_age_proof_l1556_155656


namespace NUMINAMATH_GPT_product_of_axes_l1556_155663

-- Definitions based on conditions
def ellipse (a b : ℝ) : Prop :=
  a^2 - b^2 = 64

def triangle_incircle_diameter (a b : ℝ) : Prop :=
  b + 8 - a = 4

-- Proving that (AB)(CD) = 240
theorem product_of_axes (a b : ℝ) (h₁ : ellipse a b) (h₂ : triangle_incircle_diameter a b) : 
  (2 * a) * (2 * b) = 240 :=
by
  sorry

end NUMINAMATH_GPT_product_of_axes_l1556_155663


namespace NUMINAMATH_GPT_simplify_expression_l1556_155686

theorem simplify_expression (x : ℝ) : (5 * x + 2 * x + 7 * x) = 14 * x :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1556_155686


namespace NUMINAMATH_GPT_number_of_B_students_l1556_155633

/-- Let x be the number of students who earn a B. 
    Given the conditions:
    - The number of students who earn an A is 0.5x.
    - The number of students who earn a C is 2x.
    - The number of students who earn a D is 0.3x.
    - The total number of students in the class is 40.
    Prove the number of students who earn a B is 40 / 3.8 = 200 / 19, approximately 11. -/
theorem number_of_B_students (x : ℝ) (h_bA: x * 0.5 + x + x * 2 + x * 0.3 = 40) : 
  x = 40 / 3.8 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_B_students_l1556_155633


namespace NUMINAMATH_GPT_age_of_B_l1556_155652

variable (A B C : ℕ)

theorem age_of_B (h1 : A + B + C = 84) (h2 : A + C = 58) : B = 26 := by
  sorry

end NUMINAMATH_GPT_age_of_B_l1556_155652


namespace NUMINAMATH_GPT_shared_friends_count_l1556_155613

theorem shared_friends_count (james_friends : ℕ) (total_combined : ℕ) (john_factor : ℕ) 
  (h1 : james_friends = 75) 
  (h2 : john_factor = 3) 
  (h3 : total_combined = 275) : 
  james_friends + (john_factor * james_friends) - total_combined = 25 := 
by
  sorry

end NUMINAMATH_GPT_shared_friends_count_l1556_155613


namespace NUMINAMATH_GPT_total_candies_l1556_155604

theorem total_candies (n p r : ℕ) (H1 : n = 157) (H2 : p = 235) (H3 : r = 98) :
  n * p + r = 36993 := by
  sorry

end NUMINAMATH_GPT_total_candies_l1556_155604


namespace NUMINAMATH_GPT_icosahedron_path_count_l1556_155643

noncomputable def icosahedron_paths : ℕ := 
  sorry

theorem icosahedron_path_count : icosahedron_paths = 45 :=
  sorry

end NUMINAMATH_GPT_icosahedron_path_count_l1556_155643


namespace NUMINAMATH_GPT_lucille_house_difference_l1556_155698

-- Define the heights of the houses as given in the conditions.
def height_lucille : ℕ := 80
def height_neighbor1 : ℕ := 70
def height_neighbor2 : ℕ := 99

-- Define the total height of the houses.
def total_height : ℕ := height_neighbor1 + height_lucille + height_neighbor2

-- Define the average height of the houses.
def average_height : ℕ := total_height / 3

-- Define the height difference between Lucille's house and the average height.
def height_difference : ℕ := average_height - height_lucille

-- The theorem to prove.
theorem lucille_house_difference :
  height_difference = 3 := by
  sorry

end NUMINAMATH_GPT_lucille_house_difference_l1556_155698


namespace NUMINAMATH_GPT_solution_set_of_x_squared_geq_four_l1556_155680

theorem solution_set_of_x_squared_geq_four :
  {x : ℝ | x^2 ≥ 4} = {x : ℝ | x ≤ -2 ∨ x ≥ 2} :=
sorry

end NUMINAMATH_GPT_solution_set_of_x_squared_geq_four_l1556_155680


namespace NUMINAMATH_GPT_four_edge_trips_count_l1556_155609

-- Defining points and edges of the cube
inductive Point
| A | B | C | D | E | F | G | H

open Point

-- Edges of the cube are connections between points
def Edge (p1 p2 : Point) : Prop :=
  ∃ (edges : List (Point × Point)), 
    edges = [(A, B), (A, D), (A, E), (B, C), (B, E), (B, F), (C, D), (C, F), (C, G), (D, E), (D, F), (D, H), (E, F), (E, H), (F, G), (F, H), (G, H)] ∧ 
    ((p1, p2) ∈ edges ∨ (p2, p1) ∈ edges)

-- Define the proof statement
theorem four_edge_trips_count : 
  ∃ (num_paths : ℕ), num_paths = 12 :=
sorry

end NUMINAMATH_GPT_four_edge_trips_count_l1556_155609


namespace NUMINAMATH_GPT_Emily_sixth_score_l1556_155654

theorem Emily_sixth_score :
  let scores := [91, 94, 88, 90, 101]
  let current_sum := scores.sum
  let desired_average := 95
  let num_quizzes := 6
  let total_score_needed := num_quizzes * desired_average
  let sixth_score := total_score_needed - current_sum
  sixth_score = 106 :=
by
  sorry

end NUMINAMATH_GPT_Emily_sixth_score_l1556_155654


namespace NUMINAMATH_GPT_pet_store_dogs_l1556_155616

theorem pet_store_dogs (cats dogs : ℕ) (h1 : 18 = cats) (h2 : 3 * dogs = 4 * cats) : dogs = 24 :=
by
  sorry

end NUMINAMATH_GPT_pet_store_dogs_l1556_155616


namespace NUMINAMATH_GPT_fiftieth_statement_l1556_155632

-- Define the types
inductive Inhabitant : Type
| knight : Inhabitant
| liar : Inhabitant

-- Define the function telling the statement
def statement (inhabitant : Inhabitant) : String :=
  match inhabitant with
  | Inhabitant.knight => "Knight"
  | Inhabitant.liar => "Liar"

-- Define the condition: knights tell the truth and liars lie
def tells_truth (inhabitant : Inhabitant) (statement_about_neighbor : String) : Prop :=
  match inhabitant with
  | Inhabitant.knight => statement_about_neighbor = "Knight"
  | Inhabitant.liar => statement_about_neighbor ≠ "Knight"

-- Define a function that determines what each inhabitant says about their right-hand neighbor
def what_they_say (idx : ℕ) : String :=
  if idx % 2 = 0 then "Liar" else "Knight"

-- Define the inhabitant pattern
def inhabitant_at (idx : ℕ) : Inhabitant :=
  if idx % 2 = 0 then Inhabitant.liar else Inhabitant.knight

-- The main theorem statement
theorem fiftieth_statement : tells_truth (inhabitant_at 49) (what_they_say 50) :=
by 
  -- This proof outlines the theorem statement only
  sorry

end NUMINAMATH_GPT_fiftieth_statement_l1556_155632


namespace NUMINAMATH_GPT_factorize_x_squared_minus_4_factorize_2mx_squared_minus_4mx_plus_2m_factorize_y_quad_l1556_155650

-- Problem 1
theorem factorize_x_squared_minus_4 (x : ℝ) :
  x^2 - 4 = (x + 2) * (x - 2) :=
by { 
  sorry
}

-- Problem 2
theorem factorize_2mx_squared_minus_4mx_plus_2m (x m : ℝ) :
  2 * m * x^2 - 4 * m * x + 2 * m = 2 * m * (x - 1)^2 :=
by { 
  sorry
}

-- Problem 3
theorem factorize_y_quad (y : ℝ) :
  (y^2 - 1)^2 - 6 * (y^2 - 1) + 9 = (y + 2)^2 * (y - 2)^2 :=
by { 
  sorry
}

end NUMINAMATH_GPT_factorize_x_squared_minus_4_factorize_2mx_squared_minus_4mx_plus_2m_factorize_y_quad_l1556_155650


namespace NUMINAMATH_GPT_gcd_sequence_terms_l1556_155649

theorem gcd_sequence_terms (d m : ℕ) (hd : d > 1) (hm : m > 0) :
    ∃ k l : ℕ, k ≠ l ∧ gcd (2 ^ (2 ^ k) + d) (2 ^ (2 ^ l) + d) > m := 
sorry

end NUMINAMATH_GPT_gcd_sequence_terms_l1556_155649


namespace NUMINAMATH_GPT_least_subtraction_to_divisible_by_prime_l1556_155622

theorem least_subtraction_to_divisible_by_prime :
  ∃ k : ℕ, (k = 46) ∧ (856324 - k) % 101 = 0 :=
by
  sorry

end NUMINAMATH_GPT_least_subtraction_to_divisible_by_prime_l1556_155622


namespace NUMINAMATH_GPT_right_triangle_power_inequality_l1556_155621

theorem right_triangle_power_inequality {a b c x : ℝ} (hpos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a^2 = b^2 + c^2) (h_longest : a > b ∧ a > c) :
  (x > 2) → (a^x > b^x + c^x) :=
by sorry

end NUMINAMATH_GPT_right_triangle_power_inequality_l1556_155621


namespace NUMINAMATH_GPT_side_length_square_eq_4_l1556_155676

theorem side_length_square_eq_4 (s : ℝ) (h : s^2 - 3 * s = 4) : s = 4 :=
sorry

end NUMINAMATH_GPT_side_length_square_eq_4_l1556_155676


namespace NUMINAMATH_GPT_simplify_expression_l1556_155655

variable (x y z : ℝ)

theorem simplify_expression (hxz : x > z) (hzy : z > y) (hy0 : y > 0) :
  (x^z * z^y * y^x) / (z^z * y^y * x^x) = x^(z-x) * z^(y-z) * y^(x-y) :=
sorry

end NUMINAMATH_GPT_simplify_expression_l1556_155655


namespace NUMINAMATH_GPT_polygon_is_hexagon_l1556_155682

-- Definitions
def side_length : ℝ := 8
def perimeter : ℝ := 48

-- The main theorem to prove
theorem polygon_is_hexagon : (perimeter / side_length = 6) ∧ (48 / 8 = 6) := 
by
  sorry

end NUMINAMATH_GPT_polygon_is_hexagon_l1556_155682


namespace NUMINAMATH_GPT_subway_boarding_probability_l1556_155664

theorem subway_boarding_probability :
  ∀ (total_interval boarding_interval : ℕ),
  total_interval = 10 →
  boarding_interval = 1 →
  (boarding_interval : ℚ) / total_interval = 1 / 10 := by
  intros total_interval boarding_interval ht hb
  rw [hb, ht]
  norm_num

end NUMINAMATH_GPT_subway_boarding_probability_l1556_155664


namespace NUMINAMATH_GPT_ab_value_l1556_155691

theorem ab_value (a b : ℕ) (ha : a > 0) (hb : b > 0) (h : a^2 + 3 * b = 33) : a * b = 24 := 
by 
  sorry

end NUMINAMATH_GPT_ab_value_l1556_155691


namespace NUMINAMATH_GPT_clusters_per_spoonful_l1556_155699

theorem clusters_per_spoonful (spoonfuls_per_bowl : ℕ) (clusters_per_box : ℕ) (bowls_per_box : ℕ) 
  (h_spoonfuls : spoonfuls_per_bowl = 25) 
  (h_clusters : clusters_per_box = 500)
  (h_bowls : bowls_per_box = 5) : 
  clusters_per_box / bowls_per_box / spoonfuls_per_bowl = 4 := 
by 
  have clusters_per_bowl := clusters_per_box / bowls_per_box
  have clusters_per_spoonful := clusters_per_bowl / spoonfuls_per_bowl
  sorry

end NUMINAMATH_GPT_clusters_per_spoonful_l1556_155699


namespace NUMINAMATH_GPT_total_amount_after_refunds_l1556_155634

def individual_bookings : ℕ := 12000
def group_bookings : ℕ := 16000
def refunds : ℕ := 1600

theorem total_amount_after_refunds : 
  individual_bookings + group_bookings - refunds = 26400 :=
by 
  -- The proof goes here.
  sorry

end NUMINAMATH_GPT_total_amount_after_refunds_l1556_155634


namespace NUMINAMATH_GPT_area_of_parallelogram_l1556_155653

theorem area_of_parallelogram
  (angle_deg : ℝ := 150)
  (side1 : ℝ := 10)
  (side2 : ℝ := 20)
  (adj_angle_deg : ℝ := 180 - angle_deg)
  (angle_rad : ℝ := (adj_angle_deg * Real.pi) / 180) :
  let height := side1 * (Real.sqrt 3 / 2)
  let area := side2 * height
  area = 100 * Real.sqrt 3 :=
by
  /- Proof skipped -/
  sorry

end NUMINAMATH_GPT_area_of_parallelogram_l1556_155653


namespace NUMINAMATH_GPT_percent_area_shaded_l1556_155646

-- Conditions: Square $ABCD$ has a side length of 10, and square $PQRS$ has a side length of 15.
-- The overlap of these squares forms a rectangle $AQRD$ with dimensions $20 \times 25$.

theorem percent_area_shaded 
  (side_ABCD : ℕ := 10) 
  (side_PQRS : ℕ := 15) 
  (dim_AQRD_length : ℕ := 25) 
  (dim_AQRD_width : ℕ := 20) 
  (area_AQRD : ℕ := dim_AQRD_length * dim_AQRD_width)
  (overlap_side : ℕ := 10) 
  (area_shaded : ℕ := overlap_side * overlap_side)
  : (area_shaded * 100) / area_AQRD = 20 := 
by 
  sorry

end NUMINAMATH_GPT_percent_area_shaded_l1556_155646


namespace NUMINAMATH_GPT_constant_term_binomial_expansion_l1556_155607

theorem constant_term_binomial_expansion (a : ℝ) (h : 15 * a^2 = 120) : a = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_constant_term_binomial_expansion_l1556_155607


namespace NUMINAMATH_GPT_buoy_radius_l1556_155623

-- Define the conditions based on the given problem
def is_buoy_hole (width : ℝ) (depth : ℝ) : Prop :=
  width = 30 ∧ depth = 10

-- Define the statement to prove the radius of the buoy
theorem buoy_radius : ∀ r x : ℝ, is_buoy_hole 30 10 → (x^2 + 225 = (x + 10)^2) → r = x + 10 → r = 16.25 := by
  intros r x h_cond h_eq h_add
  sorry

end NUMINAMATH_GPT_buoy_radius_l1556_155623


namespace NUMINAMATH_GPT_number_of_avocados_l1556_155600

-- Constants for the given problem
def banana_cost : ℕ := 1
def apple_cost : ℕ := 2
def strawberry_cost_per_12 : ℕ := 4
def avocado_cost : ℕ := 3
def grape_cost_half_bunch : ℕ := 2
def total_cost : ℤ := 28

-- Quantities of the given fruits
def banana_qty : ℕ := 4
def apple_qty : ℕ := 3
def strawberry_qty : ℕ := 24
def grape_qty_full_bunch_cost : ℕ := 4 -- since half bunch cost $2, full bunch cost $4

-- Definition to calculate the cost of the known fruits
def known_fruit_cost : ℤ :=
  (banana_qty * banana_cost) +
  (apple_qty * apple_cost) +
  (strawberry_qty / 12 * strawberry_cost_per_12) +
  grape_qty_full_bunch_cost

-- The cost of avocados needed to fill the total cost
def avocado_cost_needed : ℤ := total_cost - known_fruit_cost

-- Finally, we need to prove that the number of avocados is 2
theorem number_of_avocados (n : ℕ) : n * avocado_cost = avocado_cost_needed → n = 2 :=
by
  -- Problem data
  have h_banana : ℕ := banana_qty * banana_cost
  have h_apple : ℕ := apple_qty * apple_cost
  have h_strawberry : ℕ := (strawberry_qty / 12) * strawberry_cost_per_12
  have h_grape : ℕ := grape_qty_full_bunch_cost
  have h_known : ℕ := h_banana + h_apple + h_strawberry + h_grape
  
  -- Calculation for number of avocados
  have h_avocado : ℤ := total_cost - h_known
  
  -- Proving number of avocados
  sorry

end NUMINAMATH_GPT_number_of_avocados_l1556_155600


namespace NUMINAMATH_GPT_Xiaogang_raised_arm_exceeds_head_l1556_155668

theorem Xiaogang_raised_arm_exceeds_head :
  ∀ (height shadow_no_arm shadow_with_arm : ℝ),
    height = 1.7 → shadow_no_arm = 0.85 → shadow_with_arm = 1.1 →
    (height / shadow_no_arm) = ((shadow_with_arm - shadow_no_arm) * (height / shadow_no_arm)) →
    shadow_with_arm - shadow_no_arm = 0.25 →
    ((height / shadow_no_arm) * 0.25) = 0.5 :=
by
  intros height shadow_no_arm shadow_with_arm h_eq1 h_eq2 h_eq3 h_eq4 h_eq5
  sorry

end NUMINAMATH_GPT_Xiaogang_raised_arm_exceeds_head_l1556_155668


namespace NUMINAMATH_GPT_sunflower_mix_is_50_percent_l1556_155696

-- Define the proportions and percentages given in the problem
def prop_A : ℝ := 0.60 -- 60% of the mix is Brand A
def prop_B : ℝ := 0.40 -- 40% of the mix is Brand B
def sf_A : ℝ := 0.60 -- Brand A is 60% sunflower
def sf_B : ℝ := 0.35 -- Brand B is 35% sunflower

-- Define the final percentage of sunflower in the mix
noncomputable def sunflower_mix_percentage : ℝ :=
  (sf_A * prop_A) + (sf_B * prop_B)

-- Statement to prove that the percentage of sunflower in the mix is 50%
theorem sunflower_mix_is_50_percent : sunflower_mix_percentage = 0.50 :=
by
  sorry

end NUMINAMATH_GPT_sunflower_mix_is_50_percent_l1556_155696


namespace NUMINAMATH_GPT_area_excluding_garden_proof_l1556_155611

noncomputable def area_land_excluding_garden (length width r : ℝ) : ℝ :=
  let area_rec := length * width
  let area_circle := Real.pi * (r ^ 2)
  area_rec - area_circle

theorem area_excluding_garden_proof :
  area_land_excluding_garden 8 12 3 = 96 - 9 * Real.pi :=
by
  unfold area_land_excluding_garden
  sorry

end NUMINAMATH_GPT_area_excluding_garden_proof_l1556_155611


namespace NUMINAMATH_GPT_fibby_numbers_l1556_155673

def is_fibby (k : ℕ) : Prop :=
  k ≥ 3 ∧ ∃ (n : ℕ) (d : ℕ → ℕ),
  (∀ j, 1 ≤ j ∧ j ≤ k - 2 → d (j + 2) = d (j + 1) + d j) ∧
  (∀ (j : ℕ), 1 ≤ j ∧ j ≤ k → d j ∣ n) ∧
  (∀ (m : ℕ), m ∣ n → m < d 1 ∨ m > d k)

theorem fibby_numbers : ∀ (k : ℕ), is_fibby k → k = 3 ∨ k = 4 :=
sorry

end NUMINAMATH_GPT_fibby_numbers_l1556_155673


namespace NUMINAMATH_GPT_Tricia_is_five_years_old_l1556_155603

noncomputable def Vincent_age : ℕ := 22
noncomputable def Rupert_age : ℕ := Vincent_age - 2
noncomputable def Khloe_age : ℕ := Rupert_age - 10
noncomputable def Eugene_age : ℕ := 3 * Khloe_age
noncomputable def Yorick_age : ℕ := 2 * Eugene_age
noncomputable def Amilia_age : ℕ := Yorick_age / 4
noncomputable def Tricia_age : ℕ := Amilia_age / 3

theorem Tricia_is_five_years_old : Tricia_age = 5 := by
  unfold Tricia_age Amilia_age Yorick_age Eugene_age Khloe_age Rupert_age Vincent_age
  sorry

end NUMINAMATH_GPT_Tricia_is_five_years_old_l1556_155603


namespace NUMINAMATH_GPT_inequality_proof_l1556_155645

theorem inequality_proof 
  {x₁ x₂ x₃ x₄ x₅ x₆ : ℝ} (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0) (h₆ : x₆ > 0) :
  (x₂ / x₁)^5 + (x₄ / x₂)^5 + (x₆ / x₃)^5 + (x₁ / x₄)^5 + (x₃ / x₅)^5 + (x₅ / x₆)^5 ≥ 
  (x₁ / x₂) + (x₂ / x₄) + (x₃ / x₆) + (x₄ / x₁) + (x₅ / x₃) + (x₆ / x₅) := 
  sorry

end NUMINAMATH_GPT_inequality_proof_l1556_155645


namespace NUMINAMATH_GPT_least_four_digit_integer_has_3_7_11_as_factors_l1556_155694

theorem least_four_digit_integer_has_3_7_11_as_factors :
  ∃ x : ℕ, (1000 ≤ x ∧ x < 10000) ∧ (3 ∣ x) ∧ (7 ∣ x) ∧ (11 ∣ x) ∧ x = 1155 := by
  sorry

end NUMINAMATH_GPT_least_four_digit_integer_has_3_7_11_as_factors_l1556_155694


namespace NUMINAMATH_GPT_right_rectangular_prism_volume_l1556_155675

theorem right_rectangular_prism_volume
    (a b c : ℝ)
    (H1 : a * b = 56)
    (H2 : b * c = 63)
    (H3 : a * c = 72)
    (H4 : c = 3 * a) :
    a * b * c = 2016 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_right_rectangular_prism_volume_l1556_155675


namespace NUMINAMATH_GPT_square_points_sum_of_squares_l1556_155639

theorem square_points_sum_of_squares 
  (a b c d : ℝ) 
  (h₀_a : 0 ≤ a ∧ a ≤ 1)
  (h₀_b : 0 ≤ b ∧ b ≤ 1)
  (h₀_c : 0 ≤ c ∧ c ≤ 1)
  (h₀_d : 0 ≤ d ∧ d ≤ 1) 
  :
  2 ≤ a^2 + (1 - d)^2 + b^2 + (1 - a)^2 + c^2 + (1 - b)^2 + d^2 + (1 - c)^2 ∧
  a^2 + (1 - d)^2 + b^2 + (1 - a)^2 + c^2 + (1 - b)^2 + d^2 + (1 - c)^2 ≤ 4 := 
by
  sorry

end NUMINAMATH_GPT_square_points_sum_of_squares_l1556_155639


namespace NUMINAMATH_GPT_cell_division_50_closest_to_10_15_l1556_155689

theorem cell_division_50_closest_to_10_15 :
  10^14 < 2^50 ∧ 2^50 < 10^16 :=
sorry

end NUMINAMATH_GPT_cell_division_50_closest_to_10_15_l1556_155689


namespace NUMINAMATH_GPT_largest_of_numbers_l1556_155674

theorem largest_of_numbers (a b c d : ℝ) 
  (ha : a = 0) (hb : b = -1) (hc : c = 3.5) (hd : d = Real.sqrt 13) : 
  ∃ x, x = Real.sqrt 13 ∧ (x > a) ∧ (x > b) ∧ (x > c) ∧ (x > d) :=
by
  sorry

end NUMINAMATH_GPT_largest_of_numbers_l1556_155674


namespace NUMINAMATH_GPT_fourth_group_students_l1556_155678

theorem fourth_group_students (total_students group1 group2 group3 group4 : ℕ)
  (h_total : total_students = 24)
  (h_group1 : group1 = 5)
  (h_group2 : group2 = 8)
  (h_group3 : group3 = 7)
  (h_groups_sum : group1 + group2 + group3 + group4 = total_students) :
  group4 = 4 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_fourth_group_students_l1556_155678


namespace NUMINAMATH_GPT_total_revenue_correct_l1556_155685

def original_price_sneakers : ℝ := 80
def discount_sneakers : ℝ := 0.25
def pairs_sneakers_sold : ℕ := 2

def original_price_sandals : ℝ := 60
def discount_sandals : ℝ := 0.35
def pairs_sandals_sold : ℕ := 4

def original_price_boots : ℝ := 120
def discount_boots : ℝ := 0.40
def pairs_boots_sold : ℕ := 11

def calculate_total_revenue : ℝ := 
  let revenue_sneakers := pairs_sneakers_sold * (original_price_sneakers * (1 - discount_sneakers))
  let revenue_sandals := pairs_sandals_sold * (original_price_sandals * (1 - discount_sandals))
  let revenue_boots := pairs_boots_sold * (original_price_boots * (1 - discount_boots))
  revenue_sneakers + revenue_sandals + revenue_boots

theorem total_revenue_correct : calculate_total_revenue = 1068 := by
  sorry

end NUMINAMATH_GPT_total_revenue_correct_l1556_155685


namespace NUMINAMATH_GPT_seats_capacity_l1556_155666

theorem seats_capacity (x : ℕ) (h1 : 15 * x + 12 * x + 8 = 89) : x = 3 :=
by
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_seats_capacity_l1556_155666


namespace NUMINAMATH_GPT_parallel_planes_transitivity_l1556_155697

-- Define different planes α, β, γ
variables (α β γ : Plane)

-- Define the parallel relation between planes
axiom parallel : Plane → Plane → Prop

-- Conditions
axiom diff_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ
axiom β_parallel_α : parallel β α
axiom γ_parallel_α : parallel γ α

-- Statement to prove
theorem parallel_planes_transitivity (α β γ : Plane) 
  (h1 : parallel β α) 
  (h2 : parallel γ α) 
  (h3 : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) : parallel β γ :=
sorry

end NUMINAMATH_GPT_parallel_planes_transitivity_l1556_155697


namespace NUMINAMATH_GPT_number_of_n_l1556_155631

theorem number_of_n (n : ℕ) (hn : n ≤ 500) (hk : ∃ k : ℕ, 21 * n = k^2) : 
  ∃ m : ℕ, m = 4 := by
  sorry

end NUMINAMATH_GPT_number_of_n_l1556_155631
