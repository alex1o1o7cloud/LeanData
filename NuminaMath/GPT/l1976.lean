import Mathlib

namespace NUMINAMATH_GPT_volume_CO2_is_7_l1976_197650

-- Definitions based on conditions
def Avogadro_law (V1 V2 : ℝ) : Prop := V1 = V2
def molar_ratio (V_CO2 V_O2 : ℝ) : Prop := V_CO2 = 1 / 2 * V_O2
def volume_O2 : ℝ := 14

-- Statement to be proved
theorem volume_CO2_is_7 : ∃ V_CO2 : ℝ, molar_ratio V_CO2 volume_O2 ∧ V_CO2 = 7 := by
  sorry

end NUMINAMATH_GPT_volume_CO2_is_7_l1976_197650


namespace NUMINAMATH_GPT_evaluate_expression_l1976_197685

theorem evaluate_expression (x : ℝ) (h1 : x^5 + 1 ≠ 0) (h2 : x^5 - 1 ≠ 0) :
  ( ((x^2 - 2*x + 2)^2 * (x^3 - x^2 + 1)^2 / (x^5 + 1)^2)^2 *
    ((x^2 + 2*x + 2)^2 * (x^3 + x^2 + 1)^2 / (x^5 - 1)^2)^2 )
  = 1 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1976_197685


namespace NUMINAMATH_GPT_new_student_weight_l1976_197692

theorem new_student_weight (avg_weight : ℝ) (x : ℝ) :
  (avg_weight * 10 - 120) = ((avg_weight - 6) * 10 + x) → x = 60 :=
by
  intro h
  -- The proof would go here, but it's skipped.
  sorry

end NUMINAMATH_GPT_new_student_weight_l1976_197692


namespace NUMINAMATH_GPT_area_of_triangle_l1976_197656

theorem area_of_triangle {a c : ℝ} (B : ℝ) (h1 : a = 1) (h2 : c = 2) (h3 : B = 60) :
    (1 / 2) * a * c * Real.sin (B * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_GPT_area_of_triangle_l1976_197656


namespace NUMINAMATH_GPT_relationship_among_a_b_c_l1976_197601

theorem relationship_among_a_b_c :
  let a := (1/6) ^ (1/2)
  let b := Real.log (1/3) / Real.log 6
  let c := Real.log (1/7) / Real.log (1/6)
  c > a ∧ a > b :=
by
  sorry

end NUMINAMATH_GPT_relationship_among_a_b_c_l1976_197601


namespace NUMINAMATH_GPT_find_m_n_calculate_expression_l1976_197611

-- Define the polynomials A and B
def A (m x : ℝ) := 5 * x^2 - m * x + 1
def B (n x : ℝ) := 2 * x^2 - 2 * x - n

-- The conditions
variable (x : ℝ) (m n : ℝ)
def no_linear_or_constant_terms (m : ℝ) (n : ℝ) : Prop :=
  ∀ x : ℝ, 3 * x^2 + (2 - m) * x + (1 + n) = 3 * x^2

-- The target theorem
theorem find_m_n 
  (h : no_linear_or_constant_terms m n) : 
  m = 2 ∧ n = -1 := sorry

-- Calculate the expression when m = 2 and n = -1
theorem calculate_expression
  (hm : m = 2)
  (hn : n = -1) : 
  m^2 + n^2 - 2 * m * n = 9 := sorry

end NUMINAMATH_GPT_find_m_n_calculate_expression_l1976_197611


namespace NUMINAMATH_GPT_initial_cats_count_l1976_197632

theorem initial_cats_count :
  ∀ (initial_birds initial_puppies initial_spiders final_total initial_cats: ℕ),
    initial_birds = 12 →
    initial_puppies = 9 →
    initial_spiders = 15 →
    final_total = 25 →
    (initial_birds / 2 + initial_puppies - 3 + initial_spiders - 7 + initial_cats = final_total) →
    initial_cats = 5 := by
  intros initial_birds initial_puppies initial_spiders final_total initial_cats h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_initial_cats_count_l1976_197632


namespace NUMINAMATH_GPT_johns_speed_final_push_l1976_197648

-- Definitions for the given conditions
def john_behind_steve : ℝ := 14
def steve_speed : ℝ := 3.7
def john_ahead_steve : ℝ := 2
def john_final_push_time : ℝ := 32

-- Proving the statement
theorem johns_speed_final_push : 
  (∃ (v : ℝ), v * john_final_push_time = steve_speed * john_final_push_time + john_behind_steve + john_ahead_steve) -> 
  ∃ (v : ℝ), v = 4.2 :=
by
  sorry

end NUMINAMATH_GPT_johns_speed_final_push_l1976_197648


namespace NUMINAMATH_GPT_hairstylist_earnings_per_week_l1976_197613

theorem hairstylist_earnings_per_week :
  let cost_normal := 5
  let cost_special := 6
  let cost_trendy := 8
  let haircuts_normal := 5
  let haircuts_special := 3
  let haircuts_trendy := 2
  let days_per_week := 7
  let daily_earnings := cost_normal * haircuts_normal + cost_special * haircuts_special + cost_trendy * haircuts_trendy
  let weekly_earnings := daily_earnings * days_per_week
  weekly_earnings = 413 := sorry

end NUMINAMATH_GPT_hairstylist_earnings_per_week_l1976_197613


namespace NUMINAMATH_GPT_quadratic_eq_has_two_distinct_real_roots_l1976_197652

theorem quadratic_eq_has_two_distinct_real_roots (m : ℝ) : 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (∀ x : ℝ, x^2 - 2*m*x - m - 1 = 0 ↔ x = x1 ∨ x = x2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_eq_has_two_distinct_real_roots_l1976_197652


namespace NUMINAMATH_GPT_find_a7_l1976_197623

variable (a : ℕ → ℝ)

def arithmetic_sequence (d : ℝ) (a1 : ℝ) :=
  ∀ n, a n = a1 + (n - 1) * d

theorem find_a7
  (a : ℕ → ℝ)
  (d : ℝ)
  (a1 : ℝ)
  (h_arith : arithmetic_sequence a d a1)
  (h_a3 : a 3 = 7)
  (h_a5 : a 5 = 13):
  a 7 = 19 :=
by
  sorry

end NUMINAMATH_GPT_find_a7_l1976_197623


namespace NUMINAMATH_GPT_radius_squared_l1976_197649

-- Definitions of the conditions
def point_A := (2, -1)
def line_l1 (x y : ℝ) := x + y = 1
def line_l2 (x y : ℝ) := 2 * x + y = 0

-- Circle with center (h, k) and radius r
def circle_equation (h k r x y : ℝ) := (x - h) ^ 2 + (y - k) ^ 2 = r ^ 2

-- Prove statement: r^2 = 2 given the conditions
theorem radius_squared (h k r : ℝ) 
  (H1 : circle_equation h k r 2 (-1))
  (H2 : line_l1 h k)
  (H3 : line_l2 h k):
  r ^ 2 = 2 := sorry

end NUMINAMATH_GPT_radius_squared_l1976_197649


namespace NUMINAMATH_GPT_correct_factorization_l1976_197671

theorem correct_factorization:
  (∃ a : ℝ, (a + 3) * (a - 3) = a ^ 2 - 9) ∧
  (∃ x : ℝ, x ^ 2 + x - 5 = x * (x + 1) - 5) ∧
  ¬ (∃ x : ℝ, x ^ 2 + 1 = x * (x + 1 / x)) ∧
  (∃ x : ℝ, x ^ 2 + 4 * x + 4 = (x + 2) ^ 2) →
  (∃ x : ℝ, x ^ 2 + 4 * x + 4 = (x + 2) ^ 2)
  := by
  sorry

end NUMINAMATH_GPT_correct_factorization_l1976_197671


namespace NUMINAMATH_GPT_broken_marbles_total_l1976_197675

theorem broken_marbles_total :
  let broken_set_1 := 0.10 * 50
  let broken_set_2 := 0.20 * 60
  let broken_set_3 := 0.30 * 70
  let broken_set_4 := 0.15 * 80
  let total_broken := broken_set_1 + broken_set_2 + broken_set_3 + broken_set_4
  total_broken = 50 :=
by
  sorry


end NUMINAMATH_GPT_broken_marbles_total_l1976_197675


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1976_197643

theorem necessary_but_not_sufficient (a : ℝ) : 
  (a > 2 → a^2 > 2 * a) ∧ (a^2 > 2 * a → (a > 2 ∨ a < 0)) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1976_197643


namespace NUMINAMATH_GPT_find_k_l1976_197687

noncomputable def proof_problem (x1 x2 x3 x4 : ℝ) (k : ℝ) : Prop :=
  (x1 + x2) / (x3 + x4) = k ∧
  (x3 + x4) / (x1 + x2) = k ∧
  (x1 + x3) / (x2 + x4) = k ∧
  (x2 + x4) / (x1 + x3) = k ∧
  (x1 + x4) / (x2 + x3) = k ∧
  (x2 + x3) / (x1 + x4) = k ∧
  x1 ≠ x2 ∨ x2 ≠ x3 ∨ x3 ≠ x4 ∨ x4 ≠ x1

theorem find_k (x1 x2 x3 x4 : ℝ) (h : proof_problem x1 x2 x3 x4 k) : k = -1 :=
  sorry

end NUMINAMATH_GPT_find_k_l1976_197687


namespace NUMINAMATH_GPT_michael_digging_time_equals_700_l1976_197668

-- Conditions defined
def digging_rate := 4
def father_depth := digging_rate * 400
def michael_depth := 2 * father_depth - 400
def time_for_michael := michael_depth / digging_rate

-- Statement to prove
theorem michael_digging_time_equals_700 : time_for_michael = 700 :=
by
  -- Here we would provide the proof steps, but we use sorry for now
  sorry

end NUMINAMATH_GPT_michael_digging_time_equals_700_l1976_197668


namespace NUMINAMATH_GPT_max_withdrawal_l1976_197621

def initial_balance : ℕ := 500
def withdraw_amount : ℕ := 300
def add_amount : ℕ := 198
def remaining_balance (x : ℕ) : Prop := 
  x % 6 = 0 ∧ x ≤ initial_balance

theorem max_withdrawal : ∃(max_withdrawal_amount : ℕ), 
  max_withdrawal_amount = initial_balance - 498 :=
sorry

end NUMINAMATH_GPT_max_withdrawal_l1976_197621


namespace NUMINAMATH_GPT_total_cost_of_vacation_l1976_197629

variable (C : ℚ)

def cost_per_person_divided_among_3 := C / 3
def cost_per_person_divided_among_4 := C / 4
def per_person_difference := 40

theorem total_cost_of_vacation
  (h : cost_per_person_divided_among_3 C - cost_per_person_divided_among_4 C = per_person_difference) :
  C = 480 := by
  sorry

end NUMINAMATH_GPT_total_cost_of_vacation_l1976_197629


namespace NUMINAMATH_GPT_kanul_spent_on_raw_materials_eq_500_l1976_197698

variable (total_amount : ℕ)
variable (machinery_cost : ℕ)
variable (cash_percentage : ℕ)

def amount_spent_on_raw_materials (total_amount machinery_cost cash_percentage : ℕ) : ℕ :=
  total_amount - machinery_cost - (total_amount * cash_percentage / 100)

theorem kanul_spent_on_raw_materials_eq_500 :
  total_amount = 1000 →
  machinery_cost = 400 →
  cash_percentage = 10 →
  amount_spent_on_raw_materials total_amount machinery_cost cash_percentage = 500 :=
by
  intros
  sorry

end NUMINAMATH_GPT_kanul_spent_on_raw_materials_eq_500_l1976_197698


namespace NUMINAMATH_GPT_composite_numbers_quotient_l1976_197627

theorem composite_numbers_quotient :
  (14 * 15 * 16 * 18 * 20 * 21 * 22 * 24 * 25 * 26 : ℚ) /
  (27 * 28 * 30 * 32 * 33 * 34 * 35 * 36 * 38 * 39) =
  (14 * 15 * 16 * 18 * 20 * 21 * 22 * 24 * 25 * 26 : ℚ) / 
  (27 * 28 * 30 * 32 * 33 * 34 * 35 * 36 * 38 * 39) :=
by sorry

end NUMINAMATH_GPT_composite_numbers_quotient_l1976_197627


namespace NUMINAMATH_GPT_new_average_mark_of_remaining_students_l1976_197688

def new_average (total_students : ℕ) (excluded_students : ℕ) (avg_marks : ℕ) (excluded_avg_marks : ℕ) : ℕ :=
  ((total_students * avg_marks) - (excluded_students * excluded_avg_marks)) / (total_students - excluded_students)

theorem new_average_mark_of_remaining_students 
  (total_students : ℕ) (excluded_students : ℕ) (avg_marks : ℕ) (excluded_avg_marks : ℕ)
  (h1 : total_students = 33)
  (h2 : excluded_students = 3)
  (h3 : avg_marks = 90)
  (h4 : excluded_avg_marks = 40) : 
  new_average total_students excluded_students avg_marks excluded_avg_marks = 95 :=
by
  sorry

end NUMINAMATH_GPT_new_average_mark_of_remaining_students_l1976_197688


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1976_197663

theorem isosceles_triangle_perimeter {a b c : ℝ} (h1 : a = 4) (h2 : b = 8) 
  (isosceles : a = c ∨ b = c) (triangle_inequality : a + a > b) :
  a + b + c = 20 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1976_197663


namespace NUMINAMATH_GPT_Ryan_spit_distance_correct_l1976_197636

-- Definitions of given conditions
def Billy_spit_distance : ℝ := 30
def Madison_spit_distance : ℝ := Billy_spit_distance * 1.20
def Ryan_spit_distance : ℝ := Madison_spit_distance * 0.50

-- Goal statement
theorem Ryan_spit_distance_correct : Ryan_spit_distance = 18 := by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_Ryan_spit_distance_correct_l1976_197636


namespace NUMINAMATH_GPT_percentage_of_second_division_l1976_197604

theorem percentage_of_second_division
  (total_students : ℕ)
  (students_first_division : ℕ)
  (students_just_passed : ℕ)
  (h1: total_students = 300)
  (h2: students_first_division = 75)
  (h3: students_just_passed = 63) :
  (total_students - (students_first_division + students_just_passed)) * 100 / total_students = 54 := 
by
  -- Proof will be added later
  sorry

end NUMINAMATH_GPT_percentage_of_second_division_l1976_197604


namespace NUMINAMATH_GPT_percentage_B_of_C_l1976_197630

theorem percentage_B_of_C 
  (A C B : ℝ)
  (h1 : A = (7 / 100) * C)
  (h2 : A = (50 / 100) * B) :
  B = (14 / 100) * C := 
sorry

end NUMINAMATH_GPT_percentage_B_of_C_l1976_197630


namespace NUMINAMATH_GPT_sale_price_relative_to_original_l1976_197646

variable (x : ℝ)

def increased_price (x : ℝ) := 1.30 * x
def sale_price (increased_price : ℝ) := 0.90 * increased_price

theorem sale_price_relative_to_original (x : ℝ) :
  sale_price (increased_price x) = 1.17 * x :=
by
  sorry

end NUMINAMATH_GPT_sale_price_relative_to_original_l1976_197646


namespace NUMINAMATH_GPT_number_of_boys_in_class_l1976_197635

theorem number_of_boys_in_class (n : ℕ) (h : 182 * n - 166 + 106 = 180 * n) : n = 30 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_boys_in_class_l1976_197635


namespace NUMINAMATH_GPT_complement_of_M_in_U_is_14_l1976_197680

def U : Set ℕ := {x | x < 5 ∧ x > 0}

def M : Set ℕ := {x | x^2 - 5 * x + 6 = 0}

theorem complement_of_M_in_U_is_14 : 
  {x | x ∈ U ∧ x ∉ M} = {1, 4} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_M_in_U_is_14_l1976_197680


namespace NUMINAMATH_GPT_swimming_speed_still_water_l1976_197696

theorem swimming_speed_still_water 
  (v t : ℝ) 
  (h1 : 3 = (v + 3) * t / (v - 3)) 
  (h2 : t ≠ 0) :
  v = 9 :=
by
  sorry

end NUMINAMATH_GPT_swimming_speed_still_water_l1976_197696


namespace NUMINAMATH_GPT_lindy_total_distance_l1976_197658

def meet_distance (d v_j v_c : ℕ) : ℕ :=
  d / (v_j + v_c)

def lindy_distance (v_l t : ℕ) : ℕ :=
  v_l * t

theorem lindy_total_distance
  (d : ℕ)
  (v_j : ℕ)
  (v_c : ℕ)
  (v_l : ℕ)
  (h1 : d = 360)
  (h2 : v_j = 5)
  (h3 : v_c = 7)
  (h4 : v_l = 12)
  :
  lindy_distance v_l (meet_distance d v_j v_c) = 360 :=
by
  sorry

end NUMINAMATH_GPT_lindy_total_distance_l1976_197658


namespace NUMINAMATH_GPT_total_people_in_club_after_5_years_l1976_197631

noncomputable def club_initial_people := 18
noncomputable def executives_per_year := 6
noncomputable def initial_regular_members := club_initial_people - executives_per_year

-- Define the function for regular members growth
noncomputable def regular_members_after_n_years (n : ℕ) : ℕ := initial_regular_members * 2 ^ n

-- Total people in the club after 5 years
theorem total_people_in_club_after_5_years : 
  club_initial_people + regular_members_after_n_years 5 - initial_regular_members = 390 :=
by
  sorry

end NUMINAMATH_GPT_total_people_in_club_after_5_years_l1976_197631


namespace NUMINAMATH_GPT_james_louise_age_sum_l1976_197640

variables (J L : ℝ)

theorem james_louise_age_sum
  (h₁ : J = L + 9)
  (h₂ : J + 5 = 3 * (L - 3)) :
  J + L = 32 :=
by
  /- Proof goes here -/
  sorry

end NUMINAMATH_GPT_james_louise_age_sum_l1976_197640


namespace NUMINAMATH_GPT_quadrilateral_with_exactly_two_axes_of_symmetry_is_either_rectangle_or_rhombus_l1976_197697

structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

def has_exactly_two_axes_of_symmetry (q : Quadrilateral) : Prop :=
  -- Definition to be developed further based on symmetry conditions
  sorry

def is_rectangle (q : Quadrilateral) : Prop :=
  -- Definition to be developed further based on properties of rectangle
  sorry

def is_rhombus (q : Quadrilateral) : Prop :=
  -- Definition to be developed further based on properties of rhombus
  sorry

theorem quadrilateral_with_exactly_two_axes_of_symmetry_is_either_rectangle_or_rhombus
  (q : Quadrilateral)
  (h : has_exactly_two_axes_of_symmetry q) :
  is_rectangle q ∨ is_rhombus q := by
  sorry

end NUMINAMATH_GPT_quadrilateral_with_exactly_two_axes_of_symmetry_is_either_rectangle_or_rhombus_l1976_197697


namespace NUMINAMATH_GPT_prob_part1_prob_part2_l1976_197628

-- Define the probability that Person A hits the target
def pA : ℚ := 2 / 3

-- Define the probability that Person B hits the target
def pB : ℚ := 3 / 4

-- Define the number of shots
def nShotsA : ℕ := 3
def nShotsB : ℕ := 2

-- The problem posed to Person A
def probA_miss_at_least_once : ℚ := 1 - (pA ^ nShotsA)

-- The problem posed to Person A (exactly twice in 2 shots)
def probA_hits_exactly_twice : ℚ := pA ^ 2

-- The problem posed to Person B (exactly once in 2 shots)
def probB_hits_exactly_once : ℚ :=
  2 * (pB * (1 - pB))

-- The combined probability for Part 2
def combined_prob : ℚ := probA_hits_exactly_twice * probB_hits_exactly_once

theorem prob_part1 :
  probA_miss_at_least_once = 19 / 27 := by
  sorry

theorem prob_part2 :
  combined_prob = 1 / 6 := by
  sorry

end NUMINAMATH_GPT_prob_part1_prob_part2_l1976_197628


namespace NUMINAMATH_GPT_points_lie_on_line_l1976_197607

theorem points_lie_on_line (t : ℝ) (ht : t ≠ 0) :
    let x := (t + 2) / t
    let y := (t - 2) / t
    x + y = 2 :=
by
  let x := (t + 2) / t
  let y := (t - 2) / t
  sorry

end NUMINAMATH_GPT_points_lie_on_line_l1976_197607


namespace NUMINAMATH_GPT_num_distinct_units_digits_of_cubes_l1976_197633

theorem num_distinct_units_digits_of_cubes : 
  ∃ S : Finset ℕ, (∀ d : ℕ, (d < 10) → (d^3 % 10) ∈ S) ∧ S.card = 10 := by
  sorry

end NUMINAMATH_GPT_num_distinct_units_digits_of_cubes_l1976_197633


namespace NUMINAMATH_GPT_eval_expression_l1976_197684

theorem eval_expression :
  (2011 * (2012 * 10001) * (2013 * 100010001)) - (2013 * (2011 * 10001) * (2012 * 100010001)) =
  -2 * 2012 * 2013 * 10001 * 100010001 :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_l1976_197684


namespace NUMINAMATH_GPT_carrie_worked_days_l1976_197662

theorem carrie_worked_days (d : ℕ) 
  (h1: ∀ n : ℕ, d = n → (2 * 22 * n - 54 = 122)) : d = 4 :=
by
  -- The proof will go here.
  sorry

end NUMINAMATH_GPT_carrie_worked_days_l1976_197662


namespace NUMINAMATH_GPT_quadratic_function_positive_difference_l1976_197665

/-- Given a quadratic function y = ax^2 + bx + c, where the coefficient a
indicates a downward-opening parabola (a < 0) and the y-intercept is positive (c > 0),
prove that the expression (c - a) is always positive. -/
theorem quadratic_function_positive_difference (a b c : ℝ) (h1 : a < 0) (h2 : c > 0) : c - a > 0 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_function_positive_difference_l1976_197665


namespace NUMINAMATH_GPT_compare_variables_l1976_197682

theorem compare_variables (a b c : ℝ) (h1 : a = 2 ^ (1 / 2)) (h2 : b = Real.log 3 / Real.log π) (h3 : c = Real.log (1 / 3) / Real.log 2) : 
  a > b ∧ b > c :=
by
  sorry

end NUMINAMATH_GPT_compare_variables_l1976_197682


namespace NUMINAMATH_GPT_minimum_value_op_dot_fp_l1976_197669

theorem minimum_value_op_dot_fp (x y : ℝ) (h_ellipse : x^2 / 2 + y^2 = 1) :
  let OP := (x, y)
  let FP := (x - 1, y)
  let dot_product := x * (x - 1) + y^2
  dot_product ≥ 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_op_dot_fp_l1976_197669


namespace NUMINAMATH_GPT_linear_function_of_additivity_l1976_197610

theorem linear_function_of_additivity (f : ℝ → ℝ) 
  (h_add : ∀ x y : ℝ, f (x + y) = f x + f y) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
sorry

end NUMINAMATH_GPT_linear_function_of_additivity_l1976_197610


namespace NUMINAMATH_GPT_max_angle_C_l1976_197612

-- Define the necessary context and conditions
variable {a b c : ℝ}

-- Condition that a^2 + b^2 = 2c^2 in a triangle
axiom triangle_condition : a^2 + b^2 = 2 * c^2

-- Theorem statement
theorem max_angle_C (h : a^2 + b^2 = 2 * c^2) : ∃ C : ℝ, C = Real.pi / 3 := sorry

end NUMINAMATH_GPT_max_angle_C_l1976_197612


namespace NUMINAMATH_GPT_systematic_sampling_interval_l1976_197606

-- Definition of the population size and sample size
def populationSize : Nat := 800
def sampleSize : Nat := 40

-- The main theorem stating that the interval k in systematic sampling is 20
theorem systematic_sampling_interval : populationSize / sampleSize = 20 := by
  sorry

end NUMINAMATH_GPT_systematic_sampling_interval_l1976_197606


namespace NUMINAMATH_GPT_red_peppers_weight_correct_l1976_197602

def weight_of_red_peppers : Prop :=
  ∀ (T G : ℝ), (T = 0.66) ∧ (G = 0.33) → (T - G = 0.33)

theorem red_peppers_weight_correct : weight_of_red_peppers :=
  sorry

end NUMINAMATH_GPT_red_peppers_weight_correct_l1976_197602


namespace NUMINAMATH_GPT_missing_jar_size_l1976_197691

theorem missing_jar_size (x : ℕ) (h₁ : 3 * 16 + 3 * x + 3 * 40 = 252) 
                          (h₂ : 3 + 3 + 3 = 9) : x = 28 := 
by 
  sorry

end NUMINAMATH_GPT_missing_jar_size_l1976_197691


namespace NUMINAMATH_GPT_probability_diamond_or_ace_l1976_197609

theorem probability_diamond_or_ace (total_cards : ℕ) (diamonds : ℕ) (aces : ℕ) (jokers : ℕ)
  (not_diamonds_nor_aces : ℕ) (p_not_diamond_nor_ace : ℚ) (p_both_not_diamond_nor_ace : ℚ) : 
  total_cards = 54 →
  diamonds = 13 →
  aces = 4 →
  jokers = 2 →
  not_diamonds_nor_aces = 38 →
  p_not_diamond_nor_ace = 19 / 27 →
  p_both_not_diamond_nor_ace = (19 / 27) ^ 2 →
  1 - p_both_not_diamond_nor_ace = 368 / 729 :=
by 
  intros
  sorry

end NUMINAMATH_GPT_probability_diamond_or_ace_l1976_197609


namespace NUMINAMATH_GPT_sets_of_three_teams_l1976_197625

-- Definitions based on the conditions
def total_teams : ℕ := 20
def won_games : ℕ := 12
def lost_games : ℕ := 7

-- Main theorem to prove
theorem sets_of_three_teams : 
  (total_teams * (total_teams - 1) * (total_teams - 2)) / 6 / 2 = 570 := by
  sorry

end NUMINAMATH_GPT_sets_of_three_teams_l1976_197625


namespace NUMINAMATH_GPT_math_problem_l1976_197699

theorem math_problem : 3 * 3^4 + 9^60 / 9^59 - 27^3 = -19431 := by
  sorry

end NUMINAMATH_GPT_math_problem_l1976_197699


namespace NUMINAMATH_GPT_minimum_seats_l1976_197657

-- Condition: 150 seats in a row.
def seats : ℕ := 150

-- Assertion: The fewest number of seats that must be occupied so that any additional person seated must sit next to someone.
def minOccupiedSeats : ℕ := 50

theorem minimum_seats (s : ℕ) (m : ℕ) (h_seats : s = 150) (h_min : m = 50) :
  (∀ x, x = 150 → ∀ n, n ≥ 0 ∧ n ≤ m → 
    ∃ y, y ≥ 0 ∧ y ≤ x ∧ ∀ z, z = n + 1 → ∃ w, w ≥ 0 ∧ w ≤ x ∧ w = n ∨ w = n + 1) := 
sorry

end NUMINAMATH_GPT_minimum_seats_l1976_197657


namespace NUMINAMATH_GPT_remaining_watermelons_l1976_197618

def initial_watermelons : ℕ := 4
def eaten_watermelons : ℕ := 3

theorem remaining_watermelons : initial_watermelons - eaten_watermelons = 1 :=
by sorry

end NUMINAMATH_GPT_remaining_watermelons_l1976_197618


namespace NUMINAMATH_GPT_not_necessarily_divisible_by_66_l1976_197624

theorem not_necessarily_divisible_by_66 (m : ℤ) (h1 : ∃ k : ℤ, m = k * (k + 1) * (k + 2) * (k + 3) * (k + 4)) (h2 : 11 ∣ m) : ¬ (66 ∣ m) :=
sorry

end NUMINAMATH_GPT_not_necessarily_divisible_by_66_l1976_197624


namespace NUMINAMATH_GPT_derivative_at_x1_is_12_l1976_197679

theorem derivative_at_x1_is_12 : 
  (deriv (fun x : ℝ => (2 * x + 1) ^ 2) 1) = 12 :=
by
  sorry

end NUMINAMATH_GPT_derivative_at_x1_is_12_l1976_197679


namespace NUMINAMATH_GPT_base_8_to_base_10_2671_to_1465_l1976_197659

theorem base_8_to_base_10_2671_to_1465 :
  (2 * 8^3 + 6 * 8^2 + 7 * 8^1 + 1 * 8^0) = 1465 := by
  sorry

end NUMINAMATH_GPT_base_8_to_base_10_2671_to_1465_l1976_197659


namespace NUMINAMATH_GPT_condition1_not_sufficient_nor_necessary_condition2_necessary_l1976_197655

variable (x y : ℝ)

-- ① Neither sufficient nor necessary
theorem condition1_not_sufficient_nor_necessary (h1 : x ≠ 1 ∧ y ≠ 2) : ¬ ((x ≠ 1 ∧ y ≠ 2) → x + y ≠ 3) ∧ ¬ (x + y ≠ 3 → x ≠ 1 ∧ y ≠ 2) := sorry

-- ② Necessary condition
theorem condition2_necessary (h2 : x ≠ 1 ∨ y ≠ 2) : x + y ≠ 3 → (x ≠ 1 ∨ y ≠ 2) := sorry

end NUMINAMATH_GPT_condition1_not_sufficient_nor_necessary_condition2_necessary_l1976_197655


namespace NUMINAMATH_GPT_area_of_circle_segment_l1976_197637

-- Definitions for the conditions in the problem
def circle_eq (x y : ℝ) : Prop := x^2 - 10 * x + y^2 = 9
def line_eq (x y : ℝ) : Prop := y = x - 5

-- The area of the portion of the circle that lies above the x-axis and to the left of the line y = x - 5
theorem area_of_circle_segment :
  let area_of_circle := 34 * Real.pi
  let portion_fraction := 1 / 8
  portion_fraction * area_of_circle = 4.25 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_area_of_circle_segment_l1976_197637


namespace NUMINAMATH_GPT_suitable_comprehensive_survey_l1976_197693

-- Definitions based on conditions

def heights_of_students (n : Nat) : Prop := n = 45
def disease_rate_wheat (area : Type) : Prop := True
def love_for_chrysanthemums (population : Type) : Prop := True
def food_safety_hotel (time : Type) : Prop := True

-- The theorem to prove

theorem suitable_comprehensive_survey : 
  (heights_of_students 45 → True) ∧ 
  (disease_rate_wheat ℕ → False) ∧ 
  (love_for_chrysanthemums ℕ → False) ∧ 
  (food_safety_hotel ℕ → False) →
  heights_of_students 45 :=
by
  intros
  sorry

end NUMINAMATH_GPT_suitable_comprehensive_survey_l1976_197693


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l1976_197677

variables {a_n : ℕ → ℝ} {S_n q : ℝ}

axiom a1_eq : a_n 1 = 2
axiom an_eq : ∀ n, a_n n = if n > 0 then 2 * q^(n-1) else 0
axiom Sn_eq : ∀ n, a_n n = -64 → S_n = -42 → q = -2

theorem geometric_sequence_common_ratio (q : ℝ) :
  (∀ n, a_n n = if n > 0 then 2 * q^(n-1) else 0) →
  a_n 1 = 2 →
  (∀ n, a_n n = -64 → S_n = -42 → q = -2) :=
by intros _ _ _; sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l1976_197677


namespace NUMINAMATH_GPT_ratio_of_sums_equiv_seven_eighths_l1976_197641

variable (p q r u v w : ℝ)
variable (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
variable (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
variable (h1 : p^2 + q^2 + r^2 = 49)
variable (h2 : u^2 + v^2 + w^2 = 64)
variable (h3 : p * u + q * v + r * w = 56)

theorem ratio_of_sums_equiv_seven_eighths :
  (p + q + r) / (u + v + w) = 7 / 8 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_sums_equiv_seven_eighths_l1976_197641


namespace NUMINAMATH_GPT_tickets_count_l1976_197654

theorem tickets_count (x y: ℕ) (h : 3 * x + 5 * y = 78) : 
  ∃ n : ℕ , n = 6 :=
sorry

end NUMINAMATH_GPT_tickets_count_l1976_197654


namespace NUMINAMATH_GPT_typing_time_l1976_197690

theorem typing_time (typing_speed : ℕ) (words_per_page : ℕ) (number_of_pages : ℕ) (h1 : typing_speed = 90) (h2 : words_per_page = 450) (h3 : number_of_pages = 10) : 
  (words_per_page / typing_speed) * number_of_pages = 50 := 
by
  sorry

end NUMINAMATH_GPT_typing_time_l1976_197690


namespace NUMINAMATH_GPT_extracurricular_books_l1976_197622

theorem extracurricular_books (a b c d : ℕ) 
  (h1 : b + c + d = 110)
  (h2 : a + c + d = 108)
  (h3 : a + b + d = 104)
  (h4 : a + b + c = 119) :
  a = 37 ∧ b = 39 ∧ c = 43 ∧ d = 28 :=
by {
  -- Proof to be done here
  sorry
}

end NUMINAMATH_GPT_extracurricular_books_l1976_197622


namespace NUMINAMATH_GPT_men_hours_per_day_l1976_197666

theorem men_hours_per_day
  (H : ℕ)
  (men_days := 15 * 21 * H)
  (women_days := 21 * 20 * 9)
  (conversion_ratio := 3 / 2)
  (equivalent_man_hours := women_days * conversion_ratio)
  (same_work : men_days = equivalent_man_hours) :
  H = 8 :=
by
  sorry

end NUMINAMATH_GPT_men_hours_per_day_l1976_197666


namespace NUMINAMATH_GPT_quadratic_rewrite_l1976_197670

theorem quadratic_rewrite (x : ℝ) (b c : ℝ) : 
  (x^2 + 1560 * x + 2400 = (x + b)^2 + c) → 
  c / b = -300 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_rewrite_l1976_197670


namespace NUMINAMATH_GPT_prob_correct_last_digit_no_more_than_two_attempts_prob_correct_last_digit_no_more_than_two_attempts_if_even_l1976_197626

/-
Prove that if a person forgets the last digit of their 6-digit password, which can be any digit from 0 to 9,
the probability of pressing the correct last digit in no more than 2 attempts is 1/5.
-/

theorem prob_correct_last_digit_no_more_than_two_attempts :
  let correct_prob := 1 / 10 
  let incorrect_prob := 9 / 10 
  let second_attempt_prob := 1 / 9 
  correct_prob + (incorrect_prob * second_attempt_prob) = 1 / 5 :=
by
  sorry

/-
Prove that if a person forgets the last digit of their 6-digit password, but remembers that the last digit is an even number,
the probability of pressing the correct last digit in no more than 2 attempts is 2/5.
-/

theorem prob_correct_last_digit_no_more_than_two_attempts_if_even :
  let correct_prob := 1 / 5 
  let incorrect_prob := 4 / 5 
  let second_attempt_prob := 1 / 4 
  correct_prob + (incorrect_prob * second_attempt_prob) = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_prob_correct_last_digit_no_more_than_two_attempts_prob_correct_last_digit_no_more_than_two_attempts_if_even_l1976_197626


namespace NUMINAMATH_GPT_twins_ages_sum_equals_20_l1976_197667

def sum_of_ages (A K : ℕ) := 2 * A + K

theorem twins_ages_sum_equals_20 (A K : ℕ) (h1 : A = A) (h2 : A * A * K = 256) : 
  sum_of_ages A K = 20 :=
by
  sorry

end NUMINAMATH_GPT_twins_ages_sum_equals_20_l1976_197667


namespace NUMINAMATH_GPT_evaluateExpression_at_3_l1976_197681

noncomputable def evaluateExpression (x : ℚ) : ℚ :=
  (x - 1 + (2 - 2 * x) / (x + 1)) / ((x * x - x) / (x + 1))

theorem evaluateExpression_at_3 : evaluateExpression 3 = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_evaluateExpression_at_3_l1976_197681


namespace NUMINAMATH_GPT_chocolate_bars_percentage_l1976_197664

noncomputable def total_chocolate_bars (milk dark almond white caramel : ℕ) : ℕ :=
  milk + dark + almond + white + caramel

noncomputable def percentage (count total : ℕ) : ℚ :=
  (count : ℚ) / (total : ℚ) * 100

theorem chocolate_bars_percentage :
  let milk := 36
  let dark := 21
  let almond := 40
  let white := 15
  let caramel := 28
  let total := total_chocolate_bars milk dark almond white caramel
  total = 140 ∧
  percentage milk total = 25.71 ∧
  percentage dark total = 15 ∧
  percentage almond total = 28.57 ∧
  percentage white total = 10.71 ∧
  percentage caramel total = 20 :=
by
  sorry

end NUMINAMATH_GPT_chocolate_bars_percentage_l1976_197664


namespace NUMINAMATH_GPT_no_consecutive_days_played_l1976_197639

theorem no_consecutive_days_played (john_interval mary_interval : ℕ) :
  john_interval = 16 ∧ mary_interval = 25 → 
  ¬ ∃ (n : ℕ), (n * john_interval + 1 = m * mary_interval ∨ n * john_interval = m * mary_interval + 1) :=
by
  sorry

end NUMINAMATH_GPT_no_consecutive_days_played_l1976_197639


namespace NUMINAMATH_GPT_altered_solution_contains_correct_detergent_volume_l1976_197619

-- Define the original and altered ratios.
def original_ratio : ℝ × ℝ × ℝ := (2, 25, 100)
def altered_ratio_bleach_to_detergent : ℝ × ℝ := (6, 25)
def altered_ratio_detergent_to_water : ℝ × ℝ := (25, 200)

-- Define the given condition about the amount of water in the altered solution.
def altered_solution_water_volume : ℝ := 300

-- Define a function for the total altered solution volume and detergent volume
noncomputable def altered_solution_detergent_volume (water_volume : ℝ) : ℝ :=
  let detergent_volume := (altered_ratio_detergent_to_water.1 * water_volume) / altered_ratio_detergent_to_water.2
  detergent_volume

-- The proof statement asserting the amount of detergent in the altered solution.
theorem altered_solution_contains_correct_detergent_volume :
  altered_solution_detergent_volume altered_solution_water_volume = 37.5 :=
by
  sorry

end NUMINAMATH_GPT_altered_solution_contains_correct_detergent_volume_l1976_197619


namespace NUMINAMATH_GPT_x_minus_y_possible_values_l1976_197673

theorem x_minus_y_possible_values (x y : ℝ) (hx : x^2 = 9) (hy : |y| = 4) (hxy : x < y) : x - y = -1 ∨ x - y = -7 := 
sorry

end NUMINAMATH_GPT_x_minus_y_possible_values_l1976_197673


namespace NUMINAMATH_GPT_natural_number_pairs_int_l1976_197615

theorem natural_number_pairs_int {
  a b : ℕ
} : 
  (∃ a b : ℕ, 
    (b^2 - a ≠ 0 ∧ (a^2 + b) % (b^2 - a) = 0) ∧ 
    (a^2 - b ≠ 0 ∧ (b^2 + a) % (a^2 - b) = 0)
  ) ↔ ((a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3)) :=
by sorry

end NUMINAMATH_GPT_natural_number_pairs_int_l1976_197615


namespace NUMINAMATH_GPT_paint_cost_contribution_l1976_197644

theorem paint_cost_contribution
  (paint_cost_per_gallon : ℕ) 
  (coverage_per_gallon : ℕ) 
  (total_wall_area : ℕ) 
  (two_coats : ℕ) 
  : paint_cost_per_gallon = 45 → coverage_per_gallon = 400 → total_wall_area = 1600 → two_coats = 2 → 
    ((total_wall_area / coverage_per_gallon) * two_coats * paint_cost_per_gallon) / 2 = 180 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_paint_cost_contribution_l1976_197644


namespace NUMINAMATH_GPT_correct_statement_about_residuals_l1976_197605

-- Define the properties and characteristics of residuals as per the definition
axiom residuals_definition : Prop
axiom residuals_usefulness : residuals_definition → Prop

-- The theorem to prove that the correct statement about residuals is that they can be used to assess the effectiveness of model fitting
theorem correct_statement_about_residuals (h : residuals_definition) : residuals_usefulness h :=
sorry

end NUMINAMATH_GPT_correct_statement_about_residuals_l1976_197605


namespace NUMINAMATH_GPT_part1_part2_l1976_197686

noncomputable def f (x : ℝ) (k : ℝ) : ℝ :=
  k - |x - 3|

theorem part1 (k : ℝ) (h : ∀ x, f (x + 3) k ≥ 0 ↔ x ∈ [-1, 1]) : k = 1 :=
sorry

variable (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c)

theorem part2 (h : (1 / a) + (1 / (2 * b)) + (1 / (3 * c)) = 1) : 
  (1 / 9) * a + (2 / 9) * b + (3 / 9) * c ≥ 1 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1976_197686


namespace NUMINAMATH_GPT_mapping_f_of_neg2_and_3_l1976_197616

-- Define the mapping f
def f (x y : ℝ) : ℝ × ℝ := (x + y, x * y)

-- Define the given point
def p : ℝ × ℝ := (-2, 3)

-- Define the expected corresponding point
def expected_p : ℝ × ℝ := (1, -6)

-- The theorem stating the problem to be proved
theorem mapping_f_of_neg2_and_3 :
  f p.1 p.2 = expected_p := by
  sorry

end NUMINAMATH_GPT_mapping_f_of_neg2_and_3_l1976_197616


namespace NUMINAMATH_GPT_ellipse_h_k_a_c_sum_l1976_197653

theorem ellipse_h_k_a_c_sum :
  let h := -3
  let k := 1
  let a := 4
  let c := 2
  h + k + a + c = 4 :=
by
  let h := -3
  let k := 1
  let a := 4
  let c := 2
  show h + k + a + c = 4
  sorry

end NUMINAMATH_GPT_ellipse_h_k_a_c_sum_l1976_197653


namespace NUMINAMATH_GPT_female_students_count_l1976_197672

theorem female_students_count 
  (total_average : ℝ)
  (male_count : ℕ)
  (male_average : ℝ)
  (female_average : ℝ)
  (female_count : ℕ) 
  (correct_female_count : female_count = 12)
  (h1 : total_average = 90)
  (h2 : male_count = 8)
  (h3 : male_average = 87)
  (h4 : female_average = 92) :
  total_average * (male_count + female_count) = male_count * male_average + female_count * female_average :=
by sorry

end NUMINAMATH_GPT_female_students_count_l1976_197672


namespace NUMINAMATH_GPT_set_intersection_l1976_197674

theorem set_intersection (M N : Set ℝ) 
  (hM : M = {x | 2 * x - 3 < 1}) 
  (hN : N = {x | -1 < x ∧ x < 3}) : 
  (M ∩ N) = {x | -1 < x ∧ x < 2} := 
by 
  sorry

end NUMINAMATH_GPT_set_intersection_l1976_197674


namespace NUMINAMATH_GPT_find_triplets_l1976_197638

theorem find_triplets (x y z : ℕ) :
  (x^2 + y^2 = 3 * 2016^z + 77) →
  (x, y, z) = (77, 14, 1) ∨ (x, y, z) = (14, 77, 1) ∨ 
  (x, y, z) = (70, 35, 1) ∨ (x, y, z) = (35, 70, 1) ∨ 
  (x, y, z) = (8, 4, 0) ∨ (x, y, z) = (4, 8, 0) :=
by
  sorry

end NUMINAMATH_GPT_find_triplets_l1976_197638


namespace NUMINAMATH_GPT_AM_GM_inequality_l1976_197614

theorem AM_GM_inequality (a : List ℝ) (h : ∀ x ∈ a, 0 < x) :
  (a.sum / a.length) ≥ a.prod ^ (1 / a.length) := 
sorry

end NUMINAMATH_GPT_AM_GM_inequality_l1976_197614


namespace NUMINAMATH_GPT_smallest_m_plus_n_l1976_197695

theorem smallest_m_plus_n (m n : ℕ) (hmn : m > n) (hid : (2012^m : ℕ) % 1000 = (2012^n) % 1000) : m + n = 104 :=
sorry

end NUMINAMATH_GPT_smallest_m_plus_n_l1976_197695


namespace NUMINAMATH_GPT_probability_of_team_with_2_girls_2_boys_l1976_197608

open Nat

-- Define the combinatorics function for binomial coefficients
def binomial (n k : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem probability_of_team_with_2_girls_2_boys :
  let total_women := 8
  let total_men := 6
  let team_size := 4
  let ways_to_choose_2_girls := binomial total_women 2
  let ways_to_choose_2_boys := binomial total_men 2
  let total_ways_to_form_team := binomial (total_women + total_men) team_size
  let favorable_outcomes := ways_to_choose_2_girls * ways_to_choose_2_boys
  (favorable_outcomes : ℚ) / total_ways_to_form_team = 60 / 143 := 
by sorry

end NUMINAMATH_GPT_probability_of_team_with_2_girls_2_boys_l1976_197608


namespace NUMINAMATH_GPT_find_f_l1976_197603

noncomputable def f (x : ℕ) : ℚ := (1/4) * x * (x + 1) * (2 * x + 1)

lemma f_initial_condition : f 1 = 3 / 2 := by
  sorry

lemma f_functional_equation (x y : ℕ) :
  f (x + y) = (1 + y / (x + 1)) * f x + (1 + x / (y + 1)) * f y + x^2 * y + x * y + x * y^2 := by
  sorry

theorem find_f (x : ℕ) : f x = (1 / 4) * x * (x + 1) * (2 * x + 1) := by
  sorry

end NUMINAMATH_GPT_find_f_l1976_197603


namespace NUMINAMATH_GPT_a_b_sum_possible_values_l1976_197645

theorem a_b_sum_possible_values (a b : ℝ) 
  (h1 : a^3 - 12 * a^2 + 9 * a - 18 = 0)
  (h2 : 9 * b^3 - 135 * b^2 + 450 * b - 1650 = 0) :
  a + b = 6 ∨ a + b = 14 :=
sorry

end NUMINAMATH_GPT_a_b_sum_possible_values_l1976_197645


namespace NUMINAMATH_GPT_division_remainder_l1976_197676

def remainder (x y : ℕ) : ℕ := x % y

theorem division_remainder (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : (x : ℚ) / y = 96.15) (h4 : y = 20) : remainder x y = 3 :=
by
  sorry

end NUMINAMATH_GPT_division_remainder_l1976_197676


namespace NUMINAMATH_GPT_no_real_intersection_l1976_197600

def parabola_line_no_real_intersection : Prop :=
  let a := 3
  let b := -6
  let c := 5
  (b^2 - 4 * a * c) < 0

theorem no_real_intersection (h : parabola_line_no_real_intersection) : 
  ∀ x : ℝ, 3*x^2 - 4*x + 2 ≠ 2*x - 3 :=
by sorry

end NUMINAMATH_GPT_no_real_intersection_l1976_197600


namespace NUMINAMATH_GPT_change_digit_correct_sum_l1976_197660

theorem change_digit_correct_sum :
  ∃ d e, 
  d = 2 ∧ e = 8 ∧ 
  653479 + 938521 ≠ 1616200 ∧
  (658479 + 938581 = 1616200) ∧ 
  d + e = 10 := 
by {
  -- our proof goes here
  sorry
}

end NUMINAMATH_GPT_change_digit_correct_sum_l1976_197660


namespace NUMINAMATH_GPT_average_player_time_l1976_197620

theorem average_player_time:
  let pg := 130
  let sg := 145
  let sf := 85
  let pf := 60
  let c := 180
  let total_secs := pg + sg + sf + pf + c
  let total_mins := total_secs / 60
  let num_players := 5
  let avg_mins_per_player := total_mins / num_players
  avg_mins_per_player = 2 :=
by
  sorry

end NUMINAMATH_GPT_average_player_time_l1976_197620


namespace NUMINAMATH_GPT_tens_digit_of_9_pow_1010_l1976_197642

theorem tens_digit_of_9_pow_1010 : (9 ^ 1010) % 100 = 1 :=
by sorry

end NUMINAMATH_GPT_tens_digit_of_9_pow_1010_l1976_197642


namespace NUMINAMATH_GPT_increase_in_volume_eq_l1976_197634

theorem increase_in_volume_eq (x : ℝ) (l w h : ℝ) (h₀ : l = 6) (h₁ : w = 4) (h₂ : h = 5) :
  (6 + x) * 4 * 5 = 6 * 4 * (5 + x) :=
by
  sorry

end NUMINAMATH_GPT_increase_in_volume_eq_l1976_197634


namespace NUMINAMATH_GPT_speed_of_man_rowing_upstream_l1976_197647

theorem speed_of_man_rowing_upstream (V_m V_downstream V_upstream : ℝ) 
  (H1 : V_m = 60) 
  (H2 : V_downstream = 65) 
  (H3 : V_upstream = V_m - (V_downstream - V_m)) : 
  V_upstream = 55 := 
by 
  subst H1 
  subst H2 
  rw [H3] 
  norm_num

end NUMINAMATH_GPT_speed_of_man_rowing_upstream_l1976_197647


namespace NUMINAMATH_GPT_triangular_weight_l1976_197689

noncomputable def rectangular_weight := 90
variables {C T : ℕ}

-- Conditions
axiom cond1 : C + T = 3 * C
axiom cond2 : 4 * C + T = T + C + rectangular_weight

-- Question: How much does the triangular weight weigh?
theorem triangular_weight : T = 60 :=
sorry

end NUMINAMATH_GPT_triangular_weight_l1976_197689


namespace NUMINAMATH_GPT_continuity_at_x0_l1976_197651

noncomputable def f (x : ℝ) : ℝ := -4 * x^2 - 7

theorem continuity_at_x0 :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 1| < δ → |f x - f 1| < ε :=
by
  sorry

end NUMINAMATH_GPT_continuity_at_x0_l1976_197651


namespace NUMINAMATH_GPT_inequality_proof_l1976_197678

variable (x y : ℝ)
variable (h1 : x ≥ 0)
variable (h2 : y ≥ 0)
variable (h3 : x + y ≤ 1)

theorem inequality_proof (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + y ≤ 1) : 
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) := 
by 
  sorry

end NUMINAMATH_GPT_inequality_proof_l1976_197678


namespace NUMINAMATH_GPT_g_h_value_l1976_197661

def g (x : ℕ) : ℕ := 3 * x^2 + 2
def h (x : ℕ) : ℕ := 5 * x^3 - 2

theorem g_h_value : g (h 2) = 4334 := by
  sorry

end NUMINAMATH_GPT_g_h_value_l1976_197661


namespace NUMINAMATH_GPT_car_traveled_miles_per_gallon_city_l1976_197617

noncomputable def miles_per_gallon_city (H C G : ℝ) : Prop :=
  (C = H - 18) ∧ (462 = H * G) ∧ (336 = C * G)

theorem car_traveled_miles_per_gallon_city :
  ∃ H G, miles_per_gallon_city H 48 G :=
by
  sorry

end NUMINAMATH_GPT_car_traveled_miles_per_gallon_city_l1976_197617


namespace NUMINAMATH_GPT_complex_roots_eqn_l1976_197694

open Complex

theorem complex_roots_eqn (a b c d e k : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (he : e ≠ 0) 
  (hk1 : a * k^3 + b * k^2 + c * k + d = e)
  (hk2 : b * k^3 + c * k^2 + d * k + e = a) :
  k = 1 ∨ k = -1 ∨ k = Complex.I ∨ k = -Complex.I := 
sorry

end NUMINAMATH_GPT_complex_roots_eqn_l1976_197694


namespace NUMINAMATH_GPT_other_store_pools_l1976_197683

variable (P A : ℕ)
variable (three_times : P = 3 * A)
variable (total_pools : P + A = 800)

theorem other_store_pools (three_times : P = 3 * A) (total_pools : P + A = 800) : A = 266 := 
by
  sorry

end NUMINAMATH_GPT_other_store_pools_l1976_197683
